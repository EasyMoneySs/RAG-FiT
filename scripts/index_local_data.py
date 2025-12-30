import os
import time
import glob
import argparse
import pickle
from pathlib import Path
import pandas as pd
from haystack import Document

try:
    from haystack import Pipeline
    from haystack.components.converters import TextFileToDocument
    from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.utils import ComponentDevice
    from qdrant_client import QdrantClient, models
except ImportError:
    print("Error: Haystack or Qdrant integration not found.")
    print("Please ensure you have installed the necessary dependencies:")
    print("pip install haystack-ai haystack-qdrant sentence-transformers pandas")
    exit(1)

import re

def remove_repetitions(text):
    """
    Remove repetitive substrings from text using regex.
    Handles patterns like "Text...Text...Text..." where "Text..." is > 10 chars.
    """
    if not text:
        return ""
        
    # Pattern explanation:
    # (.{10,}?)   -> Capture group 1: any character (dot), at least 10 times, non-greedy
    # \1+         -> Match content of group 1 one or more times immediately after
    # flags=re.DOTALL -> Make dot match newlines too
    pattern = r'(.{10,}?)\1+'
    
    # Run twice to handle nested or complex repeats
    try:
        text = re.sub(pattern, r'\1', text, flags=re.DOTALL)
        text = re.sub(pattern, r'\1', text, flags=re.DOTALL)
    except Exception:
        pass # Fallback to original text if regex is too complex/slow
        
    return text.strip()

def load_csv_as_documents(file_path):
    """
    Load a CSV file and convert each row into a Haystack Document.
    Combines all columns into a structured text format.
    Includes deduplication to handle repetitive data.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return []

    documents = []
    
    # Fill NaN with empty string
    df = df.fillna("")
    
    # Define columns to completely ignore (noise/internal data)
    SKIP_COLUMNS = {
        "标题链接", "链接", "url", "URL", "汉语拼音", "批准文号", "药品分类", "生产企业", "相关疾病", "主要成份", "规格", "贮藏", "有效期", "用法用量",
        "药品性质", "编号", "id", "ID", 
        "r3", "R3", 
        "Unnamed: 0"
    }
    
    # Use to_dict('records') for faster iteration compared to iterrows
    for row in df.to_dict("records"):
        content_parts = []
        meta = {}
        seen_values = set()
        
        # Determine title for metadata (prioritize '通用名称' or '商品名称')
        title = row.get("通用名称", row.get("商品名称", "Unknown Drug"))
        meta["title"] = str(title)
        
        for col in df.columns:
            # Skip blacklisted columns
            if col in SKIP_COLUMNS:
                continue
                
            val = str(row[col]).strip()
            
            # Skip empty or NaN
            if not val or val.lower() == "nan":
                continue
            
            # 0. Robust Deduplication using Regex
            val = remove_repetitions(val)
                
            # 1. Simple Deduplication: Skip if this exact text appeared in another column
            # (Check again after cleaning)
            if val in seen_values:
                continue
            seen_values.add(val)

            # Add to content string
            content_parts.append(f"{col}: {val}")
            
            # Add to metadata
            if len(val) < 200:
                meta[col] = val

        if not content_parts:
            continue
            
        text_content = "\n".join(content_parts)
        
        doc = Document(content=text_content, meta=meta)
        documents.append(doc)
        
    return documents

def index_data(data_dir, url, port, index_name, recreate_index, file_type="txt", embedder_model="BAAI/llm-embedder", embedding_dim=768, batch_size=256, write_batch_size=100, cache_path=None, start_batch=0):
    start_time = time.time()
    print(f"--> Connecting to Qdrant at {url}:{port}...")
    
    # 0. Ensure collection exists (Manual handling to avoid Haystack-Qdrant version mismatches)
    try:
        client = QdrantClient(url=url, port=port)
        if client.collection_exists(index_name):
            if recreate_index:
                print(f"--> Deleting existing collection '{index_name}'...")
                client.delete_collection(index_name)
                print(f"--> Creating collection '{index_name}' manually...")
                client.create_collection(
                    collection_name=index_name,
                    vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.DOT)
                )
            else:
                print(f"--> Collection '{index_name}' already exists. Appending to it.")
        else:
            print(f"--> Creating collection '{index_name}' manually...")
            client.create_collection(
                collection_name=index_name,
                vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.DOT)
            )
    except Exception as e:
        print(f"Warning: Failed to ensure collection exists manually: {e}")

    # 1. Initialize Document Store
    # We set recreate_index=False because we handled it manually (or didn't want it)
    document_store = QdrantDocumentStore(
        url=url,
        port=port,
        index=index_name,
        embedding_dim=embedding_dim,        # Configurable dimension
        similarity="dot_product", 
        recreate_index=False,
        write_batch_size=write_batch_size # Decoupled write batch size
    )

    # 2. Build Pipeline
    print("--> Building Indexing Pipeline...")
    pipeline = Pipeline()

    # Components
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False
    )
    # Split long texts into chunks
    # Optimized for Chinese text: use 'sentence' as Chinese words lack spaces
    splitter = DocumentSplitter(
        split_by="sentence", 
        split_length=6, 
        split_overlap=2
    )
    # Embedder - Using the provided model
    embedder = SentenceTransformersDocumentEmbedder(
        model=embedder_model,
        meta_fields_to_embed=["title"],
        device=ComponentDevice.from_str("cuda"),  # Use GPU if available
        batch_size=batch_size # Optimized embedding batch size
    )
    writer = DocumentWriter(document_store=document_store)

    # Add components to pipeline
    # Note: We don't add a Converter here because we might feed Documents directly if CSV
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)

    # Connect components
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # 3. Load Files
    documents = []

    if cache_path and os.path.exists(cache_path):
        print(f"--> Loading documents from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            documents = pickle.load(f)
    else:
        if file_type == "txt":
            print(f"--> Looking for .txt files in {data_dir}")
            files = glob.glob(os.path.join(data_dir, "*.txt"))
            if not files:
                print(f"No .txt files found in {data_dir}")
                return
            
            # For TXT, we use the converter dynamically or just use TextFileToDocument manually
            converter = TextFileToDocument()
            results = converter.run(sources=files)
            documents = results["documents"]
            
        elif file_type == "csv":
            print(f"--> Looking for .csv files in {data_dir}")
            files = glob.glob(os.path.join(data_dir, "*.csv"))
            if not files:
                print(f"No .csv files found in {data_dir}")
                return
                
            for f in files:
                print(f"    Loading {f}...")
                docs = load_csv_as_documents(f)
                documents.extend(docs)
        
        if cache_path:
            print(f"--> Saving {len(documents)} documents to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
            
    print(f"--> Prepared {len(documents)} documents. Starting indexing...")
    
    # 4. Run Pipeline (Batched)
    # Run in batches to save memory and provide progress updates
    total_docs = len(documents)
    
    for i in range(start_batch * batch_size, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        current_batch_idx = i // batch_size
        print(f"    Processing batch {current_batch_idx + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} docs)...")
        pipeline.run({"cleaner": {"documents": batch}})
    
    print("--> Indexing successfully completed!")
    print(f"--> Data is now in Qdrant index: '{index_name}'")
    print(f"--> Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index local text/csv files into Qdrant for RAG-FiT")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing data files")
    parser.add_argument("--type", type=str, default="txt", choices=["txt", "csv"], help="File type to index: 'txt' or 'csv'")
    parser.add_argument("--url", type=str, default="localhost", help="Qdrant URL (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant Port (default: 6333)")
    parser.add_argument("--index", type=str, default="wikipedia", help="Qdrant Index Name (default: wikipedia)")
    parser.add_argument("--recreate", action="store_true", help="Force recreate index (WILL DELETE EXISTING DATA)")
    parser.add_argument("--embedder_model", type=str, default="BAAI/llm-embedder", help="HuggingFace model name for embeddings")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Dimension of the embeddings (must match model)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding and indexing (default: 256)")
    parser.add_argument("--write_batch_size", type=int, default=100, help="Batch size for Qdrant storage writes (default: 100)")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to cache file for documents (e.g., docs.pkl)")
    parser.add_argument("--start_batch", type=int, default=0, help="Batch index to start processing from (for resuming)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        exit(1)
        
    index_data(args.data_dir, args.url, args.port, args.index, args.recreate, args.type, args.embedder_model, args.embedding_dim, args.batch_size, args.write_batch_size, args.cache_path, args.start_batch)
