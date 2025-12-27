import os
import glob
import argparse
from pathlib import Path

try:
    from haystack import Pipeline
    from haystack.components.converters import TextFileToDocument
    from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
except ImportError:
    print("Error: Haystack or Qdrant integration not found.")
    print("Please ensure you have installed the necessary dependencies:")
    print("pip install haystack-ai haystack-qdrant sentence-transformers")
    exit(1)

def index_data(data_dir, url, port, index_name, recreate_index):
    print(f"--> Connecting to Qdrant at {url}:{port}...")
    
    # 1. Initialize Document Store
    # Note: Parameters must match your configs/external/haystack/qdrant.yaml
    document_store = QdrantDocumentStore(
        url=url,
        port=port,
        index=index_name,
        embedding_dim=768,        # Matches BAAI/llm-embedder
        similarity="dot_product", # Matches config
        recreate_index=recreate_index # CAUTION: True will delete existing index
    )

    # 2. Build Pipeline
    print("--> Building Indexing Pipeline...")
    pipeline = Pipeline()

    # Components
    converter = TextFileToDocument()
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False
    )
    # Split long texts into chunks (e.g., 200 words)
    splitter = DocumentSplitter(
        split_by="word", 
        split_length=200, 
        split_overlap=20
    )
    # Embedder - Using the same model as in your retrieval config
    embedder = SentenceTransformersDocumentEmbedder(
        model="BAAI/llm-embedder",
        meta_fields_to_embed=["title"]
    )
    writer = DocumentWriter(document_store=document_store)

    # Add components to pipeline
    pipeline.add_component("converter", converter)
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)

    # Connect components
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # 3. Load Files
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        print(f"No .txt files found in {data_dir}")
        return

    print(f"--> Found {len(files)} files. Starting indexing...")
    
    # 4. Run Pipeline
    pipeline.run({"converter": {"sources": files}})
    
    print("--> Indexing successfully completed!")
    print(f"--> Data is now in Qdrant index: '{index_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index local text files into Qdrant for RAG-FiT")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing .txt files")
    parser.add_argument("--url", type=str, default="localhost", help="Qdrant URL (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant Port (default: 6333)")
    # index 就是像一个数据库名称一样
    parser.add_argument("--index", type=str, default="wikipedia", help="Qdrant Index Name (default: wikipedia)")
    parser.add_argument("--recreate", action="store_true", help="Force recreate index (WILL DELETE EXISTING DATA)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        exit(1)
        
    index_data(args.data_dir, args.url, args.port, args.index, args.recreate)
