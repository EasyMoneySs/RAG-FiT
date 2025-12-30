import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import os

def analyze(data_file, model_name, input_key, target_key):
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Loading dataset: {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")
    
    input_lengths = []
    target_lens = []
    total_lens = []

    print("Processing tokens (this may take a while)...")
    for ex in dataset:
        # 统计输入字段
        val_in = str(ex.get(input_key) or "")
        in_tokens = len(tokenizer.encode(val_in))
        input_lengths.append(in_tokens)
        
        # 统计目标字段
        val_tar = str(ex.get(target_key) or "")
        tar_tokens = len(tokenizer.encode(val_tar))
        target_lens.append(tar_tokens)
        
        # 统计总长度 (Input + Target)
        total_lens.append(in_tokens + tar_tokens)

    def print_stats(name, lens):
        lens = np.array(lens)
        print(f"\n--- Statistics for {name} ---")
        print(f"Max length:    {np.max(lens)}")
        print(f"Mean length:   {np.mean(lens):.2f}")
        print(f"90th percentile: {np.percentile(lens, 90)}")
        print(f"95th percentile: {np.percentile(lens, 95)}")
        print(f"99th percentile: {np.percentile(lens, 99)}")

    print_stats(f"Input ({input_key})", input_lengths)
    print_stats(f"Target ({target_key})", target_lens)
    print_stats("Total (Input + Target)", total_lens)

if __name__ == "__main__":
    analyze(
        data_file="data/processed/drug_retrieval_v2-train.jsonl",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        input_key="my_prompt",
        target_key="answer"
    )
