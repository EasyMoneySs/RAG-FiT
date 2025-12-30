import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

def check_model(model_name):
    # 基础缓存路径
    cache_root = "/root/shared-nvme/hf_cache/hub"
    
    # 转换模型名称格式: Organization/ModelName -> models--Organization--ModelName
    # 例如: Qwen/Qwen3-Embedding-8B -> models--Qwen--Qwen3-Embedding-8B
    dir_name = "models--" + model_name.replace("/", "--")
    base_path = os.path.join(cache_root, dir_name, "snapshots")

    print(f"正在检查模型: {model_name}")
    print(f"预期路径: {base_path}")

    try:
        if not os.path.exists(base_path):
             raise FileNotFoundError(f"目录不存在: {base_path}\n请确认模型是否已下载到该路径。")

        # 自动获取第一个文件夹名（即哈希文件夹）
        snapshots = os.listdir(base_path)
        if not snapshots:
            raise Exception("snapshots 文件夹是空的，模型可能根本没下载成功。")
        
        model_path = os.path.join(base_path, snapshots[0])
        print(f"检测到模型实际路径: {model_path}")

        print("正在尝试加载分词器...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("分词器加载成功。")
        except Exception as e:
            print(f"分词器加载警告: {e}")
        
        print("正在尝试加载模型结构（Meta 模式）...")
        try:
            # 优先尝试作为 CausalLM 加载
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                device_map="meta"
            )
        except Exception as e_causal:
            print(f"AutoModelForCausalLM 加载失败 ({e_causal})，尝试通用 AutoModel...")
            # 如果是 Embedding 模型或其他结构，尝试通用加载
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                device_map="meta"
            )
            
        print("✅ 恭喜！模型结构加载成功，文件元数据完整。")

    except IndexError:
        print("❌ 错误：找不到 snapshots 目录下的哈希文件夹，请检查路径。")
    except Exception as e:
        print(f"❌ 加载失败，文件可能损坏或不完整。错误信息: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 HuggingFace 本地缓存中的模型完整性")
    parser.add_argument("--model_name", type=str, help="模型名称，例如: Qwen/Qwen3-Embedding-8B")
    
    args = parser.parse_args()
    check_model(args.model_name)