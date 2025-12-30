import json
import logging
import re
from functools import partial
from pathlib import Path

import hydra
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./configs/my_configs", config_name="inference-vllm")
def main(args):
    # 【步骤 1】打印配置信息
    logging.info(OmegaConf.to_yaml(args))

    # 【步骤 2】加载测试数据
    # 从 JSONL 文件加载数据(通常是处理好的验证集或测试集)
    logging.info(f"Loading data file: {args.data_file}")
    data = load_dataset(
        "json", data_files=to_absolute_path(args.data_file), split="train"
    )

    # 【步骤 3】实例化模型
    # 通过 Hydra 配置动态加载模型
    # 支持: HuggingFace Transformers, vLLM, OpenAI 等
    model = hydra.utils.instantiate(args.model, _convert_="object")
    logging.info(f"Loaded model: {model}")

    # 【步骤 4】检查缓存文件
    # 如果之前运行过推理,可以从缓存恢复,避免重复计算
    logging.info(f"Generated (opt. cache) file: {args.generated_file}")
    
    if Path(args.generated_file).exists():
        # 加载已生成的预测结果
        saved_data = load_dataset("json", data_files=args.generated_file, split="train")
    else:
        saved_data = []

    # 【步骤 5】限制数据量(可选)
    # 用于快速测试或调试
    if args.limit:
        data = data.select(range(args.limit))

    # 【步骤 6】定义推理函数
    def map_generate(model, examples, indices):
        # 确定 batch 中哪些索引需要生成
        todo_indices = [i for i in indices if i >= len(saved_data)]
        
        # 如果当前 batch 全部在缓存中
        if not todo_indices:
            examples[args.generation_key] = [saved_data[i][args.generation_key] for i in indices]
            return examples

        # 初始化结果列表（先填充 None 或缓存值）
        batch_results = [None] * len(indices)
        for i, idx in enumerate(indices):
            if idx < len(saved_data):
                batch_results[i] = saved_data[idx][args.generation_key]

        # 提取需要生成的 prompt
        prompts = [examples[args.input_key][i] for i, idx in enumerate(indices) if idx >= len(saved_data)]
        
        # 批量调用模型生成预测
        all_raw_outputs = model.generate(prompts)
        
        # 处理生成结果
        raw_out_iter = iter(all_raw_outputs)
        for i, idx in enumerate(indices):
            if idx >= len(saved_data):
                raw_out = next(raw_out_iter)
                
                # 正则清洗
                matches = re.findall(r'\[.*?\]', raw_out, re.DOTALL)
                # 确保输出始终为字符串，避免 datasets 加载时因类型不一致报错
                final_out = matches[0] if matches else raw_out
                
                batch_results[i] = final_out

                # 实时写入文件
                with open(args.generated_file, "a") as f:
                    f.write(
                        json.dumps({
                            args.generation_key: final_out,
                            "target": examples[args.target_key][i],
                            "raw_output": raw_out
                        }, ensure_ascii=False) + "\n"
                    )

        examples[args.generation_key] = batch_results
        return examples

    # 【步骤 7】批量推理
    # 使用 datasets 的 map 函数并行处理
    data = data.map(
        partial(map_generate, model), 
        with_indices=True, 
        batched=True, 
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
