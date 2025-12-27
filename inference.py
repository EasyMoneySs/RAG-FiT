import json
import logging
from functools import partial
from pathlib import Path

import hydra
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
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
    def map_generate(model, example, idx):
        # 如果当前样本还没有预测结果
        if idx >= len(saved_data):
            # 调用模型生成预测
            # args.input_key: 输入字段(如 "prompt")
            out = model.generate(example[args.input_key])
            
            # 将生成结果保存到样本中
            example[args.generation_key] = out

            # 实时写入文件(增量保存,防止中断丢失结果)
            with open(args.generated_file, "a") as f:
                f.write(
                    json.dumps({
                        "text": out,                          # 模型预测
                        "target": example[args.target_key]    # 标准答案
                    }) + "\n"
                )
        else:
            # 从缓存加载已有结果
            example[args.generation_key] = saved_data[idx]["text"]

        return example

    # 【步骤 7】批量推理
    # 使用 datasets 的 map 函数并行处理
    data = data.map(partial(map_generate, model), with_indices=True)


if __name__ == "__main__":
    main()
