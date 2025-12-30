# 2025-12-29 推理脚本修复记录

## 问题 1: Baichuan 模型缺少 Chat Template
**现象**: `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set...`
**原因**: HuggingFace 上的 Baichuan2-7B-Chat tokenizer 没有默认定义 `chat_template` 属性，导致调用 `apply_chat_template` 失败。
**修复**:
- 修改 `ragfit/models/vllm.py`: 在初始化时检测 `baichuan` 模型，如果缺少模板，手动注入正确的 Jinja2 模板（包含 `<reserved_106>` 和 `<reserved_107>` 特殊 Token）。

## 问题 2: LoRA Rank 限制
**现象**: `ValueError: LoRA rank 32 is greater than max_lora_rank 16.`
**原因**: vLLM 默认 `max_lora_rank` 为 16，而当前使用的 LoRA adapter rank 为 32。
**修复**:
- 修改 `configs/my_configs/inference-vllm.yaml`: 在 `llm_params` 中添加 `max_lora_rank: 32`。

## 问题 3: 上下文长度不足
**现象**: `ValueError: The decoder prompt (length ...) is longer than the maximum model length of 2048.`
**原因**: 配置文件中 `max_model_len` 默认为 2048，不足以处理包含长 System Prompt 的输入（提示词长度超过 2000）。
**修复**:
- 修改 `configs/my_configs/inference-vllm.yaml`: 将 `max_model_len` 增加至 `4096`，以匹配 Baichuan2 的支持长度。

## 问题 4: 输出重复与格式混乱 (严重)
**现象**: 模型输出无限循环的文本，且未遵循 JSON 格式指令。
**原因**:
1. **Prompt 构建方式错误**: 最初尝试手动拼接字符串 `<reserved_106>...`，但 vLLM/Tokenizer 未将其识别为特殊 Token，导致模型理解混乱。
2. **缺乏停止条件**: 未设置 `stop` token，导致模型生成完答案后继续“复读”。
**修复**:
- **代码层面 (`ragfit/models/vllm.py`)**:
    - 弃用字符串拼接。改用 `apply_chat_template(..., tokenize=True)` 直接生成 Token IDs。这确保了特殊 Token 被正确编码。
- **配置层面 (`inference-vllm.yaml`)**:
    - 添加 `stop: ["\n", "</s>"]`, 强制模型在换行或结束符处停止。
    - 添加 `repetition_penalty: 1.1` 以抑制重复。
    - 调整 `max_tokens` 为 `512`。

## 问题 5: vLLM API 兼容性 (v0.13.0)
**现象**: 
1. `TypeError: LLM.generate() got an unexpected keyword argument 'prompt'`
2. `TypeError: LLM.generate() got an unexpected keyword argument 'prompt_token_ids'`
**原因**: vLLM 0.13.0 的 `generate` 方法签名与旧代码不兼容：
1. 输入参数名应为 `prompts` 而非 `prompt`。
2. 不再支持 `prompt_token_ids` 作为独立关键字参数，需将其包装在字典 `{'prompt_token_ids': [...]}` 中作为 `TokensPrompt` 类型传给 `prompts`。
**修复**:
- 修改 `ragfit/models/vllm.py`: 重构 `generate` 方法调用，适配 vLLM 0.13.0 的 API 规范。

## 问题 6: 推理速度慢 (Batch Size = 1)
**现象**: 脚本默认逐条进行推理，无法利用 vLLM 的高吞吐特性，导致在大规模数据集上耗时极长。
**修复**:
- **配置层面 (`inference-vllm.yaml`)**: 新增 `batch_size: 32` 参数。
- **模型层面 (`ragfit/models/vllm.py`)**: 重构 `generate` 方法，使其支持接收 Prompt 列表，并实现批量 Token 化与 vLLM 批量提交。
- **脚本层面 (`inference.py`)**: 将数据映射函数 `data.map` 升级为 `batched=True` 模式，并优化了批量模式下的缓存跳过与结果持久化逻辑。

## 问题 7: 加载缓存时 Arrow 类型错误
**现象**: `pyarrow.lib.ArrowInvalid: JSON parse error: Column(/output) changed from string to array`.
**原因**: `re.findall` 在匹配成功时返回列表，失败时返回原始字符串。这导致 JSONL 文件中同一列出现了不同的数据类型，使 `datasets` 库无法正确解析。
**修复**:
- **脚本层面 (`inference.py`)**: 修改清洗逻辑，如果匹配成功则取 `matches[0]`。确保 `output` 列始终为字符串类型。

## 问题 8: RAG 分块策略不适用于医疗场景
**现象**: 默认按单词（word）切分且长度为 200，在中文医疗说明书场景下可能导致语义断裂，影响检索质量。
**修复**: 
- **脚本层面 (`scripts/index_local_data.py`)**: 将 `split_by` 更改为 `passage`（段落），`split_length` 设为 250，`split_overlap` 设为 30。这能更完整地捕获药品用药建议并保持语义连贯。

## 问题 9: 检索上下文过长导致冗余
**现象**: 默认 `top_k` 为 10，检索到的背景知识块过多，不仅增加了模型处理负担，还可能引入无关噪音。
**修复**: 
- **配置层面 (`configs/external/haystack/qdrant.yaml`)**: 将 `top_k` 从 10 降低到 5。这能显著缩短 Prompt 长度并提高信息密度。

---
**最终状态**: 
1. 解决了所有环境报错与 API 兼容性问题。
2. 修复了 Baichuan 模型特有的 Prompt 编码与格式混乱问题。
3. 新增批量推理功能，并修复了批量模式下的结果一致性问题。
4. 确保了持久化数据的类型一致性，支持断点续传。
5. **优化了 RAG 架构**: 调整了分块策略（Chunking）以适配医疗文本，并缩减了检索 Top-K 以提升上下文针对性。目前系统在稳定性、速度和检索质量上均得到了显著提升。
