# Medical RAG Agent - 训练操作指南 (Training Guide)

这份文档记录了针对“医疗病历用药推荐 Agent”进行模型训练的详细配置与操作流程。

## 1. 环境准备

在开始训练前，请确保环境配置正确：

```bash
# 1. 激活虚拟环境
conda activate ragfit  # 或你的环境名

# 2. 确认依赖安装
pip install -e .

# 3. 登录 WandB (只需执行一次)
wandb login
# 提示输入 API Key 时，从 https://wandb.ai/authorize 获取并粘贴
```

---

## 2. 配置文件说明 (`configs/training.yaml`)

我们使用 `configs/training.yaml` 来控制训练参数。以下是关键修改点，特别是针对 **全量微调 (Full Fine-tuning)** 的设置。

### 核心修改点

打开 `configs/training.yaml` 进行以下确认：

#### A. 模型与微调方式
**方案一：全量微调 (Full Fine-tuning) [推荐]**
适用于追求最佳效果，且显存充足（或使用 Gradient Accumulation）。

```yaml
model:
    _target_: ragfit.models.hf.HFTrain
    model_name_or_path: microsoft/Phi-3-mini-128k-instruct # 或其他基座模型
    
    # [重要] 全量微调必须关闭量化
    load_in_4bit: false
    load_in_8bit: false 
    
    # [重要] 移除或注释掉整个 lora 模块
    # lora:
    #    ...
```

**方案二：LoRA 微调 (Parameter-Efficient)**
适用于显存有限的情况。

```yaml
model:
    # ...
    # 开启量化以节省显存
    load_in_4bit: true 
    
    # 启用 LoRA
    lora:
        r: 16
        lora_alpha: 16
        target_modules: ["qkv_proj", ...] # 使用 tools/find_lora_targets.py 获取
        # ...
```

#### B. 显存优化与训练参数 (在 `train` 块中)

```yaml
train:
    output_dir: ./trained_models/medical_v1
    
    # 显存优化策略
    per_device_train_batch_size: 1      # 保持最小
    gradient_accumulation_steps: 8      # 累积梯度，相当于 Batch Size = 8
    
    # 精度设置
    bf16: true    # 30系/40系/A100/H100 显卡开启
    fp16: false   # 旧显卡开启这个，关闭 bf16
    
    # 学习率 (全量微调建议 2e-5, LoRA 可稍大 2e-4)
    learning_rate: 2e-5
    num_train_epochs: 3
    
    # 监控
    report_to: wandb
```

#### C. WandB 监控配置 (文件末尾)

```yaml
# 必须配置，否则无法追踪实验
use_wandb: true
wandb_entity: "your-username"    # 你的 WandB 用户名
project: "medical-rag-agent"     # 项目名称
experiment: "full-ft-run-001"    # 本次实验名称
```

#### D. Hugging Face Hub 上传配置 (可选)

如果你希望训练结束后**自动**将模型上传到 Hugging Face Hub（方便分享或在其他机器下载），请配置 `hfhub_tag`。

1.  **认证**: 必须先在终端登录 Hugging Face。
    ```bash
    huggingface-cli login
    # 输入你的 Access Token (从 https://huggingface.co/settings/tokens 获取，需有 Write 权限)
    ```

2.  **配置**:
    ```yaml
    # 格式: 用户名/模型仓库名
    hfhub_tag: "your-hf-username/medical-rag-phi3-full"
    ```
    *   如果留空，则只保存在本地，不上传。
    *   **注意**: 如果仓库不存在，它会自动创建（前提是 Token 有权限）。如果是私有模型，上传后默认为 Private。

---

## 3. 辅助工具

在配置 LoRA 时，如果不知道模型的 `target_modules` 写什么，可以使用我们在 `tools` 下创建的脚本：

```bash
# 查看模型结构并获取建议的 LoRA target modules
python tools/find_lora_targets.py microsoft/Phi-3-mini-128k-instruct
```

---

## 4. 启动训练

确认 `data_file` 指向了处理好的 JSONL 数据路径，然后运行：

```bash
# 基本运行命令
python training.py

# 命令行覆盖参数 (推荐) - 灵活修改实验名，避免改 YAML
python training.py experiment="medical-test-run-02" train.num_train_epochs=5
```

### 常见问题 Troubleshooting

1.  **OOM (Out of Memory)**:
    *   减小 `per_device_train_batch_size` 到 1。
    *   开启 `gradient_checkpointing: true` (在 `train` 块中添加)。
    *   如果是全量微调 OOM，考虑换回 LoRA 或使用 DeepSpeed (需要额外配置)。

2.  **WandB 报错 `KeyError: 'project'`**:
    *   确保 `configs/training.yaml` 文件末尾添加了 `project: "..."` 字段。

3.  **模型不收敛**:
    *   检查 `learning_rate` 是否太大。
    *   检查数据格式是否正确 (Input/Output 是否对应)。

---

## 5. 训练后操作

训练完成后，模型会保存在 `trained_models/checkpoint/` 目录下。

*   **推理验证**: 使用 `inference.py` 加载该路径进行测试。
*   **模型合并 (如果用了 LoRA)**: 如果需要部署，可能需要将 Adapter 合并回 Base Model。
