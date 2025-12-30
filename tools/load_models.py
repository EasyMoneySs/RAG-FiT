from huggingface_hub import snapshot_download

# 指定模型名称
model_id = "Qwen/Qwen3-8B"

# 指定本地保存路径（你可以根据你的 NVME 挂载点调整）
# local_dir = "/root/shared-nvme/models/Qwen3-4B-Thinking-2507"

# print(f"正在下载模型 {model_id} 到 {local_dir}...")

snapshot_download(
    repo_id=model_id,
    local_dir_use_symlinks=False, # 直接保存文件而非链接，方便后续移动
    resume_download=True,
    ignore_patterns=["*.msgpack", "*.h5"], # 忽略不需要的格式，节省空间
)

print("下载完成！")