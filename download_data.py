import os
# 1. 强制走国内镜像（必须保留）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("🚀 开始下载 Commonsense170k (修复源)...")

try:
    # 2. 替换为有效的公开仓库 ID
    dataset_path = snapshot_download(
        repo_id="zwhe99/commonsense_170k",  # 这是一个确认存活的公开源
        repo_type="dataset",
        local_dir="/root/autodl-tmp/subspace/data/commonsense170k",
        local_dir_use_symlinks=False,  # 下载真实文件
        resume_download=True,
        max_workers=8  # H800 多线程拉取
    )
    print(f"✅ 下载成功！数据已保存在: {dataset_path}")
    print("💡 提示：该数据集通常是 parquet 格式，如果代码需要 json，可能需要简单转换。")

except Exception as e:
    print(f"❌ 下载失败，错误详情:\n{e}")