from modelscope import snapshot_download

# 下载路径：会自动下载到你当前的 cache 目录，或者你可以指定 cache_dir='./models'
model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B-Instruct', 
    cache_dir='./models'  # 建议指定一个目录，方便管理
)

print(f"模型下载完成，路径在: {model_dir}")