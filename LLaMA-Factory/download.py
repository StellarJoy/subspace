from modelscope import snapshot_download
# 指定下载目录到数据盘
model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B-Instruct', 
    cache_dir='/root/autodl-tmp/models', 
    revision='master'
)
print(f'Download finished! Model path: {model_dir}')
