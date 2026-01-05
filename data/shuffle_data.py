import json
import random

# 原始文件路径 (请根据实际路径修改)
input_file = "./commonsense170k/train.json"
# 新的乱序文件路径
output_file = "./commonsense170k/train_shuffled.json"

print(f"正在读取文件: {input_file} ...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据量: {len(data)}")
print("正在打乱数据顺序...")

# 核心步骤：全局打乱
random.seed(42) # 设置种子保证可复现
random.shuffle(data)

# 可选：如果只想保留前 10k 用于快速实验，可以在这里截断，
# 但更建议保存全量乱序版，在配置文件里控制 max_samples
# data = data[:10000] 

print(f"正在写入新文件: {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("完成！请修改 dataset_info.json 指向新文件。")