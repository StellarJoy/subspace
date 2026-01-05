import json
import os
from datasets import load_dataset

def main():
    print("🚀 开始准备数据 (终极修正版)...")

    # ==================== 1. 下载并清洗 MetaMathQA 数据集 ====================
    print("⏳ 正在加载 meta-math/MetaMathQA ...")
    
    try:
        # 下载数据
        dataset = load_dataset("meta-math/MetaMathQA", split="train[:10000]")
        
        # 1. 只保留需要的列
        print("🧹 正在清洗数据，仅保留 query 和 response 列...")
        dataset = dataset.select_columns(["query", "response"])
        
        # 2. [关键修复] 将 Dataset 对象转换为标准的 Python List
        # 这样我们可以完全控制保存格式，确保它是标准的 JSON 数组 [{}, {}, ...]
        data_list = dataset.to_list()
        
        # 创建保存目录
        os.makedirs("MetaMathQA", exist_ok=True)
        save_path = "MetaMathQA/train.json"
        
        # 3. [关键修复] 使用 python 原生 json.dump 保存
        print(f"💾 正在保存标准 JSON 格式至 {save_path}...")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 数据保存成功！(共 {len(data_list)} 条)")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    # ==================== 2. 更新 dataset_info.json ====================
    info_file = "dataset_info.json"
    
    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            try:
                data_info = json.load(f)
            except json.JSONDecodeError:
                data_info = {}
    else:
        data_info = {}

    print(f"📂 正在更新 {info_file} ...")

    # 添加 commonsense_170k
    data_info["commonsense_170k"] = {
        "file_name": "commonsense170k/train_shuffled.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }

    # 添加 meta_math
    data_info["meta_math"] = {
        "file_name": "MetaMathQA/train.json",
        "columns": {
            "prompt": "query",
            "response": "response"
        }
    }

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)

    print("🎉 修复完成！生成的文件现在包含逗号和外层 []，格式正确。")

if __name__ == "__main__":
    main()