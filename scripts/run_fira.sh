#!/bin/bash

# ==================== 1. 魔法前缀 (自动定位工程目录) ====================
# 获取当前脚本所在的目录 (即 subspace/scripts)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 获取项目根目录 (即 subspace/)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# "瞬移"到根目录执行，确保所有相对路径都从 subspace/ 开始
cd "$PROJECT_ROOT"

# 将根目录加入 Python 搜索路径，确保能 import Fira 等模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "📍 当前工作目录: $(pwd)"

# ==================== 2. 参数配置 ====================

# [路径设置]
# 模型路径 (保持你原本的绝对路径，或者移入项目内用 $PROJECT_ROOT/models/...)
MODEL_PATH="/root/autodl-tmp/subspace/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# 数据路径 (修改为指向我们新建的 data 目录)
DATA_PATH="$PROJECT_ROOT/data/commonsense170k/train.json"

# 输出路径 (统一放入 outputs 文件夹，方便管理)
OUTPUT_DIR="$PROJECT_ROOT/outputs/fira"

# [显卡设置]
export CUDA_VISIBLE_DEVICES=0

# ==================== 3. 启动训练 ====================
echo "🚀 开始运行 Fira 训练..."
echo "📂 数据路径: $DATA_PATH"
echo "💾 输出路径: $OUTPUT_DIR"

# 注意：这里的 python 脚本路径是相对于 subspace/ 的
python Fira/fine_tuning/finetune.py \
  --base_model "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 64 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --cutoff_len 1024 \
  --val_set_size 1000 \
  --adapter_name lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
  --save_step 1000 \
  --eval_step 50 \
  --optimizer_name fira_adamw

echo "✅ 训练结束！"