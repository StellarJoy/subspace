#!/bin/bash
set -e
set -o xtrace

# ==================== 1. 魔法前缀 (自动定位工程目录) ====================
# 获取当前脚本所在的目录 (即 subspace/scripts)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 获取项目根目录 (即 subspace/)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# "瞬移"到根目录执行，确保所有相对路径都从 subspace/ 开始
cd "$PROJECT_ROOT"

# 同时加入 "项目根目录" (为了找 utils) 和 "stella 仓库目录" (为了找 stella 包)
export PYTHONPATH="$PROJECT_ROOT/stella:$PROJECT_ROOT:$PYTHONPATH"

echo "📍 当前工作目录: $(pwd)"

# ==================== 2. 参数配置 ====================

# [路径设置] - 全部修改为基于 $PROJECT_ROOT 的相对路径
# 这样即使你把 subspace 文件夹移动到任何地方，或者换了电脑，都能跑
MODEL_PATH="$PROJECT_ROOT/models/LLM-Research/Meta-Llama-3-8B-Instruct"
DATA_PATH="$PROJECT_ROOT/data/MetaMathQA/train.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stella"

# [环境设置]
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# ==================== 3. 启动训练 ====================
echo "🚀 开始运行 Stella (H800 Speed Mode)..."
echo "🧠 模型路径: $MODEL_PATH"
echo "📂 数据路径: $DATA_PATH"
echo "💾 输出路径: $OUTPUT_DIR"

# 运行 Python 脚本
python stella/experiments/commonsense/tools/finetune.py \
  --base_model "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_train_samples 10000 \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-4 \
  --cutoff_len 1024 \
  --val_set_size 1000 \
  --eval_step 50 \
  --save_step 1000 \
  --adapter_name Stella \
  --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
  --lora_r 32 \
  --lora_alpha 64 \
  --stella_init rando \
  --stella_retraction polar \
  --stella_diag_s True \
  --bf16 True \
  --fp16 False \
  2>&1 | tee -a "$OUTPUT_DIR/finetune.log"

echo "✅ 训练结束！"