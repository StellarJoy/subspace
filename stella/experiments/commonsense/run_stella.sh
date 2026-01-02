#!/bin/bash
set -e
set -o xtrace

# ================= 配置区域 =================
# 模型绝对路径
BASE_MODEL='/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct' 

# 数据绝对路径 (直接复用 LLaMA-Factory 的数据)
DATA_PATH='/root/autodl-tmp/LLaMA-Factory/data/alpaca_en_demo.json'

# 输出路径 (建议放在 tmp 下，防止系统盘爆满)
WORK_DIR='/root/autodl-tmp/train_output/llama3_stella_experiment'

# 3. 实验名称与参数
MODEL='LLaMA3-8B'
ADAPTER='Stella'
LORA_R=32             # Rank，StelLA 可以在较小 Rank 下工作，但 32 更稳
LORA_ALPHA=64         # Alpha 通常是 Rank 的 2 倍
LR=0.0002             # 学习率，Llama3 建议 2e-4 或 1e-4
CUTOFF=512          # 序列长度，Llama3 支持 8k，视你显存决定 (推荐 2048 或 4096)
RETRACTION="polar"    # 核心：Stiefel 流形的收缩方法 (Polar 分解)
INIT="rando"          # 初始化方式：rando (随机) 或 svd (SVD分解初始化)

export TOKENIZERS_PARALLELISM=true

mkdir -p $WORK_DIR

# ================= 启动命令 =================
python tools/finetune.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$WORK_DIR" \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --learning_rate $LR \
    --cutoff_len $CUTOFF \
    --val_set_size 0 \
    --eval_step 200 \
    --save_step 200 \
    --adapter_name $ADAPTER \
    --target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --stella_init $INIT \
    --stella_retraction $RETRACTION \
    --stella_diag_s True \
    --bf16 True \
    --fp16 False \
    --use_gradient_checkpointing
    2>&1 | tee -a "$WORK_DIR/finetune.log"