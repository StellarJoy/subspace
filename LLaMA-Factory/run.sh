#!/bin/bash
echo "=== 开始训练 LoRA ==="
export EXP_NAME="LoRA"
llamafactory-cli train config/llama3_lora.yaml

echo "=== 开始训练 DoRA ==="
export EXP_NAME="DoRA"
llamafactory-cli train config/llama3_dora.yaml

export EXP_NAME="pissa"
llamafactory-cli train config/llama3_pissa.yaml

echo "=== 全部搞定 ==="