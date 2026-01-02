# 运行前请确保你在 fine_tuning 目录下
# 并且已经下载好了 commonsense_170k.json

# 设置使用单卡 GPU (例如 4090)
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct' \
#  --load_8bit \
  --data_path '/root/autodl-tmp/LLaMA-Factory/data/alpaca_en_demo.json' \
  --output_dir './result/fira_llama3_8b' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --save_step 15000 \
  --eval_step 1000 \
  --optimizer_name fira_adamw