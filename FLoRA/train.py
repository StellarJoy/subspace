import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
# 确保你是在 FLoRA 目录下运行此脚本，这样才能 import flora
from flora import FLoRA

# ================= 配置区域 =================
# 1. 模型绝对路径
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# 2. 数据集绝对路径
DATA_PATH = "/root/autodl-tmp/LLaMA-Factory/data/alpaca_en_demo.json"

# 3. 输出路径 (训练日志和权重会保存在当前目录下的 output 文件夹)
OUTPUT_DIR = "./output_llama3_flora"
# ===========================================

class AlpacaDataset(Dataset):
    """
    专门读取 Alpaca 格式 json 的数据集类，并处理成 Llama-3 的对话格式
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"正在加载数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"加载完成，共 {len(self.data)} 条数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # 构建 Llama-3 格式的 prompt
        # 格式: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        if input_text:
            user_prompt = f"{instruction}\nInput:\n{input_text}"
        else:
            user_prompt = instruction

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output_text},
        ]
        
        # 使用 tokenizer 的 apply_chat_template 自动处理特殊符号
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        # 转换为 token id
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # 简单的做法：labels 就是 input_ids (Masking 会由模型自动处理，或者全量学习)
        # 如果想更严谨只计算 answer 的 loss，需要手动处理掩码，这里为了测试跑通暂时全量
        labels = input_ids.clone()
        
        # 将 pad token 的 label 设为 -100，避免计算 loss
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    # 1. 加载模型和 Tokenizer
    print("正在加载 Tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Llama3 必须设置 pad_token

    # 推荐使用 bfloat16 加载，如果你显卡不支持，可以改为 float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 配置 FLoRA
    # Llama3 的主要层都在 layers.x.self_attn 和 mlp 中
    # 名称通常为: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    flora_params = dict(
        target_keys=['q_proj', 'v_proj', 'gate_proj', 'down_proj'], # 选取几个关键层进行微调
        base_name='model',
        cls_types=['linear'], # Llama3 全是 Linear 层
        flora_cfg=dict(
            N=2,            # Linear 对应 N=2
            r=[16, 16],     # 秩
            scale=4.0,      # 缩放因子
            drop_rate=0.05,
        ),
    )

    print("正在包装 FLoRA...")
    flora_model = FLoRA(model, **flora_params)
    
    
    # 打印参数量看是否生效
    trainable_params = sum(p.numel() for p in flora_model.model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in flora_model.model.parameters())
    print(f"Trainable params: {trainable_params} | All params: {all_params} | Ratio: {trainable_params/all_params:.4%}")

    # 3. 准备数据
    dataset = AlpacaDataset(DATA_PATH, tokenizer)

    # 4. 训练参数设置
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # 显存不够可以调小
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        num_train_epochs=1,            # 测试跑 1 个 epoch 即可
        learning_rate=2e-4,            # FLoRA 学习率可以稍大
        logging_steps=5,
        save_steps=300,
        bf16=True,                     # 开启 bf16 加速
        remove_unused_columns=False,   # 防止删除我们需要的数据列
    )

    if args.gradient_checkpointing:
        print("检测到开启了梯度检查点，正在解冻输入层...")
        if hasattr(flora_model.model, "model") and hasattr(flora_model.model.model, "embed_tokens"):
             flora_model.model.model.embed_tokens.weight.requires_grad_(True)
        elif hasattr(flora_model.model, "get_input_embeddings"):
             flora_model.model.get_input_embeddings().weight.requires_grad_(True)
        else:
             print("警告：未找到输入嵌入层")
             
        flora_model.model.config.use_cache = False
    # 5. 开始训练
    trainer = Trainer(
        model=flora_model.model,
        args=args,
        train_dataset=dataset,
    )

    print("开始训练...")
    trainer.train()
    
    # 6. 保存权重
    print(f"训练结束，保存权重到 {OUTPUT_DIR}")
    save_path = os.path.join(OUTPUT_DIR, "flora_final.pt")
    
    # 手动筛选 FLoRA 权重进行保存
    state_dict = {k: v.cpu() for k, v in flora_model.model.state_dict().items() if "param_" in k or "param_G" in k}
    torch.save(state_dict, save_path)
    print("权重保存完成。")

if __name__ == "__main__":
    main()