import os
import subprocess
import sys

# ==================== 1. 魔法前缀 (自动定位工程目录) ====================
# 获取当前脚本位置 (subspace/scripts/batch_eval.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (subspace/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

print(f"📍 项目根目录已定位: {PROJECT_ROOT}")

# ==================== 2. 核心配置区 ====================

# [基座模型路径]
# 使用 os.path.join 拼接，兼容各种操作系统，且不依赖绝对路径
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/LLM-Research/Meta-Llama-3-8B-Instruct")

# [测试任务]
TASKS = "piqa,boolq,arc_easy"

# [待测模型列表]
# 格式: "显示名称": "相对于 outputs 的文件夹名" (或者完整路径)
# 我根据之前帮你改的脚本，填好了这三个路径，你可以根据实际跑出来的情况调整
MODELS_TO_TEST = {
    # LLaMA-Factory 的结果 (之前改名为 llama3-8b-dora-sft)
    "LoRA_Factory": os.path.join(PROJECT_ROOT, "outputs/lora_rank8"),

    # Fira 的结果 (之前脚本里设定的)
    #"Fira_LoRA":    os.path.join(PROJECT_ROOT, "outputs/fira_llama3_8b"),

    # Stella 的结果 (之前脚本里设定的)
    #"Stella_H800":  os.path.join(PROJECT_ROOT, "outputs/llama3_stella_h800_speed_run"),
}

# [评估结果保存位置]
# 建议也放在 outputs 里，或者单独一个 eval_results 文件夹
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval_results")

# [显卡设置]
GPU_ID = "0"

# ====================================================

def run_eval():
    print(f"\n🚀 开始批量评估任务: {TASKS}")
    print(f"📂 结果将保存在: {EVAL_OUTPUT_DIR}\n")

    # 确保结果目录存在
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # 设置环境变量 (如果需要)
    os.environ["HF_TOKEN"] = "" # 如果服务器已有环境无需重复设置
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    for algo_name, model_path in MODELS_TO_TEST.items():
        print(f"{'='*60}")
        print(f"🧪 正在评估: {algo_name}")
        print(f"🔍 模型路径: {model_path}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"❌ 警告: 路径不存在，跳过此模型！")
            continue

        # --- 智能判断模型类型 ---
        # 检查目录下是否有 adapter_config.json
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_adapter:
            print("🤖 类型识别: PEFT Adapter (LoRA/DoRA/PiSSA)")
            # 语法: pretrained=BaseModel,peft=AdapterPath
            model_args = f"pretrained={BASE_MODEL_PATH},peft={model_path},dtype=float16"
        else:
            print("🤖 类型识别: Full Model (全量权重)")
            # 语法: pretrained=ModelPath
            model_args = f"pretrained={model_path},dtype=float16"

        # 构造输出文件路径
        output_file = os.path.join(EVAL_OUTPUT_DIR, f"result_{algo_name}.json")
        
        # 构造 lm_eval 命令
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", TASKS,
            "--num_fewshot", "0",
            "--batch_size", "auto",
            "--device", f"cuda:{GPU_ID}",
            "--output_path", output_file
        ]

        # 打印并执行命令
        print(f"🏃 执行命令: {' '.join(cmd)}")
        
        try:
            # 实时输出日志
            subprocess.run(cmd, check=True)
            print(f"✅ {algo_name} 评估完成！")
        except subprocess.CalledProcessError:
            print(f"❌ {algo_name} 评估过程中出错！")
        except FileNotFoundError:
            print("❌ 错误: 未找到 'lm_eval' 命令。请确保你已经 pip install lm-eval 并且环境已激活。")

    print(f"\n🎉 所有测试结束！请查看 {EVAL_OUTPUT_DIR} 文件夹。")

if __name__ == "__main__":
    run_eval()