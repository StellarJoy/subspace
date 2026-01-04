#!/bin/bash

# ==================== 1. 魔法前缀 ====================
# 自动定位到 subspace 根目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

# 这一步是为了防止你以后修改了 LLaMA-Factory 代码但系统调用了旧的包
# 把本地的 LLaMA-Factory/src 加入环境变量，确保用的是你仓库里的代码
export PYTHONPATH="$PROJECT_ROOT/LLaMA-Factory/src:$PYTHONPATH"

echo "📍 工作目录已切换至: $(pwd)"

# ==================== 2. 配置区域 ====================
# 指定你的配置文件路径 (建议把 yaml 移到 configs/llama_factory/ 下)
CONFIG_FILE="$PROJECT_ROOT/configs/llama_factory/llama3_pissa.yaml"

# 显卡设置
export CUDA_VISIBLE_DEVICES=0

# ==================== 3. 运行命令 ====================
echo "🚀 启动 LLaMA-Factory 训练..."
echo "📄 配置文件: $CONFIG_FILE"

# 调用 CLI，但路径全部动态生成
python -m llamafactory.cli train "$CONFIG_FILE"