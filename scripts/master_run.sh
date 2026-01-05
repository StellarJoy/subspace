#!/bin/bash

# --- 待训练清单：你可以根据需要自由调整顺序，或注释掉不想跑的 ---
scripts=(
    "run_dora.sh"
    #"run_fira.sh"
    #"run_galore.sh"
    "run_lora_dora.sh"
    "run_lora_fira.sh"
    "run_lora_galore.sh"
    #"run_lora_pissa.sh"
    "run_lora_stella.sh"
    #"run_pissa.sh"
    #"run_stella.sh"
)

echo "==== 自动化批量训练开始 ===="
echo "总共有 ${#scripts[@]} 个任务待执行"

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "----------------------------------------"
        echo "正在执行: $script"
        
        # 执行脚本
        # 使用 bash 运行，即使脚本没有执行权限也能跑
        bash "$script"
        
        # 核心逻辑：无论上一个脚本成功还是失败，$? 都会被捕捉，
        # 我们这里不退出，直接进入下一次循环
        if [ $? -eq 0 ]; then
            echo "✅ $script 执行成功"
        else
            echo "❌ $script 运行出错（已自动跳过，准备下一个）"
        fi
    else
        echo "⚠️ 找不到文件: $script，跳过。"
    fi
done

echo "----------------------------------------"
echo "所有任务已尝试完毕！可以去睡觉了，晚安。"