import os
import json
import matplotlib.pyplot as plt
import time

def read_loss_data(file_path):
    """读取 jsonl 文件，提取 step, loss 和 method"""
    steps = []
    losses = []
    method_name = "Unknown"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'loss' in data and data['loss'] is not None:
                        steps.append(data['step'])
                        losses.append(data['loss'])
                        if 'method' in data:
                            method_name = data['method']
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None, None, None

    return steps, losses, method_name

def main():
    # 1. 确定路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(current_dir, '..', 'outputs')
    
    # 定义保存图片的目录: outputs/loss
    save_dir = os.path.join(outputs_dir, 'loss')

    if not os.path.exists(outputs_dir):
        print(f"错误: 找不到 outputs 目录: {outputs_dir}")
        return

    # 2. 扫描文件夹
    subfolders = []
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        # 排除掉新建的 loss 文件夹，只看实验记录文件夹
        if item == 'loss': 
            continue
            
        log_file = os.path.join(item_path, 'unified_log.jsonl')
        if os.path.isdir(item_path) and os.path.exists(log_file):
            subfolders.append(item)
    
    subfolders.sort()

    if not subfolders:
        print("在 outputs 目录中没有找到包含 unified_log.jsonl 的文件夹。")
        return

    # 3. 用户交互
    print(f"\n在 '{outputs_dir}' 中发现以下实验记录:")
    for i, folder in enumerate(subfolders):
        print(f"[{i}] {folder}")
    
    print("\n请输入要绘制的实验序号 (例如: 0 2 3)，输入 'all' 绘制所有:")
    user_input = input(">> ").strip()

    selected_indices = []
    if user_input.lower() == 'all':
        selected_indices = range(len(subfolders))
    else:
        parts = user_input.replace(',', ' ').split()
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if 0 <= idx < len(subfolders):
                    selected_indices.append(idx)
            else:
                print(f"警告: '{p}' 不是有效的序号，已忽略。")

    if not selected_indices:
        print("未选择任何有效实验，程序退出。")
        return

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors 
    
    print("\n正在处理...")
    has_data = False
    
    for idx in selected_indices:
        folder_name = subfolders[idx]
        file_path = os.path.join(outputs_dir, folder_name, 'unified_log.jsonl')
        
        steps, losses, method = read_loss_data(file_path)
        
        if steps and losses:
            has_data = True
            plt.plot(steps, losses, label=method, linewidth=1.5, alpha=0.8)
            print(f" -> 添加曲线: {folder_name} (Method: {method})")
        else:
            print(f" -> 跳过: {folder_name} (无有效数据)")

    if not has_data:
        print("没有可绘制的数据。")
        return

    # 5. 图表设置与保存
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # --- 核心修改部分开始 ---
    
    # 确保 outputs/loss 目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"\n已自动创建保存目录: {save_dir}")

    # 生成带时间戳的文件名，防止覆盖
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"loss_plot_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    # 保存图片 (dpi=300 保证清晰度)
    plt.savefig(save_path, dpi=300)
    print(f"图片已保存至: {save_path}")
    
    # --- 核心修改部分结束 ---

    # 如果在图形界面环境下，依然显示出来
    try:
        plt.show()
    except Exception:
        pass # 如果在纯命令行服务器上，show可能会报错，直接忽略即可

if __name__ == "__main__":
    main()