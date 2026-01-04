import json
import time
import torch
import os
from transformers import TrainerCallback

class UnifiedLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path, method_name):
        self.log_file_path = log_file_path
        self.method_name = method_name
        self.start_time = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # 新实验清理旧日志
        if os.path.exists(log_file_path):
            try:
                os.remove(log_file_path)
            except OSError:
                pass

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        # 尝试重置显存统计（有些环境可能不支持，加个try）
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            current_time = time.time()
            # 获取显存峰值 (GB)
            try:
                peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            except:
                peak_reserved_gb = 0.0
            
            log_entry = {
                "method": self.method_name,
                "step": state.global_step,
                "epoch": logs.get("epoch", 0.0),
                "loss": logs.get("loss", None),
                "eval_loss": logs.get("eval_loss", None),
                "learning_rate": logs.get("learning_rate", 0.0),
                "elapsed_seconds": current_time - self.start_time if self.start_time else 0,
                "peak_memory_gb": peak_reserved_gb
            }
            
            # 使用 jsonl 格式追加写入
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")