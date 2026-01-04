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
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # 如果是新实验，清理旧日志；如果是断点续训，可以注释掉下面两行
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            current_time = time.time()
            # 获取显存峰值 (GB)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
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
            
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")