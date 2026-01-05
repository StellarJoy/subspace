# 🌌 Project Subspace: Unified Subspace Optimization for LLM Fine-tuning

这是一个基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 构建的大模型微调算法复现与评测框架。

本项目致力于对子空间优化（Subspace Optimization）类算法进行**工程化统一**，实现了包括 Fira, Stella, FLoRA 等算法的复现，并提供了一致的训练接口、统一的日志记录系统以及批量化评测工具。

## ✨ 核心特性 (Key Features)

* **🔧 统一接口 (Unified Interface)**: 无论是运行官方的 LoRA 还是自定义的 Fira/Stella，均通过 `scripts/` 下的标准化 Shell 脚本一键启动。
* **📊 集中管理 (Centralized Management)**: 所有的实验配置 (`configs/`)、运行脚本 (`scripts/`) 和输出结果 (`outputs/`) 均由根目录统一管理，告别散乱的文件夹。
* **📝 全局日志 (Unified Logger)**: 内置自定义 Callback，自动将不同框架（LLaMA-Factory/Fira/Stella）的训练日志统一格式化为 `jsonl`，方便后续对比分析。

## 🧠 支持算法 (Supported Algorithms)

本项目集成了以下微调方法：

| 算法       | 来源                                                         | 说明                         | 运行方式                     |
| :--------- | :----------------------------------------------------------- | :--------------------------- | :--------------------------- |
| **LoRA**  | LLaMA-Factory                                                | 基座框架自带的主流 PEFT 方法 | `bash scripts/run_lora.sh`   |
| **DoRA**   | LLaMA-Factory                                                | 基座框架自带的主流 PEFT 方法 | `bash scripts/run_dora.sh`   |
| **GaLore** | LLaMA-Factory                                                | 基座框架自带的主流 PEFT 方法 | `bash scripts/run_galore.sh` |
| **pissa**  | LLaMA-Factory                                                | 基座框架自带的主流 PEFT 方法 | `bash scripts/run_pissa.sh`  |
| **Fira**   | [Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/pdf/2410.01623) | 复现 Fira 子空间分解算法     | `bash scripts/run_fira.sh`   |
| **Stella** | [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](https://arxiv.org/pdf/2510.01938) | 复现 Stella 算法             | `bash scripts/run_stella.sh` |

## 📂 项目结构 (Directory Structure)

```text
subspace/
├── configs/               # [配置] 存放所有 yaml 配置文件
│   ├── llama_factory/     # LLaMA-Factory 相关配置 (lora, dora 等)
├── scripts/               # [入口] 统一运行脚本 (Unified Entry Points)
│	├── run_lora.sh		   # 运行 lora 算法
│   ├── run_dora.sh        # 运行 dora 算法
│   ├── run_pissa.sh       # 运行 pissa 算法
│   ├── run_galore.sh      # 运行 galore 算法
│   ├── run_fira.sh        # 运行 Fira 算法
│   ├── run_stella.sh      # 运行 Stella 算法
│   └── eval.py            # 批量评测脚本
│   └── loss_plot.py       # 绘制损失曲线脚本
├── outputs/               # [输出] 所有的 Checkpoints 和 Logs 统一存放在此
├── data/                  # [数据] 数据集与 dataset_info.json,模型下载脚本
├── models/                # [模型] 基座模型 (Git ignored)
├── utils/                 # [工具] 通用工具包 (如 UnifiedLogger)
├── LLaMA-Factory/         # [核心] 基座训练框架 (作为子模块)
├── lm-evaluation-harness/ # [评测] 评估工具库
├── Fira/                  # [算法] Fira 源码
└── stella/                # [算法] Stella 源码

```

## 🛠️ 环境安装 (Installation)

本项目采用**分层依赖**管理。

Bash

```bash
# 1. 克隆仓库
git clone [https://github.com/your_username/subspace.git](https://github.com/your_username/subspace.git)
cd subspace

# 2. 安装基础环境 (LLaMA-Factory + lm-eval)
pip install -e "./LLaMA-Factory[metrics]"
pip install -e "./lm-evaluation-harness"

# 3. 安装扩展算法依赖 (如 Stella)
pip install -e "./stella"
```

## 🚀 快速开始 (Quick Start)

所有训练脚本均已配置了**自动根目录定位**，你可以在任何路径下运行它们。

### 1. 准备数据与模型

将你的数据放入 `data/` 目录，并在 `data/dataset_info.json` 中注册,将模型放入'models/'目录

### 2. 启动训练

Bash

```bash
# 运行 Fira
bash scripts/run_fira.sh

# 运行 Stella
bash scripts/run_stella.sh

# 运行标准 LoRA/DoRA等 (需修改 configs/llama_factory/ 下的 yaml)
bash scripts/run_lora.sh
```

### 3. 批量评测

训练完成后，使用统一评测脚本对 `outputs/` 下的所有模型进行评估：

Bash

```bash
# 脚本会自动扫描 outputs 目录下的模型并调用 lm-eval
python scripts/eval.py
```

结果将保存在 `eval_results/` 目录下。

## 📈 结果可视化 (Visualization)

由于集成了 `UnifiedLogger`，每个实验文件夹下都会生成 `unified_log.jsonl`。你可以使用简单的 Python 脚本读取并绘制 Loss 曲线对比图。