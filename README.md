# Unsloth/Opensloth Fine-Tuning Project (Chat Export from "Fomo Sapiens" Telegram group)

This repository contains a fine-tuning script for LLMs using **Unsloth/Opensloth**.

## 🌟 Key Features
* **Pinned Dependencies:** Includes a `requirements.txt` with specific versions of `unsloth` and `torchao` to avoid the "int1" crash bug found in late Jan 2026.
* **Memory Efficient:** Uses Hugging Face **Streaming Mode** to handle large datasets without exhausting system RAM.
* **Kaggle Compatible:** Designed to be easily exported and run in Kaggle or local environments.

## 🚀 Getting Started - Run the script with multi-gpu support (2 GPUs):

torchrun --nproc_per_node=2 multigpu_torchrun.py

### 1. Clone the repository
```bash
git clone [https://github.com/NguyenGeorge/llama3-8B-Fomo-Sapiens-fine-tune.git](https://github.com/NguyenGeorge/llama3-8B-Fomo-Sapiens-fine-tune.git)
cd llama3-8B-Fomo-Sapiens-fine-tune
