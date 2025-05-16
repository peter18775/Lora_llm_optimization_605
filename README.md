# LoRA-Fine-Tuning on DistilBERT for Disaster-Tweet Classification 🚀

Generative AI workloads—in particular, large language models (LLMs) and text-to-image diffusion models—demand heavy compute and memory resources for both fine-tuning and inference. Parameter-efficient fine-tuning techniques (e.g., LoRA, QLoRA) and post-training quantization (e.g., INT8) promise to reduce these requirements, but relative performance across different hardware tiers (CPUs, GPUs, TPUs) remains underexplored. In this work, we conduct a two-part study: (1) disaster-tweet classification using DistilBERT on CPU and GPU, comparing full fine-tuning versus LoRA; (2) Stable Diffusion 1.5 personalization on CPU and GPU, comparing LoRA, SVDiff, Custom Diffusion, and DreamBooth. We measure accuracy/quality, end-to-end training time, inference latency, peak memory, parameter efficiency, and energy consumption. Our results show that GPUs deliver 10–20× speedups and roughly 3× lower memory usage versus CPUs; LoRA on GPU achieves 95 % of full fine-tune quality at 11× faster training and one-quarter the memory footprint; among diffusion personalization methods, SVDiff offers the best trade-off (0.4 ‰ trainable params, zero latency penalty) followed by LoRA, Custom Diffusion, and DreamBooth. These findings provide concrete guidance for selecting cost-effective, hardware-aware tuning strategies in resource-constrained 

---

## 📁 Repository layout

| Path | What’s inside |
|------|---------------|
| `notebooks/` | (Optional) original exploratory notebooks kept for reference |
| `src/` | **Production code** – dataset utils, LoRA wrapper, training & inference CLIs |
| `src/data/` | `tweet_dataset.py` for loading, splitting, and tokenising the Kaggle corpus |
| `src/models/` | `lora_wrapper.py` that inserts / loads LoRA adapters into any HF model |
| `src/train.py` | CLI script: launches a Hugging-Face `Trainer` run with LoRA |
| `src/inference.py` | Tiny CLI for single-tweet prediction with a saved adapter |
| `requirements.txt` | Locked package versions (PyTorch 2.7 + Transformers 4.51, etc.) |
| `README.md` | **← you are here** |



---

## 🔧 Quick start

```bash
# 1. Clone and enter
git clone https://github.com/peter18775/Hardware-Aware_Neural_network.git
cd lora-llm-finetune

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run a notebook (CPU example)
jupyter lab notebooks/gpu-lora-llm.ipynb #we are running gpu code only because we know the outcome of how cpu will take 15 hrs to complete the task.
