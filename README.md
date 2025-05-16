# LoRA-Fine-Tuning on DistilBERT for Disaster-Tweet Classification 🚀

Fine-tuning **DistilBERT** with **Low-Rank Adaptation (LoRA)** on both CPU-only and GPU (CUDA) hardware.  
The project demonstrates how LoRA slashes train-time memory footprints while matching full-fine-tune accuracy.

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
