# LoRA-Fine-Tuning on DistilBERT for Disaster-Tweet Classification 🚀

Fine-tuning **DistilBERT** with **Low-Rank Adaptation (LoRA)** on both CPU-only and GPU (CUDA) hardware.  
The project demonstrates how LoRA slashes train-time memory footprints while matching full-fine-tune accuracy.

---

## 📁 Repository layout

| Path | What’s inside |
|------|---------------|
| `notebooks/` | Two Jupyter notebooks: **CPU** and **GPU** variations of the workflow |
| 'Group-21.pdf' | The final presentation of our project |
| 'requirements.txt | Requirements file that contains the dependencies list for the project |


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
