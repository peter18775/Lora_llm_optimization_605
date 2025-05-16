"""
CLI for single-text inference with the saved LoRA adapter.

Example:
    python -m src.inference \
        --adapter artifacts/lora_adapter \
        --text "Forest fire near La Ronge Sask. Canada"
"""
import argparse

import torch
from transformers import AutoTokenizer

from src.models.lora_wrapper import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument(
        "--base_model", default="distilbert-base-uncased", help="Same as during training"
    )
    p.add_argument("--text", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = build_model(
        args.base_model, num_labels=2, load_adapter=args.adapter
    ).eval()

    inputs = tok(args.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(-1).item()
        prob = logits.softmax(-1)[0, pred].item()
    label = "DISASTER 🚨" if pred == 1 else "NON-DISASTER ✅"
    print(f"{label} ({prob:.2%})")


if __name__ == "__main__":
    main()
