"""
Hydra-free minimal trainer (for clarity).  Invoke via:
    python -m src.train --csv data/train.csv --output_dir artifacts/
"""
import argparse
from pathlib import Path

import evaluate
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.data.tweet_dataset import load_csv, tokenise, train_val_test_split
from src.models.lora_wrapper import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to raw Kaggle CSV")
    p.add_argument("--base_model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output_dir", default="artifacts/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds_raw = load_csv(args.csv)
    train_ds, val_ds, _ = train_val_test_split(ds_raw)
    train_tok = tokenise(train_ds, args.base_model)
    val_tok = tokenise(val_ds, args.base_model)

    model = build_model(args.base_model, num_labels=2).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    collator = DataCollatorWithPadding(tok)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return accuracy.compute(predictions=preds, references=labels)

    ta = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(Path(args.output_dir) / "lora_adapter")


if __name__ == "__main__":
    main()
