"""
Dataset utilities for the Kaggle “Real or Not Disaster Tweets” corpus.
"""
from pathlib import Path
from typing import Literal, Tuple

import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased"


def load_csv(path: str | Path) -> Dataset:
    """Load the Kaggle CSV (expects `text`, `target` columns)."""
    ds = datasets.load_dataset("csv", data_files=str(path))["train"]
    ds = ds.rename_column("target", "labels")
    return ds


def train_val_test_split(
    ds: Dataset, seed: int = 42, val_size: float = 0.1, test_size: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    train_val, test = train_test_split(  # first carve out test
        ds, test_size=test_size, shuffle=True, random_state=seed, stratify=ds["labels"]
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        shuffle=True,
        random_state=seed,
        stratify=train_val["labels"],
    )
    return Dataset.from_list(train), Dataset.from_list(val), Dataset.from_list(test)


def tokenise(
    ds: Dataset, model_name: str = DEFAULT_MODEL, max_len: int = 128
) -> Dataset:
    tok = AutoTokenizer.from_pretrained(model_name)

    def _tok(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    return ds.map(_tok, batched=True).remove_columns(["text"])


__all__: Tuple[Literal["load_csv", "train_val_test_split", "tokenise"]] = (
    "load_csv",
    "train_val_test_split",
    "tokenise",
)
