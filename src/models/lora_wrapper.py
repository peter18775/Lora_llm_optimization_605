"""
Insert LoRA adapters into any HF sequence-classification model.
"""
from dataclasses import dataclass
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification


@dataclass
class LoRAHyperParams:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "value")


def build_model(
    base_model: str,
    num_labels: int,
    lora_cfg: LoRAHyperParams | None = None,
    load_adapter: str | None = None,
) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )

    if load_adapter:  # inference path
        model.load_adapter(load_adapter)
        return model

    lora = lora_cfg or LoRAHyperParams()
    config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=lora.target_modules,
    )
    return get_peft_model(model, config)
