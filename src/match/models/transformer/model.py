"""Transformer model construction and weighted Hugging Face trainer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from ...pair_encoding import add_pair_special_tokens


class WeightedSequenceTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.detach().to(dtype=torch.float32)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights")
        outputs = model(**inputs)
        losses = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            reduction="none",
        )
        weights = sample_weights.to(outputs.logits.device)
        loss = torch.sum(losses * weights) / torch.sum(weights)
        return (loss, outputs) if return_outputs else loss


def model_factory(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
):
    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            ignore_mismatched_sizes=True,
        )
        if use_field_tokens:
            add_pair_special_tokens(tokenizer, model)
        return model

    return initialize_model


__all__ = ["WeightedSequenceTrainer", "model_factory"]
