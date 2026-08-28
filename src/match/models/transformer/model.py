"""Transformer model construction and weighted Hugging Face trainer."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, Trainer

from .construction import model_factory
from .optimizer import (
    LearningRateMultipliers,
    build_transformer_optimizer,
)
from .objective import weighted_classification_loss


class WeightedSequenceTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        learning_rate_multipliers: LearningRateMultipliers | None = None,
        **kwargs: Any,
    ) -> None:
        self.learning_rate_multipliers = (
            learning_rate_multipliers or LearningRateMultipliers()
        )
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.detach().to(dtype=torch.float32)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = build_transformer_optimizer(
                self.model,
                backbone_lr=float(self.args.learning_rate),
                multipliers=self.learning_rate_multipliers,
                weight_decay=float(self.args.weight_decay),
            )
        return self.optimizer

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        sample_weights = model_inputs.pop("sample_weights")
        outputs = model(**model_inputs)
        loss = weighted_classification_loss(
            outputs.logits,
            labels,
            sample_weights,
            self.class_weights,
        )
        return (loss, outputs) if return_outputs else loss


__all__ = ["WeightedSequenceTrainer", "model_factory"]
