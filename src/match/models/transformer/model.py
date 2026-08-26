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
from .head import PoolingHeadConfig, PoolingSequenceClassifier
from .optimizer import LearningRateMultipliers, build_transformer_optimizer


AUGMENTED_INPUT_PREFIX = "augmented_"


class WeightedSequenceTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        learning_rate_multipliers: LearningRateMultipliers | None = None,
        augmentation_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= augmentation_alpha <= 1.0:
            raise ValueError("augmentation_alpha must be in [0, 1]")
        self.learning_rate_multipliers = (
            learning_rate_multipliers or LearningRateMultipliers()
        )
        self.augmentation_alpha = augmentation_alpha
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
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights")
        augmented_inputs = {
            name.removeprefix(AUGMENTED_INPUT_PREFIX): inputs.pop(name)
            for name in tuple(inputs)
            if name.startswith(AUGMENTED_INPUT_PREFIX)
        }
        outputs = model(**inputs)
        original_loss = self._weighted_cross_entropy(
            outputs.logits,
            labels,
            sample_weights,
        )
        if augmented_inputs:
            augmented_outputs = model(**augmented_inputs)
            augmented_loss = self._weighted_cross_entropy(
                augmented_outputs.logits,
                labels,
                sample_weights,
            )
            loss = (
                self.augmentation_alpha * original_loss
                + (1.0 - self.augmentation_alpha) * augmented_loss
            )
        else:
            loss = original_loss
        return (loss, outputs) if return_outputs else loss

    def _weighted_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        losses = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
            reduction="none",
        )
        weights = sample_weights.to(logits.device)
        return torch.sum(losses * weights) / torch.sum(weights)


def model_factory(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
    head_type: str = "default",
    head_config: PoolingHeadConfig | None = None,
):
    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        if head_type == "pooling":
            model = PoolingSequenceClassifier.from_backbone_pretrained(
                model_path,
                head_config=head_config or PoolingHeadConfig(),
                num_labels=2,
                id2label={0: "different", 1: "match"},
                label2id={"different": 0, "match": 1},
            )
        elif head_type == "default":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                id2label={0: "different", 1: "match"},
                label2id={"different": 0, "match": 1},
                ignore_mismatched_sizes=True,
            )
        else:
            raise ValueError("head_type must be 'default' or 'pooling'")
        if use_field_tokens:
            add_pair_special_tokens(tokenizer, model)
        return model

    return initialize_model


__all__ = ["WeightedSequenceTrainer", "model_factory"]
