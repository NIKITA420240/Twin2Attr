"""Backend-independent training objective for binary Transformer classifiers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted BCE for one logit or cross-entropy for two logits."""
    if logits.ndim != 2 or logits.shape[1] not in {1, 2}:
        raise ValueError("classifier must return shape [batch, 1 or 2]")
    batch_size = logits.shape[0]
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("labels must have shape [batch]")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != batch_size:
        raise ValueError("sample_weights must have shape [batch]")
    if class_weights.shape != (2,):
        raise ValueError("class_weights must have shape [2]")

    device = logits.device
    labels = labels.to(device=device, dtype=torch.long)
    sample_weights = sample_weights.to(device=device, dtype=logits.dtype)
    class_weights = class_weights.to(device=device, dtype=logits.dtype)
    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must contain only 0 and 1")
    if torch.any(sample_weights <= 0) or torch.any(class_weights <= 0):
        raise ValueError("sample and class weights must be positive")

    if logits.shape[1] == 1:
        losses = F.binary_cross_entropy_with_logits(
            logits[:, 0],
            labels.to(dtype=logits.dtype),
            reduction="none",
        )
        losses = losses * class_weights[labels]
    else:
        losses = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            reduction="none",
        )
    return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


__all__ = ["weighted_classification_loss"]
