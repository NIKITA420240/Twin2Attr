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
    """Compute weighted binary loss for hard or soft targets in ``[0, 1]``."""
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
    labels = labels.to(device=device, dtype=logits.dtype)
    sample_weights = sample_weights.to(device=device, dtype=logits.dtype)
    class_weights = class_weights.to(device=device, dtype=logits.dtype)
    if not torch.isfinite(labels).all() or torch.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must contain finite values in [0, 1]")
    if torch.any(sample_weights <= 0) or torch.any(class_weights <= 0):
        raise ValueError("sample and class weights must be positive")

    if logits.shape[1] == 1:
        scores = logits[:, 0]
        losses = -(
            (1.0 - labels) * class_weights[0] * F.logsigmoid(-scores)
            + labels * class_weights[1] * F.logsigmoid(scores)
        )
    else:
        log_probabilities = F.log_softmax(logits, dim=-1)
        weighted_classes = torch.stack(
            (
                (1.0 - labels) * class_weights[0],
                labels * class_weights[1],
            ),
            dim=-1,
        )
        losses = -torch.sum(
            weighted_classes * log_probabilities,
            dim=-1,
        )
    return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


__all__ = ["weighted_classification_loss"]
