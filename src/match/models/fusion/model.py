"""Configuration, result and neural head for feature fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FusionConfig:
    hidden_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")


@dataclass(frozen=True)
class FusionTrainingResult:
    model_path: Path
    best_validation_macro_pr_auc: float
    trained_epochs: int
    cls_dim: int
    maxpooling_dim: int


class FusionClassifier(nn.Module):
    def __init__(
        self,
        cls_dim: int,
        maxpooling_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if cls_dim < 1 or maxpooling_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding and hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.cls_dim = cls_dim
        self.maxpooling_dim = maxpooling_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cls_normalization = nn.LayerNorm(cls_dim)
        self.maxpooling_normalization = nn.LayerNorm(maxpooling_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim + maxpooling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cls_embeddings: torch.Tensor,
        maxpooling_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if cls_embeddings.ndim != 2 or cls_embeddings.shape[1] != self.cls_dim:
            raise ValueError(f"cls_embeddings must have shape [B, {self.cls_dim}]")
        if (
            maxpooling_embeddings.ndim != 2
            or maxpooling_embeddings.shape[1] != self.maxpooling_dim
        ):
            raise ValueError(
                "maxpooling_embeddings must have shape "
                f"[B, {self.maxpooling_dim}]"
            )
        if cls_embeddings.shape[0] != maxpooling_embeddings.shape[0]:
            raise ValueError("embedding batches must contain the same number of rows")
        combined = torch.cat(
            (
                self.cls_normalization(cls_embeddings),
                self.maxpooling_normalization(maxpooling_embeddings),
            ),
            dim=1,
        )
        return self.classifier(combined).squeeze(1)


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_embedding_pair(
    cls_embeddings: np.ndarray,
    maxpooling_embeddings: np.ndarray,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for values, name in (
        (cls_embeddings, f"{split_name}_cls_embeddings"),
        (maxpooling_embeddings, f"{split_name}_maxpooling_embeddings"),
    ):
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty two-dimensional array")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(np.ascontiguousarray(array, dtype=np.float32))
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError(
            f"{split_name} embedding matrices must have the same row count"
        )
    return arrays[0], arrays[1]


__all__ = [
    "FusionClassifier",
    "FusionConfig",
    "FusionTrainingResult",
    "resolve_device",
    "validate_embedding_pair",
]
