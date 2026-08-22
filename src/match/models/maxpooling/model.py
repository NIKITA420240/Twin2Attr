"""Model state and neural classifier for max-pooling features."""

from dataclasses import dataclass

import torch
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
from torch import nn


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


@dataclass(slots=True)
class MaxPoolingModel:
    _fasttext: FastText
    _scaler: StandardScaler
    _classifier: PairMLP
    vector_size: int
    best_validation_auc: float
    best_validation_pr_auc: float


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = ["MaxPoolingModel", "PairMLP", "resolve_device"]
