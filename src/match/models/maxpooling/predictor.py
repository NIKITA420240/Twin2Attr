"""Inference adapter for the max-pooling model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..contracts import PredictionBatch
from .features import encode_attribute_pairs
from .model import MaxPoolingModel, resolve_device
from .serialization import load_maxpooling_model


def predict_maxpooling_probabilities(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    model: MaxPoolingModel,
    *,
    attributes_column: str = "attributes",
    batch_size: int = 512,
    device: str | torch.device | None = None,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features = encode_attribute_pairs(
        items,
        matches,
        model,
        attributes_column=attributes_column,
    )
    if not len(features):
        return np.empty(0, dtype=np.float32)

    scaled = model._scaler.transform(features).astype(np.float32, copy=False)
    target_device = resolve_device(device)
    classifier = model._classifier.to(target_device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (feature_batch,) in DataLoader(
            TensorDataset(torch.from_numpy(scaled)),
            batch_size=min(batch_size, len(scaled)),
            shuffle=False,
        ):
            logits = classifier(feature_batch.to(target_device))
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


@dataclass(slots=True)
class MaxPoolingPredictor:
    model: MaxPoolingModel
    batch_size: int = 512
    device: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        batch_size: int = 512,
        device: str | None = None,
    ) -> MaxPoolingPredictor:
        return cls(
            model=load_maxpooling_model(model_path),
            batch_size=batch_size,
            device=device,
        )

    @property
    def output_dim(self) -> int:
        return 2 * int(self.model.vector_size)

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        return encode_attribute_pairs(
            batch.items,
            batch.matches,
            self.model,
            attributes_column=batch.attributes_column,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        return predict_maxpooling_probabilities(
            batch.items,
            batch.matches,
            self.model,
            attributes_column=batch.attributes_column,
            batch_size=self.batch_size,
            device=self.device,
        )


__all__ = ["MaxPoolingPredictor", "predict_maxpooling_probabilities"]
