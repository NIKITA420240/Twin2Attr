"""Inference adapter for the fusion head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..contracts import PairEncoder, PredictionBatch
from .model import FusionClassifier, validate_embedding_pair
from .serialization import load_fusion_classifier


def predict_fusion_probabilities(
    model: FusionClassifier,
    cls_embeddings: np.ndarray,
    maxpooling_embeddings: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cls_array, maxpooling_array = validate_embedding_pair(
        cls_embeddings,
        maxpooling_embeddings,
        split_name="inference",
    )
    if cls_array.shape[1] != model.cls_dim:
        raise ValueError(f"CLS dimension must equal trained dimension {model.cls_dim}")
    if maxpooling_array.shape[1] != model.maxpooling_dim:
        raise ValueError(
            "max-pooling dimension must equal trained dimension "
            f"{model.maxpooling_dim}"
        )
    dataset = TensorDataset(
        torch.from_numpy(cls_array),
        torch.from_numpy(maxpooling_array),
    )
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for cls_batch, maxpooling_batch in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        ):
            logits = model(cls_batch.to(device), maxpooling_batch.to(device))
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


@dataclass(slots=True)
class FusionPredictor:
    model: FusionClassifier
    transformer: PairEncoder
    maxpooling: PairEncoder
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @classmethod
    def load(
        cls,
        fusion_path: str | Path,
        *,
        transformer: PairEncoder,
        maxpooling: PairEncoder,
        batch_size: int = 512,
        device: str | None = None,
    ) -> FusionPredictor:
        return cls(
            model=load_fusion_classifier(fusion_path, device=device),
            transformer=transformer,
            maxpooling=maxpooling,
            batch_size=batch_size,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        return predict_fusion_probabilities(
            self.model,
            self.transformer.encode(batch),
            self.maxpooling.encode(batch),
            batch_size=self.batch_size,
        )


__all__ = [
    "FusionPredictor",
    "predict_fusion_probabilities",
]
