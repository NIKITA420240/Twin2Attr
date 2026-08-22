"""Metrics and class balancing for Transformer training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import EvalPrediction


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not set(np.unique(label_array)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    counts = np.bincount(label_array, minlength=2)
    if np.any(counts == 0):
        raise ValueError("both classes must be present in training labels")
    weights = label_array.size / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _prediction_values(
    prediction: EvalPrediction | tuple[Any, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(prediction, EvalPrediction):
        logits = prediction.predictions
        labels = prediction.label_ids
    else:
        logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("expected logits with shape (n_samples, 2)")
    return logits, labels


def _positive_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated[:, 1] / exponentiated.sum(axis=1)


def compute_pr_auc(
    prediction: EvalPrediction | tuple[Any, Any],
) -> dict[str, float]:
    logits, labels = _prediction_values(prediction)
    probabilities = _positive_probabilities(logits)
    return {"pr_auc": float(average_precision_score(labels, probabilities))}


def compute_macro_pr_auc(
    prediction: EvalPrediction | tuple[Any, Any],
    categories: Sequence[str],
) -> dict[str, float]:
    logits, labels = _prediction_values(prediction)
    probabilities = _positive_probabilities(logits)
    category_values = np.asarray(categories)
    category_scores = [
        average_precision_score(
            labels[category_values == category],
            probabilities[category_values == category],
        )
        for category in np.unique(category_values)
    ]
    return {"macro_pr_auc": float(np.mean(category_scores))}


__all__ = ["compute_class_weights", "compute_macro_pr_auc", "compute_pr_auc"]
