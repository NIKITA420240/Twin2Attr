"""Configurable pooling heads for Transformer sequence classification."""

from .config import PoolingHeadConfig
from .classifier import (
    HybridSequenceClassifier,
    HybridSequenceClassifierConfig,
    PoolingSequenceClassifier,
    PoolingSequenceClassifierConfig,
)
from .model import TransformerPoolingHead, TypedAttributeFusionHead

__all__ = [
    "PoolingHeadConfig",
    "PoolingSequenceClassifier",
    "PoolingSequenceClassifierConfig",
    "HybridSequenceClassifier",
    "HybridSequenceClassifierConfig",
    "TransformerPoolingHead",
    "TypedAttributeFusionHead",
]
