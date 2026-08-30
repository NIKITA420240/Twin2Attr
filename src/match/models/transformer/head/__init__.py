"""Configurable pooling heads for Transformer sequence classification."""

from .config import PoolingHeadConfig
from .classifier import (
    GatedResidualFusionSequenceClassifier,
    GatedResidualFusionSequenceClassifierConfig,
    HybridSequenceClassifier,
    HybridSequenceClassifierConfig,
    PoolingSequenceClassifier,
    PoolingSequenceClassifierConfig,
)
from .gated_residual import GatedResidualFusionHead
from .model import TransformerPoolingHead

__all__ = [
    "PoolingHeadConfig",
    "GatedResidualFusionHead",
    "GatedResidualFusionSequenceClassifier",
    "GatedResidualFusionSequenceClassifierConfig",
    "PoolingSequenceClassifier",
    "PoolingSequenceClassifierConfig",
    "HybridSequenceClassifier",
    "HybridSequenceClassifierConfig",
    "TransformerPoolingHead",
]
