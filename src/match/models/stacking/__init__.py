"""Transformer-to-CatBoost stacking model."""

from .predictor import StackingPredictor
from .training import StackingTrainer

__all__ = ["StackingPredictor", "StackingTrainer"]
