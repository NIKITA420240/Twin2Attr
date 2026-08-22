"""Common model capabilities and concrete inference adapters."""

from .contracts import MatchPredictor, PairEncoder, PredictionBatch
from .predictors import FusionPredictor, MaxPoolingPredictor, TransformerPredictor

__all__ = [
    "FusionPredictor",
    "MatchPredictor",
    "MaxPoolingPredictor",
    "PairEncoder",
    "PredictionBatch",
    "TransformerPredictor",
]
