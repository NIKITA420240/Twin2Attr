"""Common model capabilities and concrete inference adapters."""

from importlib import import_module
from typing import Any

from .contracts import MatchPredictor, PairEncoder, PredictionBatch
from .predictors import FusionPredictor, MaxPoolingPredictor, TransformerPredictor

_LAZY_EXPORTS = {
    "FusionTrainer": ".training",
    "MaxPoolingTrainer": ".training",
    "ModelTrainer": ".training",
    "TrainingArtifacts": ".artifacts",
    "TransformerTrainer": ".training",
    "build_trainer": ".training",
    "save_solution_manifest": ".artifacts",
}

__all__ = [
    "FusionPredictor",
    "FusionTrainer",
    "MatchPredictor",
    "MaxPoolingPredictor",
    "MaxPoolingTrainer",
    "ModelTrainer",
    "PairEncoder",
    "PredictionBatch",
    "TrainingArtifacts",
    "TransformerPredictor",
    "TransformerTrainer",
    "build_trainer",
    "save_solution_manifest",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
