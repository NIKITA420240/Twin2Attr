"""Lazily exposed fusion components."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FusionClassifier": (".model", "FusionClassifier"),
    "FusionConfig": (".model", "FusionConfig"),
    "FusionPredictor": (".predictor", "FusionPredictor"),
    "FusionTrainer": (".training", "FusionTrainer"),
    "FusionTrainingResult": (".model", "FusionTrainingResult"),
    "load_fusion_classifier": (".serialization", "load_fusion_classifier"),
    "predict_fusion_probabilities": (".predictor", "predict_fusion_probabilities"),
    "save_fusion_classifier": (".serialization", "save_fusion_classifier"),
    "train_fusion_classifier": (".training", "train_fusion_classifier"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
