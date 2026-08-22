"""Lazily exposed max-pooling components."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MaxPoolingModel": (".model", "MaxPoolingModel"),
    "MaxPoolingPredictor": (".predictor", "MaxPoolingPredictor"),
    "MaxPoolingTrainer": (".training", "MaxPoolingTrainer"),
    "PairMLP": (".model", "PairMLP"),
    "encode_attribute_pairs": (".features", "encode_attribute_pairs"),
    "load_maxpooling_model": (".serialization", "load_maxpooling_model"),
    "predict_maxpooling_probabilities": (
        ".predictor",
        "predict_maxpooling_probabilities",
    ),
    "save_maxpooling_model": (".serialization", "save_maxpooling_model"),
    "train_maxpooling_model": (".training", "train_maxpooling_model"),
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
