"""Lazily exposed CatBoost matcher components."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BoostingFeatureBuilder": (".features", "BoostingFeatureBuilder"),
    "BoostingPredictor": (".predictor", "BoostingPredictor"),
    "BoostingTrainer": (".training", "BoostingTrainer"),
    "load_boosting_model": (".serialization", "load_boosting_model"),
    "save_boosting_model": (".serialization", "save_boosting_model"),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
