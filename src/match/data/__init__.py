"""Lazily exposed data APIs for training and inference workflows."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "TrainingMatchPaths": (".loading", "TrainingMatchPaths"),
    "load_training_matches": (".loading", "load_training_matches"),
    "read_parquet": (".loading", "read_parquet"),
    "TrainingData": (".preparation", "TrainingData"),
    "prepare_pair_rows": (".preparation", "prepare_pair_rows"),
    "prepare_training_data": (".preparation", "prepare_training_data"),
    "prepare_configured_items": (
        ".preprocessing",
        "prepare_configured_items",
    ),
    "prepare_manifest_items": (
        ".preprocessing",
        "prepare_manifest_items",
    ),
}

__all__ = sorted(_EXPORTS)


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


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
