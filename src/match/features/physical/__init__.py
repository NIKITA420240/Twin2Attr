"""Lazy exports for rule-based physical attribute extraction."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PhysicalAttributeParser": (".parser", "PhysicalAttributeParser"),
    "PhysicalItemEnricher": (".enrichment", "PhysicalItemEnricher"),
    "PhysicalUnitNormalizer": (".normalizer", "PhysicalUnitNormalizer"),
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
