"""Optional item-level feature enrichment with lazy public exports."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FeaturePipeline": (".pipeline", "FeaturePipeline"),
    "ItemEnricher": (".contracts", "ItemEnricher"),
    "ItemEnricherFactory": (".factory", "ItemEnricherFactory"),
    "PreparedItems": (".contracts", "PreparedItems"),
    "build_feature_pipeline": (".factory", "build_feature_pipeline"),
    "build_manifest_feature_pipeline": (
        ".factory",
        "build_manifest_feature_pipeline",
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
