"""Lazily exposed word-level NER components."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NerEntity": (".entities", "NerEntity"),
    "NerExtractor": (".entities", "NerExtractor"),
    "NerItemEnricher": (".enrichment", "NerItemEnricher"),
    "WordNERModel": (".model", "WordNERModel"),
    "WordNerPredictor": (".predictor", "WordNerPredictor"),
    "load_word_ner_predictor": (".serialization", "load_word_ner_predictor"),
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
