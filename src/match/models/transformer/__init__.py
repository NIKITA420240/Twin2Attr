"""Lazily exposed Transformer components."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ResolvedTrainingConfig": (".config", "ResolvedTrainingConfig"),
    "SequenceClassifierConfig": (".config", "SequenceClassifierConfig"),
    "TrainingResult": (".config", "TrainingResult"),
    "TransformerPredictor": (".predictor", "TransformerPredictor"),
    "TransformerTrainer": (".training", "TransformerTrainer"),
    "compute_class_weights": (".metrics", "compute_class_weights"),
    "compute_macro_pr_auc": (".metrics", "compute_macro_pr_auc"),
    "compute_pr_auc": (".metrics", "compute_pr_auc"),
    "encode_pair_cls": (".predictor", "encode_pair_cls"),
    "load_trained_classifier": (".predictor", "load_trained_classifier"),
    "predict_match_probabilities": (".predictor", "predict_match_probabilities"),
    "predict_logit_margins": (".predictor", "predict_logit_margins"),
    "predict_pair_logits": (".predictor", "predict_pair_logits"),
    "train_sequence_classifier": (".training", "train_sequence_classifier"),
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
