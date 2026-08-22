"""Twin2Attr public API with lazy imports for lightweight CLI startup."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AppConfig": (".config", "AppConfig"),
    "CONFIG_DIR": (".paths", "CONFIG_DIR"),
    "DATA_DIR": (".paths", "DATA_DIR"),
    "DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS": (
        ".pair_encoding",
        "DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS",
    ),
    "DataSplitConfig": (".data_split", "DataSplitConfig"),
    "DataSplitResult": (".data_split", "DataSplitResult"),
    "FusionClassifier": (".fusion", "FusionClassifier"),
    "FusionConfig": (".fusion", "FusionConfig"),
    "FusionPredictor": (".models", "FusionPredictor"),
    "FusionTrainingResult": (".fusion", "FusionTrainingResult"),
    "KEY_TOKEN": (".pair_encoding", "KEY_TOKEN"),
    "MatchPredictor": (".models", "MatchPredictor"),
    "MaxPoolingPredictor": (".models", "MaxPoolingPredictor"),
    "PROJECT_ROOT": (".paths", "PROJECT_ROOT"),
    "PairEncodingCollator": (".pair_encoding", "PairEncodingCollator"),
    "PairEncoder": (".models", "PairEncoder"),
    "PredictionBatch": (".models", "PredictionBatch"),
    "PreparedCard": (".prepare_data", "PreparedCard"),
    "PreparedPair": (".prepare_data", "PreparedPair"),
    "PreparedPairDataset": (".pair_encoding", "PreparedPairDataset"),
    "SequenceClassifierConfig": (".transformer", "SequenceClassifierConfig"),
    "TrainingResult": (".transformer", "TrainingResult"),
    "TransformerPredictor": (".models", "TransformerPredictor"),
    "VAL_TOKEN": (".pair_encoding", "VAL_TOKEN"),
    "add_pair_special_tokens": (".pair_encoding", "add_pair_special_tokens"),
    "encode_attribute_pairs": (".maxpooling", "encode_attribute_pairs"),
    "encode_pair_cls": (".transformer", "encode_pair_cls"),
    "encode_prepared_pair": (".pair_encoding", "encode_prepared_pair"),
    "infer_pair_max_length": (".pair_encoding", "infer_pair_max_length"),
    "inspect_max_length": (".workflows.inspect", "inspect_max_length"),
    "load_fusion_classifier": (".fusion", "load_fusion_classifier"),
    "load_app_config": (".config", "load_app_config"),
    "load_app_config_file": (".config", "load_app_config_file"),
    "load_trained_classifier": (".transformer", "load_trained_classifier"),
    "normalize_attributes": (".normalization", "normalize_attributes"),
    "predict_fusion_probabilities": (".fusion", "predict_fusion_probabilities"),
    "predict_match_probabilities": (".transformer", "predict_match_probabilities"),
    "predict_maxpooling_probabilities": (
        ".maxpooling",
        "predict_maxpooling_probabilities",
    ),
    "prepare_cards": (".prepare_data", "prepare_cards"),
    "prepare_pairs": (".prepare_data", "prepare_pairs"),
    "resolve_project_path": (".paths", "resolve_project_path"),
    "serialize_card": (".pair_encoding", "serialize_card"),
    "serialize_pair": (".pair_encoding", "serialize_pair"),
    "split_matches": (".data_split", "split_matches"),
    "train_fusion_classifier": (".fusion", "train_fusion_classifier"),
    "train_maxpooling_model": (".maxpooling", "train_maxpooling_model"),
    "train": (".workflows.train", "train"),
    "train_sequence_classifier": (".transformer", "train_sequence_classifier"),
    "validate_predefined_split": (".data_split", "validate_predefined_split"),
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


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
