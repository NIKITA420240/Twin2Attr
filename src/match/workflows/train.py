"""Explicit training workflow for Twin2Attr experiments.

This module deliberately contains workflow code only. Attribute normalization,
card preparation, pair encoding and model optimization remain owned by their
specialized modules.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import joblib
import polars as pl
from loguru import logger

from ..config import AppConfig, save_app_config
from ..data import (
    TrainingData,
    TrainingMatchPaths,
    load_training_matches,
    prepare_training_data,
    read_parquet,
)
from ..data_split import DataSplitConfig
from ..fusion import (
    FusionConfig,
    train_fusion_classifier,
)
from ..maxpooling import MaxPoolingModel, encode_attribute_pairs, train_maxpooling_model
from ..transformer import (
    SequenceClassifierConfig,
    TrainingResult,
    encode_pair_cls,
    load_trained_classifier,
    train_sequence_classifier,
)
from ..prepare_data import PreparedPair
from ._common import (
    check_optional_features,
    normalization_enabled,
    prepare_items,
    resolve_max_length,
    workflow_logging,
)

__all__ = [
    "train",
]


def _sequence_config(config: AppConfig) -> SequenceClassifierConfig:
    model = config.model
    encoding = config.pair_encoding
    return SequenceClassifierConfig(
        model_path=model.pretrained_model_path,
        max_epochs=model.max_epochs,
        hpo_trials=model.hpo_trials,
        seed=model.seed,
        use_field_tokens=encoding.use_field_tokens,
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        max_length=encoding.max_length,
        max_length_quantile=encoding.quantile,
        max_length_sample_size=encoding.sample_size,
        max_length_hard_cap=encoding.hard_cap,
    )


def _fusion_config(config: AppConfig) -> FusionConfig:
    settings = config.fusion
    return FusionConfig(
        hidden_dim=settings.hidden_dim,
        dropout=settings.dropout,
        batch_size=settings.batch_size,
        max_epochs=settings.max_epochs,
        patience=settings.patience,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        seed=config.model.seed,
    )


def _train_maxpooling(
    config: AppConfig,
    items: pl.DataFrame,
    train_matches: pl.DataFrame,
    attributes_column: str,
) -> MaxPoolingModel:
    settings = config.features.maxpooling
    model_path = settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = train_maxpooling_model(
        items,
        train_matches,
        attributes_column=attributes_column,
        vector_size=settings.vector_size,
        window=settings.window,
        min_count=settings.min_count,
        workers=settings.workers,
        fasttext_epochs=settings.fasttext_epochs,
        classifier_epochs=settings.classifier_epochs,
        batch_size=settings.batch_size,
        validation_fraction=settings.validation_fraction,
        patience=settings.patience,
        dropout=settings.dropout,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        random_state=config.model.seed,
        device=config.runtime.device,
    )
    joblib.dump(model, model_path)
    logger.info("Saved max-pooling model to {!s}", model_path)
    return model


def _train_fusion(
    config: AppConfig,
    items: pl.DataFrame,
    attributes_column: str,
    train_matches: pl.DataFrame,
    validation_matches: pl.DataFrame,
    train_pairs: list[PreparedPair],
    validation_pairs: list[PreparedPair],
    maxpooling_model: MaxPoolingModel | None,
) -> None:
    if maxpooling_model is None:
        raise RuntimeError("fusion training requires a trained max-pooling model")

    started_at = perf_counter()
    tokenizer, transformer = load_trained_classifier(
        config.paths.model_dir,
        device=config.runtime.device,
    )
    embedding_batch_size = config.fusion.embedding_batch_size
    train_cls = encode_pair_cls(
        transformer,
        tokenizer,
        train_pairs,
        batch_size=embedding_batch_size,
    )
    validation_cls = encode_pair_cls(
        transformer,
        tokenizer,
        validation_pairs,
        batch_size=embedding_batch_size,
    )

    fusion_path = config.fusion.model_path
    train_maxpooling = encode_attribute_pairs(
        items,
        train_matches,
        maxpooling_model,
        attributes_column=attributes_column,
    )
    validation_maxpooling = encode_attribute_pairs(
        items,
        validation_matches,
        maxpooling_model,
        attributes_column=attributes_column,
    )
    fusion_result = train_fusion_classifier(
        train_cls,
        train_maxpooling,
        [int(pair.label) for pair in train_pairs],
        validation_cls,
        validation_maxpooling,
        [int(pair.label) for pair in validation_pairs],
        [pair.category for pair in validation_pairs],
        _fusion_config(config),
        output_path=fusion_path,
        device=config.runtime.device,
    )
    logger.info(
        "Fusion training finished: best_val_macro_pr_auc={:.6f}, "
        "elapsed_seconds={:.3f}",
        fusion_result.best_validation_macro_pr_auc,
        perf_counter() - started_at,
    )


def _train_transformer(
    config: AppConfig,
    data: TrainingData,
    max_length: int,
) -> TrainingResult:
    model_dir = config.paths.model_dir
    sequence_config = replace(
        _sequence_config(config),
        max_length=max_length,
    )
    return train_sequence_classifier(
        data.train_pairs,
        data.validation_pairs,
        sequence_config,
        output_dir=model_dir,
    )


def _training_match_paths(config: AppConfig) -> TrainingMatchPaths:
    split_settings = config.split
    return TrainingMatchPaths(
        source=config.paths.train_matches,
        validation=config.paths.validation_matches,
        generated_train=split_settings.train_output_path,
        generated_validation=split_settings.validation_output_path,
    )


def _data_split_config(config: AppConfig) -> DataSplitConfig:
    settings = config.split
    return DataSplitConfig(
        validation_fraction=settings.validation_fraction,
        leakage_scope=settings.leakage_scope,
        seed=settings.seed,
        candidate_splits=settings.candidate_splits,
    )


def train(config: AppConfig) -> TrainingResult:
    """Run the explicit training workflow and return the Transformer result."""
    with workflow_logging(config, workflow_name="train"):
        check_optional_features(config)
        maxpooling_enabled = config.features.maxpooling.enabled
        fusion_enabled = config.fusion.enabled
        if fusion_enabled and not maxpooling_enabled:
            raise ValueError(
                "fusion.enabled=true requires features.maxpooling.enabled=true"
            )

        model_dir = config.paths.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        save_app_config(config, model_dir / "pipeline_config.yaml")

        items = read_parquet(
            config.paths.items,
            label="items",
        )
        attributes_column = config.normalization.source_column
        if normalization_enabled(config):
            items, attributes_column = prepare_items(items, config)

        train_matches, validation_matches = load_training_matches(
            items,
            _training_match_paths(config),
            _data_split_config(config),
            mode=config.split.mode,
        )
        data = prepare_training_data(
            items,
            train_matches,
            validation_matches,
            attributes_column=attributes_column,
        )
        max_length = resolve_max_length(config, data.train_pairs)
        transformer_result = _train_transformer(config, data, max_length)

        maxpooling_model: MaxPoolingModel | None = None
        if maxpooling_enabled:
            maxpooling_model = _train_maxpooling(
                config,
                data.items,
                data.train_matches,
                data.attributes_column,
            )

        if fusion_enabled:
            _train_fusion(
                config,
                data.items,
                data.attributes_column,
                data.train_matches,
                data.validation_matches,
                data.train_pairs,
                data.validation_pairs,
                maxpooling_model,
            )

        return transformer_result
