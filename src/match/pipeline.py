"""Explicit training and inspection workflows for Twin2Attr experiments.

This module deliberately contains workflow code only. Attribute normalization,
card preparation, pair encoding and model optimization remain owned by their
specialized modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import joblib
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from .data_split import DataSplitConfig, split_matches, validate_predefined_split
from .fusion import (
    FusionConfig,
    train_fusion_classifier,
)
from .maxpooling import MaxPoolingModel, encode_attribute_pairs, train_maxpooling_model
from .transformer import (
    SequenceClassifierConfig,
    TrainingResult,
    encode_pair_cls,
    load_trained_classifier,
    train_sequence_classifier,
)
from .normalization import normalize_attributes
from .pair_encoding import add_pair_special_tokens, infer_pair_max_length
from .paths import resolve_project_path
from .prepare_data import PreparedPair, prepare_pairs

__all__ = [
    "inspect_max_length",
    "train_pipeline",
]


Config = DictConfig | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingData:
    """Prepared inputs shared by the explicitly called training components."""

    items: pl.DataFrame
    attributes_column: str
    train_matches: pl.DataFrame
    validation_matches: pl.DataFrame
    train_pairs: list[PreparedPair]
    validation_pairs: list[PreparedPair]


def _config(config: Config) -> DictConfig:
    if isinstance(config, DictConfig):
        return config
    return OmegaConf.create(config)


def _path(value: Any, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"config value {name!r} must contain a path")
    return resolve_project_path(str(value))


def _read_parquet(path: Path, *, label: str) -> pl.DataFrame:
    started_at = perf_counter()
    frame = pl.read_parquet(path)
    logger.info(
        "Loaded {}: path={!s}, rows={}, columns={}, elapsed_seconds={:.3f}",
        label,
        path,
        frame.height,
        len(frame.columns),
        perf_counter() - started_at,
    )
    return frame


def _prepare_items(items: pl.DataFrame, config: DictConfig) -> tuple[pl.DataFrame, str]:
    normalization = config.normalization
    source_column = str(normalization.source_column)
    logger.info("Running attribute normalization")
    normalized = normalize_attributes(
        items,
        _path(normalization.synonyms_path, "normalization.synonyms_path"),
        _path(normalization.unique_attributes_path, "normalization.unique_attributes_path"),
        source_column=source_column,
        output_column=str(normalization.output_column),
        n_jobs=int(normalization.n_jobs),
        chunk_size=int(normalization.chunk_size),
    )
    return normalized, str(normalization.output_column)


def _check_optional_features(config: DictConfig) -> None:
    if config.features.ner.enabled:
        raise NotImplementedError(
            "features.ner.enabled=true, but no NER card transformer has been "
            "implemented yet; keep it false until an NER provider is added"
        )


def _prepare_pair_rows(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    attributes_column: str,
    *,
    split_name: str,
) -> list[PreparedPair]:
    started_at = perf_counter()
    pairs = prepare_pairs(items, matches, attributes_column=attributes_column)
    logger.info(
        "Prepared {} pairs: rows={}, attributes_column={!r}, elapsed_seconds={:.3f}",
        split_name,
        len(pairs),
        attributes_column,
        perf_counter() - started_at,
    )
    return pairs


def _load_training_matches(
    items: pl.DataFrame,
    config: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load predefined splits or generate and persist an automatic split."""
    settings = config.split
    train_source_path = _path(config.paths.train_matches, "paths.train_matches")
    if settings.mode == "predefined":
        train_matches = _read_parquet(train_source_path, label="training matches")
        validation_source_path = _path(config.paths.validation_matches, "paths.validation_matches")
        validation_matches = _read_parquet(validation_source_path, label="validation matches")
        validate_predefined_split(
            items,
            train_matches,
            validation_matches,
            leakage_scope=str(settings.leakage_scope),
        )
        return train_matches, validation_matches

    if settings.mode != "auto":
        raise ValueError("split.mode must be one of: predefined, auto")
    all_matches = _read_parquet(train_source_path, label="all labeled matches")
    split_result = split_matches(
        items,
        all_matches,
        DataSplitConfig(
            validation_fraction=float(settings.validation_fraction),
            leakage_scope=str(settings.leakage_scope),
            seed=int(settings.seed),
            candidate_splits=int(settings.candidate_splits),
        ),
    )
    generated_train_path = _path(
        settings.train_output_path,
        "split.train_output_path",
    )
    generated_validation_path = _path(
        settings.validation_output_path,
        "split.validation_output_path",
    )
    generated_train_path.parent.mkdir(parents=True, exist_ok=True)
    generated_validation_path.parent.mkdir(parents=True, exist_ok=True)
    split_result.train_matches.write_parquet(generated_train_path)
    split_result.validation_matches.write_parquet(generated_validation_path)
    logger.info(
        "Saved generated split: train_path={!s}, validation_path={!s}",
        generated_train_path,
        generated_validation_path,
    )
    return split_result.train_matches, split_result.validation_matches


def _sequence_config(config: DictConfig) -> SequenceClassifierConfig:
    model = config.model
    encoding = config.pair_encoding
    return SequenceClassifierConfig(
        model_path=str(model.pretrained_model_path),
        max_epochs=int(model.max_epochs),
        hpo_trials=int(model.hpo_trials),
        seed=int(model.seed),
        use_field_tokens=bool(encoding.use_field_tokens),
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        max_length=encoding.max_length,
        max_length_quantile=float(encoding.quantile),
        max_length_sample_size=int(encoding.sample_size),
        max_length_hard_cap=int(encoding.hard_cap),
    )


def _fusion_config(config: DictConfig) -> FusionConfig:
    settings = config.fusion
    return FusionConfig(
        hidden_dim=int(settings.hidden_dim),
        dropout=float(settings.dropout),
        batch_size=int(settings.batch_size),
        max_epochs=int(settings.max_epochs),
        patience=int(settings.patience),
        learning_rate=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
        seed=int(config.model.seed),
    )


def _train_maxpooling(
    config: DictConfig,
    items: pl.DataFrame,
    train_matches: pl.DataFrame,
    attributes_column: str,
) -> MaxPoolingModel:
    settings = config.features.maxpooling
    model_path = _path(settings.model_path, "features.maxpooling.model_path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = train_maxpooling_model(
        items,
        train_matches,
        attributes_column=attributes_column,
        vector_size=int(settings.vector_size),
        window=int(settings.window),
        min_count=int(settings.min_count),
        workers=int(settings.workers),
        fasttext_epochs=int(settings.fasttext_epochs),
        classifier_epochs=int(settings.classifier_epochs),
        batch_size=int(settings.batch_size),
        validation_fraction=float(settings.validation_fraction),
        patience=int(settings.patience),
        dropout=float(settings.dropout),
        learning_rate=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
        random_state=int(config.model.seed),
        device=config.runtime.device,
    )
    joblib.dump(model, model_path)
    logger.info("Saved max-pooling model to {!s}", model_path)
    return model


def _train_fusion(
    config: DictConfig,
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
        _path(config.paths.model_dir, "paths.model_dir"),
        device=config.runtime.device,
    )
    embedding_batch_size = int(config.fusion.embedding_batch_size)
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

    fusion_path = _path(config.fusion.model_path, "fusion.model_path")
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


def _normalization_enabled(config: DictConfig) -> bool:
    return bool(config.normalization.get("enabled", False))


def _resolve_max_length(
    config: DictConfig,
    pairs: list[PreparedPair],
) -> int:
    encoding = config.pair_encoding
    if encoding.max_length is not None:
        return int(encoding.max_length)

    tokenizer = AutoTokenizer.from_pretrained(str(config.model.pretrained_model_path))
    if encoding.use_field_tokens:
        add_pair_special_tokens(tokenizer)
    max_length = infer_pair_max_length(
        tokenizer,
        pairs,
        quantile=float(encoding.quantile),
        sample_size=int(encoding.sample_size),
        hard_cap=int(encoding.hard_cap),
        use_field_tokens=bool(encoding.use_field_tokens),
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
    )
    logger.info("Pair encoding resolved max_length={}", max_length)
    return max_length


def _prepare_training_data(config: DictConfig) -> TrainingData:
    items_path = _path(config.paths.items, "paths.items")
    items = _read_parquet(items_path, label="items")
    attributes_column = str(config.normalization.source_column)
    if _normalization_enabled(config):
        items, attributes_column = _prepare_items(items, config)

    train_matches, validation_matches = _load_training_matches(items, config)
    train_size = train_matches.height
    combined_matches = pl.concat(
        [
            train_matches.select("id1", "id2", "target"),
            validation_matches.select("id1", "id2", "target"),
        ],
        how="vertical_relaxed",
    )
    pairs = _prepare_pair_rows(
        items,
        combined_matches,
        attributes_column,
        split_name="training and validation",
    )
    return TrainingData(
        items=items,
        attributes_column=attributes_column,
        train_matches=train_matches,
        validation_matches=validation_matches,
        train_pairs=pairs[:train_size],
        validation_pairs=pairs[train_size:],
    )


def _train_transformer(
    config: DictConfig,
    data: TrainingData,
    max_length: int,
) -> TrainingResult:
    model_dir = _path(config.paths.model_dir, "paths.model_dir")
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


@contextmanager
def _pipeline_logging(config: DictConfig, *, pipeline_name: str) -> Iterator[None]:
    """Configure the optional file sink for one pipeline invocation."""
    log_sink: int | None = None
    if config.logging.file:
        log_path = _path(config.logging.file, "logging.file")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_sink = logger.add(
            log_path,
            level=str(config.logging.level),
            rotation=str(config.logging.rotation),
            enqueue=True,
        )
    try:
        logger.info(
            "Starting Twin2Attr workflow: name={}, normalization={}, "
            "maxpooling={}, fusion={}, ner={}",
            pipeline_name,
            _normalization_enabled(config),
            bool(config.features.maxpooling.get("enabled", False)),
            bool(config.fusion.get("enabled", False)),
            config.features.ner.enabled,
        )
        yield
    finally:
        if log_sink is not None:
            logger.remove(log_sink)


def train_pipeline(config: Config) -> TrainingResult:
    """Run the explicit training workflow and return the Transformer result."""
    cfg = _config(config)
    with _pipeline_logging(cfg, pipeline_name="train"):
        _check_optional_features(cfg)
        maxpooling_enabled = bool(cfg.features.maxpooling.get("enabled", False))
        fusion_enabled = bool(cfg.fusion.get("enabled", False))
        if fusion_enabled and not maxpooling_enabled:
            raise ValueError(
                "fusion.enabled=true requires features.maxpooling.enabled=true"
            )

        model_dir = _path(cfg.paths.model_dir, "paths.model_dir")
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, model_dir / "pipeline_config.yaml", resolve=True)

        data = _prepare_training_data(cfg)
        max_length = _resolve_max_length(cfg, data.train_pairs)
        transformer_result = _train_transformer(cfg, data, max_length)

        maxpooling_model: MaxPoolingModel | None = None
        if maxpooling_enabled:
            maxpooling_model = _train_maxpooling(
                cfg,
                data.items,
                data.train_matches,
                data.attributes_column,
            )

        if fusion_enabled:
            _train_fusion(
                cfg,
                data.items,
                data.attributes_column,
                data.train_matches,
                data.validation_matches,
                data.train_pairs,
                data.validation_pairs,
                maxpooling_model,
            )

        return transformer_result


def inspect_max_length(config: Config) -> int:
    """Prepare inspection pairs and return their recommended encoded length."""
    cfg = _config(config)
    with _pipeline_logging(cfg, pipeline_name="inspect"):
        _check_optional_features(cfg)
        items = _read_parquet(_path(cfg.paths.items, "paths.items"), label="items")
        attributes_column = str(cfg.normalization.source_column)
        if _normalization_enabled(cfg):
            items, attributes_column = _prepare_items(items, cfg)

        configured_path = cfg.paths.get("inspect_matches") or cfg.paths.train_matches
        inspect_matches = _read_parquet(
            _path(configured_path, "paths.inspect_matches"),
            label="inspection matches",
        )
        inspect_pairs = _prepare_pair_rows(
            items,
            inspect_matches,
            attributes_column,
            split_name="inspection",
        )
        return _resolve_max_length(cfg, inspect_pairs)
