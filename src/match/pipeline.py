"""Configuration-driven orchestration for Twin2Attr experiments.

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
import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from .data_split import DataSplitConfig, split_matches, validate_predefined_split
from .fusion import (
    FusionConfig,
    load_fusion_classifier,
    predict_fusion_probabilities,
    train_fusion_classifier,
)
from .maxpooling import MaxPoolingModel, encode_attribute_pairs, train_maxpooling_model
from .transformer import (
    SequenceClassifierConfig,
    TrainingResult,
    encode_pair_cls,
    load_trained_classifier,
    predict_match_probabilities,
    train_sequence_classifier,
)
from .normalization import normalize_attributes
from .pair_encoding import add_pair_special_tokens, infer_pair_max_length
from .paths import resolve_project_path
from .prepare_data import PreparedPair, prepare_pairs

__all__ = [
    "inference_pipeline",
    "inspect_max_length",
    "run_pipeline",
    "train_pipeline",
]


Config = DictConfig | Mapping[str, Any]


@dataclass(slots=True)
class PipelineContext:
    """Mutable artifacts passed between explicitly ordered pipeline stages."""

    action: str
    config: DictConfig
    stages: tuple[str, ...]
    items_path: Path | None = None
    items: pl.DataFrame | None = None
    attributes_column: str = "attributes"
    train_matches: pl.DataFrame | None = None
    validation_matches: pl.DataFrame | None = None
    train_matches_path: Path | None = None
    validation_matches_path: Path | None = None
    inference_matches: pl.DataFrame | None = None
    inference_matches_path: Path | None = None
    train_pairs: list[PreparedPair] | None = None
    validation_pairs: list[PreparedPair] | None = None
    inference_pairs: list[PreparedPair] | None = None
    resolved_max_length: int | None = None
    tokenizer: Any = None
    transformer_model: Any = None
    transformer_result: TrainingResult | None = None
    maxpooling_model: MaxPoolingModel | None = None
    cls_embeddings: np.ndarray | None = None
    maxpooling_embeddings: np.ndarray | None = None
    predictions: pl.DataFrame | None = None


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
) -> tuple[pl.DataFrame, pl.DataFrame, Path, Path]:
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
        return (
            train_matches,
            validation_matches,
            train_source_path,
            validation_source_path,
        )

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
    return (
        split_result.train_matches,
        split_result.validation_matches,
        generated_train_path,
        generated_validation_path,
    )


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


@contextmanager
def _items_parquet(
    items: pl.DataFrame,
    original_path: Path,
    *,
    attributes_column: str,
    temporary_path: Path,
) -> Iterator[Path]:
    """Expose the selected attributes as ``attributes`` for max-pooling APIs."""
    if attributes_column == "attributes":
        yield original_path
        return

    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    items.with_columns(pl.col(attributes_column).alias("attributes")).write_parquet(temporary_path)
    logger.info("Wrote normalized max-pooling input to {!s}", temporary_path)
    try:
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _train_maxpooling(
    config: DictConfig,
    items: pl.DataFrame,
    items_path: Path,
    train_matches_path: Path,
    attributes_column: str,
) -> MaxPoolingModel | None:
    settings = config.features.maxpooling
    model_path = _path(settings.model_path, "features.maxpooling.model_path")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = model_path.parent / ".maxpooling_items.parquet"
    with _items_parquet(
        items,
        items_path,
        attributes_column=attributes_column,
        temporary_path=staging_path,
    ) as maxpooling_items_path:
        model = train_maxpooling_model(
            maxpooling_items_path,
            train_matches_path,
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


def _encode_maxpooling_vectors(
    items: pl.DataFrame,
    items_path: Path,
    attributes_column: str,
    matches_path: Path,
    model: MaxPoolingModel,
    *,
    temporary_path: Path,
) -> np.ndarray:
    with _items_parquet(
        items,
        items_path,
        attributes_column=attributes_column,
        temporary_path=temporary_path,
    ) as maxpooling_items_path:
        return encode_attribute_pairs(maxpooling_items_path, matches_path, model)


def _train_fusion(
    config: DictConfig,
    items: pl.DataFrame,
    items_path: Path,
    attributes_column: str,
    train_matches_path: Path,
    validation_matches_path: Path,
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
    temporary_items_path = fusion_path.parent / ".fusion_items.parquet"
    train_maxpooling = _encode_maxpooling_vectors(
        items,
        items_path,
        attributes_column,
        train_matches_path,
        maxpooling_model,
        temporary_path=temporary_items_path,
    )
    validation_maxpooling = _encode_maxpooling_vectors(
        items,
        items_path,
        attributes_column,
        validation_matches_path,
        maxpooling_model,
        temporary_path=temporary_items_path,
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


def _require(context: PipelineContext, attribute: str, stage: str) -> Any:
    value = getattr(context, attribute)
    if value is None:
        raise RuntimeError(f"stage {stage!r} requires context.{attribute}")
    return value


def _validate_stage_sequence(action: str, stages: tuple[str, ...]) -> None:
    known = {
        "read_data",
        "normalize",
        "split_data",
        "prepare_data",
        "pair_encoding",
        "transformer",
        "maxpooling",
        "fusion",
        "save_predictions",
    }
    unknown = set(stages) - known
    if unknown:
        raise ValueError(f"unknown pipeline stages: {sorted(unknown)}")
    if len(stages) != len(set(stages)):
        raise ValueError("pipeline stages must not contain duplicates")

    positions = {stage: index for index, stage in enumerate(stages)}

    def require_before(stage: str, dependency: str) -> None:
        if stage in positions and (
            dependency not in positions or positions[dependency] > positions[stage]
        ):
            raise ValueError(f"stage {stage!r} requires earlier stage {dependency!r}")

    for stage in stages:
        if stage != "read_data":
            require_before(stage, "read_data")
    if action == "train":
        for stage in ("prepare_data", "maxpooling"):
            require_before(stage, "split_data")
    require_before("pair_encoding", "prepare_data")
    require_before("transformer", "pair_encoding")
    if "normalize" in positions:
        for stage in ("prepare_data", "maxpooling"):
            if stage in positions and positions["normalize"] > positions[stage]:
                raise ValueError(f"stage {stage!r} must run after 'normalize'")
    if "fusion" in positions:
        require_before("fusion", "transformer")
        require_before("fusion", "maxpooling")
    if action == "inference":
        require_before("save_predictions", "transformer")


def _stage_read_data(context: PipelineContext) -> None:
    context.items_path = _path(context.config.paths.items, "paths.items")
    context.items = _read_parquet(context.items_path, label="items")
    context.attributes_column = str(context.config.normalization.source_column)
    if context.action in {"inference", "inspect"}:
        configured_path = (
            context.config.paths.inference_matches
            if context.action == "inference"
            else context.config.paths.get("inspect_matches")
            or context.config.paths.train_matches
        )
        context.inference_matches_path = _path(configured_path, "inspection matches")
        context.inference_matches = _read_parquet(
            context.inference_matches_path,
            label=f"{context.action} matches",
        )


def _stage_normalize(context: PipelineContext) -> None:
    if any(
        pairs is not None
        for pairs in (
            context.train_pairs,
            context.validation_pairs,
            context.inference_pairs,
        )
    ):
        raise RuntimeError("normalize must run before prepare_data")
    items = _require(context, "items", "normalize")
    context.items, context.attributes_column = _prepare_items(items, context.config)


def _stage_split_data(context: PipelineContext) -> None:
    if context.action != "train":
        raise ValueError("split_data is only valid in the training pipeline")
    items = _require(context, "items", "split_data")
    (
        context.train_matches,
        context.validation_matches,
        context.train_matches_path,
        context.validation_matches_path,
    ) = _load_training_matches(items, context.config)


def _stage_prepare_data(context: PipelineContext) -> None:
    items = _require(context, "items", "prepare_data")
    if context.action == "train":
        train_matches = _require(context, "train_matches", "prepare_data")
        validation_matches = _require(context, "validation_matches", "prepare_data")
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
            context.attributes_column,
            split_name="training and validation",
        )
        context.train_pairs = pairs[:train_size]
        context.validation_pairs = pairs[train_size:]
        return

    matches = _require(context, "inference_matches", "prepare_data")
    context.inference_pairs = _prepare_pair_rows(
        items,
        matches,
        context.attributes_column,
        split_name="inference",
    )


def _stage_pair_encoding(context: PipelineContext) -> None:
    if context.action == "inference":
        _require(context, "inference_pairs", "pair_encoding")
        context.tokenizer, context.transformer_model = load_trained_classifier(
            _path(context.config.paths.model_dir, "paths.model_dir"),
            device=context.config.runtime.device,
        )
        model_config = getattr(context.transformer_model, "config", None)
        context.resolved_max_length = getattr(
            model_config,
            "match_max_length",
            None,
        )
        return

    pairs_attribute = "train_pairs" if context.action == "train" else "inference_pairs"
    train_pairs = _require(context, pairs_attribute, "pair_encoding")
    encoding = context.config.pair_encoding
    if encoding.max_length is not None:
        context.resolved_max_length = int(encoding.max_length)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(context.config.model.pretrained_model_path))
    if encoding.use_field_tokens:
        add_pair_special_tokens(tokenizer)
    context.resolved_max_length = infer_pair_max_length(
        tokenizer,
        train_pairs,
        quantile=float(encoding.quantile),
        sample_size=int(encoding.sample_size),
        hard_cap=int(encoding.hard_cap),
        use_field_tokens=bool(encoding.use_field_tokens),
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
    )
    logger.info("Pair encoding resolved max_length={}", context.resolved_max_length)


def _stage_transformer(context: PipelineContext) -> None:
    if context.action == "train":
        train_pairs = _require(context, "train_pairs", "transformer")
        validation_pairs = _require(context, "validation_pairs", "transformer")
        max_length = _require(context, "resolved_max_length", "transformer")
        model_dir = _path(context.config.paths.model_dir, "paths.model_dir")
        model_dir.mkdir(parents=True, exist_ok=True)
        sequence_config = replace(
            _sequence_config(context.config),
            max_length=int(max_length),
        )
        context.transformer_result = train_sequence_classifier(
            train_pairs,
            validation_pairs,
            sequence_config,
            output_dir=model_dir,
        )
        return

    pairs = _require(context, "inference_pairs", "transformer")
    tokenizer = _require(context, "tokenizer", "transformer")
    model = _require(context, "transformer_model", "transformer")
    matches = _require(context, "inference_matches", "transformer")
    probabilities = predict_match_probabilities(
        model,
        tokenizer,
        pairs,
        batch_size=int(context.config.inference.batch_size),
    )
    context.predictions = matches.with_columns(
        pl.Series(str(context.config.inference.probability_column), probabilities)
    )
    if context.config.inference.save_cls_embeddings or "fusion" in context.stages:
        context.cls_embeddings = encode_pair_cls(
            model,
            tokenizer,
            pairs,
            batch_size=int(context.config.inference.batch_size),
        )
    if context.config.inference.save_cls_embeddings:
        cls_path = _path(context.config.paths.cls_embeddings, "paths.cls_embeddings")
        cls_path.parent.mkdir(parents=True, exist_ok=True)
        with cls_path.open("wb") as output_file:
            np.save(output_file, context.cls_embeddings)


def _stage_maxpooling(context: PipelineContext) -> None:
    items = _require(context, "items", "maxpooling")
    items_path = _require(context, "items_path", "maxpooling")
    settings = context.config.features.maxpooling
    if context.action == "train":
        train_matches_path = _require(
            context,
            "train_matches_path",
            "maxpooling",
        )
        context.maxpooling_model = _train_maxpooling(
            context.config,
            items,
            items_path,
            train_matches_path,
            context.attributes_column,
        )
        return

    matches_path = _require(context, "inference_matches_path", "maxpooling")
    context.maxpooling_model = joblib.load(
        _path(settings.model_path, "features.maxpooling.model_path")
    )
    output_path = _path(
        context.config.paths.maxpooling_embeddings,
        "paths.maxpooling_embeddings",
    )
    context.maxpooling_embeddings = _encode_maxpooling_vectors(
        items,
        items_path,
        context.attributes_column,
        matches_path,
        context.maxpooling_model,
        temporary_path=output_path.parent / ".maxpooling_items.parquet",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        np.save(output_file, context.maxpooling_embeddings)


def _stage_fusion(context: PipelineContext) -> None:
    if context.action == "train":
        _require(context, "transformer_result", "fusion")
        _train_fusion(
            context.config,
            _require(context, "items", "fusion"),
            _require(context, "items_path", "fusion"),
            context.attributes_column,
            _require(context, "train_matches_path", "fusion"),
            _require(context, "validation_matches_path", "fusion"),
            _require(context, "train_pairs", "fusion"),
            _require(context, "validation_pairs", "fusion"),
            _require(context, "maxpooling_model", "fusion"),
        )
        return

    predictions = _require(context, "predictions", "fusion")
    fusion_model = load_fusion_classifier(
        _path(context.config.fusion.model_path, "fusion.model_path"),
        device=context.config.runtime.device,
    )
    probabilities = predict_fusion_probabilities(
        fusion_model,
        _require(context, "cls_embeddings", "fusion"),
        _require(context, "maxpooling_embeddings", "fusion"),
        batch_size=int(context.config.fusion.batch_size),
    )
    context.predictions = predictions.with_columns(
        pl.Series(str(context.config.fusion.probability_column), probabilities)
    )


def _stage_save_predictions(context: PipelineContext) -> None:
    predictions = _require(context, "predictions", "save_predictions")
    output_path = _path(context.config.paths.predictions, "paths.predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_path)
    logger.info("Saved {} predictions to {!s}", predictions.height, output_path)


_STAGES = {
    "read_data": _stage_read_data,
    "normalize": _stage_normalize,
    "split_data": _stage_split_data,
    "prepare_data": _stage_prepare_data,
    "pair_encoding": _stage_pair_encoding,
    "transformer": _stage_transformer,
    "maxpooling": _stage_maxpooling,
    "fusion": _stage_fusion,
    "save_predictions": _stage_save_predictions,
}


def _run_stage_pipeline(config: DictConfig, *, action: str) -> PipelineContext:
    stages = tuple(str(stage) for stage in config.pipelines[action])
    _validate_stage_sequence(action, stages)
    context = PipelineContext(action=action, config=config, stages=stages)
    for index, stage in enumerate(stages, start=1):
        started_at = perf_counter()
        logger.info("Starting stage {}/{}: {}", index, len(stages), stage)
        _STAGES[stage](context)
        logger.info(
            "Finished stage {}/{}: {}, elapsed_seconds={:.3f}",
            index,
            len(stages),
            stage,
            perf_counter() - started_at,
        )
    return context


def train_pipeline(config: Config) -> TrainingResult | None:
    """Execute the ordered training stages declared in the config."""
    cfg = _config(config)
    _check_optional_features(cfg)
    model_dir = _path(cfg.paths.model_dir, "paths.model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, model_dir / "pipeline_config.yaml", resolve=True)
    context = _run_stage_pipeline(cfg, action="train")
    return context.transformer_result


def inference_pipeline(config: Config) -> pl.DataFrame:
    """Execute the ordered inference stages declared in the config."""
    cfg = _config(config)
    _check_optional_features(cfg)
    context = _run_stage_pipeline(cfg, action="inference")
    if context.predictions is not None:
        return context.predictions
    return _require(context, "inference_matches", "inference_pipeline")


def inspect_max_length(config: Config) -> int:
    """Run inspection stages and return the resolved pair length."""
    cfg = _config(config)
    _check_optional_features(cfg)
    context = _run_stage_pipeline(cfg, action="inspect")
    return int(_require(context, "resolved_max_length", "inspect_max_length"))


def run_pipeline(config: Config) -> TrainingResult | pl.DataFrame | int | None:
    """Dispatch the configured ``train``, ``inference`` or ``inspect`` action."""
    cfg = _config(config)
    log_sink: int | None = None
    if cfg.logging.file:
        log_path = _path(cfg.logging.file, "logging.file")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_sink = logger.add(
            log_path,
            level=str(cfg.logging.level),
            rotation=str(cfg.logging.rotation),
            enqueue=True,
        )
    try:
        logger.info(
            "Starting Twin2Attr pipeline: action={}, stages={}, ner={}",
            cfg.action,
            list(cfg.pipelines[cfg.action])
            if cfg.action in {"train", "inference", "inspect"}
            else [],
            cfg.features.ner.enabled,
        )
        if cfg.action == "train":
            return train_pipeline(cfg)
        if cfg.action == "inference":
            return inference_pipeline(cfg)
        if cfg.action == "inspect":
            return inspect_max_length(cfg)
        raise ValueError("action must be one of: train, inference, inspect")
    finally:
        if log_sink is not None:
            logger.remove(log_sink)
