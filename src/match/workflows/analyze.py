"""Explicit offline analysis workflow."""

from __future__ import annotations

import polars as pl
from loguru import logger

from ..analysis import (
    AttributeImportanceResult,
    analyze_transformer_attribute_importance,
)
from ..augmentations import apply_pair_augmentation
from ..config import AppConfig
from ..data import prepare_configured_items, prepare_pair_rows
from ..data_models import build_data_model
from ..data_postprocessing import apply_pair_postprocessing
from ._common import workflow_logging


def _sample_matches(
    matches: pl.DataFrame,
    *,
    sample_size: int | None,
    seed: int,
) -> pl.DataFrame:
    if sample_size is None or matches.height <= sample_size:
        return matches
    return matches.sample(n=sample_size, shuffle=True, seed=seed)


def _selected_items(
    items: pl.DataFrame,
    matches: pl.DataFrame,
) -> pl.DataFrame:
    ids = pl.concat(
        [
            matches.select(pl.col("id1").alias("id")),
            matches.select(pl.col("id2").alias("id")),
        ]
    ).unique()
    return items.join(ids, on="id", how="semi")


def analyze(config: AppConfig) -> AttributeImportanceResult:
    """Run the configured analysis model without changing model weights."""
    if config.analysis.analysis_model != "attribute_importance":
        raise ValueError(
            f"Unsupported analysis model: {config.analysis.analysis_model!r}"
        )
    settings = config.analysis_models.attribute_importance
    if settings.model != "transformer":
        raise ValueError(
            "attribute_importance currently supports only model='transformer'"
        )
    with workflow_logging(config, workflow_name="analyze"):
        frames = build_data_model(
            config,
            name=config.analysis.data_model,
        ).load_inspection_frames()
        matches = _sample_matches(
            frames.matches,
            sample_size=settings.sample_size,
            seed=config.runtime.seed,
        )
        items = _selected_items(frames.items, matches)
        prepared_items = prepare_configured_items(items, config)
        pairs = prepare_pair_rows(
            prepared_items.frame,
            matches,
            prepared_items.attributes_column,
            split_name="attribute importance analysis",
        )
        augmented = apply_pair_augmentation(
            pairs,
            config,
            model_name=config.analysis.augmentation_model,
        )
        processed = apply_pair_postprocessing(
            augmented.pairs,
            config,
            model_name=config.analysis.data_postprocessing_model,
        )
        logger.info(
            "Attribute analysis sample selected: data_model={}, source_rows={}, "
            "augmented_rows={}",
            config.analysis.data_model,
            len(pairs),
            len(processed),
        )
        return analyze_transformer_attribute_importance(
            config,
            processed,
        )


__all__ = ["analyze"]
