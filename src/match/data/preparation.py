"""Prepare structured model inputs from already loaded data frames."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import polars as pl
from loguru import logger

from ..prepare_data import PreparedPair, prepare_pairs


@dataclass(frozen=True, slots=True)
class TrainingData:
    """Prepared inputs shared by the explicitly called training components."""

    items: pl.DataFrame
    attributes_column: str
    train_matches: pl.DataFrame
    validation_matches: pl.DataFrame
    train_pairs: list[PreparedPair]
    validation_pairs: list[PreparedPair]


def prepare_pair_rows(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    attributes_column: str,
    *,
    split_name: str,
) -> list[PreparedPair]:
    """Build structured pairs and log the resulting split size."""
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


def prepare_training_data(
    items: pl.DataFrame,
    train_matches: pl.DataFrame,
    validation_matches: pl.DataFrame,
    *,
    attributes_column: str,
) -> TrainingData:
    """Create aligned train and validation pairs without reading any files."""
    train_size = train_matches.height
    def selected(frame: pl.DataFrame) -> pl.DataFrame:
        if "sample_weight" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(1.0).cast(pl.Float32).alias("sample_weight")
            )
        return frame.select("id1", "id2", "target", "sample_weight")

    combined_matches = pl.concat(
        [
            selected(train_matches),
            selected(validation_matches),
        ],
        how="vertical_relaxed",
    )
    pairs = prepare_pair_rows(
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
