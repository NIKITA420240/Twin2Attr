"""Parquet loading and train/validation split persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl
from loguru import logger

from ..data_split import DataSplitConfig, split_matches, validate_predefined_split


@dataclass(frozen=True, slots=True)
class TrainingMatchPaths:
    """Input and generated-output paths used to obtain labeled splits."""

    source: Path
    validation: Path
    generated_train: Path
    generated_validation: Path


def read_parquet(path: Path, *, label: str) -> pl.DataFrame:
    """Read a parquet file and report its basic shape and loading time."""
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


def load_training_matches(
    items: pl.DataFrame,
    paths: TrainingMatchPaths,
    split_config: DataSplitConfig,
    *,
    mode: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load predefined splits or create, persist and return an automatic split."""
    if mode == "predefined":
        train_matches = read_parquet(paths.source, label="training matches")
        validation_matches = read_parquet(
            paths.validation,
            label="validation matches",
        )
        validate_predefined_split(
            items,
            train_matches,
            validation_matches,
            leakage_scope=split_config.leakage_scope,
        )
        return train_matches, validation_matches

    if mode != "auto":
        raise ValueError("split mode must be one of: predefined, auto")

    all_matches = read_parquet(paths.source, label="all labeled matches")
    split_result = split_matches(items, all_matches, split_config)
    paths.generated_train.parent.mkdir(parents=True, exist_ok=True)
    paths.generated_validation.parent.mkdir(parents=True, exist_ok=True)
    split_result.train_matches.write_parquet(paths.generated_train)
    split_result.validation_matches.write_parquet(paths.generated_validation)
    logger.info(
        "Saved generated split: train_path={!s}, validation_path={!s}",
        paths.generated_train,
        paths.generated_validation,
    )
    return split_result.train_matches, split_result.validation_matches
