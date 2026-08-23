"""Target conversion, source weighting and bounded dataset sampling."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from ..config import DatasetSourceSettings, DatasetSplitterSettings

_REQUIRED_MATCH_COLUMNS = {"id1", "id2", "target"}


def _validate_match_columns(matches: pl.DataFrame, *, source_name: str) -> None:
    missing = _REQUIRED_MATCH_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(
            f"dataset source {source_name!r} is missing columns: {sorted(missing)}"
        )
    if matches.height == 0:
        raise ValueError(f"dataset source {source_name!r} must not be empty")
    if any(matches.get_column(name).null_count() for name in _REQUIRED_MATCH_COLUMNS):
        raise ValueError(f"dataset source {source_name!r} contains null match values")


def _binary_targets(
    matches: pl.DataFrame,
    splitter: DatasetSplitterSettings,
    *,
    source_name: str,
) -> pl.DataFrame:
    targets = matches.get_column("target")
    if splitter.score_type == "label":
        if not set(targets.unique().to_list()).issubset({0, 1}):
            raise ValueError(
                f"dataset source {source_name!r} target must contain only 0 and 1"
            )
        return matches.with_columns(pl.col("target").cast(pl.Int8))

    total_votes = splitter.total_votes
    if total_votes is None:
        raise RuntimeError("votes splitter is missing total_votes")
    numeric_targets = targets.cast(pl.Float64)
    values = numeric_targets.to_numpy()
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            f"dataset source {source_name!r} vote scores must be in [0, 1]"
        )
    scaled = values * total_votes
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"dataset source {source_name!r} scores must use steps of "
            f"1/{total_votes}"
        )
    with_votes = matches.with_columns(
        pl.Series("_votes", rounded.astype(np.int16, copy=False))
    )
    selected = with_votes.filter(
        (pl.col("_votes") <= splitter.negative_threshold)
        | (pl.col("_votes") >= splitter.positive_threshold)
    ).with_columns(
        (pl.col("_votes") >= splitter.positive_threshold)
        .cast(pl.Int8)
        .alias("target")
    )
    dropped = matches.height - selected.height
    logger.info(
        "Prepared vote labels: source={}, input_rows={}, selected_rows={}, "
        "dropped_uncertain_rows={}",
        source_name,
        matches.height,
        selected.height,
        dropped,
    )
    if selected.height == 0:
        raise ValueError(
            f"dataset source {source_name!r} thresholds removed every row"
        )
    return selected.drop("_votes")


def _proportional_allocations(
    counts: pl.DataFrame,
    max_rows: int,
) -> pl.DataFrame:
    group_counts = counts.get_column("_group_count").to_numpy()
    raw = group_counts.astype(np.float64) * (max_rows / group_counts.sum())
    allocations = np.floor(raw).astype(np.int64)
    remaining = max_rows - int(allocations.sum())
    if remaining:
        order = np.argsort(-(raw - allocations), kind="stable")
        for index in order:
            if remaining == 0:
                break
            if allocations[index] < group_counts[index]:
                allocations[index] += 1
                remaining -= 1
    return counts.with_columns(pl.Series("_allocation", allocations))


def _category_target_sample(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> pl.DataFrame:
    lookup = items.select(
        pl.col("id").alias("id1"),
        pl.col("category").alias("_sample_category"),
    )
    enriched = matches.join(lookup, on="id1", how="left", validate="m:1")
    if enriched.get_column("_sample_category").null_count():
        raise ValueError("cannot sample matches containing unknown left item ids")
    groups = ["_sample_category", "target"]
    counts = enriched.group_by(groups).len(name="_group_count")
    allocations = _proportional_allocations(counts, max_rows)
    ranked = (
        enriched.with_columns(
            pl.struct("id1", "id2", "target")
            .hash(seed=seed)
            .alias("_sample_hash")
        )
        .sort([*groups, "_sample_hash"])
        .with_columns(pl.int_range(pl.len()).over(groups).alias("_sample_rank"))
        .join(allocations.select(*groups, "_allocation"), on=groups, how="left")
    )
    return (
        ranked.filter(pl.col("_sample_rank") < pl.col("_allocation"))
        .drop("_sample_category", "_sample_hash", "_sample_rank", "_allocation")
    )


def prepare_source_matches(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    source: DatasetSourceSettings,
    *,
    seed: int,
) -> pl.DataFrame:
    """Normalize one source to binary labels and attach its loss weight."""
    _validate_match_columns(matches, source_name=source.name)
    prepared = _binary_targets(
        matches,
        source.splitter,
        source_name=source.name,
    )
    if source.max_rows is not None and prepared.height > source.max_rows:
        if source.sampling_strategy == "random":
            prepared = prepared.sample(
                n=source.max_rows,
                shuffle=True,
                seed=seed,
            )
        else:
            prepared = _category_target_sample(
                items,
                prepared,
                max_rows=source.max_rows,
                seed=seed,
            )
        logger.info(
            "Sampled dataset source: source={}, strategy={}, rows={}",
            source.name,
            source.sampling_strategy,
            prepared.height,
        )
    return prepared.select("id1", "id2", "target").with_columns(
        pl.lit(source.weight).cast(pl.Float32).alias("sample_weight"),
        pl.lit(source.name).alias("data_source"),
    )


__all__ = ["prepare_source_matches"]
