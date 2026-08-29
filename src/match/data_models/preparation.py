"""Target conversion, source weighting and bounded dataset sampling."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from ..config import (
    DatasetOverlapResolutionSettings,
    DatasetSourceSettings,
    DatasetSplitterSettings,
)
from ..sample_weight_models import build_sample_weight_model

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
        return matches.with_columns(
            pl.col("target").cast(pl.Int8),
            pl.col("target").cast(pl.Float32).alias("training_target"),
            pl.lit(1.0).cast(pl.Float32).alias("_annotation_confidence"),
        )

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
        pl.Series("annotation_votes", rounded.astype(np.int16, copy=False)),
        numeric_targets.cast(pl.Float32).alias("training_target"),
        pl.Series(
            "_annotation_confidence",
            np.abs(2.0 * values - 1.0).astype(np.float32, copy=False),
        ),
    )
    if splitter.uncertain_action == "drop":
        selected = with_votes.filter(
            (pl.col("annotation_votes") <= splitter.negative_threshold)
            | (pl.col("annotation_votes") >= splitter.positive_threshold)
        )
        hard_target = (
            pl.col("annotation_votes") >= splitter.positive_threshold
        )
    else:
        selected = with_votes
        hard_target = (
            pl.col("annotation_votes") * 2 >= total_votes
        )
    selected = selected.with_columns(
        hard_target.cast(pl.Int8).alias("target")
    )
    if splitter.target_mode == "hard":
        selected = selected.with_columns(
            pl.col("target").cast(pl.Float32).alias("training_target")
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
    return selected


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
    strategy: str = "category_target_balanced",
    confidence_power: float = 2.0,
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
    ranked = enriched.with_columns(
        pl.struct("id1", "id2", "target")
        .hash(seed=seed)
        .alias("_sample_hash")
    )
    if strategy == "category_target_confidence_weighted":
        uniform = (
            (pl.col("_sample_hash").cast(pl.Float64) + 1.0)
            / 18_446_744_073_709_551_616.0
        )
        weight = (
            pl.col("_annotation_confidence")
            .cast(pl.Float64)
            .pow(confidence_power)
        )
        ranked = ranked.with_columns(
            (-uniform.log() / weight).alias("_sample_score")
        )
    elif strategy == "category_target_confidence_priority":
        ranked = ranked.with_columns(
            (-pl.col("_annotation_confidence")).alias("_sample_score")
        )
    else:
        ranked = ranked.with_columns(
            pl.col("_sample_hash").alias("_sample_score")
        )
    return (
        ranked.sort([*groups, "_sample_score", "_sample_hash"])
        .with_columns(pl.int_range(pl.len()).over(groups).alias("_sample_rank"))
        .join(allocations.select(*groups, "_allocation"), on=groups, how="left")
        .filter(pl.col("_sample_rank") < pl.col("_allocation"))
        .drop(
            "_sample_category",
            "_sample_hash",
            "_sample_score",
            "_sample_rank",
            "_allocation",
        )
    )


def prepare_source_labels(
    matches: pl.DataFrame,
    source: DatasetSourceSettings,
) -> pl.DataFrame:
    """Validate one source and normalize its selected rows to binary labels."""
    _validate_match_columns(matches, source_name=source.name)
    return _binary_targets(
        matches,
        source.splitter,
        source_name=source.name,
    )


def _with_pair_key(matches: pl.DataFrame) -> pl.DataFrame:
    return matches.with_columns(
        pl.min_horizontal("id1", "id2").alias("_overlap_id1"),
        pl.max_horizontal("id1", "id2").alias("_overlap_id2"),
    )


def resolve_source_overlaps(
    sources: dict[str, pl.DataFrame],
    settings: DatasetOverlapResolutionSettings,
) -> dict[str, pl.DataFrame]:
    """Remove lower-priority cross-source copies of symmetric match pairs."""
    if not settings.enabled:
        return sources
    ranks = {name: rank for rank, name in enumerate(settings.source_priority)}
    keyed = {
        name: _with_pair_key(frame)
        for name, frame in sources.items()
    }
    keys = pl.concat(
        [
            frame.select(
                "_overlap_id1",
                "_overlap_id2",
                "target",
            ).with_columns(
                pl.lit(ranks[name]).cast(pl.UInt32).alias("_overlap_rank"),
            )
            for name, frame in keyed.items()
        ],
        how="vertical_relaxed",
    )
    summary = keys.group_by("_overlap_id1", "_overlap_id2").agg(
        pl.col("_overlap_rank").n_unique().alias("_source_count"),
        pl.col("target").n_unique().alias("_target_count"),
        pl.col("_overlap_rank").min().alias("_winning_rank"),
    )
    overlaps = summary.filter(pl.col("_source_count") > 1)
    conflicts = overlaps.filter(pl.col("_target_count") > 1).height
    result: dict[str, pl.DataFrame] = {}
    removed_by_source: dict[str, int] = {}
    for name, frame in keyed.items():
        losing_keys = overlaps.filter(
            pl.col("_winning_rank") < ranks[name]
        ).select("_overlap_id1", "_overlap_id2")
        selected = frame.join(
            losing_keys,
            on=["_overlap_id1", "_overlap_id2"],
            how="anti",
        )
        removed_by_source[name] = frame.height - selected.height
        result[name] = selected.drop("_overlap_id1", "_overlap_id2")
    logger.info(
        "Resolved cross-source overlaps: overlapping_pairs={}, "
        "conflicting_target_pairs={}, removed_rows={}, priority={}",
        overlaps.height,
        conflicts,
        removed_by_source,
        list(settings.source_priority),
    )
    return result


def finalize_source_matches(
    items: pl.DataFrame,
    prepared: pl.DataFrame,
    source: DatasetSourceSettings,
    *,
    seed: int,
) -> pl.DataFrame:
    """Apply source weighting and sampling after overlap resolution."""
    weight_model = build_sample_weight_model(source.weight_model)
    prepared = weight_model.apply(prepared, source_name=source.name)
    confidence = source.confidence_weighting
    if confidence.enabled:
        prepared = prepared.with_columns(
            pl.max_horizontal(
                pl.lit(confidence.min_weight_multiplier),
                pl.col("_annotation_confidence")
                .cast(pl.Float64)
                .pow(confidence.power),
            )
            .cast(pl.Float32)
            .alias("confidence_multiplier")
        )
    else:
        prepared = prepared.with_columns(
            pl.lit(1.0).cast(pl.Float32).alias("confidence_multiplier")
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
                strategy=source.sampling_strategy,
                confidence_power=source.confidence_power,
            )
        logger.info(
            "Sampled dataset source: source={}, strategy={}, rows={}",
            source.name,
            source.sampling_strategy,
            prepared.height,
        )
    diagnostics = [
        name
        for name in prepared.columns
        if name in {
            "annotation_votes",
            "training_target",
            "confidence_multiplier",
            "weight_multiplier",
        }
        or name.startswith("transitivity_")
    ]
    return (
        prepared.select("id1", "id2", "target", *diagnostics)
        .with_columns(
            (
                pl.lit(source.weight).cast(pl.Float32)
                * pl.col("confidence_multiplier")
                * pl.col("weight_multiplier")
            )
            .cast(pl.Float32)
            .alias("sample_weight"),
            pl.lit(source.name).alias("data_source"),
        )
    )


def prepare_source_matches(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    source: DatasetSourceSettings,
    *,
    seed: int,
) -> pl.DataFrame:
    """Normalize, weight and sample one standalone dataset source."""
    prepared = prepare_source_labels(matches, source)
    return finalize_source_matches(
        items,
        prepared,
        source,
        seed=seed,
    )


__all__ = [
    "finalize_source_matches",
    "prepare_source_labels",
    "prepare_source_matches",
    "resolve_source_overlaps",
]
