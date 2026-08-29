"""Select uncertain pairs, label them with an LLM and persist one dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl
from loguru import logger

from ..config import AppConfig, DatasetLabelingSettings
from ..labeling import (
    LlmLabelingResult,
    label_pairs_with_llm,
    labeling_fingerprint,
)
from ._common import workflow_logging


ANNOTATION_KEY_COLUMNS = (
    "pair_left",
    "pair_right",
    "labeling_fingerprint",
)
ANNOTATION_COLUMNS = (
    "id1",
    "id2",
    "pair_left",
    "pair_right",
    "source_score",
    "category1",
    "category2",
    "llm_score",
    "reason",
    "llm_model",
    "labeling_fingerprint",
    "labeled_at",
)


@dataclass(frozen=True, slots=True)
class SelectedLabelingSample:
    pairs: pl.DataFrame
    source_rows: int
    interval_rows: int
    interval_unique_pairs: int
    already_labeled_rows: int
    remaining_rows: int


@dataclass(frozen=True, slots=True)
class DatasetLabelingResult:
    output_path: Path
    labeling_fingerprint: str
    source_rows: int
    interval_rows: int
    interval_unique_pairs: int
    already_labeled_rows: int
    remaining_rows: int
    selected_rows: int
    successful_rows: int
    failed_rows: int
    output_rows: int
    rounds_completed: int


def _required_columns(path: Path, required: set[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    columns = set(pl.scan_parquet(path).collect_schema().names())
    missing = required - columns
    if missing:
        raise ValueError(
            f"Parquet file {path} is missing columns: {', '.join(sorted(missing))}"
        )


def _load_existing_annotations(path: Path) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    frame = pl.read_parquet(path)
    missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing annotation file {path} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    return frame.select(ANNOTATION_COLUMNS)


def select_unlabeled_pairs(
    settings: DatasetLabelingSettings,
    *,
    fingerprint: str,
    existing_annotations: pl.DataFrame | None,
) -> SelectedLabelingSample:
    """Select a deterministic sample after an exact canonical-pair anti-join."""
    _required_columns(
        settings.source_matches_path,
        {"id1", "id2", settings.score_column},
    )
    source = pl.scan_parquet(settings.source_matches_path).select(
        "id1",
        "id2",
        pl.col(settings.score_column).cast(pl.Float64).alias("source_score"),
    )
    in_interval = pl.col("source_score").is_between(
        settings.lower_p,
        settings.upper_p,
        closed="both",
    )
    statistics = source.select(
        pl.len().alias("source_rows"),
        in_interval.sum().alias("interval_rows"),
        (
            (pl.col("id1").is_null() | pl.col("id2").is_null())
            & in_interval
        )
        .sum()
        .alias("null_id_rows"),
    ).collect(engine="streaming").row(0, named=True)
    if statistics["null_id_rows"]:
        raise ValueError(
            "Source matches contain null id1/id2 values inside the selected interval"
        )

    candidates = (
        source.filter(in_interval)
        .with_columns(
            pl.when(pl.col("id1") <= pl.col("id2"))
            .then(pl.col("id1"))
            .otherwise(pl.col("id2"))
            .alias("pair_left"),
            pl.when(pl.col("id1") <= pl.col("id2"))
            .then(pl.col("id2"))
            .otherwise(pl.col("id1"))
            .alias("pair_right"),
        )
        .unique(subset=["pair_left", "pair_right"], maintain_order=False)
        .with_columns(
            pl.col("pair_left").alias("id1"),
            pl.col("pair_right").alias("id2"),
        )
    )
    interval_unique_pairs = int(
        candidates.select(pl.len()).collect(engine="streaming").item()
    )

    remaining = candidates
    if existing_annotations is not None and not existing_annotations.is_empty():
        completed_keys = (
            existing_annotations.filter(
                pl.col("labeling_fingerprint") == fingerprint
            )
            .select("pair_left", "pair_right")
            .unique()
        )
        if not completed_keys.is_empty():
            remaining = remaining.join(
                completed_keys.lazy(),
                on=["pair_left", "pair_right"],
                how="anti",
            )

    remaining_rows = int(
        remaining.select(pl.len()).collect(engine="streaming").item()
    )
    selected = (
        remaining.with_columns(
            pl.struct("pair_left", "pair_right")
            .hash(seed=settings.seed)
            .alias("_sample_order")
        )
        .sort("_sample_order", "pair_left", "pair_right")
        .head(settings.sample_size)
        .drop("_sample_order")
        .collect(engine="streaming")
    )
    return SelectedLabelingSample(
        pairs=selected,
        source_rows=int(statistics["source_rows"]),
        interval_rows=int(statistics["interval_rows"]),
        interval_unique_pairs=interval_unique_pairs,
        already_labeled_rows=interval_unique_pairs - remaining_rows,
        remaining_rows=remaining_rows,
    )


def _attach_card_fields(
    selected: pl.DataFrame,
    items_path: Path,
) -> pl.DataFrame:
    _required_columns(
        items_path,
        {"id", "name", "category", "attributes"},
    )
    needed_ids = pl.concat(
        [
            selected.select(pl.col("id1").alias("id")),
            selected.select(pl.col("id2").alias("id")),
        ]
    ).unique()
    cards = (
        pl.scan_parquet(items_path)
        .select("id", "name", "category", "attributes")
        .join(needed_ids.lazy(), on="id", how="semi")
        .collect(engine="streaming")
    )
    left_cards = cards.select(
        pl.col("id").alias("id1"),
        pl.col("name").fill_null("").alias("name1"),
        pl.col("category").fill_null("").alias("category1"),
        pl.col("attributes").fill_null("{}").alias("attributes1"),
        pl.lit(True).alias("_left_found"),
    )
    right_cards = cards.select(
        pl.col("id").alias("id2"),
        pl.col("name").fill_null("").alias("name2"),
        pl.col("category").fill_null("").alias("category2"),
        pl.col("attributes").fill_null("{}").alias("attributes2"),
        pl.lit(True).alias("_right_found"),
    )
    pairs = (
        selected.join(left_cards, on="id1", how="left", validate="m:1")
        .join(right_cards, on="id2", how="left", validate="m:1")
    )
    missing_cards = pairs.filter(
        pl.col("_left_found").is_null() | pl.col("_right_found").is_null()
    ).height
    if missing_cards:
        raise ValueError(
            f"Selected matches reference {missing_cards} pairs with missing cards"
        )
    return pairs.drop("_left_found", "_right_found").with_row_index(
        "_label_row_id"
    )


def _write_annotations(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary_path)
    temporary_path.replace(path)


def label_dataset(
    config: AppConfig,
    *,
    labeler: Callable[..., LlmLabelingResult] = label_pairs_with_llm,
) -> DatasetLabelingResult:
    """Run deterministic, resumable LLM labeling for one configured sample."""
    settings = config.labeling
    fingerprint = labeling_fingerprint(settings.llm)
    started_at = perf_counter()
    with workflow_logging(config, workflow_name="label"):
        existing_annotations = _load_existing_annotations(settings.output_path)
        existing_rows = (
            0 if existing_annotations is None else existing_annotations.height
        )
        selection = select_unlabeled_pairs(
            settings,
            fingerprint=fingerprint,
            existing_annotations=existing_annotations,
        )
        logger.info(
            "LLM labeling selection: source_rows={}, interval=[{}, {}], "
            "interval_rows={}, interval_unique_pairs={}, "
            "already_labeled_rows={}, remaining_rows={}, requested_sample_size={}, "
            "selected_rows={}, fingerprint={}",
            selection.source_rows,
            settings.lower_p,
            settings.upper_p,
            selection.interval_rows,
            selection.interval_unique_pairs,
            selection.already_labeled_rows,
            selection.remaining_rows,
            settings.sample_size,
            selection.pairs.height,
            fingerprint,
        )

        if selection.pairs.is_empty():
            logger.info("No unlabeled pairs remain for the configured selection")
            return DatasetLabelingResult(
                output_path=settings.output_path,
                labeling_fingerprint=fingerprint,
                source_rows=selection.source_rows,
                interval_rows=selection.interval_rows,
                interval_unique_pairs=selection.interval_unique_pairs,
                already_labeled_rows=selection.already_labeled_rows,
                remaining_rows=selection.remaining_rows,
                selected_rows=0,
                successful_rows=0,
                failed_rows=0,
                output_rows=existing_rows,
                rounds_completed=0,
            )

        pairs = _attach_card_fields(selection.pairs, settings.items_path)
        accumulated = existing_annotations

        def checkpoint(new_annotations: pl.DataFrame) -> None:
            nonlocal accumulated
            accumulated = (
                new_annotations
                if accumulated is None or accumulated.is_empty()
                else pl.concat(
                    [accumulated, new_annotations],
                    how="vertical_relaxed",
                )
            )
            accumulated = (
                accumulated.unique(
                    subset=ANNOTATION_KEY_COLUMNS,
                    keep="last",
                    maintain_order=True,
                )
                .select(ANNOTATION_COLUMNS)
                .sort("labeling_fingerprint", "pair_left", "pair_right")
            )
            _write_annotations(accumulated, settings.output_path)
            logger.info(
                "Saved LLM labeling checkpoint: new_rows={}, total_rows={}, path={!s}",
                new_annotations.height,
                accumulated.height,
                settings.output_path,
            )

        llm_result = labeler(
            pairs,
            settings.llm,
            checkpoint_every_batches=settings.checkpoint_every_batches,
            checkpoint=checkpoint,
        )
        if llm_result.error_counts:
            logger.warning(
                "Top LLM labeling errors: {}",
                dict(llm_result.error_counts),
            )
        output_rows = 0 if accumulated is None else accumulated.height
        logger.info(
            "Finished LLM labeling: selected_rows={}, successful_rows={}, "
            "failed_rows={}, rounds_completed={}, output_rows_before={}, "
            "output_rows_after={}, elapsed_seconds={:.3f}",
            pairs.height,
            llm_result.successful_rows,
            llm_result.failed_rows,
            llm_result.rounds_completed,
            existing_rows,
            output_rows,
            perf_counter() - started_at,
        )
        return DatasetLabelingResult(
            output_path=settings.output_path,
            labeling_fingerprint=fingerprint,
            source_rows=selection.source_rows,
            interval_rows=selection.interval_rows,
            interval_unique_pairs=selection.interval_unique_pairs,
            already_labeled_rows=selection.already_labeled_rows,
            remaining_rows=selection.remaining_rows,
            selected_rows=pairs.height,
            successful_rows=llm_result.successful_rows,
            failed_rows=llm_result.failed_rows,
            output_rows=output_rows,
            rounds_completed=llm_result.rounds_completed,
        )


__all__ = [
    "DatasetLabelingResult",
    "SelectedLabelingSample",
    "label_dataset",
    "select_unlabeled_pairs",
]
