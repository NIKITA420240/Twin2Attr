"""Evaluator-compatible inference over packaged Twin2Attr model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from .data.preprocessing import prepare_manifest_items
from .models.contracts import PredictionBatch
from .models.factory import build_predictor
from .paths import PROJECT_ROOT


PAIR_COLUMNS = ("id1", "id2")
ITEM_COLUMNS = ("id", "name", "attributes", "category")


def _load_solution(solution_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    path = Path(solution_path) if solution_path is not None else PROJECT_ROOT / "solution.json"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Solution manifest does not exist: {path}")
    solution = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(solution, dict):
        raise ValueError("solution.json must contain a JSON object")
    return solution, path.parent


def _read_inputs(
    items_path: str | Path,
    matches_path: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    matches = pl.read_parquet(matches_path)
    missing_pair_columns = set(PAIR_COLUMNS) - set(matches.columns)
    if missing_pair_columns:
        raise ValueError(
            f"matches parquet is missing columns: {sorted(missing_pair_columns)}"
        )
    if matches.select(
        pl.col("id1").is_null().sum() + pl.col("id2").is_null().sum()
    ).item():
        raise ValueError("matches parquet contains null item ids")

    required_ids = pl.concat(
        [
            matches.select(pl.col("id1").alias("id")),
            matches.select(pl.col("id2").alias("id")),
        ]
    ).unique()
    items_scan = pl.scan_parquet(items_path)
    available_columns = set(items_scan.collect_schema().names())
    missing_item_columns = set(ITEM_COLUMNS) - available_columns
    if missing_item_columns:
        raise ValueError(
            f"items parquet is missing columns: {sorted(missing_item_columns)}"
        )
    items = (
        items_scan.select(ITEM_COLUMNS)
        .join(required_ids.lazy(), on="id", how="semi")
        .collect(engine="streaming")
    )
    duplicate_count = items.height - items.get_column("id").n_unique()
    if duplicate_count:
        raise ValueError(f"items parquet contains {duplicate_count} duplicate ids")
    missing_ids = required_ids.join(items.select("id"), on="id", how="anti")
    if missing_ids.height:
        raise ValueError(
            f"matches parquet references {missing_ids.height} unknown item ids"
        )
    return items, matches


def _predict(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> np.ndarray:
    prepared_items = prepare_manifest_items(items, solution, solution_root)
    batch = PredictionBatch(
        items=prepared_items.frame,
        matches=matches,
        attributes_column=prepared_items.attributes_column,
    )
    return build_predictor(solution, solution_root).predict_proba(batch)


def create_submission(
    items_path: str | Path,
    matches_path: str | Path,
    output_path: str | Path,
    *,
    solution_path: str | Path | None = None,
) -> pl.DataFrame:
    """Create a validated ``id1,id2,predict`` CSV in original pair order."""
    solution, solution_root = _load_solution(solution_path)
    items, matches = _read_inputs(items_path, matches_path)
    predictions = np.asarray(
        _predict(items, matches, solution, solution_root),
        dtype=np.float64,
    )
    if predictions.shape != (matches.height,):
        raise RuntimeError(
            f"predictor returned shape {predictions.shape}; expected {(matches.height,)}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("predictions contain NaN or infinity")

    result = matches.select(PAIR_COLUMNS).with_columns(
        pl.Series("predict", predictions)
    )
    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(target_path)
    return result


__all__ = ["create_submission"]
