"""Evaluator-compatible inference over packaged Twin2Attr model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from .paths import PROJECT_ROOT
from .prepare_data import prepare_pairs


PAIR_COLUMNS = ("id1", "id2")
ITEM_COLUMNS = ("id", "name", "attributes", "category")


def _resolve_from_solution(value: Any, *, root: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"solution field {name!r} must contain a path")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


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


def _prepare_attributes(
    items: pl.DataFrame,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> tuple[pl.DataFrame, str]:
    settings = solution.get("normalization")
    if not isinstance(settings, Mapping) or not bool(settings.get("enabled", False)):
        return items, "attributes"

    from .normalization import normalize_attributes

    output_column = str(settings.get("output_column", "normalized_attributes"))
    normalized = normalize_attributes(
        items,
        _resolve_from_solution(
            settings.get("synonyms_path"),
            root=solution_root,
            name="normalization.synonyms_path",
        ),
        _resolve_from_solution(
            settings.get("unique_attributes_path"),
            root=solution_root,
            name="normalization.unique_attributes_path",
        ),
        source_column=str(settings.get("source_column", "attributes")),
        output_column=output_column,
        n_jobs=int(settings.get("n_jobs", 1)),
        chunk_size=int(settings.get("chunk_size", 5_000)),
    )
    return normalized, output_column


def _transformer_artifacts(
    solution: Mapping[str, Any],
    solution_root: Path,
):
    from .transformer import load_trained_classifier

    model_directory = _resolve_from_solution(
        solution.get("model_directory"),
        root=solution_root,
        name="model_directory",
    )
    return load_trained_classifier(model_directory, device=solution.get("device"))


def _maxpooling_embeddings(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    attributes_column: str,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> np.ndarray:
    import joblib

    from .maxpooling import MaxPoolingModel, encode_attribute_pairs

    model_path = _resolve_from_solution(
        solution.get("maxpooling_path"),
        root=solution_root,
        name="maxpooling_path",
    )
    model = joblib.load(model_path)
    if not isinstance(model, MaxPoolingModel):
        raise TypeError("maxpooling_path does not contain a MaxPoolingModel")

    return encode_attribute_pairs(
        items,
        matches,
        model,
        attributes_column=attributes_column,
    )


def _predict(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> np.ndarray:
    from .transformer import encode_pair_cls, predict_match_probabilities

    items, attributes_column = _prepare_attributes(items, solution, solution_root)
    pairs = prepare_pairs(items, matches, attributes_column=attributes_column)
    tokenizer, transformer = _transformer_artifacts(solution, solution_root)
    batch_size = int(solution.get("batch_size", 64))
    predictor = str(solution.get("predictor", "transformer"))

    if predictor == "transformer":
        return predict_match_probabilities(
            transformer,
            tokenizer,
            pairs,
            batch_size=batch_size,
        )
    if predictor == "fusion":
        from .fusion import load_fusion_classifier, predict_fusion_probabilities

        cls_embeddings = encode_pair_cls(
            transformer,
            tokenizer,
            pairs,
            batch_size=batch_size,
        )
        maxpooling_embeddings = _maxpooling_embeddings(
            items,
            matches,
            attributes_column,
            solution,
            solution_root,
        )
        fusion_path = _resolve_from_solution(
            solution.get("fusion_path"),
            root=solution_root,
            name="fusion_path",
        )
        fusion = load_fusion_classifier(fusion_path, device=solution.get("device"))
        return predict_fusion_probabilities(
            fusion,
            cls_embeddings,
            maxpooling_embeddings,
            batch_size=int(solution.get("fusion_batch_size", 512)),
        )
    raise ValueError(f"Unsupported predictor in solution.json: {predictor!r}")


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
