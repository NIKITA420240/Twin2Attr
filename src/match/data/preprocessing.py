"""One item preprocessing sequence shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import polars as pl

from ..features.contracts import PreparedItems
from ..features.factory import (
    build_feature_pipeline,
    build_manifest_feature_pipeline,
)

if TYPE_CHECKING:
    from ..config import AppConfig


def prepare_configured_items(
    items: pl.DataFrame,
    config: AppConfig,
) -> PreparedItems:
    normalization = config.normalization
    attributes_column = normalization.source_column
    if normalization.enabled:
        from ..normalization import normalize_attributes

        items = normalize_attributes(
            items,
            normalization.synonyms_path,
            normalization.unique_attributes_path,
            source_column=normalization.source_column,
            output_column=normalization.output_column,
            n_jobs=normalization.n_jobs,
            chunk_size=normalization.chunk_size,
        )
        attributes_column = normalization.output_column
    return build_feature_pipeline(config).enrich(
        PreparedItems(items, attributes_column)
    )


def _solution_path(value: Any, *, root: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"solution field {name!r} must contain a path")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def prepare_manifest_items(
    items: pl.DataFrame,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> PreparedItems:
    attributes_column = "attributes"
    normalization = solution.get("normalization")
    if isinstance(normalization, Mapping):
        attributes_column = str(
            normalization.get("source_column", "attributes")
        )
    if isinstance(normalization, Mapping) and bool(
        normalization.get("enabled", False)
    ):
        from ..normalization import normalize_attributes

        attributes_column = str(
            normalization.get("output_column", "normalized_attributes")
        )
        items = normalize_attributes(
            items,
            _solution_path(
                normalization.get("synonyms_path"),
                root=solution_root,
                name="normalization.synonyms_path",
            ),
            _solution_path(
                normalization.get("unique_attributes_path"),
                root=solution_root,
                name="normalization.unique_attributes_path",
            ),
            source_column=str(
                normalization.get("source_column", "attributes")
            ),
            output_column=attributes_column,
            n_jobs=int(normalization.get("n_jobs", 1)),
            chunk_size=int(normalization.get("chunk_size", 5_000)),
        )
    return build_manifest_feature_pipeline(solution, solution_root).enrich(
        PreparedItems(items, attributes_column)
    )


__all__ = [
    "prepare_configured_items",
    "prepare_manifest_items",
]
