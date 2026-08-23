"""One item preprocessing sequence shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

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
    return build_feature_pipeline(config).enrich(
        PreparedItems(items, config.features.normalization.source_column)
    )


def prepare_manifest_items(
    items: pl.DataFrame,
    solution: Mapping[str, Any],
    solution_root: Path,
) -> PreparedItems:
    feature_values = solution.get("features")
    features = feature_values if isinstance(feature_values, Mapping) else {}
    normalization_value = features.get(
        "normalization",
        solution.get("normalization"),
    )
    normalization = (
        normalization_value if isinstance(normalization_value, Mapping) else {}
    )
    attributes_column = str(normalization.get("source_column", "attributes"))
    return build_manifest_feature_pipeline(solution, solution_root).enrich(
        PreparedItems(items, attributes_column)
    )


__all__ = [
    "prepare_configured_items",
    "prepare_manifest_items",
]
