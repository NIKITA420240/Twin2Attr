"""Explicit workflow for inspecting encoded pair lengths."""

from __future__ import annotations

from ..data import prepare_pair_rows, read_parquet
from ..config import AppConfig
from ._common import (
    check_optional_features,
    normalization_enabled,
    prepare_items,
    resolve_max_length,
    workflow_logging,
)


def inspect_max_length(config: AppConfig) -> int:
    """Prepare inspection pairs and return their recommended encoded length."""
    with workflow_logging(config, workflow_name="inspect"):
        check_optional_features(config)
        items = read_parquet(
            config.paths.items,
            label="items",
        )
        attributes_column = config.normalization.source_column
        if normalization_enabled(config):
            items, attributes_column = prepare_items(items, config)

        configured_path = config.paths.inspect_matches or config.paths.train_matches
        inspect_matches = read_parquet(
            configured_path,
            label="inspection matches",
        )
        inspect_pairs = prepare_pair_rows(
            items,
            inspect_matches,
            attributes_column,
            split_name="inspection",
        )
        return resolve_max_length(config, inspect_pairs)


__all__ = ["inspect_max_length"]
