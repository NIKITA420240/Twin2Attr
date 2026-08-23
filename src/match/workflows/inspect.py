"""Explicit workflow for inspecting encoded pair lengths."""

from __future__ import annotations

from ..config import AppConfig
from ..data import prepare_configured_items, prepare_pair_rows, read_parquet
from ._common import resolve_max_length, workflow_logging


def inspect_max_length(config: AppConfig) -> int:
    """Prepare inspection pairs and return their recommended encoded length."""
    with workflow_logging(config, workflow_name="inspect"):
        items = read_parquet(
            config.training.data.items,
            label="items",
        )
        prepared_items = prepare_configured_items(items, config)
        items = prepared_items.frame
        attributes_column = prepared_items.attributes_column

        data = config.training.data
        configured_path = data.inspect_matches or data.train_matches
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
