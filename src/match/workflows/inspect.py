"""Explicit workflow for inspecting encoded pair lengths."""

from __future__ import annotations

from ..config import AppConfig
from ..data import prepare_configured_items, prepare_pair_rows
from ..data_models import build_data_model
from ._common import resolve_max_length, workflow_logging


def inspect_max_length(config: AppConfig) -> int:
    """Prepare inspection pairs and return their recommended encoded length."""
    with workflow_logging(config, workflow_name="inspect"):
        frames = build_data_model(config).load_inspection_frames()
        items = frames.items
        prepared_items = prepare_configured_items(items, config)
        items = prepared_items.frame
        attributes_column = prepared_items.attributes_column

        inspect_pairs = prepare_pair_rows(
            items,
            frames.matches,
            attributes_column,
            split_name="inspection",
        )
        return resolve_max_length(config, inspect_pairs)


__all__ = ["inspect_max_length"]
