"""Contracts for per-row training sample-weight models."""

from __future__ import annotations

from typing import Protocol

import polars as pl


class SampleWeightModel(Protocol):
    """Calculate a positive multiplier for every labeled source row."""

    def apply(self, matches: pl.DataFrame, *, source_name: str) -> pl.DataFrame:
        ...


__all__ = ["SampleWeightModel"]
