"""Contracts shared by optional item feature providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl


@dataclass(frozen=True, slots=True)
class PreparedItems:
    frame: pl.DataFrame
    attributes_column: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pl.DataFrame):
            raise TypeError("frame must be a polars.DataFrame")
        if not self.attributes_column.strip():
            raise ValueError("attributes_column must not be empty")
        if self.attributes_column not in self.frame.columns:
            raise ValueError(
                f"items does not contain attributes column {self.attributes_column!r}"
            )


@runtime_checkable
class ItemEnricher(Protocol):
    def enrich(self, items: PreparedItems) -> PreparedItems:
        """Return items with additional feature columns."""
        ...


__all__ = ["ItemEnricher", "PreparedItems"]
