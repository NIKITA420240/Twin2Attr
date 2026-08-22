"""Small capability interfaces shared by model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl

from ..prepare_data import PreparedPair, prepare_pairs


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """Aligned source frames and structured pairs for model inference."""

    items: pl.DataFrame
    matches: pl.DataFrame
    attributes_column: str
    pairs: Sequence[PreparedPair] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.items, pl.DataFrame):
            raise TypeError("items must be a polars.DataFrame")
        if not isinstance(self.matches, pl.DataFrame):
            raise TypeError("matches must be a polars.DataFrame")
        if self.pairs is not None and len(self.pairs) != self.matches.height:
            raise ValueError("pairs and matches must contain the same number of rows")
        if not self.attributes_column.strip():
            raise ValueError("attributes_column must not be empty")

    def prepared_pairs(self) -> Sequence[PreparedPair]:
        """Return supplied structured pairs or create them from source frames."""
        if self.pairs is not None:
            return self.pairs
        return prepare_pairs(
            self.items,
            self.matches,
            attributes_column=self.attributes_column,
        )


@runtime_checkable
class PairEncoder(Protocol):
    """A component capable of converting aligned pairs into feature vectors."""

    @property
    def output_dim(self) -> int:
        ...

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        ...


@runtime_checkable
class MatchPredictor(Protocol):
    """A component producing one match probability for every input pair."""

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        ...


__all__ = ["MatchPredictor", "PairEncoder", "PredictionBatch"]
