"""Small capability interfaces shared by model implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import polars as pl

from ..prepare_data import PreparedPair, prepare_pairs

if TYPE_CHECKING:
    from ..data import TrainingData
    from .artifacts import TrainingArtifacts


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

    def take_indices(self, indices: Sequence[int]) -> PredictionBatch:
        """Select pair rows while keeping the shared item table unchanged."""
        selected = [int(index) for index in indices]
        if any(index < 0 or index >= self.matches.height for index in selected):
            raise IndexError("prediction batch index is out of range")
        pairs = self.prepared_pairs()
        return PredictionBatch(
            items=self.items,
            matches=self.matches[selected],
            attributes_column=self.attributes_column,
            pairs=[pairs[index] for index in selected],
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


@runtime_checkable
class ModelTrainer(Protocol):
    """A selected model strategy capable of consuming prepared training data."""

    def train(self, data: TrainingData) -> TrainingArtifacts:
        ...


__all__ = ["MatchPredictor", "ModelTrainer", "PairEncoder", "PredictionBatch"]
