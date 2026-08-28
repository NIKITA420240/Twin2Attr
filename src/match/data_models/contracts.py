"""Contracts shared by configurable training-data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True, slots=True)
class LoadedTrainingSplits:
    """Raw items and labeled matches prepared for model-independent processing."""

    items: pl.DataFrame
    train_matches: pl.DataFrame
    validation_matches: pl.DataFrame
    stacking_matches: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class InspectionFrames:
    """Items and representative matches used by analysis workflows."""

    items: pl.DataFrame
    matches: pl.DataFrame


class TrainingDataModel(Protocol):
    """A configured recipe capable of constructing training data splits."""

    def load_training_splits(self) -> LoadedTrainingSplits:
        ...

    def load_inspection_frames(self) -> InspectionFrames:
        ...


__all__ = ["InspectionFrames", "LoadedTrainingSplits", "TrainingDataModel"]
