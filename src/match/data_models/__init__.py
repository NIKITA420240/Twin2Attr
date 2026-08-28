"""Configurable recipes for constructing model training datasets."""

from .contracts import InspectionFrames, LoadedTrainingSplits, TrainingDataModel
from .factory import build_data_model
from .models import BaseDatasetModel, MixedDatasetModel
from .preparation import (
    finalize_source_matches,
    prepare_source_labels,
    prepare_source_matches,
    resolve_source_overlaps,
)

__all__ = [
    "BaseDatasetModel",
    "InspectionFrames",
    "LoadedTrainingSplits",
    "MixedDatasetModel",
    "TrainingDataModel",
    "build_data_model",
    "finalize_source_matches",
    "prepare_source_labels",
    "prepare_source_matches",
    "resolve_source_overlaps",
]
