"""Factory for configured training-data recipes."""

from __future__ import annotations

from ..config import AppConfig
from .contracts import TrainingDataModel
from .models import BaseDatasetModel, MixedDatasetModel


def build_data_model(config: AppConfig) -> TrainingDataModel:
    """Build the data recipe selected by ``training.data_model``."""
    name = config.training.data_model
    descriptions = config.data_model_description
    if name == "base_dataset":
        return BaseDatasetModel(
            descriptions.base_dataset,
            create_stacking_split=config.training.model == "stacking",
        )
    if name == "mix_dataset":
        return MixedDatasetModel(descriptions.mix_dataset)
    raise ValueError(f"Unsupported training data model: {name!r}")


__all__ = ["build_data_model"]
