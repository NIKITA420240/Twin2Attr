"""Factory for configured training-data recipes."""

from __future__ import annotations

from ..config import AppConfig
from .contracts import TrainingDataModel
from .models import BaseDatasetModel, MixedDatasetModel


def build_data_model(
    config: AppConfig,
    *,
    name: str | None = None,
) -> TrainingDataModel:
    """Build a named data recipe or the one selected for training."""
    selected_name = config.training.data_model if name is None else name
    descriptions = config.data_model_description
    if selected_name == "base_dataset":
        return BaseDatasetModel(
            descriptions.base_dataset,
            create_stacking_split=(
                name is None and config.training.model == "stacking"
            ),
        )
    if selected_name == "mix_dataset":
        return MixedDatasetModel(descriptions.mix_dataset)
    if selected_name == "mix_dataset_codex":
        return MixedDatasetModel(descriptions.mix_dataset_codex)
    raise ValueError(f"Unsupported data model: {selected_name!r}")


__all__ = ["build_data_model"]
