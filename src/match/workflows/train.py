"""Explicit application workflow for model training."""

from __future__ import annotations

from dataclasses import replace

from ..config import AppConfig, save_app_config
from ..data import prepare_configured_items, prepare_training_data
from ..data_models import build_data_model
from ..models.artifacts import TrainingArtifacts, save_solution_manifest
from ..models.factory import build_trainer
from ._common import workflow_logging


def train(config: AppConfig) -> TrainingArtifacts:
    """Prepare data, train the selected model and persist its manifest."""
    with workflow_logging(config, workflow_name="train"):
        splits = build_data_model(config).load_training_splits()
        items = splits.items
        prepared_items = prepare_configured_items(items, config)
        items = prepared_items.frame
        attributes_column = prepared_items.attributes_column

        data = prepare_training_data(
            items,
            splits.train_matches,
            splits.validation_matches,
            attributes_column=attributes_column,
        )
        trainer = build_trainer(config)
        artifacts = trainer.train(data)
        save_app_config(config, config.artifacts.resolved_config_path)
        solution_path = save_solution_manifest(config, artifacts)
        return replace(
            artifacts,
            resolved_config_path=config.artifacts.resolved_config_path,
            solution_path=solution_path,
        )


__all__ = ["train"]
