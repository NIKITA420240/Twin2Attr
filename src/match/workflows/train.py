"""Explicit application workflow for model training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..augmentations import apply_pair_augmentation
from ..config import AppConfig, save_app_config
from ..data import prepare_configured_items, prepare_training_data
from ..data_models import build_data_model
from ..data_postprocessing import apply_pair_postprocessing
from ..experiments import save_experiment_record
from ..models.artifacts import TrainingArtifacts, save_solution_manifest
from ..models.factory import build_trainer
from ._common import workflow_logging


def train(
    config: AppConfig,
    *,
    experiment_name: str | None = None,
    experiment_registry_path: Path | None = None,
) -> TrainingArtifacts:
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
            stacking_matches=getattr(splits, "stacking_matches", None),
        )
        if config.training.augmentation_model is not None:
            augmented = apply_pair_augmentation(
                data.train_pairs,
                config,
                model_name=config.training.augmentation_model,
            )
            source_indices = list(augmented.source_indices)
            data = replace(
                data,
                train_matches=data.train_matches[source_indices],
                train_pairs=list(augmented.pairs),
            )
        if config.training.data_postprocessing_model is not None:
            data = replace(
                data,
                train_pairs=list(
                    apply_pair_postprocessing(
                        data.train_pairs,
                        config,
                        model_name=config.training.data_postprocessing_model,
                    )
                ),
            )
        trainer = build_trainer(config)
        artifacts = trainer.train(data)
        save_app_config(config, config.training.resolved_config_path)
        solution_path = save_solution_manifest(config, artifacts)
        result = replace(
            artifacts,
            resolved_config_path=config.training.resolved_config_path,
            solution_path=solution_path,
        )
        if experiment_name is not None:
            if experiment_registry_path is None:
                raise ValueError(
                    "experiment_registry_path is required with experiment_name"
                )
            save_experiment_record(
                config,
                result,
                splits,
                experiment_name=experiment_name,
                registry_path=experiment_registry_path,
            )
        return result


__all__ = ["train"]
