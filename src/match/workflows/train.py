"""Explicit application workflow for model training."""

from __future__ import annotations

from dataclasses import replace

from ..config import AppConfig, save_app_config
from ..data import (
    TrainingMatchPaths,
    load_training_matches,
    prepare_configured_items,
    prepare_training_data,
    read_parquet,
)
from ..data_split import DataSplitConfig
from ..models.artifacts import TrainingArtifacts, save_solution_manifest
from ..models.factory import build_trainer
from ._common import workflow_logging


def _training_match_paths(config: AppConfig) -> TrainingMatchPaths:
    split = config.split
    return TrainingMatchPaths(
        source=config.paths.train_matches,
        validation=config.paths.validation_matches,
        generated_train=split.train_output_path,
        generated_validation=split.validation_output_path,
    )


def _data_split_config(config: AppConfig) -> DataSplitConfig:
    split = config.split
    return DataSplitConfig(
        validation_fraction=split.validation_fraction,
        leakage_scope=split.leakage_scope,
        seed=split.seed,
        candidate_splits=split.candidate_splits,
    )


def train(config: AppConfig) -> TrainingArtifacts:
    """Prepare data, train the selected model and persist its manifest."""
    with workflow_logging(config, workflow_name="train"):
        items = read_parquet(config.paths.items, label="items")
        prepared_items = prepare_configured_items(items, config)
        items = prepared_items.frame
        attributes_column = prepared_items.attributes_column

        train_matches, validation_matches = load_training_matches(
            items,
            _training_match_paths(config),
            _data_split_config(config),
            mode=config.split.mode,
        )
        data = prepare_training_data(
            items,
            train_matches,
            validation_matches,
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
