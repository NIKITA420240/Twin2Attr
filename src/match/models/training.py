"""Training strategies selected from ``training.model``."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import joblib
from loguru import logger

from ..config import AppConfig
from ..data import TrainingData
from ..fusion import FusionConfig, train_fusion_classifier
from ..maxpooling import train_maxpooling_model
from ..transformer import SequenceClassifierConfig, train_sequence_classifier
from .artifacts import TrainingArtifacts
from .contracts import PredictionBatch
from .predictors import MaxPoolingPredictor, TransformerPredictor


class ModelTrainer(Protocol):
    """Common training capability used by the train workflow."""

    def train(self, data: TrainingData) -> TrainingArtifacts:
        ...


def _sequence_config(config: AppConfig) -> SequenceClassifierConfig:
    parameters = config.models_parameters.transformer
    encoding = config.pair_encoding
    return SequenceClassifierConfig(
        model_path=parameters.pretrained_model_path,
        max_epochs=parameters.max_epochs,
        hpo_trials=parameters.hpo_trials,
        seed=config.runtime.seed,
        use_field_tokens=encoding.use_field_tokens,
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        max_length=encoding.max_length,
        max_length_quantile=encoding.quantile,
        max_length_sample_size=encoding.sample_size,
        max_length_hard_cap=encoding.hard_cap,
    )


def _fusion_config(config: AppConfig) -> FusionConfig:
    parameters = config.models_parameters.fusion
    return FusionConfig(
        hidden_dim=parameters.hidden_dim,
        dropout=parameters.dropout,
        batch_size=parameters.batch_size,
        max_epochs=parameters.max_epochs,
        patience=parameters.patience,
        learning_rate=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
        seed=config.runtime.seed,
    )


@dataclass(frozen=True, slots=True)
class TransformerTrainer:
    config: AppConfig

    def train(self, data: TrainingData) -> TrainingArtifacts:
        output_dir = self.config.artifacts.transformer_dir
        result = train_sequence_classifier(
            data.train_pairs,
            data.validation_pairs,
            _sequence_config(self.config),
            output_dir=output_dir,
        )
        return TrainingArtifacts(
            predictor="transformer",
            transformer_dir=result.model_dir,
            metrics=(
                (
                    "transformer.validation_macro_pr_auc",
                    result.validation_macro_pr_auc,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class MaxPoolingTrainer:
    config: AppConfig

    def train(self, data: TrainingData) -> TrainingArtifacts:
        parameters = self.config.models_parameters.maxpooling
        output_path = self.config.artifacts.maxpooling_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model = train_maxpooling_model(
            data.items,
            data.train_matches,
            attributes_column=data.attributes_column,
            vector_size=parameters.vector_size,
            window=parameters.window,
            min_count=parameters.min_count,
            workers=parameters.workers,
            fasttext_epochs=parameters.fasttext_epochs,
            classifier_epochs=parameters.classifier_epochs,
            batch_size=parameters.batch_size,
            validation_fraction=parameters.validation_fraction,
            patience=parameters.patience,
            dropout=parameters.dropout,
            learning_rate=parameters.learning_rate,
            weight_decay=parameters.weight_decay,
            random_state=self.config.runtime.seed,
            device=self.config.runtime.device,
        )
        joblib.dump(model, output_path)
        logger.info("Saved max-pooling model to {!s}", output_path)
        return TrainingArtifacts(
            predictor="maxpooling",
            maxpooling_path=output_path,
            metrics=(
                ("maxpooling.validation_roc_auc", model.best_validation_auc),
                ("maxpooling.validation_pr_auc", model.best_validation_pr_auc),
            ),
        )


@dataclass(frozen=True, slots=True)
class FusionTrainer:
    config: AppConfig
    transformer: TransformerTrainer
    maxpooling: MaxPoolingTrainer

    def train(self, data: TrainingData) -> TrainingArtifacts:
        transformer_artifacts = self.transformer.train(data)
        maxpooling_artifacts = self.maxpooling.train(data)
        if transformer_artifacts.transformer_dir is None:
            raise RuntimeError("transformer trainer did not produce its artifact")
        if maxpooling_artifacts.maxpooling_path is None:
            raise RuntimeError("max-pooling trainer did not produce its artifact")

        started_at = perf_counter()
        parameters = self.config.models_parameters.fusion
        transformer = TransformerPredictor.load(
            transformer_artifacts.transformer_dir,
            batch_size=parameters.embedding_batch_size,
            device=self.config.runtime.device,
        )
        maxpooling = MaxPoolingPredictor.load(
            maxpooling_artifacts.maxpooling_path,
            batch_size=parameters.batch_size,
            device=self.config.runtime.device,
        )
        train_batch = PredictionBatch(
            items=data.items,
            matches=data.train_matches,
            attributes_column=data.attributes_column,
            pairs=data.train_pairs,
        )
        validation_batch = PredictionBatch(
            items=data.items,
            matches=data.validation_matches,
            attributes_column=data.attributes_column,
            pairs=data.validation_pairs,
        )
        output_path = self.config.artifacts.fusion_path
        result = train_fusion_classifier(
            transformer.encode(train_batch),
            maxpooling.encode(train_batch),
            [int(pair.label) for pair in data.train_pairs],
            transformer.encode(validation_batch),
            maxpooling.encode(validation_batch),
            [int(pair.label) for pair in data.validation_pairs],
            [pair.category for pair in data.validation_pairs],
            _fusion_config(self.config),
            output_path=output_path,
            device=self.config.runtime.device,
        )
        logger.info(
            "Fusion training finished: best_val_macro_pr_auc={:.6f}, "
            "elapsed_seconds={:.3f}",
            result.best_validation_macro_pr_auc,
            perf_counter() - started_at,
        )
        return TrainingArtifacts(
            predictor="fusion",
            transformer_dir=transformer_artifacts.transformer_dir,
            maxpooling_path=maxpooling_artifacts.maxpooling_path,
            fusion_path=result.model_path,
            metrics=(
                *transformer_artifacts.metrics,
                *maxpooling_artifacts.metrics,
                (
                    "fusion.validation_macro_pr_auc",
                    result.best_validation_macro_pr_auc,
                ),
            ),
        )


def build_trainer(config: AppConfig) -> ModelTrainer:
    """Construct the training strategy selected by ``training.model``."""
    if config.training.model == "transformer":
        return TransformerTrainer(config)
    if config.training.model == "maxpooling":
        return MaxPoolingTrainer(config)
    if config.training.model == "fusion":
        return FusionTrainer(
            config=config,
            transformer=TransformerTrainer(config),
            maxpooling=MaxPoolingTrainer(config),
        )
    raise ValueError(f"Unsupported training model: {config.training.model!r}")


__all__ = [
    "FusionTrainer",
    "MaxPoolingTrainer",
    "ModelTrainer",
    "TrainingArtifacts",
    "TransformerTrainer",
    "build_trainer",
]
