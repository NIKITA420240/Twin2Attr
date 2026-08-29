"""Two-stage Transformer/CatBoost stacking training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score

from ...config import AppConfig
from ...data import TrainingData
from ..artifacts import TrainingArtifacts
from ..boosting.features import BoostingFeatureBuilder
from ..contracts import PredictionBatch
from ..transformer.predictor import TransformerPredictor
from ..transformer.training import TransformerTrainer
from .features import build_stacking_features
from .serialization import save_stacking_model


def _labels(pairs) -> np.ndarray:
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    if labels.size == 0 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("stacking train and validation must contain both classes")
    return labels


def _macro_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    categories: list[str],
) -> float:
    values = np.asarray(categories, dtype=object)
    scores: list[float] = []
    for category in np.unique(values):
        mask = values == category
        if np.unique(labels[mask]).size < 2:
            continue
        scores.append(float(average_precision_score(labels[mask], probabilities[mask])))
    if not scores:
        raise ValueError("stacking validation has no category with both classes")
    return float(np.mean(scores))


@dataclass(frozen=True, slots=True)
class StackingTrainer:
    config: AppConfig
    transformer: TransformerTrainer

    def train(self, data: TrainingData) -> TrainingArtifacts:
        if data.stacking_matches is None or data.stacking_pairs is None:
            raise ValueError("stacking training requires a dedicated stacking split")

        transformer_artifacts = self.transformer.train(data)
        if transformer_artifacts.transformer_dir is None:
            raise RuntimeError("Transformer trainer did not produce an artifact")
        length_bucketing = self.config.inference.transformer.length_bucketing
        batch_fields = (
            self.config.model_description.transformer.tokenizer.batch_fields
        )
        transformer = TransformerPredictor.load(
            transformer_artifacts.transformer_dir,
            batch_size=self.config.inference.transformer.batch_size,
            dtype=self.config.inference.transformer.dtype,
            num_workers=self.config.inference.transformer.num_workers,
            prefetch_factor=self.config.inference.transformer.prefetch_factor,
            pin_memory=self.config.inference.transformer.pin_memory,
            non_blocking_transfer=(
                self.config.inference.transformer.non_blocking_transfer
            ),
            length_bucketing=length_bucketing.enabled,
            padding_length_buckets=length_bucketing.padding_length_buckets,
            batch_fields=batch_fields.enabled,
            field_chunk_size=batch_fields.chunk_size,
            compile_enabled=(
                self.config.inference.transformer.torch_compile.enabled
            ),
            compile_mode=self.config.inference.transformer.torch_compile.mode,
            compile_dynamic=(
                self.config.inference.transformer.torch_compile.dynamic
            ),
            device=self.config.runtime.device,
        )
        stacking_batch = PredictionBatch(
            data.items,
            data.stacking_matches,
            data.attributes_column,
            data.stacking_pairs,
        )
        validation_batch = PredictionBatch(
            data.items,
            data.validation_matches,
            data.attributes_column,
            data.validation_pairs,
        )
        builder = BoostingFeatureBuilder(
            self.config.pair_features.typed_attributes
        )
        train_features = build_stacking_features(
            stacking_batch,
            transformer.predict_logit_margin(stacking_batch),
            structured_builder=builder,
        )
        validation_features = build_stacking_features(
            validation_batch,
            transformer.predict_logit_margin(validation_batch),
            structured_builder=builder,
        )
        train_labels = _labels(data.stacking_pairs)
        validation_labels = _labels(data.validation_pairs)
        train_weights = np.asarray(
            [pair.sample_weight for pair in data.stacking_pairs],
            dtype=np.float32,
        )

        from catboost import CatBoostClassifier

        parameters = self.config.model_description.boosting
        model = CatBoostClassifier(
            iterations=parameters.iterations,
            depth=parameters.depth,
            learning_rate=parameters.learning_rate,
            loss_function=parameters.loss_function,
            eval_metric="PRAUC:type=Classic",
            random_seed=self.config.runtime.seed,
            thread_count=parameters.thread_count,
            allow_writing_files=False,
            verbose=50,
        )
        model.fit(
            train_features,
            train_labels,
            sample_weight=train_weights,
            cat_features=list(builder.categorical_features),
            eval_set=(validation_features, validation_labels),
            early_stopping_rounds=parameters.early_stopping_rounds,
            use_best_model=True,
        )
        probabilities = np.asarray(
            model.predict_proba(
                validation_features,
                thread_count=parameters.thread_count,
            )[:, 1],
            dtype=np.float32,
        )
        score = _macro_pr_auc(
            validation_labels,
            probabilities,
            [pair.category for pair in data.validation_pairs],
        )
        output_dir = save_stacking_model(
            model,
            self.config.model_description.stacking.artifact_dir,
            list(train_features.columns),
            feature_options=builder.feature_options,
        )
        logger.info(
            "Stacking training finished: validation_macro_pr_auc={:.6f}, "
            "artifact={!s}",
            score,
            output_dir,
        )
        return TrainingArtifacts(
            predictor="stacking",
            transformer_dir=transformer_artifacts.transformer_dir,
            stacking_dir=output_dir,
            metrics=(
                *transformer_artifacts.metrics,
                ("stacking.validation_macro_pr_auc", score),
            ),
        )


__all__ = ["StackingTrainer"]
