"""Training strategy for the fast CatBoost matcher."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score

from ...config import AppConfig
from ...data import TrainingData
from ..artifacts import TrainingArtifacts
from ..contracts import PredictionBatch
from .features import BoostingFeatureBuilder
from .serialization import save_boosting_model


def _labels(data: TrainingData, *, validation: bool) -> np.ndarray:
    pairs = data.validation_pairs if validation else data.train_pairs
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int64)
    if labels.size == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(
            "boosting train and validation labels must be non-empty and binary"
        )
    return labels


def _macro_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    categories: list[str],
) -> float:
    scores = []
    category_values = np.asarray(categories, dtype=object)
    for category in np.unique(category_values):
        mask = category_values == category
        if np.unique(labels[mask]).size < 2:
            logger.warning(
                "Skipping category {!r} in boosting macro PR-AUC: only one class",
                category,
            )
            continue
        scores.append(float(average_precision_score(labels[mask], probabilities[mask])))
    if not scores:
        raise ValueError("boosting validation has no category containing both classes")
    return float(np.mean(scores))


class BoostingTrainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def train(self, data: TrainingData) -> TrainingArtifacts:
        from catboost import CatBoostClassifier

        parameters = self.config.model_description.boosting
        builder = BoostingFeatureBuilder()
        train_batch = PredictionBatch(
            data.items,
            data.train_matches,
            data.attributes_column,
            data.train_pairs,
        )
        validation_batch = PredictionBatch(
            data.items,
            data.validation_matches,
            data.attributes_column,
            data.validation_pairs,
        )
        train_features = builder.transform(train_batch)
        validation_features = builder.transform(validation_batch)
        train_labels = _labels(data, validation=False)
        validation_labels = _labels(data, validation=True)
        train_weights = np.asarray(
            [pair.sample_weight for pair in data.train_pairs],
            dtype=np.float32,
        )
        if np.unique(train_labels).size != 2 or np.unique(validation_labels).size != 2:
            raise ValueError(
                "boosting train and validation splits must contain both classes"
            )

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
        output_dir = save_boosting_model(
            model,
            self.config.model_description.boosting.artifact_dir,
            list(train_features.columns),
        )
        logger.info(
            "Boosting training finished: validation_macro_pr_auc={:.6f}, artifact={!s}",
            score,
            output_dir,
        )
        return TrainingArtifacts(
            predictor="boosting",
            boosting_dir=output_dir,
            metrics=(("boosting.validation_macro_pr_auc", score),),
        )


__all__ = ["BoostingTrainer"]
