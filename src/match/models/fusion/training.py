"""Training strategy and optimization loop for the fusion head."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from ...config import AppConfig
from ...data import TrainingData
from ..artifacts import TrainingArtifacts
from ..contracts import PredictionBatch
from ..maxpooling.predictor import MaxPoolingPredictor
from ..maxpooling.training import MaxPoolingTrainer
from ..transformer.predictor import TransformerPredictor
from ..transformer.training import TransformerTrainer
from .model import (
    FusionClassifier,
    FusionConfig,
    FusionTrainingResult,
    resolve_device,
    validate_embedding_pair,
)
from .serialization import save_fusion_classifier


def _binary_labels(
    values: Sequence[int],
    *,
    name: str,
    expected_rows: int,
) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected_rows:
        raise ValueError(f"{name} must contain one label per embedding row")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _macro_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    categories: np.ndarray,
) -> float:
    scores: list[float] = []
    for category in np.unique(categories):
        mask = categories == category
        category_labels = labels[mask]
        if len(np.unique(category_labels)) < 2:
            logger.warning(
                "Skipping category {!r} in fusion macro PR-AUC: only one class",
                category,
            )
            continue
        scores.append(
            float(average_precision_score(category_labels, probabilities[mask]))
        )
    if not scores:
        raise ValueError("validation categories contain no category with both classes")
    return float(np.mean(scores))


def _validation_probabilities(
    model: FusionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for cls_batch, maxpooling_batch, _ in loader:
            logits = model(
                cls_batch.to(device, non_blocking=True),
                maxpooling_batch.to(device, non_blocking=True),
            )
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def train_fusion_classifier(
    train_cls_embeddings: np.ndarray,
    train_maxpooling_embeddings: np.ndarray,
    train_labels: Sequence[int],
    train_weights: Sequence[float],
    validation_cls_embeddings: np.ndarray,
    validation_maxpooling_embeddings: np.ndarray,
    validation_labels: Sequence[int],
    validation_categories: Sequence[str],
    config: FusionConfig,
    *,
    output_path,
    device: str | torch.device | None = None,
) -> FusionTrainingResult:
    train_cls, train_maxpooling = validate_embedding_pair(
        train_cls_embeddings,
        train_maxpooling_embeddings,
        split_name="train",
    )
    validation_cls, validation_maxpooling = validate_embedding_pair(
        validation_cls_embeddings,
        validation_maxpooling_embeddings,
        split_name="validation",
    )
    if train_cls.shape[1] != validation_cls.shape[1]:
        raise ValueError("train and validation CLS dimensions must match")
    if train_maxpooling.shape[1] != validation_maxpooling.shape[1]:
        raise ValueError("train and validation max-pooling dimensions must match")
    train_targets = _binary_labels(
        train_labels,
        name="train_labels",
        expected_rows=len(train_cls),
    )
    weights = np.asarray(train_weights, dtype=np.float32)
    if weights.shape != train_targets.shape or np.any(weights <= 0.0):
        raise ValueError("train_weights must be positive and aligned with labels")
    validation_targets = _binary_labels(
        validation_labels,
        name="validation_labels",
        expected_rows=len(validation_cls),
    )
    categories = np.asarray(validation_categories)
    if categories.ndim != 1 or len(categories) != len(validation_cls):
        raise ValueError("validation_categories must contain one value per row")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = resolve_device(device)
    model = FusionClassifier(
        train_cls.shape[1],
        train_maxpooling.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(target_device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_cls),
        torch.from_numpy(train_maxpooling),
        torch.from_numpy(train_targets.astype(np.float32)),
        torch.from_numpy(weights),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_cls),
        torch.from_numpy(validation_maxpooling),
        torch.from_numpy(validation_targets.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=target_device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.batch_size, len(validation_dataset)),
        shuffle=False,
        pin_memory=target_device.type == "cuda",
    )
    negative_count = float(weights[train_targets == 0].sum())
    positive_count = float(weights[train_targets == 1].sum())
    positive_weight = torch.tensor(
        negative_count / positive_count,
        device=target_device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    trained_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for cls_batch, maxpooling_batch, label_batch, weight_batch in train_loader:
            cls_batch = cls_batch.to(target_device, non_blocking=True)
            maxpooling_batch = maxpooling_batch.to(target_device, non_blocking=True)
            label_batch = label_batch.to(target_device, non_blocking=True)
            weight_batch = weight_batch.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                model(cls_batch, maxpooling_batch),
                label_batch,
                pos_weight=positive_weight,
                reduction="none",
            )
            loss = torch.sum(losses * weight_batch) / torch.sum(weight_batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(label_batch)
        score = _macro_pr_auc(
            validation_targets,
            _validation_probabilities(model, validation_loader, target_device),
            categories,
        )
        trained_epochs = epoch
        logger.info(
            "Fusion epoch {}/{}: train_loss={:.6f}, val_macro_pr_auc={:.6f}",
            epoch,
            config.max_epochs,
            loss_sum / len(train_dataset),
            score,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Fusion early stopping at epoch {}", epoch)
                break
    if best_state is None:
        raise RuntimeError("fusion training did not produce a valid model state")
    model.load_state_dict(best_state)
    model_path = save_fusion_classifier(
        model,
        config,
        best_state,
        best_validation_macro_pr_auc=float(best_score),
        model_path=output_path,
    )
    return FusionTrainingResult(
        model_path=model_path,
        best_validation_macro_pr_auc=float(best_score),
        trained_epochs=trained_epochs,
        cls_dim=model.cls_dim,
        maxpooling_dim=model.maxpooling_dim,
    )


def _fusion_config(config: AppConfig) -> FusionConfig:
    parameters = config.model_description.fusion
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
        parameters = self.config.model_description.fusion
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
        result = train_fusion_classifier(
            transformer.encode(train_batch),
            maxpooling.encode(train_batch),
            [int(pair.label) for pair in data.train_pairs],
            [pair.sample_weight for pair in data.train_pairs],
            transformer.encode(validation_batch),
            maxpooling.encode(validation_batch),
            [int(pair.label) for pair in data.validation_pairs],
            [pair.category for pair in data.validation_pairs],
            _fusion_config(self.config),
            output_path=self.config.model_description.fusion.artifact_path,
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


__all__ = ["FusionTrainer", "train_fusion_classifier"]
