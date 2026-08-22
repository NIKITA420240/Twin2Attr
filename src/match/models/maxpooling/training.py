"""Training of FastText, max-pooling features and their MLP classifier."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ...config import AppConfig
from ...data import TrainingData
from ..artifacts import TrainingArtifacts
from .features import (
    TRAIN_MATCH_COLUMNS,
    build_corpus,
    encode_loaded_pairs,
    pair_groups,
    train_fasttext,
    validate_columns,
    validate_items,
)
from .model import MaxPoolingModel, PairMLP, resolve_device
from .serialization import save_maxpooling_model


def _evaluate_classifier(
    classifier: PairMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    classifier.eval()
    total_loss = 0.0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(features)
            total_loss += criterion(logits, labels).item() * features.size(0)
            targets.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    target_values = np.concatenate(targets)
    probability_values = np.concatenate(probabilities)
    return (
        total_loss / len(loader.dataset),
        float(roc_auc_score(target_values, probability_values)),
        float(average_precision_score(target_values, probability_values)),
    )


def _train_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
    batch_size: int,
    epochs: int,
    patience: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device | None,
) -> tuple[StandardScaler, PairMLP, float, float]:
    if len(features) < 4:
        raise ValueError("at least four matched pairs are required for training")
    classes, counts = np.unique(targets, return_counts=True)
    if set(classes.tolist()) != {0, 1} or counts.min() < 2:
        raise ValueError("target must contain at least two examples of both classes")
    splitter = GroupShuffleSplit(
        n_splits=32,
        test_size=validation_fraction,
        random_state=random_state,
    )
    split = next(
        (
            (train_indices, validation_indices)
            for train_indices, validation_indices in splitter.split(
                features,
                targets,
                groups,
            )
            if len(np.unique(targets[train_indices])) == 2
            and len(np.unique(targets[validation_indices])) == 2
        ),
        None,
    )
    if split is None:
        raise ValueError(
            "validation split cannot contain both target classes without "
            "leaking duplicate or reversed pairs; add more distinct pairs or "
            "increase validation_fraction"
        )
    train_indices, validation_indices = split
    train_y = targets[train_indices]
    validation_y = targets[validation_indices]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[train_indices]).astype(np.float32)
    validation_x = scaler.transform(features[validation_indices]).astype(np.float32)

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    target_device = resolve_device(device)
    classifier = PairMLP(features.shape[1], dropout=dropout).to(target_device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_x),
        torch.from_numpy(validation_y.astype(np.float32)),
    )
    effective_batch_size = min(batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
        drop_last=len(train_dataset) % effective_batch_size == 1,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    positive_count = max(int(np.sum(train_y == 1)), 1)
    negative_count = int(np.sum(train_y == 0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=target_device,
        )
    )
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    best_auc = -np.inf
    best_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_train_loss = 0.0
        trained_samples = 0
        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(target_device, non_blocking=True)
            targets_batch = targets_batch.to(target_device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(classifier(features_batch), targets_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * features_batch.size(0)
            trained_samples += features_batch.size(0)
        validation_loss, validation_auc, validation_pr_auc = _evaluate_classifier(
            classifier,
            validation_loader,
            criterion,
            target_device,
        )
        scheduler.step(validation_auc)
        logger.info(
            "Classifier epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, "
            "val_roc_auc={:.4f}, val_pr_auc={:.4f}",
            epoch,
            epochs,
            total_train_loss / trained_samples,
            validation_loss,
            validation_auc,
            validation_pr_auc,
        )
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_pr_auc = validation_pr_auc
            best_state = copy.deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Classifier early stopping at epoch {}", epoch)
                break
    if best_state is None:
        raise RuntimeError("classifier training did not produce a valid state")
    classifier.load_state_dict(best_state)
    return scaler, classifier.cpu(), float(best_auc), float(best_pr_auc)


def train_maxpooling_model(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    attributes_column: str = "attributes",
    vector_size: int = 256,
    window: int = 5,
    min_count: int = 2,
    workers: int = 8,
    fasttext_epochs: int = 10,
    classifier_epochs: int = 50,
    batch_size: int = 512,
    validation_fraction: float = 0.2,
    patience: int = 7,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
    device: str | torch.device | None = None,
) -> MaxPoolingModel:
    if vector_size < 1 or window < 1 or min_count < 1 or workers < 1:
        raise ValueError("FastText numeric parameters must be positive")
    if fasttext_epochs < 1 or classifier_epochs < 1 or batch_size < 2:
        raise ValueError("epoch counts must be positive and batch_size at least two")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if patience < 1:
        raise ValueError("patience must be positive")
    started_at = perf_counter()
    validate_items(items, attributes_column)
    validate_columns(matches, TRAIN_MATCH_COLUMNS, "training matches")
    corpus = build_corpus(items, attributes_column)
    fasttext = train_fasttext(
        corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=fasttext_epochs,
        random_state=random_state,
    )
    features = encode_loaded_pairs(items, matches, fasttext, attributes_column)
    targets = matches.get_column("target").to_numpy()
    scaler, classifier, best_auc, best_pr_auc = _train_classifier(
        features,
        targets,
        pair_groups(matches),
        validation_fraction=validation_fraction,
        random_state=random_state,
        batch_size=batch_size,
        epochs=classifier_epochs,
        patience=patience,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )
    logger.info(
        "Finished max-pooling training: items={}, pairs={}, "
        "best_val_roc_auc={:.4f}, best_val_pr_auc={:.4f}, elapsed_seconds={:.3f}",
        items.height,
        matches.height,
        best_auc,
        best_pr_auc,
        perf_counter() - started_at,
    )
    return MaxPoolingModel(
        _fasttext=fasttext,
        _scaler=scaler,
        _classifier=classifier,
        vector_size=vector_size,
        best_validation_auc=best_auc,
        best_validation_pr_auc=best_pr_auc,
    )


@dataclass(frozen=True, slots=True)
class MaxPoolingTrainer:
    config: AppConfig

    def train(self, data: TrainingData) -> TrainingArtifacts:
        parameters = self.config.models_parameters.maxpooling
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
        output_path = save_maxpooling_model(
            model,
            self.config.artifacts.maxpooling_path,
        )
        return TrainingArtifacts(
            predictor="maxpooling",
            maxpooling_path=output_path,
            metrics=(
                ("maxpooling.validation_roc_auc", model.best_validation_auc),
                ("maxpooling.validation_pr_auc", model.best_validation_pr_auc),
            ),
        )


__all__ = ["MaxPoolingTrainer", "train_maxpooling_model"]
