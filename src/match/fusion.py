"""Train and apply a classifier over Transformer and max-pooling embeddings."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "FusionClassifier",
    "FusionConfig",
    "FusionTrainingResult",
    "load_fusion_classifier",
    "predict_fusion_probabilities",
    "train_fusion_classifier",
]


@dataclass(frozen=True)
class FusionConfig:
    """Training parameters for the classifier over concatenated embeddings."""

    hidden_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs and patience must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")


@dataclass(frozen=True)
class FusionTrainingResult:
    """Summary of a trained and persisted fusion head."""

    model_path: Path
    best_validation_macro_pr_auc: float
    trained_epochs: int
    cls_dim: int
    maxpooling_dim: int


class FusionClassifier(nn.Module):
    """Classify a pair after concatenating two independently produced vectors.

    Normalization is performed separately because CLS and max-pooling vectors
    can have different scales. Concatenation happens inside each batch, so the
    full ``[N, D_cls + D_max]`` matrix is never duplicated in host memory.
    """

    def __init__(
        self,
        cls_dim: int,
        maxpooling_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if cls_dim < 1 or maxpooling_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding and hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.cls_dim = cls_dim
        self.maxpooling_dim = maxpooling_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cls_normalization = nn.LayerNorm(cls_dim)
        self.maxpooling_normalization = nn.LayerNorm(maxpooling_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim + maxpooling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cls_embeddings: torch.Tensor,
        maxpooling_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if cls_embeddings.ndim != 2 or cls_embeddings.shape[1] != self.cls_dim:
            raise ValueError(f"cls_embeddings must have shape [B, {self.cls_dim}]")
        if (maxpooling_embeddings.ndim != 2 or maxpooling_embeddings.shape[1] != self.maxpooling_dim):
            raise ValueError(
                "maxpooling_embeddings must have shape "
                f"[B, {self.maxpooling_dim}]"
            )
        if cls_embeddings.shape[0] != maxpooling_embeddings.shape[0]:
            raise ValueError("embedding batches must contain the same number of rows")
        combined = torch.cat(
            (
                self.cls_normalization(cls_embeddings),
                self.maxpooling_normalization(maxpooling_embeddings),
            ),
            dim=1,
        )
        return self.classifier(combined).squeeze(1)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embedding_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _binary_labels(values: Sequence[int], *, name: str, expected_rows: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected_rows:
        raise ValueError(f"{name} must contain one label per embedding row")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def _validate_embedding_pair(
    cls_embeddings: np.ndarray,
    maxpooling_embeddings: np.ndarray,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    cls_array = _embedding_array(cls_embeddings, name=f"{split_name}_cls_embeddings")
    maxpooling_array = _embedding_array(
        maxpooling_embeddings,
        name=f"{split_name}_maxpooling_embeddings",
    )
    if len(cls_array) != len(maxpooling_array):
        raise ValueError(f"{split_name} embedding matrices must have the same row count")
    return cls_array, maxpooling_array


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
        scores.append(float(average_precision_score(category_labels, probabilities[mask])))
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
    validation_cls_embeddings: np.ndarray,
    validation_maxpooling_embeddings: np.ndarray,
    validation_labels: Sequence[int],
    validation_categories: Sequence[str],
    config: FusionConfig,
    *,
    output_path: str | Path,
    device: str | torch.device | None = None,
) -> FusionTrainingResult:
    """Train only the fusion head and select its epoch by macro PR-AUC.

    Transformer and max-pooling encoders are outside this function and remain
    frozen. Rows of both embedding matrices must describe the same pairs in
    exactly the same order.
    """
    train_cls, train_maxpooling = _validate_embedding_pair(
        train_cls_embeddings,
        train_maxpooling_embeddings,
        split_name="train",
    )
    validation_cls, validation_maxpooling = _validate_embedding_pair(
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
    target_device = _resolve_device(device)
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
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_cls),
        torch.from_numpy(validation_maxpooling),
        torch.from_numpy(validation_targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        pin_memory=target_device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(config.batch_size, len(validation_dataset)),
        shuffle=False,
        pin_memory=target_device.type == "cuda",
    )

    negative_count = int(np.sum(train_targets == 0))
    positive_count = int(np.sum(train_targets == 1))
    positive_weight = negative_count / positive_count
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=target_device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    logger.info(
        "Training fusion head: train_rows={}, validation_rows={}, cls_dim={}, "
        "maxpooling_dim={}, positive_weight={:.6f}, device={}",
        len(train_dataset),
        len(validation_dataset),
        model.cls_dim,
        model.maxpooling_dim,
        positive_weight,
        target_device,
    )

    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    trained_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for cls_batch, maxpooling_batch, label_batch in train_loader:
            cls_batch = cls_batch.to(target_device, non_blocking=True)
            maxpooling_batch = maxpooling_batch.to(target_device, non_blocking=True)
            label_batch = label_batch.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(cls_batch, maxpooling_batch), label_batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(label_batch)

        validation_probabilities = _validation_probabilities(
            model,
            validation_loader,
            target_device,
        )
        score = _macro_pr_auc(
            validation_targets,
            validation_probabilities,
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
    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "cls_dim": model.cls_dim,
            "maxpooling_dim": model.maxpooling_dim,
            "config": asdict(config),
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "best_validation_macro_pr_auc": float(best_score),
        },
        target_path,
    )
    logger.info(
        "Saved fusion head: path={!s}, best_val_macro_pr_auc={:.6f}",
        target_path,
        best_score,
    )
    return FusionTrainingResult(
        model_path=target_path,
        best_validation_macro_pr_auc=float(best_score),
        trained_epochs=trained_epochs,
        cls_dim=model.cls_dim,
        maxpooling_dim=model.maxpooling_dim,
    )


def load_fusion_classifier(
    model_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> FusionClassifier:
    """Load a fusion head saved by :func:`train_fusion_classifier`."""
    target_device = _resolve_device(device)
    checkpoint = torch.load(
        Path(model_path).expanduser().resolve(),
        map_location=target_device,
        weights_only=True,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported fusion checkpoint format")
    saved_config = FusionConfig(**checkpoint["config"])
    model = FusionClassifier(
        int(checkpoint["cls_dim"]),
        int(checkpoint["maxpooling_dim"]),
        hidden_dim=saved_config.hidden_dim,
        dropout=saved_config.dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(target_device).eval()


def predict_fusion_probabilities(
    model: FusionClassifier,
    cls_embeddings: np.ndarray,
    maxpooling_embeddings: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Return positive-class probabilities for aligned embedding matrices."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cls_array, maxpooling_array = _validate_embedding_pair(
        cls_embeddings,
        maxpooling_embeddings,
        split_name="inference",
    )
    if cls_array.shape[1] != model.cls_dim:
        raise ValueError(f"CLS dimension must equal trained dimension {model.cls_dim}")
    if maxpooling_array.shape[1] != model.maxpooling_dim:
        raise ValueError(
            "max-pooling dimension must equal trained dimension "
            f"{model.maxpooling_dim}"
        )

    dataset = TensorDataset(
        torch.from_numpy(cls_array),
        torch.from_numpy(maxpooling_array),
    )
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for cls_batch, maxpooling_batch in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        ):
            logits = model(cls_batch.to(device), maxpooling_batch.to(device))
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)
