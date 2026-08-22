"""Train and apply the FastText/max-pooling pair pipeline."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import orjson
import polars as pl
import torch
from gensim.models import FastText
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["encode_attribute_pairs", "train_maxpooling_model"]


_TRAIN_MATCH_COLUMNS = {"id1", "id2", "target"}
_INFERENCE_MATCH_COLUMNS = {"id1", "id2"}
_NON_ALNUM_PATTERN = re.compile(r"[^а-яёa-z0-9]+")
_SPACE_PATTERN = re.compile(r"\s+")

CardEmbedding = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class PairMLP(nn.Module):
    """Binary classifier from the original max-pooling notebook."""

    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


@dataclass(slots=True)
class MaxPoolingModel:
    """Fitted pipeline state kept behind the two public functions."""

    _fasttext: FastText
    _scaler: StandardScaler
    _classifier: PairMLP
    vector_size: int
    best_validation_auc: float
    best_validation_pr_auc: float


def _validate_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a polars.DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_items(items: pl.DataFrame, attributes_column: str) -> None:
    if not attributes_column.strip():
        raise ValueError("attributes_column must not be empty")
    _validate_columns(items, {"id", attributes_column}, "items")
    if items.get_column("id").null_count():
        raise ValueError("items contains null ids")
    duplicate_count = items.height - items.get_column("id").n_unique()
    if duplicate_count:
        raise ValueError(f"items contains {duplicate_count} duplicate ids")


def _normalize_text(text: Any) -> str:
    normalized = _NON_ALNUM_PATTERN.sub(" ", str(text).lower())
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def _parse_attributes(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        try:
            parsed = orjson.loads(raw)
        except (orjson.JSONDecodeError, TypeError) as error:
            raise ValueError("attributes must contain a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError("attributes must contain a JSON object")
    return {_normalize_text(key): _normalize_text(value) for key, value in parsed.items()}


def _attributes_to_tokens(attributes: Mapping[str, str]) -> list[str]:
    tokens: list[str] = []
    for key, value in attributes.items():
        tokens.extend(key.split())
        tokens.extend(value.split())
    return tokens


def _build_corpus(items: pl.DataFrame, attributes_column: str) -> list[list[str]]:
    corpus = []
    for raw in items.get_column(attributes_column):
        tokens = _attributes_to_tokens(_parse_attributes(raw))
        if tokens:
            corpus.append(tokens)
    if not corpus:
        raise ValueError("cannot train FastText on an empty attribute corpus")
    return corpus


def _train_fasttext(
    corpus: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    epochs: int,
    random_state: int,
) -> FastText:
    logger.info(
        "Training internal FastText: documents={}, vector_size={}, epochs={}, workers={}",
        len(corpus),
        vector_size,
        epochs,
        workers,
    )
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        min_n=2,
        max_n=5,
        epochs=epochs,
        seed=random_state,
    )
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model


def _text_embedding(text: str, model: FastText) -> np.ndarray:
    tokens = text.split()
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)

    embedding = np.zeros(model.vector_size, dtype=np.float32)
    for token in tokens:
        index = model.wv.key_to_index.get(token)
        if index is None:
            embedding += model.wv.get_vector(token)
        else:
            embedding += model.wv.vectors[index]
    return embedding * np.float32(1.0 / len(tokens))


def _unit_embedding(text: str, model: FastText) -> np.ndarray:
    embedding = _text_embedding(text, model)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


def _pool_card(raw_attributes: Any, model: FastText) -> CardEmbedding:
    attributes = _parse_attributes(raw_attributes)
    if not attributes:
        empty = np.zeros(model.vector_size, dtype=np.float32)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    keys = np.asarray([_unit_embedding(key, model) for key in attributes], dtype=np.float32)
    values = np.asarray([_unit_embedding(value, model) for value in attributes.values()], dtype=np.float32)
    return (
        keys.min(axis=0),
        keys.max(axis=0),
        values.min(axis=0),
        values.max(axis=0),
    )


def _max_pairwise_product(
    out: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    temporary: np.ndarray,
) -> None:
    np.multiply(a_min, b_min, out=out)
    np.multiply(a_min, b_max, out=temporary)
    np.maximum(out, temporary, out=out)
    np.multiply(a_max, b_min, out=temporary)
    np.maximum(out, temporary, out=out)
    np.multiply(a_max, b_max, out=temporary)
    np.maximum(out, temporary, out=out)


def _pair_embedding(
    out: np.ndarray,
    card1: CardEmbedding,
    card2: CardEmbedding,
    temporary: np.ndarray,
) -> None:
    dimension = len(card1[0])
    _max_pairwise_product(
        out[:dimension],
        card1[0],
        card1[1],
        card2[0],
        card2[1],
        temporary[:dimension],
    )
    _max_pairwise_product(
        out[dimension:],
        card1[2],
        card1[3],
        card2[2],
        card2[3],
        temporary[dimension:],
    )


def _required_item_ids(matches: pl.DataFrame) -> set[Any]:
    return set(matches.get_column("id1")) | set(matches.get_column("id2"))


def _pair_groups(matches: pl.DataFrame) -> np.ndarray:
    """Assign the same group to duplicate and reversed item pairs."""
    group_by_pair: dict[frozenset[Any], int] = {}
    groups = np.empty(matches.height, dtype=np.int64)
    for index, (id1, id2) in enumerate(matches.select("id1", "id2").iter_rows()):
        pair = frozenset((id1, id2))
        groups[index] = group_by_pair.setdefault(pair, len(group_by_pair))
    return groups


def _build_card_cache(
    items: pl.DataFrame,
    required_ids: set[Any],
    model: FastText,
    attributes_column: str,
) -> dict[Any, CardEmbedding]:
    available_ids = set(items.get_column("id"))
    missing_ids = required_ids - available_ids
    if missing_ids:
        sample = sorted(map(str, missing_ids))[:10]
        raise ValueError(f"match parquet references {len(missing_ids)} unknown item ids: {sample}")

    selected = items.filter(pl.col("id").is_in(list(required_ids)))
    return {
        item_id: _pool_card(raw_attributes, model)
        for item_id, raw_attributes in selected.select(
            "id", attributes_column
        ).iter_rows()
    }


def _encode_loaded_pairs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    model: FastText,
    attributes_column: str,
) -> np.ndarray:
    feature_size = 2 * model.vector_size
    embeddings = np.empty((matches.height, feature_size), dtype=np.float32)
    temporary = np.empty(feature_size, dtype=np.float32)
    cache = _build_card_cache(
        items,
        _required_item_ids(matches),
        model,
        attributes_column,
    )

    for index, (id1, id2) in enumerate(matches.select("id1", "id2").iter_rows()):
        _pair_embedding(embeddings[index], cache[id1], cache[id2], temporary)
    return embeddings


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            for train_indices, validation_indices in splitter.split(features, targets, groups)
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
    train_x = features[train_indices]
    validation_x = features[validation_indices]
    train_y = targets[train_indices]
    validation_y = targets[validation_indices]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    validation_x = scaler.transform(validation_x).astype(np.float32)

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    target_device = _resolve_device(device)
    classifier = PairMLP(features.shape[1], dropout=dropout).to(target_device)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count], dtype=torch.float32, device=target_device))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

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

        validation_loss, validation_auc, validation_pr_auc = _evaluate_classifier(classifier, validation_loader, criterion, target_device)
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
    """Train the internal FastText encoder and MLP from loaded data frames.

    ``items`` must contain ``id`` and ``attributes_column``. ``matches`` must
    contain ``id1``, ``id2`` and binary ``target`` columns. Neither frame is
    modified.
    """
    if vector_size < 1 or window < 1 or min_count < 1 or workers < 1:
        raise ValueError("FastText numeric parameters must be positive")
    if fasttext_epochs < 1 or classifier_epochs < 1 or batch_size < 2:
        raise ValueError("epoch counts must be positive and batch_size at least two")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if patience < 1:
        raise ValueError("patience must be positive")

    started_at = perf_counter()
    _validate_items(items, attributes_column)
    _validate_columns(matches, _TRAIN_MATCH_COLUMNS, "training matches")
    corpus = _build_corpus(items, attributes_column)
    fasttext = _train_fasttext(
        corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=fasttext_epochs,
        random_state=random_state,
    )
    del corpus
    pair_features = _encode_loaded_pairs(
        items,
        matches,
        fasttext,
        attributes_column,
    )
    targets = matches.get_column("target").to_numpy()
    scaler, classifier, best_auc, best_pr_auc = _train_classifier(
        pair_features,
        targets,
        _pair_groups(matches),
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
        "Finished max-pooling model training: items={}, pairs={}, "
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


def encode_attribute_pairs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    model: MaxPoolingModel,
    *,
    attributes_column: str = "attributes",
) -> np.ndarray:
    """Return raw max-pooling embeddings for the supplied pairs.

    Rows remain in ``matches`` order. Only ``id1`` and ``id2`` are required;
    ``target`` is deliberately ignored when present. The result has shape
    ``(pair_count, 2 * model.vector_size)`` and dtype ``float32``.
    """
    if not isinstance(model, MaxPoolingModel):
        raise TypeError("model must be returned by train_maxpooling_model")
    started_at = perf_counter()
    _validate_items(items, attributes_column)
    _validate_columns(matches, _INFERENCE_MATCH_COLUMNS, "matches")
    result = _encode_loaded_pairs(
        items,
        matches,
        model._fasttext,
        attributes_column,
    )
    logger.info(
        "Encoded max-pooling pair embeddings: pairs={}, dimensions={}, elapsed_seconds={:.3f}",
        matches.height,
        result.shape[1],
        perf_counter() - started_at,
    )
    return result
