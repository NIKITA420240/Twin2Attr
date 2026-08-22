"""FastText training and max-pooling pair feature extraction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import orjson
import polars as pl
from gensim.models import FastText
from loguru import logger

from .model import MaxPoolingModel

TRAIN_MATCH_COLUMNS = {"id1", "id2", "target"}
INFERENCE_MATCH_COLUMNS = {"id1", "id2"}
_NON_ALNUM_PATTERN = re.compile(r"[^а-яёa-z0-9]+")
_SPACE_PATTERN = re.compile(r"\s+")
CardEmbedding = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def validate_columns(
    frame: pl.DataFrame,
    required: set[str],
    label: str,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a polars.DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def validate_items(items: pl.DataFrame, attributes_column: str) -> None:
    if not attributes_column.strip():
        raise ValueError("attributes_column must not be empty")
    validate_columns(items, {"id", attributes_column}, "items")
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
    return {
        _normalize_text(key): _normalize_text(value)
        for key, value in parsed.items()
    }


def build_corpus(
    items: pl.DataFrame,
    attributes_column: str,
) -> list[list[str]]:
    corpus: list[list[str]] = []
    for raw in items.get_column(attributes_column):
        tokens = [
            token
            for key, value in _parse_attributes(raw).items()
            for token in (*key.split(), *value.split())
        ]
        if tokens:
            corpus.append(tokens)
    if not corpus:
        raise ValueError("cannot train FastText on an empty attribute corpus")
    return corpus


def train_fasttext(
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


def _unit_embedding(text: str, model: FastText) -> np.ndarray:
    tokens = text.split()
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    embedding = np.zeros(model.vector_size, dtype=np.float32)
    for token in tokens:
        index = model.wv.key_to_index.get(token)
        embedding += (
            model.wv.get_vector(token)
            if index is None
            else model.wv.vectors[index]
        )
    embedding *= np.float32(1.0 / len(tokens))
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


def _pool_card(raw_attributes: Any, model: FastText) -> CardEmbedding:
    attributes = _parse_attributes(raw_attributes)
    if not attributes:
        empty = np.zeros(model.vector_size, dtype=np.float32)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()
    keys = np.asarray(
        [_unit_embedding(key, model) for key in attributes],
        dtype=np.float32,
    )
    values = np.asarray(
        [_unit_embedding(value, model) for value in attributes.values()],
        dtype=np.float32,
    )
    return keys.min(0), keys.max(0), values.min(0), values.max(0)


def _max_pairwise_product(
    out: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    temporary: np.ndarray,
) -> None:
    np.multiply(a_min, b_min, out=out)
    for left, right in ((a_min, b_max), (a_max, b_min), (a_max, b_max)):
        np.multiply(left, right, out=temporary)
        np.maximum(out, temporary, out=out)


def _pair_embedding(
    out: np.ndarray,
    left: CardEmbedding,
    right: CardEmbedding,
    temporary: np.ndarray,
) -> None:
    dimension = len(left[0])
    _max_pairwise_product(
        out[:dimension],
        left[0],
        left[1],
        right[0],
        right[1],
        temporary[:dimension],
    )
    _max_pairwise_product(
        out[dimension:],
        left[2],
        left[3],
        right[2],
        right[3],
        temporary[dimension:],
    )


def pair_groups(matches: pl.DataFrame) -> np.ndarray:
    group_by_pair: dict[frozenset[Any], int] = {}
    groups = np.empty(matches.height, dtype=np.int64)
    for index, (id1, id2) in enumerate(matches.select("id1", "id2").iter_rows()):
        pair = frozenset((id1, id2))
        groups[index] = group_by_pair.setdefault(pair, len(group_by_pair))
    return groups


def encode_loaded_pairs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    model: FastText,
    attributes_column: str,
) -> np.ndarray:
    required_ids = set(matches.get_column("id1")) | set(matches.get_column("id2"))
    missing_ids = required_ids - set(items.get_column("id"))
    if missing_ids:
        sample = sorted(map(str, missing_ids))[:10]
        raise ValueError(
            f"match parquet references {len(missing_ids)} unknown item ids: {sample}"
        )
    selected = items.filter(pl.col("id").is_in(list(required_ids)))
    cache = {
        item_id: _pool_card(raw_attributes, model)
        for item_id, raw_attributes in selected.select(
            "id",
            attributes_column,
        ).iter_rows()
    }
    feature_size = 2 * model.vector_size
    embeddings = np.empty((matches.height, feature_size), dtype=np.float32)
    temporary = np.empty(feature_size, dtype=np.float32)
    for index, (id1, id2) in enumerate(matches.select("id1", "id2").iter_rows()):
        _pair_embedding(embeddings[index], cache[id1], cache[id2], temporary)
    return embeddings


def encode_attribute_pairs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    model: MaxPoolingModel,
    *,
    attributes_column: str = "attributes",
) -> np.ndarray:
    if not isinstance(model, MaxPoolingModel):
        raise TypeError("model must be a MaxPoolingModel")
    started_at = perf_counter()
    validate_items(items, attributes_column)
    validate_columns(matches, INFERENCE_MATCH_COLUMNS, "matches")
    result = encode_loaded_pairs(
        items,
        matches,
        model._fasttext,
        attributes_column,
    )
    logger.info(
        "Encoded max-pooling pairs: rows={}, dimensions={}, elapsed_seconds={:.3f}",
        matches.height,
        result.shape[1],
        perf_counter() - started_at,
    )
    return result


__all__ = ["encode_attribute_pairs"]
