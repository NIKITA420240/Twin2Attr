"""Typed-attribute tensors and batching for Transformer fusion heads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ...pair_encoding import PairEncodingCollator
from ...pair_features import (
    TypedAttributeComparator,
    TypedAttributeOptions,
    aggregate_typed_attribute_features,
    typed_attribute_feature_names,
)
from ...prepare_data import PreparedPair


TYPED_FUSION_SCHEMA_VERSION = 1


def build_typed_feature_matrix(
    pairs: Sequence[PreparedPair],
    options: TypedAttributeOptions,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build one finite float32 row per prepared pair in a stable column order."""
    if not options.enabled:
        raise ValueError("typed fusion requires enabled typed attribute options")
    names = typed_attribute_feature_names(enabled_types=options.enabled_types)
    comparator = TypedAttributeComparator(options)
    matrix = np.empty((len(pairs), len(names)), dtype=np.float32)
    for row_index, pair in enumerate(pairs):
        values = aggregate_typed_attribute_features(
            comparator.compare_pair(pair),
            enabled_types=options.enabled_types,
        )
        if tuple(values) != names:
            raise RuntimeError("typed feature schema changed while building a matrix")
        matrix[row_index] = tuple(float(values[name]) for name in names)
    if not np.isfinite(matrix).all():
        raise ValueError("typed fusion features must contain only finite values")
    return names, matrix


@dataclass(frozen=True, slots=True)
class TypedPreparedPair:
    pair: PreparedPair
    features: np.ndarray


class TypedPreparedPairDataset(Dataset):
    """Pair dataset with typed features precomputed from unaugmented attributes."""

    def __init__(
        self,
        pairs: Sequence[PreparedPair],
        features: np.ndarray,
        *,
        transform: Callable[[PreparedPair], PreparedPair] | None = None,
    ) -> None:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(pairs):
            raise ValueError("typed feature matrix must align with prepared pairs")
        if not np.isfinite(matrix).all():
            raise ValueError("typed feature matrix must contain only finite values")
        self._pairs = pairs
        self._features = np.ascontiguousarray(matrix)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> TypedPreparedPair:
        pair = self._pairs[index]
        if self._transform is not None:
            pair = self._transform(pair)
        return TypedPreparedPair(pair, self._features[index])


class TypedPairEncodingCollator:
    """Add a fixed-width typed tensor to the existing pair encoding batch."""

    def __init__(
        self,
        base_collator: PairEncodingCollator,
        *,
        options: TypedAttributeOptions,
        feature_names: Sequence[str],
    ) -> None:
        if not options.enabled:
            raise ValueError("typed fusion collator requires enabled options")
        self.base_collator = base_collator
        self.options = options
        self.feature_names = tuple(feature_names)
        expected = typed_attribute_feature_names(enabled_types=options.enabled_types)
        if self.feature_names != expected:
            raise ValueError("typed fusion feature names do not match options")

    def __call__(
        self,
        examples: list[PreparedPair | TypedPreparedPair],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty typed fusion batch")
        if all(isinstance(example, TypedPreparedPair) for example in examples):
            typed_examples = [
                example for example in examples if isinstance(example, TypedPreparedPair)
            ]
            pairs = [example.pair for example in typed_examples]
            matrix = np.stack(
                [example.features for example in typed_examples],
                axis=0,
            ).astype(np.float32, copy=False)
        elif all(isinstance(example, PreparedPair) for example in examples):
            pairs = [example for example in examples if isinstance(example, PreparedPair)]
            names, matrix = build_typed_feature_matrix(pairs, self.options)
            if names != self.feature_names:
                raise RuntimeError("inference typed feature schema does not match artifact")
        else:
            raise TypeError("typed fusion batches cannot mix pair example types")
        batch = self.base_collator(pairs)
        batch["typed_features"] = torch.from_numpy(
            np.ascontiguousarray(matrix, dtype=np.float32)
        )
        return batch


__all__ = [
    "TYPED_FUSION_SCHEMA_VERSION",
    "TypedPairEncodingCollator",
    "TypedPreparedPair",
    "TypedPreparedPairDataset",
    "build_typed_feature_matrix",
]
