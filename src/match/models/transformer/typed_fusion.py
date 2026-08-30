"""Typed-attribute tensors and batching for Transformer fusion heads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    """Build one finite float32 feature row per prepared pair."""
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
        matrix[row_index] = [float(values[name]) for name in names]
    if not np.isfinite(matrix).all():
        raise ValueError("typed fusion features must contain only finite values")
    return names, matrix


@dataclass(frozen=True, slots=True)
class TypedDatasetItem:
    base: Any
    features: np.ndarray


class TypedFeatureDataset(Dataset[TypedDatasetItem]):
    """Attach precomputed typed rows to cached or uncached pair datasets."""

    def __init__(self, base: Dataset[Any], features: np.ndarray) -> None:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(base):
            raise ValueError("typed feature matrix must align with the base dataset")
        if not np.isfinite(matrix).all():
            raise ValueError("typed feature matrix must contain only finite values")
        self.base = base
        self.features = np.ascontiguousarray(matrix)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> TypedDatasetItem:
        return TypedDatasetItem(self.base[index], self.features[index])


class TypedPairEncodingCollator:
    """Add a fixed-width typed tensor to the normal encoded pair batch."""

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

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty typed fusion batch")
        if all(isinstance(example, TypedDatasetItem) for example in examples):
            typed_examples = [
                example for example in examples if isinstance(example, TypedDatasetItem)
            ]
            base_examples = [example.base for example in typed_examples]
            matrix = np.stack([example.features for example in typed_examples])
        elif all(isinstance(example, PreparedPair) for example in examples):
            base_examples = examples
            names, matrix = build_typed_feature_matrix(examples, self.options)
            if names != self.feature_names:
                raise RuntimeError("inference typed schema does not match artifact")
        else:
            raise TypeError("typed fusion batch items have incompatible types")
        batch = self.base_collator(base_examples)
        batch["typed_features"] = torch.from_numpy(
            np.ascontiguousarray(matrix, dtype=np.float32)
        )
        return batch


__all__ = [
    "TYPED_FUSION_SCHEMA_VERSION",
    "TypedFeatureDataset",
    "TypedPairEncodingCollator",
    "build_typed_feature_matrix",
]
