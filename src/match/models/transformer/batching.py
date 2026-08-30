"""Backend-independent batching for Transformer pair inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from ...pair_encoding import PairEncodingCollator, PreparedPairDataset
from ...prepare_data import PreparedCard, PreparedPair
from .profile import SEQUENCE_CLASSIFIER_PROFILE


def _estimated_card_length(
    card: PreparedCard,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> int:
    """Estimate token demand cheaply, without running the tokenizer twice."""
    value_character_limit = max_attribute_value_chars
    if max_attribute_value_tokens is not None:
        token_estimate_limit = max_attribute_value_tokens * 4
        value_character_limit = (
            token_estimate_limit
            if value_character_limit is None
            else min(value_character_limit, token_estimate_limit)
        )
    demand = len(card.name) + len(card.category)
    for key, value in card.attributes:
        value_length = len(value)
        if value_character_limit is not None:
            value_length = min(value_length, value_character_limit)
        demand += len(key) + value_length
    return demand


def length_bucket_order(
    pairs: Sequence[PreparedPair],
    max_attribute_value_tokens: int | None,
    max_attribute_value_chars: int | None = None,
) -> np.ndarray:
    """Return sorted-position to original-position indices for bucketing."""
    estimates = np.fromiter(
        (
            _estimated_card_length(
                pair.left,
                max_attribute_value_chars,
                max_attribute_value_tokens,
            )
            + _estimated_card_length(
                pair.right,
                max_attribute_value_chars,
                max_attribute_value_tokens,
            )
            for pair in pairs
        ),
        dtype=np.int64,
        count=len(pairs),
    )
    return np.argsort(estimates, kind="stable")


class _OrderedPreparedPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PreparedPair],
        order: np.ndarray,
    ) -> None:
        self._pairs = pairs
        self._order = order

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index: int) -> PreparedPair:
        return self._pairs[int(self._order[index])]


def inference_dataset(
    pairs: Sequence[PreparedPair],
    *,
    length_bucketing: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> tuple[Dataset, np.ndarray | None]:
    if not length_bucketing or len(pairs) < 2:
        return PreparedPairDataset(pairs), None
    order = length_bucket_order(
        pairs,
        max_attribute_value_tokens,
        max_attribute_value_chars,
    )
    return _OrderedPreparedPairDataset(pairs, order), order


def inference_loader(
    dataset: Dataset,
    collator: PairEncodingCollator,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> tuple[DataLoader, bool]:
    use_pinned_memory = pin_memory and device.type == "cuda"
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": use_pinned_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs), use_pinned_memory


def restore_original_order(
    values: np.ndarray,
    order: np.ndarray | None,
) -> np.ndarray:
    if order is None:
        return values
    restored = np.empty_like(values)
    restored[order] = values
    return restored


@dataclass(frozen=True, slots=True)
class TransformerBatchingSettings:
    batch_size: int = 64
    retry_on_oom: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    length_bucketing: bool = False
    padding_length_buckets: tuple[int, ...] | None = None
    batch_fields: bool = False
    field_chunk_size: int = 16_384

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        if self.field_chunk_size < 1:
            raise ValueError("field_chunk_size must be positive")

    def collator(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        use_field_tokens: bool,
        max_attribute_value_chars: int | None,
        max_attribute_value_tokens: int | None,
        profile: str = SEQUENCE_CLASSIFIER_PROFILE,
    ) -> PairEncodingCollator:
        return PairEncodingCollator(
            tokenizer,
            max_length,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
            include_labels=False,
            padding_length_buckets=(
                self.padding_length_buckets if self.length_bucketing else None
            ),
            batch_fields=self.batch_fields,
            field_chunk_size=self.field_chunk_size,
            profile=profile,
        )


__all__ = [
    "TransformerBatchingSettings",
    "inference_dataset",
    "inference_loader",
    "length_bucket_order",
    "restore_original_order",
]
