"""Deterministic attribute-order augmentation for prepared product pairs."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..config import AppConfig, AttributeShuffleSettings
from ..prepare_data import PreparedCard, PreparedPair


@dataclass(frozen=True, slots=True)
class AugmentedPairs:
    pairs: tuple[PreparedPair, ...]
    source_indices: tuple[int, ...]


def _stable_seed(
    settings: AttributeShuffleSettings,
    pair: PreparedPair,
    *,
    copy_index: int,
    side: str,
) -> int:
    side_value = side if settings.shuffle_cards_independently else "pair"
    value = "\x1f".join(
        (
            str(settings.seed),
            repr(pair.left.item_id),
            repr(pair.right.item_id),
            str(copy_index),
            side_value,
        )
    )
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _shuffle_card(
    card: PreparedCard,
    *,
    seed: int,
) -> PreparedCard:
    attributes = list(card.attributes)
    random.Random(seed).shuffle(attributes)
    return replace(card, attributes=tuple(attributes))


def _shuffle_pair(
    pair: PreparedPair,
    settings: AttributeShuffleSettings,
    *,
    copy_index: int,
) -> PreparedPair:
    return replace(
        pair,
        left=_shuffle_card(
            pair.left,
            seed=_stable_seed(
                settings,
                pair,
                copy_index=copy_index,
                side="left",
            ),
        ),
        right=_shuffle_card(
            pair.right,
            seed=_stable_seed(
                settings,
                pair,
                copy_index=copy_index,
                side="right",
            ),
        ),
        preserve_attribute_order=True,
        skip_oversized_attributes=settings.skip_oversized,
    )


def apply_pair_augmentation(
    pairs: Sequence[PreparedPair],
    config: AppConfig,
    *,
    model_name: str | None,
) -> AugmentedPairs:
    """Apply a configured augmentation and retain source-row alignment."""
    if model_name is None:
        return AugmentedPairs(tuple(pairs), tuple(range(len(pairs))))
    if model_name != "attribute_shuffle":
        raise ValueError(f"Unsupported augmentation model: {model_name!r}")

    settings = config.augmentation_models.attribute_shuffle
    return apply_attribute_shuffle(pairs, settings)


def apply_attribute_shuffle(
    pairs: Sequence[PreparedPair],
    settings: AttributeShuffleSettings,
) -> AugmentedPairs:
    """Create stable shuffled copies according to one resolved descriptor."""
    augmented: list[PreparedPair] = []
    source_indices: list[int] = []
    for source_index, pair in enumerate(pairs):
        if settings.keep_original:
            augmented.append(pair)
            source_indices.append(source_index)
        for copy_index in range(settings.shuffled_copies):
            augmented.append(
                _shuffle_pair(pair, settings, copy_index=copy_index)
            )
            source_indices.append(source_index)
    return AugmentedPairs(tuple(augmented), tuple(source_indices))


def apply_manifest_augmentation(
    pairs: Sequence[PreparedPair],
    solution: Mapping[str, Any],
) -> AugmentedPairs:
    """Apply the pair augmentation serialized in an inference manifest."""
    model_name = solution.get("augmentation_model")
    if model_name is None:
        return AugmentedPairs(tuple(pairs), tuple(range(len(pairs))))
    if str(model_name) != "attribute_shuffle":
        raise ValueError(f"Unsupported augmentation model: {model_name!r}")
    models = solution.get("augmentation_models")
    if not isinstance(models, Mapping):
        raise TypeError("solution augmentation_models must contain a mapping")
    values = models.get("attribute_shuffle")
    if not isinstance(values, Mapping):
        raise TypeError(
            "solution augmentation_models.attribute_shuffle must be a mapping"
        )

    def boolean(name: str) -> bool:
        value = values.get(name)
        if not isinstance(value, bool):
            raise TypeError(f"attribute_shuffle.{name} must be a boolean")
        return value

    settings = AttributeShuffleSettings(
        type=str(values.get("type", "")),
        shuffled_copies=int(values.get("shuffled_copies", 0)),
        keep_original=boolean("keep_original"),
        seed=int(values.get("seed", 0)),
        shuffle_cards_independently=boolean("shuffle_cards_independently"),
        skip_oversized=boolean("skip_oversized"),
    )
    return apply_attribute_shuffle(pairs, settings)


__all__ = [
    "AugmentedPairs",
    "apply_attribute_shuffle",
    "apply_manifest_augmentation",
    "apply_pair_augmentation",
]
