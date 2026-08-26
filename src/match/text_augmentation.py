"""Structure-aware text augmentation for product-card pairs."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .prepare_data import PreparedCard, PreparedPair


@dataclass(frozen=True, slots=True)
class TextAugmentationConfig:
    """Probabilities and loss mixing weight for training-time augmentation.

    ``alpha`` is the weight of the original-batch loss. The augmented-batch
    loss therefore receives ``1 - alpha``.
    """

    enabled: bool = False
    alpha: float = 0.7
    attribute_dropout_probability: float = 0.15
    word_shuffle_probability: float = 0.15
    keyboard_typo_probability: float = 0.01
    word_dropout_probability: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("augmentation.alpha must be in [0, 1]")
        probabilities = {
            "attribute_dropout_probability": self.attribute_dropout_probability,
            "word_shuffle_probability": self.word_shuffle_probability,
            "keyboard_typo_probability": self.keyboard_typo_probability,
            "word_dropout_probability": self.word_dropout_probability,
        }
        for name, probability in probabilities.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"augmentation.{name} must be in [0, 1]")


def _keyboard_neighbors() -> dict[str, tuple[str, ...]]:
    layouts = (
        ("1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"),
        ("1234567890-=", "йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю."),
    )
    neighbors: dict[str, set[str]] = {}
    for rows in layouts:
        for row in rows:
            for index, character in enumerate(row):
                adjacent = neighbors.setdefault(character, set())
                if index:
                    adjacent.add(row[index - 1])
                if index + 1 < len(row):
                    adjacent.add(row[index + 1])
    return {
        character: tuple(sorted(adjacent))
        for character, adjacent in neighbors.items()
        if adjacent
    }


_KEYBOARD_NEIGHBORS = _keyboard_neighbors()


class ProductTextAugmenter:
    """Randomly perturb text while preserving product-card structure.

    Categories and attribute keys are intentionally kept intact because they
    carry schema semantics. Product names and attribute values receive word
    and typo noise, while complete attributes may be dropped independently.
    """

    def __init__(self, config: TextAugmentationConfig, *, seed: int) -> None:
        self.config = config
        self._random = random.Random(seed)

    def augment_pair(self, pair: PreparedPair) -> PreparedPair:
        if not self.config.enabled:
            return pair
        return replace(
            pair,
            left=self._augment_card(pair.left),
            right=self._augment_card(pair.right),
        )

    def _augment_card(self, card: PreparedCard) -> PreparedCard:
        attributes = tuple(
            (key, self._augment_text(value))
            for key, value in card.attributes
            if self._random.random() >= self.config.attribute_dropout_probability
        )
        return replace(
            card,
            name=self._augment_text(card.name),
            attributes=attributes,
        )

    def _augment_text(self, text: str) -> str:
        words = text.split()
        words = self._drop_words(words)
        if (
            len(words) > 1
            and self._random.random() < self.config.word_shuffle_probability
        ):
            original = list(words)
            self._random.shuffle(words)
            if words == original:
                words = words[1:] + words[:1]
        augmented = " ".join(words)
        return "".join(self._keyboard_typo(character) for character in augmented)

    def _drop_words(self, words: list[str]) -> list[str]:
        if len(words) < 2 or self.config.word_dropout_probability == 0.0:
            return words
        retained = [
            word
            for word in words
            if self._random.random() >= self.config.word_dropout_probability
        ]
        if retained:
            return retained
        return [self._random.choice(words)]

    def _keyboard_typo(self, character: str) -> str:
        neighbors = _KEYBOARD_NEIGHBORS.get(character.lower())
        if (
            not neighbors
            or self._random.random() >= self.config.keyboard_typo_probability
        ):
            return character
        replacement = self._random.choice(neighbors)
        return replacement.upper() if character.isupper() else replacement


__all__ = [
    "ProductTextAugmenter",
    "TextAugmentationConfig",
]
