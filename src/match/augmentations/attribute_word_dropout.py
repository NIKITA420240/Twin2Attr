"""Dynamic structure-aware text noise for Transformer training."""

from __future__ import annotations

import random
from dataclasses import replace

from ..config import AttributeWordDropoutSettings
from ..prepare_data import PreparedCard, PreparedPair


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


class AttributeWordDropoutAugmenter:
    """Create a fresh noisy pair each time a training row is requested.

    Category and attribute keys retain their schema meaning. Noise is applied
    only to product names and surviving attribute values.
    """

    def __init__(self, settings: AttributeWordDropoutSettings) -> None:
        self.settings = settings
        self._random = random.Random(settings.seed)

    def __call__(self, pair: PreparedPair) -> PreparedPair:
        if self._random.random() >= self.settings.pair_probability:
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
            if self._random.random()
            >= self.settings.attribute_dropout_probability
        )
        return replace(
            card,
            name=self._augment_text(card.name),
            attributes=attributes,
        )

    def _augment_text(self, text: str) -> str:
        words = self._drop_words(text.split())
        if (
            len(words) > 1
            and self._random.random()
            < self.settings.word_shuffle_probability
        ):
            original = list(words)
            self._random.shuffle(words)
            if words == original:
                words = words[1:] + words[:1]
        return "".join(
            self._keyboard_typo(character)
            for character in " ".join(words)
        )

    def _drop_words(self, words: list[str]) -> list[str]:
        if len(words) < 2 or self.settings.word_dropout_probability == 0.0:
            return words
        retained = [
            word
            for word in words
            if self._random.random()
            >= self.settings.word_dropout_probability
        ]
        return retained or [self._random.choice(words)]

    def _keyboard_typo(self, character: str) -> str:
        neighbors = _KEYBOARD_NEIGHBORS.get(character.lower())
        if (
            not neighbors
            or self._random.random()
            >= self.settings.keyboard_typo_probability
        ):
            return character
        replacement = self._random.choice(neighbors)
        return replacement.upper() if character.isupper() else replacement


__all__ = ["AttributeWordDropoutAugmenter"]
