"""Typed NER output values."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class NerEntity:
    label: str
    text: str
    start: int
    end: int
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.label.strip() or self.label == "O":
            raise ValueError("entity label must be a non-empty non-O class")
        if not self.text:
            raise ValueError("entity text must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("entity span must be positive and non-empty")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("entity score must be in [0, 1]")


@runtime_checkable
class NerExtractor(Protocol):
    def extract(self, texts: Sequence[str]) -> list[list[NerEntity]]:
        ...


__all__ = ["NerEntity", "NerExtractor"]
