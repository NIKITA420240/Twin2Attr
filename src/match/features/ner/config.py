"""NER architecture metadata stored with its weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_CLASS_NAMES = (
    "O",
    "бренд",
    "тип",
    "модель",
    "материал",
    "цвет",
    "размер",
    "назначение",
    "артикул",
)


@dataclass(frozen=True, slots=True)
class WordNerModelConfig:
    model_name: str
    num_attention_heads: int = 4
    pos_dim: int = 16
    max_subwords_per_word: int = 12
    attention_hidden: int = 32
    sequence_heads: int = 4
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> WordNerModelConfig:
        class_names = values.get("class_names", DEFAULT_CLASS_NAMES)
        return cls(
            model_name=str(values["model_name"]),
            num_attention_heads=int(values.get("num_attention_heads", 4)),
            pos_dim=int(values.get("pos_dim", 16)),
            max_subwords_per_word=int(values.get("max_subwords_per_word", 12)),
            attention_hidden=int(values.get("attention_hidden", 32)),
            sequence_heads=int(values.get("sequence_heads", 4)),
            class_names=tuple(str(value) for value in class_names),
        )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("NER model_name must not be empty")
        if min(
            self.num_attention_heads,
            self.pos_dim,
            self.max_subwords_per_word,
            self.attention_hidden,
            self.sequence_heads,
        ) < 1:
            raise ValueError("NER architecture dimensions must be positive")
        if not self.class_names or self.class_names[0] != "O":
            raise ValueError("NER class_names must start with O")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("NER class_names must be unique")


__all__ = ["DEFAULT_CLASS_NAMES", "WordNerModelConfig"]
