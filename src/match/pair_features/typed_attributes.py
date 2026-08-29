"""Symmetric, type-aware comparison of structured product attributes."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Final

from ..prepare_data import PreparedCard, PreparedPair

ATTRIBUTE_TYPES: Final = ("CODE", "PHYSICAL", "NUMERIC", "SET", "TEXT")

_CODE_MARKERS: Final = (
    "артикул",
    "oem",
    "оем",
    "part number",
    "партномер",
    "номер детали",
    "номер модели",
    "модель",
    "код модели",
    "код товара",
    "код производителя",
    "sku",
)
_PHYSICAL_MARKERS: Final = (
    "вес",
    "масса",
    "длина",
    "ширина",
    "высота",
    "глубина",
    "толщина",
    "диаметр",
    "объем",
    "объём",
    "мощность",
    "напряжение",
    "частота",
    "давление",
    "температура",
    "скорость",
    "площадь",
    "емкость",
    "ёмкость",
)
_NUMERIC_MARKERS: Final = (
    "количество",
    "год",
    "число",
    "номер",
)
_SET_MARKERS: Final = (
    "цвет",
    "материал",
    "комплектация",
    "совместим",
    "назначение",
    "сезон",
    "покрытие",
    "тип",
    "вид",
    "форма",
    "бренд",
    "производитель",
    "страна",
)
_MODEL_CODE_MARKERS: Final = (
    "артикул",
    "part number",
    "партномер",
    "номер детали",
    "номер модели",
    "модель",
    "код модели",
    "код товара",
    "код производителя",
    "sku",
)
_OEM_MARKERS: Final = ("oem", "оем")
_BRAND_MARKERS: Final = ("бренд", "brand", "производитель")
_NUMBER_RE: Final = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_SET_SEPARATOR_RE: Final = re.compile(r"\s*[;,|]+\s*")
_TOKEN_RE: Final = re.compile(r"\w+", re.UNICODE)


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return " ".join(text.casefold().replace("ё", "е").split())


def _normalize_key(value: Any) -> str:
    return _normalize_text(value).strip(" :;,.-")


def _normalize_code(value: Any) -> str:
    return re.sub(r"[^0-9a-zа-я]", "", _normalize_text(value))


def _split_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = value
    else:
        raw_values = _SET_SEPARATOR_RE.split(str(value))
    normalized = (_normalize_text(item) for item in raw_values)
    return tuple(dict.fromkeys(item for item in normalized if item))


def _numbers(value: Any) -> tuple[float, ...]:
    result: list[float] = []
    for match in _NUMBER_RE.findall("" if value is None else str(value)):
        try:
            number = float(match.replace(",", "."))
        except ValueError:
            continue
        if math.isfinite(number):
            result.append(number)
    return tuple(result)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-8)


def _ratio(left: float, right: float) -> float:
    largest = max(abs(left), abs(right))
    return min(abs(left), abs(right)) / largest if largest else 1.0


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _dice(left: set[str], right: set[str]) -> float:
    total = len(left) + len(right)
    return 2.0 * len(left & right) / total if total else 1.0


def _containment(left: set[str], right: set[str]) -> float:
    smallest = min(len(left), len(right))
    return len(left & right) / smallest if smallest else 1.0


def _char_ngrams(value: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", value).strip()
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


@dataclass(frozen=True, slots=True)
class TypedAttributeOptions:
    """Stable feature options persisted with CatBoost artifacts."""

    enabled: bool = False
    detector: str = "rules"
    preserve_semantic_type_when_missing: bool = True
    symmetric: bool = True
    code: bool = True
    physical: bool = True
    numeric: bool = True
    set: bool = True
    text: bool = True

    def __post_init__(self) -> None:
        if self.detector != "rules":
            raise ValueError("typed attribute detector currently must be 'rules'")
        if not self.preserve_semantic_type_when_missing:
            raise ValueError("typed attributes must preserve semantic type when missing")
        if not self.symmetric:
            raise ValueError("typed attribute features currently must be symmetric")
        if self.enabled and not self.enabled_types:
            raise ValueError("enabled typed attributes require at least one type")

    @property
    def enabled_types(self) -> tuple[str, ...]:
        flags = {
            "CODE": self.code,
            "PHYSICAL": self.physical,
            "NUMERIC": self.numeric,
            "SET": self.set,
            "TEXT": self.text,
        }
        return tuple(name for name in ATTRIBUTE_TYPES if flags[name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "detector": self.detector,
            "preserve_semantic_type_when_missing": self.preserve_semantic_type_when_missing,
            "symmetric": self.symmetric,
            "types": {
                "code": self.code,
                "physical": self.physical,
                "numeric": self.numeric,
                "set": self.set,
                "text": self.text,
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> TypedAttributeOptions:
        source = dict(values or {})
        types = source.get("types", {})
        if not isinstance(types, Mapping):
            raise ValueError("typed attribute option 'types' must be an object")
        return cls(
            enabled=bool(source.get("enabled", False)),
            detector=str(source.get("detector", "rules")),
            preserve_semantic_type_when_missing=bool(
                source.get("preserve_semantic_type_when_missing", True)
            ),
            symmetric=bool(source.get("symmetric", True)),
            code=bool(types.get("code", True)),
            physical=bool(types.get("physical", True)),
            numeric=bool(types.get("numeric", True)),
            set=bool(types.get("set", True)),
            text=bool(types.get("text", True)),
        )


@dataclass(frozen=True, slots=True)
class TypedAttributeComparison:
    key: str
    semantic_type: str
    both_present: bool
    one_missing: bool
    exact: float
    similarity: float
    conflict: float
    metrics: Mapping[str, float]


def _card_attributes(card: PreparedCard) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for raw_key, raw_value in card.attributes:
        key = _normalize_key(raw_key)
        value = _normalize_text(raw_value)
        if key and value and value not in grouped[key]:
            grouped[key].append(value)
    return {key: ";".join(values) for key, values in grouped.items()}


class TypedAttributeComparator:
    """Compare aligned attributes using deterministic type-specific metrics."""

    def __init__(self, options: TypedAttributeOptions | None = None) -> None:
        self.options = options or TypedAttributeOptions(enabled=True)

    @staticmethod
    def detect_type(key: str, left: str | None, right: str | None) -> str:
        normalized_key = _normalize_key(key)
        if any(marker in normalized_key for marker in _CODE_MARKERS):
            return "CODE"
        if any(marker in normalized_key for marker in _PHYSICAL_MARKERS):
            return "PHYSICAL"
        if any(marker in normalized_key for marker in _NUMERIC_MARKERS):
            return "NUMERIC"
        if any(marker in normalized_key for marker in _SET_MARKERS):
            return "SET"
        values = tuple(value for value in (left, right) if value)
        if any(re.search(r"[;|]", value) for value in values):
            return "SET"
        return "TEXT"

    @staticmethod
    def _code_metrics(left: str, right: str) -> dict[str, float]:
        left_values = {_normalize_code(value) for value in _split_values(left)} - {""}
        right_values = {_normalize_code(value) for value in _split_values(right)} - {""}
        pairs = [(a, b) for a in left_values for b in right_values]
        similarities = [SequenceMatcher(None, a, b).ratio() for a, b in pairs]
        left_digits = {"".join(re.findall(r"\d", value)) for value in left_values}
        right_digits = {"".join(re.findall(r"\d", value)) for value in right_values}
        digit_exact = float(bool((left_digits & right_digits) - {""}))
        exact = float(bool(left_values & right_values))
        similarity = max(similarities, default=0.0)
        return {
            "exact": exact,
            "set_exact": float(left_values == right_values and bool(left_values)),
            "similarity": similarity,
            "digit_exact": digit_exact,
            "jaccard": _jaccard(left_values, right_values),
            "containment": _containment(left_values, right_values),
        }

    @staticmethod
    def _numeric_metrics(left: str, right: str) -> dict[str, float]:
        left_values = _numbers(left)
        right_values = _numbers(right)
        pairs = [(a, b) for a in left_values for b in right_values]
        if not pairs:
            return {"exact": 0.0, "similarity": 0.0, "relative_difference": 1.0, "ratio": 0.0}
        best_left, best_right = min(pairs, key=lambda pair: _relative_difference(*pair))
        relative_difference = _relative_difference(best_left, best_right)
        ratio = _ratio(best_left, best_right)
        return {
            "exact": float(relative_difference <= 1e-8),
            "similarity": ratio,
            "relative_difference": relative_difference,
            "ratio": ratio,
        }

    @staticmethod
    def _set_metrics(left: str, right: str) -> dict[str, float]:
        left_values = set(_split_values(left))
        right_values = set(_split_values(right))
        jaccard = _jaccard(left_values, right_values)
        return {
            "exact": float(left_values == right_values and bool(left_values)),
            "similarity": jaccard,
            "jaccard": jaccard,
            "dice": _dice(left_values, right_values),
            "containment": _containment(left_values, right_values),
        }

    @staticmethod
    def _text_metrics(left: str, right: str) -> dict[str, float]:
        left_text = _normalize_text(left)
        right_text = _normalize_text(right)
        token_jaccard = _jaccard(set(_TOKEN_RE.findall(left_text)), set(_TOKEN_RE.findall(right_text)))
        char_jaccard = _jaccard(_char_ngrams(left_text), _char_ngrams(right_text))
        similarity = (token_jaccard + char_jaccard) / 2.0
        return {
            "exact": float(left_text == right_text and bool(left_text)),
            "similarity": similarity,
            "token_jaccard": token_jaccard,
            "char_jaccard": char_jaccard,
        }

    def compare_pair(self, pair: PreparedPair) -> tuple[TypedAttributeComparison, ...]:
        left_attributes = _card_attributes(pair.left)
        right_attributes = _card_attributes(pair.right)
        result: list[TypedAttributeComparison] = []
        for key in sorted(set(left_attributes) | set(right_attributes)):
            left = left_attributes.get(key)
            right = right_attributes.get(key)
            semantic_type = self.detect_type(key, left, right)
            if semantic_type not in self.options.enabled_types:
                continue
            both_present = left is not None and right is not None
            if not both_present:
                result.append(
                    TypedAttributeComparison(
                        key=key,
                        semantic_type=semantic_type,
                        both_present=False,
                        one_missing=True,
                        exact=0.0,
                        similarity=0.0,
                        conflict=0.0,
                        metrics={},
                    )
                )
                continue
            if semantic_type == "CODE":
                metrics = self._code_metrics(left, right)
            elif semantic_type in {"PHYSICAL", "NUMERIC"}:
                metrics = self._numeric_metrics(left, right)
            elif semantic_type == "SET":
                metrics = self._set_metrics(left, right)
            else:
                metrics = self._text_metrics(left, right)
            exact = float(metrics["exact"])
            similarity = float(metrics["similarity"])
            conflict = float(
                not exact
                and (
                    similarity < 0.5
                    or (
                        semantic_type == "CODE"
                        and metrics.get("digit_exact", 0.0) == 0.0
                    )
                )
            )
            result.append(
                TypedAttributeComparison(
                    key=key,
                    semantic_type=semantic_type,
                    both_present=True,
                    one_missing=False,
                    exact=exact,
                    similarity=similarity,
                    conflict=conflict,
                    metrics=metrics,
                )
            )
        return tuple(result)


def _critical_group(key: str) -> str | None:
    if any(marker in key for marker in _OEM_MARKERS):
        return "oem"
    if any(marker in key for marker in _MODEL_CODE_MARKERS):
        return "model_code"
    if any(marker in key for marker in _BRAND_MARKERS):
        return "brand"
    return None


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_typed_attribute_features(
    comparisons: Sequence[TypedAttributeComparison],
    *,
    enabled_types: Sequence[str] = ATTRIBUTE_TYPES,
) -> dict[str, float]:
    """Flatten variable-length comparisons into a stable symmetric schema."""
    result: dict[str, float] = {
        "typed_attribute_comparisons": float(len(comparisons)),
        "typed_attribute_both_present": float(sum(item.both_present for item in comparisons)),
        "typed_attribute_one_missing": float(sum(item.one_missing for item in comparisons)),
        "typed_attribute_exact_matches": float(sum(item.exact for item in comparisons)),
        "typed_attribute_strong_conflicts": float(sum(item.conflict for item in comparisons)),
    }
    for semantic_type in enabled_types:
        prefix = f"typed_{semantic_type.casefold()}"
        selected = [item for item in comparisons if item.semantic_type == semantic_type]
        present = [item for item in selected if item.both_present]
        similarities = [item.similarity for item in present]
        result[f"{prefix}_comparisons"] = float(len(selected))
        result[f"{prefix}_both_present"] = float(len(present))
        result[f"{prefix}_one_missing"] = float(sum(item.one_missing for item in selected))
        result[f"{prefix}_exact_matches"] = float(sum(item.exact for item in present))
        result[f"{prefix}_mean_similarity"] = _mean(similarities)
        result[f"{prefix}_min_similarity"] = min(similarities, default=0.0)
        result[f"{prefix}_strong_conflicts"] = float(sum(item.conflict for item in present))

    for group in ("model_code", "oem", "brand"):
        selected = [item for item in comparisons if _critical_group(item.key) == group]
        present = [item for item in selected if item.both_present]
        result[f"typed_{group}_both_present"] = float(bool(present))
        result[f"typed_{group}_exact_match"] = float(any(item.exact for item in present))
        result[f"typed_{group}_conflict"] = float(any(item.conflict for item in present))
        result[f"typed_{group}_one_missing"] = float(any(item.one_missing for item in selected))
    return result


__all__ = [
    "ATTRIBUTE_TYPES",
    "TypedAttributeComparison",
    "TypedAttributeComparator",
    "TypedAttributeOptions",
    "aggregate_typed_attribute_features",
]
