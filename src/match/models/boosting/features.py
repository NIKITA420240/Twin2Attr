"""Deterministic pair features for the fast CatBoost matcher."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...prepare_data import PreparedPair
from ..contracts import PredictionBatch

FEATURE_SCHEMA_VERSION = 1
CATEGORICAL_FEATURES = ("category",)
_DIGIT_RE = re.compile(r"\d")
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _pair_values(
    output: pd.DataFrame,
    name: str,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    """Add symmetric aggregate values for a pair of numeric vectors."""
    output[f"{name}_median"] = ((left + right) / 2).astype(np.float32)
    output[f"{name}_absolute_difference"] = np.abs(left - right).astype(np.float32)


def _minmax_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    largest = np.maximum(left, right)
    return np.divide(
        np.minimum(left, right),
        largest,
        out=np.ones_like(largest),
        where=largest != 0,
    )


def _set_metrics(left: set[str], right: set[str]) -> tuple[float, float, float, float]:
    common = len(left & right)
    union = len(left | right)
    total = len(left) + len(right)
    smallest = min(len(left), len(right))
    return (
        float(common),
        common / union if union else 1.0,
        2 * common / total if total else 1.0,
        common / smallest if smallest else 1.0,
    )


def _overlaps(
    left: pd.Series,
    right: pd.Series,
    kinds: tuple[str, ...],
) -> np.ndarray:
    rows: list[list[float]] = []
    for left_text, right_text in zip(left, right):
        base_left = set(left_text.casefold().split())
        base_right = set(right_text.casefold().split())
        numeric_left = {token for token in base_left if _DIGIT_RE.search(token)}
        numeric_right = {token for token in base_right if _DIGIT_RE.search(token)}
        values: list[float] = []
        for kind in kinds:
            if kind == "numeric":
                left_tokens, right_tokens = numeric_left, numeric_right
            elif kind == "model_code":
                left_tokens = {
                    token for token in numeric_left if _ALPHA_RE.search(token)
                }
                right_tokens = {
                    token for token in numeric_right if _ALPHA_RE.search(token)
                }
            else:
                left_tokens, right_tokens = base_left, base_right
            values.extend(_set_metrics(left_tokens, right_tokens))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _attribute_dict(pair: PreparedPair, *, left: bool) -> dict[str, str]:
    card = pair.left if left else pair.right
    return {
        " ".join(str(key).casefold().split()): " ".join(str(value).casefold().split())
        for key, value in card.attributes
    }


def _attribute_text(pair: PreparedPair, *, left: bool) -> str:
    values = _attribute_dict(pair, left=left)
    return " ".join(f"{key} {value}" for key, value in values.items())


def _name_char_metrics(left: pd.Series, right: pd.Series) -> np.ndarray:
    rows = []
    for left_name, right_name in zip(left, right):
        names: list[set[str]] = []
        for value in (left_name, right_name):
            normalized = "".join(char for char in value.casefold() if char.isalnum())
            names.append(
                {
                    normalized[index : index + 3]
                    for index in range(max(1, len(normalized) - 2))
                }
                if normalized
                else set()
            )
        rows.append(_set_metrics(names[0], names[1]))
    return np.asarray(rows, dtype=np.float32)


def _structured_features(
    output: pd.DataFrame,
    pairs: Sequence[PreparedPair],
) -> None:
    rows = []
    article_keys = ("артикул", "article", "sku", "код товара", "код модели")
    brand_keys = ("бренд", "brand", "производитель")
    for pair in pairs:
        left_dict = _attribute_dict(pair, left=True)
        right_dict = _attribute_dict(pair, left=False)
        left_keys, right_keys = set(left_dict), set(right_dict)
        common_keys = left_keys & right_keys
        exact = sum(left_dict[key] == right_dict[key] for key in common_keys)
        conflicts = len(common_keys) - exact
        left_articles = {
            value
            for key, value in left_dict.items()
            if any(candidate in key for candidate in article_keys)
        }
        right_articles = {
            value
            for key, value in right_dict.items()
            if any(candidate in key for candidate in article_keys)
        }
        left_brand = next(
            (
                value
                for key, value in left_dict.items()
                if any(candidate in key for candidate in brand_keys)
            ),
            "",
        )
        right_brand = next(
            (
                value
                for key, value in right_dict.items()
                if any(candidate in key for candidate in brand_keys)
            ),
            "",
        )
        rows.append(
            (
                *_set_metrics(left_keys, right_keys),
                *_set_metrics(set(left_dict.values()), set(right_dict.values())),
                *_set_metrics(left_articles, right_articles),
                exact,
                conflicts,
                exact / len(common_keys) if common_keys else 1.0,
                conflicts / len(common_keys) if common_keys else 0.0,
                float(bool(left_brand and right_brand and left_brand == right_brand)),
            )
        )
    values = np.asarray(rows, dtype=np.float32)
    for group, prefix in enumerate(
        ("attribute_keys", "attribute_values", "article_tokens")
    ):
        for index, suffix in enumerate(("common", "jaccard", "dice", "containment")):
            output[f"{prefix}_{suffix}"] = values[:, group * 4 + index]
    names = (
        "exact_common_attribute_values",
        "conflicting_common_attribute_values",
        "exact_value_share_among_common_keys",
        "conflict_share_among_common_keys",
        "brand_exact_match",
    )
    for index, name in enumerate(names, start=12):
        output[name] = values[:, index]


def build_pair_features(pairs: Sequence[PreparedPair]) -> pd.DataFrame:
    """Build an order-stable feature frame from structured product pairs."""
    if not pairs:
        raise ValueError("boosting feature input must not be empty")
    text: dict[str, pd.Series] = {
        "name_1": pd.Series([pair.left.name for pair in pairs], dtype="string"),
        "name_2": pd.Series([pair.right.name for pair in pairs], dtype="string"),
        "attributes_1": pd.Series(
            [_attribute_text(pair, left=True) for pair in pairs], dtype="string"
        ),
        "attributes_2": pd.Series(
            [_attribute_text(pair, left=False) for pair in pairs], dtype="string"
        ),
    }
    text["card_1"] = text["name_1"] + " " + text["attributes_1"]
    text["card_2"] = text["name_2"] + " " + text["attributes_2"]

    output = pd.DataFrame(index=range(len(pairs)))
    output["category"] = [
        left if left == right else " <> ".join(sorted((left, right)))
        for left, right in ((pair.left.category, pair.right.category) for pair in pairs)
    ]

    for field in ("name", "attributes", "card"):
        left, right = text[f"{field}_1"], text[f"{field}_2"]
        left_chars = left.str.len().to_numpy(np.float32)
        right_chars = right.str.len().to_numpy(np.float32)
        left_words = left.str.split().str.len().to_numpy(np.float32)
        right_words = right.str.split().str.len().to_numpy(np.float32)
        _pair_values(output, f"{field}_chars", left_chars, right_chars)
        _pair_values(output, f"{field}_words", left_words, right_words)
        if field == "name":
            output["name_chars_minmax_ratio"] = _minmax_ratio(left_chars, right_chars)
            output["name_words_minmax_ratio"] = _minmax_ratio(left_words, right_words)
        elif field == "card":
            output["card_words_minmax_ratio"] = _minmax_ratio(left_words, right_words)

    card_1, card_2 = text["card_1"], text["card_2"]
    digits_1 = card_1.str.count(r"\d").to_numpy(np.float32)
    digits_2 = card_2.str.count(r"\d").to_numpy(np.float32)
    _pair_values(output, "digits_total", digits_1, digits_2)
    output["digits_total_minmax_ratio"] = _minmax_ratio(digits_1, digits_2)
    for digit in "0123456789":
        _pair_values(
            output,
            f"digit_{digit}",
            card_1.str.count(digit).to_numpy(np.float32),
            card_2.str.count(digit).to_numpy(np.float32),
        )

    groups = (
        (("name_words",), text["name_1"], text["name_2"], ("all",)),
        (
            ("attribute_words",),
            text["attributes_1"],
            text["attributes_2"],
            ("all",),
        ),
        (
            ("card_words", "numeric_tokens", "model_code_tokens"),
            card_1,
            card_2,
            ("all", "numeric", "model_code"),
        ),
    )
    suffixes = ("common", "jaccard", "dice", "containment")
    for prefixes, left, right, kinds in groups:
        values = _overlaps(left, right, kinds)
        for group_index, prefix in enumerate(prefixes):
            for metric_index, suffix in enumerate(suffixes):
                output[f"{prefix}_{suffix}"] = values[:, group_index * 4 + metric_index]
    name_1 = text["name_1"].str.casefold()
    name_2 = text["name_2"].str.casefold()
    output["exact_name"] = name_1.eq(name_2).to_numpy(np.int8)
    output["name_contains_other"] = [
        float(bool(left and right) and (left in right or right in left))
        for left, right in zip(name_1, name_2)
    ]
    output["model_code_overlap_present"] = (
        output["model_code_tokens_common"].gt(0).to_numpy(np.int8)
    )
    output["numeric_token_conflict"] = (
        output["numeric_tokens_containment"].eq(0).to_numpy(np.int8)
    )
    output["model_code_token_conflict"] = (
        output["model_code_tokens_containment"].eq(0).to_numpy(np.int8)
    )
    char_metrics = _name_char_metrics(text["name_1"], text["name_2"])
    for index, suffix in enumerate(("common", "jaccard", "dice", "containment")):
        output[f"name_char_trigrams_{suffix}"] = char_metrics[:, index]
    _structured_features(output, pairs)
    return output


class BoostingFeatureBuilder:
    """Model-specific adapter from the shared batch to CatBoost features."""

    schema_version = FEATURE_SCHEMA_VERSION
    categorical_features = CATEGORICAL_FEATURES

    def transform(self, batch: PredictionBatch) -> pd.DataFrame:
        return build_pair_features(batch.prepared_pairs())


__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_SCHEMA_VERSION",
    "BoostingFeatureBuilder",
    "build_pair_features",
]
