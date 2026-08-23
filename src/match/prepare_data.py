"""Model-independent preparation of structured product-card pairs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

__all__ = [
    "PreparedCard",
    "PreparedPair",
    "prepare_cards",
    "prepare_pairs",
]


AttributePair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class PreparedCard:
    """A product card with fields kept separate for later tokenization."""

    item_id: Any
    name: str
    category: str
    attributes: tuple[AttributePair, ...]


@dataclass(frozen=True, slots=True)
class PreparedPair:
    """A structured product pair in the original matches-row order.

    ``category`` is taken from the left card and is intended for category-level
    evaluation such as macro PR-AUC. ``label`` is ``None`` for inference pairs.
    """

    left: PreparedCard
    right: PreparedCard
    label: int | None
    category: str
    sample_weight: float = 1.0


def _parse_attributes(raw: Any, *, item_id: Any) -> tuple[AttributePair, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
            raise ValueError(f"item {item_id!r} attributes must contain a JSON object") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"item {item_id!r} attributes must contain a JSON object")
    return tuple(
        (str(key), "" if value is None else str(value))
        for key, value in parsed.items()
    )


def prepare_cards(
    items: pl.DataFrame,
    *,
    id_column: str = "id",
    name_column: str = "name",
    category_column: str = "category",
    attributes_column: str = "attributes",
) -> dict[Any, PreparedCard]:
    """Convert an items frame into structured cards keyed by item id.

    The caller is responsible for selecting and, if necessary, normalizing the
    attributes column. Attribute order from the source JSON is preserved so a
    later encoding policy can decide which fields to prioritize.
    """
    if not isinstance(items, pl.DataFrame):
        raise TypeError("items must be a polars.DataFrame")

    required_columns = {
        id_column,
        name_column,
        category_column,
        attributes_column,
    }
    missing_columns = required_columns - set(items.columns)
    if missing_columns:
        raise ValueError(f"items frame is missing columns: {sorted(missing_columns)}")
    if items.get_column(id_column).null_count():
        raise ValueError("items frame contains null ids")
    duplicate_count = items.height - items.get_column(id_column).n_unique()
    if duplicate_count:
        raise ValueError(f"items frame contains {duplicate_count} duplicate ids")
    if items.get_column(category_column).null_count():
        raise ValueError("items frame contains null categories")

    cards: dict[Any, PreparedCard] = {}
    selected = items.select(
        id_column,
        name_column,
        category_column,
        attributes_column,
    )
    for item_id, name, category, raw_attributes in selected.iter_rows():
        cards[item_id] = PreparedCard(
            item_id=item_id,
            name="" if name is None else str(name),
            category=str(category),
            attributes=_parse_attributes(raw_attributes, item_id=item_id),
        )
    return cards


def prepare_pairs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    id_column: str = "id",
    name_column: str = "name",
    category_column: str = "category",
    attributes_column: str = "attributes",
    left_id_column: str = "id1",
    right_id_column: str = "id2",
    label_column: str = "target",
    weight_column: str = "sample_weight",
) -> list[PreparedPair]:
    """Build structured training or inference pairs without tokenization.

    Pair order and duplicate rows are preserved. If ``label_column`` is absent,
    every returned pair has ``label=None``. All referenced item ids must exist.
    """
    if not isinstance(matches, pl.DataFrame):
        raise TypeError("matches must be a polars.DataFrame")
    required_match_columns = {left_id_column, right_id_column}
    missing_columns = required_match_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"matches frame is missing columns: {sorted(missing_columns)}")

    cards = prepare_cards(
        items,
        id_column=id_column,
        name_column=name_column,
        category_column=category_column,
        attributes_column=attributes_column,
    )
    has_labels = label_column in matches.columns
    selected_columns = [left_id_column, right_id_column]
    if has_labels:
        selected_columns.append(label_column)
    has_weights = weight_column in matches.columns
    if has_weights:
        selected_columns.append(weight_column)

    pairs: list[PreparedPair] = []
    for row in matches.select(selected_columns).iter_rows():
        left_id, right_id = row[:2]
        try:
            left = cards[left_id]
            right = cards[right_id]
        except KeyError as error:
            missing_id = error.args[0]
            raise ValueError(f"matches frame references unknown item id: {missing_id!r}") from error

        label: int | None = None
        if has_labels:
            raw_label = row[2]
            if raw_label is None or raw_label not in (0, 1, False, True):
                raise ValueError("target labels must contain only 0 and 1")
            label = int(raw_label)
        raw_weight = row[-1] if has_weights else 1.0
        sample_weight = float(raw_weight)
        if not sample_weight > 0.0:
            raise ValueError("sample weights must be positive")
        pairs.append(
            PreparedPair(
                left=left,
                right=right,
                label=label,
                category=left.category,
                sample_weight=sample_weight,
            )
        )
    return pairs
