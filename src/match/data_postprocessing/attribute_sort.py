"""Order prepared-card attributes using an analysis priority table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl

from ..config import AppConfig
from ..prepare_data import PreparedCard, PreparedPair

_REQUIRED_COLUMNS = {"scope", "category", "attribute", "priority"}


@dataclass(frozen=True, slots=True)
class AttributePriorityTable:
    global_priorities: dict[str, int]
    category_priorities: dict[tuple[str, str], int]

    @classmethod
    def load(cls, path: str | Path) -> AttributePriorityTable:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"Attribute priorities file does not exist: {source}. "
                "Run `python run.py analyze` first."
            )
        frame = pl.read_parquet(source)
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                "attribute priorities parquet is missing columns: "
                f"{sorted(missing)}"
            )

        global_priorities: dict[str, int] = {}
        category_priorities: dict[tuple[str, str], int] = {}
        for row in frame.select(_REQUIRED_COLUMNS).iter_rows(named=True):
            scope = str(row["scope"])
            attribute = str(row["attribute"]).strip()
            priority = int(row["priority"])
            if not attribute or priority < 1:
                raise ValueError(
                    "attribute priorities require non-empty attributes and "
                    "positive priorities"
                )
            if scope == "global":
                if attribute in global_priorities:
                    raise ValueError(
                        f"duplicate global attribute priority: {attribute!r}"
                    )
                global_priorities[attribute] = priority
                continue
            if scope != "category" or row["category"] is None:
                raise ValueError(
                    "attribute priorities scope must be 'global' or a category "
                    "row with a non-null category"
                )
            key = (str(row["category"]), attribute)
            if key in category_priorities:
                raise ValueError(
                    "duplicate category attribute priority: "
                    f"category={key[0]!r}, attribute={attribute!r}"
                )
            category_priorities[key] = priority
        if not global_priorities:
            raise ValueError("attribute priorities contain no global rows")
        return cls(global_priorities, category_priorities)

    def _priority(self, category: str, attribute: str) -> int | None:
        return self.category_priorities.get(
            (category, attribute),
            self.global_priorities.get(attribute),
        )

    def sort_card(self, card: PreparedCard, *, category: str) -> PreparedCard:
        indexed = list(enumerate(card.attributes))

        def key(value: tuple[int, tuple[str, str]]) -> tuple[int, int, int]:
            source_index, (attribute, _) = value
            priority = self._priority(category, attribute)
            if priority is None:
                return (1, 0, source_index)
            return (0, priority, source_index)

        ordered = tuple(attribute for _, attribute in sorted(indexed, key=key))
        return replace(card, attributes=ordered)

    def sort_pairs(
        self,
        pairs: Sequence[PreparedPair],
    ) -> tuple[PreparedPair, ...]:
        return tuple(
            replace(
                pair,
                left=self.sort_card(pair.left, category=pair.category),
                right=self.sort_card(pair.right, category=pair.category),
                preserve_attribute_order=True,
                skip_oversized_attributes=True,
            )
            for pair in pairs
        )


def apply_pair_postprocessing(
    pairs: Sequence[PreparedPair],
    config: AppConfig,
    *,
    model_name: str | None,
) -> tuple[PreparedPair, ...]:
    if model_name is None:
        return tuple(pairs)
    if model_name != "attribute_sort":
        raise ValueError(f"Unsupported data postprocessing model: {model_name!r}")
    table = AttributePriorityTable.load(
        config.data_postprocessing_models.attribute_sort.priorities_path
    )
    return table.sort_pairs(pairs)


def apply_manifest_postprocessing(
    pairs: Sequence[PreparedPair],
    solution: Mapping[str, Any],
    solution_root: Path,
) -> tuple[PreparedPair, ...]:
    model_name = solution.get("data_postprocessing_model")
    if model_name is None:
        return tuple(pairs)
    if str(model_name) != "attribute_sort":
        raise ValueError(f"Unsupported data postprocessing model: {model_name!r}")
    models = solution.get("data_postprocessing_models")
    if not isinstance(models, Mapping):
        raise TypeError(
            "solution data_postprocessing_models must contain a mapping"
        )
    values = models.get("attribute_sort")
    if not isinstance(values, Mapping):
        raise TypeError(
            "solution data_postprocessing_models.attribute_sort must be a mapping"
        )
    value = values.get("priorities_path")
    if value is None or not str(value).strip():
        raise ValueError("attribute_sort.priorities_path must contain a path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = solution_root / path
    return AttributePriorityTable.load(path).sort_pairs(pairs)


__all__ = [
    "AttributePriorityTable",
    "apply_manifest_postprocessing",
    "apply_pair_postprocessing",
]
