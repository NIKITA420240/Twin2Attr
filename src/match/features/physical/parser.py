"""Algorithmic parser for values with measurement units in product names."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from joblib import Parallel, delayed

from .config import (
    DEFAULT_UNIT_ATTRIBUTES,
    DIMENSION_KEYWORDS,
    DIMENSION_UNITS,
    physical_unit_aliases,
)


_WORKER_PARSER: PhysicalAttributeParser | None = None


def _worker_parser() -> PhysicalAttributeParser:
    global _WORKER_PARSER
    if _WORKER_PARSER is None:
        _WORKER_PARSER = PhysicalAttributeParser(n_jobs=1)
    return _WORKER_PARSER


def _parse_chunk(values: list[str]) -> list[dict[str, str]]:
    return _worker_parser().parse_batch(values)


@dataclass(slots=True)
class PhysicalAttributeParser:
    """Extract one- and multi-dimensional physical values from names."""

    n_jobs: int = 1
    chunk_size: int = 10_000
    _aliases: dict[str, str] = field(init=False, repr=False)
    _multidimensional_pattern: re.Pattern[str] = field(init=False, repr=False)
    _physical_pattern: re.Pattern[str] = field(init=False, repr=False)
    _dimension_keyword_pattern: re.Pattern[str] = field(init=False, repr=False)
    _diameter_short_pattern: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_jobs == 0 or self.chunk_size < 1:
            raise ValueError("physical n_jobs must not be zero and chunk_size positive")
        self._aliases = physical_unit_aliases()
        dimensions = self._unit_pattern(only_dimensions=True)
        units = self._unit_pattern(only_dimensions=False)
        number = r"\d+(?:[.,]\d+)?"
        self._multidimensional_pattern = re.compile(
            rf"(?P<a>{number})\s*[xх×*]\s*(?P<b>{number})"
            rf"(?:\s*[xх×*]\s*(?P<c>{number}))?\s*"
            rf"(?P<unit>{dimensions})(?![a-zа-яё0-9])",
            re.IGNORECASE,
        )
        self._physical_pattern = re.compile(
            rf"(?P<number>{number})\s*(?P<unit>{units})"
            rf"(?![a-zа-яё0-9])",
            re.IGNORECASE,
        )
        keywords = "|".join(
            map(re.escape, sorted(DIMENSION_KEYWORDS, key=len, reverse=True))
        )
        self._dimension_keyword_pattern = re.compile(
            rf"(?<!\w)({keywords})(?!\w)",
            re.IGNORECASE,
        )
        self._diameter_short_pattern = re.compile(
            r"(?:\bdia\s*|\bd\s*|[ø⌀]\s*)$",
            re.IGNORECASE,
        )

    def _unit_pattern(self, *, only_dimensions: bool) -> str:
        aliases = (
            alias
            for alias, unit in self._aliases.items()
            if not only_dimensions or unit in DIMENSION_UNITS
        )
        return "|".join(map(re.escape, sorted(aliases, key=len, reverse=True)))

    def _dimension_attribute(self, text: str, number_start: int) -> str:
        context = text[max(0, number_start - 40) : number_start]
        separator = max((context.rfind(char) for char in ",;/|()"), default=-1)
        local_context = context[separator + 1 :]
        matches = list(self._dimension_keyword_pattern.finditer(local_context))
        if matches:
            return matches[-1].group(1).casefold()
        if self._diameter_short_pattern.search(local_context):
            return "диаметр"
        return "длина"

    def parse(self, name: str) -> dict[str, str]:
        text = str(name).casefold()
        result: dict[str, str] = {}
        multidimensional_spans: list[tuple[int, int]] = []
        for match in self._multidimensional_pattern.finditer(text):
            unit = self._aliases[match.group("unit").casefold()]
            multidimensional_spans.append(match.span())
            for attribute, value in zip(
                ("длина", "ширина", "высота"),
                match.group("a", "b", "c"),
            ):
                if value is not None:
                    result.setdefault(
                        f"{attribute}, {unit}", value.replace(",", ".")
                    )

        for match in self._physical_pattern.finditer(text):
            if any(start <= match.start() < end for start, end in multidimensional_spans):
                continue
            raw_number = match.group("number")
            unit = self._aliases[match.group("unit").casefold()]
            if (
                unit == "г"
                and match.end("number") == match.start("unit")
                and raw_number.isdigit()
                and len(raw_number) == 4
                and 1900 <= int(raw_number) <= 2100
            ):
                continue
            attribute = (
                self._dimension_attribute(text, match.start("number"))
                if unit in DIMENSION_UNITS
                else DEFAULT_UNIT_ATTRIBUTES.get(unit)
            )
            if attribute is not None:
                result.setdefault(
                    f"{attribute}, {unit}", raw_number.replace(",", ".")
                )
        return result

    def parse_batch(self, names: Sequence[str]) -> list[dict[str, str]]:
        values = ["" if name is None else str(name) for name in names]
        if not values:
            return []
        if self.n_jobs == 1 or len(values) <= self.chunk_size:
            return [self.parse(value) for value in values]
        chunks = [
            values[index : index + self.chunk_size]
            for index in range(0, len(values), self.chunk_size)
        ]
        parts = Parallel(
            n_jobs=self.n_jobs,
            backend="loky",
            pre_dispatch=self.n_jobs,
            batch_size=1,
        )(delayed(_parse_chunk)(chunk) for chunk in chunks)
        return [item for part in parts for item in part]


__all__ = ["PhysicalAttributeParser"]
