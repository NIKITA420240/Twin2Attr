"""Polars adapter for physical attributes extracted from product names."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from loguru import logger

from ..attributes import (
    merge_missing_only,
    parse_attribute_mapping,
    serialize_attribute_mapping,
)
from ..contracts import PreparedItems
from .normalizer import PhysicalUnitNormalizer
from .parser import PhysicalAttributeParser


@dataclass(frozen=True, slots=True)
class PhysicalItemEnricher:
    parser: PhysicalAttributeParser
    source_column: str = "name"
    output_column: str = "physical_attributes"
    enriched_column: str = "feature_attributes"
    merge_policy: str = "missing_only"
    normalize_units: bool = True
    normalizer: PhysicalUnitNormalizer = field(
        default_factory=PhysicalUnitNormalizer,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.merge_policy != "missing_only":
            raise ValueError("only the missing_only physical merge policy is supported")
        if not self.source_column.strip():
            raise ValueError("physical source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("physical output columns must not be empty")

    def enrich(self, items: PreparedItems) -> PreparedItems:
        frame = items.frame
        missing = {self.source_column, items.attributes_column} - set(frame.columns)
        if missing:
            raise ValueError(f"items is missing physical columns: {sorted(missing)}")
        names = [
            "" if value is None else str(value)
            for value in frame.get_column(self.source_column)
        ]
        extracted = self.parser.parse_batch(names)
        if len(extracted) != frame.height:
            raise RuntimeError(
                "physical parser returned a different number of rows: "
                f"{len(extracted)} != {frame.height}"
            )

        physical_values: list[str] = []
        enriched_values: list[str] = []
        for row, (raw_attributes, parsed) in enumerate(
            zip(frame.get_column(items.attributes_column), extracted)
        ):
            physical = (
                self.normalizer.normalize(parsed) if self.normalize_units else parsed
            )
            source = parse_attribute_mapping(raw_attributes, row=row)
            physical_values.append(serialize_attribute_mapping(physical))
            enriched_values.append(
                serialize_attribute_mapping(merge_missing_only(physical, source))
            )
        result = frame.with_columns(
            pl.Series(self.output_column, physical_values, dtype=pl.String),
            pl.Series(self.enriched_column, enriched_values, dtype=pl.String),
        )
        logger.info(
            "Enriched items with physical attributes: rows={}, output={}",
            frame.height,
            self.enriched_column,
        )
        return PreparedItems(result, self.enriched_column)


__all__ = ["PhysicalItemEnricher"]
