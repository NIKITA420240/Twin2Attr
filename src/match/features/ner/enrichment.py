"""Polars integration and safe merging of NER-derived attributes."""

from __future__ import annotations

from dataclasses import dataclass
import polars as pl
from loguru import logger

from ..attributes import (
    merge_missing_only,
    parse_attribute_mapping,
    serialize_attribute_mapping,
)
from ..contracts import PreparedItems
from .entities import NerEntity, NerExtractor


def _entities_to_attributes(entities: list[NerEntity]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for entity in entities:
        values = grouped.setdefault(entity.label, [])
        if entity.text not in values:
            values.append(entity.text)
    return {label: ", ".join(values) for label, values in grouped.items()}


@dataclass(frozen=True, slots=True)
class NerItemEnricher:
    extractor: NerExtractor
    source_column: str = "name"
    output_column: str = "ner_attributes"
    enriched_column: str = "enriched_attributes"
    merge_policy: str = "missing_only"

    def __post_init__(self) -> None:
        if self.merge_policy != "missing_only":
            raise ValueError("only the missing_only NER merge policy is supported")
        if not self.source_column.strip():
            raise ValueError("source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("NER output columns must not be empty")

    def enrich(self, items: PreparedItems) -> PreparedItems:
        frame = items.frame
        required = {self.source_column, items.attributes_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"items is missing NER columns: {sorted(missing)}")
        names = [
            "" if value is None else str(value)
            for value in frame.get_column(self.source_column)
        ]
        extracted = self.extractor.extract(names)
        if len(extracted) != frame.height:
            raise RuntimeError(
                "NER extractor returned a different number of rows: "
                f"{len(extracted)} != {frame.height}"
            )
        ner_values: list[str] = []
        enriched_values: list[str] = []
        for row, (raw_attributes, entities) in enumerate(
            zip(frame.get_column(items.attributes_column), extracted)
        ):
            ner_attributes = _entities_to_attributes(entities)
            source_attributes = parse_attribute_mapping(raw_attributes, row=row)
            ner_values.append(serialize_attribute_mapping(ner_attributes))
            enriched_values.append(
                serialize_attribute_mapping(
                    merge_missing_only(ner_attributes, source_attributes)
                )
            )
        enriched = frame.with_columns(
            pl.Series(self.output_column, ner_values, dtype=pl.String),
            pl.Series(self.enriched_column, enriched_values, dtype=pl.String),
        )
        logger.info(
            "Enriched items with NER: rows={}, source={}, output={}",
            frame.height,
            self.source_column,
            self.enriched_column,
        )
        return PreparedItems(enriched, self.enriched_column)


__all__ = ["NerItemEnricher"]
