"""Composition of independent item-level feature providers."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ItemEnricher, PreparedItems


@dataclass(frozen=True, slots=True)
class FeaturePipeline:
    enrichers: tuple[ItemEnricher, ...] = ()

    def enrich(self, items: PreparedItems) -> PreparedItems:
        result = items
        for enricher in self.enrichers:
            result = enricher.enrich(result)
        return result


__all__ = ["FeaturePipeline"]
