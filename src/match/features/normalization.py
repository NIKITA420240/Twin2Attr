"""Feature-pipeline adapter for deterministic attribute normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..normalization import normalize_attributes
from .contracts import PreparedItems


@dataclass(frozen=True, slots=True)
class NormalizationItemEnricher:
    synonyms_path: Path
    unique_attributes_path: Path
    output_column: str = "normalized_attributes"
    n_jobs: int = 1
    chunk_size: int = 5_000

    def enrich(self, items: PreparedItems) -> PreparedItems:
        normalized = normalize_attributes(
            items.frame,
            self.synonyms_path,
            self.unique_attributes_path,
            source_column=items.attributes_column,
            output_column=self.output_column,
            n_jobs=self.n_jobs,
            chunk_size=self.chunk_size,
        )
        return PreparedItems(normalized, self.output_column)


__all__ = ["NormalizationItemEnricher"]
