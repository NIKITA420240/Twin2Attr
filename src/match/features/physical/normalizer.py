"""Canonical unit conversion for attributes extracted from names."""

from __future__ import annotations

from typing import Any, Mapping

from ...normalization import normalize_physical_attributes


class PhysicalUnitNormalizer:
    """Object adapter around the shared notebook-derived normalizer."""

    def normalize(self, attributes: Mapping[Any, Any]) -> dict[str, str]:
        return normalize_physical_attributes(attributes)


__all__ = ["PhysicalUnitNormalizer"]
