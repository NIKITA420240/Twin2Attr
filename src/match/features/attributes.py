"""Serialization and precedence rules shared by item enrichers."""

from __future__ import annotations

from typing import Any, Mapping

import orjson


def parse_attribute_mapping(raw: Any, *, row: int) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    try:
        parsed = orjson.loads(raw)
    except (orjson.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"attributes row {row} must contain a JSON object"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(f"attributes row {row} must contain a JSON object")
    return {str(key): value for key, value in parsed.items()}


def merge_missing_only(
    extracted: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge extracted values without replacing source card attributes."""
    source_keys = {str(key).strip().casefold() for key in source}
    return {
        **{
            key: value
            for key, value in extracted.items()
            if key.strip().casefold() not in source_keys
        },
        **source,
    }


def serialize_attribute_mapping(value: Mapping[str, Any]) -> str:
    return orjson.dumps(value).decode("utf-8")


__all__ = [
    "merge_missing_only",
    "parse_attribute_mapping",
    "serialize_attribute_mapping",
]
