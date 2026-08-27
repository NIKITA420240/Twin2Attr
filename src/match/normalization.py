"""Normalize product-card attributes in Polars data frames."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Any

import orjson
import polars as pl
import pymorphy3
from joblib import Parallel, delayed
from loguru import logger
from pint import UnitRegistry
from pint.errors import PintError

from .normalization_rules import (
    DIMENSION_KEYWORDS,
    KEYWORD_POSSIBLE_UNITS,
    KEYWORD_STANDARD_UNIT,
    RU_TO_PINT,
    SKIP_KEYWORDS,
    UNIT_ALIASES,
)

__all__ = ["normalize_attributes", "normalize_physical_attributes"]


_SPACE_PATTERN = re.compile(r"\s+")
_RUSSIAN_WORD_PATTERN = re.compile(r"[а-яё]+", flags=re.IGNORECASE)
_DIMENSION_PATTERN = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s*[xх×*]\s*(\d+(?:[.,]\d+)?)"
    r"(?:\s*[xх×*]\s*(\d+(?:[.,]\d+)?))?\s*$",
    flags=re.IGNORECASE,
)

_ALIAS_TO_UNIT = {
    alias.lower().strip(): canonical_unit
    for canonical_unit, aliases in UNIT_ALIASES.items()
    for alias in aliases
}
_UNIT_ALIAS_LISTS = {
    canonical_unit: sorted(
        UNIT_ALIASES.get(canonical_unit, {canonical_unit}),
        key=len,
        reverse=True,
    )
    for units in KEYWORD_POSSIBLE_UNITS.values()
    for canonical_unit in units
}
_KEYWORD_PATTERNS = {
    keyword: re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
    for keyword in KEYWORD_POSSIBLE_UNITS
}
_KEYWORD_UNIT_PATTERNS: dict[str, re.Pattern[str]] = {}
_KEYWORD_ALIAS_TO_UNIT: dict[str, dict[str, str]] = {}

for _keyword, _allowed_units in KEYWORD_POSSIBLE_UNITS.items():
    _alias_map = {
        alias.lower().strip(): canonical_unit
        for canonical_unit in _allowed_units
        for alias in _UNIT_ALIAS_LISTS[canonical_unit]
    }
    if _alias_map:
        _alternatives = "|".join(
            map(re.escape, sorted(_alias_map, key=len, reverse=True))
        )
        _KEYWORD_UNIT_PATTERNS[_keyword] = re.compile(
            rf"(?<!\w)({_alternatives})(?!\w)",
            re.IGNORECASE,
        )
    _KEYWORD_ALIAS_TO_UNIT[_keyword] = _alias_map

_UNIT_CLEANUP_PATTERNS = {}
for _canonical_unit, _aliases in _UNIT_ALIAS_LISTS.items():
    _alternatives = "|".join(map(re.escape, sorted(_aliases, key=len, reverse=True)))
    _UNIT_CLEANUP_PATTERNS[_canonical_unit] = (
        re.compile(rf"\(\s*(?:{_alternatives})\s*\)", re.IGNORECASE),
        re.compile(rf"(?<!\w)(?:{_alternatives})(?!\w)", re.IGNORECASE),
    )

_EXPLICIT_ALIASES = "|".join(
    map(re.escape, sorted(_ALIAS_TO_UNIT, key=len, reverse=True))
)
_EXPLICIT_COMMA_PATTERN = re.compile(
    rf",\s*({_EXPLICIT_ALIASES})(?=$|\s)", re.IGNORECASE
)
_EXPLICIT_BRACKET_PATTERN = re.compile(
    rf"\(\s*({_EXPLICIT_ALIASES})\s*\)", re.IGNORECASE
)
_STANDARD_KEYWORD_PATTERNS = [
    (
        keyword,
        re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE),
    )
    for keyword in sorted(KEYWORD_STANDARD_UNIT, key=len, reverse=True)
]

_UREG: UnitRegistry | None = None
_MORPH: pymorphy3.MorphAnalyzer | None = None
_WORKER_ATTRIBUTE_NAME_MAP: Mapping[str, str] | None = None


def _get_unit_registry() -> UnitRegistry:
    global _UREG
    if _UREG is None:
        _UREG = UnitRegistry()
    return _UREG


def _get_morphology() -> pymorphy3.MorphAnalyzer:
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


def _contains_keyword(text: str, keyword: str) -> bool:
    return (
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE)
        is not None
    )


def _clean_key_unit(key: str, canonical_unit: str) -> str:
    bracket_pattern, standalone_pattern = _UNIT_CLEANUP_PATTERNS[canonical_unit]
    key = bracket_pattern.sub("", key)
    key = standalone_pattern.sub("", key, count=1)
    return _SPACE_PATTERN.sub(" ", key).strip(" ,()")


def _normalize_physical_unit_pair(old_key: Any, old_value: Any) -> tuple[str, str]:
    key = str(old_key).strip()
    value = str(old_value).strip()
    key_lower = key.lower()

    if any(word in key_lower for word in SKIP_KEYWORDS):
        return key, value

    for keyword, keyword_pattern in _KEYWORD_PATTERNS.items():
        if keyword_pattern.search(key) is None:
            continue
        unit_pattern = _KEYWORD_UNIT_PATTERNS.get(keyword)
        if unit_pattern is None:
            continue

        value_match = unit_pattern.search(value)
        if value_match is not None:
            alias = value_match.group(1).lower().strip()
            canonical_unit = _KEYWORD_ALIAS_TO_UNIT[keyword][alias]
            new_value = (
                value[: value_match.start()] + value[value_match.end() :]
            ).strip()
            return (
                f"{_clean_key_unit(key, canonical_unit)}, {canonical_unit}",
                new_value,
            )

        key_match = unit_pattern.search(key)
        if key_match is not None:
            alias = key_match.group(1).lower().strip()
            canonical_unit = _KEYWORD_ALIAS_TO_UNIT[keyword][alias]
            return f"{_clean_key_unit(key, canonical_unit)}, {canonical_unit}", value

    explicit_match = _EXPLICIT_COMMA_PATTERN.search(
        key
    ) or _EXPLICIT_BRACKET_PATTERN.search(key)
    if explicit_match is None:
        return key, value

    alias = explicit_match.group(1).lower().strip()
    canonical_unit = _ALIAS_TO_UNIT[alias]
    clean_key = key[: explicit_match.start()].strip()
    value_unit_pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
    new_value = value_unit_pattern.sub("", value, count=1).strip()
    return f"{clean_key}, {canonical_unit}", new_value


@cache
def _conversion(from_unit: str, to_unit: str) -> tuple[float, float] | None:
    from_pint = RU_TO_PINT.get(from_unit)
    to_pint = RU_TO_PINT.get(to_unit)
    if from_pint is None or to_pint is None:
        return None
    ureg = _get_unit_registry()
    try:
        y0 = ureg.Quantity(0, from_pint).to(to_pint).magnitude
        y1 = ureg.Quantity(1, from_pint).to(to_pint).magnitude
    except (PintError, TypeError, ValueError):
        return None
    return float(y1 - y0), float(y0)


def _convert_physical_value(value: str, from_unit: str, to_unit: str) -> float | None:
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if from_unit == to_unit:
        return number
    conversion = _conversion(from_unit, to_unit)
    if conversion is None:
        return None
    scale, offset = conversion
    converted = number * scale + offset
    return converted if math.isfinite(converted) else None


def _format_number(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=1e-12, abs_tol=1e-12):
        return str(rounded)
    return format(value, ".12g")


def _standardize_physical_unit_pair(key: str, value: str) -> tuple[str, str]:
    if "," not in key:
        return key, value
    attribute, unit = map(str.strip, key.rsplit(",", 1))
    matched_keyword = next(
        (
            keyword
            for keyword, pattern in _STANDARD_KEYWORD_PATTERNS
            if pattern.search(attribute)
        ),
        None,
    )
    if matched_keyword is None:
        return key, value
    target_unit = KEYWORD_STANDARD_UNIT[matched_keyword]
    converted = _convert_physical_value(value, unit.lower(), target_unit)
    if converted is None:
        return key, value
    return f"{attribute}, {target_unit}", _format_number(converted)


def _normalize_physical_attributes(attributes: Mapping[Any, Any]) -> dict[str, str]:
    result = {}
    for old_key, old_value in attributes.items():
        key, value = _normalize_physical_unit_pair(old_key, old_value)
        key, value = _standardize_physical_unit_pair(key, value)
        result[key] = value
    return result


def _dimension_suffix(key: str) -> str | None:
    match = re.search(r"(?<!\w)размер(?:ы)?\s+([а-яёa-z0-9_-]+)", key, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _normalize_multidimensional_attributes(
    attributes: Mapping[Any, Any],
) -> dict[str, str]:
    result = {}
    for old_key, old_value in attributes.items():
        key = str(old_key).strip()
        value = str(old_value).strip()
        is_dimension = any(_contains_keyword(key, word) for word in DIMENSION_KEYWORDS)
        if len(value) > 20 or not is_dimension:
            result[key] = value
            continue

        normalized_key, normalized_value = _normalize_physical_unit_pair(key, value)
        if "," not in normalized_key:
            result[key] = value
            continue
        _, unit = map(str.strip, normalized_key.rsplit(",", 1))
        match = _DIMENSION_PATTERN.fullmatch(normalized_value)
        if match is None:
            result[key] = value
            continue

        suffix = _dimension_suffix(key)
        for name, dimension in zip(("длина", "ширина", "высота"), match.groups()):
            if dimension is None:
                continue
            attribute = f"{name} {suffix}" if suffix else name
            new_key, new_value = _standardize_physical_unit_pair(
                f"{attribute}, {unit}", dimension
            )
            result[new_key] = new_value
    return result


def normalize_physical_attributes(
    attributes: Mapping[Any, Any],
) -> dict[str, str]:
    """Normalize physical attribute names, units and numeric values.

    This is the reusable, mapping-level part of the notebook's
    ``PhysicalUnitNormalizer``. It is shared by full card normalization and
    by physical values extracted from product names.
    """
    multidimensional = _normalize_multidimensional_attributes(attributes)
    return _normalize_physical_attributes(multidimensional)


def _load_synonym_replacements(path: str | Path) -> dict[str, str]:
    synonyms = pl.read_parquet(path)
    required_columns = {"replacer", "synonyms"}
    missing_columns = required_columns - set(synonyms.columns)
    if missing_columns:
        raise ValueError(f"Synonym file is missing columns: {sorted(missing_columns)}")

    replacements = {}
    for replacer, aliases in synonyms.select("replacer", "synonyms").iter_rows():
        if replacer is None or aliases is None:
            continue
        for alias in aliases:
            replacements[str(alias).lower()] = str(replacer).lower()
    return replacements


def _normalize_synonym_word(
    word: str,
    replacements: Mapping[str, str],
    cache: dict[str, str],
) -> str:
    word = word.lower()
    if word in cache:
        return cache[word]

    morphology = _get_morphology()
    source = morphology.parse(word)[0]
    replacer = replacements.get(source.normal_form)
    if replacer is None:
        replacer = replacements.get(word)
    if replacer is None:
        result = word
    else:
        target = morphology.parse(replacer)[0]
        grammemes = {
            grammeme
            for grammeme in (source.tag.case, source.tag.number, source.tag.gender)
            if grammeme is not None
        }
        inflected = target.inflect(grammemes)
        result = inflected.word if inflected is not None else replacer

    cache[word] = result
    return result


def _normalize_attribute_name(
    text: str,
    replacements: Mapping[str, str],
    cache: dict[str, str],
) -> str:
    return _RUSSIAN_WORD_PATTERN.sub(
        lambda match: _normalize_synonym_word(match.group(), replacements, cache),
        text.lower(),
    )


def _load_attribute_name_map(
    path: str | Path,
    replacements: Mapping[str, str],
    cache: dict[str, str],
) -> dict[str, str]:
    unique_attributes = pl.read_parquet(path)
    if "attribute" not in unique_attributes.columns:
        raise ValueError("Unique-attributes file is missing column: 'attribute'")

    return {
        str(attribute): _normalize_attribute_name(
            str(attribute),
            replacements,
            cache,
        )
        for attribute in unique_attributes["attribute"].drop_nulls()
    }


def _normalize_synonym_attributes(
    attributes: Mapping[Any, Any],
    attribute_name_map: Mapping[str, str],
) -> dict[str, str]:
    result = {}
    for attribute, value in attributes.items():
        normalized_attribute = attribute_name_map.get(str(attribute), str(attribute))
        result.setdefault(normalized_attribute, str(value))
    return result


def _normalize_attribute_json(
    raw: Any,
    attribute_name_map: Mapping[str, str],
) -> str | None:
    if raw is None:
        return None
    original = raw
    try:
        attributes = (
            dict(raw) if isinstance(raw, Mapping) else orjson.loads(str(raw))
        )
    except (orjson.JSONDecodeError, TypeError, ValueError):
        return str(original)
    if not isinstance(attributes, dict):
        return str(original)
    attributes = normalize_physical_attributes(attributes)
    attributes = _normalize_synonym_attributes(attributes, attribute_name_map)
    return json.dumps(attributes, ensure_ascii=False)


def _initialize_normalization_worker(attribute_name_map: Mapping[str, str]) -> None:
    global _WORKER_ATTRIBUTE_NAME_MAP
    _WORKER_ATTRIBUTE_NAME_MAP = attribute_name_map


def _normalize_chunk(raw_values: list[Any]) -> list[str | None]:
    if _WORKER_ATTRIBUTE_NAME_MAP is None:
        raise RuntimeError("Normalization worker was not initialized")
    return [_normalize_attribute_json(raw, _WORKER_ATTRIBUTE_NAME_MAP) for raw in raw_values]


def _iter_chunks(series: pl.Series, chunk_size: int):
    for offset in range(0, len(series), chunk_size):
        yield series.slice(offset, chunk_size).to_list()


def normalize_attributes(
    frame: pl.DataFrame,
    synonyms_path: str | Path,
    unique_attributes_path: str | Path,
    *,
    source_column: str = "attributes",
    output_column: str = "normalized_attributes",
    n_jobs: int = 4,
    chunk_size: int = 5_000,
) -> pl.DataFrame:
    """Return a Polars frame with normalized product attributes.

    The source column may contain JSON objects encoded as strings.
    ``synonyms_path`` must point to a parquet file with ``replacer`` and
    list-valued ``synonyms`` columns. ``unique_attributes_path`` must point to
    a parquet file with an ``attribute`` column. The input frame is not
    mutated; the normalized JSON is written to ``output_column``.
    """
    if not isinstance(frame, pl.DataFrame):
        logger.error("Expected polars.DataFrame, received {}", type(frame).__name__)
        raise TypeError("frame must be a polars.DataFrame")
    if source_column not in frame.columns:
        logger.error("Source column {!r} is missing; available columns: {}", source_column, frame.columns)
        raise ValueError(f"Missing source column: {source_column!r}")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    started_at = perf_counter()
    logger.info("Starting attribute normalization: rows={}, source_column={!r}, output_column={!r}", frame.height, source_column, output_column)

    try:
        replacements = _load_synonym_replacements(synonyms_path)
        synonym_cache: dict[str, str] = {}
        attribute_name_map = _load_attribute_name_map(unique_attributes_path, replacements, synonym_cache)
    except (OSError, pl.exceptions.PolarsError, ValueError):
        logger.exception(
            "Failed to load normalization metadata: synonyms_path={!s}, "
            "unique_attributes_path={!s}",
            synonyms_path,
            unique_attributes_path,
        )
        raise

    logger.info("Loaded normalization metadata: synonym_replacements={}, unique_attributes={}", len(replacements), len(attribute_name_map))

    if n_jobs == 1 or frame.height <= chunk_size:
        logger.info("Using sequential normalization")
        result = frame.with_columns(
            pl.col(source_column)
            .map_elements(
                lambda raw: _normalize_attribute_json(raw, attribute_name_map),
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .alias(output_column)
        )
    else:
        chunk_count = math.ceil(frame.height / chunk_size)
        logger.info("Using process-based normalization: n_jobs={}, chunk_size={}, chunks={}", n_jobs, chunk_size, chunk_count)
        normalized_chunks = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            pre_dispatch=n_jobs,
            initializer=_initialize_normalization_worker,
            initargs=(attribute_name_map,),
        )(delayed(_normalize_chunk)(chunk) for chunk in _iter_chunks(frame.get_column(source_column), chunk_size))
        normalized_values = [value for normalized_chunk in normalized_chunks for value in normalized_chunk]
        result = frame.with_columns(pl.Series(output_column, normalized_values, dtype=pl.String))
    logger.info("Finished attribute normalization: rows={}, elapsed_seconds={:.3f}", result.height, perf_counter() - started_at)
    return result
