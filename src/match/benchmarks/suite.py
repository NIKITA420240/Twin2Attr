"""Typed YAML benchmark suites with validated configuration overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig, app_config_to_mapping, load_app_config


@dataclass(frozen=True, slots=True)
class BenchmarkOverride:
    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class BenchmarkTest:
    key: str
    name: str
    overrides: tuple[BenchmarkOverride, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    test_type: str
    reference_test: str | None
    tests: dict[str, BenchmarkTest]
    seeds: tuple[int, ...] = ()


def load_benchmark_suite(
    path: Path,
    *,
    allowed_test_types: set[str],
) -> BenchmarkSuite:
    """Load one suite and reject malformed or ambiguous cases."""
    from omegaconf import OmegaConf

    if not path.is_file():
        raise FileNotFoundError(f"Benchmark tests file does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("tests"), dict):
        raise ValueError("benchmark tests YAML must contain a 'tests' mapping")
    test_type = str(raw.get("test_type", "speed")).lower()
    if test_type not in allowed_test_types:
        expected = ", ".join(sorted(allowed_test_types))
        raise ValueError(f"benchmark test_type must be one of: {expected}")
    reference_value = raw.get("reference_test")
    reference_test = None if reference_value is None else str(reference_value)
    seeds_value = raw.get("seeds", [])
    if not isinstance(seeds_value, list):
        raise ValueError("benchmark seeds must be a list")
    seeds = tuple(int(seed) for seed in seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be unique")

    result: dict[str, BenchmarkTest] = {}
    names: set[str] = set()
    for raw_key, value in raw["tests"].items():
        key = str(raw_key)
        if not isinstance(value, dict):
            raise ValueError(f"benchmark test {key!r} must be a mapping")
        name = str(value.get("name", key)).strip()
        if not name:
            raise ValueError(f"benchmark test {key!r} name must not be empty")
        if name in names:
            raise ValueError(f"benchmark test name {name!r} is duplicated")
        names.add(name)
        overrides_value = value.get("overrides")
        if not isinstance(overrides_value, list) or not overrides_value:
            raise ValueError(
                f"benchmark test {key!r} overrides must be a non-empty list"
            )
        overrides: list[BenchmarkOverride] = []
        paths: set[str] = set()
        for index, override in enumerate(overrides_value):
            if not isinstance(override, dict):
                raise ValueError(
                    f"benchmark test {key!r} override {index} must be a mapping"
                )
            path_value = override.get("path")
            if path_value is None or not str(path_value).strip():
                raise ValueError(
                    f"benchmark test {key!r} override {index} requires path"
                )
            if "value" not in override:
                raise ValueError(
                    f"benchmark test {key!r} override {index} requires value"
                )
            dotted_path = str(path_value)
            if dotted_path in paths:
                raise ValueError(
                    f"benchmark test {key!r} overrides path {dotted_path!r} twice"
                )
            paths.add(dotted_path)
            overrides.append(BenchmarkOverride(dotted_path, override["value"]))
        result[key] = BenchmarkTest(key, name, tuple(overrides))
    if not result:
        raise ValueError("benchmark tests YAML contains no tests")
    if reference_test is not None and reference_test not in result:
        raise ValueError(
            f"benchmark reference_test {reference_test!r} is absent from tests"
        )
    return BenchmarkSuite(test_type, reference_test, result, seeds)


def select_benchmark_tests(
    suite: BenchmarkSuite,
    selected_tests: tuple[str, ...] | None,
) -> tuple[BenchmarkTest, ...]:
    """Resolve an optional ordered subset from a suite."""
    selected = tuple(suite.tests) if selected_tests is None else selected_tests
    missing = [key for key in selected if key not in suite.tests]
    if missing:
        raise ValueError(
            f"Unknown selected benchmark tests {missing}; "
            f"available: {list(suite.tests)}"
        )
    return tuple(suite.tests[key] for key in selected)


def _set_path(values: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid benchmark override path: {dotted_path!r}")
    current: dict[str, Any] = values
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(
                f"Unknown benchmark override path {dotted_path!r}: "
                f"{part!r} is absent or is not a mapping"
            )
        current = child
    leaf = parts[-1]
    if leaf not in current:
        raise ValueError(f"Unknown benchmark override path: {dotted_path!r}")
    current[leaf] = value


def apply_benchmark_test(config: AppConfig, test: BenchmarkTest) -> AppConfig:
    """Apply one test to a fresh mutable snapshot and run normal validation."""
    values = app_config_to_mapping(config)
    for override in test.overrides:
        _set_path(values, override.path, override.value)
    try:
        return load_app_config(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid configuration produced by benchmark test {test.key!r}: {error}"
        ) from error


__all__ = [
    "BenchmarkOverride",
    "BenchmarkSuite",
    "BenchmarkTest",
    "apply_benchmark_test",
    "load_benchmark_suite",
    "select_benchmark_tests",
]
