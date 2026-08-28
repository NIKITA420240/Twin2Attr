"""Graph-local transitivity weighting for noisy pair labels."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import polars as pl
from loguru import logger

_REQUIRED_COLUMNS = {"id1", "id2", "target"}
_CONFIDENCE_COLUMN = "_annotation_confidence"

EdgeKey = frozenset[Any]
Neighbor = tuple[int, float]


def _edge_key(left: Any, right: Any) -> EdgeKey:
    try:
        return frozenset((left, right))
    except TypeError as error:
        raise ValueError("transitivity weighting requires hashable item ids") from error


@dataclass(frozen=True, slots=True)
class _Edge:
    left: Any
    right: Any
    target: int
    confidence: float


@dataclass(frozen=True, slots=True)
class _Metrics:
    comparable: int
    violations: int
    violation_rate: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class TransitivitySampleWeightModel:
    """Downweight edges participating in contradictory observed triangles."""

    penalty_strength: float = 1.0
    min_weight_multiplier: float = 0.25
    min_comparable_neighbors: int = 2
    confidence_weighted_violations: bool = True

    def __post_init__(self) -> None:
        if self.penalty_strength < 0.0:
            raise ValueError("penalty_strength must not be negative")
        if not 0.0 < self.min_weight_multiplier <= 1.0:
            raise ValueError("min_weight_multiplier must be in (0, 1]")
        if self.min_comparable_neighbors < 1:
            raise ValueError("min_comparable_neighbors must be positive")

    def apply(self, matches: pl.DataFrame, *, source_name: str) -> pl.DataFrame:
        started_at = perf_counter()
        missing = _REQUIRED_COLUMNS - set(matches.columns)
        if missing:
            raise ValueError(
                f"transitivity weight model is missing columns: {sorted(missing)}"
            )
        if matches.height == 0:
            return matches.with_columns(
                pl.lit(1.0).cast(pl.Float32).alias("weight_multiplier")
            )

        rows = self._rows(matches)
        labels: dict[EdgeKey, set[int]] = defaultdict(set)
        confidences: dict[EdgeKey, list[float]] = defaultdict(list)
        endpoints: dict[EdgeKey, tuple[Any, Any]] = {}
        for edge in rows:
            key = _edge_key(edge.left, edge.right)
            labels[key].add(edge.target)
            confidences[key].append(edge.confidence)
            endpoints.setdefault(key, (edge.left, edge.right))

        conflicting = {key for key, values in labels.items() if len(values) > 1}
        stable: dict[EdgeKey, _Edge] = {}
        adjacency: dict[Any, dict[Any, Neighbor]] = defaultdict(dict)
        for key, values in labels.items():
            if key in conflicting:
                continue
            left, right = endpoints[key]
            target = next(iter(values))
            confidence = sum(confidences[key]) / len(confidences[key])
            edge = _Edge(left, right, target, confidence)
            stable[key] = edge
            if left == right:
                continue
            adjacency[left][right] = (target, confidence)
            adjacency[right][left] = (target, confidence)

        metrics = {
            key: self._edge_metrics(edge, adjacency)
            for key, edge in stable.items()
        }
        conflict_metrics = self._conflict_metrics()
        row_metrics = [
            conflict_metrics if (key := _edge_key(edge.left, edge.right)) in conflicting
            else metrics[key]
            for edge in rows
        ]
        result = matches.with_columns(
            pl.Series(
                "transitivity_comparable_neighbors",
                [value.comparable for value in row_metrics],
                dtype=pl.UInt32,
            ),
            pl.Series(
                "transitivity_violations",
                [value.violations for value in row_metrics],
                dtype=pl.UInt32,
            ),
            pl.Series(
                "transitivity_violation_rate",
                [value.violation_rate for value in row_metrics],
                dtype=pl.Float32,
            ),
            pl.Series(
                "weight_multiplier",
                [value.multiplier for value in row_metrics],
                dtype=pl.Float32,
            ),
        )
        affected = sum(value.multiplier < 1.0 for value in row_metrics)
        violating = sum(value.violations > 0 for value in row_metrics)
        comparable = sum(
            value.comparable >= self.min_comparable_neighbors
            for value in row_metrics
        )
        logger.info(
            "Applied transitivity sample weights: source={}, rows={}, "
            "comparable_rows={}, violating_rows={}, affected_rows={}, "
            "affected_fraction={:.6f}, mean_multiplier={:.6f}, "
            "min_multiplier={:.6f}, conflicting_duplicate_edges={}, "
            "elapsed_seconds={:.3f}",
            source_name,
            matches.height,
            comparable,
            violating,
            affected,
            affected / matches.height,
            sum(value.multiplier for value in row_metrics) / matches.height,
            min(value.multiplier for value in row_metrics),
            len(conflicting),
            perf_counter() - started_at,
        )
        return result

    def _rows(self, matches: pl.DataFrame) -> list[_Edge]:
        confidence = (
            pl.col(_CONFIDENCE_COLUMN).cast(pl.Float64)
            if _CONFIDENCE_COLUMN in matches.columns
            else pl.lit(1.0)
        )
        selected = matches.select(
            "id1",
            "id2",
            "target",
            confidence.alias("_confidence"),
        )
        rows: list[_Edge] = []
        for left, right, raw_target, raw_confidence in selected.iter_rows():
            target = int(raw_target)
            if target not in {0, 1}:
                raise ValueError("transitivity weighting requires binary targets")
            value = float(raw_confidence)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("annotation confidence must be in [0, 1]")
            rows.append(_Edge(left, right, target, value))
        return rows

    def _edge_metrics(
        self,
        edge: _Edge,
        adjacency: dict[Any, dict[Any, Neighbor]],
    ) -> _Metrics:
        if edge.left == edge.right:
            return self._direct_metrics(violation=edge.target == 0)
        left_neighbors = adjacency.get(edge.left, {})
        right_neighbors = adjacency.get(edge.right, {})
        if len(left_neighbors) <= len(right_neighbors):
            candidates, other = left_neighbors, right_neighbors
        else:
            candidates, other = right_neighbors, left_neighbors
        comparable = 0
        violations = 0
        weighted_violations = 0.0
        for neighbor, first in candidates.items():
            second = other.get(neighbor)
            if second is None:
                continue
            comparable += 1
            first_target, first_confidence = first
            second_target, second_confidence = second
            violation = (
                first_target != second_target
                if edge.target == 1
                else first_target == second_target == 1
            )
            if not violation:
                continue
            violations += 1
            if self.confidence_weighted_violations:
                evidence = min(first_confidence, second_confidence)
                denominator = max(edge.confidence + evidence, 1e-12)
                weighted_violations += 2.0 * evidence / denominator
            else:
                weighted_violations += 1.0
        if comparable == 0 or violations == 0:
            return _Metrics(comparable, violations, 0.0, 1.0)
        rate = weighted_violations / comparable
        if comparable < self.min_comparable_neighbors:
            return _Metrics(comparable, violations, rate, 1.0)
        multiplier = max(
            self.min_weight_multiplier,
            math.exp(-self.penalty_strength * rate),
        )
        return _Metrics(comparable, violations, rate, multiplier)

    def _direct_metrics(self, *, violation: bool) -> _Metrics:
        if not violation:
            return _Metrics(0, 0, 0.0, 1.0)
        multiplier = max(
            self.min_weight_multiplier,
            math.exp(-self.penalty_strength),
        )
        return _Metrics(1, 1, 1.0, multiplier)

    def _conflict_metrics(self) -> _Metrics:
        # Opposite labels for the same unordered pair are direct evidence and
        # do not require a third graph node.
        multiplier = max(
            self.min_weight_multiplier,
            math.exp(-self.penalty_strength),
        )
        return _Metrics(1, 1, 1.0, multiplier)


__all__ = ["TransitivitySampleWeightModel"]
