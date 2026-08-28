"""Constant sample-weight model."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ConstantSampleWeightModel:
    """Keep every source row at the configured source-level weight."""

    def apply(self, matches: pl.DataFrame, *, source_name: str) -> pl.DataFrame:
        del source_name
        return matches.with_columns(
            pl.lit(1.0).cast(pl.Float32).alias("weight_multiplier")
        )


__all__ = ["ConstantSampleWeightModel"]
