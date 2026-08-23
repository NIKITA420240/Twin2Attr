"""Shared Parquet loading with structured logging."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
from loguru import logger


def read_parquet(
    path: Path,
    *,
    label: str,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read a parquet file and report its basic shape and loading time."""
    started_at = perf_counter()
    frame = pl.read_parquet(path, columns=columns)
    logger.info(
        "Loaded {}: path={!s}, rows={}, columns={}, elapsed_seconds={:.3f}",
        label,
        path,
        frame.height,
        len(frame.columns),
        perf_counter() - started_at,
    )
    return frame
