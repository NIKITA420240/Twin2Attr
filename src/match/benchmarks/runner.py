"""Top-level orchestration for all configured benchmark jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from ..analysis.backend_benchmark import run_backend_benchmark
from ..config import AppConfig, save_app_config
from .training_quality import run_training_quality_benchmark


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    output_dir: Path
    report_path: Path
    completed_jobs: int
    failed_jobs: int
    skipped_jobs: int

    def summary(self) -> str:
        return (
            f"Benchmark run saved to {self.output_dir}; "
            f"completed={self.completed_jobs}, failed={self.failed_jobs}, "
            f"skipped={self.skipped_jobs}"
        )


def _unique_run_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root.expanduser().resolve() / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root.expanduser().resolve() / f"{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _run_inference_job(
    config: AppConfig,
    *,
    tests_path: Path,
    selected_tests: tuple[str, ...] | None,
    output_dir: Path,
) -> Any:
    settings = config.analysis_models.backend_benchmark
    if settings is None:
        raise ValueError(
            "inference benchmark requires analysis_models.backend_benchmark "
            "runtime settings"
        )
    configured = replace(
        config,
        analysis_models=replace(
            config.analysis_models,
            backend_benchmark=replace(
                settings,
                tests_path=tests_path,
                selected_tests=selected_tests,
                output_dir=output_dir,
            ),
        ),
    )
    return run_backend_benchmark(configured)


def run_configured_benchmarks(config: AppConfig) -> BenchmarkRunResult:
    """Run every enabled benchmark job and write one master report."""
    settings = config.benchmark
    if settings is None:
        raise ValueError("config section 'benchmark' is required")
    run_dir = _unique_run_dir(settings.output_dir)
    save_app_config(config, run_dir / "config.yaml")
    rows: list[dict[str, Any]] = []
    stop = False
    for job in settings.jobs:
        row: dict[str, Any] = {
            "job": job.name,
            "type": job.type,
            "enabled": job.enabled,
            "tests_path": str(job.tests_path),
            "selected_tests": (
                None if job.selected_tests is None else list(job.selected_tests)
            ),
            "status": "skipped",
            "elapsed_seconds": 0.0,
            "output_dir": None,
            "report_path": None,
            "failed_cases": None,
            "error": None,
        }
        if not job.enabled or stop:
            rows.append(row)
            continue
        job_dir = run_dir / job.name
        started = perf_counter()
        try:
            logger.info("Running benchmark job: {} ({})", job.name, job.type)
            if job.type == "inference":
                result = _run_inference_job(
                    config,
                    tests_path=job.tests_path,
                    selected_tests=job.selected_tests,
                    output_dir=job_dir,
                )
            else:
                result = run_training_quality_benchmark(
                    config,
                    tests_path=job.tests_path,
                    output_dir=job_dir,
                    selected_tests=job.selected_tests,
                )
            row.update(
                {
                    "status": "completed",
                    "output_dir": str(job_dir),
                    "report_path": str(result.report_path),
                }
            )
            failed_cases = int(
                getattr(
                    result,
                    "failed_tests",
                    getattr(result, "failed_runs", 0),
                )
            )
            row["failed_cases"] = failed_cases
            if failed_cases:
                row["status"] = "failed"
                row["error"] = f"{failed_cases} benchmark cases failed"
                stop = settings.fail_fast
        except Exception as error:  # Preserve results from independent jobs.
            row.update(
                {
                    "status": "failed",
                    "output_dir": str(job_dir),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            logger.exception("Benchmark job {} failed", job.name)
            stop = settings.fail_fast
        finally:
            row["elapsed_seconds"] = perf_counter() - started
        rows.append(row)

    report_path = run_dir / "report.json"
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    report = {
        "schema_version": 1,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped,
        "jobs": rows,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BenchmarkRunResult(
        output_dir=run_dir,
        report_path=report_path,
        completed_jobs=completed,
        failed_jobs=failed,
        skipped_jobs=skipped,
    )


__all__ = ["BenchmarkRunResult", "run_configured_benchmarks"]
