"""Paired training ablations evaluated on an identical validation split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import polars as pl
from loguru import logger

from ..config import AppConfig
from ..experiments import configure_experiment
from ..workflows.train import train
from .suite import (
    BenchmarkOverride,
    BenchmarkTest,
    apply_benchmark_test,
    load_benchmark_suite,
    select_benchmark_tests,
)


@dataclass(frozen=True, slots=True)
class TrainingQualityBenchmarkResult:
    output_dir: Path
    results_path: Path
    aggregates_path: Path
    report_path: Path
    completed_runs: int
    failed_runs: int

    def summary(self) -> str:
        return (
            f"Training-quality benchmark saved to {self.output_dir}; "
            f"completed={self.completed_runs}, failed={self.failed_runs}"
        )


def _seeded_test(test: BenchmarkTest, seed: int) -> BenchmarkTest:
    """Keep every random component and both data splits paired by seed."""
    seed_overrides = tuple(
        BenchmarkOverride(path, seed)
        for path in (
            "runtime.seed",
            "data_model_description.base_dataset.seed",
            "data_model_description.mix_dataset.seed",
            "augmentation_models.attribute_shuffle.seed",
            "augmentation_models.attribute_word_dropout.seed",
        )
    )
    return BenchmarkTest(test.key, test.name, (*test.overrides, *seed_overrides))


def _validation_metric(metrics: tuple[tuple[str, float], ...]) -> float:
    candidates = [
        float(value)
        for name, value in metrics
        if name.endswith(".validation_macro_pr_auc")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "training benchmark requires exactly one validation_macro_pr_auc metric"
        )
    return candidates[0]


def _run_case(
    base_config: AppConfig,
    test: BenchmarkTest,
    seed: int,
    *,
    experiments_root: Path,
) -> dict[str, Any]:
    experiment_name = f"{test.key}-seed-{seed}"
    row: dict[str, Any] = {
        "test": test.key,
        "name": test.name,
        "seed": seed,
        "experiment_name": experiment_name,
        "status": "failed",
        "validation_macro_pr_auc": None,
        "delta_validation_macro_pr_auc": None,
        "validation_pairs_hash": None,
        "same_validation_split_as_reference": None,
        "training_seconds": None,
        "llm_mean_weight_multiplier": None,
        "llm_downweighted_fraction": None,
        "llm_violating_fraction": None,
        "experiment_path": None,
        "error": None,
    }
    started = perf_counter()
    try:
        configured = apply_benchmark_test(base_config, _seeded_test(test, seed))
        if configured.training.data_model != "mix_dataset":
            raise ValueError(
                "training-quality benchmark for sample weighting requires "
                "training.data_model=mix_dataset"
            )
        configured, experiment_dir, registry_path = configure_experiment(
            configured,
            experiment_name,
            experiments_root=experiments_root,
        )
        artifacts = train(
            configured,
            experiment_name=experiment_name,
            experiment_registry_path=registry_path,
        )
        record_path = experiment_dir / "experiment.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        llm_weighting = record.get("sample_weighting", {}).get("llm", {})
        row.update(
            {
                "status": "completed",
                "validation_macro_pr_auc": _validation_metric(artifacts.metrics),
                "validation_pairs_hash": record["split"][
                    "validation_pairs_hash"
                ],
                "experiment_path": str(experiment_dir),
                "llm_mean_weight_multiplier": llm_weighting.get(
                    "mean_weight_multiplier"
                ),
                "llm_downweighted_fraction": llm_weighting.get(
                    "downweighted_fraction"
                ),
                "llm_violating_fraction": llm_weighting.get(
                    "violating_fraction"
                ),
            }
        )
    except Exception as error:  # Keep independent, expensive cases running.
        row["error"] = f"{type(error).__name__}: {error}"
        logger.exception(
            "Training-quality benchmark case {} seed {} failed",
            test.key,
            seed,
        )
    finally:
        row["training_seconds"] = perf_counter() - started
    return row


def _compare_with_reference(
    rows: list[dict[str, Any]],
    *,
    reference_test: str,
) -> None:
    references = {
        int(row["seed"]): row
        for row in rows
        if row["test"] == reference_test and row["status"] == "completed"
    }
    for row in rows:
        if row["status"] != "completed":
            continue
        reference = references.get(int(row["seed"]))
        if reference is None:
            continue
        same_split = (
            row["validation_pairs_hash"] == reference["validation_pairs_hash"]
        )
        row["same_validation_split_as_reference"] = same_split
        if not same_split:
            row["status"] = "failed"
            row["error"] = (
                "validation split differs from the paired reference run"
            )
            continue
        row["delta_validation_macro_pr_auc"] = float(
            row["validation_macro_pr_auc"]
            - reference["validation_macro_pr_auc"]
        )


def _aggregate(
    rows: list[dict[str, Any]],
    tests: tuple[BenchmarkTest, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for test in tests:
        attempted = [row for row in rows if row["test"] == test.key]
        completed = [
            row
            for row in rows
            if row["test"] == test.key and row["status"] == "completed"
        ]
        metrics = [float(row["validation_macro_pr_auc"]) for row in completed]
        deltas = [
            float(row["delta_validation_macro_pr_auc"])
            for row in completed
            if row["delta_validation_macro_pr_auc"] is not None
        ]
        result.append(
            {
                "test": test.key,
                "name": test.name,
                "completed_runs": len(completed),
                "failed_runs": sum(
                    row["test"] == test.key and row["status"] == "failed"
                    for row in rows
                ),
                "mean_validation_macro_pr_auc": (
                    None if not metrics else mean(metrics)
                ),
                "std_validation_macro_pr_auc": (
                    None if not metrics else pstdev(metrics)
                ),
                "mean_delta_validation_macro_pr_auc": (
                    None if not deltas else mean(deltas)
                ),
                "std_delta_validation_macro_pr_auc": (
                    None if not deltas else pstdev(deltas)
                ),
                "wins_vs_reference": sum(delta > 0.0 for delta in deltas),
                "paired_runs": len(deltas),
                "mean_training_seconds": (
                    None
                    if not completed
                    else mean(float(row["training_seconds"]) for row in completed)
                ),
                "total_training_seconds": sum(
                    float(row["training_seconds"])
                    for row in attempted
                    if row["training_seconds"] is not None
                ),
            }
        )
    return result


def run_training_quality_benchmark(
    config: AppConfig,
    *,
    tests_path: Path,
    output_dir: Path,
    selected_tests: tuple[str, ...] | None = None,
) -> TrainingQualityBenchmarkResult:
    """Train every ablation and compare paired validation PR-AUC values."""
    benchmark_started = perf_counter()
    suite = load_benchmark_suite(
        tests_path,
        allowed_test_types={"training_quality"},
    )
    tests = select_benchmark_tests(suite, selected_tests)
    if suite.reference_test is None:
        raise ValueError("training-quality benchmark requires reference_test")
    if suite.reference_test not in {test.key for test in tests}:
        raise ValueError("training-quality reference_test must be selected")
    seeds = suite.seeds or (config.runtime.seed,)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    experiments_root = output_dir / "experiments"

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for test in tests:
            logger.info(
                "Running training-quality benchmark: test={}, seed={}",
                test.key,
                seed,
            )
            rows.append(
                _run_case(
                    config,
                    test,
                    seed,
                    experiments_root=experiments_root,
                )
            )
    _compare_with_reference(rows, reference_test=suite.reference_test)
    aggregates = _aggregate(rows, tests)

    results_path = output_dir / "results.parquet"
    aggregates_path = output_dir / "aggregates.parquet"
    report_path = output_dir / "report.json"
    pl.DataFrame(rows).write_parquet(results_path)
    pl.DataFrame(aggregates).write_parquet(aggregates_path)
    total_elapsed_seconds = perf_counter() - benchmark_started
    report = {
        "schema_version": 1,
        "test_type": suite.test_type,
        "reference_test": suite.reference_test,
        "tests_path": str(tests_path),
        "seeds": list(seeds),
        "total_elapsed_seconds": total_elapsed_seconds,
        "rows": rows,
        "aggregates": aggregates,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    completed = sum(row["status"] == "completed" for row in rows)
    return TrainingQualityBenchmarkResult(
        output_dir=output_dir,
        results_path=results_path,
        aggregates_path=aggregates_path,
        report_path=report_path,
        completed_runs=completed,
        failed_runs=len(rows) - completed,
    )


__all__ = [
    "TrainingQualityBenchmarkResult",
    "run_training_quality_benchmark",
]
