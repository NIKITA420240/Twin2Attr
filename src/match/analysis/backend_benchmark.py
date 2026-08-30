"""Reproducible comparison of Transformer inference backends."""

from __future__ import annotations

import copy
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..benchmarks.suite import (
    BenchmarkOverride,
    BenchmarkSuite,
    BenchmarkTest,
    apply_benchmark_test,
    load_benchmark_suite,
    select_benchmark_tests,
)
from ..batch_profiles import (
    BatchMeasurement,
    PerformanceWorkload,
    current_environment,
    model_content_hash,
    profile_fingerprint,
    save_profile,
    update_profile,
    utc_now,
)
from ..config import (
    AppConfig,
    app_config_to_mapping,
    save_app_config,
)
from ..data import prepare_configured_items, prepare_pair_rows
from ..data_models import build_data_model
from ..models.contracts import PredictionBatch
from ..models.factory import build_predictor


@dataclass(frozen=True, slots=True)
class BackendBenchmarkResult:
    output_path: Path
    report_path: Path
    sample_rows: int
    completed_tests: int
    failed_tests: int

    def summary(self) -> str:
        return (
            f"Backend benchmark saved to {self.output_path}; "
            f"sample_rows={self.sample_rows}, completed={self.completed_tests}, "
            f"failed={self.failed_tests}"
        )


def _load_suite(path: Path) -> BenchmarkSuite:
    return load_benchmark_suite(path, allowed_test_types={"speed", "quality"})


def _select_tests(
    config: AppConfig,
) -> tuple[tuple[BenchmarkTest, ...], str, str]:
    settings = config.analysis_models.backend_benchmark
    if settings is None:
        raise ValueError(
            "analysis_models.backend_benchmark is required for backend_benchmark"
        )
    suite = _load_suite(settings.tests_path)
    selected = select_benchmark_tests(suite, settings.selected_tests)
    reference_test = suite.reference_test or settings.reference_test
    if reference_test not in {test.key for test in selected}:
        raise ValueError(
            f"benchmark reference_test {reference_test!r} must be selected"
        )
    return (
        selected,
        reference_test,
        suite.test_type,
    )


def _load_solution(config: AppConfig) -> tuple[dict[str, Any], Path]:
    path = config.inference.solution_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark solution manifest does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark solution manifest must contain an object")
    if "model_directory" not in value:
        raise ValueError("benchmark solution manifest has no model_directory")
    return value, path.parent


def _runtime_solution(base: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Overlay validated Transformer runtime settings on a trained manifest."""
    values = app_config_to_mapping(config)
    transformer = values["inference"]["transformer"]
    result = copy.deepcopy(base)
    result["predictor"] = "transformer"
    for key in (
        "backend",
        "batch_size",
        "dtype",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "non_blocking_transfer",
        "attention",
        "length_bucketing",
        "torch_compile",
        "onnxruntime",
        "tensorrt",
    ):
        result[key] = copy.deepcopy(transformer[key])
    result["onnxruntime"]["fallback_to_pytorch"] = False
    result["tensorrt"]["fallback_to_onnxruntime"] = False
    result["tokenizer"] = copy.deepcopy(
        values["model_description"]["transformer"]["tokenizer"]
    )
    result["pair_encoding"] = {
        "max_length": transformer["max_length"],
        "max_attribute_value_chars": transformer[
            "max_attribute_value_chars"
        ],
        "max_attribute_value_tokens": transformer[
            "max_attribute_value_tokens"
        ],
    }
    if "onnx_artifacts" not in result:
        export = values["model_description"]["transformer"]["export"]["onnx"]
        result["onnx_artifacts"] = {
            **copy.deepcopy(export),
            "classifier_path": "onnx/classifier.onnx",
            "encoder_path": "onnx/encoder.onnx",
        }
    return result


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        return


def _release_backend() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _gpu_memory_used_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return float(total - free) / 1024**3
    except (ImportError, RuntimeError):
        return None


def _gpu_peak_memory_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / 1024**3
    except (ImportError, RuntimeError):
        return None


def _token_statistics(
    predictor: Any,
    batch: PredictionBatch,
    *,
    max_length: int | None,
) -> tuple[int, dict[str, int]] | None:
    """Count real tokenizer tokens once, outside measured inference time."""
    tokenizer = getattr(predictor, "tokenizer", None)
    batching = getattr(predictor, "_batching", None)
    resolved_length = getattr(predictor, "_resolved_max_length", None)
    if tokenizer is None or batching is None or not callable(resolved_length):
        return None
    try:
        pairs = batch.prepared_pairs()
        collator = batching.collator(
            tokenizer,
            max_length=resolved_length(pairs, max_length),
        )
        lengths: list[int] = []
        for offset in range(0, len(pairs), 4_096):
            encoded = collator(pairs[offset : offset + 4_096])
            attention_mask = encoded["attention_mask"]
            lengths.extend(
                int(value)
                for value in attention_mask.sum(dim=1).detach().cpu().tolist()
            )
        if not lengths:
            return 0, {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
        values = np.asarray(lengths, dtype=np.int64)
        return int(values.sum()), {
            name: int(np.percentile(values, percentile, method="higher"))
            for name, percentile in (
                ("p50", 50),
                ("p90", 90),
                ("p95", 95),
                ("p99", 99),
            )
        }
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        logger.warning("Could not collect exact token statistics for benchmark")
        return None


def _quality_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predicted = probabilities >= 0.5
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def _binary_labels(matches: pl.DataFrame) -> np.ndarray | None:
    if "target" not in matches.columns:
        return None
    labels = matches.get_column("target").to_numpy()
    if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1}):
        return None
    if len(np.unique(labels)) != 2:
        return None
    return labels.astype(np.int64, copy=False)


def _sample_matches(
    matches: pl.DataFrame,
    *,
    sample_size: int | None,
    seed: int,
) -> pl.DataFrame:
    if sample_size is None or matches.height <= sample_size:
        return matches
    return matches.sample(n=sample_size, shuffle=True, seed=seed)


def _selected_items(items: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    ids = pl.concat(
        [
            matches.select(pl.col("id1").alias("id")),
            matches.select(pl.col("id2").alias("id")),
        ]
    ).unique()
    return items.join(ids, on="id", how="semi")


def _run_test(
    base_config: AppConfig,
    base_solution: dict[str, Any],
    solution_root: Path,
    test: BenchmarkTest,
    batch: PredictionBatch,
    *,
    warmup_batches: int,
    measured_runs: int,
    output_dir: Path,
    test_type: str,
    token_statistics_cache: dict[
        tuple[int | None, int | None, int | None],
        tuple[int, dict[str, int]] | None,
    ],
) -> tuple[dict[str, Any], np.ndarray | None]:
    config = apply_benchmark_test(base_config, test)
    test_dir = output_dir / test.key
    save_app_config(config, test_dir / "config.yaml")
    solution = _runtime_solution(base_solution, config)
    (test_dir / "solution.json").write_text(
        json.dumps(solution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    transformer = config.inference.transformer
    row: dict[str, Any] = {
        "test": test.key,
        "name": test.name,
        "test_type": test_type,
        "backend": transformer.backend,
        "provider": (
            transformer.onnxruntime.provider
            if transformer.backend == "onnxruntime"
            else None
        ),
        "batch_size": transformer.batch_size,
        "dtype": transformer.dtype,
        "attention": transformer.attention.implementation,
        "torch_compile": transformer.torch_compile.enabled,
        "max_length": transformer.max_length,
        "padding_buckets": (
            None
            if (
                not transformer.length_bucketing.enabled
                or transformer.length_bucketing.padding_length_buckets is None
            )
            else list(transformer.length_bucketing.padding_length_buckets)
        ),
        "max_attribute_value_chars": transformer.max_attribute_value_chars,
        "max_attribute_value_tokens": transformer.max_attribute_value_tokens,
        "sample_rows": batch.matches.height,
        "status": "failed",
        "prepare_seconds": None,
        "warmup_seconds": None,
        "mean_inference_seconds": None,
        "min_inference_seconds": None,
        "std_inference_seconds": None,
        "pairs_per_second": None,
        "tokens_per_second": None,
        "real_tokens": None,
        "length_quantiles": None,
        "gpu_memory_used_gb": None,
        "peak_vram_gib": None,
        "p95_step_ms": None,
        "mean_probability_difference": None,
        "max_probability_difference": None,
        "class_disagreement_rate": None,
        "quality_gate_passed": None,
        "pr_auc": None,
        "roc_auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "delta_pr_auc": None,
        "delta_roc_auc": None,
        "delta_accuracy": None,
        "delta_precision": None,
        "delta_recall": None,
        "delta_f1": None,
        "error": None,
    }
    predictor = None
    predictions: np.ndarray | None = None
    _release_backend()
    try:
        started = perf_counter()
        predictor = build_predictor(solution, solution_root)
        _cuda_synchronize()
        row["prepare_seconds"] = perf_counter() - started

        token_key = (
            transformer.max_length,
            transformer.max_attribute_value_chars,
            transformer.max_attribute_value_tokens,
        )
        if token_key not in token_statistics_cache:
            token_statistics_cache[token_key] = _token_statistics(
                predictor,
                batch,
                max_length=transformer.max_length,
            )
        token_statistics = token_statistics_cache[token_key]
        if token_statistics is not None:
            row["real_tokens"], row["length_quantiles"] = token_statistics

        warmup_started = perf_counter()
        if warmup_batches:
            warmup_rows = min(
                batch.matches.height,
                warmup_batches * transformer.batch_size,
            )
            predictor.predict_proba(batch.take_indices(range(warmup_rows)))
            _cuda_synchronize()
        row["warmup_seconds"] = perf_counter() - warmup_started

        durations: list[float] = []
        for _ in range(measured_runs):
            _cuda_synchronize()
            run_started = perf_counter()
            current = np.asarray(predictor.predict_proba(batch), dtype=np.float64)
            _cuda_synchronize()
            durations.append(perf_counter() - run_started)
            predictions = current
        if predictions is None or predictions.shape != (batch.matches.height,):
            raise RuntimeError("benchmark predictor returned an invalid shape")
        if not np.isfinite(predictions).all():
            raise RuntimeError("benchmark predictions contain NaN or infinity")
        average = mean(durations)
        batch_count = max(
            1,
            math.ceil(batch.matches.height / transformer.batch_size),
        )
        per_step_milliseconds = [
            duration * 1_000.0 / batch_count for duration in durations
        ]
        row.update(
            {
                "status": "completed",
                "mean_inference_seconds": average,
                "min_inference_seconds": min(durations),
                "std_inference_seconds": pstdev(durations),
                "pairs_per_second": batch.matches.height / average,
                "tokens_per_second": (
                    None
                    if row["real_tokens"] is None
                    else float(row["real_tokens"]) / average
                ),
                "gpu_memory_used_gb": _gpu_memory_used_gb(),
                "peak_vram_gib": _gpu_peak_memory_gib(),
                "p95_step_ms": float(
                    np.percentile(
                        np.asarray(per_step_milliseconds),
                        95,
                        method="higher",
                    )
                ),
            }
        )
    except Exception as error:  # Keep independent benchmark cases running.
        error_text = f"{type(error).__name__}: {error}"
        if "out of memory" in error_text.lower() or type(error).__name__ in {
            "OutOfMemoryError",
            "CUDAOutOfMemoryError",
        }:
            row["status"] = "oom"
        row["error"] = error_text
        logger.exception("Backend benchmark test {} failed", test.key)
    finally:
        del predictor
        _release_backend()
    return row, predictions


def _model_directory(base_solution: dict[str, Any], solution_root: Path) -> Path:
    value = Path(str(base_solution["model_directory"]))
    return value if value.is_absolute() else solution_root / value


def _persist_batch_profiles(
    settings: Any,
    base_solution: dict[str, Any],
    solution_root: Path,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    options = settings.batch_profiles
    if not options.enabled:
        return []
    environment = current_environment()
    model_hash = model_content_hash(
        _model_directory(base_solution, solution_root)
    )
    grouped: dict[str, tuple[PerformanceWorkload, list[BatchMeasurement]]] = {}
    for row in rows:
        if row["backend"] != "pytorch":
            continue
        workload = PerformanceWorkload(
            model_hash=model_hash,
            mode="inference",
            dtype=str(row["dtype"]),
            attention=str(row["attention"]),
            torch_compile=bool(row["torch_compile"]),
            max_length=(
                None if row["max_length"] is None else int(row["max_length"])
            ),
            padding_buckets=tuple(int(value) for value in (row["padding_buckets"] or ())),
            length_quantiles=(
                None
                if row["length_quantiles"] is None
                else {
                    str(key): int(value)
                    for key, value in row["length_quantiles"].items()
                }
            ),
        )
        fingerprint = profile_fingerprint(environment, workload)
        status = str(row["status"])
        if status not in {"completed", "oom"}:
            status = "failed"
        measurement = BatchMeasurement(
            batch_size=int(row["batch_size"]),
            status=status,
            tokens_per_second=(
                None
                if row["tokens_per_second"] is None
                else float(row["tokens_per_second"])
            ),
            pairs_per_second=(
                None
                if row["pairs_per_second"] is None
                else float(row["pairs_per_second"])
            ),
            peak_vram_gib=(
                None
                if row["peak_vram_gib"] is None
                else float(row["peak_vram_gib"])
            ),
            p95_step_ms=(
                None
                if row["p95_step_ms"] is None
                else float(row["p95_step_ms"])
            ),
            measured_at=utc_now(),
        )
        if fingerprint not in grouped:
            grouped[fingerprint] = (workload, [])
        grouped[fingerprint][1].append(measurement)

    summaries: list[dict[str, Any]] = []
    snapshot_directory = output_dir / "batch_profiles"
    for workload, measurements in grouped.values():
        profile = update_profile(
            options.directory,
            environment,
            workload,
            tuple(measurements),
            safe_batch_fraction=options.safe_batch_fraction,
            batch_size_multiple=options.batch_size_multiple,
        )
        snapshot_path = save_profile(snapshot_directory, profile)
        summaries.append(
            {
                "fingerprint": profile.fingerprint,
                "attention": workload.attention,
                "dtype": workload.dtype,
                "torch_compile": workload.torch_compile,
                "best_batch_size": profile.best_batch_size,
                "safe_batch_size": profile.safe_batch_size,
                "cache_path": str(
                    options.directory
                    / workload.mode
                    / f"{profile.fingerprint}.json"
                ),
                "snapshot_path": str(snapshot_path),
            }
        )
    return summaries


def run_backend_benchmark(config: AppConfig) -> BackendBenchmarkResult:
    """Run selected backend cases on one shared, prepared pair sample."""
    settings = config.analysis_models.backend_benchmark
    if settings is None:
        raise ValueError("backend_benchmark settings are missing")
    tests, reference_test, test_type = _select_tests(config)
    base_solution, solution_root = _load_solution(config)
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preparation_started = perf_counter()
    frames = build_data_model(
        config,
        name=config.analysis.data_model,
    ).load_inspection_frames()
    matches = _sample_matches(
        frames.matches,
        sample_size=settings.sample_size,
        seed=config.runtime.seed,
    )
    if matches.is_empty():
        raise ValueError("backend benchmark selected no match pairs")
    items = _selected_items(frames.items, matches)
    prepared_items = prepare_configured_items(items, config)
    pairs = prepare_pair_rows(
        prepared_items.frame,
        matches,
        prepared_items.attributes_column,
        split_name="backend benchmark",
    )
    batch = PredictionBatch(
        items=prepared_items.frame,
        matches=matches,
        attributes_column=prepared_items.attributes_column,
        pairs=pairs,
    )
    preparation_seconds = perf_counter() - preparation_started
    logger.info(
        "Backend benchmark sample prepared: rows={}, seconds={:.3f}",
        matches.height,
        preparation_seconds,
    )

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    token_statistics_cache: dict[
        tuple[int | None, int | None, int | None],
        tuple[int, dict[str, int]] | None,
    ] = {}
    for test in tests:
        logger.info("Running backend benchmark test: {} ({})", test.key, test.name)
        row, values = _run_test(
            config,
            base_solution,
            solution_root,
            test,
            batch,
            warmup_batches=settings.warmup_batches,
            measured_runs=settings.measured_runs,
            output_dir=output_dir,
            test_type=test_type,
            token_statistics_cache=token_statistics_cache,
        )
        rows.append(row)
        if values is not None:
            predictions[test.key] = values

    labels = _binary_labels(matches)
    if test_type == "quality" and labels is None:
        raise ValueError(
            "quality benchmark requires a binary target column containing both classes"
        )
    if labels is not None:
        for row in rows:
            values = predictions.get(str(row["test"]))
            if values is not None:
                row.update(_quality_metrics(labels, values))

    reference = predictions.get(reference_test)
    if settings.compare_predictions and reference is not None:
        for row in rows:
            values = predictions.get(str(row["test"]))
            if values is None:
                continue
            difference = np.abs(values - reference)
            row["mean_probability_difference"] = float(difference.mean())
            row["max_probability_difference"] = float(difference.max())
            row["class_disagreement_rate"] = float(
                np.mean((values >= 0.5) != (reference >= 0.5))
            )
            if test_type == "speed":
                row["quality_gate_passed"] = bool(
                    row["max_probability_difference"] <= 1.0e-3
                    and row["class_disagreement_rate"] == 0.0
                )
                if not row["quality_gate_passed"]:
                    row["status"] = "failed"
                    row["error"] = (
                        "prediction quality gate failed: "
                        f"max_probability_difference="
                        f"{row['max_probability_difference']:.6g}, "
                        f"class_disagreement_rate="
                        f"{row['class_disagreement_rate']:.6g}"
                    )
    elif settings.compare_predictions:
        logger.warning(
            "Reference benchmark test {} failed; prediction deltas are unavailable",
            reference_test,
        )

    completed = sum(row["status"] == "completed" for row in rows)
    reference_row = next(
        (row for row in rows if row["test"] == reference_test),
        None,
    )
    reference_seconds = (
        None if reference_row is None else reference_row["mean_inference_seconds"]
    )
    for row in rows:
        seconds = row["mean_inference_seconds"]
        row["speedup_vs_reference"] = (
            None
            if reference_seconds is None or seconds is None
            else float(reference_seconds / seconds)
        )
        row["common_preparation_seconds"] = preparation_seconds
        if reference_row is not None:
            for metric in (
                "pr_auc",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ):
                value = row[metric]
                reference_value = reference_row[metric]
                if value is not None and reference_value is not None:
                    row[f"delta_{metric}"] = float(value - reference_value)

    batch_profiles = _persist_batch_profiles(
        settings,
        base_solution,
        solution_root,
        rows,
        output_dir,
    )

    output_path = output_dir / "results.parquet"
    report_path = output_dir / "report.json"
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(output_path)
    report = {
        "sample_rows": matches.height,
        "common_preparation_seconds": preparation_seconds,
        "test_type": test_type,
        "reference_test": reference_test,
        "compare_predictions": settings.compare_predictions,
        "completed_tests": completed,
        "failed_tests": len(rows) - completed,
        "batch_profiles": batch_profiles,
        "tests": rows,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BackendBenchmarkResult(
        output_path=output_path,
        report_path=report_path,
        sample_rows=matches.height,
        completed_tests=completed,
        failed_tests=len(rows) - completed,
    )


__all__ = [
    "BackendBenchmarkResult",
    "BenchmarkOverride",
    "BenchmarkTest",
    "apply_benchmark_test",
    "run_backend_benchmark",
]
