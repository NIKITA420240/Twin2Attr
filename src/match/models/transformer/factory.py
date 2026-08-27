"""Build a Transformer predictor from a packaged solution manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from transformers import AutoTokenizer

from .onnx_runtime import (
    OnnxRuntimeInitializationError,
    OnnxRuntimeTransformerExecutor,
    OnnxRuntimeUnavailableError,
)
from .predictor import TransformerPredictor

TransformerUsage = Literal["classifier", "encoder"]


def _length_bucketing_options(
    solution: Mapping[str, Any],
) -> tuple[bool, tuple[int, ...] | None]:
    value = solution.get("length_bucketing", False)
    if isinstance(value, Mapping):
        buckets_value = value.get("padding_length_buckets")
        if buckets_value is None:
            buckets = None
        elif isinstance(buckets_value, (list, tuple)):
            if any(
                isinstance(bucket, bool) or not isinstance(bucket, int)
                for bucket in buckets_value
            ):
                raise ValueError(
                    "solution field 'length_bucketing."
                    "padding_length_buckets' must contain integers"
                )
            buckets = tuple(buckets_value)
        else:
            raise ValueError(
                "solution field 'length_bucketing.padding_length_buckets' "
                "must be an array or null"
            )
        return bool(value.get("enabled", True)), buckets
    if isinstance(value, bool):
        return value, None
    raise ValueError(
        "solution field 'length_bucketing' must be an object or boolean"
    )


def _batching_options(solution: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = solution.get("tokenizer", {})
    if not isinstance(tokenizer, Mapping):
        raise ValueError("solution field 'tokenizer' must be an object")
    batch_fields = tokenizer.get("batch_fields", {})
    if not isinstance(batch_fields, Mapping):
        raise ValueError("solution field 'tokenizer.batch_fields' must be an object")
    length_bucketing, padding_length_buckets = _length_bucketing_options(solution)
    return {
        "batch_size": int(solution.get("batch_size", 64)),
        "num_workers": int(solution.get("num_workers", 0)),
        "prefetch_factor": int(solution.get("prefetch_factor", 2)),
        "pin_memory": bool(solution.get("pin_memory", True)),
        "non_blocking_transfer": bool(solution.get("non_blocking_transfer", True)),
        "length_bucketing": length_bucketing,
        "padding_length_buckets": padding_length_buckets,
        "batch_fields": bool(batch_fields.get("enabled", False)),
        "field_chunk_size": int(batch_fields.get("chunk_size", 16_384)),
    }


def _artifact_path(model_directory: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else model_directory / path


def _build_pytorch_predictor(
    solution: Mapping[str, Any],
    model_directory: Path,
    batching: Mapping[str, Any],
) -> TransformerPredictor:
    torch_compile = solution.get("torch_compile", {})
    if not isinstance(torch_compile, Mapping):
        raise ValueError("solution field 'torch_compile' must be an object")
    return TransformerPredictor.load(
        model_directory,
        **batching,
        dtype=str(solution.get("dtype", "float32")),
        compile_enabled=bool(torch_compile.get("enabled", False)),
        compile_mode=str(torch_compile.get("mode", "reduce-overhead")),
        compile_dynamic=bool(torch_compile.get("dynamic", True)),
        device=solution.get("device"),
    )


def build_transformer_predictor(
    solution: Mapping[str, Any],
    solution_root: Path,
    *,
    usage: TransformerUsage,
) -> TransformerPredictor:
    """Build the configured backend behind one common predictor."""
    if usage not in {"classifier", "encoder"}:
        raise ValueError("Transformer usage must be classifier or encoder")
    model_value = solution.get("model_directory")
    if model_value is None or not str(model_value).strip():
        raise ValueError("solution field 'model_directory' must contain a path")
    model_path = Path(str(model_value)).expanduser()
    model_directory = (
        model_path if model_path.is_absolute() else solution_root / model_path
    )
    batching = _batching_options(solution)
    backend = str(solution.get("backend", "pytorch")).lower()
    if backend == "pytorch":
        return _build_pytorch_predictor(solution, model_directory, batching)
    if backend != "onnxruntime":
        raise ValueError(
            "solution field 'backend' must be 'pytorch' or 'onnxruntime'"
        )

    runtime = solution.get("onnxruntime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("solution field 'onnxruntime' must be an object")
    artifacts = solution.get("onnx_artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("solution field 'onnx_artifacts' must be an object")

    fallback = bool(runtime.get("fallback_to_pytorch", True))
    tokenizer = AutoTokenizer.from_pretrained(model_directory, use_fast=True)
    model_config = json.loads(
        (model_directory / "config.json").read_text(encoding="utf-8")
    )
    try:
        executor = OnnxRuntimeTransformerExecutor(
            model_directory=model_directory,
            model_config=model_config,
            provider=str(runtime.get("provider", "cuda")).lower(),
            device_id=int(runtime.get("device_id", 0)),
            io_binding=bool(runtime.get("io_binding", True)),
            graph_optimization=str(runtime.get("graph_optimization", "all")).lower(),
            classifier_path=_artifact_path(
                model_directory,
                artifacts.get("classifier_path"),
            ),
            encoder_path=_artifact_path(
                model_directory,
                artifacts.get("encoder_path"),
            ),
        )
        executor.validate(
            classifier=usage == "classifier",
            encoder=usage == "encoder",
        )
        return TransformerPredictor(tokenizer, executor, **batching)
    except (
        FileNotFoundError,
        OnnxRuntimeInitializationError,
        OnnxRuntimeUnavailableError,
    ) as error:
        if not fallback:
            raise
        logger.warning(
            "ONNX Runtime initialization failed; using PyTorch: {}",
            error,
        )
        return _build_pytorch_predictor(solution, model_directory, batching)


__all__ = ["TransformerUsage", "build_transformer_predictor"]
