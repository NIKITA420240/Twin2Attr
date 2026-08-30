"""Build a Transformer predictor from a packaged solution manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from loguru import logger
from transformers import AutoTokenizer

from .executor import TransformerExecutor
from .onnx_runtime import (
    OnnxRuntimeInitializationError,
    OnnxRuntimeTransformerExecutor,
    OnnxRuntimeUnavailableError,
    OrtTensorRTProviderOptions,
)
from .predictor import TransformerPredictor
from .profile import TransformerRuntimeContract
from .tensorrt_common import (
    TensorRTInitializationError,
    TensorRTProfile,
    TensorRTUnavailableError,
)
from .tensorrt_runtime import TensorRTEngineOptions, TensorRTTransformerExecutor

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
    raise ValueError("solution field 'length_bucketing' must be an object or boolean")


def _batching_options(solution: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = solution.get("tokenizer", {})
    if not isinstance(tokenizer, Mapping):
        raise ValueError("solution field 'tokenizer' must be an object")
    batch_fields = tokenizer.get("batch_fields", {})
    if not isinstance(batch_fields, Mapping):
        raise ValueError("solution field 'tokenizer.batch_fields' must be an object")
    pair_encoding = solution.get("pair_encoding", {})
    if not isinstance(pair_encoding, Mapping):
        raise ValueError("solution field 'pair_encoding' must be an object")
    length_bucketing, padding_length_buckets = _length_bucketing_options(solution)
    def optional_int(name: str) -> int | None:
        value = pair_encoding.get(name)
        return None if value is None else int(value)

    options = {
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
    for name in (
        "max_length",
        "max_attribute_value_chars",
        "max_attribute_value_tokens",
    ):
        value = optional_int(name)
        if value is not None:
            options[name] = value
    return options


def _artifact_path(model_directory: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else model_directory / path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"solution field {name!r} must be an object")
    return value


def _profile(
    value: Mapping[str, Any],
    *,
    default_batch_size: int,
    default_lengths: tuple[int, ...],
    input_names: tuple[str, ...],
    name: str,
) -> TensorRTProfile:
    lengths_value = value.get("sequence_lengths", default_lengths)
    if not isinstance(lengths_value, (list, tuple)):
        raise ValueError(f"solution field '{name}.sequence_lengths' must be an array")
    try:
        return TensorRTProfile.from_sequence_lengths(
            lengths_value,
            min_batch_size=int(value.get("min_batch_size", 1)),
            opt_batch_size=int(value.get("opt_batch_size", default_batch_size)),
            max_batch_size=int(value.get("max_batch_size", default_batch_size)),
            input_names=input_names,
        )
    except ValueError as error:
        raise ValueError(f"invalid solution field '{name}': {error}") from error


def _model_input_names(
    tokenizer: Any,
    model_config: Mapping[str, Any],
    *,
    classifier: bool,
) -> tuple[str, ...]:
    names = ["input_ids", "attention_mask"]
    contract = TransformerRuntimeContract.from_config(model_config)
    if (
        not contract.output.uses_prompted_pairs
        and "token_type_ids" in getattr(tokenizer, "model_input_names", ())
    ):
        names.append("token_type_ids")
    if classifier and contract.typed_feature_names:
        names.append("typed_features")
    return tuple(names)


def _ort_tensorrt_options(
    runtime: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    model_directory: Path,
    tokenizer: Any,
    model_config: Mapping[str, Any],
    *,
    classifier: bool,
) -> OrtTensorRTProviderOptions:
    value = runtime.get("tensorrt")
    if value is None:
        return OrtTensorRTProviderOptions(
            fp16_enabled=str(artifacts.get("precision", "float32")) == "float16"
        )
    tensorrt = _mapping(value, "onnxruntime.tensorrt")
    engine_cache = _mapping(
        tensorrt.get("engine_cache", {}), "onnxruntime.tensorrt.engine_cache"
    )
    timing_cache = _mapping(
        tensorrt.get("timing_cache", {}), "onnxruntime.tensorrt.timing_cache"
    )
    profiles = _mapping(
        tensorrt.get("profiles", {}), "onnxruntime.tensorrt.profiles"
    )
    timing_path = timing_cache.get("path")
    return OrtTensorRTProviderOptions(
        engine_cache_enabled=bool(engine_cache.get("enabled", True)),
        engine_cache_path=_artifact_path(
            model_directory, engine_cache.get("path", "onnx/trt_cache")
        ),
        timing_cache_enabled=bool(timing_cache.get("enabled", True)),
        timing_cache_path=_artifact_path(model_directory, timing_path),
        profile=_profile(
            profiles,
            default_batch_size=2048,
            default_lengths=(64, 96, 128),
            input_names=_model_input_names(
                tokenizer,
                model_config,
                classifier=classifier,
            ),
            name="onnxruntime.tensorrt.profiles",
        ),
        fp16_enabled=str(artifacts.get("precision", "float32")) == "float16",
    )


def _validate_tensorrt_profile(
    profile: TensorRTProfile | None,
    batching: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    if profile is None:
        return
    batch_size = int(batching["batch_size"])
    if batch_size > profile.max_batch_size:
        raise ValueError(
            f"Transformer batch_size={batch_size} exceeds TensorRT profile "
            f"max_batch_size={profile.max_batch_size}"
        )
    model_max_length = TransformerRuntimeContract.from_config(
        model_config
    ).max_length
    if (
        model_max_length is not None
        and int(model_max_length) > profile.max_sequence_length
    ):
        raise ValueError(
            f"Transformer max_length={model_max_length} exceeds TensorRT "
            f"profile maximum sequence length={profile.max_sequence_length}"
        )
    if not bool(batching["length_bucketing"]) and profile.min_sequence_length > 1:
        raise ValueError(
            "TensorRT profile with minimum sequence length greater than 1 "
            "requires length_bucketing to be enabled"
        )
    buckets = batching.get("padding_length_buckets")
    if buckets and (
        buckets[0] < profile.min_sequence_length
        or buckets[-1] > profile.max_sequence_length
    ):
        raise ValueError(
            "length_bucketing padding lengths must fit inside the TensorRT "
            "sequence-length profile"
        )


def _lifecycle(executor: TransformerExecutor, usage: TransformerUsage) -> None:
    requested = {
        "classifier": usage == "classifier",
        "encoder": usage == "encoder",
    }
    started = perf_counter()
    executor.prepare(**requested)
    prepared = perf_counter()
    executor.warmup(**requested)
    warmed_up = perf_counter()
    logger.info(
        "Prepared Transformer backend: executor={}, prepare_seconds={:.3f}, "
        "warmup_seconds={:.3f}",
        type(executor).__name__,
        prepared - started,
        warmed_up - prepared,
    )


def _predictor(
    tokenizer: Any,
    executor: TransformerExecutor,
    batching: Mapping[str, Any],
) -> TransformerPredictor:
    return TransformerPredictor(tokenizer, executor, **batching)


def _build_pytorch_predictor(
    solution: Mapping[str, Any],
    model_directory: Path,
    batching: Mapping[str, Any],
) -> TransformerPredictor:
    torch_compile = solution.get("torch_compile", {})
    if not isinstance(torch_compile, Mapping):
        raise ValueError("solution field 'torch_compile' must be an object")
    attention = solution.get("attention")
    if attention is None:
        attention = {}
    if not isinstance(attention, Mapping):
        raise ValueError("solution field 'attention' must be an object")
    attention_kwargs = (
        {}
        if "attention" not in solution
        else {
            "attention_implementation": str(
                attention.get("implementation", "auto")
            )
        }
    )
    predictor = TransformerPredictor.load(
        model_directory,
        **batching,
        dtype=str(solution.get("dtype", "float32")),
        compile_enabled=bool(torch_compile.get("enabled", False)),
        compile_mode=str(torch_compile.get("mode", "reduce-overhead")),
        compile_dynamic=bool(torch_compile.get("dynamic", True)),
        device=solution.get("device"),
        **attention_kwargs,
    )
    return predictor


def _load_components(model_directory: Path) -> tuple[Any, dict[str, Any]]:
    model_config = json.loads(
        (model_directory / "config.json").read_text(encoding="utf-8")
    )
    runtime_contract = TransformerRuntimeContract.from_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        use_fast=True,
        trust_remote_code=runtime_contract.output.uses_prompted_pairs,
    )
    return tokenizer, model_config


def _build_onnx_predictor(
    solution: Mapping[str, Any],
    model_directory: Path,
    batching: Mapping[str, Any],
    *,
    usage: TransformerUsage,
    tokenizer: Any | None = None,
    model_config: dict[str, Any] | None = None,
    provider_override: str | None = None,
    fallback_to_pytorch: bool | None = None,
) -> TransformerPredictor:
    runtime = _mapping(solution.get("onnxruntime", {}), "onnxruntime")
    artifacts = _mapping(solution.get("onnx_artifacts", {}), "onnx_artifacts")
    if tokenizer is None or model_config is None:
        tokenizer, model_config = _load_components(model_directory)
    provider = provider_override or str(runtime.get("provider", "cuda")).lower()
    fallback = (
        bool(runtime.get("fallback_to_pytorch", True))
        if fallback_to_pytorch is None
        else fallback_to_pytorch
    )
    try:
        tensorrt = _ort_tensorrt_options(
            runtime,
            artifacts,
            model_directory,
            tokenizer,
            model_config,
            classifier=usage == "classifier",
        )
        if provider == "tensorrt":
            _validate_tensorrt_profile(tensorrt.profile, batching, model_config)
        executor = OnnxRuntimeTransformerExecutor(
            model_directory=model_directory,
            model_config=model_config,
            provider=provider,
            device_id=int(runtime.get("device_id", 0)),
            io_binding=bool(runtime.get("io_binding", True)),
            graph_optimization=str(runtime.get("graph_optimization", "all")).lower(),
            classifier_path=_artifact_path(
                model_directory, artifacts.get("classifier_path")
            ),
            encoder_path=_artifact_path(
                model_directory, artifacts.get("encoder_path")
            ),
            tensorrt=tensorrt,
        )
        _lifecycle(executor, usage)
        return _predictor(tokenizer, executor, batching)
    except (
        FileNotFoundError,
        OnnxRuntimeInitializationError,
        OnnxRuntimeUnavailableError,
    ) as error:
        if not fallback:
            raise
        logger.warning("ONNX Runtime initialization failed; using PyTorch: {}", error)
        return _build_pytorch_predictor(solution, model_directory, batching)


def _native_tensorrt_options(
    value: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    model_directory: Path,
    tokenizer: Any,
    model_config: Mapping[str, Any],
    *,
    classifier: bool,
) -> TensorRTEngineOptions:
    engine_cache = _mapping(value.get("engine_cache", {}), "tensorrt.engine_cache")
    timing_cache = _mapping(value.get("timing_cache", {}), "tensorrt.timing_cache")
    profiles = _mapping(value.get("profiles", {}), "tensorrt.profiles")
    timing_path = timing_cache.get("path")
    return TensorRTEngineOptions(
        device_id=int(value.get("device_id", 0)),
        engine_cache_enabled=bool(engine_cache.get("enabled", True)),
        engine_cache_path=_artifact_path(
            model_directory, engine_cache.get("path", "onnx/native_trt_cache")
        ),
        timing_cache_enabled=bool(timing_cache.get("enabled", True)),
        timing_cache_path=_artifact_path(model_directory, timing_path),
        workspace_size_bytes=int(float(value.get("workspace_size_gb", 8.0)) * 1024**3),
        builder_optimization_level=int(value.get("builder_optimization_level", 3)),
        profile=_profile(
            profiles,
            default_batch_size=512,
            default_lengths=(1, 256, 472),
            input_names=_model_input_names(
                tokenizer,
                model_config,
                classifier=classifier,
            ),
            name="tensorrt.profiles",
        ),
        fp16_enabled=str(artifacts.get("precision", "float32")) == "float16",
    )


def _build_native_tensorrt_predictor(
    solution: Mapping[str, Any],
    model_directory: Path,
    batching: Mapping[str, Any],
    *,
    usage: TransformerUsage,
) -> TransformerPredictor:
    runtime = _mapping(solution.get("tensorrt", {}), "tensorrt")
    artifacts = _mapping(solution.get("onnx_artifacts", {}), "onnx_artifacts")
    tokenizer, model_config = _load_components(model_directory)
    options = _native_tensorrt_options(
        runtime,
        artifacts,
        model_directory,
        tokenizer,
        model_config,
        classifier=usage == "classifier",
    )
    _validate_tensorrt_profile(options.profile, batching, model_config)
    try:
        executor = TensorRTTransformerExecutor(
            model_directory=model_directory,
            model_config=model_config,
            options=options,
            classifier_path=_artifact_path(
                model_directory, artifacts.get("classifier_path")
            ),
            encoder_path=_artifact_path(model_directory, artifacts.get("encoder_path")),
        )
        _lifecycle(executor, usage)
        return _predictor(tokenizer, executor, batching)
    except (
        FileNotFoundError,
        TensorRTInitializationError,
        TensorRTUnavailableError,
    ) as error:
        if not bool(runtime.get("fallback_to_onnxruntime", True)):
            raise
        logger.warning(
            "native TensorRT initialization failed; using ONNX Runtime CUDA: {}",
            error,
        )
        return _build_onnx_predictor(
            solution,
            model_directory,
            batching,
            usage=usage,
            tokenizer=tokenizer,
            model_config=model_config,
            provider_override="cuda",
            fallback_to_pytorch=False,
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
    if backend == "onnxruntime":
        return _build_onnx_predictor(
            solution, model_directory, batching, usage=usage
        )
    if backend == "tensorrt":
        return _build_native_tensorrt_predictor(
            solution, model_directory, batching, usage=usage
        )
    raise ValueError(
        "solution field 'backend' must be 'pytorch', 'onnxruntime' or 'tensorrt'"
    )


__all__ = ["TransformerUsage", "build_transformer_predictor"]
