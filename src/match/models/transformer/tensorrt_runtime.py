"""Native TensorRT implementation of the Transformer execution contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...pair_encoding import DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS
from .executor import TransformerExecutorOutOfMemoryError
from .onnx_export import CLASSIFIER_ONNX_NAME, ENCODER_ONNX_NAME, ONNX_DIRECTORY_NAME
from .tensorrt_builder import TensorRTEngineBuilder
from .tensorrt_cache import TensorRTEngineCache
from .tensorrt_common import (
    TensorRTInitializationError,
    TensorRTProfile,
    TensorRTUnavailableError,
)


@dataclass(frozen=True, slots=True)
class TensorRTEngineOptions:
    device_id: int = 0
    engine_cache_enabled: bool = True
    engine_cache_path: Path | None = None
    timing_cache_enabled: bool = True
    timing_cache_path: Path | None = None
    workspace_size_bytes: int = 8 * 1024**3
    builder_optimization_level: int = 3
    profile: TensorRTProfile = TensorRTProfile()
    fp16_enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or self.device_id < 0:
            raise ValueError("TensorRT device_id must not be negative")
        if self.workspace_size_bytes < 1:
            raise ValueError("TensorRT workspace size must be positive")
        if self.builder_optimization_level not in range(6):
            raise ValueError("TensorRT builder optimization level must be 0..5")


def _require_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise TensorRTUnavailableError(
            "native TensorRT inference requires NVIDIA TensorRT Python bindings"
        ) from error
    return trt


def _is_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


def _torch_dtype(trt: Any, dtype: Any) -> torch.dtype:
    numpy_dtype = np.dtype(trt.nptype(dtype))
    try:
        return {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.bool_): torch.bool,
        }[numpy_dtype]
    except KeyError as error:
        raise TypeError(f"unsupported TensorRT tensor dtype: {dtype}") from error


@dataclass(slots=True)
class _EngineState:
    engine: Any
    context: Any


@dataclass(slots=True)
class TensorRTTransformerExecutor:
    model_directory: Path
    model_config: dict[str, Any]
    options: TensorRTEngineOptions = TensorRTEngineOptions()
    classifier_path: Path | None = None
    encoder_path: Path | None = None
    _trt: Any = field(init=False, repr=False)
    _logger: Any = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _cache: TensorRTEngineCache = field(init=False, repr=False)
    _builder: TensorRTEngineBuilder = field(init=False, repr=False)
    _states: dict[str, _EngineState] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._trt = _require_tensorrt()
        self._logger = self._trt.Logger(self._trt.Logger.WARNING)
        self._device = torch.device("cuda", self.options.device_id)
        try:
            with torch.cuda.device(self._device):
                torch.cuda.current_stream(self._device)
        except RuntimeError as error:
            raise TensorRTUnavailableError(
                f"CUDA device {self.options.device_id} is unavailable"
            ) from error
        onnx_dir = self.model_directory / ONNX_DIRECTORY_NAME
        self.classifier_path = self.classifier_path or (
            onnx_dir / CLASSIFIER_ONNX_NAME
        )
        self.encoder_path = self.encoder_path or (onnx_dir / ENCODER_ONNX_NAME)
        self._cache = TensorRTEngineCache(
            model_directory=self.model_directory,
            enabled=self.options.engine_cache_enabled,
            directory=self.options.engine_cache_path,
            timing_enabled=self.options.timing_cache_enabled,
            timing_path=self.options.timing_cache_path,
        )
        self._builder = TensorRTEngineBuilder(
            trt=self._trt,
            logger=self._logger,
            profile=self.options.profile,
            cache=self._cache,
            workspace_size_bytes=self.options.workspace_size_bytes,
            optimization_level=self.options.builder_optimization_level,
            fp16_enabled=self.options.fp16_enabled,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def output_dim(self) -> int:
        value = self.model_config.get("hidden_size") or self.model_config.get(
            "backbone_config", {}
        ).get("hidden_size")
        if not value:
            raise ValueError("Transformer config does not expose hidden_size")
        return int(value)

    @property
    def max_length(self) -> int | None:
        value = self.model_config.get("match_max_length")
        return None if value is None else int(value)

    @property
    def use_field_tokens(self) -> bool:
        return bool(self.model_config.get("match_use_field_tokens", False))

    @property
    def max_attribute_value_chars(self) -> int | None:
        return self.model_config.get("match_max_attribute_value_chars")

    @property
    def max_attribute_value_tokens(self) -> int | None:
        return self.model_config.get(
            "match_max_attribute_value_tokens",
            DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        )

    def _cache_identity(self, kind: str) -> dict[str, Any]:
        properties = torch.cuda.get_device_properties(self.options.device_id)
        return {
            "kind": kind,
            "tensorrt": self._trt.__version__,
            "device": properties.name,
            "capability": [properties.major, properties.minor],
            "workspace": self.options.workspace_size_bytes,
            "optimization_level": self.options.builder_optimization_level,
            "batch": self.options.profile.batch_sizes,
            "sequence": self.options.profile.sequence_lengths,
            "inputs": self.options.profile.input_names,
            "fp16": self.options.fp16_enabled,
        }

    def _onnx_path(self, kind: str) -> Path:
        path = self.classifier_path if kind == "classifier" else self.encoder_path
        if path is None or not path.is_file():
            raise FileNotFoundError(f"TensorRT source ONNX graph is missing: {path}")
        return path

    def _deserialize(self, serialized: bytes) -> Any | None:
        try:
            runtime = self._trt.Runtime(self._logger)
            return runtime.deserialize_cuda_engine(serialized)
        except Exception as error:
            raise TensorRTInitializationError(
                f"failed to deserialize TensorRT engine: {error}"
            ) from error

    def _load_state(self, kind: str) -> _EngineState:
        try:
            onnx_path = self._onnx_path(kind)
            engine_path = self._cache.engine_path(
                onnx_path,
                kind,
                self._cache_identity(kind),
            )
            serialized = self._cache.load_engine(engine_path)
            engine = None
            if serialized is not None:
                try:
                    engine = self._deserialize(serialized)
                except TensorRTInitializationError:
                    # A stale or corrupted cache is recoverable from the ONNX graph.
                    engine = None
            if engine is None:
                serialized = self._builder.build(onnx_path)
                engine = self._deserialize(serialized)
                if engine is None:
                    raise TensorRTInitializationError(
                        "failed to deserialize newly built TensorRT engine"
                    )
                self._cache.store_engine(engine_path, serialized)
            context = engine.create_execution_context()
            if context is None:
                raise TensorRTInitializationError(
                    "failed to create TensorRT execution context"
                )
            return _EngineState(engine, context)
        except (FileNotFoundError, TensorRTInitializationError):
            raise
        except Exception as error:
            raise TensorRTInitializationError(
                f"failed to prepare native TensorRT {kind} engine: {error}"
            ) from error

    def _state(self, kind: str) -> _EngineState:
        if kind not in self._states:
            raise RuntimeError(
                f"TensorRT {kind} engine is not prepared; call prepare() first"
            )
        return self._states[kind]

    def _execute(
        self,
        kind: str,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        state = self._state(kind)
        engine, context = state.engine, state.context
        stream = torch.cuda.current_stream(self.device)
        buffers: dict[str, torch.Tensor] = {}
        try:
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                if engine.get_tensor_mode(name) != self._trt.TensorIOMode.INPUT:
                    continue
                if name not in batch:
                    raise RuntimeError(f"TensorRT batch is missing input {name!r}")
                tensor = batch[name].to(
                    device=self.device,
                    dtype=_torch_dtype(self._trt, engine.get_tensor_dtype(name)),
                    non_blocking=non_blocking,
                ).contiguous()
                if not context.set_input_shape(name, tuple(tensor.shape)):
                    raise RuntimeError(
                        "TensorRT rejected input shape "
                        f"{tuple(tensor.shape)} for {name}"
                    )
                buffers[name] = tensor
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                if engine.get_tensor_mode(name) != self._trt.TensorIOMode.OUTPUT:
                    continue
                shape = tuple(context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(
                        f"TensorRT could not resolve output shape for {name}: {shape}"
                    )
                buffers[name] = torch.empty(
                    shape,
                    dtype=_torch_dtype(self._trt, engine.get_tensor_dtype(name)),
                    device=self.device,
                )
            for name, tensor in buffers.items():
                if not context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT rejected buffer for {name!r}")
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            outputs = [
                buffers[engine.get_tensor_name(index)]
                for index in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(index))
                == self._trt.TensorIOMode.OUTPUT
            ]
            if len(outputs) != 1:
                raise RuntimeError(
                    f"expected one TensorRT output, found {len(outputs)}"
                )
            return outputs[0].float().cpu().numpy()
        except Exception as error:
            if _is_oom(error):
                raise TransformerExecutorOutOfMemoryError from error
            raise

    def predict_logits(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        values = self._execute("classifier", batch, non_blocking=non_blocking)
        if values.ndim != 2 or values.shape[1] != 2:
            raise RuntimeError(
                f"TensorRT classifier returned shape {values.shape}, "
                "expected [batch, 2]"
            )
        return values.astype(np.float32, copy=False)

    def encode_cls(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        values = self._execute("encoder", batch, non_blocking=non_blocking)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise RuntimeError(
                f"TensorRT encoder returned shape {values.shape}, expected "
                f"[batch, {self.output_dim}]"
            )
        return values.astype(np.float32, copy=False)

    def prepare(self, *, classifier: bool, encoder: bool) -> None:
        if classifier and "classifier" not in self._states:
            self._states["classifier"] = self._load_state("classifier")
        if encoder and "encoder" not in self._states:
            self._states["encoder"] = self._load_state("encoder")

    def _warmup_kind(self, kind: str) -> None:
        state = self._state(kind)
        batch_size = min(8, self.options.profile.opt_batch_size)
        sequence_length = self.options.profile.opt_sequence_length
        batch: dict[str, torch.Tensor] = {}
        for index in range(state.engine.num_io_tensors):
            name = state.engine.get_tensor_name(index)
            if state.engine.get_tensor_mode(name) != self._trt.TensorIOMode.INPUT:
                continue
            declared_shape = tuple(state.engine.get_tensor_shape(name))
            shape = tuple(
                dimension
                if dimension > 0
                else (batch_size if axis == 0 else sequence_length)
                for axis, dimension in enumerate(declared_shape)
            )
            batch[name] = torch.zeros(
                shape,
                dtype=_torch_dtype(self._trt, state.engine.get_tensor_dtype(name)),
            )
        self._execute(kind, batch, non_blocking=False)

    def warmup(self, *, classifier: bool, encoder: bool) -> None:
        try:
            if classifier:
                self._warmup_kind("classifier")
            if encoder:
                self._warmup_kind("encoder")
        except TensorRTInitializationError:
            raise
        except Exception as error:
            raise TensorRTInitializationError(
                f"native TensorRT warmup failed: {error}"
            ) from error

    def clear_cache(self) -> None:
        torch.cuda.empty_cache()


__all__ = [
    "TensorRTEngineOptions",
    "TensorRTInitializationError",
    "TensorRTProfile",
    "TensorRTTransformerExecutor",
    "TensorRTUnavailableError",
]
