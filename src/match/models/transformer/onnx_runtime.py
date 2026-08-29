"""ONNX Runtime implementation of the Transformer execution contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .executor import TransformerExecutorOutOfMemoryError
from .onnx_export import CLASSIFIER_ONNX_NAME, ENCODER_ONNX_NAME, ONNX_DIRECTORY_NAME
from .profile import TransformerArtifactContract, TransformerRuntimeContract
from .tensorrt_common import TensorRTProfile


class OnnxRuntimeUnavailableError(RuntimeError):
    """The requested ONNX Runtime installation or provider is unavailable."""


class OnnxRuntimeInitializationError(RuntimeError):
    """An exported graph could not be initialized by ONNX Runtime."""


@dataclass(frozen=True, slots=True)
class OrtTensorRTProviderOptions:
    engine_cache_enabled: bool = True
    engine_cache_path: Path | None = None
    timing_cache_enabled: bool = True
    timing_cache_path: Path | None = None
    profile: TensorRTProfile | None = None
    fp16_enabled: bool = False


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise OnnxRuntimeUnavailableError(
            "ONNX Runtime inference requires the optional dependencies; "
            "install the project with the 'onnx-cpu' or 'onnx-gpu' extra"
        ) from error
    return ort


def _graph_optimization_level(ort: Any, value: str) -> Any:
    return {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[value]


def _is_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


def _numpy_dtype(dtype: torch.dtype) -> type[np.generic]:
    try:
        return {
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bool: np.bool_,
        }[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported ONNX input dtype: {dtype}") from error


@dataclass(slots=True)
class OnnxRuntimeTransformerExecutor:
    model_directory: Path
    model_config: dict[str, Any]
    provider: str = "cuda"
    device_id: int = 0
    io_binding: bool = True
    graph_optimization: str = "all"
    disabled_optimizers: tuple[str, ...] = ()
    classifier_path: Path | None = None
    encoder_path: Path | None = None
    tensorrt: OrtTensorRTProviderOptions = OrtTensorRTProviderOptions()
    _ort: Any = field(init=False, repr=False)
    _providers: list[Any] = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _classifier_session: Any | None = field(init=False, default=None, repr=False)
    _encoder_session: Any | None = field(init=False, default=None, repr=False)
    _runtime_contract: TransformerRuntimeContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._runtime_contract = TransformerRuntimeContract.from_config(
            self.model_config
        )
        if self.provider not in {"cpu", "cuda", "tensorrt"}:
            raise ValueError("ONNX Runtime provider must be cpu, cuda or tensorrt")
        if self.device_id < 0:
            raise ValueError("ONNX Runtime device_id must not be negative")
        if self.graph_optimization not in {
            "disabled",
            "basic",
            "extended",
            "all",
        }:
            raise ValueError("unsupported ONNX graph optimization level")

        self._ort = _require_onnxruntime()
        requested = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }[self.provider]
        available = set(self._ort.get_available_providers())
        if requested not in available:
            raise OnnxRuntimeUnavailableError(
                f"requested ONNX Runtime provider {requested!r} is unavailable; "
                f"available providers: {sorted(available)}"
            )
        self._providers, self._device = self._provider_configuration(requested)
        onnx_dir = self.model_directory / ONNX_DIRECTORY_NAME
        self.classifier_path = self.classifier_path or (
            onnx_dir / CLASSIFIER_ONNX_NAME
        )
        self.encoder_path = self.encoder_path or (onnx_dir / ENCODER_ONNX_NAME)

    def _provider_configuration(
        self,
        requested: str,
    ) -> tuple[list[Any], torch.device]:
        if self.provider == "cpu":
            return [requested], torch.device("cpu")
        try:
            compute_stream = str(
                torch.cuda.current_stream(self.device_id).cuda_stream
            )
        except RuntimeError as error:
            raise OnnxRuntimeUnavailableError(
                f"CUDA device {self.device_id} is unavailable to PyTorch"
            ) from error
        cuda_options = {
            "device_id": self.device_id,
            "user_compute_stream": compute_stream,
            "do_copy_in_default_stream": True,
        }
        device = torch.device("cuda", self.device_id)
        if self.provider == "cuda":
            return [
                (requested, cuda_options),
                "CPUExecutionProvider",
            ], device

        default_cache_path = (
            self.model_directory / ONNX_DIRECTORY_NAME / "trt_cache"
        )
        engine_cache_path = self.tensorrt.engine_cache_path or default_cache_path
        timing_cache_path = self.tensorrt.timing_cache_path or engine_cache_path
        try:
            if self.tensorrt.engine_cache_enabled:
                engine_cache_path.mkdir(parents=True, exist_ok=True)
            if self.tensorrt.timing_cache_enabled:
                timing_cache_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OnnxRuntimeInitializationError(
                f"failed to create TensorRT cache directory: {error}"
            ) from error
        tensorrt_options = {
            "device_id": self.device_id,
            "user_compute_stream": compute_stream,
            "trt_fp16_enable": self.tensorrt.fp16_enabled,
            "trt_engine_cache_enable": self.tensorrt.engine_cache_enabled,
            "trt_engine_cache_path": str(engine_cache_path),
            "trt_timing_cache_enable": self.tensorrt.timing_cache_enabled,
            "trt_timing_cache_path": str(timing_cache_path),
        }
        tensorrt_options.update(self._profile_options())
        return [
            (requested, tensorrt_options),
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ], device

    def _profile_options(self) -> dict[str, str]:
        profile = self.tensorrt.profile
        if profile is None:
            return {}

        def shapes(batch_size: int, sequence_length: int) -> str:
            return ",".join(
                f"{name}:{batch_size}x{sequence_length}"
                for name in profile.input_names
            )

        return {
            "trt_profile_min_shapes": shapes(
                profile.min_batch_size,
                profile.min_sequence_length,
            ),
            "trt_profile_opt_shapes": shapes(
                profile.opt_batch_size,
                profile.opt_sequence_length,
            ),
            "trt_profile_max_shapes": shapes(
                profile.max_batch_size,
                profile.max_sequence_length,
            ),
        }

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def output_dim(self) -> int:
        return self.runtime_contract.hidden_size

    @property
    def num_logits(self) -> int:
        return self.output_contract.num_logits

    @property
    def profile(self) -> str:
        return self.output_contract.profile

    @property
    def output_contract(self) -> TransformerArtifactContract:
        return self.runtime_contract.output

    @property
    def runtime_contract(self) -> TransformerRuntimeContract:
        return self._runtime_contract

    @property
    def max_length(self) -> int | None:
        return self.runtime_contract.max_length

    @property
    def use_field_tokens(self) -> bool:
        return self.runtime_contract.use_field_tokens

    @property
    def max_attribute_value_chars(self) -> int | None:
        return self.runtime_contract.max_attribute_value_chars

    @property
    def max_attribute_value_tokens(self) -> int | None:
        return self.runtime_contract.max_attribute_value_tokens

    def _new_session(self, path: Path | None) -> Any:
        if path is None or not path.is_file():
            raise FileNotFoundError(f"ONNX graph is missing: {path}")
        options = self._ort.SessionOptions()
        options.graph_optimization_level = _graph_optimization_level(
            self._ort,
            self.graph_optimization,
        )
        try:
            session_kwargs: dict[str, Any] = {}
            if self.disabled_optimizers:
                session_kwargs["disabled_optimizers"] = set(
                    self.disabled_optimizers
                )
            return self._ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=self._providers,
                **session_kwargs,
            )
        except Exception as error:
            raise OnnxRuntimeInitializationError(
                f"failed to initialize ONNX graph {path}: {error}"
            ) from error

    def _session(self, kind: str) -> Any:
        if kind == "classifier":
            if self._classifier_session is None:
                self._classifier_session = self._new_session(self.classifier_path)
            return self._classifier_session
        if kind == "encoder":
            if self._encoder_session is None:
                self._encoder_session = self._new_session(self.encoder_path)
            return self._encoder_session
        raise ValueError(f"unknown ONNX graph kind: {kind}")

    def _run_session(
        self,
        session: Any,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        input_names = {value.name for value in session.get_inputs()}
        inputs = {name: value for name, value in batch.items() if name in input_names}
        missing = input_names - inputs.keys()
        if missing:
            raise RuntimeError(f"ONNX batch is missing inputs: {sorted(missing)}")
        try:
            if self.io_binding and self.device.type == "cuda":
                return self._run_with_io_binding(
                    session,
                    inputs,
                    non_blocking=non_blocking,
                )
            numpy_inputs = {
                name: value.detach().cpu().numpy()
                for name, value in inputs.items()
            }
            output_name = session.get_outputs()[0].name
            return np.asarray(session.run([output_name], numpy_inputs)[0])
        except Exception as error:
            if _is_oom(error):
                raise TransformerExecutorOutOfMemoryError from error
            raise

    def _run_with_io_binding(
        self,
        session: Any,
        inputs: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        binding = session.io_binding()
        gpu_inputs: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            tensor = value.to(
                self.device,
                non_blocking=non_blocking,
            ).contiguous()
            gpu_inputs[name] = tensor
            binding.bind_input(
                name,
                "cuda",
                self.device_id,
                _numpy_dtype(tensor.dtype),
                tuple(tensor.shape),
                tensor.data_ptr(),
            )
        output_name = session.get_outputs()[0].name
        binding.bind_output(output_name, "cuda", self.device_id)
        session.run_with_iobinding(binding)
        return np.asarray(binding.copy_outputs_to_cpu()[0])

    def predict_logits(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        values = self._run_session(
            self._session("classifier"),
            batch,
            non_blocking=non_blocking,
        )
        if values.ndim != 2 or values.shape[1] != self.num_logits:
            raise RuntimeError(
                f"ONNX classifier returned shape {values.shape}, expected "
                f"[batch, {self.num_logits}]"
            )
        return values.astype(np.float32, copy=False)

    def encode_pooled(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        values = self._run_session(
            self._session("encoder"),
            batch,
            non_blocking=non_blocking,
        )
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise RuntimeError(
                f"ONNX encoder returned shape {values.shape}, expected "
                f"[batch, {self.output_dim}]"
            )
        return values.astype(np.float32, copy=False)

    def encode_cls(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        """Compatibility alias for the profile-specific pooled encoder."""
        return self.encode_pooled(batch, non_blocking=non_blocking)

    def prepare(self, *, classifier: bool, encoder: bool) -> None:
        if classifier:
            self._session("classifier")
        if encoder:
            self._session("encoder")

    def _warmup_session(self, kind: str) -> None:
        session = self._session(kind)
        profile = self.tensorrt.profile if self.provider == "tensorrt" else None
        sequence_length = profile.opt_sequence_length if profile is not None else 8
        batch: dict[str, torch.Tensor] = {}
        dtypes = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bool)": torch.bool,
        }
        for value in session.get_inputs():
            try:
                dtype = dtypes[value.type]
            except KeyError as error:
                raise OnnxRuntimeInitializationError(
                    f"unsupported ONNX warmup input type {value.type!r}"
                ) from error
            declared_shape = getattr(value, "shape", (None, None))
            shape = tuple(
                dimension
                if isinstance(dimension, int) and dimension > 0
                else (1 if index == 0 else sequence_length)
                for index, dimension in enumerate(declared_shape)
            )
            batch[value.name] = torch.zeros(shape, dtype=dtype)
        self._run_session(session, batch, non_blocking=False)

    def warmup(self, *, classifier: bool, encoder: bool) -> None:
        try:
            if classifier:
                self._warmup_session("classifier")
            if encoder:
                self._warmup_session("encoder")
        except OnnxRuntimeInitializationError:
            raise
        except Exception as error:
            raise OnnxRuntimeInitializationError(
                f"ONNX Runtime warmup failed: {error}"
            ) from error

    def clear_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "OnnxRuntimeInitializationError",
    "OnnxRuntimeTransformerExecutor",
    "OnnxRuntimeUnavailableError",
    "OrtTensorRTProviderOptions",
]
