"""Native TensorRT engine construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensorrt_cache import TensorRTEngineCache
from .tensorrt_common import TensorRTInitializationError, TensorRTProfile


@dataclass(slots=True)
class TensorRTEngineBuilder:
    trt: Any
    logger: Any
    profile: TensorRTProfile
    cache: TensorRTEngineCache
    workspace_size_bytes: int
    optimization_level: int
    fp16_enabled: bool

    def build(self, onnx_path: Path) -> bytes:
        try:
            return self._build(onnx_path)
        except TensorRTInitializationError:
            raise
        except Exception as error:
            raise TensorRTInitializationError(
                f"failed to build TensorRT engine from {onnx_path}: {error}"
            ) from error

    def _build(self, onnx_path: Path) -> bytes:
        builder = self.trt.Builder(self.logger)
        explicit_batch = 0
        flag = getattr(
            self.trt.NetworkDefinitionCreationFlag,
            "EXPLICIT_BATCH",
            None,
        )
        if flag is not None:
            explicit_batch = 1 << int(flag)
        network = builder.create_network(explicit_batch)
        parser = self.trt.OnnxParser(network, self.logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "; ".join(
                str(parser.get_error(index)) for index in range(parser.num_errors)
            )
            raise TensorRTInitializationError(
                f"failed to parse ONNX graph {onnx_path}: {errors}"
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            self.trt.MemoryPoolType.WORKSPACE,
            self.workspace_size_bytes,
        )
        config.builder_optimization_level = self.optimization_level
        if self.fp16_enabled:
            config.set_flag(self.trt.BuilderFlag.FP16)

        optimization_profile = builder.create_optimization_profile()
        has_dynamic_inputs = False
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shape = tuple(tensor.shape)
            if -1 not in shape:
                continue
            has_dynamic_inputs = True
            if not optimization_profile.set_shape(
                tensor.name,
                self.profile.resolve_shape(shape, "min"),
                self.profile.resolve_shape(shape, "opt"),
                self.profile.resolve_shape(shape, "max"),
            ):
                raise TensorRTInitializationError(
                    f"invalid optimization profile for input {tensor.name!r}"
                )
        if has_dynamic_inputs:
            config.add_optimization_profile(optimization_profile)

        timing_cache = None
        if self.cache.resolved_timing_path() is not None:
            timing_cache = config.create_timing_cache(self.cache.load_timing())
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise TensorRTInitializationError(
                f"TensorRT failed to build an engine from {onnx_path}"
            )
        if timing_cache is not None:
            current_cache = config.get_timing_cache()
            self.cache.store_timing(bytes(current_cache.serialize()))
        return bytes(serialized)


__all__ = ["TensorRTEngineBuilder"]
