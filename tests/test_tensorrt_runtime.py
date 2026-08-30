import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from match.models.transformer.tensorrt_runtime import (
    TensorRTEngineOptions,
    TensorRTProfile,
    TensorRTTransformerExecutor,
    _torch_dtype,
)
from match.models.transformer.tensorrt_builder import TensorRTEngineBuilder
from match.models.transformer.tensorrt_cache import TensorRTEngineCache
from match.models.transformer.tensorrt_common import TensorRTInitializationError


class _FakeTensorRT:
    class TensorIOMode:
        INPUT = "input"

    @staticmethod
    def nptype(value):
        return value


class NativeTensorRTRuntimeTests(unittest.TestCase):
    def test_warmup_uses_small_optimal_profile_shape(self) -> None:
        executor = object.__new__(TensorRTTransformerExecutor)
        object.__setattr__(
            executor,
            "options",
            TensorRTEngineOptions(
                profile=TensorRTProfile(
                    min_batch_size=1,
                    opt_batch_size=256,
                    max_batch_size=256,
                    min_sequence_length=32,
                    opt_sequence_length=96,
                    max_sequence_length=128,
                )
            ),
        )
        object.__setattr__(executor, "_trt", _FakeTensorRT)
        engine = Mock()
        engine.num_io_tensors = 2
        engine.get_tensor_name.side_effect = ["input_ids", "attention_mask"]
        engine.get_tensor_mode.return_value = _FakeTensorRT.TensorIOMode.INPUT
        engine.get_tensor_shape.return_value = (-1, -1)
        engine.get_tensor_dtype.return_value = np.int32
        object.__setattr__(
            executor,
            "_states",
            {"classifier": SimpleNamespace(engine=engine, context=object())},
        )
        observed = {}

        def execute(instance, kind, batch, *, non_blocking):
            self.assertIs(instance, executor)
            observed.update({name: tuple(value.shape) for name, value in batch.items()})

        with patch.object(
            TensorRTTransformerExecutor,
            "_execute",
            autospec=True,
            side_effect=execute,
        ):
            executor._warmup_kind("classifier")

        self.assertEqual(
            observed,
            {"input_ids": (8, 96), "attention_mask": (8, 96)},
        )

    def test_profile_replaces_only_dynamic_batch_and_sequence_dimensions(self) -> None:
        profile = TensorRTProfile(
            min_batch_size=1,
            opt_batch_size=256,
            max_batch_size=512,
            min_sequence_length=1,
            opt_sequence_length=256,
            max_sequence_length=472,
        )

        self.assertEqual(profile.resolve_shape((-1, -1), "min"), (1, 1))
        self.assertEqual(profile.resolve_shape((-1, -1), "opt"), (256, 256))
        self.assertEqual(profile.resolve_shape((-1, -1), "max"), (512, 472))
        self.assertEqual(profile.resolve_shape((-1, 32), "max"), (512, 32))

    def test_rejects_invalid_profiles_and_builder_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch profile"):
            TensorRTProfile(
                min_batch_size=4,
                opt_batch_size=2,
                max_batch_size=8,
            )
        with self.assertRaisesRegex(ValueError, "0..5"):
            TensorRTEngineOptions(builder_optimization_level=6)

    def test_maps_tensorrt_dtypes_to_torch(self) -> None:
        self.assertEqual(_torch_dtype(_FakeTensorRT, np.float16), torch.float16)
        self.assertEqual(_torch_dtype(_FakeTensorRT, np.int32), torch.int32)

    def test_engine_and_timing_cache_have_separate_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "classifier.onnx"
            onnx_path.write_bytes(b"onnx")
            cache = TensorRTEngineCache(root)
            engine_path = cache.engine_path(onnx_path, "classifier", {"gpu": "H100"})

            self.assertIsNotNone(engine_path)
            cache.store_engine(engine_path, b"engine")
            cache.store_timing(b"timing")

            self.assertEqual(cache.load_engine(engine_path), b"engine")
            self.assertEqual(cache.load_timing(), b"timing")

    def test_builder_normalizes_backend_errors(self) -> None:
        builder = TensorRTEngineBuilder(
            trt=object(),
            logger=object(),
            profile=TensorRTProfile(),
            cache=TensorRTEngineCache(Path("/tmp/model"), enabled=False),
            workspace_size_bytes=1,
            optimization_level=0,
            fp16_enabled=False,
        )
        with (
            patch.object(
                TensorRTEngineBuilder,
                "_build",
                side_effect=RuntimeError("builder failed"),
            ),
            self.assertRaisesRegex(TensorRTInitializationError, "builder failed"),
        ):
            builder.build(Path("model.onnx"))


if __name__ == "__main__":
    unittest.main()
