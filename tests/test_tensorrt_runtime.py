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
    _EngineState,
    _torch_dtype,
)
from match.models.transformer.tensorrt_builder import TensorRTEngineBuilder
from match.models.transformer.tensorrt_cache import TensorRTEngineCache
from match.models.transformer.tensorrt_common import TensorRTInitializationError


class _FakeTensorRT:
    class TensorIOMode:
        INPUT = "input"
        OUTPUT = "output"

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

    def test_reuses_buffers_and_bindings_for_repeated_shape(self) -> None:
        executor = object.__new__(TensorRTTransformerExecutor)
        object.__setattr__(executor, "_trt", _FakeTensorRT)
        object.__setattr__(executor, "_device", torch.device("cpu"))
        names = ("input_ids", "attention_mask", "logits")
        engine = Mock()
        engine.num_io_tensors = len(names)
        engine.get_tensor_name.side_effect = lambda index: names[index]
        engine.get_tensor_mode.side_effect = lambda name: (
            _FakeTensorRT.TensorIOMode.OUTPUT
            if name == "logits"
            else _FakeTensorRT.TensorIOMode.INPUT
        )
        engine.get_tensor_dtype.side_effect = lambda name: (
            np.float32 if name == "logits" else np.int32
        )
        context = Mock()
        context.set_input_shape.return_value = True
        context.get_tensor_shape.return_value = (2, 1)
        context.set_tensor_address.return_value = True
        context.execute_async_v3.return_value = True
        state = _EngineState(engine, context)
        object.__setattr__(executor, "_states", {"classifier": state})
        stream = SimpleNamespace(cuda_stream=7, synchronize=Mock())
        batch = {
            "input_ids": torch.ones((2, 32), dtype=torch.int64),
            "attention_mask": torch.ones((2, 32), dtype=torch.int64),
        }

        with patch.object(torch.cuda, "current_stream", return_value=stream):
            first = executor._execute("classifier", batch, non_blocking=False)
            buffer_ids = {
                name: id(tensor)
                for name, tensor in next(iter(state.buffers.values())).device.items()
            }
            second = executor._execute("classifier", batch, non_blocking=False)

        self.assertEqual(first.shape, (2, 1))
        self.assertEqual(second.shape, (2, 1))
        self.assertEqual(len(state.buffers), 1)
        self.assertEqual(
            buffer_ids,
            {
                name: id(tensor)
                for name, tensor in next(iter(state.buffers.values())).device.items()
            },
        )
        self.assertEqual(context.set_input_shape.call_count, 2)
        self.assertEqual(context.set_tensor_address.call_count, 3)
        self.assertEqual(context.execute_async_v3.call_count, 2)
        self.assertEqual(stream.synchronize.call_count, 2)

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
