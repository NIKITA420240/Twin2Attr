import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    @staticmethod
    def nptype(value):
        return value


class NativeTensorRTRuntimeTests(unittest.TestCase):
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

    def test_builder_accepts_none_returned_by_set_shape(self) -> None:
        tensor = SimpleNamespace(name="input_ids", shape=(-1, -1))
        network = SimpleNamespace(
            num_inputs=1,
            get_input=lambda _index: tensor,
        )
        parser = SimpleNamespace(parse=lambda _value: True, num_errors=0)
        config = Mock()
        optimization_profile = Mock()
        optimization_profile.set_shape.return_value = None
        native_builder = Mock()
        native_builder.create_network.return_value = network
        native_builder.create_builder_config.return_value = config
        native_builder.create_optimization_profile.return_value = optimization_profile
        native_builder.build_serialized_network.return_value = b"engine"
        trt = SimpleNamespace(
            Builder=Mock(return_value=native_builder),
            OnnxParser=Mock(return_value=parser),
            NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
            MemoryPoolType=SimpleNamespace(WORKSPACE=object()),
            BuilderFlag=SimpleNamespace(FP16=object()),
        )

        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "classifier.onnx"
            onnx_path.write_bytes(b"onnx")
            builder = TensorRTEngineBuilder(
                trt=trt,
                logger=object(),
                profile=TensorRTProfile(),
                cache=TensorRTEngineCache(
                    Path(directory),
                    enabled=False,
                    timing_enabled=False,
                ),
                workspace_size_bytes=1,
                optimization_level=0,
                fp16_enabled=False,
            )

            result = builder.build(onnx_path)

        self.assertEqual(result, b"engine")
        optimization_profile.set_shape.assert_called_once_with(
            "input_ids",
            (1, 1),
            (512, 256),
            (512, 472),
        )


if __name__ == "__main__":
    unittest.main()
