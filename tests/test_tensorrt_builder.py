import tempfile
import unittest
from pathlib import Path

from match.models.transformer.tensorrt_builder import TensorRTEngineBuilder
from match.models.transformer.tensorrt_common import (
    TensorRTInitializationError,
    TensorRTProfile,
)


class _Parser:
    def __init__(self, result=False) -> None:
        self.calls = []
        self.num_errors = 1
        self.result = result

    def parse_from_file(self, path):
        self.calls.append(path)
        return self.result

    def get_error(self, index):
        return "external initializer is unavailable"


class _Tensor:
    def __init__(self, name):
        self.name = name
        self.shape = (-1, -1)


class _Network:
    def __init__(self):
        self.inputs = [_Tensor("input_ids"), _Tensor("attention_mask")]

    @property
    def num_inputs(self):
        return len(self.inputs)

    def get_input(self, index):
        return self.inputs[index]


class _OptimizationProfile:
    def __init__(self):
        self.calls = []

    def set_shape(self, name, minimum, optimum, maximum):
        self.calls.append((name, minimum, optimum, maximum))
        return None


class _Config:
    def __init__(self):
        self.builder_optimization_level = None
        self.profiles = []

    def set_memory_pool_limit(self, pool, size):
        self.workspace = (pool, size)

    def set_flag(self, flag):
        self.flag = flag

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)


class _NativeBuilder:
    def __init__(self):
        self.network = _Network()
        self.profile = _OptimizationProfile()
        self.config = _Config()

    def create_network(self, flags):
        return self.network

    def create_builder_config(self):
        return self.config

    def create_optimization_profile(self):
        return self.profile

    def build_serialized_network(self, network, config):
        return b"serialized-engine"


class _Cache:
    def resolved_timing_path(self):
        return None


class _FakeTensorRT:
    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = None

    class Logger:
        WARNING = 2

    class MemoryPoolType:
        WORKSPACE = "workspace"

    class BuilderFlag:
        FP16 = "fp16"

    def __init__(self, *, parser_result=False) -> None:
        self.parser = _Parser(parser_result)
        self.builder = _NativeBuilder()

    def Builder(self, logger):
        return self.builder

    def OnnxParser(self, network, logger):
        return self.parser


class TensorRTEngineBuilderTests(unittest.TestCase):
    @staticmethod
    def _builder(runtime):
        return TensorRTEngineBuilder(
            trt=runtime,
            logger=object(),
            profile=TensorRTProfile(
                min_batch_size=1,
                opt_batch_size=128,
                max_batch_size=128,
                min_sequence_length=32,
                opt_sequence_length=96,
                max_sequence_length=128,
            ),
            cache=_Cache(),
            workspace_size_bytes=1024,
            optimization_level=5,
            fp16_enabled=True,
        )

    def test_parses_onnx_from_file_to_resolve_external_weights(self) -> None:
        runtime = _FakeTensorRT()
        builder = self._builder(runtime)
        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "classifier.onnx"
            onnx_path.write_bytes(b"external-data-model")
            with self.assertRaisesRegex(
                TensorRTInitializationError,
                "external initializer is unavailable",
            ):
                builder.build(onnx_path)

        self.assertEqual(runtime.parser.calls, [str(onnx_path)])

    def test_accepts_none_returned_by_set_shape(self) -> None:
        runtime = _FakeTensorRT(parser_result=True)
        builder = self._builder(runtime)
        with tempfile.TemporaryDirectory() as directory:
            onnx_path = Path(directory) / "classifier.onnx"
            onnx_path.write_bytes(b"external-data-model")
            result = builder.build(onnx_path)

        self.assertEqual(result, b"serialized-engine")
        self.assertEqual(
            runtime.builder.profile.calls,
            [
                ("input_ids", (1, 32), (128, 96), (128, 128)),
                ("attention_mask", (1, 32), (128, 96), (128, 128)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
