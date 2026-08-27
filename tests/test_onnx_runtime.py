import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from match.models.transformer.onnx_runtime import OnnxRuntimeTransformerExecutor


class _FakeSessionOptions:
    graph_optimization_level = None


class _FakeSession:
    def __init__(self, path, *, sess_options, providers):
        self.path = path
        self.options = sess_options
        self.providers = providers

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, inputs):
        assert output_names == ["logits"]
        batch_size = inputs["input_ids"].shape[0]
        return [np.zeros((batch_size, 2), dtype=np.float32)]


class _FakeOrt:
    class GraphOptimizationLevel:
        ORT_DISABLE_ALL = 0
        ORT_ENABLE_BASIC = 1
        ORT_ENABLE_EXTENDED = 2
        ORT_ENABLE_ALL = 3

    SessionOptions = _FakeSessionOptions
    InferenceSession = _FakeSession

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]


class OnnxRuntimePredictorTests(unittest.TestCase):
    def test_cpu_session_runs_torch_batches_without_pytorch_model(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "match.models.transformer.onnx_runtime._require_onnxruntime",
                return_value=_FakeOrt,
            ),
        ):
            root = Path(directory)
            onnx_dir = root / "onnx"
            onnx_dir.mkdir()
            classifier = onnx_dir / "classifier.onnx"
            classifier.write_bytes(b"graph")
            executor = OnnxRuntimeTransformerExecutor(
                model_directory=root,
                model_config={"hidden_size": 16, "match_max_length": 32},
                provider="cpu",
                io_binding=False,
            )

            session = executor._session("classifier")
            logits = executor._run_session(
                session,
                {
                    "input_ids": torch.ones((3, 8), dtype=torch.long),
                    "attention_mask": torch.ones((3, 8), dtype=torch.long),
                    "unused": torch.ones((3, 8), dtype=torch.long),
                },
                non_blocking=False,
            )

        self.assertEqual(executor.output_dim, 16)
        np.testing.assert_array_equal(logits, np.zeros((3, 2), dtype=np.float32))
        self.assertEqual(session.providers, ["CPUExecutionProvider"])
        self.assertEqual(session.options.graph_optimization_level, 3)

    def test_rejects_unavailable_gpu_provider(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "match.models.transformer.onnx_runtime._require_onnxruntime",
                return_value=_FakeOrt,
            ),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            OnnxRuntimeTransformerExecutor(
                model_directory=Path(directory),
                model_config={"hidden_size": 16},
                provider="cuda",
            )


if __name__ == "__main__":
    unittest.main()
