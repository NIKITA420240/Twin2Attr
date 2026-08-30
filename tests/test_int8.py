import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from match.models.transformer.int8 import (
    Int8WeightEmbedding,
    Int8WeightLinear,
    quantize_module_weights_to_int8,
    replace_module_weights_with_int8_shells,
)


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 4)


class Int8WeightQuantizationTests(unittest.TestCase):
    def test_replaces_supported_layers_and_reports_storage(self) -> None:
        torch.manual_seed(7)
        model = _TinyModule()
        original_embedding = model.embedding.weight.detach().clone()
        original_projection = model.projection.weight.detach().clone()

        result = quantize_module_weights_to_int8(model)

        self.assertIsInstance(model.embedding, Int8WeightEmbedding)
        self.assertIsInstance(model.projection, Int8WeightLinear)
        self.assertEqual(result.linear_layers, 1)
        self.assertEqual(result.embedding_layers, 1)
        self.assertEqual(
            result.int8_weight_bytes,
            original_embedding.numel() + original_projection.numel(),
        )
        reconstructed_embedding = (
            model.embedding.weight.float()
            * model.embedding.weight_scale.float().unsqueeze(1)
        )
        reconstructed_projection = (
            model.projection.weight.float()
            * model.projection.weight_scale.float().unsqueeze(1)
        )
        embedding_tolerance = model.embedding.weight_scale.float().max().item()
        projection_tolerance = model.projection.weight_scale.float().max().item()
        torch.testing.assert_close(
            reconstructed_embedding,
            original_embedding,
            atol=embedding_tolerance,
            rtol=0,
        )
        torch.testing.assert_close(
            reconstructed_projection,
            original_projection,
            atol=projection_tolerance,
            rtol=0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "optional ONNX dependency is not installed",
    )
    def test_exports_int8_weights_through_dequantize_linear(self) -> None:
        import onnx

        layer = Int8WeightLinear(nn.Linear(8, 4, bias=False)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "int8-linear.onnx"
            export_kwargs = {
                "opset_version": 19,
                "input_names": ["input"],
                "output_names": ["output"],
            }
            if "dynamo" in torch.onnx.export.__code__.co_varnames:
                export_kwargs["dynamo"] = False
            torch.onnx.export(
                layer,
                (torch.ones((2, 8), dtype=torch.float16),),
                str(path),
                **export_kwargs,
            )
            model = onnx.load(str(path))
            onnx.checker.check_model(model)

        self.assertIn("DequantizeLinear", {node.op_type for node in model.graph.node})
        self.assertTrue(
            any(initializer.data_type == onnx.TensorProto.INT8 for initializer in model.graph.initializer)
        )

    def test_empty_int8_shells_restore_quantized_state_dict(self) -> None:
        torch.manual_seed(11)
        source = _TinyModule()
        quantize_module_weights_to_int8(source)
        state = source.state_dict()
        with torch.device("meta"):
            restored = _TinyModule()

        result = replace_module_weights_with_int8_shells(restored)
        restored.load_state_dict(state, strict=True, assign=True)

        self.assertEqual(result.linear_layers, 1)
        self.assertEqual(result.embedding_layers, 1)
        self.assertFalse(any(value.is_meta for value in restored.state_dict().values()))
        inputs = torch.tensor([[1, 2], [3, 4]])
        torch.testing.assert_close(
            restored.projection(restored.embedding(inputs)),
            source.projection(source.embedding(inputs)),
        )


if __name__ == "__main__":
    unittest.main()
