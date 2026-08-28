import gc
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from match.models.transformer.onnx_export import (
    ONNX_EXTERNAL_DATA_SUFFIX,
    _convert_to_float16,
    export_transformer_to_onnx,
)
from match.models.transformer.onnx_runtime import (
    OnnxRuntimeTransformerExecutor,
)
from match.models.transformer.predictor import (
    TransformerPredictor,
    encode_pair_cls,
    load_trained_classifier,
    predict_pair_logits,
)
from match.prepare_data import PreparedCard, PreparedPair


_ONNX_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime")
)


@unittest.skipUnless(_ONNX_AVAILABLE, "optional ONNX dependencies are not installed")
class OnnxExportIntegrationTests(unittest.TestCase):
    def test_float16_conversion_uses_path_shape_inference(self) -> None:
        import onnx
        from onnx import TensorProto, external_data_helper, helper, numpy_helper
        from onnxconverter_common import float16

        shape = [1, 1024]
        source_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Add", ["input", "weights"], ["output"])],
                "external-data",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
                [
                    numpy_helper.from_array(
                        np.ones(shape, dtype=np.float32),
                        name="weights",
                    )
                ],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            destination = root / "destination.onnx"
            external_data = destination.with_name(
                f"{destination.name}{ONNX_EXTERNAL_DATA_SUFFIX}"
            )
            onnx.save(source_model, str(source))
            with (
                patch.object(
                    float16,
                    "convert_float_to_float16_model_path",
                    side_effect=AssertionError("Windows-unsafe path helper used"),
                ),
                patch.object(
                    onnx.shape_inference,
                    "infer_shapes_path",
                    wraps=onnx.shape_inference.infer_shapes_path,
                ) as infer_path,
                patch.object(
                    float16,
                    "convert_float_to_float16",
                    wraps=float16.convert_float_to_float16,
                ) as convert,
            ):
                _convert_to_float16(source, destination)

            onnx.checker.check_model(onnx.load(str(destination)))
            unloaded = onnx.load(str(destination), load_external_data=False)
            self.assertTrue(external_data.is_file())
            self.assertTrue(
                any(
                    external_data_helper.uses_external_data(initializer)
                    for initializer in unloaded.graph.initializer
                )
            )
            original_size = external_data.stat().st_size
            with external_data.open("ab") as file:
                file.write(b"stale")
            _convert_to_float16(source, destination)
            self.assertEqual(external_data.stat().st_size, original_size)

        infer_path.assert_called_once()
        convert.assert_called_once_with(
            ANY,
            keep_io_types=True,
            disable_shape_infer=True,
        )

    def test_exported_graphs_match_pytorch_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "left",
                "right",
                "product",
                "category",
            ]
            (root / "vocab.txt").write_text(
                "\n".join(vocabulary),
                encoding="utf-8",
            )
            tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
            tokenizer.save_pretrained(root)
            config = BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=2,
            )
            config.match_max_length = 24
            config.match_use_field_tokens = False
            config.match_max_attribute_value_chars = 256
            config.match_max_attribute_value_tokens = 16
            BertForSequenceClassification(config).save_pretrained(root)
            export_transformer_to_onnx(root, precision="float16")

            tokenizer, model = load_trained_classifier(root, device="cpu")
            pair = PreparedPair(
                PreparedCard(1, "left product", "category", ()),
                PreparedCard(2, "right product", "category", ()),
                None,
                "category",
            )
            pairs = [pair, pair, pair]
            executor = OnnxRuntimeTransformerExecutor(
                model_directory=root,
                model_config=config.to_dict(),
                provider="cpu",
                io_binding=False,
            )
            runtime = TransformerPredictor(tokenizer, executor, batch_size=2)

            onnx_logits = runtime.predict_pair_logits(pairs)
            onnx_embeddings = runtime.encode_pairs(pairs)
            torch_logits = predict_pair_logits(model, tokenizer, pairs, batch_size=2)
            torch_embeddings = encode_pair_cls(model, tokenizer, pairs, batch_size=2)
            del runtime, executor, model, tokenizer
            gc.collect()

        np.testing.assert_allclose(onnx_logits, torch_logits, atol=5e-3, rtol=5e-3)
        np.testing.assert_allclose(
            onnx_embeddings,
            torch_embeddings,
            atol=5e-3,
            rtol=5e-3,
        )

    def test_one_logit_prompted_graph_matches_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocabulary = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "question",
                "passage",
                "name",
                "left",
                "right",
                "category",
            ]
            (root / "vocab.txt").write_text(
                "\n".join(vocabulary), encoding="utf-8"
            )
            tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
            tokenizer.save_pretrained(root)
            config = BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=1,
            )
            config.match_profile = "prompted_binary_reranker"
            config.match_head_type = "native"
            config.match_num_logits = 1
            config.match_probability_transform = "sigmoid"
            config.match_max_length = 24
            config.match_use_field_tokens = False
            config.match_max_attribute_value_chars = 256
            config.match_max_attribute_value_tokens = 16
            BertForSequenceClassification(config).save_pretrained(root)
            export_transformer_to_onnx(root, precision="float32")

            tokenizer, model = load_trained_classifier(root, device="cpu")
            pair = PreparedPair(
                PreparedCard(1, "left", "category", ()),
                PreparedCard(2, "right", "category", ()),
                None,
                "category",
            )
            pairs = [pair, pair, pair]
            executor = OnnxRuntimeTransformerExecutor(
                model_directory=root,
                model_config=config.to_dict(),
                provider="cpu",
                io_binding=False,
            )
            runtime = TransformerPredictor(tokenizer, executor, batch_size=2)

            onnx_logits = runtime.predict_pair_logits(pairs)
            onnx_probabilities = runtime.predict_proba(
                SimpleNamespace(prepared_pairs=lambda: pairs)
            )
            torch_logits = predict_pair_logits(model, tokenizer, pairs, batch_size=2)
            del runtime, executor, model, tokenizer
            gc.collect()

        self.assertEqual(onnx_logits.shape, (3, 1))
        np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            onnx_probabilities,
            1.0 / (1.0 + np.exp(-torch_logits[:, 0])),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
