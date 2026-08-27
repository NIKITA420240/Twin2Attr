import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from match.models.transformer.onnx_export import export_transformer_to_onnx
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
            export_transformer_to_onnx(root, precision="float32")

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

        np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(
            onnx_embeddings,
            torch_embeddings,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
