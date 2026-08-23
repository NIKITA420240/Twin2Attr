import unittest
import tempfile
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from match.models.transformer.head import PoolingHeadConfig, TransformerPoolingHead
from match.models.transformer.model import model_factory
from match.models.transformer.predictor import load_trained_classifier


class TransformerPoolingHeadTests(unittest.TestCase):
    def test_combines_selected_poolings_and_ignores_padding(self) -> None:
        head = TransformerPoolingHead(
            hidden_size=2,
            num_labels=2,
            config=PoolingHeadConfig(
                poolings=("mean", "max"),
                mlp_hidden_dims=(),
                dropout=0.0,
            ),
        )
        hidden_states = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]]
        )
        attention_mask = torch.tensor([[1, 1, 0]])

        pooled = head.pool(hidden_states, attention_mask)

        torch.testing.assert_close(
            pooled,
            torch.tensor([[2.0, 3.0, 3.0, 4.0]]),
        )

    def test_attention_pooling_ignores_padding(self) -> None:
        head = TransformerPoolingHead(
            hidden_size=2,
            num_labels=2,
            config=PoolingHeadConfig(
                poolings=("attention",),
                dropout=0.0,
                attention_hidden_dim=1,
            ),
        )
        hidden_states = torch.tensor(
            [[[2.0, 3.0], [2.0, 3.0], [1000.0, 1000.0]]]
        )

        pooled = head.pool(hidden_states, torch.tensor([[1, 1, 0]]))

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))

    def test_multi_head_attention_pooling_concatenates_head_outputs(self) -> None:
        head = TransformerPoolingHead(
            hidden_size=2,
            num_labels=2,
            config=PoolingHeadConfig(
                poolings=("attention",),
                dropout=0.0,
                attention_hidden_dim=1,
                attention_num_heads=3,
            ),
        )
        hidden_states = torch.tensor(
            [[[2.0, 3.0], [2.0, 3.0], [1000.0, 1000.0]]]
        )

        pooled = head.pool(hidden_states, torch.tensor([[1, 1, 0]]))

        torch.testing.assert_close(
            pooled,
            torch.tensor([[2.0, 3.0, 2.0, 3.0, 2.0, 3.0]]),
        )

    def test_rejects_non_positive_attention_head_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention_num_heads"):
            PoolingHeadConfig(
                poolings=("attention",),
                attention_num_heads=0,
            )

    def test_fully_masked_input_produces_finite_zero_features(self) -> None:
        head = TransformerPoolingHead(
            hidden_size=2,
            num_labels=2,
            config=PoolingHeadConfig(poolings=("cls", "mean", "max", "attention")),
        )

        pooled = head.pool(
            torch.ones(1, 2, 2),
            torch.zeros(1, 2, dtype=torch.long),
        )

        torch.testing.assert_close(pooled, torch.zeros(1, 8))

    def test_factory_model_survives_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint"
            trained = root / "trained"
            checkpoint.mkdir()
            (checkpoint / "vocab.txt").write_text(
                "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "item"]),
                encoding="utf-8",
            )
            tokenizer = BertTokenizerFast(
                vocab_file=str(checkpoint / "vocab.txt"),
            )
            tokenizer.save_pretrained(checkpoint)
            BertModel(
                BertConfig(
                    vocab_size=tokenizer.vocab_size,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                    max_position_embeddings=32,
                )
            ).save_pretrained(checkpoint)
            default_model = model_factory(
                str(checkpoint),
                tokenizer,
                use_field_tokens=False,
            )()
            self.assertFalse(hasattr(default_model.config, "match_head_type"))
            initialize = model_factory(
                str(checkpoint),
                tokenizer,
                use_field_tokens=True,
                head_type="pooling",
                head_config=PoolingHeadConfig(
                    poolings=("cls", "mean", "attention"),
                    mlp_hidden_dims=(8,),
                    dropout=0.0,
                    attention_hidden_dim=4,
                    attention_num_heads=2,
                ),
            )
            model = initialize().eval()
            inputs = {
                "input_ids": torch.tensor([[2, 5, 3, 5, 3]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
                "token_type_ids": torch.tensor([[0, 0, 0, 1, 1]]),
            }
            expected = model(**inputs).logits
            self.assertTrue(
                torch.isfinite(model(**inputs, labels=torch.tensor([1])).loss)
            )
            model.save_pretrained(trained)
            tokenizer.save_pretrained(trained)

            _, restored = load_trained_classifier(trained, device="cpu")

            self.assertEqual(restored.config.match_head_type, "pooling")
            self.assertEqual(restored.config.head_config["attention_num_heads"], 2)
            torch.testing.assert_close(restored(**inputs).logits, expected)
            self.assertIsInstance(restored(**inputs, return_dict=False), tuple)


if __name__ == "__main__":
    unittest.main()
