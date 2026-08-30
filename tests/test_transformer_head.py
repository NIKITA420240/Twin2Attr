import unittest
import tempfile
from pathlib import Path

import torch
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    XLMRobertaConfig,
)

from match.models.transformer.head import (
    GatedResidualFusionHead,
    GatedResidualFusionSequenceClassifier,
    GatedResidualFusionSequenceClassifierConfig,
    HybridSequenceClassifier,
    HybridSequenceClassifierConfig,
    PoolingHeadConfig,
    TransformerPoolingHead,
)
from match.models.transformer.model import model_factory
from match.models.transformer.predictor import load_trained_classifier


class TransformerPoolingHeadTests(unittest.TestCase):
    def test_gated_residual_model_survives_save_and_load(self) -> None:
        backbone = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            num_labels=1,
        )
        model = GatedResidualFusionSequenceClassifier(
            GatedResidualFusionSequenceClassifierConfig(
                backbone_config=backbone.to_dict(),
                head_config=PoolingHeadConfig(
                    poolings=("mean", "attention"),
                    typed_hidden_dims=(4,),
                    dropout=0.0,
                ).to_dict(),
                typed_feature_count=6,
                num_labels=2,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = GatedResidualFusionSequenceClassifier.from_pretrained(
                directory
            )

        self.assertEqual(restored.config.typed_feature_count, 6)
        self.assertEqual(restored.config.match_head_type, "gated_residual_fusion")

    def test_gated_residual_head_is_exact_noop_after_reset(self) -> None:
        head = GatedResidualFusionHead(
            hidden_size=4,
            typed_feature_count=6,
            config=PoolingHeadConfig(
                poolings=("mean", "attention"),
                mlp_hidden_dims=(5,),
                typed_hidden_dims=(3,),
                dropout=0.0,
            ),
        ).eval()
        head.reset_residual_outputs()

        correction = head(
            torch.randn(2, 4, 4),
            torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            torch.randn(2, 6),
        )

        torch.testing.assert_close(correction, torch.zeros(2, 1))

    def test_gated_residual_sequence_classifier_starts_as_native(self) -> None:
        backbone = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            num_labels=1,
        )
        model = GatedResidualFusionSequenceClassifier(
            GatedResidualFusionSequenceClassifierConfig(
                backbone_config=backbone.to_dict(),
                head_config=PoolingHeadConfig(
                    poolings=("mean", "attention"),
                    typed_hidden_dims=(4,),
                    dropout=0.0,
                ).to_dict(),
                typed_feature_count=6,
                num_labels=2,
            )
        ).eval()
        inputs = {
            "input_ids": torch.tensor([[0, 5, 6, 2], [0, 7, 2, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        }

        with torch.inference_mode():
            hidden_states = model.backbone(**inputs).last_hidden_state
            native_score = model.native_head(hidden_states)
            logits = model(**inputs, typed_features=torch.randn(2, 6)).logits

        torch.testing.assert_close(logits[:, :1], torch.zeros_like(native_score))
        torch.testing.assert_close(logits[:, 1:], native_score)

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

        weights = head.attention_weights(
            hidden_states,
            torch.tensor([[1, 1, 0]]),
        )
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 1))
        torch.testing.assert_close(weights[:, 2], torch.zeros(1, 1))

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

    def test_hybrid_rejects_multi_logit_native_head(self) -> None:
        backbone = BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            num_labels=2,
        )
        model = HybridSequenceClassifier(
            HybridSequenceClassifierConfig(
                backbone_config=backbone.to_dict(),
                head_config=PoolingHeadConfig(poolings=("attention",)).to_dict(),
                num_labels=2,
            )
        )

        with self.assertRaisesRegex(ValueError, "scalar reranker head"):
            model(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )

    def test_fixed_native_only_hybrid_disables_attention_branch(self) -> None:
        backbone = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            num_labels=1,
        )
        model = HybridSequenceClassifier(
            HybridSequenceClassifierConfig(
                backbone_config=backbone.to_dict(),
                head_config=PoolingHeadConfig(
                    poolings=("attention",),
                    dropout=0.0,
                    attention_hidden_dim=4,
                    native_logit_weight=1.0,
                    attention_logit_weight=0.0,
                    train_logit_weights=False,
                ).to_dict(),
                num_labels=2,
            )
        ).eval()
        inputs = {
            "input_ids": torch.tensor([[0, 5, 6, 2]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

        with torch.inference_mode():
            hidden_states = model.backbone(
                **inputs,
                return_dict=True,
            ).last_hidden_state
            expected_native = model.native_head(hidden_states)
            logits = model(**inputs).logits

        self.assertNotIn("logit_weights", dict(model.named_parameters()))
        self.assertIn("logit_weights", dict(model.named_buffers()))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.attention_head.parameters())
        )
        torch.testing.assert_close(logits[:, 1:], expected_native)
        torch.testing.assert_close(logits[:, :1], torch.zeros_like(expected_native))

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
                    output_hidden_states=True,
                    output_attentions=True,
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
            outputs = model(**inputs)
            self.assertIsNone(outputs.hidden_states)
            self.assertIsNone(outputs.attentions)
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
