import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from match.models.transformer import (
    SequenceClassifierConfig,
    compute_class_weights,
    compute_pr_auc,
    train_sequence_classifier,
)
from match.models.transformer.onnx_export import _ClassifierGraph, _EncoderGraph
from match.models.transformer.predictor import (
    _length_bucket_order,
    _restore_original_order,
    predict_logit_margins,
)
from match.pair_encoding import (
    PairEncodingCollator,
    PreparedPairDataset,
    encode_prepared_pair,
    infer_pair_max_length,
)
from match.prepare_data import PreparedCard, PreparedPair


class FakeTokenizer:
    model_max_length = 512

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, *, pair=False):
        return 3 if pair else 2


class TransformersV5BertTokenizer(FakeTokenizer):
    """Minimal v5-style tokenizer without legacy pair builder methods."""

    cls_token_id = 101
    sep_token_id = 102
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
        return_attention_mask,
        return_token_type_ids,
    ):
        del padding, truncation, return_attention_mask, return_token_type_ids
        return {
            "input_ids": [
                self.encode(text, add_special_tokens=add_special_tokens)
                for text in texts
            ]
        }

    def pad(self, rows, *, padding, max_length=None, return_tensors):
        self.last_padding = padding
        self.last_max_length = max_length
        target = max_length or max(len(row["input_ids"]) for row in rows)
        result = {}
        for key in rows[0]:
            result[key] = torch.tensor(
                [row[key] + [0] * (target - len(row[key])) for row in rows]
            )
        return result


class RecordingBertTokenizer(TransformersV5BertTokenizer):
    def __init__(self):
        self.encoded_texts = []

    def encode(self, text, *, add_special_tokens=False):
        self.encoded_texts.append(text)
        return super().encode(text, add_special_tokens=add_special_tokens)


class TransformersV5BertTokenizerWithMissingPairMetadata(
    TransformersV5BertTokenizer
):
    """V5 backend that does not report the BERT pair special-token layout."""

    def num_special_tokens_to_add(self, *, pair=False):
        return 2


class TransformersV5XLMRobertaTokenizer(FakeTokenizer):
    """Minimal v5-style XLM-R tokenizer without legacy pair builders."""

    cls_token = "<s>"
    sep_token = "</s>"
    cls_token_id = 0
    sep_token_id = 2
    model_input_names = ["input_ids", "attention_mask"]

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(10, 10 + len(text.split())))

    def num_special_tokens_to_add(self, *, pair=False):
        return 4 if pair else 2


def _card(item_id: int, name: str, category: str = "category") -> PreparedCard:
    return PreparedCard(item_id, name, category, ())


def _pair(
    left_id: int,
    left_name: str,
    right_id: int,
    right_name: str,
    label: int,
    category: str = "category",
) -> PreparedPair:
    return PreparedPair(
        _card(left_id, left_name, category),
        _card(right_id, right_name, category),
        label,
        category,
    )


class SequenceClassifierModelTests(unittest.TestCase):
    def test_onnx_wrappers_match_pytorch_outputs(self) -> None:
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=2,
            )
        ).eval()
        input_ids = torch.randint(0, 32, (3, 8))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        with torch.inference_mode():
            expected_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
            expected_cls = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state[:, 0, :]
            logits = _ClassifierGraph(model, token_type_ids=True)(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            cls_embedding = _EncoderGraph(model, token_type_ids=True)(
                input_ids,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(cls_embedding, expected_cls)

    def test_config_rejects_invalid_compute_budget(self) -> None:
        with self.assertRaises(ValueError):
            SequenceClassifierConfig("checkpoint", hpo_trials=0)

    def test_dataset_preserves_prepared_pairs(self) -> None:
        pair = _pair(1, "left", 2, "right", 1)
        dataset = PreparedPairDataset([pair])

        self.assertEqual(len(dataset), 1)
        self.assertIs(dataset[0], pair)

    def test_length_bucketing_restores_original_pair_order(self) -> None:
        pairs = [
            _pair(1, "a much longer product name", 2, "another long name", 0),
            _pair(3, "short", 4, "tiny", 1),
            _pair(5, "medium product", 6, "medium", 0),
        ]
        original = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)

        order = _length_bucket_order(pairs, max_attribute_value_tokens=16)
        restored = _restore_original_order(original[order], order)

        self.assertEqual(order.tolist(), [1, 2, 0])
        np.testing.assert_array_equal(restored, original)

    def test_collator_pads_to_the_next_configured_length_bucket(self) -> None:
        tokenizer = TransformersV5BertTokenizer()
        pair = _pair(1, "left name", 2, "right name", 1)
        encoded = encode_prepared_pair(
            tokenizer,
            pair,
            max_length=32,
            use_field_tokens=False,
        )
        expected_length = next(
            bucket for bucket in (8, 16, 32)
            if bucket >= len(encoded["input_ids"])
        )
        collator = PairEncodingCollator(
            tokenizer,
            32,
            use_field_tokens=False,
            include_labels=False,
            padding_length_buckets=(8, 16, 32),
        )

        batch = collator([pair])

        self.assertEqual(batch["input_ids"].shape[1], expected_length)
        self.assertEqual(tokenizer.last_padding, "max_length")
        self.assertEqual(tokenizer.last_max_length, expected_length)

    def test_batched_field_tokenization_matches_scalar_encoding(self) -> None:
        tokenizer = TransformersV5BertTokenizer()
        pairs = [
            PreparedPair(
                PreparedCard(
                    1,
                    "left product",
                    "category",
                    (("color", "deep black"), ("memory", "256 gb")),
                ),
                PreparedCard(
                    2,
                    "right product",
                    "category",
                    (("color", "black"),),
                ),
                None,
                "category",
            ),
            _pair(3, "short", 4, "another product", 1),
        ]
        scalar = PairEncodingCollator(
            tokenizer,
            32,
            use_field_tokens=False,
            include_labels=False,
        )(pairs)
        batched = PairEncodingCollator(
            tokenizer,
            32,
            use_field_tokens=False,
            include_labels=False,
            batch_fields=True,
            field_chunk_size=3,
        )(pairs)

        self.assertEqual(set(scalar), set(batched))
        for name in scalar:
            torch.testing.assert_close(scalar[name], batched[name])

    def test_collator_requires_final_bucket_to_match_max_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal max_length"):
            PairEncodingCollator(
                TransformersV5BertTokenizer(),
                32,
                use_field_tokens=False,
                padding_length_buckets=(8, 16),
            )

    def test_attribute_value_character_limit_is_applied_before_tokenization(self) -> None:
        tokenizer = RecordingBertTokenizer()
        pair = PreparedPair(
            PreparedCard(1, "left", "category", (("description", "x" * 500),)),
            PreparedCard(2, "right", "category", ()),
            1,
            "category",
        )

        encode_prepared_pair(
            tokenizer,
            pair,
            max_length=32,
            use_field_tokens=False,
            max_attribute_value_chars=256,
            max_attribute_value_tokens=16,
        )

        self.assertIn(" " + "x" * 256, tokenizer.encoded_texts)
        self.assertNotIn(" " + "x" * 500, tokenizer.encoded_texts)

    def test_infers_quantile_length_and_rounds_to_multiple_of_eight(self) -> None:
        pairs = [
            _pair(1, "one", 2, "two", 1),
            _pair(
                3,
                "one two three four",
                4,
                "five six seven eight nine",
                0,
            ),
        ]

        max_length = infer_pair_max_length(
            FakeTokenizer(),
            pairs,
            use_field_tokens=False,
        )

        self.assertEqual(max_length, 24)

    def test_encodes_bert_pair_without_removed_transformers_v5_methods(self) -> None:
        encoded = encode_prepared_pair(
            TransformersV5BertTokenizer(),
            _pair(1, "left name", 2, "right name", 1),
            max_length=32,
            use_field_tokens=False,
        )

        self.assertEqual(encoded["input_ids"][0], 101)
        self.assertEqual(encoded["input_ids"].count(102), 2)
        self.assertEqual(
            len(encoded["token_type_ids"]),
            len(encoded["input_ids"]),
        )
        self.assertIn(1, encoded["token_type_ids"])

    def test_encodes_bert_pair_when_v5_backend_misses_pair_metadata(self) -> None:
        encoded = encode_prepared_pair(
            TransformersV5BertTokenizerWithMissingPairMetadata(),
            _pair(1, "left name", 2, "right name", 1),
            max_length=32,
            use_field_tokens=False,
        )

        self.assertEqual(encoded["input_ids"][0], 101)
        self.assertEqual(encoded["input_ids"].count(102), 2)
        self.assertLessEqual(len(encoded["input_ids"]), 32)

    def test_encodes_xlm_roberta_pair_without_v5_pair_builder(self) -> None:
        encoded = encode_prepared_pair(
            TransformersV5XLMRobertaTokenizer(),
            _pair(1, "left name", 2, "right name", 1),
            max_length=32,
            use_field_tokens=False,
        )

        self.assertEqual(encoded["input_ids"][0], 0)
        self.assertEqual(encoded["input_ids"].count(2), 3)
        self.assertLessEqual(len(encoded["input_ids"]), 32)

    def test_balanced_class_weights_give_rare_class_more_weight(self) -> None:
        weights = compute_class_weights([0, 0, 0, 1])

        torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.0]))

    def test_class_weights_include_dataset_sample_weights(self) -> None:
        weights = compute_class_weights([0, 1], [3.0, 1.0])

        torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.0]))

    def test_pr_auc_uses_continuous_positive_class_scores(self) -> None:
        logits = np.array(
            [
                [3.0, 0.0],
                [0.0, 3.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        )
        labels = np.array([0, 1, 0, 1])

        metrics = compute_pr_auc((logits, labels))

        self.assertEqual(metrics["pr_auc"], 1.0)

    def test_stacking_logit_margin_is_positive_minus_negative_logit(self) -> None:
        logits = np.array([[3.0, 1.0], [-1.0, 2.5]], dtype=np.float32)
        with patch(
            "match.models.transformer.predictor.predict_pair_logits",
            return_value=logits,
        ):
            margins = predict_logit_margins(object(), object(), [object(), object()])

        np.testing.assert_array_equal(margins, np.array([-2.0, 3.5]))

    def test_trains_and_saves_tiny_local_sequence_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            checkpoint = directory / "checkpoint"
            output = directory / "trained"
            checkpoint.mkdir()
            vocabulary = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "black",
                "chair",
                "phone",
                "same",
                "different",
                "category",
            ]
            (checkpoint / "vocab.txt").write_text(
                "\n".join(vocabulary),
                encoding="utf-8",
            )
            tokenizer = BertTokenizerFast(
                vocab_file=str(checkpoint / "vocab.txt"),
                do_lower_case=True,
            )
            tokenizer.save_pretrained(checkpoint)
            model = BertForSequenceClassification(
                BertConfig(
                    vocab_size=tokenizer.vocab_size,
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                    max_position_embeddings=64,
                    num_labels=2,
                )
            )
            model.save_pretrained(checkpoint)

            train_pairs = [
                _pair(1, "black chair", 2, "black chair", 1),
                _pair(3, "phone", 4, "different chair", 0),
                _pair(5, "same chair", 6, "black chair", 1),
                _pair(7, "phone", 8, "chair", 0),
            ]
            validation_pairs = [
                _pair(9, "same chair", 10, "black chair", 1),
                _pair(11, "phone", 12, "different chair", 0),
            ]

            result = train_sequence_classifier(
                train_pairs,
                validation_pairs,
                SequenceClassifierConfig(
                    str(checkpoint),
                    max_epochs=1,
                    hpo_trials=1,
                    seed=7,
                    max_attribute_value_chars=256,
                ),
                output_dir=output,
            )

            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "training_metadata.json").is_file())
            saved_config = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config["match_max_attribute_value_chars"], 256)
            self.assertTrue(np.isfinite(result.validation_macro_pr_auc))


if __name__ == "__main__":
    unittest.main()
