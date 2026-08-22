import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from match.pair_encoding import PreparedPairDataset, infer_pair_max_length
from match.prepare_data import PreparedCard, PreparedPair
from match.transformer import (
    SequenceClassifierConfig,
    compute_class_weights,
    compute_pr_auc,
    train_sequence_classifier,
)


class FakeTokenizer:
    model_max_length = 512

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, *, pair=False):
        return 3 if pair else 2


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
    def test_config_rejects_invalid_compute_budget(self) -> None:
        with self.assertRaises(ValueError):
            SequenceClassifierConfig("checkpoint", hpo_trials=0)

    def test_dataset_preserves_prepared_pairs(self) -> None:
        pair = _pair(1, "left", 2, "right", 1)
        dataset = PreparedPairDataset([pair])

        self.assertEqual(len(dataset), 1)
        self.assertIs(dataset[0], pair)

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

    def test_balanced_class_weights_give_rare_class_more_weight(self) -> None:
        weights = compute_class_weights([0, 0, 0, 1])

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
                ),
                output_dir=output,
            )

            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "training_metadata.json").is_file())
            self.assertTrue(np.isfinite(result.validation_macro_pr_auc))


if __name__ == "__main__":
    unittest.main()
