import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from match.model import (
    SequenceClassifierConfig,
    SequencePairDataset,
    compute_class_weights,
    compute_pr_auc,
    infer_max_length,
    train_sequence_classifier,
)


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, text, text_pair, **kwargs):
        del kwargs
        return {
            "input_ids": [
                list(range(len(left.split()) + len(right.split()) + 3))
                for left, right in zip(text, text_pair)
            ]
        }


class SequenceClassifierModelTests(unittest.TestCase):
    def test_config_rejects_invalid_compute_budget(self) -> None:
        with self.assertRaises(ValueError):
            SequenceClassifierConfig("checkpoint", hpo_trials=0)

    def test_dataset_validates_pair_and_label_counts(self) -> None:
        with self.assertRaises(ValueError):
            SequencePairDataset([("left", "right")], labels=[])

    def test_infers_quantile_length_and_rounds_to_multiple_of_eight(self) -> None:
        pairs = [
            ("one", "two"),
            ("one two three four", "five six seven eight nine"),
        ]

        max_length = infer_max_length(FakeTokenizer(), pairs)

        self.assertEqual(max_length, 16)

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
                ("black chair", "black chair"),
                ("phone", "different chair"),
                ("same chair", "black chair"),
                ("phone", "chair"),
            ]
            train_labels = [1, 0, 1, 0]
            validation_pairs = [
                ("same chair", "black chair"),
                ("phone", "different chair"),
            ]
            validation_labels = [1, 0]

            result = train_sequence_classifier(
                train_pairs,
                train_labels,
                validation_pairs,
                validation_labels,
                SequenceClassifierConfig(
                    str(checkpoint),
                    max_epochs=1,
                    hpo_trials=2,
                    seed=7,
                ),
                output_dir=output,
            )

            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "training_metadata.json").is_file())
            self.assertTrue(np.isfinite(result.validation_pr_auc))


if __name__ == "__main__":
    unittest.main()
