import unittest

import torch

from match.models.transformer.objective import weighted_classification_loss


class TransformerObjectiveTests(unittest.TestCase):
    def test_one_logit_uses_weighted_binary_cross_entropy(self) -> None:
        logits = torch.tensor([[-2.0], [2.0]])
        labels = torch.tensor([0, 1])
        sample_weights = torch.tensor([1.0, 3.0])
        class_weights = torch.tensor([2.0, 0.5])

        actual = weighted_classification_loss(
            logits, labels, sample_weights, class_weights
        )
        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], labels.float(), reduction="none"
        ) * class_weights[labels]
        expected = torch.sum(per_row * sample_weights) / sample_weights.sum()

        torch.testing.assert_close(actual, expected)

    def test_two_logits_uses_weighted_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        labels = torch.tensor([0, 1])
        sample_weights = torch.tensor([2.0, 1.0])
        class_weights = torch.tensor([0.5, 2.0])

        actual = weighted_classification_loss(
            logits, labels, sample_weights, class_weights
        )
        per_row = torch.nn.functional.cross_entropy(
            logits, labels, weight=class_weights, reduction="none"
        )
        expected = torch.sum(per_row * sample_weights) / sample_weights.sum()

        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_output_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch, 1 or 2"):
            weighted_classification_loss(
                torch.zeros(2, 3),
                torch.tensor([0, 1]),
                torch.ones(2),
                torch.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
