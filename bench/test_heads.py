from __future__ import annotations

import unittest

import torch

from frozen_features import pair_segment_masks
from heads import BidirectionalCategoryConditionedHead, head_loss


class FinalHeadTests(unittest.TestCase):
    def test_end_to_end_order_symmetry(self) -> None:
        torch.manual_seed(7)
        head = BidirectionalCategoryConditionedHead(16, 8, 12, 0.0, 3, 4).eval()
        forward = torch.randn(11, 5, 16)
        reverse = torch.randn(11, 5, 16)
        categories = torch.randint(0, 3, (11,))
        first = head(
            forward,
            reverse_features=reverse,
            category_ids=categories,
        ).logit
        second = head(
            reverse,
            reverse_features=forward,
            category_ids=categories,
        ).logit
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_loss_is_finite(self) -> None:
        head = BidirectionalCategoryConditionedHead(16, 8, 12, 0.0, 3, 4)
        output = head(
            torch.randn(5, 5, 16),
            reverse_features=torch.randn(5, 5, 16),
            category_ids=torch.tensor([0, 1, 2, 1, 0]),
        )
        loss = head_loss(
            output,
            torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            positive_weight=torch.tensor(1.0),
            consistency_weight=0.05,
        )
        self.assertTrue(bool(torch.isfinite(loss)))


class SegmentMaskTests(unittest.TestCase):
    def test_bert_pair_layout(self) -> None:
        ids = torch.tensor([[101, 10, 11, 102, 20, 21, 102, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
        types = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]])
        left, right = pair_segment_masks(
            ids,
            mask,
            sep_token_id=102,
            cls_token_id=101,
            pad_token_id=0,
            token_type_ids=types,
        )
        self.assertEqual(left.sum().item(), 2)
        self.assertEqual(right.sum().item(), 2)

    def test_xlm_roberta_pair_layout(self) -> None:
        ids = torch.tensor([[0, 10, 11, 2, 2, 20, 21, 2, 1]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
        left, right = pair_segment_masks(
            ids,
            mask,
            sep_token_id=2,
            cls_token_id=0,
            pad_token_id=1,
        )
        self.assertEqual(left.sum().item(), 2)
        self.assertEqual(right.sum().item(), 2)


if __name__ == "__main__":
    unittest.main()
