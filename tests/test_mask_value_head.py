from __future__ import annotations

import unittest

import torch

from src.model.mask_value_head import MaskValueHead, ordinal_mask_value_loss


class MaskValueHeadTest(unittest.TestCase):
    def test_logits_are_monotone_and_loss_backpropagates(self) -> None:
        head = MaskValueHead(model_dim=16, heads=4, layers=1)
        packets = torch.randn(2, 12, 16)
        question = torch.randn(2, 16)
        logits = head(packets, question, torch.tensor([0, 3, 4095]), torch.tensor([0, 0, 1]))
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertTrue(bool((logits[:, 1:] <= logits[:, :-1]).all()))
        loss = ordinal_mask_value_loss(
            logits,
            torch.tensor([0, 2, 5]),
            torch.tensor([0.5, 1.0, 0.25]),
            torch.tensor([0, 0, 1]),
        )
        loss.backward()
        self.assertGreater(loss.item(), 0)

    def test_loss_is_query_balanced_and_inverse_probability_weighted(self) -> None:
        logits = torch.zeros(3, 5, requires_grad=True)
        loss = ordinal_mask_value_loss(
            logits,
            torch.tensor([0, 5, 5]),
            torch.tensor([0.5, 0.25, 1.0]),
            torch.tensor([0, 0, 1]),
        )
        self.assertAlmostEqual(float(loss.item()), 0.69314718, places=6)


if __name__ == "__main__":
    unittest.main()
