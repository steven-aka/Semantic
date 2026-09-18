from __future__ import annotations

import math
import unittest

import torch

from src.model.partial_order_plackett_luce import partial_order_plackett_luce_nll


class PartialOrderPlackettLuceTests(unittest.TestCase):
    def test_two_packet_probability_is_exact(self) -> None:
        scores = torch.tensor([[math.log(3.0), 0.0]], requires_grad=True)
        loss = partial_order_plackett_luce_nll(scores, [[(0, 1)]])
        self.assertAlmostEqual(float(loss.detach()), -math.log(0.75), places=6)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_set_valued_extensions_are_marginalized(self) -> None:
        scores = torch.zeros((1, 3))
        # Exactly two of six permutations put packet 0 before both 1 and 2.
        loss = partial_order_plackett_luce_nll(scores, [[(0, 1), (0, 2)]])
        self.assertAlmostEqual(float(loss), -math.log(2 / 6), places=6)

    def test_aligned_order_has_lower_loss(self) -> None:
        preferences = [[(0, 1), (1, 2)]]
        aligned = partial_order_plackett_luce_nll(
            torch.tensor([[3.0, 0.0, -3.0]]), preferences
        )
        reversed_loss = partial_order_plackett_luce_nll(
            torch.tensor([[-3.0, 0.0, 3.0]]), preferences
        )
        self.assertLess(float(aligned), float(reversed_loss))

    def test_empty_preference_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no identifiable"):
            partial_order_plackett_luce_nll(torch.zeros((1, 2)), [[]])


if __name__ == "__main__":
    unittest.main()
