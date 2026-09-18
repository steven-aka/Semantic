import unittest

import torch

from src.model.critical_boundary_loss import worst_boundary_rank_loss


class CriticalBoundaryLossTest(unittest.TestCase):
    def test_only_worst_pair_controls_each_example(self) -> None:
        scores = torch.tensor([[2.0, 1.0, 3.0]], requires_grad=True)
        loss = worst_boundary_rank_loss(scores, [[(0, 1), (0, 2)]])
        expected = torch.nn.functional.softplus(torch.tensor(1.0))
        self.assertTrue(torch.allclose(loss.detach(), expected))
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_examples_are_balanced(self) -> None:
        scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = worst_boundary_rank_loss(scores, [[(0, 1)], [(0, 1)]])
        expected = (
            torch.nn.functional.softplus(torch.tensor(-1.0))
            + torch.nn.functional.softplus(torch.tensor(1.0))
        ) / 2
        self.assertTrue(torch.allclose(loss, expected))

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no critical-boundary"):
            worst_boundary_rank_loss(torch.zeros(1, 2), [[]])


if __name__ == "__main__":
    unittest.main()
