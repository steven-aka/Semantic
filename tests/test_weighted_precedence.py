import itertools
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.model.relational_boundary_loss import worst_relational_boundary_loss
from src.model.relational_packet_ranker import RelationalPacketRanker
from src.search.weighted_precedence import exact_weighted_precedence_order


def score(order, logits):
    return sum(logits[first][second] for i, first in enumerate(order) for second in order[i + 1 :])


class WeightedPrecedenceTest(unittest.TestCase):
    def test_relational_head_is_antisymmetric(self) -> None:
        class FakeLM(nn.Module):
            config = SimpleNamespace(hidden_size=4)

            def forward(self, input_ids, **kwargs):
                hidden = torch.nn.functional.one_hot(input_ids % 4, 4).float()
                return SimpleNamespace(hidden_states=[hidden])

        model = RelationalPacketRanker(FakeLM(), head_hidden_size=3)
        logits = model(
            torch.tensor([[0, 1, 2, 3]]),
            torch.ones(1, 4, dtype=torch.long),
            [[(0, 1), (1, 2), (2, 3)]],
        )["precedence_logits"]
        self.assertTrue(torch.allclose(logits, -logits.transpose(1, 2)))
        self.assertTrue(torch.allclose(torch.diagonal(logits, dim1=1, dim2=2), torch.zeros(1, 3)))

    def test_exact_decoder_matches_brute_force(self) -> None:
        logits = [
            [0.0, 2.0, -1.0, 0.5],
            [-2.0, 0.0, 3.0, -0.2],
            [1.0, -3.0, 0.0, 2.0],
            [-0.5, 0.2, -2.0, 0.0],
        ]
        actual = exact_weighted_precedence_order(logits)
        brute = max(itertools.permutations(range(4)), key=lambda order: (score(order, logits), tuple(-x for x in order)))
        self.assertEqual(score(actual, logits), score(brute, logits))

    def test_zero_tie_is_deterministic(self) -> None:
        self.assertEqual(exact_weighted_precedence_order([[0.0] * 3 for _ in range(3)]), (0, 1, 2))

    def test_relational_loss_targets_worst_edge(self) -> None:
        logits = torch.tensor([[[0.0, 2.0, -1.0], [-2.0, 0.0, 0.5], [1.0, -0.5, 0.0]]])
        loss = worst_relational_boundary_loss(logits, [[(0, 1), (1, 2)]])
        self.assertTrue(torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(-0.5))))

    def test_relational_loss_rejects_empty_supervision(self) -> None:
        with self.assertRaises(ValueError):
            worst_relational_boundary_loss(torch.zeros(1, 2, 2), [[]])


if __name__ == "__main__":
    unittest.main()
