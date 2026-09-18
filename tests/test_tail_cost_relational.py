import math
import unittest

import torch
from torch.nn import functional as F

from src.data.tail_cost_boundary_dataset import build_tail_cost_row
from src.model.tail_cost_relational_loss import tail_cost_relational_loss


class TailCostDatasetTest(unittest.TestCase):
    def test_separates_safety_and_rate_edges_and_normalizes_each_pool(self) -> None:
        row = {
            "example_id": "x",
            "pairwise_preferences": [[0, 2]],
            "all_identifiable_preferences": [[0, 2], [1, 2]],
            "packet_tokens": [10, 20, 100],
            "packet_inclusion_fraction_by_level": [
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ],
        }
        result = build_tail_cost_row(row)
        self.assertEqual([(edge["winner"], edge["loser"]) for edge in result["safety_preferences"]], [(0, 2)])
        self.assertEqual([(edge["winner"], edge["loser"]) for edge in result["rate_preferences"]], [(1, 2)])
        self.assertAlmostEqual(result["safety_preferences"][0]["weight"], 1.0)
        self.assertAlmostEqual(result["rate_preferences"][0]["weight"], 1.0)
        self.assertEqual(result["safety_preferences"][0]["anchor_coverage"], 2)


class TailCostLossTest(unittest.TestCase):
    def test_top_quartile_uses_multiple_edges_and_adds_rate_term(self) -> None:
        logits = torch.zeros(1, 5, 5)
        logits[0, 0, 1] = -3.0
        logits[0, 0, 2] = -2.0
        logits[0, 0, 3] = -1.0
        logits[0, 0, 4] = 1.0
        logits[0, 1, 4] = -0.5
        result = tail_cost_relational_loss(
            logits,
            [[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)]],
            [[(1, 4, 1.0)]],
            tail_fraction=0.5,
            rate_lambda=0.25,
        )
        expected_safety = (F.softplus(torch.tensor(3.0)) + F.softplus(torch.tensor(2.0))) / 2
        expected_rate = F.softplus(torch.tensor(0.5))
        self.assertTrue(torch.allclose(result.safety_tail, expected_safety))
        self.assertTrue(torch.allclose(result.rate, expected_rate))
        self.assertTrue(torch.allclose(result.total, expected_safety + 0.25 * expected_rate))

    def test_zero_safety_example_can_train_on_rate(self) -> None:
        logits = torch.zeros(1, 2, 2)
        result = tail_cost_relational_loss(
            logits, [[]], [[(0, 1, 1.0)]], tail_fraction=0.25, rate_lambda=0.25
        )
        self.assertIsNone(result.safety_tail)
        self.assertAlmostEqual(float(result.total), 0.25 * math.log(2), places=6)


if __name__ == "__main__":
    unittest.main()
