import math
import unittest

import torch

from src.training.train_v17cut_b0_structured_fixed_v8 import (
    best_stop_vector, log_partition, oracle_optimal_paths,
)


class StructuredStopTests(unittest.TestCase):
    def test_oracle_handles_fidelity_rollback_and_ordering(self):
        fidelity = [0.0] * 13
        fidelity[2] = .7
        fidelity[4] = .9
        fidelity[5] = .4
        fidelity[8] = .95
        tokens = list(range(13))
        cost, paths = oracle_optimal_paths(fidelity, tokens, [.6, .9, .95])
        self.assertEqual(cost, 14)
        self.assertEqual(paths, [(2, 4, 8)])

    def test_partition_counts_all_monotone_paths(self):
        scores = torch.zeros((2, 13))
        self.assertAlmostEqual(log_partition(scores).item(), math.log(91), places=5)

    def test_inference_uses_scores_without_fidelity(self):
        scores = torch.zeros((2, 13))
        scores[0, 3] = 10
        scores[1, 2] = 20  # forbidden by monotonicity after choosing 3
        scores[1, 4] = 9
        path = best_stop_vector(scores)
        self.assertEqual(path, (0, 2))
        self.assertLessEqual(path[0], path[1])


if __name__ == "__main__":
    unittest.main()
