from __future__ import annotations

import itertools
import random
import unittest

from src.search.predicted_mask_trajectory import best_order_from_predicted_attainment


class PredictedMaskTrajectoryTest(unittest.TestCase):
    @staticmethod
    def objective(order, attained, tokens):
        reached = attained[0]
        cumulative_tokens = 0
        mask = 0
        for packet in order:
            mask |= 1 << packet
            next_reached = max(reached, attained[mask])
            cumulative_tokens += (next_reached - reached) * tokens[mask]
            reached = next_reached
        return -reached, cumulative_tokens

    def test_prefers_early_attainment_then_lower_cost(self) -> None:
        attained = [0, 1, 0, 2, 0, 1, 1, 2]
        tokens = [0, 10, 0, 20, 0, 5, 8, 30]
        order = best_order_from_predicted_attainment(attained, tokens, 2)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[1], 1)
        self.assertEqual(sorted(order), [0, 1, 2])

    def test_matches_exhaustive_orders_on_small_random_lattices(self) -> None:
        rng = random.Random(17)
        width = 4
        for _ in range(20):
            attained = [rng.randrange(4) for _ in range(1 << width)]
            tokens = [rng.randrange(1, 100) for _ in range(1 << width)]
            order = best_order_from_predicted_attainment(attained, tokens, 3)
            best = min(
                self.objective(candidate, attained, tokens)
                for candidate in itertools.permutations(range(width))
            )
            self.assertEqual(self.objective(order, attained, tokens), best)


if __name__ == "__main__":
    unittest.main()
