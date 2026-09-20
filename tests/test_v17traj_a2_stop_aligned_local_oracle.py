import unittest

from src.evaluation.v17traj_a2_stop_aligned_local_oracle import (
    choose,
    local_orders,
    path,
    safe,
    score,
)


class FixedStopLocalOracleTests(unittest.TestCase):
    def test_kendall_two_neighborhood(self):
        base = list(range(12))
        orders = local_orders(base)
        self.assertEqual(len(orders), len(set(orders)))
        self.assertEqual(orders[0], tuple(base))
        rank = {packet: i for i, packet in enumerate(base)}
        for order in orders:
            inversions = sum(rank[order[i]] > rank[order[j]] for i in range(12) for j in range(i + 1, 12))
            self.assertLessEqual(inversions, 2)

    def test_oracle_preserves_hits_and_token_budget(self):
        base = list(range(12))
        moved = list(range(9)) + [10, 9, 11]
        fidelity = [0.0] * 4096
        tokens = [mask.bit_count() * 10 + int(bool(mask & 1)) * 5 for mask in range(4096)]
        fidelity[path(base)[10]] = .9
        fidelity[path(moved)[10]] = .95
        levels = {.6, .7, .8, .9, .95}
        base_score = score(base, path(base), fidelity, tokens, [10] * 5, levels)
        moved_score = score(moved, path(moved), fidelity, tokens, [10] * 5, levels)
        self.assertTrue(safe(moved_score, base_score, True))
        self.assertEqual(choose([base_score, moved_score], base_score, True)["order"], moved)


if __name__ == "__main__":
    unittest.main()
