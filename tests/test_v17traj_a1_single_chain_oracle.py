import unittest

from src.evaluation.v17traj_a1_single_chain_oracle import (
    inversion_distance,
    nearest_stable_order,
    trajectory,
)


class SingleChainOracleTests(unittest.TestCase):
    def test_nested_anchors_have_ordered_single_chain_stops(self):
        order = list(range(12))
        fidelity = [0.0] * 4096
        tokens = [mask.bit_count() for mask in range(4096)]
        for mask in (1, 3, 7):
            fidelity[mask] = 0.95
        result = trajectory(order, fidelity, tokens, [.6, .7, .8, .9, .95])
        self.assertTrue(result["joint_stable_complete"])
        self.assertEqual(result["stable_start_tokens"], [1] * 5)
        self.assertEqual(result["stable_confirm_tokens"], [3] * 5)


    def test_nearest_witness_uses_exact_inversion_distance(self):
        base = [2, 1, 0] + list(range(3, 12))
        fidelity = [0.0] * 4096
        for mask in (1, 3, 7):
            fidelity[mask] = .9
        order, distance = nearest_stable_order(base, fidelity, .9)
        self.assertEqual(order[:3], [0, 1, 2])
        self.assertEqual(distance, inversion_distance(order, base))
        self.assertEqual(distance, 3)
        self.assertIsNone(nearest_stable_order(base, [0.0] * 4096, .9))


if __name__ == "__main__":
    unittest.main()
