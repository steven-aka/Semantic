import unittest

from src.evaluation.v17f0_state_sufficiency_target_identifiability import backward_reachability, robustness


class V17F0RobustnessTest(unittest.TestCase):
    def test_backward_reachability_and_shallow_fraction(self):
        fidelity = [0.0] * 8
        fidelity[0b011] = 0.9
        reach = backward_reachability(fidelity, width=3)
        self.assertTrue(reach[0])
        self.assertTrue(reach[0b001])
        self.assertFalse(reach[0b100])
        met = [value >= 0.9 for value in fidelity]
        r1, r2, families = robustness(0, 0, reach, met, width=3)
        self.assertEqual(families, 1)
        self.assertEqual(r1, 0.5)
        self.assertEqual(r2, 0.5)


if __name__ == "__main__":
    unittest.main()
