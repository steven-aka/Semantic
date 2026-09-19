import unittest

from src.evaluation.v17e2_evidence_progress_target_audit import preference_state


class V17E2ProgressTargetTest(unittest.TestCase):
    def test_preference_state_preserves_all_tied_best_actions(self):
        row = {"values": [(4, 0.2), (7, 0.2), (9, -0.1)]}
        result = preference_state(row, 1e-9)
        self.assertEqual(result["positive"], [4, 7])
        self.assertEqual(result["negative"], [9])

    def test_all_tied_state_is_excluded(self):
        self.assertIsNone(preference_state({"values": [(1, 0.0), (2, 0.0)]}, 0.0))


if __name__ == "__main__":
    unittest.main()
