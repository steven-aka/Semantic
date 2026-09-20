import unittest

from src.evaluation.v17traj_a3_0_local_edit_label_audit import label


class LocalEditLabels(unittest.TestCase):
    def test_strict_benefit_and_incomparable_tradeoffs(self):
        baseline = ((True, True, False, False, None), 100)
        self.assertEqual(label(((True, True, True, False, None), 100), baseline), "beneficial")
        self.assertEqual(label(((True, True, False, False, None), 90), baseline), "beneficial")
        self.assertEqual(label(((True, True, True, False, None), 110), baseline), "tradeoff")
        self.assertEqual(label(((False, True, True, False, None), 90), baseline), "tradeoff")
        self.assertEqual(label(((False, True, False, False, None), 100), baseline), "harmful")
        self.assertEqual(label(baseline, baseline), "neutral")


if __name__ == "__main__":
    unittest.main()
