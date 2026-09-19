import unittest

from src.evaluation.v17e1ab_frontier_information_audit import random_recall


class V17E1FrontierAuditTest(unittest.TestCase):
    def test_random_recall_accounts_for_viable_prevalence(self):
        self.assertAlmostEqual(random_recall(10, 8, 1), 0.8)
        self.assertAlmostEqual(random_recall(10, 8, 2), 1.0 - 1.0 / 45.0)
        self.assertEqual(random_recall(10, 1, 10), 1.0)


if __name__ == "__main__":
    unittest.main()
