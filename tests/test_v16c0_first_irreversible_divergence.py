import unittest

from src.evaluation.v16c0_first_irreversible_divergence import first_zero_depth, state_viability
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from tests.test_sequential_trajectory_dp import row


class V16C0Test(unittest.TestCase):
    def test_contract_and_regret_viability_are_distinct(self):
        fidelity = [0.0, 0.5, 0.7, 0.9, 0.0, 0.5, 0.7, 0.9]
        dp = SequentialTrajectoryDP([row(mask, value) for mask, value in enumerate(fidelity)], [0.5, 0.9])
        oracle = dp.value(0, dp.attained[0]).additional_cumulative_tokens
        contract, qualified = state_viability(dp, 0, dp.attained[0], 0, oracle_tokens=oracle, full_tokens=dp.tokens[-1], regret_limit=0.0)
        self.assertTrue(contract)
        self.assertTrue(qualified)
        contract, qualified = state_viability(dp, 0, dp.attained[0], 1, oracle_tokens=oracle, full_tokens=dp.tokens[-1], regret_limit=0.0)
        self.assertTrue(contract)
        self.assertFalse(qualified)

    def test_first_zero_depth(self):
        trace = [{"depth": 1, "alive": 2}, {"depth": 2, "alive": 0}, {"depth": 3, "alive": 0}]
        self.assertEqual(first_zero_depth(trace, "alive"), 2)
        self.assertIsNone(first_zero_depth(trace, "missing") if False else first_zero_depth([{"depth": 1, "alive": 1}], "alive"))


if __name__ == "__main__":
    unittest.main()
