import unittest

from src.evaluation.v5_tail_cost_decision import development_headroom


def summary(successes: int, trajectory: float, regret: float) -> dict:
    return {
        "per_level": {"0.9": {"examples": 300, "contract_successes": successes}},
        "oracle_cutoff_all_active_contracts_success_fraction": trajectory,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": regret,
    }


class V5TailCostDecisionTest(unittest.TestCase):
    def test_requires_all_three_development_checks(self) -> None:
        self.assertTrue(development_headroom(summary(282, 0.90, 0.03))["passed"])
        self.assertFalse(development_headroom(summary(281, 0.90, 0.03))["passed"])
        self.assertFalse(development_headroom(summary(282, 0.89, 0.03))["passed"])
        self.assertFalse(development_headroom(summary(282, 0.90, 0.031))["passed"])


if __name__ == "__main__":
    unittest.main()
