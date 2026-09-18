import unittest

from src.evaluation.v4_relational_decision import development_headroom, fresh_confirmation


def summary(success_09=282, trajectories=0.91):
    levels = {}
    for level, examples, successes in ((0.6,300,299),(0.7,300,299),(0.8,300,299),(0.9,300,success_09),(0.95,240,238)):
        levels[str(level)] = {"examples": examples, "contract_successes": successes, "contract_success_fraction": successes/examples}
    return {
        "per_level": levels,
        "oracle_cutoff_active_contract_success_fraction": 0.98,
        "oracle_cutoff_all_active_contracts_success_fraction": trajectories,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": 0.02,
    }


class V4RelationalDecisionTest(unittest.TestCase):
    def test_development_requires_282(self):
        self.assertTrue(development_headroom(summary())["passed"])
        self.assertFalse(development_headroom(summary(281))["passed"])

    def test_fresh_risk_gate_is_stricter_than_fraction(self):
        failed = fresh_confirmation(summary(280))
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["per_anchor_risk"]["0.9"]["passed"])


if __name__ == "__main__":
    unittest.main()
