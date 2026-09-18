import unittest

from src.evaluation.v7_data_scale_decision import development_headroom


def _summary(successes: int, trajectory: float, regret: float) -> dict:
    return {
        "per_level": {"0.9": {"examples": 300, "contract_successes": successes}},
        "oracle_cutoff_all_active_contracts_success_fraction": trajectory,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": regret,
    }


class V7DataScaleDecisionTest(unittest.TestCase):
    def test_passes_only_when_all_frozen_checks_pass(self) -> None:
        result = development_headroom(_summary(282, 0.90, 0.03))
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "GO_FREEZE_NEW_TARGET_BLIND_CONFIRM_POPULATION")

    def test_failure_closes_static_one_shot_ranking(self) -> None:
        result = development_headroom(_summary(281, 0.95, 0.02))
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "STOP_STATIC_ONE_SHOT_RANKING_AFTER_V7")


if __name__ == "__main__":
    unittest.main()
