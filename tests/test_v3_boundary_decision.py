import unittest

from src.evaluation.v3_boundary_decision import decide_boundary_confirmation


class V3BoundaryDecisionTest(unittest.TestCase):
    def _summary(self, successes: int) -> dict[str, object]:
        return {
            "oracle_cutoff_active_contract_success_fraction": successes / 210,
            "oracle_cutoff_all_active_contracts_success_fraction": successes / 210,
            "mean_oracle_cutoff_ranking_regret_normalized_feasible": 0.02,
            "per_level": {
                "0.9": {
                    "examples": 210,
                    "contract_success_fraction": successes / 210,
                }
            },
        }

    def test_199_of_210_passes_one_percent_cp_bound(self) -> None:
        result = decide_boundary_confirmation(
            self._summary(199),
            familywise_alpha=0.01,
            required_lower_bound=0.90,
            minimum_active_success=0.90,
            minimum_trajectory_success=0.90,
            maximum_regret=0.03,
        )
        self.assertEqual(result["decision"], "GO_IMPLEMENT_CUTOFF")

    def test_198_of_210_fails_one_percent_cp_bound(self) -> None:
        result = decide_boundary_confirmation(
            self._summary(198),
            familywise_alpha=0.01,
            required_lower_bound=0.90,
            minimum_active_success=0.90,
            minimum_trajectory_success=0.90,
            maximum_regret=0.03,
        )
        self.assertEqual(result["decision"], "STOP_V3_RANKING")


if __name__ == "__main__":
    unittest.main()
