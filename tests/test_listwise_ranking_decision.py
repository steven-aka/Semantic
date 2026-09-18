from __future__ import annotations

import unittest
from unittest.mock import patch

from src.evaluation.listwise_ranking_decision import decide_listwise_revision


class ListwiseRankingDecisionTests(unittest.TestCase):
    def _record(self, seed: int, successes: int, loss: float) -> dict:
        details = [
            {
                "anchors": [
                    {"fidelity_level": 0.90, "contract_success": index < successes}
                ]
            }
            for index in range(300)
        ]
        return {
            "seed": seed,
            "checkpoint": f"checkpoint-{seed}",
            "best_validation_listwise_nll": loss,
            "gate_metrics": {
                "active_contract_success": 0.98,
                "all_active_trajectory_success": 0.95,
                "highest_active_anchor_success": 0.95,
                "normalized_ranking_regret_feasible": 0.02,
                "identifiable_pairwise_accuracy": 0.90,
            },
            "details": details,
        }

    @patch(
        "pathlib.Path.read_text",
        return_value='{"ranking_only_gate_before_cutoff": {"minimum_mean_active_contract_success": 0.97, "minimum_mean_all_active_trajectory_success": 0.90, "minimum_mean_highest_active_anchor_success": 0.90, "maximum_mean_normalized_ranking_regret_feasible": 0.03, "minimum_mean_identifiable_pairwise_accuracy": 0.85}}',
    )
    def test_selects_lowest_loss_eligible_seed(self, _read_text) -> None:
        config = {"parent_protocol": "unused"}
        result = decide_listwise_revision(
            [self._record(1, 282, 2.0), self._record(2, 281, 1.0)], config
        )
        self.assertEqual(result["decision"], "GO_FREEZE_CUTOFF")
        self.assertEqual(result["selected_seed"], 1)

    @patch(
        "pathlib.Path.read_text",
        return_value='{"ranking_only_gate_before_cutoff": {"minimum_mean_active_contract_success": 0.97, "minimum_mean_all_active_trajectory_success": 0.90, "minimum_mean_highest_active_anchor_success": 0.90, "maximum_mean_normalized_ranking_regret_feasible": 0.03, "minimum_mean_identifiable_pairwise_accuracy": 0.85}}',
    )
    def test_stops_without_risk_headroom(self, _read_text) -> None:
        config = {"parent_protocol": "unused"}
        result = decide_listwise_revision([self._record(1, 281, 1.0)], config)
        self.assertEqual(result["decision"], "STOP_V2_1_RANKING")
        self.assertIsNone(result["selected_seed"])


if __name__ == "__main__":
    unittest.main()
