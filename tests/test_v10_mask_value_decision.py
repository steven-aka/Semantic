from __future__ import annotations

import unittest

from src.evaluation.v10_mask_value_decision import decide_v10_probe


def summary(successes: int, trajectories: float = 0.90, regret: float = 0.03):
    return {
        "complete": True,
        "examples": 300,
        "per_level": {"0.9": {"examples": 300, "contract_successes": successes}},
        "oracle_cutoff_all_active_contracts_success_fraction": trajectories,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": regret,
    }


class V10MaskValueDecisionTest(unittest.TestCase):
    def test_go_requires_all_three_gates(self) -> None:
        self.assertEqual(
            decide_v10_probe(summary(282))["decision"],
            "GO_FREEZE_SAME_HEAD_FOR_NEW_CONFIRMATION",
        )
        self.assertEqual(
            decide_v10_probe(summary(282, trajectories=0.899))["decision"],
            "STOP_MASK_VALUE_HYPOTHESIS",
        )
        self.assertEqual(
            decide_v10_probe(summary(282, regret=0.0301))["decision"],
            "STOP_MASK_VALUE_HYPOTHESIS",
        )

    def test_primary_boundary_is_frozen(self) -> None:
        self.assertEqual(
            decide_v10_probe(summary(281))["decision"],
            "INCONCLUSIVE_NO_FRESH_ROLE",
        )
        self.assertEqual(
            decide_v10_probe(summary(278))["decision"],
            "STOP_MASK_VALUE_HYPOTHESIS",
        )

    def test_incomplete_or_wrong_denominator_is_invalid(self) -> None:
        value = summary(282)
        value["complete"] = False
        self.assertEqual(decide_v10_probe(value)["decision"], "INVALID_SUMMARY")
        value = summary(282)
        value["examples"] = 400
        self.assertEqual(decide_v10_probe(value)["decision"], "INVALID_SUMMARY")

    def test_missing_regret_is_invalid(self) -> None:
        value = summary(282)
        value["mean_oracle_cutoff_ranking_regret_normalized_feasible"] = None
        self.assertEqual(decide_v10_probe(value)["decision"], "INVALID_SUMMARY")


if __name__ == "__main__":
    unittest.main()
