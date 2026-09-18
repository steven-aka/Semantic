from __future__ import annotations

import unittest

from src.evaluation.v10_fresh_confirmation_decision import decide_v10_fresh_confirmation


def summary(successes=300):
    return {
        "complete": True,
        "examples": 300,
        "per_level": {
            str(level): {"examples": 300, "contract_successes": successes}
            for level in (0.6, 0.7, 0.8, 0.9, 0.95)
        },
        "oracle_cutoff_active_contract_success_fraction": successes / 300,
        "oracle_cutoff_all_active_contracts_success_fraction": successes / 300,
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": 0.02,
    }


class V10FreshConfirmationDecisionTest(unittest.TestCase):
    def test_complete_strong_result_passes(self):
        self.assertEqual(
            decide_v10_fresh_confirmation(summary())["decision"],
            "GO_CUTOFF_AND_CALIBRATION",
        )

    def test_risk_failure_stops(self):
        self.assertEqual(
            decide_v10_fresh_confirmation(summary(277))["decision"],
            "STOP_MASK_VALUE_AFTER_FRESH_CONFIRM",
        )

    def test_wrong_population_is_invalid(self):
        value = summary()
        value["examples"] = 299
        self.assertEqual(decide_v10_fresh_confirmation(value)["decision"], "INVALID_SUMMARY")


if __name__ == "__main__":
    unittest.main()
