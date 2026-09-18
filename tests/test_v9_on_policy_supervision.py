from __future__ import annotations

import unittest

from src.data.build_v9_on_policy_supervision import summarize


class V9OnPolicySupervisionTest(unittest.TestCase):
    def test_summary_splits_covered_and_uncovered_states(self) -> None:
        details = [
            {
                "example_id": "x",
                "prefixes": [
                    {
                        "fixed_pool_covered": True,
                        "chosen_is_optimal": True,
                        "chosen_lost_reachable_anchors": 0,
                        "chosen_advantage": 0.0,
                    },
                    {
                        "fixed_pool_covered": False,
                        "chosen_is_optimal": False,
                        "chosen_lost_reachable_anchors": 1,
                        "chosen_advantage": 1.5,
                    },
                ],
            }
        ]
        result = summarize(details)
        self.assertEqual(result["fixed_pool_coverage_fraction"], 0.5)
        self.assertEqual(result["covered"]["optimal_action_fraction"], 1.0)
        self.assertEqual(result["uncovered"]["lost_anchor_action_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
