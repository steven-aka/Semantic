from __future__ import annotations

import unittest

from src.evaluation.v8_first_divergence_audit import summarize


class V8FirstDivergenceAuditTest(unittest.TestCase):
    def test_summary_separates_local_rank_beam_and_pool_coverage(self) -> None:
        rows = [
            {
                "first_divergence": {
                    "depth": 2,
                    "optimal_best_rank": 9,
                    "optimal_probability_mass": 0.01,
                    "optimal_in_local_top8": False,
                    "fixed_pool_covered": True,
                    "lost_reachable_anchors": 1,
                },
                "oracle_beam_extinction_depth": 3,
                "prefixes": [{"fixed_pool_covered": True}] * 12,
            },
            {
                "first_divergence": {
                    "depth": 4,
                    "optimal_best_rank": 2,
                    "optimal_probability_mass": 0.3,
                    "optimal_in_local_top8": True,
                    "fixed_pool_covered": False,
                    "lost_reachable_anchors": 0,
                },
                "oracle_beam_extinction_depth": None,
                "prefixes": [{"fixed_pool_covered": False}] * 12,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["first_divergence"]["mean_depth"], 3)
        self.assertEqual(result["first_divergence"]["optimal_best_rank_gt8"], 1)
        self.assertEqual(result["first_divergence"]["fixed_pool_covered"], 1)
        self.assertEqual(result["beam"]["oracle_consistent_path_extinguished"], 1)
        self.assertEqual(result["decoded_prefix_fixed_pool_coverage_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
