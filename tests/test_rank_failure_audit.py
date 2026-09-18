from __future__ import annotations

import unittest

from src.data.schemas import ExactSearchResult
from src.evaluation.rank_failure_audit import (
    audit_failed_order,
    clopper_pearson_lower,
    minimum_successes_for_lower_bound,
)


class RankFailureAuditTests(unittest.TestCase):
    def test_clopper_pearson_bonferroni_requirement(self) -> None:
        self.assertEqual(minimum_successes_for_lower_bound(300, 0.90, 0.01), 282)
        self.assertLess(clopper_pearson_lower(278, 300, 0.01), 0.90)
        self.assertGreaterEqual(clopper_pearson_lower(282, 300, 0.01), 0.90)

    def test_single_boundary_swap_is_identifiable(self) -> None:
        rows = []
        for mask in range(8):
            state = tuple((mask >> index) & 1 for index in range(3))
            fidelity = 0.95 if state == (1, 0, 1) else 0.5
            rows.append(
                ExactSearchResult(
                    "x", state, sum(state), "", 0.0, fidelity, 0.0, fidelity
                )
            )
        source = {
            "example_id": "x",
            "question": "which?",
            "packet_ids": [0, 1, 2],
            "packet_texts": ["first", "intruder", "missing"],
            "active_levels": [0.9],
            "pairwise_preferences": [[2, 1]],
            "never_reveal_labels": [False, True, False],
            "packet_anchor_ambiguity_fraction": 0.0,
            "identifiable_pairs": 1,
        }
        prediction = {
            "example_id": "x",
            "scores": [3.0, 2.0, 1.0],
            "learned_order": [0, 1, 2],
            "all_active_contracts_success": False,
        }
        audit = audit_failed_order(source, rows, prediction, run_label="seed")
        self.assertEqual(audit["minimum_adjacent_swaps_to_a_feasible_state"], 1)
        self.assertEqual(audit["minimum_boundary_replacements"], 1)
        self.assertEqual([row["packet_index"] for row in audit["missing_packets"]], [2])
        self.assertEqual([row["packet_index"] for row in audit["intruder_packets"]], [1])
        self.assertEqual(
            audit["repair_pair_relation_counts"],
            {"identifiable_model_violation": 1},
        )
        self.assertTrue(audit["one_edit_counterfactual"]["swap"])


if __name__ == "__main__":
    unittest.main()
