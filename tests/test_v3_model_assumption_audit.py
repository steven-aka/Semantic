import unittest

from src.data.schemas import ExactSearchResult
from src.search.near_optimal_chain_set import near_optimal_chain_membership

from src.evaluation.v3_model_assumption_audit import (
    graph_has_cycle,
    harmfulness_stability,
    local_boundary_edges,
    memberships_for_slacks,
    summarize_prediction,
)


class V3ModelAssumptionAuditTest(unittest.TestCase):
    def test_multi_slack_dp_matches_frozen_single_slack_oracle(self) -> None:
        exact = [
            ExactSearchResult(
                example_id="x",
                state=tuple(int(bool(mask & (1 << bit))) for bit in range(3)),
                tokens=10 + 3 * mask.bit_count(),
                prediction="",
                answer_em=0.0,
                answer_f1=0.0,
                fact_recall=0.0,
                fidelity=(0.95 if mask & 1 and not mask & 2 else 0.65 if mask & 1 else 0.0),
            )
            for mask in range(8)
        ]
        slacks = [0.0, 0.005, 0.05]
        joint = memberships_for_slacks(exact, [0.6, 0.9], slacks)
        for slack in slacks:
            separate = near_optimal_chain_membership(exact, [0.6, 0.9], normalized_slack=slack)
            self.assertEqual(joint[slack]["state_masks_by_level"], separate["state_masks_by_level"])

    def test_cycles_are_distinct_from_direct_conflict(self) -> None:
        self.assertTrue(graph_has_cycle(3, {(0, 1), (1, 2), (2, 0)}))
        self.assertFalse(graph_has_cycle(3, {(0, 1), (0, 2)}))

    def test_harmful_sign_flip_does_not_imply_order_impossibility(self) -> None:
        # Packet 1 crosses the contract at mask 1 but is helpful at mask 5.
        fidelity = {mask: 0.0 for mask in range(8)}
        fidelity[1] = 0.9
        fidelity[5] = 0.9
        fidelity[3] = 0.2
        fidelity[7] = 1.0
        groups = harmfulness_stability(fidelity, [[1, 5]], [0.9], 3)
        self.assertEqual(groups["harmful_packet_anchor_groups"], 1)
        self.assertEqual(groups["strict_sign_flip_groups"], 1)
        self.assertEqual(groups["contract_crossing_flip_groups"], 1)
        self.assertEqual(local_boundary_edges(fidelity, [[1]], [0.9], 3), {(0, 1)})

    def test_complete_boundary_uses_strict_score_separation(self) -> None:
        predictions = {
            "x": {"scores": [1.0, 0.0, 2.0], "all_active_contracts_success": True},
            "y": {"scores": [0.0, 0.0], "all_active_contracts_success": False},
        }
        boundaries = {
            "x": {"pairwise_preferences": [[0, 1], [2, 1]]},
            "y": {"pairwise_preferences": [[0, 1]]},
        }
        result = summarize_prediction(predictions, boundaries)
        self.assertEqual(result["boundary_pair_accuracy"], 2 / 3)
        self.assertEqual(result["complete_boundary_separation"], 0.5)


if __name__ == "__main__":
    unittest.main()
