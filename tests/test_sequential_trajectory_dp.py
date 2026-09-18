from __future__ import annotations

import unittest

from src.data.schemas import ExactSearchResult
from src.search.atomic_nested_chain import best_binary_nested_chain, mask_to_state
from src.search.rank_then_cut import best_prefix_nested_chain
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def row(mask: int, fidelity: float, tokens: int | None = None) -> ExactSearchResult:
    state = mask_to_state(mask, 3)
    return ExactSearchResult(
        example_id="example",
        state=state,
        tokens=mask.bit_count() if tokens is None else tokens,
        prediction="",
        answer_em=0.0,
        answer_f1=0.0,
        fact_recall=0.0,
        fidelity=fidelity,
    )


class SequentialTrajectoryDPTest(unittest.TestCase):
    def test_matches_existing_nested_chain_objective(self) -> None:
        fidelity = [0.0, 0.6, 0.0, 0.8, 0.7, 0.6, 0.9, 0.95]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        levels = [0.6, 0.7, 0.8, 0.9]
        dp = SequentialTrajectoryDP(exact, levels)
        rollout = dp.canonical_rollout()
        nested = best_binary_nested_chain(exact, levels)
        evaluated = best_prefix_nested_chain(exact, rollout["order"], levels)
        nested_cost = sum(item["tokens"] for item in nested if item["feasible"])
        evaluated_cost = sum(item["tokens"] for item in evaluated if item["feasible"])
        self.assertEqual(len(levels), rollout["reached_levels"])
        self.assertEqual(nested_cost, rollout["cumulative_tokens"])
        self.assertEqual(nested_cost, evaluated_cost)

    def test_history_is_part_of_the_oracle_state(self) -> None:
        # At selected set {0}, a history that already reached 0.8 has no need
        # to prioritize packet 1, while a history that only reached 0.6 does.
        fidelity = [0.0, 0.6, 0.8, 0.8, 0.0, 0.6, 0.8, 0.8]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.8])
        low_history = dp.optimal_actions(1, 1)
        high_history = dp.optimal_actions(1, 2)
        self.assertNotEqual(low_history, high_history)

    def test_slack_expands_set_without_admitting_less_feasible_action(self) -> None:
        fidelity = [0.0, 0.6, 0.6, 0.8, 0.0, 0.6, 0.8, 0.8]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.8])
        exact_actions = set(dp.optimal_actions(0, 0))
        slack_actions = set(dp.optimal_actions(0, 0, normalized_slack=1.0))
        self.assertTrue(exact_actions <= slack_actions)
        best_reached = dp.value(0, 0).reached_levels
        for packet, value in dp.action_values(0, 0):
            if packet in slack_actions:
                self.assertEqual(best_reached, value.reached_levels)

    def test_action_advantages_preserve_lexicographic_optimum(self) -> None:
        fidelity = [0.0, 0.6, 0.0, 0.8, 0.7, 0.6, 0.9, 0.95]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.7, 0.8, 0.9])
        advantages = dp.action_advantages(0, dp.attained[0])
        optimal = set(dp.optimal_actions(0, dp.attained[0]))
        zero = {item.packet for item in advantages if item.scalar == 0.0}
        self.assertEqual(zero, optimal)
        self.assertTrue(all(item.scalar >= 0.0 for item in advantages))
        self.assertTrue(
            all(item.scalar > 1.0 for item in advantages if item.lost_reachable_anchors > 0)
        )

    def test_one_run_supervision_adds_deterministic_recovery_histories(self) -> None:
        fidelity = [0.0, 0.6, 0.0, 0.8, 0.0, 0.6, 0.8, 0.9]
        exact = [row(mask, value) for mask, value in enumerate(fidelity)]
        dp = SequentialTrajectoryDP(exact, [0.6, 0.8, 0.9])
        first = dp.one_run_supervision(seed=17, oracle_rollouts=2, random_rollouts=2)
        second = dp.one_run_supervision(seed=17, oracle_rollouts=2, random_rollouts=2)
        self.assertEqual(first, second)
        self.assertGreater(len(first), dp.width)
        self.assertTrue(any(item.history and item.history[0] not in dp.optimal_actions(0, 0) for item in first))
        for item in first:
            self.assertEqual(
                item.optimal_actions,
                dp.optimal_actions(item.selected_mask, item.reached_levels),
            )


if __name__ == "__main__":
    unittest.main()
