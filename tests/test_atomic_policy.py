from __future__ import annotations

import unittest
from itertools import product

import torch

from src.model.atomic_reveal_policy import AtomicRevealPolicy, masked_ordinal_bce, trajectory_logits
from src.model.atomic_packet_ranker import AtomicPacketRanker, partial_order_rank_loss
from src.model.input_builder import build_atomic_model_input
from src.data.schemas import ExactSearchResult
from src.data.build_rank_learning_curve import build_nested_learning_curve
from src.evaluation.rank_only_gate import apply_conjunctive_gate, summarize_gate_run
from src.search.atomic_nested_chain import mask_to_state
from src.search.near_optimal_chain_set import (
    identifiable_pair_relations,
    near_optimal_chain_membership,
)
from src.search.rank_then_cut import (
    best_nested_chain_at_counts,
    best_prefix_nested_chain,
    prefix_state,
)


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return type("Encoding", (), {"input_ids": list(range(1, len(text.split()) + 1))})()


class AtomicPolicyTests(unittest.TestCase):
    @staticmethod
    def _exact_rows() -> list[ExactSearchResult]:
        fidelity = [0.0, 0.55, 0.65, 0.82, 0.60, 0.90, 0.85, 1.0]
        tokens = [0, 2, 4, 7, 3, 6, 8, 11]
        return [
            ExactSearchResult(
                example_id="rank",
                state=mask_to_state(mask, 3),
                tokens=tokens[mask],
                prediction="",
                answer_em=0.0,
                answer_f1=fidelity[mask],
                fact_recall=fidelity[mask],
                fidelity=fidelity[mask],
            )
            for mask in range(8)
        ]

    def test_policy_predicts_one_threshold_per_packet(self) -> None:
        class FakeLM(torch.nn.Module):
            config = type("Config", (), {"hidden_size": 2})()

            def forward(self, input_ids, **kwargs):
                del kwargs
                hidden = torch.arange(
                    input_ids.numel() * 2, dtype=torch.float32
                ).reshape(*input_ids.shape, 2)
                return type("Output", (), {"hidden_states": (hidden,)})()

        policy = AtomicRevealPolicy(FakeLM())
        output = policy(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 1, 1, 1]]),
            [[(0, 2), (2, 3)]],
            torch.tensor([0.6, 0.8]),
        )
        self.assertEqual(tuple(output["thresholds"].shape), (1, 2))
        self.assertEqual(tuple(output["trajectory_logits"].shape), (1, 2, 2))
        self.assertTrue(torch.all(output["thresholds"] >= 0))
        self.assertTrue(torch.all(output["thresholds"] <= 1))

    def test_ranker_scores_packets_and_uses_relative_preferences(self) -> None:
        class FakeLM(torch.nn.Module):
            config = type("Config", (), {"hidden_size": 2})()

            def forward(self, input_ids, **kwargs):
                del kwargs
                hidden = torch.arange(
                    input_ids.numel() * 2, dtype=torch.float32
                ).reshape(*input_ids.shape, 2)
                return type("Output", (), {"hidden_states": (hidden,)})()

        ranker = AtomicPacketRanker(FakeLM(), head_hidden_size=4)
        output = ranker(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 1, 1, 1]]),
            [[(0, 2), (2, 3)]],
            [[(0, 1)]],
        )
        self.assertEqual(tuple(output["scores"].shape), (1, 2))
        self.assertTrue(torch.isfinite(output["loss"]))
        good = partial_order_rank_loss(torch.tensor([[2.0, -1.0]]), [[(0, 1)]])
        bad = partial_order_rank_loss(torch.tensor([[-1.0, 2.0]]), [[(0, 1)]])
        self.assertLess(float(good), float(bad))

    def test_learning_curve_is_nested_and_requires_frozen_size(self) -> None:
        rows = [
            {"example_id": str(index), "pairwise_preferences": [[0, 1]]}
            for index in range(2000)
        ]
        curve = build_nested_learning_curve(rows)
        self.assertEqual(list(curve), [500, 1000, 2000])
        self.assertEqual(curve[500], curve[1000][:500])
        self.assertEqual(curve[1000], curve[2000][:1000])
        with self.assertRaises(ValueError):
            build_nested_learning_curve(rows[:-1])

    def test_rank_only_gate_uses_highest_active_anchor_and_is_conjunctive(self) -> None:
        summary = {
            "complete": True,
            "examples": 2,
            "oracle_cutoff_active_contract_success_fraction": 0.98,
            "oracle_cutoff_all_active_contracts_success_fraction": 0.95,
            "mean_oracle_cutoff_ranking_regret_normalized_feasible": 0.02,
            "identifiable_pairwise_accuracy": 0.90,
        }
        details = [
            {"anchors": [{"contract_success": False}, {"contract_success": True}]},
            {"anchors": [{"contract_success": True}, {"contract_success": False}]},
        ]
        run = summarize_gate_run(summary, details)
        self.assertEqual(run["highest_active_anchor_success"], 0.5)
        thresholds = {
            "minimum_mean_active_contract_success": 0.97,
            "minimum_mean_all_active_trajectory_success": 0.90,
            "minimum_mean_highest_active_anchor_success": 0.90,
            "maximum_mean_normalized_ranking_regret_feasible": 0.03,
            "minimum_mean_identifiable_pairwise_accuracy": 0.85,
        }
        gate = apply_conjunctive_gate([run, run, run], thresholds)
        self.assertFalse(gate["scientific_gate_passed"])
        self.assertEqual(gate["decision"], "NO_GO_STOP_AT_RANKING")

    def test_input_builder_records_nonoverlapping_packet_spans(self) -> None:
        value = build_atomic_model_input(_Tokenizer(), "who?", ["alpha beta", "gamma"])
        self.assertEqual(len(value.packet_spans), 2)
        self.assertLessEqual(value.packet_spans[0][1], value.packet_spans[1][0])
        self.assertEqual(value.packet_spans[0][1] - value.packet_spans[0][0], 2)
        self.assertEqual(value.packet_spans[1][1] - value.packet_spans[1][0], 1)

    def test_ordinal_logits_are_monotone_and_masked(self) -> None:
        thresholds = torch.tensor([[0.75]])
        levels = torch.tensor([0.60, 0.70, 0.80, 0.90])
        logits = trajectory_logits(thresholds, levels, 0.05)
        self.assertTrue(torch.all(logits[..., 1:] > logits[..., :-1]))
        labels = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
        mask = torch.tensor([[[True, True, True, False]]])
        loss_a = masked_ordinal_bce(
            thresholds, levels, labels, mask, temperature=0.05
        )
        labels[..., -1] = 1.0
        loss_b = masked_ordinal_bce(
            thresholds, levels, labels, mask, temperature=0.05
        )
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=7)

    def test_fixed_order_oracle_cutoff_matches_brute_force(self) -> None:
        rows = self._exact_rows()
        levels = (0.6, 0.8)
        order = (2, 0, 1)
        chain = best_prefix_nested_chain(rows, order, levels)
        candidates = [prefix_state(order, count) for count in range(4)]
        by_state = {row.state: row for row in rows}
        brute = min(
            by_state[low].tokens + by_state[high].tokens
            for low, high in product(candidates, repeat=2)
            if sum(low) <= sum(high)
            and by_state[low].fidelity >= levels[0]
            and by_state[high].fidelity >= levels[1]
        )
        self.assertEqual(chain[-1]["cumulative_tokens"], brute)
        self.assertLessEqual(chain[0]["cutoff_count"], chain[1]["cutoff_count"])

    def test_oracle_order_with_fixed_counts_maximizes_contracts(self) -> None:
        rows = self._exact_rows()
        levels = (0.6, 0.8)
        chain = best_nested_chain_at_counts(rows, (1, 2), levels)
        self.assertTrue(all(row["feasible"] for row in chain))
        self.assertEqual([sum(row["state"]) for row in chain], [1, 2])
        low, high = (tuple(row["state"]) for row in chain)
        self.assertTrue(all(a <= b for a, b in zip(low, high)))

    def test_near_optimal_membership_matches_chain_brute_force(self) -> None:
        rows = self._exact_rows()
        levels = (0.6, 0.8)
        by_mask = {mask: row for mask, row in enumerate(rows)}
        chains = [
            (low, high)
            for low, high in product(range(8), repeat=2)
            if low & high == low
            and by_mask[low].fidelity >= levels[0]
            and by_mask[high].fidelity >= levels[1]
        ]
        optimum = min(sum(by_mask[mask].tokens for mask in chain) for chain in chains)
        best = [
            chain
            for chain in chains
            if sum(by_mask[mask].tokens for mask in chain) == optimum
        ]
        membership = near_optimal_chain_membership(rows, levels, normalized_slack=0.0)
        self.assertEqual(
            [set(value) for value in membership["state_masks_by_level"]],
            [{chain[index] for chain in best} for index in range(2)],
        )
        relations = identifiable_pair_relations(membership, 3)
        self.assertTrue(all(value in (-1, 1) for value in relations.values()))


if __name__ == "__main__":
    unittest.main()
