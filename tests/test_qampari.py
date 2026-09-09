from __future__ import annotations

import unittest

from src.data.qampari_controlled import build_controlled_examples, select_query_sentence
from src.data.qampari_select import select_qampari_development
from src.data.schemas import ExactSearchResult, SemanticPacket
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.qampari_full_context import run_qampari_full_context
from src.evaluation.qampari_gate import _checks
from src.search.qampari_exact_search import (
    run_qampari_exact_search,
    validate_qampari_exact_rows,
)
from src.representation.lossless_packetizer import (
    build_lossless_packets,
    validate_lossless_partition,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


class _Target:
    def generate_batch(self, questions, contexts):
        del questions
        return [
            "<answer>Item 0 # Item 1</answer>" if context else "<answer>wrong</answer>"
            for context in contexts
        ]


def _row(qid: str, offset: int = 0) -> dict:
    return {
        "qid": qid,
        "question_text": f"List items for {qid}?",
        "answer_list": [
            {
                "answer_text": f"Item {offset + index}",
                "aliases": [f"Alias {offset + index}"],
                "answer_url": f"https://example/{offset + index}",
                "proof": [{"proof_text": f"Item {offset + index} is supported."}],
            }
            for index in range(10)
        ],
    }


class QampariTests(unittest.TestCase):
    def test_query_sentence_selection_is_label_free_and_source_preserving(self) -> None:
        proof = (
            "He studied engineering.\n"
            "He served in the Singapore Armed Forces from 1992 to 1995.\n"
            "He later became an executive."
        )
        selected = select_query_sentence(proof, "Who served with the Singapore Armed Forces?")
        self.assertEqual(
            selected, "He served in the Singapore Armed Forces from 1992 to 1995."
        )

    def test_alias_aware_list_metric_penalizes_hallucinations(self) -> None:
        atoms = [
            {"answer_text": "Alpha", "aliases": ["Alpha alias"]},
            {"answer_text": "Beta", "aliases": ["Beta alias"]},
        ]
        metrics = qampari_list_metrics(["Alpha alias", "wrong"], atoms)
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_list_parser_is_deterministic_and_deduplicates(self) -> None:
        self.assertEqual(
            parse_list_prediction("<answer>Alpha # Beta; alpha\n3. Gamma</answer>"),
            ["Alpha", "Beta", "Gamma"],
        )

    def test_controlled_builder_has_ten_atoms_five_relevant_groups_and_distractor(self) -> None:
        examples, annotations = build_controlled_examples(
            [_row("q1"), _row("q2", 100), _row("q3", 200)],
            _Tokenizer(),
            n_examples=2,
        )
        self.assertEqual(len(examples), 2)
        for example, annotation in zip(examples, annotations):
            self.assertEqual(len(example.units), 6)
            self.assertFalse(any(unit.supporting for unit in example.units))
            self.assertEqual(len(annotation["answer_atoms"]), 10)
            self.assertEqual(sorted(map(len, annotation["unit_answer_indices"])), [0, 2, 2, 2, 2, 2])
            self.assertLessEqual(max(len(unit.text.split()) for unit in example.units), 256)

    def test_full_context_audit_uses_same_hard_list_metric_for_empty_and_full(self) -> None:
        examples, annotation_rows = build_controlled_examples(
            [_row("q1"), _row("q2", 100)], _Tokenizer(), n_examples=1
        )
        annotations = {row["example_id"]: row for row in annotation_rows}
        rows = run_qampari_full_context(
            examples, annotations, _Target(), _Tokenizer(), batch_size=1
        )
        self.assertEqual(rows[0]["full_metrics"]["correct"], 2)
        self.assertEqual(rows[0]["full_metrics"]["recall"], 0.2)
        self.assertEqual(rows[0]["empty_metrics"]["correct"], 0)
        self.assertGreater(rows[0]["f1_context_gain"], 0)

    def test_development_selector_applies_one_conjunctive_rule_in_source_order(self) -> None:
        examples, annotations = build_controlled_examples(
            [_row("q1"), _row("q2", 100), _row("q3", 200)],
            _Tokenizer(),
            n_examples=2,
        )
        baseline = [
            {
                "example_id": examples[0].example_id,
                "full_metrics": {"f1": 0.9},
                "empty_metrics": {"f1": 0.1},
                "f1_context_gain": 0.8,
            },
            {
                "example_id": examples[1].example_id,
                "full_metrics": {"f1": 0.7},
                "empty_metrics": {"f1": 0.0},
                "f1_context_gain": 0.7,
            },
        ]
        selected, selected_annotations = select_qampari_development(
            examples, annotations, baseline, n_examples=1
        )
        self.assertEqual(selected[0].example_id, examples[0].example_id)
        self.assertEqual(selected_annotations[0]["example_id"], examples[0].example_id)

    def test_exact_search_covers_all_states_and_validates_list_f1(self) -> None:
        examples, annotation_rows = build_controlled_examples(
            [_row("q1"), _row("q2", 100)], _Tokenizer(), n_examples=1
        )
        example = examples[0]
        packets = [
            SemanticPacket(
                example.example_id,
                unit.unit_id,
                unit.text,
                "gist",
                "residual",
                len(unit.text.split()),
                1,
                1,
            )
            for unit in example.units
        ]
        rows = run_qampari_exact_search(
            example,
            annotation_rows[0],
            packets,
            _Target(),
            _Tokenizer(),
            batch_size=100,
        )
        self.assertEqual(len(rows), 729)
        self.assertEqual(validate_qampari_exact_rows(example, rows), [])
        invalid = list(rows[:-1])
        self.assertTrue(validate_qampari_exact_rows(example, invalid))
        wrong_fidelity = list(rows)
        first = wrong_fidelity[0]
        wrong_fidelity[0] = ExactSearchResult(
            first.example_id,
            first.state,
            first.tokens,
            first.prediction,
            first.answer_em,
            first.answer_f1,
            first.fact_recall,
            1.0,
        )
        self.assertIn(
            "fidelity/list-F1 mismatch",
            validate_qampari_exact_rows(example, wrong_fidelity),
        )

    def test_lossless_packetizer_is_disjoint_complete_and_label_free(self) -> None:
        examples, _ = build_controlled_examples(
            [_row("q1"), _row("q2", 100)], _Tokenizer(), n_examples=1
        )
        packets = build_lossless_packets(examples, _Tokenizer())[examples[0].example_id]
        self.assertEqual(len(packets), 6)
        for packet in packets:
            self.assertEqual(validate_lossless_partition(packet, _Tokenizer()), [])
            self.assertLessEqual(packet.gist_tokens / packet.source_tokens, 0.5)
            self.assertNotEqual(packet.gist, packet.residual)

    def test_frozen_gate_is_conjunctive(self) -> None:
        thresholds = {
            "minimum_complete_examples": 30,
            "minimum_adjacent_feasible_pairs": 30,
            "minimum_strict_rate_transition_fraction": 0.2,
            "minimum_nested_reuse_fraction": 0.8,
            "maximum_example_weighted_mean_structural_gap_normalized": 0.1,
            "maximum_structural_gap_normalized_p90": 0.2,
            "minimum_mean_atom_recall_gain_0_60_to_0_90": 0.15,
            "maximum_atom_recall_decline_fraction": 0.1,
            "minimum_full_state_f1_0_8_fraction": 0.9,
            "minimum_max_f1_0_9_fraction": 0.9,
            "minimum_nonnested_independent_switches": 1,
            "require_lossless_partition": True,
        }
        metrics = {
            "errors": [], "examples": 30, "adjacent_feasible_pairs": 100,
            "strict_rate_transition_fraction": 1.0, "nested_reuse_fraction_any_tie": 0.9,
            "example_weighted_mean_structural_gap_normalized": 0.01,
            "structural_gap_normalized_p90": 0.02,
            "mean_atom_recall_gain_0_60_to_0_90": 0.4,
            "atom_recall_decline_fraction": 0.0, "full_state_f1_0_8_fraction": 0.9,
            "max_f1_0_9_fraction": 0.89, "nonnested_independent_switches": 2,
            "lossless_partition_complete": True,
        }
        checks = _checks(metrics, thresholds)
        self.assertFalse(checks["minimum_max_f1_0_9_fraction"])
        self.assertTrue(all(value for key, value in checks.items() if key != "minimum_max_f1_0_9_fraction"))


if __name__ == "__main__":
    unittest.main()
