from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from src.data.schemas import ExactSearchResult, QAExample, SemanticPacket, SemanticUnit
from src.evaluation.frontier_diagnostics import diagnose_frontier
from src.representation.state_builder import build_representation, states_at_fidelity
from src.representation.token_counter import WhitespaceTokenizer
from src.search.best_nested_chain import best_nested_chain, is_nested, structural_gap_rows
from src.data.schemas import write_jsonl
from src.search.exact_frontier import attainable_levels, independent_frontier, parse_levels
from src.search.exact_search import run_exact_search
from src.target.qwen_runner import CallableTargetRunner


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()
        units = [SemanticUnit(0, "Alpha source", True), SemanticUnit(1, "Beta source", False)]
        self.example = QAExample("x", "test", "Where?", "Moon", "Alpha source Beta source", units)
        self.packets = [
            SemanticPacket("x", 0, "Alpha source", "Alpha", "Moon", 2, 1, 1),
            SemanticPacket("x", 1, "Beta source", "Beta", "detail", 2, 1, 1),
        ]

    def test_representation_and_threshold_nestedness(self) -> None:
        self.assertEqual(build_representation(self.packets, (1, 2)), "Alpha\nBeta\ndetail")
        low = states_at_fidelity(0.5, [0.4, 0.6], [0.8, 0.9])
        high = states_at_fidelity(0.85, [0.4, 0.6], [0.8, 0.9])
        self.assertTrue(is_nested(low, high))

    def test_exact_search_batches_all_states(self) -> None:
        batch_sizes = []

        class RecordingTarget(CallableTargetRunner):
            def answer_batch(inner_self, questions, contexts):
                batch_sizes.append(len(contexts))
                return super().answer_batch(questions, contexts)

        target = RecordingTarget(lambda _q, context: "Moon" if "Moon" in context else "unknown")
        results = run_exact_search(self.example, self.packets, target, self.tokenizer, batch_size=9)
        self.assertEqual(len(results), 9)
        self.assertEqual(batch_sizes, [9])
        self.assertTrue(any(result.fidelity == 1.0 for result in results))

    def test_nested_chain_and_gap(self) -> None:
        results = run_exact_search(
            self.example,
            self.packets,
            CallableTargetRunner(lambda _q, context: "Moon" if "Moon" in context else "unknown"),
            self.tokenizer,
            batch_size=9,
        )
        levels = (0.6, 0.9)
        independent = independent_frontier(results, levels)
        nested = best_nested_chain(results, levels)
        self.assertEqual(len(nested), 2)
        self.assertTrue(is_nested(nested[0]["state"], nested[1]["state"]))
        gaps = structural_gap_rows(independent, nested)
        self.assertTrue(all(row["structural_gap"] >= 0 for row in gaps))

    def test_positive_control_has_structural_gap(self) -> None:
        results = [
            ExactSearchResult("x", (1, 0), 1, "low", 1, 0.6, 1, 0.6),
            ExactSearchResult("x", (0, 2), 3, "high", 1, 0.9, 1, 0.9),
            ExactSearchResult("x", (1, 2), 4, "both", 1, 0.9, 1, 0.9),
        ]
        levels = (0.6, 0.9)
        independent = independent_frontier(results, levels)
        nested = best_nested_chain(results, levels)
        gaps = structural_gap_rows(independent, nested)
        self.assertEqual([row["structural_gap"] for row in gaps], [0, 1])

    def test_nested_chain_keeps_feasible_prefix(self) -> None:
        results = [
            ExactSearchResult("x", (0, 0), 0, "", 0, 0, 1, 0),
            ExactSearchResult("x", (1, 0), 1, "Moon", 1, 0.7, 1, 0.7),
        ]
        chain = best_nested_chain(results, (0.6, 0.8))
        self.assertTrue(chain[0]["feasible"])
        self.assertFalse(chain[1]["feasible"])

    def test_custom_fidelity_grid_parser(self) -> None:
        self.assertEqual(parse_levels("0.2, 0.5,1.0"), (0.2, 0.5, 1.0))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_levels("0.5,0.5")

    def test_attainable_fidelity_grid(self) -> None:
        rows = [
            ExactSearchResult("x", (0,), 0, "", 0, 0, 0, 0),
            ExactSearchResult("x", (1,), 1, "", 0, 0.5, 1, 0.5),
            ExactSearchResult("x", (1,), 1, "", 0, 0.5, 1, 0.5000000000001),
            ExactSearchResult("x", (2,), 2, "", 1, 1, 1, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(attainable_levels([path]), (0.5, 1.0))

    def test_frontier_diagnostic_declares_flat_frontier_no_go(self) -> None:
        independent = [
            {
                "example_id": "x",
                "fidelity_level": str(level),
                "state": "[1, 0]",
                "tokens": "10",
                "fact_recall": "1.0",
            }
            for level in (0.6, 0.8)
        ]
        gaps = [{"structural_gap": "0"}, {"structural_gap": "0"}]
        result = diagnose_frontier(independent, gaps)
        self.assertEqual(result["state_switches"], 0)
        self.assertTrue(result["assessment"].startswith("no_go"))


if __name__ == "__main__":
    unittest.main()
