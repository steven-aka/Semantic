from __future__ import annotations

import unittest

from src.data.schemas import QAExample, SemanticUnit
from src.evaluation.fact_coverage import (
    binary_verdict,
    map_facts_to_units,
    validate_fact_coverage,
)
from src.evaluation.full_context import cached_full_context_valid
from src.evaluation.qa_metrics import exact_match, supporting_fact_recall, token_f1
from src.evaluation.rescore_fact_fidelity import content_fact_recall
from src.representation.token_counter import WhitespaceTokenizer


class MetricTests(unittest.TestCase):
    def test_hotpot_style_metrics(self) -> None:
        self.assertEqual(exact_match("The Moon", "moon"), 1.0)
        self.assertAlmostEqual(token_f1("red moon", "moon"), 2 / 3)
        self.assertEqual(supporting_fact_recall((1, 0, 2), [True, False, True]), 1.0)

    def test_gold_fact_maps_to_exact_current_unit(self) -> None:
        example = QAExample(
            "x",
            "hotpotqa",
            "Question?",
            "answer",
            "Alpha fact. Other text.",
            [
                SemanticUnit(0, "Other text.", False, "Other"),
                SemanticUnit(1, "Alpha fact. More detail.", True, "Alpha"),
            ],
        )
        facts = [{"title": "Alpha", "sentence_id": 0, "text": " Alpha fact. "}]
        self.assertEqual(map_facts_to_units(example, facts)[0]["unit_id"], 1)

    def test_content_fact_recall_distinguishes_gist_and_full(self) -> None:
        coverage = {
            "facts": [
                {"unit_id": 0, "gist_supported": False, "full_supported": True},
                {"unit_id": 2, "gist_supported": True, "full_supported": True},
            ]
        }
        self.assertEqual(content_fact_recall((1, 0, 1), coverage), 0.5)
        self.assertEqual(content_fact_recall((2, 0, 1), coverage), 1.0)
        self.assertTrue(binary_verdict("<answer>yes</answer>"))
        self.assertFalse(binary_verdict("no."))

    def test_missing_label_free_fact_contributes_zero_recall(self) -> None:
        example = QAExample(
            "x", "hotpotqa", "Question?", "answer", "Noise.",
            [SemanticUnit(0, "Title: Noise\nNoise.", False, "Noise", 0)],
        )
        facts = [{"title": "Gold", "sentence_id": 0, "text": "Gold fact."}]
        mapped = map_facts_to_units(example, facts, allow_missing=True)
        self.assertIsNone(mapped[0]["unit_id"])
        coverage = {"facts": [{**mapped[0], "gist_supported": False, "full_supported": False}]}
        self.assertEqual(content_fact_recall((2,), coverage), 0.0)

    def test_fact_coverage_validation_rejects_wrong_closure(self) -> None:
        example = QAExample(
            "x",
            "hotpotqa",
            "Question?",
            "answer",
            "Alpha fact.",
            [SemanticUnit(0, "Alpha fact.", True, "Alpha")],
        )
        raw = {
            "context": {"title": ["Alpha"], "sentences": [["Alpha fact."]]},
            "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        }
        coverage = [{
            "example_id": "x",
            "facts": [{
                "title": "Alpha", "sentence_id": 0, "text": "Alpha fact.",
                "unit_id": 0, "gist_prediction": "yes", "full_prediction": "no",
                "gist_supported": True, "full_supported_raw": False,
                "full_supported": False,
            }],
        }]
        self.assertTrue(validate_fact_coverage([example], {"x": raw}, coverage))

    def test_full_context_cache_validation(self) -> None:
        example = QAExample(
            "x", "test", "Question?", "moon", "red moon",
            [SemanticUnit(0, "red moon")],
        )
        row = {
            "example_id": "x", "raw_prediction": "<answer>moon</answer>",
            "prediction": "moon", "gold": "moon", "em": 1.0, "f1": 1.0,
            "context_tokens": 2, "eligible_for_supervision": True,
        }
        self.assertTrue(
            cached_full_context_valid([example], [row], [example], WhitespaceTokenizer())
        )
        row["context_tokens"] = 3
        self.assertFalse(
            cached_full_context_valid([example], [row], [example], WhitespaceTokenizer())
        )


if __name__ == "__main__":
    unittest.main()
