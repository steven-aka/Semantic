from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.fact_atomic_pilot import build_fact_atomic_example
from src.data.label_free_pilot import (
    bm25_scores,
    build_label_free_example,
    unit_selection_signature,
)
from src.data.normalize_hotpot import normalize_row, select_controlled_units
from src.data.multihop_retrieval import POLICIES
from src.data.semantic_retrieval import parse_selection, selector_prompt
from src.data.eligible_semantic_pilot import _load_raw as load_v0_5_raw
from src.data.schemas import QAExample, SemanticUnit, read_jsonl, write_jsonl
from src.data.segment import segment_document
from src.representation.token_counter import WhitespaceTokenizer, encode_text


class DataTests(unittest.TestCase):
    def test_label_free_sentence_atomic_selection_ignores_labels(self) -> None:
        row = {
            "id": "lf1",
            "question": "Where was Ada born?",
            "answer": "London",
            "context": {
                "title": ["Ada", "Noise"],
                "sentences": [
                    ["Ada was a mathematician.", "Ada was born in London."],
                    ["Noise one.", "Noise two.", "Noise three.", "Noise four."],
                ],
            },
            "supporting_facts": {"title": ["Ada"], "sent_id": [1]},
        }
        tokenizer = WhitespaceTokenizer()
        original = build_label_free_example(row, tokenizer, n_units=6)
        self.assertIsNotNone(original)
        perturbed = {**row, "answer": "ERASED", "supporting_facts": {"title": [], "sent_id": []}}
        rebuilt = build_label_free_example(perturbed, tokenizer, n_units=6)
        assert original is not None and rebuilt is not None
        self.assertEqual(unit_selection_signature(original), unit_selection_signature(rebuilt))
        self.assertTrue(all(not unit.supporting for unit in original.units))
        self.assertEqual([unit.source_sentence_id for unit in original.units[:2]], [0, 1])

    def test_bm25_prefers_query_matching_sentence(self) -> None:
        scores = bm25_scores("Ada London", ["Ada was born in London", "unrelated text"])
        self.assertGreater(scores[0], scores[1])

    def test_multihop_policies_are_six_unit_and_label_invariant(self) -> None:
        row = {
            "id": "mh1", "question": "How are Alpha and Beta related?", "answer": "x",
            "context": {
                "title": ["Alpha", "Beta", "Noise"],
                "sentences": [
                    ["Alpha links to Beta.", "Alpha detail one.", "Alpha detail two."],
                    ["Beta links to Alpha.", "Beta detail one.", "Beta detail two."],
                    ["Noise one.", "Noise two.", "Noise three."],
                ],
            },
            "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        }
        altered = {**row, "answer": "erased", "supporting_facts": {"title": [], "sent_id": []}}
        for policy in POLICIES.values():
            selected = policy(row, 1.2, 0.75)
            self.assertEqual(len(selected), 6)
            self.assertEqual(selected, policy(altered, 1.2, 0.75))

    def test_semantic_selector_prompt_is_label_free_and_parser_is_strict(self) -> None:
        row = {
            "id": "selector1", "question": "What connects Alpha and Beta?", "answer": "secret",
            "context": {
                "title": ["Alpha", "Beta"],
                "sentences": [["A0.", "A1.", "A2."], ["B0.", "B1.", "B2."]],
            },
            "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        }
        altered = {**row, "answer": "erased", "supporting_facts": {"title": [], "sent_id": []}}
        self.assertEqual(selector_prompt(row), selector_prompt(altered))
        self.assertEqual(parse_selection('{"indices":[5,0,4,1,3,2]}', 6), [5, 0, 4, 1, 3, 2])
        with self.assertRaises(ValueError):
            parse_selection('{"indices":[0,0,1,2,3,4]}', 6)
        with self.assertRaises(ValueError):
            parse_selection('{"indices":[0,1,2,3,4,6]}', 6)

    def test_v0_5_raw_loader_symbol_is_available(self) -> None:
        # Import guard: the V0.5 population builder must remain a standalone,
        # test-importable stage rather than hidden shell logic.
        self.assertTrue(callable(load_v0_5_raw))

    def test_hotpot_normalization_and_support(self) -> None:
        row = {
            "_id": "hp1",
            "question": "Where was Ada born?",
            "answer": "London",
            "context": [
                ["Ada", ["Ada was a mathematician.", "She was born in London."]],
                ["Noise", ["A distractor sentence."]],
            ],
            "supporting_facts": [["Ada", 1]],
        }
        example = normalize_row(row)
        self.assertEqual(example.example_id, "hp1")
        self.assertTrue(example.units[0].supporting)
        self.assertFalse(example.units[1].supporting)
        selected = select_controlled_units(example, 2)
        self.assertEqual([unit.unit_id for unit in selected.units], [0, 1])

    def test_segmentation_is_bounded_and_nonoverlapping(self) -> None:
        tokenizer = WhitespaceTokenizer()
        text = "One two three four. Five six seven eight. Nine ten eleven twelve."
        chunks = segment_document(text, tokenizer, target_tokens=6, max_tokens=8)
        self.assertTrue(chunks)
        self.assertTrue(all(len(encode_text(tokenizer, chunk)) <= 8 for chunk in chunks))
        original = encode_text(tokenizer, text)
        reconstructed = [token for chunk in chunks for token in encode_text(tokenizer, chunk)]
        self.assertEqual(reconstructed, original)

    def test_fact_atomic_pilot_separates_gold_sentences(self) -> None:
        raw = {
            "id": "atomic",
            "question": "Which facts?",
            "answer": "A",
            "supporting_facts": {
                "title": ["Gold", "Gold", "Other"],
                "sent_id": [0, 1, 0],
            },
            "context": {
                "title": ["Gold", "Other", "Noise A", "Noise B", "Noise C"],
                "sentences": [
                    ["First fact.", "Second fact."],
                    ["Third fact."],
                    ["Noise one."],
                    ["Noise two."],
                    ["Noise three."],
                ],
            },
        }
        example = normalize_row(raw)
        atomic = build_fact_atomic_example(example, raw, n_units=6)
        self.assertIsNotNone(atomic)
        assert atomic is not None
        self.assertEqual(len(atomic.units), 6)
        self.assertEqual(sum(unit.supporting for unit in atomic.units), 3)
        self.assertEqual(
            [unit.text for unit in atomic.units if unit.supporting],
            ["First fact.", "Second fact.", "Third fact."],
        )

    def test_schema_jsonl_round_trip(self) -> None:
        example = QAExample("x", "test", "q", "a", "body", [SemanticUnit(0, "body")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            write_jsonl(path, [example])
            loaded = list(read_jsonl(path, QAExample))
        self.assertEqual(loaded, [example])


if __name__ == "__main__":
    unittest.main()
