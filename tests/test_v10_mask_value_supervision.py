from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.build_v10_mask_value_supervision import build_example
from src.data.schemas import ExactSearchResult, write_jsonl
from src.search.atomic_nested_chain import mask_to_state


class V10MaskValueSupervisionTest(unittest.TestCase):
    def test_stratification_retains_each_attained_class(self) -> None:
        fidelity = [0.0, 0.6, 0.7, 0.8, 0.0, 0.6, 0.7, 0.8]
        exact = [
            ExactSearchResult(
                example_id="x",
                state=mask_to_state(mask, 3),
                tokens=mask.bit_count() + 1,
                prediction="",
                answer_em=0.0,
                answer_f1=0.0,
                fact_recall=0.0,
                fidelity=value,
            )
            for mask, value in enumerate(fidelity)
        ]
        source = {
            "example_id": "x",
            "question": "q",
            "packet_texts": ["a"] * 12,
            "packet_tokens": [1] * 12,
            "active_levels": [0.6, 0.7, 0.8],
        }
        with tempfile.TemporaryDirectory() as directory:
            write_jsonl(Path(directory) / "x.jsonl", exact)
            row = build_example(source, directory, 1, 17)
        self.assertEqual({item["attained_levels"] for item in row["mask_values"]}, {0, 1, 2, 3})
        self.assertEqual(len(row["mask_values"]), 4)
        self.assertIn(0, {item["mask"] for item in row["mask_values"]})
        self.assertIn(7, {item["mask"] for item in row["mask_values"]})
        self.assertEqual(row["active_levels"], [0.6, 0.7, 0.8, 0.9, 0.95])
        self.assertEqual(row["attainable_levels"], [0.6, 0.7, 0.8])
        self.assertTrue(all(0 < item["sampling_probability"] <= 1 for item in row["mask_values"]))

    def test_missing_intermediate_class_is_allowed(self) -> None:
        fidelity = [0.0, 0.7, 0.0, 0.7, 0.0, 0.7, 0.0, 0.7]
        exact = [
            ExactSearchResult(
                example_id="x",
                state=mask_to_state(mask, 3),
                tokens=mask.bit_count() + 1,
                prediction="",
                answer_em=0.0,
                answer_f1=0.0,
                fact_recall=0.0,
                fidelity=value,
            )
            for mask, value in enumerate(fidelity)
        ]
        source = {
            "example_id": "x",
            "question": "q",
            "packet_texts": ["a"] * 12,
            "packet_tokens": [1] * 12,
            "active_levels": [0.6, 0.7],
        }
        with tempfile.TemporaryDirectory() as directory:
            write_jsonl(Path(directory) / "x.jsonl", exact)
            row = build_example(source, directory, 1, 17)
        self.assertEqual({item["attained_levels"] for item in row["mask_values"]}, {0, 2})


if __name__ == "__main__":
    unittest.main()
