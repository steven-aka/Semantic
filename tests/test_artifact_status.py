from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.schemas import (
    ExactSearchResult,
    QAExample,
    SemanticPacket,
    SemanticUnit,
    write_jsonl,
)
from src.evaluation.artifact_status import exact_status, packet_status, validate_exact_rows


class ArtifactStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.example = QAExample(
            "example",
            "test",
            "question",
            "answer",
            "Alpha beta gamma delta.",
            [SemanticUnit(0, "Alpha beta gamma delta.")],
        )
        self.examples_path = self.root / "examples.jsonl"
        write_jsonl(self.examples_path, [self.example])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_packet_status_requires_valid_complete_cache(self) -> None:
        packet_dir = self.root / "packets"
        packet_dir.mkdir()
        write_jsonl(
            packet_dir / "example.jsonl",
            [SemanticPacket("example", 0, self.example.units[0].text, "Alpha", "beta", 4, 1, 1)],
        )
        self.assertTrue(packet_status(self.examples_path, packet_dir)["complete"])

    def test_exact_status_requires_all_states(self) -> None:
        exact_dir = self.root / "exact"
        exact_dir.mkdir()
        rows = [
            ExactSearchResult("example", (state,), state, "answer", 1.0, 1.0, 1.0, 1.0)
            for state in range(3)
        ]
        write_jsonl(exact_dir / "example.jsonl", rows)
        self.assertTrue(exact_status(self.examples_path, exact_dir)["complete"])
        write_jsonl(exact_dir / "example.jsonl", rows[:2])
        self.assertFalse(exact_status(self.examples_path, exact_dir)["complete"])

    def test_exact_rows_reject_wrong_state_shape(self) -> None:
        rows = [
            ExactSearchResult("example", (state,), state, "answer", 1.0, 1.0, 1.0, 1.0)
            for state in (0, 1)
        ]
        rows.append(ExactSearchResult("example", (0, 0), 2, "answer", 1.0, 1.0, 1.0, 1.0))
        self.assertTrue(validate_exact_rows(self.example, rows))


if __name__ == "__main__":
    unittest.main()
