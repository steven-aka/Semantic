from __future__ import annotations

import unittest

from src.target.answer_parser import parse_answer


class AnswerParserTests(unittest.TestCase):
    def test_removes_answer_wrapper(self) -> None:
        self.assertEqual(parse_answer("Final answer: Paris"), "Paris")

    def test_extracts_boolean_from_explanatory_sentence(self) -> None:
        self.assertEqual(parse_answer("Yes, both people were American."), "yes")
        self.assertEqual(parse_answer("No. The dates do not overlap."), "no")

    def test_extracts_tagged_answer(self) -> None:
        self.assertEqual(
            parse_answer("The evidence supports this. <answer>Chief of Protocol</answer>"),
            "Chief of Protocol",
        )

    def test_does_not_truncate_title_starting_with_no(self) -> None:
        self.assertEqual(parse_answer("No Country for Old Men"), "No Country for Old Men")


if __name__ == "__main__":
    unittest.main()
