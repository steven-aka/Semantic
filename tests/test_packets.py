from __future__ import annotations

import unittest

from src.data.schemas import QAExample, SemanticPacket, SemanticUnit
from src.representation.token_counter import WhitespaceTokenizer, count_tokens
from src.teacher.packet_generator import (
    PacketGenerator,
    cached_packets_valid,
    collapse_redundant_residual,
    parse_packet_json,
    rebalance_gist,
)
from src.teacher.packet_validator import numeric_concepts, numeric_strings, validate_packet


class PacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def packet(self, gist: str, residual: str) -> SemanticPacket:
        source = "In 1902 Ada Lovelace opened a public school beside the old river in London city."
        return SemanticPacket(
            "x",
            0,
            source,
            gist,
            residual,
            count_tokens(self.tokenizer, source),
            count_tokens(self.tokenizer, gist),
            count_tokens(self.tokenizer, residual),
        )

    def test_packet_parser_accepts_fence(self) -> None:
        parsed = parse_packet_json('```json\n{"gist":"g","residual":"r"}\n```')
        self.assertEqual(parsed, {"gist": "g", "residual": "r"})

    def test_rebalance_moves_suffix_without_dropping_words(self) -> None:
        gist, residual = rebalance_gist(
            "alpha beta gamma delta epsilon", "zeta eta", self.tokenizer, 3
        )
        self.assertEqual(gist, "alpha beta gamma")
        self.assertEqual(residual, "delta epsilon zeta eta")

    def test_only_information_free_repeated_residual_can_collapse(self) -> None:
        gist = "Swedish higher education part of Bologna Process"
        repeated = "The Swedish higher education system is a part of the Bologna Process."
        informative = repeated + " It began in 1999."
        self.assertEqual(collapse_redundant_residual(gist, repeated), "")
        self.assertIsNone(collapse_redundant_residual(gist, informative))

    def test_numeric_validation_compares_values_not_surface_forms(self) -> None:
        source = "It was the ninth storm, third typhoon, and second super-typhoon."
        packet = SemanticPacket("x", 0, source, "9th storm, 3rd typhoon", "2nd super-typhoon", 20, 4, 2)
        self.assertTrue(validate_packet(packet).valid)
        invented = SemanticPacket("x", 0, source, "10th storm", "", 20, 2, 0)
        self.assertFalse(validate_packet(invented).valid)
        self.assertEqual(numeric_concepts("one hour, twenty-one minutes, sixteen seconds"), {"1", "21", "16"})
        self.assertEqual(numeric_concepts("1h21m16s"), {"1", "21", "16"})
        self.assertEqual(numeric_concepts("1939–1944 and 1939-1944"), {"1939", "1944"})

    def test_cache_reuse_requires_current_source_and_valid_packet(self) -> None:
        example = QAExample(
            "x",
            "test",
            "q",
            "a",
            "Alpha beta gamma delta.",
            [SemanticUnit(0, "Alpha beta gamma delta.")],
        )
        valid = SemanticPacket(
            "x", 0, "Alpha beta gamma delta.", "Alpha", "beta", 4, 1, 1
        )
        stale = SemanticPacket("x", 0, "Old source.", "Old", "source", 2, 1, 1)
        self.assertTrue(cached_packets_valid(example, [valid]))
        self.assertFalse(cached_packets_valid(example, [stale]))

    def test_validator_rejects_hallucinated_number(self) -> None:
        result = validate_packet(self.packet("Ada opened school in 1999.", "London site."))
        self.assertFalse(result.valid)
        self.assertTrue(any("numbers" in error for error in result.errors))

    def test_validator_requires_source_document_identity_closure(self) -> None:
        source = (
            "Document title: Edward Samuel Rogers\n"
            "Rogers attended Upper Canada College."
        )
        closed = SemanticPacket(
            "x", 0, source, "Edward Samuel Rogers attended college.", "", 20, 5, 0
        )
        shortened = SemanticPacket("x", 0, source, "Rogers attended college.", "", 20, 3, 0)
        self.assertTrue(validate_packet(closed).valid)
        result = validate_packet(shortened)
        self.assertFalse(result.valid)
        self.assertTrue(any("document titles" in error for error in result.errors))
        marked_source = (
            "Document title: [[Harry Potter (film)|Harry Potter]]\n"
            "The film was released in 2001."
        )
        display_title = SemanticPacket(
            "x", 0, marked_source, "Harry Potter was released in 2001.", "", 20, 7, 0
        )
        self.assertTrue(validate_packet(display_title).valid)

    def test_validator_ignores_sentence_punctuation_after_number(self) -> None:
        result = validate_packet(self.packet("Ada opened a school in 1902.", "London site."))
        self.assertTrue(result.valid)
        self.assertEqual(numeric_strings("In 1902, then 19.5%."), ("1902", "19.5%"))

    def test_generator_retries_invalid_json(self) -> None:
        calls = 0
        seen_prompts = []

        def generate(prompts):
            nonlocal calls
            calls += 1
            seen_prompts.extend(prompts)
            if calls == 1:
                return ["not json"]
            return ['{"gist":"Ada opened a school.","residual":"It was in London."}']

        source = "Ada opened a well known public school for local children beside the river in London."
        example = QAExample("x", "test", "q", "a", source, [SemanticUnit(0, source)])
        packets = PacketGenerator(generate, self.tokenizer, max_retries=1).generate_for_units(example)
        self.assertEqual(calls, 2)
        self.assertEqual(len(packets), 1)
        self.assertIn("previous output below was rejected", seen_prompts[1])
        self.assertIn("not json", seen_prompts[1])
        self.assertIn("at most 3 whitespace-separated words", seen_prompts[0])
        self.assertIn("allowed numeric values from this source: NONE", seen_prompts[0])

    def test_generator_retry_uses_each_units_word_budget(self) -> None:
        short = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
        long = (
            "Able baker charlie delta echo foxtrot golf hotel india juliet kilo lima "
            "mike november oscar papa quebec romeo sierra tango."
        )
        calls = 0
        batches = []

        def generate(prompts):
            nonlocal calls
            calls += 1
            batches.append(list(prompts))
            if calls == 1:
                return [
                    f'{{"gist":{short!r},"residual":""}}'.replace("'", '"'),
                    f'{{"gist":{long!r},"residual":""}}'.replace("'", '"'),
                ]
            return [
                '{"gist":"Alpha beta.","residual":""}',
                '{"gist":"Able baker charlie.","residual":""}',
            ]

        example = QAExample(
            "x",
            "test",
            "q",
            "a",
            short + " " + long,
            [SemanticUnit(0, short), SemanticUnit(1, long)],
        )
        packets = PacketGenerator(generate, self.tokenizer, max_retries=1).generate_for_units(example)
        self.assertEqual(len(packets), 2)
        self.assertIn("rewrite GIST in at most 2 words", batches[1][0])
        self.assertIn("rewrite GIST in at most 5 words", batches[1][1])


if __name__ == "__main__":
    unittest.main()
