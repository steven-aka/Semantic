from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from src.data.schemas import QAExample, SemanticUnit, read_jsonl, write_jsonl
from src.representation.token_counter import WhitespaceTokenizer, count_tokens, encode_text, load_tokenizer


_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])(?:[\"'”’)]*)\s+|(?<=[。！？])")


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in _SENTENCE_RE.split(text.strip()) if piece.strip()]


def _hard_split(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    ids = encode_text(tokenizer, text)
    pieces: list[str] = []
    for start in range(0, len(ids), max_tokens):
        chunk = ids[start : start + max_tokens]
        if hasattr(tokenizer, "decode"):
            decoded = tokenizer.decode(chunk, skip_special_tokens=True).strip()
        else:
            decoded = " ".join(str(token) for token in chunk)
        if decoded:
            pieces.append(decoded)
    return pieces


def segment_document(
    text: str,
    tokenizer: Any,
    target_tokens: int = 192,
    max_tokens: int = 256,
) -> list[str]:
    """Create ordered, non-overlapping chunks near target_tokens."""
    if not 0 < target_tokens <= max_tokens:
        raise ValueError("require 0 < target_tokens <= max_tokens")
    paragraphs = [part.strip() for part in _PARAGRAPH_RE.split(text) if part.strip()]
    output: list[str] = []
    for paragraph in paragraphs:
        current: list[str] = []
        current_size = 0
        for sentence in split_sentences(paragraph) or [paragraph]:
            sentence_size = count_tokens(tokenizer, sentence)
            if sentence_size > max_tokens:
                if current:
                    output.append(" ".join(current))
                    current, current_size = [], 0
                output.extend(_hard_split(sentence, tokenizer, max_tokens))
                continue
            candidate_size = current_size + sentence_size
            should_merge = (
                not current
                or candidate_size <= target_tokens
                or (
                    candidate_size <= max_tokens
                    and abs(candidate_size - target_tokens) < abs(current_size - target_tokens)
                )
            )
            if should_merge:
                current.append(sentence)
                current_size = candidate_size
            else:
                output.append(" ".join(current))
                current, current_size = [sentence], sentence_size
        if current:
            output.append(" ".join(current))
    return output


def segment_example(
    example: QAExample, tokenizer: Any, target_tokens: int = 192, max_tokens: int = 256
) -> QAExample:
    units: list[SemanticUnit] = []
    sources = example.units or [SemanticUnit(0, example.context)]
    for source in sources:
        for chunk in segment_document(source.text, tokenizer, target_tokens, max_tokens):
            units.append(
                SemanticUnit(
                    unit_id=len(units), text=chunk, supporting=source.supporting, title=source.title
                )
            )
    return replace(example, units=units, context="\n\n".join(unit.text for unit in units))


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment normalized examples into semantic units")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-tokens", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--test-tokenizer", action="store_true")
    args = parser.parse_args()
    tokenizer = WhitespaceTokenizer() if args.test_tokenizer else load_tokenizer(args.tokenizer)
    examples = (
        segment_example(example, tokenizer, args.target_tokens, args.max_tokens)
        for example in read_jsonl(args.input, QAExample)
    )
    write_jsonl(args.output, examples)


if __name__ == "__main__":
    main()

