from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.data.schemas import QAExample, read_jsonl
from src.representation.token_counter import count_tokens, load_tokenizer


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_examples(examples: list[QAExample], tokenizer: Any, max_tokens: int = 256) -> dict[str, Any]:
    token_counts = [count_tokens(tokenizer, unit.text) for example in examples for unit in example.units]
    units_per_example = [len(example.units) for example in examples]
    supports_per_example = [sum(unit.supporting for unit in example.units) for example in examples]
    contiguous_ids = all(
        [unit.unit_id for unit in example.units] == list(range(len(example.units)))
        for example in examples
    )
    return {
        "examples": len(examples),
        "semantic_units": len(token_counts),
        "unit_ids_contiguous": contiguous_ids,
        "empty_units": sum(not unit.text.strip() for example in examples for unit in example.units),
        "units_over_max_tokens": sum(value > max_tokens for value in token_counts),
        "tokens_per_unit": {
            "min": min(token_counts) if token_counts else None,
            "mean": mean(token_counts) if token_counts else None,
            "median": median(token_counts) if token_counts else None,
            "p95": _percentile(token_counts, 0.95),
            "max": max(token_counts) if token_counts else None,
        },
        "units_per_example": dict(sorted(Counter(units_per_example).items())),
        "supporting_units_per_example": dict(sorted(Counter(supports_per_example).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized/segmented experiment data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    examples = list(read_jsonl(args.input, QAExample))
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    audit = audit_examples(examples, tokenizer, args.max_tokens)
    audit.update(
        {
            "input": args.input,
            "input_sha256": sha256(args.input),
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.revision,
            "max_tokens": args.max_tokens,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

