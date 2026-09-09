from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.schemas import QAExample, SemanticUnit, read_jsonl, write_jsonl
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, write_metadata


POLICY_NAME = "sentence_atomic_label_free_bm25_v1"
DEFAULT_SAMPLE_SALT = "v0_3_label_free_sentence_atomic_20260907"
_TERM_RE = re.compile(r"\w+", re.UNICODE)


def terms(text: str) -> list[str]:
    return [term.casefold() for term in _TERM_RE.findall(text) if len(term) > 1]


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Deterministic per-example BM25; no corpus labels or learned weights."""
    if not documents:
        return []
    tokenized = [terms(document) for document in documents]
    query_terms = terms(query)
    n_docs = len(documents)
    average_length = sum(map(len, tokenized)) / n_docs or 1.0
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens)
    )
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_norm = k1 * (1.0 - b + b * len(tokens) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (n_docs - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + length_norm)
            )
        scores.append(score)
    return scores


def sentence_candidates(row: Mapping[str, Any]) -> list[tuple[int, str, int, str]]:
    """Return source-order coordinates without reading supporting_facts."""
    context = row["context"]
    output: list[tuple[int, str, int, str]] = []
    for title_index, (title, sentences) in enumerate(
        zip(context["title"], context["sentences"])
    ):
        for sentence_id, sentence in enumerate(sentences):
            sentence = str(sentence).strip()
            if sentence:
                output.append((title_index, str(title), sentence_id, sentence))
    return output


def build_label_free_example(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    n_units: int = 6,
    max_tokens: int = 256,
    k1: float = 1.2,
    b: float = 0.75,
) -> QAExample | None:
    candidates = sentence_candidates(row)
    if len(candidates) < n_units:
        return None
    rendered = [f"Title: {title}\n{sentence}" for _, title, _, sentence in candidates]
    # This bounded pilot excludes over-limit source sentences using text length
    # only.  It never uses answers or supporting-fact annotations to do so.
    if any(count_tokens(tokenizer, text) > max_tokens for text in rendered):
        return None
    scores = bm25_scores(str(row["question"]), rendered, k1=k1, b=b)
    selected = sorted(
        sorted(range(len(candidates)), key=lambda index: (-scores[index], index))[:n_units]
    )
    units = []
    for unit_id, candidate_index in enumerate(selected):
        _, title, sentence_id, _ = candidates[candidate_index]
        units.append(
            SemanticUnit(
                unit_id=unit_id,
                text=rendered[candidate_index],
                # Deliberately false: construction never reads gold labels.
                supporting=False,
                title=title,
                source_sentence_id=sentence_id,
            )
        )
    return QAExample(
        example_id=str(row["id"]),
        dataset="hotpotqa",
        question=str(row["question"]).strip(),
        answer=str(row["answer"]).strip(),
        context="\n\n".join(unit.text for unit in units),
        units=units,
    )


def unit_selection_signature(example: QAExample) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (unit.unit_id, unit.title, unit.source_sentence_id, unit.text)
        for unit in example.units
    )


def _read_excluded_ids(paths: Iterable[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            excluded.update(
                example.example_id for example in read_jsonl(candidate, QAExample)
            )
    return excluded


def _rank(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{example_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen sentence-atomic, label-free BM25 V0.3 pilot"
    )
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--output", default="data/units/hotpot_v0_3_candidates.jsonl")
    parser.add_argument("--manifest", default="results/v0_3/data_manifest.json")
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--units", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--sample-salt", default=DEFAULT_SAMPLE_SALT)
    parser.add_argument(
        "--exclude",
        action="append",
        default=["data/units/hotpot_validation.jsonl"],
    )
    args = parser.parse_args()

    import pyarrow.parquet as parquet

    tokenizer = load_tokenizer(args.tokenizer)
    excluded = _read_excluded_ids(args.exclude)
    ranked_rows: list[tuple[str, Mapping[str, Any]]] = []
    columns = ["id", "question", "answer", "context"]
    for batch in parquet.ParquetFile(args.raw).iter_batches(
        batch_size=256, columns=columns
    ):
        for row in batch.to_pylist():
            if row["id"] in excluded:
                continue
            ranked_rows.append((_rank(str(row["id"]), args.sample_salt), row))
    ranked_rows.sort(key=lambda item: (item[0], str(item[1]["id"])))
    selected: list[tuple[str, QAExample, Mapping[str, Any]]] = []
    for rank, row in ranked_rows:
        example = build_label_free_example(
            row,
            tokenizer,
            n_units=args.units,
            max_tokens=args.max_tokens,
            k1=args.k1,
            b=args.b,
        )
        if example is not None:
            selected.append((rank, example, row))
            if len(selected) == args.count:
                break
    if len(selected) < args.count:
        raise RuntimeError(f"only {len(selected)} label-free candidates for {args.count}")

    # Leakage test: perturb answer and inject arbitrary supporting_facts.  Unit
    # selection must remain byte-for-byte identical because neither field is an
    # input to build_label_free_example.
    for _, example, row in selected:
        perturbed = dict(row)
        perturbed["answer"] = "LABEL_ERASED"
        perturbed["supporting_facts"] = {
            "title": ["SHOULD_NOT_BE_READ"], "sent_id": [999999]
        }
        rebuilt = build_label_free_example(
            perturbed,
            tokenizer,
            n_units=args.units,
            max_tokens=args.max_tokens,
            k1=args.k1,
            b=args.b,
        )
        if rebuilt is None or unit_selection_signature(rebuilt) != unit_selection_signature(example):
            raise AssertionError(f"label leakage detected for {example.example_id}")

    output_examples = [item[1] for item in selected]
    write_jsonl(args.output, output_examples)
    token_lengths = [
        count_tokens(tokenizer, unit.text)
        for example in output_examples
        for unit in example.units
    ]
    write_metadata(
        args.manifest,
        {
            "policy": POLICY_NAME,
            "preregistered_before_gpu_results": True,
            "sample_salt": args.sample_salt,
            "sample_rule": "lowest_sha256(salt + ':' + id) among valid unused rows",
            "excluded_ids": len(excluded),
            "n_examples": len(output_examples),
            "n_units_per_example": args.units,
            "segmentation": "one original HotpotQA sentence per unit; title is visible and token-counted",
            "selection": "per-example BM25(question, title + sentence), top-k, source-order render",
            "bm25": {"k1": args.k1, "b": args.b},
            "max_tokens": args.max_tokens,
            "label_permutation_invariance": True,
            "example_ids": [example.example_id for example in output_examples],
            "unit_token_min": min(token_lengths),
            "unit_token_max": max(token_lengths),
            "source_raw": str(args.raw),
            "source_raw_sha256": sha256(args.raw),
            "output": str(args.output),
            "output_sha256": sha256(args.output),
        },
    )
    print(json.dumps({"examples": len(output_examples), "units": len(token_lengths), "min_tokens": min(token_lengths), "max_tokens": max(token_lengths)}, indent=2))


if __name__ == "__main__":
    main()
