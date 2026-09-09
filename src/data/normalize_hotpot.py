from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from src.data.schemas import QAExample, SemanticUnit, write_jsonl


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _contexts(row: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    context = row.get("context", [])
    if isinstance(context, dict):
        return list(zip(context.get("title", []), context.get("sentences", [])))
    result = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((str(item[0]), list(item[1])))
        elif isinstance(item, dict):
            result.append((str(item["title"]), list(item.get("sentences", []))))
    return result


def normalize_row(row: Mapping[str, Any], prefix: str = "hotpot") -> QAExample:
    support = row.get("supporting_facts", [])
    if isinstance(support, dict):
        support_pairs = set(zip(support.get("title", []), support.get("sent_id", [])))
    else:
        support_pairs = {(item[0], int(item[1])) for item in support}
    units: list[SemanticUnit] = []
    for title, sentences in _contexts(row):
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        if not text:
            continue
        is_supporting = any((title, index) in support_pairs for index in range(len(sentences)))
        units.append(SemanticUnit(len(units), text, is_supporting, title))
    example_id = str(row.get("_id") or row.get("id") or f"{prefix}_{abs(hash(str(row))) :x}")
    return QAExample(
        example_id=example_id,
        dataset="hotpotqa",
        question=str(row["question"]).strip(),
        answer=str(row["answer"]).strip(),
        context="\n\n".join(unit.text for unit in units),
        units=units,
    )


def _terms(text: str) -> set[str]:
    return {term.lower() for term in _WORD_RE.findall(text) if len(term) > 1}


def select_controlled_units(example: QAExample, n_units: int) -> QAExample:
    """Keep all supports, then lexical query-relevant distractors; preserve source order."""
    supporting = [unit for unit in example.units if unit.supporting]
    if len(supporting) > n_units:
        raise ValueError(f"{example.example_id} has more supporting units than n_units")
    query_terms = _terms(example.question)
    distractors = [unit for unit in example.units if not unit.supporting]
    ranked = sorted(
        distractors,
        key=lambda unit: (-len(query_terms & _terms(f"{unit.title or ''} {unit.text}")), unit.unit_id),
    )
    selected_ids = {unit.unit_id for unit in supporting + ranked[: n_units - len(supporting)]}
    chosen = [unit for unit in example.units if unit.unit_id in selected_ids]
    units = [SemanticUnit(i, unit.text, unit.supporting, unit.title) for i, unit in enumerate(chosen)]
    return QAExample(
        example_id=example.example_id,
        dataset=example.dataset,
        question=example.question,
        answer=example.answer,
        context="\n\n".join(unit.text for unit in units),
        units=units,
    )


def _read_local(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read Parquet input") from exc
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        else:
            payload = json.load(handle)
            yield from payload if isinstance(payload, list) else payload["data"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize HotpotQA into the unified schema")
    parser.add_argument("--input", help="Local HotpotQA JSON or JSONL; otherwise use Hugging Face")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--dataset-id",
        default="hotpotqa/hotpot_qa",
        help="Hugging Face dataset repository used when --input is omitted",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--controlled-units", type=int)
    args = parser.parse_args()
    if args.input:
        rows: Iterable[Mapping[str, Any]] = _read_local(Path(args.input))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("install datasets or pass --input") from exc
        rows = load_dataset(args.dataset_id, "distractor", split=args.split)
    examples = []
    for index, row in enumerate(rows):
        if args.limit is not None and index >= args.limit:
            break
        example = normalize_row(row)
        if args.controlled_units:
            example = select_controlled_units(example, args.controlled_units)
        examples.append(example)
    write_jsonl(args.output, examples)


if __name__ == "__main__":
    main()
