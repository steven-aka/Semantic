from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.normalize_hotpot import _terms
from src.data.schemas import QAExample, SemanticUnit, read_jsonl, write_jsonl
from src.evaluation.fact_coverage import raw_supporting_facts


def _read_raw(path: str | Path, wanted: set[str]) -> dict[str, Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    selected: dict[str, Mapping[str, Any]] = {}
    columns = ["id", "context", "supporting_facts"]
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            if row["id"] in wanted:
                selected[row["id"]] = row
        if len(selected) == len(wanted):
            break
    return selected


def build_fact_atomic_example(
    example: QAExample,
    raw_row: Mapping[str, Any],
    *,
    n_units: int = 6,
    min_facts: int = 3,
    max_facts: int = 5,
) -> QAExample | None:
    """Build a fixed-size oracle V0 stress case with one gold sentence per unit.

    Only V0 unit granularity changes. Questions, answers, gold facts and the
    three representation states remain unchanged. Distractors come solely from
    non-supporting HotpotQA titles.
    """
    facts = raw_supporting_facts(raw_row)
    if not min_facts <= len(facts) <= max_facts or len(facts) > n_units:
        return None

    support_titles = {str(fact["title"]) for fact in facts}
    title_order = {
        str(title): index for index, title in enumerate(raw_row["context"]["title"])
    }
    query_terms = _terms(example.question)
    distractors = [unit for unit in example.units if unit.title not in support_titles]
    distractors.sort(
        key=lambda unit: (
            -len(query_terms & _terms(f"{unit.title or ''} {unit.text}")),
            unit.unit_id,
        )
    )
    needed = n_units - len(facts)
    if len(distractors) < needed:
        return None

    ordered: list[tuple[tuple[int, int, int], str, bool, str | None]] = []
    for fact in facts:
        title = str(fact["title"])
        ordered.append(
            (
                (title_order[title], int(fact["sentence_id"]), 0),
                str(fact["text"]),
                True,
                title,
            )
        )
    for unit in distractors[:needed]:
        ordered.append(
            (
                (title_order.get(unit.title or "", len(title_order)), unit.unit_id, 1),
                unit.text,
                False,
                unit.title,
            )
        )
    ordered.sort(key=lambda item: item[0])
    units = [
        SemanticUnit(index, text, supporting, title)
        for index, (_, text, supporting, title) in enumerate(ordered)
    ]
    return replace(example, units=units, context="\n\n".join(unit.text for unit in units))


def build_fact_atomic_pilot(
    examples: Iterable[QAExample],
    raw_rows: Mapping[str, Mapping[str, Any]],
    *,
    n_units: int = 6,
    min_facts: int = 3,
    max_facts: int = 5,
) -> list[QAExample]:
    output = []
    for example in examples:
        row = raw_rows.get(example.example_id)
        if row is None:
            raise ValueError(f"raw HotpotQA row missing for {example.example_id}")
        atomic = build_fact_atomic_example(
            example,
            row,
            n_units=n_units,
            min_facts=min_facts,
            max_facts=max_facts,
        )
        if atomic is not None:
            output.append(atomic)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fact-atomic, fixed-unit HotpotQA V0 stress pilot"
    )
    parser.add_argument("--examples", default="data/units/hotpot_validation.jsonl")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--output", default="data/units/hotpot_fact_atomic_candidates.jsonl")
    parser.add_argument("--units", type=int, default=6)
    parser.add_argument("--min-facts", type=int, default=3)
    parser.add_argument("--max-facts", type=int, default=5)
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples, QAExample))
    raw_rows = _read_raw(args.raw, {example.example_id for example in examples})
    output = build_fact_atomic_pilot(
        examples,
        raw_rows,
        n_units=args.units,
        min_facts=args.min_facts,
        max_facts=args.max_facts,
    )
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()
