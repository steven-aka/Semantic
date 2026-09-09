from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import experiment_metadata, write_metadata


def content_fact_recall(state: Sequence[int], coverage: Mapping[str, Any]) -> float:
    facts = coverage["facts"]
    if not facts:
        return 1.0
    retained = 0
    for fact in facts:
        if fact.get("unit_id") is None:
            continue
        level = state[int(fact["unit_id"])]
        if level == 1:
            retained += int(fact["gist_supported"])
        elif level == 2:
            retained += int(fact["full_supported"])
    return retained / len(facts)


def rescore_results(
    results: Sequence[ExactSearchResult], coverage: Mapping[str, Any]
) -> list[ExactSearchResult]:
    rescored = []
    for result in results:
        fact_recall = content_fact_recall(result.state, coverage)
        rescored.append(
            replace(
                result,
                fact_recall=fact_recall,
                fidelity=min(result.answer_f1, fact_recall),
            )
        )
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore exact search with content-aware facts")
    parser.add_argument("--input-dir", default="results/exact_search")
    parser.add_argument("--coverage", default="results/v0_1/fact_coverage.jsonl")
    parser.add_argument("--output-dir", default="results/v0_1/exact_search")
    parser.add_argument("--experiment-id", default="v0_1_fact_aware_rescore")
    args = parser.parse_args()

    coverage = {
        row["example_id"]: row for row in read_jsonl(args.coverage)
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(Path(args.input_dir).glob("*.jsonl"))
    for path in input_paths:
        results = list(read_jsonl(path, ExactSearchResult))
        if not results:
            continue
        example_id = results[0].example_id
        if example_id not in coverage:
            raise ValueError(f"fact coverage missing for {example_id}")
        write_jsonl(output_dir / path.name, rescore_results(results, coverage[example_id]))
    write_metadata(
        output_dir / "metadata.json",
        experiment_metadata(
            experiment_id=args.experiment_id,
            source_exact_search=args.input_dir,
            source_fact_coverage=args.coverage,
            fidelity="min(answer_f1, target_judged_gold_fact_recall)",
            target_predictions_reused=True,
        ),
    )


if __name__ == "__main__":
    main()
