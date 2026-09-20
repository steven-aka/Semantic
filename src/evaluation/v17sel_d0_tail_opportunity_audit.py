"""Train-only, read-only audit of tail opportunity under a preserved top-4 pool.

All pool metrics are oracle ceilings, not deployable top-1 results.  In
particular, a per-query oracle choosing one tail equals the full top-10 pool.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


def best(candidates: list[dict]) -> dict | None:
    successes = [c for c in candidates if c["success"][3]]
    return min(successes, key=lambda c: (c["earliest_tokens"][3], c["rank"])) if successes else None


def outcome(row: dict, tail_ranks: tuple[int, ...]) -> dict:
    candidates = row["candidates"]
    pool = candidates[:5] + [candidates[rank] for rank in tail_ranks]
    winner = best(pool)
    return {
        "success": winner is not None,
        "complete": any(c["complete"] for c in pool),
        "earliest_090_tokens": winner["earliest_tokens"][3] if winner else None,
        "winner_rank": winner["rank"] if winner else None,
    }


def summarize(rows: list[dict], ranks: tuple[int, ...]) -> dict:
    outcomes = [outcome(row, ranks) for row in rows]
    baseline = [outcome(row, ()) for row in rows]
    repairs = sum(not old["success"] and new["success"] for old, new in zip(baseline, outcomes))
    savings = [old["earliest_090_tokens"] - new["earliest_090_tokens"]
               for old, new in zip(baseline, outcomes) if old["success"] and new["success"]]
    return {
        "tail_ranks": list(ranks),
        "queries": len(rows),
        "oracle_success_090": sum(item["success"] for item in outcomes),
        "oracle_complete": sum(item["complete"] for item in outcomes),
        "new_success_vs_top4": repairs,
        "top4_success_with_token_saving": sum(value > 0 for value in savings),
        "mean_token_saving_on_top4_success": mean(savings) if savings else None,
        "mean_failure_penalized_090_token_fraction": mean(
            item["earliest_090_tokens"] / row["full_tokens"] if item["success"] else 1.0
            for row, item in zip(rows, outcomes)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.candidates))
    if len(rows) != 2032 or len({row["example_id"] for row in rows}) != 2032:
        raise ValueError("expected 2032 unique train-side queries")
    for row in rows:
        if len(row["candidates"]) != 11 or [c["rank"] for c in row["candidates"]] != list(range(11)):
            raise ValueError("unexpected candidate rank contract")
        if row["full_tokens"] <= 0:
            raise ValueError("invalid full token count")
    train = [row for row in rows if fold(row["example_id"]) != 4]
    validation = [row for row in rows if fold(row["example_id"]) == 4]
    if len(train) != 1421 or len(validation) != 611:
        raise ValueError("unexpected C1 train/validation split")

    # The fixed-rank operating points are defined without using any labels.
    fixed = {str(rank): {"all": summarize(rows, (rank,)),
                         "train": summarize(train, (rank,)),
                         "validation": summarize(validation, (rank,))}
             for rank in range(5, 11)}
    baseline = {"all": summarize(rows, ()), "train": summarize(train, ()),
                "validation": summarize(validation, ())}
    full = {"all": summarize(rows, tuple(range(5, 11))),
            "train": summarize(train, tuple(range(5, 11))),
            "validation": summarize(validation, tuple(range(5, 11)))}

    # A label-optimized global subset is explicitly an upper bound for a
    # precommitted set of ranks, not a model or an unbiased validation result.
    two_rank_oracles = [summarize(train, pair) for pair in combinations(range(5, 11), 2)]
    best_two_train = max(two_rank_oracles, key=lambda x: (
        x["new_success_vs_top4"], x["top4_success_with_token_saving"],
        -x["mean_failure_penalized_090_token_fraction"], -x["tail_ranks"][0], -x["tail_ranks"][1]))
    two_ranks = tuple(best_two_train["tail_ranks"])
    two_fixed = {"ranks_selected_on_train": list(two_ranks),
                 "train": best_two_train,
                 "validation": summarize(validation, two_ranks)}

    diagnostic = []
    first_rank = Counter()
    positive_count = Counter()
    for row in rows:
        top4 = outcome(row, ())
        top10 = outcome(row, tuple(range(5, 11)))
        positive = [c["rank"] for c in row["candidates"][5:] if c["success"][3]]
        if not top4["success"] and top10["success"]:
            first_rank[min(positive)] += 1
            positive_count[len(positive)] += 1
        diagnostic.append({
            "example_id": row["example_id"], "fold": fold(row["example_id"]),
            "top4_success": top4["success"], "top10_success": top10["success"],
            "top4_earliest_090_tokens": top4["earliest_090_tokens"],
            "top10_earliest_090_tokens": top10["earliest_090_tokens"],
            "successful_tail_ranks": positive,
        })
    if full["all"]["new_success_vs_top4"] != sum(first_rank.values()):
        raise ValueError("new opportunity accounting mismatch")
    # Per-query adaptive oracle one-slot is a tautological copy of full top10.
    summary = {
        "protocol": "V17-SEL-D0_TRAIN_ONLY_TAIL_OPPORTUNITY_AUDIT",
        "role": "oracle pool opportunity, not deployed top-1 or a tail-gate result",
        "candidate_budget": "top4 preserved as a pool; adding tail slots increases pool size; final output still requires a selector",
        "baseline_top4": baseline,
        "fixed_single_tail_rank": fixed,
        "train_selected_fixed_two_tail_ranks": two_fixed,
        "full_top10": full,
        "per_query_oracle_one_tail": {
            "oracle_success_090": full["all"]["oracle_success_090"],
            "reason": "when a tail success exists, an adaptive oracle can choose it; this equals the full top10 existential ceiling by definition",
        },
        "new_opportunity_first_success_rank": dict(sorted(first_rank.items())),
        "new_opportunity_successful_tail_count": dict(sorted(positive_count.items())),
        "decision": "DO_NOT_INFER_ONE_ESCAPE_SLOT_DEPLOYABILITY_FROM_ORACLE_COVERAGE",
        "holdout_read": False, "development_read": False, "confirmation_read": False,
        "artifacts": {"candidates_sha256": sha256(args.candidates)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", diagnostic)
    write_metadata(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
