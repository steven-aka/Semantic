"""Train-only oracle value audit for the frozen top-4/top-10 candidate sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, (9 * len(ordered) + 9) // 10 - 1)]


def summarize(rows: list[dict]) -> dict:
    output = {"queries": len(rows)}
    for k in (4, 10):
        key = f"top{k}"
        successes = [row for row in rows if row[key]["success"]]
        complete = [row for row in rows if row[key]["complete"]]
        costs = [float(row[key]["earliest_090_tokens"]) for row in successes]
        fractions = [float(row[key]["earliest_090_token_fraction"]) for row in successes]
        regrets = [float(row[key]["global_090_token_regret_fraction"]) for row in successes if row[key]["global_090_token_regret_fraction"] is not None]
        output[key] = {
            "success_090": len(successes),
            "complete": len(complete),
            "mean_earliest_090_tokens_successful": mean(costs) if costs else None,
            "median_earliest_090_tokens_successful": median(costs) if costs else None,
            "p90_earliest_090_tokens_successful": p90(costs),
            "mean_earliest_090_token_fraction_successful": mean(fractions) if fractions else None,
            "mean_global_090_token_regret_fraction_successful": mean(regrets) if regrets else None,
            "mean_penalized_090_token_fraction_all_queries": mean(row[key]["earliest_090_token_fraction"] if row[key]["success"] else 1.0 for row in rows),
        }
    new = [row for row in rows if row["top10"]["success"] and not row["top4"]["success"]]
    output["new_top10_success"] = {
        "queries": len(new),
        "mean_earliest_090_tokens": mean(row["top10"]["earliest_090_tokens"] for row in new) if new else None,
        "median_earliest_090_tokens": median(row["top10"]["earliest_090_tokens"] for row in new) if new else None,
        "p90_earliest_090_tokens": p90([row["top10"]["earliest_090_tokens"] for row in new]),
        "mean_earliest_090_token_fraction": mean(row["top10"]["earliest_090_token_fraction"] for row in new) if new else None,
        "mean_global_090_token_regret_fraction": mean(row["top10"]["global_090_token_regret_fraction"] for row in new if row["top10"]["global_090_token_regret_fraction"] is not None) if new else None,
    }
    paired = [row for row in rows if row["top4"]["success"] and row["top10"]["success"]]
    output["paired_top4_success"] = {
        "queries": len(paired),
        "top10_cheaper": sum(row["top10"]["earliest_090_tokens"] < row["top4"]["earliest_090_tokens"] for row in paired),
        "top10_equal": sum(row["top10"]["earliest_090_tokens"] == row["top4"]["earliest_090_tokens"] for row in paired),
        "mean_token_delta_top10_minus_top4": mean(row["top10"]["earliest_090_tokens"] - row["top4"]["earliest_090_tokens"] for row in paired) if paired else None,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_BUILD_TRAIN_ONLY":
        raise ValueError("C0 protocol is not frozen")
    rows = []
    ids = set()
    for item in read_jsonl(args.candidates):
        query = item["example_id"]
        if query in ids:
            raise ValueError("duplicate query")
        ids.add(query)
        if len(item["candidates"]) != 11:
            raise ValueError("expected fallback + top10")
        result = {"example_id": query, "fold": fold(query), "unique_candidate_orders": item["unique_candidate_orders"]}
        for k in (4, 10):
            pool = item["candidates"][:k + 1]
            successful = [candidate for candidate in pool if candidate["success"][3]]
            best = min(successful, key=lambda candidate: (candidate["earliest_tokens"][3], candidate["rank"])) if successful else None
            complete = [candidate for candidate in pool if candidate["complete"]]
            result[f"top{k}"] = {
                "success": bool(successful),
                "complete": bool(complete),
                "earliest_090_tokens": best["earliest_tokens"][3] if best else None,
                "earliest_090_token_fraction": best["earliest_tokens"][3] / item["full_tokens"] if best else None,
                "global_090_token_regret_fraction": (best["earliest_tokens"][3] - item["global_min_090_tokens"]) / item["full_tokens"] if best and item["global_min_090_tokens"] is not None else None,
            }
        if result["top4"]["success"] and not result["top10"]["success"]:
            raise ValueError("top10 coverage lost a top4 success")
        rows.append(result)
    if len(rows) != 2032:
        raise ValueError(f"unexpected C0 train-only population: {len(rows)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_query.jsonl", rows)
    summary = {
        "role": "train-only descriptive oracle cost audit; not a deployable selector result",
        "all_clean_train": summarize(rows),
        "selector_train": summarize([row for row in rows if row["fold"] != 4]),
        "selector_validation": summarize([row for row in rows if row["fold"] == 4]),
        "holdout_read": False,
        "development_read": False,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates)},
    }
    write_metadata(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
