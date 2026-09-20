"""Read-only decomposition after the frozen SEL-C1 train-only STOP decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from src.data.schemas import read_jsonl
from src.reproducibility import sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--train-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.train_summary).read_text())
    if summary["decision"] != "STOP_SEL_C1_TRAIN_ONLY_GATE" or not summary["holdout_read"] is False:
        raise ValueError("diagnostic requires train-only STOP result")
    candidates = {row["example_id"]: row for row in read_jsonl(args.candidates)}
    choices = list(read_jsonl(args.choices))
    if len(choices) != summary["selector_validation_queries"]:
        raise ValueError("validation choices differ from frozen summary")
    rows = []
    for choice in choices:
        row = candidates[choice["example_id"]]
        top4 = row["candidates"][choice["top4_selected_rank"]]
        top10 = row["candidates"][choice["top10_selected_rank"]]
        rows.append((row, top4, top10))
    new = [item for item in rows if not any(candidate["success"][3] for candidate in item[0]["candidates"][:5]) and any(candidate["success"][3] for candidate in item[0]["candidates"])]
    both = [item for item in rows if item[1]["success"][3] and item[2]["success"][3]]
    deltas = [item[2]["earliest_tokens"][3] - item[1]["earliest_tokens"][3] for item in both]
    later = [item for item in rows if item[2]["rank"] >= 5]
    output = {
        "role": "read-only diagnostic after preregistered train-only STOP; no model selection",
        "queries": len(rows),
        "new_top10_oracle_opportunities": len(new),
        "new_opportunities_repaired_by_top10_selector": sum(item[2]["success"][3] for item in new),
        "top10_selected_rank_5_to_10": len(later),
        "rank_5_to_10_selected_successes": sum(item[2]["success"][3] for item in later),
        "rank_5_to_10_selected_complete": sum(item[2]["complete"] for item in later),
        "both_arms_success_090": len(both),
        "top10_longer_on_both_success": sum(delta > 0 for delta in deltas),
        "top10_shorter_on_both_success": sum(delta < 0 for delta in deltas),
        "mean_top10_minus_top4_earliest_090_tokens_both_success": mean(deltas) if deltas else None,
        "median_top10_minus_top4_earliest_090_tokens_both_success": median(deltas) if deltas else None,
        "mean_unique_projected_orders_out_of_11": mean(item[0]["unique_candidate_orders"] for item in rows),
        "holdout_read": False,
        "artifacts": {"candidates_sha256": sha256(args.candidates), "choices_sha256": sha256(args.choices), "train_summary_sha256": sha256(args.train_summary)},
    }
    write_metadata(args.output, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
