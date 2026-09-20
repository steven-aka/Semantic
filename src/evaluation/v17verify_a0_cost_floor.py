"""Read-only cost floor for an extra Target call in depth-10 omission selection."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data.schemas import read_jsonl
from src.evaluation.v17exec_a1_fresh_depth10_cache import choose, score
from src.training.train_v17traj_a3_1_boundary_editor_cv import canonical_choices


def main():
    parser = argparse.ArgumentParser()
    for name in ("fresh", "candidates", "rollouts", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    fresh = list(read_jsonl(args.fresh))
    if len(fresh) != 5684:
        raise AssertionError("incomplete fresh cache")
    by_query = defaultdict(dict)
    for row in fresh:
        by_query[row["example_id"]][row["mask"]] = row
    source = {row["example_id"]: row for row in read_jsonl(args.candidates) if row["example_id"] in by_query}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts) if row["example_id"] in by_query}
    if len(source) != 1421 or len(orders) != 1421:
        raise AssertionError("wrong source population")
    savings, edited_savings = [], []
    for query, records in by_query.items():
        active = {float(value) for value in source[query]["attainable_levels"]}
        options = []
        for action in canonical_choices(orders[query]):
            mask = sum(1 << packet for packet in action[:10])
            item = score(records[mask], active)
            item["mask"] = mask
            options.append(item)
        best = choose(options)
        saved = options[0]["tokens"] - best["tokens"]
        savings.append(saved)
        if best["mask"] != options[0]["mask"]:
            edited_savings.append(saved)
    mean_call_input = mean(row["prompt_tokens"] for row in fresh)
    mean_call_output = mean(row["generated_tokens"] for row in fresh)
    mean_call_total = mean_call_input + mean_call_output
    result = {"protocol": "V17-VERIFY-A0_DEPTH10_EXTRA_TARGET_CALL_COST_FLOOR",
              "queries": len(by_query), "oracle_edited_queries": len(edited_savings),
              "oracle_mean_context_tokens_saved_all_queries": mean(savings),
              "oracle_mean_context_tokens_saved_if_edited": mean(edited_savings),
              "observed_mean_target_call_prompt_tokens": mean_call_input,
              "observed_mean_target_call_generated_tokens": mean_call_output,
              "observed_mean_target_call_total_tokens": mean_call_total,
              "extra_call_cost_to_oracle_saving_ratio_if_all_queries_probed": mean_call_total / mean(savings),
              "extra_call_cost_to_oracle_saving_ratio_if_oracle_edits_alone_probed": mean_call_total / mean(edited_savings),
              "decision": "STOP_EXTRA_TARGET_PROBE_FOR_STATIC_DEPTH10_OMISSION_TOKEN_SAVINGS",
              "limitations": ["This compares observed generation call size with token savings under a perfect outcome-aware oracle; a verifier prompt may have a different token count.",
                              "A Target call reused as the final answer is not extra, but its signal cannot retroactively change the already submitted context.",
                              "This does not rule out closed-loop feedback when a call is already required at an earlier progressive step, or quality gains at higher total cost."]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
