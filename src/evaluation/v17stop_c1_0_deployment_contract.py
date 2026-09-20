"""Costed STOP versus retained-CONTINUE contract on the train-side cache."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.schemas import read_jsonl

LEVELS = ("0.6", "0.7", "0.8", "0.9", "0.95")
ROOT = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT = Path("results/v2_rank_then_cut/v17stop_c1_0_deployment_contract")
CONTINUE = "depth10_plus_old_context_supported_depth9"


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_c1_0_deployment_contract.json").read_text())
    assert cfg["status"] == "FROZEN_READ_ONLY_TRAIN_SIDE"
    rows = list(read_jsonl(ROOT / "per_query.jsonl"))
    assert len(rows) == 1421 and len({r["example_id"] for r in rows}) == 1421
    report = {"protocol": cfg["protocol"], "population": len(rows), "anchors": {},
              "new_target_calls": 0, "sealed_outcome_sets_read": False}
    transitions = []
    for level in LEVELS:
        eligible = [r for r in rows if level in r["outputs"]["depth9_only"]["success"]]
        counts = {"SS": 0, "SF": 0, "FS": 0, "FF": 0}
        added = []
        min_cost_max_quality = 0
        for r in eligible:
            stop = r["outputs"]["depth9_only"]["success"][level]
            cont = r["outputs"][CONTINUE]["success"][level]
            base = r["outputs"]["depth10_only"]["success"][level]
            key = ("S" if stop else "F") + ("S" if cont else "F")
            counts[key] += 1
            # For max-quality minimum-cost hindsight: continue only when STOP fails and CONTINUE succeeds.
            min_cost_max_quality += r["depth9_target_tokens"] + (r["depth10_target_tokens"] if key == "FS" else 0)
            if key == "FS":
                added.append(r["depth10_target_tokens"])
            transitions.append({"example_id": r["example_id"], "anchor": level,
                                "stop_success": stop, "continue_success": cont,
                                "depth10_success": base, "transition": key,
                                "stop_target_tokens": r["depth9_target_tokens"],
                                "continue_target_tokens": r["depth9_target_tokens"] + r["depth10_target_tokens"]})
        n = len(eligible)
        baseline_success = sum(r["outputs"]["depth10_only"]["success"][level] for r in eligible)
        stop_success = counts["SS"] + counts["SF"]
        continue_success = counts["SS"] + counts["FS"]
        ceiling = stop_success + counts["FS"]
        # Strictly an oracle cost floor: hindsight selects the cheapest successful FS cases.
        same_quality_cost = None
        if ceiling >= baseline_success:
            same_quality_cost = (sum(r["depth9_target_tokens"] for r in eligible) +
                                 sum(sorted(added)[:max(0, baseline_success - stop_success)])) / n
        report["anchors"][level] = {
            "attainable_requests": n, "transitions_stop_vs_retained_continue": counts,
            "stop_success": stop_success, "retained_continue_success": continue_success,
            "single_depth10_success": baseline_success,
            "hindsight_max_success": ceiling,
            "single_depth10_mean_target_tokens": sum(r["depth10_target_tokens"] for r in eligible) / n,
            "stop_mean_target_tokens": sum(r["depth9_target_tokens"] for r in eligible) / n,
            "always_continue_mean_target_tokens": sum(r["depth9_target_tokens"] + r["depth10_target_tokens"] for r in eligible) / n,
            "hindsight_min_mean_tokens_at_max_success": min_cost_max_quality / n,
            "hindsight_min_mean_tokens_at_single_depth10_success": same_quality_cost,
        }
    report["complete"] = {
        "single_depth10": sum(r["outputs"]["depth10_only"]["complete"] for r in rows),
        "always_continue": sum(r["outputs"][CONTINUE]["complete"] for r in rows),
        "hindsight_independent_anchor_max": sum(all(
            r["outputs"]["depth9_only"]["success"].get(level, False) or
            r["outputs"][CONTINUE]["success"].get(level, False)
            for level in r["outputs"]["depth9_only"]["success"]
        ) for r in rows),
        "note": "Independent attainable-anchor requests; not one shared stop action for all five levels."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "per_request.jsonl").open("w") as handle:
        for row in transitions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
