"""Zero-Target decomposition of insertion gate and candidate-choice ceilings."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl


BASE = Path("results/v2_rank_then_cut/v17packet_slot_b0_pilot")
OUT = Path("results/v2_rank_then_cut/v17packet_insert_d0_decomposition")
LEVEL = .90
EPS = 1e-6


def success(row: dict) -> bool:
    return row["f1"] + EPS >= LEVEL


def metrics(choices: list[tuple[dict, dict]]) -> dict:
    return {
        "success_090": sum(success(a) for _, a in choices),
        "repairs": sum(b["baseline_f1"] + EPS < LEVEL and success(a) for b, a in choices),
        "breaks": sum(b["baseline_f1"] + EPS >= LEVEL and not success(a) for b, a in choices),
        "complete": sum(all(b["baseline_hits"][i] is not False for i in (0, 1, 2, 4)) and success(a)
                        for b, a in choices),
        "interventions": sum(a["candidate_index"] is not None for _, a in choices),
        "mean_added_depth9_context_tokens": sum(a["slack"] for _, a in choices) / len(choices),
    }


def main() -> None:
    baselines = list(read_jsonl(BASE / "per_query.jsonl"))
    calls = defaultdict(list)
    for row in read_jsonl(BASE / "per_call.jsonl"):
        calls[row["example_id"]].append(row)
    assert len(baselines) == 24 and sum(map(len, calls.values())) == 84
    choices = defaultdict(list)
    per_query = []
    for b in baselines:
        q = b["example_id"]
        options = calls[q]
        assert options
        stay = {"candidate_index": None, "f1": b["baseline_f1"], "slack": 0}
        shortest = min(options, key=lambda a: (a["slack"], a["candidate_index"]))
        assert shortest["candidate_index"] == b["fixed_shortest"]["candidate_index"]
        # Frozen choice; oracle gate may only accept a genuine .90 repair.
        fixed_gate = shortest if not success(stay) and success(shortest) else stay
        # Candidate oracle is forced to insert. Its gate is permanently ON.
        forced_best = max(options, key=lambda a: (success(a), a["f1"], -a["slack"],
                                                   -a["candidate_index"]))
        # Full oracle only acts on failures and chooses the cheapest repair.
        repairs = [a for a in options if success(a)]
        full = (min(repairs, key=lambda a: (a["slack"], a["candidate_index"]))
                if not success(stay) and repairs else stay)
        for label, action in (("v8_stay", stay), ("always_shortest", shortest),
                              ("fixed_shortest_oracle_gate", fixed_gate),
                              ("always_insert_oracle_candidate", forced_best),
                              ("full_oracle", full)):
            choices[label].append((b, action))
        per_query.append({"example_id": q, "baseline_success": success(stay),
                          "shortest_repairs": not success(stay) and success(shortest),
                          "any_candidate_repairs": not success(stay) and bool(repairs),
                          "shortest_breaks": success(stay) and not success(shortest),
                          "all_candidates_break": success(stay) and not any(success(a) for a in options),
                          "candidate_count": len(options)})
    summary = {"protocol": "INSERT-D0_CACHED_GATE_CHOICE_DECOMPOSITION",
               "queries": len(baselines), "cached_candidate_actions": 84,
               "new_target_calls": 0, "new_training": False, "sealed_sets_read": False,
               "complete_correction": {"reason": "mask unattainable anchor None instead of counting it as failure",
                                       "original_summary": "original_summary_before_complete_correction.json",
                                       "source": "existing SLOT-B0 per_query.jsonl", "target_calls": 0},
               "policies": {label: metrics(rows) for label, rows in choices.items()},
               "opportunity": {
                   "failure_queries_repairable_by_any_candidate": sum(r["any_candidate_repairs"] for r in per_query),
                   "failure_queries_repairable_by_shortest": sum(r["shortest_repairs"] for r in per_query),
                   "success_queries_broken_by_shortest": sum(r["shortest_breaks"] for r in per_query),
                   "success_queries_where_all_candidates_break": sum(r["all_candidates_break"] for r in per_query),
               },
               "interpretation": [
                   "Fixed-shortest oracle gate and oracle gate for the fixed-shortest candidate are the same policy; they are not independent experiments.",
                   "Full oracle selects STAY and candidate using Target outcomes. Forced oracle candidate has no gate and exposes token cost of intervening on every query.",
                   "The 24 baseline-stratified, design-exposed train queries are not a population estimate; all oracle policies are infeasible at deployment.",
               ]}
    assert summary["policies"]["full_oracle"]["repairs"] == 7
    assert summary["policies"]["fixed_shortest_oracle_gate"]["repairs"] == 4
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
