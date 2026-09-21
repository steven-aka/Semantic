"""Compute the fixed aggressive-checkpoint -> depth10 oracle ceiling."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs/v17stop_c1_c0b_two_stage_oracle.json"
OUT = ROOT / "results/v2_rank_then_cut/v17stop_c1_c0b_two_stage_oracle"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cfg = read_json(CFG_PATH)
    states = {}
    for path in sorted(glob.glob(str(ROOT / cfg["inputs"]["per_prefix_glob"]))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            states[(row["example_id"], int(row["depth"]))] = row

    queries = [
        json.loads(line)
        for line in open(ROOT / cfg["inputs"]["per_query"], encoding="utf-8")
        if line.strip()
    ]
    expected = len(queries) * 12
    assert len(states) == expected, (len(states), expected)

    levels = list(cfg["early_depths"])
    totals = defaultdict(float)
    per_level = {}
    per_request = []
    complete = {"aggressive": 0, "depth10": 0, "oracle": 0}

    for q in queries:
        qid = q["example_id"]
        flags = {name: [] for name in complete}
        for level in levels:
            info = q["levels"].get(level)
            if info is None:
                continue
            tau = float(level)
            early_depth = int(cfg["early_depths"][level])
            early = states[(qid, early_depth)]
            late = states[(qid, int(cfg["fallback_depth"]))]
            early_ok = float(early["f1"]) + 1e-12 >= tau
            late_ok = float(late["f1"]) + 1e-12 >= tau
            selected = early if early_ok else late
            oracle_ok = early_ok or late_ok
            calls = 1 if early_depth == int(cfg["fallback_depth"]) or early_ok else 2
            compute = int(early["prompt_tokens"]) + int(early["generated_tokens"])
            if calls == 2:
                compute += int(late["prompt_tokens"]) + int(late["generated_tokens"])

            depth10_compute = int(late["prompt_tokens"]) + int(late["generated_tokens"])
            totals["requests"] += 1
            totals["aggressive_success"] += early_ok
            totals["depth10_success"] += late_ok
            totals["oracle_success"] += oracle_ok
            totals["aggressive_context"] += int(early["tokens"])
            totals["depth10_context"] += int(late["tokens"])
            totals["oracle_context"] += int(selected["tokens"])
            totals["depth10_compute"] += depth10_compute
            totals["oracle_compute"] += compute
            totals["calls"] += calls
            totals["rescues"] += (not early_ok and late_ok)
            totals["early_preserved_when_d10_fails"] += (early_ok and not late_ok)
            flags["aggressive"].append(early_ok)
            flags["depth10"].append(late_ok)
            flags["oracle"].append(oracle_ok)
            per_request.append({
                "example_id": qid,
                "fidelity": tau,
                "early_depth": early_depth,
                "early_success": early_ok,
                "depth10_success": late_ok,
                "oracle_depth": int(selected["depth"]),
                "oracle_calls": calls,
                "depth10_context_tokens": int(late["tokens"]),
                "oracle_context_tokens": int(selected["tokens"]),
                "depth10_compute_tokens": depth10_compute,
                "oracle_compute_tokens": compute
            })
        for name in complete:
            complete[name] += bool(flags[name]) and all(flags[name])

    for level in levels:
        rows = [r for r in per_request if r["fidelity"] == float(level)]
        n = len(rows)
        per_level[level] = {
            "eligible": n,
            "early_depth": cfg["early_depths"][level],
            "aggressive_success": sum(r["early_success"] for r in rows),
            "depth10_success": sum(r["depth10_success"] for r in rows),
            "two_stage_oracle_success": sum(r["early_success"] or r["depth10_success"] for r in rows),
            "depth10_rescues": sum((not r["early_success"]) and r["depth10_success"] for r in rows),
            "early_success_lost_at_depth10": sum(r["early_success"] and not r["depth10_success"] for r in rows),
            "mean_depth10_context_tokens": sum(r["depth10_context_tokens"] for r in rows) / n,
            "mean_oracle_context_tokens": sum(r["oracle_context_tokens"] for r in rows) / n,
            "mean_depth10_compute_tokens": sum(r["depth10_compute_tokens"] for r in rows) / n,
            "mean_oracle_compute_tokens": sum(r["oracle_compute_tokens"] for r in rows) / n,
            "mean_oracle_calls": sum(r["oracle_calls"] for r in rows) / n
        }

    qn = len(queries)
    context_saving = (totals["depth10_context"] - totals["oracle_context"]) / qn
    compute_saving = (totals["depth10_compute"] - totals["oracle_compute"]) / qn
    summary = {
        "protocol": cfg["protocol"],
        "status": "COMPLETED_FROM_EXISTING_FRESH_CANONICAL_CACHE",
        "queries": qn,
        "attainable_requests": int(totals["requests"]),
        "early_depths": cfg["early_depths"],
        "fallback_depth": cfg["fallback_depth"],
        "quality": {
            "aggressive_anchor_success_sum": int(totals["aggressive_success"]),
            "depth10_anchor_success_sum": int(totals["depth10_success"]),
            "two_stage_oracle_anchor_success_sum": int(totals["oracle_success"]),
            "aggressive_complete": complete["aggressive"],
            "depth10_complete": complete["depth10"],
            "two_stage_oracle_complete": complete["oracle"],
            "depth10_rescues": int(totals["rescues"]),
            "early_success_lost_at_depth10_but_preserved_by_oracle": int(totals["early_preserved_when_d10_fails"])
        },
        "cost": {
            "mean_depth10_final_context_tokens_per_query": totals["depth10_context"] / qn,
            "mean_two_stage_oracle_final_context_tokens_per_query": totals["oracle_context"] / qn,
            "mean_final_context_tokens_saved_per_query": context_saving,
            "final_context_saving_fraction": (totals["depth10_context"] - totals["oracle_context"]) / totals["depth10_context"],
            "mean_depth10_target_compute_tokens_per_query": totals["depth10_compute"] / qn,
            "mean_two_stage_oracle_target_compute_tokens_per_query": totals["oracle_compute"] / qn,
            "mean_target_compute_tokens_saved_per_query": compute_saving,
            "target_compute_saving_fraction": (totals["depth10_compute"] - totals["oracle_compute"]) / totals["depth10_compute"],
            "mean_two_stage_oracle_calls_per_attainable_request": totals["calls"] / totals["requests"],
            "maximum_calls_per_request": 2
        },
        "per_level": per_level,
        "gate": {
            "substantial_final_context_headroom": context_saving > 0,
            "positive_real_target_compute_headroom": compute_saving > 0,
            "authorize_traced_confidence_preflight": context_saving > 0 and compute_saving > 0
        },
        "decision": "GO_C1_PREFLIGHT_NATIVE_CONFIDENCE" if context_saving > 0 and compute_saving > 0 else "STOP_C1_TWO_STAGE",
        "limitations": "Clairvoyant train-side ceiling. It uses gold F1 only to define the oracle action and supplies no deployable confidence rule. Sealed splits remain closed.",
        "new_target_calls": 0
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(OUT / "per_request.jsonl", "w", encoding="utf-8") as handle:
        for row in per_request:
            handle.write(json.dumps(row) + "\n")

    report = f"""# C1-C0B two-stage oracle ceiling

This zero-call analysis fixes the aggressive first checkpoints to `[6,7,7,9,10]` and permits only one fallback, depth 10. A hindsight oracle stops early exactly when the fresh answer already reaches the requested fidelity; otherwise it pays a second full Target call at depth 10.

## Result

| Policy | Anchor successes | Complete | Mean final context/query | Mean Target compute/query |
|---|---:|---:|---:|---:|
| Aggressive fixed | {int(totals['aggressive_success'])} | {complete['aggressive']} | n/a | n/a |
| Depth10 fixed | {int(totals['depth10_success'])} | {complete['depth10']} | {totals['depth10_context']/qn:.2f} | {totals['depth10_compute']/qn:.2f} |
| Two-stage oracle | {int(totals['oracle_success'])} | {complete['oracle']} | {totals['oracle_context']/qn:.2f} | {totals['oracle_compute']/qn:.2f} |

The two-stage oracle saves **{context_saving:.2f} final-context tokens/query ({summary['cost']['final_context_saving_fraction']:.2%})** and **{compute_saving:.2f} actual Target prompt+generation tokens/query ({summary['cost']['target_compute_saving_fraction']:.2%})** relative to depth10 fixed, while using {summary['cost']['mean_two_stage_oracle_calls_per_attainable_request']:.3f} calls per attainable request (maximum two). It also preserves early successes that depth10 later loses.

Decision: `{summary['decision']}`. This authorizes only a small traced-path equivalence preflight. It does not authorize a classifier or claim deployability.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
