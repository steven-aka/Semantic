#!/usr/bin/env python3
"""Zero-call feasibility audit for a stronger two-stage stopping judge.

The deployment-aligned decision is whether depth 10 is required to rescue an
early failure.  Consequently FS states (early fail, depth-10 success) must
CONTINUE, while SS/SF/FF states may safely STOP relative to the depth-10
quality baseline.  This differs from ordinary early-sufficiency labels, which
incorrectly penalize stopping on FF states even though it saves compute without
changing quality.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)
EARLY_DEPTH = {0.60: 6, 0.70: 7, 0.80: 7, 0.90: 9, 0.95: 10}


def load_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence/traces.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v2_rank_then_cut/v17stop_c4_a0_required_stopper_feasibility"),
    )
    parser.add_argument("--grid-step", type=float, default=0.001)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--compute-ratio-cap", type=float, default=0.98)
    args = parser.parse_args()

    rows = load_jsonl(args.traces)
    by_query = defaultdict(dict)
    for row in rows:
        by_query[row["example_id"]][int(row["depth"])] = row
    assert by_query and all(all(d in states for d in (6, 7, 9, 10)) for states in by_query.values())

    query_ids = sorted(by_query)
    n = len(query_ids)
    tol_count = args.quality_tolerance * n
    decisions = []
    class_counts = {str(level): Counter() for level in LEVELS[:-1]}
    depth10_success = {}
    depth10_compute_total = 0.0
    depth10_context_total = 0.0

    for qid in query_ids:
        late = by_query[qid][10]
        late_compute = late["prompt_tokens"] + late["generated_tokens"]
        for level in LEVELS:
            late_ok = late["f1"] + 1e-12 >= level
            depth10_success[(qid, level)] = late_ok
            depth10_compute_total += late_compute
            depth10_context_total += late["context_tokens"]
            if level == 0.95:
                continue
            early = by_query[qid][EARLY_DEPTH[level]]
            early_ok = early["f1"] + 1e-12 >= level
            cls = ("S" if early_ok else "F") + ("S" if late_ok else "F")
            class_counts[str(level)][cls] += 1
            decisions.append(
                {
                    "query_id": qid,
                    "level": level,
                    "class": cls,
                    "early_compute": early["prompt_tokens"] + early["generated_tokens"],
                    "late_compute": late_compute,
                    "early_context": early["context_tokens"],
                    "late_context": late["context_tokens"],
                }
            )

    baseline_level = {
        str(level): sum(depth10_success[(qid, level)] for qid in query_ids) for level in LEVELS
    }
    baseline_complete = sum(all(depth10_success[(qid, level)] for level in LEVELS) for qid in query_ids)
    baseline_compute = depth10_compute_total / n
    baseline_context = depth10_context_total / n

    by_q_level = {(d["query_id"], d["level"]): d for d in decisions}

    def evaluate(rescue_recall: float, unnecessary_continue: float):
        # rescue_recall: P(CONTINUE | FS)
        # unnecessary_continue: P(CONTINUE | SS/SF/FF)
        per_level = {str(level): 0.0 for level in LEVELS}
        compute_total = 0.0
        context_total = 0.0
        complete = 0.0
        for qid in query_ids:
            complete_prob = 1.0
            for level in LEVELS:
                late = by_query[qid][10]
                if level == 0.95:
                    p_success = float(depth10_success[(qid, level)])
                    compute_total += late["prompt_tokens"] + late["generated_tokens"]
                    context_total += late["context_tokens"]
                else:
                    d = by_q_level[(qid, level)]
                    cls = d["class"]
                    p_continue = rescue_recall if cls == "FS" else unnecessary_continue
                    if cls == "SS":
                        p_success = 1.0
                    elif cls == "SF":
                        p_success = 1.0 - p_continue
                    elif cls == "FS":
                        p_success = p_continue
                    else:
                        p_success = 0.0
                    compute_total += d["early_compute"] + p_continue * d["late_compute"]
                    context_total += (1.0 - p_continue) * d["early_context"] + p_continue * d["late_context"]
                per_level[str(level)] += p_success
                complete_prob *= p_success
            complete += complete_prob
        mean_compute = compute_total / n
        mean_context = context_total / n
        quality_pass = all(
            per_level[str(level)] >= baseline_level[str(level)] - tol_count - 1e-9 for level in LEVELS
        ) and complete >= baseline_complete - tol_count - 1e-9
        compute_pass = mean_compute <= args.compute_ratio_cap * baseline_compute + 1e-9
        return {
            "per_level_success_expected": per_level,
            "complete_expected": complete,
            "mean_target_compute_tokens_per_query": mean_compute,
            "mean_final_context_tokens_per_query": mean_context,
            "target_compute_ratio_vs_depth10": mean_compute / baseline_compute,
            "quality_pass": quality_pass,
            "compute_pass": compute_pass,
            "feasible": quality_pass and compute_pass,
            "judge_compute_budget_tokens_per_query_at_cap": args.compute_ratio_cap * baseline_compute - mean_compute,
        }

    steps = round(1.0 / args.grid_step)
    boundary = []
    feasible_points = 0
    best_margin = None
    for fi in range(steps + 1):
        f = fi * args.grid_step
        first = None
        # Quality is monotone in rescue recall at fixed unnecessary-continue
        # rate. Find its first passing grid point, then check the upper recall
        # range allowed by the (oppositely monotone) compute constraint.
        lo, hi = 0, steps
        while lo < hi:
            mid = (lo + hi) // 2
            if evaluate(mid * args.grid_step, f)["quality_pass"]:
                hi = mid
            else:
                lo = mid + 1
        r_index = lo
        metrics = evaluate(r_index * args.grid_step, f)
        if metrics["feasible"]:
            # Count feasible r grid points explicitly; this is cheap after the
            # boundary search and avoids assuming every higher-r point fits the
            # compute cap.
            clo, chi = r_index, steps
            if evaluate(chi * args.grid_step, f)["compute_pass"]:
                max_index = chi
            else:
                while clo + 1 < chi:
                    mid = (clo + chi) // 2
                    if evaluate(mid * args.grid_step, f)["compute_pass"]:
                        clo = mid
                    else:
                        chi = mid
                max_index = clo
            feasible_points += max_index - r_index + 1
            first = {"unnecessary_continue_rate": f, "minimum_rescue_recall": r_index * args.grid_step, **metrics}
            margin = metrics["judge_compute_budget_tokens_per_query_at_cap"]
            if best_margin is None or margin > best_margin["judge_compute_budget_tokens_per_query_at_cap"]:
                best_margin = first
        if first:
            boundary.append(first)

    probes = {}
    for r, f in ((0.80, 0.01), (0.90, 0.01), (0.95, 0.01), (0.97, 0.002), (0.99, 0.0), (1.0, 0.0)):
        probes[f"rescue_{r:.3f}_unnecessary_{f:.3f}"] = evaluate(r, f)

    realistic = [p for p in boundary if p["minimum_rescue_recall"] <= 0.90 and p["unnecessary_continue_rate"] >= 0.01]
    decision = "GO_C4_AUX_RESCUE_JUDGE_PROTOCOL" if realistic else "STOP_STRONGER_STOPPER_BEFORE_TRAINING"
    result = {
        "protocol": "V17-STOP-C4-A0_REQUIRED_STOPPER_FEASIBILITY",
        "status": "COMPLETED_ZERO_TARGET_CALLS",
        "queries": n,
        "decision_examples": len(decisions),
        "decision_semantics": {
            "FS": "CONTINUE is required: early fails and depth10 succeeds",
            "SS": "STOP is safe: both early and depth10 succeed",
            "SF": "STOP is beneficial: early succeeds and depth10 rolls back",
            "FF": "STOP is quality-neutral and cheaper: both fail",
            "axes": {
                "rescue_recall": "P(CONTINUE | FS)",
                "unnecessary_continue_rate": "P(CONTINUE | SS or SF or FF)",
            },
            "why_not_plain_tpr_fpr": "Early-sufficiency labels treat FF as negative, although stopping on FF is quality-neutral and saves compute. They also hide rollback-preserving SF states. The deployment action is rescue necessity, not absolute sufficiency.",
        },
        "class_counts_per_level": {level: dict(counts) for level, counts in class_counts.items()},
        "depth10_baseline": {
            "per_level_success": baseline_level,
            "complete": baseline_complete,
            "mean_target_compute_tokens_per_query": baseline_compute,
            "mean_final_context_tokens_per_query": baseline_context,
        },
        "gate": {
            "quality_tolerance_fraction": args.quality_tolerance,
            "quality_tolerance_expected_queries": tol_count,
            "target_compute_ratio_cap": args.compute_ratio_cap,
            "realistic_region_definition": "minimum rescue recall <= 0.90 at unnecessary-continue rate >= 0.01",
            "judge_cost_note": "The grid excludes auxiliary-judge cost. The reported budget at the 0.98 cap is the maximum equivalent Target-token allowance before the point ceases to pass deployment compute.",
        },
        "grid": {
            "step": args.grid_step,
            "feasible_grid_points": feasible_points,
            "boundary": boundary,
            "best_compute_margin_boundary_point": best_margin,
            "selected_probes": probes,
        },
        "decision": decision,
        "limitations": "Design-exposed 256-query C1 cohort. Expected metrics assume a judge with class-conditional rates shared across decisions and independent decisions for expected Complete. A learned judge still requires query-grouped OOF and explicit compute accounting; this audit cannot establish learnability.",
        "new_target_calls": 0,
        "sealed_sets_read": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "feasible_boundary.jsonl").open("w") as handle:
        for point in boundary:
            handle.write(json.dumps(point) + "\n")
    print(json.dumps({
        "decision": decision,
        "class_counts_per_level": result["class_counts_per_level"],
        "baseline": result["depth10_baseline"],
        "boundary_first": boundary[0] if boundary else None,
        "boundary_last": boundary[-1] if boundary else None,
        "best_margin": best_margin,
        "selected_probes": probes,
    }, indent=2))


if __name__ == "__main__":
    main()
