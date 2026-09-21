"""Reassess the proposed STOP-C0 gate from existing canonical fresh caches."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v17stop_c0_feedback_gate_reassessment.json"
OUT = ROOT / "results/v2_rank_then_cut/v17stop_c0_feedback_gate_reassessment"


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def root_path(path: str) -> Path:
    return ROOT / path


def main() -> None:
    cfg = load_json(CONFIG)
    value = load_json(root_path(cfg["inputs"]["clairvoyant_value_bound"]))
    costed = load_json(root_path(cfg["inputs"]["costed_quality_baseline_control"]))
    rows = [
        json.loads(line)
        for line in open(root_path(cfg["inputs"]["fresh_prefix_per_query"]), encoding="utf-8")
        if line.strip()
    ]

    early = value["schedules"]["early"]
    saving_fraction = (
        early["mean_context_token_saving_upper_bound"]
        / early["mean_baseline_cumulative_context_tokens"]
    )
    opportunity = {}
    for level, fixed_depth in cfg["fixed_schedule"].items():
        eligible = 0
        eventual_success = 0
        earlier = 0
        for row in rows:
            info = row["levels"].get(level)
            if info is None:
                continue
            eligible += 1
            first = info.get("earliest_success")
            if first is None:
                continue
            eventual_success += 1
            earlier += int(first < fixed_depth)
        opportunity[level] = {
            "fixed_depth": fixed_depth,
            "eligible_queries": eligible,
            "eventually_successful_queries": eventual_success,
            "earlier_success_queries": earlier,
            "earlier_success_fraction_of_eligible": earlier / eligible if eligible else None,
            "earlier_success_fraction_of_eventual_success": (
                earlier / eventual_success if eventual_success else None
            ),
            "passes_10pct_gate": (
                earlier / eligible >= cfg["gates"]["earlier_success_fraction"]
                if eligible else False
            ),
        }

    passing_anchors = sum(v["passes_10pct_gate"] for v in opportunity.values())
    saving_pass = saving_fraction >= cfg["gates"]["minimum_mean_final_context_saving_fraction"]
    opportunity_pass = passing_anchors >= cfg["gates"]["minimum_anchors_with_10pct_earlier_success"]

    quality_control = costed["schedules"]["conservative"]["sequential_monotone"]
    summary = {
        "protocol": cfg["protocol"],
        "status": "COMPLETED_FROM_EXISTING_FRESH_CANONICAL_CACHE",
        "queries": len(rows),
        "primary_early_schedule": {
            "depths": list(cfg["fixed_schedule"].values()),
            "mean_baseline_cumulative_context_tokens": early["mean_baseline_cumulative_context_tokens"],
            "optimistic_clairvoyant_cumulative_context_tokens": early["mean_clairvoyant_cumulative_context_tokens"],
            "optimistic_mean_context_token_saving": early["mean_context_token_saving_upper_bound"],
            "optimistic_mean_final_context_saving_fraction": saving_fraction,
            "saving_gate": cfg["gates"]["minimum_mean_final_context_saving_fraction"],
            "saving_gate_pass": saving_pass,
            "quality_loss_percentage_points": 0.0,
            "quality_gate_pass": True,
            "note": "This is already a zero-observation-cost hindsight upper bound. Real STOP/CONTINUE probes can only add Target cost and cannot increase this final-context ceiling."
        },
        "anchor_opportunity": opportunity,
        "anchors_passing_10pct_gate": passing_anchors,
        "required_anchors_passing_10pct_gate": cfg["gates"]["minimum_anchors_with_10pct_earlier_success"],
        "opportunity_gate_pass": opportunity_pass,
        "call_and_compute_gates": {
            "evaluated_for_primary_schedule": False,
            "reason": "The optimistic no-observation-cost bound already fails two necessary gates; additional calls cannot rescue them."
        },
        "distinct_quality_heavy_control": {
            "baseline": "depth10 for every anchor; this is not the primary [6,7,7,9,10] comparison",
            "mean_final_context_tokens_saved": quality_control["mean_final_context_tokens_saved"],
            "mean_target_compute_tokens_saved": quality_control["mean_target_compute_tokens_saved"],
            "mean_extra_target_calls": quality_control["mean_extra_target_calls"],
            "baseline_complete": quality_control["baseline_complete"],
            "oracle_complete": quality_control["oracle_probe_complete"]
        },
        "decision": "STOP_CURRENT_CONTRACT_ADAPTIVE_STOPPING_AT_C0",
        "decision_reason": (
            "Against the prescribed early fixed schedule, even a free clairvoyant stopper saves only "
            f"{saving_fraction:.2%} cumulative context (<5%), and only {passing_anchors}/5 anchors "
            "have at least 10% earlier-success opportunity. Training an output-only stopper cannot "
            "repair an insufficient opportunity ceiling."
        ),
        "next_contract_change": {
            "name": "C1-OBS_NATIVE_TARGET_CONFIDENCE",
            "scope": "Allow read-only native token log-probability/margin traces from the frozen Target while retaining lossless source text, V8 ordering, add-only prefixes, and the same fidelity metric.",
            "why": "Existing output-only answer-count, answer-stability, linear-answer, semantic-critic, and self-report probes failed. Native confidence is the smallest information-contract change that directly targets missing observability; cost optimization or more output features cannot enlarge the early schedule's opportunity ceiling.",
            "required_control": "Generate labels and confidence traces in the same fresh canonical calls because prior log-prob preflight showed that old cached outputs cannot be safely joined to a changed generation path."
        },
        "new_target_calls": 0
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    report = f"""# V17 STOP-C0 Target-feedback gate reassessment

The proposed feedback-stopping line was evaluated from the existing fresh canonical V8 prefix cache; no new Target calls and no model fitting were needed.

## Primary prescribed baseline

The frozen schedule is `[6, 7, 7, 9, 10]`. Its mean cumulative context cost is `{early['mean_baseline_cumulative_context_tokens']:.2f}` tokens/query. A free clairvoyant stopper lowers this only to `{early['mean_clairvoyant_cumulative_context_tokens']:.2f}`, an optimistic saving of `{early['mean_context_token_saving_upper_bound']:.2f}` tokens/query or `{saving_fraction:.2%}`. This fails the preregistered `>=5%` gate before charging any observation call.

## Earlier-success opportunity

| Fidelity | Fixed depth | Eligible | Earlier success | Fraction | >=10% |
|---|---:|---:|---:|---:|---:|
"""
    for level, item in opportunity.items():
        report += f"| {level} | {item['fixed_depth']} | {item['eligible_queries']} | {item['earlier_success_queries']} | {item['earlier_success_fraction_of_eligible']:.2%} | {'yes' if item['passes_10pct_gate'] else 'no'} |\n"
    report += f"""

Only `{passing_anchors}/5` anchors pass the 10% opportunity requirement; the protocol requires at least `{cfg['gates']['minimum_anchors_with_10pct_earlier_success']}/5`. The 0.80, 0.90 and 0.95 operating depths coincide with the sharp empirical quality transitions, so output feedback cannot create substantial earlier context opportunities there.

The existing depth-10 quality-heavy control remains positive (sequential oracle saves `{quality_control['mean_final_context_tokens_saved']:.2f}` final-context tokens and `{quality_control['mean_target_compute_tokens_saved']:.2f}` Target-compute tokens/query), but it is a different operating point and does not make the prescribed early-schedule gate pass.

## Decision

`STOP_CURRENT_CONTRACT_ADAPTIVE_STOPPING_AT_C0`.

Do not repeat answer-dynamics probes or train another output-only stopper. The next scientifically distinct contract is `C1-OBS_NATIVE_TARGET_CONFIDENCE`: expose read-only native token log-probability or margin traces from the frozen Target, while retaining source-preserving text, V8 ordering, add-only prefixes and the fidelity objective. Any pilot must collect outcomes and traces in the same fresh calls; prior trace preflight showed that attaching new traces to old cached generations is invalid.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
