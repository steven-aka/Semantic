"""Read-only gate for the existing protected-insertion observability data."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/v2_rank_then_cut"
OUT = BASE / "v17packet_obs_a0_data_gate"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def hit(value):
    return value + 1e-6 >= 0.90


def main():
    pilot = BASE / "v17packet_slot_b0_pilot"
    frag = BASE / "v17packet_frag_a0_component_ablation"
    queries = {r["example_id"]: r for r in rows(pilot / "per_query.jsonl")}
    actions = rows(pilot / "per_call.jsonl")
    repeats = rows(frag / "per_case.jsonl")
    counts = Counter()
    by_query = defaultdict(Counter)
    for action in actions:
        q = action["example_id"]
        assert q in queries
        assert action["slack"] == action["context_tokens"] - queries[q]["baseline_tokens"]
        before, after = hit(queries[q]["baseline_f1"]), hit(action["f1"])
        category = "repair" if not before and after else "break" if before and not after else "same"
        counts[category] += 1
        by_query[q][category] += 1
    assert len(actions) == 84 and len(queries) == 24
    assert len({(a["example_id"], a["candidate_index"]) for a in actions}) == len(actions)
    assert all(r["reproduced"]["baseline"] for r in repeats)
    repeat_counts = Counter(r["cached_outcome"] for r in repeats)
    matched = Counter(r["cached_outcome"] for r in repeats if r["reproduced"]["title_sentence"])
    summary = {
        "contract": "V8 depth9 default preserved; one rank10 title+sentence inserted; actual slack <=48",
        "pilot_queries": len(queries),
        "pilot_actions": len(actions),
        "sampling": "12 V8 successes + 12 V8 failures, selected from design-exposed R1 train512 by pre-outcome SHA order",
        "action_outcomes_090": dict(counts),
        "query_counts_any_repair_break": {
            "repair": sum(bool(x["repair"]) for x in by_query.values()),
            "break": sum(bool(x["break"]) for x in by_query.values()),
        },
        "repeat_cases": len(repeats),
        "repeat_unique_queries": len({r["example_id"] for r in repeats}),
        "repeat_selected_by_cached_outcome": dict(repeat_counts),
        "repeat_combined_success_bit_agreement": f"{sum(matched.values())}/{len(repeats)}",
        "repeat_by_cached_outcome": {key: f"{matched[key]}/{value}" for key, value in repeat_counts.items()},
        "decision": "STOP_STRONG_OBSERVABILITY_PROBE_ON_CURRENT_INSERTION_LABELS",
        "reason": "Only 24 independent outcome-balanced queries; repeated actions are outcome-selected, so neither grouped learnability nor population repeatability is estimable reliably.",
        "next": "Freeze one insertion candidate rule and collect matched baseline/action labels on an outcome-blind train-side query sample before a single grouped paired-effect probe.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
