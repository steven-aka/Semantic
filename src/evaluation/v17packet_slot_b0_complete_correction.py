"""Correct cached SLOT-B0 Complete counts without rerunning the Target.

Absent 0.95 anchors are ``None`` and must be masked rather than counted as
failed. The original summary is preserved beside the corrected summary.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/v2_rank_then_cut")
B0 = ROOT / "v17packet_slot_b0_pilot"


def main() -> None:
    path = B0 / "summary.json"
    summary = json.loads(path.read_text())
    original = B0 / "original_summary_before_complete_correction.json"
    assert original.exists(), "preserve the original report first"
    rows = [json.loads(line) for line in (B0 / "per_query.jsonl").read_text().splitlines()]
    assert len(rows) == 24
    assert sum(r["baseline_hits"][4] is None for r in rows) == 10

    def other_levels_ok(row):
        return all(row["baseline_hits"][i] is not False for i in (0, 1, 2, 4))

    for stratum, group in summary["by_stratum"].items():
        subset = [r for r in rows if r["stratum"] == stratum]
        assert len(subset) == group["queries"]
        group["baseline_complete"] = sum(all(h is not False for h in r["baseline_hits"]) for r in subset)
        group["fixed_shortest_complete"] = sum(other_levels_ok(r) and r["fixed_shortest"]["f1"] + 1e-6 >= .90 for r in subset)
    for cap, group in summary["by_cap_oracle"].items():
        group["baseline_complete"] = sum(all(h is not False for h in r["baseline_hits"]) for r in rows)
        group["oracle_complete"] = sum(other_levels_ok(r) and r["by_cap"][cap]["f1"] + 1e-6 >= .90 for r in rows)
    summary["complete_correction"] = {
        "reason": "attainable 0.95 is undefined on four-anchor examples; None is masked, not failure",
        "original_summary": original.name,
        "target_calls": 0,
        "source": "existing per_query.jsonl",
    }
    assert sum(v["baseline_complete"] for v in summary["by_stratum"].values()) == 10
    assert sum(v["fixed_shortest_complete"] for v in summary["by_stratum"].values()) == 9
    assert summary["by_cap_oracle"]["48"]["oracle_complete"] == 10
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["complete_correction"]))


if __name__ == "__main__":
    main()
