"""Summarize paired natural insertion and the historical depth10 reference."""
import glob
import json
import random
from pathlib import Path

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17packet_obs_a1_natural_insertion"


def read(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def main():
    rows = list(read(OUT / "per_query.jsonl"))
    ids = {r["example_id"] for r in rows}
    calls = {(r["example_id"], r["arm"]): r for r in read(OUT / "per_call.jsonl")}
    d10 = {}
    for path in glob.glob(str(BASE / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for row in read(path):
            if row["example_id"] in ids and row["depth"] == 10:
                d10[row["example_id"]] = row
    assert len(rows) == len(d10) == 256 and len(calls) == 512
    rng = random.Random(20260921)
    samples = {"net_success": [], "mean_delta_f1": []}
    for _ in range(10000):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        samples["net_success"].append(sum(x["repair"] - x["break"] for x in draw))
        samples["mean_delta_f1"].append(sum(x["delta_f1"] for x in draw) / len(draw))
    ci = {}
    for key, values in samples.items():
        values.sort()
        ci[key] = [values[249], values[9749]]
    result = {
        "protocol": "OBS-A1_NATURAL_INSERTION_ANALYSIS",
        "paired_query_bootstrap_95pct": ci,
        "continuous_f1_gain_actions": sum(r["delta_f1"] > 0 for r in rows),
        "continuous_f1_loss_actions": sum(r["delta_f1"] < 0 for r in rows),
        "continuous_f1_equal_actions": sum(r["delta_f1"] == 0 for r in rows),
        "historical_p0_depth10_success_090": sum(r["f1"] + 1e-6 >= .90 for r in d10.values()),
        "historical_p0_depth10_mean_extra_context_tokens_vs_insert": sum(
            d10[r["example_id"]]["tokens"] - calls[r["example_id"], "action"]["context_tokens"]
            for r in rows) / len(rows),
        "depth10_comparison_caveat": "P0 depth10 was generated in a prior run, not paired fresh in OBS-A1.",
        "sealed_sets_read": False,
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
