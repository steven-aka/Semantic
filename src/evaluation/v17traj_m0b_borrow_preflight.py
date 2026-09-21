"""Zero-Target-call eligibility and cumulative-cost audit for early rank10 atoms."""
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments

BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "v17traj_m0b_borrow_preflight"
CFG = Path("configs/v17traj_m0b_borrow_preflight.json")


def main():
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_PREFLIGHT"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    ids = {r["example_id"] for r in read_jsonl(BASE / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")}
    data = {r["example_id"]: r for r in read_jsonl(BASE / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(BASE / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(ids) == len(data) == len(orders) == 1421
    entries = []
    for q in sorted(ids):
        packets, order = data[q]["packet_texts"], orders[q]
        assert len(order) == 12
        p10 = order[9]
        baseline = {d: render(packets, mask(order, d)) for d in (6, 7, 9, 10)}
        base_cost = {d: count(text) for d, text in baseline.items()}
        options = []
        for i, fragment in enumerate(segments(packets[p10])):
            proposed7 = render(packets, mask(order, 7) | (1 << p10), {p10: fragment})
            slack7 = count(proposed7) - base_cost[7]
            if 0 < slack7 <= cfg["max_depth7_extra_context_tokens"]:
                options.append((slack7, i, fragment))
        if options:
            slack7, i, fragment = min(options)
            proposed9 = render(packets, mask(order, 9) | (1 << p10), {p10: fragment})
            slack9 = count(proposed9) - base_cost[9]
            assert slack9 > 0
            # At depth10 the original full packet is restored; source-order
            # rendering must be byte-identical to V8's depth10 context.
            restored10 = render(packets, mask(order, 10))
            assert restored10 == baseline[10]
            assert fragment.partition("\nSource evidence: ")[0] in packets[p10]
            entries.append({"example_id": q, "eligible": True, "candidate_index": i,
                            "source_packet_id": p10, "depth7_slack": slack7,
                            "depth9_slack": slack9,
                            "cumulative_slack": 2 * slack7 + slack9,
                            "depth10_exact_recovery": True,
                            "fragment_sha256": hashlib.sha256(fragment.encode()).hexdigest()})
        else:
            entries.append({"example_id": q, "eligible": False,
                            "candidate_index": None, "depth7_slack": 0,
                            "depth9_slack": 0, "cumulative_slack": 0,
                            "depth10_exact_recovery": True})
    eligible = [r for r in entries if r["eligible"]]
    result = {"protocol": cfg["protocol"], "queries": len(entries),
              "eligible_queries": len(eligible),
              "max_depth7_slack": cfg["max_depth7_extra_context_tokens"],
              "mean_depth7_slack_among_eligible": sum(r["depth7_slack"] for r in eligible) / len(eligible),
              "mean_depth9_slack_among_eligible": sum(r["depth9_slack"] for r in eligible) / len(eligible),
              "mean_cumulative_slack_among_eligible": sum(r["cumulative_slack"] for r in eligible) / len(eligible),
              "mean_cumulative_slack_all_queries_if_uniform": sum(r["cumulative_slack"] for r in entries) / len(entries),
              "max_cumulative_slack": max(r["cumulative_slack"] for r in eligible),
              "depth10_exact_recovery_count": sum(r["depth10_exact_recovery"] for r in entries),
              "sealed_sets_read": False,
              "limitation": "Only context token accounting; future Target outputs and prompt/output costs are not observed."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in entries:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
