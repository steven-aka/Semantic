"""Zero-Target-call audit of one rank10-atom / rank7-atom reveal exchange."""
import hashlib
import json
import re
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments

BASE = Path("results/v2_rank_then_cut")
ROOT = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17traj_m2a_promote_delay_preflight"
CFG = Path("configs/v17traj_m2a_promote_delay_preflight.json")


def rank7_remainders(packet):
    title, sep, proof = packet.partition("\nSource evidence: ")
    if not sep or not title.startswith("Document title: "):
        return []
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", proof) if x.strip()]
    if len(sentences) < 2 or " ".join(sentences) != proof:
        return []
    return [(i, sentence, title + sep + " ".join(sentences[:i] + sentences[i + 1:]))
            for i, sentence in enumerate(sentences)]


def main():
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_ZERO_TARGET_CALL_AUDIT" and cfg["schedule"] == [6, 7, 7, 9, 10]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    ids = {r["example_id"] for r in read_jsonl(BASE / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    m0b = {r["example_id"]: r for r in read_jsonl(BASE / "v17traj_m0b_borrow_preflight/per_query.jsonl")}
    assert len(ids) == len(data) == len(orders) == len(m0b) == 1421
    OUT.mkdir(parents=True, exist_ok=True)
    per_query, actions = [], []
    for q in sorted(ids):
        packets, order = data[q]["packet_texts"], orders[q]
        p7, p10 = order[6], order[9]
        base6 = render(packets, mask(order, 6))
        base7 = render(packets, mask(order, 7))
        base9 = render(packets, mask(order, 9))
        base10 = render(packets, mask(order, 10))
        valid = []
        raw = 0
        if m0b[q]["eligible"]:
            f = segments(packets[p10])[m0b[q]["candidate_index"]]
            future_proof = f.partition("\nSource evidence: ")[2]
            for index, delayed, remainder in rank7_remainders(packets[p7]):
                if delayed == future_proof or delayed in base6:
                    continue
                raw += 1
                partial = {p7: remainder, p10: f}
                action7 = render(packets, mask(order, 7) | (1 << p10), partial)
                action9 = render(packets, mask(order, 9) | (1 << p10), partial)
                # The change begins after S6; full originals are restored at depth10.
                assert render(packets, mask(order, 6)) == base6
                assert render(packets, mask(order, 10)) == base10
                delta7 = count(action7) - count(base7)
                delta9 = count(action9) - count(base9)
                cumulative = 2 * delta7 + delta9
                if delta7 <= 0 and delta9 <= 0 and cumulative <= 0:
                    item = {"example_id": q, "delayed_sentence_index": index,
                            "rank7_packet_id": p7, "rank10_packet_id": p10,
                            "promoted_sentence_index": m0b[q]["candidate_index"],
                            "delta_context_tokens_at_depth7": delta7,
                            "delta_context_tokens_at_depth9": delta9,
                            "delta_cumulative_context_tokens": cumulative,
                            "delayed_sentence_sha256": hashlib.sha256(delayed.encode()).hexdigest(),
                            "promoted_fragment_sha256": m0b[q]["fragment_sha256"]}
                    valid.append(item)
                    actions.append(item)
        per_query.append({"example_id": q, "m0b_eligible": m0b[q]["eligible"],
                          "raw_nondeduplicated_pairs": raw, "legal_actions": len(valid),
                          "best_delta_cumulative_context_tokens": min((x["delta_cumulative_context_tokens"] for x in valid), default=None)})
    eligible = [r for r in per_query if r["legal_actions"]]
    result = {"protocol": cfg["protocol"], "queries": len(per_query),
              "m0b_eligible_queries": sum(r["m0b_eligible"] for r in per_query),
              "query_with_rank7_exchange_candidates": sum(r["raw_nondeduplicated_pairs"] > 0 for r in per_query),
              "raw_nondeduplicated_pairs": sum(r["raw_nondeduplicated_pairs"] for r in per_query),
              "budget_neutral_actions": len(actions), "budget_neutral_queries": len(eligible),
              "queries_with_at_least_3_actions": sum(r["legal_actions"] >= 3 for r in per_query),
              "mean_legal_actions_per_eligible_query": len(actions) / len(eligible) if eligible else 0,
              "depth7_token_delta_range": [min((x["delta_context_tokens_at_depth7"] for x in actions), default=None), max((x["delta_context_tokens_at_depth7"] for x in actions), default=None)],
              "depth9_token_delta_range": [min((x["delta_context_tokens_at_depth9"] for x in actions), default=None), max((x["delta_context_tokens_at_depth9"] for x in actions), default=None)],
              "cumulative_token_delta_range": [min((x["delta_cumulative_context_tokens"] for x in actions), default=None), max((x["delta_cumulative_context_tokens"] for x in actions), default=None)],
              "affected_levels": [0.70, 0.80, 0.90], "structurally_unchanged_levels": [0.60, 0.95],
              "target_calls": 0, "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in (("per_query.jsonl", per_query), ("legal_actions.jsonl", actions)):
        with (OUT / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
