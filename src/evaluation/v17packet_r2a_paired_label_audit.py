"""Read-only R1/R2 paired-decision audit; no Target calls or model training."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("results/v2_rank_then_cut")
R1 = ROOT / "v17packet_r1_label_scale512/per_query.jsonl"
R2 = ROOT / "v17packet_r2_semantic_probe/oof.jsonl"
OUT = ROOT / "v17packet_r2a_paired_label_audit"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def category(b: dict, a: dict) -> str:
    # This frozen action contract changes only the 0.90 result.
    assert all(b["hits"][i] == a["hits"][i] for i in (0, 1, 2, 4))
    if b["hits"][3] and not a["hits"][3]:
        return "quality_break"
    if not b["hits"][3] and a["hits"][3]:
        return "quality_repair"
    if a["context_tokens"] < b["context_tokens"]:
        return "token_only_gain"
    return "no_value"


def evaluate(rows: list[dict], chosen: dict[str, dict]) -> dict:
    counts: Counter[str] = Counter()
    total_tokens = 0
    for row in rows:
        base = row["baseline"]
        action = chosen.get(row["example_id"], base)
        total_tokens += action["context_tokens"]
        if action is not base:
            counts[category(base, action)] += 1
    return {
        "switches": sum(counts.values()),
        "quality_repairs": counts["quality_repair"],
        "quality_breaks": counts["quality_break"],
        "token_only_gains": counts["token_only_gain"],
        "success_090": sum(chosen.get(r["example_id"], r["baseline"])["hits"][3] for r in rows),
        "complete": sum(chosen.get(r["example_id"], r["baseline"])["complete"] for r in rows),
        "mean_cumulative_context_tokens": total_tokens / len(rows),
    }


def main() -> None:
    rows = read_jsonl(R1)
    preds = read_jsonl(R2)
    assert len(rows) == len(preds) == 512
    by_id = {r["example_id"]: r for r in rows}
    pred_by_id = {p["example_id"]: p for p in preds}
    assert len(by_id) == len(pred_by_id) == 512
    assert by_id.keys() == pred_by_id.keys()

    action_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    repair_multiplicities: Counter[int] = Counter()
    per_query = []
    for row in rows:
        b = row["baseline"]
        p = pred_by_id[row["example_id"]]
        assert b == p["baseline"]
        eligible = [a for a in row["sentence_options"] if a["context_tokens"] <= b["context_tokens"]]
        categories = Counter(category(b, a) for a in eligible)
        action_counts.update(categories)
        for key in ("quality_repair", "quality_break", "token_only_gain", "no_value"):
            query_counts["has_" + key] += categories[key] > 0
        if categories["quality_repair"]:
            repair_multiplicities[categories["quality_repair"]] += 1
        model_action = p["best_action"]
        if model_action is not None:
            assert model_action in eligible
        per_query.append({
            "example_id": row["example_id"],
            "fold": p["fold"],
            "baseline_success_090": b["hits"][3],
            "eligible_actions": len(eligible),
            "category_counts": dict(categories),
            "model_best_category": category(b, model_action) if model_action else "STAY",
            "model_best_score": p["best_score"],
        })

    ranked = sorted((p for p in preds if p["best_action"] is not None),
                    key=lambda p: (-p["best_score"], p["example_id"]))
    policies = {}
    for k in (5, 26, 52):
        selected = {p["example_id"] for p in ranked[:k]}
        learned = {q: pred_by_id[q]["best_action"] for q in selected}
        oracle_sentence = {q: by_id[q]["strict_oracle"] for q in selected}
        # Perfect gate is quality-repair opportunity only. Token-only value is
        # reported separately because it is much more prevalent.
        perfect_repair_gate = {r["example_id"] for r in rows if any(
            category(r["baseline"], a) == "quality_repair"
            for a in r["sentence_options"]
            if a["context_tokens"] <= r["baseline"]["context_tokens"])}
        oracle_gate_learned_choice = {q: pred_by_id[q]["best_action"] for q in perfect_repair_gate}
        oracle_gate_oracle_choice = {q: by_id[q]["strict_oracle"] for q in perfect_repair_gate}
        policies[f"top_{k}"] = {
            "learned_gate_learned_choice": evaluate(rows, learned),
            "learned_gate_oracle_choice": evaluate(rows, oracle_sentence),
            "selected_with_repair_opportunity": len(selected & perfect_repair_gate),
        }
    policies["perfect_quality_repair_gate"] = {
        "learned_choice": evaluate(rows, oracle_gate_learned_choice),
        "oracle_choice": evaluate(rows, oracle_gate_oracle_choice),
    }
    baseline = evaluate(rows, {})
    strict = evaluate(rows, {r["example_id"]: r["strict_oracle"] for r in rows
                             if r["strict_index"] >= 0})
    assert baseline["success_090"] == 381 and strict["success_090"] == 420
    assert baseline["complete"] == 273 and strict["complete"] == 288
    assert action_counts["quality_repair"] == 56 and query_counts["has_quality_repair"] == 39
    assert query_counts["has_token_only_gain"] == 369
    result = {
        "protocol": "V17-PACKET-R2A_PAIRED_LABEL_AND_FAILURE_AUDIT",
        "data": "512 design-exposed train queries, frozen R1 outcomes and R2 grouped OOF best actions",
        "eligibility": "candidate cumulative context tokens <= V8; STAY is always allowed",
        "action_categories": dict(action_counts),
        "query_categories": dict(query_counts),
        "repair_action_count_by_query": dict(repair_multiplicities),
        "baseline": baseline,
        "strict_oracle": strict,
        "counterfactuals": policies,
        "limitations": [
            "R2 OOF saved only each query's highest-scored eligible action; complete within-query ranking cannot be reconstructed.",
            "Counterfactual oracle gates and choices use Target labels and are diagnostic ceilings, not deployable policies.",
            "Quality repairs are rare (39 independent queries); action count is not independent sample size.",
            "Only 0.90 changes under this action contract; multi-anchor tradeoff is structurally absent.",
            "No new Target calls, model training, sealed-set reads, or model selection.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as f:
        for item in per_query:
            f.write(json.dumps(item) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
