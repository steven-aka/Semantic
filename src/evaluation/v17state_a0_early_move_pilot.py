"""Costed train-side oracle for one V8 move before depth 6 or 7."""
from __future__ import annotations

import hashlib
import json
import os
import argparse
from collections import defaultdict
from pathlib import Path

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
LEVELS = (.60, .70, .80, .90, .95)


def mask(order, depth):
    return sum(1 << int(p) for p in order[:depth])


def moved(order, depth, rank):
    result = list(order)
    result.insert(depth, result.pop(rank - 1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v17state_a0_early_move_pilot.json")
    ap.add_argument("--output-dir", default="results/v2_rank_then_cut/v17state_a0_early_move_pilot")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output_dir)
    assert cfg["status"] == "FROZEN_TRAIN_SIDE"
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "sel_c0_train_top10_candidates.jsonl")
              if fold(r["example_id"]) != 4}
    assert len(source) == 1421
    ids = sorted(source, key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:cfg["sample_size"]]
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl")
              if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(atoms) == len(ids)
    cache = defaultdict(dict)
    p0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    paths = list(p0.glob("shard_*_of_3/per_prefix.jsonl")) + [
        Path("results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl")]
    for path in paths:
        for row in read_jsonl(path):
            if row["example_id"] in data:
                cache[row["example_id"]][row["mask"]] = row
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "per_state.jsonl"
    pilot_path = Path("results/v2_rank_then_cut/v17state_a0_early_move_pilot/per_state.jsonl")
    if cfg["sample_size"] > 64 and pilot_path.exists():
        for row in read_jsonl(pilot_path):
            if row["example_id"] in data:
                cache[row["example_id"]][row["mask"]] = row
    if output_path.exists():
        for row in read_jsonl(output_path):
            cache[row["example_id"]][row["mask"]] = row
    action_masks = {}
    tasks = {}
    for q in ids:
        order = orders[q]
        baseline10 = mask(order, 10)
        for depth in cfg["decision_depths"]:
            for rank in range(depth + 1, cfg["latest_eligible_v8_rank"] + 1):
                action = moved(order, depth, rank)
                assert mask(action, 10) == baseline10
                key = (q, depth, rank)
                action_masks[key] = [mask(action, d) for d in cfg["primary_schedule"]]
                for d, state_mask in zip(cfg["primary_schedule"], action_masks[key]):
                    if state_mask in cache[q] or (q, state_mask) in tasks:
                        continue
                    context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"])
                                          if state_mask & (1 << i))
                    tasks[q, state_mask] = {"example_id": q, "mask": state_mask,
                                            "question": data[q]["question"], "context": context,
                                            "changed_depth": d}
    pending = list(tasks.values())
    print(json.dumps({"sample": len(ids), "actions": len(action_masks), "fresh_states": len(pending)}), flush=True)
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with output_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "mask": task["mask"],
                           "changed_depth": task["changed_depth"],
                           "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                           "prediction": " # ".join(parsed),
                           "tokens": count_tokens(target.tokenizer, task["context"]),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    handle.write(json.dumps(row) + "\n")
                    cache[row["example_id"]][row["mask"]] = row
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"completed_fresh_states": min(start + len(batch), len(pending)),
                                  "total_fresh_states": len(pending)}), flush=True)
    def evaluate(q, masks):
        active = {float(x) for x in source[q]["attainable_levels"]}
        rows = [cache[q][m] for m in masks]
        hits = [None if level not in active else row["f1"] + cfg["fidelity_epsilon"] >= level
                for level, row in zip(LEVELS, rows)]
        return {"hits": hits, "complete": all(hit is not False for hit in hits),
                "context_tokens": sum(row["tokens"] for level, row in zip(LEVELS, rows) if level in active),
                "f1": [row["f1"] if level in active else None for level, row in zip(LEVELS, rows)]}

    rows = []
    for q in ids:
        base = evaluate(q, [mask(orders[q], d) for d in cfg["primary_schedule"]])
        alternatives = []
        for depth in cfg["decision_depths"]:
            for rank in range(depth + 1, cfg["latest_eligible_v8_rank"] + 1):
                item = evaluate(q, action_masks[q, depth, rank])
                alternatives.append({"depth": depth, "rank": rank, **item})
        safe = [r for r in alternatives if not any(a is True and b is False for a, b in zip(base["hits"], r["hits"]))]
        allowed = [{"depth": None, "rank": None, **base}] + safe
        def choice_key(r):
            repairs = sum(a is False and b is True for a, b in zip(base["hits"], r["hits"]))
            return (-repairs, r["context_tokens"], r["depth"] is not None,
                    r["depth"] or 0, r["rank"] or 0)
        chosen = min(allowed, key=choice_key)
        strict = min((r for r in allowed if r["context_tokens"] <= base["context_tokens"]), key=choice_key)
        rows.append({"example_id": q, "baseline": base, "actions": alternatives,
                     "quality_first_oracle": chosen, "strict_no_extra_context_oracle": strict})
    with (out / "per_query.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    def summarize(key):
        values = [r[key] for r in rows]
        return {"success": {str(level): sum(v["hits"][i] is True for v in values)
                            for i, level in enumerate(LEVELS)},
                "complete": sum(v["complete"] for v in values),
                "mean_cumulative_context_tokens": sum(v["context_tokens"] for v in values) / len(values),
                "changed_queries": sum(v["depth"] is not None for v in values) if key != "baseline" else 0}
    fresh = list(read_jsonl(output_path)) if output_path.exists() else []
    report = {"protocol": cfg["protocol"], "sample_ids_sha256": hashlib.sha256(
        "\n".join(ids).encode()).hexdigest(), "queries": len(ids), "actions": len(action_masks),
        "fresh_states": len(fresh), "fresh_target_prompt_tokens": sum(r["prompt_tokens"] for r in fresh),
        "fresh_target_generated_tokens": sum(r["generated_tokens"] for r in fresh),
        "baseline": summarize("baseline"), "quality_first_oracle": summarize("quality_first_oracle"),
        "strict_no_extra_context_oracle": summarize("strict_no_extra_context_oracle"),
        "limitations": f"Outcome-aware selected action; train-side {len(ids)}-query audit, not generalization or deployable policy."
    }
    report["reused_pilot_states"] = 320 if cfg["sample_size"] > 64 else 0
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
