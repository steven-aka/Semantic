"""Fresh 8B counterfactuals for three bounded V8 adjacent swaps."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

LEVELS = (.60, .70, .80, .90, .95)
SWAPS = (6, 7, 9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v17prog_a0_fresh_adjacent_swap_oracle.json")
    ap.add_argument("--candidates", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
    ap.add_argument("--data", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
    ap.add_argument("--rollouts", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
    ap.add_argument("--annotations", default="data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
    ap.add_argument("--p0-root", default="results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    ap.add_argument("--output-dir", default="results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["primary_schedule"] != [6, 7, 7, 9, 10]:
        raise ValueError("protocol changed")
    source = {r["example_id"]: r for r in read_jsonl(args.candidates) if fold(r["example_id"]) != 4}
    if len(source) != 1421:
        raise AssertionError("unexpected train-only population")
    data = {r["example_id"]: r for r in read_jsonl(args.data) if r["example_id"] in source}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts) if r["example_id"] in source}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(args.annotations) if r["example_id"] in source}
    if any(len(x) != 1421 for x in (data, orders, atoms)):
        raise AssertionError("incomplete source join")
    base = defaultdict(dict)
    for path in sorted(Path(args.p0_root).glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            q, depth = row["example_id"], row["depth"]
            if depth in base[q]:
                raise AssertionError("duplicate P0")
            base[q][depth] = row
    if len(base) != 1421 or any(len(rows) != 12 for rows in base.values()):
        raise AssertionError("P0 incomplete")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_action.jsonl"
    done = {}
    if path.exists():
        for row in read_jsonl(path):
            key = row["example_id"], row["swap_depth"]
            if key in done:
                raise AssertionError("duplicate resumed action")
            done[key] = row
    tasks = []
    for q in sorted(source):
        order = orders[q]
        for d in SWAPS:
            if (q, d) in done:
                continue
            changed = list(order)
            changed[d - 1], changed[d] = changed[d], changed[d - 1]
            mask = sum(1 << p for p in changed[:d])
            if mask == base[q][d]["mask"] or sum(1 << p for p in changed[:d+1]) != base[q][d+1]["mask"]:
                raise AssertionError("swap endpoint semantics")
            context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"]) if mask & (1 << i))
            tasks.append({"example_id": q, "swap_depth": d, "mask": mask, "question": data[q]["question"], "context": context})
    if len(done) + len(tasks) != 1421 * 3:
        raise AssertionError("task accounting")
    print(json.dumps({"total": 4263, "resumed": len(done), "pending": len(tasks)}), flush=True)
    if tasks:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([x["question"] for x in batch], [x["context"] for x in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {k: task[k] for k in ("example_id", "swap_depth", "mask")}
                    row.update({"f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                                "prediction": " # ".join(parsed),
                                "tokens": count_tokens(target.tokenizer, task["context"]),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids)})
                    handle.write(json.dumps(row) + "\n")
                    done[(row["example_id"], row["swap_depth"])] = row
                handle.flush()
                os.fsync(handle.fileno())
                (out / "progress.json").write_text(json.dumps({"completed": len(done), "total": 4263}) + "\n")
                print(json.dumps({"completed": len(done), "total": 4263}), flush=True)
    if len(done) != 4263:
        raise AssertionError("incomplete action cache")

    def evaluate(q, swap_depth, schedule):
        active = {float(x) for x in source[q]["attainable_levels"]}
        rows = [done[q, d] if swap_depth == d else base[q][d] for d in schedule]
        hits = [None if level not in active else row["f1"] + cfg["fidelity_epsilon"] >= level
                for level, row in zip(LEVELS, rows)]
        cost = sum(row["tokens"] for level, row in zip(LEVELS, rows) if level in active)
        return {"hits": hits, "complete": all(x is not False for x in hits), "tokens": cost,
                "mean_context_fraction": mean(row["tokens"] / base[q][12]["full_tokens"]
                                              for level, row in zip(LEVELS, rows) if level in active)}

    result = {}
    for schedule_name in ("primary_schedule", "sensitivity_schedule"):
        schedule = cfg[schedule_name]
        records = []
        for q in sorted(source):
            baseline = evaluate(q, None, schedule)
            options = [(None, baseline)] + [(d, evaluate(q, d, schedule)) for d in SWAPS]
            permitted = [(d, v) for d, v in options if v["tokens"] <= baseline["tokens"] and
                         not any(a is True and b is False for a, b in zip(baseline["hits"], v["hits"]))]
            best_d, best = min(permitted, key=lambda item: (
                -sum(a is False and b is True for a, b in zip(baseline["hits"], item[1]["hits"])),
                item[1]["tokens"], item[0] is not None, item[0] or 0))
            records.append({"example_id": q, "stay": baseline, "oracle": best, "action": best_d})
        def summarize(which):
            rows = [r[which] for r in records]
            return {"success": {str(level): sum(row["hits"][i] is True for row in rows) for i, level in enumerate(LEVELS)},
                    "complete": sum(row["complete"] for row in rows),
                    "mean_cumulative_context_tokens": mean(row["tokens"] for row in rows),
                    "mean_normalized_cumulative_context": mean(row["mean_context_fraction"] for row in rows)}
        result[schedule_name] = {"depths": schedule, "stay": summarize("stay"), "oracle": summarize("oracle"),
                                 "edited_queries": sum(r["action"] is not None for r in records),
                                 "safe_repair_queries": sum(any(a is False and b is True for a, b in zip(r["stay"]["hits"], r["oracle"]["hits"])) for r in records),
                                 "complete_repairs": sum(not r["stay"]["complete"] and r["oracle"]["complete"] for r in records),
                                 "action_counts": {str(d): sum(r["action"] == d for r in records) for d in (None, *SWAPS)}}
        # Secondary design-side analysis only: this was not the frozen primary oracle rule.
        quality_first = []
        for q in sorted(source):
            baseline = evaluate(q, None, schedule)
            options = [(None, baseline)] + [(d, evaluate(q, d, schedule)) for d in SWAPS]
            safe = [(d, v) for d, v in options if not any(a is True and b is False
                    for a, b in zip(baseline["hits"], v["hits"]))]
            _, best = min(safe, key=lambda item: (
                -sum(a is False and b is True for a, b in zip(baseline["hits"], item[1]["hits"])),
                item[1]["tokens"], item[0] is not None, item[0] or 0))
            quality_first.append(best)
        result[schedule_name]["quality_first_exploratory"] = {
            "success": {str(level): sum(row["hits"][i] is True for row in quality_first)
                        for i, level in enumerate(LEVELS)},
            "complete": sum(row["complete"] for row in quality_first),
            "mean_cumulative_context_tokens": mean(row["tokens"] for row in quality_first),
            "mean_paired_context_token_delta": mean(row["tokens"] - record["stay"]["tokens"]
                                                     for row, record in zip(quality_first, records)),
            "role": "post-primary outcome-aware diagnostic; not a deployment or preregistered gate result"}
    summary = {"protocol": cfg["protocol"], "queries": 1421, "target_calls": len(done),
               "prompt_tokens": sum(x["prompt_tokens"] for x in done.values()),
               "generated_tokens": sum(x["generated_tokens"] for x in done.values()),
               "schedules": result,
               "limitations": "Outcome-aware oracle on design-side train queries; no deployable policy or independent quality claim."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
