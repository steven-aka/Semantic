"""Fresh 8B bounded move-to-depth9 oracle with fixed causal stop schedules."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/v2_rank_then_cut/v17prog_b0_depth9_move_oracle")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    cfg = json.loads(Path("configs/v17prog_b0_depth9_move_oracle.json").read_text())
    if cfg["status"] != "FROZEN_TRAIN_SIDE" or cfg["move_depth"] != 9:
        raise AssertionError("unfrozen B0")
    root = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
    source = {r["example_id"]: r for r in read_jsonl(root / "sel_c0_train_top10_candidates.jsonl")
              if fold(r["example_id"]) != 4}
    data = {r["example_id"]: r for r in read_jsonl(root / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in source}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(root / "sel_c0_train_v8_rollouts.jsonl")
              if r["example_id"] in source}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in source}
    if any(len(x) != 1421 for x in (source, data, orders, atoms)):
        raise AssertionError("source join mismatch")
    cache = defaultdict(dict)
    p0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    for path in sorted(p0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            cache[row["example_id"]][row["mask"]] = row
    for row in read_jsonl("results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl"):
        cache[row["example_id"]][row["mask"]] = row
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("per_state.jsonl" if args.num_shards == 1 else
                  f"per_state_shard_{args.shard_index}_of_{args.num_shards}.jsonl")
    for existing_path in sorted(out.glob("per_state*.jsonl")):
        for row in read_jsonl(existing_path):
            if row["mask"] in cache[row["example_id"]]:
                raise AssertionError("duplicate or previously cached state")
            cache[row["example_id"]][row["mask"]] = row
    tasks = []
    action_masks = {}
    for q in sorted(source):
        order = orders[q]
        for rank in (9, 10, 11, 12):
            moved = list(order)
            moved.insert(8, moved.pop(rank - 1))
            masks = {d: sum(1 << p for p in moved[:d]) for d in (6, 7, 8, 9, 10)}
            action_masks[q, rank] = masks
            for d in (9, 10):
                mask = masks[d]
                if mask in cache[q]:
                    continue
                context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"]) if mask & (1 << i))
                tasks.append({"example_id": q, "mask": mask, "question": data[q]["question"], "context": context})
                cache[q][mask] = None  # deduplicate task list
    completed = sum(1 for q in source for row in cache[q].values()
                    if row is not None and row.get("source") == "B0 depth9 move")
    if len(tasks) + completed != 5684:
        # Exactly four new depth9/10 masks per query after P0 and A0 reuse.
        raise AssertionError(f"unexpected B0 task accounting: {len(tasks)}")
    if not args.aggregate_only:
        tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
    print(json.dumps({"new_states_total": 5684, "previously_completed": completed,
                      "pending_this_shard": len(tasks)}), flush=True)
    if tasks and not args.aggregate_only:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        initial_completed = completed
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([x["question"] for x in batch], [x["context"] for x in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "mask": task["mask"],
                           "source": "B0 depth9 move",
                           "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                           "prediction": " # ".join(parsed),
                           "tokens": count_tokens(target.tokenizer, task["context"]),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    handle.write(json.dumps(row) + "\n")
                    cache[row["example_id"]][row["mask"]] = row
                handle.flush()
                os.fsync(handle.fileno())
                completed += len(batch)
                progress = {"completed_before_start": initial_completed, "completed_this_shard": start + len(batch),
                            "total": 5684, "shard_index": args.shard_index, "num_shards": args.num_shards}
                (out / f"progress_shard_{args.shard_index}_of_{args.num_shards}.json").write_text(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
    if args.num_shards > 1 and not args.aggregate_only:
        return
    # Other shard processes have exited before aggregate-only is invoked.
    if args.aggregate_only:
        for existing_path in sorted(out.glob("per_state*.jsonl")):
            for row in read_jsonl(existing_path):
                cache[row["example_id"]][row["mask"]] = row
    if any(cache[q][mask] is None for (q, _), masks in action_masks.items() for mask in masks.values()):
        raise AssertionError("missing mask")

    def evaluation(q: str, rank: int, schedule: list[int]) -> dict:
        active = {float(x) for x in source[q]["attainable_levels"]}
        selected = [cache[q][action_masks[q, rank][d]] for d in schedule]
        if any(row is None for row in selected):
            raise AssertionError("incomplete action")
        hits = [None if level not in active else row["f1"] + cfg["fidelity_epsilon"] >= level
                for level, row in zip(LEVELS, selected)]
        return {"hits": hits, "complete": all(x is not False for x in hits),
                "tokens": sum(row["tokens"] for level, row in zip(LEVELS, selected) if level in active)}

    def summary(rows: list[dict]) -> dict:
        return {"success": {str(level): sum(r["hits"][i] is True for r in rows)
                            for i, level in enumerate(LEVELS)},
                "complete": sum(r["complete"] for r in rows),
                "mean_cumulative_context_tokens": mean(r["tokens"] for r in rows)}

    schedules = {}
    detail = []
    for name in ("primary_schedule", "sensitivity_schedule"):
        schedule = cfg[name]
        stay, adjacent, strict, quality = [], [], [], []
        strict_actions, quality_actions = [], []
        for q in sorted(source):
            baseline = evaluation(q, 9, schedule)
            options = [(rank, evaluation(q, rank, schedule)) for rank in (9, 10, 11, 12)]
            safe = [(rank, r) for rank, r in options if not any(
                a is True and b is False for a, b in zip(baseline["hits"], r["hits"]))]
            strict_pool = [(rank, r) for rank, r in safe if r["tokens"] <= baseline["tokens"]]
            key = lambda item: (-sum(a is False and b is True for a, b in zip(
                baseline["hits"], item[1]["hits"])), item[1]["tokens"], item[0])
            strict_rank, strict_result = min(strict_pool, key=key)
            quality_rank, quality_result = min(safe, key=key)
            stay.append(baseline)
            adjacent.append(min([item for item in safe if item[0] <= 10], key=key)[1])
            strict.append(strict_result)
            quality.append(quality_result)
            strict_actions.append(strict_rank)
            quality_actions.append(quality_rank)
            detail.append({"schedule": name, "example_id": q, "stay": baseline,
                           "actions": {str(rank): result for rank, result in options},
                           "strict_rank": strict_rank, "quality_rank": quality_rank})
        schedules[name] = {"depths": schedule, "stay": summary(stay), "adjacent_move_oracle": summary(adjacent),
                           "strict_no_extra_context_oracle": summary(strict),
                           "quality_first_no_break_oracle": summary(quality),
                           "strict_action_counts": {str(rank): strict_actions.count(rank) for rank in (9, 10, 11, 12)},
                           "quality_action_counts": {str(rank): quality_actions.count(rank) for rank in (9, 10, 11, 12)}}
    fresh = [row for existing_path in sorted(out.glob("per_state*.jsonl"))
             for row in read_jsonl(existing_path)]
    report = {"protocol": cfg["protocol"], "queries": 1421, "fresh_target_states": len(fresh),
              "fresh_prompt_tokens": sum(r["prompt_tokens"] for r in fresh),
              "fresh_generated_tokens": sum(r["generated_tokens"] for r in fresh),
              "schedules": schedules,
              "limitations": "Outcome-aware train-side oracle. Fixed stop depths remove hindsight stopping but choosing move still uses Target truth. Not a deployable policy; sealed sets untouched."}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "per_query_decisions.jsonl").open("w") as handle:
        for row in detail:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report["schedules"], indent=2), flush=True)


if __name__ == "__main__":
    main()
