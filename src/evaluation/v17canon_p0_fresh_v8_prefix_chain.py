"""Resumeable current-Target V8 prefix-chain cache on the train-only queries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v17canon_p0_fresh_v8_prefix_chain.json")
    parser.add_argument("--candidates", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
    parser.add_argument("--data", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
    parser.add_argument("--rollouts", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
    parser.add_argument("--annotations", default="data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
    parser.add_argument("--fresh-depth10", default="results/v2_rank_then_cut/v17exec_a1_fresh_depth10_cache/per_mask.jsonl")
    parser.add_argument("--output-dir", default="results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["depths"] != list(range(1, 13)):
        raise ValueError("protocol changed")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    population = sorted((r for r in read_jsonl(args.candidates) if fold(r["example_id"]) != 4), key=lambda r: r["example_id"])
    if len(population) != 1421 or len({r["example_id"] for r in population}) != 1421:
        raise AssertionError("unexpected population")
    source = [r for i, r in enumerate(population) if i % args.shard_count == args.shard_index]
    ids = {r["example_id"] for r in source}
    data = {r["example_id"]: r for r in read_jsonl(args.data) if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts) if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(args.annotations) if r["example_id"] in ids}
    fresh = {(r["example_id"], r["mask"]): r for r in read_jsonl(args.fresh_depth10) if r["example_id"] in ids}
    if any(len(x) != len(ids) for x in (data, orders, atoms)):
        raise AssertionError("incomplete source join")
    out = Path(args.output_dir) / f"shard_{args.shard_index}_of_{args.shard_count}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_prefix.jsonl"
    done = {}
    if path.exists():
        for record in read_jsonl(path):
            key = (record["example_id"], record["depth"])
            if key in done:
                raise AssertionError("duplicate resume key")
            done[key] = record
    tasks = []
    reuse = []
    for row in source:
        q = row["example_id"]
        order = orders[q]
        if sorted(order) != list(range(12)):
            raise AssertionError(f"bad V8 permutation: {q}")
        for depth in cfg["depths"]:
            if (q, depth) in done:
                continue
            mask = sum(1 << p for p in order[:depth])
            context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"]) if mask & (1 << i))
            task = {"example_id": q, "depth": depth, "mask": mask, "question": data[q]["question"],
                    "context": context, "full_tokens": row["full_tokens"]}
            (reuse if depth == 10 else tasks).append(task)
    if len(done) + len(tasks) + len(reuse) != len(source) * 12:
        raise AssertionError("task accounting error")
    with path.open("a", encoding="utf-8") as handle:
        for task in reuse:
            cached = fresh.get((task["example_id"], task["mask"]))
            if cached is None:
                raise AssertionError("missing fresh depth10 STAY cache")
            record = {k: task[k] for k in ("example_id", "depth", "mask", "full_tokens")}
            record.update({k: cached[k] for k in ("f1", "prediction", "tokens", "prompt_tokens", "generated_tokens")})
            record["source"] = "EXEC-A1 current-contract depth10"
            handle.write(json.dumps(record) + "\n")
            done[(record["example_id"], 10)] = record
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"shard": args.shard_index, "total": len(source) * 12, "reused_or_resumed": len(done), "pending_target_calls": len(tasks)}), flush=True)
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
                    record = {k: task[k] for k in ("example_id", "depth", "mask", "full_tokens")}
                    record.update({"f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                                   "prediction": " # ".join(parsed),
                                   "tokens": count_tokens(target.tokenizer, task["context"]),
                                   "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                   "generated_tokens": len(output.outputs[0].token_ids), "source": "CANON-P0 current-contract"})
                    handle.write(json.dumps(record) + "\n")
                    done[(record["example_id"], record["depth"])] = record
                handle.flush()
                os.fsync(handle.fileno())
                (out / "progress.json").write_text(json.dumps({"completed": len(done), "total": len(source) * 12,
                    "target_calls_completed": sum(x["source"] == "CANON-P0 current-contract" for x in done.values())}, indent=2) + "\n")
                print(json.dumps({"completed": len(done), "total": len(source) * 12}), flush=True)
    if len(done) != len(source) * 12:
        raise AssertionError("incomplete prefix chain")


if __name__ == "__main__":
    main()
