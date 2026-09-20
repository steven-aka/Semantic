"""Regenerate the four A3 depth-10 contexts per train query under frozen current Target."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17act_c0_learning_chain_audit import raw_cache, rendered
from src.target.qampari_runner import QampariTargetRunner
from src.training.train_v17traj_a3_1_boundary_editor_cv import canonical_choices

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)


def masks_for(order):
    return [sum(1 << packet for packet in action[:10]) for action in canonical_choices(order)]


def score(record, active):
    return {"hits": [record["f1"] + 1e-6 >= level if level in active else None for level in LEVELS],
            "tokens": record["tokens"], "context_fraction": record["tokens"] / record["full_tokens"]}


def choose(options):
    stay = options[0]
    permitted = [item for item in options if item["tokens"] <= stay["tokens"] and
                 not any(hit is False and old is True for hit, old in zip(item["hits"], stay["hits"]))]
    if not permitted:
        raise AssertionError("STAY not permitted")
    return min(permitted, key=lambda item: (-sum(a is True and b is False for a, b in zip(item["hits"], stay["hits"])),
                                              item["tokens"], item["mask"]))


def summarize(records, field):
    rows = [row[field] for row in records]
    return {"anchor_success": [sum(row["hits"][i] is True for row in rows) for i in range(5)],
            "complete": sum(all(hit is not False for hit in row["hits"]) for row in rows),
            "mean_context_fraction": mean(row["context_fraction"] for row in rows),
            "anchor_repairs": [sum(row[field]["hits"][i] is True and row["stay"]["hits"][i] is False for row in records) for i in range(5)],
            "anchor_breaks": [sum(row[field]["hits"][i] is False and row["stay"]["hits"][i] is True for row in records) for i in range(5)],
            "complete_repairs": sum(all(h is not False for h in row[field]["hits"]) and
                                    any(h is False for h in row["stay"]["hits"]) for row in records),
            "complete_breaks": sum(any(h is False for h in row[field]["hits"]) and
                                   all(h is not False for h in row["stay"]["hits"]) for row in records)}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "data", "rollouts", "annotations", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["schedule"] != [10] * 5:
        raise ValueError("unfrozen current-Target cache protocol")
    source = [row for row in read_jsonl(args.candidates) if fold(row["example_id"]) != 4]
    if len(source) != 1421:
        raise AssertionError("unexpected train-only population")
    ids = {row["example_id"] for row in source}
    data = {row["example_id"]: row for row in read_jsonl(args.data) if row["example_id"] in ids}
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts) if row["example_id"] in ids}
    atoms = {row["example_id"]: row["answer_atoms"] for row in read_jsonl(args.annotations) if row["example_id"] in ids}
    if len(data) != 1421 or len(orders) != 1421 or len(atoms) != 1421:
        raise AssertionError("incomplete train-side source join")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_mask.jsonl"
    done = {}
    if path.exists():
        for row in read_jsonl(path):
            key = (row["example_id"], row["mask"])
            if key in done:
                raise AssertionError("duplicate resume key")
            done[key] = row
    tasks = []
    for row in source:
        query = row["example_id"]
        masks = masks_for(orders[query])
        if len(set(masks)) != 4:
            raise AssertionError("four unique endpoint masks required")
        old = raw_cache(Path(args.exact_dir) / f"{query}.jsonl", set(masks))
        for mask in masks:
            if (query, mask) not in done:
                tasks.append({"example_id": query, "mask": mask,
                              "question": data[query]["question"],
                              "context": "\n\n".join(text.strip() for i, text in enumerate(data[query]["packet_texts"]) if mask & (1 << i)),
                              "tokens": old[mask][1], "full_tokens": row["full_tokens"],
                              "old_cache_f1": old[mask][0]})
    if len(done) + len(tasks) != 5684:
        raise AssertionError("unexpected task count")
    print(json.dumps({"total": 5684, "resumed": len(done), "pending": len(tasks)}), flush=True)
    cost = {"target_calls_this_run": 0, "prompt_tokens_this_run": 0, "generated_tokens_this_run": 0}
    if tasks:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([task["question"] for task in batch], [task["context"] for task in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    record = {key: value for key, value in task.items() if key not in {"question", "context"}}
                    record.update({"f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                                   "prediction": " # ".join(parsed), "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                   "generated_tokens": len(output.outputs[0].token_ids)})
                    handle.write(json.dumps(record) + "\n")
                    done[(record["example_id"], record["mask"])] = record
                    cost["target_calls_this_run"] += 1
                    cost["prompt_tokens_this_run"] += record["prompt_tokens"]
                    cost["generated_tokens_this_run"] += record["generated_tokens"]
                handle.flush()
                os.fsync(handle.fileno())
                (out / "progress.json").write_text(json.dumps({"completed": len(done), "total": 5684, "cost_this_run": cost}, indent=2) + "\n")
                print(json.dumps({"completed": len(done), "total": 5684, "cost_this_run": cost}), flush=True)
    if len(done) != 5684:
        raise AssertionError("incomplete fresh depth10 cache")
    results = []
    for row in source:
        query = row["example_id"]
        active = {float(level) for level in row["attainable_levels"]}
        options = []
        for mask in masks_for(orders[query]):
            item = score(done[(query, mask)], active)
            item["mask"] = mask
            options.append(item)
        results.append({"example_id": query, "stay": options[0], "oracle": choose(options)})
    summary = {"protocol": cfg["protocol"], "queries": len(results), "contexts": len(done),
               "stay": summarize(results, "stay"), "oracle": summarize(results, "oracle"),
               "total_cost": {"target_calls": len(done), "prompt_tokens": sum(row["prompt_tokens"] for row in done.values()),
                              "generated_tokens": sum(row["generated_tokens"] for row in done.values())},
               "limitations": ["Four-action oracle uses outcome labels and is not deployable.",
                               "The V8 order and query pool are historical train-side, not independent upstream lineage-clean validation.",
                               "Only one fixed depth10 endpoint is regenerated; this does not revalidate other stopping schedules or full trajectories."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
