"""Rerun paired A3 STAY/beneficial masks on the frozen Qwen3-8B Target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17act_c0_learning_chain_audit import independent_label, independent_outcome, raw_cache, rendered, select_ids
from src.target.qampari_runner import QampariTargetRunner
from src.training.train_v17traj_a3_1_boundary_editor_cv import canonical_choices


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "candidates", "data", "rollouts", "oof", "annotations", "exact-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY":
        raise ValueError("unfrozen rerun protocol")
    selected = select_ids(list(read_jsonl(args.oof)), 64)
    sources = {r["example_id"]: r for r in read_jsonl(args.candidates) if r["example_id"] in selected}
    data = {r["example_id"]: r for r in read_jsonl(args.data) if r["example_id"] in selected}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts) if r["example_id"] in selected}
    annotations = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(args.annotations) if r["example_id"] in selected}
    eligible = []
    for query in sorted(selected, key=lambda q: hashlib.sha256(q.encode()).hexdigest()):
        actions = canonical_choices(orders[query])
        masks = [sum(1 << packet for packet in action[:10]) for action in actions]
        cache = raw_cache(Path(args.exact_dir) / f"{query}.jsonl", set(masks))
        active = {float(level) for level in sources[query]["attainable_levels"]}
        outcomes = [independent_outcome(action, cache, active) for action in actions]
        positives = [i for i in range(1, 4) if independent_label(outcomes[i], outcomes[0]) == 1]
        if positives:
            i = positives[0]
            eligible.append({"example_id": query, "action_index": i, "masks": [masks[0], masks[i]],
                             "cached_f1": [cache[masks[0]][0], cache[masks[i]][0]],
                             "cached_tokens": [cache[masks[0]][1], cache[masks[i]][1]],
                             "active": sorted(active), "cached_label": 1,
                             "question": data[query]["question"],
                             "contexts": [rendered(data[query]["packet_texts"], actions[0]),
                                          rendered(data[query]["packet_texts"], actions[i])], "reruns": []})
        if len(eligible) == 12:
            break
    if len(eligible) != 12 or any(row["example_id"] not in annotations for row in eligible):
        raise AssertionError("insufficient audited positive pairs or annotations")
    target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                 gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    all_inputs = [(row, i) for row in eligible for i in range(2)]
    cost = {"target_calls": 0, "prompt_tokens": 0, "generated_tokens": 0}
    for repeat in range(cfg["repeat_generations"]):
        for start in range(0, len(all_inputs), cfg["batch_size"]):
            batch = all_inputs[start:start + cfg["batch_size"]]
            prompts = target._chat_prompts([row["question"] for row, _ in batch],
                                           [row["contexts"][i] for row, i in batch])
            outputs = target.model.generate(prompts, params, use_tqdm=False)
            for (row, i), prompt, output in zip(batch, prompts, outputs):
                value = float(qampari_list_metrics(parse_list_prediction(output.outputs[0].text),
                                                   annotations[row["example_id"]])["f1"])
                while len(row["reruns"]) <= repeat:
                    row["reruns"].append([None, None])
                row["reruns"][repeat][i] = value
                cost["target_calls"] += 1
                cost["prompt_tokens"] += len(target.tokenizer.encode(prompt))
                cost["generated_tokens"] += len(output.outputs[0].token_ids)
            print(json.dumps({"repeat": repeat, "completed": min(start + len(batch), len(all_inputs))}), flush=True)
    flipped = []
    for row in eligible:
        row["rerun_labels"] = []
        for stay, edit in row["reruns"]:
            levels = row["active"]
            breaks = any(stay + 1e-6 >= level and edit + 1e-6 < level for level in levels)
            repairs = any(stay + 1e-6 < level and edit + 1e-6 >= level for level in levels)
            saving = row["cached_tokens"][1] < row["cached_tokens"][0]
            row["rerun_labels"].append(1 if not breaks and (repairs or saving) else -1 if breaks else 0)
        if any(label != 1 for label in row["rerun_labels"]):
            flipped.append(row["example_id"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_pair.jsonl", [{k: v for k, v in row.items() if k not in {"question", "contexts"}} for row in eligible])
    result = {"protocol": cfg["protocol"], "queries": len(eligible), "paired_masks": 2 * len(eligible),
              "repeat_generations": cfg["repeat_generations"], "cached_positive_label_flip_queries": flipped,
              "queries_with_within_rerun_label_disagreement": [row["example_id"] for row in eligible if len(set(row["rerun_labels"])) > 1],
              "cost": cost, "limitations": ["Enriched rare-positive pairs, not a population label-flip estimate.",
                                         "Only selected STAY/edit masks are rerun; other actions retain original cache labels."]}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
