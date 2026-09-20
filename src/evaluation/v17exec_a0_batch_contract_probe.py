"""Compare batch-1 Target generation against prior batch-24 paired reruns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "prior-pairs", "data", "annotations", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["batch_size"] != 1 or cfg["repeats"] != 1:
        raise ValueError("unfrozen batch probe")
    prior = list(read_jsonl(args.prior_pairs))
    changed = [row for row in prior if any(value != 1 for value in row["rerun_labels"])]
    stable = [row for row in prior if all(value == 1 for value in row["rerun_labels"])][:4]
    selected = changed + stable
    if len(changed) != 4 or len(selected) != 8:
        raise AssertionError("unexpected pair selection")
    ids = {row["example_id"] for row in selected}
    data = {row["example_id"]: row for row in read_jsonl(args.data) if row["example_id"] in ids}
    annotations = {row["example_id"]: row["answer_atoms"] for row in read_jsonl(args.annotations) if row["example_id"] in ids}
    target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                 gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
    from vllm import SamplingParams
    params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
    cost = {"target_calls": 0, "prompt_tokens": 0, "generated_tokens": 0}
    rows = []
    for prior_row in selected:
        query = prior_row["example_id"]
        source = data[query]
        record = {"example_id": query, "masks": prior_row["masks"], "cached_f1": prior_row["cached_f1"],
                  "batch24_f1": prior_row["reruns"][0], "batch1": [], "active": prior_row["active"],
                  "cached_tokens": prior_row["cached_tokens"]}
        for mask in prior_row["masks"]:
            context = "\n\n".join(text.strip() for index, text in enumerate(source["packet_texts"]) if mask & 1 << index)
            prompt = target._chat_prompts([source["question"]], [context])[0]
            output = target.model.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            parsed = parse_list_prediction(output.text)
            f1 = float(qampari_list_metrics(parsed, annotations[query])["f1"])
            record["batch1"].append({"f1": f1, "parsed_answers": parsed,
                "generated_token_ids": list(output.token_ids),
                "prompt_token_ids_sha256": hashlib.sha256(json.dumps(target.tokenizer.encode(prompt), separators=(",", ":")).encode()).hexdigest(),
                "finish_reason": output.finish_reason})
            cost["target_calls"] += 1
            cost["prompt_tokens"] += len(target.tokenizer.encode(prompt))
            cost["generated_tokens"] += len(output.token_ids)
        values = [item["f1"] for item in record["batch1"]]
        active = prior_row["active"]
        broken = any(values[0] + 1e-6 >= level and values[1] + 1e-6 < level for level in active)
        repaired = any(values[0] + 1e-6 < level and values[1] + 1e-6 >= level for level in active)
        saving = prior_row["cached_tokens"][1] < prior_row["cached_tokens"][0]
        record["batch1_label"] = 1 if not broken and (repaired or saving) else -1 if broken else 0
        rows.append(record)
        print(json.dumps({"done": len(rows), "query": query, "batch1_label": record["batch1_label"]}), flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_pair.jsonl", rows)
    summary = {"protocol": cfg["protocol"], "queries": len(rows),
               "batch1_vs_batch24_f1_disagreement_pairs": [row["example_id"] for row in rows if any(abs(item["f1"] - prior) > 1e-9 for item, prior in zip(row["batch1"], row["batch24_f1"]))],
               "batch1_vs_batch24_label_disagreement_pairs": [row["example_id"] for row, old in zip(rows, selected) if row["batch1_label"] != old["rerun_labels"][0]],
               "batch1_vs_old_cache_label_disagreement_pairs": [row["example_id"] for row in rows if row["batch1_label"] != 1],
               "cost": cost,
               "interpretation": "Batch-1 versus batch-24 on the current Target is a controlled submission-batch test, not a replication of the old unknown full execution environment."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
