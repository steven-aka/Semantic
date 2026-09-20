"""Replay one exact-search 512-state submission batch for a disputed pair."""
from __future__ import annotations

import argparse
import json
from itertools import islice, product
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "data", "annotations", "prior-pairs", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["batch_size"] != 512:
        raise ValueError("unfrozen exact-batch probe")
    query = cfg["example_id"]
    source = next(row for row in read_jsonl(args.data) if row["example_id"] == query)
    atoms = next(row["answer_atoms"] for row in read_jsonl(args.annotations) if row["example_id"] == query)
    old = next(row for row in read_jsonl(args.prior_pairs) if row["example_id"] == query)
    states = list(islice(product(range(2), repeat=12), cfg["original_product_order_batch_index"] * 512,
                         (cfg["original_product_order_batch_index"] + 1) * 512))
    masks = [sum(value << index for index, value in enumerate(state)) for state in states]
    if len(states) != 512 or not set(cfg["selected_masks"]) <= set(masks):
        raise AssertionError("selected masks not in exact batch")
    contexts = ["\n\n".join(text.strip() for text, bit in zip(source["packet_texts"], state) if bit)
                for state in states]
    target = QampariTargetRunner(cfg["model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                 gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
    from vllm import SamplingParams
    prompts = target._chat_prompts([source["question"]] * 512, contexts)
    outputs = target.model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False)
    chosen = []
    for mask in cfg["selected_masks"]:
        index = masks.index(mask)
        output = outputs[index].outputs[0]
        parsed = parse_list_prediction(output.text)
        f1 = float(qampari_list_metrics(parsed, atoms)["f1"])
        chosen.append({"mask": mask, "f1": f1, "parsed_answers": parsed,
                       "generated_token_ids": list(output.token_ids), "finish_reason": output.finish_reason,
                       "prompt_tokens": len(target.tokenizer.encode(prompts[index]))})
    result = {"protocol": cfg["protocol"], "example_id": query, "submitted_states": 512,
              "old_cache_f1": old["cached_f1"], "batch24_f1": old["reruns"][0], "exact_batch_f1": [row["f1"] for row in chosen],
              "selected": chosen, "cost": {"target_calls": 512,
                   "prompt_tokens": sum(len(target.tokenizer.encode(prompt)) for prompt in prompts),
                   "generated_tokens": sum(len(output.outputs[0].token_ids) for output in outputs)},
              "limitation": "One query and one exact-search submission batch; reproducing old batch geometry does not recover an unrecorded old software/model-file environment."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: value for k, value in result.items() if k != "selected"}), flush=True)


if __name__ == "__main__":
    main()
