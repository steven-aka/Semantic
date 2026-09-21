"""Collect same-call Qwen3-8B answers, labels and native confidence traces."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
PREFLIGHT = Path("results/v2_rank_then_cut/v17stop_c2a_native_trace_preflight/paired.jsonl")
OUT = Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence")


def key(salt: str, value: str) -> str:
    return hashlib.sha256((salt + value).encode()).hexdigest()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def main() -> None:
    from vllm import SamplingParams

    cfg = json.loads(Path("configs/v17stop_c1_obs0_native_confidence.json").read_text())
    assert cfg["status"] == "FROZEN_TRACE_DESIGN_COHORT"
    preflight_ids = {r["example_id"] for r in read_jsonl(PREFLIGHT)}
    population = {r["example_id"] for r in read_jsonl(P0 / "per_query.jsonl")}
    chosen = sorted(population - preflight_ids, key=lambda q: key(cfg["selection_salt"], q))[: cfg["sample_size"]]
    assert len(chosen) == cfg["sample_size"] and not (set(chosen) & preflight_ids)
    ids = set(chosen)
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    states = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in ids and row["depth"] in cfg["depths"]:
                states[(row["example_id"], row["depth"])] = row
    assert len(data) == len(atoms) == len(ids)
    assert len(states) == len(ids) * len(cfg["depths"])

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "traces.jsonl"
    done = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[(row["example_id"], row["depth"])] = row
    tasks = []
    for qid in chosen:
        for depth in cfg["depths"]:
            if (qid, depth) in done:
                continue
            state = states[(qid, depth)]
            context = "\n\n".join(
                text.strip() for i, text in enumerate(data[qid]["packet_texts"])
                if state["mask"] & (1 << i)
            )
            tasks.append({"example_id": qid, "depth": depth, "mask": state["mask"],
                          "question": data[qid]["question"], "context": context})
    print(json.dumps({"selected_queries": len(ids), "completed": len(done), "pending_calls": len(tasks)}), flush=True)
    if tasks:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"],
                                logprobs=cfg["logprobs_top_k"])
        with out_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([x["question"] for x in batch], [x["context"] for x in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, request in zip(batch, prompts, outputs):
                    output = request.outputs[0]
                    parsed = parse_list_prediction(output.text)
                    token_logs = []
                    margins = []
                    entropies = []
                    for token_id, top in zip(output.token_ids, output.logprobs or []):
                        chosen_lp = float(top[token_id].logprob)
                        token_logs.append(chosen_lp)
                        vals = sorted((float(x.logprob) for x in top.values()), reverse=True)
                        margins.append(vals[0] - vals[1] if len(vals) > 1 else float("nan"))
                        probs = [math.exp(x) for x in vals]
                        entropies.append(-sum(p * math.log(max(p, 1e-30)) for p in probs))
                    if len(token_logs) != len(output.token_ids):
                        raise AssertionError("missing chosen-token logprob")
                    record = {
                        "example_id": task["example_id"], "depth": task["depth"], "mask": task["mask"],
                        "prediction": " # ".join(parsed),
                        "answer_count": len(parsed),
                        "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                        "context_tokens": count_tokens(target.tokenizer, task["context"]),
                        "prompt_tokens": len(target.tokenizer.encode(prompt)),
                        "generated_tokens": len(output.token_ids),
                        "finish_reason": output.finish_reason,
                        "token_ids": list(output.token_ids),
                        "chosen_token_logprobs": token_logs,
                        "mean_chosen_logprob": sum(token_logs) / len(token_logs) if token_logs else float("nan"),
                        "p10_chosen_logprob": percentile(token_logs, 0.10),
                        "min_chosen_logprob": min(token_logs) if token_logs else float("nan"),
                        "mean_top2_margin": sum(margins) / len(margins) if margins else float("nan"),
                        "mean_top2_partial_entropy": sum(entropies) / len(entropies) if entropies else float("nan"),
                        "source": "C1 unified traced execution path"
                    }
                    handle.write(json.dumps(record) + "\n")
                    done[(record["example_id"], record["depth"])] = record
                handle.flush()
                os.fsync(handle.fileno())
                (OUT / "progress.json").write_text(json.dumps({"completed": len(done), "total": len(ids) * len(cfg["depths"])}, indent=2) + "\n")
                print(json.dumps({"completed": len(done), "total": len(ids) * len(cfg["depths"])}), flush=True)
    assert len(done) == len(ids) * len(cfg["depths"])
    manifest = {
        "protocol": cfg["protocol"], "queries": len(ids), "depths": cfg["depths"],
        "trace_states": len(done), "target_calls": len(done),
        "selection_role": "natural deterministic design cohort; queries previously design-exposed, traces newly collected",
        "excluded_preflight_queries": len(preflight_ids), "sealed_sets_read": False
    }
    (OUT / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
