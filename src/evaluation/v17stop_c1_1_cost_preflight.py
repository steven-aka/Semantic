"""Measure real serialized input length and small-encoder batch-1 cost."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data.schemas import read_jsonl

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT = Path("results/v2_rank_then_cut/v17stop_c1_1_cost_preflight")


def percentile(xs: list[float]) -> dict[str, float]:
    return {str(p): float(np.percentile(xs, p)) for p in (0, 50, 90, 95, 99, 100)}


def main() -> None:
    cfg = json.loads(Path("configs/v17stop_c1_1_cost_preflight.json").read_text())
    assert cfg["status"] == "FROZEN_NO_TRAINING_TRAIN_SIDE"
    rows = list(read_jsonl(C0 / "per_query.jsonl"))
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")}
    prefixes = {}
    for path in Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain").glob("shard_*_of_3/per_prefix.jsonl"):
        for r in read_jsonl(path):
            if r["depth"] == 9 and r["example_id"] in data:
                prefixes[r["example_id"]] = r
    assert len(rows) == 1421
    texts = []
    for r in rows:
        q = r["example_id"]
        d = data[q]
        answer = prefixes[q]["prediction"]
        packet = d["packet_texts"][r["added_packet"]]
        texts.append(f"[QUERY] {d['question']} [TARGET] 0.90 [ANSWER9] {answer} [NEW_PACKET] {packet}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_repo"], use_fast=True)
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    lengths = [len(x) for x in encoded["input_ids"]]
    packet_lengths = [len(tokenizer.encode(data[r["example_id"]]["packet_texts"][r["added_packet"]],
                                           add_special_tokens=False)) for r in rows]
    result = {"protocol": cfg["protocol"], "population": len(rows), "encoder_repo": cfg["encoder_repo"],
              "max_tokens": cfg["maximum_tokens"], "full_input_length_percentiles": percentile(lengths),
              "new_packet_length_percentiles": percentile(packet_lengths),
              "fraction_full_input_over_512": float(np.mean(np.asarray(lengths) > 512)),
              "mean_overflow_tokens": float(np.mean(np.maximum(np.asarray(lengths) - 512, 0))),
              "new_target_calls": 0, "sealed_outcome_sets_read": False}
    if not torch.cuda.is_available():
        result["gpu_benchmark"] = "unavailable"
    else:
        device = torch.device(f"cuda:{cfg['gpu']}")
        model = AutoModel.from_pretrained(cfg["encoder_repo"], torch_dtype=torch.float16).to(device).eval()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        selected = np.linspace(0, len(texts) - 1, cfg["benchmark_examples"], dtype=int)
        times, tokenizer_times = [], []
        peak = 0
        with torch.inference_mode():
            for i, idx in enumerate(selected):
                start = time.perf_counter()
                inputs = tokenizer(texts[idx], return_tensors="pt", truncation=True,
                                   max_length=cfg["maximum_tokens"]).to(device)
                tokenizer_times.append((time.perf_counter() - start) * 1000)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                _ = model(**inputs).last_hidden_state[:, 0, :]
                torch.cuda.synchronize(device)
                if i >= cfg["warmup_examples"]:
                    times.append((time.perf_counter() - start) * 1000)
                    peak = max(peak, torch.cuda.max_memory_allocated(device))
        result["gpu_benchmark"] = {"gpu": torch.cuda.get_device_name(device),
                                   "precision": cfg["precision"], "batch_size": 1,
                                   "forward_ms_percentiles": percentile(times),
                                   "tokenization_and_transfer_ms_percentiles": percentile(tokenizer_times),
                                   "peak_allocated_mib": peak / 2**20,
                                   "sampled_examples": len(selected), "warmup_examples": cfg["warmup_examples"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
