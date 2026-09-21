"""Training-free LLMLingua-2 adaptation screen at mild compression rates."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.frontier_r1_hard_screen import BASE, prepare_rows


OUT = BASE / "frontier_r1_llmlingua2_adapted_screen64"
RATES = (0.6, 0.7, 0.8, 0.9, 0.95)
ARMS = ("v8_global", "original_global")


def compress(args, rows):
    from llmlingua import PromptCompressor
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True, local_files_only=True)
    compressor = PromptCompressor(model_name=args.compressor_model, device_map="cuda", use_llmlingua2=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "compressed.jsonl"
    done = {(r["example_id"], r["arm"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            for arm in ARMS:
                source = row["baseline_context"] if arm == "v8_global" else row["original_context"]
                source_tokens = len(tok.encode(source, add_special_tokens=False))
                for rate in RATES:
                    key = (row["example_id"], arm, rate)
                    if key in done:
                        continue
                    begin = time.perf_counter()
                    result = compressor.compress_prompt_llmlingua2(
                        [source], rate=rate, use_context_level_filter=False, use_token_level_filter=True
                    )
                    context = result["compressed_prompt"].strip()
                    context_tokens = len(tok.encode(context, add_special_tokens=False))
                    rec = {"example_id": row["example_id"], "question": row["question"], "arm": arm,
                           "keep_rate": rate, "context": context, "source_context_tokens": source_tokens,
                           "compressed_context_tokens": context_tokens,
                           "actual_keep_rate": context_tokens / max(1, source_tokens),
                           "compressor_seconds": time.perf_counter() - begin}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                    done[key] = rec
                    print(json.dumps({"compressed": len(done), "total": len(rows) * len(ARMS) * len(RATES)}), flush=True)
    assert len(done) == len(rows) * len(ARMS) * len(RATES)


def evaluate(args, rows):
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams

    compressed = list(read_jsonl(OUT / "compressed.jsonl"))
    ids = {r["example_id"] for r in rows}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids}
    old = BASE / "frontier_r1_hard_screen64"
    refs = {
        "v8_global": {r["example_id"]: r for r in read_jsonl(old / "fresh_baseline_outputs.jsonl")},
        "original_global": {r["example_id"]: r for r in read_jsonl(old / "fresh_original_baseline_outputs.jsonl")},
    }
    path = OUT / "target_outputs.jsonl"
    done = {(r["example_id"], r["arm"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [r for r in compressed if (r["example_id"], r["arm"], r["keep_rate"]) not in done]
    if pending:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
        params = SamplingParams(temperature=0, max_tokens=256)
        with path.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                for src, prompt, generated in zip(batch, prompts, outputs):
                    prediction = parse_list_prediction(generated.outputs[0].text)
                    rec = {k: src[k] for k in ("example_id", "arm", "keep_rate", "source_context_tokens",
                                                "compressed_context_tokens", "actual_keep_rate", "compressor_seconds")}
                    rec.update({"f1": float(qampari_list_metrics(prediction, atoms[src["example_id"]])["f1"]),
                                "prediction": prediction, "raw_output": generated.outputs[0].text,
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(generated.outputs[0].token_ids),
                                "target_seconds": elapsed / len(batch)})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); done[(rec["example_id"], rec["arm"], rec["keep_rate"])] = rec
                f.flush(); os.fsync(f.fileno())
                print(json.dumps({"evaluated": len(done), "total": len(compressed)}), flush=True)
    results = []
    for arm in ARMS:
        ref = refs[arm]
        for rate in RATES:
            group = [r for (_, a, k), r in done.items() if a == arm and k == rate]
            thresholds = {f"{t:.2f}": {"baseline": sum(ref[r["example_id"]]["f1"] + 1e-6 >= t for r in group),
                                             "compressed": sum(r["f1"] + 1e-6 >= t for r in group)}
                          for t in (0.60, 0.70, 0.80, 0.90, 0.95)}
            results.append({"arm": arm, "requested_keep_rate": rate,
                            "baseline_mean_f1": sum(ref[r["example_id"]]["f1"] for r in group) / len(group),
                            "mean_source_context_tokens": sum(r["source_context_tokens"] for r in group) / len(group),
                            "mean_actual_keep_rate": sum(r["actual_keep_rate"] for r in group) / len(group),
                            "mean_context_tokens": sum(r["compressed_context_tokens"] for r in group) / len(group),
                            "mean_f1": sum(r["f1"] for r in group) / len(group),
                            "delta_f1": sum(r["f1"] - ref[r["example_id"]]["f1"] for r in group) / len(group),
                            "success_090": sum(r["f1"] + 1e-6 >= .9 for r in group),
                            "repairs_090": sum(ref[r["example_id"]]["f1"] + 1e-6 < .9 <= r["f1"] + 1e-6 for r in group),
                            "breaks_090": sum(ref[r["example_id"]]["f1"] + 1e-6 >= .9 > r["f1"] + 1e-6 for r in group),
                            "threshold_success": thresholds})
    summary = {"protocol": "FRONTIER-R1-LLMLINGUA2-ADAPTED-SCREEN64", "target": "frozen Qwen3-8B",
               "adaptation": "no training; V8 coarse selection plus global compression; mild keep-rate sweep",
               "results": results, "new_target_calls": len(done), "sealed_sets_read": False,
               "decision": "RETAIN_V8_GLOBAL_RATE095_AS_STRONG_TRAINING_FREE_BASELINE"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("compress", "evaluate", "all"), default="all")
    p.add_argument("--compressor-model", default="models/llmlingua-2-xlm-roberta-large-meetingbank")
    p.add_argument("--gpu-memory-utilization", type=float, default=.4)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(); rows = prepare_rows()
    if args.mode in ("compress", "all"): compress(args, rows)
    if args.mode in ("evaluate", "all"): evaluate(args, rows)


if __name__ == "__main__":
    main()
