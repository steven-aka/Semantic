"""Evaluate the trained QAMPARI RECOMP-style extractor with frozen Qwen3-8B."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.baselines.frontier_r1_recomp_extractive import mean_pooling, rows
from src.data.schemas import read_jsonl


BASE = Path("results/v2_rank_then_cut")
OUT = BASE / "baseline_recomp_qampari_extractive"
RATES = (0.25, 0.5, 0.75, 0.9, 0.95)


def compress(args, source_rows):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True).to(args.device).eval()
    target_tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True, local_files_only=True)
    path = OUT / "compressed.jsonl"
    done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    with path.open("a", encoding="utf-8") as f:
        for row in source_rows:
            texts = [row["question"]] + [u["text"] for u in row["units"]]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(args.device)
            begin = time.perf_counter()
            with torch.no_grad(): embeddings = torch.nn.functional.normalize(mean_pooling(model(**encoded)[0], encoded["attention_mask"]), dim=-1).cpu()
            scores = [float(embeddings[0] @ embeddings[i + 1]) for i in range(len(row["units"]))]
            source_tokens = len(target_tok.encode(row["original_context"], add_special_tokens=False))
            for rate in RATES:
                key = (row["example_id"], rate)
                if key in done: continue
                budget = round(source_tokens * rate); chosen = []
                for index in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
                    trial = sorted(chosen + [index]); context = "\n\n".join(row["units"][i]["text"] for i in trial)
                    if len(target_tok.encode(context, add_special_tokens=False)) <= budget: chosen.append(index)
                chosen = sorted(chosen or [max(range(len(scores)), key=scores.__getitem__)])
                context = "\n\n".join(row["units"][i]["text"] for i in chosen)
                tokens = len(target_tok.encode(context, add_special_tokens=False))
                rec = {"example_id": row["example_id"], "question": row["question"], "keep_rate": rate,
                       "context": context, "source_context_tokens": source_tokens, "compressed_context_tokens": tokens,
                       "actual_keep_rate": tokens / source_tokens, "selected_unit_indices": chosen,
                       "compressor_seconds": (time.perf_counter() - begin)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno()); done[key] = rec
    assert len(done) == len(source_rows) * len(RATES)


def evaluate(args, source_rows):
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    compressed = list(read_jsonl(OUT / "compressed.jsonl")); ids = {r["example_id"] for r in source_rows}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids}
    baseline = {r["example_id"]: r for r in read_jsonl(BASE / "frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl")}
    path = OUT / "target_outputs.jsonl"; done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [r for r in compressed if (r["example_id"], r["keep_rate"]) not in done]
    if pending:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
        params = SamplingParams(temperature=0, max_tokens=256)
        with path.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), 64):
                batch = pending[start:start + 64]; prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - begin
                for src, prompt, generated in zip(batch, prompts, outputs):
                    prediction = parse_list_prediction(generated.outputs[0].text)
                    rec = {k: src[k] for k in ("example_id", "keep_rate", "source_context_tokens", "compressed_context_tokens", "actual_keep_rate", "compressor_seconds")}
                    rec.update({"f1": float(qampari_list_metrics(prediction, atoms[src["example_id"]])["f1"]), "prediction": prediction,
                                "raw_output": generated.outputs[0].text, "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(generated.outputs[0].token_ids), "target_seconds": elapsed / len(batch)})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); done[(rec["example_id"], rec["keep_rate"])] = rec
                f.flush(); os.fsync(f.fileno()); print(json.dumps({"evaluated": len(done), "total": len(compressed)}), flush=True)
    results = []
    for rate in RATES:
        group = [r for (_, k), r in done.items() if k == rate]
        results.append({"requested_keep_rate": rate, "mean_actual_keep_rate": sum(r["actual_keep_rate"] for r in group) / len(group),
                        "mean_context_tokens": sum(r["compressed_context_tokens"] for r in group) / len(group),
                        "mean_f1": sum(r["f1"] for r in group) / len(group),
                        "delta_f1": sum(r["f1"] - baseline[r["example_id"]]["f1"] for r in group) / len(group),
                        "success_090": sum(r["f1"] + 1e-6 >= .9 for r in group),
                        "repairs_090": sum(baseline[r["example_id"]]["f1"] + 1e-6 < .9 <= r["f1"] + 1e-6 for r in group),
                        "breaks_090": sum(baseline[r["example_id"]]["f1"] + 1e-6 >= .9 > r["f1"] + 1e-6 for r in group)})
    summary = {"protocol": "BASELINE-RECOMP-QAMPARI-EXTRACTIVE-V1", "qualification": "RECOMP-style in-domain supervised adaptation",
               "target": "frozen Qwen3-8B", "baseline_mean_f1": sum(r["f1"] for r in baseline.values()) / 64,
               "baseline_success_090": sum(r["f1"] + 1e-6 >= .9 for r in baseline.values()), "results": results,
               "new_target_calls": len(done), "screen_queries_excluded_from_training": True, "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=("compress", "evaluate", "all"), default="all")
    p.add_argument("--checkpoint", default="checkpoints/baselines/recomp_qampari_extractive"); p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-memory-utilization", type=float, default=.4); args = p.parse_args(); source = rows()
    if args.mode in ("compress", "all"): compress(args, source)
    if args.mode in ("evaluate", "all"): evaluate(args, source)


if __name__ == "__main__": main()
