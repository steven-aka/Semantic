"""R1 hard-compression Screen-64 on frozen CANON-P0 depth-10 contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from src.data.schemas import read_jsonl


BASE = Path("results/v2_rank_then_cut")
R0 = BASE / "frontier_compression_r0"
OUT = BASE / "frontier_r1_hard_screen64"
LINEAGE = BASE / "v17sel_b2b_lineage_clean_holdout"


def render_selected(packets, selected, replacements=None):
    replacements = replacements or {}
    return "\n\n".join(replacements.get(i, packets[i]).strip() for i in range(len(packets)) if i in selected)


def prepare_rows():
    manifest = json.loads((R0 / "manifest.json").read_text())
    ids = set(manifest["screen_query_ids"])
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == 64
    rows = []
    for qid in manifest["screen_query_ids"]:
        order = orders[qid]
        assert sorted(order) == list(range(12))
        selected = set(order[:10])
        row = data[qid]
        original_context = render_selected(row["packet_texts"], set(range(len(row["packet_texts"]))))
        rows.append({"example_id": qid, "question": row["question"], "packets": row["packet_texts"],
                     "selected": sorted(selected), "baseline_context": render_selected(row["packet_texts"], selected),
                     "original_context": original_context})
    return rows


def compress(args, rows):
    from llmlingua import PromptCompressor
    from transformers import AutoTokenizer
    target_tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True, local_files_only=True)
    compressor = PromptCompressor(model_name=args.compressor_model, device_map=args.compressor_device, use_llmlingua2=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "compressed.jsonl"
    done = {}
    if output.exists():
        done = {(r["example_id"], r["variant"], r["keep_rate"]): r for r in read_jsonl(output)}
    with output.open("a", encoding="utf-8") as f:
        for row in rows:
            baseline_tokens = len(target_tok.encode(row["baseline_context"], add_special_tokens=False))
            for variant in ("native", "v8_wrapped", "original_native"):
                for rate in (0.5, 0.25, 0.125):
                    key = (row["example_id"], variant, rate)
                    if key in done:
                        continue
                    start = time.perf_counter()
                    if variant in ("native", "original_native"):
                        source_context = row["baseline_context"] if variant == "native" else row["original_context"]
                        result = compressor.compress_prompt_llmlingua2([source_context], rate=rate,
                            use_context_level_filter=False, use_token_level_filter=True)
                        context = result["compressed_prompt"].strip()
                    else:
                        replacements = {}
                        for packet_id in row["selected"]:
                            result = compressor.compress_prompt_llmlingua2([row["packets"][packet_id]], rate=rate,
                                use_context_level_filter=False, use_token_level_filter=True)
                            replacements[packet_id] = result["compressed_prompt"].strip()
                        context = render_selected(row["packets"], set(row["selected"]), replacements)
                    source_context = row["original_context"] if variant == "original_native" else row["baseline_context"]
                    baseline_tokens = len(target_tok.encode(source_context, add_special_tokens=False))
                    record = {"example_id": row["example_id"], "variant": variant, "keep_rate": rate,
                              "question": row["question"], "context": context,
                              "baseline_context_tokens": baseline_tokens,
                              "compressed_context_tokens": len(target_tok.encode(context, add_special_tokens=False)),
                              "compressor_seconds": time.perf_counter() - start}
                    record["actual_keep_rate"] = record["compressed_context_tokens"] / max(1, baseline_tokens)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                    done[key] = record
                    print(json.dumps({"compressed": len(done), "total": len(rows) * 9}), flush=True)
    assert len(done) == len(rows) * 9


def evaluate(args, rows):
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    compressed = list(read_jsonl(OUT / "compressed.jsonl"))
    screen_ids = {r["example_id"] for r in rows}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in screen_ids}
    historical_baseline = {}
    for path in sorted((BASE / "v17canon_p0_fresh_v8_prefix_chain").glob("shard_*_of_*/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in screen_ids and r["depth"] == 10:
                historical_baseline[r["example_id"]] = r
    assert len(atoms) == len(historical_baseline) == 64
    output = OUT / "target_outputs.jsonl"
    done = {(r["example_id"], r["variant"], r["keep_rate"]): r for r in read_jsonl(output)} if output.exists() else {}
    pending = [r for r in compressed if (r["example_id"], r["variant"], r["keep_rate"]) not in done]
    baseline_output = OUT / "fresh_baseline_outputs.jsonl"
    fresh_baseline = {r["example_id"]: r for r in read_jsonl(baseline_output)} if baseline_output.exists() else {}
    original_baseline_output = OUT / "fresh_original_baseline_outputs.jsonl"
    original_baseline = {r["example_id"]: r for r in read_jsonl(original_baseline_output)} if original_baseline_output.exists() else {}
    if pending or len(fresh_baseline) != 64 or len(original_baseline) != 64:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
        params = SamplingParams(temperature=0.0, max_tokens=256)
        baseline_pending = [r for r in rows if r["example_id"] not in fresh_baseline]
        if baseline_pending:
            with baseline_output.open("a", encoding="utf-8") as f:
                for start in range(0, len(baseline_pending), args.batch_size):
                    batch = baseline_pending[start:start + args.batch_size]
                    prompts = target._chat_prompts([r["question"] for r in batch], [r["baseline_context"] for r in batch])
                    begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - begin
                    for source, prompt, generated in zip(batch, prompts, outputs):
                        pred = parse_list_prediction(generated.outputs[0].text)
                        rec = {"example_id": source["example_id"],
                               "f1": float(qampari_list_metrics(pred, atoms[source["example_id"]])["f1"]),
                               "prediction": pred, "raw_output": generated.outputs[0].text,
                               "prompt_tokens": len(target.tokenizer.encode(prompt)),
                               "generated_tokens": len(generated.outputs[0].token_ids),
                               "target_batch_seconds_per_query": elapsed / len(batch)}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); fresh_baseline[rec["example_id"]] = rec
                    f.flush(); os.fsync(f.fileno())
                    print(json.dumps({"fresh_baseline": len(fresh_baseline), "total": 64}), flush=True)
        original_pending = [r for r in rows if r["example_id"] not in original_baseline]
        if original_pending:
            with original_baseline_output.open("a", encoding="utf-8") as f:
                for start in range(0, len(original_pending), args.batch_size):
                    batch = original_pending[start:start + args.batch_size]
                    prompts = target._chat_prompts([r["question"] for r in batch], [r["original_context"] for r in batch])
                    begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - begin
                    for source, prompt, generated in zip(batch, prompts, outputs):
                        pred = parse_list_prediction(generated.outputs[0].text)
                        rec = {"example_id": source["example_id"],
                               "f1": float(qampari_list_metrics(pred, atoms[source["example_id"]])["f1"]),
                               "prediction": pred, "raw_output": generated.outputs[0].text,
                               "prompt_tokens": len(target.tokenizer.encode(prompt)),
                               "generated_tokens": len(generated.outputs[0].token_ids),
                               "target_batch_seconds_per_query": elapsed / len(batch)}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); original_baseline[rec["example_id"]] = rec
                    f.flush(); os.fsync(f.fileno())
                    print(json.dumps({"fresh_original_baseline": len(original_baseline), "total": 64}), flush=True)
        with output.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - begin
                for source, prompt, generated in zip(batch, prompts, outputs):
                    pred = parse_list_prediction(generated.outputs[0].text)
                    rec = {k: source[k] for k in ("example_id", "variant", "keep_rate", "baseline_context_tokens", "compressed_context_tokens", "actual_keep_rate", "compressor_seconds")}
                    rec.update({"f1": float(qampari_list_metrics(pred, atoms[source["example_id"]])["f1"]),
                                "prediction": pred, "raw_output": generated.outputs[0].text,
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(generated.outputs[0].token_ids),
                                "target_batch_seconds_per_query": elapsed / len(batch)})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); done[(rec["example_id"], rec["variant"], rec["keep_rate"])] = rec
                f.flush(); os.fsync(f.fileno())
                print(json.dumps({"evaluated": len(done), "total": len(compressed)}), flush=True)
    assert len(done) == 576
    assert len(fresh_baseline) == 64
    assert len(original_baseline) == 64
    groups = {}
    for (qid, variant, rate), rec in done.items():
        groups.setdefault((variant, rate), []).append(rec)
    results = []
    thresholds = (0.60, 0.70, 0.80, 0.90, 0.95)
    for (variant, rate), group in sorted(groups.items()):
        reference = original_baseline if variant == "original_native" else fresh_baseline
        base = [reference[r["example_id"]]["f1"] for r in group]
        threshold_success = {
            f"{threshold:.2f}": {
                "baseline": sum(x + 1e-6 >= threshold for x in base),
                "compressed": sum(r["f1"] + 1e-6 >= threshold for r in group),
            }
            for threshold in thresholds
        }
        results.append({"variant": variant, "requested_keep_rate": rate, "queries": len(group),
                        "baseline_mean_f1": sum(base)/len(base), "compressed_mean_f1": sum(r["f1"] for r in group)/len(group),
                        "delta_mean_f1": sum(r["f1"]-fresh_baseline[r["example_id"]]["f1"] for r in group)/len(group),
                        "baseline_success_090": sum(x+1e-6 >= .9 for x in base),
                        "compressed_success_090": sum(r["f1"]+1e-6 >= .9 for r in group),
                        "repairs_090": sum(reference[r["example_id"]]["f1"]+1e-6 < .9 <= r["f1"]+1e-6 for r in group),
                        "breaks_090": sum(reference[r["example_id"]]["f1"]+1e-6 >= .9 > r["f1"]+1e-6 for r in group),
                        "mean_actual_keep_rate": sum(r["actual_keep_rate"] for r in group)/len(group),
                        "mean_context_tokens": sum(r["compressed_context_tokens"] for r in group)/len(group),
                        "mean_compressor_seconds": sum(r["compressor_seconds"] for r in group)/len(group),
                        "mean_target_seconds": sum(r["target_batch_seconds_per_query"] for r in group)/len(group),
                        "threshold_success": threshold_success})
    qualifying = [r for r in results if r["mean_actual_keep_rate"] <= .25 + 1e-9]
    passes_quality_gate = any(
        r["delta_mean_f1"] >= -.02
        and all(r["threshold_success"][key]["compressed"] >= r["threshold_success"][key]["baseline"] - 2
                for key in ("0.90", "0.95"))
        for r in qualifying
    )
    summary = {"protocol": "FRONTIER-R1-LLMLINGUA2-SCREEN64", "target": "frozen Qwen3-8B",
               "baseline": "fresh same-run-contract V8 depth10", "historical_baseline_mean_f1": sum(r["f1"] for r in historical_baseline.values())/64,
               "fresh_baseline_mean_f1": sum(r["f1"] for r in fresh_baseline.values())/64,
               "fresh_original_baseline_mean_f1": sum(r["f1"] for r in original_baseline.values())/64,
               "variant_semantics": {"native": "global LLMLingua-2 over V8 depth-10 context (legacy name)",
                                     "v8_wrapped": "packetwise LLMLingua-2 over V8 depth-10 selected packets",
                                     "original_native": "global LLMLingua-2 over the complete original 12-packet context"},
               "results": results, "target_calls": len(done) + len(fresh_baseline) + len(original_baseline),
               "screen_gate": {"requires_actual_keep_rate_lte": 0.25,
                               "max_mean_f1_drop": 0.02,
                               "max_success_drop_queries": 2,
                               "passes_quality_gate": passes_quality_gate},
               "decision": "GO_LLMLINGUA2_DESIGN256" if passes_quality_gate else "STOP_LLMLINGUA2_AFTER_SCREEN64",
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("compress", "evaluate", "all"), default="all")
    p.add_argument("--compressor-model", default="models/llmlingua-2-xlm-roberta-large-meetingbank")
    p.add_argument("--compressor-device", default="cuda")
    p.add_argument("--gpu-memory-utilization", type=float, default=.45)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    rows = prepare_rows()
    if args.mode in ("compress", "all"): compress(args, rows)
    if args.mode in ("evaluate", "all"): evaluate(args, rows)


if __name__ == "__main__": main()
