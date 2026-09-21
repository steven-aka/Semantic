"""Screen-64 LongLLMLingua mechanism port using a frozen Qwen3-1.7B compressor."""
from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.frontier_r1_hard_screen import BASE, LINEAGE, R0, render_selected


OUT = BASE / "frontier_r1_longllmlingua_qwen17b_screen64"


def install_qwen3_legacy_cache_adapter(compressor):
    """Adapt LLMLingua 0.2.2's legacy list cache to Transformers DynamicCache."""
    import torch
    from transformers import DynamicCache

    def get_ppl(self, text, granularity="sentence", input_ids=None, attention_mask=None,
                past_key_values=None, return_kv=False, end=None, condition_mode="none",
                condition_pos_id=0):
        if input_ids is None:
            tokenized = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        model_cache = DynamicCache.from_legacy_cache(tuple(tuple(x) for x in past_key_values)) if past_key_values is not None else None
        with torch.no_grad():
            response = self.model(input_ids[:, past_length:end], attention_mask=attention_mask[:, :end],
                                  past_key_values=model_cache, use_cache=True)
        legacy = [[k, v] for k, v in response.past_key_values.to_legacy_cache()]
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1:end].contiguous()
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].reshape(-1)
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            shift_logits.reshape(-1, shift_logits.size(-1))[active], shift_labels.reshape(-1)[active])
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        result = loss.mean() if granularity == "sentence" else loss
        return (result, legacy) if return_kv else result

    compressor.get_ppl = types.MethodType(get_ppl, compressor)


def rows():
    manifest = json.loads((R0 / "manifest.json").read_text())
    ids = set(manifest["screen_query_ids"])
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    return [{"example_id": qid, "question": data[qid]["question"], "packets": data[qid]["packet_texts"],
             "original_context": render_selected(data[qid]["packet_texts"], set(range(12)))}
            for qid in manifest["screen_query_ids"]]


def compress(args, source_rows):
    from llmlingua import PromptCompressor
    from transformers import AutoTokenizer
    target_tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True, local_files_only=True)
    compressor = PromptCompressor(model_name=args.compressor_model, device_map=args.compressor_device, use_llmlingua2=False)
    install_qwen3_legacy_cache_adapter(compressor)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "compressed.jsonl"
    done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    with path.open("a", encoding="utf-8") as f:
        for row in source_rows:
            source_tokens = len(target_tok.encode(row["original_context"], add_special_tokens=False))
            for rate in (0.5, 0.25, 0.125):
                key = (row["example_id"], rate)
                if key in done:
                    continue
                begin = time.perf_counter()
                result = compressor.compress_prompt(
                    row["packets"], question=row["question"], rate=rate,
                    rank_method="longllmlingua", condition_in_question="after",
                    reorder_context="original", concate_question=False,
                    use_context_level_filter=True, use_sentence_level_filter=False,
                    use_token_level_filter=True,
                )
                context = result["compressed_prompt"].strip()
                compressed_tokens = len(target_tok.encode(context, add_special_tokens=False))
                record = {"example_id": row["example_id"], "question": row["question"], "keep_rate": rate,
                          "context": context, "source_context_tokens": source_tokens,
                          "compressed_context_tokens": compressed_tokens,
                          "actual_keep_rate": compressed_tokens / max(1, source_tokens),
                          "compressor_seconds": time.perf_counter() - begin,
                          "compressor_report": result}
                f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                done[key] = record
                print(json.dumps({"compressed": len(done), "total": len(source_rows) * 3,
                                  "last_seconds": record["compressor_seconds"]}), flush=True)
    assert len(done) == len(source_rows) * 3


def evaluate(args, source_rows):
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    compressed = list(read_jsonl(OUT / "compressed.jsonl"))
    ids = {r["example_id"] for r in source_rows}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids}
    baseline_path = BASE / "frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl"
    baseline = {r["example_id"]: r for r in read_jsonl(baseline_path)}
    assert len(baseline) == 64
    output = OUT / "target_outputs.jsonl"
    done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(output)} if output.exists() else {}
    pending = [r for r in compressed if (r["example_id"], r["keep_rate"]) not in done]
    if pending:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
        params = SamplingParams(temperature=0.0, max_tokens=256)
        with output.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                begin = time.perf_counter(); outputs = target.model.generate(prompts, params, use_tqdm=False); elapsed = time.perf_counter() - begin
                for source, prompt, generated in zip(batch, prompts, outputs):
                    pred = parse_list_prediction(generated.outputs[0].text)
                    rec = {k: source[k] for k in ("example_id", "keep_rate", "source_context_tokens", "compressed_context_tokens", "actual_keep_rate", "compressor_seconds")}
                    rec.update({"f1": float(qampari_list_metrics(pred, atoms[source["example_id"]])["f1"]),
                                "prediction": pred, "raw_output": generated.outputs[0].text,
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(generated.outputs[0].token_ids),
                                "target_batch_seconds_per_query": elapsed / len(batch)})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); done[(rec["example_id"], rec["keep_rate"])] = rec
                f.flush(); os.fsync(f.fileno()); print(json.dumps({"evaluated": len(done), "total": 192}), flush=True)
    assert len(done) == 192
    thresholds = (0.60, 0.70, 0.80, 0.90, 0.95)
    results = []
    for rate in (0.125, 0.25, 0.5):
        group = [r for (qid, value), r in done.items() if value == rate]
        base = [baseline[r["example_id"]]["f1"] for r in group]
        results.append({"requested_keep_rate": rate, "queries": len(group),
                        "baseline_mean_f1": sum(base)/len(base),
                        "compressed_mean_f1": sum(r["f1"] for r in group)/len(group),
                        "delta_mean_f1": sum(r["f1"]-baseline[r["example_id"]]["f1"] for r in group)/len(group),
                        "baseline_success_090": sum(x+1e-6 >= .9 for x in base),
                        "compressed_success_090": sum(r["f1"]+1e-6 >= .9 for r in group),
                        "repairs_090": sum(baseline[r["example_id"]]["f1"]+1e-6 < .9 <= r["f1"]+1e-6 for r in group),
                        "breaks_090": sum(baseline[r["example_id"]]["f1"]+1e-6 >= .9 > r["f1"]+1e-6 for r in group),
                        "mean_actual_keep_rate": sum(r["actual_keep_rate"] for r in group)/len(group),
                        "mean_context_tokens": sum(r["compressed_context_tokens"] for r in group)/len(group),
                        "mean_compressor_seconds": sum(r["compressor_seconds"] for r in group)/len(group),
                        "threshold_success": {f"{t:.2f}": {"baseline": sum(x+1e-6 >= t for x in base),
                                                                "compressed": sum(r["f1"]+1e-6 >= t for r in group)} for t in thresholds}})
    eligible = [r for r in results if r["mean_actual_keep_rate"] <= .25]
    passed = any(r["delta_mean_f1"] >= -.02 and r["compressed_success_090"] >= r["baseline_success_090"]-2 for r in eligible)
    summary = {"protocol": "FRONTIER-R1-LONGLMLINGUA-QWEN17B-PORT-SCREEN64",
               "qualification": "official LongLLMLingua algorithm; non-paper Qwen3-1.7B compressor backbone",
               "target": "frozen Qwen3-8B", "source": "complete original 12-packet context",
               "results": results, "screen_gate_passed": passed,
               "decision": "GO_LONGLMLINGUA_PORT_DESIGN256" if passed else "STOP_LONGLMLINGUA_PORT_AFTER_SCREEN64",
               "new_target_calls": len(done), "baseline_reused_calls": 64, "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("compress", "evaluate", "all"), default="all")
    p.add_argument("--compressor-model", default="models/Qwen3-1.7B")
    p.add_argument("--compressor-device", default="cuda")
    p.add_argument("--gpu-memory-utilization", type=float, default=.40)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(); source_rows = rows()
    if args.mode in ("compress", "all"): compress(args, source_rows)
    if args.mode in ("evaluate", "all"): evaluate(args, source_rows)


if __name__ == "__main__":
    main()
