"""Zero-shot RECOMP NQ extractive compressor baseline on Screen-64."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.baselines.frontier_r1_hard_screen import BASE, LINEAGE, R0, render_selected


OUT = BASE / "frontier_r1_recomp_nq_extractive_screen64"
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def rows():
    manifest = json.loads((R0 / "manifest.json").read_text())
    ids = set(manifest["screen_query_ids"])
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    result = []
    for qid in manifest["screen_query_ids"]:
        packets = data[qid]["packet_texts"]
        units = []
        for packet_id, packet in enumerate(packets):
            for sentence_id, text in enumerate(x.strip() for x in SENTENCE.split(packet) if x.strip()):
                units.append({"packet_id": packet_id, "sentence_id": sentence_id, "text": text})
        result.append({"example_id": qid, "question": data[qid]["question"], "units": units,
                       "original_context": render_selected(packets, set(range(12)))})
    return result


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def compress(args, source_rows):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.compressor_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.compressor_model, local_files_only=True).to(args.device).eval()
    target_tok = AutoTokenizer.from_pretrained("models/Qwen3-8B", trust_remote_code=True, local_files_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "compressed.jsonl"
    done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(path)} if path.exists() else {}
    with path.open("a", encoding="utf-8") as f:
        for row in source_rows:
            texts = [row["question"]] + [u["text"] for u in row["units"]]
            begin = time.perf_counter()
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(args.device)
            with torch.no_grad():
                embeddings = mean_pooling(model(**encoded)[0], encoded["attention_mask"]).cpu()
            scores = [float(embeddings[0] @ embeddings[i + 1]) for i in range(len(row["units"]))]
            scoring_seconds = time.perf_counter() - begin
            source_tokens = len(target_tok.encode(row["original_context"], add_special_tokens=False))
            unit_tokens = [len(target_tok.encode(u["text"], add_special_tokens=False)) for u in row["units"]]
            for rate in (0.5, 0.25, 0.125):
                key = (row["example_id"], rate)
                if key in done:
                    continue
                budget = max(1, round(source_tokens * rate))
                chosen = []
                for index in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
                    trial = sorted(chosen + [index])
                    candidate = "\n\n".join(row["units"][i]["text"] for i in trial)
                    if len(target_tok.encode(candidate, add_special_tokens=False)) <= budget:
                        chosen.append(index)
                if not chosen:
                    chosen = [max(range(len(scores)), key=lambda i: scores[i])]
                chosen.sort()
                context = "\n\n".join(row["units"][i]["text"] for i in chosen)
                compressed_tokens = len(target_tok.encode(context, add_special_tokens=False))
                rec = {"example_id": row["example_id"], "question": row["question"], "keep_rate": rate,
                       "context": context, "source_context_tokens": source_tokens,
                       "compressed_context_tokens": compressed_tokens,
                       "actual_keep_rate": compressed_tokens / max(1, source_tokens),
                       "selected_unit_indices": chosen,
                       "selected_units": [row["units"][i] | {"score": scores[i], "tokens": unit_tokens[i]} for i in chosen],
                       "candidate_units": len(row["units"]), "compressor_seconds": scoring_seconds}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno()); done[key] = rec
                print(json.dumps({"compressed": len(done), "total": len(source_rows)*3}), flush=True)
    assert len(done) == len(source_rows)*3


def evaluate(args, source_rows):
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    compressed = list(read_jsonl(OUT / "compressed.jsonl")); ids = {r["example_id"] for r in source_rows}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids}
    baseline = {r["example_id"]: r for r in read_jsonl(BASE / "frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl")}
    output = OUT / "target_outputs.jsonl"
    done = {(r["example_id"], r["keep_rate"]): r for r in read_jsonl(output)} if output.exists() else {}
    pending = [r for r in compressed if (r["example_id"], r["keep_rate"]) not in done]
    if pending:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096)
        params = SamplingParams(temperature=0.0, max_tokens=256)
        with output.open("a", encoding="utf-8") as f:
            for start in range(0, len(pending), args.batch_size):
                batch=pending[start:start+args.batch_size]
                prompts=target._chat_prompts([r["question"] for r in batch],[r["context"] for r in batch])
                begin=time.perf_counter(); outputs=target.model.generate(prompts,params,use_tqdm=False); elapsed=time.perf_counter()-begin
                for source,prompt,generated in zip(batch,prompts,outputs):
                    pred=parse_list_prediction(generated.outputs[0].text)
                    rec={k:source[k] for k in ("example_id","keep_rate","source_context_tokens","compressed_context_tokens","actual_keep_rate","compressor_seconds")}
                    rec.update({"f1":float(qampari_list_metrics(pred,atoms[source["example_id"]])["f1"]),"prediction":pred,
                                "raw_output":generated.outputs[0].text,"prompt_tokens":len(target.tokenizer.encode(prompt)),
                                "generated_tokens":len(generated.outputs[0].token_ids),"target_batch_seconds_per_query":elapsed/len(batch)})
                    f.write(json.dumps(rec,ensure_ascii=False)+"\n");done[(rec["example_id"],rec["keep_rate"])]=rec
                f.flush();os.fsync(f.fileno());print(json.dumps({"evaluated":len(done),"total":192}),flush=True)
    assert len(done)==192
    thresholds=(.60,.70,.80,.90,.95);results=[]
    for rate in (.125,.25,.5):
        group=[r for (_,v),r in done.items() if v==rate];base=[baseline[r["example_id"]]["f1"] for r in group]
        results.append({"requested_keep_rate":rate,"queries":len(group),"baseline_mean_f1":sum(base)/len(base),
          "compressed_mean_f1":sum(r["f1"] for r in group)/len(group),
          "delta_mean_f1":sum(r["f1"]-baseline[r["example_id"]]["f1"] for r in group)/len(group),
          "baseline_success_090":sum(x+1e-6>=.9 for x in base),"compressed_success_090":sum(r["f1"]+1e-6>=.9 for r in group),
          "repairs_090":sum(baseline[r["example_id"]]["f1"]+1e-6<.9<=r["f1"]+1e-6 for r in group),
          "breaks_090":sum(baseline[r["example_id"]]["f1"]+1e-6>=.9>r["f1"]+1e-6 for r in group),
          "mean_actual_keep_rate":sum(r["actual_keep_rate"] for r in group)/len(group),
          "mean_context_tokens":sum(r["compressed_context_tokens"] for r in group)/len(group),
          "mean_compressor_seconds":sum(r["compressor_seconds"] for r in group)/len(group),
          "threshold_success":{f"{t:.2f}":{"baseline":sum(x+1e-6>=t for x in base),"compressed":sum(r["f1"]+1e-6>=t for r in group)} for t in thresholds}})
    eligible=[r for r in results if r["mean_actual_keep_rate"]<=.25];passed=any(r["delta_mean_f1"]>=-.02 and r["compressed_success_090"]>=r["baseline_success_090"]-2 for r in eligible)
    summary={"protocol":"FRONTIER-R1-RECOMP-NQ-EXTRACTIVE-SCREEN64","qualification":"official NQ checkpoint; zero-shot cross-domain QAMPARI baseline",
             "target":"frozen Qwen3-8B","source":"complete original 12-packet context","results":results,"screen_gate_passed":passed,
             "decision":"GO_RECOMP_EXTRACTIVE_DESIGN256" if passed else "STOP_RECOMP_EXTRACTIVE_AFTER_SCREEN64",
             "new_target_calls":len(done),"baseline_reused_calls":64,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("compress","evaluate","all"),default="all")
    p.add_argument("--compressor-model",default="models/recomp-nq-extractive");p.add_argument("--device",default="cuda")
    p.add_argument("--gpu-memory-utilization",type=float,default=.40);p.add_argument("--batch-size",type=int,default=64)
    args=p.parse_args();source_rows=rows()
    if args.mode in ("compress","all"):compress(args,source_rows)
    if args.mode in ("evaluate","all"):evaluate(args,source_rows)


if __name__=="__main__":main()
