"""Zero-shot RECOMP NQ abstractive compressor baseline on Screen-64."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.frontier_r1_hard_screen import BASE, LINEAGE, R0, render_selected


OUT = BASE / "frontier_r1_recomp_nq_abstractive_screen64"


def rows():
    manifest=json.loads((R0/"manifest.json").read_text());ids=set(manifest["screen_query_ids"])
    data={r["example_id"]:r for r in read_jsonl(LINEAGE/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    return [{"example_id":q,"question":data[q]["question"],
             "original_context":render_selected(data[q]["packet_texts"],set(range(12)))} for q in manifest["screen_query_ids"]]


def compress(args, source_rows):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.compressor_model,local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(args.compressor_model,local_files_only=True,torch_dtype=torch.bfloat16).to(args.device).eval()
    target_tok=AutoTokenizer.from_pretrained("models/Qwen3-8B",trust_remote_code=True,local_files_only=True)
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/"compressed.jsonl"
    done={r["example_id"]:r for r in read_jsonl(path)} if path.exists() else {}
    pending=[r for r in source_rows if r["example_id"] not in done]
    with path.open("a",encoding="utf-8") as f:
        for start in range(0,len(pending),args.batch_size):
            batch=pending[start:start+args.batch_size]
            prompts=[f"Question: {r['question']}\n Document: {r['original_context']}\n Summary: " for r in batch]
            encoded=tok(prompts,padding=True,truncation=True,max_length=1024,return_tensors="pt").to(args.device)
            begin=time.perf_counter()
            with torch.no_grad():
                generated=model.generate(**encoded,max_new_tokens=512,num_beams=4,no_repeat_ngram_size=3,early_stopping=True)
            elapsed=time.perf_counter()-begin
            summaries=tok.batch_decode(generated,skip_special_tokens=True)
            for source,summary,input_length in zip(batch,summaries,encoded.attention_mask.sum(dim=1).tolist()):
                context=summary.strip();source_tokens=len(target_tok.encode(source["original_context"],add_special_tokens=False));compressed_tokens=len(target_tok.encode(context,add_special_tokens=False))
                rec={"example_id":source["example_id"],"question":source["question"],"context":context,
                     "source_context_tokens":source_tokens,"compressed_context_tokens":compressed_tokens,
                     "actual_keep_rate":compressed_tokens/max(1,source_tokens),"compressor_input_tokens":int(input_length),
                     "compressor_seconds":elapsed/len(batch)}
                f.write(json.dumps(rec,ensure_ascii=False)+"\n");done[rec["example_id"]]=rec
            f.flush();os.fsync(f.fileno());print(json.dumps({"compressed":len(done),"total":64}),flush=True)
    assert len(done)==64


def evaluate(args, source_rows):
    from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    compressed=list(read_jsonl(OUT/"compressed.jsonl"));ids={r["example_id"] for r in source_rows}
    atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl(Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")) if r["example_id"] in ids}
    baseline={r["example_id"]:r for r in read_jsonl(BASE/"frontier_r1_hard_screen64/fresh_original_baseline_outputs.jsonl")}
    output=OUT/"target_outputs.jsonl";done={r["example_id"]:r for r in read_jsonl(output)} if output.exists() else {}
    pending=[r for r in compressed if r["example_id"] not in done]
    if pending:
        target=QampariTargetRunner("models/Qwen3-8B",backend="vllm",max_new_tokens=256,gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=4096)
        params=SamplingParams(temperature=0.0,max_tokens=256)
        with output.open("a",encoding="utf-8") as f:
            for start in range(0,len(pending),args.target_batch_size):
                batch=pending[start:start+args.target_batch_size];prompts=target._chat_prompts([r["question"] for r in batch],[r["context"] for r in batch])
                begin=time.perf_counter();outputs=target.model.generate(prompts,params,use_tqdm=False);elapsed=time.perf_counter()-begin
                for source,prompt,generated in zip(batch,prompts,outputs):
                    pred=parse_list_prediction(generated.outputs[0].text);rec={k:source[k] for k in ("example_id","source_context_tokens","compressed_context_tokens","actual_keep_rate","compressor_input_tokens","compressor_seconds")}
                    rec.update({"f1":float(qampari_list_metrics(pred,atoms[source["example_id"]])["f1"]),"prediction":pred,"raw_output":generated.outputs[0].text,
                                "prompt_tokens":len(target.tokenizer.encode(prompt)),"generated_tokens":len(generated.outputs[0].token_ids),"target_batch_seconds_per_query":elapsed/len(batch)})
                    f.write(json.dumps(rec,ensure_ascii=False)+"\n");done[rec["example_id"]]=rec
                f.flush();os.fsync(f.fileno());print(json.dumps({"evaluated":len(done),"total":64}),flush=True)
    group=list(done.values());base=[baseline[r["example_id"]]["f1"] for r in group];thresholds=(.60,.70,.80,.90,.95)
    result={"queries":64,"baseline_mean_f1":sum(base)/64,"compressed_mean_f1":sum(r["f1"] for r in group)/64,
            "delta_mean_f1":sum(r["f1"]-baseline[r["example_id"]]["f1"] for r in group)/64,
            "baseline_success_090":sum(x+1e-6>=.9 for x in base),"compressed_success_090":sum(r["f1"]+1e-6>=.9 for r in group),
            "repairs_090":sum(baseline[r["example_id"]]["f1"]+1e-6<.9<=r["f1"]+1e-6 for r in group),
            "breaks_090":sum(baseline[r["example_id"]]["f1"]+1e-6>=.9>r["f1"]+1e-6 for r in group),
            "mean_actual_keep_rate":sum(r["actual_keep_rate"] for r in group)/64,"mean_context_tokens":sum(r["compressed_context_tokens"] for r in group)/64,
            "empty_summaries":sum(r["compressed_context_tokens"]==0 for r in group),"mean_compressor_seconds":sum(r["compressor_seconds"] for r in group)/64,
            "threshold_success":{f"{t:.2f}":{"baseline":sum(x+1e-6>=t for x in base),"compressed":sum(r["f1"]+1e-6>=t for r in group)} for t in thresholds}}
    passed=result["mean_actual_keep_rate"]<=.25 and result["delta_mean_f1"]>=-.02 and result["compressed_success_090"]>=result["baseline_success_090"]-2
    summary={"protocol":"FRONTIER-R1-RECOMP-NQ-ABSTRACTIVE-SCREEN64","qualification":"official NQ checkpoint; zero-shot cross-domain QAMPARI baseline",
             "target":"frozen Qwen3-8B","source":"complete original 12-packet context","generation":{"max_source_tokens":1024,"max_new_tokens":512,"num_beams":4,"no_repeat_ngram_size":3},
             "result":result,"screen_gate_passed":passed,"decision":"GO_RECOMP_ABSTRACTIVE_DESIGN256" if passed else "STOP_RECOMP_ABSTRACTIVE_AFTER_SCREEN64",
             "new_target_calls":len(done),"baseline_reused_calls":64,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("compress","evaluate","all"),default="all");p.add_argument("--compressor-model",default="models/recomp-nq-abstractive")
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=2);p.add_argument("--target-batch-size",type=int,default=64);p.add_argument("--gpu-memory-utilization",type=float,default=.40)
    args=p.parse_args();source_rows=rows()
    if args.mode in ("compress","all"):compress(args,source_rows)
    if args.mode in ("evaluate","all"):evaluate(args,source_rows)


if __name__=="__main__":main()
