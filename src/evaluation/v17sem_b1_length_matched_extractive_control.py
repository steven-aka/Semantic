"""Target-blind length-matched extractive control for the ideal SEM-B1 facts."""
import argparse
import hashlib
import json
import os
import re
import random
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17sem_b1_ideal_representation_positive_control import BASE,C0,LINEAGE,LEVELS,DEPTHS
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask,render

OUT=BASE/"v17sem_b1_length_matched_extractive_control";B1=BASE/"v17sem_b1_ideal_representation_positive_control"
CFG=Path("configs/v17sem_b1_length_matched_extractive_control.json")


def terms(text):return set(re.findall(r"[a-z0-9]+",text.lower()))


def best_span(source,fact,question,tok):
    pieces=list(re.finditer(r"\S+",source));ft,qt=terms(fact),terms(question);cap=len(tok.encode(fact,add_special_tokens=False));best=None
    for i in range(len(pieces)):
        for j in range(i,len(pieces)):
            span=source[pieces[i].start():pieces[j].end()];n=len(tok.encode(span,add_special_tokens=False))
            if n>cap:break
            st=terms(span);fo=len(st&ft);qo=len(st&qt);key=(2*fo+qo,fo+qo,n,-pieces[i].start())
            if best is None or key>best[0]:best=(key,span,n)
    assert best and best[1] in source and best[2]<=cap
    return best[1],best[2],cap


def prepare(cfg,tok):
    manifest=json.loads((C0/"manifest.json").read_text());ids=manifest["query_ids"]
    audit={r["audit_id"]:r for r in read_jsonl(C0/"audit_items.jsonl")};ann={r["audit_id"]:r for r in read_jsonl(C0/"annotation.jsonl")}
    data={r["example_id"]:r for r in read_jsonl(LINEAGE/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids};orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(LINEAGE/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    approved={int(k):v for k,v in json.loads((OUT/"approved_spans.json").read_text()).items()}
    tasks=[];meta=[]
    for aid,q in enumerate(ids,1):
        if ann[aid]["label"]!="ELIGIBLE":continue
        _,_,cap=best_span(audit[aid]["future_atom"],ann[aid]["canonical_fact"],audit[aid]["question"],tok)
        span=approved[aid];n=len(tok.encode(span,add_special_tokens=False))
        assert span in audit[aid]["future_atom"] and n<=cap
        payload="Document title: Extractive evidence\nSource evidence: "+span;packets,order=data[q]["packet_texts"],orders[q]
        for d in (7,9,10):
            context=render(packets,mask(order,d))+"\n\n"+payload
            tasks.append({"example_id":q,"arm":f"extractive{d}","question":data[q]["question"],"context":context,"context_tokens":len(tok.encode(context,add_special_tokens=False))})
        meta.append({"audit_id":aid,"example_id":q,"span":span,"span_tokens":n,"fact_tokens":cap})
    return tasks,meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");args=ap.parse_args();cfg=json.loads(CFG.read_text())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg["target_model"],trust_remote_code=True);tasks,meta=prepare(cfg,tok);OUT.mkdir(parents=True,exist_ok=True)
    manifest={"protocol":cfg["protocol"],"queries":len(meta),"planned_target_calls":len(tasks),"task_sha256":hashlib.sha256("\n".join(f'{x["example_id"]}:{x["arm"]}:{hashlib.sha256(x["context"].encode()).hexdigest()}' for x in tasks).encode()).hexdigest(),"sealed_sets_read":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (OUT/"spans.jsonl").open("w") as h:
        for x in meta:h.write(json.dumps(x,ensure_ascii=False)+"\n")
    if args.preflight_only:print(json.dumps({**manifest,"mean_span_tokens":sum(x["span_tokens"] for x in meta)/len(meta),"mean_fact_tokens":sum(x["fact_tokens"] for x in meta)/len(meta)},indent=2));return
    from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    ids={x["example_id"] for x in meta};atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    call=OUT/"per_call.jsonl";done={(r["example_id"],r["arm"]):r for r in read_jsonl(call)} if call.exists() else {};pending=[x for x in tasks if (x["example_id"],x["arm"]) not in done]
    if pending:
        target=QampariTargetRunner(cfg["target_model"],backend="vllm",max_new_tokens=cfg["max_new_tokens"],gpu_memory_utilization=cfg["gpu_memory_utilization"],max_model_len=cfg["max_model_len"]);params=SamplingParams(temperature=0,max_tokens=cfg["max_new_tokens"])
        with call.open("a") as h:
            for start in range(0,len(pending),cfg["batch_size"]):
                batch=pending[start:start+cfg["batch_size"]];prompts=target._chat_prompts([x["question"] for x in batch],[x["context"] for x in batch]);outs=target.model.generate(prompts,params,use_tqdm=False)
                for task,prompt,out in zip(batch,prompts,outs):
                    pred=parse_list_prediction(out.outputs[0].text);row={"example_id":task["example_id"],"arm":task["arm"],"context_tokens":task["context_tokens"],"f1":float(qampari_list_metrics(pred,atoms[task["example_id"]])["f1"]),"raw_output":out.outputs[0].text,"generated_token_ids":list(out.outputs[0].token_ids),"prompt_tokens":len(target.tokenizer.encode(prompt)),"generated_tokens":len(out.outputs[0].token_ids)}
                    done[row["example_id"],row["arm"]]=row;h.write(json.dumps(row,ensure_ascii=False)+"\n")
                h.flush();os.fsync(h.fileno());print(json.dumps({"completed":len(done),"total":len(tasks)}),flush=True)
    prior={r["example_id"]:r for r in read_jsonl(B1/"per_query.jsonl")};rows=[]
    for m in meta:
        q=m["example_id"];r=prior[q];att=[x is not None for x in r["v8_hits"]];arms=("extractive7","extractive7","extractive9","extractive10")
        f1=[done[q,a]["f1"] for a in arms];hits=[r["v8_hits"][0]]+[None if not att[i] else f1[i-1]+cfg["fidelity_epsilon"]>=LEVELS[i] for i in range(1,5)]
        complete=all(x is not False for x in hits);context=r["v8_context"]+sum((done[q,a]["context_tokens"]-next(x["context_tokens"] for x in read_jsonl(B1/"per_call.jsonl") if x["example_id"]==q and x["arm"]==f"base{d}")) for i,(a,d) in enumerate(zip(arms,(7,7,9,10)),1) if att[i])
        rows.append({**m,"v8_hits":r["v8_hits"],"semantic_hits":r["semantic_hits"],"extractive_hits":hits,"v8_complete":r["v8_complete"],"semantic_complete":r["semantic_complete"],"extractive_complete":complete,"v8_context":r["v8_context"],"semantic_context":r["semantic_context"],"extractive_context":context})
    def stats(name):return {"success":[sum(x[name+"_hits"][i] is True for x in rows) for i in range(5)],"complete":sum(x[name+"_complete"] for x in rows),"complete_repairs":sum(not x["v8_complete"] and x[name+"_complete"] for x in rows),"complete_breaks":sum(x["v8_complete"] and not x[name+"_complete"] for x in rows),"queries_with_any_anchor_break":sum(any(a is True and b is False for a,b in zip(x["v8_hits"],x[name+"_hits"])) for x in rows),"mean_extra_cumulative_context_active40":sum(x[name+"_context"]-x["v8_context"] for x in rows)/len(rows)}
    ext,sem=stats("extractive"),stats("semantic");rng=random.Random(20260921);boot=[]
    for _ in range(10000):
        draw=[rows[rng.randrange(len(rows))] for _ in rows]
        boot.append(sum(x["semantic_complete"]-x["extractive_complete"] for x in draw))
    boot.sort()
    sem_dominates=(sem["complete"]>=ext["complete"] and all(a>=b for a,b in zip(sem["success"],ext["success"])) and
                   sem["queries_with_any_anchor_break"]<=ext["queries_with_any_anchor_break"] and
                   sem["mean_extra_cumulative_context_active40"]<=ext["mean_extra_cumulative_context_active40"])
    result={"protocol":cfg["protocol"],"queries":40,"target_calls":len(done),"extractive":ext,"semantic":sem,
            "paired_complete_difference_semantic_minus_extractive":sem["complete"]-ext["complete"],
            "paired_bootstrap_95pct_complete_difference":[boot[249],boot[9749]],
            "semantic_pareto_dominates_extractive":sem_dominates,
            "interpretation":"SEMANTIC_SPECIFIC_PARETO_GAIN" if sem_dominates else "INCONCLUSIVE_NO_PARETO_DOMINANCE",
            "sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
    with (OUT/"per_query.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")


if __name__=="__main__":main()
