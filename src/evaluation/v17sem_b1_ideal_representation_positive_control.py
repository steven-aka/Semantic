"""Fresh V8/raw/ideal-semantic Target comparison on the SEM-C0 cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render

BASE=Path("results/v2_rank_then_cut");LINEAGE=BASE/"v17sel_b2b_lineage_clean_holdout"
C0=BASE/"v17sem_c0_two_atom_closure";OUT=BASE/"v17sem_b1_ideal_representation_positive_control"
CFG=Path("configs/v17sem_b1_ideal_representation_positive_control.json")
LEVELS=(.60,.70,.80,.90,.95);DEPTHS=(6,7,7,9,10)


def prepare(cfg,tokenizer):
    ids=json.loads((C0/"manifest.json").read_text())["query_ids"]
    audit={r["audit_id"]:r for r in read_jsonl(C0/"audit_items.jsonl")}
    ann={r["audit_id"]:r for r in read_jsonl(C0/"annotation.jsonl")}
    data={r["example_id"]:r for r in read_jsonl(LINEAGE/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(LINEAGE/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    tasks=[];meta=[]
    for aid,q in enumerate(ids,1):
        packets,order=data[q]["packet_texts"],orders[q]
        base={d:render(packets,mask(order,d)) for d in (6,7,9,10)}
        eligible=ann[aid]["label"]=="ELIGIBLE"
        payloads={}
        if eligible:
            payloads["raw"]=audit[aid]["future_atom"]
            payloads["semantic"]="Document title: Semantic evidence\nSource evidence: "+ann[aid]["canonical_fact"]
        arms={f"base{d}":base[d] for d in base}
        for name,payload in payloads.items():
            for d in (7,9,10):arms[f"{name}{d}"]=base[d]+"\n\n"+payload
        for arm,context in arms.items():
            tasks.append({"example_id":q,"arm":arm,"question":data[q]["question"],"context":context,
                          "context_tokens":len(tokenizer.encode(context,add_special_tokens=False))})
        meta.append({"audit_id":aid,"example_id":q,"eligible":eligible,"attainable_levels":data[q]["attainable_levels"]})
    return ids,tasks,meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");args=ap.parse_args()
    cfg=json.loads(CFG.read_text());assert cfg["status"]=="FROZEN_BEFORE_TARGET_CALLS" and cfg["schedule"]==list(DEPTHS)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg["target_model"],trust_remote_code=True)
    ids,tasks,meta=prepare(cfg,tok);OUT.mkdir(parents=True,exist_ok=True)
    manifest={"protocol":cfg["protocol"],"query_ids":ids,"eligible_actions":sum(x["eligible"] for x in meta),
              "planned_target_calls":len(tasks),"task_sha256":hashlib.sha256("\n".join(f'{x["example_id"]}:{x["arm"]}:{hashlib.sha256(x["context"].encode()).hexdigest()}' for x in tasks).encode()).hexdigest(),
              "sealed_sets_read":False}
    mp=OUT/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    if args.preflight_only:print(json.dumps(manifest,indent=2));return
    from src.evaluation.qampari_metrics import parse_list_prediction,qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in set(ids)}
    call=OUT/"per_call.jsonl";done={(r["example_id"],r["arm"]):r for r in read_jsonl(call)} if call.exists() else {}
    pending=[x for x in tasks if (x["example_id"],x["arm"]) not in done]
    if pending:
        target=QampariTargetRunner(cfg["target_model"],backend="vllm",max_new_tokens=cfg["max_new_tokens"],gpu_memory_utilization=cfg["gpu_memory_utilization"],max_model_len=cfg["max_model_len"])
        params=SamplingParams(temperature=cfg["temperature"],max_tokens=cfg["max_new_tokens"])
        with call.open("a") as h:
            for start in range(0,len(pending),cfg["batch_size"]):
                batch=pending[start:start+cfg["batch_size"]];prompts=target._chat_prompts([x["question"] for x in batch],[x["context"] for x in batch]);outs=target.model.generate(prompts,params,use_tqdm=False)
                for task,prompt,out in zip(batch,prompts,outs):
                    pred=parse_list_prediction(out.outputs[0].text);row={"example_id":task["example_id"],"arm":task["arm"],"context_tokens":task["context_tokens"],
                        "f1":float(qampari_list_metrics(pred,atoms[task["example_id"]])["f1"]),"raw_output":out.outputs[0].text,
                        "generated_token_ids":list(out.outputs[0].token_ids),"prompt_tokens":len(target.tokenizer.encode(prompt)),"generated_tokens":len(out.outputs[0].token_ids)}
                    done[row["example_id"],row["arm"]]=row;h.write(json.dumps(row,ensure_ascii=False)+"\n")
                h.flush();os.fsync(h.fileno());print(json.dumps({"completed":len(done),"total":len(tasks)}),flush=True)
    rows=[]
    for m in meta:
        q=m["example_id"];att={float(x) for x in m["attainable_levels"]};armsets={"v8":("base6","base7","base7","base9","base10")}
        for name in ("raw","semantic"):
            armsets[name]=armsets["v8"] if not m["eligible"] else ("base6",f"{name}7",f"{name}7",f"{name}9",f"{name}10")
        out={**m}
        for name,arms in armsets.items():
            hits=[None if level not in att else done[q,a]["f1"]+cfg["fidelity_epsilon"]>=level for level,a in zip(LEVELS,arms)]
            out[name+"_hits"]=hits;out[name+"_complete"]=all(x is not False for x in hits)
            out[name+"_context"]=sum(done[q,a]["context_tokens"] for level,a in zip(LEVELS,arms) if level in att)
        rows.append(out)
    active=[r for r in rows if r["eligible"]]
    def metrics(name):
        return {"success":[sum(r[name+"_hits"][i] is True for r in rows) for i in range(5)],"complete":sum(r[name+"_complete"] for r in rows),
          "complete_repairs":sum(not r["v8_complete"] and r[name+"_complete"] for r in active),"complete_breaks":sum(r["v8_complete"] and not r[name+"_complete"] for r in active),
          "safe_complete_repairs":sum(not r["v8_complete"] and r[name+"_complete"] and not any(x is True and y is False for x,y in zip(r["v8_hits"],r[name+"_hits"])) for r in active),
          "queries_with_any_anchor_break":sum(any(x is True and y is False for x,y in zip(r["v8_hits"],r[name+"_hits"])) for r in active),
          "mean_extra_cumulative_context_all64":sum(r[name+"_context"]-r["v8_context"] for r in rows)/len(rows),
          "mean_extra_cumulative_context_active40":sum(r[name+"_context"]-r["v8_context"] for r in active)/len(active)}
    result={"protocol":cfg["protocol"],"queries":64,"eligible_actions":len(active),"target_calls":len(done),"v8":metrics("v8"),"raw":metrics("raw"),"semantic":metrics("semantic")}
    improvements=sum([result["semantic"]["safe_complete_repairs"]>result["raw"]["safe_complete_repairs"],result["semantic"]["queries_with_any_anchor_break"]<result["raw"]["queries_with_any_anchor_break"],result["semantic"]["mean_extra_cumulative_context_all64"]<result["raw"]["mean_extra_cumulative_context_all64"]])
    go=result["semantic"]["complete"]>=result["v8"]["complete"] and result["semantic"]["safe_complete_repairs"]>=4 and improvements>=2
    result.update({"semantic_vs_raw_strict_improvements":improvements,"decision":"GO_LENGTH_MATCHED_EXTRACTIVE_CONTROL" if go else "STOP_IDEAL_SEMANTIC_REPRESENTATION","sealed_sets_read":False})
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (OUT/"per_query.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
