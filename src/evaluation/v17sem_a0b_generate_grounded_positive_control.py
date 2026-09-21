"""Generate the frozen 32-item SEM-A0B grounding-audit set with Qwen3-14B."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments
from src.teacher.packet_validator import numeric_concepts

BASE = Path("results/v2_rank_then_cut")
ROOT = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17sem_a0b_query_conditioned_positive_control"
CFG = Path("configs/v17sem_a0b_query_conditioned_positive_control.json")

PROMPT_V1 = """Extract one compact fact from the source that helps answer the question.
Use only the source. Do not infer an unstated relation or use outside knowledge.
The fact must be at most {budget} Qwen3-8B tokenizer tokens.
Return JSON only with exactly:
{{"source_quote":"an exact contiguous quote copied from SOURCE", "fact":"one source-supported query-relevant fact"}}

QUESTION:
{question}

SOURCE:
{source}
"""

PROMPT_V2 = """Extract one affirmative evidence fact from SOURCE that helps answer QUESTION.
The fact must be fully supported by one exact contiguous SOURCE quote.
Do not say that the source lacks, omits, or does not mention information.
Do not infer a cast member, relation, category, date, or location that the quote does not state.
FACT must contain at most 10 whitespace-separated words and at most {budget} Qwen3-8B tokens.
Return JSON only with exactly:
{{"source_quote":"exact contiguous SOURCE substring that states the fact", "fact":"short affirmative supported fact"}}

QUESTION:
{question}

SOURCE:
{source}
"""


def parse(text):
    start,end=text.find("{"),text.rfind("}")
    if start<0 or end<=start: raise ValueError("no JSON object")
    value=json.loads(text[start:end+1])
    if set(value)!={"source_quote","fact"} or not all(isinstance(value[k],str) for k in value):
        raise ValueError("wrong schema")
    return {k:value[k].strip() for k in value}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");ap.add_argument("--revision",choices=("v1","v2"),default="v1");args=ap.parse_args()
    cfg=json.loads(CFG.read_text());assert cfg["status"]=="DESIGN_FROZEN_NO_TEACHER_OR_TARGET_CALLS_YET"
    m0b={r["example_id"]:r for r in read_jsonl(BASE/"v17traj_m0b_borrow_preflight/per_query.jsonl") if r["eligible"]}
    excluded=set()
    for name in ("v17traj_m0c_borrow_pilot","v17traj_m1_low_cost_borrow","v17traj_m2b_promote_delay_pilot","v17packet_obs_a1_natural_insertion","v17packet_frag_b0_title_only_natural_pilot"):
        excluded.update(json.loads((BASE/name/"manifest.json").read_text())["query_ids"])
    pool=sorted(set(m0b)-excluded,key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))
    assert len(pool)==235
    offset=0 if args.revision=="v1" else 32
    chosen=pool[offset:offset+32]
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen}
    orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen}
    assert len(data)==len(orders)==32
    tasks=[]
    for q in chosen:
        packet=data[q]["packet_texts"][orders[q][9]]
        source=segments(packet)[m0b[q]["candidate_index"]]
        template=PROMPT_V1 if args.revision=="v1" else PROMPT_V2
        tasks.append({"example_id":q,"question":data[q]["question"],"source":source,
                      "prompt":template.format(budget=cfg["max_fact_tokens_qwen3_8b"],question=data[q]["question"],source=source)})
    out=OUT if args.revision=="v1" else Path(str(OUT)+"_v2")
    out.mkdir(parents=True,exist_ok=True)
    manifest={"protocol":cfg["protocol"] if args.revision=="v1" else cfg["protocol"]+"_V2","query_ids":chosen,"audit_items":32,
              "task_sha256":hashlib.sha256("\n".join(hashlib.sha256(t["prompt"].encode()).hexdigest() for t in tasks).encode()).hexdigest(),
              "gold_or_target_outcomes_loaded":False,"sealed_sets_read":False}
    if args.revision=="v2": manifest["cohort_offset"]=offset
    mp=out/"manifest.json"
    if mp.exists(): assert json.loads(mp.read_text())==manifest
    else: mp.write_text(json.dumps(manifest,indent=2)+"\n")
    if args.preflight_only:
        print(json.dumps({k:v for k,v in manifest.items() if k!="query_ids"},indent=2));return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer=AutoTokenizer.from_pretrained("models/Qwen3-8B",trust_remote_code=True)
    teacher_tokenizer=AutoTokenizer.from_pretrained(cfg["teacher_model"],trust_remote_code=True)
    path=out/"generations.jsonl";done={r["example_id"]:r for r in read_jsonl(path)} if path.exists() else {}
    pending=[t for t in tasks if t["example_id"] not in done]
    model=(LLM(model=cfg["teacher_model"],trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=.85,max_model_len=2048,max_num_seqs=1)
           if pending else None)
    params=SamplingParams(temperature=0,max_tokens=128)
    with path.open("a") as handle:
        for index,t in enumerate(tasks):
            if t["example_id"] in done:continue
            formatted=teacher_tokenizer.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            start=time.perf_counter();model_output=model.generate([formatted],params,use_tqdm=False)[0];latency=time.perf_counter()-start
            raw=model_output.outputs[0].text
            row={**t,"raw_output":raw,"teacher_prompt_tokens":len(teacher_tokenizer.encode(formatted)),
                 "teacher_generated_tokens":len(model_output.outputs[0].token_ids),"latency_seconds_batch1":latency}
            try:
                parsed=parse(raw);fact=parsed["fact"];quote=parsed["source_quote"]
                entities=set(re.findall(r"\b[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+)*",fact))
                row.update(parsed);row["fact_tokens_qwen3_8b"]=len(tokenizer.encode(fact,add_special_tokens=False))
                row["quote_exact_substring"]=bool(quote) and quote in t["source"]
                row["numbers_grounded"]=numeric_concepts(fact)<=numeric_concepts(t["source"])
                row["entities_grounded"]=all(e.casefold() in t["source"].casefold() for e in entities)
                row["automatic_valid"]=bool(fact) and row["fact_tokens_qwen3_8b"]<=cfg["max_fact_tokens_qwen3_8b"] and row["quote_exact_substring"] and row["numbers_grounded"] and row["entities_grounded"]
            except Exception as exc:
                row.update({"parse_error":str(exc),"automatic_valid":False})
            done[t["example_id"]]=row;handle.write(json.dumps(row,ensure_ascii=False)+"\n");handle.flush();os.fsync(handle.fileno())
            print(json.dumps({"completed":len(done),"total":32}),flush=True)
    rows=[done[q] for q in chosen]
    blind=[{"audit_id":i+1,"question":r["question"],"source":r["source"],"source_quote":r.get("source_quote"),"fact":r.get("fact"),
            "review_supported_all_predicates":None,"review_notes":None} for i,r in enumerate(rows)]
    with (out/"blind_grounding_audit.jsonl").open("w") as h:
        for r in blind:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    automatic_valid=sum(r["automatic_valid"] for r in rows)
    summary={"protocol":manifest["protocol"],"generator_revision":args.revision,"items":32,"automatic_valid":automatic_valid,
             "exact_quote":sum(r.get("quote_exact_substring",False) for r in rows),"fact_within_16_tokens":sum(r.get("fact_tokens_qwen3_8b",999)<=16 for r in rows),
             "mean_fact_tokens_qwen3_8b":sum(r.get("fact_tokens_qwen3_8b",0) for r in rows)/32,
             "teacher_prompt_tokens":sum(r["teacher_prompt_tokens"] for r in rows),"teacher_generated_tokens_including_quote":sum(r["teacher_generated_tokens"] for r in rows),
             "mean_batch1_latency_seconds":sum(r["latency_seconds_batch1"] for r in rows)/32,
             "automatic_precheck":"PASS_TO_INDEPENDENT_REVIEW" if automatic_valid>=31 else "FAIL_AUTOMATIC_PRECHECK",
             "manual_blind_audit_status":"PENDING_INDEPENDENT_REVIEW" if automatic_valid>=31 else "NOT_OPENED_AUTOMATIC_PRECHECK_FAILED","sem_a1_authorized":False,
             "reason":"Frozen protocol requires >=31/32 strict source-supported judgments; necessary automatic checks are below that ceiling." if automatic_valid<31 else "Independent predicate-level review remains required.",
             "new_target_calls":0,"sealed_sets_read":False}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
