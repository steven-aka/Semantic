"""Run the frozen Qwen3-14B constrained proposition-linker positive control."""
import argparse,hashlib,json,os,re,time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17sem_e0_query_schema import schema

BASE=Path("results/v2_rank_then_cut");IN=BASE/"v17sem_d05_relation_closure_verifier/fresh_audit_items.jsonl";OUT=BASE/"v17sem_e0q_constrained_linker";CFG=Path("configs/v17sem_e0q_constrained_linker.json")

PROMPT="""You are a conservative source proposition linker. Do not answer the question and do not write a new fact.
Select exact quotes that jointly prove the requested relation for the answer-bearing document entity. If any required predicate, argument, answer type, constraint, or relation link is missing or ambiguous, ABSTAIN.

Return JSON only. For ABSTAIN:
{{"decision":"ABSTAIN","reason":"short reason"}}
For LINK:
{{"decision":"LINK","sentence_id":INTEGER,"subject_quote":"exact quote","predicate_quote":"exact quote","object_quotes":["exact quote"],"constraint_quotes":["exact quote"],"link_type":"SAME_CLAUSE|PASSIVE|RELATIVE_CLAUSE|APPOSITION|DOCUMENT_TITLE_EXPLICIT_LINK"}}

Quotes must be exact substrings of one numbered SOURCE sentence (the subject may instead be the exact DOCUMENT TITLE). Do not combine unrelated clauses. A document title plus a predicate fragment is invalid unless that sentence explicitly links the title/document entity to the predicate. In particular, do not change an album's action into its performer's action.

QUESTION SCHEMA:
{schema}

QUESTION:
{question}

DOCUMENT TITLE:
{title}

SOURCE SENTENCES:
{sentences}
"""

def split_source(packet):
    title=next((x.removeprefix("Document title: ").strip() for x in packet.splitlines() if x.startswith("Document title: ")),"")
    evidence=" ".join(x.removeprefix("Source evidence: ") for x in packet.splitlines() if x.startswith("Source evidence: "))
    raw=[x.strip() for x in re.split(r"(?<=[.!?])\s+",evidence) if x.strip()]
    return title,{i+1:x for i,x in enumerate(raw)}

def parse(raw):
    a,b=raw.find("{"),raw.rfind("}")
    if a<0 or b<=a:raise ValueError("no JSON")
    return json.loads(raw[a:b+1])

def hard_validate(value,title,sents,cfg):
    if value.get("decision")=="ABSTAIN":return True,None
    if value.get("decision")!="LINK":return False,"BAD_DECISION"
    required={"decision","sentence_id","subject_quote","predicate_quote","object_quotes","constraint_quotes","link_type"}
    if set(value)!=required:return False,"BAD_KEYS"
    if value["sentence_id"] not in sents:return False,"BAD_SENTENCE_ID"
    if value["link_type"] not in cfg["link_types"]:return False,"BAD_LINK_TYPE"
    sent=sents[value["sentence_id"]]
    if not isinstance(value["object_quotes"],list) or not isinstance(value["constraint_quotes"],list):return False,"BAD_LIST"
    vals=[value["predicate_quote"],*value["object_quotes"],*value["constraint_quotes"]]
    if not value["subject_quote"] or (value["subject_quote"] not in sent and value["subject_quote"]!=title):return False,"SUBJECT_NOT_EXACT"
    if any(not isinstance(x,str) or not x or x not in sent for x in vals):return False,"QUOTE_NOT_EXACT"
    return True,None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");args=ap.parse_args();cfg=json.loads(CFG.read_text());assert cfg["status"]=="FROZEN_BEFORE_TEACHER_CALLS"
    tasks=[]
    for r in read_jsonl(IN):
        title,sents=split_source(r["rank10_packet"]);sch=schema(r["question"])
        text="\n".join(f"[{k}] {v}" for k,v in sents.items())
        prompt=PROMPT.format(schema=json.dumps(sch,ensure_ascii=False),question=r["question"],title=title,sentences=text)
        tasks.append({"audit_id":r["audit_id"],"question":r["question"],"title":title,"sentences":sents,"schema":sch,"prompt":prompt})
    OUT.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256("\n".join(hashlib.sha256(x["prompt"].encode()).hexdigest() for x in tasks).encode()).hexdigest()
    manifest={"protocol":cfg["protocol"],"audit_ids":[x["audit_id"] for x in tasks],"items":len(tasks),"prompt_sha256":digest,"model":cfg["teacher_model"],"gold_or_target_outcomes_loaded":False,"sealed_sets_read":False}
    mp=OUT/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    if args.preflight_only:
        print(json.dumps({"items":len(tasks),"prompt_sha256":digest,"status":"FROZEN_READY_FOR_TEACHER"},indent=2));return
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(cfg["teacher_model"],trust_remote_code=True)
    path=OUT/"generations.jsonl";done={r["audit_id"]:r for r in read_jsonl(path)} if path.exists() else {}
    pending=[x for x in tasks if x["audit_id"] not in done]
    model=LLM(model=cfg["teacher_model"],trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=.82,max_model_len=4096,max_num_seqs=1) if pending else None
    params=SamplingParams(temperature=0,max_tokens=cfg["generation"]["max_tokens"])
    with path.open("a") as h:
        for t in tasks:
            if t["audit_id"] in done:continue
            formatted=tokenizer.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            st=time.perf_counter();out=model.generate([formatted],params,use_tqdm=False)[0];lat=time.perf_counter()-st;raw=out.outputs[0].text
            row={"audit_id":t["audit_id"],"raw_output":raw,"teacher_prompt_tokens":len(tokenizer.encode(formatted)),"teacher_generated_tokens":len(out.outputs[0].token_ids),"latency_seconds_batch1":lat}
            try:
                value=parse(raw);ok,err=hard_validate(value,t["title"],t["sentences"],cfg);row.update({"parsed":value,"hard_valid":ok,"hard_error":err})
                if ok and value.get("decision")=="LINK":
                    sent=t["sentences"][value["sentence_id"]];row["offsets"]={k:[sent.find(v),sent.find(v)+len(v)] for k,v in (("subject",value["subject_quote"]),("predicate",value["predicate_quote"])) if v in sent}
            except Exception as e:row.update({"hard_valid":False,"hard_error":str(e)})
            done[t["audit_id"]]=row;h.write(json.dumps(row,ensure_ascii=False)+"\n");h.flush();os.fsync(h.fileno());print(json.dumps({"completed":len(done),"total":len(tasks)}),flush=True)
    rows=[done[x["audit_id"]] for x in tasks];summary={"protocol":cfg["protocol"],"items":len(rows),"hard_valid_outputs":sum(r["hard_valid"] for r in rows),"links":sum(r.get("parsed",{}).get("decision")=="LINK" and r["hard_valid"] for r in rows),"abstains":sum(r.get("parsed",{}).get("decision")=="ABSTAIN" and r["hard_valid"] for r in rows),"teacher_prompt_tokens":sum(r["teacher_prompt_tokens"] for r in rows),"teacher_generated_tokens":sum(r["teacher_generated_tokens"] for r in rows),"manual_semantic_review":"PENDING","fresh_e0c2_authorized":False,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"automatic_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
