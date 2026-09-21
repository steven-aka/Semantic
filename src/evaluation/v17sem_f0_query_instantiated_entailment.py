"""Final semantic-evidence rescue: deterministic hypotheses plus entailment."""
import argparse,hashlib,json,os,time
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17sem_e0_query_schema import schema

BASE=Path("results/v2_rank_then_cut");IN=BASE/"v17sem_d05_relation_closure_verifier/fresh_audit_items.jsonl";OUT=BASE/"v17sem_f0_query_instantiated_entailment";CFG=Path("configs/v17sem_f0_query_instantiated_entailment.json")

PROMPT="""Judge whether SOURCE entails the complete HYPOTHESIS exactly as written.
ENTAILED requires the same answer entity, predicate, arguments, answer type, and every constraint. A related entity, substring location (for example Higher Broughton vs Broughton), different subject, different role, missing condition, or merely associated fact is NOT_ENTAILED. Use no outside knowledge. If SOURCE is ambiguous, ABSTAIN.
Return JSON only:
{{"decision":"ENTAILED|NOT_ENTAILED|ABSTAIN","support_quote":"exact contiguous SOURCE quote if ENTAILED, otherwise empty","reason":"short reason"}}

HYPOTHESIS:
{hypothesis}

SOURCE:
{source}
"""

def hypothesis(title,s):
    f=s["parser_family"];a=s["known_arguments"];p=s["predicate_slots"]
    if f=="FILM_CAST":
        bits=[]
        for pred,arg in zip(p,a):bits.append((f"has {arg} as a cast member") if pred=="cast_member" else f"was directed by {arg}")
        return f"{title} is a film that "+" and ".join(bits)+"."
    simple={"PLAYED_FOR":f"{title} played for {a[0]}.","BORN_IN":f"{title} was born in {a[0]}.","DIED_IN":f"{title} died in {a[0]}.","WON_AWARD":f"{title} won {a[0]}.","RECEIVED_AWARD":f"{title} received {a[0]}.","BURIED_IN":f"{title} was buried in {a[0]}.","PERFORMED_BY":f"{title} is an album performed by {a[0]}.","WRITTEN_BY":f"{title} is a song whose lyrics were written by {a[0]}.","FORMED_IN":f"{title} is a group formed in {a[0]}.","LOCATED_IN":f"{title} is a spatial entity located in {a[0]}.","PARTY_MEMBER":f"{title} was a member of the political party {a[0]}.","PART_OF":f"{title} is a competition that is part of {a[0]}.","RECORD_LABEL":f"{title} is an album on the record label {a[0]}.","CAST_MEMBER":f"{title} is a film with {a[0]} as a cast member.","PLAYED_IN_LEAGUE":f"{title} is a sports team that played in {a[0]}."}
    if f in simple:return simple[f]
    if f=="TABLE_COMPOSITION":
        domain=(" and belongs to the set of "+" and ".join(s["constraints"])) if s["constraints"] else ""
        return f"{title} is a {s['answer_slot']}{domain}. The {p[0]} of {title} is {a[0]}."
    if f=="TEMPORAL_FILTER":return f"{title} is {s['answer_slot']} created or occurring after {a[0]}."
    if f=="CLASSIFIED_UNDER":return f"{title} is {s['answer_slot']} classified under {a[0]}."
    if f=="FILM_CAST_MUSIC":return f"{title} is a film with {a[0]} as a cast member and music composed by {a[1]}."
    if f=="COMPETITION_LOCATION_PARTICIPANT":return f"{title} is a competition located in {a[0]} with {a[1]} as a participant."
    raise ValueError(f)

def parse(raw):
    a,b=raw.find("{"),raw.rfind("}")
    if a<0 or b<=a:raise ValueError("no JSON")
    return json.loads(raw[a:b+1])

def validate(v,source,cfg):
    if set(v)!={"decision","support_quote","reason"}:return False,"BAD_KEYS"
    if v["decision"] not in cfg["decisions"]:return False,"BAD_DECISION"
    if not all(isinstance(v[k],str) for k in v):return False,"BAD_TYPES"
    if v["decision"]=="ENTAILED" and (not v["support_quote"] or v["support_quote"] not in source):return False,"QUOTE_NOT_EXACT"
    if v["decision"]!="ENTAILED" and v["support_quote"]:return False,"NONEMPTY_NEGATIVE_QUOTE"
    return True,None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight-only",action="store_true");args=ap.parse_args();cfg=json.loads(CFG.read_text());assert cfg["status"]=="FROZEN_BEFORE_TEACHER_CALLS"
    tasks=[]
    for r in read_jsonl(IN):
        title=next(x.removeprefix("Document title: ").strip() for x in r["rank10_packet"].splitlines() if x.startswith("Document title: "));h=hypothesis(title,schema(r["question"]))
        for arm,source in (("future",r["rank10_packet"]),("current_s6",r["state_s6"])):
            tasks.append({"audit_id":r["audit_id"],"arm":arm,"title":title,"hypothesis":h,"source":source,"prompt":PROMPT.format(hypothesis=h,source=source)})
    OUT.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256("\n".join(hashlib.sha256(x["prompt"].encode()).hexdigest() for x in tasks).encode()).hexdigest();manifest={"protocol":cfg["protocol"],"items":len(tasks),"queries":96,"prompt_sha256":digest,"model":cfg["teacher_model"],"gold_or_target_outcomes_loaded":False,"sealed_sets_read":False}
    mp=OUT/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    with (OUT/"hypotheses.jsonl").open("w") as h:
        for t in tasks[::2]:h.write(json.dumps({k:t[k] for k in ("audit_id","title","hypothesis")},ensure_ascii=False)+"\n")
    if args.preflight_only:print(json.dumps({"tasks":len(tasks),"prompt_sha256":digest,"status":"FROZEN_READY_FOR_TEACHER"},indent=2));return
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained(cfg["teacher_model"],trust_remote_code=True);path=OUT/"generations.jsonl";done={(r["audit_id"],r["arm"]):r for r in read_jsonl(path)} if path.exists() else {};pending=[t for t in tasks if (t["audit_id"],t["arm"]) not in done]
    model=LLM(model=cfg["teacher_model"],trust_remote_code=True,dtype="bfloat16",gpu_memory_utilization=.82,max_model_len=4096,max_num_seqs=1) if pending else None;params=SamplingParams(temperature=0,max_tokens=cfg["generation"]["max_tokens"])
    with path.open("a") as out:
        for t in tasks:
            key=t["audit_id"],t["arm"]
            if key in done:continue
            formatted=tok.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False);st=time.perf_counter();o=model.generate([formatted],params,use_tqdm=False)[0];lat=time.perf_counter()-st;raw=o.outputs[0].text;row={"audit_id":t["audit_id"],"arm":t["arm"],"hypothesis":t["hypothesis"],"raw_output":raw,"teacher_prompt_tokens":len(tok.encode(formatted)),"teacher_generated_tokens":len(o.outputs[0].token_ids),"latency_seconds_batch1":lat}
            try:v=parse(raw);ok,err=validate(v,t["source"],cfg);row.update(parsed=v,hard_valid=ok,hard_error=err)
            except Exception as e:row.update(hard_valid=False,hard_error=str(e))
            done[key]=row;out.write(json.dumps(row,ensure_ascii=False)+"\n");out.flush();os.fsync(out.fileno());print(json.dumps({"completed":len(done),"total":len(tasks)}),flush=True)
    rows=[done[(t["audit_id"],t["arm"])] for t in tasks];summary={"protocol":cfg["protocol"],"tasks":len(rows),"hard_valid":sum(r["hard_valid"] for r in rows),"future_entailed":sum(r["arm"]=="future" and r.get("parsed",{}).get("decision")=="ENTAILED" and r["hard_valid"] for r in rows),"current_entailed":sum(r["arm"]=="current_s6" and r.get("parsed",{}).get("decision")=="ENTAILED" and r["hard_valid"] for r in rows),"manual_regression_review":"PENDING","fresh_f0b_authorized":False,"teacher_calls":len(rows),"new_target_calls":0,"sealed_sets_read":False};(OUT/"automatic_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
