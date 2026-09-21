"""Freeze SEM-D0A feasibility items and the label-blind SEM-D0B policy."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask,render,segments
from src.evaluation.v17sem_b0_single_atom_eligibility import canonical_tokenizer,validate_annotation
from src.representation.token_counter import count_tokens

BASE=Path("results/v2_rank_then_cut");ROOT=BASE/"v17sel_b2b_lineage_clean_holdout"
OUT=BASE/"v17sem_d0_automatic_grounded_extractive";CFG=Path("configs/v17sem_d0_automatic_grounded_extractive.json")
GENERIC={"which","what","who","are","the","a","an","of","in","as","such","that","their","value","values","name","title","film","album","member","members","played","performer","artist","team","organization"}


def source_pool():
    m0b={r["example_id"]:r for r in read_jsonl(BASE/"v17traj_m0b_borrow_preflight/per_query.jsonl") if r["eligible"]}
    excluded=set()
    for name in ("v17traj_m0c_borrow_pilot","v17traj_m1_low_cost_borrow","v17traj_m2b_promote_delay_pilot","v17packet_obs_a1_natural_insertion","v17packet_frag_b0_title_only_natural_pilot"):
        excluded.update(json.loads((BASE/name/"manifest.json").read_text())["query_ids"])
    pool=sorted(set(m0b)-excluded,key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))
    for name in ("v17sem_a0b_query_conditioned_positive_control","v17sem_a0b_query_conditioned_positive_control_v2","v17sem_b0_single_atom_eligibility"):
        excluded.update(json.loads((BASE/name/"manifest.json").read_text())["query_ids"])
    remaining=[q for q in pool if q not in excluded]
    assert len(pool)==235 and len(remaining)==107
    return remaining[:96],remaining[96:],m0b


def words_with_offsets(text):return list(re.finditer(r"\S+",text))
def terms(text):return set(re.findall(r"[a-z0-9]+",text.lower()))


def automatic_candidate(question,s6,atom1,atom2,tok):
    source=atom1+"\n\n"+atom2;title=atom1.splitlines()[0].removeprefix("Document title: ").strip()
    title_span=title;query_terms=terms(question)-GENERIC;s6_terms=terms(s6)
    if not title or terms(title)<=s6_terms:return None,"TITLE_NOT_NOVEL"
    evidence=" ".join(x.removeprefix("Source evidence: ") for x in source.splitlines() if x.startswith("Source evidence: "))
    pieces=words_with_offsets(evidence);best=None
    for i in range(len(pieces)):
        for j in range(i,len(pieces)):
            span=evidence[pieces[i].start():pieces[j].end()]
            serialized=title_span+" ; "+span;n=count_tokens(tok,serialized)
            if n>16:break
            overlap=len(terms(span)&query_terms)
            key=(overlap,n,-pieces[i].start())
            if best is None or key>best[0]:best=(key,serialized,span,n)
    if best is None or best[0][0]==0:return None,"NO_QUERY_TERM_OVERLAP"
    return {"serialized":best[1],"title_span":title_span,"evidence_span":best[2],"tokens":best[3],"query_overlap":best[0][0]},None


def build():
    chosen,reserved,m0b=source_pool();data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen};orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen};tok=canonical_tokenizer();rows=[];auto=[]
    for aid,q in enumerate(chosen,1):
        packets,order=data[q]["packet_texts"],orders[q];atoms=segments(packets[order[9]]);i=m0b[q]["candidate_index"];j=i+1 if i+1<len(atoms) else i-1
        pair=atoms[i]+"\n\n"+atoms[j];s6=render(packets,mask(order,6));cand,reason=automatic_candidate(data[q]["question"],s6,atoms[i],atoms[j],tok)
        rows.append({"audit_id":aid,"question":data[q]["question"],"state_s6":s6,"primary_atom":atoms[i],"adjacent_atom":atoms[j],"future_atom":pair,"source_tokens_qwen3_8b":count_tokens(tok,pair)})
        auto.append({"audit_id":aid,"decision":"EMIT" if cand else "ABSTAIN","candidate":cand,"abstain_reason":reason})
    return chosen,reserved,rows,auto


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--score",action="store_true");args=ap.parse_args();cfg=json.loads(CFG.read_text());assert cfg["status"]=="FROZEN_BEFORE_FEASIBILITY_LABELS"
    chosen,reserved,rows,auto=build();OUT.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256("\n".join(json.dumps(r,sort_keys=True) for r in rows).encode()).hexdigest()
    manifest={"protocol":cfg["protocol"],"query_ids":chosen,"reserved_query_ids":reserved,"items":96,"content_sha256":digest,"gold_or_outcomes_loaded":False,"new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    mp=OUT/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    for name,vals in (("audit_items.jsonl",rows),("automatic_policy.jsonl",auto)):
        with (OUT/name).open("w") as h:
            for r in vals:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    if not args.score:print(json.dumps({"items":96,"reserved":len(reserved),"automatic_emissions":sum(x["decision"]=="EMIT" for x in auto),"content_sha256":digest,"status":"AWAITING_AI_ASSISTED_FEASIBILITY_REVIEW"},indent=2));return
    ann=list(read_jsonl(OUT/"annotation.jsonl"));assert len(ann)==96;byid={r["audit_id"]:r for r in rows};tok=canonical_tokenizer();bad={r["audit_id"]:validate_annotation(byid[r["audit_id"]],r,tok) for r in ann};bad={k:v for k,v in bad.items() if v}
    if bad:raise ValueError(bad)
    for r in ann:
        if r["label"]=="ELIGIBLE":
            spans=r.get("source_spans") or [];serialized=" ; ".join(spans)
            if not 1<=len(spans)<=2 or any(s not in byid[r["audit_id"]]["future_atom"] for s in spans) or count_tokens(tok,serialized)>16:
                raise ValueError(f'invalid compact spans for {r["audit_id"]}')
    truth={r["audit_id"]:r["label"]=="ELIGIBLE" for r in ann};emitted={r["audit_id"]:r for r in auto if r["decision"]=="EMIT"}
    auto_audit={r["audit_id"]:r for r in read_jsonl(OUT/"automatic_audit.jsonl")};assert set(auto_audit)==set(emitted)
    correct=[auto_audit[aid]["valid_compact_evidence"] for aid in emitted]
    eligible=sum(truth.values());tp=sum(correct);precision=tp/len(emitted) if emitted else 0;recall=tp/eligible if eligible else 0;coverage=len(emitted)/96
    gate=precision>=cfg["automatic_gate"]["emission_grounding_precision"] and recall>=cfg["automatic_gate"]["eligible_recall"] and coverage>=cfg["automatic_gate"]["minimum_emission_coverage"]
    summary={"protocol":cfg["protocol"],"items":96,"audited_eligible":eligible,"automatic_emissions":len(emitted),"automatic_semantically_correct":tp,"emission_precision":precision,"eligible_recall":recall,"emission_coverage":coverage,"decision":"GO_SEM_D1_FRESH_TARGET" if gate else "STOP_AUTOMATIC_POLICY_REDESIGN_ON_NEW_COHORT_ONLY","human_iaa":"NOT_AVAILABLE","new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
