"""Create inspectable sentence-level proof objects for accepted E0B designs.

Sentence witnesses are conservative upper bounds on minimal closure length.
They do not claim that offsets alone prove semantic entailment.
"""
import json,re
from pathlib import Path

from src.evaluation.v17sem_b0_single_atom_eligibility import canonical_tokenizer
from src.representation.token_counter import count_tokens

BASE=Path("results/v2_rank_then_cut");D=BASE/"v17sem_d05_relation_closure_verifier";E=BASE/"v17sem_e0_slot_first_source_backed";OUT=BASE/"v17sem_e0p_proposition_backed"

def read(path,key="audit_id"):return {r[key]:r for r in map(json.loads,path.open())}

def sentence_bounds(text,pos):
    starts=[0]+[m.end() for m in re.finditer(r"(?<=[.!?])\s+",text)]
    ends=[m.start() for m in re.finditer(r"(?<=[.!?])\s+",text)]+[len(text)]
    for a,b in zip(starts,ends):
        if a<=pos<b:return a,b
    return 0,len(text)

def find_ci(text,needle,start=0):
    p=text.lower().find(needle.lower(),start)
    return None if p<0 else [p,p+len(needle)]

def main():
    items=read(D/"fresh_audit_items.jsonl"); cand=read(E/"e0b_design_candidates.jsonl"); review=read(E/"e0b_design_review.jsonl");tok=canonical_tokenizer();rows=[]
    for aid,r in sorted(review.items()):
        if not r["valid_relation_closure"]:continue
        item=items[aid]; c=cand[aid]["candidate"]; packet=item["rank10_packet"]; quote=c["evidence_quote"]
        qoff=find_ci(packet,quote); assert qoff,(aid,quote)
        a,b=sentence_bounds(packet,qoff[0]); sentence=packet[a:b].strip(); shift=packet.find(sentence,a,b+1); sent=[shift,shift+len(sentence)]
        sch=c["schema"]; objects=[]
        for arg in sch["known_arguments"]:
            off=find_ci(packet,arg,sent[0])
            if off and off[1]<=sent[1]:objects.append({"text":packet[off[0]:off[1]],"offsets":off})
        title=c["title"]; toff=find_ci(packet,title)
        full=title+" ; "+sentence
        rows.append({"audit_id":aid,"subject_span":{"text":title,"offsets":toff},"predicate_schema":sch["predicate_slots"],"object_spans":objects,"constraint_schema":sch.get("constraints",[]),"source_sentence":{"text":sentence,"offsets":sent},"link_type":"DOCUMENT_ENTITY_PLUS_EXPLICIT_CLAUSE","compact_candidate":c["serialized"],"compact_tokens":c["tokens"],"sentence_witness_tokens":count_tokens(tok,full),"note":"Sentence witness length is an upper bound, not proven minimal L_closure."})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"design_proof_objects.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    lens=[r["sentence_witness_tokens"] for r in rows]
    summary={"protocol":"SEM-E0P_PROPOSITION_BACKED_PREFLIGHT","proof_objects":len(rows),"sentence_witness_length":{"min":min(lens),"median":sorted(lens)[len(lens)//2],"max":max(lens),"le16":sum(x<=16 for x in lens),"le24":sum(x<=24 for x in lens),"le32":sum(x<=32 for x in lens)},"available_dependency_or_srl_runtime":False,"interpretation":"Offsets and complete sentences establish provenance but not entailment. No reproducible dependency/SRL runtime is installed, so a fresh automatic proposition gate is not yet authorized.","decision":"STOP_BEFORE_FRESH_E0C2_IMPLEMENT_OR_FREEZE_PROPOSITION_LINK_VALIDATOR","new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
