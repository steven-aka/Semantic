"""Precision-first relation-closure verifier for SEM-D0.5.

The verifier accepts only frozen, explicitly supported question schemas. Unknown
schemas and truncated required arguments abstain. It never completes evidence.
"""
import json
import math
import re
from pathlib import Path

BASE=Path("results/v2_rank_then_cut")
SRC=BASE/"v17sem_d0_automatic_grounded_extractive"
OUT=BASE/"v17sem_d05_relation_closure_verifier"

def norm(s):
    return re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()

def contains(text,arg):
    t=" "+norm(text)+" "; toks=norm(arg).split()
    while toks and toks[-1] in {"fc","f","c"}:toks.pop()
    return bool(toks) and all((" "+x+" ") in t for x in toks)

def decide(question,candidate):
    q=question.strip(); c=candidate.strip(); evidence=c.split(";",1)[-1]; lo=norm(evidence)
    m=re.fullmatch(r"Which film has (.+?) as a member of its cast(?: and was directed by (.+?))?\??",q,re.I)
    if m:
        persons=[x for x in m.groups() if x]
        cue=bool(re.search(r"\b(stars?|starring|features?|cast|supporting cast)\b",lo))
        return cue and all(contains(c,x) for x in persons),"FILM_CAST"
    m=re.fullmatch(r"What's the name of a film with (.+?) in it\??",q,re.I)
    if m:
        return contains(c,m.group(1)) and bool(re.search(r"\b(stars?|starring|features?|cast)\b",lo)),"FILM_CAST"
    m=re.fullmatch(r"Who played for the (.+?)\??",q,re.I)
    if m:
        return contains(c,m.group(1)) and bool(re.search(r"\b(play|played|playing)\b",lo)),"PLAYED_FOR"
    m=re.fullmatch(r"Who was born in (.+?)\??",q,re.I)
    if m:return contains(c,m.group(1)) and " born " in (" "+lo+" "),"BORN_IN"
    m=re.fullmatch(r"What is the name of the person born in (.+?)\??",q,re.I)
    if m:return contains(c,m.group(1)) and " born " in (" "+lo+" "),"BORN_IN"
    m=re.fullmatch(r"Which album has (.+?) as performer\??",q,re.I)
    if m:
        return contains(c,m.group(1)) and " album " in (" "+lo+" ") and " by " in (" "+lo+" "),"ALBUM_BY"
    m=re.fullmatch(r"Which songs had its lyrics written by (.+?)\??",q,re.I)
    if m:return contains(c,m.group(1)) and bool(re.search(r"\b(written|lyrics)\b",lo)),"WRITTEN_BY"
    m=re.fullmatch(r"Who is or was buried in (.+?)\??",q,re.I)
    if m:return contains(c,m.group(1)) and " buried " in (" "+lo+" "),"BURIED_IN"
    m=re.fullmatch(r"Which spatial entity is located in (.+?)\??",q,re.I)
    if m:return contains(c,m.group(1)) and " located " in (" "+lo+" "),"LOCATED_IN"
    return False,"UNKNOWN_OR_UNSUPPORTED_SCHEMA"

def wilson(k,n,z=1.96):
    if not n:return [0.0,0.0]
    p=k/n; d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def read(name):return {r["audit_id"]:r for r in map(json.loads,(SRC/name).open())}

def main():
    items,policy,audit=read("audit_items.jsonl"),read("automatic_policy.jsonl"),read("automatic_audit.jsonl")
    rows=[]
    for aid,p in sorted(policy.items()):
        if p["decision"]!="EMIT":continue
        emit,schema=decide(items[aid]["question"],p["candidate"]["serialized"])
        rows.append({"audit_id":aid,"schema":schema,"decision":"EMIT" if emit else "ABSTAIN","candidate":p["candidate"]["serialized"],"valid_compact_evidence":audit[aid]["valid_compact_evidence"]})
    emitted=[r for r in rows if r["decision"]=="EMIT"]
    tp=sum(r["valid_compact_evidence"] for r in emitted); eligible=sum(r["valid_compact_evidence"] for r in rows)
    s={"protocol":"SEM-D0.5_DESIGN_SET_SANITY_ONLY","design_exposed":True,"input_emissions":len(rows),"verified_emissions":len(emitted),"true_emissions":tp,"precision":tp/len(emitted) if emitted else 0,"precision_wilson_95":wilson(tp,len(emitted)),"recall_among_valid_D0_emissions":tp/eligible if eligible else 0,"note":"This result may guide implementation but cannot pass the fresh gate."}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"design_set_verifier_outputs.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    (OUT/"design_set_summary.json").write_text(json.dumps(s,indent=2)+"\n")
    print(json.dumps(s,indent=2))

if __name__=="__main__":main()
