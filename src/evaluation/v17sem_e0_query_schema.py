"""Question-only schema extraction for SEM-E0.

The parser preserves raw task relation labels where a reliable ontology mapping
is unavailable. It does not inspect source text or generate answers.
"""
import json,re
from collections import Counter
from pathlib import Path

BASE=Path("results/v2_rank_then_cut")
IN=BASE/"v17sem_d05_relation_closure_verifier/fresh_audit_items.jsonl"
OUT=BASE/"v17sem_e0_slot_first_source_backed"

def schema(question):
    q=question.strip().rstrip("?")
    # QAMPARI table-composition template. Preserve the raw fields because they
    # carry dataset-specific constraints that must not be silently normalized.
    m=re.fullmatch(r"What are the (.+?) of (.+?) such that their (.+?) value are (.+)",q,re.I)
    if m:
        ans,domain,pred,arg=m.groups()
        return {"answer_slot":ans,"predicate_slots":[pred],"known_arguments":[arg],"constraints":[domain],"parser_family":"TABLE_COMPOSITION"}
    m=re.fullmatch(r"What are the (.+?) such that their (.+?) value are (.+)",q,re.I)
    if m:
        ans,pred,arg=m.groups()
        return {"answer_slot":ans,"predicate_slots":[pred],"known_arguments":[arg],"constraints":[],"parser_family":"TABLE_COMPOSITION"}
    m=re.fullmatch(r"What are the (.+?) such that they were created or happened after (\d+)",q,re.I)
    if m:
        ans,year=m.groups()
        return {"answer_slot":ans,"predicate_slots":["created_or_happened_after"],"known_arguments":[year],"constraints":[],"parser_family":"TEMPORAL_FILTER"}
    m=re.fullmatch(r"What is (.+?) that is classified under (.+)",q,re.I)
    if m:
        ans,arg=m.groups()
        return {"answer_slot":ans,"predicate_slots":["classified_under"],"known_arguments":[arg],"constraints":[],"parser_family":"CLASSIFIED_UNDER"}
    m=re.fullmatch(r"Which film has (.+?) as a member of its cast(?: and (?:has (.+?) as a member of its cast|was directed by (.+?)))?",q,re.I)
    if m:
        args=[x for x in m.groups() if x]
        preds=["cast_member"]+( ["cast_member"] if m.group(2) else ["directed_by"] if m.group(3) else [] )
        return {"answer_slot":"film","predicate_slots":preds,"known_arguments":args,"constraints":[],"parser_family":"FILM_CAST"}
    patterns=[
      (r"Who(?:'s| has)? played for the (.+)","person","played_for"),
      (r"Who was born in (.+)","person","born_in"),(r"What is the name of the person born in (.+)","person","born_in"),
      (r"Who died in (.+)","person","died_in"),(r"Who won the (.+)","person_or_group","won_award"),
      (r"Who received the award (.+)","person_or_group","received_award"),
      (r"Who is or was buried in (.+)","person","buried_in"),
      (r"Which album has (.+) as performer","album","performed_by"),
      (r"Which songs had its lyrics written by (.+)","song","written_by"),
      (r"Which groups were formed in (.+)","group","formed_in"),
      (r"Which spatial entity is located in (.+)","spatial_entity","located_in"),
      (r"Which natural area, domicile or venue is located in (.+)","place","located_in"),
      (r"What sports team played in the league (.+)","sports_team","played_in_league"),
      (r"Who was a member of the political party (.+)","person","party_member"),
      (r"Which competition is a part of (.+)","competition","part_of"),
      (r"Which album is part of the record label (.+)","album","record_label"),
      (r"What's the name of a film with (.+) in it","film","cast_member"),
      (r"(.+) is a cast member in which film","film","cast_member"),
    ]
    for pat,ans,pred in patterns:
        m=re.fullmatch(pat,q,re.I)
        if m:return {"answer_slot":ans,"predicate_slots":[pred],"known_arguments":[m.group(1)],"constraints":[],"parser_family":pred.upper()}
    # Preserve conjunctions as separate required clauses; unsupported clauses
    # remain explicit rather than being discarded.
    if " and Which " in q:
        m=re.fullmatch(r"Which film has (.+?) as a member of its cast and Which musics were composed by (.+)",q,re.I)
        if m:return {"answer_slot":"film","predicate_slots":["cast_member","music_composed_by"],"known_arguments":[m.group(1),m.group(2)],"constraints":[],"parser_family":"FILM_CAST_MUSIC"}
        m=re.fullmatch(r"Which competition is located in (.+?) and Which competitions had (.+?) as participant",q,re.I)
        if m:return {"answer_slot":"competition","predicate_slots":["located_in","participant"],"known_arguments":[m.group(1),m.group(2)],"constraints":[],"parser_family":"COMPETITION_LOCATION_PARTICIPANT"}
        clauses=q.split(" and Which ")
        return {"answer_slot":"shared_entity","predicate_slots":clauses,"known_arguments":[],"constraints":["ALL_CLAUSES_REQUIRED"],"parser_family":"RAW_CONJUNCTION"}
    return None

def main():
    rows=[]
    for r in map(json.loads,IN.open()):
        s=schema(r["question"])
        rows.append({"audit_id":r["audit_id"],"question":r["question"],"decision":"PARSED" if s else "ABSTAIN","schema":s,"source_inspected":False})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"e0a_design_schemas.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    counts=Counter(r["schema"]["parser_family"] if r["schema"] else "ABSTAIN" for r in rows)
    summary={"protocol":"SEM-E0A_QUERY_SCHEMA_DESIGN_AUDIT","items":len(rows),"parsed":sum(r["decision"]=="PARSED" for r in rows),"abstained":sum(r["decision"]=="ABSTAIN" for r in rows),"families":dict(sorted(counts.items())),"question_only":True,"new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False,"status":"AWAITING_SCHEMA_CORRECTNESS_REVIEW"}
    (OUT/"e0a_design_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
