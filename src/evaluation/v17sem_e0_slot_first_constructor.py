"""Closure-constrained exact-span constructor for SEM-E0B design audit."""
import json,re
from pathlib import Path

from src.evaluation.v17sem_b0_single_atom_eligibility import canonical_tokenizer
from src.evaluation.v17sem_e0_query_schema import schema
from src.representation.token_counter import count_tokens

BASE=Path("results/v2_rank_then_cut");IN=BASE/"v17sem_d05_relation_closure_verifier/fresh_audit_items.jsonl";OUT=BASE/"v17sem_e0_slot_first_source_backed"

STOP={"the","a","an","of","in","and","or","their","value","are","is","such","that","which","what","name"}
def toks(s):return [x for x in re.findall(r"[a-z0-9]+",s.lower()) if x not in STOP]
def phrase_group(s):
    z=toks(s)
    while z and z[-1] in {"fc","f","c"}:z.pop()
    return [set(z)] if z else []

PRED={
 "cast_member":[{"stars"},{"starring"},{"features"},{"cast"},{"acted"}],
 "directed_by":[{"directed"}],"played_for":[{"played"},{"playing"}],"played_in_league":[{"competes"},{"played"},{"league"}],
 "born_in":[{"born"}],"died_in":[{"died"}],"won_award":[{"won"}],"received_award":[{"received"},{"awarded"}],
 "buried_in":[{"buried"}],"performed_by":[{"album"}],"written_by":[{"written"},{"lyrics"}],"formed_in":[{"formed"}],
 "located_in":[{"located"},{"jurisdiction"}],"party_member":[{"member"}],"part_of":[{"part"},{"competition"}],
 "record_label":[{"released"},{"records"}],"music_composed_by":[{"music"},{"composed"}],"participant":[{"opposed"},{"participant"}],
 "classified_under":[{"insurer"},{"insurance"}],"created_or_happened_after":[{"created"},{"founded"},{"established"}],
}

TYPE_CUES=[
 (("suburb",),[{"suburb"}]),(("municipalit",),[{"municipality"},{"village"},{"city"},{"town"}]),
 (("nature","center"),[{"nature","center"}]),(("financial","service"),[{"insurer"},{"insurance","company"}]),
 (("composer",),[{"composer"}]),(("explorer",),[{"explorer"}]),(("district",),[{"district"}]),
 (("national","park"),[{"park"},{"seashore"},{"monument"}]),(("wilderness",),[{"wilderness"}]),
 (("airline",),[{"airline"}]),(("tenor",),[{"tenor"}]),(("radio","station"),[{"radio","station"}]),
 (("urban","area"),[{"locality"},{"urban","area"}]),(("mountain",),[{"mountain"}]),
 (("shopping","mall"),[{"mall"},{"shopping","centre"},{"shopping","center"}]),
 (("competition",),[{"competition"},{"event"}]),(("sports","team"),[{"team"}]),
]

def answer_type_groups(s):
    text=" ".join([s["answer_slot"]]+s.get("constraints",[])).lower();out=[]
    for needles,alts in TYPE_CUES:
        if all(x in text for x in needles):out.append(alts)
    return out

def table_pred_groups(raw):
    r=set(toks(raw)); groups=[]
    lex={
      "county":[{"county"}],"region":[{"region"}],"location":[{"located"},{"location"}],"state":[{"state"}],
      "province":[{"province"}],"municipality":[{"municipality"}],"district":[{"district"}],"nationality":[{"german"},{"canadian"},{"english"},{"italian"}],
      "party":[{"party"}],"director":[{"directed"},{"director"}],"performer":[{"album"},{"singer"}],"moving":[{"joined"},{"transferred"},{"signed"}],
      "league":[{"league"},{"competes"}],"formed":[{"formed"},{"founded"}],"license":[{"licensed"}],"canton":[{"canton"}],
      "local":[{"city"},{"jurisdiction"},{"municipality"}],"music":[{"music"},{"composed"}],"lyrics":[{"written"},{"lyrics"}],
    }
    for key,alts in lex.items():
        if key in r:groups.append(alts)
    return groups or [[r]]

def requirements(s):
    groups=[]
    if s["parser_family"]=="CLASSIFIED_UNDER":return None
    for a in s["known_arguments"]:groups.append(phrase_group(a))
    for p in s["predicate_slots"]:
        if s["parser_family"]=="TABLE_COMPOSITION":groups.extend(table_pred_groups(p))
        elif p=="performed_by":groups.extend([[{"album"}],[{"by"},{"singer"},{"artist"},{"band"}]])
        else:groups.append(PRED.get(p,[{p.replace("_"," ")}]))
    groups.extend(answer_type_groups(s))
    # Flatten: every item is a list of alternative token sets.
    return [g for g in groups if g]

def covers(words,groups):
    return all(any(alt <= words for alt in alternatives) for alternatives in groups)

def construct(question,packet,s6,tok):
    s=schema(question)
    if not s:return None,"QUERY_SCHEMA_ERROR"
    lines=packet.splitlines(); title=next((x.removeprefix("Document title: ").strip() for x in lines if x.startswith("Document title: ")),"")
    evidence=" ".join(x.removeprefix("Source evidence: ") for x in lines if x.startswith("Source evidence: "))
    if not title or not evidence:return None,"SOURCE_MATCH_MISS"
    if title.lower() in s6.lower():return None,"NOVELTY_FAIL"
    groups=requirements(s)
    if groups is None:return None,"UNSUPPORTED_SEMANTIC_COMPARISON"
    matches=list(re.finditer(r"\S+",evidence)); best=None
    for i in range(len(matches)):
        for j in range(i,min(len(matches),i+40)):
            span=evidence[matches[i].start():matches[j].end()]; serial=title+" ; "+span; n=count_tokens(tok,serial)
            if n>16:break
            ws=set(toks(span))
            if covers(ws,groups):
                key=(n,j-i,matches[i].start())
                if best is None or key<best[0]:best=(key,span,n,matches[i].start(),matches[j].end())
    if not best:return None,"BUDGET_INFEASIBLE_OR_SLOT_MISSING"
    return {"serialized":title+" ; "+best[1],"title":title,"evidence_quote":best[1],"evidence_offsets":[best[3],best[4]],"tokens":best[2],"schema":s},None

def main():
    tok=canonical_tokenizer(); rows=[]
    for r in map(json.loads,IN.open()):
        c,reason=construct(r["question"],r["rank10_packet"],r["state_s6"],tok)
        rows.append({"audit_id":r["audit_id"],"decision":"EMIT" if c else "ABSTAIN","candidate":c,"failure":reason})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"e0b_design_candidates.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary={"protocol":"SEM-E0B_SLOT_FIRST_DESIGN_AUDIT","items":len(rows),"emissions":sum(r["decision"]=="EMIT" for r in rows),"abstentions":sum(r["decision"]=="ABSTAIN" for r in rows),"new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False,"status":"AWAITING_EXACT_WITNESS_REVIEW"}
    (OUT/"e0b_design_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
