"""Summarize the frozen SEM-D0 relation-closure failure taxonomy.

This is a design-set audit.  It neither trains a verifier nor calls Teacher/Target.
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path("results/v2_rank_then_cut")
SRC = BASE / "v17sem_d0_automatic_grounded_extractive"
OUT = BASE / "v17sem_d05_relation_closure_verifier"

# One primary mechanism per invalid emission. Secondary mechanisms are retained
# separately because truncation, wrong predicates, and unresolved references can
# co-occur. Categories describe the emitted candidate, not the source-pair oracle.
PRIMARY = {
    1:"WRONG_PREDICATE",2:"WRONG_PREDICATE",4:"MISSING_ARGUMENT",5:"MISSING_PREDICATE",
    6:"CROSS_SPAN_UNRESOLVED",7:"NO_RELEVANT_FACT",8:"NO_RELEVANT_FACT",12:"MISSING_PREDICATE",
    15:"NO_RELEVANT_FACT",16:"MISSING_ARGUMENT",17:"CONSTRAINT_MISSING",19:"CONSTRAINT_MISSING",
    23:"MISSING_PREDICATE",24:"MISSING_PREDICATE",26:"WRONG_PREDICATE",27:"NO_RELEVANT_FACT",
    29:"NO_RELEVANT_FACT",30:"MISSING_ARGUMENT",31:"NO_RELEVANT_FACT",32:"NO_RELEVANT_FACT",
    33:"MISSING_ARGUMENT",35:"MISSING_PREDICATE",37:"NO_RELEVANT_FACT",38:"MISSING_ARGUMENT",
    40:"MISSING_ARGUMENT",41:"NO_RELEVANT_FACT",42:"MISSING_PREDICATE",43:"MISSING_ARGUMENT",
    46:"WRONG_PREDICATE",47:"NO_RELEVANT_FACT",48:"MISSING_PREDICATE",49:"AMBIGUOUS_RELATION",
    50:"MISSING_ARGUMENT",52:"NO_RELEVANT_FACT",53:"CONSTRAINT_MISSING",54:"NO_RELEVANT_FACT",
    55:"MISSING_PREDICATE",56:"MISSING_ARGUMENT",57:"NO_RELEVANT_FACT",59:"NO_RELEVANT_FACT",
    61:"WRONG_PREDICATE",62:"WRONG_PREDICATE",63:"AMBIGUOUS_RELATION",64:"CONSTRAINT_MISSING",
    66:"MISSING_PREDICATE",67:"NO_RELEVANT_FACT",70:"CONSTRAINT_MISSING",75:"MISSING_ARGUMENT",
    76:"MISSING_ARGUMENT",80:"MISSING_PREDICATE",81:"WRONG_PREDICATE",82:"MISSING_PREDICATE",
    88:"MISSING_ARGUMENT",91:"WRONG_PREDICATE",94:"MISSING_ARGUMENT",96:"AMBIGUOUS_RELATION",
}

TRUNCATED = {1,2,4,5,12,16,17,23,24,30,33,35,38,40,42,43,48,50,55,56,70,75,76,80,82,88,94}
MULTI_CONSTRAINT = {1,4,7,16,17,19,26,27,29,30,31,32,33,46,47,50,52,53,54,57,59,62,63,64,70,75,80,91,94}

def read(name):
    return {r["audit_id"]: r for r in map(json.loads, (SRC/name).open())}

def wilson(k,n,z=1.96):
    if not n:return [0.0,0.0]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return [c-h,c+h]

def main():
    items, policy, audit = read("audit_items.jsonl"), read("automatic_policy.jsonl"), read("automatic_audit.jsonl")
    invalid=sorted(k for k,r in audit.items() if not r["valid_compact_evidence"])
    assert invalid == sorted(PRIMARY), (set(invalid)-set(PRIMARY),set(PRIMARY)-set(invalid))
    rows=[]
    for aid in invalid:
        rows.append({
            "audit_id":aid,
            "question":items[aid]["question"],
            "candidate":policy[aid]["candidate"]["serialized"],
            "primary_failure":PRIMARY[aid],
            "truncation_contributed":aid in TRUNCATED,
            "multi_constraint_query":aid in MULTI_CONSTRAINT,
            "provenance":"AI_ASSISTED_MECHANISM_TAXONOMY; no human IAA"
        })
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"d0_error_taxonomy.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    counts=Counter(r["primary_failure"] for r in rows)
    summary={
        "protocol":"SEM-D0.5A_RELATION_CLOSURE_ERROR_TAXONOMY",
        "design_exposed_items":96,
        "invalid_emissions":len(rows),
        "primary_failure_counts":dict(sorted(counts.items())),
        "truncation_contributed":sum(r["truncation_contributed"] for r in rows),
        "multi_constraint_queries":sum(r["multi_constraint_query"] for r in rows),
        "interpretation":"The dominant failure is incomplete or irrelevant relation expression, not source provenance. Exact spans alone do not establish relation closure.",
        "decision":"PROCEED_TO_FRESH_RELATION_CLOSURE_GATE_WITH_CONSERVATIVE_ABSTENTION",
        "new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False,
        "human_iaa":"NOT_AVAILABLE"
    }
    (OUT/"taxonomy_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
