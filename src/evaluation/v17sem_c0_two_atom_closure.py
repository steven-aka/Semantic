"""Build and score the paired-adjacent-atom SEM-C0 closure audit."""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments
from src.evaluation.v17sem_b0_single_atom_eligibility import canonical_tokenizer, validate_annotation
from src.representation.token_counter import count_tokens

BASE=Path("results/v2_rank_then_cut")
B0=BASE/"v17sem_b0_single_atom_eligibility"
ROOT=BASE/"v17sel_b2b_lineage_clean_holdout"
OUT=BASE/"v17sem_c0_two_atom_closure"


def build():
    manifest=json.loads((B0/"manifest.json").read_text());ids=manifest["query_ids"]
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    m0b={r["example_id"]:r for r in read_jsonl(BASE/"v17traj_m0b_borrow_preflight/per_query.jsonl") if r["example_id"] in ids}
    tok=canonical_tokenizer();rows=[]
    for audit_id,q in enumerate(ids,1):
        packet=data[q]["packet_texts"][orders[q][9]];atoms=segments(packet);i=m0b[q]["candidate_index"]
        j=i+1 if i+1<len(atoms) else i-1
        assert 0<=j<len(atoms) and j!=i
        source=atoms[i]+"\n\n"+atoms[j]
        rows.append({"audit_id":audit_id,"question":data[q]["question"],"state_s6":next(r["state_s6"] for r in read_jsonl(B0/"audit_items.jsonl") if r["audit_id"]==audit_id),
                     "primary_atom":atoms[i],"adjacent_atom":atoms[j],"adjacent_index":j,
                     "future_atom":source,"source_tokens_qwen3_8b":count_tokens(tok,source)})
    return ids,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--score",action="store_true");args=ap.parse_args()
    ids,rows=build();OUT.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256("\n".join(json.dumps(r,sort_keys=True) for r in rows).encode()).hexdigest()
    manifest={"protocol":"SEM-C0_FIXED_ADJACENT_TWO_ATOM_CLOSURE","query_ids":ids,"items":64,
              "adjacent_rule":"next atom, or previous when primary is last","content_sha256":digest,
              "gold_or_outcomes_loaded":False,"new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (OUT/"audit_items.jsonl").open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    if not args.score:
        print(json.dumps({"items":64,"content_sha256":digest,"status":"AWAITING_AI_ASSISTED_REVIEW"},indent=2));return
    annotations=list(read_jsonl(OUT/"annotation.jsonl"));assert len(annotations)==64
    tok=canonical_tokenizer();byid={r["audit_id"]:r for r in rows}
    bad={r["audit_id"]:validate_annotation(byid[r["audit_id"]],r,tok) for r in annotations};bad={k:v for k,v in bad.items() if v}
    if bad:raise ValueError(bad)
    eligible=sum(r["label"]=="ELIGIBLE" for r in annotations)
    reasons={k:sum(r.get("abstain_reason")==k for r in annotations) for k in sorted({r.get("abstain_reason") for r in annotations if r.get("abstain_reason")})}
    summary={"protocol":manifest["protocol"],"items":64,"eligible":eligible,"abstain":64-eligible,"abstain_reasons":reasons,
             "review":"AI-assisted primary plus adversarial audit; no independent-human IAA claim",
             "single_atom_eligible":json.loads((B0/"single_review_summary.json").read_text())["eligible"],
             "eligibility_gain_over_single_atom":eligible-json.loads((B0/"single_review_summary.json").read_text())["eligible"],
             "decision":"GO_IDEAL_SEMANTIC_REPRESENTATION_POSITIVE_CONTROL" if eligible>=24 else "STOP_OR_EXPAND_CLOSURE",
             "new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
