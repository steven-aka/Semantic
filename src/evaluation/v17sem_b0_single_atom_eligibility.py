"""Build and score the frozen human SEM-B0 single-atom eligibility audit."""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens, load_tokenizer
from src.teacher.packet_validator import numeric_concepts

BASE=Path("results/v2_rank_then_cut")
ROOT=BASE/"v17sel_b2b_lineage_clean_holdout"
OUT=BASE/"v17sem_b0_single_atom_eligibility"
CFG=Path("configs/v17sem_b0_single_atom_eligibility.json")


def canonical_tokenizer():
    """Load the frozen local tokenizer without requiring the CUDA torch wheel."""
    try:
        return load_tokenizer("models/Qwen3-8B")
    except RuntimeError:
        from tokenizers import Tokenizer
        raw=Tokenizer.from_file("models/Qwen3-8B/tokenizer.json")
        class LocalTokenizer:
            def encode(self,text,add_special_tokens=False):
                return raw.encode(text,add_special_tokens=add_special_tokens).ids
        return LocalTokenizer()


def choose(cfg):
    m0b={r["example_id"]:r for r in read_jsonl(BASE/"v17traj_m0b_borrow_preflight/per_query.jsonl") if r["eligible"]}
    excluded=set()
    for name in ("v17traj_m0c_borrow_pilot","v17traj_m1_low_cost_borrow","v17traj_m2b_promote_delay_pilot","v17packet_obs_a1_natural_insertion","v17packet_frag_b0_title_only_natural_pilot"):
        excluded.update(json.loads((BASE/name/"manifest.json").read_text())["query_ids"])
    pool=sorted(set(m0b)-excluded,key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))
    used=set(json.loads((BASE/"v17sem_a0b_query_conditioned_positive_control/manifest.json").read_text())["query_ids"])
    used.update(json.loads((BASE/"v17sem_a0b_query_conditioned_positive_control_v2/manifest.json").read_text())["query_ids"])
    remaining=[q for q in pool if q not in used]
    assert len(pool)==235 and len(remaining)==171
    return remaining[:cfg["queries"]],m0b


def build(cfg):
    chosen,m0b=choose(cfg)
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen}
    orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen}
    assert len(data)==len(orders)==len(chosen)==64
    tokenizer=canonical_tokenizer()
    rows=[]
    for index,q in enumerate(chosen,1):
        packets,order=data[q]["packet_texts"],orders[q]
        atom=segments(packets[order[9]])[m0b[q]["candidate_index"]]
        rows.append({"audit_id":index,"question":data[q]["question"],"state_s6":render(packets,mask(order,6)),
                     "future_atom":atom,"future_atom_tokens_qwen3_8b":count_tokens(tokenizer,atom)})
    return chosen,rows


def validate_annotation(source,row,tokenizer):
    allowed={"ELIGIBLE","ABSTAIN"}; label=row.get("label")
    errors=[]
    if label not in allowed:errors.append("label must be ELIGIBLE or ABSTAIN")
    if label=="ELIGIBLE":
        quote=(row.get("exact_quote") or "").strip();fact=(row.get("canonical_fact") or "").strip()
        if not quote or quote not in source["future_atom"]:errors.append("exact_quote is not a nonempty exact substring")
        if not fact:errors.append("canonical_fact empty")
        if count_tokens(tokenizer,fact)>16:errors.append("canonical_fact exceeds 16 tokens")
        if numeric_concepts(fact)-numeric_concepts(source["future_atom"]):errors.append("fact has source-absent numeric concept")
        if not (row.get("relevance_reason") or "").strip():errors.append("relevance_reason empty")
        if not (row.get("novelty_reason") or "").strip():errors.append("novelty_reason empty")
    if label=="ABSTAIN" and row.get("abstain_reason") not in {"NO_RELEVANT_FACT","ALREADY_IN_S6","RELATION_NOT_CLOSED","CANNOT_FIT_16_TOKENS","AMBIGUOUS_GROUNDING"}:
        errors.append("invalid abstain_reason")
    return errors


def kappa(a,b):
    n=len(a);po=sum(x==y for x,y in zip(a,b))/n
    pa=sum(x=="ELIGIBLE" for x in a)/n;pb=sum(x=="ELIGIBLE" for x in b)/n
    pe=pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe<1 else 1.0


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--score",action="store_true");ap.add_argument("--single-review",action="store_true");args=ap.parse_args()
    cfg=json.loads(CFG.read_text());assert cfg["status"]=="FROZEN_BEFORE_ANNOTATION"
    chosen,rows=build(cfg);OUT.mkdir(parents=True,exist_ok=True)
    manifest={"protocol":cfg["protocol"],"query_ids":chosen,"items":64,
              "content_sha256":hashlib.sha256("\n".join(json.dumps(r,sort_keys=True) for r in rows).encode()).hexdigest(),
              "gold_or_outcomes_loaded":False,"sealed_sets_read":False}
    mp=OUT/"manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    source_path=OUT/"audit_items.jsonl"
    with source_path.open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")
    for name in ("annotator_a.jsonl","annotator_b.jsonl"):
        path=OUT/name
        if not path.exists():
            with path.open("w") as h:
                for r in rows:h.write(json.dumps({"audit_id":r["audit_id"],"label":None,"exact_quote":None,"canonical_fact":None,"relevance_reason":None,"novelty_reason":None,"abstain_reason":None,"notes":None},ensure_ascii=False)+"\n")
    if args.single_review:
        tokenizer=canonical_tokenizer();byid={r["audit_id"]:r for r in rows}
        values=list(read_jsonl(OUT/"annotator_a.jsonl"));assert len(values)==64
        bad={r["audit_id"]:validate_annotation(byid[r["audit_id"]],r,tokenizer) for r in values}
        bad={k:v for k,v in bad.items() if v}
        if bad:raise ValueError(f"annotator_a.jsonl invalid: {bad}")
        eligible=sum(r["label"]=="ELIGIBLE" for r in values)
        reasons={k:sum(r.get("abstain_reason")==k for r in values) for k in sorted({r.get("abstain_reason") for r in values if r.get("abstain_reason")})}
        route="SEM_B1_IDEAL_REPRESENTATION_POSITIVE_CONTROL" if eligible>=24 else ("STOP_SINGLE_ATOM_CONTRACT" if eligible<16 else "GRAY_ZONE_MECHANISM_AUDIT")
        summary={"protocol":"SEM-B0R_AI_ASSISTED_SINGLE_REVIEW","items":64,"eligible":eligible,"abstain":64-eligible,
                 "abstain_reasons":reasons,"human_iaa":"NOT_AVAILABLE","cohen_kappa":None,
                 "formal_human_gate_claimed":False,"decision_threshold_only":route,
                 "new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
        (OUT/"single_review_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return
    if not args.score:
        print(json.dumps({"protocol":cfg["protocol"],"items":64,"status":"AWAITING_TWO_INDEPENDENT_HUMAN_ANNOTATIONS","content_sha256":manifest["content_sha256"]},indent=2));return
    tokenizer=canonical_tokenizer();byid={r["audit_id"]:r for r in rows};annotations=[]
    for name in ("annotator_a.jsonl","annotator_b.jsonl"):
        values=list(read_jsonl(OUT/name));assert len(values)==64
        errors={r["audit_id"]:validate_annotation(byid[r["audit_id"]],r,tokenizer) for r in values}
        bad={k:v for k,v in errors.items() if v}
        if bad:raise ValueError(f"{name} invalid: {bad}")
        annotations.append(values)
    a,b=annotations;labels_a=[r["label"] for r in a];labels_b=[r["label"] for r in b]
    agreement=sum(x==y for x,y in zip(labels_a,labels_b))/64;kap=kappa(labels_a,labels_b)
    direct_eligible=sum(x==y=="ELIGIBLE" for x,y in zip(labels_a,labels_b))
    summary={"protocol":cfg["protocol"],"items":64,"raw_label_agreement":agreement,"cohen_kappa":kap,
             "annotator_a_eligible":labels_a.count("ELIGIBLE"),"annotator_b_eligible":labels_b.count("ELIGIBLE"),
             "direct_agreement_eligible":direct_eligible,"disagreements":sum(x!=y for x,y in zip(labels_a,labels_b)),
             "reliability_gate_passed":agreement>=.85 and kap>=.70,
             "status":"AWAITING_ADJUDICATION" if agreement<1 else "READY_FOR_FINAL_GATE",
             "new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    (OUT/"pre_adjudication_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
