"""Build and score the fresh SEM-D0.5 relation-closure gate."""
import argparse, hashlib, json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask,render,segments
from src.evaluation.v17sem_b0_single_atom_eligibility import canonical_tokenizer
from src.evaluation.v17sem_d0_automatic_grounded_extractive import automatic_candidate
from src.evaluation.v17sem_d05_conservative_verifier import decide,wilson

BASE=Path("results/v2_rank_then_cut"); ROOT=BASE/"v17sel_b2b_lineage_clean_holdout"
OUT=BASE/"v17sem_d05_relation_closure_verifier"

def all_manifest_ids():
    seen=set()
    for p in BASE.rglob("manifest.json"):
        try:o=json.loads(p.read_text())
        except Exception:continue
        for key in ("query_ids","example_ids"):
            if isinstance(o.get(key),list):seen.update(map(str,o[key]))
    return seen

def build():
    data=list(read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl")); orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl")}
    unseen=[r for r in data if str(r["example_id"]) not in all_manifest_ids() and r["example_id"] in orders and segments(r["packet_texts"][orders[r["example_id"]][9]])]
    unseen.sort(key=lambda r:(hashlib.sha256(("SEM-D05-FRESH|"+str(r["example_id"])).encode()).hexdigest(),str(r["example_id"])))
    chosen=unseen[:96]; tok=canonical_tokenizer(); items=[]; policy=[]
    for aid,r in enumerate(chosen,1):
        q=r["example_id"]; order=orders[q]; packet=r["packet_texts"][order[9]]; atoms=segments(packet); s6=render(r["packet_texts"],mask(order,6)); candidates=[]
        pairs=[(i,i+1) for i in range(max(0,len(atoms)-1))] or [(0,0)]
        for i,j in pairs:
            c,reason=automatic_candidate(r["question"],s6,atoms[i],atoms[j],tok)
            if c:candidates.append((c["query_overlap"],c["tokens"],-i,c,i,j))
        if candidates:
            _,_,_,cand,i,j=max(candidates,key=lambda x:x[:3]); raw="\n\n".join([atoms[i]] if i==j else [atoms[i],atoms[j]])
            verified,schema=decide(r["question"],cand["serialized"]); decision="EMIT" if verified else "ABSTAIN"
            policy.append({"audit_id":aid,"decision":decision,"schema":schema,"candidate":cand,"source_atom_indices":[i,j],"generator_emitted":True})
        else:
            raw="\n\n".join(atoms[:2]);policy.append({"audit_id":aid,"decision":"ABSTAIN","schema":"NO_GENERATED_CANDIDATE","candidate":None,"source_atom_indices":None,"generator_emitted":False})
        items.append({"audit_id":aid,"example_id":q,"question":r["question"],"state_s6":s6,"rank10_packet":packet,"selected_source_pair":raw})
    return items,policy,len(unseen)

def write_jsonl(path,rows):
    with path.open("w") as h:
        for r in rows:h.write(json.dumps(r,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--score",action="store_true");args=ap.parse_args();items,policy,n=build();OUT.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256("\n".join(json.dumps(r,sort_keys=True) for r in items).encode()).hexdigest()
    manifest={"protocol":"SEM-D0.5_FRESH_RELATION_CLOSURE_GATE","query_ids":[r["example_id"] for r in items],"items":96,"unseen_canonical_pool":n,"content_sha256":digest,"candidate_source":"all adjacent atom pairs in V8 rank10 packet","selection":"SHA-ordered queries; label-blind lexical candidate generation; frozen conservative verifier","new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False}
    mp=OUT/"fresh_manifest.json"
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+"\n")
    write_jsonl(OUT/"fresh_audit_items.jsonl",items);write_jsonl(OUT/"fresh_policy.jsonl",policy)
    if not args.score:
        print(json.dumps({"items":96,"unseen_pool":n,"generator_emissions":sum(x["generator_emitted"] for x in policy),"verified_emissions":sum(x["decision"]=="EMIT" for x in policy),"status":"AWAITING_AI_ASSISTED_BLIND_ANNOTATION"},indent=2));return
    ann={r["audit_id"]:r for r in read_jsonl(OUT/"fresh_annotation.jsonl")};assert set(ann)==set(range(1,97))
    pol={r["audit_id"]:r for r in policy}; eligible=sum(r["source_universe_eligible"] for r in ann.values()); emitted=[p for p in policy if p["decision"]=="EMIT"]
    tp=sum(ann[p["audit_id"]]["emitted_candidate_valid"] for p in emitted); prec=tp/len(emitted) if emitted else 0; recall=tp/eligible if eligible else 0; cov=len(emitted)/96; ci=wilson(tp,len(emitted))
    gate=prec>=.95 and ci[0]>=.80 and recall>=.50 and len(emitted)>=20 and cov>=.15
    summary={"protocol":"SEM-D0.5_FRESH_RELATION_CLOSURE_GATE","items":96,"source_universe_eligible":eligible,"generator_emissions":sum(x["generator_emitted"] for x in policy),"verified_emissions":len(emitted),"valid_verified_emissions":tp,"precision":prec,"precision_wilson_95":ci,"eligible_recall":recall,"coverage":cov,"decision":"GO_SEM_D1_FRESH_TARGET" if gate else "STOP_OR_REDESIGN_RELATION_CLOSURE","new_teacher_calls":0,"new_target_calls":0,"sealed_sets_read":False,"human_iaa":"NOT_AVAILABLE"}
    (OUT/"fresh_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
