from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.evaluation.v17c0_beam_trajectory_failure_audit import replay
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from src.training.train_v17b1_multi_anchor_viability import load_policy


MODES=("residual_off","current","force_target_090","target_local_090","force_target_local_090","oracle_unresolved_target_local_090")


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path,"rt",encoding="utf-8") as handle:return [json.loads(line) for line in handle]


def strict_signature(row: dict[str, Any]) -> str:
    counts=[]
    for anchor in range(5):
        values=[a["future_viability"][anchor] for a in row["actions"] if a["future_viability"][anchor] is not None]
        counts.append((sum(v==1 for v in values),sum(v==0 for v in values)))
    return repr((tuple(row["anchor_status"]),row["reached_level_count"],row["depth"],len(row["actions"]),tuple(counts)))


def metric_for_order(source: dict[str,Any], exact: list[ExactSearchResult], order: list[int]) -> dict[str,Any]:
    levels=trajectory_levels(source); learned=best_prefix_nested_chain(exact,order,levels); oracle=best_binary_nested_chain(exact,levels); success={str(level):bool(x["feasible"]) for level,x in zip(levels,learned)}; complete=all(success.values()); regret=None
    if complete:
        full=next(x.tokens for x in exact if all(x.state)); regret=(sum(int(x["tokens"]) for x in learned)-sum(int(x["tokens"]) for x in oracle))/(len(levels)*full)
    return {"per_level":success,"complete":complete,"regret":regret}


def summarize(rows:list[dict[str,Any]], baseline:dict[str,dict[str,Any]])->dict[str,Any]:
    levels=sorted({k for x in rows for k in x["metrics"]["per_level"]},key=float); per={}
    for level in levels:
        applicable=[x for x in rows if level in x["metrics"]["per_level"]]; repairs=sum(x["metrics"]["per_level"][level] and not baseline[x["sample"]]["per_level"][level] for x in applicable); breaks=sum(not x["metrics"]["per_level"][level] and baseline[x["sample"]]["per_level"][level] for x in applicable)
        per[level]={"examples":len(applicable),"successes":sum(x["metrics"]["per_level"][level] for x in applicable),"repairs_vs_residual_off":repairs,"breaks_vs_residual_off":breaks}
    regrets=[x["metrics"]["regret"] for x in rows if x["metrics"]["regret"] is not None]
    return {"per_level":per,"complete_successes":sum(x["metrics"]["complete"] for x in rows),"mean_complete_regret":mean(regrets),"primary_failure_ids":sorted(x["sample"] for x in rows if not x["metrics"]["per_level"]["0.9"]),"first_prune_events":sum(x["first_prune"] is not None for x in rows),"median_minimum_rescue_margin":median(x["first_prune"]["minimum_rescue_margin"] for x in rows if x["first_prune"] is not None)}


def main()->None:
    p=argparse.ArgumentParser(description="V17-D1A mask, target-local, and boundary-coverage audit")
    p.add_argument("--data",required=True);p.add_argument("--embeddings",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--viability-head",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--d0-wrong-events",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--hidden-dim",type=int,default=128)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0");data=list(read_jsonl(a.data));cache=torch.load(a.embeddings,map_location="cpu",weights_only=True);cache_index={x:i for i,x in enumerate(cache["example_ids"])};wrong={x["sample"]:x for x in read_gzip(a.d0_wrong_events)}
    model=load_policy(a.checkpoint,device,a.hidden_dim);model.viability_head.load_state_dict(torch.load(a.viability_head,map_location=device,weights_only=True));model.eval();outputs={mode:[] for mode in MODES};fixed=[]
    with torch.no_grad():
        for n,source in enumerate(data,1):
            idx=cache_index[source["example_id"]];packets=cache["packets"][idx].to(device=device,dtype=torch.float32);question=cache["questions"][idx].to(device=device,dtype=torch.float32);fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32);fractions/=fractions.sum();exact=list(read_jsonl(Path(a.exact_dir)/f"{source['example_id']}.jsonl",ExactSearchResult));dp=SequentialTrajectoryDP(exact,trajectory_levels(source))
            current_event=None
            for mode in MODES:
                result=replay(model,packets,question,fractions,dp,residual_enabled=mode!="residual_off",scoring_mode="current" if mode=="residual_off" else mode);metrics=metric_for_order(source,exact,result["decoded_order"]);outputs[mode].append({"sample":source["example_id"],"metrics":metrics,"first_prune":result["first_irreversible_prune"]})
                if mode=="current":current_event=result["first_irreversible_prune"]
            if source["example_id"] in wrong:
                event=current_event;zv=event["viable_per_level_logits"];zb=event["boundary_per_level_logits"];mv=event["viable_active_target_mask"];mb=event["boundary_active_target_mask"];scale=float(model.viability_head.residual_scale)
                def aggregate(logits,mask):return scale*sum(z for z,m in zip(logits,mask) if m)/max(1,sum(mask))
                mode_margin={"current":aggregate(zv,mv)-aggregate(zb,mb),"force_target_090":aggregate(zv,[*mv[:3],True,*mv[4:]])-aggregate(zb,[*mb[:3],True,*mb[4:]]),"target_local_090":scale*((zv[3] if mv[3] else 0)-(zb[3] if mb[3] else 0)),"force_target_local_090":scale*(zv[3]-zb[3])}
                fixed.append({"sample":source["example_id"],"mode_raw_residual_margin":mode_margin,"direction_correct":{k:v>0 for k,v in mode_margin.items()}})
            if n%25==0:print(json.dumps({"processed":n,"total":len(data)}),flush=True)
    baseline={x["sample"]:x["metrics"] for x in outputs["residual_off"]};summaries={mode:summarize(rows,baseline) for mode,rows in outputs.items()};base_b1={x["sample"]:x["metrics"] for x in outputs["current"]}
    for mode,rows in outputs.items():
        if mode in {"residual_off","current"}:continue
        summary=summaries[mode];summary["primary_repairs_vs_current_b1"]=sum(x["metrics"]["per_level"]["0.9"] and not base_b1[x["sample"]]["per_level"]["0.9"] for x in rows);summary["primary_breaks_vs_current_b1"]=sum(not x["metrics"]["per_level"]["0.9"] and base_b1[x["sample"]]["per_level"]["0.9"] for x in rows)
    train=read_gzip(a.training_artifact);signature=Counter(strict_signature(x) for x in train);boundary=Counter();pair_count=0
    for row in train:
        values=[a["future_viability"][3] for a in row["actions"] if a["future_viability"][3] is not None];pos=sum(x==1 for x in values);neg=sum(x==0 for x in values)
        if pos and neg: boundary[(row["provenance"],strict_signature(row))]+=1;pair_count+=pos*neg
    coverage={"allowed_states":len(train),"strict_primary_boundary_states":sum(boundary.values()),"strict_primary_action_pairs":pair_count,"by_provenance":dict(Counter(k[0] for k,v in boundary.items() for _ in range(v))),"unique_strict_signatures":len({k[1] for k in boundary})}
    write_jsonl(out/"fixed_event_counterfactuals.jsonl",fixed);write_jsonl(out/"canonical_mode_results.jsonl",[{"mode":m,**s} for m,s in summaries.items()]);write_metadata(out/"boundary_coverage.json",coverage)
    result={"complete":True,"modes":summaries,"fixed_wrong_direction_sign_accuracy":{mode:sum(x["direction_correct"][mode] for x in fixed) for mode in ("current","force_target_090","target_local_090","force_target_local_090")},"boundary_coverage":coverage,"sealed":{"development":True,"hop2":True,"training":True,"trajectory_loss":True,"terminal_reranker":True,"continuous_cost":True,"learned_cutoff":True,"confirmation":True},"artifacts":{"data_sha256":sha256(a.data),"embeddings_sha256":sha256(a.embeddings),"viability_head_sha256":sha256(a.viability_head),"training_artifact_sha256":sha256(a.training_artifact)}};write_metadata(out/"summary.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
