from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.evaluation.v17c0_beam_trajectory_failure_audit import replay
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from src.training.train_v17b1_multi_anchor_viability import load_policy
from src.training.v17b_viability import FIDELITY_GRID


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle: return [json.loads(line) for line in handle]


def action_targets(dp: SequentialTrajectoryDP, history: Sequence[int]) -> dict[int, list[int | None]]:
    mask=0; reached=dp.attained[0]
    for packet in history: mask|=1<<packet; reached=max(reached,dp.attained[mask])
    grid={round(x,2):i for i,x in enumerate(dp.levels)}; result={}
    for packet,value in dp.action_values(mask,reached):
        values=[]
        for level in FIDELITY_GRID:
            local=grid.get(round(level,2)); values.append(None if local is None or local<reached else int(value.reached_levels>=local+1))
        result[packet]=values
    return result


def state_signature(targets: dict[int, list[int | None]], depth: int) -> str:
    return repr((depth,tuple(sorted(tuple(x) for x in targets.values()))))


def trajectory_stats(dp: SequentialTrajectoryDP, history: Sequence[int], target_index: int) -> dict[str, Any]:
    mask=0; reached=dp.attained[0]; cumulative=reached*dp.tokens[0]; first_depth=0 if reached>=target_index+1 else None; first_tokens=dp.tokens[0] if first_depth==0 else None
    for depth,packet in enumerate(history,1):
        mask|=1<<packet; next_reached=max(reached,dp.attained[mask]); cumulative+=(next_reached-reached)*dp.tokens[mask]; reached=next_reached
        if first_depth is None and reached>=target_index+1: first_depth=depth; first_tokens=dp.tokens[mask]
    return {"highest_attained_anchor_count":reached,"target_success":reached>=target_index+1,"target_first_attainment_depth":first_depth,"target_first_attainment_tokens":first_tokens,"cumulative_trajectory_tokens":cumulative}


def main() -> None:
    p=argparse.ArgumentParser(description="V17-D0 canonical harness and two-bottleneck audit")
    p.add_argument("--data",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--exact-dir",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--viability-head",required=True); p.add_argument("--training-artifact",required=True); p.add_argument("--c0-summary",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--hidden-dim",type=int,default=128)
    args=p.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda:0")
    data=list(read_jsonl(args.data)); cache=torch.load(args.embeddings,map_location="cpu",weights_only=True); cache_index={x:i for i,x in enumerate(cache["example_ids"])}
    model=load_policy(args.checkpoint,device,args.hidden_dim); model.viability_head.load_state_dict(torch.load(args.viability_head,map_location=device,weights_only=True)); model.eval()
    train=read_gzip(args.training_artifact); exact_train={(x["example_id"],tuple(x["history"])):x for x in train}; signature_counts=Counter()
    for row in train: signature_counts[state_signature({a["packet"]:a["future_viability"] for a in row["actions"]},row["depth"])]+=1
    wrong=[]; terminals=[]; summaries=[]
    with torch.no_grad():
        for n,source in enumerate(data,1):
            i=cache_index[source["example_id"]]; packets=cache["packets"][i].to(device=device,dtype=torch.float32); question=cache["questions"][i].to(device=device,dtype=torch.float32); fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32); fractions/=fractions.sum()
            exact=list(read_jsonl(Path(args.exact_dir)/f"{source['example_id']}.jsonl",ExactSearchResult)); dp=SequentialTrajectoryDP(exact,trajectory_levels(source)); result=replay(model,packets,question,fractions,dp,residual_enabled=True); target=next(j for j,x in enumerate(dp.levels) if abs(x-.9)<1e-9)
            event=result["first_irreversible_prune"]
            if event is not None and event["residual_margin"]<=0:
                vt=action_targets(dp,event["viable_parent_history"]); bt=action_targets(dp,event["boundary_parent_history"]); vr=exact_train.get((source["example_id"],tuple(event["viable_parent_history"]))); br=exact_train.get((source["example_id"],tuple(event["boundary_parent_history"])))
                vi=event["viable_per_level_logits"]; bi=event["boundary_per_level_logits"]; active=[j for j,(a,b) in enumerate(zip(event["viable_active_target_mask"],event["boundary_active_target_mask"])) if a and b]
                wrong.append({"sample":source["example_id"],"step":event["step"],"target_fidelity":.9,"attainable_levels":list(dp.levels),"viable_parent_history":event["viable_parent_history"],"viable_action":event["viable_action"],"boundary_parent_history":event["boundary_parent_history"],"boundary_action":event["boundary_action"],"viable_action_targets":vt[event["viable_action"]],"boundary_action_targets":bt[event["boundary_action"]],"viable_per_level_logits":vi,"boundary_per_level_logits":bi,"per_level_logit_margin":[a-b for a,b in zip(vi,bi)],"active_in_both":active,"raw_residual_margin":event["viable_raw_residual_logit"]-event["boundary_raw_residual_logit"],"normalized_cumulative_residual_margin":event["residual_margin"],"base_cumulative_margin":event["base_margin"],"total_cumulative_margin":event["total_margin"],"feature_cosine_similarity":event["feature_cosine_similarity"],"feature_l2":event["feature_l2"],"viable_training_provenance":None if vr is None else vr["provenance"],"boundary_training_provenance":None if br is None else br["provenance"],"viable_seen_exact_state":vr is not None,"boundary_seen_exact_state":br is not None,"viable_equivalent_training_states":signature_counts[state_signature(vt,len(event["viable_parent_history"]))],"boundary_equivalent_training_states":signature_counts[state_signature(bt,len(event["boundary_parent_history"]))],"viable_parent_primary_positive_actions":sum(x[target]==1 for x in vt.values()),"viable_parent_primary_negative_actions":sum(x[target]==0 for x in vt.values()),"boundary_parent_primary_positive_actions":sum(x[target]==1 for x in bt.values()),"boundary_parent_primary_negative_actions":sum(x[target]==0 for x in bt.values())})
            for rank,candidate in enumerate(result["terminal"],1):
                stats=trajectory_stats(dp,candidate["history"],target); terminals.append({"sample":source["example_id"],"rank_under_search_score":rank,**candidate,**stats})
            summaries.append({"sample":source["example_id"],"top1_success":result["top1_success"],"any_terminal_viable":result["any_terminal_viable"],"first_irreversible_prune_step":None if event is None else event["step"]})
            if n%50==0: print(json.dumps({"processed":n,"total":len(data)}),flush=True)
    write_jsonl(out/"wrong_direction_events.jsonl",wrong); write_jsonl(out/"terminal_candidates.jsonl",terminals)
    natural_failures=[x for x in summaries if not x["top1_success"]]; natural_terminal=[x for x in natural_failures if x["any_terminal_viable"]]
    same_target=sum(x["viable_action_targets"][3]==1 and x["boundary_action_targets"][3]==0 for x in wrong)
    primary_logit_correct=sum(x["per_level_logit_margin"][3]>0 for x in wrong); aggregate_conflict=sum(x["per_level_logit_margin"][3]>0 and x["raw_residual_margin"]<=0 for x in wrong)
    result={"complete":True,"canonical_harness":{"causal_baseline_primary_090":"residual-off 279/300","historical_deployed_reference":"online-bfloat16 V8 280/300; not causal repair/break baseline","embedding_mode":"frozen cached V8 packets/questions","embedding_dtype":str(cache["packets"].dtype),"head_dtype":"torch.float32","beam_width":8,"candidate_order":"packet index ascending before score/history deterministic sort","tie_breaking":"descending cumulative normalized log probability, then lexicographic history","terminal_selection":"highest cumulative search score"},"wrong_direction":{"events":len(wrong),"exact_primary_label_separation":same_target,"primary_logit_favors_viable":primary_logit_correct,"primary_logit_correct_but_aggregate_wrong":aggregate_conflict,"both_states_seen_exact":sum(x["viable_seen_exact_state"] and x["boundary_seen_exact_state"] for x in wrong),"mean_feature_cosine_similarity":mean(x["feature_cosine_similarity"] for x in wrong),"median_feature_l2":median(x["feature_l2"] for x in wrong)},"terminal":{"standard_beam_primary_failures":len(natural_failures),"failures_with_naturally_surviving_viable_terminal":len(natural_terminal),"failures_requiring_survival_repair_before_reranking":len(natural_failures)-len(natural_terminal),"standalone_terminal_reranker_ceiling_on_current_failures":len(natural_terminal)},"sealed":{"development":True,"hop2":True,"learned_cutoff":True,"fresh_confirmation":True,"training":True},"artifacts":{"data_sha256":sha256(args.data),"embeddings_sha256":sha256(args.embeddings),"checkpoint_head_sha256":sha256(Path(args.checkpoint)/"sequential_head.pt"),"viability_head_sha256":sha256(args.viability_head),"training_artifact_sha256":sha256(args.training_artifact)}}
    write_metadata(out/"summary.json",result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
