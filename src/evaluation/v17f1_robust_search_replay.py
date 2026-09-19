from __future__ import annotations

import argparse
import hashlib
import json
import random
from types import SimpleNamespace
from pathlib import Path
from statistics import mean

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.evaluation.v17c0_beam_trajectory_failure_audit import replay
from src.evaluation.v17d1a_scoring_counterfactual_audit import metric_for_order
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17b1_multi_anchor_viability import load_policy, read_gzip


class ReachabilityDP:
    """Only the target reachability fields needed by the no-training replay."""
    def __init__(self, exact, levels):
        self.levels=tuple(levels);size=1<<len(exact[0].state);fidelity=[0.0]*size
        for row in exact:fidelity[state_to_mask(row.state)]=float(row.fidelity)
        self.attained=tuple(sum(value+1e-12>=level for level in self.levels) for value in fidelity);primary=next(i for i,level in enumerate(self.levels) if abs(level-.9)<1e-9);reach=[value+1e-12>=.9 for value in fidelity]
        for count in range(len(exact[0].state)-1,-1,-1):
            for mask in range(size):
                if mask.bit_count()==count and not reach[mask]:reach[mask]=any(not(mask&(1<<packet)) and reach[mask|(1<<packet)] for packet in range(len(exact[0].state)))
        self.reach=reach;self.primary=primary
    def value(self,mask,reached):return SimpleNamespace(reached_levels=self.primary+1 if reached>self.primary or self.reach[mask] else 0)


def fast_replay(model,packets,question,fractions,dp,rule,seed,beam_width=8):
    primary=next(i for i,level in enumerate(dp.levels) if abs(level-.9)<1e-9);beams=[(0.0,(),torch.tanh(model.initial_history(question[None]))[0],0,dp.attained[0])];first_prune=False
    for depth in range(1,13):
        count=len(beams);states=torch.stack([row[2] for row in beams]);selected=torch.zeros((count,12),dtype=torch.bool,device=packets.device)
        for i,row in enumerate(beams):
            if row[1]:selected[i,list(row[1])]=True
        ps=packets[None].expand(count,-1,-1);qs=question[None].expand(count,-1);indices=torch.arange(count,device=packets.device);fs=fractions[None].expand(count,-1);features,progress=model.action_features(ps,qs,states,selected,indices,fs);base=model.action_head(features).squeeze(-1).masked_fill(selected,-torch.inf);active=model.deployment_active_target_mask(progress,torch.full((count,),len(dp.levels),device=packets.device)).clone();active[:,:,3]=True;_,residual=model.viability_head(features,active);logp=torch.log_softmax((base+residual).float(),dim=-1).cpu();next_states=model.history_gru(ps.reshape(-1,model.model_dim),states[:,None,:].expand(-1,12,-1).reshape(-1,model.model_dim)).reshape(count,12,model.model_dim);expanded=[]
        for i,(total,history,_,mask,reached) in enumerate(beams):
            for packet in range(12):
                if selected[i,packet]:continue
                next_mask=mask|(1<<packet);next_reached=max(reached,dp.attained[next_mask]);expanded.append((total+float(logp[i,packet]),history+(packet,),next_states[i,packet],next_mask,next_reached,history,dp.value(next_mask,next_reached).reached_levels>=primary+1))
        expanded.sort(key=lambda row:(-row[0],row[1]));before=any(row[6] for row in expanded)
        if rule=="global_topk":kept=expanded[:beam_width]
        elif rule in {"top6_parent_rescue","random_parent_control"}:
            kept=expanded[:6];represented={row[5] for row in kept};candidates=[];seen=set()
            for row in expanded[6:]:
                if row[5] not in represented and row[5] not in seen:candidates.append(row);seen.add(row[5])
            if rule=="random_parent_control":random.Random(seed+depth).shuffle(candidates)
            kept.extend(candidates[:beam_width-len(kept)]);kept.extend(row for row in expanded if row not in kept and len(kept)<beam_width)
        elif rule=="parent_balanced":
            kept=[];seen=set()
            for row in expanded:
                if row[5] not in seen:kept.append(row);seen.add(row[5])
                if len(kept)==beam_width:break
            kept.extend(row for row in expanded if row not in kept and len(kept)<beam_width)
        else:raise ValueError(rule)
        kept.sort(key=lambda row:(-row[0],row[1]));first_prune|=before and not any(row[6] for row in kept);beams=[row[:5] for row in kept]
    return list(beams[0][1]),first_prune


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--train-data",required=True);p.add_argument("--embeddings",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--viability-head",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();config=json.loads(Path(a.protocol_config).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    boundary_ids=sorted({row["example_id"] for row in read_gzip(a.training_artifact) if 0.9 in row["critical_levels"]});sources={row["example_id"]:row for row in read_jsonl(a.train_data)};cache=torch.load(a.embeddings,map_location="cpu",weights_only=True);index={value:i for i,value in enumerate(cache["example_ids"])};model=load_policy(a.checkpoint,device,128);model.viability_head.load_state_dict(torch.load(a.viability_head,map_location=device,weights_only=True));model.eval();rules=list(config["rules"]);rows=[]
    with torch.no_grad():
        for number,example_id in enumerate(boundary_ids,1):
            source=sources[example_id];i=index[example_id];packets=cache["packets"][i].to(device=device,dtype=torch.float32);question=cache["questions"][i].to(device=device,dtype=torch.float32);fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32);fractions/=fractions.sum();exact=list(read_jsonl(Path(a.exact_dir)/f"{example_id}.jsonl",ExactSearchResult));dp=ReachabilityDP(exact,trajectory_levels(source));seed=int.from_bytes(hashlib.sha256(example_id.encode()).digest()[:4],"big")
            for rule in rules:
                order,first_prune=fast_replay(model,packets,question,fractions,dp,rule,seed);rows.append({"example_id":example_id,"rule":rule,"metrics":metric_for_order(source,exact,order),"first_irreversible_prune":first_prune})
            if number%50==0:print(json.dumps({"processed":number,"total":len(boundary_ids)}),flush=True)
    baseline={row["example_id"]:row for row in rows if row["rule"]=="global_topk"};summaries={}
    for rule in rules:
        selected=[row for row in rows if row["rule"]==rule];primary=[row for row in selected if "0.9" in row["metrics"]["per_level"]];repairs=sum(row["metrics"]["per_level"]["0.9"] and not baseline[row["example_id"]]["metrics"]["per_level"]["0.9"] for row in primary);breaks=sum(not row["metrics"]["per_level"]["0.9"] and baseline[row["example_id"]]["metrics"]["per_level"]["0.9"] for row in primary);complete_repairs=sum(row["metrics"]["complete"] and not baseline[row["example_id"]]["metrics"]["complete"] for row in selected);complete_breaks=sum(not row["metrics"]["complete"] and baseline[row["example_id"]]["metrics"]["complete"] for row in selected);regrets=[row["metrics"]["regret"] for row in selected if row["metrics"]["regret"] is not None];summaries[rule]={"examples":len(selected),"primary_090_successes":sum(row["metrics"]["per_level"]["0.9"] for row in primary),"primary_repairs":repairs,"primary_breaks":breaks,"complete_successes":sum(row["metrics"]["complete"] for row in selected),"complete_repairs":complete_repairs,"complete_breaks":complete_breaks,"mean_complete_regret":mean(regrets),"first_irreversible_prunes":sum(row["first_irreversible_prune"] for row in selected)}
    gate=config["opening_gate"];random_net=summaries["random_parent_control"]["primary_repairs"]-summaries["random_parent_control"]["primary_breaks"];passing=[]
    for rule in ("top6_parent_rescue","parent_balanced"):
        value=summaries[rule];net=value["primary_repairs"]-value["primary_breaks"]
        if net>=gate["primary_090_net_repairs_min"] and value["primary_breaks"]<=gate["primary_090_breaks_max"] and value["complete_breaks"]<=gate["complete_breaks_max"] and value["mean_complete_regret"]<=gate["mean_complete_regret_max"] and net>random_net:passing.append(rule)
    decision="GO_V17F2_ROBUST_SEARCH_INTERNAL_PROTOCOL_FREEZE" if passing else "STOP_V17F1_ROBUST_SEARCH_BRANCH"
    write_jsonl(out/"per_example_results.jsonl",rows);summary={"complete":True,"population_examples":len(boundary_ids),"summaries":summaries,"passing_rules":passing,"decision":decision,"internal_used":False,"development_used":False,"confirmation_used":False,"artifacts":{"protocol_sha256":sha256(a.protocol_config),"training_artifact_sha256":sha256(a.training_artifact),"viability_head_sha256":sha256(a.viability_head)}};write_metadata(out/"summary.json",summary);write_metadata(out/"decision.json",{"decision":decision,"passing_rules":passing,"training_authorized":False,"internal_used":False,"development_used":False,"confirmation_used":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
