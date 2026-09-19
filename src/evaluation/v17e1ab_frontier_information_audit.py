from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import histories_tensor, stable_fold
from src.evaluation.v17e0_frozen_information_bottleneck import evaluate_scores, normalize, paired_bootstrap, train_probe
from src.reproducibility import sha256, write_metadata
from src.training.train_v17b1_multi_anchor_viability import load_policy, read_gzip


def frontier_features(model: Any, records: list[dict[str, Any]], states: list[dict[str, Any]], sources: dict[str, dict[str, Any]], cache: dict[str, Any], cache_index: dict[str,int], device: torch.device, batch_size: int = 64) -> dict[str, torch.Tensor]:
    total = max(max(row["positive"] + row["negative"]) for row in states) + 1
    scalar = torch.empty((total, 10), dtype=torch.float16); pooled = torch.empty((total, 1024), dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows=records[start:start+batch_size]; meta=states[start:start+batch_size]
            ids=torch.tensor([cache_index[row["example_id"]] for row in rows]);packets=cache["packets"][ids].to(device=device,dtype=torch.float32);questions=cache["questions"][ids].to(device=device,dtype=torch.float32)
            histories,lengths=histories_tensor(rows,device);indices=torch.arange(len(rows),device=device);fractions=torch.tensor([sources[row["example_id"]]["packet_tokens"] for row in rows],device=device,dtype=torch.float32);fractions/=fractions.sum(1,keepdim=True)
            current,selected=model.history_states(packets,questions,histories,lengths,indices)
            parents=[];actions=[];global_indices=[]
            for i,(row,state) in enumerate(zip(rows,meta)):
                action_rows=sorted(row["actions"],key=lambda x:x["packet"]); feature_indices=sorted(state["positive"]+state["negative"])
                if len(action_rows)!=len(feature_indices):raise ValueError("action/feature alignment mismatch")
                for action_row,feature_index in zip(action_rows,feature_indices):parents.append(i);actions.append(int(action_row["packet"]));global_indices.append(feature_index)
            parent=torch.tensor(parents,device=device);action=torch.tensor(actions,device=device);next_state=model.history_gru(packets[parent,action],current[parent]);selected_after=selected[parent].clone();selected_after[torch.arange(len(parent),device=device),action]=True
            successor, _=model.action_features(packets,questions,next_state,selected_after,parent,fractions);base=model.action_head(successor).squeeze(-1).masked_fill(selected_after,-torch.inf);legal=~selected_after;count=legal.sum(1).float();safe_count=count.clamp_min(1.0);top4=torch.topk(base,4,dim=1).values;top4=torch.where(torch.isfinite(top4),top4,torch.full_like(top4,-20.0))
            finite=base.masked_fill(~legal,0.0);average=finite.sum(1)/safe_count;variance=((finite-average[:,None]).square()*legal).sum(1)/safe_count;safe_logits=base.masked_fill(~legal,-20.0);probability=torch.softmax(safe_logits,dim=1)*legal;probability=probability/probability.sum(1,keepdim=True).clamp_min(1e-12);entropy=-(probability*probability.clamp_min(1e-12).log()).sum(1);margin=torch.where(count>1,top4[:,0]-top4[:,1],torch.zeros_like(count))
            scalar_batch=torch.cat((count[:,None],top4,average[:,None],variance.sqrt()[:,None],entropy[:,None],margin[:,None],top4[:,:1]),1)
            candidates=packets[parent];candidate_mean=(candidates*legal[:,:,None]).sum(1)/safe_count[:,None];candidate_max=candidates.masked_fill(~legal[:,:,None],-torch.inf).max(1).values;candidate_max=torch.where(torch.isfinite(candidate_max),candidate_max,torch.zeros_like(candidate_max));pooled_batch=torch.cat((candidate_mean,candidate_max),1)
            global_tensor=torch.tensor(global_indices);scalar[global_tensor]=scalar_batch.cpu().to(torch.float16);pooled[global_tensor]=pooled_batch.cpu().to(torch.float16)
            if (start//batch_size+1)%20==0:print(json.dumps({"frontier_states":min(start+batch_size,len(records)),"total":len(records)}),flush=True)
    return {"scalar":scalar,"pooled":pooled}


def random_recall(n:int,p:int,k:int)->float:
    if k>=n:return 1.0
    return 1.0-(comb(n-p,k)/comb(n,k) if n-p>=k else 0.0)


def rank_profile(scores:torch.Tensor,states:list[dict[str,Any]])->dict[str,Any]:
    observed={k:[] for k in (1,2,4,8)};expected={k:[] for k in (1,2,4,8)};prevalence=[]
    for row in states:
        indices=row["positive"]+row["negative"];values=scores[indices];labels=torch.tensor([1]*len(row["positive"])+[0]*len(row["negative"]));order=torch.argsort(values,descending=True);best=next(rank+1 for rank,index in enumerate(order.tolist()) if labels[index]);n=len(indices);p=len(row["positive"]);prevalence.append(p/n)
        for k in observed:observed[k].append(float(best<=k));expected[k].append(random_recall(n,p,min(k,n)))
    return {"states":len(states),"mean_viable_prevalence":mean(prevalence),"recall":{str(k):mean(observed[k]) for k in observed},"random_expected":{str(k):mean(expected[k]) for k in expected},"lift":{str(k):mean(observed[k])-mean(expected[k]) for k in observed}}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--boundary-features",required=True);p.add_argument("--train-data",required=True);p.add_argument("--embeddings",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--viability-head",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();config=json.loads(Path(a.protocol_config).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle=torch.load(a.boundary_features,map_location="cpu",weights_only=False);local=bundle["features"];states=bundle["states"];all_records=read_gzip(a.training_artifact);records=[]
    for row in all_records:
        values=[x["future_viability"][3] for x in row["actions"] if x["future_viability"][3] is not None]
        if any(x==1 for x in values) and any(x==0 for x in values):records.append(row)
    if len(records)!=len(states):raise ValueError("boundary state count mismatch")
    source={row["example_id"]:row for row in read_jsonl(a.train_data)};cache=torch.load(a.embeddings,map_location="cpu",weights_only=True);cache_index={value:i for i,value in enumerate(cache["example_ids"])};model=load_policy(a.checkpoint,device,128);model.viability_head.load_state_dict(torch.load(a.viability_head,map_location=device,weights_only=True));model.eval()
    frontier_path=out/"frontier_features.pt"
    if frontier_path.exists():frontier=torch.load(frontier_path,map_location="cpu",weights_only=False)
    else:frontier=frontier_features(model,records,states,source,cache,cache_index,device);torch.save(frontier,frontier_path)
    taps={"local_full":local,"frontier_scalar":frontier["scalar"],"frontier_pooled":frontier["pooled"],"local_plus_scalar":torch.cat((local,frontier["scalar"]),-1),"local_plus_pooled":torch.cat((local,frontier["pooled"]),-1)};assign=[stable_fold(row["example_id"],3) for row in states];results=[];oof={tap:torch.empty(len(local)) for tap in taps};query_results={}
    for tap,raw in taps.items():
        tap_queries={}
        for fold in range(3):
            train=[i for i,value in enumerate(assign) if value!=fold];test=[i for i,value in enumerate(assign) if value==fold];train_actions=sorted({x for i in train for x in states[i]["positive"]+states[i]["negative"]});norm=normalize(raw[train_actions],raw);probe=train_probe(norm,states,train,config,device,config["probe"]["seed"]+fold)
            with torch.no_grad():scores=torch.cat([probe(norm[start:start+4096].to(device=device,dtype=torch.float32)).cpu() for start in range(0,len(norm),4096)])
            test_actions=sorted({x for i in test for x in states[i]["positive"]+states[i]["negative"]});oof[tap][test_actions]=scores[test_actions];metrics,queries=evaluate_scores(scores,states,test);results.append({"tap":tap,"fold":fold,"dimension":raw.shape[1],**metrics});tap_queries.update(queries);print(json.dumps(results[-1]),flush=True)
        query_results[tap]=tap_queries
    local_queries=query_results["local_full"];gate=config["frontier_gate"];aggregate={}
    for tap in taps:
        rows=[x for x in results if x["tap"]==tap];boot=paired_bootstrap(query_results[tap],local_queries,gate["bootstrap_replicates"],config["probe"]["seed"]);aggregate[tap]={"macro_state_accuracy":mean(x["macro_state_accuracy"] for x in rows),"macro_query_accuracy":mean(x["macro_query_accuracy"] for x in rows),"fold_macro_state":[x["macro_state_accuracy"] for x in rows],"paired_vs_local":boot};aggregate[tap]["frontier_gate_passed"]=(tap!="local_full" and aggregate[tap]["macro_state_accuracy"]>=gate["macro_state_accuracy_min"] and aggregate[tap]["macro_query_accuracy"]>=gate["macro_query_accuracy_min"] and min(aggregate[tap]["fold_macro_state"])>=gate["every_fold_macro_state_min"] and boot["lower_95"]>0)
    profiles={tap:rank_profile(scores,states) for tap,scores in oof.items()};robust=config["robust_search_opening_gate"];frontier_pass=[tap for tap in taps if aggregate[tap]["frontier_gate_passed"]];robust_taps=[tap for tap in taps if profiles[tap]["recall"]["4"]>=robust["oof_best_viable_recall_at_4_min"] and profiles[tap]["lift"]["4"]>=robust["random_adjusted_lift_at_4_min"]]
    if frontier_pass:decision="GO_V17E2_FRONTIER_AWARE_CRITIC_DESIGN"
    elif robust_taps:decision="GO_V17E1C_ROBUST_BEAM_CAUSAL_REPLAY"
    else:decision="GO_V17E2_EVIDENCE_PROGRESS_TARGET_AUDIT"
    write_jsonl(out/"fold_results.jsonl",results);summary={"complete":True,"aggregates":aggregate,"rank_profiles":profiles,"frontier_passing_taps":frontier_pass,"robust_search_opening_taps":robust_taps,"decision":decision,"internal_used":False,"development_used":False,"confirmation_used":False,"artifacts":{"protocol_sha256":sha256(a.protocol_config),"training_artifact_sha256":sha256(a.training_artifact),"boundary_features_sha256":sha256(a.boundary_features)}};write_metadata(out/"summary.json",summary);write_metadata(out/"decision.json",{"decision":decision,"frontier_passing_taps":frontier_pass,"robust_search_opening_taps":robust_taps,"training_authorized":False,"internal_used":False,"development_used":False,"confirmation_used":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
