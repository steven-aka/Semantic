from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.evaluation.v17e0_frozen_information_bottleneck import normalize
from src.evaluation.v17e2_evidence_progress_target_audit import evaluate, preference_state, train
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17b1_multi_anchor_viability import read_gzip


def backward_reachability(fidelity:list[float],target:float=.9,width:int=12)->list[bool]:
    reach=[value+1e-12>=target for value in fidelity]
    for count in range(width-1,-1,-1):
        for mask in range(1<<width):
            if mask.bit_count()!=count or reach[mask]:continue
            reach[mask]=any(not(mask&(1<<packet)) and reach[mask|(1<<packet)] for packet in range(width))
    return reach


def robustness(mask:int,action:int,reach:list[bool],met:list[bool]|None=None,width:int=12)->tuple[float,float,int]:
    met = reach if met is None else met
    first=mask|(1<<action);remaining=[packet for packet in range(width) if not(first&(1<<packet))]
    if met[first]:return (1.0,1.0,len(remaining))
    if not remaining:return (float(reach[first]),float(reach[first]),0)
    one=[reach[first|(1<<packet)] for packet in remaining];families=sum(one);r1=families/len(one)
    grandchildren=[]
    for packet in remaining:
        second=first|(1<<packet);rest=[other for other in remaining if other!=packet]
        if rest:grandchildren.extend(met[second] or reach[second|(1<<other)] for other in rest)
        else:grandchildren.append(reach[second])
    return r1,sum(grandchildren)/len(grandchildren),families


def build_rows(records:list[dict[str,Any]],states:list[dict[str,Any]],exact_dir:str)->dict[str,list[dict[str,Any]]]:
    grouped=defaultdict(list)
    for record,state in zip(records,states):grouped[record["example_id"]].append((record,state))
    output={name:[] for name in ("r1","r2","successful_families")};checked=0
    for number,(example_id,items) in enumerate(grouped.items(),1):
        exact=list(read_jsonl(Path(exact_dir)/f"{example_id}.jsonl",ExactSearchResult));fidelity=[0.0]*4096
        for row in exact:fidelity[state_to_mask(row.state)]=float(row.fidelity)
        met=[value+1e-12>=.9 for value in fidelity];reach=backward_reachability(fidelity)
        for record,state in items:
            action_rows=sorted(record["actions"],key=lambda x:x["packet"]);indices=sorted(state["positive"]+state["negative"]);values={name:[] for name in output}
            for action,index in zip(action_rows,indices):
                viable=int(action["future_viability"][3]);actual=int(reach[record["selected_mask"]|(1<<int(action["packet"]))]);checked+=1
                if viable!=actual:raise RuntimeError("binary viability/reachability mismatch")
                if viable:
                    r1,r2,families=robustness(record["selected_mask"],int(action["packet"]),reach,met);values["r1"].append((index,r1));values["r2"].append((index,r2));values["successful_families"].append((index,float(families)))
            for name in output:output[name].append({"example_id":example_id,"history":record["history"],"values":values[name]})
        if number%100==0:print(json.dumps({"robustness_examples":number,"total":len(grouped)}),flush=True)
    return output


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--boundary-features",required=True);p.add_argument("--frontier-features",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();config=json.loads(Path(a.protocol_config).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");bundle=torch.load(a.boundary_features,map_location="cpu",weights_only=False);local=bundle["features"];states=bundle["states"];frontier=torch.load(a.frontier_features,map_location="cpu",weights_only=False)["pooled"]
    records=[]
    for row in read_gzip(a.training_artifact):
        values=[x["future_viability"][3] for x in row["actions"] if x["future_viability"][3] is not None]
        if any(x==1 for x in values) and any(x==0 for x in values):records.append(row)
    targets=build_rows(records,states,a.exact_dir);features={"local_full":local,"frontier_pooled":frontier,"local_plus_frontier_pooled":torch.cat((local,frontier),-1)};results=[];summaries={};gate=config["opening_gate"]
    for target,rows in targets.items():
        usable=[x for x in (preference_state(row,1e-9) if len(row["values"])>1 else None for row in rows) if x is not None];ranges=[max(value for _,value in row["values"])-min(value for _,value in row["values"]) for row in rows if row["values"]];target_summary={"positive_states":sum(bool(row["values"]) for row in rows),"heterogeneous_positive_states":len(usable),"heterogeneous_positive_state_fraction":len(usable)/sum(bool(row["values"]) for row in rows),"median_within_state_range":median(ranges),"features":{}}
        assign=[stable_fold(row["example_id"],3) for row in usable]
        for feature_name,raw in features.items():
            folds=[]
            for fold in range(3):
                train_indices=[i for i,value in enumerate(assign) if value!=fold];test_indices=[i for i,value in enumerate(assign) if value==fold];train_actions=sorted({x for i in train_indices for x in usable[i]["positive"]+usable[i]["negative"]});normalized=normalize(raw[train_actions],raw);probe=train(normalized,usable,train_indices,config,device,config["probe"]["seed"]+fold)
                with torch.no_grad():scores=torch.cat([probe(normalized[start:start+4096].to(device=device,dtype=torch.float32)).cpu() for start in range(0,len(normalized),4096)])
                metrics=evaluate(scores,usable,test_indices);entry={"target":target,"feature":feature_name,"fold":fold,**metrics};results.append(entry);folds.append(entry);print(json.dumps(entry),flush=True)
            summary={"macro_state_accuracy":mean(x["macro_state_accuracy"] for x in folds),"macro_query_accuracy":mean(x["macro_query_accuracy"] for x in folds),"fold_macro_state":[x["macro_state_accuracy"] for x in folds]};summary["opening_gate_passed"]=(target_summary["heterogeneous_positive_state_fraction"]>=gate["heterogeneous_positive_state_fraction_min"] and summary["macro_state_accuracy"]>=gate["macro_state_accuracy_min"] and summary["macro_query_accuracy"]>=gate["macro_query_accuracy_min"] and min(summary["fold_macro_state"])>=gate["every_fold_macro_state_min"]);target_summary["features"][feature_name]=summary
        summaries[target]=target_summary
    passing=[f"{target}:{feature}" for target,value in summaries.items() for feature,metrics in value["features"].items() if metrics["opening_gate_passed"]]
    if passing:decision="GO_V17F1_ROBUSTNESS_TARGET_CONTROLLER_DESIGN"
    elif max(value["heterogeneous_positive_state_fraction"] for value in summaries.values())>=gate["heterogeneous_positive_state_fraction_min"]:decision="GO_V17F1_ROBUST_SEARCH_WITH_WEAK_HEURISTIC_DESIGN"
    else:decision="STOP_ROBUSTNESS_TARGET_REASSESS_INPUT_CONTRACT"
    write_jsonl(out/"fold_results.jsonl",results);summary={"complete":True,"targets":summaries,"passing_target_features":passing,"decision":decision,"oracle_features_used_as_inputs":False,"internal_used":False,"development_used":False,"confirmation_used":False,"artifacts":{"protocol_sha256":sha256(a.protocol_config),"training_artifact_sha256":sha256(a.training_artifact),"boundary_features_sha256":sha256(a.boundary_features),"frontier_features_sha256":sha256(a.frontier_features)}};write_metadata(out/"summary.json",summary);write_metadata(out/"decision.json",{"decision":decision,"passing_target_features":passing,"training_authorized":False,"internal_used":False,"development_used":False,"confirmation_used":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
