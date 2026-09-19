from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.evaluation.v17e0_frozen_information_bottleneck import normalize
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17b1_multi_anchor_viability import read_gzip


class LinearScorer(nn.Module):
    def __init__(self, dimension: int): super().__init__(); self.layer=nn.Linear(dimension,1,bias=False)
    def forward(self,value:torch.Tensor)->torch.Tensor:return self.layer(value).squeeze(-1)


def make_targets(records:list[dict[str,Any]],states:list[dict[str,Any]],exact_dir:str,sources:dict[str,dict[str,Any]])->dict[str,list[dict[str,Any]]]:
    by_example=defaultdict(list)
    for record,state in zip(records,states):by_example[record["example_id"]].append((record,state))
    output={name:[] for name in ("fidelity_delta","clipped_090_gap_reduction","new_anchor_gain")}
    for number,(example_id,rows) in enumerate(by_example.items(),1):
        exact=list(read_jsonl(Path(exact_dir)/f"{example_id}.jsonl",ExactSearchResult));by_mask={state_to_mask(row.state):row for row in exact};levels=[float(x) for x in sources[example_id]["attainable_levels"]]
        for record,state in rows:
            current=by_mask[record["selected_mask"]];action_rows=sorted(record["actions"],key=lambda x:x["packet"]);indices=sorted(state["positive"]+state["negative"]);values={name:[] for name in output}
            for action,feature_index in zip(action_rows,indices):
                nxt=by_mask[record["selected_mask"]|(1<<int(action["packet"]))];values["fidelity_delta"].append((feature_index,float(nxt.fidelity-current.fidelity)));values["clipped_090_gap_reduction"].append((feature_index,float(min(.9,nxt.fidelity)-min(.9,current.fidelity))));values["new_anchor_gain"].append((feature_index,float(sum(nxt.fidelity+1e-12>=x for x in levels)-sum(current.fidelity+1e-12>=x for x in levels))))
            viability={feature_index:int(action["future_viability"][3]) for action,feature_index in zip(action_rows,indices)}
            for name in output:output[name].append({"example_id":example_id,"history":record["history"],"values":values[name],"viability":viability})
        if number%500==0:print(json.dumps({"target_examples":number,"total":len(by_example)}),flush=True)
    return output


def preference_state(row:dict[str,Any],epsilon:float)->dict[str,Any]|None:
    high=[];low=[];values=[value for _,value in row["values"]];maximum=max(values);minimum=min(values)
    if maximum-minimum<=epsilon:return None
    high=[index for index,value in row["values"] if maximum-value<=epsilon];low=[index for index,value in row["values"] if maximum-value>epsilon]
    return {**row,"positive":high,"negative":low}


def train(features:torch.Tensor,states:list[dict[str,Any]],indices:list[int],config:dict[str,Any],device:torch.device,seed:int)->LinearScorer:
    settings=config["probe"];rng=random.Random(seed);torch.manual_seed(seed);model=LinearScorer(features.shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    for _ in range(settings["epochs"]):
        order=indices[:];rng.shuffle(order)
        for start in range(0,len(order),settings["batch_states"]):
            rows=order[start:start+settings["batch_states"]];positive=[rng.choice(states[i]["positive"]) for i in rows];negative=[rng.choice(states[i]["negative"]) for i in rows];margin=model(features[positive].to(device=device,dtype=torch.float32))-model(features[negative].to(device=device,dtype=torch.float32));loss=F.softplus(-margin).mean();optimizer.zero_grad();loss.backward();optimizer.step()
    return model


def evaluate(scores:torch.Tensor,states:list[dict[str,Any]],indices:list[int])->dict[str,float]:
    state_values=[];queries=defaultdict(list);correct=total=0
    for i in indices:
        row=states[i];matrix=scores[row["positive"]][:,None]-scores[row["negative"]][None,:];accuracy=float((matrix>0).float().mean());state_values.append(accuracy);queries[row["example_id"]].append(accuracy);correct+=int((matrix>0).sum());total+=matrix.numel()
    return {"pair_accuracy":correct/total,"macro_state_accuracy":mean(state_values),"macro_query_accuracy":mean(mean(x) for x in queries.values())}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--boundary-features",required=True);p.add_argument("--train-data",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();config=json.loads(Path(a.protocol_config).read_text());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");bundle=torch.load(a.boundary_features,map_location="cpu",weights_only=False);features=bundle["features"];feature_states=bundle["states"]
    all_records=read_gzip(a.training_artifact);records=[]
    for row in all_records:
        values=[x["future_viability"][3] for x in row["actions"] if x["future_viability"][3] is not None]
        if any(x==1 for x in values) and any(x==0 for x in values):records.append(row)
    sources={row["example_id"]:row for row in read_jsonl(a.train_data)};targets=make_targets(records,feature_states,a.exact_dir,sources);results=[];summaries={};gate=config["opening_gate"]
    for name,rows in targets.items():
        epsilon=float(config["tie_epsilon"][name]);usable=[x for x in (preference_state(row,epsilon) for row in rows) if x is not None];assign=[stable_fold(row["example_id"],3) for row in usable];folds=[]
        for fold in range(3):
            train_indices=[i for i,value in enumerate(assign) if value!=fold];test_indices=[i for i,value in enumerate(assign) if value==fold];train_actions=sorted({x for i in train_indices for x in usable[i]["positive"]+usable[i]["negative"]});normalized=normalize(features[train_actions],features);probe=train(normalized,usable,train_indices,config,device,config["probe"]["seed"]+fold)
            with torch.no_grad():scores=torch.cat([probe(normalized[start:start+4096].to(device=device,dtype=torch.float32)).cpu() for start in range(0,len(normalized),4096)])
            metrics=evaluate(scores,usable,test_indices);row={"target":name,"fold":fold,**metrics};results.append(row);folds.append(row);print(json.dumps(row),flush=True)
        greedy_safe=[];greedy_strict=[]
        for row in usable:
            maximum=max(value for _,value in row["values"]);best=[index for index,value in row["values"] if maximum-value<=epsilon];safe=[row["viability"][index] for index in best];greedy_safe.append(any(safe));greedy_strict.append(all(safe))
        summary={"states":len(rows),"non_tied_states":len(usable),"non_tied_state_fraction":len(usable)/len(rows),"macro_state_accuracy":mean(x["macro_state_accuracy"] for x in folds),"macro_query_accuracy":mean(x["macro_query_accuracy"] for x in folds),"fold_macro_state":[x["macro_state_accuracy"] for x in folds],"greedy_any_best_preserves_090_viability":mean(greedy_safe),"greedy_all_best_preserve_090_viability":mean(greedy_strict)};summary["opening_gate_passed"]=(summary["non_tied_state_fraction"]>=gate["non_tied_state_fraction_min"] and summary["macro_state_accuracy"]>=gate["macro_state_accuracy_min"] and summary["macro_query_accuracy"]>=gate["macro_query_accuracy_min"] and min(summary["fold_macro_state"])>=gate["every_fold_macro_state_min"] and summary["greedy_any_best_preserves_090_viability"]>=gate["greedy_best_preserves_090_viability_min"]);summaries[name]=summary
    passing=[name for name,value in summaries.items() if value["opening_gate_passed"]];decision="GO_V17E3_EVIDENCE_PROGRESS_CONTROLLER_DESIGN" if passing else "STOP_EVIDENCE_PROGRESS_AUDIT_LABEL_AND_INPUT_SUFFICIENCY"
    write_jsonl(out/"fold_results.jsonl",results);summary={"complete":True,"targets":summaries,"passing_targets":passing,"decision":decision,"internal_used":False,"development_used":False,"confirmation_used":False,"artifacts":{"protocol_sha256":sha256(a.protocol_config),"training_artifact_sha256":sha256(a.training_artifact),"boundary_features_sha256":sha256(a.boundary_features)}};write_metadata(out/"summary.json",summary);write_metadata(out/"decision.json",{"decision":decision,"passing_targets":passing,"training_authorized":False,"internal_used":False,"development_used":False,"confirmation_used":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
