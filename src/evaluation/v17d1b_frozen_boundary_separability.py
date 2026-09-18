from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17b1_multi_anchor_viability import load_policy


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path,"rt",encoding="utf-8") as handle:return [json.loads(line) for line in handle]


def stable_fold(value: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],"big")%folds


def strict_signature(row: dict[str,Any])->str:
    counts=[]
    for anchor in range(5):
        values=[a["future_viability"][anchor] for a in row["actions"] if a["future_viability"][anchor] is not None];counts.append((sum(v==1 for v in values),sum(v==0 for v in values)))
    return repr((tuple(row["anchor_status"]),row["reached_level_count"],row["depth"],len(row["actions"]),tuple(counts)))


class LinearScorer(nn.Module):
    def __init__(self,dim:int):super().__init__();self.score=nn.Linear(dim,1,bias=False)
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.score(x).squeeze(-1)


class MatchedCapacityScorer(nn.Module):
    def __init__(self,dim:int,hidden:int=128):super().__init__();self.score=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:return self.score(x).squeeze(-1)


def histories_tensor(rows:Sequence[dict[str,Any]],device:torch.device)->tuple[torch.Tensor,torch.Tensor]:
    width=max(len(x["history"]) for x in rows);hist=torch.full((len(rows),width),-1,dtype=torch.long,device=device);length=torch.tensor([len(x["history"]) for x in rows],device=device)
    for i,row in enumerate(rows):
        if row["history"]:hist[i,:len(row["history"])]=torch.tensor(row["history"],device=device)
    return hist,length


def extract_features(model:Any,records:list[dict[str,Any]],sources:dict[str,dict[str,Any]],cache:dict[str,Any],cache_index:dict[str,int],device:torch.device,batch_size:int)->dict[str,Any]:
    features=[];states=[];offset=0;model.eval()
    with torch.no_grad():
        for start in range(0,len(records),batch_size):
            rows=records[start:start+batch_size];ids=torch.tensor([cache_index[x["example_id"]] for x in rows]);packets=cache["packets"][ids].to(device=device,dtype=torch.float32);questions=cache["questions"][ids].to(device=device,dtype=torch.float32);hist,length=histories_tensor(rows,device);idx=torch.arange(len(rows),device=device);fractions=torch.tensor([sources[x["example_id"]]["packet_tokens"] for x in rows],device=device,dtype=torch.float32);fractions/=fractions.sum(1,keepdim=True);history_state,selected=model.history_states(packets,questions,hist,length,idx);phi,_=model.action_features(packets,questions,history_state,selected,idx,fractions)
            for i,row in enumerate(rows):
                actions=sorted(row["actions"],key=lambda x:x["packet"]);action_ids=[x["packet"] for x in actions];local=phi[i,action_ids].detach().cpu().to(torch.float16);features.append(local);labels=[x["future_viability"][3] for x in actions];pos=[offset+j for j,x in enumerate(labels) if x==1];neg=[offset+j for j,x in enumerate(labels) if x==0];states.append({"example_id":row["example_id"],"history":row["history"],"provenance":row["provenance"],"signature":strict_signature(row),"positive":pos,"negative":neg});offset+=len(actions)
            if (start//batch_size+1)%20==0:print(json.dumps({"feature_states":min(start+batch_size,len(records)),"total":len(records)}),flush=True)
    return {"features":torch.cat(features),"states":states}


def train_probe(kind:str,features:torch.Tensor,states:list[dict[str,Any]],train_indices:list[int],device:torch.device,seed:int,epochs:int=20,batch_size:int=512,sampling_unit:str="state")->nn.Module:
    torch.manual_seed(seed);model=(LinearScorer(features.shape[1]) if kind=="linear" else MatchedCapacityScorer(features.shape[1])).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=random.Random(seed)
    for epoch in range(epochs):
        if sampling_unit=="example":
            grouped=defaultdict(list)
            for index in train_indices:grouped[states[index]["example_id"]].append(index)
            order=[rng.choice(values) for values in grouped.values()]
        else:order=train_indices[:]
        rng.shuffle(order);model.train()
        for start in range(0,len(order),batch_size):
            chosen=order[start:start+batch_size];pos=[];neg=[];orientation=[]
            for state_index in chosen:
                row=states[state_index];pos.append(rng.choice(row["positive"]));neg.append(rng.choice(row["negative"]));orientation.append(rng.randrange(2))
            pf=features[pos].to(device=device,dtype=torch.float32);nf=features[neg].to(device=device,dtype=torch.float32);margin=model(pf)-model(nf);orient=torch.tensor(orientation,device=device,dtype=torch.bool);signed=torch.where(orient,margin,-margin);target=orient.float();loss=F.binary_cross_entropy_with_logits(signed,target)
            optimizer.zero_grad();loss.backward();optimizer.step()
    return model


def evaluate_probe(model:nn.Module,features:torch.Tensor,states:list[dict[str,Any]],indices:list[int],device:torch.device)->dict[str,Any]:
    state_accuracy=[];signature_values=defaultdict(list);margins=[];correct=total=0;model.eval()
    with torch.no_grad():
        for state_index in indices:
            row=states[state_index];pos=torch.tensor(row["positive"]);neg=torch.tensor(row["negative"]);ps=model(features[pos].to(device=device,dtype=torch.float32));ns=model(features[neg].to(device=device,dtype=torch.float32));matrix=ps[:,None]-ns[None,:];values=matrix.flatten().cpu();acc=float((values>0).float().mean());state_accuracy.append(acc);signature_values[row["signature"]].append(acc);margins.extend(values.tolist());correct+=int((values>0).sum());total+=len(values)
    ordered=sorted(margins)
    return {"pair_accuracy":correct/total,"macro_state_accuracy":mean(state_accuracy),"macro_signature_accuracy":mean(mean(x) for x in signature_values.values()),"pairs":total,"states":len(indices),"margin_median":ordered[len(ordered)//2],"margin_p10":ordered[len(ordered)//10]}


def state_action_feature(model:Any,source:dict[str,Any],history:list[int],action:int,cache:dict[str,Any],cache_index:dict[str,int],device:torch.device)->torch.Tensor:
    idx=cache_index[source["example_id"]];packets=cache["packets"][idx:idx+1].to(device=device,dtype=torch.float32);question=cache["questions"][idx:idx+1].to(device=device,dtype=torch.float32);row={"history":history};hist,length=histories_tensor([row],device);fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32)[None];fractions/=fractions.sum(1,keepdim=True);state,selected=model.history_states(packets,question,hist,length,torch.zeros(1,dtype=torch.long,device=device));phi,_=model.action_features(packets,question,state,selected,torch.zeros(1,dtype=torch.long,device=device),fractions);return phi[0,action].detach().cpu().to(torch.float16)


def main()->None:
    p=argparse.ArgumentParser(description="V17-D1B frozen boundary separability audit");p.add_argument("--train-data",required=True);p.add_argument("--internal-data",required=True);p.add_argument("--embeddings",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--training-artifact",required=True);p.add_argument("--wrong-events",required=True);p.add_argument("--fixed-events",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--folds",type=int,default=3);p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=20260919)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0");train_source={x["example_id"]:x for x in read_jsonl(a.train_data)};internal_source={x["example_id"]:x for x in read_jsonl(a.internal_data)};cache=torch.load(a.embeddings,map_location="cpu",weights_only=True);cache_index={x:i for i,x in enumerate(cache["example_ids"])};model=load_policy(a.checkpoint,device,128);model.eval()
    all_records=read_gzip(a.training_artifact);records=[]
    for row in all_records:
        values=[x["future_viability"][3] for x in row["actions"] if x["future_viability"][3] is not None]
        if any(x==1 for x in values) and any(x==0 for x in values):records.append(row)
    feature_cache=out/"boundary_features.pt"
    if feature_cache.exists():bundle=torch.load(feature_cache,map_location="cpu",weights_only=False)
    else:bundle=extract_features(model,records,train_source,cache,cache_index,device,a.batch_size);torch.save(bundle,feature_cache)
    features=bundle["features"];states=bundle["states"];splitters={"state_grouped":lambda i:f"{states[i]['example_id']}|{states[i]['history']}","example_grouped":lambda i:states[i]["example_id"],"signature_held_out":lambda i:states[i]["signature"]};fold_results=[];trained={}
    for split_name,key_fn in splitters.items():
        assignments=[stable_fold(key_fn(i),a.folds) for i in range(len(states))]
        for kind in ("linear","matched_capacity"):
            for fold in range(a.folds):
                train_idx=[i for i,x in enumerate(assignments) if x!=fold];test_idx=[i for i,x in enumerate(assignments) if x==fold];probe=train_probe(kind,features,states,train_idx,device,a.seed+fold,epochs=a.epochs);metrics=evaluate_probe(probe,features,states,test_idx,device);fold_results.append({"split":split_name,"probe":kind,"fold":fold,**metrics});print(json.dumps(fold_results[-1]),flush=True)
    for kind in ("linear","matched_capacity"):
        trained[kind]=train_probe(kind,features,states,list(range(len(states))),device,a.seed+100,epochs=a.epochs).cpu()
    example_assignments=[stable_fold(states[i]["example_id"],a.folds) for i in range(len(states))]
    for fold in range(a.folds):
        train_idx=[i for i,x in enumerate(example_assignments) if x!=fold];test_idx=[i for i,x in enumerate(example_assignments) if x==fold];probe=train_probe("linear",features,states,train_idx,device,a.seed+200+fold,epochs=a.epochs,sampling_unit="example");metrics=evaluate_probe(probe,features,states,test_idx,device);fold_results.append({"split":"example_grouped_query_balanced","probe":"linear","fold":fold,**metrics});print(json.dumps(fold_results[-1]),flush=True)
    trained["linear_query_balanced"]=train_probe("linear",features,states,list(range(len(states))),device,a.seed+300,epochs=a.epochs,sampling_unit="example").cpu()
    wrong=read_gzip(a.wrong_events);fixed={x["sample"]:x for x in read_gzip(a.fixed_events)};confirm=[]
    with torch.no_grad():
        for row in wrong:
            source=internal_source[row["sample"]];vp=state_action_feature(model,source,row["viable_parent_history"],row["viable_action"],cache,cache_index,device);bp=state_action_feature(model,source,row["boundary_parent_history"],row["boundary_action"],cache,cache_index,device);item={"sample":row["sample"],"had_structure_support":row["viable_equivalent_training_states"]>0 and row["boundary_equivalent_training_states"]>0,"d1a_force_direction_correct":fixed[row["sample"]]["direction_correct"]["force_target_090"]}
            for kind,probe in trained.items():probe=probe.to(device);item[f"{kind}_margin"]=float(probe(vp[None].to(device=device,dtype=torch.float32))-probe(bp[None].to(device=device,dtype=torch.float32)));item[f"{kind}_direction_correct"]=item[f"{kind}_margin"]>0
            confirm.append(item)
    write_jsonl(out/"fold_results.jsonl",fold_results);write_jsonl(out/"internal_confirmatory_events.jsonl",confirm)
    aggregate={}
    split_names=[*splitters,"example_grouped_query_balanced"]
    for split in split_names:
        aggregate[split]={}
        kinds=("linear",) if split=="example_grouped_query_balanced" else ("linear","matched_capacity")
        for kind in kinds:
            rows=[x for x in fold_results if x["split"]==split and x["probe"]==kind];aggregate[split][kind]={key:mean(x[key] for x in rows) for key in ("pair_accuracy","macro_state_accuracy","macro_signature_accuracy","margin_median","margin_p10")}
    result={"complete":True,"boundary_states":len(states),"action_features":len(features),"feature_dim":features.shape[1],"splits":aggregate,"internal_confirmatory":{"events":len(confirm),"linear_correct":sum(x["linear_direction_correct"] for x in confirm),"linear_preserved_d1a_correct":sum(x["linear_direction_correct"] for x in confirm if x["d1a_force_direction_correct"]),"d1a_correct_events":sum(x["d1a_force_direction_correct"] for x in confirm),"linear_correct_supported":sum(x["linear_direction_correct"] for x in confirm if x["had_structure_support"]),"supported_events":sum(x["had_structure_support"] for x in confirm),"matched_capacity_correct":sum(x["matched_capacity_direction_correct"] for x in confirm),"query_balanced_linear_correct":sum(x["linear_query_balanced_direction_correct"] for x in confirm),"query_balanced_preserved_d1a_correct":sum(x["linear_query_balanced_direction_correct"] for x in confirm if x["d1a_force_direction_correct"])},"gates":{"example_grouped_linear_min":.80,"signature_held_out_linear_min":.70,"internal_linear_min":"7/10","preserve_all_d1a_correct":True},"probe_policy":{"diagnostic_only":True,"checkpoint_deployable":False,"primary_sampling":"state equal weight","query_balanced_sensitivity":"one state per example per epoch; post-hoc diagnostic, not primary gate","pair_orientation":"shared scorer difference with randomized orientation; no intercept shortcut","epochs":a.epochs,"weight_decay":1e-4},"sealed":{"development":True,"all_model_training":True,"terminal_reranker":True,"hop2":True,"learned_cutoff":True,"confirmation":True},"artifacts":{"train_sha256":sha256(a.train_data),"embedding_sha256":sha256(a.embeddings),"training_artifact_sha256":sha256(a.training_artifact)}};write_metadata(out/"summary.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
