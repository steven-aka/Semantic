from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v11_mask_retrieval_audit import evaluate, project, summarize
from src.model.evidence_sufficiency_selector import EvidenceSufficiencySelector
from src.reproducibility import sha256, write_metadata
from src.training.train_v14_conservative_selector import CandidateData, choose, predict


def main():
    p=argparse.ArgumentParser();p.add_argument("--candidate-data",required=True);p.add_argument("--cache",required=True);p.add_argument("--selector",required=True);p.add_argument("--selection",required=True);p.add_argument("--mask-data",required=True);p.add_argument("--v8-details",required=True);p.add_argument("--output",required=True);p.add_argument("--summary",required=True);p.add_argument("--batch-size",type=int,default=64);a=p.parse_args()
    device=torch.device("cuda:0");model=EvidenceSufficiencySelector();model.load_state_dict(torch.load(a.selector,map_location="cpu",weights_only=True));model.to(device)
    data=CandidateData(a.candidate_data,a.cache);rows,scores=predict(model,data,device,a.batch_size);selection=json.loads(Path(a.selection).read_text());chosen=choose(rows,scores,selection["threshold"])
    lattice={r["example_id"]:r for r in read_jsonl(a.mask_data)};base={r["example_id"]:r for r in read_jsonl(a.v8_details)};details=[]
    for row,index,values in zip(rows,chosen,scores):
        source=lattice[row["example_id"]];reference=base[row["example_id"]];base_order=reference["decoded_order"];mask=row["candidates"][index]["mask"];order=base_order if index==0 else project(base_order,mask);oracle=[int(x["oracle_nested_tokens"]) for x in reference["anchors"]]
        result=evaluate(order,source,oracle);details.append({"example_id":row["example_id"],"selected_index":index,"selected_mask":mask,"scores":values,"result":result})
    summary={"complete":True,"scientific_role":"frozen V14 selector evaluated with oracle cutoff on consumed development","result":summarize(details,"result"),"switches":sum(x["selected_index"]!=0 for x in details),"artifacts":{"candidate_data_sha256":sha256(a.candidate_data),"cache_sha256":sha256(a.cache),"selector_sha256":sha256(a.selector)}}
    write_jsonl(a.output,details);write_metadata(a.summary,summary);print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
