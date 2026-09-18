from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v11_mask_retrieval_audit import evaluate, project, summarize
from src.model.counterfactual_repair_verifier import CounterfactualRepairVerifier
from src.reproducibility import sha256, write_metadata
from src.training.train_v15_counterfactual_repair_verifier import PairData, choose, predict


def ensemble_predictions(data, model_dir, device, batch_size):
    accumulated={}
    for fold in range(5):
        model=CounterfactualRepairVerifier(); model.load_state_dict(torch.load(Path(model_dir)/f"fold{fold}_verifier.pt",map_location="cpu",weights_only=True)); model.to(device)
        current=predict(model,data,list(range(len(data))),device,batch_size)
        for eid,value in current.items():
            if eid not in accumulated: accumulated[eid]={"probabilities":torch.tensor(value["probabilities"]),"delta_predictions":torch.tensor(value["delta_predictions"])}
            else:
                accumulated[eid]["probabilities"]+=torch.tensor(value["probabilities"]); accumulated[eid]["delta_predictions"]+=torch.tensor(value["delta_predictions"])
    return {eid:{k:(value/5).tolist() for k,value in values.items()} for eid,values in accumulated.items()}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--pair-data",required=True); p.add_argument("--cache",required=True); p.add_argument("--model-dir",required=True); p.add_argument("--rule",required=True); p.add_argument("--mask-data",required=True); p.add_argument("--v8-details",required=True); p.add_argument("--output",required=True); p.add_argument("--summary",required=True); p.add_argument("--scientific-role",required=True); p.add_argument("--batch-size",type=int,default=128); a=p.parse_args()
    device=torch.device("cuda:0"); data=PairData(a.pair_data,a.cache); predictions=ensemble_predictions(data,a.model_dir,device,a.batch_size); rule=json.loads(Path(a.rule).read_text())
    chosen=choose(data.rows,predictions,rule["risk_multiplier"],rule["threshold"]); lattice={r["example_id"]:r for r in read_jsonl(a.mask_data)}; base={r["example_id"]:r for r in read_jsonl(a.v8_details)}; details=[]
    for row,index in zip(data.rows,chosen):
        eid=row["example_id"]; reference=base[eid]; base_order=reference["decoded_order"]
        mask=0 if index==0 else row["pairs"][index-1]["candidate_mask"]; order=base_order if index==0 else project(base_order,mask)
        oracle=[int(x["oracle_nested_tokens"]) for x in reference["anchors"]]; result=evaluate(order,lattice[eid],oracle)
        probabilities=predictions[eid]["probabilities"]; scores=[x[1]-rule["risk_multiplier"]*x[2] for x in probabilities]
        details.append({"example_id":eid,"selected_index":index,"selected_mask":mask,"scores":scores,"probabilities":probabilities,"delta_predictions":predictions[eid]["delta_predictions"],"result":result})
    model_hashes={f"fold{i}":sha256(Path(a.model_dir)/f"fold{i}_verifier.pt") for i in range(5)}
    summary={"complete":True,"scientific_role":a.scientific_role,"result":summarize(details,"result"),"switches":sum(x["selected_index"]!=0 for x in details),"artifacts":{"pair_data_sha256":sha256(a.pair_data),"cache_sha256":sha256(a.cache),"rule_sha256":sha256(a.rule),"models":model_hashes}}
    write_jsonl(a.output,details); write_metadata(a.summary,summary); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
