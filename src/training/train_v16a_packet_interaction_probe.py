from __future__ import annotations

import argparse,json,random
from pathlib import Path

import torch
from torch.utils.data import DataLoader,Subset

from src.data.schemas import write_jsonl
from src.model.packet_interaction_probe import PacketInteractionProbe,interaction_loss
from src.reproducibility import experiment_metadata,sha256,write_metadata
from src.training.train_v15_counterfactual_repair_verifier import PairData,class_weights,collate,make_folds,select_rule


def targets(batch,device):
    values=[];active=[]
    for row in batch["rows"]:
        base=row["fallback"]
        for pair in row["pairs"]:
            values.append([int(pair["candidate_success"][i])-int(base["success"][i])+1 for i in range(5)])
            active.append(base["active"])
    return torch.tensor(values,device=device),torch.tensor(active,device=device)


def forward(model,batch,device):
    keys=("candidate","matched","added","removed","ranks","retrieval","fractions","indices")
    return model(batch["packets"].to(device),batch["questions"].to(device),*(batch[k].to(device) for k in keys))


def predict(model,data,indices,device,batch_size):
    output={};model.eval();loader=DataLoader(Subset(data,indices),batch_size=batch_size,shuffle=False,collate_fn=collate)
    with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for batch in loader:
            classes,anchors=forward(model,batch,device);cp=torch.softmax(classes.float(),1).cpu().reshape(len(batch["rows"]),4,4);ap=torch.softmax(anchors.float(),2).cpu().reshape(len(batch["rows"]),4,5,3)
            for row,c,a in zip(batch["rows"],cp,ap):output[row["example_id"]]={"probabilities":c.tolist(),"anchor_delta_probabilities":a.tolist()}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--train-data",required=True);p.add_argument("--train-cache",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch-size",type=int,default=96);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=20260918);a=p.parse_args()
    protocol=json.loads(Path(a.protocol_config).read_text());expected=protocol["arguments"]
    if protocol["status"]!="APPROVED_TO_RUN":raise ValueError("protocol not approved")
    mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
    if mismatch:raise ValueError(mismatch)
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device("cuda:0");out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/"protocol_snapshot.json",protocol)
    data=PairData(a.train_data,a.train_cache);folds=make_folds(data.rows);oof={};history=[]
    for fold in range(5):
        train=[i for i,f in enumerate(folds) if f!=fold];held=[i for i,f in enumerate(folds) if f==fold];weights,counts=class_weights(data.rows,train)
        model=PacketInteractionProbe().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps)
        loader=DataLoader(Subset(data,train),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed+fold),collate_fn=collate);iterator=iter(loader)
        for step in range(1,a.steps+1):
            model.train()
            try:b=next(iterator)
            except StopIteration:iterator=iter(loader);b=next(iterator)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                c,anchor=forward(model,b,device);at,active=targets(b,device);loss=interaction_loss(c,anchor,b["classes"].to(device),at,active,b["indices"].to(device),weights.to(device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if step%100==0:print(json.dumps({"fold":fold,"step":step,"loss":float(loss.detach())}),flush=True)
        checkpoint=out/f"fold{fold}_probe.pt";torch.save({k:v.float().cpu() for k,v in model.state_dict().items()},checkpoint);oof.update(predict(model,data,held,device,a.batch_size));history.append({"fold":fold,"class_counts":counts,"final_loss":float(loss.detach()),"checkpoint_sha256":sha256(checkpoint)})
    selected,fallback=select_rule(data.rows,oof,protocol["frozen"]["risk_multiplier_grid"],protocol["frozen"]["decision_threshold_grid"]);key,multiplier,threshold,summary,chosen=selected
    rule={"risk_multiplier":multiplier,"threshold":threshold,"oof_summary":summary,"fallback":fallback,"selection_key":list(key)};write_metadata(out/"selected_rule.json",rule);write_jsonl(out/"oof_predictions.jsonl",[{"example_id":r["example_id"],"selected_index":c,**oof[r["example_id"]]} for r,c in zip(data.rows,chosen)]);write_jsonl(out/"history.jsonl",history);write_metadata(out/"metadata.json",experiment_metadata(stage="v16a_packet_interaction_diagnostic",development300_used=False,locked_roles_used=False));print(json.dumps(rule,indent=2))


if __name__=="__main__":main()
