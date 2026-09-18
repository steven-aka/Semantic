from __future__ import annotations

import argparse, json, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.schemas import read_jsonl, write_jsonl
from src.model.evidence_sufficiency_selector import EvidenceSufficiencySelector, conservative_sufficiency_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata


class CandidateData(Dataset):
    def __init__(self, data, cache):
        self.rows = list(read_jsonl(data)); payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload["example_ids"] != [row["example_id"] for row in self.rows]: raise ValueError("cache mismatch")
        self.packets = payload["packets"]; self.questions = payload["questions"]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index], self.packets[index], self.questions[index]


def collate(items):
    rows=[x[0] for x in items]; masks=[];ranks=[];retrieval=[];fractions=[];targets=[];active=[];indices=[]
    for i,row in enumerate(rows):
        for candidate in row["candidates"]:
            masks.append(candidate["mask"]);ranks.append(candidate["rank"]);retrieval.append(candidate["retrieval_logits"])
            fractions.append(candidate["mask_tokens"] / max(1,row["full_tokens"]));targets.append(candidate["success"]);active.append(candidate["active"]);indices.append(i)
    return {"rows":rows,"packets":torch.stack([x[1] for x in items]),"questions":torch.stack([x[2] for x in items]),
            "masks":torch.tensor(masks),"ranks":torch.tensor(ranks),"retrieval":torch.tensor(retrieval),"fractions":torch.tensor(fractions),
            "targets":torch.tensor(targets,dtype=torch.bool),"active":torch.tensor(active,dtype=torch.bool),"indices":torch.tensor(indices)}


def forward(model,b,device):
    return model(b["packets"].to(device),b["questions"].to(device),b["masks"].to(device),b["ranks"].to(device),b["retrieval"].to(device),b["fractions"].to(device),b["indices"].to(device))


def predict(model,data,device,batch_size):
    rows=[];scores=[];model.eval()
    with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for b in DataLoader(data,batch_size=batch_size,shuffle=False,collate_fn=collate):
            probs=torch.sigmoid(forward(model,b,device).float()).reshape(len(b["rows"]),5,5)
            # Fixed, label-free five-anchor utility; 0.90 receives the preregistered bottleneck weight two.
            weight=torch.tensor([1,1,1,2,1],device=device)
            utility=(probs*weight).sum(2).cpu()
            rows.extend(b["rows"]);scores.extend(utility.tolist())
    return rows,scores


def choose(rows,scores,threshold):
    chosen=[]
    for row,values in zip(rows,scores):
        best=max(range(1,5),key=lambda i:(values[i],-i));margin=values[best]-values[0]
        chosen.append(best if margin>=threshold else 0)
    return chosen


def summarize(rows,chosen):
    per=[]
    for anchor in range(5):
        eligible=[(r,c) for r,c in zip(rows,chosen) if r["candidates"][c]["active"][anchor]]
        per.append({"examples":len(eligible),"successes":sum(r["candidates"][c]["success"][anchor] for r,c in eligible)})
    complete=sum(r["candidates"][c]["complete"] for r,c in zip(rows,chosen))
    switches=sum(c!=0 for c in chosen)
    return {"per_anchor":per,"complete":complete,"switches":switches}


def select_threshold(rows,scores):
    margins=sorted({values[max(range(1,5),key=lambda i:(values[i],-i))]-values[0] for values in scores})
    candidates=[float("inf"),*margins]
    base=summarize(rows,[0]*len(rows));records=[]
    for threshold in candidates:
        selected=choose(rows,scores,threshold);summary=summarize(rows,selected)
        no_regression=all(value["successes"]>=base["per_anchor"][i]["successes"] for i,value in enumerate(summary["per_anchor"])) and summary["complete"]>=base["complete"]
        key=(int(no_regression),summary["per_anchor"][3]["successes"],summary["complete"],-summary["switches"],threshold)
        records.append((key,threshold,summary,selected))
    return max(records,key=lambda x:x[0]),base


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol-config",required=True);p.add_argument("--train-data",required=True);p.add_argument("--train-cache",required=True);p.add_argument("--validation-data",required=True);p.add_argument("--validation-cache",required=True);p.add_argument("--initial-head",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--steps",type=int,default=400);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=20260918);a=p.parse_args()
    protocol=json.loads(Path(a.protocol_config).read_text());expected=protocol["arguments"]
    if protocol["status"]!="APPROVED_TO_RUN":raise ValueError("protocol not approved")
    mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
    if mismatch:raise ValueError(mismatch)
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device("cuda:0");out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);write_metadata(out/"protocol_snapshot.json",protocol)
    train=CandidateData(a.train_data,a.train_cache);valid=CandidateData(a.validation_data,a.validation_cache)
    model=EvidenceSufficiencySelector();old=torch.load(a.initial_head,map_location="cpu",weights_only=True);state=model.state_dict()
    for key in state:
        if key in old and state[key].shape==old[key].shape:state[key]=old[key]
    model.load_state_dict(state);model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=collate);it=iter(loader);history=[]
    for step in range(1,a.steps+1):
        model.train()
        try:b=next(it)
        except StopIteration:it=iter(loader);b=next(it)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            logits=forward(model,b,device);loss=conservative_sufficiency_loss(logits,b["targets"].to(device),b["active"].to(device),b["indices"].to(device))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if step%100==0:
            rows,scores=predict(model,valid,device,a.batch_size);selected,base=select_threshold(rows,scores);key,threshold,summary,chosen=selected
            record={"step":step,"train_loss":float(loss.detach()),"threshold":threshold,"internal_validation":summary,"fallback":base,"selection_key":list(key)};history.append(record);print(json.dumps(record),flush=True);write_jsonl(out/"history.jsonl",history)
            torch.save({k:v.float().cpu() for k,v in model.state_dict().items()},out/f"selector_step{step}.pt")
    selected=max(history,key=lambda r:tuple(r["selection_key"]));Path(out/"selected_selector.pt").write_bytes((out/f"selector_step{selected['step']}.pt").read_bytes());write_metadata(out/"selected.json",selected)
    # Emit frozen validation choices for auditability.
    model.load_state_dict(torch.load(out/"selected_selector.pt",map_location="cpu",weights_only=True));rows,scores=predict(model,valid,device,a.batch_size);chosen=choose(rows,scores,selected["threshold"])
    write_jsonl(out/"internal_validation_selection.jsonl",[{"example_id":r["example_id"],"selected_index":c,"scores":s} for r,c,s in zip(rows,chosen,scores)])
    write_metadata(out/"metadata.json",experiment_metadata(stage="v14_conservative_evidence_sufficiency_selector",selector_sha256=sha256(out/"selected_selector.pt"),locked_roles_used=False,development300_used=False))

if __name__=="__main__":main()
