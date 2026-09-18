from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.schemas import read_jsonl, write_jsonl
from src.model.counterfactual_repair_verifier import CounterfactualRepairVerifier, verifier_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata


class PairData(Dataset):
    def __init__(self, data: str, cache: str):
        self.rows = list(read_jsonl(data))
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload["example_ids"] != [row["example_id"] for row in self.rows]:
            raise ValueError("cache mismatch")
        self.packets = payload["packets"]
        self.questions = payload["questions"]

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index], self.packets[index], self.questions[index]


def collate(items):
    rows = [x[0] for x in items]
    fields = {key: [] for key in ("candidate", "matched", "added", "removed", "ranks", "retrieval", "fractions", "classes", "deltas", "indices")}
    for i, row in enumerate(rows):
        for pair in row["pairs"]:
            fields["candidate"].append(pair["candidate_mask"]); fields["matched"].append(pair["matched_fallback_mask"])
            fields["added"].append(pair["added_mask"]); fields["removed"].append(pair["removed_mask"])
            fields["ranks"].append(pair["rank"]); fields["retrieval"].append(pair["retrieval_logits"])
            fields["fractions"].append(pair["mask_token_fraction"]); fields["classes"].append(pair["class_index"])
            fields["deltas"].append(pair["delta_max_fidelity"]); fields["indices"].append(i)
    return {"rows": rows, "packets": torch.stack([x[1] for x in items]), "questions": torch.stack([x[2] for x in items]),
            **{k: torch.tensor(v, dtype=torch.float32 if k in ("retrieval", "fractions", "deltas") else torch.long) for k, v in fields.items()}}


def forward(model, batch, device):
    keys = ("candidate", "matched", "added", "removed", "ranks", "retrieval", "fractions", "indices")
    return model(batch["packets"].to(device), batch["questions"].to(device), *(batch[k].to(device) for k in keys))


def make_folds(rows, count=5):
    # Stable, label-stratified assignment keeps the rare repair and break queries represented in every fold.
    groups = {}
    for i, row in enumerate(rows):
        signature = (any(p["class_index"] == 1 for p in row["pairs"]), any(p["class_index"] == 2 for p in row["pairs"]))
        groups.setdefault(signature, []).append(i)
    folds = [-1] * len(rows)
    for indices in groups.values():
        indices.sort(key=lambda i: hashlib.sha256(rows[i]["example_id"].encode()).hexdigest())
        for offset, index in enumerate(indices): folds[index] = offset % count
    return folds


def class_weights(rows, indices):
    counts = Counter(pair["class_index"] for i in indices for pair in rows[i]["pairs"])
    total = sum(counts.values()); raw = [min(20.0, total / (4 * max(1, counts[c]))) for c in range(4)]
    scale = 4 / sum(raw)
    return torch.tensor([x * scale for x in raw]), dict(sorted(counts.items()))


def predict(model, data, indices, device, batch_size):
    output = {}; model.eval()
    loader = DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            logits, delta = forward(model, batch, device); probs = torch.softmax(logits.float(), 1).cpu().reshape(len(batch["rows"]), 4, 4)
            delta = delta.float().cpu().reshape(len(batch["rows"]), 4)
            for row, p, d in zip(batch["rows"], probs, delta): output[row["example_id"]] = {"probabilities": p.tolist(), "delta_predictions": d.tolist()}
    return output


def choose(rows, predictions, risk_multiplier, threshold):
    chosen=[]
    for row in rows:
        probabilities=predictions[row["example_id"]]["probabilities"]
        scores=[p[1] - risk_multiplier * p[2] for p in probabilities]
        best=max(range(4), key=lambda i:(scores[i],-i))
        chosen.append(best + 1 if scores[best] >= threshold else 0)
    return chosen


def summarize(rows, chosen):
    def item(row, index): return row["fallback"] if index == 0 else row["pairs"][index - 1]
    per=[]
    for anchor in range(5):
        eligible=[item(r,c) for r,c in zip(rows,chosen) if item(r,c)["active"][anchor]]
        per.append({"examples":len(eligible),"successes":sum(x["success"][anchor] if "success" in x else x["candidate_success"][anchor] for x in eligible)})
    return {"per_anchor":per,"complete":sum((x["complete"] if "complete" in x else x["candidate_complete"]) for x in (item(r,c) for r,c in zip(rows,chosen))),"switches":sum(c != 0 for c in chosen)}


def select_rule(rows, predictions, multipliers, thresholds):
    fallback=summarize(rows,[0]*len(rows)); records=[]
    for multiplier in multipliers:
        for threshold in thresholds:
            selected=choose(rows,predictions,multiplier,threshold); summary=summarize(rows,selected)
            safe=all(x["successes"]>=fallback["per_anchor"][i]["successes"] for i,x in enumerate(summary["per_anchor"])) and summary["complete"]>=fallback["complete"]
            key=(int(safe),summary["per_anchor"][3]["successes"],summary["complete"],-summary["switches"],multiplier,threshold)
            records.append((key,multiplier,threshold,summary,selected))
    return max(records,key=lambda x:x[0]), fallback


def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol-config",required=True); p.add_argument("--train-data",required=True); p.add_argument("--train-cache",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--steps",type=int,default=300); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--seed",type=int,default=20260918); a=p.parse_args()
    protocol=json.loads(Path(a.protocol_config).read_text()); expected=protocol["arguments"]
    if protocol["status"] != "APPROVED_TO_RUN": raise ValueError("protocol not approved")
    mismatch={k:(getattr(a,k),v) for k,v in expected.items() if getattr(a,k)!=v}
    if mismatch: raise ValueError(mismatch)
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda:0"); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); write_metadata(out/"protocol_snapshot.json",protocol)
    data=PairData(a.train_data,a.train_cache); folds=make_folds(data.rows); write_jsonl(out/"fold_assignments.jsonl",[{"example_id":r["example_id"],"fold":f} for r,f in zip(data.rows,folds)])
    oof={}; history=[]
    for fold in range(5):
        train_indices=[i for i,f in enumerate(folds) if f!=fold]; heldout=[i for i,f in enumerate(folds) if f==fold]
        weights,counts=class_weights(data.rows,train_indices); model=CounterfactualRepairVerifier().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps)
        loader=DataLoader(Subset(data,train_indices),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed+fold),collate_fn=collate); iterator=iter(loader)
        for step in range(1,a.steps+1):
            model.train()
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                logits,delta=forward(model,batch,device); loss=verifier_loss(logits,delta,batch["classes"].to(device),batch["deltas"].to(device),batch["indices"].to(device),weights.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0: print(json.dumps({"fold":fold,"step":step,"loss":float(loss.detach())}),flush=True)
        checkpoint=out/f"fold{fold}_verifier.pt"; torch.save({k:v.float().cpu() for k,v in model.state_dict().items()},checkpoint)
        oof.update(predict(model,data,heldout,device,a.batch_size)); history.append({"fold":fold,"class_counts":counts,"class_weights":weights.tolist(),"heldout_examples":len(heldout),"final_loss":float(loss.detach()),"checkpoint_sha256":sha256(checkpoint)})
    selected,fallback=select_rule(data.rows,oof,protocol["frozen"]["risk_multiplier_grid"],protocol["frozen"]["decision_threshold_grid"])
    key,multiplier,threshold,summary,chosen=selected
    rule={"risk_multiplier":multiplier,"threshold":threshold,"oof_summary":summary,"fallback":fallback,"selection_key":list(key)}
    write_metadata(out/"selected_rule.json",rule); write_jsonl(out/"oof_predictions.jsonl",[{"example_id":r["example_id"],"selected_index":c,**oof[r["example_id"]]} for r,c in zip(data.rows,chosen)])
    write_jsonl(out/"history.jsonl",history); write_metadata(out/"metadata.json",experiment_metadata(stage="v15_counterfactual_repair_verifier",development300_used=False,locked_roles_used=False))
    print(json.dumps(rule,indent=2))


if __name__ == "__main__": main()
