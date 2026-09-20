"""One frozen grouped OOF semantic probe for late sentence selection."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments

CFG=Path("configs/v17packet_r2_semantic_probe.json")
ROOT=Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
R1=Path("results/v2_rank_then_cut/v17packet_r1_label_scale512")
OUT=Path("results/v2_rank_then_cut/v17packet_r2_semantic_probe")


def fold(q):
    return int.from_bytes(hashlib.sha256(q.encode()).digest()[:8],"big")%4


def title(packet):
    return packet.split("\n",1)[0]


def main():
    cfg=json.loads(CFG.read_text())
    assert cfg["status"]=="FROZEN_TRAIN_SIDE_OOF"
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    rows=list(read_jsonl(R1/"per_query.jsonl"))
    assert len(rows)==512
    ids={r["example_id"] for r in rows}
    source={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(ROOT/"sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    texts=[];labels=[];keys=[];folds=[]
    for row in rows:
        q=row["example_id"]
        packet_texts=source[q]["packet_texts"]
        order=orders[q]
        current_titles="; ".join(title(packet_texts[p]) for p in order[:8])
        displaced=packet_texts[order[8]]
        parts=segments(packet_texts[order[9]])
        assert len(parts)==len(row["sentence_options"])
        for option,fragment in zip(row["sentence_options"],parts):
            texts.append(f"[QUERY] {source[q]['question']} [STATE_TITLES] {current_titles} "
                         f"[CANDIDATE] {fragment} [DISPLACED] {displaced}")
            labels.append(int(option["hits"][3] is True))
            keys.append((q,option["index"]))
            folds.append(fold(q))
    assert len(keys)==1485
    tok=AutoTokenizer.from_pretrained(cfg["encoder_repo"],use_fast=True)
    encoded=tok(texts,truncation=True,max_length=cfg["max_tokens"],padding="max_length",return_tensors="pt")
    ids_tensor=encoded["input_ids"];mask_tensor=encoded["attention_mask"]
    label_tensor=torch.tensor(labels,dtype=torch.long)
    predictions=np.zeros(len(keys),dtype=np.float32)
    histories=[]
    device=torch.device("cuda:0")
    for held in range(cfg["folds"]):
        train=torch.tensor([i for i,f in enumerate(folds) if f!=held])
        valid=torch.tensor([i for i,f in enumerate(folds) if f==held])
        assert len(train)+len(valid)==len(keys)
        model=AutoModelForSequenceClassification.from_pretrained(cfg["encoder_repo"],num_labels=2).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
        scaler=torch.amp.GradScaler("cuda")
        loader=DataLoader(TensorDataset(ids_tensor[train],mask_tensor[train],label_tensor[train]),
                          batch_size=cfg["batch_size"],shuffle=True,
                          generator=torch.Generator().manual_seed(cfg["seed"]+held))
        history={"fold":held,"train_actions":len(train),"valid_actions":len(valid),"epoch_loss":[]}
        for epoch in range(cfg["epochs"]):
            model.train();losses=[]
            for ids_batch,attn_batch,y_batch in loader:
                ids_batch=ids_batch.to(device);attn_batch=attn_batch.to(device);y_batch=y_batch.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    out=model(input_ids=ids_batch,attention_mask=attn_batch,labels=y_batch)
                if not torch.isfinite(out.loss):raise ValueError("nonfinite semantic probe loss")
                scaler.scale(out.loss).backward();scaler.step(opt);scaler.update()
                losses.append(float(out.loss.detach()))
            history["epoch_loss"].append(float(np.mean(losses)))
            print(json.dumps({"fold":held,"epoch":epoch+1,"loss":history["epoch_loss"][-1]}),flush=True)
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(valid),cfg["batch_size"]):
                idx=valid[start:start+cfg["batch_size"]]
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(input_ids=ids_tensor[idx].to(device),
                                 attention_mask=mask_tensor[idx].to(device)).logits
                predictions[idx.numpy()]=logits.float().softmax(-1)[:,1].cpu().numpy()
        histories.append(history)
        del model,opt,scaler;torch.cuda.empty_cache()
    score={key:float(value) for key,value in zip(keys,predictions)}
    outrows=[]
    for row in rows:
        q=row["example_id"]
        eligible=[a for a in row["sentence_options"] if a["context_tokens"]<=row["baseline"]["context_tokens"]]
        best=max(eligible,key=lambda a:(score[q,a["index"]],-a["index"])) if eligible else None
        outrows.append({"example_id":q,"fold":fold(q),"baseline":row["baseline"],
                        "best_action":best,"best_score":score[q,best["index"]] if best else None})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"oof.jsonl").open("w") as handle:
        for row in outrows:handle.write(json.dumps(row)+"\n")
    policies=[]
    for fraction in cfg["descriptive_fractions"]:
        n=math.ceil(len(outrows)*fraction)
        ranked=sorted((r for r in outrows if r["best_action"]),key=lambda r:r["best_score"],reverse=True)
        selected={r["example_id"] for r in ranked[:n]}
        chosen=[r["best_action"] if r["example_id"] in selected else r["baseline"] for r in outrows]
        policies.append({"switch_fraction":fraction,"switches":len(selected),
                         "success_090":sum(a["hits"][3] is True for a in chosen),
                         "complete":sum(a["complete"] for a in chosen),
                         "mean_cumulative_context_tokens":sum(a["context_tokens"] for a in chosen)/len(chosen),
                         "repairs":sum(r["baseline"]["hits"][3] is False and a["hits"][3] is True for r,a in zip(outrows,chosen)),
                         "breaks":sum(r["baseline"]["hits"][3] is True and a["hits"][3] is False for r,a in zip(outrows,chosen))})
    primary=next(p for p in policies if p["switch_fraction"]==cfg["primary_switch_fraction"])
    passed=(primary["repairs"]-primary["breaks"]>=3 and primary["complete"]>=273
            and primary["mean_cumulative_context_tokens"]<=2227.728515625)
    summary={"protocol":cfg["protocol"],"queries":len(rows),"actions":len(keys),
             "fold_history":histories,"action_label_prevalence":sum(labels)/len(labels),
             "baseline_090":381,"baseline_complete":273,"baseline_mean_context":2227.728515625,
             "policies":policies,"primary_gate":"GO_R2_SEMANTIC_RESEARCH_VALIDATION" if passed else
             "STOP_R2_SEMANTIC_OOF_GATE","new_target_calls":0,
             "checkpoint_saved":False,"sealed_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
