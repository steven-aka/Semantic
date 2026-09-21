"""Three-fold train-side semantic probe for safe M2 exchange identification."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.evaluation.v17traj_m2a_promote_delay_preflight import rank7_remainders

CFG = Path("configs/v17traj_m2c_relative_exchange_probe.json")
BASE = Path("results/v2_rank_then_cut")
ROOT = BASE / "v17sel_b2b_lineage_clean_holdout"
SOURCE = BASE / "v17traj_m2b_promote_delay_pilot/per_query.jsonl"
OUT = BASE / "v17traj_m2c_relative_exchange_probe"


def fold(q, n):
    return int.from_bytes(hashlib.sha256(q.encode()).digest()[:8], "big") % n


def action_label(base, action):
    breaks = any(x is True and y is False for x, y in zip(base["hits"], action["hits"]))
    if breaks:
        return 0
    if not base["complete"] and action["complete"]:
        return 2
    return 1


def main():
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_OOF_DIAGNOSTIC"
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    rows = list(read_jsonl(SOURCE)); assert len(rows) == 128
    ids = {r["example_id"] for r in rows}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(ids)
    tok = AutoTokenizer.from_pretrained(cfg["encoder_repo"], use_fast=True)
    keys=[]; labels=[]; folds=[]; token_ids=[]; attention=[]; trunc=Counter()
    budget=cfg["field_token_budgets"]
    for r in rows:
        q=r["example_id"]; packets=data[q]["packet_texts"]; order=orders[q]; p7,p10=order[6],order[9]
        s6=render(packets,mask(order,6))
        delayed={i:s for i,s,_ in rank7_remainders(packets[p7])}
        promoted={i:s for i,s in enumerate(segments(packets[p10]))}
        base=r["scores"]["baseline"]
        for a in r["actions"]:
            fields={"query":data[q]["question"],"state_s6":s6,"promote":promoted[a["promoted_sentence_index"]],"delay":delayed[a["delayed_sentence_index"]]}
            built=[tok.cls_token_id]
            for name in ("query","state_s6","promote","delay"):
                raw=tok.encode(fields[name],add_special_tokens=False);trunc[name]+=len(raw)>budget[name]
                built.extend(raw[:budget[name]]);built.append(tok.sep_token_id)
            assert len(built)<=cfg["max_tokens"]
            pad=cfg["max_tokens"]-len(built)
            token_ids.append(built+[tok.pad_token_id]*pad);attention.append([1]*len(built)+[0]*pad)
            keys.append((q,a["arm"]));labels.append(action_label(base,r["scores"][a["arm"]]));folds.append(fold(q,cfg["folds"]))
    assert len(keys)==223 and Counter(labels)=={0:70,1:134,2:19}
    x=torch.tensor(token_ids);m=torch.tensor(attention);y=torch.tensor(labels);f=torch.tensor(folds)
    probs=np.zeros((len(keys),3),dtype=np.float32);hist=[];device=torch.device("cuda:0")
    for held in range(cfg["folds"]):
        train=torch.nonzero(f!=held,as_tuple=True)[0];valid=torch.nonzero(f==held,as_tuple=True)[0]
        counts=torch.bincount(y[train],minlength=3).float();assert torch.all(counts>0)
        weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
        model=AutoModelForSequenceClassification.from_pretrained(cfg["encoder_repo"],num_labels=3).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
        scaler=torch.amp.GradScaler("cuda")
        loader=DataLoader(TensorDataset(x[train],m[train],y[train]),batch_size=cfg["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(cfg["seed"]+held))
        losses=[]
        for epoch in range(cfg["epochs"]):
            model.train();e=[]
            for xb,mb,yb in loader:
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(input_ids=xb.to(device),attention_mask=mb.to(device)).logits
                    loss=torch.nn.functional.cross_entropy(logits.float(),yb.to(device),weight=weights.to(device))
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();e.append(float(loss.detach()))
            losses.append(float(np.mean(e)));print(json.dumps({"fold":held,"epoch":epoch+1,"loss":losses[-1]}),flush=True)
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(valid),cfg["batch_size"]):
                idx=valid[start:start+cfg["batch_size"]]
                with torch.autocast("cuda",dtype=torch.float16):
                    out=model(input_ids=x[idx].to(device),attention_mask=m[idx].to(device)).logits.softmax(-1)
                probs[idx.numpy()]=out.float().cpu().numpy()
        hist.append({"fold":held,"train":len(train),"valid":len(valid),"train_class_counts":counts.int().tolist(),"loss":losses})
        del model,opt,scaler;torch.cuda.empty_cache()
    pred={k:p for k,p in zip(keys,probs)}
    action_oof=[]
    for (q,arm), label_value, probability in zip(keys,labels,probs):
        action_oof.append({"example_id":q,"arm":arm,"label":cfg["labels"][label_value],
                           "probabilities":probability.tolist(),"score":float(probability[2]-probability[0])})
    ranked=[]
    for r in rows:
        q=r["example_id"]
        best=max(r["actions"],key=lambda a:(float(pred[q,a["arm"]][2]-pred[q,a["arm"]][0]),-a["delayed_sentence_index"]))
        ranked.append({"example_id":q,"fold":fold(q,cfg["folds"]),"score":float(pred[q,best["arm"]][2]-pred[q,best["arm"]][0]),"arm":best["arm"],"probabilities":pred[q,best["arm"]].tolist()})
    ranked.sort(key=lambda z:(-z["score"],z["example_id"]))
    rowmap={r["example_id"]:r for r in rows};reports=[]
    for fraction in cfg["report_intervention_budgets"]:
        selected=ranked[:math.ceil(len(rows)*fraction)];selected_map={z["example_id"]:z for z in selected}
        repairs=breaks=0;fold_repairs=set();ctx=tgt=0;success=[0]*5;complete=0
        opportunities=oracle_selected_repairs=0
        for r in rows:
            b=r["scores"]["baseline"];choice=r["scores"][selected_map[r["example_id"]]["arm"]] if r["example_id"] in selected_map else b
            safe_actions=[r["scores"][a["arm"]] for a in r["actions"] if action_label(b,r["scores"][a["arm"]])==2]
            if r["example_id"] in selected_map and safe_actions:
                opportunities+=1;oracle_selected_repairs+=1
            rep=not b["complete"] and choice["complete"] and not any(x is True and z is False for x,z in zip(b["hits"],choice["hits"]))
            br=b["complete"] and not choice["complete"]
            repairs+=rep;breaks+=br
            if rep:fold_repairs.add(fold(r["example_id"],cfg["folds"]))
            complete+=choice["complete"];ctx+=choice["cumulative_context_tokens"];tgt+=choice["cumulative_target_tokens"]
            success=[v+(choice["hits"][i] is True) for i,v in enumerate(success)]
        reports.append({"budget":fraction,"switches":len(selected),"opportunity_queries_found":opportunities,
                        "random_expected_opportunity_queries":len(selected)*15/len(rows),
                        "safe_complete_repairs":repairs,"oracle_action_repairs_on_selected_queries":oracle_selected_repairs,
                        "complete_breaks":breaks,"repair_folds":sorted(fold_repairs),"complete":complete,"five_anchor_success":success,"mean_cumulative_context_tokens":ctx/len(rows),"mean_cumulative_target_tokens":tgt/len(rows)})
    baseline_complete=sum(r["scores"]["baseline"]["complete"] for r in rows);baseline_ctx=sum(r["scores"]["baseline"]["cumulative_context_tokens"] for r in rows)/len(rows)
    primary=next(r for r in reports if r["budget"]==.1)
    passed=primary["safe_complete_repairs"]>=3 and primary["complete_breaks"]<=1 and primary["complete"]-baseline_complete>=2 and primary["mean_cumulative_context_tokens"]<=baseline_ctx and len(primary["repair_folds"])>=2
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"oof.jsonl").open("w") as h:
        for z in ranked:h.write(json.dumps(z)+"\n")
    with (OUT/"oof_actions.jsonl").open("w") as h:
        for z in action_oof:h.write(json.dumps(z)+"\n")
    def ap(class_index):
        pairs=sorted(((float(p[class_index]),label==class_index) for p,label in zip(probs,labels)),reverse=True)
        total=sum(v for _,v in pairs)
        return sum((sum(v for _,v in pairs[:i+1])/(i+1)) for i,(_,v) in enumerate(pairs) if v)/total
    summary={"protocol":cfg["protocol"],"queries":len(rows),"actions":len(keys),"label_counts":dict(Counter(labels)),
             "safe_action_average_precision":ap(2),"harm_action_average_precision":ap(0),
             "field_truncation_counts":dict(trunc),"fold_history":hist,"baseline_complete":baseline_complete,"baseline_success":[sum(r["scores"]["baseline"]["hits"][i] is True for r in rows) for i in range(5)],"baseline_mean_cumulative_context_tokens":baseline_ctx,"policies":reports,"primary_gate":"GO_M2D_INDEPENDENT_LABEL_SCALE_DESIGN" if passed else "STOP_M2C_RELATIVE_EXCHANGE_PROBE","checkpoint_saved":False,"new_target_calls":0,"sealed_sets_read":False,"limitations":"Small design-exposed OOF diagnostic with only 19 positive actions/15 positive queries; no checkpoint is retained."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps({"primary_gate":summary["primary_gate"],"policies":reports},indent=2))


if __name__=="__main__":main()
