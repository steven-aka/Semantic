"""Fixed grouped OOF lexical probe for R1 sentence safety; no deployment model."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import segments

CFG = Path("configs/v17packet_r2_lexical_probe.json")
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
R1 = Path("results/v2_rank_then_cut/v17packet_r1_label_scale512")
OUT = Path("results/v2_rank_then_cut/v17packet_r2_lexical_probe")


def hashed_features(texts, dimension):
    v = torch.zeros(dimension, dtype=torch.float32)
    for prefix, text in texts:
        terms = re.findall(r"[\w]+", text.lower())
        grams = terms + [terms[i] + "_" + terms[i+1] for i in range(len(terms)-1)]
        for token in grams:
            digest = hashlib.blake2b((prefix+":"+token).encode(), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % dimension
            v[idx] += 1.0
    norm = torch.linalg.vector_norm(v)
    return v / norm if norm > 0 else v


def main():
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_DIAGNOSTIC"
    rows = list(read_jsonl(R1 / "per_query.jsonl"))
    assert len(rows) == 512
    ids = {r["example_id"] for r in rows}
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
              if r["example_id"] in ids}
    order = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl")
             if r["example_id"] in ids}
    x, labels, keys, folds = [], [], [], []
    dimension = cfg["hash_features"]
    for row in rows:
        q = row["example_id"]
        question = source[q]["question"]
        rank9 = source[q]["packet_texts"][order[q][8]]
        rank10 = source[q]["packet_texts"][order[q][9]]
        pieces = segments(rank10)
        assert len(pieces) == len(row["sentence_options"])
        group_fold = int(hashlib.sha256(q.encode()).hexdigest(),16) % cfg["folds"]
        for action, fragment in zip(row["sentence_options"],pieces):
            x.append(hashed_features([("q",question),("candidate",fragment),("displaced",rank9)],dimension))
            labels.append(float(action["hits"][3] is True))
            keys.append((q,action["index"]))
            folds.append(group_fold)
    features = torch.stack(x)
    targets = torch.tensor(labels,dtype=torch.float32)
    fold_ids = torch.tensor(folds)
    preds = torch.zeros(len(keys))
    for heldout in range(cfg["folds"]):
        torch.manual_seed(20260920)
        model = torch.nn.Linear(dimension,1)
        optim = torch.optim.Adam(model.parameters(),lr=0.03,weight_decay=0.001)
        train = fold_ids != heldout
        for _ in range(100):
            optim.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features[train]).flatten(),targets[train])
            loss.backward()
            optim.step()
        with torch.no_grad():
            preds[~train] = torch.sigmoid(model(features[~train]).flatten())
    scores = {key:float(value) for key,value in zip(keys,preds)}
    outrows=[]
    for row in rows:
        q=row["example_id"]
        eligible=[a for a in row["sentence_options"] if a["context_tokens"]<=row["baseline"]["context_tokens"]]
        best=max(eligible,key=lambda a:(scores[q,a["index"]],-a["index"])) if eligible else None
        outrows.append({"example_id":q,"baseline":row["baseline"],
                        "best_action":best,"best_score":scores[q,best["index"]] if best else None})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"oof.jsonl").open("w") as handle:
        for row in outrows:handle.write(json.dumps(row)+"\n")
    reports=[]
    for threshold in cfg["thresholds"]:
        chosen=[r["best_action"] if r["best_action"] and r["best_score"]>=threshold else r["baseline"] for r in outrows]
        reports.append({"threshold":threshold,"switches":sum(a is not r["baseline"] for a,r in zip(chosen,outrows)),
                        "success_090":sum(a["hits"][3] is True for a in chosen),
                        "complete":sum(a["complete"] for a in chosen),
                        "mean_cumulative_context_tokens":sum(a["context_tokens"] for a in chosen)/len(chosen),
                        "repairs":sum(r["baseline"]["hits"][3] is False and a["hits"][3] is True for r,a in zip(outrows,chosen)),
                        "breaks":sum(r["baseline"]["hits"][3] is True and a["hits"][3] is False for r,a in zip(outrows,chosen))})
    summary={"protocol":cfg["protocol"],"queries":len(rows),"candidate_sentences":len(keys),
             "action_success_prevalence":sum(labels)/len(labels),
             "baseline_090":sum(r["baseline"]["hits"][3] is True for r in rows),
             "policies":reports,"limitations":cfg["interpretation"]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
