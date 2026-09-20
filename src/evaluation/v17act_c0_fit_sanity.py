"""Train-only A3 linear-head memorization diagnostic on a frozen 64-query subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.schemas import read_jsonl
from src.evaluation.v17act_c0_learning_chain_audit import independent_label, independent_outcome, raw_cache, select_ids
from src.training.train_v17traj_a3_1_boundary_editor_cv import action_feature, canonical_choices


def main():
    parser = argparse.ArgumentParser()
    for name in ("candidates", "data", "rollouts", "oof", "embeddings", "exact-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    oof = list(read_jsonl(args.oof))
    selected = select_ids(oof, 64)
    source = {r["example_id"]: r for r in read_jsonl(args.candidates) if r["example_id"] in selected}
    data = {r["example_id"]: r for r in read_jsonl(args.data) if r["example_id"] in selected}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts) if r["example_id"] in selected}
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    positions = {query: index for index, query in enumerate(embeddings["example_ids"])}
    xs, ys = [], []
    for query in sorted(selected):
        row = source[query]
        actions = canonical_choices(orders[query])
        masks = {sum(1 << packet for packet in action[:10]) for action in actions}
        cache = raw_cache(Path(args.exact_dir) / f"{query}.jsonl", masks)
        active = {float(level) for level in row["attainable_levels"]}
        outcomes = [independent_outcome(action, cache, active) for action in actions]
        ys.append([independent_label(item, outcomes[0]) for item in outcomes[1:]])
        index = positions[query]
        xs.append(torch.stack([action_feature(orders[query], action,
                                             embeddings["packets"][index], embeddings["questions"][index],
                                             data[query]["packet_tokens"]) for action in actions[1:]]))
    torch.set_num_threads(4)
    x = torch.stack(xs).float()
    y = torch.tensor(ys, dtype=torch.float32)
    mu = x.reshape(-1, x.shape[-1]).mean(0)
    sigma = x.reshape(-1, x.shape[-1]).std(0).clamp_min(1e-5)
    x = (x - mu) / sigma
    torch.manual_seed(20260920)
    model = torch.nn.Linear(x.shape[-1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    checkpoints = {}
    for step in range(1, 3001):
        logits = model(x).squeeze(-1)
        used = y != 0
        loss = F.softplus(-y[used] * logits[used]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (300, 3000):
            with torch.no_grad():
                logits = model(x).squeeze(-1)
                choice = torch.cat((torch.zeros((len(x), 1)), logits), dim=1).argmax(dim=1)
                picked = [int(y[i, action - 1]) if action else 0 for i, action in enumerate(choice.tolist())]
                checkpoints[str(step)] = {"loss": float(loss),
                    "labeled_sign_accuracy": float(((logits * y > 0) & used).sum() / used.sum()),
                    "selected_non_stay": int((choice != 0).sum()),
                    "selected_positive": sum(value == 1 for value in picked),
                    "selected_negative": sum(value == -1 for value in picked),
                    "selected_ignored": sum(value == 0 for value in picked) - int((choice == 0).sum())}
    result = {"protocol": "V17-ACT-C0_A3_SMALL_FIT_SANITY", "queries": len(x),
              "labels": {"positive": int((y == 1).sum()), "negative": int((y == -1).sum()), "ignored": int((y == 0).sum())},
              "checkpoints": checkpoints,
              "interpretation": "This reuses the A3 linear head on 64 selected train queries. Failure to memorize does not by itself establish a plumbing bug or absence of generalizable signal."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
