from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schemas import write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.reproducibility import sha256, write_metadata
from src.training.train_v17b1_multi_anchor_viability import load_policy


class LinearScorer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__(); self.weight = nn.Linear(dimension, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight(value).squeeze(-1)


def tap_features(full: torch.Tensor, transition_gru: nn.Module, tap: str, device: torch.device, batch: int = 4096) -> torch.Tensor:
    d = 512
    candidate, history, selected, query = full[:, :d], full[:, d:2*d], full[:, 2*d:3*d], full[:, 4*d:5*d]
    if tap == "candidate": return candidate
    if tap == "candidate_times_history": return full[:, 5*d:6*d]
    if tap == "candidate_times_selected": return full[:, 6*d:7*d]
    if tap == "candidate_query_interaction": return torch.cat((candidate, query, candidate * query), dim=-1)
    if tap == "full_action_features": return full
    outputs = []
    transition_gru.eval()
    with torch.no_grad():
        for start in range(0, len(full), batch):
            c = candidate[start:start+batch].to(device=device, dtype=torch.float32)
            h = history[start:start+batch].to(device=device, dtype=torch.float32)
            outputs.append(transition_gru(c, h).cpu().to(torch.float16))
    nxt = torch.cat(outputs)
    if tap == "transition_next": return nxt
    if tap == "transition_delta": return nxt - history
    if tap == "transition_query_interaction": return torch.cat((nxt, query, nxt * query), dim=-1)
    raise ValueError(tap)


def normalize(train_features: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    average = train_features.float().mean(0)
    std = train_features.float().std(0, unbiased=False).clamp_min(1e-6)
    return ((value.float() - average) / std).to(torch.float16)


def train_probe(features: torch.Tensor, states: list[dict[str, Any]], indices: list[int], config: dict[str, Any], device: torch.device, seed: int) -> LinearScorer:
    settings = config["probe"]; rng = random.Random(seed); torch.manual_seed(seed)
    model = LinearScorer(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    for _ in range(settings["epochs"]):
        order = indices[:]; rng.shuffle(order); model.train()
        for start in range(0, len(order), settings["batch_states"]):
            rows = order[start:start+settings["batch_states"]]
            positive = [rng.choice(states[index]["positive"]) for index in rows]
            negative = [rng.choice(states[index]["negative"]) for index in rows]
            pos = features[positive].to(device=device, dtype=torch.float32)
            neg = features[negative].to(device=device, dtype=torch.float32)
            loss = F.softplus(-(model(pos) - model(neg))).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model


def evaluate_scores(scores: torch.Tensor, states: list[dict[str, Any]], indices: list[int]) -> tuple[dict[str, float], dict[str, float]]:
    state_accuracy = []; query_values = defaultdict(list); correct = total = 0
    for index in indices:
        row = states[index]; matrix = scores[row["positive"]][:, None] - scores[row["negative"]][None, :]
        accuracy = float((matrix > 0).float().mean()); state_accuracy.append(accuracy); query_values[row["example_id"]].append(accuracy)
        correct += int((matrix > 0).sum()); total += matrix.numel()
    query_accuracy = {query: mean(values) for query, values in query_values.items()}
    return {"pair_accuracy": correct/total, "macro_state_accuracy": mean(state_accuracy), "macro_query_accuracy": mean(query_accuracy.values())}, query_accuracy


def paired_bootstrap(candidate: dict[str, float], baseline: dict[str, float], replicates: int, seed: int) -> dict[str, float]:
    keys = sorted(set(candidate) & set(baseline)); differences = [candidate[key] - baseline[key] for key in keys]
    rng = random.Random(seed); values = []
    for _ in range(replicates): values.append(mean(rng.choice(differences) for _ in keys))
    values.sort()
    return {"mean_query_improvement": mean(differences), "lower_95": values[int(.025*replicates)], "upper_95": values[min(replicates-1,int(.975*replicates))]}


def main() -> None:
    parser = argparse.ArgumentParser(description="V17-E0 frozen information bottleneck localization")
    parser.add_argument("--protocol-config", required=True); parser.add_argument("--boundary-features", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--viability-head", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN": raise ValueError("protocol is not approved")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); bundle = torch.load(args.boundary_features, map_location="cpu", weights_only=False)
    full = bundle["features"]; states = bundle["states"]; assignments = [stable_fold(row["example_id"], 3) for row in states]
    model = load_policy(args.checkpoint, device, 128); model.viability_head.load_state_dict(torch.load(args.viability_head, map_location=device, weights_only=True)); model.eval()
    results = []; query_results: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    frozen_head_scores = []
    with torch.no_grad():
        for start in range(0, len(full), 4096): frozen_head_scores.append(model.viability_head.viability(full[start:start+4096].to(device=device,dtype=torch.float32))[...,3].cpu())
    frozen_head_scores = torch.cat(frozen_head_scores)
    for fold in range(3):
        test = [i for i,value in enumerate(assignments) if value == fold]
        metrics, queries = evaluate_scores(frozen_head_scores, states, test); results.append({"tap":"frozen_head_090","fold":fold,**metrics}); query_results["frozen_head_090"][fold]=queries
    for tap in config["taps"]:
        raw = tap_features(full, model.history_gru, tap, device)
        for fold in range(3):
            train = [i for i,value in enumerate(assignments) if value != fold]; test = [i for i,value in enumerate(assignments) if value == fold]
            train_action_indices = sorted({action for index in train for action in states[index]["positive"] + states[index]["negative"]})
            normalized = normalize(raw[train_action_indices], raw)
            probe = train_probe(normalized, states, train, config, device, config["probe"]["seed"] + fold)
            with torch.no_grad():
                score_chunks = [probe(normalized[start:start+4096].to(device=device,dtype=torch.float32)).cpu() for start in range(0,len(normalized),4096)]
            metrics, queries = evaluate_scores(torch.cat(score_chunks), states, test); row={"tap":tap,"fold":fold,"dimension":raw.shape[1],**metrics}; results.append(row);query_results[tap][fold]=queries;print(json.dumps(row),flush=True)
    aggregates = {}
    full_queries = {key:value for fold in query_results["full_action_features"].values() for key,value in fold.items()}
    gate = config["sidecar_gate"]
    for tap in ["frozen_head_090",*config["taps"]]:
        rows=[row for row in results if row["tap"]==tap]; queries={key:value for fold in query_results[tap].values() for key,value in fold.items()}
        bootstrap=paired_bootstrap(queries,full_queries,gate["bootstrap_replicates"],gate["bootstrap_seed"])
        aggregates[tap]={"macro_state_accuracy":mean(row["macro_state_accuracy"] for row in rows),"macro_query_accuracy":mean(row["macro_query_accuracy"] for row in rows),"fold_macro_state":[row["macro_state_accuracy"] for row in rows],"paired_vs_full":bootstrap}
        aggregates[tap]["sidecar_gate_passed"]=(aggregates[tap]["macro_state_accuracy"]>=gate["macro_state_accuracy_min"] and aggregates[tap]["macro_query_accuracy"]>=gate["macro_query_accuracy_min"] and min(aggregates[tap]["fold_macro_state"])>=gate["every_fold_macro_state_min"] and bootstrap["lower_95"]>gate["paired_query_bootstrap_lower_improvement_over_full"])
    passing=[tap for tap in config["taps"] if aggregates[tap]["sidecar_gate_passed"]]
    interaction_best=max(aggregates[tap]["macro_query_accuracy"] for tap in ("candidate_query_interaction","transition_query_interaction"))
    simple_best=max(aggregates[tap]["macro_query_accuracy"] for tap in ("candidate","transition_next","transition_delta"))
    if passing: decision="GO_V17E1_UPSTREAM_BOUNDARY_SIDECAR_DESIGN"
    elif interaction_best > simple_best: decision="GO_V17E1_QUERY_CONDITIONED_REPRESENTATION_DESIGN"
    else: decision="STOP_FROZEN_LINEAR_TAP_BRANCH_AUDIT_TARGET_AND_FEATURE_SUFFICIENCY"
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);write_jsonl(out/"fold_results.jsonl",results)
    summary={"complete":True,"aggregates":aggregates,"passing_taps":passing,"decision":decision,"frozen_head_role":"descriptive in-sample reference only; B1 trained it on all train2863 queries, so it is not an out-of-fold gate candidate","internal_used":False,"development_used":False,"confirmation_used":False,"artifacts":{"protocol_sha256":sha256(args.protocol_config),"boundary_features_sha256":sha256(args.boundary_features),"viability_head_sha256":sha256(args.viability_head)}}
    write_metadata(out/"summary.json",summary);write_metadata(out/"decision.json",{"decision":decision,"passing_taps":passing,"model_training_authorized":False,"internal_used":False,"development_used":False,"confirmation_used":False});print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
