from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import histories_tensor, stable_fold
from src.model.v17d2_branch_adapter import Branch090AdaptedViabilityHead
from src.reproducibility import sha256, write_metadata
from src.training.train_v17b1_multi_anchor_viability import load_policy


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def boundary_scores(head: Branch090AdaptedViabilityHead, features: torch.Tensor) -> torch.Tensor:
    return head.frozen_head.viability(head.adapter(features))[..., 3]


def extract_reference_features(
    model: Any, records: list[dict[str, Any]], sources: dict[str, dict[str, Any]], cache: dict[str, Any],
    cache_index: dict[str, int], device: torch.device, batch_size: int,
) -> dict[str, Any]:
    chosen = []
    for row in records:
        values = [action["future_viability"][3] for action in row["actions"] if action["future_viability"][3] is not None]
        boundary = any(value == 1 for value in values) and any(value == 0 for value in values)
        if row["provenance"] == "deployed" or (row["provenance"] == "one_hop" and not boundary):
            chosen.append(row)
    features = []
    metadata = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chosen), batch_size):
            rows = chosen[start:start + batch_size]
            ids = torch.tensor([cache_index[row["example_id"]] for row in rows])
            packets = cache["packets"][ids].to(device=device, dtype=torch.float32)
            questions = cache["questions"][ids].to(device=device, dtype=torch.float32)
            histories, lengths = histories_tensor(rows, device)
            indices = torch.arange(len(rows), device=device)
            fractions = torch.tensor([sources[row["example_id"]]["packet_tokens"] for row in rows], device=device, dtype=torch.float32)
            fractions /= fractions.sum(1, keepdim=True)
            state, selected = model.history_states(packets, questions, histories, lengths, indices)
            phi, _ = model.action_features(packets, questions, state, selected, indices, fractions)
            for i, row in enumerate(rows):
                action = min(int(item["packet"]) for item in row["actions"])
                features.append(phi[i, action].detach().cpu().to(torch.float16))
                metadata.append({"example_id": row["example_id"], "provenance": row["provenance"]})
            if (start // batch_size + 1) % 20 == 0:
                print(json.dumps({"reference_extracted": min(start + batch_size, len(chosen)), "total": len(chosen)}), flush=True)
    return {"features": torch.stack(features), "metadata": metadata}


def train_fold(
    original_head: Any, features: torch.Tensor, states: list[dict[str, Any]], reference: dict[str, Any],
    train_indices: list[int], device: torch.device, config: dict[str, Any], seed: int,
) -> tuple[Branch090AdaptedViabilityHead, list[dict[str, float]]]:
    architecture = config["architecture"]
    settings = config["training_if_authorized"]
    head = Branch090AdaptedViabilityHead(original_head, features.shape[1], architecture["rank"], architecture["layer_norm_eps"]).to(device)
    optimizer = torch.optim.AdamW(head.trainable_parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = random.Random(seed)
    by_reference = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(reference["metadata"]):
        by_reference[row["provenance"]][row["example_id"]].append(index)
    history = []
    for epoch in range(settings["epochs"]):
        order = train_indices[:]; rng.shuffle(order); losses = []
        head.train()
        for start in range(0, len(order), settings["batch_states"]):
            selected_states = order[start:start + settings["batch_states"]]
            positive = [rng.choice(states[index]["positive"]) for index in selected_states]
            negative = [rng.choice(states[index]["negative"]) for index in selected_states]
            pos = features[positive].to(device=device, dtype=torch.float32)
            neg = features[negative].to(device=device, dtype=torch.float32)
            rank_loss = F.softplus(-(boundary_scores(head, pos) - boundary_scores(head, neg))).mean()
            ref_indices = []
            half = max(1, len(selected_states) // 2)
            for provenance, count in (("deployed", half), ("one_hop", len(selected_states) - half)):
                queries = sorted(by_reference[provenance])
                for _ in range(count):
                    query = rng.choice(queries); ref_indices.append(rng.choice(by_reference[provenance][query]))
            ref = reference["features"][ref_indices].to(device=device, dtype=torch.float32)
            drift_features = torch.cat((pos, neg, ref), dim=0)
            correction = head.adapter.correction(drift_features)
            relative_drift = (correction.square().sum(-1) / drift_features.square().sum(-1).clamp_min(1e-12)).mean()
            loss = rank_loss + settings["lambda_drift"] * relative_drift
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.trainable_parameters(), settings["gradient_clip_norm"])
            optimizer.step(); losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": mean(losses)})
    return head, history


def evaluate(head: Branch090AdaptedViabilityHead, features: torch.Tensor, states: list[dict[str, Any]], indices: list[int], device: torch.device) -> dict[str, float]:
    state_values = []; query_values = defaultdict(list); correct = total = 0
    head.eval()
    with torch.no_grad():
        for index in indices:
            row = states[index]
            pos = features[row["positive"]].to(device=device, dtype=torch.float32)
            neg = features[row["negative"]].to(device=device, dtype=torch.float32)
            matrix = boundary_scores(head, pos)[:, None] - boundary_scores(head, neg)[None, :]
            accuracy = float((matrix > 0).float().mean())
            state_values.append(accuracy); query_values[row["example_id"]].append(accuracy)
            correct += int((matrix > 0).sum()); total += matrix.numel()
    return {
        "pair_accuracy": correct / total,
        "macro_state_accuracy": mean(state_values),
        "macro_query_accuracy": mean(mean(values) for values in query_values.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V17-D2B frozen 0.90 branch-adapter training gate")
    parser.add_argument("--protocol-config", required=True); parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--boundary-features", required=True); parser.add_argument("--train-data", required=True)
    parser.add_argument("--embeddings", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--viability-head", required=True); parser.add_argument("--d1b-fold-results", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_PENDING_ARCHITECTURE_AUDIT": raise ValueError("unexpected protocol state")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.boundary_features, map_location="cpu", weights_only=False); features = bundle["features"]; states = bundle["states"]
    records = read_gzip(args.training_artifact); sources = {row["example_id"]: row for row in read_jsonl(args.train_data)}
    cache = torch.load(args.embeddings, map_location="cpu", weights_only=True); cache_index = {value: i for i, value in enumerate(cache["example_ids"])}
    model = load_policy(args.checkpoint, device, config["architecture"]["viability_hidden_dim"])
    model.viability_head.load_state_dict(torch.load(args.viability_head, map_location=device, weights_only=True)); original_head = model.viability_head
    reference_path = out / "reference_features.pt"
    if reference_path.exists(): reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    else:
        reference = extract_reference_features(model, records, sources, cache, cache_index, device, args.batch_size)
        torch.save(reference, reference_path)
    baseline_rows = read_gzip(args.d1b_fold_results)
    baseline = {row["fold"]: row["macro_state_accuracy"] for row in baseline_rows if row["split"] == "example_grouped" and row["probe"] == "linear"}
    assignments = [stable_fold(row["example_id"], 3) for row in states]; folds = []; histories = []
    for fold in range(3):
        train_indices = [i for i, value in enumerate(assignments) if value != fold]; test_indices = [i for i, value in enumerate(assignments) if value == fold]
        head, history = train_fold(original_head, features, states, reference, train_indices, device, config, config["training_if_authorized"]["seed"] + fold)
        metrics = evaluate(head, features, states, test_indices, device); metrics.update({"fold": fold, "d1b_macro_state_accuracy": baseline[fold], "not_below_d1b": metrics["macro_state_accuracy"] >= baseline[fold]})
        folds.append(metrics); histories.append({"fold": fold, "history": history}); print(json.dumps(metrics), flush=True)
    macro_state = mean(row["macro_state_accuracy"] for row in folds); macro_query = mean(row["macro_query_accuracy"] for row in folds)
    gate = config["held_out_query_gate"]
    passed = macro_state >= gate["primary_macro_state_accuracy_min"] and macro_query >= gate["query_balanced_macro_state_accuracy_min"] and all(row["not_below_d1b"] for row in folds)
    summary = {
        "complete": True, "folds": folds, "oof_macro_state_accuracy": macro_state, "oof_macro_query_accuracy": macro_query,
        "reference_states": len(reference["metadata"]), "reference_by_provenance": {key: sum(row["provenance"] == key for row in reference["metadata"]) for key in ("deployed", "one_hop")},
        "held_out_query_gate_passed": passed, "internal_read": False,
        "artifacts": {"protocol_sha256": sha256(args.protocol_config), "training_artifact_sha256": sha256(args.training_artifact), "viability_head_sha256": sha256(args.viability_head)},
    }
    write_jsonl(out / "cv_history.jsonl", histories); write_metadata(out / "held_out_query_summary.json", summary)
    decision = {"decision": "GO_V17D2B_FINAL_TRAIN_AND_INTERNAL_REPLAY" if passed else "STOP_V17D2B_HELD_OUT_QUERY_GATE", "gate_passed": passed, "internal_used": False, "development_used": False, "confirmation_used": False}
    write_metadata(out / "decision.json", decision); print(json.dumps({**decision, "summary": summary}, indent=2))


if __name__ == "__main__": main()
