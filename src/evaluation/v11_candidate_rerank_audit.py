"""Consumed-development audit of V10 value reranking over frozen sequential beams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import load_head_state
from src.model.mask_value_head import MaskValueHead
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain


class _NoBackbone(nn.Module):
    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)


def terminal_beams(
    model: SequentialPacketPolicy,
    packets: torch.Tensor,
    question: torch.Tensor,
    packet_token_fractions: torch.Tensor,
    *,
    beam_width: int,
) -> list[tuple[float, tuple[int, ...]]]:
    dtype = model.initial_history.weight.dtype
    packets = packets.to(dtype)
    question = question.to(dtype)
    beams: list[tuple[float, tuple[int, ...], torch.Tensor]] = [
        (0.0, (), torch.tanh(model.initial_history(question[None]))[0])
    ]
    for _ in range(12):
        count = len(beams)
        states = torch.stack([row[2] for row in beams])
        selected = torch.zeros((count, 12), dtype=torch.bool, device=packets.device)
        for index, (_, history, _) in enumerate(beams):
            if history:
                selected[index, list(history)] = True
        logits, _ = model.score_states(
            packets[None].expand(count, -1, -1),
            question[None].expand(count, -1),
            states,
            selected,
            torch.arange(count, device=packets.device),
            packet_token_fractions[None].expand(count, -1),
        )
        log_prob = torch.log_softmax(logits.float(), dim=-1).cpu()
        next_states = model.history_gru(
            packets[None].expand(count, -1, -1).reshape(-1, model.model_dim),
            states[:, None, :].expand(-1, 12, -1).reshape(-1, model.model_dim),
        ).reshape(count, 12, model.model_dim)
        expanded = []
        for index, (score, history, _) in enumerate(beams):
            for packet in range(12):
                if selected[index, packet]:
                    continue
                expanded.append(
                    (score + float(log_prob[index, packet]), history + (packet,), next_states[index, packet])
                )
        expanded.sort(key=lambda row: (-row[0], row[1]))
        beams = expanded[:beam_width]
    return [(score, order) for score, order, _ in beams]


def prefix_masks(order: Sequence[int]) -> list[int]:
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << int(packet)))
    return masks


def predicted_path_key(order: Sequence[int], predicted: dict[int, int], tokens: Sequence[int]) -> tuple[int, int]:
    reached = int(predicted[0])
    cumulative = reached * int(tokens[0])
    for mask in prefix_masks(order)[1:]:
        next_reached = max(reached, int(predicted[mask]))
        cumulative += (next_reached - reached) * int(tokens[mask])
        reached = next_reached
    return -reached, cumulative


def true_path_key(anchors: Sequence[dict[str, Any]]) -> tuple[int, int]:
    successes = sum(bool(row["contract_success"]) for row in anchors)
    tokens = sum(int(row["tokens"]) for row in anchors if row["contract_success"])
    return -successes, tokens


def summarize(details: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    chosen = [row[field] for row in details]
    levels = sorted({float(a["fidelity_level"]) for row in chosen for a in row["anchors"]})
    per_level = {}
    for level in levels:
        values = [next(a for a in row["anchors"] if float(a["fidelity_level"]) == level) for row in chosen if any(float(a["fidelity_level"]) == level for a in row["anchors"])]
        per_level[str(level)] = {
            "examples": len(values),
            "contract_successes": sum(bool(value["contract_success"]) for value in values),
        }
    regrets = [row["trajectory_regret"] for row in chosen if row["trajectory_regret"] is not None]
    active = [a for row in chosen for a in row["anchors"]]
    return {
        "examples": len(chosen),
        "per_level": per_level,
        "active_contract_success_fraction": sum(a["contract_success"] for a in active) / len(active),
        "complete_trajectory_successes": sum(row["all_success"] for row in chosen),
        "complete_trajectory_success_fraction": sum(row["all_success"] for row in chosen) / len(chosen),
        "feasible_trajectory_examples": len(regrets),
        "mean_normalized_regret_feasible": mean(regrets) if regrets else None,
    }


def evaluate_order(order, exact, levels, oracle, full_tokens):
    learned = best_prefix_nested_chain(exact, order, levels)
    anchors = []
    learned_total = oracle_total = 0
    for level, estimate, baseline in zip(levels, learned, oracle):
        success = bool(estimate["feasible"])
        value = int(estimate["tokens"]) if estimate["tokens"] is not None else None
        learned_total += value or 0
        oracle_total += int(baseline["tokens"])
        anchors.append({"fidelity_level": level, "contract_success": success, "tokens": value})
    all_success = all(row["contract_success"] for row in anchors)
    return {
        "order": list(order),
        "anchors": anchors,
        "all_success": all_success,
        "trajectory_regret": (learned_total - oracle_total) / (len(levels) * full_tokens) if all_success else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--v10-head", required=True)
    parser.add_argument("--sequential-head", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    rows = list(read_jsonl(args.data))
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache["example_ids"] != [row["example_id"] for row in rows]:
        raise ValueError("cache IDs differ from data")
    packets = cache["packets"]
    questions = cache["questions"]
    candidate_sets: dict[str, list[list[tuple[float, tuple[int, ...]]]]] = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for label, path in args.sequential_head:
            model = SequentialPacketPolicy(_NoBackbone(), model_dim=512)
            load_head_state(model, torch.load(path, map_location="cpu", weights_only=True))
            model.move_head(device=device, dtype=torch.bfloat16)
            model.eval()
            all_beams = []
            for index, row in enumerate(rows):
                fractions = torch.tensor(row["packet_tokens"], device=device, dtype=torch.float32)
                fractions /= fractions.sum()
                all_beams.append(terminal_beams(model, packets[index].to(device), questions[index].to(device), fractions, beam_width=args.beam_width))
            candidate_sets[label] = all_beams
            del model
            torch.cuda.empty_cache()
        value_head = MaskValueHead(model_dim=512, layers=2, heads=8, dropout=0.1)
        value_head.load_state_dict(torch.load(args.v10_head, map_location="cpu", weights_only=True))
        value_head.to(device).eval()
        details = []
        for index, row in enumerate(rows):
            sources = {}
            unique = {}
            for label, beams in candidate_sets.items():
                for rank, (score, order) in enumerate(beams[index], 1):
                    unique.setdefault(order, []).append({"model": label, "rank": rank, "log_probability": score})
                    if rank == 1:
                        sources[label] = order
            masks = sorted({mask for order in unique for mask in prefix_masks(order)})
            mask_tensor = torch.tensor(masks, device=device)
            logits = value_head(
                packets[index:index+1].to(device), questions[index:index+1].to(device), mask_tensor,
                torch.zeros(len(masks), device=device, dtype=torch.long),
            )
            predicted = {mask: int(value) for mask, value in zip(masks, (logits >= 0).sum(dim=1).cpu())}
            tokens = [int(item["tokens"]) for item in row["mask_values"]]
            predicted_choice = min(unique, key=lambda order: (predicted_path_key(order, predicted, tokens), order))
            exact = list(read_jsonl(Path(args.exact_dir) / f"{row['example_id']}.jsonl", ExactSearchResult))
            levels = [float(value) for value in row["attainable_levels"]]
            oracle = best_binary_nested_chain(exact, levels)
            full_tokens = next(item.tokens for item in exact if item.state == (1,) * 12)
            evaluated = {
                order: evaluate_order(order, exact, levels, oracle, full_tokens)
                for order in unique
            }
            oracle_choice = min(unique, key=lambda order: (true_path_key(evaluated[order]["anchors"]), order))
            record = {
                "example_id": row["example_id"],
                "candidate_orders": len(unique),
                "v10_rerank": evaluated[predicted_choice],
                "candidate_oracle": evaluated[oracle_choice],
                "v10_predicted_key": list(predicted_path_key(predicted_choice, predicted, tokens)),
                "v10_selected_provenance": unique[predicted_choice],
                "oracle_selected_provenance": unique[oracle_choice],
            }
            for label, order in sources.items():
                record[label] = evaluated[order]
            details.append(record)
    summary = {
        "complete": True,
        "scientific_role": "post-stop consumed-development diagnostic; no fresh role opened",
        "candidate_pool": f"union of terminal beam-{args.beam_width} orders from frozen sequential heads",
        "selector": "frozen V10 hard five-anchor mask values; lexicographic reached anchors then cumulative first-crossing tokens",
        "mean_unique_candidate_orders": mean(row["candidate_orders"] for row in details),
        "results": {field: summarize(details, field) for field in [*[label for label, _ in args.sequential_head], "v10_rerank", "candidate_oracle"]},
        "artifacts": {
            "data_sha256": sha256(args.data), "cache_sha256": sha256(args.cache), "v10_head_sha256": sha256(args.v10_head),
            "sequential_heads": {label: sha256(path) for label, path in args.sequential_head},
        },
    }
    write_jsonl(args.output, details)
    write_metadata(args.summary, summary)
    write_metadata(f"{args.output}.metadata.json", experiment_metadata(stage="v11_candidate_rerank_audit", **summary["artifacts"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
