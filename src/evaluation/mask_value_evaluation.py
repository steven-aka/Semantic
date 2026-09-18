from __future__ import annotations

from statistics import mean
from pathlib import Path
from typing import Any, Sequence

import torch

from src.data.schemas import ExactSearchResult, read_jsonl
from src.model.mask_value_head import MaskValueHead
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.predicted_mask_trajectory import best_order_from_predicted_attainment
from src.search.rank_then_cut import best_prefix_nested_chain


class CachedMaskValueEncoder:
    def __init__(self, packets: torch.Tensor, questions: torch.Tensor):
        self.packets = packets
        self.questions = questions

    def eval(self):
        return self

    def encode(self, indices, *, pad_token_id: int, device: torch.device):
        selected = torch.tensor(indices, dtype=torch.long)
        return self.packets[selected].to(device), self.questions[selected].to(device)


def logits_to_attainment(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("V10 deployment requires all five fixed anchor logits")
    return (logits >= 0).sum(dim=1)


def predict_lattice(
    head: MaskValueHead,
    packets: torch.Tensor,
    question: torch.Tensor,
    *,
    chunk_size: int,
) -> list[int]:
    if chunk_size <= 0:
        raise ValueError("mask chunk size must be positive")
    output = []
    device = packets.device
    for start in range(0, 4096, chunk_size):
        masks = torch.arange(start, min(start + chunk_size, 4096), device=device)
        logits = head(
            packets[None],
            question[None],
            masks,
            torch.zeros(len(masks), dtype=torch.long, device=device),
        )
        output.extend(int(value) for value in logits_to_attainment(logits).cpu())
    return output


def evaluate_mask_value_policy(
    encoder: SequentialPacketPolicy,
    head: MaskValueHead,
    rows: Sequence[dict[str, Any]],
    inputs: Sequence[Any],
    *,
    exact_dir: str | Path,
    pad_token_id: int,
    device: torch.device,
    encoder_batch_size: int = 8,
    mask_chunk_size: int = 256,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoder.eval()
    head.eval()
    outputs = []
    anchor_success = anchors = trajectory_success = 0
    regrets = []
    exact_state_correct = exact_state_total = 0
    per_level: dict[float, list[tuple[bool, float | None]]] = {}
    predicted_cutoff_success = predicted_cutoff_anchors = predicted_cutoff_trajectories = 0
    classification = {level: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for level in range(5)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), encoder_batch_size):
            batch_rows = rows[start : start + encoder_batch_size]
            batch_inputs = inputs[start : start + encoder_batch_size]
            packets, questions = encoder.encode(
                batch_inputs, pad_token_id=pad_token_id, device=device
            )
            for batch_index, source in enumerate(batch_rows):
                levels = [float(value) for value in source["attainable_levels"]]
                predicted = predict_lattice(
                    head,
                    packets[batch_index],
                    questions[batch_index],
                    chunk_size=mask_chunk_size,
                )
                true_attained = [int(item["attained_levels"]) for item in source["mask_values"]]
                if len(true_attained) != 4096:
                    raise ValueError("evaluation requires a complete mask lattice")
                if any(int(item["mask"]) != index for index, item in enumerate(source["mask_values"])):
                    raise ValueError("evaluation mask lattice must be ordered from 0 through 4095")
                exact_state_correct += sum(a == b for a, b in zip(predicted, true_attained))
                exact_state_total += 4096
                for truth, guess in zip(true_attained, predicted):
                    for level in range(5):
                        actual = truth > level
                        estimate = guess > level
                        key = "tp" if actual and estimate else "fp" if estimate else "fn" if actual else "tn"
                        classification[level][key] += 1
                tokens = [int(item["tokens"]) for item in source["mask_values"]]
                order = best_order_from_predicted_attainment(predicted, tokens, 5)
                exact = list(
                    read_jsonl(Path(exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult)
                )
                oracle = best_binary_nested_chain(exact, levels)
                learned = best_prefix_nested_chain(exact, order, levels)
                by_state = {item.state: item for item in exact}
                full_tokens = by_state[(1,) * 12].tokens
                prefix_masks = [0]
                for packet in order:
                    prefix_masks.append(prefix_masks[-1] | (1 << packet))
                predicted_cutoff_rows = []
                predicted_all_success = True
                for level_index, level in enumerate(levels, 1):
                    cutoff = next(
                        (count for count, mask in enumerate(prefix_masks) if predicted[mask] >= level_index),
                        None,
                    )
                    actual = None if cutoff is None else by_state[
                        tuple(int(prefix_masks[cutoff] & (1 << bit) != 0) for bit in range(12))
                    ]
                    success = actual is not None and actual.fidelity >= level
                    predicted_cutoff_success += int(success)
                    predicted_cutoff_anchors += 1
                    predicted_all_success &= success
                    predicted_cutoff_rows.append({
                        "fidelity_level": level,
                        "predicted_cutoff_count": cutoff,
                        "contract_success": success,
                        "tokens": None if actual is None else actual.tokens,
                        "achieved_fidelity": None if actual is None else actual.fidelity,
                    })
                predicted_cutoff_trajectories += int(predicted_all_success)
                all_success = True
                learned_tokens = oracle_tokens = 0
                anchor_rows = []
                for level, estimate, baseline in zip(levels, learned, oracle):
                    success = bool(estimate["feasible"])
                    all_success &= success
                    anchor_success += int(success)
                    anchors += 1
                    value = int(estimate["tokens"]) if estimate["tokens"] is not None else None
                    learned_tokens += value or 0
                    oracle_tokens += int(baseline["tokens"])
                    regret = (
                        (value - int(baseline["tokens"])) / full_tokens
                        if success and value is not None
                        else None
                    )
                    per_level.setdefault(level, []).append((success, regret))
                    anchor_rows.append(
                        {
                            "fidelity_level": level,
                            "contract_success": success,
                            "cutoff_count": estimate["cutoff_count"],
                            "tokens": value,
                            "oracle_nested_tokens": baseline["tokens"],
                            "rate_regret_normalized_if_success": regret,
                            "state": estimate["state"],
                        }
                    )
                trajectory_success += int(all_success)
                trajectory_regret = (
                    (learned_tokens - oracle_tokens) / (len(levels) * full_tokens)
                    if all_success
                    else None
                )
                if trajectory_regret is not None:
                    regrets.append(trajectory_regret)
                outputs.append(
                    {
                        "example_id": source["example_id"],
                        "decoded_order": list(order),
                        "all_active_contracts_success": all_success,
                        "oracle_cutoff_ranking_regret_normalized": trajectory_regret,
                        "predicted_cutoff_all_active_contracts_success": predicted_all_success,
                        "predicted_cutoffs": predicted_cutoff_rows,
                        "anchors": anchor_rows,
                    }
                )
    return {
        "complete": True,
        "examples": len(outputs),
        "exact_attained_class_accuracy": exact_state_correct / exact_state_total,
        "mask_classification": {
            str(level): {
                **values,
                "recall": values["tp"] / (values["tp"] + values["fn"]),
                "precision": values["tp"] / (values["tp"] + values["fp"])
                if values["tp"] + values["fp"]
                else None,
            }
            for level, values in classification.items()
            if values["tp"] + values["fn"] + values["fp"] + values["tn"] > 0
        },
        "oracle_cutoff_active_contract_success_fraction": anchor_success / anchors,
        "oracle_cutoff_all_active_contracts_success_fraction": trajectory_success / len(outputs),
        "oracle_cutoff_feasible_trajectory_examples": len(regrets),
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": mean(regrets) if regrets else None,
        "predicted_cutoff_active_contract_success_fraction": (
            predicted_cutoff_success / predicted_cutoff_anchors
        ),
        "predicted_cutoff_all_active_contracts_success_fraction": (
            predicted_cutoff_trajectories / len(outputs)
        ),
        "per_level": {
            str(level): {
                "examples": len(values),
                "contract_successes": sum(success for success, _ in values),
                "contract_success_fraction": mean(success for success, _ in values),
                "mean_rate_regret_normalized_successful": (
                    mean(regret for _, regret in values if regret is not None)
                    if any(regret is not None for _, regret in values)
                    else None
                ),
            }
            for level, values in sorted(per_level.items())
        },
        "decoder": "label-free exact DP over five fixed predicted ordinal anchors",
    }, outputs
