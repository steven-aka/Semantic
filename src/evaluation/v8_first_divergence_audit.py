from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import load_model
from src.model.sequential_packet_policy import (
    SequentialPacketPolicy,
    build_sequential_encoder_input,
)
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def _mask_and_reached(
    dp: SequentialTrajectoryDP, history: Sequence[int]
) -> tuple[int, int]:
    mask = 0
    reached = dp.attained[0]
    for packet in history:
        mask |= 1 << int(packet)
        reached = max(reached, dp.attained[mask])
    return mask, reached


def _beam_trace(
    model: SequentialPacketPolicy,
    packets: torch.Tensor,
    question: torch.Tensor,
    packet_token_fractions: torch.Tensor,
    dp: SequentialTrajectoryDP,
    *,
    beam_width: int,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    dtype = model.initial_history.weight.dtype
    packets = packets.to(dtype)
    question = question.to(dtype)
    initial_reached = dp.attained[0]
    beams: list[tuple[float, tuple[int, ...], torch.Tensor, int, int, bool]] = [
        (
            0.0,
            (),
            torch.tanh(model.initial_history(question[None]))[0],
            0,
            initial_reached,
            True,
        )
    ]
    trace: list[dict[str, Any]] = []
    for depth in range(12):
        beam_count = len(beams)
        states = torch.stack([value[2] for value in beams])
        selected = torch.zeros((beam_count, 12), dtype=torch.bool, device=packets.device)
        for beam_index, (_, history, _, _, _, _) in enumerate(beams):
            if history:
                selected[beam_index, list(history)] = True
        logits, _ = model.score_states(
            packets[None].expand(beam_count, -1, -1),
            question[None].expand(beam_count, -1),
            states,
            selected,
            torch.arange(beam_count, dtype=torch.long, device=packets.device),
            packet_token_fractions[None].expand(beam_count, -1),
        )
        log_prob = torch.log_softmax(logits.float(), dim=-1).cpu()
        next_states = model.history_gru(
            packets[None].expand(beam_count, -1, -1).reshape(-1, model.model_dim),
            states[:, None, :].expand(-1, 12, -1).reshape(-1, model.model_dim),
        ).reshape(beam_count, 12, model.model_dim)
        expanded: list[tuple[float, tuple[int, ...], torch.Tensor, int, int, bool]] = []
        for beam_index, (score, history, _, mask, reached, consistent) in enumerate(beams):
            optimal = dp.optimal_actions(mask, reached)
            for packet in range(12):
                if mask & (1 << packet):
                    continue
                next_mask = mask | (1 << packet)
                next_reached = max(reached, dp.attained[next_mask])
                expanded.append(
                    (
                        score + float(log_prob[beam_index, packet].item()),
                        history + (packet,),
                        next_states[beam_index, packet],
                        next_mask,
                        next_reached,
                        consistent and packet in optimal,
                    )
                )
        expanded.sort(key=lambda value: (-value[0], value[1]))
        beams = expanded[:beam_width]
        trace.append(
            {
                "depth": depth + 1,
                "oracle_consistent_beams": sum(value[5] for value in beams),
                "best_score": beams[0][0],
                "worst_retained_score": beams[-1][0],
                "histories": [list(value[1]) for value in beams],
            }
        )
    return beams[0][1], trace


def _prefix_policy_rows(
    model: SequentialPacketPolicy,
    packets: torch.Tensor,
    question: torch.Tensor,
    packet_token_fractions: torch.Tensor,
    order: Sequence[int],
    dp: SequentialTrajectoryDP,
    fixed_pool: set[tuple[int, ...]],
) -> list[dict[str, Any]]:
    device = packets.device
    histories = torch.full((12, 11), -1, dtype=torch.long, device=device)
    lengths = torch.arange(12, dtype=torch.long, device=device)
    for depth in range(1, 12):
        histories[depth, :depth] = torch.tensor(order[:depth], dtype=torch.long, device=device)
    indices = torch.zeros(12, dtype=torch.long, device=device)
    states, selected = model.history_states(
        packets[None], question[None], histories, lengths, indices
    )
    logits, _ = model.score_states(
        packets[None],
        question[None],
        states,
        selected,
        indices,
        packet_token_fractions[None],
    )
    rows = []
    for depth, chosen in enumerate(order):
        history = tuple(int(value) for value in order[:depth])
        mask, reached = _mask_and_reached(dp, history)
        optimal = dp.optimal_actions(mask, reached)
        remaining = [packet for packet in range(12) if not mask & (1 << packet)]
        scores = logits[depth].float()
        ranked = sorted(remaining, key=lambda packet: (-float(scores[packet].item()), packet))
        ranks = {packet: index + 1 for index, packet in enumerate(ranked)}
        probabilities = torch.softmax(scores[remaining], dim=0)
        probability_by_packet = {
            packet: float(probabilities[index].item()) for index, packet in enumerate(remaining)
        }
        action_values = dict(dp.action_values(mask, reached))
        best = min(action_values.values(), key=lambda value: value.comparison_key)
        selected_value = action_values[int(chosen)]
        rows.append(
            {
                "depth": depth,
                "history": list(history),
                "fixed_pool_covered": history in fixed_pool,
                "chosen": int(chosen),
                "chosen_rank": ranks[int(chosen)],
                "chosen_is_optimal": int(chosen) in optimal,
                "optimal_actions": list(optimal),
                "optimal_best_rank": min(ranks[action] for action in optimal),
                "optimal_probability_mass": sum(probability_by_packet[action] for action in optimal),
                "optimal_in_local_top8": min(ranks[action] for action in optimal) <= min(8, len(remaining)),
                "best_final_reached": best.reached_levels,
                "chosen_final_reached": selected_value.reached_levels,
                "lost_reachable_anchors": best.reached_levels - selected_value.reached_levels,
                "chosen_additional_tokens": selected_value.additional_cumulative_tokens,
                "best_additional_tokens": best.additional_cumulative_tokens,
                "additional_token_excess_if_equal_reach": (
                    selected_value.additional_cumulative_tokens - best.additional_cumulative_tokens
                    if selected_value.reached_levels == best.reached_levels
                    else None
                ),
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = [row["first_divergence"] for row in rows if row["first_divergence"] is not None]
    ranks = [int(row["optimal_best_rank"]) for row in first]
    masses = [float(row["optimal_probability_mass"]) for row in first]
    extinction = [row["oracle_beam_extinction_depth"] for row in rows]
    extinct = [int(value) for value in extinction if value is not None]
    return {
        "complete": True,
        "examples": len(rows),
        "examples_with_nonoptimal_decoded_action": len(first),
        "first_divergence": {
            "mean_depth": mean(row["depth"] for row in first) if first else None,
            "median_depth": median(row["depth"] for row in first) if first else None,
            "mean_optimal_best_rank": mean(ranks) if ranks else None,
            "median_optimal_best_rank": median(ranks) if ranks else None,
            "optimal_best_rank_gt8": sum(rank > 8 for rank in ranks),
            "optimal_in_local_top8": sum(row["optimal_in_local_top8"] for row in first),
            "mean_optimal_probability_mass": mean(masses) if masses else None,
            "median_optimal_probability_mass": median(masses) if masses else None,
            "fixed_pool_covered": sum(row["fixed_pool_covered"] for row in first),
            "immediately_loses_reachable_anchor": sum(
                row["lost_reachable_anchors"] > 0 for row in first
            ),
        },
        "beam": {
            "oracle_consistent_path_survives_to_depth12": sum(value is None for value in extinction),
            "oracle_consistent_path_extinguished": len(extinct),
            "mean_extinction_depth": mean(extinct) if extinct else None,
            "median_extinction_depth": median(extinct) if extinct else None,
        },
        "decoded_prefix_fixed_pool_coverage_fraction": (
            sum(sum(item["fixed_pool_covered"] for item in row["prefixes"]) for row in rows)
            / (12 * len(rows))
            if rows
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V8 first divergences and beam survival")
    parser.add_argument("--data", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fidelity-level", type=float, default=0.9)
    args = parser.parse_args()

    details = list(read_jsonl(args.details))
    failed_ids = {
        row["example_id"]
        for row in details
        if not next(
            anchor
            for anchor in row["anchors"]
            if abs(float(anchor["fidelity_level"]) - args.fidelity_level) < 1e-9
        )["contract_success"]
    }
    sources = [row for row in read_jsonl(args.data) if row["example_id"] in failed_ids]
    if len(sources) != len(failed_ids):
        raise RuntimeError("failed detail IDs do not match the supplied data")

    tokenizer, model = load_model(args)
    model.eval()
    device = torch.device("cuda:0")
    output_rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(sources), args.batch_size):
            batch = sources[start : start + args.batch_size]
            inputs = [
                build_sequential_encoder_input(
                    tokenizer,
                    row["question"],
                    row["packet_texts"],
                    max_sequence_length=args.max_sequence_length,
                )
                for row in batch
            ]
            encoded_packets, encoded_questions = model.encode(
                inputs, pad_token_id=int(tokenizer.pad_token_id), device=device
            )
            for index, source in enumerate(batch):
                exact = list(
                    read_jsonl(Path(args.exact_dir) / f"{source['example_id']}.jsonl", ExactSearchResult)
                )
                dp = SequentialTrajectoryDP(exact, source["active_levels"])
                fractions = torch.tensor(source["packet_tokens"], device=device, dtype=torch.float32)
                fractions /= fractions.sum()
                order, beam_trace = _beam_trace(
                    model,
                    encoded_packets[index],
                    encoded_questions[index],
                    fractions,
                    dp,
                    beam_width=args.beam_width,
                )
                fixed_pool = {tuple(item["history"]) for item in source["histories"]}
                prefix_rows = _prefix_policy_rows(
                    model,
                    encoded_packets[index],
                    encoded_questions[index],
                    fractions,
                    order,
                    dp,
                    fixed_pool,
                )
                first = next((row for row in prefix_rows if not row["chosen_is_optimal"]), None)
                extinction = next(
                    (row["depth"] for row in beam_trace if row["oracle_consistent_beams"] == 0),
                    None,
                )
                output_rows.append(
                    {
                        "example_id": source["example_id"],
                        "decoded_order": list(order),
                        "first_divergence": first,
                        "oracle_beam_extinction_depth": extinction,
                        "prefixes": prefix_rows,
                        "beam_trace": beam_trace,
                    }
                )
            print(json.dumps({"processed": min(start + len(batch), len(sources)), "total": len(sources)}), flush=True)

    summary = summarize(output_rows)
    summary["artifacts"] = {
        "data": args.data,
        "data_sha256": sha256(args.data),
        "details": args.details,
        "details_sha256": sha256(args.details),
        "checkpoint": args.checkpoint,
        "checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"),
        "exact_dir": args.exact_dir,
        "locked_roles_used": False,
    }
    write_jsonl(args.output, output_rows)
    write_metadata(args.summary, summary)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(stage="v8_first_divergence_audit", **summary["artifacts"]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
