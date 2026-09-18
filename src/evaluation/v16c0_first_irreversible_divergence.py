from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import load_model
from src.model.sequential_packet_policy import build_sequential_encoder_input
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def trajectory_levels(row: dict[str, Any]) -> list[float]:
    """Use attainable contracts rather than V10's fixed classifier grid."""
    return [float(value) for value in row.get("attainable_levels", row["active_levels"])]


def state_viability(
    dp: SequentialTrajectoryDP,
    mask: int,
    reached: int,
    incurred_tokens: int,
    *,
    oracle_tokens: int,
    full_tokens: int,
    regret_limit: float,
) -> tuple[bool, bool]:
    """Return future contract feasibility and per-trajectory regret feasibility."""
    suffix = dp.value(mask, reached)
    contract = suffix.reached_levels == len(dp.levels)
    minimum_total = incurred_tokens + suffix.additional_cumulative_tokens
    budget = oracle_tokens + regret_limit * len(dp.levels) * full_tokens
    return contract, contract and minimum_total <= budget + 1e-9


def first_zero_depth(trace: Sequence[dict[str, Any]], key: str) -> int | None:
    return next((int(row["depth"]) for row in trace if int(row[key]) == 0), None)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def distribution(values: list[int]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }

    failed = [row for row in rows if not row["top1_complete_contract"]]
    regret_failed = [row for row in rows if not row["top1_regret_qualified"]]
    primary_failed = [row for row in rows if row["primary_090_active"] and not row["top1_primary_090"]]
    primary_depths = [int(row["first_primary_090_extinction_depth"]) for row in rows if row["first_primary_090_extinction_depth"] is not None]
    exact_depths = [int(row["first_exact_optimal_extinction_depth"]) for row in rows if row["first_exact_optimal_extinction_depth"] is not None]
    contract_depths = [int(row["first_contract_extinction_depth"]) for row in rows if row["first_contract_extinction_depth"] is not None]
    regret_depths = [int(row["first_regret_qualified_extinction_depth"]) for row in rows if row["first_regret_qualified_extinction_depth"] is not None]

    def mechanisms(population: Sequence[dict[str, Any]]) -> dict[str, int]:
        return {
            "oracle_infeasible": sum(not row["oracle_complete_contract_feasible"] for row in population),
            "beam_pruning_extinction": sum(row["first_contract_extinction_depth"] is not None for row in population),
            "survived_but_terminal_top1_failed": sum(
                row["oracle_complete_contract_feasible"]
                and row["first_contract_extinction_depth"] is None
                and not row["top1_complete_contract"]
                for row in population
            ),
        }

    return {
        "complete": True,
        "examples": len(rows),
        "top1": {
            "primary_090_active_examples": sum(row["primary_090_active"] for row in rows),
            "primary_090_successes": sum(row["top1_primary_090"] for row in rows),
            "complete_contract_successes": sum(row["top1_complete_contract"] for row in rows),
            "regret_qualified_successes": sum(row["top1_regret_qualified"] for row in rows),
            "mean_regret_among_complete": (
                mean(row["top1_regret"] for row in rows if row["top1_regret"] is not None)
                if any(row["top1_regret"] is not None for row in rows) else None
            ),
        },
        "extinction_depth": {
            "primary_090_viable": distribution(primary_depths),
            "exact_dp_optimal": distribution(exact_depths),
            "contract_viable": distribution(contract_depths),
            "regret_qualified_contract_viable": distribution(regret_depths),
        },
        "failure_mechanism": {
            "primary_090_failures": len(primary_failed),
            "complete_contract_failures": len(failed),
            "regret_qualified_failures": len(regret_failed),
            "intrinsically_no_complete_contract_trajectory": sum(not row["oracle_complete_contract_feasible"] for row in rows),
            "among_primary_090_failures": {
                "beam_pruning_extinction": sum(row["first_primary_090_extinction_depth"] is not None for row in primary_failed),
                "survived_but_terminal_top1_failed": sum(row["first_primary_090_extinction_depth"] is None for row in primary_failed),
            },
            "among_complete_contract_failures": mechanisms(failed),
        },
    }


def beam_trace(
    model: Any,
    packets: torch.Tensor,
    question: torch.Tensor,
    fractions: torch.Tensor,
    dp: SequentialTrajectoryDP,
    *,
    beam_width: int,
    regret_limit: float,
) -> dict[str, Any]:
    dtype = model.initial_history.weight.dtype
    packets, question = packets.to(dtype), question.to(dtype)
    initial_reached = dp.attained[0]
    initial_cost = initial_reached * dp.tokens[0]
    oracle_tokens = initial_cost + dp.value(0, initial_reached).additional_cumulative_tokens
    full_tokens = dp.tokens[-1]
    oracle_complete, oracle_qualified = state_viability(
        dp, 0, initial_reached, initial_cost, oracle_tokens=oracle_tokens,
        full_tokens=full_tokens, regret_limit=regret_limit,
    )
    primary_index = next((i for i, level in enumerate(dp.levels) if abs(level - 0.9) < 1e-9), None)
    # score, history, recurrent state, mask, reached, incurred tokens, exact-optimal
    beams = [(0.0, (), torch.tanh(model.initial_history(question[None]))[0], 0, initial_reached, initial_cost, True)]
    trace = []
    for depth in range(1, 13):
        count = len(beams)
        states = torch.stack([row[2] for row in beams])
        selected = torch.zeros((count, 12), dtype=torch.bool, device=packets.device)
        for index, row in enumerate(beams):
            if row[1]:
                selected[index, list(row[1])] = True
        logits, _ = model.score_states(
            packets[None].expand(count, -1, -1), question[None].expand(count, -1),
            states, selected, torch.arange(count, device=packets.device), fractions[None].expand(count, -1),
        )
        log_prob = torch.log_softmax(logits.float(), dim=-1).cpu()
        next_states = model.history_gru(
            packets[None].expand(count, -1, -1).reshape(-1, model.model_dim),
            states[:, None, :].expand(-1, 12, -1).reshape(-1, model.model_dim),
        ).reshape(count, 12, model.model_dim)
        expanded = []
        for index, (score, history, _, mask, reached, incurred, exact) in enumerate(beams):
            optimal = set(dp.optimal_actions(mask, reached))
            for packet in range(12):
                if mask & (1 << packet):
                    continue
                next_mask = mask | (1 << packet)
                next_reached = max(reached, dp.attained[next_mask])
                next_incurred = incurred + (next_reached - reached) * dp.tokens[next_mask]
                contract, qualified = state_viability(
                    dp, next_mask, next_reached, next_incurred,
                    oracle_tokens=oracle_tokens, full_tokens=full_tokens, regret_limit=regret_limit,
                )
                primary_viable = primary_index is not None and dp.value(next_mask, next_reached).reached_levels >= primary_index + 1
                expanded.append((
                    score + float(log_prob[index, packet]), history + (packet,), next_states[index, packet],
                    next_mask, next_reached, next_incurred, exact and packet in optimal, contract, qualified, primary_viable,
                ))
        expanded.sort(key=lambda row: (-row[0], row[1]))
        threshold = float(expanded[min(beam_width, len(expanded)) - 1][0])
        best_contract = max((row[0] for row in expanded if row[7]), default=None)
        best_qualified = max((row[0] for row in expanded if row[8]), default=None)
        best_primary = max((row[0] for row in expanded if row[9]), default=None)
        kept = expanded[:beam_width]
        beams = [row[:7] for row in kept]
        trace.append({
            "depth": depth,
            "pruning_threshold": threshold,
            "exact_optimal_beams": sum(row[6] for row in kept),
            "contract_viable_beams": sum(row[7] for row in kept),
            "regret_qualified_beams": sum(row[8] for row in kept),
            "primary_090_viable_beams": sum(row[9] for row in kept),
            "expanded_exact_optimal_children": sum(row[6] for row in expanded),
            "expanded_contract_viable_children": sum(row[7] for row in expanded),
            "expanded_regret_qualified_children": sum(row[8] for row in expanded),
            "expanded_primary_090_viable_children": sum(row[9] for row in expanded),
            "best_contract_margin_to_threshold": None if best_contract is None else float(best_contract - threshold),
            "best_regret_qualified_margin_to_threshold": None if best_qualified is None else float(best_qualified - threshold),
            "best_primary_090_margin_to_threshold": None if best_primary is None else float(best_primary - threshold),
            "histories": [list(row[1]) for row in kept],
        })
    top = beams[0]
    top_contract, top_qualified = state_viability(
        dp, top[3], top[4], top[5], oracle_tokens=oracle_tokens,
        full_tokens=full_tokens, regret_limit=regret_limit,
    )
    regret = ((top[5] - oracle_tokens) / (len(dp.levels) * full_tokens)) if top_contract else None
    return {
        "decoded_order": list(top[1]),
        "primary_090_active": primary_index is not None,
        "top1_primary_090": primary_index is not None and top[4] >= primary_index + 1,
        "top1_complete_contract": top_contract,
        "top1_regret_qualified": top_qualified,
        "top1_regret": regret,
        "oracle_complete_contract_feasible": oracle_complete,
        "oracle_regret_qualified_feasible": oracle_qualified,
        "oracle_cumulative_tokens": oracle_tokens,
        "top1_cumulative_tokens": top[5],
        "first_exact_optimal_extinction_depth": first_zero_depth(trace, "exact_optimal_beams"),
        "first_primary_090_extinction_depth": first_zero_depth(trace, "primary_090_viable_beams"),
        "first_contract_extinction_depth": first_zero_depth(trace, "contract_viable_beams") if oracle_complete else None,
        "first_regret_qualified_extinction_depth": first_zero_depth(trace, "regret_qualified_beams") if oracle_qualified else None,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V16-C0 deployed first irreversible divergence audit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--role", choices=("train2863", "internal_validation300"), required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--regret-limit", type=float, default=0.03)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    all_rows = list(read_jsonl(args.data))
    rows = [row for index, row in enumerate(all_rows) if index % args.shard_count == args.shard_index]
    tokenizer, model = load_model(args)
    model.eval()
    device = torch.device("cuda:0")
    outputs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            inputs = [build_sequential_encoder_input(tokenizer, row["question"], row["packet_texts"], max_sequence_length=args.max_sequence_length) for row in batch]
            encoded_packets, encoded_questions = model.encode(inputs, pad_token_id=int(tokenizer.pad_token_id), device=device)
            for index, row in enumerate(batch):
                exact = list(read_jsonl(Path(args.exact_dir) / f"{row['example_id']}.jsonl", ExactSearchResult))
                dp = SequentialTrajectoryDP(exact, trajectory_levels(row))
                fractions = torch.tensor(row["packet_tokens"], device=device, dtype=torch.float32)
                fractions /= fractions.sum()
                result = beam_trace(model, encoded_packets[index], encoded_questions[index], fractions, dp, beam_width=args.beam_width, regret_limit=args.regret_limit)
                outputs.append({"example_id": row["example_id"], **result})
            if len(outputs) % 100 < len(batch) or len(outputs) == len(rows):
                print(json.dumps({"role": args.role, "processed": len(outputs), "total": len(rows)}), flush=True)
                write_jsonl(args.output, outputs)
    result = summarize(outputs)
    result.update({
        "role": args.role,
        "shard": {"index": args.shard_index, "count": args.shard_count, "source_examples": len(all_rows)},
        "definition": {
            "exact_optimal": "prefix has selected only exact lexicographic DP-optimal actions",
            "contract_viable": "some suffix can still attain every active fidelity anchor",
            "regret_qualified": f"contract viable and minimum final per-trajectory normalized regret <= {args.regret_limit}",
        },
        "artifacts": {"data": args.data, "data_sha256": sha256(args.data), "checkpoint": args.checkpoint, "checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"), "locked_roles_used": False},
    })
    write_jsonl(args.output, outputs)
    write_metadata(args.summary, result)
    write_metadata(f"{args.output}.metadata.json", experiment_metadata(stage="v16c0_first_irreversible_divergence", role=args.role, **result["artifacts"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
