from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.model.atomic_reveal_policy import AtomicRevealPolicy
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.exact_frontier import independent_frontier
from src.training.train_atomic_ordinal import OrdinalTrajectoryDataset, make_collator
from src.reproducibility import experiment_metadata, sha256, write_metadata


def _load_policy(args: argparse.Namespace) -> tuple[Any, AtomicRevealPolicy]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    lm = PeftModel.from_pretrained(base, Path(args.checkpoint) / "adapter")
    lm.config.use_cache = False
    policy = AtomicRevealPolicy(lm)
    state = torch.load(
        Path(args.checkpoint) / "threshold_head.pt", map_location="cpu", weights_only=True
    )
    policy.threshold_head.load_state_dict(state)
    policy.threshold_head.to(device="cuda:0", dtype=torch.bfloat16)
    return tokenizer, policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned atomic reveal trajectories")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--calibration")
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.data))
    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    tokenizer, policy = _load_policy(args)
    dataset = OrdinalTrajectoryDataset(source_rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collator(int(tokenizer.pad_token_id)),
    )
    device = torch.device("cuda:0")
    policy.eval()
    predicted_thresholds: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            output = policy(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["packet_spans"],
                batch["levels"].to(device),
                temperature=args.temperature,
            )
            for example_id, values in zip(batch["example_ids"], output["thresholds"]):
                predicted_thresholds[example_id] = values.float().cpu().tolist()

    output_rows = []
    ordinal_correct = ordinal_count = exact_trajectories = 0
    anchors = successes = all_success = 0
    feasible_policy_gaps = []
    feasible_total_gaps = []
    decomposition_residuals = []
    for source in source_rows:
        example_id = source["example_id"]
        levels = [float(value) for value in source["fidelity_levels"]]
        thresholds = predicted_thresholds[example_id]
        raw_states = [
            tuple(int(level >= threshold) for threshold in thresholds)
            for level in levels
        ]
        control_levels = (
            [float(calibration["control_levels"][str(level)]) for level in levels]
            if calibration else levels
        )
        if any(a > b for a, b in zip(control_levels, control_levels[1:])):
            raise ValueError("calibrated control levels must be nondecreasing")
        states = [
            tuple(int(level >= threshold) for threshold in thresholds)
            for level in control_levels
        ]
        if any(any(a > b for a, b in zip(low, high)) for low, high in zip(states, states[1:])):
            raise RuntimeError(f"{example_id}: threshold policy violated nestedness")
        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        by_state = {row.state: row for row in exact}
        independent = independent_frontier(exact, levels)
        nested = best_binary_nested_chain(exact, levels)
        full_tokens = by_state[(1,) * len(thresholds)].tokens
        trajectory_match = True
        contract_successes = []
        learned_sum = nested_sum = independent_sum = 0
        rows_for_example = []
        for index, (level, control_level, state) in enumerate(
            zip(levels, control_levels, states)
        ):
            mask = bool(source["anchor_mask"][index])
            gold = source["nested_states"][index]
            if mask:
                for packet_index, value in enumerate(state):
                    ordinal_correct += int(
                        raw_states[index][packet_index]
                        == int(source["ordinal_labels"][packet_index][index])
                    )
                    ordinal_count += 1
                trajectory_match &= list(raw_states[index]) == gold
            learned = by_state[state]
            success = learned.fidelity >= level
            if mask:
                anchors += 1
                successes += int(success)
                contract_successes.append(success)
            nested_tokens = nested[index]["tokens"]
            independent_tokens = independent[index]["tokens"]
            if mask and success:
                learned_sum += learned.tokens
                nested_sum += int(nested_tokens)
                independent_sum += int(independent_tokens)
            rows_for_example.append(
                {
                    "fidelity_level": level,
                    "control_fidelity_level": control_level,
                    "anchor_mask": mask,
                    "raw_predicted_state": list(raw_states[index]),
                    "predicted_state": list(state),
                    "predicted_tokens": learned.tokens,
                    "achieved_fidelity": learned.fidelity,
                    "contract_success": success,
                    "independent_tokens": independent_tokens,
                    "nested_tokens": nested_tokens,
                    "learned_minus_nested_normalized": (
                        (learned.tokens - int(nested_tokens)) / full_tokens
                        if mask and success else None
                    ),
                }
            )
        exact_trajectories += int(trajectory_match)
        every_contract = bool(contract_successes) and all(contract_successes)
        all_success += int(every_contract)
        policy_gap = total_gap = residual = None
        if every_contract:
            denominator = len(contract_successes) * full_tokens
            policy_gap = (learned_sum - nested_sum) / denominator
            structural_gap = (nested_sum - independent_sum) / denominator
            total_gap = (learned_sum - independent_sum) / denominator
            residual = total_gap - (structural_gap + policy_gap)
            feasible_policy_gaps.append(policy_gap)
            feasible_total_gaps.append(total_gap)
            decomposition_residuals.append(abs(residual))
        output_rows.append(
            {
                "example_id": example_id,
                "predicted_thresholds": thresholds,
                "representation_nested": True,
                "all_active_contracts_success": every_contract,
                "trajectory_policy_gap_normalized": policy_gap,
                "trajectory_total_gap_normalized": total_gap,
                "gap_decomposition_residual": residual,
                "anchors": rows_for_example,
            }
        )

    write_jsonl(args.output, output_rows)
    summary = {
        "complete": True,
        "examples": len(source_rows),
        "representation_nested_fraction": 1.0,
        "ordinal_accuracy": ordinal_correct / ordinal_count,
        "exact_trajectory_fraction": exact_trajectories / len(source_rows),
        "active_contract_success_fraction": successes / anchors,
        "all_active_contracts_success_fraction": all_success / len(source_rows),
        "feasible_trajectory_examples": len(feasible_policy_gaps),
        "mean_policy_gap_normalized_feasible": (
            mean(feasible_policy_gaps) if feasible_policy_gaps else None
        ),
        "mean_total_gap_normalized_feasible": (
            mean(feasible_total_gaps) if feasible_total_gaps else None
        ),
        "maximum_gap_decomposition_residual": (
            max(decomposition_residuals) if decomposition_residuals else None
        ),
        "infeasible_predictions_are_violations_not_negative_gaps": True,
        "calibrated": calibration is not None,
        "calibration_valid": (
            bool(calibration.get("calibration_valid", True)) if calibration else None
        ),
        "calibration_sha256": sha256(args.calibration) if args.calibration else None,
        "artifacts": {
            "data_sha256": sha256(args.data),
            "checkpoint_metadata_sha256": sha256(
                Path(args.checkpoint) / "training_metadata.json"
            ),
        },
    }
    write_metadata(args.summary, summary)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="v1_atomic_ordinal_policy_evaluation",
            data=args.data,
            exact_dir=args.exact_dir,
            checkpoint=args.checkpoint,
            temperature=args.temperature,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
