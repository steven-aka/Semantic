from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.model.sequential_policy_loss import sequential_policy_loss
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.training.sequential_policy_data import SequentialHistoryCollator, SequentialHistoryDataset


def head_state_dict(model: SequentialPacketPolicy) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().float().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("lm.")
    }


def load_head_state(model: SequentialPacketPolicy, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("lm.") for key in result.missing_keys):
        raise ValueError(f"invalid sequential head state: {result}")


def evaluate_loaded_policy(
    model: SequentialPacketPolicy,
    loader: DataLoader,
    *,
    exact_dir: str | Path,
    pad_token_id: int,
    device: torch.device,
    beam_width: int = 8,
    progress_weight: float = 0.2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    outputs: list[dict[str, Any]] = []
    action_losses = []
    progress_losses = []
    action_accuracies = []
    anchor_success = anchors = trajectory_success = 0
    regrets = []
    per_level: dict[float, list[tuple[bool, float | None]]] = {}
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action_logits, progress_logits, packets, question = model.forward_states(
                    batch["inputs"],
                    batch["histories"].to(device),
                    batch["history_lengths"].to(device),
                    batch["example_indices"].to(device),
                    batch["packet_token_fractions"].to(device),
                    pad_token_id=pad_token_id,
                    device=device,
                )
                loss = sequential_policy_loss(
                    action_logits,
                    progress_logits,
                    batch["optimal_actions"],
                    batch["reached_levels"].to(device),
                    batch["active_level_counts"].to(device),
                    progress_weight=progress_weight,
                )
            action_losses.append(float(loss.action.item()))
            progress_losses.append(float(loss.progress.item()))
            action_accuracies.append(loss.action_accuracy)
            for batch_index, source in enumerate(batch["rows"]):
                order = model.beam_order(
                    packets[batch_index],
                    question[batch_index],
                    batch["packet_token_fractions"][batch_index].to(device),
                    beam_width=beam_width,
                )
                example_id = source["example_id"]
                levels = [float(value) for value in source["active_levels"]]
                exact = list(
                    read_jsonl(Path(exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
                )
                by_state = {row.state: row for row in exact}
                full_tokens = by_state[(1,) * 12].tokens
                oracle = best_binary_nested_chain(exact, levels)
                learned = best_prefix_nested_chain(exact, order, levels)
                all_success = True
                learned_tokens = oracle_tokens = 0
                anchor_rows = []
                for level, predicted, baseline in zip(levels, learned, oracle):
                    success = bool(predicted["feasible"])
                    all_success &= success
                    anchor_success += int(success)
                    anchors += 1
                    tokens = int(predicted["tokens"]) if predicted["tokens"] is not None else None
                    learned_tokens += tokens or 0
                    oracle_tokens += int(baseline["tokens"])
                    regret = (
                        (tokens - int(baseline["tokens"])) / full_tokens
                        if success and tokens is not None else None
                    )
                    per_level.setdefault(level, []).append((success, regret))
                    anchor_rows.append(
                        {
                            "fidelity_level": level,
                            "contract_success": success,
                            "cutoff_count": predicted["cutoff_count"],
                            "tokens": tokens,
                            "oracle_nested_tokens": baseline["tokens"],
                            "rate_regret_normalized_if_success": regret,
                            "state": predicted["state"],
                        }
                    )
                trajectory_success += int(all_success)
                trajectory_regret = (
                    (learned_tokens - oracle_tokens) / (len(levels) * full_tokens)
                    if all_success else None
                )
                if trajectory_regret is not None:
                    regrets.append(trajectory_regret)
                outputs.append(
                    {
                        "example_id": example_id,
                        "decoded_order": list(order),
                        "all_active_contracts_success": all_success,
                        "oracle_cutoff_ranking_regret_normalized": trajectory_regret,
                        "anchors": anchor_rows,
                    }
                )
    summary = {
        "complete": True,
        "examples": len(outputs),
        "validation_action_loss": mean(action_losses),
        "validation_progress_loss": mean(progress_losses),
        "validation_set_action_accuracy": mean(action_accuracies),
        "oracle_cutoff_active_contract_success_fraction": anchor_success / anchors,
        "oracle_cutoff_all_active_contracts_success_fraction": trajectory_success / len(outputs),
        "oracle_cutoff_feasible_trajectory_examples": len(regrets),
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": mean(regrets) if regrets else None,
        "per_level": {
            str(level): {
                "examples": len(values),
                "contract_successes": sum(success for success, _ in values),
                "contract_success_fraction": mean(success for success, _ in values),
                "mean_rate_regret_normalized_successful": (
                    mean(regret for _, regret in values if regret is not None)
                    if any(regret is not None for _, regret in values) else None
                ),
            }
            for level, values in sorted(per_level.items())
        },
        "decoder": f"history-aware sequential policy beam width {beam_width}",
    }
    return summary, outputs


def load_model(args: argparse.Namespace) -> tuple[Any, SequentialPacketPolicy]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    lm = PeftModel.from_pretrained(base, Path(args.checkpoint) / "adapter")
    lm.config.use_cache = False
    model = SequentialPacketPolicy(lm, model_dim=args.model_dim)
    state = torch.load(Path(args.checkpoint) / "sequential_head.pt", map_location="cpu", weights_only=True)
    load_head_state(model, state)
    model.move_head(device="cuda:0", dtype=torch.bfloat16)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V8 sequential packet ordering")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    args = parser.parse_args()
    rows = list(read_jsonl(args.data))
    tokenizer, model = load_model(args)
    dataset = SequentialHistoryDataset(rows, tokenizer, args.max_sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SequentialHistoryCollator(20260912),
    )
    summary, outputs = evaluate_loaded_policy(
        model,
        loader,
        exact_dir=args.exact_dir,
        pad_token_id=int(tokenizer.pad_token_id),
        device=torch.device("cuda:0"),
        beam_width=args.beam_width,
    )
    summary["artifacts"] = {
        "data_sha256": sha256(args.data),
        "checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"),
    }
    write_jsonl(args.output, outputs)
    write_metadata(args.summary, summary)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(stage="v8_sequential_policy_evaluation", data=args.data, checkpoint=args.checkpoint),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
