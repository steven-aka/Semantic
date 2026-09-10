from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.model.atomic_packet_ranker import AtomicPacketRanker
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.training.train_atomic_ranker import RankOracleDataset, make_rank_collator


def _load(args: argparse.Namespace) -> tuple[Any, AtomicPacketRanker]:
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
    model = AtomicPacketRanker(lm, head_hidden_size=args.head_hidden_size)
    state = torch.load(Path(args.checkpoint) / "rank_head.pt", map_location="cpu", weights_only=True)
    model.rank_head.load_state_dict(state)
    model.rank_head.to(device="cuda:0", dtype=torch.bfloat16)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned order with oracle monotone cutoff")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-hidden-size", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.data))
    tokenizer, model = _load(args)
    dataset = RankOracleDataset(source_rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_rank_collator(int(tokenizer.pad_token_id)),
    )
    model.eval()
    device = torch.device("cuda:0")
    scores_by_id = {}
    correct = pairs = 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["packet_spans"],
            )
            scores = output["scores"].float().cpu()
            for batch_index, example_id in enumerate(batch["example_ids"]):
                values = scores[batch_index].tolist()
                scores_by_id[example_id] = values
                for winner, loser in batch["preferences"][batch_index]:
                    correct += int(values[winner] > values[loser])
                    pairs += 1

    outputs = []
    anchor_success = anchors = all_successes = 0
    regrets = []
    by_level: dict[float, list[tuple[bool, float | None]]] = {}
    for source in source_rows:
        example_id = source["example_id"]
        scores = scores_by_id[example_id]
        order = tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))
        levels = [float(value) for value in source["active_levels"]]
        exact = list(
            read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult)
        )
        by_state = {row.state: row for row in exact}
        full_tokens = by_state[(1,) * len(scores)].tokens
        oracle = best_binary_nested_chain(exact, levels)
        learned = best_prefix_nested_chain(exact, order, levels)
        rows = []
        all_success = True
        learned_tokens = oracle_tokens = 0
        for level, predicted, baseline in zip(levels, learned, oracle):
            success = bool(predicted["feasible"])
            all_success &= success
            anchor_success += int(success)
            anchors += 1
            tokens = int(predicted["tokens"]) if predicted["tokens"] is not None else None
            if tokens is not None:
                learned_tokens += tokens
            oracle_tokens += int(baseline["tokens"])
            anchor_regret = (
                (tokens - int(baseline["tokens"])) / full_tokens
                if success and tokens is not None else None
            )
            by_level.setdefault(level, []).append((success, anchor_regret))
            rows.append(
                {
                    "fidelity_level": level,
                    "contract_success": success,
                    "cutoff_count": predicted["cutoff_count"],
                    "tokens": tokens,
                    "oracle_nested_tokens": baseline["tokens"],
                    "rate_regret_normalized_if_success": anchor_regret,
                    "state": predicted["state"],
                }
            )
        all_successes += int(all_success)
        trajectory_regret = None
        if all_success:
            trajectory_regret = (learned_tokens - oracle_tokens) / (len(levels) * full_tokens)
            regrets.append(trajectory_regret)
        outputs.append(
            {
                "example_id": example_id,
                "scores": scores,
                "learned_order": list(order),
                "all_active_contracts_success": all_success,
                "oracle_cutoff_ranking_regret_normalized": trajectory_regret,
                "anchors": rows,
            }
        )

    summary = {
        "complete": True,
        "examples": len(outputs),
        "identifiable_pairs": pairs,
        "identifiable_pairwise_accuracy": correct / pairs,
        "oracle_cutoff_active_contract_success_fraction": anchor_success / anchors,
        "oracle_cutoff_all_active_contracts_success_fraction": all_successes / len(outputs),
        "oracle_cutoff_feasible_trajectory_examples": len(regrets),
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": mean(regrets) if regrets else None,
        "per_level": {
            str(level): {
                "examples": len(values),
                "contract_success_fraction": mean(int(success) for success, _ in values),
                "mean_rate_regret_normalized_successful": mean(
                    regret for _, regret in values if regret is not None
                ) if any(regret is not None for _, regret in values) else None,
            }
            for level, values in sorted(by_level.items())
        },
        "exact_trajectory_fraction_is_not_a_primary_metric": True,
        "artifacts": {
            "data_sha256": sha256(args.data),
            "checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"),
        },
    }
    write_jsonl(args.output, outputs)
    write_metadata(args.summary, summary)
    write_metadata(
        f"{args.output}.metadata.json",
        experiment_metadata(
            stage="v2_ranking_only_oracle_cutoff_evaluation",
            data=args.data,
            exact_dir=args.exact_dir,
            checkpoint=args.checkpoint,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
