from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.model.relational_packet_ranker import RelationalPacketRanker
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.search.weighted_precedence import exact_weighted_precedence_order
from src.training.train_atomic_ranker import RankOracleDataset, make_rank_collator


def _load(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ), device_map={"": 0},
    )
    lm = PeftModel.from_pretrained(base, Path(args.checkpoint) / "adapter")
    lm.config.use_cache = False
    model = RelationalPacketRanker(lm, args.head_hidden_size)
    state = torch.load(Path(args.checkpoint) / "precedence_head.pt", map_location="cpu", weights_only=True)
    model.precedence_head.load_state_dict(state)
    model.precedence_head.to(device="cuda:0", dtype=torch.bfloat16)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V4 relational order with global oracle cutoff")
    parser.add_argument("--data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--head-hidden-size", type=int, default=256)
    args = parser.parse_args()
    source_rows = list(read_jsonl(args.data))
    tokenizer, model = _load(args)
    loader = DataLoader(
        RankOracleDataset(source_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=make_rank_collator(int(tokenizer.pad_token_id)),
    )
    model.eval()
    device = torch.device("cuda:0")
    logits_by_id = {}
    edge_correct = edges = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["packet_spans"]
            )["precedence_logits"].float().cpu()
            for index, example_id in enumerate(batch["example_ids"]):
                values = logits[index].tolist()
                logits_by_id[example_id] = values
                for winner, loser in batch["preferences"][index]:
                    edge_correct += int(values[winner][loser] > 0)
                    edges += 1
    outputs = []
    anchor_success = anchors = trajectory_success = 0
    regrets = []
    per_level: dict[float, list[tuple[bool, float | None]]] = {}
    for source in source_rows:
        example_id = source["example_id"]
        logits = logits_by_id[example_id]
        order = exact_weighted_precedence_order(logits)
        levels = [float(value) for value in source["active_levels"]]
        exact = list(read_jsonl(Path(args.exact_dir) / f"{example_id}.jsonl", ExactSearchResult))
        by_state = {row.state: row for row in exact}
        full_tokens = by_state[(1,) * len(order)].tokens
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
            regret = (tokens - int(baseline["tokens"])) / full_tokens if success and tokens is not None else None
            per_level.setdefault(level, []).append((success, regret))
            anchor_rows.append({
                "fidelity_level": level, "contract_success": success,
                "cutoff_count": predicted["cutoff_count"], "tokens": tokens,
                "oracle_nested_tokens": baseline["tokens"],
                "rate_regret_normalized_if_success": regret, "state": predicted["state"],
            })
        trajectory_success += int(all_success)
        trajectory_regret = (learned_tokens - oracle_tokens) / (len(levels) * full_tokens) if all_success else None
        if trajectory_regret is not None:
            regrets.append(trajectory_regret)
        outputs.append({
            "example_id": example_id, "precedence_logits": logits,
            "decoded_order": list(order), "all_active_contracts_success": all_success,
            "oracle_cutoff_ranking_regret_normalized": trajectory_regret, "anchors": anchor_rows,
        })
    summary = {
        "complete": True, "examples": len(outputs), "stable_boundary_edges": edges,
        "stable_boundary_edge_accuracy": edge_correct / edges,
        "oracle_cutoff_active_contract_success_fraction": anchor_success / anchors,
        "oracle_cutoff_all_active_contracts_success_fraction": trajectory_success / len(outputs),
        "oracle_cutoff_feasible_trajectory_examples": len(regrets),
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": mean(regrets) if regrets else None,
        "per_level": {
            str(level): {
                "examples": len(values),
                "contract_successes": sum(success for success, _ in values),
                "contract_success_fraction": mean(success for success, _ in values),
                "mean_rate_regret_normalized_successful": mean(regret for _, regret in values if regret is not None)
                if any(regret is not None for _, regret in values) else None,
            } for level, values in sorted(per_level.items())
        },
        "decoder": "exact maximum-weight total order over antisymmetric pair logits",
        "artifacts": {
            "data_sha256": sha256(args.data),
            "checkpoint_metadata_sha256": sha256(Path(args.checkpoint) / "training_metadata.json"),
        },
    }
    write_jsonl(args.output, outputs)
    write_metadata(args.summary, summary)
    write_metadata(f"{args.output}.metadata.json", experiment_metadata(
        stage="v4_relational_precedence_oracle_cutoff_evaluation",
        data=args.data, exact_dir=args.exact_dir, checkpoint=args.checkpoint,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
