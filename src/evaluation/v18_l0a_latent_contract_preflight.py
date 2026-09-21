#!/usr/bin/env python3
"""Static/API and cost-accounting preflight for V18 progressive latents.

This performs no model inference. It verifies that the installed Qwen3 model
implementation exposes the required white-box hooks and estimates a transparent
parameter-position proxy from the existing C1 depth-10 traces. Runtime
equivalence and latency remain mandatory gates before any training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-config", type=Path, default=Path("models/Qwen3-8B/config.json"))
    p.add_argument("--compressor-config", type=Path, default=Path("models/Qwen3-1.7B/config.json"))
    p.add_argument(
        "--model-source",
        type=Path,
        default=Path(".venv/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py"),
    )
    p.add_argument(
        "--traces",
        type=Path,
        default=Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence/traces.jsonl"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/v2_rank_then_cut/v18_l0a_latent_contract_preflight/summary.json"),
    )
    p.add_argument("--latent-budgets", default="4,8,16,24,32")
    args = p.parse_args()

    target = json.loads(args.target_config.read_text())
    compressor = json.loads(args.compressor_config.read_text())
    source = args.model_source.read_text()
    rows = read_jsonl(args.traces)
    d10 = [r for r in rows if int(r["depth"]) == 10]
    assert len(d10) == 256
    budgets = [int(x) for x in args.latent_budgets.split(",")]
    assert len(budgets) == 5 and budgets == sorted(set(budgets))

    mean_prompt = sum(r["prompt_tokens"] for r in d10) / len(d10)
    mean_context = sum(r["context_tokens"] for r in d10) / len(d10)
    mean_generation = sum(r["generated_tokens"] for r in d10) / len(d10)
    mean_fixed_text = mean_prompt - mean_context
    baseline_five_call_target = 5 * (mean_prompt + mean_generation)
    latent_target = 5 * mean_fixed_text + sum(budgets) + 5 * mean_generation
    target_params = 8.0
    compressor_params = 1.7
    compressor_once_proxy = mean_prompt * compressor_params / target_params
    total_proxy = latent_target + compressor_once_proxy

    static_checks = {
        "target_architecture_is_qwen3_causal_lm": target.get("architectures") == ["Qwen3ForCausalLM"],
        "target_hidden_size_4096": target.get("hidden_size") == 4096,
        "compressor_model_present": args.compressor_config.exists(),
        "forward_exposes_inputs_embeds": "inputs_embeds" in source,
        "forward_exposes_past_key_values": "past_key_values" in source,
        "forward_constructs_dynamic_cache": "DynamicCache" in source,
    }
    result = {
        "protocol": "V18-L0A_PROGRESSIVE_LATENT_CONTRACT_PREFLIGHT",
        "status": "STATIC_PREFLIGHT_COMPLETE_RUNTIME_PREFLIGHT_REQUIRED",
        "contract_change": {
            "retired": ["lossless representation", "source-preserving text-only Target interface", "black-box Target API"],
            "preserved": ["frozen Qwen3-8B Target", "query-conditioned", "progressive", "add-only nested prefixes", "five fidelity levels", "quality-cost Pareto objective"],
            "claim_boundary": "lossy white-box progressive representation; it must not be reported as text-token compression",
        },
        "fixed_candidate": {
            "encoder": "frozen Qwen3-1.7B, run once on canonical query plus full original context",
            "trainable_module": "one small cross-attention resampler plus 2048-to-4096 projection",
            "latent_budgets": budgets,
            "target": "Qwen3-8B fully frozen; gradients may pass through it only to the resampler",
            "forbidden": ["architecture sweep", "latent-budget sweep", "adaptive halting in L0", "gold answers or teacher outputs as compressor inputs"],
        },
        "static_api_checks": static_checks,
        "cost_preflight": {
            "empirical_c1_queries": len(d10),
            "mean_depth10_prompt_tokens": mean_prompt,
            "mean_depth10_context_tokens": mean_context,
            "mean_generated_tokens_per_call": mean_generation,
            "mean_noncontext_text_positions_per_call": mean_fixed_text,
            "depth10_five_call_target_token_compute": baseline_five_call_target,
            "latent_five_call_target_token_compute_if_generation_unchanged": latent_target,
            "compressor_once_8b_equivalent_parameter_position_proxy": compressor_once_proxy,
            "combined_parameter_position_proxy": total_proxy,
            "proxy_ratio_vs_depth10": total_proxy / baseline_five_call_target,
            "warning": "Parameter-position scaling is only a screening proxy. GO requires measured batch-1 latency, throughput, peak VRAM, Target positions, compressor positions and cache memory; latent positions are not text tokens.",
        },
        "runtime_gates_before_training": [
            "token embeddings passed through inputs_embeds reproduce input_ids next-token logits within frozen tolerance",
            "incremental cached latent-prefix forward reproduces one-shot latent-prefix logits within frozen tolerance",
            "gradient reaches latent inputs/resampler while every Target parameter has no gradient and remains unchanged",
            "measured compressor plus Target cost retains positive headroom against V8 fixed depth10",
        ],
        "evaluation_gates_after_training": {
            "quality": "each fidelity and Complete no worse than V8 fixed by more than 1 percentage point on query-held-out data",
            "target_positions": "strictly fewer than V8 at at least 3 of 5 anchors",
            "measured_total_cost": "positive Pareto improvement after compressor latency/FLOPs/VRAM and Target prefill/decode are charged",
            "nonmonotonicity": "SF count falls by at least 50% at matched per-anchor success; shorter-prefix success counts may not fall by more than 1 percentage point",
            "anti_gaming": "report SS/SF/FS/FF counts, per-prefix success, final success and continuous teacher-answer NLL; SF alone is invalid",
            "generalization": "query-grouped train/validation only; sealed sets remain closed",
        },
        "decision": "GO_V18_L0A_RUNTIME_INTERFACE_PREFLIGHT" if all(static_checks.values()) and total_proxy < baseline_five_call_target else "STOP_V18_BEFORE_RUNTIME_PREFLIGHT",
        "new_target_calls": 0,
        "sealed_sets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
