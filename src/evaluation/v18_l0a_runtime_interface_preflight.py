#!/usr/bin/env python3
"""Synthetic runtime equivalence preflight for the V18 latent interface."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-8B")
    p.add_argument("--output", type=Path, default=Path("results/v2_rank_then_cut/v18_l0a_latent_contract_preflight/runtime.json"))
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    text = "Question: Which city is the capital of France? Context: Paris is the capital of France. Answer:"
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
    mask = torch.ones_like(ids)
    split = max(2, ids.shape[1] // 2)

    with torch.inference_mode():
        out_ids = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        embeds = model.get_input_embeddings()(ids)
        out_embeds = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits
        input_embed_max_abs = (out_ids.float() - out_embeds.float()).abs().max().item()

        prefix = model(input_ids=ids[:, :split], attention_mask=mask[:, :split], use_cache=True)
        suffix_ids = model(
            input_ids=ids[:, split:],
            attention_mask=mask,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits
        # Recompute the prefix because DynamicCache is mutable.
        prefix_for_embeds = model(input_ids=ids[:, :split], attention_mask=mask[:, :split], use_cache=True)
        suffix_embeds = model.get_input_embeddings()(ids[:, split:])
        suffix = model(
            inputs_embeds=suffix_embeds,
            attention_mask=mask,
            past_key_values=prefix_for_embeds.past_key_values,
            use_cache=True,
        ).logits
        one_shot_vs_cached_ids_max_abs = (out_ids[:, split:].float() - suffix_ids.float()).abs().max().item()
        cached_ids_vs_cached_embeds_max_abs = (suffix_ids.float() - suffix.float()).abs().max().item()
        cached_top1_agreement = float((suffix_ids.argmax(-1) == suffix.argmax(-1)).float().mean().item())

    latent = model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=latent, attention_mask=mask, use_cache=False).logits
    loss = logits[:, -1].float().square().mean()
    loss.backward()
    latent_grad_finite = bool(latent.grad is not None and torch.isfinite(latent.grad).all())
    latent_grad_norm = float(latent.grad.float().norm().item()) if latent.grad is not None else 0.0
    target_grads_none = all(parameter.grad is None for parameter in model.parameters())

    def elapsed(kind: str, repeats: int = 5) -> float:
        samples = []
        with torch.inference_mode():
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if kind == "ids":
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                else:
                    model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask, use_cache=False)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
        return sum(samples) / len(samples)

    ids_latency = elapsed("ids")
    embeds_latency = elapsed("embeds")
    result = {
        "protocol": "V18-L0A_RUNTIME_INTERFACE_PREFLIGHT",
        "model": args.model,
        "synthetic_prompt_tokens": int(ids.shape[1]),
        "dtype": "bfloat16",
        "input_ids_vs_token_inputs_embeds_max_abs_logit_difference": input_embed_max_abs,
        "one_shot_vs_cached_token_ids_max_abs_logit_difference": one_shot_vs_cached_ids_max_abs,
        "cached_token_ids_vs_cached_inputs_embeds_max_abs_logit_difference": cached_ids_vs_cached_embeds_max_abs,
        "cached_token_ids_vs_cached_inputs_embeds_top1_agreement": cached_top1_agreement,
        "latent_gradient_finite": latent_grad_finite,
        "latent_gradient_norm": latent_grad_norm,
        "all_target_parameter_gradients_none": target_grads_none,
        "mean_input_ids_forward_seconds": ids_latency,
        "mean_inputs_embeds_forward_seconds": embeds_latency,
        "inputs_embeds_latency_ratio": embeds_latency / ids_latency,
        "gates": {
            "embedding_equivalence": input_embed_max_abs <= 1e-5,
            "cache_interface_equivalence": cached_ids_vs_cached_embeds_max_abs <= 1e-5 and cached_top1_agreement == 1.0,
            "gradient_path": latent_grad_finite and latent_grad_norm > 0 and target_grads_none,
        },
        "decision": "GO_V18_L0B_TINY_MEMORIZATION_PREFLIGHT" if input_embed_max_abs <= 1e-5 and cached_ids_vs_cached_embeds_max_abs <= 1e-5 and cached_top1_agreement == 1.0 and latent_grad_finite and latent_grad_norm > 0 and target_grads_none else "STOP_V18_RUNTIME_INTERFACE",
        "new_target_generation_calls": 0,
        "synthetic_only": True,
        "sealed_sets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
