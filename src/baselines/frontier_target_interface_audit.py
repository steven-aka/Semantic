"""Strict read-only identity audit for frozen-Target text/embed/cache backends."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sample_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in list(model.state_dict().items())[:8]:
        h.update(name.encode())
        h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-8B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=Path("results/v2_rank_then_cut/frontier_compression_r0/target_interface.json"))
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={"": args.device}
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = sample_hash(model)
    texts = [
        "Question: Name the cities in France. Context: Paris and Lyon are cities in France. Answer:",
        "Question: Which people won the award? Context: Ada won it in 2020; Lin won it in 2021. Answer:",
    ]
    rows = []
    with torch.inference_mode():
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids.to(args.device)
            mask = torch.ones_like(ids)
            split = ids.shape[1] // 2
            one = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            emb = model.get_input_embeddings()(ids)
            via_emb = model(inputs_embeds=emb, attention_mask=mask, use_cache=False).logits
            prefix = model(input_ids=ids[:, :split], attention_mask=mask[:, :split], use_cache=True)
            cached = model(input_ids=ids[:, split:], attention_mask=mask,
                           past_key_values=prefix.past_key_values, use_cache=True).logits
            expected = one[:, split:]
            rows.append({
                "tokens": int(ids.shape[1]),
                "ids_vs_embeds_max_abs": float((one.float() - via_emb.float()).abs().max()),
                "ids_vs_embeds_top1": float((one.argmax(-1) == via_emb.argmax(-1)).float().mean()),
                "oneshot_vs_cache_max_abs": float((expected.float() - cached.float()).abs().max()),
                "oneshot_vs_cache_mean_abs": float((expected.float() - cached.float()).abs().mean()),
                "oneshot_vs_cache_top1": float((expected.argmax(-1) == cached.argmax(-1)).float().mean()),
                "cache_prefix_length": int(prefix.past_key_values.get_seq_length()),
            })
    after = sample_hash(model)
    gates = {
        "target_parameters_unchanged": before == after,
        "target_parameters_require_no_grad": all(not p.requires_grad for p in model.parameters()),
        "token_embedding_identity": all(r["ids_vs_embeds_max_abs"] == 0 and r["ids_vs_embeds_top1"] == 1 for r in rows),
        "cache_continuation_decision_identity": all(r["oneshot_vs_cache_top1"] == 1 for r in rows),
        "cache_length_correct": all(r["cache_prefix_length"] == r["tokens"] for r in rows),
    }
    result = {
        "protocol": "FRONTIER-COMPRESSION-R0-TARGET-INTERFACE",
        "model": args.model,
        "dtype": "bfloat16",
        "rows": rows,
        "gates": gates,
        "decision": "GO_HARD_SCREEN" if all(gates.values()) else "STOP_TARGET_INTERFACE",
        "interpretation": "BF16 one-shot/cache logits may differ numerically; exact top-1 identity is the binding continuation gate.",
        "new_target_generation_calls": 0,
        "sealed_sets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
