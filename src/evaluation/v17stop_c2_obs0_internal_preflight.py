"""Paired plain versus last-layer-hook generation for internal-state C2."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, build_qampari_prompt


CFG = Path("configs/v17stop_c2_obs0_internal_preflight.json")
C1 = Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence")
DATA = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
ANN = Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
OUT = Path("results/v2_rank_then_cut/v17stop_c2_obs0_internal_preflight")


def main() -> None:
    cfg = json.loads(CFG.read_text())
    manifest = json.loads((C1 / "collection_manifest.json").read_text())
    traces = {(r["example_id"], int(r["depth"])): r for r in read_jsonl(C1 / "traces.jsonl")}
    selected = sorted({q for q, _ in traces})[: cfg["sample_size"]]
    ids = set(selected)
    data = {r["example_id"]: r for r in read_jsonl(DATA) if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(ANN) if r["example_id"] in ids}
    assert len(data) == len(atoms) == cfg["sample_size"] and manifest["queries"] == 256

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"], torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, device_map=cfg["device"]
    ).eval()

    tasks = []
    for qid in selected:
        for depth in cfg["depths"]:
            state = traces[(qid, depth)]
            context = "\n\n".join(text.strip() for i, text in enumerate(data[qid]["packet_texts"])
                                    if state["mask"] & (1 << i))
            messages = [{"role": "system", "content": QAMPARI_SYSTEM_PROMPT},
                        {"role": "user", "content": build_qampari_prompt(data[qid]["question"], context)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tasks.append({"example_id": qid, "depth": depth, "prompt": prompt})

    rows = []
    plain_seconds = hook_seconds = 0.0
    plain_peak = hook_peak = 0
    for start in range(0, len(tasks), cfg["batch_size"]):
        batch = tasks[start:start + cfg["batch_size"]]
        encoded = tokenizer([x["prompt"] for x in batch], return_tensors="pt", padding=True).to(cfg["device"])
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with torch.inference_mode():
            plain = model.generate(**encoded, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                                   pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(); plain_seconds += time.perf_counter() - begin
        plain_peak = max(plain_peak, torch.cuda.max_memory_allocated())

        captures = []
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures.append(hidden[:, -1, :].detach().float().cpu())
        handle = model.model.layers[-1].register_forward_hook(hook)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with torch.inference_mode():
            observed = model.generate(**encoded, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                                      pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(); hook_seconds += time.perf_counter() - begin
        hook_peak = max(hook_peak, torch.cuda.max_memory_allocated())
        handle.remove()

        input_len = encoded["input_ids"].shape[1]
        for i, task in enumerate(batch):
            p_ids = plain[i, input_len:].tolist()
            h_ids = observed[i, input_len:].tolist()
            # Trim padding after the first EOS, retaining EOS itself.
            def trim(xs):
                return xs[: xs.index(tokenizer.eos_token_id) + 1] if tokenizer.eos_token_id in xs else xs
            p_ids, h_ids = trim(p_ids), trim(h_ids)
            p_text = tokenizer.decode(p_ids, skip_special_tokens=True)
            h_text = tokenizer.decode(h_ids, skip_special_tokens=True)
            p_ans, h_ans = parse_list_prediction(p_text), parse_list_prediction(h_text)
            p_f1 = float(qampari_list_metrics(p_ans, atoms[task["example_id"]])["f1"])
            h_f1 = float(qampari_list_metrics(h_ans, atoms[task["example_id"]])["f1"])
            usable = captures[:max(1, len(h_ids))]
            vecs = torch.stack([x[i] for x in usable])
            h1, h2, h3 = vecs[0], vecs.mean(0), vecs[-1]
            rows.append({"example_id": task["example_id"], "depth": task["depth"],
                         "plain_token_ids": p_ids, "hook_token_ids": h_ids,
                         "plain_prediction": " # ".join(p_ans), "hook_prediction": " # ".join(h_ans),
                         "plain_f1": p_f1, "hook_f1": h_f1,
                         "token_ids_match": p_ids == h_ids,
                         "parsed_match": p_ans == h_ans, "f1_match": abs(p_f1 - h_f1) < 1e-12,
                         "hidden_finite": bool(torch.isfinite(torch.stack([h1, h2, h3])).all()),
                         "h1_norm": float(h1.norm()), "h2_norm": float(h2.norm()), "h3_norm": float(h3.norm())})

    n = len(rows)
    summary = {"protocol": cfg["protocol"], "queries": len(selected), "states": n,
               "token_id_matches": sum(r["token_ids_match"] for r in rows),
               "parsed_matches": sum(r["parsed_match"] for r in rows),
               "f1_matches": sum(r["f1_match"] for r in rows),
               "finite_hidden_states": sum(r["hidden_finite"] for r in rows),
               "plain_seconds": plain_seconds, "hook_seconds": hook_seconds,
               "hook_latency_ratio": hook_seconds / plain_seconds,
               "plain_peak_vram_bytes": plain_peak, "hook_peak_vram_bytes": hook_peak,
               "peak_vram_delta_bytes": hook_peak - plain_peak,
               "same_execution_path_as_c1_vllm": False,
               "requires_c2_path_baseline_regeneration": True,
               "decision": "GO_C2_DESIGN_COLLECTION" if all(sum(r[k] for r in rows) == n for k in ("token_ids_match", "parsed_match", "f1_match", "hidden_finite")) else "STOP_C2_INTERNAL_PREFLIGHT",
               "new_target_generations": 2 * n, "sealed_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "paired.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
