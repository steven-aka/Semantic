"""Paired ordinary/logprob Target runs on 32 stratified train-only prefixes."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT = Path("results/v2_rank_then_cut/v17stop_c2a_native_trace_preflight")
CONTINUE = "depth10_plus_old_context_supported_depth9"
CLASSES = ("SS", "SF", "FS", "FF")


def stable_key(q: str) -> str:
    return hashlib.sha256(("STOP-C2A|" + q).encode()).hexdigest()


def main() -> None:
    from vllm import SamplingParams

    cfg = json.loads(Path("configs/v17stop_c2a_native_trace_preflight.json").read_text())
    assert cfg["status"] == "FROZEN_SMALL_PAIRED_TRAIN_ONLY"
    c0 = list(read_jsonl(C0 / "per_query.jsonl"))
    by_class = defaultdict(list)
    for r in c0:
        a = r["outputs"]["depth9_only"]["success"]["0.9"]
        b = r["outputs"][CONTINUE]["success"]["0.9"]
        by_class[("S" if a else "F") + ("S" if b else "F")].append(r)
    chosen = [r for name in CLASSES for r in sorted(by_class[name], key=lambda r: stable_key(r["example_id"]))[:8]]
    assert len(chosen) == 32 and len({r["example_id"] for r in chosen}) == 32
    ids = {r["example_id"] for r in chosen}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    prefix = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in ids and r["depth"] == 9:
                prefix[r["example_id"]] = r
    assert all(len(x) == 32 for x in (data, atoms, prefix))
    target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                 max_new_tokens=cfg["max_new_tokens"],
                                 gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                 max_model_len=cfg["max_model_len"])
    prompts = []
    for r in chosen:
        q = r["example_id"]
        context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"])
                                if prefix[q]["mask"] & (1 << i))
        prompts.extend(target._chat_prompts([data[q]["question"]], [context]))
    base = SamplingParams(temperature=0, max_tokens=cfg["max_new_tokens"])
    traced = SamplingParams(temperature=0, max_tokens=cfg["max_new_tokens"],
                            logprobs=cfg["logprobs_top_k"])
    start = time.perf_counter()
    plain_outputs = target.model.generate(prompts, base, use_tqdm=False)
    plain_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scored_outputs = target.model.generate(prompts, traced, use_tqdm=False)
    scored_seconds = time.perf_counter() - start
    eos = target.tokenizer.eos_token_id
    rows = []
    for r, plain, scored in zip(chosen, plain_outputs, scored_outputs):
        q = r["example_id"]
        one, two = plain.outputs[0], scored.outputs[0]
        a1, a2 = parse_list_prediction(one.text), parse_list_prediction(two.text)
        f1 = qampari_list_metrics(a1, atoms[q])["f1"]
        f2 = qampari_list_metrics(a2, atoms[q])["f1"]
        logs = two.logprobs or []
        selected_available = len(logs) == len(two.token_ids) and all(
            token in top for token, top in zip(two.token_ids, logs))
        eos_in_generated = eos in two.token_ids
        eos_in_last_top2 = bool(logs and eos in logs[-1])
        rows.append({"example_id": q,
                     "transition_class": next(name for name in CLASSES if r in by_class[name]),
                     "cache_parsed": prefix[q]["prediction"],
                     "plain_parsed": " # ".join(a1), "logprob_parsed": " # ".join(a2),
                     "plain_f1": f1, "logprob_f1": f2,
                     "cache_090_success": r["outputs"]["depth9_only"]["success"]["0.9"],
                     "plain_090_success": f1 + 1e-6 >= .9,
                     "logprob_090_success": f2 + 1e-6 >= .9,
                     "raw_token_ids_match": list(one.token_ids) == list(two.token_ids),
                     "plain_stop_reason": one.finish_reason,
                     "logprob_stop_reason": two.finish_reason,
                     "logprob_count_matches_generated": len(logs) == len(two.token_ids),
                     "chosen_token_logprob_available": selected_available,
                     "eos_token_in_generated": eos_in_generated,
                     "eos_token_in_last_top2": eos_in_last_top2,
                     "generated_tokens_plain": len(one.token_ids),
                     "generated_tokens_logprob": len(two.token_ids)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {"protocol": cfg["protocol"], "queries": len(rows),
               "classes": {name: sum(r["transition_class"] == name for r in rows) for name in CLASSES},
               "paired_raw_token_ids_match": sum(r["raw_token_ids_match"] for r in rows),
               "paired_parsed_match": sum(r["plain_parsed"] == r["logprob_parsed"] for r in rows),
               "plain_vs_cache_parsed_match": sum(r["plain_parsed"] == r["cache_parsed"] for r in rows),
               "logprob_vs_cache_parsed_match": sum(r["logprob_parsed"] == r["cache_parsed"] for r in rows),
               "paired_090_label_match": sum(r["plain_090_success"] == r["logprob_090_success"] for r in rows),
               "plain_vs_cache_090_label_match": sum(r["plain_090_success"] == r["cache_090_success"] for r in rows),
               "chosen_token_scores_available": sum(r["chosen_token_logprob_available"] for r in rows),
               "eos_in_generated": sum(r["eos_token_in_generated"] for r in rows),
               "eos_in_last_top2": sum(r["eos_token_in_last_top2"] for r in rows),
               "plain_batch_seconds": plain_seconds,
               "logprob_batch_seconds": scored_seconds,
               "prompt_tokens_per_pass": sum(len(target.tokenizer.encode(p)) for p in prompts),
               "generated_tokens_plain": sum(r["generated_tokens_plain"] for r in rows),
               "generated_tokens_logprob": sum(r["generated_tokens_logprob"] for r in rows),
               "extra_target_calls_for_pilot": 64,
               "sealed_outcome_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
