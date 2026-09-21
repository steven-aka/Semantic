"""Paired natural train-side V8 versus rank10 title-only reveal at depth9."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
OUT = ROOT / "v17packet_frag_b0_title_only_natural_pilot"
CFG = Path("configs/v17packet_frag_b0_title_only_natural_pilot.json")


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def select():
    p0 = {r["example_id"] for r in read(ROOT / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")}
    r1 = {r["example_id"] for r in read(ROOT / "v17packet_r1_label_scale512/per_query.jsonl")}
    available = p0 - r1
    data = {r["example_id"]: r for r in read(LINEAGE / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in available}
    order = {r["example_id"]: r["decoded_order"] for r in read(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in available}
    assert len(p0) == 1421 and len(r1) == 512 and len(available) == len(data) == len(order) == 909
    eligible = []
    for q in available:
        packet = data[q]["packet_texts"][order[q][9]]
        title = packet.partition("\nSource evidence: ")[0]
        if title.startswith("Document title: ") and title.removeprefix("Document title: ").strip():
            eligible.append(q)
    selected = sorted(eligible, key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))[:128]
    assert len(selected) == 128
    return selected, data, order


def tasks_for(selected, data, order, tokenizer):
    result = []
    for q in selected:
        ranking, packets = order[q], data[q]["packet_texts"]
        candidate_id = ranking[9]
        title = packets[candidate_id].partition("\nSource evidence: ")[0]
        m9 = mask(ranking, 9)
        contexts = {"v8": render(packets, m9),
                    "title_only": render(packets, m9 | (1 << candidate_id), {candidate_id: title})}
        for arm in ("v8", "title_only"):
            context = contexts[arm]
            result.append({"example_id": q, "arm": arm, "question": data[q]["question"],
                           "context": context,
                           "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS"
    selected, data, order = select()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    tasks = tasks_for(selected, data, order, tokenizer)
    assert len(tasks) == cfg["expected_target_calls"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "query_ids": selected,
                "task_sha256": hashlib.sha256("\n".join(
                    f'{r["example_id"]}:{r["arm"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
                    for r in tasks).encode()).hexdigest(),
                "planned_target_calls": len(tasks), "sealed_sets_read": False}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids"}))
        return
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    atoms = {r["example_id"]: r["answer_atoms"] for r in read(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in set(selected)}
    assert len(atoms) == 128
    call_path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read(call_path)} if call_path.exists() else {}
    pending = [r for r in tasks if (r["example_id"], r["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with call_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"],
                           "context_tokens": task["context_tokens"],
                           "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                           "prediction": " # ".join(parsed),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[task["example_id"], task["arm"]] = row
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    eps = cfg["fidelity_epsilon"]
    paired = []
    for q in selected:
        a, b = done[q, "v8"], done[q, "title_only"]
        base, title = a["f1"] + eps >= .90, b["f1"] + eps >= .90
        paired.append({"example_id": q, "baseline_f1": a["f1"], "title_f1": b["f1"],
                       "delta_f1": b["f1"] - a["f1"], "baseline_success_090": base,
                       "title_success_090": title, "repair": not base and title,
                       "break": base and not title,
                       "context_delta": b["context_tokens"] - a["context_tokens"],
                       "target_token_delta": b["prompt_tokens"] + b["generated_tokens"]
                                             - a["prompt_tokens"] - a["generated_tokens"]})
    rng = random.Random(20260921)
    net_samples = []
    f1_samples = []
    for _ in range(10000):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        net_samples.append(sum(r["repair"] - r["break"] for r in sample))
        f1_samples.append(sum(r["delta_f1"] for r in sample) / len(sample))
    net_samples.sort(); f1_samples.sort()
    summary = {"protocol": cfg["protocol"], "queries": len(selected),
               "target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "baseline_success_090": sum(r["baseline_success_090"] for r in paired),
               "title_success_090": sum(r["title_success_090"] for r in paired),
               "repairs": sum(r["repair"] for r in paired),
               "breaks": sum(r["break"] for r in paired),
               "mean_added_context_tokens": sum(r["context_delta"] for r in paired) / len(paired),
               "mean_target_token_delta": sum(r["target_token_delta"] for r in paired) / len(paired),
               "mean_delta_f1": sum(r["delta_f1"] for r in paired) / len(paired),
               "paired_query_bootstrap_95pct_net_success_count": [net_samples[249], net_samples[9749]],
               "paired_query_bootstrap_95pct_mean_delta_f1": [f1_samples[249], f1_samples[9749]],
               "sealed_sets_read": False,
               "limitations": ["Natural train-side sample, not independent final confirmation.",
                               "Only the 0.90 depth9 endpoint was paired fresh; unchanged other anchors were not re-called and no Complete claim is made.",
                               "All rank10 titles come from a constructed gold-answer packet pool; answer-cue shortcut risk applies."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in paired:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
