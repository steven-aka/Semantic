"""Outcome-blind, paired Qwen3-8B labels for protected depth9 insertion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments

BASE = Path("results/v2_rank_then_cut")
LINEAGE = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17packet_obs_a1_natural_insertion"
CFG = Path("configs/v17packet_obs_a1_natural_insertion.json")


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def prepare(cfg, tokenizer):
    p0 = {r["example_id"] for r in read(BASE / "v17canon_p0_fresh_v8_prefix_chain/per_query.jsonl")}
    r1 = {r["example_id"] for r in read(BASE / "v17packet_r1_label_scale512/per_query.jsonl")}
    b0 = set(json.loads((BASE / "v17packet_frag_b0_title_only_natural_pilot/manifest.json").read_text())["query_ids"])
    frame = p0 - r1 - b0
    assert len(p0) == 1421 and len(r1) == 512 and len(b0) == 128 and len(frame) == 781
    data = {r["example_id"]: r for r in read(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in frame}
    order = {r["example_id"]: r["decoded_order"] for r in read(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in frame}
    assert len(data) == len(order) == len(frame)
    eligible = []
    for q in frame:
        packets, ranking = data[q]["packet_texts"], order[q]
        candidate_id = ranking[9]
        base = render(packets, mask(ranking, 9))
        base_tokens = len(tokenizer.encode(base, add_special_tokens=False))
        options = []
        for i, sentence in enumerate(segments(packets[candidate_id])):
            context = render(packets, mask(ranking, 9) | (1 << candidate_id), {candidate_id: sentence})
            tokens = len(tokenizer.encode(context, add_special_tokens=False))
            slack = tokens - base_tokens
            if 0 < slack <= cfg["slack_max"]:
                options.append((slack, i, context, tokens))
        if options:
            slack, i, context, tokens = min(options)
            eligible.append({"example_id": q, "question": data[q]["question"],
                             "baseline_context": base, "action_context": context,
                             "baseline_tokens": base_tokens, "action_tokens": tokens,
                             "slack": slack, "candidate_index": i})
    selected = sorted(eligible, key=lambda r: (hashlib.sha256(r["example_id"].encode()).hexdigest(), r["example_id"]))[:cfg["queries"]]
    assert len(selected) == cfg["queries"]
    return selected, len(eligible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    selected, eligible_count = prepare(cfg, tokenizer)
    tasks = []
    for row in selected:
        for arm in ("baseline", "action"):
            tasks.append({"example_id": row["example_id"], "arm": arm,
                          "question": row["question"], "context": row[f"{arm}_context"],
                          "context_tokens": row[f"{arm}_tokens"]})
    manifest = {"protocol": cfg["protocol"], "query_ids": [r["example_id"] for r in selected],
                "eligible_query_count": eligible_count, "planned_target_calls": len(tasks),
                "task_sha256": hashlib.sha256("\n".join(
                    f'{t["example_id"]}:{t["arm"]}:{hashlib.sha256(t["context"].encode()).hexdigest()}' for t in tasks
                ).encode()).hexdigest(), "sealed_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids"}, indent=2))
        return
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    selected_ids = set(manifest["query_ids"])
    atoms = {r["example_id"]: r["answer_atoms"] for r in read("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in selected_ids}
    assert len(atoms) == len(selected)
    output_path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read(output_path)} if output_path.exists() else {}
    pending = [t for t in tasks if (t["example_id"], t["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with output_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([t["question"] for t in batch], [t["context"] for t in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"], "context_tokens": task["context_tokens"],
                           "f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                           "raw_output": output.outputs[0].text, "prediction": answers,
                           "generated_token_ids": list(output.outputs[0].token_ids),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    paired = []
    for source in selected:
        q = source["example_id"]
        baseline, action = done[q, "baseline"], done[q, "action"]
        before = baseline["f1"] + cfg["fidelity_epsilon"] >= .90
        after = action["f1"] + cfg["fidelity_epsilon"] >= .90
        paired.append({"example_id": q, "candidate_index": source["candidate_index"],
                       "baseline_f1": baseline["f1"], "action_f1": action["f1"],
                       "delta_f1": action["f1"] - baseline["f1"], "repair": not before and after,
                       "break": before and not after, "slack": source["slack"],
                       "target_token_delta": action["prompt_tokens"] + action["generated_tokens"]
                                             - baseline["prompt_tokens"] - baseline["generated_tokens"]})
    summary = {"protocol": cfg["protocol"], "queries": len(paired), "eligible_queries": eligible_count,
               "target_calls": len(done), "baseline_success_090": sum(r["baseline_f1"] + 1e-6 >= .90 for r in paired),
               "action_success_090": sum(r["action_f1"] + 1e-6 >= .90 for r in paired),
               "repairs": sum(r["repair"] for r in paired), "breaks": sum(r["break"] for r in paired),
               "mean_delta_f1": sum(r["delta_f1"] for r in paired) / len(paired),
               "mean_extra_context_tokens": sum(r["slack"] for r in paired) / len(paired),
               "mean_target_token_delta": sum(r["target_token_delta"] for r in paired) / len(paired),
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in paired:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
