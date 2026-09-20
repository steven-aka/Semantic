"""Measure break risk of all budget-neutral swaps on V8-success train queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


CFG = Path("configs/v17packet_mech_a3_success_safety.json")
ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
R1 = ROOT / "v17packet_r1_label_scale512"
OUT = ROOT / "v17packet_mech_a3_success_safety"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_SAFETY"
    success = [r for r in read_jsonl(R1 / "per_query.jsonl") if r["baseline"]["hits"][3] is True]
    assert len(success) == 381
    selected = sorted(success, key=lambda r: hashlib.sha256(r["example_id"].encode()).hexdigest())[:64]
    ids = {r["example_id"] for r in selected}
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(selected) == len(data) == len(orders) == len(atoms) == cfg["sample_queries"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    tasks = []
    for row in selected:
        q = row["example_id"]
        packets = data[q]["packet_texts"]
        order = orders[q]
        stay_id, candidate_id = order[8], order[9]
        stay_parts, candidate_parts = segments(packets[stay_id]), segments(packets[candidate_id])
        if len(stay_parts) < 2 or not candidate_parts:
            continue
        title, separator, proof = packets[stay_id].partition("\nSource evidence: ")
        assert separator
        sentences = [x.split(separator, 1)[1] for x in stay_parts]
        assert " ".join(sentences) == proof
        base_context = render(packets, mask(order, 9))
        base_tokens = count_tokens(tokenizer, base_context)
        assert base_tokens > 0  # row baseline tokens sum all attainable anchors
        for omit_index in range(len(sentences)):
            short = title + separator + " ".join(
                sentence for i, sentence in enumerate(sentences) if i != omit_index)
            for candidate_index, fragment in enumerate(candidate_parts):
                context = render(packets, mask(order, 9) | (1 << candidate_id),
                                 {stay_id: short, candidate_id: fragment})
                tokens = count_tokens(tokenizer, context)
                if tokens > base_tokens:
                    continue
                tasks.append({"example_id": q, "omit_index": omit_index,
                              "candidate_index": candidate_index, "context": context,
                              "context_tokens": tokens, "baseline_tokens": base_tokens,
                              "question": data[q]["question"]})
    assert len(tasks) == cfg["expected_new_calls"] == 153
    assert len({r["example_id"] for r in tasks}) == cfg["expected_eligible_queries"] == 32
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "sample_queries": len(selected),
                "eligible_queries": 32, "new_target_calls": len(tasks),
                "ids_sha256": hashlib.sha256("\n".join(r["example_id"] for r in selected).encode()).hexdigest(),
                "task_signature_sha256": hashlib.sha256("\n".join(
                    f'{r["example_id"]}:{r["omit_index"]}:{r["candidate_index"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
                    for r in tasks).encode()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    if args.preflight_only:
        return
    path = OUT / "per_action.jsonl"
    done = {(r["example_id"], r["omit_index"], r["candidate_index"]): r
            for r in read_jsonl(path)} if path.exists() else {}
    pending = [r for r in tasks if (r["example_id"], r["omit_index"], r["candidate_index"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start+cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch],
                                               [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {k: task[k] for k in ("example_id", "omit_index", "candidate_index",
                                                  "context_tokens", "baseline_tokens")}
                    row.update({"f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids),
                                "prediction": " # ".join(answers)})
                    done[row["example_id"], row["omit_index"], row["candidate_index"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed_calls": len(done), "total_calls": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    per_query = []
    for row in selected:
        q = row["example_id"]
        options = [r for key,r in done.items() if key[0] == q]
        frozen = min(options, key=lambda r: (r["context_tokens"], r["candidate_index"], r["omit_index"])) if options else None
        safe = [r for r in options if r["f1"]+cfg["fidelity_epsilon"] >= 0.90]
        per_query.append({"example_id": q, "feasible_actions": len(options),
                          "safe_actions": len(safe),
                          "any_safe_action": bool(safe),
                          "all_actions_break": bool(options) and not safe,
                          "frozen_policy_break": frozen is not None and frozen["f1"]+cfg["fidelity_epsilon"] < 0.90,
                          "frozen_policy_token_delta": frozen["context_tokens"]-frozen["baseline_tokens"] if frozen else 0,
                          "frozen_policy_action": [frozen["candidate_index"], frozen["omit_index"]] if frozen else None})
    eligible = [r for r in per_query if r["feasible_actions"]]
    summary = {"protocol": cfg["protocol"], "sample_queries": 64,
               "eligible_queries": len(eligible), "new_target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "breaking_actions": sum(r["f1"]+cfg["fidelity_epsilon"] < 0.90 for r in done.values()),
               "queries_with_any_safe_action": sum(r["any_safe_action"] for r in eligible),
               "queries_where_all_actions_break": sum(r["all_actions_break"] for r in eligible),
               "frozen_min_token_policy_breaks": sum(r["frozen_policy_break"] for r in eligible),
               "frozen_min_token_policy_mean_token_delta_all_64": sum(r["frozen_policy_token_delta"] for r in per_query)/64,
               "limitations": "Training-side R1-exposed sample; safety of a deterministic action rule, not a trained selector or fresh confirmation.",
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for r in per_query:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
