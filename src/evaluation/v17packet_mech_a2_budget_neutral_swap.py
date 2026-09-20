"""Train-side oracle pilot for sentence-granularity, budget-neutral replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


CFG = Path("configs/v17packet_mech_a2_budget_neutral_swap.json")
ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
A1 = ROOT / "v17packet_mech_a1_factorial_replay"
OUT = ROOT / "v17packet_mech_a2_budget_neutral_swap"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_MECHANISM"
    prior = list(read_jsonl(A1 / "per_case.jsonl"))
    ids = {r["example_id"] for r in prior}
    source = {r["example_id"]: r for r in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(prior) == len(source) == len(orders) == len(atoms) == 63
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    tasks = []
    for case in prior:
        q = case["example_id"]
        packets = source[q]["packet_texts"]
        order = orders[q]
        stay_id, candidate_id = order[8], order[9]
        stay_fragments = segments(packets[stay_id])
        candidate_fragments = segments(packets[candidate_id])
        candidate = candidate_fragments[case["sentence_index"]]
        if len(stay_fragments) < 2:
            continue
        title, separator, original_proof = packets[stay_id].partition("\nSource evidence: ")
        assert separator
        proof_sentences = [x.split(separator, 1)[1] for x in stay_fragments]
        assert " ".join(proof_sentences) == original_proof
        base_context = render(packets, mask(order, 9))
        base_tokens = count_tokens(tokenizer, base_context)
        assert base_tokens == case["cached_B_tokens"]
        for omit_index in range(len(proof_sentences)):
            short_stay = title + separator + " ".join(
                sentence for i, sentence in enumerate(proof_sentences) if i != omit_index)
            assert short_stay != packets[stay_id]
            context = render(packets, mask(order, 9) | (1 << candidate_id),
                             {stay_id: short_stay, candidate_id: candidate})
            tokens = count_tokens(tokenizer, context)
            if tokens > base_tokens:
                continue
            tasks.append({"example_id": q, "group": case["group"],
                          "omit_index": omit_index, "candidate_index": case["sentence_index"],
                          "context": context, "context_tokens": tokens,
                          "baseline_tokens": base_tokens, "question": source[q]["question"]})
    groups = Counter(r["group"] for r in tasks)
    assert len(tasks) == cfg["expected_new_calls"] == 80
    assert len({r["example_id"] for r in tasks}) == cfg["expected_cases_with_action"] == 41
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "cases_with_action": 41,
                "new_target_calls": len(tasks), "tasks_by_prior_group": dict(groups),
                "task_signature_sha256": hashlib.sha256("\n".join(
                    f'{r["example_id"]}:{r["omit_index"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
                    for r in tasks).encode()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    if args.preflight_only:
        return
    path = OUT / "per_action.jsonl"
    done = {(r["example_id"], r["omit_index"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [r for r in tasks if (r["example_id"], r["omit_index"]) not in done]
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
                    row = {k: task[k] for k in ("example_id", "group", "omit_index",
                                                  "candidate_index", "context_tokens", "baseline_tokens")}
                    row.update({"f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids),
                                "prediction": " # ".join(answers)})
                    done[row["example_id"], row["omit_index"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed_calls": len(done), "total_calls": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    by_case = {}
    for case in prior:
        q = case["example_id"]
        actions = [r for (qid,_),r in done.items() if qid == q]
        if not actions:
            continue
        successful = [r for r in actions if r["f1"] + cfg["fidelity_epsilon"] >= 0.90]
        best = min(successful, key=lambda r: (r["context_tokens"], r["omit_index"])) if successful else None
        by_case[q] = {"example_id": q, "group": case["group"],
                      "prior_repeat_pair_valid": case.get("repeat_pair_valid"),
                      "prior_A_success": case["A_success"],
                      "baseline_tokens": case["cached_B_tokens"],
                      "candidate_actions": len(actions), "successful_actions": len(successful),
                      "oracle_success": bool(best),
                      "best_success_tokens": best["context_tokens"] if best else None,
                      "best_success_omission_index": best["omit_index"] if best else None}
    stable_repair = [r for r in by_case.values() if r["group"] == "repair" and r["prior_repeat_pair_valid"]]
    summary = {"protocol": cfg["protocol"], "new_target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "cases_with_action": len(by_case),
               "successful_cases_total": sum(r["oracle_success"] for r in by_case.values()),
               "stable_repair_cases_with_action": len(stable_repair),
               "stable_repair_cases_with_successful_budget_neutral_swap": sum(r["oracle_success"] for r in stable_repair),
               "controls_with_action": sum(r["group"] != "repair" for r in by_case.values()),
               "controls_with_successful_budget_neutral_swap": sum(r["group"] != "repair" and r["oracle_success"] for r in by_case.values()),
               "limitations": "Outcome-selected mechanism sample and oracle omission choice. No deployable selector; evaluate safety on untouched V8-success queries before any candidate policy.",
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_case.jsonl").open("w") as handle:
        for row in by_case.values():
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
