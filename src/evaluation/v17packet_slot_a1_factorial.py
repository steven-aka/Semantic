"""Paired four-arm Target replay isolating deferral and addition at depth 9."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
A0 = ROOT / "v17packet_slot_a0_functional_provenance"
OUT = ROOT / "v17packet_slot_a1_factorial"
CFG = Path("configs/v17packet_slot_a1_factorial.json")
ARM_ORDER = ("neither", "baseline", "swap", "both")


def key(row: dict) -> tuple:
    return (hashlib.sha256(row["example_id"].encode()).hexdigest(),
            row["omit_index"], row["candidate_index"])


def pick(rows: list[dict], n: int, excluded: set[str]) -> list[dict]:
    chosen = []
    for row in sorted(rows, key=key):
        if row["example_id"] in excluded:
            continue
        chosen.append(row)
        excluded.add(row["example_id"])
        if len(chosen) == n:
            break
    assert len(chosen) == n
    return chosen


def selected_cases() -> list[dict]:
    rows = list(read_jsonl(A0 / "per_action.jsonl"))
    used = set()
    groups = [
        ("break_late_only", 2, lambda r: r["outcome"] == "break" and r["late_lost"] > 0 and r["early_lost"] == 0),
        ("break_early_only", 2, lambda r: r["outcome"] == "break" and r["early_lost"] > 0 and r["late_lost"] == 0),
        ("break_mixed", 2, lambda r: r["outcome"] == "break" and r["early_lost"] > 0 and r["late_lost"] > 0),
        ("break_no_loss", 1, lambda r: r["outcome"] == "break" and r["lost_correct"] == 0),
        ("selected_repair", 4, lambda r: r["outcome"] == "selected_repair"),
    ]
    chosen = []
    for group, n, predicate in groups:
        chosen += [{**r, "selection_group": group} for r in pick(
            [r for r in rows if predicate(r)], n, used)]
    safety_queries = {r["example_id"] for r in chosen if r["outcome"] == "break"}
    safe = [r for r in rows if r["outcome"] == "safe" and r["example_id"] in safety_queries]
    safe_queries = set()
    chosen += [{**r, "selection_group": "matched_safe"} for r in pick(safe, 2, safe_queries)]
    assert len(chosen) == 13
    return chosen


def build_tasks(cases: list[dict], tokenizer) -> list[dict]:
    ids = {r["example_id"] for r in cases}
    data = {r["example_id"]: r for r in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(ids)
    tasks = []
    for case_id, row in enumerate(cases):
        q = row["example_id"]
        packets, order = data[q]["packet_texts"], orders[q]
        default_id, candidate_id = order[8], order[9]
        default_parts = segments(packets[default_id])
        candidate_parts = segments(packets[candidate_id])
        title, sep, proof = packets[default_id].partition("\nSource evidence: ")
        assert sep
        sentences = [x.split(sep, 1)[1] for x in default_parts]
        assert " ".join(sentences) == proof
        assert row["omit_index"] < len(sentences) and row["candidate_index"] < len(candidate_parts)
        shortened = title + sep + " ".join(s for i, s in enumerate(sentences)
                                            if i != row["omit_index"])
        candidate = candidate_parts[row["candidate_index"]]
        m9 = mask(order, 9)
        contexts = {
            "neither": render(packets, m9, {default_id: shortened}),
            "baseline": render(packets, m9),
            "swap": render(packets, m9 | (1 << candidate_id),
                           {default_id: shortened, candidate_id: candidate}),
            "both": render(packets, m9 | (1 << candidate_id), {candidate_id: candidate}),
        }
        assert len(set(contexts.values())) == 4
        for arm in ARM_ORDER:
            context = contexts[arm]
            tokens = count_tokens(tokenizer, context)
            tasks.append({"case_id": case_id, "example_id": q, "selection_group": row["selection_group"],
                          "arm": arm, "omit_index": row["omit_index"],
                          "candidate_index": row["candidate_index"],
                          "question": data[q]["question"], "context": context,
                          "context_tokens": tokens})
        assert tasks[-2]["context_tokens"] - tasks[-3]["context_tokens"] == row["context_token_delta"]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS"
    cases = selected_cases()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    tasks = build_tasks(cases, tokenizer)
    assert len(cases) == cfg["expected_cases"] and len(tasks) == cfg["expected_new_calls"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "cases": [
        {k: r[k] for k in ("example_id", "selection_group", "omit_index", "candidate_index")}
        for r in cases],
        "arms": list(ARM_ORDER),
        "task_sha256": hashlib.sha256("\n".join(
            f'{r["case_id"]}:{r["arm"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
            for r in tasks).encode()).hexdigest(),
        "planned_new_calls": len(tasks), "sealed_sets_read": False}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps(manifest, indent=2))
        return
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
        if r["example_id"] in {x["example_id"] for x in cases}}
    assert len(atoms) == len({x["example_id"] for x in cases})
    path = OUT / "per_call.jsonl"
    done = {(r["case_id"], r["arm"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [r for r in tasks if (r["case_id"], r["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch],
                                               [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {k: task[k] for k in ("case_id", "example_id", "selection_group",
                                                  "arm", "omit_index", "candidate_index", "context_tokens")}
                    row.update({"f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                                "prediction": " # ".join(answers),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids)})
                    done[row["case_id"], row["arm"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    by_case = defaultdict(dict)
    for row in done.values():
        by_case[row["case_id"]][row["arm"]] = row
    details = []
    for i, case in enumerate(cases):
        arms = by_case[i]
        assert set(arms) == set(ARM_ORDER)
        f = {a: arms[a]["f1"] for a in ARM_ORDER}
        ok = {a: f[a] + cfg["fidelity_epsilon"] >= .90 for a in ARM_ORDER}
        details.append({"case_id": i, "example_id": case["example_id"],
                        "selection_group": case["selection_group"], "cached_outcome": case["outcome"],
                        "f1": f, "success_090": ok,
                        "candidate_gain_without_default": f["swap"] - f["neither"],
                        "candidate_gain_with_default": f["both"] - f["baseline"],
                        "default_gain_without_candidate": f["baseline"] - f["neither"],
                        "default_gain_with_candidate": f["both"] - f["swap"],
                        "interaction": f["both"] - f["baseline"] - f["swap"] + f["neither"],
                        "both_extra_context_tokens": arms["both"]["context_tokens"] - arms["baseline"]["context_tokens"],
                        "baseline_reproduced": ok["baseline"] == (case["baseline_f1"] + 1e-6 >= .90),
                        "swap_reproduced": ok["swap"] == (case["action_f1"] + 1e-6 >= .90)})
    summary = {"protocol": cfg["protocol"], "cases": len(details),
               "new_target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "baseline_reproduced": sum(r["baseline_reproduced"] for r in details),
               "swap_reproduced": sum(r["swap_reproduced"] for r in details),
               "by_group": {}, "sealed_sets_read": False,
               "limitations": ["Selected train-side cases are mechanism-enriched, not a population estimate.",
                               "Both adds context tokens and is a causal diagnostic, not a compression policy.",
                               "Text is rendered in canonical source order; reveal-order persistence is not tested."]}
    for group in sorted({r["selection_group"] for r in details}):
        rows = [r for r in details if r["selection_group"] == group]
        summary["by_group"][group] = {"cases": len(rows),
            "baseline_success": sum(r["success_090"]["baseline"] for r in rows),
            "swap_success": sum(r["success_090"]["swap"] for r in rows),
            "both_success": sum(r["success_090"]["both"] for r in rows),
            "neither_success": sum(r["success_090"]["neither"] for r in rows),
            "baseline_reproduced": sum(r["baseline_reproduced"] for r in rows),
            "swap_reproduced": sum(r["swap_reproduced"] for r in rows)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_case.jsonl").open("w") as handle:
        for row in details:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
