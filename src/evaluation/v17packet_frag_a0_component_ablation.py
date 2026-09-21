"""Fresh paired 2x2 ablation of title and sentence at one V8 reveal slot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
B0 = ROOT / "v17packet_slot_b0_pilot"
OUT = ROOT / "v17packet_frag_a0_component_ablation"
CFG = Path("configs/v17packet_frag_a0_component_ablation.json")
ARMS = ("baseline", "title_only", "sentence_only", "title_sentence")


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def key(row):
    return (hashlib.sha256(row["example_id"].encode()).hexdigest(), row["candidate_index"])


def cases():
    rows = list(read(ROOT / "v17packet_insert_c0_visible_signal/per_action.jsonl"))
    decisive = [r for r in rows if r["outcome"] in ("repair", "break")]
    assert len(decisive) == 21
    used = {r["example_id"] for r in decisive}
    controls = []
    for outcome in ("unchanged", "same_threshold_f1_gain", "same_threshold_f1_loss"):
        selected = []
        for row in sorted((r for r in rows if r["outcome"] == outcome), key=key):
            if row["example_id"] not in used:
                selected.append(row)
                used.add(row["example_id"])
                if len(selected) == 2:
                    break
        assert len(selected) == 2, outcome
        controls += selected
    assert len(controls) == 6
    return sorted(decisive + controls, key=key)


def tasks_for(cases, tokenizer):
    ids = {r["example_id"] for r in cases}
    data = {r["example_id"]: r for r in read(LINEAGE / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    order = {r["example_id"]: r["decoded_order"] for r in read(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(data) == len(order) == len(ids)
    result = []
    for case_id, case in enumerate(cases):
        q, index = case["example_id"], case["candidate_index"]
        packets, ranking = data[q]["packet_texts"], order[q]
        candidate_id = ranking[9]
        fragment = segments(packets[candidate_id])[index]
        title, separator, sentence = fragment.partition("\nSource evidence: ")
        assert separator and title.startswith("Document title: ") and sentence
        selected_mask = mask(ranking, 9)
        variants = {
            "baseline": render(packets, selected_mask),
            "title_only": render(packets, selected_mask | (1 << candidate_id),
                                 {candidate_id: title}),
            "sentence_only": render(packets, selected_mask | (1 << candidate_id),
                                    {candidate_id: "Source evidence: " + sentence}),
            "title_sentence": render(packets, selected_mask | (1 << candidate_id),
                                     {candidate_id: fragment}),
        }
        assert len(set(variants.values())) == 4
        for arm in ARMS:
            context = variants[arm]
            result.append({"case_id": case_id, "example_id": q,
                           "candidate_index": index, "cached_outcome": case["outcome"],
                           "arm": arm, "question": data[q]["question"],
                           "context": context,
                           "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
        assert result[-1]["context_tokens"] - result[-4]["context_tokens"] == case["slack"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS" and cfg["arms"] == list(ARMS)
    selected = cases()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    tasks = tasks_for(selected, tokenizer)
    assert len(selected) == 27 and len(tasks) == cfg["expected_target_calls"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "case_count": len(selected),
                "cached_outcomes": dict(Counter(r["outcome"] for r in selected)),
                "arms": list(ARMS),
                "task_sha256": hashlib.sha256("\n".join(
                    f'{r["case_id"]}:{r["arm"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
                    for r in tasks).encode()).hexdigest(),
                "cases": [{"example_id": r["example_id"], "candidate_index": r["candidate_index"],
                           "cached_outcome": r["outcome"]} for r in selected],
                "sealed_sets_read": False, "expected_new_calls": len(tasks)}
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "cases"}))
        return
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    atoms = {r["example_id"]: r["answer_atoms"] for r in read(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl")
        if r["example_id"] in {t["example_id"] for t in tasks}}
    assert len(atoms) == len({t["example_id"] for t in tasks})
    call_path = OUT / "per_call.jsonl"
    done = {(r["case_id"], r["arm"]): r for r in read(call_path)} if call_path.exists() else {}
    pending = [r for r in tasks if (r["case_id"], r["arm"]) not in done]
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
                    answer = parse_list_prediction(output.outputs[0].text)
                    row = {k: task[k] for k in ("case_id", "example_id", "candidate_index", "cached_outcome", "arm", "context_tokens")}
                    row.update({"f1": float(qampari_list_metrics(answer, atoms[task["example_id"]])["f1"]),
                                "prediction": " # ".join(answer),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids)})
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[task["case_id"], task["arm"]] = row
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    eps = cfg["fidelity_epsilon"]
    grouped = defaultdict(dict)
    for row in done.values():
        grouped[row["case_id"]][row["arm"]] = row
    detail = []
    counts = defaultdict(Counter)
    patterns = defaultdict(Counter)
    for i, case in enumerate(selected):
        arms = grouped[i]
        assert set(arms) == set(ARMS)
        f1 = {arm: arms[arm]["f1"] for arm in ARMS}
        success = {arm: f1[arm] + eps >= .90 for arm in ARMS}
        original = {"baseline": case["baseline_success_090"],
                    "title_sentence": case["action_success_090"]}
        reproduced = {arm: success[arm] == original[arm] for arm in original}
        category = case["outcome"]
        counts[category]["cases"] += 1
        patterns[category]["".join("1" if success[a] else "0"
                                    for a in ("title_only", "sentence_only", "title_sentence"))] += 1
        for arm in ARMS:
            counts[category][f"{arm}_success"] += success[arm]
        for arm in reproduced:
            counts[category][f"{arm}_reproduced"] += reproduced[arm]
        detail.append({"case_id": i, "example_id": case["example_id"],
                       "candidate_index": case["candidate_index"], "cached_outcome": category,
                       "f1": f1, "success_090": success, "reproduced": reproduced,
                       "context_tokens": {arm: arms[arm]["context_tokens"] for arm in ARMS}})
    summary = {"protocol": cfg["protocol"], "cases": len(selected), "target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "by_cached_outcome": {k: dict(v) for k, v in counts.items()},
               "component_success_patterns_T_S_TS": {k: dict(v) for k, v in patterns.items()},
               "unique_queries_by_cached_outcome": {name: len({r["example_id"] for r in detail
                    if r["cached_outcome"] == name}) for name in counts},
               "repair_queries_with_any_success_by_arm": {arm: len({r["example_id"] for r in detail
                    if r["cached_outcome"] == "repair" and r["success_090"][arm]})
                    for arm in ARMS},
               "sealed_sets_read": False,
               "limitations": ["Cases selected by cached outcome; not a population estimate.",
                               "A title-only or sentence-only arm changes actual context length and is a mechanism ablation, not a deployable compression policy.",
                               "If fresh baseline/title+sentence does not reproduce cached outcomes, component attribution is restricted to fresh paired comparisons."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_case.jsonl").open("w") as handle:
        for row in detail:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
