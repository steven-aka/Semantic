"""Small frozen B/A/D/AD Target replay on train-side packet repairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


CFG = Path("configs/v17packet_mech_a1_factorial_replay.json")
ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
R1 = ROOT / "v17packet_r1_label_scale512"
P0 = ROOT / "v17canon_p0_fresh_v8_prefix_chain"
M0 = ROOT / "v17packet_mech_a0_cached_audit"
OUT = ROOT / "v17packet_mech_a1_factorial_replay"


def sha(q: str) -> str:
    return hashlib.sha256(q.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_MECHANISM"
    r1 = {r["example_id"]: r for r in read_jsonl(R1 / "per_query.jsonl")}
    mech = list(read_jsonl(M0 / "per_query.jsonl"))
    repairs = sorted((x for x in mech if x["kind"] == "repairable_failure"),
                     key=lambda x: sha(x["example_id"]))
    novel_controls = sorted((x for x in mech if x["kind"] == "unrepairable_failure"
                             and x["candidate_count"] and x["any_candidate_novel_gold_mention"]),
                            key=lambda x: sha(x["example_id"]))[:20]
    other_controls = sorted((x for x in mech if x["kind"] == "unrepairable_failure"
                             and x["candidate_count"] and not x["any_candidate_novel_gold_mention"]),
                            key=lambda x: sha(x["example_id"]))
    assert len(repairs) == 39 and len(novel_controls) == 20 and len(other_controls) == 4
    selected = [("repair", x) for x in repairs] + [
        ("control_novel_gold_mention", x) for x in novel_controls] + [
        ("control_no_novel_gold_mention", x) for x in other_controls]
    ids = {x["example_id"] for _, x in selected}
    data = {x["example_id"]: x for x in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl")
            if x["example_id"] in ids}
    orders = {x["example_id"]: x["decoded_order"] for x in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if x["example_id"] in ids}
    atoms = {x["example_id"]: x["answer_atoms"] for x in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if x["example_id"] in ids}
    cache = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in ids:
                cache[row["example_id"], row["mask"]] = row
    old_ad = {(x["example_id"], x["sentence_index"]): x for x in read_jsonl(R1 / "per_sentence.jsonl")
              if x["example_id"] in ids}
    assert len(data) == len(orders) == len(atoms) == len(ids) == 63
    tasks, cases = [], []
    for group, item in selected:
        q = item["example_id"]
        source, order = data[q], orders[q]
        packet_id = order[9]
        packet = source["packet_texts"][packet_id]
        fragments = segments(packet)
        base = r1[q]["baseline"]
        if group == "repair":
            index = item["selected_sentence_index"]
        else:
            eligible = [a for a in r1[q]["sentence_options"]
                        if a["context_tokens"] <= base["context_tokens"]]
            assert eligible
            index = min(eligible, key=lambda a: (a["context_tokens"], a["index"]))["index"]
        fragment = fragments[index]
        m8, m9 = mask(order, 8), mask(order, 9)
        b = cache[q,m9]
        d = cache[q,m8]
        ad = old_ad[q,index]
        assert abs(b["f1"]-item["baseline_f1"]) < 1e-8
        if group == "repair":
            assert abs(ad["f1"]-item["repair_f1"]) < 1e-8
        a_context = render(source["packet_texts"], m9 | (1 << packet_id), {packet_id: fragment})
        ad_context = render(source["packet_texts"], m8 | (1 << packet_id), {packet_id: fragment})
        assert ad_context != a_context and fragment in a_context and fragment in ad_context
        assert render(source["packet_texts"], m9) != a_context
        cases.append({"example_id": q, "group": group, "sentence_index": index,
                      "cached_B_f1": b["f1"], "cached_D_f1": d["f1"],
                      "cached_AD_f1": ad["f1"], "cached_B_tokens": b["tokens"],
                      "cached_D_tokens": d["tokens"], "cached_AD_tokens": ad["tokens"]})
        tasks.append({"example_id": q, "group": group, "arm": "A", "context": a_context,
                      "question": source["question"]})
        if group == "repair":
            tasks.append({"example_id": q, "group": group, "arm": "B_repeat",
                          "context": render(source["packet_texts"], m9), "question": source["question"]})
            tasks.append({"example_id": q, "group": group, "arm": "AD_repeat",
                          "context": ad_context, "question": source["question"]})
    assert len(tasks) == cfg["new_target_call_budget"] == 141
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "selected_queries": len(cases),
                "repair_queries": 39, "controls_with_novel_gold_mention": 20,
                "controls_without_novel_gold_mention": 4,
                "new_target_calls": len(tasks),
                "task_signature_sha256": hashlib.sha256("\n".join(
                    f'{x["example_id"]}:{x["arm"]}:{hashlib.sha256(x["context"].encode()).hexdigest()}'
                    for x in tasks).encode()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    if args.preflight_only:
        return
    path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [x for x in tasks if (x["example_id"], x["arm"]) not in done]
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
                prompts = target._chat_prompts([x["question"] for x in batch],
                                               [x["context"] for x in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "group": task["group"],
                           "arm": task["arm"], "f1": float(qampari_list_metrics(
                               parsed, atoms[task["example_id"]])["f1"]),
                           "context_tokens": count_tokens(target.tokenizer, task["context"]),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids),
                           "prediction": " # ".join(parsed)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed_calls": len(done), "total_calls": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    detail = []
    for case in cases:
        q = case["example_id"]
        a = done[q,"A"]
        result = {**case, "A_f1": a["f1"], "A_context_tokens": a["context_tokens"],
                  "A_success": a["f1"]+cfg["fidelity_epsilon"] >= 0.90,
                  "D_success": case["cached_D_f1"]+cfg["fidelity_epsilon"] >= 0.90}
        if case["group"] == "repair":
            b, ad = done[q,"B_repeat"], done[q,"AD_repeat"]
            result.update({"B_repeat_f1": b["f1"], "AD_repeat_f1": ad["f1"],
                           "repeat_pair_valid": b["f1"]+cfg["fidelity_epsilon"] < 0.90 <=
                           ad["f1"]+cfg["fidelity_epsilon"],
                           "A_delta_over_B_repeat": a["f1"]-b["f1"],
                           "AD_delta_over_B_repeat": ad["f1"]-b["f1"]})
        detail.append(result)
    repairs = [r for r in detail if r["group"] == "repair"]
    stable = [r for r in repairs if r["repeat_pair_valid"]]
    summary = {"protocol": cfg["protocol"], "new_target_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "repair_cases": len(repairs), "repeat_pair_valid": len(stable),
               "B_repeat_success_among_repairs": sum(r["B_repeat_f1"]+cfg["fidelity_epsilon"] >= 0.90 for r in repairs),
               "AD_repeat_success_among_repairs": sum(r["AD_repeat_f1"]+cfg["fidelity_epsilon"] >= 0.90 for r in repairs),
               "A_success_among_all_repairs": sum(r["A_success"] for r in repairs),
               "A_success_among_repeat_valid": sum(r["A_success"] for r in stable),
               "A_failure_but_AD_success_among_repeat_valid": sum(not r["A_success"] for r in stable),
               "D_success_among_all_repairs": sum(r["D_success"] for r in repairs),
               "mean_A_context_tokens_minus_B_among_repeat_valid": sum(
                   r["A_context_tokens"]-r["cached_B_tokens"] for r in stable)/len(stable),
               "mean_AD_context_tokens_minus_B_among_repeat_valid": sum(
                   r["cached_AD_tokens"]-r["cached_B_tokens"] for r in stable)/len(stable),
               "controls": {group: {"queries": sum(r["group"] == group for r in detail),
                                      "A_success": sum(r["group"] == group and r["A_success"] for r in detail)}
                            for group in ("control_novel_gold_mention", "control_no_novel_gold_mention")},
               "limitations": "Outcome-selected repairs and single-run cached D; A adds context and D removes context. This is mechanism diagnosis, not deployable quality-token Pareto evidence.",
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_case.jsonl").open("w") as handle:
        for row in detail:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
