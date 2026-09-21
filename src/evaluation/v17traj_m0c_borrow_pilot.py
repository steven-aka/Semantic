"""Paired fresh five-anchor Target audit for one early-borrow trajectory."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments

BASE = Path("results/v2_rank_then_cut")
LINEAGE = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17traj_m0c_borrow_pilot"
CFG = Path("configs/v17traj_m0c_borrow_pilot.json")
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)
DEPTHS = (6, 7, 7, 9, 10)


def prepare(cfg, tokenizer):
    eligible = {r["example_id"]: r for r in read_jsonl(BASE / "v17traj_m0b_borrow_preflight/per_query.jsonl") if r["eligible"]}
    r1 = {r["example_id"] for r in read_jsonl(BASE / "v17packet_r1_label_scale512/per_query.jsonl")}
    obs = set(json.loads((BASE / "v17packet_obs_a1_natural_insertion/manifest.json").read_text())["query_ids"])
    frag = set(json.loads((BASE / "v17packet_frag_b0_title_only_natural_pilot/manifest.json").read_text())["query_ids"])
    frame = set(eligible) - r1 - obs - frag
    chosen = sorted(frame, key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))[:cfg["queries"]]
    assert len(chosen) == cfg["queries"]
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen}
    assert len(data) == len(orders) == len(chosen)
    tasks, metadata = [], []
    for q in chosen:
        packets, order = data[q]["packet_texts"], orders[q]
        p10 = order[9]
        fragment = segments(packets[p10])[eligible[q]["candidate_index"]]
        baseline = {d: render(packets, mask(order, d)) for d in (6, 7, 9, 10)}
        action = {d: render(packets, mask(order, d) | (1 << p10), {p10: fragment}) for d in (7, 9)}
        assert render(packets, mask(order, 10)) == baseline[10]
        for arm, context in (("base6", baseline[6]), ("base7", baseline[7]),
                             ("action7", action[7]), ("base9", baseline[9]),
                             ("action9", action[9]), ("base10", baseline[10])):
            tasks.append({"example_id": q, "arm": arm, "question": data[q]["question"],
                          "context": context,
                          "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
        metadata.append({"example_id": q, "attainable_levels": data[q]["attainable_levels"],
                         "candidate_index": eligible[q]["candidate_index"],
                         "depth7_slack": eligible[q]["depth7_slack"],
                         "depth9_slack": eligible[q]["depth9_slack"],
                         "cumulative_slack": eligible[q]["cumulative_slack"]})
    assert len(tasks) == len(chosen) * 6
    return chosen, tasks, metadata, len(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS" and cfg["schedule"] == list(DEPTHS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    chosen, tasks, metadata, frame_size = prepare(cfg, tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "query_ids": chosen,
                "eligible_after_exclusions": frame_size, "planned_target_calls": len(tasks),
                "task_sha256": hashlib.sha256("\n".join(
                    f'{t["example_id"]}:{t["arm"]}:{hashlib.sha256(t["context"].encode()).hexdigest()}' for t in tasks
                ).encode()).hexdigest(), "sealed_sets_read": False}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids"}, indent=2))
        return
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in set(chosen)}
    assert len(atoms) == len(chosen)
    call_path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(call_path)} if call_path.exists() else {}
    pending = [t for t in tasks if (t["example_id"], t["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with call_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([t["question"] for t in batch], [t["context"] for t in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    prediction = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"],
                           "context_tokens": task["context_tokens"],
                           "f1": float(qampari_list_metrics(prediction, atoms[task["example_id"]])["f1"]),
                           "raw_output": output.outputs[0].text,
                           "generated_token_ids": list(output.outputs[0].token_ids),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    per_query = []
    for item in metadata:
        q = item["example_id"]
        attainable = {float(x) for x in item["attainable_levels"]}
        base_arms = ("base6", "base7", "base7", "base9", "base10")
        act_arms = ("base6", "action7", "action7", "action9", "base10")
        vectors, costs = {}, {}
        for name, arms in (("baseline", base_arms), ("action", act_arms)):
            vectors[name] = [None if level not in attainable else done[q, arm]["f1"] + cfg["fidelity_epsilon"] >= level
                             for level, arm in zip(LEVELS, arms)]
            costs[name] = {
                "context": sum(done[q, arm]["context_tokens"] for level, arm in zip(LEVELS, arms) if level in attainable),
                "target": sum(done[q, arm]["prompt_tokens"] + done[q, arm]["generated_tokens"]
                              for level, arm in zip(LEVELS, arms) if level in attainable)}
        assert costs["action"]["context"] - costs["baseline"]["context"] == item["cumulative_slack"]
        before, after = vectors["baseline"], vectors["action"]
        per_query.append({**item, "baseline_hits": before, "action_hits": after,
                          "baseline_complete": all(x is not False for x in before),
                          "action_complete": all(x is not False for x in after),
                          "anchor_breaks": sum(x is True and y is False for x, y in zip(before, after)),
                          "anchor_repairs": sum(x is False and y is True for x, y in zip(before, after)),
                          "baseline_cost": costs["baseline"], "action_cost": costs["action"]})
    safe_repairs = [r for r in per_query if not r["baseline_complete"] and r["action_complete"] and r["anchor_breaks"] == 0]
    result = {"protocol": cfg["protocol"], "queries": len(per_query), "target_calls": len(done),
              "baseline_success": [sum(r["baseline_hits"][i] is True for r in per_query) for i in range(5)],
              "action_success": [sum(r["action_hits"][i] is True for r in per_query) for i in range(5)],
              "baseline_complete": sum(r["baseline_complete"] for r in per_query),
              "action_complete": sum(r["action_complete"] for r in per_query),
              "complete_repairs": sum(not r["baseline_complete"] and r["action_complete"] for r in per_query),
              "complete_breaks": sum(r["baseline_complete"] and not r["action_complete"] for r in per_query),
              "queries_with_any_anchor_break": sum(r["anchor_breaks"] > 0 for r in per_query),
              "safe_complete_oracle_repairs": len(safe_repairs),
              "mean_extra_cumulative_context_uniform": sum(r["action_cost"]["context"] - r["baseline_cost"]["context"] for r in per_query) / len(per_query),
              "mean_extra_cumulative_target_uniform": sum(r["action_cost"]["target"] - r["baseline_cost"]["target"] for r in per_query) / len(per_query),
              "mean_extra_cumulative_context_safe_complete_oracle": sum(r["action_cost"]["context"] - r["baseline_cost"]["context"] for r in safe_repairs) / len(per_query),
              "opportunity_gate": "GO_M0C_LEARNABILITY_DESIGN_ONLY" if len(safe_repairs) >= 8 and sum(r["cumulative_slack"] for r in safe_repairs) / len(per_query) <= 10 else "STOP_M0C_COSTED_ORACLE_GATE",
              "sealed_sets_read": False,
              "limitations": "Outcome-aware safe oracle uses forbidden Target outcomes; no controller is trained or validated on unseen queries."}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
