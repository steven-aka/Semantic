"""Prospective low-cumulative-cost version of the frozen M0C borrow action."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments

BASE = Path("results/v2_rank_then_cut")
LINEAGE = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17traj_m1_low_cost_borrow"
CFG = Path("configs/v17traj_m1_low_cost_borrow.json")
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)
DEPTHS = (6, 7, 7, 9, 10)


def prepare(cfg, tokenizer):
    earlier = {r["example_id"]: r for r in read_jsonl(BASE / "v17traj_m0b_borrow_preflight/per_query.jsonl") if r["eligible"]}
    exclusion = {r["example_id"] for r in read_jsonl(BASE / "v17packet_r1_label_scale512/per_query.jsonl")}
    for name in ("v17packet_obs_a1_natural_insertion", "v17packet_frag_b0_title_only_natural_pilot", "v17traj_m0c_borrow_pilot"):
        exclusion.update(json.loads((BASE / name / "manifest.json").read_text())["query_ids"])
    chosen = sorted(set(earlier) - exclusion, key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))
    assert len(chosen) == 116
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen}
    assert len(data) == len(orders) == len(chosen)
    tasks, metadata = [], []
    for q in chosen:
        packets, order = data[q]["packet_texts"], orders[q]
        p10 = order[9]
        fragment = segments(packets[p10])[earlier[q]["candidate_index"]]
        base = {d: render(packets, mask(order, d)) for d in (6, 7, 9, 10)}
        use_action = earlier[q]["cumulative_slack"] <= cfg["max_cumulative_context_slack"]
        action = {d: render(packets, mask(order, d) | (1 << p10), {p10: fragment}) for d in (7, 9)} if use_action else {}
        assert render(packets, mask(order, 10)) == base[10]
        for arm, context in (("base6", base[6]), ("base7", base[7]),
                             ("base9", base[9]), ("base10", base[10])):
            tasks.append({"example_id": q, "arm": arm, "question": data[q]["question"],
                          "context": context, "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
        if use_action:
            for arm, context in (("action7", action[7]), ("action9", action[9])):
                tasks.append({"example_id": q, "arm": arm, "question": data[q]["question"],
                              "context": context, "context_tokens": len(tokenizer.encode(context, add_special_tokens=False))})
        metadata.append({"example_id": q, "attainable_levels": data[q]["attainable_levels"],
                         "candidate_index": earlier[q]["candidate_index"], "action_eligible": use_action,
                         "cumulative_slack": earlier[q]["cumulative_slack"] if use_action else 0})
    return chosen, tasks, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS" and cfg["schedule"] == list(DEPTHS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    chosen, tasks, meta = prepare(cfg, tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "query_ids": chosen, "eligible_action_queries": sum(r["action_eligible"] for r in meta),
                "planned_target_calls": len(tasks),
                "task_sha256": hashlib.sha256("\n".join(
                    f'{t["example_id"]}:{t["arm"]}:{hashlib.sha256(t["context"].encode()).hexdigest()}' for t in tasks
                ).encode()).hexdigest(), "sealed_sets_read": False}
    mpath = OUT / "manifest.json"
    if mpath.exists():
        assert json.loads(mpath.read_text()) == manifest
    else:
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids"}, indent=2))
        return
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in set(chosen)}
    assert len(atoms) == len(chosen)
    path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(path)} if path.exists() else {}
    pending = [t for t in tasks if (t["example_id"], t["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([t["question"] for t in batch], [t["context"] for t in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"], "context_tokens": task["context_tokens"],
                           "f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                           "raw_output": output.outputs[0].text, "generated_token_ids": list(output.outputs[0].token_ids),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    per_query = []
    for item in meta:
        q = item["example_id"]
        attainable = {float(x) for x in item["attainable_levels"]}
        arms = {"baseline": ("base6", "base7", "base7", "base9", "base10"),
                "action": ("base6", "action7", "action7", "action9", "base10") if item["action_eligible"]
                          else ("base6", "base7", "base7", "base9", "base10")}
        hits, cost = {}, {}
        for name, choices in arms.items():
            hits[name] = [None if level not in attainable else done[q, arm]["f1"] + cfg["fidelity_epsilon"] >= level
                          for level, arm in zip(LEVELS, choices)]
            cost[name] = {"context": sum(done[q, arm]["context_tokens"] for level, arm in zip(LEVELS, choices) if level in attainable),
                          "target": sum(done[q, arm]["prompt_tokens"] + done[q, arm]["generated_tokens"]
                                        for level, arm in zip(LEVELS, choices) if level in attainable)}
        assert cost["action"]["context"] - cost["baseline"]["context"] == item["cumulative_slack"]
        before, after = hits["baseline"], hits["action"]
        per_query.append({**item, "baseline_hits": before, "action_hits": after,
                          "baseline_complete": all(x is not False for x in before),
                          "action_complete": all(x is not False for x in after),
                          "anchor_breaks": sum(x is True and y is False for x, y in zip(before, after)),
                          "anchor_repairs": sum(x is False and y is True for x, y in zip(before, after)),
                          "baseline_cost": cost["baseline"], "action_cost": cost["action"]})
    safe = [r for r in per_query if not r["baseline_complete"] and r["action_complete"] and r["anchor_breaks"] == 0]
    cost_per_repair = sum(r["cumulative_slack"] for r in safe) / len(safe) if safe else None
    baseline_complete = sum(r["baseline_complete"] for r in per_query)
    action_complete = sum(r["action_complete"] for r in per_query)
    result = {"protocol": cfg["protocol"], "queries": len(per_query), "eligible_action_queries": manifest["eligible_action_queries"],
              "target_calls": len(done),
              "baseline_success": [sum(r["baseline_hits"][i] is True for r in per_query) for i in range(5)],
              "action_success": [sum(r["action_hits"][i] is True for r in per_query) for i in range(5)],
              "baseline_complete": baseline_complete, "action_complete": action_complete,
              "complete_repairs": sum(not r["baseline_complete"] and r["action_complete"] for r in per_query),
              "complete_breaks": sum(r["baseline_complete"] and not r["action_complete"] for r in per_query),
              "safe_complete_oracle_repairs": len(safe),
              "safe_complete_oracle_extra_cumulative_context_per_repair": cost_per_repair,
              "mean_extra_cumulative_context_uniform": sum(r["action_cost"]["context"] - r["baseline_cost"]["context"] for r in per_query) / len(per_query),
              "mean_extra_cumulative_target_uniform": sum(r["action_cost"]["target"] - r["baseline_cost"]["target"] for r in per_query) / len(per_query),
              "opportunity_gate": "GO_M1_LEARNABILITY_DESIGN_ONLY" if len(safe) >= 8 and cost_per_repair is not None and cost_per_repair <= 50 and action_complete >= baseline_complete else "STOP_M1_EXTRACTIVE_BORROW_GATE",
              "sealed_sets_read": False,
              "limitations": "Safe oracle is hindsight-only; GO authorizes design, not training, deployment or sealed evaluation."}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
