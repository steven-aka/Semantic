"""Fresh paired five-anchor audit of frozen budget-neutral promote/delay actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.evaluation.v17traj_m2a_promote_delay_preflight import rank7_remainders

BASE = Path("results/v2_rank_then_cut")
LINEAGE = BASE / "v17sel_b2b_lineage_clean_holdout"
OUT = BASE / "v17traj_m2b_promote_delay_pilot"
CFG = Path("configs/v17traj_m2b_promote_delay_pilot.json")
LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95)


def prepare(cfg, tokenizer):
    legal = {}
    for row in read_jsonl(BASE / "v17traj_m2a_promote_delay_preflight/legal_actions.jsonl"):
        legal.setdefault(row["example_id"], []).append(row)
    exclusion = set()
    for name in ("v17traj_m0c_borrow_pilot", "v17traj_m1_low_cost_borrow",
                 "v17packet_obs_a1_natural_insertion", "v17packet_frag_b0_title_only_natural_pilot"):
        exclusion.update(json.loads((BASE / name / "manifest.json").read_text())["query_ids"])
    pool = sorted(set(legal) - exclusion, key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))
    assert len(pool) == 198
    chosen = pool[:cfg["queries"]]
    data = {r["example_id"]: r for r in read_jsonl(LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in chosen}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in chosen}
    assert len(data) == len(orders) == len(chosen) == cfg["queries"]
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    tasks, metadata = [], []
    for q in chosen:
        packets, order = data[q]["packet_texts"], orders[q]
        p7, p10 = order[6], order[9]
        base = {d: render(packets, mask(order, d)) for d in (6, 7, 9, 10)}
        options = sorted(legal[q], key=lambda a: (-a["delta_cumulative_context_tokens"], a["delayed_sentence_index"]))
        actions, seen = [], set()
        for row in options:
            delayed_index = row["delayed_sentence_index"]
            remainder = {i: r for i, _, r in rank7_remainders(packets[p7])}[delayed_index]
            fragment = segments(packets[p10])[row["promoted_sentence_index"]]
            override = {p7: remainder, p10: fragment}
            at7 = render(packets, mask(order, 7) | (1 << p10), override)
            at9 = render(packets, mask(order, 9) | (1 << p10), override)
            key = (at7, at9)
            if key in seen:
                continue
            seen.add(key)
            delta7, delta9 = count(at7) - count(base[7]), count(at9) - count(base[9])
            assert delta7 == row["delta_context_tokens_at_depth7"] and delta9 == row["delta_context_tokens_at_depth9"]
            assert delta7 <= 0 and delta9 <= 0 and 2 * delta7 + delta9 <= 0
            assert render(packets, mask(order, 10)) == base[10]
            arm = f"action{len(actions)}"
            actions.append({**row, "arm": arm})
            for depth, context in ((7, at7), (9, at9)):
                tasks.append({"example_id": q, "arm": f"{arm}_{depth}", "question": data[q]["question"],
                              "context": context, "context_tokens": count(context)})
            if len(actions) == cfg["max_actions_per_query"]:
                break
        for depth, context in base.items():
            tasks.append({"example_id": q, "arm": f"base{depth}", "question": data[q]["question"],
                          "context": context, "context_tokens": count(context)})
        metadata.append({"example_id": q, "attainable_levels": data[q]["attainable_levels"], "actions": actions})
    return chosen, pool, tasks, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS" and cfg["schedule"] == [6, 7, 7, 9, 10]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    chosen, pool, tasks, metadata = prepare(cfg, tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "query_ids": chosen, "available_query_pool": len(pool),
                "planned_target_calls": len(tasks), "action_counts": [len(m["actions"]) for m in metadata],
                "task_sha256": hashlib.sha256("\n".join(f'{t["example_id"]}:{t["arm"]}:{hashlib.sha256(t["context"].encode()).hexdigest()}' for t in tasks).encode()).hexdigest(),
                "sealed_sets_read": False}
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "query_ids" and k != "action_counts"}, indent=2))
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
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"], "context_tokens": task["context_tokens"],
                           "f1": float(qampari_list_metrics(answers, atoms[task["example_id"]])["f1"]),
                           "raw_output": output.outputs[0].text, "generated_token_ids": list(output.outputs[0].token_ids),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)), "generated_tokens": len(output.outputs[0].token_ids)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    results = []
    for m in metadata:
        q = m["example_id"]
        attainable = {float(x) for x in m["attainable_levels"]}
        paths = [("baseline", None), *((a["arm"], a) for a in m["actions"])]
        scores = {}
        for name, action in paths:
            arms = ("base6", "base7", "base7", "base9", "base10") if action is None else ("base6", name + "_7", name + "_7", name + "_9", "base10")
            hits = [None if level not in attainable else done[q, arm]["f1"] + cfg["fidelity_epsilon"] >= level for level, arm in zip(LEVELS, arms)]
            cost = sum(done[q, arm]["context_tokens"] for level, arm in zip(LEVELS, arms) if level in attainable)
            target_cost = sum(done[q, arm]["prompt_tokens"] + done[q, arm]["generated_tokens"] for level, arm in zip(LEVELS, arms) if level in attainable)
            scores[name] = {"hits": hits, "complete": all(x is not False for x in hits), "cumulative_context_tokens": cost, "cumulative_target_tokens": target_cost}
        for a in m["actions"]:
            assert scores[a["arm"]]["cumulative_context_tokens"] - scores["baseline"]["cumulative_context_tokens"] == a["delta_cumulative_context_tokens"]
        results.append({"example_id": q, "attainable_levels": m["attainable_levels"], "actions": m["actions"], "scores": scores})
    fixed = []; safe = []; failure_masks = set()
    for row in results:
        before = row["scores"]["baseline"]
        chosen_action = row["scores"]["action0"]
        fixed.append(chosen_action)
        for a in row["actions"]:
            candidate = row["scores"][a["arm"]]
            if (not before["complete"] and candidate["complete"] and
                    not any(x is True and y is False for x, y in zip(before["hits"], candidate["hits"]))):
                safe.append(row["example_id"])
                failure_masks.add(tuple(x is False for x in before["hits"]))
                break
    result = {"protocol": cfg["protocol"], "queries": len(results), "target_calls": len(done),
              "actions_tested": sum(len(r["actions"]) for r in results),
              "baseline_success": [sum(r["scores"]["baseline"]["hits"][i] is True for r in results) for i in range(5)],
              "fixed_success": [sum(r["scores"]["action0"]["hits"][i] is True for r in results) for i in range(5)],
              "baseline_complete": sum(r["scores"]["baseline"]["complete"] for r in results),
              "fixed_complete": sum(x["complete"] for x in fixed),
              "fixed_complete_repairs": sum(not r["scores"]["baseline"]["complete"] and r["scores"]["action0"]["complete"] for r in results),
              "fixed_complete_breaks": sum(r["scores"]["baseline"]["complete"] and not r["scores"]["action0"]["complete"] for r in results),
              "safe_oracle_complete_repairs": len(safe), "safe_oracle_failure_masks": len(failure_masks),
              "mean_fixed_cumulative_context_delta": sum(r["scores"]["action0"]["cumulative_context_tokens"] - r["scores"]["baseline"]["cumulative_context_tokens"] for r in results) / len(results),
              "mean_fixed_cumulative_target_delta": sum(r["scores"]["action0"]["cumulative_target_tokens"] - r["scores"]["baseline"]["cumulative_target_tokens"] for r in results) / len(results),
              "opportunity_gate": "GO_M2_LEARNABILITY_DESIGN_ONLY" if len(safe) >= 12 and len(failure_masks) >= 2 else "STOP_M2_ACTION_SPACE_GATE",
              "sealed_sets_read": False,
              "limitations": "Safe oracle chooses actions using forbidden Target outcomes; fixed action is a frozen deployment-visible rule. The 0.70 read at depth7 changes. Exact depth10 recovery makes 0.95 identical by construction."}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
