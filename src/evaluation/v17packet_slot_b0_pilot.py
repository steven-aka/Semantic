"""Bounded-slack insertion pilot on query-selected train examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner


ROOT = Path("results/v2_rank_then_cut")
LINEAGE = ROOT / "v17sel_b2b_lineage_clean_holdout"
R1 = ROOT / "v17packet_r1_label_scale512"
P0 = ROOT / "v17canon_p0_fresh_v8_prefix_chain"
OUT = ROOT / "v17packet_slot_b0_pilot"
CFG = Path("configs/v17packet_slot_b0_pilot.json")


def sha(q: str) -> str:
    return hashlib.sha256(q.encode()).hexdigest()


def prepare(cfg: dict, tokenizer) -> tuple[list[dict], list[dict], dict]:
    rows = list(read_jsonl(R1 / "per_query.jsonl"))
    ids = {r["example_id"] for r in rows}
    data = {r["example_id"]: r for r in read_jsonl(
        LINEAGE / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(
        LINEAGE / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    assert len(rows) == len(data) == len(orders) == 512
    eligible = defaultdict(list)
    for r in sorted(rows, key=lambda x: sha(x["example_id"])):
        q = r["example_id"]
        packets, order = data[q]["packet_texts"], orders[q]
        candidate_id = order[9]
        fragments = segments(packets[candidate_id])
        if not fragments:
            continue
        m9 = mask(order, 9)
        base_context = render(packets, m9)
        base_tokens = count_tokens(tokenizer, base_context)
        assert base_tokens > 0
        candidates = []
        for i, sentence in enumerate(fragments):
            context = render(packets, m9 | (1 << candidate_id), {candidate_id: sentence})
            cost = count_tokens(tokenizer, context)
            slack = cost - base_tokens
            if 0 < slack <= max(cfg["slack_caps"]):
                candidates.append({"example_id": q, "candidate_index": i,
                                   "context": context, "context_tokens": cost,
                                   "baseline_tokens": base_tokens, "slack": slack,
                                   "question": data[q]["question"]})
        if candidates:
            stratum = "v8_success" if r["baseline"]["hits"][3] else "v8_failure"
            eligible[stratum].append((r, candidates))
    selected = []
    tasks = []
    for stratum in ("v8_success", "v8_failure"):
        chosen = eligible[stratum][:cfg["sample_per_baseline_stratum"]]
        assert len(chosen) == cfg["sample_per_baseline_stratum"]
        for r, candidates in chosen:
            selected.append({"example_id": r["example_id"], "stratum": stratum,
                             "candidate_count": len(candidates),
                             "shortest_index": min(candidates, key=lambda c: (c["slack"], c["candidate_index"]))["candidate_index"]})
            tasks.extend(candidates)
    assert len(selected) == 24 and len(tasks) <= cfg["max_fresh_calls"]
    return selected, tasks, {r["example_id"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_BEFORE_TARGET_CALLS"
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    selected, tasks, r1 = prepare(cfg, tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": cfg["protocol"], "selected": selected,
                "candidate_tasks": len(tasks),
                "slack_distribution": {str(cap): sum(t["slack"] <= cap for t in tasks)
                                       for cap in cfg["slack_caps"]},
                "task_sha256": hashlib.sha256("\n".join(
                    f'{r["example_id"]}:{r["candidate_index"]}:{hashlib.sha256(r["context"].encode()).hexdigest()}'
                    for r in tasks).encode()).hexdigest(),
                "sealed_sets_read": False}
    mpath = OUT / "manifest.json"
    if mpath.exists():
        assert json.loads(mpath.read_text()) == manifest
    else:
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps({k: manifest[k] for k in ("candidate_tasks", "slack_distribution", "task_sha256")}, indent=2))
        return
    chosen_ids = {r["example_id"] for r in selected}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in chosen_ids}
    assert len(atoms) == len(chosen_ids)
    cache = {}
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in chosen_ids:
                cache[row["example_id"], row["depth"]] = row
    assert all((q, d) in cache for q in chosen_ids for d in (9, 10))
    per_call = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["candidate_index"]): r for r in read_jsonl(per_call)} if per_call.exists() else {}
    pending = [t for t in tasks if (t["example_id"], t["candidate_index"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with per_call.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([t["question"] for t in batch], [t["context"] for t in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for t, prompt, output in zip(batch, prompts, outputs):
                    answers = parse_list_prediction(output.outputs[0].text)
                    row = {k: t[k] for k in ("example_id", "candidate_index", "context_tokens", "baseline_tokens", "slack")}
                    row.update({"f1": float(qampari_list_metrics(answers, atoms[t["example_id"]])["f1"]),
                                "prediction": " # ".join(answers),
                                "prompt_tokens": len(target.tokenizer.encode(prompt)),
                                "generated_tokens": len(output.outputs[0].token_ids)})
                    done[row["example_id"], row["candidate_index"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    per_query = []
    for entry in selected:
        q = entry["example_id"]
        candidates = [r for (qid, _), r in done.items() if qid == q]
        b9, b10 = cache[q, 9], cache[q, 10]
        fixed = next(r for r in candidates if r["candidate_index"] == entry["shortest_index"])
        row = {"example_id": q, "stratum": entry["stratum"],
               "baseline_f1": b9["f1"], "baseline_tokens": b9["tokens"],
               "baseline_hits": r1[q]["baseline"]["hits"],
               "depth10_f1": b10["f1"], "depth10_tokens": b10["tokens"],
               "fixed_shortest": {k: fixed[k] for k in ("candidate_index", "f1", "slack")},
               "by_cap": {}}
        for cap in cfg["slack_caps"]:
            options = [r for r in candidates if r["slack"] <= cap]
            # Outcome-based upper bound with STAY. The strict .90 objective
            # chooses no action when baseline succeeds or no repair exists.
            repairing = [r for r in options if r["f1"] + cfg["fidelity_epsilon"] >= .90]
            if b9["f1"] + cfg["fidelity_epsilon"] >= .90 or not repairing:
                row["by_cap"][str(cap)] = {"candidate_index": None, "f1": b9["f1"],
                                          "slack": 0, "eligible_options": len(options)}
            else:
                best = min(repairing, key=lambda r: (r["slack"], r["candidate_index"]))
                row["by_cap"][str(cap)] = {**{k: best[k] for k in ("candidate_index", "f1", "slack")},
                                          "eligible_options": len(options)}
        per_query.append(row)
    summary = {"protocol": cfg["protocol"], "queries": len(per_query),
               "candidate_calls": len(done),
               "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
               "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
               "v8_depth10_success_090": sum(r["depth10_f1"] + 1e-6 >= .90 for r in per_query),
               "v8_depth10_mean_extra_context_tokens": sum(r["depth10_tokens"] - r["baseline_tokens"]
                                                           for r in per_query) / len(per_query),
               "by_stratum": {}, "by_cap_oracle": {}, "sealed_sets_read": False,
               "limitations": ["R1 train512 is design-exposed; this is a bounded train-side action-space pilot, not independent validation.",
                               "Candidate selection for by_cap_oracle uses Target outcomes and is not deployable.",
                               "V8 depths are discrete; there may be no V8 prefix at exactly the insertion cost.",
                               "The depth10 context is the original canonical full context, with no duplicate candidate text."]}
    for stratum in ("v8_success", "v8_failure"):
        rows = [r for r in per_query if r["stratum"] == stratum]
        summary["by_stratum"][stratum] = {"queries": len(rows),
            "baseline_five_level_success": [sum(r["baseline_hits"][i] is True for r in rows) for i in range(5)],
            "fixed_shortest_five_level_success": [sum((r["fixed_shortest"]["f1"] + 1e-6 >= .90)
                                                       if i == 3 else (r["baseline_hits"][i] is True)
                                                       for r in rows) for i in range(5)],
            "baseline_success_090": sum(r["baseline_f1"] + 1e-6 >= .90 for r in rows),
            "fixed_shortest_success_090": sum(r["fixed_shortest"]["f1"] + 1e-6 >= .90 for r in rows),
            "fixed_shortest_repairs": sum(r["baseline_f1"] + 1e-6 < .90 and r["fixed_shortest"]["f1"] + 1e-6 >= .90 for r in rows),
            "fixed_shortest_breaks": sum(r["baseline_f1"] + 1e-6 >= .90 and r["fixed_shortest"]["f1"] + 1e-6 < .90 for r in rows),
            "baseline_complete": sum(all(hit is not False for hit in r["baseline_hits"]) for r in rows),
            "fixed_shortest_complete": sum(all(r["baseline_hits"][i] is not False for i in (0,1,2,4)) and
                                           r["fixed_shortest"]["f1"] + 1e-6 >= .90 for r in rows),
            "fixed_shortest_mean_slack": sum(r["fixed_shortest"]["slack"] for r in rows) / len(rows)}
    for cap in cfg["slack_caps"]:
        rows = [(r, r["by_cap"][str(cap)]) for r in per_query]
        summary["by_cap_oracle"][str(cap)] = {"eligible_queries": sum(c["eligible_options"] > 0 for _,c in rows),
            "queries": len(rows),
            "oracle_five_level_success": [sum((c["f1"] + 1e-6 >= .90)
                                              if i == 3 else (r["baseline_hits"][i] is True)
                                              for r,c in rows) for i in range(5)],
            "baseline_success_090": sum(r["baseline_f1"] + 1e-6 >= .90 for r, _ in rows),
            "oracle_success_090": sum(c["f1"] + 1e-6 >= .90 for _, c in rows),
            "oracle_repairs": sum(r["baseline_f1"] + 1e-6 < .90 and c["f1"] + 1e-6 >= .90 for r, c in rows),
            "oracle_breaks": sum(r["baseline_f1"] + 1e-6 >= .90 and c["f1"] + 1e-6 < .90 for r, c in rows),
            "baseline_complete": sum(all(hit is not False for hit in r["baseline_hits"]) for r, _ in rows),
            "oracle_complete": sum(all(r["baseline_hits"][i] is not False for i in (0,1,2,4)) and c["f1"] + 1e-6 >= .90 for r,c in rows),
            "mean_selected_slack": sum(c["slack"] for _, c in rows) / len(rows) if rows else None,
            "above_depth10_cost_queries": sum(r["baseline_tokens"] + c["slack"] >= r["depth10_tokens"] for r, c in rows)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
