"""New-prompt repeated Target self-report pilot on train-only prefixes."""
from __future__ import annotations

import hashlib
import json
import re
import time
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17stop_c0_prefix_output_retention import normalized_map, rules
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, QampariTargetRunner, build_qampari_prompt

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT = Path("results/v2_rank_then_cut/v17samecall_a0_sufficiency_pilot")
CONTINUE = "depth10_plus_old_context_supported_depth9"
CLASSES = ("SS", "SF", "FS", "FF")


def stable_key(q: str) -> str:
    return hashlib.sha256(("SAMECALL-A0|" + q).encode()).hexdigest()


def main() -> None:
    from vllm import SamplingParams
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path("configs/v17samecall_a0_sufficiency_pilot.json").read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_PILOT" and cfg["repeat_count"] == 2
    c0 = list(read_jsonl(C0 / "per_query.jsonl"))
    by_class = defaultdict(list)
    for r in c0:
        a = r["outputs"]["depth9_only"]["success"]["0.9"]
        b = r["outputs"][CONTINUE]["success"]["0.9"]
        by_class[("S" if a else "F") + ("S" if b else "F")].append(r)
    chosen = [r for name in CLASSES for r in sorted(by_class[name], key=lambda x: stable_key(x["example_id"]))
              [:17 if name == "SF" else 37]]
    assert len(chosen) == 128 and len({r["example_id"] for r in chosen}) == 128
    ids = {r["example_id"] for r in chosen}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in ids and r["depth"] in (9, 10):
                chain[r["example_id"]][r["depth"]] = r
    assert len(data) == len(atoms) == len(chain) == 128 and all(len(x) == 2 for x in chain.values())
    if args.analyze_only:
        from transformers import AutoTokenizer
        target = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(cfg["target_model"]))
    else:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
    status_re = re.compile(cfg["status_regex"], flags=re.IGNORECASE)
    tasks = []
    for repeat in range(cfg["repeat_count"]):
        for depth in cfg["depths"]:
            for r in chosen:
                q = r["example_id"]
                state = chain[q][depth]
                context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"])
                                        if state["mask"] & (1 << i))
                messages = [
                    {"role": "system", "content": QAMPARI_SYSTEM_PROMPT + " " + cfg["prompt_suffix_system"]},
                    {"role": "user", "content": build_qampari_prompt(data[q]["question"], context) +
                     "\n" + cfg["prompt_suffix_user"]},
                ]
                prompt = target.tokenizer.apply_chat_template(messages, tokenize=False,
                                                               add_generation_prompt=True, enable_thinking=False)
                tasks.append({"example_id": q, "repeat": repeat, "depth": depth, "prompt": prompt,
                              "context": context, "historical_class": next(name for name in CLASSES if r in by_class[name])})
    assert len(tasks) == 512
    params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_call.jsonl"
    completed = {}
    if path.exists():
        for row in read_jsonl(path):
            key = (row["example_id"], row["repeat"], row["depth"])
            if key in completed:
                raise AssertionError("duplicate resume row")
            completed[key] = row
    if args.analyze_only and len(completed) != len(tasks):
        raise AssertionError("analysis-only requires all 512 cached calls")
    started = time.perf_counter()
    with path.open("a") as handle:
        for offset in range(0, len(tasks), cfg["batch_size"]):
            batch = [t for t in tasks[offset:offset + cfg["batch_size"]]
                     if (t["example_id"], t["repeat"], t["depth"]) not in completed]
            if not batch:
                continue
            outputs = target.model.generate([t["prompt"] for t in batch], params, use_tqdm=False)
            for task, result in zip(batch, outputs):
                q = task["example_id"]
                raw = result.outputs[0].text
                tagged = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
                match = status_re.search(raw)
                parsed = parse_list_prediction(tagged.group(0)) if tagged else []
                f1 = float(qampari_list_metrics(parsed, atoms[q])["f1"])
                row = {"example_id": q, "repeat": task["repeat"], "depth": task["depth"],
                       "historical_class": task["historical_class"],
                       "raw_output": raw, "answer_tag_present": tagged is not None,
                       "answer": " # ".join(parsed), "status": match.group(1).upper() if match else None,
                       "f1": f1, "success_090": f1 + 1e-6 >= .9,
                       "prompt_tokens": len(target.tokenizer.encode(task["prompt"])),
                       "generated_tokens": len(result.outputs[0].token_ids)}
                handle.write(json.dumps(row) + "\n")
                completed[(q, task["repeat"], task["depth"])] = row
            handle.flush()
            print(json.dumps({"completed": len(completed), "total": len(tasks)}), flush=True)
    assert len(completed) == 512
    by_rep = defaultdict(dict)
    for row in completed.values():
        by_rep[(row["example_id"], row["repeat"])][row["depth"]] = row
    replay = []
    for r in chosen:
        q = r["example_id"]
        old_context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"])
                                  if chain[q][9]["mask"] & (1 << i))
        new_packet = (chain[q][9]["mask"] ^ chain[q][10]["mask"]).bit_length() - 1
        for repeat in range(2):
            a, b = by_rep[(q, repeat)][9], by_rep[(q, repeat)][10]
            kept = rules(normalized_map(a["answer"]), normalized_map(b["answer"]),
                         old_context, data[q]["packet_texts"][new_packet])[CONTINUE]
            retained_f1 = float(qampari_list_metrics(kept, atoms[q])["f1"])
            stop = a["status"] == "YES"
            replay.append({"example_id": q, "repeat": repeat, "historical_class": a["historical_class"],
                           "stop": stop, "status_valid": a["status"] is not None,
                           "current_success": a["success_090"],
                           "retained_continue_success": retained_f1 + 1e-6 >= .9,
                           "direct_depth10_success": b["success_090"],
                           "policy_success": a["success_090"] if stop else retained_f1 + 1e-6 >= .9,
                           "target_tokens_policy": a["prompt_tokens"] + a["generated_tokens"] +
                               (0 if stop else b["prompt_tokens"] + b["generated_tokens"]),
                           "target_tokens_direct_depth10": b["prompt_tokens"] + b["generated_tokens"]})
    paired = defaultdict(dict)
    for x in completed.values():
        paired[(x["example_id"], x["depth"])][x["repeat"]] = x
    stats = {"protocol": cfg["protocol"], "queries": 128, "calls": 512,
             "historical_class_counts": {n: sum(t in by_class[n] for t in chosen) for n in CLASSES},
             "answer_tag_fraction": sum(x["answer_tag_present"] for x in completed.values()) / 512,
             "status_valid_fraction": sum(x["status"] in ("YES", "NO") for x in completed.values()) / 512,
             "repeat_status_agreement_depth9": sum(p[0]["status"] == p[1]["status"] for (q,d),p in paired.items() if d == 9) / 128,
             "repeat_answer_agreement_depth9": sum(p[0]["answer"] == p[1]["answer"] for (q,d),p in paired.items() if d == 9) / 128,
             "repeat_090_label_agreement_depth9": sum(p[0]["success_090"] == p[1]["success_090"] for (q,d),p in paired.items() if d == 9) / 128,
             "repeat_answer_agreement_depth10": sum(p[0]["answer"] == p[1]["answer"] for (q,d),p in paired.items() if d == 10) / 128,
             "repeat_090_label_agreement_depth10": sum(p[0]["success_090"] == p[1]["success_090"] for (q,d),p in paired.items() if d == 10) / 128,
             "elapsed_seconds_excluding_model_init": None if args.analyze_only else time.perf_counter() - started,
             "analyzed_from_cached_calls": bool(args.analyze_only),
             "repeat_metrics": {}}
    for repeat in (0, 1):
        rr = [x for x in replay if x["repeat"] == repeat]
        calls = [x for x in completed.values() if x["repeat"] == repeat]
        stats["repeat_metrics"][str(repeat)] = {
            "new_prompt_depth9_success_090": sum(x["current_success"] for x in rr),
            "new_prompt_depth10_success_090": sum(x["direct_depth10_success"] for x in rr),
            "selfreport_policy_success_090": sum(x["policy_success"] for x in rr),
            "selfreport_stop_count": sum(x["stop"] for x in rr),
            "selfreport_current_success_true_yes": sum(x["stop"] and x["current_success"] for x in rr),
            "selfreport_current_failure_false_yes": sum(x["stop"] and not x["current_success"] for x in rr),
            "mean_target_tokens_policy": sum(x["target_tokens_policy"] for x in rr) / 128,
            "mean_target_tokens_direct_depth10": sum(x["target_tokens_direct_depth10"] for x in rr) / 128,
            "mean_extra_output_tokens_vs_historical_depth9": sum(
                x["generated_tokens"] - chain[x["example_id"]][9]["generated_tokens"]
                for x in calls if x["depth"] == 9) / 128,
        }
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    with (OUT / "replay.jsonl").open("w") as handle:
        for row in replay:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
