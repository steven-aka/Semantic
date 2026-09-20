"""Isolate addition, displacement and sentence-granularity effects in early moves."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

CFG = Path("configs/v17packet_r0_targeted_atomicity_pilot.json")
OUT = Path("results/v2_rank_then_cut/v17packet_r0_targeted_atomicity_pilot")
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
STATE = Path("results/v2_rank_then_cut/v17state_a0_early_move_expand256")


def mask(order, depth):
    return sum(1 << int(p) for p in order[:depth])


def render(packet_texts, mask_value, override=None):
    override = override or {}
    return "\n\n".join((override.get(i, text)).strip() for i, text in enumerate(packet_texts)
                       if mask_value & (1 << i))


def segments(packet):
    title, separator, proof = packet.partition("\nSource evidence: ")
    if not separator or not title.startswith("Document title: "):
        return [packet]
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", proof) if x.strip()]
    assert " ".join(sentences) == proof
    if len(sentences) < 2:
        return []
    return [title + separator + sentence for sentence in sentences]


def main():
    cfg = json.loads(CFG.read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE"
    rows = list(read_jsonl(STATE / "per_query.jsonl"))[64:]
    assert len(rows) == 192
    for x in rows:
        assert fold(x["example_id"]) != 4
    repairs90 = [x for x in rows if x["baseline"]["hits"][3] is False and
                 x["quality_first_oracle"]["hits"][3] is True]
    repairs80 = [x for x in rows if x["baseline"]["hits"][2] is False and
                 x["quality_first_oracle"]["hits"][2] is True and x not in repairs90]
    controls = [x for x in rows if x not in repairs90 and x not in repairs80]
    key = lambda x: hashlib.sha256(x["example_id"].encode()).hexdigest()
    selected = [("repair_090", x, x["quality_first_oracle"]["depth"],
                 x["quality_first_oracle"]["rank"], .90, 9) for x in repairs90]
    selected += [("repair_080_only", x, x["quality_first_oracle"]["depth"],
                  x["quality_first_oracle"]["rank"], .80, 7)
                 for x in sorted(repairs80, key=key)[:15]]
    selected += [("control", x, 6, 10, .90, 9) for x in sorted(controls, key=key)[:15]]
    assert len(repairs90) == 15 and len(selected) == 45
    ids = {x["example_id"] for _, x, *_ in selected}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl")
              if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(atoms) == 45
    cache = defaultdict(dict)
    paths = list(Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain").glob("shard_*_of_3/per_prefix.jsonl"))
    paths += [Path("results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl"),
              Path("results/v2_rank_then_cut/v17state_a0_early_move_pilot/per_state.jsonl"),
              STATE / "per_state.jsonl"]
    for path in paths:
        for row in read_jsonl(path):
            if row["example_id"] in ids:
                cache[row["example_id"]][row["mask"]] = row
    OUT.mkdir(parents=True, exist_ok=True)
    per_call = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(per_call)} if per_call.exists() else {}
    tasks = []
    meta = []
    for group, original, depth, rank, level, stop in selected:
        q = original["example_id"]
        order = orders[q]
        target_packet_id = order[rank - 1]
        original_mask = mask(order, stop)
        shifted = list(order)
        shifted.insert(depth, shifted.pop(rank - 1))
        moved_mask = mask(shifted, stop)
        assert not original_mask & (1 << target_packet_id)
        assert moved_mask & (1 << target_packet_id)
        packet = data[q]["packet_texts"][target_packet_id]
        arms = {"baseline": render(data[q]["packet_texts"], original_mask),
                "full_move": render(data[q]["packet_texts"], moved_mask),
                "baseline_plus_full": render(data[q]["packet_texts"], original_mask | (1 << target_packet_id))}
        for i, fragment in enumerate(segments(packet)):
            arms[f"baseline_plus_sentence_{i}"] = render(data[q]["packet_texts"],
                original_mask | (1 << target_packet_id), {target_packet_id: fragment})
            arms[f"full_move_sentence_{i}"] = render(data[q]["packet_texts"],
                moved_mask, {target_packet_id: fragment})
        meta.append({"group": group, "example_id": q, "decision_depth": depth,
                     "rank": rank, "stop_depth": stop, "level": level,
                     "target_packet_id": target_packet_id, "packet_chars": len(packet),
                     "sentence_count": len(segments(packet)), "arms": list(arms)})
        for name, context in arms.items():
            if name in ("baseline", "full_move", "baseline_plus_full"):
                state_mask = {"baseline": original_mask, "full_move": moved_mask,
                              "baseline_plus_full": original_mask | (1 << target_packet_id)}[name]
                if state_mask in cache[q]:
                    row = cache[q][state_mask]
                    done[q, name] = {"example_id": q, "arm": name, "f1": row["f1"],
                                     "tokens": row["tokens"], "cached": True}
                    continue
            if (q, name) not in done:
                tasks.append({"example_id": q, "arm": name,
                              "question": data[q]["question"], "context": context})
    print(json.dumps({"cases": len(meta), "new_calls": len(tasks),
                      "cached_or_completed": len(done)}), flush=True)
    if tasks:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with per_call.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"],
                           "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                           "tokens": count_tokens(target.tokenizer, task["context"]),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids),
                           "prediction": " # ".join(parsed), "cached": False}
                    done[task["example_id"], task["arm"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"completed_new_calls": min(start + len(batch), len(tasks)),
                                  "total_new_calls": len(tasks)}), flush=True)
    detail = []
    for item in meta:
        q = item["example_id"]
        arm_rows = {name: done[q, name] for name in item["arms"]}
        base = arm_rows["baseline"]
        full = arm_rows["full_move"]
        level = item["level"]
        good = lambda row: row["f1"] + cfg["fidelity_epsilon"] >= level
        sentence_arms = [name for name in arm_rows if "sentence_" in name]
        successful = [name for name in sentence_arms if good(arm_rows[name])]
        cheapest = min(successful, key=lambda name: (arm_rows[name]["tokens"], name)) if successful else None
        detail.append({**item, "baseline_good": good(base), "full_move_good": good(full),
                       "arms": arm_rows, "cheapest_successful_fragment_arm": cheapest})
    with (OUT / "per_case.jsonl").open("w") as handle:
        for item in detail:
            handle.write(json.dumps(item) + "\n")
    fresh = list(read_jsonl(per_call)) if per_call.exists() else []
    report = {"protocol": cfg["protocol"], "cases": len(detail),
              "groups": {group: sum(x["group"] == group for x in detail)
                         for group in ("repair_090", "repair_080_only", "control")},
              "new_target_calls": len(fresh),
              "new_prompt_tokens": sum(x["prompt_tokens"] for x in fresh),
              "new_generated_tokens": sum(x["generated_tokens"] for x in fresh),
              "repair_090": {}, "repair_080_only": {},
              "limitations": cfg["scope"]}
    for group in ("repair_090", "repair_080_only"):
        members = [x for x in detail if x["group"] == group]
        report[group] = {
            "full_move_success": sum(x["full_move_good"] for x in members),
            "baseline_plus_full_success": sum(x["arms"]["baseline_plus_full"]["f1"] >= x["level"] - cfg["fidelity_epsilon"] for x in members),
            "any_fragment_success": sum(x["cheapest_successful_fragment_arm"] is not None for x in members),
            "fragment_success_no_extra_context": sum(x["cheapest_successful_fragment_arm"] is not None and
                x["arms"][x["cheapest_successful_fragment_arm"]]["tokens"] <= x["arms"]["baseline"]["tokens"] for x in members),
            "fragment_success_cheaper_than_full_move": sum(x["cheapest_successful_fragment_arm"] is not None and
                x["arms"][x["cheapest_successful_fragment_arm"]]["tokens"] < x["arms"]["full_move"]["tokens"] for x in members)
        }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
