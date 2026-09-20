"""Paired rerun of the nine design-exposed no-extra-context 0.90 fragments."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

OUT = Path("results/v2_rank_then_cut/v17packet_r0_paired_replay")
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")


def main():
    cases = list(read_jsonl("results/v2_rank_then_cut/v17packet_r0_targeted_atomicity_pilot/per_case.jsonl"))
    chosen = []
    for case in cases:
        if case["group"] != "repair_090":
            continue
        arms = case["arms"]
        candidates = [(name, row) for name, row in arms.items()
                      if name.startswith("full_move_sentence_") and row["f1"] + 1e-6 >= .90
                      and row["tokens"] <= arms["baseline"]["tokens"]]
        if candidates:
            name, _ = min(candidates, key=lambda x: (x[1]["tokens"], x[0]))
            chosen.append((case, int(name.rsplit("_", 1)[1])))
    assert len(chosen) == 9 and all(c["decision_depth"] == 6 and c["rank"] == 10 for c, _ in chosen)
    ids = {c["example_id"] for c, _ in chosen}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(atoms) == 9
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(path)} if path.exists() else {}
    tasks = []
    for case, sentence_index in chosen:
        q = case["example_id"]
        order = orders[q]
        target_packet = order[9]
        moved = list(order)
        moved.insert(6, moved.pop(9))
        fragment = segments(data[q]["packet_texts"][target_packet])[sentence_index]
        contexts = {
            "depth7_fragment": render(data[q]["packet_texts"], mask(moved, 7), {target_packet: fragment}),
            "depth9_baseline": render(data[q]["packet_texts"], mask(order, 9)),
            "depth9_full_move": render(data[q]["packet_texts"], mask(moved, 9)),
            "depth9_fragment": render(data[q]["packet_texts"], mask(moved, 9), {target_packet: fragment}),
        }
        for name, context in contexts.items():
            if (q, name) not in done:
                tasks.append({"example_id": q, "arm": name, "question": data[q]["question"],
                              "context": context})
    print(json.dumps({"selected_queries": len(chosen), "pending_calls": len(tasks)}), flush=True)
    if tasks:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=0.6, max_model_len=4096)
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=256)
        prompts = target._chat_prompts([r["question"] for r in tasks], [r["context"] for r in tasks])
        outputs = target.model.generate(prompts, params, use_tqdm=False)
        with path.open("a", encoding="utf-8") as handle:
            for task, prompt, output in zip(tasks, prompts, outputs):
                parsed = parse_list_prediction(output.outputs[0].text)
                row = {"example_id": task["example_id"], "arm": task["arm"],
                       "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                       "tokens": count_tokens(target.tokenizer, task["context"]),
                       "prompt_tokens": len(target.tokenizer.encode(prompt)),
                       "generated_tokens": len(output.outputs[0].token_ids),
                       "prediction": " # ".join(parsed)}
                handle.write(json.dumps(row) + "\n")
                done[row["example_id"], row["arm"]] = row
    details = []
    for case, index in chosen:
        q = case["example_id"]
        rows = {arm: done[q, arm] for arm in
                ("depth7_fragment", "depth9_baseline", "depth9_full_move", "depth9_fragment")}
        details.append({"example_id": q, "sentence_index": index, "rows": rows})
    with (OUT / "per_case.jsonl").open("w") as handle:
        for item in details:
            handle.write(json.dumps(item) + "\n")
    report = {"protocol": "V17-PACKET-R0_PAIRED_REPLAY", "queries": 9,
              "target_calls": len(done),
              "prompt_tokens": sum(r["prompt_tokens"] for r in done.values()),
              "generated_tokens": sum(r["generated_tokens"] for r in done.values()),
              "depth9_090_success": {arm: sum(x["rows"][arm]["f1"] + 1e-6 >= .90 for x in details)
                                     for arm in ("depth9_baseline", "depth9_full_move", "depth9_fragment")},
              "depth7_fragment_080_success": sum(x["rows"]["depth7_fragment"]["f1"] + 1e-6 >= .80 for x in details),
              "fragment_no_extra_depth9_context": sum(x["rows"]["depth9_fragment"]["tokens"] <=
                  x["rows"]["depth9_baseline"]["tokens"] for x in details),
              "limitations": "These 9 were selected by earlier Target outcomes. Paired replay is a robustness diagnostic, not an independent generalization estimate or five-anchor Pareto proof."}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
