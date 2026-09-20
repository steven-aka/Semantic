"""Same-batch paired replay of R1's 11 design-exposed oracle repairs."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
R1 = Path("results/v2_rank_then_cut/v17packet_r1_rank10_late_fragment_oracle")
OUT = Path("results/v2_rank_then_cut/v17packet_r1_repair_replay")


def main():
    oracle_rows = [r for r in read_jsonl(R1 / "per_query.jsonl")
                   if r["baseline"]["hits"][3] is False and r["strict_oracle"]["hits"][3] is True]
    assert len(oracle_rows) == 11 and all(r["strict_index"] >= 0 for r in oracle_rows)
    ids = {r["example_id"] for r in oracle_rows}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read_jsonl(path)} if path.exists() else {}
    tasks = []
    for row in oracle_rows:
        q = row["example_id"]
        order = orders[q]
        packet_id = order[9]
        fragment = segments(data[q]["packet_texts"][packet_id])[row["strict_index"]]
        moved = list(order)
        moved.insert(8, moved.pop(9))
        contexts = {"baseline": render(data[q]["packet_texts"], mask(order, 9)),
                    "fragment": render(data[q]["packet_texts"], mask(moved, 9), {packet_id: fragment})}
        for arm, context in contexts.items():
            if (q, arm) not in done:
                tasks.append({"example_id": q, "arm": arm, "context": context,
                              "question": data[q]["question"]})
    print(json.dumps({"paired_queries": len(oracle_rows), "new_calls": len(tasks)}), flush=True)
    if tasks:
        target = QampariTargetRunner("models/Qwen3-8B", backend="vllm", max_new_tokens=256,
                                     gpu_memory_utilization=0.6, max_model_len=4096)
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=256)
        prompts = target._chat_prompts([t["question"] for t in tasks], [t["context"] for t in tasks])
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
    details = [{"example_id": r["example_id"], "sentence_index": r["strict_index"],
                "baseline": done[r["example_id"], "baseline"],
                "fragment": done[r["example_id"], "fragment"]} for r in oracle_rows]
    with (OUT / "per_case.jsonl").open("w") as handle:
        for row in details:
            handle.write(json.dumps(row) + "\n")
    report = {"protocol": "V17-PACKET-R1_REPAIR_PAIRED_REPLAY", "queries": 11,
              "target_calls": len(done),
              "baseline_090_success": sum(x["baseline"]["f1"] + 1e-6 >= .90 for x in details),
              "fragment_090_success": sum(x["fragment"]["f1"] + 1e-6 >= .90 for x in details),
              "paired_repairs": sum(x["baseline"]["f1"] + 1e-6 < .90 <= x["fragment"]["f1"] + 1e-6
                                    for x in details),
              "fragment_no_extra_context": sum(x["fragment"]["tokens"] <= x["baseline"]["tokens"] for x in details),
              "prompt_tokens": sum(x["prompt_tokens"] for x in done.values()),
              "generated_tokens": sum(x["generated_tokens"] for x in done.values()),
              "limitations": "Only the 11 outcome-selected repairs were rerun. This tests reproducibility, not unselected generalization."}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
