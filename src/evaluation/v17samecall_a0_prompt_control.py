"""Same-sample canonical-prompt rerun to isolate new-prompt answer changes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
NEW = Path("results/v2_rank_then_cut/v17samecall_a0_sufficiency_pilot")
OUT = Path("results/v2_rank_then_cut/v17samecall_a0_prompt_control")


def main() -> None:
    from vllm import SamplingParams

    cfg = json.loads(Path("configs/v17samecall_a0_prompt_control.json").read_text())
    assert cfg["status"] == "FROZEN_TRAIN_SIDE_CONTROL"
    new = {(r["example_id"], r["depth"]): r for r in read_jsonl(NEW / "per_call.jsonl") if r["repeat"] == 0}
    ids = sorted({q for q, _ in new})
    assert len(ids) == 128 and len(new) == 256
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    cache = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in data and r["depth"] in (9, 10):
                cache[r["example_id"]][r["depth"]] = r
    assert len(data) == len(atoms) == len(cache) == 128
    target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                 max_model_len=cfg["max_model_len"],
                                 gpu_memory_utilization=cfg["gpu_memory_utilization"])
    tasks = []
    for depth in (9, 10):
        for q in ids:
            context = "\n\n".join(t.strip() for i, t in enumerate(data[q]["packet_texts"])
                                    if cache[q][depth]["mask"] & (1 << i))
            prompt = target._chat_prompts([data[q]["question"]], [context])[0]
            tasks.append((q, depth, prompt))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_call.jsonl"
    done = {(r["example_id"], r["depth"]): r for r in read_jsonl(path)} if path.exists() else {}
    params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
    with path.open("a") as handle:
        for offset in range(0, len(tasks), cfg["batch_size"]):
            batch = [(q,d,p) for q,d,p in tasks[offset:offset+cfg["batch_size"]] if (q,d) not in done]
            if not batch:
                continue
            outputs = target.model.generate([x[2] for x in batch], params, use_tqdm=False)
            for (q, depth, prompt), output in zip(batch, outputs):
                answer = parse_list_prediction(output.outputs[0].text)
                row = {"example_id": q, "depth": depth, "answer": " # ".join(answer),
                       "f1": float(qampari_list_metrics(answer, atoms[q])["f1"]),
                       "prompt_tokens": len(target.tokenizer.encode(prompt)),
                       "generated_tokens": len(output.outputs[0].token_ids)}
                done[(q,depth)] = row
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == 256
    report = {"protocol": cfg["protocol"], "queries": 128, "target_calls": 256,
              "depths": {}, "sealed_outcome_sets_read": False}
    for depth in (9,10):
        old = [done[(q,depth)] for q in ids]
        fresh = [new[(q,depth)] for q in ids]
        historical = [cache[q][depth] for q in ids]
        report["depths"][str(depth)] = {
            "old_prompt_rerun_090_success": sum(r["f1"] + 1e-6 >= .9 for r in old),
            "new_prompt_repeat0_090_success": sum(r["success_090"] for r in fresh),
            "historical_cache_090_success": sum(r["f1"] + 1e-6 >= .9 for r in historical),
            "new_vs_old_repairs": sum(n["success_090"] and o["f1"] + 1e-6 < .9 for n,o in zip(fresh,old)),
            "new_vs_old_breaks": sum(not n["success_090"] and o["f1"] + 1e-6 >= .9 for n,o in zip(fresh,old)),
            "old_rerun_vs_cache_answer_agreement": sum(o["answer"] == h["prediction"] for o,h in zip(old,historical)),
            "mean_old_prompt_target_tokens": sum(o["prompt_tokens"]+o["generated_tokens"] for o in old)/128,
            "mean_new_prompt_target_tokens": sum(n["prompt_tokens"]+n["generated_tokens"] for n in fresh)/128}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
