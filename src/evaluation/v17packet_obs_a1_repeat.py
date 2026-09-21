"""Matched repeat of natural protected-insertion outcomes, by frozen strata."""
import hashlib
import json
import os
from pathlib import Path

from src.evaluation.v17packet_obs_a1_natural_insertion import CFG, OUT, prepare, read


def main():
    cfg = json.loads(CFG.read_text())
    original = list(read(OUT / "per_query.jsonl"))
    groups = {
        "repair": [r for r in original if r["repair"]],
        "break": [r for r in original if r["break"]],
        "neutral": [r for r in original if not r["repair"] and not r["break"]],
    }
    selected = []
    for key in ("repair", "break", "neutral"):
        choice = sorted(groups[key], key=lambda r: hashlib.sha256(r["example_id"].encode()).hexdigest())[:10]
        assert len(choice) == 10
        selected.extend(choice)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    prepared, _ = prepare(cfg, tokenizer)
    indexed = {r["example_id"]: r for r in prepared}
    tasks = []
    for row in selected:
        source = indexed[row["example_id"]]
        for arm in ("baseline", "action"):
            tasks.append({"example_id": row["example_id"], "arm": arm,
                          "question": source["question"], "context": source[f"{arm}_context"]})
    manifest = {"protocol": "OBS-A1_NATURAL_INSERTION_CONDITIONAL_REPEAT",
                "query_ids": [r["example_id"] for r in selected], "planned_target_calls": len(tasks),
                "task_sha256": hashlib.sha256("\n".join(
                    f'{t["example_id"]}:{t["arm"]}:{hashlib.sha256(t["context"].encode()).hexdigest()}' for t in tasks
                ).encode()).hexdigest(), "selection": "SHA-first 10 each of original repair/break/neutral",
                "sealed_sets_read": False}
    path = OUT / "repeat_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
    from src.target.qampari_runner import QampariTargetRunner
    from vllm import SamplingParams
    ids = set(manifest["query_ids"])
    atoms = {r["example_id"]: r["answer_atoms"] for r in read("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    call_path = OUT / "repeat_per_call.jsonl"
    done = {(r["example_id"], r["arm"]): r for r in read(call_path)} if call_path.exists() else {}
    pending = [t for t in tasks if (t["example_id"], t["arm"]) not in done]
    if pending:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"], max_model_len=cfg["max_model_len"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with call_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), cfg["batch_size"]):
                batch = pending[start:start + cfg["batch_size"]]
                outputs = target.model.generate(target._chat_prompts([t["question"] for t in batch],
                                               [t["context"] for t in batch]), params, use_tqdm=False)
                for task, output in zip(batch, outputs):
                    prediction = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "arm": task["arm"],
                           "f1": float(qampari_list_metrics(prediction, atoms[task["example_id"]])["f1"]),
                           "raw_output": output.outputs[0].text,
                           "generated_token_ids": list(output.outputs[0].token_ids)}
                    done[row["example_id"], row["arm"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(json.dumps({"completed": len(done), "total": len(tasks)}), flush=True)
    assert len(done) == len(tasks)
    by_stratum = {}
    for label in ("repair", "break", "neutral"):
        subset = [r for r in selected if ("repair" if r["repair"] else "break" if r["break"] else "neutral") == label]
        agreement = 0
        for row in subset:
            q = row["example_id"]
            before = done[q, "baseline"]["f1"] + 1e-6 >= .90
            after = done[q, "action"]["f1"] + 1e-6 >= .90
            repeated = "repair" if not before and after else "break" if before and not after else "neutral"
            agreement += repeated == label
        by_stratum[label] = {"cases": len(subset), "repeat_category_agreement": agreement}
    summary = {"protocol": manifest["protocol"], "queries": len(selected), "calls": len(done),
               "by_original_stratum": by_stratum, "sealed_sets_read": False,
               "limitation": "Outcome-stratified repeat checks conditional stability, not population noise or independent validation."}
    (OUT / "repeat_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
