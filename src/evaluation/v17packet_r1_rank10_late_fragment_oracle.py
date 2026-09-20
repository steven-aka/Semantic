"""Five-anchor train-side oracle for late rank-10 sentence promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.evaluation.v17packet_r0_targeted_atomicity_pilot import mask, render, segments
from src.representation.token_counter import count_tokens
from src.target.qampari_runner import QampariTargetRunner

CFG = Path("configs/v17packet_r1_rank10_late_fragment_oracle.json")
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
OUT = Path("results/v2_rank_then_cut/v17packet_r1_rank10_late_fragment_oracle")
LEVELS = (.60, .70, .80, .90, .95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--output-dir", default=str(OUT))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output_dir)
    assert cfg["status"] == "FROZEN_TRAIN_SIDE" and cfg["schedule"] == [6, 7, 7, 9, 10]
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "sel_c0_train_top10_candidates.jsonl")
              if fold(r["example_id"]) != 4}
    assert len(source) == 1421
    sorted_ids = sorted(source, key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    ids = sorted_ids[cfg["sample_offset"]:cfg["sample_offset"] + cfg["sample_size"]]
    assert len(ids) == cfg["sample_size"]
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(ROOT / "sel_c0_train_v8_rollouts.jsonl")
              if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    assert len(data) == len(orders) == len(atoms) == len(ids)
    cache = defaultdict(dict)
    paths = list(Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain").glob("shard_*_of_3/per_prefix.jsonl"))
    paths += [Path("results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl")]
    for path in paths:
        for row in read_jsonl(path):
            if row["example_id"] in data:
                cache[row["example_id"]][row["mask"]] = row
    assert all(len(cache[q]) >= 12 for q in ids)
    out.mkdir(parents=True, exist_ok=True)
    call_path = out / "per_sentence.jsonl"
    done = {(r["example_id"], r["sentence_index"]): r for r in read_jsonl(call_path)} if call_path.exists() else {}
    tasks = []
    manifest = []
    for q in ids:
        order = orders[q]
        packet_id = order[9]
        packet = data[q]["packet_texts"][packet_id]
        fragments = segments(packet)
        moved = list(order)
        moved.insert(8, moved.pop(9))
        d8 = mask(order, 8)
        d9 = mask(moved, 9)
        d10 = mask(order, 10)
        assert d9 == d8 | (1 << packet_id)
        assert d10 == d9 | (1 << order[8])
        assert mask(moved, 10) == d10
        original_full = "\n\n".join(t.strip() for t in data[q]["packet_texts"])
        assert render(data[q]["packet_texts"], mask(order, 12)) == original_full
        # The refined state at depth9 contains one complete title and one
        # original sentence. At depth10 the residual and displaced packet are
        # revealed, then canonical source-order rendering yields exact original
        # text, without repeating the sentence or title.
        title, sep, proof = packet.partition("\nSource evidence: ")
        if fragments:
            assert sep and " ".join(frag.split(sep, 1)[1] for frag in fragments) == proof
            assert all(frag.startswith(title + sep) for frag in fragments)
        assert render(data[q]["packet_texts"], d10) == render(data[q]["packet_texts"], mask(moved, 10))
        assert d9 in cache[q] and d10 in cache[q] and mask(order, 9) in cache[q]
        manifest.append({"example_id": q, "packet_id": packet_id, "sentence_count": len(fragments),
                         "full_context_sha256": hashlib.sha256(original_full.encode()).hexdigest(),
                         "depth9_baseline_mask": mask(order, 9), "depth9_full_move_mask": d9,
                         "depth10_restored_mask": d10})
        for i, fragment in enumerate(fragments):
            if (q, i) in done:
                continue
            context = render(data[q]["packet_texts"], d9, {packet_id: fragment})
            assert fragment in context and packet not in context
            tasks.append({"example_id": q, "sentence_index": i, "context": context,
                          "question": data[q]["question"]})
    (out / "manifest.json").write_text(json.dumps({"protocol": cfg["protocol"],
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "cases": manifest, "new_calls_total": len(tasks) + len(done)}, indent=2) + "\n")
    print(json.dumps({"queries": len(ids), "splittable": sum(r["sentence_count"]>0 for r in manifest),
                      "candidate_sentences": len(tasks) + len(done), "pending_target_calls": len(tasks)}), flush=True)
    if args.preflight_only:
        return
    if tasks:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm",
                                     max_new_tokens=cfg["max_new_tokens"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"],
                                     max_model_len=cfg["max_model_len"])
        from vllm import SamplingParams
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_new_tokens"])
        with call_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(tasks), cfg["batch_size"]):
                batch = tasks[start:start + cfg["batch_size"]]
                prompts = target._chat_prompts([r["question"] for r in batch], [r["context"] for r in batch])
                outputs = target.model.generate(prompts, params, use_tqdm=False)
                for task, prompt, output in zip(batch, prompts, outputs):
                    parsed = parse_list_prediction(output.outputs[0].text)
                    row = {"example_id": task["example_id"], "sentence_index": task["sentence_index"],
                           "f1": float(qampari_list_metrics(parsed, atoms[task["example_id"]])["f1"]),
                           "tokens": count_tokens(target.tokenizer, task["context"]),
                           "prompt_tokens": len(target.tokenizer.encode(prompt)),
                           "generated_tokens": len(output.outputs[0].token_ids),
                           "prediction": " # ".join(parsed)}
                    done[row["example_id"], row["sentence_index"]] = row
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"completed_new_calls": min(start+len(batch), len(tasks)),
                                  "total_new_calls": len(tasks)}), flush=True)
    def evaluate(q, depth9row):
        active = {float(x) for x in source[q]["attainable_levels"]}
        order = orders[q]
        stop_rows = [cache[q][mask(order,d)] if d != 9 else depth9row for d in cfg["schedule"]]
        hits = [None if level not in active else row["f1"] + cfg["fidelity_epsilon"] >= level
                for level,row in zip(LEVELS,stop_rows)]
        return {"hits": hits, "complete": all(h is not False for h in hits),
                "context_tokens": sum(row["tokens"] for level,row in zip(LEVELS,stop_rows) if level in active)}
    details=[]
    for item in manifest:
        q=item["example_id"]
        base=evaluate(q,cache[q][item["depth9_baseline_mask"]])
        full=evaluate(q,cache[q][item["depth9_full_move_mask"]])
        candidates=[(i,evaluate(q,done[q,i])) for i in range(item["sentence_count"])]
        safe=[(i,row) for i,row in candidates if not any(x is True and y is False for x,y in zip(base["hits"],row["hits"]))]
        cheap=[(i,row) for i,row in safe if row["context_tokens"]<=base["context_tokens"]]
        choose=lambda pool: min(pool,key=lambda pair:(-
            sum(x is False and y is True for x,y in zip(base["hits"],pair[1]["hits"])),
            pair[1]["context_tokens"],pair[0]))
        strict_index,strict=choose([(-1,base)]+cheap)
        quality_index,quality=choose([(-1,base)]+safe)
        details.append({"example_id":q,"baseline":base,"full_move":full,
                        "candidate_sentences":len(candidates),"sentence_options":[{"index":i,**row} for i,row in candidates],
                        "strict_index":strict_index,"strict_oracle":strict,
                        "quality_index":quality_index,"quality_oracle":quality})
    with (out / "per_query.jsonl").open("w") as handle:
        for row in details:handle.write(json.dumps(row)+"\n")
    def summarize(key):
        rows=[x[key] for x in details]
        return {"success":{str(level):sum(x["hits"][i] is True for x in rows)
                           for i,level in enumerate(LEVELS)},
                "complete":sum(x["complete"] for x in rows),
                "mean_cumulative_context_tokens":sum(x["context_tokens"] for x in rows)/len(rows)}
    all_calls=list(read_jsonl(call_path)) if call_path.exists() else []
    report={"protocol":cfg["protocol"],"queries":len(ids),
            "splittable":sum(x["sentence_count"]>0 for x in manifest),
            "fresh_target_calls":len(all_calls),
            "fresh_prompt_tokens":sum(x["prompt_tokens"] for x in all_calls),
            "fresh_generated_tokens":sum(x["generated_tokens"] for x in all_calls),
            "baseline":summarize("baseline"),"full_move":summarize("full_move"),
            "strict_no_extra_context_oracle":summarize("strict_oracle"),
            "quality_first_oracle":summarize("quality_oracle"),
            "strict_changed_queries":sum(x["strict_index"]>=0 for x in details),
            "quality_changed_queries":sum(x["quality_index"]>=0 for x in details),
            "limitations":"Outcome-aware sentence selection on train-side sample; no deployable policy or held-out model generalization."}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
