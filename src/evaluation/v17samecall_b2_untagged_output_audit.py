"""Capture raw old-prompt outputs for B0 missing-tag query/depth cases."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QampariTargetRunner

ROOT=Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0=Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
B0=Path("results/v2_rank_then_cut/v17samecall_b0_unstratified_prompt_gate")
OUT=Path("results/v2_rank_then_cut/v17samecall_b2_untagged_output_audit")


def main() -> None:
    from vllm import SamplingParams
    cfg=json.loads(Path("configs/v17samecall_b2_untagged_output_audit.json").read_text())
    assert cfg["status"]=="FROZEN_TARGETED_DESIGN_SIDE_DIAGNOSTIC"
    missing=sorted((r["example_id"],r["depth"]) for r in read_jsonl(B0/"per_call.jsonl")
                   if r["arm"]=="canonical_old_prompt" and not r["answer_tag_present"])
    assert len(missing)==23
    ids={q for q,d in missing}
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain=defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if (r["example_id"],r["depth"]) in missing:chain[r["example_id"]][r["depth"]]=r
    target=QampariTargetRunner(cfg["target_model"],backend="vllm",max_new_tokens=cfg["max_new_tokens"],
                                max_model_len=cfg["max_model_len"],gpu_memory_utilization=cfg["gpu_memory_utilization"])
    prompts=[]
    for q,depth in missing:
        context="\n\n".join(t.strip() for i,t in enumerate(data[q]["packet_texts"])
                             if chain[q][depth]["mask"]&(1<<i))
        prompts.extend(target._chat_prompts([data[q]["question"]],[context]))
    outputs=target.model.generate(prompts,SamplingParams(temperature=0,max_tokens=cfg["max_new_tokens"]),use_tqdm=False)
    rows=[]
    for (q,depth),out in zip(missing,outputs):
        raw=out.outputs[0].text
        parsed=parse_list_prediction(raw)
        rows.append({"example_id":q,"depth":depth,"raw_output":raw,"parsed":" # ".join(parsed),
                     "f1":float(qampari_list_metrics(parsed,atoms[q])["f1"]),
                     "finish_reason":out.outputs[0].finish_reason,
                     "generated_tokens":len(out.outputs[0].token_ids),
                     "has_answer_tag":"<answer>" in raw.casefold() and "</answer>" in raw.casefold()})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"per_call.jsonl").open("w") as handle:
        for r in rows:handle.write(json.dumps(r)+"\n")
    summary={"protocol":cfg["protocol"],"queries_or_states":23,
             "rerun_answer_tag_present":sum(r["has_answer_tag"] for r in rows),
             "rerun_090_success":sum(r["f1"]+1e-6>=.9 for r in rows),
             "finish_reason_counts":{k:sum(r["finish_reason"]==k for r in rows)
                                     for k in sorted({r["finish_reason"] for r in rows})},
             "new_target_calls":23,"sealed_outcome_sets_read":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
