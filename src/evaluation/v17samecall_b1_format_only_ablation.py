"""One-factor format-only prompt ablation on B0's design-exposed IDs."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, QampariTargetRunner, build_qampari_prompt

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
B0 = Path("results/v2_rank_then_cut/v17samecall_b0_unstratified_prompt_gate")
OUT = Path("results/v2_rank_then_cut/v17samecall_b1_format_only_ablation")
LEVELS = (.60,.70,.80,.90,.95)


def main() -> None:
    from vllm import SamplingParams
    cfg=json.loads(Path("configs/v17samecall_b1_format_only_ablation.json").read_text())
    assert cfg["status"]=="FROZEN_MECHANISM_ABLATION_TRAIN_SIDE"
    old={(r["example_id"],r["depth"],r["arm"]):r for r in read_jsonl(B0/"per_call.jsonl")}
    ids=sorted({q for q,_,_ in old})
    assert len(ids)==128 and len(old)==512
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain=defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in data and r["depth"] in (9,10):chain[r["example_id"]][r["depth"]]=r
    assert len(data)==len(atoms)==len(chain)==128
    target=QampariTargetRunner(cfg["target_model"],backend="vllm",max_new_tokens=cfg["max_new_tokens"],
                                max_model_len=cfg["max_model_len"],gpu_memory_utilization=cfg["gpu_memory_utilization"])
    tasks=[]
    for depth in (9,10):
        for q in ids:
            context="\n\n".join(t.strip() for i,t in enumerate(data[q]["packet_texts"])
                                 if chain[q][depth]["mask"]&(1<<i))
            messages=[{"role":"system","content":QAMPARI_SYSTEM_PROMPT+" "+cfg["system_suffix"]},
                      {"role":"user","content":build_qampari_prompt(data[q]["question"],context)+"\n"+cfg["user_suffix"]}]
            prompt=target.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            tasks.append((q,depth,prompt))
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"per_call.jsonl"
    done={(r["example_id"],r["depth"]):r for r in read_jsonl(path)} if path.exists() else {}
    params=SamplingParams(temperature=cfg["temperature"],max_tokens=cfg["max_new_tokens"])
    with path.open("a") as handle:
        for offset in range(0,len(tasks),cfg["batch_size"]):
            batch=[x for x in tasks[offset:offset+cfg["batch_size"]] if (x[0],x[1]) not in done]
            if not batch:continue
            outputs=target.model.generate([x[2] for x in batch],params,use_tqdm=False)
            for (q,depth,prompt),out in zip(batch,outputs):
                raw=out.outputs[0].text
                tag=bool(re.search(r"<answer>\s*(.*?)\s*</answer>",raw,re.IGNORECASE|re.DOTALL))
                answers=parse_list_prediction(raw)
                metric=qampari_list_metrics(answers,atoms[q])
                row={"example_id":q,"depth":depth,"answer":" # ".join(answers),"raw_output":raw,
                     "answer_tag_present":tag,"f1":float(metric["f1"]),"predicted":int(metric["predicted"]),
                     "correct":int(metric["correct"]),"prompt_tokens":len(target.tokenizer.encode(prompt)),
                     "generated_tokens":len(out.outputs[0].token_ids)}
                done[(q,depth)]=row
                handle.write(json.dumps(row)+"\n")
            handle.flush()
            print(json.dumps({"completed":len(done),"total":256}),flush=True)
    assert len(done)==256
    report={"protocol":cfg["protocol"],"queries":128,"target_calls":256,"depths":{},"sealed_outcome_sets_read":False}
    for depth in (9,10):
        formatrows=[done[q,depth] for q in ids]
        oldrows=[old[q,depth,"canonical_old_prompt"] for q in ids]
        newrows=[old[q,depth,"samecall_a0_new_prompt"] for q in ids]
        anchors={}
        for level in LEVELS:
            idx=[i for i,q in enumerate(ids) if level in {float(z) for z in data[q]["attainable_levels"]}]
            anchors[str(level)]={"eligible":len(idx),
                "old_success":sum(oldrows[i]["f1"]+1e-6>=level for i in idx),
                "format_success":sum(formatrows[i]["f1"]+1e-6>=level for i in idx),
                "selfreport_prompt_success":sum(newrows[i]["f1"]+1e-6>=level for i in idx)}
        tagged=[i for i,r in enumerate(oldrows) if r["answer_tag_present"]]
        missing=[i for i,r in enumerate(oldrows) if not r["answer_tag_present"]]
        report["depths"][str(depth)]={"anchors":anchors,
            "old_tag_present":len(tagged),"format_tag_present":sum(r["answer_tag_present"] for r in formatrows),
            "selfreport_tag_present":sum(r["answer_tag_present"] for r in newrows),
            "old_tagged_090_success":sum(oldrows[i]["f1"]+1e-6>=.9 for i in tagged),
            "format_on_old_tagged_090_success":sum(formatrows[i]["f1"]+1e-6>=.9 for i in tagged),
            "format_on_old_untagged_090_success":sum(formatrows[i]["f1"]+1e-6>=.9 for i in missing),
            "mean_target_tokens_old":sum(r["prompt_tokens"]+r["generated_tokens"] for r in oldrows)/128,
            "mean_target_tokens_format":sum(r["prompt_tokens"]+r["generated_tokens"] for r in formatrows)/128,
            "mean_target_tokens_selfreport":sum(r["prompt_tokens"]+r["generated_tokens"] for r in newrows)/128}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
