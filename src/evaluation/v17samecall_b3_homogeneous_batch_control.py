"""Old/new prompt comparison in separate homogeneous batches on B0 IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction, qampari_list_metrics
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, QampariTargetRunner, build_qampari_prompt

ROOT=Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0=Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
B0=Path("results/v2_rank_then_cut/v17samecall_b0_unstratified_prompt_gate")
PILOT=Path("results/v2_rank_then_cut/v17samecall_a0_sufficiency_pilot")
C0=Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
OUT=Path("results/v2_rank_then_cut/v17samecall_b3_homogeneous_batch_control")
LEVELS=(.60,.70,.80,.90,.95)


def main() -> None:
    from vllm import SamplingParams
    parser=argparse.ArgumentParser()
    parser.add_argument("--analyze-only",action="store_true")
    parser.add_argument("--config",default="configs/v17samecall_b3_homogeneous_batch_control.json")
    parser.add_argument("--output-dir",default=str(OUT))
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    a0=json.loads(Path(cfg["new_prompt_config"]).read_text())
    assert cfg["status"] in ("FROZEN_EXECUTION_CONTRACT_CORRECTION","FROZEN_FRESH_REPLICATION")
    if cfg.get("sample_strategy")=="fresh_replication":
        excluded={r["example_id"] for r in read_jsonl(B0/"per_call.jsonl")}
        excluded.update(r["example_id"] for r in read_jsonl(PILOT/"per_call.jsonl"))
        train={r["example_id"] for r in read_jsonl(C0/"per_query.jsonl")}
        ids=sorted(train-excluded,key=lambda q:hashlib.sha256(("SAMECALL-B4|"+q).encode()).hexdigest())[:128]
    else:
        ids=sorted({r["example_id"] for r in read_jsonl(B0/"per_call.jsonl")})
    assert len(ids)==128
    data={r["example_id"]:r for r in read_jsonl(ROOT/"data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms={r["example_id"]:r["answer_atoms"] for r in read_jsonl("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain=defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in data and r["depth"] in (9,10):chain[r["example_id"]][r["depth"]]=r
    assert len(data)==len(atoms)==len(chain)==128
    if args.analyze_only:
        from transformers import AutoTokenizer
        target=SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(cfg["target_model"]))
    else:
        target=QampariTargetRunner(cfg["target_model"],backend="vllm",max_new_tokens=cfg["max_new_tokens"],
                                    max_model_len=cfg["max_model_len"],gpu_memory_utilization=cfg["gpu_memory_utilization"])
    tasks=[]
    for arm in cfg["arms"]:
        for depth in cfg["depths"]:
            for q in ids:
                context="\n\n".join(t.strip() for i,t in enumerate(data[q]["packet_texts"])
                                     if chain[q][depth]["mask"]&(1<<i))
                user=build_qampari_prompt(data[q]["question"],context)
                system=QAMPARI_SYSTEM_PROMPT
                if arm=="samecall_a0_new_prompt":
                    system+=" "+a0["prompt_suffix_system"]
                    user+="\n"+a0["prompt_suffix_user"]
                messages=[{"role":"system","content":system},{"role":"user","content":user}]
                prompt=target.tokenizer.apply_chat_template(messages,tokenize=False,
                                                            add_generation_prompt=True,enable_thinking=False)
                tasks.append((q,depth,arm,prompt))
    assert len(tasks)==512
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/"per_call.jsonl"
    done={(r["example_id"],r["depth"],r["arm"]):r for r in read_jsonl(path)} if path.exists() else {}
    if args.analyze_only and len(done)!=512:raise AssertionError("incomplete cache")
    params=SamplingParams(temperature=cfg["temperature"],max_tokens=cfg["max_new_tokens"])
    with path.open("a") as handle:
        for offset in range(0,len(tasks),cfg["batch_size"]):
            batch=[t for t in tasks[offset:offset+cfg["batch_size"]] if (t[0],t[1],t[2]) not in done]
            if not batch:continue
            outputs=target.model.generate([t[3] for t in batch],params,use_tqdm=False)
            for (q,depth,arm,prompt),out in zip(batch,outputs):
                raw=out.outputs[0].text
                tag=bool(re.search(r"<answer>\s*(.*?)\s*</answer>",raw,re.IGNORECASE|re.DOTALL))
                answer=parse_list_prediction(raw)
                metric=qampari_list_metrics(answer,atoms[q])
                row={"example_id":q,"depth":depth,"arm":arm,"raw_output":raw,
                     "answer":" # ".join(answer),"answer_tag_present":tag,
                     "f1":float(metric["f1"]),"predicted":int(metric["predicted"]),
                     "prompt_tokens":len(target.tokenizer.encode(prompt)),
                     "generated_tokens":len(out.outputs[0].token_ids)}
                done[q,depth,arm]=row
                handle.write(json.dumps(row)+"\n")
            handle.flush()
            print(json.dumps({"completed":len(done),"total":512}),flush=True)
    assert len(done)==512
    b0={(r["example_id"],r["depth"],r["arm"]):r for r in read_jsonl(B0/"per_call.jsonl")}
    report={"protocol":cfg["protocol"],"queries":128,"target_calls":512,
            "sample_strategy":cfg.get("sample_strategy","B0_same_sample"),
            "query_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "depths":{},"sealed_outcome_sets_read":False}
    for depth in (9,10):
        old=[done[q,depth,"canonical_old_prompt"] for q in ids]
        new=[done[q,depth,"samecall_a0_new_prompt"] for q in ids]
        anchors={}
        for level in LEVELS:
            idx=[i for i,q in enumerate(ids) if level in {float(z) for z in data[q]["attainable_levels"]}]
            anchors[str(level)]={"eligible":len(idx),
                                 "old_success":sum(old[i]["f1"]+1e-6>=level for i in idx),
                                 "new_success":sum(new[i]["f1"]+1e-6>=level for i in idx),
                                 "repairs":sum(old[i]["f1"]+1e-6<level and new[i]["f1"]+1e-6>=level for i in idx),
                                 "breaks":sum(old[i]["f1"]+1e-6>=level and new[i]["f1"]+1e-6<level for i in idx)}
        report["depths"][str(depth)]={"anchors":anchors,
            "old_complete":sum(all(old[i]["f1"]+1e-6>=float(z) for z in data[q]["attainable_levels"])
                               for i,q in enumerate(ids)),
            "new_complete":sum(all(new[i]["f1"]+1e-6>=float(z) for z in data[q]["attainable_levels"])
                               for i,q in enumerate(ids)),
            "old_answer_tag_count":sum(r["answer_tag_present"] for r in old),
            "new_answer_tag_count":sum(r["answer_tag_present"] for r in new),
            "mean_old_target_tokens":sum(r["prompt_tokens"]+r["generated_tokens"] for r in old)/128,
            "mean_new_target_tokens":sum(r["prompt_tokens"]+r["generated_tokens"] for r in new)/128,
            "old_vs_B0_old_answer_agreement":sum(
                old[i]["answer"]==b0[q,depth,"canonical_old_prompt"]["answer"]
                for i,q in enumerate(ids)) if cfg.get("sample_strategy")!="fresh_replication" else None}
    (output_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
