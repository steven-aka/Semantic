"""Matched old/new prompt comparison on fresh unstratified train-side IDs."""
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

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
C0 = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")
A0 = Path("results/v2_rank_then_cut/v17samecall_a0_sufficiency_pilot")
OUT = Path("results/v2_rank_then_cut/v17samecall_b0_unstratified_prompt_gate")
LEVELS = (.60, .70, .80, .90, .95)


def main() -> None:
    from vllm import SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path("configs/v17samecall_b0_unstratified_prompt_gate.json").read_text())
    a0cfg = json.loads(Path(cfg["new_prompt_config"]).read_text())
    assert cfg["status"] == "FROZEN_FRESH_TRAIN_SIDE_HOLDOUT"
    excluded = {r["example_id"] for r in read_jsonl(A0 / "per_call.jsonl")}
    train = {r["example_id"] for r in read_jsonl(C0 / "per_query.jsonl")}
    ids = sorted(train - excluded, key=lambda q: hashlib.sha256(("SAMECALL-B0|" + q).encode()).hexdigest())[:128]
    assert len(ids) == 128 and not set(ids) & excluded
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["example_id"] in data and r["depth"] in (9,10):
                chain[r["example_id"]][r["depth"]] = r
    assert len(data) == len(atoms) == len(chain) == 128 and all(len(x)==2 for x in chain.values())
    if args.analyze_only:
        from transformers import AutoTokenizer
        target = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(cfg["target_model"]))
    else:
        target = QampariTargetRunner(cfg["target_model"], backend="vllm", max_new_tokens=cfg["max_new_tokens"],
                                     max_model_len=cfg["max_model_len"],
                                     gpu_memory_utilization=cfg["gpu_memory_utilization"])
    tasks=[]
    for depth in cfg["depths"]:
        for q in ids:
            context="\n\n".join(t.strip() for i,t in enumerate(data[q]["packet_texts"])
                                 if chain[q][depth]["mask"] & (1<<i))
            user=build_qampari_prompt(data[q]["question"],context)
            for arm in cfg["arms"]:
                if arm=="canonical_old_prompt":
                    system=QAMPARI_SYSTEM_PROMPT
                else:
                    system=QAMPARI_SYSTEM_PROMPT+" "+a0cfg["prompt_suffix_system"]
                    user_new=user+"\n"+a0cfg["prompt_suffix_user"]
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user if arm=="canonical_old_prompt" else user_new}]
                prompt=target.tokenizer.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False)
                tasks.append({"example_id":q,"depth":depth,"arm":arm,"prompt":prompt})
    assert len(tasks)==512
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"per_call.jsonl"
    done={(r["example_id"],r["depth"],r["arm"]):r for r in read_jsonl(path)} if path.exists() else {}
    if args.analyze_only and len(done)!=512:
        raise AssertionError("analysis-only requires all 512 cached calls")
    params=SamplingParams(temperature=cfg["temperature"],max_tokens=cfg["max_new_tokens"])
    status_re=re.compile(a0cfg["status_regex"],re.IGNORECASE)
    with path.open("a") as handle:
        for offset in range(0,len(tasks),cfg["batch_size"]):
            batch=[t for t in tasks[offset:offset+cfg["batch_size"]]
                   if (t["example_id"],t["depth"],t["arm"]) not in done]
            if not batch:
                continue
            outputs=target.model.generate([t["prompt"] for t in batch],params,use_tqdm=False)
            for task,result in zip(batch,outputs):
                raw=result.outputs[0].text
                answer_tag=re.search(r"<answer>\s*(.*?)\s*</answer>",raw,re.IGNORECASE|re.DOTALL)
                parsed=parse_list_prediction(answer_tag.group(0)) if answer_tag else []
                status=status_re.search(raw) if task["arm"]=="samecall_a0_new_prompt" else None
                metric=qampari_list_metrics(parsed,atoms[task["example_id"]])
                row={k:task[k] for k in ("example_id","depth","arm")}
                row.update({"answer":" # ".join(parsed),"status":status.group(1).upper() if status else None,
                            "answer_tag_present":answer_tag is not None,
                            "f1":float(metric["f1"]),"correct":int(metric["correct"]),
                            "predicted":int(metric["predicted"]),"gold":int(metric["gold"]),
                            "prompt_tokens":len(target.tokenizer.encode(task["prompt"])),
                            "generated_tokens":len(result.outputs[0].token_ids)})
                done[(task["example_id"],task["depth"],task["arm"])]=row
                handle.write(json.dumps(row)+"\n")
            handle.flush()
            print(json.dumps({"completed":len(done),"total":len(tasks)}),flush=True)
    assert len(done)==512
    report={"protocol":cfg["protocol"],"queries":128,"target_calls":512,
            "design_pilot_ids_excluded":len(excluded),"depths":{},"sealed_outcome_sets_read":False}
    for depth in cfg["depths"]:
        old=[done[(q,depth,"canonical_old_prompt")] for q in ids]
        new=[done[(q,depth,"samecall_a0_new_prompt")] for q in ids]
        eligible={str(level):[i for i,q in enumerate(ids) if level in {float(z) for z in data[q]["attainable_levels"]}]
                  for level in LEVELS}
        counts={}
        for level in LEVELS:
            idx=eligible[str(level)]
            a=sum(old[i]["f1"]+1e-6>=level for i in idx)
            b=sum(new[i]["f1"]+1e-6>=level for i in idx)
            counts[str(level)]={"eligible":len(idx),"old_success":a,"new_success":b,"net":b-a,
                                "repairs":sum(old[i]["f1"]+1e-6<level and new[i]["f1"]+1e-6>=level for i in idx),
                                "breaks":sum(old[i]["f1"]+1e-6>=level and new[i]["f1"]+1e-6<level for i in idx)}
        old_complete=sum(all(old[i]["f1"]+1e-6>=float(z) for z in data[q]["attainable_levels"])
                         for i,q in enumerate(ids))
        new_complete=sum(all(new[i]["f1"]+1e-6>=float(z) for z in data[q]["attainable_levels"])
                         for i,q in enumerate(ids))
        report["depths"][str(depth)]={"anchors":counts,"old_complete":old_complete,"new_complete":new_complete,
            "mean_old_f1":sum(r["f1"] for r in old)/128,"mean_new_f1":sum(r["f1"] for r in new)/128,
            "mean_old_answer_count":sum(r["predicted"] for r in old)/128,
            "mean_new_answer_count":sum(r["predicted"] for r in new)/128,
            "old_answer_tag_fraction":sum(r["answer_tag_present"] for r in old)/128,
            "new_answer_tag_fraction":sum(r["answer_tag_present"] for r in new)/128,
            "new_status_valid_fraction":sum(r["status"] in ("YES","NO") for r in new)/128,
            "mean_old_target_tokens":sum(r["prompt_tokens"]+r["generated_tokens"] for r in old)/128,
            "mean_new_target_tokens":sum(r["prompt_tokens"]+r["generated_tokens"] for r in new)/128,
            "mean_old_prompt_tokens":sum(r["prompt_tokens"] for r in old)/128,
            "mean_new_prompt_tokens":sum(r["prompt_tokens"] for r in new)/128,
            "mean_old_generated_tokens":sum(r["generated_tokens"] for r in old)/128,
            "mean_new_generated_tokens":sum(r["generated_tokens"] for r in new)/128}
    gate=report["depths"]["10"]
    quality=gate["anchors"]["0.9"]["net"]>=5 and gate["new_complete"]>=gate["old_complete"] and all(
        x["net"]>=-3 for x in gate["anchors"].values())
    cost=gate["mean_new_target_tokens"]<=1.25*gate["mean_old_target_tokens"]
    report["research_gate"]={"quality_pass":quality,"cost_pass":cost,"pass":quality and cost,
                              "decision":"GO_PROMPT_ONLY_METHOD_DESIGN" if quality and cost else "STOP_PROMPT_ONLY_BRANCH"}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
