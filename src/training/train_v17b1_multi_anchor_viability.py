from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import head_state_dict
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.model.v17b_viability_policy import V17BViabilityPolicy
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain
from src.training.v17b_viability import StratifiedStateSampler, masked_viability_loss, set_valued_action_loss, viability_mass_loss


class DummyLM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__(); self.config = SimpleNamespace(hidden_size=hidden_size)


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_policy(checkpoint: str, device: torch.device, hidden_dim: int) -> V17BViabilityPolicy:
    state = torch.load(Path(checkpoint) / "sequential_head.pt", map_location="cpu", weights_only=True)
    lm_hidden = int(state["input_projection.0.weight"].shape[1])
    model_dim = int(state["initial_history.weight"].shape[0])
    model = V17BViabilityPolicy(DummyLM(lm_hidden), model_dim=model_dim, viability_hidden_dim=hidden_dim)
    result = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("viability_head.")}
    if result.unexpected_keys or set(result.missing_keys) != expected_missing:
        raise ValueError(f"invalid V8 initialization: {result}")
    model.freeze_v8(); model.move_head(device=device, dtype=torch.float32); return model


def make_batch(records: list[dict[str, Any]], indices: list[int], source: dict[str, dict[str, Any]], embedding_index: dict[str, int], cache: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    rows = [records[index] for index in indices]; width = max(len(row["history"]) for row in rows)
    histories = torch.full((len(rows), width), -1, dtype=torch.long, device=device)
    lengths = torch.tensor([len(row["history"]) for row in rows], dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        if row["history"]: histories[i, :len(row["history"])] = torch.tensor(row["history"], device=device)
    embedding_rows = torch.tensor([embedding_index[row["example_id"]] for row in rows])
    packets = cache["packets"][embedding_rows].to(device=device, dtype=torch.float32)
    questions = cache["questions"][embedding_rows].to(device=device, dtype=torch.float32)
    fractions = torch.tensor([source[row["example_id"]]["packet_tokens"] for row in rows], device=device, dtype=torch.float32)
    fractions /= fractions.sum(dim=1, keepdim=True)
    states, selected = None, None
    target = torch.zeros((len(rows), 12, 5), device=device); target_mask = torch.zeros_like(target, dtype=torch.bool)
    optimal = torch.zeros((len(rows), 12), device=device, dtype=torch.bool); legal = torch.zeros_like(optimal)
    for i, row in enumerate(rows):
        optimal[i, row["exact_dp_optimal_actions"]] = True
        for action in row["actions"]:
            packet = int(action["packet"]); legal[i, packet] = True
            for anchor, value in enumerate(action["future_viability"]):
                if value is not None: target[i, packet, anchor] = float(value); target_mask[i, packet, anchor] = True
    return {"rows": rows, "packets": packets, "questions": questions, "histories": histories, "lengths": lengths, "fractions": fractions, "target": target, "target_mask": target_mask, "optimal": optimal, "legal": legal}


def evaluate(model: V17BViabilityPolicy, data: list[dict[str, Any]], cache: dict[str, torch.Tensor], embedding_index: dict[str, int], exact_dir: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details=[]; per_level={}; complete=0; regrets=[]
    model.eval()
    with torch.no_grad():
        for source in data:
            idx=embedding_index[source["example_id"]]; packets=cache["packets"][idx].to(device=device,dtype=torch.float32); question=cache["questions"][idx].to(device=device,dtype=torch.float32)
            fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32); fractions/=fractions.sum()
            levels=trajectory_levels(source); order=model.beam_order_v17(packets,question,fractions,active_level_count=len(levels),beam_width=8)
            exact=list(read_jsonl(Path(exact_dir)/f"{source['example_id']}.jsonl",ExactSearchResult)); learned=best_prefix_nested_chain(exact,order,levels); oracle=best_binary_nested_chain(exact,levels)
            successes=[]; anchor_rows=[]
            for level,predicted,baseline in zip(levels,learned,oracle):
                success=bool(predicted["feasible"]); successes.append(success); per_level.setdefault(level,[]).append(success)
                anchor_rows.append({"fidelity_level":level,"contract_success":success,"tokens":predicted["tokens"]})
            regret=None
            if all(successes):
                complete+=1; full=next(row.tokens for row in exact if all(row.state)); regret=(sum(int(x["tokens"]) for x in learned)-sum(int(x["tokens"]) for x in oracle))/(len(levels)*full); regrets.append(regret)
            details.append({"example_id":source["example_id"],"decoded_order":list(order),"all_active_contracts_success":all(successes),"oracle_cutoff_ranking_regret_normalized":regret,"anchors":anchor_rows})
    return {"complete":True,"examples":len(data),"per_level":{str(k):{"examples":len(v),"contract_successes":sum(v)} for k,v in sorted(per_level.items())},"complete_trajectory_successes":complete,"mean_complete_regret":mean(regrets),"decoder":"V17 viability residual add-only beam8"},details


def main() -> None:
    p=argparse.ArgumentParser(description="V17-B1 frozen multi-anchor viability training")
    p.add_argument("--protocol-config",required=True); p.add_argument("--artifact",required=True); p.add_argument("--train-data",required=True); p.add_argument("--internal-data",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--exact-dir",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--v8-internal-details",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--steps",type=int,default=750); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--warmup-steps",type=int,default=50); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--survival-weight",type=float,default=.25); p.add_argument("--dp-weight",type=float,default=.1); p.add_argument("--seed",type=int,default=20260918)
    args=p.parse_args(); protocol=json.loads(Path(args.protocol_config).read_text()); expected=protocol["arguments"]
    mismatch={k:(getattr(args,k),v) for k,v in expected.items() if getattr(args,k)!=v}
    if protocol["status"]!="APPROVED_TO_RUN" or mismatch: raise ValueError({"status":protocol["status"],"mismatch":mismatch})
    random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device("cuda:0"); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); write_metadata(out/"protocol_snapshot.json",protocol)
    records=read_gzip(args.artifact); source={row["example_id"]:row for row in read_jsonl(args.train_data)}; cache=torch.load(args.embeddings,map_location="cpu",weights_only=True); embedding_index={x:i for i,x in enumerate(cache["example_ids"])}
    if not set(source)<=set(embedding_index): raise ValueError("embedding cache does not cover train2863")
    model=load_policy(args.checkpoint,device,args.hidden_dim); params=[p for p in model.parameters() if p.requires_grad]; optimizer=torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=.01)
    sampler=StratifiedStateSampler(records,args.seed); history=[]; started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train(); batch=make_batch(records,sampler.sample(args.batch_size,step),source,embedding_index,cache,device)
        states,selected=model.history_states(batch["packets"],batch["questions"],batch["histories"],batch["lengths"],torch.arange(args.batch_size,device=device))
        logits,_,viability=model.score_states_with_viability(batch["packets"],batch["questions"],states,selected,torch.arange(args.batch_size,device=device),batch["fractions"],batch["target_mask"])
        lv=masked_viability_loss(viability,batch["target"],batch["target_mask"]); ls=viability_mass_loss(logits,batch["target"],batch["target_mask"],batch["legal"]); ld=set_valued_action_loss(logits,batch["optimal"],batch["legal"]); loss=lv+args.survival_weight*ls+args.dp_weight*ld
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0)
        scale=min(1.0,step/args.warmup_steps)*.5*(1+math.cos(math.pi*max(0,step-args.warmup_steps)/max(1,args.steps-args.warmup_steps)))
        for group in optimizer.param_groups: group["lr"]=args.learning_rate*scale
        optimizer.step()
        if step==1 or step%25==0:
            row={"optimizer_step":step,"loss":float(loss),"viability":float(lv),"survival":float(ls),"dp":float(ld),"residual_scale":float(model.viability_head.residual_scale.detach()),"learning_rate":optimizer.param_groups[0]["lr"]}; history.append(row); write_jsonl(out/"training_history.jsonl",history); print(json.dumps(row),flush=True)
    torch.save(model.viability_head.state_dict(),out/"viability_head.pt")
    internal=list(read_jsonl(args.internal_data)); summary,details=evaluate(model,internal,cache,embedding_index,args.exact_dir,device); write_jsonl(out/"internal_details.jsonl",details); write_metadata(out/"internal_summary.json",summary)
    with gzip.open(args.v8_internal_details,"rt") as f: baseline={x["example_id"]:x for line in f if (x:=json.loads(line))}
    baseline_success={}
    for source_row in internal:
        exact=list(read_jsonl(Path(args.exact_dir)/f"{source_row['example_id']}.jsonl",ExactSearchResult)); levels=trajectory_levels(source_row)
        predicted=best_prefix_nested_chain(exact,baseline[source_row["example_id"]]["decoded_order"],levels)
        baseline_success[source_row["example_id"]]={level:bool(row["feasible"]) for level,row in zip(levels,predicted)}
    changes={}
    for level in (0.6,0.7,0.8,0.9,0.95):
        repairs=breaks=0; examples=0
        for row in details:
            current=next((a["contract_success"] for a in row["anchors"] if abs(a["fidelity_level"]-level)<1e-9),None)
            old=baseline_success[row["example_id"]].get(level)
            if current is not None and old is not None: examples+=1; repairs+=int(current and not old); breaks+=int(old and not current)
        changes[str(level)]={"examples":examples,"repairs":repairs,"breaks":breaks,"net":repairs-breaks}
    gate=summary["per_level"]["0.9"]["contract_successes"]>=283 and changes["0.9"]["net"]>=3 and changes["0.9"]["breaks"]<=1 and summary["complete_trajectory_successes"]>=274 and summary["mean_complete_regret"]<=.03
    decision={"decision":"GO_CONSUMED_DEVELOPMENT_ONCE" if gate else "STOP_V17B1_INTERNAL_GATE","gate_passed":gate,"internal":summary,"changes_from_v8":changes,"development_used":False,"fresh_confirmation_opened":False}; write_metadata(out/"decision.json",decision)
    write_metadata(out/"training_metadata.json",experiment_metadata(stage="v17b1_multi_anchor_viability",base_checkpoint=args.checkpoint,artifact_sha256=sha256(args.artifact),optimizer_steps=args.steps,trainable="viability_head_and_residual_scale_only",elapsed_seconds=time.perf_counter()-started,development300_used=False,locked_roles_used=False))
    print(json.dumps(decision,indent=2))


if __name__=="__main__": main()
