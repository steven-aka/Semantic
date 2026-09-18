from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.reproducibility import sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP
from src.training.train_v17b1_multi_anchor_viability import load_policy


def read_gzip(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def replay(
    model: Any,
    packets: torch.Tensor,
    question: torch.Tensor,
    fractions: torch.Tensor,
    dp: SequentialTrajectoryDP,
    *,
    residual_enabled: bool,
    oracle_keep_one: bool = False,
    full_trace: bool = False,
    beam_width: int = 8,
) -> dict[str, Any]:
    primary = next(i for i, level in enumerate(dp.levels) if abs(level - .9) < 1e-9)
    # total score, base score, normalized residual effect, history, recurrent state, mask, reached
    beams = [(0.0, 0.0, 0.0, (), torch.tanh(model.initial_history(question[None]))[0], 0, dp.attained[0])]
    trace=[]; first_prune=None
    for depth in range(1, 13):
        count=len(beams); states=torch.stack([row[4] for row in beams]); selected=torch.zeros((count,12),dtype=torch.bool,device=packets.device)
        for i,row in enumerate(beams):
            if row[3]: selected[i,list(row[3])]=True
        ps=packets[None].expand(count,-1,-1); qs=question[None].expand(count,-1); indices=torch.arange(count,device=packets.device); fs=fractions[None].expand(count,-1)
        features,progress=model.action_features(ps,qs,states,selected,indices,fs); base_logits=model.action_head(features).squeeze(-1).masked_fill(selected,-torch.inf)
        active=model.deployment_active_target_mask(progress,torch.full((count,),len(dp.levels),device=packets.device)); _,raw_residual=model.viability_head(features,active)
        combined_logits=base_logits+raw_residual if residual_enabled else base_logits
        base_logp=torch.log_softmax(base_logits.float(),dim=-1).cpu(); combined_logp=torch.log_softmax(combined_logits.float(),dim=-1).cpu()
        next_states=model.history_gru(ps.reshape(-1,model.model_dim),states[:,None,:].expand(-1,12,-1).reshape(-1,model.model_dim)).reshape(count,12,model.model_dim)
        expanded=[]
        for i,(total,base,effect,history,_,mask,reached) in enumerate(beams):
            for packet in range(12):
                if selected[i,packet]: continue
                next_mask=mask|(1<<packet); next_reached=max(reached,dp.attained[next_mask]); bp=float(base_logp[i,packet]); cp=float(combined_logp[i,packet])
                viable=dp.value(next_mask,next_reached).reached_levels>=primary+1
                expanded.append({"total":total+cp,"base":base+bp,"residual_effect":effect+(cp-bp),"history":history+(packet,),"state":next_states[i,packet],"mask":next_mask,"reached":next_reached,"viable":viable,"parent_history":history,"action":packet,"base_increment":bp,"raw_residual_logit":float(raw_residual[i,packet]),"combined_increment":cp,"normalized_residual_increment":cp-bp})
        expanded.sort(key=lambda row:(-row["total"],row["history"])); cutoff=expanded[min(beam_width,len(expanded))-1]; viable_expanded=[row for row in expanded if row["viable"]]; best_viable=max(viable_expanded,key=lambda row:(row["total"],tuple(-x for x in row["history"])),default=None)
        kept=expanded[:beam_width]; forced=False
        if oracle_keep_one and best_viable is not None and not any(row["viable"] for row in kept):
            kept[-1]=best_viable; kept.sort(key=lambda row:(-row["total"],row["history"])); forced=True
        viable_before=sum(row["viable"] for row in expanded); viable_after=sum(row["viable"] for row in kept)
        event=None
        if best_viable is not None:
            event={"base_margin":best_viable["base"]-cutoff["base"],"residual_margin":best_viable["residual_effect"]-cutoff["residual_effect"],"total_margin":best_viable["total"]-cutoff["total"],"minimum_rescue_margin":max(0.0,cutoff["total"]-best_viable["total"]+1e-9),"best_viable_history":list(best_viable["history"]),"boundary_history":list(cutoff["history"]),"residual_direction":"positive" if best_viable["residual_effect"]-cutoff["residual_effect"]>0 else "nonpositive"}
        if first_prune is None and viable_before and not viable_after:
            first_prune={"step":depth,"viable_count_before":viable_before,"viable_count_after":viable_after,**(event or {})}
        row={"step":depth,"beam_cutoff_total":cutoff["total"],"viable_count_before":viable_before,"viable_count_after":viable_after,"forced_keep":forced,"best_viable_margin":event,"beam":[{"history":list(x["history"]),"total":x["total"],"base":x["base"],"residual_effect":x["residual_effect"],"viable":x["viable"],"rank":j+1} for j,x in enumerate(kept)]}
        if full_trace: row["expanded"]=[{k:(list(v) if isinstance(v,tuple) else v) for k,v in x.items() if k!="state"} for x in expanded]
        trace.append(row); beams=[(x["total"],x["base"],x["residual_effect"],x["history"],x["state"],x["mask"],x["reached"]) for x in kept]
    terminal=[]
    for row in beams:
        viable=dp.value(row[5],row[6]).reached_levels>=primary+1
        terminal.append({"history":list(row[3]),"score":row[0],"viable":viable})
    return {"decoded_order":list(beams[0][3]),"top1_success":terminal[0]["viable"],"any_terminal_viable":any(x["viable"] for x in terminal),"oracle_terminal_select_success":any(x["viable"] for x in terminal),"first_irreversible_prune":first_prune,"trace":trace}


def main() -> None:
    p=argparse.ArgumentParser(description="V17-C0 paired cumulative beam trajectory failure audit")
    p.add_argument("--data",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--exact-dir",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--viability-head",required=True); p.add_argument("--historical-v8",required=True); p.add_argument("--b1-details",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--break-id",default="56847__wikitables_composition__train")
    args=p.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda:0")
    data=list(read_jsonl(args.data)); by_id={x["example_id"]:x for x in data}; cache=torch.load(args.embeddings,map_location="cpu",weights_only=True); index={x:i for i,x in enumerate(cache["example_ids"])}
    historical={x["example_id"]:x for x in read_gzip(args.historical_v8)}; b1={x["example_id"]:x for x in read_jsonl(args.b1_details)}
    model=load_policy(args.checkpoint,device,args.hidden_dim); model.viability_head.load_state_dict(torch.load(args.viability_head,map_location=device,weights_only=True)); model.eval()
    off_orders={}; off_success={}; off_results={}; on_results={}; dps={}; tensors={}
    with torch.no_grad():
        for n,source in enumerate(data,1):
            i=index[source["example_id"]]; packets=cache["packets"][i].to(device=device,dtype=torch.float32); question=cache["questions"][i].to(device=device,dtype=torch.float32); fractions=torch.tensor(source["packet_tokens"],device=device,dtype=torch.float32); fractions/=fractions.sum()
            exact=list(read_jsonl(Path(args.exact_dir)/f"{source['example_id']}.jsonl",ExactSearchResult)); dp=SequentialTrajectoryDP(exact,trajectory_levels(source)); dps[source["example_id"]]=dp; tensors[source["example_id"]]=(packets,question,fractions)
            off=replay(model,packets,question,fractions,dp,residual_enabled=False); off_orders[source["example_id"]]=off["decoded_order"]; off_success[source["example_id"]]=off["top1_success"]
            primary_success=next(a["contract_success"] for a in b1[source["example_id"]]["anchors"] if abs(a["fidelity_level"]-.9)<1e-9)
            if not primary_success or source["example_id"]==args.break_id:
                off_results[source["example_id"]]=off
                on_results[source["example_id"]]=replay(model,packets,question,fractions,dp,residual_enabled=True)
            if n%50==0: print(json.dumps({"processed":n,"total":len(data)}),flush=True)
    exact_historical=sum(off_orders[i]==historical[i]["decoded_order"] for i in off_orders); exact_b1=sum((on_results[i]["decoded_order"]==b1[i]["decoded_order"]) for i in on_results)
    failure_rows=[]
    for example_id,on in on_results.items():
        primary_success=next(a["contract_success"] for a in b1[example_id]["anchors"] if abs(a["fidelity_level"]-.9)<1e-9)
        if primary_success: continue
        packets,question,fractions=tensors[example_id]; keep=replay(model,packets,question,fractions,dps[example_id],residual_enabled=True,oracle_keep_one=True)
        event=on["first_irreversible_prune"]
        if not off_success[example_id]: causal_status="persistent_same_harness_failure"
        else: causal_status="b1_induced_break"
        if not keep["any_terminal_viable"]: failure_class="D_oracle_keep_and_terminal_select_failed"
        elif event is None: failure_class="terminal_selection"
        elif event["residual_margin"]<=0: failure_class="B_residual_wrong_direction"
        else: failure_class="A_inherited_cumulative_deficit"
        first_composition=next((j+1 for j,(a,b) in enumerate(zip(off_results[example_id]["trace"],on["trace"])) if [x["history"] for x in a["beam"]] != [x["history"] for x in b["beam"]]),None)
        failure_rows.append({"sample":example_id,"residual_off_success":off_success[example_id],"causal_status":causal_status,"first_divergence_step":first_composition,"first_irreversible_prune_step":None if event is None else event["step"],"viable_count_before":None if event is None else event["viable_count_before"],"viable_count_after":None if event is None else event["viable_count_after"],"base_margin":None if event is None else event["base_margin"],"residual_margin":None if event is None else event["residual_margin"],"total_margin":None if event is None else event["total_margin"],"minimum_rescue_margin":None if event is None else event["minimum_rescue_margin"],"oracle_keep_one_has_terminal_viable":keep["any_terminal_viable"],"oracle_keep_one_top1_repairs":keep["top1_success"],"oracle_keep_one_plus_terminal_select_repairs":keep["oracle_terminal_select_success"],"residual_direction_at_failure":None if event is None else event["residual_direction"],"failure_class":failure_class})
    write_jsonl(out/"failure_table.jsonl",failure_rows)
    packets,question,fractions=tensors[args.break_id]; paired={"example_id":args.break_id,"residual_off":replay(model,packets,question,fractions,dps[args.break_id],residual_enabled=False,full_trace=True),"v17b1":replay(model,packets,question,fractions,dps[args.break_id],residual_enabled=True,full_trace=True),"oracle_keep_one":replay(model,packets,question,fractions,dps[args.break_id],residual_enabled=True,oracle_keep_one=True,full_trace=True)}; write_metadata(out/f"{args.break_id}_paired_trace.json",paired)
    classes=Counter(x["failure_class"] for x in failure_rows); repaired=sum(x["oracle_keep_one_plus_terminal_select_repairs"] for x in failure_rows); positive=sum(x["residual_direction_at_failure"]=="positive" for x in failure_rows)
    summary={"complete":True,"examples":len(data),"same_harness_residual_off_primary_090_successes":sum(off_success.values()),"b1_primary_090_failures":len(failure_rows),"b1_failures_that_are_persistent_under_residual_off":sum(not x["residual_off_success"] for x in failure_rows),"b1_induced_breaks_under_same_harness":sum(x["residual_off_success"] for x in failure_rows),"same_harness_residual_off_matches_historical_v8_orders":exact_historical,"same_harness_b1_matches_saved_b1_orders_in_audited_subset":exact_b1,"failure_classes":dict(classes),"oracle_keep_one_plus_terminal_select_repairs":repaired,"positive_residual_direction_at_first_prune":positive,"development_used":False,"hop2_used":False,"learned_cutoff_opened":False,"fresh_confirmation_opened":False,"artifacts":{"data_sha256":sha256(args.data),"embeddings_sha256":sha256(args.embeddings),"viability_head_sha256":sha256(args.viability_head)}}; write_metadata(out/"summary.json",summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
