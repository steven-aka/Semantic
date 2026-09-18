from __future__ import annotations

import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.data.schemas import ExactSearchResult,read_jsonl
from src.evaluation.v16b_boundary_causal_audit import LEVELS,order_result
from src.reproducibility import sha256,write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def worker(payload):
    source,order,reference,exact_dir=payload;levels=tuple(source["attainable_levels"]);exact=list(read_jsonl(Path(exact_dir)/f"{source['example_id']}.jsonl",ExactSearchResult));by={state_to_mask(x.state):x for x in exact};dp=SequentialTrajectoryDP(exact,levels)
    masks=[0]
    for p in order:masks.append(masks[-1]|(1<<p))
    reached=[];value=dp.attained[0]
    for mask in masks:value=max(value,dp.attained[mask]);reached.append(value)
    oracle=[x["oracle_nested_tokens"] for x in reference["anchors"]];base=order_result(order,by,levels,oracle);candidates=[base];seen=set()
    for depth,mask in enumerate(masks[:-1]):
        current=tuple(int(float(by[mask].fidelity)+1e-12>=x) for x in levels);selected=[p for p in range(12) if mask&(1<<p)];remaining=[p for p in range(12) if not mask&(1<<p)]
        for remove in selected:
            entry=order.index(remove);optimal=set(dp.optimal_actions(masks[entry],reached[entry]))
            if remove in optimal:continue
            for add in remaining:
                if add not in optimal:continue
                swapped=(mask&~(1<<remove))|(1<<add);after=tuple(int(float(by[swapped].fidelity)+1e-12>=x) for x in levels);delta=tuple(b-a for a,b in zip(current,after))
                if not any(x>0 for x in delta) or any(x<0 for x in delta):continue
                new=list(order);i=new.index(remove);j=new.index(add);new[i],new[j]=new[j],new[i];key=tuple(new)
                if key not in seen:seen.add(key);candidates.append(order_result(new,by,levels,oracle))
    best=min(candidates,key=lambda x:(-sum(x["success"]),x["cumulative_tokens"]));return {"example_id":source["example_id"],"baseline":base,"strict_oracle":best,"candidate_orders":len(candidates)}


def summarize(rows,field):
    values=[x[field] for x in rows];reg=[x["regret"] for x in values if x["regret"] is not None]
    return {"per_anchor":{str(level):{"examples":sum(len(x["success"])>i for x in values),"successes":sum(len(x["success"])>i and x["success"][i] for x in values)} for i,level in enumerate(LEVELS)},"complete":sum(x["complete"] for x in values),"mean_regret_feasible":sum(reg)/len(reg)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--orders",required=True);p.add_argument("--v8-details",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--output",required=True);p.add_argument("--workers",type=int,default=12);a=p.parse_args();data=list(read_jsonl(a.data));orders={x["example_id"]:x["decoded_order"] for x in read_jsonl(a.orders)};refs={x["example_id"]:x for x in read_jsonl(a.v8_details)}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(worker,((x,orders[x["example_id"]],refs[x["example_id"]],a.exact_dir) for x in data),chunksize=2))
    result={"complete":True,"scientific_role":"consumed-development oracle ceiling for DP-consistent causal swaps rendered as legal add-only orders","baseline":summarize(rows,"baseline"),"strict_dp_consistent_single_swap_add_only_oracle":summarize(rows,"strict_oracle"),"mean_candidate_orders":sum(x["candidate_orders"] for x in rows)/len(rows),"examples_with_alternative":sum(x["candidate_orders"]>1 for x in rows),"artifacts_sha256":{x:sha256(x) for x in (a.data,a.orders,a.v8_details)}};write_metadata(a.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
