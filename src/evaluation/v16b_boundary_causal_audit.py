from __future__ import annotations

import argparse,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.data.schemas import ExactSearchResult,read_jsonl,write_jsonl
from src.reproducibility import sha256,write_metadata
from src.search.atomic_nested_chain import state_to_mask


LEVELS=(.6,.7,.8,.9,.95)


def group(example_id):
    for value in ("wikitables_composition","wikidata_intersection","wikidata_simple"):
        if value in example_id:return value
    return "other"


def analyze_one(payload):
    source,order,exact_dir,fixed_histories,on_policy_histories=payload;eid=source["example_id"];levels=tuple(source["attainable_levels"])
    exact=list(read_jsonl(Path(exact_dir)/f"{eid}.jsonl",ExactSearchResult));by={state_to_mask(x.state):x for x in exact}
    def vector(mask):return tuple(int(float(by[mask].fidelity)+1e-12>=x) for x in LEVELS if x in levels)
    prefixes=[0]
    for packet in order:prefixes.append(prefixes[-1]|(1<<packet))
    reached=[];value=0
    for mask in prefixes: value=max(value,sum(vector(mask)));reached.append(value)
    fixed={tuple(x["history"]):x for x in fixed_histories};on_policy={tuple(x["history"]):x for x in on_policy_histories}
    boundary_states=0;add_positive_states=0;swap_positive_states=0;safe_adds=0;safe_swaps=[];labels={}
    for depth,mask in enumerate(prefixes[:-1]):
        current=vector(mask);remaining=[p for p in range(12) if not mask&(1<<p)];selected=[p for p in range(12) if mask&(1<<p)]
        add_positive=False
        for add in remaining:
            after=vector(mask|(1<<add));delta=tuple(b-a for a,b in zip(current,after));safe=any(x>0 for x in delta) and not any(x<0 for x in delta)
            safe_adds+=int(safe);add_positive|=safe
        swap_positive=False
        for remove in selected:
            for add in remaining:
                swapped=(mask&~(1<<remove))|(1<<add);after=vector(swapped);delta=tuple(b-a for a,b in zip(current,after));safe=any(x>0 for x in delta) and not any(x<0 for x in delta)
                if not safe:continue
                swap_positive=True;entry=order.index(remove);history=tuple(order[:entry]);record=on_policy.get(history);key=(history,add,remove)
                labels[key]={"example_id":eid,"boundary_depth":depth,"predecessor_depth":entry,"predecessor_mask":prefixes[entry],"preferred_add":add,"deferred_add":remove,"crossing_delta":delta,"fixed_v8_pool_covered":history in fixed,"v9_on_policy_covered":history in on_policy,"optimal_action_count":len(record["optimal_actions"]) if record else None,"preferred_is_dp_optimal":add in record["optimal_actions"] if record else None,"pattern":[group(eid),entry,reached[entry],list(delta)]}
                safe_swaps.append((depth,remove,add,swapped))
        add_positive_states+=int(add_positive);swap_positive_states+=int(swap_positive);boundary_states+=int(add_positive or swap_positive)
    return {"example_id":eid,"deployed_prefix_states":12,"boundary_positive_states":boundary_states,"add_positive_states":add_positive_states,"swap_positive_states":swap_positive_states,"safe_add_actions":safe_adds,"safe_swap_actions":len(safe_swaps),"causal_labels":list(labels.values()),"safe_swaps":safe_swaps,"prefixes":prefixes,"active_levels":levels,"order":order,"exact":exact}


def run(rows,orders,exact_dir,fixed,on_policy,workers):
    payloads=((r,orders[r["example_id"]],exact_dir,fixed[r["example_id"]],on_policy[r["example_id"]]) for r in rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:return list(pool.map(analyze_one,payloads,chunksize=2))


def aggregate(rows):
    labels=[x for r in rows for x in r["causal_labels"]];states=sum(r["deployed_prefix_states"] for r in rows);patterns=Counter(json.dumps(x["pattern"],sort_keys=True) for x in labels)
    return {"examples":len(rows),"deployed_prefix_states":states,"boundary_positive_states":sum(r["boundary_positive_states"] for r in rows),"boundary_positive_density":sum(r["boundary_positive_states"] for r in rows)/states,"states_with_safe_add":sum(r["add_positive_states"] for r in rows),"states_with_safe_swap":sum(r["swap_positive_states"] for r in rows),"safe_add_actions":sum(r["safe_add_actions"] for r in rows),"safe_swap_actions":sum(r["safe_swap_actions"] for r in rows),"unique_causal_predecessor_labels":len(labels),"fixed_v8_pool_coverage":sum(x["fixed_v8_pool_covered"] for x in labels)/len(labels) if labels else None,"v9_on_policy_coverage":sum(x["v9_on_policy_covered"] for x in labels)/len(labels) if labels else None,"preferred_is_dp_optimal_fraction":sum(x["preferred_is_dp_optimal"] for x in labels if x["preferred_is_dp_optimal"] is not None)/sum(x["preferred_is_dp_optimal"] is not None for x in labels) if labels else None,"mean_optimal_action_count":sum(x["optimal_action_count"] for x in labels if x["optimal_action_count"] is not None)/sum(x["optimal_action_count"] is not None for x in labels) if labels else None,"pattern_counts":dict(patterns)}


def order_result(order,by,levels,oracle_tokens):
    masks=[0]
    for p in order:masks.append(masks[-1]|(1<<p))
    chosen=[]
    for level in levels:
        possible=[by[m].tokens for m in masks if float(by[m].fidelity)+1e-12>=level];chosen.append(min(possible) if possible else None)
    success=[x is not None for x in chosen];cost=sum(x for x in chosen if x is not None);regret=(cost-sum(oracle_tokens))/(len(levels)*by[4095].tokens) if all(success) else None
    return {"success":success,"complete":all(success),"cumulative_tokens":cost,"regret":regret}


def ceiling(rows,v8_details):
    refs={x["example_id"]:x for x in v8_details};details=[]
    for row in rows:
        by={state_to_mask(x.state):x for x in row["exact"]};ref=refs[row["example_id"]];oracle=[x["oracle_nested_tokens"] for x in ref["anchors"]];base=order_result(row["order"],by,row["active_levels"],oracle);candidates=[base]
        seen=set()
        for _,remove,add,_ in row["safe_swaps"]:
            order=list(row["order"]);i=order.index(remove);j=order.index(add);order[i],order[j]=order[j],order[i];key=tuple(order)
            if key not in seen:seen.add(key);candidates.append(order_result(order,by,row["active_levels"],oracle))
        best=min(candidates,key=lambda x:(-sum(x["success"]),x["cumulative_tokens"]));details.append({"example_id":row["example_id"],"baseline":base,"restricted_oracle":best,"candidate_orders":len(candidates)})
    def summary(field):
        values=[x[field] for x in details];return {"per_anchor":{str(level):{"examples":sum(len(x["success"])>i for x in values),"successes":sum(len(x["success"])>i and x["success"][i] for x in values)} for i,level in enumerate(LEVELS)},"complete":sum(x["complete"] for x in values),"mean_regret_feasible":sum(x["regret"] for x in values if x["regret"] is not None)/sum(x["regret"] is not None for x in values)}
    return {"baseline":summary("baseline"),"restricted_single_swap_add_only_oracle":summary("restricted_oracle"),"mean_candidate_orders":sum(x["candidate_orders"] for x in details)/len(details)},details


def main():
    p=argparse.ArgumentParser();p.add_argument("--train-data",required=True);p.add_argument("--validation-data",required=True);p.add_argument("--orders",required=True);p.add_argument("--fixed-v9",required=True);p.add_argument("--on-policy-v9",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--development-data",required=True);p.add_argument("--development-orders",required=True);p.add_argument("--development-v8-details",required=True);p.add_argument("--output",required=True);p.add_argument("--details",required=True);p.add_argument("--workers",type=int,default=12);a=p.parse_args()
    train=list(read_jsonl(a.train_data));valid=list(read_jsonl(a.validation_data));orders={x["example_id"]:x["decoded_order"] for x in read_jsonl(a.orders)};fixed={x["example_id"]:x["histories"] for x in read_jsonl(a.fixed_v9)};on={x["example_id"]:x["histories"] for x in read_jsonl(a.on_policy_v9)}
    train_rows=run(train,orders,a.exact_dir,fixed,on,a.workers);valid_rows=run(valid,orders,a.exact_dir,fixed,on,a.workers);train_summary=aggregate(train_rows);valid_summary=aggregate(valid_rows);train_patterns=set(train_summary.pop("pattern_counts"));valid_patterns=valid_summary.pop("pattern_counts");validation_pattern_coverage=sum(count for pattern,count in valid_patterns.items() if pattern in train_patterns)/sum(valid_patterns.values()) if valid_patterns else None
    dev=list(read_jsonl(a.development_data));dev_orders={x["example_id"]:x["decoded_order"] for x in read_jsonl(a.development_orders)};empty={x["example_id"]:[] for x in dev};dev_rows=run(dev,dev_orders,a.exact_dir,empty,empty,a.workers);oracle,oracle_details=ceiling(dev_rows,list(read_jsonl(a.development_v8_details)))
    result={"complete":True,"scientific_role":"V16-B0 boundary-causal data feasibility audit; development is consumed and oracle-only","train2863":train_summary,"internal_validation300":valid_summary,"validation_causal_pattern_coverage_from_train":validation_pattern_coverage,"consumed_development300_ceiling":oracle,"comparison_to_v9":{"v9_on_policy_deployed_states":37956,"v9_on_policy_lost_anchor_action_fraction":.009063125724523132,"note":"V9 already supplied cost-to-go advantages; V16-B novelty requires materially denser fidelity-crossing causal labels, not another advantage loss."},"artifacts_sha256":{x:sha256(x) for x in (a.train_data,a.validation_data,a.orders,a.fixed_v9,a.on_policy_v9,a.development_data,a.development_orders,a.development_v8_details)},"confirmation_or_calibration_used":False}
    write_metadata(a.output,result);write_jsonl(a.details,[{"split":"train",**x} for r in train_rows for x in r["causal_labels"]]+[{"split":"internal_validation",**x} for r in valid_rows for x in r["causal_labels"]]+oracle_details);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
