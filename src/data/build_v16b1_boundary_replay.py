from __future__ import annotations

import argparse,json
from pathlib import Path

from src.data.schemas import read_jsonl,write_jsonl
from src.reproducibility import sha256,write_metadata


def main():
    p=argparse.ArgumentParser();p.add_argument("--on-policy",required=True);p.add_argument("--orders",required=True);p.add_argument("--causal-details",required=True);p.add_argument("--train-ids",required=True);p.add_argument("--validation-ids",required=True);p.add_argument("--train-output",required=True);p.add_argument("--validation-output",required=True);p.add_argument("--manifest",required=True);a=p.parse_args()
    rows={r["example_id"]:r for r in read_jsonl(a.on_policy)};orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(a.orders)};details=[r for r in read_jsonl(a.causal_details) if r.get("split") in ("train","internal_validation")]
    train_ids={r["example_id"] for r in read_jsonl(a.train_ids)};validation_ids={r["example_id"] for r in read_jsonl(a.validation_ids)};boundary={eid:set() for eid in rows}
    by_history={eid:{tuple(x["history"]):x for x in row["histories"]} for eid,row in rows.items()}
    for item in details:
        eid=item["example_id"];history=tuple(orders[eid][:item["predecessor_depth"]]);record=by_history[eid][history];optimal=set(record["optimal_actions"])
        if item["preferred_add"] in optimal and item["deferred_add"] not in optimal:boundary[eid].add(history)
    output={}
    for eid,row in rows.items():
        order=orders[eid];b=[by_history[eid][h] for h in sorted(boundary[eid],key=lambda x:(len(x),x))];retention=[]
        for record in row["histories"]:
            h=tuple(record["history"])
            if not record["source"].startswith("model_induced_") or h in boundary[eid] or len(h)>=12:continue
            if order[len(h)] in record["optimal_actions"]:retention.append(record)
        if b or retention:output[eid]={**row,"boundary_histories":b,"retention_histories":retention}
    def select(ids):return [output[eid] for eid in rows if eid in ids and eid in output]
    train=select(train_ids);valid=select(validation_ids);write_jsonl(a.train_output,train);write_jsonl(a.validation_output,valid)
    def summary(values):return {"queries":len(values),"queries_with_boundary":sum(bool(x["boundary_histories"]) for x in values),"queries_with_retention":sum(bool(x["retention_histories"]) for x in values),"boundary_states":sum(len(x["boundary_histories"]) for x in values),"retention_states":sum(len(x["retention_histories"]) for x in values)}
    manifest={"complete":True,"train":summary(train),"validation":summary(valid),"sampling_contract":"each optimizer step: four uniformly sampled boundary queries x 44 boundary states plus four uniformly sampled all-train retention queries x 44 matched-correct states; within-query sampling uniform with replacement","artifacts_sha256":{x:sha256(x) for x in (a.on_policy,a.orders,a.causal_details,a.train_ids,a.validation_ids,a.train_output,a.validation_output)},"locked_roles_used":False};write_metadata(a.manifest,manifest);print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
