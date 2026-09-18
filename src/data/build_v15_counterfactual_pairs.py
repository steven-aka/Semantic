from __future__ import annotations

import argparse, json
from pathlib import Path

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask


NAMES = ("both_fail", "repair", "break", "both_success")


def prefix_masks(order):
    masks=[0]
    for packet in order:masks.append(masks[-1] | (1 << packet))
    return masks


def main():
    p=argparse.ArgumentParser();p.add_argument("--candidates",required=True);p.add_argument("--rollouts",required=True);p.add_argument("--exact-dir",required=True);p.add_argument("--output",required=True);p.add_argument("--manifest",required=True);a=p.parse_args()
    rows=list(read_jsonl(a.candidates));orders={r["example_id"]:r["decoded_order"] for r in read_jsonl(a.rollouts)};output=[];counts={name:0 for name in NAMES}
    for number,row in enumerate(rows,1):
        eid=row["example_id"];order=orders[eid];exact=list(read_jsonl(Path(a.exact_dir)/f"{eid}.jsonl",ExactSearchResult));by={state_to_mask(x.state):x for x in exact}
        base_prefixes=prefix_masks(order);base_max=max(float(by[m].fidelity) for m in base_prefixes);base90=row["candidates"][0]["success"][3];pairs=[]
        for candidate in row["candidates"][1:]:
            mask=int(candidate["mask"]);size=mask.bit_count();matched=base_prefixes[size];candidate_order=[x for x in order if mask & (1<<x)]+[x for x in order if not mask & (1<<x)];candidate_max=max(float(by[m].fidelity) for m in prefix_masks(candidate_order));candidate90=candidate["success"][3]
            cls=1 if not base90 and candidate90 else 2 if base90 and not candidate90 else 3 if base90 and candidate90 else 0;counts[NAMES[cls]]+=1
            pairs.append({"rank":candidate["rank"],"candidate_mask":mask,"matched_fallback_mask":matched,"added_mask":mask & ~matched,"removed_mask":matched & ~mask,"retrieval_logits":candidate["retrieval_logits"],"mask_token_fraction":candidate["mask_tokens"]/max(1,row["full_tokens"]),"class_index":cls,"class_name":NAMES[cls],"delta_max_fidelity":candidate_max-base_max,"candidate_success":candidate["success"],"active":candidate["active"],"candidate_complete":candidate["complete"],"candidate_cumulative_tokens":candidate["cumulative_tokens"]})
        output.append({"example_id":eid,"full_tokens":row["full_tokens"],"fallback":row["candidates"][0],"pairs":pairs})
        if number%500==0:print(json.dumps({"built":number,"total":len(rows)}),flush=True)
    write_jsonl(a.output,output);write_metadata(a.manifest,{"complete":True,"examples":len(output),"pair_class_counts":counts,"candidates_sha256":sha256(a.candidates),"rollouts_sha256":sha256(a.rollouts),"output_sha256":sha256(a.output),"locked_roles_used":False})
    print(json.dumps(counts),flush=True)

if __name__=="__main__":main()
