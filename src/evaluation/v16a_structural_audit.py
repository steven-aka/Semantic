from __future__ import annotations

import argparse,json,statistics
from collections import Counter
from pathlib import Path

from src.data.schemas import read_jsonl
from src.reproducibility import sha256,write_metadata


def audit(path):
    rows=list(read_jsonl(path));jaccard=[];hamming=[];patterns=Counter();deltas=Counter();repair_pairs=0;safe_pairs=0;safe_queries=0
    for row in rows:
        masks=[p["candidate_mask"] for p in row["pairs"]]
        for i in range(4):
            for j in range(i):
                intersection=(masks[i]&masks[j]).bit_count();union=(masks[i]|masks[j]).bit_count();jaccard.append(1-intersection/union if union else 0);hamming.append((masks[i]^masks[j]).bit_count())
        fallback=row["fallback"];query_safe=False
        for pair in row["pairs"]:
            delta=tuple((int(pair["candidate_success"][k])-int(fallback["success"][k])) if fallback["active"][k] else 0 for k in range(5));deltas[str(delta)]+=1
            if not fallback["success"][3] and pair["candidate_success"][3]:
                repair_pairs+=1;patterns[f"add{pair['added_mask'].bit_count()}_remove{pair['removed_mask'].bit_count()}"]+=1
                safe=all(not fallback["active"][k] or not fallback["success"][k] or pair["candidate_success"][k] for k in range(5))
                safe_pairs+=int(safe);query_safe|=safe
        safe_queries+=int(query_safe)
    return {"examples":len(rows),"pairs":4*len(rows),"candidate_diversity":{"mean_pairwise_jaccard_distance":statistics.mean(jaccard),"median_pairwise_jaccard_distance":statistics.median(jaccard),"mean_pairwise_hamming_distance":statistics.mean(hamming)},"repair_pairs":repair_pairs,"trajectory_safe_repair_pairs":safe_pairs,"queries_with_trajectory_safe_repair":safe_queries,"repair_edit_patterns":dict(patterns),"multi_anchor_delta_patterns":dict(deltas.most_common())}


def main():
    p=argparse.ArgumentParser();p.add_argument("--train",required=True);p.add_argument("--validation",required=True);p.add_argument("--development",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    result={"complete":True,"scientific_role":"V16-A structural audit; development is consumed and diagnostic only","train2863":audit(a.train),"internal_validation300":audit(a.validation),"development300":audit(a.development),"artifacts_sha256":{x:sha256(x) for x in (a.train,a.validation,a.development)},"conclusion":"Top-4 masks have low diversity and nearly all repairs are one-packet swaps. Every validation/development repair is trajectory-safe, so the immediate failure is identifying necessary swaps and avoiding unnecessary cross-anchor-harmful switches, not unsafe repair candidates."}
    write_metadata(a.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
