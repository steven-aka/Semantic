"""Measure exact prompt-prefix KV reuse headroom for the best C1 diagnostic point."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.target.qampari_runner import QAMPARI_SYSTEM_PROMPT, build_qampari_prompt


TRACE = Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence")
DATA = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
MODEL = "models/Qwen3-8B"


def main() -> None:
    summary = json.loads((TRACE / "summary.json").read_text())
    point = summary["diagnosis"]["best_native_quality_context_point"]
    if point is None:
        raise AssertionError("no qualifying native-confidence point")
    scores = {(r["example_id"], float(r["tau"])): r for r in read_jsonl(TRACE / "oof_scores.jsonl")}
    traces = {(r["example_id"], int(r["depth"])): r for r in read_jsonl(TRACE / "traces.jsonl")}
    qids = sorted({q for q, _ in traces})
    data = {r["example_id"]: r for r in read_jsonl(DATA) if r["example_id"] in set(qids)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)

    prompt_cache = {}
    def prompt_ids(qid, depth):
        key = (qid, depth)
        if key in prompt_cache:
            return prompt_cache[key]
        state = traces[key]
        context = "\n\n".join(text.strip() for i, text in enumerate(data[qid]["packet_texts"])
                                if state["mask"] & (1 << i))
        messages = [{"role": "system", "content": QAMPARI_SYSTEM_PROMPT},
                    {"role": "user", "content": build_qampari_prompt(data[qid]["question"], context)}]
        out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prompt_cache[key] = out
        return out

    depths = {0.6: 6, 0.7: 7, 0.8: 7, 0.9: 9}
    by_depth = defaultdict(list)
    saved_prefill = 0
    fallback_requests = 0
    threshold = float(point["threshold"])
    for qid in qids:
        late = prompt_ids(qid, 10)
        for tau, depth in depths.items():
            early = prompt_ids(qid, depth)
            lcp = 0
            for a, b in zip(early, late):
                if a != b:
                    break
                lcp += 1
            by_depth[depth].append((len(early), len(late), lcp))
            stop = scores[(qid, tau)]["B1_native_only"] >= threshold
            if not stop:
                fallback_requests += 1
                saved_prefill += lcp

    depth_stats = {}
    for depth, rows in sorted(by_depth.items()):
        depth_stats[str(depth)] = {
            "pairs": len(rows),
            "mean_early_prompt_tokens": sum(x[0] for x in rows) / len(rows),
            "mean_depth10_prompt_tokens": sum(x[1] for x in rows) / len(rows),
            "mean_longest_common_prefix_tokens": sum(x[2] for x in rows) / len(rows),
            "mean_depth10_prompt_fraction_reusable": sum(x[2] / x[1] for x in rows) / len(rows)
        }
    current_extra = -float(point["target_compute_tokens_saved_vs_depth10"])
    optimistic_saving = saved_prefill / len(qids)
    residual_extra = current_extra - optimistic_saving
    out = {
        "protocol": "V17-STOP-C2-COST_PREFIX_STATE_REUSE_PREFLIGHT",
        "queries": len(qids),
        "policy": "B1 native-confidence OOF threshold 0.95",
        "prompt_rendering": "Canonical context is re-rendered in original packet-index order; later states are not simple token append operations.",
        "per_depth": depth_stats,
        "fallback_requests": fallback_requests,
        "current_extra_target_tokens_per_query_vs_depth10": current_extra,
        "optimistic_exact_lcp_prefill_tokens_saved_per_query": optimistic_saving,
        "residual_extra_target_tokens_per_query_vs_depth10": residual_extra,
        "exact_current_prompt_kv_reuse_restores_compute_pareto": residual_extra <= 0,
        "decision": "STOP_C2_COST_UNDER_CURRENT_PROMPT_RENDERING" if residual_extra > 0 else "GO_C2_COST_IMPLEMENTATION",
        "next_contract_question": "Changing to V8 reveal-order append-only rendering could increase KV reuse but changes token order and Target behavior; it requires a new paired quality/cost baseline rather than being treated as a serving-only optimization.",
        "new_target_calls": 0,
        "sealed_sets_read": False
    }
    (TRACE / "c2_cost_preflight.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
