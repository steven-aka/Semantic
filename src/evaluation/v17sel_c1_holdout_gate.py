"""One design-exposed holdout read for frozen top-4/top-10 selector endpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch

from src.data.schemas import write_jsonl
from src.model.topk_set_selector import TopKSetSelector
from src.reproducibility import sha256, write_metadata
from src.training.train_v17sel_c1_set_selector import load_features, metrics


def p90(values: list[int]) -> int | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(.9 * len(ordered)) - 1)] if ordered else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--b2b-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train = json.loads((Path(args.train_output) / "summary.json").read_text())
    if train["decision"] != "GO_SEL_C1_ONE_DESIGN_EXPOSED_HOLDOUT_READ":
        raise ValueError("train-only gate did not authorize holdout read")
    rows, features = load_features(args.candidates, args.cache)
    if len(rows) != 581:
        raise ValueError("unexpected lineage-clean holdout size")
    original = json.loads(Path(args.b2b_summary).read_text())
    for k in (4, 10):
        reproduced = sum(any(item["success"][3] for item in row["candidates"][:k + 1]) for row in rows)
        if reproduced != original["oracle_pool_090_by_top_k"][str(k)]:
            raise ValueError(f"B2B oracle pool mismatch at top{k}: {reproduced}")
    device = torch.device("cuda:0")
    choices = {}
    summaries = {}
    for arm, k in config["candidate_budget"]["arms"].items():
        selected = json.loads((Path(args.train_output) / arm / "selected.json").read_text())
        model = TopKSetSelector().to(device)
        model.load_state_dict(torch.load(selected["checkpoint"], map_location=device, weights_only=True))
        model.eval()
        choice = []
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                score = model(features[start:start + 128, :k + 1].to(device))
                choice.extend(torch.argmax(score, dim=1).cpu().tolist())
        choices[arm] = choice
        report = metrics(rows, choice, k)
        tokens = [row["candidates"][rank]["earliest_tokens"][3] for row, rank in zip(rows, choice) if row["candidates"][rank]["success"][3]]
        report.update(median_earliest_090_tokens_successful=median(tokens) if tokens else None, p90_earliest_090_tokens_successful=p90(tokens))
        summaries[arm] = report
    a = choices["top4"]
    b = choices["top10"]
    repairs = sum(not row["candidates"][i]["success"][3] and row["candidates"][j]["success"][3] for row, i, j in zip(rows, a, b))
    breaks = sum(row["candidates"][i]["success"][3] and not row["candidates"][j]["success"][3] for row, i, j in zip(rows, a, b))
    top4 = summaries["top4"]
    top10 = summaries["top10"]
    gate = config["holdout_research_gate"]
    passed = (
        top10["success_090"] >= math.ceil(gate["top10_min_090_success_fraction"] * len(rows))
        and top10["success_090"] - top4["success_090"] >= gate["top10_vs_top4_same_architecture_min_net_successes"]
        and top10["complete"] - top4["complete"] >= gate["top10_vs_top4_complete_min_delta"]
        and top10["mean_penalized_090_token_fraction"] - top4["mean_penalized_090_token_fraction"] <= gate["top10_vs_top4_mean_penalized_090_token_fraction_max_delta"]
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_query.jsonl", [
        {"example_id": row["example_id"], "top4_selected_rank": i, "top10_selected_rank": j,
         "top4_success_090": row["candidates"][i]["success"][3],
         "top10_success_090": row["candidates"][j]["success"][3],
         "top4_earliest_090_tokens": row["candidates"][i]["earliest_tokens"][3],
         "top10_earliest_090_tokens": row["candidates"][j]["earliest_tokens"][3]}
        for row, i, j in zip(rows, a, b)
    ])
    summary = {
        "decision": "GO_FREEZE_SEL_C1_CANDIDATE_STRATEGY" if passed else gate["otherwise"],
        "role": "design-exposed lineage-clean research holdout, not confirmation",
        "queries": len(rows), "top4": top4, "top10": top10, "repairs": repairs, "breaks": breaks,
        "v8_fallback": metrics(rows, [0] * len(rows), 0),
        "v13_rank1": metrics(rows, [1] * len(rows), 1),
        "oracle_pool_090": original["oracle_pool_090_by_top_k"],
        "development_read": False, "confirmation_read": False, "cutoff_trained": False,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates), "cache_sha256": sha256(args.cache)},
    }
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
