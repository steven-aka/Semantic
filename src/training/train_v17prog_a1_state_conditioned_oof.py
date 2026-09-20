"""One train-only OOF test of fresh-target local swap utility from V8 states."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.training.train_v17b1_multi_anchor_viability import DummyLM

LEVELS = (.60, .70, .80, .90, .95)
SWAPS = (6, 7, 9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v17prog_a1_state_conditioned_oof.json")
    ap.add_argument("--candidates", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_top10_candidates.jsonl")
    ap.add_argument("--data", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl")
    ap.add_argument("--rollouts", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl")
    ap.add_argument("--embeddings", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/v8_train_embeddings.pt")
    ap.add_argument("--v8-checkpoint", default="results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/v8_run/checkpoints/step500")
    ap.add_argument("--p0-root", default="results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
    ap.add_argument("--a0-cache", default="results/v2_rank_then_cut/v17prog_a0_fresh_adjacent_swap_oracle/per_action.jsonl")
    ap.add_argument("--output-dir", default="results/v2_rank_then_cut/v17prog_a1_state_conditioned_oof")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["status"] != "FROZEN_TRAIN_ONLY" or cfg["primary_schedule"] != [6, 7, 7, 9, 10]:
        raise ValueError("unfrozen training protocol")
    source = {r["example_id"]: r for r in read_jsonl(args.candidates) if fold(r["example_id"]) != 4}
    queries = sorted(source)
    if len(queries) != 1421:
        raise AssertionError("unexpected train population")
    data = {r["example_id"]: r for r in read_jsonl(args.data) if r["example_id"] in source}
    orders = {r["example_id"]: r["decoded_order"] for r in read_jsonl(args.rollouts) if r["example_id"] in source}
    base = defaultdict(dict)
    for path in sorted(Path(args.p0_root).glob("shard_*_of_3/per_prefix.jsonl")):
        for r in read_jsonl(path):
            if r["depth"] in base[r["example_id"]]:
                raise AssertionError("duplicate P0 record")
            base[r["example_id"]][r["depth"]] = r
    action = {(r["example_id"], r["swap_depth"]): r for r in read_jsonl(args.a0_cache)}
    if len(base) != 1421 or len(action) != 4263 or any(len(base[q]) != 12 for q in queries):
        raise AssertionError("incomplete fresh outcomes")
    cache = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    cache_index = {q: i for i, q in enumerate(cache["example_ids"])}
    if any(q not in cache_index for q in queries):
        raise AssertionError("missing frozen V8 embedding")
    state = torch.load(Path(args.v8_checkpoint) / "sequential_head.pt", map_location="cpu", weights_only=True)
    model = SequentialPacketPolicy(DummyLM(int(state["input_projection.0.weight"].shape[1])),
                                   model_dim=int(state["initial_history.weight"].shape[0]))
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ids = torch.tensor([cache_index[q] for q in queries])
    packets = cache["packets"][ids].to(device=device, dtype=torch.float32)
    questions = cache["questions"][ids].to(device=device, dtype=torch.float32)
    fractions = torch.tensor([data[q]["packet_tokens"] for q in queries], device=device, dtype=torch.float32)
    fractions /= fractions.sum(1, keepdim=True)
    indices = torch.arange(len(queries), device=device)
    features = []
    with torch.no_grad():
        for depth in SWAPS:
            histories = torch.tensor([orders[q][:depth - 1] for q in queries], device=device, dtype=torch.long)
            lengths = torch.full((len(queries),), depth - 1, device=device, dtype=torch.long)
            history_state, selected = model.history_states(packets, questions, histories, lengths, indices)
            phi, _ = model.action_features(packets, questions, history_state, selected, indices, fractions)
            old_ids = torch.tensor([orders[q][depth - 1] for q in queries], device=device)
            new_ids = torch.tensor([orders[q][depth] for q in queries], device=device)
            delta = phi[indices, new_ids] - phi[indices, old_ids]
            features.append(delta.detach().cpu())
            print(json.dumps({"extracted_swap_depth": depth, "feature_dim": delta.shape[1]}), flush=True)
    x = torch.stack(features, dim=1).float().to(device)
    del model, packets, questions, cache

    def outcome(q, swap, depths):
        active = {float(v) for v in source[q]["attainable_levels"]}
        rows = [action[q, d] if swap == d else base[q][d] for d in depths]
        hits = [None if level not in active else r["f1"] + 1e-6 >= level
                for level, r in zip(LEVELS, rows)]
        return {"hits": hits, "complete": all(v is not False for v in hits),
                "tokens": sum(r["tokens"] for level, r in zip(LEVELS, rows) if level in active)}

    schedules = {"primary": cfg["primary_schedule"], "sensitivity": cfg["sensitivity_schedule"]}
    outcomes = {name: [[outcome(q, swap, depths) for swap in (None, *SWAPS)] for q in queries]
                for name, depths in schedules.items()}
    positive = torch.zeros((len(queries), 4), device=device, dtype=torch.bool)
    for i, rows in enumerate(outcomes["primary"]):
        base_hits = rows[0]["hits"]
        repairs = []
        for item in rows[1:]:
            breaks = any(a is True and b is False for a, b in zip(base_hits, item["hits"]))
            repairs.append(0 if breaks else sum(a is False and b is True for a, b in zip(base_hits, item["hits"])))
        maximum = max(repairs)
        if maximum == 0:
            positive[i, 0] = True
        else:
            for j, count in enumerate(repairs, 1):
                positive[i, j] = count == maximum
    folds = torch.tensor([stable_fold(q, 4) for q in queries], device=device)
    opt_cfg = cfg["optimizer"]
    decisions = [None] * len(queries)
    fold_info = []
    for held_out in range(4):
        train = torch.where(folds != held_out)[0]
        valid = torch.where(folds == held_out)[0]
        mu = x[train].mean((0, 1))
        sigma = x[train].std((0, 1)).clamp_min(1e-5)
        train_x = (x[train] - mu) / sigma
        valid_x = (x[valid] - mu) / sigma
        torch.manual_seed(opt_cfg["seed"] + held_out)
        head = torch.nn.Linear(x.shape[-1], 1).to(device)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.AdamW(head.parameters(), lr=opt_cfg["learning_rate"],
                                      weight_decay=opt_cfg["weight_decay"])
        for step in range(opt_cfg["steps"]):
            sampled = torch.randint(len(train), (opt_cfg["batch_queries"],), device=device)
            logits = torch.cat((torch.zeros((len(sampled), 1), device=device),
                                head(train_x[sampled]).squeeze(-1)), dim=1)
            target = positive[train[sampled]]
            loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~target, -1e9), dim=1)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = torch.cat((torch.zeros((len(valid), 1), device=device),
                                head(valid_x).squeeze(-1)), dim=1)
            choices = logits.argmax(dim=1).tolist()
        for index, choice in zip(valid.tolist(), choices):
            decisions[index] = choice
        fold_info.append({"fold": held_out, "train_queries": len(train), "valid_queries": len(valid),
                          "train_queries_with_repair_target": int((~positive[train, 0]).sum()),
                          "final_loss": float(loss.detach().cpu()),
                          "chosen_swaps": sum(choice != 0 for choice in choices)})
        print(json.dumps(fold_info[-1]), flush=True)
    if any(choice is None for choice in decisions):
        raise AssertionError("missing OOF decision")

    summary = {"protocol": cfg["protocol"], "queries": len(queries), "folds": fold_info,
               "target_repair_queries": int((~positive[:, 0]).sum()), "schedules": {},
               "limitations": "Train-side query-grouped OOF for new head; frozen V8 representation was learned on this lineage. No independent claim."}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_decisions.jsonl").open("w") as handle:
        for q, choice, f in zip(queries, decisions, folds.tolist()):
            handle.write(json.dumps({"example_id": q, "fold": f, "choice": (None, *SWAPS)[choice]}) + "\n")
    for name, rows in outcomes.items():
        baseline = [r[0] for r in rows]
        learned = [r[choice] for r, choice in zip(rows, decisions)]
        def stats(values):
            return {"success": {str(level): sum(r["hits"][i] is True for r in values) for i, level in enumerate(LEVELS)},
                    "complete": sum(r["complete"] for r in values),
                    "mean_cumulative_context_tokens": mean(r["tokens"] for r in values)}
        summary["schedules"][name] = {"v8": stats(baseline), "learned": stats(learned),
                                      "repairs": {str(level): sum(old["hits"][i] is False and new["hits"][i] is True
                                                                 for old, new in zip(baseline, learned)) for i, level in enumerate(LEVELS)},
                                      "breaks": {str(level): sum(old["hits"][i] is True and new["hits"][i] is False
                                                                for old, new in zip(baseline, learned)) for i, level in enumerate(LEVELS)},
                                      "complete_repairs": sum(not old["complete"] and new["complete"] for old, new in zip(baseline, learned)),
                                      "complete_breaks": sum(old["complete"] and not new["complete"] for old, new in zip(baseline, learned))}
    b = summary["schedules"]["primary"]["v8"]
    l = summary["schedules"]["primary"]["learned"]
    no_regressions = all(l["success"][str(level)] >= b["success"][str(level)] for level in LEVELS) and l["complete"] >= b["complete"]
    strict = any(l["success"][str(level)] > b["success"][str(level)] for level in LEVELS) or l["complete"] > b["complete"] or l["mean_cumulative_context_tokens"] < b["mean_cumulative_context_tokens"]
    summary["primary_gate"] = "GO_TRAIN_ONLY_PARETO_RESEARCH" if no_regressions and strict and l["mean_cumulative_context_tokens"] <= b["mean_cumulative_context_tokens"] else "STOP_PROG_A1_NO_OOF_PARETO"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
