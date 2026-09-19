from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.search.rank_then_cut import validate_packet_order


LEVELS = (.6, .7, .8, .9, .95)
WIDTH = 12
CHOICES = 5
PREFIXES = WIDTH + 1
FIDELITY_EPS = 1e-6


def meets_fidelity(value: float, level: float) -> bool:
    """Match float32 tensor labels when exact-cache decimals lie on a threshold."""
    return float(value) + FIDELITY_EPS >= level


def project(order: list[int], mask: int) -> list[int]:
    return [packet for packet in order if mask & (1 << packet)] + [packet for packet in order if not mask & (1 << packet)]


def read_exact(path: Path) -> tuple[list[float], list[int]]:
    fidelity = [0.0] * (1 << WIDTH)
    tokens = [0] * (1 << WIDTH)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask = state_to_mask(row["state"])
            fidelity[mask] = float(row["fidelity"])
            tokens[mask] = int(row["tokens"])
    return fidelity, tokens


class PrefixData:
    def __init__(self, candidates_path: str, cache_path: str, rollouts_path: str, oracle_path: str, exact_dir: str):
        self.rows = list(read_jsonl(candidates_path))
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.ids = [row["example_id"] for row in self.rows]
        if cache["example_ids"] != self.ids:
            raise ValueError("candidate/embedding order mismatch")
        self.packets = cache["packets"]
        self.questions = cache["questions"]
        orders = {row["example_id"]: validate_packet_order(row["decoded_order"], WIDTH) for row in read_jsonl(rollouts_path)}
        oracle = {row["example_id"]: int(row["oracle_cumulative_tokens"]) for row in read_jsonl(oracle_path)}
        self.oracle_tokens = torch.tensor([oracle[query] for query in self.ids], dtype=torch.float32)
        n = len(self.rows)
        self.masks = torch.empty((n, CHOICES, PREFIXES), dtype=torch.int32)
        self.fidelity = torch.empty((n, CHOICES, PREFIXES), dtype=torch.float32)
        self.tokens = torch.empty((n, CHOICES, PREFIXES), dtype=torch.float32)
        self.active = torch.tensor([[bool(value) for value in row["candidates"][0]["active"]] for row in self.rows], dtype=torch.bool)
        self.full_tokens = torch.tensor([row["full_tokens"] for row in self.rows], dtype=torch.float32)
        for index, row in enumerate(self.rows):
            query = row["example_id"]
            full_fidelity, full_tokens = read_exact(Path(exact_dir) / f"{query}.jsonl")
            base = list(orders[query])
            if int(row["full_tokens"]) != full_tokens[-1]:
                raise ValueError("full-context token count mismatch")
            for candidate_index, candidate in enumerate(row["candidates"]):
                order = base if candidate_index == 0 else project(base, int(candidate["mask"]))
                mask = 0
                for depth in range(PREFIXES):
                    if depth:
                        mask |= 1 << order[depth - 1]
                    self.masks[index, candidate_index, depth] = mask
                    self.fidelity[index, candidate_index, depth] = full_fidelity[mask]
                    self.tokens[index, candidate_index, depth] = full_tokens[mask]
                for anchor, level in enumerate(LEVELS):
                    observed = bool((self.fidelity[index, candidate_index] + FIDELITY_EPS >= level).any())
                    if observed != bool(candidate["success"][anchor]):
                        raise ValueError(f"candidate oracle ceiling mismatch {query} candidate {candidate_index} anchor {anchor}")
            if (index + 1) % 500 == 0:
                print(json.dumps({"prefix_data": index + 1, "total": n}), flush=True)

    def batch(self, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name)[indices].to(device) for name in ("packets", "questions", "masks", "fidelity", "tokens", "active", "full_tokens")}


class MultiAnchorCutoff(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512 * 3 + 3, 128), nn.GELU(), nn.Linear(128, len(LEVELS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        packets = batch["packets"].float()
        question = batch["questions"].float()
        masks = batch["masks"]
        bits = ((masks[..., None] >> torch.arange(WIDTH, device=masks.device)) & 1).float()
        selected = torch.einsum("bctn,bnd->bctd", bits, packets) / bits.sum(-1).clamp_min(1)[..., None]
        query = question[:, None, None, :].expand_as(selected)
        count = bits.sum(-1, keepdim=True) / WIDTH
        token_fraction = batch["tokens"][..., None] / batch["full_tokens"][:, None, None, None].clamp_min(1)
        candidate_rank = torch.arange(CHOICES, device=masks.device)[None, :, None, None].expand(*masks.shape, 1).float() / (CHOICES - 1)
        features = torch.cat((query, selected, query * selected, count, token_fraction, candidate_rank), -1)
        return self.net(features)


def cutoff_counts(probabilities: torch.Tensor, active: torch.Tensor, threshold: float) -> list[int | None]:
    previous = 0
    selected = []
    for anchor in range(len(LEVELS)):
        if not bool(active[anchor]):
            selected.append(None)
            continue
        eligible = list(range(previous, PREFIXES))
        passing = [count for count in eligible if float(probabilities[count, anchor]) >= threshold]
        count = passing[0] if passing else max(eligible, key=lambda value: (float(probabilities[value, anchor]), -value))
        selected.append(count)
        previous = count
    return selected


def candidate_oracle_index(row: dict[str, Any]) -> int:
    active = [bool(value) for value in row["candidates"][0]["active"]]
    return max(range(CHOICES), key=lambda index: (int(row["candidates"][index]["complete"]), int(row["candidates"][index]["success"][3]), -int(row["candidates"][index]["cumulative_tokens"]), -index))


def summarize_predictions(data: PrefixData, probabilities: torch.Tensor, chosen_indices: list[int], threshold: float) -> dict[str, Any]:
    per_anchor = [{"examples": 0, "successes": 0, "oracle_same_order_successes": 0} for _ in LEVELS]
    complete = oracle_complete = 0
    regrets = []
    details = []
    for index, candidate_index in enumerate(chosen_indices):
        row = data.rows[index]
        candidate = row["candidates"][candidate_index]
        active = data.active[index]
        cutoffs = cutoff_counts(probabilities[index, candidate_index], active, threshold)
        successes = []
        selected_tokens = []
        for anchor, count in enumerate(cutoffs):
            if count is None:
                successes.append(None)
                selected_tokens.append(None)
                continue
            success = meets_fidelity(float(data.fidelity[index, candidate_index, count]), LEVELS[anchor])
            successes.append(success)
            selected_tokens.append(int(data.tokens[index, candidate_index, count]))
            per_anchor[anchor]["examples"] += 1
            per_anchor[anchor]["successes"] += int(success)
            per_anchor[anchor]["oracle_same_order_successes"] += int(candidate["success"][anchor])
        is_complete = all(value is not False for value in successes)
        complete += int(is_complete)
        oracle_complete += int(candidate["complete"])
        regret = None
        if is_complete:
            regret = (sum(value for value in selected_tokens if value is not None) - float(data.oracle_tokens[index])) / (sum(active.tolist()) * max(1, int(data.full_tokens[index])))
            regrets.append(regret)
        details.append({"example_id": row["example_id"], "candidate_index": candidate_index, "cutoffs": cutoffs, "success": successes, "complete": is_complete, "same_order_oracle_complete": bool(candidate["complete"]), "regret": regret})
    return {"per_anchor": per_anchor, "complete": complete, "oracle_same_order_complete": oracle_complete, "mean_regret_if_complete": mean(regrets) if regrets else None, "details": details}


def predict_all(model: MultiAnchorCutoff, data: PrefixData, indices: list[int], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = data.batch(indices[start:start + batch_size], device)
            chunks.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--internal-candidates", required=True)
    parser.add_argument("--internal-cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--internal-oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--g0a-results", required=True)
    parser.add_argument("--v14-internal-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN":
        raise ValueError("unfrozen protocol")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = PrefixData(args.train_candidates, args.train_cache, args.rollouts, args.train_oracle, args.exact_dir)
    holdout_indices = [index for index, query in enumerate(train.ids) if stable_fold(query, 5) == 0]
    train_indices = [index for index, query in enumerate(train.ids) if stable_fold(query, 5) != 0]
    unstable = {(row["example_id"], int(row["mask"])) for row in read_jsonl(args.g0a_results) if any(item["success"] != row["cached_success"] for item in row["reruns"])}
    ambiguous = torch.zeros(train.masks.shape, dtype=torch.bool)
    for index, query in enumerate(train.ids):
        for mask in train.masks[index].unique().tolist():
            if (query, mask) in unstable:
                ambiguous[index] |= train.masks[index] == mask
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = MultiAnchorCutoff().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history = []
    for step in range(1, config["steps"] + 1):
        indices = random.sample(train_indices, config["batch_queries"])
        batch = train.batch(indices, device)
        logits = model(batch)
        target = batch["fidelity"][..., None] + FIDELITY_EPS >= torch.tensor(LEVELS, device=device)
        valid = batch["active"][:, None, None, :].expand_as(target).clone()
        valid[..., 3] &= ~ambiguous[indices].to(device)
        loss = F.binary_cross_entropy_with_logits(logits[valid], target.float()[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            record = {"step": step, "train_loss": float(loss.detach())}
            history.append(record)
            print(json.dumps(record), flush=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, out / "cutoff_step400.pt")
    write_jsonl(out / "history.jsonl", history)
    holdout = PrefixDataSubset(train, holdout_indices)
    heldout_prob = predict_all(model, train, holdout_indices, device)
    heldout_results = evaluate_role(holdout, heldout_prob, [0] * len(holdout.rows), config["sufficiency_probability_threshold"])
    internal = PrefixData(args.internal_candidates, args.internal_cache, args.rollouts, args.internal_oracle, args.exact_dir)
    internal_prob = predict_all(model, internal, list(range(len(internal.rows))), device)
    choices = {row["example_id"]: int(row["selected_index"]) for row in read_jsonl(args.v14_internal_selection)}
    if set(choices) != set(internal.ids):
        raise ValueError("V14 selector choice population mismatch")
    selected = [choices[query] for query in internal.ids]
    internal_results = {
        "v8_fallback": evaluate_role(internal, internal_prob, [0] * len(internal.rows), config["sufficiency_probability_threshold"]),
        "v14_frozen_selector": evaluate_role(internal, internal_prob, selected, config["sufficiency_probability_threshold"]),
        "oracle_candidate_diagnostic": evaluate_role(internal, internal_prob, [candidate_oracle_index(row) for row in internal.rows], config["sufficiency_probability_threshold"]),
    }
    write_jsonl(out / "internal_v14_selector_details.jsonl", internal_results["v14_frozen_selector"].pop("details"))
    for key in ("v8_fallback", "oracle_candidate_diagnostic"):
        internal_results[key].pop("details")
    heldout_results.pop("details")
    summary = {"complete": True, "train_queries": len(train_indices), "heldout_queries": len(holdout_indices), "masked_known_unstable_090_prefixes": int(ambiguous[train_indices].sum()), "heldout_v8_fallback": heldout_results, "internal_design_exposed": internal_results, "decision": "CUTOFF_PILOT_COMPLETE_NO_CONFIRMATION", "development_used": False, "confirmation_used": False, "artifacts": {"protocol_sha256": sha256(args.protocol_config), "train_candidates_sha256": sha256(args.train_candidates), "internal_candidates_sha256": sha256(args.internal_candidates), "g0a_results_sha256": sha256(args.g0a_results)}}
    write_metadata(out / "summary.json", summary)
    write_metadata(out / "decision.json", {"decision": summary["decision"], "development_used": False, "confirmation_used": False, "fresh_confirmation_authorized": False})
    print(json.dumps(summary, indent=2))


class PrefixDataSubset:
    def __init__(self, parent: PrefixData, indices: list[int]):
        self.rows = [parent.rows[index] for index in indices]
        self.ids = [parent.ids[index] for index in indices]
        self.active = parent.active[indices]
        self.fidelity = parent.fidelity[indices]
        self.tokens = parent.tokens[indices]
        self.oracle_tokens = parent.oracle_tokens[indices]
        self.full_tokens = parent.full_tokens[indices]


def evaluate_role(data: Any, probabilities: torch.Tensor, indices: list[int], threshold: float) -> dict[str, Any]:
    return summarize_predictions(data, probabilities, indices, threshold)


if __name__ == "__main__":
    main()
