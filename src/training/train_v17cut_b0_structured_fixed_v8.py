"""Set-valued structured stopping on frozen V8 trajectories."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.build_v17sel_c0_train_candidates import exact_values, outcome
from src.data.schemas import read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.training.train_v17h0_multianchor_cutoff import LEVELS, meets_fidelity


def prefix_masks(order: list[int]) -> list[int]:
    masks = [0]
    for packet in order:
        masks.append(masks[-1] | (1 << packet))
    if len(masks) != 13 or masks[-1] != 4095:
        raise ValueError("invalid V8 order")
    return masks


def oracle_optimal_paths(fidelity: list[float], tokens: list[int], levels: list[float]) -> tuple[int | None, list[tuple[int, ...]]]:
    """All minimum-token fidelity-valid nondecreasing stop vectors."""
    if not levels:
        return 0, [()]
    state: dict[int, tuple[int, list[tuple[int, ...]]]] = {
        depth: (tokens[depth], [(depth,)]) for depth in range(13)
        if meets_fidelity(fidelity[depth], levels[0])
    }
    for level in levels[1:]:
        next_state = {}
        for depth in range(13):
            if not meets_fidelity(fidelity[depth], level):
                continue
            parents = [(cost, paths) for old_depth, (cost, paths) in state.items() if old_depth <= depth]
            if not parents:
                continue
            cheapest = min(cost for cost, _ in parents)
            paths = [path + (depth,) for cost, group in parents if cost == cheapest for path in group]
            next_state[depth] = (cheapest + tokens[depth], paths)
        state = next_state
    if not state:
        return None, []
    cheapest = min(cost for cost, _ in state.values())
    paths = [path for cost, group in state.values() if cost == cheapest for path in group]
    return cheapest, paths


def log_partition(scores: torch.Tensor) -> torch.Tensor:
    """Log sum of exp(path score) over all nondecreasing stop vectors."""
    alpha = scores[0]
    for level in range(1, scores.shape[0]):
        alpha = torch.logcumsumexp(alpha, dim=0) + scores[level]
    return torch.logsumexp(alpha, dim=0)


def best_stop_vector(scores: torch.Tensor) -> tuple[int, ...]:
    """Viterbi using scores alone; never examines fidelity labels."""
    alpha = scores[0]
    back = []
    for level in range(1, scores.shape[0]):
        values = []
        parents = []
        for depth in range(13):
            best = int(torch.argmax(alpha[:depth + 1]).item())
            parents.append(best)
            values.append(alpha[best] + scores[level, depth])
        back.append(parents)
        alpha = torch.stack(values)
    depth = int(torch.argmax(alpha).item())
    path = [depth]
    for parents in reversed(back):
        depth = parents[depth]
        path.append(depth)
    return tuple(reversed(path))


class StructuredCutoff(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512 * 3 + 2, 128), nn.GELU(), nn.Linear(128, 5))

    def forward(self, questions: torch.Tensor, packets: torch.Tensor, prefix_tokens: torch.Tensor,
                full_tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bits = ((masks[..., None] >> torch.arange(12, device=masks.device)) & 1).float()
        selected = torch.einsum("btn,bnd->btd", bits, packets.float()) / bits.sum(-1).clamp_min(1)[..., None]
        question = questions.float()[:, None, :].expand_as(selected)
        depth = torch.arange(13, device=masks.device).float()[None, :, None].expand(len(masks), -1, -1) / 12
        token_fraction = (prefix_tokens / full_tokens[:, None].clamp_min(1))[..., None]
        features = torch.cat((question, selected, question * selected, depth, token_fraction), dim=-1)
        return self.net(features).transpose(1, 2)


def evaluate(model: StructuredCutoff, data: dict, indices: list[int], device: torch.device) -> dict:
    model.eval()
    per_anchor = [{"level": level, "eligible": 0, "success": 0,
                   "same_order_oracle_success": 0} for level in LEVELS]
    complete = 0
    oracle_complete = 0
    complete_cost = []
    complete_regret = []
    decisions = []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            batch_ids = indices[start:start + 128]
            scores = model(data["questions"][batch_ids].to(device), data["packets"][batch_ids].to(device),
                           data["tokens"][batch_ids].to(device), data["full_tokens"][batch_ids].to(device),
                           data["masks"][batch_ids].to(device)).cpu()
            for local, index in enumerate(batch_ids):
                active = data["active_indices"][index]
                path = best_stop_vector(scores[local, active])
                good = True
                cost = 0
                for anchor, depth in zip(active, path):
                    ok = meets_fidelity(data["fidelity"][index][depth], LEVELS[anchor])
                    per_anchor[anchor]["eligible"] += 1
                    per_anchor[anchor]["success"] += int(ok)
                    per_anchor[anchor]["same_order_oracle_success"] += int(data["oracle_success"][index][anchor])
                    good &= ok
                    cost += int(data["tokens"][index, depth])
                complete += int(good)
                oracle_complete += int(data["oracle_cost"][index] is not None)
                if good:
                    complete_cost.append(cost)
                    complete_regret.append((cost - data["oracle_cost"][index]) /
                                           (len(active) * data["full_tokens"][index].item()))
                decisions.append({"example_id": data["ids"][index], "stop_vector": path,
                                  "complete": bool(good), "cumulative_tokens": cost,
                                  "oracle_cumulative_tokens": data["oracle_cost"][index]})
    return {
        "queries": len(indices), "per_anchor": per_anchor, "complete": complete,
        "same_order_oracle_complete": oracle_complete,
        "mean_cumulative_tokens_on_complete": mean(complete_cost) if complete_cost else None,
        "mean_regret_on_complete": mean(complete_regret) if complete_regret else None,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["status"] != "FROZEN_TRAIN_ONCE":
        raise ValueError("protocol not frozen")
    rows = list(read_jsonl(args.candidates))
    if len(rows) != 2032 or len({row["example_id"] for row in rows}) != 2032:
        raise ValueError("unexpected clean-train population")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    ids = [row["example_id"] for row in rows]
    if cache["example_ids"] != ids:
        raise ValueError("embedding/candidate mismatch")
    orders = {row["example_id"]: row["decoded_order"] for row in read_jsonl(args.rollouts)}
    if set(orders) != set(ids):
        raise ValueError("rollout/candidate mismatch")
    masks = []
    prefix_tokens = []
    fidelity_rows = []
    active_indices = []
    optimal_paths = []
    oracle_cost = []
    oracle_success = []
    for position, row in enumerate(rows, 1):
        query = row["example_id"]
        mask_row = prefix_masks(orders[query])
        full_fidelity, full_tokens = exact_values(Path(args.exact_dir) / f"{query}.jsonl")
        fidelity = [full_fidelity[mask] for mask in mask_row]
        tokens = [full_tokens[mask] for mask in mask_row]
        active = [i for i, level in enumerate(LEVELS) if level in row["attainable_levels"]]
        cost, paths = oracle_optimal_paths(fidelity, tokens, [LEVELS[i] for i in active])
        cached = row["candidates"][0]
        if cost != cached["oracle_ordered_complete_cumulative_tokens"] or bool(paths) != cached["complete"]:
            raise ValueError(f"oracle stopping mismatch: {query}")
        if [any(meets_fidelity(f, level) for f in fidelity) for level in LEVELS] != cached["success"]:
            raise ValueError(f"candidate ceiling mismatch: {query}")
        masks.append(mask_row)
        prefix_tokens.append(tokens)
        fidelity_rows.append(fidelity)
        active_indices.append(active)
        optimal_paths.append(paths)
        oracle_cost.append(cost)
        oracle_success.append(cached["success"])
        if position % 500 == 0:
            print(json.dumps({"loaded": position, "total": len(rows)}), flush=True)
    data = {
        "ids": ids, "questions": cache["questions"], "packets": cache["packets"],
        "masks": torch.tensor(masks, dtype=torch.int32),
        "tokens": torch.tensor(prefix_tokens, dtype=torch.float32),
        "full_tokens": torch.tensor([r["full_tokens"] for r in rows], dtype=torch.float32),
        "fidelity": fidelity_rows, "active_indices": active_indices,
        "optimal_paths": optimal_paths, "oracle_cost": oracle_cost,
        "oracle_success": oracle_success,
    }
    train = [i for i, query in enumerate(ids) if fold(query) != 4 and optimal_paths[i]]
    exposed = [i for i, query in enumerate(ids) if fold(query) == 4]
    if len(exposed) != 611:
        raise ValueError("unexpected diagnostic split")
    opt = config["optimization"]
    random.seed(opt["seed"])
    torch.manual_seed(opt["seed"])
    torch.cuda.manual_seed_all(opt["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StructuredCutoff().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt["steps"])
    generator = torch.Generator().manual_seed(opt["seed"])
    order = torch.randperm(len(train), generator=generator).tolist()
    cursor = 0
    history = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, opt["steps"] + 1):
        batch = []
        while len(batch) < opt["batch_size"]:
            if cursor == len(order):
                order = torch.randperm(len(train), generator=generator).tolist()
                cursor = 0
            take = min(opt["batch_size"] - len(batch), len(order) - cursor)
            batch.extend(train[position] for position in order[cursor:cursor + take])
            cursor += take
        scores = model(data["questions"][batch].to(device), data["packets"][batch].to(device),
                       data["tokens"][batch].to(device), data["full_tokens"][batch].to(device),
                       data["masks"][batch].to(device))
        losses = []
        for local, index in enumerate(batch):
            active = data["active_indices"][index]
            current = scores[local, active]
            log_all = log_partition(current)
            paths = data["optimal_paths"][index]
            path_index = torch.tensor(paths, dtype=torch.long, device=device)
            log_good = torch.logsumexp(current[torch.arange(len(active), device=device)[None, :], path_index].sum(dim=1), dim=0)
            losses.append(log_all - log_good)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step in (1, 100, 200, opt["steps"]):
            history.append({"step": step, "loss": float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, output / "step400.pt")
    write_jsonl(output / "history.jsonl", history)
    result = evaluate(model, data, exposed, device)
    write_jsonl(output / "exposed_611_decisions.jsonl", result.pop("decisions"))
    result.update({
        "protocol": config["protocol"], "evaluation_role": "design-exposed C1 fold; not independent confirmation",
        "trained_complete_orders": len(train), "endpoint": opt["endpoint"],
        "holdout_581_read": False, "internal300_read": False, "development_read": False,
        "confirmation_read": False, "new_target_calls": 0,
        "artifacts": {"config_sha256": sha256(args.config), "candidates_sha256": sha256(args.candidates),
                      "cache_sha256": sha256(args.cache), "rollouts_sha256": sha256(args.rollouts)},
    })
    write_metadata(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
