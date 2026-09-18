from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import head_state_dict, load_model
from src.evaluation.v16c0_first_irreversible_divergence import trajectory_levels
from src.model.sequential_packet_policy import build_sequential_encoder_input
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.search.sequential_trajectory_dp import SequentialTrajectoryDP


def read_jsonl_gzip(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mask_reached(dp: SequentialTrajectoryDP, history: Sequence[int]) -> tuple[int, int]:
    mask, reached = 0, dp.attained[0]
    for packet in history:
        mask |= 1 << int(packet)
        reached = max(reached, dp.attained[mask])
    return mask, reached


def prepare_records(details: Sequence[dict[str, Any]], rows: dict[str, dict[str, Any]], exact_dir: str | Path) -> list[dict[str, Any]]:
    output = []
    for detail in details:
        depth = detail["first_primary_090_extinction_depth"]
        if depth is None:
            continue
        source = rows[detail["example_id"]]
        exact = list(read_jsonl(Path(exact_dir) / f"{detail['example_id']}.jsonl", ExactSearchResult))
        dp = SequentialTrajectoryDP(exact, trajectory_levels(source))
        primary = next(i for i, level in enumerate(dp.levels) if abs(level - 0.9) < 1e-9)
        parents = [()] if depth == 1 else [tuple(x) for x in detail["trace"][depth - 2]["histories"]]
        retention_depth = max(0, depth - 2)
        retention = [()] if retention_depth == 0 else [tuple(x) for x in detail["trace"][retention_depth - 1]["histories"]]
        parent_rows = []
        for history in parents:
            mask, reached = mask_reached(dp, history)
            viable = []
            for packet in range(12):
                if mask & (1 << packet):
                    continue
                next_mask = mask | (1 << packet)
                next_reached = max(reached, dp.attained[next_mask])
                if dp.value(next_mask, next_reached).reached_levels >= primary + 1:
                    viable.append(packet)
            parent_rows.append({"history": history, "viable_actions": tuple(viable), "optimal_actions": dp.optimal_actions(mask, reached)})
        if not any(parent["viable_actions"] for parent in parent_rows):
            raise RuntimeError(f"FID {detail['example_id']} has no viable expansion")
        output.append({"example_id": detail["example_id"], "source": source, "fid_depth": depth, "parents": parent_rows, "retention": retention})
    return output


def score_histories(model: Any, packets: torch.Tensor, question: torch.Tensor, fractions: torch.Tensor, histories: Sequence[Sequence[int]]) -> torch.Tensor:
    width = max((len(x) for x in histories), default=0)
    tensor = torch.full((len(histories), width), -1, dtype=torch.long, device=packets.device)
    lengths = torch.tensor([len(x) for x in histories], dtype=torch.long, device=packets.device)
    for index, history in enumerate(histories):
        if history:
            tensor[index, :len(history)] = torch.tensor(history, dtype=torch.long, device=packets.device)
    indices = torch.zeros(len(histories), dtype=torch.long, device=packets.device)
    states, selected = model.history_states(packets[None], question[None], tensor, lengths, indices)
    logits, _ = model.score_states(packets[None], question[None], states, selected, indices, fractions[None])
    return logits


def sequence_and_parent_logits(model: Any, packets: torch.Tensor, question: torch.Tensor, fractions: torch.Tensor, parents: Sequence[dict[str, Any]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    prefixes = sorted(
        {tuple(parent["history"][:depth]) for parent in parents for depth in range(len(parent["history"]) + 1)},
        key=lambda x: (len(x), x),
    )
    logits = score_histories(model, packets, question, fractions, prefixes)
    by_prefix = {history: logits[index] for index, history in enumerate(prefixes)}
    scores, parent_logits = [], []
    for parent in parents:
        history = tuple(parent["history"])
        score = torch.zeros((), device=packets.device, dtype=torch.float32)
        for depth, action in enumerate(history):
            score = score + torch.log_softmax(by_prefix[history[:depth]].float(), dim=-1)[int(action)]
        scores.append(score)
        parent_logits.append(by_prefix[history])
    return scores, parent_logits


def beam_survival_loss(parent_scores: Sequence[torch.Tensor], parent_logits: Sequence[torch.Tensor], parents: Sequence[dict[str, Any]], *, beam_width: int, margin: float) -> tuple[torch.Tensor, dict[str, float]]:
    expansions, positives = [], []
    dp_losses = []
    for score, logits, parent in zip(parent_scores, parent_logits, parents):
        history = set(parent["history"])
        remaining = [packet for packet in range(12) if packet not in history]
        log_probs = torch.log_softmax(logits.float()[remaining], dim=-1)
        by_action = {packet: score + log_probs[index] for index, packet in enumerate(remaining)}
        expansions.extend(by_action.values())
        positives.extend(by_action[action] for action in parent["viable_actions"])
        optimal_indices = [remaining.index(action) for action in parent["optimal_actions"]]
        dp_losses.append(torch.logsumexp(log_probs, dim=0) - torch.logsumexp(log_probs[optimal_indices], dim=0))
    expanded = torch.stack(expansions)
    kth = torch.topk(expanded, min(beam_width, len(expansions))).values[-1]
    best_positive = torch.stack(positives).max()
    survival = F.relu(torch.as_tensor(margin, device=expanded.device) + kth - best_positive)
    return survival, {"survival": float(survival.detach()), "margin": float((best_positive - kth).detach()), "dp": float(torch.stack(dp_losses).mean().detach())}


def retention_kl(current_logits: torch.Tensor, teacher_probabilities: Sequence[torch.Tensor], histories: Sequence[Sequence[int]]) -> torch.Tensor:
    losses = []
    for index, history in enumerate(histories):
        remaining = [packet for packet in range(12) if packet not in set(history)]
        current = torch.log_softmax(current_logits[index, remaining].float(), dim=-1)
        teacher = teacher_probabilities[index].to(current.device)
        losses.append(F.kl_div(current, teacher, reduction="sum"))
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V16-C1 failure-triggered beam survival policy")
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--fid-details", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-optimizer-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--survival-margin", type=float, default=.1)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--dp-weight", type=float, default=.1)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol_config).read_text())
    expected = protocol["arguments"]
    mismatch = {key: (getattr(args, key), value) for key, value in expected.items() if getattr(args, key) != value}
    if protocol["status"] != "APPROVED_TO_RUN" or mismatch:
        raise ValueError({"status": protocol["status"], "mismatch": mismatch})
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0"); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_metadata(out / "protocol_snapshot.json", protocol)
    tokenizer, model = load_model(args)
    for parameter in model.lm.parameters(): parameter.requires_grad = False
    rows = {row["example_id"]: row for row in read_jsonl(args.train_data)}
    records = prepare_records(read_jsonl_gzip(args.fid_details), rows, args.exact_dir)
    inputs = {record["example_id"]: build_sequential_encoder_input(tokenizer, record["source"]["question"], record["source"]["packet_texts"], max_sequence_length=args.max_sequence_length) for record in records}
    # Freeze the initial policy distribution on same-rollout states before any update.
    teacher: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            packets, questions = model.encode([inputs[x["example_id"]] for x in batch], pad_token_id=int(tokenizer.pad_token_id), device=device)
            for index, record in enumerate(batch):
                fractions = torch.tensor(record["source"]["packet_tokens"], device=device, dtype=torch.float32); fractions /= fractions.sum()
                logits = score_histories(model, packets[index], questions[index], fractions, record["retention"])
                teacher[record["example_id"]] = [torch.softmax(logits[j, [p for p in range(12) if p not in set(history)]].float(), dim=-1).cpu() for j, history in enumerate(record["retention"])]
    params = list(model.head_parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_optimizer_steps)
    rng = random.Random(args.seed); history_rows = []; started = time.perf_counter()
    for step in range(1, args.max_optimizer_steps + 1):
        model.train(); batch = rng.sample(records, args.batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packets, questions = model.encode([inputs[x["example_id"]] for x in batch], pad_token_id=int(tokenizer.pad_token_id), device=device)
            losses = []; diagnostics = []
            for index, record in enumerate(batch):
                fractions = torch.tensor(record["source"]["packet_tokens"], device=device, dtype=torch.float32); fractions /= fractions.sum()
                scores, logits = sequence_and_parent_logits(model, packets[index], questions[index], fractions, record["parents"])
                survival, diag = beam_survival_loss(scores, logits, record["parents"], beam_width=args.beam_width, margin=args.survival_margin)
                retention_logits = score_histories(model, packets[index], questions[index], fractions, record["retention"])
                retention = retention_kl(retention_logits, teacher[record["example_id"]], record["retention"])
                dp = torch.stack([
                    torch.logsumexp(torch.log_softmax(logit.float(), dim=-1), dim=0) - torch.logsumexp(torch.log_softmax(logit.float(), dim=-1)[list(parent["optimal_actions"])], dim=0)
                    for logit, parent in zip(logits, record["parents"])
                ]).mean()
                losses.append(survival + args.retention_weight * retention + args.dp_weight * dp)
                diagnostics.append({**diag, "retention": float(retention.detach())})
            loss = torch.stack(losses).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            row = {"optimizer_step": step, "loss": float(loss.detach()), **{key: mean([x[key] for x in diagnostics]) for key in diagnostics[0]}}
            history_rows.append(row); write_jsonl(out / "training_history.jsonl", history_rows); print(json.dumps(row), flush=True)
    torch.save(head_state_dict(model), out / "sequential_head.pt")
    metadata = experiment_metadata(stage="v16c1_failure_triggered_beam_survival", base_checkpoint=args.checkpoint, train_data_sha256=sha256(args.train_data), fid_details_sha256=sha256(args.fid_details), sequential_head_sha256=sha256(out / "sequential_head.pt"), fid_examples=len(records), optimizer_steps=args.max_optimizer_steps, backbone_and_lora_frozen=True, complete_sequential_head_trainable=True, elapsed_seconds=time.perf_counter() - started, development300_used=False, locked_roles_used=False)
    write_metadata(out / "training_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
