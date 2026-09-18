from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.sequential_policy_evaluation import head_state_dict, load_head_state
from src.model.high_fidelity_retrieval_loss import high_fidelity_retrieval_loss
from src.model.mask_value_head import MaskValueHead
from src.model.sequential_packet_policy import SequentialPacketPolicy
from src.reproducibility import experiment_metadata, sha256, write_metadata
from src.training.mask_value_data import MaskValueCollator, MaskValueDataset


def load_trainable_encoder(args):
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lm = PeftModel.from_pretrained(base, Path(args.checkpoint) / "adapter", is_trainable=True)
    lm.config.use_cache = False
    model = SequentialPacketPolicy(lm, model_dim=args.model_dim)
    state = torch.load(Path(args.checkpoint) / "sequential_head.pt", map_location="cpu", weights_only=True)
    load_head_state(model, state)
    model.move_head(device="cuda:0", dtype=torch.bfloat16)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("lm.") and "lora_" in name
    return tokenizer, model


def validation_loss(encoder, head, loader, pad_token_id, device):
    encoder.eval(); head.eval(); losses = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            packets, questions = encoder.encode(batch["inputs"], pad_token_id=pad_token_id, device=device)
            logits = head(packets, questions, batch["masks"].to(device), batch["example_indices"].to(device))
            loss = high_fidelity_retrieval_loss(logits, batch["attained_levels"].to(device), batch["example_indices"].to(device))
            losses.append(float(loss))
    return sum(losses) / len(losses)


def save_endpoint(path, encoder, head, record):
    path.mkdir(parents=True, exist_ok=True)
    encoder.lm.save_pretrained(path / "adapter")
    torch.save(head_state_dict(encoder), path / "sequential_head.pt")
    torch.save({k: v.detach().float().cpu() for k, v in head.state_dict().items()}, path / "mask_retrieval_head.pt")
    write_metadata(path / "selection_record.json", record)


def main():
    p = argparse.ArgumentParser(description="V13 joint V8-LoRA and fidelity-0.90 mask retriever")
    p.add_argument("--protocol-config", required=True); p.add_argument("--train-data", required=True)
    p.add_argument("--validation-data", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--initial-head", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="models/Qwen3-4B"); p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--max-sequence-length", type=int, default=512); p.add_argument("--steps", type=int, default=300)
    p.add_argument("--validation-interval", type=int, default=100); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=2); p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4); p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--smoke-only", action="store_true"); a = p.parse_args()
    protocol = json.loads(Path(a.protocol_config).read_text())
    if protocol["status"] != "APPROVED_TO_RUN": raise ValueError("protocol is not approved")
    expected = protocol["arguments"]
    mismatches = {key: (getattr(a, key), value) for key, value in expected.items() if getattr(a, key) != value}
    if mismatches: raise ValueError(f"arguments differ from protocol: {mismatches}")
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda:0"); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer, encoder = load_trainable_encoder(a)
    head = MaskValueHead(model_dim=a.model_dim, layers=2, heads=8, dropout=.1)
    head.load_state_dict(torch.load(a.initial_head, map_location="cpu", weights_only=True)); head.to(device)
    train = MaskValueDataset(list(read_jsonl(a.train_data)), tokenizer, a.max_sequence_length)
    valid = MaskValueDataset(list(read_jsonl(a.validation_data)), tokenizer, a.max_sequence_length)
    collator = MaskValueCollator()
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=torch.Generator().manual_seed(a.seed), collate_fn=collator)
    valid_loader = DataLoader(valid, batch_size=a.batch_size, shuffle=False, collate_fn=collator)
    lora = [x for n, x in encoder.named_parameters() if x.requires_grad]
    frozen_bad = [n for n, x in encoder.named_parameters() if x.requires_grad and "lora_" not in n]
    if not lora or frozen_bad: raise ValueError(f"invalid trainable encoder parameters: count={len(lora)} bad={frozen_bad[:5]}")
    optimizer = torch.optim.AdamW([{"params": lora, "lr": a.lora_lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=.01)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(10, a.steps // 10), a.steps)
    iterator = iter(train_loader); history=[]; checkpoints=[]; started=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
    total_steps = 1 if a.smoke_only else a.steps
    for step in range(1, total_steps + 1):
        encoder.train(); head.train(); window=0.0
        for _ in range(a.gradient_accumulation):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(train_loader); batch=next(iterator)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                packets, questions = encoder.encode(batch["inputs"], pad_token_id=int(tokenizer.pad_token_id), device=device)
                logits=head(packets, questions, batch["masks"].to(device), batch["example_indices"].to(device))
                loss=high_fidelity_retrieval_loss(logits,batch["attained_levels"].to(device),batch["example_indices"].to(device))/a.gradient_accumulation
            loss.backward(); window += float(loss.detach())
        torch.nn.utils.clip_grad_norm_([*lora,*head.parameters()],1.0);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == total_steps:
            record={"step":step,"train_retrieval_loss":window,"lr_lora":scheduler.get_last_lr()[0],"lr_head":scheduler.get_last_lr()[1]}
            history.append(record);write_jsonl(out/("smoke_history.jsonl" if a.smoke_only else "history.jsonl"),history);print(json.dumps(record),flush=True)
        if not a.smoke_only and step % a.validation_interval == 0:
            value=validation_loss(encoder,head,valid_loader,int(tokenizer.pad_token_id),device)
            record={"step":step,"internal_validation_retrieval_loss":value};checkpoint=out/"checkpoints"/f"step{step}"
            save_endpoint(checkpoint,encoder,head,record);checkpoints.append({**record,"checkpoint":str(checkpoint)});write_metadata(out/"checkpoint_selection.json",{"checkpoints":checkpoints});print(json.dumps(record),flush=True)
    if a.smoke_only:
        write_metadata(out/"smoke.json",{"complete":True,"peak_gpu_bytes":torch.cuda.max_memory_allocated(device),"trainable_lora_parameters":sum(x.numel() for x in lora),"trainable_head_parameters":sum(x.numel() for x in head.parameters())});return
    selected=min(checkpoints,key=lambda x:(x["internal_validation_retrieval_loss"],x["step"]));write_metadata(out/"selected_checkpoint.json",selected)
    write_metadata(out/"training_metadata.json",experiment_metadata(stage="v13_joint_representation_retrieval",elapsed_seconds=time.perf_counter()-started,selected=selected,locked_roles_used=False,development300_used_during_training=False))
    print(json.dumps({"selected":selected},indent=2),flush=True)

if __name__ == "__main__": main()
