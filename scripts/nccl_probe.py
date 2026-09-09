"""Minimal two-GPU NCCL health probe for the V0 experiment monitor."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([float(local_rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    print(
        f"rank={dist.get_rank()} local_rank={local_rank} "
        f"device={torch.cuda.get_device_name(local_rank)!r} all_reduce={value.item()}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
