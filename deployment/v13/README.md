# V13 continuation bundle

This directory makes the selected V13 endpoint and its expensive supervision
portable to another machine. The Git repository contains five split archive
parts. Together they restore 5,018 files (6,264,246,994 uncompressed bytes),
including:

- the selected V13 step300 LoRA adapter, sequential head, and retrieval head;
- the frozen V13 internal train/validation mask supervision;
- the consumed development300 full mask lattice and V8 reference orders;
- V11 candidate supervision used by the downstream selector work;
- the 5,000-example source units and the 5.6 GB exact-search lattice.

The Qwen3-4B base model is not redistributed. It is downloaded from
`Qwen/Qwen3-4B` during bootstrap.

From a fresh clone, run:

```bash
bash deployment/v13/bootstrap_environment.sh
bash deployment/v13/restore_artifacts.sh
source scripts/cuda_env.sh
.venv/bin/python -m unittest discover -s tests -q
```

`scripts/cuda_env.sh` contains paths from the original shared CUDA environment.
On a standalone machine with CUDA libraries supplied by the Python packages or
system installation, edit or skip those site-specific paths before running.

The selected endpoint is restored at:

```text
results/v2_rank_then_cut/v13_joint_retriever_seed20260918/checkpoints/step300
```

The complete per-file sizes and SHA-256 checksums are in
`artifact_manifest.json`. `restore_artifacts.sh` verifies both the split archive
and every restored file before reporting success.
