#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
TARGET_MODEL=${TARGET_MODEL:-models/Qwen3-8B}
DATASET_REVISION=${DATASET_REVISION:-hotpot_qa_distractor_validation}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

"$PROJECT_PYTHON" -m src.evaluation.full_context \
  --examples data/units/hotpot_exact_pilot.jsonl \
  --output results/full_context.jsonl \
  --eligible-output data/units/hotpot_exact_pilot_eligible.jsonl \
  --model "$TARGET_MODEL" \
  --dataset-revision "$DATASET_REVISION" \
  --backend vllm \
  --gpu-memory-utilization 0.65 \
  --max-model-len 8192
