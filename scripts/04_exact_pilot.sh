#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
TARGET_MODEL=${TARGET_MODEL:-models/Qwen3-8B}
EXAMPLES=${EXAMPLES:-data/units/hotpot_exact_pilot_eligible.jsonl}
PACKET_DIR=${PACKET_DIR:-data/packets_smoke}
OUTPUT_DIR=${OUTPUT_DIR:-results/exact_search_smoke}
DATASET_REVISION=${DATASET_REVISION:-hotpot_qa_distractor_validation}
EXPERIMENT_ID=${EXPERIMENT_ID:-v0_exact}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.65}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
BATCH_SIZE=${BATCH_SIZE:-729}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

"$PROJECT_PYTHON" -m src.search.exact_search \
  --examples "$EXAMPLES" \
  --packet-dir "$PACKET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model "$TARGET_MODEL" \
  --dataset-revision "$DATASET_REVISION" \
  --experiment-id "$EXPERIMENT_ID" \
  --backend vllm \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --batch-size "$BATCH_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
