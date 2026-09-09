#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
EXAMPLES=${EXAMPLES:-data/units/hotpot_exact_pilot_eligible.jsonl}
PACKET_DIR=${PACKET_DIR:-data/packets_smoke}
TEACHER_MODEL=${TEACHER_MODEL:-models/Qwen3-14B}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-models/Qwen3-8B}
DATASET_REVISION=${DATASET_REVISION:-hotpot_qa_distractor_validation}
EXPERIMENT_ID=${EXPERIMENT_ID:-v0_teacher_packets}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
TEACHER_BACKEND=${TEACHER_BACKEND:-vllm}
TRANSFORMERS_GPU_MEMORY_GB=${TRANSFORMERS_GPU_MEMORY_GB:-12}
TRANSFORMERS_CPU_MEMORY_GB=${TRANSFORMERS_CPU_MEMORY_GB:-60}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

"$PROJECT_PYTHON" -m src.teacher.packet_generator \
  --examples "$EXAMPLES" \
  --output-dir "$PACKET_DIR" \
  --model "$TEACHER_MODEL" \
  --tokenizer "$TOKENIZER_MODEL" \
  --dataset-revision "$DATASET_REVISION" \
  --experiment-id "$EXPERIMENT_ID" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --backend "$TEACHER_BACKEND" \
  --transformers-gpu-memory-gb "$TRANSFORMERS_GPU_MEMORY_GB" \
  --transformers-cpu-memory-gb "$TRANSFORMERS_CPU_MEMORY_GB"
