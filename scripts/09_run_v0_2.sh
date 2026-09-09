#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
TARGET_MODEL=${TARGET_MODEL:-models/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-models/Qwen3-14B}
STAGE=${1:-all}

prepare() {
  "$PROJECT_PYTHON" -m src.data.fact_atomic_pilot \
    --examples data/units/hotpot_validation.jsonl \
    --raw data/raw/hotpot_validation.parquet \
    --output data/units/hotpot_fact_atomic_candidates.jsonl \
    --units 6 --min-facts 3 --max-facts 5
  "$PROJECT_PYTHON" -m src.evaluation.data_audit \
    --input data/units/hotpot_fact_atomic_candidates.jsonl \
    --output results/v0_2_data_audit.json \
    --tokenizer "$TARGET_MODEL"
}

baseline() {
  "$PROJECT_PYTHON" -m src.evaluation.full_context \
    --examples data/units/hotpot_fact_atomic_candidates.jsonl \
    --output results/v0_2/full_context.jsonl \
    --eligible-output data/units/hotpot_fact_atomic_eligible.jsonl \
    --model "$TARGET_MODEL" \
    --dataset-revision hotpot_qa_distractor_validation \
    --backend vllm --batch-size 33 --gpu-memory-utilization 0.50 \
    --max-model-len 2048 --experiment-id v0_2_fact_atomic_full_context \
    --reuse-valid
}

packets() {
  EXAMPLES=data/units/hotpot_fact_atomic_eligible.jsonl \
  PACKET_DIR=data/packets_v0_2 \
  TEACHER_MODEL="$TEACHER_MODEL" \
  TOKENIZER_MODEL="$TARGET_MODEL" \
  GPU_MEMORY_UTILIZATION=0.65 \
  MAX_MODEL_LEN=2048 \
  MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=4096 \
  EXPERIMENT_ID=v0_2_fact_atomic_packets \
    bash scripts/03_generate_packets.sh
}

target() {
  "$PROJECT_PYTHON" -m src.evaluation.run_fact_atomic_target \
    --examples data/units/hotpot_fact_atomic_eligible.jsonl \
    --raw data/raw/hotpot_validation.parquet \
    --packet-dir data/packets_v0_2 \
    --output-dir results/v0_2 \
    --model "$TARGET_MODEL" \
    --gpu-memory-utilization 0.60 --max-model-len 2048 --batch-size 729
}

analyze() {
  bash scripts/07_analyze_v0_2.sh
}

verify() {
  bash scripts/08_verify_v0_2.sh
}

case "$STAGE" in
  prepare|baseline|packets|target|analyze|verify)
    "$STAGE"
    ;;
  all)
    prepare
    baseline
    packets
    target
    analyze
    verify
    ;;
  *)
    echo "usage: $0 [all|prepare|baseline|packets|target|analyze|verify]" >&2
    exit 2
    ;;
esac

echo "V0.2 stage '$STAGE' complete. STOP before V1/QLoRA."
