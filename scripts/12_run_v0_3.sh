#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
TARGET_MODEL=${TARGET_MODEL:-models/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-models/Qwen3-14B}
STAGE=${1:-all}

prepare() {
  "$PROJECT_PYTHON" -m src.data.label_free_pilot \
    --raw data/raw/hotpot_validation.parquet \
    --output data/units/hotpot_v0_3_candidates.jsonl \
    --manifest results/v0_3/data_manifest.json \
    --tokenizer "$TARGET_MODEL" --count 30 --units 6 --max-tokens 256 \
    --k1 1.2 --b 0.75 \
    --sample-salt v0_3_label_free_sentence_atomic_20260907
  "$PROJECT_PYTHON" -m src.evaluation.data_audit \
    --input data/units/hotpot_v0_3_candidates.jsonl \
    --output results/v0_3/data_audit.json \
    --tokenizer "$TARGET_MODEL"
}

baseline() {
  "$PROJECT_PYTHON" -m src.evaluation.full_context \
    --examples data/units/hotpot_v0_3_candidates.jsonl \
    --output results/v0_3/full_context.jsonl \
    --eligible-output data/units/hotpot_v0_3_full_context_capable.jsonl \
    --model "$TARGET_MODEL" \
    --dataset-revision hotpot_qa_distractor_validation \
    --backend vllm --batch-size 30 --gpu-memory-utilization 0.50 \
    --max-model-len 2048 --experiment-id v0_3_label_free_full_context \
    --reuse-valid
}

packets() {
  EXAMPLES=data/units/hotpot_v0_3_candidates.jsonl \
  PACKET_DIR=data/packets_v0_3 \
  TEACHER_MODEL="$TEACHER_MODEL" \
  TOKENIZER_MODEL="$TARGET_MODEL" \
  GPU_MEMORY_UTILIZATION=0.65 \
  MAX_MODEL_LEN=2048 \
  MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=4096 \
  EXPERIMENT_ID=v0_3_label_free_packets \
    bash scripts/03_generate_packets.sh
}

target() {
  "$PROJECT_PYTHON" -m src.evaluation.run_fact_atomic_target \
    --examples data/units/hotpot_v0_3_candidates.jsonl \
    --raw data/raw/hotpot_validation.parquet \
    --packet-dir data/packets_v0_3 \
    --output-dir results/v0_3 \
    --model "$TARGET_MODEL" \
    --gpu-memory-utilization 0.60 --max-model-len 2048 --batch-size 729 \
    --allow-missing-facts --experiment-id-prefix v0_3_label_free
}

analyze() { bash scripts/10_analyze_v0_3.sh; }
verify() { bash scripts/11_verify_v0_3.sh; }

case "$STAGE" in
  prepare|baseline|packets|target|analyze|verify) "$STAGE" ;;
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

echo "V0.3 stage '$STAGE' complete. Gate result is results/v0_3/gate_result.json. STOP before V1/QLoRA."
