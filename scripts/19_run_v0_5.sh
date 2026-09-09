#!/usr/bin/env bash
set -euo pipefail
. scripts/cuda_env.sh

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
TARGET_MODEL=${TARGET_MODEL:-models/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-models/Qwen3-14B}
STAGE=${1:-all}

prepare() {
  "$PROJECT_PYTHON" -m src.data.eligible_semantic_pilot prepare \
    --config configs/v0_5_gate.json --raw data/raw/hotpot_validation.parquet \
    --output-dir results/v0_5 --candidate-output data/units/hotpot_v0_5_candidates.jsonl
}
baseline() {
  "$PROJECT_PYTHON" -m src.evaluation.full_context \
    --examples data/units/hotpot_v0_5_pool.jsonl \
    --output results/v0_5/pool_full_context.jsonl \
    --eligible-output data/units/hotpot_v0_5_pool_eligible.jsonl \
    --model "$TARGET_MODEL" --dataset-revision hotpot_qa_distractor_validation \
    --backend vllm --batch-size 64 --gpu-memory-utilization 0.60 \
    --max-model-len 2048 --experiment-id v0_5_answerability_pool --reuse-valid
  "$PROJECT_PYTHON" -m src.data.eligible_semantic_pilot finalize \
    --config configs/v0_5_gate.json --raw data/raw/hotpot_validation.parquet \
    --output-dir results/v0_5 --candidate-output data/units/hotpot_v0_5_candidates.jsonl
  "$PROJECT_PYTHON" -m src.evaluation.data_audit \
    --input data/units/hotpot_v0_5_candidates.jsonl \
    --output results/v0_5/data_audit.json --tokenizer "$TARGET_MODEL"
}
packets() {
  EXAMPLES=data/units/hotpot_v0_5_candidates.jsonl PACKET_DIR=data/packets_v0_5 \
  TEACHER_MODEL="$TEACHER_MODEL" TOKENIZER_MODEL="$TARGET_MODEL" \
  GPU_MEMORY_UTILIZATION=0.65 MAX_MODEL_LEN=2048 MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=4096 EXPERIMENT_ID=v0_5_answerable_semantic_packets \
    bash scripts/03_generate_packets.sh
}
target() {
  "$PROJECT_PYTHON" -m src.evaluation.run_fact_atomic_target \
    --examples data/units/hotpot_v0_5_candidates.jsonl \
    --raw data/raw/hotpot_validation.parquet --packet-dir data/packets_v0_5 \
    --output-dir results/v0_5 --model "$TARGET_MODEL" \
    --gpu-memory-utilization 0.60 --max-model-len 2048 --batch-size 729 \
    --allow-missing-facts --experiment-id-prefix v0_5_answerable_semantic
}
analyze() { RESULT_DIR=results/v0_5 bash scripts/15_analyze_v0_4.sh; }
verify() { bash scripts/18_verify_v0_5.sh; }

case "$STAGE" in
  prepare|baseline|packets|target|analyze|verify) "$STAGE" ;;
  all) prepare; baseline; packets; target; analyze; verify ;;
  *) echo "usage: $0 [all|prepare|baseline|packets|target|analyze|verify]" >&2; exit 2 ;;
esac

echo "V0.5 stage '$STAGE' complete. Inspect results/v0_5/gate_result.json; do not start V1 automatically."
