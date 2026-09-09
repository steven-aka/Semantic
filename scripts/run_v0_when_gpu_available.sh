#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT_PYTHON=${PROJECT_PYTHON:-.venv/bin/python}
LOG_DIR=${LOG_DIR:-logs/v0_autorun}
POLL_SECONDS=${POLL_SECONDS:-30}
STABILITY_SECONDS=${STABILITY_SECONDS:-10}
TEACHER_MIN_FREE_MIB=${TEACHER_MIN_FREE_MIB:-36000}
TARGET_MIN_FREE_MIB=${TARGET_MIN_FREE_MIB:-34000}
PAIR_MIN_FREE_MIB=${PAIR_MIN_FREE_MIB:-14500}
PAIR_GPU_MEMORY_UTILIZATION=${PAIR_GPU_MEMORY_UTILIZATION:-0.29}
PAIR_TEACHER_CPU_OFFLOAD_GB=${PAIR_TEACHER_CPU_OFFLOAD_GB:-2}
PAIR_TEACHER_MAX_MODEL_LEN=${PAIR_TEACHER_MAX_MODEL_LEN:-1536}
PAIR_TEACHER_MAX_NUM_BATCHED_TOKENS=${PAIR_TEACHER_MAX_NUM_BATCHED_TOKENS:-2048}
GPU_LIST=${GPU_LIST:-0,1,2,3}
PAIR_GPU_LIST=${PAIR_GPU_LIST:-0,1}
mkdir -p "$LOG_DIR"

exec 9>"$LOG_DIR/monitor.lock"
if ! flock -n 9; then
  echo "another V0 monitor already holds $LOG_DIR/monitor.lock" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG_DIR/monitor.log" >&2
}

write_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "${3:-}" >"$LOG_DIR/status.tsv"
}

best_gpu() {
  local minimum="$1"
  timeout 30 nvidia-smi -i "$GPU_LIST" \
    --query-gpu=index,memory.free \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v minimum="$minimum" '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        if (($2 + 0) >= minimum && ($2 + 0) > best) {
          best = $2 + 0
          gpu = $1
        }
      }
      END { if (gpu != "") print gpu }
    '
}

gpu_free_mib() {
  timeout 30 nvidia-smi -i "$1" \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null |
    awk 'NR == 1 { gsub(/[[:space:]]/, ""); print; exit }'
}

best_gpu_pair() {
  local minimum="$1"
  timeout 30 nvidia-smi -i "$PAIR_GPU_LIST" \
    --query-gpu=index,memory.free \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v minimum="$minimum" '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        free = $2 + 0
        if (free < minimum) next
        if (free > first_free) {
          second_free = first_free
          second_gpu = first_gpu
          first_free = free
          first_gpu = $1
        } else if (free > second_free) {
          second_free = free
          second_gpu = $1
        }
      }
      END { if (first_gpu != "" && second_gpu != "") print first_gpu "," second_gpu }
    '
}

devices_have_free() {
  local devices="$1"
  local minimum="$2"
  local expected=1
  [[ "$devices" == *,* ]] && expected=2
  timeout 30 nvidia-smi -i "$devices" \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -v minimum="$minimum" -v expected="$expected" '
      {
        gsub(/[[:space:]]/, "")
        if (($0 + 0) >= minimum) good += 1
      }
      END { exit(good == expected ? 0 : 1) }
    '
}

wait_for_devices() {
  local stage="$1"
  local single_minimum="$2"
  local pair_minimum="$3"
  local polls=0
  local candidate candidate_minimum mode
  while true; do
    candidate="$(best_gpu "$single_minimum")"
    candidate_minimum="$single_minimum"
    mode=single
    if [[ -z "$candidate" ]]; then
      candidate="$(best_gpu_pair "$pair_minimum")"
      candidate_minimum="$pair_minimum"
      mode=pair
    fi
    if [[ -n "$candidate" ]]; then
      sleep "$STABILITY_SECONDS"
      if devices_have_free "$candidate" "$candidate_minimum"; then
        log "$stage selected $mode physical GPU set $candidate"
        printf '%s\n' "$candidate"
        return 0
      fi
      log "$stage candidate GPU set $candidate lost headroom during stability check"
    fi
    ((polls += 1))
    if (( polls == 1 || polls % 20 == 0 )); then
      log "$stage waiting for one GPU >= ${single_minimum} MiB on $GPU_LIST or TP pair $PAIR_GPU_LIST each >= ${pair_minimum} MiB"
      write_status "$stage" waiting "single_min=$single_minimum pair=$PAIR_GPU_LIST pair_each_min=$pair_minimum"
    fi
    sleep "$POLL_SECONDS"
  done
}

is_transient_gpu_failure() {
  rg -q \
    'Free memory on device .* is less than desired|CUDA out of memory|CUDA error: out of memory|No available memory for the cache blocks' \
    "$1"
}

run_gpu_stage() {
  local stage="$1"
  local single_minimum="$2"
  local pair_minimum="$3"
  local single_utilization="$4"
  shift 4
  local attempt=0 devices tensor_parallel utilization max_model_len max_num_seqs
  local max_num_batched_tokens cpu_offload_gb teacher_backend transformers_gpu_memory_gb
  local attempt_log exit_code
  while true; do
    ((attempt += 1))
    devices="$(wait_for_devices "$stage" "$single_minimum" "$pair_minimum")"
    tensor_parallel=1
    utilization="$single_utilization"
    max_model_len=8192
    max_num_seqs=8
    max_num_batched_tokens=4096
    cpu_offload_gb=0
    teacher_backend=vllm
    transformers_gpu_memory_gb=12
    if [[ "$stage" == packet_* ]]; then
      max_model_len=2048
    fi
    if [[ "$devices" == *,* ]]; then
      tensor_parallel=2
      utilization="$PAIR_GPU_MEMORY_UTILIZATION"
      if [[ "$stage" == packet_* ]]; then
        # A failed physical GPU makes NCCL enumerate-and-fail even when hidden.
        # Transformers device_map shards in one process and avoids NCCL/NVML.
        tensor_parallel=1
        teacher_backend=transformers
        if [[ "$devices" == "1,0" ]]; then
          transformers_gpu_memory_gb=18,11.5
        else
          transformers_gpu_memory_gb=11.5,18
        fi
        max_model_len="$PAIR_TEACHER_MAX_MODEL_LEN"
        max_num_seqs=6
        max_num_batched_tokens="$PAIR_TEACHER_MAX_NUM_BATCHED_TOKENS"
        cpu_offload_gb="$PAIR_TEACHER_CPU_OFFLOAD_GB"
      fi
    fi
    attempt_log="$LOG_DIR/${stage}_$(date '+%Y%m%d_%H%M%S')_attempt_${attempt}.log"
    write_status "$stage" running "gpus=$devices tp=$tensor_parallel attempt=$attempt"
    log "$stage starting on physical GPU set $devices with TP=$tensor_parallel (attempt $attempt)"
    CUDA_VISIBLE_DEVICES="$devices" \
      TENSOR_PARALLEL_SIZE="$tensor_parallel" \
      GPU_MEMORY_UTILIZATION="$utilization" \
      MAX_MODEL_LEN="$max_model_len" \
      MAX_NUM_SEQS="$max_num_seqs" \
      MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens" \
      CPU_OFFLOAD_GB="$cpu_offload_gb" \
      TEACHER_BACKEND="$teacher_backend" \
      TRANSFORMERS_GPU_MEMORY_GB="$transformers_gpu_memory_gb" \
      "$@" >"$attempt_log" 2>&1
    exit_code=$?
    if (( exit_code == 0 )); then
      log "$stage completed on physical GPU set $devices"
      write_status "$stage" complete "gpus=$devices tp=$tensor_parallel attempt=$attempt"
      return 0
    fi
    if is_transient_gpu_failure "$attempt_log"; then
      log "$stage lost the GPU race (exit $exit_code); returning to monitor queue"
      write_status "$stage" waiting "transient_gpu_failure attempt=$attempt"
      sleep "$POLL_SECONDS"
      continue
    fi
    log "$stage failed with a non-transient error; inspect $attempt_log"
    write_status "$stage" failed "log=$attempt_log exit=$exit_code"
    return "$exit_code"
  done
}

artifacts_complete() {
  "$PROJECT_PYTHON" -m src.evaluation.artifact_status "$1" \
    --examples "$2" --artifact-dir "$3" >/dev/null 2>&1
}

SMOKE_EXAMPLES=data/units/hotpot_exact_pilot_eligible.jsonl
CONTROLLED_EXAMPLES=data/units/hotpot_controlled_pilot_eligible.jsonl

log "V0 automatic monitor started"
write_status monitor active

if ! artifacts_complete packets "$SMOKE_EXAMPLES" data/packets_smoke; then
  run_gpu_stage packet_smoke "$TEACHER_MIN_FREE_MIB" "$PAIR_MIN_FREE_MIB" 0.70 \
    env EXAMPLES="$SMOKE_EXAMPLES" PACKET_DIR=data/packets_smoke \
    EXPERIMENT_ID=v0_teacher_packets_smoke bash scripts/03_generate_packets.sh || exit $?
else
  log "packet_smoke already complete; skipping"
fi

if ! artifacts_complete exact "$SMOKE_EXAMPLES" results/exact_search_smoke; then
  run_gpu_stage exact_smoke "$TARGET_MIN_FREE_MIB" 99999 0.65 \
    env EXAMPLES="$SMOKE_EXAMPLES" PACKET_DIR=data/packets_smoke \
    OUTPUT_DIR=results/exact_search_smoke EXPERIMENT_ID=v0_exact_smoke \
    BATCH_SIZE=243 bash scripts/04_exact_pilot.sh || exit $?
else
  log "exact_smoke already complete; skipping"
fi

write_status analyze_smoke running
bash scripts/05_analyze_v0_smoke.sh >"$LOG_DIR/analyze_smoke.log" 2>&1 || {
  write_status analyze_smoke failed "log=$LOG_DIR/analyze_smoke.log"
  exit 1
}
log "analyze_smoke completed"

if ! artifacts_complete packets "$CONTROLLED_EXAMPLES" data/packets; then
  run_gpu_stage packet_controlled "$TEACHER_MIN_FREE_MIB" "$PAIR_MIN_FREE_MIB" 0.70 \
    env EXAMPLES="$CONTROLLED_EXAMPLES" PACKET_DIR=data/packets \
    EXPERIMENT_ID=v0_teacher_packets_controlled bash scripts/03_generate_packets.sh || exit $?
else
  log "packet_controlled already complete; skipping"
fi

if ! artifacts_complete exact "$CONTROLLED_EXAMPLES" results/exact_search; then
  run_gpu_stage exact_controlled "$TARGET_MIN_FREE_MIB" 99999 0.65 \
    env EXAMPLES="$CONTROLLED_EXAMPLES" PACKET_DIR=data/packets \
    OUTPUT_DIR=results/exact_search EXPERIMENT_ID=v0_exact_controlled \
    BATCH_SIZE=729 bash scripts/04_exact_pilot.sh || exit $?
else
  log "exact_controlled already complete; skipping"
fi

write_status analyze_controlled running
bash scripts/05_analyze_v0.sh >"$LOG_DIR/analyze_controlled.log" 2>&1 || {
  write_status analyze_controlled failed "log=$LOG_DIR/analyze_controlled.log"
  exit 1
}
log "V0 automatic pipeline completed; QLoRA/V1 remains stopped for human review"
write_status v0 complete "summary=results/v0_summary.json"
