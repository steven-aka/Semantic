#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/qwen3_14b_download"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
mkdir -p "${LOG_DIR}"

ONLY_METADATA=1 "${ROOT_DIR}/scripts/download_qwen3_14b.sh" >"${LOG_DIR}/metadata.log" 2>&1

pids=()
status=0
launched=0

cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

wait_one() {
  if ! wait -n; then
    status=1
  fi
  mapfile -t pids < <(jobs -pr)
}

# Start the smaller final shard first, then refill a slot whenever any shard exits.
for shard in 8 1 2 3 4 5 6 7; do
  if [[ -f "${ROOT_DIR}/models/Qwen3-14B/model-0000${shard}-of-00008.safetensors" ]]; then
    continue
  fi
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait_one
  done
  (
    cd "${ROOT_DIR}"
    ONLY_SHARD="${shard}" scripts/download_qwen3_14b.sh
  ) >"${LOG_DIR}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
  ((launched += 1))
done

while (( ${#pids[@]} > 0 )); do
  wait_one
done

if (( status != 0 )); then
  echo "At least one shard download failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

touch "${ROOT_DIR}/models/Qwen3-14B/.download_complete"
echo "All Qwen3-14B shards passed size and SHA-256 verification (${launched} launched this run)."
