#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/qwen3_8b_download"
mkdir -p "${LOG_DIR}"

pids=()
for shard in 1 2 3 4 5; do
  if [[ -f "${ROOT_DIR}/models/Qwen3-8B/model-0000${shard}-of-00005.safetensors" ]]; then
    continue
  fi
  (
    cd "${ROOT_DIR}"
    ONLY_SHARD="${shard}" scripts/download_qwen3_8b.sh
  ) >"${LOG_DIR}/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

if (( ${#pids[@]} == 0 )); then
  touch "${ROOT_DIR}/models/Qwen3-8B/.download_complete"
  exit 0
fi

cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if (( status != 0 )); then
  echo "At least one shard download failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

touch "${ROOT_DIR}/models/Qwen3-8B/.download_complete"
echo "All Qwen3-8B shards passed size and SHA-256 verification."
