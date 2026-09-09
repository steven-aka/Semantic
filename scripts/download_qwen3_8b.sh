#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-8B}"
SNAPSHOT_DIR="${ROOT_DIR}/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
BASE_URL="https://huggingface.co/Qwen/Qwen3-8B/resolve/main"
ONLY_SHARD="${ONLY_SHARD:-}"
ARIA2_BIN="${ARIA2_BIN:-${ROOT_DIR}/.tools/aria2/bin/aria2c}"

if [[ ! -x "${ARIA2_BIN}" ]]; then
  echo "aria2c not found at ${ARIA2_BIN}" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

for name in config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json merges.txt vocab.json; do
  cp -L "${SNAPSHOT_DIR}/${name}" "${MODEL_DIR}/${name}"
done

download_shard() {
  local name="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local output="${MODEL_DIR}/${name}"
  local partial="${output}.part"

  if [[ -f "${output}" ]] \
    && [[ "$(stat -c %s "${output}")" == "${expected_size}" ]] \
    && echo "${expected_sha}  ${output}" | sha256sum --check --status; then
    echo "[ready] ${name}"
    return 0
  fi

  if [[ -f "${partial}" ]] \
    && [[ ! -f "${partial}.aria2" ]] \
    && [[ "$(stat -c %s "${partial}")" == "${expected_size}" ]] \
    && echo "${expected_sha}  ${partial}" | sha256sum --check --status; then
    mv "${partial}" "${output}"
    echo "[verified] ${name}"
    return 0
  fi

  echo "[download] ${name} (resume at $(stat -c %s "${partial}" 2>/dev/null || echo 0) bytes)"
  local attempt=1
  while ! "${ARIA2_BIN}" \
      --continue=true \
      --max-connection-per-server=8 \
      --split=8 \
      --min-split-size=4M \
      --file-allocation=none \
      --max-tries=0 \
      --retry-wait=2 \
      --timeout=30 \
      --connect-timeout=30 \
      --lowest-speed-limit=1K \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --check-integrity=true \
      --checksum="sha-256=${expected_sha}" \
      --console-log-level=warn \
      --summary-interval=30 \
      --dir="${MODEL_DIR}" \
      --out="${name}.part" \
      "${BASE_URL}/${name}"; do
    echo "[aria2 retry ${attempt}] ${name}" >&2
    ((attempt += 1))
    sleep 2
  done

  local actual_size
  actual_size="$(stat -c %s "${partial}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "size mismatch for ${name}: expected ${expected_size}, got ${actual_size}" >&2
    return 1
  fi
  echo "${expected_sha}  ${partial}" | sha256sum --check --status
  mv "${partial}" "${output}"
  echo "[verified] ${name}"
}

[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 1 ]] || download_shard model-00001-of-00005.safetensors 3996250744 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 2 ]] || download_shard model-00002-of-00005.safetensors 3993160032 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 3 ]] || download_shard model-00003-of-00005.safetensors 3959604768 c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 4 ]] || download_shard model-00004-of-00005.safetensors 3187841392 b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 5 ]] || download_shard model-00005-of-00005.safetensors 1244659840 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff

if [[ -n "${ONLY_SHARD}" ]]; then
  echo "Qwen3-8B shard ${ONLY_SHARD} is ready at ${MODEL_DIR}"
else
  echo "Qwen3-8B is ready at ${MODEL_DIR}"
fi
