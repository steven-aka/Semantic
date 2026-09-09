#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-14B}"
REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18"
BASE_URL="https://huggingface.co/Qwen/Qwen3-14B/resolve/${REVISION}"
ONLY_SHARD="${ONLY_SHARD:-}"
ONLY_METADATA="${ONLY_METADATA:-0}"
ARIA2_BIN="${ARIA2_BIN:-${ROOT_DIR}/.tools/aria2/bin/aria2c}"

if [[ ! -x "${ARIA2_BIN}" ]]; then
  echo "aria2c not found at ${ARIA2_BIN}" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

aria_download() {
  local name="$1"
  local expected_size="$2"
  local checksum="${3:-}"
  local output="${MODEL_DIR}/${name}"
  local partial="${output}.part"

  if [[ -f "${output}" ]] && [[ "$(stat -c %s "${output}")" == "${expected_size}" ]]; then
    if [[ -z "${checksum}" ]] || echo "${checksum}  ${output}" | sha256sum --check --status; then
      echo "[ready] ${name}"
      return 0
    fi
  fi

  local checksum_args=()
  if [[ -n "${checksum}" ]]; then
    checksum_args+=(--check-integrity=true --checksum="sha-256=${checksum}")
  fi
  echo "[download] ${name}"
  local attempt=1
  while ! "${ARIA2_BIN}" \
      --continue=true \
      --max-connection-per-server=4 \
      --split=4 \
      --min-split-size=4M \
      --file-allocation=none \
      --max-tries=0 \
      --retry-wait=2 \
      --timeout=60 \
      --connect-timeout=30 \
      --lowest-speed-limit=0 \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --console-log-level=warn \
      --summary-interval=30 \
      "${checksum_args[@]}" \
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
  if [[ -n "${checksum}" ]]; then
    echo "${checksum}  ${partial}" | sha256sum --check --status
  fi
  mv "${partial}" "${output}"
  echo "[verified] ${name}"
}

download_metadata() {
  aria_download config.json 728
  aria_download generation_config.json 239
  aria_download merges.txt 1671853
  aria_download model.safetensors.index.json 36514
  aria_download tokenizer.json 11422654 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
  aria_download tokenizer_config.json 9732
  aria_download vocab.json 2776833
}

download_metadata
if [[ "${ONLY_METADATA}" == 1 ]]; then
  echo "Qwen3-14B metadata is ready at ${MODEL_DIR}"
  exit 0
fi

[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 1 ]] || aria_download model-00001-of-00008.safetensors 3841788544 e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 2 ]] || aria_download model-00002-of-00008.safetensors 3963750816 f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 3 ]] || aria_download model-00003-of-00008.safetensors 3963750880 dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 4 ]] || aria_download model-00004-of-00008.safetensors 3963750880 eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 5 ]] || aria_download model-00005-of-00008.safetensors 3963750880 97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 6 ]] || aria_download model-00006-of-00008.safetensors 3963750880 9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 7 ]] || aria_download model-00007-of-00008.safetensors 3963750880 0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750
[[ -n "${ONLY_SHARD}" && "${ONLY_SHARD}" != 8 ]] || aria_download model-00008-of-00008.safetensors 1912371880 0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0

echo "Qwen3-14B ${ONLY_SHARD:+shard ${ONLY_SHARD} }is ready at ${MODEL_DIR}"
