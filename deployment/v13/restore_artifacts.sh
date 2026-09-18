#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
bundle_dir=deployment/v13
archive="$bundle_dir/v13-continuation.restore.tar.zst"
trap 'rm -f "$archive"' EXIT

sha256sum -c "$bundle_dir/parts.sha256"
cat "$bundle_dir"/artifacts/v13-continuation.tar.zst.part-* > "$archive"

expected="$(cut -d' ' -f1 "$bundle_dir/archive.sha256")"
actual="$(sha256sum "$archive" | cut -d' ' -f1)"
if [[ "$actual" != "$expected" ]]; then
  echo "archive checksum mismatch: expected $expected, got $actual" >&2
  exit 1
fi

tar --zstd -xf "$archive"
.venv/bin/python "$bundle_dir/verify_artifacts.py"
echo "V13 continuation artifacts restored and verified."
