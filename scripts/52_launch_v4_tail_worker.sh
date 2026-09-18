#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/cuda_env.sh

root=results/v2_rank_then_cut
out="$root/v4_candidates500_exact"
logs="$root/v4_candidate_exact_logs"
queue="$logs/gpu3_tail_examples.jsonl"
log="$logs/gpu3_tail.log"
pidfile="$logs/gpu3_tail.pid"
mkdir -p "$logs"

if pgrep -f '[s]rc.search.atomic_exact_search.*gpu3_tail_examples.jsonl' >/dev/null; then
  echo "V4 GPU3 tail worker is already running"
  exit 0
fi

for shard in 0 1; do
  if ! pgrep -f "[s]rc.search.atomic_exact_search.*v4_candidates500_exact.*--shard-index $shard --num-shards 2" >/dev/null; then
    echo "original V4 shard $shard is not running; leave its restart to the scheduler" >&2
    exit 2
  fi
done

gpu_ready() {
  local free util
  read -r free util < <(
    nvidia-smi -i 3 --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr ',' ' '
  )
  (( free >= 32000 && util <= 20 ))
}

if ! gpu_ready; then
  echo "GPU3 has insufficient stable capacity for the V4 tail worker" >&2
  exit 3
fi
sleep 30
if ! gpu_ready; then
  echo "GPU3 capacity changed during the 30-second check" >&2
  exit 3
fi

python3 - "$out" "$queue" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
queue = Path(sys.argv[2])
source = Path('data/units/qampari_rank_v4_candidates500.jsonl')
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
if len(rows) != 500:
    raise SystemExit(f'expected 500 frozen V4 candidates; found {len(rows)}')
shards = [rows[0::2], rows[1::2]]
counts = [sum((out / f"{row['example_id']}.jsonl").is_file() for row in shard) for shard in shards]
remaining = [250 - count for count in counts]
# Five examples of margin give the original workers time to reuse completed tail files.
target = math.ceil(sum(remaining) / 3) + 5
tails = [max(0, count - target) for count in remaining]
selected = shards[1][250 - tails[1]:] + shards[0][250 - tails[0]:]
selected = [row for row in selected if not (out / f"{row['example_id']}.jsonl").is_file()]
if not selected:
    raise SystemExit('No useful V4 tail work remains')
temporary = queue.with_suffix(queue.suffix + '.tmp')
with temporary.open('w') as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(queue)
print(f'original counts={counts}, tail allocation={tails}, GPU3 queue={len(selected)}')
PY

echo "[V4 GPU3 tail launch] $(date -Is)" >>"$log"
nohup setsid env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  .venv/bin/python -m src.search.atomic_exact_search \
    --examples "$queue" \
    --annotations data/units/qampari_rank_v4_candidates500_annotations.jsonl \
    --packet-dir data/packets_qampari_rank_v4_candidates500 \
    --output-dir "$out" \
    --model models/Qwen3-8B \
    --batch-size 512 \
    --max-new-tokens 256 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096 \
    --shard-index 0 \
    --num-shards 1 >>"$log" 2>&1 < /dev/null &
echo "$!" >"$pidfile"
echo "V4 GPU3 tail worker launched pid=$! queue=$queue log=$log"
