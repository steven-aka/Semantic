#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root=results/v2_rank_then_cut
out="$root/v4_candidates500_exact"
logs="$root/v4_candidate_exact_logs"
complete=$(find "$out" -maxdepth 1 -name '*.jsonl' -type f 2>/dev/null | wc -l)
echo "V4 new target-blind candidate exact progress: $complete/500"
mapfile -t shard_counts < <(.venv/bin/python - <<'PY'
from pathlib import Path
from src.data.schemas import read_jsonl
rows=list(read_jsonl('data/units/qampari_rank_v4_candidates500.jsonl'))
out=Path('results/v2_rank_then_cut/v4_candidates500_exact')
for shard in range(2):
    selected=[row for index,row in enumerate(rows) if index%2==shard]
    print(sum((out/f"{row['example_id']}.jsonl").is_file() for row in selected))
PY
)
for shard in 0 1; do
  status=PENDING
  running=$(pgrep -f "[s]rc.search.atomic_exact_search.*v4_candidates500_exact.*--shard-index $shard --num-shards 2" | head -1 || true)
  if [[ -n "$running" ]]; then
    status="RUNNING pid=$running"
  elif [[ -s "$logs/shard${shard}.log" ]]; then
    if grep -Eq 'Traceback|CUDA out of memory|Engine core initialization failed' "$logs/shard${shard}.log"; then
      status=ERROR
    else
      status=STOPPED_OR_COMPLETE
    fi
  fi
  last=$(grep -E '\[atomic exact ' "$logs/shard${shard}.log" 2>/dev/null | tail -1 || true)
  echo "shard$shard: ${shard_counts[$shard]}/250 $status ${last:-no completed example in current detached log yet}"
done
echo "GPU index, used MiB, free MiB, utilization %, temperature C"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo "Physical GPU4 is excluded from this project."
