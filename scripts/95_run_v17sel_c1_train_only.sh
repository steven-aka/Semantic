#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
export CUDA_VISIBLE_DEVICES="${SEL_C1_GPU:-1}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

root=results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout
config=configs/v17sel_c0_top10_set_selector.json
data="$root/data/v10_v12_train_train_clean.jsonl"
cache="$root/sel_c0_train_v13_embeddings.pt"
rollouts="$root/sel_c0_train_v8_rollouts.jsonl"
head="$root/v13_run/checkpoints/step100/mask_retrieval_head.pt"
candidates="$root/sel_c0_train_top10_candidates.jsonl"
candidate_manifest="$root/sel_c0_train_top10_candidates_manifest.json"
audit="$root/sel_c0_train_cost_audit"
train="$root/sel_c1_train_only"
holdout_candidates="$root/sel_c1_holdout_top10_candidates.jsonl"
holdout_manifest="$root/sel_c1_holdout_top10_candidates_manifest.json"
holdout="$root/sel_c1_design_exposed_holdout"
status="$root/sel_c1_train_only_status.txt"
trap 'printf "FAILED at line %s\n" "$LINENO" > "$status"' ERR

printf 'WAITING_FOR_V8_ROLLOUT\n' > "$status"
while [[ ! -s "$rollouts.metadata.json" ]]; do
  if ! pgrep -f '^.venv/bin/python -u -m src.evaluation.rollout_v8_policy' >/dev/null; then
    echo 'V8 clean-train rollout stopped before completion' >&2
    exit 1
  fi
  sleep 20
done
[[ -s "$root/sel_c0_train_v13_embeddings_manifest.json" ]] || { echo 'V13 clean-train cache incomplete' >&2; exit 1; }

if [[ ! -s "$candidate_manifest" ]]; then
  printf 'BUILDING_TOP10_TRAIN_CANDIDATES\n' > "$status"
  .venv/bin/python -u -m src.data.build_v17sel_c0_train_candidates \
    --config "$config" --data "$data" --cache "$cache" \
    --rollouts "$rollouts" --head "$head" \
    --exact-dir results/v2_rank_then_cut/candidates5000_exact \
    --output "$candidates" --manifest "$candidate_manifest" \
    > "$root/sel_c0_build.log" 2>&1
fi

if [[ ! -s "$audit/summary.json" ]]; then
  printf 'AUDITING_TRAIN_ONLY_ORACLE_TOKEN_COST\n' > "$status"
  .venv/bin/python -u -m src.evaluation.v17sel_c0_train_cost_audit \
    --config "$config" --candidates "$candidates" --output-dir "$audit" \
    > "$root/sel_c0_cost_audit.log" 2>&1
fi

if [[ ! -s "$train/summary.json" ]]; then
  printf 'TRAINING_TOP4_AND_TOP10_SET_SELECTORS\n' > "$status"
  .venv/bin/python -u -m src.training.train_v17sel_c1_set_selector \
    --config "$config" --candidates "$candidates" --cache "$cache" \
    --output-dir "$train" > "$root/sel_c1_train_only.log" 2>&1
fi
decision="$(.venv/bin/python - "$train/summary.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['decision'])
PY
)"

if [[ "$decision" == GO_SEL_C1_ONE_DESIGN_EXPOSED_HOLDOUT_READ ]]; then
  if [[ ! -s "$holdout_manifest" ]]; then
    printf 'BUILDING_ONE_READ_HOLDOUT_CANDIDATES\n' > "$status"
    .venv/bin/python -u -m src.data.build_v17sel_c0_train_candidates \
      --config "$config" --role holdout \
      --data "$root/data/v10_v12_train_holdout.jsonl" \
      --cache "$root/v13_holdout_embeddings.pt" \
      --rollouts "$root/v8_holdout_rollouts.jsonl" \
      --head "$head" --exact-dir results/v2_rank_then_cut/candidates5000_exact \
      --output "$holdout_candidates" --manifest "$holdout_manifest" \
      > "$root/sel_c1_holdout_build.log" 2>&1
  fi
  if [[ ! -s "$holdout/summary.json" ]]; then
    printf 'EVALUATING_ONE_READ_HOLDOUT\n' > "$status"
    .venv/bin/python -u -m src.evaluation.v17sel_c1_holdout_gate \
      --config "$config" --candidates "$holdout_candidates" \
      --cache "$root/v13_holdout_embeddings.pt" --train-output "$train" \
      --b2b-summary "$root/pool_audit/summary.json" --output-dir "$holdout" \
      > "$root/sel_c1_holdout_gate.log" 2>&1
  fi
fi

printf 'PUBLISHING_SUMMARIES\n' > "$status"
publish=("$audit/summary.json" "$train/summary.json")
if [[ -s "$holdout/summary.json" ]]; then publish+=("$holdout/summary.json"); fi
git --git-dir=.git-sync --work-tree=. add -- "${publish[@]}"
if ! git --git-dir=.git-sync --work-tree=. diff --cached --quiet -- "${publish[@]}"; then
  git --git-dir=.git-sync --work-tree=. commit -m 'Record SEL-C0/C1 quality-token experiment results' -- "${publish[@]}"
fi
git --git-dir=.git-sync --work-tree=. push origin main
printf 'COMPLETE: %s\n' "$decision" > "$status"
