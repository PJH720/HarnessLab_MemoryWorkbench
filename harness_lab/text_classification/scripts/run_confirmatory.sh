#!/usr/bin/env bash
# Confirmatory evaluation on the Symptom2DiseaseHoldout test set (scripts/build_holdout.py).
# Frozen seed 42/123/456 memories of the seven preregistered systems are loaded and
# evaluated once; nothing is trained. At most 12 requests are in flight (one job at a time).
# Before each job the endpoint must answer and serve MODEL: a short outage is waited out,
# a long one stops the chain so the remaining jobs do not fail one by one.
#   usage: HARNESS_API_BASE=<endpoint>/v1 RUN_DIR=<logs/confirm-YYYYMMDD> scripts/run_confirmatory.sh
#   RUN_DIR must contain frozen_memories/<system>/seed<N>/memory.json.
set -euo pipefail
: "${HARNESS_API_BASE:?set HARNESS_API_BASE to the model endpoint; it is never stored in the repository}"
: "${RUN_DIR:?set RUN_DIR to the run folder under logs/}"
LAB="$(cd "$(dirname "$0")/.." && pwd)"; cd "$LAB"
MODEL="openai/Inferact/Qwen3.8-Flash-Next-NVFP4"
# Order only; every system still runs. Jobs with both result files are skipped on rerun.
SYSTEMS="${SYSTEMS:-no_memory fewshot_all knn_top6 contrast_online reflexion_notes cue_bank_verified cue_bank}"
SEEDS="${SEEDS:-42 123 456}"
ENDPOINT_RETRIES="${ENDPOINT_RETRIES:-30}"
ENDPOINT_RETRY_SECONDS="${ENDPOINT_RETRY_SECONDS:-60}"
export TC_DISABLE_THINKING=1 OPENAI_API_KEY=onprem-placeholder TC_MAX_WORKERS=12
export TEXT_CLASSIFICATION_LLM_CACHE_DIR="${TEXT_CLASSIFICATION_LLM_CACHE_DIR:-$HOME/.cache/text-classification/litellm-holdout}"

endpoint_ready() {
  local models
  models="$(curl -sf -m 15 "$HARNESS_API_BASE/models")" || return 1
  [[ "$models" == *"\"${MODEL#openai/}\""* ]]
}

wait_for_endpoint() {
  local attempt
  for ((attempt = 1; attempt <= ENDPOINT_RETRIES; attempt++)); do
    if endpoint_ready; then return 0; fi
    if ((attempt < ENDPOINT_RETRIES)); then sleep "$ENDPOINT_RETRY_SECONDS"; fi
  done
  return 1
}

for s in $SEEDS; do
  for m in $SYSTEMS; do
    mem="$RUN_DIR/frozen_memories/$m/seed$s/memory.json"
    rd="$RUN_DIR/eval/$m/seed$s"; mkdir -p "$rd"
    if [ -f "$rd/test.json" ] && [ -f "$rd/predictions.jsonl" ]; then echo "seed$s $m already done"; continue; fi
    if ! wait_for_endpoint; then
      echo "seed$s $m not started: endpoint unreachable or not serving ${MODEL#openai/}; stopping"
      exit 3
    fi
    env PYTHONPATH="$(dirname "$LAB")" uv run --project "$LAB" python -m text_classification.inner_loop \
      --dataset Symptom2DiseaseHoldout --model "$MODEL" --api-base "$HARNESS_API_BASE" \
      --mode online --temperature 0.0 --num-train 60 --num-val 50 --num-test 572 \
      --memory "agents/$m.py" --seed "$s" --load-memory "$mem" \
      --test-output "$rd/test.json" --predictions-output "$rd/predictions.jsonl" \
      --log "$rd/log.jsonl" > "$rd/run.out" 2>&1 \
      && echo "seed$s $m done" || echo "seed$s $m FAILED (see $rd/run.out)"
  done
done
