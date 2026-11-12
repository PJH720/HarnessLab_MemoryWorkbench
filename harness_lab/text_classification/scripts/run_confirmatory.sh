#!/usr/bin/env bash
# Confirmatory evaluation on the Symptom2DiseaseHoldout test set (scripts/build_holdout.py).
# Frozen seed 42/123/456 memories of the seven preregistered systems are loaded and
# evaluated once; nothing is trained. At most 12 requests are in flight (one job at a time).
#   usage: HARNESS_API_BASE=<endpoint>/v1 RUN_DIR=<logs/confirm-YYYYMMDD> scripts/run_confirmatory.sh
#   RUN_DIR must contain frozen_memories/<system>/seed<N>/memory.json.
set -euo pipefail
: "${HARNESS_API_BASE:?set HARNESS_API_BASE to the model endpoint; it is never stored in the repository}"
: "${RUN_DIR:?set RUN_DIR to the run folder under logs/}"
LAB="$(cd "$(dirname "$0")/.." && pwd)"; cd "$LAB"
MODEL="openai/Inferact/Qwen3.8-Flash-Next-NVFP4"
SYSTEMS="no_memory fewshot_all knn_top6 contrast_online reflexion_notes cue_bank_verified cue_bank"
SEEDS="${SEEDS:-42 123 456}"
export TC_DISABLE_THINKING=1 OPENAI_API_KEY=onprem-placeholder TC_MAX_WORKERS=12
export TEXT_CLASSIFICATION_LLM_CACHE_DIR="${TEXT_CLASSIFICATION_LLM_CACHE_DIR:-$HOME/.cache/text-classification/litellm-holdout}"

for s in $SEEDS; do
  for m in $SYSTEMS; do
    mem="$RUN_DIR/frozen_memories/$m/seed$s/memory.json"
    rd="$RUN_DIR/eval/$m/seed$s"; mkdir -p "$rd"
    if [ -f "$rd/test.json" ]; then echo "seed$s $m already done"; continue; fi
    env PYTHONPATH="$(dirname "$LAB")" uv run --project "$LAB" python -m text_classification.inner_loop \
      --dataset Symptom2DiseaseHoldout --model "$MODEL" --api-base "$HARNESS_API_BASE" \
      --mode online --temperature 0.0 --num-train 60 --num-val 50 --num-test 572 \
      --memory "agents/$m.py" --seed "$s" --load-memory "$mem" \
      --test-output "$rd/test.json" --predictions-output "$rd/predictions.jsonl" \
      --log "$rd/log.jsonl" > "$rd/run.out" 2>&1 \
      && echo "seed$s $m done" || echo "seed$s $m FAILED (see $rd/run.out)"
  done
done
