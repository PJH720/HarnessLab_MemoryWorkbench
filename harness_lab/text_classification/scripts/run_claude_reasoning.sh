#!/usr/bin/env bash
# Claude Sonnet 5.5 reasoning ablation on the frozen online-v2 memories.
# Two arms on the same model, same prompts, same effort: extended thinking on vs off.
# Needs ANTHROPIC_API_KEY in the environment; no call goes to the on-prem server.
#   usage: scripts/run_claude_reasoning.sh [--limit N]
set -euo pipefail
: "${ANTHROPIC_API_KEY:?export ANTHROPIC_API_KEY first}"
LAB="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$LAB/logs/reasoning-on"
mkdir -p "$LOGS"
run() {
  uv run --project "$LAB" --directory "$LAB" --with anthropic --with scipy --with pandas \
    python scripts/eval_reasoning.py --provider anthropic --model claude-sonnet-5-5 \
    --effort high --max-tokens 16000 --workers 8 --thinking "$1" "${@:2}"
}
run on "$@" > "$LOGS/claude-thinking-on.log" 2>&1 &
on=$!
run off "$@" > "$LOGS/claude-thinking-off.log" 2>&1 &
off=$!
wait $on && echo "thinking on: done" || echo "thinking on: FAILED (see $LOGS/claude-thinking-on.log)"
wait $off && echo "thinking off: done" || echo "thinking off: FAILED (see $LOGS/claude-thinking-off.log)"
