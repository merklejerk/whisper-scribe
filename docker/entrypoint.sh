#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-all}"
ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-8771}"

echo $ASR_HOST : $ASR_PORT

wait_for_port() {
  local host="${1:-127.0.0.1}"
  local port="${2:-$ASR_PORT}"
  for i in {1..60}; do
    if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "ASR server did not become ready on $host:$port" >&2
  return 1
}

start_asr() {
  cd /app/py
  # Allow user-provided ASR_ARGS to override defaults by appending them last
  asr_cmd=(uv run -q -m src.server --host "$ASR_HOST" --port "$ASR_PORT")
  if [[ -n "${ASR_ARGS:-}" ]]; then
    extra_asr_args=($ASR_ARGS)
    asr_cmd+=("${extra_asr_args[@]}")
  fi
  exec "${asr_cmd[@]}"
}

augment_bot_args() {
  local bargs="${BOT_ARGS:-}"
  # Add explicit --ai-service-url if ASR_HOST and ASR_PORT are set
  if [[ -n "${ASR_HOST:-}" && -n "${ASR_PORT:-}" ]]; then
    bargs+=" --ai-service-url ws://${ASR_HOST}:${ASR_PORT}"
  fi
  if [[ -n "${SESSION:-}" ]]; then
    bargs+=" --session-name ${SESSION}"
  fi
  if [[ -n "${PROFILE:-}" ]]; then
    bargs+=" --profile ${PROFILE}"
  fi
  if [[ -n "${GIST:-}" ]]; then
    shopt -s nocasematch || true
    if [[ "$GIST" =~ ^(1|true|yes|on)$ ]]; then
      bargs+=" --gist"
    fi
    shopt -u nocasematch || true
  fi
  if [[ -n "${PREV_SESSION:-}" ]]; then
    bargs+=" --prev-session ${PREV_SESSION}"
  fi
  export BOT_ARGS="$bargs"
}

start_bot() {
  : "${VOICE_CHANNEL_ID:?VOICE_CHANNEL_ID not set}"
  bot_cmd=(node js/dist/index.js bot "$VOICE_CHANNEL_ID")
  if [[ -n "${BOT_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_bot_args=($BOT_ARGS)
    bot_cmd+=("${extra_bot_args[@]}")
  fi
  exec "${bot_cmd[@]}"
}

case "$MODE" in
  asr)
    start_asr
    ;;
  bot)
    augment_bot_args "ws://asr:${ASR_PORT}"
    start_bot
    ;;
  all)
    # Start ASR locally, bound to ASR_HOST/ASR_PORT
    (cd /app/py && uv run -q -m src.server --host "$ASR_HOST" --port "$ASR_PORT") &
    ASR_PID=$!
    trap 'kill -TERM "$ASR_PID" 2>/dev/null || true' TERM INT EXIT
    # Wait for local port; use loopback which will work for 0.0.0.0 binds
    wait_for_port 127.0.0.1 "$ASR_PORT"
    augment_bot_args "ws://127.0.0.1:${ASR_PORT}"
    start_bot
    ;;
  *)
    echo "Unknown MODE: $MODE (expected: asr | bot | all)" >&2
    exit 64
    ;;
esac
