#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# ctrl.sh — one entry point for the A5B TP2 stack on 2× DGX Spark.
#
#   ./ctrl.sh start              launch both ranks (env file below)
#   ./ctrl.sh stop               stop both ranks
#   ./ctrl.sh restart            stop, wait, start
#   ./ctrl.sh status             containers + served model + KV/spec metrics
#   ./ctrl.sh gates              run the acceptance harness against the live stack
#   ./ctrl.sh env                show the effective configuration
#   ./ctrl.sh start <env-file>   use a different profile (e.g. the 1M one)
#
# Profiles live in config/*.env. Nothing is hardcoded here: every knob is read from
# the env file, so a new experiment is a new file, never an edit to the launcher.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_ENV="$HERE/config/a5b-tp2-512k-ep.env"

CMD="${1:-status}"
ENV_FILE="${2:-$DEFAULT_ENV}"

die() { echo "error: $*" >&2; exit 1; }

load_env() {
  [ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
  : "${MODEL_DIR:?MODEL_DIR missing from $ENV_FILE}"
  : "${CTX:?CTX missing from $ENV_FILE}"
  : "${PORT:?PORT missing from $ENV_FILE}"
}

case "$CMD" in
  start)
    load_env
    echo "── profile: $(basename "$ENV_FILE")"
    echo "   ctx=$CTX  gmu=$GPU_MEM_UTIL  kv=${KV_BYTES:-(GMU-driven)}  mtp=$MTP"
    echo "   extra_args=${EXTRA_ARGS:-<none>}  allow_long=$ALLOW_LONG  ple_prefetch=$PLE_PREFETCH"
    exec "$HERE/tp2-serve.sh"
    ;;

  stop)
    exec "$HERE/tp2-serve.sh" stop
    ;;

  restart)
    "$HERE/tp2-serve.sh" stop || true
    echo "waiting for VRAM to drain…"; sleep 8
    exec "$HERE/tp2-serve.sh"
    ;;

  status)
    load_env
    echo "── containers ─────────────────────────────────────────"
    printf '  head:   '; docker ps -a --format '{{.Names}} {{.Status}}' | grep -E "^${CONTAINER:-qwen38-flash-tp2} " || echo "not running"
    printf '  worker: '; ssh -o ConnectTimeout=8 "<user>@${WORKER_ROCE_IP:-<WORKER_ROCE_IP>}" \
      "docker ps -a --format '{{.Names}} {{.Status}}' | grep -E '^${CONTAINER:-qwen38-flash-tp2} ' || echo 'not running'" 2>/dev/null || echo "unreachable"
    echo
    echo "── served model ───────────────────────────────────────"
    curl -s --max-time 8 "http://127.0.0.1:${PORT}/v1/models" \
      | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)['data'][0]
    print('  ', d['id'], ' max_model_len=', d.get('max_model_len'))
except Exception:
    print('  (not serving)')" 2>/dev/null || echo "  (not serving)"
    echo
    echo "── metrics ────────────────────────────────────────────"
    curl -s --max-time 8 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | grep -E \
      '^vllm:prefix_cache_(hits|queries)_total|^vllm:kv_cache_usage_perc' | sed 's/^/  /' || true
    echo
    echo "── spec decode (last window) ──────────────────────────"
    docker logs "${CONTAINER:-qwen38-flash-tp2}" 2>&1 | grep -oE 'Avg Draft acceptance rate: [0-9.]+%' | tail -1 | sed 's/^/  /' || echo "  (none)"
    echo
    echo "── PLE mmap (SSD pressure) ────────────────────────────"
    docker logs "${CONTAINER:-qwen38-flash-tp2}" 2>&1 | grep -oE 'PLE mmap stats.*' | tail -1 | sed 's/^/  /' || echo "  (none)"
    ;;

  gates)
    load_env
    BASE_URL="http://127.0.0.1:${PORT}" MODEL="$SERVED_NAME" EXPECT_CTX="$CTX" exec "$HERE/gate.sh"
    ;;

  env)
    load_env
    echo "profile: $ENV_FILE"
    for v in IMAGE CONTAINER SERVED_NAME PORT MODEL_DIR TABLE_DIR DRAFT_DIR TP_SIZE \
             EXTRA_ARGS CTX ALLOW_LONG GPU_MEM_UTIL KV_BYTES KV_DTYPE MTP BATCHED_TOKENS \
             SEQS PLE_PREFETCH PLE_WORKERS PLE_FAST_ROWS FP8_HYBRID QPATCH SHM; do
      printf '  %-14s = %s\n' "$v" "${!v-}"
    done
    ;;

  *)
    die "usage: $0 {start|stop|restart|status|gates|env} [env-file]"
    ;;
esac
