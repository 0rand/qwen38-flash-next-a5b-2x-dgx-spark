#!/usr/bin/env bash
# Lane-3 acceptance gates. Stages run in order; abort on first failure.
#   ./gate.sh            # stages A,B,C (fast, ~1 min + 2 long prefills)
#   ./gate.sh --perf     # also stage D (llama-benchy via TEB --perf-only)
#   ./gate.sh --quality  # also stage E (TEB hardmode; slow, ~15 min)
#
# Exit codes: 10 health, 20 functional, 30 PREFIX CACHE (the kill switch), 40 perf tool, 50 quality tool
set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:8100}"
MODEL="${MODEL:-qwen3.8-flash-next}"
EXPECT_CTX="${EXPECT_CTX:-262144}"
TOKENS="${TOKENS:-8000}"
STAGE_PERF=0; STAGE_QUALITY=0
for a in "$@"; do
  case "$a" in
    --perf) STAGE_PERF=1 ;;
    --quality) STAGE_QUALITY=1; STAGE_PERF=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

c()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
no() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

api() { curl -s --max-time "${2:-120}" "$BASE_URL$1"; }

# ─────────────────────────────── Gate A — health ───────────────────────────────
c "Gate A — health"
MODELS=$(api /v1/models 30)
if [ -z "$MODELS" ]; then no "no response from $BASE_URL/v1/models"; exit 10; fi
SERVED=$(printf '%s' "$MODELS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$SERVED" ] || { no "could not parse /v1/models"; exit 10; }
ok "serving '$SERVED'"
CTX=$(api /v1/models 30 | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len","?"))' 2>/dev/null)
ok "max_model_len=$CTX (expected $EXPECT_CTX)"
METRICS=$(curl -s --max-time 10 "$BASE_URL/metrics" || true)
ENGINE=$(docker logs "${CONTAINER:-qwen38fn-tp2}" 2>&1 | grep -oE 'vLLM [0-9][^ ]*' | tail -1)
[ -n "$ENGINE" ] && ok "engine: $ENGINE"

# ────────────────────────────── Gate B — functional ────────────────────────────
c "Gate B — functional (4 gates)"
PLAIN=$(curl -s --max-time 180 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: READY\"}],
  \"max_tokens\":32,\"temperature\":0}" | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d["choices"][0]["message"].get("content") or "").strip())' 2>/dev/null)
[ -n "$PLAIN" ] && ok "plain chat: '$PLAIN'" || { no "plain chat empty"; exit 20; }

TOOLREQ=$(curl -s --max-time 180 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Prague? Use the tool.\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",
    \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
  \"tool_choice\":\"required\",\"max_tokens\":128,\"temperature\":0}" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);tc=d["choices"][0]["message"].get("tool_calls");print("yes" if tc else "no")' 2>/dev/null)
[ "$TOOLREQ" = "yes" ] && ok "tool_choice=required → valid tool_calls" || { no "tool_choice=required ignored"; exit 20; }

PREFILL=$(curl -s --max-time 300 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$(python3 - <<PY
import json
p = ("The kinetic strike protocol evaluates momentum exhaustion across correlated instruments. ") * 400
print(json.dumps({"model":"$MODEL","messages":[{"role":"user","content":p + "\nReply with one word."}],
                  "max_tokens":96,"temperature":0}))
PY
)" | python3 -c '
import json,sys
d=json.load(sys.stdin)
m=d["choices"][0]["message"]
# Thinking-enabled models put the answer in "reasoning" until thinking completes;
# a small max_tokens budget can be consumed entirely by reasoning.
print((m.get("content") or m.get("reasoning") or "").strip()[:40]) if (m.get("content") or m.get("reasoning")) else print("")' 2>/dev/null)
[ -n "$PREFILL" ] && ok "chunked prefill (~3.7K tok): '$PREFILL'" || { no "long prefill failed"; exit 20; }

# ──────────────────── Gate C — PREFIX CACHE (the kill switch) ──────────────────
c "Gate C — prefix cache (kill switch)"
m() { curl -s --max-time 10 "$BASE_URL/metrics" | grep -E '^vllm:prefix_cache_hits_total' | awk '{print $2}' | head -1; }
H0=$(m); [ -n "$H0" ] || { no "no vllm:prefix_cache_hits_total in /metrics — cannot verify"; exit 30; }
ok "hits_total before: $H0"

# A prefix hit requires a shared prefix of at least ONE full scheduler block. That block
# size is model/hybrid-specific and is NOT the --block-size you passed: read the resolved
# value from cache_config_info. Measured examples: 16 (flash-next), 3584 (GLM-5.3 NVFP4).
# Probe with ~4 blocks so a hit is possible, with a floor for small-block models.
BS=$(curl -s --max-time 10 "$BASE_URL/metrics" | grep -oE 'block_size="[0-9]+"' | head -1 | grep -oE '[0-9]+')
BS=${BS:-16}
TARGET=$(( BS * 4 )); [ "$TARGET" -lt 8192 ] && TARGET=8192
ok "resolved block_size=$BS -> probing with ~$TARGET tokens (4 blocks, floor 8192)"

LONG=$(python3 -c "
import random
bs = $TARGET
# CRITICAL: use DIVERSE text, not repeated filler. Repetition compresses under BPE
# (~2.4x fewer tokens than the char count suggests) and can leave the prompt below one
# block, producing a false 'cache dead' verdict. Random word sequences tokenize near
# ~4.5 chars/token and never repeat a block.
random.seed(42)
words = ('kinetic strike protocol momentum exhaustion volatility options chain depth '
         'signal regime liquidity gamma delta hedging notional exposure drawdown '
         'correlation residual basis spread carry convexity skew kurtosis tail risk').split()
n = int(bs * 4.5 / 7) + 10
print(' '.join(random.choice(words) for _ in range(n * 3)))
" > /tmp/gate-c-probe.txt)
ok "probe corpus written: $(wc -c < /tmp/gate-c-probe.txt) chars"
for i in 1 2; do
  # Large prompts exceed the shell's argv limit if inlined -> write the JSON payload to a
  # file and use curl -d @file (hit this at ~14K tokens / 64KB on GLM).
  python3 -c "
import json
p = open('/tmp/gate-c-probe.txt').read()
json.dump({'model':'$MODEL',
           'messages':[{'role':'user','content':p + chr(10) + 'Reply with the single word READY.'}],
           'max_tokens':64,'temperature':0}, open('/tmp/gate-c-body.json','w'))
"
  curl -s --max-time 900 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d @/tmp/gate-c-body.json > /dev/null
  sleep 2
done
H1=$(m)
ok "hits_total after 2 identical sends: $H1"
DELTA=$(python3 -c "print(int(float('${H1:-0}') - float('${H0:-0}')))")
if [ "$DELTA" -gt 0 ]; then
  ok "PREFIX CACHE WORKING (+$DELTA cached tokens)"
else
  no "PREFIX CACHE DEAD (delta=$DELTA) — long context is unusable, ABORT the lane"
  exit 30
fi

# ───────────────────────── Gate D — speed (informational) ─────────────────────
if [ "$STAGE_PERF" = 1 ]; then
  c "Gate D — speed (llama-benchy via TEB --perf-only)"
  TEB=$(command -v tool-eval-bench || true)
  if [ -z "$TEB" ]; then no "tool-eval-bench not on PATH"; exit 40; fi
  LOG="/var/tmp/lane3-perf-$(date +%H%M%S).log"
  tool-eval-bench --perf-only --benchy-runs 1 --concurrency 1,2,4 \
    --depth 0,2048,8192 --tg 512 --pp 1024 --base-url "$BASE_URL" \
    --tokenizer "${TOKENIZER:-}" > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { no "perf run failed (see $LOG)"; exit 40; }
  ok "perf logged: $LOG"
  grep -A14 'llama-benchy Results' "$LOG" | tail -14
fi

# ───────────────────────── Gate E — quality (informational) ───────────────────
if [ "$STAGE_QUALITY" = 1 ]; then
  c "Gate E — quality (TEB hardmode)"
  tool-eval-bench --hardmode --seed 42 --temperature 0 --max-tokens 16384 \
    --base-url "$BASE_URL" --model "$MODEL" 2>&1 | tail -30
  ok "quality run complete (compare vs 86-87 current stack; 89=good, 93-94=exceptional)"
fi

c "ALL GATES PASSED"
