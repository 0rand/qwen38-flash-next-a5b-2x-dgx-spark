#!/usr/bin/env bash
# TP2 launcher for lane 3 — Qwen3.8-Flash-Next A5B INT4-AutoRound across BOTH Sparks.
#
# DRAFT FOR REVIEW — nothing here has been booted yet.
#   ./tp2-serve.sh          both: worker (rank 1) first, then head (rank 0)
#   ./tp2-serve.sh head     head only
#   ./tp2-serve.sh worker   worker only
#   ./tp2-serve.sh stop     both ranks down
#
# Design: this file only ships env + runs tp2-rank.sh on each node. All quoting is plain
# argument passing — the ssh->heredoc->docker-run->sh-lc nesting is what gave the GLM stack
# five launch bugs. Per-rank logic lives in tp2-rank.sh (same file on both nodes).
#
# WHY TP2: TP1 gives 644,732 KV tokens and one 262,144 context. The spec is 512K context with
# >=2M KV. KV bf16 ~31 KB/token (measured: 20g -> 644,732 tok), so 2M = ~62 GB, plus ~71 GB
# weights — cannot fit one Spark.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RANK_SH="$HERE/tp2-rank.sh"

# ─────────────────────────── SETTINGS ───────────────────────────
export IMAGE="${IMAGE:-qwen38-flash-dgx:a5b-int4}"
export CONTAINER="${CONTAINER:-qwen38-flash-tp2}"
export SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next-a5b}"
export PORT="${PORT:-8100}"                 # host port on the HEAD (8000 = our MediaLLMProxy)

export MODELS_DIR="${MODELS_DIR:-/var/tmp/models}"
export MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"
export TABLE_DIR="${TABLE_DIR:-$MODEL_DIR/ple-table}"
export DRAFT_DIR="${DRAFT_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-draft-k10}"

export TP_SIZE="${TP_SIZE:-2}"
export CTX="${CTX:-262144}"                 # native; raise to 524288 once YaRN is proven
export SEQS="${SEQS:-8}"
# 2M tok -> ~62g | 1.5M -> ~47g | 1M -> ~31g.  Semantics of --kv-cache-memory-bytes under TP
# are UNVERIFIED here: start at 62g, read "GPU KV cache size" from the log, adjust to hit 2M.
export KV_BYTES="${KV_BYTES:-62g}"
export KV_DTYPE="${KV_DTYPE:-auto}"         # auto(bf16) | fp8_e4m3 (~1.9x ctx, ~10% slower)
export MTP="${MTP:-3}"
# A5B ships a hybrid FP8-blockwise quant (300 layers). Its SHARED-EXPERT gate_up_proj has
# intermediate 640 -> TP2 splits to 320, and fp8 block_n=128 does not divide 320, so weight
# creation fails. FP8_HYBRID=0 turns that patch off (layers keep their original quant).
export FP8_HYBRID="${FP8_HYBRID:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"    # pass-through for experiments (e.g. --enable-expert-parallel)
export BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.01}" # tiny on purpose: KV_BYTES sets the cache
export SHM="${SHM:-16g}"

# fabric (values proven on our GLM stack)
export HEAD_ROCE_IP="${HEAD_ROCE_IP:-<HEAD_ROCE_IP>}"
export WORKER_ROCE_IP="${WORKER_ROCE_IP:-<WORKER_ROCE_IP>}"
export WORKER_SSH="${WORKER_SSH:-<user>@<WORKER_ROCE_IP>}"   # fabric, NOT 'dragonforce' (LAN)
export WORKER_DIR="${WORKER_DIR:-/home/<user>/dockers/qwen38fn-a5b-lane3}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0}"   # both rails; verified GID 3 = RoCE v2
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export CONTROL_IF="${CONTROL_IF:-enp1s0f0np0}"
export MASTER_ADDR="${MASTER_ADDR:-$HEAD_ROCE_IP}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 512K needs scaling: checkpoint is native 262,144 (rope_theta 1e7, no rope_scaling baked).
# UNVERIFIED on this stack's sparse-attention path. If boot fails at 512K, fall back to
# CTX=262144 with ROPE='{}' to establish a working TP2 baseline before attacking context.
#
# NB: do NOT write this as ROPE="${ROPE:-{...json...}}" — bash's ${VAR:-word} delimiter matching
# is confused by braces inside the word when VAR is already set, and silently appends a stray
# '}' (verified: ROPE='{}' became '{}}', which vLLM rejects as
# "argument --hf-overrides: Value {}} cannot be converted").
ROPE="${ROPE:-}"        # EMPTY = omit --hf-overrides entirely (native context, no scaling).
                        # 512K needs a YaRN overlay config (our 27B pattern: rope_type=yarn,
                        # factor=2.0, original_max_position_embeddings=262144, nested under
                        # text_config) — deferred until TP2 plumbing is proven.
export ROPE
# ────────────────────────────────────────────────────────────────

rank_env() {
  # One VAR=value per line, exported for the remote shell.
  env | grep -E '^(IMAGE|CONTAINER|SERVED_NAME|PORT|MODELS_DIR|MODEL_DIR|TABLE_DIR|DRAFT_DIR|TP_SIZE|CTX|SEQS|KV_BYTES|KV_DTYPE|MTP|FP8_HYBRID|EXTRA_ARGS|BATCHED_TOKENS|GPU_MEM_UTIL|SHM|HEAD_ROCE_IP|WORKER_ROCE_IP|NCCL_IB_HCA|NCCL_SOCKET_IFNAME|CONTROL_IF|MASTER_ADDR|MASTER_PORT|ROPE)=' | sed 's/^/export /'
}

run_local() { bash "$RANK_SH" "$1" "$2"; }

run_remote() {
  local rank="$1" headless="$2"
  ssh -o ConnectTimeout=10 "$WORKER_SSH" "mkdir -p '$WORKER_DIR'"
  rsync -a "$RANK_SH" "$WORKER_SSH:$WORKER_DIR/tp2-rank.sh"
  { rank_env; echo "bash $WORKER_DIR/tp2-rank.sh $rank $headless"; } | ssh "$WORKER_SSH" 'bash -s'
}

case "${1:-both}" in
  worker) run_local 1 1 ;;
  head)   run_local 0 0 ;;
  both)
    echo "[a5b-tp2] rank 1 (worker @ $WORKER_ROCE_IP) first…"
    run_remote 1 1
    sleep 20
    echo "[a5b-tp2] rank 0 (head @ $HEAD_ROCE_IP)…"
    run_local 0 0
    echo
    echo "[a5b-tp2] follow:  docker logs -f $CONTAINER"
    echo "[a5b-tp2] health:  curl -s http://localhost:$PORT/health"
    echo "[a5b-tp2] then:    cd $HERE && BASE_URL=http://127.0.0.1:$PORT MODEL=$SERVED_NAME ./gate.sh"
    ;;
  stop)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    ssh "$WORKER_SSH" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
    echo "[a5b-tp2] both ranks stopped"
    ;;
  *) echo "usage: $0 [both|head|worker|stop]" >&2; exit 1 ;;
esac
