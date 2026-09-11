#!/usr/bin/env bash
# Runs ON ONE NODE (head or worker) — launched by tp2-serve.sh, not by hand.
#   ./tp2-rank.sh <node_rank> <headless:0|1>
# Everything is passed via env so this file contains no nested-quoting tricks: the fragile
# ssh -> heredoc -> docker run -> sh -lc nesting produced five launch bugs on the GLM stack.
set -euo pipefail

RANK="${1:?usage: tp2-rank.sh <node_rank> <headless 0|1>}"
HEADLESS="${2:-0}"
IMAGE="${IMAGE:?}"; CONTAINER="${CONTAINER:?}"; MODEL_DIR="${MODEL_DIR:?}"; TABLE_DIR="${TABLE_DIR:?}"
DRAFT_DIR="${DRAFT_DIR:?}"; SERVED_NAME="${SERVED_NAME:?}"; PORT="${PORT:?}"
TP_SIZE="${TP_SIZE:?}"; CTX="${CTX:?}"; SEQS="${SEQS:?}"; KV_BYTES="${KV_BYTES-}"; KV_DTYPE="${KV_DTYPE:?}"
MTP="${MTP:?}"; BATCHED_TOKENS="${BATCHED_TOKENS:?}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:?}"; SHM="${SHM:?}"
NCCL_IB_HCA="${NCCL_IB_HCA:?}"; FP8_HYBRID="${FP8_HYBRID:-1}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
ALLOW_LONG="${ALLOW_LONG:-0}"   # 1 = permit --max-model-len beyond the checkpoint native length
QPATCH="${QPATCH:-0}"           # 1 = mount patch/gptq-moe/auto_gptq.py over the engine's copy
# PLE (n-gram) table knobs. The table is ~49 GB and lives on NVMe, read via file-backed mmap
# (np.memmap + MADV_RANDOM). On GB10 the GPU and CPU share one 121 GB pool, so at GMU 0.83 only
# ~21 GB remains for page cache -- the table can NEVER be fully resident. PREFETCH is the designed
# mitigation: it hashes n-grams at batch-assembly time and overlaps the gather with decode instead
# of serialising ~31-280 ms/op into the critical path. It is OFF by default in the image.
PLE_PREFETCH="${PLE_PREFETCH:-0}"
PLE_WORKERS="${PLE_WORKERS:-32}"
PLE_FAST_ROWS="${PLE_FAST_ROWS:-512}"
HERE="$(cd "$(dirname "$0")" && pwd)"; NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?}"; CONTROL_IF="${CONTROL_IF:?}"
MASTER_ADDR="${MASTER_ADDR:?}"; MASTER_PORT="${MASTER_PORT:?}"; HEAD_ROCE_IP="${HEAD_ROCE_IP:?}"
WORKER_ROCE_IP="${WORKER_ROCE_IP:?}"
ROPE="${ROPE:-}"   # EMPTY = no --hf-overrides (native context); set by tp2-serve.sh
if [ -z "$ROPE" ] && [ -n "${ROPE_B64:-}" ]; then
  ROPE="$(printf %s "$ROPE_B64" | base64 -d)"   # base64 survives the env->ssh->bash hop intact
fi

# ---- RoCE-v2 IPv4-mapped GID index (re-numbers across reboots; never hardcode) --------------
# The GID value alone is NOT enough: gid[2] and gid[3] can both map our IP while only one is
# RoCE v2 (verified 2026-09-11: gid[2]=IB/RoCE v1, gid[3]=RoCE v2 for <HEAD_ROCE_IP>). Selecting
# v1 by address pattern alone would break NCCL. Require BOTH: type == "RoCE v2" AND our subnet.
gid_index() {
  local hca g entry gtype
  for hca in $(echo "$NCCL_IB_HCA" | tr ',' ' '); do
    for g in 0 1 2 3 4 5 6 7 8 9; do
      entry="$(cat "/sys/class/infiniband/$hca/ports/1/gids/$g" 2>/dev/null || true)"
      gtype="$(cat "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/$g" 2>/dev/null || true)"
      case "$gtype" in
        "RoCE v2") ;;
        *) continue ;;
      esac
      case "$entry" in
        0000:0000:0000:0000:0000:ffff:c0a8:*)     # <your-subnet>, IPv4-mapped
          echo "$g"; return 0 ;;
      esac
    done
  done
  echo 3   # our measured default (rocep1s0f0 RoCE v2 IPv4-mapped)
}
export NCCL_IB_GID_INDEX="$(gid_index)"
echo "[a5b-tp2] rank=$RANK hca=$NCCL_IB_HCA gid=$NCCL_IB_GID_INDEX headless=$HEADLESS"

HEADLESS_ARGS=()
[ "$HEADLESS" = 1 ] && HEADLESS_ARGS=(--headless)
# NB: never pass a possibly-empty variable as "$FLAG" — an empty string becomes a real (empty)
# argument and vLLM dies with "unrecognized arguments: ". Use arrays + the +idiom instead.

# MTP=0 must OMIT the flag entirely: num_speculative_tokens=0 fails pydantic (greater_than).
# VLLM_HOST_IP must be THIS node's own RoCE address: binding the head's IP on the worker dies
# with ZMQError "Cannot assign requested address" (the address does not exist there).
if [ "$RANK" = 0 ]; then SELF_ROCE_IP="$HEAD_ROCE_IP"; else SELF_ROCE_IP="$WORKER_ROCE_IP"; fi

# KV sizing: either an explicit byte pin (KV_BYTES) OR GMU-driven. Never both -- passing
# --kv-cache-memory-bytes makes vLLM ignore gpu-memory-utilization for the KV pool entirely.
KV_ARGS=()
if [ -n "${KV_BYTES:-}" ]; then
  KV_ARGS=(--kv-cache-memory-bytes "$KV_BYTES")
fi

# Optional patch-file bind mounts. File-level binds work fine over a .py; keep them OPT-IN so the
# default path is byte-identical to the unpatched image.
QPATCH_MOUNTS=()
if [ "$QPATCH" = 1 ]; then
  QPATCH_MOUNTS=(-v "${HERE}/patch/gptq-moe/auto_gptq.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_gptq.py:ro")
fi

# --max-model-len beyond the checkpoint's native max_position_embeddings requires explicit consent
# (vLLM refuses otherwise: "To allow overriding this maximum, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1").
ALLOW_LONG_ENV=()
[ "$ALLOW_LONG" = 1 ] && ALLOW_LONG_ENV=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)

SPEC_ARGS=()
if [ "${MTP:-3}" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"model\":\"${DRAFT_DIR}\"}")
fi

# --hf-overrides must be OMITTED when there is nothing to override: vLLM's parser rejects the
# empty object with "Value {}} cannot be converted to <function loads>" (2026-09-11).
ROPE_ARGS=()
case "${ROPE:-}" in
  ""|"{}") ;;
  *) ROPE_ARGS=(--hf-overrides "$ROPE") ;;
esac

# Mount the whole MODELS_DIR, exactly as the A5B single-node serve.sh does: the k10 draft dir
# lives beside the model, not inside it, so mounting only MODEL_DIR hides it from the container.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# --network host is REQUIRED for TP2: vLLM's message queue binds VLLM_HOST_IP (our RoCE address),
# which does not exist in the default bridge namespace -> ZMQError on bind (2026-09-11).
# Consequence: -p port mapping is invalid under host networking, so the in-container --port must
# BE the desired host port (8000 is our MediaLLMProxy, hence PORT=8100).
# --device /dev/infiniband + --ulimit memlock=-1 are BOTH required for RoCE: without the verbs
# device nodes NCCL cannot init any NET plugin ("Failed to initialize any NET plugin" ->
# "NCCL error: invalid usage"), and RDMA needs unlimited locked memory.
docker run -d --name "$CONTAINER" --gpus all --ipc=host --shm-size "$SHM" --network host \
  --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1 \
  -v "${MODELS_DIR}:${MODELS_DIR}:ro" \
  ${QPATCH_MOUNTS[@]+"${QPATCH_MOUNTS[@]}"} \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$CONTROL_IF" -e TP_SOCKET_IFNAME="$CONTROL_IF" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  -e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 \
  -e VLLM_HOST_IP="$SELF_ROCE_IP" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="$PLE_WORKERS" -e VLLM_PLE_MMAP_PREWARM=1 \
  -e VLLM_PLE_MMAP_PREFETCH="$PLE_PREFETCH" -e VLLM_PLE_MMAP_FAST_ROWS="$PLE_FAST_ROWS" \
  -e VLLM_PLE_MMAP_DIR="$TABLE_DIR" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID="$FP8_HYBRID" -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  ${ALLOW_LONG_ENV[@]+"${ALLOW_LONG_ENV[@]}"} \
  "$IMAGE" \
  "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" ${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"} \
  --load-format fastsafetensors --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size "$TP_SIZE" --nnodes 2 --node-rank "$RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${KV_ARGS[@]+"${KV_ARGS[@]}"} --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens "$BATCHED_TOKENS" \
  ${ROPE_ARGS[@]+"${ROPE_ARGS[@]}"} \
  -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]' --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} $EXTRA_ARGS

echo "[a5b-tp2] rank=$RANK launched as $CONTAINER"
