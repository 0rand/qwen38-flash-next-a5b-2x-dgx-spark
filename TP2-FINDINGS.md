# Qwen3.8-Flash-Next A5B INT4-AutoRound on **2× DGX Spark (GB10)**, TP=2

**Status: WORKING.** This document records how we got a single-Spark–only recipe running across two
nodes, the two blockers that had to be solved, and the exact numbers. It is written so that a future
MTP / DFlash variant of the same checkpoint family can be brought up without repeating the hunt.

Environment: 2× NVIDIA DGX Spark (GB10, sm_121), RoCE fabric between them, image
`qwen38-flash-dgx:a5b-int4` (built from Saren's `qwen3.8-Flash-DGX-AutoRound` @ `01c5914f…`, vLLM
`0.1.dev20073+g8e685d198`), checkpoint `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`.

---

## 1. TL;DR

1. **The checkpoint cannot be TP-sharded as-is.** Its hybrid quant stores 300 layers as
   **blockwise-fp8 (128×128)** with `weight_scale_inv`. vLLM's fp8 block-shape validator requires
   `(dim / TP) % 128 == 0`. The **shared expert** has a **640** dimension → TP2 gives 320 → not a
   multiple of 128 → hard failure. 640/TP is only a multiple of 128 when **TP=1**, so *no* TP>1 can
   work while those weights stay fp8.
   **Fix: requant — dequantize exactly those tensors to bf16** (§3).
2. **With that fixed, MTP still fails at TP2** — the MTP layer's experts fail
   `check_moe_marlin_supports_layer`, fall back to `MoeWNA16`, and the image's loader expects Marlin
   parameter names → `AttributeError: … has no parameter 'w2_weight'`.
   **Fix: `--enable-expert-parallel`** (§4). Config-level `dynamic` exclusions do **not** work here.
3. Result: both ranks load, all functional gates pass, prefix caching works, MTP acceptance
   ~61-64% on agentic traffic. Measured trade-off in §6 — **TP2 is ~25-30% slower single-stream
   than TP1** (interconnect latency × MTP's multiple forward passes per step); TP2's wins are
   aggregate throughput under concurrency and a ~2× larger KV pool.

---

## 2. Pre-flight analysis (do this first — it is cheap and it de-risks everything)

Before changing anything, enumerate **which** quantized tensors would break under TP. Script pattern:

```python
# read model.safetensors.index.json + safetensors metadata
# for every '.weight' that has a '.weight_scale_inv' sibling:
#     a dimension is TP-safe iff (dim / TP) % 128 == 0 on the sharded axis
```

Our result for this checkpoint at TP=2:

```
300 fp8 weights with scales
  156 already TP2-safe
  144 PROBLEMATIC  -> all shared_expert (48 layers x down_proj/gate_proj/up_proj)
                   -> all inside ONE file: model-healed-shared-expert.safetensors
```

Shapes that break: `down_proj (2560, 640)` and `gate/up_proj (640, 2560)` — the **640** dimension.
Note the fused `gate_up_proj` does *not* rescue it: `MergedColumnParallelLinear` validates each
sub-partition (640/TP) separately.

Having this list meant one fix instead of N boot cycles. **Do this before touching weights.**

---

## 3. Fix #1 — requant the shared expert to bf16 (the "requant")

### Why it works
vLLM's fp8-hybrid detection rule is:

```python
fp8_layers = {name[:-len(".weight")] for name, info in metadata.items()
              if name.endswith(".weight")
              and info["dtype"] == "F8_E4M3"
              and name[:-len(".weight")] + ".weight_scale_inv" in metadata}
```

A layer is treated as fp8 **iff its `.weight` is F8_E4M3 AND a `.weight_scale_inv` sibling exists.**
So: write the weight as **bf16** and **drop the scale sibling** → the layer is no longer fp8 → it loads
as a plain (unquantized) bf16 layer → **no block-shape constraint → shards freely under any TP.**

The checkpoint's `quantization_config.dynamic` already excludes `.*shared_expert.*` from GPTQ, so the
unquantized path is exactly what the config intends.

### Procedure
```python
from safetensors import safe_open
from safetensors.torch import save_file
import torch

BLOCK = 128
with safe_open(SRC + "/model-healed-shared-expert.safetensors", "pt") as h:
    out = {}
    for k in [x for x in h.keys() if x.endswith(".weight")]:
        w = h.get_tensor(k)                       # fp8 e4m3
        s = h.get_tensor(k + "_scale_inv")        # fp32 [ceil(o/128), ceil(i/128)]
        o, i = w.shape
        s_exp = s.repeat_interleave(BLOCK, 0)[:o].repeat_interleave(BLOCK, 1)[:, :i]
        dq = (w.to(torch.float32) * s_exp).to(torch.bfloat16)
        assert torch.isfinite(dq).all()
        out[k] = dq
save_file(out, NEWDIR + "/model-healed-shared-expert-bf16.safetensors")
```

Then build a new checkpoint directory that **symlinks every other file** from the original (so the
original is untouched) and contains a **patched `model.safetensors.index.json`**:

* retarget each converted `…shared_expert….weight` to the new file,
* **delete** its `…weight_scale_inv` entry (this is what disarms the fp8 detector).

Cost on this checkpoint: 144 tensors, 235.9M params, fp8 236 MB → **bf16 472 MB** (+236 MB total).

### Verify before booting
```python
still_fp8 = [k for k in weight_map
             if "shared_expert" in k and k.endswith(".weight")
             and dtype(k) == "F8_E4M3" and (k + "_scale_inv") in weight_map]
assert not still_fp8   # must be empty
```

> **Warning (we hit this):** if the new checkpoint dir is a symlink farm, `config.json` etc. are
> symlinks. Writing to a symlinked file **writes through into the original checkpoint**. Either
> `test -L` before every write, or replace the link with a real file first. We restored ours from a
> backup and verified by diff — but it is an easy way to corrupt a 127 GB checkpoint.

---

## 4. Fix #2 — `--enable-expert-parallel` (the part that unlocks MTP)

### Symptom
```
WARNING Layer 'mtp.layers.48.mlp.experts' is not supported by GPTQMoeMarlin.
        Falling back to Moe WNA16 kernels.
ERROR   AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight'
        for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight'
```

### Root cause
At TP=2 the expert weights get split along their dimensions, so the **640** dimension again becomes
320, `check_moe_marlin_supports_layer()` fails, and vLLM falls back to `MoeWNA16` — whose parameter
naming does not match what the model's MTP loader (`models/qwen3_8_flash_next/nvidia/mtp.py`)
expects. Note the fallback is decided on **layer shape**, not on the weights' dtype: the MTP experts
are in fact **bf16** in the checkpoint (they are shipped in `model_extra_tensors.safetensors`), yet
the layer is still evaluated as a quantized MoE.

### What does NOT work (we tried it, so you don't have to)
**Config-level exclusions cannot fix this.** We added `-:mtp\..*` (and the checkpoint already has
`-:.*layers\.48\..*`) to `quantization_config.dynamic`. The warning persists, because
`auto_gptq.AutoGPTQConfig.get_quant_method()` decides per `RoutedExperts` layer purely via the
Marlin support check — **it never consults `dynamic`**. Verified by reading the code path.
(Also verified: `VLLM_FP8_HYBRID=0` is not a workaround — it breaks the A5B loader with
`AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'`; the fp8-hybrid patch
and the loader are interdependent.)

### What works
```
--enable-expert-parallel
```
EP distributes *experts* across ranks instead of splitting each expert's matrices, so the 640
dimension stays whole → the Marlin support check passes → naming stays consistent → the MTP layer
loads. This single flag took the stack from "engine core initialization failed" to serving.

---

## 5. Recipe (working launch)

Both nodes, **container must use host networking** (see §7). Values in `<…>` are yours.

```bash
IMAGE=qwen38-flash-dgx:a5b-int4
MODEL_DIR=<checkpoint root>/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe
DRAFT_DIR=<checkpoint root>/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe-draft-k10

docker run -d --name qwen38-flash-tp2 --gpus all --ipc=host --shm-size 16g \
  --network host \
  --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1 \
  -v <checkpoint root>:<checkpoint root>:ro \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=<hca0>,<hca1> -e NCCL_SOCKET_IFNAME=<roce_netdev> \
  -e GLOO_SOCKET_IFNAME=<roce_netdev> -e TP_SOCKET_IFNAME=<roce_netdev> \
  -e NCCL_IB_GID_INDEX=<gid> \
  -e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 \
  -e VLLM_HOST_IP=<THIS_NODE_own_ip> \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 \
  -e VLLM_PLE_MMAP_DIR=$MODEL_DIR/ple-table \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" \
  "$MODEL_DIR" --served-model-name qwen3.8-flash-next-a5b \
  --host 0.0.0.0 --port 8100 [--headless] \
  --load-format fastsafetensors --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 --nnodes 2 --node-rank <0|1> \
  --master-addr <HEAD_ROCE_IP> --master-port 29500 \
  --enable-expert-parallel \
  --max-model-len 262144 --max-num-seqs 8 \
  --gpu-memory-utilization 0.01 --kv-cache-memory-bytes 20g --kv-cache-dtype auto \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]' \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"model":"'"$DRAFT_DIR"'"}'
```

Notes:
* `--gpu-memory-utilization 0.01` is intentional: `--kv-cache-memory-bytes` sizes the cache instead.
* The draft dir is the model with `num_experts_per_tok: 10` ("k10", the faster draft) and symlinks to
  the same (tp2-safe) weights.
* Rank 1 is the same command plus `--headless` and its own `--node-rank`; only rank 0 binds the port.
* **The draft dir must exist on every node** — it is created by `setup.sh` on the head only, and at
  TP it is opened by the worker too.

---

## 6. Measured results

### Gating (both configurations)
| gate | TP1 | TP2 + MTP3 + EP |
|---|---|---|
| plain chat | PASS | PASS |
| `tool_choice=required` → valid tool_calls | PASS | PASS |
| chunked prefill (~3.7K tok) | PASS | PASS |
| **prefix cache** (`prefix_cache_hits_total` delta) | PASS | **PASS (+16,800 tokens)** |

### Memory / capacity
| | TP1 | TP2 |
|---|---|---|
| weights per rank | 71.09 GiB | **34.98-36.99 GiB** |
| KV cache @ `--kv-cache-memory-bytes 20g` | 644,732 tok | **1,251,206 tok** |
| per-rank usage | ~93 GiB | **~55 GiB of 121 GiB** |

### Throughput (llama-benchy, pp2048/tg1024, `--tg 1024 --pp 2048`, depth/concurrency sweep)
| depth | c1 | c2 | c4 |
|---|---|---|---|
| d0 | 38.2 | 49.7 | 69.3 |
| d2048 | 35.0 | 56.6 | **90.8** |
| d8192 | 29.3 | 51.8 | 74.1 |

Same harness on **TP1** (single Spark): c1 = **48.9 / 45.4 / 58.5**.

**Conclusion: at TP2 the single-stream rate is ~25-30% LOWER.** Cross-node all-reduce per layer
costs more than the halved weight traffic saves, and MTP multiplies it (multiple forward passes per
verification step) — i.e. latency-bound decode pays the interconnect tax. TP2's real wins are
**aggregate throughput at concurrency** and the **~2× KV pool**.

MTP acceptance on agentic traffic: **61-64%** (on synthetic prose: 38-42% — prose is the worst case
for speculative decoding; do not judge acceptance from a prose benchmark).

---

## 7. Gotchas that cost us a boot cycle each

1. **`--network host` is mandatory** for multi-node. vLLM's message queue binds `VLLM_HOST_IP`; in
   the default bridge namespace that address does not exist → `ZMQError: Cannot assign requested
   address`. Consequence: `-p` port mapping is invalid, so the in-container `--port` **must be the
   desired host port**.
2. **`VLLM_HOST_IP` must be *this node's own* RoCE IP.** Using the head's IP on the worker reproduces
   the ZMQ error above.
3. **`/dev/infiniband` + `--ulimit memlock=-1` are both required** for RoCE. Without the verbs device
   nodes NCCL logs `Failed to initialize any NET plugin` and then
   `RuntimeError: NCCL error: invalid usage`.
4. **Check your HCA names against the actual hardware** (`ls /sys/class/infiniband/`). A stale
   `NCCL_IB_HCA` entry naming a non-existent device reproduces the same NET-plugin failure.
5. **Pick the RoCE **v2** GID, not v1.** Two GIDs can map the same IPv4 address:
   `gid[2] type=IB/RoCE v1 …ffff:c0a8:0008` vs `gid[3] type=RoCE v2 …ffff:c0a8:0008`. Match on
   `gid_attrs/types/N == "RoCE v2"` **and** your subnet — never on the address pattern alone.
6. **`-cc.splitting_ops` is not optional.** The PLE lookup is a CPU op + host-to-device copy; if it is
   captured into a CUDA graph you get
   `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU
   tensor is pinned`. The full op list from the original recipe must be carried over.
7. **Omit flags rather than zeroing them.**
   * `MTP=0` passed as `num_speculative_tokens: 0` fails pydantic (`greater_than`).
   * an empty JSON `--hf-overrides '{}'` fails as `Value {}} cannot be converted to <function loads>`.
   * a possibly-empty shell variable passed as `"$FLAG"` becomes a real **empty argument** →
     `vllm: error: unrecognized arguments:` (use an array + `${ARR[@]+"${ARR[@]}"}`).
8. **Bash `${VAR:-{json}}` mangles the value when VAR is already set** — braces inside the default
   word confuse delimiter matching and a stray `}` is appended (`'{}'` → `'{}}'`). Use a brace-free
   idiom.
9. **The draft/K10 directory must exist on the worker.** It is created head-only; at TP the worker
   opens it and dies with `Invalid repository ID or local directory specified`. It is small (348 KB,
   relative symlinks) — rsync it.
10. **Ship the image once** (`docker save | ssh docker load`), never pull per node. Also note
    `docker load` leaves `RepoDigests` empty — verify by **image ID**, not digest.

---

## 8. If you are building a new MTP / DFlash variant — read this first

The two walls above are **properties of the quantisation + TP sharding**, not of MTP specifically:

* **Any blockwise-fp8 tensor whose sharded dimension is not a multiple of 128 will fail at TP>1.**
  For NC=1-side layers like the shared expert (640) that means it must be requantised to bf16 (or
  padded to a multiple of `128 × TP`). Please consider shipping a **TP-safe variant** — it could be
  as simple as dequantising the shared expert as done here.
* **If your new drafter keeps expert matrices shardable-per-expert, `--enable-expert-parallel` is the
  lever** that avoids the Marlin→WNA16 naming trap. If the drafter's experts still fail the Marlin
  support check under EP, that is the next thing to look at.
* **DLFlash/DFlash-style drafters are worth trying for a different reason than quality**: a separate
  drafter with its own KV cache avoids the prefix-caching trap that self-speculation (MTP) creates on
  hybrid GDN targets (all KV groups get flagged as draft groups → reuse silently drops to zero; see
  vllm#53670). Our TP2 numbers here still have MTP, and prefix caching *does* work on this stack/VL
  build — but it is a known family of failure that a DFlash-style drafter sidesteps by construction.

### What we would test next, in order
1. **KV pool to spec**: at only ~55 of 121 GiB per rank, `--kv-cache-memory-bytes 60g` should give
   roughly **4M tokens** (we measured 1.25M at 20g, so ~62.5K tokens/GiB).
2. **512K context** via a YaRN overlay (the checkpoint is native 262,144; the overlay sets
   `max_position_embeddings`, `rope_type: yarn`, `factor: 2.0`,
   `original_max_position_embeddings: 262144` — nested under `text_config`).
3. **Single-stream speed**: the fact that TP2 loses to TP1 at c1 despite identical MTP suggests the
   all-reduce path is the cost. Worth A/B-ing tensor-parallel sizes / fusion settings before
   concluding, and worth re-measuring with a **real agentic workload** rather than synthetic prose
   (see the acceptance caveat in §6).
4. **Quality gate**: full tool-eval hardmode on the final configuration. Our single-Spark run scored
   **88/100** (154/176) on the same checkpoint, so that is the number to hold or beat.

---

## 9. Honesty notes

* Throughput figures come from llama-benchy, which **randomises its corpus** on purpose so prefill is
  not inflated by cache hits. They are cold-prefill numbers by design, and its prose corpus is the
  worst case for speculative decoding. Report real-workload rates (vLLM's own
  `Avg generation throughput` / `Accepted throughput`) alongside them.
* A single run cannot resolve small differences when the model's own spread is several points — use
  `--trials N` and quote a median, not one number.
* We initially mis-read concurrent aggregate throughput as single-stream; the c1 column is the only
  single-stream number. Do not repeat our mistake.

— Canglong & Primo, 2026-09-11. Tested on 2× DGX Spark (GB10), RoCE fabric.

---

## 11. Does MTP need EP? No — but the MoE kernels do (measured 2026-09-11)

We asked whether MTP could work at TP2 **without** `--enable-expert-parallel`, on the theory
that the Marlin shape check only disqualified the MTP layer. Half right, and the useful half
was wrong.

### The shape check disqualifies EVERY routed-expert layer

At TP2 without EP, all **48** MoE layers fail the check — not just MTP:

```
WARNING Layer 'language_model.model.layers.0.mlp.experts'  is not supported by GPTQMoeMarlin → WNA16
...
WARNING Layer 'language_model.model.layers.47.mlp.experts' is not supported by GPTQMoeMarlin → WNA16
```

Because each expert's `moe_intermediate_size` 640 is split to **320**, and
`320 % max(64, group_size=128) = 64 ≠ 0`. It is the *consequence* that differs:

| | routed experts (0–47) | MTP experts |
|---|---|---|
| checkpoint dtype | int4 GPTQ (`I32`/`F16`) | **bf16** |
| under WNA16 | loads, silently slower | **crashes** — no `w2_weight` |

So EP was never "needed for MTP". **EP is what keeps Marlin eligible across the entire MoE
stack**; the MTP layer was merely the one that erupted instead of degrading quietly.

### The cost of losing Marlin (pp2048/tg1024, c1)

| arm | d0 | d2048 | d8192 |
|---|---|---|---|
| **EP + Marlin** | **45.3** | **43.4** | **44.3** |
| no-EP + WNA16 | 17.7 | 21.3 | 21.8 |
| penalty | 2.6x | 2.0x | 2.0x |

At concurrency: no-EP c4 = 33.5–36.4 t/s vs **90.8** with EP at c4/d2048. **Prefill is
unaffected** (859–1,243 vs 1,001–1,198) — this is purely the MoE kernel.

### Two side results

1. **MTP-without-EP is achievable** with a 3-line reorder in `auto_gptq.py`: the `RoutedExperts`
   branch runs the shape check and returns the WNA16 fallback *before* `get_moe_quant_method()`,
   the only code that honours `-:<regex>` dynamic exclusions. So an explicit "do not quantize this"
   rule is shadowed by an automatic fallback — and our config already carried `-:mtp\..*`.
   Fix + verified applier: `tools/patch-gptq-moe-exclusion.py`, mounted via `QPATCH=1`.
   MTP then loads with **0** `w2_weight` errors and no MTP-layer warning.
2. **EP also costs nothing in cache**: KV 3,505,062 tokens with EP vs 3,591,798 without — a 2.5%
   difference, in favour of no-EP but far too small to offset a 2x decode penalty.

### Verdict

**`--enable-expert-parallel` is not optional.** It is load-bearing for MoE kernel selection,
independent of MTP — and that reason alone settles it. The patch is a diagnostic and an upstream
bug report, not a production change.

Upstream relevance: the ordering bug affects any GPTQ MoE model whose per-rank expert intermediate
is not group-aligned — a much wider class than this checkpoint.

---

## 12. The PLE table: it IS a budget question, not a wall (2026-09-11)

**Question:** can we disable the SSD offload and keep the PLE table in RAM?

**Correction first.** An earlier version of this section claimed full RAM residency was
impossible. That was wrong, and it came from anchoring on `GPU_MEM_UTIL=0.83` and reasoning
outward. The table is 45.6 GiB; the question is only whether the GPU is allowed to take so
much of the 121 GB unified pool that nothing is left for it. It is a **GMU budget**, and the
numbers land where you'd want them:

| GMU | KV tokens | table cached | host free |
|---|---|---|---|
| 0.85 (our current default) | 3,505,062 | 22% | 10.3 GiB |
| 0.75 | 2,779,175 | 49% | 22.4 GiB |
| 0.65 | 2,053,289 | 76% | 34.6 GiB |
| 0.58 | 1,545,168 | 94% | 43.1 GiB |
| **0.55** | **1,327,402** | **100%** | **46.8 GiB** |

(Derived from measured values: 36.99 GiB weights/rank, 44.7 GiB non-KV GPU footprint, and
59,650 KV tokens per GiB. Table = 49 GB.)

**So at GMU 0.55 the whole table is RAM-resident and you still keep 1.33M KV tokens** — which
covers a 512K or even 750K working context with concurrency to spare. This is not a new
capability either: the earlier FP8 lane on this hardware ran the n-gram table fully in RAM
(the trade then was "too little cache").

**Mechanism.** The mmap stays; it simply stops faulting to NVMe. `np.memmap` pages that are
resident cost nothing extra — a tmpfs copy would consume the same unified memory with no
benefit, so don't bother copying. Two caveats worth engineering around:

1. **Warm it after the checkpoint stream, not before.** Loading a 127 GB checkpoint streams
   through the page cache and will evict table pages. `VLLM_PLE_MMAP_PREWARM=1` runs at engine
   start; if boot order puts the weight load after it, re-touch the table afterwards. The
   upstream disk-offload PR notes the same trap ("UNCAPPED ... checkpoint stream evicts table
   cache; TTFT 1807-2853ms") and answers it by capping container memory, which bounds the
   page cache the container may hold.
2. **Consider locking it.** The container already runs with `--ulimit memlock=-1` (required for
   RDMA), so `mlock`-based pinning of the table directory is available if the page cache proves
   unstable under other workloads.

**Verify residency with:** `grep Cached /proc/meminfo` (expect ~45+ GiB) and the engine's own
telemetry — `PLE mmap stats` `gather ms/op` should fall from the 31–280 ms/op we measure when
part of the table is on NVMe.

**`VLLM_PLE_MMAP_PREFETCH` (default 0)** remains a separate, complementary lever: it hashes
n-grams at batch-assembly time so the gather overlaps decode instead of blocking it. With the
table resident it matters much less; with a partially cached table it is the cheapest win.

