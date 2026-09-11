# Forum post draft — for review before posting

**Suggested title:** *A5B INT4-AutoRound Qwen3.8-Flash-Next on 2× DGX Spark (TP=2): the requant it needs, and the flag that unlocks MTP*

**Suggested tags/location:** DGX Spark / GB10 — reply to azampatti's "Up to 70tok/s Qwen3.8-Flash-Next-Int4-AutoRound" thread, and/or a standalone post.

---

azampatti's A5B INT4-AutoRound build is a beautiful piece of work — it fits a model that has no right
to fit on one GB10, and it does it with real headroom. Since he doesn't have a second Spark, we tried
running it across two with `--tensor-parallel-size 2` on our pair. It does **not** work out of the box,
and both failures are interesting in their own right. Everything below is measured, not theorised; full
write-up, launchers and tools: **https://github.com/0rand/qwen38-flash-next-a5b-2x-dgx-spark**

## Blocker 1 — the checkpoint can't be TP-sharded as-is (needs a requant)

The hybrid quant stores 300 layers as **blockwise-fp8 (128×128)** with a `weight_scale_inv` sibling.
vLLM's fp8 block validator requires `(dim / TP) % 128 == 0`. The **shared expert** has a **640**
dimension:

```
Qwen3_8FlashNextSparseMoeBlock.shared_expert (Qwen3NextMLP).gate_up_proj
  -> fp8.py validate_fp8_block_shape()
  ValueError: Weight output_partition_size = 320 is not divisible by weight quantization block_n = 128
```

At TP2 that dimension halves to 320, and 128 does not divide 320. Since `640 / TP` must be a multiple
of 128, **only TP=1 can ever work** while those weights stay fp8 — no flag or config fixes it.

**The fix** is a small, targeted requant. vLLM decides a layer is fp8 with this rule:

```python
fp8_layers = {name[:-7] for name, info in metadata.items()
              if name.endswith(".weight") and info["dtype"] == "F8_E4M3"
              and name[:-7] + ".weight_scale_inv" in metadata}
```

so writing the weight as **bf16** and **dropping the scale sibling** moves it onto the unquantized
path — which shards freely. A pre-flight scan showed **exactly 144 tensors** needed it (all
`shared_expert`, all in one file); the other 156 fp8 weights were already TP-safe. Cost: fp8 236 MB →
bf16 472 MB. The original checkpoint is untouched (new dir = symlinks + one bf16 file + patched index).
Script: `tools/dequant-shared-expert.py`.

If a **TP-safe variant** could be published (or the shared expert simply shipped in bf16), this whole
step would disappear for anyone with two Sparks.

## Blocker 2 — MTP then fails, and the answer is EP, not config

With the requant done the model loads, but MTP dies at load:

```
WARNING Layer 'mtp.layers.48.mlp.experts' is not supported by GPTQMoeMarlin.
        Falling back to Moe WNA16 kernels.
ERROR   AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight'
        for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight'
```

At TP2 the expert matrices are split, so the Marlin support check fails and vLLM falls back to
`MoeWNA16`, whose parameter naming doesn't match what the MTP loader expects. (Note the fallback is
decided on **layer shape**, not weight dtype — the MTP experts are actually bf16 in the checkpoint.)

**What does not work:** adding `-:mtp\..*` (or relying on the existing `-:.*layers\.48\..*`) to
`quantization_config.dynamic`. We tested it: the warning persists, because
`AutoGPTQConfig.get_quant_method()` decides per `RoutedExperts` layer via the Marlin check and never
consults `dynamic`. We also tried `VLLM_FP8_HYBRID=0`, which breaks the loading path
(`AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'`) — the fp8-hybrid patch
and the loader are interdependent.

**What works — one flag:**

```
--enable-expert-parallel
```

EP distributes *experts* across ranks instead of splitting each expert's matrices, so the 640
dimension stays whole and the Marlin path is used with consistent naming. That single flag took us
from "Engine core initialization failed" to serving.

## Where it lands

Both ranks load (**34.98–36.99 GiB/rank**), all functional gates pass, and **prefix caching works**
(`prefix_cache_hits_total` 0 → 16,800 across two identical sends). MTP acceptance is **61–64%** on
agentic traffic. KV pool at `--kv-cache-memory-bytes 20g` is **1,251,206 tokens**, with only ~55 GiB
used per rank — so there is a lot of room left.

The honest part: **at TP=2 single-stream is ~25–30% slower than single-Spark** (38.2 vs 48.9 tok/s,
llama-benchy pp2048/tg1024, c1/d0). A cross-node all-reduce per layer costs more than the halved
weight traffic saves, and MTP multiplies it because it runs several forward passes per verification
step. TP2's real wins are **aggregate throughput under concurrency** (90.8 tok/s at c4/d2048 vs 78.2
on one Spark) and the **~2× KV pool** — the latter being what makes very long contexts practical.

## Gotchas worth knowing (each cost us a boot)

`--network host` is mandatory for multi-node (vLLM's message queue binds `VLLM_HOST_IP`; in the
default bridge namespace you get `ZMQError: Cannot assign requested address`) · that address must be
**each node's own** · `/dev/infiniband` **and** `--ulimit memlock=-1` are both required, otherwise
NCCL logs `Failed to initialize any NET plugin` then `NCCL error: invalid usage` · match the RoCE **v2**
GID via `gid_attrs/types/N == "RoCE v2"` (both v1 and v2 can map the same IPv4 address) ·
`-cc.splitting_ops` is not optional — the PLE lookup is a CPU op + H2D copy and CUDA-graph capture
fails without it · the draft/K10 directory must exist on **both** nodes · omit flags rather than
zeroing them (`MTP=0` → pydantic `greater_than`; `--hf-overrides '{}'` → parse error) · and beware
`${VAR:-{json}}` in bash, which silently appends a stray `}` when the variable is already set.

## Next

Raising the KV pool (the memory is there), a 1M-token context via YaRN, and a full quality run. If
azampatti's next iteration moves MTP → DFlash, that's interesting for a second reason: a separate
drafter with its own KV cache sidesteps the prefix-caching trap that self-speculation creates on
hybrid GDN targets (all KV groups get flagged as draft groups and reuse silently drops to zero).

Thanks for the checkpoint — it's a genuinely clever piece of engineering, and it scales further than
one Spark.

---

## FOLLOW-UP (added after measurement): can you skip `--enable-expert-parallel`?

We tested it, because "EP is required" is a claim worth attacking. Short answer: **MTP can be made
to load without EP, but you should still use EP**, and the reason is not MTP at all.

**The shape check fails for every routed-expert layer, not just MTP.** At TP2 each expert's
`moe_intermediate_size` 640 is split to 320, and `320 % group_size(128) = 64 ≠ 0`, so all **48** MoE
layers are ineligible for `GPTQMoeMarlin` and fall back to WNA16. What differs is the consequence:

| | routed experts (0–47) | MTP experts |
|---|---|---|
| checkpoint dtype | int4 GPTQ | **bf16** |
| under WNA16 | loads, silently slower | **crashes**: `no parameter 'w2_weight'` |

So EP was never "needed for MTP" — **EP is what preserves Marlin across the whole MoE stack**, and
the MTP layer was simply the one that erupted instead of degrading quietly.

Measured cost (pp2048/tg1024, c1, 1M ctx, GMU 0.85):

| arm | d0 | d2048 | d8192 |
|---|---|---|---|
| EP + Marlin | **45.3** | **43.4** | **44.3** |
| no-EP + WNA16 | 17.7 | 21.3 | 21.8 |

~2x decode; c4 33–36 t/s vs **90.8** with EP. Prefill unaffected — it is purely the MoE kernel.
KV cache was 3,591,798 tokens without EP vs 3,505,062 with (2.5%, not worth 2x decode).

**If you still want MTP without EP**, it is a 3-line reorder in `auto_gptq.py`: the `RoutedExperts`
branch returns the WNA16 fallback before `get_moe_quant_method()` is ever called — and that function
is the only place `-:<regex>` exclusions in `dynamic` are honoured. An explicit "do not quantize this
module" is therefore shadowed by an automatic fallback. Our config already carried `-:mtp\..*`; it
was simply never consulted. With the reorder, MTP loads with zero `w2_weight` errors. Fix and a
verified, idempotent applier are in the repo (`tools/patch-gptq-moe-exclusion.py`).

Worth reporting upstream: any GPTQ MoE model whose per-rank expert intermediate is not group-aligned
hits this, so the exclusion can silently do nothing in cases well beyond this checkpoint.

### And one on the PLE table

If you are memory-constrained: at GMU 0.83 the PLE table **cannot** be page-cache resident. On a
unified-memory part it shares the same pool as weights and KV — we measured 16.1 GB of cache against
a 49 GB table, so most gathers come off NVMe. The engine's own stats show the cost (`gather`
31–280 ms/op, serialised into decode). Note `VLLM_PLE_MMAP_PREFETCH` defaults to **0** — the
prefetch pipeline that hashes n-grams at batch-assembly time and overlaps the gather is simply not
enabled, and our windows report `prefetch hit 0 miss 0`. That looks like the cheapest available win.
