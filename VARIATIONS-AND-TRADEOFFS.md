# Variations and trade-offs

Every number below was measured on **2× DGX Spark (GB10, 121.7 GiB unified RAM each, RoCE fabric)**
with the azampatti A5B INT4-AutoRound checkpoint. Nothing here is extrapolated from a datasheet.

If you only read one section, read the **decision table** and **§2 (EP)**.

---

## 0. Decision table

| I want… | Profile | Rope | KV | Set-up cost |
|---|---|---|---|---|
| **Best quality** (codebase, research, agentic) | `a5b-tp2-ple-pin-nativerope.env` | native (×1.0) | pin 16g ≈ 1.04M tok | none |
| **General reach past native** | `a5b-tp2-ple-pin-384k.env` | YaRN ×1.5 | pin 16g ≈ 1.04M tok | −1 quality pt |
| **Long context (512K+)** | `a5b-tp2-ple-pin.env` | YaRN ×2.0 | pin 20g ≈ 1.33M tok | −3 quality pts |
| GMU-driven KV (no pin) | `a5b-tp2-512k-ep.env` | native | GMU 0.83 | see §4 caveat |
| Compare GMU-lowering vs pinning | `a5b-tp2-ple-ram.env` | native | GMU 0.55 | superseded by the pin |

```
./ctrl.sh start config/a5b-tp2-ple-pin-384k.env
./ctrl.sh status      # containers, served ctx, KV pool, prefix cache, acceptance, PLE telemetry
./ctrl.sh gates       # acceptance harness
./ctrl.sh stop
```

---

## 1. The one rule that matters most

**`--enable-expert-parallel` is not optional.** Measured, same checkpoint, same everything else:

| arm | d0 | d2048 | d8192 | c4 @d2048 |
|---|---|---|---|---|
| **EP + Marlin** | **45.3** | **43.4** | **44.3** | **90.8** |
| no-EP + WNA16 | 17.7 | 21.3 | 21.8 | 36.2 |

Roughly **2× decode**. Why: at TP2 every expert's `moe_intermediate_size` 640 is split to **320**,
and `320 % group_size(128) ≠ 0`, so *all 48 routed-expert layers* fail the `GPTQMoeMarlin` shape
check and fall back to WNA16. EP distributes whole experts across ranks, so the dimension stays 640
and Marlin stays eligible. Prefill is unaffected — it is purely the MoE kernel.

MTP is a *separate* story (it needs EP too, but for a different reason — see `TP2-FINDINGS.md` §4).
**EP is required for the MoE kernels regardless of MTP.**

---

## 2. The requant (why this repo has a weight-preparation step)

300 layers are blockwise-fp8 (128×128). The shared expert's **640** dimension cannot be split:
`(640 / TP) % 128 == 0` only holds for **TP=1**. So the checkpoint as shipped is not TP-shardable.

`tools/dequant-shared-expert.py` dequantises **exactly 144 tensors** (all `shared_expert`, all in one
file) to bf16 and drops their `weight_scale_inv` siblings, which moves them onto the unquantized path.
Cost: 236 MB → 472 MB. The original checkpoint is untouched — the result is a symlink farm plus one
bf16 file.

**Measured quality cost: none.** TP2-native scores **87/100** against TP1-native's **88** (the 1-point
gap is parallelism, not the requant). So if you have two Sparks, the requant is free.

---

## 3. Context length vs quality — the YaRN ladder

All runs: 88 scenarios / 176 points, temp 0, seed 42, parallel 4, TP2+MTP3+EP. **Only the rope
config varies.**

| factor | ctx | score | Hard Mode | **Structured Reasoning** | Multi-Step |
|---|---|---|---|---|---|
| **1.0 native** | 262,144 | **87** | 87% | **100%** | 88% |
| **1.5** | 393,216 | **86** | 82% | 83% | 75% |
| **2.0** | 524,288 | **84** | 79% | 67% | 75% |
| 4.0 | 1,048,576 | unmeasured | — | — | — |

Three things to take away:

1. **The cost is mild and roughly linear in the factor** — 1 point to ×1.5, 2 more to ×2.0.
2. **The damage lands on reasoning, not tool use.** Tool Selection, Parameter Precision, Error
   Recovery, Localization, Toolset Scale and Creative Composition stay at **100%** on every rung.
   Structured Reasoning goes 100 → 83 → 67. So if your work is *retrieval-shaped*, scaling is nearly
   free; if it is *reasoning-shaped* (codebase work, research), it is expensive.
3. **The model's native window is 262,144.** Above it you are off-distribution without scaling, and
   paying reasoning quality with it. Model the trade explicitly rather than leaving `max_model_len`
   at something large "just in case".

> **Gotcha we hit:** serving `max_model_len=524288` with *native* rope looks harmless (prompts under
> 262K are fine) but leaves positions 262K–524K undefined. Pair native rope with a native ceiling.

---

## 4. KV sizing: GMU vs pin

**GMU is a fill-to-budget target; `--kv-cache-memory-bytes` is an exact allocation.**

| | GMU | KV pin |
|---|---|---|
| behaviour | vLLM profiles, then *fills* the budget with KV | allocates exactly what you ask |
| measured | GMU 0.85 → 58.76 GiB → 3,505,062 tokens | 16g → 1,041,212 tokens |
| changing it | moves the GPU cap **and** the KV pool together | moves KV only |

**Reach for the pin when you need to reserve memory for a non-vLLM consumer** — here the 48.67 GiB
PLE table. One chosen variable beats two inferred ones.

> **In pin mode GMU must go tiny (`0.01`),** or vLLM fills the GMU budget with KV and the pin becomes
> decorative. The two levers fight. `ctrl.sh` profiles set this for you.

Measured density (pin mode): **~62,600–66,300 tokens/GiB**.

| pin | KV tokens | conc @262K | conc @384K |
|---|---|---|---|
| 12g | 751,007 | 2.86× | 1.91× |
| **16g** | **1,041,212** | 3.97× | 2.65× |
| 20g | 1,326,493 | 5.06× | 3.37× |

**Lowering `max_model_len` does not free KV** — it only changes concurrency for a fixed pool. The
pool is the pin.

---

## 5. The PLE table: what "offload" actually means

The 48.67 GiB n-gram table is served by **file-backed mmap** (`np.memmap` + `MADV_RANDOM`) from
NVMe. **That mmap *is* the offload path** — there is no separate switch to turn off.

On GB10 the GPU and CPU **share one memory pool**, so the table competes directly with weights and
KV. Whether it is resident is a budget question:

* At GMU 0.85 the table got **22%** cached; gathers ran ~124 µs/row (NVMe).
* With a pin (GMU 0.01 + KV_BYTES=16g) the host keeps ~45 GiB and gathers run **3.36 µs/row ≈ 47 GB/s
  — memory bandwidth**, i.e. genuinely resident.

**Residency percentage is the wrong health metric.** mmap is demand-paged: un-touched pages are the
cold tail your traffic never asked for, not pages that were evicted. The metric that matters is
**per-row gather latency**:

| path | per-row | implies |
|---|---|---|
| prefill, resident | 3.36 µs | ~47 GB/s (RAM) |
| prefill, SSD-era | ~124 µs | ~1.3 GB/s (NVMe) |
| decode, resident | 22–43 µs | latency-bound on small gathers |
| decode, SSD-era | 17–35 ms/op | serialised into decode |

**Why it is mostly a prefill story:** prefill gathers one PLE row for **every prompt token**
(3,646 rows in a single op, measured), while decode touches only tens-to-hundreds per step. Making
the table resident took prefill from ~1,100 to **~3,200–3,670 t/s** (≈3×, depth-invariant) and left
decode roughly unchanged. Since long-context latency is dominated by prefill, this is where the win
lives.

**Measuring it (do not infer from `Cached`):**
```
docker exec <container> bash -c 'for d in /proc/[0-9]*; do p=${d#/proc/}; \
  if grep -q ple-table $d/smaps 2>/dev/null; then \
  awk "/ple-table/{n=1} /^Rss:/{if(n)r+=\$2} /^\$/{n=0} END{print r/1048576}" $d/smaps; fi; done'
```
Then read the engine's own telemetry: `docker logs … | grep 'PLE mmap stats'`.

**Two things that don't work / do:**
* `cat`-warming: 37.9 s of reads bought **0.2 GiB** — sequential reads create inactive pages.
* `drop_caches` *after* the checkpoint load: releases the ~9 GiB of dead shard pages that were read
  after prewarm and outrank the table in the LRU (measured 73% → 77%). `ctrl.sh start` does this.

---

## 6. PLE prefetch (`VLLM_PLE_MMAP_PREFETCH`)

Defaults to **0** in the image, documented upstream as experimental ("run the n-gram hash at batch
assembly time"). It exists to overlap the gather with decode instead of blocking it — aimed at the
22–43 µs/row decode gathers, which are latency-bound rather than I/O-bound once the table is resident.

**Measured verdict on this hardware: leave it OFF (the default).**

Same profile, one variable, single runs each (pp2048/tg1024, c1):

| | PREFETCH OFF | PREFETCH ON |
|---|---|---|
| d0 / d2048 / d8192 (tg t/s) | 29.0 / 35.7 / 36.6 | 26.9 / 28.5 / 32.6 |
| per-op PLE overhead | 5.1-9.3 ms | 5.7-11.8 ms |

It engages and works — `backend PrefetchingMmapTable`, thousands of hits per window
(1991 -> 4250) against 22 misses, and `rows/op` drops to 0.0 because the synchronous gather path
stops being used. **But it buys nothing when the table is resident**, because there is no latency
left to hide, and it costs ~7-20% on c1 decode across every depth. c2/c4 improve slightly (up to
+13% at c4/d8192); c1 is what interactive work cares about.

**Enable it only if gathers are actually faulting to NVMe** — tight host memory, GMU held high, or a
larger table. Diagnose from the `PLE mmap stats` line: tens-to-hundreds of ms/op with high per-row
cost means NVMe is in the path and prefetch is the right tool. RAM-speed gathers mean it is pure
overhead. Full data: `RUNBOOK-mtp-no-ep.md` §15.

---

## 7. Speculative decoding (MTP)

`MTP=3` with a k=10 draft. Acceptance is **workload-dependent and should never be read in isolation**:

| content | per-position acceptance | avg |
|---|---|---|
| free prose | 0.579 / 0.281 / 0.174 | 34% |
| structured / code-like | 0.912 / 0.794 / 0.755 | 82% |

Position 1 acceptance is high and decays with depth, as expected. Upstream's own recipe notes MTP
measured *worse* than no-MTP on H100 at ~36% acceptance — consistent with the prose row above. If
your traffic is prose-heavy, test MTP off before assuming it helps.

---

## 8. Verification cookbook

| claim | how to check |
|---|---|
| EP is active (Marlin, not WNA16) | `docker logs … \| grep -c "not supported by GPTQMoeMarlin"` → **0** |
| rope override applied | `docker logs … \| grep -c hf_overrides` → **0** means native |
| served context | `curl -s localhost:8100/v1/models \| grep max_model_len` |
| KV pool | `docker logs … \| grep 'GPU KV cache size'` |
| prefix caching works | `curl -s localhost:8100/metrics \| grep prefix_cache_hits_total` — must increase across two identical sends (`gate.sh` does this) |
| PLE residency | smaps one-liner in §5 |
| PLE gather health | `docker logs … \| grep 'PLE mmap stats'` — watch µs/row, not % |
| full acceptance | `./ctrl.sh gates` |

**Always check the point denominator before comparing benchmark scores.** A tool-eval run without
`--hardmode` scores **69 scenarios / 138 points** and looks artificially strong; comparable runs are
**88 scenarios / 176 points**. We nearly published a 92 that way — it was really a 138-point subset
against a 176-point one.

---

## 9. Throughput reference (pp2048/tg1024, llama-benchy)

**384K profile, pin 16g:**

| depth | c1 tg | c2 | c4 | c1 pp | c1 TTFT |
|---|---|---|---|---|---|
| d0 | 39.2 | 64.0 | 100.7 | 3,560 | 767 ms |
| d2048 | 41.3 | 74.5 | **116.7** | 3,158 | 1,491 ms |
| d8192 | 38.4 | 62.6 | 94.0 | 3,207 | 3,388 ms |

**Versus the GMU 0.85 / 1M configuration:** prefill 1,001–1,198 t/s and 8,833 ms TTFT at d8192 —
i.e. **~3× better prefill, ~2.6× better TTFT**, at the cost of ~10% on c1 decode.

Long context holds up: at **d500000**, decode was **29.8 t/s** — the same as at d128000, confirming
depth-invariant decode for this architecture. TTFT scales ~1 s per 1,000 tokens of context.

---

## 10. Footnotes

* MTP + EP vs no-EP, the requant, and the PLE analysis in full: `TP2-FINDINGS.md`
* The experiment that produced the EP verdict, with its three-arm design: `RUNBOOK-mtp-no-ep.md`
* Everything here was measured with `tool-eval-bench` (88 scenarios, temp 0, seed 42, parallel 4) and
  `llama-benchy` via the same harness.
* Two of my own measurement errors are documented rather than hidden: the 69-scenario score above,
  and an early claim that the PLE table "cannot be RAM-resident" that was true only at GMU 0.85.
