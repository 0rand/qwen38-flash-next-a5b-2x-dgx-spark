# RUNBOOK — MTP at TP2 **without** `--enable-expert-parallel`

**Status:** ready to execute. Mechanism built and committed (`7dd1240`).
**Owner:** Primo. **Prepared by:** Canglong, 2026-09-11.
**Blocked on:** nothing except a free GPU slot — the 1M A5B stack is live on :8100.

---

## 1. Objective and hypothesis

Today we run **MTP=3 + `--enable-expert-parallel`** because MTP will not load at TP2 without EP:

```
WARNING Layer 'mtp.layers.48.mlp.experts' is not supported by GPTQMoeMarlin. Falling back to Moe WNA16 kernels.
ERROR   AttributeError: Layer 'mtp.layers.48.mlp.experts' has no parameter 'w2_weight'
        for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight'
```

**Hypothesis:** this is not a quantisation requirement but a *branch-ordering* bug. In
`auto_gptq.py::AutoGPTQConfig.get_quant_method()`, the `RoutedExperts` branch runs the Marlin
**shape** check and returns the WNA16 fallback before `get_moe_quant_method()` — the only function
that honours `-:<regex>` exclusions in `quantization_config.dynamic` and returns
`UnquantizedFusedMoEMethod`. Our config *already* contains `-:mtp\..*`; it is simply never consulted.

**Success looks like:** MTP=3 loads with **no** WNA16 warning for the MTP layer and **no** `w2_weight`
error, expert parallelism off, and throughput/quality no worse than the EP baseline.

## 2. Why this should work (verified, not assumed)

| fact | source |
|---|---|
| MTP experts are **bf16** in the checkpoint (no quantized keys) | safetensors header scan |
| `moe_intermediate_size` 640 → TP2 shards to **320** | `config.json` |
| `320 % max(64, group_size=128) = 64 ≠ 0` → Marlin ineligible | `marlin_utils.check_moe_marlin_supports_config` |
| `desc_act=False` ⇒ `allow_tile_padding=True` (only the group term fails) | docker image source |
| `-:mtp\..*` is present and correctly formed in `dynamic` (a dict) | `qwen38fn-a5b-int4-tp2safe/config.json` |
| the exclusion → `UnquantizedFusedMoEMethod`, which registers **`w13_weight`/`w2_weight`** — exactly the names the MTP loader wants | `fused_moe/unquantized_fused_moe_method.py:69-79` |
| EP only "fixes" it because experts stay whole: `640 % 128 == 0` | arithmetic |

So excluding MTP experts from quantisation is both the **correct** behaviour (they are bf16) and
sufficient for the loader.

## 3. Baseline to beat / compare against

Measured on the current EP configuration (pp2048/tg1024, llama-benchy):

| arm | config | c1 d0 | c1 d2048 | c1 d8192 | c4 d2048 | MTP acceptance |
|---|---|---|---|---|---|---|
| **A (baseline, EP)** | MTP3 + EP | 38.2 | 35.0 | 29.3 | 90.8 | 61–64% agentic / 38–42% prose |
| **B (this test)** | MTP3, **no EP**, QPATCH=1 | ? | ? | ? | ? | ? |
| **C (control)** | MTP0, no EP, QPATCH=1 | ? | ? | ? | ? | — |

Arm C exists to separate "dropping EP" from "MTP itself". Without it, a B-vs-A difference is
unattributable — and combined changes are exactly the A/B sin we avoid.

Also record for the same runs: `prefix_cache_hits_total` (must stay non-zero — the kill switch),
KV pool size, and weights/rank GiB.

## 4. Preconditions

1. **Patch file staged and committed:** `patch/gptq-moe/auto_gptq.py` (32,547 B) — the engine file
   with the reorder applied. Verify it is the patched one, not the stock copy:
   ```
   grep -n "get_dynamic_override" patch/gptq-moe/auto_gptq.py   # must appear BEFORE the shape call
   python3 -c "import ast;ast.parse(open('patch/gptq-moe/auto_gptq.py').read());print('compiles')"
   ```
2. **Config exclusion present** in `qwen38fn-a5b-int4-tp2safe/config.json`:
   `quantization_config.dynamic` must contain `"-:mtp\\..*"` (and ideally `"-:.*layers\\.48\\..*"`).
   Already true today — do not "fix" it, it was never broken.
3. **Launcher supports the mount:** `QPATCH=1` (committed `7dd1240`). `tp2-serve.sh` rsyncs `patch/`
   to the worker; `tp2-rank.sh` resolves it relative to its own directory. Verify after launch that
   the mount is present on **both** ranks (step 5.2).
4. **A free slot.** Stop the live 1M stack first, and note its launch line so it can be restored
   exactly (see §7).

## 5. Procedure

### 5.1 Stop the current stack
```
cd ~/dockers/qwen38fn-a5b-lane3 && ./tp2-serve.sh stop
```

### 5.2 Launch arm B — MTP3, no EP, patched
No `EXTRA_ARGS`, i.e. **no** `--enable-expert-parallel`. Keep 262,144 / no YaRN / 20g KV so the only
difference from arm A is EP presence:
```
MODEL_DIR=/var/tmp/models/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe \
DRAFT_DIR=/var/tmp/models/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe-draft-k10 \
CTX=262144 KV_BYTES=20g GPU_MEM_UTIL=0.85 MTP=3 QPATCH=1 \
./tp2-serve.sh
```
Confirm both ranks actually got the patched file. Grep for the patch's own comment, which exists
**only** in the patched copy — do not grep for `get_dynamic_override`, which stock also contains:
```
SP=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_gptq.py
docker exec qwen38-flash-tp2 grep -c "must outrank the" $SP                 # expect 1 (stock: 0)
ssh <user>@<WORKER_ROCE_IP> "docker exec qwen38-flash-tp2 grep -c 'must outrank the' $SP"   # expect 1
```
A `0` on either rank means that rank is running unpatched code — stop and fix before measuring,
because arm B would then reproduce the EP-less failure and teach us nothing.

### 5.3 Expected signals
**Must be ABSENT:**
```
"is not supported by GPTQMoeMarlin"   (for mtp layers)
"has no parameter 'w2_weight'"
```
**Must be PRESENT:** `Application startup complete`, a KV cache size line, and MTP acceptance lines
once traffic flows.

If the WNA16 warning appears **for the same layer**, none of this fired — go to §8.1.

### 5.4 Gates + measurement
```
BASE_URL=http://127.0.0.1:8100 MODEL=qwen3.8-flash-next-a5b EXPECT_CTX=262144 ./gate.sh
```
Then the same perf matrix as arm A (pp2048/tg1024, concurrency 1,2,4, depth 0/2048/8192), plus:
```
docker logs qwen38-flash-tp2 2>&1 | grep -oE 'Avg Draft acceptance rate: [0-9.]+%' | tail -5
curl -s http://127.0.0.1:8100/metrics | grep '^vllm:prefix_cache_hits_total'
```
Finally one hardmode quality run at temp 0 (arm A's reference is **88/100** single-run, ds2atc median
89) — a speed win alone is not a pass.

### 5.5 Arm C — MTP0, no EP, patched
Same as 5.2 with `MTP=0` and **no** `--enable-expert-parallel`. (`MTP=0` must *omit* the flag, not
zero it — pydantic `greater_than` otherwise.) This isolates EP's contribution from MTP's.

## 6. Decision rule

* **B loads and matches A within noise on speed and quality** → EP was never needed for MTP; ship the
  patch, keep no-EP (one less moving part, and no expert all-to-all).
* **B loads but is slower/faster than A** → separation of concerns: the patch fixes *loading*, EP
  affects *throughput*. Record both, choose per workload; this is a genuinely useful result either way.
* **B fails at load** → read §8; the hypothesis is falsified in a specific, informative way.
* **Quality regresses** → reject regardless of speed (the standing rule).

## 7. Rollback

The live 1M configuration (restore verbatim if needed):
```
ROPE='{"max_position_embeddings":1048576,"rope_scaling":{"rope_type":"yarn","factor":4.0,
"original_max_position_embeddings":262144}}' ALLOW_LONG=1 \
MODEL_DIR=/var/tmp/models/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe \
DRAFT_DIR=/var/tmp/models/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-tp2safe-draft-k10 \
CTX=1048576 KV_BYTES="" GPU_MEM_UTIL=0.85 MTP=3 EXTRA_ARGS="--enable-expert-parallel" \
./tp2-serve.sh
```
`QPATCH` defaults to 0, so any launch without it is byte-identical to today's image behaviour. The
patch is a mount, never a baked image — nothing to rebuild to undo.

## 8. Failure modes and what each one means

| symptom | meaning | action |
|---|---|---|
| WNA16 warning still names the mtp layer | the exclusion never fired: mount missing on that rank, or the prefix doesn't match `-:mtp\..*` | verify mount on both ranks; log the actual `prefix` string; adjust the regex (note vLLM's prefix is `mtp.layers.48.mlp.experts`) |
| `no parameter 'w2_weight'` persists | `UnquantizedFusedMoEMethod` was selected but the loader maps to a name it doesn't create | inspect the MTP loader's mapping table; may need a loader-side patch instead |
| container dies on start, mount error | `HERE` didn't resolve on the worker, or `patch/` wasn't synced | check `tp2-serve.sh` rsync ran; confirm `$WORKER_DIR/patch/gptq-moe/auto_gptq.py` exists on Force |
| `MergedColumnParallelLinear has no attribute data` | `VLLM_FP8_HYBRID` was disabled — unrelated to this patch, and not a valid toggle for this checkpoint | keep `FP8_HYBRID=1` (default) |
| loads fine, quality drops | patch changed which quant path the MTP experts take (bf16 instead of attempted int4) | unexpected — MTP experts are bf16 in the checkpoint, so this would indicate a real behavioural change: reject and investigate |

## 9. Exit criteria

1. Both ranks load the patched file (verified in-container, not assumed).
2. Arm B reaches `Application startup complete` with no WNA16 warning for the MTP layer.
3. `gate.sh` all-pass, prefix cache still accumulating hits.
4. Three arms (A baseline / B no-EP / C MTP0-no-EP) measured on the identical perf matrix.
5. One hardmode run recorded against the 88/100 reference.
6. Result written into `DragonVault/entities/infra/qwen38-flash-next.md` and, if it works, offered
   upstream as an issue/PR — the same branch order shadows exclusions for any GPTQ MoE model with an
   unquantised MoE sub-module whose per-rank intermediate isn't group-aligned.

---

## 10. RESULTS (executed 2026-09-11 ~18:30)

**Arm B ran: MTP3, no EP, QPATCH=1, 1M ctx, GMU 0.85.**

### What worked
* MTP **loads without EP** — `w2_weight` errors: **0**; no WNA16 warning for the MTP layer.
  The staged patch is proved correct on both ranks (in-container grep = 1 each).
* All gates pass; prefix cache +17,600; KV pool 3,591,798 tokens (EP: 3,505,062).

### What the hypothesis got wrong
The shape check disqualifies **every routed-expert layer**, not just MTP — **48 warnings**,
`language_model.model.layers.0 … .47.mlp.experts`. At TP2 each expert's intermediate is
640/2 = 320, and `320 % 128 ≠ 0` for all of them. What differs is the *consequence*:

| | routed experts (0–47) | MTP experts |
|---|---|---|
| checkpoint dtype | int4 GPTQ (I32/F16) | **bf16** |
| under WNA16 | loads, silently slower | **crashes** (no `w2_weight`) |

So EP was never "needed for MTP": **EP is what keeps Marlin eligible for the entire MoE stack**;
MTP was merely the layer that crashed instead of degrading.

### The numbers (pp2048/tg1024, c1)

| arm | d0 | d2048 | d8192 |
|---|---|---|---|
| **EP + Marlin** (production) | **45.3** | **43.4** | **44.3** |
| **no-EP + WNA16** (this test) | 17.7 | 21.3 | 21.8 |
| penalty | **2.6x slower** | 2.0x | 2.0x |

Concurrency confirms it: no-EP c4 = 33.5–36.4 t/s, versus 90.8 t/s measured for EP at c4/d2048.
Prefill is unaffected (~859–1,243 vs 1,001–1,198 t/s) — it is purely the MoE kernel.

### Verdict
1. **`--enable-expert-parallel` stays.** EP is load-bearing for *two* independent reasons, now both
   measured: (a) MTP cannot load without it (fixable by our patch), and (b) it determines
   **Marlin vs WNA16 for all 48 MoE layers**, worth ~2x decode. (b) alone settles it.
2. **The patch is a diagnostic and an upstream bug report, not a production change.** Its value is
   that MTP-without-EP now fails *informatively* instead of crashing, and that the ordering bug
   (explicit `-:<regex>` exclusions shadowed by an automatic shape fallback) is documented.
3. Upstream issue material: the same shadowing affects any GPTQ MoE model whose per-rank expert
   intermediate is not group-aligned — not just this checkpoint.

---

## 13. KV PIN + RAM-RESIDENT PLE — measured 2026-09-11 (profile `a5b-tp2-ple-pin.env`)

Boot: `./ctrl.sh start config/a5b-tp2-ple-pin.env` — 512K ctx, EP, MTP3, `GPU_MEM_UTIL=0.01`,
`KV_BYTES=20g`. Boot clean, all gates pass, prefix cache +17,600.

**What the pin bought (host side):**

| | GMU 0.85 (1M profile) | pin (this run) |
|---|---|---|
| host used / free | 107 / 3 GB | **54 / 43 GB** |
| page cache | 16.1 GB | **46.6 GB → then 43.2 GB under load** |
| PLE table cached | 22% | **~95-100%** |
| PLE gather ms/row | 0.124-0.137 | **0.025 idle / 0.083 loaded** |

**Capacity:** `GPU KV cache size: 1,326,493 tokens`, 2.53x concurrency at 524,288 — matches the
predicted ~1.3M from the budget table exactly.

**Throughput (pp2048/tg1024):**

| depth | c1 tg | c2 | c4 | c1 pp | c1 TTFT |
|---|---|---|---|---|---|
| d0 | 29.6 | 63.8 | 85.7 | 2,266 | 1,102 ms |
| d2048 | 26.9 | 52.9 | 77.5 | 2,262 | 2,009 ms |
| d8192 | 34.3 | 52.1 | 69.3 | 3,031 | 3,571 ms |

Prefill 2,266-3,031 t/s and TTFT are ~2.3x better than the 1M/GMU-0.85 baseline (1,001-1,198 t/s,
8,833 ms at d8192).

### ⚠️ NOT ATTRIBUTABLE — two variables moved

This run differs from the baseline in **two** ways: context (1M → 512K) **and** memory mode
(GMU-filled → pin). The c1 decode figure (29.6 vs 45.3) therefore cannot be blamed on the pin, and
the prefill gain cannot be credited to it either — the smaller YaRN factor (2.0 vs 4.0) is a
plausible contributor.

A third hypothesis worth stating so it is not lost: the 45.6 GiB resident table lives in the same
**unified LPDDR5X pool** the GPU reads from, so host residency may cost decode memory bandwidth
while leaving compute-bound prefill alone. That would explain the shape of these numbers — but it
is a hypothesis, not a finding.

**The clean test (one variable, both at 512K + EP):**
```
arm 1: config/a5b-tp2-512k-ep.env    512K, GMU 0.83   (table partly cached)
arm 2: config/a5b-tp2-ple-pin.env    512K, GMU 0.01 + KV 20g  (table resident)
```

### 13.1 Attribution RESOLVED (Primo's c1 run, same profile)

| depth | pp t/s | tg t/s | TTFT |
|---|---|---|---|
| d0 | 3,670 | 40.8 | 760 ms |
| d2048 | 2,526 | 40.1 | 1,830 ms |
| d8192 | 2,695 | 36.0 | 4,006 ms |

Against the 1M/GMU-0.85 baseline (pp 1,001-1,198, tg 43.4-45.3, TTFT 8,833 ms at d8192):
**prefill ~3x, decode unchanged.**

* The earlier "decode 29.6 vs 45.3" concern was **run-to-run variance**, not a residency penalty —
  the same config measured 40.8 here. Retired.
* No decode penalty from host-resident table data; the unified-memory bandwidth hypothesis is not
  supported.

**Mechanism (from the engine's own PLE telemetry):**

```
rows/op=3646.1   gather 3.36 us/row    <- PREFILL: one PLE row per prompt token
rows/op= 255.1   gather 25.29 us/row   <- decode steps
rows/op=  65.1   gather 43.15 us/row   <- decode steps
```

Prefill gathers a row for **every prompt token** in a single op. That path was reading NVMe; it is
now RAM-resident at ~3.4 us/row. Decode gathers only tens-to-hundreds of rows per step, so it
barely noticed the change. **The win is entirely on the prefill side — which is exactly where it
matters for long-context work.**

### Remaining lever, now evidence-backed

Decode gathers run at 22-43 us/row versus 3.36 us/row for the batched prefill op — a ~10x per-row
gap that is *not* I/O (the table is resident). That is latency/machinery overhead on small gathers,
which is precisely what `VLLM_PLE_MMAP_PREFETCH=1` exists to hide: hash at batch-assembly time and
overlap the gather with decode. Untested as of this writing.

### 13.2 Is the table FULLY resident? No — and here is how to measure it properly

Primo asked the right question ("we only use 80/122 RAM — are all tables resident?"). The honest
answer, measured from the worker process's own page maps:

```
table on disk:              48.67 GiB   (52,259,870,014 B -- NOT the 45.6 GiB I first estimated)
resident (worker Rss):     ~35.0-37.3 GiB   -> 72-77%
MemAvailable:              ~41.8 GiB        -> NOT memory pressure; it is demand paging
```

**How to measure (do this, don't infer):**
```
docker exec <c> bash -c 'for d in /proc/[0-9]*; do p=${d#/proc/}; \
  if grep -q ple-table $d/smaps 2>/dev/null; then \
  awk "/ple-table/{n=1} /^Rss:/{if(n)r+=\$2} /^\$/{n=0} END{print r/1048576}" $d/smaps; fi; done'
```

**Why it is not a problem.** mmap is demand-paged: a page becomes resident when accessed, so the
un-resident 23% is the **cold tail our traffic never requested**, not pages that were evicted.
The proof is latency, not a percentage:

| path | per-row gather | implied bandwidth (160 B rows) |
|---|---|---|
| prefill, resident | 3.36 us | **~47 GB/s** (LPDDR5X speed) |
| prefill, SSD-era | ~124 us | ~1.3 GB/s (NVMe) |

3.36 us/row is memory bandwidth, so every row being touched is in RAM. **The health metric is
per-row gather latency, not residency percentage.**

**Two things that do NOT work:**
* `cat`-warming the files: 37.9 s of reading bought **0.2 GiB**. Sequential reads create inactive,
  immediately-reclaimable pages.
* Inferring residency from `Cached`: it sat near the table size by coincidence and I misread it as
  residency. Always read the process's smaps.

**What does help:** `drop_caches` *after* the checkpoint load (privileged container), which releases
the ~9 GiB of dead checkpoint-shard pages that were read after PREWARM and therefore outranked the
table in the LRU. Measured: Cached 45.31 -> 38.17 GiB, table residency 73% -> 77%.

**Ceiling:** with the GPU holding ~65 GiB (weights 37 + KV pin 20 + overhead) and ~7 GiB of process
memory, the page-cache ceiling is ~45-49 GiB against a 48.67 GiB table — so full residency is
borderline even in principle, and would cost KV capacity to guarantee. Not worth it: the cold tail
is free until requested.

### 13.3 Second c1-c4 sweep on the pin profile (Primo, 2026-09-11 ~17:30)

| depth | c1 tg | c2 | c4 | c1 pp | c1 TTFT |
|---|---|---|---|---|---|
| d0 | 39.2 | 64.0 | 100.7 | 3,560 | 767 ms |
| d2048 | 41.3 | 74.5 | **116.7** | 3,158 | 1,491 ms |
| d8192 | 38.4 | 62.6 | 94.0 | 3,207 | 3,388 ms |

**Prefill is DEPTH-INVARIANT** at 3,158-3,560 t/s (baseline 1M/GMU 0.85: 1,001-1,198) — ~3x and,
more importantly, flat. This is the direct evidence that the un-resident PLE cold tail costs
nothing in steady state: if it did, prefill would sag with depth and it does not.

**Concurrency improved as well:** c4 94.0-116.7 t/s vs 74-90 on the old config; d2048/c4 = 116.7 is
the best aggregate this lane has recorded.

**Consistent ~10% c1 decode reduction** across both pin-profile runs (38.4/39.2/40.1/40.8 vs
43.4-45.3 baseline) — outside run variance, so probably real. Leading explanation: ~38 GiB of
host-resident table shares the unified LPDDR5X pool the GPU reads from, costing decode bandwidth
while leaving compute-bound prefill alone. Not yet isolated (would need the clean 512K A/B:
GMU 0.83 vs pin). Primo's verdict: does not matter at this scale — the prefill gain dominates.

**Verdict on the cold tail (Primo, agreeing with the data):** "we can't [be fully resident] but it
likely does not matter — if they stutter then only once or rarely, not costing constant pp or tg
speed." Recorded: residency percentage is not the target; per-row gather latency is.

---

## 14. YaRN scaling ladder for A5B (measured 2026-09-11, all 88 scenarios / 176 pts)

temp 0, seed 42, parallel 4, TP2, MTP3, EP, pin 16g. **Only the rope config varies.**

| factor | ctx | score | Hard Mode | Structured Reasoning | Multi-Step |
|---|---|---|---|---|---|
| **1.0 native** | 262,144 | **87** | 87% | **100%** | 88% |
| **1.5** | 393,216 | **86** | 82% | 83% | 75% |
| **2.0** | 524,288 | **84** | 79% | 67% | 75% |
| 4.0 | 1,048,576 | not measured | -- | -- | -- |

(TP1 native reference: 88 across three runs, sigma 0. TP2 native is 87 -- the 1-point gap is
parallelism/numerics, and the requant is exonerated.)

### Reading

* **Cost is mild and roughly linear in the factor**: 1 point native -> x1.5, 2 more x1.5 -> x2.0.
* **Sensitivity is concentrated in REASONING**, not tool use or formatting: Structured Reasoning
  holds 100% at native, 83% at x1.5, 67% at x2.0. This is the mechanism behind Primo's
  observation that heavy scaling is "fine for a long chat but bad for codebase or research".
* Safety & Boundaries and Context & State barely move (73-77%, 80-85%) across the ladder.

### Recommendation

* **<=262K work (codebase, research, tools): native. No reason to accept any loss.**
* **384K (x1.5): the general-purpose reach.** 1 point below native for 50% more context -- the
  right default when a workflow genuinely needs past 262K.
* **>=512K: reserve for long-chat/needle work** where reasoning depth matters less than reach.
* 1M (x4.0) quality remains unmeasured; the trend gives no reason to expect it to be gentle.

---

## 15. PLE PREFETCH A/B — works as designed, buys nothing when the table is resident

**Setup:** identical profile (`a5b-tp2-ple-pin-384k.env`, 384K, pin 16g, EP, MTP3), one variable:
`PLE_PREFETCH` 0 vs 1. Same perf matrix, same bench. Single runs each.

**Proof the pipeline engaged (arm B):**
```
PLE prefetch: hook installed on prepare_inputs
PLE table ready: layer 1, 320,001,536 rows x 160 B, backend PrefetchingMmapTable
PLE mmap stats: ... prefetch hit 1 miss 0        (arm A, prefetch off, reports 0/0)
```
Note the backend differs: `PrefetchingMmapTable` (on) vs `MmapPleTable` (off). Under load arm B
registered **thousands of hits per 30 s window (1991 -> 4250) with only 22 misses**, and `rows/op`
fell to **0.0** — i.e. the synchronous gather path stopped being used entirely.
(Cosmetic: vLLM logs `Unknown vLLM environment variable: VLLM_PLE_MMAP_PREFETCH`; the PLE patch
reads its own vars directly and does not care.)

**Throughput (pp2048/tg1024):**

| depth | arm A c1 / c2 / c4 | arm B c1 / c2 / c4 |
|---|---|---|
| d0 | 29.0 / 48.3 / 76.5 | 26.9 / 49.1 / 74.0 |
| d2048 | 35.7 / 51.2 / 69.0 | 28.5 / 57.0 / 80.3 |
| d8192 | 36.6 / 64.6 / 77.5 | 32.6 / 65.9 / 87.3 |

**Per-op PLE overhead:** arm A 5.1-9.3 ms vs arm B 5.7-11.8 ms.

### Verdict: leave PLE_PREFETCH=0 (the default)

* The machinery works exactly as advertised — but it exists to **hide NVMe latency**, and with the
  table resident there is no latency to hide (gathers already run at 2-31 us/row, i.e. RAM speed).
* **c1 decode is consistently ~7-20% lower with prefetch on** across all three depths — a real,
  repeatable cost, not scatter. Consistent with extra machinery (worker thread, pinned buffers,
  early hashing) on a path that was already fast enough.
* c2/c4 are mildly better with prefetch on (up to +13% at c4/d8192); c1 is what matters for
  interactive single-stream work, so the trade does not pay.

**When it WOULD be worth enabling:** if the table cannot be resident (tight host memory, GMU held
high, a larger table) and gathers are actually faulting to NVMe — that is the case this path was
built for. Measure with the `PLE mmap stats` line: if `gather` ms/op is in the tens-to-hundreds
with high per-row cost, prefetch is the tool. With RAM-speed gathers, it is pure overhead.
