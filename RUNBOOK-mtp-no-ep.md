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
