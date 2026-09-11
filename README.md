# Qwen3.8-Flash-Next A5B INT4-AutoRound — TP=2 across 2× DGX Spark (GB10)

Getting the azampatti/Saren **A5B INT4-AutoRound** recipe — written for a *single* DGX Spark — running
across **two** of them with tensor parallelism, native MTP speculative decoding and working prefix
caching. It did not work out of the box; two distinct blockers had to be solved.

**→ The full write-up is [`TP2-FINDINGS.md`](TP2-FINDINGS.md). Read that first.**

## The two blockers, in one line each

1. **The checkpoint could not be TP-sharded.** 300 layers are blockwise-fp8 (128×128); the *shared
   expert* has a 640 dimension, and fp8 requires `(dim / TP) % 128 == 0`, so 640/2 = 320 fails.
   640/TP is a multiple of 128 only when **TP=1**. → **Requant those 144 tensors to bf16**
   (`tools/dequant-shared-expert.py`).
2. **MTP then failed to load at TP2** — the MTP layer's experts fail the Marlin support check, fall
   back to `MoeWNA16`, and the loader expects Marlin names
   (`AttributeError: … has no parameter 'w2_weight'`). → **`--enable-expert-parallel`**.
   Config-level `dynamic` exclusions *cannot* fix this (the check never consults them).

## Files

| file | what it is |
|---|---|
| `TP2-FINDINGS.md` | the complete write-up: blockers, root causes, requant procedure, working recipe, measurements, 10 gotchas, guidance for new MTP/DFlash variants |
| `tools/dequant-shared-expert.py` | the **requant**: dequantizes the blockwise-fp8 shared expert to bf16 in a new checkpoint dir (originals untouched, symlink farm + patched index) |
| `tp2-serve.sh` / `tp2-rank.sh` | two-node launcher (head + worker over RoCE), env-driven |
| `gate.sh` | acceptance harness — health, functional gates, and a **prefix-cache check** that reads the resolved block size from `/metrics` and probes with a diverse ≥4-block corpus |
| `PLAN.md` | the frozen acceptance criteria written *before* the experiment (gates, decision rules, exit codes) |

## Verified state

* both ranks load (34.98–36.99 GiB/rank), engine starts, all gates pass
* **prefix caching works** (`prefix_cache_hits_total` 0 → 16,800 on two identical sends)
* MTP acceptance **61–64%** on agentic traffic
* KV pool **1,251,206 tokens** at `--kv-cache-memory-bytes 20g`, ~55 GiB/rank of 121 GiB

## Honest measurement note

At TP=2 the **single-stream rate is ~25–30% *lower*** than single-Spark (38.2 vs 48.9 tok/s at c1/d0,
llama-benchy pp2048/tg1024). Cross-node all-reduce per layer costs more than the halved weight traffic
saves, and MTP multiplies it. TP2's wins are **aggregate throughput under concurrency** (90.8 tok/s at
c4/d2048 vs 78.2) and a **~2× larger KV pool**. See §6 of the write-up.

## Credit

Checkpoint by **azampatti**, upstream recipe/image patching by **Saren-Arterius** (a fork of
**blazux**'s single-Spark work). This repo only documents what it took to run it on two nodes.
