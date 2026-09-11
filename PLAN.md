# LANE 3 — Qwen3.8-Flash-Next NVFP4 INT4-AutoRound A5B on 2x DGX Spark: frozen acceptance plan

Written 2026-09-11 **before** the checkpoint landed. Purpose: fix the acceptance criteria while we
have no emotional investment in the result, so a bad outcome cannot be rationalised later.

## Why this lane exists

| Stack | hardmode | real-world decode | prefix caching under MTP | long context |
|---|---|---|---|---|
| NVFP4 / Tony nightly (current) | 87 | 40.8 tok/s median | **ZERO hits** — vllm#53670 class | unusable (full re-prefill) |
| FP8 (Aug 27) | 82-84 | ~35 tok/s | worked (official image) | ok |
| **A5B INT4-AR (this lane)** | 89 median (5-trial, ds2atc) | 60-70 claimed, 1 Spark | **expected to work** (official-image lineage) | 512K + 2M KV target |

Author's own caveat: Saren's fork README says the repo (LLM-prepared) *"is very likely something will
break or cannot reproduce, especially the claimed prefix cache fix part."* A5B builds on Saren's image.
That is exactly why Gate C exists and runs FIRST.

## Gates — run in order, abort on first failure

| Gate | Checks | Pass criterion | Exit code on fail |
|---|---|---|---|
| **A** | server up, served model id, max_model_len, engine version | model answers `/v1/models`; id matches expectation | 10 |
| **B** | functional: plain chat; `tool_choice=required` → valid `tool_calls`; chunked prefill on a ~3.7K-token prompt; multi-turn tool result | all four return sane, non-empty results | 20 |
| **C** | **PREFIX CACHE** — `vllm:prefix_cache_hits_total` before → send same ~8K prompt twice → after | **delta > 0** | 30 |
| **D** | speed: llama-benchy depth 0/2K/8K × c 1/2/4 (TEB `--perf-only`) | recorded; no pass/fail — comparison only | 40 (tool error) |
| **E** | quality: hardmode tool-eval, seed 42, temp 0, 1200s timeout, 16k max_tokens | recorded; 89=good, 93-94=exceptional | 50 (tool error) |

**Gate C is the kill switch.** Zero hits ⇒ long context is unusable (full re-prefill every turn) and
the lane is dead regardless of speed or quality. Do not proceed to D/E on a Gate C failure.

## Decision rules (frozen)

1. **Gate C fails → stop.** Report, keep the checkpoint, do not benchmark. The download is the only cost.
2. **Gates A-C pass, D/E run →** compare honestly against the table above:
   - Speed win alone is NOT a pass. Primo's law: no speed trick is worth a quality point.
   - Quality must land ≥ 86 (current NVFP4 stack on the same suite) to be worth a lane.
   - Long-context usability is the differentiator: it is the one thing the current stack cannot do.
3. **TP2 only after TP1 passes all gates.** TP1 first: reproduce, measure, quality-gate. TP2 is a
   separate experiment for the 2M-KV / 512K spec — and it is unproven for this checkpoint (no prior art:
   zero TP/rank/world_size references in the upstream repo or the 739-line PLE patch).
4. **Numbers must come from workloads, not synthetic prose.** llama-benchy understates real decode by
   ~1.7x (measured: 24.3 vs 40.8 tok/s median). Report both, label both.

## Known TP2 unknowns (to be resolved by experiment, not assumed)

- PLE table mmap under TP: per-rank local copy assumed; untested.
- YaRN to 512K on the QSA sparse-attention indexer: untested at that depth.
- GDN/mamba state under TP may need an align-mode fix (GLM's lesson: `--mamba-cache-mode align`).
- Memory arithmetic (says it fits): 36GB weights/rank + 31GB KV/rank (2M @ bf16) ≈ 67GB of ~121GB.
  fp8-KV patch halves the KV term.
