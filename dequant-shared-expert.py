#!/usr/bin/env python3
"""Make the A5B INT4-AutoRound checkpoint TP2-shardable by dequantizing the 144
blockwise-fp8 shared_expert tensors to bf16.

WHY: vLLM's vllm_fp8_hybrid patch treats a layer as fp8 iff its .weight dtype is F8_E4M3
AND a .weight_scale_inv sibling exists. Those weights are block-128x128 quantized, and
fp8.py::validate_fp8_block_shape() requires (dim / TP) % 128 == 0. The shared expert's
gate/up/down projections have a 640 dimension -> TP2 gives 320 -> 128 does not divide 320,
so weight creation fails. 640/TP must be a multiple of 128, so no TP>1 can work while those
stay fp8. Dequantizing them to bf16 (which has no block constraint) makes them shard freely.

Pre-flight (measured): of 300 fp8 weights, 156 are already TP2-safe and exactly 144 are
problematic -- all shared_expert, all in model-healed-shared-expert.safetensors.

The ORIGINAL checkpoint is never modified: every other shard is symlinked into the new dir.
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/var/tmp/models/qwen38fn-a5b-int4"
DST = "/var/tmp/models/qwen38fn-a5b-int4-tp2safe"
HEALED = "model-healed-shared-expert.safetensors"
NEWFILE = "model-healed-shared-expert-bf16.safetensors"
BLOCK = 128

os.makedirs(DST, exist_ok=True)
print(f"source : {SRC}\nnew dir: {DST}\n")

# ── 1. dequantize ────────────────────────────────────────────────────────────────────────
out_tensors = {}
with safe_open(os.path.join(SRC, HEALED), "pt") as h:
    keys = list(h.keys())
    weights = [k for k in keys if k.endswith(".weight")]
    scales = [k for k in keys if k.endswith(".weight_scale_inv")]
    print(f"healed file: {len(weights)} weights, {len(scales)} scales")
    assert len(weights) == 144, f"expected 144 fp8 weights, found {len(weights)}"

    for k in weights:
        w = h.get_tensor(k)                                  # fp8 e4m3
        s = h.get_tensor(k + "_scale_inv")                   # fp32 [ceil(o/128), ceil(i/128)]
        o, i = w.shape
        so, si = s.shape
        assert so == -(-o // BLOCK) and si == -(-i // BLOCK), (
            f"scale/shape mismatch for {k}: w={tuple(w.shape)} s={tuple(s.shape)}")
        # expand per-block scales to full tensor, then dequant
        s_exp = s.repeat_interleave(BLOCK, dim=0)[:o].repeat_interleave(BLOCK, dim=1)[:, :i]
        dq = (w.to(torch.float32) * s_exp).to(torch.bfloat16)
        assert torch.isfinite(dq).all(), f"non-finite values after dequant: {k}"
        out_tensors[k] = dq

print(f"dequantized {len(out_tensors)} tensors -> bf16 "
      f"({sum(t.numel() for t in out_tensors.values())/1e6:.1f}M params)")
save_file(out_tensors, os.path.join(DST, NEWFILE), metadata={"format": "pt"})
print(f"wrote {NEWFILE} ({os.path.getsize(os.path.join(DST, NEWFILE))/1e6:.0f} MB)")

# ── 2. symlink everything else (originals untouched) ─────────────────────────────────────
linked = 0
for f in os.listdir(SRC):
    if f in (HEALED, "model.safetensors.index.json"):
        continue
    dst = os.path.join(DST, f)
    if os.path.lexists(dst):
        continue
    os.symlink(os.path.join(SRC, f), dst)
    linked += 1
print(f"symlinked {linked} other files from the original")

# ── 3. patched index: retarget shared_expert weights, DROP their scale entries ───────────
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
retargeted = dropped = 0
for k in list(wm):
    if "shared_expert" not in k:
        continue
    if k.endswith(".weight") and (k + "_scale_inv") in wm:
        wm[k] = NEWFILE
        retargeted += 1
    elif k.endswith(".weight_scale_inv"):
        del wm[k]
        dropped += 1
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
print(f"index: retargeted {retargeted} weights, dropped {dropped} scale entries")

# ── 4. verify the fp8-hybrid detector no longer flags them ───────────────────────────────
dtypes = {}
for f in set(wm.values()):
    p = os.path.join(DST, f)
    with safe_open(p, "pt") as h:
        for k in h.keys():
            dtypes[k] = str(h.get_slice(k).get_dtype())
still_fp8 = [k for k in wm
             if "shared_expert" in k and k.endswith(".weight")
             and "F8_E4M3" in dtypes.get(k, "")
             and (k + "_scale_inv") in wm]
print()
print(f"VERIFY: shared_expert weights still detected as fp8 = {len(still_fp8)} (want 0)")
if still_fp8:
    print("  !! " + ", ".join(still_fp8[:3]))
    sys.exit(1)
print("OK — the patch will now load shared_expert as plain bf16, which shards under TP2.")
