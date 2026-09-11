#!/usr/bin/env python3
r"""
Apply the auto_gptq RoutedExperts exclusion-order fix to a vLLM image tree.

THE BUG
-------
`AutoGPTQConfig.get_quant_method()` handles MoE layers in this order:

    1. shape check  -> check_moe_marlin_supports_layer(layer, group_size, ...)
    2. if unsupported: return MoeWNA16Config...get_quant_method(layer, prefix)   # <-- RETURNS
    3. else: get_moe_quant_method(...)   # <-- only HERE are dynamic exclusions honoured

`get_moe_quant_method()` is the function that reads the config's `dynamic` rules and, for a
`-:<regex>` (negative/exclusion) match, returns `UnquantizedFusedMoEMethod` — the correct path
for experts that are stored unquantized. But step 2 returns before it is ever consulted, so an
explicit "do not quantize this module" is silently overridden by an automatic fallback.

Consequence on Qwen3.8-Flash-Next A5B at TP=2: the MTP layer's experts have
moe_intermediate_size 640. At TP2 each rank gets 640/2 = 320, and

    320 % max(64, group_size=128) = 64 != 0   ->  Marlin unsupported -> WNA16 fallback

WNA16 then does not register the names the MTP weight loader expects, and load fails with:

    AttributeError: Layer 'mtp.layers.48.mlp.experts' has no parameter 'w2_weight'
                    for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight'

...even though the checkpoint's MTP experts are plain bf16 (no quantized keys at all) and the
config already carries `-:mtp\..*` in `dynamic`. `--enable-expert-parallel` "fixes" it only
because EP keeps the expert dimension whole (640 % 128 == 0), so Marlin stays eligible.

THE FIX
-------
Consult the dynamic exclusion BEFORE the shape-based fallback. An explicit user exclusion should
outrank an automatic fallback. After the reorder, `-:mtp\..*` selects UnquantizedFusedMoEMethod,
which registers `w13_weight`/`w2_weight` — exactly the names the MTP loader expects.

Idempotent: re-running is a no-op. Verifies the result compiles and that the new ordering is
actually present, so it fails loudly rather than reporting success on a no-op.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET_REL = "usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_gptq.py"

ORIGINAL = """        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=not self.desc_act
            ):"""

PATCHED = """        if isinstance(layer, RoutedExperts):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            # An explicit `-:<regex>` exclusion in the quantization config must outrank the
            # automatic shape-based WNA16 fallback. get_moe_quant_method() is what honours those
            # exclusions (returning UnquantizedFusedMoEMethod, which registers w13_weight/w2_weight),
            # but the fallback below returns before it is consulted -- so e.g. MTP experts stored
            # unquantized, whose per-rank intermediate (640/TP) is not a multiple of group_size,
            # were silently routed to WNA16 and failed to load with "no parameter 'w2_weight'".
            if (
                get_dynamic_override(  # noqa: E712
                    deepcopy(self),
                    layer_name=prefix,
                )
                == False
            ):  # noqa: E712
                return get_moe_quant_method(self, layer, prefix, AutoGPTQMoEMethod)

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=not self.desc_act
            ):"""

MARKER = "An explicit `-:<regex>` exclusion in the quantization config must outrank the"
SHAPE_CALL = "check_moe_marlin_supports_layer("
DYNAMIC_CALL = "get_dynamic_override("


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "/")
    target = root / TARGET_REL
    if not target.exists():
        print(f"FAIL: not found: {target}", file=sys.stderr)
        return 2

    src = target.read_text()
    if MARKER in src:
        print("already applied — skipping")
        return 0
    if ORIGINAL not in src:
        print("FAIL: anchor not found; image's auto_gptq.py differs from the expected revision",
              file=sys.stderr)
        return 3

    target.write_text(src.replace(ORIGINAL, PATCHED, 1))

    # verify: compiles, and the new branch precedes the shape check
    src = target.read_text()
    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover
        print(f"FAIL: patched file does not compile: {exc}", file=sys.stderr)
        return 4
    at = src.index(MARKER)
    if src.index(SHAPE_CALL, at) < src.index(DYNAMIC_CALL, at):
        print("FAIL: exclusion check is not ordered before the shape check", file=sys.stderr)
        return 5

    print(f"OK: patched {target.name}; dynamic exclusion now precedes the Marlin shape check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
