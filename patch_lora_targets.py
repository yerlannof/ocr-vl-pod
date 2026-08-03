#!/usr/bin/env python3
"""Patch ERNIE default LoRA target modules for PaddleOCR-VL.

The stock else-branch in ernie/utils/peft_utils.py targets fused module
names (.*qkv_proj.* / .*up_gate_proj.*) that DO NOT exist in PaddleOCR-VL —
its language layers use separate q/k/v/o_proj and gate/up/down_proj.  With
the stock list only o_proj+down_proj match (36 LoRA pairs of 126), which is
exactly the v1 failure mode.  This pins the full LM-part target list,
anchored to model.layers so the vision encoder stays frozen.
"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "ernie/utils/peft_utils.py"
src = f.read_text()
if "PATCHED-V2-TARGETS" in src:
    print("targets: already patched")
    sys.exit(0)
old = '''            target_modules = [
                ".*qkv_proj.*",
                ".*o_proj.*",
                ".*up_gate_proj.*",
                ".*down_proj.*",'''
new = '''            # PATCHED-V2-TARGETS: real PaddleOCR-VL module names (LM part)
            target_modules = [
                ".*model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\.q_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\.k_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\.v_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.self_attn\\\\.o_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.mlp\\\\.gate_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.mlp\\\\.up_proj.*",
                ".*model\\\\.layers\\\\.\\\\d+\\\\.mlp\\\\.down_proj.*",'''
assert old in src, "target_modules block not found — ERNIE layout changed?"
f.write_text(src.replace(old, new, 1))
print("targets: patched")
