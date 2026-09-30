"""One-off probe: is qwen_enc causal, so that the padding after the prompt cannot leak into it?

The graph has no attention-mask input. diffusers masks the pad tokens; a causal graph makes the
mask irrelevant for the valid prefix, a bidirectional one would not. Runs the encoder twice with
different padding and compares the prompt rows.
"""

import sys

import numpy as np

import zimage_math as zm
import zimage_ref as zr

g = zr.Graphs(zr.GRAPH_DIR, threads=8, keep=False)
host = zr.load_host(zr.HOST_DIR)
ids = host["tok"].encode(zm.chat_prompt(sys.argv[1] if len(sys.argv) > 1 else "a red apple"), add_special_tokens=False).ids
n = len(ids)
print("tokens", n, ids)

a = zm.bf16_rows_to_f32(host["embed"], ids + [zm.PAD_TOKEN_ID] * (zm.TEXT_LEN - n))
rng = np.random.default_rng(0)
b = zm.bf16_rows_to_f32(host["embed"], ids + list(rng.integers(0, 150000, zm.TEXT_LEN - n)))
ha = g.run("qwen_enc", a[None])[0]
hb = g.run("qwen_enc", b[None])[0]
d = np.abs(ha[:n] - hb[:n]).max()
print(f"prefix max|diff| = {d:.3e} (0 => causal, padding cannot leak)")
print(f"tail   max|diff| = {np.abs(ha[n:] - hb[n:]).max():.3e}")
print(f"prefix std {ha[:n].std():.3f}, nan {np.isnan(ha).any()}")
