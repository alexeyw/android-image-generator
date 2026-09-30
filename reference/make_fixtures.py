"""Write the golden fixtures the Kotlin JVM tests compare against.

    python reference/make_fixtures.py

Needs models/host (tokenizer + t_embedder weights) and out/golden_seed42_steps8.npz from
`zimage_ref.py --dump`. Output goes to app/src/test/resources/golden/ and is committed.
"""

import json
import os

import numpy as np
from tokenizers import Tokenizer

import zimage_math as zm
import zimage_ref as zr

OUT = os.path.join(zr.ROOT, "app", "src", "test", "resources", "golden")

TOKENIZER_CASES = [
    "a red apple on a wooden table, studio lighting",
    "A corgi astronaut floating in space, digital art",
    " leading space and trailing space ",
    "numbers 12345 and 3.14159",
    "don't won't it's we're I'll they'd",
    "line one\nline two\n\nline four",
    "tabs\tand   multiple    spaces",
    "Кот в сапогах на фоне заката",
    "東京の夜景、ネオンサイン",
    "emoji 🍎🚀✨ mixed",
    "café naïve résumé",  # NFC composed
    "café decomposed",  # NFD input, must normalize to NFC
    "!!!???...,,,;;;",
    "",
]


def f32list(a):
    return [float(x) for x in np.asarray(a, np.float32).reshape(-1)]


def main():
    os.makedirs(OUT, exist_ok=True)
    host = zr.load_host(zr.HOST_DIR)
    tok: Tokenizer = host["tok"]

    fx = {}
    fx["tokenizer"] = [
        {"prompt": p, "ids": tok.encode(zm.chat_prompt(p), add_special_tokens=False).ids} for p in TOKENIZER_CASES
    ]
    fx["sigmas"] = {str(n): f32list(zm.sigmas(n)) for n in (4, 8, 9)}
    fx["model_t"] = f32list([zm.model_t(s) for s in zm.sigmas(8)[:-1]])
    fx["timestep_freq_t500"] = f32list(zm.timestep_freq(500.0))
    fx["adaln"] = {
        str(t): f32list(zm.adaln_input(t, host["w1"], host["b1"], host["w2"], host["b2"]))
        for t in (0.0, 0.3, 0.7)
    }
    ccos, csin = zm.rope_cos_sin(zm.caption_pos_ids())
    icos, isin = zm.rope_cos_sin(zm.image_pos_ids())
    fx["rope_cap_cos"], fx["rope_cap_sin"] = f32list(ccos), f32list(csin)
    rows = [0, 1, 15, 16, 17, 128, 255]
    fx["rope_img_rows"] = rows
    fx["rope_img_cos"], fx["rope_img_sin"] = f32list(icos[rows]), f32list(isin[rows])
    fx["noise_seed42_head"] = f32list(zm.gaussian_noise(42, 64))
    fx["noise_seed7_sum"] = float(zm.gaussian_noise(7, 16384).astype(np.float64).sum())
    fx["splitmix_seed42"] = [str(v) for v in (lambda r: [r.next_u64() for _ in range(4)])(zm.SplitMix64(42))]
    with open(os.path.join(OUT, "math.json"), "w") as f:
        json.dump(fx, f)

    # Host-loop parity on the recorded run: noise + the step-0 graph output -> the step-0 latent.
    g = np.load(os.path.join(zr.ROOT, "out", "golden_seed42_steps8.npz"))
    for name in ("noise", "step0_final", "step0_latent"):
        np.asarray(g[name], "<f4").tofile(os.path.join(OUT, name + ".f32"))
    with open(os.path.join(OUT, "run.json"), "w") as f:
        json.dump({"prompt": str(g["prompt"]), "seed": 42, "steps": 8, "token_ids": g["token_ids"].tolist()}, f)
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
