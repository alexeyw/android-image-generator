# Z-Image Turbo on Android

[![CI](https://github.com/alexeyw/android-image-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/alexeyw/android-image-generator/actions/workflows/ci.yml)

An open-source Android demo that runs **[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)**
(Alibaba Tongyi-MAI, 6B single-stream diffusion transformer, Apache-2.0) **fully on the phone**:
prompt in, 256×256 image out, no server. Inference uses Google's LiteRT `CompiledModel` over the int8
graphs published at [`litert-community/Z-Image-Turbo-LiteRT`](https://huggingface.co/litert-community/Z-Image-Turbo-LiteRT).

| On a Galaxy S25 Ultra, 29 s | Reference pipeline (Mac) | Reference pipeline (Mac) |
|---|---|---|
| ![cabin](docs/s25_cabin_seed11.png) | ![apple](docs/reference_apple_seed42.png) | ![corgi](docs/reference_corgi_seed7.png) |
| "a cozy cabin in a snowy forest at night, warm light in the windows" · seed 11 | "a red apple on a wooden table, studio lighting" · seed 42 | "a corgi astronaut floating in space, digital art" · seed 7 |

The phone and the reference agree: the same prompt and seed give the same image at 35 dB PSNR (the
residue is XNNPACK kernel differences between ARM and x86 CPUs).

## What the app does

The published graphs cover the heavy compute; everything around them runs on the host in Kotlin:

```
prompt ─► Qwen3 BPE (Kotlin) ─► embed_tokens (mmap, bf16) ─► qwen_enc ─► z_embc ─► [cap pad token] ─► z_refc ─┐
                                                                                         once per prompt │
seeded noise ─► for each of 8 steps:                                                                      │
   patchify ─► z_embx ─► z_refx ─► concat [image | caption] ─► zc_main0..5 ─► zc_final ◄───────────────────┘
   ─► unpatchify ─► Euler step (flow matching, shift 3)
─► denormalize ─► zvae ─► 256×256 RGB
```

- **Turbo needs no CFG** (it is distilled for guidance 0), so each step is one transformer pass.
- **Everything runs on the CPU by default** (XNNPACK), one graph at a time. The first generation packs
  the weights into an on-disk XNNPACK cache (~9 GB); after that a 0.9 GB DiT shard "loads" in ~10 ms
  as a memory map, and memory stays low because the weights are file-backed pages.
- **GPU is an experimental switch.** See [Performance](#performance) for why it is not the default on 12 GB.
- **Prompt budget:** the 256 px graphs take a fixed 32-token caption, chat template included, so a prompt
  can be about 23 tokens. The app counts tokens as you type.
- **History:** every image is kept with its prompt, seed, steps, accelerator and thread count
  (`files/history/<id>.png` + `<id>.json`); from the history you can reuse those settings, save the image
  to the gallery, or delete one or all entries. The form itself is remembered between launches.

### Things the model card does not say

Found while building the reference, and pinned in code and tests:

- `z_refx` and `zc_main*` also take RoPE `cos`/`sin` `[1,1,S,64]` and the adaln vector `[1,256]`;
  `z_refc` takes `cos`/`sin`; `zc_final` takes adaln. Input order is `(x, cos, sin[, adaln])`.
- Four host tensors exist in no graph: `embed_tokens`, the timestep-embedder MLP (`t_embedder.mlp.{0,2}`)
  and `cap_pad_token`. They are read from the upstream checkpoint with HTTP Range requests (0.78 GB of a
  32 GB checkpoint); nothing is re-hosted.
- `qwen_enc` has no attention-mask input but is causal: padding changes the prompt rows by exactly 0.
- RoPE positions: caption `(1..32, 0, 0)`, image `(33, h, w)`; the unified sequence is `[image, caption]`.

## Requirements

- An arm64 phone with Android 12+ and **12 GB RAM**. Tested on a Galaxy S25 Ultra (Android 16) and an
  API 36 emulator; 8 GB phones are untested (the model card reports a GPU run on an 8 GB Pixel 8a).
- **~21 GB free storage**: 10.6 GB of models plus a 9.1 GB XNNPACK weight cache (regenerated if the
  system clears the app cache).
- Wi-Fi for the first download (10.6 GB), or a Mac/PC to `adb push` the files.

## Limitations

- **256×256 only.** The published graphs have fixed shapes; other sizes need a new conversion.
- **~23-word prompts.** The caption slot is 32 tokens including the chat template.
- **The app must stay open** during the download and the generation (no background service yet).
- **No negative prompt or CFG.** Turbo is distilled for guidance 0; the graphs would support a second
  branch, but it doubles the time and is not wired up.
- **GPU is experimental** and not faster on 12 GB phones (see [Performance](#performance)).

## Build and run

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

On first launch tap **Download**: the app fetches the 13 graphs from `litert-community` and the host
tensors from `Tongyi-MAI`, both at pinned revisions. Checked on a Galaxy S25 Ultra from a clean install:
10.56 GB in ~10 minutes over Wi-Fi (14–29 MB/s), resumed byte-exactly after *Pause* and after the process
was killed mid-file, and all 21 files matched the upstream SHA-256.

Faster for development: download once on the computer and push over USB:

```bash
scripts/download_graphs.sh                 # 9.78 GB of .tflite into models/litert
python3 -m venv .venv && .venv/bin/pip install -r reference/requirements.txt   # Python 3.10+
.venv/bin/python reference/host_assets.py  # 0.78 GB of host tensors into models/host
scripts/push_models.sh
```

Headless run from a shell (the result goes to the history, `Android/data/io.github.alexeyw.zimage/files/history/`;
timings to logcat under the `ZImage` / `ZImageRuntime` tags; the extras do not change the saved form):

```bash
adb shell am start -n io.github.alexeyw.zimage/.MainActivity --ez autorun true \
  --es prompt "a red fox in fresh snow" --el seed 7 --ei steps 8 --es backend CPU --ei threads 6
```

## Performance

Galaxy S25 Ultra (Snapdragon 8 Elite, 12 GB, Android 16), 256×256, 8 steps, measured 2026-09-29/30:

| Mode | Per image | Notes |
|---|---|---|
| **CPU, 6 threads, warm cache (default)** | **29 s** | text encoder 2.0 s, DiT 21.8 s, VAE 1.1 s |
| CPU, first run after install | 60 s | includes writing the 9.1 GB XNNPACK weight cache |
| GPU, graphs streamed | 284 s | a shard runs in 0.2 s but reloads in ~4 s every step |
| GPU, shards resident | killed | ~1.3 GB per resident shard; lmkd kills the app at the 5th |

On the GPU, with `precision = FP32` (FP16 overflows to NaN in adaLN) lmkd kills the app while the first
0.9 GB shard loads (7.6 GB RSS+swap), consistent with the delegate expanding the int8 weights to FP32.
`allowSrcQuantizedFcConvOps` makes it fit (4.4 GB peak), but quantizes the activations too: the image stays coherent but drifts from the
CPU result (17 dB PSNR). A phone with 16 GB+ may be able to keep all shards resident; untested.

### CPU threads

[`scripts/bench_threads.py`](scripts/bench_threads.py) runs the same prompt and seed with each thread count,
interleaved (4, 6, 8, 6, 8, 4, 8, 4, 6) and with a cool-down before every run:

| XNNPACK threads | Runs | Median |
|---|---|---|
| 4 | 43.5 · 31.8 · 36.6 s | 36.6 s |
| **6** | **28.8 · 28.1 · 28.8 s** | **28.8 s** |
| 8 | 51.4 · 29.8 · 36.3 s | 36.3 s |

Six threads was both the fastest and the steadiest (0.7 s spread against 12–22 s for 4 and 8); all nine
images were pixel-identical. Later single runs agreed: 29.3 and 29.4 s with 6 threads. The S25 Ultra
has six 3.53 GHz and two 4.47 GHz cores; the app uses "cores minus two, at most six", which is 6 there and
an untested guess elsewhere. The *CPU threads* slider in the app overrides it (it also applies to the text
encoder, which always runs on the CPU), and the result line under the image shows the count used. Caveats: the phone was charging over USB, and the cool-down usually hit its
7-minute cap at 31–35 °C instead of returning to the 28 °C idle temperature.

## Reference pipeline (Mac / Linux)

[`reference/`](reference) runs the same 13 graphs with the `ai-edge-litert` Python package on the CPU.
It is the source of truth for the host math and produces the golden fixtures the Kotlin tests use.

```bash
.venv/bin/python reference/zimage_ref.py "a red apple on a wooden table, studio lighting" --seed 42 --dump
.venv/bin/python reference/make_fixtures.py   # refresh app/src/test/resources/golden
```

On an Apple-silicon Mac, one image takes ~130 s. The time goes to loading and weight packing: each DiT
shard takes ~2.3 s to load and ~0.27 s to run.

## Testing

```bash
./gradlew testDebugUnitTest
```

JVM tests compare the Kotlin port against fixtures recorded from the verified reference run: the
tokenizer token-for-token on 14 prompts (Cyrillic, CJK, emoji, NFD, whitespace), the sigma schedule,
timestep embedding, RoPE tables, the seeded noise stream (bit-identical SplitMix64), and a full Euler
step on real graph output. Tests that need `models/host` skip without it; CI fetches the ~17 MB they
need (`reference/host_assets.py --skip-embed`) so all of them run on every push.

## Project layout

| Path | What |
|---|---|
| `app/.../pipeline/ZImageMath.kt` | Scheduler, timestep MLP, RoPE, (un)patchify, noise, VAE boundary |
| `app/.../pipeline/QwenTokenizer.kt` | Qwen3 byte-level BPE with the Z-Image chat template |
| `app/.../pipeline/GraphRunner.kt` | LiteRT `CompiledModel` loading, CPU/GPU choice, weight caches, fallback |
| `app/.../pipeline/ZImagePipeline.kt` | The generation loop |
| `app/.../download/ModelDownloader.kt` | Hugging Face download with Range-based safetensors extraction |
| `app/.../data/HistoryStore.kt`, `SettingsStore.kt` | Generation history (PNG + JSON files) and the saved form |
| `app/.../ui/` | Compose screens: generator, history grid, history entry |
| `reference/` | Python reference pipeline, host-asset extractor, fixture generator |

## Credits and licenses

- Code: Apache-2.0 (see [LICENSE](LICENSE)).
- Model: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI, Apache-2.0;
  LiteRT conversion by [litert-community](https://huggingface.co/litert-community/Z-Image-Turbo-LiteRT), Apache-2.0.
- `QwenTokenizer.kt` is adapted from the Bonsai Image Android sample in
  [john-rocky/hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm), Apache-2.0 (see [NOTICE](NOTICE)).
