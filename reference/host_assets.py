"""Fetch the host-side tensors the LiteRT graphs do not contain.

The Z-Image-Turbo LiteRT graphs cover the heavy compute, but the host loop still needs:
  - the Qwen3 token embedding table (`embed_tokens`, bf16 [151936, 2560]) to build `inputs_embeds`,
  - the timestep embedder MLP (`t_embedder.mlp.{0,2}`) to build the per-step adaln vector,
  - the learned caption pad token (`cap_pad_token`) the host substitutes after `z_embc`,
  - the Qwen2 BPE tables (`vocab.json`, `merges.txt`).

All of them live in the Apache-2.0 upstream checkpoint `Tongyi-MAI/Z-Image-Turbo`. Only the needed
byte ranges of the multi-GB safetensors shards are downloaded (HTTP Range), so nothing is re-hosted.

Output layout (identical on the Mac and on the phone, so `adb push` works):
  embed_tokens.bf16      raw little-endian bf16, row-major [151936, 2560]
  t_emb_w1.f32 ...       raw little-endian float32, PyTorch Linear layout [out, in]
  cap_pad_token.f32      float32 [3840]
  vocab.json, merges.txt, tokenizer.json
"""

import argparse
import json
import os
import struct
import urllib.request

import numpy as np

REPO = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/f332072aa78be7aecdf3ee76d5c247082da564a6"
TEXT_SHARD = "text_encoder/model-00001-of-00003.safetensors"
DIT_SHARD = "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"

# safetensors name -> (output file, output dtype)
DIT_TENSORS = {
    "t_embedder.mlp.0.weight": "t_emb_w1.f32",  # [1024, 256]
    "t_embedder.mlp.0.bias": "t_emb_b1.f32",  # [1024]
    "t_embedder.mlp.2.weight": "t_emb_w2.f32",  # [256, 1024]
    "t_embedder.mlp.2.bias": "t_emb_b2.f32",  # [256]
    "cap_pad_token": "cap_pad_token.f32",  # [1, 3840]
}
TOKENIZER_FILES = ["tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer.json"]


def http_range(url: str, start: int, end_exclusive: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_exclusive - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    if len(data) != end_exclusive - start:
        raise IOError(f"short read {len(data)} != {end_exclusive - start} for {url}")
    return data


def safetensors_header(url: str) -> tuple[dict, int]:
    """Returns (header json, absolute offset of the data section)."""
    (n,) = struct.unpack("<Q", http_range(url, 0, 8))
    header = json.loads(http_range(url, 8, 8 + n))
    return header, 8 + n


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def fetch_tensor(url, header, base, name) -> tuple[bytes, dict]:
    meta = header[name]
    a, b = meta["data_offsets"]
    return http_range(url, base + a, base + b), meta


def stream_range_to_file(url, start, end_exclusive, path, chunk=64 << 20):
    tmp = path + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    with open(tmp, "ab") as f:
        pos = start + done
        while pos < end_exclusive:
            nxt = min(pos + chunk, end_exclusive)
            f.write(http_range(url, pos, nxt))
            pos = nxt
            print(f"  {path}: {(pos - start) / (end_exclusive - start):6.1%}", end="\r", flush=True)
    os.replace(tmp, path)
    print()


def main(out_dir: str, skip_embed: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for rel in TOKENIZER_FILES:
        dst = os.path.join(out_dir, os.path.basename(rel))
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{REPO}/{rel}", dst)
            print("got", dst)

    url = f"{REPO}/{DIT_SHARD}"
    header, base = safetensors_header(url)
    for name, fname in DIT_TENSORS.items():
        raw, meta = fetch_tensor(url, header, base, name)
        assert meta["dtype"] in ("BF16", "F32"), meta
        arr = bf16_to_f32(raw) if meta["dtype"] == "BF16" else np.frombuffer(raw, "<f4")
        arr.astype("<f4").tofile(os.path.join(out_dir, fname))
        print(f"got {fname:18s} {meta['dtype']} {meta['shape']}")

    dst = os.path.join(out_dir, "embed_tokens.bf16")
    if not skip_embed and not os.path.exists(dst):
        url = f"{REPO}/{TEXT_SHARD}"
        header, base = safetensors_header(url)
        meta = header["model.embed_tokens.weight"]
        assert meta["dtype"] == "BF16" and meta["shape"] == [151936, 2560], meta
        a, b = meta["data_offsets"]
        stream_range_to_file(url, base + a, base + b, dst)
    print("host assets ready in", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "models", "host"))
    ap.add_argument("--skip-embed", action="store_true", help="skip the 0.78 GB embedding table (the JVM tests do not need it)")
    a = ap.parse_args()
    main(a.out_dir, a.skip_embed)
