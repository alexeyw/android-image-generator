"""Reference Z-Image-Turbo pipeline over the published LiteRT graphs, on the Mac CPU.

    python reference/zimage_ref.py "a red apple on a wooden table, studio lighting" --seed 42

Graph order per image (graphs are loaded one at a time, like on an 8-12 GB phone):
  qwen_enc                                  once per prompt
  z_embc -> [host pad token] -> z_refc      once per prompt (the context refiner has no adaln)
  per step: z_embx -> z_refx -> zc_main0..5 -> zc_final -> [host unpatchify, Euler]
  zvae                                      once
"""

import argparse
import os
import time

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel, _create_default_environment
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import CpuOptions, Options
from PIL import Image
from tokenizers import Tokenizer

import zimage_math as zm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GRAPH_DIR = os.path.join(ROOT, "models", "litert")
HOST_DIR = os.path.join(ROOT, "models", "host")


class Graphs:
    def __init__(self, graph_dir: str, threads: int, keep: bool):
        self.dir, self.threads, self.keep = graph_dir, threads, keep
        self.env = _create_default_environment()  # one Environment shared by every graph
        self.cache: dict[str, CompiledModel] = {}
        self.timings: dict[str, list[float]] = {}

    def _load(self, name: str) -> CompiledModel:
        if name in self.cache:
            return self.cache[name]
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=self.threads))
        m = CompiledModel.from_file(os.path.join(self.dir, name + ".tflite"), options=opts, environment=self.env)
        if self.keep:
            self.cache[name] = m
        return m

    def run(self, name: str, *inputs: np.ndarray) -> np.ndarray:
        t0 = time.time()
        m = self._load(name)
        t1 = time.time()
        sig = list(m.get_signature_list())[0]
        ind = m.get_input_tensor_details(sig)
        outd = m.get_output_tensor_details(sig)
        in_names = m.get_signature_list()[sig]["inputs"]
        out_name = m.get_signature_list()[sig]["outputs"][0]
        ib, ob = m.create_input_buffers(0), m.create_output_buffers(0)
        for n, buf, x in zip(in_names, ib, inputs):
            want = tuple(ind[n]["shape"])
            assert x.size == int(np.prod(want)), f"{name}.{n}: got {x.shape}, graph wants {want}"
            buf.write(np.ascontiguousarray(x, np.float32).reshape(-1))
        m.run_by_index(0, ib, ob)
        shape = tuple(outd[out_name]["shape"])
        out = ob[0].read(int(np.prod(shape)), np.float32).reshape(shape)
        if not self.keep:
            m.close()
        self.timings.setdefault(name, []).append(time.time() - t0)
        print(f"  {name:9s} load {t1 - t0:5.2f}s total {time.time() - t0:5.2f}s", flush=True)
        return out


def load_host(host_dir: str):
    f32 = lambda n, shape: np.fromfile(os.path.join(host_dir, n), "<f4").reshape(shape)
    return dict(
        tok=Tokenizer.from_file(os.path.join(host_dir, "tokenizer.json")),
        embed=np.memmap(os.path.join(host_dir, "embed_tokens.bf16"), "<u2", "r", shape=(151936, zm.TEXT_DIM)),
        w1=f32("t_emb_w1.f32", (1024, 256)),
        b1=f32("t_emb_b1.f32", (1024,)),
        w2=f32("t_emb_w2.f32", (256, 1024)),
        b2=f32("t_emb_b2.f32", (256,)),
        cap_pad=f32("cap_pad_token.f32", (zm.DIM,)),
    )


def encode_prompt(g: Graphs, host, prompt: str, dump: dict | None = None) -> np.ndarray:
    """-> refined caption tokens [CAP_LEN, DIM] (timestep-independent, computed once)."""
    ids = host["tok"].encode(zm.chat_prompt(prompt), add_special_tokens=False).ids
    n = len(ids)
    if n > zm.CAP_LEN:
        raise ValueError(f"prompt is {n} tokens with the chat template; the 256 px graphs take {zm.CAP_LEN}")
    padded = ids + [zm.PAD_TOKEN_ID] * (zm.TEXT_LEN - n)
    embeds = zm.bf16_rows_to_f32(host["embed"], padded)  # [64, 2560]
    hidden = g.run("qwen_enc", embeds[None])[0]  # [64, 2560], penultimate layer
    cap = hidden[:n]
    cap = np.concatenate([cap, np.repeat(cap[-1:], zm.CAP_LEN - n, 0)])  # _pad_with_ids: repeat last
    emb = g.run("z_embc", cap[None])[0]  # [32, 3840]
    emb[n:] = host["cap_pad"]  # torch.where(pad_mask, cap_pad_token, feats)
    cos, sin = zm.rope_cos_sin(zm.caption_pos_ids())
    ref = g.run("z_refc", emb[None], cos[None, None], sin[None, None])[0]
    if dump is not None:
        dump.update(token_ids=np.array(ids, np.int32), text_hidden=cap, cap_emb=emb, cap_ref=ref)
    return ref


def dit(g: Graphs, host, latent: np.ndarray, cap_ref: np.ndarray, t: float, dump: dict | None = None) -> np.ndarray:
    """One transformer forward -> model output in latent layout [16, 32, 32]."""
    adaln = zm.adaln_input(t, host["w1"], host["b1"], host["w2"], host["b2"])[None]
    x_cos, x_sin = zm.rope_cos_sin(zm.image_pos_ids())
    c_cos, c_sin = zm.rope_cos_sin(zm.caption_pos_ids())
    x = g.run("z_embx", zm.patchify(latent)[None])
    x = g.run("z_refx", x, x_cos[None, None], x_sin[None, None], adaln)[0]
    u = np.concatenate([x, cap_ref])[None]  # [1, 288, 3840], basic mode [image, caption]
    u_cos = np.concatenate([x_cos, c_cos])[None, None]
    u_sin = np.concatenate([x_sin, c_sin])[None, None]
    for k in range(6):
        u = g.run(f"zc_main{k}", u, u_cos, u_sin, adaln)
    out = g.run("zc_final", u, adaln)[0]  # [288, 64]
    if dump is not None:
        dump.update(adaln=adaln[0], x_ref=x, final=out)
    return zm.unpatchify(out[: zm.IMG_TOKENS])


def generate(prompt: str, seed: int, steps: int, g: Graphs, host, dump: dict | None = None) -> np.ndarray:
    cap_ref = encode_prompt(g, host, prompt, dump)
    latent = zm.gaussian_noise(seed, zm.LATENT_C * zm.LATENT_HW * zm.LATENT_HW).reshape(
        zm.LATENT_C, zm.LATENT_HW, zm.LATENT_HW
    )
    if dump is not None:
        dump["noise"] = latent.copy()
    sig = zm.sigmas(steps)
    for i in range(steps):
        print(f"step {i + 1}/{steps} sigma {sig[i]:.4f}", flush=True)
        step_dump = {} if (dump is not None and i == 0) else None
        v = dit(g, host, latent, cap_ref, zm.model_t(sig[i]), step_dump)
        latent = (latent + (sig[i + 1] - sig[i]) * -v).astype(np.float32)  # noise_pred = -model_out
        if step_dump is not None:
            dump.update({f"step0_{k}": val for k, val in step_dump.items()})
            dump["step0_latent"] = latent.copy()
    if dump is not None:
        dump["final_latent"] = latent.copy()
    img = g.run("zvae", zm.vae_input(latent)[None])[0]
    return zm.to_rgb8(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--keep", action="store_true", help="keep DiT graphs compiled between steps (needs RAM)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--dump", action="store_true", help="save intermediate tensors as golden fixtures")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    g = Graphs(GRAPH_DIR, a.threads, a.keep)
    host = load_host(HOST_DIR)
    dump = {} if a.dump else None
    t0 = time.time()
    rgb = generate(a.prompt, a.seed, a.steps, g, host, dump)
    print(f"total {time.time() - t0:.1f}s")
    for name, ts in g.timings.items():
        print(f"  {name:9s} x{len(ts):3d}  mean {np.mean(ts):6.2f}s")
    path = os.path.join(a.out, f"ref_seed{a.seed}_steps{a.steps}.png")
    Image.fromarray(rgb).save(path)
    print("saved", path)
    if dump is not None:
        np.savez_compressed(os.path.join(a.out, f"golden_seed{a.seed}_steps{a.steps}.npz"), prompt=a.prompt, **dump)


if __name__ == "__main__":
    main()
