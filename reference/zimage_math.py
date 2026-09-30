"""Host-side math of the Z-Image-Turbo LiteRT pipeline (numpy, no torch).

Everything here is ported 1:1 to Kotlin (`ZImageMath.kt`) and pinned by golden fixtures, so the
phone never debugs math — only the runtime. Source of truth for each function is diffusers'
`transformer_z_image.py` / `pipeline_z_image.py`; the line references below point there.
"""

import numpy as np

# --- fixed geometry of the published 256 px graphs -------------------------------------------
TEXT_LEN = 64  # qwen_enc sequence length
CAP_LEN = 32  # z_embc / z_refc caption slot (SEQ_MULTI_OF = 32)
LATENT_C = 16
LATENT_HW = 32  # 256 px / VAE factor 8
PATCH = 2
GRID = LATENT_HW // PATCH  # 16 x 16 patches
IMG_TOKENS = GRID * GRID  # 256
UNIFIED = IMG_TOKENS + CAP_LEN  # 288, basic-mode order is [image, caption]
DIM = 3840
TEXT_DIM = 2560
ADALN_DIM = 256
ROPE_AXES = (32, 48, 48)  # head_dim 128 split over (t, h, w)
ROPE_THETA = 256.0
T_SCALE = 1000.0
SCHED_SHIFT = 3.0
VAE_SCALING = 0.3611
VAE_SHIFT = 0.1159

PAD_TOKEN_ID = 151643  # <|endoftext|>


def chat_prompt(prompt: str) -> str:
    """Qwen3 chat template, add_generation_prompt=True, enable_thinking=True."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# --- scheduler ---------------------------------------------------------------------------------
def sigmas(num_steps: int) -> np.ndarray:
    """FlowMatchEulerDiscrete with shift=3, no dynamic shifting; returns num_steps + 1 values."""
    s = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float64).astype(np.float32)
    s = (SCHED_SHIFT * s / (1.0 + (SCHED_SHIFT - 1.0) * s)).astype(np.float32)
    return np.concatenate([s, np.zeros(1, np.float32)])


def model_t(sigma: np.float32) -> np.float32:
    """Pipeline passes (1000 - sigma * 1000) / 1000 to the transformer."""
    ts = np.float32(sigma) * np.float32(1000.0)
    return np.float32((np.float32(1000.0) - ts) / np.float32(1000.0))


# --- timestep embedder (runs on the host: 0.5 M params) ----------------------------------------
def timestep_freq(t: float, dim: int = 256, max_period: float = 10000.0) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
    args = np.float32(t) * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)


def adaln_input(t: float, w1, b1, w2, b2) -> np.ndarray:
    """t_embedder(t * t_scale) -> [256]. Weights in PyTorch Linear layout [out, in]."""
    x = timestep_freq(np.float32(t) * np.float32(T_SCALE))
    h = w1 @ x + b1
    h = h / (1.0 + np.exp(-h))  # SiLU
    return (w2 @ h + b2).astype(np.float32)


# --- RoPE ---------------------------------------------------------------------------------------
def rope_cos_sin(pos_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pos_ids [S, 3] -> (cos, sin), each [S, 64]; concat over axes (t:16, h:24, w:24)."""
    cos, sin = [], []
    for axis, d in enumerate(ROPE_AXES):
        freqs = 1.0 / (ROPE_THETA ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = np.outer(pos_ids[:, axis].astype(np.float64), freqs).astype(np.float32)
        cos.append(np.cos(ang))
        sin.append(np.sin(ang))
    return np.concatenate(cos, 1).astype(np.float32), np.concatenate(sin, 1).astype(np.float32)


def caption_pos_ids() -> np.ndarray:
    """Caption padded to CAP_LEN: positions (1..CAP_LEN, 0, 0) — pad slots keep counting."""
    ids = np.zeros((CAP_LEN, 3), np.int64)
    ids[:, 0] = np.arange(1, CAP_LEN + 1)
    return ids


def image_pos_ids() -> np.ndarray:
    """Image patches: (CAP_LEN + 1, h, w), row-major over (h, w)."""
    h, w = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    ids = np.zeros((IMG_TOKENS, 3), np.int64)
    ids[:, 0] = CAP_LEN + 1
    ids[:, 1] = h.reshape(-1)
    ids[:, 2] = w.reshape(-1)
    return ids


# --- (un)patchify -------------------------------------------------------------------------------
def patchify(latent: np.ndarray) -> np.ndarray:
    """[16, 32, 32] -> [256, 64]; patch vector index = (ph * 2 + pw) * 16 + c."""
    x = latent.reshape(LATENT_C, GRID, PATCH, GRID, PATCH)  # c, ht, ph, wt, pw
    return x.transpose(1, 3, 2, 4, 0).reshape(IMG_TOKENS, PATCH * PATCH * LATENT_C)


def unpatchify(tokens: np.ndarray) -> np.ndarray:
    """[256, 64] -> [16, 32, 32]; exact inverse of patchify."""
    x = tokens.reshape(GRID, GRID, PATCH, PATCH, LATENT_C)  # ht, wt, ph, pw, c
    return x.transpose(4, 0, 2, 1, 3).reshape(LATENT_C, LATENT_HW, LATENT_HW)


# --- portable seeded noise (bit-identical in Kotlin) --------------------------------------------
M64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & M64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def next_double(self) -> float:
        """Uniform in [0, 1) with 53 bits."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


def gaussian_noise(seed: int, n: int) -> np.ndarray:
    """Box-Muller over SplitMix64, computed in float64, stored as float32."""
    rng = SplitMix64(seed)
    out = np.empty(n, np.float32)
    i = 0
    while i < n:
        u1 = 1.0 - rng.next_double()  # (0, 1]
        u2 = rng.next_double()
        r = np.sqrt(-2.0 * np.log(u1))
        out[i] = r * np.cos(2.0 * np.pi * u2)
        if i + 1 < n:
            out[i + 1] = r * np.sin(2.0 * np.pi * u2)
        i += 2
    return out


# --- misc -----------------------------------------------------------------------------------------
def bf16_rows_to_f32(table: np.ndarray, ids) -> np.ndarray:
    """table: uint16 view of the bf16 embedding matrix [vocab, dim]."""
    return (table[np.asarray(ids)].astype(np.uint32) << 16).view(np.float32)


def vae_input(latent: np.ndarray) -> np.ndarray:
    return (latent / np.float32(VAE_SCALING) + np.float32(VAE_SHIFT)).astype(np.float32)


def to_rgb8(image_chw: np.ndarray) -> np.ndarray:
    """VAE output in [-1, 1], [3, H, W] -> uint8 [H, W, 3] (VaeImageProcessor.postprocess)."""
    x = np.clip(image_chw / 2.0 + 0.5, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
