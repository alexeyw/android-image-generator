package io.github.alexeyw.zimage.pipeline

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Host-side math of the Z-Image-Turbo pipeline: a 1:1 port of `reference/zimage_math.py`,
 * pinned by the JVM golden tests so the phone never debugs math, only the runtime.
 * Float32 where the reference computes in float32, double where it computes in float64.
 */
object ZImageMath {
    // Fixed geometry of the published 256 px graphs.
    const val TEXT_LEN = 64 // qwen_enc sequence length
    const val CAP_LEN = 32 // caption slot of z_embc / z_refc (SEQ_MULTI_OF)
    const val LATENT_C = 16
    const val LATENT_HW = 32 // 256 px / VAE factor 8
    const val PATCH = 2
    const val GRID = LATENT_HW / PATCH // 16 x 16 patches
    const val IMG_TOKENS = GRID * GRID // 256
    const val UNIFIED = IMG_TOKENS + CAP_LEN // 288, basic-mode order [image, caption]
    const val PATCH_DIM = PATCH * PATCH * LATENT_C // 64
    const val LATENT_SIZE = LATENT_C * LATENT_HW * LATENT_HW
    const val DIM = 3840
    const val TEXT_DIM = 2560
    const val ADALN_DIM = 256
    const val T_EMB_MID = 1024
    const val ROPE_HALF = 64 // head_dim 128 / 2 (cos/sin width)
    const val IMAGE_PX = 256

    private val ROPE_AXES = intArrayOf(32, 48, 48) // (t, h, w)
    private const val ROPE_THETA = 256.0
    private const val T_SCALE = 1000f
    private const val SCHED_SHIFT = 3f
    private const val VAE_SCALING = 0.3611f
    private const val VAE_SHIFT = 0.1159f

    // --- scheduler ------------------------------------------------------------------------------

    /** FlowMatchEuler, shift 3, no dynamic shifting: `numSteps + 1` sigmas ending in 0. */
    fun sigmas(numSteps: Int): FloatArray {
        val out = FloatArray(numSteps + 1)
        val stop = 1.0 / numSteps
        val step = if (numSteps > 1) (stop - 1.0) / (numSteps - 1) else 0.0
        for (i in 0 until numSteps) {
            val s = (if (i == numSteps - 1) stop else 1.0 + i * step).toFloat() // np.linspace
            out[i] = SCHED_SHIFT * s / (1f + (SCHED_SHIFT - 1f) * s)
        }
        return out
    }

    /** The pipeline hands the transformer (1000 - sigma * 1000) / 1000. */
    fun modelT(sigma: Float): Float {
        val ts = sigma * 1000f
        return (1000f - ts) / 1000f
    }

    // --- timestep embedder (host MLP, 0.5 M params) -----------------------------------------------

    fun timestepFreq(t: Float, dim: Int = ADALN_DIM): FloatArray {
        val half = dim / 2
        val out = FloatArray(dim)
        for (i in 0 until half) {
            val freq = exp(-ln(10000.0) * i / half).toFloat()
            val arg = t * freq
            out[i] = cos(arg.toDouble()).toFloat()
            out[half + i] = sin(arg.toDouble()).toFloat()
        }
        return out
    }

    /** `t_embedder(t * t_scale)` -> [256]; weights in PyTorch Linear layout [out, in]. */
    fun adalnInput(t: Float, w: TimestepWeights): FloatArray {
        val x = timestepFreq(t * T_SCALE)
        val h = FloatArray(T_EMB_MID)
        for (o in 0 until T_EMB_MID) {
            var acc = w.b1[o].toDouble()
            val row = o * ADALN_DIM
            for (i in 0 until ADALN_DIM) acc += w.w1[row + i].toDouble() * x[i]
            val v = acc.toFloat()
            h[o] = v / (1f + exp(-v)) // SiLU
        }
        val out = FloatArray(ADALN_DIM)
        for (o in 0 until ADALN_DIM) {
            var acc = w.b2[o].toDouble()
            val row = o * T_EMB_MID
            for (i in 0 until T_EMB_MID) acc += w.w2[row + i].toDouble() * h[i]
            out[o] = acc.toFloat()
        }
        return out
    }

    class TimestepWeights(val w1: FloatArray, val b1: FloatArray, val w2: FloatArray, val b2: FloatArray)

    // --- RoPE -----------------------------------------------------------------------------------

    class Rope(val cos: FloatArray, val sin: FloatArray) // each [S, 64]

    /** posIds: S rows of (t, h, w). */
    fun ropeCosSin(posIds: Array<IntArray>): Rope {
        val s = posIds.size
        val cosOut = FloatArray(s * ROPE_HALF)
        val sinOut = FloatArray(s * ROPE_HALF)
        var col0 = 0
        for ((axis, d) in ROPE_AXES.withIndex()) {
            val freqs = DoubleArray(d / 2) { k -> 1.0 / ROPE_THETA.pow((2.0 * k) / d) }
            for (r in 0 until s) {
                val pos = posIds[r][axis].toDouble()
                for (k in freqs.indices) {
                    val ang = (pos * freqs[k]).toFloat().toDouble()
                    cosOut[r * ROPE_HALF + col0 + k] = cos(ang).toFloat()
                    sinOut[r * ROPE_HALF + col0 + k] = sin(ang).toFloat()
                }
            }
            col0 += d / 2
        }
        return Rope(cosOut, sinOut)
    }

    /** Caption padded to CAP_LEN: (1..CAP_LEN, 0, 0) — pad slots keep counting. */
    fun captionPosIds(): Array<IntArray> = Array(CAP_LEN) { i -> intArrayOf(i + 1, 0, 0) }

    /** Image patches: (CAP_LEN + 1, h, w), row-major over (h, w). */
    fun imagePosIds(): Array<IntArray> = Array(IMG_TOKENS) { i -> intArrayOf(CAP_LEN + 1, i / GRID, i % GRID) }

    // --- (un)patchify -----------------------------------------------------------------------------

    /** [16, 32, 32] -> [256, 64]; patch vector index = (ph * 2 + pw) * 16 + c. */
    fun patchify(latent: FloatArray): FloatArray {
        val out = FloatArray(IMG_TOKENS * PATCH_DIM)
        for (ht in 0 until GRID) for (wt in 0 until GRID) for (ph in 0 until PATCH) for (pw in 0 until PATCH) {
            val token = ht * GRID + wt
            val y = ht * PATCH + ph
            val x = wt * PATCH + pw
            for (c in 0 until LATENT_C) {
                out[token * PATCH_DIM + (ph * PATCH + pw) * LATENT_C + c] = latent[(c * LATENT_HW + y) * LATENT_HW + x]
            }
        }
        return out
    }

    /** First IMG_TOKENS rows of [N, 64] -> [16, 32, 32]; exact inverse of [patchify]. */
    fun unpatchify(tokens: FloatArray): FloatArray {
        val out = FloatArray(LATENT_SIZE)
        for (ht in 0 until GRID) for (wt in 0 until GRID) for (ph in 0 until PATCH) for (pw in 0 until PATCH) {
            val token = ht * GRID + wt
            val y = ht * PATCH + ph
            val x = wt * PATCH + pw
            for (c in 0 until LATENT_C) {
                out[(c * LATENT_HW + y) * LATENT_HW + x] = tokens[token * PATCH_DIM + (ph * PATCH + pw) * LATENT_C + c]
            }
        }
        return out
    }

    /** Euler step of the flow: latent += (sigmaNext - sigma) * noisePred, noisePred = -modelOut. */
    fun eulerStep(latent: FloatArray, modelOut: FloatArray, sigma: Float, sigmaNext: Float) {
        val dt = sigmaNext - sigma
        for (i in latent.indices) latent[i] = latent[i] + dt * -modelOut[i]
    }

    // --- portable seeded noise (bit-identical with the Python reference) ---------------------------

    class SplitMix64(seed: Long) {
        private var state = seed

        fun nextLong(): Long {
            state += -0x61c8864680b583ebL // 0x9E3779B97F4A7C15
            var z = state
            z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L // 0xBF58476D1CE4E5B9
            z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L // 0x94D049BB133111EB
            return z xor (z ushr 31)
        }

        /** Uniform in [0, 1) with 53 bits. */
        fun nextDouble(): Double = (nextLong() ushr 11) * (1.0 / (1L shl 53))
    }

    fun gaussianNoise(seed: Long, n: Int = LATENT_SIZE): FloatArray {
        val rng = SplitMix64(seed)
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val u1 = 1.0 - rng.nextDouble() // (0, 1]
            val u2 = rng.nextDouble()
            val r = sqrt(-2.0 * ln(u1))
            out[i] = (r * cos(2.0 * PI * u2)).toFloat()
            if (i + 1 < n) out[i + 1] = (r * sin(2.0 * PI * u2)).toFloat()
            i += 2
        }
        return out
    }

    // --- VAE boundary -----------------------------------------------------------------------------

    fun vaeInput(latent: FloatArray): FloatArray = FloatArray(latent.size) { latent[it] / VAE_SCALING + VAE_SHIFT }

    /** VAE output in [-1, 1], planar [3, H, W] -> packed ARGB_8888 pixels. */
    fun toArgb(chw: FloatArray, px: Int = IMAGE_PX): IntArray {
        val plane = px * px
        fun u8(v: Float): Int = ((v / 2f + 0.5f).coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return IntArray(plane) { p ->
            (0xFF shl 24) or (u8(chw[p]) shl 16) or (u8(chw[plane + p]) shl 8) or u8(chw[2 * plane + p])
        }
    }
}
