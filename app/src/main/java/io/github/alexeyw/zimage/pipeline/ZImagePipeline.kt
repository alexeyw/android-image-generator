package io.github.alexeyw.zimage.pipeline

import io.github.alexeyw.zimage.pipeline.ZImageMath as M
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File

/**
 * Z-Image-Turbo over the 13 published LiteRT graphs; the Kotlin twin of `reference/zimage_ref.py`.
 *
 *   qwen_enc                                   once per prompt (cached across generations)
 *   z_embc -> [host pad token] -> z_refc       once per prompt (the context refiner has no adaln)
 *   per step: z_embx -> z_refx -> zc_main0..5 -> zc_final -> [host unpatchify + Euler]
 *   zvae                                       once
 *
 * Turbo is distilled for guidance 0, so each step is a single transformer pass (no CFG branch).
 */
class ZImagePipeline(modelDir: File, cacheDir: File, policy: RuntimePolicy) : AutoCloseable {
    private val hostDir = File(modelDir, "host")
    private val runner = GraphRunner(File(modelDir, "litert"), cacheDir, policy)
    private val host by lazy { HostAssets.load(hostDir) }
    val tokenizer by lazy {
        QwenTokenizer(File(hostDir, "vocab.json").inputStream(), File(hostDir, "merges.txt").inputStream())
    }

    private val imageRope = M.ropeCosSin(M.imagePosIds())
    private val captionRope = M.ropeCosSin(M.captionPosIds())
    private val unifiedCos = imageRope.cos + captionRope.cos
    private val unifiedSin = imageRope.sin + captionRope.sin

    private var cachedPrompt: String? = null
    private var cachedCaption: FloatArray? = null

    class Progress(val stage: String, val fraction: Float)

    class Result(val argb: IntArray, val seconds: Double, val timings: Map<String, List<Pair<Long, Long>>>)

    /** Prompt length in tokens with the chat template; the graphs take at most [M.CAP_LEN]. */
    fun tokenCount(prompt: String): Int = tokenizer.encodePrompt(prompt).size

    suspend fun generate(prompt: String, seed: Long, steps: Int, onProgress: (Progress) -> Unit): Result {
        val start = System.nanoTime()
        runner.timings.clear()
        val needsText = prompt != cachedPrompt
        val units = (if (needsText) 1f else 0f) + steps + 1f
        var done = 0f

        val caption = if (needsText) {
            onProgress(Progress("Encoding prompt", 0f))
            runner.releaseAll() // make room for the 3.5 GB text encoder
            encodePrompt(prompt).also {
                cachedPrompt = prompt
                cachedCaption = it
                done += 1f
            }
        } else {
            cachedCaption!!
        }

        val latent = M.gaussianNoise(seed)
        val sigmas = M.sigmas(steps)
        for (i in 0 until steps) {
            currentCoroutineContext().ensureActive()
            onProgress(Progress("Step ${i + 1} / $steps", done / units))
            val v = transformer(latent, caption, M.modelT(sigmas[i]))
            M.eulerStep(latent, v, sigmas[i], sigmas[i + 1])
            done += 1f
        }

        currentCoroutineContext().ensureActive()
        onProgress(Progress("Decoding image", done / units))
        val rgb = runner.run(GraphRunner.VAE, M.vaeInput(latent))
        onProgress(Progress("Done", 1f))
        return Result(M.toArgb(rgb), (System.nanoTime() - start) / 1e9, runner.timings.mapValues { it.value.toList() })
    }

    /** -> refined caption tokens [CAP_LEN, DIM], timestep-independent. */
    private suspend fun encodePrompt(prompt: String): FloatArray {
        val ids = tokenizer.encodePrompt(prompt)
        val n = ids.size
        require(n <= M.CAP_LEN) { "Prompt is $n tokens with the chat template; the 256 px graphs take ${M.CAP_LEN}." }
        val padded = IntArray(M.TEXT_LEN) { if (it < n) ids[it] else QwenTokenizer.PAD_ID }
        // Causal graph without a mask input: the padding cannot reach the prompt rows (verified: diff 0).
        val hidden = runner.run(GraphRunner.TEXT_ENCODER, host.embedTokens(padded))
        currentCoroutineContext().ensureActive()

        // Pad the caption to CAP_LEN by repeating the last valid row (diffusers `_pad_with_ids`).
        val d = M.TEXT_DIM
        val cap = FloatArray(M.CAP_LEN * d)
        for (r in 0 until M.CAP_LEN) System.arraycopy(hidden, minOf(r, n - 1) * d, cap, r * d, d)
        val emb = runner.run("z_embc", cap)
        // torch.where(pad_mask, cap_pad_token, feats): the pad slots get the learned token.
        for (r in n until M.CAP_LEN) System.arraycopy(host.capPadToken, 0, emb, r * M.DIM, M.DIM)
        return runner.run("z_refc", emb, captionRope.cos, captionRope.sin)
    }

    /** One transformer forward -> model output in latent layout [16, 32, 32]. */
    private suspend fun transformer(latent: FloatArray, caption: FloatArray, t: Float): FloatArray {
        val adaln = M.adalnInput(t, host.timestep)
        var x = runner.run("z_embx", M.patchify(latent))
        x = runner.run("z_refx", x, imageRope.cos, imageRope.sin, adaln)
        var u = x + caption // [288, 3840], basic-mode order [image, caption]
        for (k in 0 until 6) {
            currentCoroutineContext().ensureActive()
            u = runner.run("zc_main$k", u, unifiedCos, unifiedSin, adaln)
        }
        return M.unpatchify(runner.run("zc_final", u, adaln)) // first 256 rows are the image
    }

    override fun close() = runner.close()
}
