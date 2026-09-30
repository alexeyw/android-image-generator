package io.github.alexeyw.zimage.pipeline

import io.github.alexeyw.zimage.pipeline.Golden.assertClose
import io.github.alexeyw.zimage.pipeline.Golden.floats
import io.github.alexeyw.zimage.pipeline.ZImageMath as M
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class ZImageMathTest {
    @Test
    fun sigmasMatchDiffusersSchedule() {
        val all = Golden.math.getJSONObject("sigmas")
        for (n in listOf(4, 8, 9)) {
            assertArrayEquals("steps=$n", floats(all.getJSONArray("$n")), M.sigmas(n), 0f)
        }
    }

    @Test
    fun modelTimestepIsOneMinusSigma() {
        val expected = floats(Golden.math.getJSONArray("model_t"))
        val sig = M.sigmas(8)
        assertArrayEquals(expected, FloatArray(8) { M.modelT(sig[it]) }, 0f)
    }

    @Test
    fun timestepFrequencies() {
        assertClose(floats(Golden.math.getJSONArray("timestep_freq_t500")), M.timestepFreq(500f), atol = 2e-6f)
    }

    @Test
    fun adalnInputFromUpstreamWeights() {
        assumeTrue("needs models/host (reference/host_assets.py)", File(Golden.hostDir, "t_emb_w1.f32").exists())
        val w = HostAssets.loadTimestepWeights(Golden.hostDir)
        val all = Golden.math.getJSONObject("adaln")
        for (t in listOf("0.0", "0.3", "0.7")) {
            val expected = floats(all.getJSONArray(t))
            assertClose(expected, M.adalnInput(t.toFloat(), w), atol = 2e-4f, rtol = 1e-4f, what = "adaln t=$t")
        }
    }

    @Test
    fun ropeCaption() {
        val r = M.ropeCosSin(M.captionPosIds())
        assertClose(floats(Golden.math.getJSONArray("rope_cap_cos")), r.cos, atol = 1e-6f, what = "cos")
        assertClose(floats(Golden.math.getJSONArray("rope_cap_sin")), r.sin, atol = 1e-6f, what = "sin")
    }

    @Test
    fun ropeImageRows() {
        val rows = Golden.ints(Golden.math.getJSONArray("rope_img_rows"))
        val r = M.ropeCosSin(M.imagePosIds())
        val cos = rows.flatMap { row -> (0 until M.ROPE_HALF).map { r.cos[row * M.ROPE_HALF + it] } }.toFloatArray()
        val sin = rows.flatMap { row -> (0 until M.ROPE_HALF).map { r.sin[row * M.ROPE_HALF + it] } }.toFloatArray()
        assertClose(floats(Golden.math.getJSONArray("rope_img_cos")), cos, atol = 1e-6f, what = "cos")
        assertClose(floats(Golden.math.getJSONArray("rope_img_sin")), sin, atol = 1e-6f, what = "sin")
    }

    @Test
    fun splitMixMatchesPythonBitForBit() {
        val expected = Golden.math.getJSONArray("splitmix_seed42")
        val rng = M.SplitMix64(42)
        for (i in 0 until expected.length()) {
            assertEquals(expected.getString(i), java.lang.Long.toUnsignedString(rng.nextLong()))
        }
    }

    @Test
    fun gaussianNoiseMatchesPython() {
        assertClose(floats(Golden.math.getJSONArray("noise_seed42_head")), M.gaussianNoise(42, 64), atol = 1e-6f)
        assertEquals(Golden.math.getDouble("noise_seed7_sum"), M.gaussianNoise(7).sumOf { it.toDouble() }, 1e-3)
        assertClose(Golden.f32("noise"), M.gaussianNoise(42), atol = 1e-6f, what = "run noise")
    }

    @Test
    fun patchifyRoundTrip() {
        val lat = M.gaussianNoise(3)
        assertArrayEquals(lat, M.unpatchify(M.patchify(lat)), 0f)
    }

    /** Recorded run: noise + the step-0 graph output -> the recorded step-0 latent. */
    @Test
    fun firstEulerStepMatchesRecordedRun() {
        val latent = Golden.f32("noise")
        val out = M.unpatchify(Golden.f32("step0_final"))
        val sig = M.sigmas(8)
        M.eulerStep(latent, out, sig[0], sig[1])
        assertClose(Golden.f32("step0_latent"), latent, atol = 1e-6f, what = "latent after step 0")
    }

    @Test
    fun vaeBoundaryAndPixels() {
        val z = M.vaeInput(floatArrayOf(0f, 0.3611f))
        assertEquals(0.1159f, z[0], 0f)
        assertEquals(1.1159f, z[1], 1e-6f)
        val px = M.toArgb(floatArrayOf(-1f, 0f, 1f), px = 1)
        assertEquals(0xFF0080FF.toInt(), px[0])
    }
}
