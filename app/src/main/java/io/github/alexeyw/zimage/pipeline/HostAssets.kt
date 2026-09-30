package io.github.alexeyw.zimage.pipeline

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * Host-side tensors no LiteRT graph contains (see `reference/host_assets.py` for provenance):
 * the Qwen3 token embedding table, the timestep-embedder MLP and the caption pad token.
 */
class HostAssets private constructor(
    private val embed: ShortBuffer,
    val timestep: ZImageMath.TimestepWeights,
    val capPadToken: FloatArray,
) {
    /** bf16 rows of `embed_tokens` for [ids] -> float32 [ids.size, 2560]. */
    fun embedTokens(ids: IntArray): FloatArray {
        val d = ZImageMath.TEXT_DIM
        val out = FloatArray(ids.size * d)
        for ((row, id) in ids.withIndex()) {
            val base = id * d
            for (k in 0 until d) out[row * d + k] = Float.fromBits((embed.get(base + k).toInt() and 0xFFFF) shl 16)
        }
        return out
    }

    companion object {
        const val VOCAB = 151936
        const val EMBED_FILE = "embed_tokens.bf16"

        fun load(dir: File): HostAssets {
            val file = File(dir, EMBED_FILE)
            val expected = VOCAB.toLong() * ZImageMath.TEXT_DIM * 2
            require(file.length() == expected) { "$file is ${file.length()} bytes, expected $expected" }
            // Memory-mapped: 778 MB stay in the page cache, only the prompt's rows are touched.
            val map = RandomAccessFile(file, "r").use { raf ->
                raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, expected)
            }
            val capPad = readF32(File(dir, "cap_pad_token.f32"), ZImageMath.DIM)
            return HostAssets(map.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer(), loadTimestepWeights(dir), capPad)
        }

        fun loadTimestepWeights(dir: File): ZImageMath.TimestepWeights {
            val a = ZImageMath.ADALN_DIM
            val m = ZImageMath.T_EMB_MID
            return ZImageMath.TimestepWeights(
                w1 = readF32(File(dir, "t_emb_w1.f32"), m * a),
                b1 = readF32(File(dir, "t_emb_b1.f32"), m),
                w2 = readF32(File(dir, "t_emb_w2.f32"), a * m),
                b2 = readF32(File(dir, "t_emb_b2.f32"), a),
            )
        }

        private fun readF32(file: File, count: Int): FloatArray {
            val bytes = file.readBytes()
            require(bytes.size == count * 4) { "$file is ${bytes.size} bytes, expected ${count * 4}" }
            val fb = java.nio.ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(count).also { fb.get(it) }
        }
    }
}
