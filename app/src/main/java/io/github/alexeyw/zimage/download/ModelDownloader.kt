package io.github.alexeyw.zimage.download

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

/**
 * Fetches everything the pipeline needs straight from Hugging Face, nothing re-hosted:
 *  - the 13 LiteRT graphs of `litert-community/Z-Image-Turbo-LiteRT` (9.78 GB),
 *  - four host tensors and the BPE tables of the Apache-2.0 upstream `Tongyi-MAI/Z-Image-Turbo`;
 *    only the needed byte ranges of its multi-GB safetensors shards are read (0.78 GB).
 * Both revisions are pinned, and every file resumes from its `.part` after an interruption.
 * Mirrors `reference/host_assets.py`, so a Mac download can be `adb push`ed instead.
 */
class ModelDownloader(private val root: File) {
    class Progress(val file: String, val doneBytes: Long, val totalBytes: Long)

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    val graphDir = File(root, "litert")
    val hostDir = File(root, "host")

    /** Files still missing or incomplete, as paths relative to [root]. */
    fun missing(): List<String> = buildList {
        for ((name, size) in GRAPHS) if (File(graphDir, "$name.tflite").length() != size) add("litert/$name.tflite")
        for ((name, size) in HOST_FILES) if (File(hostDir, name).length() != size) add("host/$name")
    }

    fun totalBytes(): Long = GRAPHS.values.sum() + HOST_FILES.values.sum()

    fun presentBytes(): Long =
        GRAPHS.entries.sumOf { (n, s) -> File(graphDir, "$n.tflite").length().coerceAtMost(s) } +
            HOST_FILES.entries.sumOf { (n, s) -> File(hostDir, n).length().coerceAtMost(s) }

    suspend fun downloadAll(onProgress: (Progress) -> Unit) = withContext(Dispatchers.IO) {
        graphDir.mkdirs()
        hostDir.mkdirs()
        val total = totalBytes()
        var base = presentBytes()
        val report = { file: String, fileDone: Long -> onProgress(Progress(file, base + fileDone, total)) }

        for (name in listOf("vocab.json", "merges.txt")) {
            val dst = File(hostDir, name)
            if (dst.length() == HOST_FILES.getValue(name)) continue
            download("$UPSTREAM/tokenizer/$name", 0, HOST_FILES.getValue(name), dst) { report(name, it) }
            base += HOST_FILES.getValue(name)
        }

        // Small transformer tensors: F32 in the upstream shard, stored as-is.
        if (DIT_TENSORS.values.any { File(hostDir, it).length() != HOST_FILES.getValue(it) }) {
            val url = "$UPSTREAM/transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
            val (header, dataStart) = safetensorsHeader(url)
            for ((tensor, file) in DIT_TENSORS) {
                val meta = header.getJSONObject(tensor)
                check(meta.getString("dtype") == "F32") { "$tensor is ${meta.getString("dtype")}" }
                val off = meta.getJSONArray("data_offsets")
                download(url, dataStart + off.getLong(0), dataStart + off.getLong(1), File(hostDir, file)) {}
                base += HOST_FILES.getValue(file)
                report(file, 0)
            }
        }

        val embed = File(hostDir, EMBED_FILE)
        if (embed.length() != HOST_FILES.getValue(EMBED_FILE)) {
            val url = "$UPSTREAM/text_encoder/model-00001-of-00003.safetensors"
            val (header, dataStart) = safetensorsHeader(url)
            val meta = header.getJSONObject("model.embed_tokens.weight")
            check(meta.getString("dtype") == "BF16") { "embed_tokens is ${meta.getString("dtype")}" }
            val off = meta.getJSONArray("data_offsets")
            download(url, dataStart + off.getLong(0), dataStart + off.getLong(1), embed) { report(EMBED_FILE, it) }
            base += HOST_FILES.getValue(EMBED_FILE)
        }

        for ((name, size) in GRAPHS) {
            val dst = File(graphDir, "$name.tflite")
            if (dst.length() == size) continue
            download("$LITERT/$name.tflite", 0, size, dst) { report("$name.tflite", it) }
            base += size
        }
    }

    /** Downloads bytes [start, end) of [url] into [dst], resuming from `dst.part`. */
    private suspend fun download(url: String, start: Long, end: Long, dst: File, onBytes: (Long) -> Unit) {
        val part = File(dst.path + ".part")
        val want = end - start
        var have = part.length().coerceAtMost(want)
        if (have < want) {
            val req = Request.Builder().url(url).header("Range", "bytes=${start + have}-${end - 1}").build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("HTTP ${resp.code} for $url")
                // Small non-LFS files come back whole (200) regardless of Range: restart them.
                val append = resp.code == 206
                if (!append) {
                    check(start == 0L) { "server ignored a mid-file Range for $url" }
                    have = 0
                }
                FileOutputStream(part, append).use { out ->
                    val src = resp.body.byteStream()
                    val buf = ByteArray(1 shl 20)
                    var lastReport = 0L
                    while (have < want) {
                        currentCoroutineContext().ensureActive()
                        val r = src.read(buf, 0, minOf(buf.size.toLong(), want - have).toInt())
                        if (r < 0) break
                        out.write(buf, 0, r)
                        have += r
                        if (have - lastReport >= (8 shl 20)) {
                            lastReport = have
                            onBytes(have)
                        }
                    }
                }
            }
        }
        if (part.length() != want) throw IOException("${dst.name}: got ${part.length()} of $want bytes")
        if (!part.renameTo(dst)) throw IOException("cannot move $part to $dst")
        onBytes(want)
    }

    /** -> (header JSON, absolute offset of the data section). */
    private fun safetensorsHeader(url: String): Pair<JSONObject, Long> {
        val n = ByteBuffer.wrap(range(url, 0, 8)).order(ByteOrder.LITTLE_ENDIAN).long
        return JSONObject(String(range(url, 8, 8 + n), Charsets.UTF_8)) to 8 + n
    }

    private fun range(url: String, start: Long, end: Long): ByteArray {
        val req = Request.Builder().url(url).header("Range", "bytes=$start-${end - 1}").build()
        client.newCall(req).execute().use { resp ->
            if (resp.code != 206) throw IOException("HTTP ${resp.code} for a Range of $url")
            return resp.body.bytes().also { check(it.size.toLong() == end - start) }
        }
    }

    companion object {
        const val LITERT_REVISION = "09ea3ae2ef44d04d0ad1591de36d1c13504e1521"
        const val UPSTREAM_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
        private const val LITERT = "https://huggingface.co/litert-community/Z-Image-Turbo-LiteRT/resolve/$LITERT_REVISION"
        private const val UPSTREAM = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/$UPSTREAM_REVISION"
        private const val EMBED_FILE = "embed_tokens.bf16"

        /** Graph name -> size in bytes at [LITERT_REVISION]. */
        val GRAPHS: Map<String, Long> = linkedMapOf(
            "z_embx" to 308272L,
            "z_embc" to 9906096L,
            "zc_final" to 1295936L,
            "zvae" to 50139872L,
            "z_refc" to 355025040L,
            "z_refx" to 363386160L,
            "zc_main0" to 908480960L,
            "zc_main1" to 908480960L,
            "zc_main2" to 908480960L,
            "zc_main3" to 908480960L,
            "zc_main4" to 908480960L,
            "zc_main5" to 908480960L,
            "qwen_enc" to 3547652208L,
        )

        private val DIT_TENSORS = linkedMapOf(
            "t_embedder.mlp.0.weight" to "t_emb_w1.f32",
            "t_embedder.mlp.0.bias" to "t_emb_b1.f32",
            "t_embedder.mlp.2.weight" to "t_emb_w2.f32",
            "t_embedder.mlp.2.bias" to "t_emb_b2.f32",
            "cap_pad_token" to "cap_pad_token.f32",
        )

        private val HOST_FILES: Map<String, Long> = linkedMapOf(
            "vocab.json" to 2776833L,
            "merges.txt" to 1671853L,
            "t_emb_w1.f32" to 1048576L,
            "t_emb_b1.f32" to 4096L,
            "t_emb_w2.f32" to 1048576L,
            "t_emb_b2.f32" to 1024L,
            "cap_pad_token.f32" to 15360L,
            EMBED_FILE to 777912320L,
        )
    }
}
