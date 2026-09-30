// Adapted from QwenTokenizer.kt of the Bonsai Image Android sample,
// https://github.com/john-rocky/hf-to-litertlm (bonsai_image_work/device/BonsaiAppAndroid),
// Copyright 2026 Daisuke Majima, licensed under the Apache License, Version 2.0.
// Changes: Z-Image chat template (enable_thinking=True), 32-token caption budget.

package io.github.alexeyw.zimage.pipeline

import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

/**
 * Qwen3 byte-level BPE (vocab.json + merges.txt of `Tongyi-MAI/Z-Image-Turbo/tokenizer`).
 *
 * The chat template is applied structurally: `[<|im_start|>] + BPE("user\n" + prompt) + suffix`,
 * which is what `apply_chat_template(add_generation_prompt=True, enable_thinking=True)` renders.
 * `"user\n"` is BPE'd together with the prompt because whitespace merges across that boundary.
 * Special-token strings typed inside a prompt are tokenized as plain text.
 */
class QwenTokenizer(vocabJson: InputStream, mergesTxt: InputStream) {
    companion object {
        const val PAD_ID = 151643 // <|endoftext|>
        const val IM_START_ID = 151644 // <|im_start|>

        /** `<|im_end|>\n<|im_start|>assistant\n` */
        val SUFFIX_IDS = intArrayOf(151645, 198, 151644, 77091, 198)

        /** Tokens the template itself costs, so the prompt budget is CAP_LEN minus this. */
        val TEMPLATE_TOKENS = 1 + 2 + SUFFIX_IDS.size // <|im_start|>, "user", "\n", suffix

        // Qwen2 pre-tokenization pattern, verbatim from tokenizer.json.
        private val PRETOKEN = Regex(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
                " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            setOf(RegexOption.UNIX_LINES),
        )

        // GPT-2 bytes-to-unicode: every byte maps to a distinct printable char.
        private val BYTE_CHAR: CharArray = run {
            val bs = ((33..126) + (161..172) + (174..255)).toMutableList()
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            val table = CharArray(256)
            for (i in bs.indices) table[bs[i]] = cs[i].toChar()
            table
        }
    }

    private val vocab = HashMap<String, Int>(160_000)
    private val ranks = HashMap<String, Int>(160_000)
    private val cache = HashMap<String, IntArray>()

    init {
        val jo = JSONObject(vocabJson.bufferedReader().readText())
        for (key in jo.keys()) vocab[key] = jo.getInt(key)
        mergesTxt.bufferedReader().forEachLine { line ->
            if (line.isNotEmpty() && !line.startsWith("#")) ranks[line] = ranks.size
        }
    }

    /** Full token sequence for one prompt, chat template included (not padded). */
    fun encodePrompt(prompt: String): IntArray =
        intArrayOf(IM_START_ID) + encode("user\n" + prompt) + SUFFIX_IDS

    /** Byte-level BPE of plain text (no special-token splitting), NFC-normalized like tokenizer.json. */
    fun encode(text: String): IntArray {
        val nfc = Normalizer.normalize(text, Normalizer.Form.NFC)
        val out = ArrayList<Int>(nfc.length / 3 + 8)
        for (m in PRETOKEN.findAll(nfc)) {
            if (m.value.isEmpty()) continue
            bpe(m.value).forEach { out.add(it) }
        }
        return out.toIntArray()
    }

    @Synchronized
    private fun bpe(pretoken: String): IntArray {
        cache[pretoken]?.let { return it }
        var word = pretoken.toByteArray(Charsets.UTF_8).map { BYTE_CHAR[it.toInt() and 0xFF].toString() }
        while (word.size > 1) {
            var best = Int.MAX_VALUE
            var at = -1
            for (i in 0 until word.size - 1) {
                val r = ranks[word[i] + " " + word[i + 1]] ?: continue
                if (r < best) {
                    best = r
                    at = i
                }
            }
            if (at < 0) break
            val a = word[at]
            val b = word[at + 1]
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == a && word[i + 1] == b) {
                    merged.add(a + b)
                    i += 2
                } else {
                    merged.add(word[i])
                    i += 1
                }
            }
            word = merged
        }
        // Byte-level alphabet: every symbol and merge result exists in the vocab.
        val ids = word.mapNotNull { vocab[it] }.toIntArray()
        cache[pretoken] = ids
        return ids
    }
}
