package io.github.alexeyw.zimage.pipeline

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/** Token-exact against the Python `tokenizers` output with the Z-Image chat template. */
class QwenTokenizerTest {
    companion object {
        private lateinit var tok: QwenTokenizer

        @JvmStatic
        @BeforeClass
        fun load() {
            val vocab = File(Golden.hostDir, "vocab.json")
            assumeTrue("needs models/host (reference/host_assets.py)", vocab.exists())
            tok = QwenTokenizer(vocab.inputStream(), File(Golden.hostDir, "merges.txt").inputStream())
        }
    }

    @Test
    fun goldenCases() {
        val cases = Golden.math.getJSONArray("tokenizer")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val prompt = c.getString("prompt")
            assertArrayEquals("prompt ${prompt.quoted()}", Golden.ints(c.getJSONArray("ids")), tok.encodePrompt(prompt))
        }
    }

    @Test
    fun recordedRunPrompt() {
        val expected = Golden.ints(Golden.run.getJSONArray("token_ids"))
        assertArrayEquals(expected, tok.encodePrompt(Golden.run.getString("prompt")))
    }

    @Test
    fun templateCost() {
        assertEquals(QwenTokenizer.TEMPLATE_TOKENS, tok.encodePrompt("").size)
    }

    private fun String.quoted() = "\"" + replace("\n", "\\n").replace("\t", "\\t") + "\""
}
