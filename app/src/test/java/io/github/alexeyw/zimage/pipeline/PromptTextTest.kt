package io.github.alexeyw.zimage.pipeline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class PromptTextTest {
    @Test
    fun plainTextIsReturnedAsIs() {
        val s = "a red fox in fresh snow, watercolor \u2014 \u041A\u043E\u0442 \u6771\u4EAC \uD83C\uDF4E"
        assertSame(s, PromptText.clean(s))
    }

    @Test
    fun zeroWidthAndSoftHyphenAreDroppedEverywhere() {
        assertEquals("Vintage astronaut", PromptText.clean("\u200BVintage astro\u00ADnaut\u2060"))
        assertEquals("helmet", PromptText.clean("\uFEFFhel\u200Bmet"))
    }

    @Test
    fun joinersStayInsideEmojiAndWordsButNotAtEdges() {
        val family = "\uD83D\uDC68\u200D\uD83D\uDC69\u200D\uD83D\uDC67" // man ZWJ woman ZWJ girl
        assertEquals("a $family photo", PromptText.clean("a $family photo"))
        val persian = "\u0645\u06CC\u200C\u062E\u0648\u0627\u0647\u0645" // "mi-khaham", ZWNJ after the prefix
        assertEquals(persian, PromptText.clean(persian))
        assertEquals("cat dog", PromptText.clean("\u200Dcat \u200Cdog\u200D"))
    }

    @Test
    fun noBreakSpacesBecomeSpaces() {
        assertEquals("35 mm photo", PromptText.clean("35\u00A0mm\u202Fphoto"))
    }

    /** The prompt seen on the test phone: the leading zero width space cost a whole token. */
    @Test
    fun cleaningSavesTheTokenTheInvisibleCharacterCost() {
        val vocab = File(Golden.hostDir, "vocab.json")
        assumeTrue("needs models/host (reference/host_assets.py)", vocab.exists())
        val tok = QwenTokenizer(vocab.inputStream(), File(Golden.hostDir, "merges.txt").inputStream())
        val raw = "\u200BVintage astronaut helmet overgrown with red bioluminescent mushrooms, cinematic 35mm photograph"
        assertEquals(28, tok.encodePrompt(raw).size)
        assertEquals(27, tok.encodePrompt(PromptText.clean(raw)).size)
    }
}
