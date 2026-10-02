package io.github.alexeyw.zimage.pipeline

/**
 * Removes characters that are invisible in the prompt field but still cost tokens of the 32-token
 * caption budget; text copied from web pages and messengers carries them routinely.
 *
 * The characters are written as escapes on purpose: literal invisible characters in source are
 * themselves invisible in review and may be normalised away by an editor.
 */
object PromptText {
    // Zero width space, word joiner, zero width no-break space (BOM), soft hyphen: never meaningful here.
    private val ALWAYS_DROP = setOf('\u200B', '\u2060', '\uFEFF', '\u00AD')

    // Zero width joiner / non-joiner: they glue emoji sequences (man + ZWJ + woman = family) and
    // shape Persian and Indic words, so they stay between two visible characters and go only at
    // word edges.
    private val JOINERS = setOf('\u200D', '\u200C')

    // No-break space, figure space, narrow no-break space: shown as a space, tokenized as other bytes.
    private val SPACES = setOf('\u00A0', '\u2007', '\u202F')

    fun clean(raw: String): String {
        if (raw.none { it in ALWAYS_DROP || it in JOINERS || it in SPACES }) return raw
        val s = raw.filterNot { it in ALWAYS_DROP }
        val out = StringBuilder(s.length)
        for (i in s.indices) {
            val c = s[i]
            when {
                c in SPACES -> out.append(' ')
                c in JOINERS -> {
                    val inside = i > 0 && i < s.lastIndex && !s[i - 1].isWhitespace() && !s[i + 1].isWhitespace()
                    if (inside) out.append(c)
                }
                else -> out.append(c)
            }
        }
        return out.toString()
    }
}
