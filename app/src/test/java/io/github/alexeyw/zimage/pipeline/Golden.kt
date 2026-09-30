package io.github.alexeyw.zimage.pipeline

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Fixtures written by `reference/make_fixtures.py` from the verified Mac reference run. */
object Golden {
    val math: JSONObject by lazy { JSONObject(text("golden/math.json")) }
    val run: JSONObject by lazy { JSONObject(text("golden/run.json")) }

    /** `models/host` of the checkout; tests that need the upstream tables skip without it. */
    val hostDir: File = File("../models/host")

    fun f32(name: String): FloatArray {
        val bytes = stream("golden/$name.f32").readBytes()
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    fun floats(a: JSONArray): FloatArray = FloatArray(a.length()) { a.getDouble(it).toFloat() }

    fun ints(a: JSONArray): IntArray = IntArray(a.length()) { a.getInt(it) }

    fun assertClose(expected: FloatArray, actual: FloatArray, atol: Float, rtol: Float = 0f, what: String = "") {
        assertEquals("$what size", expected.size, actual.size)
        var worst = 0f
        var at = -1
        for (i in expected.indices) {
            val err = Math.abs(expected[i] - actual[i]) - rtol * Math.abs(expected[i])
            if (err > worst) {
                worst = err
                at = i
            }
        }
        if (worst > atol) {
            throw AssertionError("$what: [$at] expected ${expected[at]} got ${actual[at]} (excess ${worst - atol})")
        }
    }

    private fun stream(path: String) =
        requireNotNull(javaClass.classLoader!!.getResourceAsStream(path)) { "missing test resource $path" }

    private fun text(path: String) = stream(path).bufferedReader().readText()
}
