package io.github.alexeyw.zimage.pipeline

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import kotlin.system.measureTimeMillis

enum class Backend { GPU, CPU }

/**
 * Runtime choices that decide speed versus memory. The graphs themselves are fixed.
 *
 * Measured on a Galaxy S25 Ultra (12 GB), 8 steps, 2026-09-29:
 *  - CPU (default): ~29 s per image with 6 threads once the XNNPACK weight cache exists (a shard
 *    then loads in ~10 ms); matches the Mac reference at 35 dB PSNR.
 *  - GPU streaming: a shard runs in 0.2 s but loads in ~4 s every step, 284 s per image.
 *  - GPU resident: ~1.3 GB per shard; lmkd kills the app at the fifth shard on 12 GB.
 *
 * @param keepDitResident keep the per-step graphs compiled between steps (GPU only matters:
 *   CPU loads are already memory maps of the weight cache).
 * @param cpuThreads XNNPACK threads. Interleaved runs on the S25 Ultra (6 x 3.53 GHz + 2 x 4.47 GHz
 *   cores, `scripts/bench_threads.py`, 2026-09-30): 4 threads 36.6 s median (31.8-43.5), 6 threads
 *   28.8 s (28.1-28.8), 8 threads 36.3 s (29.8-51.4); identical pixels for all three. "Cores minus
 *   two, at most six" gives 6 there; it is a guess for other core layouts.
 */
data class RuntimePolicy(
    val ditBackend: Backend = Backend.CPU,
    val keepDitResident: Boolean = false,
    val cpuThreads: Int = defaultCpuThreads(),
)

fun defaultCpuThreads(): Int = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(1, 6)

/**
 * Loads `.tflite` graphs with LiteRT `CompiledModel`, one shared [Environment] for all of them
 * (a fresh Environment per graph leaks the OpenCL context), and falls back to the CPU when the
 * GPU delegate cannot compile a graph. Not thread-safe: drive it from one thread.
 */
class GraphRunner(
    private val modelDir: File,
    private val cacheDir: File,
    private val policy: RuntimePolicy,
) : AutoCloseable {
    private val env: Environment = Environment.create().also { Log.i(TAG, "runtime $policy") }
    private val resident = LinkedHashMap<String, Loaded>()
    private val gpuRejected = HashSet<String>()

    /** Wall-clock per graph for the last generation: name -> (load ms, run ms). */
    val timings = LinkedHashMap<String, MutableList<Pair<Long, Long>>>()

    private class Loaded(
        val model: CompiledModel,
        val inputs: List<TensorBuffer>,
        val outputs: List<TensorBuffer>,
        val backend: Backend,
    ) {
        fun close() {
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()
        }
    }

    fun run(name: String, vararg inputs: FloatArray): FloatArray {
        var loadMs = 0L
        val loaded = resident[name] ?: run {
            lateinit var l: Loaded
            loadMs = measureTimeMillis { l = load(name) }
            if (keepResident(name)) resident[name] = l
            l
        }
        lateinit var out: FloatArray
        val runMs = measureTimeMillis {
            require(inputs.size == loaded.inputs.size) { "$name takes ${loaded.inputs.size} inputs, got ${inputs.size}" }
            inputs.forEachIndexed { i, a -> loaded.inputs[i].writeFloat(a) }
            loaded.model.run(loaded.inputs, loaded.outputs)
            out = loaded.outputs[0].readFloat()
        }
        if (resident[name] !== loaded) loaded.close()
        timings.getOrPut(name) { mutableListOf() }.add(loadMs to runMs)
        Log.i(TAG, "$name [${loaded.backend}] load ${loadMs}ms run ${runMs}ms")
        return out
    }

    /** Drops every resident graph, e.g. before the 3.5 GB text encoder of a new prompt. */
    fun releaseAll() {
        resident.values.forEach { it.close() }
        resident.clear()
    }

    override fun close() {
        releaseAll()
        env.close()
    }

    private fun keepResident(name: String) = policy.keepDitResident && name != TEXT_ENCODER && name != VAE

    private fun backendFor(name: String): Backend = when {
        // Fails to compile on the GPU and on the NPU (model card, Galaxy S26).
        name == TEXT_ENCODER -> Backend.CPU
        // Tiny heads: faster on the CPU than a GPU dispatch (model card, Pixel 8a).
        name in CPU_HEADS -> Backend.CPU
        name in gpuRejected -> Backend.CPU
        else -> policy.ditBackend
    }

    private fun load(name: String): Loaded {
        val path = File(modelDir, "$name.tflite").absolutePath
        val backend = backendFor(name)
        if (backend == Backend.GPU) {
            try {
                return compile(path, gpuOptions(name), Backend.GPU)
            } catch (e: LiteRtException) {
                Log.w(TAG, "$name: GPU compile failed, falling back to CPU", e)
                gpuRejected += name
            }
        }
        return compile(path, cpuOptions(name), Backend.CPU)
    }

    private fun compile(path: String, options: CompiledModel.Options, backend: Backend): Loaded {
        val model = CompiledModel.create(path, options, env)
        return Loaded(model, model.createInputBuffers(), model.createOutputBuffers(), backend)
    }

    private fun gpuOptions(name: String) = CompiledModel.Options(Accelerator.GPU).apply {
        gpuOptions = CompiledModel.GpuOptions(
            // FP16 overflows in the adaLN/attention path and turns the latent into NaN.
            precision = CompiledModel.GpuOptions.Precision.FP32,
            // Without this lmkd kills the app while the 0.9 GB zc_main0 loads (S25 Ultra, 7.6 GB
            // RSS+swap; the int8 weights seem to be expanded to FP32). With it the shard fits, but
            // activations are quantized too: the image drifts from the CPU one (17 dB PSNR).
            allowSrcQuantizedFcConvOps = true,
            // Compiled OpenCL programs are cached on disk: later loads skip kernel compilation.
            serializationDir = File(cacheDir, "gpu").apply { mkdirs() }.absolutePath,
            modelCacheKey = name,
            serializeProgramCache = true,
        )
    }

    private fun cpuOptions(name: String) = CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(
            numThreads = policy.cpuThreads,
            xnnPackFlags = null,
            // XNNPACK repacks the weights on every load; the cache keeps the packed copy on disk
            // (file-backed, so the OS can page it) and makes later loads a memory map.
            xnnPackWeightCachePath = File(cacheDir, "xnnpack").apply { mkdirs() }.let { File(it, "$name.xnn") }.absolutePath,
        )
    }

    companion object {
        private const val TAG = "ZImageRuntime"
        const val TEXT_ENCODER = "qwen_enc"
        const val VAE = "zvae"
        private val CPU_HEADS = setOf("z_embx", "z_embc", "zc_final")
    }
}
