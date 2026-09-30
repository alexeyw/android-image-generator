package io.github.alexeyw.zimage.ui

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.alexeyw.zimage.download.ModelDownloader
import io.github.alexeyw.zimage.pipeline.Backend
import io.github.alexeyw.zimage.pipeline.RuntimePolicy
import io.github.alexeyw.zimage.pipeline.ZImageMath
import io.github.alexeyw.zimage.pipeline.ZImagePipeline
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.random.Random

data class UiState(
    val missingFiles: List<String> = emptyList(),
    val modelBytesTotal: Long = 0,
    val modelBytesPresent: Long = 0,
    val downloading: Boolean = false,
    val downloadFile: String = "",
    val prompt: String = "a red apple on a wooden table, studio lighting",
    val tokenCount: Int? = null,
    val seed: Long = 42,
    val randomSeed: Boolean = false,
    val steps: Int = 8,
    val policy: RuntimePolicy = RuntimePolicy(),
    val generating: Boolean = false,
    val stage: String = "",
    val progress: Float = 0f,
    val image: Bitmap? = null,
    val lastSeed: Long? = null,
    val summary: String? = null,
    val error: String? = null,
) {
    val modelsReady get() = missingFiles.isEmpty()
    val tokenBudgetExceeded get() = (tokenCount ?: 0) > ZImageMath.CAP_LEN
}

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val modelRoot = File(requireNotNull(app.getExternalFilesDir(null)), "models")
    private val outputDir = File(requireNotNull(app.getExternalFilesDir(null)), "outputs")
    private val downloader = ModelDownloader(modelRoot)

    // Weight caches are keyed by the pinned model revision. XNNPACK's staleness check appears to
    // cover its own build stamp (per its log strings), with no sign of comparing model bytes, so
    // packed weights of an older revision must never be found at the path of a newer one.
    private val cacheDir = File(app.cacheDir, "graphs-" + ModelDownloader.LITERT_REVISION.take(12))

    // LiteRT objects are created, run and closed on this one thread.
    private val engine = Dispatchers.Default.limitedParallelism(1)
    private var pipeline: ZImagePipeline? = null
    private var job: Job? = null

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    init {
        // Created by the app itself: directories that `adb shell mkdir` makes under Android/data
        // belong to the shell user (mode 2770) and stay unreadable to the app.
        downloader.graphDir.mkdirs()
        downloader.hostDir.mkdirs()
        viewModelScope.launch(Dispatchers.IO) { dropStaleCaches(app.cacheDir) }
        refreshModels()
    }

    /** Deletes weight caches of other model revisions (and the pre-0.1 unkeyed layout). */
    private fun dropStaleCaches(root: File) {
        root.listFiles()?.forEach { f ->
            val stale = f.name in setOf("xnnpack", "gpu") || (f.name.startsWith("graphs-") && f != cacheDir)
            if (stale) {
                Log.i(TAG, "dropping stale cache ${f.name}")
                f.deleteRecursively()
            }
        }
    }

    fun refreshModels() {
        _state.update {
            it.copy(
                missingFiles = downloader.missing(),
                modelBytesTotal = downloader.totalBytes(),
                modelBytesPresent = downloader.presentBytes(),
            )
        }
        if (_state.value.modelsReady) countTokens()
    }

    fun download() {
        if (_state.value.downloading) return
        job = viewModelScope.launch {
            _state.update { it.copy(downloading = true, error = null) }
            try {
                downloader.downloadAll { p ->
                    _state.update { it.copy(downloadFile = p.file, modelBytesPresent = p.doneBytes) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "download failed", e)
                _state.update { it.copy(error = "Download failed: ${e.message}") }
            } finally {
                _state.update { it.copy(downloading = false) }
                refreshModels()
            }
        }
    }

    fun setPrompt(p: String) {
        _state.update { it.copy(prompt = p) }
        countTokens()
    }

    fun setSeed(s: Long) = _state.update { it.copy(seed = s) }

    fun setRandomSeed(r: Boolean) = _state.update { it.copy(randomSeed = r) }

    fun setSteps(n: Int) = _state.update { it.copy(steps = n.coerceIn(1, 16)) }

    fun setPolicy(p: RuntimePolicy) {
        if (p == _state.value.policy) return
        _state.update { it.copy(policy = p) }
        viewModelScope.launch(engine) {
            pipeline?.close()
            pipeline = null
        }
    }

    private var countJob: Job? = null

    private fun countTokens() {
        countJob?.cancel()
        val prompt = _state.value.prompt
        countJob = viewModelScope.launch(engine) {
            val n = runCatching { pipelineLocked().tokenCount(prompt) }.getOrNull()
            _state.update { if (it.prompt == prompt) it.copy(tokenCount = n) else it }
        }
    }

    fun generate() {
        val s = _state.value
        if (s.generating || !s.modelsReady || s.tokenBudgetExceeded) return
        val seed = if (s.randomSeed) Random.nextLong(0, 1L shl 31) else s.seed
        job = viewModelScope.launch {
            _state.update { it.copy(generating = true, error = null, stage = "Loading", progress = 0f) }
            try {
                val result = withContext(engine) {
                    pipelineLocked().generate(s.prompt, seed, s.steps) { p ->
                        _state.update { it.copy(stage = p.stage, progress = p.fraction) }
                    }
                }
                val bmp = Bitmap.createBitmap(result.argb, ZImageMath.IMAGE_PX, ZImageMath.IMAGE_PX, Bitmap.Config.ARGB_8888)
                val file = withContext(Dispatchers.IO) { saveOutput(bmp, seed, s.steps) }
                val summary = summarize(result)
                Log.i(TAG, "generated ${file.name} in ${"%.1f".format(result.seconds)}s\n$summary")
                _state.update {
                    it.copy(image = bmp, lastSeed = seed, summary = "${"%.1f".format(result.seconds)} s · seed $seed\n$summary")
                }
            } catch (e: CancellationException) {
                _state.update { it.copy(stage = "Cancelled") }
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "generation failed", e)
                _state.update { it.copy(error = "${e.javaClass.simpleName}: ${e.message}") }
            } finally {
                _state.update { it.copy(generating = false) }
            }
        }
    }

    fun cancel() {
        job?.cancel()
    }

    fun saveToGallery() {
        val bmp = _state.value.image ?: return
        val name = "zimage_${_state.value.lastSeed}_${System.currentTimeMillis()}.png"
        viewModelScope.launch(Dispatchers.IO) {
            val resolver = getApplication<Application>().contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Z-Image")
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return@launch
            resolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            _state.update { it.copy(summary = (it.summary ?: "") + "\nSaved to Pictures/Z-Image") }
        }
    }

    /** Must run on [engine]. */
    private fun pipelineLocked(): ZImagePipeline =
        pipeline ?: ZImagePipeline(modelRoot, cacheDir, _state.value.policy)
            .also { pipeline = it }

    private fun saveOutput(bmp: Bitmap, seed: Long, steps: Int): File {
        outputDir.mkdirs()
        val f = File(outputDir, "zimage_seed${seed}_steps${steps}_${System.currentTimeMillis()}.png")
        f.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return f
    }

    private fun summarize(r: ZImagePipeline.Result): String = r.timings.entries.joinToString("\n") { (name, runs) ->
        val load = runs.sumOf { it.first } / 1000.0
        val run = runs.sumOf { it.second } / 1000.0
        "%-9s ×%d  load %.1fs  run %.1fs".format(name, runs.size, load, run)
    }

    override fun onCleared() {
        val p = pipeline
        pipeline = null
        // Close on the engine thread once the in-flight graph returns (the scope is already
        // cancelled); never block the main thread on a multi-second GPU call.
        if (p != null) CoroutineScope(engine).launch { p.close() }
    }

    companion object {
        private const val TAG = "ZImage"
    }
}
