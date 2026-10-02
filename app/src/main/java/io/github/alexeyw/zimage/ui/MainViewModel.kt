package io.github.alexeyw.zimage.ui

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.alexeyw.zimage.data.HistoryEntry
import io.github.alexeyw.zimage.data.HistoryStore
import io.github.alexeyw.zimage.data.Settings
import io.github.alexeyw.zimage.data.SettingsStore
import io.github.alexeyw.zimage.download.ModelDownloader
import io.github.alexeyw.zimage.pipeline.Backend
import io.github.alexeyw.zimage.pipeline.PromptText
import io.github.alexeyw.zimage.pipeline.RuntimePolicy
import io.github.alexeyw.zimage.pipeline.ZImageMath
import io.github.alexeyw.zimage.pipeline.ZImagePipeline
import io.github.alexeyw.zimage.pipeline.defaultCpuThreads
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import kotlin.random.Random

data class UiState(
    val missingFiles: List<String> = emptyList(),
    val modelBytesTotal: Long = 0,
    val modelBytesPresent: Long = 0,
    val downloading: Boolean = false,
    val downloadFile: String = "",
    val prompt: String = DEFAULT_PROMPT,
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
    val history: List<HistoryEntry> = emptyList(),
) {
    val modelsReady get() = missingFiles.isEmpty()
    val tokenBudgetExceeded get() = (tokenCount ?: 0) > ZImageMath.CAP_LEN

    fun settings() = Settings(prompt, seed, randomSeed, steps, policy)

    companion object {
        const val DEFAULT_PROMPT = "a red apple on a wooden table, studio lighting"
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val filesRoot = requireNotNull(app.getExternalFilesDir(null))
    private val modelRoot = File(filesRoot, "models")
    private val downloader = ModelDownloader(modelRoot)

    // In the app's external files so `adb pull` can reach it: <id>.png + <id>.json per generation.
    private val historyStore = HistoryStore(File(filesRoot, "history"))

    // Weight caches are keyed by the pinned model revision. XNNPACK's staleness check appears to
    // cover its own build stamp (per its log strings), with no sign of comparing model bytes, so
    // packed weights of an older revision must never be found at the path of a newer one.
    private val cacheDir = File(app.cacheDir, "graphs-" + ModelDownloader.LITERT_REVISION.take(12))

    /** Upper bound of the thread picker: every core the process may use. */
    val maxCpuThreads: Int = Runtime.getRuntime().availableProcessors()
    val defaultCpuThreads: Int = defaultCpuThreads()

    private val settingsStore = SettingsStore(app, maxCpuThreads)

    // LiteRT objects are created, run and closed on this one thread.
    private val engine = Dispatchers.Default.limitedParallelism(1)
    private var pipeline: ZImagePipeline? = null
    private var job: Job? = null

    private val _state = MutableStateFlow(
        settingsStore.load(UiState().settings()).let { s ->
            UiState(prompt = PromptText.clean(s.prompt), seed = s.seed, randomSeed = s.randomSeed, steps = s.steps, policy = s.policy)
        },
    )
    val state: StateFlow<UiState> = _state

    init {
        // Created by the app itself: directories that `adb shell mkdir` makes under Android/data
        // belong to the shell user (mode 2770) and stay unreadable to the app.
        downloader.graphDir.mkdirs()
        downloader.hostDir.mkdirs()
        viewModelScope.launch(Dispatchers.IO) { dropStaleCaches(app.cacheDir) }
        viewModelScope.launch(Dispatchers.IO) { loadHistory() }
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

    /** Lists the history and, on a fresh start, shows the newest image again. */
    private fun loadHistory() {
        val entries = historyStore.list()
        val latest = entries.firstOrNull()
        val bitmap = latest?.let { BitmapFactory.decodeFile(it.image.path) }
        _state.update {
            if (it.image != null || bitmap == null) {
                it.copy(history = entries)
            } else {
                it.copy(history = entries, image = bitmap, lastSeed = latest.seed, summary = headline(latest))
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

    // --- the generator form: every change is persisted -----------------------------------------------

    /** Invisible characters are dropped here, so every input path (typing, any paste) is covered. */
    fun setPrompt(p: String) {
        _state.update { it.copy(prompt = PromptText.clean(p)) }
        persist()
        countTokens()
    }

    fun setSeed(s: Long) = edit { it.copy(seed = s) }

    fun setRandomSeed(r: Boolean) = edit { it.copy(randomSeed = r) }

    fun setSteps(n: Int) = edit { it.copy(steps = n.coerceIn(1, 16)) }

    fun setCpuThreads(n: Int) = setPolicy(_state.value.policy.copy(cpuThreads = n.coerceIn(1, maxCpuThreads)))

    fun setPolicy(p: RuntimePolicy) {
        applyPolicy(p)
        persist()
    }

    /** Loads prompt, seed and configuration of a past generation into the form. */
    fun reuse(e: HistoryEntry) {
        _state.update { it.copy(prompt = PromptText.clean(e.prompt), seed = e.seed, randomSeed = false, steps = e.steps) }
        applyPolicy(
            RuntimePolicy(
                ditBackend = e.backend,
                keepDitResident = e.keepDitResident,
                cpuThreads = e.cpuThreads.coerceIn(1, maxCpuThreads),
            ),
        )
        persist()
        countTokens()
    }

    /**
     * Shell-driven run (see MainActivity): uses the given values for this generation without
     * overwriting what the user saved in the form.
     */
    fun runHeadless(prompt: String?, seed: Long?, steps: Int?, backend: Backend?, keep: Boolean?, threads: Int?) {
        _state.update {
            it.copy(
                prompt = prompt?.let(PromptText::clean) ?: it.prompt,
                seed = seed ?: it.seed,
                randomSeed = false,
                steps = (steps ?: it.steps).coerceIn(1, 16),
            )
        }
        val p = _state.value.policy
        applyPolicy(
            p.copy(
                ditBackend = backend ?: p.ditBackend,
                keepDitResident = keep ?: p.keepDitResident,
                cpuThreads = (threads ?: p.cpuThreads).coerceIn(1, maxCpuThreads),
            ),
        )
        generate()
    }

    private fun edit(change: (UiState) -> UiState) {
        _state.update(change)
        persist()
    }

    private fun persist() = settingsStore.save(_state.value.settings())

    private fun applyPolicy(p: RuntimePolicy) {
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

    // --- generation -----------------------------------------------------------------------------------

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
                val entry = withContext(Dispatchers.IO) {
                    val png = ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
                    historyStore.add(
                        png = png,
                        prompt = s.prompt,
                        seed = seed,
                        steps = s.steps,
                        backend = s.policy.ditBackend,
                        keepDitResident = s.policy.keepDitResident,
                        cpuThreads = s.policy.cpuThreads,
                        seconds = result.seconds,
                        modelRevision = ModelDownloader.LITERT_REVISION,
                    )
                }
                val timings = summarize(result)
                Log.i(TAG, "generated ${entry.image.name} in ${"%.1f".format(result.seconds)}s\n$timings")
                _state.update {
                    it.copy(image = bmp, lastSeed = seed, summary = headline(entry) + "\n" + timings, history = listOf(entry) + it.history)
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

    // --- history ---------------------------------------------------------------------------------------

    fun deleteHistory(e: HistoryEntry) {
        _state.update { it.copy(history = it.history - e) }
        viewModelScope.launch(Dispatchers.IO) { historyStore.delete(e) }
    }

    fun clearHistory() {
        _state.update { it.copy(history = emptyList()) }
        viewModelScope.launch(Dispatchers.IO) { historyStore.clear() }
    }

    fun saveToGallery() {
        val bmp = _state.value.image ?: return
        saveToGallery(bmp, _state.value.lastSeed)
    }

    fun saveToGallery(e: HistoryEntry) {
        viewModelScope.launch(Dispatchers.IO) {
            BitmapFactory.decodeFile(e.image.path)?.let { saveToGallery(it, e.seed) }
        }
    }

    private fun saveToGallery(bmp: Bitmap, seed: Long?) {
        val name = "zimage_${seed}_${System.currentTimeMillis()}.png"
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

        fun headline(e: HistoryEntry): String =
            String.format(Locale.getDefault(), "%.1f s · seed %d · %d CPU threads", e.seconds, e.seed, e.cpuThreads)
    }
}
