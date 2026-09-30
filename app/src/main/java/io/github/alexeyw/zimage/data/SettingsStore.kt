package io.github.alexeyw.zimage.data

import android.content.Context
import io.github.alexeyw.zimage.pipeline.Backend
import io.github.alexeyw.zimage.pipeline.RuntimePolicy

/** The generator form as the user left it. */
data class Settings(
    val prompt: String,
    val seed: Long,
    val randomSeed: Boolean,
    val steps: Int,
    val policy: RuntimePolicy,
)

/** Persists [Settings] in SharedPreferences; reads are synchronous so the first frame is already right. */
class SettingsStore(context: Context, private val maxCpuThreads: Int) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    fun load(defaults: Settings): Settings {
        val d = defaults.policy
        return Settings(
            prompt = prefs.getString(PROMPT, null) ?: defaults.prompt,
            seed = prefs.getLong(SEED, defaults.seed),
            randomSeed = prefs.getBoolean(RANDOM_SEED, defaults.randomSeed),
            steps = prefs.getInt(STEPS, defaults.steps).coerceIn(1, 16),
            policy = RuntimePolicy(
                ditBackend = prefs.getString(BACKEND, null)
                    ?.let { runCatching { Backend.valueOf(it) }.getOrNull() } ?: d.ditBackend,
                keepDitResident = prefs.getBoolean(KEEP_RESIDENT, d.keepDitResident),
                // Clamped to this phone's cores, whatever was stored.
                cpuThreads = prefs.getInt(CPU_THREADS, d.cpuThreads).coerceIn(1, maxCpuThreads),
            ),
        )
    }

    fun save(s: Settings) {
        prefs.edit()
            .putString(PROMPT, s.prompt)
            .putLong(SEED, s.seed)
            .putBoolean(RANDOM_SEED, s.randomSeed)
            .putInt(STEPS, s.steps)
            .putString(BACKEND, s.policy.ditBackend.name)
            .putBoolean(KEEP_RESIDENT, s.policy.keepDitResident)
            .putInt(CPU_THREADS, s.policy.cpuThreads)
            .apply()
    }

    private companion object {
        const val PROMPT = "prompt"
        const val SEED = "seed"
        const val RANDOM_SEED = "random_seed"
        const val STEPS = "steps"
        const val BACKEND = "backend"
        const val KEEP_RESIDENT = "keep_dit_resident"
        const val CPU_THREADS = "cpu_threads"
    }
}
