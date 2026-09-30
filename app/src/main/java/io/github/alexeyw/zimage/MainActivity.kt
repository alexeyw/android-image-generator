package io.github.alexeyw.zimage

import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.platform.LocalContext
import io.github.alexeyw.zimage.pipeline.Backend
import io.github.alexeyw.zimage.ui.MainScreen
import io.github.alexeyw.zimage.ui.MainViewModel

/**
 * Single-screen demo. Also drivable from a shell for device benchmarks:
 *
 *   adb shell am start -n io.github.alexeyw.zimage/.MainActivity --ez autorun true \
 *     --es prompt "a red fox in fresh snow" --el seed 7 --ei steps 8 \
 *     --es backend CPU --ei threads 6
 *
 * The PNG lands in Android/data/io.github.alexeyw.zimage/files/outputs/, timings in logcat (tag ZImage).
 */
class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val dark = isSystemInDarkTheme()
            val ctx = LocalContext.current
            val colors = when {
                Build.VERSION.SDK_INT >= 31 && dark -> dynamicDarkColorScheme(ctx)
                Build.VERSION.SDK_INT >= 31 -> dynamicLightColorScheme(ctx)
                dark -> darkColorScheme()
                else -> lightColorScheme()
            }
            MaterialTheme(colorScheme = colors) { MainScreen(vm) }
        }
        if (savedInstanceState == null) handleAutorun(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleAutorun(intent)
    }

    private fun handleAutorun(intent: Intent) {
        if (!intent.getBooleanExtra("autorun", false)) return
        intent.getStringExtra("prompt")?.let(vm::setPrompt)
        if (intent.hasExtra("seed")) vm.setSeed(intent.getLongExtra("seed", 42))
        if (intent.hasExtra("steps")) vm.setSteps(intent.getIntExtra("steps", 8))
        val backend = intent.getStringExtra("backend")?.let { runCatching { Backend.valueOf(it) }.getOrNull() }
        val policy = vm.state.value.policy
        vm.setPolicy(
            policy.copy(
                ditBackend = backend ?: policy.ditBackend,
                keepDitResident = intent.getBooleanExtra("keep", policy.keepDitResident),
                cpuThreads = intent.getIntExtra("threads", policy.cpuThreads),
            ),
        )
        vm.setRandomSeed(false)
        vm.generate()
    }
}
