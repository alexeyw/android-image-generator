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
import io.github.alexeyw.zimage.ui.ZImageApp
import io.github.alexeyw.zimage.ui.MainViewModel

/**
 * The generator and its history. Also drivable from a shell for device benchmarks:
 *
 *   adb shell am start -n io.github.alexeyw.zimage/.MainActivity --ez autorun true \
 *     --es prompt "a red fox in fresh snow" --el seed 7 --ei steps 8 \
 *     --es backend CPU --ei threads 6
 *
 * The run is saved to the history (Android/data/io.github.alexeyw.zimage/files/history/, PNG + JSON),
 * timings go to logcat (tag ZImage). The extras apply to this run only; the saved form is untouched.
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
            MaterialTheme(colorScheme = colors) { ZImageApp(vm) }
        }
        if (savedInstanceState == null) handleAutorun(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleAutorun(intent)
    }

    private fun handleAutorun(intent: Intent) {
        if (!intent.getBooleanExtra("autorun", false)) return
        vm.runHeadless(
            prompt = intent.getStringExtra("prompt"),
            seed = if (intent.hasExtra("seed")) intent.getLongExtra("seed", 42) else null,
            steps = if (intent.hasExtra("steps")) intent.getIntExtra("steps", 8) else null,
            backend = intent.getStringExtra("backend")?.let { runCatching { Backend.valueOf(it) }.getOrNull() },
            keep = if (intent.hasExtra("keep")) intent.getBooleanExtra("keep", false) else null,
            threads = if (intent.hasExtra("threads")) intent.getIntExtra("threads", 0) else null,
        )
    }
}
