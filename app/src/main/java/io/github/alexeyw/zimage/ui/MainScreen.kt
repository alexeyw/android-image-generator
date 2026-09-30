package io.github.alexeyw.zimage.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.alexeyw.zimage.pipeline.Backend
import io.github.alexeyw.zimage.pipeline.ZImageMath
import java.util.Locale
import kotlin.math.roundToInt

@Composable
fun MainScreen(vm: MainViewModel, onOpenHistory: () -> Unit) {
    val s by vm.state.collectAsStateWithLifecycle()
    KeepScreenOn(s.downloading || s.generating)

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Z-Image Turbo", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
            TextButton(onClick = onOpenHistory) { Text("History (${s.history.size})") }
        }
        Text(
            "6B text-to-image, fully on this phone via LiteRT. 256 × 256.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        if (!s.modelsReady) ModelsCard(s, vm) else GeneratorCard(s, vm)

        s.error?.let {
            Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
        }

        ImageArea(s)

        if (s.image != null && !s.generating) {
            OutlinedButton(onClick = vm::saveToGallery) { Text("Save to gallery") }
        }
        s.summary?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
private fun ModelsCard(s: UiState, vm: MainViewModel) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Model files", style = MaterialTheme.typography.titleMedium)
            Text(
                "${gb(s.modelBytesPresent)} of ${gb(s.modelBytesTotal)} on the device. " +
                    "The download comes straight from Hugging Face; Wi-Fi recommended, keep the app open.",
                style = MaterialTheme.typography.bodyMedium,
            )
            val total = s.modelBytesTotal.coerceAtLeast(1)
            LinearProgressIndicator(
                progress = { s.modelBytesPresent.toFloat() / total },
                modifier = Modifier.fillMaxWidth(),
            )
            if (s.downloading) {
                Text(s.downloadFile, style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = vm::cancel) { Text("Pause") }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = vm::download) {
                        Text(if (s.modelBytesPresent > 0) "Resume download" else "Download ${gb(s.modelBytesTotal)}")
                    }
                    OutlinedButton(onClick = vm::refreshModels) { Text("Re-check") }
                }
                Text(
                    "Missing: " + s.missingFiles.joinToString(", "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun GeneratorCard(s: UiState, vm: MainViewModel) {
    val busy = s.generating
    // Same pattern as the seed: typing goes to local state first, so a late state echo cannot
    // reset the cursor or drop characters; external changes (history) are adopted.
    var promptText by rememberSaveable { mutableStateOf(s.prompt) }
    var sentPrompt by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(s.prompt) {
        val current = vm.state.value.prompt
        if (current != sentPrompt) {
            promptText = current
            sentPrompt = current
        }
    }
    OutlinedTextField(
        value = promptText,
        onValueChange = {
            promptText = it
            sentPrompt = it
            vm.setPrompt(it)
        },
        label = { Text("Prompt") },
        enabled = !busy,
        minLines = 2,
        isError = s.tokenBudgetExceeded,
        supportingText = {
            val n = s.tokenCount
            Text(
                if (n == null) "…" else "$n / ${ZImageMath.CAP_LEN} tokens (chat template included)" +
                    if (s.tokenBudgetExceeded) " — too long for the 256 px graphs" else "",
            )
        },
        modifier = Modifier.fillMaxWidth(),
    )

    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        // Own text state: the field may be empty or mid-edit; only a valid number reaches the form.
        var seedText by rememberSaveable { mutableStateOf(s.seed.toString()) }
        var sentSeed by remember { mutableStateOf<Long?>(null) }
        LaunchedEffect(s.seed) {
            // Adopt the seed only when it changed elsewhere (history "Use these settings"), not an
            // echo of this field's own edit: echoes arrive late and would undo a fast delete.
            val current = vm.state.value.seed
            if (current != sentSeed) {
                seedText = current.toString()
                sentSeed = current
            }
        }
        OutlinedTextField(
            value = seedText,
            onValueChange = { t ->
                seedText = t.filter(Char::isDigit).take(18)
                seedText.toLongOrNull()?.let {
                    sentSeed = it
                    vm.setSeed(it)
                }
            },
            label = { Text("Seed") },
            enabled = !busy && !s.randomSeed,
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.width(140.dp),
        )
        Text("Random")
        Switch(checked = s.randomSeed, onCheckedChange = vm::setRandomSeed, enabled = !busy)
    }

    Text("Steps: ${s.steps}")
    Slider(
        value = s.steps.toFloat(),
        onValueChange = { vm.setSteps(it.toInt()) },
        valueRange = 1f..16f,
        steps = 14,
        enabled = !busy,
    )

    if (vm.maxCpuThreads > 1) {
        val threads = s.policy.cpuThreads
        Text("CPU threads: $threads" + if (threads == vm.defaultCpuThreads) " (default)" else "")
        Slider(
            value = threads.toFloat(),
            onValueChange = { vm.setCpuThreads(it.roundToInt()) },
            valueRange = 1f..vm.maxCpuThreads.toFloat(),
            steps = vm.maxCpuThreads - 2,
            enabled = !busy,
        )
        if (threads != vm.defaultCpuThreads) {
            Text(
                "More threads is not always faster: they share the cores with the system. " +
                    "The default is cores minus two, at most six.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        val gpu = s.policy.ditBackend == Backend.GPU
        FilterChip(
            selected = gpu,
            onClick = { vm.setPolicy(s.policy.copy(ditBackend = if (gpu) Backend.CPU else Backend.GPU)) },
            label = { Text("GPU (experimental)") },
            enabled = !busy,
        )
        if (gpu) {
            FilterChip(
                selected = s.policy.keepDitResident,
                onClick = { vm.setPolicy(s.policy.copy(keepDitResident = !s.policy.keepDitResident)) },
                label = { Text("Keep on GPU (16 GB+)") },
                enabled = !busy,
            )
        }
    }
    if (s.policy.ditBackend == Backend.GPU) {
        Text(
            "GPU runs quantized activations (a slightly different image) and reloads every shard each " +
                "step unless kept resident, which needs ~8 GB free.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }

    if (busy) {
        Text(s.stage, style = MaterialTheme.typography.bodyMedium)
        LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth())
        OutlinedButton(onClick = vm::cancel) { Text("Cancel") }
    } else {
        Button(
            onClick = vm::generate,
            enabled = !s.tokenBudgetExceeded && s.tokenCount != null,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Generate") }
    }
}

@Composable
private fun ImageArea(s: UiState) {
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        val img = s.image
        if (img != null) {
            Image(
                bitmap = img.asImageBitmap(),
                contentDescription = s.prompt,
                contentScale = ContentScale.Fit,
                filterQuality = FilterQuality.High,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Text(
                if (s.generating) "Generating…" else "Your image appears here",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    Spacer(Modifier)
}

@Composable
private fun KeepScreenOn(on: Boolean) {
    val view = LocalView.current
    DisposableEffect(on) {
        view.keepScreenOn = on
        onDispose { view.keepScreenOn = false }
    }
}

private fun gb(bytes: Long) = String.format(Locale.US, "%.2f GB", bytes / 1e9)
