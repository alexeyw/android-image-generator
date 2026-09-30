package io.github.alexeyw.zimage.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.alexeyw.zimage.data.HistoryEntry
import io.github.alexeyw.zimage.pipeline.Backend
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.DateFormat
import java.util.Date
import java.util.Locale

private enum class Screen { Main, History, Detail }

/** Root: the generator, the history grid and one history entry, with system Back between them. */
@Composable
fun ZImageApp(vm: MainViewModel) {
    var screen by rememberSaveable { mutableStateOf(Screen.Main) }
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    val s by vm.state.collectAsStateWithLifecycle()
    val selected = s.history.firstOrNull { it.id == selectedId }

    when {
        screen == Screen.Detail && selected != null -> {
            BackHandler { screen = Screen.History }
            HistoryDetail(
                entry = selected,
                busy = s.generating,
                onBack = { screen = Screen.History },
                onReuse = {
                    vm.reuse(selected)
                    screen = Screen.Main
                },
                onSave = { vm.saveToGallery(selected) },
                onDelete = {
                    vm.deleteHistory(selected)
                    screen = Screen.History
                },
            )
        }
        screen != Screen.Main -> {
            BackHandler { screen = Screen.Main }
            HistoryGrid(
                entries = s.history,
                onBack = { screen = Screen.Main },
                onOpen = {
                    selectedId = it.id
                    screen = Screen.Detail
                },
                onClear = vm::clearHistory,
            )
        }
        else -> MainScreen(vm, onOpenHistory = { screen = Screen.History })
    }
}

@Composable
private fun HistoryGrid(
    entries: List<HistoryEntry>,
    onBack: () -> Unit,
    onOpen: (HistoryEntry) -> Unit,
    onClear: () -> Unit,
) {
    var confirmClear by remember { mutableStateOf(false) }
    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .safeDrawingPadding()
            .padding(horizontal = 16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("‹ Back") }
            Text("History (${entries.size})", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            if (entries.isNotEmpty()) TextButton(onClick = { confirmClear = true }) { Text("Clear all") }
        }
        if (entries.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Generated images appear here.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 104.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxSize(),
            ) {
                items(entries, key = { it.id }) { e ->
                    HistoryImage(
                        e,
                        Modifier
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { onOpen(e) },
                    )
                }
            }
        }
    }
    if (confirmClear) {
        ConfirmDialog(
            title = if (entries.size == 1) "Delete the only image?" else "Delete all ${entries.size} images?",
            text = "The images and their settings are removed from this phone. Copies saved to the gallery stay.",
            confirm = "Delete all",
            onConfirm = {
                confirmClear = false
                onClear()
            },
            onDismiss = { confirmClear = false },
        )
    }
}

@Composable
private fun HistoryDetail(
    entry: HistoryEntry,
    busy: Boolean,
    onBack: () -> Unit,
    onReuse: () -> Unit,
    onSave: () -> Unit,
    onDelete: () -> Unit,
) {
    var confirmDelete by remember { mutableStateOf(false) }
    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        TextButton(onClick = onBack) { Text("‹ History") }
        HistoryImage(
            entry,
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(12.dp)),
        )
        SelectionContainer { Text(entry.prompt, style = MaterialTheme.typography.bodyLarge) }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Detail("Seed", entry.seed.toString())
            Detail("Steps", entry.steps.toString())
            Detail("DiT", if (entry.backend == Backend.GPU) "GPU" + if (entry.keepDitResident) ", kept resident" else "" else "CPU")
            Detail("CPU threads", entry.cpuThreads.toString())
            if (!entry.seconds.isNaN()) Detail("Time", String.format(Locale.getDefault(), "%.1f s", entry.seconds))
            Detail("Created", DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(entry.createdAt)))
        }
        Button(onClick = onReuse, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("Use these settings") }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onSave) { Text("Save to gallery") }
            OutlinedButton(onClick = { confirmDelete = true }) { Text("Delete") }
        }
        Spacer(Modifier)
    }
    if (confirmDelete) {
        ConfirmDialog(
            title = "Delete this image?",
            text = "The image and its settings are removed from this phone.",
            confirm = "Delete",
            onConfirm = {
                confirmDelete = false
                onDelete()
            },
            onDismiss = { confirmDelete = false },
        )
    }
}

@Composable
private fun Detail(label: String, value: String) {
    Row {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(0.4f))
        Text(value, modifier = Modifier.weight(0.6f), maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

/** Decodes the 256 px PNG off the main thread; a grid of them needs no image library. */
@Composable
private fun HistoryImage(entry: HistoryEntry, modifier: Modifier) {
    val bitmap by produceState<Bitmap?>(null, entry.image) {
        value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(entry.image.path) }
    }
    Box(modifier.background(MaterialTheme.colorScheme.surfaceVariant)) {
        bitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = entry.prompt,
                contentScale = ContentScale.Crop,
                filterQuality = FilterQuality.High,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
}

@Composable
private fun ConfirmDialog(title: String, text: String, confirm: String, onConfirm: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(text) },
        confirmButton = { TextButton(onClick = onConfirm) { Text(confirm) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
