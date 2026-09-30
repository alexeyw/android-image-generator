package io.github.alexeyw.zimage.data

import io.github.alexeyw.zimage.pipeline.Backend
import org.json.JSONObject
import java.io.File
import java.io.IOException

/** One finished generation: the image plus everything needed to reproduce it. */
data class HistoryEntry(
    val id: String,
    val createdAt: Long,
    val prompt: String,
    val seed: Long,
    val steps: Int,
    val backend: Backend,
    val keepDitResident: Boolean,
    val cpuThreads: Int,
    val seconds: Double,
    val modelRevision: String,
    val image: File,
)

/**
 * Generation history as plain files, `<id>.png` next to `<id>.json`, so it stays readable with
 * `adb pull` and needs no database. The JSON is written last and atomically: an entry exists once its
 * JSON does, and a crash mid-save leaves at most an orphan PNG that [list] ignores and [clear] removes.
 */
class HistoryStore(private val dir: File) {

    /** Newest first. Entries whose JSON is unreadable or whose PNG is gone are skipped. */
    fun list(): List<HistoryEntry> =
        (dir.listFiles { f -> f.name.endsWith(".json") } ?: emptyArray())
            .mapNotNull { runCatching { read(it) }.getOrNull() }
            .filter { it.image.isFile }
            .sortedWith(compareByDescending<HistoryEntry> { it.createdAt }.thenByDescending { it.id })

    fun add(
        png: ByteArray,
        prompt: String,
        seed: Long,
        steps: Int,
        backend: Backend,
        keepDitResident: Boolean,
        cpuThreads: Int,
        seconds: Double,
        modelRevision: String,
        createdAt: Long = System.currentTimeMillis(),
    ): HistoryEntry {
        dir.mkdirs()
        val id = uniqueId(createdAt, seed)
        val image = File(dir, "$id.png")
        writeAtomically(image, png)
        val entry = HistoryEntry(id, createdAt, prompt, seed, steps, backend, keepDitResident, cpuThreads, seconds, modelRevision, image)
        writeAtomically(File(dir, "$id.json"), toJson(entry).toString(2).toByteArray())
        return entry
    }

    fun delete(entry: HistoryEntry) {
        File(dir, "${entry.id}.json").delete()
        entry.image.delete()
    }

    /** Removes every entry, including orphan PNGs of interrupted saves. */
    fun clear() {
        dir.listFiles()?.forEach { if (it.isFile) it.delete() }
    }

    private fun uniqueId(createdAt: Long, seed: Long): String {
        var id = "${createdAt}_seed$seed"
        var n = 1
        while (File(dir, "$id.json").exists() || File(dir, "$id.png").exists()) id = "${createdAt}_seed${seed}_${n++}"
        return id
    }

    private fun writeAtomically(target: File, bytes: ByteArray) {
        val tmp = File(target.path + ".tmp")
        tmp.writeBytes(bytes)
        if (!tmp.renameTo(target)) throw IOException("cannot move $tmp to $target")
    }

    private fun read(json: File): HistoryEntry {
        val o = JSONObject(json.readText())
        return HistoryEntry(
            id = json.name.removeSuffix(".json"),
            createdAt = o.getLong("createdAt"),
            prompt = o.getString("prompt"),
            seed = o.getLong("seed"),
            steps = o.getInt("steps"),
            backend = Backend.valueOf(o.getString("backend")),
            keepDitResident = o.optBoolean("keepDitResident", false),
            cpuThreads = o.getInt("cpuThreads"),
            seconds = o.optDouble("seconds", Double.NaN),
            modelRevision = o.optString("modelRevision", ""),
            image = File(dir, o.getString("image")),
        )
    }

    private fun toJson(e: HistoryEntry) = JSONObject()
        .put("version", 1)
        .put("createdAt", e.createdAt)
        .put("prompt", e.prompt)
        .put("seed", e.seed)
        .put("steps", e.steps)
        .put("backend", e.backend.name)
        .put("keepDitResident", e.keepDitResident)
        .put("cpuThreads", e.cpuThreads)
        .put("seconds", e.seconds)
        .put("modelRevision", e.modelRevision)
        .put("image", e.image.name)
}
