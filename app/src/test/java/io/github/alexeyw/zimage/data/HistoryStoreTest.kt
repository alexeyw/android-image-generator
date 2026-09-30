package io.github.alexeyw.zimage.data

import io.github.alexeyw.zimage.pipeline.Backend
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class HistoryStoreTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val png = byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 1, 2, 3)

    private fun HistoryStore.addSample(createdAt: Long, seed: Long = 7, prompt: String = "a fox") = add(
        png = png,
        prompt = prompt,
        seed = seed,
        steps = 8,
        backend = Backend.GPU,
        keepDitResident = true,
        cpuThreads = 6,
        seconds = 28.8,
        modelRevision = "09ea3ae2",
        createdAt = createdAt,
    )

    @Test
    fun roundTripKeepsEverythingNeededToReproduce() {
        val dir = tmp.newFolder("history")
        val added = HistoryStore(dir).addSample(1000, seed = 42, prompt = "Кот в сапогах, \"quoted\"\nline two")
        val read = HistoryStore(dir).list().single()
        assertEquals(added, read)
        assertEquals("Кот в сапогах, \"quoted\"\nline two", read.prompt)
        assertEquals(42L, read.seed)
        assertEquals(Backend.GPU, read.backend)
        assertTrue(read.keepDitResident)
        assertEquals(6, read.cpuThreads)
        assertArrayEquals(png, read.image.readBytes())
    }

    @Test
    fun newestFirst() {
        val store = HistoryStore(tmp.newFolder("history"))
        store.addSample(1000, seed = 1)
        store.addSample(3000, seed = 3)
        store.addSample(2000, seed = 2)
        assertEquals(listOf(3L, 2L, 1L), store.list().map { it.seed })
    }

    @Test
    fun sameMillisecondAndSeedGetDistinctEntries() {
        val store = HistoryStore(tmp.newFolder("history"))
        val a = store.addSample(1000)
        val b = store.addSample(1000)
        assertNotEquals(a.id, b.id)
        assertEquals(2, store.list().size)
    }

    @Test
    fun deleteRemovesImageAndSettings() {
        val dir = tmp.newFolder("history")
        val store = HistoryStore(dir)
        val keep = store.addSample(1000, seed = 1)
        val gone = store.addSample(2000, seed = 2)
        store.delete(gone)
        assertEquals(listOf(keep), store.list())
        assertFalse(gone.image.exists())
        assertFalse(File(dir, "${gone.id}.json").exists())
    }

    @Test
    fun unreadableOrIncompleteEntriesAreSkipped() {
        val dir = tmp.newFolder("history")
        val store = HistoryStore(dir)
        val ok = store.addSample(1000)
        File(dir, "broken.json").writeText("{ not json")
        File(dir, "orphan.png").writeBytes(png) // interrupted save: PNG without JSON
        val noImage = store.addSample(2000, seed = 9)
        noImage.image.delete()
        assertEquals(listOf(ok), store.list())
    }

    @Test
    fun clearRemovesEntriesAndLeftovers() {
        val dir = tmp.newFolder("history")
        val store = HistoryStore(dir)
        store.addSample(1000)
        File(dir, "orphan.png").writeBytes(png)
        File(dir, "x.json.tmp").writeText("{}")
        store.clear()
        assertTrue(store.list().isEmpty())
        assertEquals(0, dir.listFiles()!!.size)
    }

    @Test
    fun missingDirectoryIsAnEmptyHistory() {
        assertTrue(HistoryStore(File(tmp.root, "never-created")).list().isEmpty())
    }
}
