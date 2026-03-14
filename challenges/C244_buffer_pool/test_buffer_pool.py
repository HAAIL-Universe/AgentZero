"""
Tests for C244: Buffer Pool Manager

Comprehensive tests covering:
- Page operations
- Disk manager
- Frame descriptor
- LRU replacer
- Clock replacer
- LRU-K replacer
- Buffer pool manager (fetch, unpin, new, delete, flush)
- Eviction behavior
- Dirty page handling
- Prefetch
- Sequential scan buffer
- Thread safety
- Statistics
- Edge cases
"""

import unittest
import threading
import time
from buffer_pool import (
    Page, PAGE_SIZE, DiskManager, FrameDescriptor,
    EvictionPolicy, LRUReplacer, ClockReplacer, LRUKReplacer,
    BufferPoolManager, BufferPoolStats, ScanBuffer,
)


# ===========================================================================
# Page Tests
# ===========================================================================

class TestPage(unittest.TestCase):

    def test_page_creation(self):
        p = Page(42)
        self.assertEqual(p.page_id, 42)
        self.assertEqual(len(p.data), PAGE_SIZE)

    def test_page_custom_size(self):
        p = Page(0, size=1024)
        self.assertEqual(len(p.data), 1024)

    def test_page_write_and_read(self):
        p = Page(0)
        p.write(0, b'hello')
        self.assertEqual(p.read(0, 5), b'hello')

    def test_page_write_offset(self):
        p = Page(0)
        p.write(100, b'data')
        self.assertEqual(p.read(100, 4), b'data')
        self.assertEqual(p.read(0, 4), b'\x00\x00\x00\x00')

    def test_page_clear(self):
        p = Page(0)
        p.write(0, b'hello')
        p.clear()
        self.assertEqual(p.read(0, 5), b'\x00\x00\x00\x00\x00')

    def test_page_repr(self):
        p = Page(5)
        self.assertIn('5', repr(p))

    def test_page_initial_zeros(self):
        p = Page(0)
        self.assertEqual(p.read(0, 10), b'\x00' * 10)

    def test_page_overwrite(self):
        p = Page(0)
        p.write(0, b'aaaa')
        p.write(0, b'bb')
        self.assertEqual(p.read(0, 4), b'bbaa')


# ===========================================================================
# Disk Manager Tests
# ===========================================================================

class TestDiskManager(unittest.TestCase):

    def test_allocate_page(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        self.assertEqual(pid, 0)
        pid2 = dm.allocate_page()
        self.assertEqual(pid2, 1)

    def test_read_write_page(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        data = b'\x01' * PAGE_SIZE
        dm.write_page(pid, data)
        result = dm.read_page(pid)
        self.assertEqual(result, data)

    def test_read_nonexistent_page(self):
        dm = DiskManager()
        with self.assertRaises(ValueError):
            dm.read_page(999)

    def test_write_wrong_size(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        with self.assertRaises(ValueError):
            dm.write_page(pid, b'short')

    def test_page_exists(self):
        dm = DiskManager()
        self.assertFalse(dm.page_exists(0))
        dm.allocate_page()
        self.assertTrue(dm.page_exists(0))

    def test_num_pages(self):
        dm = DiskManager()
        self.assertEqual(dm.num_pages(), 0)
        dm.allocate_page()
        dm.allocate_page()
        self.assertEqual(dm.num_pages(), 2)

    def test_io_counters(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        dm.read_page(pid)
        dm.write_page(pid, b'\x00' * PAGE_SIZE)
        self.assertEqual(dm.read_count, 1)
        self.assertEqual(dm.write_count, 1)

    def test_reset_counters(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        dm.read_page(pid)
        dm.reset_counters()
        self.assertEqual(dm.read_count, 0)
        self.assertEqual(dm.write_count, 0)

    def test_allocated_page_zeroed(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        data = dm.read_page(pid)
        self.assertEqual(data, b'\x00' * PAGE_SIZE)


# ===========================================================================
# Frame Descriptor Tests
# ===========================================================================

class TestFrameDescriptor(unittest.TestCase):

    def test_initial_state(self):
        fd = FrameDescriptor(frame_id=0)
        self.assertTrue(fd.is_free())
        self.assertFalse(fd.is_pinned())
        self.assertFalse(fd.is_dirty)

    def test_pinned(self):
        fd = FrameDescriptor(frame_id=0, pin_count=1)
        self.assertTrue(fd.is_pinned())

    def test_reset(self):
        fd = FrameDescriptor(frame_id=0, page_id=5, pin_count=2, is_dirty=True)
        fd.reset()
        self.assertTrue(fd.is_free())
        self.assertFalse(fd.is_pinned())
        self.assertFalse(fd.is_dirty)
        self.assertIsNone(fd.page)


# ===========================================================================
# LRU Replacer Tests
# ===========================================================================

class TestLRUReplacer(unittest.TestCase):

    def test_basic_eviction(self):
        r = LRUReplacer()
        frames = [FrameDescriptor(i) for i in range(3)]

        r.access(0)
        r.unpin(0)
        r.access(1)
        r.unpin(1)
        r.access(2)
        r.unpin(2)

        # Least recently used is 0
        victim = r.victim(frames)
        self.assertEqual(victim, 0)

    def test_recently_accessed_not_evicted(self):
        r = LRUReplacer()
        frames = [FrameDescriptor(i) for i in range(3)]

        r.access(0)
        r.unpin(0)
        r.access(1)
        r.unpin(1)
        r.access(0)  # Re-access 0, making it most recent

        victim = r.victim(frames)
        self.assertEqual(victim, 1)

    def test_pinned_not_evicted(self):
        r = LRUReplacer()
        frames = [FrameDescriptor(i) for i in range(3)]

        r.access(0)
        r.unpin(0)
        r.access(1)
        r.pin(1)  # Pin 1

        victim = r.victim(frames)
        self.assertEqual(victim, 0)

    def test_all_pinned(self):
        r = LRUReplacer()
        frames = [FrameDescriptor(i) for i in range(2)]

        r.access(0)
        r.pin(0)
        r.access(1)
        r.pin(1)

        victim = r.victim(frames)
        self.assertIsNone(victim)

    def test_empty_replacer(self):
        r = LRUReplacer()
        frames = [FrameDescriptor(i) for i in range(2)]
        victim = r.victim(frames)
        self.assertIsNone(victim)


# ===========================================================================
# Clock Replacer Tests
# ===========================================================================

class TestClockReplacer(unittest.TestCase):

    def test_basic_eviction(self):
        r = ClockReplacer(4)
        frames = [FrameDescriptor(i) for i in range(4)]

        r.unpin(0)
        r.unpin(1)
        r.unpin(2)

        victim = r.victim(frames)
        self.assertIn(victim, [0, 1, 2])

    def test_second_chance(self):
        r = ClockReplacer(4)
        frames = [FrameDescriptor(i) for i in range(4)]

        r.unpin(0)
        r.unpin(1)
        r.access(0)  # Set ref bit for 0

        victim = r.victim(frames)
        # 0 gets second chance (ref bit cleared), 1 should be evicted
        self.assertEqual(victim, 1)

    def test_pinned_skipped(self):
        r = ClockReplacer(3)
        frames = [FrameDescriptor(i) for i in range(3)]

        r.unpin(0)
        r.unpin(1)
        r.unpin(2)
        r.pin(0)

        victim = r.victim(frames)
        self.assertIn(victim, [1, 2])

    def test_all_pinned(self):
        r = ClockReplacer(3)
        frames = [FrameDescriptor(i) for i in range(3)]
        r.pin(0)
        r.pin(1)
        r.pin(2)
        victim = r.victim(frames)
        self.assertIsNone(victim)

    def test_empty_replacer(self):
        r = ClockReplacer(3)
        frames = [FrameDescriptor(i) for i in range(3)]
        victim = r.victim(frames)
        self.assertIsNone(victim)


# ===========================================================================
# LRU-K Replacer Tests
# ===========================================================================

class TestLRUKReplacer(unittest.TestCase):

    def test_evict_less_than_k_accesses(self):
        r = LRUKReplacer(k=2)
        frames = [FrameDescriptor(i) for i in range(3)]

        # Frame 0: 2 accesses, Frame 1: 1 access
        r.access(0)
        r.access(0)
        r.unpin(0)
        r.access(1)
        r.unpin(1)

        # Frame 1 has < K accesses, should be evicted first
        victim = r.victim(frames)
        self.assertEqual(victim, 1)

    def test_evict_oldest_kth_access(self):
        r = LRUKReplacer(k=2)
        frames = [FrameDescriptor(i) for i in range(3)]

        # Frame 0: 2 accesses (timestamps 1, 2)
        r.access(0)
        r.access(0)
        r.unpin(0)
        # Frame 1: 2 accesses (timestamps 3, 4)
        r.access(1)
        r.access(1)
        r.unpin(1)

        # Frame 0's 2nd-to-last (K-th) access is older
        victim = r.victim(frames)
        self.assertEqual(victim, 0)

    def test_pinned_not_evicted(self):
        r = LRUKReplacer(k=2)
        frames = [FrameDescriptor(i) for i in range(3)]

        r.access(0)
        r.pin(0)
        r.access(1)
        r.unpin(1)

        victim = r.victim(frames)
        self.assertEqual(victim, 1)

    def test_remove(self):
        r = LRUKReplacer(k=2)
        r.access(0)
        r.access(1)
        r.unpin(0)
        r.unpin(1)
        r.remove(0)

        frames = [FrameDescriptor(i) for i in range(3)]
        victim = r.victim(frames)
        self.assertEqual(victim, 1)

    def test_k_equals_1_is_lru(self):
        """LRU-1 should behave like LRU."""
        r = LRUKReplacer(k=1)
        frames = [FrameDescriptor(i) for i in range(3)]

        r.access(0)
        r.unpin(0)
        r.access(1)
        r.unpin(1)
        r.access(2)
        r.unpin(2)

        # Oldest single access is frame 0
        victim = r.victim(frames)
        self.assertEqual(victim, 0)


# ===========================================================================
# Buffer Pool Manager -- Basic Tests
# ===========================================================================

class TestBufferPoolBasic(unittest.TestCase):

    def _make_bpm(self, pool_size=10, policy=EvictionPolicy.LRU):
        dm = DiskManager()
        return BufferPoolManager(pool_size, dm, policy), dm

    def test_new_page(self):
        bpm, dm = self._make_bpm()
        result = bpm.new_page()
        self.assertIsNotNone(result)
        page_id, page = result
        self.assertEqual(page_id, 0)
        self.assertIsInstance(page, Page)

    def test_new_page_multiple(self):
        bpm, dm = self._make_bpm()
        ids = []
        for _ in range(5):
            pid, _ = bpm.new_page()
            ids.append(pid)
        self.assertEqual(ids, [0, 1, 2, 3, 4])

    def test_fetch_page(self):
        bpm, dm = self._make_bpm()
        pid, page = bpm.new_page()
        page.write(0, b'test data')
        bpm.unpin_page(pid, is_dirty=True)
        bpm.flush_page(pid)

        # Fetch same page
        fetched = bpm.fetch_page(pid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.read(0, 9), b'test data')

    def test_unpin_page(self):
        bpm, dm = self._make_bpm()
        pid, page = bpm.new_page()
        self.assertEqual(bpm.get_pin_count(pid), 1)
        bpm.unpin_page(pid)
        self.assertEqual(bpm.get_pin_count(pid), 0)

    def test_unpin_nonexistent(self):
        bpm, dm = self._make_bpm()
        self.assertFalse(bpm.unpin_page(999))

    def test_unpin_already_unpinned(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid)
        self.assertFalse(bpm.unpin_page(pid))  # Already at 0

    def test_pin_count_multiple_fetches(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid)

        bpm.fetch_page(pid)
        bpm.fetch_page(pid)
        self.assertEqual(bpm.get_pin_count(pid), 2)

    def test_dirty_flag(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        self.assertFalse(bpm.is_dirty(pid))
        bpm.unpin_page(pid, is_dirty=True)
        self.assertTrue(bpm.is_dirty(pid))

    def test_contains(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        self.assertTrue(bpm.contains(pid))
        self.assertFalse(bpm.contains(999))

    def test_delete_page(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid)
        self.assertTrue(bpm.delete_page(pid))
        self.assertFalse(bpm.contains(pid))

    def test_delete_pinned_page(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        self.assertFalse(bpm.delete_page(pid))

    def test_delete_nonexistent(self):
        bpm, dm = self._make_bpm()
        self.assertTrue(bpm.delete_page(999))

    def test_invalid_pool_size(self):
        dm = DiskManager()
        with self.assertRaises(ValueError):
            BufferPoolManager(0, dm)

    def test_fetch_nonexistent_page(self):
        bpm, dm = self._make_bpm()
        with self.assertRaises(ValueError):
            bpm.fetch_page(999)

    def test_num_free_frames(self):
        bpm, dm = self._make_bpm(pool_size=5)
        self.assertEqual(bpm.num_free_frames(), 5)
        bpm.new_page()
        self.assertEqual(bpm.num_free_frames(), 4)

    def test_num_pinned(self):
        bpm, dm = self._make_bpm(pool_size=5)
        bpm.new_page()
        bpm.new_page()
        self.assertEqual(bpm.num_pinned(), 2)


# ===========================================================================
# Buffer Pool Manager -- Flush Tests
# ===========================================================================

class TestBufferPoolFlush(unittest.TestCase):

    def _make_bpm(self, pool_size=10):
        dm = DiskManager()
        return BufferPoolManager(pool_size, dm), dm

    def test_flush_dirty_page(self):
        bpm, dm = self._make_bpm()
        pid, page = bpm.new_page()
        page.write(0, b'dirty data')
        bpm.unpin_page(pid, is_dirty=True)
        bpm.flush_page(pid)

        # Read from disk to verify
        disk_data = dm.read_page(pid)
        self.assertEqual(disk_data[:10], b'dirty data')

    def test_flush_clean_page(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid)
        initial_writes = dm.write_count
        bpm.flush_page(pid)
        # Clean page should not cause write
        self.assertEqual(dm.write_count, initial_writes)

    def test_flush_nonexistent(self):
        bpm, dm = self._make_bpm()
        self.assertFalse(bpm.flush_page(999))

    def test_flush_all(self):
        bpm, dm = self._make_bpm()
        for _ in range(5):
            pid, page = bpm.new_page()
            page.write(0, b'data')
            bpm.unpin_page(pid, is_dirty=True)

        bpm.flush_all()
        self.assertEqual(len(bpm.get_dirty_pages()), 0)

    def test_get_dirty_pages(self):
        bpm, dm = self._make_bpm()
        p1, _ = bpm.new_page()
        p2, _ = bpm.new_page()
        p3, _ = bpm.new_page()
        bpm.unpin_page(p1, is_dirty=True)
        bpm.unpin_page(p2, is_dirty=False)
        bpm.unpin_page(p3, is_dirty=True)
        dirty = bpm.get_dirty_pages()
        self.assertEqual(set(dirty), {p1, p3})

    def test_flush_clears_dirty_flag(self):
        bpm, dm = self._make_bpm()
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid, is_dirty=True)
        self.assertTrue(bpm.is_dirty(pid))
        bpm.flush_page(pid)
        self.assertFalse(bpm.is_dirty(pid))


# ===========================================================================
# Eviction Tests
# ===========================================================================

class TestEviction(unittest.TestCase):

    def test_lru_eviction(self):
        dm = DiskManager()
        bpm = BufferPoolManager(3, dm, EvictionPolicy.LRU)

        # Fill pool
        pages = []
        for _ in range(3):
            pid, _ = bpm.new_page()
            pages.append(pid)

        # Unpin all
        for pid in pages:
            bpm.unpin_page(pid)

        # Fetch page 0 to make it recently used
        bpm.fetch_page(pages[0])
        bpm.unpin_page(pages[0])

        # Allocate new page -- should evict page 1 (LRU)
        new_pid, _ = bpm.new_page()
        self.assertFalse(bpm.contains(pages[1]))  # Evicted
        self.assertTrue(bpm.contains(pages[0]))    # Recently used

    def test_eviction_writes_dirty(self):
        dm = DiskManager()
        bpm = BufferPoolManager(2, dm, EvictionPolicy.LRU)

        p0, page0 = bpm.new_page()
        page0.write(0, b'important')
        bpm.unpin_page(p0, is_dirty=True)

        p1, _ = bpm.new_page()
        bpm.unpin_page(p1)

        # New page should evict p0, writing dirty data
        p2, _ = bpm.new_page()
        disk_data = dm.read_page(p0)
        self.assertEqual(disk_data[:9], b'important')

    def test_all_pinned_no_eviction(self):
        dm = DiskManager()
        bpm = BufferPoolManager(2, dm, EvictionPolicy.LRU)

        bpm.new_page()
        bpm.new_page()
        # Both pinned, can't allocate more
        result = bpm.new_page()
        self.assertIsNone(result)

    def test_clock_eviction(self):
        dm = DiskManager()
        bpm = BufferPoolManager(3, dm, EvictionPolicy.CLOCK)

        pages = []
        for _ in range(3):
            pid, _ = bpm.new_page()
            pages.append(pid)
            bpm.unpin_page(pid)

        # Allocate new page -- should evict one of the existing
        new_pid, _ = bpm.new_page()
        evicted_count = sum(1 for p in pages if not bpm.contains(p))
        self.assertEqual(evicted_count, 1)

    def test_lru_k_eviction(self):
        dm = DiskManager()
        bpm = BufferPoolManager(3, dm, EvictionPolicy.LRU_K, k=2)

        pages = []
        for _ in range(3):
            pid, _ = bpm.new_page()
            pages.append(pid)

        for pid in pages:
            bpm.unpin_page(pid)

        # Access page 0 twice, page 1 once
        for _ in range(2):
            p = bpm.fetch_page(pages[0])
            bpm.unpin_page(pages[0])

        p = bpm.fetch_page(pages[1])
        bpm.unpin_page(pages[1])

        # New page should evict one with fewer accesses (not page 0)
        new_pid, _ = bpm.new_page()
        self.assertTrue(bpm.contains(pages[0]))

    def test_eviction_count(self):
        dm = DiskManager()
        bpm = BufferPoolManager(2, dm, EvictionPolicy.LRU)

        p0, _ = bpm.new_page()
        bpm.unpin_page(p0)
        p1, _ = bpm.new_page()
        bpm.unpin_page(p1)

        p2, _ = bpm.new_page()  # Evicts one
        self.assertEqual(bpm.stats.evictions, 1)

    def test_data_persists_through_eviction(self):
        dm = DiskManager()
        bpm = BufferPoolManager(1, dm, EvictionPolicy.LRU)

        p0, page0 = bpm.new_page()
        page0.write(0, b'data0')
        bpm.unpin_page(p0, is_dirty=True)

        p1, page1 = bpm.new_page()  # Evicts p0
        page1.write(0, b'data1')
        bpm.unpin_page(p1, is_dirty=True)

        # Fetch p0 back -- should read from disk
        page0_again = bpm.fetch_page(p0)
        self.assertEqual(page0_again.read(0, 5), b'data0')


# ===========================================================================
# Statistics Tests
# ===========================================================================

class TestBufferPoolStats(unittest.TestCase):

    def test_stats_creation(self):
        s = BufferPoolStats()
        self.assertEqual(s.total_requests, 0)
        self.assertEqual(s.hit_rate, 0.0)

    def test_hit_rate(self):
        s = BufferPoolStats(hits=80, misses=20)
        self.assertAlmostEqual(s.hit_rate, 0.8)

    def test_reset(self):
        s = BufferPoolStats(hits=10, misses=5, evictions=3)
        s.reset()
        self.assertEqual(s.total_requests, 0)

    def test_stats_repr(self):
        s = BufferPoolStats(hits=10, misses=5)
        self.assertIn('hit_rate', repr(s))

    def test_bpm_stats_tracking(self):
        dm = DiskManager()
        bpm = BufferPoolManager(5, dm)

        pid, _ = bpm.new_page()
        bpm.unpin_page(pid)

        # First fetch is a miss (but we did new_page which is also a miss pattern)
        bpm.fetch_page(pid)  # Should be a hit (already in pool)
        self.assertEqual(bpm.stats.hits, 1)

        bpm.unpin_page(pid)

    def test_reset_stats(self):
        dm = DiskManager()
        bpm = BufferPoolManager(5, dm)
        bpm.new_page()
        bpm.reset_stats()
        stats = bpm.get_stats()
        self.assertEqual(stats.total_requests, 0)

    def test_get_stats_copy(self):
        dm = DiskManager()
        bpm = BufferPoolManager(5, dm)
        pid = dm.allocate_page()
        bpm.fetch_page(pid)  # miss
        stats1 = bpm.get_stats()
        bpm.unpin_page(pid)
        bpm.fetch_page(pid)  # hit
        stats2 = bpm.get_stats()
        # stats1 should not change (it's a copy)
        self.assertNotEqual(stats1.hits, stats2.hits)

    def test_miss_tracking(self):
        dm = DiskManager()
        bpm = BufferPoolManager(5, dm)
        pid = dm.allocate_page()
        bpm.fetch_page(pid)
        self.assertEqual(bpm.stats.misses, 1)

    def test_dirty_write_tracking(self):
        dm = DiskManager()
        bpm = BufferPoolManager(5, dm)
        pid, _ = bpm.new_page()
        bpm.unpin_page(pid, is_dirty=True)
        bpm.flush_page(pid)
        self.assertEqual(bpm.stats.dirty_writes, 1)


# ===========================================================================
# Prefetch Tests
# ===========================================================================

class TestPrefetch(unittest.TestCase):

    def test_basic_prefetch(self):
        dm = DiskManager()
        for _ in range(10):
            dm.allocate_page()

        bpm = BufferPoolManager(10, dm)
        bpm.prefetch(0, 5)

        # Pages should be in pool but not pinned
        for pid in range(5):
            self.assertTrue(bpm.contains(pid))
            self.assertEqual(bpm.get_pin_count(pid), 0)

    def test_prefetch_only_uses_free_frames(self):
        dm = DiskManager()
        for _ in range(10):
            dm.allocate_page()

        bpm = BufferPoolManager(3, dm)
        # Fill 2 frames
        bpm.new_page()  # Uses page from disk.allocate, takes frame
        # Actually let's use fetch:
        bpm2 = BufferPoolManager(3, dm)
        bpm2.prefetch(0, 10)
        # Should only prefetch 3 (pool size)
        count = sum(1 for pid in range(10) if bpm2.contains(pid))
        self.assertLessEqual(count, 3)

    def test_prefetch_existing_page_counted_as_hit(self):
        dm = DiskManager()
        for _ in range(5):
            dm.allocate_page()

        bpm = BufferPoolManager(10, dm)
        bpm.fetch_page(0)
        bpm.unpin_page(0)

        bpm.prefetch(0, 3)
        self.assertGreater(bpm.stats.prefetch_hits, 0)

    def test_prefetch_nonexistent_stops(self):
        dm = DiskManager()
        dm.allocate_page()  # Only page 0

        bpm = BufferPoolManager(10, dm)
        bpm.prefetch(0, 5)
        self.assertTrue(bpm.contains(0))
        self.assertFalse(bpm.contains(1))

    def test_set_prefetch_distance(self):
        dm = DiskManager()
        bpm = BufferPoolManager(10, dm)
        bpm.set_prefetch_distance(8)
        self.assertEqual(bpm._prefetch_distance, 8)

    def test_set_prefetch_distance_minimum(self):
        dm = DiskManager()
        bpm = BufferPoolManager(10, dm)
        bpm.set_prefetch_distance(0)
        self.assertEqual(bpm._prefetch_distance, 1)

    def test_prefetch_stats(self):
        dm = DiskManager()
        for _ in range(5):
            dm.allocate_page()

        bpm = BufferPoolManager(10, dm)
        bpm.prefetch(0, 5)
        self.assertEqual(bpm.stats.prefetch_requests, 5)


# ===========================================================================
# Scan Buffer Tests
# ===========================================================================

class TestScanBuffer(unittest.TestCase):

    def _setup(self, num_pages=10, pool_size=5):
        dm = DiskManager()
        for i in range(num_pages):
            pid = dm.allocate_page()
            dm.write_page(pid, bytes([i]) * PAGE_SIZE)
        bpm = BufferPoolManager(pool_size, dm)
        return bpm, dm

    def test_sequential_scan(self):
        bpm, dm = self._setup(num_pages=5, pool_size=10)
        scan = ScanBuffer(bpm, 0, 5)

        pages_read = []
        while True:
            page = scan.next_page()
            if page is None:
                break
            pages_read.append(page.page_id)
            scan.unpin_current()

        self.assertEqual(pages_read, [0, 1, 2, 3, 4])

    def test_pages_remaining(self):
        bpm, dm = self._setup(num_pages=5, pool_size=10)
        scan = ScanBuffer(bpm, 0, 5)
        self.assertEqual(scan.pages_remaining, 5)
        scan.next_page()
        self.assertEqual(scan.pages_remaining, 4)

    def test_current_page_id(self):
        bpm, dm = self._setup(num_pages=5, pool_size=10)
        scan = ScanBuffer(bpm, 0, 5)
        self.assertEqual(scan.current_page_id, -1)
        scan.next_page()
        self.assertEqual(scan.current_page_id, 0)

    def test_reset(self):
        bpm, dm = self._setup(num_pages=5, pool_size=10)
        scan = ScanBuffer(bpm, 0, 5)
        scan.next_page()
        scan.unpin_current()
        scan.next_page()
        scan.unpin_current()
        scan.reset()
        self.assertEqual(scan.pages_remaining, 5)

    def test_empty_range(self):
        bpm, dm = self._setup(num_pages=5, pool_size=10)
        scan = ScanBuffer(bpm, 3, 3)
        self.assertIsNone(scan.next_page())

    def test_prefetch_triggered(self):
        bpm, dm = self._setup(num_pages=20, pool_size=20)
        scan = ScanBuffer(bpm, 0, 20, prefetch_distance=4)

        # Reading first page should trigger prefetch
        page = scan.next_page()
        self.assertIsNotNone(page)
        scan.unpin_current()

    def test_data_integrity(self):
        bpm, dm = self._setup(num_pages=3, pool_size=10)
        scan = ScanBuffer(bpm, 0, 3)

        for i in range(3):
            page = scan.next_page()
            self.assertEqual(page.data[0], i)
            scan.unpin_current()


# ===========================================================================
# Thread Safety Tests
# ===========================================================================

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_new_page(self):
        dm = DiskManager()
        bpm = BufferPoolManager(100, dm)
        results = []
        errors = []

        def create_page():
            try:
                result = bpm.new_page()
                if result:
                    pid, page = result
                    results.append(pid)
                    bpm.unpin_page(pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_page) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        # All page IDs should be unique
        self.assertEqual(len(results), len(set(results)))

    def test_concurrent_fetch_and_unpin(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        bpm = BufferPoolManager(10, dm)
        errors = []

        def fetch_unpin():
            try:
                page = bpm.fetch_page(pid)
                if page:
                    bpm.unpin_page(pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fetch_unpin) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(bpm.get_pin_count(pid), 0)

    def test_concurrent_mixed_operations(self):
        dm = DiskManager()
        for _ in range(50):
            dm.allocate_page()
        bpm = BufferPoolManager(20, dm)
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    pid = (worker_id * 10 + i) % 50
                    page = bpm.fetch_page(pid)
                    if page:
                        bpm.unpin_page(pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration(unittest.TestCase):

    def test_full_lifecycle(self):
        """Create, write, flush, evict, re-fetch, verify."""
        dm = DiskManager()
        bpm = BufferPoolManager(2, dm, EvictionPolicy.LRU)

        # Create and write
        p0, page0 = bpm.new_page()
        page0.write(0, b'page zero')
        bpm.unpin_page(p0, is_dirty=True)

        p1, page1 = bpm.new_page()
        page1.write(0, b'page one')
        bpm.unpin_page(p1, is_dirty=True)

        # Pool is full. Create another -- evicts p0
        p2, page2 = bpm.new_page()
        page2.write(0, b'page two')
        bpm.unpin_page(p2, is_dirty=True)

        self.assertFalse(bpm.contains(p0))
        self.assertEqual(bpm.stats.evictions, 1)

        # Flush remaining
        bpm.flush_all()

        # Unpin and re-fetch p0
        bpm.unpin_page(p1)  # Already unpinned, returns False
        # Need to evict to make room
        page0_back = bpm.fetch_page(p0)
        self.assertEqual(page0_back.read(0, 9), b'page zero')

    def test_working_set_fits_in_pool(self):
        """When working set fits in pool, no evictions needed."""
        dm = DiskManager()
        for _ in range(5):
            dm.allocate_page()

        bpm = BufferPoolManager(10, dm)

        # Access all 5 pages repeatedly
        for _ in range(10):
            for pid in range(5):
                page = bpm.fetch_page(pid)
                bpm.unpin_page(pid)

        self.assertEqual(bpm.stats.evictions, 0)
        # Only first 5 fetches are misses
        self.assertEqual(bpm.stats.misses, 5)
        self.assertEqual(bpm.stats.hits, 45)

    def test_thrashing_pattern(self):
        """Access pattern larger than pool -- high eviction rate."""
        dm = DiskManager()
        for _ in range(10):
            dm.allocate_page()

        bpm = BufferPoolManager(3, dm)

        # Sequential access over 10 pages with pool of 3
        for _ in range(3):
            for pid in range(10):
                page = bpm.fetch_page(pid)
                bpm.unpin_page(pid)

        self.assertGreater(bpm.stats.evictions, 0)
        self.assertLess(bpm.stats.hit_rate, 0.5)

    def test_hot_cold_access(self):
        """Hot pages should stay in pool, cold pages get evicted."""
        dm = DiskManager()
        for _ in range(20):
            dm.allocate_page()

        bpm = BufferPoolManager(5, dm, EvictionPolicy.LRU)

        # Access hot pages frequently
        hot_pages = [0, 1, 2]
        cold_pages = list(range(3, 20))

        for _ in range(10):
            # Access hot pages
            for pid in hot_pages:
                page = bpm.fetch_page(pid)
                bpm.unpin_page(pid)

            # Access one cold page
            cold = cold_pages[_ % len(cold_pages)]
            page = bpm.fetch_page(cold)
            bpm.unpin_page(cold)

        # Hot pages should still be in pool
        for pid in hot_pages:
            self.assertTrue(bpm.contains(pid))

    def test_sequential_scan_with_prefetch(self):
        dm = DiskManager()
        for i in range(100):
            pid = dm.allocate_page()
            dm.write_page(pid, bytes([i % 256]) * PAGE_SIZE)

        bpm = BufferPoolManager(20, dm, EvictionPolicy.CLOCK)
        scan = ScanBuffer(bpm, 0, 100, prefetch_distance=8)

        count = 0
        while True:
            page = scan.next_page()
            if page is None:
                break
            count += 1
            scan.unpin_current()

        self.assertEqual(count, 100)

    def test_multiple_readers(self):
        """Multiple readers of the same page."""
        dm = DiskManager()
        pid = dm.allocate_page()
        dm.write_page(pid, b'shared' + b'\x00' * (PAGE_SIZE - 6))

        bpm = BufferPoolManager(5, dm)

        # Fetch same page 3 times
        p1 = bpm.fetch_page(pid)
        p2 = bpm.fetch_page(pid)
        p3 = bpm.fetch_page(pid)

        self.assertEqual(bpm.get_pin_count(pid), 3)
        self.assertIs(p1, p2)  # Same page object
        self.assertIs(p2, p3)

        bpm.unpin_page(pid)
        self.assertEqual(bpm.get_pin_count(pid), 2)
        bpm.unpin_page(pid)
        bpm.unpin_page(pid)
        self.assertEqual(bpm.get_pin_count(pid), 0)

    def test_eviction_policy_comparison(self):
        """All eviction policies should correctly manage the pool."""
        for policy in EvictionPolicy:
            dm = DiskManager()
            for _ in range(10):
                dm.allocate_page()

            bpm = BufferPoolManager(3, dm, policy, k=2)

            for pid in range(10):
                page = bpm.fetch_page(pid)
                bpm.unpin_page(pid)

            # Pool should have exactly 3 pages
            in_pool = sum(1 for pid in range(10) if bpm.contains(pid))
            self.assertEqual(in_pool, 3, f"Policy {policy} has {in_pool} pages")

    def test_dirty_eviction_persistence(self):
        """Dirty pages evicted should be readable from disk."""
        dm = DiskManager()
        bpm = BufferPoolManager(1, dm, EvictionPolicy.LRU)

        pages_data = {}
        for i in range(5):
            pid, page = bpm.new_page()
            data = bytes([i]) * PAGE_SIZE
            page.data[:] = data
            pages_data[pid] = data
            bpm.unpin_page(pid, is_dirty=True)

        # Flush remaining in-pool pages to disk
        bpm.flush_all()

        # All should be on disk
        for pid, expected in pages_data.items():
            disk_data = dm.read_page(pid)
            self.assertEqual(disk_data, expected, f"Page {pid} data mismatch")


if __name__ == '__main__':
    unittest.main()
