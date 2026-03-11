"""
Tests for C117: LSM Tree (Log-Structured Merge Tree)
=====================================================
Tests organized by component:
1. BloomFilter
2. WAL
3. SSTable
4. MemTable
5. Merge iterator
6. LSMTree core (put/get/delete)
7. LSMTree flush and compaction
8. LSMTree range queries
9. LSMTree snapshots and recovery
10. LSMTreeMap dict interface
11. Stress / integration tests
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from lsm_tree import (
    BloomFilter, WAL, WALEntry, SSTable, MemTable, Level,
    LSMTree, LSMTreeMap, _TOMBSTONE, _merge_iterators
)


# ===================================================================
# 1. BloomFilter
# ===================================================================

class TestBloomFilter:
    def test_basic_add_and_check(self):
        bf = BloomFilter(expected_items=100)
        bf.add("hello")
        bf.add("world")
        assert bf.might_contain("hello")
        assert bf.might_contain("world")

    def test_definitely_not_present(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.001)
        bf.add("apple")
        bf.add("banana")
        # These should almost certainly return False
        missing = sum(1 for i in range(100) if not bf.might_contain(f"missing_{i}"))
        assert missing > 90  # At most ~1% false positive rate

    def test_integer_keys(self):
        bf = BloomFilter(expected_items=50)
        for i in range(50):
            bf.add(i)
        for i in range(50):
            assert bf.might_contain(i)

    def test_count(self):
        bf = BloomFilter()
        assert bf.count == 0
        bf.add("a")
        bf.add("b")
        assert bf.count == 2

    def test_repr(self):
        bf = BloomFilter(expected_items=10)
        assert "BloomFilter" in repr(bf)

    def test_small_expected_items(self):
        bf = BloomFilter(expected_items=0)
        bf.add("x")
        assert bf.might_contain("x")

    def test_low_fp_rate(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.0001)
        for i in range(100):
            bf.add(i)
        # Very low FP rate
        fps = sum(1 for i in range(1000, 2000) if bf.might_contain(i))
        assert fps < 20  # Should be very few


# ===================================================================
# 2. WAL (Write-Ahead Log)
# ===================================================================

class TestWAL:
    def test_append_put(self):
        wal = WAL()
        seq = wal.append_put("key1", "val1")
        assert seq == 1
        assert wal.size == 1

    def test_append_delete(self):
        wal = WAL()
        seq = wal.append_delete("key1")
        assert seq == 1

    def test_sequential_seqs(self):
        wal = WAL()
        s1 = wal.append_put("a", 1)
        s2 = wal.append_delete("b")
        s3 = wal.append_put("c", 3)
        assert s1 == 1
        assert s2 == 2
        assert s3 == 3
        assert wal.seq == 3

    def test_entries(self):
        wal = WAL()
        wal.append_put("x", 10)
        wal.append_delete("y")
        entries = wal.entries()
        assert len(entries) == 2
        assert entries[0].op == 'put'
        assert entries[0].key == 'x'
        assert entries[0].value == 10
        assert entries[1].op == 'delete'
        assert entries[1].key == 'y'

    def test_truncate(self):
        wal = WAL()
        wal.append_put("a", 1)
        wal.append_put("b", 2)
        wal.truncate()
        assert wal.size == 0
        assert wal.entries() == []
        # Seq continues
        s = wal.append_put("c", 3)
        assert s == 3

    def test_replay(self):
        wal = WAL()
        wal.append_put("k1", "v1")
        wal.append_delete("k2")
        replay = wal.replay()
        assert len(replay) == 2

    def test_repr(self):
        wal = WAL()
        assert "WAL" in repr(wal)

    def test_entry_repr(self):
        e1 = WALEntry(1, 'put', 'k', 'v')
        assert "PUT" in repr(e1)
        e2 = WALEntry(2, 'delete', 'k')
        assert "DEL" in repr(e2)


# ===================================================================
# 3. SSTable
# ===================================================================

class TestSSTable:
    def test_create_and_lookup(self):
        data = [(1, "a"), (3, "c"), (5, "e"), (7, "g")]
        sst = SSTable(data, table_id=1)
        found, val = sst.get(3)
        assert found and val == "c"
        found, val = sst.get(5)
        assert found and val == "e"

    def test_key_not_found(self):
        data = [(1, "a"), (3, "c"), (5, "e")]
        sst = SSTable(data, table_id=1)
        found, val = sst.get(2)
        assert not found
        found, val = sst.get(6)
        assert not found

    def test_bloom_filter_integration(self):
        data = [(i, f"v{i}") for i in range(100)]
        sst = SSTable(data, table_id=1)
        # Keys that exist should be found
        for i in range(100):
            found, val = sst.get(i)
            assert found

    def test_min_max_keys(self):
        data = [(10, "a"), (20, "b"), (30, "c")]
        sst = SSTable(data, table_id=1)
        assert sst.min_key == 10
        assert sst.max_key == 30

    def test_size(self):
        data = [(1, "a"), (2, "b"), (3, "c")]
        sst = SSTable(data, table_id=1)
        assert sst.size == 3
        assert len(sst) == 3

    def test_overlaps(self):
        data = [(10, "a"), (20, "b"), (30, "c")]
        sst = SSTable(data, table_id=1)
        assert sst.overlaps(15, 25)
        assert sst.overlaps(5, 15)
        assert sst.overlaps(25, 35)
        assert sst.overlaps(5, 35)
        assert not sst.overlaps(31, 40)
        assert not sst.overlaps(1, 9)

    def test_scan_full(self):
        data = [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
        sst = SSTable(data, table_id=1)
        items = list(sst.scan())
        assert items == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]

    def test_scan_range(self):
        data = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
        sst = SSTable(data, table_id=1)
        items = list(sst.scan(low=2, high=4))
        assert items == [(2, "b"), (3, "c"), (4, "d")]

    def test_tombstone_entry(self):
        data = [(1, "a"), (2, _TOMBSTONE), (3, "c")]
        sst = SSTable(data, table_id=1)
        found, val = sst.get(2)
        assert found and val is _TOMBSTONE

    def test_tombstone_count(self):
        data = [(1, "a"), (2, _TOMBSTONE), (3, _TOMBSTONE), (4, "d")]
        sst = SSTable(data, table_id=1)
        assert sst.tombstone_count() == 2

    def test_live_items(self):
        data = [(1, "a"), (2, _TOMBSTONE), (3, "c")]
        sst = SSTable(data, table_id=1)
        live = list(sst.live_items())
        assert live == [(1, "a"), (3, "c")]

    def test_all_items(self):
        data = [(1, "a"), (2, _TOMBSTONE), (3, "c")]
        sst = SSTable(data, table_id=1)
        items = list(sst.all_items())
        assert len(items) == 3

    def test_repr(self):
        data = [(1, "a")]
        sst = SSTable(data, table_id=42)
        assert "42" in repr(sst)

    def test_empty_sstable(self):
        sst = SSTable([], table_id=1)
        assert sst.size == 0
        found, _ = sst.get(1)
        assert not found
        assert not sst.overlaps(0, 100)
        assert list(sst.scan()) == []

    def test_single_entry(self):
        sst = SSTable([(5, "five")], table_id=1)
        found, val = sst.get(5)
        assert found and val == "five"
        found, _ = sst.get(4)
        assert not found


# ===================================================================
# 4. MemTable
# ===================================================================

class TestMemTable:
    def test_put_and_get(self):
        mt = MemTable(max_size=100)
        mt.put(1, "a")
        mt.put(2, "b")
        found, val = mt.get(1)
        assert found and val == "a"

    def test_get_missing(self):
        mt = MemTable(max_size=100)
        found, val = mt.get(999)
        assert not found

    def test_delete_writes_tombstone(self):
        mt = MemTable(max_size=100)
        mt.put(1, "a")
        mt.delete(1)
        found, val = mt.get(1)
        assert found and val is _TOMBSTONE

    def test_is_full(self):
        mt = MemTable(max_size=3)
        mt.put(1, "a")
        mt.put(2, "b")
        assert not mt.is_full()
        mt.put(3, "c")
        assert mt.is_full()

    def test_flush_to_sstable(self):
        mt = MemTable(max_size=100)
        mt.put(3, "c")
        mt.put(1, "a")
        mt.put(2, "b")
        sst = mt.flush_to_sstable(0, 3, table_id=0)
        assert sst.size == 3
        found, val = sst.get(2)
        assert found and val == "b"

    def test_items_sorted(self):
        mt = MemTable(max_size=100)
        mt.put(3, "c")
        mt.put(1, "a")
        mt.put(2, "b")
        items = list(mt.items())
        keys = [k for k, v in items]
        assert keys == [1, 2, 3]

    def test_clear(self):
        mt = MemTable(max_size=100)
        mt.put(1, "a")
        mt.clear()
        assert mt.size == 0

    def test_contains(self):
        mt = MemTable(max_size=100)
        mt.put(1, "a")
        assert 1 in mt
        assert 2 not in mt

    def test_update_existing_key(self):
        mt = MemTable(max_size=100)
        mt.put(1, "a")
        mt.put(1, "b")
        found, val = mt.get(1)
        assert val == "b"

    def test_repr(self):
        mt = MemTable(max_size=100)
        assert "MemTable" in repr(mt)


# ===================================================================
# 5. Level
# ===================================================================

class TestLevel:
    def test_add_and_get(self):
        level = Level(0, max_tables=4)
        sst = SSTable([(1, "a"), (2, "b")], table_id=0)
        level.add_table(sst)
        found, val = level.get(1)
        assert found and val == "a"

    def test_multiple_tables(self):
        level = Level(0, max_tables=4)
        sst1 = SSTable([(1, "a")], table_id=0)
        sst2 = SSTable([(1, "b")], table_id=1)  # Newer, same key
        level.add_table(sst1)
        level.add_table(sst2)
        # Newest table (last added) should win
        found, val = level.get(1)
        assert found and val == "b"

    def test_needs_compaction(self):
        level = Level(0, max_tables=2)
        assert not level.needs_compaction()
        level.add_table(SSTable([(1, "a")], table_id=0))
        level.add_table(SSTable([(2, "b")], table_id=1))
        assert not level.needs_compaction()
        level.add_table(SSTable([(3, "c")], table_id=2))
        assert level.needs_compaction()

    def test_overlapping_tables(self):
        level = Level(0, max_tables=10)
        sst1 = SSTable([(1, "a"), (5, "e")], table_id=0)
        sst2 = SSTable([(10, "j"), (15, "o")], table_id=1)
        level.add_table(sst1)
        level.add_table(sst2)
        overlapping = level.overlapping_tables(3, 7)
        assert len(overlapping) == 1
        assert overlapping[0] is sst1

    def test_remove_table(self):
        level = Level(0, max_tables=10)
        sst = SSTable([(1, "a")], table_id=0)
        level.add_table(sst)
        level.remove_table(sst)
        assert len(level) == 0

    def test_total_entries(self):
        level = Level(0, max_tables=10)
        level.add_table(SSTable([(1, "a"), (2, "b")], table_id=0))
        level.add_table(SSTable([(3, "c")], table_id=1))
        assert level.total_entries() == 3

    def test_repr(self):
        level = Level(0, max_tables=4)
        assert "Level" in repr(level)


# ===================================================================
# 6. Merge iterator
# ===================================================================

class TestMergeIterators:
    def test_single_iterator(self):
        it = iter([(1, "a"), (2, "b"), (3, "c")])
        result = list(_merge_iterators([it]))
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_two_non_overlapping(self):
        it1 = iter([(1, "a"), (3, "c")])
        it2 = iter([(2, "b"), (4, "d")])
        result = list(_merge_iterators([it1, it2]))
        assert result == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]

    def test_duplicate_key_first_wins(self):
        it1 = iter([(1, "new")])
        it2 = iter([(1, "old")])
        result = list(_merge_iterators([it1, it2]))
        assert result == [(1, "new")]

    def test_empty_iterators(self):
        result = list(_merge_iterators([iter([]), iter([])]))
        assert result == []

    def test_no_iterators(self):
        result = list(_merge_iterators([]))
        assert result == []

    def test_complex_merge(self):
        it1 = iter([(1, "a"), (3, "c"), (5, "e")])
        it2 = iter([(2, "b"), (3, "C"), (4, "d")])
        it3 = iter([(1, "A"), (6, "f")])
        result = list(_merge_iterators([it1, it2, it3]))
        keys = [k for k, v in result]
        assert keys == [1, 2, 3, 4, 5, 6]
        # it1 wins for key 1 and 3
        assert result[0] == (1, "a")
        assert result[2] == (3, "c")


# ===================================================================
# 7. LSMTree core (put/get/delete)
# ===================================================================

class TestLSMTreeCore:
    def test_basic_put_get(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("key1", "val1")
        assert lsm.get("key1") == "val1"

    def test_put_overwrite(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("key1", "v1")
        lsm.put("key1", "v2")
        assert lsm.get("key1") == "v2"

    def test_get_missing(self):
        lsm = LSMTree(memtable_size=100)
        assert lsm.get("nope") is None
        assert lsm.get("nope", "default") == "default"

    def test_delete(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("key1", "val1")
        lsm.delete("key1")
        assert lsm.get("key1") is None

    def test_contains(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("key1", "val1")
        assert "key1" in lsm
        assert "key2" not in lsm

    def test_getitem_setitem(self):
        lsm = LSMTree(memtable_size=100)
        lsm["a"] = 1
        assert lsm["a"] == 1

    def test_getitem_missing_raises(self):
        lsm = LSMTree(memtable_size=100)
        with pytest.raises(KeyError):
            _ = lsm["missing"]

    def test_delitem(self):
        lsm = LSMTree(memtable_size=100)
        lsm["a"] = 1
        del lsm["a"]
        assert "a" not in lsm

    def test_integer_keys(self):
        lsm = LSMTree(memtable_size=100)
        for i in range(50):
            lsm.put(i, i * 10)
        for i in range(50):
            assert lsm.get(i) == i * 10

    def test_delete_nonexistent_is_safe(self):
        lsm = LSMTree(memtable_size=100)
        lsm.delete("ghost")  # Should not error
        assert lsm.get("ghost") is None

    def test_put_after_delete(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("k", "v1")
        lsm.delete("k")
        lsm.put("k", "v2")
        assert lsm.get("k") == "v2"

    def test_stats_tracking(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put("a", 1)
        lsm.put("b", 2)
        lsm.get("a")
        lsm.get("missing")
        lsm.delete("a")
        stats = lsm.stats
        assert stats['puts'] == 2
        assert stats['deletes'] == 1
        assert stats['gets'] == 2
        assert stats['get_hits'] == 1
        assert stats['get_misses'] == 1

    def test_no_wal_mode(self):
        lsm = LSMTree(memtable_size=100, enable_wal=False)
        lsm.put("a", 1)
        assert lsm.get("a") == 1
        assert lsm.wal is None


# ===================================================================
# 8. LSMTree flush and compaction
# ===================================================================

class TestLSMTreeFlushCompaction:
    def test_auto_flush(self):
        lsm = LSMTree(memtable_size=5)
        for i in range(6):
            lsm.put(i, f"v{i}")
        # Memtable should have flushed
        assert lsm.stats['flushes'] >= 1
        # All values still accessible
        for i in range(6):
            assert lsm.get(i) == f"v{i}"

    def test_force_flush(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(2, "b")
        lsm.force_flush()
        assert lsm.stats['flushes'] == 1
        assert len(lsm.levels[0]) == 1
        # Still readable
        assert lsm.get(1) == "a"

    def test_multiple_flushes(self):
        lsm = LSMTree(memtable_size=5, level0_max=10)
        for batch in range(3):
            for i in range(5):
                lsm.put(batch * 10 + i, f"b{batch}v{i}")
            lsm.force_flush()
        assert lsm.stats['flushes'] == 3
        assert len(lsm.levels[0]) == 3

    def test_compaction_triggered(self):
        lsm = LSMTree(memtable_size=5, level0_max=2)
        # Insert enough to trigger multiple flushes and compaction
        for i in range(20):
            lsm.put(i, f"v{i}")
        # Should have triggered at least one compaction
        assert lsm.stats['compactions'] >= 1
        # All data still accessible
        for i in range(20):
            assert lsm.get(i) == f"v{i}"

    def test_full_compact(self):
        lsm = LSMTree(memtable_size=5, level0_max=2)
        for i in range(30):
            lsm.put(i, f"v{i}")
        lsm.full_compact()
        # All data still accessible
        for i in range(30):
            assert lsm.get(i) == f"v{i}"

    def test_tombstone_cleanup_at_bottom_level(self):
        lsm = LSMTree(memtable_size=3, num_levels=2, level0_max=2)
        lsm.put(1, "a")
        lsm.put(2, "b")
        lsm.put(3, "c")
        lsm.force_flush()
        lsm.delete(2)
        lsm.force_flush()
        # Compact L0 -> L1 (bottom level)
        lsm.force_compact()
        # Key 2 should still appear deleted
        assert lsm.get(2) is None

    def test_force_compact(self):
        lsm = LSMTree(memtable_size=5, level0_max=10)
        for i in range(15):
            lsm.put(i, i)
        lsm.force_flush()
        lsm.force_compact()
        for i in range(15):
            assert lsm.get(i) == i

    def test_delete_then_flush_then_read(self):
        lsm = LSMTree(memtable_size=5)
        lsm.put(1, "a")
        lsm.put(2, "b")
        lsm.force_flush()
        lsm.delete(1)
        lsm.force_flush()
        assert lsm.get(1) is None
        assert lsm.get(2) == "b"

    def test_level_info(self):
        lsm = LSMTree(memtable_size=5)
        for i in range(6):
            lsm.put(i, i)
        info = lsm.level_info()
        assert len(info) == 4
        assert info[0]['level'] == 0

    def test_overwrite_across_flush(self):
        lsm = LSMTree(memtable_size=5)
        lsm.put(1, "old")
        lsm.force_flush()
        lsm.put(1, "new")
        assert lsm.get(1) == "new"


# ===================================================================
# 9. LSMTree range queries
# ===================================================================

class TestLSMTreeRange:
    def test_range_memtable_only(self):
        lsm = LSMTree(memtable_size=100)
        for i in range(10):
            lsm.put(i, f"v{i}")
        result = lsm.range_query(3, 7)
        keys = [k for k, v in result]
        assert keys == [3, 4, 5, 6, 7]

    def test_range_across_levels(self):
        lsm = LSMTree(memtable_size=5, level0_max=10)
        for i in range(20):
            lsm.put(i, f"v{i}")
        result = lsm.range_query(5, 15)
        keys = [k for k, v in result]
        assert keys == list(range(5, 16))

    def test_range_unbounded(self):
        lsm = LSMTree(memtable_size=100)
        for i in range(5):
            lsm.put(i, i)
        result = lsm.range_query()
        assert len(result) == 5

    def test_range_excludes_tombstones(self):
        lsm = LSMTree(memtable_size=100)
        for i in range(5):
            lsm.put(i, f"v{i}")
        lsm.delete(2)
        result = lsm.range_query(0, 4)
        keys = [k for k, v in result]
        assert 2 not in keys
        assert len(keys) == 4

    def test_items_keys_values(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(2, "b")
        lsm.put(3, "c")
        assert lsm.keys() == [1, 2, 3]
        assert lsm.values() == ["a", "b", "c"]
        items = lsm.items()
        assert items == [(1, "a"), (2, "b"), (3, "c")]

    def test_range_after_compaction(self):
        lsm = LSMTree(memtable_size=5, level0_max=2)
        for i in range(30):
            lsm.put(i, i * 10)
        lsm.full_compact()
        result = lsm.range_query(10, 20)
        keys = [k for k, v in result]
        assert keys == list(range(10, 21))

    def test_range_empty_result(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(10, "j")
        result = lsm.range_query(5, 8)
        assert result == []


# ===================================================================
# 10. LSMTree snapshots and recovery
# ===================================================================

class TestLSMTreeSnapshotsRecovery:
    def test_create_snapshot(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(2, "b")
        snap_id = lsm.create_snapshot()
        assert snap_id == 0

    def test_read_snapshot(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(2, "b")
        snap = lsm.create_snapshot()
        # Modify after snapshot
        lsm.put(3, "c")
        lsm.delete(1)
        # Snapshot should have original data
        data = lsm.read_snapshot(snap)
        keys = [k for k, v in data]
        assert 1 in keys
        assert 2 in keys
        assert 3 not in keys

    def test_release_snapshot(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        snap = lsm.create_snapshot()
        lsm.release_snapshot(snap)
        with pytest.raises(KeyError):
            lsm.read_snapshot(snap)

    def test_multiple_snapshots(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        s1 = lsm.create_snapshot()
        lsm.put(2, "b")
        s2 = lsm.create_snapshot()
        d1 = lsm.read_snapshot(s1)
        d2 = lsm.read_snapshot(s2)
        assert len(d1) == 1
        assert len(d2) == 2

    def test_wal_recovery(self):
        lsm = LSMTree(memtable_size=100, enable_wal=True)
        lsm.put("a", 1)
        lsm.put("b", 2)
        # Simulate that WAL has entries but memtable was lost
        wal_entries = lsm.wal.replay()
        assert len(wal_entries) == 2

    def test_recover_from_wal(self):
        lsm = LSMTree(memtable_size=100, enable_wal=True)
        lsm.put("a", 1)
        lsm.put("b", 2)
        lsm.delete("a")
        # WAL has 3 entries
        assert lsm.wal.size == 3

    def test_release_nonexistent_snapshot(self):
        lsm = LSMTree(memtable_size=100)
        lsm.release_snapshot(999)  # Should not error


# ===================================================================
# 11. LSMTreeMap dict interface
# ===================================================================

class TestLSMTreeMap:
    def test_basic_operations(self):
        m = LSMTreeMap(memtable_size=100)
        m["a"] = 1
        m["b"] = 2
        assert m["a"] == 1
        assert m["b"] == 2
        assert len(m) == 2

    def test_delete(self):
        m = LSMTreeMap(memtable_size=100)
        m["a"] = 1
        del m["a"]
        assert len(m) == 0
        assert "a" not in m

    def test_delete_missing_raises(self):
        m = LSMTreeMap(memtable_size=100)
        with pytest.raises(KeyError):
            del m["missing"]

    def test_get_default(self):
        m = LSMTreeMap(memtable_size=100)
        assert m.get("x") is None
        assert m.get("x", 42) == 42

    def test_contains(self):
        m = LSMTreeMap(memtable_size=100)
        m["key"] = "val"
        assert "key" in m
        assert "other" not in m

    def test_bool(self):
        m = LSMTreeMap(memtable_size=100)
        assert not m
        m["a"] = 1
        assert m

    def test_items_keys_values(self):
        m = LSMTreeMap(memtable_size=100)
        m[1] = "a"
        m[2] = "b"
        m[3] = "c"
        assert m.keys() == [1, 2, 3]
        assert m.values() == ["a", "b", "c"]

    def test_update(self):
        m = LSMTreeMap(memtable_size=100)
        m.update({"a": 1, "b": 2})
        assert m["a"] == 1
        assert m["b"] == 2

    def test_update_from_pairs(self):
        m = LSMTreeMap(memtable_size=100)
        m.update([("x", 10), ("y", 20)])
        assert m["x"] == 10

    def test_clear(self):
        m = LSMTreeMap(memtable_size=100)
        m["a"] = 1
        m.clear()
        assert len(m) == 0

    def test_range_query(self):
        m = LSMTreeMap(memtable_size=100)
        for i in range(10):
            m[i] = i * 10
        result = m.range_query(3, 7)
        keys = [k for k, v in result]
        assert keys == [3, 4, 5, 6, 7]

    def test_overwrite_count(self):
        m = LSMTreeMap(memtable_size=100)
        m["a"] = 1
        m["a"] = 2
        assert len(m) == 1

    def test_put_and_delete_methods(self):
        m = LSMTreeMap(memtable_size=100)
        m.put("key", "val")
        assert m.get("key") == "val"
        m.delete("key")
        assert m.get("key") is None

    def test_stats(self):
        m = LSMTreeMap(memtable_size=100)
        m["a"] = 1
        assert m.stats['puts'] >= 1

    def test_repr(self):
        m = LSMTreeMap(memtable_size=100)
        assert "LSMTreeMap" in repr(m)

    def test_lsm_property(self):
        m = LSMTreeMap(memtable_size=100)
        assert isinstance(m.lsm, LSMTree)


# ===================================================================
# 12. Stress / integration tests
# ===================================================================

class TestLSMTreeStress:
    def test_1000_sequential_writes(self):
        lsm = LSMTree(memtable_size=50, level0_max=4)
        for i in range(1000):
            lsm.put(i, f"value_{i}")
        # Verify all readable
        for i in range(1000):
            assert lsm.get(i) == f"value_{i}"

    def test_write_delete_mixed(self):
        lsm = LSMTree(memtable_size=20, level0_max=3)
        # Write 100, delete evens
        for i in range(100):
            lsm.put(i, i * 10)
        for i in range(0, 100, 2):
            lsm.delete(i)
        # Verify
        for i in range(100):
            if i % 2 == 0:
                assert lsm.get(i) is None
            else:
                assert lsm.get(i) == i * 10

    def test_overwrite_stress(self):
        lsm = LSMTree(memtable_size=10, level0_max=3)
        # Write same keys multiple times
        for iteration in range(5):
            for i in range(20):
                lsm.put(i, iteration)
        # Should have latest values
        for i in range(20):
            assert lsm.get(i) == 4

    def test_range_after_heavy_writes(self):
        lsm = LSMTree(memtable_size=20, level0_max=3)
        for i in range(200):
            lsm.put(i, i)
        result = lsm.range_query(50, 150)
        keys = [k for k, v in result]
        assert keys == list(range(50, 151))

    def test_delete_all_then_rewrite(self):
        lsm = LSMTree(memtable_size=10, level0_max=3)
        for i in range(50):
            lsm.put(i, "old")
        for i in range(50):
            lsm.delete(i)
        # All should be gone
        for i in range(50):
            assert lsm.get(i) is None
        # Rewrite
        for i in range(50):
            lsm.put(i, "new")
        for i in range(50):
            assert lsm.get(i) == "new"

    def test_reverse_order_inserts(self):
        lsm = LSMTree(memtable_size=10, level0_max=3)
        for i in range(99, -1, -1):
            lsm.put(i, i)
        result = lsm.range_query(0, 99)
        keys = [k for k, v in result]
        assert keys == list(range(100))

    def test_string_keys(self):
        lsm = LSMTree(memtable_size=20)
        words = ["apple", "banana", "cherry", "date", "elderberry",
                 "fig", "grape", "honeydew", "kiwi", "lemon"]
        for w in words:
            lsm.put(w, len(w))
        for w in words:
            assert lsm.get(w) == len(w)
        # Range query on strings
        result = lsm.range_query("cherry", "grape")
        keys = [k for k, v in result]
        assert "cherry" in keys
        assert "date" in keys
        assert "elderberry" in keys
        assert "fig" in keys
        assert "grape" in keys

    def test_map_large_dataset(self):
        m = LSMTreeMap(memtable_size=50, level0_max=3)
        for i in range(500):
            m[i] = i * 2
        assert len(m) == 500
        for i in range(500):
            assert m[i] == i * 2

    def test_interleaved_read_write(self):
        lsm = LSMTree(memtable_size=10, level0_max=3)
        for i in range(100):
            lsm.put(i, i)
            if i > 0 and i % 10 == 0:
                # Read some previous values
                for j in range(max(0, i - 5), i):
                    assert lsm.get(j) == j

    def test_total_entries_property(self):
        lsm = LSMTree(memtable_size=10)
        for i in range(15):
            lsm.put(i, i)
        assert lsm.total_entries > 0

    def test_memtable_size_property(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "a")
        lsm.put(2, "b")
        assert lsm.memtable_size == 2

    def test_repr(self):
        lsm = LSMTree(memtable_size=10)
        assert "LSMTree" in repr(lsm)

    def test_snapshot_after_compaction(self):
        lsm = LSMTree(memtable_size=5, level0_max=2)
        for i in range(20):
            lsm.put(i, i)
        snap = lsm.create_snapshot()
        data = lsm.read_snapshot(snap)
        assert len(data) == 20

    def test_mixed_types_same_type_keys(self):
        """All keys must be same type for ordering."""
        lsm = LSMTree(memtable_size=100)
        lsm.put(1, "int1")
        lsm.put(2, "int2")
        lsm.put(3, "int3")
        assert lsm.get(1) == "int1"
        assert lsm.get(3) == "int3"

    def test_compaction_preserves_newest(self):
        """When compacting, newer values for same key should win."""
        lsm = LSMTree(memtable_size=3, level0_max=2, num_levels=3)
        # Write batch 1
        lsm.put(1, "old_a")
        lsm.put(2, "old_b")
        lsm.force_flush()
        # Write batch 2 (newer)
        lsm.put(1, "new_a")
        lsm.put(2, "new_b")
        lsm.force_flush()
        # Compact
        lsm.force_compact()
        assert lsm.get(1) == "new_a"
        assert lsm.get(2) == "new_b"

    def test_none_value_distinct_from_missing(self):
        """None as a value should be distinguishable from key not found."""
        lsm = LSMTree(memtable_size=100)
        lsm.put("key", None)
        assert "key" in lsm
        # get returns None for both, but contains tells them apart
        assert lsm.get("key") is None
        assert lsm.get("missing") is None
        assert "key" in lsm
        assert "missing" not in lsm

    def test_many_flushes_no_compaction(self):
        """Many small flushes without triggering compaction."""
        lsm = LSMTree(memtable_size=3, level0_max=100)
        for batch in range(20):
            for i in range(3):
                lsm.put(batch * 3 + i, batch)
            lsm.force_flush()
        assert lsm.stats['flushes'] == 20
        # Verify all data
        for i in range(60):
            assert lsm.get(i) is not None


# ===================================================================
# 13. Edge cases
# ===================================================================

class TestLSMTreeEdgeCases:
    def test_empty_tree_operations(self):
        lsm = LSMTree(memtable_size=100)
        assert lsm.get("any") is None
        assert lsm.items() == []
        assert lsm.keys() == []
        assert lsm.values() == []
        assert lsm.range_query(0, 100) == []

    def test_single_entry(self):
        lsm = LSMTree(memtable_size=100)
        lsm.put(42, "answer")
        assert lsm.get(42) == "answer"
        assert lsm.items() == [(42, "answer")]

    def test_flush_empty_memtable(self):
        lsm = LSMTree(memtable_size=100)
        lsm.force_flush()  # Nothing to flush
        assert lsm.stats['flushes'] == 0

    def test_compact_empty_level(self):
        lsm = LSMTree(memtable_size=100)
        lsm.force_compact()  # Nothing to compact
        assert lsm.stats['compactions'] == 0

    def test_very_small_memtable(self):
        lsm = LSMTree(memtable_size=1, level0_max=2)
        for i in range(10):
            lsm.put(i, i)
        for i in range(10):
            assert lsm.get(i) == i

    def test_delete_from_sstable(self):
        """Delete a key that's only in an SSTable (flushed)."""
        lsm = LSMTree(memtable_size=5)
        lsm.put(1, "a")
        lsm.put(2, "b")
        lsm.force_flush()
        lsm.delete(1)
        assert lsm.get(1) is None
        assert lsm.get(2) == "b"

    def test_range_with_tombstones_across_levels(self):
        lsm = LSMTree(memtable_size=5, level0_max=10)
        for i in range(10):
            lsm.put(i, i)
        lsm.force_flush()
        # Delete some
        lsm.delete(3)
        lsm.delete(7)
        result = lsm.range_query(0, 9)
        keys = [k for k, v in result]
        assert 3 not in keys
        assert 7 not in keys
        assert len(keys) == 8
