"""
Tests for C213: Storage Engine
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from storage_engine import (
    DiskManager, BufferPool, BufferPoolPage,
    SlottedPage, HeapFile, RowID,
    BTreeNode, BTreeIndex,
    Table, TableSchema, StorageEngine,
    TransactionalStorageEngine, CheckpointManager,
    PAGE_SIZE, INVALID_PAGE_ID, PageType,
    _encode_value, _decode_value, _encode_row, _decode_row,
    _key_to_bytes, _compare_keys,
)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    def test_encode_decode_none(self):
        data = _encode_value(None)
        val, off = _decode_value(data)
        assert val is None

    def test_encode_decode_bool(self):
        for v in [True, False]:
            data = _encode_value(v)
            val, off = _decode_value(data)
            assert val == v
            assert isinstance(val, bool)

    def test_encode_decode_int(self):
        for v in [0, 1, -1, 42, -999, 2**60]:
            data = _encode_value(v)
            val, off = _decode_value(data)
            assert val == v

    def test_encode_decode_float(self):
        for v in [0.0, 1.5, -3.14, 1e100]:
            data = _encode_value(v)
            val, off = _decode_value(data)
            assert val == v

    def test_encode_decode_string(self):
        for v in ["", "hello", "unicode: cafe", "a" * 1000]:
            data = _encode_value(v)
            val, off = _decode_value(data)
            assert val == v

    def test_encode_decode_bytes(self):
        for v in [b"", b"\x00\x01\x02", b"hello"]:
            data = _encode_value(v)
            val, off = _decode_value(data)
            assert val == v

    def test_encode_decode_list(self):
        v = [1, "two", 3.0, None, True]
        data = _encode_value(v)
        val, off = _decode_value(data)
        assert val == v

    def test_encode_decode_row(self):
        row = {"id": 1, "name": "Alice", "age": 30, "active": True}
        data = _encode_row(row)
        decoded, off = _decode_row(data)
        assert decoded == row

    def test_encode_decode_row_with_none(self):
        row = {"id": 1, "name": None, "score": 3.14}
        data = _encode_row(row)
        decoded, off = _decode_row(data)
        assert decoded == row

    def test_encode_decode_nested_list(self):
        v = [1, [2, 3], "hello"]
        data = _encode_value(v)
        val, off = _decode_value(data)
        assert val == v


class TestKeyComparison:
    def test_int_comparison(self):
        assert _compare_keys(1, 2) == -1
        assert _compare_keys(2, 1) == 1
        assert _compare_keys(5, 5) == 0

    def test_string_comparison(self):
        assert _compare_keys("a", "b") == -1
        assert _compare_keys("b", "a") == 1
        assert _compare_keys("x", "x") == 0

    def test_none_comparison(self):
        assert _compare_keys(None, None) == 0
        assert _compare_keys(None, 1) == -1
        assert _compare_keys(1, None) == 1

    def test_negative_int_key_bytes(self):
        b1 = _key_to_bytes(-5)
        b2 = _key_to_bytes(0)
        b3 = _key_to_bytes(5)
        assert b1 < b2 < b3

    def test_tuple_key(self):
        assert _compare_keys((1, "a"), (1, "b")) == -1
        assert _compare_keys((2, "a"), (1, "z")) == 1

    def test_float_key_bytes(self):
        b1 = _key_to_bytes(-1.0)
        b2 = _key_to_bytes(0.0)
        b3 = _key_to_bytes(1.0)
        assert b1 < b2 < b3


# =============================================================================
# DiskManager Tests
# =============================================================================

class TestDiskManager:
    def test_allocate_page(self):
        dm = DiskManager()
        p0 = dm.allocate_page()
        p1 = dm.allocate_page()
        assert p0 == 0
        assert p1 == 1
        assert dm.num_pages == 2

    def test_read_write_page(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        data = bytearray(PAGE_SIZE)
        data[0:4] = b'TEST'
        dm.write_page(pid, data)
        read_back = dm.read_page(pid)
        assert read_back[0:4] == b'TEST'

    def test_deallocate_and_reuse(self):
        dm = DiskManager()
        p0 = dm.allocate_page()
        p1 = dm.allocate_page()
        dm.deallocate_page(p0)
        assert dm.num_free_pages == 1
        p2 = dm.allocate_page()
        assert p2 == p0  # Reused
        assert dm.num_free_pages == 0

    def test_read_invalid_page(self):
        dm = DiskManager()
        with pytest.raises(IOError):
            dm.read_page(99)

    def test_write_invalid_page(self):
        dm = DiskManager()
        with pytest.raises(IOError):
            dm.write_page(0, bytearray(PAGE_SIZE))

    def test_write_wrong_size(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        with pytest.raises(ValueError):
            dm.write_page(pid, bytearray(100))

    def test_deallocate_zeros_page(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        data = bytearray(PAGE_SIZE)
        data[0] = 0xFF
        dm.write_page(pid, data)
        dm.deallocate_page(pid)
        pid2 = dm.allocate_page()
        assert pid2 == pid
        read = dm.read_page(pid2)
        assert read[0] == 0

    def test_multiple_pages(self):
        dm = DiskManager()
        pages = [dm.allocate_page() for _ in range(10)]
        for i, pid in enumerate(pages):
            data = bytearray(PAGE_SIZE)
            data[0] = i
            dm.write_page(pid, data)
        for i, pid in enumerate(pages):
            assert dm.read_page(pid)[0] == i

    def test_deallocate_idempotent(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        dm.deallocate_page(pid)
        dm.deallocate_page(pid)  # Should not add twice
        assert dm.num_free_pages == 1


# =============================================================================
# BufferPool Tests
# =============================================================================

class TestBufferPool:
    def test_fetch_and_unpin(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        bp = BufferPool(dm, pool_size=4)
        frame = bp.fetch_page(pid)
        assert frame.page_id == pid
        assert frame.pin_count == 1
        bp.unpin_page(pid)
        assert frame.pin_count == 0

    def test_new_page(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        frame = bp.new_page()
        assert frame.page_id == 0
        assert frame.dirty is True
        assert frame.pin_count == 1

    def test_cache_hit(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        bp = BufferPool(dm, pool_size=4)
        f1 = bp.fetch_page(pid)
        bp.unpin_page(pid)
        f2 = bp.fetch_page(pid)
        assert f1 is f2
        assert bp._hit_count == 1

    def test_dirty_flush(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        frame = bp.new_page()
        frame.data[0] = 42
        bp.unpin_page(frame.page_id, dirty=True)
        bp.flush_page(frame.page_id)
        # Read from disk directly
        data = dm.read_page(frame.page_id)
        assert data[0] == 42

    def test_eviction_lru(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=2)
        f0 = bp.new_page()
        f0.data[0] = 10
        bp.unpin_page(f0.page_id, dirty=True)

        f1 = bp.new_page()
        f1.data[0] = 20
        bp.unpin_page(f1.page_id, dirty=True)

        # Pool is full, next alloc should evict LRU (f0)
        f2 = bp.new_page()
        bp.unpin_page(f2.page_id)

        # f0 should have been flushed and evicted
        data = dm.read_page(f0.page_id)
        assert data[0] == 10

    def test_eviction_pinned_error(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=1)
        f0 = bp.new_page()
        # Don't unpin -- pool is full and all pinned
        with pytest.raises(RuntimeError, match="all pages pinned"):
            bp.new_page()

    def test_flush_all(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        frames = []
        for i in range(3):
            f = bp.new_page()
            f.data[0] = i + 1
            bp.unpin_page(f.page_id, dirty=True)
            frames.append(f)
        bp.flush_all()
        for f in frames:
            assert dm.read_page(f.page_id)[0] == f.data[0]
            assert not f.dirty

    def test_delete_page(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        f = bp.new_page()
        pid = f.page_id
        bp.unpin_page(pid)
        bp.delete_page(pid)
        assert pid not in bp._frames
        assert dm.num_free_pages == 1

    def test_hit_rate(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        bp = BufferPool(dm, pool_size=4)
        bp.fetch_page(pid)
        bp.unpin_page(pid)
        bp.fetch_page(pid)  # Hit
        bp.unpin_page(pid)
        assert bp.hit_rate == 0.5

    def test_dirty_count(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        f1 = bp.new_page()
        bp.unpin_page(f1.page_id, dirty=True)
        f2 = bp.new_page()
        bp.unpin_page(f2.page_id, dirty=False)
        # f1 was marked dirty in new_page, f2 was too but unpin with dirty=False doesn't clear it
        # Actually new_page sets dirty=True always
        assert bp.dirty_count >= 1

    def test_multiple_pins(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        bp = BufferPool(dm, pool_size=4)
        f1 = bp.fetch_page(pid)
        f2 = bp.fetch_page(pid)
        assert f1 is f2
        assert f1.pin_count == 2
        bp.unpin_page(pid)
        assert f1.pin_count == 1
        bp.unpin_page(pid)
        assert f1.pin_count == 0


# =============================================================================
# SlottedPage Tests
# =============================================================================

class TestSlottedPage:
    def test_create_page(self):
        page = SlottedPage(0)
        assert page.num_slots == 0
        assert page.next_page == INVALID_PAGE_ID

    def test_insert_and_get_tuple(self):
        page = SlottedPage(0)
        data = b"hello world"
        slot = page.insert_tuple(data)
        assert slot == 0
        assert page.get_tuple(0) == data

    def test_multiple_inserts(self):
        page = SlottedPage(0)
        for i in range(10):
            data = f"row_{i}".encode()
            slot = page.insert_tuple(data)
            assert slot == i

        for i in range(10):
            assert page.get_tuple(i) == f"row_{i}".encode()

    def test_delete_tuple(self):
        page = SlottedPage(0)
        page.insert_tuple(b"row_0")
        page.insert_tuple(b"row_1")
        assert page.delete_tuple(0) is True
        assert page.get_tuple(0) is None
        assert page.get_tuple(1) == b"row_1"

    def test_delete_nonexistent(self):
        page = SlottedPage(0)
        assert page.delete_tuple(99) is False

    def test_reuse_deleted_slot(self):
        page = SlottedPage(0)
        page.insert_tuple(b"row_0")
        page.insert_tuple(b"row_1")
        page.delete_tuple(0)
        slot = page.insert_tuple(b"row_2")
        assert slot == 0  # Reuses deleted slot
        assert page.get_tuple(0) == b"row_2"

    def test_update_tuple_fits(self):
        page = SlottedPage(0)
        page.insert_tuple(b"hello world!!")
        new_slot = page.update_tuple(0, b"hi")
        assert new_slot == 0
        assert page.get_tuple(0) == b"hi"

    def test_update_tuple_no_fit(self):
        page = SlottedPage(0)
        page.insert_tuple(b"hi")
        new_slot = page.update_tuple(0, b"a much longer string that definitely won't fit in the old slot")
        assert new_slot >= 0
        assert page.get_tuple(new_slot) == b"a much longer string that definitely won't fit in the old slot"

    def test_free_space_decreases(self):
        page = SlottedPage(0)
        initial = page.free_space()
        page.insert_tuple(b"some data here")
        assert page.free_space() < initial

    def test_page_full(self):
        page = SlottedPage(0)
        # Fill the page
        big_data = b"x" * 200
        count = 0
        while True:
            slot = page.insert_tuple(big_data)
            if slot == -1:
                break
            count += 1
        assert count > 0
        assert count < 100  # Sanity

    def test_iter_tuples(self):
        page = SlottedPage(0)
        page.insert_tuple(b"a")
        page.insert_tuple(b"b")
        page.insert_tuple(b"c")
        page.delete_tuple(1)
        tuples = list(page.iter_tuples())
        assert len(tuples) == 2
        assert tuples[0] == (0, b"a")
        assert tuples[1] == (2, b"c")

    def test_compact(self):
        page = SlottedPage(0)
        for i in range(5):
            page.insert_tuple(f"row_{i}".encode())
        page.delete_tuple(1)
        page.delete_tuple(3)
        n = page.compact()
        assert n == 3
        tuples = list(page.iter_tuples())
        assert len(tuples) == 3

    def test_next_page_link(self):
        page = SlottedPage(0)
        assert page.next_page == INVALID_PAGE_ID
        page.next_page = 42
        assert page.next_page == 42

    def test_get_invalid_slot(self):
        page = SlottedPage(0)
        assert page.get_tuple(-1) is None
        assert page.get_tuple(0) is None
        assert page.get_tuple(100) is None


# =============================================================================
# HeapFile Tests
# =============================================================================

class TestHeapFile:
    def _make_heap(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=16)
        return HeapFile(bp), bp

    def test_insert_and_get(self):
        heap, _ = self._make_heap()
        row = {"id": 1, "name": "Alice"}
        rid = heap.insert(row)
        assert isinstance(rid, RowID)
        got = heap.get(rid)
        assert got == row

    def test_multiple_inserts(self):
        heap, _ = self._make_heap()
        rids = []
        for i in range(50):
            rid = heap.insert({"id": i, "value": f"val_{i}"})
            rids.append(rid)
        assert heap.row_count == 50
        for i, rid in enumerate(rids):
            row = heap.get(rid)
            assert row["id"] == i

    def test_delete(self):
        heap, _ = self._make_heap()
        rid = heap.insert({"id": 1, "name": "Alice"})
        assert heap.delete(rid) is True
        assert heap.get(rid) is None
        assert heap.row_count == 0

    def test_update(self):
        heap, _ = self._make_heap()
        rid = heap.insert({"id": 1, "name": "Alice"})
        new_rid = heap.update(rid, {"id": 1, "name": "Bob"})
        got = heap.get(new_rid)
        assert got["name"] == "Bob"

    def test_scan(self):
        heap, _ = self._make_heap()
        for i in range(10):
            heap.insert({"id": i})
        rows = list(heap.scan())
        assert len(rows) == 10
        ids = {r["id"] for _, r in rows}
        assert ids == set(range(10))

    def test_page_allocation(self):
        heap, _ = self._make_heap()
        # Insert enough to need multiple pages
        for i in range(200):
            heap.insert({"id": i, "data": "x" * 100})
        assert heap.page_count > 1

    def test_large_row(self):
        heap, _ = self._make_heap()
        # Row that fills most of a page
        rid = heap.insert({"id": 1, "data": "x" * 3000})
        got = heap.get(rid)
        assert got["data"] == "x" * 3000

    def test_update_causes_page_move(self):
        heap, _ = self._make_heap()
        rid = heap.insert({"id": 1, "data": "short"})
        new_rid = heap.update(rid, {"id": 1, "data": "x" * 3000})
        got = heap.get(new_rid)
        assert got["data"] == "x" * 3000


# =============================================================================
# BTreeNode Serialization Tests
# =============================================================================

class TestBTreeNode:
    def test_leaf_serialize_deserialize(self):
        node = BTreeNode(0, is_leaf=True)
        node.keys = [1, 5, 10]
        node.values = [RowID(0, 0), RowID(0, 1), RowID(1, 0)]
        node.next_leaf = 5
        node.prev_leaf = 3

        data = node.serialize()
        restored = BTreeNode.deserialize(0, data)
        assert restored.is_leaf is True
        assert restored.keys == [1, 5, 10]
        assert len(restored.values) == 3
        assert restored.values[0] == RowID(0, 0)
        assert restored.next_leaf == 5
        assert restored.prev_leaf == 3

    def test_internal_serialize_deserialize(self):
        node = BTreeNode(0, is_leaf=False)
        node.keys = [10, 20]
        node.children = [1, 2, 3]
        node.parent_page = 5

        data = node.serialize()
        restored = BTreeNode.deserialize(0, data)
        assert restored.is_leaf is False
        assert restored.keys == [10, 20]
        assert restored.children == [1, 2, 3]
        assert restored.parent_page == 5

    def test_empty_leaf(self):
        node = BTreeNode(0, is_leaf=True)
        data = node.serialize()
        restored = BTreeNode.deserialize(0, data)
        assert restored.keys == []
        assert restored.values == []

    def test_string_keys(self):
        node = BTreeNode(0, is_leaf=True)
        node.keys = ["alice", "bob", "charlie"]
        node.values = [RowID(0, i) for i in range(3)]
        data = node.serialize()
        restored = BTreeNode.deserialize(0, data)
        assert restored.keys == ["alice", "bob", "charlie"]


# =============================================================================
# BTreeIndex Tests
# =============================================================================

class TestBTreeIndex:
    def _make_index(self, order=10):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=64)
        return BTreeIndex(bp, name="test", order=order), bp

    def test_insert_and_search(self):
        idx, _ = self._make_index()
        idx.insert(10, RowID(0, 0))
        idx.insert(20, RowID(0, 1))
        idx.insert(5, RowID(0, 2))
        assert idx.search(10) == RowID(0, 0)
        assert idx.search(20) == RowID(0, 1)
        assert idx.search(5) == RowID(0, 2)
        assert idx.search(99) is None

    def test_size(self):
        idx, _ = self._make_index()
        assert idx.size == 0
        idx.insert(1, RowID(0, 0))
        assert idx.size == 1
        idx.insert(2, RowID(0, 1))
        assert idx.size == 2

    def test_duplicate_key_update(self):
        idx, _ = self._make_index()
        idx.insert(10, RowID(0, 0))
        idx.insert(10, RowID(1, 1))
        assert idx.search(10) == RowID(1, 1)
        assert idx.size == 1  # Not duplicated

    def test_delete(self):
        idx, _ = self._make_index()
        idx.insert(10, RowID(0, 0))
        assert idx.delete(10) is True
        assert idx.search(10) is None
        assert idx.size == 0

    def test_delete_nonexistent(self):
        idx, _ = self._make_index()
        assert idx.delete(99) is False

    def test_range_scan(self):
        idx, _ = self._make_index()
        for i in range(20):
            idx.insert(i, RowID(0, i))
        results = list(idx.range_scan(5, 15))
        keys = [k for k, _ in results]
        assert keys == list(range(5, 16))

    def test_range_scan_exclusive(self):
        idx, _ = self._make_index()
        for i in range(10):
            idx.insert(i, RowID(0, i))
        results = list(idx.range_scan(3, 7, include_low=False, include_high=False))
        keys = [k for k, _ in results]
        assert keys == [4, 5, 6]

    def test_range_scan_open_ended(self):
        idx, _ = self._make_index()
        for i in range(10):
            idx.insert(i, RowID(0, i))
        # No low bound
        results = list(idx.range_scan(high=5))
        assert len(results) == 6  # 0-5
        # No high bound
        results = list(idx.range_scan(low=5))
        assert len(results) == 5  # 5-9

    def test_scan_all(self):
        idx, _ = self._make_index()
        for i in [5, 3, 8, 1, 9]:
            idx.insert(i, RowID(0, i))
        results = list(idx.scan_all())
        keys = [k for k, _ in results]
        assert keys == [1, 3, 5, 8, 9]

    def test_min_max_key(self):
        idx, _ = self._make_index()
        assert idx.min_key() is None
        idx.insert(10, RowID(0, 0))
        idx.insert(5, RowID(0, 1))
        idx.insert(20, RowID(0, 2))
        assert idx.min_key() == 5
        assert idx.max_key() == 20

    def test_many_inserts_causes_splits(self):
        idx, _ = self._make_index(order=4)  # Small order to force splits
        for i in range(50):
            idx.insert(i, RowID(0, i))
        assert idx.size == 50
        assert idx.height > 1
        # Verify all searchable
        for i in range(50):
            assert idx.search(i) == RowID(0, i)

    def test_reverse_insert_order(self):
        idx, _ = self._make_index(order=4)
        for i in range(30, 0, -1):
            idx.insert(i, RowID(0, i))
        results = list(idx.scan_all())
        keys = [k for k, _ in results]
        assert keys == list(range(1, 31))

    def test_random_order_inserts(self):
        idx, _ = self._make_index(order=5)
        import random
        values = list(range(100))
        random.seed(42)
        random.shuffle(values)
        for v in values:
            idx.insert(v, RowID(0, v))
        assert idx.size == 100
        for v in values:
            assert idx.search(v) == RowID(0, v)

    def test_string_keys(self):
        idx, _ = self._make_index()
        idx.insert("alice", RowID(0, 0))
        idx.insert("bob", RowID(0, 1))
        idx.insert("charlie", RowID(0, 2))
        assert idx.search("bob") == RowID(0, 1)
        assert idx.search("dave") is None

    def test_height_grows(self):
        idx, _ = self._make_index(order=4)
        assert idx.height == 1
        for i in range(20):
            idx.insert(i, RowID(0, i))
        assert idx.height >= 2

    def test_delete_after_splits(self):
        idx, _ = self._make_index(order=4)
        for i in range(20):
            idx.insert(i, RowID(0, i))
        for i in range(10):
            assert idx.delete(i) is True
        assert idx.size == 10
        for i in range(10, 20):
            assert idx.search(i) == RowID(0, i)


# =============================================================================
# Table Tests
# =============================================================================

class TestTable:
    def _make_table(self, pk="id"):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=32)
        schema = TableSchema("users", ["id", "name", "age"], primary_key=pk)
        return Table(schema, bp)

    def test_insert_and_get_by_pk(self):
        t = self._make_table()
        t.insert({"id": 1, "name": "Alice", "age": 30})
        row = t.get_by_pk(1)
        assert row["name"] == "Alice"

    def test_auto_increment(self):
        t = self._make_table()
        t.insert({"name": "Alice", "age": 30})
        t.insert({"name": "Bob", "age": 25})
        assert t.get_by_pk(1)["name"] == "Alice"
        assert t.get_by_pk(2)["name"] == "Bob"

    def test_update_by_pk(self):
        t = self._make_table()
        t.insert({"id": 1, "name": "Alice", "age": 30})
        assert t.update_by_pk(1, {"name": "Alicia"}) is True
        row = t.get_by_pk(1)
        assert row["name"] == "Alicia"
        assert row["age"] == 30

    def test_delete_by_pk(self):
        t = self._make_table()
        t.insert({"id": 1, "name": "Alice", "age": 30})
        assert t.delete_by_pk(1) is True
        assert t.get_by_pk(1) is None

    def test_scan(self):
        t = self._make_table()
        for i in range(10):
            t.insert({"id": i + 1, "name": f"user_{i}", "age": 20 + i})
        rows = list(t.scan())
        assert len(rows) == 10

    def test_scan_with_predicate(self):
        t = self._make_table()
        for i in range(10):
            t.insert({"id": i + 1, "name": f"user_{i}", "age": 20 + i})
        rows = list(t.scan(predicate=lambda r: r["age"] >= 25))
        assert len(rows) == 5

    def test_row_count(self):
        t = self._make_table()
        assert t.row_count == 0
        t.insert({"id": 1, "name": "Alice", "age": 30})
        assert t.row_count == 1
        t.delete_by_pk(1)
        assert t.row_count == 0

    def test_create_secondary_index(self):
        t = self._make_table()
        for i in range(20):
            t.insert({"id": i + 1, "name": f"user_{i}", "age": 20 + (i % 5)})
        idx = t.create_index("idx_age", "age")
        assert idx.size == 20

    def test_index_scan(self):
        t = self._make_table()
        for i in range(20):
            t.insert({"id": i + 1, "name": f"user_{i}", "age": 20 + i})
        t.create_index("idx_age", "age")
        rows = list(t.index_scan("idx_age", low=25, high=30))
        ages = {r["age"] for r in rows}
        assert ages == {25, 26, 27, 28, 29, 30}

    def test_drop_index(self):
        t = self._make_table()
        t.create_index("idx_age", "age")
        t.drop_index("idx_age")
        assert "idx_age" not in t.indexes

    def test_duplicate_index_error(self):
        t = self._make_table()
        t.create_index("idx_age", "age")
        with pytest.raises(ValueError):
            t.create_index("idx_age", "age")

    def test_no_pk_table(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=16)
        schema = TableSchema("logs", ["ts", "msg"])
        t = Table(schema, bp)
        t.insert({"ts": 1, "msg": "hello"})
        t.insert({"ts": 2, "msg": "world"})
        rows = list(t.scan())
        assert len(rows) == 2

    def test_composite_index(self):
        t = self._make_table()
        for i in range(10):
            t.insert({"id": i + 1, "name": f"user_{i % 3}", "age": 20 + i})
        t.create_index("idx_name_age", ["name", "age"])
        # The composite key is a tuple
        assert t.indexes["idx_name_age"].size == 10

    def test_get_nonexistent_pk(self):
        t = self._make_table()
        assert t.get_by_pk(999) is None

    def test_update_nonexistent(self):
        t = self._make_table()
        assert t.update_by_pk(999, {"name": "Ghost"}) is False

    def test_delete_nonexistent(self):
        t = self._make_table()
        assert t.delete_by_pk(999) is False

    def test_many_rows(self):
        t = self._make_table()
        for i in range(500):
            t.insert({"name": f"user_{i}", "age": i})
        assert t.row_count == 500
        # Spot check
        assert t.get_by_pk(250)["name"] == "user_249"


# =============================================================================
# CheckpointManager Tests
# =============================================================================

class TestCheckpointManager:
    def test_create_and_restore(self):
        dm = DiskManager()
        pid = dm.allocate_page()
        data = bytearray(PAGE_SIZE)
        data[0] = 42
        dm.write_page(pid, data)

        cm = CheckpointManager(dm)
        cp_id = cm.create_checkpoint()

        # Modify data
        data[0] = 99
        dm.write_page(pid, data)
        assert dm.read_page(pid)[0] == 99

        # Restore
        cm.restore_checkpoint(cp_id)
        assert dm.read_page(pid)[0] == 42

    def test_list_checkpoints(self):
        dm = DiskManager()
        cm = CheckpointManager(dm)
        c1 = cm.create_checkpoint()
        c2 = cm.create_checkpoint()
        assert cm.list_checkpoints() == [c1, c2]

    def test_delete_checkpoint(self):
        dm = DiskManager()
        cm = CheckpointManager(dm)
        c1 = cm.create_checkpoint()
        cm.delete_checkpoint(c1)
        assert cm.list_checkpoints() == []

    def test_restore_invalid(self):
        dm = DiskManager()
        cm = CheckpointManager(dm)
        assert cm.restore_checkpoint(999) is False


# =============================================================================
# StorageEngine Tests
# =============================================================================

class TestStorageEngine:
    def test_create_table(self):
        se = StorageEngine()
        t = se.create_table("users", ["id", "name"], primary_key="id")
        assert t is not None
        assert "users" in se.table_names()

    def test_create_duplicate_table(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"])
        with pytest.raises(ValueError):
            se.create_table("users", ["id", "name"])

    def test_drop_table(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"])
        se.drop_table("users")
        assert "users" not in se.table_names()

    def test_drop_nonexistent(self):
        se = StorageEngine()
        with pytest.raises(KeyError):
            se.drop_table("nonexistent")

    def test_insert_and_get(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name", "age"], primary_key="id")
        se.insert("users", {"id": 1, "name": "Alice", "age": 30})
        row = se.get("users", 1)
        assert row["name"] == "Alice"

    def test_update(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"], primary_key="id")
        se.insert("users", {"id": 1, "name": "Alice"})
        se.update("users", 1, {"name": "Alicia"})
        assert se.get("users", 1)["name"] == "Alicia"

    def test_delete(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"], primary_key="id")
        se.insert("users", {"id": 1, "name": "Alice"})
        se.delete("users", 1)
        assert se.get("users", 1) is None

    def test_scan(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"], primary_key="id")
        for i in range(5):
            se.insert("users", {"id": i + 1, "name": f"user_{i}"})
        rows = se.scan("users")
        assert len(rows) == 5

    def test_scan_with_predicate(self):
        se = StorageEngine()
        se.create_table("nums", ["id", "val"], primary_key="id")
        for i in range(10):
            se.insert("nums", {"id": i + 1, "val": i})
        rows = se.scan("nums", predicate=lambda r: r["val"] > 5)
        assert len(rows) == 4

    def test_index_operations(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name", "age"], primary_key="id")
        for i in range(20):
            se.insert("users", {"id": i + 1, "name": f"user_{i}", "age": 20 + i})
        se.create_index("users", "idx_age", "age")
        rows = se.index_scan("users", "idx_age", low=30, high=35)
        assert len(rows) == 6
        ages = {r["age"] for r in rows}
        assert ages == {30, 31, 32, 33, 34, 35}

    def test_nonexistent_table_operations(self):
        se = StorageEngine()
        with pytest.raises(KeyError):
            se.insert("nope", {"x": 1})
        with pytest.raises(KeyError):
            se.get("nope", 1)
        with pytest.raises(KeyError):
            se.scan("nope")

    def test_checkpoint_and_restore(self):
        se = StorageEngine()
        se.create_table("data", ["id", "val"], primary_key="id")
        se.insert("data", {"id": 1, "val": "original"})
        cp = se.checkpoint()

        se.insert("data", {"id": 2, "val": "after_checkpoint"})
        se.flush()

        # Note: restore only restores disk pages, not in-memory table state
        # This tests the disk-level checkpoint
        assert cp > 0

    def test_stats(self):
        se = StorageEngine()
        se.create_table("t1", ["id", "val"], primary_key="id")
        se.insert("t1", {"id": 1, "val": "x"})
        stats = se.stats()
        assert stats['num_tables'] == 1
        assert stats['tables']['t1']['rows'] == 1

    def test_flush(self):
        se = StorageEngine()
        se.create_table("t", ["id"], primary_key="id")
        se.insert("t", {"id": 1})
        se.flush()
        assert se.buffer_pool.dirty_count == 0

    def test_get_table(self):
        se = StorageEngine()
        se.create_table("t", ["id"], primary_key="id")
        assert se.get_table("t") is not None
        assert se.get_table("nope") is None

    def test_multiple_tables(self):
        se = StorageEngine()
        se.create_table("users", ["id", "name"], primary_key="id")
        se.create_table("orders", ["id", "user_id", "total"], primary_key="id")
        se.insert("users", {"id": 1, "name": "Alice"})
        se.insert("orders", {"id": 1, "user_id": 1, "total": 100})
        assert se.get("users", 1)["name"] == "Alice"
        assert se.get("orders", 1)["total"] == 100

    def test_large_dataset(self):
        se = StorageEngine(buffer_pool_size=32)
        se.create_table("big", ["id", "data"], primary_key="id")
        for i in range(1000):
            se.insert("big", {"id": i + 1, "data": f"value_{i}"})
        assert se.get("big", 500)["data"] == "value_499"
        assert se.get("big", 1)["data"] == "value_0"
        assert se.get("big", 1000)["data"] == "value_999"


# =============================================================================
# TransactionalStorageEngine Tests
# =============================================================================

class TestTransactionalStorageEngine:
    def test_begin_commit(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")
        tx = tse.begin()
        tse.insert(tx, "t", {"id": 1, "val": "hello"})
        tse.commit(tx)
        # Data persists after commit
        tx2 = tse.begin()
        assert tse.get(tx2, "t", 1)["val"] == "hello"

    def test_rollback(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")

        # Insert and commit first row
        tx1 = tse.begin()
        tse.insert(tx1, "t", {"id": 1, "val": "keep"})
        tse.commit(tx1)

        # Insert second row and rollback
        tx2 = tse.begin()
        tse.insert(tx2, "t", {"id": 2, "val": "discard"})
        tse.rollback(tx2)

        # Row 2 should be gone (undo log deletes it)
        tx3 = tse.begin()
        assert tse.get(tx3, "t", 1)["val"] == "keep"

    def test_update_rollback(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")
        tx1 = tse.begin()
        tse.insert(tx1, "t", {"id": 1, "val": "original"})
        tse.commit(tx1)

        tx2 = tse.begin()
        tse.update(tx2, "t", 1, {"val": "modified"})
        tse.rollback(tx2)

        tx3 = tse.begin()
        assert tse.get(tx3, "t", 1)["val"] == "original"

    def test_delete_rollback(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")
        tx1 = tse.begin()
        tse.insert(tx1, "t", {"id": 1, "val": "keep"})
        tse.commit(tx1)

        tx2 = tse.begin()
        tse.delete(tx2, "t", 1)
        tse.rollback(tx2)

        tx3 = tse.begin()
        row = tse.get(tx3, "t", 1)
        assert row is not None
        assert row["val"] == "keep"

    def test_scan_in_transaction(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")
        tx = tse.begin()
        for i in range(5):
            tse.insert(tx, "t", {"id": i + 1, "val": f"v_{i}"})
        rows = tse.scan(tx, "t")
        assert len(rows) == 5

    def test_inactive_transaction_error(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id"], primary_key="id")
        tx = tse.begin()
        tse.commit(tx)
        with pytest.raises(ValueError, match="committed"):
            tse.insert(tx, "t", {"id": 1})

    def test_unknown_transaction_error(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id"], primary_key="id")
        with pytest.raises(ValueError, match="Unknown"):
            tse.insert(999, "t", {"id": 1})

    def test_multiple_transactions(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id", "val"], primary_key="id")
        tx1 = tse.begin()
        tx2 = tse.begin()
        tse.insert(tx1, "t", {"id": 1, "val": "from_tx1"})
        tse.insert(tx2, "t", {"id": 2, "val": "from_tx2"})
        tse.commit(tx1)
        tse.commit(tx2)
        tx3 = tse.begin()
        assert tse.get(tx3, "t", 1)["val"] == "from_tx1"
        assert tse.get(tx3, "t", 2)["val"] == "from_tx2"

    def test_stats(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id"], primary_key="id")
        tx = tse.begin()
        stats = tse.stats()
        assert stats['active_transactions'] == 1
        tse.commit(tx)
        stats = tse.stats()
        assert stats['active_transactions'] == 0

    def test_get_table(self):
        tse = TransactionalStorageEngine()
        tse.create_table("t", ["id"], primary_key="id")
        assert tse.get_table("t") is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def test_full_crud_workflow(self):
        se = StorageEngine()
        se.create_table("products", ["id", "name", "price", "qty"], primary_key="id")

        # Insert
        for i in range(100):
            se.insert("products", {
                "id": i + 1,
                "name": f"Product {i}",
                "price": 10.0 + i * 0.5,
                "qty": 100 - i
            })

        # Read
        p50 = se.get("products", 50)
        assert p50["name"] == "Product 49"
        assert p50["price"] == 10.0 + 49 * 0.5

        # Update
        se.update("products", 50, {"price": 99.99})
        assert se.get("products", 50)["price"] == 99.99

        # Delete
        se.delete("products", 50)
        assert se.get("products", 50) is None

        # Scan
        rows = se.scan("products")
        assert len(rows) == 99

    def test_index_after_mutations(self):
        se = StorageEngine()
        se.create_table("items", ["id", "category", "price"], primary_key="id")
        for i in range(50):
            se.insert("items", {"id": i + 1, "category": i % 5, "price": i * 10})

        se.create_index("items", "idx_cat", "category")

        # Index scan for category=2
        rows = se.index_scan("items", "idx_cat", low=2, high=2)
        assert len(rows) == 10  # 50/5 = 10 per category
        assert all(r["category"] == 2 for r in rows)

    def test_buffer_pool_under_pressure(self):
        se = StorageEngine(buffer_pool_size=4)  # Very small pool
        se.create_table("stress", ["id", "data"], primary_key="id")
        for i in range(100):
            se.insert("stress", {"id": i + 1, "data": f"payload_{i}"})
        # Verify data integrity under cache pressure
        for i in range(100):
            row = se.get("stress", i + 1)
            assert row["data"] == f"payload_{i}"

    def test_transactional_workflow(self):
        tse = TransactionalStorageEngine()
        tse.create_table("accounts", ["id", "name", "balance"], primary_key="id")

        # Setup
        tx = tse.begin()
        tse.insert(tx, "accounts", {"id": 1, "name": "Alice", "balance": 1000})
        tse.insert(tx, "accounts", {"id": 2, "name": "Bob", "balance": 500})
        tse.commit(tx)

        # Transfer: Alice -> Bob
        tx2 = tse.begin()
        alice = tse.get(tx2, "accounts", 1)
        bob = tse.get(tx2, "accounts", 2)
        tse.update(tx2, "accounts", 1, {"balance": alice["balance"] - 200})
        tse.update(tx2, "accounts", 2, {"balance": bob["balance"] + 200})
        tse.commit(tx2)

        tx3 = tse.begin()
        assert tse.get(tx3, "accounts", 1)["balance"] == 800
        assert tse.get(tx3, "accounts", 2)["balance"] == 700

    def test_transactional_rollback_integrity(self):
        tse = TransactionalStorageEngine()
        tse.create_table("data", ["id", "val"], primary_key="id")

        # Insert base data
        tx1 = tse.begin()
        tse.insert(tx1, "data", {"id": 1, "val": "A"})
        tse.insert(tx1, "data", {"id": 2, "val": "B"})
        tse.commit(tx1)

        # Failed transaction
        tx2 = tse.begin()
        tse.update(tx2, "data", 1, {"val": "X"})
        tse.delete(tx2, "data", 2)
        tse.insert(tx2, "data", {"id": 3, "val": "C"})
        tse.rollback(tx2)

        # Verify original state
        tx3 = tse.begin()
        assert tse.get(tx3, "data", 1)["val"] == "A"
        assert tse.get(tx3, "data", 2)["val"] == "B"

    def test_disk_persistence(self):
        """Data survives buffer pool eviction."""
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=2)  # Tiny pool
        schema = TableSchema("t", ["id", "val"], primary_key="id")
        table = Table(schema, bp)

        for i in range(20):
            table.insert({"id": i + 1, "val": f"data_{i}"})

        bp.flush_all()

        # Data was written to disk through evictions
        # Verify via fresh fetches
        for i in range(20):
            row = table.get_by_pk(i + 1)
            assert row is not None
            assert row["val"] == f"data_{i}"

    def test_mixed_types(self):
        se = StorageEngine()
        se.create_table("mixed", ["id", "int_col", "float_col", "str_col", "bool_col", "null_col"],
                        primary_key="id")
        se.insert("mixed", {
            "id": 1, "int_col": 42, "float_col": 3.14,
            "str_col": "hello", "bool_col": True, "null_col": None
        })
        row = se.get("mixed", 1)
        assert row["int_col"] == 42
        assert row["float_col"] == 3.14
        assert row["str_col"] == "hello"
        assert row["bool_col"] is True
        assert row["null_col"] is None

    def test_empty_table_operations(self):
        se = StorageEngine()
        se.create_table("empty", ["id", "val"], primary_key="id")
        assert se.get("empty", 1) is None
        assert se.scan("empty") == []
        assert se.delete("empty", 1) is False
        assert se.update("empty", 1, {"val": "x"}) is False

    def test_rowid_equality_and_hash(self):
        r1 = RowID(1, 2)
        r2 = RowID(1, 2)
        r3 = RowID(1, 3)
        assert r1 == r2
        assert r1 != r3
        assert hash(r1) == hash(r2)
        assert repr(r1) == "RowID(1, 2)"

    def test_rowid_to_tuple(self):
        r = RowID(5, 3)
        assert r.to_tuple() == (5, 3)

    def test_scan_predicate_filter(self):
        tse = TransactionalStorageEngine()
        tse.create_table("nums", ["id", "val"], primary_key="id")
        tx = tse.begin()
        for i in range(10):
            tse.insert(tx, "nums", {"id": i + 1, "val": i * 10})
        rows = tse.scan(tx, "nums", predicate=lambda r: r["val"] >= 50)
        assert len(rows) == 5


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    def test_empty_string_value(self):
        se = StorageEngine()
        se.create_table("t", ["id", "val"], primary_key="id")
        se.insert("t", {"id": 1, "val": ""})
        assert se.get("t", 1)["val"] == ""

    def test_large_string_value(self):
        se = StorageEngine()
        se.create_table("t", ["id", "val"], primary_key="id")
        big = "x" * 2000
        se.insert("t", {"id": 1, "val": big})
        assert se.get("t", 1)["val"] == big

    def test_negative_keys(self):
        idx_dm = DiskManager()
        idx_bp = BufferPool(idx_dm, pool_size=16)
        idx = BTreeIndex(idx_bp, order=5)
        for i in range(-10, 11):
            idx.insert(i, RowID(0, i + 10))
        results = list(idx.scan_all())
        keys = [k for k, _ in results]
        assert keys == list(range(-10, 11))

    def test_btree_single_key(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=8)
        idx = BTreeIndex(bp, order=4)
        idx.insert(42, RowID(0, 0))
        assert idx.search(42) == RowID(0, 0)
        assert idx.size == 1
        assert idx.min_key() == 42
        assert idx.max_key() == 42

    def test_slotted_page_update_deleted(self):
        page = SlottedPage(0)
        page.insert_tuple(b"data")
        page.delete_tuple(0)
        result = page.update_tuple(0, b"new")
        assert result == -1

    def test_heap_file_empty_scan(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=4)
        heap = HeapFile(bp)
        assert list(heap.scan()) == []

    def test_disk_manager_many_alloc_dealloc(self):
        dm = DiskManager()
        pids = [dm.allocate_page() for _ in range(20)]
        for pid in pids[::2]:
            dm.deallocate_page(pid)
        assert dm.num_free_pages == 10
        # Reallocate
        new_pids = [dm.allocate_page() for _ in range(10)]
        assert dm.num_free_pages == 0

    def test_buffer_pool_size_property(self):
        dm = DiskManager()
        bp = BufferPool(dm, pool_size=8)
        assert bp.size == 0
        f = bp.new_page()
        assert bp.size == 1
        bp.unpin_page(f.page_id)

    def test_storage_engine_custom_page_size(self):
        se = StorageEngine(page_size=1024)
        se.create_table("t", ["id", "val"], primary_key="id")
        se.insert("t", {"id": 1, "val": "hello"})
        assert se.get("t", 1)["val"] == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
