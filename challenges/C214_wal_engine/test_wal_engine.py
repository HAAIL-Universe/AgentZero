"""
Tests for C214: Write-Ahead Log Engine
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from wal_engine import (
    LogType, LSN, LogRecord, WALBuffer, WALWriter, WALReader,
    WALManager, RecoveryManager, WALStorageEngine, WALAnalyzer,
    _serialize_record, _deserialize_record, TransactionEntry
)


# =============================================================================
# LSN Tests
# =============================================================================

class TestLSN:
    def test_create(self):
        lsn = LSN(42)
        assert lsn.value == 42

    def test_comparison(self):
        a, b = LSN(1), LSN(2)
        assert a < b
        assert b > a
        assert a <= b
        assert a <= LSN(1)
        assert b >= a
        assert a == LSN(1)
        assert a != b

    def test_comparison_with_int(self):
        lsn = LSN(5)
        assert lsn == 5
        assert lsn < 6
        assert lsn > 4
        assert lsn <= 5
        assert lsn >= 5

    def test_hash(self):
        a, b = LSN(1), LSN(1)
        assert hash(a) == hash(b)
        s = {a}
        assert b in s

    def test_repr(self):
        assert "10" in repr(LSN(10))

    def test_int_conversion(self):
        assert int(LSN(7)) == 7

    def test_default_zero(self):
        assert LSN().value == 0


# =============================================================================
# LogRecord Tests
# =============================================================================

class TestLogRecord:
    def test_create_begin(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        assert r.lsn == LSN(1)
        assert r.txn_id == 1
        assert r.log_type == LogType.BEGIN

    def test_create_insert(self):
        r = LogRecord(lsn=2, txn_id=1, log_type=LogType.INSERT,
                      table_name="users", page_id=0, slot_idx=0,
                      after_image=["alice", 30])
        assert r.table_name == "users"
        assert r.after_image == ["alice", 30]

    def test_create_update(self):
        r = LogRecord(lsn=3, txn_id=1, log_type=LogType.UPDATE,
                      table_name="users", page_id=0, slot_idx=0,
                      before_image=["alice", 30], after_image=["alice", 31])
        assert r.before_image == ["alice", 30]
        assert r.after_image == ["alice", 31]

    def test_create_delete(self):
        r = LogRecord(lsn=4, txn_id=1, log_type=LogType.DELETE,
                      table_name="users", page_id=0, slot_idx=0,
                      before_image=["alice", 30])
        assert r.before_image == ["alice", 30]

    def test_is_undoable(self):
        assert LogRecord(1, 1, LogType.INSERT).is_undoable()
        assert LogRecord(1, 1, LogType.UPDATE).is_undoable()
        assert LogRecord(1, 1, LogType.DELETE).is_undoable()
        assert not LogRecord(1, 1, LogType.BEGIN).is_undoable()
        assert not LogRecord(1, 1, LogType.COMMIT).is_undoable()
        assert not LogRecord(1, 1, LogType.CLR).is_undoable()

    def test_is_redoable(self):
        assert LogRecord(1, 1, LogType.INSERT).is_redoable()
        assert LogRecord(1, 1, LogType.UPDATE).is_redoable()
        assert LogRecord(1, 1, LogType.DELETE).is_redoable()
        assert LogRecord(1, 1, LogType.CLR).is_redoable()
        assert not LogRecord(1, 1, LogType.BEGIN).is_redoable()
        assert not LogRecord(1, 1, LogType.COMMIT).is_redoable()

    def test_prev_lsn_chain(self):
        r = LogRecord(lsn=5, txn_id=1, log_type=LogType.INSERT, prev_lsn=LSN(2))
        assert r.prev_lsn == LSN(2)

    def test_clr_undo_next(self):
        r = LogRecord(lsn=6, txn_id=1, log_type=LogType.CLR, undo_next_lsn=LSN(3))
        assert r.undo_next_lsn == LSN(3)

    def test_checkpoint_record(self):
        r = LogRecord(lsn=7, txn_id=0, log_type=LogType.CHECKPOINT,
                      active_txns={1: LSN(5), 2: LSN(6)},
                      dirty_pages={("users", 0): LSN(3)})
        assert len(r.active_txns) == 2
        assert len(r.dirty_pages) == 1

    def test_repr(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.INSERT, table_name="t")
        s = repr(r)
        assert "INSERT" in s
        assert "t" in s


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    def test_roundtrip_begin(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.lsn == r.lsn
        assert r2.txn_id == r.txn_id
        assert r2.log_type == r.log_type

    def test_roundtrip_insert(self):
        r = LogRecord(lsn=2, txn_id=1, log_type=LogType.INSERT,
                      table_name="users", page_id=5, slot_idx=3,
                      after_image=["alice", 30], prev_lsn=LSN(1))
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.table_name == "users"
        assert r2.page_id == 5
        assert r2.slot_idx == 3
        assert r2.after_image == ["alice", 30]
        assert r2.prev_lsn == LSN(1)

    def test_roundtrip_update(self):
        r = LogRecord(lsn=3, txn_id=1, log_type=LogType.UPDATE,
                      table_name="t", page_id=0, slot_idx=0,
                      before_image=[1, "old"], after_image=[1, "new"])
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.before_image == [1, "old"]
        assert r2.after_image == [1, "new"]

    def test_roundtrip_checkpoint(self):
        r = LogRecord(lsn=10, txn_id=0, log_type=LogType.CHECKPOINT,
                      active_txns={1: LSN(5), 2: LSN(8)},
                      dirty_pages={("users", 0): LSN(3), ("orders", 1): LSN(7)})
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.active_txns[1] == LSN(5)
        assert r2.active_txns[2] == LSN(8)
        assert r2.dirty_pages[("users", 0)] == LSN(3)
        assert r2.dirty_pages[("orders", 1)] == LSN(7)

    def test_roundtrip_clr(self):
        r = LogRecord(lsn=20, txn_id=1, log_type=LogType.CLR,
                      table_name="t", page_id=0, slot_idx=0,
                      undo_next_lsn=LSN(5), prev_lsn=LSN(15))
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.undo_next_lsn == LSN(5)
        assert r2.prev_lsn == LSN(15)

    def test_roundtrip_none_values(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.table_name is None
        assert r2.page_id is None
        assert r2.slot_idx is None
        assert r2.before_image is None
        assert r2.after_image is None

    def test_roundtrip_string_data(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.INSERT,
                      table_name="t", page_id=0, slot_idx=0,
                      after_image="hello world")
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.after_image == "hello world"

    def test_roundtrip_int_data(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.INSERT,
                      table_name="t", page_id=0, slot_idx=0,
                      after_image=42)
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.after_image == 42

    def test_roundtrip_nested_list(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.INSERT,
                      table_name="t", page_id=0, slot_idx=0,
                      after_image=[1, "two", 3.0, None, True])
        data = _serialize_record(r)
        r2, _ = _deserialize_record(data)
        assert r2.after_image == [1, "two", 3.0, None, True]

    def test_multiple_records_sequential(self):
        records = [
            LogRecord(lsn=i, txn_id=1, log_type=LogType.INSERT,
                      table_name="t", page_id=0, slot_idx=i,
                      after_image=[i, f"val{i}"])
            for i in range(1, 6)
        ]
        data = bytearray()
        for r in records:
            data.extend(_serialize_record(r))
        offset = 0
        for i in range(5):
            r, offset = _deserialize_record(bytes(data), offset)
            assert r.lsn == LSN(i + 1)
            assert r.slot_idx == i + 1

    def test_crc_corruption_detected(self):
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        data = bytearray(_serialize_record(r))
        # Corrupt a byte in the body
        data[10] ^= 0xFF
        with pytest.raises(ValueError, match="CRC"):
            _deserialize_record(bytes(data))


# =============================================================================
# WAL Buffer Tests
# =============================================================================

class TestWALBuffer:
    def test_append(self):
        buf = WALBuffer(capacity=10)
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        full = buf.append(r)
        assert not full
        assert buf.pending_count == 1

    def test_buffer_full_signal(self):
        buf = WALBuffer(capacity=3)
        for i in range(2):
            assert not buf.append(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.BEGIN))
        assert buf.append(LogRecord(lsn=3, txn_id=1, log_type=LogType.BEGIN))

    def test_flush(self):
        buf = WALBuffer()
        writer = WALWriter()
        for i in range(5):
            buf.append(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.BEGIN))
        buf.flush(writer)
        assert buf.pending_count == 0
        assert writer.record_count == 5
        assert buf.flushed_lsn == LSN(5)

    def test_flush_up_to(self):
        buf = WALBuffer()
        writer = WALWriter()
        for i in range(5):
            buf.append(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.BEGIN))
        buf.flush_up_to(LSN(3), writer)
        assert buf.pending_count == 2
        assert writer.record_count == 3
        assert buf.flushed_lsn == LSN(3)

    def test_empty_flush(self):
        buf = WALBuffer()
        writer = WALWriter()
        buf.flush(writer)
        assert buf.flushed_lsn == LSN(0)


# =============================================================================
# WAL Writer Tests
# =============================================================================

class TestWALWriter:
    def test_write(self):
        writer = WALWriter()
        r = LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN)
        writer.write(r)
        assert writer.record_count == 1
        assert writer.size > 0

    def test_multiple_writes(self):
        writer = WALWriter()
        for i in range(10):
            writer.write(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.INSERT,
                                   table_name="t", page_id=0, slot_idx=i))
        assert writer.record_count == 10

    def test_truncate_after(self):
        writer = WALWriter()
        for i in range(5):
            writer.write(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.BEGIN))
        writer.truncate_after(LSN(3))
        assert writer.record_count == 3

    def test_truncate_beyond_end(self):
        writer = WALWriter()
        writer.write(LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN))
        writer.truncate_after(LSN(100))  # No effect
        assert writer.record_count == 1

    def test_get_data(self):
        writer = WALWriter()
        writer.write(LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN))
        data = writer.get_data()
        assert isinstance(data, bytes)
        assert len(data) > 0


# =============================================================================
# WAL Reader Tests
# =============================================================================

class TestWALReader:
    def _make_reader_with_records(self, n=5):
        writer = WALWriter()
        for i in range(n):
            writer.write(LogRecord(lsn=i+1, txn_id=1, log_type=LogType.INSERT,
                                   table_name="t", page_id=0, slot_idx=i))
        return WALReader(writer)

    def test_scan_forward(self):
        reader = self._make_reader_with_records(5)
        records = list(reader.scan_forward())
        assert len(records) == 5
        assert records[0].lsn == LSN(1)
        assert records[-1].lsn == LSN(5)

    def test_scan_forward_from_lsn(self):
        reader = self._make_reader_with_records(5)
        records = list(reader.scan_forward(from_lsn=LSN(3)))
        assert len(records) == 3
        assert records[0].lsn == LSN(3)

    def test_scan_backward(self):
        reader = self._make_reader_with_records(5)
        records = list(reader.scan_backward())
        assert len(records) == 5
        assert records[0].lsn == LSN(5)
        assert records[-1].lsn == LSN(1)

    def test_find_by_lsn(self):
        reader = self._make_reader_with_records(5)
        r = reader.find_by_lsn(LSN(3))
        assert r is not None
        assert r.lsn == LSN(3)

    def test_find_by_lsn_int(self):
        reader = self._make_reader_with_records(5)
        r = reader.find_by_lsn(3)
        assert r is not None

    def test_find_by_lsn_not_found(self):
        reader = self._make_reader_with_records(5)
        assert reader.find_by_lsn(99) is None

    def test_find_by_txn(self):
        writer = WALWriter()
        writer.write(LogRecord(lsn=1, txn_id=1, log_type=LogType.INSERT, table_name="t", page_id=0, slot_idx=0))
        writer.write(LogRecord(lsn=2, txn_id=2, log_type=LogType.INSERT, table_name="t", page_id=0, slot_idx=1))
        writer.write(LogRecord(lsn=3, txn_id=1, log_type=LogType.INSERT, table_name="t", page_id=0, slot_idx=2))
        reader = WALReader(writer)
        txn1 = reader.find_by_txn(1)
        assert len(txn1) == 2
        txn2 = reader.find_by_txn(2)
        assert len(txn2) == 1

    def test_find_last_checkpoint(self):
        writer = WALWriter()
        writer.write(LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN))
        writer.write(LogRecord(lsn=2, txn_id=0, log_type=LogType.CHECKPOINT))
        writer.write(LogRecord(lsn=3, txn_id=1, log_type=LogType.INSERT, table_name="t", page_id=0, slot_idx=0))
        writer.write(LogRecord(lsn=4, txn_id=0, log_type=LogType.CHECKPOINT))
        reader = WALReader(writer)
        cp = reader.find_last_checkpoint()
        assert cp.lsn == LSN(4)

    def test_find_last_checkpoint_none(self):
        writer = WALWriter()
        writer.write(LogRecord(lsn=1, txn_id=1, log_type=LogType.BEGIN))
        reader = WALReader(writer)
        assert reader.find_last_checkpoint() is None

    def test_empty_log(self):
        writer = WALWriter()
        reader = WALReader(writer)
        assert list(reader.scan_forward()) == []
        assert list(reader.scan_backward()) == []


# =============================================================================
# WAL Manager Tests
# =============================================================================

class TestWALManager:
    def test_begin(self):
        wal = WALManager()
        lsn = wal.begin(1)
        assert lsn == LSN(1)
        assert 1 in wal.active_transactions

    def test_commit(self):
        wal = WALManager()
        wal.begin(1)
        wal.commit(1)
        assert 1 not in wal.active_transactions
        assert 1 in wal.committed_transactions

    def test_abort(self):
        wal = WALManager()
        wal.begin(1)
        wal.abort(1)
        assert 1 not in wal.active_transactions

    def test_insert_log(self):
        wal = WALManager()
        wal.begin(1)
        lsn = wal.log_insert(1, "users", 0, 0, ["alice", 30])
        assert lsn.value > 0

    def test_update_log(self):
        wal = WALManager()
        wal.begin(1)
        lsn = wal.log_update(1, "users", 0, 0, ["alice", 30], ["alice", 31])
        assert lsn.value > 0

    def test_delete_log(self):
        wal = WALManager()
        wal.begin(1)
        lsn = wal.log_delete(1, "users", 0, 0, ["alice", 30])
        assert lsn.value > 0

    def test_checkpoint(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        lsn = wal.checkpoint()
        assert lsn.value > 0

    def test_lsn_monotonic(self):
        wal = WALManager()
        lsns = []
        wal.begin(1)
        lsns.append(wal.log_insert(1, "t", 0, 0, [1]))
        lsns.append(wal.log_insert(1, "t", 0, 1, [2]))
        lsns.append(wal.log_insert(1, "t", 0, 2, [3]))
        for i in range(len(lsns) - 1):
            assert lsns[i] < lsns[i+1]

    def test_prev_lsn_chain(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(1, "t", 0, 1, [2])
        wal.flush()
        records = list(wal.reader.scan_forward())
        # Third record (second insert) should have prev_lsn pointing to second record (first insert)
        assert records[2].prev_lsn == records[1].lsn

    def test_multiple_txns(self):
        wal = WALManager()
        wal.begin(1)
        wal.begin(2)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(2, "t", 0, 1, [2])
        wal.commit(1)
        wal.commit(2)
        assert 1 in wal.committed_transactions
        assert 2 in wal.committed_transactions

    def test_log_count(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        # begin + insert + commit = 3
        assert wal.log_count() == 3

    def test_get_all_records(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, ["a"])
        wal.commit(1)
        records = wal.get_all_records()
        assert len(records) == 3
        assert records[0].log_type == LogType.BEGIN
        assert records[1].log_type == LogType.INSERT
        assert records[2].log_type == LogType.COMMIT

    def test_flush_on_commit(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        # Before commit, records may be in buffer
        wal.commit(1)
        # After commit, all should be flushed
        assert wal.buffer.pending_count == 0

    def test_checkpoint_captures_active_txns(self):
        wal = WALManager()
        wal.begin(1)
        wal.begin(2)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        wal.log_insert(2, "t", 0, 1, [2])
        wal.checkpoint()
        records = wal.get_all_records()
        cp = [r for r in records if r.log_type == LogType.CHECKPOINT][0]
        # Only txn 2 should be active at checkpoint time
        assert 2 in cp.active_txns
        assert 1 not in cp.active_txns


# =============================================================================
# Recovery Manager Tests
# =============================================================================

class TestRecoveryManager:
    def test_analysis_identifies_active_txns(self):
        wal = WALManager()
        wal.begin(1)
        wal.begin(2)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(2, "t", 0, 1, [2])
        wal.commit(1)
        # txn 2 still active
        rm = wal.create_recovery_manager()
        rm._analysis_pass()
        assert 2 in rm.txn_table
        assert rm.txn_table[2].status == 'active'

    def test_analysis_committed_txn_removed(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        rm = wal.create_recovery_manager()
        rm._analysis_pass()
        # txn 1 committed -- status should be committed
        assert rm.txn_table[1].status == 'committed'

    def test_analysis_dirty_pages_tracked(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "users", 3, 0, ["a"])
        wal.log_update(1, "users", 3, 0, ["a"], ["b"])
        wal.flush()
        rm = wal.create_recovery_manager()
        rm._analysis_pass()
        assert ("users", 3) in rm.dirty_page_table

    def test_analysis_from_checkpoint(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        wal.checkpoint()
        wal.begin(2)
        wal.log_insert(2, "t", 0, 1, [2])
        rm = wal.create_recovery_manager()
        rm._analysis_pass()
        assert 2 in rm.txn_table
        assert rm.txn_table[2].status == 'active'

    def test_recover_counts(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(1, "t", 0, 1, [2])
        # Don't commit -- txn 1 is active, needs undo
        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        # Should have undo operations for the 2 inserts
        assert undo == 2

    def test_recover_committed_no_undo(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 0

    def test_recover_multiple_txns(self):
        wal = WALManager()
        wal.begin(1)
        wal.begin(2)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(2, "t", 0, 1, [2])
        wal.commit(1)
        # txn 2 not committed
        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 1  # Only txn 2's insert undone

    def test_clr_in_undo_chain(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(1, "t", 0, 1, [2])
        wal.log_insert(1, "t", 0, 2, [3])
        # Simulate crash and recovery
        rm = wal.create_recovery_manager()
        rm.recover()
        # CLRs should have been written
        all_records = wal.get_all_records()
        clrs = [r for r in all_records if r.log_type == LogType.CLR]
        assert len(clrs) == 3  # One CLR per undone insert

    def test_end_records_written(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        rm = wal.create_recovery_manager()
        rm.recover()
        all_records = wal.get_all_records()
        ends = [r for r in all_records if r.log_type == LogType.END]
        assert len(ends) >= 1


# =============================================================================
# WAL Storage Engine Tests
# =============================================================================

class TestWALStorageEngine:
    def _make_engine(self):
        engine = WALStorageEngine()
        engine.create_table("users", ["name", "age"])
        return engine

    def _rows(self, engine, table):
        """Helper: scan and return row dicts."""
        return engine.scan_table(table)

    def test_create_table(self):
        engine = WALStorageEngine()
        engine.create_table("users", ["name", "age"])
        assert "users" in engine.engine._tables

    def test_insert_and_scan(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.insert(txn, "users", {"name": "bob", "age": 25})
        engine.commit(txn)
        rows = self._rows(engine, "users")
        assert len(rows) == 2

    def test_update(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        row_id = engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)

        txn2 = engine.begin_transaction()
        engine.update(txn2, "users", row_id, {"name": "alice", "age": 31})
        engine.commit(txn2)

        rows = self._rows(engine, "users")
        assert any(r["age"] == 31 for r in rows)

    def test_delete(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        row_id = engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)

        txn2 = engine.begin_transaction()
        engine.delete(txn2, "users", row_id)
        engine.commit(txn2)

        assert len(self._rows(engine, "users")) == 0

    def test_abort_insert(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.abort(txn)
        assert len(self._rows(engine, "users")) == 0

    def test_abort_update(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        row_id = engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)

        txn2 = engine.begin_transaction()
        engine.update(txn2, "users", row_id, {"name": "alice", "age": 99})
        engine.abort(txn2)

        rows = self._rows(engine, "users")
        assert rows[0]["age"] == 30  # Restored

    def test_abort_delete(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        row_id = engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)

        txn2 = engine.begin_transaction()
        engine.delete(txn2, "users", row_id)
        engine.abort(txn2)

        assert len(self._rows(engine, "users")) == 1

    def test_multiple_inserts_abort(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.insert(txn, "users", {"name": "bob", "age": 25})
        engine.insert(txn, "users", {"name": "charlie", "age": 35})
        engine.abort(txn)
        assert len(self._rows(engine, "users")) == 0

    def test_mixed_operations_abort(self):
        engine = self._make_engine()
        txn1 = engine.begin_transaction()
        r1 = engine.insert(txn1, "users", {"name": "alice", "age": 30})
        r2 = engine.insert(txn1, "users", {"name": "bob", "age": 25})
        engine.commit(txn1)

        txn2 = engine.begin_transaction()
        engine.update(txn2, "users", r1, {"name": "alice", "age": 99})
        engine.delete(txn2, "users", r2)
        engine.abort(txn2)

        rows = self._rows(engine, "users")
        assert len(rows) == 2
        names = sorted([r["name"] for r in rows])
        assert names == ["alice", "bob"]

    def test_wal_records_generated(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)
        records = engine.get_log_records()
        types = [r.log_type for r in records]
        assert LogType.BEGIN in types
        assert LogType.INSERT in types
        assert LogType.COMMIT in types

    def test_checkpoint(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.commit(txn)
        engine.checkpoint()
        records = engine.get_log_records()
        assert any(r.log_type == LogType.CHECKPOINT for r in records)

    def test_table_not_found(self):
        engine = WALStorageEngine()
        txn = engine.begin_transaction()
        with pytest.raises(KeyError):
            engine.insert(txn, "nonexistent", {"x": 1})

    def test_scan_nonexistent_table(self):
        engine = WALStorageEngine()
        with pytest.raises(KeyError):
            engine.scan_table("nonexistent")

    def test_concurrent_txns(self):
        engine = self._make_engine()
        txn1 = engine.begin_transaction()
        txn2 = engine.begin_transaction()
        engine.insert(txn1, "users", {"name": "alice", "age": 30})
        engine.insert(txn2, "users", {"name": "bob", "age": 25})
        engine.commit(txn1)
        engine.abort(txn2)
        rows = self._rows(engine, "users")
        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

    def test_multiple_tables(self):
        engine = WALStorageEngine()
        engine.create_table("users", ["name", "age"])
        engine.create_table("orders", ["user", "amount"])
        txn = engine.begin_transaction()
        engine.insert(txn, "users", {"name": "alice", "age": 30})
        engine.insert(txn, "orders", {"user": "alice", "amount": 100})
        engine.commit(txn)
        assert len(engine.scan_table("users")) == 1
        assert len(engine.scan_table("orders")) == 1

    def test_large_transaction(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        for i in range(100):
            engine.insert(txn, "users", {"name": f"user{i}", "age": i})
        engine.commit(txn)
        assert len(self._rows(engine, "users")) == 100

    def test_large_transaction_abort(self):
        engine = self._make_engine()
        txn = engine.begin_transaction()
        for i in range(50):
            engine.insert(txn, "users", {"name": f"user{i}", "age": i})
        engine.abort(txn)
        assert len(self._rows(engine, "users")) == 0

    def test_sequential_transactions(self):
        engine = self._make_engine()
        for i in range(10):
            txn = engine.begin_transaction()
            engine.insert(txn, "users", {"name": f"user{i}", "age": i})
            if i % 2 == 0:
                engine.commit(txn)
            else:
                engine.abort(txn)
        rows = self._rows(engine, "users")
        assert len(rows) == 5  # Only even-numbered committed


# =============================================================================
# WAL Analyzer Tests
# =============================================================================

class TestWALAnalyzer:
    def _make_wal_with_data(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "users", 0, 0, ["alice", 30])
        wal.log_insert(1, "users", 0, 1, ["bob", 25])
        wal.commit(1)
        wal.begin(2)
        wal.log_update(2, "users", 0, 0, ["alice", 30], ["alice", 31])
        wal.log_delete(2, "orders", 1, 0, ["item1"])
        wal.commit(2)
        return wal

    def test_transaction_summary(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        summary = analyzer.transaction_summary()
        assert 1 in summary
        assert 2 in summary
        assert summary[1]['status'] == 'committed'
        assert summary[2]['status'] == 'committed'

    def test_transaction_summary_operations(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        summary = analyzer.transaction_summary()
        assert summary[1]['operations']['INSERT'] == 2
        assert summary[2]['operations']['UPDATE'] == 1
        assert summary[2]['operations']['DELETE'] == 1

    def test_transaction_summary_tables(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        summary = analyzer.transaction_summary()
        assert "users" in summary[1]['tables']
        assert "orders" in summary[2]['tables']

    def test_page_history(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        history = analyzer.page_history("users", 0)
        assert len(history) == 3  # 2 inserts + 1 update

    def test_page_history_empty(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        history = analyzer.page_history("nonexistent", 99)
        assert len(history) == 0

    def test_log_statistics(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        stats = analyzer.log_statistics()
        assert stats['total_records'] > 0
        assert stats['unique_txns'] == 2
        assert stats['unique_tables'] == 2
        assert 'INSERT' in stats['by_type']
        assert 'COMMIT' in stats['by_type']

    def test_verify_integrity(self):
        wal = self._make_wal_with_data()
        analyzer = WALAnalyzer(wal)
        result = analyzer.verify_integrity()
        assert result['valid']
        assert result['record_count'] > 0

    def test_prev_lsn_chain(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(1, "t", 0, 1, [2])
        wal.log_insert(1, "t", 0, 2, [3])
        wal.commit(1)
        analyzer = WALAnalyzer(wal)
        chain = analyzer.prev_lsn_chain(1)
        assert len(chain) == 5  # commit + 3 inserts + begin
        # Chain should be in reverse LSN order
        for i in range(len(chain) - 1):
            assert chain[i]['lsn'] > chain[i+1]['lsn']

    def test_prev_lsn_chain_nonexistent(self):
        wal = WALManager()
        analyzer = WALAnalyzer(wal)
        chain = analyzer.prev_lsn_chain(99)
        assert chain == []

    def test_active_txn_summary(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        # Not committed
        analyzer = WALAnalyzer(wal)
        summary = analyzer.transaction_summary()
        assert summary[1]['status'] == 'active'

    def test_aborted_txn_summary(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.abort(1)
        analyzer = WALAnalyzer(wal)
        summary = analyzer.transaction_summary()
        assert summary[1]['status'] == 'aborted'


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def _rows(self, engine, table):
        return engine.scan_table(table)

    def test_full_lifecycle(self):
        """Full lifecycle: create, insert, update, delete, commit."""
        engine = WALStorageEngine()
        engine.create_table("items", ["name", "qty"])
        txn = engine.begin_transaction()
        r1 = engine.insert(txn, "items", {"name": "apple", "qty": 10})
        r2 = engine.insert(txn, "items", {"name": "banana", "qty": 20})
        engine.update(txn, "items", r1, {"name": "apple", "qty": 15})
        engine.delete(txn, "items", r2)
        engine.commit(txn)
        rows = self._rows(engine, "items")
        assert len(rows) == 1
        assert rows[0]["name"] == "apple"
        assert rows[0]["qty"] == 15

    def test_crash_recovery_simulation(self):
        """Simulate crash: active txn not committed, verify recovery identifies it."""
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.log_insert(1, "t", 0, 1, [2])
        wal.commit(1)

        wal.begin(2)
        wal.log_insert(2, "t", 0, 2, [3])
        # Crash! txn 2 not committed

        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 1  # txn 2's insert undone

    def test_checkpoint_recovery(self):
        """Recovery starting from checkpoint."""
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        wal.checkpoint()

        wal.begin(2)
        wal.log_insert(2, "t", 0, 1, [2])
        # Crash! txn 2 active

        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 1

    def test_multiple_checkpoints(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.checkpoint()
        wal.commit(1)
        wal.checkpoint()
        wal.begin(2)
        wal.log_insert(2, "t", 0, 1, [2])
        # Crash

        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 1

    def test_wal_and_engine_consistency(self):
        """WAL records match engine state."""
        engine = WALStorageEngine()
        engine.create_table("t", ["x"])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {"x": 1})
        engine.insert(txn, "t", {"x": 2})
        engine.insert(txn, "t", {"x": 3})
        engine.commit(txn)

        records = engine.get_log_records()
        inserts = [r for r in records if r.log_type == LogType.INSERT]
        assert len(inserts) == 3
        rows = engine.scan_table("t")
        assert len(rows) == 3

    def test_interleaved_transactions(self):
        """Multiple transactions interleaved."""
        engine = WALStorageEngine()
        engine.create_table("t", ["v"])
        txn1 = engine.begin_transaction()
        txn2 = engine.begin_transaction()
        txn3 = engine.begin_transaction()

        engine.insert(txn1, "t", {"v": "a"})
        engine.insert(txn2, "t", {"v": "b"})
        engine.insert(txn3, "t", {"v": "c"})
        engine.insert(txn1, "t", {"v": "d"})

        engine.commit(txn1)
        engine.abort(txn2)
        engine.commit(txn3)

        rows = self._rows(engine, "t")
        values = sorted([r["v"] for r in rows])
        assert values == ["a", "c", "d"]

    def test_analyzer_on_real_engine(self):
        """Analyzer works on real WAL from engine operations."""
        engine = WALStorageEngine()
        engine.create_table("t", ["x"])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {"x": 1})
        engine.commit(txn)

        analyzer = WALAnalyzer(engine.wal)
        stats = analyzer.log_statistics()
        assert stats['total_records'] >= 3
        integrity = analyzer.verify_integrity()
        assert integrity['valid']

    def test_update_restore_on_abort(self):
        """Abort after update restores original value."""
        engine = WALStorageEngine()
        engine.create_table("t", ["name", "val"])
        txn1 = engine.begin_transaction()
        rid = engine.insert(txn1, "t", {"name": "x", "val": 100})
        engine.commit(txn1)

        txn2 = engine.begin_transaction()
        rid = engine.update(txn2, "t", rid, {"name": "x", "val": 200})
        rid = engine.update(txn2, "t", rid, {"name": "x", "val": 300})
        engine.abort(txn2)

        rows = self._rows(engine, "t")
        assert rows[0]["val"] == 100

    def test_delete_restore_on_abort(self):
        """Abort after delete restores deleted row."""
        engine = WALStorageEngine()
        engine.create_table("t", ["v"])
        txn1 = engine.begin_transaction()
        rid = engine.insert(txn1, "t", {"v": "hello"})
        engine.commit(txn1)

        txn2 = engine.begin_transaction()
        engine.delete(txn2, "t", rid)
        engine.abort(txn2)

        rows = self._rows(engine, "t")
        assert len(rows) == 1
        assert rows[0]["v"] == "hello"

    def test_log_truncation(self):
        """WAL truncation after checkpoint."""
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        cp_lsn = wal.checkpoint()
        wal.begin(2)
        wal.log_insert(2, "t", 0, 1, [2])
        wal.commit(2)

        # Truncate everything before checkpoint
        wal.flush()
        wal.writer.truncate_after(cp_lsn)
        records = wal.get_all_records()
        # Only records up to checkpoint remain
        assert all(r.lsn <= cp_lsn for r in records)

    def test_serialization_stress(self):
        """Many diverse records serialize/deserialize correctly."""
        wal = WALManager()
        for i in range(20):
            wal.begin(i + 1)
            wal.log_insert(i + 1, f"table{i % 3}", i, i % 5, [i, f"val{i}", float(i)])
            if i % 3 == 0:
                wal.log_update(i + 1, f"table{i % 3}", i, i % 5,
                              [i, f"val{i}", float(i)],
                              [i, f"new{i}", float(i) + 0.5])
            if i % 2 == 0:
                wal.commit(i + 1)
            else:
                wal.abort(i + 1)

        records = wal.get_all_records()
        assert len(records) > 40

        analyzer = WALAnalyzer(wal)
        integrity = analyzer.verify_integrity()
        assert integrity['valid']

    def test_empty_wal_recovery(self):
        """Recovery on empty WAL succeeds with no operations."""
        wal = WALManager()
        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert redo == 0
        assert undo == 0

    def test_all_committed_recovery(self):
        """Recovery with all committed transactions -- no undo needed."""
        wal = WALManager()
        for i in range(5):
            wal.begin(i + 1)
            wal.log_insert(i + 1, "t", 0, i, [i])
            wal.commit(i + 1)
        rm = wal.create_recovery_manager()
        redo, undo = rm.recover()
        assert undo == 0

    def test_txn_id_monotonic(self):
        engine = WALStorageEngine()
        ids = [engine.begin_transaction() for _ in range(5)]
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    def _rows(self, engine, table):
        return engine.scan_table(table)

    def test_empty_row_insert(self):
        engine = WALStorageEngine()
        engine.create_table("t", [])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {})
        engine.commit(txn)

    def test_null_values_in_row(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["a", "b"])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {"a": None, "b": None})
        engine.commit(txn)
        rows = self._rows(engine, "t")
        assert rows[0]["a"] is None
        assert rows[0]["b"] is None

    def test_boolean_values(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["flag"])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {"flag": True})
        engine.insert(txn, "t", {"flag": False})
        engine.commit(txn)
        rows = self._rows(engine, "t")
        vals = sorted([r["flag"] for r in rows])
        assert vals == [False, True]

    def test_float_values(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["x"])
        txn = engine.begin_transaction()
        engine.insert(txn, "t", {"x": 3.14159})
        engine.commit(txn)
        rows = self._rows(engine, "t")
        assert abs(rows[0]["x"] - 3.14159) < 1e-10

    def test_long_string_values(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["data"])
        txn = engine.begin_transaction()
        long_str = "x" * 1000
        engine.insert(txn, "t", {"data": long_str})
        engine.commit(txn)
        rows = self._rows(engine, "t")
        assert rows[0]["data"] == long_str

    def test_begin_without_operations(self):
        wal = WALManager()
        wal.begin(1)
        wal.commit(1)
        records = wal.get_all_records()
        assert len(records) == 2

    def test_multiple_aborts(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["x"])
        for i in range(10):
            txn = engine.begin_transaction()
            engine.insert(txn, "t", {"x": i})
            engine.abort(txn)
        assert len(engine.scan_table("t")) == 0

    def test_commit_then_new_txn(self):
        engine = WALStorageEngine()
        engine.create_table("t", ["x"])
        for i in range(5):
            txn = engine.begin_transaction()
            engine.insert(txn, "t", {"x": i})
            engine.commit(txn)
        assert len(engine.scan_table("t")) == 5

    def test_analyzer_empty_wal(self):
        wal = WALManager()
        analyzer = WALAnalyzer(wal)
        stats = analyzer.log_statistics()
        assert stats['total_records'] == 0

    def test_page_history_multiple_txns(self):
        wal = WALManager()
        wal.begin(1)
        wal.log_insert(1, "t", 0, 0, [1])
        wal.commit(1)
        wal.begin(2)
        wal.log_update(2, "t", 0, 0, [1], [2])
        wal.commit(2)
        wal.begin(3)
        wal.log_delete(3, "t", 0, 0, [2])
        wal.commit(3)
        analyzer = WALAnalyzer(wal)
        history = analyzer.page_history("t", 0)
        assert len(history) == 3
        assert history[0]['type'] == 'INSERT'
        assert history[1]['type'] == 'UPDATE'
        assert history[2]['type'] == 'DELETE'

    def test_lsn_default_comparison(self):
        assert LSN() == LSN(0)
        assert LSN(0) < LSN(1)


# =============================================================================
# TransactionEntry Tests
# =============================================================================

class TestTransactionEntry:
    def test_create(self):
        e = TransactionEntry(1, 'active', LSN(5))
        assert e.txn_id == 1
        assert e.status == 'active'
        assert e.last_lsn == LSN(5)

    def test_defaults(self):
        e = TransactionEntry(1)
        assert e.status == 'active'
        assert e.last_lsn is None
        assert e.undo_next_lsn is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
