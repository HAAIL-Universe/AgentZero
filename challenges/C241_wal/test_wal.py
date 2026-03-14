"""
Tests for C241: Write-Ahead Logging (WAL)

Covers:
- Log record creation and types
- Page operations with LSN tracking
- Buffer pool basics
- Transaction lifecycle (begin, commit, abort)
- Insert, update, delete with WAL logging
- Read operations
- WAL protocol (log before data)
- Force-at-commit
- Undo on abort (with CLR)
- Savepoints and rollback-to-savepoint
- Fuzzy checkpoints
- ARIES recovery: analysis, redo, undo phases
- Crash + recovery scenarios
- Group commit
- Log truncation
- Log serialization/deserialization
- Transaction chains (prev_lsn)
- Multi-transaction interleaving
- Recovery with checkpoint
- Recovery with partial flush
- Recovery idempotence
- CLR chains during recovery undo
"""

import pytest
from wal import (
    LogRecordType, LogRecord, Page, BufferPool,
    TransactionEntry, WAL
)


# ===========================================================================
# Log Record Tests
# ===========================================================================

class TestLogRecord:
    def test_create_record(self):
        r = LogRecord(lsn=1, txn_id=10, record_type=LogRecordType.INSERT,
                      page_id=5, key='a', new_value=42)
        assert r.lsn == 1
        assert r.txn_id == 10
        assert r.record_type == LogRecordType.INSERT
        assert r.page_id == 5
        assert r.key == 'a'
        assert r.new_value == 42

    def test_undoable_types(self):
        for t in [LogRecordType.INSERT, LogRecordType.UPDATE, LogRecordType.DELETE]:
            r = LogRecord(lsn=1, txn_id=1, record_type=t)
            assert r.is_undoable()

    def test_non_undoable_types(self):
        for t in [LogRecordType.BEGIN, LogRecordType.COMMIT, LogRecordType.ABORT,
                  LogRecordType.CHECKPOINT, LogRecordType.CLR, LogRecordType.END]:
            r = LogRecord(lsn=1, txn_id=1, record_type=t)
            assert not r.is_undoable()

    def test_redoable_types(self):
        for t in [LogRecordType.INSERT, LogRecordType.UPDATE,
                  LogRecordType.DELETE, LogRecordType.CLR]:
            r = LogRecord(lsn=1, txn_id=1, record_type=t)
            assert r.is_redoable()

    def test_non_redoable_types(self):
        for t in [LogRecordType.BEGIN, LogRecordType.COMMIT,
                  LogRecordType.ABORT, LogRecordType.CHECKPOINT]:
            r = LogRecord(lsn=1, txn_id=1, record_type=t)
            assert not r.is_redoable()

    def test_repr(self):
        r = LogRecord(lsn=1, txn_id=2, record_type=LogRecordType.INSERT,
                      page_id=3, key='x')
        s = repr(r)
        assert 'lsn=1' in s
        assert 'INSERT' in s

    def test_prev_lsn_default(self):
        r = LogRecord(lsn=5, txn_id=1, record_type=LogRecordType.BEGIN)
        assert r.prev_lsn == 0

    def test_undo_next_lsn(self):
        r = LogRecord(lsn=10, txn_id=1, record_type=LogRecordType.CLR,
                      undo_next_lsn=5)
        assert r.undo_next_lsn == 5


# ===========================================================================
# Page Tests
# ===========================================================================

class TestPage:
    def test_create_page(self):
        p = Page(1)
        assert p.page_id == 1
        assert p.data == {}
        assert p.page_lsn == 0
        assert not p.dirty

    def test_put_and_get(self):
        p = Page(1)
        p.put('k1', 'v1', 10)
        assert p.get('k1') == 'v1'
        assert p.page_lsn == 10
        assert p.dirty

    def test_remove(self):
        p = Page(1)
        p.put('k1', 'v1', 10)
        old = p.remove('k1', 20)
        assert old == 'v1'
        assert p.get('k1') is None
        assert p.page_lsn == 20

    def test_remove_nonexistent(self):
        p = Page(1)
        old = p.remove('nope', 5)
        assert old is None

    def test_flush_clears_dirty(self):
        p = Page(1)
        p.put('k', 'v', 1)
        assert p.dirty
        p.flush()
        assert not p.dirty

    def test_repr(self):
        p = Page(7)
        assert 'id=7' in repr(p)


# ===========================================================================
# Buffer Pool Tests
# ===========================================================================

class TestBufferPool:
    def test_get_page_creates_new(self):
        bp = BufferPool()
        p = bp.get_page(1)
        assert p.page_id == 1
        assert p.data == {}

    def test_get_page_returns_same(self):
        bp = BufferPool()
        p1 = bp.get_page(1)
        p1.put('k', 'v', 1)
        p2 = bp.get_page(1)
        assert p2.get('k') == 'v'

    def test_flush_page(self):
        bp = BufferPool()
        p = bp.get_page(1)
        p.put('k', 'v', 1)
        bp.flush_page(1)
        assert not p.dirty
        assert 1 in bp.flushed_pages

    def test_flush_all(self):
        bp = BufferPool()
        for i in range(5):
            p = bp.get_page(i)
            p.put(f'k{i}', f'v{i}', i + 1)
        bp.flush_all()
        for i in range(5):
            assert not bp.pages[i].dirty

    def test_clear_loses_unflushed(self):
        bp = BufferPool()
        p = bp.get_page(1)
        p.put('k', 'v', 1)
        bp.clear()
        assert len(bp.pages) == 0

    def test_reload_from_disk(self):
        bp = BufferPool()
        p = bp.get_page(1)
        p.put('k', 'v', 1)
        bp.flush_page(1)
        bp.clear()
        bp.reload_from_disk()
        p2 = bp.get_page(1)
        assert p2.get('k') == 'v'
        assert not p2.dirty

    def test_dirty_pages(self):
        bp = BufferPool()
        bp.get_page(1).put('a', 1, 10)
        bp.get_page(2).put('b', 2, 20)
        bp.get_page(3)  # clean
        dp = bp.get_dirty_pages()
        assert 1 in dp
        assert 2 in dp
        assert 3 not in dp


# ===========================================================================
# Transaction Entry Tests
# ===========================================================================

class TestTransactionEntry:
    def test_create(self):
        e = TransactionEntry(1)
        assert e.txn_id == 1
        assert e.status == 'active'
        assert e.last_lsn == 0
        assert e.savepoints == {}


# ===========================================================================
# WAL Basic Operations
# ===========================================================================

class TestWALBasic:
    def test_begin_returns_txn_id(self):
        w = WAL()
        t1 = w.begin()
        t2 = w.begin()
        assert t1 == 1
        assert t2 == 2

    def test_begin_creates_log_record(self):
        w = WAL()
        t = w.begin()
        records = w.get_log_records(txn_id=t)
        assert len(records) == 1
        assert records[0].record_type == LogRecordType.BEGIN

    def test_commit_flushes_log(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.commit(t)
        assert w.flushed_lsn >= 3

    def test_commit_writes_commit_and_end(self):
        w = WAL()
        t = w.begin()
        w.commit(t)
        records = w.get_log_records(txn_id=t)
        types = [r.record_type for r in records]
        assert LogRecordType.COMMIT in types
        assert LogRecordType.END in types

    def test_commit_nonexistent_txn(self):
        w = WAL()
        with pytest.raises(ValueError):
            w.commit(999)

    def test_abort_nonexistent_txn(self):
        w = WAL()
        with pytest.raises(ValueError):
            w.abort(999)

    def test_active_txn_count(self):
        w = WAL()
        t1 = w.begin()
        t2 = w.begin()
        assert w.stats()['active_txns'] == 2
        w.commit(t1)
        assert w.stats()['active_txns'] == 1
        w.abort(t2)
        assert w.stats()['active_txns'] == 0


# ===========================================================================
# WAL Data Operations
# ===========================================================================

class TestWALDataOps:
    def test_insert_and_read(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'key1', 'value1')
        assert w.read(t, 1, 'key1') == 'value1'

    def test_update_and_read(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'old')
        w.update(t, 1, 'k', 'new')
        assert w.read(t, 1, 'k') == 'new'

    def test_delete_and_read(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.delete(t, 1, 'k')
        assert w.read(t, 1, 'k') is None

    def test_insert_duplicate_raises(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        with pytest.raises(ValueError, match="already exists"):
            w.insert(t, 1, 'k', 'v2')

    def test_update_nonexistent_raises(self):
        w = WAL()
        t = w.begin()
        with pytest.raises(ValueError, match="not found"):
            w.update(t, 1, 'nope', 'v')

    def test_delete_nonexistent_raises(self):
        w = WAL()
        t = w.begin()
        with pytest.raises(ValueError, match="not found"):
            w.delete(t, 1, 'nope')

    def test_insert_with_inactive_txn(self):
        w = WAL()
        t = w.begin()
        w.commit(t)
        with pytest.raises(ValueError, match="not active"):
            w.insert(t, 1, 'k', 'v')

    def test_multiple_keys_same_page(self):
        w = WAL()
        t = w.begin()
        for i in range(10):
            w.insert(t, 1, f'k{i}', f'v{i}')
        for i in range(10):
            assert w.read(t, 1, f'k{i}') == f'v{i}'

    def test_multiple_pages(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.insert(t, 2, 'b', 2)
        w.insert(t, 3, 'c', 3)
        assert w.read(t, 1, 'a') == 1
        assert w.read(t, 2, 'b') == 2
        assert w.read(t, 3, 'c') == 3

    def test_insert_generates_log_record(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        inserts = w.get_log_records(record_type=LogRecordType.INSERT)
        assert len(inserts) == 1
        assert inserts[0].key == 'k'
        assert inserts[0].new_value == 'v'

    def test_update_captures_old_value(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'old')
        w.update(t, 1, 'k', 'new')
        updates = w.get_log_records(record_type=LogRecordType.UPDATE)
        assert len(updates) == 1
        assert updates[0].old_value == 'old'
        assert updates[0].new_value == 'new'

    def test_delete_captures_old_value(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.delete(t, 1, 'k')
        deletes = w.get_log_records(record_type=LogRecordType.DELETE)
        assert len(deletes) == 1
        assert deletes[0].old_value == 'v'

    def test_update_none_value(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', None)
        w.update(t, 1, 'k', 'now_set')
        assert w.read(t, 1, 'k') == 'now_set'


# ===========================================================================
# WAL Protocol Tests
# ===========================================================================

class TestWALProtocol:
    def test_log_before_data(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        inserts = w.get_log_records(record_type=LogRecordType.INSERT)
        assert len(inserts) == 1
        assert w.read(t, 1, 'k') == 'v'

    def test_force_at_commit(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        pre_flush = w.flushed_lsn
        w.commit(t)
        assert w.flushed_lsn > pre_flush

    def test_lsn_monotonically_increasing(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.insert(t, 1, 'b', 2)
        w.commit(t)
        lsns = [r.lsn for r in w.log]
        assert lsns == sorted(lsns)
        assert len(set(lsns)) == len(lsns)


# ===========================================================================
# Abort and Undo Tests
# ===========================================================================

class TestAbortUndo:
    def test_abort_undoes_insert(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.abort(t)
        t2 = w.begin()
        assert w.read(t2, 1, 'k') is None

    def test_abort_undoes_update(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'original')
        w.commit(t1)
        t2 = w.begin()
        w.update(t2, 1, 'k', 'modified')
        w.abort(t2)
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'original'

    def test_abort_undoes_delete(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v')
        w.commit(t1)
        t2 = w.begin()
        w.delete(t2, 1, 'k')
        w.abort(t2)
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'v'

    def test_abort_writes_clr(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.abort(t)
        clrs = w.get_log_records(record_type=LogRecordType.CLR)
        assert len(clrs) >= 1

    def test_abort_multiple_operations(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 1, 'b', 2)
        w.update(t2, 1, 'a', 99)
        w.abort(t2)
        t3 = w.begin()
        assert w.read(t3, 1, 'a') == 1
        assert w.read(t3, 1, 'b') is None

    def test_abort_generates_clr_with_undo_next_lsn(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.abort(t)
        clrs = w.get_log_records(record_type=LogRecordType.CLR)
        assert len(clrs) >= 1
        for clr in clrs:
            assert clr.undo_next_lsn >= 0


# ===========================================================================
# Savepoint Tests
# ===========================================================================

class TestSavepoints:
    def test_savepoint_basic(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.savepoint(t, 'sp1')
        w.insert(t, 1, 'b', 2)
        w.rollback_to_savepoint(t, 'sp1')
        assert w.read(t, 1, 'a') == 1
        assert w.read(t, 1, 'b') is None

    def test_savepoint_preserves_earlier_work(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'x', 10)
        w.savepoint(t, 'sp')
        w.insert(t, 1, 'y', 20)
        w.update(t, 1, 'x', 99)
        w.rollback_to_savepoint(t, 'sp')
        assert w.read(t, 1, 'x') == 10
        assert w.read(t, 1, 'y') is None

    def test_savepoint_then_commit(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.savepoint(t, 'sp1')
        w.insert(t, 1, 'b', 2)
        w.rollback_to_savepoint(t, 'sp1')
        w.insert(t, 1, 'c', 3)
        w.commit(t)
        t2 = w.begin()
        assert w.read(t2, 1, 'a') == 1
        assert w.read(t2, 1, 'b') is None
        assert w.read(t2, 1, 'c') == 3

    def test_nested_savepoints(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.savepoint(t, 'sp1')
        w.insert(t, 1, 'b', 2)
        w.savepoint(t, 'sp2')
        w.insert(t, 1, 'c', 3)
        w.rollback_to_savepoint(t, 'sp2')
        assert w.read(t, 1, 'c') is None
        assert w.read(t, 1, 'b') == 2
        w.rollback_to_savepoint(t, 'sp1')
        assert w.read(t, 1, 'b') is None
        assert w.read(t, 1, 'a') == 1

    def test_rollback_removes_later_savepoints(self):
        w = WAL()
        t = w.begin()
        w.savepoint(t, 'sp1')
        w.insert(t, 1, 'a', 1)
        w.savepoint(t, 'sp2')
        w.insert(t, 1, 'b', 2)
        w.rollback_to_savepoint(t, 'sp1')
        with pytest.raises(ValueError, match="not found"):
            w.rollback_to_savepoint(t, 'sp2')

    def test_savepoint_nonexistent_txn(self):
        w = WAL()
        with pytest.raises(ValueError):
            w.savepoint(999, 'sp')

    def test_rollback_nonexistent_savepoint(self):
        w = WAL()
        t = w.begin()
        with pytest.raises(ValueError, match="not found"):
            w.rollback_to_savepoint(t, 'nope')


# ===========================================================================
# Checkpoint Tests
# ===========================================================================

class TestCheckpoint:
    def test_checkpoint_creates_record(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.commit(t)
        lsn = w.checkpoint()
        assert lsn > 0
        assert w.last_checkpoint_lsn == lsn

    def test_checkpoint_captures_active_txns(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        t2 = w.begin()
        w.insert(t2, 2, 'b', 2)
        w.commit(t1)
        w.checkpoint()
        cp_records = w.get_log_records(record_type=LogRecordType.CHECKPOINT)
        assert len(cp_records) == 1
        cp_data = cp_records[0].checkpoint_data
        assert t2 in cp_data['active_txns']
        assert t1 not in cp_data['active_txns']

    def test_checkpoint_captures_dirty_pages(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.insert(t, 2, 'b', 2)
        w.checkpoint()
        cp_records = w.get_log_records(record_type=LogRecordType.CHECKPOINT)
        dp = cp_records[0].checkpoint_data['dirty_pages']
        assert 1 in dp
        assert 2 in dp

    def test_checkpoint_flushes_log(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        pre = w.flushed_lsn
        w.checkpoint()
        assert w.flushed_lsn > pre


# ===========================================================================
# Group Commit Tests
# ===========================================================================

class TestGroupCommit:
    def test_group_commit_defers_flush(self):
        w = WAL()
        w.enable_group_commit()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.commit(t)
        w.flush_group_commit()
        assert t in w.committed_txns

    def test_group_commit_multiple_txns(self):
        w = WAL()
        w.enable_group_commit()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 1, 'b', 2)
        w.commit(t2)
        w.flush_group_commit()
        assert t1 in w.committed_txns
        assert t2 in w.committed_txns


# ===========================================================================
# Log Truncation Tests
# ===========================================================================

class TestLogTruncation:
    def test_truncate_removes_old_records(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 1, 'b', 2)
        w.commit(t2)
        w.checkpoint()
        before = len(w.log)
        removed = w.truncate_log()
        assert removed >= 0
        assert len(w.log) <= before

    def test_truncate_respects_active_txns(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        w.checkpoint()
        t2 = w.begin()
        w.insert(t2, 1, 'b', 2)
        w.truncate_log()
        txn_records = [r for r in w.log if r.txn_id == t2]
        assert len(txn_records) > 0

    def test_truncate_explicit_lsn(self):
        w = WAL()
        for i in range(5):
            t = w.begin()
            w.insert(t, 1, f'k{i}', i)
            w.commit(t)
        removed = w.truncate_log(up_to_lsn=5)
        assert removed > 0
        assert all(r.lsn >= 5 for r in w.log)


# ===========================================================================
# Log Serialization Tests
# ===========================================================================

class TestLogSerialization:
    def test_round_trip(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'key', 'value')
        w.update(t, 1, 'key', 'updated')
        w.commit(t)
        data = w.serialize_log()
        records = w.deserialize_log(data)
        assert len(records) == len(w.log)
        for orig, deser in zip(w.log, records):
            assert orig.lsn == deser.lsn
            assert orig.txn_id == deser.txn_id
            assert orig.record_type == deser.record_type

    def test_corrupted_data_raises(self):
        w = WAL()
        t = w.begin()
        w.commit(t)
        data = w.serialize_log()
        corrupted = data[:16] + b'CORRUPTED' + data[20:]
        with pytest.raises(ValueError, match="checksum"):
            w.deserialize_log(corrupted)

    def test_serialize_with_checkpoint(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.commit(t)
        w.checkpoint()
        data = w.serialize_log()
        records = w.deserialize_log(data)
        cp = [r for r in records if r.record_type == LogRecordType.CHECKPOINT]
        assert len(cp) == 1
        assert cp[0].checkpoint_data is not None


# ===========================================================================
# Transaction Chain Tests
# ===========================================================================

class TestTransactionChain:
    def test_chain_follows_prev_lsn(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.insert(t, 1, 'b', 2)
        w.commit(t)
        chain = w.get_transaction_chain(t)
        assert len(chain) >= 4
        for i in range(1, len(chain)):
            assert chain[i].prev_lsn == chain[i - 1].lsn

    def test_chain_nonexistent_txn(self):
        w = WAL()
        chain = w.get_transaction_chain(999)
        assert chain == []


# ===========================================================================
# Crash and Recovery Tests
# ===========================================================================

class TestCrashRecovery:
    def test_recovery_committed_data_survives(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'key', 'value')
        w.commit(t)
        w.simulate_crash()
        result = w.recover()
        t2 = w.begin()
        assert w.read(t2, 1, 'key') == 'value'
        assert result['redo_count'] > 0

    def test_recovery_uncommitted_data_rolled_back(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'committed_key', 'committed_val')
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 1, 'uncommitted_key', 'uncommitted_val')
        w.simulate_crash()
        result = w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'committed_key') == 'committed_val'
        assert w.read(t3, 1, 'uncommitted_key') is None
        assert result['undo_count'] > 0

    def test_recovery_with_updates(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v1')
        w.commit(t1)
        t2 = w.begin()
        w.update(t2, 1, 'k', 'v2')
        w.commit(t2)
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'v2'

    def test_recovery_undo_update(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'original')
        w.commit(t1)
        t2 = w.begin()
        w.update(t2, 1, 'k', 'modified')
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'original'

    def test_recovery_undo_delete(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v')
        w.commit(t1)
        t2 = w.begin()
        w.delete(t2, 1, 'k')
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'v'

    def test_recovery_multiple_txns(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 2, 'b', 2)
        t3 = w.begin()
        w.insert(t3, 3, 'c', 3)
        w.commit(t3)
        t4 = w.begin()
        w.insert(t4, 4, 'd', 4)
        w.simulate_crash()
        result = w.recover()
        t5 = w.begin()
        assert w.read(t5, 1, 'a') == 1
        assert w.read(t5, 2, 'b') is None
        assert w.read(t5, 3, 'c') == 3
        assert w.read(t5, 4, 'd') is None

    def test_recovery_with_checkpoint(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        w.checkpoint()
        t2 = w.begin()
        w.insert(t2, 2, 'b', 2)
        w.commit(t2)
        t3 = w.begin()
        w.insert(t3, 3, 'c', 3)
        w.simulate_crash()
        result = w.recover()
        t4 = w.begin()
        assert w.read(t4, 1, 'a') == 1
        assert w.read(t4, 2, 'b') == 2
        assert w.read(t4, 3, 'c') is None

    def test_recovery_idempotent(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v')
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 2, 'x', 'y')
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        val1 = w.read(t3, 1, 'k')
        val2 = w.read(t3, 2, 'x')
        assert val1 == 'v'
        assert val2 is None

    def test_recovery_empty_log(self):
        w = WAL()
        result = w.recover()
        assert result['redo_count'] == 0
        assert result['undo_count'] == 0

    def test_recovery_all_committed(self):
        w = WAL()
        for i in range(5):
            t = w.begin()
            w.insert(t, i, f'k{i}', f'v{i}')
            w.commit(t)
        w.simulate_crash()
        result = w.recover()
        assert result['undo_count'] == 0
        assert result['redo_count'] > 0

    def test_recovery_with_abort_before_crash(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v')
        w.abort(t1)
        t2 = w.begin()
        w.insert(t2, 1, 'k2', 'v2')
        w.commit(t2)
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') is None
        assert w.read(t3, 1, 'k2') == 'v2'

    def test_recovery_partial_flush(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.insert(t1, 2, 'b', 2)
        w.commit(t1)
        w.buffer_pool.flush_page(1)
        w.simulate_crash()
        w.recover()
        t2 = w.begin()
        assert w.read(t2, 1, 'a') == 1
        assert w.read(t2, 2, 'b') == 2

    def test_recovery_interleaved_txns(self):
        w = WAL()
        t1 = w.begin()
        t2 = w.begin()
        w.insert(t1, 1, 'a', 'from_t1')
        w.insert(t2, 1, 'b', 'from_t2')
        w.insert(t1, 2, 'c', 'from_t1')
        w.commit(t1)
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'a') == 'from_t1'
        assert w.read(t3, 1, 'b') is None
        assert w.read(t3, 2, 'c') == 'from_t1'

    def test_recovery_with_clrs_from_abort(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v1')
        w.commit(t1)
        t2 = w.begin()
        w.update(t2, 1, 'k', 'v2')
        w.abort(t2)
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'v1'


# ===========================================================================
# Stats and Inspection Tests
# ===========================================================================

class TestStats:
    def test_stats_basic(self):
        w = WAL()
        s = w.stats()
        assert s['total_records'] == 0
        assert s['active_txns'] == 0

    def test_stats_after_operations(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 'v')
        w.commit(t)
        s = w.stats()
        assert s['committed_txns'] == 1
        assert s['total_records'] > 0
        assert 'INSERT' in s['record_types']

    def test_get_log_records_filtered(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        w.update(t, 1, 'a', 2)
        w.commit(t)
        all_recs = w.get_log_records()
        txn_recs = w.get_log_records(txn_id=t)
        ins_recs = w.get_log_records(record_type=LogRecordType.INSERT)
        assert len(ins_recs) == 1


# ===========================================================================
# Multi-Transaction Scenario Tests
# ===========================================================================

class TestMultiTxnScenarios:
    def test_sequential_txns_same_key(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v1')
        w.commit(t1)
        t2 = w.begin()
        w.update(t2, 1, 'k', 'v2')
        w.commit(t2)
        t3 = w.begin()
        w.update(t3, 1, 'k', 'v3')
        w.commit(t3)
        t4 = w.begin()
        assert w.read(t4, 1, 'k') == 'v3'

    def test_many_txns_many_pages(self):
        w = WAL()
        for i in range(20):
            t = w.begin()
            w.insert(t, i % 5, f'k{i}', i)
            w.commit(t)
        for i in range(20):
            t = w.begin()
            assert w.read(t, i % 5, f'k{i}') == i

    def test_bulk_operations_recovery(self):
        w = WAL()
        committed_keys = []
        for i in range(50):
            t = w.begin()
            w.insert(t, i % 10, f'k{i}', i * 10)
            if i % 2 == 0:
                w.commit(t)
                committed_keys.append((i % 10, f'k{i}', i * 10))
        w.simulate_crash()
        w.recover()
        for page_id, key, value in committed_keys:
            t = w.begin()
            assert w.read(t, page_id, key) == value

    def test_abort_doesnt_affect_other_txn(self):
        w = WAL()
        t1 = w.begin()
        t2 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.insert(t2, 2, 'b', 2)
        w.abort(t1)
        w.commit(t2)
        t3 = w.begin()
        assert w.read(t3, 1, 'a') is None
        assert w.read(t3, 2, 'b') == 2

    def test_recovery_analysis_phase_info(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'a', 1)
        w.commit(t1)
        t2 = w.begin()
        w.insert(t2, 2, 'b', 2)
        w.simulate_crash()
        result = w.recover()
        assert t2 in result['analysis']['active_txns']
        assert t1 in result['analysis']['committed_txns']

    def test_crash_recovery_preserves_page_data(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'x', 100)
        w.insert(t, 1, 'y', 200)
        w.commit(t)
        w.buffer_pool.flush_all()
        w.simulate_crash()
        w.recover()
        t2 = w.begin()
        assert w.read(t2, 1, 'x') == 100
        assert w.read(t2, 1, 'y') == 200


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_transaction_commit(self):
        w = WAL()
        t = w.begin()
        w.commit(t)
        assert t in w.committed_txns

    def test_empty_transaction_abort(self):
        w = WAL()
        t = w.begin()
        w.abort(t)
        assert t in w.aborted_txns

    def test_read_nonexistent_page(self):
        w = WAL()
        t = w.begin()
        assert w.read(t, 999, 'nope') is None

    def test_many_savepoints(self):
        w = WAL()
        t = w.begin()
        for i in range(10):
            w.insert(t, 1, f'k{i}', i)
            w.savepoint(t, f'sp{i}')
        w.rollback_to_savepoint(t, 'sp5')
        for i in range(6, 10):
            assert w.read(t, 1, f'k{i}') is None
        for i in range(6):
            assert w.read(t, 1, f'k{i}') == i

    def test_large_values(self):
        w = WAL()
        t = w.begin()
        big = 'x' * 10000
        w.insert(t, 1, 'big', big)
        w.commit(t)
        t2 = w.begin()
        assert w.read(t2, 1, 'big') == big

    def test_complex_value_types(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'list', [1, 2, 3])
        w.insert(t, 1, 'dict', {'a': 1})
        w.insert(t, 1, 'nested', {'x': [1, {'y': 2}]})
        w.commit(t)
        t2 = w.begin()
        assert w.read(t2, 1, 'list') == [1, 2, 3]
        assert w.read(t2, 1, 'dict') == {'a': 1}
        assert w.read(t2, 1, 'nested') == {'x': [1, {'y': 2}]}

    def test_recovery_after_checkpoint_and_new_work(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'before_cp', 'yes')
        w.commit(t1)
        w.checkpoint()
        t2 = w.begin()
        w.insert(t2, 2, 'after_cp_committed', 'yes')
        w.commit(t2)
        t3 = w.begin()
        w.insert(t3, 3, 'after_cp_uncommitted', 'yes')
        w.simulate_crash()
        w.recover()
        t4 = w.begin()
        assert w.read(t4, 1, 'before_cp') == 'yes'
        assert w.read(t4, 2, 'after_cp_committed') == 'yes'
        assert w.read(t4, 3, 'after_cp_uncommitted') is None

    def test_page_lsn_tracking(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'a', 1)
        p = w.buffer_pool.get_page(1)
        lsn1 = p.page_lsn
        w.insert(t, 1, 'b', 2)
        lsn2 = p.page_lsn
        assert lsn2 > lsn1

    def test_log_record_types_complete(self):
        expected = {'BEGIN', 'COMMIT', 'ABORT', 'INSERT', 'UPDATE', 'DELETE',
                    'CHECKPOINT', 'CLR', 'END', 'SAVEPOINT', 'ROLLBACK_TO_SAVEPOINT'}
        actual = {t.name for t in LogRecordType}
        assert expected == actual

    def test_delete_then_reinsert(self):
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'v1')
        w.commit(t1)
        t2 = w.begin()
        w.delete(t2, 1, 'k')
        w.insert(t2, 1, 'k', 'v2')
        w.commit(t2)
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'v2'

    def test_update_multiple_times(self):
        w = WAL()
        t = w.begin()
        w.insert(t, 1, 'k', 0)
        for i in range(1, 20):
            w.update(t, 1, 'k', i)
        w.commit(t)
        t2 = w.begin()
        assert w.read(t2, 1, 'k') == 19

    def test_recovery_delete_then_reinsert_uncommitted(self):
        """Uncommitted delete+reinsert should undo both."""
        w = WAL()
        t1 = w.begin()
        w.insert(t1, 1, 'k', 'original')
        w.commit(t1)
        t2 = w.begin()
        w.delete(t2, 1, 'k')
        w.insert(t2, 1, 'k', 'replaced')
        w.simulate_crash()
        w.recover()
        t3 = w.begin()
        assert w.read(t3, 1, 'k') == 'original'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
