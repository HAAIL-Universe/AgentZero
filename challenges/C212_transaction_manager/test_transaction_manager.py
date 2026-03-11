"""
Tests for C212: Transaction Manager
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from transaction_manager import (
    TransactionManager, TransactionalDatabase, Transaction,
    MVCCTable, RowVersion, Snapshot, Savepoint, WAL, WALRecord,
    WALRecordType, LockManager, LockMode, LockEntry,
    IsolationLevel, TxState,
    TransactionError, DeadlockError, LockConflictError,
    SerializationError, SavepointError, TransactionAbortedError,
)


# ============================================================
# WAL Tests
# ============================================================

class TestWAL:
    def test_append_and_lsn(self):
        wal = WAL()
        r1 = wal.append(1, WALRecordType.BEGIN)
        r2 = wal.append(1, WALRecordType.INSERT, table_name="t", row_id=1)
        assert r1.lsn == 1
        assert r2.lsn == 2
        assert wal.last_lsn == 2

    def test_get_tx_records(self):
        wal = WAL()
        wal.append(1, WALRecordType.BEGIN)
        wal.append(2, WALRecordType.BEGIN)
        wal.append(1, WALRecordType.INSERT, table_name="t", row_id=1)
        wal.append(2, WALRecordType.INSERT, table_name="t", row_id=2)
        wal.append(1, WALRecordType.COMMIT)
        recs = wal.get_tx_records(1)
        assert len(recs) == 3
        assert recs[0].record_type == WALRecordType.BEGIN
        assert recs[-1].record_type == WALRecordType.COMMIT

    def test_get_tx_records_since(self):
        wal = WAL()
        wal.append(1, WALRecordType.BEGIN)
        wal.append(1, WALRecordType.INSERT, table_name="t", row_id=1)
        wal.append(1, WALRecordType.INSERT, table_name="t", row_id=2)
        recs = wal.get_tx_records_since(1, 1)
        assert len(recs) == 2  # LSN 2 and 3

    def test_uncommitted_txs(self):
        wal = WAL()
        wal.append(1, WALRecordType.BEGIN)
        wal.append(2, WALRecordType.BEGIN)
        wal.append(1, WALRecordType.COMMIT)
        assert wal.get_uncommitted_txs() == {2}

    def test_truncate_before(self):
        wal = WAL()
        wal.append(1, WALRecordType.BEGIN)
        wal.append(1, WALRecordType.INSERT, table_name="t", row_id=1)
        wal.append(1, WALRecordType.COMMIT)
        wal.truncate_before(2)
        assert len(wal.records) == 2

    def test_checkpoint_record(self):
        wal = WAL()
        rec = wal.checkpoint_record()
        assert rec.record_type == WALRecordType.CHECKPOINT
        assert rec.tx_id == 0

    def test_wal_record_fields(self):
        wal = WAL()
        rec = wal.append(5, WALRecordType.INSERT, table_name="users",
                         row_id=42, after_image={"name": "Alice"})
        assert rec.tx_id == 5
        assert rec.table_name == "users"
        assert rec.row_id == 42
        assert rec.after_image == {"name": "Alice"}
        assert rec.timestamp > 0

    def test_empty_wal(self):
        wal = WAL()
        assert wal.last_lsn == 0
        assert wal.get_uncommitted_txs() == set()
        assert wal.get_tx_records(1) == []


# ============================================================
# Snapshot Tests
# ============================================================

class TestSnapshot:
    def test_own_tx_visible(self):
        snap = Snapshot(tx_id=5, active_tx_ids={3}, min_tx_id=3, max_tx_id=6)
        assert snap.is_visible(5) is True  # Own tx

    def test_committed_visible(self):
        snap = Snapshot(tx_id=5, active_tx_ids={3}, min_tx_id=3, max_tx_id=6)
        assert snap.is_visible(1) is True
        assert snap.is_visible(2) is True
        assert snap.is_visible(4) is True

    def test_active_not_visible(self):
        snap = Snapshot(tx_id=5, active_tx_ids={3}, min_tx_id=3, max_tx_id=6)
        assert snap.is_visible(3) is False

    def test_future_not_visible(self):
        snap = Snapshot(tx_id=5, active_tx_ids={3}, min_tx_id=3, max_tx_id=6)
        assert snap.is_visible(6) is False
        assert snap.is_visible(100) is False


# ============================================================
# MVCCTable Tests
# ============================================================

class TestMVCCTable:
    def test_insert_and_scan(self):
        t = MVCCTable("t", ["id", "name"])
        rid = t.insert({"id": 1, "name": "Alice"}, tx_id=1)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        rows = t.scan(snap)
        assert len(rows) == 1
        assert rows[0][1] == {"id": 1, "name": "Alice"}

    def test_insert_invisible_to_other_active_tx(self):
        t = MVCCTable("t", ["id"])
        t.insert({"id": 1}, tx_id=2)
        snap = Snapshot(tx_id=1, active_tx_ids={2}, min_tx_id=1, max_tx_id=3)
        rows = t.scan(snap)
        assert len(rows) == 0

    def test_insert_visible_after_commit(self):
        t = MVCCTable("t", ["id"])
        t.insert({"id": 1}, tx_id=2)
        # Snapshot where tx 2 is NOT active (committed)
        snap = Snapshot(tx_id=3, active_tx_ids=set(), min_tx_id=3, max_tx_id=4)
        rows = t.scan(snap)
        assert len(rows) == 1

    def test_delete(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        t.delete(rid, tx_id=1)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        rows = t.scan(snap)
        assert len(rows) == 0

    def test_delete_invisible_to_snapshot_before_delete(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        # Delete by tx 3 (which is active in our snapshot)
        t.delete(rid, tx_id=3)
        snap = Snapshot(tx_id=2, active_tx_ids={3}, min_tx_id=2, max_tx_id=4)
        rows = t.scan(snap)
        assert len(rows) == 1  # Delete not visible

    def test_update(self):
        t = MVCCTable("t", ["id", "name"])
        rid = t.insert({"id": 1, "name": "Alice"}, tx_id=1)
        old = t.update(rid, {"id": 1, "name": "Bob"}, tx_id=1)
        assert old == {"id": 1, "name": "Alice"}
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        data = t.get(rid, snap)
        assert data == {"id": 1, "name": "Bob"}

    def test_update_old_version_visible_to_older_snapshot(self):
        t = MVCCTable("t", ["id", "name"])
        rid = t.insert({"id": 1, "name": "Alice"}, tx_id=1)
        t.update(rid, {"id": 1, "name": "Bob"}, tx_id=3)
        # Snapshot where tx 3 is active (not committed yet)
        snap = Snapshot(tx_id=2, active_tx_ids={3}, min_tx_id=2, max_tx_id=4)
        data = t.get(rid, snap)
        assert data == {"id": 1, "name": "Alice"}

    def test_get_nonexistent(self):
        t = MVCCTable("t", ["id"])
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        assert t.get(999, snap) is None

    def test_undo_insert(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        t.undo_insert(rid, 1)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        assert t.scan(snap) == []

    def test_undo_delete(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        t.delete(rid, tx_id=1)
        t.undo_delete(rid, 1)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        assert len(t.scan(snap)) == 1

    def test_undo_update(self):
        t = MVCCTable("t", ["id", "name"])
        rid = t.insert({"id": 1, "name": "Alice"}, tx_id=1)
        t.update(rid, {"id": 1, "name": "Bob"}, tx_id=2)
        t.undo_update(rid, 2)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=3)
        data = t.get(rid, snap)
        assert data["name"] == "Alice"

    def test_multiple_inserts(self):
        t = MVCCTable("t", ["id"])
        t.insert({"id": 1}, tx_id=1)
        t.insert({"id": 2}, tx_id=1)
        t.insert({"id": 3}, tx_id=1)
        snap = Snapshot(tx_id=1, active_tx_ids=set(), min_tx_id=1, max_tx_id=2)
        rows = t.scan(snap)
        assert len(rows) == 3

    def test_row_ids_sequential(self):
        t = MVCCTable("t", ["id"])
        r1 = t.insert({"id": 1}, tx_id=1)
        r2 = t.insert({"id": 2}, tx_id=1)
        r3 = t.insert({"id": 3}, tx_id=1)
        assert r1 == 1
        assert r2 == 2
        assert r3 == 3

    def test_vacuum(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        t.update(rid, {"id": 2}, tx_id=2)
        # Vacuum with oldest_active = 3 (all txs < 3 are done)
        t.vacuum(3)
        # The old version (created_by=1, deleted_by=2) should be gone
        snap = Snapshot(tx_id=3, active_tx_ids=set(), min_tx_id=3, max_tx_id=4)
        data = t.get(rid, snap)
        assert data == {"id": 2}

    def test_delete_already_deleted(self):
        t = MVCCTable("t", ["id"])
        rid = t.insert({"id": 1}, tx_id=1)
        t.delete(rid, 1)
        assert t.delete(rid, 1) is None

    def test_update_nonexistent(self):
        t = MVCCTable("t", ["id"])
        assert t.update(999, {"id": 1}, tx_id=1) is None


# ============================================================
# Lock Manager Tests
# ============================================================

class TestLockManager:
    def test_acquire_exclusive(self):
        lm = LockManager()
        assert lm.acquire(1, "t", 1, LockMode.EXCLUSIVE) is True

    def test_acquire_shared(self):
        lm = LockManager()
        assert lm.acquire(1, "t", 1, LockMode.SHARED) is True

    def test_multiple_shared(self):
        lm = LockManager()
        assert lm.acquire(1, "t", 1, LockMode.SHARED) is True
        assert lm.acquire(2, "t", 1, LockMode.SHARED) is True

    def test_shared_blocks_exclusive(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.SHARED)
        with pytest.raises(LockConflictError):
            lm.acquire(2, "t", 1, LockMode.EXCLUSIVE)

    def test_exclusive_blocks_shared(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        with pytest.raises(LockConflictError):
            lm.acquire(2, "t", 1, LockMode.SHARED)

    def test_exclusive_blocks_exclusive(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        with pytest.raises(LockConflictError):
            lm.acquire(2, "t", 1, LockMode.EXCLUSIVE)

    def test_reentrant_same_mode(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        assert lm.acquire(1, "t", 1, LockMode.EXCLUSIVE) is True

    def test_reentrant_shared_on_exclusive(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        assert lm.acquire(1, "t", 1, LockMode.SHARED) is True

    def test_upgrade_shared_to_exclusive(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.SHARED)
        assert lm.acquire(1, "t", 1, LockMode.EXCLUSIVE) is True

    def test_upgrade_blocked_by_other_shared(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.SHARED)
        lm.acquire(2, "t", 1, LockMode.SHARED)
        with pytest.raises(LockConflictError):
            lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)

    def test_release_all(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        lm.acquire(1, "t", 2, LockMode.EXCLUSIVE)
        lm.release_all(1)
        # Now tx 2 can acquire
        assert lm.acquire(2, "t", 1, LockMode.EXCLUSIVE) is True

    def test_table_level_lock(self):
        lm = LockManager()
        assert lm.acquire(1, "t", mode=LockMode.EXCLUSIVE) is True
        with pytest.raises(LockConflictError):
            lm.acquire(2, "t", mode=LockMode.EXCLUSIVE)

    def test_get_tx_locks(self):
        lm = LockManager()
        lm.acquire(1, "t", 1, LockMode.EXCLUSIVE)
        lm.acquire(1, "t", 2, LockMode.SHARED)
        locks = lm.get_tx_locks(1)
        assert len(locks) == 2


# ============================================================
# Transaction Manager Core Tests
# ============================================================

class TestTransactionManager:
    def test_create_table(self):
        tm = TransactionManager()
        t = tm.create_table("users", ["id", "name"])
        assert t.name == "users"
        assert "users" in tm.tables

    def test_create_duplicate_table(self):
        tm = TransactionManager()
        tm.create_table("users", ["id"])
        with pytest.raises(TransactionError):
            tm.create_table("users", ["id"])

    def test_drop_table(self):
        tm = TransactionManager()
        tm.create_table("users", ["id"])
        tm.drop_table("users")
        assert "users" not in tm.tables

    def test_drop_nonexistent(self):
        tm = TransactionManager()
        with pytest.raises(TransactionError):
            tm.drop_table("nope")

    def test_begin_tx(self):
        tm = TransactionManager()
        tx = tm.begin()
        assert tx.tx_id == 1
        assert tx.state == TxState.ACTIVE
        assert tx.isolation == IsolationLevel.REPEATABLE_READ

    def test_begin_with_isolation(self):
        tm = TransactionManager()
        tx = tm.begin(IsolationLevel.SERIALIZABLE)
        assert tx.isolation == IsolationLevel.SERIALIZABLE

    def test_sequential_tx_ids(self):
        tm = TransactionManager()
        tx1 = tm.begin()
        tx2 = tm.begin()
        assert tx1.tx_id == 1
        assert tx2.tx_id == 2

    def test_active_transactions(self):
        tm = TransactionManager()
        tx1 = tm.begin()
        tx2 = tm.begin()
        assert set(tm.get_active_transactions()) == {1, 2}

    def test_get_transaction(self):
        tm = TransactionManager()
        tx = tm.begin()
        assert tm.get_transaction(1) is tx
        assert tm.get_transaction(99) is None


# ============================================================
# Transaction CRUD Tests
# ============================================================

class TestTransactionCRUD:
    def test_insert_and_read(self):
        tm = TransactionManager()
        tm.create_table("users", ["id", "name"])
        tx = tm.begin()
        rid = tx.insert("users", {"id": 1, "name": "Alice"})
        data = tx.read("users", rid)
        assert data == {"id": 1, "name": "Alice"}

    def test_insert_scan(self):
        tm = TransactionManager()
        tm.create_table("users", ["id", "name"])
        tx = tm.begin()
        tx.insert("users", {"id": 1, "name": "Alice"})
        tx.insert("users", {"id": 2, "name": "Bob"})
        rows = tx.read("users")
        assert len(rows) == 2

    def test_update(self):
        tm = TransactionManager()
        tm.create_table("users", ["id", "name"])
        tx = tm.begin()
        rid = tx.insert("users", {"id": 1, "name": "Alice"})
        old = tx.update("users", rid, {"id": 1, "name": "Bob"})
        assert old == {"id": 1, "name": "Alice"}
        data = tx.read("users", rid)
        assert data == {"id": 1, "name": "Bob"}

    def test_delete(self):
        tm = TransactionManager()
        tm.create_table("users", ["id", "name"])
        tx = tm.begin()
        rid = tx.insert("users", {"id": 1, "name": "Alice"})
        old = tx.delete("users", rid)
        assert old == {"id": 1, "name": "Alice"}
        data = tx.read("users", rid)
        assert data is None

    def test_read_nonexistent_table(self):
        tm = TransactionManager()
        tx = tm.begin()
        with pytest.raises(TransactionError):
            tx.read("nope")

    def test_insert_nonexistent_table(self):
        tm = TransactionManager()
        tx = tm.begin()
        with pytest.raises(TransactionError):
            tx.insert("nope", {"id": 1})

    def test_update_nonexistent_row(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        assert tx.update("t", 999, {"id": 1}) is None

    def test_delete_nonexistent_row(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        assert tx.delete("t", 999) is None

    def test_multiple_operations(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx = tm.begin()
        r1 = tx.insert("t", {"id": 1, "val": "a"})
        r2 = tx.insert("t", {"id": 2, "val": "b"})
        r3 = tx.insert("t", {"id": 3, "val": "c"})
        tx.update("t", r2, {"id": 2, "val": "B"})
        tx.delete("t", r3)
        rows = tx.read("t")
        assert len(rows) == 2
        vals = sorted([r[1]["val"] for r in rows])
        assert vals == ["B", "a"]


# ============================================================
# Commit and Rollback Tests
# ============================================================

class TestCommitRollback:
    def test_commit_makes_visible(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.commit()

        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 1

    def test_rollback_undoes_insert(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.rollback()

        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 0

    def test_rollback_undoes_update(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "name"])
        tx1 = tm.begin()
        rid = tx1.insert("t", {"id": 1, "name": "Alice"})
        tx1.commit()

        tx2 = tm.begin()
        tx2.update("t", rid, {"id": 1, "name": "Bob"})
        tx2.rollback()

        tx3 = tm.begin()
        data = tx3.read("t", rid)
        assert data["name"] == "Alice"

    def test_rollback_undoes_delete(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        rid = tx1.insert("t", {"id": 1})
        tx1.commit()

        tx2 = tm.begin()
        tx2.delete("t", rid)
        tx2.rollback()

        tx3 = tm.begin()
        data = tx3.read("t", rid)
        assert data is not None

    def test_commit_state(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.commit()
        assert tx.state == TxState.COMMITTED

    def test_rollback_state(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.rollback()
        assert tx.state == TxState.ABORTED

    def test_operations_after_commit_fail(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.commit()
        with pytest.raises(TransactionAbortedError):
            tx.insert("t", {"id": 1})

    def test_operations_after_rollback_fail(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.rollback()
        with pytest.raises(TransactionAbortedError):
            tx.insert("t", {"id": 1})

    def test_double_rollback_safe(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.rollback()
        tx.rollback()  # Should be safe (no-op)

    def test_commit_releases_locks(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.commit()
        assert 1 not in tm.lock_manager._tx_locks

    def test_rollback_releases_locks(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.rollback()
        assert 1 not in tm.lock_manager._tx_locks

    def test_rollback_undoes_create_table(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.create_table("t", ["id"])
        assert "t" in tm.tables
        tx.rollback()
        assert "t" not in tm.tables

    def test_rollback_undoes_drop_table(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.drop_table("t")
        assert "t" not in tm.tables
        tx.rollback()
        assert "t" in tm.tables


# ============================================================
# Savepoint Tests
# ============================================================

class TestSavepoints:
    def test_basic_savepoint(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.insert("t", {"id": 2})
        tx.rollback_to_savepoint("sp1")
        rows = tx.read("t")
        assert len(rows) == 1
        assert rows[0][1]["id"] == 1

    def test_multiple_savepoints(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.insert("t", {"id": 2})
        tx.savepoint("sp2")
        tx.insert("t", {"id": 3})
        tx.rollback_to_savepoint("sp2")
        rows = tx.read("t")
        assert len(rows) == 2

    def test_rollback_to_first_savepoint(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.insert("t", {"id": 2})
        tx.savepoint("sp2")
        tx.insert("t", {"id": 3})
        tx.rollback_to_savepoint("sp1")
        rows = tx.read("t")
        assert len(rows) == 1

    def test_savepoint_update_rollback(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "name"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1, "name": "Alice"})
        tx.savepoint("sp1")
        tx.update("t", rid, {"id": 1, "name": "Bob"})
        tx.rollback_to_savepoint("sp1")
        data = tx.read("t", rid)
        assert data["name"] == "Alice"

    def test_savepoint_delete_rollback(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.delete("t", rid)
        tx.rollback_to_savepoint("sp1")
        data = tx.read("t", rid)
        assert data is not None

    def test_nonexistent_savepoint(self):
        tm = TransactionManager()
        tx = tm.begin()
        with pytest.raises(SavepointError):
            tx.rollback_to_savepoint("nope")

    def test_release_savepoint(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.savepoint("sp1")
        tx.release_savepoint("sp1")
        with pytest.raises(SavepointError):
            tx.rollback_to_savepoint("sp1")

    def test_release_nonexistent_savepoint(self):
        tm = TransactionManager()
        tx = tm.begin()
        with pytest.raises(SavepointError):
            tx.release_savepoint("nope")

    def test_rollback_removes_later_savepoints(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.savepoint("sp1")
        tx.insert("t", {"id": 1})
        tx.savepoint("sp2")
        tx.insert("t", {"id": 2})
        tx.rollback_to_savepoint("sp1")
        with pytest.raises(SavepointError):
            tx.rollback_to_savepoint("sp2")

    def test_savepoint_then_commit(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.insert("t", {"id": 2})
        tx.rollback_to_savepoint("sp1")
        tx.commit()
        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 1
        assert rows[0][1]["id"] == 1

    def test_savepoint_reuse_name(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.savepoint("sp1")
        tx.insert("t", {"id": 2})
        tx.savepoint("sp1")  # Overwrite
        tx.insert("t", {"id": 3})
        tx.rollback_to_savepoint("sp1")
        rows = tx.read("t")
        assert len(rows) == 2  # id=1 and id=2


# ============================================================
# Isolation Level Tests
# ============================================================

class TestIsolation:
    def test_read_uncommitted_sees_dirty(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin(IsolationLevel.REPEATABLE_READ)
        tx1.insert("t", {"id": 1})
        # tx1 not committed yet

        tx2 = tm.begin(IsolationLevel.READ_UNCOMMITTED)
        rows = tx2.read("t")
        assert len(rows) == 1  # Dirty read

    def test_read_committed_no_dirty_read(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        # tx1 not committed

        tx2 = tm.begin(IsolationLevel.READ_COMMITTED)
        rows = tx2.read("t")
        assert len(rows) == 0  # No dirty read

    def test_read_committed_sees_after_commit(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})

        tx2 = tm.begin(IsolationLevel.READ_COMMITTED)
        rows = tx2.read("t")
        assert len(rows) == 0

        tx1.commit()
        rows = tx2.read("t")
        assert len(rows) == 1  # Now visible

    def test_repeatable_read_consistent(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.commit()

        tx2 = tm.begin(IsolationLevel.REPEATABLE_READ)
        rows1 = tx2.read("t")
        assert len(rows1) == 1

        # Another tx inserts
        tx3 = tm.begin()
        tx3.insert("t", {"id": 2})
        tx3.commit()

        # tx2 still sees the same snapshot
        rows2 = tx2.read("t")
        assert len(rows2) == 1  # No phantom

    def test_repeatable_read_no_phantom(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])

        tx1 = tm.begin(IsolationLevel.REPEATABLE_READ)
        rows = tx1.read("t")
        assert len(rows) == 0

        tx2 = tm.begin()
        tx2.insert("t", {"id": 1})
        tx2.commit()

        rows = tx1.read("t")
        assert len(rows) == 0  # Still empty -- snapshot isolation


# ============================================================
# Serializable Isolation Tests
# ============================================================

class TestSerializable:
    def test_write_write_conflict(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx0 = tm.begin()
        rid = tx0.insert("t", {"id": 1, "val": "a"})
        tx0.commit()

        tx1 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx2 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx1.update("t", rid, {"id": 1, "val": "b"})
        with pytest.raises((SerializationError, LockConflictError)):
            tx2.update("t", rid, {"id": 1, "val": "c"})

    def test_serializable_commit_after_conflict(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx0 = tm.begin()
        rid = tx0.insert("t", {"id": 1, "val": "a"})
        tx0.commit()

        tx1 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx1.update("t", rid, {"id": 1, "val": "b"})
        tx1.commit()

        tx2 = tm.begin(IsolationLevel.SERIALIZABLE)
        data = tx2.read("t", rid)
        assert data["val"] == "b"

    def test_serialization_failure_on_commit(self):
        """If tx reads data that another concurrent tx modified and committed,
        serialization failure should occur on commit."""
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx0 = tm.begin()
        rid = tx0.insert("t", {"id": 1, "val": "a"})
        tx0.commit()

        tx1 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx1.read("t", rid)  # Read row

        tx2 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx2.update("t", rid, {"id": 1, "val": "b"})
        tx2.commit()

        with pytest.raises(SerializationError):
            tx1.commit()  # Should fail -- read set was modified


# ============================================================
# DDL in Transaction Tests
# ============================================================

class TestDDLInTransaction:
    def test_create_table_in_tx(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.create_table("t", ["id"])
        assert "t" in tm.tables
        tx.commit()
        assert "t" in tm.tables

    def test_create_table_rollback(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.create_table("t", ["id"])
        tx.rollback()
        assert "t" not in tm.tables

    def test_drop_table_in_tx(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.drop_table("t")
        tx.commit()
        assert "t" not in tm.tables

    def test_drop_table_rollback(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.drop_table("t")
        tx.rollback()
        assert "t" in tm.tables

    def test_create_duplicate_in_tx(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        with pytest.raises(TransactionError):
            tx.create_table("t", ["id"])

    def test_drop_nonexistent_in_tx(self):
        tm = TransactionManager()
        tx = tm.begin()
        with pytest.raises(TransactionError):
            tx.drop_table("nope")


# ============================================================
# WAL Recovery Tests
# ============================================================

class TestRecovery:
    def test_recover_uncommitted(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1})
        # Simulate crash: tx never committed
        # WAL has BEGIN + INSERT but no COMMIT

        tm.recover()
        # The uncommitted insert should be undone
        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 0

    def test_committed_survives_recovery(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.commit()

        tm.recover()  # Nothing to undo
        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 1


# ============================================================
# Checkpoint Tests
# ============================================================

class TestCheckpoint:
    def test_checkpoint_returns_lsn(self):
        tm = TransactionManager()
        lsn = tm.checkpoint()
        assert lsn > 0

    def test_checkpoint_after_operations(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.commit()
        lsn = tm.checkpoint()
        assert lsn > 2  # BEGIN, INSERT, COMMIT all before checkpoint


# ============================================================
# Vacuum Tests
# ============================================================

class TestVacuum:
    def test_vacuum_cleans_old_versions(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        rid = tx1.insert("t", {"id": 1})
        tx1.commit()

        tx2 = tm.begin()
        tx2.update("t", rid, {"id": 2})
        tx2.commit()

        tm.vacuum()

        tx3 = tm.begin()
        data = tx3.read("t", rid)
        assert data["id"] == 2

    def test_vacuum_preserves_active_tx_data(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        rid = tx1.insert("t", {"id": 1})
        tx1.commit()

        tx2 = tm.begin()  # Still active, holds snapshot
        tx3 = tm.begin()
        tx3.update("t", rid, {"id": 2})
        tx3.commit()

        tm.vacuum()  # Should keep old version since tx2 is active

        data = tx2.read("t", rid)
        assert data["id"] == 1  # tx2 should still see old value


# ============================================================
# TransactionalDatabase Tests
# ============================================================

class TestTransactionalDatabase:
    def test_basic_workflow(self):
        db = TransactionalDatabase()
        db.create_table("users", ["id", "name"])
        tx = db.begin()
        tx.insert("users", {"id": 1, "name": "Alice"})
        tx.commit()

        tx2 = db.begin()
        rows = tx2.read("users")
        assert len(rows) == 1

    def test_execute_in_transaction(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])

        def fn(tx):
            tx.insert("t", {"id": 1})
            tx.insert("t", {"id": 2})
            return "ok"

        result = db.execute_in_transaction(fn)
        assert result == "ok"

        tx = db.begin()
        rows = tx.read("t")
        assert len(rows) == 2

    def test_execute_in_transaction_rollback_on_error(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])

        def fn(tx):
            tx.insert("t", {"id": 1})
            raise ValueError("oops")

        with pytest.raises(ValueError):
            db.execute_in_transaction(fn)

        tx = db.begin()
        rows = tx.read("t")
        assert len(rows) == 0  # Rolled back

    def test_checkpoint_and_vacuum(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])
        tx = db.begin()
        tx.insert("t", {"id": 1})
        tx.commit()
        lsn = db.checkpoint()
        assert lsn > 0
        db.vacuum()  # Should not error

    def test_drop_table(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])
        db.drop_table("t")
        assert "t" not in db.tm.tables

    def test_wal_property(self):
        db = TransactionalDatabase()
        assert isinstance(db.wal, WAL)

    def test_lock_manager_property(self):
        db = TransactionalDatabase()
        assert isinstance(db.lock_manager, LockManager)

    def test_recover(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])
        tx = db.begin()
        tx.insert("t", {"id": 1})
        # Don't commit
        db.recover()
        tx2 = db.begin()
        rows = tx2.read("t")
        assert len(rows) == 0


# ============================================================
# Concurrent Transaction Scenarios
# ============================================================

class TestConcurrentScenarios:
    def test_two_tx_insert_different_rows(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx2 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx2.insert("t", {"id": 2})
        tx1.commit()
        tx2.commit()

        tx3 = tm.begin()
        rows = tx3.read("t")
        assert len(rows) == 2

    def test_read_own_inserts(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx1 = tm.begin()
        tx1.insert("t", {"id": 1})
        tx1.insert("t", {"id": 2})
        rows = tx1.read("t")
        assert len(rows) == 2

    def test_tx_cant_see_concurrent_inserts(self):
        """With REPEATABLE_READ, a tx can't see inserts from concurrent txs."""
        tm = TransactionManager()
        tm.create_table("t", ["id"])

        tx1 = tm.begin(IsolationLevel.REPEATABLE_READ)
        tx1.read("t")  # Take snapshot

        tx2 = tm.begin()
        tx2.insert("t", {"id": 1})
        tx2.commit()

        rows = tx1.read("t")
        assert len(rows) == 0

    def test_lost_update_prevention(self):
        """Two txs reading and writing same row -- one should fail."""
        tm = TransactionManager()
        tm.create_table("t", ["id", "balance"])
        tx0 = tm.begin()
        rid = tx0.insert("t", {"id": 1, "balance": 100})
        tx0.commit()

        tx1 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx2 = tm.begin(IsolationLevel.SERIALIZABLE)
        tx1.read("t", rid)
        tx2.read("t", rid)
        tx1.update("t", rid, {"id": 1, "balance": 150})
        tx1.commit()

        # tx2 should fail due to serialization conflict
        with pytest.raises((SerializationError, LockConflictError)):
            tx2.update("t", rid, {"id": 1, "balance": 200})

    def test_insert_delete_same_tx(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1})
        tx.delete("t", rid)
        rows = tx.read("t")
        assert len(rows) == 0
        tx.commit()

        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 0

    def test_update_then_delete(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1, "val": "a"})
        tx.update("t", rid, {"id": 1, "val": "b"})
        tx.delete("t", rid)
        rows = tx.read("t")
        assert len(rows) == 0

    def test_many_transactions_sequential(self):
        tm = TransactionManager()
        tm.create_table("counter", ["val"])
        tx0 = tm.begin()
        rid = tx0.insert("counter", {"val": 0})
        tx0.commit()

        for i in range(1, 11):
            tx = tm.begin()
            data = tx.read("counter", rid)
            tx.update("counter", rid, {"val": data["val"] + 1})
            tx.commit()

        tx_final = tm.begin()
        data = tx_final.read("counter", rid)
        assert data["val"] == 10


# ============================================================
# WAL Record Types Completeness
# ============================================================

class TestWALRecordTypes:
    def test_begin_record(self):
        tm = TransactionManager()
        tx = tm.begin()
        recs = tm.wal.get_tx_records(tx.tx_id)
        assert recs[0].record_type == WALRecordType.BEGIN

    def test_insert_record(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        recs = tm.wal.get_tx_records(tx.tx_id)
        insert_recs = [r for r in recs if r.record_type == WALRecordType.INSERT]
        assert len(insert_recs) == 1
        assert insert_recs[0].after_image == {"id": 1}

    def test_update_record(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "name"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1, "name": "a"})
        tx.update("t", rid, {"id": 1, "name": "b"})
        recs = tm.wal.get_tx_records(tx.tx_id)
        upd_recs = [r for r in recs if r.record_type == WALRecordType.UPDATE]
        assert len(upd_recs) == 1
        assert upd_recs[0].before_image == {"id": 1, "name": "a"}
        assert upd_recs[0].after_image == {"id": 1, "name": "b"}

    def test_delete_record(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1})
        tx.delete("t", rid)
        recs = tm.wal.get_tx_records(tx.tx_id)
        del_recs = [r for r in recs if r.record_type == WALRecordType.DELETE]
        assert len(del_recs) == 1
        assert del_recs[0].before_image == {"id": 1}

    def test_commit_record(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.commit()
        recs = tm.wal.get_tx_records(tx.tx_id)
        assert recs[-1].record_type == WALRecordType.COMMIT

    def test_abort_record(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.rollback()
        recs = tm.wal.get_tx_records(tx.tx_id)
        assert recs[-1].record_type == WALRecordType.ABORT

    def test_savepoint_record(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.savepoint("sp1")
        recs = tm.wal.get_tx_records(tx.tx_id)
        sp_recs = [r for r in recs if r.record_type == WALRecordType.SAVEPOINT]
        assert len(sp_recs) == 1
        assert sp_recs[0].savepoint_name == "sp1"

    def test_create_table_record(self):
        tm = TransactionManager()
        tx = tm.begin()
        tx.create_table("t", ["id", "name"])
        recs = tm.wal.get_tx_records(tx.tx_id)
        ct_recs = [r for r in recs if r.record_type == WALRecordType.CREATE_TABLE]
        assert len(ct_recs) == 1
        assert ct_recs[0].table_columns == ["id", "name"]

    def test_drop_table_record(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.drop_table("t")
        recs = tm.wal.get_tx_records(tx.tx_id)
        dt_recs = [r for r in recs if r.record_type == WALRecordType.DROP_TABLE]
        assert len(dt_recs) == 1


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_table_scan(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        rows = tx.read("t")
        assert rows == []

    def test_large_transaction(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        for i in range(100):
            tx.insert("t", {"id": i})
        tx.commit()
        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 100

    def test_rollback_large_transaction(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        for i in range(100):
            tx.insert("t", {"id": i})
        tx.rollback()
        tx2 = tm.begin()
        rows = tx2.read("t")
        assert len(rows) == 0

    def test_write_set_tracking(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx = tm.begin()
        tx.insert("t", {"id": 1})
        tx.insert("t", {"id": 2})
        assert len(tx.write_set) == 2

    def test_read_set_tracking(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx0 = tm.begin()
        rid = tx0.insert("t", {"id": 1})
        tx0.commit()

        tx = tm.begin()
        tx.read("t", rid)
        assert ("t", rid) in tx.read_set

    def test_read_scan_tracks_all_rows(self):
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx0 = tm.begin()
        tx0.insert("t", {"id": 1})
        tx0.insert("t", {"id": 2})
        tx0.commit()

        tx = tm.begin()
        tx.read("t")  # Scan
        assert len(tx.read_set) == 2

    def test_multiple_updates_same_row(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx = tm.begin()
        rid = tx.insert("t", {"id": 1, "val": "a"})
        tx.update("t", rid, {"id": 1, "val": "b"})
        tx.update("t", rid, {"id": 1, "val": "c"})
        data = tx.read("t", rid)
        assert data["val"] == "c"
        tx.commit()

        tx2 = tm.begin()
        data = tx2.read("t", rid)
        assert data["val"] == "c"

    def test_tx_isolation_between_tables(self):
        tm = TransactionManager()
        tm.create_table("t1", ["id"])
        tm.create_table("t2", ["id"])
        tx = tm.begin()
        tx.insert("t1", {"id": 1})
        tx.insert("t2", {"id": 2})
        rows1 = tx.read("t1")
        rows2 = tx.read("t2")
        assert len(rows1) == 1
        assert len(rows2) == 1
        assert rows1[0][1]["id"] == 1
        assert rows2[0][1]["id"] == 2

    def test_savepoint_with_mixed_ops(self):
        tm = TransactionManager()
        tm.create_table("t", ["id", "val"])
        tx = tm.begin()
        r1 = tx.insert("t", {"id": 1, "val": "a"})
        r2 = tx.insert("t", {"id": 2, "val": "b"})
        tx.savepoint("sp1")
        tx.update("t", r1, {"id": 1, "val": "A"})
        tx.delete("t", r2)
        r3 = tx.insert("t", {"id": 3, "val": "c"})
        tx.rollback_to_savepoint("sp1")
        rows = tx.read("t")
        assert len(rows) == 2
        vals = sorted([r[1]["val"] for r in rows])
        assert vals == ["a", "b"]

    def test_read_committed_fresh_snapshot_each_read(self):
        """READ_COMMITTED should see newly committed data on each read."""
        tm = TransactionManager()
        tm.create_table("t", ["id"])
        tx0 = tm.begin()
        tx0.insert("t", {"id": 1})
        tx0.commit()

        tx_reader = tm.begin(IsolationLevel.READ_COMMITTED)
        rows = tx_reader.read("t")
        assert len(rows) == 1

        tx_writer = tm.begin()
        tx_writer.insert("t", {"id": 2})
        tx_writer.commit()

        rows = tx_reader.read("t")
        assert len(rows) == 2  # Sees new data


# ============================================================
# Stress / Integration Tests
# ============================================================

class TestIntegration:
    def test_full_lifecycle(self):
        db = TransactionalDatabase()
        db.create_table("accounts", ["id", "balance"])

        # Insert initial data
        tx = db.begin()
        r1 = tx.insert("accounts", {"id": 1, "balance": 1000})
        r2 = tx.insert("accounts", {"id": 2, "balance": 500})
        tx.commit()

        # Transfer
        tx = db.begin()
        acct1 = tx.read("accounts", r1)
        acct2 = tx.read("accounts", r2)
        tx.update("accounts", r1, {"id": 1, "balance": acct1["balance"] - 200})
        tx.update("accounts", r2, {"id": 2, "balance": acct2["balance"] + 200})
        tx.commit()

        # Verify
        tx = db.begin()
        assert tx.read("accounts", r1)["balance"] == 800
        assert tx.read("accounts", r2)["balance"] == 700

    def test_execute_in_transaction_with_serializable(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id", "val"])

        def fn(tx):
            tx.insert("t", {"id": 1, "val": "hello"})
            return tx.read("t")

        rows = db.execute_in_transaction(fn, IsolationLevel.SERIALIZABLE)
        assert len(rows) == 1

    def test_many_small_transactions(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])
        for i in range(50):
            tx = db.begin()
            tx.insert("t", {"id": i})
            tx.commit()

        tx = db.begin()
        rows = tx.read("t")
        assert len(rows) == 50

    def test_interleaved_reads_and_writes(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id", "val"])

        tx1 = db.begin()
        r1 = tx1.insert("t", {"id": 1, "val": "a"})
        tx1.commit()

        tx2 = db.begin(IsolationLevel.REPEATABLE_READ)
        tx2.read("t")  # Take snapshot

        tx3 = db.begin()
        tx3.insert("t", {"id": 2, "val": "b"})
        tx3.commit()

        # tx2 should not see id=2
        rows = tx2.read("t")
        assert len(rows) == 1

        tx2.commit()

        # New tx sees both
        tx4 = db.begin()
        rows = tx4.read("t")
        assert len(rows) == 2

    def test_checkpoint_then_vacuum(self):
        db = TransactionalDatabase()
        db.create_table("t", ["id"])
        tx = db.begin()
        rid = tx.insert("t", {"id": 1})
        tx.commit()

        tx2 = db.begin()
        tx2.update("t", rid, {"id": 2})
        tx2.commit()

        db.checkpoint()
        db.vacuum()

        tx3 = db.begin()
        data = tx3.read("t", rid)
        assert data["id"] == 2

    def test_savepoint_nested_scenario(self):
        db = TransactionalDatabase()
        db.create_table("t", ["step"])
        tx = db.begin()
        tx.insert("t", {"step": "initial"})
        tx.savepoint("sp_a")

        tx.insert("t", {"step": "after_a"})
        tx.savepoint("sp_b")

        tx.insert("t", {"step": "after_b"})

        # Rollback to sp_b
        tx.rollback_to_savepoint("sp_b")
        rows = tx.read("t")
        assert len(rows) == 2

        # Continue after rollback
        tx.insert("t", {"step": "new_after_b"})

        # Rollback to sp_a
        tx.rollback_to_savepoint("sp_a")
        rows = tx.read("t")
        assert len(rows) == 1
        assert rows[0][1]["step"] == "initial"

        tx.commit()

        tx2 = db.begin()
        rows = tx2.read("t")
        assert len(rows) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
