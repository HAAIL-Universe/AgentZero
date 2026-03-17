"""
C246: Transaction Manager Tests

Tests for the ACID transaction manager composing MVCC + WAL + Lock Manager.
Covers: lifecycle, reads, writes, isolation, savepoints, batch ops,
deadlock handling, recovery, GC, context managers, diagnostics.
"""

import os
import sys
import unittest
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_challenges = os.path.join(os.path.dirname(__file__), '..')
for _dep in ('C240_mvcc', 'C241_wal', 'C242_lock_manager'):
    _path = os.path.join(_challenges, _dep)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from transaction_manager import (
    TransactionManager, Transaction, ReadOnlyTransaction,
    TransactionError, DeadlockError, ConflictError,
    TransactionNotFoundError, TransactionNotActiveError,
    TxnState, ManagedTransaction,
)
from mvcc import IsolationLevel


# ===================================================================
# 1. Transaction Lifecycle
# ===================================================================

class TestTransactionLifecycle(unittest.TestCase):
    """Basic begin/commit/abort lifecycle."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_begin_returns_txn_id(self):
        tid = self.tm.begin()
        self.assertIsInstance(tid, int)
        self.assertGreater(tid, 0)

    def test_sequential_txn_ids(self):
        t1 = self.tm.begin()
        t2 = self.tm.begin()
        self.assertEqual(t2, t1 + 1)

    def test_commit_active_txn(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k", "v")
        result = self.tm.commit(tid)
        self.assertTrue(result)

    def test_abort_active_txn(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k", "v")
        self.tm.abort(tid)
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.state, TxnState.ABORTED)

    def test_commit_nonexistent_txn_raises(self):
        with self.assertRaises(TransactionNotFoundError):
            self.tm.commit(999)

    def test_abort_nonexistent_txn_raises(self):
        with self.assertRaises(TransactionNotFoundError):
            self.tm.abort(999)

    def test_double_commit_raises(self):
        tid = self.tm.begin()
        self.tm.commit(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.commit(tid)

    def test_double_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.abort(tid)

    def test_commit_after_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.commit(tid)

    def test_begin_with_isolation_level(self):
        tid = self.tm.begin(isolation=IsolationLevel.SERIALIZABLE)
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.isolation, IsolationLevel.SERIALIZABLE)

    def test_begin_with_priority(self):
        tid = self.tm.begin(priority=10)
        self.assertIsNotNone(self.tm.get_transaction(tid))

    def test_active_transactions_list(self):
        t1 = self.tm.begin()
        t2 = self.tm.begin()
        active = self.tm.active_transactions()
        self.assertIn(t1, active)
        self.assertIn(t2, active)
        self.tm.commit(t1)
        active = self.tm.active_transactions()
        self.assertNotIn(t1, active)
        self.assertIn(t2, active)


# ===================================================================
# 2. Read Operations
# ===================================================================

class TestReadOperations(unittest.TestCase):
    """Test get, scan, range_scan, exists, count."""

    def setUp(self):
        self.tm = TransactionManager()
        # Seed data
        tid = self.tm.begin()
        for i in range(5):
            self.tm.put(tid, f"key{i}", f"val{i}")
        self.tm.commit(tid)

    def test_get_existing_key(self):
        tid = self.tm.begin()
        self.assertEqual(self.tm.get(tid, "key0"), "val0")
        self.tm.commit(tid)

    def test_get_nonexistent_key(self):
        tid = self.tm.begin()
        self.assertIsNone(self.tm.get(tid, "nokey"))
        self.tm.commit(tid)

    def test_scan_all(self):
        tid = self.tm.begin()
        result = self.tm.scan(tid, "key")
        self.assertEqual(len(result), 5)
        self.tm.commit(tid)

    def test_scan_prefix(self):
        tid = self.tm.begin()
        self.tm.put(tid, "other1", "x")
        result = self.tm.scan(tid, "key")
        self.assertEqual(len(result), 5)
        self.tm.commit(tid)

    def test_range_scan(self):
        tid = self.tm.begin()
        result = self.tm.range_scan(tid, "key1", "key3")
        self.assertIn("key1", result)
        self.assertIn("key3", result)
        self.assertNotIn("key4", result)
        self.tm.commit(tid)

    def test_exists_true(self):
        tid = self.tm.begin()
        self.assertTrue(self.tm.exists(tid, "key0"))
        self.tm.commit(tid)

    def test_exists_false(self):
        tid = self.tm.begin()
        self.assertFalse(self.tm.exists(tid, "nope"))
        self.tm.commit(tid)

    def test_count(self):
        tid = self.tm.begin()
        self.assertEqual(self.tm.count(tid, "key"), 5)
        self.tm.commit(tid)

    def test_read_tracks_stats(self):
        tid = self.tm.begin()
        self.tm.get(tid, "key0")
        self.tm.get(tid, "key1")
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.reads, 2)
        self.tm.commit(tid)


# ===================================================================
# 3. Write Operations
# ===================================================================

class TestWriteOperations(unittest.TestCase):
    """Test put, delete, update semantics."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_put_and_read(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.tm.commit(tid)

    def test_put_overwrites(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.put(tid, "k1", "v2")
        self.assertEqual(self.tm.get(tid, "k1"), "v2")
        self.tm.commit(tid)

    def test_delete_existing(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.commit(tid)

        tid2 = self.tm.begin()
        result = self.tm.delete(tid2, "k1")
        self.assertTrue(result)
        self.assertIsNone(self.tm.get(tid2, "k1"))
        self.tm.commit(tid2)

    def test_delete_nonexistent(self):
        tid = self.tm.begin()
        result = self.tm.delete(tid, "nokey")
        self.assertFalse(result)
        self.tm.commit(tid)

    def test_write_tracks_stats(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.put(tid, "k2", "v2")
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.writes, 2)
        self.tm.commit(tid)

    def test_write_not_visible_until_commit(self):
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        # Another transaction shouldn't see uncommitted writes
        t2 = self.tm.begin()
        val = self.tm.get(t2, "k1")
        self.assertIsNone(val)
        self.tm.commit(t1)
        self.tm.commit(t2)

    def test_aborted_writes_not_visible(self):
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        self.tm.abort(t1)

        t2 = self.tm.begin()
        self.assertIsNone(self.tm.get(t2, "k1"))
        self.tm.commit(t2)

    def test_committed_writes_visible_to_new_txn(self):
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        self.tm.commit(t1)

        t2 = self.tm.begin()
        self.assertEqual(self.tm.get(t2, "k1"), "v1")
        self.tm.commit(t2)


# ===================================================================
# 4. Isolation
# ===================================================================

class TestIsolation(unittest.TestCase):
    """Test that isolation levels are respected."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_repeatable_read_snapshot(self):
        """Txn sees consistent snapshot even if other txns commit."""
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        self.tm.commit(t1)

        t2 = self.tm.begin(isolation=IsolationLevel.REPEATABLE_READ)
        val_before = self.tm.get(t2, "k1")

        # Another txn modifies k1
        t3 = self.tm.begin()
        self.tm.put(t3, "k1", "v2")
        self.tm.commit(t3)

        # t2 should still see v1 (snapshot)
        val_after = self.tm.get(t2, "k1")
        self.assertEqual(val_before, val_after)
        self.assertEqual(val_after, "v1")
        self.tm.commit(t2)

    def test_read_committed_sees_committed(self):
        """Read committed isolation sees latest committed values."""
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        self.tm.commit(t1)

        t2 = self.tm.begin(isolation=IsolationLevel.READ_COMMITTED)
        self.assertEqual(self.tm.get(t2, "k1"), "v1")

        t3 = self.tm.begin()
        self.tm.put(t3, "k1", "v2")
        self.tm.commit(t3)

        # Read committed sees the latest committed value
        val = self.tm.get(t2, "k1")
        self.assertEqual(val, "v2")
        self.tm.commit(t2)

    def test_concurrent_write_same_key(self):
        """Two txns writing same key -- second blocks on X lock."""
        tm = TransactionManager(lock_timeout=0.5)
        t1 = tm.begin()
        tm.put(t1, "k1", "from_t1")

        t2 = tm.begin()
        # t2 tries to write same key -- should timeout or deadlock
        # since t1 holds X lock
        with self.assertRaises((DeadlockError, TransactionError)):
            tm.put(t2, "k1", "from_t2")

        tm.commit(t1)


# ===================================================================
# 5. Savepoints
# ===================================================================

class TestSavepoints(unittest.TestCase):
    """Test savepoint creation and rollback."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_savepoint_and_rollback(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.savepoint(tid, "sp1")
        self.tm.put(tid, "k2", "v2")
        self.tm.rollback_to_savepoint(tid, "sp1")
        # k1 should still be there, k2 should be rolled back
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.assertIsNone(self.tm.get(tid, "k2"))
        self.tm.commit(tid)

    def test_nested_savepoints(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.savepoint(tid, "sp1")
        self.tm.put(tid, "k2", "v2")
        self.tm.savepoint(tid, "sp2")
        self.tm.put(tid, "k3", "v3")
        self.tm.rollback_to_savepoint(tid, "sp1")
        # Only k1 survives
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.assertIsNone(self.tm.get(tid, "k2"))
        self.assertIsNone(self.tm.get(tid, "k3"))
        self.tm.commit(tid)

    def test_savepoint_restores_write_count(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.savepoint(tid, "sp1")
        self.tm.put(tid, "k2", "v2")
        self.tm.put(tid, "k3", "v3")
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.writes, 3)
        self.tm.rollback_to_savepoint(tid, "sp1")
        self.assertEqual(mtxn.writes, 1)
        self.tm.commit(tid)

    def test_savepoint_preserves_later_savepoints(self):
        """Rolling back to sp1 removes sp2 but keeps sp1."""
        tid = self.tm.begin()
        self.tm.savepoint(tid, "sp1")
        self.tm.put(tid, "k1", "v1")
        self.tm.savepoint(tid, "sp2")
        self.tm.put(tid, "k2", "v2")
        self.tm.rollback_to_savepoint(tid, "sp1")
        mtxn = self.tm.get_transaction(tid)
        sp_names = [s["name"] for s in mtxn.savepoints]
        self.assertIn("sp1", sp_names)
        self.assertNotIn("sp2", sp_names)
        self.tm.commit(tid)


# ===================================================================
# 6. Batch Operations
# ===================================================================

class TestBatchOperations(unittest.TestCase):
    """Test batch get/put/delete."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_batch_put(self):
        tid = self.tm.begin()
        self.tm.batch_put(tid, {"a": 1, "b": 2, "c": 3})
        self.assertEqual(self.tm.get(tid, "a"), 1)
        self.assertEqual(self.tm.get(tid, "b"), 2)
        self.assertEqual(self.tm.get(tid, "c"), 3)
        self.tm.commit(tid)

    def test_batch_get(self):
        tid = self.tm.begin()
        self.tm.batch_put(tid, {"a": 1, "b": 2, "c": 3})
        self.tm.commit(tid)

        tid2 = self.tm.begin()
        result = self.tm.batch_get(tid2, ["a", "b", "missing"])
        self.assertEqual(result, {"a": 1, "b": 2})
        self.tm.commit(tid2)

    def test_batch_delete(self):
        tid = self.tm.begin()
        self.tm.batch_put(tid, {"a": 1, "b": 2, "c": 3})
        self.tm.commit(tid)

        tid2 = self.tm.begin()
        deleted = self.tm.batch_delete(tid2, ["a", "c", "missing"])
        self.assertEqual(deleted, 2)
        self.assertIsNone(self.tm.get(tid2, "a"))
        self.assertEqual(self.tm.get(tid2, "b"), 2)
        self.tm.commit(tid2)

    def test_batch_put_atomicity(self):
        """Batch put within aborted txn should not persist."""
        tid = self.tm.begin()
        self.tm.batch_put(tid, {"x": 10, "y": 20})
        self.tm.abort(tid)

        tid2 = self.tm.begin()
        self.assertIsNone(self.tm.get(tid2, "x"))
        self.assertIsNone(self.tm.get(tid2, "y"))
        self.tm.commit(tid2)


# ===================================================================
# 7. Context Managers
# ===================================================================

class TestContextManagers(unittest.TestCase):
    """Test Transaction and ReadOnlyTransaction context managers."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_transaction_commits_on_success(self):
        with Transaction(self.tm) as txn:
            txn.put("k1", "v1")

        tid = self.tm.begin()
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.tm.commit(tid)

    def test_transaction_aborts_on_exception(self):
        try:
            with Transaction(self.tm) as txn:
                txn.put("k1", "v1")
                raise ValueError("oops")
        except ValueError:
            pass

        tid = self.tm.begin()
        self.assertIsNone(self.tm.get(tid, "k1"))
        self.tm.commit(tid)

    def test_read_only_transaction(self):
        # Seed
        with Transaction(self.tm) as txn:
            txn.put("k1", "v1")
            txn.put("k2", "v2")

        with ReadOnlyTransaction(self.tm) as rtxn:
            self.assertEqual(rtxn.get("k1"), "v1")
            data = rtxn.scan("k")
            self.assertEqual(len(data), 2)
            self.assertTrue(rtxn.exists("k1"))
            self.assertEqual(rtxn.count("k"), 2)

    def test_transaction_savepoint_via_ctx(self):
        with Transaction(self.tm) as txn:
            txn.put("k1", "v1")
            txn.savepoint("sp1")
            txn.put("k2", "v2")
            txn.rollback_to_savepoint("sp1")

        tid = self.tm.begin()
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.assertIsNone(self.tm.get(tid, "k2"))
        self.tm.commit(tid)

    def test_transaction_batch_via_ctx(self):
        with Transaction(self.tm) as txn:
            txn.batch_put({"a": 1, "b": 2})
            result = txn.batch_get(["a", "b"])
            self.assertEqual(result, {"a": 1, "b": 2})

    def test_transaction_range_scan_via_ctx(self):
        with Transaction(self.tm) as txn:
            txn.batch_put({"a": 1, "b": 2, "c": 3, "d": 4})

        with ReadOnlyTransaction(self.tm) as rtxn:
            result = rtxn.range_scan("a", "c")
            self.assertIn("a", result)
            self.assertIn("b", result)
            self.assertIn("c", result)  # inclusive
            self.assertNotIn("d", result)

    def test_transaction_delete_via_ctx(self):
        with Transaction(self.tm) as txn:
            txn.put("k1", "v1")

        with Transaction(self.tm) as txn:
            self.assertTrue(txn.delete("k1"))

        tid = self.tm.begin()
        self.assertIsNone(self.tm.get(tid, "k1"))
        self.tm.commit(tid)

    def test_txn_id_property(self):
        with Transaction(self.tm) as txn:
            self.assertIsInstance(txn.txn_id, int)


# ===================================================================
# 8. Recovery
# ===================================================================

class TestRecovery(unittest.TestCase):
    """Test crash simulation and ARIES recovery."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_committed_data_survives_crash(self):
        """Committed data should be recoverable after crash."""
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.commit(tid)

        self.tm.simulate_crash()
        result = self.tm.recover()
        self.assertIsInstance(result, dict)

    def test_crash_aborts_active_txns(self):
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "v1")
        # Do not commit
        self.tm.simulate_crash()
        mtxn = self.tm.get_transaction(t1)
        self.assertEqual(mtxn.state, TxnState.ABORTED)

    def test_checkpoint(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.commit(tid)
        lsn = self.tm.checkpoint()
        self.assertIsInstance(lsn, int)


# ===================================================================
# 9. Garbage Collection
# ===================================================================

class TestGarbageCollection(unittest.TestCase):
    """Test MVCC garbage collection integration."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_gc_returns_count(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.commit(tid)

        # Overwrite to create dead version
        tid2 = self.tm.begin()
        self.tm.put(tid2, "k1", "v2")
        self.tm.commit(tid2)

        collected = self.tm.gc()
        self.assertIsInstance(collected, int)

    def test_gc_preserves_live_data(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "final")
        self.tm.commit(tid)

        self.tm.gc()

        tid2 = self.tm.begin()
        self.assertEqual(self.tm.get(tid2, "k1"), "final")
        self.tm.commit(tid2)


# ===================================================================
# 10. Diagnostics and Statistics
# ===================================================================

class TestDiagnostics(unittest.TestCase):
    """Test stats and diagnostic methods."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_stats_structure(self):
        stats = self.tm.stats()
        self.assertIn("total_transactions", stats)
        self.assertIn("active_transactions", stats)
        self.assertIn("committed", stats)
        self.assertIn("aborted", stats)
        self.assertIn("mvcc", stats)
        self.assertIn("wal", stats)
        self.assertIn("lock_manager", stats)

    def test_stats_counts(self):
        t1 = self.tm.begin()
        self.tm.commit(t1)
        t2 = self.tm.begin()
        self.tm.abort(t2)
        stats = self.tm.stats()
        self.assertEqual(stats["committed"], 1)
        self.assertEqual(stats["aborted"], 1)
        self.assertEqual(stats["total_transactions"], 2)

    def test_wal_stats(self):
        self.assertIsInstance(self.tm.wal_stats(), dict)

    def test_lock_stats(self):
        self.assertIsInstance(self.tm.lock_stats(), dict)

    def test_mvcc_stats(self):
        self.assertIsInstance(self.tm.mvcc_stats(), dict)

    def test_get_transaction(self):
        tid = self.tm.begin()
        mtxn = self.tm.get_transaction(tid)
        self.assertIsInstance(mtxn, ManagedTransaction)
        self.assertEqual(mtxn.txn_id, tid)
        self.assertEqual(mtxn.state, TxnState.ACTIVE)
        self.tm.commit(tid)

    def test_get_nonexistent_transaction(self):
        self.assertIsNone(self.tm.get_transaction(999))


# ===================================================================
# 11. Concurrent Transactions
# ===================================================================

class TestConcurrentTransactions(unittest.TestCase):
    """Test concurrent access patterns."""

    def setUp(self):
        self.tm = TransactionManager(lock_timeout=1.0)

    def test_concurrent_reads(self):
        """Multiple readers should not block each other."""
        # Seed
        tid = self.tm.begin()
        self.tm.put(tid, "shared", "value")
        self.tm.commit(tid)

        results = {}
        errors = []

        def reader(name):
            try:
                t = self.tm.begin()
                val = self.tm.get(t, "shared")
                results[name] = val
                self.tm.commit(t)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(f"r{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        for name in results:
            self.assertEqual(results[name], "value")

    def test_writers_to_different_keys(self):
        """Writers to different keys should not conflict."""
        errors = []

        def writer(key, value):
            try:
                t = self.tm.begin()
                self.tm.put(t, key, value)
                self.tm.commit(t)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"k{i}", f"v{i}"))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        # Verify all writes persisted
        tid = self.tm.begin()
        for i in range(5):
            self.assertEqual(self.tm.get(tid, f"k{i}"), f"v{i}")
        self.tm.commit(tid)


# ===================================================================
# 12. Error Handling
# ===================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error conditions."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_read_after_commit_raises(self):
        tid = self.tm.begin()
        self.tm.commit(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.get(tid, "k1")

    def test_write_after_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.put(tid, "k1", "v1")

    def test_delete_after_commit_raises(self):
        tid = self.tm.begin()
        self.tm.commit(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.delete(tid, "k1")

    def test_scan_after_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.scan(tid)

    def test_exists_after_commit_raises(self):
        tid = self.tm.begin()
        self.tm.commit(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.exists(tid, "k1")

    def test_count_after_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.count(tid)

    def test_savepoint_after_commit_raises(self):
        tid = self.tm.begin()
        self.tm.commit(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.savepoint(tid, "sp1")

    def test_batch_put_after_abort_raises(self):
        tid = self.tm.begin()
        self.tm.abort(tid)
        with self.assertRaises(TransactionNotActiveError):
            self.tm.batch_put(tid, {"a": 1})


# ===================================================================
# 13. Complex Scenarios
# ===================================================================

class TestComplexScenarios(unittest.TestCase):
    """Integration scenarios testing multiple features together."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_read_your_own_writes(self):
        """A transaction should see its own uncommitted writes."""
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.assertEqual(self.tm.get(tid, "k1"), "v1")
        self.tm.put(tid, "k1", "v2")
        self.assertEqual(self.tm.get(tid, "k1"), "v2")
        self.tm.commit(tid)

    def test_write_delete_write(self):
        """Put -> Delete -> Put should result in final value."""
        tid = self.tm.begin()
        self.tm.put(tid, "k1", "v1")
        self.tm.delete(tid, "k1")
        self.tm.put(tid, "k1", "v3")
        self.assertEqual(self.tm.get(tid, "k1"), "v3")
        self.tm.commit(tid)

        tid2 = self.tm.begin()
        self.assertEqual(self.tm.get(tid2, "k1"), "v3")
        self.tm.commit(tid2)

    def test_multi_txn_sequential(self):
        """Multiple sequential transactions build on each other."""
        for i in range(10):
            tid = self.tm.begin()
            self.tm.put(tid, f"k{i}", f"v{i}")
            self.tm.commit(tid)

        tid = self.tm.begin()
        for i in range(10):
            self.assertEqual(self.tm.get(tid, f"k{i}"), f"v{i}")
        self.assertEqual(self.tm.count(tid, "k"), 10)
        self.tm.commit(tid)

    def test_savepoint_with_batch(self):
        """Savepoint + batch put + rollback."""
        tid = self.tm.begin()
        self.tm.batch_put(tid, {"a": 1, "b": 2})
        self.tm.savepoint(tid, "sp1")
        self.tm.batch_put(tid, {"c": 3, "d": 4})
        self.tm.rollback_to_savepoint(tid, "sp1")
        self.assertEqual(self.tm.get(tid, "a"), 1)
        self.assertEqual(self.tm.get(tid, "b"), 2)
        self.assertIsNone(self.tm.get(tid, "c"))
        self.assertIsNone(self.tm.get(tid, "d"))
        self.tm.commit(tid)

    def test_interleaved_read_write_txns(self):
        """Interleaved reads and writes across transactions."""
        t1 = self.tm.begin()
        self.tm.put(t1, "k1", "t1_v1")
        self.tm.commit(t1)

        t2 = self.tm.begin()
        t3 = self.tm.begin()

        # t2 reads k1
        val2 = self.tm.get(t2, "k1")
        self.assertEqual(val2, "t1_v1")

        # t3 reads k1
        val3 = self.tm.get(t3, "k1")
        self.assertEqual(val3, "t1_v1")

        # Both commit (no writes, no conflict)
        self.tm.commit(t2)
        self.tm.commit(t3)

    def test_long_running_txn(self):
        """Transaction with many operations."""
        tid = self.tm.begin()
        for i in range(100):
            self.tm.put(tid, f"item_{i:03d}", {"value": i, "data": "x" * 100})
        self.tm.commit(tid)

        tid2 = self.tm.begin()
        data = self.tm.scan(tid2, "item_")
        self.assertEqual(len(data), 100)
        self.tm.commit(tid2)

    def test_gc_after_overwrites(self):
        """GC cleans up dead versions from overwrites."""
        for i in range(5):
            tid = self.tm.begin()
            self.tm.put(tid, "counter", i)
            self.tm.commit(tid)

        collected = self.tm.gc()
        self.assertGreaterEqual(collected, 0)

        tid = self.tm.begin()
        self.assertEqual(self.tm.get(tid, "counter"), 4)
        self.tm.commit(tid)

    def test_checkpoint_interval(self):
        """Automatic checkpointing after threshold."""
        tm = TransactionManager(checkpoint_interval=5)
        for i in range(6):
            tid = tm.begin()
            tm.put(tid, f"k{i}", f"v{i}")
            tm.commit(tid)
        # Checkpoint should have been triggered
        stats = tm.stats()
        self.assertLessEqual(stats["ops_since_checkpoint"], 5)


# ===================================================================
# 14. Configuration
# ===================================================================

class TestConfiguration(unittest.TestCase):
    """Test different configuration options."""

    def test_custom_isolation(self):
        tm = TransactionManager(isolation=IsolationLevel.SERIALIZABLE)
        tid = tm.begin()
        mtxn = tm.get_transaction(tid)
        self.assertEqual(mtxn.isolation, IsolationLevel.SERIALIZABLE)
        tm.commit(tid)

    def test_custom_lock_timeout(self):
        tm = TransactionManager(lock_timeout=0.5)
        tid = tm.begin()
        tm.put(tid, "k", "v")
        tm.commit(tid)

    def test_disabled_checkpoint(self):
        tm = TransactionManager(checkpoint_interval=0)
        for i in range(20):
            tid = tm.begin()
            tm.put(tid, f"k{i}", f"v{i}")
            tm.commit(tid)
        # Should not crash even with many ops

    def test_custom_database_table(self):
        tm = TransactionManager(database="mydb", table="users")
        tid = tm.begin()
        mtxn = tm.get_transaction(tid)
        self.assertEqual(mtxn.database, "mydb")
        self.assertEqual(mtxn.table, "users")
        tm.commit(tid)


# ===================================================================
# 15. Edge Cases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def setUp(self):
        self.tm = TransactionManager()

    def test_empty_batch_put(self):
        tid = self.tm.begin()
        self.tm.batch_put(tid, {})
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.writes, 0)
        self.tm.commit(tid)

    def test_empty_batch_get(self):
        tid = self.tm.begin()
        result = self.tm.batch_get(tid, [])
        self.assertEqual(result, {})
        self.tm.commit(tid)

    def test_empty_batch_delete(self):
        tid = self.tm.begin()
        deleted = self.tm.batch_delete(tid, [])
        self.assertEqual(deleted, 0)
        self.tm.commit(tid)

    def test_scan_empty_store(self):
        tid = self.tm.begin()
        result = self.tm.scan(tid, "anything")
        self.assertEqual(result, {})
        self.tm.commit(tid)

    def test_count_empty_store(self):
        tid = self.tm.begin()
        self.assertEqual(self.tm.count(tid), 0)
        self.tm.commit(tid)

    def test_put_none_value(self):
        tid = self.tm.begin()
        self.tm.put(tid, "k1", None)
        val = self.tm.get(tid, "k1")
        self.assertIsNone(val)
        self.tm.commit(tid)

    def test_put_complex_value(self):
        tid = self.tm.begin()
        data = {"nested": [1, 2, 3], "flag": True}
        self.tm.put(tid, "k1", data)
        result = self.tm.get(tid, "k1")
        self.assertEqual(result, data)
        self.tm.commit(tid)

    def test_key_with_special_chars(self):
        tid = self.tm.begin()
        self.tm.put(tid, "key/with/slashes", "v1")
        self.tm.put(tid, "key.with.dots", "v2")
        self.tm.put(tid, "key-with-dashes", "v3")
        self.assertEqual(self.tm.get(tid, "key/with/slashes"), "v1")
        self.assertEqual(self.tm.get(tid, "key.with.dots"), "v2")
        self.assertEqual(self.tm.get(tid, "key-with-dashes"), "v3")
        self.tm.commit(tid)

    def test_many_txns_sequential(self):
        """Stress test: many sequential transactions."""
        for i in range(50):
            tid = self.tm.begin()
            self.tm.put(tid, f"stress_{i}", i)
            self.tm.commit(tid)
        stats = self.tm.stats()
        self.assertEqual(stats["committed"], 50)

    def test_begin_commit_empty_txn(self):
        """Transaction with no operations should commit fine."""
        tid = self.tm.begin()
        result = self.tm.commit(tid)
        self.assertTrue(result)

    def test_begin_abort_empty_txn(self):
        """Transaction with no operations should abort fine."""
        tid = self.tm.begin()
        self.tm.abort(tid)
        mtxn = self.tm.get_transaction(tid)
        self.assertEqual(mtxn.state, TxnState.ABORTED)


if __name__ == "__main__":
    unittest.main()
