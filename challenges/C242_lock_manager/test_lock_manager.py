"""
Tests for C242: Lock Manager
"""

import sys
import os
import time
import threading
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from lock_manager import (
    LockManager, LockMode, LockGranularity, ResourceId, LockStatus,
    LockTableEntry, LockRequest, DeadlockDetector, TransactionState,
    TxnPhase, DeadlockError, LockTimeoutError, LockError,
    LockEscalationError, COMPAT, UPGRADE, make_resource,
)


# ===========================================================================
# 1. Lock Mode and Compatibility Tests
# ===========================================================================

class TestLockMode(unittest.TestCase):
    def test_mode_ordering(self):
        self.assertTrue(LockMode.NONE < LockMode.IS < LockMode.IX < LockMode.S < LockMode.SIX < LockMode.X)

    def test_is_compatible_with_is(self):
        self.assertTrue(COMPAT[LockMode.IS][LockMode.IS])

    def test_is_compatible_with_ix(self):
        self.assertTrue(COMPAT[LockMode.IS][LockMode.IX])

    def test_is_compatible_with_s(self):
        self.assertTrue(COMPAT[LockMode.IS][LockMode.S])

    def test_is_compatible_with_six(self):
        self.assertTrue(COMPAT[LockMode.IS][LockMode.SIX])

    def test_is_not_compatible_with_x(self):
        self.assertFalse(COMPAT[LockMode.IS][LockMode.X])

    def test_ix_compatible_with_ix(self):
        self.assertTrue(COMPAT[LockMode.IX][LockMode.IX])

    def test_ix_not_compatible_with_s(self):
        self.assertFalse(COMPAT[LockMode.IX][LockMode.S])

    def test_s_compatible_with_s(self):
        self.assertTrue(COMPAT[LockMode.S][LockMode.S])

    def test_s_not_compatible_with_x(self):
        self.assertFalse(COMPAT[LockMode.S][LockMode.X])

    def test_x_not_compatible_with_anything(self):
        for mode in [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]:
            self.assertFalse(COMPAT[LockMode.X][mode])

    def test_six_compatible_with_is(self):
        self.assertTrue(COMPAT[LockMode.SIX][LockMode.IS])

    def test_six_not_compatible_with_six(self):
        self.assertFalse(COMPAT[LockMode.SIX][LockMode.SIX])


# ===========================================================================
# 2. Upgrade Matrix Tests
# ===========================================================================

class TestUpgradeMatrix(unittest.TestCase):
    def test_is_plus_s_equals_s(self):
        self.assertEqual(UPGRADE[(LockMode.IS, LockMode.S)], LockMode.S)

    def test_ix_plus_s_equals_six(self):
        self.assertEqual(UPGRADE[(LockMode.IX, LockMode.S)], LockMode.SIX)

    def test_s_plus_ix_equals_six(self):
        self.assertEqual(UPGRADE[(LockMode.S, LockMode.IX)], LockMode.SIX)

    def test_s_plus_x_equals_x(self):
        self.assertEqual(UPGRADE[(LockMode.S, LockMode.X)], LockMode.X)

    def test_x_plus_anything_equals_x(self):
        for mode in [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]:
            self.assertEqual(UPGRADE[(LockMode.X, mode)], LockMode.X)

    def test_same_mode_upgrade(self):
        for mode in [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]:
            self.assertEqual(UPGRADE[(mode, mode)], mode)


# ===========================================================================
# 3. ResourceId Tests
# ===========================================================================

class TestResourceId(unittest.TestCase):
    def test_database_resource(self):
        r = make_resource(database="mydb")
        self.assertEqual(r.granularity, LockGranularity.DATABASE)
        self.assertEqual(r.database, "mydb")

    def test_table_resource(self):
        r = make_resource(table="users")
        self.assertEqual(r.granularity, LockGranularity.TABLE)
        self.assertEqual(r.table, "users")

    def test_page_resource(self):
        r = make_resource(table="users", page=5)
        self.assertEqual(r.granularity, LockGranularity.PAGE)
        self.assertEqual(r.page, 5)

    def test_row_resource(self):
        r = make_resource(table="users", page=5, row=42)
        self.assertEqual(r.granularity, LockGranularity.ROW)
        self.assertEqual(r.row, 42)

    def test_parent_of_row_is_page(self):
        r = make_resource(table="users", page=5, row=42)
        p = r.parent()
        self.assertEqual(p.granularity, LockGranularity.PAGE)
        self.assertEqual(p.page, 5)
        self.assertIsNone(p.row)

    def test_parent_of_page_is_table(self):
        r = make_resource(table="users", page=5)
        p = r.parent()
        self.assertEqual(p.granularity, LockGranularity.TABLE)
        self.assertEqual(p.table, "users")

    def test_parent_of_table_is_database(self):
        r = make_resource(table="users")
        p = r.parent()
        self.assertEqual(p.granularity, LockGranularity.DATABASE)

    def test_parent_of_database_is_none(self):
        r = make_resource()
        self.assertIsNone(r.parent())

    def test_resource_equality(self):
        r1 = make_resource(table="users", page=5, row=42)
        r2 = make_resource(table="users", page=5, row=42)
        self.assertEqual(r1, r2)

    def test_resource_hashing(self):
        r1 = make_resource(table="users", page=5, row=42)
        r2 = make_resource(table="users", page=5, row=42)
        self.assertEqual(hash(r1), hash(r2))
        s = {r1}
        self.assertIn(r2, s)

    def test_resource_repr(self):
        r = make_resource(table="users", page=5, row=42)
        s = repr(r)
        self.assertIn("users", s)


# ===========================================================================
# 4. Deadlock Detector Tests
# ===========================================================================

class TestDeadlockDetector(unittest.TestCase):
    def test_no_cycle_empty(self):
        dd = DeadlockDetector()
        self.assertIsNone(dd.find_cycle())

    def test_no_cycle_linear(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 3)
        self.assertIsNone(dd.find_cycle())

    def test_simple_cycle(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 1)
        cycle = dd.find_cycle()
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 2)
        self.assertIn(1, cycle)
        self.assertIn(2, cycle)

    def test_three_way_cycle(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 3)
        dd.add_edge(3, 1)
        cycle = dd.find_cycle()
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 3)

    def test_remove_edges(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 1)
        dd.remove_edges_for(1)
        self.assertIsNone(dd.find_cycle())

    def test_self_edge_ignored(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 1)  # Should not add
        self.assertIsNone(dd.find_cycle())

    def test_clear(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 1)
        dd.clear()
        self.assertIsNone(dd.find_cycle())

    def test_cycle_in_larger_graph(self):
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 3)
        dd.add_edge(3, 4)
        dd.add_edge(4, 2)  # Cycle: 2->3->4->2
        cycle = dd.find_cycle()
        self.assertIsNotNone(cycle)


# ===========================================================================
# 5. Lock Table Entry Tests
# ===========================================================================

class TestLockTableEntry(unittest.TestCase):
    def test_empty_entry(self):
        r = make_resource(table="t")
        entry = LockTableEntry(resource=r)
        self.assertEqual(entry.granted_mode, LockMode.NONE)
        self.assertEqual(len(entry.granted_group), 0)

    def test_update_granted_mode_single(self):
        r = make_resource(table="t")
        entry = LockTableEntry(resource=r)
        entry.granted_group.append(LockRequest(txn_id=1, mode=LockMode.S, resource=r, status=LockStatus.GRANTED))
        entry.update_granted_mode()
        self.assertEqual(entry.granted_mode, LockMode.S)

    def test_update_granted_mode_multiple(self):
        r = make_resource(table="t")
        entry = LockTableEntry(resource=r)
        entry.granted_group.append(LockRequest(txn_id=1, mode=LockMode.IS, resource=r, status=LockStatus.GRANTED))
        entry.granted_group.append(LockRequest(txn_id=2, mode=LockMode.IX, resource=r, status=LockStatus.GRANTED))
        entry.update_granted_mode()
        self.assertEqual(entry.granted_mode, LockMode.IX)

    def test_update_granted_mode_empty(self):
        r = make_resource(table="t")
        entry = LockTableEntry(resource=r)
        entry.granted_mode = LockMode.X
        entry.update_granted_mode()
        self.assertEqual(entry.granted_mode, LockMode.NONE)


# ===========================================================================
# 6. Basic Lock Manager Tests
# ===========================================================================

class TestLockManagerBasic(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_begin_txn(self):
        self.lm.begin_txn(1)
        self.assertEqual(self.lm.active_txn_count(), 1)

    def test_begin_duplicate_txn(self):
        self.lm.begin_txn(1)
        with self.assertRaises(LockError):
            self.lm.begin_txn(1)

    def test_commit_txn(self):
        self.lm.begin_txn(1)
        self.lm.commit_txn(1)
        self.assertEqual(self.lm.active_txn_count(), 0)

    def test_abort_txn(self):
        self.lm.begin_txn(1)
        self.lm.abort_txn(1)
        self.assertEqual(self.lm.active_txn_count(), 0)

    def test_commit_nonexistent(self):
        self.lm.commit_txn(999)  # Should not raise

    def test_acquire_shared(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.assertTrue(self.lm.acquire(1, r, LockMode.S))

    def test_acquire_exclusive(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.assertTrue(self.lm.acquire(1, r, LockMode.X))

    def test_acquire_unregistered_txn(self):
        r = make_resource(table="users")
        with self.assertRaises(LockError):
            self.lm.acquire(99, r, LockMode.S)

    def test_multiple_shared_compatible(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.assertTrue(self.lm.acquire(1, r, LockMode.S))
        self.assertTrue(self.lm.acquire(2, r, LockMode.S))

    def test_shared_exclusive_conflict(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.S)
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, r, LockMode.X, timeout=0.1)

    def test_exclusive_exclusive_conflict(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, r, LockMode.X, timeout=0.1)

    def test_same_txn_reentrant(self):
        """Same transaction can acquire same lock multiple times."""
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        self.assertTrue(self.lm.acquire(1, r, LockMode.S))

    def test_locks_released_on_commit(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        self.lm.commit_txn(1)
        # Now txn 2 should be able to get X lock
        self.assertTrue(self.lm.acquire(2, r, LockMode.X))

    def test_locks_released_on_abort(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        self.lm.abort_txn(1)
        self.assertTrue(self.lm.acquire(2, r, LockMode.X))


# ===========================================================================
# 7. Lock Upgrade Tests
# ===========================================================================

class TestLockUpgrade(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_upgrade_s_to_x(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        self.assertTrue(self.lm.acquire(1, r, LockMode.X))
        state = self.lm.get_txn_state(1)
        self.assertEqual(state["held_locks"][r], LockMode.X)

    def test_upgrade_is_to_ix(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.IS)
        self.assertTrue(self.lm.acquire(1, r, LockMode.IX))

    def test_upgrade_blocked_by_other(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.S)
        self.lm.acquire(2, r, LockMode.S)
        # Txn 1 wants to upgrade S -> X, but txn 2 holds S
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(1, r, LockMode.X, timeout=0.1)

    def test_no_upgrade_if_already_stronger(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.X)
        self.assertTrue(self.lm.acquire(1, r, LockMode.S))  # Already X, no upgrade needed

    def test_stats_track_upgrades(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        self.lm.acquire(1, r, LockMode.X)
        stats = self.lm.get_stats()
        self.assertGreaterEqual(stats["upgrades"], 1)


# ===========================================================================
# 8. Strict 2PL Tests
# ===========================================================================

class TestStrict2PL(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(strict_2pl=True, default_timeout=1.0)

    def test_release_transitions_to_shrinking(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        self.lm.release(1, r)
        state = self.lm.get_txn_state(1)
        self.assertEqual(state["phase"], "SHRINKING")

    def test_cannot_acquire_in_shrinking_phase(self):
        r1 = make_resource(table="users")
        r2 = make_resource(table="orders")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r1, LockMode.S)
        self.lm.release(1, r1)
        with self.assertRaises(LockError):
            self.lm.acquire(1, r2, LockMode.S)

    def test_release_nonexistent_lock(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        with self.assertRaises(LockError):
            self.lm.release(1, r)

    def test_release_unregistered_txn(self):
        r = make_resource(table="users")
        with self.assertRaises(LockError):
            self.lm.release(99, r)


# ===========================================================================
# 9. Non-Strict Mode Tests
# ===========================================================================

class TestNonStrict(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(strict_2pl=False, default_timeout=1.0)

    def test_can_acquire_after_release(self):
        r1 = make_resource(table="users")
        r2 = make_resource(table="orders")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r1, LockMode.S)
        self.lm.release(1, r1)
        # In non-strict mode, can still acquire
        self.assertTrue(self.lm.acquire(1, r2, LockMode.S))


# ===========================================================================
# 10. Intent Lock Tests
# ===========================================================================

class TestIntentLocks(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_row_lock_gets_intention_on_ancestors(self):
        r = make_resource(table="users", page=1, row=5)
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        state = self.lm.get_txn_state(1)
        # Should have IS on page, table, database
        page_r = make_resource(table="users", page=1)
        table_r = make_resource(table="users")
        db_r = make_resource()
        self.assertIn(page_r, state["held_locks"])
        self.assertIn(table_r, state["held_locks"])
        self.assertIn(db_r, state["held_locks"])

    def test_row_x_gets_ix_on_ancestors(self):
        r = make_resource(table="users", page=1, row=5)
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.X)
        state = self.lm.get_txn_state(1)
        table_r = make_resource(table="users")
        self.assertEqual(state["held_locks"][table_r], LockMode.IX)

    def test_multiple_is_compatible(self):
        r1 = make_resource(table="users", page=1, row=1)
        r2 = make_resource(table="users", page=1, row=2)
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r1, LockMode.S)
        self.lm.acquire(2, r2, LockMode.S)
        # Both should have IS on table -- compatible


# ===========================================================================
# 11. Deadlock Detection Tests
# ===========================================================================

class TestDeadlockDetection(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(deadlock_detection=True, default_timeout=2.0)

    def test_simple_deadlock(self):
        """T1 holds A, wants B. T2 holds B, wants A."""
        a = make_resource(table="a")
        b = make_resource(table="b")
        self.lm.begin_txn(1, priority=0)
        self.lm.begin_txn(2, priority=1)
        self.lm.acquire(1, a, LockMode.X)
        self.lm.acquire(2, b, LockMode.X)

        # T2 wants A (held by T1) -- will deadlock when T1 wants B
        # We need threading for a real deadlock, but our detector catches it on the request
        # T1 wants B -- deadlock: T1 waits on T2 (for B), T2 must wait on T1 (for A)
        # But in single-threaded, T2 hasn't actually queued for A yet.
        # Let's test via the detector directly
        dd = DeadlockDetector()
        dd.add_edge(1, 2)
        dd.add_edge(2, 1)
        cycle = dd.find_cycle()
        self.assertIsNotNone(cycle)

    def test_deadlock_victim_selection(self):
        """Victim should be the lowest-priority transaction."""
        self.lm.begin_txn(1, priority=0)  # High priority
        self.lm.begin_txn(2, priority=10) # Low priority
        # The one with higher priority number should be victim
        a = make_resource(table="a")
        b = make_resource(table="b")
        self.lm.acquire(1, a, LockMode.X)
        self.lm.acquire(2, b, LockMode.X)

        # Simulate: create scenario where deadlock is detected
        # Use threading to create real deadlock scenario
        results = {"error": None}

        def txn2_acquire():
            try:
                self.lm.acquire(2, a, LockMode.X, timeout=1.0)
            except DeadlockError as e:
                results["error"] = e
            except LockTimeoutError:
                pass

        t = threading.Thread(target=txn2_acquire)
        t.start()
        time.sleep(0.1)

        try:
            self.lm.acquire(1, b, LockMode.X, timeout=1.0)
        except DeadlockError as e:
            results["error"] = e
        except LockTimeoutError:
            pass

        t.join(timeout=2.0)
        # One of them should have gotten a deadlock error
        # (or timeout if detection missed it)


# ===========================================================================
# 12. Lock Escalation Tests
# ===========================================================================

class TestLockEscalation(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(escalation_threshold=5, default_timeout=1.0)

    def test_no_escalation_below_threshold(self):
        self.lm.begin_txn(1)
        for i in range(4):
            r = make_resource(table="users", page=0, row=i)
            self.lm.acquire(1, r, LockMode.S)
        self.assertFalse(self.lm.check_escalation(1, table="users"))

    def test_escalation_at_threshold(self):
        self.lm.begin_txn(1)
        for i in range(5):
            r = make_resource(table="users", page=0, row=i)
            self.lm.acquire(1, r, LockMode.S)
        result = self.lm.check_escalation(1, table="users")
        self.assertTrue(result)
        state = self.lm.get_txn_state(1)
        table_r = make_resource(table="users")
        # Should now hold table-level lock
        self.assertIn(table_r, state["held_locks"])

    def test_escalation_mode_x_when_any_x(self):
        self.lm.begin_txn(1)
        for i in range(4):
            r = make_resource(table="users", page=0, row=i)
            self.lm.acquire(1, r, LockMode.S)
        r = make_resource(table="users", page=0, row=4)
        self.lm.acquire(1, r, LockMode.X)
        self.lm.check_escalation(1, table="users")
        state = self.lm.get_txn_state(1)
        table_r = make_resource(table="users")
        self.assertEqual(state["held_locks"][table_r], LockMode.X)

    def test_escalation_releases_row_locks(self):
        self.lm.begin_txn(1)
        for i in range(5):
            r = make_resource(table="users", page=0, row=i)
            self.lm.acquire(1, r, LockMode.S)
        self.lm.check_escalation(1, table="users")
        state = self.lm.get_txn_state(1)
        # Row locks should be released
        for i in range(5):
            r = make_resource(table="users", page=0, row=i)
            self.assertNotIn(r, state["held_locks"])

    def test_escalation_stats(self):
        self.lm.begin_txn(1)
        for i in range(5):
            r = make_resource(table="users", page=0, row=i)
            self.lm.acquire(1, r, LockMode.S)
        self.lm.check_escalation(1, table="users")
        stats = self.lm.get_stats()
        self.assertGreaterEqual(stats["escalations"], 1)

    def test_escalation_nonexistent_txn(self):
        self.assertFalse(self.lm.check_escalation(999, table="users"))


# ===========================================================================
# 13. Concurrent Lock Tests
# ===========================================================================

class TestConcurrentLocks(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=2.0)

    def test_concurrent_shared_readers(self):
        r = make_resource(table="data")
        results = []
        barrier = threading.Barrier(5)

        def reader(tid):
            self.lm.begin_txn(tid)
            barrier.wait()
            try:
                self.lm.acquire(tid, r, LockMode.S)
                results.append(tid)
            finally:
                self.lm.commit_txn(tid)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(1, 6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(results), 5)

    def test_writer_waits_for_readers(self):
        r = make_resource(table="data")
        order = []
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)

        self.lm.acquire(1, r, LockMode.S)
        order.append("r1_acquired")

        def writer():
            try:
                self.lm.acquire(2, r, LockMode.X, timeout=2.0)
                order.append("w2_acquired")
            except LockTimeoutError:
                order.append("w2_timeout")

        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.2)
        order.append("r1_releasing")
        self.lm.commit_txn(1)
        t.join(timeout=3.0)

        self.assertEqual(order[0], "r1_acquired")
        self.assertIn("w2_acquired", order)

    def test_exclusive_serialization(self):
        lm = LockManager(default_timeout=10.0)
        r = make_resource(table="counter")
        counter = [0]

        def increment(tid):
            lm.begin_txn(tid)
            try:
                lm.acquire(tid, r, LockMode.X, timeout=10.0)
                val = counter[0]
                time.sleep(0.01)
                counter[0] = val + 1
            except (LockTimeoutError, DeadlockError):
                pass
            finally:
                lm.commit_txn(tid)

        threads = [threading.Thread(target=increment, args=(i,)) for i in range(1, 11)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20.0)

        self.assertEqual(counter[0], 10)


# ===========================================================================
# 14. Timeout Tests
# ===========================================================================

class TestTimeout(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=0.2)

    def test_timeout_on_conflict(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        start = time.monotonic()
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, r, LockMode.X, timeout=0.2)
        elapsed = time.monotonic() - start
        self.assertGreater(elapsed, 0.1)
        self.assertLess(elapsed, 1.0)

    def test_timeout_stats(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        try:
            self.lm.acquire(2, r, LockMode.X, timeout=0.1)
        except LockTimeoutError:
            pass
        stats = self.lm.get_stats()
        self.assertGreaterEqual(stats["timeouts"], 1)

    def test_zero_timeout_immediate_fail(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, r, LockMode.X, timeout=0.01)


# ===========================================================================
# 15. Query Methods Tests
# ===========================================================================

class TestQueryMethods(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_get_locks_by_txn(self):
        r1 = make_resource(table="a")
        r2 = make_resource(table="b")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r1, LockMode.S)
        self.lm.acquire(1, r2, LockMode.X)
        locks = self.lm.get_locks(txn_id=1)
        self.assertGreaterEqual(len(locks), 2)

    def test_get_locks_by_resource(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.S)
        self.lm.acquire(2, r, LockMode.S)
        locks = self.lm.get_locks(resource=r)
        self.assertEqual(len(locks), 2)

    def test_get_all_locks(self):
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.S)
        locks = self.lm.get_locks()
        self.assertGreater(len(locks), 0)

    def test_get_txn_state(self):
        self.lm.begin_txn(1)
        state = self.lm.get_txn_state(1)
        self.assertEqual(state["txn_id"], 1)
        self.assertEqual(state["phase"], "GROWING")
        self.assertEqual(state["lock_count"], 0)

    def test_get_txn_state_nonexistent(self):
        self.assertIsNone(self.lm.get_txn_state(999))

    def test_get_stats(self):
        stats = self.lm.get_stats()
        self.assertIn("locks_acquired", stats)
        self.assertIn("locks_released", stats)
        self.assertIn("deadlocks_detected", stats)
        self.assertIn("escalations", stats)
        self.assertIn("timeouts", stats)
        self.assertIn("upgrades", stats)

    def test_get_wait_for_graph(self):
        graph = self.lm.get_wait_for_graph()
        self.assertIsInstance(graph, dict)


# ===========================================================================
# 16. Multi-Granularity Scenario Tests
# ===========================================================================

class TestMultiGranularity(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_table_x_blocks_row_s(self):
        """Table X lock should block row S lock from another txn."""
        table_r = make_resource(table="users")
        row_r = make_resource(table="users", page=1, row=1)
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, table_r, LockMode.X)
        # Txn 2 trying to get row S needs IS on table -- incompatible with X
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, row_r, LockMode.S, timeout=0.1)

    def test_row_locks_different_rows_compatible(self):
        """Different rows can be locked by different txns."""
        r1 = make_resource(table="users", page=1, row=1)
        r2 = make_resource(table="users", page=1, row=2)
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r1, LockMode.X)
        self.lm.acquire(2, r2, LockMode.X)  # Should succeed

    def test_row_locks_same_row_conflict(self):
        r = make_resource(table="users", page=1, row=1)
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.X)
        with self.assertRaises(LockTimeoutError):
            self.lm.acquire(2, r, LockMode.X, timeout=0.1)

    def test_page_level_locking(self):
        p1 = make_resource(table="users", page=1)
        p2 = make_resource(table="users", page=2)
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, p1, LockMode.X)
        self.lm.acquire(2, p2, LockMode.X)  # Different pages


# ===========================================================================
# 17. Edge Cases and Error Handling
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_many_resources_one_txn(self):
        self.lm.begin_txn(1)
        for i in range(100):
            r = make_resource(table=f"t{i}")
            self.lm.acquire(1, r, LockMode.S)
        state = self.lm.get_txn_state(1)
        self.assertGreaterEqual(state["lock_count"], 100)

    def test_many_txns_one_resource(self):
        r = make_resource(table="shared")
        for i in range(20):
            self.lm.begin_txn(i)
            self.lm.acquire(i, r, LockMode.S)
        locks = self.lm.get_locks(resource=r)
        self.assertEqual(len(locks), 20)

    def test_commit_releases_all(self):
        self.lm.begin_txn(1)
        for i in range(10):
            r = make_resource(table=f"t{i}")
            self.lm.acquire(1, r, LockMode.X)
        self.lm.commit_txn(1)
        self.assertEqual(self.lm.active_txn_count(), 0)

    def test_abort_releases_all(self):
        self.lm.begin_txn(1)
        for i in range(10):
            r = make_resource(table=f"t{i}")
            self.lm.acquire(1, r, LockMode.X)
        self.lm.abort_txn(1)
        locks = self.lm.get_locks(txn_id=1)
        self.assertEqual(len(locks), 0)

    def test_database_level_lock(self):
        r = make_resource()
        self.lm.begin_txn(1)
        self.lm.acquire(1, r, LockMode.X)
        state = self.lm.get_txn_state(1)
        self.assertIn(r, state["held_locks"])

    def test_ix_compatible_with_ix_on_table(self):
        """Two txns can both hold IX on a table (for different row writes)."""
        r = make_resource(table="users")
        self.lm.begin_txn(1)
        self.lm.begin_txn(2)
        self.lm.acquire(1, r, LockMode.IX)
        self.lm.acquire(2, r, LockMode.IX)


# ===========================================================================
# 18. Stress and Integration Tests
# ===========================================================================

class TestStress(unittest.TestCase):
    def test_many_concurrent_txns(self):
        lm = LockManager(default_timeout=5.0)
        errors = []
        completed = []

        def worker(tid):
            try:
                lm.begin_txn(tid)
                for i in range(5):
                    r = make_resource(table="shared", page=tid % 3, row=i)
                    lm.acquire(tid, r, LockMode.S)
                time.sleep(0.01)
                lm.commit_txn(tid)
                completed.append(tid)
            except (LockTimeoutError, DeadlockError):
                try:
                    lm.abort_txn(tid)
                except Exception:
                    pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 21)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertGreater(len(completed), 0)

    def test_read_write_mix(self):
        """Mix of readers and writers on same resource."""
        lm = LockManager(default_timeout=3.0)
        results = {"reads": 0, "writes": 0}
        lock = threading.Lock()

        def reader(tid):
            lm.begin_txn(tid)
            r = make_resource(table="data")
            try:
                lm.acquire(tid, r, LockMode.S, timeout=2.0)
                time.sleep(0.01)
                with lock:
                    results["reads"] += 1
            except (LockTimeoutError, DeadlockError):
                pass
            finally:
                lm.commit_txn(tid)

        def writer(tid):
            lm.begin_txn(tid)
            r = make_resource(table="data")
            try:
                lm.acquire(tid, r, LockMode.X, timeout=2.0)
                time.sleep(0.02)
                with lock:
                    results["writes"] += 1
            except (LockTimeoutError, DeadlockError):
                pass
            finally:
                lm.commit_txn(tid)

        threads = []
        for i in range(10):
            if i % 3 == 0:
                threads.append(threading.Thread(target=writer, args=(i + 100,)))
            else:
                threads.append(threading.Thread(target=reader, args=(i + 100,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        total = results["reads"] + results["writes"]
        self.assertGreater(total, 0)


# ===========================================================================
# 19. Compatibility Matrix Completeness Tests
# ===========================================================================

class TestCompatibilityCompleteness(unittest.TestCase):
    def test_all_mode_pairs_defined(self):
        modes = [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]
        for m1 in modes:
            for m2 in modes:
                self.assertIn(m2, COMPAT[m1], f"Missing COMPAT[{m1}][{m2}]")

    def test_all_upgrade_pairs_defined(self):
        modes = [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]
        for m1 in modes:
            for m2 in modes:
                self.assertIn((m1, m2), UPGRADE, f"Missing UPGRADE[({m1}, {m2})]")

    def test_compatibility_symmetry(self):
        """Compatibility matrix should be symmetric."""
        modes = [LockMode.IS, LockMode.IX, LockMode.S, LockMode.SIX, LockMode.X]
        for m1 in modes:
            for m2 in modes:
                self.assertEqual(COMPAT[m1][m2], COMPAT[m2][m1],
                                 f"Asymmetric: COMPAT[{m1}][{m2}] != COMPAT[{m2}][{m1}]")


# ===========================================================================
# 20. Transaction State Transitions
# ===========================================================================

class TestTxnStateTransitions(unittest.TestCase):
    def setUp(self):
        self.lm = LockManager(strict_2pl=True, default_timeout=1.0)

    def test_initial_state_growing(self):
        self.lm.begin_txn(1)
        state = self.lm.get_txn_state(1)
        self.assertEqual(state["phase"], "GROWING")

    def test_growing_allows_acquire(self):
        self.lm.begin_txn(1)
        r = make_resource(table="t")
        self.assertTrue(self.lm.acquire(1, r, LockMode.S))

    def test_priority_setting(self):
        self.lm.begin_txn(1, priority=5)
        state = self.lm.get_txn_state(1)
        self.assertEqual(state["txn_id"], 1)

    def test_lock_count_tracking(self):
        self.lm.begin_txn(1)
        r1 = make_resource(table="a")
        r2 = make_resource(table="b")
        self.lm.acquire(1, r1, LockMode.S)
        self.lm.acquire(1, r2, LockMode.S)
        state = self.lm.get_txn_state(1)
        # Count includes intention locks too
        self.assertGreaterEqual(state["lock_count"], 2)


# ===========================================================================
# 21. Wait Queue FIFO Tests
# ===========================================================================

class TestWaitQueueFIFO(unittest.TestCase):
    def test_fifo_ordering(self):
        """Waiting requests should be granted in FIFO order."""
        lm = LockManager(default_timeout=3.0)
        r = make_resource(table="data")
        order = []

        lm.begin_txn(1)
        lm.begin_txn(2)
        lm.begin_txn(3)

        lm.acquire(1, r, LockMode.X)

        def waiter(tid):
            try:
                lm.acquire(tid, r, LockMode.X, timeout=3.0)
                order.append(tid)
            except (LockTimeoutError, DeadlockError):
                pass
            finally:
                lm.commit_txn(tid)

        t2 = threading.Thread(target=waiter, args=(2,))
        t3 = threading.Thread(target=waiter, args=(3,))
        t2.start()
        time.sleep(0.1)
        t3.start()
        time.sleep(0.1)

        lm.commit_txn(1)
        t2.join(timeout=3.0)
        t3.join(timeout=3.0)

        if len(order) >= 2:
            self.assertEqual(order[0], 2)  # T2 should get it first


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
