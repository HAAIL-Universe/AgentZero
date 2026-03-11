"""
Tests for C216: Lock Manager

Tests cover:
- Lock modes and compatibility matrix
- Resource hierarchy and identification
- Basic acquire/release
- Multiple readers (shared locks)
- Exclusive lock exclusivity
- Lock upgrades (S->X, IS->IX, etc.)
- Lock downgrades
- Intention locks (IS, IX, SIX)
- Hierarchical locking (auto intention)
- Wait-for graph and deadlock detection
- Lock timeouts
- Lock escalation
- Transaction lock tracking
- Two-phase locking protocol
- Multi-granularity locker
- Concurrent access patterns
- Lock stats and diagnostics
- Analyzer reports
"""

import pytest
import threading
import time
from lock_manager import (
    LockMode, LockResult, ResourceLevel, ResourceId,
    LockManager, LockEntry, LockRequest, LockStats,
    WaitForGraph, LockManagerAnalyzer, TwoPhaseLockHelper,
    MultiGranularityLocker,
    DeadlockError, LockTimeoutError, LockError,
    COMPATIBILITY, UPGRADE,
    make_db, make_table, make_page, make_row,
)


# ============================================================
# Lock Mode and Compatibility
# ============================================================

class TestLockMode:
    def test_mode_ordering(self):
        assert LockMode.IS < LockMode.IX < LockMode.S < LockMode.SIX < LockMode.X

    def test_mode_values(self):
        assert LockMode.IS.value == 0
        assert LockMode.X.value == 4

    def test_compatibility_ss(self):
        """Shared locks are compatible with each other."""
        assert COMPATIBILITY[LockMode.S][LockMode.S] is True

    def test_compatibility_sx(self):
        """Shared and exclusive are incompatible."""
        assert COMPATIBILITY[LockMode.S][LockMode.X] is False

    def test_compatibility_xx(self):
        """Exclusive locks are incompatible with everything."""
        assert COMPATIBILITY[LockMode.X][LockMode.X] is False
        assert COMPATIBILITY[LockMode.X][LockMode.S] is False
        assert COMPATIBILITY[LockMode.X][LockMode.IS] is False

    def test_compatibility_is_is(self):
        assert COMPATIBILITY[LockMode.IS][LockMode.IS] is True

    def test_compatibility_ix_ix(self):
        assert COMPATIBILITY[LockMode.IX][LockMode.IX] is True

    def test_compatibility_is_x(self):
        assert COMPATIBILITY[LockMode.IS][LockMode.X] is False

    def test_compatibility_ix_s(self):
        assert COMPATIBILITY[LockMode.IX][LockMode.S] is False

    def test_compatibility_six_is(self):
        assert COMPATIBILITY[LockMode.SIX][LockMode.IS] is True

    def test_compatibility_six_ix(self):
        assert COMPATIBILITY[LockMode.SIX][LockMode.IX] is False

    def test_compatibility_six_s(self):
        assert COMPATIBILITY[LockMode.SIX][LockMode.S] is False

    def test_compatibility_symmetric_check(self):
        """Verify asymmetric entries in compatibility matrix."""
        # IS + SIX: True
        assert COMPATIBILITY[LockMode.IS][LockMode.SIX] is True
        # SIX + IS: True
        assert COMPATIBILITY[LockMode.SIX][LockMode.IS] is True

    def test_upgrade_is_ix(self):
        assert UPGRADE[(LockMode.IS, LockMode.IX)] == LockMode.IX

    def test_upgrade_ix_s(self):
        assert UPGRADE[(LockMode.IX, LockMode.S)] == LockMode.SIX

    def test_upgrade_s_x(self):
        assert UPGRADE[(LockMode.S, LockMode.X)] == LockMode.X

    def test_upgrade_idempotent(self):
        for mode in LockMode:
            assert UPGRADE[(mode, mode)] == mode


# ============================================================
# Resource Hierarchy
# ============================================================

class TestResourceId:
    def test_database_key(self):
        r = make_db("mydb")
        assert r.key == "db:mydb"
        assert r.level == ResourceLevel.DATABASE

    def test_table_key(self):
        r = make_table("mydb", "users")
        assert r.key == "tbl:mydb.users"
        assert r.level == ResourceLevel.TABLE

    def test_page_key(self):
        r = make_page("mydb", "users", 5)
        assert r.key == "pg:mydb.users.5"
        assert r.level == ResourceLevel.PAGE

    def test_row_key(self):
        r = make_row("mydb", "users", 5, 42)
        assert r.key == "row:mydb.users.5.42"
        assert r.level == ResourceLevel.ROW

    def test_parent_row(self):
        r = make_row("db", "t", 1, 2)
        p = r.parent
        assert p == make_page("db", "t", 1)

    def test_parent_page(self):
        r = make_page("db", "t", 1)
        p = r.parent
        assert p == make_table("db", "t")

    def test_parent_table(self):
        r = make_table("db", "t")
        p = r.parent
        assert p == make_db("db")

    def test_parent_database(self):
        r = make_db("db")
        assert r.parent is None

    def test_equality(self):
        r1 = make_row("db", "t", 1, 2)
        r2 = make_row("db", "t", 1, 2)
        assert r1 == r2

    def test_hash(self):
        r1 = make_row("db", "t", 1, 2)
        r2 = make_row("db", "t", 1, 2)
        assert hash(r1) == hash(r2)
        s = {r1}
        assert r2 in s

    def test_inequality(self):
        r1 = make_row("db", "t", 1, 2)
        r2 = make_row("db", "t", 1, 3)
        assert r1 != r2

    def test_repr(self):
        r = make_table("db", "users")
        assert "tbl:db.users" in repr(r)


# ============================================================
# LockEntry
# ============================================================

class TestLockEntry:
    def test_is_free(self):
        e = LockEntry(resource=make_table("db", "t"))
        assert e.is_free

    def test_not_free_with_grant(self):
        e = LockEntry(resource=make_table("db", "t"))
        e.granted[1] = LockMode.S
        assert not e.is_free

    def test_group_mode_none(self):
        e = LockEntry(resource=make_table("db", "t"))
        assert e.group_mode() is None

    def test_group_mode_single(self):
        e = LockEntry(resource=make_table("db", "t"))
        e.granted[1] = LockMode.S
        assert e.group_mode() == LockMode.S

    def test_group_mode_multiple(self):
        e = LockEntry(resource=make_table("db", "t"))
        e.granted[1] = LockMode.IS
        e.granted[2] = LockMode.IX
        assert e.group_mode() == LockMode.IX

    def test_is_compatible(self):
        e = LockEntry(resource=make_table("db", "t"))
        e.granted[1] = LockMode.S
        assert e.is_compatible(LockMode.S)
        assert not e.is_compatible(LockMode.X)

    def test_is_compatible_exclude(self):
        e = LockEntry(resource=make_table("db", "t"))
        e.granted[1] = LockMode.X
        assert not e.is_compatible(LockMode.S)
        assert e.is_compatible(LockMode.S, exclude_tx=1)


# ============================================================
# Wait-For Graph
# ============================================================

class TestWaitForGraph:
    def test_empty(self):
        g = WaitForGraph()
        assert g.detect_cycle() == []

    def test_no_cycle(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert g.detect_cycle() == []

    def test_simple_cycle(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 1)
        cycle = g.detect_cycle()
        assert len(cycle) > 0

    def test_three_node_cycle(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        cycle = g.detect_cycle()
        assert len(cycle) > 0
        assert cycle[0] == cycle[-1]  # Cycle starts and ends at same node

    def test_remove_node(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        g.remove_node(2)
        assert g.detect_cycle() == []

    def test_remove_edges(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 1)
        g.remove_edges_for(1)
        assert g.detect_cycle() == []

    def test_has_cycle_from_node(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        cycle = g.has_cycle(1)
        assert len(cycle) > 0

    def test_no_self_edge(self):
        g = WaitForGraph()
        g.add_edge(1, 1)
        assert 1 not in g._edges.get(1, set())

    def test_clear(self):
        g = WaitForGraph()
        g.add_edge(1, 2)
        g.clear()
        assert g.edges == {}


# ============================================================
# Basic Lock Acquire / Release
# ============================================================

class TestBasicLocking:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)
        self.res = make_table("db", "users")

    def test_acquire_shared(self):
        result = self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_acquire_exclusive(self):
        result = self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_release(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert self.lm.release(1, self.res)

    def test_release_nonexistent(self):
        assert not self.lm.release(1, self.res)

    def test_is_locked(self):
        assert not self.lm.is_locked(self.res)
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert self.lm.is_locked(self.res)

    def test_get_lock_mode(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert self.lm.get_lock_mode(1, self.res) == LockMode.S
        assert self.lm.get_lock_mode(2, self.res) is None

    def test_get_holders(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        self.lm.acquire(2, self.res, LockMode.S, auto_intention=False)
        holders = self.lm.get_holders(self.res)
        assert holders == {1: LockMode.S, 2: LockMode.S}

    def test_multiple_shared(self):
        r1 = self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        r2 = self.lm.acquire(2, self.res, LockMode.S, auto_intention=False)
        r3 = self.lm.acquire(3, self.res, LockMode.S, auto_intention=False)
        assert r1 == r2 == r3 == LockResult.GRANTED

    def test_exclusive_blocks_shared(self):
        self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        result = self.lm.try_acquire(2, self.res, LockMode.S, auto_intention=False)
        assert result == LockResult.DENIED

    def test_shared_blocks_exclusive(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        result = self.lm.try_acquire(2, self.res, LockMode.X, auto_intention=False)
        assert result == LockResult.DENIED

    def test_release_grants_waiter(self):
        self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        granted = threading.Event()

        def waiter():
            self.lm.acquire(2, self.res, LockMode.S, auto_intention=False, timeout=2.0)
            granted.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        self.lm.release(1, self.res)
        t.join(timeout=2.0)
        assert granted.is_set()

    def test_reentrant_same_mode(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        result = self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_release_all(self):
        r1 = make_table("db", "t1")
        r2 = make_table("db", "t2")
        self.lm.acquire(1, r1, LockMode.S, auto_intention=False)
        self.lm.acquire(1, r2, LockMode.X, auto_intention=False)
        count = self.lm.release_all(1)
        assert count == 2
        assert not self.lm.is_locked(r1)
        assert not self.lm.is_locked(r2)

    def test_lock_count(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert self.lm.lock_count() == 1

    def test_tx_count(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        self.lm.acquire(2, self.res, LockMode.S, auto_intention=False)
        assert self.lm.tx_count() == 2

    def test_reset(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        self.lm.reset()
        assert self.lm.lock_count() == 0
        assert self.lm.tx_count() == 0


# ============================================================
# Lock Upgrade
# ============================================================

class TestLockUpgrade:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)
        self.res = make_table("db", "users")

    def test_upgrade_s_to_x(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        result = self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        assert result == LockResult.UPGRADED
        assert self.lm.get_lock_mode(1, self.res) == LockMode.X

    def test_upgrade_is_to_ix(self):
        self.lm.acquire(1, self.res, LockMode.IS, auto_intention=False)
        result = self.lm.acquire(1, self.res, LockMode.IX, auto_intention=False)
        assert result == LockResult.UPGRADED
        assert self.lm.get_lock_mode(1, self.res) == LockMode.IX

    def test_upgrade_ix_to_six(self):
        self.lm.acquire(1, self.res, LockMode.IX, auto_intention=False)
        result = self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert result == LockResult.UPGRADED
        assert self.lm.get_lock_mode(1, self.res) == LockMode.SIX

    def test_upgrade_blocked_by_other_shared(self):
        """S->X upgrade blocked when another txn holds S."""
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        self.lm.acquire(2, self.res, LockMode.S, auto_intention=False)
        result = self.lm.try_acquire(1, self.res, LockMode.X, auto_intention=False)
        assert result == LockResult.DENIED

    def test_no_upgrade_needed(self):
        """Already holding X, requesting S should be a no-op."""
        self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        result = self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert result == LockResult.GRANTED  # Already sufficient


# ============================================================
# Lock Downgrade
# ============================================================

class TestLockDowngrade:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)
        self.res = make_table("db", "users")

    def test_downgrade_x_to_s(self):
        self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        assert self.lm.downgrade(1, self.res, LockMode.S)
        assert self.lm.get_lock_mode(1, self.res) == LockMode.S

    def test_downgrade_not_holding(self):
        assert not self.lm.downgrade(1, self.res, LockMode.S)

    def test_downgrade_not_lower(self):
        self.lm.acquire(1, self.res, LockMode.S, auto_intention=False)
        assert not self.lm.downgrade(1, self.res, LockMode.X)

    def test_downgrade_grants_waiters(self):
        """Downgrade from X to S should allow other shared readers."""
        self.lm.acquire(1, self.res, LockMode.X, auto_intention=False)
        granted = threading.Event()

        def waiter():
            self.lm.acquire(2, self.res, LockMode.S, auto_intention=False, timeout=2.0)
            granted.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        self.lm.downgrade(1, self.res, LockMode.S)
        t.join(timeout=2.0)
        assert granted.is_set()


# ============================================================
# Intention Locks
# ============================================================

class TestIntentionLocks:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_is_compatible_with_is(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.IS, auto_intention=False)
        result = self.lm.acquire(2, res, LockMode.IS, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_ix_compatible_with_ix(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.IX, auto_intention=False)
        result = self.lm.acquire(2, res, LockMode.IX, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_ix_incompatible_with_s(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.IX, auto_intention=False)
        result = self.lm.try_acquire(2, res, LockMode.S, auto_intention=False)
        assert result == LockResult.DENIED

    def test_six_allows_is(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.SIX, auto_intention=False)
        result = self.lm.acquire(2, res, LockMode.IS, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_six_blocks_ix(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.SIX, auto_intention=False)
        result = self.lm.try_acquire(2, res, LockMode.IX, auto_intention=False)
        assert result == LockResult.DENIED


# ============================================================
# Hierarchical Locking (Auto Intention)
# ============================================================

class TestHierarchicalLocking:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_row_lock_creates_intention(self):
        """Locking a row should auto-acquire IS/IX on parent table and db."""
        row = make_row("db", "users", 0, 1)
        self.lm.acquire(1, row, LockMode.S)
        # Should have IS on db, table, page
        assert self.lm.get_lock_mode(1, make_db("db")) == LockMode.IS
        assert self.lm.get_lock_mode(1, make_table("db", "users")) == LockMode.IS
        assert self.lm.get_lock_mode(1, make_page("db", "users", 0)) == LockMode.IS
        assert self.lm.get_lock_mode(1, row) == LockMode.S

    def test_exclusive_row_creates_ix(self):
        row = make_row("db", "users", 0, 1)
        self.lm.acquire(1, row, LockMode.X)
        assert self.lm.get_lock_mode(1, make_db("db")) == LockMode.IX
        assert self.lm.get_lock_mode(1, make_table("db", "users")) == LockMode.IX
        assert self.lm.get_lock_mode(1, make_page("db", "users", 0)) == LockMode.IX

    def test_intention_allows_multiple_writers(self):
        """Two txns can lock different rows exclusively (IX is compatible with IX)."""
        r1 = make_row("db", "users", 0, 1)
        r2 = make_row("db", "users", 0, 2)
        self.lm.acquire(1, r1, LockMode.X)
        result = self.lm.acquire(2, r2, LockMode.X)
        assert result == LockResult.GRANTED

    def test_table_x_blocks_row_lock(self):
        """If txn1 holds X on table, txn2 can't get IS (needed for row lock)."""
        table = make_table("db", "users")
        row = make_row("db", "users", 0, 1)
        self.lm.acquire(1, table, LockMode.X, auto_intention=True)
        result = self.lm.try_acquire(2, row, LockMode.S)
        assert result == LockResult.DENIED

    def test_mixed_read_write_different_rows(self):
        """Reader and writer on different rows should coexist."""
        r1 = make_row("db", "t", 0, 1)
        r2 = make_row("db", "t", 0, 2)
        self.lm.acquire(1, r1, LockMode.S)
        # tx1 has IS on table, tx2 needs IX -- IS+IX compatible
        # Actually tx1 gets IS on table/page, tx2 needs IX on table/page (for X row)
        # IS+IX compatible: yes
        result = self.lm.acquire(2, r2, LockMode.X)
        assert result == LockResult.GRANTED


# ============================================================
# Deadlock Detection
# ============================================================

class TestDeadlockDetection:
    def setup_method(self):
        self.lm = LockManager(default_timeout=2.0, deadlock_detection=True)

    def test_deadlock_two_txns(self):
        """Classic deadlock: tx1 holds A, wants B; tx2 holds B, wants A."""
        a = make_table("db", "a")
        b = make_table("db", "b")
        self.lm.acquire(1, a, LockMode.X, auto_intention=False)
        self.lm.acquire(2, b, LockMode.X, auto_intention=False)

        deadlock_detected = threading.Event()

        def tx1_wait():
            try:
                self.lm.acquire(1, b, LockMode.X, auto_intention=False, timeout=2.0)
            except DeadlockError:
                deadlock_detected.set()

        t = threading.Thread(target=tx1_wait)
        t.start()
        time.sleep(0.15)

        try:
            self.lm.acquire(2, a, LockMode.X, auto_intention=False, timeout=2.0)
        except DeadlockError:
            deadlock_detected.set()

        t.join(timeout=3.0)
        assert deadlock_detected.is_set()

    def test_no_deadlock_compatible(self):
        """Two shared locks on same resource: no deadlock."""
        a = make_table("db", "a")
        self.lm.acquire(1, a, LockMode.S, auto_intention=False)
        result = self.lm.acquire(2, a, LockMode.S, auto_intention=False)
        assert result == LockResult.GRANTED

    def test_deadlock_stats(self):
        a = make_table("db", "a")
        b = make_table("db", "b")
        self.lm.acquire(1, a, LockMode.X, auto_intention=False)
        self.lm.acquire(2, b, LockMode.X, auto_intention=False)

        def tx1_wait():
            try:
                self.lm.acquire(1, b, LockMode.X, auto_intention=False, timeout=2.0)
            except DeadlockError:
                pass

        t = threading.Thread(target=tx1_wait)
        t.start()
        time.sleep(0.15)
        try:
            self.lm.acquire(2, a, LockMode.X, auto_intention=False, timeout=2.0)
        except DeadlockError:
            pass
        t.join(timeout=3.0)
        assert self.lm.stats.deadlocks > 0


# ============================================================
# Timeouts
# ============================================================

class TestTimeouts:
    def setup_method(self):
        self.lm = LockManager(default_timeout=0.5)

    def test_timeout(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.X, auto_intention=False)
        with pytest.raises(LockTimeoutError):
            self.lm.acquire(2, res, LockMode.X, auto_intention=False, timeout=0.2)

    def test_try_acquire_denied(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.X, auto_intention=False)
        result = self.lm.try_acquire(2, res, LockMode.X, auto_intention=False)
        assert result == LockResult.DENIED

    def test_timeout_stats(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.X, auto_intention=False)
        try:
            self.lm.acquire(2, res, LockMode.X, auto_intention=False, timeout=0.1)
        except LockTimeoutError:
            pass
        assert self.lm.stats.timeouts >= 1


# ============================================================
# Lock Escalation
# ============================================================

class TestLockEscalation:
    def setup_method(self):
        self.lm = LockManager(escalation_threshold=5, default_timeout=1.0)

    def test_check_escalation_below(self):
        for i in range(4):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        assert not self.lm.check_escalation(1, "db", "t")

    def test_check_escalation_at_threshold(self):
        for i in range(5):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        assert self.lm.check_escalation(1, "db", "t")

    def test_escalate(self):
        for i in range(5):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        result = self.lm.escalate(1, "db", "t")
        assert result in (LockResult.GRANTED, LockResult.UPGRADED)
        # Row locks should be gone, table lock should exist
        assert self.lm.get_lock_mode(1, make_table("db", "t")) == LockMode.X
        for i in range(5):
            assert self.lm.get_lock_mode(1, make_row("db", "t", 0, i)) is None

    def test_auto_escalate_needed(self):
        for i in range(5):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        result = self.lm.auto_escalate(1, "db", "t")
        assert result is not None

    def test_auto_escalate_not_needed(self):
        self.lm.acquire(1, make_row("db", "t", 0, 0), LockMode.X, auto_intention=False)
        result = self.lm.auto_escalate(1, "db", "t")
        assert result is None

    def test_escalation_stats(self):
        for i in range(5):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        self.lm.escalate(1, "db", "t")
        assert self.lm.stats.escalations == 1

    def test_escalate_page_locks(self):
        for i in range(5):
            self.lm.acquire(1, make_page("db", "t", i), LockMode.X, auto_intention=False)
        result = self.lm.escalate(1, "db", "t")
        assert result in (LockResult.GRANTED, LockResult.UPGRADED)
        assert self.lm.get_lock_mode(1, make_table("db", "t")) == LockMode.X


# ============================================================
# Transaction Lock Tracking
# ============================================================

class TestTxLockTracking:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_get_tx_locks(self):
        r1 = make_table("db", "t1")
        r2 = make_table("db", "t2")
        self.lm.acquire(1, r1, LockMode.S, auto_intention=False)
        self.lm.acquire(1, r2, LockMode.X, auto_intention=False)
        locks = self.lm.get_tx_locks(1)
        assert len(locks) == 2
        assert locks[r1.key] == LockMode.S
        assert locks[r2.key] == LockMode.X

    def test_get_tx_locks_empty(self):
        assert self.lm.get_tx_locks(99) == {}

    def test_release_all_clears_tracking(self):
        self.lm.acquire(1, make_table("db", "t1"), LockMode.S, auto_intention=False)
        self.lm.release_all(1)
        assert self.lm.get_tx_locks(1) == {}

    def test_get_waiter_count(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.X, auto_intention=False)
        assert self.lm.get_waiter_count(res) == 0


# ============================================================
# Two-Phase Locking
# ============================================================

class TestTwoPhaseLocking:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)
        self.tpl = TwoPhaseLockHelper(self.lm)

    def test_growing_phase(self):
        self.tpl.begin(1)
        assert self.tpl.phase(1) == 'growing'
        result = self.tpl.acquire(1, make_table("db", "t"), LockMode.S)
        assert result == LockResult.GRANTED

    def test_shrinking_phase_blocks_acquire(self):
        self.tpl.begin(1)
        res = make_table("db", "t")
        self.tpl.acquire(1, res, LockMode.S)
        self.tpl.release(1, res)
        assert self.tpl.phase(1) == 'shrinking'
        with pytest.raises(LockError, match="shrinking"):
            self.tpl.acquire(1, make_table("db", "t2"), LockMode.S)

    def test_release_all_resets(self):
        self.tpl.begin(1)
        self.tpl.acquire(1, make_table("db", "t"), LockMode.S)
        self.tpl.release_all(1)
        assert self.tpl.phase(1) == 'unknown'

    def test_strict_2pl_hold_until_commit(self):
        """In strict 2PL, all locks held until release_all (commit)."""
        self.tpl.begin(1)
        r1 = make_table("db", "t1")
        r2 = make_table("db", "t2")
        self.tpl.acquire(1, r1, LockMode.S)
        self.tpl.acquire(1, r2, LockMode.X)
        # Don't release individually -- release all at commit
        # auto_intention acquires IS/IX on db too, so count >= 2
        count = self.tpl.release_all(1)
        assert count >= 2


# ============================================================
# Multi-Granularity Locker
# ============================================================

class TestMultiGranularityLocker:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)
        self.mgl = MultiGranularityLocker(self.lm, db="mydb")

    def test_lock_table_shared(self):
        result = self.mgl.lock_table_shared(1, "users")
        assert result == LockResult.GRANTED

    def test_lock_table_exclusive(self):
        result = self.mgl.lock_table_exclusive(1, "users")
        assert result == LockResult.GRANTED

    def test_lock_row_shared(self):
        result = self.mgl.lock_row_shared(1, "users", 0, 1)
        assert result == LockResult.GRANTED
        # Check intention locks
        assert self.lm.get_lock_mode(1, make_table("mydb", "users")) == LockMode.IS

    def test_lock_row_exclusive(self):
        result = self.mgl.lock_row_exclusive(1, "users", 0, 1)
        assert result == LockResult.GRANTED
        assert self.lm.get_lock_mode(1, make_table("mydb", "users")) == LockMode.IX

    def test_lock_page(self):
        result = self.mgl.lock_page_shared(1, "users", 0)
        assert result == LockResult.GRANTED

    def test_unlock_row(self):
        self.mgl.lock_row_shared(1, "users", 0, 1)
        assert self.mgl.unlock_row(1, "users", 0, 1)

    def test_unlock_table(self):
        self.mgl.lock_table_shared(1, "users")
        assert self.mgl.unlock_table(1, "users")

    def test_unlock_all(self):
        self.mgl.lock_row_shared(1, "users", 0, 1)
        self.mgl.lock_row_exclusive(1, "users", 0, 2)
        count = self.mgl.unlock_all(1)
        assert count > 0

    def test_two_readers_different_rows(self):
        self.mgl.lock_row_shared(1, "users", 0, 1)
        result = self.mgl.lock_row_shared(2, "users", 0, 2)
        assert result == LockResult.GRANTED

    def test_two_writers_different_rows(self):
        self.mgl.lock_row_exclusive(1, "users", 0, 1)
        result = self.mgl.lock_row_exclusive(2, "users", 0, 2)
        assert result == LockResult.GRANTED

    def test_reader_writer_same_row_blocked(self):
        self.mgl.lock_row_exclusive(1, "users", 0, 1)
        result = self.lm.try_acquire(2, make_row("mydb", "users", 0, 1), LockMode.S)
        assert result == LockResult.DENIED


# ============================================================
# Concurrent Access
# ============================================================

class TestConcurrentAccess:
    def setup_method(self):
        self.lm = LockManager(default_timeout=3.0)

    def test_concurrent_shared_readers(self):
        """Multiple threads can acquire shared locks simultaneously."""
        res = make_table("db", "t")
        results = []
        barrier = threading.Barrier(5)

        def reader(tx_id):
            barrier.wait()
            r = self.lm.acquire(tx_id, res, LockMode.S, auto_intention=False)
            results.append(r)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert all(r == LockResult.GRANTED for r in results)

    def test_exclusive_serialization(self):
        """Exclusive locks serialize access."""
        res = make_table("db", "t")
        order = []
        lock = threading.Lock()

        def writer(tx_id):
            self.lm.acquire(tx_id, res, LockMode.X, auto_intention=False)
            with lock:
                order.append(tx_id)
            time.sleep(0.05)
            self.lm.release(tx_id, res)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.02)
        for t in threads:
            t.join(timeout=5.0)
        assert len(order) == 3

    def test_reader_writer_interleave(self):
        """Readers and writers interleave correctly."""
        res = make_table("db", "t")
        events = []

        def reader(tx_id):
            self.lm.acquire(tx_id, res, LockMode.S, auto_intention=False)
            events.append(('read', tx_id))
            time.sleep(0.05)
            self.lm.release(tx_id, res)

        def writer(tx_id):
            self.lm.acquire(tx_id, res, LockMode.X, auto_intention=False)
            events.append(('write', tx_id))
            time.sleep(0.05)
            self.lm.release(tx_id, res)

        # Start reader, then writer, then reader
        t1 = threading.Thread(target=reader, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t3 = threading.Thread(target=reader, args=(3,))
        t1.start()
        time.sleep(0.02)
        t2.start()
        time.sleep(0.02)
        t3.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        t3.join(timeout=5.0)
        assert len(events) == 3


# ============================================================
# Stats
# ============================================================

class TestLockStats:
    def test_initial(self):
        s = LockStats()
        assert s.grants == 0
        assert s.summary()['grants'] == 0

    def test_grants_counted(self):
        lm = LockManager()
        lm.acquire(1, make_table("db", "t"), LockMode.S, auto_intention=False)
        assert lm.stats.grants == 1

    def test_releases_counted(self):
        lm = LockManager()
        res = make_table("db", "t")
        lm.acquire(1, res, LockMode.S, auto_intention=False)
        lm.release(1, res)
        assert lm.stats.releases == 1

    def test_summary_keys(self):
        s = LockStats()
        keys = s.summary().keys()
        assert 'grants' in keys
        assert 'deadlocks' in keys
        assert 'escalations' in keys


# ============================================================
# Analyzer
# ============================================================

class TestLockManagerAnalyzer:
    def setup_method(self):
        self.lm = LockManager(escalation_threshold=5, default_timeout=1.0)
        self.analyzer = LockManagerAnalyzer(self.lm)

    def test_contention_report_empty(self):
        report = self.analyzer.contention_report()
        assert report['total_locks'] == 0
        assert report['hot_resources'] == []

    def test_contention_report_shared(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.S, auto_intention=False)
        self.lm.acquire(2, res, LockMode.S, auto_intention=False)
        report = self.analyzer.contention_report()
        assert report['total_locks'] == 2
        assert len(report['hot_resources']) == 1

    def test_tx_report(self):
        self.lm.acquire(1, make_table("db", "t1"), LockMode.S, auto_intention=False)
        self.lm.acquire(1, make_table("db", "t2"), LockMode.X, auto_intention=False)
        report = self.analyzer.tx_report(1)
        assert report['tx_id'] == 1
        assert report['total_locks'] == 2

    def test_deadlock_report_clean(self):
        report = self.analyzer.deadlock_report()
        assert not report['cycle_detected']
        assert report['cycle'] == []

    def test_escalation_report(self):
        for i in range(5):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        report = self.analyzer.escalation_report()
        assert report['escalation_threshold'] == 5

    def test_escalation_candidates(self):
        for i in range(3):
            self.lm.acquire(1, make_row("db", "t", 0, i), LockMode.X, auto_intention=False)
        report = self.analyzer.escalation_report()
        # 3 >= 5//2 (2), so should appear as candidate
        assert len(report['candidates']) >= 1


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def setup_method(self):
        self.lm = LockManager(default_timeout=1.0)

    def test_release_already_released(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.S, auto_intention=False)
        self.lm.release(1, res)
        assert not self.lm.release(1, res)

    def test_get_all_locks_empty(self):
        assert self.lm.get_all_locks() == {}

    def test_get_all_waiters_empty(self):
        assert self.lm.get_all_waiters() == {}

    def test_multiple_resources(self):
        for i in range(10):
            self.lm.acquire(1, make_table("db", f"t{i}"), LockMode.S, auto_intention=False)
        assert self.lm.lock_count() == 10

    def test_release_all_no_locks(self):
        assert self.lm.release_all(99) == 0

    def test_is_locked_after_all_release(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.S, auto_intention=False)
        self.lm.acquire(2, res, LockMode.S, auto_intention=False)
        self.lm.release(1, res)
        assert self.lm.is_locked(res)  # tx2 still holds
        self.lm.release(2, res)
        assert not self.lm.is_locked(res)

    def test_downgrade_same_level(self):
        res = make_table("db", "t")
        self.lm.acquire(1, res, LockMode.S, auto_intention=False)
        assert not self.lm.downgrade(1, res, LockMode.S)  # Same level, not a downgrade

    def test_get_lock_mode_no_resource(self):
        assert self.lm.get_lock_mode(1, make_table("db", "nonexistent")) is None

    def test_get_holders_no_resource(self):
        assert self.lm.get_holders(make_table("db", "nonexistent")) == {}

    def test_detect_deadlock_no_waiters(self):
        assert self.lm.detect_deadlock() == []

    def test_get_wait_graph_empty(self):
        assert self.lm.get_wait_graph() == {}
