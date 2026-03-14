"""
Tests for C240: Multi-Version Concurrency Control (MVCC)

Covers:
- Basic CRUD operations
- Transaction lifecycle (begin, commit, abort)
- Snapshot isolation
- Read-committed isolation
- Serializable isolation (SSI)
- Write-write conflict detection
- Version chain management
- Savepoints and partial rollback
- Garbage collection
- Secondary indexes
- Range queries
- Concurrent transaction scenarios
- Edge cases
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mvcc import (
    MVCCEngine, IsolationLevel, TxnStatus, Version, Snapshot,
    WriteConflictError, SerializationError, MVCCIndex
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def engine():
    return MVCCEngine()


@pytest.fixture
def si_engine():
    """Engine with a pre-populated dataset."""
    e = MVCCEngine()
    t = e.begin()
    e.put(t, "a", 1)
    e.put(t, "b", 2)
    e.put(t, "c", 3)
    e.commit(t)
    return e


# ===========================================================================
# Basic CRUD
# ===========================================================================

class TestBasicCRUD:
    def test_put_and_get(self, engine):
        t = engine.begin()
        engine.put(t, "key1", "value1")
        assert engine.get(t, "key1") == "value1"
        engine.commit(t)

    def test_get_nonexistent(self, engine):
        t = engine.begin()
        assert engine.get(t, "missing") is None
        engine.commit(t)

    def test_put_overwrite(self, engine):
        t = engine.begin()
        engine.put(t, "key", "v1")
        engine.put(t, "key", "v2")
        assert engine.get(t, "key") == "v2"
        engine.commit(t)

    def test_delete_existing(self, si_engine):
        t = si_engine.begin()
        assert si_engine.delete(t, "a") is True
        assert si_engine.get(t, "a") is None
        si_engine.commit(t)

    def test_delete_nonexistent(self, engine):
        t = engine.begin()
        assert engine.delete(t, "missing") is False
        engine.commit(t)

    def test_exists(self, si_engine):
        t = si_engine.begin()
        assert si_engine.exists(t, "a") is True
        assert si_engine.exists(t, "z") is False
        si_engine.commit(t)

    def test_count(self, si_engine):
        t = si_engine.begin()
        assert si_engine.count(t) == 3
        si_engine.commit(t)

    def test_count_with_prefix(self, engine):
        t = engine.begin()
        engine.put(t, "user:1", "alice")
        engine.put(t, "user:2", "bob")
        engine.put(t, "item:1", "widget")
        assert engine.count(t, "user:") == 2
        assert engine.count(t, "item:") == 1
        engine.commit(t)

    def test_multiple_keys(self, engine):
        t = engine.begin()
        for i in range(100):
            engine.put(t, f"key{i}", i)
        for i in range(100):
            assert engine.get(t, f"key{i}") == i
        engine.commit(t)

    def test_various_value_types(self, engine):
        t = engine.begin()
        engine.put(t, "int", 42)
        engine.put(t, "str", "hello")
        engine.put(t, "list", [1, 2, 3])
        engine.put(t, "dict", {"a": 1})
        engine.put(t, "none", None)
        engine.put(t, "bool", True)
        assert engine.get(t, "int") == 42
        assert engine.get(t, "str") == "hello"
        assert engine.get(t, "list") == [1, 2, 3]
        assert engine.get(t, "dict") == {"a": 1}
        assert engine.get(t, "none") is None  # Same as nonexistent
        assert engine.get(t, "bool") is True
        engine.commit(t)


# ===========================================================================
# Transaction Lifecycle
# ===========================================================================

class TestTransactionLifecycle:
    def test_begin_returns_unique_ids(self, engine):
        t1 = engine.begin()
        t2 = engine.begin()
        t3 = engine.begin()
        assert t1 != t2 != t3
        assert t1 < t2 < t3
        engine.abort(t1)
        engine.abort(t2)
        engine.abort(t3)

    def test_commit_success(self, engine):
        t = engine.begin()
        engine.put(t, "x", 1)
        assert engine.commit(t) is True

    def test_commit_makes_visible(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        assert engine.get(t2, "x") == 1
        engine.commit(t2)

    def test_abort_rolls_back(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        engine.put(t2, "x", 2)
        engine.abort(t2)

        t3 = engine.begin()
        assert engine.get(t3, "x") == 1  # Original value
        engine.commit(t3)

    def test_abort_new_key(self, engine):
        t = engine.begin()
        engine.put(t, "ephemeral", 42)
        engine.abort(t)

        t2 = engine.begin()
        assert engine.get(t2, "ephemeral") is None
        engine.commit(t2)

    def test_double_commit_fails(self, engine):
        t = engine.begin()
        engine.commit(t)
        with pytest.raises(ValueError, match="already committed"):
            engine.commit(t)

    def test_double_abort_fails(self, engine):
        t = engine.begin()
        engine.abort(t)
        with pytest.raises(ValueError, match="already aborted"):
            engine.abort(t)

    def test_commit_then_abort_fails(self, engine):
        t = engine.begin()
        engine.commit(t)
        with pytest.raises(ValueError, match="already committed"):
            engine.abort(t)

    def test_nonexistent_txn(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.get(999, "x")

    def test_active_transactions(self, engine):
        assert engine.active_transactions() == []
        t1 = engine.begin()
        t2 = engine.begin()
        assert engine.active_transactions() == [t1, t2]
        engine.commit(t1)
        assert engine.active_transactions() == [t2]
        engine.abort(t2)
        assert engine.active_transactions() == []


# ===========================================================================
# Snapshot Isolation (Repeatable Read)
# ===========================================================================

class TestSnapshotIsolation:
    def test_read_own_writes(self, engine):
        t = engine.begin()
        engine.put(t, "x", 1)
        assert engine.get(t, "x") == 1
        engine.commit(t)

    def test_isolation_from_concurrent_writes(self, si_engine):
        """T2 should not see T1's uncommitted writes."""
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        si_engine.put(t1, "a", 100)
        assert si_engine.get(t2, "a") == 1  # Still sees old value
        si_engine.commit(t1)
        # Even after T1 commits, T2's snapshot doesn't include it (repeatable read)
        assert si_engine.get(t2, "a") == 1
        si_engine.commit(t2)

    def test_new_txn_sees_committed(self, si_engine):
        t1 = si_engine.begin()
        si_engine.put(t1, "a", 100)
        si_engine.commit(t1)

        t2 = si_engine.begin()
        assert si_engine.get(t2, "a") == 100
        si_engine.commit(t2)

    def test_phantom_reads_prevented(self, engine):
        """Repeatable read prevents phantom reads."""
        t1 = engine.begin()
        engine.put(t1, "item:1", "a")
        engine.put(t1, "item:2", "b")
        engine.commit(t1)

        t2 = engine.begin()
        items = engine.scan(t2, "item:")
        assert len(items) == 2

        # Concurrent txn adds a new item
        t3 = engine.begin()
        engine.put(t3, "item:3", "c")
        engine.commit(t3)

        # T2 still sees only 2 items
        items2 = engine.scan(t2, "item:")
        assert len(items2) == 2
        engine.commit(t2)

    def test_repeatable_read_consistency(self, si_engine):
        """Same read returns same value throughout transaction."""
        t1 = si_engine.begin()
        v1 = si_engine.get(t1, "a")

        # Concurrent update
        t2 = si_engine.begin()
        si_engine.put(t2, "a", 999)
        si_engine.commit(t2)

        v2 = si_engine.get(t1, "a")
        assert v1 == v2  # Same value both times
        si_engine.commit(t1)

    def test_snapshot_sees_pre_begin_commits_only(self, engine):
        """Snapshot should only see transactions committed before begin."""
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()  # Takes snapshot here

        t3 = engine.begin()
        engine.put(t3, "y", 2)
        engine.commit(t3)

        assert engine.get(t2, "x") == 1   # Committed before t2 began
        assert engine.get(t2, "y") is None  # Committed after t2 began
        engine.commit(t2)


# ===========================================================================
# Read Committed Isolation
# ===========================================================================

class TestReadCommitted:
    def test_sees_committed_changes(self, engine):
        t1 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin(IsolationLevel.READ_COMMITTED)

        t3 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t3, "x", 2)
        engine.commit(t3)

        # Read committed should see the latest committed value
        assert engine.get(t2, "x") == 2
        engine.commit(t2)

    def test_does_not_see_uncommitted(self, engine):
        t1 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin(IsolationLevel.READ_COMMITTED)
        t3 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t3, "x", 2)
        # T3 not yet committed

        assert engine.get(t2, "x") == 1  # Still sees committed value
        engine.commit(t3)
        assert engine.get(t2, "x") == 2  # Now sees T3's committed value
        engine.commit(t2)

    def test_non_repeatable_reads(self, engine):
        """Read committed allows non-repeatable reads (by design)."""
        t1 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin(IsolationLevel.READ_COMMITTED)
        v1 = engine.get(t2, "x")
        assert v1 == 1

        t3 = engine.begin(IsolationLevel.READ_COMMITTED)
        engine.put(t3, "x", 2)
        engine.commit(t3)

        v2 = engine.get(t2, "x")
        assert v2 == 2  # Different from v1! Non-repeatable read
        engine.commit(t2)


# ===========================================================================
# Serializable Isolation (SSI)
# ===========================================================================

class TestSerializable:
    def test_basic_serializable(self, engine):
        """Serializable transactions that don't conflict should succeed."""
        t1 = engine.begin(IsolationLevel.SERIALIZABLE)
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin(IsolationLevel.SERIALIZABLE)
        assert engine.get(t2, "x") == 1
        engine.put(t2, "y", 2)
        engine.commit(t2)

    def test_write_skew_detected(self, engine):
        """SSI should detect write skew anomalies."""
        # Setup: two accounts with constraint sum >= 0
        t0 = engine.begin()
        engine.put(t0, "acct_a", 100)
        engine.put(t0, "acct_b", 100)
        engine.commit(t0)

        # T1 reads both, writes to a
        t1 = engine.begin(IsolationLevel.SERIALIZABLE)
        a1 = engine.get(t1, "acct_a")
        b1 = engine.get(t1, "acct_b")

        # T2 reads both, writes to b (concurrent)
        t2 = engine.begin(IsolationLevel.SERIALIZABLE)
        a2 = engine.get(t2, "acct_a")
        b2 = engine.get(t2, "acct_b")

        engine.put(t1, "acct_a", a1 - 150)  # -50 in a
        engine.commit(t1)

        engine.put(t2, "acct_b", b2 - 150)  # Would make both negative
        with pytest.raises(SerializationError):
            engine.commit(t2)

    def test_serializable_no_conflict(self, engine):
        """Non-overlapping reads/writes should pass SSI."""
        t0 = engine.begin()
        engine.put(t0, "x", 1)
        engine.put(t0, "y", 2)
        engine.commit(t0)

        t1 = engine.begin(IsolationLevel.SERIALIZABLE)
        engine.get(t1, "x")
        engine.put(t1, "y", 20)

        t2 = engine.begin(IsolationLevel.SERIALIZABLE)
        engine.get(t2, "y")
        engine.put(t2, "x", 10)

        engine.commit(t1)  # T1 commits first
        # T2 read y (which T1 wrote) -- conflict!
        with pytest.raises(SerializationError):
            engine.commit(t2)


# ===========================================================================
# Write-Write Conflicts
# ===========================================================================

class TestWriteConflicts:
    def test_concurrent_write_same_key(self, si_engine):
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        si_engine.put(t1, "a", 10)
        with pytest.raises(WriteConflictError):
            si_engine.put(t2, "a", 20)

        si_engine.commit(t1)
        si_engine.abort(t2)

    def test_write_after_commit_ok(self, si_engine):
        t1 = si_engine.begin()
        si_engine.put(t1, "a", 10)
        si_engine.commit(t1)

        t2 = si_engine.begin()
        si_engine.put(t2, "a", 20)  # No conflict -- t1 already committed
        si_engine.commit(t2)

    def test_write_after_abort_ok(self, si_engine):
        t1 = si_engine.begin()
        si_engine.put(t1, "a", 10)
        si_engine.abort(t1)

        t2 = si_engine.begin()
        si_engine.put(t2, "a", 20)  # No conflict -- t1 aborted
        si_engine.commit(t2)

    def test_concurrent_delete_conflict(self, si_engine):
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        si_engine.delete(t1, "a")
        with pytest.raises(WriteConflictError):
            si_engine.delete(t2, "a")

        si_engine.commit(t1)
        si_engine.abort(t2)


# ===========================================================================
# Version Chains
# ===========================================================================

class TestVersionChains:
    def test_version_count_increases(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)
        assert engine.version_count("x") == 1

        t2 = engine.begin()
        engine.put(t2, "x", 2)
        engine.commit(t2)
        assert engine.version_count("x") == 2

        t3 = engine.begin()
        engine.put(t3, "x", 3)
        engine.commit(t3)
        assert engine.version_count("x") == 3

    def test_abort_removes_version(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        engine.put(t2, "x", 2)
        engine.abort(t2)

        assert engine.version_count("x") == 1

    def test_version_chain_after_multiple_updates(self, engine):
        for i in range(10):
            t = engine.begin()
            engine.put(t, "counter", i)
            engine.commit(t)

        assert engine.version_count("counter") == 10

        t = engine.begin()
        assert engine.get(t, "counter") == 9
        engine.commit(t)


# ===========================================================================
# Savepoints
# ===========================================================================

class TestSavepoints:
    def test_basic_savepoint(self, engine):
        t = engine.begin()
        engine.put(t, "x", 1)
        engine.savepoint(t, "sp1")
        engine.put(t, "x", 2)
        engine.put(t, "y", 10)

        assert engine.get(t, "x") == 2
        assert engine.get(t, "y") == 10

        engine.rollback_to_savepoint(t, "sp1")
        assert engine.get(t, "x") == 1
        assert engine.get(t, "y") is None
        engine.commit(t)

    def test_nested_savepoints(self, engine):
        t = engine.begin()
        engine.put(t, "x", 1)
        engine.savepoint(t, "sp1")

        engine.put(t, "x", 2)
        engine.savepoint(t, "sp2")

        engine.put(t, "x", 3)
        assert engine.get(t, "x") == 3

        engine.rollback_to_savepoint(t, "sp2")
        assert engine.get(t, "x") == 2

        engine.rollback_to_savepoint(t, "sp1")
        assert engine.get(t, "x") == 1
        engine.commit(t)

    def test_savepoint_not_found(self, engine):
        t = engine.begin()
        with pytest.raises(ValueError, match="not found"):
            engine.rollback_to_savepoint(t, "nonexistent")
        engine.abort(t)

    def test_savepoint_preserves_earlier_writes(self, engine):
        t = engine.begin()
        engine.put(t, "a", 1)
        engine.put(t, "b", 2)
        engine.savepoint(t, "sp")
        engine.put(t, "c", 3)
        engine.delete(t, "a")

        engine.rollback_to_savepoint(t, "sp")
        assert engine.get(t, "a") == 1  # Restored
        assert engine.get(t, "b") == 2  # Preserved
        assert engine.get(t, "c") is None  # Rolled back
        engine.commit(t)

    def test_savepoint_with_new_key_rollback(self, engine):
        t = engine.begin()
        engine.savepoint(t, "sp")
        engine.put(t, "new_key", 42)
        assert engine.get(t, "new_key") == 42

        engine.rollback_to_savepoint(t, "sp")
        assert engine.get(t, "new_key") is None
        engine.commit(t)

        # Verify it's really gone
        t2 = engine.begin()
        assert engine.get(t2, "new_key") is None
        engine.commit(t2)


# ===========================================================================
# Garbage Collection
# ===========================================================================

class TestGarbageCollection:
    def test_gc_no_active_txns(self, engine):
        # Create multiple versions
        for i in range(5):
            t = engine.begin()
            engine.put(t, "x", i)
            engine.commit(t)

        assert engine.version_count("x") == 5
        collected = engine.gc()
        assert collected > 0
        # After GC with no active txns, only latest version should remain
        assert engine.version_count("x") == 1

        # Value is still correct
        t = engine.begin()
        assert engine.get(t, "x") == 4
        engine.commit(t)

    def test_gc_preserves_active_txn_versions(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()  # Takes snapshot seeing x=1
        engine.put(t2, "x", 2)
        engine.commit(t2)

        t3 = engine.begin()  # Active, needs to see x=2

        # GC should not collect version visible to t3
        engine.gc()
        assert engine.get(t3, "x") == 2
        engine.commit(t3)

    def test_gc_deleted_keys(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        engine.delete(t2, "x")
        engine.commit(t2)

        engine.gc()
        assert engine.version_count("x") == 0

    def test_gc_aborted_txns(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        engine.put(t2, "x", 2)
        engine.abort(t2)

        # After abort, aborted version is already removed
        assert engine.version_count("x") == 1

        t3 = engine.begin()
        assert engine.get(t3, "x") == 1
        engine.commit(t3)

    def test_gc_returns_count(self, engine):
        for i in range(3):
            t = engine.begin()
            engine.put(t, "x", i)
            engine.commit(t)

        collected = engine.gc()
        assert collected >= 2  # At least 2 old versions collected

    def test_gc_multiple_keys(self, engine):
        for i in range(5):
            t = engine.begin()
            engine.put(t, "a", i)
            engine.put(t, "b", i * 10)
            engine.commit(t)

        engine.gc()
        assert engine.version_count("a") == 1
        assert engine.version_count("b") == 1


# ===========================================================================
# Scan Operations
# ===========================================================================

class TestScanOperations:
    def test_scan_all(self, si_engine):
        t = si_engine.begin()
        result = si_engine.scan(t)
        assert result == {"a": 1, "b": 2, "c": 3}
        si_engine.commit(t)

    def test_scan_prefix(self, engine):
        t = engine.begin()
        engine.put(t, "user:1", "alice")
        engine.put(t, "user:2", "bob")
        engine.put(t, "item:1", "widget")
        engine.commit(t)

        t2 = engine.begin()
        users = engine.scan(t2, "user:")
        assert users == {"user:1": "alice", "user:2": "bob"}
        engine.commit(t2)

    def test_range_scan(self, engine):
        t = engine.begin()
        for c in "abcdefghij":
            engine.put(t, c, ord(c))
        engine.commit(t)

        t2 = engine.begin()
        result = engine.range_scan(t2, "c", "f")
        assert set(result.keys()) == {"c", "d", "e", "f"}
        engine.commit(t2)

    def test_scan_isolation(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.put(t1, "y", 2)
        engine.commit(t1)

        t2 = engine.begin()
        result1 = engine.scan(t2)
        assert len(result1) == 2

        # Concurrent add
        t3 = engine.begin()
        engine.put(t3, "z", 3)
        engine.commit(t3)

        result2 = engine.scan(t2)
        assert len(result2) == 2  # Repeatable read: still 2
        engine.commit(t2)

    def test_scan_empty(self, engine):
        t = engine.begin()
        assert engine.scan(t) == {}
        engine.commit(t)

    def test_range_scan_empty_range(self, engine):
        t = engine.begin()
        engine.put(t, "a", 1)
        engine.put(t, "z", 2)
        engine.commit(t)

        t2 = engine.begin()
        result = engine.range_scan(t2, "m", "n")
        assert result == {}
        engine.commit(t2)


# ===========================================================================
# Secondary Indexes
# ===========================================================================

class TestSecondaryIndexes:
    def test_create_index(self, engine):
        t = engine.begin()
        engine.put(t, "user:1", {"name": "alice", "age": 30})
        engine.put(t, "user:2", {"name": "bob", "age": 25})
        engine.commit(t)

        idx = engine.create_index("by_age", lambda v: v["age"])
        assert isinstance(idx, MVCCIndex)

    def test_index_lookup(self, engine):
        t = engine.begin()
        engine.put(t, "user:1", {"name": "alice", "age": 30})
        engine.put(t, "user:2", {"name": "bob", "age": 25})
        engine.put(t, "user:3", {"name": "carol", "age": 30})
        engine.commit(t)

        engine.create_index("by_age", lambda v: v["age"])

        t2 = engine.begin()
        result = engine.index_lookup(t2, "by_age", 30)
        assert len(result) == 2
        names = {v["name"] for v in result.values()}
        assert names == {"alice", "carol"}
        engine.commit(t2)

    def test_index_updated_on_write(self, engine):
        engine.create_index("by_age", lambda v: v["age"])

        t = engine.begin()
        engine.put(t, "user:1", {"name": "alice", "age": 30})
        engine.commit(t)

        t2 = engine.begin()
        result = engine.index_lookup(t2, "by_age", 30)
        assert "user:1" in result

        engine.put(t2, "user:1", {"name": "alice", "age": 31})
        engine.commit(t2)

        t3 = engine.begin()
        result30 = engine.index_lookup(t3, "by_age", 30)
        result31 = engine.index_lookup(t3, "by_age", 31)
        assert "user:1" not in result30
        assert "user:1" in result31
        engine.commit(t3)

    def test_index_not_found(self, engine):
        t = engine.begin()
        with pytest.raises(ValueError, match="not found"):
            engine.index_lookup(t, "nonexistent", 42)
        engine.abort(t)


# ===========================================================================
# Concurrent Transaction Scenarios
# ===========================================================================

class TestConcurrentScenarios:
    def test_lost_update_prevented(self, si_engine):
        """Two txns reading then writing same key -- second should conflict."""
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        v1 = si_engine.get(t1, "a")
        v2 = si_engine.get(t2, "a")

        si_engine.put(t1, "a", v1 + 10)  # a = 11
        si_engine.commit(t1)

        with pytest.raises(WriteConflictError):
            si_engine.put(t2, "a", v2 + 20)

        si_engine.abort(t2)

    def test_read_only_txn_no_conflict(self, si_engine):
        """Read-only transactions should never conflict."""
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        si_engine.get(t1, "a")
        si_engine.get(t2, "a")

        si_engine.commit(t1)
        si_engine.commit(t2)

    def test_disjoint_writes_no_conflict(self, si_engine):
        """Concurrent writes to different keys should succeed."""
        t1 = si_engine.begin()
        t2 = si_engine.begin()

        si_engine.put(t1, "a", 10)
        si_engine.put(t2, "b", 20)

        si_engine.commit(t1)
        si_engine.commit(t2)

        t3 = si_engine.begin()
        assert si_engine.get(t3, "a") == 10
        assert si_engine.get(t3, "b") == 20
        si_engine.commit(t3)

    def test_long_running_txn(self, engine):
        """Long-running txn should see consistent snapshot."""
        t0 = engine.begin()
        engine.put(t0, "x", 0)
        engine.commit(t0)

        long_txn = engine.begin()
        assert engine.get(long_txn, "x") == 0

        # Many concurrent updates
        for i in range(1, 11):
            t = engine.begin()
            engine.put(t, "x", i)
            engine.commit(t)

        # Long txn still sees original
        assert engine.get(long_txn, "x") == 0
        engine.commit(long_txn)

    def test_three_way_conflict(self, engine):
        """Three txns trying to write same key."""
        t0 = engine.begin()
        engine.put(t0, "x", 0)
        engine.commit(t0)

        t1 = engine.begin()
        t2 = engine.begin()
        t3 = engine.begin()

        engine.put(t1, "x", 1)  # t1 gets the lock

        with pytest.raises(WriteConflictError):
            engine.put(t2, "x", 2)

        with pytest.raises(WriteConflictError):
            engine.put(t3, "x", 3)

        engine.commit(t1)
        engine.abort(t2)
        engine.abort(t3)

    def test_delete_then_insert(self, si_engine):
        """Delete a key, then re-insert it in same txn."""
        t = si_engine.begin()
        si_engine.delete(t, "a")
        assert si_engine.get(t, "a") is None

        si_engine.put(t, "a", 999)
        assert si_engine.get(t, "a") == 999
        si_engine.commit(t)

        t2 = si_engine.begin()
        assert si_engine.get(t2, "a") == 999
        si_engine.commit(t2)

    def test_interleaved_read_write(self, engine):
        """Interleaved reads and writes across transactions."""
        t0 = engine.begin()
        engine.put(t0, "x", 0)
        engine.put(t0, "y", 0)
        engine.commit(t0)

        t1 = engine.begin()
        t2 = engine.begin()

        # T1 reads x, writes y
        engine.get(t1, "x")
        engine.put(t1, "y", 1)

        # T2 reads y (sees 0, not T1's uncommitted 1)
        v = engine.get(t2, "y")
        assert v == 0

        engine.put(t2, "x", 2)

        engine.commit(t1)
        engine.commit(t2)  # No conflict -- disjoint writes


# ===========================================================================
# Stats and Diagnostics
# ===========================================================================

class TestStats:
    def test_stats_empty(self, engine):
        s = engine.stats()
        assert s['total_keys'] == 0
        assert s['total_versions'] == 0
        assert s['active_txns'] == 0
        assert s['committed_txns'] == 0
        assert s['aborted_txns'] == 0
        assert s['indexes'] == 0

    def test_stats_populated(self, si_engine):
        s = si_engine.stats()
        assert s['total_keys'] == 3
        assert s['total_versions'] == 3
        assert s['committed_txns'] >= 1

    def test_stats_after_operations(self, engine):
        t1 = engine.begin()
        engine.put(t1, "x", 1)
        engine.commit(t1)

        t2 = engine.begin()
        engine.put(t2, "x", 2)
        engine.abort(t2)

        s = engine.stats()
        assert s['total_keys'] == 1
        assert s['committed_txns'] == 1
        assert s['aborted_txns'] == 1


# ===========================================================================
# Snapshot Object
# ===========================================================================

class TestSnapshot:
    def test_visibility_committed_before(self):
        snap = Snapshot(xmin=5, xmax=10, active_txns=frozenset({7, 8}))
        assert snap.is_visible(3) is True   # Committed before window
        assert snap.is_visible(6) is True   # In window, not active

    def test_visibility_active(self):
        snap = Snapshot(xmin=5, xmax=10, active_txns=frozenset({7, 8}))
        assert snap.is_visible(7) is False  # Active at snapshot time
        assert snap.is_visible(8) is False

    def test_visibility_future(self):
        snap = Snapshot(xmin=5, xmax=10, active_txns=frozenset({7}))
        assert snap.is_visible(10) is False  # >= xmax
        assert snap.is_visible(15) is False


# ===========================================================================
# Version Object
# ===========================================================================

class TestVersion:
    def test_version_creation(self):
        v = Version("key", "val", xmin=1)
        assert v.key == "key"
        assert v.value == "val"
        assert v.xmin == 1
        assert v.xmax == 0
        assert v.is_deleted() is False

    def test_version_deleted(self):
        v = Version("key", "val", xmin=1, xmax=5)
        assert v.is_deleted() is True

    def test_version_chain(self):
        v1 = Version("key", "v1", xmin=1)
        v2 = Version("key", "v2", xmin=2)
        v2.prev_version = v1
        assert v2.prev_version.value == "v1"

    def test_version_repr(self):
        v = Version("k", 42, xmin=1, xmax=3)
        r = repr(v)
        assert "k" in r
        assert "42" in r


# ===========================================================================
# MVCCIndex Object
# ===========================================================================

class TestMVCCIndex:
    def test_add_and_lookup(self):
        idx = MVCCIndex("test", lambda v: v)
        idx.add("pk1", "a")
        idx.add("pk2", "b")
        idx.add("pk3", "a")
        assert idx.lookup("a") == {"pk1", "pk3"}
        assert idx.lookup("b") == {"pk2"}

    def test_remove(self):
        idx = MVCCIndex("test", lambda v: v)
        idx.add("pk1", "a")
        idx.add("pk2", "a")
        idx.remove("pk1", "a")
        assert idx.lookup("a") == {"pk2"}

    def test_remove_last_entry(self):
        idx = MVCCIndex("test", lambda v: v)
        idx.add("pk1", "a")
        idx.remove("pk1", "a")
        assert idx.lookup("a") == set()

    def test_range_lookup(self):
        idx = MVCCIndex("test", lambda v: v)
        for i in range(10):
            idx.add(f"pk{i}", i)
        result = idx.range_lookup(3, 7)
        assert result == {f"pk{i}" for i in range(3, 8)}


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_engine(self, engine):
        t = engine.begin()
        assert engine.scan(t) == {}
        assert engine.get(t, "anything") is None
        assert engine.count(t) == 0
        engine.commit(t)

    def test_single_txn_multiple_ops(self, engine):
        t = engine.begin()
        engine.put(t, "x", 1)
        engine.put(t, "x", 2)
        engine.put(t, "x", 3)
        engine.delete(t, "x")
        engine.put(t, "x", 4)
        assert engine.get(t, "x") == 4
        engine.commit(t)

    def test_many_concurrent_txns(self, engine):
        """Many concurrent read-only transactions."""
        t0 = engine.begin()
        engine.put(t0, "shared", "data")
        engine.commit(t0)

        txns = [engine.begin() for _ in range(50)]
        for t in txns:
            assert engine.get(t, "shared") == "data"
        for t in txns:
            engine.commit(t)

    def test_rapid_update_cycle(self, engine):
        """Rapid sequential updates to same key."""
        for i in range(100):
            t = engine.begin()
            engine.put(t, "counter", i)
            engine.commit(t)

        t = engine.begin()
        assert engine.get(t, "counter") == 99
        engine.commit(t)

    def test_gc_with_active_readers(self, engine):
        """GC should not collect versions needed by active readers."""
        for i in range(5):
            t = engine.begin()
            engine.put(t, "x", i)
            engine.commit(t)

        reader = engine.begin()
        assert engine.get(reader, "x") == 4

        # GC while reader is active
        engine.gc()

        # Reader should still work
        assert engine.get(reader, "x") == 4
        engine.commit(reader)

    def test_abort_after_many_writes(self, engine):
        """Aborting a transaction with many writes."""
        t0 = engine.begin()
        for i in range(10):
            engine.put(t0, f"key{i}", f"original{i}")
        engine.commit(t0)

        t = engine.begin()
        for i in range(10):
            engine.put(t, f"key{i}", f"modified{i}")
        engine.abort(t)

        t2 = engine.begin()
        for i in range(10):
            assert engine.get(t2, f"key{i}") == f"original{i}"
        engine.commit(t2)

    def test_put_after_delete_same_txn(self, engine):
        t0 = engine.begin()
        engine.put(t0, "x", "old")
        engine.commit(t0)

        t = engine.begin()
        engine.delete(t, "x")
        engine.put(t, "x", "new")
        engine.commit(t)

        t2 = engine.begin()
        assert engine.get(t2, "x") == "new"
        engine.commit(t2)

    def test_mixed_isolation_levels(self, engine):
        """Different transactions with different isolation levels."""
        t0 = engine.begin()
        engine.put(t0, "x", 1)
        engine.commit(t0)

        rc = engine.begin(IsolationLevel.READ_COMMITTED)
        rr = engine.begin(IsolationLevel.REPEATABLE_READ)

        t = engine.begin()
        engine.put(t, "x", 2)
        engine.commit(t)

        assert engine.get(rc, "x") == 2   # Read committed sees new value
        assert engine.get(rr, "x") == 1   # Repeatable read sees old value

        engine.commit(rc)
        engine.commit(rr)


# ===========================================================================
# Integration Scenarios
# ===========================================================================

class TestIntegration:
    def test_bank_transfer(self, engine):
        """Classic bank transfer scenario."""
        t0 = engine.begin()
        engine.put(t0, "acct:alice", 1000)
        engine.put(t0, "acct:bob", 500)
        engine.commit(t0)

        # Transfer $200 from alice to bob
        t = engine.begin()
        alice = engine.get(t, "acct:alice")
        bob = engine.get(t, "acct:bob")
        engine.put(t, "acct:alice", alice - 200)
        engine.put(t, "acct:bob", bob + 200)
        engine.commit(t)

        t2 = engine.begin()
        assert engine.get(t2, "acct:alice") == 800
        assert engine.get(t2, "acct:bob") == 700
        engine.commit(t2)

    def test_inventory_update(self, engine):
        """Inventory management with conflict detection."""
        t0 = engine.begin()
        engine.put(t0, "product:widget", {"stock": 10, "price": 9.99})
        engine.commit(t0)

        # Two concurrent orders
        order1 = engine.begin()
        order2 = engine.begin()

        item = engine.get(order1, "product:widget")
        engine.put(order1, "product:widget",
                   {"stock": item["stock"] - 3, "price": item["price"]})
        engine.commit(order1)

        # Order2 should conflict
        item2 = engine.get(order2, "product:widget")
        with pytest.raises(WriteConflictError):
            engine.put(order2, "product:widget",
                       {"stock": item2["stock"] - 5, "price": item2["price"]})
        engine.abort(order2)

    def test_session_management(self, engine):
        """Session store with expiry simulation."""
        t0 = engine.begin()
        for i in range(5):
            engine.put(t0, f"session:{i}", {"user": f"user{i}", "active": True})
        engine.commit(t0)

        # Expire some sessions
        t1 = engine.begin()
        for i in range(3):
            engine.delete(t1, f"session:{i}")
        engine.commit(t1)

        # Check remaining
        t2 = engine.begin()
        active = engine.scan(t2, "session:")
        assert len(active) == 2
        assert "session:3" in active
        assert "session:4" in active
        engine.commit(t2)

    def test_full_lifecycle(self, engine):
        """Complete lifecycle: create, read, update, delete, GC."""
        # Create
        t1 = engine.begin()
        engine.put(t1, "entity:1", {"name": "test", "version": 1})
        engine.commit(t1)

        # Read
        t2 = engine.begin()
        val = engine.get(t2, "entity:1")
        assert val["version"] == 1
        engine.commit(t2)

        # Update
        t3 = engine.begin()
        engine.put(t3, "entity:1", {"name": "test", "version": 2})
        engine.commit(t3)

        # Verify update
        t4 = engine.begin()
        assert engine.get(t4, "entity:1")["version"] == 2
        engine.commit(t4)

        # Delete
        t5 = engine.begin()
        engine.delete(t5, "entity:1")
        engine.commit(t5)

        # Verify delete
        t6 = engine.begin()
        assert engine.get(t6, "entity:1") is None
        engine.commit(t6)

        # GC
        collected = engine.gc()
        assert collected > 0
        assert engine.version_count("entity:1") == 0

    def test_savepoint_in_complex_operation(self, engine):
        """Savepoints used in a multi-step operation with partial rollback."""
        t = engine.begin()
        engine.put(t, "order:status", "pending")
        engine.put(t, "order:total", 0)

        engine.savepoint(t, "before_items")

        # Add items
        engine.put(t, "order:item:1", {"name": "A", "price": 10})
        engine.put(t, "order:item:2", {"name": "B", "price": 20})
        engine.put(t, "order:total", 30)

        engine.savepoint(t, "before_discount")

        # Apply discount that turns out to be invalid
        engine.put(t, "order:discount", "INVALID_CODE")
        engine.put(t, "order:total", 15)

        # Rollback discount
        engine.rollback_to_savepoint(t, "before_discount")
        assert engine.get(t, "order:total") == 30
        assert engine.get(t, "order:discount") is None

        engine.put(t, "order:status", "confirmed")
        engine.commit(t)

        t2 = engine.begin()
        assert engine.get(t2, "order:status") == "confirmed"
        assert engine.get(t2, "order:total") == 30
        assert engine.get(t2, "order:item:1")["name"] == "A"
        engine.commit(t2)


# ===========================================================================
# Stress / Volume Tests
# ===========================================================================

class TestStress:
    def test_many_keys(self, engine):
        """Test with many keys."""
        t = engine.begin()
        for i in range(1000):
            engine.put(t, f"key:{i:04d}", i)
        engine.commit(t)

        t2 = engine.begin()
        assert engine.count(t2) == 1000
        assert engine.get(t2, "key:0500") == 500
        engine.commit(t2)

    def test_many_versions_single_key(self, engine):
        """Test many versions of a single key."""
        for i in range(50):
            t = engine.begin()
            engine.put(t, "hot_key", i)
            engine.commit(t)

        t = engine.begin()
        assert engine.get(t, "hot_key") == 49
        engine.commit(t)

        # GC should clean up
        engine.gc()
        assert engine.version_count("hot_key") == 1

    def test_sequential_txn_throughput(self, engine):
        """Many sequential short transactions."""
        for i in range(200):
            t = engine.begin()
            engine.put(t, f"seq:{i}", i)
            if i % 3 == 0:
                engine.abort(t)
            else:
                engine.commit(t)

        t = engine.begin()
        count = engine.count(t, "seq:")
        # 200 txns, 1/3 aborted
        assert count > 100
        engine.commit(t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
