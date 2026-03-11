"""
V166: Lock Order Verification -- Comprehensive Tests
"""

import pytest
from lock_order_verifier import (
    LockOrderVerifier, LockTrace, LockEvent, LockHierarchy,
    EventType, OrderEdge, Cycle, VerificationResult,
    make_trace, make_acquire_sequence,
)


# ============================================================
# Basic: Consistent orderings (no deadlock)
# ============================================================

class TestConsistentOrdering:
    """Tests where lock ordering is consistent (no cycles)."""

    def test_single_lock_single_tx(self):
        """One transaction, one lock -- trivially consistent."""
        t = make_acquire_sequence("tx1", "A")
        v = LockOrderVerifier()
        r = v.verify_traces([t])
        assert r.consistent
        assert r.num_cycles == 0

    def test_two_locks_same_order(self):
        """Two transactions both acquire A then B -- consistent."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "A", "B")
        r = LockOrderVerifier().verify_traces([t1, t2])
        assert r.consistent

    def test_three_locks_same_order(self):
        """Multiple transactions all using same order A->B->C."""
        traces = [
            make_acquire_sequence(f"tx{i}", "A", "B", "C")
            for i in range(5)
        ]
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent
        assert r.num_cycles == 0

    def test_disjoint_lock_sets(self):
        """Transactions using completely different locks -- no conflict."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "X", "Y")
        r = LockOrderVerifier().verify_traces([t1, t2])
        assert r.consistent

    def test_partial_overlap_consistent(self):
        """Overlapping locks but consistent order."""
        t1 = make_acquire_sequence("tx1", "A", "B", "C")
        t2 = make_acquire_sequence("tx2", "B", "C")
        r = LockOrderVerifier().verify_traces([t1, t2])
        assert r.consistent

    def test_single_tx_many_locks(self):
        """One transaction acquires many locks in sequence."""
        locks = [f"L{i}" for i in range(10)]
        t = make_acquire_sequence("tx1", *locks)
        r = LockOrderVerifier().verify_traces([t])
        assert r.consistent

    def test_empty_traces(self):
        """No traces at all -- trivially consistent."""
        r = LockOrderVerifier().verify_traces([])
        assert r.consistent
        assert r.num_cycles == 0

    def test_empty_trace_events(self):
        """A trace with no events -- consistent."""
        t = LockTrace(transaction_id="tx1")
        r = LockOrderVerifier().verify_traces([t])
        assert r.consistent

    def test_acquire_release_acquire_different(self):
        """Acquire A, release A, acquire B -- no nesting, no edge."""
        t = make_trace("tx1", ("A", "A"), ("R", "A"), ("A", "B"), ("R", "B"))
        r = LockOrderVerifier().verify_traces([t])
        assert r.consistent
        # No edge A->B because A was released before B acquired
        assert "B" not in r.order_graph.get("A", set())


# ============================================================
# Basic: Inconsistent orderings (deadlock potential)
# ============================================================

class TestInconsistentOrdering:
    """Tests detecting potential deadlocks (cycles)."""

    def test_classic_AB_BA_deadlock(self):
        """Classic deadlock: tx1 acquires A,B; tx2 acquires B,A."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "A")
        r = LockOrderVerifier().verify_traces([t1, t2])
        assert not r.consistent
        assert r.num_cycles >= 1

    def test_deadlock_reports_locks(self):
        """Cycle should contain the conflicting locks."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "A")
        r = LockOrderVerifier().verify_traces([t1, t2])
        cycle = r.cycles[0]
        lock_set = set(cycle.locks)
        assert "A" in lock_set
        assert "B" in lock_set

    def test_deadlock_reports_transactions(self):
        """Cycle should identify the transactions involved."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "A")
        r = LockOrderVerifier().verify_traces([t1, t2])
        txns = r.all_involved_transactions
        assert "tx1" in txns
        assert "tx2" in txns

    def test_three_way_cycle(self):
        """A->B, B->C, C->A forms a 3-node cycle."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "C")
        t3 = make_acquire_sequence("tx3", "C", "A")
        r = LockOrderVerifier().verify_traces([t1, t2, t3])
        assert not r.consistent
        assert r.num_cycles >= 1

    def test_four_way_cycle(self):
        """A->B, B->C, C->D, D->A."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "C")
        t3 = make_acquire_sequence("tx3", "C", "D")
        t4 = make_acquire_sequence("tx4", "D", "A")
        r = LockOrderVerifier().verify_traces([t1, t2, t3, t4])
        assert not r.consistent

    def test_multiple_cycles(self):
        """Two independent deadlock cycles."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "A")
        t3 = make_acquire_sequence("tx3", "X", "Y")
        t4 = make_acquire_sequence("tx4", "Y", "X")
        r = LockOrderVerifier().verify_traces([t1, t2, t3, t4])
        assert not r.consistent
        assert r.num_cycles >= 2

    def test_self_loop_not_possible(self):
        """Acquiring the same lock twice doesn't create a self-loop edge."""
        t = make_trace("tx1", ("A", "X"), ("A", "X"))
        r = LockOrderVerifier().verify_traces([t])
        assert "X" not in r.order_graph.get("X", set())


# ============================================================
# Graph construction
# ============================================================

class TestGraphConstruction:
    """Tests for correct lock-order graph building."""

    def test_single_edge(self):
        """Acquiring A then B creates edge A->B."""
        t = make_acquire_sequence("tx1", "A", "B")
        r = LockOrderVerifier().verify_traces([t])
        assert "B" in r.order_graph.get("A", set())

    def test_three_locks_creates_three_edges(self):
        """Acquiring A,B,C creates edges A->B, A->C, B->C."""
        t = make_acquire_sequence("tx1", "A", "B", "C")
        r = LockOrderVerifier().verify_traces([t])
        assert "B" in r.order_graph["A"]
        assert "C" in r.order_graph["A"]
        assert "C" in r.order_graph["B"]

    def test_release_breaks_ordering(self):
        """After releasing A, acquiring C does not create A->C."""
        t = make_trace("tx1",
            ("A", "A"), ("A", "B"), ("R", "A"), ("A", "C"), ("R", "C"), ("R", "B"))
        r = LockOrderVerifier().verify_traces([t])
        assert "B" in r.order_graph.get("A", set())
        # A was released before C acquired, so no A->C
        assert "C" not in r.order_graph.get("A", set())
        # But B was still held when C was acquired, so B->C
        assert "C" in r.order_graph.get("B", set())

    def test_edge_info_tracks_transaction(self):
        """Edge info records which transaction created the edge."""
        t = make_acquire_sequence("tx1", "A", "B")
        r = LockOrderVerifier().verify_traces([t])
        edges = r.edge_info[("A", "B")]
        assert len(edges) == 1
        assert edges[0].transaction_id == "tx1"

    def test_multiple_tx_same_edge(self):
        """Multiple transactions creating the same edge are all recorded."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "A", "B")
        r = LockOrderVerifier().verify_traces([t1, t2])
        edges = r.edge_info[("A", "B")]
        assert len(edges) == 2
        txns = {e.transaction_id for e in edges}
        assert txns == {"tx1", "tx2"}

    def test_no_edges_for_sequential_non_nested(self):
        """Sequential acquire-release without nesting creates no ordering edges."""
        t = make_trace("tx1",
            ("A", "A"), ("R", "A"),
            ("A", "B"), ("R", "B"),
            ("A", "C"), ("R", "C"))
        r = LockOrderVerifier().verify_traces([t])
        assert len(r.order_graph) == 0


# ============================================================
# Cycle properties
# ============================================================

class TestCycleProperties:
    """Tests for Cycle dataclass methods."""

    def test_cycle_lock_pairs(self):
        """lock_pairs returns adjacent pairs in the cycle."""
        c = Cycle(
            locks=["A", "B", "C", "A"],
            edges=[
                OrderEdge("A", "B", "tx1"),
                OrderEdge("B", "C", "tx2"),
                OrderEdge("C", "A", "tx3"),
            ]
        )
        assert c.lock_pairs == [("A", "B"), ("B", "C"), ("C", "A")]

    def test_cycle_transactions(self):
        c = Cycle(
            locks=["A", "B", "A"],
            edges=[
                OrderEdge("A", "B", "tx1"),
                OrderEdge("B", "A", "tx2"),
            ]
        )
        assert c.transactions == {"tx1", "tx2"}

    def test_all_conflicting_pairs(self):
        """VerificationResult aggregates pairs from all cycles."""
        t1 = make_acquire_sequence("tx1", "A", "B")
        t2 = make_acquire_sequence("tx2", "B", "A")
        r = LockOrderVerifier().verify_traces([t1, t2])
        pairs = r.all_conflicting_pairs
        assert len(pairs) >= 2  # A->B and B->A


# ============================================================
# Warnings and edge cases
# ============================================================

class TestWarnings:
    """Tests for warning generation."""

    def test_release_unheld_lock_warns(self):
        """Releasing a lock not held generates a warning."""
        t = make_trace("tx1", ("R", "X"))
        r = LockOrderVerifier().verify_traces([t])
        assert len(r.warnings) == 1
        assert "unheld" in r.warnings[0].lower()

    def test_double_release_warns(self):
        """Releasing a lock twice warns on the second release."""
        t = make_trace("tx1", ("A", "X"), ("R", "X"), ("R", "X"))
        r = LockOrderVerifier().verify_traces([t])
        assert len(r.warnings) == 1


# ============================================================
# Hierarchical locks
# ============================================================

class TestHierarchicalLocks:
    """Tests for hierarchical lock support."""

    def test_hierarchy_creation(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        h.add_child("row", "table")
        assert h.get_parent("table") == "db"
        assert h.get_parent("row") == "table"
        assert h.get_parent("db") is None

    def test_hierarchy_ancestors(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        h.add_child("row", "table")
        assert h.get_ancestors("row") == ["table", "db"]

    def test_hierarchy_descendants(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        h.add_child("row", "table")
        desc = h.get_all_descendants("db")
        assert set(desc) == {"table", "row"}

    def test_hierarchy_is_ancestor(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        h.add_child("row", "table")
        assert h.is_ancestor_of("db", "row")
        assert h.is_ancestor_of("table", "row")
        assert not h.is_ancestor_of("row", "db")

    def test_hierarchy_root(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        h.add_child("row", "table")
        assert h.get_root("row") == "db"
        assert h.get_root("db") == "db"

    def test_hierarchy_children(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("t1", "db")
        h.add_child("t2", "db")
        assert set(h.get_children("db")) == {"t1", "t2"}

    def test_hierarchy_implicit_locking(self):
        """Acquiring parent implicitly locks children -- edges to children."""
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")

        # tx1 acquires X then db (which implicitly locks table)
        t = make_acquire_sequence("tx1", "X", "db")
        v = LockOrderVerifier(hierarchy=h)
        r = v.verify_traces([t])
        # Should have edge X -> db and X -> table (implicit)
        assert "db" in r.order_graph.get("X", set())
        assert "table" in r.order_graph.get("X", set())

    def test_hierarchy_violation_parent_after_child(self):
        """Acquiring parent while child is already held is a violation."""
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")

        t = make_trace("tx1", ("A", "table"), ("A", "db"), ("R", "db"), ("R", "table"))
        v = LockOrderVerifier(hierarchy=h)
        r = v.verify_traces([t])
        assert not r.consistent
        assert len(r.hierarchy_violations) >= 1
        assert "parent" in r.hierarchy_violations[0].lower()

    def test_hierarchy_consistent_parent_before_child(self):
        """Parent then child is the correct hierarchy order -- no violation."""
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")

        t = make_trace("tx1", ("A", "db"), ("A", "table"), ("R", "table"), ("R", "db"))
        v = LockOrderVerifier(hierarchy=h)
        r = v.verify_traces([t])
        assert len(r.hierarchy_violations) == 0

    def test_hierarchy_deadlock_with_implicit(self):
        """Deadlock involving implicit child locks."""
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")

        # tx1: acquire X then db (implicitly table)
        # tx2: acquire table then X
        t1 = make_acquire_sequence("tx1", "X", "db")
        t2 = make_acquire_sequence("tx2", "table", "X")
        v = LockOrderVerifier(hierarchy=h)
        r = v.verify_traces([t1, t2])
        assert not r.consistent

    def test_hierarchy_contains(self):
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("table", "db")
        assert h.contains("db")
        assert h.contains("table")
        assert not h.contains("other")

    def test_deep_hierarchy(self):
        """Deep hierarchy: db -> schema -> table -> row."""
        h = LockHierarchy()
        h.add_root("db")
        h.add_child("schema", "db")
        h.add_child("table", "schema")
        h.add_child("row", "table")
        assert h.get_ancestors("row") == ["table", "schema", "db"]
        desc = h.get_all_descendants("db")
        assert set(desc) == {"schema", "table", "row"}


# ============================================================
# check_new_trace (speculative check)
# ============================================================

class TestCheckNewTrace:
    """Tests for speculative trace checking."""

    def test_check_new_trace_no_conflict(self):
        """New trace that is consistent doesn't flag."""
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        t2 = make_acquire_sequence("tx2", "A", "B")
        r = v.check_new_trace(t2)
        assert r.consistent

    def test_check_new_trace_conflict(self):
        """New trace that would create a cycle is flagged."""
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        t2 = make_acquire_sequence("tx2", "B", "A")
        r = v.check_new_trace(t2)
        assert not r.consistent

    def test_check_new_trace_does_not_persist(self):
        """Speculative check doesn't modify the verifier state."""
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        t2 = make_acquire_sequence("tx2", "B", "A")
        v.check_new_trace(t2)
        # Original state should be unchanged -- only tx1
        r = v.verify()
        assert r.consistent


# ============================================================
# Helper functions
# ============================================================

class TestHelpers:
    """Tests for helper/convenience functions."""

    def test_make_trace(self):
        t = make_trace("tx1", ("A", "X"), ("A", "Y"), ("R", "Y"), ("R", "X"))
        assert t.transaction_id == "tx1"
        assert len(t.events) == 4
        assert t.events[0].event_type == EventType.ACQUIRE
        assert t.events[0].lock_name == "X"
        assert t.events[2].event_type == EventType.RELEASE

    def test_make_acquire_sequence(self):
        t = make_acquire_sequence("tx1", "A", "B", "C")
        assert len(t.events) == 6
        # First 3: acquires
        assert all(e.event_type == EventType.ACQUIRE for e in t.events[:3])
        # Last 3: releases in reverse
        assert all(e.event_type == EventType.RELEASE for e in t.events[3:])
        assert t.events[3].lock_name == "C"
        assert t.events[4].lock_name == "B"
        assert t.events[5].lock_name == "A"

    def test_lock_trace_chaining(self):
        t = LockTrace(transaction_id="tx1")
        result = t.add_acquire("A").add_acquire("B").add_release("B").add_release("A")
        assert result is t
        assert len(t.events) == 4


# ============================================================
# Complex / realistic scenarios
# ============================================================

class TestComplexScenarios:
    """Realistic multi-transaction scenarios."""

    def test_database_scenario_consistent(self):
        """Simulated DB: all transactions lock table before row."""
        traces = [
            make_acquire_sequence("select_1", "table_lock", "row_42"),
            make_acquire_sequence("update_1", "table_lock", "row_42"),
            make_acquire_sequence("insert_1", "table_lock", "row_99"),
        ]
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent

    def test_database_scenario_deadlock(self):
        """Two transactions locking rows in opposite order."""
        t1 = make_acquire_sequence("tx1", "row_1", "row_2")
        t2 = make_acquire_sequence("tx2", "row_2", "row_1")
        r = LockOrderVerifier().verify_traces([t1, t2])
        assert not r.consistent

    def test_philosopher_dining(self):
        """Classic dining philosophers: 5 philosophers, 5 forks, circular."""
        traces = []
        for i in range(5):
            left = f"fork_{i}"
            right = f"fork_{(i+1) % 5}"
            traces.append(make_acquire_sequence(f"phil_{i}", left, right))
        r = LockOrderVerifier().verify_traces(traces)
        # This creates a cycle: fork_0->fork_1->...->fork_4->fork_0
        assert not r.consistent

    def test_philosopher_fixed(self):
        """Fixed dining philosophers: last philosopher reverses order."""
        traces = []
        for i in range(4):
            left = f"fork_{i}"
            right = f"fork_{(i+1) % 5}"
            traces.append(make_acquire_sequence(f"phil_{i}", left, right))
        # Last philosopher acquires in reverse to break cycle
        traces.append(make_acquire_sequence("phil_4", "fork_0", "fork_4"))
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent

    def test_interleaved_acquire_release(self):
        """Complex interleaving: A held while B acquired/released, then C acquired."""
        t = make_trace("tx1",
            ("A", "A"),
            ("A", "B"),
            ("R", "B"),
            ("A", "C"),
            ("R", "C"),
            ("R", "A"))
        r = LockOrderVerifier().verify_traces([t])
        # Edges: A->B, A->C (B released before C, so no B->C)
        assert "B" in r.order_graph.get("A", set())
        assert "C" in r.order_graph.get("A", set())
        assert "C" not in r.order_graph.get("B", set())
        assert r.consistent

    def test_many_transactions_no_conflict(self):
        """20 transactions all using the same lock order."""
        traces = [
            make_acquire_sequence(f"tx{i}", "M1", "M2", "M3")
            for i in range(20)
        ]
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent

    def test_single_conflicting_pair_among_many(self):
        """One bad transaction among many good ones."""
        traces = [
            make_acquire_sequence(f"tx{i}", "A", "B")
            for i in range(10)
        ]
        traces.append(make_acquire_sequence("bad_tx", "B", "A"))
        r = LockOrderVerifier().verify_traces(traces)
        assert not r.consistent
        assert "bad_tx" in r.all_involved_transactions

    def test_nested_locking_consistent(self):
        """Deeply nested consistent locking."""
        t = make_acquire_sequence("tx1", "L1", "L2", "L3", "L4", "L5")
        r = LockOrderVerifier().verify_traces([t])
        assert r.consistent
        # Should have edges for all pairs
        for i in range(1, 6):
            for j in range(i+1, 6):
                assert f"L{j}" in r.order_graph[f"L{i}"]

    def test_reentrant_same_lock(self):
        """Same lock acquired twice (reentrant) -- no self-edge."""
        t = make_trace("tx1", ("A", "M"), ("A", "M"), ("R", "M"), ("R", "M"))
        r = LockOrderVerifier().verify_traces([t])
        assert "M" not in r.order_graph.get("M", set())


# ============================================================
# Edge cases and data classes
# ============================================================

class TestDataClasses:
    """Tests for data class behavior."""

    def test_lock_event_frozen(self):
        e = LockEvent(EventType.ACQUIRE, "X", 0)
        with pytest.raises(AttributeError):
            e.lock_name = "Y"

    def test_order_edge_frozen(self):
        e = OrderEdge("A", "B", "tx1")
        with pytest.raises(AttributeError):
            e.from_lock = "C"

    def test_verification_result_defaults(self):
        r = VerificationResult(consistent=True)
        assert r.num_cycles == 0
        assert r.all_conflicting_pairs == set()
        assert r.all_involved_transactions == set()

    def test_event_type_values(self):
        assert EventType.ACQUIRE != EventType.RELEASE

    def test_lock_trace_default_events(self):
        t = LockTrace(transaction_id="tx")
        assert t.events == []


# ============================================================
# Incremental usage
# ============================================================

class TestIncrementalUsage:
    """Tests for adding traces incrementally."""

    def test_add_then_verify(self):
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        v.add_trace(make_acquire_sequence("tx2", "A", "B"))
        r = v.verify()
        assert r.consistent

    def test_add_creates_conflict(self):
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        v.add_trace(make_acquire_sequence("tx2", "B", "A"))
        r = v.verify()
        assert not r.consistent

    def test_reverify_clears_state(self):
        """verify() rebuilds the graph each time."""
        v = LockOrderVerifier()
        v.add_trace(make_acquire_sequence("tx1", "A", "B"))
        r1 = v.verify()
        assert r1.consistent
        v.add_trace(make_acquire_sequence("tx2", "B", "A"))
        r2 = v.verify()
        assert not r2.consistent

    def test_add_traces_bulk(self):
        v = LockOrderVerifier()
        v.add_traces([
            make_acquire_sequence("tx1", "A", "B"),
            make_acquire_sequence("tx2", "B", "C"),
        ])
        r = v.verify()
        assert r.consistent


# ============================================================
# Large-scale tests
# ============================================================

class TestScalability:
    """Tests with larger numbers of locks/transactions."""

    def test_chain_no_cycle(self):
        """Linear chain: L0->L1->...->L9, no cycle."""
        traces = []
        for i in range(9):
            traces.append(make_acquire_sequence(f"tx{i}", f"L{i}", f"L{i+1}"))
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent

    def test_chain_with_back_edge(self):
        """Chain plus one back edge creates a cycle."""
        traces = []
        for i in range(5):
            traces.append(make_acquire_sequence(f"tx{i}", f"L{i}", f"L{i+1}"))
        traces.append(make_acquire_sequence("txback", "L5", "L0"))
        r = LockOrderVerifier().verify_traces(traces)
        assert not r.consistent

    def test_star_topology_consistent(self):
        """All transactions lock center first, then one leaf -- consistent."""
        traces = [
            make_acquire_sequence(f"tx{i}", "center", f"leaf_{i}")
            for i in range(10)
        ]
        r = LockOrderVerifier().verify_traces(traces)
        assert r.consistent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
