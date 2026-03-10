"""Tests for V036: Concurrent Separation Logic."""

import pytest
import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from concurrent_separation_logic import (
    # Core types
    Owner, Resource, LockInvariant, ThreadState,
    CmdKind, Cmd,
    # Command constructors
    CNew, CAssign, CLoad, CStore, CDispose, CNull,
    CAcquire, CRelease, CParallel, CAtomic, CSkip, CSeq, CSeqList,
    # Rely-guarantee
    RelyGuarantee,
    # Results
    CSLVerdict, CSLResult, OwnershipReport,
    # Analysis
    detect_races, analyze_ownership,
    # Verifier
    CSLVerifier,
    # High-level API
    verify_concurrent, check_race_freedom, ownership_analysis,
    check_rely_guarantee,
    # Helpers
    _collect_vars_cmd, _writes_cmd, _heap_accesses,
)

# Import SL primitives
sys.path.insert(0, os.path.join(os.path.dirname(_dir), "V031_separation_logic"))
from separation_logic import (
    EVar, ENull, EInt,
    Emp, PointsTo, Star, StarList, Pure, PureEq, PureNeq, LSeg,
    SLFormula, SymbolicHeap,
    fresh_var, reset_fresh,
)


# ════════════════════════════════════════════════════════════════════
# Command constructors
# ════════════════════════════════════════════════════════════════════

class TestCommandConstructors:
    def test_skip(self):
        c = CSkip()
        assert c.kind == CmdKind.SKIP

    def test_new(self):
        c = CNew("x")
        assert c.kind == CmdKind.NEW
        assert c.args == ["x"]

    def test_assign(self):
        c = CAssign("x", "y")
        assert c.kind == CmdKind.ASSIGN
        assert c.args == ["x", "y"]

    def test_load(self):
        c = CLoad("x", "p")
        assert c.kind == CmdKind.LOAD

    def test_store(self):
        c = CStore("p", "v")
        assert c.kind == CmdKind.STORE

    def test_dispose(self):
        c = CDispose("p")
        assert c.kind == CmdKind.DISPOSE

    def test_null(self):
        c = CNull("x")
        assert c.kind == CmdKind.NULL

    def test_acquire(self):
        c = CAcquire("mutex")
        assert c.kind == CmdKind.ACQUIRE
        assert c.args == ["mutex"]

    def test_release(self):
        c = CRelease("mutex")
        assert c.kind == CmdKind.RELEASE

    def test_parallel(self):
        c = CParallel(CSkip(), CSkip())
        assert c.kind == CmdKind.PARALLEL

    def test_atomic(self):
        c = CAtomic(CStore("p", "v"))
        assert c.kind == CmdKind.ATOMIC

    def test_seq(self):
        c = CSeq(CNew("x"), CStore("x", "v"))
        assert c.kind == CmdKind.SEQ

    def test_seq_list_empty(self):
        c = CSeqList([])
        assert c.kind == CmdKind.SKIP

    def test_seq_list_single(self):
        c = CSeqList([CNew("x")])
        assert c.kind == CmdKind.NEW

    def test_seq_list_multiple(self):
        c = CSeqList([CNew("x"), CStore("x", "v"), CDispose("x")])
        assert c.kind == CmdKind.SEQ
        assert c.args[0].kind == CmdKind.NEW
        assert c.args[1].kind == CmdKind.SEQ

    def test_repr(self):
        c = CParallel(CNew("x"), CNew("y"))
        s = repr(c)
        assert "||" in s


# ════════════════════════════════════════════════════════════════════
# Variable collection
# ════════════════════════════════════════════════════════════════════

class TestVarCollection:
    def test_collect_vars_new(self):
        assert _collect_vars_cmd(CNew("x")) == {"x"}

    def test_collect_vars_assign(self):
        assert _collect_vars_cmd(CAssign("x", "y")) == {"x", "y"}

    def test_collect_vars_parallel(self):
        c = CParallel(CNew("x"), CNew("y"))
        assert _collect_vars_cmd(c) == {"x", "y"}

    def test_collect_vars_seq(self):
        c = CSeq(CNew("x"), CStore("x", "v"))
        assert _collect_vars_cmd(c) == {"x", "v"}

    def test_writes_new(self):
        assert _writes_cmd(CNew("x")) == {"x"}

    def test_writes_load(self):
        assert _writes_cmd(CLoad("y", "p")) == {"y"}

    def test_writes_store(self):
        assert _writes_cmd(CStore("p", "v")) == {"p"}

    def test_writes_parallel(self):
        c = CParallel(CNew("x"), CAssign("y", "z"))
        assert _writes_cmd(c) == {"x", "y"}

    def test_heap_accesses_load(self):
        acc = _heap_accesses(CLoad("x", "p"))
        assert ('read', 'p') in acc

    def test_heap_accesses_store(self):
        acc = _heap_accesses(CStore("p", "v"))
        assert ('write', 'p') in acc

    def test_heap_accesses_parallel(self):
        c = CParallel(CLoad("x", "p"), CStore("q", "v"))
        acc = _heap_accesses(c)
        assert ('read', 'p') in acc
        assert ('write', 'q') in acc


# ════════════════════════════════════════════════════════════════════
# Data race detection
# ════════════════════════════════════════════════════════════════════

class TestRaceDetection:
    def test_no_race_disjoint(self):
        """Disjoint variable access: no race."""
        c = CParallel(CNew("x"), CNew("y"))
        races = detect_races(c)
        assert len(races) == 0

    def test_write_write_race(self):
        """Both threads write the same variable: race."""
        c = CParallel(CAssign("x", "a"), CAssign("x", "b"))
        races = detect_races(c)
        assert len(races) > 0
        assert any(r['var'] == 'x' for r in races)

    def test_write_read_race(self):
        """One writes, other reads: race."""
        c = CParallel(CAssign("x", "a"), CLoad("y", "x"))
        races = detect_races(c)
        assert len(races) > 0

    def test_read_read_no_race(self):
        """Both only read: no race."""
        c = CParallel(CLoad("a", "p"), CLoad("b", "q"))
        # p and q are different, no shared write
        races = detect_races(c)
        # a and b are written (load writes to dest), p and q are read
        # No overlap in writes-vs-reads for same var
        write_write = [r for r in races if r['kind'] == 'write-write']
        # a, b are disjoint writes
        assert len(write_write) == 0

    def test_lock_protected_no_race(self):
        """Lock-protected access: no race."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        t1 = CSeqList([CAcquire("m"), CStore("x", "a"), CRelease("m")])
        t2 = CSeqList([CAcquire("m"), CStore("x", "b"), CRelease("m")])
        c = CParallel(t1, t2)
        races = detect_races(c, {"m": inv})
        # x is protected by lock m
        x_races = [r for r in races if r['var'] == 'x']
        assert len(x_races) == 0

    def test_heap_write_write_race(self):
        """Both threads store to same heap cell: race."""
        c = CParallel(CStore("p", "a"), CStore("p", "b"))
        races = detect_races(c)
        assert len(races) > 0

    def test_nested_parallel_race(self):
        """Race in nested parallel."""
        inner = CParallel(CAssign("x", "a"), CAssign("x", "b"))
        c = CSeq(CNew("x"), inner)
        races = detect_races(c)
        assert any(r['var'] == 'x' for r in races)


# ════════════════════════════════════════════════════════════════════
# Lock invariants
# ════════════════════════════════════════════════════════════════════

class TestLockInvariants:
    def test_create_lock_invariant(self):
        inv = LockInvariant("mutex", PointsTo(EVar("x"), EVar("v")))
        assert inv.lock_name == "mutex"
        assert inv.invariant.kind.name == "POINTS_TO"

    def test_copy(self):
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        inv2 = inv.copy()
        assert inv2.lock_name == "m"
        assert inv2 is not inv


# ════════════════════════════════════════════════════════════════════
# Ownership analysis
# ════════════════════════════════════════════════════════════════════

class TestOwnershipAnalysis:
    def test_disjoint_threads(self):
        """Each thread uses different variables."""
        c = CParallel(CNew("x"), CNew("y"))
        report = ownership_analysis(c)
        assert report.is_race_free()
        assert len(report.shared_resources) == 0

    def test_shared_unprotected(self):
        """Shared variable without lock protection."""
        c = CParallel(CAssign("x", "a"), CAssign("x", "b"))
        report = ownership_analysis(c)
        assert "x" in report.shared_resources
        assert "x" in report.unprotected_shared
        assert not report.is_race_free()

    def test_shared_protected(self):
        """Shared variable with lock protection."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        t1 = CSeqList([CAcquire("m"), CStore("x", "a"), CRelease("m")])
        t2 = CSeqList([CAcquire("m"), CStore("x", "b"), CRelease("m")])
        c = CParallel(t1, t2)
        report = ownership_analysis(c, {"m": inv})
        assert "x" in report.protected_resources.get("m", set())
        assert "x" not in report.unprotected_shared

    def test_thread_resources_collected(self):
        """Each thread's resources are properly tracked."""
        c = CParallel(
            CSeq(CNew("x"), CStore("x", "a")),
            CSeq(CNew("y"), CStore("y", "b")),
        )
        report = ownership_analysis(c)
        left_vars = report.thread_resources.get("main.L", set())
        right_vars = report.thread_resources.get("main.R", set())
        assert "x" in left_vars
        assert "y" in right_vars


# ════════════════════════════════════════════════════════════════════
# CSL Verification: Sequential base
# ════════════════════════════════════════════════════════════════════

class TestCSLSequential:
    def test_skip(self):
        """Skip preserves precondition."""
        pre = PointsTo(EVar("x"), EVar("v"))
        result = verify_concurrent(pre, CSkip(), pre)
        assert result.is_safe()

    def test_acquire_release(self):
        """Acquire then release returns to original state."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        pre = Emp()
        post = Emp()
        cmd = CSeq(CAcquire("m"), CRelease("m"))
        result = verify_concurrent(pre, cmd, post, {"m": inv})
        assert result.is_safe()

    def test_acquire_gives_invariant(self):
        """After acquire, thread has the lock invariant."""
        inv_formula = PointsTo(EVar("x"), EVar("v"))
        inv = LockInvariant("m", inv_formula)
        pre = Emp()
        post = PointsTo(EVar("x"), EVar("v"))
        result = verify_concurrent(pre, CAcquire("m"), post, {"m": inv})
        assert result.is_safe()

    def test_acquire_unknown_lock_fails(self):
        """Acquiring unknown lock fails."""
        result = verify_concurrent(Emp(), CAcquire("nonexistent"), Emp())
        assert not result.is_safe()

    def test_release_without_invariant_fails(self):
        """Releasing without holding invariant fails."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        # Pre is Emp but we need x|->v to release
        result = verify_concurrent(Emp(), CRelease("m"), Emp(), {"m": inv})
        assert not result.is_safe()

    def test_double_acquire_fails(self):
        """Double acquire is a deadlock."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        cmd = CSeq(CAcquire("m"), CAcquire("m"))
        result = verify_concurrent(Emp(), cmd, Emp(), {"m": inv})
        assert not result.is_safe()
        assert any("deadlock" in e.lower() for e in result.errors)


# ════════════════════════════════════════════════════════════════════
# CSL Verification: Parallel composition
# ════════════════════════════════════════════════════════════════════

class TestCSLParallel:
    def test_disjoint_parallel_safe(self):
        """Two threads with disjoint resources: safe."""
        pre = Star(PointsTo(EVar("x"), ENull()), PointsTo(EVar("y"), ENull()))
        t1 = CSkip()
        t2 = CSkip()
        post = Star(PointsTo(EVar("x"), ENull()), PointsTo(EVar("y"), ENull()))
        result = verify_concurrent(pre, CParallel(t1, t2), post)
        assert result.is_safe()

    def test_parallel_race_detected(self):
        """Parallel with shared writes: race."""
        cmd = CParallel(CAssign("x", "a"), CAssign("x", "b"))
        result = verify_concurrent(Emp(), cmd, Emp())
        assert result.verdict == CSLVerdict.RACE
        assert len(result.race_reports) > 0

    def test_lock_protected_parallel_no_race(self):
        """Lock-protected parallel access: no race for protected vars."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        t1 = CSeqList([CAcquire("m"), CSkip(), CRelease("m")])
        t2 = CSeqList([CAcquire("m"), CSkip(), CRelease("m")])
        pre = Emp()
        post = Emp()
        result = verify_concurrent(pre, CParallel(t1, t2), post, {"m": inv})
        # x is protected -- no race on x
        x_races = [r for r in result.race_reports if r['var'] == 'x']
        assert len(x_races) == 0


# ════════════════════════════════════════════════════════════════════
# CSL Verification: Atomic blocks
# ════════════════════════════════════════════════════════════════════

class TestCSLAtomic:
    def test_atomic_skip(self):
        """Atomic skip is safe."""
        pre = PointsTo(EVar("x"), EVar("v"))
        result = verify_concurrent(pre, CAtomic(CSkip()), pre)
        assert result.is_safe()

    def test_atomic_in_parallel(self):
        """Atomic blocks in parallel threads."""
        pre = Star(PointsTo(EVar("x"), ENull()), PointsTo(EVar("y"), ENull()))
        t1 = CAtomic(CSkip())
        t2 = CAtomic(CSkip())
        post = Star(PointsTo(EVar("x"), ENull()), PointsTo(EVar("y"), ENull()))
        result = verify_concurrent(pre, CParallel(t1, t2), post)
        assert result.is_safe()


# ════════════════════════════════════════════════════════════════════
# Rely-guarantee
# ════════════════════════════════════════════════════════════════════

class TestRelyGuarantee:
    def test_compatible_rely_guarantee(self):
        """Thread guarantees match other's rely."""
        x = EVar("x")
        v = EVar("v")
        w = EVar("w")
        inv = PointsTo(x, v)

        specs = [
            {
                'thread_id': 't1',
                'pre': PointsTo(x, v),
                'cmd': CSkip(),
                'post': PointsTo(x, v),
                'guarantee': [(PointsTo(x, v), PointsTo(x, v))],
                'rely': [(PointsTo(x, v), PointsTo(x, v))],
            },
            {
                'thread_id': 't2',
                'pre': Emp(),
                'cmd': CSkip(),
                'post': Emp(),
                'guarantee': [(PointsTo(x, v), PointsTo(x, v))],
                'rely': [(PointsTo(x, v), PointsTo(x, v))],
            },
        ]

        result = check_rely_guarantee(specs)
        assert result.verdict == CSLVerdict.SAFE

    def test_incompatible_guarantee(self):
        """Thread guarantee not in other's rely.

        t1 guarantees it may change x|->v to x|->w (a heap mutation).
        t2 relies on y|->v staying as y|->v (a different resource).
        t1's guarantee (x|->v -> x|->w) does NOT entail t2's rely (y|->v -> y|->v)
        because the spatial atoms don't match (x != y).
        """
        x = EVar("x")
        y = EVar("y")
        v = EVar("v")
        w = EVar("w")

        specs = [
            {
                'thread_id': 't1',
                'pre': PointsTo(x, v),
                'cmd': CSkip(),
                'post': PointsTo(x, v),
                'guarantee': [(PointsTo(x, v), PointsTo(x, w))],  # Changes x
                'rely': [(PointsTo(y, v), PointsTo(y, v))],
            },
            {
                'thread_id': 't2',
                'pre': PointsTo(y, v),
                'cmd': CSkip(),
                'post': PointsTo(y, v),
                'guarantee': [(PointsTo(y, v), PointsTo(y, v))],
                'rely': [(PointsTo(y, v), PointsTo(y, v))],  # t1's guar doesn't match
            },
        ]

        result = check_rely_guarantee(specs)
        assert result.verdict == CSLVerdict.UNSAFE
        assert len(result.errors) > 0


# ════════════════════════════════════════════════════════════════════
# Resource and ThreadState
# ════════════════════════════════════════════════════════════════════

class TestResourceTypes:
    def test_resource_copy(self):
        r = Resource(PointsTo(EVar("x"), ENull()), Owner.THREAD_LOCAL, "t1")
        r2 = r.copy()
        assert r2.owner == Owner.THREAD_LOCAL
        assert r2.owner_id == "t1"
        assert r2 is not r

    def test_thread_state_copy(self):
        ts = ThreadState("t1", PointsTo(EVar("x"), ENull()), ["m1"])
        ts2 = ts.copy()
        assert ts2.thread_id == "t1"
        assert ts2.held_locks == ["m1"]
        assert ts2 is not ts

    def test_resource_owners(self):
        assert Owner.THREAD_LOCAL != Owner.SHARED
        assert Owner.SHARED != Owner.UNOWNED


# ════════════════════════════════════════════════════════════════════
# Race freedom API
# ════════════════════════════════════════════════════════════════════

class TestRaceFreedomAPI:
    def test_race_free_disjoint(self):
        assert check_race_freedom(CParallel(CNew("x"), CNew("y")))

    def test_race_on_shared_write(self):
        assert not check_race_freedom(CParallel(CAssign("x", "a"), CAssign("x", "b")))

    def test_race_free_with_lock(self):
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        t1 = CSeqList([CAcquire("m"), CStore("x", "a"), CRelease("m")])
        t2 = CSeqList([CAcquire("m"), CStore("x", "b"), CRelease("m")])
        c = CParallel(t1, t2)
        assert check_race_freedom(c, {"m": inv})


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_parallel(self):
        """Parallel of two skips."""
        result = verify_concurrent(Emp(), CParallel(CSkip(), CSkip()), Emp())
        assert result.is_safe()

    def test_deeply_nested_seq(self):
        """Deep sequential chain."""
        cmd = CSeqList([CSkip()] * 5)
        result = verify_concurrent(Emp(), cmd, Emp())
        assert result.is_safe()

    def test_multiple_locks(self):
        """Multiple lock invariants."""
        inv1 = LockInvariant("m1", PointsTo(EVar("x"), ENull()))
        inv2 = LockInvariant("m2", PointsTo(EVar("y"), ENull()))
        t1 = CSeqList([CAcquire("m1"), CSkip(), CRelease("m1")])
        t2 = CSeqList([CAcquire("m2"), CSkip(), CRelease("m2")])
        result = verify_concurrent(Emp(), CParallel(t1, t2), Emp(),
                                    {"m1": inv1, "m2": inv2})
        assert result.is_safe()

    def test_release_unknown_lock(self):
        """Releasing unknown lock fails."""
        result = verify_concurrent(Emp(), CRelease("unknown"), Emp())
        assert not result.is_safe()

    def test_seq_with_acquire_release(self):
        """Sequence: acquire, work, release."""
        inv = LockInvariant("m", PointsTo(EVar("x"), EVar("v")))
        cmd = CSeqList([CAcquire("m"), CSkip(), CRelease("m")])
        result = verify_concurrent(Emp(), cmd, Emp(), {"m": inv})
        assert result.is_safe()

    def test_formula_with_lock_raw(self):
        """Pass raw formula as lock invariant (not LockInvariant)."""
        result = verify_concurrent(
            Emp(),
            CSeq(CAcquire("m"), CRelease("m")),
            Emp(),
            {"m": PointsTo(EVar("x"), ENull())},
        )
        assert result.is_safe()


# ════════════════════════════════════════════════════════════════════
# Integration: realistic concurrent patterns
# ════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_producer_consumer_skeleton(self):
        """Producer-consumer with shared buffer protected by lock."""
        inv = LockInvariant("buf_lock", PointsTo(EVar("buf"), EVar("data")))

        producer = CSeqList([
            CAcquire("buf_lock"),
            CSkip(),  # write to buf
            CRelease("buf_lock"),
        ])

        consumer = CSeqList([
            CAcquire("buf_lock"),
            CSkip(),  # read from buf
            CRelease("buf_lock"),
        ])

        result = verify_concurrent(
            Emp(),
            CParallel(producer, consumer),
            Emp(),
            {"buf_lock": inv},
        )
        assert result.is_safe()

    def test_two_lock_protocol(self):
        """Two different locks protecting different resources."""
        inv_a = LockInvariant("lock_a", PointsTo(EVar("a"), ENull()))
        inv_b = LockInvariant("lock_b", PointsTo(EVar("b"), ENull()))

        t1 = CSeqList([
            CAcquire("lock_a"),
            CSkip(),
            CRelease("lock_a"),
        ])

        t2 = CSeqList([
            CAcquire("lock_b"),
            CSkip(),
            CRelease("lock_b"),
        ])

        result = verify_concurrent(
            Emp(),
            CParallel(t1, t2),
            Emp(),
            {"lock_a": inv_a, "lock_b": inv_b},
        )
        assert result.is_safe()

    def test_unprotected_shared_detected(self):
        """Shared access without lock: race detected."""
        cmd = CParallel(
            CStore("shared", "val1"),
            CLoad("result", "shared"),
        )
        result = verify_concurrent(
            PointsTo(EVar("shared"), ENull()),
            cmd,
            Emp(),
        )
        assert result.verdict == CSLVerdict.RACE

    def test_ownership_producer_consumer(self):
        """Ownership analysis of producer-consumer."""
        inv = LockInvariant("m", PointsTo(EVar("buf"), EVar("d")))
        producer = CSeqList([CAcquire("m"), CStore("buf", "new_data"), CRelease("m")])
        consumer = CSeqList([CAcquire("m"), CLoad("val", "buf"), CRelease("m")])
        cmd = CParallel(producer, consumer)

        report = ownership_analysis(cmd, {"m": inv})
        assert "buf" in report.protected_resources.get("m", set())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
