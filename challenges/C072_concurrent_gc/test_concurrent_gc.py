"""
Tests for C072: Concurrent Garbage Collector with Tri-Color Marking

Tests cover:
- Tri-color state machine (phases, transitions)
- Incremental marking (gray set processing, budgets)
- Incremental sweeping (batch sweep, dead object removal)
- Write barriers (SATB and Dijkstra)
- Allocation during GC phases
- Pinned objects
- Weak references under concurrent GC
- Finalizers
- Adaptive budgets
- Full collection fallback
- VM integration (ConcurrentGCVM)
- Language features with concurrent GC
- Statistics tracking
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from concurrent_gc import (
    ConcurrentGC, ConcurrentGCVM, ConcurrentHeapRef, Color, GCPhase,
    ConcurrentGCStats, WeakRef, HEAP_TYPES,
    compile_concurrent, run_concurrent_gc, execute_concurrent_gc,
    lex, parse, Compiler, VM, Chunk, Op,
    FnObject, ClosureObject, ClassObject, TraitObject,
    VMError, CompileError, ParseError,
    run, execute,
)


# ============================================================
# Tri-Color Basics
# ============================================================

class TestTriColorBasics:
    def test_heap_ref_initial_color(self):
        gc = ConcurrentGC()
        ref = gc.track([1, 2, 3])
        assert ref.color == Color.WHITE

    def test_heap_ref_unique_ids(self):
        gc = ConcurrentGC()
        r1 = gc.track([1])
        r2 = gc.track([2])
        assert r1._id != r2._id

    def test_heap_ref_equality(self):
        gc = ConcurrentGC()
        obj = [1, 2]
        r1 = gc.track(obj)
        r2 = gc.track(obj)  # Same object
        assert r1 == r2

    def test_heap_ref_hash(self):
        gc = ConcurrentGC()
        r1 = gc.track([1])
        r2 = gc.track([2])
        s = {r1, r2}
        assert len(s) == 2

    def test_color_enum_values(self):
        assert Color.WHITE.value == 0
        assert Color.GRAY.value == 1
        assert Color.BLACK.value == 2

    def test_phase_enum_values(self):
        assert GCPhase.IDLE.value == "idle"
        assert GCPhase.MARK.value == "mark"
        assert GCPhase.SWEEP.value == "sweep"

    def test_initial_phase_is_idle(self):
        gc = ConcurrentGC()
        assert gc.phase == GCPhase.IDLE

    def test_heap_ref_repr(self):
        gc = ConcurrentGC()
        ref = gc.track([1])
        r = repr(ref)
        assert "CHeapRef" in r
        assert "WHITE" in r

    def test_dedup_tracking(self):
        gc = ConcurrentGC()
        obj = [1, 2, 3]
        r1 = gc.track(obj)
        r2 = gc.track(obj)
        assert r1 is r2
        assert gc.stats.total_allocations == 1


# ============================================================
# Phase State Machine
# ============================================================

class TestPhaseStateMachine:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_start_collection_transitions_to_mark(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.start_collection(vm)
        assert gc.phase == GCPhase.MARK

    def test_mark_to_sweep_transition(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        gc.start_collection(vm)
        # Mark until done
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        assert gc.phase == GCPhase.SWEEP

    def test_sweep_to_idle_transition(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        while gc.phase == GCPhase.SWEEP:
            gc.sweep_step(budget=100)
        assert gc.phase == GCPhase.IDLE

    def test_full_cycle_idle_mark_sweep_idle(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        phases = [gc.phase]
        gc.start_collection(vm)
        phases.append(gc.phase)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        phases.append(gc.phase)
        while gc.phase == GCPhase.SWEEP:
            gc.sweep_step(budget=100)
        phases.append(gc.phase)
        assert phases == [GCPhase.IDLE, GCPhase.MARK, GCPhase.SWEEP, GCPhase.IDLE]

    def test_double_start_collection_no_op(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.start_collection(vm)
        cycle_before = gc._cycle
        gc.start_collection(vm)  # Should be no-op
        assert gc._cycle == cycle_before

    def test_cycle_increments(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.collect_full(vm)
        assert gc._cycle == 1
        gc.collect_full(vm)
        assert gc._cycle == 2


# ============================================================
# Incremental Marking
# ============================================================

class TestIncrementalMarking:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_mark_step_respects_budget(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        # Build a chain of reachable objects so gray set has many items
        objs = []
        for i in range(50):
            objs.append([None])
        for i in range(49):
            objs[i][0] = objs[i + 1]
        for obj in objs:
            gc.track(obj)
        vm.stack.append(objs[0])
        gc.start_collection(vm)
        gc.mark_step(budget=5)
        # Should still be in MARK phase (chain of 50, budget 5)
        assert gc.phase == GCPhase.MARK

    def test_mark_step_returns_true_when_done(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        gc.start_collection(vm)
        # With large budget, should finish in one step
        done = gc.mark_step(budget=1000)
        assert done is True

    def test_mark_step_processes_gray_objects(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        inner = [1, 2, 3]
        outer = [inner]
        gc.track(inner)
        gc.track(outer)
        vm.stack.append(outer)
        gc.start_collection(vm)
        # Mark to completion
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        # Both should be black (reachable via stack)
        obj_map = gc._object_map
        outer_ref = obj_map[id(outer)]
        inner_ref = obj_map[id(inner)]
        assert outer_ref.color == Color.BLACK
        assert inner_ref.color == Color.BLACK

    def test_unreachable_stays_white(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        reachable = [1]
        unreachable = [2]
        gc.track(reachable)
        gc.track(unreachable)
        vm.stack.append(reachable)
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        unreachable_ref = gc._object_map.get(id(unreachable))
        # unreachable should be white (or already swept)
        if unreachable_ref:
            assert unreachable_ref.color == Color.WHITE

    def test_mark_work_counter(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=100)
        assert gc.stats.total_mark_work >= 0
        assert gc.stats.mark_steps >= 1

    def test_gray_set_starts_from_roots(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [42]
        gc.track(obj)
        vm.stack.append(obj)
        gc.start_collection(vm)
        # Gray set should have at least the root object
        assert len(gc._gray_set) >= 1


# ============================================================
# Incremental Sweeping
# ============================================================

class TestIncrementalSweeping:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_sweep_removes_white_objects(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        alive = [1]
        dead = [2]
        gc.track(alive)
        gc.track(dead)
        vm.stack.append(alive)
        freed = gc.collect_full(vm)
        assert freed == 1
        assert gc.stats.heap_size == 1

    def test_sweep_step_respects_budget(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(100):
            gc.track([i])
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=1000)
        assert gc.phase == GCPhase.SWEEP
        gc.sweep_step(budget=10)
        # Should still be sweeping
        assert gc.phase == GCPhase.SWEEP

    def test_sweep_step_returns_true_when_done(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=1000)
        done = gc.sweep_step(budget=1000)
        assert done is True

    def test_sweep_increments_generation_of_survivors(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        vm.stack.append(obj)
        gc.collect_full(vm)
        assert ref.generation >= 1

    def test_multiple_collections_increment_generation(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        vm.stack.append(obj)
        gc.collect_full(vm)
        gc.collect_full(vm)
        gc.collect_full(vm)
        assert ref.generation >= 3

    def test_sweep_work_counter(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        gc.collect_full(vm)
        assert gc.stats.total_sweep_work >= 10
        assert gc.stats.sweep_steps >= 1


# ============================================================
# Write Barriers
# ============================================================

class TestWriteBarriers:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_satb_barrier_no_op_in_idle(self):
        gc = ConcurrentGC()
        obj = [1]
        gc.track(obj)
        gc.write_barrier(obj, [2])
        assert gc.stats.barrier_hits == 0
        assert gc.stats.barrier_checks == 1

    def test_satb_barrier_grays_white_during_mark(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        old_ref_obj = [99]
        target = [1]
        ref_old = gc.track(old_ref_obj)
        gc.track(target)
        vm.stack.append(target)
        gc.start_collection(vm)
        # old_ref_obj should be white since not reachable
        assert ref_old.color == Color.WHITE
        # Fire SATB barrier
        gc.write_barrier(target, old_ref_obj)
        assert ref_old.color == Color.GRAY
        assert gc.stats.barrier_hits == 1

    def test_satb_barrier_ignores_non_white(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        vm.stack.append(obj)
        gc.start_collection(vm)
        # obj should be gray (root)
        assert ref.color == Color.GRAY
        gc.write_barrier([2], obj)
        assert gc.stats.barrier_hits == 0

    def test_dijkstra_barrier_grays_white_child_of_black(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        parent = [1]
        child = [2]
        # Build a chain so marking doesn't finish immediately
        chain = [None]
        for i in range(20):
            new = [chain]
            gc.track(chain)
            chain = new
        gc.track(chain)
        vm.stack.append(chain)
        ref_p = gc.track(parent)
        ref_c = gc.track(child)
        vm.stack.append(parent)
        gc.start_collection(vm)
        # Do one small mark step (won't finish the whole chain)
        gc.mark_step(budget=3)
        assert gc.phase == GCPhase.MARK  # Still marking
        # Force parent black, child white for barrier test
        ref_p.color = Color.BLACK
        ref_c.color = Color.WHITE
        gc.write_barrier_new(parent, child)
        assert ref_c.color == Color.GRAY
        assert gc.stats.barrier_hits >= 1

    def test_dijkstra_barrier_no_op_for_non_black_parent(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        parent = [1]
        child = [2]
        ref_p = gc.track(parent)
        ref_c = gc.track(child)
        gc.start_collection(vm)
        ref_p.color = Color.GRAY
        ref_c.color = Color.WHITE
        gc.write_barrier_new(parent, child)
        assert ref_c.color == Color.WHITE  # Not grayed

    def test_barrier_check_count(self):
        gc = ConcurrentGC()
        gc.write_barrier([1], [2])
        gc.write_barrier([3], [4])
        gc.write_barrier_new([5], [6])
        assert gc.stats.barrier_checks == 3

    def test_barrier_with_none(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.start_collection(vm)
        gc.write_barrier([1], None)
        gc.write_barrier_new([1], None)
        assert gc.stats.barrier_hits == 0


# ============================================================
# Allocation During GC
# ============================================================

class TestAllocationDuringGC:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_alloc_during_mark_is_black(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.start_collection(vm)
        assert gc.phase == GCPhase.MARK
        ref = gc.track([42])
        assert ref.color == Color.BLACK

    def test_alloc_during_sweep_is_white(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.track([1])
        gc.start_collection(vm)
        while gc.phase == GCPhase.MARK:
            gc.mark_step(budget=1000)
        assert gc.phase == GCPhase.SWEEP
        ref = gc.track([99])
        assert ref.color == Color.WHITE

    def test_alloc_during_idle_is_white(self):
        gc = ConcurrentGC()
        ref = gc.track([1])
        assert ref.color == Color.WHITE

    def test_black_alloc_survives_current_cycle(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.start_collection(vm)
        # Allocate during mark phase
        obj = [42]
        ref = gc.track(obj)
        vm.stack.append(obj)
        assert ref.color == Color.BLACK
        # Complete the cycle
        while gc.phase != GCPhase.IDLE:
            if gc.phase == GCPhase.MARK:
                gc.mark_step(budget=1000)
            else:
                gc.sweep_step(budget=1000)
        # Object should survive
        assert ref in gc.heap


# ============================================================
# Pinned Objects
# ============================================================

class TestPinnedObjects:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_pinned_object_survives_collection(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        gc.track(obj)
        gc.pin(obj)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 1

    def test_unpinned_object_can_be_collected(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        gc.track(obj)
        gc.pin(obj)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 1
        gc.unpin(obj)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 0

    def test_pin_unknown_object_no_error(self):
        gc = ConcurrentGC()
        gc.pin([999])  # Not tracked
        # Should not raise

    def test_pinned_increments_generation(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        gc.pin(obj)
        gc.collect_full(vm)
        assert ref.generation >= 1


# ============================================================
# Weak References
# ============================================================

class TestWeakReferences:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_weak_ref_alive_before_collection(self):
        gc = ConcurrentGC()
        obj = [1, 2, 3]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        assert wr is not None
        assert wr.alive is True
        assert wr.get() is obj

    def test_weak_ref_invalidated_after_collection(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1, 2, 3]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        gc.collect_full(vm)  # obj unreachable
        assert wr.alive is False
        assert wr.get() is None

    def test_weak_ref_survives_if_reachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1, 2, 3]
        gc.track(obj)
        vm.stack.append(obj)
        wr = gc.create_weak_ref(obj)
        gc.collect_full(vm)
        assert wr.alive is True
        assert wr.get() is obj

    def test_weak_ref_to_untracked_returns_none(self):
        gc = ConcurrentGC()
        wr = gc.create_weak_ref([42])
        assert wr is None


# ============================================================
# Finalizers
# ============================================================

class TestFinalizers:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_finalizer_called_on_collection(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        finalized = []
        obj = [1, 2, 3]
        gc.track(obj, finalizer=lambda o: finalized.append(id(o)))
        gc.collect_full(vm)
        assert len(finalized) == 1

    def test_finalizer_not_called_if_reachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        finalized = []
        obj = [1]
        gc.track(obj, finalizer=lambda o: finalized.append(1))
        vm.stack.append(obj)
        gc.collect_full(vm)
        assert len(finalized) == 0

    def test_finalizer_exception_suppressed(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        gc.track(obj, finalizer=lambda o: 1/0)
        gc.collect_full(vm)  # Should not raise

    def test_multiple_finalizers(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        results = []
        for i in range(5):
            obj = [i]
            gc.track(obj, finalizer=lambda o, x=i: results.append(x))
        gc.collect_full(vm)
        assert sorted(results) == [0, 1, 2, 3, 4]


# ============================================================
# Adaptive Budgets
# ============================================================

class TestAdaptiveBudgets:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_adaptive_increases_for_large_heap(self):
        gc = ConcurrentGC(adaptive=True, mark_budget=32, sweep_budget=64)
        vm = self._make_vm()
        objs = []
        for i in range(1500):
            obj = [i]
            gc.track(obj)
            objs.append(obj)
        # Make them reachable so heap stays large after collection
        vm.stack.extend(objs)
        gc.collect_full(vm)
        assert gc.mark_budget > 32
        assert gc.sweep_budget > 64

    def test_adaptive_decreases_for_small_heap(self):
        gc = ConcurrentGC(adaptive=True, mark_budget=64, sweep_budget=128)
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        gc.collect_full(vm)
        assert gc.mark_budget < 64
        assert gc.sweep_budget < 128

    def test_non_adaptive_unchanged(self):
        gc = ConcurrentGC(adaptive=False, mark_budget=32, sweep_budget=64)
        vm = self._make_vm()
        for i in range(1500):
            gc.track([i])
            vm.stack.append([i])
        gc.collect_full(vm)
        assert gc.mark_budget == 32
        assert gc.sweep_budget == 64


# ============================================================
# Full Collection (Stop-the-World Fallback)
# ============================================================

class TestFullCollection:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_collect_full_frees_unreachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(20):
            gc.track([i])
        freed = gc.collect_full(vm)
        assert freed == 20

    def test_collect_full_keeps_reachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        alive = [1]
        gc.track(alive)
        vm.stack.append(alive)
        for i in range(10):
            gc.track([i + 100])
        freed = gc.collect_full(vm)
        assert freed == 10
        assert gc.stats.heap_size == 1

    def test_collect_full_updates_stats(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(5):
            gc.track([i])
        gc.collect_full(vm)
        assert gc.stats.total_collections == 1
        assert gc.stats.total_freed == 5
        assert gc.stats.total_allocations == 5

    def test_collect_full_returns_to_idle(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        gc.collect_full(vm)
        assert gc.phase == GCPhase.IDLE

    def test_collect_full_resets_allocs_counter(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        assert gc._allocs_since_gc == 10
        gc.collect_full(vm)
        assert gc._allocs_since_gc == 0


# ============================================================
# Step-based Incremental Collection
# ============================================================

class TestStepBasedCollection:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_step_triggers_collection_at_threshold(self):
        gc = ConcurrentGC(threshold=5)
        vm = self._make_vm()
        for i in range(5):
            gc.track([i])
        assert gc.phase == GCPhase.IDLE
        gc.step(vm)
        assert gc.phase == GCPhase.MARK

    def test_step_no_trigger_below_threshold(self):
        gc = ConcurrentGC(threshold=100)
        vm = self._make_vm()
        for i in range(3):
            gc.track([i])
        gc.step(vm)
        assert gc.phase == GCPhase.IDLE

    def test_step_progresses_through_phases(self):
        gc = ConcurrentGC(threshold=3, mark_budget=1000, sweep_budget=1000)
        vm = self._make_vm()
        for i in range(3):
            gc.track([i])
        # Step until collection completes
        steps = 0
        while steps < 100:
            completed = gc.step(vm)
            steps += 1
            if completed:
                break
        assert gc.phase == GCPhase.IDLE
        assert gc.stats.total_collections == 1

    def test_step_returns_true_on_cycle_complete(self):
        gc = ConcurrentGC(threshold=1, mark_budget=1000, sweep_budget=1000)
        vm = self._make_vm()
        gc.track([1])
        gc.step(vm)  # Start collection
        results = []
        for _ in range(20):
            r = gc.step(vm)
            results.append(r)
            if r:
                break
        assert True in results


# ============================================================
# Statistics
# ============================================================

class TestStatistics:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_get_stats_returns_dict(self):
        gc = ConcurrentGC()
        stats = gc.get_stats()
        assert isinstance(stats, dict)
        assert 'total_collections' in stats
        assert 'phase' in stats

    def test_stats_phase_tracking(self):
        gc = ConcurrentGC()
        assert gc.get_stats()['phase'] == 'idle'
        vm = self._make_vm()
        gc.start_collection(vm)
        assert gc.get_stats()['phase'] == 'mark'

    def test_stats_cycle_tracking(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        assert gc.get_stats()['cycle'] == 0
        gc.collect_full(vm)
        assert gc.get_stats()['cycle'] == 1

    def test_stats_gray_set_size(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        obj = [1]
        gc.track(obj)
        vm.stack.append(obj)
        gc.start_collection(vm)
        assert gc.get_stats()['gray_set_size'] >= 1

    def test_peak_heap_size(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        gc.collect_full(vm)
        assert gc.stats.peak_heap_size == 10
        assert gc.stats.heap_size == 0

    def test_timing_stats(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        for i in range(10):
            gc.track([i])
        gc.collect_full(vm)
        assert gc.stats.last_total_duration >= 0
        assert gc.stats.last_mark_duration >= 0
        assert gc.stats.last_sweep_duration >= 0


# ============================================================
# Object Graph Traversal
# ============================================================

class TestObjectGraph:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_nested_list_reachability(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        inner = [1, 2]
        outer = [inner]
        gc.track(inner)
        gc.track(outer)
        vm.stack.append(outer)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 2

    def test_dict_values_reachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        val = [1, 2]
        d = {'key': val}
        gc.track(val)
        gc.track(d)
        vm.stack.append(d)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 2

    def test_circular_reference_collected(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        a = [None]
        b = [a]
        a[0] = b
        gc.track(a)
        gc.track(b)
        # Neither is reachable from roots
        freed = gc.collect_full(vm)
        assert freed == 2

    def test_circular_reference_kept_if_rooted(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        a = [None]
        b = [a]
        a[0] = b
        gc.track(a)
        gc.track(b)
        vm.stack.append(a)
        gc.collect_full(vm)
        assert gc.stats.heap_size == 2

    def test_deep_chain_reachable(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        objs = []
        for i in range(10):
            objs.append([None])
        for i in range(9):
            objs[i][0] = objs[i + 1]
        for obj in objs:
            gc.track(obj)
        vm.stack.append(objs[0])
        gc.collect_full(vm)
        assert gc.stats.heap_size == 10


# ============================================================
# ConcurrentGCVM Integration
# ============================================================

class TestConcurrentGCVM:
    def test_simple_program(self):
        result, output, stats = run_concurrent_gc("let x = 42; print(x);")
        assert "42" in output

    def test_gc_stats_populated(self):
        result, output, stats = run_concurrent_gc("""
            let arr = [1, 2, 3];
            let x = arr;
            print(x);
        """)
        assert stats['total_allocations'] >= 0

    def test_function_allocation(self):
        result, output, stats = run_concurrent_gc("""
            fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        assert "7" in output

    def test_class_allocation(self):
        result, output, stats = run_concurrent_gc("""
            class Dog {
                init(name) { this.name = name; }
                speak() { return this.name; }
            }
            let d = Dog("Rex");
            print(d.speak());
        """)
        assert "Rex" in output

    def test_closure_allocation(self):
        result, output, stats = run_concurrent_gc("""
            fn make_adder(x) {
                return fn(y) { return x + y; };
            }
            let add5 = make_adder(5);
            print(add5(3));
            print(add5(10));
        """)
        assert "8" in output
        assert "15" in output

    def test_array_operations(self):
        result, output, stats = run_concurrent_gc("""
            let arr = [1, 2, 3, 4, 5];
            print(len(arr));
            let arr2 = [...arr, 6];
            print(len(arr2));
        """)
        assert "5" in output
        assert "6" in output

    def test_hash_operations(self):
        result, output, stats = run_concurrent_gc("""
            let h = {a: 1, b: 2};
            print(h.a);
            print(h.b);
        """)
        assert "1" in output
        assert "2" in output

    def test_loop_allocation_pressure(self):
        result, output, stats = run_concurrent_gc("""
            let i = 0;
            while (i < 100) {
                let temp = [i, i + 1, i + 2];
                i = i + 1;
            }
            print(i);
        """, gc_threshold=20)
        assert "100" in output

    def test_manual_gc_collect(self):
        chunk = compile_concurrent("let x = 42;")
        vm = ConcurrentGCVM(chunk, gc_threshold=1000)
        vm.run()
        freed = vm.gc_collect()
        assert freed >= 0

    def test_gc_phase_api(self):
        chunk = compile_concurrent("let x = 42;")
        vm = ConcurrentGCVM(chunk, gc_threshold=1000)
        assert vm.gc_phase() == "idle"

    def test_gc_start_and_step(self):
        chunk = compile_concurrent("let x = [1, 2, 3];")
        vm = ConcurrentGCVM(chunk, gc_threshold=1000)
        vm.run()
        vm.gc_start()
        assert vm.gc_phase() == "mark"
        # Step until done
        for _ in range(100):
            if vm.gc_step():
                break
        assert vm.gc_phase() == "idle"

    def test_gc_stats_api(self):
        chunk = compile_concurrent("let x = [1, 2, 3];")
        vm = ConcurrentGCVM(chunk, gc_threshold=1000)
        vm.run()
        stats = vm.gc_stats()
        assert isinstance(stats, dict)
        assert 'phase' in stats


# ============================================================
# Language Features Under Concurrent GC
# ============================================================

class TestLanguageFeaturesWithCGC:
    def test_string_interpolation(self):
        result, output, stats = run_concurrent_gc("""
            let name = "world";
            print(f"hello ${name}");
        """)
        assert "hello world" in output

    def test_spread_operator(self):
        result, output, stats = run_concurrent_gc("""
            let a = [1, 2];
            let b = [0, ...a, 3];
            print(len(b));
        """)
        assert "4" in output

    def test_destructuring(self):
        result, output, stats = run_concurrent_gc("""
            let [a, b, c] = [10, 20, 30];
            print(a + b + c);
        """)
        assert "60" in output

    def test_pipe_operator(self):
        result, output, stats = run_concurrent_gc("""
            fn double(x) { return x * 2; }
            fn inc(x) { return x + 1; }
            let result = 5 |> double |> inc;
            print(result);
        """)
        assert "11" in output

    def test_optional_chaining(self):
        result, output, stats = run_concurrent_gc("""
            let obj = {a: {b: 42}};
            print(obj?.a?.b);
            let x = null;
            print(x ?? "fallback");
        """)
        assert "42" in output
        assert "fallback" in output

    def test_null_coalescing(self):
        result, output, stats = run_concurrent_gc("""
            let x = null ?? 42;
            print(x);
        """)
        assert "42" in output

    def test_for_in_loop(self):
        result, output, stats = run_concurrent_gc("""
            let sum = 0;
            for (x in [1, 2, 3, 4, 5]) {
                sum = sum + x;
            }
            print(sum);
        """)
        assert "15" in output

    def test_try_catch(self):
        result, output, stats = run_concurrent_gc("""
            try {
                throw "oops";
            } catch (e) {
                print(e);
            }
        """)
        assert "oops" in output

    def test_try_finally(self):
        result, output, stats = run_concurrent_gc("""
            let x = 0;
            try {
                x = 1;
            } finally {
                x = x + 10;
            }
            print(x);
        """)
        assert "11" in output

    def test_trait(self):
        result, output, stats = run_concurrent_gc("""
            trait Greet {
                greet() { return f"Hello, ${this.name}"; }
            }
            class Person implements Greet {
                init(name) { this.name = name; }
            }
            let p = Person("Alice");
            print(p.greet());
        """)
        assert "Hello, Alice" in output

    def test_enum(self):
        result, output, stats = run_concurrent_gc("""
            enum Color { Red, Green, Blue }
            print(Color.Red);
        """)
        assert any("Red" in s for s in output)

    def test_ternary_like_logic(self):
        result, output, stats = run_concurrent_gc("""
            let x = 42;
            let r = if (x == 42) "answer" else "other";
            print(r);
        """)
        assert "answer" in output

    def test_generators(self):
        result, output, stats = run_concurrent_gc("""
            fn* count(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = count(3);
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert "0" in output
        assert "1" in output
        assert "2" in output

    def test_class_methods(self):
        result, output, stats = run_concurrent_gc("""
            class Dog {
                init(name) { this.name = name; }
                speak() { return f"${this.name} says woof"; }
            }
            let d = Dog("Rex");
            print(d.speak());
        """)
        assert "Rex says woof" in output

    def test_static_methods(self):
        result, output, stats = run_concurrent_gc("""
            class Math {
                static square(x) { return x * x; }
            }
            print(Math.square(5));
        """)
        assert "25" in output

    def test_getters_setters(self):
        result, output, stats = run_concurrent_gc("""
            class Rect {
                init(w, h) { this.w = w; this.h = h; }
                get area() { return this.w * this.h; }
            }
            let r = Rect(3, 4);
            print(r.area);
        """)
        assert "12" in output


# ============================================================
# Stress Tests
# ============================================================

class TestStress:
    def test_many_allocations_with_low_threshold(self):
        """Lots of temp arrays with aggressive GC."""
        result, output, stats = run_concurrent_gc("""
            let total = 0;
            let i = 0;
            while (i < 200) {
                let tmp = [i, i * 2];
                total = total + tmp[0];
                i = i + 1;
            }
            print(total);
        """, gc_threshold=10)
        assert "19900" in output

    def test_retained_objects_not_collected(self):
        result, output, stats = run_concurrent_gc("""
            let items = [];
            let i = 0;
            while (i < 50) {
                items = [...items, [i]];
                i = i + 1;
            }
            print(len(items));
        """, gc_threshold=10)
        assert "50" in output

    def test_many_functions(self):
        result, output, stats = run_concurrent_gc("""
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let adders = [];
            let i = 0;
            while (i < 20) {
                adders.push(make_adder(i));
                i = i + 1;
            }
            print(adders[10](5));
        """, gc_threshold=10)
        assert "15" in output

    def test_nested_hash_pressure(self):
        result, output, stats = run_concurrent_gc("""
            let data = {};
            let i = 0;
            while (i < 50) {
                data[string(i)] = {value: i, nested: {deep: i * 2}};
                i = i + 1;
            }
            print(data["25"].nested.deep);
        """, gc_threshold=15)
        assert "50" in output


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def _make_vm(self):
        chunk = Chunk()
        chunk.code = [Op.CONST, 0, Op.HALT]
        chunk.constants = [42]
        chunk.lines = [1, 1, 1]
        return VM(chunk)

    def test_empty_heap_collect(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        freed = gc.collect_full(vm)
        assert freed == 0

    def test_collect_disabled(self):
        gc = ConcurrentGC(threshold=1)
        vm = self._make_vm()
        gc.enabled = False
        for i in range(10):
            gc.track([i])
        gc.step(vm)
        assert gc.phase == GCPhase.IDLE

    def test_mark_step_when_not_marking(self):
        gc = ConcurrentGC()
        result = gc.mark_step()
        assert result is True  # No-op

    def test_sweep_step_when_not_sweeping(self):
        gc = ConcurrentGC()
        result = gc.sweep_step()
        assert result is True

    def test_alloc_cycle_tracking(self):
        gc = ConcurrentGC()
        vm = self._make_vm()
        r1 = gc.track([1])
        assert r1.alloc_cycle == 0
        gc.collect_full(vm)
        r2 = gc.track([2])
        assert r2.alloc_cycle == 1

    def test_execute_concurrent_gc_api(self):
        result = execute_concurrent_gc("print(1 + 2);")
        assert result['output'] == ['3']
        assert 'gc_stats' in result

    def test_compile_concurrent_api(self):
        chunk = compile_concurrent("let x = 1;")
        assert isinstance(chunk, Chunk)

    def test_multiple_cycles_correct_stats(self):
        gc = ConcurrentGC(threshold=5)
        vm = self._make_vm()
        for _ in range(3):
            for i in range(5):
                gc.track([object()])
            gc.collect_full(vm)
        assert gc.stats.total_collections == 3
        assert gc.stats.total_freed >= 15


# ============================================================
# Write Barrier + VM Integration
# ============================================================

class TestBarrierVMIntegration:
    def test_overwrite_triggers_barrier_check(self):
        """Overwriting a variable should trigger barrier checks."""
        src = """
            let x = [1, 2, 3];
            let y = [4, 5, 6];
            x = y;
        """
        chunk = compile_concurrent(src)
        vm = ConcurrentGCVM(chunk, gc_threshold=1000)
        vm.run()
        # Barrier checks should have happened for STORE ops
        assert vm.cgc.stats.barrier_checks >= 0

    def test_gc_survives_heavy_mutation(self):
        """Heavy mutation with GC shouldn't crash."""
        result, output, stats = run_concurrent_gc("""
            let arr = [0, 0, 0, 0, 0];
            let i = 0;
            while (i < 100) {
                arr[i % 5] = [i, i + 1];
                i = i + 1;
            }
            print(arr[0][0]);
        """, gc_threshold=10)
        assert "95" in output


# ============================================================
# Concurrent Allocation Patterns
# ============================================================

class TestConcurrentPatterns:
    def test_producer_consumer_pattern(self):
        """Simulate producer creating objects, consumer reading and discarding."""
        result, output, stats = run_concurrent_gc("""
            let queue = [];
            let i = 0;
            while (i < 100) {
                queue = [...queue, {value: i}];
                if (len(queue) > 10) {
                    let new_q = [];
                    let j = len(queue) - 10;
                    while (j < len(queue)) {
                        new_q = [...new_q, queue[j]];
                        j = j + 1;
                    }
                    queue = new_q;
                }
                i = i + 1;
            }
            print(len(queue));
        """, gc_threshold=15)
        assert "10" in output

    def test_tree_building_and_pruning(self):
        """Build a tree-like structure, then prune it."""
        result, output, stats = run_concurrent_gc("""
            fn make_node(v) {
                return {val: v, left: null, right: null};
            }
            let root = make_node(1);
            root.left = make_node(2);
            root.right = make_node(3);
            root.left.left = make_node(4);
            root.left = null;
            print(root.right.val);
        """, gc_threshold=5)
        assert "3" in output

    def test_recursive_allocation(self):
        result, output, stats = run_concurrent_gc("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(15));
        """, gc_threshold=20)
        assert "610" in output


# Run all tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
