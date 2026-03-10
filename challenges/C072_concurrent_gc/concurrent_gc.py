"""
C072: Concurrent Garbage Collector with Tri-Color Marking

An incremental, concurrent-safe garbage collector that uses tri-color marking
and write barriers to allow collection without stop-the-world pauses.

Key features:
- Tri-color marking: White (unvisited), Gray (to scan), Black (done)
- Write barriers: Snapshot-at-the-beginning (SATB) barrier prevents lost objects
- Incremental marking: Process N gray objects per VM step (configurable)
- Incremental sweeping: Sweep K objects per step
- GC phases: IDLE -> MARK -> SWEEP -> IDLE (state machine)
- Concurrent-safe: Mutations during GC are safe via write barriers
- Back-pressure: Auto-adjust mark/sweep budget based on allocation rate
- Generational hints: Young/old partitioning for efficiency
- Finalizer scheduling: Finalizers run between phases (not during)
- Statistics: Phase timing, barrier hits, incremental step counts

Composes: C071 (garbage_collector) -- extends with concurrent collection
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C071_garbage_collector'))

from garbage_collector import (
    HeapRef, WeakRef, GCStats, HEAP_TYPES,
    GarbageCollector, GCVM,
    compile_source, run_with_gc, execute_with_gc,
    # Re-export VM types
    lex, parse, Parser, Compiler, VM, Chunk, Op,
    FnObject, ClosureObject, GeneratorObject, PromiseObject,
    AsyncCoroutine, AsyncGeneratorObject, TraitObject, ClassObject,
    BoundMethod, EnumObject, EnumVariant, NativeFunction, NativeModule,
    VMError, CompileError, ParseError,
    run, execute,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, Dict, List
from enum import Enum


# ---------- GC Phase State Machine ----------

class GCPhase(Enum):
    IDLE = "idle"
    MARK = "mark"
    SWEEP = "sweep"


# ---------- Tri-Color States ----------

class Color(Enum):
    WHITE = 0   # Not yet visited (candidate for collection)
    GRAY = 1    # Visited but children not scanned
    BLACK = 2   # Visited and all children scanned


# ---------- Concurrent GC Stats ----------

@dataclass
class ConcurrentGCStats:
    # Collection counts
    total_collections: int = 0
    total_freed: int = 0
    total_allocations: int = 0
    heap_size: int = 0
    peak_heap_size: int = 0

    # Incremental stats
    mark_steps: int = 0        # Number of incremental mark steps
    sweep_steps: int = 0       # Number of incremental sweep steps
    total_mark_work: int = 0   # Total objects marked across all cycles
    total_sweep_work: int = 0  # Total objects swept across all cycles

    # Write barrier stats
    barrier_hits: int = 0      # Times write barrier triggered re-graying
    barrier_checks: int = 0    # Times write barrier was checked

    # Phase timing (seconds)
    last_mark_duration: float = 0.0
    last_sweep_duration: float = 0.0
    last_total_duration: float = 0.0

    # Generational
    gen0_collections: int = 0
    gen1_collections: int = 0


# ---------- Concurrent Heap Ref ----------

class ConcurrentHeapRef:
    """Extended HeapRef with tri-color support."""
    __slots__ = ('obj', 'color', 'generation', 'weak_refs', 'finalizer',
                 '_id', 'alloc_cycle', 'pinned')
    _next_id = 0

    def __init__(self, obj, finalizer=None, alloc_cycle=0):
        self.obj = obj
        self.color = Color.WHITE
        self.generation = 0
        self.weak_refs = []
        self.finalizer = finalizer
        ConcurrentHeapRef._next_id += 1
        self._id = ConcurrentHeapRef._next_id
        self.alloc_cycle = alloc_cycle  # Which GC cycle allocated this
        self.pinned = False  # Pinned objects are never collected

    def __repr__(self):
        return f"CHeapRef({self._id}, {self.color.name}, gen={self.generation})"

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, ConcurrentHeapRef) and self._id == other._id


# ---------- Concurrent Garbage Collector ----------

class ConcurrentGC:
    """Incremental tri-color mark-sweep garbage collector.

    The collector runs in phases:
    1. IDLE: No collection in progress. New allocations are white.
    2. MARK: Incrementally process gray objects. Write barriers active.
       New allocations during mark are colored black (already visited).
    3. SWEEP: Incrementally sweep white objects (unreachable).
       New allocations during sweep are white (for next cycle).
    """

    def __init__(self, threshold=256, mark_budget=32, sweep_budget=64,
                 gen1_threshold=5, adaptive=True):
        self.heap: Set[ConcurrentHeapRef] = set()
        self.stats = ConcurrentGCStats()
        self.threshold = threshold
        self.mark_budget = mark_budget    # Objects to mark per step
        self.sweep_budget = sweep_budget  # Objects to sweep per step
        self.gen1_threshold = gen1_threshold
        self.adaptive = adaptive          # Auto-adjust budgets
        self.enabled = True

        # Phase state
        self.phase = GCPhase.IDLE
        self._gray_set: List[ConcurrentHeapRef] = []  # Work list for marking
        self._sweep_iter: Optional[list] = None   # Iterator for sweeping
        self._sweep_idx: int = 0
        self._cycle: int = 0
        self._allocs_since_gc = 0
        self._phase_start_time: float = 0.0
        self._mark_start_time: float = 0.0

        # Object map: id(obj) -> ConcurrentHeapRef
        self._object_map: Dict[int, ConcurrentHeapRef] = {}

        # Finalizer queue (run between phases)
        self._finalizer_queue: List[ConcurrentHeapRef] = []

        # Write barrier log (SATB)
        self._satb_buffer: List[Any] = []

    # ---------- Allocation ----------

    def track(self, obj, finalizer=None) -> ConcurrentHeapRef:
        """Register an object on the managed heap."""
        obj_id = id(obj)
        if obj_id in self._object_map:
            return self._object_map[obj_id]

        ref = ConcurrentHeapRef(obj, finalizer=finalizer, alloc_cycle=self._cycle)

        # Color depends on phase:
        # - During MARK: new objects are black (already considered visited)
        #   This prevents them from being collected in the current cycle
        # - During IDLE/SWEEP: new objects are white (for next cycle)
        if self.phase == GCPhase.MARK:
            ref.color = Color.BLACK
        else:
            ref.color = Color.WHITE

        self.heap.add(ref)
        self._object_map[obj_id] = ref
        self.stats.total_allocations += 1
        self.stats.heap_size += 1
        if self.stats.heap_size > self.stats.peak_heap_size:
            self.stats.peak_heap_size = self.stats.heap_size
        self._allocs_since_gc += 1
        return ref

    def pin(self, obj):
        """Pin an object so it's never collected."""
        obj_id = id(obj)
        if obj_id in self._object_map:
            self._object_map[obj_id].pinned = True

    def unpin(self, obj):
        """Unpin a previously pinned object."""
        obj_id = id(obj)
        if obj_id in self._object_map:
            self._object_map[obj_id].pinned = False

    def create_weak_ref(self, obj) -> Optional[WeakRef]:
        """Create a weak reference to a tracked object."""
        obj_id = id(obj)
        if obj_id not in self._object_map:
            return None
        ref = self._object_map[obj_id]
        wr = WeakRef.__new__(WeakRef)
        wr._heap_ref = ref
        wr._alive = True
        ref.weak_refs.append(wr)
        return wr

    # ---------- Write Barrier (SATB) ----------

    def write_barrier(self, target_obj, old_ref_obj):
        """Snapshot-at-the-beginning write barrier.

        Called when a reference in target_obj is being overwritten.
        old_ref_obj is the value being overwritten (the old referent).

        If we're in MARK phase and the old referent is white, gray it
        to ensure it gets scanned (preserving the snapshot invariant).
        """
        self.stats.barrier_checks += 1

        if self.phase != GCPhase.MARK:
            return

        if old_ref_obj is None:
            return

        old_id = id(old_ref_obj)
        if old_id in self._object_map:
            old_ref = self._object_map[old_id]
            if old_ref.color == Color.WHITE:
                # Re-gray: the old referent might still be reachable through
                # another path, and we need to check
                old_ref.color = Color.GRAY
                self._gray_set.append(old_ref)
                self.stats.barrier_hits += 1

    def write_barrier_new(self, target_obj, new_ref_obj):
        """Dijkstra-style write barrier (alternative/complementary).

        When a black object gets a reference to a white object,
        gray the white object to maintain the tri-color invariant:
        no black object points directly to a white object.
        """
        self.stats.barrier_checks += 1

        if self.phase != GCPhase.MARK:
            return

        if new_ref_obj is None:
            return

        target_id = id(target_obj)
        new_id = id(new_ref_obj)

        target_ref = self._object_map.get(target_id)
        new_ref = self._object_map.get(new_id)

        if target_ref and new_ref:
            if target_ref.color == Color.BLACK and new_ref.color == Color.WHITE:
                new_ref.color = Color.GRAY
                self._gray_set.append(new_ref)
                self.stats.barrier_hits += 1

    # ---------- Root Scanning ----------

    def _find_roots(self, vm: VM) -> List[ConcurrentHeapRef]:
        """Find all root objects from VM state, return as gray set."""
        root_ids = set()
        self._scan_value_ids(vm.env, root_ids)
        for val in vm.stack:
            self._scan_value_ids(val, root_ids)
        for frame in vm.call_stack:
            if isinstance(frame, (list, tuple)):
                for item in frame:
                    self._scan_value_ids(item, root_ids)
            else:
                self._scan_value_ids(frame, root_ids)
        for handler in vm.handler_stack:
            if isinstance(handler, (list, tuple)):
                for item in handler:
                    self._scan_value_ids(item, root_ids)
            else:
                self._scan_value_ids(handler, root_ids)
        for item in vm._async_queue:
            if isinstance(item, (list, tuple)):
                for v in item:
                    self._scan_value_ids(v, root_ids)
            else:
                self._scan_value_ids(item, root_ids)
        if vm._current_async is not None:
            self._scan_value_ids(vm._current_async, root_ids)

        # Convert to HeapRefs
        roots = []
        for ref in self.heap:
            if id(ref.obj) in root_ids:
                roots.append(ref)
        return roots

    def _scan_value_ids(self, val, ids: set):
        """Collect ids of all heap-type values reachable from val (shallow)."""
        val_id = id(val)
        if val_id in ids:
            return
        if val_id in self._object_map:
            ids.add(val_id)
        if isinstance(val, dict):
            ids.add(val_id)
            for k, v in val.items():
                if isinstance(v, HEAP_TYPES) and id(v) not in ids:
                    ids.add(id(v))
        elif isinstance(val, (list, tuple)):
            ids.add(val_id)
            for item in val:
                if isinstance(item, HEAP_TYPES) and id(item) not in ids:
                    ids.add(id(item))

    # ---------- Incremental Mark ----------

    def _get_children(self, ref: ConcurrentHeapRef) -> List[ConcurrentHeapRef]:
        """Get all heap-ref children of an object."""
        children = []
        obj = ref.obj

        if isinstance(obj, dict):
            for v in obj.values():
                child_id = id(v)
                if child_id in self._object_map:
                    children.append(self._object_map[child_id])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                child_id = id(item)
                if child_id in self._object_map:
                    children.append(self._object_map[child_id])
        elif isinstance(obj, ClosureObject):
            for child_obj in [obj.fn, obj.env]:
                cid = id(child_obj)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, GeneratorObject):
            for child_obj in [obj.fn, obj.env]:
                cid = id(child_obj)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for item in obj.stack:
                cid = id(item)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for item in obj.call_stack:
                cid = id(item)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for item in obj.handler_stack:
                cid = id(item)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, AsyncCoroutine):
            for child_obj in [obj.fn, obj.env]:
                cid = id(child_obj)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for item in obj.stack:
                cid = id(item)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            if obj.promise is not None:
                cid = id(obj.promise)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, AsyncGeneratorObject):
            for child_obj in [obj.fn, obj.env]:
                cid = id(child_obj)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for item in obj.stack:
                cid = id(item)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            if obj.next_promise is not None:
                cid = id(obj.next_promise)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, PromiseObject):
            if obj.value is not None:
                cid = id(obj.value)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, ClassObject):
            for attr in [obj.methods, obj.static_methods, obj.getters, obj.setters]:
                cid = id(attr)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            if obj.parent is not None:
                cid = id(obj.parent)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
            for trait in obj.traits:
                cid = id(trait)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, TraitObject):
            cid = id(obj.methods)
            if cid in self._object_map:
                children.append(self._object_map[cid])
            if obj.parent is not None:
                cid = id(obj.parent)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, BoundMethod):
            for child_obj in [obj.instance, obj.method, obj.klass]:
                cid = id(child_obj)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, EnumObject):
            for attr in [obj.variants, obj.methods]:
                cid = id(attr)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, EnumVariant):
            if obj.enum_ref is not None:
                cid = id(obj.enum_ref)
                if cid in self._object_map:
                    children.append(self._object_map[cid])
        elif isinstance(obj, NativeModule):
            cid = id(obj.exports)
            if cid in self._object_map:
                children.append(self._object_map[cid])

        return children

    def start_collection(self, vm: VM):
        """Begin an incremental collection cycle."""
        if self.phase != GCPhase.IDLE:
            return  # Already collecting

        self._cycle += 1
        self._mark_start_time = time.monotonic()
        self._phase_start_time = time.monotonic()

        # Reset all to white
        for ref in self.heap:
            ref.color = Color.WHITE

        # Find roots and color them gray
        roots = self._find_roots(vm)
        self._gray_set = []
        for ref in roots:
            ref.color = Color.GRAY
            self._gray_set.append(ref)

        # Pinned objects are always roots
        for ref in self.heap:
            if ref.pinned and ref.color == Color.WHITE:
                ref.color = Color.GRAY
                self._gray_set.append(ref)

        self.phase = GCPhase.MARK

    def mark_step(self, budget=None) -> bool:
        """Process up to `budget` gray objects. Returns True if marking is done."""
        if self.phase != GCPhase.MARK:
            return True

        if budget is None:
            budget = self.mark_budget

        processed = 0
        while self._gray_set and processed < budget:
            ref = self._gray_set.pop()
            if ref.color != Color.GRAY:
                continue  # Already processed (possible duplicate from barrier)

            # Scan children
            children = self._get_children(ref)
            for child in children:
                if child.color == Color.WHITE:
                    child.color = Color.GRAY
                    self._gray_set.append(child)

            # This object is now fully scanned
            ref.color = Color.BLACK
            processed += 1

        self.stats.mark_steps += 1
        self.stats.total_mark_work += processed

        if not self._gray_set:
            # Mark phase complete -- transition to sweep
            now = time.monotonic()
            self.stats.last_mark_duration = now - self._mark_start_time
            self._begin_sweep()
            return True

        return False

    def _begin_sweep(self):
        """Transition from MARK to SWEEP phase."""
        self._phase_start_time = time.monotonic()
        # Build sweep list (snapshot of heap)
        self._sweep_iter = list(self.heap)
        self._sweep_idx = 0
        self.phase = GCPhase.SWEEP

    def sweep_step(self, budget=None) -> bool:
        """Sweep up to `budget` objects. Returns True if sweeping is done."""
        if self.phase != GCPhase.SWEEP:
            return True

        if budget is None:
            budget = self.sweep_budget

        freed = 0
        processed = 0
        dead = []

        while self._sweep_idx < len(self._sweep_iter) and processed < budget:
            ref = self._sweep_iter[self._sweep_idx]
            self._sweep_idx += 1
            processed += 1

            if ref not in self.heap:
                continue  # Already removed (e.g. by re-tracking)

            if ref.pinned:
                ref.generation += 1
                continue

            if ref.color == Color.WHITE:
                # Unreachable -- collect
                if ref.finalizer is not None:
                    self._finalizer_queue.append(ref)
                for wr in ref.weak_refs:
                    wr._invalidate()
                dead.append(ref)
                freed += 1
            else:
                # Survived -- increment generation
                ref.generation += 1

        # Remove dead objects
        for ref in dead:
            self.heap.discard(ref)
            obj_id = id(ref.obj)
            if obj_id in self._object_map:
                del self._object_map[obj_id]

        self.stats.total_freed += freed
        self.stats.sweep_steps += 1
        self.stats.total_sweep_work += processed

        if self._sweep_idx >= len(self._sweep_iter):
            # Sweep complete
            self._finish_collection()
            return True

        return False

    def _finish_collection(self):
        """Finalize a collection cycle."""
        now = time.monotonic()
        self.stats.last_sweep_duration = now - self._phase_start_time
        self.stats.last_total_duration = now - self._mark_start_time
        self.stats.total_collections += 1
        self.stats.heap_size = len(self.heap)
        self._allocs_since_gc = 0

        # Run finalizers
        self._run_finalizers()

        # Adaptive budget adjustment
        if self.adaptive:
            self._adjust_budgets()

        self.phase = GCPhase.IDLE
        self._sweep_iter = None
        self._gray_set = []

    def _run_finalizers(self):
        """Run queued finalizers."""
        while self._finalizer_queue:
            ref = self._finalizer_queue.pop(0)
            try:
                if ref.finalizer is not None:
                    ref.finalizer(ref.obj)
            except Exception:
                pass

    def _adjust_budgets(self):
        """Adaptive budget: increase if heap is large, decrease if small."""
        heap_sz = len(self.heap)
        if heap_sz > 1000:
            self.mark_budget = min(128, self.mark_budget + 8)
            self.sweep_budget = min(256, self.sweep_budget + 16)
        elif heap_sz < 100:
            self.mark_budget = max(8, self.mark_budget - 4)
            self.sweep_budget = max(16, self.sweep_budget - 8)

    # ---------- Convenience ----------

    def step(self, vm: VM) -> bool:
        """Run one incremental GC step. Returns True if a cycle completed."""
        if self.phase == GCPhase.IDLE:
            if not self.enabled:
                return False
            if self._allocs_since_gc >= self.threshold:
                self.start_collection(vm)
                return False
            return False
        elif self.phase == GCPhase.MARK:
            self.mark_step()
            return False
        elif self.phase == GCPhase.SWEEP:
            done = self.sweep_step()
            return done

    def collect_full(self, vm: VM) -> int:
        """Run a complete collection synchronously (stop-the-world fallback)."""
        before = len(self.heap)
        self.start_collection(vm)
        # Run mark to completion
        while self.phase == GCPhase.MARK:
            self.mark_step(budget=len(self.heap) + 1)
        # Run sweep to completion
        while self.phase == GCPhase.SWEEP:
            self.sweep_step(budget=len(self._sweep_iter) + 1)
        return before - len(self.heap)

    def get_stats(self) -> dict:
        """Return statistics as a dict."""
        return {
            'total_collections': self.stats.total_collections,
            'total_freed': self.stats.total_freed,
            'total_allocations': self.stats.total_allocations,
            'heap_size': self.stats.heap_size,
            'peak_heap_size': self.stats.peak_heap_size,
            'mark_steps': self.stats.mark_steps,
            'sweep_steps': self.stats.sweep_steps,
            'total_mark_work': self.stats.total_mark_work,
            'total_sweep_work': self.stats.total_sweep_work,
            'barrier_hits': self.stats.barrier_hits,
            'barrier_checks': self.stats.barrier_checks,
            'last_mark_duration': self.stats.last_mark_duration,
            'last_sweep_duration': self.stats.last_sweep_duration,
            'last_total_duration': self.stats.last_total_duration,
            'phase': self.phase.value,
            'cycle': self._cycle,
            'gray_set_size': len(self._gray_set),
        }


# ---------- Concurrent GC VM ----------

class ConcurrentGCVM(VM):
    """VM with incremental tri-color garbage collection.

    Hooks into the execution loop to:
    1. Track heap allocations
    2. Fire write barriers on mutations
    3. Run incremental GC steps between instructions
    """

    def __init__(self, chunk: Chunk, trace=False, gc_threshold=256,
                 mark_budget=32, sweep_budget=64, adaptive=True):
        self.cgc = ConcurrentGC(
            threshold=gc_threshold,
            mark_budget=mark_budget,
            sweep_budget=sweep_budget,
            adaptive=adaptive,
        )
        super().__init__(chunk, trace=trace)
        self._gc_track_env()

    def _gc_track_env(self):
        """Track all current env values on the GC heap."""
        for key, val in self.env.items():
            if isinstance(val, HEAP_TYPES):
                self.cgc.track(val)

    def _gc_track(self, val):
        """Track a value if it's a heap type."""
        if isinstance(val, HEAP_TYPES):
            self.cgc.track(val)
        return val

    def _gc_scan_stack_top(self):
        """Track the top of stack if it's a heap object."""
        if self.stack:
            self._gc_track(self.stack[-1])

    def _execute_op(self, op):
        """Override to add write barriers and incremental GC."""
        result = super()._execute_op(op)

        # Track new allocations after creation ops
        if op in (Op.MAKE_CLOSURE, Op.MAKE_ARRAY, Op.MAKE_HASH,
                  Op.MAKE_CLASS, Op.MAKE_TRAIT, Op.CALL, Op.CALL_SPREAD,
                  Op.ARRAY_SPREAD, Op.HASH_SPREAD, Op.STORE):
            self._gc_scan_stack_top()

        # Write barrier: after STORE or INDEX_SET, the env/container was mutated.
        # We fire a general barrier check on any newly stored heap value.
        if op == Op.STORE and self.cgc.phase == GCPhase.MARK:
            # The stored value is now in env -- scan env for new references
            self._gc_barrier_scan_env()
        elif op == Op.INDEX_SET and self.cgc.phase == GCPhase.MARK:
            # Container was mutated -- scan stack for new references
            self._gc_barrier_scan_stack()

        # Run incremental GC step
        self.cgc.step(self)

        return result

    def _gc_barrier_scan_env(self):
        """After a STORE, scan env for any white objects that need graying."""
        for val in self.env.values():
            if isinstance(val, HEAP_TYPES):
                obj_id = id(val)
                ref = self.cgc._object_map.get(obj_id)
                if ref and ref.color == Color.WHITE:
                    ref.color = Color.GRAY
                    self.cgc._gray_set.append(ref)
                    self.cgc.stats.barrier_hits += 1

    def _gc_barrier_scan_stack(self):
        """After an INDEX_SET, scan top of stack for white objects."""
        for val in self.stack[-3:] if len(self.stack) >= 3 else self.stack:
            if isinstance(val, HEAP_TYPES):
                obj_id = id(val)
                ref = self.cgc._object_map.get(obj_id)
                if ref and ref.color == Color.WHITE:
                    ref.color = Color.GRAY
                    self.cgc._gray_set.append(ref)
                    self.cgc.stats.barrier_hits += 1

    def gc_collect(self) -> int:
        """Manually trigger a full synchronous collection."""
        return self.cgc.collect_full(self)

    def gc_start(self):
        """Start an incremental collection cycle."""
        self.cgc.start_collection(self)

    def gc_step(self) -> bool:
        """Run one incremental GC step. Returns True if cycle completed."""
        return self.cgc.step(self)

    def gc_phase(self) -> str:
        """Get current GC phase."""
        return self.cgc.phase.value

    def gc_stats(self) -> dict:
        """Get GC statistics."""
        return self.cgc.get_stats()


# ---------- Public API ----------

def compile_concurrent(source: str) -> Chunk:
    """Compile source code to a chunk."""
    return compile_source(source)


def run_concurrent_gc(source: str, gc_threshold=256, mark_budget=32,
                      sweep_budget=64) -> tuple:
    """Run source code with concurrent GC VM.
    Returns (result, output, gc_stats).
    """
    chunk = compile_concurrent(source)
    vm = ConcurrentGCVM(chunk, gc_threshold=gc_threshold,
                        mark_budget=mark_budget, sweep_budget=sweep_budget)
    result = vm.run()
    # Finalize any in-progress GC
    if vm.cgc.phase != GCPhase.IDLE:
        vm.gc_collect()
    return result, vm.output, vm.gc_stats()


def execute_concurrent_gc(source: str, gc_threshold=256) -> dict:
    """Execute source with concurrent GC. Returns dict with result, output, gc_stats."""
    result, output, stats = run_concurrent_gc(source, gc_threshold=gc_threshold)
    return {
        'result': result,
        'output': output,
        'gc_stats': stats,
    }
