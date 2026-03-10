"""
C071: Garbage Collector for the Stack VM

A mark-sweep garbage collector that tracks heap-allocated objects in the VM,
identifies reachable objects via root scanning and graph traversal, and
reclaims unreachable objects.

Key features:
- Heap tracking: all VM-allocated objects registered in a heap set
- Root scanning: stack, env, call_stack, handler_stack, async queue
- Mark phase: recursive graph traversal from roots
- Sweep phase: remove unmarked objects
- Auto-collection: triggers after N allocations (configurable threshold)
- GC statistics: allocation count, collection count, objects freed
- Generational hints: objects surviving collections get promoted (longer-lived)
- Weak references: references that don't prevent collection
- Finalizers: callbacks invoked before collection

Composes: C070 (trait_decorators) -- full VM with all language features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C070_trait_decorators'))

from trait_decorators import (
    # Core
    lex, parse, Parser, Compiler, VM, Chunk, Op,
    # Object types
    FnObject, ClosureObject, GeneratorObject, PromiseObject,
    AsyncCoroutine, AsyncGeneratorObject, TraitObject, ClassObject,
    BoundMethod, EnumObject, EnumVariant, NativeFunction, NativeModule,
    # Errors
    VMError, CompileError, ParseError,
    # Helpers
    run, execute,
    compile_source as _compile_source,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, Dict, List


# ---------- Heap object wrapper ----------

class HeapRef:
    """Wraps a VM value to track it on the managed heap."""
    __slots__ = ('obj', 'marked', 'generation', 'weak_refs', 'finalizer', '_id')
    _next_id = 0

    def __init__(self, obj, finalizer=None):
        self.obj = obj
        self.marked = False
        self.generation = 0  # incremented on survival
        self.weak_refs = []  # list of WeakRef pointing at this
        self.finalizer = finalizer  # callable or None
        HeapRef._next_id += 1
        self._id = HeapRef._next_id

    def __repr__(self):
        return f"HeapRef({self._id}, gen={self.generation}, {type(self.obj).__name__})"

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, HeapRef) and self._id == other._id


class WeakRef:
    """A weak reference that doesn't prevent GC collection."""
    __slots__ = ('_heap_ref', '_alive')

    def __init__(self, heap_ref: HeapRef):
        self._heap_ref = heap_ref
        self._alive = True
        heap_ref.weak_refs.append(self)

    @property
    def alive(self):
        return self._alive

    def get(self):
        """Returns the referenced object, or None if collected."""
        if self._alive:
            return self._heap_ref.obj
        return None

    def _invalidate(self):
        self._alive = False
        self._heap_ref = None


# ---------- GC Statistics ----------

@dataclass
class GCStats:
    total_allocations: int = 0
    total_collections: int = 0
    total_freed: int = 0
    heap_size: int = 0
    peak_heap_size: int = 0
    gen0_collections: int = 0
    gen1_collections: int = 0
    full_collections: int = 0


# ---------- GC-aware VM ----------

# Types that are heap-allocated (contain references to other objects)
HEAP_TYPES = (
    FnObject, ClosureObject, GeneratorObject, PromiseObject,
    AsyncCoroutine, AsyncGeneratorObject, TraitObject, ClassObject,
    BoundMethod, EnumObject, EnumVariant, NativeFunction, NativeModule,
    list, dict,
)


class GarbageCollector:
    """Mark-sweep garbage collector for VM objects."""

    def __init__(self, threshold=256, gen1_threshold=5):
        self.heap: Set[HeapRef] = set()
        self.stats = GCStats()
        self.threshold = threshold  # allocations before auto-collect
        self._allocs_since_gc = 0
        self.gen1_threshold = gen1_threshold  # collections before gen1 promotion
        self.enabled = True
        self._finalizer_queue: List[HeapRef] = []
        self._object_map: Dict[int, HeapRef] = {}  # id(obj) -> HeapRef

    def track(self, obj, finalizer=None) -> HeapRef:
        """Register an object on the managed heap."""
        obj_id = id(obj)
        if obj_id in self._object_map:
            return self._object_map[obj_id]

        ref = HeapRef(obj, finalizer=finalizer)
        self.heap.add(ref)
        self._object_map[obj_id] = ref
        self.stats.total_allocations += 1
        self.stats.heap_size += 1
        if self.stats.heap_size > self.stats.peak_heap_size:
            self.stats.peak_heap_size = self.stats.heap_size
        self._allocs_since_gc += 1
        return ref

    def create_weak_ref(self, obj) -> Optional[WeakRef]:
        """Create a weak reference to a tracked object."""
        obj_id = id(obj)
        if obj_id not in self._object_map:
            return None
        return WeakRef(self._object_map[obj_id])

    def _find_roots(self, vm: VM) -> Set[int]:
        """Find all root object ids from VM state."""
        roots = set()
        self._scan_value(vm.env, roots)
        for val in vm.stack:
            self._scan_value(val, roots)
        for frame in vm.call_stack:
            if isinstance(frame, (list, tuple)):
                for item in frame:
                    self._scan_value(item, roots)
            else:
                self._scan_value(frame, roots)
        for handler in vm.handler_stack:
            if isinstance(handler, (list, tuple)):
                for item in handler:
                    self._scan_value(item, roots)
            else:
                self._scan_value(handler, roots)
        # Async queue
        for item in vm._async_queue:
            if isinstance(item, (list, tuple)):
                for v in item:
                    self._scan_value(v, roots)
            else:
                self._scan_value(item, roots)
        if vm._current_async is not None:
            self._scan_value(vm._current_async, roots)
        return roots

    def _scan_value(self, val, roots: set):
        """Recursively scan a value for heap object references."""
        val_id = id(val)
        if val_id in roots:
            return  # already visited
        if val_id in self._object_map:
            roots.add(val_id)
        if isinstance(val, dict):
            roots.add(val_id)
            for k, v in val.items():
                self._scan_value(v, roots)
        elif isinstance(val, (list, tuple)):
            roots.add(val_id)
            for item in val:
                self._scan_value(item, roots)
        elif isinstance(val, ClosureObject):
            roots.add(val_id)
            self._scan_value(val.fn, roots)
            self._scan_value(val.env, roots)
        elif isinstance(val, GeneratorObject):
            roots.add(val_id)
            self._scan_value(val.fn, roots)
            self._scan_value(val.env, roots)
            for item in val.stack:
                self._scan_value(item, roots)
            for item in val.call_stack:
                self._scan_value(item, roots)
            for item in val.handler_stack:
                self._scan_value(item, roots)
        elif isinstance(val, AsyncCoroutine):
            roots.add(val_id)
            self._scan_value(val.fn, roots)
            self._scan_value(val.env, roots)
            for item in val.stack:
                self._scan_value(item, roots)
            if val.promise is not None:
                self._scan_value(val.promise, roots)
        elif isinstance(val, AsyncGeneratorObject):
            roots.add(val_id)
            self._scan_value(val.fn, roots)
            self._scan_value(val.env, roots)
            for item in val.stack:
                self._scan_value(item, roots)
            if val.next_promise is not None:
                self._scan_value(val.next_promise, roots)
        elif isinstance(val, PromiseObject):
            roots.add(val_id)
            if val.value is not None:
                self._scan_value(val.value, roots)
        elif isinstance(val, ClassObject):
            roots.add(val_id)
            self._scan_value(val.methods, roots)
            self._scan_value(val.static_methods, roots)
            self._scan_value(val.getters, roots)
            self._scan_value(val.setters, roots)
            if val.parent is not None:
                self._scan_value(val.parent, roots)
            for trait in val.traits:
                self._scan_value(trait, roots)
        elif isinstance(val, TraitObject):
            roots.add(val_id)
            self._scan_value(val.methods, roots)
            if val.parent is not None:
                self._scan_value(val.parent, roots)
        elif isinstance(val, BoundMethod):
            roots.add(val_id)
            self._scan_value(val.instance, roots)
            self._scan_value(val.method, roots)
            self._scan_value(val.klass, roots)
        elif isinstance(val, EnumObject):
            roots.add(val_id)
            self._scan_value(val.variants, roots)
            self._scan_value(val.methods, roots)
        elif isinstance(val, EnumVariant):
            roots.add(val_id)
            if val.enum_ref is not None:
                self._scan_value(val.enum_ref, roots)
        elif isinstance(val, FnObject):
            roots.add(val_id)
        elif isinstance(val, NativeFunction):
            roots.add(val_id)
        elif isinstance(val, NativeModule):
            roots.add(val_id)
            self._scan_value(val.exports, roots)

    def mark(self, vm: VM):
        """Mark phase: find all reachable objects from roots."""
        # Clear all marks
        for ref in self.heap:
            ref.marked = False

        # Find roots and mark
        root_ids = self._find_roots(vm)
        for ref in self.heap:
            if id(ref.obj) in root_ids:
                ref.marked = True

    def sweep(self) -> int:
        """Sweep phase: remove unmarked objects. Returns count freed."""
        freed = 0
        dead = set()
        for ref in self.heap:
            if not ref.marked:
                # Queue finalizer
                if ref.finalizer is not None:
                    self._finalizer_queue.append(ref)
                # Invalidate weak refs
                for wr in ref.weak_refs:
                    wr._invalidate()
                dead.add(ref)
                if id(ref.obj) in self._object_map:
                    del self._object_map[id(ref.obj)]
                freed += 1
            else:
                # Survived -- increment generation
                ref.generation += 1

        self.heap -= dead
        self.stats.total_freed += freed
        self.stats.heap_size = len(self.heap)
        return freed

    def _run_finalizers(self):
        """Run queued finalizers."""
        while self._finalizer_queue:
            ref = self._finalizer_queue.pop(0)
            try:
                if ref.finalizer is not None:
                    ref.finalizer(ref.obj)
            except Exception:
                pass  # finalizers must not propagate exceptions

    def collect(self, vm: VM) -> int:
        """Run a full mark-sweep collection. Returns objects freed."""
        self.mark(vm)
        freed = self.sweep()
        self._run_finalizers()
        self.stats.total_collections += 1
        self.stats.full_collections += 1
        self._allocs_since_gc = 0
        return freed

    def collect_gen0(self, vm: VM) -> int:
        """Collect only generation-0 objects (young generation)."""
        # Mark all
        for ref in self.heap:
            ref.marked = False

        root_ids = self._find_roots(vm)
        for ref in self.heap:
            if id(ref.obj) in root_ids:
                ref.marked = True
            elif ref.generation >= self.gen1_threshold:
                # Gen1+ objects are not collected in gen0 sweep
                ref.marked = True

        freed = self.sweep()
        self._run_finalizers()
        self.stats.total_collections += 1
        self.stats.gen0_collections += 1
        self._allocs_since_gc = 0
        return freed

    def maybe_collect(self, vm: VM) -> int:
        """Auto-collect if allocation threshold is reached."""
        if not self.enabled:
            return 0
        if self._allocs_since_gc >= self.threshold:
            return self.collect_gen0(vm)
        return 0

    def get_stats(self) -> dict:
        """Return GC statistics as a dict."""
        return {
            'total_allocations': self.stats.total_allocations,
            'total_collections': self.stats.total_collections,
            'total_freed': self.stats.total_freed,
            'heap_size': self.stats.heap_size,
            'peak_heap_size': self.stats.peak_heap_size,
            'gen0_collections': self.stats.gen0_collections,
            'gen1_collections': self.stats.gen1_collections,
            'full_collections': self.stats.full_collections,
        }


# ---------- GC-instrumented VM ----------

class GCVM(VM):
    """VM with integrated garbage collection.

    Hooks into the execution loop to track heap-allocated objects
    and periodically run collection between instructions.
    """

    def __init__(self, chunk: Chunk, trace=False, gc_threshold=256):
        self.gc = GarbageCollector(threshold=gc_threshold)
        super().__init__(chunk, trace=trace)
        self._gc_track_env()

    def _gc_track_env(self):
        """Track all current env values on the GC heap."""
        for key, val in self.env.items():
            if isinstance(val, HEAP_TYPES):
                self.gc.track(val)

    def _gc_track(self, val):
        """Track a value if it's a heap type."""
        if isinstance(val, HEAP_TYPES):
            self.gc.track(val)
        return val

    def _gc_scan_stack_top(self):
        """Track the top of stack if it's a heap object."""
        if self.stack:
            self._gc_track(self.stack[-1])

    def _execute_op(self, op):
        """Override to track allocations after each instruction."""
        result = super()._execute_op(op)
        # After creation ops, track the result on the heap
        if op in (Op.MAKE_CLOSURE, Op.MAKE_ARRAY, Op.MAKE_HASH,
                  Op.MAKE_CLASS, Op.MAKE_TRAIT, Op.CALL, Op.CALL_SPREAD,
                  Op.ARRAY_SPREAD, Op.HASH_SPREAD, Op.STORE):
            self._gc_scan_stack_top()
            self.gc.maybe_collect(self)
        return result

    def gc_collect(self) -> int:
        """Manually trigger garbage collection."""
        return self.gc.collect(self)

    def gc_stats(self) -> dict:
        """Get GC statistics."""
        return self.gc.get_stats()


# ---------- Public API ----------

def compile_source(source: str) -> Chunk:
    """Compile source code to a chunk."""
    chunk, compiler = _compile_source(source)
    return chunk


def run_with_gc(source: str, gc_threshold=256) -> tuple:
    """Run source code with GC-instrumented VM.
    Returns (result, output, gc_stats).
    """
    chunk = compile_source(source)
    vm = GCVM(chunk, gc_threshold=gc_threshold)
    result = vm.run()
    return result, vm.output, vm.gc.get_stats()


def execute_with_gc(source: str, gc_threshold=256) -> dict:
    """Execute source code with GC. Returns dict with result, output, gc_stats."""
    result, output, stats = run_with_gc(source, gc_threshold=gc_threshold)
    return {
        'result': result,
        'output': output,
        'gc_stats': stats,
    }
