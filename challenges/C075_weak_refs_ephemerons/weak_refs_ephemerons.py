"""
C075: Weak References + Ephemerons
==================================
Standalone challenge -- new domain: GC-integrated reference types

Weak references allow observing objects without preventing their collection.
Ephemerons are the hard problem: a (key, value) pair where the value is only
considered reachable if the key is reachable through paths NOT going through
the ephemeron itself. This requires iterative fixpoint computation during GC.

Key components:
- GCObject: base managed object with mark/sweep support
- WeakRef: reference that doesn't prevent collection, with callbacks
- WeakValueDict: dictionary with weak values (auto-cleanup on collection)
- WeakKeyDict: dictionary with weak keys (auto-cleanup on collection)
- Ephemeron: (key, value) pair with conditional reachability
- EphemeronTable: collection of ephemerons with GC fixpoint
- MarkSweepGC: garbage collector with ephemeron-aware mark phase
- Finalizer ordering: finalizers run in dependency order
- Resurrection: objects revived by finalizers get re-tracked

The ephemeron fixpoint algorithm:
1. Mark all strongly reachable objects from roots
2. Scan ephemerons: if key is marked, mark value and trace it
3. Repeat step 2 until no new marks (fixpoint)
4. Remaining ephemerons with unmarked keys are dead
"""

from typing import Any, Optional, Callable, Dict, List, Set, Tuple
from collections import defaultdict
from enum import Enum, auto


# ---------------------------------------------------------------------------
# GC Object -- base for all managed objects
# ---------------------------------------------------------------------------

class GCObject:
    """A garbage-collected object with explicit reference tracking."""

    __slots__ = ('value', 'obj_id', 'marked', 'generation', 'finalizer',
                 'refs', '_weak_refs', '_alive', '_attached_ephemerons')

    _next_id = 0

    def __init__(self, value: Any = None, finalizer: Optional[Callable] = None):
        self.value = value
        self.obj_id = GCObject._next_id
        GCObject._next_id += 1
        self.marked = False
        self.generation = 0
        self.finalizer = finalizer
        self.refs: List['GCObject'] = []         # strong references to other objects
        self._weak_refs: List['WeakRef'] = []    # weak refs pointing at this
        self._alive = True
        self._attached_ephemerons: List['Ephemeron'] = []  # ephemerons with this as key

    def set_references(self, *refs: 'GCObject'):
        """Set this object's outgoing strong references."""
        self.refs = list(refs)

    def add_reference(self, ref: 'GCObject'):
        """Add a strong reference."""
        self.refs.append(ref)

    def remove_reference(self, ref: 'GCObject'):
        """Remove a strong reference."""
        self.refs = [r for r in self.refs if r is not ref]

    def is_alive(self) -> bool:
        return self._alive

    def __repr__(self):
        status = "alive" if self._alive else "dead"
        return f"GCObject({self.obj_id}, {self.value!r}, {status})"

    def __hash__(self):
        return self.obj_id

    def __eq__(self, other):
        return isinstance(other, GCObject) and self.obj_id == other.obj_id


# ---------------------------------------------------------------------------
# Weak Reference -- doesn't prevent collection
# ---------------------------------------------------------------------------

class WeakRef:
    """A weak reference that doesn't prevent collection.

    Supports optional callback invoked when the referent is collected.
    """

    __slots__ = ('_referent', '_alive', '_callback', '_ref_id')
    _next_id = 0

    def __init__(self, referent: GCObject, callback: Optional[Callable] = None):
        if not isinstance(referent, GCObject):
            raise TypeError("WeakRef target must be a GCObject")
        if not referent.is_alive():
            raise ValueError("Cannot create weak reference to dead object")
        self._referent = referent
        self._alive = True
        self._callback = callback
        self._ref_id = WeakRef._next_id
        WeakRef._next_id += 1
        referent._weak_refs.append(self)

    @property
    def alive(self) -> bool:
        return self._alive

    def get(self) -> Optional[GCObject]:
        """Return the referent if alive, else None."""
        if self._alive:
            return self._referent
        return None

    def _invalidate(self):
        """Called by GC when referent is collected."""
        if not self._alive:
            return
        self._alive = False
        cb = self._callback
        referent = self._referent
        self._referent = None
        if cb is not None:
            try:
                cb(referent)
            except Exception:
                pass  # swallow callback exceptions

    def __repr__(self):
        if self._alive:
            return f"WeakRef({self._referent})"
        return "WeakRef(<dead>)"

    def __hash__(self):
        return self._ref_id

    def __eq__(self, other):
        return isinstance(other, WeakRef) and self._ref_id == other._ref_id


# ---------------------------------------------------------------------------
# Weak Value Dictionary -- values are weak references
# ---------------------------------------------------------------------------

class WeakValueDict:
    """Dictionary where values are weakly referenced.

    When a value object is collected, its entry is automatically removed.
    """

    def __init__(self):
        self._data: Dict[Any, WeakRef] = {}

    def __setitem__(self, key: Any, value: GCObject):
        if not isinstance(value, GCObject):
            raise TypeError("WeakValueDict values must be GCObjects")
        # Create weak ref with cleanup callback
        def _remove(ref, k=key):
            self._data.pop(k, None)
        self._data[key] = WeakRef(value, callback=_remove)

    def __getitem__(self, key: Any) -> Optional[GCObject]:
        wr = self._data.get(key)
        if wr is None:
            raise KeyError(key)
        obj = wr.get()
        if obj is None:
            del self._data[key]
            raise KeyError(key)
        return obj

    def get(self, key: Any, default=None) -> Optional[GCObject]:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        # Prune dead entries
        self._prune()
        return len(self._data)

    def __delitem__(self, key: Any):
        del self._data[key]

    def _prune(self):
        """Remove entries whose referents have been collected."""
        dead_keys = [k for k, wr in self._data.items() if not wr.alive]
        for k in dead_keys:
            del self._data[k]

    def keys(self):
        self._prune()
        return self._data.keys()

    def values(self):
        self._prune()
        return [wr.get() for wr in self._data.values()]

    def items(self):
        self._prune()
        return [(k, wr.get()) for k, wr in self._data.items()]

    def __repr__(self):
        self._prune()
        items = [(k, wr.get()) for k, wr in self._data.items()]
        return f"WeakValueDict({dict(items)})"


# ---------------------------------------------------------------------------
# Weak Key Dictionary -- keys are weakly referenced
# ---------------------------------------------------------------------------

class WeakKeyDict:
    """Dictionary where keys are weakly referenced.

    When a key object is collected, its entry is automatically removed.
    """

    def __init__(self):
        self._data: Dict[int, Tuple[WeakRef, Any]] = {}  # obj_id -> (weakref, value)

    def __setitem__(self, key: GCObject, value: Any):
        if not isinstance(key, GCObject):
            raise TypeError("WeakKeyDict keys must be GCObjects")
        oid = key.obj_id
        def _remove(ref, k=oid):
            self._data.pop(k, None)
        self._data[oid] = (WeakRef(key, callback=_remove), value)

    def __getitem__(self, key: GCObject) -> Any:
        oid = key.obj_id
        entry = self._data.get(oid)
        if entry is None:
            raise KeyError(key)
        wr, value = entry
        if not wr.alive:
            del self._data[oid]
            raise KeyError(key)
        return value

    def get(self, key: GCObject, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: GCObject) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        self._prune()
        return len(self._data)

    def __delitem__(self, key: GCObject):
        del self._data[key.obj_id]

    def _prune(self):
        dead = [k for k, (wr, _) in self._data.items() if not wr.alive]
        for k in dead:
            del self._data[k]

    def keys(self):
        self._prune()
        return [wr.get() for wr, _ in self._data.values()]

    def values(self):
        self._prune()
        return [v for _, v in self._data.values()]

    def items(self):
        self._prune()
        return [(wr.get(), v) for wr, v in self._data.values()]


# ---------------------------------------------------------------------------
# Ephemeron -- (key, value) with conditional reachability
# ---------------------------------------------------------------------------

class Ephemeron:
    """An ephemeron holds a (key, value) pair where:
    - The key is weakly held
    - The value is only considered reachable if the key is reachable
      through paths that do NOT go through this ephemeron

    This is the fundamental building block for ephemeron tables,
    weak-keyed caches, and observer patterns.
    """

    __slots__ = ('_key', '_value', '_alive', '_eph_id', '_callback')
    _next_id = 0

    def __init__(self, key: GCObject, value: Any,
                 callback: Optional[Callable] = None):
        if not isinstance(key, GCObject):
            raise TypeError("Ephemeron key must be a GCObject")
        self._key = key
        self._value = value
        self._alive = True
        self._callback = callback
        self._eph_id = Ephemeron._next_id
        Ephemeron._next_id += 1
        key._attached_ephemerons.append(self)

    @property
    def key(self) -> Optional[GCObject]:
        return self._key if self._alive else None

    @property
    def value(self) -> Any:
        return self._value if self._alive else None

    @property
    def alive(self) -> bool:
        return self._alive

    def _invalidate(self):
        """Called by GC when key becomes unreachable."""
        if not self._alive:
            return
        self._alive = False
        cb = self._callback
        key = self._key
        value = self._value
        self._key = None
        self._value = None
        if cb is not None:
            try:
                cb(key, value)
            except Exception:
                pass

    def __repr__(self):
        if self._alive:
            return f"Ephemeron({self._key}, {self._value!r})"
        return "Ephemeron(<dead>)"

    def __hash__(self):
        return self._eph_id

    def __eq__(self, other):
        return isinstance(other, Ephemeron) and self._eph_id == other._eph_id


# ---------------------------------------------------------------------------
# Ephemeron Table -- collection of ephemerons with GC-aware iteration
# ---------------------------------------------------------------------------

class EphemeronTable:
    """A table of ephemerons supporting lookup, iteration, and GC integration.

    This is essentially a weak-keyed map with ephemeron semantics:
    values are only retained while keys are independently reachable.
    """

    def __init__(self, callback: Optional[Callable] = None):
        self._entries: Dict[int, Ephemeron] = {}  # obj_id -> Ephemeron
        self._default_callback = callback

    def set(self, key: GCObject, value: Any):
        """Add or update an ephemeron entry."""
        oid = key.obj_id
        # Remove old entry if exists
        old = self._entries.get(oid)
        if old is not None and old._alive:
            # Detach old ephemeron from key
            key._attached_ephemerons = [
                e for e in key._attached_ephemerons if e is not old
            ]
        eph = Ephemeron(key, value, callback=self._default_callback)
        self._entries[oid] = eph

    def get(self, key: GCObject, default=None) -> Any:
        """Get value for key, or default if not found or dead."""
        oid = key.obj_id
        eph = self._entries.get(oid)
        if eph is None or not eph.alive:
            if eph is not None and not eph.alive:
                del self._entries[oid]
            return default
        return eph.value

    def __contains__(self, key: GCObject) -> bool:
        oid = key.obj_id
        eph = self._entries.get(oid)
        if eph is None:
            return False
        if not eph.alive:
            del self._entries[oid]
            return False
        return True

    def remove(self, key: GCObject):
        """Remove an entry."""
        oid = key.obj_id
        eph = self._entries.get(oid)
        if eph is not None:
            # Detach from key
            if eph._key is not None:
                eph._key._attached_ephemerons = [
                    e for e in eph._key._attached_ephemerons if e is not eph
                ]
            del self._entries[oid]

    def __len__(self) -> int:
        self._prune()
        return len(self._entries)

    def _prune(self):
        dead = [k for k, e in self._entries.items() if not e.alive]
        for k in dead:
            del self._entries[k]

    def all_ephemerons(self) -> List[Ephemeron]:
        """Return all live ephemerons."""
        self._prune()
        return list(self._entries.values())

    def keys(self) -> List[GCObject]:
        self._prune()
        return [e.key for e in self._entries.values()]

    def values(self) -> List[Any]:
        self._prune()
        return [e.value for e in self._entries.values()]

    def items(self) -> List[Tuple[GCObject, Any]]:
        self._prune()
        return [(e.key, e.value) for e in self._entries.values()]

    def __repr__(self):
        self._prune()
        items = [(e.key, e.value) for e in self._entries.values()]
        return f"EphemeronTable({items})"


# ---------------------------------------------------------------------------
# Finalizer Queue -- ordered finalization
# ---------------------------------------------------------------------------

class FinalizerState(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()


class FinalizerEntry:
    """Tracks a finalizer and its execution state."""

    __slots__ = ('obj', 'finalizer', 'state', 'dependencies')

    def __init__(self, obj: GCObject, finalizer: Callable):
        self.obj = obj
        self.finalizer = finalizer
        self.state = FinalizerState.PENDING
        self.dependencies: List['FinalizerEntry'] = []  # must run after these


# ---------------------------------------------------------------------------
# Mark-Sweep GC with Ephemeron Support
# ---------------------------------------------------------------------------

class MarkSweepGC:
    """Garbage collector with full ephemeron support.

    The mark phase uses iterative fixpoint for ephemerons:
    1. Mark all objects strongly reachable from roots
    2. Scan all ephemerons: if key is marked, mark value (trace transitively)
    3. Repeat until no new objects are marked (fixpoint)
    4. Unmarked ephemeron keys -> invalidate ephemeron
    """

    def __init__(self, threshold: int = 100):
        self.heap: Set[GCObject] = set()
        self.roots: Set[GCObject] = set()
        self.ephemerons: Set[Ephemeron] = set()
        self.ephemeron_tables: List[EphemeronTable] = []
        self.threshold = threshold
        self._allocs_since_gc = 0
        self.enabled = True

        # Stats
        self.total_allocations = 0
        self.total_collections = 0
        self.total_freed = 0
        self.total_ephemerons_cleared = 0
        self.total_weak_refs_cleared = 0
        self.total_finalizers_run = 0
        self.peak_heap_size = 0

        # Resurrection tracking
        self._resurrected: Set[GCObject] = set()
        self._in_finalization = False

        # Finalizer ordering
        self._finalizer_queue: List[FinalizerEntry] = []

    def alloc(self, value: Any = None, finalizer: Optional[Callable] = None) -> GCObject:
        """Allocate a new GC-managed object."""
        obj = GCObject(value, finalizer=finalizer)
        self.heap.add(obj)
        self.total_allocations += 1
        self._allocs_since_gc += 1
        if len(self.heap) > self.peak_heap_size:
            self.peak_heap_size = len(self.heap)
        return obj

    def add_root(self, obj: GCObject):
        """Add a root reference."""
        self.roots.add(obj)

    def remove_root(self, obj: GCObject):
        """Remove a root reference."""
        self.roots.discard(obj)

    def track_ephemeron(self, eph: Ephemeron):
        """Register an ephemeron with the GC."""
        self.ephemerons.add(eph)

    def track_ephemeron_table(self, table: EphemeronTable):
        """Register an ephemeron table with the GC."""
        if table not in self.ephemeron_tables:
            self.ephemeron_tables.append(table)

    def create_ephemeron(self, key: GCObject, value: Any,
                         callback: Optional[Callable] = None) -> Ephemeron:
        """Create and track an ephemeron."""
        eph = Ephemeron(key, value, callback=callback)
        self.ephemerons.add(eph)
        return eph

    def create_weak_ref(self, obj: GCObject,
                        callback: Optional[Callable] = None) -> WeakRef:
        """Create a weak reference to a managed object."""
        return WeakRef(obj, callback=callback)

    def should_collect(self) -> bool:
        """Check if auto-collection should trigger."""
        return self.enabled and self._allocs_since_gc >= self.threshold

    # ----- Mark Phase -----

    def _mark_object(self, obj: GCObject, worklist: List[GCObject]) -> bool:
        """Mark an object and add its references to worklist.
        Returns True if the object was newly marked."""
        if obj.marked:
            return False
        obj.marked = True
        worklist.append(obj)
        return True

    def _trace_worklist(self, worklist: List[GCObject]):
        """Trace all objects in worklist, marking their references."""
        while worklist:
            obj = worklist.pop()
            for ref in obj.refs:
                if ref._alive and not ref.marked:
                    ref.marked = True
                    worklist.append(ref)

    def _mark_roots(self) -> List[GCObject]:
        """Mark all root objects and return worklist."""
        worklist: List[GCObject] = []
        for root in self.roots:
            if root._alive:
                self._mark_object(root, worklist)
        return worklist

    def _process_ephemerons(self) -> bool:
        """Process ephemerons: if key is marked, mark value.
        Returns True if any new marks were made."""
        new_marks = False
        worklist: List[GCObject] = []

        # Collect all ephemerons (standalone + from tables)
        all_ephs = set(self.ephemerons)
        for table in self.ephemeron_tables:
            for eph in table._entries.values():
                if eph.alive:
                    all_ephs.add(eph)

        for eph in all_ephs:
            if not eph.alive:
                continue
            key = eph._key
            if key is None or not key._alive:
                continue
            # If key is marked (reachable), then value should be traced
            if key.marked:
                val = eph._value
                if isinstance(val, GCObject) and val._alive and not val.marked:
                    self._mark_object(val, worklist)
                    new_marks = True

        # Trace any newly marked objects transitively
        if worklist:
            self._trace_worklist(worklist)
            new_marks = True

        return new_marks

    def mark(self):
        """Full mark phase with ephemeron fixpoint."""
        # Clear all marks
        for obj in self.heap:
            obj.marked = False

        # Phase 1: mark from roots
        worklist = self._mark_roots()
        self._trace_worklist(worklist)

        # Phase 2: ephemeron fixpoint
        # Repeat until no new marks are produced
        iterations = 0
        while self._process_ephemerons():
            iterations += 1
            if iterations > len(self.heap) + len(self.ephemerons) + 1:
                break  # safety: avoid infinite loop

    # ----- Sweep Phase -----

    def _build_finalizer_queue(self, dead: Set[GCObject]) -> List[FinalizerEntry]:
        """Build ordered finalizer queue from dead objects.
        Objects referenced by other dead objects finalize first (bottom-up)."""
        entries: Dict[int, FinalizerEntry] = {}
        for obj in dead:
            if obj.finalizer is not None:
                entries[obj.obj_id] = FinalizerEntry(obj, obj.finalizer)

        # Build dependency graph: if A references B and both are dead+finalizable,
        # B's finalizer should run before A's
        for oid, entry in entries.items():
            for ref in entry.obj.refs:
                if ref.obj_id in entries:
                    entry.dependencies.append(entries[ref.obj_id])

        # Topological sort (Kahn's algorithm)
        in_degree: Dict[int, int] = {oid: 0 for oid in entries}
        for oid, entry in entries.items():
            for dep in entry.dependencies:
                in_degree[dep.obj.obj_id] = in_degree.get(dep.obj.obj_id, 0)
                # dep must run before entry -> entry depends on dep
                # but in_degree counts how many depend on this node
                pass

        # Simpler: just sort by reference depth (objects with no refs to other
        # dead objects go first, then objects that reference them, etc.)
        ordered: List[FinalizerEntry] = []
        visited: Set[int] = set()

        def visit(entry: FinalizerEntry):
            if entry.obj.obj_id in visited:
                return
            visited.add(entry.obj.obj_id)
            for dep in entry.dependencies:
                visit(dep)
            ordered.append(entry)

        for entry in entries.values():
            visit(entry)

        return ordered

    def _run_finalizers(self, dead: Set[GCObject]):
        """Run finalizers for dead objects in dependency order."""
        queue = self._build_finalizer_queue(dead)
        self._in_finalization = True

        for entry in queue:
            entry.state = FinalizerState.RUNNING
            try:
                entry.finalizer(entry.obj)
            except Exception:
                pass  # swallow finalizer exceptions
            entry.state = FinalizerState.DONE
            self.total_finalizers_run += 1

        self._in_finalization = False

    def _invalidate_weak_refs(self, dead: Set[GCObject]):
        """Invalidate all weak references to dead objects."""
        for obj in dead:
            for wr in obj._weak_refs:
                wr._invalidate()
                self.total_weak_refs_cleared += 1
            obj._weak_refs.clear()

    def _invalidate_ephemerons(self, dead: Set[GCObject]):
        """Invalidate ephemerons whose keys are dead or not in heap."""
        dead_ids = {obj.obj_id for obj in dead}
        heap_ids = {obj.obj_id for obj in self.heap}

        # Standalone ephemerons
        dead_ephs = set()
        for eph in self.ephemerons:
            if not eph.alive:
                dead_ephs.add(eph)
                continue
            # Key is dead, or key is not alive, or key is not in heap
            key = eph._key
            if key is not None and (key.obj_id in dead_ids or not key._alive
                                    or key.obj_id not in heap_ids):
                eph._invalidate()
                dead_ephs.add(eph)
                self.total_ephemerons_cleared += 1
        self.ephemerons -= dead_ephs

        # Ephemerons in tables
        for table in self.ephemeron_tables:
            dead_table_keys = []
            for oid, eph in table._entries.items():
                key = eph._key
                if key is not None and (key.obj_id in dead_ids or not key._alive
                                        or key.obj_id not in heap_ids):
                    eph._invalidate()
                    dead_table_keys.append(oid)
                    self.total_ephemerons_cleared += 1
            for k in dead_table_keys:
                del table._entries[k]

    def _check_resurrection(self, dead: Set[GCObject]) -> Set[GCObject]:
        """Check if any dead objects were resurrected by finalizers.
        An object is resurrected if a finalizer stored a reference to it
        in a live object."""
        resurrected = set()

        # Re-mark from roots to find newly reachable objects
        for obj in self.heap:
            obj.marked = False

        worklist = self._mark_roots()
        self._trace_worklist(worklist)

        # Ephemeron fixpoint again
        while self._process_ephemerons():
            pass

        for obj in dead:
            if obj.marked:
                resurrected.add(obj)
                obj.generation += 1  # promoted for surviving

        return resurrected

    def sweep(self) -> int:
        """Sweep phase: collect unmarked objects. Returns count freed."""
        dead = {obj for obj in self.heap if not obj.marked and obj._alive}

        if not dead:
            return 0

        # Run finalizers (may resurrect objects)
        has_finalizers = any(obj.finalizer is not None for obj in dead)
        if has_finalizers:
            self._run_finalizers(dead)
            resurrected = self._check_resurrection(dead)
            dead -= resurrected
            self._resurrected |= resurrected

        # Invalidate weak refs and ephemerons for truly dead objects
        self._invalidate_weak_refs(dead)
        self._invalidate_ephemerons(dead)

        # Remove from heap
        freed = len(dead)
        for obj in dead:
            obj._alive = False
        self.heap -= dead

        self.total_freed += freed
        return freed

    def collect(self) -> int:
        """Full collection cycle: mark + sweep."""
        self.mark()
        freed = self.sweep()
        self.total_collections += 1
        self._allocs_since_gc = 0

        # Increment generation for survivors
        for obj in self.heap:
            if obj._alive:
                obj.generation += 1

        return freed

    def stats(self) -> dict:
        """Return GC statistics."""
        return {
            'heap_size': len(self.heap),
            'total_allocations': self.total_allocations,
            'total_collections': self.total_collections,
            'total_freed': self.total_freed,
            'peak_heap_size': self.peak_heap_size,
            'ephemerons_cleared': self.total_ephemerons_cleared,
            'weak_refs_cleared': self.total_weak_refs_cleared,
            'finalizers_run': self.total_finalizers_run,
            'active_ephemerons': len(self.ephemerons),
        }

    # ----- Generational support -----

    def minor_collect(self, gen_threshold: int = 3) -> int:
        """Collect only young objects (generation < threshold)."""
        # Clear marks
        for obj in self.heap:
            obj.marked = False

        # Mark from roots (all roots are considered live)
        worklist = self._mark_roots()
        self._trace_worklist(worklist)

        # Ephemeron fixpoint
        while self._process_ephemerons():
            pass

        # Only sweep young objects
        dead = {obj for obj in self.heap
                if not obj.marked and obj._alive and obj.generation < gen_threshold}

        if not dead:
            self.total_collections += 1
            self._allocs_since_gc = 0
            for obj in self.heap:
                if obj._alive:
                    obj.generation += 1
            return 0

        # Finalizers, weak refs, ephemerons
        has_finalizers = any(obj.finalizer is not None for obj in dead)
        if has_finalizers:
            self._run_finalizers(dead)
            resurrected = self._check_resurrection(dead)
            dead -= resurrected

        self._invalidate_weak_refs(dead)
        self._invalidate_ephemerons(dead)

        freed = len(dead)
        for obj in dead:
            obj._alive = False
        self.heap -= dead

        self.total_freed += freed
        self.total_collections += 1
        self._allocs_since_gc = 0

        for obj in self.heap:
            if obj._alive:
                obj.generation += 1

        return freed


# ---------------------------------------------------------------------------
# Convenience: WeakSet -- a set with weak membership
# ---------------------------------------------------------------------------

class WeakSet:
    """A set where membership is weak -- collected objects are auto-removed."""

    def __init__(self):
        self._refs: Dict[int, WeakRef] = {}

    def add(self, obj: GCObject):
        oid = obj.obj_id
        if oid in self._refs and self._refs[oid].alive:
            return
        def _remove(ref, k=oid):
            self._refs.pop(k, None)
        self._refs[oid] = WeakRef(obj, callback=_remove)

    def discard(self, obj: GCObject):
        self._refs.pop(obj.obj_id, None)

    def __contains__(self, obj: GCObject) -> bool:
        wr = self._refs.get(obj.obj_id)
        if wr is None:
            return False
        if not wr.alive:
            del self._refs[obj.obj_id]
            return False
        return True

    def __len__(self) -> int:
        self._prune()
        return len(self._refs)

    def _prune(self):
        dead = [k for k, wr in self._refs.items() if not wr.alive]
        for k in dead:
            del self._refs[k]

    def __iter__(self):
        self._prune()
        for wr in list(self._refs.values()):
            obj = wr.get()
            if obj is not None:
                yield obj
