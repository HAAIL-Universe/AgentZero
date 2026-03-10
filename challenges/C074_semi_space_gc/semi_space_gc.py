"""
C074: Semi-Space Garbage Collector (Cheney's Algorithm)
=======================================================
Composing: C073 (Memory Pools / Arena Allocator)

A copying/moving garbage collector using two semi-spaces:
- From-space: where new objects are allocated (bump pointer)
- To-space: where live objects are copied during collection
- Forwarding pointers: update references after copying
- Cheney's algorithm: BFS traversal using to-space as implicit queue
- Large object space: objects too big for semi-spaces
- Pinned objects: immovable objects (FFI, roots)
- Generational mode: nursery semi-space + tenured region
- Statistics: allocation rate, survival rate, copy overhead
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict, List, Set, Tuple
from enum import Enum, auto
import sys


# ---------------------------------------------------------------------------
# Managed Object -- wrapper for GC-tracked objects
# ---------------------------------------------------------------------------

class ManagedObject:
    """A GC-managed object with forwarding pointer support."""

    __slots__ = ('value', 'forwarded', 'forward_to', 'pinned',
                 'finalizer', 'obj_id', 'size', 'age', '_alive')

    _next_id = 0

    def __init__(self, value: Any, finalizer: Optional[Callable] = None,
                 pinned: bool = False, size: int = 0):
        self.value = value
        self.forwarded = False
        self.forward_to: Optional['ManagedObject'] = None
        self.pinned = pinned
        self.finalizer = finalizer
        self.obj_id = ManagedObject._next_id
        ManagedObject._next_id += 1
        self.size = size if size > 0 else self._estimate_size(value)
        self.age = 0
        self._alive = True

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Rough size estimate for an object."""
        if value is None:
            return 8
        if isinstance(value, (int, float, bool)):
            return 16
        if isinstance(value, str):
            return 40 + len(value)
        if isinstance(value, (list, tuple)):
            return 56 + 8 * len(value)
        if isinstance(value, dict):
            return 64 + 16 * len(value)
        return 64

    def is_alive(self):
        return self._alive and not self.forwarded

    def resolve(self) -> 'ManagedObject':
        """Follow forwarding chain to find the current copy."""
        obj = self
        while obj.forwarded and obj.forward_to is not None:
            obj = obj.forward_to
        return obj


# ---------------------------------------------------------------------------
# Semi-Space -- one half of the copying collector's memory
# ---------------------------------------------------------------------------

class SemiSpace:
    """A contiguous region for bump-pointer allocation."""

    def __init__(self, capacity: int, name: str = ""):
        self.capacity = capacity
        self.name = name
        self.objects: List[ManagedObject] = []
        self.bump: int = 0  # Next allocation offset (in size units)
        self.used_bytes: int = 0

    def can_fit(self, size: int) -> bool:
        return self.used_bytes + size <= self.capacity

    def allocate(self, managed_obj: ManagedObject) -> bool:
        """Bump-pointer allocation. Returns True if successful."""
        if not self.can_fit(managed_obj.size):
            return False
        self.objects.append(managed_obj)
        self.bump += 1
        self.used_bytes += managed_obj.size
        return True

    def reset(self):
        """Clear this semi-space for reuse."""
        self.objects.clear()
        self.bump = 0
        self.used_bytes = 0

    @property
    def live_count(self) -> int:
        return len([o for o in self.objects if o._alive and not o.forwarded])

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.used_bytes / self.capacity

    def __repr__(self):
        return f"SemiSpace({self.name}, {self.used_bytes}/{self.capacity}, {len(self.objects)} objs)"


# ---------------------------------------------------------------------------
# Large Object Space -- for objects too big for semi-spaces
# ---------------------------------------------------------------------------

class LargeObjectSpace:
    """Separate space for large objects (not copied, just mark-swept)."""

    def __init__(self, threshold: int = 1024):
        self.threshold = threshold
        self.objects: List[ManagedObject] = []
        self._marked: Set[int] = set()

    def should_use_los(self, size: int) -> bool:
        return size >= self.threshold

    def allocate(self, managed_obj: ManagedObject):
        self.objects.append(managed_obj)

    def mark(self, obj_id: int):
        self._marked.add(obj_id)

    def sweep(self) -> int:
        """Remove unmarked large objects. Returns count freed."""
        freed = 0
        surviving = []
        for obj in self.objects:
            if obj.obj_id in self._marked:
                surviving.append(obj)
            else:
                if obj.finalizer:
                    try:
                        obj.finalizer(obj.value)
                    except Exception:
                        pass
                obj._alive = False
                obj.value = None
                freed += 1
        self.objects = surviving
        self._marked.clear()
        return freed

    def reset_marks(self):
        self._marked.clear()

    @property
    def live_count(self) -> int:
        return len([o for o in self.objects if o._alive])

    @property
    def total_size(self) -> int:
        return sum(o.size for o in self.objects if o._alive)


# ---------------------------------------------------------------------------
# GC Statistics
# ---------------------------------------------------------------------------

@dataclass
class GCStats:
    total_allocations: int = 0
    total_bytes_allocated: int = 0
    total_collections: int = 0
    total_bytes_copied: int = 0
    total_objects_copied: int = 0
    total_objects_freed: int = 0
    total_bytes_freed: int = 0
    total_finalizers_run: int = 0
    los_allocations: int = 0
    los_freed: int = 0
    peak_live_bytes: int = 0
    peak_live_count: int = 0
    # Per-collection stats (last collection)
    last_live_before: int = 0
    last_live_after: int = 0
    last_bytes_copied: int = 0
    last_survival_rate: float = 0.0


# ---------------------------------------------------------------------------
# Semi-Space Collector (Cheney's Algorithm)
# ---------------------------------------------------------------------------

class SemiSpaceCollector:
    """Copying garbage collector using Cheney's algorithm.

    Design:
    - Two semi-spaces of equal size: from-space and to-space
    - Objects allocated in from-space via bump pointer
    - On collection: copy live objects from from-space to to-space, then swap
    - Forwarding pointers ensure references are updated
    - Large objects go to a separate LOS (mark-swept, not copied)
    - Pinned objects stay in a separate set (never moved)
    - Optional generational mode: semi-space acts as nursery
    """

    def __init__(self, space_size: int = 4096,
                 los_threshold: int = 1024,
                 auto_collect: bool = True):
        self.space_size = space_size
        self.auto_collect = auto_collect

        # The two semi-spaces
        self.from_space = SemiSpace(space_size, "from")
        self.to_space = SemiSpace(space_size, "to")

        # Large object space
        self.los = LargeObjectSpace(los_threshold)

        # Pinned objects (never moved)
        self.pinned: List[ManagedObject] = []

        # Object lookup by obj_id
        self._obj_map: Dict[int, ManagedObject] = {}

        # Root set (registered by mutator)
        self._roots: List[ManagedObject] = []

        # Reference graph: obj_id -> list of child obj_ids
        self._references: Dict[int, List[int]] = {}

        # Stats
        self.stats = GCStats()

        # Callbacks
        self._on_collect: Optional[Callable] = None

        # Tenured space (for generational mode)
        self._tenured: Optional[SemiSpace] = None
        self._tenure_threshold: int = 3  # Collections survived before tenuring
        self._generational: bool = False

    # -- Configuration --

    def enable_generational(self, tenured_size: int = 8192,
                            tenure_threshold: int = 3):
        """Enable generational mode with a tenured space."""
        self._generational = True
        self._tenured = SemiSpace(tenured_size, "tenured")
        self._tenure_threshold = tenure_threshold

    def set_collection_callback(self, callback: Callable):
        """Register a callback to run after each collection."""
        self._on_collect = callback

    # -- Allocation --

    def allocate(self, value: Any, finalizer: Optional[Callable] = None,
                 pinned: bool = False, size: int = 0) -> ManagedObject:
        """Allocate a new managed object.

        Routes to:
        - Pinned set if pinned=True
        - LOS if object is large
        - From-space otherwise (bump pointer)
        """
        managed = ManagedObject(value, finalizer=finalizer, pinned=pinned, size=size)
        self.stats.total_allocations += 1
        self.stats.total_bytes_allocated += managed.size

        if pinned:
            self.pinned.append(managed)
            self._obj_map[managed.obj_id] = managed
            self._update_peak()
            return managed

        if self.los.should_use_los(managed.size):
            self.los.allocate(managed)
            self._obj_map[managed.obj_id] = managed
            self.stats.los_allocations += 1
            self._update_peak()
            return managed

        # Try from-space
        if not self.from_space.can_fit(managed.size):
            if self.auto_collect:
                self.collect()
                if not self.from_space.can_fit(managed.size):
                    # Still can't fit -- grow or fail
                    raise MemoryError(
                        f"Out of memory: cannot fit {managed.size} bytes "
                        f"(from-space {self.from_space.used_bytes}/{self.from_space.capacity})"
                    )
            else:
                raise MemoryError(
                    f"Out of memory: from-space full "
                    f"({self.from_space.used_bytes}/{self.from_space.capacity})"
                )

        success = self.from_space.allocate(managed)
        assert success, "Allocation should succeed after space check"
        self._obj_map[managed.obj_id] = managed
        self._update_peak()
        return managed

    def _update_peak(self):
        live = self.live_count
        live_bytes = self.live_bytes
        if live > self.stats.peak_live_count:
            self.stats.peak_live_count = live
        if live_bytes > self.stats.peak_live_bytes:
            self.stats.peak_live_bytes = live_bytes

    # -- Reference graph --

    def set_references(self, parent: ManagedObject, children: List[ManagedObject]):
        """Declare that parent references the given children."""
        self._references[parent.obj_id] = [c.obj_id for c in children]

    def add_reference(self, parent: ManagedObject, child: ManagedObject):
        """Add a single reference from parent to child."""
        if parent.obj_id not in self._references:
            self._references[parent.obj_id] = []
        if child.obj_id not in self._references[parent.obj_id]:
            self._references[parent.obj_id].append(child.obj_id)

    def remove_reference(self, parent: ManagedObject, child: ManagedObject):
        """Remove a reference from parent to child."""
        if parent.obj_id in self._references:
            refs = self._references[parent.obj_id]
            if child.obj_id in refs:
                refs.remove(child.obj_id)

    def get_children(self, obj: ManagedObject) -> List[ManagedObject]:
        """Get all live children of an object."""
        if obj.obj_id not in self._references:
            return []
        children = []
        for child_id in self._references[obj.obj_id]:
            if child_id in self._obj_map:
                child = self._obj_map[child_id]
                resolved = child.resolve()
                if resolved._alive:
                    children.append(resolved)
        return children

    # -- Root management --

    def add_root(self, obj: ManagedObject):
        """Register an object as a GC root."""
        if obj not in self._roots:
            self._roots.append(obj)

    def remove_root(self, obj: ManagedObject):
        """Unregister a GC root."""
        if obj in self._roots:
            self._roots.remove(obj)

    def clear_roots(self):
        """Remove all roots."""
        self._roots.clear()

    # -- Collection (Cheney's Algorithm) --

    def collect(self) -> int:
        """Run a collection cycle. Returns number of objects freed."""
        if self._generational:
            return self._collect_generational()
        return self._collect_full()

    def _collect_full(self) -> int:
        """Full semi-space collection using Cheney's algorithm."""
        live_before = self.from_space.live_count + len(self.pinned) + self.los.live_count
        self.stats.last_live_before = live_before

        bytes_copied = 0
        objects_copied = 0

        # Reset to-space
        self.to_space.reset()

        # Phase 1: Copy roots to to-space
        scan_queue_start = 0  # Cheney's scan pointer into to_space.objects

        for root in self._roots:
            root = root.resolve()
            if root._alive and not root.pinned:
                self._copy_to_to_space(root)

        # Also copy objects referenced by pinned objects
        for pin in self.pinned:
            if pin._alive:
                for child in self.get_children(pin):
                    child = child.resolve()
                    if child._alive and not child.pinned and not child.forwarded:
                        self._copy_to_to_space(child)

        # Phase 2: Cheney scan -- BFS using to-space as implicit queue
        while scan_queue_start < len(self.to_space.objects):
            obj = self.to_space.objects[scan_queue_start]
            scan_queue_start += 1

            # Process children
            children = self.get_children(obj)
            for child in children:
                child = child.resolve()
                if child._alive and not child.pinned and not child.forwarded:
                    self._copy_to_to_space(child)

        # Phase 3: Update reference graph with new obj_ids
        self._update_references_after_copy()

        # Phase 4: Run finalizers on dead objects in from-space
        freed = 0
        for obj in self.from_space.objects:
            if obj._alive and not obj.forwarded:
                # This object was not copied -- it's garbage
                if obj.finalizer:
                    try:
                        obj.finalizer(obj.value)
                    except Exception:
                        pass
                    self.stats.total_finalizers_run += 1
                obj._alive = False
                obj.value = None
                freed += 1
                # Remove from obj_map
                if obj.obj_id in self._obj_map:
                    del self._obj_map[obj.obj_id]
                if obj.obj_id in self._references:
                    del self._references[obj.obj_id]

        # Phase 5: LOS mark-sweep
        self.los.reset_marks()
        self._mark_los_from_roots()
        los_freed = self.los.sweep()
        for obj in list(self._obj_map.values()):
            if not obj._alive:
                if obj.obj_id in self._obj_map:
                    del self._obj_map[obj.obj_id]

        # Phase 6: Swap spaces
        self.from_space, self.to_space = self.to_space, self.from_space
        self.to_space.reset()

        # Update roots to resolved versions
        self._roots = [r.resolve() for r in self._roots if r.resolve()._alive]

        # Count copied
        objects_copied = len(self.from_space.objects)
        bytes_copied = self.from_space.used_bytes

        # Stats
        total_freed = freed + los_freed
        self.stats.total_collections += 1
        self.stats.total_objects_copied += objects_copied
        self.stats.total_bytes_copied += bytes_copied
        self.stats.total_objects_freed += total_freed
        self.stats.total_bytes_freed += freed  # approximate
        self.stats.los_freed += los_freed
        self.stats.last_bytes_copied = bytes_copied

        live_after = self.from_space.live_count + len(self.pinned) + self.los.live_count
        self.stats.last_live_after = live_after
        if live_before > 0:
            self.stats.last_survival_rate = live_after / live_before
        else:
            self.stats.last_survival_rate = 0.0

        if self._on_collect:
            self._on_collect(self.stats)

        return total_freed

    def _copy_to_to_space(self, obj: ManagedObject) -> ManagedObject:
        """Copy an object to to-space. Returns the new copy (or existing if already forwarded)."""
        if obj.forwarded:
            return obj.forward_to

        if obj.pinned:
            return obj  # Pinned objects don't move

        # LOS objects are mark-swept, not copied
        if any(lo is obj or lo.obj_id == obj.obj_id for lo in self.los.objects):
            return obj

        # Create copy in to-space
        copy = ManagedObject(
            obj.value,
            finalizer=obj.finalizer,
            pinned=False,
            size=obj.size
        )
        copy.age = obj.age + 1
        copy._alive = True

        success = self.to_space.allocate(copy)
        if not success:
            # To-space full -- this shouldn't happen if spaces are sized correctly
            # but handle gracefully by keeping object in place
            return obj

        # Set forwarding pointer
        obj.forwarded = True
        obj.forward_to = copy

        # Update obj_map
        self._obj_map[obj.obj_id] = copy
        copy.obj_id = obj.obj_id  # Preserve identity

        return copy

    def _update_references_after_copy(self):
        """Update reference graph to point to forwarded copies."""
        new_refs: Dict[int, List[int]] = {}
        for parent_id, child_ids in self._references.items():
            if parent_id in self._obj_map:
                parent = self._obj_map[parent_id]
                if parent._alive:
                    new_children = []
                    for child_id in child_ids:
                        if child_id in self._obj_map:
                            child = self._obj_map[child_id]
                            resolved = child.resolve()
                            if resolved._alive:
                                new_children.append(resolved.obj_id)
                    if new_children:
                        new_refs[parent_id] = new_children
        self._references = new_refs

    def _mark_los_from_roots(self):
        """Mark large objects reachable from roots."""
        visited: Set[int] = set()
        worklist: List[ManagedObject] = []

        # Start from roots
        for root in self._roots:
            root = root.resolve()
            if root._alive:
                worklist.append(root)

        # Include pinned
        for pin in self.pinned:
            if pin._alive:
                worklist.append(pin)

        # BFS
        while worklist:
            obj = worklist.pop()
            if obj.obj_id in visited:
                continue
            visited.add(obj.obj_id)

            # If it's in LOS, mark it
            if any(lo is obj or lo.obj_id == obj.obj_id for lo in self.los.objects):
                self.los.mark(obj.obj_id)

            # Traverse children
            for child in self.get_children(obj):
                child = child.resolve()
                if child._alive and child.obj_id not in visited:
                    worklist.append(child)

    # -- Generational collection --

    def _collect_generational(self) -> int:
        """Generational collection: semi-space is nursery, promote old objects."""
        assert self._tenured is not None

        # Minor collection: copy live nursery objects
        live_before = self.from_space.live_count
        self.stats.last_live_before = live_before

        self.to_space.reset()
        scan_queue_start = 0

        # Copy roots
        for root in self._roots:
            root = root.resolve()
            if root._alive and not root.pinned:
                # Check if in from-space (nursery)
                if self._in_from_space(root):
                    self._copy_or_promote(root)

        # Copy pinned children
        for pin in self.pinned:
            if pin._alive:
                for child in self.get_children(pin):
                    child = child.resolve()
                    if child._alive and not child.pinned and not child.forwarded:
                        if self._in_from_space(child):
                            self._copy_or_promote(child)

        # Also trace from tenured -> nursery references
        if self._tenured:
            for obj in self._tenured.objects:
                if obj._alive:
                    for child in self.get_children(obj):
                        child = child.resolve()
                        if child._alive and not child.pinned and not child.forwarded:
                            if self._in_from_space(child):
                                self._copy_or_promote(child)

        # Cheney scan (to-space as nursery survivors)
        while scan_queue_start < len(self.to_space.objects):
            obj = self.to_space.objects[scan_queue_start]
            scan_queue_start += 1
            for child in self.get_children(obj):
                child = child.resolve()
                if child._alive and not child.pinned and not child.forwarded:
                    if self._in_from_space(child):
                        self._copy_or_promote(child)

        # Free dead nursery objects
        freed = 0
        for obj in self.from_space.objects:
            if obj._alive and not obj.forwarded:
                if obj.finalizer:
                    try:
                        obj.finalizer(obj.value)
                    except Exception:
                        pass
                    self.stats.total_finalizers_run += 1
                obj._alive = False
                obj.value = None
                freed += 1
                if obj.obj_id in self._obj_map:
                    del self._obj_map[obj.obj_id]
                if obj.obj_id in self._references:
                    del self._references[obj.obj_id]

        # LOS sweep
        self.los.reset_marks()
        self._mark_los_from_roots()
        los_freed = self.los.sweep()

        # Swap spaces
        self.from_space, self.to_space = self.to_space, self.from_space
        self.to_space.reset()

        # Update roots
        self._roots = [r.resolve() for r in self._roots if r.resolve()._alive]

        # Update references
        self._update_references_after_copy()

        total_freed = freed + los_freed
        self.stats.total_collections += 1
        self.stats.total_objects_freed += total_freed
        self.stats.los_freed += los_freed

        live_after = self.from_space.live_count + len(self.pinned) + self.los.live_count
        if self._tenured:
            live_after += self._tenured.live_count
        self.stats.last_live_after = live_after
        if live_before > 0:
            self.stats.last_survival_rate = live_after / live_before

        if self._on_collect:
            self._on_collect(self.stats)

        return total_freed

    def _in_from_space(self, obj: ManagedObject) -> bool:
        """Check if object is in from-space (not tenured, not pinned, not LOS)."""
        return any(o is obj for o in self.from_space.objects)

    def _copy_or_promote(self, obj: ManagedObject) -> ManagedObject:
        """Copy to to-space, or promote to tenured if old enough."""
        if obj.forwarded:
            return obj.forward_to

        if obj.age >= self._tenure_threshold and self._tenured is not None:
            return self._promote_to_tenured(obj)

        return self._copy_to_to_space(obj)

    def _promote_to_tenured(self, obj: ManagedObject) -> ManagedObject:
        """Promote an object to tenured space."""
        if obj.forwarded:
            return obj.forward_to

        copy = ManagedObject(
            obj.value,
            finalizer=obj.finalizer,
            pinned=False,
            size=obj.size
        )
        copy.age = obj.age + 1
        copy._alive = True
        copy.obj_id = obj.obj_id

        success = self._tenured.allocate(copy)
        if not success:
            # Tenured full -- fall back to to-space
            return self._copy_to_to_space(obj)

        obj.forwarded = True
        obj.forward_to = copy
        self._obj_map[obj.obj_id] = copy
        return copy

    # -- Full collection (including tenured) --

    def collect_full(self) -> int:
        """Major collection: collect everything including tenured space."""
        if not self._generational or self._tenured is None:
            return self.collect()

        # Collect nursery first
        nursery_freed = self._collect_generational()

        # Now collect tenured: mark-sweep (don't copy tenured objects)
        tenured_freed = 0
        visited: Set[int] = set()
        worklist: List[ManagedObject] = []

        # Mark from roots
        for root in self._roots:
            root = root.resolve()
            if root._alive:
                worklist.append(root)
        for pin in self.pinned:
            if pin._alive:
                worklist.append(pin)
        # Mark from from-space (nursery survivors)
        for obj in self.from_space.objects:
            if obj._alive:
                worklist.append(obj)

        while worklist:
            obj = worklist.pop()
            if obj.obj_id in visited:
                continue
            visited.add(obj.obj_id)
            for child in self.get_children(obj):
                child = child.resolve()
                if child._alive and child.obj_id not in visited:
                    worklist.append(child)

        # Sweep tenured
        surviving = []
        for obj in self._tenured.objects:
            if obj.obj_id in visited:
                surviving.append(obj)
            else:
                if obj.finalizer:
                    try:
                        obj.finalizer(obj.value)
                    except Exception:
                        pass
                    self.stats.total_finalizers_run += 1
                obj._alive = False
                obj.value = None
                tenured_freed += 1
                if obj.obj_id in self._obj_map:
                    del self._obj_map[obj.obj_id]
                if obj.obj_id in self._references:
                    del self._references[obj.obj_id]

        # Rebuild tenured
        old_tenured = self._tenured
        self._tenured = SemiSpace(old_tenured.capacity, "tenured")
        for obj in surviving:
            self._tenured.allocate(obj)

        self.stats.total_objects_freed += tenured_freed
        return nursery_freed + tenured_freed

    # -- Queries --

    @property
    def live_count(self) -> int:
        count = self.from_space.live_count
        count += len([p for p in self.pinned if p._alive])
        count += self.los.live_count
        if self._tenured:
            count += self._tenured.live_count
        return count

    @property
    def live_bytes(self) -> int:
        total = 0
        for obj in self.from_space.objects:
            if obj._alive and not obj.forwarded:
                total += obj.size
        for obj in self.pinned:
            if obj._alive:
                total += obj.size
        total += self.los.total_size
        if self._tenured:
            for obj in self._tenured.objects:
                if obj._alive:
                    total += obj.size
        return total

    @property
    def total_capacity(self) -> int:
        cap = self.from_space.capacity + self.to_space.capacity
        if self._tenured:
            cap += self._tenured.capacity
        return cap

    def is_alive(self, obj: ManagedObject) -> bool:
        """Check if a managed object is still alive (following forwards)."""
        resolved = obj.resolve()
        return resolved._alive

    def get_value(self, obj: ManagedObject) -> Any:
        """Get the current value of a managed object (following forwards)."""
        resolved = obj.resolve()
        if resolved._alive:
            return resolved.value
        return None

    def set_value(self, obj: ManagedObject, value: Any):
        """Update the value of a managed object."""
        resolved = obj.resolve()
        if resolved._alive:
            resolved.value = value

    def get_age(self, obj: ManagedObject) -> int:
        """Get the age (collections survived) of an object."""
        resolved = obj.resolve()
        return resolved.age

    def lookup(self, obj_id: int) -> Optional[ManagedObject]:
        """Look up an object by its ID."""
        if obj_id in self._obj_map:
            obj = self._obj_map[obj_id]
            resolved = obj.resolve()
            if resolved._alive:
                return resolved
        return None

    def get_stats(self) -> dict:
        """Return comprehensive GC statistics."""
        return {
            'total_allocations': self.stats.total_allocations,
            'total_bytes_allocated': self.stats.total_bytes_allocated,
            'total_collections': self.stats.total_collections,
            'total_objects_copied': self.stats.total_objects_copied,
            'total_bytes_copied': self.stats.total_bytes_copied,
            'total_objects_freed': self.stats.total_objects_freed,
            'total_finalizers_run': self.stats.total_finalizers_run,
            'los_allocations': self.stats.los_allocations,
            'los_freed': self.stats.los_freed,
            'peak_live_count': self.stats.peak_live_count,
            'peak_live_bytes': self.stats.peak_live_bytes,
            'live_count': self.live_count,
            'live_bytes': self.live_bytes,
            'from_space_used': self.from_space.used_bytes,
            'from_space_capacity': self.from_space.capacity,
            'to_space_used': self.to_space.used_bytes,
            'to_space_capacity': self.to_space.capacity,
            'last_survival_rate': self.stats.last_survival_rate,
            'generational': self._generational,
        }

    def dump_heap(self) -> dict:
        """Dump heap state for debugging."""
        return {
            'from_space': [
                {'id': o.obj_id, 'value': repr(o.value), 'age': o.age,
                 'alive': o._alive, 'forwarded': o.forwarded, 'size': o.size}
                for o in self.from_space.objects
            ],
            'to_space': [
                {'id': o.obj_id, 'value': repr(o.value), 'age': o.age,
                 'alive': o._alive, 'forwarded': o.forwarded, 'size': o.size}
                for o in self.to_space.objects
            ],
            'pinned': [
                {'id': o.obj_id, 'value': repr(o.value), 'alive': o._alive}
                for o in self.pinned
            ],
            'los': [
                {'id': o.obj_id, 'value': repr(o.value), 'alive': o._alive, 'size': o.size}
                for o in self.los.objects
            ],
            'tenured': [
                {'id': o.obj_id, 'value': repr(o.value), 'age': o.age, 'alive': o._alive}
                for o in (self._tenured.objects if self._tenured else [])
            ],
            'roots': [r.resolve().obj_id for r in self._roots if r.resolve()._alive],
            'references': dict(self._references),
        }
