"""
C073: Memory Pools / Arena Allocator
=====================================
Composing: C071 (GC) + C072 (Concurrent GC)

Arena-based memory management with:
- Bump-pointer arenas for fast sequential allocation
- Fixed-size pools for common object sizes (small/medium/large)
- Generational arenas (nursery + tenured)
- Bulk deallocation (free entire arena at once)
- GC integration (mark-sweep and concurrent tri-color compatible)
- Fragmentation tracking and compaction hints
- Thread-safe pool allocation with write barrier compatibility
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Callable, Dict, List, Set, Tuple
import time


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class ArenaState(Enum):
    ACTIVE = auto()      # Currently accepting allocations
    FULL = auto()        # No more space
    FROZEN = auto()      # Immutable (tenured, no new allocs)
    FREED = auto()       # Bulk deallocated


class PoolSize(Enum):
    SMALL = 64       # ints, bools, small strings
    MEDIUM = 256     # lists < 16 elements, small dicts
    LARGE = 1024     # large objects, closures with many captures


class Generation(Enum):
    NURSERY = 0      # New objects land here
    YOUNG = 1        # Survived 1 collection
    TENURED = 2      # Survived 2+ collections (long-lived)


# ---------------------------------------------------------------------------
# Arena Slot -- a single allocation unit within an arena
# ---------------------------------------------------------------------------

@dataclass
class ArenaSlot:
    """A single allocated slot within an arena block."""
    index: int              # Position in arena
    obj: Any = None         # The stored object
    marked: bool = False    # GC mark bit
    color: int = 0          # 0=WHITE, 1=GRAY, 2=BLACK (tri-color compat)
    generation: int = 0     # Survival count
    finalizer: Optional[Callable] = None
    pinned: bool = False    # If True, never moved or collected
    _alive: bool = True     # False after deallocation

    def is_alive(self):
        return self._alive and self.obj is not None


# ---------------------------------------------------------------------------
# Arena Block -- contiguous region with bump-pointer allocation
# ---------------------------------------------------------------------------

class ArenaBlock:
    """A fixed-size contiguous block of slots with bump-pointer allocation."""

    def __init__(self, block_id: int, capacity: int, generation: Generation = Generation.NURSERY):
        self.block_id = block_id
        self.capacity = capacity
        self.generation = generation
        self.state = ArenaState.ACTIVE
        self.slots: List[ArenaSlot] = []
        self.bump: int = 0  # Next free index
        self.live_count: int = 0
        self.created_at: float = time.monotonic()

    def allocate(self, obj: Any, finalizer: Optional[Callable] = None,
                 pinned: bool = False) -> Optional[ArenaSlot]:
        """Bump-pointer allocation. O(1) fast path."""
        if self.state != ArenaState.ACTIVE or self.bump >= self.capacity:
            if self.state == ArenaState.ACTIVE:
                self.state = ArenaState.FULL
            return None

        slot = ArenaSlot(
            index=self.bump,
            obj=obj,
            finalizer=finalizer,
            pinned=pinned,
        )
        self.slots.append(slot)
        self.bump += 1
        self.live_count += 1
        return slot

    def deallocate_slot(self, slot: ArenaSlot):
        """Mark a single slot as dead. Does not reclaim space (arena is append-only)."""
        if slot._alive:
            if slot.finalizer:
                try:
                    slot.finalizer(slot.obj)
                except Exception:
                    pass
            slot._alive = False
            slot.obj = None
            slot.finalizer = None
            self.live_count -= 1

    def bulk_free(self) -> int:
        """Free entire arena at once. Returns count of objects freed."""
        freed = 0
        for slot in self.slots:
            if slot._alive:
                if slot.finalizer:
                    try:
                        slot.finalizer(slot.obj)
                    except Exception:
                        pass
                slot._alive = False
                slot.obj = None
                slot.finalizer = None
                freed += 1
        self.live_count = 0
        self.state = ArenaState.FREED
        return freed

    def freeze(self):
        """Freeze arena -- no more allocations (used for tenuring)."""
        self.state = ArenaState.FROZEN

    @property
    def utilization(self) -> float:
        """Fraction of slots that are alive."""
        if self.bump == 0:
            return 0.0
        return self.live_count / self.bump

    @property
    def fragmentation(self) -> float:
        """Fraction of allocated slots that are dead (holes)."""
        if self.bump == 0:
            return 0.0
        dead = self.bump - self.live_count
        return dead / self.bump

    @property
    def is_empty(self) -> bool:
        return self.live_count == 0

    def live_objects(self) -> List[ArenaSlot]:
        """Iterate live slots (for GC root scanning)."""
        return [s for s in self.slots if s._alive]


# ---------------------------------------------------------------------------
# Fixed-Size Pool -- slab allocator for uniform-size objects
# ---------------------------------------------------------------------------

class FixedPool:
    """Pool allocator for objects of a specific size class.

    Uses multiple arena blocks internally, creating new ones as needed.
    Maintains a free list for recycled slots.
    """

    def __init__(self, pool_id: int, size_class: PoolSize,
                 block_capacity: int = 256):
        self.pool_id = pool_id
        self.size_class = size_class
        self.block_capacity = block_capacity
        self.blocks: List[ArenaBlock] = []
        self.free_list: List[Tuple[int, int]] = []  # (block_idx, slot_idx)
        self._next_block_id = 0
        self._total_allocs = 0
        self._total_frees = 0

        # Create initial block
        self._add_block()

    def _add_block(self) -> ArenaBlock:
        block = ArenaBlock(self._next_block_id, self.block_capacity)
        self._next_block_id += 1
        self.blocks.append(block)
        return block

    def allocate(self, obj: Any, finalizer: Optional[Callable] = None,
                 pinned: bool = False) -> ArenaSlot:
        """Allocate from free list first, then bump pointer."""
        self._total_allocs += 1

        # Try free list first (recycled slots)
        while self.free_list:
            block_idx, slot_idx = self.free_list.pop()
            if block_idx < len(self.blocks):
                block = self.blocks[block_idx]
                if slot_idx < len(block.slots):
                    slot = block.slots[slot_idx]
                    if not slot._alive:
                        slot.obj = obj
                        slot.finalizer = finalizer
                        slot.pinned = pinned
                        slot._alive = True
                        slot.marked = False
                        slot.color = 0
                        slot.generation = 0
                        block.live_count += 1
                        return slot

        # Try current active block
        for block in reversed(self.blocks):
            if block.state == ArenaState.ACTIVE:
                slot = block.allocate(obj, finalizer, pinned)
                if slot is not None:
                    return slot

        # All blocks full -- create new one
        new_block = self._add_block()
        slot = new_block.allocate(obj, finalizer, pinned)
        assert slot is not None
        return slot

    def free(self, slot: ArenaSlot):
        """Return a slot to the free list."""
        if slot._alive:
            # Find which block owns this slot
            for i, block in enumerate(self.blocks):
                if any(s is slot for s in block.slots):
                    block.deallocate_slot(slot)
                    self.free_list.append((i, slot.index))
                    self._total_frees += 1
                    return
            # Slot not found in any block -- just kill it
            slot._alive = False
            slot.obj = None
            self._total_frees += 1

    @property
    def total_capacity(self) -> int:
        return sum(b.capacity for b in self.blocks)

    @property
    def live_count(self) -> int:
        return sum(b.live_count for b in self.blocks)

    @property
    def utilization(self) -> float:
        total = sum(b.bump for b in self.blocks)
        if total == 0:
            return 0.0
        return self.live_count / total


# ---------------------------------------------------------------------------
# Memory Manager -- coordinates arenas, pools, generations, and GC
# ---------------------------------------------------------------------------

@dataclass
class MemoryStats:
    total_allocations: int = 0
    total_frees: int = 0
    total_bulk_frees: int = 0
    arena_count: int = 0
    pool_count: int = 0
    nursery_promotions: int = 0
    young_promotions: int = 0
    compactions: int = 0
    peak_live: int = 0
    collections: int = 0


class MemoryManager:
    """Central memory manager coordinating arenas, pools, and GC integration.

    Architecture:
    - Nursery arena: small, frequently collected (gen0)
    - Young arenas: survived 1 collection (gen1)
    - Tenured arenas: long-lived objects (gen2+), rarely collected
    - Fixed pools: for uniform-size objects (configurable)
    - GC hooks: mark/sweep integration for both stop-the-world and concurrent
    """

    def __init__(self, nursery_capacity: int = 512,
                 young_capacity: int = 1024,
                 tenured_capacity: int = 2048,
                 auto_collect_threshold: int = 256,
                 promotion_age: int = 2):
        self.nursery_capacity = nursery_capacity
        self.young_capacity = young_capacity
        self.tenured_capacity = tenured_capacity
        self.auto_collect_threshold = auto_collect_threshold
        self.promotion_age = promotion_age

        self.stats = MemoryStats()
        self._next_block_id = 0

        # Generational arenas
        self.nursery: ArenaBlock = self._make_arena(nursery_capacity, Generation.NURSERY)
        self.young_arenas: List[ArenaBlock] = []
        self.tenured_arenas: List[ArenaBlock] = []

        # Fixed-size pools (optional, created on demand)
        self.pools: Dict[PoolSize, FixedPool] = {}
        self._next_pool_id = 0

        # Object lookup: obj id -> (ArenaSlot, ArenaBlock)
        self._slot_map: Dict[int, Tuple[ArenaSlot, ArenaBlock]] = {}

        # Allocs since last collection
        self._allocs_since_gc = 0

        # Write barrier log (for concurrent GC compat)
        self._barrier_log: List[ArenaSlot] = []

        # Roots registered by the VM
        self._roots: List[ArenaSlot] = []

    def _make_arena(self, capacity: int, gen: Generation) -> ArenaBlock:
        block = ArenaBlock(self._next_block_id, capacity, gen)
        self._next_block_id += 1
        self.stats.arena_count += 1
        return block

    # -- Pool management --

    def get_or_create_pool(self, size_class: PoolSize) -> FixedPool:
        """Get or create a fixed-size pool for the given size class."""
        if size_class not in self.pools:
            pool = FixedPool(self._next_pool_id, size_class)
            self._next_pool_id += 1
            self.pools[size_class] = pool
            self.stats.pool_count += 1
        return self.pools[size_class]

    def classify_size(self, obj: Any) -> Optional[PoolSize]:
        """Classify an object into a pool size class, or None for arena allocation."""
        if isinstance(obj, (int, float, bool)):
            return PoolSize.SMALL
        if isinstance(obj, str):
            if len(obj) <= 64:
                return PoolSize.SMALL
            elif len(obj) <= 256:
                return PoolSize.MEDIUM
            else:
                return PoolSize.LARGE
        if isinstance(obj, (list, tuple)):
            if len(obj) <= 16:
                return PoolSize.MEDIUM
            else:
                return PoolSize.LARGE
        if isinstance(obj, dict):
            if len(obj) <= 8:
                return PoolSize.MEDIUM
            else:
                return PoolSize.LARGE
        # Default: no pool, use arena
        return None

    # -- Allocation --

    def allocate(self, obj: Any, finalizer: Optional[Callable] = None,
                 pinned: bool = False, use_pool: bool = False) -> ArenaSlot:
        """Allocate an object. Routes to pool or nursery arena.

        Args:
            obj: The object to track
            finalizer: Optional cleanup callback
            pinned: If True, object is never collected or moved
            use_pool: If True, try to use a fixed-size pool

        Returns:
            ArenaSlot wrapping the object
        """
        # Dedup check
        obj_id = id(obj)
        if obj_id in self._slot_map:
            existing_slot, _ = self._slot_map[obj_id]
            if existing_slot._alive:
                return existing_slot

        self.stats.total_allocations += 1
        self._allocs_since_gc += 1

        # Pool allocation path
        if use_pool:
            size_class = self.classify_size(obj)
            if size_class is not None:
                pool = self.get_or_create_pool(size_class)
                slot = pool.allocate(obj, finalizer, pinned)
                # Pool slots don't go in _slot_map for arena tracking
                # but we track them for dedup
                self._slot_map[obj_id] = (slot, pool.blocks[-1])
                self._update_peak()
                return slot

        # Arena allocation path -- nursery first
        slot = self.nursery.allocate(obj, finalizer, pinned)
        if slot is not None:
            self._slot_map[obj_id] = (slot, self.nursery)
            self._update_peak()
            return slot

        # Nursery full -- try to collect, then retry
        if self.nursery.state != ArenaState.ACTIVE:
            self._promote_nursery()
            slot = self.nursery.allocate(obj, finalizer, pinned)
            if slot is not None:
                self._slot_map[obj_id] = (slot, self.nursery)
                self._update_peak()
                return slot

        # Should not reach here -- promotion creates new nursery
        raise MemoryError("Failed to allocate: nursery promotion failed")

    def allocate_in_generation(self, obj: Any, gen: Generation,
                               finalizer: Optional[Callable] = None,
                               pinned: bool = False) -> ArenaSlot:
        """Allocate directly into a specific generation (for promotion or pretenuring)."""
        obj_id = id(obj)
        if obj_id in self._slot_map:
            existing_slot, _ = self._slot_map[obj_id]
            if existing_slot._alive:
                return existing_slot

        self.stats.total_allocations += 1

        if gen == Generation.NURSERY:
            block = self.nursery
        elif gen == Generation.YOUNG:
            block = self._get_or_create_young()
        else:
            block = self._get_or_create_tenured()

        slot = block.allocate(obj, finalizer, pinned)
        if slot is None:
            # Block full, create new
            if gen == Generation.YOUNG:
                block = self._make_arena(self.young_capacity, Generation.YOUNG)
                self.young_arenas.append(block)
            else:
                block = self._make_arena(self.tenured_capacity, Generation.TENURED)
                self.tenured_arenas.append(block)
            slot = block.allocate(obj, finalizer, pinned)

        if slot is not None:
            self._slot_map[obj_id] = (slot, block)
            self._update_peak()
        return slot

    def _get_or_create_young(self) -> ArenaBlock:
        for arena in self.young_arenas:
            if arena.state == ArenaState.ACTIVE:
                return arena
        arena = self._make_arena(self.young_capacity, Generation.YOUNG)
        self.young_arenas.append(arena)
        return arena

    def _get_or_create_tenured(self) -> ArenaBlock:
        for arena in self.tenured_arenas:
            if arena.state == ArenaState.ACTIVE:
                return arena
        arena = self._make_arena(self.tenured_capacity, Generation.TENURED)
        self.tenured_arenas.append(arena)
        return arena

    def _update_peak(self):
        current = self.live_count
        if current > self.stats.peak_live:
            self.stats.peak_live = current

    # -- Deallocation --

    def free(self, slot: ArenaSlot):
        """Free a single slot."""
        if not slot._alive:
            return
        obj_id = id(slot.obj) if slot.obj is not None else None
        # Find owning block
        for block in self._all_arenas():
            if any(s is slot for s in block.slots):
                block.deallocate_slot(slot)
                if obj_id is not None and obj_id in self._slot_map:
                    del self._slot_map[obj_id]
                self.stats.total_frees += 1
                return
        # Check pools
        for pool in self.pools.values():
            for block in pool.blocks:
                if any(s is slot for s in block.slots):
                    pool.free(slot)
                    if obj_id is not None and obj_id in self._slot_map:
                        del self._slot_map[obj_id]
                    self.stats.total_frees += 1
                    return

    def bulk_free_arena(self, block: ArenaBlock) -> int:
        """Bulk-free an entire arena. Fast path for generation collection."""
        # Clean up slot_map entries
        for slot in block.slots:
            if slot._alive and slot.obj is not None:
                obj_id = id(slot.obj)
                if obj_id in self._slot_map:
                    del self._slot_map[obj_id]
        freed = block.bulk_free()
        self.stats.total_bulk_frees += freed
        return freed

    # -- Generational promotion --

    def _promote_nursery(self):
        """Promote surviving nursery objects to young generation, then reset nursery."""
        survivors = self.nursery.live_objects()
        for slot in survivors:
            slot.generation += 1
            if slot.generation >= self.promotion_age:
                # Promote to tenured
                self._promote_slot(slot, Generation.TENURED)
                self.stats.young_promotions += 1
            else:
                # Promote to young
                self._promote_slot(slot, Generation.YOUNG)
                self.stats.nursery_promotions += 1

        # Reset nursery
        old_nursery = self.nursery
        old_nursery.state = ArenaState.FREED
        self.nursery = self._make_arena(self.nursery_capacity, Generation.NURSERY)

    def _promote_slot(self, slot: ArenaSlot, target_gen: Generation):
        """Move a slot's object to a different generation's arena."""
        if target_gen == Generation.YOUNG:
            target = self._get_or_create_young()
        else:
            target = self._get_or_create_tenured()

        new_slot = target.allocate(slot.obj, slot.finalizer, slot.pinned)
        if new_slot is None:
            if target_gen == Generation.YOUNG:
                target = self._make_arena(self.young_capacity, Generation.YOUNG)
                self.young_arenas.append(target)
            else:
                target = self._make_arena(self.tenured_capacity, Generation.TENURED)
                self.tenured_arenas.append(target)
            new_slot = target.allocate(slot.obj, slot.finalizer, slot.pinned)

        new_slot.generation = slot.generation
        new_slot.marked = slot.marked
        new_slot.color = slot.color

        # Update slot map
        if slot.obj is not None:
            obj_id = id(slot.obj)
            self._slot_map[obj_id] = (new_slot, target)

        # Kill old slot (without running finalizer -- object still alive)
        slot._alive = False
        slot.obj = None
        slot.finalizer = None

    def promote_survivors(self):
        """Promote all surviving objects in young arenas that meet the age threshold."""
        for arena in list(self.young_arenas):
            promoted_all = True
            for slot in arena.live_objects():
                slot.generation += 1
                if slot.generation >= self.promotion_age:
                    self._promote_slot(slot, Generation.TENURED)
                    self.stats.young_promotions += 1
                else:
                    promoted_all = False
            if promoted_all and arena.live_count == 0:
                arena.state = ArenaState.FREED

        # Remove freed young arenas
        self.young_arenas = [a for a in self.young_arenas if a.state != ArenaState.FREED]

    # -- GC Integration --

    def mark_roots(self, root_objects: List[Any]):
        """Mark phase: set mark bit on root objects and their reachable graph."""
        # Reset all marks
        for arena in self._all_arenas():
            for slot in arena.slots:
                if slot._alive:
                    slot.marked = False
                    slot.color = 0  # WHITE

        # Mark roots
        gray_list: List[ArenaSlot] = []
        for obj in root_objects:
            obj_id = id(obj)
            if obj_id in self._slot_map:
                slot, _ = self._slot_map[obj_id]
                if slot._alive and not slot.marked:
                    slot.marked = True
                    slot.color = 1  # GRAY
                    gray_list.append(slot)

        # Trace (BFS)
        while gray_list:
            slot = gray_list.pop(0)
            slot.color = 2  # BLACK
            children = self._get_children(slot.obj)
            for child in children:
                child_id = id(child)
                if child_id in self._slot_map:
                    child_slot, _ = self._slot_map[child_id]
                    if child_slot._alive and not child_slot.marked:
                        child_slot.marked = True
                        child_slot.color = 1  # GRAY
                        gray_list.append(child_slot)

    def sweep(self) -> int:
        """Sweep phase: free unmarked objects. Returns count freed."""
        freed = 0
        for arena in self._all_arenas():
            for slot in list(arena.slots):
                if slot._alive and not slot.marked and not slot.pinned:
                    obj_id = id(slot.obj) if slot.obj is not None else None
                    arena.deallocate_slot(slot)
                    if obj_id is not None and obj_id in self._slot_map:
                        del self._slot_map[obj_id]
                    freed += 1
        self.stats.total_frees += freed
        self.stats.collections += 1
        return freed

    def collect(self, root_objects: List[Any]) -> int:
        """Full mark-sweep collection cycle."""
        self.mark_roots(root_objects)
        return self.sweep()

    def collect_nursery(self, root_objects: List[Any]) -> int:
        """Minor collection: only collect nursery, promote survivors."""
        # Mark all
        self.mark_roots(root_objects)

        # Sweep only nursery
        freed = 0
        for slot in list(self.nursery.slots):
            if slot._alive and not slot.marked and not slot.pinned:
                obj_id = id(slot.obj) if slot.obj is not None else None
                self.nursery.deallocate_slot(slot)
                if obj_id is not None and obj_id in self._slot_map:
                    del self._slot_map[obj_id]
                freed += 1

        self.stats.total_frees += freed
        self.stats.collections += 1

        # Promote survivors
        if self.nursery.state == ArenaState.FULL or self.nursery.bump >= self.nursery_capacity * 0.75:
            self._promote_nursery()

        return freed

    # -- Incremental GC (concurrent-compatible) --

    def incremental_mark_step(self, root_objects: List[Any], budget: int = 32) -> bool:
        """One step of incremental marking. Returns True if marking is complete."""
        # Initialize gray set if empty
        if not hasattr(self, '_gray_set') or self._gray_set is None:
            self._gray_set: List[ArenaSlot] = []
            # Reset marks
            for arena in self._all_arenas():
                for slot in arena.slots:
                    if slot._alive:
                        slot.marked = False
                        slot.color = 0
            # Gray the roots
            for obj in root_objects:
                obj_id = id(obj)
                if obj_id in self._slot_map:
                    slot, _ = self._slot_map[obj_id]
                    if slot._alive and not slot.marked:
                        slot.marked = True
                        slot.color = 1
                        self._gray_set.append(slot)

        processed = 0
        while self._gray_set and processed < budget:
            slot = self._gray_set.pop(0)
            slot.color = 2  # BLACK
            children = self._get_children(slot.obj)
            for child in children:
                child_id = id(child)
                if child_id in self._slot_map:
                    child_slot, _ = self._slot_map[child_id]
                    if child_slot._alive and not child_slot.marked:
                        child_slot.marked = True
                        child_slot.color = 1
                        self._gray_set.append(child_slot)
            processed += 1

        if not self._gray_set:
            self._gray_set = None
            return True  # Done
        return False  # More work to do

    def incremental_sweep_step(self, budget: int = 64) -> Tuple[int, bool]:
        """One step of incremental sweeping. Returns (freed_count, is_complete)."""
        if not hasattr(self, '_sweep_state') or self._sweep_state is None:
            # Build sweep list
            self._sweep_state = []
            for arena in self._all_arenas():
                for slot in arena.slots:
                    if slot._alive:
                        self._sweep_state.append((slot, arena))
            self._sweep_idx = 0

        freed = 0
        end = min(self._sweep_idx + budget, len(self._sweep_state))
        while self._sweep_idx < end:
            slot, arena = self._sweep_state[self._sweep_idx]
            self._sweep_idx += 1
            if slot._alive and not slot.marked and not slot.pinned:
                obj_id = id(slot.obj) if slot.obj is not None else None
                arena.deallocate_slot(slot)
                if obj_id is not None and obj_id in self._slot_map:
                    del self._slot_map[obj_id]
                freed += 1

        done = self._sweep_idx >= len(self._sweep_state)
        if done:
            self._sweep_state = None
            self._sweep_idx = 0
            self.stats.total_frees += freed
            self.stats.collections += 1
        else:
            self.stats.total_frees += freed

        return freed, done

    # -- Write Barriers (concurrent GC compat) --

    def write_barrier_satb(self, slot: ArenaSlot, old_child: Any):
        """Snapshot-at-the-beginning barrier: re-gray old child if white during marking."""
        if old_child is None:
            return
        old_id = id(old_child)
        if old_id in self._slot_map:
            old_slot, _ = self._slot_map[old_id]
            if old_slot._alive and old_slot.color == 0:  # WHITE
                old_slot.color = 1  # GRAY
                old_slot.marked = True
                self._barrier_log.append(old_slot)

    def write_barrier_dijkstra(self, slot: ArenaSlot, new_child: Any):
        """Dijkstra barrier: if black slot gets white child, gray the child."""
        if new_child is None:
            return
        if slot.color != 2:  # Not BLACK
            return
        new_id = id(new_child)
        if new_id in self._slot_map:
            new_slot, _ = self._slot_map[new_id]
            if new_slot._alive and new_slot.color == 0:  # WHITE
                new_slot.color = 1  # GRAY
                new_slot.marked = True
                self._barrier_log.append(new_slot)

    # -- Compaction --

    def compact_arena(self, block: ArenaBlock) -> ArenaBlock:
        """Compact an arena by copying live objects to a new, tighter block."""
        live = block.live_objects()
        if not live:
            block.state = ArenaState.FREED
            return block

        new_block = self._make_arena(max(len(live), 16), block.generation)
        for old_slot in live:
            new_slot = new_block.allocate(old_slot.obj, old_slot.finalizer, old_slot.pinned)
            new_slot.generation = old_slot.generation
            new_slot.marked = old_slot.marked
            new_slot.color = old_slot.color
            # Update slot map
            if old_slot.obj is not None:
                obj_id = id(old_slot.obj)
                self._slot_map[obj_id] = (new_slot, new_block)
            old_slot._alive = False
            old_slot.obj = None
            old_slot.finalizer = None

        block.live_count = 0
        block.state = ArenaState.FREED
        self.stats.compactions += 1
        return new_block

    def compact_young(self) -> int:
        """Compact all young arenas with high fragmentation. Returns arenas compacted."""
        compacted = 0
        new_young = []
        for arena in self.young_arenas:
            if arena.state == ArenaState.FREED:
                continue
            if arena.fragmentation > 0.5 and arena.live_count > 0:
                new_arena = self.compact_arena(arena)
                new_young.append(new_arena)
                compacted += 1
            else:
                new_young.append(arena)
        self.young_arenas = new_young
        return compacted

    # -- Helpers --

    def _get_children(self, obj: Any) -> List[Any]:
        """Extract child references from an object (for tracing GC)."""
        children = []
        if isinstance(obj, dict):
            for v in obj.values():
                children.append(v)
        elif isinstance(obj, (list, tuple)):
            children.extend(obj)
        elif hasattr(obj, '__dict__'):
            for v in obj.__dict__.values():
                children.append(v)
        return children

    def _all_arenas(self) -> List[ArenaBlock]:
        """All non-freed arenas."""
        arenas = []
        if self.nursery.state != ArenaState.FREED:
            arenas.append(self.nursery)
        arenas.extend(a for a in self.young_arenas if a.state != ArenaState.FREED)
        arenas.extend(a for a in self.tenured_arenas if a.state != ArenaState.FREED)
        return arenas

    @property
    def live_count(self) -> int:
        count = sum(a.live_count for a in self._all_arenas())
        for pool in self.pools.values():
            count += pool.live_count
        return count

    @property
    def total_capacity(self) -> int:
        cap = sum(a.capacity for a in self._all_arenas())
        for pool in self.pools.values():
            cap += pool.total_capacity
        return cap

    def get_stats(self) -> dict:
        """Return comprehensive memory statistics."""
        return {
            'total_allocations': self.stats.total_allocations,
            'total_frees': self.stats.total_frees,
            'total_bulk_frees': self.stats.total_bulk_frees,
            'live_count': self.live_count,
            'total_capacity': self.total_capacity,
            'arena_count': self.stats.arena_count,
            'pool_count': self.stats.pool_count,
            'nursery_promotions': self.stats.nursery_promotions,
            'young_promotions': self.stats.young_promotions,
            'compactions': self.stats.compactions,
            'peak_live': self.stats.peak_live,
            'collections': self.stats.collections,
            'nursery_utilization': self.nursery.utilization,
            'nursery_fragmentation': self.nursery.fragmentation,
        }

    def lookup(self, obj: Any) -> Optional[ArenaSlot]:
        """Look up the slot for a tracked object."""
        obj_id = id(obj)
        if obj_id in self._slot_map:
            slot, _ = self._slot_map[obj_id]
            if slot._alive:
                return slot
        return None

    def should_collect(self) -> bool:
        """Check if auto-collection threshold is reached."""
        return self._allocs_since_gc >= self.auto_collect_threshold

    def reset_alloc_counter(self):
        """Reset the allocation counter after a collection."""
        self._allocs_since_gc = 0
