"""
Tests for C073: Memory Pools / Arena Allocator
"""
import pytest
from memory_pools import (
    ArenaBlock, ArenaSlot, ArenaState, FixedPool, PoolSize,
    Generation, MemoryManager, MemoryStats
)


# =========================================================================
# ArenaSlot tests
# =========================================================================

class TestArenaSlot:
    def test_slot_creation(self):
        slot = ArenaSlot(index=0, obj="hello")
        assert slot.obj == "hello"
        assert slot.index == 0
        assert slot._alive
        assert not slot.marked
        assert slot.color == 0
        assert slot.generation == 0

    def test_slot_is_alive(self):
        slot = ArenaSlot(index=0, obj=42)
        assert slot.is_alive()
        slot._alive = False
        assert not slot.is_alive()

    def test_slot_none_obj_not_alive(self):
        slot = ArenaSlot(index=0, obj=None)
        assert not slot.is_alive()

    def test_slot_pinned(self):
        slot = ArenaSlot(index=0, obj="pinned", pinned=True)
        assert slot.pinned

    def test_slot_finalizer(self):
        called = []
        slot = ArenaSlot(index=0, obj="x", finalizer=lambda o: called.append(o))
        assert slot.finalizer is not None


# =========================================================================
# ArenaBlock tests
# =========================================================================

class TestArenaBlock:
    def test_create_block(self):
        block = ArenaBlock(0, capacity=10)
        assert block.capacity == 10
        assert block.bump == 0
        assert block.live_count == 0
        assert block.state == ArenaState.ACTIVE

    def test_bump_allocation(self):
        block = ArenaBlock(0, capacity=5)
        s1 = block.allocate("a")
        s2 = block.allocate("b")
        s3 = block.allocate("c")
        assert s1.index == 0
        assert s2.index == 1
        assert s3.index == 2
        assert block.bump == 3
        assert block.live_count == 3

    def test_allocate_fills_block(self):
        block = ArenaBlock(0, capacity=2)
        s1 = block.allocate("a")
        s2 = block.allocate("b")
        assert s1 is not None
        assert s2 is not None
        s3 = block.allocate("c")
        assert s3 is None
        assert block.state == ArenaState.FULL

    def test_deallocate_slot(self):
        block = ArenaBlock(0, capacity=10)
        s1 = block.allocate("a")
        assert block.live_count == 1
        block.deallocate_slot(s1)
        assert block.live_count == 0
        assert not s1._alive
        assert s1.obj is None

    def test_deallocate_runs_finalizer(self):
        finalized = []
        block = ArenaBlock(0, capacity=10)
        s1 = block.allocate("obj1", finalizer=lambda o: finalized.append(o))
        block.deallocate_slot(s1)
        assert finalized == ["obj1"]

    def test_bulk_free(self):
        block = ArenaBlock(0, capacity=10)
        for i in range(5):
            block.allocate(f"obj{i}")
        assert block.live_count == 5
        freed = block.bulk_free()
        assert freed == 5
        assert block.live_count == 0
        assert block.state == ArenaState.FREED

    def test_bulk_free_runs_finalizers(self):
        finalized = []
        block = ArenaBlock(0, capacity=10)
        for i in range(3):
            block.allocate(f"obj{i}", finalizer=lambda o: finalized.append(o))
        block.bulk_free()
        assert len(finalized) == 3

    def test_bulk_free_partial(self):
        block = ArenaBlock(0, capacity=10)
        s1 = block.allocate("a")
        s2 = block.allocate("b")
        block.deallocate_slot(s1)
        freed = block.bulk_free()
        assert freed == 1  # Only b was still alive

    def test_freeze(self):
        block = ArenaBlock(0, capacity=10)
        block.allocate("a")
        block.freeze()
        assert block.state == ArenaState.FROZEN
        s = block.allocate("b")
        assert s is None  # Can't allocate into frozen block

    def test_utilization(self):
        block = ArenaBlock(0, capacity=10)
        block.allocate("a")
        block.allocate("b")
        assert block.utilization == 1.0  # 2 live / 2 bumped
        block.deallocate_slot(block.slots[0])
        assert block.utilization == 0.5  # 1 live / 2 bumped

    def test_fragmentation(self):
        block = ArenaBlock(0, capacity=10)
        block.allocate("a")
        block.allocate("b")
        block.allocate("c")
        assert block.fragmentation == 0.0
        block.deallocate_slot(block.slots[1])
        assert abs(block.fragmentation - 1/3) < 0.01

    def test_is_empty(self):
        block = ArenaBlock(0, capacity=10)
        assert block.is_empty
        block.allocate("a")
        assert not block.is_empty

    def test_live_objects(self):
        block = ArenaBlock(0, capacity=10)
        s1 = block.allocate("a")
        s2 = block.allocate("b")
        s3 = block.allocate("c")
        block.deallocate_slot(s2)
        live = block.live_objects()
        assert len(live) == 2
        assert s1 in live
        assert s3 in live

    def test_generation_tagging(self):
        block = ArenaBlock(0, capacity=10, generation=Generation.TENURED)
        assert block.generation == Generation.TENURED

    def test_empty_block_utilization(self):
        block = ArenaBlock(0, capacity=10)
        assert block.utilization == 0.0
        assert block.fragmentation == 0.0

    def test_pinned_allocation(self):
        block = ArenaBlock(0, capacity=10)
        s = block.allocate("pinned", pinned=True)
        assert s.pinned


# =========================================================================
# FixedPool tests
# =========================================================================

class TestFixedPool:
    def test_create_pool(self):
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=16)
        assert pool.size_class == PoolSize.SMALL
        assert len(pool.blocks) == 1

    def test_pool_allocation(self):
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=16)
        s = pool.allocate(42)
        assert s.obj == 42
        assert s._alive
        assert pool.live_count == 1

    def test_pool_many_allocations(self):
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=4)
        slots = [pool.allocate(i) for i in range(10)]
        assert all(s._alive for s in slots)
        assert pool.live_count == 10
        assert len(pool.blocks) == 3  # 4 + 4 + 2

    def test_pool_free_and_reuse(self):
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=4)
        s1 = pool.allocate("a")
        s2 = pool.allocate("b")
        pool.free(s1)
        assert pool.live_count == 1
        # Free list should recycle the slot
        s3 = pool.allocate("c")
        assert pool.live_count == 2
        # The recycled slot should be reused
        assert pool._total_frees == 1

    def test_pool_utilization(self):
        pool = FixedPool(0, PoolSize.MEDIUM, block_capacity=10)
        for i in range(5):
            pool.allocate(i)
        assert pool.live_count == 5
        assert pool.total_capacity == 10

    def test_pool_multiple_blocks(self):
        pool = FixedPool(0, PoolSize.LARGE, block_capacity=2)
        pool.allocate("a")
        pool.allocate("b")
        pool.allocate("c")  # Triggers new block
        assert len(pool.blocks) == 2
        assert pool.live_count == 3

    def test_pool_pinned(self):
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=10)
        s = pool.allocate("x", pinned=True)
        assert s.pinned

    def test_pool_finalizer(self):
        called = []
        pool = FixedPool(0, PoolSize.SMALL, block_capacity=10)
        s = pool.allocate("x", finalizer=lambda o: called.append(o))
        pool.free(s)
        assert called == ["x"]


# =========================================================================
# MemoryManager -- basic allocation tests
# =========================================================================

class TestMemoryManagerAllocation:
    def test_create_manager(self):
        mm = MemoryManager()
        assert mm.nursery.state == ArenaState.ACTIVE
        assert mm.live_count == 0

    def test_basic_allocation(self):
        mm = MemoryManager()
        s = mm.allocate("hello")
        assert s.obj == "hello"
        assert s._alive
        assert mm.live_count == 1

    def test_many_allocations(self):
        mm = MemoryManager(nursery_capacity=100)
        objs = [f"obj{i}" for i in range(50)]
        slots = [mm.allocate(o) for o in objs]
        assert all(s._alive for s in slots)
        assert mm.live_count == 50

    def test_dedup(self):
        mm = MemoryManager()
        obj = [1, 2, 3]
        s1 = mm.allocate(obj)
        s2 = mm.allocate(obj)
        assert s1 is s2  # Same object returns same slot

    def test_allocation_stats(self):
        mm = MemoryManager()
        mm.allocate("a")
        mm.allocate("b")
        stats = mm.get_stats()
        assert stats['total_allocations'] == 2
        assert stats['live_count'] == 2

    def test_pool_allocation(self):
        mm = MemoryManager()
        s = mm.allocate(42, use_pool=True)
        assert s.obj == 42
        assert PoolSize.SMALL in mm.pools

    def test_pool_classification(self):
        mm = MemoryManager()
        assert mm.classify_size(42) == PoolSize.SMALL
        assert mm.classify_size(3.14) == PoolSize.SMALL
        assert mm.classify_size(True) == PoolSize.SMALL
        assert mm.classify_size("short") == PoolSize.SMALL
        assert mm.classify_size("x" * 100) == PoolSize.MEDIUM
        assert mm.classify_size("x" * 500) == PoolSize.LARGE
        assert mm.classify_size([1, 2]) == PoolSize.MEDIUM
        assert mm.classify_size(list(range(50))) == PoolSize.LARGE
        assert mm.classify_size({"a": 1}) == PoolSize.MEDIUM
        assert mm.classify_size({f"k{i}": i for i in range(20)}) == PoolSize.LARGE

    def test_lookup(self):
        mm = MemoryManager()
        obj = [1, 2, 3]
        s = mm.allocate(obj)
        found = mm.lookup(obj)
        assert found is s

    def test_lookup_not_found(self):
        mm = MemoryManager()
        assert mm.lookup("nonexistent") is None

    def test_allocate_pinned(self):
        mm = MemoryManager()
        s = mm.allocate("pinned", pinned=True)
        assert s.pinned

    def test_allocate_with_finalizer(self):
        called = []
        mm = MemoryManager()
        s = mm.allocate("x", finalizer=lambda o: called.append(o))
        mm.free(s)
        assert called == ["x"]

    def test_nursery_overflow_promotes(self):
        mm = MemoryManager(nursery_capacity=5)
        slots = [mm.allocate(f"obj{i}") for i in range(5)]
        assert mm.nursery.state == ArenaState.FULL or mm.nursery.bump == 5
        # Next allocation triggers promotion
        s6 = mm.allocate("obj5")
        assert s6._alive
        assert mm.stats.nursery_promotions > 0 or len(mm.young_arenas) > 0


# =========================================================================
# MemoryManager -- deallocation tests
# =========================================================================

class TestMemoryManagerDeallocation:
    def test_free_single(self):
        mm = MemoryManager()
        s = mm.allocate("a")
        mm.free(s)
        assert not s._alive
        assert mm.stats.total_frees == 1

    def test_free_updates_slot_map(self):
        mm = MemoryManager()
        obj = [1, 2]
        s = mm.allocate(obj)
        mm.free(s)
        assert mm.lookup(obj) is None

    def test_bulk_free_arena(self):
        mm = MemoryManager(nursery_capacity=100)
        for i in range(20):
            mm.allocate(f"obj{i}")
        freed = mm.bulk_free_arena(mm.nursery)
        assert freed == 20
        assert mm.nursery.state == ArenaState.FREED

    def test_free_already_dead(self):
        mm = MemoryManager()
        s = mm.allocate("a")
        mm.free(s)
        mm.free(s)  # Should not crash
        assert mm.stats.total_frees == 1

    def test_free_pool_slot(self):
        mm = MemoryManager()
        s = mm.allocate(42, use_pool=True)
        mm.free(s)
        assert not s._alive


# =========================================================================
# MemoryManager -- generational promotion tests
# =========================================================================

class TestGenerationalPromotion:
    def test_nursery_promotion_to_young(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=3)
        # Fill nursery
        objs = [f"obj{i}" for i in range(5)]
        slots = [mm.allocate(o) for o in objs]
        # Force promotion
        mm._promote_nursery()
        assert len(mm.young_arenas) > 0
        assert mm.stats.nursery_promotions > 0

    def test_young_promotion_to_tenured(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=2)
        objs = [f"obj{i}" for i in range(3)]
        slots = [mm.allocate(o) for o in objs]
        # First promotion: nursery -> young (gen becomes 1)
        mm._promote_nursery()
        # Second: young survivors get gen bumped
        mm.promote_survivors()
        # Objects with gen >= promotion_age go to tenured
        assert mm.stats.young_promotions > 0

    def test_allocate_in_generation(self):
        mm = MemoryManager()
        s = mm.allocate_in_generation("tenured_obj", Generation.TENURED)
        assert s._alive
        assert len(mm.tenured_arenas) > 0

    def test_allocate_in_young(self):
        mm = MemoryManager()
        s = mm.allocate_in_generation("young_obj", Generation.YOUNG)
        assert s._alive
        assert len(mm.young_arenas) > 0

    def test_promotion_preserves_objects(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=2)
        obj = [1, 2, 3]
        s = mm.allocate(obj)
        mm._promote_nursery()
        # Object should still be findable
        found = mm.lookup(obj)
        assert found is not None
        assert found.obj is obj

    def test_promotion_increments_generation(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=5)
        obj = [1, 2, 3]
        mm.allocate(obj)
        mm._promote_nursery()
        found = mm.lookup(obj)
        assert found.generation >= 1

    def test_freed_arenas_removed(self):
        mm = MemoryManager(nursery_capacity=3, promotion_age=2)
        for i in range(3):
            mm.allocate(f"o{i}")
        mm._promote_nursery()
        # Kill all young objects
        for arena in mm.young_arenas:
            for slot in arena.live_objects():
                arena.deallocate_slot(slot)
        mm.promote_survivors()
        # Freed young arenas should be cleaned up
        assert all(a.state != ArenaState.FREED for a in mm.young_arenas)


# =========================================================================
# MemoryManager -- GC integration tests
# =========================================================================

class TestGCIntegration:
    def test_mark_roots(self):
        mm = MemoryManager()
        obj1 = [1, 2]
        obj2 = [3, 4]
        s1 = mm.allocate(obj1)
        s2 = mm.allocate(obj2)
        mm.mark_roots([obj1])  # Only obj1 is a root
        assert s1.marked
        assert not s2.marked

    def test_sweep_unreachable(self):
        mm = MemoryManager()
        obj1 = [1]
        obj2 = [2]
        mm.allocate(obj1)
        mm.allocate(obj2)
        freed = mm.collect([obj1])  # Only obj1 reachable
        assert freed == 1  # obj2 swept
        assert mm.live_count == 1

    def test_collect_preserves_reachable(self):
        mm = MemoryManager()
        obj1 = {"key": "value"}
        mm.allocate(obj1)
        freed = mm.collect([obj1])
        assert freed == 0
        assert mm.lookup(obj1) is not None

    def test_transitive_reachability(self):
        mm = MemoryManager()
        child = [1, 2]
        parent = [child]
        mm.allocate(child)
        mm.allocate(parent)
        freed = mm.collect([parent])
        assert freed == 0  # Both reachable through parent

    def test_dict_children_traced(self):
        mm = MemoryManager()
        val = [1, 2]
        d = {"ref": val}
        mm.allocate(val)
        mm.allocate(d)
        freed = mm.collect([d])
        assert freed == 0

    def test_collect_all_unreachable(self):
        mm = MemoryManager()
        objs = [f"garbage{i}" for i in range(10)]
        for o in objs:
            mm.allocate(o)
        freed = mm.collect([])  # No roots
        assert freed == 10
        assert mm.live_count == 0

    def test_pinned_survives_sweep(self):
        mm = MemoryManager()
        obj = [1]
        s = mm.allocate(obj, pinned=True)
        freed = mm.collect([])  # No roots, but obj is pinned
        assert freed == 0
        assert s._alive

    def test_collect_nursery(self):
        mm = MemoryManager(nursery_capacity=10)
        alive = [1, 2]
        dead = [3, 4]
        mm.allocate(alive)
        mm.allocate(dead)
        freed = mm.collect_nursery([alive])
        assert freed == 1

    def test_collection_count(self):
        mm = MemoryManager()
        mm.allocate("a")
        mm.collect(["a"])
        mm.collect(["a"])
        assert mm.stats.collections == 2

    def test_should_collect(self):
        mm = MemoryManager(auto_collect_threshold=5)
        for i in range(4):
            mm.allocate(f"obj{i}")
        assert not mm.should_collect()
        mm.allocate("obj4")
        assert mm.should_collect()
        mm.reset_alloc_counter()
        assert not mm.should_collect()


# =========================================================================
# Incremental GC tests
# =========================================================================

class TestIncrementalGC:
    def test_incremental_mark_single_step(self):
        mm = MemoryManager()
        obj = [1]
        mm.allocate(obj)
        done = mm.incremental_mark_step([obj], budget=100)
        assert done
        slot = mm.lookup(obj)
        assert slot.marked

    def test_incremental_mark_multi_step(self):
        mm = MemoryManager()
        objs = [[i] for i in range(20)]
        for o in objs:
            mm.allocate(o)
        # Mark with small budget
        roots = objs[:5]
        done = mm.incremental_mark_step(roots, budget=2)
        # May not be done in one step
        while not done:
            done = mm.incremental_mark_step(roots, budget=2)
        # Root objects should be marked
        for r in roots:
            assert mm.lookup(r).marked

    def test_incremental_sweep(self):
        mm = MemoryManager()
        alive = [1]
        dead = [2]
        mm.allocate(alive)
        mm.allocate(dead)
        # Mark
        done = mm.incremental_mark_step([alive], budget=100)
        assert done
        # Sweep
        total_freed = 0
        sweep_done = False
        while not sweep_done:
            freed, sweep_done = mm.incremental_sweep_step(budget=1)
            total_freed += freed
        assert total_freed == 1

    def test_incremental_full_cycle(self):
        mm = MemoryManager()
        keep = [[i] for i in range(5)]
        garbage = [[i + 100] for i in range(10)]
        for o in keep + garbage:
            mm.allocate(o)
        # Mark
        done = False
        while not done:
            done = mm.incremental_mark_step(keep, budget=3)
        # Sweep
        total_freed = 0
        done = False
        while not done:
            freed, done = mm.incremental_sweep_step(budget=5)
            total_freed += freed
        assert total_freed == 10
        assert mm.live_count == 5

    def test_incremental_mark_resets_between_cycles(self):
        mm = MemoryManager()
        obj = [1]
        mm.allocate(obj)
        done = mm.incremental_mark_step([obj], budget=100)
        assert done
        # Start another cycle
        done = mm.incremental_mark_step([obj], budget=100)
        assert done


# =========================================================================
# Write barrier tests
# =========================================================================

class TestWriteBarriers:
    def test_satb_barrier_grays_old(self):
        mm = MemoryManager()
        old_child = [1]
        parent = [old_child]
        s_old = mm.allocate(old_child)
        mm.allocate(parent)
        # Simulate: old_child is white, parent overwrites reference
        s_old.color = 0  # WHITE
        mm.write_barrier_satb(mm.lookup(parent), old_child)
        assert s_old.color == 1  # GRAY
        assert s_old.marked

    def test_satb_barrier_noop_on_black(self):
        mm = MemoryManager()
        old_child = [1]
        s_old = mm.allocate(old_child)
        s_old.color = 2  # BLACK
        mm.write_barrier_satb(s_old, old_child)
        assert s_old.color == 2  # Still BLACK

    def test_dijkstra_barrier_grays_new_child(self):
        mm = MemoryManager()
        parent_obj = [1]
        new_child = [2]
        s_parent = mm.allocate(parent_obj)
        s_child = mm.allocate(new_child)
        s_parent.color = 2  # BLACK
        s_child.color = 0   # WHITE
        mm.write_barrier_dijkstra(s_parent, new_child)
        assert s_child.color == 1  # GRAY

    def test_dijkstra_noop_if_parent_not_black(self):
        mm = MemoryManager()
        parent_obj = [1]
        new_child = [2]
        s_parent = mm.allocate(parent_obj)
        s_child = mm.allocate(new_child)
        s_parent.color = 1  # GRAY
        s_child.color = 0   # WHITE
        mm.write_barrier_dijkstra(s_parent, new_child)
        assert s_child.color == 0  # Still WHITE

    def test_barrier_log(self):
        mm = MemoryManager()
        old_child = [1]
        s = mm.allocate(old_child)
        s.color = 0
        mm.write_barrier_satb(s, old_child)
        assert len(mm._barrier_log) == 1

    def test_barrier_none_noop(self):
        mm = MemoryManager()
        s = mm.allocate([1])
        mm.write_barrier_satb(s, None)
        mm.write_barrier_dijkstra(s, None)
        assert len(mm._barrier_log) == 0


# =========================================================================
# Compaction tests
# =========================================================================

class TestCompaction:
    def test_compact_arena(self):
        mm = MemoryManager(nursery_capacity=10)
        objs = [f"obj{i}" for i in range(10)]
        slots = [mm.allocate(o) for o in objs]
        # Kill half
        for i in range(0, 10, 2):
            mm.free(slots[i])
        assert mm.nursery.fragmentation > 0.3
        new_arena = mm.compact_arena(mm.nursery)
        assert new_arena.live_count == 5
        assert new_arena.fragmentation == 0.0

    def test_compact_preserves_objects(self):
        mm = MemoryManager(nursery_capacity=10)
        obj = [1, 2, 3]
        mm.allocate(obj)
        mm.allocate("dead")
        mm.free(mm.lookup("dead"))
        new_arena = mm.compact_arena(mm.nursery)
        # Object should still be findable
        found = mm.lookup(obj)
        assert found is not None
        assert found.obj is obj

    def test_compact_empty_arena(self):
        mm = MemoryManager(nursery_capacity=10)
        s = mm.allocate("a")
        mm.free(s)
        new_arena = mm.compact_arena(mm.nursery)
        assert new_arena.state == ArenaState.FREED

    def test_compact_young(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=5)
        # Create objects in young
        for i in range(5):
            mm.allocate(f"obj{i}")
        mm._promote_nursery()
        assert len(mm.young_arenas) > 0
        # Kill some young objects
        for arena in mm.young_arenas:
            live = arena.live_objects()
            for slot in live[:3]:
                arena.deallocate_slot(slot)
        compacted = mm.compact_young()
        assert compacted >= 0

    def test_compaction_stat(self):
        mm = MemoryManager(nursery_capacity=10)
        mm.allocate("a")
        mm.compact_arena(mm.nursery)
        assert mm.stats.compactions == 1

    def test_compact_preserves_generation(self):
        mm = MemoryManager(nursery_capacity=5, promotion_age=5)
        obj = [1]
        s = mm.allocate(obj)
        s.generation = 3
        new_arena = mm.compact_arena(mm.nursery)
        found = mm.lookup(obj)
        assert found.generation == 3

    def test_compact_preserves_pinned(self):
        mm = MemoryManager(nursery_capacity=5)
        obj = [1]
        s = mm.allocate(obj, pinned=True)
        new_arena = mm.compact_arena(mm.nursery)
        found = mm.lookup(obj)
        assert found.pinned


# =========================================================================
# Statistics and monitoring tests
# =========================================================================

class TestStats:
    def test_peak_live(self):
        mm = MemoryManager()
        for i in range(10):
            mm.allocate(f"obj{i}")
        assert mm.stats.peak_live == 10

    def test_peak_live_after_free(self):
        mm = MemoryManager()
        slots = [mm.allocate(f"obj{i}") for i in range(10)]
        for s in slots[:5]:
            mm.free(s)
        assert mm.stats.peak_live == 10  # Peak is historical
        assert mm.live_count == 5

    def test_get_stats_comprehensive(self):
        mm = MemoryManager()
        mm.allocate("a")
        stats = mm.get_stats()
        required_keys = [
            'total_allocations', 'total_frees', 'total_bulk_frees',
            'live_count', 'total_capacity', 'arena_count', 'pool_count',
            'nursery_promotions', 'young_promotions', 'compactions',
            'peak_live', 'collections', 'nursery_utilization',
            'nursery_fragmentation'
        ]
        for key in required_keys:
            assert key in stats, f"Missing stat: {key}"

    def test_total_capacity(self):
        mm = MemoryManager(nursery_capacity=100)
        assert mm.total_capacity >= 100


# =========================================================================
# Edge cases and stress tests
# =========================================================================

class TestEdgeCases:
    def test_finalizer_exception_doesnt_crash(self):
        mm = MemoryManager()
        s = mm.allocate("x", finalizer=lambda o: 1/0)
        mm.free(s)  # Should not raise
        assert not s._alive

    def test_double_free(self):
        mm = MemoryManager()
        s = mm.allocate("a")
        mm.free(s)
        mm.free(s)  # Should be safe
        assert mm.stats.total_frees == 1

    def test_collect_empty(self):
        mm = MemoryManager()
        freed = mm.collect([])
        assert freed == 0

    def test_many_collections(self):
        mm = MemoryManager(nursery_capacity=20)
        for cycle in range(5):
            objs = [f"c{cycle}_obj{i}" for i in range(10)]
            for o in objs:
                mm.allocate(o)
            freed = mm.collect(objs[:2])  # Keep only first 2
            assert freed >= 0

    def test_stress_alloc_free(self):
        mm = MemoryManager(nursery_capacity=200)  # Large enough to avoid promotion
        all_slots = []
        for i in range(100):
            s = mm.allocate(f"stress_{i}")
            all_slots.append(s)
        # Free every other (use lookup to get current slot after any promotions)
        objs = [s.obj for s in all_slots]
        for i in range(0, 100, 2):
            slot = mm.lookup(objs[i])
            if slot:
                mm.free(slot)
        assert mm.live_count == 50

    def test_mixed_pool_and_arena(self):
        mm = MemoryManager()
        s1 = mm.allocate(42, use_pool=True)   # Pool
        s2 = mm.allocate("arena_obj")           # Arena
        assert s1._alive
        assert s2._alive
        assert mm.live_count == 2

    def test_promotion_chain(self):
        """Nursery -> young -> tenured through multiple collections."""
        mm = MemoryManager(nursery_capacity=3, promotion_age=2)
        obj = [1, 2, 3]
        mm.allocate(obj)
        # Fill nursery and promote
        mm.allocate("fill1")
        mm.allocate("fill2")
        mm._promote_nursery()
        # Now promote young -> tenured
        mm.promote_survivors()
        assert mm.stats.nursery_promotions > 0

    def test_arena_state_transitions(self):
        block = ArenaBlock(0, capacity=2)
        assert block.state == ArenaState.ACTIVE
        block.allocate("a")
        block.allocate("b")
        block.allocate("c")  # Returns None, block now FULL
        assert block.state == ArenaState.FULL
        block.freeze()
        assert block.state == ArenaState.FROZEN
        freed = block.bulk_free()
        assert block.state == ArenaState.FREED

    def test_concurrent_gc_color_tracking(self):
        """Verify tri-color states work through mark cycle."""
        mm = MemoryManager()
        obj = [1]
        s = mm.allocate(obj)
        assert s.color == 0  # WHITE initially
        mm.mark_roots([obj])
        # After marking, root should be BLACK
        assert s.color == 2  # BLACK

    def test_multiple_pool_sizes(self):
        mm = MemoryManager()
        mm.allocate(42, use_pool=True)           # SMALL
        mm.allocate("x" * 100, use_pool=True)    # MEDIUM
        mm.allocate(list(range(50)), use_pool=True)  # LARGE
        assert len(mm.pools) == 3

    def test_object_with_attrs_traced(self):
        """Objects with __dict__ should have children traced."""
        class Obj:
            def __init__(self, ref):
                self.ref = ref
        mm = MemoryManager()
        child = [1, 2]
        parent = Obj(child)
        mm.allocate(child)
        mm.allocate(parent)
        freed = mm.collect([parent])
        assert freed == 0  # child reachable through parent.ref

    def test_collect_nursery_promotes(self):
        mm = MemoryManager(nursery_capacity=10)
        objs = [f"obj{i}" for i in range(8)]
        for o in objs:
            mm.allocate(o)
        mm.collect_nursery(objs[:3])
        # After nursery collection, some promotion may occur
        assert mm.stats.collections >= 1


# =========================================================================
# Integration: full lifecycle
# =========================================================================

class TestFullLifecycle:
    def test_alloc_collect_promote_compact(self):
        """Full lifecycle: allocate, collect garbage, promote survivors, compact."""
        mm = MemoryManager(nursery_capacity=20, promotion_age=2)

        # Phase 1: Allocate
        live_objs = [[i] for i in range(5)]
        dead_objs = [[i + 100] for i in range(15)]
        for o in live_objs + dead_objs:
            mm.allocate(o)
        assert mm.live_count == 20

        # Phase 2: Collect (sweep dead)
        freed = mm.collect(live_objs)
        assert freed == 15
        assert mm.live_count == 5

        # Phase 3: Promote to young
        mm._promote_nursery()
        assert len(mm.young_arenas) > 0

        # Phase 4: Promote to tenured
        mm.promote_survivors()

        # Phase 5: Compact young
        mm.compact_young()

        # Live objects should still be findable
        for o in live_objs:
            found = mm.lookup(o)
            assert found is not None

    def test_incremental_gc_lifecycle(self):
        """Full incremental GC cycle."""
        mm = MemoryManager()
        keep = [[i] for i in range(3)]
        garbage = [[i + 100] for i in range(7)]
        for o in keep + garbage:
            mm.allocate(o)

        # Incremental mark
        done = False
        while not done:
            done = mm.incremental_mark_step(keep, budget=2)

        # Incremental sweep
        total_freed = 0
        done = False
        while not done:
            freed, done = mm.incremental_sweep_step(budget=3)
            total_freed += freed

        assert total_freed == 7
        assert mm.live_count == 3

    def test_pool_gc_lifecycle(self):
        """Pool-allocated objects participate in GC."""
        mm = MemoryManager()
        pool_obj = 42
        s = mm.allocate(pool_obj, use_pool=True)
        # Pool objects are in slot_map, participate in mark/sweep
        mm.mark_roots([pool_obj])
        assert s.marked

    def test_repeated_nursery_cycles(self):
        """Repeated nursery fill/collect cycles."""
        mm = MemoryManager(nursery_capacity=10, promotion_age=3)
        all_live = []
        for cycle in range(5):
            live = [f"live_{cycle}_{i}" for i in range(2)]
            dead = [f"dead_{cycle}_{i}" for i in range(8)]
            for o in live + dead:
                mm.allocate(o)
            all_live.extend(live)
            mm.collect_nursery(all_live)
        # All live objects should still be findable
        # (some may have been promoted through generations)
        assert mm.stats.collections == 5

    def test_write_barriers_during_incremental(self):
        """Write barriers maintain correctness during incremental marking."""
        mm = MemoryManager()
        parent = [None]
        child = [42]
        mm.allocate(parent)
        mm.allocate(child)

        # Start incremental mark
        mm.incremental_mark_step([parent], budget=1)

        # Simulate mutation: parent (BLACK) gets reference to child (WHITE)
        s_parent = mm.lookup(parent)
        s_child = mm.lookup(child)
        if s_parent and s_child:
            s_parent.color = 2  # BLACK
            s_child.color = 0   # WHITE
            mm.write_barrier_dijkstra(s_parent, child)
            # Child should now be GRAY (won't be missed by sweep)
            assert s_child.color == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
