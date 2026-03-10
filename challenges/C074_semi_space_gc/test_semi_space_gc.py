"""
Tests for C074: Semi-Space Garbage Collector (Cheney's Algorithm)
"""

import pytest
from semi_space_gc import (
    ManagedObject, SemiSpace, LargeObjectSpace, GCStats,
    SemiSpaceCollector,
)


# ===========================================================================
# ManagedObject tests
# ===========================================================================

class TestManagedObject:
    def test_create_basic(self):
        obj = ManagedObject(42)
        assert obj.value == 42
        assert obj._alive
        assert not obj.forwarded
        assert obj.forward_to is None
        assert not obj.pinned
        assert obj.age == 0

    def test_unique_ids(self):
        a = ManagedObject("a")
        b = ManagedObject("b")
        assert a.obj_id != b.obj_id

    def test_size_estimation_none(self):
        obj = ManagedObject(None)
        assert obj.size == 8

    def test_size_estimation_int(self):
        obj = ManagedObject(42)
        assert obj.size == 16

    def test_size_estimation_string(self):
        obj = ManagedObject("hello")
        assert obj.size == 40 + 5

    def test_size_estimation_list(self):
        obj = ManagedObject([1, 2, 3])
        assert obj.size == 56 + 8 * 3

    def test_size_estimation_dict(self):
        obj = ManagedObject({"a": 1})
        assert obj.size == 64 + 16

    def test_explicit_size(self):
        obj = ManagedObject(42, size=100)
        assert obj.size == 100

    def test_is_alive(self):
        obj = ManagedObject(42)
        assert obj.is_alive()
        obj._alive = False
        assert not obj.is_alive()

    def test_is_alive_forwarded(self):
        obj = ManagedObject(42)
        obj.forwarded = True
        assert not obj.is_alive()

    def test_resolve_no_forward(self):
        obj = ManagedObject(42)
        assert obj.resolve() is obj

    def test_resolve_chain(self):
        a = ManagedObject("a")
        b = ManagedObject("b")
        c = ManagedObject("c")
        a.forwarded = True
        a.forward_to = b
        b.forwarded = True
        b.forward_to = c
        assert a.resolve() is c

    def test_pinned(self):
        obj = ManagedObject(42, pinned=True)
        assert obj.pinned

    def test_finalizer_stored(self):
        called = []
        obj = ManagedObject(42, finalizer=lambda v: called.append(v))
        assert obj.finalizer is not None


# ===========================================================================
# SemiSpace tests
# ===========================================================================

class TestSemiSpace:
    def test_create(self):
        space = SemiSpace(1024, "test")
        assert space.capacity == 1024
        assert space.bump == 0
        assert space.used_bytes == 0

    def test_allocate(self):
        space = SemiSpace(1024)
        obj = ManagedObject(42)
        assert space.allocate(obj)
        assert space.bump == 1
        assert space.used_bytes == obj.size

    def test_allocate_full(self):
        space = SemiSpace(16)
        obj = ManagedObject(42)  # size 16
        assert space.allocate(obj)
        obj2 = ManagedObject(43)  # size 16
        assert not space.allocate(obj2)

    def test_can_fit(self):
        space = SemiSpace(100)
        assert space.can_fit(50)
        assert space.can_fit(100)
        assert not space.can_fit(101)

    def test_reset(self):
        space = SemiSpace(1024)
        space.allocate(ManagedObject(1))
        space.allocate(ManagedObject(2))
        space.reset()
        assert space.bump == 0
        assert space.used_bytes == 0
        assert len(space.objects) == 0

    def test_live_count(self):
        space = SemiSpace(1024)
        a = ManagedObject(1)
        b = ManagedObject(2)
        space.allocate(a)
        space.allocate(b)
        assert space.live_count == 2
        a._alive = False
        assert space.live_count == 1

    def test_utilization(self):
        space = SemiSpace(100)
        obj = ManagedObject(None, size=50)
        space.allocate(obj)
        assert space.utilization == 0.5

    def test_utilization_empty(self):
        space = SemiSpace(0)
        assert space.utilization == 0.0


# ===========================================================================
# LargeObjectSpace tests
# ===========================================================================

class TestLargeObjectSpace:
    def test_should_use_los(self):
        los = LargeObjectSpace(threshold=1024)
        assert not los.should_use_los(100)
        assert los.should_use_los(1024)
        assert los.should_use_los(2000)

    def test_allocate_and_sweep(self):
        los = LargeObjectSpace(threshold=100)
        obj = ManagedObject("big", size=200)
        los.allocate(obj)
        assert los.live_count == 1
        # Don't mark it -> should be swept
        freed = los.sweep()
        assert freed == 1
        assert los.live_count == 0

    def test_mark_survives_sweep(self):
        los = LargeObjectSpace(threshold=100)
        obj = ManagedObject("big", size=200)
        los.allocate(obj)
        los.mark(obj.obj_id)
        freed = los.sweep()
        assert freed == 0
        assert los.live_count == 1

    def test_finalizer_on_sweep(self):
        los = LargeObjectSpace(threshold=100)
        called = []
        obj = ManagedObject("big", finalizer=lambda v: called.append(v), size=200)
        los.allocate(obj)
        los.sweep()
        assert called == ["big"]

    def test_total_size(self):
        los = LargeObjectSpace(threshold=100)
        a = ManagedObject("a", size=200)
        b = ManagedObject("b", size=300)
        los.allocate(a)
        los.allocate(b)
        assert los.total_size == 500


# ===========================================================================
# SemiSpaceCollector -- basic allocation
# ===========================================================================

class TestCollectorAllocation:
    def test_allocate_basic(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42)
        assert obj.value == 42
        assert obj._alive

    def test_allocate_multiple(self):
        gc = SemiSpaceCollector(space_size=4096)
        objs = [gc.allocate(i) for i in range(10)]
        assert len(objs) == 10
        assert gc.live_count == 10

    def test_allocate_pinned(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42, pinned=True)
        assert obj.pinned
        assert obj in gc.pinned

    def test_allocate_large(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100)
        obj = gc.allocate("x" * 200)
        assert gc.los.live_count == 1
        assert gc.stats.los_allocations == 1

    def test_allocate_out_of_memory(self):
        gc = SemiSpaceCollector(space_size=32, auto_collect=False)
        gc.allocate(1, size=16)
        gc.allocate(2, size=16)
        with pytest.raises(MemoryError):
            gc.allocate(3, size=16)

    def test_auto_collect_on_full(self):
        gc = SemiSpaceCollector(space_size=64, auto_collect=True)
        a = gc.allocate(1, size=32)
        gc.add_root(a)
        b = gc.allocate(2, size=32)
        # from-space full, auto-collect should trigger
        c = gc.allocate(3, size=32)
        assert c._alive
        assert gc.stats.total_collections >= 1

    def test_stats_tracking(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(1)
        gc.allocate(2)
        assert gc.stats.total_allocations == 2
        assert gc.stats.total_bytes_allocated > 0


# ===========================================================================
# SemiSpaceCollector -- reference graph
# ===========================================================================

class TestReferenceGraph:
    def test_set_references(self):
        gc = SemiSpaceCollector(space_size=4096)
        parent = gc.allocate("parent")
        child = gc.allocate("child")
        gc.set_references(parent, [child])
        children = gc.get_children(parent)
        assert len(children) == 1
        assert children[0].value == "child"

    def test_add_reference(self):
        gc = SemiSpaceCollector(space_size=4096)
        a = gc.allocate("a")
        b = gc.allocate("b")
        gc.add_reference(a, b)
        assert len(gc.get_children(a)) == 1

    def test_remove_reference(self):
        gc = SemiSpaceCollector(space_size=4096)
        a = gc.allocate("a")
        b = gc.allocate("b")
        gc.add_reference(a, b)
        gc.remove_reference(a, b)
        assert len(gc.get_children(a)) == 0

    def test_no_duplicate_references(self):
        gc = SemiSpaceCollector(space_size=4096)
        a = gc.allocate("a")
        b = gc.allocate("b")
        gc.add_reference(a, b)
        gc.add_reference(a, b)
        assert len(gc.get_children(a)) == 1


# ===========================================================================
# SemiSpaceCollector -- root management
# ===========================================================================

class TestRoots:
    def test_add_root(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42)
        gc.add_root(obj)
        assert obj in gc._roots

    def test_remove_root(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42)
        gc.add_root(obj)
        gc.remove_root(obj)
        assert obj not in gc._roots

    def test_clear_roots(self):
        gc = SemiSpaceCollector(space_size=4096)
        a = gc.allocate(1)
        b = gc.allocate(2)
        gc.add_root(a)
        gc.add_root(b)
        gc.clear_roots()
        assert len(gc._roots) == 0

    def test_no_duplicate_roots(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42)
        gc.add_root(obj)
        gc.add_root(obj)
        assert gc._roots.count(obj) == 1


# ===========================================================================
# SemiSpaceCollector -- basic collection
# ===========================================================================

class TestBasicCollection:
    def test_collect_empty(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        freed = gc.collect()
        assert freed == 0

    def test_collect_all_garbage(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(1)
        gc.allocate(2)
        gc.allocate(3)
        # No roots -> everything is garbage
        freed = gc.collect()
        assert freed == 3
        assert gc.live_count == 0

    def test_collect_preserves_roots(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        a = gc.allocate("keep")
        gc.allocate("discard")
        gc.add_root(a)
        freed = gc.collect()
        assert freed == 1
        assert gc.live_count == 1

    def test_collect_preserves_reachable(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        child = gc.allocate("child")
        gc.set_references(root, [child])
        gc.add_root(root)
        gc.allocate("garbage")
        freed = gc.collect()
        assert freed == 1
        assert gc.live_count == 2

    def test_collect_deep_graph(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        prev = root
        for i in range(10):
            child = gc.allocate(f"child_{i}")
            gc.set_references(prev, [child])
            prev = child
        gc.allocate("garbage1")
        gc.allocate("garbage2")
        freed = gc.collect()
        assert freed == 2
        assert gc.live_count == 11

    def test_collect_diamond_graph(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        left = gc.allocate("left")
        right = gc.allocate("right")
        shared = gc.allocate("shared")
        gc.set_references(root, [left, right])
        gc.set_references(left, [shared])
        gc.set_references(right, [shared])
        gc.add_root(root)
        gc.allocate("garbage")
        freed = gc.collect()
        assert freed == 1
        assert gc.live_count == 4

    def test_collect_spaces_swap(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        old_from = gc.from_space
        gc.allocate(1)
        gc.collect()
        # After collection, spaces swapped
        assert gc.from_space is not old_from

    def test_collect_updates_stats(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(1)
        gc.allocate(2)
        gc.collect()
        assert gc.stats.total_collections == 1
        assert gc.stats.total_objects_freed == 2

    def test_multiple_collections(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        for _ in range(5):
            gc.allocate("garbage")
            gc.collect()
        assert gc.stats.total_collections == 5
        assert gc.live_count == 1


# ===========================================================================
# Forwarding pointers
# ===========================================================================

class TestForwarding:
    def test_forwarding_after_collect(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        gc.collect()
        # Object should be resolvable
        resolved = obj.resolve()
        assert resolved._alive
        assert resolved.value == "hello"

    def test_resolve_follows_chain(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        gc.collect()
        gc.collect()
        # After two collections, forwarding chain exists
        resolved = obj.resolve()
        assert resolved.value == "hello"

    def test_get_value_follows_forward(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        gc.collect()
        assert gc.get_value(obj) == "hello"

    def test_set_value_follows_forward(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("old")
        gc.add_root(obj)
        gc.collect()
        gc.set_value(obj, "new")
        assert gc.get_value(obj) == "new"

    def test_is_alive_follows_forward(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        gc.collect()
        assert gc.is_alive(obj)

    def test_dead_object_after_collect(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("goodbye")
        gc.collect()
        assert not gc.is_alive(obj)
        assert gc.get_value(obj) is None


# ===========================================================================
# Pinned objects
# ===========================================================================

class TestPinnedObjects:
    def test_pinned_survives_without_root(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("pinned", pinned=True)
        gc.allocate("garbage")
        gc.collect()
        assert obj._alive
        assert obj.value == "pinned"

    def test_pinned_not_moved(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("pinned", pinned=True)
        gc.collect()
        assert not obj.forwarded

    def test_pinned_children_preserved(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        pinned = gc.allocate("pinned", pinned=True)
        child = gc.allocate("child")
        gc.set_references(pinned, [child])
        gc.allocate("garbage")
        gc.collect()
        # Child should be kept alive via pinned parent
        assert gc.live_count == 2


# ===========================================================================
# Finalizers
# ===========================================================================

class TestFinalizers:
    def test_finalizer_called_on_collect(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        called = []
        gc.allocate(42, finalizer=lambda v: called.append(v))
        gc.collect()
        assert called == [42]
        assert gc.stats.total_finalizers_run == 1

    def test_finalizer_not_called_for_live(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        called = []
        obj = gc.allocate(42, finalizer=lambda v: called.append(v))
        gc.add_root(obj)
        gc.collect()
        assert called == []

    def test_multiple_finalizers(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        called = []
        gc.allocate(1, finalizer=lambda v: called.append(v))
        gc.allocate(2, finalizer=lambda v: called.append(v))
        gc.allocate(3, finalizer=lambda v: called.append(v))
        gc.collect()
        assert sorted(called) == [1, 2, 3]

    def test_finalizer_exception_ignored(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(42, finalizer=lambda v: 1/0)
        gc.collect()  # Should not raise
        assert gc.stats.total_finalizers_run == 1


# ===========================================================================
# Large Object Space integration
# ===========================================================================

class TestLOSIntegration:
    def test_large_object_goes_to_los(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100)
        obj = gc.allocate("x" * 200)
        assert gc.los.live_count == 1
        assert gc.from_space.live_count == 0

    def test_los_collected_with_gc(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100, auto_collect=False)
        gc.allocate("x" * 200)
        freed = gc.collect()
        assert freed >= 1
        assert gc.los.live_count == 0

    def test_los_reachable_preserved(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100, auto_collect=False)
        root = gc.allocate("root")
        big = gc.allocate("x" * 200)
        gc.set_references(root, [big])
        gc.add_root(root)
        gc.collect()
        assert gc.los.live_count == 1

    def test_los_finalizer(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100, auto_collect=False)
        called = []
        gc.allocate("x" * 200, finalizer=lambda v: called.append(len(v)))
        gc.collect()
        assert called == [200]


# ===========================================================================
# Aging and generational
# ===========================================================================

class TestAging:
    def test_age_increments(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("survivor")
        gc.add_root(obj)
        gc.collect()
        resolved = obj.resolve()
        assert resolved.age == 1

    def test_age_increments_multiple(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("survivor")
        gc.add_root(obj)
        gc.collect()
        gc.collect()
        gc.collect()
        resolved = obj.resolve()
        assert resolved.age == 3


class TestGenerational:
    def test_enable_generational(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.enable_generational(tenured_size=8192, tenure_threshold=2)
        assert gc._generational
        assert gc._tenured is not None
        assert gc._tenure_threshold == 2

    def test_promotion_to_tenured(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=2)
        obj = gc.allocate("survivor")
        gc.add_root(obj)
        # Survive 2 collections -> age reaches threshold
        gc.collect()
        gc.collect()
        gc.collect()  # This should promote (age >= tenure_threshold)
        assert gc._tenured.live_count >= 1

    def test_generational_frees_garbage(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=3)
        root = gc.allocate("root")
        gc.add_root(root)
        gc.allocate("garbage")
        freed = gc.collect()
        assert freed == 1

    def test_tenured_reachable_from_root(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=1)
        root = gc.allocate("root")
        child = gc.allocate("child")
        gc.set_references(root, [child])
        gc.add_root(root)
        gc.collect()
        gc.collect()  # Should promote
        # Both should still be alive
        total_live = gc.live_count
        assert total_live >= 2

    def test_tenured_to_nursery_reference(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=1)
        old_obj = gc.allocate("old")
        gc.add_root(old_obj)
        gc.collect()
        gc.collect()  # Promote old_obj
        # Now allocate young object referenced by tenured
        young = gc.allocate("young")
        resolved_old = old_obj.resolve()
        gc.set_references(resolved_old, [young])
        gc.collect()
        # Young should survive via tenured reference
        assert gc.live_count >= 2


# ===========================================================================
# Full collection (major GC)
# ===========================================================================

class TestFullCollection:
    def test_collect_full_non_generational(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(1)
        gc.allocate(2)
        freed = gc.collect_full()
        assert freed == 2

    def test_collect_full_tenured(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=1)
        root = gc.allocate("root")
        garbage = gc.allocate("garbage_tenured")
        gc.add_root(root)
        gc.add_root(garbage)
        # Promote both
        gc.collect()
        gc.collect()
        # Now remove garbage from roots
        gc.remove_root(garbage.resolve())
        # Full collection should reclaim tenured garbage
        freed = gc.collect_full()
        assert freed >= 1

    def test_collect_full_preserves_live(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.enable_generational(tenured_size=8192, tenure_threshold=1)
        root = gc.allocate("keep")
        gc.add_root(root)
        gc.collect()
        gc.collect()  # Promote
        freed = gc.collect_full()
        assert gc.live_count >= 1


# ===========================================================================
# Object lookup and queries
# ===========================================================================

class TestLookup:
    def test_lookup_by_id(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate("hello")
        found = gc.lookup(obj.obj_id)
        assert found is not None
        assert found.value == "hello"

    def test_lookup_after_collect(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        oid = obj.obj_id
        gc.collect()
        found = gc.lookup(oid)
        assert found is not None
        assert found.value == "hello"

    def test_lookup_dead_returns_none(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("dead")
        oid = obj.obj_id
        gc.collect()
        found = gc.lookup(oid)
        assert found is None

    def test_get_age(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("hello")
        gc.add_root(obj)
        assert gc.get_age(obj) == 0
        gc.collect()
        assert gc.get_age(obj) >= 1


# ===========================================================================
# Statistics
# ===========================================================================

class TestStatistics:
    def test_initial_stats(self):
        gc = SemiSpaceCollector(space_size=4096)
        stats = gc.get_stats()
        assert stats['total_allocations'] == 0
        assert stats['total_collections'] == 0

    def test_allocation_stats(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(1)
        gc.allocate(2)
        stats = gc.get_stats()
        assert stats['total_allocations'] == 2
        assert stats['total_bytes_allocated'] > 0

    def test_collection_stats(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(1)
        gc.allocate(2)
        gc.collect()
        stats = gc.get_stats()
        assert stats['total_collections'] == 1
        assert stats['total_objects_freed'] == 2

    def test_peak_tracking(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        gc.allocate(1)
        gc.allocate(2)
        gc.allocate(3)
        gc.collect()
        stats = gc.get_stats()
        assert stats['peak_live_count'] == 3

    def test_survival_rate(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("keep")
        gc.add_root(root)
        gc.allocate("discard")
        gc.collect()
        stats = gc.get_stats()
        assert 0.0 < stats['last_survival_rate'] <= 1.0

    def test_collection_callback(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        called = []
        gc.set_collection_callback(lambda s: called.append(s.total_collections))
        gc.allocate(1)
        gc.collect()
        assert len(called) == 1

    def test_space_utilization_in_stats(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(1)
        stats = gc.get_stats()
        assert stats['from_space_capacity'] == 4096
        assert stats['from_space_used'] > 0


# ===========================================================================
# Heap dump
# ===========================================================================

class TestHeapDump:
    def test_dump_heap(self):
        gc = SemiSpaceCollector(space_size=4096)
        obj = gc.allocate(42)
        gc.add_root(obj)
        dump = gc.dump_heap()
        assert 'from_space' in dump
        assert 'to_space' in dump
        assert 'pinned' in dump
        assert 'los' in dump
        assert 'tenured' in dump
        assert 'roots' in dump
        assert 'references' in dump

    def test_dump_from_space_contents(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(42)
        dump = gc.dump_heap()
        assert len(dump['from_space']) == 1
        assert dump['from_space'][0]['alive']


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_collect_with_no_objects(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        freed = gc.collect()
        assert freed == 0
        assert gc.stats.total_collections == 1

    def test_collect_preserves_after_many_cycles(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("persistent")
        gc.add_root(root)
        for _ in range(20):
            gc.allocate("temp")
            gc.collect()
        assert gc.live_count == 1
        assert gc.get_value(root) == "persistent"

    def test_self_referencing_object(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        obj = gc.allocate("self_ref")
        gc.set_references(obj, [obj])  # Self-reference
        gc.add_root(obj)
        freed = gc.collect()
        assert freed == 0
        assert gc.live_count == 1

    def test_cycle_without_root(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        a = gc.allocate("a")
        b = gc.allocate("b")
        gc.set_references(a, [b])
        gc.set_references(b, [a])
        # No roots -> cycle should be collected
        freed = gc.collect()
        assert freed == 2

    def test_branching_graph(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        children = [gc.allocate(f"c{i}") for i in range(5)]
        gc.set_references(root, children)
        gc.allocate("garbage")
        freed = gc.collect()
        assert freed == 1
        assert gc.live_count == 6

    def test_zero_capacity_space(self):
        gc = SemiSpaceCollector(space_size=0, auto_collect=False)
        with pytest.raises(MemoryError):
            gc.allocate(1)

    def test_large_allocation_count(self):
        gc = SemiSpaceCollector(space_size=100000, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        for i in range(100):
            gc.allocate(f"obj_{i}")
        freed = gc.collect()
        assert freed == 100
        assert gc.live_count == 1

    def test_alternating_alloc_collect(self):
        gc = SemiSpaceCollector(space_size=4096, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        for i in range(10):
            gc.allocate(f"temp_{i}")
            gc.collect()
        assert gc.live_count == 1
        assert gc.stats.total_collections == 10


# ===========================================================================
# Properties
# ===========================================================================

class TestProperties:
    def test_live_count(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(1)
        gc.allocate(2)
        gc.allocate(3, pinned=True)
        assert gc.live_count == 3

    def test_live_bytes(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.allocate(1, size=100)
        gc.allocate(2, size=200)
        assert gc.live_bytes == 300

    def test_total_capacity(self):
        gc = SemiSpaceCollector(space_size=4096)
        assert gc.total_capacity == 4096 * 2

    def test_total_capacity_generational(self):
        gc = SemiSpaceCollector(space_size=4096)
        gc.enable_generational(tenured_size=8192)
        assert gc.total_capacity == 4096 * 2 + 8192


# ===========================================================================
# Stress tests
# ===========================================================================

class TestStress:
    def test_many_objects_survive(self):
        gc = SemiSpaceCollector(space_size=100000, auto_collect=False)
        roots = []
        for i in range(50):
            obj = gc.allocate(f"keep_{i}")
            gc.add_root(obj)
            roots.append(obj)
        for i in range(50):
            gc.allocate(f"discard_{i}")
        freed = gc.collect()
        assert freed == 50
        assert gc.live_count == 50

    def test_deep_chain_survives(self):
        gc = SemiSpaceCollector(space_size=100000, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        prev = root
        for i in range(50):
            child = gc.allocate(f"link_{i}")
            gc.set_references(prev, [child])
            prev = child
        freed = gc.collect()
        assert freed == 0
        assert gc.live_count == 51

    def test_repeated_gc_stability(self):
        gc = SemiSpaceCollector(space_size=100000, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        for cycle in range(30):
            for i in range(10):
                gc.allocate(f"temp_{cycle}_{i}")
            gc.collect()
        assert gc.live_count == 1
        assert gc.stats.total_collections == 30
        assert gc.get_value(root) == "root"

    def test_generational_stress(self):
        gc = SemiSpaceCollector(space_size=50000, auto_collect=False)
        gc.enable_generational(tenured_size=100000, tenure_threshold=2)
        root = gc.allocate("root")
        gc.add_root(root)
        for cycle in range(10):
            for i in range(5):
                gc.allocate(f"temp_{cycle}_{i}")
            gc.collect()
        # Root should be promoted to tenured by now
        assert gc.live_count >= 1

    def test_pinned_and_normal_mixed(self):
        gc = SemiSpaceCollector(space_size=100000, auto_collect=False)
        pinned = gc.allocate("pinned", pinned=True)
        normal = gc.allocate("normal")
        gc.add_root(normal)
        gc.allocate("garbage")
        gc.collect()
        assert gc.live_count == 2
        assert pinned._alive
        assert gc.get_value(normal) == "normal"

    def test_los_and_normal_mixed(self):
        gc = SemiSpaceCollector(space_size=4096, los_threshold=100, auto_collect=False)
        root = gc.allocate("root")
        gc.add_root(root)
        big = gc.allocate("x" * 200)
        gc.set_references(root, [big])
        gc.allocate("garbage")
        gc.allocate("y" * 200)  # unreferenced large object
        gc.collect()
        assert gc.live_count == 2
        assert gc.los.live_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
