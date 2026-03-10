"""
Tests for C075: Weak References + Ephemerons
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from weak_refs_ephemerons import (
    GCObject, WeakRef, WeakValueDict, WeakKeyDict,
    Ephemeron, EphemeronTable, MarkSweepGC, WeakSet,
    FinalizerState, FinalizerEntry,
)


# ===================================================================
# GCObject basics
# ===================================================================

class TestGCObject:
    def test_create(self):
        obj = GCObject("hello")
        assert obj.value == "hello"
        assert obj.is_alive()
        assert obj.marked is False
        assert obj.generation == 0

    def test_unique_ids(self):
        a = GCObject("a")
        b = GCObject("b")
        assert a.obj_id != b.obj_id

    def test_set_references(self):
        a = GCObject("a")
        b = GCObject("b")
        c = GCObject("c")
        a.set_references(b, c)
        assert len(a.refs) == 2
        assert b in a.refs
        assert c in a.refs

    def test_add_remove_reference(self):
        a = GCObject("a")
        b = GCObject("b")
        a.add_reference(b)
        assert b in a.refs
        a.remove_reference(b)
        assert b not in a.refs

    def test_repr(self):
        obj = GCObject("test")
        r = repr(obj)
        assert "test" in r
        assert "alive" in r

    def test_hash_and_eq(self):
        a = GCObject("a")
        b = GCObject("b")
        s = {a, b}
        assert len(s) == 2
        assert a in s
        assert a == a
        assert a != b

    def test_none_value(self):
        obj = GCObject(None)
        assert obj.value is None
        assert obj.is_alive()

    def test_finalizer_attribute(self):
        called = []
        obj = GCObject("x", finalizer=lambda o: called.append(o.value))
        assert obj.finalizer is not None


# ===================================================================
# WeakRef basics
# ===================================================================

class TestWeakRef:
    def test_create_and_get(self):
        obj = GCObject("target")
        wr = WeakRef(obj)
        assert wr.alive
        assert wr.get() is obj

    def test_invalidate(self):
        obj = GCObject("target")
        wr = WeakRef(obj)
        wr._invalidate()
        assert not wr.alive
        assert wr.get() is None

    def test_callback_on_invalidate(self):
        obj = GCObject("target")
        called = []
        wr = WeakRef(obj, callback=lambda ref: called.append(ref.value))
        wr._invalidate()
        assert len(called) == 1
        assert called[0] == "target"

    def test_double_invalidate_safe(self):
        obj = GCObject("x")
        called = []
        wr = WeakRef(obj, callback=lambda ref: called.append(1))
        wr._invalidate()
        wr._invalidate()
        assert len(called) == 1

    def test_callback_exception_swallowed(self):
        obj = GCObject("x")
        wr = WeakRef(obj, callback=lambda ref: 1/0)
        wr._invalidate()  # should not raise
        assert not wr.alive

    def test_type_error_on_non_gcobject(self):
        with pytest.raises(TypeError):
            WeakRef("not a GCObject")

    def test_error_on_dead_object(self):
        obj = GCObject("x")
        obj._alive = False
        with pytest.raises(ValueError):
            WeakRef(obj)

    def test_multiple_weak_refs(self):
        obj = GCObject("shared")
        wr1 = WeakRef(obj)
        wr2 = WeakRef(obj)
        assert len(obj._weak_refs) == 2
        wr1._invalidate()
        assert not wr1.alive
        assert wr2.alive  # independent

    def test_repr(self):
        obj = GCObject("val")
        wr = WeakRef(obj)
        assert "val" in repr(wr)
        wr._invalidate()
        assert "dead" in repr(wr)

    def test_hash_and_eq(self):
        obj = GCObject("x")
        wr1 = WeakRef(obj)
        wr2 = WeakRef(obj)
        assert wr1 != wr2  # different WeakRef instances
        assert wr1 == wr1
        s = {wr1, wr2}
        assert len(s) == 2


# ===================================================================
# WeakValueDict
# ===================================================================

class TestWeakValueDict:
    def test_set_get(self):
        d = WeakValueDict()
        obj = GCObject("val")
        d["key"] = obj
        assert d["key"] is obj

    def test_missing_key(self):
        d = WeakValueDict()
        with pytest.raises(KeyError):
            d["missing"]

    def test_get_default(self):
        d = WeakValueDict()
        assert d.get("missing") is None
        assert d.get("missing", 42) == 42

    def test_contains(self):
        d = WeakValueDict()
        obj = GCObject("v")
        d["k"] = obj
        assert "k" in d
        assert "nope" not in d

    def test_auto_cleanup(self):
        d = WeakValueDict()
        obj = GCObject("v")
        d["k"] = obj
        # Simulate collection
        wr = d._data["k"]
        wr._invalidate()
        assert "k" not in d
        assert len(d) == 0

    def test_len(self):
        d = WeakValueDict()
        d["a"] = GCObject("1")
        d["b"] = GCObject("2")
        assert len(d) == 2

    def test_del(self):
        d = WeakValueDict()
        d["k"] = GCObject("v")
        del d["k"]
        assert "k" not in d

    def test_type_error(self):
        d = WeakValueDict()
        with pytest.raises(TypeError):
            d["k"] = "not a GCObject"

    def test_keys_values_items(self):
        d = WeakValueDict()
        a = GCObject("a")
        b = GCObject("b")
        d["x"] = a
        d["y"] = b
        assert set(d.keys()) == {"x", "y"}
        assert len(d.values()) == 2
        assert len(d.items()) == 2

    def test_overwrite(self):
        d = WeakValueDict()
        a = GCObject("a")
        b = GCObject("b")
        d["k"] = a
        d["k"] = b
        assert d["k"] is b

    def test_repr(self):
        d = WeakValueDict()
        r = repr(d)
        assert "WeakValueDict" in r


# ===================================================================
# WeakKeyDict
# ===================================================================

class TestWeakKeyDict:
    def test_set_get(self):
        d = WeakKeyDict()
        key = GCObject("key")
        d[key] = "value"
        assert d[key] == "value"

    def test_missing_key(self):
        d = WeakKeyDict()
        key = GCObject("key")
        with pytest.raises(KeyError):
            d[key]

    def test_get_default(self):
        d = WeakKeyDict()
        key = GCObject("k")
        assert d.get(key) is None
        assert d.get(key, 99) == 99

    def test_contains(self):
        d = WeakKeyDict()
        key = GCObject("k")
        d[key] = "v"
        assert key in d

    def test_auto_cleanup(self):
        d = WeakKeyDict()
        key = GCObject("k")
        d[key] = "v"
        # Simulate collection
        entry = d._data[key.obj_id]
        entry[0]._invalidate()
        assert key not in d
        assert len(d) == 0

    def test_len(self):
        d = WeakKeyDict()
        a = GCObject("a")
        b = GCObject("b")
        d[a] = 1
        d[b] = 2
        assert len(d) == 2

    def test_del(self):
        d = WeakKeyDict()
        key = GCObject("k")
        d[key] = "v"
        del d[key]
        assert key not in d

    def test_type_error(self):
        d = WeakKeyDict()
        with pytest.raises(TypeError):
            d["not a GCObject"] = "v"

    def test_keys_values_items(self):
        d = WeakKeyDict()
        a = GCObject("a")
        b = GCObject("b")
        d[a] = 1
        d[b] = 2
        assert len(d.keys()) == 2
        assert set(d.values()) == {1, 2}
        assert len(d.items()) == 2


# ===================================================================
# Ephemeron basics
# ===================================================================

class TestEphemeron:
    def test_create(self):
        key = GCObject("key")
        eph = Ephemeron(key, "value")
        assert eph.key is key
        assert eph.value == "value"
        assert eph.alive

    def test_invalidate(self):
        key = GCObject("key")
        eph = Ephemeron(key, "value")
        eph._invalidate()
        assert not eph.alive
        assert eph.key is None
        assert eph.value is None

    def test_callback(self):
        key = GCObject("key")
        results = []
        eph = Ephemeron(key, "value", callback=lambda k, v: results.append((k.value, v)))
        eph._invalidate()
        assert results == [("key", "value")]

    def test_double_invalidate(self):
        key = GCObject("key")
        results = []
        eph = Ephemeron(key, "value", callback=lambda k, v: results.append(1))
        eph._invalidate()
        eph._invalidate()
        assert len(results) == 1

    def test_type_error(self):
        with pytest.raises(TypeError):
            Ephemeron("not a GCObject", "value")

    def test_attached_to_key(self):
        key = GCObject("key")
        eph = Ephemeron(key, "value")
        assert eph in key._attached_ephemerons

    def test_value_is_gcobject(self):
        key = GCObject("key")
        val = GCObject("val")
        eph = Ephemeron(key, val)
        assert eph.value is val

    def test_repr(self):
        key = GCObject("k")
        eph = Ephemeron(key, "v")
        assert "k" in repr(eph)
        eph._invalidate()
        assert "dead" in repr(eph)

    def test_hash_and_eq(self):
        k1 = GCObject("k1")
        k2 = GCObject("k2")
        e1 = Ephemeron(k1, "v1")
        e2 = Ephemeron(k2, "v2")
        assert e1 != e2
        assert e1 == e1
        s = {e1, e2}
        assert len(s) == 2


# ===================================================================
# EphemeronTable
# ===================================================================

class TestEphemeronTable:
    def test_set_get(self):
        t = EphemeronTable()
        key = GCObject("key")
        t.set(key, "value")
        assert t.get(key) == "value"

    def test_get_default(self):
        t = EphemeronTable()
        key = GCObject("key")
        assert t.get(key) is None
        assert t.get(key, 42) == 42

    def test_contains(self):
        t = EphemeronTable()
        key = GCObject("key")
        t.set(key, "value")
        assert key in t
        other = GCObject("other")
        assert other not in t

    def test_remove(self):
        t = EphemeronTable()
        key = GCObject("key")
        t.set(key, "value")
        t.remove(key)
        assert key not in t

    def test_len(self):
        t = EphemeronTable()
        a = GCObject("a")
        b = GCObject("b")
        t.set(a, 1)
        t.set(b, 2)
        assert len(t) == 2

    def test_overwrite(self):
        t = EphemeronTable()
        key = GCObject("key")
        t.set(key, "old")
        t.set(key, "new")
        assert t.get(key) == "new"
        assert len(t) == 1

    def test_keys_values_items(self):
        t = EphemeronTable()
        a = GCObject("a")
        b = GCObject("b")
        t.set(a, 1)
        t.set(b, 2)
        assert len(t.keys()) == 2
        assert set(t.values()) == {1, 2}
        assert len(t.items()) == 2

    def test_all_ephemerons(self):
        t = EphemeronTable()
        a = GCObject("a")
        t.set(a, 1)
        ephs = t.all_ephemerons()
        assert len(ephs) == 1
        assert ephs[0].key is a

    def test_dead_entry_cleaned(self):
        t = EphemeronTable()
        key = GCObject("key")
        t.set(key, "value")
        # Simulate key death
        eph = t._entries[key.obj_id]
        eph._invalidate()
        assert key not in t
        assert len(t) == 0

    def test_callback(self):
        results = []
        t = EphemeronTable(callback=lambda k, v: results.append((k.value, v)))
        key = GCObject("key")
        t.set(key, "value")
        eph = t._entries[key.obj_id]
        eph._invalidate()
        assert len(results) == 1

    def test_repr(self):
        t = EphemeronTable()
        r = repr(t)
        assert "EphemeronTable" in r


# ===================================================================
# WeakSet
# ===================================================================

class TestWeakSet:
    def test_add_contains(self):
        ws = WeakSet()
        obj = GCObject("x")
        ws.add(obj)
        assert obj in ws

    def test_discard(self):
        ws = WeakSet()
        obj = GCObject("x")
        ws.add(obj)
        ws.discard(obj)
        assert obj not in ws

    def test_len(self):
        ws = WeakSet()
        a = GCObject("a")
        b = GCObject("b")
        ws.add(a)
        ws.add(b)
        assert len(ws) == 2

    def test_auto_cleanup(self):
        ws = WeakSet()
        obj = GCObject("x")
        ws.add(obj)
        # Simulate collection
        wr = ws._refs[obj.obj_id]
        wr._invalidate()
        assert obj not in ws
        assert len(ws) == 0

    def test_iter(self):
        ws = WeakSet()
        a = GCObject("a")
        b = GCObject("b")
        ws.add(a)
        ws.add(b)
        items = list(ws)
        assert len(items) == 2

    def test_duplicate_add(self):
        ws = WeakSet()
        obj = GCObject("x")
        ws.add(obj)
        ws.add(obj)
        assert len(ws) == 1


# ===================================================================
# MarkSweepGC -- basic collection
# ===================================================================

class TestGCBasic:
    def test_alloc(self):
        gc = MarkSweepGC()
        obj = gc.alloc("hello")
        assert obj.is_alive()
        assert obj in gc.heap
        assert gc.total_allocations == 1

    def test_collect_unreachable(self):
        gc = MarkSweepGC()
        obj = gc.alloc("unreachable")
        freed = gc.collect()
        assert freed == 1
        assert not obj.is_alive()
        assert obj not in gc.heap

    def test_collect_reachable(self):
        gc = MarkSweepGC()
        obj = gc.alloc("root")
        gc.add_root(obj)
        freed = gc.collect()
        assert freed == 0
        assert obj.is_alive()

    def test_transitive_reachability(self):
        gc = MarkSweepGC()
        a = gc.alloc("a")
        b = gc.alloc("b")
        c = gc.alloc("c")
        a.set_references(b)
        b.set_references(c)
        gc.add_root(a)
        freed = gc.collect()
        assert freed == 0
        assert a.is_alive()
        assert b.is_alive()
        assert c.is_alive()

    def test_partial_collection(self):
        gc = MarkSweepGC()
        a = gc.alloc("root")
        b = gc.alloc("reachable")
        c = gc.alloc("unreachable")
        a.set_references(b)
        gc.add_root(a)
        freed = gc.collect()
        assert freed == 1
        assert not c.is_alive()
        assert a.is_alive()
        assert b.is_alive()

    def test_remove_root(self):
        gc = MarkSweepGC()
        obj = gc.alloc("root")
        gc.add_root(obj)
        gc.collect()
        assert obj.is_alive()
        gc.remove_root(obj)
        gc.collect()
        assert not obj.is_alive()

    def test_cycle_collection(self):
        gc = MarkSweepGC()
        a = gc.alloc("a")
        b = gc.alloc("b")
        a.set_references(b)
        b.set_references(a)
        # No root -> both should be collected
        freed = gc.collect()
        assert freed == 2
        assert not a.is_alive()
        assert not b.is_alive()

    def test_rooted_cycle(self):
        gc = MarkSweepGC()
        a = gc.alloc("a")
        b = gc.alloc("b")
        a.set_references(b)
        b.set_references(a)
        gc.add_root(a)
        freed = gc.collect()
        assert freed == 0
        assert a.is_alive()
        assert b.is_alive()

    def test_stats(self):
        gc = MarkSweepGC()
        gc.alloc("a")
        gc.alloc("b")
        gc.collect()
        s = gc.stats()
        assert s['total_allocations'] == 2
        assert s['total_collections'] == 1
        assert s['total_freed'] == 2
        assert s['peak_heap_size'] == 2

    def test_multiple_collections(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        for i in range(10):
            gc.alloc(f"temp_{i}")
        freed = gc.collect()
        assert freed == 10
        assert len(gc.heap) == 1

    def test_generation_increment(self):
        gc = MarkSweepGC()
        obj = gc.alloc("survivor")
        gc.add_root(obj)
        gc.collect()
        assert obj.generation >= 1
        gc.collect()
        assert obj.generation >= 2


# ===================================================================
# MarkSweepGC -- weak references
# ===================================================================

class TestGCWeakRefs:
    def test_weak_ref_doesnt_prevent_collection(self):
        gc = MarkSweepGC()
        obj = gc.alloc("target")
        wr = gc.create_weak_ref(obj)
        assert wr.alive
        gc.collect()
        assert not wr.alive
        assert wr.get() is None

    def test_weak_ref_survives_with_root(self):
        gc = MarkSweepGC()
        obj = gc.alloc("target")
        gc.add_root(obj)
        wr = gc.create_weak_ref(obj)
        gc.collect()
        assert wr.alive
        assert wr.get() is obj

    def test_weak_ref_callback_on_collection(self):
        gc = MarkSweepGC()
        obj = gc.alloc("target")
        results = []
        wr = gc.create_weak_ref(obj, callback=lambda ref: results.append("collected"))
        gc.collect()
        assert results == ["collected"]

    def test_multiple_weak_refs_same_object(self):
        gc = MarkSweepGC()
        obj = gc.alloc("target")
        wr1 = gc.create_weak_ref(obj)
        wr2 = gc.create_weak_ref(obj)
        gc.collect()
        assert not wr1.alive
        assert not wr2.alive

    def test_weak_ref_stats(self):
        gc = MarkSweepGC()
        obj = gc.alloc("target")
        gc.create_weak_ref(obj)
        gc.create_weak_ref(obj)
        gc.collect()
        assert gc.total_weak_refs_cleared == 2


# ===================================================================
# MarkSweepGC -- ephemerons (the hard part)
# ===================================================================

class TestGCEphemerons:
    def test_ephemeron_key_reachable_keeps_value(self):
        """When key is rooted, value should survive."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("value")
        gc.add_root(key)
        eph = gc.create_ephemeron(key, val)
        gc.collect()
        assert key.is_alive()
        assert val.is_alive()
        assert eph.alive

    def test_ephemeron_key_unreachable_clears(self):
        """When key is unreachable, ephemeron and value should be cleared."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("value")
        eph = gc.create_ephemeron(key, val)
        gc.collect()
        assert not key.is_alive()
        assert not val.is_alive()
        assert not eph.alive

    def test_ephemeron_value_not_root(self):
        """Value alone does not keep key alive."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("value")
        gc.add_root(val)  # root value, not key
        eph = gc.create_ephemeron(key, val)
        gc.collect()
        # Key is unreachable -> ephemeron dies, but value is separately rooted
        assert not key.is_alive()
        assert val.is_alive()
        assert not eph.alive

    def test_ephemeron_chain(self):
        """Chain: root -> key1, eph(key1, key2), eph(key2, val).
        key1 reachable -> key2 reachable -> val reachable."""
        gc = MarkSweepGC()
        key1 = gc.alloc("key1")
        key2 = gc.alloc("key2")
        val = gc.alloc("val")
        gc.add_root(key1)
        gc.create_ephemeron(key1, key2)
        gc.create_ephemeron(key2, val)
        gc.collect()
        assert key1.is_alive()
        assert key2.is_alive()
        assert val.is_alive()

    def test_ephemeron_chain_broken(self):
        """If root is removed, entire chain dies."""
        gc = MarkSweepGC()
        key1 = gc.alloc("key1")
        key2 = gc.alloc("key2")
        val = gc.alloc("val")
        gc.add_root(key1)
        gc.create_ephemeron(key1, key2)
        gc.create_ephemeron(key2, val)
        gc.collect()
        assert key1.is_alive()
        gc.remove_root(key1)
        gc.collect()
        assert not key1.is_alive()
        assert not key2.is_alive()
        assert not val.is_alive()

    def test_ephemeron_fixpoint_iteration(self):
        """Requires multiple fixpoint iterations:
        root -> A, eph(A, B), eph(B, C), eph(C, D)
        A marked -> B marked -> C marked -> D marked (3 iterations)."""
        gc = MarkSweepGC()
        objs = [gc.alloc(f"obj{i}") for i in range(5)]
        gc.add_root(objs[0])
        for i in range(4):
            gc.create_ephemeron(objs[i], objs[i + 1])
        gc.collect()
        for obj in objs:
            assert obj.is_alive(), f"{obj} should be alive"

    def test_ephemeron_diamond(self):
        """Diamond pattern: root -> A, eph(A, B), eph(A, C), eph(B, D), eph(C, D).
        All should survive."""
        gc = MarkSweepGC()
        a = gc.alloc("A")
        b = gc.alloc("B")
        c = gc.alloc("C")
        d = gc.alloc("D")
        gc.add_root(a)
        gc.create_ephemeron(a, b)
        gc.create_ephemeron(a, c)
        gc.create_ephemeron(b, d)
        gc.create_ephemeron(c, d)
        gc.collect()
        assert all(obj.is_alive() for obj in [a, b, c, d])

    def test_ephemeron_independent_dead(self):
        """Two independent ephemerons, both keys unreachable."""
        gc = MarkSweepGC()
        k1 = gc.alloc("k1")
        v1 = gc.alloc("v1")
        k2 = gc.alloc("k2")
        v2 = gc.alloc("v2")
        gc.create_ephemeron(k1, v1)
        gc.create_ephemeron(k2, v2)
        gc.collect()
        assert not any(o.is_alive() for o in [k1, v1, k2, v2])

    def test_ephemeron_callback(self):
        gc = MarkSweepGC()
        key = gc.alloc("key")
        results = []
        gc.create_ephemeron(key, "val", callback=lambda k, v: results.append((k.value, v)))
        gc.collect()
        assert len(results) == 1
        assert results[0] == ("key", "val")

    def test_ephemeron_stats(self):
        gc = MarkSweepGC()
        k1 = gc.alloc("k1")
        k2 = gc.alloc("k2")
        gc.create_ephemeron(k1, "v1")
        gc.create_ephemeron(k2, "v2")
        gc.collect()
        assert gc.total_ephemerons_cleared == 2

    def test_ephemeron_with_gc_value(self):
        """Value is a GCObject -- should be collected when key dies."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("value_obj")
        gc.create_ephemeron(key, val)
        gc.collect()
        assert not val.is_alive()

    def test_ephemeron_value_reachable_through_other_path(self):
        """Value is reachable through root, but key is not.
        Value survives, ephemeron dies."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("value")
        root = gc.alloc("root")
        root.set_references(val)
        gc.add_root(root)
        eph = gc.create_ephemeron(key, val)
        gc.collect()
        assert not key.is_alive()
        assert val.is_alive()  # kept alive by root
        assert not eph.alive

    def test_ephemeron_self_referencing_key(self):
        """Key references itself through a cycle."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        key.set_references(key)
        val = gc.alloc("val")
        gc.create_ephemeron(key, val)
        # Key is in a cycle but not rooted -> dead
        gc.collect()
        assert not key.is_alive()
        assert not val.is_alive()


# ===================================================================
# MarkSweepGC -- ephemeron tables with GC
# ===================================================================

class TestGCEphemeronTable:
    def test_table_entries_traced(self):
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        key = gc.alloc("key")
        val = gc.alloc("val")
        gc.add_root(key)
        table.set(key, val)
        gc.collect()
        assert key.is_alive()
        assert val.is_alive()

    def test_table_entries_cleared(self):
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        key = gc.alloc("key")
        val = gc.alloc("val")
        table.set(key, val)
        gc.collect()
        assert not key.is_alive()
        assert not val.is_alive()
        assert len(table) == 0

    def test_table_mixed(self):
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        alive_key = gc.alloc("alive_key")
        alive_val = gc.alloc("alive_val")
        dead_key = gc.alloc("dead_key")
        dead_val = gc.alloc("dead_val")
        gc.add_root(alive_key)
        table.set(alive_key, alive_val)
        table.set(dead_key, dead_val)
        gc.collect()
        assert alive_key.is_alive()
        assert alive_val.is_alive()
        assert not dead_key.is_alive()
        assert not dead_val.is_alive()
        assert len(table) == 1

    def test_table_chain_through_table(self):
        """Ephemeron chain through table entries."""
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        a = gc.alloc("a")
        b = gc.alloc("b")
        c = gc.alloc("c")
        gc.add_root(a)
        table.set(a, b)
        table.set(b, c)
        gc.collect()
        assert a.is_alive()
        assert b.is_alive()
        assert c.is_alive()

    def test_standalone_and_table_ephemerons_interact(self):
        """Mix of standalone ephemerons and table entries."""
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        root = gc.alloc("root")
        a = gc.alloc("a")
        b = gc.alloc("b")
        gc.add_root(root)
        # Standalone: root -> a
        gc.create_ephemeron(root, a)
        # Table: a -> b
        table.set(a, b)
        gc.collect()
        assert all(obj.is_alive() for obj in [root, a, b])


# ===================================================================
# MarkSweepGC -- finalizers
# ===================================================================

class TestGCFinalizers:
    def test_finalizer_called(self):
        gc = MarkSweepGC()
        results = []
        gc.alloc("obj", finalizer=lambda o: results.append(o.value))
        gc.collect()
        assert results == ["obj"]

    def test_finalizer_order(self):
        """B references C -> C's finalizer runs before B's."""
        gc = MarkSweepGC()
        order = []
        c = gc.alloc("C", finalizer=lambda o: order.append("C"))
        b = gc.alloc("B", finalizer=lambda o: order.append("B"))
        b.set_references(c)
        gc.collect()
        assert order.index("C") < order.index("B")

    def test_finalizer_exception_swallowed(self):
        gc = MarkSweepGC()
        gc.alloc("boom", finalizer=lambda o: 1/0)
        gc.collect()  # should not raise

    def test_finalizer_stats(self):
        gc = MarkSweepGC()
        gc.alloc("a", finalizer=lambda o: None)
        gc.alloc("b", finalizer=lambda o: None)
        gc.collect()
        assert gc.total_finalizers_run == 2

    def test_resurrection(self):
        """Finalizer stores reference to dead object in a root -> resurrection."""
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        target = gc.alloc("target", finalizer=lambda o: root.add_reference(o))
        gc.collect()
        # target was resurrected by the finalizer
        assert target.is_alive()
        assert target in root.refs

    def test_resurrection_increments_generation(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        target = gc.alloc("target", finalizer=lambda o: root.add_reference(o))
        gc.collect()
        assert target.generation > 0  # promoted for surviving via resurrection


# ===================================================================
# MarkSweepGC -- generational
# ===================================================================

class TestGCGenerational:
    def test_minor_collect(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        young = gc.alloc("young")
        freed = gc.minor_collect()
        assert freed == 1
        assert not young.is_alive()
        assert root.is_alive()

    def test_minor_skips_old_objects(self):
        gc = MarkSweepGC()
        old = gc.alloc("old")
        old.generation = 10  # artificially aged
        gc.minor_collect(gen_threshold=3)
        # Old objects with generation >= threshold are not swept in minor collect
        # But they're also not rooted, so they ARE swept (minor_collect sweeps young only)
        assert old.is_alive()  # generation >= threshold -> not swept

    def test_minor_collect_with_ephemerons(self):
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("val")
        gc.add_root(key)
        gc.create_ephemeron(key, val)
        freed = gc.minor_collect()
        assert freed == 0
        assert val.is_alive()


# ===================================================================
# Complex scenarios
# ===================================================================

class TestComplexScenarios:
    def test_weak_ref_and_ephemeron_same_object(self):
        """Object is both weak ref target and ephemeron key."""
        gc = MarkSweepGC()
        obj = gc.alloc("shared")
        val = gc.alloc("val")
        wr = gc.create_weak_ref(obj)
        eph = gc.create_ephemeron(obj, val)
        gc.collect()
        assert not wr.alive
        assert not eph.alive
        assert not obj.is_alive()
        assert not val.is_alive()

    def test_weak_ref_and_ephemeron_rooted(self):
        gc = MarkSweepGC()
        obj = gc.alloc("shared")
        val = gc.alloc("val")
        gc.add_root(obj)
        wr = gc.create_weak_ref(obj)
        eph = gc.create_ephemeron(obj, val)
        gc.collect()
        assert wr.alive
        assert eph.alive
        assert obj.is_alive()
        assert val.is_alive()

    def test_large_ephemeron_chain(self):
        """Long chain of ephemerons (10 deep)."""
        gc = MarkSweepGC()
        objs = [gc.alloc(f"node_{i}") for i in range(10)]
        gc.add_root(objs[0])
        for i in range(9):
            gc.create_ephemeron(objs[i], objs[i + 1])
        gc.collect()
        for obj in objs:
            assert obj.is_alive()

    def test_large_ephemeron_chain_broken(self):
        gc = MarkSweepGC()
        objs = [gc.alloc(f"node_{i}") for i in range(10)]
        gc.add_root(objs[0])
        for i in range(9):
            gc.create_ephemeron(objs[i], objs[i + 1])
        gc.collect()
        gc.remove_root(objs[0])
        gc.collect()
        for obj in objs:
            assert not obj.is_alive()

    def test_weak_value_dict_with_gc(self):
        """WeakValueDict entries cleaned when GC collects values."""
        gc = MarkSweepGC()
        d = WeakValueDict()
        alive = gc.alloc("alive")
        dead = gc.alloc("dead")
        gc.add_root(alive)
        d["a"] = alive
        d["d"] = dead
        gc.collect()
        assert "a" in d
        assert d["a"] is alive
        assert "d" not in d

    def test_weak_key_dict_with_gc(self):
        gc = MarkSweepGC()
        d = WeakKeyDict()
        alive = gc.alloc("alive")
        dead = gc.alloc("dead")
        gc.add_root(alive)
        d[alive] = "kept"
        d[dead] = "cleared"
        gc.collect()
        assert d[alive] == "kept"
        assert dead not in d

    def test_ephemeron_table_with_gc(self):
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        keys = [gc.alloc(f"k{i}") for i in range(5)]
        vals = [gc.alloc(f"v{i}") for i in range(5)]
        # Root first 3 keys
        for i in range(3):
            gc.add_root(keys[i])
            table.set(keys[i], vals[i])
        # Last 2 unrooted
        table.set(keys[3], vals[3])
        table.set(keys[4], vals[4])
        gc.collect()
        assert len(table) == 3
        for i in range(3):
            assert keys[i].is_alive()
            assert vals[i].is_alive()
        assert not keys[3].is_alive()
        assert not keys[4].is_alive()

    def test_cross_references_ephemerons(self):
        """Ephemeron values reference each other's keys."""
        gc = MarkSweepGC()
        a = gc.alloc("a")
        b = gc.alloc("b")
        gc.add_root(a)
        # eph1: key=a, value=b
        # eph2: key=b, value points to nothing extra
        gc.create_ephemeron(a, b)
        gc.create_ephemeron(b, gc.alloc("leaf"))
        gc.collect()
        assert a.is_alive()
        assert b.is_alive()

    def test_multiple_collections_stability(self):
        """Multiple collections don't corrupt state."""
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        for _ in range(5):
            gc.alloc("temp")
            gc.create_ephemeron(root, gc.alloc("eph_val"))
            gc.collect()
        assert root.is_alive()
        assert len(gc.heap) >= 1  # at least root

    def test_weak_set_with_gc(self):
        gc = MarkSweepGC()
        ws = WeakSet()
        alive = gc.alloc("alive")
        dead = gc.alloc("dead")
        gc.add_root(alive)
        ws.add(alive)
        ws.add(dead)
        gc.collect()
        assert alive in ws
        assert dead not in ws
        assert len(ws) == 1

    def test_ephemeron_key_is_value_of_another(self):
        """key2 is the value of eph1. If key1 is rooted, key2 becomes reachable
        through eph1's value, which makes eph2 keep val alive."""
        gc = MarkSweepGC()
        key1 = gc.alloc("key1")
        key2 = gc.alloc("key2")
        val = gc.alloc("val")
        gc.add_root(key1)
        gc.create_ephemeron(key1, key2)  # key1 rooted -> key2 reachable
        gc.create_ephemeron(key2, val)    # key2 reachable -> val reachable
        gc.collect()
        assert all(obj.is_alive() for obj in [key1, key2, val])

    def test_auto_collect_threshold(self):
        gc = MarkSweepGC(threshold=5)
        root = gc.alloc("root")
        gc.add_root(root)
        for i in range(3):
            gc.alloc(f"temp_{i}")
        assert not gc.should_collect()  # 4 allocs < 5
        gc.alloc("trigger")
        assert gc.should_collect()  # 5 allocs >= 5

    def test_dead_object_references_cleared(self):
        """After collection, dead objects' references shouldn't prevent future collections."""
        gc = MarkSweepGC()
        a = gc.alloc("a")
        b = gc.alloc("b")
        a.set_references(b)
        gc.collect()
        assert not a.is_alive()
        assert not b.is_alive()

    def test_finalizer_with_ephemeron(self):
        """Finalizer on ephemeron key."""
        gc = MarkSweepGC()
        results = []
        key = gc.alloc("key", finalizer=lambda o: results.append("finalized"))
        val = gc.alloc("val")
        gc.create_ephemeron(key, val)
        gc.collect()
        assert results == ["finalized"]
        assert not key.is_alive()

    def test_ephemeron_value_is_non_gcobject(self):
        """Ephemeron value can be a plain Python value."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        gc.add_root(key)
        eph = gc.create_ephemeron(key, {"data": [1, 2, 3]})
        gc.collect()
        assert eph.alive
        assert eph.value == {"data": [1, 2, 3]}

    def test_many_ephemerons_same_key(self):
        gc = MarkSweepGC()
        key = gc.alloc("key")
        gc.add_root(key)
        vals = [gc.alloc(f"v{i}") for i in range(10)]
        ephs = [gc.create_ephemeron(key, v) for v in vals]
        gc.collect()
        for v in vals:
            assert v.is_alive()
        for e in ephs:
            assert e.alive

    def test_many_ephemerons_same_key_unrooted(self):
        gc = MarkSweepGC()
        key = gc.alloc("key")
        vals = [gc.alloc(f"v{i}") for i in range(10)]
        ephs = [gc.create_ephemeron(key, v) for v in vals]
        gc.collect()
        for v in vals:
            assert not v.is_alive()
        for e in ephs:
            assert not e.alive


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_heap_collect(self):
        gc = MarkSweepGC()
        freed = gc.collect()
        assert freed == 0

    def test_only_roots_collect(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        freed = gc.collect()
        assert freed == 0

    def test_ephemeron_key_dead_before_collect(self):
        """Key was already dead when ephemeron was created -- edge case."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("val")
        key._alive = False
        gc.heap.discard(key)
        # Ephemeron with dead key
        eph = gc.create_ephemeron(key, val)
        gc.collect()
        # Key is dead -> ephemeron clears
        assert not eph.alive

    def test_weak_ref_to_rooted_then_unrooted(self):
        gc = MarkSweepGC()
        obj = gc.alloc("obj")
        gc.add_root(obj)
        wr = gc.create_weak_ref(obj)
        gc.collect()
        assert wr.alive
        gc.remove_root(obj)
        gc.collect()
        assert not wr.alive

    def test_ephemeron_with_no_gc_tracking(self):
        """Ephemeron created outside GC -- should still work internally."""
        key = GCObject("key")
        val = GCObject("val")
        eph = Ephemeron(key, val)
        assert eph.alive
        eph._invalidate()
        assert not eph.alive

    def test_gc_disabled(self):
        gc = MarkSweepGC()
        gc.enabled = False
        gc.alloc("x")
        gc.alloc("y")
        assert not gc.should_collect()

    def test_collect_after_all_roots_removed(self):
        gc = MarkSweepGC()
        objs = [gc.alloc(f"obj{i}") for i in range(5)]
        for obj in objs:
            gc.add_root(obj)
        gc.collect()
        assert all(obj.is_alive() for obj in objs)
        for obj in objs:
            gc.remove_root(obj)
        gc.collect()
        assert all(not obj.is_alive() for obj in objs)

    def test_ephemeron_value_references_key(self):
        """Value references key -- still ephemeron semantics apply.
        Key must be reachable through non-ephemeron path."""
        gc = MarkSweepGC()
        key = gc.alloc("key")
        val = gc.alloc("val")
        val.set_references(key)  # val -> key
        gc.create_ephemeron(key, val)
        # Neither is rooted -> both die (val references key, but key
        # is not independently reachable)
        gc.collect()
        assert not key.is_alive()
        assert not val.is_alive()

    def test_ephemeron_tree(self):
        """Binary tree of ephemerons from a single root.
        root -> eph(root, a), eph(root, b), eph(a, c), eph(b, d)."""
        gc = MarkSweepGC()
        root = gc.alloc("root")
        a = gc.alloc("a")
        b = gc.alloc("b")
        c = gc.alloc("c")
        d = gc.alloc("d")
        gc.add_root(root)
        gc.create_ephemeron(root, a)
        gc.create_ephemeron(root, b)
        gc.create_ephemeron(a, c)
        gc.create_ephemeron(b, d)
        gc.collect()
        for obj in [root, a, b, c, d]:
            assert obj.is_alive()

    def test_weak_ref_created_during_no_gc(self):
        """WeakRef created when GC is disabled works fine."""
        gc = MarkSweepGC()
        gc.enabled = False
        obj = gc.alloc("x")
        wr = gc.create_weak_ref(obj)
        assert wr.alive
        gc.enabled = True
        gc.collect()
        assert not wr.alive


# ===================================================================
# Stress tests
# ===================================================================

class TestStress:
    def test_many_objects(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        for i in range(100):
            obj = gc.alloc(f"node_{i}")
            root.add_reference(obj)
        gc.collect()
        assert len(gc.heap) == 101

    def test_many_ephemerons(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        for i in range(50):
            gc.create_ephemeron(root, gc.alloc(f"val_{i}"))
        gc.collect()
        # root is alive -> all values alive
        assert len(gc.heap) == 51

    def test_many_weak_refs(self):
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        wrs = [gc.create_weak_ref(root) for _ in range(100)]
        gc.collect()
        assert all(wr.alive for wr in wrs)

    def test_churn(self):
        """Allocate and collect repeatedly."""
        gc = MarkSweepGC()
        root = gc.alloc("root")
        gc.add_root(root)
        for cycle in range(20):
            for i in range(10):
                gc.alloc(f"temp_{cycle}_{i}")
            gc.collect()
        assert len(gc.heap) == 1
        assert gc.total_freed == 200

    def test_ephemeron_table_churn(self):
        gc = MarkSweepGC()
        table = EphemeronTable()
        gc.track_ephemeron_table(table)
        root = gc.alloc("root")
        gc.add_root(root)
        for cycle in range(10):
            key = gc.alloc(f"key_{cycle}")
            gc.add_root(key)
            table.set(key, gc.alloc(f"val_{cycle}"))
            gc.collect()
        assert len(table) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
