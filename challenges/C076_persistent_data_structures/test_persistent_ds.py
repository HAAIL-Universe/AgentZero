"""
Tests for C076: Persistent Data Structures
"""
import pytest
from persistent_ds import (
    PersistentVector, TransientVector,
    PersistentList, NIL,
    PersistentHashMap, TransientHashMap,
    PersistentSortedSet,
)


# ============================================================
# PersistentVector Tests
# ============================================================

class TestPersistentVectorBasic:
    def test_empty(self):
        v = PersistentVector.empty()
        assert len(v) == 0
        assert not v

    def test_append_and_get(self):
        v = PersistentVector.empty()
        v1 = v.append(10)
        assert len(v1) == 1
        assert v1.get(0) == 10
        assert len(v) == 0  # original unchanged

    def test_multiple_appends(self):
        v = PersistentVector.empty()
        for i in range(100):
            v = v.append(i)
        assert len(v) == 100
        for i in range(100):
            assert v.get(i) == i

    def test_of(self):
        v = PersistentVector.of(1, 2, 3)
        assert len(v) == 3
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

    def test_from_iterable(self):
        v = PersistentVector.from_iterable(range(50))
        assert len(v) == 50
        for i in range(50):
            assert v[i] == i

    def test_negative_index(self):
        v = PersistentVector.of(10, 20, 30)
        assert v[-1] == 30
        assert v[-2] == 20
        assert v[-3] == 10

    def test_index_error(self):
        v = PersistentVector.of(1, 2, 3)
        with pytest.raises(IndexError):
            v.get(5)
        with pytest.raises(IndexError):
            v.get(-4)


class TestPersistentVectorSet:
    def test_set_basic(self):
        v = PersistentVector.of(1, 2, 3)
        v2 = v.set(1, 99)
        assert v2[1] == 99
        assert v[1] == 2  # original unchanged

    def test_set_first_and_last(self):
        v = PersistentVector.of(10, 20, 30)
        v2 = v.set(0, 100)
        v3 = v.set(2, 300)
        assert v2[0] == 100
        assert v3[2] == 300
        assert v[0] == 10
        assert v[2] == 30

    def test_set_out_of_range(self):
        v = PersistentVector.of(1)
        with pytest.raises(IndexError):
            v.set(5, 99)


class TestPersistentVectorPop:
    def test_pop_basic(self):
        v = PersistentVector.of(1, 2, 3)
        v2 = v.pop()
        assert len(v2) == 2
        assert v2[0] == 1
        assert v2[1] == 2
        assert len(v) == 3  # original unchanged

    def test_pop_to_empty(self):
        v = PersistentVector.of(1)
        v2 = v.pop()
        assert len(v2) == 0

    def test_pop_empty_error(self):
        v = PersistentVector.empty()
        with pytest.raises(IndexError):
            v.pop()

    def test_pop_many(self):
        v = PersistentVector.from_iterable(range(100))
        for i in range(100, 0, -1):
            assert len(v) == i
            v = v.pop()
        assert len(v) == 0


class TestPersistentVectorStructuralSharing:
    def test_shared_structure(self):
        v1 = PersistentVector.from_iterable(range(1000))
        v2 = v1.append(1000)
        # v1 and v2 share most of the trie
        assert len(v1) == 1000
        assert len(v2) == 1001
        # Both return correct values
        for i in range(1000):
            assert v1[i] == v2[i] == i
        assert v2[1000] == 1000

    def test_multiple_branches(self):
        base = PersistentVector.from_iterable(range(100))
        branches = [base.set(50, i * 100) for i in range(10)]
        # Each branch has its own value at index 50
        for i, b in enumerate(branches):
            assert b[50] == i * 100
        # Base unchanged
        assert base[50] == 50


class TestPersistentVectorLarge:
    def test_large_vector(self):
        v = PersistentVector.from_iterable(range(10000))
        assert len(v) == 10000
        assert v[0] == 0
        assert v[5000] == 5000
        assert v[9999] == 9999

    def test_large_with_updates(self):
        v = PersistentVector.from_iterable(range(5000))
        v = v.set(0, -1)
        v = v.set(2500, -2)
        v = v.set(4999, -3)
        assert v[0] == -1
        assert v[2500] == -2
        assert v[4999] == -3
        assert v[1] == 1


class TestPersistentVectorIterAndOps:
    def test_iter(self):
        v = PersistentVector.of(1, 2, 3, 4, 5)
        assert list(v) == [1, 2, 3, 4, 5]

    def test_eq(self):
        v1 = PersistentVector.of(1, 2, 3)
        v2 = PersistentVector.of(1, 2, 3)
        v3 = PersistentVector.of(1, 2, 4)
        assert v1 == v2
        assert v1 != v3

    def test_hash_equal(self):
        v1 = PersistentVector.of(1, 2, 3)
        v2 = PersistentVector.of(1, 2, 3)
        assert hash(v1) == hash(v2)

    def test_repr(self):
        v = PersistentVector.of(1, 2, 3)
        assert "PVec" in repr(v)
        assert "1" in repr(v)

    def test_bool(self):
        assert not PersistentVector.empty()
        assert PersistentVector.of(1)

    def test_map(self):
        v = PersistentVector.of(1, 2, 3)
        v2 = v.map(lambda x: x * 2)
        assert list(v2) == [2, 4, 6]

    def test_filter(self):
        v = PersistentVector.of(1, 2, 3, 4, 5)
        v2 = v.filter(lambda x: x % 2 == 0)
        assert list(v2) == [2, 4]

    def test_reduce(self):
        v = PersistentVector.of(1, 2, 3, 4, 5)
        assert v.reduce(lambda a, b: a + b) == 15

    def test_reduce_with_init(self):
        v = PersistentVector.of(1, 2, 3)
        assert v.reduce(lambda a, b: a + b, 10) == 16

    def test_reduce_empty_error(self):
        v = PersistentVector.empty()
        with pytest.raises(TypeError):
            v.reduce(lambda a, b: a + b)

    def test_slice(self):
        v = PersistentVector.of(0, 1, 2, 3, 4)
        s = v.slice(1, 4)
        assert list(s) == [1, 2, 3]

    def test_slice_negative(self):
        v = PersistentVector.of(0, 1, 2, 3, 4)
        s = v.slice(-3)
        assert list(s) == [2, 3, 4]

    def test_concat(self):
        v1 = PersistentVector.of(1, 2, 3)
        v2 = PersistentVector.of(4, 5, 6)
        v3 = v1.concat(v2)
        assert list(v3) == [1, 2, 3, 4, 5, 6]

    def test_index_of(self):
        v = PersistentVector.of(10, 20, 30)
        assert v.index_of(20) == 1
        assert v.index_of(99) == -1

    def test_contains(self):
        v = PersistentVector.of(10, 20, 30)
        assert v.contains(20)
        assert not v.contains(99)

    def test_to_list(self):
        v = PersistentVector.of(1, 2, 3)
        assert v.to_list() == [1, 2, 3]

    def test_getitem_slice(self):
        v = PersistentVector.of(0, 1, 2, 3, 4)
        s = v[1:4]
        assert list(s) == [1, 2, 3]


class TestTransientVector:
    def test_basic(self):
        t = TransientVector(PersistentVector.empty())
        for i in range(100):
            t.append_mut(i)
        v = t.persistent()
        assert len(v) == 100
        for i in range(100):
            assert v[i] == i

    def test_set_mut(self):
        v = PersistentVector.from_iterable(range(50))
        t = v.transient()
        t.set_mut(25, 999)
        v2 = t.persistent()
        assert v2[25] == 999
        assert v[25] == 25  # original unchanged

    def test_persisted_error(self):
        t = TransientVector(PersistentVector.empty())
        t.append_mut(1)
        t.persistent()
        with pytest.raises(RuntimeError):
            t.append_mut(2)

    def test_large_batch(self):
        t = TransientVector(PersistentVector.empty())
        for i in range(10000):
            t.append_mut(i)
        v = t.persistent()
        assert len(v) == 10000
        assert v[9999] == 9999


# ============================================================
# PersistentList Tests
# ============================================================

class TestPersistentListBasic:
    def test_empty(self):
        assert len(NIL) == 0
        assert not NIL

    def test_cons(self):
        lst = PersistentList.of(1, 2, 3)
        assert len(lst) == 3
        assert lst.head == 1
        assert lst.first == 1

    def test_tail(self):
        lst = PersistentList.of(1, 2, 3)
        t = lst.tail
        assert len(t) == 2
        assert t.head == 2

    def test_cons_creates_new(self):
        lst = PersistentList.of(2, 3)
        lst2 = lst.cons(1)
        assert len(lst2) == 3
        assert lst2.head == 1
        assert len(lst) == 2  # original unchanged

    def test_of(self):
        lst = PersistentList.of(10, 20, 30)
        assert list(lst) == [10, 20, 30]

    def test_from_iterable(self):
        lst = PersistentList.from_iterable(range(5))
        assert list(lst) == [0, 1, 2, 3, 4]

    def test_get(self):
        lst = PersistentList.of(10, 20, 30)
        assert lst[0] == 10
        assert lst[1] == 20
        assert lst[2] == 30

    def test_get_negative(self):
        lst = PersistentList.of(10, 20, 30)
        assert lst[-1] == 30
        assert lst[-2] == 20

    def test_get_error(self):
        lst = PersistentList.of(1)
        with pytest.raises(IndexError):
            lst[5]


class TestPersistentListOps:
    def test_iter(self):
        lst = PersistentList.of(1, 2, 3)
        assert list(lst) == [1, 2, 3]

    def test_eq(self):
        l1 = PersistentList.of(1, 2, 3)
        l2 = PersistentList.of(1, 2, 3)
        l3 = PersistentList.of(1, 2, 4)
        assert l1 == l2
        assert l1 != l3

    def test_nil_eq(self):
        assert NIL == NIL
        assert NIL == PersistentList.empty()

    def test_hash_equal(self):
        l1 = PersistentList.of(1, 2, 3)
        l2 = PersistentList.of(1, 2, 3)
        assert hash(l1) == hash(l2)

    def test_repr(self):
        lst = PersistentList.of(1, 2)
        assert "PList" in repr(lst)

    def test_map(self):
        lst = PersistentList.of(1, 2, 3)
        lst2 = lst.map(lambda x: x * 10)
        assert list(lst2) == [10, 20, 30]

    def test_filter(self):
        lst = PersistentList.of(1, 2, 3, 4, 5)
        lst2 = lst.filter(lambda x: x > 3)
        assert list(lst2) == [4, 5]

    def test_reduce(self):
        lst = PersistentList.of(1, 2, 3)
        assert lst.reduce(lambda a, b: a + b) == 6

    def test_reduce_with_init(self):
        lst = PersistentList.of(1, 2, 3)
        assert lst.reduce(lambda a, b: a + b, 100) == 106

    def test_reverse(self):
        lst = PersistentList.of(1, 2, 3)
        rev = lst.reverse()
        assert list(rev) == [3, 2, 1]
        assert list(lst) == [1, 2, 3]  # original unchanged

    def test_take(self):
        lst = PersistentList.of(1, 2, 3, 4, 5)
        t = lst.take(3)
        assert list(t) == [1, 2, 3]

    def test_drop(self):
        lst = PersistentList.of(1, 2, 3, 4, 5)
        d = lst.drop(2)
        assert list(d) == [3, 4, 5]

    def test_drop_all(self):
        lst = PersistentList.of(1, 2)
        d = lst.drop(5)
        assert len(d) == 0

    def test_concat(self):
        l1 = PersistentList.of(1, 2)
        l2 = PersistentList.of(3, 4)
        l3 = l1.concat(l2)
        assert list(l3) == [1, 2, 3, 4]

    def test_to_list(self):
        lst = PersistentList.of(1, 2, 3)
        assert lst.to_list() == [1, 2, 3]

    def test_structural_sharing(self):
        base = PersistentList.of(3, 4, 5)
        a = base.cons(2)
        b = base.cons(99)
        # Both share the same tail
        assert list(a) == [2, 3, 4, 5]
        assert list(b) == [99, 3, 4, 5]
        assert a.rest is b.rest  # literally same object


# ============================================================
# PersistentHashMap Tests
# ============================================================

class TestPersistentHashMapBasic:
    def test_empty(self):
        m = PersistentHashMap.empty()
        assert len(m) == 0
        assert not m

    def test_set_and_get(self):
        m = PersistentHashMap.empty()
        m1 = m.set("a", 1)
        assert m1["a"] == 1
        assert len(m1) == 1
        assert len(m) == 0  # original unchanged

    def test_multiple_sets(self):
        m = PersistentHashMap.empty()
        m = m.set("a", 1).set("b", 2).set("c", 3)
        assert m["a"] == 1
        assert m["b"] == 2
        assert m["c"] == 3
        assert len(m) == 3

    def test_overwrite(self):
        m = PersistentHashMap.empty().set("a", 1)
        m2 = m.set("a", 99)
        assert m2["a"] == 99
        assert m["a"] == 1
        assert len(m2) == 1  # count unchanged

    def test_get_default(self):
        m = PersistentHashMap.empty()
        assert m.get("missing") is None
        assert m.get("missing", 42) == 42

    def test_contains(self):
        m = PersistentHashMap.empty().set("x", 10)
        assert "x" in m
        assert "y" not in m

    def test_key_error(self):
        m = PersistentHashMap.empty()
        with pytest.raises(KeyError):
            m["missing"]

    def test_of(self):
        m = PersistentHashMap.of(a=1, b=2)
        assert m["a"] == 1
        assert m["b"] == 2

    def test_from_dict(self):
        m = PersistentHashMap.from_dict({"x": 10, "y": 20})
        assert m["x"] == 10
        assert m["y"] == 20
        assert len(m) == 2

    def test_from_pairs(self):
        m = PersistentHashMap.from_pairs([("a", 1), ("b", 2)])
        assert m["a"] == 1
        assert m["b"] == 2


class TestPersistentHashMapDelete:
    def test_delete_basic(self):
        m = PersistentHashMap.empty().set("a", 1).set("b", 2)
        m2 = m.delete("a")
        assert "a" not in m2
        assert m2["b"] == 2
        assert len(m2) == 1
        assert "a" in m  # original unchanged

    def test_delete_missing(self):
        m = PersistentHashMap.empty().set("a", 1)
        m2 = m.delete("missing")
        assert m2 is m  # returns same object

    def test_delete_all(self):
        m = PersistentHashMap.empty().set("a", 1).set("b", 2)
        m = m.delete("a").delete("b")
        assert len(m) == 0

    def test_delete_and_readd(self):
        m = PersistentHashMap.empty().set("a", 1)
        m = m.delete("a")
        m = m.set("a", 99)
        assert m["a"] == 99
        assert len(m) == 1


class TestPersistentHashMapLarge:
    def test_many_entries(self):
        m = PersistentHashMap.empty()
        for i in range(500):
            m = m.set(f"key_{i}", i)
        assert len(m) == 500
        for i in range(500):
            assert m[f"key_{i}"] == i

    def test_integer_keys(self):
        m = PersistentHashMap.empty()
        for i in range(200):
            m = m.set(i, i * 10)
        assert len(m) == 200
        for i in range(200):
            assert m[i] == i * 10

    def test_structural_sharing(self):
        m1 = PersistentHashMap.empty()
        for i in range(100):
            m1 = m1.set(i, i)
        m2 = m1.set(50, 999)
        assert m1[50] == 50
        assert m2[50] == 999
        # Both still correct for other keys
        for i in range(100):
            if i != 50:
                assert m1[i] == m2[i] == i


class TestPersistentHashMapIterAndOps:
    def test_keys(self):
        m = PersistentHashMap.of(a=1, b=2, c=3)
        assert sorted(m.keys()) == ["a", "b", "c"]

    def test_values(self):
        m = PersistentHashMap.of(a=1, b=2, c=3)
        assert sorted(m.values()) == [1, 2, 3]

    def test_items(self):
        m = PersistentHashMap.of(a=1, b=2)
        assert sorted(m.items()) == [("a", 1), ("b", 2)]

    def test_iter(self):
        m = PersistentHashMap.of(x=10)
        assert list(m) == ["x"]

    def test_eq(self):
        m1 = PersistentHashMap.of(a=1, b=2)
        m2 = PersistentHashMap.of(a=1, b=2)
        m3 = PersistentHashMap.of(a=1, b=3)
        assert m1 == m2
        assert m1 != m3

    def test_hash_equal(self):
        m1 = PersistentHashMap.of(a=1, b=2)
        m2 = PersistentHashMap.of(a=1, b=2)
        assert hash(m1) == hash(m2)

    def test_repr(self):
        m = PersistentHashMap.of(a=1)
        assert "PMap" in repr(m)

    def test_merge(self):
        m1 = PersistentHashMap.of(a=1, b=2)
        m2 = PersistentHashMap.of(b=20, c=30)
        m3 = m1.merge(m2)
        assert m3["a"] == 1
        assert m3["b"] == 20  # m2 wins
        assert m3["c"] == 30
        assert len(m3) == 3

    def test_map_values(self):
        m = PersistentHashMap.of(a=1, b=2, c=3)
        m2 = m.map_values(lambda v: v * 10)
        assert m2["a"] == 10
        assert m2["b"] == 20

    def test_filter_entries(self):
        m = PersistentHashMap.of(a=1, b=2, c=3)
        m2 = m.filter_entries(lambda k, v: v > 1)
        assert "a" not in m2
        assert m2["b"] == 2
        assert m2["c"] == 3

    def test_to_dict(self):
        m = PersistentHashMap.of(a=1, b=2)
        assert m.to_dict() == {"a": 1, "b": 2}

    def test_update(self):
        m = PersistentHashMap.of(count=5)
        m2 = m.update("count", lambda x: x + 1)
        assert m2["count"] == 6
        assert m["count"] == 5

    def test_update_missing(self):
        m = PersistentHashMap.empty()
        m2 = m.update("x", lambda x: x + 1, 0)
        assert m2["x"] == 1


class TestTransientHashMap:
    def test_basic(self):
        t = TransientHashMap(PersistentHashMap.empty())
        for i in range(100):
            t.set_mut(f"k{i}", i)
        m = t.persistent()
        assert len(m) == 100
        assert m["k50"] == 50

    def test_delete_mut(self):
        m = PersistentHashMap.from_dict({"a": 1, "b": 2, "c": 3})
        t = m.transient()
        t.delete_mut("b")
        m2 = t.persistent()
        assert len(m2) == 2
        assert "b" not in m2

    def test_persisted_error(self):
        t = TransientHashMap(PersistentHashMap.empty())
        t.set_mut("a", 1)
        t.persistent()
        with pytest.raises(RuntimeError):
            t.set_mut("b", 2)


class TestHashMapCollisions:
    """Test hash collision handling."""

    def test_collision_keys(self):
        """Use objects with controlled hashes to force collisions."""
        class HashCollider:
            def __init__(self, val, h):
                self.val = val
                self._h = h
            def __hash__(self):
                return self._h
            def __eq__(self, other):
                return isinstance(other, HashCollider) and self.val == other.val
            def __repr__(self):
                return f"HC({self.val})"

        k1 = HashCollider("a", 42)
        k2 = HashCollider("b", 42)  # same hash
        k3 = HashCollider("c", 42)  # same hash

        m = PersistentHashMap.empty()
        m = m.set(k1, 1).set(k2, 2).set(k3, 3)
        assert len(m) == 3
        assert m[k1] == 1
        assert m[k2] == 2
        assert m[k3] == 3

        # Update collision entry
        m2 = m.set(k2, 99)
        assert m2[k2] == 99
        assert len(m2) == 3

        # Delete from collision
        m3 = m.delete(k2)
        assert len(m3) == 2
        assert k2 not in m3
        assert m3[k1] == 1
        assert m3[k3] == 3


# ============================================================
# PersistentSortedSet Tests
# ============================================================

class TestPersistentSortedSetBasic:
    def test_empty(self):
        s = PersistentSortedSet.empty()
        assert len(s) == 0
        assert not s

    def test_add(self):
        s = PersistentSortedSet.empty()
        s1 = s.add(5)
        assert len(s1) == 1
        assert 5 in s1
        assert len(s) == 0  # original unchanged

    def test_add_multiple(self):
        s = PersistentSortedSet.of(3, 1, 4, 1, 5, 9)
        assert len(s) == 5  # duplicates ignored
        assert list(s) == [1, 3, 4, 5, 9]  # sorted

    def test_add_duplicate(self):
        s = PersistentSortedSet.of(1, 2, 3)
        s2 = s.add(2)
        assert len(s2) == 3  # no change

    def test_contains(self):
        s = PersistentSortedSet.of(10, 20, 30)
        assert 10 in s
        assert 20 in s
        assert 15 not in s

    def test_of(self):
        s = PersistentSortedSet.of(5, 3, 7, 1)
        assert list(s) == [1, 3, 5, 7]

    def test_from_iterable(self):
        s = PersistentSortedSet.from_iterable(range(10))
        assert len(s) == 10
        assert list(s) == list(range(10))


class TestPersistentSortedSetRemove:
    def test_remove_basic(self):
        s = PersistentSortedSet.of(1, 2, 3, 4, 5)
        s2 = s.remove(3)
        assert len(s2) == 4
        assert 3 not in s2
        assert list(s2) == [1, 2, 4, 5]
        assert 3 in s  # original unchanged

    def test_remove_missing(self):
        s = PersistentSortedSet.of(1, 2, 3)
        s2 = s.remove(99)
        assert s2 is s  # returns same object

    def test_remove_all(self):
        s = PersistentSortedSet.of(1, 2, 3)
        s = s.remove(1).remove(2).remove(3)
        assert len(s) == 0

    def test_remove_and_readd(self):
        s = PersistentSortedSet.of(1, 2, 3)
        s = s.remove(2).add(2)
        assert 2 in s
        assert list(s) == [1, 2, 3]


class TestPersistentSortedSetMinMax:
    def test_min(self):
        s = PersistentSortedSet.of(5, 3, 8, 1, 9)
        assert s.min() == 1

    def test_max(self):
        s = PersistentSortedSet.of(5, 3, 8, 1, 9)
        assert s.max() == 9

    def test_min_empty_error(self):
        with pytest.raises(ValueError):
            PersistentSortedSet.empty().min()

    def test_max_empty_error(self):
        with pytest.raises(ValueError):
            PersistentSortedSet.empty().max()


class TestPersistentSortedSetOps:
    def test_iter_sorted(self):
        s = PersistentSortedSet.of(5, 1, 3, 7, 2)
        assert list(s) == [1, 2, 3, 5, 7]

    def test_eq(self):
        s1 = PersistentSortedSet.of(1, 2, 3)
        s2 = PersistentSortedSet.of(3, 2, 1)  # same set
        s3 = PersistentSortedSet.of(1, 2, 4)
        assert s1 == s2
        assert s1 != s3

    def test_hash_equal(self):
        s1 = PersistentSortedSet.of(1, 2, 3)
        s2 = PersistentSortedSet.of(1, 2, 3)
        assert hash(s1) == hash(s2)

    def test_repr(self):
        s = PersistentSortedSet.of(1, 2, 3)
        assert "PSortedSet" in repr(s)

    def test_range_query(self):
        s = PersistentSortedSet.from_iterable(range(20))
        result = list(s.range_query(5, 10))
        assert result == [5, 6, 7, 8, 9, 10]

    def test_range_query_empty(self):
        s = PersistentSortedSet.of(1, 5, 10)
        result = list(s.range_query(6, 9))
        assert result == []

    def test_union(self):
        s1 = PersistentSortedSet.of(1, 2, 3)
        s2 = PersistentSortedSet.of(3, 4, 5)
        s3 = s1.union(s2)
        assert list(s3) == [1, 2, 3, 4, 5]

    def test_intersection(self):
        s1 = PersistentSortedSet.of(1, 2, 3, 4)
        s2 = PersistentSortedSet.of(3, 4, 5, 6)
        s3 = s1.intersection(s2)
        assert list(s3) == [3, 4]

    def test_difference(self):
        s1 = PersistentSortedSet.of(1, 2, 3, 4)
        s2 = PersistentSortedSet.of(2, 4)
        s3 = s1.difference(s2)
        assert list(s3) == [1, 3]

    def test_to_list(self):
        s = PersistentSortedSet.of(3, 1, 2)
        assert s.to_list() == [1, 2, 3]

    def test_nth(self):
        s = PersistentSortedSet.of(10, 20, 30, 40, 50)
        assert s.nth(0) == 10
        assert s.nth(2) == 30
        assert s.nth(4) == 50

    def test_nth_error(self):
        s = PersistentSortedSet.of(1)
        with pytest.raises(IndexError):
            s.nth(5)

    def test_custom_comparator(self):
        # Reverse order
        s = PersistentSortedSet.of(1, 2, 3, comparator=lambda a, b: -1 if a > b else 1 if a < b else 0)
        assert list(s) == [3, 2, 1]


class TestPersistentSortedSetLarge:
    def test_large_insertions(self):
        s = PersistentSortedSet.empty()
        for i in range(500):
            s = s.add(i)
        assert len(s) == 500
        assert list(s) == list(range(500))

    def test_large_deletions(self):
        s = PersistentSortedSet.from_iterable(range(200))
        for i in range(0, 200, 2):
            s = s.remove(i)
        assert len(s) == 100
        assert list(s) == list(range(1, 200, 2))

    def test_structural_sharing_sorted(self):
        s1 = PersistentSortedSet.from_iterable(range(100))
        s2 = s1.add(999)
        assert len(s1) == 100
        assert len(s2) == 101
        assert 999 not in s1
        assert 999 in s2


# ============================================================
# Cross-structure tests
# ============================================================

class TestCrossStructure:
    def test_vector_of_maps(self):
        v = PersistentVector.of(
            PersistentHashMap.of(name="Alice", age=30),
            PersistentHashMap.of(name="Bob", age=25),
        )
        assert v[0]["name"] == "Alice"
        assert v[1]["age"] == 25

    def test_map_of_vectors(self):
        m = PersistentHashMap.empty()
        m = m.set("nums", PersistentVector.of(1, 2, 3))
        m = m.set("chars", PersistentVector.of("a", "b", "c"))
        assert m["nums"][1] == 2
        assert m["chars"][2] == "c"

    def test_vector_to_set(self):
        v = PersistentVector.of(3, 1, 4, 1, 5, 9, 2, 6, 5)
        s = PersistentSortedSet.from_iterable(v)
        assert list(s) == [1, 2, 3, 4, 5, 6, 9]

    def test_list_to_vector(self):
        lst = PersistentList.of(1, 2, 3, 4, 5)
        v = PersistentVector.from_iterable(lst)
        assert list(v) == [1, 2, 3, 4, 5]

    def test_map_keys_to_sorted_set(self):
        m = PersistentHashMap.of(c=3, a=1, b=2)
        s = PersistentSortedSet.from_iterable(m.keys())
        assert list(s) == ["a", "b", "c"]


class TestImmutabilityGuarantees:
    def test_vector_immutability_chain(self):
        versions = [PersistentVector.empty()]
        for i in range(20):
            versions.append(versions[-1].append(i))
        # Each version has correct length and contents
        for i, v in enumerate(versions):
            assert len(v) == i
            for j in range(i):
                assert v[j] == j

    def test_map_immutability_chain(self):
        versions = [PersistentHashMap.empty()]
        for i in range(20):
            versions.append(versions[-1].set(f"k{i}", i))
        for i, m in enumerate(versions):
            assert len(m) == i
            for j in range(i):
                assert m[f"k{j}"] == j

    def test_sorted_set_immutability_chain(self):
        versions = [PersistentSortedSet.empty()]
        for i in range(20):
            versions.append(versions[-1].add(i))
        for i, s in enumerate(versions):
            assert len(s) == i
            for j in range(i):
                assert j in s

    def test_list_sharing(self):
        base = PersistentList.of(4, 5, 6)
        a = base.cons(3).cons(2).cons(1)
        b = base.cons(30).cons(20).cons(10)
        assert list(a) == [1, 2, 3, 4, 5, 6]
        assert list(b) == [10, 20, 30, 4, 5, 6]
        # Shared tail
        assert a.rest.rest.rest is b.rest.rest.rest
