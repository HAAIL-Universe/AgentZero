"""Tests for C105: Van Emde Boas Tree and variants."""
import pytest
import random
from van_emde_boas import VEBTree, VEBSet, VEBMap, XFastTrie, YFastTrie


# ===================================================================
# VEBTree -- Core
# ===================================================================

class TestVEBTreeBasic:
    def test_create_empty(self):
        t = VEBTree(16)
        assert t.u == 16
        assert t.min is None
        assert t.max is None
        assert len(t) == 0
        assert not t

    def test_universe_rounds_up(self):
        t = VEBTree(5)
        assert t.u == 8  # Next power of 2

    def test_universe_power_of_2(self):
        t = VEBTree(16)
        assert t.u == 16

    def test_min_universe(self):
        t = VEBTree(2)
        assert t.u == 2

    def test_invalid_universe(self):
        with pytest.raises(ValueError):
            VEBTree(1)

    def test_insert_single(self):
        t = VEBTree(16)
        t.insert(5)
        assert t.min == 5
        assert t.max == 5
        assert len(t) == 1
        assert 5 in t

    def test_insert_two(self):
        t = VEBTree(16)
        t.insert(5)
        t.insert(10)
        assert t.min == 5
        assert t.max == 10
        assert len(t) == 2

    def test_insert_duplicate(self):
        t = VEBTree(16)
        t.insert(5)
        t.insert(5)
        assert len(t) == 1

    def test_insert_all(self):
        t = VEBTree(8)
        for i in range(8):
            t.insert(i)
        assert len(t) == 8
        assert t.min == 0
        assert t.max == 7

    def test_insert_reverse_order(self):
        t = VEBTree(16)
        for i in [15, 10, 5, 0]:
            t.insert(i)
        assert t.min == 0
        assert t.max == 15
        assert len(t) == 4

    def test_insert_out_of_range(self):
        t = VEBTree(16)
        with pytest.raises(ValueError):
            t.insert(16)
        with pytest.raises(ValueError):
            t.insert(-1)

    def test_member(self):
        t = VEBTree(16)
        t.insert(3)
        t.insert(7)
        t.insert(11)
        assert t.member(3)
        assert t.member(7)
        assert t.member(11)
        assert not t.member(0)
        assert not t.member(5)
        assert not t.member(15)

    def test_member_empty(self):
        t = VEBTree(16)
        assert not t.member(0)

    def test_member_out_of_range(self):
        t = VEBTree(16)
        assert not t.member(-1)
        assert not t.member(16)

    def test_contains(self):
        t = VEBTree(16)
        t.insert(5)
        assert 5 in t
        assert 6 not in t


class TestVEBTreeDelete:
    def test_delete_only_element(self):
        t = VEBTree(16)
        t.insert(5)
        t.delete(5)
        assert t.min is None
        assert t.max is None
        assert len(t) == 0

    def test_delete_min(self):
        t = VEBTree(16)
        t.insert(3)
        t.insert(7)
        t.delete(3)
        assert t.min == 7
        assert t.max == 7
        assert len(t) == 1

    def test_delete_max(self):
        t = VEBTree(16)
        t.insert(3)
        t.insert(7)
        t.delete(7)
        assert t.min == 3
        assert t.max == 3
        assert len(t) == 1

    def test_delete_middle(self):
        t = VEBTree(16)
        t.insert(2)
        t.insert(5)
        t.insert(10)
        t.delete(5)
        assert 2 in t
        assert 5 not in t
        assert 10 in t
        assert len(t) == 2

    def test_delete_nonexistent(self):
        t = VEBTree(16)
        t.insert(5)
        t.delete(3)  # No-op
        assert len(t) == 1

    def test_delete_from_empty(self):
        t = VEBTree(16)
        t.delete(5)  # No-op
        assert len(t) == 0

    def test_delete_all(self):
        t = VEBTree(16)
        elems = [1, 5, 9, 13]
        for e in elems:
            t.insert(e)
        for e in elems:
            t.delete(e)
        assert len(t) == 0
        assert t.min is None

    def test_delete_out_of_range(self):
        t = VEBTree(16)
        t.insert(5)
        t.delete(-1)  # No-op
        t.delete(16)  # No-op
        assert len(t) == 1

    def test_delete_and_reinsert(self):
        t = VEBTree(16)
        t.insert(5)
        t.delete(5)
        t.insert(5)
        assert 5 in t
        assert len(t) == 1


class TestVEBTreeSuccessorPredecessor:
    def test_successor_basic(self):
        t = VEBTree(16)
        for x in [2, 5, 9, 14]:
            t.insert(x)
        assert t.successor(0) == 2
        assert t.successor(2) == 5
        assert t.successor(3) == 5
        assert t.successor(5) == 9
        assert t.successor(9) == 14
        assert t.successor(14) is None

    def test_successor_empty(self):
        t = VEBTree(16)
        assert t.successor(5) is None

    def test_successor_below_min(self):
        t = VEBTree(16)
        t.insert(5)
        assert t.successor(3) == 5

    def test_predecessor_basic(self):
        t = VEBTree(16)
        for x in [2, 5, 9, 14]:
            t.insert(x)
        assert t.predecessor(15) == 14
        assert t.predecessor(14) == 9
        assert t.predecessor(10) == 9
        assert t.predecessor(9) == 5
        assert t.predecessor(5) == 2
        assert t.predecessor(2) is None

    def test_predecessor_empty(self):
        t = VEBTree(16)
        assert t.predecessor(5) is None

    def test_predecessor_above_max(self):
        t = VEBTree(16)
        t.insert(5)
        assert t.predecessor(10) == 5

    def test_successor_consecutive(self):
        t = VEBTree(8)
        for i in range(8):
            t.insert(i)
        for i in range(7):
            assert t.successor(i) == i + 1
        assert t.successor(7) is None

    def test_predecessor_consecutive(self):
        t = VEBTree(8)
        for i in range(8):
            t.insert(i)
        assert t.predecessor(0) is None
        for i in range(1, 8):
            assert t.predecessor(i) == i - 1


class TestVEBTreeIteration:
    def test_iter_empty(self):
        t = VEBTree(16)
        assert list(t) == []

    def test_iter_sorted(self):
        t = VEBTree(16)
        for x in [10, 2, 7, 15, 0]:
            t.insert(x)
        assert list(t) == [0, 2, 7, 10, 15]

    def test_to_sorted_list(self):
        t = VEBTree(32)
        elems = [20, 5, 15, 0, 31]
        for x in elems:
            t.insert(x)
        assert t.to_sorted_list() == sorted(elems)

    def test_range_query(self):
        t = VEBTree(32)
        for x in [1, 5, 10, 15, 20, 25, 30]:
            t.insert(x)
        assert t.range_query(5, 20) == [5, 10, 15, 20]
        assert t.range_query(6, 14) == [10]
        assert t.range_query(0, 100) == [1, 5, 10, 15, 20, 25, 30]
        assert t.range_query(50, 60) == []

    def test_repr(self):
        t = VEBTree(16)
        t.insert(3)
        r = repr(t)
        assert "VEBTree" in r


class TestVEBTreeLargeUniverse:
    def test_large_universe(self):
        t = VEBTree(65536)
        t.insert(0)
        t.insert(65535)
        t.insert(32768)
        assert t.min == 0
        assert t.max == 65535
        assert len(t) == 3
        assert t.successor(0) == 32768
        assert t.predecessor(65535) == 32768

    def test_sparse_large(self):
        t = VEBTree(1024)
        elems = [0, 100, 200, 500, 999]
        for x in elems:
            t.insert(x)
        assert list(t) == sorted(elems)
        assert t.successor(100) == 200
        assert t.predecessor(500) == 200

    def test_random_operations(self):
        """Fuzz test: random insert/delete/query against a Python set."""
        random.seed(42)
        t = VEBTree(256)
        reference = set()
        for _ in range(200):
            op = random.choice(['insert', 'insert', 'delete', 'member', 'succ', 'pred'])
            x = random.randint(0, 255)
            if op == 'insert':
                t.insert(x)
                reference.add(x)
            elif op == 'delete':
                t.delete(x)
                reference.discard(x)
            elif op == 'member':
                assert t.member(x) == (x in reference)
            elif op == 'succ':
                expected = min((e for e in reference if e > x), default=None)
                assert t.successor(x) == expected, f"successor({x}): got {t.successor(x)}, expected {expected}"
            elif op == 'pred':
                expected = max((e for e in reference if e < x), default=None)
                assert t.predecessor(x) == expected, f"predecessor({x}): got {t.predecessor(x)}, expected {expected}"
        assert len(t) == len(reference)

    def test_base_case_u2(self):
        """Test universe size 2 (base case)."""
        t = VEBTree(2)
        assert t.u == 2
        t.insert(0)
        assert t.min == 0
        assert t.max == 0
        t.insert(1)
        assert t.min == 0
        assert t.max == 1
        assert t.successor(0) == 1
        assert t.predecessor(1) == 0
        t.delete(0)
        assert t.min == 1
        assert t.max == 1

    def test_universe_4(self):
        t = VEBTree(4)
        t.insert(0)
        t.insert(3)
        assert t.successor(0) == 3
        assert t.predecessor(3) == 0
        t.insert(1)
        assert t.successor(0) == 1
        assert t.successor(1) == 3


# ===================================================================
# VEBSet
# ===================================================================

class TestVEBSet:
    def test_create_empty(self):
        s = VEBSet(16)
        assert len(s) == 0
        assert not s

    def test_create_from_iterable(self):
        s = VEBSet(16, [3, 7, 11])
        assert len(s) == 3
        assert 3 in s
        assert 7 in s
        assert 11 in s

    def test_add_discard_remove(self):
        s = VEBSet(16)
        s.add(5)
        assert 5 in s
        s.discard(5)
        assert 5 not in s
        s.add(10)
        s.remove(10)
        assert 10 not in s

    def test_remove_missing_raises(self):
        s = VEBSet(16)
        with pytest.raises(KeyError):
            s.remove(5)

    def test_discard_missing(self):
        s = VEBSet(16)
        s.discard(5)  # No error

    def test_min_max(self):
        s = VEBSet(16, [3, 7, 11])
        assert s.min == 3
        assert s.max == 11

    def test_successor_predecessor(self):
        s = VEBSet(16, [2, 5, 10])
        assert s.successor(2) == 5
        assert s.predecessor(10) == 5

    def test_iter_sorted(self):
        s = VEBSet(16, [10, 3, 7, 15])
        assert list(s) == [3, 7, 10, 15]

    def test_union(self):
        a = VEBSet(16, [1, 3, 5])
        b = VEBSet(16, [2, 3, 6])
        c = a.union(b)
        assert list(c) == [1, 2, 3, 5, 6]

    def test_intersection(self):
        a = VEBSet(16, [1, 3, 5, 7])
        b = VEBSet(16, [3, 5, 9])
        c = a.intersection(b)
        assert list(c) == [3, 5]

    def test_difference(self):
        a = VEBSet(16, [1, 3, 5, 7])
        b = VEBSet(16, [3, 7])
        c = a.difference(b)
        assert list(c) == [1, 5]

    def test_symmetric_difference(self):
        a = VEBSet(16, [1, 3, 5])
        b = VEBSet(16, [3, 5, 7])
        c = a.symmetric_difference(b)
        assert list(c) == [1, 7]

    def test_issubset(self):
        a = VEBSet(16, [3, 5])
        b = VEBSet(16, [1, 3, 5, 7])
        assert a.issubset(b)
        assert not b.issubset(a)

    def test_issuperset(self):
        a = VEBSet(16, [1, 3, 5, 7])
        b = VEBSet(16, [3, 5])
        assert a.issuperset(b)

    def test_range_query(self):
        s = VEBSet(32, [1, 5, 10, 15, 20])
        assert s.range_query(5, 15) == [5, 10, 15]

    def test_repr(self):
        s = VEBSet(16, [3, 7])
        r = repr(s)
        assert "VEBSet" in r


# ===================================================================
# VEBMap
# ===================================================================

class TestVEBMap:
    def test_create_empty(self):
        m = VEBMap(16)
        assert len(m) == 0
        assert not m

    def test_setitem_getitem(self):
        m = VEBMap(16)
        m[5] = "hello"
        assert m[5] == "hello"
        assert len(m) == 1

    def test_overwrite(self):
        m = VEBMap(16)
        m[5] = "a"
        m[5] = "b"
        assert m[5] == "b"

    def test_delitem(self):
        m = VEBMap(16)
        m[5] = "x"
        del m[5]
        assert 5 not in m
        assert len(m) == 0

    def test_delitem_missing(self):
        m = VEBMap(16)
        with pytest.raises(KeyError):
            del m[5]

    def test_getitem_missing(self):
        m = VEBMap(16)
        with pytest.raises(KeyError):
            _ = m[5]

    def test_get_default(self):
        m = VEBMap(16)
        assert m.get(5) is None
        assert m.get(5, "default") == "default"
        m[5] = "val"
        assert m.get(5) == "val"

    def test_contains(self):
        m = VEBMap(16)
        m[3] = "x"
        assert 3 in m
        assert 4 not in m

    def test_keys_values_items(self):
        m = VEBMap(16)
        m[10] = "ten"
        m[3] = "three"
        m[7] = "seven"
        assert list(m.keys()) == [3, 7, 10]
        assert list(m.values()) == ["three", "seven", "ten"]
        assert list(m.items()) == [(3, "three"), (7, "seven"), (10, "ten")]

    def test_iter(self):
        m = VEBMap(16)
        m[5] = "a"
        m[2] = "b"
        assert list(m) == [2, 5]

    def test_min_max(self):
        m = VEBMap(16)
        m[3] = "a"
        m[10] = "b"
        assert m.min_key == 3
        assert m.max_key == 10
        assert m.min_item() == (3, "a")
        assert m.max_item() == (10, "b")

    def test_min_max_empty(self):
        m = VEBMap(16)
        assert m.min_key is None
        assert m.max_key is None
        assert m.min_item() is None
        assert m.max_item() is None

    def test_successor_predecessor(self):
        m = VEBMap(16)
        m[2] = "a"
        m[5] = "b"
        m[10] = "c"
        assert m.successor(2) == (5, "b")
        assert m.successor(5) == (10, "c")
        assert m.successor(10) is None
        assert m.predecessor(10) == (5, "b")
        assert m.predecessor(5) == (2, "a")
        assert m.predecessor(2) is None

    def test_range_query(self):
        m = VEBMap(32)
        for i in [1, 5, 10, 15, 20]:
            m[i] = str(i)
        result = m.range_query(5, 15)
        assert result == [(5, "5"), (10, "10"), (15, "15")]

    def test_pop(self):
        m = VEBMap(16)
        m[5] = "x"
        assert m.pop(5) == "x"
        assert 5 not in m
        assert m.pop(5, "default") == "default"
        with pytest.raises(KeyError):
            m.pop(5)

    def test_repr(self):
        m = VEBMap(16)
        m[3] = "a"
        r = repr(m)
        assert "VEBMap" in r


# ===================================================================
# XFastTrie
# ===================================================================

class TestXFastTrie:
    def test_create_empty(self):
        t = XFastTrie(16)
        assert t.u == 16
        assert len(t) == 0
        assert not t

    def test_insert_member(self):
        t = XFastTrie(16)
        t.insert(5)
        assert t.member(5)
        assert 5 in t
        assert not t.member(6)

    def test_insert_multiple(self):
        t = XFastTrie(16)
        for x in [3, 7, 11, 15]:
            t.insert(x)
        assert len(t) == 4
        for x in [3, 7, 11, 15]:
            assert x in t

    def test_insert_duplicate(self):
        t = XFastTrie(16)
        t.insert(5)
        t.insert(5)
        assert len(t) == 1

    def test_min_max(self):
        t = XFastTrie(16)
        t.insert(5)
        t.insert(10)
        t.insert(2)
        assert t.min == 2
        assert t.max == 10

    def test_delete(self):
        t = XFastTrie(16)
        t.insert(5)
        t.insert(10)
        t.delete(5)
        assert 5 not in t
        assert 10 in t
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = XFastTrie(16)
        t.insert(5)
        t.delete(10)  # No-op
        assert len(t) == 1

    def test_delete_all(self):
        t = XFastTrie(16)
        for x in [3, 7, 11]:
            t.insert(x)
        for x in [3, 7, 11]:
            t.delete(x)
        assert len(t) == 0

    def test_iter_sorted(self):
        t = XFastTrie(16)
        for x in [10, 3, 7, 0]:
            t.insert(x)
        assert list(t) == [0, 3, 7, 10]

    def test_successor(self):
        t = XFastTrie(16)
        for x in [2, 5, 9, 14]:
            t.insert(x)
        assert t.successor(0) == 2
        assert t.successor(2) == 5
        assert t.successor(3) == 5
        assert t.successor(14) is None

    def test_predecessor(self):
        t = XFastTrie(16)
        for x in [2, 5, 9, 14]:
            t.insert(x)
        assert t.predecessor(15) == 14
        assert t.predecessor(14) == 9
        assert t.predecessor(2) is None

    def test_successor_empty(self):
        t = XFastTrie(16)
        assert t.successor(5) is None

    def test_predecessor_empty(self):
        t = XFastTrie(16)
        assert t.predecessor(5) is None

    def test_insert_out_of_range(self):
        t = XFastTrie(16)
        with pytest.raises(ValueError):
            t.insert(16)
        with pytest.raises(ValueError):
            t.insert(-1)

    def test_member_out_of_range(self):
        t = XFastTrie(16)
        assert not t.member(-1)
        assert not t.member(16)

    def test_range_query(self):
        t = XFastTrie(32)
        for x in [1, 5, 10, 15, 20]:
            t.insert(x)
        assert t.range_query(5, 15) == [5, 10, 15]

    def test_repr(self):
        t = XFastTrie(16)
        t.insert(3)
        assert "XFastTrie" in repr(t)

    def test_random_ops(self):
        """Fuzz test against a sorted set."""
        random.seed(123)
        t = XFastTrie(64)
        ref = set()
        for _ in range(100):
            op = random.choice(['insert', 'insert', 'delete', 'member'])
            x = random.randint(0, 63)
            if op == 'insert':
                t.insert(x)
                ref.add(x)
            elif op == 'delete':
                t.delete(x)
                ref.discard(x)
            elif op == 'member':
                assert t.member(x) == (x in ref)
        assert sorted(t) == sorted(ref)


# ===================================================================
# YFastTrie
# ===================================================================

class TestYFastTrie:
    def test_create_empty(self):
        t = YFastTrie(16)
        assert len(t) == 0
        assert not t

    def test_insert_member(self):
        t = YFastTrie(16)
        t.insert(5)
        assert t.member(5)
        assert 5 in t
        assert len(t) == 1

    def test_insert_multiple(self):
        t = YFastTrie(16)
        for x in [3, 7, 11]:
            t.insert(x)
        assert len(t) == 3
        for x in [3, 7, 11]:
            assert x in t

    def test_insert_duplicate(self):
        t = YFastTrie(16)
        t.insert(5)
        t.insert(5)
        assert len(t) == 1

    def test_min_max(self):
        t = YFastTrie(16)
        for x in [5, 2, 10]:
            t.insert(x)
        assert t.min == 2
        assert t.max == 10

    def test_delete(self):
        t = YFastTrie(16)
        t.insert(5)
        t.insert(10)
        t.delete(5)
        assert 5 not in t
        assert 10 in t
        assert len(t) == 1

    def test_delete_nonexistent(self):
        t = YFastTrie(16)
        t.insert(5)
        t.delete(10)
        assert len(t) == 1

    def test_iter_sorted(self):
        t = YFastTrie(16)
        for x in [10, 3, 7, 0]:
            t.insert(x)
        assert list(t) == [0, 3, 7, 10]

    def test_successor(self):
        t = YFastTrie(32)
        for x in [2, 5, 10, 20]:
            t.insert(x)
        assert t.successor(2) == 5
        assert t.successor(5) == 10
        assert t.successor(20) is None

    def test_predecessor(self):
        t = YFastTrie(32)
        for x in [2, 5, 10, 20]:
            t.insert(x)
        assert t.predecessor(20) == 10
        assert t.predecessor(10) == 5
        assert t.predecessor(2) is None

    def test_successor_empty(self):
        t = YFastTrie(16)
        assert t.successor(5) is None

    def test_predecessor_empty(self):
        t = YFastTrie(16)
        assert t.predecessor(5) is None

    def test_range_query(self):
        t = YFastTrie(32)
        for x in [1, 5, 10, 15, 20]:
            t.insert(x)
        assert t.range_query(5, 15) == [5, 10, 15]

    def test_insert_out_of_range(self):
        t = YFastTrie(16)
        with pytest.raises(ValueError):
            t.insert(16)

    def test_repr(self):
        t = YFastTrie(16)
        t.insert(3)
        assert "YFastTrie" in repr(t)

    def test_many_inserts_group_splitting(self):
        """Test that group splitting works with many elements."""
        t = YFastTrie(256)
        elems = list(range(0, 256, 3))
        for x in elems:
            t.insert(x)
        assert len(t) == len(elems)
        assert list(t) == sorted(elems)

    def test_delete_triggers_merge(self):
        """Insert many, delete most, verify merging doesn't break."""
        t = YFastTrie(64)
        for x in range(64):
            t.insert(x)
        for x in range(0, 64, 2):
            t.delete(x)
        remaining = [x for x in range(64) if x % 2 == 1]
        assert list(t) == remaining
        assert len(t) == len(remaining)

    def test_delete_representative(self):
        """Deleting the representative of a group."""
        t = YFastTrie(16)
        t.insert(5)
        t.insert(6)
        # One of these is the representative; delete it
        t.delete(5)
        assert 5 not in t
        assert 6 in t
        assert len(t) == 1


# ===================================================================
# Cross-variant consistency
# ===================================================================

class TestCrossVariant:
    def test_all_variants_agree(self):
        """All four structures give same results for same operations."""
        random.seed(99)
        U = 64
        veb = VEBTree(U)
        xfast = XFastTrie(U)
        yfast = YFastTrie(U)
        vset = VEBSet(U)

        elems = random.sample(range(U), 20)
        for x in elems:
            veb.insert(x)
            xfast.insert(x)
            yfast.insert(x)
            vset.add(x)

        sorted_elems = sorted(elems)
        assert list(veb) == sorted_elems
        assert list(xfast) == sorted_elems
        assert list(yfast) == sorted_elems
        assert list(vset) == sorted_elems

        for x in range(U):
            assert veb.member(x) == xfast.member(x) == yfast.member(x) == (x in vset)

    def test_successor_predecessor_agree(self):
        """All variants give same successor/predecessor."""
        U = 32
        veb = VEBTree(U)
        xfast = XFastTrie(U)
        yfast = YFastTrie(U)

        elems = [3, 7, 15, 20, 28]
        for x in elems:
            veb.insert(x)
            xfast.insert(x)
            yfast.insert(x)

        for x in range(U):
            vs = veb.successor(x)
            xs = xfast.successor(x)
            ys = yfast.successor(x)
            assert vs == xs == ys, f"successor({x}): veb={vs}, xfast={xs}, yfast={ys}"

            vp = veb.predecessor(x)
            xp = xfast.predecessor(x)
            yp = yfast.predecessor(x)
            assert vp == xp == yp, f"predecessor({x}): veb={vp}, xfast={xp}, yfast={yp}"


# ===================================================================
# Edge cases and stress
# ===================================================================

class TestEdgeCases:
    def test_veb_single_element_succ_pred(self):
        t = VEBTree(16)
        t.insert(8)
        assert t.successor(7) == 8
        assert t.successor(8) is None
        assert t.predecessor(9) == 8
        assert t.predecessor(8) is None

    def test_veb_two_adjacent(self):
        t = VEBTree(16)
        t.insert(5)
        t.insert(6)
        assert t.successor(5) == 6
        assert t.predecessor(6) == 5

    def test_veb_boundary_values(self):
        t = VEBTree(16)
        t.insert(0)
        t.insert(15)
        assert t.min == 0
        assert t.max == 15
        assert t.successor(0) == 15
        assert t.predecessor(15) == 0

    def test_veb_delete_and_query(self):
        """After deleting, successor/predecessor should skip deleted."""
        t = VEBTree(16)
        for x in [2, 5, 8, 11]:
            t.insert(x)
        t.delete(5)
        assert t.successor(2) == 8
        assert t.predecessor(8) == 2

    def test_xfast_adjacent_elements(self):
        t = XFastTrie(8)
        for x in range(8):
            t.insert(x)
        for x in range(7):
            assert t.successor(x) == x + 1

    def test_veb_u256_sparse(self):
        """Sparse tree in large universe."""
        t = VEBTree(256)
        t.insert(0)
        t.insert(255)
        assert t.successor(0) == 255
        assert t.predecessor(255) == 0

    def test_vebset_empty_ops(self):
        s = VEBSet(16)
        assert s.min is None
        assert s.max is None
        assert list(s) == []
        assert s.successor(5) is None
        assert s.predecessor(5) is None

    def test_vebmap_iter_empty(self):
        m = VEBMap(16)
        assert list(m.keys()) == []
        assert list(m.values()) == []
        assert list(m.items()) == []

    def test_bool_semantics(self):
        """Empty structures are falsy, non-empty are truthy."""
        for cls in [VEBTree, XFastTrie, YFastTrie]:
            t = cls(16)
            assert not t
            t.insert(5)
            assert t

    def test_veb_insert_delete_insert_sequence(self):
        """Repeated insert-delete cycles."""
        t = VEBTree(16)
        for _ in range(5):
            t.insert(7)
            assert 7 in t
            t.delete(7)
            assert 7 not in t
        assert len(t) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
