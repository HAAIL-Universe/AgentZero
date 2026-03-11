"""
C104: Treap -- Randomized Binary Search Tree

A treap combines a BST (on keys) with a max-heap (on random priorities).
This gives probabilistic O(log n) for insert, delete, search, split, merge.

Features:
- TreapMap: ordered key-value store with split/merge primitives
- TreapSet: ordered set built on TreapMap
- Implicit Treap: sequence with O(log n) insert/delete/reverse (rope-like)
- Persistent Treap: path-copying for immutable versions
- Bulk operations: union, intersection, difference via split/merge
"""

import random
from typing import Any, Optional, Iterator, Tuple, List


# ============================================================
# TreapNode
# ============================================================

class TreapNode:
    """Node for explicit-key treap."""
    __slots__ = ('key', 'value', 'priority', 'left', 'right', 'size')

    def __init__(self, key, value=None, priority=None):
        self.key = key
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.size = 1

    def __repr__(self):
        return f"TreapNode({self.key!r}, pri={self.priority:.3f})"


def _size(node):
    return node.size if node else 0


def _update(node):
    if node:
        node.size = 1 + _size(node.left) + _size(node.right)
    return node


# ============================================================
# Split / Merge primitives
# ============================================================

def split(node, key):
    """Split treap into (left, right) where left has keys < key."""
    if node is None:
        return None, None
    if key <= node.key:
        left, node.left = split(node.left, key)
        _update(node)
        return left, node
    else:
        node.right, right = split(node.right, key)
        _update(node)
        return node, right


def merge(left, right):
    """Merge two treaps where all keys in left < all keys in right."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority >= right.priority:
        left.right = merge(left.right, right)
        _update(left)
        return left
    else:
        right.left = merge(left, right.left)
        _update(right)
        return right


# ============================================================
# TreapMap
# ============================================================

class TreapMap:
    """Ordered key-value map backed by a treap."""

    def __init__(self):
        self._root = None

    def __len__(self):
        return _size(self._root)

    def __bool__(self):
        return self._root is not None

    def __contains__(self, key):
        return self._find(self._root, key) is not None

    def _find(self, node, key):
        while node:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def get(self, key, default=None):
        node = self._find(self._root, key)
        return node.value if node else default

    def __getitem__(self, key):
        node = self._find(self._root, key)
        if node is None:
            raise KeyError(key)
        return node.value

    def __setitem__(self, key, value):
        self.put(key, value)

    def put(self, key, value=None):
        """Insert or update key-value pair."""
        node = self._find(self._root, key)
        if node is not None:
            node.value = value
            return
        new_node = TreapNode(key, value)
        left, right = split(self._root, key)
        self._root = merge(merge(left, new_node), right)

    def delete(self, key):
        """Remove key. Raises KeyError if not found."""
        if key not in self:
            raise KeyError(key)
        self._root = self._delete(self._root, key)

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
            _update(node)
            return node
        elif key > node.key:
            node.right = self._delete(node.right, key)
            _update(node)
            return node
        else:
            return merge(node.left, node.right)

    def pop(self, key, *args):
        """Remove and return value for key."""
        node = self._find(self._root, key)
        if node is None:
            if args:
                return args[0]
            raise KeyError(key)
        val = node.value
        self._root = self._delete(self._root, key)
        return val

    # -- Order statistics --

    def min(self):
        """Return (key, value) of minimum key."""
        if not self._root:
            raise ValueError("empty treap")
        node = self._root
        while node.left:
            node = node.left
        return (node.key, node.value)

    def max(self):
        """Return (key, value) of maximum key."""
        if not self._root:
            raise ValueError("empty treap")
        node = self._root
        while node.right:
            node = node.right
        return (node.key, node.value)

    def kth(self, k):
        """Return (key, value) of k-th element (0-indexed)."""
        if k < 0 or k >= len(self):
            raise IndexError(f"index {k} out of range")
        return self._kth(self._root, k)

    def _kth(self, node, k):
        left_size = _size(node.left)
        if k < left_size:
            return self._kth(node.left, k)
        elif k == left_size:
            return (node.key, node.value)
        else:
            return self._kth(node.right, k - left_size - 1)

    def rank(self, key):
        """Return number of keys strictly less than key."""
        r = 0
        node = self._root
        while node:
            if key <= node.key:
                node = node.left
            else:
                r += _size(node.left) + 1
                node = node.right
        return r

    # -- Range queries --

    def floor(self, key):
        """Largest key <= given key, or None."""
        result = None
        node = self._root
        while node:
            if key == node.key:
                return (node.key, node.value)
            elif key < node.key:
                node = node.left
            else:
                result = (node.key, node.value)
                node = node.right
        return result

    def ceiling(self, key):
        """Smallest key >= given key, or None."""
        result = None
        node = self._root
        while node:
            if key == node.key:
                return (node.key, node.value)
            elif key > node.key:
                node = node.right
            else:
                result = (node.key, node.value)
                node = node.left
        return result

    def range_query(self, lo, hi):
        """Return list of (key, value) pairs where lo <= key <= hi."""
        result = []
        self._range(self._root, lo, hi, result)
        return result

    def _range(self, node, lo, hi, result):
        if node is None:
            return
        if lo <= node.key:
            self._range(node.left, lo, hi, result)
        if lo <= node.key <= hi:
            result.append((node.key, node.value))
        if node.key <= hi:
            self._range(node.right, lo, hi, result)

    # -- Iteration --

    def __iter__(self):
        return self._inorder(self._root)

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield node.key
        yield from self._inorder(node.right)

    def items(self):
        """Iterate (key, value) pairs in order."""
        return self._inorder_items(self._root)

    def _inorder_items(self, node):
        if node is None:
            return
        yield from self._inorder_items(node.left)
        yield (node.key, node.value)
        yield from self._inorder_items(node.right)

    def keys(self):
        return iter(self)

    def values(self):
        for _, v in self.items():
            yield v

    # -- Split / Merge on map level --

    def split_at(self, key):
        """Split into two TreapMaps: keys < key, keys >= key."""
        left, right = split(self._root, key)
        lm, rm = TreapMap(), TreapMap()
        lm._root = left
        rm._root = right
        return lm, rm

    # -- Bulk set operations --

    def update(self, other):
        """Merge all entries from other TreapMap into self."""
        for k, v in other.items():
            self.put(k, v)

    # -- Heap property verification (for testing) --

    def _verify(self):
        """Verify BST and heap invariants. Returns True or raises."""
        self._verify_node(self._root, None, None)
        self._verify_sizes(self._root)
        return True

    def _verify_node(self, node, lo, hi):
        if node is None:
            return
        if lo is not None and node.key <= lo:
            raise AssertionError(f"BST violation: {node.key} <= {lo}")
        if hi is not None and node.key >= hi:
            raise AssertionError(f"BST violation: {node.key} >= {hi}")
        if node.left and node.left.priority > node.priority:
            raise AssertionError(f"Heap violation: child {node.left.priority} > parent {node.priority}")
        if node.right and node.right.priority > node.priority:
            raise AssertionError(f"Heap violation: child {node.right.priority} > parent {node.priority}")
        self._verify_node(node.left, lo, node.key)
        self._verify_node(node.right, node.key, hi)

    def _verify_sizes(self, node):
        if node is None:
            return 0
        ls = self._verify_sizes(node.left)
        rs = self._verify_sizes(node.right)
        expected = ls + rs + 1
        if node.size != expected:
            raise AssertionError(f"Size mismatch at {node.key}: {node.size} != {expected}")
        return expected

    def clear(self):
        self._root = None


# ============================================================
# TreapSet
# ============================================================

class TreapSet:
    """Ordered set backed by a treap."""

    def __init__(self, iterable=None):
        self._map = TreapMap()
        if iterable:
            for item in iterable:
                self.add(item)

    def __len__(self):
        return len(self._map)

    def __bool__(self):
        return bool(self._map)

    def __contains__(self, item):
        return item in self._map

    def add(self, item):
        self._map.put(item)

    def discard(self, item):
        if item in self._map:
            self._map.delete(item)

    def remove(self, item):
        self._map.delete(item)

    def pop_min(self):
        k, _ = self._map.min()
        self._map.delete(k)
        return k

    def pop_max(self):
        k, _ = self._map.max()
        self._map.delete(k)
        return k

    def min(self):
        return self._map.min()[0]

    def max(self):
        return self._map.max()[0]

    def kth(self, k):
        return self._map.kth(k)[0]

    def rank(self, item):
        return self._map.rank(item)

    def __iter__(self):
        return iter(self._map)

    def floor(self, item):
        r = self._map.floor(item)
        return r[0] if r else None

    def ceiling(self, item):
        r = self._map.ceiling(item)
        return r[0] if r else None

    def range_query(self, lo, hi):
        return [k for k, _ in self._map.range_query(lo, hi)]

    def union(self, other):
        result = TreapSet()
        for item in self:
            result.add(item)
        for item in other:
            result.add(item)
        return result

    def intersection(self, other):
        result = TreapSet()
        for item in self:
            if item in other:
                result.add(item)
        return result

    def difference(self, other):
        result = TreapSet()
        for item in self:
            if item not in other:
                result.add(item)
        return result

    def symmetric_difference(self, other):
        result = TreapSet()
        for item in self:
            if item not in other:
                result.add(item)
        for item in other:
            if item not in self:
                result.add(item)
        return result

    def is_subset(self, other):
        for item in self:
            if item not in other:
                return False
        return True

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def clear(self):
        self._map.clear()

    def _verify(self):
        return self._map._verify()


# ============================================================
# ImplicitTreapNode -- for sequence operations
# ============================================================

class ImplicitTreapNode:
    """Node for implicit-key treap (array-like with O(log n) ops)."""
    __slots__ = ('value', 'priority', 'left', 'right', 'size', 'reversed')

    def __init__(self, value, priority=None):
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.size = 1
        self.reversed = False


def _isize(node):
    return node.size if node else 0


def _iupdate(node):
    if node:
        node.size = 1 + _isize(node.left) + _isize(node.right)
    return node


def _push_down(node):
    """Push lazy reverse flag down to children."""
    if node and node.reversed:
        node.left, node.right = node.right, node.left
        if node.left:
            node.left.reversed = not node.left.reversed
        if node.right:
            node.right.reversed = not node.right.reversed
        node.reversed = False


def isplit(node, pos):
    """Split implicit treap at position pos (0-indexed).
    Returns (left, right) where left has pos elements."""
    if node is None:
        return None, None
    _push_down(node)
    left_size = _isize(node.left)
    if pos <= left_size:
        left, node.left = isplit(node.left, pos)
        _iupdate(node)
        return left, node
    else:
        node.right, right = isplit(node.right, pos - left_size - 1)
        _iupdate(node)
        return node, right


def imerge(left, right):
    """Merge two implicit treaps."""
    if left is None:
        return right
    if right is None:
        return left
    _push_down(left)
    _push_down(right)
    if left.priority >= right.priority:
        left.right = imerge(left.right, right)
        _iupdate(left)
        return left
    else:
        right.left = imerge(left, right.left)
        _iupdate(right)
        return right


class ImplicitTreap:
    """Sequence supporting O(log n) insert, delete, reverse, and access by index."""

    def __init__(self, iterable=None):
        self._root = None
        if iterable:
            for item in iterable:
                self.append(item)

    def __len__(self):
        return _isize(self._root)

    def __bool__(self):
        return self._root is not None

    def append(self, value):
        """Append value to end."""
        node = ImplicitTreapNode(value)
        self._root = imerge(self._root, node)

    def insert(self, index, value):
        """Insert value at index."""
        if index < 0:
            index = max(0, len(self) + index)
        index = min(index, len(self))
        node = ImplicitTreapNode(value)
        left, right = isplit(self._root, index)
        self._root = imerge(imerge(left, node), right)

    def delete(self, index):
        """Delete element at index."""
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range")
        left, rest = isplit(self._root, index)
        _, right = isplit(rest, 1)
        self._root = imerge(left, right)

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range")
        return self._get(self._root, index)

    def _get(self, node, index):
        _push_down(node)
        left_size = _isize(node.left)
        if index < left_size:
            return self._get(node.left, index)
        elif index == left_size:
            return node.value
        else:
            return self._get(node.right, index - left_size - 1)

    def __setitem__(self, index, value):
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range")
        self._set(self._root, index, value)

    def _set(self, node, index, value):
        _push_down(node)
        left_size = _isize(node.left)
        if index < left_size:
            self._set(node.left, index, value)
        elif index == left_size:
            node.value = value
        else:
            self._set(node.right, index - left_size - 1, value)

    def reverse(self, l=None, r=None):
        """Reverse elements in range [l, r) (0-indexed). Defaults to full reverse."""
        n = len(self)
        if l is None:
            l = 0
        if r is None:
            r = n
        if l < 0:
            l += n
        if r < 0:
            r += n
        if l >= r or l >= n:
            return
        r = min(r, n)
        left, mid_right = isplit(self._root, l)
        mid, right = isplit(mid_right, r - l)
        if mid:
            mid.reversed = not mid.reversed
        self._root = imerge(imerge(left, mid), right)

    def to_list(self):
        result = []
        self._collect(self._root, result)
        return result

    def _collect(self, node, result):
        if node is None:
            return
        _push_down(node)
        self._collect(node.left, result)
        result.append(node.value)
        self._collect(node.right, result)

    def __iter__(self):
        return iter(self.to_list())

    def split_at(self, index):
        """Split into two ImplicitTreaps at index."""
        left, right = isplit(self._root, index)
        lt, rt = ImplicitTreap(), ImplicitTreap()
        lt._root = left
        rt._root = right
        return lt, rt

    def concat(self, other):
        """Concatenate other ImplicitTreap to end of self."""
        self._root = imerge(self._root, other._root)
        other._root = None


# ============================================================
# PersistentTreapNode -- path-copying for immutable versions
# ============================================================

class PersistentTreapNode:
    """Immutable treap node -- operations return new roots via path-copying."""
    __slots__ = ('key', 'value', 'priority', 'left', 'right', 'size')

    def __init__(self, key, value=None, priority=None, left=None, right=None):
        self.key = key
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = left
        self.right = right
        self.size = 1 + _psize(left) + _psize(right)


def _psize(node):
    return node.size if node else 0


def _pcopy(node, left=None, right=None):
    """Create a copy of node with optionally different children."""
    if node is None:
        return None
    l = left if left is not ... else node.left
    r = right if right is not ... else node.right
    return PersistentTreapNode(node.key, node.value, node.priority, l, r)


def psplit(node, key):
    """Persistent split -- returns new nodes, originals unchanged."""
    if node is None:
        return None, None
    if key <= node.key:
        left, new_right_child = psplit(node.left, key)
        new_node = PersistentTreapNode(node.key, node.value, node.priority, new_right_child, node.right)
        return left, new_node
    else:
        new_left_child, right = psplit(node.right, key)
        new_node = PersistentTreapNode(node.key, node.value, node.priority, node.left, new_left_child)
        return new_node, right


def pmerge(left, right):
    """Persistent merge -- returns new nodes, originals unchanged."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority >= right.priority:
        new_right = pmerge(left.right, right)
        return PersistentTreapNode(left.key, left.value, left.priority, left.left, new_right)
    else:
        new_left = pmerge(left, right.left)
        return PersistentTreapNode(right.key, right.value, right.priority, new_left, right.right)


class PersistentTreap:
    """Immutable ordered map with structural sharing."""

    def __init__(self, root=None):
        self._root = root

    def __len__(self):
        return _psize(self._root)

    def __bool__(self):
        return self._root is not None

    def __contains__(self, key):
        return self._find(self._root, key) is not None

    def _find(self, node, key):
        while node:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def get(self, key, default=None):
        node = self._find(self._root, key)
        return node.value if node else default

    def __getitem__(self, key):
        node = self._find(self._root, key)
        if node is None:
            raise KeyError(key)
        return node.value

    def put(self, key, value=None):
        """Return new PersistentTreap with key set to value."""
        # If key exists, we need to delete it first
        node = self._find(self._root, key)
        if node is not None:
            root = self._pdelete(self._root, key)
        else:
            root = self._root
        new_node = PersistentTreapNode(key, value)
        left, right = psplit(root, key)
        new_root = pmerge(pmerge(left, new_node), right)
        return PersistentTreap(new_root)

    def delete(self, key):
        """Return new PersistentTreap without key. Raises KeyError if not found."""
        if key not in self:
            raise KeyError(key)
        new_root = self._pdelete(self._root, key)
        return PersistentTreap(new_root)

    def _pdelete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            new_left = self._pdelete(node.left, key)
            return PersistentTreapNode(node.key, node.value, node.priority, new_left, node.right)
        elif key > node.key:
            new_right = self._pdelete(node.right, key)
            return PersistentTreapNode(node.key, node.value, node.priority, node.left, new_right)
        else:
            return pmerge(node.left, node.right)

    def min(self):
        if not self._root:
            raise ValueError("empty treap")
        node = self._root
        while node.left:
            node = node.left
        return (node.key, node.value)

    def max(self):
        if not self._root:
            raise ValueError("empty treap")
        node = self._root
        while node.right:
            node = node.right
        return (node.key, node.value)

    def __iter__(self):
        return self._inorder(self._root)

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield node.key
        yield from self._inorder(node.right)

    def items(self):
        return self._inorder_items(self._root)

    def _inorder_items(self, node):
        if node is None:
            return
        yield from self._inorder_items(node.left)
        yield (node.key, node.value)
        yield from self._inorder_items(node.right)

    def kth(self, k):
        if k < 0 or k >= len(self):
            raise IndexError(f"index {k} out of range")
        return self._kth(self._root, k)

    def _kth(self, node, k):
        left_size = _psize(node.left)
        if k < left_size:
            return self._kth(node.left, k)
        elif k == left_size:
            return (node.key, node.value)
        else:
            return self._kth(node.right, k - left_size - 1)

    def rank(self, key):
        r = 0
        node = self._root
        while node:
            if key <= node.key:
                node = node.left
            else:
                r += _psize(node.left) + 1
                node = node.right
        return r
