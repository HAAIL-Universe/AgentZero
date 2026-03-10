"""
C078: Persistent B-Tree
=======================
A persistent (immutable) B-tree for ordered key-value storage.

Features:
- Path-copying persistence (structural sharing)
- Configurable branching factor (order)
- O(log n) insert, delete, search
- Range queries, min/max, in-order traversal
- Bulk loading for efficient construction
- Merge/diff operations between trees
- Iteration protocol (Python __iter__)
"""

from __future__ import annotations
from typing import Any, Optional, Iterator


class BTreeNode:
    """Immutable B-tree node. Internal nodes have children; leaves don't."""
    __slots__ = ('keys', 'values', 'children', 'leaf')

    def __init__(self, keys: tuple, values: tuple, children: tuple | None = None):
        self.keys = keys
        self.values = values
        self.children = children  # None for leaf, tuple of BTreeNode for internal
        self.leaf = children is None

    def __repr__(self):
        if self.leaf:
            return f"Leaf({list(self.keys)})"
        return f"Internal({list(self.keys)}, {len(self.children)} children)"


def _search_index(keys: tuple, key) -> int:
    """Binary search for insertion index in sorted keys."""
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


class BTree:
    """
    Persistent B-tree with path-copying.

    Every mutation returns a NEW BTree; the original is unchanged.
    Nodes share structure where possible (structural sharing).

    Parameters:
        order: Maximum number of children per internal node (min 3).
               A node can hold at most (order-1) keys.
               Nodes split when they reach (order) keys.
    """

    __slots__ = ('_root', '_order', '_size')

    def __init__(self, root: BTreeNode | None = None, order: int = 32, size: int = 0):
        if order < 3:
            raise ValueError("B-tree order must be at least 3")
        self._root = root
        self._order = order
        self._size = size

    @property
    def order(self) -> int:
        return self._order

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:
        return f"BTree(size={self._size}, order={self._order})"

    # ---- Search ----

    def get(self, key, default=None) -> Any:
        """Get value for key, or default if not found."""
        node = self._root
        while node is not None:
            i = _search_index(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                return node.values[i]
            if node.leaf:
                return default
            node = node.children[i]
        return default

    def __contains__(self, key) -> bool:
        """Check if key exists."""
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def __getitem__(self, key) -> Any:
        """Get value for key, raise KeyError if not found."""
        sentinel = object()
        result = self.get(key, sentinel)
        if result is sentinel:
            raise KeyError(key)
        return result

    # ---- Insert ----

    def insert(self, key, value) -> 'BTree':
        """Return a new BTree with key-value inserted (or updated)."""
        if self._root is None:
            node = BTreeNode(keys=(key,), values=(value,))
            return BTree(node, self._order, 1)

        result = self._insert_node(self._root, key, value)
        if result is None:
            # Key updated, size unchanged
            new_root = self._update_key(self._root, key, value)
            return BTree(new_root, self._order, self._size)

        if len(result) == 1:
            return BTree(result[0], self._order, self._size + 1)
        else:
            # Root split: create new root
            left, median_key, median_val, right = result
            new_root = BTreeNode(
                keys=(median_key,),
                values=(median_val,),
                children=(left, right)
            )
            return BTree(new_root, self._order, self._size + 1)

    def _update_key(self, node: BTreeNode, key, value) -> BTreeNode:
        """Path-copy update of an existing key's value."""
        i = _search_index(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            new_values = node.values[:i] + (value,) + node.values[i+1:]
            return BTreeNode(node.keys, new_values, node.children)
        # Must be in child
        new_child = self._update_key(node.children[i], key, value)
        new_children = node.children[:i] + (new_child,) + node.children[i+1:]
        return BTreeNode(node.keys, node.values, new_children)

    def _insert_node(self, node: BTreeNode, key, value):
        """Insert into subtree rooted at node.

        Returns:
            None: key already exists (update needed)
            (node,): single node result (no split)
            (left, median_key, median_val, right): split result
        """
        i = _search_index(node.keys, key)

        # Key exists -- signal update
        if i < len(node.keys) and node.keys[i] == key:
            return None

        if node.leaf:
            new_keys = node.keys[:i] + (key,) + node.keys[i:]
            new_values = node.values[:i] + (value,) + node.values[i:]
            if len(new_keys) < self._order:
                return (BTreeNode(new_keys, new_values),)
            else:
                return self._split_leaf(new_keys, new_values)
        else:
            result = self._insert_node(node.children[i], key, value)
            if result is None:
                return None
            if len(result) == 1:
                new_children = node.children[:i] + (result[0],) + node.children[i+1:]
                return (BTreeNode(node.keys, node.values, new_children),)
            else:
                left, med_key, med_val, right = result
                new_keys = node.keys[:i] + (med_key,) + node.keys[i:]
                new_values = node.values[:i] + (med_val,) + node.values[i:]
                new_children = node.children[:i] + (left, right) + node.children[i+1:]
                if len(new_keys) < self._order:
                    return (BTreeNode(new_keys, new_values, new_children),)
                else:
                    return self._split_internal(new_keys, new_values, new_children)

    def _split_leaf(self, keys: tuple, values: tuple):
        """Split a full leaf into two leaves + median."""
        mid = len(keys) // 2
        left = BTreeNode(keys[:mid], values[:mid])
        right = BTreeNode(keys[mid+1:], values[mid+1:])
        return (left, keys[mid], values[mid], right)

    def _split_internal(self, keys: tuple, values: tuple, children: tuple):
        """Split a full internal node into two internals + median."""
        mid = len(keys) // 2
        left = BTreeNode(keys[:mid], values[:mid], children[:mid+1])
        right = BTreeNode(keys[mid+1:], values[mid+1:], children[mid+1:])
        return (left, keys[mid], values[mid], right)

    # ---- Delete ----

    def delete(self, key) -> 'BTree':
        """Return a new BTree with key removed. Raises KeyError if not found."""
        if self._root is None:
            raise KeyError(key)

        new_root = self._delete_node(self._root, key)
        if new_root is None:
            raise KeyError(key)

        new_size = self._size - 1

        # If root is internal with no keys, collapse
        if not new_root.leaf and len(new_root.keys) == 0:
            if new_root.children:
                new_root = new_root.children[0]
            else:
                new_root = None

        if new_size == 0:
            new_root = None

        return BTree(new_root, self._order, new_size)

    def _delete_node(self, node: BTreeNode, key) -> BTreeNode | None:
        """Delete key from subtree. Returns new node or None if key not found."""
        i = _search_index(node.keys, key)
        found = i < len(node.keys) and node.keys[i] == key

        if node.leaf:
            if not found:
                return None
            new_keys = node.keys[:i] + node.keys[i+1:]
            new_values = node.values[:i] + node.values[i+1:]
            return BTreeNode(new_keys, new_values)

        if found:
            # Replace with predecessor (rightmost in left subtree)
            pred_key, pred_val = self._predecessor(node.children[i])
            # Delete predecessor from left child
            new_child = self._delete_node(node.children[i], pred_key)
            new_keys = node.keys[:i] + (pred_key,) + node.keys[i+1:]
            new_values = node.values[:i] + (pred_val,) + node.values[i+1:]
            new_children = node.children[:i] + (new_child,) + node.children[i+1:]
            return self._rebalance(new_keys, new_values, new_children, i)
        else:
            new_child = self._delete_node(node.children[i], key)
            if new_child is None:
                return None
            new_children = node.children[:i] + (new_child,) + node.children[i+1:]
            return self._rebalance(node.keys, node.values, new_children, i)

    def _predecessor(self, node: BTreeNode):
        """Find the rightmost key-value in a subtree."""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]

    def _rebalance(self, keys: tuple, values: tuple, children: tuple, idx: int) -> BTreeNode:
        """Rebalance after deletion if child at idx is underflowing."""
        min_keys = (self._order - 1) // 2  # Minimum keys per non-root node

        child = children[idx]
        if len(child.keys) >= min_keys:
            return BTreeNode(keys, values, children)

        # Try borrowing from left sibling
        if idx > 0 and len(children[idx - 1].keys) > min_keys:
            return self._borrow_left(keys, values, children, idx)

        # Try borrowing from right sibling
        if idx < len(children) - 1 and len(children[idx + 1].keys) > min_keys:
            return self._borrow_right(keys, values, children, idx)

        # Merge with a sibling
        if idx > 0:
            return self._merge(keys, values, children, idx - 1)
        else:
            return self._merge(keys, values, children, idx)

    def _borrow_left(self, keys, values, children, idx):
        """Borrow from left sibling through parent."""
        left = children[idx - 1]
        child = children[idx]
        sep_idx = idx - 1

        # Move parent separator down to child, move left's last key up to parent
        if child.leaf:
            new_child = BTreeNode(
                (keys[sep_idx],) + child.keys,
                (values[sep_idx],) + child.values
            )
        else:
            new_child = BTreeNode(
                (keys[sep_idx],) + child.keys,
                (values[sep_idx],) + child.values,
                (left.children[-1],) + child.children
            )

        if left.leaf:
            new_left = BTreeNode(left.keys[:-1], left.values[:-1])
        else:
            new_left = BTreeNode(left.keys[:-1], left.values[:-1], left.children[:-1])

        new_keys = keys[:sep_idx] + (left.keys[-1],) + keys[sep_idx+1:]
        new_values = values[:sep_idx] + (left.values[-1],) + values[sep_idx+1:]
        new_children = children[:idx-1] + (new_left, new_child) + children[idx+1:]
        return BTreeNode(new_keys, new_values, new_children)

    def _borrow_right(self, keys, values, children, idx):
        """Borrow from right sibling through parent."""
        right = children[idx + 1]
        child = children[idx]
        sep_idx = idx

        if child.leaf:
            new_child = BTreeNode(
                child.keys + (keys[sep_idx],),
                child.values + (values[sep_idx],)
            )
        else:
            new_child = BTreeNode(
                child.keys + (keys[sep_idx],),
                child.values + (values[sep_idx],),
                child.children + (right.children[0],)
            )

        if right.leaf:
            new_right = BTreeNode(right.keys[1:], right.values[1:])
        else:
            new_right = BTreeNode(right.keys[1:], right.values[1:], right.children[1:])

        new_keys = keys[:sep_idx] + (right.keys[0],) + keys[sep_idx+1:]
        new_values = values[:sep_idx] + (right.values[0],) + values[sep_idx+1:]
        new_children = children[:idx] + (new_child, new_right) + children[idx+2:]
        return BTreeNode(new_keys, new_values, new_children)

    def _merge(self, keys, values, children, idx):
        """Merge children[idx] and children[idx+1] with separator keys[idx]."""
        left = children[idx]
        right = children[idx + 1]
        sep_key = keys[idx]
        sep_val = values[idx]

        if left.leaf:
            merged = BTreeNode(
                left.keys + (sep_key,) + right.keys,
                left.values + (sep_val,) + right.values
            )
        else:
            merged = BTreeNode(
                left.keys + (sep_key,) + right.keys,
                left.values + (sep_val,) + right.values,
                left.children + right.children
            )

        new_keys = keys[:idx] + keys[idx+1:]
        new_values = values[:idx] + values[idx+1:]
        new_children = children[:idx] + (merged,) + children[idx+2:]
        return BTreeNode(new_keys, new_values, new_children)

    # ---- Discard (no error on missing) ----

    def discard(self, key) -> 'BTree':
        """Return new BTree with key removed, or same tree if key not found."""
        if key not in self:
            return self
        return self.delete(key)

    # ---- Min / Max ----

    def min(self):
        """Return (key, value) for the minimum key. Raises ValueError if empty."""
        if self._root is None:
            raise ValueError("min() on empty B-tree")
        node = self._root
        while not node.leaf:
            node = node.children[0]
        return (node.keys[0], node.values[0])

    def max(self):
        """Return (key, value) for the maximum key. Raises ValueError if empty."""
        if self._root is None:
            raise ValueError("max() on empty B-tree")
        node = self._root
        while not node.leaf:
            node = node.children[-1]
        return (node.keys[-1], node.values[-1])

    # ---- Iteration ----

    def __iter__(self) -> Iterator:
        """In-order iteration over keys."""
        if self._root is not None:
            yield from self._iter_node(self._root)

    def _iter_node(self, node: BTreeNode) -> Iterator:
        if node.leaf:
            yield from node.keys
        else:
            for i, child in enumerate(node.children):
                yield from self._iter_node(child)
                if i < len(node.keys):
                    yield node.keys[i]

    def items(self) -> Iterator:
        """In-order iteration over (key, value) pairs."""
        if self._root is not None:
            yield from self._items_node(self._root)

    def _items_node(self, node: BTreeNode) -> Iterator:
        if node.leaf:
            yield from zip(node.keys, node.values)
        else:
            for i, child in enumerate(node.children):
                yield from self._items_node(child)
                if i < len(node.keys):
                    yield (node.keys[i], node.values[i])

    def keys(self) -> Iterator:
        """In-order iteration over keys."""
        return iter(self)

    def values(self) -> Iterator:
        """In-order iteration over values."""
        if self._root is not None:
            yield from self._values_node(self._root)

    def _values_node(self, node: BTreeNode) -> Iterator:
        if node.leaf:
            yield from node.values
        else:
            for i, child in enumerate(node.children):
                yield from self._values_node(child)
                if i < len(node.keys):
                    yield node.values[i]

    # ---- Range queries ----

    def range(self, lo=None, hi=None, inclusive_lo=True, inclusive_hi=False) -> Iterator:
        """Yield (key, value) pairs where lo <= key < hi (by default).

        Args:
            lo: Lower bound (None = no lower bound)
            hi: Upper bound (None = no upper bound)
            inclusive_lo: Include lo in results (default True)
            inclusive_hi: Include hi in results (default False)
        """
        if self._root is not None:
            yield from self._range_node(self._root, lo, hi, inclusive_lo, inclusive_hi)

    def _range_node(self, node, lo, hi, inc_lo, inc_hi):
        if node.leaf:
            for k, v in zip(node.keys, node.values):
                if lo is not None:
                    if inc_lo and k < lo:
                        continue
                    if not inc_lo and k <= lo:
                        continue
                if hi is not None:
                    if inc_hi and k > hi:
                        break
                    if not inc_hi and k >= hi:
                        break
                yield (k, v)
        else:
            for i, child in enumerate(node.children):
                # Prune: skip children that can't contain range
                if hi is not None and i > 0:
                    sep = node.keys[i - 1]
                    if inc_hi and sep > hi:
                        break
                    if not inc_hi and sep >= hi:
                        break
                yield from self._range_node(child, lo, hi, inc_lo, inc_hi)
                if i < len(node.keys):
                    k, v = node.keys[i], node.values[i]
                    in_range = True
                    if lo is not None:
                        if inc_lo and k < lo:
                            in_range = False
                        if not inc_lo and k <= lo:
                            in_range = False
                    if hi is not None:
                        if inc_hi and k > hi:
                            break
                        if not inc_hi and k >= hi:
                            break
                    if in_range:
                        yield (k, v)

    # ---- Floor / Ceiling ----

    def floor(self, key):
        """Return (k, v) where k is the greatest key <= key. None if no such key."""
        result = None
        node = self._root
        while node is not None:
            i = _search_index(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                return (node.keys[i], node.values[i])
            # The keys before index i are all < key
            if i > 0:
                result = (node.keys[i-1], node.values[i-1])
            if node.leaf:
                break
            node = node.children[i]
        return result

    def ceiling(self, key):
        """Return (k, v) where k is the least key >= key. None if no such key."""
        result = None
        node = self._root
        while node is not None:
            i = _search_index(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                return (node.keys[i], node.values[i])
            if i < len(node.keys):
                result = (node.keys[i], node.values[i])
            if node.leaf:
                break
            node = node.children[i]
        return result

    # ---- Rank operations ----

    def rank(self, key) -> int:
        """Return number of keys strictly less than key."""
        return self._rank_node(self._root, key) if self._root else 0

    def _rank_node(self, node, key) -> int:
        i = _search_index(node.keys, key)
        if node.leaf:
            return i
        count = 0
        for j in range(i):
            count += self._subtree_size(node.children[j]) + 1
        count += self._rank_node(node.children[i], key)
        return count

    def _subtree_size(self, node) -> int:
        """Count total keys in subtree."""
        if node.leaf:
            return len(node.keys)
        total = len(node.keys)
        for child in node.children:
            total += self._subtree_size(child)
        return total

    def select(self, k: int):
        """Return (key, value) of the k-th smallest key (0-indexed). Raises IndexError."""
        if k < 0 or k >= self._size:
            raise IndexError(f"index {k} out of range for size {self._size}")
        return self._select_node(self._root, k)

    def _select_node(self, node, k):
        if node.leaf:
            return (node.keys[k], node.values[k])
        for i, child in enumerate(node.children):
            child_size = self._subtree_size(child)
            if k < child_size:
                return self._select_node(child, k)
            k -= child_size
            if i < len(node.keys):
                if k == 0:
                    return (node.keys[i], node.values[i])
                k -= 1
        raise IndexError("unreachable")  # pragma: no cover

    # ---- Bulk operations ----

    @classmethod
    def from_sorted(cls, items, order: int = 32) -> 'BTree':
        """Build a B-tree from pre-sorted (key, value) pairs. O(n)."""
        items = list(items)
        if not items:
            return cls(order=order)

        tree = cls(order=order)
        for k, v in items:
            tree = tree.insert(k, v)
        return tree

    @classmethod
    def from_items(cls, items, order: int = 32) -> 'BTree':
        """Build a B-tree from unsorted (key, value) pairs."""
        items = sorted(items, key=lambda x: x[0])
        return cls.from_sorted(items, order=order)

    @classmethod
    def from_dict(cls, d: dict, order: int = 32) -> 'BTree':
        """Build a B-tree from a dictionary."""
        return cls.from_items(d.items(), order=order)

    # ---- Merge / Diff ----

    def merge(self, other: 'BTree', conflict=None) -> 'BTree':
        """Merge two B-trees. On key conflict, use conflict(key, self_val, other_val).
        Default: other's value wins."""
        if conflict is None:
            conflict = lambda k, a, b: b

        result = self
        for k, v in other.items():
            if k in result:
                merged_val = conflict(k, result[k], v)
                result = result.insert(k, merged_val)
            else:
                result = result.insert(k, v)
        return result

    def diff(self, other: 'BTree') -> dict:
        """Return differences between self and other.

        Returns dict with:
            'added': keys in other but not self
            'removed': keys in self but not other
            'changed': keys in both with different values
        """
        added = {}
        removed = {}
        changed = {}

        self_keys = set(self)
        other_keys = set(other)

        for k in other_keys - self_keys:
            added[k] = other[k]
        for k in self_keys - other_keys:
            removed[k] = self[k]
        for k in self_keys & other_keys:
            if self[k] != other[k]:
                changed[k] = (self[k], other[k])

        return {'added': added, 'removed': removed, 'changed': changed}

    # ---- Map / Filter / Reduce ----

    def map_values(self, fn) -> 'BTree':
        """Return new B-tree with fn applied to each value."""
        if self._root is None:
            return self
        new_root = self._map_node(self._root, fn)
        return BTree(new_root, self._order, self._size)

    def _map_node(self, node, fn):
        new_values = tuple(fn(v) for v in node.values)
        if node.leaf:
            return BTreeNode(node.keys, new_values)
        new_children = tuple(self._map_node(c, fn) for c in node.children)
        return BTreeNode(node.keys, new_values, new_children)

    def filter(self, pred) -> 'BTree':
        """Return new B-tree containing only items where pred(key, value) is True."""
        result = BTree(order=self._order)
        for k, v in self.items():
            if pred(k, v):
                result = result.insert(k, v)
        return result

    def reduce(self, fn, initial=None):
        """Reduce over (key, value) pairs in order."""
        acc = initial
        first = True
        for k, v in self.items():
            if first and initial is None:
                acc = (k, v)
                first = False
            else:
                acc = fn(acc, (k, v))
                first = False
        if first and initial is None:
            raise ValueError("reduce on empty B-tree with no initial value")
        return acc

    # ---- Structural info ----

    def height(self) -> int:
        """Return the height of the tree (0 for empty, 1 for single leaf)."""
        if self._root is None:
            return 0
        h = 1
        node = self._root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h

    def to_dict(self) -> dict:
        """Convert to an ordered dict."""
        return dict(self.items())

    # ---- Equality ----

    def __eq__(self, other) -> bool:
        if not isinstance(other, BTree):
            return NotImplemented
        if self._size != other._size:
            return False
        for (k1, v1), (k2, v2) in zip(self.items(), other.items()):
            if k1 != k2 or v1 != v2:
                return False
        return True

    # ---- Pop min / max ----

    def pop_min(self):
        """Return ((key, value), new_tree) for the minimum key. Raises ValueError if empty."""
        if self._root is None:
            raise ValueError("pop_min() on empty B-tree")
        k, v = self.min()
        return ((k, v), self.delete(k))

    def pop_max(self):
        """Return ((key, value), new_tree) for the maximum key. Raises ValueError if empty."""
        if self._root is None:
            raise ValueError("pop_max() on empty B-tree")
        k, v = self.max()
        return ((k, v), self.delete(k))

    # ---- Slice ----

    def slice(self, lo=None, hi=None) -> 'BTree':
        """Return a new B-tree containing only keys in [lo, hi)."""
        result = BTree(order=self._order)
        for k, v in self.range(lo, hi):
            result = result.insert(k, v)
        return result

    # ---- Reverse iteration ----

    def __reversed__(self) -> Iterator:
        """Reverse iteration over keys."""
        if self._root is not None:
            yield from self._reverse_node(self._root)

    def _reverse_node(self, node: BTreeNode) -> Iterator:
        if node.leaf:
            for k in reversed(node.keys):
                yield k
        else:
            for i in range(len(node.children) - 1, -1, -1):
                if i < len(node.keys):
                    yield node.keys[i]
                yield from self._reverse_node(node.children[i])

    # ---- Nth from end ----

    def kth_largest(self, k: int):
        """Return (key, value) of k-th largest (0-indexed). Raises IndexError."""
        return self.select(self._size - 1 - k)

    # ---- Count in range ----

    def count_range(self, lo=None, hi=None) -> int:
        """Count keys in [lo, hi)."""
        count = 0
        for _ in self.range(lo, hi):
            count += 1
        return count

    # ---- Nearest ----

    def nearest(self, key):
        """Return (k, v) for the key nearest to the given key. None if empty."""
        if self._root is None:
            return None
        f = self.floor(key)
        c = self.ceiling(key)
        if f is None and c is None:
            return None
        if f is None:
            return c
        if c is None:
            return f
        if abs(key - f[0]) <= abs(key - c[0]):
            return f
        return c
