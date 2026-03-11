"""
C116: B+ Tree
=============
A mutable B+ tree -- the classic database index structure.

Key difference from B-tree: ALL data lives in leaf nodes. Internal nodes
are pure index (separator keys only). Leaves form a doubly-linked list
for O(1) sequential access after any point lookup.

Components:
- BPlusTree: Core mutable B+ tree with configurable order
- BPlusTreeMap: Full ordered map interface (get/set/del/iter/range)
- BPlusTreeSet: Ordered set built on BPlusTreeMap
- BulkLoader: Efficient bottom-up construction from sorted data
- BPlusTreeIterator: Bidirectional iterator via leaf chain

Features:
- O(log_b n) search, insert, delete
- O(k + log n) range queries via leaf chain
- Bulk loading in O(n) from sorted input
- Min/max in O(log n) via leftmost/rightmost leaf
- Floor/ceiling queries
- Forward and reverse iteration
- Merge and difference operations
"""

from __future__ import annotations
from typing import Any, Optional, Iterator


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class LeafNode:
    """Leaf node: holds actual key-value pairs + sibling links."""
    __slots__ = ('keys', 'values', 'next_leaf', 'prev_leaf', 'parent')

    def __init__(self):
        self.keys: list = []
        self.values: list = []
        self.next_leaf: Optional[LeafNode] = None
        self.prev_leaf: Optional[LeafNode] = None
        self.parent: Optional[InternalNode] = None

    @property
    def is_leaf(self) -> bool:
        return True

    def __repr__(self):
        return f"Leaf({self.keys})"


class InternalNode:
    """Internal node: holds separator keys + child pointers. No values."""
    __slots__ = ('keys', 'children', 'parent')

    def __init__(self):
        self.keys: list = []
        self.children: list = []  # len(children) == len(keys) + 1
        self.parent: Optional[InternalNode] = None

    @property
    def is_leaf(self) -> bool:
        return False

    def __repr__(self):
        return f"Internal({self.keys}, {len(self.children)} children)"


# ---------------------------------------------------------------------------
# Binary search helper
# ---------------------------------------------------------------------------

def _bisect_right(keys: list, key) -> int:
    """Find rightmost insertion point for key in sorted list."""
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_left(keys: list, key) -> int:
    """Find leftmost insertion point for key in sorted list."""
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# BPlusTree core
# ---------------------------------------------------------------------------

class BPlusTree:
    """
    Mutable B+ tree.

    Parameters:
        order: Maximum number of children per internal node (min 3).
               Leaves hold at most (order - 1) key-value pairs.
               Internal nodes hold at most (order - 1) separator keys.
    """

    def __init__(self, order: int = 32):
        if order < 3:
            raise ValueError("B+ tree order must be at least 3")
        self._order = order
        self._root: LeafNode | InternalNode = LeafNode()
        self._size = 0
        # Track head/tail of leaf chain for O(1) min/max access
        self._head: LeafNode = self._root  # leftmost leaf
        self._tail: LeafNode = self._root  # rightmost leaf

    @property
    def order(self) -> int:
        return self._order

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:
        return f"BPlusTree(order={self._order}, size={self._size})"

    # -- Search --

    def _find_leaf(self, key) -> LeafNode:
        """Navigate from root to the leaf that should contain key."""
        node = self._root
        while not node.is_leaf:
            idx = _bisect_right(node.keys, key)
            node = node.children[idx]
        return node

    def get(self, key, default=None):
        """Get value for key, or default if not found."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)
        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            return leaf.values[idx]
        return default

    def __getitem__(self, key):
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)
        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            return leaf.values[idx]
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)
        return idx < len(leaf.keys) and leaf.keys[idx] == key

    # -- Insert --

    def insert(self, key, value) -> None:
        """Insert or update key-value pair."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)

        # Update existing
        if idx < len(leaf.keys) and leaf.keys[idx] == key:
            leaf.values[idx] = value
            return

        # Insert new
        leaf.keys.insert(idx, key)
        leaf.values.insert(idx, value)
        self._size += 1

        # Split if overflow
        if len(leaf.keys) >= self._order:
            self._split_leaf(leaf)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def _split_leaf(self, leaf: LeafNode) -> None:
        """Split a full leaf node."""
        mid = len(leaf.keys) // 2
        new_leaf = LeafNode()

        # Move upper half to new leaf
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        # Update linked list
        new_leaf.next_leaf = leaf.next_leaf
        new_leaf.prev_leaf = leaf
        if leaf.next_leaf:
            leaf.next_leaf.prev_leaf = new_leaf
        leaf.next_leaf = new_leaf

        # Update tail if needed
        if self._tail is leaf:
            self._tail = new_leaf

        # Promote first key of new leaf to parent
        promote_key = new_leaf.keys[0]
        self._insert_into_parent(leaf, promote_key, new_leaf)

    def _split_internal(self, node: InternalNode) -> None:
        """Split a full internal node."""
        mid = len(node.keys) // 2
        promote_key = node.keys[mid]

        new_node = InternalNode()
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        # Update parent pointers
        for child in new_node.children:
            child.parent = new_node

        self._insert_into_parent(node, promote_key, new_node)

    def _insert_into_parent(self, left, key, right) -> None:
        """Insert a key and right child into the parent of left."""
        if left is self._root:
            # Create new root
            new_root = InternalNode()
            new_root.keys = [key]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self._root = new_root
            return

        parent = left.parent
        idx = _bisect_right(parent.keys, key)
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right)
        right.parent = parent

        if len(parent.keys) >= self._order:
            self._split_internal(parent)

    # -- Delete --

    def delete(self, key) -> bool:
        """Delete key. Returns True if found, False otherwise."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)

        if idx >= len(leaf.keys) or leaf.keys[idx] != key:
            return False

        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        self._size -= 1

        # If root is a leaf, no underflow to fix
        if leaf is self._root:
            return True

        # Check underflow: min keys = ceil(order/2) - 1
        min_keys = (self._order + 1) // 2 - 1
        if len(leaf.keys) < min_keys:
            self._fix_leaf_underflow(leaf)

        return True

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    def pop(self, key, *args):
        """Remove key and return its value. Raises KeyError if not found and no default."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)

        if idx >= len(leaf.keys) or leaf.keys[idx] != key:
            if args:
                return args[0]
            raise KeyError(key)

        value = leaf.values[idx]
        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        self._size -= 1

        if leaf is not self._root:
            min_keys = (self._order + 1) // 2 - 1
            if len(leaf.keys) < min_keys:
                self._fix_leaf_underflow(leaf)

        return value

    def _fix_leaf_underflow(self, leaf: LeafNode) -> None:
        """Fix underflow in a leaf by borrowing or merging."""
        parent = leaf.parent
        idx = parent.children.index(leaf)

        # Try borrow from right sibling
        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            min_keys = (self._order + 1) // 2 - 1
            if len(right.keys) > min_keys:
                # Borrow first key from right
                leaf.keys.append(right.keys.pop(0))
                leaf.values.append(right.values.pop(0))
                parent.keys[idx] = right.keys[0]
                return

        # Try borrow from left sibling
        if idx > 0:
            left = parent.children[idx - 1]
            min_keys = (self._order + 1) // 2 - 1
            if len(left.keys) > min_keys:
                # Borrow last key from left
                leaf.keys.insert(0, left.keys.pop())
                leaf.values.insert(0, left.values.pop())
                parent.keys[idx - 1] = leaf.keys[0]
                return

        # Merge: prefer merging with right sibling
        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            # Merge right into leaf
            leaf.keys.extend(right.keys)
            leaf.values.extend(right.values)
            leaf.next_leaf = right.next_leaf
            if right.next_leaf:
                right.next_leaf.prev_leaf = leaf
            if self._tail is right:
                self._tail = leaf
            # Remove separator and right child from parent
            parent.keys.pop(idx)
            parent.children.pop(idx + 1)
        else:
            # Merge leaf into left sibling
            left = parent.children[idx - 1]
            left.keys.extend(leaf.keys)
            left.values.extend(leaf.values)
            left.next_leaf = leaf.next_leaf
            if leaf.next_leaf:
                leaf.next_leaf.prev_leaf = left
            if self._tail is leaf:
                self._tail = left
            parent.keys.pop(idx - 1)
            parent.children.pop(idx)

        # Fix parent underflow
        if parent is self._root:
            if not parent.keys:
                self._root = parent.children[0]
                self._root.parent = None
        else:
            min_keys = (self._order + 1) // 2 - 1
            if len(parent.keys) < min_keys:
                self._fix_internal_underflow(parent)

    def _fix_internal_underflow(self, node: InternalNode) -> None:
        """Fix underflow in an internal node by borrowing or merging."""
        parent = node.parent
        idx = parent.children.index(node)

        # Try borrow from right sibling
        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            min_keys = (self._order + 1) // 2 - 1
            if len(right.keys) > min_keys:
                # Pull separator down, push right's first key up
                node.keys.append(parent.keys[idx])
                node.children.append(right.children.pop(0))
                node.children[-1].parent = node
                parent.keys[idx] = right.keys.pop(0)
                return

        # Try borrow from left sibling
        if idx > 0:
            left = parent.children[idx - 1]
            min_keys = (self._order + 1) // 2 - 1
            if len(left.keys) > min_keys:
                node.keys.insert(0, parent.keys[idx - 1])
                node.children.insert(0, left.children.pop())
                node.children[0].parent = node
                parent.keys[idx - 1] = left.keys.pop()
                return

        # Merge
        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            node.keys.append(parent.keys.pop(idx))
            node.keys.extend(right.keys)
            for child in right.children:
                child.parent = node
            node.children.extend(right.children)
            parent.children.pop(idx + 1)
        else:
            left = parent.children[idx - 1]
            left.keys.append(parent.keys.pop(idx - 1))
            left.keys.extend(node.keys)
            for child in node.children:
                child.parent = left
            left.children.extend(node.children)
            parent.children.pop(idx)

        if parent is self._root and not parent.keys:
            self._root = parent.children[0]
            self._root.parent = None
        elif parent is not self._root:
            min_keys = (self._order + 1) // 2 - 1
            if len(parent.keys) < min_keys:
                self._fix_internal_underflow(parent)

    # -- Queries --

    def min_key(self):
        """Return the smallest key, or None if empty."""
        if self._size == 0:
            return None
        return self._head.keys[0]

    def max_key(self):
        """Return the largest key, or None if empty."""
        if self._size == 0:
            return None
        return self._tail.keys[-1]

    def min_item(self):
        """Return (key, value) for smallest key."""
        if self._size == 0:
            return None
        return (self._head.keys[0], self._head.values[0])

    def max_item(self):
        """Return (key, value) for largest key."""
        if self._size == 0:
            return None
        return (self._tail.keys[-1], self._tail.values[-1])

    def floor(self, key):
        """Return largest key <= given key, or None."""
        leaf = self._find_leaf(key)
        idx = _bisect_right(leaf.keys, key) - 1
        if idx >= 0:
            return leaf.keys[idx]
        # Check previous leaf
        if leaf.prev_leaf and leaf.prev_leaf.keys:
            return leaf.prev_leaf.keys[-1]
        return None

    def ceiling(self, key):
        """Return smallest key >= given key, or None."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)
        if idx < len(leaf.keys):
            return leaf.keys[idx]
        # Check next leaf
        if leaf.next_leaf and leaf.next_leaf.keys:
            return leaf.next_leaf.keys[0]
        return None

    def floor_item(self, key):
        """Return (k, v) for largest k <= key, or None."""
        leaf = self._find_leaf(key)
        idx = _bisect_right(leaf.keys, key) - 1
        if idx >= 0:
            return (leaf.keys[idx], leaf.values[idx])
        if leaf.prev_leaf and leaf.prev_leaf.keys:
            return (leaf.prev_leaf.keys[-1], leaf.prev_leaf.values[-1])
        return None

    def ceiling_item(self, key):
        """Return (k, v) for smallest k >= key, or None."""
        leaf = self._find_leaf(key)
        idx = _bisect_left(leaf.keys, key)
        if idx < len(leaf.keys):
            return (leaf.keys[idx], leaf.values[idx])
        if leaf.next_leaf and leaf.next_leaf.keys:
            return (leaf.next_leaf.keys[0], leaf.next_leaf.values[0])
        return None

    # -- Range queries (via leaf chain) --

    def range_query(self, low=None, high=None, include_low=True, include_high=True) -> list:
        """
        Return list of (key, value) pairs in [low, high] range.
        None means unbounded on that side.
        """
        result = []

        if low is None:
            leaf = self._head
            idx = 0
        else:
            leaf = self._find_leaf(low)
            if include_low:
                idx = _bisect_left(leaf.keys, low)
            else:
                idx = _bisect_right(leaf.keys, low)

        while leaf is not None:
            while idx < len(leaf.keys):
                k = leaf.keys[idx]
                if high is not None:
                    if include_high:
                        if k > high:
                            return result
                    else:
                        if k >= high:
                            return result
                result.append((k, leaf.values[idx]))
                idx += 1
            leaf = leaf.next_leaf
            idx = 0

        return result

    def range_keys(self, low=None, high=None, include_low=True, include_high=True) -> list:
        """Return keys in range."""
        return [k for k, v in self.range_query(low, high, include_low, include_high)]

    def range_values(self, low=None, high=None, include_low=True, include_high=True) -> list:
        """Return values in range."""
        return [v for k, v in self.range_query(low, high, include_low, include_high)]

    def count_range(self, low=None, high=None, include_low=True, include_high=True) -> int:
        """Count keys in range without collecting results."""
        count = 0

        if low is None:
            leaf = self._head
            idx = 0
        else:
            leaf = self._find_leaf(low)
            if include_low:
                idx = _bisect_left(leaf.keys, low)
            else:
                idx = _bisect_right(leaf.keys, low)

        while leaf is not None:
            while idx < len(leaf.keys):
                k = leaf.keys[idx]
                if high is not None:
                    if include_high:
                        if k > high:
                            return count
                    else:
                        if k >= high:
                            return count
                count += 1
                idx += 1
            leaf = leaf.next_leaf
            idx = 0

        return count

    # -- Iteration --

    def __iter__(self) -> Iterator:
        """Iterate all keys in order via leaf chain."""
        leaf = self._head
        while leaf is not None:
            yield from leaf.keys
            leaf = leaf.next_leaf

    def items(self) -> Iterator:
        """Iterate (key, value) pairs in order."""
        leaf = self._head
        while leaf is not None:
            yield from zip(leaf.keys, leaf.values)
            leaf = leaf.next_leaf

    def values(self) -> Iterator:
        """Iterate values in key order."""
        leaf = self._head
        while leaf is not None:
            yield from leaf.values
            leaf = leaf.next_leaf

    def keys(self) -> Iterator:
        """Iterate keys in order."""
        return iter(self)

    def reversed_keys(self) -> Iterator:
        """Iterate keys in reverse order via leaf chain."""
        leaf = self._tail
        while leaf is not None:
            for i in range(len(leaf.keys) - 1, -1, -1):
                yield leaf.keys[i]
            leaf = leaf.prev_leaf

    def reversed_items(self) -> Iterator:
        """Iterate (key, value) pairs in reverse order."""
        leaf = self._tail
        while leaf is not None:
            for i in range(len(leaf.keys) - 1, -1, -1):
                yield (leaf.keys[i], leaf.values[i])
            leaf = leaf.prev_leaf

    # -- Rank operations --

    def rank(self, key) -> int:
        """Return number of keys strictly less than given key."""
        count = 0
        leaf = self._head
        while leaf is not None:
            idx = _bisect_left(leaf.keys, key)
            count += idx
            if idx < len(leaf.keys):
                return count
            leaf = leaf.next_leaf
        return count

    def select(self, k: int):
        """Return the k-th smallest key (0-indexed). Raises IndexError if out of range."""
        if k < 0:
            k += self._size
        if k < 0 or k >= self._size:
            raise IndexError(f"index {k} out of range for tree of size {self._size}")
        leaf = self._head
        while leaf is not None:
            if k < len(leaf.keys):
                return leaf.keys[k]
            k -= len(leaf.keys)
            leaf = leaf.next_leaf
        raise IndexError("should not reach here")

    # -- Bulk operations --

    def update(self, iterable) -> None:
        """Insert multiple key-value pairs from iterable of (key, value)."""
        for k, v in iterable:
            self.insert(k, v)

    def clear(self) -> None:
        """Remove all entries."""
        self._root = LeafNode()
        self._head = self._root
        self._tail = self._root
        self._size = 0

    # -- Height / stats --

    def height(self) -> int:
        """Return tree height (0 for empty, 1 for root-only)."""
        if self._size == 0:
            return 0
        h = 1
        node = self._root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def leaf_count(self) -> int:
        """Count leaf nodes by walking the chain."""
        count = 0
        leaf = self._head
        while leaf is not None:
            count += 1
            leaf = leaf.next_leaf
        return count

    def _verify(self) -> bool:
        """Verify B+ tree invariants. Returns True if valid."""
        if self._size == 0:
            return True

        # Check leaf chain covers all entries
        count = 0
        prev = None
        leaf = self._head
        last_key = None
        while leaf is not None:
            if len(leaf.keys) != len(leaf.values):
                raise AssertionError("key/value length mismatch")
            for i, k in enumerate(leaf.keys):
                if last_key is not None and k <= last_key:
                    raise AssertionError(f"leaf keys not sorted: {last_key} >= {k}")
                last_key = k
                count += 1
            if leaf.prev_leaf is not prev:
                raise AssertionError("prev_leaf link broken")
            prev = leaf
            leaf = leaf.next_leaf

        if count != self._size:
            raise AssertionError(f"leaf chain has {count} entries but size is {self._size}")

        if prev is not self._tail:
            raise AssertionError("tail pointer incorrect")

        # Check internal node key constraints
        self._verify_node(self._root, None, None)
        return True

    def _verify_node(self, node, low, high) -> int:
        """Verify subtree rooted at node. Returns height."""
        if node.is_leaf:
            for k in node.keys:
                if low is not None and k < low:
                    raise AssertionError(f"leaf key {k} < lower bound {low}")
                if high is not None and k >= high:
                    raise AssertionError(f"leaf key {k} >= upper bound {high}")
            return 1

        # Internal node
        if len(node.children) != len(node.keys) + 1:
            raise AssertionError(f"internal node: {len(node.children)} children but {len(node.keys)} keys")

        heights = set()
        for i, child in enumerate(node.children):
            if child.parent is not node:
                raise AssertionError("parent pointer mismatch")
            child_low = node.keys[i - 1] if i > 0 else low
            child_high = node.keys[i] if i < len(node.keys) else high
            h = self._verify_node(child, child_low, child_high)
            heights.add(h)

        if len(heights) != 1:
            raise AssertionError(f"unbalanced: heights = {heights}")

        return heights.pop() + 1


# ---------------------------------------------------------------------------
# BPlusTreeMap -- dict-like interface
# ---------------------------------------------------------------------------

class BPlusTreeMap:
    """
    Ordered map backed by a B+ tree.

    Supports dict-like interface: map[key] = value, del map[key], key in map.
    Also supports ordered operations: min, max, floor, ceiling, range queries.
    """

    def __init__(self, order: int = 32, items=None):
        self._tree = BPlusTree(order=order)
        if items:
            if isinstance(items, dict):
                items = items.items()
            for k, v in items:
                self._tree.insert(k, v)

    def __len__(self):
        return len(self._tree)

    def __bool__(self):
        return bool(self._tree)

    def __repr__(self):
        pairs = list(self._tree.items())
        if len(pairs) > 5:
            shown = ', '.join(f'{k}: {v}' for k, v in pairs[:5])
            return f"BPlusTreeMap({{{shown}, ...}}, size={len(self._tree)})"
        shown = ', '.join(f'{k}: {v}' for k, v in pairs)
        return f"BPlusTreeMap({{{shown}}})"

    def __getitem__(self, key):
        return self._tree[key]

    def __setitem__(self, key, value):
        self._tree[key] = value

    def __delitem__(self, key):
        del self._tree[key]

    def __contains__(self, key):
        return key in self._tree

    def __iter__(self):
        return iter(self._tree)

    def get(self, key, default=None):
        return self._tree.get(key, default)

    def pop(self, key, *args):
        return self._tree.pop(key, *args)

    def items(self):
        return self._tree.items()

    def keys(self):
        return self._tree.keys()

    def values(self):
        return self._tree.values()

    def update(self, iterable):
        if isinstance(iterable, dict):
            iterable = iterable.items()
        self._tree.update(iterable)

    def clear(self):
        self._tree.clear()

    def min_key(self):
        return self._tree.min_key()

    def max_key(self):
        return self._tree.max_key()

    def min_item(self):
        return self._tree.min_item()

    def max_item(self):
        return self._tree.max_item()

    def floor(self, key):
        return self._tree.floor(key)

    def ceiling(self, key):
        return self._tree.ceiling(key)

    def floor_item(self, key):
        return self._tree.floor_item(key)

    def ceiling_item(self, key):
        return self._tree.ceiling_item(key)

    def range_query(self, low=None, high=None, include_low=True, include_high=True):
        return self._tree.range_query(low, high, include_low, include_high)

    def range_keys(self, low=None, high=None, include_low=True, include_high=True):
        return self._tree.range_keys(low, high, include_low, include_high)

    def count_range(self, low=None, high=None, include_low=True, include_high=True):
        return self._tree.count_range(low, high, include_low, include_high)

    def rank(self, key):
        return self._tree.rank(key)

    def select(self, k):
        return self._tree.select(k)

    def reversed_keys(self):
        return self._tree.reversed_keys()

    def reversed_items(self):
        return self._tree.reversed_items()

    def height(self):
        return self._tree.height()


# ---------------------------------------------------------------------------
# BPlusTreeSet -- ordered set
# ---------------------------------------------------------------------------

class BPlusTreeSet:
    """Ordered set backed by a B+ tree."""

    _SENTINEL = object()

    def __init__(self, order: int = 32, items=None):
        self._tree = BPlusTree(order=order)
        if items:
            for item in items:
                self._tree.insert(item, self._SENTINEL)

    def __len__(self):
        return len(self._tree)

    def __bool__(self):
        return bool(self._tree)

    def __repr__(self):
        elems = list(self._tree)
        if len(elems) > 8:
            shown = ', '.join(str(e) for e in elems[:8])
            return f"BPlusTreeSet({{{shown}, ...}}, size={len(self._tree)})"
        shown = ', '.join(str(e) for e in elems)
        return f"BPlusTreeSet({{{shown}}})"

    def __contains__(self, item):
        return item in self._tree

    def __iter__(self):
        return iter(self._tree)

    def add(self, item):
        self._tree.insert(item, self._SENTINEL)

    def discard(self, item):
        self._tree.delete(item)

    def remove(self, item):
        if not self._tree.delete(item):
            raise KeyError(item)

    def pop_min(self):
        """Remove and return the smallest element."""
        k = self._tree.min_key()
        if k is None:
            raise KeyError("pop from empty set")
        self._tree.delete(k)
        return k

    def pop_max(self):
        """Remove and return the largest element."""
        k = self._tree.max_key()
        if k is None:
            raise KeyError("pop from empty set")
        self._tree.delete(k)
        return k

    def min(self):
        return self._tree.min_key()

    def max(self):
        return self._tree.max_key()

    def floor(self, item):
        return self._tree.floor(item)

    def ceiling(self, item):
        return self._tree.ceiling(item)

    def range(self, low=None, high=None, include_low=True, include_high=True):
        """Return list of elements in range."""
        return self._tree.range_keys(low, high, include_low, include_high)

    def count_range(self, low=None, high=None, include_low=True, include_high=True):
        return self._tree.count_range(low, high, include_low, include_high)

    def rank(self, item):
        return self._tree.rank(item)

    def select(self, k):
        return self._tree.select(k)

    def reversed(self):
        return self._tree.reversed_keys()

    def union(self, other: BPlusTreeSet) -> BPlusTreeSet:
        """Return new set with elements from both sets."""
        result = BPlusTreeSet(order=self._tree.order)
        for item in self:
            result.add(item)
        for item in other:
            result.add(item)
        return result

    def intersection(self, other: BPlusTreeSet) -> BPlusTreeSet:
        """Return new set with elements common to both sets."""
        result = BPlusTreeSet(order=self._tree.order)
        for item in self:
            if item in other:
                result.add(item)
        return result

    def difference(self, other: BPlusTreeSet) -> BPlusTreeSet:
        """Return new set with elements in self but not in other."""
        result = BPlusTreeSet(order=self._tree.order)
        for item in self:
            if item not in other:
                result.add(item)
        return result


# ---------------------------------------------------------------------------
# BulkLoader -- efficient bottom-up construction
# ---------------------------------------------------------------------------

class BulkLoader:
    """Build a B+ tree from sorted data in O(n) time.

    Instead of n individual inserts (O(n log n)), constructs the tree
    bottom-up by filling leaves sequentially and building internal nodes.
    """

    @staticmethod
    def load(data, order: int = 32, sorted_data: bool = False) -> BPlusTree:
        """
        Build a B+ tree from data.

        Args:
            data: Iterable of (key, value) pairs.
            order: Tree order.
            sorted_data: If True, data is assumed sorted. If False, will sort.

        Returns:
            A new BPlusTree with all data inserted.
        """
        items = list(data)
        if not items:
            return BPlusTree(order=order)

        if not sorted_data:
            items.sort(key=lambda x: x[0])

        # Deduplicate (keep last value for duplicate keys)
        deduped = []
        for i, (k, v) in enumerate(items):
            if i == 0 or k != items[i - 1][0]:
                deduped.append((k, v))
            else:
                deduped[-1] = (k, v)
        items = deduped

        tree = BPlusTree(order=order)
        tree._size = len(items)

        # Build leaves
        max_keys = order - 1
        leaves = []
        for i in range(0, len(items), max_keys):
            chunk = items[i:i + max_keys]
            leaf = LeafNode()
            leaf.keys = [k for k, v in chunk]
            leaf.values = [v for k, v in chunk]
            leaves.append(leaf)

        # Link leaves
        for i in range(len(leaves)):
            if i > 0:
                leaves[i].prev_leaf = leaves[i - 1]
            if i < len(leaves) - 1:
                leaves[i].next_leaf = leaves[i + 1]

        tree._head = leaves[0]
        tree._tail = leaves[-1]

        if len(leaves) == 1:
            tree._root = leaves[0]
            return tree

        # Build internal nodes bottom-up
        children = leaves
        while len(children) > 1:
            parents = []
            i = 0
            while i < len(children):
                node = InternalNode()
                # Take up to 'order' children
                end = min(i + order, len(children))
                node.children = children[i:end]
                for child in node.children:
                    child.parent = node

                # Separator keys: for leaf children, use first key of each child after the first
                # For internal children, pull up keys similarly
                node.keys = []
                for j in range(1, len(node.children)):
                    child = node.children[j]
                    if child.is_leaf:
                        node.keys.append(child.keys[0])
                    else:
                        # For internal nodes, the separator is the smallest key in the subtree
                        n = child
                        while not n.is_leaf:
                            n = n.children[0]
                        node.keys.append(n.keys[0])

                parents.append(node)
                i = end

            children = parents

        tree._root = children[0]
        return tree


# ---------------------------------------------------------------------------
# Merge / diff operations
# ---------------------------------------------------------------------------

def merge_trees(tree1: BPlusTree, tree2: BPlusTree, order: int = None) -> BPlusTree:
    """Merge two B+ trees into one. On duplicate keys, tree2's value wins."""
    if order is None:
        order = max(tree1.order, tree2.order)

    # Merge-sort the leaf chains
    merged = []
    it1 = tree1.items()
    it2 = tree2.items()

    item1 = next(it1, None)
    item2 = next(it2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            merged.append(item1)
            item1 = next(it1, None)
        elif item1[0] > item2[0]:
            merged.append(item2)
            item2 = next(it2, None)
        else:
            merged.append(item2)  # tree2 wins
            item1 = next(it1, None)
            item2 = next(it2, None)

    while item1 is not None:
        merged.append(item1)
        item1 = next(it1, None)

    while item2 is not None:
        merged.append(item2)
        item2 = next(it2, None)

    return BulkLoader.load(merged, order=order, sorted_data=True)


def diff_trees(tree1: BPlusTree, tree2: BPlusTree) -> dict:
    """
    Compute difference between two trees.

    Returns dict with:
        'only_in_first': [(key, value), ...] -- keys only in tree1
        'only_in_second': [(key, value), ...] -- keys only in tree2
        'different': [(key, val1, val2), ...] -- same key, different values
        'same': int -- count of identical key-value pairs
    """
    result = {
        'only_in_first': [],
        'only_in_second': [],
        'different': [],
        'same': 0
    }

    it1 = tree1.items()
    it2 = tree2.items()

    item1 = next(it1, None)
    item2 = next(it2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            result['only_in_first'].append(item1)
            item1 = next(it1, None)
        elif item1[0] > item2[0]:
            result['only_in_second'].append(item2)
            item2 = next(it2, None)
        else:
            if item1[1] == item2[1]:
                result['same'] += 1
            else:
                result['different'].append((item1[0], item1[1], item2[1]))
            item1 = next(it1, None)
            item2 = next(it2, None)

    while item1 is not None:
        result['only_in_first'].append(item1)
        item1 = next(it1, None)

    while item2 is not None:
        result['only_in_second'].append(item2)
        item2 = next(it2, None)

    return result
