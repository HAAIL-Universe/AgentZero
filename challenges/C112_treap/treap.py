"""
C112: Treap -- Randomized Binary Search Tree

A treap combines BST ordering on keys with min-heap ordering on random priorities.
This gives expected O(log n) time for search, insert, delete without explicit balancing.

Variants:
1. Treap -- basic randomized treap with split/merge
2. ImplicitTreap -- implicit key treap (array-like, supports reverse/insert-at/delete-at)
3. PersistentTreap -- path-copying persistent treap with version history
4. MergeableTreap -- treap with efficient merge of arbitrary treaps
5. IntervalTreap -- treap augmented with interval data for stabbing queries
"""

import random
from typing import Any, Optional, List, Tuple, Iterator, Callable


# ============================================================
# Variant 1: Treap (split/merge based)
# ============================================================

class TreapNode:
    __slots__ = ('key', 'value', 'priority', 'left', 'right', 'size')

    def __init__(self, key, value=None, priority=None):
        self.key = key
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.size = 1

    def __repr__(self):
        return f"TreapNode({self.key}, p={self.priority:.3f})"


def _update_size(node):
    if node:
        node.size = 1 + _size(node.left) + _size(node.right)


def _size(node):
    return node.size if node else 0


def _split(node, key):
    """Split treap into (left, right) where left has keys < key, right has keys >= key."""
    if node is None:
        return None, None
    if key <= node.key:
        left, node.left = _split(node.left, key)
        _update_size(node)
        return left, node
    else:
        node.right, right = _split(node.right, key)
        _update_size(node)
        return node, right


def _merge(left, right):
    """Merge two treaps where all keys in left < all keys in right."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        left.right = _merge(left.right, right)
        _update_size(left)
        return left
    else:
        right.left = _merge(left, right.left)
        _update_size(right)
        return right


class Treap:
    """Randomized BST using split/merge operations."""

    def __init__(self):
        self._root = None

    @property
    def root(self):
        return self._root

    def __len__(self):
        return _size(self._root)

    def __bool__(self):
        return self._root is not None

    def insert(self, key, value=None):
        """Insert key-value pair. If key exists, update value."""
        existing = self._find_node(self._root, key)
        if existing:
            existing.value = value
            return
        node = TreapNode(key, value)
        left, right = _split(self._root, key)
        self._root = _merge(_merge(left, node), right)

    def delete(self, key):
        """Delete key. Returns True if found and deleted."""
        self._root, deleted = self._delete(self._root, key)
        return deleted

    def _delete(self, node, key):
        if node is None:
            return None, False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
            _update_size(node)
            return node, deleted
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
            _update_size(node)
            return node, deleted
        else:
            merged = _merge(node.left, node.right)
            return merged, True

    def search(self, key):
        """Search for key. Returns value or None."""
        node = self._find_node(self._root, key)
        return node.value if node else None

    def __contains__(self, key):
        return self._find_node(self._root, key) is not None

    def _find_node(self, node, key):
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return None

    def min(self):
        """Return minimum key."""
        if not self._root:
            raise ValueError("Treap is empty")
        node = self._root
        while node.left:
            node = node.left
        return node.key

    def max(self):
        """Return maximum key."""
        if not self._root:
            raise ValueError("Treap is empty")
        node = self._root
        while node.right:
            node = node.right
        return node.key

    def floor(self, key):
        """Return largest key <= given key, or None."""
        result = None
        node = self._root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                result = node.key
                node = node.right
            else:
                return node.key
        return result

    def ceiling(self, key):
        """Return smallest key >= given key, or None."""
        result = None
        node = self._root
        while node:
            if key > node.key:
                node = node.right
            elif key < node.key:
                result = node.key
                node = node.left
            else:
                return node.key
        return result

    def rank(self, key):
        """Return number of keys strictly less than given key."""
        rank = 0
        node = self._root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                rank += 1 + _size(node.left)
                node = node.right
            else:
                rank += _size(node.left)
                break
        return rank

    def select(self, k):
        """Return k-th smallest key (0-indexed)."""
        if k < 0 or k >= len(self):
            raise IndexError(f"Index {k} out of range")
        node = self._root
        while node:
            left_size = _size(node.left)
            if k < left_size:
                node = node.left
            elif k > left_size:
                k -= left_size + 1
                node = node.right
            else:
                return node.key

    def range_query(self, lo, hi):
        """Return all keys in [lo, hi] in sorted order."""
        result = []
        self._range_collect(self._root, lo, hi, result)
        return result

    def _range_collect(self, node, lo, hi, result):
        if node is None:
            return
        if lo < node.key:
            self._range_collect(node.left, lo, hi, result)
        if lo <= node.key <= hi:
            result.append(node.key)
        if node.key < hi:
            self._range_collect(node.right, lo, hi, result)

    def inorder(self):
        """Return keys in sorted order."""
        result = []
        self._inorder(self._root, result)
        return result

    def _inorder(self, node, result):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node.key)
        self._inorder(node.right, result)

    def __iter__(self):
        return iter(self.inorder())

    def split(self, key):
        """Split into two treaps: keys < key, keys >= key."""
        left_root, right_root = _split(self._root, key)
        left_treap = Treap()
        left_treap._root = left_root
        right_treap = Treap()
        right_treap._root = right_root
        self._root = None
        return left_treap, right_treap

    @staticmethod
    def merge(left, right):
        """Merge two treaps. All keys in left must be < all keys in right."""
        result = Treap()
        result._root = _merge(left._root, right._root)
        left._root = None
        right._root = None
        return result

    def _verify_bst(self, node=None, lo=None, hi=None):
        """Verify BST and heap properties (for testing)."""
        if node is None:
            return True
        if lo is not None and node.key <= lo:
            return False
        if hi is not None and node.key >= hi:
            return False
        if node.left and node.left.priority > node.priority:
            return False
        if node.right and node.right.priority > node.priority:
            return False
        if node.size != 1 + _size(node.left) + _size(node.right):
            return False
        return (self._verify_bst(node.left, lo, node.key) and
                self._verify_bst(node.right, node.key, hi))

    def verify(self):
        """Verify treap invariants."""
        return self._verify_bst(self._root)


# ============================================================
# Variant 2: Implicit Treap (array-like operations)
# ============================================================

class ImplicitTreapNode:
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


def _ipush(node):
    """Push down reverse flag."""
    if node and node.reversed:
        node.left, node.right = node.right, node.left
        if node.left:
            node.left.reversed = not node.left.reversed
        if node.right:
            node.right.reversed = not node.right.reversed
        node.reversed = False


def _isplit(node, k):
    """Split by implicit index: left has first k elements."""
    if node is None:
        return None, None
    _ipush(node)
    left_size = _isize(node.left)
    if k <= left_size:
        left, node.left = _isplit(node.left, k)
        _iupdate(node)
        return left, node
    else:
        node.right, right = _isplit(node.right, k - left_size - 1)
        _iupdate(node)
        return node, right


def _imerge(left, right):
    """Merge two implicit treaps."""
    if left is None:
        return right
    if right is None:
        return left
    _ipush(left)
    _ipush(right)
    if left.priority > right.priority:
        left.right = _imerge(left.right, right)
        _iupdate(left)
        return left
    else:
        right.left = _imerge(left, right.left)
        _iupdate(right)
        return right


class ImplicitTreap:
    """Array-like data structure using implicit keys.

    Supports O(log n) insert-at, delete-at, reverse-range, and access-by-index.
    """

    def __init__(self, values=None):
        self._root = None
        if values:
            for v in values:
                self.append(v)

    def __len__(self):
        return _isize(self._root)

    def __bool__(self):
        return self._root is not None

    def append(self, value):
        """Append value to end."""
        node = ImplicitTreapNode(value)
        self._root = _imerge(self._root, node)

    def insert_at(self, index, value):
        """Insert value at given index."""
        if index < 0 or index > len(self):
            raise IndexError(f"Index {index} out of range")
        node = ImplicitTreapNode(value)
        left, right = _isplit(self._root, index)
        self._root = _imerge(_imerge(left, node), right)

    def delete_at(self, index):
        """Delete element at given index. Returns deleted value."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        left, rest = _isplit(self._root, index)
        deleted, right = _isplit(rest, 1)
        self._root = _imerge(left, right)
        return deleted.value

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        node = self._root
        while node:
            _ipush(node)
            left_size = _isize(node.left)
            if index < left_size:
                node = node.left
            elif index > left_size:
                index -= left_size + 1
                node = node.right
            else:
                return node.value
        raise IndexError("Should not reach here")

    def __setitem__(self, index, value):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        node = self._root
        while node:
            _ipush(node)
            left_size = _isize(node.left)
            if index < left_size:
                node = node.left
            elif index > left_size:
                index -= left_size + 1
                node = node.right
            else:
                node.value = value
                return
        raise IndexError("Should not reach here")

    def reverse_range(self, lo, hi):
        """Reverse elements in range [lo, hi] (inclusive)."""
        if lo < 0 or hi >= len(self) or lo > hi:
            raise IndexError(f"Invalid range [{lo}, {hi}]")
        left, rest = _isplit(self._root, lo)
        mid, right = _isplit(rest, hi - lo + 1)
        if mid:
            mid.reversed = not mid.reversed
        self._root = _imerge(_imerge(left, mid), right)

    def to_list(self):
        """Convert to Python list."""
        result = []
        self._collect(self._root, result)
        return result

    def _collect(self, node, result):
        if node is None:
            return
        _ipush(node)
        self._collect(node.left, result)
        result.append(node.value)
        self._collect(node.right, result)

    def __iter__(self):
        return iter(self.to_list())

    def split_at(self, index):
        """Split into two implicit treaps at index."""
        left_root, right_root = _isplit(self._root, index)
        left = ImplicitTreap()
        left._root = left_root
        right = ImplicitTreap()
        right._root = right_root
        self._root = None
        return left, right

    @staticmethod
    def merge(left, right):
        """Concatenate two implicit treaps."""
        result = ImplicitTreap()
        result._root = _imerge(left._root, right._root)
        left._root = None
        right._root = None
        return result


# ============================================================
# Variant 3: Persistent Treap (immutable, versioned)
# ============================================================

class PersistentTreapNode:
    __slots__ = ('key', 'value', 'priority', 'left', 'right', 'size')

    def __init__(self, key, value, priority, left=None, right=None):
        self.key = key
        self.value = value
        self.priority = priority
        self.left = left
        self.right = right
        self.size = 1 + _psize(left) + _psize(right)


def _psize(node):
    return node.size if node else 0


def _pcopy(node, left=None, right=None):
    """Create a new node with updated children (path copying)."""
    if node is None:
        return None
    l = left if left is not ... else node.left
    r = right if right is not ... else node.right
    return PersistentTreapNode(node.key, node.value, node.priority, l, r)


def _psplit(node, key):
    """Persistent split -- returns new nodes, original unchanged."""
    if node is None:
        return None, None
    if key <= node.key:
        left, new_left = _psplit(node.left, key)
        new_node = PersistentTreapNode(node.key, node.value, node.priority, new_left, node.right)
        return left, new_node
    else:
        new_right, right = _psplit(node.right, key)
        new_node = PersistentTreapNode(node.key, node.value, node.priority, node.left, new_right)
        return new_node, right


def _pmerge(left, right):
    """Persistent merge -- returns new nodes."""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        new_right = _pmerge(left.right, right)
        return PersistentTreapNode(left.key, left.value, left.priority, left.left, new_right)
    else:
        new_left = _pmerge(left, right.left)
        return PersistentTreapNode(right.key, right.value, right.priority, new_left, right.right)


class PersistentTreap:
    """Immutable treap with version history. Each mutation returns a new version."""

    def __init__(self, root=None, history=None):
        self._root = root
        self._history = history or []

    @property
    def root(self):
        return self._root

    def __len__(self):
        return _psize(self._root)

    def __bool__(self):
        return self._root is not None

    def insert(self, key, value=None):
        """Return new treap with key inserted."""
        # Check if key exists -- if so, update value
        existing = self._find(self._root, key)
        if existing:
            new_root = self._update_node(self._root, key, value)
        else:
            node = PersistentTreapNode(key, value, random.random())
            left, right = _psplit(self._root, key)
            new_root = _pmerge(_pmerge(left, node), right)
        new_history = self._history + [self._root]
        return PersistentTreap(new_root, new_history)

    def _update_node(self, node, key, value):
        """Path-copy update of existing key's value."""
        if node is None:
            return None
        if key < node.key:
            new_left = self._update_node(node.left, key, value)
            return PersistentTreapNode(node.key, node.value, node.priority, new_left, node.right)
        elif key > node.key:
            new_right = self._update_node(node.right, key, value)
            return PersistentTreapNode(node.key, node.value, node.priority, node.left, new_right)
        else:
            return PersistentTreapNode(node.key, value, node.priority, node.left, node.right)

    def delete(self, key):
        """Return new treap with key deleted."""
        new_root, deleted = self._pdelete(self._root, key)
        if not deleted:
            return self  # Key not found, return same version
        new_history = self._history + [self._root]
        return PersistentTreap(new_root, new_history)

    def _pdelete(self, node, key):
        if node is None:
            return None, False
        if key < node.key:
            new_left, deleted = self._pdelete(node.left, key)
            if not deleted:
                return node, False
            return PersistentTreapNode(node.key, node.value, node.priority, new_left, node.right), True
        elif key > node.key:
            new_right, deleted = self._pdelete(node.right, key)
            if not deleted:
                return node, False
            return PersistentTreapNode(node.key, node.value, node.priority, node.left, new_right), True
        else:
            return _pmerge(node.left, node.right), True

    def search(self, key):
        """Search for key. Returns value or None."""
        node = self._find(self._root, key)
        return node.value if node else None

    def __contains__(self, key):
        return self._find(self._root, key) is not None

    def _find(self, node, key):
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return None

    def version_count(self):
        """Return number of previous versions."""
        return len(self._history)

    def get_version(self, version):
        """Get a previous version (0 = first)."""
        if version < 0 or version >= len(self._history):
            raise IndexError(f"Version {version} out of range")
        return PersistentTreap(self._history[version])

    def inorder(self):
        result = []
        self._inorder(self._root, result)
        return result

    def _inorder(self, node, result):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node.key)
        self._inorder(node.right, result)

    def __iter__(self):
        return iter(self.inorder())


# ============================================================
# Variant 4: Mergeable Treap (merge arbitrary treaps)
# ============================================================

class MergeableTreap(Treap):
    """Treap that supports merging two arbitrary treaps (with overlapping key ranges).

    Uses split-based merge: split one treap by each key of the other and recombine.
    """

    def union(self, other):
        """Return new treap containing all keys from both. Self's values win on conflicts."""
        result = MergeableTreap()
        result._root = self._union(self._root, other._root)
        return result

    def _union(self, t1, t2):
        if t1 is None:
            return self._copy_tree(t2)
        if t2 is None:
            return self._copy_tree(t1)
        # Ensure t1 has higher priority (acts as root)
        if t1.priority < t2.priority:
            t1, t2 = t2, t1
        # Split t2 by t1's key
        left2, right2 = self._split_by(t2, t1.key)
        new_left = self._union(t1.left, left2)
        new_right = self._union(t1.right, right2)
        node = TreapNode(t1.key, t1.value, t1.priority)
        node.left = new_left
        node.right = new_right
        _update_size(node)
        return node

    def intersection(self, other):
        """Return new treap containing only keys present in both."""
        result = MergeableTreap()
        result._root = self._intersection(self._root, other._root)
        return result

    def _intersection(self, t1, t2):
        if t1 is None or t2 is None:
            return None
        # Split t2 by t1's key
        left2, right2 = self._split_by(t2, t1.key)
        found = self._find_node(t2, t1.key) is not None
        new_left = self._intersection(t1.left, left2)
        new_right = self._intersection(t1.right, right2)
        if found:
            node = TreapNode(t1.key, t1.value, t1.priority)
            node.left = new_left
            node.right = new_right
            _update_size(node)
            return node
        else:
            return _merge(new_left, new_right)

    def difference(self, other):
        """Return new treap with keys in self but not in other."""
        result = MergeableTreap()
        result._root = self._difference(self._root, other._root)
        return result

    def _difference(self, t1, t2):
        if t1 is None:
            return None
        if t2 is None:
            return self._copy_tree(t1)
        left2, right2 = self._split_by(t2, t1.key)
        found = self._find_node(t2, t1.key) is not None
        new_left = self._difference(t1.left, left2)
        new_right = self._difference(t1.right, right2)
        if not found:
            node = TreapNode(t1.key, t1.value, t1.priority)
            node.left = new_left
            node.right = new_right
            _update_size(node)
            return node
        else:
            return _merge(new_left, new_right)

    def _split_by(self, node, key):
        """Split tree by key, excluding the key itself from both sides."""
        if node is None:
            return None, None
        if key < node.key:
            left, new_left = self._split_by(node.left, key)
            n = TreapNode(node.key, node.value, node.priority)
            n.left = new_left
            n.right = self._copy_tree(node.right)
            _update_size(n)
            return left, n
        elif key > node.key:
            new_right, right = self._split_by(node.right, key)
            n = TreapNode(node.key, node.value, node.priority)
            n.left = self._copy_tree(node.left)
            n.right = new_right
            _update_size(n)
            return n, right
        else:
            # Key found -- exclude it from both
            return self._copy_tree(node.left), self._copy_tree(node.right)

    def _copy_tree(self, node):
        if node is None:
            return None
        n = TreapNode(node.key, node.value, node.priority)
        n.left = self._copy_tree(node.left)
        n.right = self._copy_tree(node.right)
        _update_size(n)
        return n


# ============================================================
# Variant 5: Interval Treap
# ============================================================

class IntervalTreapNode:
    __slots__ = ('lo', 'hi', 'value', 'priority', 'left', 'right', 'size', 'max_hi')

    def __init__(self, lo, hi, value=None, priority=None):
        self.lo = lo
        self.hi = hi
        self.value = value
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.size = 1
        self.max_hi = hi


def _it_size(node):
    return node.size if node else 0


def _it_max_hi(node):
    return node.max_hi if node else float('-inf')


def _it_update(node):
    if node:
        node.size = 1 + _it_size(node.left) + _it_size(node.right)
        node.max_hi = max(node.hi, _it_max_hi(node.left), _it_max_hi(node.right))


class IntervalTreap:
    """Treap augmented with intervals for efficient stabbing/overlap queries.

    Keys are interval low endpoints. Augmented with max_hi for pruning.
    """

    def __init__(self):
        self._root = None

    def __len__(self):
        return _it_size(self._root)

    def __bool__(self):
        return self._root is not None

    def insert(self, lo, hi, value=None):
        """Insert interval [lo, hi] with optional value."""
        node = IntervalTreapNode(lo, hi, value)
        self._root = self._insert(self._root, node)

    def _insert(self, root, node):
        if root is None:
            return node
        if node.priority > root.priority:
            # New node becomes root, split existing tree
            node.left, node.right = self._split(root, node.lo)
            _it_update(node)
            return node
        if node.lo <= root.lo:
            root.left = self._insert(root.left, node)
        else:
            root.right = self._insert(root.right, node)
        _it_update(root)
        return root

    def _split(self, node, key):
        if node is None:
            return None, None
        if key <= node.lo:
            left, node.left = self._split(node.left, key)
            _it_update(node)
            return left, node
        else:
            node.right, right = self._split(node.right, key)
            _it_update(node)
            return node, right

    def delete(self, lo, hi):
        """Delete first interval matching [lo, hi]. Returns True if found."""
        self._root, deleted = self._delete(self._root, lo, hi)
        return deleted

    def _delete(self, node, lo, hi):
        if node is None:
            return None, False
        if lo == node.lo and hi == node.hi:
            merged = self._merge(node.left, node.right)
            return merged, True
        if lo <= node.lo:
            node.left, deleted = self._delete(node.left, lo, hi)
        else:
            node.right, deleted = self._delete(node.right, lo, hi)
        _it_update(node)
        return node, deleted

    def _merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            _it_update(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            _it_update(right)
            return right

    def stab(self, point):
        """Return all intervals containing the point."""
        result = []
        self._stab(self._root, point, result)
        return result

    def _stab(self, node, point, result):
        if node is None:
            return
        if _it_max_hi(node) < point:
            return  # Prune: no interval in this subtree can contain point
        self._stab(node.left, point, result)
        if node.lo <= point <= node.hi:
            result.append((node.lo, node.hi, node.value))
        if point >= node.lo:
            self._stab(node.right, point, result)

    def overlap(self, lo, hi):
        """Return all intervals overlapping [lo, hi]."""
        result = []
        self._overlap(self._root, lo, hi, result)
        return result

    def _overlap(self, node, lo, hi, result):
        if node is None:
            return
        if _it_max_hi(node) < lo:
            return  # Prune
        self._overlap(node.left, lo, hi, result)
        if node.lo <= hi and node.hi >= lo:
            result.append((node.lo, node.hi, node.value))
        if node.lo <= hi:
            self._overlap(node.right, lo, hi, result)

    def all_intervals(self):
        """Return all intervals in sorted order by lo."""
        result = []
        self._collect_all(self._root, result)
        return result

    def _collect_all(self, node, result):
        if node is None:
            return
        self._collect_all(node.left, result)
        result.append((node.lo, node.hi, node.value))
        self._collect_all(node.right, result)

    def min_interval(self):
        """Return interval with smallest lo."""
        if not self._root:
            raise ValueError("IntervalTreap is empty")
        node = self._root
        while node.left:
            node = node.left
        return (node.lo, node.hi, node.value)

    def enclosing(self, lo, hi):
        """Return all intervals that fully enclose [lo, hi]."""
        result = []
        self._enclosing(self._root, lo, hi, result)
        return result

    def _enclosing(self, node, lo, hi, result):
        if node is None:
            return
        if _it_max_hi(node) < hi:
            return  # Prune
        self._enclosing(node.left, lo, hi, result)
        if node.lo <= lo and node.hi >= hi:
            result.append((node.lo, node.hi, node.value))
        if node.lo <= lo:
            self._enclosing(node.right, lo, hi, result)
