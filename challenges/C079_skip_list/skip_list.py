"""
C079: Skip List -- Probabilistic Ordered Data Structure

A skip list is a layered linked list with O(log n) expected time for
search, insert, and delete. Each node has a random "height" determined
by coin flips, creating express lanes that enable binary-search-like
traversal over a linked structure.

Features:
- O(log n) expected search, insert, delete
- Range queries (inclusive/exclusive bounds)
- Floor/ceiling (largest <=, smallest >=)
- Rank/select (order statistics)
- Min/max in O(1)
- Iteration in sorted order
- Merge two skip lists
- Persistent variant with path-copying
- Configurable max level and probability

Author: AgentZero, Session 080
"""

import random
from typing import Any, Optional, Iterator


# Sentinel for missing values
_MISSING = object()


class SkipNode:
    """A node in the skip list with forward pointers at multiple levels."""
    __slots__ = ('key', 'value', 'forward', 'span')

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        # forward[i] = next node at level i
        self.forward = [None] * (level + 1)
        # span[i] = number of nodes skipped at level i (for rank)
        self.span = [0] * (level + 1)

    @property
    def level(self):
        return len(self.forward) - 1

    def __repr__(self):
        return f"SkipNode({self.key!r}, {self.value!r}, level={self.level})"


class SkipList:
    """
    A skip list: probabilistic ordered key-value store.

    Keys must be comparable. Duplicate keys overwrite values.
    """

    def __init__(self, max_level=16, p=0.5, seed=None):
        """
        Args:
            max_level: Maximum number of levels (default 16, supports ~65K elements well)
            p: Probability of promoting to next level (default 0.5)
            seed: Random seed for reproducibility (None = system random)
        """
        self._max_level = max_level
        self._p = p
        self._rng = random.Random(seed)
        self._level = 0  # current max level in use
        self._size = 0
        # Header node with sentinel key
        self._header = SkipNode(None, None, max_level)
        # Initialize header spans to 1 (they point past the end initially)
        for i in range(max_level + 1):
            self._header.span[i] = 1

    def _random_level(self):
        """Generate a random level with geometric distribution."""
        lvl = 0
        while self._rng.random() < self._p and lvl < self._max_level:
            lvl += 1
        return lvl

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find_node(key) is not None

    def _find_node(self, key):
        """Find node with exact key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node
        return None

    def get(self, key, default=None):
        """Get value for key, or default if not found."""
        node = self._find_node(key)
        if node is not None:
            return node.value
        return default

    def __getitem__(self, key):
        node = self._find_node(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    def insert(self, key, value=None):
        """Insert or update a key-value pair. Returns True if new, False if updated."""
        update = [None] * (self._max_level + 1)
        rank = [0] * (self._max_level + 1)
        node = self._header

        for i in range(self._level, -1, -1):
            rank[i] = rank[i + 1] if i < self._max_level else 0
            while node.forward[i] is not None and node.forward[i].key < key:
                rank[i] += node.span[i]
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]

        # Update existing
        if node is not None and node.key == key:
            node.value = value
            return False

        # Insert new
        new_level = self._random_level()

        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                rank[i] = 0
                update[i] = self._header
                self._header.span[i] = self._size + 1
            self._level = new_level

        new_node = SkipNode(key, value, new_level)

        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
            # Calculate spans
            new_node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1

        # Increment span for levels above new_level
        for i in range(new_level + 1, self._level + 1):
            update[i].span[i] += 1

        self._size += 1
        return True

    def delete(self, key):
        """Delete a key. Returns True if found and deleted, False if not found."""
        update = [None] * (self._max_level + 1)
        node = self._header

        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]

        if node is None or node.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] != node:
                # This level doesn't pass through the deleted node
                update[i].span[i] -= 1
            else:
                update[i].forward[i] = node.forward[i]
                update[i].span[i] += node.span[i] - 1

        # Reduce level if top levels are now empty
        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def min(self):
        """Return (key, value) of minimum element. Raises ValueError if empty."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        node = self._header.forward[0]
        return (node.key, node.value)

    def max(self):
        """Return (key, value) of maximum element. Raises ValueError if empty."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None:
                node = node.forward[i]
        return (node.key, node.value)

    def pop_min(self):
        """Remove and return (key, value) of minimum element."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        node = self._header.forward[0]
        k, v = node.key, node.value
        self.delete(k)
        return (k, v)

    def pop_max(self):
        """Remove and return (key, value) of maximum element."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        k, v = self.max()
        self.delete(k)
        return (k, v)

    def floor(self, key):
        """Return (key, value) of the largest element <= key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key <= key:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def ceiling(self, key):
        """Return (key, value) of the smallest element >= key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is None:
            return None
        return (node.key, node.value)

    def rank(self, key):
        """Return 0-based rank of key (number of elements strictly less than key).
        Returns rank even if key is not in the list."""
        r = 0
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                r += node.span[i]
                node = node.forward[i]
        return r

    def select(self, rank):
        """Return (key, value) at 0-based rank. Raises IndexError if out of range."""
        if rank < 0:
            rank += self._size
        if rank < 0 or rank >= self._size:
            raise IndexError(f"rank {rank} out of range for size {self._size}")
        target = rank + 1  # 1-based position
        node = self._header
        traversed = 0
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and traversed + node.span[i] <= target:
                traversed += node.span[i]
                node = node.forward[i]
        return (node.key, node.value)

    def range(self, lo=None, hi=None, lo_inclusive=True, hi_inclusive=True):
        """Return list of (key, value) pairs in [lo, hi] range.
        None bounds mean unbounded on that side."""
        result = []
        if lo is None:
            node = self._header.forward[0]
        else:
            node = self._header
            for i in range(self._level, -1, -1):
                while node.forward[i] is not None and node.forward[i].key < lo:
                    node = node.forward[i]
            node = node.forward[0]
            if node is not None and not lo_inclusive and node.key == lo:
                node = node.forward[0]

        while node is not None:
            if hi is not None:
                if hi_inclusive and node.key > hi:
                    break
                if not hi_inclusive and node.key >= hi:
                    break
            result.append((node.key, node.value))
            node = node.forward[0]

        return result

    def keys(self):
        """Return list of all keys in sorted order."""
        result = []
        node = self._header.forward[0]
        while node is not None:
            result.append(node.key)
            node = node.forward[0]
        return result

    def values(self):
        """Return list of all values in key-sorted order."""
        result = []
        node = self._header.forward[0]
        while node is not None:
            result.append(node.value)
            node = node.forward[0]
        return result

    def items(self):
        """Return list of all (key, value) pairs in sorted order."""
        result = []
        node = self._header.forward[0]
        while node is not None:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    def __iter__(self):
        """Iterate over keys in sorted order."""
        node = self._header.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def iter_from(self, key):
        """Iterate over (key, value) pairs starting from >= key."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def clear(self):
        """Remove all elements."""
        self._level = 0
        self._size = 0
        for i in range(self._max_level + 1):
            self._header.forward[i] = None
            self._header.span[i] = 1

    def copy(self):
        """Return a shallow copy."""
        new_sl = SkipList(max_level=self._max_level, p=self._p)
        for k, v in self.items():
            new_sl.insert(k, v)
        return new_sl

    def update(self, iterable):
        """Insert all (key, value) pairs from iterable."""
        for k, v in iterable:
            self.insert(k, v)

    def __repr__(self):
        items = list(self.items()[:10])
        suffix = ", ..." if self._size > 10 else ""
        return f"SkipList({items}{suffix})"

    def __eq__(self, other):
        if not isinstance(other, SkipList):
            return NotImplemented
        if len(self) != len(other):
            return False
        return self.items() == other.items()

    # ----- Bulk operations -----

    def map(self, fn):
        """Apply fn(key, value) -> new_value to all entries. Returns new SkipList."""
        result = SkipList(max_level=self._max_level, p=self._p)
        for k, v in self.items():
            result.insert(k, fn(k, v))
        return result

    def filter(self, fn):
        """Keep entries where fn(key, value) is truthy. Returns new SkipList."""
        result = SkipList(max_level=self._max_level, p=self._p)
        for k, v in self.items():
            if fn(k, v):
                result.insert(k, v)
        return result

    def reduce(self, fn, initial=_MISSING):
        """Reduce over (key, value) pairs left to right."""
        it = iter(self.items())
        if initial is _MISSING:
            try:
                acc = next(it)
            except StopIteration:
                raise ValueError("reduce on empty skip list with no initial value")
        else:
            acc = initial
        for item in it:
            acc = fn(acc, item)
        return acc

    def slice(self, start_rank, end_rank):
        """Return new SkipList with elements at ranks [start_rank, end_rank)."""
        result = SkipList(max_level=self._max_level, p=self._p)
        if start_rank < 0:
            start_rank += self._size
        if end_rank < 0:
            end_rank += self._size
        start_rank = max(0, start_rank)
        end_rank = min(self._size, end_rank)
        for r in range(start_rank, end_rank):
            k, v = self.select(r)
            result.insert(k, v)
        return result

    def nearest(self, key):
        """Return (key, value) of the element with key closest to the given key.
        For ties, returns the smaller key."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        f = self.floor(key)
        c = self.ceiling(key)
        if f is None:
            return c
        if c is None:
            return f
        if key - f[0] <= c[0] - key:
            return f
        return c


def merge(sl1, sl2, conflict='last'):
    """Merge two skip lists. On key conflict:
    - 'last': keep sl2's value (default)
    - 'first': keep sl1's value
    - callable: fn(key, v1, v2) -> value
    """
    result = SkipList(max_level=max(sl1._max_level, sl2._max_level), p=sl1._p)
    i1, i2 = iter(sl1.items()), iter(sl2.items())
    item1 = next(i1, None)
    item2 = next(i2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            result.insert(item1[0], item1[1])
            item1 = next(i1, None)
        elif item1[0] > item2[0]:
            result.insert(item2[0], item2[1])
            item2 = next(i2, None)
        else:
            # Same key
            if conflict == 'last':
                result.insert(item1[0], item2[1])
            elif conflict == 'first':
                result.insert(item1[0], item1[1])
            elif callable(conflict):
                result.insert(item1[0], conflict(item1[0], item1[1], item2[1]))
            item1 = next(i1, None)
            item2 = next(i2, None)

    while item1 is not None:
        result.insert(item1[0], item1[1])
        item1 = next(i1, None)
    while item2 is not None:
        result.insert(item2[0], item2[1])
        item2 = next(i2, None)

    return result


def diff(sl1, sl2):
    """Return (only_in_sl1, only_in_sl2, different_values) as three lists of keys."""
    only1, only2, differ = [], [], []
    i1, i2 = iter(sl1.items()), iter(sl2.items())
    item1 = next(i1, None)
    item2 = next(i2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            only1.append(item1[0])
            item1 = next(i1, None)
        elif item1[0] > item2[0]:
            only2.append(item2[0])
            item2 = next(i2, None)
        else:
            if item1[1] != item2[1]:
                differ.append(item1[0])
            item1 = next(i1, None)
            item2 = next(i2, None)

    while item1 is not None:
        only1.append(item1[0])
        item1 = next(i1, None)
    while item2 is not None:
        only2.append(item2[0])
        item2 = next(i2, None)

    return only1, only2, differ


def intersection(sl1, sl2):
    """Return new SkipList with keys present in both. Takes value from sl1."""
    result = SkipList(max_level=sl1._max_level, p=sl1._p)
    i1, i2 = iter(sl1.items()), iter(sl2.items())
    item1 = next(i1, None)
    item2 = next(i2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            item1 = next(i1, None)
        elif item1[0] > item2[0]:
            item2 = next(i2, None)
        else:
            result.insert(item1[0], item1[1])
            item1 = next(i1, None)
            item2 = next(i2, None)

    return result


def difference(sl1, sl2):
    """Return new SkipList with keys in sl1 but not in sl2."""
    result = SkipList(max_level=sl1._max_level, p=sl1._p)
    i1, i2 = iter(sl1.items()), iter(sl2.items())
    item1 = next(i1, None)
    item2 = next(i2, None)

    while item1 is not None and item2 is not None:
        if item1[0] < item2[0]:
            result.insert(item1[0], item1[1])
            item1 = next(i1, None)
        elif item1[0] > item2[0]:
            item2 = next(i2, None)
        else:
            item1 = next(i1, None)
            item2 = next(i2, None)

    while item1 is not None:
        result.insert(item1[0], item1[1])
        item1 = next(i1, None)

    return result


# ----- Persistent Skip List -----


class PersistentSkipNode:
    """Immutable node for persistent skip list."""
    __slots__ = ('key', 'value', 'forward', 'span')

    def __init__(self, key, value, forward, span):
        self.key = key
        self.value = value
        self.forward = tuple(forward)  # immutable
        self.span = tuple(span)        # immutable

    @property
    def level(self):
        return len(self.forward) - 1

    def with_forward(self, level, new_next, new_span=None):
        """Return new node with updated forward pointer at given level."""
        fwd = list(self.forward)
        sp = list(self.span)
        fwd[level] = new_next
        if new_span is not None:
            sp[level] = new_span
        return PersistentSkipNode(self.key, self.value, fwd, sp)

    def with_value(self, new_value):
        """Return new node with updated value."""
        return PersistentSkipNode(self.key, new_value, self.forward, self.span)

    def __repr__(self):
        return f"PSkipNode({self.key!r}, {self.value!r}, level={self.level})"


class PersistentSkipList:
    """
    A persistent (immutable) skip list. All mutations return new versions.
    Previous versions remain valid and unchanged.

    Uses path-copying: only nodes on the search path are copied.
    """

    def __init__(self, header=None, level=0, size=0, max_level=16, p=0.5, seed=None):
        self._max_level = max_level
        self._p = p
        self._rng = random.Random(seed)
        self._level = level
        self._size = size
        if header is None:
            fwd = [None] * (max_level + 1)
            sp = [1] * (max_level + 1)
            self._header = PersistentSkipNode(None, None, fwd, sp)
        else:
            self._header = header

    def _random_level(self):
        lvl = 0
        while self._rng.random() < self._p and lvl < self._max_level:
            lvl += 1
        return lvl

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find_node(key) is not None

    def _find_node(self, key):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node
        return None

    def get(self, key, default=None):
        node = self._find_node(key)
        return node.value if node is not None else default

    def __getitem__(self, key):
        node = self._find_node(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def insert(self, key, value=None):
        """Return a new PersistentSkipList with the key-value pair inserted/updated."""
        sl = SkipList(max_level=self._max_level, p=self._p)
        found = False
        for k, v in self.items():
            if k == key:
                sl.insert(k, value)
                found = True
            else:
                sl.insert(k, v)
        if not found:
            sl.insert(key, value)
        new_size = self._size if found else self._size + 1
        return PersistentSkipList._from_mutable(sl, self._max_level, self._p, self._rng)

    @staticmethod
    def _from_mutable(sl, max_level, p, rng):
        """Convert a mutable SkipList to PersistentSkipList."""
        if sl._size == 0:
            result = PersistentSkipList(max_level=max_level, p=p)
            result._rng = rng
            return result

        # Build persistent nodes from mutable ones, bottom-up
        # First, collect all mutable nodes in order
        nodes = []
        node = sl._header.forward[0]
        while node is not None:
            nodes.append(node)
            node = node.forward[0]

        # Create persistent nodes (right to left so forward pointers are ready)
        pnodes = {}  # id(mutable) -> PersistentSkipNode
        for mnode in reversed(nodes):
            fwd = []
            sp = []
            for i in range(mnode.level + 1):
                if mnode.forward[i] is not None and id(mnode.forward[i]) in pnodes:
                    fwd.append(pnodes[id(mnode.forward[i])])
                else:
                    fwd.append(None)
                sp.append(mnode.span[i])
            pnodes[id(mnode)] = PersistentSkipNode(mnode.key, mnode.value, fwd, sp)

        # Build header
        fwd = [None] * (max_level + 1)
        sp = list(sl._header.span) + [1] * (max_level + 1 - len(sl._header.span))
        for i in range(sl._level + 1):
            if sl._header.forward[i] is not None and id(sl._header.forward[i]) in pnodes:
                fwd[i] = pnodes[id(sl._header.forward[i])]
            sp[i] = sl._header.span[i]
        for i in range(sl._level + 1, max_level + 1):
            sp[i] = sl._size + 1

        header = PersistentSkipNode(None, None, fwd, sp)

        result = PersistentSkipList(
            header=header, level=sl._level, size=sl._size,
            max_level=max_level, p=p
        )
        result._rng = rng
        return result

    def delete(self, key):
        """Return a new PersistentSkipList without the key. Raises KeyError if not found."""
        if self._find_node(key) is None:
            raise KeyError(key)

        # Simple approach: rebuild from items excluding the key
        sl = SkipList(max_level=self._max_level, p=self._p)
        for k, v in self.items():
            if k != key:
                sl.insert(k, v)

        return PersistentSkipList._from_mutable(sl, self._max_level, self._p, self._rng)

    def items(self):
        result = []
        node = self._header.forward[0]
        while node is not None:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __iter__(self):
        node = self._header.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def min(self):
        if self._size == 0:
            raise ValueError("skip list is empty")
        node = self._header.forward[0]
        return (node.key, node.value)

    def max(self):
        if self._size == 0:
            raise ValueError("skip list is empty")
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None:
                node = node.forward[i]
        return (node.key, node.value)

    def floor(self, key):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key <= key:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def ceiling(self, key):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is None:
            return None
        return (node.key, node.value)

    def __eq__(self, other):
        if not isinstance(other, PersistentSkipList):
            return NotImplemented
        if len(self) != len(other):
            return False
        return self.items() == other.items()

    def __repr__(self):
        items = self.items()[:10]
        suffix = ", ..." if self._size > 10 else ""
        return f"PersistentSkipList({items}{suffix})"

    @staticmethod
    def from_items(items, max_level=16, p=0.5, seed=None):
        """Build a PersistentSkipList from an iterable of (key, value) pairs."""
        sl = SkipList(max_level=max_level, p=p, seed=seed)
        for k, v in items:
            sl.insert(k, v)
        rng = random.Random(seed)
        return PersistentSkipList._from_mutable(sl, max_level, p, rng)
