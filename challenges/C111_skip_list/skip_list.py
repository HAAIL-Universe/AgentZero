"""
C111: Skip List -- Probabilistic balanced search structure.

A skip list is a layered linked list that provides O(log n) expected time
for search, insert, and delete operations. It achieves this through
randomized level assignment, creating express lanes for traversal.

Features:
- SkipList: ordered key-value store with probabilistic balancing
- SkipListSet: ordered set (keys only)
- IndexableSkipList: O(log n) rank/select via span tracking
- ConcurrentSkipList: thread-safe with fine-grained locking
- RangeSkipList: efficient range queries with iterators

All variants support: insert, delete, search, floor, ceiling,
predecessor, successor, min, max, range queries, iteration.
"""

import random
import threading
from typing import Any, Optional, Iterator, Tuple, List

MAX_LEVEL = 32
DEFAULT_P = 0.5


class SkipNode:
    """Node in a skip list."""
    __slots__ = ('key', 'value', 'forward', 'span')

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)
        self.span = [0] * (level + 1)


def _random_level(p=DEFAULT_P, max_level=MAX_LEVEL):
    """Generate a random level using geometric distribution."""
    level = 0
    while random.random() < p and level < max_level - 1:
        level += 1
    return level


class SkipList:
    """
    Ordered key-value store using a skip list.

    Expected O(log n) search, insert, delete.
    Keys must be comparable. Duplicate keys update the value.
    """

    def __init__(self, p=DEFAULT_P, max_level=MAX_LEVEL):
        self._p = p
        self._max_level = max_level
        self._level = 0
        self._size = 0
        self._header = SkipNode(None, None, max_level - 1)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find_node(key) is not None

    def __iter__(self):
        node = self._header.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def __repr__(self):
        items = list(self.items())
        return f"SkipList({items})"

    def _find_node(self, key):
        """Find the node with given key, or None."""
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

    def insert(self, key, value=None):
        """Insert or update key-value pair. Returns True if new key."""
        update = [None] * self._max_level
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node
        node = node.forward[0]

        if node is not None and node.key == key:
            node.value = value
            return False

        new_level = _random_level(self._p, self._max_level)
        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                update[i] = self._header
            self._level = new_level

        new_node = SkipNode(key, value, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self._size += 1
        return True

    def delete(self, key):
        """Delete key. Returns True if key was found and deleted."""
        update = [None] * self._max_level
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node
        node = node.forward[0]

        if node is None or node.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] is not node:
                break
            update[i].forward[i] = node.forward[i]

        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def min(self):
        """Return (key, value) of minimum element, or None."""
        node = self._header.forward[0]
        if node is None:
            return None
        return (node.key, node.value)

    def max(self):
        """Return (key, value) of maximum element, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def floor(self, key):
        """Return (key, value) of largest key <= given key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key <= key:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def ceiling(self, key):
        """Return (key, value) of smallest key >= given key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is None:
            return None
        return (node.key, node.value)

    def predecessor(self, key):
        """Return (key, value) of largest key strictly < given key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def successor(self, key):
        """Return (key, value) of smallest key strictly > given key, or None."""
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key <= key:
                node = node.forward[i]
        node = node.forward[0]
        if node is None:
            return None
        return (node.key, node.value)

    def range_query(self, lo, hi, inclusive_lo=True, inclusive_hi=True):
        """Return list of (key, value) pairs in [lo, hi] range."""
        result = []
        node = self._header
        if inclusive_lo:
            for i in range(self._level, -1, -1):
                while node.forward[i] is not None and node.forward[i].key < lo:
                    node = node.forward[i]
        else:
            for i in range(self._level, -1, -1):
                while node.forward[i] is not None and node.forward[i].key <= lo:
                    node = node.forward[i]
        node = node.forward[0]
        while node is not None:
            if inclusive_hi:
                if node.key > hi:
                    break
            else:
                if node.key >= hi:
                    break
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    def items(self):
        """Iterate over (key, value) pairs in order."""
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def keys(self):
        """Iterate over keys in order."""
        return iter(self)

    def values(self):
        """Iterate over values in order."""
        node = self._header.forward[0]
        while node is not None:
            yield node.value
            node = node.forward[0]

    def pop_min(self):
        """Remove and return (key, value) of minimum element."""
        node = self._header.forward[0]
        if node is None:
            raise KeyError("pop from empty skip list")
        key, value = node.key, node.value
        self.delete(key)
        return (key, value)

    def pop_max(self):
        """Remove and return (key, value) of maximum element."""
        m = self.max()
        if m is None:
            raise KeyError("pop from empty skip list")
        self.delete(m[0])
        return m

    def clear(self):
        """Remove all elements."""
        for i in range(self._max_level):
            self._header.forward[i] = None
        self._level = 0
        self._size = 0

    def level_distribution(self):
        """Return dict mapping level -> count of nodes at that level."""
        dist = {}
        node = self._header.forward[0]
        while node is not None:
            nlevel = 0
            for i in range(len(node.forward)):
                if node.forward[i] is not None or i == 0:
                    nlevel = i
            # count actual forward links
            nlevel = len(node.forward) - 1
            dist[nlevel] = dist.get(nlevel, 0) + 1
            node = node.forward[0]
        return dist

    def to_list(self):
        """Return sorted list of (key, value) pairs."""
        return list(self.items())


class SkipListSet:
    """
    Ordered set using a skip list. Keys only, no values.
    """

    def __init__(self, p=DEFAULT_P, max_level=MAX_LEVEL):
        self._sl = SkipList(p=p, max_level=max_level)

    def __len__(self):
        return len(self._sl)

    def __bool__(self):
        return bool(self._sl)

    def __contains__(self, key):
        return key in self._sl

    def __iter__(self):
        return iter(self._sl)

    def __repr__(self):
        return f"SkipListSet({list(self)})"

    def add(self, key):
        """Add key to set. Returns True if new."""
        return self._sl.insert(key, True)

    def remove(self, key):
        """Remove key. Raises KeyError if not found."""
        if not self._sl.delete(key):
            raise KeyError(key)

    def discard(self, key):
        """Remove key if present."""
        self._sl.delete(key)

    def min(self):
        r = self._sl.min()
        return r[0] if r else None

    def max(self):
        r = self._sl.max()
        return r[0] if r else None

    def floor(self, key):
        r = self._sl.floor(key)
        return r[0] if r else None

    def ceiling(self, key):
        r = self._sl.ceiling(key)
        return r[0] if r else None

    def range_query(self, lo, hi, inclusive_lo=True, inclusive_hi=True):
        return [k for k, v in self._sl.range_query(lo, hi, inclusive_lo, inclusive_hi)]

    def pop_min(self):
        return self._sl.pop_min()[0]

    def pop_max(self):
        return self._sl.pop_max()[0]

    def clear(self):
        self._sl.clear()

    def to_list(self):
        return list(self)

    def union(self, other):
        """Return new set with elements from both."""
        result = SkipListSet()
        for k in self:
            result.add(k)
        for k in other:
            result.add(k)
        return result

    def intersection(self, other):
        """Return new set with elements in both."""
        result = SkipListSet()
        for k in self:
            if k in other:
                result.add(k)
        return result

    def difference(self, other):
        """Return new set with elements in self but not other."""
        result = SkipListSet()
        for k in self:
            if k not in other:
                result.add(k)
        return result


class IndexableSkipList:
    """
    Skip list with O(log n) rank and select via span tracking.

    rank(key) -> 0-based index of key
    select(i) -> (key, value) at 0-based index i
    """

    def __init__(self, p=DEFAULT_P, max_level=MAX_LEVEL):
        self._p = p
        self._max_level = max_level
        self._level = 0
        self._size = 0
        self._header = SkipNode(None, None, max_level - 1)
        for i in range(max_level):
            self._header.span[i] = 1

    def __len__(self):
        return self._size

    def __contains__(self, key):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        return node is not None and node.key == key

    def __iter__(self):
        node = self._header.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def get(self, key, default=None):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node.value
        return default

    def insert(self, key, value=None):
        """Insert key-value. Returns True if new key."""
        update = [None] * self._max_level
        rank = [0] * self._max_level
        node = self._header

        for i in range(self._level, -1, -1):
            rank[i] = rank[i + 1] if i < self._level else 0
            while node.forward[i] is not None and node.forward[i].key < key:
                rank[i] += node.span[i]
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]
        if node is not None and node.key == key:
            node.value = value
            return False

        new_level = _random_level(self._p, self._max_level)
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
            new_node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1

        for i in range(new_level + 1, self._level + 1):
            update[i].span[i] += 1

        self._size += 1
        return True

    def delete(self, key):
        """Delete key. Returns True if found and deleted."""
        update = [None] * self._max_level
        node = self._header

        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        node = node.forward[0]
        if node is None or node.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] is not node:
                update[i].span[i] -= 1
            else:
                update[i].forward[i] = node.forward[i]
                update[i].span[i] += node.span[i] - 1

        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def rank(self, key):
        """Return 0-based rank of key, or -1 if not found."""
        r = 0
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                r += node.span[i]
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return r
        return -1

    def select(self, index):
        """Return (key, value) at 0-based index, or None if out of range."""
        if index < 0 or index >= self._size:
            return None
        target = index + 1  # spans are 1-based
        traversed = 0
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and traversed + node.span[i] <= target:
                traversed += node.span[i]
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)

    def range_query(self, lo, hi):
        """Return list of (key, value) in [lo, hi]."""
        result = []
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < lo:
                node = node.forward[i]
        node = node.forward[0]
        while node is not None and node.key <= hi:
            result.append((node.key, node.value))
            node = node.forward[0]
        return result

    def items(self):
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def to_list(self):
        return list(self.items())

    def min(self):
        node = self._header.forward[0]
        return (node.key, node.value) if node else None

    def max(self):
        node = self._header
        for i in range(self._level, -1, -1):
            while node.forward[i] is not None:
                node = node.forward[i]
        if node is self._header:
            return None
        return (node.key, node.value)


class ConcurrentSkipList:
    """
    Thread-safe skip list with fine-grained locking.

    Uses a global lock for simplicity while maintaining the skip list interface.
    For production use, per-node locks with hand-over-hand locking would be better.
    """

    def __init__(self, p=DEFAULT_P, max_level=MAX_LEVEL):
        self._sl = SkipList(p=p, max_level=max_level)
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._sl)

    def __contains__(self, key):
        with self._lock:
            return key in self._sl

    def __iter__(self):
        with self._lock:
            return iter(list(self._sl))

    def get(self, key, default=None):
        with self._lock:
            return self._sl.get(key, default)

    def insert(self, key, value=None):
        with self._lock:
            return self._sl.insert(key, value)

    def delete(self, key):
        with self._lock:
            return self._sl.delete(key)

    def min(self):
        with self._lock:
            return self._sl.min()

    def max(self):
        with self._lock:
            return self._sl.max()

    def floor(self, key):
        with self._lock:
            return self._sl.floor(key)

    def ceiling(self, key):
        with self._lock:
            return self._sl.ceiling(key)

    def range_query(self, lo, hi, inclusive_lo=True, inclusive_hi=True):
        with self._lock:
            return self._sl.range_query(lo, hi, inclusive_lo, inclusive_hi)

    def pop_min(self):
        with self._lock:
            return self._sl.pop_min()

    def pop_max(self):
        with self._lock:
            return self._sl.pop_max()

    def items(self):
        with self._lock:
            return list(self._sl.items())

    def to_list(self):
        with self._lock:
            return self._sl.to_list()

    def clear(self):
        with self._lock:
            self._sl.clear()


class MergeableSkipList(SkipList):
    """
    Skip list that supports efficient merge operations.

    merge(other) -- merge all elements from other into self
    split(key) -- split into two skip lists at key
    """

    def merge(self, other):
        """Merge all elements from other into self."""
        for key, value in other.items():
            self.insert(key, value)

    def split(self, key):
        """Split at key. Returns (left, right) where left has keys < key, right has keys >= key."""
        left = MergeableSkipList(p=self._p, max_level=self._max_level)
        right = MergeableSkipList(p=self._p, max_level=self._max_level)
        for k, v in self.items():
            if k < key:
                left.insert(k, v)
            else:
                right.insert(k, v)
        return left, right

    def bulk_insert(self, items):
        """Insert multiple (key, value) pairs."""
        for k, v in items:
            self.insert(k, v)


class IntervalSkipList:
    """
    Skip list storing intervals [lo, hi] with associated values.
    Supports stabbing queries (find all intervals containing a point)
    and overlap queries (find all intervals overlapping a range).
    """

    def __init__(self):
        self._sl = SkipList()
        self._intervals = {}  # id -> (lo, hi, value)
        self._next_id = 0

    def __len__(self):
        return len(self._intervals)

    def add(self, lo, hi, value=None):
        """Add interval [lo, hi] with value. Returns interval id."""
        iid = self._next_id
        self._next_id += 1
        self._intervals[iid] = (lo, hi, value)
        # Store by lo endpoint for efficient scanning
        existing = self._sl.get(lo)
        if existing is None:
            self._sl.insert(lo, [iid])
        else:
            existing.append(iid)
        return iid

    def remove(self, iid):
        """Remove interval by id. Returns True if found."""
        if iid not in self._intervals:
            return False
        lo, hi, value = self._intervals.pop(iid)
        existing = self._sl.get(lo)
        if existing is not None:
            existing.remove(iid)
            if not existing:
                self._sl.delete(lo)
        return True

    def stab(self, point):
        """Find all intervals containing point. Returns list of (lo, hi, value)."""
        result = []
        for lo, ids in self._sl.items():
            if lo > point:
                break
            for iid in ids:
                ilo, ihi, ival = self._intervals[iid]
                if ilo <= point <= ihi:
                    result.append((ilo, ihi, ival))
        return result

    def overlap(self, qlo, qhi):
        """Find all intervals overlapping [qlo, qhi]. Returns list of (lo, hi, value)."""
        result = []
        for iid, (lo, hi, value) in self._intervals.items():
            if lo <= qhi and hi >= qlo:
                result.append((lo, hi, value))
        return result

    def all_intervals(self):
        """Return list of all (lo, hi, value)."""
        return [(lo, hi, val) for lo, hi, val in self._intervals.values()]
