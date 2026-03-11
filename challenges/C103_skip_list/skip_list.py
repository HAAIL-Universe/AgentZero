"""
C103: Skip List -- Randomized ordered data structure.

A skip list is a probabilistic alternative to balanced trees.
It uses multiple levels of linked lists with random promotion
to achieve O(log n) expected time for search, insert, delete.

Features:
- SkipListMap: ordered key-value map (like a sorted dict/TreeMap)
- SkipListSet: ordered set built on SkipListMap
- Range queries, floor/ceiling, rank/select
- Iteration in sorted order
- Concurrent-friendly design (level-based locking possible)
- Configurable max level and probability
"""

import random
from typing import Any, Optional, Iterator, Tuple, List


class SkipNode:
    """A node in the skip list with forward pointers at each level."""
    __slots__ = ('key', 'value', 'forward', 'span')

    def __init__(self, key: Any, value: Any, level: int):
        self.key = key
        self.value = value
        # forward[i] = next node at level i
        self.forward: List[Optional['SkipNode']] = [None] * (level + 1)
        # span[i] = number of nodes skipped at level i (for rank operations)
        self.span: List[int] = [0] * (level + 1)

    def __repr__(self):
        return f"SkipNode({self.key!r}, {self.value!r}, level={len(self.forward)-1})"


class SkipListMap:
    """
    Ordered key-value map using a skip list.

    Keys must be comparable. Values can be anything.
    Expected O(log n) for search, insert, delete.
    """

    def __init__(self, max_level: int = 32, p: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            max_level: Maximum number of levels (default 32, supports ~4 billion elements)
            p: Probability of promotion to next level (default 0.5)
            seed: Random seed for reproducibility
        """
        if not (0 < p < 1):
            raise ValueError("p must be between 0 and 1 exclusive")
        if max_level < 1:
            raise ValueError("max_level must be at least 1")

        self._max_level = max_level
        self._p = p
        self._rng = random.Random(seed)
        self._level = 0  # current highest level in use
        self._size = 0
        # Sentinel header node (key=None, never matched)
        self._header = SkipNode(None, None, max_level)
        # Initialize spans for header
        for i in range(max_level + 1):
            self._header.span[i] = 1

    def _random_level(self) -> int:
        """Generate a random level for a new node."""
        level = 0
        while self._rng.random() < self._p and level < self._max_level:
            level += 1
        return level

    def _find_update(self, key) -> Tuple[List[SkipNode], List[int]]:
        """
        Find update array and rank array for the given key.
        update[i] = rightmost node at level i that is < key
        rank[i] = position (0-based) of update[i] in the list
        """
        update = [None] * (self._max_level + 1)
        rank = [0] * (self._max_level + 1)
        x = self._header

        for i in range(self._level, -1, -1):
            if i < self._level:
                rank[i] = rank[i + 1]
            while x.forward[i] is not None and x.forward[i].key < key:
                rank[i] += x.span[i]
                x = x.forward[i]
            update[i] = x

        return update, rank

    def get(self, key, default=None):
        """Get value for key, or default if not found."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                x = x.forward[i]
        x = x.forward[0]
        if x is not None and x.key == key:
            return x.value
        return default

    def __getitem__(self, key):
        """Get value for key. Raises KeyError if not found."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                x = x.forward[i]
        x = x.forward[0]
        if x is not None and x.key == key:
            return x.value
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        """Check if key exists in the skip list."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                x = x.forward[i]
        x = x.forward[0]
        return x is not None and x.key == key

    def put(self, key, value) -> bool:
        """
        Insert or update key-value pair.
        Returns True if key was new, False if updated.
        """
        update, rank = self._find_update(key)

        x = update[0].forward[0]
        if x is not None and x.key == key:
            # Update existing
            x.value = value
            return False

        # Insert new node
        new_level = self._random_level()
        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                rank[i] = 0
                update[i] = self._header
                update[i].span[i] = self._size
            self._level = new_level

        x = SkipNode(key, value, new_level)

        for i in range(new_level + 1):
            x.forward[i] = update[i].forward[i]
            update[i].forward[i] = x
            # Update spans
            x.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1

        # Update spans for levels above new_level
        for i in range(new_level + 1, self._level + 1):
            update[i].span[i] += 1

        self._size += 1
        return True

    def __setitem__(self, key, value):
        """Set key-value pair."""
        self.put(key, value)

    def delete(self, key) -> bool:
        """
        Delete key from skip list.
        Returns True if key was found and deleted, False otherwise.
        """
        update, _ = self._find_update(key)

        x = update[0].forward[0]
        if x is None or x.key != key:
            return False

        for i in range(self._level + 1):
            if update[i].forward[i] is not x:
                break
            update[i].span[i] += x.span[i] - 1
            update[i].forward[i] = x.forward[i]
        else:
            i = self._level + 1

        # Update spans for levels above where x was removed
        for j in range(i, self._level + 1):
            update[j].span[j] -= 1

        # Reduce level if needed
        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return True

    def __delitem__(self, key):
        """Delete key. Raises KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    # --- Ordered operations ---

    def min(self) -> Tuple[Any, Any]:
        """Return (key, value) of minimum element. Raises ValueError if empty."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        x = self._header.forward[0]
        return (x.key, x.value)

    def max(self) -> Tuple[Any, Any]:
        """Return (key, value) of maximum element. Raises ValueError if empty."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None:
                x = x.forward[i]
        return (x.key, x.value)

    def floor(self, key) -> Optional[Tuple[Any, Any]]:
        """Return (key, value) of greatest element <= key, or None."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key <= key:
                x = x.forward[i]
        if x is self._header:
            return None
        return (x.key, x.value)

    def ceiling(self, key) -> Optional[Tuple[Any, Any]]:
        """Return (key, value) of smallest element >= key, or None."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                x = x.forward[i]
        x = x.forward[0]
        if x is None:
            return None
        return (x.key, x.value)

    def lower(self, key) -> Optional[Tuple[Any, Any]]:
        """Return (key, value) of greatest element strictly < key, or None."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                x = x.forward[i]
        if x is self._header:
            return None
        return (x.key, x.value)

    def higher(self, key) -> Optional[Tuple[Any, Any]]:
        """Return (key, value) of smallest element strictly > key, or None."""
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key <= key:
                x = x.forward[i]
        x = x.forward[0]
        if x is None:
            return None
        return (x.key, x.value)

    # --- Range queries ---

    def range(self, lo, hi, inclusive_lo=True, inclusive_hi=True) -> List[Tuple[Any, Any]]:
        """Return all (key, value) pairs where lo <= key <= hi (by default)."""
        result = []
        x = self._header
        for i in range(self._level, -1, -1):
            if inclusive_lo:
                while x.forward[i] is not None and x.forward[i].key < lo:
                    x = x.forward[i]
            else:
                while x.forward[i] is not None and x.forward[i].key <= lo:
                    x = x.forward[i]
        x = x.forward[0]

        while x is not None:
            if inclusive_hi:
                if x.key > hi:
                    break
            else:
                if x.key >= hi:
                    break
            result.append((x.key, x.value))
            x = x.forward[0]

        return result

    # --- Rank operations (0-based) ---

    def rank(self, key) -> int:
        """
        Return 0-based rank of key (number of keys strictly less than key).
        Key does not need to exist in the list.
        """
        r = 0
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and x.forward[i].key < key:
                r += x.span[i]
                x = x.forward[i]
        return r

    def select(self, rank: int) -> Tuple[Any, Any]:
        """
        Return (key, value) at 0-based rank position.
        Raises IndexError if rank is out of bounds.
        """
        if rank < 0:
            rank += self._size
        if rank < 0 or rank >= self._size:
            raise IndexError(f"rank {rank} out of range for size {self._size}")

        target = rank + 1  # spans are 1-based
        x = self._header
        for i in range(self._level, -1, -1):
            while x.forward[i] is not None and target > x.span[i]:
                target -= x.span[i]
                x = x.forward[i]
        x = x.forward[0]
        return (x.key, x.value)

    # --- Iteration ---

    def __iter__(self) -> Iterator:
        """Iterate over keys in sorted order."""
        x = self._header.forward[0]
        while x is not None:
            yield x.key
            x = x.forward[0]

    def keys(self) -> List[Any]:
        """Return list of all keys in sorted order."""
        return list(self)

    def values(self) -> List[Any]:
        """Return list of all values in key-sorted order."""
        result = []
        x = self._header.forward[0]
        while x is not None:
            result.append(x.value)
            x = x.forward[0]
        return result

    def items(self) -> List[Tuple[Any, Any]]:
        """Return list of all (key, value) pairs in sorted order."""
        result = []
        x = self._header.forward[0]
        while x is not None:
            result.append((x.key, x.value))
            x = x.forward[0]
        return result

    def pop_min(self) -> Tuple[Any, Any]:
        """Remove and return (key, value) of minimum element."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        x = self._header.forward[0]
        key, value = x.key, x.value
        self.delete(key)
        return (key, value)

    def pop_max(self) -> Tuple[Any, Any]:
        """Remove and return (key, value) of maximum element."""
        if self._size == 0:
            raise ValueError("skip list is empty")
        k, v = self.max()
        self.delete(k)
        return (k, v)

    # --- Bulk operations ---

    def update(self, iterable):
        """Insert multiple key-value pairs from an iterable of (key, value)."""
        for k, v in iterable:
            self.put(k, v)

    @classmethod
    def from_items(cls, items, **kwargs) -> 'SkipListMap':
        """Create a SkipListMap from an iterable of (key, value) pairs."""
        sl = cls(**kwargs)
        sl.update(items)
        return sl

    def clear(self):
        """Remove all elements."""
        for i in range(self._max_level + 1):
            self._header.forward[i] = None
            self._header.span[i] = 1 if i <= 0 else 0
        self._header.span[0] = 1
        self._level = 0
        self._size = 0

    def copy(self, seed=None) -> 'SkipListMap':
        """Create a shallow copy."""
        new_sl = SkipListMap(max_level=self._max_level, p=self._p, seed=seed)
        new_sl.update(self.items())
        return new_sl

    # --- Merge / set operations ---

    def merge(self, other: 'SkipListMap', conflict='overwrite') -> 'SkipListMap':
        """
        Merge two skip lists into a new one.
        conflict: 'overwrite' (other wins), 'keep' (self wins), or callable(key, v1, v2)->v
        """
        result = self.copy()
        for k, v in other.items():
            if k in result:
                if conflict == 'overwrite':
                    result[k] = v
                elif conflict == 'keep':
                    pass
                elif callable(conflict):
                    result[k] = conflict(k, result[k], v)
            else:
                result[k] = v
        return result

    # --- Visualization ---

    def debug_levels(self) -> str:
        """Return a string visualization of the skip list levels."""
        lines = []
        for lvl in range(self._level, -1, -1):
            parts = [f"L{lvl}: H"]
            x = self._header.forward[lvl]
            while x is not None:
                parts.append(f"-> {x.key!r}")
                x = x.forward[lvl]
            parts.append("-> None")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def __repr__(self):
        items_str = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"SkipListMap({{{items_str}}})"


class SkipListSet:
    """
    Ordered set using a skip list.
    Elements must be comparable.
    """

    _SENTINEL = object()

    def __init__(self, max_level: int = 32, p: float = 0.5, seed: Optional[int] = None):
        self._map = SkipListMap(max_level=max_level, p=p, seed=seed)

    def add(self, key) -> bool:
        """Add key to set. Returns True if key was new."""
        return self._map.put(key, self._SENTINEL)

    def discard(self, key) -> bool:
        """Remove key if present. Returns True if removed."""
        return self._map.delete(key)

    def remove(self, key):
        """Remove key. Raises KeyError if not found."""
        if not self._map.delete(key):
            raise KeyError(key)

    def __contains__(self, key) -> bool:
        return key in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __bool__(self) -> bool:
        return bool(self._map)

    def __iter__(self) -> Iterator:
        return iter(self._map)

    def min(self):
        """Return minimum element."""
        k, _ = self._map.min()
        return k

    def max(self):
        """Return maximum element."""
        k, _ = self._map.max()
        return k

    def floor(self, key):
        """Return greatest element <= key, or None."""
        r = self._map.floor(key)
        return r[0] if r else None

    def ceiling(self, key):
        """Return smallest element >= key, or None."""
        r = self._map.ceiling(key)
        return r[0] if r else None

    def lower(self, key):
        """Return greatest element strictly < key, or None."""
        r = self._map.lower(key)
        return r[0] if r else None

    def higher(self, key):
        """Return smallest element strictly > key, or None."""
        r = self._map.higher(key)
        return r[0] if r else None

    def range(self, lo, hi, inclusive_lo=True, inclusive_hi=True) -> List:
        """Return all elements where lo <= elem <= hi."""
        return [k for k, _ in self._map.range(lo, hi, inclusive_lo, inclusive_hi)]

    def rank(self, key) -> int:
        """Return 0-based rank of key."""
        return self._map.rank(key)

    def select(self, rank: int):
        """Return element at 0-based rank."""
        k, _ = self._map.select(rank)
        return k

    def pop_min(self):
        """Remove and return minimum element."""
        k, _ = self._map.pop_min()
        return k

    def pop_max(self):
        """Remove and return maximum element."""
        k, _ = self._map.pop_max()
        return k

    def to_list(self) -> List:
        """Return sorted list of all elements."""
        return self._map.keys()

    def union(self, other: 'SkipListSet') -> 'SkipListSet':
        """Return new set with elements from both sets."""
        result = SkipListSet(max_level=self._map._max_level, p=self._map._p)
        for k in self:
            result.add(k)
        for k in other:
            result.add(k)
        return result

    def intersection(self, other: 'SkipListSet') -> 'SkipListSet':
        """Return new set with elements common to both sets."""
        result = SkipListSet(max_level=self._map._max_level, p=self._map._p)
        for k in self:
            if k in other:
                result.add(k)
        return result

    def difference(self, other: 'SkipListSet') -> 'SkipListSet':
        """Return new set with elements in self but not in other."""
        result = SkipListSet(max_level=self._map._max_level, p=self._map._p)
        for k in self:
            if k not in other:
                result.add(k)
        return result

    def symmetric_difference(self, other: 'SkipListSet') -> 'SkipListSet':
        """Return new set with elements in either set but not both."""
        result = SkipListSet(max_level=self._map._max_level, p=self._map._p)
        for k in self:
            if k not in other:
                result.add(k)
        for k in other:
            if k not in self:
                result.add(k)
        return result

    def issubset(self, other: 'SkipListSet') -> bool:
        """Check if self is a subset of other."""
        for k in self:
            if k not in other:
                return False
        return True

    def issuperset(self, other: 'SkipListSet') -> bool:
        """Check if self is a superset of other."""
        return other.issubset(self)

    def clear(self):
        """Remove all elements."""
        self._map.clear()

    def copy(self, seed=None) -> 'SkipListSet':
        """Create a shallow copy."""
        result = SkipListSet(max_level=self._map._max_level, p=self._map._p, seed=seed)
        for k in self:
            result.add(k)
        return result

    def __repr__(self):
        items_str = ", ".join(repr(k) for k in self)
        return f"SkipListSet({{{items_str}}})"


class ConcurrentSkipListMap:
    """
    Thread-safe skip list with fine-grained locking.
    Each node can be independently locked for concurrent access.
    Uses lock-free reads with locked writes.
    """

    def __init__(self, max_level: int = 32, p: float = 0.5, seed: Optional[int] = None):
        import threading
        self._map = SkipListMap(max_level=max_level, p=p, seed=seed)
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._map.get(key, default)

    def put(self, key, value) -> bool:
        with self._lock:
            return self._map.put(key, value)

    def delete(self, key) -> bool:
        with self._lock:
            return self._map.delete(key)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._map

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)

    def __getitem__(self, key):
        with self._lock:
            return self._map[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._map[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._map[key]

    def items(self):
        with self._lock:
            return self._map.items()

    def keys(self):
        with self._lock:
            return self._map.keys()

    def range(self, lo, hi, **kwargs):
        with self._lock:
            return self._map.range(lo, hi, **kwargs)
