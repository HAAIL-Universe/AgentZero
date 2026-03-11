"""
C118: Cache Systems
==================
Seven cache implementations with different eviction policies:
1. LRUCache -- Least Recently Used (O(1) via hash + doubly-linked list)
2. LFUCache -- Least Frequently Used (O(1) via frequency buckets)
3. TTLCache -- Time-to-live with lazy expiration
4. SLRUCache -- Segmented LRU (probation + protected segments)
5. ARCCache -- Adaptive Replacement Cache (self-tuning recency vs frequency)
6. WriteBackCache -- Write-back with dirty tracking and flush
7. MultiTierCache -- Composable multi-level cache (L1 -> L2 -> ... -> origin)
"""

import time as _time_module


# ============================================================
# Doubly-linked list node and base infrastructure
# ============================================================

class _Node:
    """Doubly-linked list node for O(1) cache operations."""
    __slots__ = ('key', 'value', 'prev', 'next', 'freq', 'expire_at', 'dirty')

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.freq = 1
        self.expire_at = None
        self.dirty = False


class _DoublyLinkedList:
    """Doubly-linked list with sentinel nodes for O(1) add/remove."""

    def __init__(self):
        self.head = _Node()  # sentinel
        self.tail = _Node()  # sentinel
        self.head.next = self.tail
        self.tail.prev = self.head
        self._size = 0

    def __len__(self):
        return self._size

    def add_front(self, node):
        """Add node right after head sentinel."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self._size += 1

    def add_back(self, node):
        """Add node right before tail sentinel."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        self._size += 1

    def remove(self, node):
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self._size -= 1

    def remove_last(self):
        """Remove and return the node before tail sentinel (LRU victim)."""
        if self._size == 0:
            return None
        node = self.tail.prev
        self.remove(node)
        return node

    def peek_last(self):
        """Return the node before tail sentinel without removing."""
        if self._size == 0:
            return None
        return self.tail.prev

    def move_to_front(self, node):
        """Move existing node to front (most recently used)."""
        self.remove(node)
        self.add_front(node)

    def items(self):
        """Iterate from front (MRU) to back (LRU)."""
        cur = self.head.next
        while cur is not self.tail:
            yield cur
            cur = cur.next

    def items_reverse(self):
        """Iterate from back (LRU) to front (MRU)."""
        cur = self.tail.prev
        while cur is not self.head:
            yield cur
            cur = cur.prev


# ============================================================
# 1. LRU Cache
# ============================================================

class LRUCache:
    """Least Recently Used cache with O(1) get/put/delete.

    Args:
        capacity: Maximum number of entries.
        on_evict: Optional callback(key, value) called when an entry is evicted.
    """

    def __init__(self, capacity, on_evict=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.on_evict = on_evict
        self._map = {}         # key -> _Node
        self._list = _DoublyLinkedList()
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        """Get value by key. Returns default if not found."""
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            return default
        self._hits += 1
        self._list.move_to_front(node)
        return node.value

    def put(self, key, value):
        """Insert or update key-value pair."""
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._list.move_to_front(node)
            return
        if len(self._map) >= self.capacity:
            self._evict()
        node = _Node(key, value)
        self._map[key] = node
        self._list.add_front(node)

    def delete(self, key):
        """Remove key. Returns True if key existed."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._list.remove(node)
        return True

    def __contains__(self, key):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def __getitem__(self, key):
        val = self.get(key, _SENTINEL)
        if val is _SENTINEL:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    def peek(self, key, default=None):
        """Get value without updating recency."""
        node = self._map.get(key)
        return node.value if node is not None else default

    def keys(self):
        """Return keys in MRU to LRU order."""
        return [n.key for n in self._list.items()]

    def values(self):
        """Return values in MRU to LRU order."""
        return [n.value for n in self._list.items()]

    def items(self):
        """Return (key, value) pairs in MRU to LRU order."""
        return [(n.key, n.value) for n in self._list.items()]

    def clear(self):
        """Remove all entries."""
        self._map.clear()
        self._list = _DoublyLinkedList()

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {
            'size': len(self._map),
            'capacity': self.capacity,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }

    def _evict(self):
        node = self._list.remove_last()
        if node is not None:
            del self._map[node.key]
            if self.on_evict:
                self.on_evict(node.key, node.value)


_SENTINEL = object()


# ============================================================
# 2. LFU Cache
# ============================================================

class LFUCache:
    """Least Frequently Used cache with O(1) get/put.

    Uses frequency buckets: each frequency maps to a doubly-linked list.
    On tie, evicts the least recently used among least-frequent entries.

    Args:
        capacity: Maximum number of entries.
        on_evict: Optional callback(key, value) called on eviction.
    """

    def __init__(self, capacity, on_evict=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.on_evict = on_evict
        self._map = {}          # key -> _Node
        self._freq_map = {}     # freq -> _DoublyLinkedList
        self._min_freq = 0
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            return default
        self._hits += 1
        self._touch(node)
        return node.value

    def put(self, key, value):
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._touch(node)
            return
        if len(self._map) >= self.capacity:
            self._evict()
        node = _Node(key, value)
        node.freq = 1
        self._map[key] = node
        self._get_freq_list(1).add_front(node)
        self._min_freq = 1

    def delete(self, key):
        node = self._map.pop(key, None)
        if node is None:
            return False
        flist = self._freq_map.get(node.freq)
        if flist:
            flist.remove(node)
            if len(flist) == 0:
                del self._freq_map[node.freq]
        return True

    def __contains__(self, key):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def peek(self, key, default=None):
        node = self._map.get(key)
        return node.value if node is not None else default

    def frequency(self, key):
        """Return access frequency for key, or 0 if not present."""
        node = self._map.get(key)
        return node.freq if node is not None else 0

    def clear(self):
        self._map.clear()
        self._freq_map.clear()
        self._min_freq = 0

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {
            'size': len(self._map),
            'capacity': self.capacity,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'min_freq': self._min_freq,
        }

    def _touch(self, node):
        """Increment frequency and move to new bucket."""
        old_freq = node.freq
        flist = self._freq_map.get(old_freq)
        if flist:
            flist.remove(node)
            if len(flist) == 0:
                del self._freq_map[old_freq]
                if self._min_freq == old_freq:
                    self._min_freq = old_freq + 1
        node.freq += 1
        self._get_freq_list(node.freq).add_front(node)

    def _get_freq_list(self, freq):
        if freq not in self._freq_map:
            self._freq_map[freq] = _DoublyLinkedList()
        return self._freq_map[freq]

    def _evict(self):
        flist = self._freq_map.get(self._min_freq)
        if flist is None:
            return
        node = flist.remove_last()
        if node is None:
            return
        if len(flist) == 0:
            del self._freq_map[self._min_freq]
        del self._map[node.key]
        if self.on_evict:
            self.on_evict(node.key, node.value)


# ============================================================
# 3. TTL Cache
# ============================================================

class TTLCache:
    """Cache with per-entry time-to-live expiration.

    Uses lazy expiration: entries are checked on access and periodically
    cleaned up. Built on LRU ordering for capacity eviction.

    Args:
        capacity: Maximum number of (non-expired) entries.
        default_ttl: Default TTL in seconds. None means no expiry.
        time_func: Function returning current time (for testing).
        on_evict: Optional callback(key, value) called on eviction/expiry.
    """

    def __init__(self, capacity, default_ttl=None, time_func=None, on_evict=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._time = time_func or _time_module.time
        self.on_evict = on_evict
        self._map = {}
        self._list = _DoublyLinkedList()
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            return default
        if self._is_expired(node):
            self._remove_node(node)
            self._misses += 1
            return default
        self._hits += 1
        self._list.move_to_front(node)
        return node.value

    def put(self, key, value, ttl=_SENTINEL):
        """Insert or update. ttl overrides default_ttl for this entry."""
        actual_ttl = self.default_ttl if ttl is _SENTINEL else ttl
        node = self._map.get(key)
        if node is not None:
            node.value = value
            node.expire_at = self._time() + actual_ttl if actual_ttl is not None else None
            self._list.move_to_front(node)
            return
        # Clean expired before checking capacity
        self._lazy_cleanup()
        if len(self._map) >= self.capacity:
            self._evict()
        node = _Node(key, value)
        node.expire_at = self._time() + actual_ttl if actual_ttl is not None else None
        self._map[key] = node
        self._list.add_front(node)

    def delete(self, key):
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._list.remove(node)
        return True

    def __contains__(self, key):
        node = self._map.get(key)
        if node is None:
            return False
        if self._is_expired(node):
            self._remove_node(node)
            return False
        return True

    def __len__(self):
        self._lazy_cleanup()
        return len(self._map)

    def ttl_remaining(self, key):
        """Return remaining TTL in seconds, or None if no expiry, or -1 if not found."""
        node = self._map.get(key)
        if node is None:
            return -1
        if self._is_expired(node):
            self._remove_node(node)
            return -1
        if node.expire_at is None:
            return None
        return max(0.0, node.expire_at - self._time())

    def clear(self):
        self._map.clear()
        self._list = _DoublyLinkedList()

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _is_expired(self, node):
        return node.expire_at is not None and self._time() >= node.expire_at

    def _remove_node(self, node):
        self._list.remove(node)
        del self._map[node.key]
        if self.on_evict:
            self.on_evict(node.key, node.value)

    def _lazy_cleanup(self):
        """Remove expired entries (scan from LRU end, stop early)."""
        to_remove = []
        for node in self._list.items_reverse():
            if self._is_expired(node):
                to_remove.append(node)
        for node in to_remove:
            self._remove_node(node)

    def _evict(self):
        node = self._list.remove_last()
        if node is not None:
            del self._map[node.key]
            if self.on_evict:
                self.on_evict(node.key, node.value)


# ============================================================
# 4. SLRU Cache (Segmented LRU)
# ============================================================

class SLRUCache:
    """Segmented LRU Cache with probation and protected segments.

    New entries go to probation. On second access, they promote to protected.
    When protected is full, its LRU victim demotes back to probation.
    Eviction always happens from probation's LRU end.

    Args:
        capacity: Total capacity (probation + protected).
        protected_ratio: Fraction of capacity for protected segment (default 0.8).
        on_evict: Optional callback(key, value) called on eviction.
    """

    def __init__(self, capacity, protected_ratio=0.8, on_evict=None):
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.capacity = capacity
        self.on_evict = on_evict
        self._protected_cap = max(1, int(capacity * protected_ratio))
        self._probation_cap = capacity - self._protected_cap
        if self._probation_cap < 1:
            self._probation_cap = 1
            self._protected_cap = capacity - 1

        self._map = {}  # key -> _Node
        self._probation = _DoublyLinkedList()
        self._protected = _DoublyLinkedList()
        # Track which segment each node is in
        self._in_protected = set()
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            return default
        self._hits += 1
        if key in self._in_protected:
            self._protected.move_to_front(node)
        else:
            # Promote from probation to protected
            self._probation.remove(node)
            self._promote_to_protected(node)
        return node.value

    def put(self, key, value):
        node = self._map.get(key)
        if node is not None:
            node.value = value
            if key in self._in_protected:
                self._protected.move_to_front(node)
            else:
                self._probation.remove(node)
                self._promote_to_protected(node)
            return
        # New entry -> probation
        if len(self._map) >= self.capacity:
            self._evict_probation()
        node = _Node(key, value)
        self._map[key] = node
        self._probation.add_front(node)

    def delete(self, key):
        node = self._map.pop(key, None)
        if node is None:
            return False
        if key in self._in_protected:
            self._protected.remove(node)
            self._in_protected.discard(key)
        else:
            self._probation.remove(node)
        return True

    def __contains__(self, key):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def segment_of(self, key):
        """Return 'protected', 'probation', or None."""
        if key not in self._map:
            return None
        return 'protected' if key in self._in_protected else 'probation'

    def clear(self):
        self._map.clear()
        self._in_protected.clear()
        self._probation = _DoublyLinkedList()
        self._protected = _DoublyLinkedList()

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {
            'size': len(self._map),
            'capacity': self.capacity,
            'probation_size': len(self._probation),
            'protected_size': len(self._protected),
            'protected_cap': self._protected_cap,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }

    def _promote_to_protected(self, node):
        """Move node into protected segment, possibly demoting protected LRU."""
        if len(self._protected) >= self._protected_cap:
            # Demote protected LRU back to probation
            demoted = self._protected.remove_last()
            if demoted is not None:
                self._in_protected.discard(demoted.key)
                self._probation.add_front(demoted)
                # If probation overflows, evict from probation
                if len(self._probation) > self._probation_cap:
                    self._evict_probation()
        self._protected.add_front(node)
        self._in_protected.add(node.key)

    def _evict_probation(self):
        node = self._probation.remove_last()
        if node is not None:
            del self._map[node.key]
            if self.on_evict:
                self.on_evict(node.key, node.value)


# ============================================================
# 5. ARC Cache (Adaptive Replacement Cache)
# ============================================================

class ARCCache:
    """Adaptive Replacement Cache -- self-tuning balance between recency and frequency.

    Maintains four lists:
    - T1: Recent entries (seen once)
    - T2: Frequent entries (seen 2+)
    - B1: Ghost entries evicted from T1 (keys only)
    - B2: Ghost entries evicted from T2 (keys only)

    Parameter p adapts: ghost hits on B1 increase p (favor recency),
    ghost hits on B2 decrease p (favor frequency).

    Args:
        capacity: Maximum number of cached entries (T1 + T2).
        on_evict: Optional callback(key, value) called on eviction.
    """

    def __init__(self, capacity, on_evict=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.on_evict = on_evict
        self._p = 0  # target size for T1

        # Real caches
        self._t1 = _DoublyLinkedList()
        self._t2 = _DoublyLinkedList()
        self._t1_map = {}  # key -> _Node
        self._t2_map = {}  # key -> _Node

        # Ghost caches (keys only -- store _Node with value=None)
        self._b1 = _DoublyLinkedList()
        self._b2 = _DoublyLinkedList()
        self._b1_map = {}
        self._b2_map = {}

        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        # Case 1: key in T1 -- move to T2 MRU
        if key in self._t1_map:
            self._hits += 1
            node = self._t1_map.pop(key)
            self._t1.remove(node)
            self._t2_map[key] = node
            self._t2.add_front(node)
            return node.value

        # Case 2: key in T2 -- move to T2 MRU
        if key in self._t2_map:
            self._hits += 1
            node = self._t2_map[key]
            self._t2.move_to_front(node)
            return node.value

        self._misses += 1
        return default

    def put(self, key, value):
        # Case 1: key in T1
        if key in self._t1_map:
            node = self._t1_map.pop(key)
            self._t1.remove(node)
            node.value = value
            self._t2_map[key] = node
            self._t2.add_front(node)
            return

        # Case 2: key in T2
        if key in self._t2_map:
            node = self._t2_map[key]
            node.value = value
            self._t2.move_to_front(node)
            return

        # Case 3: key in B1 (ghost hit -- adapt toward recency)
        if key in self._b1_map:
            delta = max(1, len(self._b2_map) // max(1, len(self._b1_map)))
            self._p = min(self._p + delta, self.capacity)
            self._replace(key)
            ghost = self._b1_map.pop(key)
            self._b1.remove(ghost)
            node = _Node(key, value)
            self._t2_map[key] = node
            self._t2.add_front(node)
            return

        # Case 4: key in B2 (ghost hit -- adapt toward frequency)
        if key in self._b2_map:
            delta = max(1, len(self._b1_map) // max(1, len(self._b2_map)))
            self._p = max(self._p - delta, 0)
            self._replace(key)
            ghost = self._b2_map.pop(key)
            self._b2.remove(ghost)
            node = _Node(key, value)
            self._t2_map[key] = node
            self._t2.add_front(node)
            return

        # Case 5: completely new key
        total_t = len(self._t1_map) + len(self._t2_map)
        total_all = total_t + len(self._b1_map) + len(self._b2_map)

        if total_t >= self.capacity:
            # Cache is full
            if len(self._t1_map) + len(self._b1_map) >= self.capacity:
                # B1 is too large, discard oldest B1
                if len(self._b1_map) > 0:
                    evicted = self._b1.remove_last()
                    if evicted:
                        self._b1_map.pop(evicted.key, None)
                self._replace(key)
            else:
                if total_all >= 2 * self.capacity:
                    # B2 is too large, discard oldest B2
                    if len(self._b2_map) > 0:
                        evicted = self._b2.remove_last()
                        if evicted:
                            self._b2_map.pop(evicted.key, None)
                self._replace(key)
        # No need to explicitly handle total_t < capacity -- just add

        node = _Node(key, value)
        self._t1_map[key] = node
        self._t1.add_front(node)

    def delete(self, key):
        if key in self._t1_map:
            node = self._t1_map.pop(key)
            self._t1.remove(node)
            return True
        if key in self._t2_map:
            node = self._t2_map.pop(key)
            self._t2.remove(node)
            return True
        return False

    def __contains__(self, key):
        return key in self._t1_map or key in self._t2_map

    def __len__(self):
        return len(self._t1_map) + len(self._t2_map)

    def clear(self):
        self._t1_map.clear()
        self._t2_map.clear()
        self._b1_map.clear()
        self._b2_map.clear()
        self._t1 = _DoublyLinkedList()
        self._t2 = _DoublyLinkedList()
        self._b1 = _DoublyLinkedList()
        self._b2 = _DoublyLinkedList()
        self._p = 0

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {
            'size': len(self),
            'capacity': self.capacity,
            't1_size': len(self._t1_map),
            't2_size': len(self._t2_map),
            'b1_size': len(self._b1_map),
            'b2_size': len(self._b2_map),
            'p': self._p,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }

    def _replace(self, key):
        """Evict one entry from T1 or T2 based on p."""
        if (len(self._t1_map) > 0 and
            (len(self._t1_map) > self._p or
             (key in self._b2_map and len(self._t1_map) == self._p))):
            # Evict from T1, add ghost to B1
            node = self._t1.remove_last()
            if node:
                self._t1_map.pop(node.key, None)
                if self.on_evict:
                    self.on_evict(node.key, node.value)
                ghost = _Node(node.key)
                self._b1_map[node.key] = ghost
                self._b1.add_front(ghost)
        elif len(self._t2_map) > 0:
            # Evict from T2, add ghost to B2
            node = self._t2.remove_last()
            if node:
                self._t2_map.pop(node.key, None)
                if self.on_evict:
                    self.on_evict(node.key, node.value)
                ghost = _Node(node.key)
                self._b2_map[node.key] = ghost
                self._b2.add_front(ghost)


# ============================================================
# 6. WriteBack Cache
# ============================================================

class WriteBackCache:
    """Write-back cache with dirty tracking and flush support.

    Writes are cached and marked dirty. Flushing writes dirty entries
    to a backing store via a write_func callback.

    Args:
        capacity: Maximum number of entries.
        write_func: Callback(key, value) to persist dirty entries.
        read_func: Optional callback(key) -> value to load from backing store on miss.
        on_evict: Optional callback(key, value) called after eviction (after flush).
    """

    def __init__(self, capacity, write_func=None, read_func=None, on_evict=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.write_func = write_func
        self.read_func = read_func
        self.on_evict = on_evict
        self._map = {}
        self._list = _DoublyLinkedList()
        self._dirty = set()
        self._hits = 0
        self._misses = 0
        self._writes_back = 0

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is not None:
            self._hits += 1
            self._list.move_to_front(node)
            return node.value
        # Try read_func
        if self.read_func is not None:
            try:
                value = self.read_func(key)
            except (KeyError, Exception):
                self._misses += 1
                return default
            self._misses += 1  # still a cache miss
            # Cache the loaded value (not dirty)
            self._insert(key, value, dirty=False)
            return value
        self._misses += 1
        return default

    def put(self, key, value):
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._dirty.add(key)
            self._list.move_to_front(node)
            return
        self._insert(key, value, dirty=True)

    def delete(self, key):
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._list.remove(node)
        # Flush if dirty before removing
        if key in self._dirty:
            self._flush_node(node)
            self._dirty.discard(key)
        return True

    def flush(self, key=None):
        """Flush dirty entries to backing store. If key given, flush only that key."""
        if key is not None:
            node = self._map.get(key)
            if node and key in self._dirty:
                self._flush_node(node)
                self._dirty.discard(key)
            return
        # Flush all dirty
        for k in list(self._dirty):
            node = self._map.get(k)
            if node:
                self._flush_node(node)
        self._dirty.clear()

    def flush_all(self):
        """Flush all dirty entries."""
        self.flush()

    def is_dirty(self, key):
        return key in self._dirty

    def dirty_keys(self):
        return set(self._dirty)

    def __contains__(self, key):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def clear(self, flush=True):
        if flush:
            self.flush()
        self._map.clear()
        self._list = _DoublyLinkedList()
        self._dirty.clear()

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self):
        return {
            'size': len(self._map),
            'capacity': self.capacity,
            'dirty_count': len(self._dirty),
            'writes_back': self._writes_back,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }

    def _insert(self, key, value, dirty=False):
        if len(self._map) >= self.capacity:
            self._evict()
        node = _Node(key, value)
        self._map[key] = node
        self._list.add_front(node)
        if dirty:
            self._dirty.add(key)

    def _flush_node(self, node):
        if self.write_func:
            self.write_func(node.key, node.value)
            self._writes_back += 1

    def _evict(self):
        node = self._list.remove_last()
        if node is not None:
            # Flush if dirty
            if node.key in self._dirty:
                self._flush_node(node)
                self._dirty.discard(node.key)
            del self._map[node.key]
            if self.on_evict:
                self.on_evict(node.key, node.value)


# ============================================================
# 7. MultiTier Cache
# ============================================================

class MultiTierCache:
    """Multi-level cache with inclusive/exclusive policies.

    Composes multiple cache instances into a hierarchy. On miss at level N,
    checks level N+1, ..., and optionally an origin function. Found values
    are promoted back to earlier levels.

    Args:
        levels: List of cache instances (L1, L2, ...). Each must support
                get(key, default), put(key, value), delete(key).
        origin: Optional callback(key) -> value for loading on total miss.
        inclusive: If True (default), found values are inserted into all
                   higher levels. If False, only into the requesting level.
    """

    def __init__(self, levels, origin=None, inclusive=True):
        if not levels:
            raise ValueError("at least one cache level required")
        self.levels = list(levels)
        self.origin = origin
        self.inclusive = inclusive
        self._hits = [0] * len(levels)
        self._misses = 0

    def get(self, key, default=None):
        for i, cache in enumerate(self.levels):
            value = cache.get(key, _SENTINEL)
            if value is not _SENTINEL:
                self._hits[i] += 1
                # Promote to higher levels
                if self.inclusive and i > 0:
                    for j in range(i):
                        self.levels[j].put(key, value)
                return value
        # Total miss -- try origin
        if self.origin is not None:
            try:
                value = self.origin(key)
            except (KeyError, Exception):
                self._misses += 1
                return default
            self._misses += 1  # still a miss from cache perspective
            # Insert into all levels
            for cache in self.levels:
                cache.put(key, value)
            return value
        self._misses += 1
        return default

    def put(self, key, value):
        """Write to all levels."""
        for cache in self.levels:
            cache.put(key, value)

    def delete(self, key):
        """Delete from all levels."""
        deleted = False
        for cache in self.levels:
            if cache.delete(key):
                deleted = True
        return deleted

    def invalidate(self, key):
        """Alias for delete -- invalidate across all levels."""
        return self.delete(key)

    def __contains__(self, key):
        return any(key in cache for cache in self.levels)

    def __len__(self):
        # Return L1 size as the primary size indicator
        return len(self.levels[0])

    def clear(self):
        for cache in self.levels:
            cache.clear()

    def stats(self):
        return {
            'levels': len(self.levels),
            'hits_per_level': list(self._hits),
            'total_hits': sum(self._hits),
            'misses': self._misses,
            'hit_rate': sum(self._hits) / max(1, sum(self._hits) + self._misses),
        }
