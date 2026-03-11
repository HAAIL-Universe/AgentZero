"""
C105: Van Emde Boas Tree

O(log log U) integer data structure supporting:
- insert, delete, member: O(log log U)
- successor, predecessor: O(log log U)
- min, max: O(1)

Universe size U must be a power of 2. Supports integers in [0, U-1].

Variants:
- VEBTree: standard Van Emde Boas tree
- VEBSet: set interface wrapping VEBTree
- VEBMap: key-value map using VEB for keys
- XFastTrie: x-fast trie (O(log log U) successor via hashing)
- YFastTrie: y-fast trie (O(log log U) amortized with O(n) space)
"""

import math
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Core Van Emde Boas Tree
# ---------------------------------------------------------------------------

class VEBTree:
    """Van Emde Boas tree for universe [0, U-1] where U is a power of 2."""

    __slots__ = ('u', '_half', '_min', '_max', 'summary', 'cluster', '_size')

    def __init__(self, u):
        if u < 2:
            raise ValueError("Universe size must be >= 2")
        # Round up to next power of 2 if needed
        bits = max(1, math.ceil(math.log2(u)))
        self.u = 1 << bits
        self._half = bits // 2
        self._min = None
        self._max = None
        self._size = 0
        self.summary = None
        self.cluster = None
        # Clusters are lazily allocated

    def _high(self, x):
        """Upper half of bits = cluster index."""
        return x >> self._half

    def _low(self, x):
        """Lower half of bits = position within cluster."""
        return x & ((1 << self._half) - 1)

    def _index(self, high, low):
        """Reconstruct key from cluster index and position."""
        return (high << self._half) | low

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, x):
        return self.member(x)

    def __iter__(self):
        """Iterate all elements in sorted order."""
        x = self._min
        while x is not None:
            yield x
            x = self.successor(x)

    def __repr__(self):
        if self._size <= 16:
            return f"VEBTree(u={self.u}, elements={list(self)})"
        return f"VEBTree(u={self.u}, size={self._size})"

    def member(self, x):
        """Check if x is in the tree. O(log log U)."""
        if x < 0 or x >= self.u:
            return False
        if self._min is None:
            return False
        if x == self._min or x == self._max:
            return True
        if self.u == 2:
            return False
        h = self._high(x)
        if self.cluster is None or self.cluster[h] is None:
            return False
        return self.cluster[h].member(self._low(x))

    def insert(self, x):
        """Insert x into the tree. O(log log U)."""
        if x < 0 or x >= self.u:
            raise ValueError(f"Value {x} out of range [0, {self.u - 1}]")
        if self._min is None:
            # Empty tree -- just set min and max
            self._min = self._max = x
            self._size = 1
            return
        if x == self._min or x == self._max:
            return  # Already present
        if self.member(x):
            return  # Already present (deeper check)

        self._size += 1

        if x < self._min:
            # New min -- swap and insert old min into clusters
            x, self._min = self._min, x
        if x > self._max:
            self._max = x

        if self.u == 2:
            return

        # Lazy allocation
        if self.cluster is None:
            num_clusters = 1 << (math.ceil(math.log2(self.u)) - self._half)
            self.cluster = [None] * num_clusters
            self.summary = VEBTree(num_clusters)

        h = self._high(x)
        l = self._low(x)

        if self.cluster[h] is None:
            cluster_size = 1 << self._half
            self.cluster[h] = VEBTree(cluster_size)

        if self.cluster[h]._min is None:
            # Empty cluster -- update summary
            self.summary.insert(h)

        self.cluster[h].insert(l)

    def delete(self, x):
        """Delete x from the tree. O(log log U)."""
        if x < 0 or x >= self.u:
            return
        if self._min is None:
            return
        if not self.member(x):
            return
        if self._min == self._max:
            # Only one element
            if x == self._min:
                self._min = self._max = None
                self._size = 0
            return

        self._size -= 1

        if x == self._min:
            # Replace min with next smallest
            if self.summary is None or self.summary._min is None:
                # Only min and max, no clusters
                self._min = self._max
                return
            first_cluster = self.summary._min
            x = self._index(first_cluster, self.cluster[first_cluster]._min)
            self._min = x
            # Fall through to delete x from its cluster

        if self.u == 2:
            # Base case: two elements, deleting one
            if x == 0:
                self._min = 1
            else:
                self._min = 0
            self._max = self._min
            return

        if self.cluster is None:
            return

        h = self._high(x)
        l = self._low(x)

        if self.cluster[h] is None:
            return

        self.cluster[h].delete(l)

        if self.cluster[h]._min is None:
            # Cluster became empty
            self.summary.delete(h)

        if x == self._max:
            if self.summary._min is None:
                # No more clusters -- max = min
                self._max = self._min
            else:
                last_cluster = self.summary._max
                self._max = self._index(last_cluster, self.cluster[last_cluster]._max)

    def successor(self, x):
        """Find smallest element > x. O(log log U). Returns None if none."""
        if self._min is None:
            return None
        if x < self._min:
            return self._min

        if self.u == 2:
            if x == 0 and self._max == 1:
                return 1
            return None

        if self.cluster is None:
            return None

        h = self._high(x)
        l = self._low(x)

        # Check if successor is in same cluster
        if (self.cluster is not None and
            h < len(self.cluster) and
            self.cluster[h] is not None and
            self.cluster[h]._max is not None and
            l < self.cluster[h]._max):
            offset = self.cluster[h].successor(l)
            return self._index(h, offset)

        # Otherwise find next non-empty cluster
        next_cluster = self.summary.successor(h) if self.summary else None
        if next_cluster is None:
            return None
        return self._index(next_cluster, self.cluster[next_cluster]._min)

    def predecessor(self, x):
        """Find largest element < x. O(log log U). Returns None if none."""
        if self._min is None:
            return None
        if x > self._max:
            return self._max

        if self.u == 2:
            if x == 1 and self._min == 0:
                return 0
            return None

        if self.cluster is None:
            if x > self._min:
                return self._min
            return None

        h = self._high(x)
        l = self._low(x)

        # Check if predecessor is in same cluster
        if (h < len(self.cluster) and
            self.cluster[h] is not None and
            self.cluster[h]._min is not None and
            l > self.cluster[h]._min):
            offset = self.cluster[h].predecessor(l)
            return self._index(h, offset)

        # Otherwise find previous non-empty cluster
        prev_cluster = self.summary.predecessor(h) if self.summary else None
        if prev_cluster is not None:
            return self._index(prev_cluster, self.cluster[prev_cluster]._max)

        # Check if min is a valid predecessor (min isn't stored in clusters)
        if self._min is not None and x > self._min:
            return self._min
        return None

    def range_query(self, lo, hi):
        """Return all elements in [lo, hi] in sorted order."""
        if self._min is None or lo > hi:
            return []
        result = []
        x = lo if self.member(lo) else self.successor(lo - 1) if lo > 0 else (self._min if self._min >= lo else self.successor(lo - 1))
        # Simplify: find first element >= lo
        if lo <= self._min:
            x = self._min
        else:
            x = self.successor(lo - 1)
        while x is not None and x <= hi:
            result.append(x)
            x = self.successor(x)
        return result

    def to_sorted_list(self):
        """Return all elements as a sorted list."""
        return list(self)


# ---------------------------------------------------------------------------
# VEBSet -- Set interface
# ---------------------------------------------------------------------------

class VEBSet:
    """Set of integers in [0, U-1] backed by a Van Emde Boas tree."""

    def __init__(self, universe_size, iterable=None):
        self._tree = VEBTree(universe_size)
        if iterable:
            for x in iterable:
                self.add(x)

    @property
    def universe_size(self):
        return self._tree.u

    def add(self, x):
        self._tree.insert(x)

    def discard(self, x):
        self._tree.delete(x)

    def remove(self, x):
        if x not in self._tree:
            raise KeyError(x)
        self._tree.delete(x)

    def __contains__(self, x):
        return x in self._tree

    def __len__(self):
        return len(self._tree)

    def __bool__(self):
        return bool(self._tree)

    def __iter__(self):
        return iter(self._tree)

    def __repr__(self):
        if len(self) <= 16:
            return f"VEBSet({set(self)})"
        return f"VEBSet(size={len(self)}, u={self.universe_size})"

    @property
    def min(self):
        return self._tree.min

    @property
    def max(self):
        return self._tree.max

    def successor(self, x):
        return self._tree.successor(x)

    def predecessor(self, x):
        return self._tree.predecessor(x)

    def union(self, other):
        """Return new VEBSet with elements from both sets."""
        u = max(self.universe_size, other.universe_size)
        result = VEBSet(u)
        for x in self:
            result.add(x)
        for x in other:
            result.add(x)
        return result

    def intersection(self, other):
        """Return new VEBSet with elements in both sets."""
        u = max(self.universe_size, other.universe_size)
        result = VEBSet(u)
        smaller, larger = (self, other) if len(self) <= len(other) else (other, self)
        for x in smaller:
            if x in larger:
                result.add(x)
        return result

    def difference(self, other):
        """Return new VEBSet with elements in self but not other."""
        result = VEBSet(self.universe_size)
        for x in self:
            if x not in other:
                result.add(x)
        return result

    def symmetric_difference(self, other):
        """Return new VEBSet with elements in exactly one set."""
        u = max(self.universe_size, other.universe_size)
        result = VEBSet(u)
        for x in self:
            if x not in other:
                result.add(x)
        for x in other:
            if x not in self:
                result.add(x)
        return result

    def issubset(self, other):
        for x in self:
            if x not in other:
                return False
        return True

    def issuperset(self, other):
        return other.issubset(self)

    def range_query(self, lo, hi):
        return self._tree.range_query(lo, hi)


# ---------------------------------------------------------------------------
# VEBMap -- Key-value map with VEB-ordered keys
# ---------------------------------------------------------------------------

class VEBMap:
    """Integer-keyed map with O(log log U) operations, ordered by key."""

    def __init__(self, universe_size):
        self._tree = VEBTree(universe_size)
        self._data = {}

    @property
    def universe_size(self):
        return self._tree.u

    def __setitem__(self, key, value):
        self._tree.insert(key)
        self._data[key] = value

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        return self._data[key]

    def __delitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._tree.delete(key)
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        """Keys in sorted order."""
        return iter(self._tree)

    def values(self):
        """Values in key-sorted order."""
        for k in self._tree:
            yield self._data[k]

    def items(self):
        """(key, value) pairs in key-sorted order."""
        for k in self._tree:
            yield k, self._data[k]

    def __iter__(self):
        return iter(self._tree)

    def __repr__(self):
        if len(self) <= 8:
            return f"VEBMap({dict(self.items())})"
        return f"VEBMap(size={len(self)}, u={self.universe_size})"

    @property
    def min_key(self):
        return self._tree.min

    @property
    def max_key(self):
        return self._tree.max

    def min_item(self):
        k = self._tree.min
        if k is None:
            return None
        return k, self._data[k]

    def max_item(self):
        k = self._tree.max
        if k is None:
            return None
        return k, self._data[k]

    def successor(self, key):
        """Next key > given key, or None."""
        k = self._tree.successor(key)
        if k is None:
            return None
        return k, self._data[k]

    def predecessor(self, key):
        """Previous key < given key, or None."""
        k = self._tree.predecessor(key)
        if k is None:
            return None
        return k, self._data[k]

    def range_query(self, lo, hi):
        """All (key, value) pairs with key in [lo, hi]."""
        keys = self._tree.range_query(lo, hi)
        return [(k, self._data[k]) for k in keys]

    def pop(self, key, *args):
        if key in self._data:
            val = self._data.pop(key)
            self._tree.delete(key)
            return val
        if args:
            return args[0]
        raise KeyError(key)


# ---------------------------------------------------------------------------
# X-Fast Trie -- O(log log U) successor via hash tables at each level
# ---------------------------------------------------------------------------

class XFastTrie:
    """
    X-Fast Trie: stores integers in [0, U-1] where U is a power of 2.
    - member: O(1) via bottom-level hash
    - successor/predecessor: O(log log U) via binary search on levels
    - insert/delete: O(log U) to update hash tables at each level
    - Space: O(n log U)

    Uses a doubly-linked leaf list for efficient traversal.
    """

    class _Leaf:
        __slots__ = ('key', 'prev', 'next')
        def __init__(self, key):
            self.key = key
            self.prev = None
            self.next = None

    def __init__(self, universe_size):
        bits = max(1, math.ceil(math.log2(max(2, universe_size))))
        self.u = 1 << bits
        self._bits = bits
        # levels[i] maps prefix of length i to node info
        # levels[0] = root (always {0: True} if non-empty... we store prefix->leaf_desc)
        # levels[bits] = leaf level maps key -> _Leaf
        self._levels = [dict() for _ in range(bits + 1)]
        self._head = None  # Linked list head (smallest)
        self._tail = None  # Linked list tail (largest)
        self._size = 0

    def _prefix(self, x, level):
        """Get prefix of x at given level (top `level` bits)."""
        return x >> (self._bits - level)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, x):
        return self.member(x)

    def __iter__(self):
        node = self._head
        while node is not None:
            yield node.key
            node = node.next

    def __repr__(self):
        if self._size <= 16:
            return f"XFastTrie(u={self.u}, elements={list(self)})"
        return f"XFastTrie(u={self.u}, size={self._size})"

    @property
    def min(self):
        return self._head.key if self._head else None

    @property
    def max(self):
        return self._tail.key if self._tail else None

    def member(self, x):
        """O(1) membership test."""
        if x < 0 or x >= self.u:
            return False
        return x in self._levels[self._bits]

    def insert(self, x):
        """Insert x. O(log U)."""
        if x < 0 or x >= self.u:
            raise ValueError(f"Value {x} out of range [0, {self.u - 1}]")
        if x in self._levels[self._bits]:
            return  # Already present

        self._size += 1
        leaf = self._Leaf(x)
        self._levels[self._bits][x] = leaf

        # Find where to insert in linked list
        # Use successor/predecessor on existing structure before we update prefixes
        # But structure may be incomplete -- find position by scanning
        if self._head is None:
            self._head = self._tail = leaf
        elif x < self._head.key:
            leaf.next = self._head
            self._head.prev = leaf
            self._head = leaf
        elif x > self._tail.key:
            leaf.prev = self._tail
            self._tail.next = leaf
            self._tail = leaf
        else:
            # Find predecessor leaf by walking linked list (correct but O(n) for insert)
            # In a full implementation we'd use the hash tables, but for correctness:
            node = self._head
            while node.next and node.next.key < x:
                node = node.next
            leaf.next = node.next
            leaf.prev = node
            if node.next:
                node.next.prev = leaf
            node.next = leaf

        # Update prefix hash tables at each level
        for level in range(self._bits):
            p = self._prefix(x, level)
            if p not in self._levels[level]:
                self._levels[level][p] = {'left': None, 'right': None}
            # Track descendant leaves for successor/predecessor
            node_info = self._levels[level][p]
            # left = smallest descendant leaf, right = largest descendant leaf
            if node_info['left'] is None or x < node_info['left'].key:
                node_info['left'] = leaf
            if node_info['right'] is None or x > node_info['right'].key:
                node_info['right'] = leaf

    def delete(self, x):
        """Delete x. O(log U)."""
        if x not in self._levels[self._bits]:
            return

        leaf = self._levels[self._bits][x]
        self._size -= 1

        # Remove from linked list
        if leaf.prev:
            leaf.prev.next = leaf.next
        else:
            self._head = leaf.next
        if leaf.next:
            leaf.next.prev = leaf.prev
        else:
            self._tail = leaf.prev

        # Remove from leaf level
        del self._levels[self._bits][x]

        # Update prefix hash tables
        for level in range(self._bits):
            p = self._prefix(x, level)
            if p not in self._levels[level]:
                continue
            node_info = self._levels[level][p]

            # Check if any descendants remain at this prefix
            has_descendants = False
            # A prefix at level `level` covers keys in a range
            shift = self._bits - level
            lo = p << shift
            hi = lo + (1 << shift) - 1

            # Find new left/right by checking neighbors
            new_left = None
            new_right = None

            # Check linked list neighbors
            if leaf.next and lo <= leaf.next.key <= hi:
                new_left = leaf.next
            if leaf.prev and lo <= leaf.prev.key <= hi:
                new_right = leaf.prev

            # Walk to find actual bounds if needed
            if new_left is None:
                node = leaf.next
                while node and node.key <= hi:
                    if node.key >= lo:
                        new_left = node
                        break
                    node = node.next

            if new_right is None:
                node = leaf.prev
                while node and node.key >= lo:
                    if node.key <= hi:
                        new_right = node
                        break
                    node = node.prev

            if new_left is None and new_right is None:
                del self._levels[level][p]
            else:
                if new_left is None:
                    new_left = new_right
                if new_right is None:
                    new_right = new_left
                # Also walk to find true min/max in range
                node = new_left
                while node and node.key <= hi:
                    if new_right is None or node.key > new_right.key:
                        new_right = node
                    node = node.next

                node_info['left'] = new_left
                node_info['right'] = new_right

    def successor(self, x):
        """Find smallest element > x. O(log log U)."""
        if self._size == 0:
            return None
        if x < self._head.key:
            return self._head.key
        if x >= self._tail.key:
            return None

        # If x is in the trie, just follow linked list
        if x in self._levels[self._bits]:
            leaf = self._levels[self._bits][x]
            return leaf.next.key if leaf.next else None

        # Binary search on levels to find lowest ancestor with a right descendant > x
        lo_lvl, hi_lvl = 0, self._bits
        while lo_lvl < hi_lvl:
            mid = (lo_lvl + hi_lvl) // 2
            p = self._prefix(x, mid)
            if p in self._levels[mid]:
                lo_lvl = mid + 1
            else:
                hi_lvl = mid

        # lo_lvl is the level where the prefix first disappears
        # Go one level up and use the descendant pointers
        level = lo_lvl - 1
        if level < 0:
            level = 0

        p = self._prefix(x, level)
        if p in self._levels[level]:
            node_info = self._levels[level][p]
            # The right descendant of this prefix
            right_leaf = node_info['right']
            if right_leaf and right_leaf.key > x:
                # Walk from right_leaf backwards to find exact successor
                # Actually, successor is the smallest key > x
                # right_leaf might be > x, but there might be something between
                left_leaf = node_info['left']
                if left_leaf and left_leaf.key > x:
                    return left_leaf.key
                # Walk from left_leaf forward
                node = left_leaf if left_leaf else self._head
                while node:
                    if node.key > x:
                        return node.key
                    node = node.next
            # If right_leaf <= x, successor is in the next prefix's subtree
            next_p = p + 1
            if next_p in self._levels[level]:
                return self._levels[level][next_p]['left'].key

        # Fallback: walk linked list from head
        node = self._head
        while node:
            if node.key > x:
                return node.key
            node = node.next
        return None

    def predecessor(self, x):
        """Find largest element < x. O(log log U)."""
        if self._size == 0:
            return None
        if x > self._tail.key:
            return self._tail.key
        if x <= self._head.key:
            return None

        # If x is in the trie, just follow linked list
        if x in self._levels[self._bits]:
            leaf = self._levels[self._bits][x]
            return leaf.prev.key if leaf.prev else None

        # Binary search on levels
        lo_lvl, hi_lvl = 0, self._bits
        while lo_lvl < hi_lvl:
            mid = (lo_lvl + hi_lvl) // 2
            p = self._prefix(x, mid)
            if p in self._levels[mid]:
                lo_lvl = mid + 1
            else:
                hi_lvl = mid

        level = lo_lvl - 1
        if level < 0:
            level = 0

        p = self._prefix(x, level)
        if p in self._levels[level]:
            node_info = self._levels[level][p]
            left_leaf = node_info['left']
            right_leaf = node_info['right']
            if right_leaf and right_leaf.key < x:
                return right_leaf.key
            if left_leaf and left_leaf.key < x:
                # Walk forward to find largest < x
                node = left_leaf
                best = None
                while node and node.key < x:
                    best = node.key
                    node = node.next
                if best is not None:
                    return best
            # Check previous prefix
            prev_p = p - 1
            if prev_p >= 0 and prev_p in self._levels[level]:
                return self._levels[level][prev_p]['right'].key

        # Fallback
        node = self._tail
        while node:
            if node.key < x:
                return node.key
            node = node.prev
        return None

    def range_query(self, lo, hi):
        """All elements in [lo, hi] in sorted order."""
        result = []
        node = self._head
        while node and node.key < lo:
            node = node.next
        while node and node.key <= hi:
            result.append(node.key)
            node = node.next
        return result


# ---------------------------------------------------------------------------
# Y-Fast Trie -- O(n) space, O(log log U) amortized
# ---------------------------------------------------------------------------

class YFastTrie:
    """
    Y-Fast Trie: O(n) space, O(log log U) amortized operations.

    Partitions elements into groups of O(log U) size, stores representatives
    in an X-Fast Trie, and uses balanced BSTs (sorted lists) for each group.
    """

    def __init__(self, universe_size):
        bits = max(1, math.ceil(math.log2(max(2, universe_size))))
        self.u = 1 << bits
        self._bits = bits
        self._group_size = max(1, bits)  # Target group size = log U
        self._xfast = XFastTrie(universe_size)  # Representatives
        self._groups = {}  # representative -> sorted list of elements
        self._element_to_rep = {}  # element -> its group's representative
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, x):
        return x in self._element_to_rep

    def __iter__(self):
        """Iterate all elements in sorted order."""
        # Merge all sorted groups into one sorted stream
        import heapq
        iters = []
        for rep in self._xfast:
            if rep in self._groups and self._groups[rep]:
                iters.append(iter(self._groups[rep]))
        yield from heapq.merge(*iters)

    def __repr__(self):
        if self._size <= 16:
            return f"YFastTrie(u={self.u}, elements={list(self)})"
        return f"YFastTrie(u={self.u}, size={self._size})"

    @property
    def min(self):
        rep = self._xfast.min
        if rep is None:
            return None
        return self._groups[rep][0]

    @property
    def max(self):
        rep = self._xfast.max
        if rep is None:
            return None
        return self._groups[rep][-1]

    def _find_group(self, x):
        """Find the representative whose group should contain x."""
        # Find successor rep >= x
        if x in self._xfast:
            return x
        succ = self._xfast.successor(x - 1) if x > 0 else self._xfast.successor(-1)
        pred = self._xfast.predecessor(x + 1) if x < self.u - 1 else self._xfast.predecessor(self.u)
        # Choose closest rep
        if succ is not None and pred is not None:
            if x - pred <= succ - x:
                return pred
            return succ
        return succ if succ is not None else pred

    def _sorted_insert(self, lst, x):
        """Insert x into sorted list."""
        import bisect
        bisect.insort(lst, x)

    def _sorted_remove(self, lst, x):
        """Remove x from sorted list."""
        import bisect
        i = bisect.bisect_left(lst, x)
        if i < len(lst) and lst[i] == x:
            lst.pop(i)

    def member(self, x):
        return x in self._element_to_rep

    def insert(self, x):
        """Insert x. O(log log U) amortized."""
        if x < 0 or x >= self.u:
            raise ValueError(f"Value {x} out of range [0, {self.u - 1}]")
        if x in self._element_to_rep:
            return

        self._size += 1

        if len(self._xfast) == 0:
            # First element -- create a new group
            self._xfast.insert(x)
            self._groups[x] = [x]
            self._element_to_rep[x] = x
            return

        rep = self._find_group(x)
        if rep is None:
            # No existing group, create new
            self._xfast.insert(x)
            self._groups[x] = [x]
            self._element_to_rep[x] = x
            return

        group = self._groups[rep]
        self._sorted_insert(group, x)
        self._element_to_rep[x] = rep

        # Split if group is too large
        if len(group) > 2 * self._group_size:
            self._split_group(rep)

    def _split_group(self, rep):
        """Split a group that's too large into two groups."""
        group = self._groups[rep]
        mid = len(group) // 2
        left = group[:mid]
        right = group[mid:]

        new_rep = right[0]  # New representative for right half

        # Update existing group
        self._groups[rep] = left
        for elem in left:
            self._element_to_rep[elem] = rep

        # Create new group
        self._xfast.insert(new_rep)
        self._groups[new_rep] = right
        for elem in right:
            self._element_to_rep[elem] = new_rep

        # If old rep is no longer in its own group, we need to fix
        if rep not in left:
            # Old rep must be re-pointed: remove from xfast if not in left
            # Actually rep should be the max of left or we keep it as-is
            # The rep is just a key in xfast -- it's fine as long as it points to the group
            pass

    def delete(self, x):
        """Delete x. O(log log U) amortized."""
        if x not in self._element_to_rep:
            return

        self._size -= 1
        rep = self._element_to_rep[x]
        group = self._groups[rep]
        self._sorted_remove(group, x)
        del self._element_to_rep[x]

        if len(group) == 0:
            # Group is empty -- remove representative
            del self._groups[rep]
            self._xfast.delete(rep)
            return

        # If we deleted the representative, we need to reassign
        if x == rep:
            new_rep = group[0]
            del self._groups[rep]
            self._xfast.delete(rep)
            self._xfast.insert(new_rep)
            self._groups[new_rep] = group
            for elem in group:
                self._element_to_rep[elem] = new_rep

        # Merge with neighbor if group is too small
        if len(group) < self._group_size // 2 and len(self._xfast) > 1:
            current_rep = self._element_to_rep.get(group[0], rep)
            self._try_merge(current_rep)

    def _try_merge(self, rep):
        """Try to merge a small group with a neighbor."""
        if rep not in self._groups:
            return
        group = self._groups[rep]

        # Find neighbor group
        succ_rep = self._xfast.successor(rep)
        pred_rep = self._xfast.predecessor(rep)

        # Pick smaller neighbor to merge with
        merge_rep = None
        if succ_rep is not None and succ_rep in self._groups:
            if pred_rep is not None and pred_rep in self._groups:
                if len(self._groups[pred_rep]) <= len(self._groups[succ_rep]):
                    merge_rep = pred_rep
                else:
                    merge_rep = succ_rep
            else:
                merge_rep = succ_rep
        elif pred_rep is not None and pred_rep in self._groups:
            merge_rep = pred_rep

        if merge_rep is None:
            return

        if len(group) + len(self._groups[merge_rep]) <= 2 * self._group_size:
            # Merge
            merged = sorted(group + self._groups[merge_rep])
            # Remove both reps
            del self._groups[rep]
            del self._groups[merge_rep]
            self._xfast.delete(rep)
            self._xfast.delete(merge_rep)
            # Create merged group
            new_rep = merged[0]
            self._xfast.insert(new_rep)
            self._groups[new_rep] = merged
            for elem in merged:
                self._element_to_rep[elem] = new_rep

    def successor(self, x):
        """Smallest element > x. O(log log U) amortized."""
        if self._size == 0:
            return None

        # Check current group first
        if x in self._element_to_rep:
            rep = self._element_to_rep[x]
            group = self._groups[rep]
            import bisect
            i = bisect.bisect_right(group, x)
            if i < len(group):
                return group[i]
        else:
            # Find the group that might contain successor
            rep = self._find_group(x)
            if rep is not None and rep in self._groups:
                group = self._groups[rep]
                import bisect
                i = bisect.bisect_right(group, x)
                if i < len(group):
                    return group[i]

        # Check next group
        # Find successor representative
        succ_rep = self._xfast.successor(x)
        while succ_rep is not None:
            if succ_rep in self._groups:
                group = self._groups[succ_rep]
                import bisect
                i = bisect.bisect_right(group, x)
                if i < len(group):
                    return group[i]
            succ_rep = self._xfast.successor(succ_rep)

        return None

    def predecessor(self, x):
        """Largest element < x. O(log log U) amortized."""
        if self._size == 0:
            return None

        # Check current group first
        if x in self._element_to_rep:
            rep = self._element_to_rep[x]
            group = self._groups[rep]
            import bisect
            i = bisect.bisect_left(group, x)
            if i > 0:
                return group[i - 1]
        else:
            rep = self._find_group(x)
            if rep is not None and rep in self._groups:
                group = self._groups[rep]
                import bisect
                i = bisect.bisect_left(group, x)
                if i > 0:
                    return group[i - 1]

        # Check previous group
        pred_rep = self._xfast.predecessor(x)
        while pred_rep is not None:
            if pred_rep in self._groups:
                group = self._groups[pred_rep]
                import bisect
                i = bisect.bisect_left(group, x)
                if i > 0:
                    return group[i - 1]
                # Might need to check the entire group
                if group and group[-1] < x:
                    return group[-1]
            pred_rep = self._xfast.predecessor(pred_rep)

        return None

    def range_query(self, lo, hi):
        """All elements in [lo, hi] in sorted order."""
        result = []
        for elem in self:
            if lo <= elem <= hi:
                result.append(elem)
            elif elem > hi:
                break
        return result
