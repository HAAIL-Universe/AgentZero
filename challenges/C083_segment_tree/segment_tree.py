"""
C083: Segment Tree with Lazy Propagation

A versatile range query data structure supporting:
- Range queries: sum, min, max, gcd, product (O(log n))
- Range updates: set, add (O(log n) with lazy propagation)
- Point queries and updates (O(log n))
- Persistent segment tree (path-copying, O(log n) per operation)
- 2D segment tree for rectangle queries
- Merge sort tree for order-statistic range queries
- Beats tree (Segment Tree Beats) for range chmin/chmax

Standalone implementation -- no dependencies on other challenges.
"""

from math import gcd, inf
from functools import reduce


# ---------------------------------------------------------------------------
# Monoid definitions for segment tree nodes
# ---------------------------------------------------------------------------

class Monoid:
    """Base class for segment tree monoids (combine + identity)."""
    @staticmethod
    def identity():
        raise NotImplementedError
    @staticmethod
    def combine(a, b):
        raise NotImplementedError


class SumMonoid(Monoid):
    @staticmethod
    def identity():
        return 0
    @staticmethod
    def combine(a, b):
        return a + b


class MinMonoid(Monoid):
    @staticmethod
    def identity():
        return inf
    @staticmethod
    def combine(a, b):
        return min(a, b)


class MaxMonoid(Monoid):
    @staticmethod
    def identity():
        return -inf
    @staticmethod
    def combine(a, b):
        return max(a, b)


class GCDMonoid(Monoid):
    @staticmethod
    def identity():
        return 0
    @staticmethod
    def combine(a, b):
        return gcd(a, b)


class ProductMonoid(Monoid):
    @staticmethod
    def identity():
        return 1
    @staticmethod
    def combine(a, b):
        return a * b


# ---------------------------------------------------------------------------
# Lazy operation definitions
# ---------------------------------------------------------------------------

class LazyOp:
    """Base class for lazy propagation operations."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        """Apply lazy value to a node covering 'count' elements."""
        raise NotImplementedError
    @staticmethod
    def compose(outer, inner):
        """Compose two lazy values (outer applied after inner)."""
        raise NotImplementedError


class AddLazy(LazyOp):
    """Lazy add: adds a value to all elements in range."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return node_val + lazy_val * count
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        if inner is None:
            return outer
        return outer + inner


class SetLazy(LazyOp):
    """Lazy set: sets all elements in range to a value."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return lazy_val * count
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        return outer


class AddLazyMin(LazyOp):
    """Lazy add for min-monoid trees (no count multiplication)."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return node_val + lazy_val
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        if inner is None:
            return outer
        return outer + inner


class SetLazyMin(LazyOp):
    """Lazy set for min-monoid trees (no count multiplication)."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return lazy_val
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        return outer


class SetLazyMax(LazyOp):
    """Lazy set for max-monoid trees."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return lazy_val
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        return outer


class AddLazyMax(LazyOp):
    """Lazy add for max-monoid trees."""
    @staticmethod
    def identity():
        return None
    @staticmethod
    def apply_to_node(lazy_val, node_val, count):
        if lazy_val is None:
            return node_val
        return node_val + lazy_val
    @staticmethod
    def compose(outer, inner):
        if outer is None:
            return inner
        if inner is None:
            return outer
        return outer + inner


# ---------------------------------------------------------------------------
# Segment Tree with Lazy Propagation
# ---------------------------------------------------------------------------

class SegmentTree:
    """
    Array-based segment tree with lazy propagation.

    Supports arbitrary monoids for queries and lazy operations for updates.
    Default: sum queries with add-range updates.
    """

    def __init__(self, data, monoid=None, lazy_op=None):
        """Build segment tree from initial data array."""
        if monoid is None:
            monoid = SumMonoid
        if lazy_op is None:
            lazy_op = AddLazy

        self._monoid = monoid
        self._lazy_op = lazy_op
        self._n = len(data)

        if self._n == 0:
            self._tree = []
            self._lazy = []
            return

        # Tree array: 1-indexed, size 4*n for safety
        size = 4 * self._n
        self._tree = [monoid.identity()] * size
        self._lazy = [lazy_op.identity()] * size

        self._build(data, 1, 0, self._n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self._tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self._tree[node] = self._monoid.combine(
            self._tree[2 * node], self._tree[2 * node + 1]
        )

    def _push_down(self, node, start, end):
        """Push lazy value to children."""
        if self._lazy[node] is not None:
            mid = (start + end) // 2
            self._apply_lazy(2 * node, start, mid, self._lazy[node])
            self._apply_lazy(2 * node + 1, mid + 1, end, self._lazy[node])
            self._lazy[node] = self._lazy_op.identity()

    def _apply_lazy(self, node, start, end, val):
        """Apply a lazy value to a node."""
        count = end - start + 1
        self._tree[node] = self._lazy_op.apply_to_node(val, self._tree[node], count)
        self._lazy[node] = self._lazy_op.compose(val, self._lazy[node])

    def update_range(self, l, r, val):
        """Apply lazy operation to all elements in [l, r]."""
        if self._n == 0 or l > r:
            return
        self._update_range(1, 0, self._n - 1, l, r, val)

    def _update_range(self, node, start, end, l, r, val):
        if l > end or r < start:
            return
        if l <= start and end <= r:
            self._apply_lazy(node, start, end, val)
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, l, r, val)
        self._update_range(2 * node + 1, mid + 1, end, l, r, val)
        self._tree[node] = self._monoid.combine(
            self._tree[2 * node], self._tree[2 * node + 1]
        )

    def update_point(self, idx, val):
        """Set element at idx to val."""
        if self._n == 0:
            return
        self._update_point(1, 0, self._n - 1, idx, val)

    def _update_point(self, node, start, end, idx, val):
        if start == end:
            self._tree[node] = val
            self._lazy[node] = self._lazy_op.identity()
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        if idx <= mid:
            self._update_point(2 * node, start, mid, idx, val)
        else:
            self._update_point(2 * node + 1, mid + 1, end, idx, val)
        self._tree[node] = self._monoid.combine(
            self._tree[2 * node], self._tree[2 * node + 1]
        )

    def query(self, l, r):
        """Query monoid aggregate over [l, r]."""
        if self._n == 0 or l > r:
            return self._monoid.identity()
        return self._query(1, 0, self._n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if l > end or r < start:
            return self._monoid.identity()
        if l <= start and end <= r:
            return self._tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return self._monoid.combine(left, right)

    def query_point(self, idx):
        """Get value at a single index."""
        return self.query(idx, idx)

    def __len__(self):
        return self._n

    def to_list(self):
        """Extract all current values as a list."""
        return [self.query_point(i) for i in range(self._n)]

    def find_first(self, l, r, predicate):
        """
        Find the leftmost index in [l, r] where predicate(prefix_aggregate) is True.
        Returns -1 if no such index exists.
        Useful for: find first position where prefix sum >= k.
        """
        if self._n == 0 or l > r:
            return -1
        result = [self._monoid.identity()]
        idx = self._find_first(1, 0, self._n - 1, l, r, predicate, result)
        return idx

    def _find_first(self, node, start, end, l, r, predicate, acc):
        if l > end or r < start:
            return -1
        if l <= start and end <= r:
            combined = self._monoid.combine(acc[0], self._tree[node])
            if not predicate(combined):
                acc[0] = combined
                return -1
            if start == end:
                acc[0] = combined
                return start
        self._push_down(node, start, end)
        mid = (start + end) // 2
        left_result = self._find_first(2 * node, start, mid, l, r, predicate, acc)
        if left_result != -1:
            return left_result
        return self._find_first(2 * node + 1, mid + 1, end, l, r, predicate, acc)

    def find_last(self, l, r, predicate):
        """
        Find the rightmost index in [l, r] where predicate(suffix_aggregate) is True.
        Returns -1 if no such index exists.
        """
        if self._n == 0 or l > r:
            return -1
        result = [self._monoid.identity()]
        idx = self._find_last(1, 0, self._n - 1, l, r, predicate, result)
        return idx

    def _find_last(self, node, start, end, l, r, predicate, acc):
        if l > end or r < start:
            return -1
        if l <= start and end <= r:
            combined = self._monoid.combine(self._tree[node], acc[0])
            if not predicate(combined):
                acc[0] = combined
                return -1
            if start == end:
                acc[0] = combined
                return start
        self._push_down(node, start, end)
        mid = (start + end) // 2
        right_result = self._find_last(2 * node + 1, mid + 1, end, l, r, predicate, acc)
        if right_result != -1:
            return right_result
        return self._find_last(2 * node, start, mid, l, r, predicate, acc)

    def kth_element(self, k):
        """
        Find kth smallest element (1-indexed) in a frequency segment tree.
        The tree should store counts, and this walks down using counts.
        Returns the index whose prefix count >= k.
        """
        if self._n == 0 or k <= 0:
            return -1
        return self._kth_element(1, 0, self._n - 1, k)

    def _kth_element(self, node, start, end, k):
        if start == end:
            return start
        self._push_down(node, start, end)
        left_count = self._tree[2 * node]
        if k <= left_count:
            return self._kth_element(2 * node, start, (start + end) // 2, k)
        else:
            return self._kth_element(2 * node + 1, (start + end) // 2 + 1, end, k - left_count)


# ---------------------------------------------------------------------------
# Persistent Segment Tree (path-copying)
# ---------------------------------------------------------------------------

class _PNode:
    """Immutable node for persistent segment tree."""
    __slots__ = ('val', 'left', 'right', 'lazy', 'count')

    def __init__(self, val, left=None, right=None, lazy=None, count=1):
        self.val = val
        self.left = left
        self.right = right
        self.lazy = lazy
        self.count = count


class PersistentSegmentTree:
    """
    Persistent segment tree using path-copying.
    Each update returns a new version (root) while old versions remain accessible.
    """

    def __init__(self, data=None, monoid=None, lazy_op=None):
        if monoid is None:
            monoid = SumMonoid
        if lazy_op is None:
            lazy_op = AddLazy

        self._monoid = monoid
        self._lazy_op = lazy_op
        self._versions = []

        if data is not None and len(data) > 0:
            self._n = len(data)
            root = self._build(data, 0, self._n - 1)
            self._versions.append(root)
        else:
            self._n = 0
            self._versions.append(None)

    def _build(self, data, start, end):
        if start == end:
            return _PNode(data[start], count=1)
        mid = (start + end) // 2
        left = self._build(data, start, mid)
        right = self._build(data, mid + 1, end)
        return _PNode(
            self._monoid.combine(left.val, right.val),
            left, right, count=end - start + 1
        )

    def _push_down(self, node, start, end):
        """Create new children with lazy pushed down. Returns new node."""
        if node is None or node.lazy is None:
            return node
        mid = (start + end) // 2
        left = self._apply_lazy_node(node.left, start, mid, node.lazy)
        right = self._apply_lazy_node(node.right, mid + 1, end, node.lazy)
        return _PNode(node.val, left, right, None, node.count)

    def _apply_lazy_node(self, node, start, end, lazy_val):
        """Apply lazy value to node, creating a new node."""
        if node is None:
            return None
        count = end - start + 1
        new_val = self._lazy_op.apply_to_node(lazy_val, node.val, count)
        new_lazy = self._lazy_op.compose(lazy_val, node.lazy)
        return _PNode(new_val, node.left, node.right, new_lazy, count)

    def update_point(self, version, idx, val):
        """Set element at idx to val in given version, return new version index."""
        root = self._versions[version]
        new_root = self._update_point(root, 0, self._n - 1, idx, val)
        self._versions.append(new_root)
        return len(self._versions) - 1

    def _update_point(self, node, start, end, idx, val):
        if start == end:
            return _PNode(val, count=1)
        node = self._push_down(node, start, end)
        mid = (start + end) // 2
        if idx <= mid:
            new_left = self._update_point(node.left, start, mid, idx, val)
            return _PNode(
                self._monoid.combine(new_left.val, node.right.val if node.right else self._monoid.identity()),
                new_left, node.right, count=node.count
            )
        else:
            new_right = self._update_point(node.right, mid + 1, end, idx, val)
            return _PNode(
                self._monoid.combine(node.left.val if node.left else self._monoid.identity(), new_right.val),
                node.left, new_right, count=node.count
            )

    def update_range(self, version, l, r, val):
        """Apply lazy operation to [l, r] in given version, return new version index."""
        root = self._versions[version]
        new_root = self._update_range(root, 0, self._n - 1, l, r, val)
        self._versions.append(new_root)
        return len(self._versions) - 1

    def _update_range(self, node, start, end, l, r, val):
        if l > end or r < start:
            return node
        if l <= start and end <= r:
            return self._apply_lazy_node(node, start, end, val)
        node = self._push_down(node, start, end)
        mid = (start + end) // 2
        new_left = self._update_range(node.left, start, mid, l, r, val)
        new_right = self._update_range(node.right, mid + 1, end, l, r, val)
        left_val = new_left.val if new_left else self._monoid.identity()
        right_val = new_right.val if new_right else self._monoid.identity()
        return _PNode(
            self._monoid.combine(left_val, right_val),
            new_left, new_right, count=end - start + 1
        )

    def query(self, version, l, r):
        """Query monoid aggregate over [l, r] in given version."""
        root = self._versions[version]
        if root is None:
            return self._monoid.identity()
        return self._query(root, 0, self._n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if node is None or l > end or r < start:
            return self._monoid.identity()
        if l <= start and end <= r:
            return node.val
        node = self._push_down(node, start, end)
        mid = (start + end) // 2
        left = self._query(node.left, start, mid, l, r)
        right = self._query(node.right, mid + 1, end, l, r)
        return self._monoid.combine(left, right)

    def query_point(self, version, idx):
        """Get value at a single index in given version."""
        return self.query(version, idx, idx)

    @property
    def version_count(self):
        return len(self._versions)

    @property
    def n(self):
        return self._n


# ---------------------------------------------------------------------------
# Segment Tree Beats (range chmin / chmax)
# ---------------------------------------------------------------------------

class _BeatsNode:
    """Node for Segment Tree Beats supporting range chmin/chmax."""
    __slots__ = ('max_val', 'max_cnt', 'second_max',
                 'min_val', 'min_cnt', 'second_min',
                 'sum_val', 'lazy_add')

    def __init__(self):
        self.max_val = -inf
        self.max_cnt = 0
        self.second_max = -inf
        self.min_val = inf
        self.min_cnt = 0
        self.second_min = inf
        self.sum_val = 0
        self.lazy_add = 0


class SegmentTreeBeats:
    """
    Segment Tree Beats (Ji driver segment tree).

    Supports:
    - range_chmin(l, r, v): set a[i] = min(a[i], v) for i in [l, r]
    - range_chmax(l, r, v): set a[i] = max(a[i], v) for i in [l, r]
    - range_add(l, r, v): add v to all elements in [l, r]
    - query_sum(l, r): sum of elements in [l, r]
    - query_min(l, r): minimum in [l, r]
    - query_max(l, r): maximum in [l, r]

    All operations O(log^2 n) amortized.
    """

    def __init__(self, data):
        self._n = len(data)
        if self._n == 0:
            self._tree = []
            return

        self._tree = [_BeatsNode() for _ in range(4 * self._n)]
        self._build(data, 1, 0, self._n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            v = data[start]
            t = self._tree[node]
            t.max_val = v
            t.max_cnt = 1
            t.second_max = -inf
            t.min_val = v
            t.min_cnt = 1
            t.second_min = inf
            t.sum_val = v
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self._pull_up(node)

    def _pull_up(self, node):
        l = self._tree[2 * node]
        r = self._tree[2 * node + 1]
        t = self._tree[node]

        t.sum_val = l.sum_val + r.sum_val

        # Max
        if l.max_val > r.max_val:
            t.max_val = l.max_val
            t.max_cnt = l.max_cnt
            t.second_max = max(l.second_max, r.max_val)
        elif l.max_val < r.max_val:
            t.max_val = r.max_val
            t.max_cnt = r.max_cnt
            t.second_max = max(l.max_val, r.second_max)
        else:
            t.max_val = l.max_val
            t.max_cnt = l.max_cnt + r.max_cnt
            t.second_max = max(l.second_max, r.second_max)

        # Min
        if l.min_val < r.min_val:
            t.min_val = l.min_val
            t.min_cnt = l.min_cnt
            t.second_min = min(l.second_min, r.min_val)
        elif l.min_val > r.min_val:
            t.min_val = r.min_val
            t.min_cnt = r.min_cnt
            t.second_min = min(l.min_val, r.second_min)
        else:
            t.min_val = l.min_val
            t.min_cnt = l.min_cnt + r.min_cnt
            t.second_min = min(l.second_min, r.second_min)

    def _push_add(self, node, start, end, val):
        t = self._tree[node]
        t.sum_val += val * (end - start + 1)
        t.max_val += val
        if t.second_max != -inf:
            t.second_max += val
        t.min_val += val
        if t.second_min != inf:
            t.second_min += val
        t.lazy_add += val

    def _push_chmin(self, node, start, end, val):
        """Apply chmin to node: clamp max_val down to val."""
        t = self._tree[node]
        if val >= t.max_val:
            return
        t.sum_val -= (t.max_val - val) * t.max_cnt
        # If max == min (all same), update min too
        if t.max_val == t.min_val:
            t.min_val = val
        elif val <= t.second_min:
            # val becomes the new min
            t.second_min = val
        if t.max_val == t.second_min:
            t.second_min = val
        t.max_val = val
        # Update min tracking
        if val < t.min_val:
            t.min_val = val
            t.min_cnt = end - start + 1
            t.second_min = inf

    def _push_chmax(self, node, start, end, val):
        """Apply chmax to node: clamp min_val up to val."""
        t = self._tree[node]
        if val <= t.min_val:
            return
        t.sum_val += (val - t.min_val) * t.min_cnt
        # If min == max (all same), update max too
        if t.min_val == t.max_val:
            t.max_val = val
        elif val >= t.second_max:
            t.second_max = val
        if t.min_val == t.second_max:
            t.second_max = val
        t.min_val = val
        # Update max tracking
        if val > t.max_val:
            t.max_val = val
            t.max_cnt = end - start + 1
            t.second_max = -inf

    def _push_down(self, node, start, end):
        if start == end:
            return
        mid = (start + end) // 2
        t = self._tree[node]

        # Push add first
        if t.lazy_add != 0:
            self._push_add(2 * node, start, mid, t.lazy_add)
            self._push_add(2 * node + 1, mid + 1, end, t.lazy_add)
            t.lazy_add = 0

        # Push chmin (clamp children's max down to this node's max)
        self._push_chmin(2 * node, start, mid, t.max_val)
        self._push_chmin(2 * node + 1, mid + 1, end, t.max_val)

        # Push chmax (clamp children's min up to this node's min)
        self._push_chmax(2 * node, start, mid, t.min_val)
        self._push_chmax(2 * node + 1, mid + 1, end, t.min_val)

    def range_chmin(self, l, r, val):
        """Set a[i] = min(a[i], val) for i in [l, r]."""
        if self._n == 0 or l > r:
            return
        self._range_chmin(1, 0, self._n - 1, l, r, val)

    def _range_chmin(self, node, start, end, l, r, val):
        if l > end or r < start or val >= self._tree[node].max_val:
            return
        if l <= start and end <= r and val > self._tree[node].second_max:
            self._push_chmin(node, start, end, val)
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_chmin(2 * node, start, mid, l, r, val)
        self._range_chmin(2 * node + 1, mid + 1, end, l, r, val)
        self._pull_up(node)

    def range_chmax(self, l, r, val):
        """Set a[i] = max(a[i], val) for i in [l, r]."""
        if self._n == 0 or l > r:
            return
        self._range_chmax(1, 0, self._n - 1, l, r, val)

    def _range_chmax(self, node, start, end, l, r, val):
        if l > end or r < start or val <= self._tree[node].min_val:
            return
        if l <= start and end <= r and val < self._tree[node].second_min:
            self._push_chmax(node, start, end, val)
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_chmax(2 * node, start, mid, l, r, val)
        self._range_chmax(2 * node + 1, mid + 1, end, l, r, val)
        self._pull_up(node)

    def range_add(self, l, r, val):
        """Add val to all elements in [l, r]."""
        if self._n == 0 or l > r:
            return
        self._range_add(1, 0, self._n - 1, l, r, val)

    def _range_add(self, node, start, end, l, r, val):
        if l > end or r < start:
            return
        if l <= start and end <= r:
            self._push_add(node, start, end, val)
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_add(2 * node, start, mid, l, r, val)
        self._range_add(2 * node + 1, mid + 1, end, l, r, val)
        self._pull_up(node)

    def query_sum(self, l, r):
        """Sum of elements in [l, r]."""
        if self._n == 0 or l > r:
            return 0
        return self._query_sum(1, 0, self._n - 1, l, r)

    def _query_sum(self, node, start, end, l, r):
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            return self._tree[node].sum_val
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query_sum(2 * node, start, mid, l, r) +
                self._query_sum(2 * node + 1, mid + 1, end, l, r))

    def query_min(self, l, r):
        """Minimum in [l, r]."""
        if self._n == 0 or l > r:
            return inf
        return self._query_min(1, 0, self._n - 1, l, r)

    def _query_min(self, node, start, end, l, r):
        if l > end or r < start:
            return inf
        if l <= start and end <= r:
            return self._tree[node].min_val
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return min(self._query_min(2 * node, start, mid, l, r),
                   self._query_min(2 * node + 1, mid + 1, end, l, r))

    def query_max(self, l, r):
        """Maximum in [l, r]."""
        if self._n == 0 or l > r:
            return -inf
        return self._query_max(1, 0, self._n - 1, l, r)

    def _query_max(self, node, start, end, l, r):
        if l > end or r < start:
            return -inf
        if l <= start and end <= r:
            return self._tree[node].max_val
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return max(self._query_max(2 * node, start, mid, l, r),
                   self._query_max(2 * node + 1, mid + 1, end, l, r))

    def to_list(self):
        """Extract current values."""
        result = []
        for i in range(self._n):
            result.append(self.query_min(i, i))
        return result


# ---------------------------------------------------------------------------
# Merge Sort Tree (for order-statistic range queries)
# ---------------------------------------------------------------------------

class MergeSortTree:
    """
    Merge sort tree: each node stores sorted subarray.
    Supports count_less_than(l, r, v) and kth_smallest(l, r, k) queries.
    Build: O(n log n), Query: O(log^2 n) or O(log^3 n) for kth.
    """

    def __init__(self, data):
        self._n = len(data)
        self._data = list(data)

        if self._n == 0:
            self._tree = []
            return

        self._tree = [None] * (4 * self._n)
        self._build(1, 0, self._n - 1)

    def _build(self, node, start, end):
        if start == end:
            self._tree[node] = [self._data[start]]
            return
        mid = (start + end) // 2
        self._build(2 * node, start, mid)
        self._build(2 * node + 1, mid + 1, end)
        # Merge two sorted arrays
        self._tree[node] = self._merge(self._tree[2 * node], self._tree[2 * node + 1])

    def _merge(self, a, b):
        result = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result

    def count_less_than(self, l, r, val):
        """Count elements in [l, r] strictly less than val. O(log^2 n)."""
        if self._n == 0 or l > r:
            return 0
        return self._count_less(1, 0, self._n - 1, l, r, val)

    def _count_less(self, node, start, end, l, r, val):
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            # Binary search in sorted array
            return self._bisect_left(self._tree[node], val)
        mid = (start + end) // 2
        return (self._count_less(2 * node, start, mid, l, r, val) +
                self._count_less(2 * node + 1, mid + 1, end, l, r, val))

    def count_less_equal(self, l, r, val):
        """Count elements in [l, r] <= val."""
        if self._n == 0 or l > r:
            return 0
        return self._count_leq(1, 0, self._n - 1, l, r, val)

    def _count_leq(self, node, start, end, l, r, val):
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            return self._bisect_right(self._tree[node], val)
        mid = (start + end) // 2
        return (self._count_leq(2 * node, start, mid, l, r, val) +
                self._count_leq(2 * node + 1, mid + 1, end, l, r, val))

    def count_in_range(self, l, r, lo, hi):
        """Count elements in [l, r] with value in [lo, hi]."""
        return self.count_less_equal(l, r, hi) - self.count_less_than(l, r, lo)

    def kth_smallest(self, l, r, k):
        """
        Find kth smallest element in subarray [l, r] (1-indexed).
        O(log^3 n) via binary search + count_less_equal.
        Returns None if k > r - l + 1.
        """
        if self._n == 0 or l > r or k <= 0 or k > r - l + 1:
            return None

        # Get sorted values for binary search bounds
        all_vals = sorted(set(self._data[l:r + 1]))

        # Binary search on value
        lo_idx, hi_idx = 0, len(all_vals) - 1
        while lo_idx < hi_idx:
            mid_idx = (lo_idx + hi_idx) // 2
            cnt = self.count_less_equal(l, r, all_vals[mid_idx])
            if cnt < k:
                lo_idx = mid_idx + 1
            else:
                hi_idx = mid_idx
        return all_vals[lo_idx]

    def _bisect_left(self, arr, val):
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _bisect_right(self, arr, val):
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= val:
                lo = mid + 1
            else:
                hi = mid
        return lo


# ---------------------------------------------------------------------------
# 2D Segment Tree (for rectangle queries)
# ---------------------------------------------------------------------------

class SegmentTree2D:
    """
    2D segment tree for rectangle sum queries and point updates.
    Build: O(n*m*log(n)*log(m)), Query: O(log(n)*log(m)).
    """

    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            self._rows = 0
            self._cols = 0
            return

        self._rows = len(matrix)
        self._cols = len(matrix[0])

        # Outer tree nodes, each containing an inner segment tree (as array)
        self._tree = [[0] * (4 * self._cols) for _ in range(4 * self._rows)]
        self._build_y(matrix, 1, 0, self._rows - 1)

    def _build_x(self, node_y, node_x, lx, rx, matrix_row=None, ly=None, ry=None):
        if lx == rx:
            if matrix_row is not None:
                self._tree[node_y][node_x] = matrix_row[lx]
            else:
                self._tree[node_y][node_x] = (
                    self._tree[2 * node_y][node_x] +
                    self._tree[2 * node_y + 1][node_x]
                )
            return
        mid = (lx + rx) // 2
        self._build_x(node_y, 2 * node_x, lx, mid, matrix_row, ly, ry)
        self._build_x(node_y, 2 * node_x + 1, mid + 1, rx, matrix_row, ly, ry)
        self._tree[node_y][node_x] = (
            self._tree[node_y][2 * node_x] +
            self._tree[node_y][2 * node_x + 1]
        )

    def _build_y(self, matrix, node_y, ly, ry):
        if ly == ry:
            self._build_x(node_y, 1, 0, self._cols - 1, matrix_row=matrix[ly])
            return
        mid = (ly + ry) // 2
        self._build_y(matrix, 2 * node_y, ly, mid)
        self._build_y(matrix, 2 * node_y + 1, mid + 1, ry)
        self._build_x(node_y, 1, 0, self._cols - 1, ly=ly, ry=ry)

    def update(self, row, col, val):
        """Set matrix[row][col] = val."""
        self._update_y(1, 0, self._rows - 1, row, col, val)

    def _update_y(self, node_y, ly, ry, row, col, val):
        if ly == ry:
            self._update_x(node_y, 1, 0, self._cols - 1, col, val, is_leaf_y=True)
            return
        mid = (ly + ry) // 2
        if row <= mid:
            self._update_y(2 * node_y, ly, mid, row, col, val)
        else:
            self._update_y(2 * node_y + 1, mid + 1, ry, row, col, val)
        self._update_x(node_y, 1, 0, self._cols - 1, col, val, is_leaf_y=False)

    def _update_x(self, node_y, node_x, lx, rx, col, val, is_leaf_y):
        if lx == rx:
            if is_leaf_y:
                self._tree[node_y][node_x] = val
            else:
                self._tree[node_y][node_x] = (
                    self._tree[2 * node_y][node_x] +
                    self._tree[2 * node_y + 1][node_x]
                )
            return
        mid = (lx + rx) // 2
        if col <= mid:
            self._update_x(node_y, 2 * node_x, lx, mid, col, val, is_leaf_y)
        else:
            self._update_x(node_y, 2 * node_x + 1, mid + 1, rx, col, val, is_leaf_y)
        self._tree[node_y][node_x] = (
            self._tree[node_y][2 * node_x] +
            self._tree[node_y][2 * node_x + 1]
        )

    def query(self, r1, c1, r2, c2):
        """Sum of elements in rectangle [r1..r2] x [c1..c2]."""
        if self._rows == 0:
            return 0
        return self._query_y(1, 0, self._rows - 1, r1, r2, c1, c2)

    def _query_y(self, node_y, ly, ry, r1, r2, c1, c2):
        if r1 > ry or r2 < ly:
            return 0
        if r1 <= ly and ry <= r2:
            return self._query_x(node_y, 1, 0, self._cols - 1, c1, c2)
        mid = (ly + ry) // 2
        return (self._query_y(2 * node_y, ly, mid, r1, r2, c1, c2) +
                self._query_y(2 * node_y + 1, mid + 1, ry, r1, r2, c1, c2))

    def _query_x(self, node_y, node_x, lx, rx, c1, c2):
        if c1 > rx or c2 < lx:
            return 0
        if c1 <= lx and rx <= c2:
            return self._tree[node_y][node_x]
        mid = (lx + rx) // 2
        return (self._query_x(node_y, 2 * node_x, lx, mid, c1, c2) +
                self._query_x(node_y, 2 * node_x + 1, mid + 1, rx, c1, c2))


# ---------------------------------------------------------------------------
# Sparse Segment Tree (for large index ranges)
# ---------------------------------------------------------------------------

class _SparseNode:
    __slots__ = ('val', 'left', 'right', 'lazy')
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
        self.lazy = None


class SparseSegmentTree:
    """
    Sparse (dynamic) segment tree for very large index ranges.
    Only allocates nodes as needed. Supports sum queries + range add.
    """

    def __init__(self, lo, hi):
        """Create sparse segment tree covering index range [lo, hi]."""
        self._lo = lo
        self._hi = hi
        self._root = _SparseNode()

    def _push_down(self, node, start, end):
        if node.lazy is not None and start < end:
            mid = (start + end) // 2
            if node.left is None:
                node.left = _SparseNode()
            if node.right is None:
                node.right = _SparseNode()
            node.left.val += node.lazy * (mid - start + 1)
            node.left.lazy = (node.left.lazy or 0) + node.lazy
            node.right.val += node.lazy * (end - mid)
            node.right.lazy = (node.right.lazy or 0) + node.lazy
            node.lazy = None

    def update_range(self, l, r, val):
        """Add val to all elements in [l, r]."""
        self._update(self._root, self._lo, self._hi, l, r, val)

    def _update(self, node, start, end, l, r, val):
        if l > end or r < start:
            return
        if l <= start and end <= r:
            node.val += val * (end - start + 1)
            node.lazy = (node.lazy or 0) + val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        if node.left is None:
            node.left = _SparseNode()
        if node.right is None:
            node.right = _SparseNode()
        self._update(node.left, start, mid, l, r, val)
        self._update(node.right, mid + 1, end, l, r, val)
        node.val = node.left.val + node.right.val

    def update_point(self, idx, val):
        """Set element at idx to val (absolute set, not add)."""
        self._set_point(self._root, self._lo, self._hi, idx, val)

    def _set_point(self, node, start, end, idx, val):
        if start == end:
            node.val = val
            node.lazy = None
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        if node.left is None:
            node.left = _SparseNode()
        if node.right is None:
            node.right = _SparseNode()
        if idx <= mid:
            self._set_point(node.left, start, mid, idx, val)
        else:
            self._set_point(node.right, mid + 1, end, idx, val)
        node.val = node.left.val + node.right.val

    def query(self, l, r):
        """Sum of elements in [l, r]."""
        return self._query(self._root, self._lo, self._hi, l, r)

    def _query(self, node, start, end, l, r):
        if node is None or l > end or r < start:
            return 0
        if l <= start and end <= r:
            return node.val
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(node.left, start, mid, l, r) +
                self._query(node.right, mid + 1, end, l, r))

    def query_point(self, idx):
        """Get value at a single index."""
        return self.query(idx, idx)
