"""
C082: Interval Tree -- Composing C081 Finger Tree with Interval Monoid.

An interval tree supporting:
  - O(log n) insert/delete of intervals
  - O(log n + k) stabbing queries (find all intervals containing a point)
  - O(log n + k) overlap queries (find all intervals overlapping a range)
  - O(1) min/max interval endpoints
  - O(n) coverage, gaps, merge-overlapping
  - Bulk operations, statistics, sweep-line algorithms

Intervals are stored sorted by low endpoint. The monoid tracks the maximum
high endpoint, enabling efficient pruning during queries.

Composes: C081 Finger Tree (finger_tree.py)
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple

# Import C081 finger tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C081_finger_tree'))
from finger_tree import (
    Monoid, Elem, Node2, Node3, Empty, Single, Deep, EMPTY,
    cons, snoc, head, last, tail, init, concat, split,
    to_list, from_list, _digit_measure, _head_deep, _last_deep,
    _list_to_tree, _deep_left, _deep_right, _split_digit, _split_tree,
    _unwrap, fold_left, fold_right
)


# ============================================================
# Interval representation
# ============================================================

@dataclass(frozen=True, order=True)
class Interval:
    """A closed interval [lo, hi] with optional associated data."""
    lo: float
    hi: float
    data: Any = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"Invalid interval: lo={self.lo} > hi={self.hi}")

    def contains_point(self, p):
        """Check if point p is within this interval."""
        return self.lo <= p <= self.hi

    def overlaps(self, other):
        """Check if this interval overlaps with another."""
        if isinstance(other, Interval):
            return self.lo <= other.hi and self.hi >= other.lo
        # (lo, hi) tuple
        return self.lo <= other[1] and self.hi >= other[0]

    def contains_interval(self, other):
        """Check if this interval fully contains another."""
        if isinstance(other, Interval):
            return self.lo <= other.lo and self.hi >= other.hi
        return self.lo <= other[0] and self.hi >= other[1]

    @property
    def length(self):
        """Length of the interval."""
        return self.hi - self.lo

    @property
    def midpoint(self):
        """Midpoint of the interval."""
        return (self.lo + self.hi) / 2

    def __repr__(self):
        if self.data is not None:
            return f"Interval({self.lo}, {self.hi}, {self.data!r})"
        return f"Interval({self.lo}, {self.hi})"


# ============================================================
# Interval Monoid -- tracks max high endpoint
# ============================================================

class IntervalMonoid(Monoid):
    """Monoid for interval trees.

    Measures: (max_lo, min_lo, max_hi) triple.
    - max_lo: enables monotone split predicate for sorted insertion
    - min_lo: enables overlap pruning (skip if all lo > query hi)
    - max_hi: enables stab/overlap pruning (skip if all hi < query point)
    """

    @staticmethod
    def empty():
        return None  # (max_lo, min_lo, max_hi) or None

    @staticmethod
    def combine(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]))

    @staticmethod
    def measure(x):
        """Measure an interval element."""
        if isinstance(x, Interval):
            return (x.lo, x.lo, x.hi)  # (max_lo, min_lo, max_hi)
        # Already a measurement tuple
        return x


# ============================================================
# Tree traversal with monoid-based pruning
# ============================================================

def _collect_stab(tree, point, monoid, results):
    """Collect all intervals containing point, using max_hi for pruning."""
    if isinstance(tree, Empty):
        return

    # Check monoid: if max_hi < point, no interval here contains point
    m = tree.measure(monoid)
    if m is None:
        return
    max_hi = m[2]
    if max_hi < point:
        return

    if isinstance(tree, Single):
        elem = tree.elem
        iv = elem.value if isinstance(elem, Elem) else elem
        if isinstance(iv, Interval) and iv.contains_point(point):
            results.append(iv)
        return

    # Deep: check left digit, middle, right digit
    _collect_stab_digit(tree.left, point, monoid, results)
    _collect_stab(tree.middle, point, monoid, results)
    _collect_stab_digit(tree.right, point, monoid, results)


def _collect_stab_digit(digit, point, monoid, results):
    """Collect from a digit (list of Elem/Node)."""
    for elem in digit:
        if isinstance(elem, Elem):
            iv = elem.value
            if isinstance(iv, Interval) and iv.contains_point(point):
                results.append(iv)
        elif isinstance(elem, (Node2, Node3)):
            # Check monoid measure for pruning
            m = elem.measure(monoid)
            if m is not None and m[2] >= point:
                _collect_stab_node(elem, point, monoid, results)


def _collect_stab_node(node, point, monoid, results):
    """Collect from a Node2/Node3."""
    m = node.measure(monoid)
    if m is None or m[2] < point:
        return

    if isinstance(node, Node2):
        _collect_stab_child(node.a, point, monoid, results)
        _collect_stab_child(node.b, point, monoid, results)
    elif isinstance(node, Node3):
        _collect_stab_child(node.a, point, monoid, results)
        _collect_stab_child(node.b, point, monoid, results)
        _collect_stab_child(node.c, point, monoid, results)


def _collect_stab_child(child, point, monoid, results):
    """Collect from a child which may be Elem, Node2, or Node3."""
    if isinstance(child, Elem):
        iv = child.value
        if isinstance(iv, Interval) and iv.contains_point(point):
            results.append(iv)
    elif isinstance(child, (Node2, Node3)):
        _collect_stab_node(child, point, monoid, results)


def _collect_overlap(tree, qlo, qhi, monoid, results):
    """Collect all intervals overlapping [qlo, qhi], using max_hi for pruning."""
    if isinstance(tree, Empty):
        return

    m = tree.measure(monoid)
    if m is None:
        return
    min_lo = m[1]
    max_hi = m[2]
    # Pruning: if max_hi < qlo, no interval overlaps
    if max_hi < qlo:
        return
    # Pruning: if min_lo > qhi, no interval overlaps
    if min_lo > qhi:
        return

    if isinstance(tree, Single):
        elem = tree.elem
        iv = elem.value if isinstance(elem, Elem) else elem
        if isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi)):
            results.append(iv)
        return

    _collect_overlap_digit(tree.left, qlo, qhi, monoid, results)
    _collect_overlap(tree.middle, qlo, qhi, monoid, results)
    _collect_overlap_digit(tree.right, qlo, qhi, monoid, results)


def _collect_overlap_digit(digit, qlo, qhi, monoid, results):
    """Collect overlapping intervals from a digit."""
    for elem in digit:
        if isinstance(elem, Elem):
            iv = elem.value
            if isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi)):
                results.append(iv)
        elif isinstance(elem, (Node2, Node3)):
            m = elem.measure(monoid)
            if m is not None and m[2] >= qlo and m[1] <= qhi:
                _collect_overlap_node(elem, qlo, qhi, monoid, results)


def _collect_overlap_node(node, qlo, qhi, monoid, results):
    """Collect from a Node2/Node3."""
    m = node.measure(monoid)
    if m is None or m[2] < qlo or m[1] > qhi:
        return

    if isinstance(node, Node2):
        _collect_overlap_child(node.a, qlo, qhi, monoid, results)
        _collect_overlap_child(node.b, qlo, qhi, monoid, results)
    elif isinstance(node, Node3):
        _collect_overlap_child(node.a, qlo, qhi, monoid, results)
        _collect_overlap_child(node.b, qlo, qhi, monoid, results)
        _collect_overlap_child(node.c, qlo, qhi, monoid, results)


def _collect_overlap_child(child, qlo, qhi, monoid, results):
    """Collect from a child."""
    if isinstance(child, Elem):
        iv = child.value
        if isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi)):
            results.append(iv)
    elif isinstance(child, (Node2, Node3)):
        _collect_overlap_node(child, qlo, qhi, monoid, results)


# ============================================================
# Interval Tree
# ============================================================

class IntervalTree:
    """Persistent interval tree backed by a finger tree.

    Intervals are stored sorted by low endpoint. The IntervalMonoid
    tracks the maximum high endpoint for efficient query pruning.

    All operations return new trees (persistent/immutable).
    """

    def __init__(self, tree=None):
        self._tree = tree if tree is not None else EMPTY
        self._monoid = IntervalMonoid

    def is_empty(self):
        return isinstance(self._tree, Empty)

    def __len__(self):
        return len(to_list(self._tree))

    def __iter__(self):
        return iter(to_list(self._tree))

    def __repr__(self):
        items = to_list(self._tree)
        if len(items) > 8:
            return f"IntervalTree([{', '.join(repr(x) for x in items[:8])}, ...] ({len(items)} intervals))"
        return f"IntervalTree({items!r})"

    def __eq__(self, other):
        if isinstance(other, IntervalTree):
            return sorted(to_list(self._tree), key=lambda iv: (iv.lo, iv.hi)) == \
                   sorted(to_list(other._tree), key=lambda iv: (iv.lo, iv.hi))
        return NotImplemented

    def __contains__(self, interval):
        """Check if exact interval exists in the tree."""
        for iv in to_list(self._tree):
            if iv.lo == interval.lo and iv.hi == interval.hi:
                if interval.data is not None:
                    if iv.data == interval.data:
                        return True
                else:
                    return True
        return False

    # --------------------------------------------------------
    # Core operations
    # --------------------------------------------------------

    def insert(self, lo_or_interval, hi=None, data=None):
        """Insert an interval. O(log n).

        Can be called as:
          tree.insert(Interval(1, 5))
          tree.insert(1, 5)
          tree.insert(1, 5, "label")
        """
        if isinstance(lo_or_interval, Interval):
            iv = lo_or_interval
        else:
            iv = Interval(lo_or_interval, hi, data)

        if isinstance(self._tree, Empty):
            return IntervalTree(Single(Elem(iv)))

        # Find insertion point (sorted by lo, then hi for stability)
        m = self._tree.measure(self._monoid)
        if m is None or (iv.lo > m[0] and iv.lo > self._max_lo()):
            return IntervalTree(snoc(self._tree, iv, self._monoid))

        # Split at the insertion point: max_lo (m[0]) >= iv.lo is monotone
        try:
            left, x, right = split(
                self._tree,
                lambda m: m is not None and m[0] >= iv.lo,
                self._monoid
            )
            # x has lo >= iv.lo. We need to insert iv before x.
            # But we need to handle equal lo values (sort by hi as tiebreaker)
            if x.lo == iv.lo and iv.hi > x.hi:
                # iv goes after x
                right = cons(right, x, self._monoid)
                # Find correct position among equal-lo intervals
                return IntervalTree(
                    concat(snoc(left, iv, self._monoid), right, self._monoid)
                )
            right = cons(right, x, self._monoid)
            return IntervalTree(
                concat(snoc(left, iv, self._monoid), right, self._monoid)
            )
        except ValueError:
            # predicate never satisfied -- all lo values < iv.lo
            return IntervalTree(snoc(self._tree, iv, self._monoid))

    def _max_lo(self):
        """Get the maximum lo value (last element's lo)."""
        lst = last(self._tree)
        return lst.lo if isinstance(lst, Interval) else float('-inf')

    def delete(self, lo_or_interval, hi=None, data=None):
        """Delete an interval. O(n) -- scans for exact match.

        Returns new tree without the first matching interval.
        """
        if isinstance(lo_or_interval, Interval):
            target = lo_or_interval
        else:
            target = Interval(lo_or_interval, hi, data)

        items = to_list(self._tree)
        found = False
        result = []
        for iv in items:
            if not found and iv.lo == target.lo and iv.hi == target.hi:
                if target.data is not None:
                    if iv.data == target.data:
                        found = True
                        continue
                else:
                    found = True
                    continue
            result.append(iv)

        if not found:
            return self  # Nothing to delete

        return IntervalTree(from_list(result, self._monoid))

    def delete_all(self, lo_or_interval, hi=None):
        """Delete all intervals matching [lo, hi]. Returns new tree."""
        if isinstance(lo_or_interval, Interval):
            lo, hi = lo_or_interval.lo, lo_or_interval.hi
        else:
            lo = lo_or_interval

        items = [iv for iv in to_list(self._tree)
                 if not (iv.lo == lo and iv.hi == hi)]
        return IntervalTree(from_list(items, self._monoid))

    # --------------------------------------------------------
    # Query operations
    # --------------------------------------------------------

    def stab(self, point):
        """Find all intervals containing point. O(log n + k) with pruning.

        Returns list of Interval objects.
        """
        results = []
        _collect_stab(self._tree, point, self._monoid, results)
        return results

    def overlap(self, lo_or_interval, hi=None):
        """Find all intervals overlapping [lo, hi]. O(log n + k) with pruning.

        Can be called as:
          tree.overlap(Interval(1, 5))
          tree.overlap(1, 5)
        """
        if isinstance(lo_or_interval, Interval):
            qlo, qhi = lo_or_interval.lo, lo_or_interval.hi
        else:
            qlo, qhi = lo_or_interval, hi

        results = []
        _collect_overlap(self._tree, qlo, qhi, self._monoid, results)
        return results

    def any_overlap(self, lo_or_interval, hi=None):
        """Check if any interval overlaps [lo, hi]. O(log n) -- early exit."""
        if isinstance(lo_or_interval, Interval):
            qlo, qhi = lo_or_interval.lo, lo_or_interval.hi
        else:
            qlo, qhi = lo_or_interval, hi

        return self._any_overlap_tree(self._tree, qlo, qhi)

    def _any_overlap_tree(self, tree, qlo, qhi):
        """Check if any interval in tree overlaps [qlo, qhi]."""
        if isinstance(tree, Empty):
            return False

        m = tree.measure(self._monoid)
        if m is None:
            return False
        min_lo = m[1]
        max_hi = m[2]
        if max_hi < qlo or min_lo > qhi:
            return False

        if isinstance(tree, Single):
            elem = tree.elem
            iv = elem.value if isinstance(elem, Elem) else elem
            return isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi))

        # Check digits and middle
        if self._any_overlap_digit(tree.left, qlo, qhi):
            return True
        if self._any_overlap_tree(tree.middle, qlo, qhi):
            return True
        if self._any_overlap_digit(tree.right, qlo, qhi):
            return True
        return False

    def _any_overlap_digit(self, digit, qlo, qhi):
        """Check digit for any overlap."""
        for elem in digit:
            if isinstance(elem, Elem):
                iv = elem.value
                if isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi)):
                    return True
            elif isinstance(elem, (Node2, Node3)):
                m = elem.measure(self._monoid)
                if m is not None and m[2] >= qlo and m[1] <= qhi:
                    if self._any_overlap_node(elem, qlo, qhi):
                        return True
        return False

    def _any_overlap_node(self, node, qlo, qhi):
        """Check node for any overlap."""
        m = node.measure(self._monoid)
        if m is None or m[2] < qlo or m[1] > qhi:
            return False

        children = [node.a, node.b]
        if isinstance(node, Node3):
            children.append(node.c)

        for child in children:
            if isinstance(child, Elem):
                iv = child.value
                if isinstance(iv, Interval) and iv.overlaps(Interval(qlo, qhi)):
                    return True
            elif isinstance(child, (Node2, Node3)):
                if self._any_overlap_node(child, qlo, qhi):
                    return True
        return False

    def containing(self, lo_or_interval, hi=None):
        """Find all intervals that fully contain [lo, hi]."""
        if isinstance(lo_or_interval, Interval):
            qlo, qhi = lo_or_interval.lo, lo_or_interval.hi
        else:
            qlo, qhi = lo_or_interval, hi

        results = []
        for iv in to_list(self._tree):
            if iv.lo <= qlo and iv.hi >= qhi:
                results.append(iv)
        return results

    def contained_by(self, lo_or_interval, hi=None):
        """Find all intervals fully contained within [lo, hi]."""
        if isinstance(lo_or_interval, Interval):
            qlo, qhi = lo_or_interval.lo, lo_or_interval.hi
        else:
            qlo, qhi = lo_or_interval, hi

        results = []
        for iv in to_list(self._tree):
            if iv.lo >= qlo and iv.hi <= qhi:
                results.append(iv)
        return results

    def nearest(self, point, count=1):
        """Find the nearest interval(s) to a point.

        Returns list of (distance, interval) sorted by distance.
        Distance is 0 if point is inside the interval.
        """
        items = to_list(self._tree)
        if not items:
            return []

        distances = []
        for iv in items:
            if iv.contains_point(point):
                d = 0
            elif point < iv.lo:
                d = iv.lo - point
            else:
                d = point - iv.hi
            distances.append((d, iv))

        distances.sort(key=lambda x: (x[0], x[1].lo, x[1].hi))
        return distances[:count]

    # --------------------------------------------------------
    # Endpoint queries
    # --------------------------------------------------------

    def min_lo(self):
        """Get the minimum low endpoint. O(1)."""
        if self.is_empty():
            raise IndexError("min_lo on empty tree")
        h = head(self._tree)
        return h.lo if isinstance(h, Interval) else h

    def max_lo(self):
        """Get the maximum low endpoint. O(1)."""
        if self.is_empty():
            raise IndexError("max_lo on empty tree")
        l = last(self._tree)
        return l.lo if isinstance(l, Interval) else l

    def max_hi(self):
        """Get the maximum high endpoint. O(1) -- from monoid."""
        if self.is_empty():
            raise IndexError("max_hi on empty tree")
        m = self._tree.measure(self._monoid)
        return m[2] if m is not None else None

    def min_hi(self):
        """Get the minimum high endpoint. O(n)."""
        if self.is_empty():
            raise IndexError("min_hi on empty tree")
        return min(iv.hi for iv in to_list(self._tree))

    def span(self):
        """Get the overall span [min_lo, max_hi]. O(1)."""
        if self.is_empty():
            return None
        return Interval(self.min_lo(), self.max_hi())

    # --------------------------------------------------------
    # Set operations
    # --------------------------------------------------------

    def merge(self, other):
        """Merge two interval trees. O(n + m)."""
        items = sorted(
            to_list(self._tree) + to_list(other._tree),
            key=lambda iv: (iv.lo, iv.hi)
        )
        return IntervalTree(from_list(items, self._monoid))

    def intersection(self, other):
        """Find intervals present in both trees. O(n * m)."""
        other_items = to_list(other._tree)
        result = []
        for iv in to_list(self._tree):
            for oiv in other_items:
                if iv.lo == oiv.lo and iv.hi == oiv.hi:
                    result.append(iv)
                    break
        result.sort(key=lambda iv: (iv.lo, iv.hi))
        return IntervalTree(from_list(result, self._monoid))

    def difference(self, other):
        """Find intervals in self but not in other. O(n * m)."""
        other_items = to_list(other._tree)
        result = []
        for iv in to_list(self._tree):
            found = False
            for oiv in other_items:
                if iv.lo == oiv.lo and iv.hi == oiv.hi:
                    found = True
                    break
            if not found:
                result.append(iv)
        return IntervalTree(from_list(result, self._monoid))

    # --------------------------------------------------------
    # Interval arithmetic
    # --------------------------------------------------------

    def merge_overlapping(self):
        """Merge all overlapping/adjacent intervals. O(n log n).

        Returns new tree with non-overlapping intervals.
        """
        items = sorted(to_list(self._tree), key=lambda iv: (iv.lo, iv.hi))
        if not items:
            return IntervalTree()

        merged = [Interval(items[0].lo, items[0].hi)]
        for iv in items[1:]:
            if iv.lo <= merged[-1].hi:
                # Overlapping or adjacent -- extend
                if iv.hi > merged[-1].hi:
                    merged[-1] = Interval(merged[-1].lo, iv.hi)
            else:
                merged.append(Interval(iv.lo, iv.hi))

        return IntervalTree(from_list(merged, self._monoid))

    def gaps(self):
        """Find gaps between intervals. O(n log n).

        Returns list of Interval objects representing uncovered regions.
        """
        merged = self.merge_overlapping()
        items = to_list(merged._tree)
        if len(items) <= 1:
            return []

        result = []
        for i in range(len(items) - 1):
            if items[i].hi < items[i + 1].lo:
                result.append(Interval(items[i].hi, items[i + 1].lo))
        return result

    def coverage(self):
        """Calculate total coverage (sum of non-overlapping lengths). O(n log n)."""
        merged = self.merge_overlapping()
        return sum(iv.length for iv in to_list(merged._tree))

    def complement(self, lo, hi):
        """Find uncovered regions within [lo, hi]. O(n log n).

        Returns list of Interval objects.
        """
        merged = self.merge_overlapping()
        items = to_list(merged._tree)

        result = []
        current = lo
        for iv in items:
            if iv.lo > hi:
                break
            if iv.hi < lo:
                continue
            start = max(iv.lo, lo)
            if current < start:
                result.append(Interval(current, start))
            current = max(current, min(iv.hi, hi))
        if current < hi:
            result.append(Interval(current, hi))
        return result

    def clip(self, lo, hi):
        """Clip all intervals to [lo, hi]. O(n).

        Returns new tree with intervals trimmed to the clip range.
        Intervals entirely outside are removed.
        """
        result = []
        for iv in to_list(self._tree):
            if iv.hi < lo or iv.lo > hi:
                continue
            clipped = Interval(max(iv.lo, lo), min(iv.hi, hi), iv.data)
            result.append(clipped)
        result.sort(key=lambda iv: (iv.lo, iv.hi))
        return IntervalTree(from_list(result, self._monoid))

    def fragment(self):
        """Fragment overlapping intervals into non-overlapping pieces. O(n log n).

        Returns new tree where every original boundary creates a split point.
        Example: [1,5] and [3,7] -> [1,3], [3,5], [5,7]
        """
        items = to_list(self._tree)
        if not items:
            return IntervalTree()

        # Collect all unique boundary points
        points = set()
        for iv in items:
            points.add(iv.lo)
            points.add(iv.hi)
        points = sorted(points)

        if len(points) < 2:
            return IntervalTree(from_list(items, self._monoid))

        # Create fragments for each segment between consecutive points
        fragments = []
        for i in range(len(points) - 1):
            seg_lo, seg_hi = points[i], points[i + 1]
            # Check if any interval covers this segment
            mid = (seg_lo + seg_hi) / 2
            for iv in items:
                if iv.lo <= seg_lo and iv.hi >= seg_hi:
                    fragments.append(Interval(seg_lo, seg_hi))
                    break

        return IntervalTree(from_list(fragments, self._monoid))

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def intervals(self):
        """Get all intervals as a sorted list."""
        return to_list(self._tree)

    def count(self):
        """Number of intervals."""
        return len(self)

    def depth_at(self, point):
        """Count how many intervals contain a given point. O(log n + k)."""
        return len(self.stab(point))

    def max_depth(self):
        """Find the maximum overlap depth. O(n log n).

        Returns (max_depth, point_of_max_depth).
        """
        items = to_list(self._tree)
        if not items:
            return (0, None)

        # Sweep line: events at each lo (+1) and hi (-1)
        events = []
        for iv in items:
            events.append((iv.lo, 1))
            events.append((iv.hi, -1))
        events.sort(key=lambda e: (e[0], -e[1]))  # opens before closes at same point

        max_d = 0
        current = 0
        max_point = None
        for pos, delta in events:
            current += delta
            if current > max_d:
                max_d = current
                max_point = pos
        return (max_d, max_point)

    def histogram(self, buckets=10):
        """Create a histogram of interval start points. O(n).

        Returns list of (bucket_range, count) tuples.
        """
        items = to_list(self._tree)
        if not items:
            return []

        lo = min(iv.lo for iv in items)
        hi = max(iv.lo for iv in items)
        if lo == hi:
            return [(Interval(lo, hi), len(items))]

        width = (hi - lo) / buckets
        result = []
        for i in range(buckets):
            blo = lo + i * width
            bhi = lo + (i + 1) * width if i < buckets - 1 else hi + 0.001
            count = sum(1 for iv in items if blo <= iv.lo < bhi)
            result.append((Interval(blo, bhi), count))
        return result

    # --------------------------------------------------------
    # Bulk operations
    # --------------------------------------------------------

    @staticmethod
    def from_intervals(intervals):
        """Build interval tree from a list of intervals. O(n log n).

        Accepts Interval objects or (lo, hi) or (lo, hi, data) tuples.
        """
        ivs = []
        for item in intervals:
            if isinstance(item, Interval):
                ivs.append(item)
            elif isinstance(item, tuple):
                if len(item) == 2:
                    ivs.append(Interval(item[0], item[1]))
                elif len(item) == 3:
                    ivs.append(Interval(item[0], item[1], item[2]))
                else:
                    raise ValueError(f"Invalid interval tuple: {item}")
            else:
                raise TypeError(f"Expected Interval or tuple, got {type(item)}")

        ivs.sort(key=lambda iv: (iv.lo, iv.hi))
        return IntervalTree(from_list(ivs, IntervalMonoid))

    def filter(self, predicate):
        """Filter intervals by predicate. O(n)."""
        items = [iv for iv in to_list(self._tree) if predicate(iv)]
        return IntervalTree(from_list(items, self._monoid))

    def map_data(self, fn):
        """Apply fn to data of each interval. O(n)."""
        items = [Interval(iv.lo, iv.hi, fn(iv.data)) for iv in to_list(self._tree)]
        return IntervalTree(from_list(items, self._monoid))

    def fold(self, fn, init_val):
        """Left fold over intervals. O(n)."""
        result = init_val
        for iv in to_list(self._tree):
            result = fn(result, iv)
        return result

    # --------------------------------------------------------
    # Sweep line algorithms
    # --------------------------------------------------------

    def intersect_all(self):
        """Find the intersection of all intervals (if it exists).

        Returns Interval or None if no common intersection.
        """
        items = to_list(self._tree)
        if not items:
            return None

        lo = max(iv.lo for iv in items)
        hi = min(iv.hi for iv in items)
        if lo > hi:
            return None
        return Interval(lo, hi)

    def union_length(self):
        """Total length of the union of all intervals. Same as coverage()."""
        return self.coverage()

    def pairwise_overlaps(self):
        """Find all pairs of overlapping intervals. O(n log n + k).

        Uses sweep line for efficiency.
        Returns list of (interval1, interval2) tuples.
        """
        items = sorted(to_list(self._tree), key=lambda iv: (iv.lo, iv.hi))
        if len(items) <= 1:
            return []

        # Sweep line with active set
        events = []
        for i, iv in enumerate(items):
            events.append((iv.lo, 0, i))   # 0 = open
            events.append((iv.hi, 1, i))   # 1 = close
        events.sort(key=lambda e: (e[0], e[1]))

        active = set()
        pairs = []
        for _, etype, idx in events:
            if etype == 0:  # open
                for a in active:
                    pairs.append((items[a], items[idx]))
                active.add(idx)
            else:  # close
                active.discard(idx)

        return pairs

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self):
        """Serialize to dict."""
        return {
            'intervals': [
                {'lo': iv.lo, 'hi': iv.hi, 'data': iv.data}
                for iv in to_list(self._tree)
            ]
        }

    @staticmethod
    def from_dict(d):
        """Deserialize from dict."""
        intervals = [
            Interval(item['lo'], item['hi'], item.get('data'))
            for item in d['intervals']
        ]
        return IntervalTree.from_intervals(intervals)

    def to_list(self):
        """Return intervals as sorted list."""
        return to_list(self._tree)
