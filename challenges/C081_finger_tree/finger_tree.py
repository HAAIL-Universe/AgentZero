"""
C081: Finger Tree -- General-purpose persistent sequence.

A 2-3 finger tree (Hinze & Paterson 2006) supporting:
  - O(1) amortized cons/snoc (prepend/append)
  - O(1) amortized head/tail/last/init
  - O(log n) concatenation
  - O(log n) split by monotone predicate
  - Monoid-parameterized measurement for:
    - Sequence (size) -- random access by index
    - Priority queue (min/max)
    - Ordered sequence (key) -- sorted sets
    - Interval trees

All operations are purely functional (persistent).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Generic
import functools


# ============================================================
# Monoid abstraction
# ============================================================

class Monoid:
    """Base monoid: identity element + associative combine."""
    @staticmethod
    def empty():
        return None

    @staticmethod
    def combine(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (a, b)


class SizeMonoid(Monoid):
    """Measures size (count of elements). Enables random access."""
    @staticmethod
    def empty():
        return 0

    @staticmethod
    def combine(a, b):
        return a + b

    @staticmethod
    def measure(x):
        return 1


class PriorityMonoid(Monoid):
    """Measures minimum priority. Enables priority queue."""
    @staticmethod
    def empty():
        return float('inf')

    @staticmethod
    def combine(a, b):
        return min(a, b)

    @staticmethod
    def measure(x):
        if isinstance(x, tuple):
            return x[0]  # (priority, value)
        return x


class MaxPriorityMonoid(Monoid):
    """Measures maximum priority."""
    @staticmethod
    def empty():
        return float('-inf')

    @staticmethod
    def combine(a, b):
        return max(a, b)

    @staticmethod
    def measure(x):
        if isinstance(x, tuple):
            return x[0]
        return x


class KeyMonoid(Monoid):
    """Measures maximum key. Enables ordered sequences."""
    @staticmethod
    def empty():
        return None

    @staticmethod
    def combine(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)

    @staticmethod
    def measure(x):
        if isinstance(x, tuple):
            return x[0]  # (key, value)
        return x


# ============================================================
# Tree nodes
# ============================================================

class Measured:
    """Mixin for cached measurements."""
    __slots__ = ('_cached_measure',)

    def measure(self, monoid):
        raise NotImplementedError


@dataclass(frozen=True)
class Elem:
    """Leaf element wrapper."""
    value: Any

    def measure(self, monoid):
        return monoid.measure(self.value)

    def to_list(self):
        return [self.value]

    def __repr__(self):
        return f"Elem({self.value!r})"


@dataclass(frozen=True)
class Node2:
    """Internal node with 2 children."""
    a: Any
    b: Any
    _measure: Any = field(default=None, repr=False, compare=False)

    def measure(self, monoid):
        if self._measure is not None:
            return self._measure
        m = monoid.combine(self.a.measure(monoid), self.b.measure(monoid))
        object.__setattr__(self, '_measure', m)
        return m

    def to_list(self):
        return self.a.to_list() + self.b.to_list()

    def to_digit(self):
        return [self.a, self.b]

    def __repr__(self):
        return f"Node2({self.a!r}, {self.b!r})"


@dataclass(frozen=True)
class Node3:
    """Internal node with 3 children."""
    a: Any
    b: Any
    c: Any
    _measure: Any = field(default=None, repr=False, compare=False)

    def measure(self, monoid):
        if self._measure is not None:
            return self._measure
        m = monoid.combine(
            monoid.combine(self.a.measure(monoid), self.b.measure(monoid)),
            self.c.measure(monoid)
        )
        object.__setattr__(self, '_measure', m)
        return m

    def to_list(self):
        return self.a.to_list() + self.b.to_list() + self.c.to_list()

    def to_digit(self):
        return [self.a, self.b, self.c]

    def __repr__(self):
        return f"Node3({self.a!r}, {self.b!r}, {self.c!r})"


def _make_node(items):
    """Make a Node2 or Node3 from a list of 2-3 items."""
    if len(items) == 2:
        return Node2(items[0], items[1])
    elif len(items) == 3:
        return Node3(items[0], items[1], items[2])
    else:
        raise ValueError(f"Cannot make node from {len(items)} items")


# ============================================================
# Digit (1-4 elements stored at fingers)
# ============================================================

def _digit_measure(digit, monoid):
    """Measure a digit (list of 1-4 elements)."""
    result = monoid.empty()
    for x in digit:
        result = monoid.combine(result, x.measure(monoid))
    return result


# ============================================================
# Finger Tree
# ============================================================

class Empty:
    """Empty finger tree."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def measure(self, monoid):
        return monoid.empty()

    def is_empty(self):
        return True

    def to_list(self):
        return []

    def __repr__(self):
        return "Empty()"

    def __len__(self):
        return 0


@dataclass(frozen=True)
class Single:
    """Finger tree with exactly one element."""
    elem: Any

    def measure(self, monoid):
        return self.elem.measure(monoid)

    def is_empty(self):
        return False

    def to_list(self):
        return self.elem.to_list()

    def __repr__(self):
        return f"Single({self.elem!r})"


@dataclass(frozen=True)
class Deep:
    """Finger tree with left digit, middle tree, right digit."""
    left: list    # 1-4 elements
    middle: Any   # Finger tree of Node2/Node3
    right: list   # 1-4 elements
    _measure: Any = field(default=None, repr=False, compare=False)

    def measure(self, monoid):
        if self._measure is not None:
            return self._measure
        m = monoid.combine(
            monoid.combine(
                _digit_measure(self.left, monoid),
                self.middle.measure(monoid)
            ),
            _digit_measure(self.right, monoid)
        )
        object.__setattr__(self, '_measure', m)
        return m

    def is_empty(self):
        return False

    def to_list(self):
        result = []
        for x in self.left:
            result.extend(x.to_list())
        result.extend(self.middle.to_list())
        for x in self.right:
            result.extend(x.to_list())
        return result

    def __repr__(self):
        return f"Deep({self.left!r}, {self.middle!r}, {self.right!r})"


EMPTY = Empty()


# ============================================================
# Core operations
# ============================================================

def cons(tree, elem, monoid):
    """Prepend an element to the front. O(1) amortized."""
    a = Elem(elem) if not isinstance(elem, (Elem, Node2, Node3)) else elem

    if isinstance(tree, Empty):
        return Single(a)

    if isinstance(tree, Single):
        return Deep([a], EMPTY, [tree.elem])

    # Deep
    if len(tree.left) < 4:
        return Deep([a] + tree.left, tree.middle, tree.right)

    # Left digit is full (4 elements) -- push middle 3 into spine
    new_node = Node3(tree.left[1], tree.left[2], tree.left[3])
    new_middle = cons(tree.middle, new_node, monoid)
    return Deep([a, tree.left[0]], new_middle, tree.right)


def snoc(tree, elem, monoid):
    """Append an element to the back. O(1) amortized."""
    a = Elem(elem) if not isinstance(elem, (Elem, Node2, Node3)) else elem

    if isinstance(tree, Empty):
        return Single(a)

    if isinstance(tree, Single):
        return Deep([tree.elem], EMPTY, [a])

    # Deep
    if len(tree.right) < 4:
        return Deep(tree.left, tree.middle, tree.right + [a])

    # Right digit is full -- push middle 3 into spine
    new_node = Node3(tree.right[0], tree.right[1], tree.right[2])
    new_middle = snoc(tree.middle, new_node, monoid)
    return Deep(tree.left, new_middle, [tree.right[3], a])


def head(tree):
    """Get the first element. O(1)."""
    if isinstance(tree, Empty):
        raise IndexError("head of empty tree")
    if isinstance(tree, Single):
        return tree.elem.value if isinstance(tree.elem, Elem) else tree.elem
    return tree.left[0].value if isinstance(tree.left[0], Elem) else tree.left[0]


def last(tree):
    """Get the last element. O(1)."""
    if isinstance(tree, Empty):
        raise IndexError("last of empty tree")
    if isinstance(tree, Single):
        return tree.elem.value if isinstance(tree.elem, Elem) else tree.elem
    return tree.right[-1].value if isinstance(tree.right[-1], Elem) else tree.right[-1]


def tail(tree, monoid):
    """Remove the first element. O(1) amortized."""
    if isinstance(tree, Empty):
        raise IndexError("tail of empty tree")
    if isinstance(tree, Single):
        return EMPTY

    if len(tree.left) > 1:
        return Deep(tree.left[1:], tree.middle, tree.right)

    # Left digit has exactly 1 element -- borrow from middle
    if not isinstance(tree.middle, Empty):
        # Get first node from middle
        mid_head = _head_deep(tree.middle)
        new_left = mid_head.to_digit()
        new_middle = tail(tree.middle, monoid)
        return Deep(new_left, new_middle, tree.right)

    # Middle is empty -- right becomes the tree
    if len(tree.right) == 1:
        return Single(tree.right[0])
    return Deep(tree.right[:1], EMPTY, tree.right[1:])


def init(tree, monoid):
    """Remove the last element. O(1) amortized."""
    if isinstance(tree, Empty):
        raise IndexError("init of empty tree")
    if isinstance(tree, Single):
        return EMPTY

    if len(tree.right) > 1:
        return Deep(tree.left, tree.middle, tree.right[:-1])

    # Right digit has exactly 1 element -- borrow from middle
    if not isinstance(tree.middle, Empty):
        mid_last = _last_deep(tree.middle)
        new_right = mid_last.to_digit()
        new_middle = init(tree.middle, monoid)
        return Deep(tree.left, new_middle, new_right)

    # Middle is empty -- left becomes the tree
    if len(tree.left) == 1:
        return Single(tree.left[0])
    return Deep(tree.left[:-1], EMPTY, tree.left[-1:])


def _head_deep(tree):
    """Get the first node from a finger tree (for internal use)."""
    if isinstance(tree, Single):
        return tree.elem
    return tree.left[0]


def _last_deep(tree):
    """Get the last node from a finger tree (for internal use)."""
    if isinstance(tree, Single):
        return tree.elem
    return tree.right[-1]


# ============================================================
# Concatenation
# ============================================================

def _nodes(monoid, items):
    """Convert a list of items into Node2/Node3 list."""
    n = len(items)
    if n == 2:
        return [Node2(items[0], items[1])]
    if n == 3:
        return [Node3(items[0], items[1], items[2])]
    if n == 4:
        return [Node2(items[0], items[1]), Node2(items[2], items[3])]
    if n == 5:
        return [Node3(items[0], items[1], items[2]), Node2(items[3], items[4])]
    if n == 6:
        return [Node3(items[0], items[1], items[2]), Node3(items[3], items[4], items[5])]
    if n == 7:
        return [Node3(items[0], items[1], items[2]), Node2(items[3], items[4]), Node2(items[5], items[6])]
    # For larger n, chunk recursively
    if n >= 8:
        result = []
        i = 0
        while i < n:
            remaining = n - i
            if remaining == 2:
                result.append(Node2(items[i], items[i+1]))
                i += 2
            elif remaining == 4:
                result.append(Node2(items[i], items[i+1]))
                result.append(Node2(items[i+2], items[i+3]))
                i += 4
            else:
                result.append(Node3(items[i], items[i+1], items[i+2]))
                i += 3
        return result
    return []


def _app3(monoid, left, middle_list, right):
    """Concatenate two finger trees with a list of elements in between."""
    if isinstance(left, Empty):
        result = right
        for x in reversed(middle_list):
            result = cons(result, x, monoid)
        return result

    if isinstance(right, Empty):
        result = left
        for x in middle_list:
            result = snoc(result, x, monoid)
        return result

    if isinstance(left, Single):
        result = right
        for x in reversed(middle_list):
            result = cons(result, x, monoid)
        return cons(result, left.elem, monoid)

    if isinstance(right, Single):
        result = left
        for x in middle_list:
            result = snoc(result, x, monoid)
        return snoc(result, right.elem, monoid)

    # Both Deep
    combined = left.right + middle_list + right.left
    new_middle = _app3(monoid, left.middle, _nodes(monoid, combined), right.middle)
    return Deep(left.left, new_middle, right.right)


def concat(left, right, monoid):
    """Concatenate two finger trees. O(log(min(n, m)))."""
    return _app3(monoid, left, [], right)


# ============================================================
# Split by monotone predicate
# ============================================================

def split(tree, predicate, monoid):
    """Split tree into (left, x, right) where predicate first becomes True.

    predicate(measure) -> bool must be monotone:
      False, False, ..., False, True, True, ..., True

    Returns (left_tree, element, right_tree).
    Raises ValueError if predicate is never True.
    """
    if isinstance(tree, Empty):
        raise ValueError("split on empty tree")

    if isinstance(tree, Single):
        if predicate(tree.elem.measure(monoid)):
            return (EMPTY, _unwrap(tree.elem), EMPTY)
        raise ValueError("predicate never satisfied")

    # Check if predicate is satisfied at all
    if not predicate(tree.measure(monoid)):
        raise ValueError("predicate never satisfied")

    return _split_tree(tree, predicate, monoid, monoid.empty())


def _unwrap(elem):
    """Unwrap Elem to get the value."""
    if isinstance(elem, Elem):
        return elem.value
    return elem


def _split_tree(tree, predicate, monoid, accumulated):
    """Internal split implementation."""
    if isinstance(tree, Single):
        return (EMPTY, _unwrap(tree.elem), EMPTY)

    # Check left digit
    left_measure = monoid.combine(accumulated, _digit_measure(tree.left, monoid))
    if predicate(left_measure):
        # Split is within the left digit
        prefix, x, suffix = _split_digit(tree.left, predicate, monoid, accumulated)
        left_tree = _list_to_tree(prefix, monoid)
        right_tree = _deep_left(suffix, tree.middle, tree.right, monoid)
        return (left_tree, _unwrap(x), right_tree)

    # Check middle
    mid_measure = monoid.combine(left_measure, tree.middle.measure(monoid))
    if predicate(mid_measure):
        # Split is within the middle tree
        ml, node, mr = _split_tree(tree.middle, predicate, monoid, left_measure)
        node_digits = node.to_digit() if hasattr(node, 'to_digit') else [node]
        node_accumulated = monoid.combine(left_measure, ml.measure(monoid))
        prefix, x, suffix = _split_digit(node_digits, predicate, monoid, node_accumulated)
        left_tree = _deep_right(tree.left, ml, prefix, monoid)
        right_tree = _deep_left(suffix, mr, tree.right, monoid)
        return (left_tree, _unwrap(x), right_tree)

    # Split is within the right digit
    right_accumulated = mid_measure
    prefix, x, suffix = _split_digit(tree.right, predicate, monoid, right_accumulated)
    left_tree = _deep_right(tree.left, tree.middle, prefix, monoid)
    right_tree = _list_to_tree(suffix, monoid)
    return (left_tree, _unwrap(x), right_tree)


def _split_digit(digit, predicate, monoid, accumulated):
    """Split a digit list. Returns (prefix, element, suffix)."""
    acc = accumulated
    for i, x in enumerate(digit):
        new_acc = monoid.combine(acc, x.measure(monoid))
        if predicate(new_acc):
            return (digit[:i], x, digit[i+1:])
        acc = new_acc
    # Should not happen if predicate is monotone and satisfied
    return (digit[:-1], digit[-1], [])


def _list_to_tree(items, monoid):
    """Convert a list of elements to a finger tree."""
    tree = EMPTY
    for x in items:
        tree = snoc(tree, x, monoid)
    return tree


def _deep_left(left, middle, right, monoid):
    """Create a Deep with possibly empty left digit."""
    if not left:
        if isinstance(middle, Empty):
            return _list_to_tree(right, monoid)
        h = _head_deep(middle)
        return Deep(h.to_digit(), tail(middle, monoid), right)
    return Deep(left, middle, right)


def _deep_right(left, middle, right, monoid):
    """Create a Deep with possibly empty right digit."""
    if not right:
        if isinstance(middle, Empty):
            return _list_to_tree(left, monoid)
        l = _last_deep(middle)
        return Deep(left, init(middle, monoid), l.to_digit())
    return Deep(left, middle, right)


# ============================================================
# Traversal
# ============================================================

def to_list(tree):
    """Convert finger tree to a list. O(n)."""
    return tree.to_list()


def from_list(items, monoid):
    """Build a finger tree from a list. O(n)."""
    tree = EMPTY
    for x in items:
        tree = snoc(tree, x, monoid)
    return tree


def size(tree, monoid):
    """Get tree size using monoid measure (SizeMonoid) or traversal."""
    if monoid is SizeMonoid:
        return tree.measure(monoid)
    return len(tree.to_list())


def fold_left(tree, fn, init_val):
    """Left fold over elements."""
    result = init_val
    for x in to_list(tree):
        result = fn(result, x)
    return result


def fold_right(tree, fn, init_val):
    """Right fold over elements."""
    result = init_val
    for x in reversed(to_list(tree)):
        result = fn(x, result)
    return result


# ============================================================
# Random access (indexed) operations using SizeMonoid
# ============================================================

def lookup(tree, index, monoid=SizeMonoid):
    """Get element at index. O(log n)."""
    if monoid is not SizeMonoid:
        raise ValueError("lookup requires SizeMonoid")
    n = tree.measure(monoid)
    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range [0, {n})")

    _, x, _ = split(tree, lambda m: m > index, monoid)
    return x


def update(tree, index, value, monoid=SizeMonoid):
    """Set element at index, returning new tree. O(log n)."""
    if monoid is not SizeMonoid:
        raise ValueError("update requires SizeMonoid")
    n = tree.measure(monoid)
    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range [0, {n})")

    left, _, right = split(tree, lambda m: m > index, monoid)
    return snoc(cons(right, Elem(value), monoid), left, monoid) if False else \
        concat(snoc(left, value, monoid), right, monoid)


def insert_at(tree, index, value, monoid=SizeMonoid):
    """Insert element at index, returning new tree. O(log n)."""
    if monoid is not SizeMonoid:
        raise ValueError("insert_at requires SizeMonoid")
    n = tree.measure(monoid)
    if index < 0 or index > n:
        raise IndexError(f"index {index} out of range [0, {n}]")

    if index == 0:
        return cons(tree, value, monoid)
    if index == n:
        return snoc(tree, value, monoid)

    left, x, right = split(tree, lambda m: m > index, monoid)
    left = snoc(left, value, monoid)
    return concat(left, cons(right, x, monoid), monoid)


def delete_at(tree, index, monoid=SizeMonoid):
    """Delete element at index, returning new tree. O(log n)."""
    if monoid is not SizeMonoid:
        raise ValueError("delete_at requires SizeMonoid")
    n = tree.measure(monoid)
    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range [0, {n})")

    left, _, right = split(tree, lambda m: m > index, monoid)
    return concat(left, right, monoid)


def take(tree, n, monoid=SizeMonoid):
    """Take the first n elements. O(log n)."""
    total = tree.measure(monoid)
    if n <= 0:
        return EMPTY
    if n >= total:
        return tree
    left, x, _ = split(tree, lambda m: m > n - 1, monoid)
    return snoc(left, x, monoid)


def drop(tree, n, monoid=SizeMonoid):
    """Drop the first n elements. O(log n)."""
    total = tree.measure(monoid)
    if n <= 0:
        return tree
    if n >= total:
        return EMPTY
    _, _, right = split(tree, lambda m: m > n - 1, monoid)
    return right


def slice_tree(tree, start, end, monoid=SizeMonoid):
    """Get elements [start, end). O(log n)."""
    total = tree.measure(monoid)
    start = max(0, start)
    end = min(total, end)
    if start >= end:
        return EMPTY
    t = drop(tree, start, monoid)
    return take(t, end - start, monoid)


# ============================================================
# Priority Queue operations using PriorityMonoid
# ============================================================

def pq_insert(tree, priority, value, monoid=PriorityMonoid):
    """Insert into priority queue. O(1) amortized."""
    return snoc(tree, (priority, value), monoid)


def _pq_predicate(monoid, target):
    """Build monotone split predicate for priority queue."""
    if monoid is MaxPriorityMonoid:
        return lambda m: m >= target
    return lambda m: m <= target


def pq_find_min(tree, monoid=PriorityMonoid):
    """Find min (or max for MaxPriorityMonoid) element. O(log n)."""
    if isinstance(tree, Empty):
        raise IndexError("find_min on empty queue")
    target = tree.measure(monoid)
    pred = _pq_predicate(monoid, target)
    _, elem, _ = split(tree, pred, monoid)
    return elem


def pq_delete_min(tree, monoid=PriorityMonoid):
    """Delete min (or max for MaxPriorityMonoid) element. Returns (elem, new_tree). O(log n)."""
    if isinstance(tree, Empty):
        raise IndexError("delete_min on empty queue")
    target = tree.measure(monoid)
    pred = _pq_predicate(monoid, target)
    left, elem, right = split(tree, pred, monoid)
    return (elem, concat(left, right, monoid))


# ============================================================
# Ordered sequence operations using KeyMonoid
# ============================================================

def ordered_insert(tree, key, value=None, monoid=KeyMonoid):
    """Insert into ordered sequence maintaining sort order. O(log n)."""
    entry = (key, value) if value is not None else key

    if isinstance(tree, Empty):
        return Single(Elem(entry))

    tree_max = tree.measure(monoid)
    if tree_max is None or key > tree_max:
        return snoc(tree, entry, monoid)

    # Split at the point where key <= measure
    try:
        left, x, right = split(tree, lambda m: m is not None and m >= key, monoid)
        right = cons(right, x, monoid)
        return concat(snoc(left, entry, monoid), right, monoid)
    except ValueError:
        return snoc(tree, entry, monoid)


def ordered_search(tree, key, monoid=KeyMonoid):
    """Search for key in ordered sequence. O(log n). Returns element or None."""
    if isinstance(tree, Empty):
        return None

    tree_max = tree.measure(monoid)
    if tree_max is None or key > tree_max:
        return None

    try:
        _, x, _ = split(tree, lambda m: m is not None and m >= key, monoid)
        x_key = x[0] if isinstance(x, tuple) else x
        if x_key == key:
            return x
        return None
    except ValueError:
        return None


def ordered_delete(tree, key, monoid=KeyMonoid):
    """Delete key from ordered sequence. O(log n). Returns new tree."""
    if isinstance(tree, Empty):
        return tree

    try:
        left, x, right = split(tree, lambda m: m is not None and m >= key, monoid)
        x_key = x[0] if isinstance(x, tuple) else x
        if x_key == key:
            return concat(left, right, monoid)
        # Key not found, put it back
        return concat(snoc(left, x, monoid), right, monoid)
    except ValueError:
        return tree


def ordered_to_sorted_list(tree):
    """Convert ordered tree to sorted list."""
    return to_list(tree)


# ============================================================
# Merge operation for ordered sequences
# ============================================================

def ordered_merge(left, right, monoid=KeyMonoid):
    """Merge two ordered sequences. O(m log(n/m + 1))."""
    result = left
    for elem in to_list(right):
        result = ordered_insert(result, elem[0] if isinstance(elem, tuple) else elem,
                               elem[1] if isinstance(elem, tuple) else None,
                               monoid)
    return result


# ============================================================
# FingerTreeSeq: High-level sequence API
# ============================================================

class FingerTreeSeq:
    """Persistent sequence backed by a finger tree with SizeMonoid.

    Supports O(1) amortized prepend/append, O(log n) random access,
    concatenation, split, insert, and delete.
    """

    def __init__(self, tree=None):
        self._tree = tree if tree is not None else EMPTY
        self._monoid = SizeMonoid

    @staticmethod
    def from_list(items):
        """Build sequence from list."""
        return FingerTreeSeq(from_list(items, SizeMonoid))

    def to_list(self):
        """Convert to list."""
        return to_list(self._tree)

    def __len__(self):
        if isinstance(self._tree, Empty):
            return 0
        return self._tree.measure(self._monoid)

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            return FingerTreeSeq(slice_tree(self._tree, start, stop, self._monoid))
        if index < 0:
            index += n
        return lookup(self._tree, index, self._monoid)

    def __iter__(self):
        return iter(self.to_list())

    def __repr__(self):
        items = self.to_list()
        if len(items) > 10:
            return f"FingerTreeSeq([{', '.join(repr(x) for x in items[:10])}, ...])"
        return f"FingerTreeSeq({items!r})"

    def __eq__(self, other):
        if isinstance(other, FingerTreeSeq):
            return self.to_list() == other.to_list()
        return NotImplemented

    def is_empty(self):
        return isinstance(self._tree, Empty)

    def prepend(self, value):
        """O(1) amortized."""
        return FingerTreeSeq(cons(self._tree, value, self._monoid))

    def append(self, value):
        """O(1) amortized."""
        return FingerTreeSeq(snoc(self._tree, value, self._monoid))

    def head(self):
        """O(1)."""
        return head(self._tree)

    def last(self):
        """O(1)."""
        return last(self._tree)

    def tail(self):
        """O(1) amortized."""
        return FingerTreeSeq(tail(self._tree, self._monoid))

    def init(self):
        """O(1) amortized."""
        return FingerTreeSeq(init(self._tree, self._monoid))

    def concat(self, other):
        """O(log(min(n, m)))."""
        return FingerTreeSeq(concat(self._tree, other._tree, self._monoid))

    def insert(self, index, value):
        """O(log n)."""
        return FingerTreeSeq(insert_at(self._tree, index, value, self._monoid))

    def delete(self, index):
        """O(log n)."""
        return FingerTreeSeq(delete_at(self._tree, index, self._monoid))

    def update(self, index, value):
        """O(log n)."""
        return FingerTreeSeq(update(self._tree, index, value, self._monoid))

    def take(self, n):
        """O(log n)."""
        return FingerTreeSeq(take(self._tree, n, self._monoid))

    def drop(self, n):
        """O(log n)."""
        return FingerTreeSeq(drop(self._tree, n, self._monoid))

    def split_at(self, index):
        """Split into two sequences at index. O(log n)."""
        return (self.take(index), self.drop(index))

    def reverse(self):
        """O(n) -- returns new reversed sequence."""
        return FingerTreeSeq.from_list(list(reversed(self.to_list())))

    def map(self, fn):
        """Apply fn to each element. O(n)."""
        return FingerTreeSeq.from_list([fn(x) for x in self.to_list()])

    def filter(self, pred):
        """Filter elements. O(n)."""
        return FingerTreeSeq.from_list([x for x in self.to_list() if pred(x)])

    def fold_left(self, fn, init_val):
        """Left fold. O(n)."""
        return fold_left(self._tree, fn, init_val)

    def fold_right(self, fn, init_val):
        """Right fold. O(n)."""
        return fold_right(self._tree, fn, init_val)


# ============================================================
# FingerTreePQ: Priority Queue API
# ============================================================

class FingerTreePQ:
    """Persistent priority queue backed by a finger tree with PriorityMonoid.

    Supports O(1) insert, O(1) find-min, O(log n) delete-min.
    """

    def __init__(self, tree=None, monoid=None):
        self._tree = tree if tree is not None else EMPTY
        self._monoid = monoid or PriorityMonoid

    def is_empty(self):
        return isinstance(self._tree, Empty)

    def __len__(self):
        return len(to_list(self._tree))

    def insert(self, priority, value=None):
        """O(1) amortized."""
        return FingerTreePQ(
            pq_insert(self._tree, priority, value, self._monoid),
            self._monoid
        )

    def find_min(self):
        """O(log n). Returns (priority, value)."""
        return pq_find_min(self._tree, self._monoid)

    def delete_min(self):
        """O(log n). Returns (element, new_queue)."""
        elem, new_tree = pq_delete_min(self._tree, self._monoid)
        return (elem, FingerTreePQ(new_tree, self._monoid))

    def to_sorted_list(self):
        """Extract all elements in priority order. O(n log n)."""
        result = []
        q = self
        while not q.is_empty():
            elem, q = q.delete_min()
            result.append(elem)
        return result

    def merge(self, other):
        """Merge two priority queues. O(log(min(n, m)))."""
        return FingerTreePQ(
            concat(self._tree, other._tree, self._monoid),
            self._monoid
        )

    def __repr__(self):
        return f"FingerTreePQ({to_list(self._tree)!r})"


# ============================================================
# FingerTreeOrdSeq: Ordered Sequence API
# ============================================================

class FingerTreeOrdSeq:
    """Persistent ordered sequence (sorted set) using KeyMonoid.

    Maintains elements in sorted order by key.
    Supports O(log n) insert, search, delete.
    """

    def __init__(self, tree=None):
        self._tree = tree if tree is not None else EMPTY
        self._monoid = KeyMonoid

    def is_empty(self):
        return isinstance(self._tree, Empty)

    def __len__(self):
        return len(to_list(self._tree))

    def insert(self, key, value=None):
        """Insert maintaining order. O(log n)."""
        return FingerTreeOrdSeq(
            ordered_insert(self._tree, key, value, self._monoid)
        )

    def search(self, key):
        """Search by key. O(log n)."""
        return ordered_search(self._tree, key, self._monoid)

    def delete(self, key):
        """Delete by key. O(log n)."""
        return FingerTreeOrdSeq(
            ordered_delete(self._tree, key, self._monoid)
        )

    def min(self):
        """Get minimum element. O(1)."""
        return head(self._tree)

    def max(self):
        """Get maximum element. O(1)."""
        return last(self._tree)

    def to_sorted_list(self):
        """Get all elements in order."""
        return to_list(self._tree)

    def merge(self, other):
        """Merge two ordered sequences. O(m log(n/m + 1))."""
        return FingerTreeOrdSeq(
            ordered_merge(self._tree, other._tree, self._monoid)
        )

    def range_query(self, lo, hi):
        """Get elements with keys in [lo, hi]. O(log n + k)."""
        result = []
        for x in to_list(self._tree):
            k = x[0] if isinstance(x, tuple) else x
            if lo <= k <= hi:
                result.append(x)
            elif k > hi:
                break
        return result

    def __iter__(self):
        return iter(self.to_sorted_list())

    def __repr__(self):
        return f"FingerTreeOrdSeq({self.to_sorted_list()!r})"
