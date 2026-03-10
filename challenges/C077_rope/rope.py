"""
C077: Rope Data Structure

A persistent rope for efficient string operations. Ropes represent strings
as balanced binary trees of string fragments, enabling O(log n) concatenation,
splitting, insertion, and deletion.

Key design decisions:
- Immutable/persistent: all operations return new ropes, originals unchanged
- Weight-balanced: automatic rebalancing via Fibonacci-based thresholds
- Leaf size limit: fragments split at MAX_LEAF to maintain tree structure
- Iteration without recursion: uses explicit stack for memory efficiency
"""

import math
from collections import deque


MAX_LEAF = 64  # Maximum characters in a leaf node


# Fibonacci thresholds for balance checking (Boehm et al.)
_FIB = [1, 2]
while _FIB[-1] < 2**31:
    _FIB.append(_FIB[-1] + _FIB[-2])
MAX_DEPTH = len(_FIB) - 2


class RopeNode:
    """Base class for rope nodes."""
    __slots__ = ('_length', '_depth')

    @property
    def length(self):
        return self._length

    @property
    def depth(self):
        return self._depth


class Leaf(RopeNode):
    """Leaf node containing a string fragment."""
    __slots__ = ('text',)

    def __init__(self, text):
        self.text = text
        self._length = len(text)
        self._depth = 0


class Branch(RopeNode):
    """Internal node with left and right children."""
    __slots__ = ('left', 'right', 'weight')

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.weight = left.length  # Weight = length of left subtree
        self._length = left.length + right.length
        self._depth = 1 + max(left.depth, right.depth)


# Singleton empty leaf
_EMPTY_LEAF = Leaf("")


class Rope:
    """
    Persistent rope data structure for efficient string manipulation.

    Operations:
    - concat(other): O(log n) concatenation
    - split(index): O(log n) split into two ropes
    - insert(index, text): O(log n) insertion
    - delete(start, end): O(log n) deletion
    - charAt(index): O(log n) character access
    - substring(start, end): O(log n) substring extraction
    - balance(): O(n) rebalance for optimal depth
    - lines(): iterate over lines
    - find(pattern): find first occurrence of pattern
    - replace(old, new): replace first occurrence
    - replace_all(old, new): replace all occurrences
    """

    __slots__ = ('_root',)

    def __init__(self, root=None):
        if root is None:
            self._root = _EMPTY_LEAF
        elif isinstance(root, str):
            self._root = _build_leaves(root)
        elif isinstance(root, RopeNode):
            self._root = root
        else:
            raise TypeError(f"Cannot create Rope from {type(root)}")

    @staticmethod
    def from_string(s):
        """Create a rope from a string."""
        return Rope(s)

    @staticmethod
    def empty():
        """Create an empty rope."""
        return Rope()

    @property
    def length(self):
        """Total length of the rope."""
        return self._root.length

    def __len__(self):
        return self._root.length

    def __str__(self):
        """Convert rope to string. O(n)."""
        parts = []
        _collect(self._root, parts)
        return ''.join(parts)

    def __repr__(self):
        s = str(self)
        if len(s) > 40:
            return f"Rope({s[:37]!r}...)"
        return f"Rope({s!r})"

    def __eq__(self, other):
        if isinstance(other, Rope):
            return str(self) == str(other)
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __bool__(self):
        return self._root.length > 0

    def __iter__(self):
        """Iterate over characters."""
        return _char_iter(self._root)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._root.length)
            if step != 1:
                # Fall back to string for non-unit steps
                return Rope(str(self)[index])
            return self.substring(start, stop)
        if index < 0:
            index += self._root.length
        if index < 0 or index >= self._root.length:
            raise IndexError(f"rope index out of range: {index}")
        return self.char_at(index)

    def __add__(self, other):
        if isinstance(other, str):
            other = Rope(other)
        if isinstance(other, Rope):
            return self.concat(other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return Rope(other).concat(self)
        return NotImplemented

    def __contains__(self, item):
        if isinstance(item, str):
            return self.find(item) >= 0
        return False

    # --- Core Operations ---

    def char_at(self, index):
        """Get character at index. O(log n)."""
        if index < 0:
            index += self._root.length
        if index < 0 or index >= self._root.length:
            raise IndexError(f"rope index out of range: {index}")
        return _char_at(self._root, index)

    def concat(self, other):
        """Concatenate with another rope. O(log n)."""
        if isinstance(other, str):
            other = Rope(other)
        if not isinstance(other, Rope):
            raise TypeError(f"Cannot concat with {type(other)}")
        if self._root.length == 0:
            return other
        if other._root.length == 0:
            return self
        new_root = _concat(self._root, other._root)
        result = Rope(new_root)
        # Auto-rebalance if too deep
        if new_root.depth > MAX_DEPTH:
            result = result.balance()
        return result

    def split(self, index):
        """Split rope at index. Returns (left, right). O(log n)."""
        if index <= 0:
            return (Rope.empty(), Rope(self._root))
        if index >= self._root.length:
            return (Rope(self._root), Rope.empty())
        left, right = _split(self._root, index)
        return (Rope(left), Rope(right))

    def insert(self, index, text):
        """Insert text at index. O(log n)."""
        if isinstance(text, Rope):
            text_rope = text
        else:
            text_rope = Rope(str(text))
        if index <= 0:
            return text_rope.concat(self)
        if index >= self._root.length:
            return self.concat(text_rope)
        left, right = self.split(index)
        return left.concat(text_rope).concat(right)

    def delete(self, start, end=None):
        """Delete characters from start to end. O(log n)."""
        if end is None:
            end = start + 1
        if start < 0:
            start = 0
        if end > self._root.length:
            end = self._root.length
        if start >= end:
            return self
        left, _ = self.split(start)
        _, right = self.split(end)
        return left.concat(right)

    def substring(self, start, end=None):
        """Extract substring. O(log n)."""
        if end is None:
            end = self._root.length
        if start < 0:
            start = 0
        if end > self._root.length:
            end = self._root.length
        if start >= end:
            return Rope.empty()
        _, right = self.split(start)
        result, _ = right.split(end - start)
        return result

    def balance(self):
        """Rebalance the rope for optimal depth. O(n)."""
        leaves = []
        _collect_leaves(self._root, leaves)
        if not leaves:
            return Rope.empty()
        root = _merge_leaves(leaves, 0, len(leaves))
        return Rope(root)

    # --- String Operations ---

    def find(self, pattern, start=0):
        """Find first occurrence of pattern. Returns index or -1."""
        if isinstance(pattern, Rope):
            pattern = str(pattern)
        if not pattern:
            return start if 0 <= start <= self._root.length else -1
        plen = len(pattern)
        if plen > self._root.length - start:
            return -1
        # Use Boyer-Moore-Horspool for patterns > 3 chars
        if plen > 3:
            return _bmh_search(self, pattern, start)
        # Simple search for short patterns
        for i in range(start, self._root.length - plen + 1):
            match = True
            for j in range(plen):
                if _char_at(self._root, i + j) != pattern[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    def find_all(self, pattern):
        """Find all occurrences of pattern. Returns list of indices."""
        results = []
        if not pattern:
            return results
        pos = 0
        while pos <= self._root.length - len(pattern):
            idx = self.find(pattern, pos)
            if idx < 0:
                break
            results.append(idx)
            pos = idx + 1
        return results

    def replace(self, old, new, count=1):
        """Replace first `count` occurrences of old with new."""
        if not old:
            return self
        if isinstance(new, Rope):
            new = str(new)
        result = self
        replaced = 0
        offset = 0
        while replaced < count:
            idx = result.find(old, offset)
            if idx < 0:
                break
            result = result.delete(idx, idx + len(old)).insert(idx, new)
            offset = idx + len(new)
            replaced += 1
        return result

    def replace_all(self, old, new):
        """Replace all occurrences of old with new."""
        return self.replace(old, new, count=self._root.length + 1)

    def upper(self):
        """Convert to uppercase."""
        return Rope(str(self).upper())

    def lower(self):
        """Convert to lowercase."""
        return Rope(str(self).lower())

    def strip(self):
        """Strip whitespace from both ends."""
        s = str(self)
        return Rope(s.strip())

    def startswith(self, prefix):
        """Check if rope starts with prefix."""
        if isinstance(prefix, Rope):
            prefix = str(prefix)
        if len(prefix) > self._root.length:
            return False
        for i, c in enumerate(prefix):
            if _char_at(self._root, i) != c:
                return False
        return True

    def endswith(self, suffix):
        """Check if rope ends with suffix."""
        if isinstance(suffix, Rope):
            suffix = str(suffix)
        if len(suffix) > self._root.length:
            return False
        offset = self._root.length - len(suffix)
        for i, c in enumerate(suffix):
            if _char_at(self._root, offset + i) != c:
                return False
        return True

    def lines(self):
        """Iterate over lines (split by \\n)."""
        current = []
        ended_with_newline = False
        for ch in self:
            if ch == '\n':
                yield ''.join(current)
                current = []
                ended_with_newline = True
            else:
                current.append(ch)
                ended_with_newline = False
        if current or ended_with_newline:
            yield ''.join(current)

    def line_count(self):
        """Count number of lines."""
        count = 1
        for ch in self:
            if ch == '\n':
                count += 1
        return count

    def line_at(self, line_num):
        """Get the content of a specific line (0-indexed)."""
        current_line = 0
        current = []
        for ch in self:
            if ch == '\n':
                if current_line == line_num:
                    return ''.join(current)
                current_line += 1
                current = []
            else:
                current.append(ch)
        if current_line == line_num:
            return ''.join(current)
        raise IndexError(f"line index out of range: {line_num}")

    def char_to_line(self, char_index):
        """Convert character index to line number."""
        if char_index < 0 or char_index >= self._root.length:
            raise IndexError(f"character index out of range: {char_index}")
        line = 0
        for i, ch in enumerate(self):
            if i == char_index:
                return line
            if ch == '\n':
                line += 1
        return line

    def line_to_char(self, line_num):
        """Convert line number to character index of line start."""
        if line_num == 0:
            return 0
        current_line = 0
        for i, ch in enumerate(self):
            if ch == '\n':
                current_line += 1
                if current_line == line_num:
                    return i + 1
        raise IndexError(f"line index out of range: {line_num}")

    # --- Tree Info ---

    @property
    def depth(self):
        """Depth of the rope tree."""
        return self._root.depth

    def is_balanced(self):
        """Check if the rope is balanced (Fibonacci-balanced)."""
        return _is_balanced(self._root)

    def leaf_count(self):
        """Count number of leaf nodes."""
        leaves = []
        _collect_leaves(self._root, leaves)
        return len(leaves)

    # --- Bulk Operations ---

    @staticmethod
    def join(separator, ropes):
        """Join multiple ropes with a separator."""
        if isinstance(separator, str):
            separator = Rope(separator)
        result = Rope.empty()
        first = True
        for r in ropes:
            if not first:
                result = result.concat(separator)
            if isinstance(r, str):
                r = Rope(r)
            result = result.concat(r)
            first = False
        return result

    def split_at_string(self, delimiter):
        """Split rope at delimiter occurrences."""
        if isinstance(delimiter, Rope):
            delimiter = str(delimiter)
        parts = []
        remaining = self
        while remaining.length > 0:
            idx = remaining.find(delimiter)
            if idx < 0:
                parts.append(remaining)
                break
            left, right = remaining.split(idx)
            parts.append(left)
            _, remaining = right.split(len(delimiter))
        if remaining.length == 0 and parts:
            pass  # Don't add trailing empty
        if not parts:
            parts.append(Rope.empty())
        return parts

    def reverse(self):
        """Reverse the rope."""
        return Rope(str(self)[::-1])

    def repeat(self, n):
        """Repeat the rope n times."""
        if n <= 0:
            return Rope.empty()
        if n == 1:
            return self
        # Use doubling for efficiency
        result = Rope.empty()
        base = self
        while n > 0:
            if n & 1:
                result = result.concat(base)
            base = base.concat(base)
            n >>= 1
        return result


# --- Internal Functions ---

def _build_leaves(text):
    """Build a balanced tree from a string."""
    if not text:
        return _EMPTY_LEAF
    if len(text) <= MAX_LEAF:
        return Leaf(text)
    # Split into leaves and merge bottom-up
    leaves = []
    for i in range(0, len(text), MAX_LEAF):
        leaves.append(Leaf(text[i:i + MAX_LEAF]))
    return _merge_leaves(leaves, 0, len(leaves))


def _merge_leaves(leaves, start, end):
    """Merge a range of leaves into a balanced tree."""
    count = end - start
    if count == 1:
        return leaves[start]
    if count == 2:
        return Branch(leaves[start], leaves[start + 1])
    mid = start + count // 2
    left = _merge_leaves(leaves, start, mid)
    right = _merge_leaves(leaves, mid, end)
    return Branch(left, right)


def _collect(node, parts):
    """Collect all text fragments into a list. Iterative."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Leaf):
            if n.text:
                parts.append(n.text)
        else:
            # Push right first so left is processed first
            stack.append(n.right)
            stack.append(n.left)


def _collect_leaves(node, leaves):
    """Collect all leaf nodes. Iterative."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Leaf):
            if n.length > 0:
                leaves.append(n)
        else:
            stack.append(n.right)
            stack.append(n.left)


def _char_at(node, index):
    """Get character at index. Iterative."""
    while True:
        if isinstance(node, Leaf):
            return node.text[index]
        if index < node.weight:
            node = node.left
        else:
            index -= node.weight
            node = node.right


def _char_iter(node):
    """Iterate over all characters. Iterative using stack."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Leaf):
            yield from n.text
        else:
            stack.append(n.right)
            stack.append(n.left)


def _concat(left, right):
    """Concatenate two nodes."""
    # Merge small adjacent leaves
    if isinstance(left, Leaf) and isinstance(right, Leaf):
        if left.length + right.length <= MAX_LEAF:
            return Leaf(left.text + right.text)
    return Branch(left, right)


def _split(node, index):
    """Split a node at index. Returns (left_node, right_node)."""
    if isinstance(node, Leaf):
        return (Leaf(node.text[:index]), Leaf(node.text[index:]))
    if index == 0:
        return (_EMPTY_LEAF, node)
    if index == node.length:
        return (node, _EMPTY_LEAF)
    if index < node.weight:
        # Split is in the left subtree
        ll, lr = _split(node.left, index)
        return (ll, _concat(lr, node.right))
    elif index > node.weight:
        # Split is in the right subtree
        rl, rr = _split(node.right, index - node.weight)
        return (_concat(node.left, rl), rr)
    else:
        # Split exactly at weight boundary
        return (node.left, node.right)


def _is_balanced(node):
    """Check Fibonacci balance condition."""
    if isinstance(node, Leaf):
        return True
    if node.depth >= len(_FIB):
        return False
    return node.length >= _FIB[node.depth]


def _bmh_search(rope, pattern, start):
    """Boyer-Moore-Horspool search on rope."""
    plen = len(pattern)
    rlen = rope.length
    if plen == 0:
        return start

    # Build bad character table
    skip = {}
    for i in range(plen - 1):
        skip[pattern[i]] = plen - 1 - i

    # Extract text for efficient access (rope char_at is O(log n))
    # For large ropes, we search in chunks
    text = str(rope)  # For correctness; optimization possible later

    i = start
    while i <= rlen - plen:
        j = plen - 1
        while j >= 0 and text[i + j] == pattern[j]:
            j -= 1
        if j < 0:
            return i
        i += skip.get(text[i + plen - 1], plen)
    return -1
