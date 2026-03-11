"""
C107: Splay Tree -- Self-Adjusting Binary Search Tree

A splay tree is a BST that moves recently accessed elements to the root
via splaying operations (zig, zig-zig, zig-zag rotations). This gives
amortized O(log n) for all operations and excellent cache locality for
workloads with temporal locality.

Variants:
1. SplayTree -- core splay tree with full BST operations
2. SplayTreeMap -- key-value mapping built on splay tree
3. SplayTreeMultiSet -- allows duplicate keys with counts
4. ImplicitSplayTree -- indexed sequence (implicit keys for rope-like ops)
5. LinkCutTree -- dynamic forest via splay trees (path queries, link/cut)
"""


class SplayNode:
    """Node for standard splay tree."""
    __slots__ = ('key', 'left', 'right', 'parent', 'size')

    def __init__(self, key, parent=None):
        self.key = key
        self.left = None
        self.right = None
        self.parent = parent
        self.size = 1

    def _update_size(self):
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class SplayTree:
    """
    Self-adjusting BST with amortized O(log n) operations.

    Operations: insert, delete, find, min, max, predecessor, successor,
    split, merge, kth, rank, range query, in-order traversal.
    """

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self.find(key) is not None

    def __iter__(self):
        """In-order traversal."""
        yield from self._inorder(self.root)

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield node.key
        yield from self._inorder(node.right)

    # -- Rotations --

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x._update_size()
        y._update_size()

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x._update_size()
        y._update_size()

    # -- Splay --

    def _splay(self, x):
        """Splay node x to the root."""
        if x is None:
            return
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g is None:
                # Zig step
                if x is p.left:
                    self._rotate_right(p)
                else:
                    self._rotate_left(p)
            elif x is p.left and p is g.left:
                # Zig-zig (left-left)
                self._rotate_right(g)
                self._rotate_right(p)
            elif x is p.right and p is g.right:
                # Zig-zig (right-right)
                self._rotate_left(g)
                self._rotate_left(p)
            elif x is p.right and p is g.left:
                # Zig-zag (left-right)
                self._rotate_left(p)
                self._rotate_right(g)
            else:
                # Zig-zag (right-left)
                self._rotate_right(p)
                self._rotate_left(g)
        self.root = x

    # -- Core operations --

    def insert(self, key):
        """Insert key. Returns True if new, False if duplicate."""
        if self.root is None:
            self.root = SplayNode(key)
            self._size = 1
            return True

        node = self.root
        while True:
            if key == node.key:
                self._splay(node)
                return False
            elif key < node.key:
                if node.left is None:
                    node.left = SplayNode(key, parent=node)
                    self._size += 1
                    self._splay(node.left)
                    return True
                node = node.left
            else:
                if node.right is None:
                    node.right = SplayNode(key, parent=node)
                    self._size += 1
                    self._splay(node.right)
                    return True
                node = node.right

    def find(self, key):
        """Find key. Returns key if found, None otherwise. Splays closest node."""
        node = self.root
        last = None
        while node is not None:
            last = node
            if key == node.key:
                self._splay(node)
                return node.key
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        if last:
            self._splay(last)
        return None

    def delete(self, key):
        """Delete key. Returns True if deleted, False if not found."""
        if self.root is None:
            return False

        # Splay the key to root
        if self.find(key) is None:
            return False
        if self.root.key != key:
            return False

        # Root is now the node to delete
        self._size -= 1
        left = self.root.left
        right = self.root.right

        if left is None:
            self.root = right
            if right:
                right.parent = None
        elif right is None:
            self.root = left
            left.parent = None
        else:
            # Splay max of left subtree
            left.parent = None
            right.parent = None
            self.root = left
            # Find max in left
            m = left
            while m.right:
                m = m.right
            self._splay(m)
            # Now root is max of left, has no right child
            self.root.right = right
            right.parent = self.root
            self.root._update_size()

        return True

    def minimum(self):
        """Return minimum key, or None if empty."""
        if self.root is None:
            return None
        node = self.root
        while node.left:
            node = node.left
        self._splay(node)
        return node.key

    def maximum(self):
        """Return maximum key, or None if empty."""
        if self.root is None:
            return None
        node = self.root
        while node.right:
            node = node.right
        self._splay(node)
        return node.key

    def predecessor(self, key):
        """Return largest key < key, or None."""
        if self.root is None:
            return None
        self.find(key)
        # After splay, root is closest to key
        if self.root.key < key:
            return self.root.key
        # Check left subtree max
        if self.root.left:
            node = self.root.left
            while node.right:
                node = node.right
            return node.key
        return None

    def successor(self, key):
        """Return smallest key > key, or None."""
        if self.root is None:
            return None
        self.find(key)
        if self.root.key > key:
            return self.root.key
        if self.root.right:
            node = self.root.right
            while node.left:
                node = node.left
            return node.key
        return None

    def kth(self, k):
        """Return k-th smallest key (0-indexed). None if out of range."""
        if k < 0 or k >= self._size:
            return None
        node = self.root
        while node:
            left_size = node.left.size if node.left else 0
            if k == left_size:
                self._splay(node)
                return node.key
            elif k < left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        return None

    def rank(self, key):
        """Return number of keys strictly less than key."""
        if self.root is None:
            return 0
        self.find(key)
        # After splay, root is closest
        left_size = self.root.left.size if self.root.left else 0
        if self.root.key < key:
            return left_size + 1
        else:
            return left_size

    def range_query(self, lo, hi):
        """Return sorted list of keys in [lo, hi]."""
        result = []
        self._range_collect(self.root, lo, hi, result)
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

    def split(self, key):
        """Split into (left_tree, right_tree) where left has keys <= key."""
        if self.root is None:
            return SplayTree(), SplayTree()

        self.find(key)
        # Root is closest to key
        if self.root.key <= key:
            left = SplayTree()
            right = SplayTree()
            left.root = self.root
            right.root = self.root.right
            if right.root:
                right.root.parent = None
            left.root.right = None
            left.root._update_size()
            left._size = left.root.size
            right._size = right.root.size if right.root else 0
        else:
            left = SplayTree()
            right = SplayTree()
            right.root = self.root
            left.root = self.root.left
            if left.root:
                left.root.parent = None
            right.root.left = None
            right.root._update_size()
            right._size = right.root.size
            left._size = left.root.size if left.root else 0

        self.root = None
        self._size = 0
        return left, right

    @staticmethod
    def merge(left, right):
        """Merge two splay trees where all keys in left < all keys in right."""
        if left.root is None:
            return right
        if right.root is None:
            return left

        # Splay max of left to root
        node = left.root
        while node.right:
            node = node.right
        left._splay(node)

        # Attach right as right child
        left.root.right = right.root
        right.root.parent = left.root
        left.root._update_size()
        left._size = left.root.size

        right.root = None
        right._size = 0
        return left

    def to_sorted_list(self):
        """Return all keys as sorted list."""
        return list(self)

    def clear(self):
        """Remove all elements."""
        self.root = None
        self._size = 0


# -- Variant 2: SplayTreeMap --

class MapNode:
    """Node for key-value splay tree."""
    __slots__ = ('key', 'value', 'left', 'right', 'parent', 'size')

    def __init__(self, key, value, parent=None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = parent
        self.size = 1

    def _update_size(self):
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class SplayTreeMap:
    """
    Ordered key-value map backed by splay tree.
    Supports get, put, delete, floor, ceiling, range queries.
    """

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        self.get(key)
        return self.root is not None and self.root.key == key

    def __getitem__(self, key):
        self.get(key)
        if self.root is None or self.root.key != key:
            raise KeyError(key)
        return self.root.value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    def __iter__(self):
        yield from self._inorder(self.root)

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield (node.key, node.value)
        yield from self._inorder(node.right)

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x._update_size()
        y._update_size()

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x._update_size()
        y._update_size()

    def _splay(self, x):
        if x is None:
            return
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g is None:
                if x is p.left:
                    self._rotate_right(p)
                else:
                    self._rotate_left(p)
            elif x is p.left and p is g.left:
                self._rotate_right(g)
                self._rotate_right(p)
            elif x is p.right and p is g.right:
                self._rotate_left(g)
                self._rotate_left(p)
            elif x is p.right and p is g.left:
                self._rotate_left(p)
                self._rotate_right(g)
            else:
                self._rotate_right(p)
                self._rotate_left(g)
        self.root = x

    def put(self, key, value):
        """Insert or update key-value pair. Returns True if new key."""
        if self.root is None:
            self.root = MapNode(key, value)
            self._size = 1
            return True

        node = self.root
        while True:
            if key == node.key:
                node.value = value
                self._splay(node)
                return False
            elif key < node.key:
                if node.left is None:
                    node.left = MapNode(key, value, parent=node)
                    self._size += 1
                    self._splay(node.left)
                    return True
                node = node.left
            else:
                if node.right is None:
                    node.right = MapNode(key, value, parent=node)
                    self._size += 1
                    self._splay(node.right)
                    return True
                node = node.right

    def get(self, key, default=None):
        """Get value for key, or default if not found."""
        node = self.root
        last = None
        while node is not None:
            last = node
            if key == node.key:
                self._splay(node)
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        if last:
            self._splay(last)
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if self.root is None:
            return False
        self.get(key)
        if self.root is None or self.root.key != key:
            return False

        self._size -= 1
        left = self.root.left
        right = self.root.right

        if left is None:
            self.root = right
            if right:
                right.parent = None
        elif right is None:
            self.root = left
            left.parent = None
        else:
            left.parent = None
            right.parent = None
            self.root = left
            m = left
            while m.right:
                m = m.right
            self._splay(m)
            self.root.right = right
            right.parent = self.root
            self.root._update_size()

        return True

    def floor(self, key):
        """Return (k, v) with largest k <= key, or None."""
        if self.root is None:
            return None
        self.get(key)
        if self.root.key <= key:
            return (self.root.key, self.root.value)
        if self.root.left:
            node = self.root.left
            while node.right:
                node = node.right
            return (node.key, node.value)
        return None

    def ceiling(self, key):
        """Return (k, v) with smallest k >= key, or None."""
        if self.root is None:
            return None
        self.get(key)
        if self.root.key >= key:
            return (self.root.key, self.root.value)
        if self.root.right:
            node = self.root.right
            while node.left:
                node = node.left
            return (node.key, node.value)
        return None

    def range_query(self, lo, hi):
        """Return sorted list of (key, value) pairs in [lo, hi]."""
        result = []
        self._range_collect(self.root, lo, hi, result)
        return result

    def _range_collect(self, node, lo, hi, result):
        if node is None:
            return
        if lo < node.key:
            self._range_collect(node.left, lo, hi, result)
        if lo <= node.key <= hi:
            result.append((node.key, node.value))
        if node.key < hi:
            self._range_collect(node.right, lo, hi, result)

    def items(self):
        """Return all (key, value) pairs in sorted order."""
        return list(self)

    def keys(self):
        """Return all keys in sorted order."""
        return [k for k, _ in self]

    def values(self):
        """Return all values in key order."""
        return [v for _, v in self]

    def min_entry(self):
        """Return (key, value) with smallest key, or None."""
        if self.root is None:
            return None
        node = self.root
        while node.left:
            node = node.left
        self._splay(node)
        return (node.key, node.value)

    def max_entry(self):
        """Return (key, value) with largest key, or None."""
        if self.root is None:
            return None
        node = self.root
        while node.right:
            node = node.right
        self._splay(node)
        return (node.key, node.value)


# -- Variant 3: SplayTreeMultiSet --

class MultiSetNode:
    """Node with count for multiset."""
    __slots__ = ('key', 'count', 'left', 'right', 'parent', 'size', 'total')

    def __init__(self, key, count=1, parent=None):
        self.key = key
        self.count = count
        self.left = None
        self.right = None
        self.parent = parent
        self.size = 1    # distinct elements
        self.total = count  # total with multiplicity

    def _update(self):
        self.size = 1
        self.total = self.count
        if self.left:
            self.size += self.left.size
            self.total += self.left.total
        if self.right:
            self.size += self.right.size
            self.total += self.right.total


class SplayTreeMultiSet:
    """
    Multiset (bag) with splay tree backing.
    Supports duplicate keys with counts.
    """

    def __init__(self):
        self.root = None
        self._distinct = 0
        self._total = 0

    def __len__(self):
        """Total count including duplicates."""
        return self._total

    @property
    def distinct_count(self):
        return self._distinct

    def __contains__(self, key):
        return self.count(key) > 0

    def __iter__(self):
        """Yields each key * its count."""
        for key, cnt in self._inorder(self.root):
            for _ in range(cnt):
                yield key

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield (node.key, node.count)
        yield from self._inorder(node.right)

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x._update()
        y._update()

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x._update()
        y._update()

    def _splay(self, x):
        if x is None:
            return
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g is None:
                if x is p.left:
                    self._rotate_right(p)
                else:
                    self._rotate_left(p)
            elif x is p.left and p is g.left:
                self._rotate_right(g)
                self._rotate_right(p)
            elif x is p.right and p is g.right:
                self._rotate_left(g)
                self._rotate_left(p)
            elif x is p.right and p is g.left:
                self._rotate_left(p)
                self._rotate_right(g)
            else:
                self._rotate_right(p)
                self._rotate_left(g)
        self.root = x

    def _find_node(self, key):
        node = self.root
        last = None
        while node:
            last = node
            if key == node.key:
                self._splay(node)
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        if last:
            self._splay(last)
        return None

    def add(self, key, count=1):
        """Add key with given count."""
        if count <= 0:
            return
        if self.root is None:
            self.root = MultiSetNode(key, count)
            self._distinct = 1
            self._total = count
            return

        node = self._find_node(key)
        if node and node.key == key:
            node.count += count
            node._update()
            self._total += count
        else:
            # Insert new node
            new = MultiSetNode(key, count)
            if self.root is None:
                self.root = new
            elif key < self.root.key:
                new.left = self.root.left
                new.right = self.root
                if new.left:
                    new.left.parent = new
                self.root.left = None
                self.root.parent = new
                self.root._update()
                self.root = new
            else:
                new.right = self.root.right
                new.left = self.root
                if new.right:
                    new.right.parent = new
                self.root.right = None
                self.root.parent = new
                self.root._update()
                self.root = new
            new.parent = None
            new._update()
            self._distinct += 1
            self._total += count

    def remove(self, key, count=1):
        """Remove count occurrences of key. Returns number actually removed."""
        node = self._find_node(key)
        if node is None or node.key != key:
            return 0

        removed = min(count, node.count)
        node.count -= removed
        self._total -= removed

        if node.count <= 0:
            # Remove node entirely
            self._distinct -= 1
            left = self.root.left
            right = self.root.right
            if left is None:
                self.root = right
                if right:
                    right.parent = None
            elif right is None:
                self.root = left
                left.parent = None
            else:
                left.parent = None
                right.parent = None
                self.root = left
                m = left
                while m.right:
                    m = m.right
                self._splay(m)
                self.root.right = right
                right.parent = self.root
                self.root._update()
        else:
            node._update()

        return removed

    def count(self, key):
        """Return count of key."""
        node = self._find_node(key)
        if node and node.key == key:
            return node.count
        return 0

    def distinct_keys(self):
        """Return sorted list of distinct keys."""
        return [k for k, _ in self._inorder(self.root)]

    def most_common(self, n=None):
        """Return keys sorted by count (descending)."""
        items = list(self._inorder(self.root))
        items.sort(key=lambda x: (-x[1], x[0]))
        if n is not None:
            items = items[:n]
        return items


# -- Variant 4: ImplicitSplayTree --

class ImplicitNode:
    """Node for implicit-key splay tree (indexed sequence)."""
    __slots__ = ('value', 'left', 'right', 'parent', 'size', 'rev')

    def __init__(self, value, parent=None):
        self.value = value
        self.left = None
        self.right = None
        self.parent = parent
        self.size = 1
        self.rev = False  # lazy reversal flag

    def _update_size(self):
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class ImplicitSplayTree:
    """
    Implicit-key splay tree for sequence operations.
    Supports O(log n) amortized: insert at index, delete at index,
    access by index, reverse range, split, merge.
    """

    def __init__(self):
        self.root = None

    def __len__(self):
        return self.root.size if self.root else 0

    def _push_down(self, node):
        if node and node.rev:
            node.left, node.right = node.right, node.left
            if node.left:
                node.left.rev = not node.left.rev
            if node.right:
                node.right.rev = not node.right.rev
            node.rev = False

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x._update_size()
        y._update_size()

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x._update_size()
        y._update_size()

    def _splay(self, x):
        if x is None:
            return
        while x.parent is not None:
            p = x.parent
            g = p.parent
            if g:
                # Push down grandparent and parent before rotations
                self._push_down(g)
            self._push_down(p)
            self._push_down(x)
            if g is None:
                if x is p.left:
                    self._rotate_right(p)
                else:
                    self._rotate_left(p)
            elif x is p.left and p is g.left:
                self._rotate_right(g)
                self._rotate_right(p)
            elif x is p.right and p is g.right:
                self._rotate_left(g)
                self._rotate_left(p)
            elif x is p.right and p is g.left:
                self._rotate_left(p)
                self._rotate_right(g)
            else:
                self._rotate_right(p)
                self._rotate_left(g)
        self._push_down(x)
        self.root = x

    def _find_kth(self, k):
        """Find and splay the k-th node (0-indexed)."""
        node = self.root
        while node:
            self._push_down(node)
            left_size = node.left.size if node.left else 0
            if k == left_size:
                self._splay(node)
                return node
            elif k < left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        return None

    def get(self, index):
        """Get value at index."""
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        node = self._find_kth(index)
        return node.value

    def set(self, index, value):
        """Set value at index."""
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        node = self._find_kth(index)
        node.value = value

    def insert(self, index, value):
        """Insert value at index (0 <= index <= len)."""
        if index < 0 or index > len(self):
            raise IndexError("index out of range")
        new_node = ImplicitNode(value)
        if self.root is None:
            self.root = new_node
            return

        if index == len(self):
            # Append: splay rightmost
            node = self.root
            self._push_down(node)
            while node.right:
                node = node.right
                self._push_down(node)
            self._splay(node)
            node.right = new_node
            new_node.parent = node
            node._update_size()
        elif index == 0:
            # Prepend: splay leftmost
            node = self.root
            self._push_down(node)
            while node.left:
                node = node.left
                self._push_down(node)
            self._splay(node)
            node.left = new_node
            new_node.parent = node
            node._update_size()
        else:
            # Split at index: splay index-1 and index
            self._find_kth(index - 1)
            right = self.root.right
            self.root.right = new_node
            new_node.parent = self.root
            new_node.right = right
            if right:
                right.parent = new_node
            new_node._update_size()
            self.root._update_size()

    def delete(self, index):
        """Delete value at index. Returns the deleted value."""
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        node = self._find_kth(index)
        value = node.value

        left = self.root.left
        right = self.root.right

        if left is None:
            self.root = right
            if right:
                right.parent = None
        elif right is None:
            self.root = left
            left.parent = None
        else:
            left.parent = None
            right.parent = None
            self.root = left
            # Splay max of left
            m = left
            self._push_down(m)
            while m.right:
                m = m.right
                self._push_down(m)
            self._splay(m)
            self.root.right = right
            right.parent = self.root
            self.root._update_size()

        return value

    def append(self, value):
        """Append value to end."""
        self.insert(len(self), value)

    def reverse_range(self, lo, hi):
        """Reverse elements in range [lo, hi] (inclusive, 0-indexed)."""
        n = len(self)
        if lo < 0 or hi >= n or lo > hi:
            return
        if lo == hi:
            return

        # We need to isolate [lo, hi] as a subtree
        # Strategy: splay lo-1 to root, splay hi+1 to right child
        # Then left child of hi+1 is the range [lo, hi]
        if lo == 0 and hi == n - 1:
            # Reverse entire tree
            if self.root:
                self.root.rev = not self.root.rev
            return

        if lo == 0:
            # Splay hi+1 to root
            self._find_kth(hi + 1)
            # Left subtree is [0, hi]
            if self.root.left:
                self.root.left.rev = not self.root.left.rev
        elif hi == n - 1:
            # Splay lo-1 to root
            self._find_kth(lo - 1)
            # Right subtree is [lo, n-1]
            if self.root.right:
                self.root.right.rev = not self.root.right.rev
        else:
            # Splay lo-1 to root
            self._find_kth(lo - 1)
            # Splay hi+1 in right subtree
            # Detach right, splay kth=0 in it (which is index lo, the first in right)
            # Actually: we need hi+1 relative to right subtree
            right = self.root.right
            if right is None:
                return
            right.parent = None
            old_root = self.root
            self.root = right
            target = hi - lo + 1  # hi+1 relative to right subtree = hi - (lo-1+1) + 1 = hi - lo + 1
            self._find_kth(target)
            # Now self.root is hi+1 node, its left child is [lo, hi]
            if self.root.left:
                self.root.left.rev = not self.root.left.rev
            # Reattach
            old_root.right = self.root
            self.root.parent = old_root
            old_root._update_size()
            self.root = old_root

    def to_list(self):
        """Convert to Python list."""
        result = []
        self._collect(self.root, result)
        return result

    def _collect(self, node, result):
        if node is None:
            return
        self._push_down(node)
        self._collect(node.left, result)
        result.append(node.value)
        self._collect(node.right, result)

    @staticmethod
    def from_list(lst):
        """Build from list in O(n)."""
        tree = ImplicitSplayTree()
        for v in lst:
            tree.append(v)
        return tree


# -- Variant 5: Link-Cut Tree --

class LCTNode:
    """Node for Link-Cut Tree (path-based access in dynamic forests)."""
    __slots__ = ('id', 'value', 'left', 'right', 'parent', 'rev',
                 'aggregate', 'size')

    def __init__(self, node_id, value=0):
        self.id = node_id
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.rev = False
        self.aggregate = value  # path aggregate (sum)
        self.size = 1

    def _update(self):
        self.size = 1
        self.aggregate = self.value
        if self.left:
            self.size += self.left.size
            self.aggregate += self.left.aggregate
        if self.right:
            self.aggregate += self.right.aggregate
            self.size += self.right.size


class LinkCutTree:
    """
    Dynamic forest supporting O(log n) amortized:
    - link(u, v): add edge between trees
    - cut(u, v): remove edge
    - connected(u, v): check if same tree
    - path_aggregate(u, v): query sum on path
    - lca(u, v): lowest common ancestor (in rooted mode)
    """

    def __init__(self, n):
        """Create forest of n isolated nodes (0-indexed)."""
        self.nodes = [LCTNode(i) for i in range(n)]

    def _is_root(self, x):
        """Check if x is root of its splay tree (auxiliary tree root)."""
        return (x.parent is None or
                (x.parent.left is not x and x.parent.right is not x))

    def _push_down(self, x):
        if x and x.rev:
            x.left, x.right = x.right, x.left
            if x.left:
                x.left.rev = not x.left.rev
            if x.right:
                x.right.rev = not x.right.rev
            x.rev = False

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if x.parent:
            if x is x.parent.left:
                x.parent.left = y
            elif x is x.parent.right:
                x.parent.right = y
        y.left = x
        x.parent = y
        x._update()
        y._update()

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if x.parent:
            if x is x.parent.right:
                x.parent.right = y
            elif x is x.parent.left:
                x.parent.left = y
        y.right = x
        x.parent = y
        x._update()
        y._update()

    def _splay(self, x):
        # Push down from root of splay tree to x
        path = []
        y = x
        while not self._is_root(y):
            path.append(y.parent)
            y = y.parent
        path.append(y)
        for node in reversed(path):
            self._push_down(node)
        self._push_down(x)

        while not self._is_root(x):
            p = x.parent
            if self._is_root(p):
                if x is p.left:
                    self._rotate_right(p)
                else:
                    self._rotate_left(p)
            else:
                g = p.parent
                if x is p.left and p is g.left:
                    self._rotate_right(g)
                    self._rotate_right(p)
                elif x is p.right and p is g.right:
                    self._rotate_left(g)
                    self._rotate_left(p)
                elif x is p.right and p is g.left:
                    self._rotate_left(p)
                    self._rotate_right(g)
                else:
                    self._rotate_right(p)
                    self._rotate_left(g)

    def _access(self, x):
        """Make x to root path a preferred path. Returns last node accessed."""
        last = None
        y = x
        while y is not None:
            self._splay(y)
            y.right = last
            y._update()
            last = y
            y = y.parent
        self._splay(x)
        return last

    def _make_root(self, x):
        """Make x the root of its represented tree."""
        self._access(x)
        x.rev = not x.rev
        self._push_down(x)

    def _find_root(self, x):
        """Find root of the represented tree containing x."""
        self._access(x)
        r = x
        self._push_down(r)
        while r.left:
            r = r.left
            self._push_down(r)
        self._splay(r)
        return r

    def link(self, u, v):
        """Link node u to node v (make u's tree a subtree of v's tree).
        Returns False if already connected."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._make_root(nu)
        if self._find_root(nv) is nu:
            return False  # Already connected
        nu.parent = nv
        return True

    def cut(self, u, v):
        """Cut edge between u and v. Returns False if no direct edge."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._make_root(nu)
        self._access(nv)
        # nv should have nu as left child if they're directly connected
        if nv.left is not nu or nu.right is not None:
            return False
        nv.left = None
        nu.parent = None
        nv._update()
        return True

    def connected(self, u, v):
        """Check if u and v are in the same tree."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._make_root(nu)
        return self._find_root(nv) is nu

    def path_aggregate(self, u, v):
        """Return sum of values on path from u to v."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._make_root(nu)
        self._access(nv)
        return nv.aggregate

    def set_value(self, u, value):
        """Set value of node u."""
        nu = self.nodes[u]
        self._access(nu)
        nu.value = value
        nu._update()

    def get_value(self, u):
        """Get value of node u."""
        return self.nodes[u].value

    def lca(self, u, v):
        """Find LCA of u and v (relative to current tree root). Returns node id."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._access(nu)
        last = self._access(nv)
        return last.id

    def path_length(self, u, v):
        """Return number of nodes on path from u to v."""
        nu = self.nodes[u]
        nv = self.nodes[v]
        self._make_root(nu)
        self._access(nv)
        return nv.size
