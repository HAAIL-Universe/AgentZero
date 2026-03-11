"""
C108: Red-Black Tree -- Self-balancing BST with O(log n) operations.

Properties (invariants):
1. Every node is red or black
2. Root is black
3. Every leaf (NIL) is black
4. Red nodes have only black children
5. All paths from a node to its leaves have the same black-height

Variants:
- RedBlackTree: ordered set (unique keys)
- RedBlackMap: ordered key-value map
- RedBlackMultiMap: ordered map allowing duplicate keys
- IntervalMap: interval-based queries using augmented RB tree
- OrderStatisticTree: rank/select queries using augmented RB tree
"""

RED = True
BLACK = False


class Node:
    __slots__ = ('key', 'value', 'color', 'left', 'right', 'parent', 'size', 'max_end')

    def __init__(self, key, value=None, color=RED):
        self.key = key
        self.value = value
        self.color = color
        self.left = None
        self.right = None
        self.parent = None
        self.size = 1
        self.max_end = None  # for interval tree augmentation


# Sentinel NIL node (shared across all trees)
NIL = Node(key=None, value=None, color=BLACK)
NIL.left = NIL
NIL.right = NIL
NIL.parent = NIL
NIL.size = 0


class RedBlackTree:
    """Ordered set backed by a red-black tree."""

    def __init__(self, items=None):
        self.root = NIL
        self._size = 0
        if items:
            for item in items:
                self.insert(item)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find(key) is not NIL

    def __iter__(self):
        return self._inorder(self.root)

    def __reversed__(self):
        return self._reverse_inorder(self.root)

    def __repr__(self):
        items = list(self)
        return f"RedBlackTree({items})"

    # --- Core operations ---

    def insert(self, key):
        """Insert key. Returns True if new, False if already existed."""
        node = Node(key, color=RED)
        node.left = NIL
        node.right = NIL

        parent = NIL
        current = self.root

        while current is not NIL:
            parent = current
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                return False  # duplicate

        node.parent = parent
        if parent is NIL:
            self.root = node
        elif key < parent.key:
            parent.left = node
        else:
            parent.right = node

        self._size += 1
        self._update_size_up(node.parent)
        self._insert_fixup(node)
        return True

    def delete(self, key):
        """Delete key. Returns True if found, False if not."""
        z = self._find(key)
        if z is NIL:
            return False
        self._delete_node(z)
        return True

    def find(self, key):
        """Return key if present, else None."""
        node = self._find(key)
        return node.key if node is not NIL else None

    def min(self):
        """Return minimum key, or None if empty."""
        if self.root is NIL:
            return None
        return self._minimum(self.root).key

    def max(self):
        """Return maximum key, or None if empty."""
        if self.root is NIL:
            return None
        return self._maximum(self.root).key

    def floor(self, key):
        """Largest key <= given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if key == node.key:
                return node.key
            elif key < node.key:
                node = node.left
            else:
                result = node.key
                node = node.right
        return result

    def ceiling(self, key):
        """Smallest key >= given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if key == node.key:
                return node.key
            elif key > node.key:
                node = node.right
            else:
                result = node.key
                node = node.left
        return result

    def range(self, lo, hi):
        """Return all keys in [lo, hi] in order."""
        result = []
        self._range_collect(self.root, lo, hi, result)
        return result

    def successor(self, key):
        """Return smallest key strictly greater than given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if node.key > key:
                result = node.key
                node = node.left
            else:
                node = node.right
        return result

    def predecessor(self, key):
        """Return largest key strictly less than given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if node.key < key:
                result = node.key
                node = node.right
            else:
                node = node.left
        return result

    def pop_min(self):
        """Remove and return the minimum key."""
        if self.root is NIL:
            raise ValueError("pop from empty tree")
        node = self._minimum(self.root)
        key = node.key
        self._delete_node(node)
        return key

    def pop_max(self):
        """Remove and return the maximum key."""
        if self.root is NIL:
            raise ValueError("pop from empty tree")
        node = self._maximum(self.root)
        key = node.key
        self._delete_node(node)
        return key

    def clear(self):
        """Remove all elements."""
        self.root = NIL
        self._size = 0

    def to_sorted_list(self):
        """Return all keys as a sorted list."""
        return list(self)

    def height(self):
        """Return the height of the tree."""
        return self._height(self.root)

    def black_height(self):
        """Return the black-height of the tree (should be consistent)."""
        h = 0
        node = self.root
        while node is not NIL:
            if node.color == BLACK:
                h += 1
            node = node.left
        return h

    def is_valid(self):
        """Validate all red-black tree invariants."""
        if self.root is NIL:
            return True
        # Property 2: root is black
        if self.root.color != BLACK:
            return False
        # Check properties 4 and 5
        return self._validate(self.root)[0]

    # --- Internal helpers ---

    def _find(self, key):
        node = self.root
        while node is not NIL:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return NIL

    def _minimum(self, node):
        while node.left is not NIL:
            node = node.left
        return node

    def _maximum(self, node):
        while node.right is not NIL:
            node = node.right
        return node

    def _inorder(self, node):
        if node is not NIL:
            yield from self._inorder(node.left)
            yield node.key
            yield from self._inorder(node.right)

    def _reverse_inorder(self, node):
        if node is not NIL:
            yield from self._reverse_inorder(node.right)
            yield node.key
            yield from self._reverse_inorder(node.left)

    def _range_collect(self, node, lo, hi, result):
        if node is NIL:
            return
        if lo < node.key:
            self._range_collect(node.left, lo, hi, result)
        if lo <= node.key <= hi:
            result.append(node.key)
        if node.key < hi:
            self._range_collect(node.right, lo, hi, result)

    def _height(self, node):
        if node is NIL:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _validate(self, node):
        """Returns (valid, black_height)."""
        if node is NIL:
            return True, 1

        # Property 4: red node must have black children
        if node.color == RED:
            if node.left.color == RED or node.right.color == RED:
                return False, 0

        left_valid, left_bh = self._validate(node.left)
        if not left_valid:
            return False, 0

        right_valid, right_bh = self._validate(node.right)
        if not right_valid:
            return False, 0

        # Property 5: equal black heights
        if left_bh != right_bh:
            return False, 0

        bh = left_bh + (1 if node.color == BLACK else 0)
        return True, bh

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # Update sizes
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        # Update sizes
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _update_size_up(self, node):
        while node is not NIL:
            node.size = node.left.size + node.right.size + 1
            node = node.parent

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right  # uncle
                if y.color == RED:
                    # Case 1: uncle is red
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        # Case 2: z is right child
                        z = z.parent
                        self._rotate_left(z)
                    # Case 3: z is left child
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                # Mirror cases
                y = z.parent.parent.left  # uncle
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    def _transplant(self, u, v):
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_node(self, z):
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
            self._update_size_up(z.parent)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
            self._update_size_up(z.parent)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y  # important when x is NIL
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self._update_size_up(x.parent if x.parent is not NIL else y)
            y.size = y.left.size + y.right.size + 1

        self._size -= 1
        if y_original_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right  # sibling
                if w.color == RED:
                    # Case 1: sibling is red
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    # Case 2: both of sibling's children are black
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        # Case 3: sibling's right child is black
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    # Case 4: sibling's right child is red
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                # Mirror cases
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK


class RedBlackMap:
    """Ordered key-value map backed by a red-black tree."""

    def __init__(self, items=None):
        self.root = NIL
        self._size = 0
        if items:
            if isinstance(items, dict):
                items = items.items()
            for k, v in items:
                self.put(k, v)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find(key) is not NIL

    def __getitem__(self, key):
        node = self._find(key)
        if node is NIL:
            raise KeyError(key)
        return node.value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        if not self.delete(key):
            raise KeyError(key)

    def __iter__(self):
        return self._inorder_keys(self.root)

    def __repr__(self):
        pairs = [(k, self[k]) for k in self]
        return f"RedBlackMap({pairs})"

    def put(self, key, value):
        """Insert or update key-value pair. Returns True if new key."""
        # Check if key exists first
        existing = self._find(key)
        if existing is not NIL:
            existing.value = value
            return False

        node = Node(key, value, color=RED)
        node.left = NIL
        node.right = NIL

        parent = NIL
        current = self.root

        while current is not NIL:
            parent = current
            if key < current.key:
                current = current.left
            else:
                current = current.right

        node.parent = parent
        if parent is NIL:
            self.root = node
        elif key < parent.key:
            parent.left = node
        else:
            parent.right = node

        self._size += 1
        self._update_size_up(node.parent)
        self._insert_fixup(node)
        return True

    def get(self, key, default=None):
        node = self._find(key)
        return node.value if node is not NIL else default

    def delete(self, key):
        z = self._find(key)
        if z is NIL:
            return False
        self._delete_node(z)
        return True

    def min_key(self):
        if self.root is NIL:
            return None
        node = self._minimum(self.root)
        return node.key

    def max_key(self):
        if self.root is NIL:
            return None
        node = self._maximum(self.root)
        return node.key

    def min_item(self):
        if self.root is NIL:
            return None
        node = self._minimum(self.root)
        return (node.key, node.value)

    def max_item(self):
        if self.root is NIL:
            return None
        node = self._maximum(self.root)
        return (node.key, node.value)

    def floor(self, key):
        """Largest key <= given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if key == node.key:
                return node.key
            elif key < node.key:
                node = node.left
            else:
                result = node.key
                node = node.right
        return result

    def ceiling(self, key):
        """Smallest key >= given key, or None."""
        result = None
        node = self.root
        while node is not NIL:
            if key == node.key:
                return node.key
            elif key > node.key:
                node = node.right
            else:
                result = node.key
                node = node.left
        return result

    def keys(self):
        return list(self._inorder_keys(self.root))

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return list(self._inorder_items(self.root))

    def range(self, lo, hi):
        """Return all (key, value) pairs with keys in [lo, hi]."""
        result = []
        self._range_collect(self.root, lo, hi, result)
        return result

    def clear(self):
        self.root = NIL
        self._size = 0

    # --- Reuse RB tree internals ---

    def _find(self, key):
        node = self.root
        while node is not NIL:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return NIL

    def _minimum(self, node):
        while node.left is not NIL:
            node = node.left
        return node

    def _maximum(self, node):
        while node.right is not NIL:
            node = node.right
        return node

    def _inorder_keys(self, node):
        if node is not NIL:
            yield from self._inorder_keys(node.left)
            yield node.key
            yield from self._inorder_keys(node.right)

    def _inorder_items(self, node):
        if node is not NIL:
            yield from self._inorder_items(node.left)
            yield (node.key, node.value)
            yield from self._inorder_items(node.right)

    def _range_collect(self, node, lo, hi, result):
        if node is NIL:
            return
        if lo < node.key:
            self._range_collect(node.left, lo, hi, result)
        if lo <= node.key <= hi:
            result.append((node.key, node.value))
        if node.key < hi:
            self._range_collect(node.right, lo, hi, result)

    def _update_size_up(self, node):
        while node is not NIL:
            node.size = node.left.size + node.right.size + 1
            node = node.parent

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    def _transplant(self, u, v):
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_node(self, z):
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
            self._update_size_up(z.parent)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
            self._update_size_up(z.parent)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self._update_size_up(x.parent if x.parent is not NIL else y)
            y.size = y.left.size + y.right.size + 1
        self._size -= 1
        if y_original_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK


class RedBlackMultiMap:
    """Ordered multimap -- allows duplicate keys, each with its own value."""

    def __init__(self, items=None):
        self.root = NIL
        self._size = 0
        if items:
            for k, v in items:
                self.insert(k, v)

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __contains__(self, key):
        return self._find_first(key) is not NIL

    def __iter__(self):
        return self._inorder_items(self.root)

    def insert(self, key, value):
        """Insert key-value pair (always inserts, even for duplicate keys)."""
        node = Node(key, value, color=RED)
        node.left = NIL
        node.right = NIL

        parent = NIL
        current = self.root

        while current is not NIL:
            parent = current
            if key <= current.key:  # <= allows duplicates on left
                current = current.left
            else:
                current = current.right

        node.parent = parent
        if parent is NIL:
            self.root = node
        elif key <= parent.key:
            parent.left = node
        else:
            parent.right = node

        self._size += 1
        self._update_size_up(node.parent)
        self._insert_fixup(node)

    def get_all(self, key):
        """Return all values associated with key."""
        results = []
        self._collect_key(self.root, key, results)
        return results

    def delete_one(self, key):
        """Delete one occurrence of key. Returns True if found."""
        node = self._find_first(key)
        if node is NIL:
            return False
        self._delete_node(node)
        return True

    def delete_all(self, key):
        """Delete all occurrences of key. Returns count deleted."""
        count = 0
        while True:
            node = self._find_first(key)
            if node is NIL:
                break
            self._delete_node(node)
            count += 1
        return count

    def count(self, key):
        """Count occurrences of key."""
        return len(self.get_all(key))

    def items(self):
        return list(self._inorder_items(self.root))

    def clear(self):
        self.root = NIL
        self._size = 0

    def _find_first(self, key):
        result = NIL
        node = self.root
        while node is not NIL:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                result = node
                node = node.left  # look for earlier occurrence
        return result

    def _collect_key(self, node, key, results):
        if node is NIL:
            return
        if key <= node.key:
            self._collect_key(node.left, key, results)
        if node.key == key:
            results.append(node.value)
        if key >= node.key:
            self._collect_key(node.right, key, results)

    def _inorder_items(self, node):
        if node is not NIL:
            yield from self._inorder_items(node.left)
            yield (node.key, node.value)
            yield from self._inorder_items(node.right)

    # Reuse core RB internals (same as RedBlackMap)
    def _update_size_up(self, node):
        while node is not NIL:
            node.size = node.left.size + node.right.size + 1
            node = node.parent

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    def _transplant(self, u, v):
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_node(self, z):
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
            self._update_size_up(z.parent)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
            self._update_size_up(z.parent)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self._update_size_up(x.parent if x.parent is not NIL else y)
            y.size = y.left.size + y.right.size + 1
        self._size -= 1
        if y_original_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK

    def _minimum(self, node):
        while node.left is not NIL:
            node = node.left
        return node


class OrderStatisticTree:
    """Red-black tree augmented with size for O(log n) rank and select."""

    def __init__(self, items=None):
        self._tree = RedBlackTree(items)

    def __len__(self):
        return len(self._tree)

    def __contains__(self, key):
        return key in self._tree

    def __iter__(self):
        return iter(self._tree)

    def insert(self, key):
        return self._tree.insert(key)

    def delete(self, key):
        return self._tree.delete(key)

    def rank(self, key):
        """Return 0-based rank (position) of key. Raises KeyError if not found."""
        node = self._tree._find(key)
        if node is NIL:
            raise KeyError(key)
        r = node.left.size
        y = node
        while y.parent is not NIL:
            if y is y.parent.right:
                r += y.parent.left.size + 1
            y = y.parent
        return r

    def select(self, rank):
        """Return the key at 0-based rank. Raises IndexError if out of range."""
        if rank < 0 or rank >= self._tree._size:
            raise IndexError(f"rank {rank} out of range")
        return self._select_node(self._tree.root, rank).key

    def count_less(self, key):
        """Count elements strictly less than key."""
        count = 0
        node = self._tree.root
        while node is not NIL:
            if key <= node.key:
                node = node.left
            else:
                count += node.left.size + 1
                node = node.right
        return count

    def count_range(self, lo, hi):
        """Count elements in [lo, hi]."""
        return self.count_less(hi + 1) - self.count_less(lo) if isinstance(lo, int) else self._count_range_generic(lo, hi)

    def _count_range_generic(self, lo, hi):
        """Count range for non-integer keys."""
        count = 0
        self._count_range_impl(self._tree.root, lo, hi, count_ref=[0])
        return count_ref[0]

    def _select_node(self, node, rank):
        left_size = node.left.size
        if rank < left_size:
            return self._select_node(node.left, rank)
        elif rank == left_size:
            return node
        else:
            return self._select_node(node.right, rank - left_size - 1)

    def min(self):
        return self._tree.min()

    def max(self):
        return self._tree.max()

    def to_sorted_list(self):
        return self._tree.to_sorted_list()


class IntervalMap:
    """Augmented red-black tree for interval queries.

    Each interval is (lo, hi) with lo <= hi. Supports:
    - insert/delete intervals
    - stab queries (point in which intervals?)
    - overlap queries (which intervals overlap a given interval?)
    """

    def __init__(self):
        self.root = NIL
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def insert(self, lo, hi, value=None):
        """Insert interval [lo, hi] with optional associated value."""
        node = Node(key=lo, value=(hi, value), color=RED)
        node.left = NIL
        node.right = NIL
        node.max_end = hi

        parent = NIL
        current = self.root

        while current is not NIL:
            parent = current
            if lo < current.key or (lo == current.key and hi < current.value[0]):
                current = current.left
            else:
                current = current.right

        node.parent = parent
        if parent is NIL:
            self.root = node
        elif lo < parent.key or (lo == parent.key and hi < parent.value[0]):
            parent.left = node
        else:
            parent.right = node

        self._size += 1
        self._update_max_up(node)
        self._insert_fixup(node)

    def delete(self, lo, hi):
        """Delete interval [lo, hi]. Returns True if found."""
        node = self._find_interval(lo, hi)
        if node is NIL:
            return False
        self._delete_node(node)
        return True

    def stab(self, point):
        """Return all intervals containing the point."""
        results = []
        self._stab_query(self.root, point, results)
        return results

    def overlap(self, lo, hi):
        """Return all intervals overlapping [lo, hi]."""
        results = []
        self._overlap_query(self.root, lo, hi, results)
        return results

    def all_intervals(self):
        """Return all intervals in sorted order."""
        result = []
        self._collect_all(self.root, result)
        return result

    def clear(self):
        self.root = NIL
        self._size = 0

    def _find_interval(self, lo, hi):
        node = self.root
        while node is not NIL:
            node_hi = node.value[0]
            if lo == node.key and hi == node_hi:
                return node
            elif lo < node.key or (lo == node.key and hi < node_hi):
                node = node.left
            else:
                node = node.right
        return NIL

    def _stab_query(self, node, point, results):
        if node is NIL:
            return
        if node.max_end < point:
            return
        self._stab_query(node.left, point, results)
        lo = node.key
        hi = node.value[0]
        if lo <= point <= hi:
            results.append((lo, hi, node.value[1]))
        if point >= lo:
            self._stab_query(node.right, point, results)

    def _overlap_query(self, node, lo, hi, results):
        if node is NIL:
            return
        if node.max_end < lo:
            return
        self._overlap_query(node.left, lo, hi, results)
        node_lo = node.key
        node_hi = node.value[0]
        if node_lo <= hi and node_hi >= lo:
            results.append((node_lo, node_hi, node.value[1]))
        if node_lo <= hi:
            self._overlap_query(node.right, lo, hi, results)

    def _collect_all(self, node, result):
        if node is NIL:
            return
        self._collect_all(node.left, result)
        result.append((node.key, node.value[0], node.value[1]))
        self._collect_all(node.right, result)

    def _update_max(self, node):
        if node is NIL:
            return
        node.max_end = node.value[0]  # own hi
        if node.left is not NIL and node.left.max_end is not None:
            node.max_end = max(node.max_end, node.left.max_end)
        if node.right is not NIL and node.right.max_end is not None:
            node.max_end = max(node.max_end, node.right.max_end)

    def _update_max_up(self, node):
        while node is not NIL:
            old = node.max_end
            self._update_max(node)
            if node.max_end == old:
                break
            node = node.parent

    def _update_size_up(self, node):
        while node is not NIL:
            node.size = node.left.size + node.right.size + 1
            node = node.parent

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1
        self._update_max(x)
        self._update_max(y)

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1
        self._update_max(x)
        self._update_max(y)

    def _insert_fixup(self, z):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK
        self._update_max_up(z)

    def _transplant(self, u, v):
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node):
        while node.left is not NIL:
            node = node.left
        return node

    def _delete_node(self, z):
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
            self._update_max_up(z.parent)
            self._update_size_up(z.parent)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
            self._update_max_up(z.parent)
            self._update_size_up(z.parent)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            update_from = x.parent if x.parent is not NIL else y
            self._update_max_up(update_from)
            self._update_size_up(update_from)
            y.size = y.left.size + y.right.size + 1
            self._update_max(y)
        self._size -= 1
        if y_original_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK
