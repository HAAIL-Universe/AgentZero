"""
C122: Binomial Heap

A binomial heap is a collection of binomial trees satisfying the heap property.
Each binomial tree B_k has exactly 2^k nodes and height k. The heap stores at
most one tree of each order, mirroring binary number representation.

Key operations and their worst-case complexities:
  - insert: O(log n)
  - find_min: O(log n), O(1) with cached min
  - merge/union: O(log n)
  - extract_min: O(log n)
  - decrease_key: O(log n)
  - delete: O(log n)

Variants:
1. BinomialHeap -- min-heap with merge, decrease-key, delete
2. MaxBinomialHeap -- max-heap variant
3. BinomialHeapMap -- key-value map with O(1) lookup by name
4. MergeableBinomialHeap -- named heaps with merge tracking
5. BinomialHeapPQ -- priority queue interface (push/pop/peek/update)
6. LazyBinomialHeap -- lazy insertion with deferred consolidation

Utilities:
  - heap_sort(items) -- sort via binomial heap
  - k_smallest(items, k) -- find k smallest elements
  - merge_sorted_streams(streams) -- merge multiple sorted iterables
"""


class BinomialNode:
    """Node in a binomial tree. Uses child-sibling representation."""
    __slots__ = ('key', 'value', 'degree', 'parent', 'child', 'sibling')

    def __init__(self, key, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.degree = 0
        self.parent = None
        self.child = None   # leftmost child
        self.sibling = None  # right sibling

    def __repr__(self):
        return f"BinomialNode(key={self.key}, value={self.value})"


def _link(child, parent):
    """Make child a subtree of parent. Both must have the same degree.
    Returns parent (new root of combined tree)."""
    child.parent = parent
    child.sibling = parent.child
    parent.child = child
    parent.degree += 1
    return parent


def _merge_root_lists(h1, h2):
    """Merge two root lists sorted by degree into one sorted list.
    Returns head of merged list."""
    if h1 is None:
        return h2
    if h2 is None:
        return h1

    head = None
    tail = None
    a, b = h1, h2

    while a is not None and b is not None:
        if a.degree <= b.degree:
            pick = a
            a = a.sibling
        else:
            pick = b
            b = b.sibling

        if head is None:
            head = pick
            tail = pick
        else:
            tail.sibling = pick
            tail = pick

    remaining = a if a is not None else b
    if tail is not None:
        tail.sibling = remaining
    else:
        head = remaining

    return head


def _union(h1, h2):
    """Union two binomial heaps (root lists). Returns new root list head.
    Core algorithm: merge root lists by degree, then combine same-degree trees."""
    head = _merge_root_lists(h1, h2)
    if head is None:
        return None

    prev = None
    curr = head
    nxt = curr.sibling

    while nxt is not None:
        # Case 1 & 2: degrees differ, or three trees of same degree (skip)
        if curr.degree != nxt.degree or (nxt.sibling is not None and nxt.sibling.degree == curr.degree):
            prev = curr
            curr = nxt
        # Case 3: curr.key <= nxt.key -- link nxt under curr
        elif curr.key <= nxt.key:
            curr.sibling = nxt.sibling
            _link(nxt, curr)
        # Case 4: curr.key > nxt.key -- link curr under nxt
        else:
            if prev is None:
                head = nxt
            else:
                prev.sibling = nxt
            _link(curr, nxt)
            curr = nxt
        nxt = curr.sibling

    return head


def _union_max(h1, h2):
    """Union for max-heaps: larger key becomes parent."""
    head = _merge_root_lists(h1, h2)
    if head is None:
        return None

    prev = None
    curr = head
    nxt = curr.sibling

    while nxt is not None:
        if curr.degree != nxt.degree or (nxt.sibling is not None and nxt.sibling.degree == curr.degree):
            prev = curr
            curr = nxt
        elif curr.key >= nxt.key:
            curr.sibling = nxt.sibling
            _link(nxt, curr)
        else:
            if prev is None:
                head = nxt
            else:
                prev.sibling = nxt
            _link(curr, nxt)
            curr = nxt
        nxt = curr.sibling

    return head


def _reverse_children(node):
    """Reverse the child list of a node, clearing parent pointers.
    Returns head of reversed list (suitable as a new root list)."""
    prev = None
    curr = node
    while curr is not None:
        nxt = curr.sibling
        curr.sibling = prev
        curr.parent = None
        prev = curr
        curr = nxt
    return prev


def _find_min_node(head):
    """Find the node with minimum key in a root list."""
    if head is None:
        return None
    min_node = head
    curr = head.sibling
    while curr is not None:
        if curr.key < min_node.key:
            min_node = curr
        curr = curr.sibling
    return min_node


def _find_max_node(head):
    """Find the node with maximum key in a root list."""
    if head is None:
        return None
    max_node = head
    curr = head.sibling
    while curr is not None:
        if curr.key > max_node.key:
            max_node = curr
        curr = curr.sibling
    return max_node


class BinomialHeap:
    """Min-Binomial Heap supporting merge, decrease-key, and arbitrary delete."""

    def __init__(self):
        self._head = None  # head of root list (sorted by degree)
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    def insert(self, key, value=None):
        """Insert a key-value pair. Returns the node for later decrease_key/delete."""
        node = BinomialNode(key, value)
        self._head = _union(self._head, node)
        self._size += 1
        return node

    def find_min(self):
        """Return (key, value) of the minimum element."""
        if self._head is None:
            raise IndexError("find_min on empty heap")
        node = _find_min_node(self._head)
        return (node.key, node.value)

    def peek(self):
        """Alias for find_min."""
        return self.find_min()

    def extract_min(self):
        """Remove and return (key, value) of the minimum element."""
        if self._head is None:
            raise IndexError("extract_min on empty heap")

        # Find min and its predecessor
        min_node = self._head
        min_prev = None
        prev = None
        curr = self._head
        while curr is not None:
            if curr.key < min_node.key:
                min_node = curr
                min_prev = prev
            prev = curr
            curr = curr.sibling

        # Remove min from root list
        if min_prev is None:
            self._head = min_node.sibling
        else:
            min_prev.sibling = min_node.sibling

        # Reverse children of min and union with remaining heap
        children = _reverse_children(min_node.child)
        self._head = _union(self._head, children)
        self._size -= 1

        return (min_node.key, min_node.value)

    def decrease_key(self, node, new_key):
        """Decrease the key of a node. new_key must be <= current key."""
        if new_key > node.key:
            raise ValueError(f"New key {new_key} is greater than current key {node.key}")
        node.key = new_key
        # Bubble up
        curr = node
        parent = curr.parent
        while parent is not None and curr.key < parent.key:
            # Swap key and value (not pointers)
            curr.key, parent.key = parent.key, curr.key
            curr.value, parent.value = parent.value, curr.value
            # If there are external references, swap node identity tracking
            curr = parent
            parent = curr.parent

    def delete(self, node):
        """Delete an arbitrary node from the heap."""
        self.decrease_key(node, float('-inf'))
        self.extract_min()

    def merge(self, other):
        """Merge another BinomialHeap into this one. The other heap becomes empty."""
        if not isinstance(other, BinomialHeap):
            raise TypeError("Can only merge with another BinomialHeap")
        self._head = _union(self._head, other._head)
        self._size += other._size
        other._head = None
        other._size = 0

    def to_sorted_list(self):
        """Extract all elements in sorted order."""
        result = []
        while self._size > 0:
            result.append(self.extract_min())
        return result

    def __iter__(self):
        """Iterate over all (key, value) pairs (not sorted, non-destructive)."""
        def _traverse(node):
            while node is not None:
                yield (node.key, node.value)
                yield from _traverse(node.child)
                node = node.sibling
        yield from _traverse(self._head)

    def __contains__(self, key):
        for k, v in self:
            if k == key:
                return True
        return False


class MaxBinomialHeap:
    """Max-Binomial Heap -- all operations use max ordering."""

    def __init__(self):
        self._head = None
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    def insert(self, key, value=None):
        node = BinomialNode(key, value)
        self._head = _union_max(self._head, node)
        self._size += 1
        return node

    def find_max(self):
        if self._head is None:
            raise IndexError("find_max on empty heap")
        node = _find_max_node(self._head)
        return (node.key, node.value)

    def peek(self):
        return self.find_max()

    def extract_max(self):
        if self._head is None:
            raise IndexError("extract_max on empty heap")

        max_node = self._head
        max_prev = None
        prev = None
        curr = self._head
        while curr is not None:
            if curr.key > max_node.key:
                max_node = curr
                max_prev = prev
            prev = curr
            curr = curr.sibling

        if max_prev is None:
            self._head = max_node.sibling
        else:
            max_prev.sibling = max_node.sibling

        children = _reverse_children(max_node.child)
        self._head = _union_max(self._head, children)
        self._size -= 1

        return (max_node.key, max_node.value)

    def increase_key(self, node, new_key):
        if new_key < node.key:
            raise ValueError(f"New key {new_key} is less than current key {node.key}")
        node.key = new_key
        curr = node
        parent = curr.parent
        while parent is not None and curr.key > parent.key:
            curr.key, parent.key = parent.key, curr.key
            curr.value, parent.value = parent.value, curr.value
            curr = parent
            parent = curr.parent

    def delete(self, node):
        self.increase_key(node, float('inf'))
        self.extract_max()

    def merge(self, other):
        if not isinstance(other, MaxBinomialHeap):
            raise TypeError("Can only merge with another MaxBinomialHeap")
        self._head = _union_max(self._head, other._head)
        self._size += other._size
        other._head = None
        other._size = 0

    def to_sorted_list(self):
        result = []
        while self._size > 0:
            result.append(self.extract_max())
        return result

    def __iter__(self):
        def _traverse(node):
            while node is not None:
                yield (node.key, node.value)
                yield from _traverse(node.child)
                node = node.sibling
        yield from _traverse(self._head)


class BinomialHeapMap:
    """Binomial Heap with O(1) lookup by name/key for decrease-key operations."""

    def __init__(self):
        self._heap = BinomialHeap()
        self._nodes = {}  # name -> node

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def insert(self, name, priority):
        """Insert with a name for later lookup. Name must be unique."""
        if name in self._nodes:
            raise KeyError(f"Name '{name}' already exists")
        node = self._heap.insert(priority, name)
        self._nodes[name] = node
        return node

    def find_min(self):
        """Return (priority, name) of minimum."""
        key, value = self._heap.find_min()
        return (key, value)

    def peek(self):
        return self.find_min()

    def extract_min(self):
        """Remove and return (priority, name) of minimum."""
        key, value = self._heap.extract_min()
        if value in self._nodes:
            del self._nodes[value]
        return (key, value)

    def decrease_key(self, name, new_priority):
        """Decrease priority of named element."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        node = self._nodes[name]
        old_key = node.key
        self._heap.decrease_key(node, new_priority)
        # After bubble-up, node references may have swapped values
        # Re-map all affected nodes
        self._rebuild_map()

    def _rebuild_map(self):
        """Rebuild the name->node map by traversing the heap."""
        self._nodes.clear()
        for key, value in self._heap:
            # Find the actual node (traverse)
            pass
        # More efficient: walk the tree
        def _traverse(node):
            while node is not None:
                self._nodes[node.value] = node
                _traverse(node.child)
                node = node.sibling
        _traverse(self._heap._head)

    def delete(self, name):
        """Delete a named element."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        node = self._nodes[name]
        self._heap.delete(node)
        # After delete, values may have been swapped during bubble-up
        self._rebuild_map()
        if name in self._nodes:
            del self._nodes[name]

    def __contains__(self, name):
        return name in self._nodes

    def get_priority(self, name):
        """Get the current priority of a named element."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        return self._nodes[name].key

    def __iter__(self):
        return iter(self._heap)


class MergeableBinomialHeap:
    """Binomial Heap with explicit merge tracking and size."""

    def __init__(self, name=None):
        self._heap = BinomialHeap()
        self.name = name
        self._merged_from = []

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def insert(self, key, value=None):
        return self._heap.insert(key, value)

    def find_min(self):
        return self._heap.find_min()

    def peek(self):
        return self.find_min()

    def extract_min(self):
        return self._heap.extract_min()

    def merge(self, other):
        """Merge another MergeableBinomialHeap into this one."""
        if not isinstance(other, MergeableBinomialHeap):
            raise TypeError("Can only merge with MergeableBinomialHeap")
        self._heap.merge(other._heap)
        if other.name:
            self._merged_from.append(other.name)
        self._merged_from.extend(other._merged_from)

    @property
    def merge_history(self):
        return list(self._merged_from)

    def to_sorted_list(self):
        return self._heap.to_sorted_list()

    def __iter__(self):
        return iter(self._heap)


class BinomialHeapPQ:
    """Priority Queue interface built on BinomialHeap."""

    def __init__(self):
        self._heap = BinomialHeap()
        self._counter = 0  # for FIFO tie-breaking

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def push(self, item, priority=0):
        """Push item with priority (lower = higher priority)."""
        self._counter += 1
        node = self._heap.insert((priority, self._counter), item)
        return node

    def pop(self):
        """Pop highest-priority (lowest key) item."""
        if self._heap.is_empty():
            raise IndexError("pop from empty priority queue")
        key, value = self._heap.extract_min()
        return value

    def peek(self):
        """Peek at highest-priority item without removing."""
        if self._heap.is_empty():
            raise IndexError("peek on empty priority queue")
        key, value = self._heap.find_min()
        return value

    def push_pop(self, item, priority=0):
        """Push an item and pop the minimum. More efficient than push then pop."""
        self.push(item, priority)
        return self.pop()

    def __iter__(self):
        """Iterate items in priority order (destructive)."""
        while self._heap:
            yield self.pop()


class LazyBinomialHeap:
    """Lazy Binomial Heap -- defers consolidation until extract_min.
    Insert is O(1) amortized by just adding trees to the root list."""

    def __init__(self):
        self._roots = []  # list of root nodes (not necessarily distinct degrees)
        self._size = 0
        self._min_node = None

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    def insert(self, key, value=None):
        """O(1) insert -- just add a single-node tree to roots."""
        node = BinomialNode(key, value)
        self._roots.append(node)
        if self._min_node is None or key < self._min_node.key:
            self._min_node = node
        self._size += 1
        return node

    def find_min(self):
        """O(1) with cached minimum."""
        if self._min_node is None:
            raise IndexError("find_min on empty heap")
        return (self._min_node.key, self._min_node.value)

    def peek(self):
        return self.find_min()

    def _consolidate(self):
        """Consolidate roots by degree, like Fibonacci heap consolidation."""
        if not self._roots:
            self._min_node = None
            return

        max_degree = 0
        for r in self._roots:
            if r.degree > max_degree:
                max_degree = r.degree
        # Extra space for potential merges
        table_size = max_degree + len(self._roots) + 1
        degree_table = [None] * table_size

        for root in self._roots:
            root.parent = None
            root.sibling = None
            curr = root
            d = curr.degree
            while d < len(degree_table) and degree_table[d] is not None:
                other = degree_table[d]
                degree_table[d] = None
                if curr.key > other.key:
                    curr, other = other, curr
                _link(other, curr)
                d = curr.degree
            if d >= len(degree_table):
                degree_table.extend([None] * (d - len(degree_table) + 1))
            degree_table[d] = curr

        # Rebuild root list
        self._roots = [t for t in degree_table if t is not None]
        self._min_node = None
        for r in self._roots:
            r.sibling = None
            r.parent = None
            if self._min_node is None or r.key < self._min_node.key:
                self._min_node = r

    def extract_min(self):
        """Remove minimum. Triggers consolidation."""
        if not self._roots:
            raise IndexError("extract_min on empty heap")

        # Find and remove min
        min_node = self._min_node
        self._roots.remove(min_node)

        # Add children to roots
        child = min_node.child
        while child is not None:
            nxt = child.sibling
            child.parent = None
            child.sibling = None
            self._roots.append(child)
            child = nxt

        self._size -= 1
        self._consolidate()

        return (min_node.key, min_node.value)

    def merge(self, other):
        """Merge another LazyBinomialHeap. O(1)."""
        if not isinstance(other, LazyBinomialHeap):
            raise TypeError("Can only merge with LazyBinomialHeap")
        self._roots.extend(other._roots)
        self._size += other._size
        if other._min_node is not None:
            if self._min_node is None or other._min_node.key < self._min_node.key:
                self._min_node = other._min_node
        other._roots = []
        other._size = 0
        other._min_node = None

    def decrease_key(self, node, new_key):
        """Decrease key with bubble-up."""
        if new_key > node.key:
            raise ValueError(f"New key {new_key} is greater than current key {node.key}")
        node.key = new_key
        curr = node
        parent = curr.parent
        while parent is not None and curr.key < parent.key:
            curr.key, parent.key = parent.key, curr.key
            curr.value, parent.value = parent.value, curr.value
            curr = parent
            parent = curr.parent
        # Update cached min
        if self._min_node is None or new_key < self._min_node.key:
            self._min_node = curr

    def delete(self, node):
        """Delete arbitrary node."""
        self.decrease_key(node, float('-inf'))
        self.extract_min()

    def to_sorted_list(self):
        result = []
        while self._size > 0:
            result.append(self.extract_min())
        return result

    def __iter__(self):
        def _traverse_roots():
            for root in self._roots:
                yield from _traverse_node(root)
        def _traverse_node(node):
            if node is None:
                return
            yield (node.key, node.value)
            yield from _traverse_node(node.child)
            yield from _traverse_node(node.sibling)
        yield from _traverse_roots()


# === Utilities ===

def heap_sort(items):
    """Sort items using a binomial heap. Returns sorted list of (key, value) pairs.
    Items can be (key, value) tuples or plain keys."""
    heap = BinomialHeap()
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            heap.insert(item[0], item[1])
        else:
            heap.insert(item)
    return heap.to_sorted_list()


def k_smallest(items, k):
    """Find the k smallest elements from items.
    Returns sorted list of (key, value) pairs."""
    if k <= 0:
        return []
    heap = BinomialHeap()
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            heap.insert(item[0], item[1])
        else:
            heap.insert(item)
    result = []
    for _ in range(min(k, len(heap))):
        result.append(heap.extract_min())
    return result


def merge_sorted_streams(*streams):
    """Merge multiple sorted iterables into a single sorted output.
    Yields elements in sorted order."""
    heap = BinomialHeap()
    iters = []
    for i, stream in enumerate(streams):
        it = iter(stream)
        try:
            val = next(it)
            iters.append(it)
            heap.insert((val, i), len(iters) - 1)
        except StopIteration:
            iters.append(iter([]))

    while heap:
        (val, stream_idx), idx = heap.extract_min()
        yield val
        try:
            nxt = next(iters[idx])
            heap.insert((nxt, stream_idx), idx)
        except StopIteration:
            pass
