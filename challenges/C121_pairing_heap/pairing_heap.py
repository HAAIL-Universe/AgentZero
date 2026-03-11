"""
C121: Pairing Heap

A pairing heap is a type of self-adjusting heap with simple implementation
and excellent amortized performance. Simpler than Fibonacci heaps while
achieving similar bounds in practice.

Variants:
1. PairingHeap -- min-heap with decrease-key, merge, arbitrary delete
2. MaxPairingHeap -- max-heap variant
3. PairingHeapMap -- key-value map with decrease-key by name
4. MergeablePairingHeap -- explicit merge support with size tracking
5. PairingHeapPQ -- priority queue interface with task scheduling
6. LazyPairingHeap -- lazy variant with delayed merging
"""


class PairingNode:
    """Node for pairing heap. Uses left-child/right-sibling representation."""
    __slots__ = ('key', 'value', 'child', 'sibling', 'parent')

    def __init__(self, key, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.child = None
        self.sibling = None
        self.parent = None

    def __repr__(self):
        return f"PairingNode(key={self.key}, value={self.value})"


def _link(a, b):
    """Link two nodes, making the one with larger key a child of the other.
    Returns the new root. Both a and b must be non-None."""
    if b.key < a.key:
        a, b = b, a
    # b becomes first child of a (prepend to child list)
    b.sibling = a.child
    b.parent = a
    a.child = b
    a.sibling = None
    a.parent = None
    return a


def _link_max(a, b):
    """Link for max-heap: larger key becomes root."""
    if b.key > a.key:
        a, b = b, a
    b.sibling = a.child
    b.parent = a
    a.child = b
    a.sibling = None
    a.parent = None
    return a


def _two_pass_merge(node):
    """Two-pass pairing: left-to-right pairing pass, then right-to-left merge.
    This is the standard deletion algorithm for pairing heaps."""
    if node is None:
        return None
    if node.sibling is None:
        node.parent = None
        return node

    # Left-to-right pairing pass: pair adjacent siblings
    pairs = []
    current = node
    while current is not None:
        a = current
        b = current.sibling
        if b is not None:
            next_node = b.sibling
            a.sibling = None
            a.parent = None
            b.sibling = None
            b.parent = None
            pairs.append(_link(a, b))
            current = next_node
        else:
            a.sibling = None
            a.parent = None
            pairs.append(a)
            current = None

    # Right-to-left merge
    result = pairs[-1]
    for i in range(len(pairs) - 2, -1, -1):
        result = _link(pairs[i], result)

    return result


def _two_pass_merge_max(node):
    """Two-pass pairing for max-heap."""
    if node is None:
        return None
    if node.sibling is None:
        node.parent = None
        return node

    pairs = []
    current = node
    while current is not None:
        a = current
        b = current.sibling
        if b is not None:
            next_node = b.sibling
            a.sibling = None
            a.parent = None
            b.sibling = None
            b.parent = None
            pairs.append(_link_max(a, b))
            current = next_node
        else:
            a.sibling = None
            a.parent = None
            pairs.append(a)
            current = None

    result = pairs[-1]
    for i in range(len(pairs) - 2, -1, -1):
        result = _link_max(pairs[i], result)

    return result


# ---------------------------------------------------------------------------
# 1. PairingHeap (min-heap)
# ---------------------------------------------------------------------------

class PairingHeap:
    """Min pairing heap with O(1) insert/merge/find-min, O(log n) amortized delete-min."""

    def __init__(self):
        self._root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    @property
    def size(self):
        return self._size

    def find_min(self):
        """Return minimum key. Raises IndexError if empty."""
        if self._root is None:
            raise IndexError("find_min on empty heap")
        return self._root.key

    def peek(self):
        """Return (key, value) of minimum. Raises IndexError if empty."""
        if self._root is None:
            raise IndexError("peek on empty heap")
        return (self._root.key, self._root.value)

    def insert(self, key, value=None):
        """Insert key with optional value. Returns the node for decrease_key."""
        node = PairingNode(key, value)
        if self._root is None:
            self._root = node
        else:
            self._root = _link(self._root, node)
        self._size += 1
        return node

    def delete_min(self):
        """Remove and return (key, value) of minimum. Raises IndexError if empty."""
        if self._root is None:
            raise IndexError("delete_min on empty heap")
        key, value = self._root.key, self._root.value
        self._root = _two_pass_merge(self._root.child)
        self._size -= 1
        return (key, value)

    def extract_min(self):
        """Alias for delete_min."""
        return self.delete_min()

    def decrease_key(self, node, new_key):
        """Decrease key of given node. Raises ValueError if new_key > current."""
        if new_key > node.key:
            raise ValueError(f"New key {new_key} is greater than current key {node.key}")
        node.key = new_key
        if node is self._root:
            return
        # Cut node from parent
        self._cut(node)
        self._root = _link(self._root, node)

    def _cut(self, node):
        """Cut node from its parent."""
        parent = node.parent
        if parent is None:
            return
        if parent.child is node:
            parent.child = node.sibling
            if node.sibling is not None:
                node.sibling.parent = parent
        else:
            # Find node among siblings
            prev = parent.child
            while prev is not None and prev.sibling is not node:
                prev = prev.sibling
            if prev is not None:
                prev.sibling = node.sibling
                if node.sibling is not None:
                    node.sibling.parent = parent
        node.parent = None
        node.sibling = None

    def delete(self, node):
        """Delete an arbitrary node from the heap."""
        if node is self._root:
            self.delete_min()
            return
        self._cut(node)
        merged_children = _two_pass_merge(node.child)
        if merged_children is not None:
            self._root = _link(self._root, merged_children)
        self._size -= 1

    def merge(self, other):
        """Merge another PairingHeap into this one. Other becomes empty."""
        if not isinstance(other, PairingHeap):
            raise TypeError("Can only merge with PairingHeap")
        if other._root is None:
            return
        if self._root is None:
            self._root = other._root
        else:
            self._root = _link(self._root, other._root)
        self._size += other._size
        other._root = None
        other._size = 0

    def __iter__(self):
        """Iterate in sorted order (destructive -- makes a copy)."""
        copy = PairingHeap()
        if self._root is not None:
            copy._root = self._deep_copy(self._root)
            copy._size = self._size
        while copy:
            key, value = copy.delete_min()
            yield (key, value)

    def _deep_copy(self, node):
        if node is None:
            return None
        new_node = PairingNode(node.key, node.value)
        new_node.child = self._deep_copy(node.child)
        new_node.sibling = self._deep_copy(node.sibling)
        if new_node.child is not None:
            new_node.child.parent = new_node
        if new_node.sibling is not None:
            new_node.sibling.parent = new_node
        return new_node

    def to_sorted_list(self):
        """Return sorted list of (key, value) pairs."""
        return list(self)

    def clear(self):
        """Remove all elements."""
        self._root = None
        self._size = 0


# ---------------------------------------------------------------------------
# 2. MaxPairingHeap
# ---------------------------------------------------------------------------

class MaxPairingHeap:
    """Max pairing heap -- find_max, delete_max, increase_key."""

    def __init__(self):
        self._root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    @property
    def size(self):
        return self._size

    def find_max(self):
        if self._root is None:
            raise IndexError("find_max on empty heap")
        return self._root.key

    def peek(self):
        if self._root is None:
            raise IndexError("peek on empty heap")
        return (self._root.key, self._root.value)

    def insert(self, key, value=None):
        node = PairingNode(key, value)
        if self._root is None:
            self._root = node
        else:
            self._root = _link_max(self._root, node)
        self._size += 1
        return node

    def delete_max(self):
        if self._root is None:
            raise IndexError("delete_max on empty heap")
        key, value = self._root.key, self._root.value
        self._root = _two_pass_merge_max(self._root.child)
        self._size -= 1
        return (key, value)

    def extract_max(self):
        return self.delete_max()

    def increase_key(self, node, new_key):
        if new_key < node.key:
            raise ValueError(f"New key {new_key} is less than current key {node.key}")
        node.key = new_key
        if node is self._root:
            return
        self._cut(node)
        self._root = _link_max(self._root, node)

    def _cut(self, node):
        parent = node.parent
        if parent is None:
            return
        if parent.child is node:
            parent.child = node.sibling
            if node.sibling is not None:
                node.sibling.parent = parent
        else:
            prev = parent.child
            while prev is not None and prev.sibling is not node:
                prev = prev.sibling
            if prev is not None:
                prev.sibling = node.sibling
                if node.sibling is not None:
                    node.sibling.parent = parent
        node.parent = None
        node.sibling = None

    def delete(self, node):
        if node is self._root:
            self.delete_max()
            return
        self._cut(node)
        merged_children = _two_pass_merge_max(node.child)
        if merged_children is not None:
            self._root = _link_max(self._root, merged_children)
        self._size -= 1

    def merge(self, other):
        if not isinstance(other, MaxPairingHeap):
            raise TypeError("Can only merge with MaxPairingHeap")
        if other._root is None:
            return
        if self._root is None:
            self._root = other._root
        else:
            self._root = _link_max(self._root, other._root)
        self._size += other._size
        other._root = None
        other._size = 0

    def clear(self):
        self._root = None
        self._size = 0


# ---------------------------------------------------------------------------
# 3. PairingHeapMap -- key-value with lookup by name
# ---------------------------------------------------------------------------

class PairingHeapMap:
    """Priority queue with named entries and decrease-key by name.
    Maps names to nodes for O(1) lookup."""

    def __init__(self):
        self._heap = PairingHeap()
        self._nodes = {}  # name -> node

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def __contains__(self, name):
        return name in self._nodes

    def is_empty(self):
        return self._heap.is_empty()

    def insert(self, name, priority):
        """Insert named entry with given priority. Raises KeyError if name exists."""
        if name in self._nodes:
            raise KeyError(f"Name '{name}' already exists")
        node = self._heap.insert(priority, name)
        self._nodes[name] = node

    def peek(self):
        """Return (priority, name) of minimum."""
        key, value = self._heap.peek()
        return (key, value)

    def extract_min(self):
        """Remove and return (priority, name) of minimum."""
        key, name = self._heap.delete_min()
        del self._nodes[name]
        return (key, name)

    def decrease_key(self, name, new_priority):
        """Decrease priority of named entry."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        node = self._nodes[name]
        self._heap.decrease_key(node, new_priority)

    def get_priority(self, name):
        """Get current priority of named entry."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        return self._nodes[name].key

    def delete(self, name):
        """Delete named entry."""
        if name not in self._nodes:
            raise KeyError(f"Name '{name}' not found")
        node = self._nodes[name]
        self._heap.delete(node)
        del self._nodes[name]

    def update(self, name, priority):
        """Insert or update: if name exists, decrease key if lower; else insert."""
        if name in self._nodes:
            node = self._nodes[name]
            if priority < node.key:
                self._heap.decrease_key(node, priority)
        else:
            self.insert(name, priority)

    def clear(self):
        self._heap.clear()
        self._nodes.clear()


# ---------------------------------------------------------------------------
# 4. MergeablePairingHeap -- merge support with size tracking
# ---------------------------------------------------------------------------

class MergeablePairingHeap:
    """Pairing heap optimized for frequent merges with size tracking and split."""

    def __init__(self):
        self._heap = PairingHeap()

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
        return self._heap.peek()

    def extract_min(self):
        return self._heap.delete_min()

    def delete_min(self):
        return self._heap.delete_min()

    def decrease_key(self, node, new_key):
        self._heap.decrease_key(node, new_key)

    def merge(self, other):
        """Merge another MergeablePairingHeap into this one."""
        if not isinstance(other, MergeablePairingHeap):
            raise TypeError("Can only merge with MergeablePairingHeap")
        self._heap.merge(other._heap)

    def merge_many(self, *heaps):
        """Merge multiple heaps into this one."""
        for h in heaps:
            self.merge(h)

    def split_min(self):
        """Extract min and return it as a new single-element heap."""
        if self.is_empty():
            raise IndexError("split_min on empty heap")
        key, value = self._heap.delete_min()
        new_heap = MergeablePairingHeap()
        new_heap.insert(key, value)
        return new_heap

    def to_sorted_list(self):
        return self._heap.to_sorted_list()

    def clear(self):
        self._heap.clear()


# ---------------------------------------------------------------------------
# 5. PairingHeapPQ -- priority queue with task scheduling
# ---------------------------------------------------------------------------

class PairingHeapPQ:
    """Priority queue for task scheduling with support for priorities,
    cancellation, and batch operations."""

    REMOVED = object()  # sentinel for cancelled tasks

    def __init__(self):
        self._heap = PairingHeap()
        self._entries = {}  # task_id -> node
        self._counter = 0

    def __len__(self):
        return len(self._entries)

    def __bool__(self):
        return bool(self._entries)

    def is_empty(self):
        return not bool(self._entries)

    def add_task(self, task_id, priority=0, data=None):
        """Add a task with given priority. Lower priority = higher urgency."""
        if task_id in self._entries:
            raise KeyError(f"Task '{task_id}' already exists")
        entry = {'task_id': task_id, 'data': data, 'seq': self._counter}
        self._counter += 1
        # Use (priority, seq) as composite key for stable ordering
        node = self._heap.insert((priority, entry['seq']), entry)
        self._entries[task_id] = node

    def get_next(self):
        """Get highest-priority task without removing it."""
        if not self._entries:
            raise IndexError("get_next on empty queue")
        key, entry = self._heap.peek()
        return (key[0], entry['task_id'], entry['data'])

    def pop_task(self):
        """Remove and return highest-priority task."""
        while self._heap:
            key, entry = self._heap.delete_min()
            task_id = entry['task_id']
            if task_id in self._entries:
                del self._entries[task_id]
                return (key[0], task_id, entry['data'])
        raise IndexError("pop_task on empty queue")

    def cancel_task(self, task_id):
        """Cancel a task by ID."""
        if task_id not in self._entries:
            raise KeyError(f"Task '{task_id}' not found")
        node = self._entries[task_id]
        self._heap.delete(node)
        del self._entries[task_id]

    def update_priority(self, task_id, new_priority):
        """Update task priority. Works for both increase and decrease."""
        if task_id not in self._entries:
            raise KeyError(f"Task '{task_id}' not found")
        node = self._entries[task_id]
        entry = node.value
        old_priority = node.key[0]
        if new_priority == old_priority:
            return
        # Remove and re-insert to handle both increase and decrease
        self._heap.delete(node)
        new_node = self._heap.insert((new_priority, entry['seq']), entry)
        self._entries[task_id] = new_node

    def has_task(self, task_id):
        return task_id in self._entries

    def get_priority(self, task_id):
        if task_id not in self._entries:
            raise KeyError(f"Task '{task_id}' not found")
        return self._entries[task_id].key[0]

    def batch_add(self, tasks):
        """Add multiple tasks: list of (task_id, priority) or (task_id, priority, data)."""
        for t in tasks:
            if len(t) == 2:
                self.add_task(t[0], t[1])
            else:
                self.add_task(t[0], t[1], t[2])

    def drain(self):
        """Remove and yield all tasks in priority order."""
        while self._entries:
            yield self.pop_task()

    def clear(self):
        self._heap.clear()
        self._entries.clear()
        self._counter = 0


# ---------------------------------------------------------------------------
# 6. LazyPairingHeap -- lazy variant with delayed merging
# ---------------------------------------------------------------------------

class LazyPairingHeap:
    """Lazy pairing heap that defers child merging.
    Insert is truly O(1) worst-case. Delete-min consolidates lazily."""

    def __init__(self):
        self._root = None
        self._size = 0
        self._aux = []  # auxiliary buffer for lazy inserts

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def is_empty(self):
        return self._size == 0

    def find_min(self):
        self._consolidate()
        if self._root is None:
            raise IndexError("find_min on empty heap")
        return self._root.key

    def peek(self):
        self._consolidate()
        if self._root is None:
            raise IndexError("peek on empty heap")
        return (self._root.key, self._root.value)

    def insert(self, key, value=None):
        """O(1) insert -- just adds to auxiliary buffer."""
        node = PairingNode(key, value)
        self._aux.append(node)
        self._size += 1
        return node

    def _consolidate(self):
        """Merge all auxiliary nodes into root."""
        if not self._aux:
            return
        for node in self._aux:
            node.parent = None
            node.sibling = None
            if self._root is None:
                self._root = node
            else:
                self._root = _link(self._root, node)
        self._aux.clear()

    def delete_min(self):
        """Remove and return minimum. Consolidates first."""
        self._consolidate()
        if self._root is None:
            raise IndexError("delete_min on empty heap")
        key, value = self._root.key, self._root.value
        self._root = _two_pass_merge(self._root.child)
        self._size -= 1
        return (key, value)

    def extract_min(self):
        return self.delete_min()

    def merge(self, other):
        """Merge another LazyPairingHeap into this one."""
        if not isinstance(other, LazyPairingHeap):
            raise TypeError("Can only merge with LazyPairingHeap")
        # Move other's aux into ours
        self._aux.extend(other._aux)
        other._aux.clear()
        if other._root is not None:
            self._aux.append(other._root)
            other._root = None
        self._size += other._size
        other._size = 0

    def clear(self):
        self._root = None
        self._size = 0
        self._aux.clear()


# ---------------------------------------------------------------------------
# Utility functions (Dijkstra, Prim) using PairingHeap
# ---------------------------------------------------------------------------

def dijkstra(graph, source):
    """Dijkstra's shortest path using PairingHeapMap.
    graph: dict of {node: [(neighbor, weight), ...]}
    Returns: (distances, predecessors)
    """
    dist = {source: 0}
    pred = {source: None}
    pq = PairingHeapMap()
    pq.insert(source, 0)

    visited = set()

    while pq:
        d, u = pq.extract_min()
        if u in visited:
            continue
        visited.add(u)

        for v, w in graph.get(u, []):
            new_dist = d + w
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = u
                pq.update(v, new_dist)

    return dist, pred


def prim_mst(graph):
    """Prim's minimum spanning tree using PairingHeapMap.
    graph: dict of {node: [(neighbor, weight), ...]}
    Returns: list of (u, v, weight) edges in MST
    """
    if not graph:
        return []

    start = next(iter(graph))
    in_mst = {start}
    mst_edges = []
    pq = PairingHeapMap()
    edge_info = {}  # node -> (from_node, weight)

    for v, w in graph.get(start, []):
        if v not in in_mst:
            pq.insert(v, w)
            edge_info[v] = (start, w)

    while pq:
        w, v = pq.extract_min()
        if v in in_mst:
            continue
        in_mst.add(v)
        from_node, weight = edge_info[v]
        mst_edges.append((from_node, v, weight))

        for u, uw in graph.get(v, []):
            if u not in in_mst:
                if u not in edge_info or uw < edge_info[u][1]:
                    edge_info[u] = (v, uw)
                    pq.update(u, uw)

    return mst_edges


def k_smallest(items, k):
    """Find k smallest items using PairingHeap. O(n + k log n)."""
    if k <= 0:
        return []
    heap = PairingHeap()
    for item in items:
        if isinstance(item, tuple):
            heap.insert(item[0], item[1])
        else:
            heap.insert(item)
    result = []
    while heap and len(result) < k:
        result.append(heap.delete_min())
    return result


def heap_sort(items):
    """Sort items using PairingHeap. O(n log n)."""
    heap = PairingHeap()
    for item in items:
        if isinstance(item, tuple):
            heap.insert(item[0], item[1])
        else:
            heap.insert(item)
    return [heap.delete_min() for _ in range(len(heap))]
