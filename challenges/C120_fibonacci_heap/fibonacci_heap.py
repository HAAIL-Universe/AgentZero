"""
C120: Fibonacci Heap

A Fibonacci heap is a collection of heap-ordered trees with lazy consolidation.
Key operations and their amortized complexities:
  - insert: O(1)
  - find_min: O(1)
  - merge: O(1)
  - decrease_key: O(1) amortized (cascading cuts)
  - delete_min: O(log n) amortized
  - delete: O(log n) amortized

Variants:
  - FibonacciHeap: min-heap with decrease_key
  - MaxFibonacciHeap: max-heap variant
  - FibonacciHeapMap: key-value heap with O(1) lookup by key
  - MergeableFibonacciHeap: named heaps with O(1) merge
  - FibonacciHeapPQ: priority queue interface (push/pop/peek/update)
"""

import math


class FibNode:
    """Node in a Fibonacci heap."""
    __slots__ = ('key', 'value', 'degree', 'mark', 'parent', 'child', 'left', 'right')

    def __init__(self, key, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self


def _insert_into_list(node, list_node):
    """Insert node into circular doubly-linked list containing list_node."""
    node.right = list_node.right
    node.left = list_node
    list_node.right.left = node
    list_node.right = node


def _remove_from_list(node):
    """Remove node from its circular doubly-linked list."""
    node.left.right = node.right
    node.right.left = node.left
    node.left = node
    node.right = node


def _merge_lists(a, b):
    """Merge two circular doubly-linked lists. Returns the new head."""
    if a is None:
        return b
    if b is None:
        return a
    # Splice b's list into a's list
    a_right = a.right
    b_left = b.left
    a.right = b
    b.left = a
    a_right.left = b_left
    b_left.right = a_right
    return a


def _iterate_list(head):
    """Iterate all nodes in a circular doubly-linked list."""
    if head is None:
        return
    node = head
    nodes = []
    while True:
        nodes.append(node)
        node = node.right
        if node is head:
            break
    return nodes


class FibonacciHeap:
    """Min-Fibonacci Heap with O(1) amortized insert/decrease_key."""

    def __init__(self):
        self.min_node = None
        self.n = 0

    def __len__(self):
        return self.n

    def __bool__(self):
        return self.n > 0

    def is_empty(self):
        return self.n == 0

    def insert(self, key, value=None):
        """Insert a key (and optional value) into the heap. Returns the node."""
        node = FibNode(key, value)
        if self.min_node is None:
            self.min_node = node
        else:
            _insert_into_list(node, self.min_node)
            if node.key < self.min_node.key:
                self.min_node = node
        self.n += 1
        return node

    def find_min(self):
        """Return the minimum node, or None if empty."""
        return self.min_node

    def peek(self):
        """Return (key, value) of the minimum element."""
        if self.min_node is None:
            raise IndexError("peek from empty heap")
        return (self.min_node.key, self.min_node.value)

    def merge(self, other):
        """Merge another FibonacciHeap into this one. The other heap is emptied."""
        if not isinstance(other, FibonacciHeap):
            raise TypeError("can only merge with another FibonacciHeap")
        if other.min_node is None:
            return
        self.min_node = _merge_lists(self.min_node, other.min_node)
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node
        self.n += other.n
        other.min_node = None
        other.n = 0

    def extract_min(self):
        """Remove and return the minimum node."""
        z = self.min_node
        if z is None:
            raise IndexError("extract_min from empty heap")

        # Add all children of z to root list
        if z.child is not None:
            children = _iterate_list(z.child)
            for child in children:
                child.parent = None
            self.min_node = _merge_lists(self.min_node, z.child)
            z.child = None

        # Remove z from root list
        if z.right is z:
            self.min_node = None
        else:
            self.min_node = z.right
            _remove_from_list(z)
            self._consolidate()

        self.n -= 1
        z.left = z
        z.right = z
        return z

    def decrease_key(self, node, new_key):
        """Decrease the key of a node."""
        if new_key > node.key:
            raise ValueError("new key is greater than current key")
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if node.key < self.min_node.key:
            self.min_node = node

    def delete(self, node):
        """Delete an arbitrary node from the heap."""
        self.decrease_key(node, float('-inf'))
        self.extract_min()

    def _cut(self, node, parent):
        """Cut node from parent and add to root list."""
        # Remove node from parent's child list
        if node.right is node:
            parent.child = None
        else:
            if parent.child is node:
                parent.child = node.right
            _remove_from_list(node)
        parent.degree -= 1
        # Add to root list
        node.parent = None
        node.mark = False
        _insert_into_list(node, self.min_node)

    def _cascading_cut(self, node):
        """Cascading cut up the tree."""
        parent = node.parent
        if parent is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

    def _consolidate(self):
        """Consolidate trees so no two roots have the same degree."""
        max_degree = int(math.log2(self.n)) + 2 if self.n > 0 else 1
        degree_table = [None] * (max_degree + 1)

        roots = _iterate_list(self.min_node)
        for root in roots:
            root.parent = None
            d = root.degree
            while d < len(degree_table) and degree_table[d] is not None:
                other = degree_table[d]
                if root.key > other.key:
                    root, other = other, root
                self._link(other, root)
                degree_table[d] = None
                d += 1
            if d >= len(degree_table):
                degree_table.extend([None] * (d - len(degree_table) + 1))
            degree_table[d] = root

        # Reconstruct root list
        self.min_node = None
        for node in degree_table:
            if node is not None:
                node.left = node
                node.right = node
                if self.min_node is None:
                    self.min_node = node
                else:
                    _insert_into_list(node, self.min_node)
                    if node.key < self.min_node.key:
                        self.min_node = node

    def _link(self, child, parent):
        """Make child a child of parent."""
        _remove_from_list(child)
        child.parent = parent
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            _insert_into_list(child, parent.child)
        parent.degree += 1
        child.mark = False

    def to_sorted_list(self):
        """Extract all elements in sorted order. Destructive."""
        result = []
        while self.n > 0:
            node = self.extract_min()
            result.append((node.key, node.value))
        return result

    def __iter__(self):
        """Iterate over all (key, value) pairs (not in sorted order)."""
        if self.min_node is None:
            return
        stack = [self.min_node]
        visited = set()
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            # Iterate siblings
            current = node
            while True:
                if id(current) not in visited:
                    visited.add(id(current))
                    yield (current.key, current.value)
                    if current.child is not None:
                        stack.append(current.child)
                current = current.right
                if current is node:
                    break


class MaxFibonacciHeap:
    """Max-Fibonacci Heap -- stores negated keys internally."""

    def __init__(self):
        self._heap = FibonacciHeap()
        self._node_map = {}  # id(wrapper) -> original_key

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def insert(self, key, value=None):
        """Insert with max-heap semantics. Returns a handle."""
        node = self._heap.insert(-key, value)
        self._node_map[id(node)] = key
        return node

    def find_max(self):
        """Return the maximum node."""
        node = self._heap.find_min()
        return node

    def peek(self):
        """Return (key, value) of the maximum element."""
        if self._heap.is_empty():
            raise IndexError("peek from empty heap")
        node = self._heap.find_min()
        return (self._node_map[id(node)], node.value)

    def extract_max(self):
        """Remove and return (key, value) of the maximum."""
        node = self._heap.extract_min()
        original_key = self._node_map.pop(id(node))
        return (original_key, node.value)

    def increase_key(self, node, new_key):
        """Increase the key of a node (decrease in negated space)."""
        old_key = self._node_map[id(node)]
        if new_key < old_key:
            raise ValueError("new key is less than current key")
        self._node_map[id(node)] = new_key
        self._heap.decrease_key(node, -new_key)

    def delete(self, node):
        """Delete a node."""
        self._node_map.pop(id(node), None)
        self._heap.delete(node)

    def merge(self, other):
        """Merge another MaxFibonacciHeap into this one."""
        if not isinstance(other, MaxFibonacciHeap):
            raise TypeError("can only merge with another MaxFibonacciHeap")
        self._heap.merge(other._heap)
        self._node_map.update(other._node_map)
        other._node_map.clear()


class FibonacciHeapMap:
    """Fibonacci heap with O(1) lookup by key name (key-value priority queue)."""

    def __init__(self):
        self._heap = FibonacciHeap()
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
        """Insert or update a named entry with given priority."""
        if name in self._nodes:
            self.update(name, priority)
            return
        node = self._heap.insert(priority, name)
        self._nodes[name] = node

    def peek(self):
        """Return (name, priority) of the minimum."""
        if self._heap.is_empty():
            raise IndexError("peek from empty heap")
        node = self._heap.find_min()
        return (node.value, node.key)

    def pop(self):
        """Remove and return (name, priority) of the minimum."""
        node = self._heap.extract_min()
        name = node.value
        del self._nodes[name]
        return (name, node.key)

    def get_priority(self, name):
        """Get the priority of a named entry."""
        if name not in self._nodes:
            raise KeyError(name)
        return self._nodes[name].key

    def update(self, name, new_priority):
        """Update the priority of a named entry."""
        if name not in self._nodes:
            raise KeyError(name)
        node = self._nodes[name]
        if new_priority < node.key:
            self._heap.decrease_key(node, new_priority)
        elif new_priority > node.key:
            # For increase: delete and re-insert
            self._heap.delete(node)
            new_node = self._heap.insert(new_priority, name)
            self._nodes[name] = new_node

    def delete(self, name):
        """Delete a named entry."""
        if name not in self._nodes:
            raise KeyError(name)
        node = self._nodes.pop(name)
        self._heap.delete(node)


class MergeableFibonacciHeap:
    """Named heaps that can be merged in O(1)."""

    def __init__(self):
        self._heaps = {}  # name -> FibonacciHeap

    def create_heap(self, name):
        """Create a new named heap."""
        if name in self._heaps:
            raise ValueError(f"heap '{name}' already exists")
        self._heaps[name] = FibonacciHeap()

    def get_heap(self, name):
        """Get a named heap."""
        if name not in self._heaps:
            raise KeyError(name)
        return self._heaps[name]

    def insert(self, heap_name, key, value=None):
        """Insert into a named heap."""
        return self._heaps[heap_name].insert(key, value)

    def find_min(self, heap_name):
        """Find min of a named heap."""
        return self._heaps[heap_name].find_min()

    def extract_min(self, heap_name):
        """Extract min from a named heap."""
        return self._heaps[heap_name].extract_min()

    def merge_heaps(self, name1, name2, new_name=None):
        """Merge heap name2 into name1 (or into new_name). name2 is removed."""
        h1 = self._heaps[name1]
        h2 = self._heaps.pop(name2)
        h1.merge(h2)
        if new_name and new_name != name1:
            self._heaps[new_name] = self._heaps.pop(name1)

    def heap_names(self):
        """Return list of heap names."""
        return list(self._heaps.keys())

    def __contains__(self, name):
        return name in self._heaps


class FibonacciHeapPQ:
    """Priority queue interface backed by Fibonacci heap.

    Supports push, pop, peek, update, and iteration.
    Items are (priority, item) tuples.
    """

    def __init__(self, items=None, max_heap=False):
        self._max_heap = max_heap
        self._heap = FibonacciHeap()
        self._item_nodes = {}  # item -> node
        self._sign = -1 if max_heap else 1
        if items:
            for priority, item in items:
                self.push(priority, item)

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def push(self, priority, item):
        """Push an item with given priority."""
        if item in self._item_nodes:
            self.update(item, priority)
            return
        node = self._heap.insert(self._sign * priority, item)
        self._item_nodes[item] = node

    def pop(self):
        """Remove and return (priority, item) with best priority."""
        node = self._heap.extract_min()
        item = node.value
        self._item_nodes.pop(item, None)
        return (self._sign * node.key, item)

    def peek(self):
        """Return (priority, item) with best priority without removing."""
        if self._heap.is_empty():
            raise IndexError("peek from empty priority queue")
        node = self._heap.find_min()
        return (self._sign * node.key, node.value)

    def update(self, item, new_priority):
        """Update the priority of an existing item."""
        if item not in self._item_nodes:
            raise KeyError(item)
        node = self._item_nodes[item]
        internal_new = self._sign * new_priority
        if internal_new < node.key:
            self._heap.decrease_key(node, internal_new)
        elif internal_new > node.key:
            self._heap.delete(node)
            new_node = self._heap.insert(internal_new, item)
            self._item_nodes[item] = new_node

    def delete(self, item):
        """Delete an item from the queue."""
        if item not in self._item_nodes:
            raise KeyError(item)
        node = self._item_nodes.pop(item)
        self._heap.delete(node)

    def __contains__(self, item):
        return item in self._item_nodes

    def get_priority(self, item):
        """Get the current priority of an item."""
        if item not in self._item_nodes:
            raise KeyError(item)
        return self._sign * self._item_nodes[item].key


# -- Utility: Dijkstra's algorithm using FibonacciHeapPQ --

def dijkstra(graph, source):
    """Dijkstra's shortest path using Fibonacci heap.

    graph: dict of {node: [(neighbor, weight), ...]}
    source: start node
    Returns: (distances, predecessors)
    """
    dist = {source: 0}
    pred = {source: None}
    pq = FibonacciHeapPQ()
    pq.push(0, source)

    for node in graph:
        if node != source:
            dist[node] = float('inf')

    while pq:
        d, u = pq.pop()
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph.get(u, []):
            alt = d + w
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                pred[v] = u
                if v in pq:
                    pq.update(v, alt)
                else:
                    pq.push(alt, v)

    return dist, pred


def prim_mst(graph):
    """Prim's MST using Fibonacci heap.

    graph: dict of {node: [(neighbor, weight), ...]}
    Returns: list of (u, v, weight) edges in MST
    """
    if not graph:
        return []

    nodes = list(graph.keys())
    start = nodes[0]
    in_mst = set()
    pq = FibonacciHeapPQ()
    edge_to = {}  # node -> (from_node, weight)

    pq.push(0, start)
    edge_to[start] = (None, 0)
    mst_edges = []

    while pq:
        w, u = pq.pop()
        if u in in_mst:
            continue
        in_mst.add(u)
        from_node, weight = edge_to[u]
        if from_node is not None:
            mst_edges.append((from_node, u, weight))

        for v, edge_w in graph.get(u, []):
            if v not in in_mst:
                if v not in pq:
                    pq.push(edge_w, v)
                    edge_to[v] = (u, edge_w)
                elif edge_w < pq.get_priority(v):
                    pq.update(v, edge_w)
                    edge_to[v] = (u, edge_w)

    return mst_edges
