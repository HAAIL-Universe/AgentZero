"""
C123: D-ary Heap -- Generalized heap with configurable branching factor.

Variants:
1. DaryHeap -- min-heap with configurable d (children per node)
2. MaxDaryHeap -- max-heap variant
3. DaryHeapMap -- heap with decrease-key via index tracking
4. MergeableDaryHeap -- merge support via batch insert
5. DaryHeapPQ -- priority queue with named entries
6. MedianDaryHeap -- dual-heap median maintenance

Utilities:
- heap_sort(items, d=4) -- sort via d-ary heap
- k_smallest(items, k, d=4) -- extract k smallest
- k_largest(items, k, d=4) -- extract k largest
- merge_sorted(iterables, d=4) -- merge sorted iterables
"""


class DaryHeap:
    """Min-heap with configurable branching factor d."""

    def __init__(self, d=4, items=None):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self._data = []
        if items:
            self._data = list(items)
            self._heapify()

    def _heapify(self):
        n = len(self._data)
        for i in range((n - 2) // self.d, -1, -1):
            self._sift_down(i)

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = i * self.d + 1
        end = min(start + self.d, len(self._data))
        return range(start, end)

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self._data[i] < self._data[p]:
                self._data[i], self._data[p] = self._data[p], self._data[i]
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            smallest = i
            for c in self._children(i):
                if c < n and self._data[c] < self._data[smallest]:
                    smallest = c
            if smallest == i:
                break
            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest

    def push(self, value):
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        if len(self._data) == 1:
            return self._data.pop()
        result = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return result

    def peek(self):
        if not self._data:
            raise IndexError("peek at empty heap")
        return self._data[0]

    def pushpop(self, value):
        if self._data and self._data[0] < value:
            result = self._data[0]
            self._data[0] = value
            self._sift_down(0)
            return result
        return value

    def replace(self, value):
        if not self._data:
            raise IndexError("replace on empty heap")
        result = self._data[0]
        self._data[0] = value
        self._sift_down(0)
        return result

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def __iter__(self):
        copy = DaryHeap(self.d, self._data)
        while copy:
            yield copy.pop()

    def __repr__(self):
        return f"DaryHeap(d={self.d}, size={len(self._data)})"


class MaxDaryHeap:
    """Max-heap with configurable branching factor d."""

    def __init__(self, d=4, items=None):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self._data = []
        if items:
            self._data = list(items)
            self._heapify()

    def _heapify(self):
        n = len(self._data)
        for i in range((n - 2) // self.d, -1, -1):
            self._sift_down(i)

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = i * self.d + 1
        end = min(start + self.d, len(self._data))
        return range(start, end)

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self._data[i] > self._data[p]:
                self._data[i], self._data[p] = self._data[p], self._data[i]
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            largest = i
            for c in self._children(i):
                if c < n and self._data[c] > self._data[largest]:
                    largest = c
            if largest == i:
                break
            self._data[i], self._data[largest] = self._data[largest], self._data[i]
            i = largest

    def push(self, value):
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        if len(self._data) == 1:
            return self._data.pop()
        result = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return result

    def peek(self):
        if not self._data:
            raise IndexError("peek at empty heap")
        return self._data[0]

    def pushpop(self, value):
        if self._data and self._data[0] > value:
            result = self._data[0]
            self._data[0] = value
            self._sift_down(0)
            return result
        return value

    def replace(self, value):
        if not self._data:
            raise IndexError("replace on empty heap")
        result = self._data[0]
        self._data[0] = value
        self._sift_down(0)
        return result

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def __iter__(self):
        copy = MaxDaryHeap(self.d, self._data)
        while copy:
            yield copy.pop()

    def __repr__(self):
        return f"MaxDaryHeap(d={self.d}, size={len(self._data)})"


class DaryHeapMap:
    """D-ary min-heap with decrease-key support via index tracking."""

    def __init__(self, d=4):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self._data = []       # list of (priority, key)
        self._index = {}      # key -> index in _data
        self._counter = 0

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = i * self.d + 1
        end = min(start + self.d, len(self._data))
        return range(start, end)

    def _swap(self, i, j):
        self._data[i], self._data[j] = self._data[j], self._data[i]
        self._index[self._data[i][1]] = i
        self._index[self._data[j][1]] = j

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self._data[i][0] < self._data[p][0]:
                self._swap(i, p)
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            smallest = i
            for c in self._children(i):
                if c < n and self._data[c][0] < self._data[smallest][0]:
                    smallest = c
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def push(self, key, priority):
        if key in self._index:
            raise KeyError(f"key {key!r} already exists, use decrease_key")
        idx = len(self._data)
        self._data.append((priority, key))
        self._index[key] = idx
        self._sift_up(idx)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        priority, key = self._data[0]
        if len(self._data) == 1:
            self._data.pop()
            del self._index[key]
            return key, priority
        last = self._data.pop()
        self._data[0] = last
        self._index[last[1]] = 0
        del self._index[key]
        self._sift_down(0)
        return key, priority

    def peek(self):
        if not self._data:
            raise IndexError("peek at empty heap")
        priority, key = self._data[0]
        return key, priority

    def decrease_key(self, key, new_priority):
        if key not in self._index:
            raise KeyError(f"key {key!r} not found")
        idx = self._index[key]
        old_priority = self._data[idx][0]
        if new_priority > old_priority:
            raise ValueError("new priority must be <= old priority")
        self._data[idx] = (new_priority, key)
        self._sift_up(idx)

    def update_key(self, key, new_priority):
        """Update priority (increase or decrease)."""
        if key not in self._index:
            raise KeyError(f"key {key!r} not found")
        idx = self._index[key]
        old_priority = self._data[idx][0]
        self._data[idx] = (new_priority, key)
        if new_priority < old_priority:
            self._sift_up(idx)
        else:
            self._sift_down(idx)

    def remove(self, key):
        if key not in self._index:
            raise KeyError(f"key {key!r} not found")
        idx = self._index[key]
        if idx == len(self._data) - 1:
            self._data.pop()
            del self._index[key]
            return
        last = self._data.pop()
        self._data[idx] = last
        self._index[last[1]] = idx
        del self._index[key]
        old_priority = last[0]
        parent_idx = self._parent(idx)
        if idx > 0 and self._data[idx][0] < self._data[parent_idx][0]:
            self._sift_up(idx)
        else:
            self._sift_down(idx)

    def __contains__(self, key):
        return key in self._index

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def get_priority(self, key):
        if key not in self._index:
            raise KeyError(f"key {key!r} not found")
        idx = self._index[key]
        return self._data[idx][0]

    def __repr__(self):
        return f"DaryHeapMap(d={self.d}, size={len(self._data)})"


class MergeableDaryHeap:
    """D-ary min-heap with efficient merge via batch insert + heapify."""

    def __init__(self, d=4, items=None):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self._data = []
        if items:
            self._data = list(items)
            self._heapify()

    def _heapify(self):
        n = len(self._data)
        for i in range((n - 2) // self.d, -1, -1):
            self._sift_down(i)

    def _parent(self, i):
        return (i - 1) // self.d

    def _children(self, i):
        start = i * self.d + 1
        end = min(start + self.d, len(self._data))
        return range(start, end)

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self._data[i] < self._data[p]:
                self._data[i], self._data[p] = self._data[p], self._data[i]
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            smallest = i
            for c in self._children(i):
                if c < n and self._data[c] < self._data[smallest]:
                    smallest = c
            if smallest == i:
                break
            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest

    def push(self, value):
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        if len(self._data) == 1:
            return self._data.pop()
        result = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return result

    def peek(self):
        if not self._data:
            raise IndexError("peek at empty heap")
        return self._data[0]

    def merge(self, other):
        """Merge another heap into this one. O(n+m) via concatenate + heapify."""
        if isinstance(other, MergeableDaryHeap):
            self._data.extend(other._data)
        else:
            self._data.extend(other)
        self._heapify()

    def merge_many(self, *others):
        """Merge multiple heaps/iterables."""
        for other in others:
            if isinstance(other, MergeableDaryHeap):
                self._data.extend(other._data)
            else:
                self._data.extend(other)
        self._heapify()

    def split(self):
        """Split heap into two roughly equal heaps."""
        mid = len(self._data) // 2
        left = MergeableDaryHeap(self.d, self._data[:mid])
        right = MergeableDaryHeap(self.d, self._data[mid:])
        return left, right

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def __iter__(self):
        copy = MergeableDaryHeap(self.d, self._data)
        while copy:
            yield copy.pop()

    def __repr__(self):
        return f"MergeableDaryHeap(d={self.d}, size={len(self._data)})"


class DaryHeapPQ:
    """Priority queue with named entries, backed by d-ary heap."""

    def __init__(self, d=4):
        if d < 2:
            raise ValueError("d must be >= 2")
        self._heap = DaryHeapMap(d)
        self.d = d

    def enqueue(self, name, priority):
        self._heap.push(name, priority)

    def dequeue(self):
        if not self._heap:
            raise IndexError("dequeue from empty queue")
        return self._heap.pop()

    def peek(self):
        if not self._heap:
            raise IndexError("peek at empty queue")
        return self._heap.peek()

    def update_priority(self, name, priority):
        self._heap.update_key(name, priority)

    def remove(self, name):
        self._heap.remove(name)

    def __contains__(self, name):
        return name in self._heap

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def __repr__(self):
        return f"DaryHeapPQ(d={self.d}, size={len(self._heap)})"


class MedianDaryHeap:
    """Maintains running median using two d-ary heaps."""

    def __init__(self, d=4):
        if d < 2:
            raise ValueError("d must be >= 2")
        self.d = d
        self._lo = MaxDaryHeap(d)    # max-heap for lower half
        self._hi = DaryHeap(d)       # min-heap for upper half
        self._size = 0

    def _balance(self):
        if len(self._lo) > len(self._hi) + 1:
            self._hi.push(self._lo.pop())
        elif len(self._hi) > len(self._lo) + 1:
            self._lo.push(self._hi.pop())

    def push(self, value):
        self._size += 1
        if not self._lo or value <= self._lo.peek():
            self._lo.push(value)
        else:
            self._hi.push(value)
        self._balance()

    def median(self):
        if self._size == 0:
            raise IndexError("median of empty collection")
        if len(self._lo) > len(self._hi):
            return self._lo.peek()
        elif len(self._hi) > len(self._lo):
            return self._hi.peek()
        else:
            return (self._lo.peek() + self._hi.peek()) / 2

    def pop_median(self):
        if self._size == 0:
            raise IndexError("pop from empty collection")
        self._size -= 1
        if len(self._lo) >= len(self._hi):
            result = self._lo.pop()
        else:
            result = self._hi.pop()
        self._balance()
        return result

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    def __repr__(self):
        return f"MedianDaryHeap(d={self.d}, size={self._size})"


# === Utility Functions ===

def heap_sort(items, d=4, reverse=False):
    """Sort items using a d-ary heap. O(n log_d n)."""
    if reverse:
        h = MaxDaryHeap(d, items)
    else:
        h = DaryHeap(d, items)
    return list(h)


def k_smallest(items, k, d=4):
    """Return k smallest items. O(n + k log_d n)."""
    h = DaryHeap(d, items)
    result = []
    for _ in range(min(k, len(h))):
        result.append(h.pop())
    return result


def k_largest(items, k, d=4):
    """Return k largest items. O(n + k log_d n)."""
    h = MaxDaryHeap(d, items)
    result = []
    for _ in range(min(k, len(h))):
        result.append(h.pop())
    return result


def merge_sorted(*iterables, d=4):
    """Merge multiple sorted iterables into one sorted sequence."""
    iters = []
    for it in iterables:
        it = iter(it)
        try:
            val = next(it)
            iters.append((val, it))
        except StopIteration:
            pass

    h = DaryHeap(d)
    for idx, (val, it) in enumerate(iters):
        h.push((val, idx))

    result = []
    while h:
        val, idx = h.pop()
        result.append(val)
        try:
            nxt = next(iters[idx][1])
            h.push((nxt, idx))
        except StopIteration:
            pass

    return result


def nsmallest(items, n, d=4):
    """Like heapq.nsmallest -- efficient for small n relative to len(items)."""
    if n <= 0:
        return []
    items = list(items)
    if n >= len(items):
        return sorted(items)
    # Use a max-heap of size n
    h = MaxDaryHeap(d, items[:n])
    for item in items[n:]:
        if item < h.peek():
            h.replace(item)
    return sorted(h)


def nlargest(items, n, d=4):
    """Like heapq.nlargest -- efficient for small n relative to len(items)."""
    if n <= 0:
        return []
    items = list(items)
    if n >= len(items):
        return sorted(items, reverse=True)
    # Use a min-heap of size n
    h = DaryHeap(d, items[:n])
    for item in items[n:]:
        if item > h.peek():
            h.replace(item)
    return sorted(h, reverse=True)
