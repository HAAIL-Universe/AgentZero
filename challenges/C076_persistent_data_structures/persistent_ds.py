"""
C076: Persistent Data Structures
Immutable collections with structural sharing for efficient functional programming.

Data structures:
- PersistentVector: Bit-partitioned trie (32-way branching), O(log32 N) ops
- PersistentHashMap: Hash Array Mapped Trie (HAMT), O(log32 N) ops
- PersistentList: Immutable cons list, O(1) prepend/head/tail
- PersistentSortedSet: Persistent red-black tree, O(log N) ops
- Transient variants for batch mutation performance
"""

from __future__ import annotations
from typing import Any, Iterator, Optional, Callable
import itertools

# ============================================================
# PersistentVector -- Bit-partitioned trie
# ============================================================

BITS = 5
BRANCH_FACTOR = 1 << BITS  # 32
MASK = BRANCH_FACTOR - 1


class _VNode:
    """Internal node of the vector trie."""
    __slots__ = ('children', 'owner')

    def __init__(self, children: list = None, owner: object = None):
        self.children = children if children is not None else []
        self.owner = owner  # None = persistent, non-None = transient owner

    def copy(self, owner: object = None) -> '_VNode':
        return _VNode(list(self.children), owner)

    def ensure_editable(self, owner: object) -> '_VNode':
        if self.owner is owner:
            return self
        return self.copy(owner)


class PersistentVector:
    """
    Immutable vector with structural sharing.
    Uses a 32-way branching trie for O(log32 N) access/update.
    Tail optimization for fast appends.
    """
    __slots__ = ('_count', '_shift', '_root', '_tail')

    def __init__(self, count: int = 0, shift: int = BITS,
                 root: _VNode = None, tail: list = None):
        self._count = count
        self._shift = shift
        self._root = root or _VNode()
        self._tail = tail if tail is not None else []

    @staticmethod
    def empty() -> 'PersistentVector':
        return PersistentVector()

    @staticmethod
    def of(*args) -> 'PersistentVector':
        v = PersistentVector()
        for item in args:
            v = v.append(item)
        return v

    @staticmethod
    def from_iterable(iterable) -> 'PersistentVector':
        t = TransientVector(PersistentVector())
        for item in iterable:
            t.append_mut(item)
        return t.persistent()

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def _tail_offset(self) -> int:
        if self._count < BRANCH_FACTOR:
            return 0
        return ((self._count - 1) >> BITS) << BITS

    def _array_for(self, i: int) -> list:
        """Find the leaf array containing index i."""
        if i < 0 or i >= self._count:
            raise IndexError(f"Index {i} out of range for vector of size {self._count}")
        if i >= self._tail_offset():
            return self._tail
        node = self._root
        level = self._shift
        while level > 0:
            idx = (i >> level) & MASK
            if idx < len(node.children):
                node = node.children[idx]
            else:
                raise IndexError(f"Internal trie error at level {level}")
            level -= BITS
        return node.children

    def get(self, index: int) -> Any:
        """Get element at index. Supports negative indexing."""
        if index < 0:
            index += self._count
        arr = self._array_for(index)
        return arr[index & MASK]

    def __getitem__(self, index) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return PersistentVector.from_iterable(
                self.get(i) for i in range(start, stop, step)
            )
        if index < 0:
            index += self._count
        return self.get(index)

    def set(self, index: int, val: Any) -> 'PersistentVector':
        """Return new vector with val at index."""
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(f"Index {index} out of range")
        if index >= self._tail_offset():
            new_tail = list(self._tail)
            new_tail[index & MASK] = val
            return PersistentVector(self._count, self._shift, self._root, new_tail)
        new_root = self._do_set(self._shift, self._root, index, val)
        return PersistentVector(self._count, self._shift, new_root, list(self._tail))

    def _do_set(self, level: int, node: _VNode, i: int, val: Any) -> _VNode:
        new_node = node.copy()
        if level == 0:
            idx = i & MASK
            while len(new_node.children) <= idx:
                new_node.children.append(None)
            new_node.children[idx] = val
        else:
            sub_idx = (i >> level) & MASK
            while len(new_node.children) <= sub_idx:
                new_node.children.append(_VNode())
            new_node.children[sub_idx] = self._do_set(
                level - BITS, new_node.children[sub_idx], i, val
            )
        return new_node

    def append(self, val: Any) -> 'PersistentVector':
        """Return new vector with val appended."""
        # Room in tail?
        if len(self._tail) < BRANCH_FACTOR:
            new_tail = list(self._tail)
            new_tail.append(val)
            return PersistentVector(self._count + 1, self._shift, self._root, new_tail)
        # Tail full -- push tail into trie
        tail_node = _VNode(list(self._tail))
        new_shift = self._shift
        new_root = self._root
        # Overflow root?
        if (self._count >> BITS) > (1 << self._shift):
            new_root = _VNode([self._root, self._new_path(self._shift, tail_node)])
            new_shift += BITS
        else:
            new_root = self._push_tail(self._shift, self._root, tail_node)
        return PersistentVector(self._count + 1, new_shift, new_root, [val])

    def _push_tail(self, level: int, parent: _VNode, tail_node: _VNode) -> _VNode:
        new_parent = parent.copy()
        sub_idx = ((self._count - 1) >> level) & MASK
        if level == BITS:
            while len(new_parent.children) <= sub_idx:
                new_parent.children.append(None)
            new_parent.children[sub_idx] = tail_node
        else:
            if sub_idx < len(new_parent.children) and new_parent.children[sub_idx] is not None:
                new_child = self._push_tail(
                    level - BITS, new_parent.children[sub_idx], tail_node
                )
                new_parent.children[sub_idx] = new_child
            else:
                while len(new_parent.children) <= sub_idx:
                    new_parent.children.append(None)
                new_parent.children[sub_idx] = self._new_path(level - BITS, tail_node)
        return new_parent

    def _new_path(self, level: int, node: _VNode) -> _VNode:
        if level == 0:
            return node
        return _VNode([self._new_path(level - BITS, node)])

    def pop(self) -> 'PersistentVector':
        """Return new vector with last element removed."""
        if self._count == 0:
            raise IndexError("Cannot pop from empty vector")
        if self._count == 1:
            return PersistentVector()
        # More than one element in tail?
        if len(self._tail) > 1:
            new_tail = list(self._tail[:-1])
            return PersistentVector(self._count - 1, self._shift, self._root, new_tail)
        # Tail has 1 element -- need to pop from trie
        new_tail = self._array_for(self._count - 2)
        new_root = self._pop_tail(self._shift, self._root)
        new_shift = self._shift
        if new_root is None:
            new_root = _VNode()
        if self._shift > BITS and len(new_root.children) == 1:
            new_root = new_root.children[0]
            new_shift -= BITS
        return PersistentVector(self._count - 1, new_shift, new_root, list(new_tail))

    def _pop_tail(self, level: int, node: _VNode) -> Optional[_VNode]:
        sub_idx = ((self._count - 2) >> level) & MASK
        if level > BITS:
            new_child = self._pop_tail(level - BITS, node.children[sub_idx])
            if new_child is None and sub_idx == 0:
                return None
            new_node = node.copy()
            if new_child is None:
                new_node.children = list(new_node.children[:sub_idx])
            else:
                new_node.children[sub_idx] = new_child
            return new_node
        elif sub_idx == 0:
            return None
        else:
            new_node = node.copy()
            new_node.children = list(new_node.children[:sub_idx])
            return new_node

    def __iter__(self) -> Iterator:
        for i in range(self._count):
            yield self.get(i)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PersistentVector):
            return NotImplemented
        if self._count != other._count:
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    def __hash__(self) -> int:
        h = 0
        for item in self:
            h = h * 31 + hash(item)
        return h

    def __repr__(self) -> str:
        items = ', '.join(repr(x) for x in self)
        return f"PVec([{items}])"

    def map(self, fn: Callable) -> 'PersistentVector':
        return PersistentVector.from_iterable(fn(x) for x in self)

    def filter(self, fn: Callable) -> 'PersistentVector':
        return PersistentVector.from_iterable(x for x in self if fn(x))

    def reduce(self, fn: Callable, init: Any = None) -> Any:
        it = iter(self)
        if init is None:
            try:
                acc = next(it)
            except StopIteration:
                raise TypeError("reduce() of empty vector with no initial value")
        else:
            acc = init
        for x in it:
            acc = fn(acc, x)
        return acc

    def slice(self, start: int = 0, end: int = None) -> 'PersistentVector':
        if end is None:
            end = self._count
        if start < 0:
            start += self._count
        if end < 0:
            end += self._count
        start = max(0, min(start, self._count))
        end = max(0, min(end, self._count))
        return PersistentVector.from_iterable(self.get(i) for i in range(start, end))

    def concat(self, other: 'PersistentVector') -> 'PersistentVector':
        t = TransientVector(self)
        for x in other:
            t.append_mut(x)
        return t.persistent()

    def index_of(self, val: Any) -> int:
        for i in range(self._count):
            if self.get(i) == val:
                return i
        return -1

    def contains(self, val: Any) -> bool:
        return self.index_of(val) != -1

    def to_list(self) -> list:
        return list(self)

    def transient(self) -> 'TransientVector':
        return TransientVector(self)


class TransientVector:
    """Mutable version for batch operations. Call persistent() when done."""

    def __init__(self, pvec: PersistentVector):
        self._owner = object()  # unique identity
        self._count = pvec._count
        self._shift = pvec._shift
        self._root = pvec._root.ensure_editable(self._owner)
        self._tail = list(pvec._tail)
        self._persisted = False

    def _ensure_mutable(self):
        if self._persisted:
            raise RuntimeError("TransientVector already persisted")

    def append_mut(self, val: Any) -> 'TransientVector':
        self._ensure_mutable()
        if len(self._tail) < BRANCH_FACTOR:
            self._tail.append(val)
            self._count += 1
            return self
        # Push tail into trie
        tail_node = _VNode(self._tail, self._owner)
        self._tail = [val]
        if (self._count >> BITS) > (1 << self._shift):
            new_root = _VNode([self._root,
                               self._new_path(self._shift, tail_node)],
                              self._owner)
            self._shift += BITS
            self._root = new_root
        else:
            self._root = self._push_tail_mut(self._shift, self._root, tail_node)
        self._count += 1
        return self

    def _push_tail_mut(self, level: int, parent: _VNode, tail_node: _VNode) -> _VNode:
        parent = parent.ensure_editable(self._owner)
        sub_idx = ((self._count - 1) >> level) & MASK
        if level == BITS:
            while len(parent.children) <= sub_idx:
                parent.children.append(None)
            parent.children[sub_idx] = tail_node
        else:
            if sub_idx < len(parent.children) and parent.children[sub_idx] is not None:
                parent.children[sub_idx] = self._push_tail_mut(
                    level - BITS, parent.children[sub_idx], tail_node
                )
            else:
                while len(parent.children) <= sub_idx:
                    parent.children.append(None)
                parent.children[sub_idx] = self._new_path(level - BITS, tail_node)
        return parent

    def _new_path(self, level: int, node: _VNode) -> _VNode:
        if level == 0:
            return node
        return _VNode([self._new_path(level - BITS, node)], self._owner)

    def set_mut(self, index: int, val: Any) -> 'TransientVector':
        self._ensure_mutable()
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(f"Index {index} out of range")
        tail_off = 0 if self._count < BRANCH_FACTOR else ((self._count - 1) >> BITS) << BITS
        if index >= tail_off:
            self._tail[index & MASK] = val
            return self
        self._root = self._do_set_mut(self._shift, self._root, index, val)
        return self

    def _do_set_mut(self, level: int, node: _VNode, i: int, val: Any) -> _VNode:
        node = node.ensure_editable(self._owner)
        if level == 0:
            idx = i & MASK
            while len(node.children) <= idx:
                node.children.append(None)
            node.children[idx] = val
        else:
            sub_idx = (i >> level) & MASK
            while len(node.children) <= sub_idx:
                node.children.append(_VNode([], self._owner))
            node.children[sub_idx] = self._do_set_mut(
                level - BITS, node.children[sub_idx], i, val
            )
        return node

    def persistent(self) -> PersistentVector:
        self._persisted = True
        self._owner = None
        return PersistentVector(self._count, self._shift, self._root, self._tail)


# ============================================================
# PersistentList -- Cons list
# ============================================================

class _Nil:
    """Empty list sentinel."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __repr__(self):
        return "PList([])"

    def __eq__(self, other):
        return isinstance(other, _Nil) or (isinstance(other, PersistentList) and other._count == 0)

    def __hash__(self):
        return 0


NIL = _Nil()


class PersistentList:
    """Immutable singly-linked list. O(1) prepend, head, tail."""
    __slots__ = ('_head', '_rest', '_count')

    def __init__(self, head: Any, rest=None, count: int = 1):
        self._head = head
        self._rest = rest if rest is not None else NIL
        self._count = count

    @staticmethod
    def empty():
        return NIL

    @staticmethod
    def of(*args) -> 'PersistentList':
        result = NIL
        for item in reversed(args):
            result = PersistentList(item, result,
                                   (result._count if isinstance(result, PersistentList) else 0) + 1)
        return result

    @staticmethod
    def from_iterable(iterable):
        items = list(iterable)
        return PersistentList.of(*items)

    def cons(self, val: Any) -> 'PersistentList':
        """Prepend val. O(1)."""
        return PersistentList(val, self, self._count + 1)

    @property
    def head(self) -> Any:
        return self._head

    @property
    def first(self) -> Any:
        return self._head

    @property
    def rest(self):
        return self._rest

    @property
    def tail(self):
        return self._rest

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return True

    def get(self, index: int) -> Any:
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(f"Index {index} out of range")
        curr = self
        for _ in range(index):
            curr = curr._rest
        return curr._head

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __iter__(self) -> Iterator:
        curr = self
        while isinstance(curr, PersistentList):
            yield curr._head
            curr = curr._rest

    def __eq__(self, other) -> bool:
        if isinstance(other, _Nil):
            return self._count == 0
        if not isinstance(other, PersistentList):
            return NotImplemented
        if self._count != other._count:
            return False
        return all(a == b for a, b in zip(self, other))

    def __hash__(self) -> int:
        h = 0
        for item in self:
            h = h * 31 + hash(item)
        return h

    def __repr__(self) -> str:
        items = ', '.join(repr(x) for x in self)
        return f"PList([{items}])"

    def map(self, fn: Callable):
        return PersistentList.from_iterable(fn(x) for x in self)

    def filter(self, fn: Callable):
        return PersistentList.from_iterable(x for x in self if fn(x))

    def reduce(self, fn: Callable, init: Any = None) -> Any:
        it = iter(self)
        if init is None:
            try:
                acc = next(it)
            except StopIteration:
                raise TypeError("reduce() of empty list with no initial value")
        else:
            acc = init
        for x in it:
            acc = fn(acc, x)
        return acc

    def reverse(self):
        result = NIL
        for x in self:
            if isinstance(result, _Nil):
                result = PersistentList(x, NIL, 1)
            else:
                result = result.cons(x)
        return result

    def take(self, n: int):
        return PersistentList.from_iterable(itertools.islice(self, n))

    def drop(self, n: int):
        curr = self
        for _ in range(n):
            if not isinstance(curr, PersistentList):
                return NIL
            curr = curr._rest
        return curr

    def concat(self, other):
        """Concatenate two lists."""
        items = list(self) + list(other)
        return PersistentList.of(*items)

    def to_list(self) -> list:
        return list(self)


# ============================================================
# PersistentHashMap -- Hash Array Mapped Trie (HAMT)
# ============================================================

class _HEmpty:
    """Empty HAMT sentinel."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_HEMPTY = _HEmpty()


class _HLeaf:
    """Single key-value pair at a hash position."""
    __slots__ = ('hash', 'key', 'value')

    def __init__(self, h: int, key: Any, value: Any):
        self.hash = h
        self.key = key
        self.value = value


class _HCollision:
    """Multiple entries with the same hash."""
    __slots__ = ('hash', 'entries')

    def __init__(self, h: int, entries: list):
        self.hash = h
        self.entries = entries  # list of (key, value)


class _HBranch:
    """Interior HAMT node with bitmap-indexed children."""
    __slots__ = ('bitmap', 'children', 'owner')

    def __init__(self, bitmap: int, children: list, owner: object = None):
        self.bitmap = bitmap
        self.children = children
        self.owner = owner

    def copy(self, owner: object = None) -> '_HBranch':
        return _HBranch(self.bitmap, list(self.children), owner)

    def ensure_editable(self, owner: object) -> '_HBranch':
        if self.owner is owner:
            return self
        return self.copy(owner)


def _hamt_index(bitmap: int, bit: int) -> int:
    """Count set bits below 'bit' in bitmap."""
    return bin(bitmap & (bit - 1)).count('1')


class PersistentHashMap:
    """
    Immutable hash map using a Hash Array Mapped Trie (HAMT).
    O(log32 N) get/set/delete with structural sharing.
    """
    __slots__ = ('_root', '_count')

    def __init__(self, root=None, count: int = 0):
        self._root = root if root is not None else _HEMPTY
        self._count = count

    @staticmethod
    def empty() -> 'PersistentHashMap':
        return PersistentHashMap()

    @staticmethod
    def of(**kwargs) -> 'PersistentHashMap':
        m = PersistentHashMap()
        for k, v in kwargs.items():
            m = m.set(k, v)
        return m

    @staticmethod
    def from_dict(d: dict) -> 'PersistentHashMap':
        t = TransientHashMap(PersistentHashMap())
        for k, v in d.items():
            t.set_mut(k, v)
        return t.persistent()

    @staticmethod
    def from_pairs(pairs) -> 'PersistentHashMap':
        t = TransientHashMap(PersistentHashMap())
        for k, v in pairs:
            t.set_mut(k, v)
        return t.persistent()

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def get(self, key: Any, default: Any = None) -> Any:
        return self._get_node(self._root, hash(key), key, 0, default)

    def _get_node(self, node, h: int, key: Any, shift: int, default: Any) -> Any:
        if isinstance(node, _HEmpty):
            return default
        if isinstance(node, _HLeaf):
            if node.key == key:
                return node.value
            return default
        if isinstance(node, _HCollision):
            for k, v in node.entries:
                if k == key:
                    return v
            return default
        # _HBranch
        bit = 1 << ((h >> shift) & MASK)
        if not (node.bitmap & bit):
            return default
        idx = _hamt_index(node.bitmap, bit)
        return self._get_node(node.children[idx], h, key, shift + BITS, default)

    def __getitem__(self, key: Any) -> Any:
        sentinel = object()
        result = self.get(key, sentinel)
        if result is sentinel:
            raise KeyError(key)
        return result

    def __contains__(self, key: Any) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def set(self, key: Any, value: Any) -> 'PersistentHashMap':
        """Return new map with key=value."""
        h = hash(key)
        new_root, added = self._set_node(self._root, h, key, value, 0)
        new_count = self._count + (1 if added else 0)
        return PersistentHashMap(new_root, new_count)

    def _set_node(self, node, h: int, key: Any, value: Any, shift: int):
        """Returns (new_node, was_added)."""
        if isinstance(node, _HEmpty):
            return _HLeaf(h, key, value), True

        if isinstance(node, _HLeaf):
            if node.hash == h:
                if node.key == key:
                    # Update existing
                    return _HLeaf(h, key, value), False
                # Hash collision
                return _HCollision(h, [(node.key, node.value), (key, value)]), True
            # Different hash -- create branch
            return self._make_branch(node, _HLeaf(h, key, value), shift), True

        if isinstance(node, _HCollision):
            if node.hash == h:
                for i, (k, v) in enumerate(node.entries):
                    if k == key:
                        new_entries = list(node.entries)
                        new_entries[i] = (key, value)
                        return _HCollision(h, new_entries), False
                return _HCollision(h, node.entries + [(key, value)]), True
            # Different hash -- wrap in branch
            return self._make_branch(node, _HLeaf(h, key, value), shift), True

        # _HBranch
        bit = 1 << ((h >> shift) & MASK)
        idx = _hamt_index(node.bitmap, bit)
        if node.bitmap & bit:
            child = node.children[idx]
            new_child, added = self._set_node(child, h, key, value, shift + BITS)
            new_children = list(node.children)
            new_children[idx] = new_child
            return _HBranch(node.bitmap, new_children), added
        else:
            new_children = list(node.children)
            new_children.insert(idx, _HLeaf(h, key, value))
            return _HBranch(node.bitmap | bit, new_children), True

    def _make_branch(self, existing, new_leaf, shift: int):
        """Create branch node(s) to distinguish two entries by their hashes."""
        if isinstance(existing, _HLeaf):
            eh = existing.hash
        else:  # _HCollision
            eh = existing.hash
        nh = new_leaf.hash

        e_idx = (eh >> shift) & MASK
        n_idx = (nh >> shift) & MASK

        if e_idx == n_idx:
            # Same slot -- recurse deeper
            child = self._make_branch(existing, new_leaf, shift + BITS)
            bit = 1 << e_idx
            return _HBranch(bit, [child])
        else:
            e_bit = 1 << e_idx
            n_bit = 1 << n_idx
            bitmap = e_bit | n_bit
            if e_idx < n_idx:
                children = [existing, new_leaf]
            else:
                children = [new_leaf, existing]
            return _HBranch(bitmap, children)

    def delete(self, key: Any) -> 'PersistentHashMap':
        """Return new map without key. Returns self if key not present."""
        if key not in self:
            return self
        h = hash(key)
        new_root = self._delete_node(self._root, h, key, 0)
        if new_root is None:
            new_root = _HEMPTY
        return PersistentHashMap(new_root, self._count - 1)

    def _delete_node(self, node, h: int, key: Any, shift: int):
        if isinstance(node, _HEmpty):
            return node
        if isinstance(node, _HLeaf):
            if node.key == key:
                return None
            return node
        if isinstance(node, _HCollision):
            new_entries = [(k, v) for k, v in node.entries if k != key]
            if len(new_entries) == len(node.entries):
                return node
            if len(new_entries) == 1:
                return _HLeaf(node.hash, new_entries[0][0], new_entries[0][1])
            return _HCollision(node.hash, new_entries)
        # _HBranch
        bit = 1 << ((h >> shift) & MASK)
        if not (node.bitmap & bit):
            return node
        idx = _hamt_index(node.bitmap, bit)
        child = node.children[idx]
        new_child = self._delete_node(child, h, key, shift + BITS)
        if new_child is None:
            # Remove this slot
            if bin(node.bitmap).count('1') == 1:
                return None
            new_children = list(node.children)
            del new_children[idx]
            return _HBranch(node.bitmap & ~bit, new_children)
        if new_child is child:
            return node
        new_children = list(node.children)
        new_children[idx] = new_child
        return _HBranch(node.bitmap, new_children)

    def keys(self) -> Iterator:
        for k, v in self.items():
            yield k

    def values(self) -> Iterator:
        for k, v in self.items():
            yield v

    def items(self) -> Iterator:
        yield from self._iter_node(self._root)

    def _iter_node(self, node):
        if isinstance(node, _HEmpty):
            return
        if isinstance(node, _HLeaf):
            yield (node.key, node.value)
            return
        if isinstance(node, _HCollision):
            for k, v in node.entries:
                yield (k, v)
            return
        for child in node.children:
            yield from self._iter_node(child)

    def __iter__(self) -> Iterator:
        return self.keys()

    def __eq__(self, other) -> bool:
        if not isinstance(other, PersistentHashMap):
            return NotImplemented
        if self._count != other._count:
            return False
        for k, v in self.items():
            if other.get(k) != v:
                return False
        return True

    def __hash__(self) -> int:
        # Order-independent hash
        h = 0
        for k, v in self.items():
            h ^= hash((k, v))
        return h

    def __repr__(self) -> str:
        pairs = ', '.join(f'{k!r}: {v!r}' for k, v in sorted(self.items(), key=lambda x: repr(x[0])))
        return f"PMap({{{pairs}}})"

    def merge(self, other: 'PersistentHashMap') -> 'PersistentHashMap':
        """Return new map with all entries from both (other wins on conflict)."""
        t = TransientHashMap(self)
        for k, v in other.items():
            t.set_mut(k, v)
        return t.persistent()

    def map_values(self, fn: Callable) -> 'PersistentHashMap':
        t = TransientHashMap(PersistentHashMap())
        for k, v in self.items():
            t.set_mut(k, fn(v))
        return t.persistent()

    def filter_entries(self, fn: Callable) -> 'PersistentHashMap':
        t = TransientHashMap(PersistentHashMap())
        for k, v in self.items():
            if fn(k, v):
                t.set_mut(k, v)
        return t.persistent()

    def to_dict(self) -> dict:
        return dict(self.items())

    def transient(self) -> 'TransientHashMap':
        return TransientHashMap(self)

    def update(self, key: Any, fn: Callable, default: Any = None) -> 'PersistentHashMap':
        """Apply fn to the current value of key (or default if absent)."""
        old = self.get(key, default)
        return self.set(key, fn(old))


class TransientHashMap:
    """Mutable HAMT for batch operations."""

    def __init__(self, phm: PersistentHashMap):
        self._owner = object()
        self._root = phm._root
        self._count = phm._count
        self._persisted = False

    def _ensure_mutable(self):
        if self._persisted:
            raise RuntimeError("TransientHashMap already persisted")

    def set_mut(self, key: Any, value: Any) -> 'TransientHashMap':
        self._ensure_mutable()
        h = hash(key)
        new_root, added = self._set_node_mut(self._root, h, key, value, 0)
        self._root = new_root
        if added:
            self._count += 1
        return self

    def _set_node_mut(self, node, h: int, key: Any, value: Any, shift: int):
        if isinstance(node, _HEmpty):
            return _HLeaf(h, key, value), True
        if isinstance(node, _HLeaf):
            if node.hash == h:
                if node.key == key:
                    return _HLeaf(h, key, value), False
                return _HCollision(h, [(node.key, node.value), (key, value)]), True
            # Different hash
            phm = PersistentHashMap()
            return phm._make_branch(node, _HLeaf(h, key, value), shift), True
        if isinstance(node, _HCollision):
            if node.hash == h:
                for i, (k, v) in enumerate(node.entries):
                    if k == key:
                        new_entries = list(node.entries)
                        new_entries[i] = (key, value)
                        return _HCollision(h, new_entries), False
                return _HCollision(h, node.entries + [(key, value)]), True
            phm = PersistentHashMap()
            return phm._make_branch(node, _HLeaf(h, key, value), shift), True
        # _HBranch
        node = node.ensure_editable(self._owner) if isinstance(node, _HBranch) else node
        bit = 1 << ((h >> shift) & MASK)
        idx = _hamt_index(node.bitmap, bit)
        if node.bitmap & bit:
            child = node.children[idx]
            new_child, added = self._set_node_mut(child, h, key, value, shift + BITS)
            node.children[idx] = new_child
            return node, added
        else:
            node.children.insert(idx, _HLeaf(h, key, value))
            node.bitmap |= bit
            return node, True

    def delete_mut(self, key: Any) -> 'TransientHashMap':
        self._ensure_mutable()
        sentinel = object()
        if PersistentHashMap(self._root, self._count).get(key, sentinel) is sentinel:
            return self
        h = hash(key)
        phm = PersistentHashMap(self._root, self._count)
        new_root = phm._delete_node(self._root, h, key, 0)
        if new_root is None:
            new_root = _HEMPTY
        self._root = new_root
        self._count -= 1
        return self

    def persistent(self) -> PersistentHashMap:
        self._persisted = True
        self._owner = None
        return PersistentHashMap(self._root, self._count)


# ============================================================
# PersistentSortedSet -- Red-Black Tree
# ============================================================

_RED = True
_BLACK = False


class _RBNode:
    """Red-black tree node."""
    __slots__ = ('key', 'left', 'right', 'color')

    def __init__(self, key: Any, left=None, right=None, color=_RED):
        self.key = key
        self.left = left
        self.right = right
        self.color = color


_RBNIL = _RBNode(None, color=_BLACK)
_RBNIL.left = _RBNIL
_RBNIL.right = _RBNIL


def _rb_is_red(node: _RBNode) -> bool:
    return node is not _RBNIL and node.color == _RED


def _rb_copy(node: _RBNode, **kwargs) -> _RBNode:
    """Create modified copy of node (persistent -- never mutate)."""
    return _RBNode(
        kwargs.get('key', node.key),
        kwargs.get('left', node.left),
        kwargs.get('right', node.right),
        kwargs.get('color', node.color),
    )


def _rb_rotate_left(node: _RBNode) -> _RBNode:
    x = node.right
    return _RBNode(x.key, _RBNode(node.key, node.left, x.left, _RED), x.right, node.color)


def _rb_rotate_right(node: _RBNode) -> _RBNode:
    x = node.left
    return _RBNode(x.key, x.left, _RBNode(node.key, x.right, node.right, _RED), node.color)


def _rb_flip_colors(node: _RBNode) -> _RBNode:
    return _RBNode(
        node.key,
        _rb_copy(node.left, color=not node.left.color),
        _rb_copy(node.right, color=not node.right.color),
        not node.color,
    )


def _rb_balance(node: _RBNode) -> _RBNode:
    if _rb_is_red(node.right) and not _rb_is_red(node.left):
        node = _rb_rotate_left(node)
    if _rb_is_red(node.left) and _rb_is_red(node.left.left):
        node = _rb_rotate_right(node)
    if _rb_is_red(node.left) and _rb_is_red(node.right):
        node = _rb_flip_colors(node)
    return node


def _rb_insert(node: _RBNode, key: Any, comparator: Callable) -> tuple:
    """Returns (new_node, was_added)."""
    if node is _RBNIL:
        return _RBNode(key, _RBNIL, _RBNIL, _RED), True

    cmp = comparator(key, node.key)
    if cmp < 0:
        new_left, added = _rb_insert(node.left, key, comparator)
        node = _rb_copy(node, left=new_left)
    elif cmp > 0:
        new_right, added = _rb_insert(node.right, key, comparator)
        node = _rb_copy(node, right=new_right)
    else:
        # Key exists -- replace (set semantics: no change)
        return node, False

    return _rb_balance(node), added


def _rb_move_red_left(node: _RBNode) -> _RBNode:
    node = _rb_flip_colors(node)
    if _rb_is_red(node.right.left):
        node = _rb_copy(node, right=_rb_rotate_right(node.right))
        node = _rb_rotate_left(node)
        node = _rb_flip_colors(node)
    return node


def _rb_move_red_right(node: _RBNode) -> _RBNode:
    node = _rb_flip_colors(node)
    if _rb_is_red(node.left.left):
        node = _rb_rotate_right(node)
        node = _rb_flip_colors(node)
    return node


def _rb_min(node: _RBNode) -> Any:
    while node.left is not _RBNIL:
        node = node.left
    return node.key


def _rb_delete_min(node: _RBNode) -> _RBNode:
    if node.left is _RBNIL:
        return _RBNIL
    if not _rb_is_red(node.left) and not _rb_is_red(node.left.left):
        node = _rb_move_red_left(node)
    node = _rb_copy(node, left=_rb_delete_min(node.left))
    return _rb_balance(node)


def _rb_delete(node: _RBNode, key: Any, comparator: Callable) -> tuple:
    """Returns (new_node, was_deleted)."""
    if node is _RBNIL:
        return _RBNIL, False

    cmp = comparator(key, node.key)
    if cmp < 0:
        if node.left is _RBNIL:
            return node, False
        if not _rb_is_red(node.left) and not _rb_is_red(node.left.left):
            node = _rb_move_red_left(node)
        new_left, deleted = _rb_delete(node.left, key, comparator)
        node = _rb_copy(node, left=new_left)
        return _rb_balance(node), deleted
    else:
        if _rb_is_red(node.left):
            node = _rb_rotate_right(node)
            # Recalculate cmp after rotation
            cmp = comparator(key, node.key)

        if cmp == 0 and node.right is _RBNIL:
            return _RBNIL, True

        if node.right is not _RBNIL and not _rb_is_red(node.right) and not _rb_is_red(node.right.left):
            node = _rb_move_red_right(node)
            cmp = comparator(key, node.key)

        if cmp == 0:
            successor = _rb_min(node.right)
            node = _rb_copy(node, key=successor, right=_rb_delete_min(node.right))
            return _rb_balance(node), True
        else:
            new_right, deleted = _rb_delete(node.right, key, comparator)
            node = _rb_copy(node, right=new_right)
            return _rb_balance(node), deleted


def _default_comparator(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


class PersistentSortedSet:
    """
    Immutable sorted set using a persistent left-leaning red-black tree.
    O(log N) add/remove/contains with structural sharing.
    """
    __slots__ = ('_root', '_count', '_comparator')

    def __init__(self, root=_RBNIL, count: int = 0, comparator: Callable = None):
        self._root = root
        self._count = count
        self._comparator = comparator or _default_comparator

    @staticmethod
    def empty(comparator: Callable = None) -> 'PersistentSortedSet':
        return PersistentSortedSet(comparator=comparator)

    @staticmethod
    def of(*args, comparator: Callable = None) -> 'PersistentSortedSet':
        s = PersistentSortedSet(comparator=comparator)
        for item in args:
            s = s.add(item)
        return s

    @staticmethod
    def from_iterable(iterable, comparator: Callable = None) -> 'PersistentSortedSet':
        s = PersistentSortedSet(comparator=comparator)
        for item in iterable:
            s = s.add(item)
        return s

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def add(self, key: Any) -> 'PersistentSortedSet':
        new_root, added = _rb_insert(self._root, key, self._comparator)
        new_root = _rb_copy(new_root, color=_BLACK)
        new_count = self._count + (1 if added else 0)
        return PersistentSortedSet(new_root, new_count, self._comparator)

    def remove(self, key: Any) -> 'PersistentSortedSet':
        if key not in self:
            return self
        new_root, deleted = _rb_delete(self._root, key, self._comparator)
        if new_root is not _RBNIL:
            new_root = _rb_copy(new_root, color=_BLACK)
        new_count = self._count - (1 if deleted else 0)
        return PersistentSortedSet(new_root, new_count, self._comparator)

    def __contains__(self, key: Any) -> bool:
        node = self._root
        while node is not _RBNIL:
            cmp = self._comparator(key, node.key)
            if cmp < 0:
                node = node.left
            elif cmp > 0:
                node = node.right
            else:
                return True
        return False

    def min(self) -> Any:
        if self._root is _RBNIL:
            raise ValueError("Empty set has no minimum")
        return _rb_min(self._root)

    def max(self) -> Any:
        if self._root is _RBNIL:
            raise ValueError("Empty set has no maximum")
        node = self._root
        while node.right is not _RBNIL:
            node = node.right
        return node.key

    def _inorder(self, node: _RBNode):
        if node is _RBNIL:
            return
        yield from self._inorder(node.left)
        yield node.key
        yield from self._inorder(node.right)

    def __iter__(self) -> Iterator:
        return self._inorder(self._root)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PersistentSortedSet):
            return NotImplemented
        if self._count != other._count:
            return False
        return all(a == b for a, b in zip(self, other))

    def __hash__(self) -> int:
        h = 0
        for item in self:
            h = h * 31 + hash(item)
        return h

    def __repr__(self) -> str:
        items = ', '.join(repr(x) for x in self)
        return f"PSortedSet([{items}])"

    def range_query(self, lo: Any, hi: Any) -> Iterator:
        """Yield all keys where lo <= key <= hi."""
        yield from self._range(self._root, lo, hi)

    def _range(self, node: _RBNode, lo: Any, hi: Any):
        if node is _RBNIL:
            return
        cmp_lo = self._comparator(lo, node.key)
        cmp_hi = self._comparator(hi, node.key)
        if cmp_lo < 0:
            yield from self._range(node.left, lo, hi)
        if cmp_lo <= 0 and cmp_hi >= 0:
            yield node.key
        if cmp_hi > 0:
            yield from self._range(node.right, lo, hi)

    def union(self, other: 'PersistentSortedSet') -> 'PersistentSortedSet':
        result = self
        for key in other:
            result = result.add(key)
        return result

    def intersection(self, other: 'PersistentSortedSet') -> 'PersistentSortedSet':
        result = PersistentSortedSet(comparator=self._comparator)
        for key in self:
            if key in other:
                result = result.add(key)
        return result

    def difference(self, other: 'PersistentSortedSet') -> 'PersistentSortedSet':
        result = self
        for key in other:
            result = result.remove(key)
        return result

    def to_list(self) -> list:
        return list(self)

    def nth(self, n: int) -> Any:
        """Get nth element in sorted order."""
        for i, key in enumerate(self):
            if i == n:
                return key
        raise IndexError(f"Index {n} out of range")
