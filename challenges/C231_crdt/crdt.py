"""
C231: Conflict-free Replicated Data Types (CRDTs)

State-based (CvRDT) implementations of common CRDTs for eventually consistent
distributed systems. Each type supports merge() for convergence without coordination.

Types implemented:
  Counters: GCounter, PNCounter
  Registers: LWWRegister, MVRegister
  Sets: GSet, TwoPSet, ORSet, LWWElementSet
  Sequences: RGASequence (Replicated Growable Array)
  Maps: ORMap (Observed-Remove Map)
  Flags: EWFlag (Enable-Wins Flag), DWFlag (Disable-Wins Flag)

All CRDTs satisfy:
  - Commutativity: merge(a, b) == merge(b, a)
  - Associativity: merge(merge(a, b), c) == merge(a, merge(b, c))
  - Idempotency: merge(a, a) == a
"""

import time
import uuid
from collections import defaultdict


# =============================================================================
# Counters
# =============================================================================

class GCounter:
    """Grow-only counter. Each replica increments its own slot."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.counts = defaultdict(int)

    def increment(self, amount=1):
        if amount < 0:
            raise ValueError("GCounter only supports non-negative increments")
        self.counts[self.replica_id] += amount

    @property
    def value(self):
        return sum(self.counts.values())

    def merge(self, other):
        result = GCounter(self.replica_id)
        all_keys = set(self.counts) | set(other.counts)
        for k in all_keys:
            result.counts[k] = max(self.counts[k], other.counts[k])
        return result

    def __eq__(self, other):
        if not isinstance(other, GCounter):
            return NotImplemented
        return dict(self.counts) == dict(other.counts)

    def __repr__(self):
        return f"GCounter({self.value})"


class PNCounter:
    """Positive-Negative counter. Two GCounters: one for increments, one for decrements."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.p = GCounter(replica_id)  # positive
        self.n = GCounter(replica_id)  # negative

    def increment(self, amount=1):
        self.p.increment(amount)

    def decrement(self, amount=1):
        self.n.increment(amount)

    @property
    def value(self):
        return self.p.value - self.n.value

    def merge(self, other):
        result = PNCounter(self.replica_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def __eq__(self, other):
        if not isinstance(other, PNCounter):
            return NotImplemented
        return self.p == other.p and self.n == other.n

    def __repr__(self):
        return f"PNCounter({self.value})"


# =============================================================================
# Registers
# =============================================================================

class LWWRegister:
    """Last-Writer-Wins Register. Timestamp breaks ties."""

    def __init__(self, replica_id, value=None, timestamp=0):
        self.replica_id = replica_id
        self._value = value
        self._timestamp = timestamp

    def set(self, value, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        if timestamp >= self._timestamp:
            self._value = value
            self._timestamp = timestamp

    @property
    def value(self):
        return self._value

    @property
    def timestamp(self):
        return self._timestamp

    def merge(self, other):
        if other._timestamp > self._timestamp:
            result = LWWRegister(self.replica_id, other._value, other._timestamp)
        elif other._timestamp == self._timestamp:
            # Tie-break by replica_id (deterministic)
            if str(other.replica_id) > str(self.replica_id):
                result = LWWRegister(self.replica_id, other._value, other._timestamp)
            else:
                result = LWWRegister(self.replica_id, self._value, self._timestamp)
        else:
            result = LWWRegister(self.replica_id, self._value, self._timestamp)
        return result

    def __eq__(self, other):
        if not isinstance(other, LWWRegister):
            return NotImplemented
        return self._value == other._value and self._timestamp == other._timestamp

    def __repr__(self):
        return f"LWWRegister({self._value!r})"


class MVRegister:
    """Multi-Value Register. Concurrent writes produce multiple values (like Amazon's shopping cart).
    Uses vector clocks to track causality."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        # Each entry: (value, vector_clock_dict)
        self._entries = []
        self._clock = defaultdict(int)

    def set(self, value):
        self._clock[self.replica_id] += 1
        # New write dominates all current entries
        self._entries = [(value, dict(self._clock))]

    @property
    def values(self):
        return [v for v, _ in self._entries]

    @property
    def value(self):
        """Returns single value if unique, list if concurrent."""
        vals = self.values
        if len(vals) == 1:
            return vals[0]
        return vals

    def merge(self, other):
        result = MVRegister(self.replica_id)
        # Merge clocks
        all_keys = set(self._clock) | set(other._clock)
        for k in all_keys:
            result._clock[k] = max(self._clock.get(k, 0), other._clock.get(k, 0))

        # Keep entries not dominated by the other's clock
        all_entries = []
        for val, vc in self._entries:
            if not _vc_dominates(other._clock, vc):
                all_entries.append((val, vc))
        for val, vc in other._entries:
            if not _vc_dominates(self._clock, vc):
                all_entries.append((val, vc))

        # Deduplicate
        seen = set()
        unique = []
        for val, vc in all_entries:
            key = (val, tuple(sorted(vc.items())))
            if key not in seen:
                seen.add(key)
                unique.append((val, vc))

        result._entries = unique if unique else list(self._entries) + list(other._entries)
        # Deduplicate again for the fallback case
        if not unique:
            seen2 = set()
            deduped = []
            for val, vc in result._entries:
                key = (val, tuple(sorted(vc.items())))
                if key not in seen2:
                    seen2.add(key)
                    deduped.append((val, vc))
            result._entries = deduped

        return result

    def __repr__(self):
        return f"MVRegister({self.values})"


def _vc_dominates(vc1, vc2):
    """Returns True if vc1 strictly dominates vc2 (vc1 >= vc2 on all, > on at least one)."""
    all_keys = set(vc1) | set(vc2)
    ge = all(vc1.get(k, 0) >= vc2.get(k, 0) for k in all_keys)
    gt = any(vc1.get(k, 0) > vc2.get(k, 0) for k in all_keys)
    return ge and gt


# =============================================================================
# Sets
# =============================================================================

class GSet:
    """Grow-only set. Elements can be added but never removed."""

    def __init__(self, replica_id=None):
        self.replica_id = replica_id
        self._elements = set()

    def add(self, element):
        self._elements.add(element)

    def contains(self, element):
        return element in self._elements

    @property
    def elements(self):
        return frozenset(self._elements)

    def __len__(self):
        return len(self._elements)

    def merge(self, other):
        result = GSet(self.replica_id)
        result._elements = self._elements | other._elements
        return result

    def __eq__(self, other):
        if not isinstance(other, GSet):
            return NotImplemented
        return self._elements == other._elements

    def __repr__(self):
        return f"GSet({self._elements})"


class TwoPSet:
    """Two-Phase Set. Elements can be added and removed, but removal is permanent."""

    def __init__(self, replica_id=None):
        self.replica_id = replica_id
        self._added = set()
        self._removed = set()  # tombstone set

    def add(self, element):
        self._added.add(element)

    def remove(self, element):
        if element in self._added:
            self._removed.add(element)

    def contains(self, element):
        return element in self._added and element not in self._removed

    @property
    def elements(self):
        return frozenset(self._added - self._removed)

    def __len__(self):
        return len(self._added - self._removed)

    def merge(self, other):
        result = TwoPSet(self.replica_id)
        result._added = self._added | other._added
        result._removed = self._removed | other._removed
        return result

    def __eq__(self, other):
        if not isinstance(other, TwoPSet):
            return NotImplemented
        return self._added == other._added and self._removed == other._removed

    def __repr__(self):
        return f"TwoPSet({self.elements})"


class ORSet:
    """Observed-Remove Set. Supports add-remove-add cycles using unique tags."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        # element -> set of (unique_tag) for active entries
        self._entries = defaultdict(set)  # element -> {tag, ...}
        self._tombstones = defaultdict(set)  # element -> {tag, ...}

    def add(self, element):
        tag = str(uuid.uuid4())
        self._entries[element].add(tag)
        return tag

    def remove(self, element):
        if element in self._entries:
            # Tombstone all currently observed tags
            tags = self._entries[element].copy()
            self._tombstones[element] |= tags
            self._entries[element] -= tags

    def contains(self, element):
        live = self._entries.get(element, set()) - self._tombstones.get(element, set())
        return len(live) > 0

    @property
    def elements(self):
        result = set()
        for elem, tags in self._entries.items():
            live = tags - self._tombstones.get(elem, set())
            if live:
                result.add(elem)
        return frozenset(result)

    def __len__(self):
        return len(self.elements)

    def merge(self, other):
        result = ORSet(self.replica_id)
        all_elems = set(self._entries) | set(other._entries)
        for elem in all_elems:
            result._entries[elem] = (self._entries.get(elem, set()) |
                                     other._entries.get(elem, set()))
            result._tombstones[elem] = (self._tombstones.get(elem, set()) |
                                        other._tombstones.get(elem, set()))
        return result

    def __repr__(self):
        return f"ORSet({self.elements})"


class LWWElementSet:
    """LWW-Element-Set. Each element has add/remove timestamps; latest wins."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self._adds = {}      # element -> timestamp
        self._removes = {}   # element -> timestamp

    def add(self, element, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        if element not in self._adds or timestamp > self._adds[element]:
            self._adds[element] = timestamp

    def remove(self, element, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        if element not in self._removes or timestamp > self._removes[element]:
            self._removes[element] = timestamp

    def contains(self, element):
        if element not in self._adds:
            return False
        add_ts = self._adds[element]
        rem_ts = self._removes.get(element, -1)
        return add_ts >= rem_ts  # bias toward add on tie

    @property
    def elements(self):
        result = set()
        for elem in self._adds:
            if self.contains(elem):
                result.add(elem)
        return frozenset(result)

    def __len__(self):
        return len(self.elements)

    def merge(self, other):
        result = LWWElementSet(self.replica_id)
        for elem in set(self._adds) | set(other._adds):
            ts1 = self._adds.get(elem, -1)
            ts2 = other._adds.get(elem, -1)
            result._adds[elem] = max(ts1, ts2)
        for elem in set(self._removes) | set(other._removes):
            ts1 = self._removes.get(elem, -1)
            ts2 = other._removes.get(elem, -1)
            result._removes[elem] = max(ts1, ts2)
        return result

    def __repr__(self):
        return f"LWWElementSet({self.elements})"


# =============================================================================
# Flags
# =============================================================================

class EWFlag:
    """Enable-Wins Flag. Concurrent enable+disable resolves to enabled."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self._enables = set()   # unique tags
        self._disables = set()  # unique tags

    def enable(self):
        tag = str(uuid.uuid4())
        self._enables.add(tag)
        # Clear observed disables
        self._disables.clear()

    def disable(self):
        # Move all enables to disables
        self._disables |= self._enables
        self._enables.clear()

    @property
    def value(self):
        live = self._enables - self._disables
        return len(live) > 0

    def merge(self, other):
        result = EWFlag(self.replica_id)
        result._enables = self._enables | other._enables
        result._disables = self._disables | other._disables
        return result

    def __repr__(self):
        return f"EWFlag({self.value})"


class DWFlag:
    """Disable-Wins Flag. Concurrent enable+disable resolves to disabled."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self._enables = set()
        self._disables = set()

    def enable(self):
        # Move all disables to enables
        self._enables |= self._disables
        self._disables.clear()
        tag = str(uuid.uuid4())
        self._enables.add(tag)

    def disable(self):
        tag = str(uuid.uuid4())
        self._disables.add(tag)
        # Clear observed enables
        self._enables.clear()

    @property
    def value(self):
        live_disables = self._disables - self._enables
        if live_disables:
            return False
        return len(self._enables) > 0

    def merge(self, other):
        result = DWFlag(self.replica_id)
        result._enables = self._enables | other._enables
        result._disables = self._disables | other._disables
        return result

    def __repr__(self):
        return f"DWFlag({self.value})"


# =============================================================================
# Sequences
# =============================================================================

class RGASequence:
    """Replicated Growable Array (RGA). Ordered sequence CRDT for collaborative editing.

    Each element has a unique ID (timestamp, replica_id) and a reference to the
    element it was inserted after. Tombstones mark deleted elements.
    """

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self._counter = 0
        # Each node: {id: (counter, replica), value: ..., after: id|None, deleted: bool}
        self._nodes = {}  # id -> node dict
        self._root_id = None  # virtual root

    def _next_id(self):
        self._counter += 1
        return (self._counter, self.replica_id)

    def _sorted_children(self, parent_id):
        """Get children of parent_id, sorted by ID descending (newest first)."""
        children = []
        for nid, node in self._nodes.items():
            if node['after'] == parent_id:
                children.append(nid)
        # Sort descending by (counter, replica) so newest inserts appear first
        children.sort(reverse=True)
        return children

    def _linearize(self):
        """Convert tree to linear sequence via DFS."""
        result = []
        stack = list(reversed(self._sorted_children(self._root_id)))
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            if not node['deleted']:
                result.append((nid, node['value']))
            # Push children in reverse order so first child is processed first
            children = self._sorted_children(nid)
            for child_id in reversed(children):
                stack.append(child_id)
        return result

    def insert(self, index, value):
        """Insert value at position index."""
        seq = self._linearize()
        if index == 0:
            after_id = self._root_id
        elif index <= len(seq):
            after_id = seq[index - 1][0]
        else:
            raise IndexError(f"Index {index} out of range (len={len(seq)})")

        nid = self._next_id()
        self._nodes[nid] = {
            'id': nid,
            'value': value,
            'after': after_id,
            'deleted': False,
        }
        return nid

    def append(self, value):
        """Append value to end."""
        seq = self._linearize()
        return self.insert(len(seq), value)

    def delete(self, index):
        """Mark element at index as deleted (tombstone)."""
        seq = self._linearize()
        if index < 0 or index >= len(seq):
            raise IndexError(f"Index {index} out of range (len={len(seq)})")
        nid = seq[index][0]
        self._nodes[nid]['deleted'] = True

    @property
    def elements(self):
        return [v for _, v in self._linearize()]

    def __len__(self):
        return len(self._linearize())

    def __getitem__(self, index):
        seq = self._linearize()
        return seq[index][1]

    def merge(self, other):
        result = RGASequence(self.replica_id)
        result._counter = max(self._counter, other._counter)
        # Merge all nodes
        all_ids = set(self._nodes) | set(other._nodes)
        for nid in all_ids:
            n1 = self._nodes.get(nid)
            n2 = other._nodes.get(nid)
            if n1 and n2:
                # Both have it -- deleted if either deleted
                result._nodes[nid] = dict(n1)
                result._nodes[nid]['deleted'] = n1['deleted'] or n2['deleted']
            elif n1:
                result._nodes[nid] = dict(n1)
            else:
                result._nodes[nid] = dict(n2)
        return result

    def __repr__(self):
        return f"RGASequence({self.elements})"


# =============================================================================
# Maps
# =============================================================================

class ORMap:
    """Observed-Remove Map. Keys map to nested CRDT values.
    Supports add, remove, and nested CRDT merging."""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self._keys = ORSet(replica_id)  # track key presence
        self._values = {}  # key -> crdt_value

    def put(self, key, crdt_value):
        """Associate key with a CRDT value. If key exists, merges."""
        self._keys.add(key)
        if key in self._values and hasattr(self._values[key], 'merge'):
            self._values[key] = self._values[key].merge(crdt_value)
        else:
            self._values[key] = crdt_value

    def get(self, key):
        if self._keys.contains(key) and key in self._values:
            return self._values[key]
        return None

    def remove(self, key):
        self._keys.remove(key)

    def contains(self, key):
        return self._keys.contains(key)

    @property
    def keys(self):
        return self._keys.elements

    def merge(self, other):
        result = ORMap(self.replica_id)
        result._keys = self._keys.merge(other._keys)
        # Merge values for all live keys
        all_val_keys = set(self._values) | set(other._values)
        for key in all_val_keys:
            v1 = self._values.get(key)
            v2 = other._values.get(key)
            if v1 and v2 and hasattr(v1, 'merge'):
                result._values[key] = v1.merge(v2)
            elif v1:
                result._values[key] = v1
            elif v2:
                result._values[key] = v2
        return result

    def __repr__(self):
        live = {k: self._values.get(k) for k in self.keys if k in self._values}
        return f"ORMap({live})"


# =============================================================================
# Vector Clock utility
# =============================================================================

class VectorClock:
    """Vector clock for causal ordering."""

    def __init__(self, replica_id=None):
        self.replica_id = replica_id
        self._clock = defaultdict(int)

    def increment(self, replica_id=None):
        rid = replica_id or self.replica_id
        self._clock[rid] += 1

    def get(self, replica_id):
        return self._clock.get(replica_id, 0)

    def set(self, replica_id, value):
        self._clock[replica_id] = value

    def merge(self, other):
        result = VectorClock(self.replica_id)
        for k in set(self._clock) | set(other._clock):
            result._clock[k] = max(self._clock.get(k, 0), other._clock.get(k, 0))
        return result

    def dominates(self, other):
        """True if self >= other on all components and > on at least one."""
        return _vc_dominates(dict(self._clock), dict(other._clock))

    def concurrent(self, other):
        """True if neither dominates the other."""
        return not self.dominates(other) and not other.dominates(self)

    def __eq__(self, other):
        if not isinstance(other, VectorClock):
            return NotImplemented
        return dict(self._clock) == dict(other._clock)

    def __le__(self, other):
        for k in set(self._clock) | set(other._clock):
            if self._clock.get(k, 0) > other._clock.get(k, 0):
                return False
        return True

    def __repr__(self):
        return f"VectorClock({dict(self._clock)})"


# =============================================================================
# Convergence testing utility
# =============================================================================

class CRDTNetwork:
    """Simulates a network of CRDT replicas for testing convergence."""

    def __init__(self):
        self._replicas = {}  # name -> crdt

    def add_replica(self, name, crdt):
        self._replicas[name] = crdt

    def get(self, name):
        return self._replicas[name]

    def sync(self, src, dst):
        """Sync src -> dst (dst merges src's state)."""
        merged = self._replicas[dst].merge(self._replicas[src])
        self._replicas[dst] = merged

    def sync_all(self):
        """Full mesh sync -- all replicas converge."""
        names = list(self._replicas.keys())
        # Multiple rounds to ensure full convergence
        for _ in range(len(names)):
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j:
                        self.sync(names[i], names[j])

    def converged(self):
        """Check if all replicas have same state."""
        names = list(self._replicas.keys())
        if len(names) < 2:
            return True
        ref = self._replicas[names[0]]
        for name in names[1:]:
            other = self._replicas[name]
            # Compare elements/value depending on type
            if hasattr(ref, 'elements'):
                if ref.elements != other.elements:
                    return False
            elif hasattr(ref, 'value'):
                if ref.value != other.value:
                    return False
        return True
