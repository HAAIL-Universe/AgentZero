"""
C202: Conflict-free Replicated Data Types (CRDTs)

A comprehensive CRDT library implementing both state-based (CvRDT) and
operation-based (CmRDT) variants for distributed systems without consensus.

State-based CRDTs (convergent):
  - GCounter: Grow-only counter
  - PNCounter: Positive-negative counter (increment and decrement)
  - GSet: Grow-only set
  - TwoPSet: Two-phase set (add and remove, remove is permanent)
  - ORSet: Observed-Remove set (add/remove with unique tags)
  - LWWRegister: Last-Writer-Wins register (timestamp-based)
  - MVRegister: Multi-Value register (concurrent writes preserved)
  - LWWElementSet: LWW-Element set (timestamps per element)
  - RGA: Replicated Growable Array (ordered sequence)

Operation-based CRDTs (commutative):
  - OpCounter: Op-based counter
  - OpSet: Op-based set with add/remove

Delta-state CRDTs:
  - DeltaGCounter: Delta-mutator GCounter (ship only changes)
  - DeltaPNCounter: Delta-mutator PNCounter

Composition:
  - CRDTMap: Map of CRDTs (nested CRDT values)
  - CausalContext: Vector clock for causal consistency

Network simulation:
  - CRDTNetwork: Simulated network with partitions, delays, message loss
"""

import copy
import time
import uuid
import random
from collections import defaultdict


# =============================================================================
# Vector Clocks / Causal Context
# =============================================================================

class VectorClock:
    """Vector clock for causal ordering."""

    def __init__(self, clock=None):
        self.clock = dict(clock) if clock else {}

    def increment(self, node_id):
        self.clock[node_id] = self.clock.get(node_id, 0) + 1
        return self

    def get(self, node_id):
        return self.clock.get(node_id, 0)

    def merge(self, other):
        result = VectorClock(self.clock)
        for node_id, count in other.clock.items():
            result.clock[node_id] = max(result.clock.get(node_id, 0), count)
        return result

    def __le__(self, other):
        """self <= other (self happened-before or concurrent with other)"""
        for node_id, count in self.clock.items():
            if count > other.clock.get(node_id, 0):
                return False
        return True

    def __lt__(self, other):
        """self < other (self strictly happened-before other)"""
        return self <= other and not other <= self

    def __eq__(self, other):
        if not isinstance(other, VectorClock):
            return False
        all_keys = set(self.clock.keys()) | set(other.clock.keys())
        return all(self.clock.get(k, 0) == other.clock.get(k, 0) for k in all_keys)

    def __hash__(self):
        return hash(tuple(sorted(self.clock.items())))

    def concurrent(self, other):
        """True if neither happened-before the other."""
        return not (self <= other) and not (other <= self)

    def copy(self):
        return VectorClock(self.clock)

    def __repr__(self):
        return f"VC({self.clock})"


class CausalContext:
    """Causal context using vector clock + dot set for delta CRDTs."""

    def __init__(self):
        self.cc = {}  # compact causal context: node -> max seq
        self.dots = set()  # dot set: (node, seq) pairs

    def next_dot(self, node_id):
        seq = self.cc.get(node_id, 0) + 1
        self.cc[node_id] = seq
        dot = (node_id, seq)
        self.dots.add(dot)
        return dot

    def has_dot(self, dot):
        node_id, seq = dot
        if seq <= self.cc.get(node_id, 0):
            return True
        return dot in self.dots

    def compact(self):
        """Compact dot set into cc where possible."""
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for dot in self.dots:
                node_id, seq = dot
                if seq == self.cc.get(node_id, 0) + 1:
                    self.cc[node_id] = seq
                    to_remove.add(dot)
                    changed = True
            self.dots -= to_remove

    def merge(self, other):
        result = CausalContext()
        for node_id in set(self.cc) | set(other.cc):
            result.cc[node_id] = max(self.cc.get(node_id, 0), other.cc.get(node_id, 0))
        result.dots = self.dots | other.dots
        result.compact()
        return result

    def copy(self):
        ctx = CausalContext()
        ctx.cc = dict(self.cc)
        ctx.dots = set(self.dots)
        return ctx


# =============================================================================
# State-based CRDTs (CvRDTs)
# =============================================================================

class GCounter:
    """Grow-only counter. Each node has its own counter; value is the sum."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.counts = {}

    def increment(self, amount=1):
        if amount < 0:
            raise ValueError("GCounter only supports non-negative increments")
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount

    @property
    def value(self):
        return sum(self.counts.values())

    def merge(self, other):
        """Merge with another GCounter (LUB in the semilattice)."""
        result = GCounter(self.node_id)
        for node_id in set(self.counts) | set(other.counts):
            result.counts[node_id] = max(
                self.counts.get(node_id, 0),
                other.counts.get(node_id, 0)
            )
        return result

    def state(self):
        return dict(self.counts)

    def __repr__(self):
        return f"GCounter({self.value}, counts={self.counts})"


class PNCounter:
    """Positive-Negative counter. Two GCounters: one for increments, one for decrements."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.p = GCounter(self.node_id)  # positive
        self.n = GCounter(self.node_id)  # negative

    def increment(self, amount=1):
        self.p.increment(amount)

    def decrement(self, amount=1):
        self.n.increment(amount)

    @property
    def value(self):
        return self.p.value - self.n.value

    def merge(self, other):
        result = PNCounter(self.node_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def __repr__(self):
        return f"PNCounter({self.value})"


class GSet:
    """Grow-only set. Elements can be added but never removed."""

    def __init__(self):
        self.elements = set()

    def add(self, element):
        self.elements.add(element)

    def lookup(self, element):
        return element in self.elements

    @property
    def value(self):
        return frozenset(self.elements)

    def merge(self, other):
        result = GSet()
        result.elements = self.elements | other.elements
        return result

    def __repr__(self):
        return f"GSet({self.elements})"


class TwoPSet:
    """Two-Phase Set. Elements can be added and removed, but removal is permanent."""

    def __init__(self):
        self.add_set = set()
        self.remove_set = set()

    def add(self, element):
        self.add_set.add(element)

    def remove(self, element):
        if element in self.add_set:
            self.remove_set.add(element)

    def lookup(self, element):
        return element in self.add_set and element not in self.remove_set

    @property
    def value(self):
        return frozenset(self.add_set - self.remove_set)

    def merge(self, other):
        result = TwoPSet()
        result.add_set = self.add_set | other.add_set
        result.remove_set = self.remove_set | other.remove_set
        return result

    def __repr__(self):
        return f"TwoPSet({self.value})"


class ORSet:
    """Observed-Remove Set. Add/remove with unique tags to handle concurrent ops."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.elements = {}  # element -> set of unique tags
        self.tombstones = set()  # removed tags

    def add(self, element):
        tag = f"{self.node_id}:{uuid.uuid4().hex[:8]}"
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)

    def remove(self, element):
        if element in self.elements:
            self.tombstones |= self.elements[element]
            del self.elements[element]

    def lookup(self, element):
        if element not in self.elements:
            return False
        live_tags = self.elements[element] - self.tombstones
        return len(live_tags) > 0

    @property
    def value(self):
        result = set()
        for elem, tags in self.elements.items():
            if tags - self.tombstones:
                result.add(elem)
        return frozenset(result)

    def merge(self, other):
        result = ORSet(self.node_id)
        result.tombstones = self.tombstones | other.tombstones
        all_elements = set(self.elements) | set(other.elements)
        for elem in all_elements:
            tags_self = self.elements.get(elem, set())
            tags_other = other.elements.get(elem, set())
            merged_tags = (tags_self | tags_other) - result.tombstones
            if merged_tags:
                result.elements[elem] = merged_tags
        return result

    def __repr__(self):
        return f"ORSet({self.value})"


class LWWRegister:
    """Last-Writer-Wins Register. Concurrent writes resolved by timestamp."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self._value = None
        self._timestamp = 0
        self._node_id = self.node_id  # for tiebreaking

    def set(self, value, timestamp=None):
        ts = timestamp if timestamp is not None else time.monotonic()
        if ts > self._timestamp or (ts == self._timestamp and self.node_id > self._node_id):
            self._value = value
            self._timestamp = ts
            self._node_id = self.node_id

    @property
    def value(self):
        return self._value

    @property
    def timestamp(self):
        return self._timestamp

    def merge(self, other):
        result = LWWRegister(self.node_id)
        if other._timestamp > self._timestamp or (
            other._timestamp == self._timestamp and other._node_id > self._node_id
        ):
            result._value = other._value
            result._timestamp = other._timestamp
            result._node_id = other._node_id
        else:
            result._value = self._value
            result._timestamp = self._timestamp
            result._node_id = self._node_id
        return result

    def __repr__(self):
        return f"LWWRegister({self._value}@{self._timestamp})"


class MVRegister:
    """Multi-Value Register. Concurrent writes are all preserved (conflict set)."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.values = {}  # VectorClock -> value
        self.clock = VectorClock()

    def set(self, value):
        self.clock = self.clock.increment(self.node_id)
        # New write supersedes all current values
        self.values = {self.clock.copy(): value}

    @property
    def value(self):
        """Returns set of concurrent values."""
        return frozenset(self.values.values())

    def merge(self, other):
        result = MVRegister(self.node_id)
        result.clock = self.clock.merge(other.clock)
        # Keep values that are not dominated by the other's clock
        for vc, val in self.values.items():
            if not vc < other.clock:
                result.values[vc] = val
        for vc, val in other.values.items():
            if not vc < self.clock:
                result.values[vc] = val
        return result

    def __repr__(self):
        return f"MVRegister({self.value})"


class LWWElementSet:
    """LWW-Element Set. Each element has add/remove timestamps."""

    def __init__(self):
        self.add_map = {}  # element -> timestamp
        self.remove_map = {}  # element -> timestamp

    def add(self, element, timestamp=None):
        ts = timestamp if timestamp is not None else time.monotonic()
        if element not in self.add_map or ts > self.add_map[element]:
            self.add_map[element] = ts

    def remove(self, element, timestamp=None):
        ts = timestamp if timestamp is not None else time.monotonic()
        if element not in self.remove_map or ts > self.remove_map[element]:
            self.remove_map[element] = ts

    def lookup(self, element):
        if element not in self.add_map:
            return False
        add_ts = self.add_map[element]
        rem_ts = self.remove_map.get(element, -1)
        return add_ts > rem_ts

    @property
    def value(self):
        return frozenset(e for e in self.add_map if self.lookup(e))

    def merge(self, other):
        result = LWWElementSet()
        for e in set(self.add_map) | set(other.add_map):
            result.add_map[e] = max(self.add_map.get(e, 0), other.add_map.get(e, 0))
        for e in set(self.remove_map) | set(other.remove_map):
            result.remove_map[e] = max(self.remove_map.get(e, 0), other.remove_map.get(e, 0))
        return result

    def __repr__(self):
        return f"LWWElementSet({self.value})"


# =============================================================================
# RGA (Replicated Growable Array) -- ordered sequence CRDT
# =============================================================================

class RGANode:
    """A node in the RGA linked list."""
    def __init__(self, value, timestamp, node_id, deleted=False):
        self.value = value
        self.timestamp = timestamp
        self.node_id = node_id
        self.deleted = deleted  # tombstone

    def identifier(self):
        return (self.timestamp, self.node_id)

    def __repr__(self):
        d = " [DEL]" if self.deleted else ""
        return f"RGANode({self.value}@{self.timestamp}:{self.node_id}{d})"


class RGA:
    """Replicated Growable Array. Ordered sequence with insert/delete."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.clock = 0
        # List of (node, after_id). We maintain a flat list with tombstones.
        self.nodes = []  # list of RGANode
        self._index = {}  # (timestamp, node_id) -> index in nodes

    def _next_timestamp(self):
        self.clock += 1
        return self.clock

    def _find_index(self, identifier):
        """Find index of node with given identifier."""
        if identifier in self._index:
            return self._index[identifier]
        return None

    def _rebuild_index(self):
        self._index = {}
        for i, node in enumerate(self.nodes):
            self._index[node.identifier()] = i

    def insert(self, position, value):
        """Insert value at position (0-indexed among live elements)."""
        ts = self._next_timestamp()
        new_node = RGANode(value, ts, self.node_id)

        if position == 0 and len(self.nodes) == 0:
            self.nodes.append(new_node)
            self._index[new_node.identifier()] = 0
            return new_node.identifier()

        # Find the actual index (accounting for tombstones)
        if position == 0:
            # Insert before everything: find insertion point
            insert_idx = 0
        else:
            live_count = 0
            insert_idx = len(self.nodes)
            for i, n in enumerate(self.nodes):
                if not n.deleted:
                    live_count += 1
                    if live_count == position:
                        insert_idx = i + 1
                        break

        # Insert right after, but before any nodes with smaller timestamps
        # (to maintain causal ordering for concurrent inserts at same position)
        while insert_idx < len(self.nodes):
            existing = self.nodes[insert_idx]
            if (existing.timestamp, existing.node_id) > (ts, self.node_id):
                break
            insert_idx += 1

        self.nodes.insert(insert_idx, new_node)
        self._rebuild_index()
        return new_node.identifier()

    def append(self, value):
        """Append value at end."""
        return self.insert(len(self.value), value)

    def delete(self, position):
        """Delete element at position (0-indexed among live elements)."""
        live_count = 0
        for i, node in enumerate(self.nodes):
            if not node.deleted:
                if live_count == position:
                    node.deleted = True
                    return node.identifier()
                live_count += 1
        raise IndexError(f"Position {position} out of range")

    def insert_remote(self, value, timestamp, node_id, after_id=None):
        """Insert a remote node. after_id is the identifier of the predecessor."""
        new_node = RGANode(value, timestamp, node_id)
        self.clock = max(self.clock, timestamp)

        if after_id is None:
            insert_idx = 0
        else:
            idx = self._find_index(after_id)
            if idx is None:
                # Predecessor not found; append
                insert_idx = len(self.nodes)
            else:
                insert_idx = idx + 1

        # Position among concurrent inserts: higher (timestamp, node_id) goes first
        while insert_idx < len(self.nodes):
            existing = self.nodes[insert_idx]
            if (existing.timestamp, existing.node_id) < (timestamp, node_id):
                break
            insert_idx += 1

        self.nodes.insert(insert_idx, new_node)
        self._rebuild_index()
        return new_node.identifier()

    def delete_remote(self, identifier):
        """Apply a remote delete by marking the node as tombstoned."""
        idx = self._find_index(identifier)
        if idx is not None:
            self.nodes[idx].deleted = True

    @property
    def value(self):
        return [n.value for n in self.nodes if not n.deleted]

    def merge(self, other):
        """Merge with another RGA."""
        result = RGA(self.node_id)
        result.clock = max(self.clock, other.clock)

        # Collect all nodes by identifier
        all_nodes = {}
        for n in self.nodes:
            ident = n.identifier()
            all_nodes[ident] = RGANode(n.value, n.timestamp, n.node_id,
                                       n.deleted)
        for n in other.nodes:
            ident = n.identifier()
            if ident in all_nodes:
                # Union tombstones
                all_nodes[ident].deleted = all_nodes[ident].deleted or n.deleted
            else:
                all_nodes[ident] = RGANode(n.value, n.timestamp, n.node_id,
                                           n.deleted)

        # Sort: higher timestamp first (within same position), stable
        result.nodes = sorted(all_nodes.values(),
                              key=lambda n: (-n.timestamp, n.node_id))
        result._rebuild_index()
        return result

    def __repr__(self):
        return f"RGA({self.value})"


# =============================================================================
# Operation-based CRDTs (CmRDTs)
# =============================================================================

class OpCounter:
    """Operation-based counter. Operations are increment/decrement with deltas."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self._value = 0
        self.ops_log = []  # for replay

    def increment(self, amount=1):
        op = ('inc', amount, self.node_id, len(self.ops_log))
        self.ops_log.append(op)
        self._value += amount
        return op

    def decrement(self, amount=1):
        op = ('dec', amount, self.node_id, len(self.ops_log))
        self.ops_log.append(op)
        self._value -= amount
        return op

    def apply_op(self, op):
        """Apply a remote operation."""
        kind, amount = op[0], op[1]
        if kind == 'inc':
            self._value += amount
        elif kind == 'dec':
            self._value -= amount
        self.ops_log.append(op)

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f"OpCounter({self._value})"


class OpSet:
    """Operation-based set with add/remove. Requires causal delivery."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.elements = set()
        self.clock = VectorClock()
        self.ops_log = []

    def add(self, element):
        self.clock = self.clock.increment(self.node_id)
        op = ('add', element, self.clock.copy())
        self.elements.add(element)
        self.ops_log.append(op)
        return op

    def remove(self, element):
        self.clock = self.clock.increment(self.node_id)
        op = ('remove', element, self.clock.copy())
        self.elements.discard(element)
        self.ops_log.append(op)
        return op

    def apply_op(self, op):
        kind, element, vc = op
        if kind == 'add':
            self.elements.add(element)
        elif kind == 'remove':
            self.elements.discard(element)
        self.clock = self.clock.merge(vc)
        self.ops_log.append(op)

    @property
    def value(self):
        return frozenset(self.elements)

    def __repr__(self):
        return f"OpSet({self.elements})"


# =============================================================================
# Delta-state CRDTs
# =============================================================================

class DeltaGCounter:
    """Delta-state GCounter. Ships only the delta (changed entries)."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.counts = {}

    def increment(self, amount=1):
        if amount < 0:
            raise ValueError("GCounter only supports non-negative increments")
        old = self.counts.get(self.node_id, 0)
        self.counts[self.node_id] = old + amount
        # Return delta
        return {self.node_id: self.counts[self.node_id]}

    def apply_delta(self, delta):
        """Apply a delta (partial state) from another replica."""
        for node_id, count in delta.items():
            self.counts[node_id] = max(self.counts.get(node_id, 0), count)

    @property
    def value(self):
        return sum(self.counts.values())

    def merge(self, other):
        result = DeltaGCounter(self.node_id)
        for node_id in set(self.counts) | set(other.counts):
            result.counts[node_id] = max(
                self.counts.get(node_id, 0),
                other.counts.get(node_id, 0)
            )
        return result

    def __repr__(self):
        return f"DeltaGCounter({self.value})"


class DeltaPNCounter:
    """Delta-state PNCounter."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.p = DeltaGCounter(self.node_id)
        self.n = DeltaGCounter(self.node_id)

    def increment(self, amount=1):
        delta_p = self.p.increment(amount)
        return ('p', delta_p)

    def decrement(self, amount=1):
        delta_n = self.n.increment(amount)
        return ('n', delta_n)

    def apply_delta(self, delta):
        kind, d = delta
        if kind == 'p':
            self.p.apply_delta(d)
        elif kind == 'n':
            self.n.apply_delta(d)

    @property
    def value(self):
        return self.p.value - self.n.value

    def merge(self, other):
        result = DeltaPNCounter(self.node_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def __repr__(self):
        return f"DeltaPNCounter({self.value})"


# =============================================================================
# CRDT Map (nested CRDTs)
# =============================================================================

class CRDTMap:
    """Map where values are CRDTs. Supports nested composition."""

    def __init__(self, node_id=None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.entries = {}  # key -> CRDT instance
        self.tombstones = set()  # removed keys

    def set(self, key, crdt):
        """Set a key to a CRDT value."""
        self.entries[key] = crdt
        self.tombstones.discard(key)

    def get(self, key):
        if key in self.entries and key not in self.tombstones:
            return self.entries[key]
        return None

    def remove(self, key):
        self.tombstones.add(key)

    def keys(self):
        return frozenset(k for k in self.entries if k not in self.tombstones)

    @property
    def value(self):
        result = {}
        for k in self.keys():
            v = self.entries[k]
            result[k] = v.value if hasattr(v, 'value') else v
        return result

    def merge(self, other):
        result = CRDTMap(self.node_id)
        all_keys = set(self.entries) | set(other.entries)
        for key in all_keys:
            in_self = key in self.entries
            in_other = key in other.entries
            tomb_self = key in self.tombstones
            tomb_other = key in other.tombstones

            if in_self and in_other:
                # Both have it -- merge the values
                merged = self.entries[key].merge(other.entries[key])
                result.entries[key] = merged
                # Only tombstone if both tombstoned
                if tomb_self and tomb_other:
                    result.tombstones.add(key)
            elif in_self:
                result.entries[key] = copy.deepcopy(self.entries[key])
                if tomb_self:
                    result.tombstones.add(key)
            elif in_other:
                result.entries[key] = copy.deepcopy(other.entries[key])
                if tomb_other:
                    result.tombstones.add(key)
        return result

    def __repr__(self):
        return f"CRDTMap({self.value})"


# =============================================================================
# Network Simulation
# =============================================================================

class Message:
    """A message in the simulated network."""
    def __init__(self, sender, receiver, payload, msg_type='state'):
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.msg_type = msg_type  # 'state', 'op', 'delta'
        self.id = uuid.uuid4().hex[:8]


class CRDTNetwork:
    """Simulated network for testing CRDT convergence."""

    def __init__(self):
        self.nodes = {}  # node_id -> CRDT instance
        self.partitions = set()  # set of frozenset pairs that cannot communicate
        self.message_loss_rate = 0.0
        self.pending_messages = []
        self.delivered_messages = []

    def add_node(self, node_id, crdt):
        self.nodes[node_id] = crdt

    def partition(self, node_a, node_b):
        """Create a network partition between two nodes."""
        self.partitions.add(frozenset([node_a, node_b]))

    def heal(self, node_a=None, node_b=None):
        """Heal partition between two nodes, or all if none specified."""
        if node_a is None:
            self.partitions.clear()
        else:
            self.partitions.discard(frozenset([node_a, node_b]))

    def can_communicate(self, node_a, node_b):
        return frozenset([node_a, node_b]) not in self.partitions

    def broadcast_state(self, sender_id):
        """Broadcast full state from sender to all reachable nodes."""
        sender = self.nodes[sender_id]
        for receiver_id in self.nodes:
            if receiver_id == sender_id:
                continue
            if not self.can_communicate(sender_id, receiver_id):
                continue
            if random.random() < self.message_loss_rate:
                continue
            msg = Message(sender_id, receiver_id, copy.deepcopy(sender), 'state')
            self.pending_messages.append(msg)

    def broadcast_op(self, sender_id, op):
        """Broadcast an operation from sender to all reachable nodes."""
        sender = self.nodes[sender_id]
        for receiver_id in self.nodes:
            if receiver_id == sender_id:
                continue
            if not self.can_communicate(sender_id, receiver_id):
                continue
            if random.random() < self.message_loss_rate:
                continue
            msg = Message(sender_id, receiver_id, copy.deepcopy(op), 'op')
            self.pending_messages.append(msg)

    def broadcast_delta(self, sender_id, delta):
        """Broadcast a delta from sender to all reachable nodes."""
        for receiver_id in self.nodes:
            if receiver_id == sender_id:
                continue
            if not self.can_communicate(sender_id, receiver_id):
                continue
            if random.random() < self.message_loss_rate:
                continue
            msg = Message(sender_id, receiver_id, copy.deepcopy(delta), 'delta')
            self.pending_messages.append(msg)

    def deliver_all(self):
        """Deliver all pending messages."""
        while self.pending_messages:
            msg = self.pending_messages.pop(0)
            receiver = self.nodes.get(msg.receiver)
            if receiver is None:
                continue
            if msg.msg_type == 'state':
                self.nodes[msg.receiver] = receiver.merge(msg.payload)
            elif msg.msg_type == 'op':
                receiver.apply_op(msg.payload)
            elif msg.msg_type == 'delta':
                receiver.apply_delta(msg.payload)
            self.delivered_messages.append(msg)

    def sync_all(self):
        """Full state sync: every node broadcasts, then deliver all."""
        for node_id in list(self.nodes):
            self.broadcast_state(node_id)
        self.deliver_all()

    def converged(self):
        """Check if all nodes have the same value."""
        values = [n.value for n in self.nodes.values()]
        return all(v == values[0] for v in values)

    def get_values(self):
        return {nid: n.value for nid, n in self.nodes.items()}


# =============================================================================
# Convenience: CRDT Registry
# =============================================================================

CRDT_TYPES = {
    'gcounter': GCounter,
    'pncounter': PNCounter,
    'gset': GSet,
    'twopset': TwoPSet,
    'orset': ORSet,
    'lwwregister': LWWRegister,
    'mvregister': MVRegister,
    'lwwelementset': LWWElementSet,
    'rga': RGA,
    'opcounter': OpCounter,
    'opset': OpSet,
    'deltagcounter': DeltaGCounter,
    'deltapncounter': DeltaPNCounter,
    'crdtmap': CRDTMap,
}


def create_crdt(crdt_type, node_id=None):
    """Factory function for creating CRDT instances."""
    cls = CRDT_TYPES.get(crdt_type.lower())
    if cls is None:
        raise ValueError(f"Unknown CRDT type: {crdt_type}")
    if node_id and cls.__init__.__code__.co_varnames[1:2] == ('node_id',):
        return cls(node_id)
    return cls()
