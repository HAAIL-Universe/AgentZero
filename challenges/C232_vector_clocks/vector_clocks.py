"""
C232: Vector Clocks -- Causal Ordering for Distributed Systems

Implements multiple vector clock variants for tracking causality:

1. VectorClock -- Classic vector clock (Lamport/Fidge/Mattern)
   - Increment, merge, compare (concurrent/before/after)
   - Causal ordering of events

2. VersionVector -- Optimized for replica versioning
   - Dominance checks, conflict detection
   - Compact representation

3. DottedVersionVector -- Accurate conflict detection (Riak-style)
   - Dot + version vector separation
   - Sibling resolution without false conflicts

4. IntervalTreeClock -- Dynamic process creation/retirement
   - Fork/join semantics
   - No fixed process set required

5. BloomClock -- Probabilistic vector clock
   - Bounded space regardless of process count
   - Tunable false positive rate

6. CausalHistory -- Full event DAG tracking
   - Transitive reduction
   - Causal path queries
"""

from __future__ import annotations
import hashlib
import math
from typing import Optional


# ---------------------------------------------------------------------------
# 1. VectorClock -- Classic Lamport/Fidge/Mattern
# ---------------------------------------------------------------------------

class VectorClock:
    """Classic vector clock for causal ordering."""

    def __init__(self, clock: Optional[dict[str, int]] = None):
        self._clock: dict[str, int] = dict(clock) if clock else {}

    def copy(self) -> VectorClock:
        return VectorClock(self._clock)

    @property
    def clock(self) -> dict[str, int]:
        return dict(self._clock)

    def get(self, node_id: str) -> int:
        return self._clock.get(node_id, 0)

    def increment(self, node_id: str) -> VectorClock:
        """Increment the clock for a local event."""
        new_clock = dict(self._clock)
        new_clock[node_id] = new_clock.get(node_id, 0) + 1
        return VectorClock(new_clock)

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge two clocks (take component-wise max)."""
        merged = dict(self._clock)
        for node_id, count in other._clock.items():
            merged[node_id] = max(merged.get(node_id, 0), count)
        return merged if isinstance(merged, VectorClock) else VectorClock(merged)

    def __le__(self, other: VectorClock) -> bool:
        """self happens-before or equals other."""
        for node_id, count in self._clock.items():
            if count > other._clock.get(node_id, 0):
                return False
        return True

    def __lt__(self, other: VectorClock) -> bool:
        """self strictly happens-before other."""
        return self <= other and self != other

    def __ge__(self, other: VectorClock) -> bool:
        return other <= self

    def __gt__(self, other: VectorClock) -> bool:
        return other < self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_keys = set(self._clock.keys()) | set(other._clock.keys())
        for k in all_keys:
            if self._clock.get(k, 0) != other._clock.get(k, 0):
                return False
        return True

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, v) for k, v in self._clock.items() if v > 0)))

    def concurrent(self, other: VectorClock) -> bool:
        """True if neither clock dominates the other."""
        return not (self <= other) and not (other <= self)

    def dominates(self, other: VectorClock) -> bool:
        """True if self strictly dominates other (self > other)."""
        return self > other

    def descends(self, other: VectorClock) -> bool:
        """True if self descends from other (other <= self)."""
        return other <= self

    def compare(self, other: VectorClock) -> str:
        """Return 'before', 'after', 'equal', or 'concurrent'."""
        if self == other:
            return "equal"
        if self < other:
            return "before"
        if self > other:
            return "after"
        return "concurrent"

    def nodes(self) -> set[str]:
        """Return set of node IDs that have entries."""
        return {k for k, v in self._clock.items() if v > 0}

    def __repr__(self) -> str:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(self._clock.items()) if v > 0)
        return f"VC({items})"


# ---------------------------------------------------------------------------
# 2. VersionVector -- Optimized for replica versioning
# ---------------------------------------------------------------------------

class VersionVector:
    """Version vector for replica conflict detection."""

    def __init__(self, vector: Optional[dict[str, int]] = None):
        self._vector: dict[str, int] = dict(vector) if vector else {}

    def copy(self) -> VersionVector:
        return VersionVector(self._vector)

    @property
    def vector(self) -> dict[str, int]:
        return dict(self._vector)

    def increment(self, replica_id: str) -> VersionVector:
        new_vec = dict(self._vector)
        new_vec[replica_id] = new_vec.get(replica_id, 0) + 1
        return VersionVector(new_vec)

    def merge(self, other: VersionVector) -> VersionVector:
        merged = dict(self._vector)
        for rid, count in other._vector.items():
            merged[rid] = max(merged.get(rid, 0), count)
        return VersionVector(merged)

    def dominates(self, other: VersionVector) -> bool:
        """Does self strictly dominate other?"""
        at_least = True
        strictly_greater = False
        all_keys = set(self._vector.keys()) | set(other._vector.keys())
        for k in all_keys:
            s = self._vector.get(k, 0)
            o = other._vector.get(k, 0)
            if s < o:
                at_least = False
                break
            if s > o:
                strictly_greater = True
        return at_least and strictly_greater

    def conflicts(self, other: VersionVector) -> bool:
        """True if neither dominates (concurrent versions)."""
        return not self.dominates(other) and not other.dominates(self) and self != other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionVector):
            return NotImplemented
        all_keys = set(self._vector.keys()) | set(other._vector.keys())
        return all(self._vector.get(k, 0) == other._vector.get(k, 0) for k in all_keys)

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, v) for k, v in self._vector.items() if v > 0)))

    def __repr__(self) -> str:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(self._vector.items()) if v > 0)
        return f"VV({items})"


# ---------------------------------------------------------------------------
# 3. DottedVersionVector -- Riak-style accurate conflict detection
# ---------------------------------------------------------------------------

class Dot:
    """A single (replica, counter) event."""

    def __init__(self, replica_id: str, counter: int):
        self.replica_id = replica_id
        self.counter = counter

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dot):
            return NotImplemented
        return self.replica_id == other.replica_id and self.counter == other.counter

    def __hash__(self) -> int:
        return hash((self.replica_id, self.counter))

    def __repr__(self) -> str:
        return f"Dot({self.replica_id}, {self.counter})"


class DottedVersionVector:
    """
    Dotted version vector (DVV) for accurate conflict detection.

    A DVV = (dot, version_vector) where:
    - dot is the event that created this version
    - version_vector is the causal context
    """

    def __init__(self, dot: Optional[Dot] = None,
                 version_vector: Optional[dict[str, int]] = None):
        self.dot = dot
        self._vv: dict[str, int] = dict(version_vector) if version_vector else {}

    @property
    def version_vector(self) -> dict[str, int]:
        return dict(self._vv)

    def new_event(self, replica_id: str) -> DottedVersionVector:
        """Create a new event at the given replica."""
        # Merge dot into VV first
        new_vv = dict(self._vv)
        if self.dot:
            new_vv[self.dot.replica_id] = max(
                new_vv.get(self.dot.replica_id, 0), self.dot.counter
            )
        # New counter
        new_counter = new_vv.get(replica_id, 0) + 1
        new_vv[replica_id] = max(new_vv.get(replica_id, 0), new_counter - 1) if new_counter > 1 else new_vv.get(replica_id, 0)
        return DottedVersionVector(Dot(replica_id, new_counter), new_vv)

    def sync(self, other: DottedVersionVector) -> DottedVersionVector:
        """Merge causal contexts (version vectors) from two DVVs."""
        merged_vv = dict(self._vv)
        if self.dot:
            merged_vv[self.dot.replica_id] = max(
                merged_vv.get(self.dot.replica_id, 0), self.dot.counter
            )
        for rid, count in other._vv.items():
            merged_vv[rid] = max(merged_vv.get(rid, 0), count)
        if other.dot:
            merged_vv[other.dot.replica_id] = max(
                merged_vv.get(other.dot.replica_id, 0), other.dot.counter
            )
        return DottedVersionVector(self.dot, merged_vv)

    def descends(self, other: DottedVersionVector) -> bool:
        """Does self causally descend from other?"""
        # other's dot must be covered by self's causal context
        if other.dot:
            self_vv = dict(self._vv)
            if self.dot:
                self_vv[self.dot.replica_id] = max(
                    self_vv.get(self.dot.replica_id, 0), self.dot.counter
                )
            if self_vv.get(other.dot.replica_id, 0) < other.dot.counter:
                return False
        # other's VV must be dominated
        self_vv = dict(self._vv)
        if self.dot:
            self_vv[self.dot.replica_id] = max(
                self_vv.get(self.dot.replica_id, 0), self.dot.counter
            )
        for rid, count in other._vv.items():
            if self_vv.get(rid, 0) < count:
                return False
        return True

    def concurrent(self, other: DottedVersionVector) -> bool:
        """True if neither descends from the other."""
        return not self.descends(other) and not other.descends(self)

    def _full_vv(self) -> dict[str, int]:
        """Get full version vector including dot."""
        vv = dict(self._vv)
        if self.dot:
            vv[self.dot.replica_id] = max(
                vv.get(self.dot.replica_id, 0), self.dot.counter
            )
        return vv

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DottedVersionVector):
            return NotImplemented
        return self.dot == other.dot and self._full_vv() == other._full_vv()

    def __repr__(self) -> str:
        vv_str = ", ".join(f"{k}:{v}" for k, v in sorted(self._vv.items()) if v > 0)
        return f"DVV({self.dot}, {{{vv_str}}})"


# ---------------------------------------------------------------------------
# 4. IntervalTreeClock -- Dynamic process creation/retirement
# ---------------------------------------------------------------------------

class ITCStamp:
    """
    Interval Tree Clock stamp.

    An ITC stamp = (id, event) where:
    - id is a binary tree of ownership fractions
    - event is a tree of counters

    Supports fork (split ownership), join (merge), and event (local tick).
    """

    def __init__(self, id_tree=None, event_tree=None):
        # id_tree: 0, 1, or (left, right) tuple
        # event_tree: int or (base, left, right) tuple
        self.id = id_tree if id_tree is not None else 1
        self.event = event_tree if event_tree is not None else 0

    @staticmethod
    def seed() -> ITCStamp:
        """Create the initial seed stamp."""
        return ITCStamp(1, 0)

    def fork(self) -> tuple[ITCStamp, ITCStamp]:
        """Split this stamp into two with complementary IDs."""
        left_id, right_id = ITCStamp._split_id(self.id)
        return (
            ITCStamp(left_id, ITCStamp._copy_event(self.event)),
            ITCStamp(right_id, ITCStamp._copy_event(self.event))
        )

    def join(self, other: ITCStamp) -> ITCStamp:
        """Merge two stamps."""
        merged_id = ITCStamp._sum_id(self.id, other.id)
        merged_event = ITCStamp._join_event(self.event, other.event)
        return ITCStamp(merged_id, merged_event)

    def event_tick(self) -> ITCStamp:
        """Record a local event (increment where we have ownership)."""
        new_event = ITCStamp._grow(self.id, self.event)
        return ITCStamp(ITCStamp._copy_id(self.id), new_event)

    def leq(self, other: ITCStamp) -> bool:
        """self <= other in causal order."""
        return ITCStamp._leq_event(self.event, other.event)

    def concurrent(self, other: ITCStamp) -> bool:
        """Neither dominates the other."""
        return not self.leq(other) and not other.leq(self)

    @staticmethod
    def _split_id(id_tree):
        """Split an ID into two complementary halves."""
        if id_tree == 0:
            return (0, 0)
        if id_tree == 1:
            return ((1, 0), (0, 1))
        left, right = id_tree
        if left == 0:
            rl, rr = ITCStamp._split_id(right)
            return ((0, rl), (0, rr))
        if right == 0:
            ll, lr = ITCStamp._split_id(left)
            return ((ll, 0), (lr, 0))
        return ((left, 0), (0, right))

    @staticmethod
    def _sum_id(a, b):
        """Sum two complementary IDs."""
        if a == 0:
            return b
        if b == 0:
            return a
        if isinstance(a, tuple) and isinstance(b, tuple):
            result = (ITCStamp._sum_id(a[0], b[0]), ITCStamp._sum_id(a[1], b[1]))
            if result == (1, 1):
                return 1
            return result
        return 1  # Both non-zero leaves sum to 1

    @staticmethod
    def _copy_id(id_tree):
        if isinstance(id_tree, tuple):
            return (ITCStamp._copy_id(id_tree[0]), ITCStamp._copy_id(id_tree[1]))
        return id_tree

    @staticmethod
    def _copy_event(event):
        if isinstance(event, tuple):
            return (event[0], ITCStamp._copy_event(event[1]), ITCStamp._copy_event(event[2]))
        return event

    @staticmethod
    def _norm_event(event):
        """Normalize an event tree by lifting common minimums."""
        if isinstance(event, tuple):
            base, left, right = event
            left = ITCStamp._norm_event(left)
            right = ITCStamp._norm_event(right)
            if isinstance(left, int) and isinstance(right, int) and left == right:
                return base + left
            left_min = ITCStamp._min_event(left)
            right_min = ITCStamp._min_event(right)
            m = min(left_min, right_min)
            if m > 0:
                left = ITCStamp._lower_event(left, m)
                right = ITCStamp._lower_event(right, m)
                return (base + m, left, right)
            return (base, left, right)
        return event

    @staticmethod
    def _min_event(event) -> int:
        if isinstance(event, int):
            return event
        base, left, right = event
        return base + min(ITCStamp._min_event(left), ITCStamp._min_event(right))

    @staticmethod
    def _max_event(event) -> int:
        if isinstance(event, int):
            return event
        base, left, right = event
        return base + max(ITCStamp._max_event(left), ITCStamp._max_event(right))

    @staticmethod
    def _lower_event(event, amount: int):
        if isinstance(event, int):
            return event - amount
        base, left, right = event
        return (base - amount, left, right)

    @staticmethod
    def _lift_event(event, amount: int):
        if isinstance(event, int):
            return event + amount
        base, left, right = event
        return (base + amount, left, right)

    @staticmethod
    def _join_event(a, b):
        """Join two event trees (component-wise max)."""
        if isinstance(a, int) and isinstance(b, int):
            return max(a, b)
        # Normalize both to tuple form
        if isinstance(a, int):
            a = (a, 0, 0)
        if isinstance(b, int):
            b = (b, 0, 0)
        base_a, la, ra = a
        base_b, lb, rb = b
        if base_a > base_b:
            lb = ITCStamp._lift_event(lb, base_b - base_a) if base_b >= base_a else ITCStamp._join_raise(lb, base_a - base_b)
            rb = ITCStamp._lift_event(rb, base_b - base_a) if base_b >= base_a else ITCStamp._join_raise(rb, base_a - base_b)
            result = (base_a,
                      ITCStamp._join_event(la, lb),
                      ITCStamp._join_event(ra, rb))
        elif base_b > base_a:
            la_adj = ITCStamp._join_raise(la, base_b - base_a)
            ra_adj = ITCStamp._join_raise(ra, base_b - base_a)
            result = (base_b,
                      ITCStamp._join_event(la_adj, lb),
                      ITCStamp._join_event(ra_adj, rb))
        else:
            result = (base_a,
                      ITCStamp._join_event(la, lb),
                      ITCStamp._join_event(ra, rb))
        return ITCStamp._norm_event(result)

    @staticmethod
    def _join_raise(event, amount):
        """Raise minimum of event by amount (for join with higher base)."""
        if isinstance(event, int):
            return max(event - amount, 0)
        base, left, right = event
        new_base = base - amount
        if new_base >= 0:
            return (new_base, left, right)
        return ITCStamp._join_event(
            ITCStamp._join_raise(left, -new_base),
            ITCStamp._join_raise(right, -new_base)
        )

    @staticmethod
    def _leq_event(a, b) -> bool:
        """a <= b in event ordering."""
        if isinstance(a, int) and isinstance(b, int):
            return a <= b
        if isinstance(a, int):
            a = (a, 0, 0)
        if isinstance(b, int):
            return ITCStamp._max_event(a) <= b
        base_a, la, ra = a
        base_b, lb, rb = b
        if base_a > base_b:
            return False
        diff = base_b - base_a
        return (ITCStamp._leq_event(la, ITCStamp._lift_event(lb, diff)) and
                ITCStamp._leq_event(ra, ITCStamp._lift_event(rb, diff)))

    @staticmethod
    def _fill(id_tree, event):
        """Try to fill the event tree where we have ownership."""
        if id_tree == 0:
            return None  # No ownership here
        if id_tree == 1:
            # Full ownership -- max out
            return ITCStamp._max_event(event)
        if isinstance(id_tree, tuple):
            id_left, id_right = id_tree
            if isinstance(event, int):
                event = (event, 0, 0)
            base, eleft, eright = event
            if id_right == 0:
                filled = ITCStamp._fill(id_left, eleft)
                if filled is not None and filled != eleft:
                    return ITCStamp._norm_event((base, filled, eright))
                return None
            if id_left == 0:
                filled = ITCStamp._fill(id_right, eright)
                if filled is not None and filled != eright:
                    return ITCStamp._norm_event((base, eleft, filled))
                return None
            fl = ITCStamp._fill(id_left, eleft)
            fr = ITCStamp._fill(id_right, eright)
            if fl is not None and fl != eleft:
                return ITCStamp._norm_event((base, fl, eright))
            if fr is not None and fr != eright:
                return ITCStamp._norm_event((base, eleft, fr))
            return None
        return None

    @staticmethod
    def _grow(id_tree, event):
        """Grow the event tree (increment) where we have ownership."""
        if id_tree == 1:
            if isinstance(event, int):
                return event + 1
            base, left, right = event
            return (base + 1, 0, 0)
        if isinstance(id_tree, tuple):
            id_left, id_right = id_tree
            if isinstance(event, int):
                event = (event, 0, 0)
            base, eleft, eright = event
            if id_right == 0:
                grown = ITCStamp._grow(id_left, eleft)
                return ITCStamp._norm_event((base, grown, eright))
            if id_left == 0:
                grown = ITCStamp._grow(id_right, eright)
                return ITCStamp._norm_event((base, eleft, grown))
            # Both sides have ownership - grow on left by default
            grown = ITCStamp._grow(id_left, eleft)
            return ITCStamp._norm_event((base, grown, eright))
        return event

    def __repr__(self) -> str:
        return f"ITC(id={self.id}, event={self.event})"


# ---------------------------------------------------------------------------
# 5. BloomClock -- Probabilistic vector clock
# ---------------------------------------------------------------------------

class BloomClock:
    """
    Probabilistic vector clock using Bloom filter bit arrays.

    Trades perfect accuracy for bounded space. Supports any number
    of processes without growing the clock size.
    """

    def __init__(self, size: int = 128, num_hashes: int = 3,
                 bits: Optional[bytearray] = None, count: int = 0):
        self._size = size
        self._num_hashes = num_hashes
        self._bits = bytearray(bits) if bits else bytearray(size)
        self._count = count  # Number of events recorded

    @property
    def size(self) -> int:
        return self._size

    @property
    def count(self) -> int:
        return self._count

    def copy(self) -> BloomClock:
        return BloomClock(self._size, self._num_hashes,
                          bytearray(self._bits), self._count)

    def _hash_positions(self, node_id: str, counter: int) -> list[int]:
        """Get bit positions for a (node_id, counter) event."""
        positions = []
        data = f"{node_id}:{counter}".encode()
        for i in range(self._num_hashes):
            h = hashlib.sha256(data + i.to_bytes(2, 'big')).digest()
            pos = int.from_bytes(h[:4], 'big') % self._size
            positions.append(pos)
        return positions

    def record(self, node_id: str) -> BloomClock:
        """Record a local event for the given node."""
        new_bc = self.copy()
        new_bc._count += 1
        positions = new_bc._hash_positions(node_id, new_bc._count)
        for pos in positions:
            new_bc._bits[pos] = 1
        return new_bc

    def merge(self, other: BloomClock) -> BloomClock:
        """Merge two Bloom clocks (bitwise OR)."""
        assert self._size == other._size
        new_bits = bytearray(self._size)
        for i in range(self._size):
            new_bits[i] = self._bits[i] | other._bits[i]
        return BloomClock(self._size, self._num_hashes, new_bits,
                          max(self._count, other._count))

    def contains(self, other: BloomClock) -> bool:
        """Check if self's bits are a superset of other's bits."""
        for i in range(self._size):
            if other._bits[i] and not self._bits[i]:
                return False
        return True

    def maybe_before(self, other: BloomClock) -> bool:
        """
        Probabilistic happens-before check.
        If True, self MIGHT happen before other (could be false positive).
        If False, self definitely does NOT happen before other.
        """
        return other.contains(self) and self._bits != other._bits

    def definitely_concurrent(self, other: BloomClock) -> bool:
        """
        If True, the clocks are definitely concurrent.
        (Neither's bits are a subset of the other's.)
        """
        return not self.contains(other) and not other.contains(self)

    def fill_ratio(self) -> float:
        """What fraction of bits are set."""
        set_bits = sum(1 for b in self._bits if b)
        return set_bits / self._size if self._size > 0 else 0.0

    def estimated_false_positive_rate(self) -> float:
        """Estimate current false positive rate."""
        p = self.fill_ratio()
        return p ** self._num_hashes

    def __repr__(self) -> str:
        set_bits = sum(1 for b in self._bits if b)
        return f"BloomClock(size={self._size}, set={set_bits}, count={self._count})"


# ---------------------------------------------------------------------------
# 6. CausalHistory -- Full event DAG
# ---------------------------------------------------------------------------

class Event:
    """An event in a causal history."""

    def __init__(self, event_id: str, node_id: str,
                 data: Optional[str] = None,
                 clock: Optional[VectorClock] = None):
        self.event_id = event_id
        self.node_id = node_id
        self.data = data
        self.clock = clock or VectorClock()
        self.causes: list[str] = []  # Event IDs of direct causes

    def __repr__(self) -> str:
        return f"Event({self.event_id}, node={self.node_id})"


class CausalHistory:
    """
    Full causal event DAG with vector clocks.

    Tracks events and their causal relationships, supporting
    queries about causal paths, concurrent events, and frontiers.
    """

    def __init__(self):
        self._events: dict[str, Event] = {}
        self._clocks: dict[str, VectorClock] = {}  # Per-node current clock
        self._children: dict[str, list[str]] = {}  # event_id -> caused events

    @property
    def events(self) -> dict[str, Event]:
        return dict(self._events)

    def add_event(self, event_id: str, node_id: str,
                  causes: Optional[list[str]] = None,
                  data: Optional[str] = None) -> Event:
        """Add an event with explicit causal dependencies."""
        if event_id in self._events:
            raise ValueError(f"Event {event_id} already exists")

        # Build clock: merge all causes' clocks, then increment
        clock = self._clocks.get(node_id, VectorClock())
        if causes:
            for cause_id in causes:
                if cause_id not in self._events:
                    raise ValueError(f"Cause event {cause_id} does not exist")
                clock = clock.merge(self._events[cause_id].clock)

        clock = clock.increment(node_id)

        event = Event(event_id, node_id, data, clock)
        event.causes = list(causes) if causes else []

        self._events[event_id] = event
        self._clocks[node_id] = clock

        # Track children
        if event_id not in self._children:
            self._children[event_id] = []
        if causes:
            for cause_id in causes:
                if cause_id not in self._children:
                    self._children[cause_id] = []
                self._children[cause_id].append(event_id)

        return event

    def local_event(self, event_id: str, node_id: str,
                    data: Optional[str] = None) -> Event:
        """Add a local event (caused by the node's last event, if any)."""
        # Find the node's latest event
        causes = []
        for eid, ev in self._events.items():
            if ev.node_id == node_id:
                # Check if it's the latest for this node
                is_latest = True
                for other_eid in self._children.get(eid, []):
                    if self._events[other_eid].node_id == node_id:
                        is_latest = False
                        break
                if is_latest:
                    causes.append(eid)
        return self.add_event(event_id, node_id, causes if causes else None, data)

    def send_event(self, event_id: str, sender_id: str,
                   data: Optional[str] = None) -> Event:
        """Record a send event."""
        return self.local_event(event_id, sender_id, data)

    def receive_event(self, event_id: str, receiver_id: str,
                      send_event_id: str,
                      data: Optional[str] = None) -> Event:
        """Record a receive event caused by a send."""
        causes = [send_event_id]
        # Also include receiver's latest
        for eid, ev in self._events.items():
            if ev.node_id == receiver_id:
                is_latest = True
                for other_eid in self._children.get(eid, []):
                    if self._events[other_eid].node_id == receiver_id:
                        is_latest = False
                        break
                if is_latest:
                    causes.append(eid)
        return self.add_event(event_id, receiver_id, causes, data)

    def happens_before(self, event_a: str, event_b: str) -> bool:
        """Does event_a causally happen before event_b?"""
        ea = self._events[event_a]
        eb = self._events[event_b]
        return ea.clock < eb.clock

    def concurrent(self, event_a: str, event_b: str) -> bool:
        """Are two events concurrent?"""
        ea = self._events[event_a]
        eb = self._events[event_b]
        return ea.clock.concurrent(eb.clock)

    def causal_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """Find a causal path from one event to another (BFS)."""
        if from_id not in self._events or to_id not in self._events:
            return None
        if from_id == to_id:
            return [from_id]

        # BFS through children
        from collections import deque
        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()
            for child in self._children.get(current, []):
                if child == to_id:
                    return path + [child]
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        return None

    def frontier(self) -> list[str]:
        """Get the frontier (events with no children)."""
        result = []
        for eid in self._events:
            if not self._children.get(eid, []):
                result.append(eid)
        return sorted(result)

    def concurrent_set(self, event_id: str) -> set[str]:
        """Get all events concurrent with the given event."""
        result = set()
        for eid in self._events:
            if eid != event_id and self.concurrent(event_id, eid):
                result.add(eid)
        return result

    def causal_cut(self, event_id: str) -> set[str]:
        """Get the causal past of an event (all events it depends on)."""
        result = set()
        ev = self._events[event_id]

        def walk(eid: str):
            if eid in result:
                return
            result.add(eid)
            for cause in self._events[eid].causes:
                walk(cause)

        for cause in ev.causes:
            walk(cause)
        return result

    def linearize(self) -> list[str]:
        """Topological sort of events respecting causality."""
        in_degree: dict[str, int] = {eid: 0 for eid in self._events}
        for eid, ev in self._events.items():
            for cause in ev.causes:
                in_degree[eid] = in_degree.get(eid, 0)  # ensure exists

        # Count in-degrees from causes
        in_degree = {eid: len(ev.causes) for eid, ev in self._events.items()}

        from collections import deque
        queue = deque(sorted(eid for eid, d in in_degree.items() if d == 0))
        result = []

        while queue:
            eid = queue.popleft()
            result.append(eid)
            for child in sorted(self._children.get(eid, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"CausalHistory({len(self._events)} events)"


# ---------------------------------------------------------------------------
# 7. CausalBroadcast -- Reliable causal broadcast protocol
# ---------------------------------------------------------------------------

class CausalBroadcast:
    """
    Causal broadcast protocol using vector clocks.

    Ensures messages are delivered in causal order:
    if send(m1) -> send(m2), then deliver(m1) before deliver(m2).
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = list(peers)
        self.clock = VectorClock()
        self._pending: list[tuple[str, VectorClock, str]] = []  # (sender, clock, data)
        self._delivered: list[tuple[str, str]] = []  # (sender, data)

    def broadcast(self, data: str) -> tuple[VectorClock, str]:
        """Broadcast a message. Returns (clock, data) to send to all peers."""
        self.clock = self.clock.increment(self.node_id)
        self._delivered.append((self.node_id, data))
        return (self.clock.copy(), data)

    def receive(self, sender: str, msg_clock: VectorClock,
                data: str) -> list[tuple[str, str]]:
        """
        Receive a message. Buffer if not causally ready.
        Returns list of newly delivered (sender, data) pairs.
        """
        self._pending.append((sender, msg_clock, data))
        return self._try_deliver()

    def _is_deliverable(self, sender: str, msg_clock: VectorClock) -> bool:
        """Check if a message is causally ready for delivery."""
        # The sender's component must be exactly our knowledge + 1
        if msg_clock.get(sender) != self.clock.get(sender) + 1:
            return False
        # All other components must be <= our clock
        for node_id in msg_clock.nodes():
            if node_id != sender:
                if msg_clock.get(node_id) > self.clock.get(node_id):
                    return False
        return True

    def _try_deliver(self) -> list[tuple[str, str]]:
        """Try to deliver buffered messages in causal order."""
        delivered = []
        changed = True
        while changed:
            changed = False
            remaining = []
            for sender, msg_clock, data in self._pending:
                if self._is_deliverable(sender, msg_clock):
                    self.clock = self.clock.merge(msg_clock)
                    self._delivered.append((sender, data))
                    delivered.append((sender, data))
                    changed = True
                else:
                    remaining.append((sender, msg_clock, data))
            self._pending = remaining
        return delivered

    @property
    def delivered(self) -> list[tuple[str, str]]:
        return list(self._delivered)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def __repr__(self) -> str:
        return f"CausalBroadcast({self.node_id}, delivered={len(self._delivered)}, pending={len(self._pending)})"


# ---------------------------------------------------------------------------
# 8. CausalConsistencyChecker -- Verify causal consistency of event logs
# ---------------------------------------------------------------------------

class CausalConsistencyChecker:
    """
    Verifies that a log of events maintains causal consistency.

    Can detect:
    - Causal violations (effect before cause)
    - Missing events in causal chains
    - Clock anomalies
    """

    def __init__(self):
        self._log: list[tuple[str, VectorClock, str]] = []  # (node, clock, data)

    def add_entry(self, node_id: str, clock: VectorClock,
                  data: str = "") -> None:
        """Add a log entry."""
        self._log.append((node_id, clock, data))

    def check_consistency(self) -> list[str]:
        """
        Check the log for causal consistency violations.
        Returns list of violation descriptions.
        """
        violations = []

        # Check 1: Per-node monotonicity
        node_latest: dict[str, VectorClock] = {}
        for i, (node_id, clock, data) in enumerate(self._log):
            if node_id in node_latest:
                prev = node_latest[node_id]
                if not prev <= clock:
                    violations.append(
                        f"Entry {i}: Non-monotonic clock for {node_id}"
                    )
            node_latest[node_id] = clock

        # Check 2: Per-node counter increments by 1
        node_counters: dict[str, int] = {}
        for i, (node_id, clock, data) in enumerate(self._log):
            expected = node_counters.get(node_id, 0) + 1
            actual = clock.get(node_id)
            if actual != expected:
                if actual > expected + 1:
                    violations.append(
                        f"Entry {i}: Gap in {node_id}'s counter "
                        f"(expected {expected}, got {actual})"
                    )
            node_counters[node_id] = actual

        # Check 3: Delivery order respects causality
        delivered_clocks: list[tuple[str, VectorClock]] = []
        for i, (node_id, clock, data) in enumerate(self._log):
            for j, (prev_node, prev_clock) in enumerate(delivered_clocks):
                if prev_clock > clock and prev_node != node_id:
                    violations.append(
                        f"Entry {i}: Delivered before causal dependency "
                        f"(entry {j} has greater clock)"
                    )
            delivered_clocks.append((node_id, clock))

        return violations

    def find_anomalies(self) -> dict[str, list[int]]:
        """Find clock anomalies grouped by type."""
        anomalies: dict[str, list[int]] = {
            "gaps": [],
            "non_monotonic": [],
            "concurrent_same_node": [],
        }

        node_latest: dict[str, VectorClock] = {}
        node_prev_counter: dict[str, int] = {}

        for i, (node_id, clock, _) in enumerate(self._log):
            if node_id in node_latest:
                prev = node_latest[node_id]
                if not prev <= clock:
                    anomalies["non_monotonic"].append(i)
                if prev.concurrent(clock):
                    anomalies["concurrent_same_node"].append(i)

            counter = clock.get(node_id)
            if node_id in node_prev_counter:
                if counter > node_prev_counter[node_id] + 1:
                    anomalies["gaps"].append(i)
            node_prev_counter[node_id] = counter
            node_latest[node_id] = clock

        return anomalies

    def __len__(self) -> int:
        return len(self._log)
