"""
C204: Vector Clocks & Causal Broadcast

Fundamental distributed systems primitives for tracking causality:
- VectorClock: Lamport-style vector timestamps for causal ordering
- VersionVector: Conflict detection for replicated data
- CausalHistory: Full causal history tracking with dotted version vectors
- CausalBroadcast: Reliable causal broadcast protocol
- CausalNetwork: Simulation environment for testing

Built from scratch. No external dependencies.
"""


# ---------------------------------------------------------------------------
# Vector Clock
# ---------------------------------------------------------------------------

class VectorClock:
    """Tracks causal ordering across distributed processes."""

    def __init__(self, clock=None):
        self._clock = dict(clock) if clock else {}

    def increment(self, node_id):
        """Increment this node's logical clock."""
        self._clock[node_id] = self._clock.get(node_id, 0) + 1
        return self

    def get(self, node_id):
        """Get the clock value for a node."""
        return self._clock.get(node_id, 0)

    def merge(self, other):
        """Merge with another vector clock (pointwise max)."""
        result = dict(self._clock)
        for node_id, val in other._clock.items():
            result[node_id] = max(result.get(node_id, 0), val)
        return VectorClock(result)

    def update(self, other):
        """Merge other into self in-place."""
        for node_id, val in other._clock.items():
            self._clock[node_id] = max(self._clock.get(node_id, 0), val)
        return self

    def copy(self):
        return VectorClock(self._clock)

    @property
    def clock(self):
        return dict(self._clock)

    def __eq__(self, other):
        if not isinstance(other, VectorClock):
            return NotImplemented
        # Equal if all entries match (missing = 0)
        all_keys = set(self._clock) | set(other._clock)
        return all(self.get(k) == other.get(k) for k in all_keys)

    def __le__(self, other):
        """self <= other: self happened-before or concurrent with other."""
        if not isinstance(other, VectorClock):
            return NotImplemented
        for k in self._clock:
            if self._clock[k] > other.get(k):
                return False
        return True

    def __lt__(self, other):
        """self < other: self strictly happened-before other."""
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, VectorClock):
            return NotImplemented
        return other <= self

    def __gt__(self, other):
        if not isinstance(other, VectorClock):
            return NotImplemented
        return other < self

    def concurrent_with(self, other):
        """True if neither clock dominates the other."""
        return not (self <= other) and not (other <= self)

    def dominates(self, other):
        """True if self strictly dominates other (happened after)."""
        return self > other

    def __repr__(self):
        items = sorted(self._clock.items())
        return "VC({})".format(", ".join("{}:{}".format(k, v) for k, v in items))

    def __hash__(self):
        return hash(tuple(sorted(self._clock.items())))


def compare_clocks(vc1, vc2):
    """Compare two vector clocks. Returns 'before', 'after', 'equal', or 'concurrent'."""
    if vc1 == vc2:
        return 'equal'
    if vc1 < vc2:
        return 'before'
    if vc1 > vc2:
        return 'after'
    return 'concurrent'


# ---------------------------------------------------------------------------
# Version Vector (for replicated data conflict detection)
# ---------------------------------------------------------------------------

class VersionVector:
    """Version vector for detecting conflicts in replicated data stores."""

    def __init__(self):
        self._versions = {}  # node_id -> version

    def increment(self, node_id):
        self._versions[node_id] = self._versions.get(node_id, 0) + 1
        return self

    def get(self, node_id):
        return self._versions.get(node_id, 0)

    def merge(self, other):
        """Merge two version vectors (pointwise max)."""
        result = VersionVector()
        all_keys = set(self._versions) | set(other._versions)
        for k in all_keys:
            result._versions[k] = max(self.get(k), other.get(k))
        return result

    def descends_from(self, other):
        """True if self >= other on all entries (self descends from other)."""
        for k in other._versions:
            if self.get(k) < other.get(k):
                return False
        return True

    def conflicts_with(self, other):
        """True if neither descends from the other (concurrent writes)."""
        return not self.descends_from(other) and not other.descends_from(self)

    def copy(self):
        vv = VersionVector()
        vv._versions = dict(self._versions)
        return vv

    @property
    def versions(self):
        return dict(self._versions)

    def __eq__(self, other):
        if not isinstance(other, VersionVector):
            return NotImplemented
        all_keys = set(self._versions) | set(other._versions)
        return all(self.get(k) == other.get(k) for k in all_keys)

    def __repr__(self):
        items = sorted(self._versions.items())
        return "VV({})".format(", ".join("{}:{}".format(k, v) for k, v in items))


# ---------------------------------------------------------------------------
# Dotted Version Vector (precise conflict detection)
# ---------------------------------------------------------------------------

class Dot:
    """A single event: (node_id, counter)."""
    __slots__ = ('node', 'counter')

    def __init__(self, node, counter):
        self.node = node
        self.counter = counter

    def __eq__(self, other):
        return isinstance(other, Dot) and self.node == other.node and self.counter == other.counter

    def __hash__(self):
        return hash((self.node, self.counter))

    def __repr__(self):
        return "Dot({},{})".format(self.node, self.counter)


class DottedVersionVector:
    """
    Dotted version vector for precise conflict tracking.
    Combines a version vector (causal context) with a dot (the event itself).
    Used in Riak-style CRDTs to avoid false conflicts.
    """

    def __init__(self, version_vector=None, dot=None):
        self.vv = version_vector if version_vector else VersionVector()
        self.dot = dot  # The specific event this value was written at

    def descends(self, other):
        """Does self's causal context include other's dot?"""
        if other.dot is None:
            return self.vv.descends_from(other.vv)
        # Check if our VV covers the other's dot
        return self.vv.get(other.dot.node) >= other.dot.counter

    def concurrent_with(self, other):
        """True if neither descends from the other."""
        return not self.descends(other) and not other.descends(self)

    def __repr__(self):
        return "DVV(vv={}, dot={})".format(self.vv, self.dot)


# ---------------------------------------------------------------------------
# Causal History
# ---------------------------------------------------------------------------

class CausalHistory:
    """
    Tracks full causal history of events using dot sets.
    Supports compact representation and garbage collection.
    """

    def __init__(self):
        self._contiguous = {}  # node -> max contiguous counter
        self._dots = set()     # additional non-contiguous dots

    def add(self, node, counter):
        """Record that we've seen event (node, counter)."""
        self._dots.add(Dot(node, counter))
        self._compact(node)
        return self

    def next_dot(self, node):
        """Generate the next dot for a node."""
        base = self._contiguous.get(node, 0)
        # Check dots above contiguous range
        counter = base + 1
        while Dot(node, counter) in self._dots:
            counter += 1
        return Dot(node, counter)

    def has(self, node, counter):
        """Check if event (node, counter) is in our history."""
        cont = self._contiguous.get(node, 0)
        if counter <= cont:
            return True
        return Dot(node, counter) in self._dots

    def merge(self, other):
        """Merge two causal histories."""
        result = CausalHistory()
        # Merge contiguous ranges
        all_nodes = set(self._contiguous) | set(other._contiguous)
        for node in all_nodes:
            result._contiguous[node] = max(
                self._contiguous.get(node, 0),
                other._contiguous.get(node, 0)
            )
        # Merge dot sets
        result._dots = self._dots | other._dots
        # Compact
        for node in all_nodes | {d.node for d in result._dots}:
            result._compact(node)
        return result

    def _compact(self, node):
        """Absorb contiguous dots into the contiguous range."""
        base = self._contiguous.get(node, 0)
        while Dot(node, base + 1) in self._dots:
            base += 1
            self._dots.discard(Dot(node, base))
        if base > 0:
            self._contiguous[node] = base

    def events_unseen_by(self, other):
        """Return dots in self not in other."""
        unseen = set()
        for node, cont in self._contiguous.items():
            other_cont = other._contiguous.get(node, 0)
            for c in range(other_cont + 1, cont + 1):
                if not Dot(node, c) in other._dots:
                    unseen.add(Dot(node, c))
        for dot in self._dots:
            if not other.has(dot.node, dot.counter):
                unseen.add(dot)
        return unseen

    @property
    def size(self):
        """Total number of events tracked."""
        return sum(self._contiguous.values()) + len(self._dots)

    def __repr__(self):
        parts = []
        for node, cont in sorted(self._contiguous.items()):
            parts.append("{}:1-{}".format(node, cont))
        if self._dots:
            parts.append("dots={}".format(sorted((d.node, d.counter) for d in self._dots)))
        return "CH({})".format(", ".join(parts))


# ---------------------------------------------------------------------------
# Interval Clock (compact representation)
# ---------------------------------------------------------------------------

class IntervalClock:
    """
    Interval-based version vector using contiguous intervals.
    More compact than tracking individual dots when events are mostly sequential.
    """

    def __init__(self):
        self._intervals = {}  # node -> list of (start, end) intervals, sorted

    def add(self, node, counter):
        """Add a single event."""
        if node not in self._intervals:
            self._intervals[node] = [(counter, counter)]
        else:
            self._intervals[node].append((counter, counter))
            self._intervals[node] = self._merge_intervals(self._intervals[node])
        return self

    def add_range(self, node, start, end):
        """Add a range of events [start, end]."""
        if node not in self._intervals:
            self._intervals[node] = [(start, end)]
        else:
            self._intervals[node].append((start, end))
            self._intervals[node] = self._merge_intervals(self._intervals[node])
        return self

    def has(self, node, counter):
        """Check if event (node, counter) is tracked."""
        for start, end in self._intervals.get(node, []):
            if start <= counter <= end:
                return True
        return False

    def max_counter(self, node):
        """Get the maximum counter for a node."""
        intervals = self._intervals.get(node, [])
        if not intervals:
            return 0
        return intervals[-1][1]

    def merge(self, other):
        """Merge two interval clocks."""
        result = IntervalClock()
        all_nodes = set(self._intervals) | set(other._intervals)
        for node in all_nodes:
            combined = list(self._intervals.get(node, [])) + list(other._intervals.get(node, []))
            result._intervals[node] = self._merge_intervals(combined)
        return result

    @staticmethod
    def _merge_intervals(intervals):
        """Merge overlapping/adjacent intervals."""
        if not intervals:
            return []
        sorted_ivs = sorted(intervals)
        merged = [sorted_ivs[0]]
        for start, end in sorted_ivs[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @property
    def gap_count(self):
        """Number of gaps in the interval representation."""
        total = 0
        for intervals in self._intervals.values():
            if len(intervals) > 1:
                total += len(intervals) - 1
        return total

    def __repr__(self):
        parts = []
        for node in sorted(self._intervals):
            ivs = self._intervals[node]
            iv_str = ",".join("[{}-{}]".format(s, e) for s, e in ivs)
            parts.append("{}:{}".format(node, iv_str))
        return "IC({})".format(", ".join(parts))


# ---------------------------------------------------------------------------
# Stamped Value (value + causal metadata)
# ---------------------------------------------------------------------------

class StampedValue:
    """A value with its causal metadata."""

    def __init__(self, value, clock, writer=None, timestamp=None):
        self.value = value
        self.clock = clock  # VectorClock at time of write
        self.writer = writer
        self.timestamp = timestamp  # logical timestamp

    def __repr__(self):
        return "SV({}, {})".format(self.value, self.clock)


# ---------------------------------------------------------------------------
# Causal Store (replicated KV with conflict detection)
# ---------------------------------------------------------------------------

class CausalStore:
    """
    Replicated key-value store with vector clock conflict detection.
    Supports sibling values (concurrent writes produce multiple values).
    """

    def __init__(self, node_id):
        self.node_id = node_id
        self.clock = VectorClock()
        self._data = {}  # key -> list of StampedValue (siblings)

    def put(self, key, value):
        """Write a value, incrementing our clock."""
        self.clock = self.clock.increment(self.node_id)
        sv = StampedValue(value, self.clock.copy(), writer=self.node_id)
        self._data[key] = [sv]
        return sv

    def get(self, key):
        """Get all sibling values for a key."""
        return list(self._data.get(key, []))

    def get_one(self, key):
        """Get a single value (latest or None). Raises if siblings exist."""
        siblings = self.get(key)
        if not siblings:
            return None
        if len(siblings) > 1:
            raise ConflictError(key, siblings)
        return siblings[0].value

    def resolve(self, key, value, context_clock=None):
        """Resolve siblings by writing a new value with merged context."""
        siblings = self.get(key)
        if not siblings:
            return self.put(key, value)
        # Merge all sibling clocks
        merged = VectorClock()
        for sib in siblings:
            merged = merged.merge(sib.clock)
        if context_clock:
            merged = merged.merge(context_clock)
        merged = merged.increment(self.node_id)
        self.clock = self.clock.merge(merged)
        sv = StampedValue(value, merged.copy(), writer=self.node_id)
        self._data[key] = [sv]
        return sv

    def receive_put(self, key, stamped_value):
        """Receive a replicated write from another node."""
        self.clock = self.clock.merge(stamped_value.clock)
        existing = self._data.get(key, [])
        if not existing:
            self._data[key] = [stamped_value]
            return

        # Filter out values dominated by the new one
        surviving = []
        new_dominated = False
        for sib in existing:
            rel = compare_clocks(sib.clock, stamped_value.clock)
            if rel == 'before' or rel == 'equal':
                # Existing is superseded by new
                continue
            elif rel == 'after':
                # New is superseded by existing
                new_dominated = True
                surviving.append(sib)
            else:
                # Concurrent -- keep both
                surviving.append(sib)

        if not new_dominated:
            surviving.append(stamped_value)

        self._data[key] = surviving

    def keys(self):
        return list(self._data.keys())

    def has_conflicts(self, key):
        return len(self._data.get(key, [])) > 1

    def sibling_count(self, key):
        return len(self._data.get(key, []))

    def sync_from(self, other):
        """Pull all data from another store (anti-entropy)."""
        for key in other.keys():
            for sv in other.get(key):
                self.receive_put(key, sv)

    def __repr__(self):
        return "CausalStore({}, keys={})".format(self.node_id, list(self._data.keys()))


class ConflictError(Exception):
    """Raised when reading a key with unresolved siblings."""
    def __init__(self, key, siblings):
        self.key = key
        self.siblings = siblings
        super().__init__("Conflict on key '{}': {} siblings".format(key, len(siblings)))


# ---------------------------------------------------------------------------
# Causal Broadcast Message
# ---------------------------------------------------------------------------

class CausalMessage:
    """A message in the causal broadcast protocol."""

    _next_id = 0

    def __init__(self, sender, payload, clock, deps=None):
        CausalMessage._next_id += 1
        self.id = CausalMessage._next_id
        self.sender = sender
        self.payload = payload
        self.clock = clock.copy()  # VectorClock at send time
        self.deps = deps or set()  # set of message IDs this depends on

    def __repr__(self):
        return "Msg({}, from={}, {})".format(self.id, self.sender, self.clock)


# ---------------------------------------------------------------------------
# Causal Broadcast Node
# ---------------------------------------------------------------------------

class CausalBroadcastNode:
    """
    Node in a causal broadcast protocol.
    Guarantees causal delivery: if msg A happened-before msg B,
    then A is delivered before B at every node.
    """

    def __init__(self, node_id):
        self.node_id = node_id
        self.clock = VectorClock()
        self.delivered = []       # delivered messages in order
        self.pending = []         # buffered messages waiting for deps
        self.delivered_ids = set()
        self.network = None       # set by CausalNetwork
        self._on_deliver = None   # callback

    def on_deliver(self, callback):
        """Set delivery callback: callback(message)."""
        self._on_deliver = callback

    def broadcast(self, payload):
        """Broadcast a message to all other nodes."""
        self.clock = self.clock.increment(self.node_id)
        msg = CausalMessage(
            sender=self.node_id,
            payload=payload,
            clock=self.clock,
            deps=set(self.delivered_ids)
        )
        # Deliver to self immediately
        self._do_deliver(msg)
        # Send to network
        if self.network:
            self.network.send(self.node_id, msg)
        return msg

    def receive(self, msg):
        """Receive a message from the network. Buffer if deps not met."""
        if msg.id in self.delivered_ids:
            return  # Already delivered

        if self._can_deliver(msg):
            self._do_deliver(msg)
            self._try_deliver_pending()
        else:
            self.pending.append(msg)

    def _can_deliver(self, msg):
        """Check if all causal dependencies are satisfied."""
        # A message can be delivered if the sender's clock entry is exactly
        # one more than what we've seen, and all other entries are <= ours
        for node_id in set(list(msg.clock._clock.keys()) + list(self.clock._clock.keys())):
            if node_id == msg.sender:
                # We need sender's previous events
                expected = msg.clock.get(node_id) - 1
                if self.clock.get(node_id) < expected:
                    return False
            else:
                # We need all events the sender had seen
                if self.clock.get(node_id) < msg.clock.get(node_id):
                    return False
        return True

    def _do_deliver(self, msg):
        """Actually deliver a message."""
        self.delivered.append(msg)
        self.delivered_ids.add(msg.id)
        self.clock = self.clock.merge(msg.clock)
        if self._on_deliver:
            self._on_deliver(msg)

    def _try_deliver_pending(self):
        """Try to deliver buffered messages."""
        changed = True
        while changed:
            changed = False
            still_pending = []
            for msg in self.pending:
                if msg.id in self.delivered_ids:
                    changed = True
                    continue
                if self._can_deliver(msg):
                    self._do_deliver(msg)
                    changed = True
                else:
                    still_pending.append(msg)
            self.pending = still_pending

    @property
    def pending_count(self):
        return len(self.pending)

    def __repr__(self):
        return "CBNode({}, delivered={}, pending={})".format(
            self.node_id, len(self.delivered), len(self.pending))


# ---------------------------------------------------------------------------
# Causal Network (simulation)
# ---------------------------------------------------------------------------

class CausalNetwork:
    """
    Simulates a network for causal broadcast testing.
    Supports message reordering, delays, partitions, and loss.
    """

    def __init__(self):
        self.nodes = {}         # node_id -> CausalBroadcastNode
        self.in_flight = []     # (target_id, message) pairs
        self.partitions = set() # set of frozenset({a, b}) pairs that are partitioned
        self.loss_rate = 0.0    # probability of message loss
        self._delivered_log = []  # (target, message) for testing
        self._rng_counter = 0   # deterministic "random" for testing

    def add_node(self, node_id):
        """Add a node to the network."""
        node = CausalBroadcastNode(node_id)
        node.network = self
        self.nodes[node_id] = node
        return node

    def get_node(self, node_id):
        return self.nodes[node_id]

    def send(self, sender_id, msg):
        """Send a message from sender to all other nodes."""
        for target_id in self.nodes:
            if target_id == sender_id:
                continue
            if self._is_partitioned(sender_id, target_id):
                continue
            if self._should_drop():
                continue
            self.in_flight.append((target_id, msg))

    def _is_partitioned(self, a, b):
        return frozenset({a, b}) in self.partitions

    def _should_drop(self):
        if self.loss_rate <= 0:
            return False
        # Deterministic pseudo-random for reproducible tests
        self._rng_counter += 1
        return (self._rng_counter * 2654435761 % (1 << 32)) / (1 << 32) < self.loss_rate

    def partition(self, node_a, node_b):
        """Create a network partition between two nodes."""
        self.partitions.add(frozenset({node_a, node_b}))

    def heal(self, node_a, node_b):
        """Heal a partition between two nodes."""
        self.partitions.discard(frozenset({node_a, node_b}))

    def heal_all(self):
        """Heal all partitions."""
        self.partitions.clear()

    def deliver_one(self):
        """Deliver the first in-flight message. Returns True if delivered."""
        if not self.in_flight:
            return False
        target_id, msg = self.in_flight.pop(0)
        node = self.nodes[target_id]
        node.receive(msg)
        self._delivered_log.append((target_id, msg))
        return True

    def deliver_all(self):
        """Deliver all in-flight messages (may produce new ones)."""
        iterations = 0
        while self.in_flight and iterations < 10000:
            self.deliver_one()
            iterations += 1
        return iterations

    def deliver_reversed(self):
        """Deliver all messages in reverse order (worst-case reordering)."""
        self.in_flight.reverse()
        return self.deliver_all()

    def deliver_random(self, seed=42):
        """Deliver messages in pseudo-random order."""
        import random
        rng = random.Random(seed)
        while self.in_flight:
            idx = rng.randint(0, len(self.in_flight) - 1)
            self.in_flight[0], self.in_flight[idx] = self.in_flight[idx], self.in_flight[0]
            self.deliver_one()

    @property
    def in_flight_count(self):
        return len(self.in_flight)

    def __repr__(self):
        return "CausalNetwork(nodes={}, in_flight={})".format(
            len(self.nodes), len(self.in_flight))


# ---------------------------------------------------------------------------
# Matrix Clock (knows what others know)
# ---------------------------------------------------------------------------

class MatrixClock:
    """
    Matrix clock: each node maintains a vector clock for every other node.
    Enables garbage collection -- you know what all nodes have seen.
    """

    def __init__(self, node_id, all_nodes):
        self.node_id = node_id
        self.all_nodes = list(all_nodes)
        # matrix[i][j] = node i's knowledge of node j's clock
        self._matrix = {}
        for i in all_nodes:
            self._matrix[i] = {}
            for j in all_nodes:
                self._matrix[i][j] = 0

    def increment(self):
        """Increment own clock."""
        self._matrix[self.node_id][self.node_id] += 1
        return self

    def get(self, observer, observed):
        """Get observer's view of observed's clock."""
        return self._matrix.get(observer, {}).get(observed, 0)

    def local_clock(self):
        """Get this node's view as a VectorClock."""
        return VectorClock(self._matrix[self.node_id])

    def send_stamp(self):
        """Generate a stamp to send with a message."""
        self.increment()
        # Return our entire row
        return dict(self._matrix[self.node_id])

    def receive_stamp(self, sender_id, sender_row):
        """Process a received matrix clock stamp."""
        # Update our view of sender's clock
        for node_id in self.all_nodes:
            val = sender_row.get(node_id, 0)
            self._matrix[self.node_id][node_id] = max(
                self._matrix[self.node_id][node_id], val
            )
        # Update our knowledge of what sender knows
        self._matrix[sender_id] = dict(sender_row)

    def min_known(self, node_id):
        """What is the minimum clock value that ALL nodes have seen for node_id?"""
        return min(self._matrix[n].get(node_id, 0) for n in self.all_nodes)

    def can_gc(self, event_node, event_counter):
        """Can we garbage collect this event? (All nodes have seen it.)"""
        return self.min_known(event_node) >= event_counter

    def __repr__(self):
        row = self._matrix[self.node_id]
        items = sorted(row.items())
        return "MC({}, {})".format(self.node_id,
            ", ".join("{}:{}".format(k, v) for k, v in items))


# ---------------------------------------------------------------------------
# Bloom Clock (probabilistic vector clock)
# ---------------------------------------------------------------------------

class BloomClock:
    """
    Probabilistic causality tracking using Bloom filter-like structure.
    Fixed-size regardless of number of nodes. Trades accuracy for space.
    False positives possible on ordering queries.
    """

    def __init__(self, size=64, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self._counters = [0] * size

    def _hash_positions(self, node_id):
        """Get hash positions for a node."""
        positions = []
        h = hash(str(node_id))
        for i in range(self.num_hashes):
            h = (h * 2654435761 + i * 1234567) % (1 << 32)
            positions.append(h % self.size)
        return positions

    def increment(self, node_id):
        """Increment counters for this node."""
        for pos in self._hash_positions(node_id):
            self._counters[pos] += 1
        return self

    def merge(self, other):
        """Merge with another bloom clock (pointwise max)."""
        result = BloomClock(self.size, self.num_hashes)
        for i in range(self.size):
            result._counters[i] = max(self._counters[i], other._counters[i])
        return result

    def possibly_before(self, other):
        """True if self might have happened before other."""
        for i in range(self.size):
            if self._counters[i] > other._counters[i]:
                return False
        return True

    def definitely_concurrent(self, other):
        """True if definitely concurrent (neither dominates)."""
        self_less = False
        other_less = False
        for i in range(self.size):
            if self._counters[i] < other._counters[i]:
                self_less = True
            if self._counters[i] > other._counters[i]:
                other_less = True
        return self_less and other_less

    def copy(self):
        result = BloomClock(self.size, self.num_hashes)
        result._counters = list(self._counters)
        return result

    def __repr__(self):
        non_zero = sum(1 for c in self._counters if c > 0)
        return "BloomClock(size={}, active={})".format(self.size, non_zero)


# ---------------------------------------------------------------------------
# Causal Delivery Verifier (for testing)
# ---------------------------------------------------------------------------

class CausalDeliveryVerifier:
    """
    Verifies that causal delivery guarantees hold.
    Checks that if msg A happened-before msg B, A is delivered first at every node.
    """

    def __init__(self):
        self._delivery_order = {}  # node_id -> [msg_id, ...]
        self._msg_clocks = {}      # msg_id -> VectorClock
        self._violations = []

    def record_delivery(self, node_id, msg):
        """Record that a message was delivered to a node."""
        if node_id not in self._delivery_order:
            self._delivery_order[node_id] = []
        self._delivery_order[node_id].append(msg.id)
        self._msg_clocks[msg.id] = msg.clock.copy()

    def verify(self):
        """Check all deliveries for causal violations. Returns list of violations."""
        self._violations = []
        for node_id, order in self._delivery_order.items():
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    msg_a = order[i]
                    msg_b = order[j]
                    clock_a = self._msg_clocks[msg_a]
                    clock_b = self._msg_clocks[msg_b]
                    # If B happened-before A but B was delivered after A
                    if clock_b < clock_a:
                        pass  # OK, A was delivered first and dominates B is wrong
                    if clock_a > clock_b:
                        pass  # A happened after B but delivered first -- violation
                        # Actually, the check is: if B < A (B happened before A)
                        # then B must be delivered before A. B is at index j > i,
                        # so B was delivered AFTER A. That's a violation.
                        # Wait -- let me re-think. order[i] was delivered first.
                        # If clock of order[j] < clock of order[i], then order[j]
                        # happened before order[i], but was delivered after. Violation.
                    if compare_clocks(clock_b, clock_a) == 'before':
                        # msg_b happened before msg_a, but msg_b delivered after msg_a
                        self._violations.append({
                            'node': node_id,
                            'msg_early': msg_a,
                            'msg_late': msg_b,
                            'issue': 'msg {} happened before msg {} but delivered after'.format(
                                msg_b, msg_a)
                        })
        return self._violations

    @property
    def is_causal(self):
        """True if no causal violations detected."""
        return len(self.verify()) == 0
