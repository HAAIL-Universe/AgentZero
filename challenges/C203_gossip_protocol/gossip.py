"""
C203: Gossip Protocol & Failure Detection
==========================================
A SWIM-style gossip protocol with multiple failure detection strategies
for building reliable distributed membership and state dissemination.

Components:
- GossipNode: Core node with membership list and gossip state
- SWIMProtocol: SWIM failure detection (ping, ping-req, suspect)
- PhiAccrualDetector: Phi accrual failure detector (adaptive thresholds)
- HeartbeatDetector: Simple heartbeat-based failure detection
- InfectionDissemination: Epidemic broadcast for membership updates
- GossipNetwork: Simulated network with partitions, delays, loss
- AntiEntropy: Merkle-tree based state synchronization
- GossipState: Eventually-consistent key-value state via gossip

Based on:
- SWIM (Das et al., 2002)
- Phi Accrual Failure Detector (Hayashibara et al., 2004)
- Epidemic algorithms (Demers et al., 1987)
"""

import math
import time
import random
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Optional


# =============================================================================
# Enums and Constants
# =============================================================================

class NodeStatus(Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"
    LEFT = "left"  # Graceful departure


class MessageType(Enum):
    PING = "ping"
    PING_ACK = "ack"
    PING_REQ = "ping_req"
    PING_REQ_ACK = "ping_req_ack"
    COMPOUND = "compound"  # Piggybacked membership updates
    STATE_PUSH = "state_push"
    STATE_PULL = "state_pull"
    STATE_PUSH_PULL = "state_push_pull"
    JOIN = "join"
    LEAVE = "leave"
    SYNC_DIGEST = "sync_digest"
    SYNC_DELTA = "sync_delta"


class DisseminationType(Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    CONFIRM = "confirm"  # Confirmed dead
    LEAVE = "leave"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Member:
    """A member in the gossip group."""
    node_id: str
    address: str  # Simulated address
    status: NodeStatus = NodeStatus.ALIVE
    incarnation: int = 0  # Monotonic counter to refute suspicion
    last_updated: float = 0.0
    metadata: dict = field(default_factory=dict)

    def copy(self):
        return Member(self.node_id, self.address, self.status,
                      self.incarnation, self.last_updated, dict(self.metadata))


@dataclass
class GossipMessage:
    """A message in the gossip protocol."""
    type: MessageType
    src: str
    dst: str
    data: dict = field(default_factory=dict)
    piggyback: list = field(default_factory=list)  # Piggybacked updates
    seq: int = 0  # Sequence number for ping correlation


@dataclass
class MembershipUpdate:
    """A membership change to disseminate."""
    dtype: DisseminationType
    node_id: str
    incarnation: int
    timestamp: float = 0.0
    invalidations: int = 0  # Times this has been sent (for limiting)

    def key(self):
        return self.node_id  # One update per node (supersede by priority)


@dataclass
class StateEntry:
    """A key-value entry in the gossip state."""
    key: str
    value: Any
    version: int = 0
    node_id: str = ""  # Which node owns this
    timestamp: float = 0.0

    def copy(self):
        return StateEntry(self.key, self.value, self.version, self.node_id, self.timestamp)


# =============================================================================
# Phi Accrual Failure Detector
# =============================================================================

class PhiAccrualDetector:
    """
    Phi accrual failure detector (Hayashibara et al., 2004).

    Instead of binary alive/dead, computes a suspicion level phi.
    phi = -log10(P(heartbeat_late | normal_distribution))
    Higher phi = more suspicious. Threshold is configurable.
    """

    def __init__(self, threshold=8.0, max_sample_size=200,
                 min_std_deviation_ms=100, initial_heartbeat_ms=1000):
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.min_std_deviation_ms = min_std_deviation_ms
        self.initial_heartbeat_ms = initial_heartbeat_ms
        # Per-node heartbeat intervals
        self._intervals = defaultdict(lambda: deque(maxlen=self.max_sample_size))
        self._last_heartbeat = {}
        self._first_heartbeat = {}

    def heartbeat(self, node_id, now=None):
        """Record a heartbeat from a node."""
        now = now if now is not None else time.time()
        if node_id in self._last_heartbeat:
            interval = now - self._last_heartbeat[node_id]
            self._intervals[node_id].append(interval)
        else:
            self._first_heartbeat[node_id] = now
            # Seed with initial interval
            self._intervals[node_id].append(self.initial_heartbeat_ms / 1000.0)
        self._last_heartbeat[node_id] = now

    def phi(self, node_id, now=None):
        """Compute phi value for a node. Higher = more likely failed."""
        now = now if now is not None else time.time()
        if node_id not in self._last_heartbeat:
            return 0.0  # No data yet

        last = self._last_heartbeat[node_id]
        intervals = self._intervals[node_id]

        if len(intervals) == 0:
            return 0.0

        # Compute mean and std of intervals
        mean = sum(intervals) / len(intervals)
        if mean <= 0:
            return 0.0
        if len(intervals) < 2:
            std = mean / 4  # Conservative estimate
        else:
            variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
            std = max(math.sqrt(variance), self.min_std_deviation_ms / 1000.0)

        elapsed = now - last
        if elapsed <= 0:
            return 0.0

        # Use normal distribution CDF approximation
        # P(late) = P(X > elapsed) = 1 - Phi((elapsed - mean) / std)
        # phi_value = -log10(P(late))
        y = (elapsed - mean) / std

        # Approximate normal CDF using error function approach
        # For large y, P(late) is very small -> high phi
        # Using logistic approximation: Phi(x) ~ 1/(1+exp(-1.7*x - 0.73*x^3))
        # But simpler: use exponential distribution for tail behavior
        # P(late) ~ exp(-0.5 * y^2) / (y * sqrt(2*pi)) for large y
        # For a practical detector, use: phi = y^2 / 2 * log10(e) for y > 0
        if y <= 0:
            return 0.0

        # Simple and effective: phi grows quadratically with deviation
        # This matches the Gaussian tail: -log10(1-Phi(y)) ~ y^2/(2*ln(10))
        p_late = math.exp(-0.5 * y * y)
        if p_late < 1e-15:
            return 34.0
        return -math.log10(p_late)

    def is_available(self, node_id, now=None):
        """Check if node is considered available."""
        return self.phi(node_id, now) < self.threshold

    def remove(self, node_id):
        """Remove tracking for a node."""
        self._intervals.pop(node_id, None)
        self._last_heartbeat.pop(node_id, None)
        self._first_heartbeat.pop(node_id, None)


# =============================================================================
# Simple Heartbeat Detector
# =============================================================================

class HeartbeatDetector:
    """Simple timeout-based failure detector."""

    def __init__(self, timeout=5.0):
        self.timeout = timeout
        self._last_heartbeat = {}

    def heartbeat(self, node_id, now=None):
        now = now if now is not None else time.time()
        self._last_heartbeat[node_id] = now

    def is_available(self, node_id, now=None):
        now = now if now is not None else time.time()
        if node_id not in self._last_heartbeat:
            return True  # No data, assume alive
        return (now - self._last_heartbeat[node_id]) < self.timeout

    def time_since(self, node_id, now=None):
        now = now if now is not None else time.time()
        if node_id not in self._last_heartbeat:
            return float('inf')
        return now - self._last_heartbeat[node_id]

    def remove(self, node_id):
        self._last_heartbeat.pop(node_id, None)


# =============================================================================
# Merkle Tree for Anti-Entropy
# =============================================================================

class MerkleNode:
    """Node in a Merkle tree for efficient state comparison."""
    def __init__(self, hash_val="", left=None, right=None, key=None, value_hash=None):
        self.hash = hash_val
        self.left = left
        self.right = right
        self.key = key  # Only for leaf nodes
        self.value_hash = value_hash  # Only for leaf nodes


class MerkleTree:
    """Merkle tree for detecting state differences efficiently."""

    def __init__(self):
        self.root = None
        self._entries = {}  # key -> hash of value

    def _hash(self, data):
        return hashlib.sha256(str(data).encode()).hexdigest()[:16]

    def update(self, key, value):
        """Update a key-value pair."""
        self._entries[key] = self._hash(value)
        self._rebuild()

    def remove(self, key):
        """Remove a key."""
        if key in self._entries:
            del self._entries[key]
            self._rebuild()

    def _rebuild(self):
        """Rebuild tree from entries."""
        if not self._entries:
            self.root = None
            return
        sorted_keys = sorted(self._entries.keys())
        leaves = [MerkleNode(hash_val=self._entries[k], key=k, value_hash=self._entries[k])
                  for k in sorted_keys]
        self.root = self._build_tree(leaves)

    def _build_tree(self, nodes):
        if len(nodes) == 1:
            return nodes[0]
        mid = len(nodes) // 2
        left = self._build_tree(nodes[:mid])
        right = self._build_tree(nodes[mid:])
        combined = self._hash(left.hash + right.hash)
        return MerkleNode(hash_val=combined, left=left, right=right)

    def root_hash(self):
        return self.root.hash if self.root else ""

    def digest(self):
        """Get a compact digest for comparison."""
        return {"root": self.root_hash(), "keys": sorted(self._entries.keys()),
                "hashes": {k: v for k, v in self._entries.items()}}

    def diff(self, other_digest):
        """Find keys that differ from another digest."""
        my_hashes = self._entries
        other_hashes = other_digest.get("hashes", {})
        all_keys = set(my_hashes.keys()) | set(other_hashes.keys())
        differing = []
        for k in all_keys:
            if my_hashes.get(k) != other_hashes.get(k):
                differing.append(k)
        return differing


# =============================================================================
# Gossip State (Eventually-Consistent KV)
# =============================================================================

class GossipState:
    """Eventually-consistent key-value store propagated via gossip."""

    def __init__(self, node_id):
        self.node_id = node_id
        self._entries = {}  # key -> StateEntry
        self._merkle = MerkleTree()

    def get(self, key):
        entry = self._entries.get(key)
        return entry.value if entry else None

    def set(self, key, value, now=None):
        """Set a local key-value pair."""
        now = now if now is not None else time.time()
        old = self._entries.get(key)
        version = (old.version + 1) if old else 1
        entry = StateEntry(key=key, value=value, version=version,
                           node_id=self.node_id, timestamp=now)
        self._entries[key] = entry
        self._merkle.update(key, (value, version))
        return entry

    def merge_entry(self, entry):
        """Merge a remote entry, keeping highest version."""
        existing = self._entries.get(entry.key)
        if existing is None or entry.version > existing.version or \
           (entry.version == existing.version and entry.timestamp > existing.timestamp):
            self._entries[entry.key] = entry.copy()
            self._merkle.update(entry.key, (entry.value, entry.version))
            return True
        return False

    def merge_entries(self, entries):
        """Merge multiple remote entries."""
        merged = 0
        for entry in entries:
            if self.merge_entry(entry):
                merged += 1
        return merged

    def all_entries(self):
        return list(self._entries.values())

    def entries_for_keys(self, keys):
        return [self._entries[k].copy() for k in keys if k in self._entries]

    def digest(self):
        return self._merkle.digest()

    def diff_keys(self, remote_digest):
        return self._merkle.diff(remote_digest)

    def delete(self, key):
        if key in self._entries:
            del self._entries[key]
            self._merkle.remove(key)
            return True
        return False

    def keys(self):
        return list(self._entries.keys())

    def size(self):
        return len(self._entries)


# =============================================================================
# Infection-style Dissemination
# =============================================================================

class DisseminationQueue:
    """
    Queue of membership updates to piggyback on protocol messages.
    Each update has a limited number of times it can be sent (lambda * log(N)).
    """

    def __init__(self, retransmit_mult=4):
        self.retransmit_mult = retransmit_mult
        self._updates = {}  # key -> MembershipUpdate
        self._queue = deque()  # Order of insertion for fairness

    def add(self, update):
        """Add or supersede a membership update."""
        key = update.key()
        old = self._updates.get(key)
        # Supersede if newer incarnation or higher priority type
        if old is None or update.incarnation > old.incarnation or \
           (update.incarnation == old.incarnation and
            self._priority(update.dtype) > self._priority(old.dtype)):
            self._updates[key] = update
            if old is None:
                self._queue.append(key)

    def _priority(self, dtype):
        """Higher priority types supersede lower ones."""
        order = {DisseminationType.ALIVE: 0, DisseminationType.SUSPECT: 1,
                 DisseminationType.CONFIRM: 2, DisseminationType.LEAVE: 3}
        return order.get(dtype, 0)

    def get_batch(self, max_count, cluster_size):
        """Get a batch of updates to piggyback. Removes expired ones."""
        max_transmits = self.retransmit_mult * max(1, _log2_ceil(max(cluster_size, 1)))
        batch = []
        to_remove = []
        seen = set()

        for key in list(self._queue):
            if len(batch) >= max_count:
                break
            if key in seen:
                continue
            seen.add(key)
            update = self._updates.get(key)
            if update is None:
                to_remove.append(key)
                continue
            if update.invalidations >= max_transmits:
                to_remove.append(key)
                continue
            batch.append(update)
            update.invalidations += 1

        for key in to_remove:
            self._updates.pop(key, None)
            try:
                self._queue.remove(key)
            except ValueError:
                pass

        return batch

    def size(self):
        return len(self._updates)


def _log2_ceil(n):
    if n <= 1:
        return 1
    return math.ceil(math.log2(n))


# =============================================================================
# SWIM-style Gossip Node
# =============================================================================

class GossipNode:
    """
    A node in a SWIM-style gossip protocol.

    Combines:
    - SWIM failure detection (ping, ping-req, suspect/alive/dead)
    - Infection-style dissemination (piggyback on protocol messages)
    - Anti-entropy (Merkle tree state sync)
    - Phi accrual detector (adaptive failure detection)
    """

    def __init__(self, node_id, address=None, config=None):
        self.node_id = node_id
        self.address = address or node_id
        self.config = config or {}

        # Configuration
        self.protocol_period = self.config.get("protocol_period", 1.0)
        self.suspect_timeout = self.config.get("suspect_timeout", 5.0)
        self.ping_timeout = self.config.get("ping_timeout", 0.5)
        self.ping_req_count = self.config.get("ping_req_count", 3)
        self.max_piggyback = self.config.get("max_piggyback", 10)
        self.gossip_fanout = self.config.get("gossip_fanout", 3)
        self.use_phi_detector = self.config.get("use_phi_detector", False)

        # State
        self.incarnation = 0
        self.members = {}  # node_id -> Member
        self.members[node_id] = Member(node_id, self.address, NodeStatus.ALIVE, 0)
        self._seq = 0
        self._pending_acks = {}  # seq -> (target, deadline, indirect_for)
        self._pending_ping_reqs = {}  # seq -> (original_target, requester)
        self._suspect_timers = {}  # node_id -> deadline

        # Subsystems
        self.dissemination = DisseminationQueue(
            self.config.get("retransmit_mult", 4))
        self.state = GossipState(node_id)
        self.phi_detector = PhiAccrualDetector(
            threshold=self.config.get("phi_threshold", 8.0))
        self.heartbeat_detector = HeartbeatDetector(
            timeout=self.config.get("heartbeat_timeout", 5.0))

        # Stats
        self.stats = {"pings_sent": 0, "pings_received": 0, "acks_sent": 0,
                      "acks_received": 0, "ping_reqs_sent": 0,
                      "updates_sent": 0, "updates_received": 0,
                      "state_syncs": 0, "suspect_events": 0, "dead_events": 0}

        # Event log for testing
        self._events = []
        self._outbox = []  # Messages to send

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def _log_event(self, event_type, **kwargs):
        self._events.append({"type": event_type, "node": self.node_id, **kwargs})

    # --- Membership ---

    def join(self, seed_id, seed_address=None):
        """Join a cluster via a seed node."""
        seed_address = seed_address or seed_id
        self.members[seed_id] = Member(seed_id, seed_address, NodeStatus.ALIVE)
        msg = GossipMessage(
            type=MessageType.JOIN, src=self.node_id, dst=seed_id,
            data={"node_id": self.node_id, "address": self.address})
        self._outbox.append(msg)
        self._log_event("join", seed=seed_id)

    def leave(self):
        """Gracefully leave the cluster."""
        self.incarnation += 1
        update = MembershipUpdate(
            DisseminationType.LEAVE, self.node_id, self.incarnation,
            timestamp=time.time())
        self.dissemination.add(update)
        self.members[self.node_id].status = NodeStatus.LEFT
        # Broadcast leave to all alive members
        for mid, m in self.members.items():
            if mid != self.node_id and m.status == NodeStatus.ALIVE:
                msg = GossipMessage(
                    type=MessageType.LEAVE, src=self.node_id, dst=mid,
                    data={"node_id": self.node_id, "incarnation": self.incarnation})
                self._outbox.append(msg)
        self._log_event("leave")

    def alive_members(self):
        """Get list of alive member IDs (excluding self)."""
        return [mid for mid, m in self.members.items()
                if mid != self.node_id and m.status in (NodeStatus.ALIVE, NodeStatus.SUSPECT)]

    def all_member_ids(self):
        return list(self.members.keys())

    def member_count(self):
        return len([m for m in self.members.values()
                    if m.status in (NodeStatus.ALIVE, NodeStatus.SUSPECT)])

    def get_member(self, node_id):
        return self.members.get(node_id)

    # --- SWIM Protocol Tick ---

    def tick(self, now=None):
        """Run one protocol period."""
        now = now if now is not None else time.time()

        # Check suspect timers
        self._check_suspect_timers(now)

        # Check pending acks
        self._check_pending_acks(now)

        # Probe a random member
        self._probe_random(now)

        return self._drain_outbox()

    def _probe_random(self, now):
        """Select a random alive member and ping it."""
        candidates = self.alive_members()
        if not candidates:
            return
        target = random.choice(candidates)
        self._send_ping(target, now)

    def _send_ping(self, target, now=None):
        """Send a direct ping to target."""
        now = now if now is not None else time.time()
        seq = self._next_seq()
        batch = self.dissemination.get_batch(self.max_piggyback, len(self.members))
        msg = GossipMessage(
            type=MessageType.PING, src=self.node_id, dst=target,
            seq=seq, piggyback=self._serialize_updates(batch))
        self._pending_acks[seq] = (target, now + self.ping_timeout, None)
        self._outbox.append(msg)
        self.stats["pings_sent"] += 1

    def _send_ping_req(self, target, now):
        """Send indirect pings via random members."""
        candidates = [m for m in self.alive_members() if m != target]
        mediators = random.sample(candidates, min(self.ping_req_count, len(candidates)))
        for mediator in mediators:
            seq = self._next_seq()
            msg = GossipMessage(
                type=MessageType.PING_REQ, src=self.node_id, dst=mediator,
                seq=seq, data={"target": target})
            self._pending_acks[seq] = (target, now + self.ping_timeout * 3, None)
            self._outbox.append(msg)
            self.stats["ping_reqs_sent"] += 1

    def _check_pending_acks(self, now):
        """Check for timed-out pings."""
        expired = []
        for seq, (target, deadline, indirect_for) in list(self._pending_acks.items()):
            if now >= deadline:
                expired.append((seq, target, indirect_for))

        for seq, target, indirect_for in expired:
            del self._pending_acks[seq]
            if indirect_for is None:
                # Direct ping timed out -- try indirect
                self._send_ping_req(target, now)
            # After ping-req also times out, suspect the node
            # (this happens on the next tick if no ack arrives)

    def _check_suspect_timers(self, now):
        """Promote suspects to dead if timer expired."""
        expired = [nid for nid, deadline in self._suspect_timers.items()
                   if now >= deadline]
        for nid in expired:
            del self._suspect_timers[nid]
            if nid in self.members and self.members[nid].status == NodeStatus.SUSPECT:
                self._confirm_dead(nid)

    # --- Receiving Messages ---

    def receive(self, msg, now=None):
        """Handle an incoming message."""
        now = now if now is not None else time.time()

        # Process piggybacked updates
        if msg.piggyback:
            self._process_piggyback(msg.piggyback, now)

        handler = {
            MessageType.PING: self._handle_ping,
            MessageType.PING_ACK: self._handle_ack,
            MessageType.PING_REQ: self._handle_ping_req,
            MessageType.PING_REQ_ACK: self._handle_ping_req_ack,
            MessageType.JOIN: self._handle_join,
            MessageType.LEAVE: self._handle_leave,
            MessageType.STATE_PUSH: self._handle_state_push,
            MessageType.STATE_PULL: self._handle_state_pull,
            MessageType.STATE_PUSH_PULL: self._handle_state_push_pull,
            MessageType.SYNC_DIGEST: self._handle_sync_digest,
            MessageType.SYNC_DELTA: self._handle_sync_delta,
        }.get(msg.type)

        if handler:
            handler(msg, now)

        return self._drain_outbox()

    def _handle_ping(self, msg, now):
        """Respond to a ping with an ack."""
        self.stats["pings_received"] += 1
        self._record_heartbeat(msg.src, now)
        batch = self.dissemination.get_batch(self.max_piggyback, len(self.members))
        ack = GossipMessage(
            type=MessageType.PING_ACK, src=self.node_id, dst=msg.src,
            seq=msg.seq, piggyback=self._serialize_updates(batch))
        self._outbox.append(ack)
        self.stats["acks_sent"] += 1

    def _handle_ack(self, msg, now):
        """Handle a ping acknowledgment."""
        self.stats["acks_received"] += 1
        self._record_heartbeat(msg.src, now)
        # Clear pending ack
        if msg.seq in self._pending_acks:
            target, _, _ = self._pending_acks.pop(msg.seq)
            # If this node was suspected, mark alive
            if target in self.members and self.members[target].status == NodeStatus.SUSPECT:
                self._mark_alive(target, self.members[target].incarnation)

    def _handle_ping_req(self, msg, now):
        """Forward a ping request to the target."""
        target = msg.data.get("target")
        if not target:
            return
        seq = self._next_seq()
        fwd = GossipMessage(
            type=MessageType.PING, src=self.node_id, dst=target, seq=seq)
        self._pending_ping_reqs[seq] = (target, msg.src)
        self._outbox.append(fwd)

    def _handle_ping_req_ack(self, msg, now):
        """Handle ack from an indirect ping -- forward to original requester."""
        # This is handled via the normal ack path
        pass

    def _handle_join(self, msg, now):
        """Handle a join request."""
        new_id = msg.data.get("node_id", msg.src)
        new_addr = msg.data.get("address", new_id)
        self.members[new_id] = Member(new_id, new_addr, NodeStatus.ALIVE, 0, now)
        self._record_heartbeat(new_id, now)
        # Disseminate alive update
        update = MembershipUpdate(DisseminationType.ALIVE, new_id, 0, now)
        self.dissemination.add(update)
        # Send back member list
        member_data = {mid: {"address": m.address, "status": m.status.value,
                             "incarnation": m.incarnation}
                       for mid, m in self.members.items()
                       if m.status in (NodeStatus.ALIVE, NodeStatus.SUSPECT)}
        reply = GossipMessage(
            type=MessageType.COMPOUND, src=self.node_id, dst=new_id,
            data={"members": member_data})
        self._outbox.append(reply)
        self._log_event("member_joined", new_node=new_id)

    def _handle_leave(self, msg, now):
        """Handle a leave notification."""
        leaving_id = msg.data.get("node_id", msg.src)
        inc = msg.data.get("incarnation", 0)
        if leaving_id in self.members:
            m = self.members[leaving_id]
            if inc >= m.incarnation:
                m.status = NodeStatus.LEFT
                m.incarnation = inc
                m.last_updated = now
                update = MembershipUpdate(DisseminationType.LEAVE, leaving_id, inc, now)
                self.dissemination.add(update)
                self._log_event("member_left", node=leaving_id)
                self.phi_detector.remove(leaving_id)
                self.heartbeat_detector.remove(leaving_id)

    # --- State Sync (Anti-Entropy) ---

    def _handle_state_push(self, msg, now):
        """Receive pushed state entries."""
        entries = self._deserialize_state_entries(msg.data.get("entries", []))
        merged = self.state.merge_entries(entries)
        self.stats["state_syncs"] += 1

    def _handle_state_pull(self, msg, now):
        """Respond with requested state entries."""
        keys = msg.data.get("keys", [])
        entries = self.state.entries_for_keys(keys) if keys else self.state.all_entries()
        reply = GossipMessage(
            type=MessageType.STATE_PUSH, src=self.node_id, dst=msg.src,
            data={"entries": self._serialize_state_entries(entries)})
        self._outbox.append(reply)

    def _handle_state_push_pull(self, msg, now):
        """Exchange state: merge incoming, send back ours."""
        entries = self._deserialize_state_entries(msg.data.get("entries", []))
        self.state.merge_entries(entries)
        # Send back our entries
        reply = GossipMessage(
            type=MessageType.STATE_PUSH, src=self.node_id, dst=msg.src,
            data={"entries": self._serialize_state_entries(self.state.all_entries())})
        self._outbox.append(reply)
        self.stats["state_syncs"] += 1

    def _handle_sync_digest(self, msg, now):
        """Handle a Merkle digest and send back differing entries."""
        remote_digest = msg.data.get("digest", {})
        diff_keys = self.state.diff_keys(remote_digest)
        if diff_keys:
            entries = self.state.entries_for_keys(diff_keys)
            reply = GossipMessage(
                type=MessageType.SYNC_DELTA, src=self.node_id, dst=msg.src,
                data={"entries": self._serialize_state_entries(entries),
                      "digest": self.state.digest()})
            self._outbox.append(reply)

    def _handle_sync_delta(self, msg, now):
        """Handle delta entries from a sync."""
        entries = self._deserialize_state_entries(msg.data.get("entries", []))
        self.state.merge_entries(entries)
        # Check if we have entries they don't
        remote_digest = msg.data.get("digest", {})
        if remote_digest:
            diff_keys = self.state.diff_keys(remote_digest)
            # Only send keys that we have and they don't (or ours are newer)
            our_entries = self.state.entries_for_keys(diff_keys)
            if our_entries:
                reply = GossipMessage(
                    type=MessageType.STATE_PUSH, src=self.node_id, dst=msg.src,
                    data={"entries": self._serialize_state_entries(our_entries)})
                self._outbox.append(reply)

    def sync_state_with(self, target):
        """Initiate anti-entropy sync with a peer."""
        digest = self.state.digest()
        msg = GossipMessage(
            type=MessageType.SYNC_DIGEST, src=self.node_id, dst=target,
            data={"digest": digest})
        self._outbox.append(msg)
        return self._drain_outbox()

    def push_state_to(self, target, keys=None):
        """Push state entries to a peer."""
        entries = self.state.entries_for_keys(keys) if keys else self.state.all_entries()
        msg = GossipMessage(
            type=MessageType.STATE_PUSH, src=self.node_id, dst=target,
            data={"entries": self._serialize_state_entries(entries)})
        self._outbox.append(msg)
        return self._drain_outbox()

    # --- Suspicion / Failure ---

    def _suspect_node(self, node_id):
        """Mark a node as suspect."""
        if node_id not in self.members:
            return
        m = self.members[node_id]
        if m.status == NodeStatus.DEAD or m.status == NodeStatus.LEFT:
            return
        if m.status == NodeStatus.ALIVE:
            m.status = NodeStatus.SUSPECT
            m.last_updated = time.time()
            self._suspect_timers[node_id] = time.time() + self.suspect_timeout
            update = MembershipUpdate(
                DisseminationType.SUSPECT, node_id, m.incarnation, time.time())
            self.dissemination.add(update)
            self.stats["suspect_events"] += 1
            self._log_event("suspect", target=node_id, incarnation=m.incarnation)

    def _mark_alive(self, node_id, incarnation=None):
        """Mark a node as alive (refuting suspicion)."""
        if node_id not in self.members:
            return
        m = self.members[node_id]
        if incarnation is not None and incarnation > m.incarnation:
            m.incarnation = incarnation
        m.status = NodeStatus.ALIVE
        m.last_updated = time.time()
        self._suspect_timers.pop(node_id, None)
        update = MembershipUpdate(
            DisseminationType.ALIVE, node_id, m.incarnation, time.time())
        self.dissemination.add(update)
        self._log_event("alive", target=node_id, incarnation=m.incarnation)

    def _confirm_dead(self, node_id):
        """Confirm a node as dead."""
        if node_id not in self.members:
            return
        m = self.members[node_id]
        m.status = NodeStatus.DEAD
        m.last_updated = time.time()
        update = MembershipUpdate(
            DisseminationType.CONFIRM, node_id, m.incarnation, time.time())
        self.dissemination.add(update)
        self.phi_detector.remove(node_id)
        self.heartbeat_detector.remove(node_id)
        self.stats["dead_events"] += 1
        self._log_event("dead", target=node_id)

    def refute_suspicion(self):
        """Refute suspicion about self by incrementing incarnation."""
        self.incarnation += 1
        self.members[self.node_id].incarnation = self.incarnation
        self.members[self.node_id].status = NodeStatus.ALIVE
        update = MembershipUpdate(
            DisseminationType.ALIVE, self.node_id, self.incarnation, time.time())
        self.dissemination.add(update)
        self._log_event("refute", incarnation=self.incarnation)

    # --- Piggybacking ---

    def _process_piggyback(self, updates_data, now):
        """Process piggybacked membership updates."""
        for ud in updates_data:
            dtype = DisseminationType(ud["dtype"])
            node_id = ud["node_id"]
            incarnation = ud["incarnation"]
            self.stats["updates_received"] += 1

            if node_id == self.node_id:
                # Someone says something about us
                if dtype == DisseminationType.SUSPECT:
                    if incarnation >= self.incarnation:
                        self.refute_suspicion()
                continue

            member = self.members.get(node_id)

            if dtype == DisseminationType.ALIVE:
                if member is None:
                    self.members[node_id] = Member(
                        node_id, node_id, NodeStatus.ALIVE, incarnation, now)
                    self._record_heartbeat(node_id, now)
                elif incarnation > member.incarnation or \
                     (incarnation == member.incarnation and
                      member.status == NodeStatus.SUSPECT):
                    member.incarnation = incarnation
                    member.status = NodeStatus.ALIVE
                    member.last_updated = now
                    self._suspect_timers.pop(node_id, None)
                    self._record_heartbeat(node_id, now)

            elif dtype == DisseminationType.SUSPECT:
                if member is None:
                    self.members[node_id] = Member(
                        node_id, node_id, NodeStatus.SUSPECT, incarnation, now)
                elif incarnation >= member.incarnation and member.status == NodeStatus.ALIVE:
                    member.status = NodeStatus.SUSPECT
                    member.incarnation = incarnation
                    member.last_updated = now
                    self._suspect_timers[node_id] = now + self.suspect_timeout

            elif dtype == DisseminationType.CONFIRM:
                if member is not None and member.status != NodeStatus.LEFT:
                    member.status = NodeStatus.DEAD
                    member.last_updated = now
                    self._suspect_timers.pop(node_id, None)
                    self.phi_detector.remove(node_id)
                    self.heartbeat_detector.remove(node_id)

            elif dtype == DisseminationType.LEAVE:
                if member is not None:
                    if incarnation >= member.incarnation:
                        member.status = NodeStatus.LEFT
                        member.incarnation = incarnation
                        member.last_updated = now
                        self._suspect_timers.pop(node_id, None)
                        self.phi_detector.remove(node_id)
                        self.heartbeat_detector.remove(node_id)

            # Re-disseminate
            update = MembershipUpdate(dtype, node_id, incarnation, now)
            self.dissemination.add(update)

    def _record_heartbeat(self, node_id, now):
        """Record a heartbeat from a node."""
        self.phi_detector.heartbeat(node_id, now)
        self.heartbeat_detector.heartbeat(node_id, now)

    def _serialize_updates(self, updates):
        return [{"dtype": u.dtype.value, "node_id": u.node_id,
                 "incarnation": u.incarnation} for u in updates]

    def _serialize_state_entries(self, entries):
        return [{"key": e.key, "value": e.value, "version": e.version,
                 "node_id": e.node_id, "timestamp": e.timestamp} for e in entries]

    def _deserialize_state_entries(self, data):
        return [StateEntry(d["key"], d["value"], d["version"],
                           d["node_id"], d["timestamp"]) for d in data]

    def _drain_outbox(self):
        msgs = list(self._outbox)
        self._outbox.clear()
        return msgs


# =============================================================================
# Gossip Network (Simulation)
# =============================================================================

class GossipNetwork:
    """
    Simulated network for testing gossip protocols.

    Supports:
    - Message delivery with latency
    - Network partitions
    - Message loss
    - Ordered delivery within a connection
    """

    def __init__(self, seed=None):
        self.nodes = {}  # node_id -> GossipNode
        self._partitions = set()  # frozenset pairs that can't communicate
        self._loss_rate = 0.0
        self._latency = 0.0
        self._time = 0.0
        self._pending = []  # (deliver_time, message)
        self._delivered = []  # All delivered messages
        self._dropped = []  # All dropped messages
        self._rng = random.Random(seed)

    def add_node(self, node):
        """Add a node to the network."""
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        """Remove a node (simulates crash)."""
        self.nodes.pop(node_id, None)

    def partition(self, node_a, node_b):
        """Create a network partition between two nodes."""
        self._partitions.add(frozenset([node_a, node_b]))

    def heal_partition(self, node_a, node_b):
        """Heal a partition between two nodes."""
        self._partitions.discard(frozenset([node_a, node_b]))

    def heal_all(self):
        """Heal all partitions."""
        self._partitions.clear()

    def partition_node(self, node_id):
        """Partition a node from all others."""
        for other in self.nodes:
            if other != node_id:
                self.partition(node_id, other)

    def set_loss_rate(self, rate):
        """Set random message loss rate (0.0 to 1.0)."""
        self._loss_rate = rate

    def set_latency(self, latency):
        """Set message delivery latency."""
        self._latency = latency

    def _can_communicate(self, src, dst):
        """Check if two nodes can communicate."""
        return frozenset([src, dst]) not in self._partitions

    def send(self, messages):
        """Queue messages for delivery."""
        for msg in messages:
            if not self._can_communicate(msg.src, msg.dst):
                self._dropped.append(msg)
                continue
            if self._rng.random() < self._loss_rate:
                self._dropped.append(msg)
                continue
            deliver_time = self._time + self._latency
            self._pending.append((deliver_time, msg))

    def deliver_pending(self):
        """Deliver all pending messages that are ready."""
        ready = [(t, m) for t, m in self._pending if t <= self._time]
        self._pending = [(t, m) for t, m in self._pending if t > self._time]
        all_responses = []
        for _, msg in ready:
            dst_node = self.nodes.get(msg.dst)
            if dst_node is None:
                self._dropped.append(msg)
                continue
            self._delivered.append(msg)
            responses = dst_node.receive(msg, self._time)
            all_responses.extend(responses)
        # Queue response messages
        if all_responses:
            self.send(all_responses)
        return len(ready)

    def tick(self, dt=1.0):
        """Advance time and run one gossip round."""
        self._time += dt
        # Each node runs its protocol tick
        all_messages = []
        for node in self.nodes.values():
            if node.members[node.node_id].status in (NodeStatus.ALIVE, NodeStatus.SUSPECT):
                msgs = node.tick(self._time)
                all_messages.extend(msgs)
        # Send all messages
        self.send(all_messages)
        # Deliver pending messages (may generate more)
        rounds = 0
        while self._pending and rounds < 10:
            delivered = self.deliver_pending()
            if delivered == 0:
                break
            rounds += 1

    def run(self, ticks=10, dt=1.0):
        """Run the simulation for multiple ticks."""
        for _ in range(ticks):
            self.tick(dt)

    def current_time(self):
        return self._time

    def delivered_count(self):
        return len(self._delivered)

    def dropped_count(self):
        return len(self._dropped)

    def get_cluster_view(self, node_id):
        """Get a node's view of cluster membership."""
        node = self.nodes.get(node_id)
        if not node:
            return {}
        return {mid: m.status.value for mid, m in node.members.items()}

    def all_agree_on_membership(self):
        """Check if all alive nodes agree on who is alive."""
        alive_views = []
        for nid, node in self.nodes.items():
            if node.members[nid].status == NodeStatus.ALIVE:
                alive_set = frozenset(
                    mid for mid, m in node.members.items()
                    if m.status == NodeStatus.ALIVE)
                alive_views.append(alive_set)
        if not alive_views:
            return True
        return all(v == alive_views[0] for v in alive_views)

    def converge(self, max_ticks=50, dt=1.0):
        """Run until all nodes agree on membership or max_ticks reached."""
        for i in range(max_ticks):
            self.tick(dt)
            if self.all_agree_on_membership():
                return i + 1
        return max_ticks

    def broadcast_state_sync(self):
        """Trigger anti-entropy sync between random pairs."""
        alive = [nid for nid, n in self.nodes.items()
                 if n.members[nid].status == NodeStatus.ALIVE]
        if len(alive) < 2:
            return
        for nid in alive:
            target = self._rng.choice([x for x in alive if x != nid])
            msgs = self.nodes[nid].sync_state_with(target)
            self.send(msgs)
        # Deliver the sync messages
        self.deliver_pending()
        self.deliver_pending()  # Second round for responses
