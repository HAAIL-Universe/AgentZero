"""
C233: Gossip Protocol

A gossip (epidemic) protocol implementation for distributed information dissemination.
Implements multiple gossip strategies with membership, failure detection, and
application-layer dissemination.

Components:
1. GossipNode -- individual node with state, rumor buffer, membership view
2. GossipCluster -- simulation harness for a cluster of nodes
3. Membership protocol -- SWIM-style failure detection (ping, ping-req, suspect, confirm)
4. Dissemination -- push, pull, push-pull gossip for key-value state
5. Anti-entropy -- Merkle-tree-based state reconciliation
6. Rumor mongering -- probabilistic broadcast with fan-out control
"""

import hashlib
import math
import random
import time
from collections import defaultdict
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeState(Enum):
    ALIVE = auto()
    SUSPECT = auto()
    DEAD = auto()
    LEFT = auto()


class GossipStrategy(Enum):
    PUSH = auto()
    PULL = auto()
    PUSH_PULL = auto()


class MessageType(Enum):
    PING = auto()
    PING_REQ = auto()
    ACK = auto()
    SUSPECT = auto()
    ALIVE = auto()
    CONFIRM_DEAD = auto()
    JOIN = auto()
    LEAVE = auto()
    PUSH_STATE = auto()
    PULL_REQUEST = auto()
    PULL_RESPONSE = auto()
    RUMOR = auto()
    ANTI_ENTROPY_REQ = auto()
    ANTI_ENTROPY_RESP = auto()


# ---------------------------------------------------------------------------
# Merkle Tree for anti-entropy
# ---------------------------------------------------------------------------

class MerkleNode:
    """Node in a Merkle hash tree for state comparison."""
    __slots__ = ('hash', 'left', 'right', 'key', 'value')

    def __init__(self, hash_val, left=None, right=None, key=None, value=None):
        self.hash = hash_val
        self.left = left
        self.right = right
        self.key = key
        self.value = value


def _hash(data):
    return hashlib.sha256(str(data).encode()).hexdigest()[:16]


def build_merkle_tree(state):
    """Build a Merkle tree from a dict of key-value pairs."""
    if not state:
        return MerkleNode(_hash("empty"))

    sorted_items = sorted(state.items(), key=lambda x: str(x[0]))
    leaves = [MerkleNode(_hash(f"{k}:{v}"), key=k, value=v) for k, v in sorted_items]

    if len(leaves) == 1:
        return leaves[0]

    # Build tree bottom-up
    level = leaves
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                combined = _hash(level[i].hash + level[i + 1].hash)
                next_level.append(MerkleNode(combined, level[i], level[i + 1]))
            else:
                next_level.append(level[i])
        level = next_level

    return level[0]


def diff_merkle_trees(local_tree, remote_tree):
    """Find keys that differ between two Merkle trees.
    Returns set of keys that need syncing."""
    if local_tree is None and remote_tree is None:
        return set()
    if local_tree is None or remote_tree is None:
        keys = set()
        _collect_keys(local_tree or remote_tree, keys)
        return keys
    if local_tree.hash == remote_tree.hash:
        return set()

    # Leaf nodes
    if local_tree.key is not None or remote_tree.key is not None:
        keys = set()
        if local_tree.key is not None:
            keys.add(local_tree.key)
        if remote_tree.key is not None:
            keys.add(remote_tree.key)
        return keys

    diffs = set()
    diffs |= diff_merkle_trees(local_tree.left, remote_tree.left)
    diffs |= diff_merkle_trees(local_tree.right, remote_tree.right)
    return diffs


def _collect_keys(node, keys):
    if node is None:
        return
    if node.key is not None:
        keys.add(node.key)
    _collect_keys(node.left, keys)
    _collect_keys(node.right, keys)


# ---------------------------------------------------------------------------
# Version Vector for conflict resolution
# ---------------------------------------------------------------------------

class VersionVector:
    """Vector clock for tracking causal ordering of updates."""

    def __init__(self):
        self.clock = {}

    def increment(self, node_id):
        self.clock[node_id] = self.clock.get(node_id, 0) + 1

    def merge(self, other):
        for k, v in other.clock.items():
            self.clock[k] = max(self.clock.get(k, 0), v)

    def dominates(self, other):
        """True if self >= other on all entries and > on at least one."""
        if not other.clock:
            return bool(self.clock)
        all_ge = all(self.clock.get(k, 0) >= v for k, v in other.clock.items())
        any_gt = any(self.clock.get(k, 0) > v for k, v in other.clock.items()) or \
                 any(k not in other.clock for k in self.clock)
        return all_ge and any_gt

    def concurrent(self, other):
        """True if neither dominates the other and they're not equal."""
        if self.clock == other.clock:
            return False
        return not self.dominates(other) and not other.dominates(self)

    def copy(self):
        vv = VersionVector()
        vv.clock = dict(self.clock)
        return vv

    def __eq__(self, other):
        return isinstance(other, VersionVector) and self.clock == other.clock

    def __repr__(self):
        return f"VV({self.clock})"


# ---------------------------------------------------------------------------
# Versioned Value
# ---------------------------------------------------------------------------

class VersionedValue:
    """A value with a version vector for conflict resolution."""

    def __init__(self, value, version=None, timestamp=None):
        self.value = value
        self.version = version or VersionVector()
        self.timestamp = timestamp or time.time()

    def copy(self):
        return VersionedValue(self.value, self.version.copy(), self.timestamp)


# ---------------------------------------------------------------------------
# Gossip Message
# ---------------------------------------------------------------------------

class GossipMessage:
    __slots__ = ('type', 'sender', 'target', 'payload', 'seq')
    _seq_counter = 0

    def __init__(self, msg_type, sender, target=None, payload=None):
        self.type = msg_type
        self.sender = sender
        self.target = target
        self.payload = payload or {}
        GossipMessage._seq_counter += 1
        self.seq = GossipMessage._seq_counter

    def __repr__(self):
        return f"Msg({self.type.name}, {self.sender}->{self.target}, seq={self.seq})"


# ---------------------------------------------------------------------------
# GossipNode
# ---------------------------------------------------------------------------

class GossipNode:
    """A single node in a gossip cluster."""

    def __init__(self, node_id, config=None):
        self.node_id = node_id
        self.config = config or {}

        # Membership
        self.state = NodeState.ALIVE
        self.members = {node_id: NodeState.ALIVE}  # node_id -> NodeState
        self.incarnation = 0  # for refuting suspicion
        self.member_incarnations = {node_id: 0}

        # Application state (key -> VersionedValue)
        self.store = {}

        # Failure detection
        self.suspect_timers = {}  # node_id -> rounds remaining
        self.ping_targets = []
        self.pending_pings = {}  # seq -> (target, indirect_requestors)
        self.ping_req_targets = {}  # seq -> original_target
        self.awaiting_ack = {}  # target -> rounds_waiting
        self.awaiting_ping_req_ack = {}  # target -> rounds_waiting

        # Rumor mongering
        self.rumors = {}  # rumor_id -> {data, infected_count, max_infections}
        self.seen_rumors = set()

        # Protocol state
        self.outbox = []  # messages to send
        self.round = 0

        # Stats
        self.messages_sent = 0
        self.messages_received = 0
        self.state_updates = 0

        # Config defaults
        self.fanout = self.config.get('fanout', 3)
        self.suspect_timeout = self.config.get('suspect_timeout', 3)
        self.rumor_max_infections = self.config.get('rumor_max_infections', 3)
        self.ping_req_count = self.config.get('ping_req_count', 2)

    # -- Membership ---------------------------------------------------------

    def join(self, seed_nodes):
        """Join the cluster by contacting seed nodes."""
        for seed in seed_nodes:
            if seed != self.node_id:
                msg = GossipMessage(MessageType.JOIN, self.node_id, seed,
                                    {'incarnation': self.incarnation})
                self.outbox.append(msg)
                self.messages_sent += 1

    def leave(self):
        """Gracefully leave the cluster."""
        self.state = NodeState.LEFT
        self.members[self.node_id] = NodeState.LEFT
        for member in list(self.members):
            if member != self.node_id and self.members[member] == NodeState.ALIVE:
                msg = GossipMessage(MessageType.LEAVE, self.node_id, member,
                                    {'node': self.node_id})
                self.outbox.append(msg)
                self.messages_sent += 1

    def get_alive_members(self):
        """Return list of alive members excluding self."""
        return [m for m, s in self.members.items()
                if s == NodeState.ALIVE and m != self.node_id]

    # -- State operations ---------------------------------------------------

    def put(self, key, value):
        """Set a key-value pair in local state."""
        if key in self.store:
            vv = self.store[key].version.copy()
        else:
            vv = VersionVector()
        vv.increment(self.node_id)
        self.store[key] = VersionedValue(value, vv)
        self.state_updates += 1

    def get(self, key):
        """Get a value from local state."""
        if key in self.store:
            return self.store[key].value
        return None

    def get_versioned(self, key):
        """Get the full versioned value."""
        return self.store.get(key)

    # -- Gossip round -------------------------------------------------------

    def gossip_round(self, strategy=GossipStrategy.PUSH_PULL):
        """Execute one round of gossip dissemination."""
        self.round += 1
        alive = self.get_alive_members()
        if not alive:
            return

        targets = random.sample(alive, min(self.fanout, len(alive)))

        for target in targets:
            if strategy == GossipStrategy.PUSH:
                self._push_state(target)
            elif strategy == GossipStrategy.PULL:
                self._pull_request(target)
            elif strategy == GossipStrategy.PUSH_PULL:
                self._push_state(target)
                self._pull_request(target)

    def _push_state(self, target):
        """Push local state to target."""
        state_snapshot = {}
        for k, vv in self.store.items():
            state_snapshot[k] = {
                'value': vv.value,
                'clock': dict(vv.version.clock),
                'timestamp': vv.timestamp
            }
        msg = GossipMessage(MessageType.PUSH_STATE, self.node_id, target,
                            {'state': state_snapshot})
        self.outbox.append(msg)
        self.messages_sent += 1

    def _pull_request(self, target):
        """Request state from target."""
        msg = GossipMessage(MessageType.PULL_REQUEST, self.node_id, target)
        self.outbox.append(msg)
        self.messages_sent += 1

    # -- Failure detection (SWIM-style) ------------------------------------

    def failure_detection_round(self):
        """Run one round of SWIM failure detection."""
        self.round += 1

        # Advance suspect timers
        expired = []
        for node_id in list(self.suspect_timers):
            self.suspect_timers[node_id] -= 1
            if self.suspect_timers[node_id] <= 0:
                expired.append(node_id)

        for node_id in expired:
            del self.suspect_timers[node_id]
            self._confirm_dead(node_id)

        # Check for unanswered direct pings -> escalate to ping_req
        escalate = []
        for target in list(self.awaiting_ack):
            self.awaiting_ack[target] -= 1
            if self.awaiting_ack[target] <= 0:
                escalate.append(target)
                del self.awaiting_ack[target]

        for target in escalate:
            self._send_ping_req(target)

        # Check for unanswered indirect pings -> suspect
        suspect_targets = []
        for target in list(self.awaiting_ping_req_ack):
            self.awaiting_ping_req_ack[target] -= 1
            if self.awaiting_ping_req_ack[target] <= 0:
                suspect_targets.append(target)
                del self.awaiting_ping_req_ack[target]

        for target in suspect_targets:
            self._suspect_node(target)

        # Pick a random alive member to ping
        alive = self.get_alive_members()
        suspects = [m for m, s in self.members.items()
                    if s == NodeState.SUSPECT and m != self.node_id]
        pingable = alive + suspects

        if not pingable:
            return

        target = random.choice(pingable)
        self._send_ping(target)

    def _send_ping(self, target):
        msg = GossipMessage(MessageType.PING, self.node_id, target)
        self.pending_pings[msg.seq] = (target, [])
        self.awaiting_ack[target] = 1  # expect ack within 1 round
        self.outbox.append(msg)
        self.messages_sent += 1

    def _send_ping_req(self, failed_target):
        """Send indirect ping requests through other members."""
        alive = self.get_alive_members()
        others = [m for m in alive if m != failed_target]
        if not others:
            self._suspect_node(failed_target)
            return

        self.awaiting_ping_req_ack[failed_target] = 1  # expect indirect ack within 1 round
        helpers = random.sample(others, min(self.ping_req_count, len(others)))
        for helper in helpers:
            msg = GossipMessage(MessageType.PING_REQ, self.node_id, helper,
                                {'target': failed_target})
            self.outbox.append(msg)
            self.messages_sent += 1

    def _suspect_node(self, node_id):
        if node_id == self.node_id:
            return
        if self.members.get(node_id) in (NodeState.DEAD, NodeState.LEFT):
            return
        self.members[node_id] = NodeState.SUSPECT
        self.suspect_timers[node_id] = self.suspect_timeout

        # Broadcast suspicion (including to the suspect so it can refute)
        targets = self.get_alive_members()
        if node_id not in targets:
            targets.append(node_id)
        for member in targets:
            msg = GossipMessage(MessageType.SUSPECT, self.node_id, member,
                                {'node': node_id,
                                 'incarnation': self.member_incarnations.get(node_id, 0)})
            self.outbox.append(msg)
            self.messages_sent += 1

    def _confirm_dead(self, node_id):
        if node_id == self.node_id:
            return
        self.members[node_id] = NodeState.DEAD

        for member in self.get_alive_members():
            msg = GossipMessage(MessageType.CONFIRM_DEAD, self.node_id, member,
                                {'node': node_id})
            self.outbox.append(msg)
            self.messages_sent += 1

    # -- Rumor mongering ---------------------------------------------------

    def spread_rumor(self, rumor_id, data):
        """Start spreading a rumor through the cluster."""
        if rumor_id in self.seen_rumors:
            return
        self.seen_rumors.add(rumor_id)
        self.rumors[rumor_id] = {
            'data': data,
            'infected_count': 0,
            'max_infections': self.rumor_max_infections
        }
        self._propagate_rumor(rumor_id)

    def _propagate_rumor(self, rumor_id):
        """Forward rumor to random members."""
        if rumor_id not in self.rumors:
            return
        rumor = self.rumors[rumor_id]
        if rumor['infected_count'] >= rumor['max_infections']:
            return

        alive = self.get_alive_members()
        if not alive:
            return

        targets = random.sample(alive, min(self.fanout, len(alive)))
        for target in targets:
            msg = GossipMessage(MessageType.RUMOR, self.node_id, target,
                                {'rumor_id': rumor_id, 'data': rumor['data']})
            self.outbox.append(msg)
            self.messages_sent += 1

        rumor['infected_count'] += 1

    # -- Anti-entropy ------------------------------------------------------

    def anti_entropy_round(self):
        """Run anti-entropy using Merkle tree comparison."""
        alive = self.get_alive_members()
        if not alive:
            return

        target = random.choice(alive)
        tree = build_merkle_tree({k: v.value for k, v in self.store.items()})
        msg = GossipMessage(MessageType.ANTI_ENTROPY_REQ, self.node_id, target,
                            {'root_hash': tree.hash,
                             'state_hashes': {k: _hash(f"{k}:{v.value}")
                                              for k, v in self.store.items()}})
        self.outbox.append(msg)
        self.messages_sent += 1

    # -- Message handling --------------------------------------------------

    def receive(self, message):
        """Process an incoming gossip message."""
        self.messages_received += 1
        handler = {
            MessageType.PING: self._handle_ping,
            MessageType.PING_REQ: self._handle_ping_req,
            MessageType.ACK: self._handle_ack,
            MessageType.SUSPECT: self._handle_suspect,
            MessageType.ALIVE: self._handle_alive,
            MessageType.CONFIRM_DEAD: self._handle_confirm_dead,
            MessageType.JOIN: self._handle_join,
            MessageType.LEAVE: self._handle_leave,
            MessageType.PUSH_STATE: self._handle_push_state,
            MessageType.PULL_REQUEST: self._handle_pull_request,
            MessageType.PULL_RESPONSE: self._handle_pull_response,
            MessageType.RUMOR: self._handle_rumor,
            MessageType.ANTI_ENTROPY_REQ: self._handle_anti_entropy_req,
            MessageType.ANTI_ENTROPY_RESP: self._handle_anti_entropy_resp,
        }.get(message.type)

        if handler:
            handler(message)

    def _handle_ping(self, msg):
        ack = GossipMessage(MessageType.ACK, self.node_id, msg.sender,
                            {'reply_to': msg.seq})
        self.outbox.append(ack)
        self.messages_sent += 1
        # Mark sender as alive
        self._mark_alive(msg.sender)

    def _handle_ping_req(self, msg):
        target = msg.payload.get('target')
        if target:
            # Forward ping to target, remember who asked
            fwd = GossipMessage(MessageType.PING, self.node_id, target,
                                {'on_behalf_of': msg.sender})
            self.pending_pings[fwd.seq] = (target, [msg.sender])
            self.outbox.append(fwd)
            self.messages_sent += 1

    def _handle_ack(self, msg):
        reply_to = msg.payload.get('reply_to')
        # Mark sender alive
        self._mark_alive(msg.sender)
        # Clear awaiting ack for this sender
        self.awaiting_ack.pop(msg.sender, None)
        self.awaiting_ping_req_ack.pop(msg.sender, None)

        # Check if this was an indirect ping
        if reply_to in self.pending_pings:
            target, requestors = self.pending_pings.pop(reply_to)
            for req in requestors:
                fwd_ack = GossipMessage(MessageType.ACK, self.node_id, req,
                                        {'reply_to': reply_to, 'target': target})
                self.outbox.append(fwd_ack)
                self.messages_sent += 1

        # Also check on_behalf_of acks
        target_node = msg.payload.get('target')
        if target_node:
            self._mark_alive(target_node)
            self.awaiting_ack.pop(target_node, None)
            self.awaiting_ping_req_ack.pop(target_node, None)

    def _handle_suspect(self, msg):
        node_id = msg.payload.get('node')
        inc = msg.payload.get('incarnation', 0)

        if node_id == self.node_id:
            # Refute: bump incarnation and announce alive
            if inc >= self.incarnation:
                self.incarnation = inc + 1
                self.member_incarnations[self.node_id] = self.incarnation
                for member in self.get_alive_members():
                    alive_msg = GossipMessage(MessageType.ALIVE, self.node_id, member,
                                              {'node': self.node_id,
                                               'incarnation': self.incarnation})
                    self.outbox.append(alive_msg)
                    self.messages_sent += 1
            return

        current_inc = self.member_incarnations.get(node_id, 0)
        if inc >= current_inc and self.members.get(node_id) == NodeState.ALIVE:
            self.members[node_id] = NodeState.SUSPECT
            self.member_incarnations[node_id] = inc
            self.suspect_timers[node_id] = self.suspect_timeout

    def _handle_alive(self, msg):
        node_id = msg.payload.get('node')
        inc = msg.payload.get('incarnation', 0)
        current_inc = self.member_incarnations.get(node_id, 0)

        if inc > current_inc:
            self._mark_alive(node_id)
            self.member_incarnations[node_id] = inc
            # Cancel suspect timer
            self.suspect_timers.pop(node_id, None)

    def _handle_confirm_dead(self, msg):
        node_id = msg.payload.get('node')
        if node_id and node_id != self.node_id:
            self.members[node_id] = NodeState.DEAD
            self.suspect_timers.pop(node_id, None)

    def _handle_join(self, msg):
        sender = msg.sender
        inc = msg.payload.get('incarnation', 0)
        self.members[sender] = NodeState.ALIVE
        self.member_incarnations[sender] = inc

        # Send back membership list
        ack = GossipMessage(MessageType.ACK, self.node_id, sender,
                            {'reply_to': msg.seq,
                             'members': {k: v.name for k, v in self.members.items()},
                             'incarnations': dict(self.member_incarnations)})
        self.outbox.append(ack)
        self.messages_sent += 1

    def _handle_leave(self, msg):
        node_id = msg.payload.get('node')
        if node_id:
            self.members[node_id] = NodeState.LEFT
            self.suspect_timers.pop(node_id, None)

    def _handle_push_state(self, msg):
        remote_state = msg.payload.get('state', {})
        self._merge_remote_state(remote_state)

    def _handle_pull_request(self, msg):
        state_snapshot = {}
        for k, vv in self.store.items():
            state_snapshot[k] = {
                'value': vv.value,
                'clock': dict(vv.version.clock),
                'timestamp': vv.timestamp
            }
        resp = GossipMessage(MessageType.PULL_RESPONSE, self.node_id, msg.sender,
                             {'state': state_snapshot})
        self.outbox.append(resp)
        self.messages_sent += 1

    def _handle_pull_response(self, msg):
        remote_state = msg.payload.get('state', {})
        self._merge_remote_state(remote_state)

    def _handle_rumor(self, msg):
        rumor_id = msg.payload.get('rumor_id')
        data = msg.payload.get('data')
        if rumor_id and rumor_id not in self.seen_rumors:
            self.spread_rumor(rumor_id, data)

    def _handle_anti_entropy_req(self, msg):
        remote_hashes = msg.payload.get('state_hashes', {})
        local_hashes = {k: _hash(f"{k}:{v.value}") for k, v in self.store.items()}

        # Find differences
        diff_keys = set()
        all_keys = set(remote_hashes.keys()) | set(local_hashes.keys())
        for k in all_keys:
            if remote_hashes.get(k) != local_hashes.get(k):
                diff_keys.add(k)

        # Send our values for differing keys
        diff_state = {}
        for k in diff_keys:
            if k in self.store:
                vv = self.store[k]
                diff_state[k] = {
                    'value': vv.value,
                    'clock': dict(vv.version.clock),
                    'timestamp': vv.timestamp
                }

        resp = GossipMessage(MessageType.ANTI_ENTROPY_RESP, self.node_id, msg.sender,
                             {'diff_state': diff_state, 'diff_keys': list(diff_keys)})
        self.outbox.append(resp)
        self.messages_sent += 1

    def _handle_anti_entropy_resp(self, msg):
        diff_state = msg.payload.get('diff_state', {})
        diff_keys = msg.payload.get('diff_keys', [])
        self._merge_remote_state(diff_state)

        # Push our state for keys the remote is missing
        push_state = {}
        for k in diff_keys:
            if k in self.store and k not in diff_state:
                vv = self.store[k]
                push_state[k] = {
                    'value': vv.value,
                    'clock': dict(vv.version.clock),
                    'timestamp': vv.timestamp
                }
        if push_state:
            push_msg = GossipMessage(MessageType.PUSH_STATE, self.node_id, msg.sender,
                                     {'state': push_state})
            self.outbox.append(push_msg)
            self.messages_sent += 1

    # -- Internal helpers --------------------------------------------------

    def _mark_alive(self, node_id):
        if node_id != self.node_id:
            if self.members.get(node_id) != NodeState.LEFT:
                self.members[node_id] = NodeState.ALIVE
            self.suspect_timers.pop(node_id, None)

    def _merge_remote_state(self, remote_state):
        """Merge remote state using version vectors for conflict resolution."""
        for key, entry in remote_state.items():
            remote_vv = VersionVector()
            remote_vv.clock = entry.get('clock', {})
            remote_val = entry.get('value')
            remote_ts = entry.get('timestamp', 0)

            if key not in self.store:
                # We don't have it -- accept
                self.store[key] = VersionedValue(remote_val, remote_vv, remote_ts)
                self.state_updates += 1
            else:
                local = self.store[key]
                if remote_vv.dominates(local.version):
                    # Remote is newer
                    self.store[key] = VersionedValue(remote_val, remote_vv, remote_ts)
                    self.state_updates += 1
                elif local.version.dominates(remote_vv):
                    # Local is newer -- keep ours
                    pass
                elif remote_vv.concurrent(local.version):
                    # Conflict -- last-writer-wins by timestamp
                    if remote_ts > local.timestamp:
                        merged_vv = local.version.copy()
                        merged_vv.merge(remote_vv)
                        self.store[key] = VersionedValue(remote_val, merged_vv, remote_ts)
                        self.state_updates += 1
                    elif remote_ts == local.timestamp:
                        # Tie-break by value comparison
                        merged_vv = local.version.copy()
                        merged_vv.merge(remote_vv)
                        if str(remote_val) > str(local.value):
                            self.store[key] = VersionedValue(remote_val, merged_vv, remote_ts)
                        else:
                            self.store[key] = VersionedValue(local.value, merged_vv, local.timestamp)
                        self.state_updates += 1
                    else:
                        # Local wins -- merge version vectors
                        local.version.merge(remote_vv)


# ---------------------------------------------------------------------------
# GossipCluster -- simulation harness
# ---------------------------------------------------------------------------

class GossipCluster:
    """Simulates a cluster of gossip nodes with message passing."""

    def __init__(self, node_count=0, config=None):
        self.nodes = {}
        self.config = config or {}
        self.network_partitions = set()  # frozenset pairs of partitioned nodes
        self.dropped_messages = 0
        self.total_messages = 0
        self.message_log = []
        self.drop_rate = self.config.get('drop_rate', 0.0)
        self.round = 0

        for i in range(node_count):
            self.add_node(f"node_{i}")

        # Auto-join all initial nodes
        if node_count > 1:
            for nid in list(self.nodes):
                others = [n for n in self.nodes if n != nid]
                self.nodes[nid].join(others)
            self._deliver_all()

    def add_node(self, node_id):
        """Add a new node to the cluster."""
        node = GossipNode(node_id, self.config)
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id):
        """Remove a node (crash simulation)."""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def partition(self, group_a, group_b):
        """Create a network partition between two groups of nodes."""
        for a in group_a:
            for b in group_b:
                self.network_partitions.add(frozenset([a, b]))

    def heal_partition(self):
        """Remove all network partitions."""
        self.network_partitions.clear()

    def is_partitioned(self, node_a, node_b):
        return frozenset([node_a, node_b]) in self.network_partitions

    def gossip_round(self, strategy=GossipStrategy.PUSH_PULL):
        """Run one round of gossip across all nodes."""
        self.round += 1
        for node in list(self.nodes.values()):
            if node.state == NodeState.ALIVE:
                node.gossip_round(strategy)
        self._deliver_all()

    def failure_detection_round(self):
        """Run one round of failure detection across all nodes."""
        self.round += 1
        for node in list(self.nodes.values()):
            if node.state == NodeState.ALIVE:
                node.failure_detection_round()
        self._deliver_all()

    def anti_entropy_round(self):
        """Run one round of anti-entropy across all nodes."""
        self.round += 1
        for node in list(self.nodes.values()):
            if node.state == NodeState.ALIVE:
                node.anti_entropy_round()
        self._deliver_all()

    def _deliver_all(self):
        """Deliver all pending messages (with partition and drop simulation)."""
        max_iterations = 20  # prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            all_messages = []
            for node in list(self.nodes.values()):
                all_messages.extend(node.outbox)
                node.outbox = []

            if not all_messages:
                break

            for msg in all_messages:
                self.total_messages += 1
                self.message_log.append(msg)

                # Check partition
                if self.is_partitioned(msg.sender, msg.target):
                    self.dropped_messages += 1
                    continue

                # Check random drop
                if self.drop_rate > 0 and random.random() < self.drop_rate:
                    self.dropped_messages += 1
                    continue

                # Deliver
                if msg.target in self.nodes:
                    self.nodes[msg.target].receive(msg)

    def run_rounds(self, count, strategy=GossipStrategy.PUSH_PULL):
        """Run multiple gossip rounds."""
        for _ in range(count):
            self.gossip_round(strategy)

    def converged(self, key=None):
        """Check if all alive nodes have converged on state."""
        alive_nodes = [n for n in self.nodes.values() if n.state == NodeState.ALIVE]
        if len(alive_nodes) < 2:
            return True

        if key:
            values = [n.get(key) for n in alive_nodes]
            return len(set(str(v) for v in values)) <= 1
        else:
            # Check all keys
            all_keys = set()
            for n in alive_nodes:
                all_keys.update(n.store.keys())

            for k in all_keys:
                values = [n.get(k) for n in alive_nodes]
                if len(set(str(v) for v in values)) > 1:
                    return False
            return True

    def get_convergence_status(self):
        """Return detailed convergence status."""
        alive_nodes = [n for n in self.nodes.values() if n.state == NodeState.ALIVE]
        all_keys = set()
        for n in alive_nodes:
            all_keys.update(n.store.keys())

        status = {}
        for k in all_keys:
            values = {}
            for n in alive_nodes:
                v = n.get(k)
                values[n.node_id] = v
            unique = set(str(v) for v in values.values())
            status[k] = {
                'converged': len(unique) <= 1,
                'values': values,
                'unique_count': len(unique)
            }
        return status

    def stats(self):
        """Return cluster-wide statistics."""
        alive = sum(1 for n in self.nodes.values() if n.state == NodeState.ALIVE)
        suspect = sum(1 for n in self.nodes.values() if n.state == NodeState.SUSPECT)
        dead = sum(1 for n in self.nodes.values() if n.state == NodeState.DEAD)
        total_state_updates = sum(n.state_updates for n in self.nodes.values())

        return {
            'total_nodes': len(self.nodes),
            'alive': alive,
            'suspect': suspect,
            'dead': dead,
            'total_messages': self.total_messages,
            'dropped_messages': self.dropped_messages,
            'state_updates': total_state_updates,
            'rounds': self.round,
        }
