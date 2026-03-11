"""
C206: Distributed Key-Value Store
=================================
THE BIG COMPOSITION: Raft + Gossip + Vector Clocks + Consistent Hashing + HTTP

Architecture:
- Consistent Hashing: partitions keyspace across nodes
- Raft Consensus: strongly consistent replication per partition group
- Vector Clocks: versioning for conflict detection / eventual consistency mode
- Gossip Protocol: cluster membership, failure detection, metadata propagation
- HTTP Server: client-facing REST API

Modes:
- STRONG: reads/writes go through Raft leader for linearizability
- EVENTUAL: reads from any replica, writes propagate via anti-entropy + vector clocks
- SESSION: read-your-writes consistency via session tokens

Components:
1. Partition       -- a key range owned by a replica group
2. ReplicaGroup    -- Raft cluster managing one partition
3. ClusterNode     -- a physical node running multiple partition replicas
4. KVCluster       -- the full distributed system
5. KVClient        -- client with routing, retries, session tracking
6. KVHttpHandler   -- REST API adapter
"""

import sys
import os
import time
import json
import hashlib
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

# Add parent paths for imports
_challenges = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_challenges, 'C201_raft_consensus'))
sys.path.insert(0, os.path.join(_challenges, 'C203_gossip_protocol'))
sys.path.insert(0, os.path.join(_challenges, 'C204_vector_clocks'))
sys.path.insert(0, os.path.join(_challenges, 'C205_consistent_hashing'))

from raft import (
    RaftNode, RaftCluster, RaftLog, KeyValueStateMachine,
    Message as RaftMessage, MessageType as RaftMessageType,
    Role as RaftRole, LogEntry, Snapshot
)
from gossip import (
    GossipNode, GossipNetwork, GossipMessage, GossipState,
    NodeStatus, MessageType as GossipMessageType, Member
)
from vector_clocks import (
    VectorClock, VersionVector, StampedValue, CausalStore
)
from consistent_hashing import (
    HashRing, ReplicatedHashRing, RING_SIZE
)


# ============================================================
# Enums and Configuration
# ============================================================

class ConsistencyLevel(Enum):
    """Read/write consistency level."""
    ONE = 1          # Any single replica
    QUORUM = 2       # Majority of replicas
    ALL = 3          # All replicas must respond
    STRONG = 4       # Through Raft leader (linearizable)


class ConsistencyMode(Enum):
    """Cluster-wide consistency mode for a partition."""
    STRONG = auto()      # All ops through Raft
    EVENTUAL = auto()    # Vector clock + anti-entropy
    SESSION = auto()     # Read-your-writes via session tokens


class OperationType(Enum):
    """KV operation types."""
    GET = "get"
    PUT = "put"
    DELETE = "delete"
    CAS = "cas"          # Compare-and-swap
    SCAN = "scan"        # Range scan
    BATCH = "batch"      # Batch operations


class NodeState(Enum):
    """Physical node lifecycle."""
    JOINING = auto()
    ACTIVE = auto()
    LEAVING = auto()
    DEAD = auto()


@dataclass
class KVConfig:
    """Cluster configuration."""
    num_partitions: int = 16
    replication_factor: int = 3
    num_vnodes: int = 150
    default_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    default_mode: ConsistencyMode = ConsistencyMode.STRONG
    read_repair: bool = True
    hinted_handoff: bool = True
    anti_entropy_interval: float = 30.0
    gossip_interval: float = 1.0
    election_timeout_range: tuple = (150, 300)
    heartbeat_interval: int = 50
    max_hints_per_node: int = 1000
    hint_ttl: float = 3600.0
    session_ttl: float = 300.0
    tombstone_ttl: float = 86400.0


# ============================================================
# Data Types
# ============================================================

@dataclass
class KVEntry:
    """A stored value with metadata."""
    key: str
    value: Any
    version: VersionVector = field(default_factory=VersionVector)
    timestamp: float = 0.0
    tombstone: bool = False
    writer: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "version": self.version.versions,
            "timestamp": self.timestamp,
            "tombstone": self.tombstone,
            "writer": self.writer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KVEntry':
        vv = VersionVector()
        for node, count in d.get("version", {}).items():
            for _ in range(count):
                vv.increment(node)
        return cls(
            key=d["key"],
            value=d["value"],
            version=vv,
            timestamp=d.get("timestamp", 0.0),
            tombstone=d.get("tombstone", False),
            writer=d.get("writer", ""),
        )


@dataclass
class HintedHandoff:
    """Hinted handoff entry for unavailable nodes."""
    target_node: str
    entry: KVEntry
    created_at: float
    partition_id: int


@dataclass
class SessionToken:
    """Session token for read-your-writes consistency."""
    session_id: str
    node_id: str
    vector_clock: VectorClock = field(default_factory=VectorClock)
    last_access: float = 0.0

    def update(self, clock: VectorClock):
        merged = self.vector_clock.merge(clock)
        self.vector_clock = merged


@dataclass
class KVRequest:
    """Client request."""
    op: OperationType
    key: str = ""
    value: Any = None
    consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    session_id: str = ""
    cas_expected: Any = None
    scan_prefix: str = ""
    scan_limit: int = 100
    batch_ops: list = field(default_factory=list)
    request_id: str = ""
    timeout: float = 5.0


@dataclass
class KVResponse:
    """Response to client."""
    success: bool
    value: Any = None
    version: dict = field(default_factory=dict)
    error: str = ""
    conflicts: list = field(default_factory=list)
    session_id: str = ""
    request_id: str = ""
    node_id: str = ""

    def to_dict(self) -> dict:
        d = {"success": self.success}
        if self.value is not None or self.success:
            d["value"] = self.value
        if self.version:
            d["version"] = self.version
        if self.error:
            d["error"] = self.error
        if self.conflicts:
            d["conflicts"] = self.conflicts
        if self.session_id:
            d["session_id"] = self.session_id
        if self.request_id:
            d["request_id"] = self.request_id
        return d


# ============================================================
# Partition: manages a key range's data
# ============================================================

class Partition:
    """A logical partition of the keyspace."""

    def __init__(self, partition_id: int, node_id: str):
        self.partition_id = partition_id
        self.node_id = node_id
        self._data: dict[str, list[KVEntry]] = {}  # key -> siblings (for conflicts)
        self._version = VersionVector()
        self._tombstones: dict[str, float] = {}  # key -> tombstone creation time

    def get(self, key: str) -> list[KVEntry]:
        """Get all versions of a key (siblings if conflicts exist)."""
        entries = self._data.get(key, [])
        return [e for e in entries if not e.tombstone]

    def get_with_tombstones(self, key: str) -> list[KVEntry]:
        """Get including tombstones."""
        return list(self._data.get(key, []))

    def put(self, key: str, value: Any, writer: str = "",
            context_version: VersionVector = None, now: float = 0.0) -> KVEntry:
        """Write a value, resolving or creating siblings."""
        self._version.increment(self.node_id)

        entry = KVEntry(
            key=key,
            value=value,
            version=VersionVector(),
            timestamp=now or time.time(),
            writer=writer or self.node_id,
        )

        existing = self._data.get(key, [])

        if context_version is not None:
            # Client-provided context: discard entries dominated by context
            surviving = []
            for e in existing:
                if not context_version.descends_from(e.version) and not e.version.descends_from(context_version):
                    surviving.append(e)
            # Merge context into new entry's version
            merged_vv = context_version.copy()
            merged_vv.increment(self.node_id)
            entry.version = merged_vv
            surviving.append(entry)
            self._data[key] = surviving
        elif existing:
            # No context: merge with all existing
            merged_vv = VersionVector()
            for e in existing:
                merged_vv = merged_vv.merge(e.version)
            merged_vv.increment(self.node_id)
            entry.version = merged_vv
            self._data[key] = [entry]
        else:
            entry.version.increment(self.node_id)
            self._data[key] = [entry]

        # Clear tombstone if overwriting
        self._tombstones.pop(key, None)
        return entry

    def delete(self, key: str, writer: str = "", now: float = 0.0) -> Optional[KVEntry]:
        """Delete via tombstone."""
        existing = self._data.get(key, [])
        if not existing:
            return None

        self._version.increment(self.node_id)
        merged_vv = VersionVector()
        for e in existing:
            merged_vv = merged_vv.merge(e.version)
        merged_vv.increment(self.node_id)

        tombstone = KVEntry(
            key=key,
            value=None,
            version=merged_vv,
            timestamp=now or time.time(),
            tombstone=True,
            writer=writer or self.node_id,
        )
        self._data[key] = [tombstone]
        self._tombstones[key] = tombstone.timestamp
        return tombstone

    def merge_entry(self, entry: KVEntry) -> bool:
        """Merge a remote entry (for replication / anti-entropy)."""
        existing = self._data.get(entry.key, [])

        if not existing:
            self._data[entry.key] = [entry]
            if entry.tombstone:
                self._tombstones[entry.key] = entry.timestamp
            return True

        # Check if already dominated
        for e in existing:
            if e.version.descends_from(entry.version):
                return False  # Already have newer

        # Remove entries dominated by incoming
        surviving = []
        for e in existing:
            if not entry.version.descends_from(e.version):
                surviving.append(e)  # Concurrent -- keep as sibling

        # Check if incoming is truly new
        if len(surviving) == len(existing):
            # Nothing dominated -- check for exact duplicate
            for e in existing:
                if e.version == entry.version:
                    return False

        surviving.append(entry)
        self._data[entry.key] = surviving

        if entry.tombstone:
            self._tombstones[entry.key] = entry.timestamp
        return True

    def all_entries(self) -> list[KVEntry]:
        """All entries for anti-entropy sync."""
        result = []
        for entries in self._data.values():
            result.extend(entries)
        return result

    def keys(self) -> list[str]:
        """All non-tombstoned keys."""
        return [k for k, entries in self._data.items()
                if any(not e.tombstone for e in entries)]

    def size(self) -> int:
        return len(self.keys())

    def has_conflicts(self, key: str) -> bool:
        entries = self.get(key)
        return len(entries) > 1

    def resolve_conflicts(self, key: str, value: Any, writer: str = "",
                          now: float = 0.0) -> KVEntry:
        """Resolve all siblings into a single value."""
        existing = self._data.get(key, [])
        merged_vv = VersionVector()
        for e in existing:
            merged_vv = merged_vv.merge(e.version)
        merged_vv.increment(self.node_id)

        entry = KVEntry(
            key=key,
            value=value,
            version=merged_vv,
            timestamp=now or time.time(),
            writer=writer or self.node_id,
        )
        self._data[key] = [entry]
        return entry

    def gc_tombstones(self, ttl: float, now: float = 0.0):
        """Remove expired tombstones."""
        now = now or time.time()
        expired = [k for k, ts in self._tombstones.items() if now - ts > ttl]
        for k in expired:
            del self._tombstones[k]
            if k in self._data:
                entries = self._data[k]
                if all(e.tombstone for e in entries):
                    del self._data[k]

    def snapshot(self) -> dict:
        """Snapshot for Raft snapshots."""
        return {
            "partition_id": self.partition_id,
            "node_id": self.node_id,
            "data": {k: [e.to_dict() for e in entries]
                     for k, entries in self._data.items()},
            "version": self._version.versions,
        }

    def restore(self, data: dict):
        """Restore from snapshot."""
        self._data.clear()
        for k, entries in data.get("data", {}).items():
            self._data[k] = [KVEntry.from_dict(e) for e in entries]
        vv = VersionVector()
        for node, count in data.get("version", {}).items():
            for _ in range(count):
                vv.increment(node)
        self._version = vv


# ============================================================
# PartitionStateMachine: Raft state machine wrapping Partition
# ============================================================

class PartitionStateMachine:
    """State machine for Raft that wraps a Partition."""

    def __init__(self, partition_id: int, node_id: str):
        self.partition = Partition(partition_id, node_id)
        self._last_applied = 0

    def apply(self, command: dict, index: int) -> Any:
        """Apply a command from Raft log."""
        self._last_applied = index
        if command is None:
            return {"ok": True}  # No-op entry (e.g., leader election)
        op = command.get("op", "")
        key = command.get("key", "")
        value = command.get("value")
        writer = command.get("writer", "")
        now = command.get("timestamp", 0.0)

        if op == "put":
            entry = self.partition.put(key, value, writer=writer, now=now)
            return {"ok": True, "version": entry.version.versions}
        elif op == "get":
            entries = self.partition.get(key)
            if not entries:
                return {"ok": True, "value": None, "found": False}
            if len(entries) == 1:
                return {"ok": True, "value": entries[0].value,
                        "version": entries[0].version.versions, "found": True}
            return {"ok": True, "value": entries[0].value,
                    "conflicts": [e.to_dict() for e in entries],
                    "version": entries[0].version.versions, "found": True}
        elif op == "delete":
            entry = self.partition.delete(key, writer=writer, now=now)
            return {"ok": True, "deleted": entry is not None}
        elif op == "cas":
            expected = command.get("expected")
            entries = self.partition.get(key)
            current = entries[0].value if entries else None
            if current != expected:
                return {"ok": False, "error": "cas_failed",
                        "current": current, "expected": expected}
            entry = self.partition.put(key, value, writer=writer, now=now)
            return {"ok": True, "version": entry.version.versions}
        else:
            return {"ok": False, "error": f"unknown op: {op}"}

    def snapshot(self) -> dict:
        return self.partition.snapshot()

    def restore(self, data: dict):
        self.partition.restore(data)


# ============================================================
# ReplicaGroup: Raft cluster managing a single partition
# ============================================================

class ReplicaGroup:
    """A group of replicas for one partition, coordinated by Raft."""

    def __init__(self, partition_id: int, node_ids: list[str], config: KVConfig = None):
        self.partition_id = partition_id
        self.config = config or KVConfig()
        self.node_ids = list(node_ids)

        # Create Raft nodes with partition-aware state machines
        self._state_machines: dict[str, PartitionStateMachine] = {}
        self._raft_nodes: dict[str, RaftNode] = {}

        for nid in node_ids:
            sm = PartitionStateMachine(partition_id, nid)
            self._state_machines[nid] = sm
            peers = [p for p in node_ids if p != nid]
            node = RaftNode(
                nid, peers,
                state_machine=sm,
                election_timeout_range=self.config.election_timeout_range,
                heartbeat_interval=self.config.heartbeat_interval,
            )
            self._raft_nodes[nid] = node

        self._messages: list[RaftMessage] = []
        self._partitions: set[tuple] = set()

    @property
    def nodes(self) -> dict[str, RaftNode]:
        return dict(self._raft_nodes)

    def get_leader(self) -> Optional[str]:
        for nid, node in self._raft_nodes.items():
            if node.role == RaftRole.LEADER:
                return nid
        return None

    def get_state_machine(self, node_id: str) -> Optional[PartitionStateMachine]:
        return self._state_machines.get(node_id)

    def tick(self, ms: int = 1):
        """Advance time for all Raft nodes."""
        for node in self._raft_nodes.values():
            node.tick(ms)

    def deliver_messages(self):
        """Route messages between Raft nodes."""
        for node in self._raft_nodes.values():
            while node.outbox:
                msg = node.outbox.pop(0)
                pair = (msg.src, msg.dst)
                rpair = (msg.dst, msg.src)
                if pair in self._partitions or rpair in self._partitions:
                    continue
                if msg.dst in self._raft_nodes:
                    self._raft_nodes[msg.dst].receive(msg)

    def tick_until_leader(self, max_ticks: int = 5000) -> bool:
        for _ in range(max_ticks):
            self.tick(1)
            self.deliver_messages()
            if self.get_leader():
                return True
        return False

    def submit(self, command: dict, node_id: str = None) -> Optional[str]:
        """Submit command to leader (or specified node)."""
        target = node_id or self.get_leader()
        if not target or target not in self._raft_nodes:
            return None
        node = self._raft_nodes[target]
        if node.role != RaftRole.LEADER:
            return None
        # Create request
        req_id = f"req-{self.partition_id}-{id(command)}"
        msg = RaftMessage(
            type=RaftMessageType.CLIENT_REQUEST,
            src="client",
            dst=target,
            term=0,
            data={"command": command, "request_id": req_id},
        )
        node.receive(msg)
        self.deliver_messages()
        return req_id

    def submit_and_commit(self, command: dict, max_ticks: int = 3000) -> Any:
        """Submit and wait for commit."""
        leader = self.get_leader()
        if not leader:
            return {"ok": False, "error": "no_leader"}

        req_id = self.submit(command, leader)
        if not req_id:
            return {"ok": False, "error": "submit_failed"}

        node = self._raft_nodes[leader]
        # Single-node: advance commit immediately since no peers to replicate to
        if not node.peers:
            node._advance_commit_index()
        for _ in range(max_ticks):
            self.tick(1)
            self.deliver_messages()
            # Check committed_results
            if req_id in node.committed_results:
                return node.committed_results.pop(req_id)
        return {"ok": False, "error": "timeout"}

    def partition_nodes(self, group_a: list[str], group_b: list[str]):
        """Create network partition between groups."""
        for a in group_a:
            for b in group_b:
                self._partitions.add((a, b))

    def heal_partitions(self):
        self._partitions.clear()

    def read_local(self, node_id: str, key: str) -> list[KVEntry]:
        """Read directly from a node's partition (eventual consistency)."""
        sm = self._state_machines.get(node_id)
        if not sm:
            return []
        return sm.partition.get(key)


# ============================================================
# ClusterNode: a physical node in the cluster
# ============================================================

class ClusterNode:
    """Represents a physical node that hosts multiple partition replicas."""

    def __init__(self, node_id: str, address: str = ""):
        self.node_id = node_id
        self.address = address or f"{node_id}:8080"
        self.state = NodeState.ACTIVE
        self.partitions: dict[int, int] = {}  # partition_id -> role (0=follower, 1=leader)
        self.hints: list[HintedHandoff] = []
        self.metadata: dict[str, Any] = {}
        self._sessions: dict[str, SessionToken] = {}

    def add_partition(self, partition_id: int, is_leader: bool = False):
        self.partitions[partition_id] = 1 if is_leader else 0

    def remove_partition(self, partition_id: int):
        self.partitions.pop(partition_id, None)

    def store_hint(self, hint: HintedHandoff):
        if len(self.hints) < 1000:
            self.hints.append(hint)

    def get_hints_for(self, target_node: str) -> list[HintedHandoff]:
        return [h for h in self.hints if h.target_node == target_node]

    def remove_hints_for(self, target_node: str):
        self.hints = [h for h in self.hints if h.target_node != target_node]

    def get_or_create_session(self, session_id: str, now: float = 0.0) -> SessionToken:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionToken(
                session_id=session_id,
                node_id=self.node_id,
                last_access=now,
            )
        session = self._sessions[session_id]
        session.last_access = now
        return session

    def gc_sessions(self, ttl: float, now: float = 0.0):
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_access > ttl]
        for sid in expired:
            del self._sessions[sid]


# ============================================================
# KVCluster: the full distributed KV store
# ============================================================

class KVCluster:
    """
    Distributed Key-Value Store.

    Composes:
    - Consistent hashing for partitioning
    - Raft for strong consistency per partition
    - Vector clocks for versioning and conflict detection
    - Gossip for membership and metadata propagation
    """

    def __init__(self, node_ids: list[str], config: KVConfig = None):
        self.config = config or KVConfig()
        self._node_ids = list(node_ids)

        # Consistent hashing ring for partition assignment
        self._ring = ReplicatedHashRing(
            nodes=node_ids,
            num_replicas=self.config.num_vnodes,
            replication_factor=self.config.replication_factor,
        )

        # Physical nodes
        self._nodes: dict[str, ClusterNode] = {}
        for nid in node_ids:
            self._nodes[nid] = ClusterNode(nid)

        # Partition -> replica group mapping
        self._partition_map: dict[int, list[str]] = {}  # partition_id -> [node_ids]
        self._replica_groups: dict[int, ReplicaGroup] = {}

        # Gossip network for membership
        self._gossip_net = GossipNetwork()
        for nid in node_ids:
            gnode = GossipNode(nid, config={
                "protocol_period": self.config.gossip_interval,
                "suspect_timeout": 5.0,
            })
            self._gossip_net.add_node(gnode)

        # Cluster-wide vector clock for ordering
        self._cluster_clock = VectorClock()

        # Event log
        self._events: list[dict] = []

        # Assign partitions
        self._assign_partitions()

    def _assign_partitions(self):
        """Assign partitions to nodes using consistent hashing."""
        for pid in range(self.config.num_partitions):
            partition_key = f"partition-{pid}"
            replicas = self._ring.get_preference_list(partition_key)

            # Ensure we have enough replicas
            if len(replicas) < self.config.replication_factor:
                # Use available nodes
                replicas = list(replicas)
                for nid in self._node_ids:
                    if nid not in replicas:
                        replicas.append(nid)
                    if len(replicas) >= self.config.replication_factor:
                        break

            self._partition_map[pid] = replicas
            group = ReplicaGroup(pid, replicas, self.config)
            self._replica_groups[pid] = group

            # Track on physical nodes
            for i, nid in enumerate(replicas):
                if nid in self._nodes:
                    self._nodes[nid].add_partition(pid, is_leader=(i == 0))

    def _key_to_partition(self, key: str) -> int:
        """Map a key to its partition ID."""
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % self.config.num_partitions

    def _get_replica_group(self, key: str) -> ReplicaGroup:
        """Get the replica group responsible for a key."""
        pid = self._key_to_partition(key)
        return self._replica_groups[pid]

    def _get_partition_nodes(self, key: str) -> list[str]:
        """Get node IDs responsible for a key."""
        pid = self._key_to_partition(key)
        return self._partition_map.get(pid, [])

    # --- Cluster lifecycle ---

    def bootstrap(self, max_ticks: int = 5000) -> bool:
        """Bootstrap all partition groups (elect leaders)."""
        success = True
        for group in self._replica_groups.values():
            if not group.tick_until_leader(max_ticks):
                success = False
        return success

    def tick(self, ms: int = 1):
        """Advance time for all components."""
        for group in self._replica_groups.values():
            group.tick(ms)
            group.deliver_messages()

    def tick_n(self, n: int, ms: int = 1):
        """Tick multiple times."""
        for _ in range(n):
            self.tick(ms)

    def gossip_tick(self):
        """Run one gossip round."""
        self._gossip_net.tick()

    # --- Core operations ---

    def put(self, key: str, value: Any, consistency: ConsistencyLevel = None,
            writer: str = "", now: float = 0.0) -> KVResponse:
        """Put a key-value pair."""
        consistency = consistency or self.config.default_consistency
        group = self._get_replica_group(key)

        if consistency == ConsistencyLevel.STRONG:
            return self._strong_put(group, key, value, writer, now)
        else:
            return self._quorum_put(group, key, value, writer, consistency, now)

    def get(self, key: str, consistency: ConsistencyLevel = None,
            session_id: str = "") -> KVResponse:
        """Get a value by key."""
        consistency = consistency or self.config.default_consistency
        group = self._get_replica_group(key)

        if consistency == ConsistencyLevel.STRONG:
            return self._strong_get(group, key)
        elif session_id and self.config.default_mode == ConsistencyMode.SESSION:
            return self._session_get(group, key, session_id)
        else:
            return self._quorum_get(group, key, consistency)

    def delete(self, key: str, consistency: ConsistencyLevel = None,
               writer: str = "") -> KVResponse:
        """Delete a key."""
        consistency = consistency or self.config.default_consistency
        group = self._get_replica_group(key)

        if consistency == ConsistencyLevel.STRONG:
            return self._strong_delete(group, key, writer)
        else:
            return self._quorum_delete(group, key, writer, consistency)

    def cas(self, key: str, expected: Any, new_value: Any,
            writer: str = "") -> KVResponse:
        """Compare-and-swap (always strong consistency)."""
        group = self._get_replica_group(key)
        command = {"op": "cas", "key": key, "expected": expected,
                   "value": new_value, "writer": writer}
        result = group.submit_and_commit(command)
        if not result.get("ok"):
            return KVResponse(
                success=False,
                error=result.get("error", "cas_failed"),
                value=result.get("current"),
            )
        return KVResponse(success=True, version=result.get("version", {}))

    def scan(self, prefix: str = "", limit: int = 100,
             node_id: str = None) -> KVResponse:
        """Scan keys across partitions."""
        results = []
        for pid, group in self._replica_groups.items():
            # Read from leader or specified node
            target = node_id or group.get_leader()
            if not target:
                continue
            sm = group.get_state_machine(target)
            if not sm:
                continue
            for key in sm.partition.keys():
                if prefix and not key.startswith(prefix):
                    continue
                entries = sm.partition.get(key)
                if entries:
                    results.append({
                        "key": key,
                        "value": entries[0].value,
                        "version": entries[0].version.versions,
                    })
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        results.sort(key=lambda r: r["key"])
        return KVResponse(success=True, value=results[:limit])

    def batch(self, operations: list[dict]) -> KVResponse:
        """Execute batch operations."""
        results = []
        for op in operations:
            op_type = op.get("op", "put")
            if op_type == "put":
                resp = self.put(op["key"], op["value"])
            elif op_type == "get":
                resp = self.get(op["key"])
            elif op_type == "delete":
                resp = self.delete(op["key"])
            else:
                resp = KVResponse(success=False, error=f"unknown op: {op_type}")
            results.append(resp.to_dict())
        return KVResponse(success=True, value=results)

    # --- Strong consistency (through Raft) ---

    def _strong_put(self, group: ReplicaGroup, key: str, value: Any,
                    writer: str, now: float) -> KVResponse:
        command = {"op": "put", "key": key, "value": value,
                   "writer": writer, "timestamp": now}
        result = group.submit_and_commit(command)
        if not result.get("ok"):
            return KVResponse(success=False, error=result.get("error", "commit_failed"))
        return KVResponse(success=True, version=result.get("version", {}))

    def _strong_get(self, group: ReplicaGroup, key: str) -> KVResponse:
        command = {"op": "get", "key": key}
        result = group.submit_and_commit(command)
        if not result.get("ok"):
            return KVResponse(success=False, error=result.get("error", "read_failed"))
        if not result.get("found"):
            return KVResponse(success=True, value=None)
        resp = KVResponse(
            success=True,
            value=result.get("value"),
            version=result.get("version", {}),
        )
        if "conflicts" in result:
            resp.conflicts = result["conflicts"]
        return resp

    def _strong_delete(self, group: ReplicaGroup, key: str,
                       writer: str) -> KVResponse:
        command = {"op": "delete", "key": key, "writer": writer}
        result = group.submit_and_commit(command)
        if not result.get("ok"):
            return KVResponse(success=False, error=result.get("error", "delete_failed"))
        return KVResponse(success=True)

    # --- Quorum consistency ---

    def _quorum_put(self, group: ReplicaGroup, key: str, value: Any,
                    writer: str, consistency: ConsistencyLevel,
                    now: float) -> KVResponse:
        """Write to N replicas based on consistency level."""
        required = self._required_responses(consistency)
        successes = 0

        for nid in group.node_ids:
            sm = group.get_state_machine(nid)
            if sm:
                sm.partition.put(key, value, writer=writer or nid, now=now)
                successes += 1

        if successes >= required:
            return KVResponse(success=True)
        return KVResponse(success=False, error="insufficient_replicas")

    def _quorum_get(self, group: ReplicaGroup, key: str,
                    consistency: ConsistencyLevel) -> KVResponse:
        """Read from N replicas and merge."""
        required = self._required_responses(consistency)
        all_entries = []
        responding = 0

        for nid in group.node_ids:
            sm = group.get_state_machine(nid)
            if sm:
                entries = sm.partition.get(key)
                all_entries.extend(entries)
                responding += 1

        if responding < required:
            return KVResponse(success=False, error="insufficient_replicas")

        if not all_entries:
            return KVResponse(success=True, value=None)

        # Merge: keep latest / detect conflicts via version vectors
        merged = self._merge_entries(all_entries)

        if len(merged) == 1:
            e = merged[0]
            # Do read repair if enabled
            if self.config.read_repair and responding > 1:
                self._read_repair(group, e)
            return KVResponse(
                success=True,
                value=e.value,
                version=e.version.versions,
            )
        else:
            return KVResponse(
                success=True,
                value=merged[0].value,
                conflicts=[e.to_dict() for e in merged],
                version=merged[0].version.versions,
            )

    def _quorum_delete(self, group: ReplicaGroup, key: str,
                       writer: str, consistency: ConsistencyLevel) -> KVResponse:
        required = self._required_responses(consistency)
        successes = 0
        for nid in group.node_ids:
            sm = group.get_state_machine(nid)
            if sm:
                sm.partition.delete(key, writer=writer or nid)
                successes += 1
        if successes >= required:
            return KVResponse(success=True)
        return KVResponse(success=False, error="insufficient_replicas")

    # --- Session consistency ---

    def _session_get(self, group: ReplicaGroup, key: str,
                     session_id: str) -> KVResponse:
        """Read-your-writes: ensure we read from a replica that has our writes."""
        # Find node with session
        for nid in group.node_ids:
            if nid in self._nodes:
                node = self._nodes[nid]
                if session_id in node._sessions:
                    # This node has the session -- read from here
                    sm = group.get_state_machine(nid)
                    if sm:
                        entries = sm.partition.get(key)
                        if entries:
                            return KVResponse(
                                success=True,
                                value=entries[0].value,
                                version=entries[0].version.versions,
                                session_id=session_id,
                            )
                        return KVResponse(success=True, value=None,
                                          session_id=session_id)

        # Fallback to quorum read
        resp = self._quorum_get(group, key, ConsistencyLevel.QUORUM)
        resp.session_id = session_id
        return resp

    # --- Helpers ---

    def _required_responses(self, level: ConsistencyLevel) -> int:
        rf = self.config.replication_factor
        if level == ConsistencyLevel.ONE:
            return 1
        elif level == ConsistencyLevel.QUORUM:
            return (rf // 2) + 1
        elif level == ConsistencyLevel.ALL:
            return rf
        elif level == ConsistencyLevel.STRONG:
            return (rf // 2) + 1
        return 1

    def _merge_entries(self, entries: list[KVEntry]) -> list[KVEntry]:
        """Merge entries from multiple replicas, keeping concurrent siblings."""
        if not entries:
            return []

        merged = []
        for entry in entries:
            dominated = False
            new_merged = []
            for existing in merged:
                if existing.version.descends_from(entry.version):
                    dominated = True
                    new_merged.append(existing)
                elif entry.version.descends_from(existing.version):
                    # Entry dominates existing -- skip existing
                    continue
                else:
                    new_merged.append(existing)
            if not dominated:
                new_merged.append(entry)
            merged = new_merged

        return merged if merged else entries[:1]

    def _read_repair(self, group: ReplicaGroup, entry: KVEntry):
        """Propagate latest value to all replicas."""
        for nid in group.node_ids:
            sm = group.get_state_machine(nid)
            if sm:
                sm.partition.merge_entry(entry)

    # --- Hinted handoff ---

    def store_hint(self, source_node: str, target_node: str, entry: KVEntry,
                   partition_id: int):
        """Store a hint when target is unavailable."""
        if source_node in self._nodes and self.config.hinted_handoff:
            hint = HintedHandoff(
                target_node=target_node,
                entry=entry,
                created_at=time.time(),
                partition_id=partition_id,
            )
            self._nodes[source_node].store_hint(hint)

    def deliver_hints(self, source_node: str, target_node: str) -> int:
        """Deliver stored hints to a recovered node."""
        if source_node not in self._nodes:
            return 0
        node = self._nodes[source_node]
        hints = node.get_hints_for(target_node)
        delivered = 0

        for hint in hints:
            group = self._replica_groups.get(hint.partition_id)
            if group:
                sm = group.get_state_machine(target_node)
                if sm:
                    sm.partition.merge_entry(hint.entry)
                    delivered += 1

        node.remove_hints_for(target_node)
        return delivered

    # --- Anti-entropy ---

    def anti_entropy_sync(self, partition_id: int, node_a: str, node_b: str) -> int:
        """Synchronize two replicas of a partition."""
        group = self._replica_groups.get(partition_id)
        if not group:
            return 0

        sm_a = group.get_state_machine(node_a)
        sm_b = group.get_state_machine(node_b)
        if not sm_a or not sm_b:
            return 0

        synced = 0
        # A -> B
        for entry in sm_a.partition.all_entries():
            if sm_b.partition.merge_entry(entry):
                synced += 1
        # B -> A
        for entry in sm_b.partition.all_entries():
            if sm_a.partition.merge_entry(entry):
                synced += 1

        return synced

    def full_anti_entropy(self) -> int:
        """Run anti-entropy across all partitions."""
        total = 0
        for pid, nodes in self._partition_map.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    total += self.anti_entropy_sync(pid, nodes[i], nodes[j])
        return total

    # --- Membership ---

    def add_node(self, node_id: str) -> bool:
        """Add a new node to the cluster."""
        if node_id in self._nodes:
            return False

        self._nodes[node_id] = ClusterNode(node_id)
        self._node_ids.append(node_id)
        self._ring.add_node(node_id)

        # Add gossip node
        gnode = GossipNode(node_id)
        self._gossip_net.add_node(gnode)

        # Rebalance: some partitions get new replicas
        self._rebalance_after_add(node_id)

        self._events.append({"type": "node_added", "node": node_id})
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster."""
        if node_id not in self._nodes:
            return False

        self._nodes[node_id].state = NodeState.DEAD
        self._ring.mark_failed(node_id)

        self._events.append({"type": "node_removed", "node": node_id})
        return True

    def _rebalance_after_add(self, new_node: str):
        """Assign new node to partitions that should include it."""
        for pid in range(self.config.num_partitions):
            partition_key = f"partition-{pid}"
            new_replicas = self._ring.get_preference_list(partition_key)
            if new_node in new_replicas and new_node not in self._partition_map.get(pid, []):
                # New node should be part of this partition
                current = self._partition_map.get(pid, [])
                if len(current) < self.config.replication_factor:
                    current.append(new_node)
                    self._partition_map[pid] = current
                    # Would need to add to ReplicaGroup -- simplified here
                    self._nodes[new_node].add_partition(pid)

    # --- Cluster info ---

    def get_partition_info(self) -> dict:
        """Get partition assignment info."""
        info = {}
        for pid, nodes in self._partition_map.items():
            group = self._replica_groups[pid]
            leader = group.get_leader()
            info[pid] = {
                "replicas": nodes,
                "leader": leader,
                "size": 0,
            }
            if leader:
                sm = group.get_state_machine(leader)
                if sm:
                    info[pid]["size"] = sm.partition.size()
        return info

    def get_node_info(self, node_id: str) -> Optional[dict]:
        node = self._nodes.get(node_id)
        if not node:
            return None
        return {
            "node_id": node.node_id,
            "address": node.address,
            "state": node.state.name,
            "partitions": dict(node.partitions),
            "hints_pending": len(node.hints),
        }

    def get_cluster_stats(self) -> dict:
        total_keys = 0
        for group in self._replica_groups.values():
            leader = group.get_leader()
            if leader:
                sm = group.get_state_machine(leader)
                if sm:
                    total_keys += sm.partition.size()
        return {
            "nodes": len(self._node_ids),
            "partitions": self.config.num_partitions,
            "replication_factor": self.config.replication_factor,
            "total_keys": total_keys,
            "events": len(self._events),
        }

    @property
    def nodes(self) -> dict[str, ClusterNode]:
        return dict(self._nodes)

    @property
    def partition_map(self) -> dict[int, list[str]]:
        return dict(self._partition_map)

    @property
    def replica_groups(self) -> dict[int, ReplicaGroup]:
        return dict(self._replica_groups)


# ============================================================
# KVClient: client-side routing and session tracking
# ============================================================

class KVClient:
    """Client for the distributed KV store."""

    def __init__(self, cluster: KVCluster, session_id: str = ""):
        self.cluster = cluster
        self.session_id = session_id or f"session-{id(self)}"
        self._request_counter = 0

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"{self.session_id}-{self._request_counter}"

    def put(self, key: str, value: Any,
            consistency: ConsistencyLevel = None) -> KVResponse:
        resp = self.cluster.put(key, value, consistency=consistency)
        resp.request_id = self._next_request_id()
        resp.session_id = self.session_id
        return resp

    def get(self, key: str,
            consistency: ConsistencyLevel = None) -> KVResponse:
        resp = self.cluster.get(key, consistency=consistency,
                                session_id=self.session_id)
        resp.request_id = self._next_request_id()
        return resp

    def delete(self, key: str,
               consistency: ConsistencyLevel = None) -> KVResponse:
        resp = self.cluster.delete(key, consistency=consistency)
        resp.request_id = self._next_request_id()
        resp.session_id = self.session_id
        return resp

    def cas(self, key: str, expected: Any, new_value: Any) -> KVResponse:
        resp = self.cluster.cas(key, expected, new_value)
        resp.request_id = self._next_request_id()
        return resp

    def scan(self, prefix: str = "", limit: int = 100) -> KVResponse:
        resp = self.cluster.scan(prefix=prefix, limit=limit)
        resp.request_id = self._next_request_id()
        return resp

    def batch(self, operations: list[dict]) -> KVResponse:
        resp = self.cluster.batch(operations)
        resp.request_id = self._next_request_id()
        return resp


# ============================================================
# KVHttpHandler: REST API adapter (composes with C016 pattern)
# ============================================================

class KVHttpHandler:
    """
    HTTP API handler for the distributed KV store.
    Designed to work with any HTTP server framework.

    Routes:
      GET    /kv/<key>           - Get value
      PUT    /kv/<key>           - Put value (body: {"value": ...})
      DELETE /kv/<key>           - Delete key
      POST   /kv/<key>/cas      - Compare-and-swap (body: {"expected": ..., "value": ...})
      GET    /kv?prefix=...     - Scan keys
      POST   /kv/_batch         - Batch operations
      GET    /cluster/info      - Cluster info
      GET    /cluster/partitions - Partition info
      GET    /cluster/node/<id> - Node info
    """

    def __init__(self, cluster: KVCluster):
        self.cluster = cluster
        self._routes = self._build_routes()

    def _build_routes(self) -> list:
        return [
            ("GET", "/kv/_scan", self._handle_scan),
            ("POST", "/kv/_batch", self._handle_batch),
            ("GET", "/kv/", self._handle_get),
            ("PUT", "/kv/", self._handle_put),
            ("DELETE", "/kv/", self._handle_delete),
            ("POST", "/kv/", self._handle_post),
            ("GET", "/cluster/info", self._handle_cluster_info),
            ("GET", "/cluster/partitions", self._handle_partition_info),
            ("GET", "/cluster/node/", self._handle_node_info_id),
        ]

    def handle_request(self, method: str, path: str, body: dict = None,
                       headers: dict = None) -> tuple:
        """Handle an HTTP request. Returns (status_code, response_dict)."""
        body = body or {}
        headers = headers or {}

        consistency = self._parse_consistency(headers.get("X-Consistency", ""))
        session_id = headers.get("X-Session-ID", "")

        # Route matching
        if method == "GET" and path == "/kv/_scan":
            return self._handle_scan(body, headers)
        elif method == "POST" and path == "/kv/_batch":
            return self._handle_batch(body, headers)
        elif path.startswith("/kv/"):
            key = path[4:]  # strip "/kv/"
            if key.endswith("/cas") and method == "POST":
                key = key[:-4]
                return self._handle_cas(key, body, consistency)
            elif method == "GET":
                return self._handle_get_key(key, consistency, session_id)
            elif method == "PUT":
                return self._handle_put_key(key, body, consistency)
            elif method == "DELETE":
                return self._handle_delete_key(key, consistency)
        elif path == "/cluster/info" and method == "GET":
            return self._handle_cluster_info(body, headers)
        elif path == "/cluster/partitions" and method == "GET":
            return self._handle_partition_info(body, headers)
        elif path.startswith("/cluster/node/") and method == "GET":
            node_id = path[len("/cluster/node/"):]
            return self._handle_node_info_id(node_id)

        return 404, {"error": "not_found"}

    def _handle_get_key(self, key: str, consistency: ConsistencyLevel,
                        session_id: str) -> tuple:
        resp = self.cluster.get(key, consistency=consistency,
                                session_id=session_id)
        if resp.success:
            return 200, resp.to_dict()
        return 500, resp.to_dict()

    def _handle_put_key(self, key: str, body: dict,
                        consistency: ConsistencyLevel) -> tuple:
        value = body.get("value")
        writer = body.get("writer", "")
        resp = self.cluster.put(key, value, consistency=consistency,
                                writer=writer)
        if resp.success:
            return 200, resp.to_dict()
        return 500, resp.to_dict()

    def _handle_delete_key(self, key: str,
                           consistency: ConsistencyLevel) -> tuple:
        resp = self.cluster.delete(key, consistency=consistency)
        if resp.success:
            return 200, resp.to_dict()
        return 500, resp.to_dict()

    def _handle_cas(self, key: str, body: dict,
                    consistency: ConsistencyLevel) -> tuple:
        expected = body.get("expected")
        value = body.get("value")
        writer = body.get("writer", "")
        resp = self.cluster.cas(key, expected, value, writer=writer)
        if resp.success:
            return 200, resp.to_dict()
        return 409, resp.to_dict()  # Conflict

    def _handle_scan(self, body: dict, headers: dict) -> tuple:
        prefix = body.get("prefix", "")
        limit = body.get("limit", 100)
        resp = self.cluster.scan(prefix=prefix, limit=limit)
        return 200, resp.to_dict()

    def _handle_batch(self, body: dict, headers: dict) -> tuple:
        operations = body.get("operations", [])
        resp = self.cluster.batch(operations)
        return 200, resp.to_dict()

    def _handle_cluster_info(self, body: dict, headers: dict) -> tuple:
        return 200, self.cluster.get_cluster_stats()

    def _handle_partition_info(self, body: dict, headers: dict) -> tuple:
        return 200, self.cluster.get_partition_info()

    def _handle_node_info_id(self, node_id: str) -> tuple:
        info = self.cluster.get_node_info(node_id)
        if info:
            return 200, info
        return 404, {"error": "node_not_found"}

    def _handle_get(self, body, headers):
        return 400, {"error": "key required"}

    def _handle_put(self, body, headers):
        return 400, {"error": "key required"}

    def _handle_delete(self, body, headers):
        return 400, {"error": "key required"}

    def _handle_post(self, body, headers):
        return 400, {"error": "key required"}

    def _parse_consistency(self, value: str) -> ConsistencyLevel:
        mapping = {
            "one": ConsistencyLevel.ONE,
            "quorum": ConsistencyLevel.QUORUM,
            "all": ConsistencyLevel.ALL,
            "strong": ConsistencyLevel.STRONG,
        }
        return mapping.get(value.lower(), self.cluster.config.default_consistency)


# ============================================================
# Convenience: create_cluster factory
# ============================================================

def create_cluster(n: int = 3, config: KVConfig = None) -> KVCluster:
    """Create an n-node cluster with default config."""
    node_ids = [f"node-{i}" for i in range(n)]
    return KVCluster(node_ids, config=config)
