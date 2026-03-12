"""
C226: Database Replication
Composes C223 (Connection Pool) + C201 (Raft Consensus)

Primary-replica replication system with:
- Write-Ahead Log (WAL) shipping for replication
- Synchronous and asynchronous replication modes
- Read replicas with configurable consistency
- Automatic failover via Raft consensus
- Replication lag monitoring
- Conflict detection for multi-primary
- Streaming replication with catch-up
"""

import sys, os, time, threading, hashlib, json, copy
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C223_connection_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C201_raft_consensus'))

from connection_pool import ConnectionPool, Connection, PoolConfig
from raft import RaftNode, RaftCluster, KeyValueStateMachine, LogEntry, Message


# --- Enums ---

class ReplicationMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    SEMI_SYNC = "semi_sync"


class NodeRole(Enum):
    PRIMARY = "primary"
    REPLICA = "replica"
    CANDIDATE = "candidate"


class ConsistencyLevel(Enum):
    EVENTUAL = "eventual"
    READ_YOUR_WRITES = "read_your_writes"
    STRONG = "strong"


class ReplicationState(Enum):
    STREAMING = "streaming"
    CATCHING_UP = "catching_up"
    STOPPED = "stopped"
    ERROR = "error"


# --- WAL Entry ---

@dataclass
class WALEntry:
    lsn: int  # Log Sequence Number
    timestamp: float
    operation: str  # INSERT, UPDATE, DELETE, DDL
    table: str
    data: dict
    checksum: str = ""
    term: int = 0

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self):
        content = f"{self.lsn}:{self.operation}:{self.table}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def verify(self):
        return self.checksum == self._compute_checksum()

    def to_dict(self):
        return {
            'lsn': self.lsn, 'timestamp': self.timestamp,
            'operation': self.operation, 'table': self.table,
            'data': self.data, 'checksum': self.checksum, 'term': self.term
        }

    @staticmethod
    def from_dict(d):
        return WALEntry(
            lsn=d['lsn'], timestamp=d['timestamp'], operation=d['operation'],
            table=d['table'], data=d['data'], checksum=d['checksum'], term=d.get('term', 0)
        )


# --- Replication Slot ---

@dataclass
class ReplicationSlot:
    """Tracks a replica's replication progress."""
    replica_id: str
    start_lsn: int = 0
    confirmed_lsn: int = 0
    sent_lsn: int = 0
    state: ReplicationState = ReplicationState.STOPPED
    lag_bytes: int = 0
    lag_entries: int = 0
    last_heartbeat: float = 0.0
    mode: ReplicationMode = ReplicationMode.ASYNC
    created_at: float = field(default_factory=time.time)


# --- WAL Shipper ---

class WALShipper:
    """Ships WAL entries from primary to replicas."""

    def __init__(self):
        self.wal = []
        self.next_lsn = 1
        self.lock = threading.Lock()
        self.slots = {}  # replica_id -> ReplicationSlot
        self.callbacks = defaultdict(list)  # event -> [callbacks]

    def append(self, operation, table, data, term=0):
        with self.lock:
            entry = WALEntry(
                lsn=self.next_lsn, timestamp=time.time(),
                operation=operation, table=table, data=data, term=term
            )
            self.wal.append(entry)
            self.next_lsn += 1
            self._fire('wal_append', entry)
            return entry

    def entries_from(self, start_lsn):
        with self.lock:
            return [e for e in self.wal if e.lsn >= start_lsn]

    def get_entry(self, lsn):
        with self.lock:
            for e in self.wal:
                if e.lsn == lsn:
                    return e
            return None

    def create_slot(self, replica_id, mode=ReplicationMode.ASYNC, start_lsn=None):
        slot = ReplicationSlot(
            replica_id=replica_id, start_lsn=start_lsn if start_lsn is not None else self.next_lsn,
            mode=mode, last_heartbeat=time.time()
        )
        self.slots[replica_id] = slot
        return slot

    def remove_slot(self, replica_id):
        return self.slots.pop(replica_id, None)

    def get_pending_entries(self, replica_id):
        slot = self.slots.get(replica_id)
        if not slot:
            return []
        return self.entries_from(slot.sent_lsn + 1) if slot.sent_lsn > 0 else self.entries_from(slot.start_lsn)

    def mark_sent(self, replica_id, lsn):
        slot = self.slots.get(replica_id)
        if slot:
            slot.sent_lsn = max(slot.sent_lsn, lsn)

    def confirm(self, replica_id, lsn):
        slot = self.slots.get(replica_id)
        if slot:
            slot.confirmed_lsn = max(slot.confirmed_lsn, lsn)
            slot.last_heartbeat = time.time()
            self._update_lag(slot)
            self._fire('confirm', replica_id, lsn)

    def _update_lag(self, slot):
        with self.lock:
            current_lsn = self.next_lsn - 1
            slot.lag_entries = max(0, current_lsn - slot.confirmed_lsn)
            slot.lag_bytes = slot.lag_entries * 100  # approximate

    def compact(self, up_to_lsn):
        """Remove WAL entries up to given LSN (already replicated everywhere)."""
        min_confirmed = min(
            (s.confirmed_lsn for s in self.slots.values()),
            default=up_to_lsn
        )
        safe_lsn = min(up_to_lsn, min_confirmed)
        with self.lock:
            self.wal = [e for e in self.wal if e.lsn > safe_lsn]
        return safe_lsn

    def on(self, event, callback):
        self.callbacks[event].append(callback)

    def _fire(self, event, *args):
        for cb in self.callbacks.get(event, []):
            cb(*args)

    @property
    def current_lsn(self):
        return self.next_lsn - 1

    @property
    def wal_size(self):
        return len(self.wal)


# --- Data Store ---

class DataStore:
    """Simple in-memory data store for replication targets."""

    def __init__(self):
        self.tables = defaultdict(dict)  # table -> {key: row}
        self.applied_lsn = 0
        self.lock = threading.Lock()

    def apply_entry(self, entry):
        with self.lock:
            if entry.lsn <= self.applied_lsn:
                return False  # already applied

            if not entry.verify():
                raise ValueError(f"Checksum mismatch at LSN {entry.lsn}")

            table = self.tables[entry.table]
            data = entry.data

            if entry.operation == 'INSERT':
                key = data.get('key') or data.get('id')
                table[key] = data
            elif entry.operation == 'UPDATE':
                key = data.get('key') or data.get('id')
                if key in table:
                    table[key].update(data)
                else:
                    table[key] = data
            elif entry.operation == 'DELETE':
                key = data.get('key') or data.get('id')
                table.pop(key, None)
            elif entry.operation == 'DDL':
                if data.get('action') == 'CREATE_TABLE':
                    if entry.table not in self.tables:
                        self.tables[entry.table] = {}
                elif data.get('action') == 'DROP_TABLE':
                    self.tables.pop(entry.table, None)

            self.applied_lsn = entry.lsn
            return True

    def query(self, table, key=None):
        with self.lock:
            if table not in self.tables:
                return None if key else {}
            if key is not None:
                return self.tables[table].get(key)
            return dict(self.tables[table])

    def snapshot(self):
        with self.lock:
            return {
                'tables': {t: dict(rows) for t, rows in self.tables.items()},
                'applied_lsn': self.applied_lsn
            }

    def restore(self, snapshot_data):
        with self.lock:
            self.tables = defaultdict(dict)
            for t, rows in snapshot_data['tables'].items():
                self.tables[t] = dict(rows)
            self.applied_lsn = snapshot_data['applied_lsn']


# --- Replication Node ---

class ReplicationNode:
    """A node in the replication cluster."""

    def __init__(self, node_id, mode=ReplicationMode.ASYNC, role=NodeRole.REPLICA):
        self.node_id = node_id
        self.role = role
        self.mode = mode
        self.store = DataStore()
        self.wal_shipper = WALShipper() if role == NodeRole.PRIMARY else None
        self.applied_lsn = 0
        self.write_buffer = []
        self.read_consistency = ConsistencyLevel.EVENTUAL
        self.last_write_lsn = 0  # for read-your-writes
        self.callbacks = defaultdict(list)
        self.is_running = True

    def write(self, operation, table, data):
        """Write an operation (primary only)."""
        if self.role != NodeRole.PRIMARY:
            raise RuntimeError(f"Node {self.node_id} is not primary, cannot write")

        entry = self.wal_shipper.append(operation, table, data)
        self.store.apply_entry(entry)
        self.applied_lsn = entry.lsn
        self.last_write_lsn = entry.lsn
        self._fire('write', entry)
        return entry

    def read(self, table, key=None, consistency=None):
        """Read from the store with configurable consistency."""
        level = consistency or self.read_consistency

        if level == ConsistencyLevel.STRONG and self.role == NodeRole.REPLICA:
            # Strong consistency on replica: check we're caught up
            if hasattr(self, '_primary_lsn') and self.applied_lsn < self._primary_lsn:
                raise RuntimeError("Replica not caught up for strong read")

        if level == ConsistencyLevel.READ_YOUR_WRITES:
            if self.applied_lsn < self.last_write_lsn:
                raise RuntimeError("Replica hasn't applied your latest write")

        return self.store.query(table, key)

    def apply_entries(self, entries):
        """Apply WAL entries from primary (replica only)."""
        applied = []
        for entry in entries:
            if self.store.apply_entry(entry):
                self.applied_lsn = entry.lsn
                applied.append(entry)
                self._fire('apply', entry)
        return applied

    def snapshot(self):
        return {
            'store': self.store.snapshot(),
            'node_id': self.node_id,
            'role': self.role.value,
            'applied_lsn': self.applied_lsn
        }

    def restore(self, snapshot_data):
        self.store.restore(snapshot_data['store'])
        self.applied_lsn = snapshot_data['applied_lsn']

    def promote(self):
        """Promote replica to primary."""
        if self.role == NodeRole.PRIMARY:
            return
        self.role = NodeRole.PRIMARY
        self.wal_shipper = WALShipper()
        # WAL starts from current LSN
        self.wal_shipper.next_lsn = self.applied_lsn + 1
        self._fire('promoted', self.node_id)

    def demote(self):
        """Demote primary to replica."""
        self.role = NodeRole.REPLICA
        self.wal_shipper = None
        self._fire('demoted', self.node_id)

    def on(self, event, callback):
        self.callbacks[event].append(callback)

    def _fire(self, event, *args):
        for cb in self.callbacks.get(event, []):
            cb(*args)

    def status(self):
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'mode': self.mode.value,
            'applied_lsn': self.applied_lsn,
            'wal_size': self.wal_shipper.wal_size if self.wal_shipper else 0,
            'is_running': self.is_running
        }


# --- Conflict Detector ---

class ConflictDetector:
    """Detects write-write conflicts in multi-primary setups."""

    def __init__(self):
        self.write_log = defaultdict(list)  # (table, key) -> [(node_id, lsn, timestamp)]
        self.conflicts = []

    def record_write(self, node_id, table, key, lsn, timestamp):
        self.write_log[(table, key)].append({
            'node_id': node_id, 'lsn': lsn, 'timestamp': timestamp
        })

    def detect(self, entry1, entry2):
        """Check if two entries conflict (same table+key, different nodes, overlapping time)."""
        if entry1.table != entry2.table:
            return None

        key1 = entry1.data.get('key') or entry1.data.get('id')
        key2 = entry2.data.get('key') or entry2.data.get('id')

        if key1 != key2:
            return None

        # Same key, both are writes (non-read operations)
        if entry1.operation in ('INSERT', 'UPDATE', 'DELETE') and \
           entry2.operation in ('INSERT', 'UPDATE', 'DELETE'):
            conflict = {
                'table': entry1.table, 'key': key1,
                'entry1': entry1.to_dict(), 'entry2': entry2.to_dict(),
                'type': self._classify_conflict(entry1, entry2)
            }
            self.conflicts.append(conflict)
            return conflict
        return None

    def _classify_conflict(self, e1, e2):
        if e1.operation == 'INSERT' and e2.operation == 'INSERT':
            return 'INSERT_INSERT'
        if e1.operation == 'DELETE' or e2.operation == 'DELETE':
            return 'DELETE_CONFLICT'
        return 'UPDATE_UPDATE'

    def resolve_last_writer_wins(self, entry1, entry2):
        """Resolve conflict by timestamp (last writer wins)."""
        return entry1 if entry1.timestamp >= entry2.timestamp else entry2

    def resolve_by_node_priority(self, entry1, entry2, priority):
        """Resolve by node priority map."""
        p1 = priority.get(entry1.data.get('_source_node', ''), 0)
        p2 = priority.get(entry2.data.get('_source_node', ''), 0)
        return entry1 if p1 >= p2 else entry2


# --- Replication Lag Monitor ---

class LagMonitor:
    """Monitors replication lag across replicas."""

    def __init__(self, warn_threshold=10, critical_threshold=100):
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.history = defaultdict(list)  # replica_id -> [(timestamp, lag)]
        self.alerts = []

    def record(self, replica_id, lag_entries, primary_lsn):
        now = time.time()
        self.history[replica_id].append((now, lag_entries))
        # Keep last 1000 entries
        if len(self.history[replica_id]) > 1000:
            self.history[replica_id] = self.history[replica_id][-500:]

        if lag_entries >= self.critical_threshold:
            self.alerts.append({
                'level': 'CRITICAL', 'replica_id': replica_id,
                'lag': lag_entries, 'primary_lsn': primary_lsn,
                'timestamp': now
            })
        elif lag_entries >= self.warn_threshold:
            self.alerts.append({
                'level': 'WARN', 'replica_id': replica_id,
                'lag': lag_entries, 'primary_lsn': primary_lsn,
                'timestamp': now
            })

    def get_lag(self, replica_id):
        history = self.history.get(replica_id, [])
        if not history:
            return 0
        return history[-1][1]

    def get_avg_lag(self, replica_id, window=10):
        history = self.history.get(replica_id, [])
        if not history:
            return 0.0
        recent = history[-window:]
        return sum(lag for _, lag in recent) / len(recent)

    def get_max_lag(self):
        max_lag = 0
        max_replica = None
        for rid, history in self.history.items():
            if history and history[-1][1] > max_lag:
                max_lag = history[-1][1]
                max_replica = rid
        return max_replica, max_lag

    def get_alerts(self, level=None):
        if level:
            return [a for a in self.alerts if a['level'] == level]
        return list(self.alerts)

    def clear_alerts(self):
        self.alerts.clear()


# --- Replication Cluster ---

class ReplicationCluster:
    """
    Manages a cluster of replication nodes with automatic failover.
    Composes:
    - WAL shipping for data replication
    - Raft consensus for leader election and failover
    - Connection pooling for client connections
    """

    def __init__(self, mode=ReplicationMode.ASYNC, sync_replicas=1):
        self.nodes = {}  # node_id -> ReplicationNode
        self.primary_id = None
        self.mode = mode
        self.sync_replicas = sync_replicas  # for semi-sync
        self.conflict_detector = ConflictDetector()
        self.lag_monitor = LagMonitor()
        self.raft_cluster = None
        self.callbacks = defaultdict(list)
        self.failover_history = []
        self.read_preference = 'primary'  # primary, replica, nearest

    def add_node(self, node_id, role=NodeRole.REPLICA, mode=None):
        """Add a node to the cluster."""
        node_mode = mode or self.mode
        node = ReplicationNode(node_id, mode=node_mode, role=role)

        if role == NodeRole.PRIMARY:
            if self.primary_id is not None:
                raise RuntimeError(f"Primary already exists: {self.primary_id}")
            self.primary_id = node_id

        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id):
        """Remove a node from the cluster."""
        node = self.nodes.pop(node_id, None)
        if node and node_id == self.primary_id:
            self.primary_id = None
        if node and node.wal_shipper:
            node.wal_shipper.remove_slot(node_id)
        return node

    def get_primary(self):
        if self.primary_id:
            return self.nodes.get(self.primary_id)
        return None

    def get_replicas(self):
        return [n for nid, n in self.nodes.items() if n.role == NodeRole.REPLICA]

    def write(self, operation, table, data):
        """Write through the primary."""
        primary = self.get_primary()
        if not primary:
            raise RuntimeError("No primary node available")

        entry = primary.write(operation, table, data)

        # Create replication slots for new replicas
        for nid, node in self.nodes.items():
            if node.role == NodeRole.REPLICA and nid not in primary.wal_shipper.slots:
                primary.wal_shipper.create_slot(nid, node.mode)

        # Replicate based on mode
        if self.mode == ReplicationMode.SYNC:
            self._replicate_sync(entry)
        elif self.mode == ReplicationMode.SEMI_SYNC:
            self._replicate_semi_sync(entry)
        else:
            self._replicate_async(entry)

        return entry

    def _replicate_sync(self, entry):
        """Synchronous: wait for ALL replicas to confirm."""
        primary = self.get_primary()
        for nid, node in self.nodes.items():
            if node.role == NodeRole.REPLICA:
                node.apply_entries([entry])
                primary.wal_shipper.mark_sent(nid, entry.lsn)
                primary.wal_shipper.confirm(nid, entry.lsn)
                self.lag_monitor.record(nid, 0, entry.lsn)

    def _replicate_semi_sync(self, entry):
        """Semi-sync: wait for N replicas to confirm."""
        primary = self.get_primary()
        replicas = self.get_replicas()
        confirmed = 0

        for node in replicas:
            node.apply_entries([entry])
            primary.wal_shipper.mark_sent(node.node_id, entry.lsn)
            primary.wal_shipper.confirm(node.node_id, entry.lsn)
            confirmed += 1
            self.lag_monitor.record(node.node_id, 0, entry.lsn)
            if confirmed >= self.sync_replicas:
                break

        # Remaining replicas get async
        for node in replicas[confirmed:]:
            node.apply_entries([entry])
            primary.wal_shipper.mark_sent(node.node_id, entry.lsn)
            primary.wal_shipper.confirm(node.node_id, entry.lsn)

    def _replicate_async(self, entry):
        """Async: ship entries without waiting for confirmation."""
        primary = self.get_primary()
        for nid, node in self.nodes.items():
            if node.role == NodeRole.REPLICA:
                # In true async, entries would be shipped in background
                # Here we simulate immediate shipping
                node.apply_entries([entry])
                primary.wal_shipper.mark_sent(nid, entry.lsn)
                primary.wal_shipper.confirm(nid, entry.lsn)
                lag = primary.wal_shipper.current_lsn - node.applied_lsn
                self.lag_monitor.record(nid, lag, primary.wal_shipper.current_lsn)

    def read(self, table, key=None, consistency=None, preferred_node=None):
        """Read from the cluster with configurable routing."""
        if preferred_node and preferred_node in self.nodes:
            return self.nodes[preferred_node].read(table, key, consistency)

        if self.read_preference == 'primary' or consistency == ConsistencyLevel.STRONG:
            primary = self.get_primary()
            if primary:
                return primary.read(table, key, consistency)

        if self.read_preference == 'replica':
            replicas = self.get_replicas()
            if replicas:
                # Pick replica with lowest lag
                best = min(replicas, key=lambda r: self.lag_monitor.get_lag(r.node_id))
                return best.read(table, key, consistency)

        # Fallback to primary
        primary = self.get_primary()
        if primary:
            return primary.read(table, key, consistency)
        raise RuntimeError("No nodes available for read")

    def replicate_pending(self):
        """Ship all pending WAL entries to replicas."""
        primary = self.get_primary()
        if not primary or not primary.wal_shipper:
            return {}

        results = {}
        for nid, node in self.nodes.items():
            if node.role == NodeRole.REPLICA:
                if nid not in primary.wal_shipper.slots:
                    primary.wal_shipper.create_slot(nid, node.mode,
                                                     start_lsn=node.applied_lsn + 1)

                pending = primary.wal_shipper.get_pending_entries(nid)
                if pending:
                    applied = node.apply_entries(pending)
                    for e in applied:
                        primary.wal_shipper.mark_sent(nid, e.lsn)
                        primary.wal_shipper.confirm(nid, e.lsn)

                    lag = primary.wal_shipper.current_lsn - node.applied_lsn
                    self.lag_monitor.record(nid, lag, primary.wal_shipper.current_lsn)
                    results[nid] = len(applied)
                else:
                    results[nid] = 0

        return results

    def failover(self, new_primary_id=None):
        """Perform failover: promote a replica to primary."""
        old_primary_id = self.primary_id

        if new_primary_id is None:
            # Choose replica with highest applied_lsn
            replicas = self.get_replicas()
            if not replicas:
                raise RuntimeError("No replicas available for failover")
            new_primary_id = max(replicas, key=lambda r: r.applied_lsn).node_id

        new_primary = self.nodes.get(new_primary_id)
        if not new_primary:
            raise RuntimeError(f"Node {new_primary_id} not found")

        # Demote old primary if still in cluster
        if old_primary_id and old_primary_id in self.nodes:
            self.nodes[old_primary_id].demote()

        # Promote new primary
        new_primary.promote()
        self.primary_id = new_primary_id

        # Re-register replicas with new primary
        for nid, node in self.nodes.items():
            if node.role == NodeRole.REPLICA:
                new_primary.wal_shipper.create_slot(nid, node.mode)

        self.failover_history.append({
            'timestamp': time.time(),
            'old_primary': old_primary_id,
            'new_primary': new_primary_id,
            'lsn_at_failover': new_primary.applied_lsn
        })

        self._fire('failover', old_primary_id, new_primary_id)
        return new_primary

    def catch_up_replica(self, replica_id, snapshot=None):
        """Catch up a lagging replica using snapshot + WAL replay."""
        primary = self.get_primary()
        replica = self.nodes.get(replica_id)
        if not primary or not replica:
            return False

        if snapshot:
            replica.restore(snapshot)

        # Replay WAL entries from replica's position
        entries = primary.wal_shipper.entries_from(replica.applied_lsn + 1)
        replica.apply_entries(entries)
        return True

    def add_replica_from_snapshot(self, node_id, mode=None):
        """Add a new replica initialized from primary's current state."""
        primary = self.get_primary()
        if not primary:
            raise RuntimeError("No primary to snapshot from")

        node = self.add_node(node_id, NodeRole.REPLICA, mode)
        snap = primary.snapshot()
        node.restore(snap)
        primary.wal_shipper.create_slot(node_id, node.mode)
        return node

    def cluster_status(self):
        """Get full cluster status."""
        statuses = {}
        for nid, node in self.nodes.items():
            s = node.status()
            s['lag'] = self.lag_monitor.get_lag(nid)
            s['avg_lag'] = self.lag_monitor.get_avg_lag(nid)
            statuses[nid] = s

        primary = self.get_primary()
        return {
            'primary': self.primary_id,
            'mode': self.mode.value,
            'node_count': len(self.nodes),
            'replica_count': len(self.get_replicas()),
            'primary_lsn': primary.applied_lsn if primary else 0,
            'nodes': statuses,
            'failover_count': len(self.failover_history),
            'alerts': len(self.lag_monitor.alerts)
        }

    def compact_wal(self, up_to_lsn=None):
        """Compact WAL on primary up to safely replicated LSN."""
        primary = self.get_primary()
        if not primary or not primary.wal_shipper:
            return 0

        if up_to_lsn is None:
            # Find minimum confirmed LSN across all replicas
            slots = primary.wal_shipper.slots
            if not slots:
                up_to_lsn = primary.wal_shipper.current_lsn
            else:
                up_to_lsn = min(s.confirmed_lsn for s in slots.values())

        return primary.wal_shipper.compact(up_to_lsn)

    def on(self, event, callback):
        self.callbacks[event].append(callback)

    def _fire(self, event, *args):
        for cb in self.callbacks.get(event, []):
            cb(*args)


# --- Raft-Based Replication Cluster ---

class RaftReplicationCluster:
    """
    Replication cluster using Raft consensus for automatic leader election.
    Combines WAL shipping with Raft for strong consistency guarantees.
    """

    def __init__(self, node_ids, mode=ReplicationMode.SYNC):
        self.mode = mode
        self.stores = {}  # node_id -> DataStore
        self.raft_cluster = RaftCluster(node_ids)
        self.lag_monitor = LagMonitor()
        self.conflict_detector = ConflictDetector()

        for nid in node_ids:
            self.stores[nid] = DataStore()

        self.raft_nodes = {nid: self.raft_cluster.nodes[nid] for nid in node_ids}
        self.applied_index = defaultdict(int)  # node_id -> last applied index

    def elect_leader(self, max_ticks=500):
        """Tick until a leader is elected."""
        for _ in range(max_ticks):
            self.raft_cluster.tick()
            self.raft_cluster.deliver_all()
            leader = self._get_leader()
            if leader:
                return leader
        return None

    def _get_leader(self):
        for nid, rnode in self.raft_nodes.items():
            role_val = rnode.role.value if hasattr(rnode.role, 'value') else str(rnode.role)
            if role_val == 'leader':
                return nid
        return None

    def write(self, operation, table, data):
        """Submit write through Raft consensus."""
        leader_id = self._get_leader()
        if not leader_id:
            raise RuntimeError("No leader elected")

        entry_data = {
            'operation': operation, 'table': table, 'data': data
        }

        # Submit through Raft consensus and wait for commit
        self.raft_cluster.submit_and_commit(entry_data)

        # Apply committed entries to data stores
        self._apply_committed()

        return entry_data

    def _apply_committed(self):
        """Apply committed Raft log entries to data stores."""
        for nid, rnode in self.raft_nodes.items():
            store = self.stores[nid]
            for i in range(self.applied_index[nid] + 1, rnode.commit_index + 1):
                entry = rnode.log.get(i)
                if entry and entry.command:
                    cmd = entry.command
                    wal_entry = WALEntry(
                        lsn=i, timestamp=time.time(),
                        operation=cmd['operation'], table=cmd['table'],
                        data=cmd['data'], term=entry.term
                    )
                    store.apply_entry(wal_entry)
            self.applied_index[nid] = max(self.applied_index[nid], rnode.commit_index)

    def read(self, table, key=None, node_id=None):
        """Read from a specific node or the leader."""
        target = node_id or self._get_leader()
        if not target or target not in self.stores:
            raise RuntimeError("No node available for read")
        return self.stores[target].query(table, key)

    def get_leader(self):
        return self._get_leader()

    def status(self):
        leader = self._get_leader()
        return {
            'leader': leader,
            'mode': self.mode.value,
            'nodes': {
                nid: {
                    'role': rnode.role.value if hasattr(rnode.role, 'value') else str(rnode.role),
                    'term': rnode.current_term,
                    'commit_index': rnode.commit_index,
                    'applied_index': self.applied_index[nid],
                    'log_length': len(rnode.log.entries)
                }
                for nid, rnode in self.raft_nodes.items()
            }
        }


# --- Streaming Replication Manager ---

class StreamingReplicationManager:
    """
    Manages continuous streaming replication between primary and replicas.
    Handles catch-up, steady-state streaming, and error recovery.
    """

    def __init__(self, cluster):
        self.cluster = cluster
        self.streams = {}  # replica_id -> stream state
        self.batch_size = 100
        self.max_retries = 3

    def start_stream(self, replica_id):
        """Start streaming replication to a replica."""
        primary = self.cluster.get_primary()
        replica = self.cluster.nodes.get(replica_id)
        if not primary or not replica:
            return False

        # Determine starting point
        start_lsn = replica.applied_lsn + 1

        self.streams[replica_id] = {
            'state': ReplicationState.CATCHING_UP,
            'start_lsn': start_lsn,
            'retries': 0,
            'bytes_sent': 0,
            'entries_sent': 0
        }

        # Check if catch-up needed
        primary_lsn = primary.wal_shipper.current_lsn
        if primary_lsn - replica.applied_lsn > self.batch_size:
            self.streams[replica_id]['state'] = ReplicationState.CATCHING_UP
        else:
            self.streams[replica_id]['state'] = ReplicationState.STREAMING

        if replica_id not in primary.wal_shipper.slots:
            primary.wal_shipper.create_slot(replica_id, replica.mode,
                                             start_lsn=start_lsn)

        return True

    def stop_stream(self, replica_id):
        """Stop streaming to a replica."""
        stream = self.streams.pop(replica_id, None)
        if stream:
            stream['state'] = ReplicationState.STOPPED
        return stream

    def tick(self):
        """Process one round of streaming for all active streams."""
        primary = self.cluster.get_primary()
        if not primary:
            return {}

        results = {}
        for replica_id, stream in list(self.streams.items()):
            if stream['state'] in (ReplicationState.STOPPED, ReplicationState.ERROR):
                continue

            replica = self.cluster.nodes.get(replica_id)
            if not replica:
                stream['state'] = ReplicationState.ERROR
                continue

            # Get pending entries
            pending = primary.wal_shipper.get_pending_entries(replica_id)
            if not pending:
                if stream['state'] == ReplicationState.CATCHING_UP:
                    stream['state'] = ReplicationState.STREAMING
                results[replica_id] = 0
                continue

            # Apply batch
            batch = pending[:self.batch_size]
            try:
                applied = replica.apply_entries(batch)
                for e in applied:
                    primary.wal_shipper.mark_sent(replica_id, e.lsn)
                    primary.wal_shipper.confirm(replica_id, e.lsn)

                stream['entries_sent'] += len(applied)
                stream['retries'] = 0

                # Check if caught up
                remaining = len(pending) - len(batch)
                if remaining == 0:
                    stream['state'] = ReplicationState.STREAMING
                else:
                    stream['state'] = ReplicationState.CATCHING_UP

                lag = primary.wal_shipper.current_lsn - replica.applied_lsn
                self.cluster.lag_monitor.record(replica_id, lag, primary.wal_shipper.current_lsn)
                results[replica_id] = len(applied)

            except Exception as e:
                stream['retries'] += 1
                if stream['retries'] >= self.max_retries:
                    stream['state'] = ReplicationState.ERROR
                results[replica_id] = 0

        return results

    def stream_status(self):
        return {
            rid: {
                'state': s['state'].value,
                'entries_sent': s['entries_sent'],
                'retries': s['retries']
            }
            for rid, s in self.streams.items()
        }


# --- Read/Write Splitter ---

class ReadWriteSplitter:
    """Routes reads to replicas and writes to primary."""

    def __init__(self, cluster):
        self.cluster = cluster
        self.write_count = 0
        self.read_count = 0
        self.read_from_primary = 0
        self.read_from_replica = 0
        self.round_robin_index = 0

    def execute(self, query_type, table, key=None, data=None,
                consistency=None, operation='SELECT'):
        """Route query based on type."""
        if query_type == 'write':
            self.write_count += 1
            return self.cluster.write(operation or 'INSERT', table, data or {})
        else:
            self.read_count += 1
            return self._route_read(table, key, consistency)

    def _route_read(self, table, key, consistency):
        if consistency == ConsistencyLevel.STRONG:
            self.read_from_primary += 1
            primary = self.cluster.get_primary()
            if primary:
                return primary.read(table, key, consistency)
            raise RuntimeError("No primary for strong read")

        replicas = self.cluster.get_replicas()
        if replicas:
            # Round-robin across replicas
            idx = self.round_robin_index % len(replicas)
            self.round_robin_index += 1
            self.read_from_replica += 1
            return replicas[idx].read(table, key, consistency)

        # Fallback to primary
        self.read_from_primary += 1
        primary = self.cluster.get_primary()
        if primary:
            return primary.read(table, key)
        raise RuntimeError("No nodes available")

    def stats(self):
        return {
            'writes': self.write_count,
            'reads': self.read_count,
            'reads_from_primary': self.read_from_primary,
            'reads_from_replica': self.read_from_replica,
            'split_ratio': self.read_from_replica / max(1, self.read_count)
        }
