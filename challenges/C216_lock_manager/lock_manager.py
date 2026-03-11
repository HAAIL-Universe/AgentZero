"""
C216: Lock Manager

Comprehensive hierarchical lock manager for database systems.

Lock modes:
- S (Shared): Multiple readers
- X (Exclusive): Single writer
- IS (Intent Shared): Intent to acquire S on child
- IX (Intent Exclusive): Intent to acquire X on child
- SIX (Shared + Intent Exclusive): Read all, write some children

Features:
- Hierarchical locking (database -> table -> page -> row)
- Lock compatibility matrix
- Deadlock detection via wait-for graph (cycle detection)
- Lock escalation (row locks -> table lock when threshold exceeded)
- Lock timeouts
- Lock upgrade (S -> X, IS -> IX, IS -> S, etc.)
- Transaction lock tracking and bulk release
- Wait queue with FIFO fairness

Composes with C212 Transaction Manager conceptually.
No external dependencies. Pure Python.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto
from collections import defaultdict, deque
import time
import threading


# ============================================================
# Lock Modes
# ============================================================

class LockMode(Enum):
    IS = 0   # Intent Shared
    IX = 1   # Intent Exclusive
    S = 2    # Shared
    SIX = 3  # Shared + Intent Exclusive
    X = 4    # Exclusive

    def __lt__(self, other):
        if not isinstance(other, LockMode):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, LockMode):
            return NotImplemented
        return self.value <= other.value


# Compatibility matrix: COMPATIBILITY[held][requested] = True if compatible
# From standard database textbook (Gray & Reuter)
#          IS    IX    S    SIX    X
# IS       Y     Y     Y    Y     N
# IX       Y     Y     N    N     N
# S        Y     N     Y    N     N
# SIX      Y     N     N    N     N
# X        N     N     N    N     N

COMPATIBILITY = {
    LockMode.IS:  {LockMode.IS: True,  LockMode.IX: True,  LockMode.S: True,  LockMode.SIX: True,  LockMode.X: False},
    LockMode.IX:  {LockMode.IS: True,  LockMode.IX: True,  LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
    LockMode.S:   {LockMode.IS: True,  LockMode.IX: False, LockMode.S: True,  LockMode.SIX: False, LockMode.X: False},
    LockMode.SIX: {LockMode.IS: True,  LockMode.IX: False, LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
    LockMode.X:   {LockMode.IS: False, LockMode.IX: False, LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
}

# Lock upgrade: what mode results from combining two modes held by same txn
# upgrade_mode(current, requested) -> resulting mode
UPGRADE = {
    (LockMode.IS, LockMode.IS): LockMode.IS,
    (LockMode.IS, LockMode.IX): LockMode.IX,
    (LockMode.IS, LockMode.S): LockMode.S,
    (LockMode.IS, LockMode.SIX): LockMode.SIX,
    (LockMode.IS, LockMode.X): LockMode.X,
    (LockMode.IX, LockMode.IS): LockMode.IX,
    (LockMode.IX, LockMode.IX): LockMode.IX,
    (LockMode.IX, LockMode.S): LockMode.SIX,
    (LockMode.IX, LockMode.SIX): LockMode.SIX,
    (LockMode.IX, LockMode.X): LockMode.X,
    (LockMode.S, LockMode.IS): LockMode.S,
    (LockMode.S, LockMode.IX): LockMode.SIX,
    (LockMode.S, LockMode.S): LockMode.S,
    (LockMode.S, LockMode.SIX): LockMode.SIX,
    (LockMode.S, LockMode.X): LockMode.X,
    (LockMode.SIX, LockMode.IS): LockMode.SIX,
    (LockMode.SIX, LockMode.IX): LockMode.SIX,
    (LockMode.SIX, LockMode.S): LockMode.SIX,
    (LockMode.SIX, LockMode.SIX): LockMode.SIX,
    (LockMode.SIX, LockMode.X): LockMode.X,
    (LockMode.X, LockMode.IS): LockMode.X,
    (LockMode.X, LockMode.IX): LockMode.X,
    (LockMode.X, LockMode.S): LockMode.X,
    (LockMode.X, LockMode.SIX): LockMode.X,
    (LockMode.X, LockMode.X): LockMode.X,
}


# ============================================================
# Resource hierarchy
# ============================================================

class ResourceLevel(Enum):
    DATABASE = 0
    TABLE = 1
    PAGE = 2
    ROW = 3


@dataclass
class ResourceId:
    """Identifies a lockable resource in the hierarchy."""
    level: ResourceLevel
    database: str = ""
    table: str = ""
    page: int = -1
    row: int = -1

    @property
    def key(self) -> str:
        if self.level == ResourceLevel.DATABASE:
            return f"db:{self.database}"
        elif self.level == ResourceLevel.TABLE:
            return f"tbl:{self.database}.{self.table}"
        elif self.level == ResourceLevel.PAGE:
            return f"pg:{self.database}.{self.table}.{self.page}"
        else:
            return f"row:{self.database}.{self.table}.{self.page}.{self.row}"

    @property
    def parent(self) -> Optional[ResourceId]:
        """Get parent resource in hierarchy."""
        if self.level == ResourceLevel.DATABASE:
            return None
        elif self.level == ResourceLevel.TABLE:
            return ResourceId(ResourceLevel.DATABASE, self.database)
        elif self.level == ResourceLevel.PAGE:
            return ResourceId(ResourceLevel.TABLE, self.database, self.table)
        else:
            return ResourceId(ResourceLevel.PAGE, self.database, self.table, self.page)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, ResourceId):
            return NotImplemented
        return self.key == other.key

    def __repr__(self):
        return f"ResourceId({self.key})"


def make_db(name: str) -> ResourceId:
    return ResourceId(ResourceLevel.DATABASE, database=name)

def make_table(db: str, table: str) -> ResourceId:
    return ResourceId(ResourceLevel.TABLE, database=db, table=table)

def make_page(db: str, table: str, page: int) -> ResourceId:
    return ResourceId(ResourceLevel.PAGE, database=db, table=table, page=page)

def make_row(db: str, table: str, page: int, row: int) -> ResourceId:
    return ResourceId(ResourceLevel.ROW, database=db, table=table, page=page, row=row)


# ============================================================
# Lock request and lock entry
# ============================================================

class LockResult(Enum):
    GRANTED = auto()
    WAITING = auto()
    DENIED = auto()       # Timeout
    DEADLOCK = auto()     # Would cause deadlock
    UPGRADED = auto()     # Lock mode upgraded


@dataclass
class LockRequest:
    """A pending or granted lock request."""
    tx_id: int
    mode: LockMode
    granted: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class LockEntry:
    """Lock state for a single resource."""
    resource: ResourceId
    granted: dict[int, LockMode] = field(default_factory=dict)  # tx_id -> mode
    wait_queue: deque[LockRequest] = field(default_factory=deque)

    @property
    def is_free(self) -> bool:
        return len(self.granted) == 0 and len(self.wait_queue) == 0

    def group_mode(self) -> Optional[LockMode]:
        """Most restrictive mode currently granted (max by enum value)."""
        if not self.granted:
            return None
        return max(self.granted.values())

    def is_compatible(self, mode: LockMode, exclude_tx: int = -1) -> bool:
        """Check if mode is compatible with all currently granted locks (excluding a txn)."""
        for tx_id, held_mode in self.granted.items():
            if tx_id == exclude_tx:
                continue
            if not COMPATIBILITY[held_mode][mode]:
                return False
        return True


# ============================================================
# Exceptions
# ============================================================

class LockError(Exception):
    pass

class DeadlockError(LockError):
    pass

class LockTimeoutError(LockError):
    pass

class LockEscalationError(LockError):
    pass


# ============================================================
# Wait-For Graph (Deadlock Detection)
# ============================================================

class WaitForGraph:
    """Directed graph: tx_id -> set of tx_ids it waits for."""

    def __init__(self):
        self._edges: dict[int, set[int]] = defaultdict(set)

    def add_edge(self, waiter: int, holder: int):
        """waiter is waiting for holder."""
        if waiter != holder:
            self._edges[waiter].add(holder)

    def remove_node(self, tx_id: int):
        """Remove all edges involving tx_id."""
        self._edges.pop(tx_id, None)
        for s in self._edges.values():
            s.discard(tx_id)

    def remove_edges_for(self, waiter: int):
        """Remove all wait-for edges from waiter."""
        self._edges.pop(waiter, None)

    def has_cycle(self, start: int) -> list[int]:
        """Check if adding edges for start creates a cycle. Returns cycle path or []."""
        if start not in self._edges:
            return []

        visited = set()
        path = []

        def dfs(node: int) -> list[int]:
            if node in visited:
                # Found cycle - extract it
                if node in path:
                    idx = path.index(node)
                    return path[idx:] + [node]
                return []
            visited.add(node)
            path.append(node)
            for neighbor in self._edges.get(node, set()):
                result = dfs(neighbor)
                if result:
                    return result
            path.pop()
            return []

        return dfs(start)

    def detect_cycle(self) -> list[int]:
        """Detect any cycle in the graph. Returns cycle path or []."""
        visited = set()
        rec_stack = set()

        def dfs(node: int, path: list[int]) -> list[int]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in self._edges.get(node, set()):
                if neighbor not in visited:
                    result = dfs(neighbor, path)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    idx = path.index(neighbor)
                    return path[idx:] + [neighbor]
            path.pop()
            rec_stack.discard(node)
            return []

        for node in list(self._edges.keys()):
            if node not in visited:
                result = dfs(node, [])
                if result:
                    return result
        return []

    @property
    def edges(self) -> dict[int, set[int]]:
        return dict(self._edges)

    def clear(self):
        self._edges.clear()


# ============================================================
# Lock Manager
# ============================================================

class LockManager:
    """
    Hierarchical lock manager with deadlock detection, escalation, and timeouts.

    Thread-safe. Uses a single mutex for simplicity (fine for moderate concurrency).
    """

    def __init__(self, escalation_threshold: int = 100,
                 default_timeout: float = 5.0,
                 deadlock_detection: bool = True):
        self._lock = threading.Lock()
        self._entries: dict[str, LockEntry] = {}  # resource key -> LockEntry
        self._tx_locks: dict[int, dict[str, LockMode]] = defaultdict(dict)  # tx -> {resource_key: mode}
        self._wait_graph = WaitForGraph()
        self._escalation_threshold = escalation_threshold
        self._default_timeout = default_timeout
        self._deadlock_detection = deadlock_detection
        # Condition for waking waiters
        self._condition = threading.Condition(self._lock)
        # Stats
        self._stats = LockStats()

    def _get_entry(self, resource: ResourceId) -> LockEntry:
        key = resource.key
        if key not in self._entries:
            self._entries[key] = LockEntry(resource=resource)
        return self._entries[key]

    def acquire(self, tx_id: int, resource: ResourceId, mode: LockMode,
                timeout: Optional[float] = None, auto_intention: bool = True) -> LockResult:
        """
        Acquire a lock on the given resource.

        If auto_intention is True, automatically acquires intention locks on parent resources.
        Returns LockResult indicating outcome.
        Raises DeadlockError if deadlock detected.
        Raises LockTimeoutError if timeout exceeded.
        """
        if timeout is None:
            timeout = self._default_timeout

        # Acquire intention locks on ancestors first
        if auto_intention:
            intention_mode = LockMode.IS if mode in (LockMode.S, LockMode.IS) else LockMode.IX
            parent = resource.parent
            while parent is not None:
                result = self._acquire_single(tx_id, parent, intention_mode, timeout)
                if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                    return result
                parent = parent.parent

        return self._acquire_single(tx_id, resource, mode, timeout)

    def _acquire_single(self, tx_id: int, resource: ResourceId,
                        mode: LockMode, timeout: float) -> LockResult:
        """Acquire a single lock (no intention lock propagation)."""
        deadline = time.time() + timeout

        with self._condition:
            entry = self._get_entry(resource)
            key = resource.key

            # Already hold this lock?
            if tx_id in entry.granted:
                current = entry.granted[tx_id]
                if current == mode or UPGRADE.get((current, mode), mode) == current:
                    # Already sufficient
                    return LockResult.GRANTED
                # Need upgrade
                upgraded = UPGRADE.get((current, mode), mode)
                if entry.is_compatible(upgraded, exclude_tx=tx_id):
                    entry.granted[tx_id] = upgraded
                    self._tx_locks[tx_id][key] = upgraded
                    self._stats.upgrades += 1
                    return LockResult.UPGRADED

                # Must wait for upgrade - check deadlock first
                if self._deadlock_detection:
                    for holder_tx in entry.granted:
                        if holder_tx != tx_id:
                            self._wait_graph.add_edge(tx_id, holder_tx)
                    cycle = self._wait_graph.has_cycle(tx_id)
                    if cycle:
                        self._wait_graph.remove_edges_for(tx_id)
                        self._stats.deadlocks += 1
                        raise DeadlockError(f"Deadlock detected: {cycle}")

                # Wait for upgrade
                req = LockRequest(tx_id=tx_id, mode=upgraded, granted=False)
                entry.wait_queue.append(req)

                while not req.granted:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        entry.wait_queue.remove(req)
                        self._wait_graph.remove_edges_for(tx_id)
                        self._stats.timeouts += 1
                        raise LockTimeoutError(f"Timeout waiting for lock on {resource}")
                    self._condition.wait(timeout=min(remaining, 0.1))
                    # Re-check
                    if req.granted:
                        break
                    if entry.is_compatible(upgraded, exclude_tx=tx_id):
                        entry.granted[tx_id] = upgraded
                        self._tx_locks[tx_id][key] = upgraded
                        req.granted = True
                        entry.wait_queue.remove(req)
                        self._wait_graph.remove_edges_for(tx_id)
                        self._stats.upgrades += 1
                        return LockResult.UPGRADED

                self._wait_graph.remove_edges_for(tx_id)
                self._stats.upgrades += 1
                return LockResult.UPGRADED

            # New lock request - check compatibility
            if entry.is_compatible(mode):
                # No waiters ahead or compatible
                if len(entry.wait_queue) == 0 or self._can_grant_ahead_of_waiters(entry, mode):
                    entry.granted[tx_id] = mode
                    self._tx_locks[tx_id][key] = mode
                    self._stats.grants += 1
                    return LockResult.GRANTED

            # Must wait - deadlock check
            if self._deadlock_detection:
                for holder_tx in entry.granted:
                    if holder_tx != tx_id:
                        self._wait_graph.add_edge(tx_id, holder_tx)
                # Also wait for txns ahead in queue whose locks conflict
                for req_ahead in entry.wait_queue:
                    if req_ahead.tx_id != tx_id and not COMPATIBILITY.get(mode, {}).get(req_ahead.mode, False):
                        self._wait_graph.add_edge(tx_id, req_ahead.tx_id)
                cycle = self._wait_graph.has_cycle(tx_id)
                if cycle:
                    self._wait_graph.remove_edges_for(tx_id)
                    self._stats.deadlocks += 1
                    raise DeadlockError(f"Deadlock detected: {cycle}")

            # Enqueue and wait
            req = LockRequest(tx_id=tx_id, mode=mode, granted=False)
            entry.wait_queue.append(req)
            self._stats.waits += 1

            while not req.granted:
                remaining = deadline - time.time()
                if remaining <= 0:
                    if req in entry.wait_queue:
                        entry.wait_queue.remove(req)
                    self._wait_graph.remove_edges_for(tx_id)
                    self._stats.timeouts += 1
                    raise LockTimeoutError(f"Timeout waiting for lock on {resource}")
                self._condition.wait(timeout=min(remaining, 0.1))

            self._wait_graph.remove_edges_for(tx_id)
            return LockResult.GRANTED

    def _can_grant_ahead_of_waiters(self, entry: LockEntry, mode: LockMode) -> bool:
        """Allow granting if all waiters are compatible (prevents starvation for shared locks)."""
        # Simple policy: don't grant ahead of waiting exclusive requests
        for req in entry.wait_queue:
            if not COMPATIBILITY.get(mode, {}).get(req.mode, False):
                return False
        return True

    def release(self, tx_id: int, resource: ResourceId) -> bool:
        """Release a specific lock held by tx_id. Returns True if released."""
        with self._condition:
            return self._release_internal(tx_id, resource)

    def _release_internal(self, tx_id: int, resource: ResourceId) -> bool:
        """Internal release (caller holds lock)."""
        key = resource.key
        if key not in self._entries:
            return False
        entry = self._entries[key]
        if tx_id not in entry.granted:
            return False

        del entry.granted[tx_id]
        if tx_id in self._tx_locks and key in self._tx_locks[tx_id]:
            del self._tx_locks[tx_id][key]

        self._stats.releases += 1

        # Grant waiters
        self._grant_waiters(entry)

        # Cleanup empty entries
        if entry.is_free:
            del self._entries[key]

        self._condition.notify_all()
        return True

    def release_all(self, tx_id: int) -> int:
        """Release all locks held by a transaction. Returns count released."""
        with self._condition:
            keys = list(self._tx_locks.get(tx_id, {}).keys())
            count = 0
            for key in keys:
                if key in self._entries:
                    entry = self._entries[key]
                    if tx_id in entry.granted:
                        del entry.granted[tx_id]
                        self._stats.releases += 1
                        count += 1
                        self._grant_waiters(entry)
                        if entry.is_free:
                            del self._entries[key]
            self._tx_locks.pop(tx_id, None)
            self._wait_graph.remove_node(tx_id)
            self._condition.notify_all()
            return count

    def _grant_waiters(self, entry: LockEntry):
        """Try to grant locks to waiters in FIFO order."""
        granted_any = True
        while granted_any:
            granted_any = False
            new_queue = deque()
            for req in entry.wait_queue:
                if entry.is_compatible(req.mode, exclude_tx=req.tx_id):
                    # Grant it
                    entry.granted[req.tx_id] = req.mode
                    self._tx_locks[req.tx_id][entry.resource.key] = req.mode
                    req.granted = True
                    self._stats.grants += 1
                    granted_any = True
                else:
                    new_queue.append(req)
            entry.wait_queue = new_queue
            if not granted_any:
                break

    def try_acquire(self, tx_id: int, resource: ResourceId, mode: LockMode,
                    auto_intention: bool = True) -> LockResult:
        """Non-blocking acquire. Returns GRANTED/UPGRADED or DENIED (never waits)."""
        try:
            return self.acquire(tx_id, resource, mode, timeout=0.0, auto_intention=auto_intention)
        except LockTimeoutError:
            return LockResult.DENIED
        except DeadlockError:
            return LockResult.DEADLOCK

    def is_locked(self, resource: ResourceId) -> bool:
        """Check if a resource has any locks."""
        with self._lock:
            key = resource.key
            return key in self._entries and len(self._entries[key].granted) > 0

    def get_lock_mode(self, tx_id: int, resource: ResourceId) -> Optional[LockMode]:
        """Get the lock mode held by tx_id on resource, or None."""
        with self._lock:
            key = resource.key
            if key in self._entries:
                return self._entries[key].granted.get(tx_id)
            return None

    def get_holders(self, resource: ResourceId) -> dict[int, LockMode]:
        """Get all lock holders on a resource."""
        with self._lock:
            key = resource.key
            if key in self._entries:
                return dict(self._entries[key].granted)
            return {}

    def get_tx_locks(self, tx_id: int) -> dict[str, LockMode]:
        """Get all locks held by a transaction."""
        with self._lock:
            return dict(self._tx_locks.get(tx_id, {}))

    def get_waiter_count(self, resource: ResourceId) -> int:
        """Get number of transactions waiting for a resource."""
        with self._lock:
            key = resource.key
            if key in self._entries:
                return len(self._entries[key].wait_queue)
            return 0

    # ============================================================
    # Lock escalation
    # ============================================================

    def escalate(self, tx_id: int, db: str, table: str,
                 target_mode: LockMode = LockMode.X) -> LockResult:
        """
        Escalate row/page locks to a table-level lock.
        Releases all row/page locks on the table and acquires a table lock.
        """
        with self._condition:
            table_res = make_table(db, table)
            prefix_pg = f"pg:{db}.{table}."
            prefix_row = f"row:{db}.{table}."

            # Count child locks
            child_keys = []
            for key in list(self._tx_locks.get(tx_id, {}).keys()):
                if key.startswith(prefix_pg) or key.startswith(prefix_row):
                    child_keys.append(key)

            # Release child locks
            for key in child_keys:
                if key in self._entries:
                    entry = self._entries[key]
                    if tx_id in entry.granted:
                        del entry.granted[tx_id]
                        self._stats.releases += 1
                        self._grant_waiters(entry)
                        if entry.is_free:
                            del self._entries[key]
                if tx_id in self._tx_locks and key in self._tx_locks[tx_id]:
                    del self._tx_locks[tx_id][key]

            self._condition.notify_all()

        # Now acquire table lock (outside inner lock to avoid re-entrancy issues)
        result = self.acquire(tx_id, table_res, target_mode, auto_intention=True)
        if result in (LockResult.GRANTED, LockResult.UPGRADED):
            self._stats.escalations += 1
        return result

    def check_escalation(self, tx_id: int, db: str, table: str) -> bool:
        """Check if tx_id should escalate locks on this table."""
        with self._lock:
            prefix_pg = f"pg:{db}.{table}."
            prefix_row = f"row:{db}.{table}."
            count = 0
            for key in self._tx_locks.get(tx_id, {}):
                if key.startswith(prefix_pg) or key.startswith(prefix_row):
                    count += 1
            return count >= self._escalation_threshold

    def auto_escalate(self, tx_id: int, db: str, table: str,
                      target_mode: LockMode = LockMode.X) -> Optional[LockResult]:
        """Escalate if threshold exceeded. Returns result or None if no escalation needed."""
        if self.check_escalation(tx_id, db, table):
            return self.escalate(tx_id, db, table, target_mode)
        return None

    # ============================================================
    # Deadlock detection (explicit)
    # ============================================================

    def detect_deadlock(self) -> list[int]:
        """Run cycle detection on wait-for graph. Returns cycle or []."""
        with self._lock:
            return self._wait_graph.detect_cycle()

    def get_wait_graph(self) -> dict[int, set[int]]:
        """Get current wait-for graph edges."""
        with self._lock:
            return self._wait_graph.edges

    # ============================================================
    # Downgrade
    # ============================================================

    def downgrade(self, tx_id: int, resource: ResourceId, new_mode: LockMode) -> bool:
        """
        Downgrade a lock to a less restrictive mode.
        Returns True if successful, False if not holding or new_mode is more restrictive.
        """
        with self._condition:
            key = resource.key
            if key not in self._entries:
                return False
            entry = self._entries[key]
            if tx_id not in entry.granted:
                return False
            current = entry.granted[tx_id]
            if new_mode.value >= current.value:
                return False  # Not a downgrade
            entry.granted[tx_id] = new_mode
            self._tx_locks[tx_id][key] = new_mode
            self._stats.downgrades += 1
            # Re-grant waiters since we've weakened
            self._grant_waiters(entry)
            self._condition.notify_all()
            return True

    # ============================================================
    # Stats and diagnostics
    # ============================================================

    @property
    def stats(self) -> 'LockStats':
        return self._stats

    def get_all_locks(self) -> dict[str, dict[int, LockMode]]:
        """Get all current locks for diagnostics."""
        with self._lock:
            result = {}
            for key, entry in self._entries.items():
                if entry.granted:
                    result[key] = dict(entry.granted)
            return result

    def get_all_waiters(self) -> dict[str, list[tuple[int, LockMode]]]:
        """Get all current waiters."""
        with self._lock:
            result = {}
            for key, entry in self._entries.items():
                if entry.wait_queue:
                    result[key] = [(r.tx_id, r.mode) for r in entry.wait_queue]
            return result

    def lock_count(self) -> int:
        """Total number of granted locks."""
        with self._lock:
            return sum(len(e.granted) for e in self._entries.values())

    def tx_count(self) -> int:
        """Number of transactions holding locks."""
        with self._lock:
            return len(self._tx_locks)

    def reset(self):
        """Clear all locks and state."""
        with self._condition:
            self._entries.clear()
            self._tx_locks.clear()
            self._wait_graph.clear()
            self._stats = LockStats()
            self._condition.notify_all()


@dataclass
class LockStats:
    """Lock manager statistics."""
    grants: int = 0
    releases: int = 0
    upgrades: int = 0
    downgrades: int = 0
    waits: int = 0
    timeouts: int = 0
    deadlocks: int = 0
    escalations: int = 0

    def summary(self) -> dict[str, int]:
        return {
            'grants': self.grants,
            'releases': self.releases,
            'upgrades': self.upgrades,
            'downgrades': self.downgrades,
            'waits': self.waits,
            'timeouts': self.timeouts,
            'deadlocks': self.deadlocks,
            'escalations': self.escalations,
        }


# ============================================================
# Lock Manager Analyzer
# ============================================================

class LockManagerAnalyzer:
    """Diagnostic tool for analyzing lock manager state."""

    def __init__(self, manager: LockManager):
        self.manager = manager

    def contention_report(self) -> dict[str, Any]:
        """Analyze lock contention."""
        all_locks = self.manager.get_all_locks()
        all_waiters = self.manager.get_all_waiters()
        stats = self.manager.stats.summary()

        hot_resources = []
        for key in set(list(all_locks.keys()) + list(all_waiters.keys())):
            holders = len(all_locks.get(key, {}))
            waiters = len(all_waiters.get(key, []))
            if holders > 1 or waiters > 0:
                hot_resources.append({
                    'resource': key,
                    'holders': holders,
                    'waiters': waiters,
                    'contention': holders + waiters,
                })
        hot_resources.sort(key=lambda x: x['contention'], reverse=True)

        return {
            'total_locks': self.manager.lock_count(),
            'total_transactions': self.manager.tx_count(),
            'hot_resources': hot_resources[:10],
            'stats': stats,
        }

    def tx_report(self, tx_id: int) -> dict[str, Any]:
        """Report locks held by a specific transaction."""
        locks = self.manager.get_tx_locks(tx_id)
        by_level = defaultdict(list)
        for key, mode in locks.items():
            level = key.split(':')[0]
            by_level[level].append({'resource': key, 'mode': mode.name})
        return {
            'tx_id': tx_id,
            'total_locks': len(locks),
            'by_level': dict(by_level),
        }

    def deadlock_report(self) -> dict[str, Any]:
        """Report on deadlock state."""
        wait_graph = self.manager.get_wait_graph()
        cycle = self.manager.detect_deadlock()
        return {
            'wait_graph_edges': {k: list(v) for k, v in wait_graph.items()},
            'cycle_detected': len(cycle) > 0,
            'cycle': cycle,
            'total_deadlocks': self.manager.stats.deadlocks,
        }

    def escalation_report(self) -> dict[str, Any]:
        """Report on lock escalation state."""
        all_locks = self.manager.get_all_locks()
        # Find tables with many row/page locks
        table_child_counts: dict[str, int] = defaultdict(int)
        for key in all_locks:
            if key.startswith('row:') or key.startswith('pg:'):
                parts = key.split(':')[1].split('.')
                if len(parts) >= 2:
                    table_key = f"{parts[0]}.{parts[1]}"
                    table_child_counts[table_key] += 1

        candidates = [
            {'table': k, 'child_locks': v, 'threshold': self.manager._escalation_threshold}
            for k, v in table_child_counts.items()
            if v >= self.manager._escalation_threshold // 2  # Report at 50% threshold
        ]
        candidates.sort(key=lambda x: x['child_locks'], reverse=True)

        return {
            'escalation_threshold': self.manager._escalation_threshold,
            'total_escalations': self.manager.stats.escalations,
            'candidates': candidates,
        }


# ============================================================
# Two-Phase Locking (2PL) Helper
# ============================================================

class TwoPhaseLockHelper:
    """
    Enforces strict two-phase locking protocol.

    Growing phase: locks can be acquired, not released.
    Shrinking phase: locks can be released, not acquired.
    Strict 2PL: all locks held until commit/abort.
    """

    def __init__(self, manager: LockManager):
        self.manager = manager
        self._phases: dict[int, str] = {}  # tx_id -> 'growing' or 'shrinking'

    def begin(self, tx_id: int):
        """Start a transaction in growing phase."""
        self._phases[tx_id] = 'growing'

    def acquire(self, tx_id: int, resource: ResourceId, mode: LockMode,
                timeout: Optional[float] = None) -> LockResult:
        """Acquire lock (only in growing phase)."""
        phase = self._phases.get(tx_id, 'growing')
        if phase == 'shrinking':
            raise LockError(f"Cannot acquire locks in shrinking phase (tx {tx_id})")
        return self.manager.acquire(tx_id, resource, mode, timeout=timeout)

    def release(self, tx_id: int, resource: ResourceId) -> bool:
        """Release lock (transitions to shrinking phase)."""
        self._phases[tx_id] = 'shrinking'
        return self.manager.release(tx_id, resource)

    def release_all(self, tx_id: int) -> int:
        """Release all locks (strict 2PL: call at commit/abort)."""
        self._phases.pop(tx_id, None)
        return self.manager.release_all(tx_id)

    def phase(self, tx_id: int) -> str:
        """Get current phase."""
        return self._phases.get(tx_id, 'unknown')


# ============================================================
# Multi-Granularity Lock Protocol
# ============================================================

class MultiGranularityLocker:
    """
    Convenience layer for hierarchical locking with automatic intention locks.

    Enforces the intention locking protocol:
    - To lock a row in S mode, must hold IS (or higher) on table, page
    - To lock a row in X mode, must hold IX (or higher) on table, page
    """

    def __init__(self, manager: LockManager, db: str = "default"):
        self.manager = manager
        self.db = db

    def lock_table_shared(self, tx_id: int, table: str, timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_table(self.db, table), LockMode.S, timeout=timeout)

    def lock_table_exclusive(self, tx_id: int, table: str, timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_table(self.db, table), LockMode.X, timeout=timeout)

    def lock_row_shared(self, tx_id: int, table: str, page: int, row: int,
                        timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_row(self.db, table, page, row), LockMode.S, timeout=timeout)

    def lock_row_exclusive(self, tx_id: int, table: str, page: int, row: int,
                           timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_row(self.db, table, page, row), LockMode.X, timeout=timeout)

    def lock_page_shared(self, tx_id: int, table: str, page: int,
                         timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_page(self.db, table, page), LockMode.S, timeout=timeout)

    def lock_page_exclusive(self, tx_id: int, table: str, page: int,
                            timeout: Optional[float] = None) -> LockResult:
        return self.manager.acquire(tx_id, make_page(self.db, table, page), LockMode.X, timeout=timeout)

    def unlock_row(self, tx_id: int, table: str, page: int, row: int) -> bool:
        return self.manager.release(tx_id, make_row(self.db, table, page, row))

    def unlock_table(self, tx_id: int, table: str) -> bool:
        return self.manager.release(tx_id, make_table(self.db, table))

    def unlock_all(self, tx_id: int) -> int:
        return self.manager.release_all(tx_id)
