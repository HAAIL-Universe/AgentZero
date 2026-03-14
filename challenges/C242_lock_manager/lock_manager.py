"""
C242: Lock Manager -- Two-Phase Locking with Deadlock Detection

A database lock manager implementing:
- Lock modes: Shared (S), Exclusive (X), Intent Shared (IS), Intent Exclusive (IX), SIX
- Lock granularity: Database, Table, Page, Row
- Two-Phase Locking (2PL) with strict and rigorous variants
- Wait-For Graph deadlock detection with victim selection
- Lock escalation (row -> page -> table) based on thresholds
- Lock compatibility matrix
- Timeout-based deadlock prevention
- Lock queuing with FIFO fairness
"""

import time
import threading
from enum import IntEnum, auto
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Lock Modes
# ---------------------------------------------------------------------------

class LockMode(IntEnum):
    """Lock modes ordered by strength."""
    NONE = 0
    IS = 1    # Intent Shared
    IX = 2    # Intent Exclusive
    S = 3     # Shared
    SIX = 4   # Shared + Intent Exclusive
    X = 5     # Exclusive


# Compatibility matrix: COMPAT[held][requested] = True if compatible
COMPAT = {
    LockMode.IS:  {LockMode.IS: True,  LockMode.IX: True,  LockMode.S: True,  LockMode.SIX: True,  LockMode.X: False},
    LockMode.IX:  {LockMode.IS: True,  LockMode.IX: True,  LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
    LockMode.S:   {LockMode.IS: True,  LockMode.IX: False, LockMode.S: True,  LockMode.SIX: False, LockMode.X: False},
    LockMode.SIX: {LockMode.IS: True,  LockMode.IX: False, LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
    LockMode.X:   {LockMode.IS: False, LockMode.IX: False, LockMode.S: False, LockMode.SIX: False, LockMode.X: False},
}

# Mode upgrade: what mode covers both?
UPGRADE = {
    (LockMode.IS, LockMode.IS): LockMode.IS,
    (LockMode.IS, LockMode.IX): LockMode.IX,
    (LockMode.IS, LockMode.S):  LockMode.S,
    (LockMode.IS, LockMode.SIX): LockMode.SIX,
    (LockMode.IS, LockMode.X):  LockMode.X,
    (LockMode.IX, LockMode.IS): LockMode.IX,
    (LockMode.IX, LockMode.IX): LockMode.IX,
    (LockMode.IX, LockMode.S):  LockMode.SIX,
    (LockMode.IX, LockMode.SIX): LockMode.SIX,
    (LockMode.IX, LockMode.X):  LockMode.X,
    (LockMode.S, LockMode.IS):  LockMode.S,
    (LockMode.S, LockMode.IX):  LockMode.SIX,
    (LockMode.S, LockMode.S):   LockMode.S,
    (LockMode.S, LockMode.SIX): LockMode.SIX,
    (LockMode.S, LockMode.X):   LockMode.X,
    (LockMode.SIX, LockMode.IS): LockMode.SIX,
    (LockMode.SIX, LockMode.IX): LockMode.SIX,
    (LockMode.SIX, LockMode.S):  LockMode.SIX,
    (LockMode.SIX, LockMode.SIX): LockMode.SIX,
    (LockMode.SIX, LockMode.X):  LockMode.X,
    (LockMode.X, LockMode.IS):  LockMode.X,
    (LockMode.X, LockMode.IX):  LockMode.X,
    (LockMode.X, LockMode.S):   LockMode.X,
    (LockMode.X, LockMode.SIX): LockMode.X,
    (LockMode.X, LockMode.X):   LockMode.X,
}


# ---------------------------------------------------------------------------
# Lock Granularity
# ---------------------------------------------------------------------------

class LockGranularity(IntEnum):
    DATABASE = 0
    TABLE = 1
    PAGE = 2
    ROW = 3


# ---------------------------------------------------------------------------
# Resource ID
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceId:
    """Identifies a lockable resource at some granularity level."""
    granularity: LockGranularity
    database: str = "default"
    table: Optional[str] = None
    page: Optional[int] = None
    row: Optional[int] = None

    def parent(self):
        """Return the parent resource (coarser granularity)."""
        if self.granularity == LockGranularity.ROW:
            return ResourceId(LockGranularity.PAGE, self.database, self.table, self.page)
        elif self.granularity == LockGranularity.PAGE:
            return ResourceId(LockGranularity.TABLE, self.database, self.table)
        elif self.granularity == LockGranularity.TABLE:
            return ResourceId(LockGranularity.DATABASE, self.database)
        return None

    def __repr__(self):
        parts = [self.database]
        if self.table is not None:
            parts.append(self.table)
        if self.page is not None:
            parts.append(f"p{self.page}")
        if self.row is not None:
            parts.append(f"r{self.row}")
        return f"Res({'/'.join(parts)})"


def make_resource(database="default", table=None, page=None, row=None):
    """Helper to create a ResourceId with auto-detected granularity."""
    if row is not None:
        return ResourceId(LockGranularity.ROW, database, table, page, row)
    elif page is not None:
        return ResourceId(LockGranularity.PAGE, database, table, page)
    elif table is not None:
        return ResourceId(LockGranularity.TABLE, database, table)
    else:
        return ResourceId(LockGranularity.DATABASE, database)


# ---------------------------------------------------------------------------
# Lock Request and Grant
# ---------------------------------------------------------------------------

class LockStatus(IntEnum):
    GRANTED = 0
    WAITING = 1
    CONVERTING = 2  # Upgrade in progress


@dataclass
class LockRequest:
    """A pending or granted lock request."""
    txn_id: int
    mode: LockMode
    resource: ResourceId
    status: LockStatus = LockStatus.WAITING
    timestamp: float = field(default_factory=time.monotonic)
    convert_mode: Optional[LockMode] = None  # Target mode for conversion


# ---------------------------------------------------------------------------
# Lock Table Entry
# ---------------------------------------------------------------------------

@dataclass
class LockTableEntry:
    """Per-resource lock state."""
    resource: ResourceId
    granted_group: list = field(default_factory=list)    # List[LockRequest]
    wait_queue: deque = field(default_factory=deque)      # Deque[LockRequest]
    granted_mode: LockMode = LockMode.NONE                # Combined granted mode

    def update_granted_mode(self):
        """Recompute the combined granted mode."""
        if not self.granted_group:
            self.granted_mode = LockMode.NONE
            return
        mode = LockMode.NONE
        for req in self.granted_group:
            if mode == LockMode.NONE:
                mode = req.mode
            else:
                mode = UPGRADE.get((mode, req.mode), LockMode.X)
        self.granted_mode = mode


# ---------------------------------------------------------------------------
# Deadlock Detector (Wait-For Graph)
# ---------------------------------------------------------------------------

class DeadlockDetector:
    """Builds and analyzes a wait-for graph to detect deadlocks."""

    def __init__(self):
        self.edges = defaultdict(set)  # txn_id -> set of txn_ids it waits for

    def add_edge(self, waiter, holder):
        if waiter != holder:
            self.edges[waiter].add(holder)

    def remove_edges_for(self, txn_id):
        self.edges.pop(txn_id, None)
        for waiters in self.edges.values():
            waiters.discard(txn_id)

    def clear(self):
        self.edges.clear()

    def find_cycle(self):
        """Find a cycle in the wait-for graph. Returns the cycle as a list of txn_ids, or None."""
        visited = set()
        rec_stack = set()
        parent = {}

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    parent[neighbor] = node
                    result = dfs(neighbor, path)
                    if result is not None:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle -- extract it
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.discard(node)
            return None

        for node in list(self.edges.keys()):
            if node not in visited:
                result = dfs(node, [])
                if result is not None:
                    return result
        return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LockError(Exception):
    pass

class DeadlockError(LockError):
    def __init__(self, txn_id, cycle=None):
        self.txn_id = txn_id
        self.cycle = cycle
        super().__init__(f"Deadlock detected: txn {txn_id} aborted (cycle: {cycle})")

class LockTimeoutError(LockError):
    def __init__(self, txn_id, resource):
        self.txn_id = txn_id
        self.resource = resource
        super().__init__(f"Lock timeout: txn {txn_id} on {resource}")

class LockEscalationError(LockError):
    pass


# ---------------------------------------------------------------------------
# Transaction State
# ---------------------------------------------------------------------------

class TxnPhase(IntEnum):
    GROWING = 0     # Can acquire locks
    SHRINKING = 1   # Can only release locks (strict 2PL releases at commit)


@dataclass
class TransactionState:
    """Tracks lock state for a transaction."""
    txn_id: int
    phase: TxnPhase = TxnPhase.GROWING
    held_locks: dict = field(default_factory=dict)  # ResourceId -> LockMode
    start_time: float = field(default_factory=time.monotonic)
    lock_count: int = 0
    row_locks_per_table: dict = field(default_factory=lambda: defaultdict(int))
    priority: int = 0  # Lower = higher priority (for victim selection)


# ---------------------------------------------------------------------------
# Lock Manager
# ---------------------------------------------------------------------------

class LockManager:
    """
    Two-Phase Lock Manager with:
    - Multi-granularity locking (IS/IX/S/SIX/X)
    - Strict 2PL (locks released at commit/abort)
    - Wait-for graph deadlock detection
    - Lock escalation
    - Timeout support
    """

    def __init__(self, deadlock_detection=True, escalation_threshold=10,
                 default_timeout=5.0, strict_2pl=True):
        self._lock = threading.Lock()
        self._lock_table = {}           # ResourceId -> LockTableEntry
        self._txn_states = {}           # txn_id -> TransactionState
        self._deadlock_detector = DeadlockDetector()
        self._waiters = {}              # txn_id -> threading.Event
        self._deadlock_detection = deadlock_detection
        self._escalation_threshold = escalation_threshold
        self._default_timeout = default_timeout
        self._strict_2pl = strict_2pl
        self._stats = {
            "locks_acquired": 0,
            "locks_released": 0,
            "deadlocks_detected": 0,
            "escalations": 0,
            "timeouts": 0,
            "upgrades": 0,
        }

    # -- Transaction lifecycle -------------------------------------------------

    def begin_txn(self, txn_id, priority=0):
        """Register a new transaction."""
        with self._lock:
            if txn_id in self._txn_states:
                raise LockError(f"Transaction {txn_id} already exists")
            self._txn_states[txn_id] = TransactionState(txn_id=txn_id, priority=priority)

    def commit_txn(self, txn_id):
        """Commit: release all locks held by this transaction."""
        self._end_txn(txn_id)

    def abort_txn(self, txn_id):
        """Abort: release all locks held by this transaction."""
        self._end_txn(txn_id)

    def _end_txn(self, txn_id):
        """Release all locks for a transaction."""
        with self._lock:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                return
            # Release all locks
            resources = list(txn.held_locks.keys())
            for resource in resources:
                self._release_lock_internal(txn_id, resource)
            # Clean up
            self._deadlock_detector.remove_edges_for(txn_id)
            del self._txn_states[txn_id]
            self._waiters.pop(txn_id, None)

    # -- Lock acquisition ------------------------------------------------------

    def acquire(self, txn_id, resource, mode, timeout=None):
        """
        Acquire a lock on a resource.

        Returns True on success.
        Raises DeadlockError or LockTimeoutError on failure.
        """
        if timeout is None:
            timeout = self._default_timeout

        with self._lock:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                raise LockError(f"Transaction {txn_id} not registered")

            if self._strict_2pl and txn.phase == TxnPhase.SHRINKING:
                raise LockError(f"Transaction {txn_id} in shrinking phase, cannot acquire locks")

            # Acquire intention locks on ancestors
            self._acquire_intention_locks(txn_id, resource, mode)

            # Check if we already hold a lock on this resource
            existing_mode = txn.held_locks.get(resource)
            if existing_mode is not None:
                if existing_mode >= mode:
                    return True  # Already have sufficient lock
                # Need upgrade
                return self._upgrade_lock(txn_id, resource, existing_mode, mode, timeout)

            # Try to grant immediately
            entry = self._get_or_create_entry(resource)
            if self._can_grant(entry, txn_id, mode):
                self._grant_lock(entry, txn_id, resource, mode)
                return True

            # Must wait
            return self._wait_for_lock(txn_id, resource, mode, entry, timeout)

    def _acquire_intention_locks(self, txn_id, resource, mode):
        """Acquire intention locks on all ancestor resources."""
        intent_mode = LockMode.IS if mode in (LockMode.S, LockMode.IS) else LockMode.IX
        parent = resource.parent()
        while parent is not None:
            txn = self._txn_states[txn_id]
            existing = txn.held_locks.get(parent)
            if existing is not None:
                needed = UPGRADE.get((existing, intent_mode), intent_mode)
                if needed <= existing:
                    parent = parent.parent()
                    continue
                # Upgrade the intention lock
                entry = self._get_or_create_entry(parent)
                self._do_upgrade(entry, txn_id, parent, existing, needed)
            else:
                entry = self._get_or_create_entry(parent)
                if self._can_grant(entry, txn_id, intent_mode):
                    self._grant_lock(entry, txn_id, parent, intent_mode)
                else:
                    # Cannot acquire intention lock -- propagate failure
                    raise LockTimeoutError(txn_id, parent)
            parent = parent.parent()

    def _can_grant(self, entry, txn_id, mode):
        """Check if a lock can be granted immediately."""
        # If wait queue is non-empty, must wait (FIFO fairness)
        if entry.wait_queue:
            return False

        # Check compatibility with all granted locks
        for req in entry.granted_group:
            if req.txn_id == txn_id:
                continue  # Same txn -- compatible
            if not COMPAT.get(req.mode, {}).get(mode, False):
                return False
        return True

    def _grant_lock(self, entry, txn_id, resource, mode):
        """Grant a lock to a transaction."""
        req = LockRequest(txn_id=txn_id, mode=mode, resource=resource, status=LockStatus.GRANTED)
        entry.granted_group.append(req)
        entry.update_granted_mode()

        txn = self._txn_states[txn_id]
        txn.held_locks[resource] = mode
        txn.lock_count += 1

        # Track row locks per table for escalation
        if resource.granularity == LockGranularity.ROW:
            table_key = (resource.database, resource.table)
            txn.row_locks_per_table[table_key] += 1

        self._stats["locks_acquired"] += 1

    def _upgrade_lock(self, txn_id, resource, old_mode, new_mode, timeout):
        """Upgrade an existing lock to a stronger mode."""
        target = UPGRADE.get((old_mode, new_mode), LockMode.X)
        if target <= old_mode:
            return True

        entry = self._get_or_create_entry(resource)

        # Check if upgrade is possible immediately
        can_upgrade = True
        for req in entry.granted_group:
            if req.txn_id == txn_id:
                continue
            if not COMPAT.get(req.mode, {}).get(target, False):
                can_upgrade = False
                break

        if can_upgrade:
            self._do_upgrade(entry, txn_id, resource, old_mode, target)
            return True

        # Must wait for upgrade
        return self._wait_for_upgrade(txn_id, resource, old_mode, target, entry, timeout)

    def _do_upgrade(self, entry, txn_id, resource, old_mode, new_mode):
        """Perform the actual lock upgrade."""
        for req in entry.granted_group:
            if req.txn_id == txn_id:
                req.mode = new_mode
                break
        entry.update_granted_mode()
        self._txn_states[txn_id].held_locks[resource] = new_mode
        self._stats["upgrades"] += 1

    def _wait_for_lock(self, txn_id, resource, mode, entry, timeout):
        """Wait for a lock to become available."""
        # Add to wait queue
        req = LockRequest(txn_id=txn_id, mode=mode, resource=resource, status=LockStatus.WAITING)
        entry.wait_queue.append(req)

        # Build wait-for edges
        for granted in entry.granted_group:
            if granted.txn_id != txn_id:
                self._deadlock_detector.add_edge(txn_id, granted.txn_id)

        # Check for deadlock
        if self._deadlock_detection:
            cycle = self._deadlock_detector.find_cycle()
            if cycle is not None:
                # Remove from wait queue
                entry.wait_queue = deque(r for r in entry.wait_queue if r.txn_id != txn_id)
                self._deadlock_detector.remove_edges_for(txn_id)
                self._stats["deadlocks_detected"] += 1
                victim = self._select_victim(cycle)
                raise DeadlockError(victim, cycle)

        # Create wait event
        event = threading.Event()
        self._waiters[txn_id] = event

        # Release main lock and wait
        self._lock.release()
        try:
            granted = event.wait(timeout=timeout)
            if not granted:
                # Timeout
                with self._lock:
                    entry.wait_queue = deque(r for r in entry.wait_queue if r.txn_id != txn_id)
                    self._deadlock_detector.remove_edges_for(txn_id)
                    self._waiters.pop(txn_id, None)
                    self._stats["timeouts"] += 1
                raise LockTimeoutError(txn_id, resource)
            return True
        finally:
            self._lock.acquire()

    def _wait_for_upgrade(self, txn_id, resource, old_mode, target_mode, entry, timeout):
        """Wait for a lock upgrade."""
        # Mark as converting
        for req in entry.granted_group:
            if req.txn_id == txn_id:
                req.status = LockStatus.CONVERTING
                req.convert_mode = target_mode
                break

        # Build wait-for edges
        for granted in entry.granted_group:
            if granted.txn_id != txn_id:
                self._deadlock_detector.add_edge(txn_id, granted.txn_id)

        # Check for deadlock
        if self._deadlock_detection:
            cycle = self._deadlock_detector.find_cycle()
            if cycle is not None:
                for req in entry.granted_group:
                    if req.txn_id == txn_id:
                        req.status = LockStatus.GRANTED
                        req.convert_mode = None
                        break
                self._deadlock_detector.remove_edges_for(txn_id)
                self._stats["deadlocks_detected"] += 1
                victim = self._select_victim(cycle)
                raise DeadlockError(victim, cycle)

        event = threading.Event()
        self._waiters[txn_id] = event

        self._lock.release()
        try:
            granted = event.wait(timeout=timeout)
            if not granted:
                with self._lock:
                    for req in entry.granted_group:
                        if req.txn_id == txn_id:
                            req.status = LockStatus.GRANTED
                            req.convert_mode = None
                            break
                    self._deadlock_detector.remove_edges_for(txn_id)
                    self._waiters.pop(txn_id, None)
                    self._stats["timeouts"] += 1
                raise LockTimeoutError(txn_id, resource)
            return True
        finally:
            self._lock.acquire()

    def _select_victim(self, cycle):
        """Select the youngest (lowest priority) transaction as deadlock victim."""
        # Pick the transaction with highest priority number (lowest priority)
        # If tie, pick youngest (latest start_time)
        best_victim = cycle[0]
        best_priority = -1
        best_time = 0

        for txn_id in cycle:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                continue
            if txn.priority > best_priority or (txn.priority == best_priority and txn.start_time > best_time):
                best_victim = txn_id
                best_priority = txn.priority
                best_time = txn.start_time

        return best_victim

    # -- Lock release ----------------------------------------------------------

    def release(self, txn_id, resource):
        """Release a single lock. In strict 2PL, transitions to shrinking phase."""
        with self._lock:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                raise LockError(f"Transaction {txn_id} not registered")

            if resource not in txn.held_locks:
                raise LockError(f"Transaction {txn_id} does not hold lock on {resource}")

            if self._strict_2pl:
                txn.phase = TxnPhase.SHRINKING

            self._release_lock_internal(txn_id, resource)

    def _release_lock_internal(self, txn_id, resource):
        """Internal lock release (no phase check)."""
        entry = self._lock_table.get(resource)
        if entry is None:
            return

        # Remove from granted group
        entry.granted_group = [r for r in entry.granted_group if r.txn_id != txn_id]
        entry.update_granted_mode()

        # Update txn state
        txn = self._txn_states.get(txn_id)
        if txn is not None:
            old_mode = txn.held_locks.pop(resource, None)
            if old_mode is not None:
                txn.lock_count -= 1
                if resource.granularity == LockGranularity.ROW:
                    table_key = (resource.database, resource.table)
                    txn.row_locks_per_table[table_key] = max(0, txn.row_locks_per_table[table_key] - 1)
                self._stats["locks_released"] += 1

        # Remove from wait queue too (if aborting while waiting)
        entry.wait_queue = deque(r for r in entry.wait_queue if r.txn_id != txn_id)

        # Try to grant waiting requests
        self._process_wait_queue(entry)

        # Clean up empty entries
        if not entry.granted_group and not entry.wait_queue:
            del self._lock_table[resource]

    def _process_wait_queue(self, entry):
        """Try to grant locks to waiting transactions."""
        changed = True
        while changed:
            changed = False

            # First: process conversions (upgrades from granted group)
            for req in entry.granted_group:
                if req.status == LockStatus.CONVERTING:
                    can_convert = True
                    for other in entry.granted_group:
                        if other.txn_id == req.txn_id:
                            continue
                        if not COMPAT.get(other.mode, {}).get(req.convert_mode, False):
                            can_convert = False
                            break
                    if can_convert:
                        req.mode = req.convert_mode
                        req.convert_mode = None
                        req.status = LockStatus.GRANTED
                        entry.update_granted_mode()
                        txn = self._txn_states.get(req.txn_id)
                        if txn:
                            txn.held_locks[req.resource] = req.mode
                        self._stats["upgrades"] += 1
                        self._deadlock_detector.remove_edges_for(req.txn_id)
                        event = self._waiters.pop(req.txn_id, None)
                        if event:
                            event.set()
                        changed = True

            # Then: process new requests from wait queue (FIFO)
            new_queue = deque()
            while entry.wait_queue:
                req = entry.wait_queue.popleft()
                if self._can_grant_to_waiter(entry, req):
                    req.status = LockStatus.GRANTED
                    entry.granted_group.append(req)
                    entry.update_granted_mode()

                    txn = self._txn_states.get(req.txn_id)
                    if txn:
                        txn.held_locks[req.resource] = req.mode
                        txn.lock_count += 1
                        if req.resource.granularity == LockGranularity.ROW:
                            table_key = (req.resource.database, req.resource.table)
                            txn.row_locks_per_table[table_key] += 1

                    self._stats["locks_acquired"] += 1
                    self._deadlock_detector.remove_edges_for(req.txn_id)
                    event = self._waiters.pop(req.txn_id, None)
                    if event:
                        event.set()
                    changed = True
                else:
                    new_queue.append(req)
                    # FIFO: remaining requests stay queued behind this one
                    while entry.wait_queue:
                        new_queue.append(entry.wait_queue.popleft())
                    break
            entry.wait_queue = new_queue

    def _can_grant_to_waiter(self, entry, request):
        """Check if a waiting request can now be granted."""
        for req in entry.granted_group:
            if req.txn_id == request.txn_id:
                continue
            if not COMPAT.get(req.mode, {}).get(request.mode, False):
                return False
        return True

    # -- Lock escalation -------------------------------------------------------

    def check_escalation(self, txn_id, database="default", table=None):
        """
        Check if row locks should be escalated to table lock.
        Returns True if escalation occurred.
        """
        with self._lock:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                return False

            table_key = (database, table)
            row_count = txn.row_locks_per_table.get(table_key, 0)

            if row_count < self._escalation_threshold:
                return False

            return self._escalate(txn_id, database, table)

    def _escalate(self, txn_id, database, table):
        """Escalate row locks to a table lock."""
        txn = self._txn_states[txn_id]
        table_resource = make_resource(database=database, table=table)

        # Determine escalation mode: if any row lock is X, escalate to X, else S
        escalation_mode = LockMode.S
        rows_to_release = []
        for resource, mode in list(txn.held_locks.items()):
            if (resource.granularity == LockGranularity.ROW and
                resource.database == database and resource.table == table):
                rows_to_release.append(resource)
                if mode == LockMode.X or mode == LockMode.IX:
                    escalation_mode = LockMode.X

        # Also check page locks
        pages_to_release = []
        for resource, mode in list(txn.held_locks.items()):
            if (resource.granularity == LockGranularity.PAGE and
                resource.database == database and resource.table == table):
                pages_to_release.append(resource)
                if mode == LockMode.X or mode == LockMode.IX:
                    escalation_mode = LockMode.X

        # Try to acquire table lock
        entry = self._get_or_create_entry(table_resource)
        existing = txn.held_locks.get(table_resource)

        if existing is not None:
            # Upgrade existing intention lock to full lock
            target = UPGRADE.get((existing, escalation_mode), escalation_mode)
            can_upgrade = True
            for req in entry.granted_group:
                if req.txn_id == txn_id:
                    continue
                if not COMPAT.get(req.mode, {}).get(target, False):
                    can_upgrade = False
                    break
            if not can_upgrade:
                return False
            self._do_upgrade(entry, txn_id, table_resource, existing, target)
        else:
            if not self._can_grant(entry, txn_id, escalation_mode):
                return False
            self._grant_lock(entry, txn_id, table_resource, escalation_mode)

        # Release individual row and page locks
        for resource in rows_to_release + pages_to_release:
            self._release_lock_internal(txn_id, resource)

        self._stats["escalations"] += 1
        return True

    # -- Helpers ---------------------------------------------------------------

    def _get_or_create_entry(self, resource):
        """Get or create a lock table entry for a resource."""
        if resource not in self._lock_table:
            self._lock_table[resource] = LockTableEntry(resource=resource)
        return self._lock_table[resource]

    # -- Query methods ---------------------------------------------------------

    def get_locks(self, txn_id=None, resource=None):
        """Query current locks, optionally filtered by txn or resource."""
        with self._lock:
            results = []
            if resource is not None:
                entry = self._lock_table.get(resource)
                if entry:
                    for req in entry.granted_group:
                        if txn_id is None or req.txn_id == txn_id:
                            results.append({"txn_id": req.txn_id, "resource": req.resource,
                                            "mode": req.mode, "status": "granted"})
                    for req in entry.wait_queue:
                        if txn_id is None or req.txn_id == txn_id:
                            results.append({"txn_id": req.txn_id, "resource": req.resource,
                                            "mode": req.mode, "status": "waiting"})
            elif txn_id is not None:
                txn = self._txn_states.get(txn_id)
                if txn:
                    for res, mode in txn.held_locks.items():
                        results.append({"txn_id": txn_id, "resource": res, "mode": mode, "status": "granted"})
            else:
                for entry in self._lock_table.values():
                    for req in entry.granted_group:
                        results.append({"txn_id": req.txn_id, "resource": req.resource,
                                        "mode": req.mode, "status": "granted"})
            return results

    def get_wait_for_graph(self):
        """Return the current wait-for graph edges."""
        with self._lock:
            return dict(self._deadlock_detector.edges)

    def get_stats(self):
        """Return lock manager statistics."""
        with self._lock:
            return dict(self._stats)

    def get_txn_state(self, txn_id):
        """Return the state of a transaction."""
        with self._lock:
            txn = self._txn_states.get(txn_id)
            if txn is None:
                return None
            return {
                "txn_id": txn.txn_id,
                "phase": txn.phase.name,
                "lock_count": txn.lock_count,
                "held_locks": dict(txn.held_locks),
                "row_locks_per_table": dict(txn.row_locks_per_table),
            }

    def active_txn_count(self):
        """Number of active transactions."""
        with self._lock:
            return len(self._txn_states)
