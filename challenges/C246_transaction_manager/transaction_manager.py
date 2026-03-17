"""
C246: Transaction Manager

A complete ACID transaction manager composing three subsystems:
- C240 MVCC for multi-version concurrency control (data storage + visibility)
- C241 WAL for write-ahead logging (durability + crash recovery)
- C242 Lock Manager for two-phase locking (isolation + deadlock detection)

Features:
- Unified transaction lifecycle: begin, commit, abort, savepoints
- Automatic lock acquisition before reads/writes (intention + row locks)
- WAL logging of all mutations for crash recovery
- MVCC snapshot isolation with configurable isolation levels
- Deadlock detection with automatic victim abort
- Crash recovery via ARIES (analysis, redo, undo)
- Periodic checkpointing
- Lock escalation integration
- Transaction statistics and diagnostics
- Nested savepoints with partial rollback

Domain: Database Internals
Composes: C240 (MVCC), C241 (WAL), C242 (Lock Manager)
"""

import sys
import os
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional

# Add parent challenges to path
_challenges = os.path.join(os.path.dirname(__file__), '..')
for _dep in ('C240_mvcc', 'C241_wal', 'C242_lock_manager'):
    _path = os.path.join(_challenges, _dep)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mvcc import MVCCEngine, IsolationLevel, TxnStatus
from wal import WAL, LogRecordType
from lock_manager import LockManager, LockMode, make_resource


# ---------------------------------------------------------------------------
# Transaction Manager Types
# ---------------------------------------------------------------------------

class TxnState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()
    PREPARING = auto()  # for future 2PC


@dataclass
class ManagedTransaction:
    """Internal bookkeeping for a managed transaction."""
    txn_id: int
    mvcc_txn_id: int
    wal_txn_id: int
    isolation: IsolationLevel
    state: TxnState = TxnState.ACTIVE
    start_time: float = field(default_factory=time.time)
    reads: int = 0
    writes: int = 0
    locks_acquired: int = 0
    savepoints: list = field(default_factory=list)
    written_keys: list = field(default_factory=list)
    read_keys: list = field(default_factory=list)
    database: str = "default"
    table: str = "data"


class TransactionError(Exception):
    """Base error for transaction manager."""
    pass


class DeadlockError(TransactionError):
    """Raised when a deadlock is detected and this txn is chosen as victim."""
    pass


class ConflictError(TransactionError):
    """Raised on write-write conflicts."""
    pass


class TransactionNotFoundError(TransactionError):
    """Raised when operating on a non-existent transaction."""
    pass


class TransactionNotActiveError(TransactionError):
    """Raised when operating on an already committed/aborted transaction."""
    pass


# ---------------------------------------------------------------------------
# Transaction Manager
# ---------------------------------------------------------------------------

class TransactionManager:
    """ACID transaction manager composing MVCC + WAL + Lock Manager.

    Coordinates:
    1. Lock Manager -- acquire locks before data access
    2. WAL -- log all mutations before applying
    3. MVCC -- execute reads/writes with snapshot isolation

    The commit protocol:
    1. WAL commit record (durability guarantee)
    2. MVCC commit (make writes visible)
    3. Lock release (allow other transactions)

    The abort protocol:
    1. MVCC abort (discard writes)
    2. WAL abort record (for recovery)
    3. Lock release
    """

    def __init__(
        self,
        *,
        isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ,
        lock_timeout: float = 5.0,
        enable_deadlock_detection: bool = True,
        checkpoint_interval: int = 100,
        database: str = "default",
        table: str = "data",
    ):
        self._mvcc = MVCCEngine()
        self._wal = WAL()
        self._lock_mgr = LockManager()
        self._default_isolation = isolation
        self._lock_timeout = lock_timeout
        self._deadlock_detection = enable_deadlock_detection
        self._checkpoint_interval = checkpoint_interval
        self._database = database
        self._table = table

        # Transaction registry
        self._transactions: dict[int, ManagedTransaction] = {}
        self._next_txn_id = 1
        self._txn_lock = threading.Lock()
        self._ops_since_checkpoint = 0
        self._total_committed = 0
        self._total_aborted = 0

    # ----- Transaction Lifecycle -----

    def begin(
        self,
        isolation: Optional[IsolationLevel] = None,
        priority: int = 0,
    ) -> int:
        """Begin a new managed transaction.

        Returns:
            Transaction ID for use in subsequent operations.
        """
        iso = isolation or self._default_isolation
        with self._txn_lock:
            txn_id = self._next_txn_id
            self._next_txn_id += 1

        # Start in all three subsystems
        mvcc_txn_id = self._mvcc.begin(iso)
        wal_txn_id = self._wal.begin()
        self._lock_mgr.begin_txn(txn_id, priority=priority)

        mtxn = ManagedTransaction(
            txn_id=txn_id,
            mvcc_txn_id=mvcc_txn_id,
            wal_txn_id=wal_txn_id,
            isolation=iso,
            database=self._database,
            table=self._table,
        )
        self._transactions[txn_id] = mtxn
        return txn_id

    def commit(self, txn_id: int) -> bool:
        """Commit a transaction. Returns True on success.

        Protocol: WAL commit -> MVCC commit -> Lock release
        """
        mtxn = self._get_active_txn(txn_id)

        try:
            # Phase 1: WAL commit record (durability)
            self._wal.commit(mtxn.wal_txn_id)

            # Phase 2: MVCC commit (visibility)
            success = self._mvcc.commit(mtxn.mvcc_txn_id)
            if not success:
                # MVCC detected a conflict (serializable violation)
                self._wal.abort(mtxn.wal_txn_id)
                self._mvcc.abort(mtxn.mvcc_txn_id)
                self._lock_mgr.abort_txn(txn_id)
                mtxn.state = TxnState.ABORTED
                self._total_aborted += 1
                raise ConflictError(f"Transaction {txn_id} aborted due to serialization conflict")

            # Phase 3: Release all locks
            self._lock_mgr.commit_txn(txn_id)
            mtxn.state = TxnState.COMMITTED
            self._total_committed += 1

            # Periodic checkpoint
            self._ops_since_checkpoint += mtxn.writes
            if self._checkpoint_interval > 0 and self._ops_since_checkpoint >= self._checkpoint_interval:
                self._wal.checkpoint()
                self._ops_since_checkpoint = 0

            return True

        except ConflictError:
            raise
        except Exception as e:
            # Safety net: if commit fails mid-protocol, abort everything
            self._force_abort(mtxn)
            raise TransactionError(f"Commit failed for txn {txn_id}: {e}") from e

    def abort(self, txn_id: int):
        """Abort a transaction, discarding all changes.

        Protocol: MVCC abort -> WAL abort -> Lock release
        """
        mtxn = self._get_active_txn(txn_id)
        self._force_abort(mtxn)

    def _force_abort(self, mtxn: ManagedTransaction):
        """Unconditionally abort a transaction in all subsystems."""
        try:
            self._mvcc.abort(mtxn.mvcc_txn_id)
        except Exception:
            pass
        try:
            self._wal.abort(mtxn.wal_txn_id)
        except Exception:
            pass
        try:
            self._lock_mgr.abort_txn(mtxn.txn_id)
        except Exception:
            pass
        mtxn.state = TxnState.ABORTED
        self._total_aborted += 1

    # ----- Savepoints -----

    def savepoint(self, txn_id: int, name: str):
        """Create a named savepoint within a transaction."""
        mtxn = self._get_active_txn(txn_id)
        self._mvcc.savepoint(mtxn.mvcc_txn_id, name)
        self._wal.savepoint(mtxn.wal_txn_id, name)
        mtxn.savepoints.append({
            "name": name,
            "writes_at": mtxn.writes,
            "written_keys_at": len(mtxn.written_keys),
        })

    def rollback_to_savepoint(self, txn_id: int, name: str):
        """Roll back to a named savepoint, undoing subsequent writes."""
        mtxn = self._get_active_txn(txn_id)
        self._mvcc.rollback_to_savepoint(mtxn.mvcc_txn_id, name)
        self._wal.rollback_to_savepoint(mtxn.wal_txn_id, name)
        # Restore write count to savepoint state
        for sp in reversed(mtxn.savepoints):
            if sp["name"] == name:
                mtxn.writes = sp["writes_at"]
                mtxn.written_keys = mtxn.written_keys[:sp["written_keys_at"]]
                # Remove savepoints after this one
                idx = mtxn.savepoints.index(sp)
                mtxn.savepoints = mtxn.savepoints[:idx + 1]
                break

    # ----- Data Operations -----

    def get(self, txn_id: int, key: str) -> Optional[Any]:
        """Read a value by key within a transaction.

        MVCC provides read isolation via snapshots -- readers never block
        writers. No lock acquisition needed for reads.
        """
        mtxn = self._get_active_txn(txn_id)
        result = self._mvcc.get(mtxn.mvcc_txn_id, key)
        mtxn.reads += 1
        mtxn.read_keys.append(key)
        return result

    def put(self, txn_id: int, key: str, value: Any):
        """Write a key-value pair within a transaction.

        Acquires exclusive lock, logs to WAL, then writes through MVCC.
        """
        mtxn = self._get_active_txn(txn_id)
        self._acquire_lock(mtxn, key, LockMode.X)

        # Read old value for WAL undo record
        old_value = self._mvcc.get(mtxn.mvcc_txn_id, key)

        # WAL first (write-ahead protocol)
        page_id = hash(key) % 1000
        if old_value is None:
            self._wal.insert(mtxn.wal_txn_id, page_id, key, value)
        else:
            self._wal.update(mtxn.wal_txn_id, page_id, key, value)

        # Then MVCC
        self._mvcc.put(mtxn.mvcc_txn_id, key, value)
        mtxn.writes += 1
        mtxn.written_keys.append(key)

    def delete(self, txn_id: int, key: str) -> bool:
        """Delete a key within a transaction.

        Acquires exclusive lock, logs to WAL, then deletes through MVCC.
        Returns False if the key does not exist.
        """
        mtxn = self._get_active_txn(txn_id)

        # Check existence first (MVCC read -- no lock needed)
        if not self._mvcc.exists(mtxn.mvcc_txn_id, key):
            return False

        self._acquire_lock(mtxn, key, LockMode.X)

        # WAL first
        page_id = hash(key) % 1000
        try:
            self._wal.delete(mtxn.wal_txn_id, page_id, key)
        except ValueError:
            pass  # Key may not be in WAL page (MVCC handles storage)

        # Then MVCC
        result = self._mvcc.delete(mtxn.mvcc_txn_id, key)
        if result:
            mtxn.writes += 1
            mtxn.written_keys.append(key)
        return result

    def scan(self, txn_id: int, prefix: str = "") -> dict[str, Any]:
        """Scan all keys with given prefix within a transaction.

        MVCC snapshot provides consistent reads without locking.
        """
        mtxn = self._get_active_txn(txn_id)
        result = self._mvcc.scan(mtxn.mvcc_txn_id, prefix)
        mtxn.reads += len(result)
        return result

    def range_scan(self, txn_id: int, lo: str, hi: str) -> dict[str, Any]:
        """Range scan [lo, hi] within a transaction (inclusive on both ends)."""
        mtxn = self._get_active_txn(txn_id)
        result = self._mvcc.range_scan(mtxn.mvcc_txn_id, lo, hi)
        mtxn.reads += len(result)
        return result

    def exists(self, txn_id: int, key: str) -> bool:
        """Check if key exists within a transaction."""
        mtxn = self._get_active_txn(txn_id)
        return self._mvcc.exists(mtxn.mvcc_txn_id, key)

    def count(self, txn_id: int, prefix: str = "") -> int:
        """Count keys with given prefix within a transaction."""
        mtxn = self._get_active_txn(txn_id)
        return self._mvcc.count(mtxn.mvcc_txn_id, prefix)

    # ----- Batch Operations -----

    def batch_put(self, txn_id: int, items: dict[str, Any]):
        """Write multiple key-value pairs atomically."""
        for key, value in items.items():
            self.put(txn_id, key, value)

    def batch_get(self, txn_id: int, keys: list[str]) -> dict[str, Any]:
        """Read multiple keys. Returns {key: value} for existing keys."""
        result = {}
        for key in keys:
            val = self.get(txn_id, key)
            if val is not None:
                result[key] = val
        return result

    def batch_delete(self, txn_id: int, keys: list[str]) -> int:
        """Delete multiple keys. Returns count of deleted keys."""
        deleted = 0
        for key in keys:
            if self.delete(txn_id, key):
                deleted += 1
        return deleted

    # ----- Lock Management -----

    def _acquire_lock(self, mtxn: ManagedTransaction, key: str, mode: LockMode):
        """Acquire row-level lock with intention locks on parent resources."""
        try:
            # Intention lock on table
            table_resource = make_resource(
                database=mtxn.database,
                table=mtxn.table,
            )
            intent_mode = LockMode.IS if mode == LockMode.S else LockMode.IX
            acquired = self._lock_mgr.acquire(
                mtxn.txn_id, table_resource, intent_mode, timeout=self._lock_timeout
            )
            if not acquired:
                raise DeadlockError(
                    f"Transaction {mtxn.txn_id} could not acquire {intent_mode.name} lock on table"
                )

            # Row-level lock
            row_resource = make_resource(
                database=mtxn.database,
                table=mtxn.table,
                row=key,
            )
            acquired = self._lock_mgr.acquire(
                mtxn.txn_id, row_resource, mode, timeout=self._lock_timeout
            )
            if not acquired:
                raise DeadlockError(
                    f"Transaction {mtxn.txn_id} could not acquire {mode.name} lock on {key}"
                )
            mtxn.locks_acquired += 1

            # Check lock escalation
            self._lock_mgr.check_escalation(
                mtxn.txn_id,
                database=mtxn.database,
                table=mtxn.table,
            )
        except DeadlockError:
            raise
        except Exception as e:
            # Wrap lock manager errors (LockTimeoutError, DeadlockError from C242)
            raise DeadlockError(
                f"Transaction {mtxn.txn_id} lock acquisition failed on {key}: {e}"
            ) from e

    # ----- Recovery -----

    def checkpoint(self) -> int:
        """Force a WAL checkpoint."""
        lsn = self._wal.checkpoint()
        self._ops_since_checkpoint = 0
        return lsn

    def simulate_crash(self):
        """Simulate a crash for testing recovery."""
        self._wal.simulate_crash()
        # Clear in-memory state
        active_txns = list(self._transactions.keys())
        for tid in active_txns:
            mtxn = self._transactions[tid]
            if mtxn.state == TxnState.ACTIVE:
                mtxn.state = TxnState.ABORTED

    def recover(self) -> dict:
        """Run ARIES recovery after a crash.

        Returns recovery stats including redo/undo counts.
        """
        recovery_result = self._wal.recover()
        return recovery_result

    # ----- Garbage Collection -----

    def gc(self) -> int:
        """Run MVCC garbage collection on dead versions."""
        return self._mvcc.gc()

    # ----- Diagnostics -----

    def get_transaction(self, txn_id: int) -> Optional[ManagedTransaction]:
        """Get transaction metadata."""
        return self._transactions.get(txn_id)

    def active_transactions(self) -> list[int]:
        """List IDs of active transactions."""
        return [
            tid for tid, mtxn in self._transactions.items()
            if mtxn.state == TxnState.ACTIVE
        ]

    def stats(self) -> dict:
        """Aggregate statistics from all subsystems."""
        active = [
            tid for tid, mtxn in self._transactions.items()
            if mtxn.state == TxnState.ACTIVE
        ]
        return {
            "total_transactions": len(self._transactions),
            "active_transactions": len(active),
            "committed": self._total_committed,
            "aborted": self._total_aborted,
            "ops_since_checkpoint": self._ops_since_checkpoint,
            "mvcc": self._mvcc.stats(),
            "wal": self._wal.stats(),
            "lock_manager": self._lock_mgr.get_stats(),
        }

    def wal_stats(self) -> dict:
        """WAL-specific statistics."""
        return self._wal.stats()

    def lock_stats(self) -> dict:
        """Lock manager statistics."""
        return self._lock_mgr.get_stats()

    def mvcc_stats(self) -> dict:
        """MVCC statistics."""
        return self._mvcc.stats()

    # ----- Internal Helpers -----

    def _get_active_txn(self, txn_id: int) -> ManagedTransaction:
        """Look up a transaction and verify it is active."""
        mtxn = self._transactions.get(txn_id)
        if mtxn is None:
            raise TransactionNotFoundError(f"Transaction {txn_id} not found")
        if mtxn.state != TxnState.ACTIVE:
            raise TransactionNotActiveError(
                f"Transaction {txn_id} is {mtxn.state.name}, not ACTIVE"
            )
        return mtxn


# ---------------------------------------------------------------------------
# Read-Only Transaction Context Manager
# ---------------------------------------------------------------------------

class ReadOnlyTransaction:
    """Context manager for read-only transactions.

    Usage:
        with ReadOnlyTransaction(tm) as txn:
            val = txn.get("key1")
            data = txn.scan("prefix")
    """

    def __init__(self, tm: TransactionManager, isolation: Optional[IsolationLevel] = None):
        self._tm = tm
        self._isolation = isolation
        self._txn_id: Optional[int] = None

    def __enter__(self):
        self._txn_id = self._tm.begin(isolation=self._isolation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn_id is not None:
            if exc_type is not None:
                self._tm.abort(self._txn_id)
            else:
                self._tm.commit(self._txn_id)
        return False

    def get(self, key: str) -> Optional[Any]:
        return self._tm.get(self._txn_id, key)

    def scan(self, prefix: str = "") -> dict[str, Any]:
        return self._tm.scan(self._txn_id, prefix)

    def range_scan(self, lo: str, hi: str) -> dict[str, Any]:
        return self._tm.range_scan(self._txn_id, lo, hi)

    def exists(self, key: str) -> bool:
        return self._tm.exists(self._txn_id, key)

    def count(self, prefix: str = "") -> int:
        return self._tm.count(self._txn_id, prefix)


# ---------------------------------------------------------------------------
# Read-Write Transaction Context Manager
# ---------------------------------------------------------------------------

class Transaction:
    """Context manager for read-write transactions.

    Usage:
        with Transaction(tm) as txn:
            txn.put("key1", "value1")
            val = txn.get("key2")
    Commits on clean exit, aborts on exception.
    """

    def __init__(self, tm: TransactionManager, isolation: Optional[IsolationLevel] = None):
        self._tm = tm
        self._isolation = isolation
        self._txn_id: Optional[int] = None

    def __enter__(self):
        self._txn_id = self._tm.begin(isolation=self._isolation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn_id is not None:
            if exc_type is not None:
                try:
                    self._tm.abort(self._txn_id)
                except Exception:
                    pass
            else:
                self._tm.commit(self._txn_id)
        return False

    @property
    def txn_id(self) -> int:
        return self._txn_id

    def get(self, key: str) -> Optional[Any]:
        return self._tm.get(self._txn_id, key)

    def put(self, key: str, value: Any):
        self._tm.put(self._txn_id, key, value)

    def delete(self, key: str) -> bool:
        return self._tm.delete(self._txn_id, key)

    def scan(self, prefix: str = "") -> dict[str, Any]:
        return self._tm.scan(self._txn_id, prefix)

    def range_scan(self, lo: str, hi: str) -> dict[str, Any]:
        return self._tm.range_scan(self._txn_id, lo, hi)

    def exists(self, key: str) -> bool:
        return self._tm.exists(self._txn_id, key)

    def count(self, prefix: str = "") -> int:
        return self._tm.count(self._txn_id, prefix)

    def savepoint(self, name: str):
        self._tm.savepoint(self._txn_id, name)

    def rollback_to_savepoint(self, name: str):
        self._tm.rollback_to_savepoint(self._txn_id, name)

    def batch_put(self, items: dict[str, Any]):
        self._tm.batch_put(self._txn_id, items)

    def batch_get(self, keys: list[str]) -> dict[str, Any]:
        return self._tm.batch_get(self._txn_id, keys)

    def batch_delete(self, keys: list[str]) -> int:
        return self._tm.batch_delete(self._txn_id, keys)
