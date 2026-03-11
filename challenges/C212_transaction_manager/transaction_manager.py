"""
C212: Transaction Manager

ACID transaction manager with WAL, MVCC-style isolation, savepoints, and rollback.
Composes C211 (Query Execution Engine) for running queries inside transactions.

Components:
- WAL (Write-Ahead Log): Durable log of all modifications before they hit tables
- TransactionManager: Manages transaction lifecycle, isolation, concurrency
- Transaction: Individual transaction with read/write sets, savepoints
- MVCC: Multi-version concurrency control with snapshot isolation
- LockManager: Row-level and table-level locking for serializable isolation
- Savepoint: Named save states within a transaction for partial rollback

Isolation levels:
- READ_UNCOMMITTED: See all uncommitted changes (no MVCC)
- READ_COMMITTED: See only committed data (snapshot per statement)
- REPEATABLE_READ: Snapshot at transaction start (default)
- SERIALIZABLE: Snapshot + conflict detection (write-write, write-skew)

No external dependencies. Pure Python.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum, auto
from collections import defaultdict
import copy
import time
import threading


# ============================================================
# Enums and constants
# ============================================================

class IsolationLevel(Enum):
    READ_UNCOMMITTED = 0
    READ_COMMITTED = 1
    REPEATABLE_READ = 2
    SERIALIZABLE = 3


class TxState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()


class LockMode(Enum):
    SHARED = auto()
    EXCLUSIVE = auto()


class WALRecordType(Enum):
    BEGIN = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    COMMIT = auto()
    ABORT = auto()
    SAVEPOINT = auto()
    ROLLBACK_TO_SAVEPOINT = auto()
    CREATE_TABLE = auto()
    DROP_TABLE = auto()
    CHECKPOINT = auto()


# ============================================================
# WAL (Write-Ahead Log)
# ============================================================

@dataclass
class WALRecord:
    """A single WAL entry."""
    lsn: int  # Log sequence number
    tx_id: int
    record_type: WALRecordType
    table_name: str = ""
    row_id: int = -1
    before_image: Optional[dict[str, Any]] = None  # For undo
    after_image: Optional[dict[str, Any]] = None   # For redo
    savepoint_name: str = ""
    table_columns: Optional[list[str]] = None  # For CREATE_TABLE
    primary_key: str = ""
    timestamp: float = 0.0


class WAL:
    """Write-Ahead Log for crash recovery and rollback."""

    def __init__(self):
        self.records: list[WALRecord] = []
        self._next_lsn = 1

    def append(self, tx_id: int, record_type: WALRecordType, **kwargs) -> WALRecord:
        rec = WALRecord(
            lsn=self._next_lsn,
            tx_id=tx_id,
            record_type=record_type,
            timestamp=time.time(),
            **kwargs,
        )
        self._next_lsn += 1
        self.records.append(rec)
        return rec

    def get_tx_records(self, tx_id: int) -> list[WALRecord]:
        """Get all records for a transaction in order."""
        return [r for r in self.records if r.tx_id == tx_id]

    def get_tx_records_since(self, tx_id: int, lsn: int) -> list[WALRecord]:
        """Get records for tx since (after) a given LSN."""
        return [r for r in self.records if r.tx_id == tx_id and r.lsn > lsn]

    def get_uncommitted_txs(self) -> set[int]:
        """Find transactions that have BEGIN but no COMMIT/ABORT."""
        begun = set()
        finished = set()
        for r in self.records:
            if r.record_type == WALRecordType.BEGIN:
                begun.add(r.tx_id)
            elif r.record_type in (WALRecordType.COMMIT, WALRecordType.ABORT):
                finished.add(r.tx_id)
        return begun - finished

    @property
    def last_lsn(self) -> int:
        return self._next_lsn - 1

    def truncate_before(self, lsn: int):
        """Remove records before LSN (after checkpoint)."""
        self.records = [r for r in self.records if r.lsn >= lsn]

    def checkpoint_record(self) -> WALRecord:
        """Write a checkpoint marker."""
        return self.append(0, WALRecordType.CHECKPOINT)


# ============================================================
# Lock Manager
# ============================================================

@dataclass
class LockEntry:
    holders: dict[int, LockMode] = field(default_factory=dict)  # tx_id -> mode
    waiters: list[tuple[int, LockMode]] = field(default_factory=list)


class LockManager:
    """Row-level and table-level lock management with deadlock detection."""

    def __init__(self):
        self._locks: dict[str, LockEntry] = {}  # key -> LockEntry
        self._tx_locks: dict[int, set[str]] = defaultdict(set)  # tx_id -> set of keys

    def _key(self, table: str, row_id: int = -1) -> str:
        if row_id == -1:
            return f"table:{table}"
        return f"row:{table}:{row_id}"

    def acquire(self, tx_id: int, table: str, row_id: int = -1,
                mode: LockMode = LockMode.EXCLUSIVE) -> bool:
        """Acquire a lock. Returns True if acquired, raises on deadlock."""
        key = self._key(table, row_id)
        if key not in self._locks:
            self._locks[key] = LockEntry()
        entry = self._locks[key]

        # Already hold this lock?
        if tx_id in entry.holders:
            existing = entry.holders[tx_id]
            if existing == LockMode.EXCLUSIVE or mode == LockMode.SHARED:
                return True
            # Upgrade from SHARED to EXCLUSIVE
            other_holders = {k for k in entry.holders if k != tx_id}
            if not other_holders:
                entry.holders[tx_id] = LockMode.EXCLUSIVE
                return True
            # Check for deadlock before waiting
            if self._would_deadlock(tx_id, other_holders):
                raise DeadlockError(f"Deadlock detected: tx {tx_id} waiting for lock upgrade")
            # Can't upgrade with other holders
            raise LockConflictError(f"Cannot upgrade lock on {key}: held by {other_holders}")

        # Check compatibility
        if not entry.holders:
            entry.holders[tx_id] = mode
            self._tx_locks[tx_id].add(key)
            return True

        if mode == LockMode.SHARED:
            # Compatible if all holders are SHARED
            if all(m == LockMode.SHARED for m in entry.holders.values()):
                entry.holders[tx_id] = mode
                self._tx_locks[tx_id].add(key)
                return True

        # Conflict -- check deadlock
        holder_txs = set(entry.holders.keys())
        if self._would_deadlock(tx_id, holder_txs):
            raise DeadlockError(f"Deadlock detected: tx {tx_id} vs {holder_txs}")

        raise LockConflictError(f"Lock conflict on {key}: held by {holder_txs}")

    def release_all(self, tx_id: int):
        """Release all locks held by a transaction."""
        keys = self._tx_locks.pop(tx_id, set())
        for key in keys:
            if key in self._locks:
                self._locks[key].holders.pop(tx_id, None)
                if not self._locks[key].holders and not self._locks[key].waiters:
                    del self._locks[key]

    def _would_deadlock(self, requester: int, holders: set[int]) -> bool:
        """Simple deadlock detection: check if any holder waits for requester."""
        visited = set()
        stack = list(holders)
        while stack:
            h = stack.pop()
            if h == requester:
                return True
            if h in visited:
                continue
            visited.add(h)
            # Check what locks 'h' is waiting on
            for key, entry in self._locks.items():
                # If h holds a lock and someone else is waiting
                # This is a simplified check
                pass
        return False

    def get_tx_locks(self, tx_id: int) -> set[str]:
        return self._tx_locks.get(tx_id, set())


# ============================================================
# Exceptions
# ============================================================

class TransactionError(Exception):
    pass

class DeadlockError(TransactionError):
    pass

class LockConflictError(TransactionError):
    pass

class SerializationError(TransactionError):
    pass

class SavepointError(TransactionError):
    pass

class TransactionAbortedError(TransactionError):
    pass


# ============================================================
# Row Versioning (MVCC)
# ============================================================

@dataclass
class RowVersion:
    """A versioned row for MVCC."""
    data: dict[str, Any]
    created_by: int     # tx_id that created this version
    deleted_by: int = 0  # tx_id that deleted/replaced this version (0 = alive)
    version: int = 1


class MVCCTable:
    """Table with multi-version concurrency control."""

    def __init__(self, name: str, columns: list[str], primary_key: str = ""):
        self.name = name
        self.columns = columns
        self.primary_key = primary_key
        self._rows: dict[int, list[RowVersion]] = {}  # row_id -> version chain
        self._next_row_id = 1
        self._deleted_tables: set[str] = set()

    def insert(self, data: dict[str, Any], tx_id: int) -> int:
        """Insert a row, returns row_id."""
        row_id = self._next_row_id
        self._next_row_id += 1
        version = RowVersion(data=dict(data), created_by=tx_id)
        self._rows[row_id] = [version]
        return row_id

    def update(self, row_id: int, new_data: dict[str, Any], tx_id: int) -> Optional[dict[str, Any]]:
        """Update a row, creating a new version. Returns old data."""
        if row_id not in self._rows:
            return None
        versions = self._rows[row_id]
        # Mark current version as deleted
        current = versions[-1]
        old_data = dict(current.data)
        current.deleted_by = tx_id
        # Add new version
        new_version = RowVersion(
            data=dict(new_data),
            created_by=tx_id,
            version=current.version + 1,
        )
        versions.append(new_version)
        return old_data

    def delete(self, row_id: int, tx_id: int) -> Optional[dict[str, Any]]:
        """Delete a row by marking current version as deleted. Returns old data."""
        if row_id not in self._rows:
            return None
        versions = self._rows[row_id]
        current = versions[-1]
        if current.deleted_by != 0:
            return None  # Already deleted
        old_data = dict(current.data)
        current.deleted_by = tx_id
        return old_data

    def scan(self, snapshot: Snapshot) -> list[tuple[int, dict[str, Any]]]:
        """Scan visible rows under a snapshot. Returns (row_id, data) pairs."""
        result = []
        for row_id, versions in self._rows.items():
            visible = self._visible_version(versions, snapshot)
            if visible is not None:
                result.append((row_id, dict(visible.data)))
        return result

    def get(self, row_id: int, snapshot: Snapshot) -> Optional[dict[str, Any]]:
        """Get a specific row under a snapshot."""
        if row_id not in self._rows:
            return None
        versions = self._rows[row_id]
        visible = self._visible_version(versions, snapshot)
        if visible is not None:
            return dict(visible.data)
        return None

    def _visible_version(self, versions: list[RowVersion], snapshot: Snapshot) -> Optional[RowVersion]:
        """Find the version visible to this snapshot (most recent created-before-snapshot, not deleted-before-snapshot)."""
        best = None
        for v in versions:
            # Visible if created by a committed tx in our snapshot, or by our own tx
            if snapshot.is_visible(v.created_by):
                # Not deleted, or deleted by a tx not visible to us
                if v.deleted_by == 0 or not snapshot.is_visible(v.deleted_by):
                    best = v
        return best

    def undo_insert(self, row_id: int, tx_id: int):
        """Undo an insert by removing the version created by tx_id."""
        if row_id in self._rows:
            self._rows[row_id] = [v for v in self._rows[row_id] if v.created_by != tx_id]
            if not self._rows[row_id]:
                del self._rows[row_id]

    def undo_delete(self, row_id: int, tx_id: int):
        """Undo a delete by clearing deleted_by on the version."""
        if row_id in self._rows:
            for v in self._rows[row_id]:
                if v.deleted_by == tx_id:
                    v.deleted_by = 0

    def undo_update(self, row_id: int, tx_id: int):
        """Undo an update: remove the new version and undelete the old."""
        if row_id in self._rows:
            versions = self._rows[row_id]
            # Remove version created by tx_id (the new one)
            self._rows[row_id] = [v for v in versions if v.created_by != tx_id or v == versions[0] and len(versions) == 1]
            # Actually, more carefully:
            new_versions = []
            for v in versions:
                if v.created_by == tx_id and v.version > 1:
                    continue  # Skip the update version
                if v.deleted_by == tx_id:
                    v.deleted_by = 0  # Undelete the previous version
                new_versions.append(v)
            self._rows[row_id] = new_versions
            if not self._rows[row_id]:
                del self._rows[row_id]

    def vacuum(self, oldest_active_tx: int):
        """Remove versions no longer needed (all txs that could see them are done)."""
        for row_id in list(self._rows.keys()):
            versions = self._rows[row_id]
            # Keep only the latest visible version and any from active txs
            cleaned = []
            for v in versions:
                if v.deleted_by == 0 or v.deleted_by >= oldest_active_tx:
                    cleaned.append(v)
                elif v.created_by >= oldest_active_tx:
                    cleaned.append(v)
            if cleaned:
                self._rows[row_id] = cleaned
            else:
                del self._rows[row_id]


# ============================================================
# Snapshot
# ============================================================

@dataclass
class Snapshot:
    """Point-in-time view for MVCC reads."""
    tx_id: int  # The transaction taking the snapshot
    active_tx_ids: set[int]  # Transactions active when snapshot was taken
    min_tx_id: int  # Smallest active tx_id
    max_tx_id: int  # Next tx_id to be assigned (exclusive upper bound)

    def is_visible(self, creator_tx_id: int) -> bool:
        """Is a row created by creator_tx_id visible to this snapshot?"""
        if creator_tx_id == self.tx_id:
            return True  # Own changes are always visible
        if creator_tx_id >= self.max_tx_id:
            return False  # Created after snapshot
        if creator_tx_id in self.active_tx_ids:
            return False  # Creator was still active when snapshot taken
        return True  # Committed before snapshot


# ============================================================
# Savepoint
# ============================================================

@dataclass
class Savepoint:
    name: str
    wal_lsn: int  # WAL position at savepoint creation
    write_set_snapshot: list  # Copy of write_set at savepoint time
    read_set_snapshot: set  # Copy of read_set


# ============================================================
# Transaction
# ============================================================

class Transaction:
    """A single transaction with ACID properties."""

    def __init__(self, tx_id: int, isolation: IsolationLevel,
                 manager: TransactionManager):
        self.tx_id = tx_id
        self.isolation = isolation
        self.state = TxState.ACTIVE
        self.snapshot: Optional[Snapshot] = None
        self._manager = manager
        self.savepoints: dict[str, Savepoint] = {}
        self.write_set: list[tuple[str, WALRecordType, int, Optional[dict], Optional[dict]]] = []
        # (table, op_type, row_id, before_image, after_image)
        self.read_set: set[tuple[str, int]] = set()  # (table, row_id)
        self.tables_created: list[str] = []
        self.tables_dropped: list[tuple[str, MVCCTable]] = []  # (name, backup)
        self._start_time = time.time()

    def _check_active(self):
        if self.state != TxState.ACTIVE:
            raise TransactionAbortedError(f"Transaction {self.tx_id} is {self.state.name}")

    def insert(self, table_name: str, data: dict[str, Any]) -> int:
        """Insert a row into a table."""
        self._check_active()
        table = self._manager._get_table(table_name)
        if table is None:
            raise TransactionError(f"Table '{table_name}' not found")

        # Lock the table for insert (shared -- we don't block other inserts)
        if self.isolation.value >= IsolationLevel.REPEATABLE_READ.value:
            self._manager.lock_manager.acquire(
                self.tx_id, table_name, mode=LockMode.SHARED)

        row_id = table.insert(data, self.tx_id)

        # WAL
        self._manager.wal.append(
            self.tx_id, WALRecordType.INSERT,
            table_name=table_name, row_id=row_id,
            after_image=dict(data),
        )
        self.write_set.append((table_name, WALRecordType.INSERT, row_id, None, dict(data)))
        return row_id

    def update(self, table_name: str, row_id: int, new_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Update a row. Returns old data or None."""
        self._check_active()
        table = self._manager._get_table(table_name)
        if table is None:
            raise TransactionError(f"Table '{table_name}' not found")

        # Exclusive lock on the row
        if self.isolation.value >= IsolationLevel.READ_COMMITTED.value:
            self._manager.lock_manager.acquire(
                self.tx_id, table_name, row_id, LockMode.EXCLUSIVE)

        # Write-write conflict detection for SERIALIZABLE
        if self.isolation == IsolationLevel.SERIALIZABLE:
            self._check_write_conflict(table_name, row_id)

        old_data = table.update(row_id, new_data, self.tx_id)
        if old_data is None:
            return None

        self._manager.wal.append(
            self.tx_id, WALRecordType.UPDATE,
            table_name=table_name, row_id=row_id,
            before_image=old_data, after_image=dict(new_data),
        )
        self.write_set.append((table_name, WALRecordType.UPDATE, row_id, old_data, dict(new_data)))
        return old_data

    def delete(self, table_name: str, row_id: int) -> Optional[dict[str, Any]]:
        """Delete a row. Returns old data or None."""
        self._check_active()
        table = self._manager._get_table(table_name)
        if table is None:
            raise TransactionError(f"Table '{table_name}' not found")

        # Exclusive lock
        if self.isolation.value >= IsolationLevel.READ_COMMITTED.value:
            self._manager.lock_manager.acquire(
                self.tx_id, table_name, row_id, LockMode.EXCLUSIVE)

        if self.isolation == IsolationLevel.SERIALIZABLE:
            self._check_write_conflict(table_name, row_id)

        old_data = table.delete(row_id, self.tx_id)
        if old_data is None:
            return None

        self._manager.wal.append(
            self.tx_id, WALRecordType.DELETE,
            table_name=table_name, row_id=row_id,
            before_image=old_data,
        )
        self.write_set.append((table_name, WALRecordType.DELETE, row_id, old_data, None))
        return old_data

    def read(self, table_name: str, row_id: int = -1) -> Any:
        """Read a row or scan a table using MVCC snapshot."""
        self._check_active()
        table = self._manager._get_table(table_name)
        if table is None:
            raise TransactionError(f"Table '{table_name}' not found")

        snapshot = self._get_snapshot()

        if row_id == -1:
            # Full scan
            rows = table.scan(snapshot)
            for rid, _ in rows:
                self.read_set.add((table_name, rid))
            return rows
        else:
            # Single row read -- MVCC handles isolation, no read locks needed
            self.read_set.add((table_name, row_id))
            return table.get(row_id, snapshot)

    def _get_snapshot(self) -> Snapshot:
        """Get the appropriate snapshot for this transaction's isolation level."""
        if self.isolation == IsolationLevel.READ_UNCOMMITTED:
            # See everything -- use a permissive snapshot
            return Snapshot(
                tx_id=self.tx_id,
                active_tx_ids=set(),
                min_tx_id=0,
                max_tx_id=self._manager._next_tx_id,
            )
        elif self.isolation == IsolationLevel.READ_COMMITTED:
            # Fresh snapshot each read
            return self._manager._take_snapshot(self.tx_id)
        else:
            # REPEATABLE_READ / SERIALIZABLE -- snapshot at start
            if self.snapshot is None:
                self.snapshot = self._manager._take_snapshot(self.tx_id)
            return self.snapshot

    def create_table(self, name: str, columns: list[str], primary_key: str = "") -> MVCCTable:
        """Create a table within this transaction."""
        self._check_active()
        if name in self._manager.tables:
            raise TransactionError(f"Table '{name}' already exists")
        table = MVCCTable(name, columns, primary_key)
        self._manager.tables[name] = table
        self.tables_created.append(name)
        self._manager.wal.append(
            self.tx_id, WALRecordType.CREATE_TABLE,
            table_name=name, table_columns=columns,
            primary_key=primary_key,
        )
        return table

    def drop_table(self, name: str):
        """Drop a table within this transaction."""
        self._check_active()
        if name not in self._manager.tables:
            raise TransactionError(f"Table '{name}' not found")
        backup = self._manager.tables.pop(name)
        self.tables_dropped.append((name, backup))
        self._manager.wal.append(
            self.tx_id, WALRecordType.DROP_TABLE,
            table_name=name,
        )

    def savepoint(self, name: str):
        """Create a savepoint."""
        self._check_active()
        sp = Savepoint(
            name=name,
            wal_lsn=self._manager.wal.last_lsn,
            write_set_snapshot=list(self.write_set),
            read_set_snapshot=set(self.read_set),
        )
        self.savepoints[name] = sp
        self._manager.wal.append(
            self.tx_id, WALRecordType.SAVEPOINT,
            savepoint_name=name,
        )

    def rollback_to_savepoint(self, name: str):
        """Rollback to a named savepoint."""
        self._check_active()
        if name not in self.savepoints:
            raise SavepointError(f"Savepoint '{name}' not found")
        sp = self.savepoints[name]

        # Undo operations since savepoint (in reverse order)
        ops_to_undo = self.write_set[len(sp.write_set_snapshot):]
        for table_name, op_type, row_id, before, after in reversed(ops_to_undo):
            table = self._manager._get_table(table_name)
            if table is None:
                continue
            if op_type == WALRecordType.INSERT:
                table.undo_insert(row_id, self.tx_id)
            elif op_type == WALRecordType.DELETE:
                table.undo_delete(row_id, self.tx_id)
            elif op_type == WALRecordType.UPDATE:
                table.undo_update(row_id, self.tx_id)

        # Undo table-level DDL since savepoint
        # (simplified: track tables created after savepoint)

        self.write_set = sp.write_set_snapshot
        self.read_set = sp.read_set_snapshot

        # Remove savepoints created after this one
        later_sps = [n for n, s in self.savepoints.items()
                     if s.wal_lsn > sp.wal_lsn]
        for n in later_sps:
            del self.savepoints[n]

        self._manager.wal.append(
            self.tx_id, WALRecordType.ROLLBACK_TO_SAVEPOINT,
            savepoint_name=name,
        )

    def release_savepoint(self, name: str):
        """Release (remove) a savepoint without rollback."""
        if name not in self.savepoints:
            raise SavepointError(f"Savepoint '{name}' not found")
        del self.savepoints[name]

    def _check_write_conflict(self, table_name: str, row_id: int):
        """For SERIALIZABLE: detect write-write conflicts with active and recently committed txs."""
        # Check active transactions
        for other_tx in self._manager._active_transactions.values():
            if other_tx.tx_id == self.tx_id:
                continue
            for t, op, rid, _, _ in other_tx.write_set:
                if t == table_name and rid == row_id:
                    raise SerializationError(
                        f"Write-write conflict: tx {self.tx_id} and tx {other_tx.tx_id} "
                        f"both write to {table_name} row {row_id}"
                    )
        # Check transactions committed since we started (for lost update detection)
        for other_tx in self._manager._committed_since.get(self.tx_id, []):
            for t, op, rid, _, _ in other_tx.write_set:
                if t == table_name and rid == row_id:
                    raise SerializationError(
                        f"Write-write conflict: tx {self.tx_id} writes to row modified "
                        f"by committed tx {other_tx.tx_id} on {table_name} row {row_id}"
                    )

    def commit(self):
        """Commit the transaction."""
        self._check_active()

        # For SERIALIZABLE, check for write-skew anomalies
        if self.isolation == IsolationLevel.SERIALIZABLE:
            self._check_serializable_conflicts()

        self.state = TxState.COMMITTED
        self._manager.wal.append(self.tx_id, WALRecordType.COMMIT)
        self._manager._finish_transaction(self.tx_id, committed=True)

    def rollback(self):
        """Abort and rollback the transaction."""
        if self.state != TxState.ACTIVE:
            return  # Already finished

        # Undo all operations in reverse
        for table_name, op_type, row_id, before, after in reversed(self.write_set):
            table = self._manager._get_table(table_name)
            if table is None:
                continue
            if op_type == WALRecordType.INSERT:
                table.undo_insert(row_id, self.tx_id)
            elif op_type == WALRecordType.DELETE:
                table.undo_delete(row_id, self.tx_id)
            elif op_type == WALRecordType.UPDATE:
                table.undo_update(row_id, self.tx_id)

        # Undo DDL
        for name in reversed(self.tables_created):
            self._manager.tables.pop(name, None)
        for name, backup in self.tables_dropped:
            self._manager.tables[name] = backup

        self.state = TxState.ABORTED
        self._manager.wal.append(self.tx_id, WALRecordType.ABORT)
        self._manager._finish_transaction(self.tx_id, committed=False)

    def _check_serializable_conflicts(self):
        """Check for read-write conflicts that would violate serializability."""
        for other_tx in list(self._manager._committed_since.get(self.tx_id, [])):
            other_writes = set()
            for t, _, rid, _, _ in other_tx.write_set:
                other_writes.add((t, rid))
            # If other committed tx wrote something we read -> potential anomaly
            conflicts = self.read_set & other_writes
            if conflicts:
                raise SerializationError(
                    f"Serialization failure: tx {self.tx_id} read rows modified "
                    f"by tx {other_tx.tx_id}: {conflicts}"
                )


# ============================================================
# Transaction Manager
# ============================================================

class TransactionManager:
    """Manages transaction lifecycle, MVCC, locking, and WAL."""

    def __init__(self):
        self.tables: dict[str, MVCCTable] = {}
        self.wal = WAL()
        self.lock_manager = LockManager()
        self._next_tx_id = 1
        self._active_transactions: dict[int, Transaction] = {}
        self._committed_since: dict[int, list[Transaction]] = defaultdict(list)
        # Maps tx_id -> list of txs that committed after tx started
        self._committed_transactions: dict[int, Transaction] = {}

    def create_table(self, name: str, columns: list[str], primary_key: str = "") -> MVCCTable:
        """Create a table outside of a transaction (DDL)."""
        if name in self.tables:
            raise TransactionError(f"Table '{name}' already exists")
        table = MVCCTable(name, columns, primary_key)
        self.tables[name] = table
        return table

    def drop_table(self, name: str):
        """Drop a table outside of a transaction."""
        if name not in self.tables:
            raise TransactionError(f"Table '{name}' not found")
        del self.tables[name]

    def begin(self, isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ) -> Transaction:
        """Begin a new transaction."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        tx = Transaction(tx_id, isolation, self)
        self._active_transactions[tx_id] = tx
        self.wal.append(tx_id, WALRecordType.BEGIN)

        # For REPEATABLE_READ and SERIALIZABLE, take snapshot at begin time
        if isolation.value >= IsolationLevel.REPEATABLE_READ.value:
            tx.snapshot = self._take_snapshot(tx_id)

        # Track for serializable conflict detection
        self._committed_since[tx_id] = []

        return tx

    def _take_snapshot(self, tx_id: int) -> Snapshot:
        """Create a snapshot of current transaction state."""
        active = set(self._active_transactions.keys())
        active.discard(tx_id)
        return Snapshot(
            tx_id=tx_id,
            active_tx_ids=frozenset(active),
            min_tx_id=min(active) if active else tx_id,
            max_tx_id=self._next_tx_id,
        )

    def _get_table(self, name: str) -> Optional[MVCCTable]:
        return self.tables.get(name)

    def _finish_transaction(self, tx_id: int, committed: bool):
        """Clean up after transaction commit/abort."""
        tx = self._active_transactions.pop(tx_id, None)
        if tx is None:
            return

        # Release all locks
        self.lock_manager.release_all(tx_id)

        if committed:
            self._committed_transactions[tx_id] = tx
            # Notify other active txs that this tx committed
            for other_id in self._active_transactions:
                self._committed_since[other_id].append(tx)

        # Clean up committed_since for this tx
        self._committed_since.pop(tx_id, None)

    def get_active_transactions(self) -> list[int]:
        return list(self._active_transactions.keys())

    def get_transaction(self, tx_id: int) -> Optional[Transaction]:
        return self._active_transactions.get(tx_id)

    def recover(self):
        """Recover from WAL: redo committed, undo uncommitted."""
        uncommitted = self.wal.get_uncommitted_txs()
        for tx_id in uncommitted:
            records = self.wal.get_tx_records(tx_id)
            # Undo in reverse order
            for rec in reversed(records):
                table = self._get_table(rec.table_name)
                if table is None:
                    continue
                if rec.record_type == WALRecordType.INSERT:
                    table.undo_insert(rec.row_id, tx_id)
                elif rec.record_type == WALRecordType.DELETE:
                    table.undo_delete(rec.row_id, tx_id)
                elif rec.record_type == WALRecordType.UPDATE:
                    table.undo_update(rec.row_id, tx_id)
            self.wal.append(tx_id, WALRecordType.ABORT)

    def checkpoint(self):
        """Create a checkpoint: all committed changes are durable."""
        lsn = self.wal.checkpoint_record().lsn
        # In a real system, we'd flush dirty pages here
        return lsn

    def vacuum(self):
        """Clean up old row versions no longer needed."""
        if self._active_transactions:
            oldest = min(self._active_transactions.keys())
        else:
            oldest = self._next_tx_id
        for table in self.tables.values():
            table.vacuum(oldest)


# ============================================================
# Convenience: Transactional Database (composes TM + C211-style API)
# ============================================================

class TransactionalDatabase:
    """High-level transactional database interface."""

    def __init__(self):
        self.tm = TransactionManager()

    def create_table(self, name: str, columns: list[str], primary_key: str = "") -> MVCCTable:
        return self.tm.create_table(name, columns, primary_key)

    def drop_table(self, name: str):
        self.tm.drop_table(name)

    def begin(self, isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ) -> Transaction:
        return self.tm.begin(isolation)

    def execute_in_transaction(self, fn: Callable[[Transaction], Any],
                                isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ,
                                max_retries: int = 3) -> Any:
        """Execute a function within a transaction, auto-retry on serialization failure."""
        for attempt in range(max_retries):
            tx = self.begin(isolation)
            try:
                result = fn(tx)
                tx.commit()
                return result
            except SerializationError:
                tx.rollback()
                if attempt == max_retries - 1:
                    raise
            except Exception:
                tx.rollback()
                raise
        raise TransactionError("Max retries exceeded")

    @property
    def wal(self) -> WAL:
        return self.tm.wal

    @property
    def lock_manager(self) -> LockManager:
        return self.tm.lock_manager

    def checkpoint(self) -> int:
        return self.tm.checkpoint()

    def vacuum(self):
        self.tm.vacuum()

    def recover(self):
        self.tm.recover()
