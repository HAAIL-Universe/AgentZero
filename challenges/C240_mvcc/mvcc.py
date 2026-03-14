"""
C240: Multi-Version Concurrency Control (MVCC)

A complete MVCC storage engine implementing:
- Multi-version tuple storage with transaction ID tagging
- Snapshot isolation (SI) and serializable snapshot isolation (SSI)
- Read-committed and repeatable-read isolation levels
- Transaction lifecycle: begin, commit, abort, savepoints
- Visibility rules based on transaction snapshots
- Garbage collection of dead versions
- Write-write conflict detection
- Predicate locking for serializable isolation
- Secondary indexes with MVCC awareness
- Range queries with consistent snapshots

Domain: Database Internals
Standalone implementation -- no external dependencies.
"""

from enum import Enum, auto
from typing import Any, Optional
import time
import threading


# ---------------------------------------------------------------------------
# Transaction States
# ---------------------------------------------------------------------------

class TxnStatus(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()


class IsolationLevel(Enum):
    READ_COMMITTED = auto()
    REPEATABLE_READ = auto()
    SERIALIZABLE = auto()


# ---------------------------------------------------------------------------
# Version Chain Entry
# ---------------------------------------------------------------------------

class Version:
    """A single version of a row/tuple."""

    __slots__ = ('key', 'value', 'xmin', 'xmax', 'created_by', 'deleted_by',
                 'prev_version')

    def __init__(self, key: str, value: Any, xmin: int, xmax: int = 0):
        self.key = key
        self.value = value
        self.xmin = xmin       # Transaction that created this version
        self.xmax = xmax       # Transaction that deleted/replaced this version (0 = alive)
        self.created_by = xmin
        self.deleted_by = xmax
        self.prev_version: Optional['Version'] = None

    def is_deleted(self) -> bool:
        return self.xmax != 0

    def __repr__(self):
        return f"Version(key={self.key!r}, val={self.value!r}, xmin={self.xmin}, xmax={self.xmax})"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot:
    """Point-in-time snapshot for a transaction."""

    __slots__ = ('xmin', 'xmax', 'active_txns')

    def __init__(self, xmin: int, xmax: int, active_txns: frozenset):
        self.xmin = xmin          # Lowest active txn ID at snapshot time
        self.xmax = xmax          # Next txn ID that will be assigned
        self.active_txns = active_txns  # Set of txn IDs active at snapshot time

    def is_visible(self, xid: int) -> bool:
        """Check if a transaction's effects are visible in this snapshot."""
        if xid >= self.xmax:
            return False
        if xid in self.active_txns:
            return False
        return True


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction:
    """Represents a database transaction."""

    def __init__(self, txn_id: int, isolation: IsolationLevel, snapshot: Snapshot):
        self.txn_id = txn_id
        self.isolation = isolation
        self.status = TxnStatus.ACTIVE
        self.snapshot = snapshot
        self.write_set: dict[str, Version] = {}  # key -> version we wrote
        self.read_set: dict[str, int] = {}        # key -> xmin of version we read
        self.savepoints: list[dict] = []           # stack of savepoint states
        self.start_time = time.monotonic()
        self.locks: set[str] = set()               # keys we hold write locks on

    def is_active(self) -> bool:
        return self.status == TxnStatus.ACTIVE


# ---------------------------------------------------------------------------
# Savepoint
# ---------------------------------------------------------------------------

class Savepoint:
    """Captures transaction state at a point for partial rollback."""

    def __init__(self, name: str, write_set: dict, read_set: dict, locks: set):
        self.name = name
        self.write_set = dict(write_set)
        self.read_set = dict(read_set)
        self.locks = set(locks)
        # Capture xmax of each version at savepoint time (for detecting post-SP deletes)
        self.version_xmax: dict[str, int] = {k: v.xmax for k, v in write_set.items()}


# ---------------------------------------------------------------------------
# Write Conflict Error
# ---------------------------------------------------------------------------

class WriteConflictError(Exception):
    """Raised when a write-write conflict is detected."""
    pass


class SerializationError(Exception):
    """Raised when serializable isolation detects a conflict."""
    pass


class DeadlockError(Exception):
    """Raised when a deadlock is detected."""
    pass


# ---------------------------------------------------------------------------
# MVCC Index
# ---------------------------------------------------------------------------

class MVCCIndex:
    """Secondary index with MVCC awareness."""

    def __init__(self, name: str, key_func):
        self.name = name
        self.key_func = key_func
        # index_key -> set of primary keys
        self.entries: dict[Any, set] = {}

    def add(self, primary_key: str, value: Any):
        index_key = self.key_func(value)
        if index_key not in self.entries:
            self.entries[index_key] = set()
        self.entries[index_key].add(primary_key)

    def remove(self, primary_key: str, value: Any):
        index_key = self.key_func(value)
        if index_key in self.entries:
            self.entries[index_key].discard(primary_key)
            if not self.entries[index_key]:
                del self.entries[index_key]

    def lookup(self, index_key: Any) -> set:
        return set(self.entries.get(index_key, set()))

    def range_lookup(self, lo: Any, hi: Any) -> set:
        """Return primary keys where lo <= index_key <= hi."""
        result = set()
        for k, pkeys in self.entries.items():
            if lo <= k <= hi:
                result.update(pkeys)
        return result


# ---------------------------------------------------------------------------
# MVCC Storage Engine
# ---------------------------------------------------------------------------

class MVCCEngine:
    """
    Multi-Version Concurrency Control storage engine.

    Supports multiple isolation levels:
    - READ_COMMITTED: each statement sees latest committed data
    - REPEATABLE_READ: transaction sees snapshot from begin time
    - SERIALIZABLE: repeatable read + conflict detection for write skew
    """

    def __init__(self):
        # Version storage: key -> latest Version (head of chain)
        self._versions: dict[str, Version] = {}
        # Transaction management
        self._next_txn_id = 1
        self._active_txns: dict[int, Transaction] = {}
        self._committed_txns: dict[int, Transaction] = {}
        self._aborted_txns: set[int] = set()
        # Write locks: key -> txn_id holding the lock
        self._write_locks: dict[str, int] = {}
        # Indexes
        self._indexes: dict[str, MVCCIndex] = {}
        # GC tracking
        self._oldest_active_xid: int = 0
        # Lock for thread safety
        self._lock = threading.Lock()
        # SSI: track read/write dependencies for serializable
        self._rw_deps: dict[int, set] = {}  # txn_id -> set of txn_ids it conflicts with

    # -------------------------------------------------------------------
    # Transaction Lifecycle
    # -------------------------------------------------------------------

    def begin(self, isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ) -> int:
        """Begin a new transaction. Returns transaction ID."""
        with self._lock:
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            snapshot = self._take_snapshot()
            txn = Transaction(txn_id, isolation, snapshot)
            self._active_txns[txn_id] = txn
            return txn_id

    def commit(self, txn_id: int) -> bool:
        """Commit a transaction. Returns True on success."""
        with self._lock:
            txn = self._get_active_txn(txn_id)

            # SSI validation for serializable
            if txn.isolation == IsolationLevel.SERIALIZABLE:
                self._ssi_validate(txn)

            # Mark all our written versions as committed
            txn.status = TxnStatus.COMMITTED
            self._committed_txns[txn_id] = txn

            # Release write locks
            for key in txn.locks:
                if self._write_locks.get(key) == txn_id:
                    del self._write_locks[key]

            del self._active_txns[txn_id]
            self._update_oldest_active()
            return True

    def abort(self, txn_id: int):
        """Abort a transaction, rolling back all changes."""
        with self._lock:
            txn = self._get_active_txn(txn_id)
            self._do_abort(txn)

    def _do_abort(self, txn: Transaction):
        """Internal abort logic (caller holds lock)."""
        txn.status = TxnStatus.ABORTED
        self._aborted_txns.add(txn.txn_id)

        # Undo written versions
        for key, version in txn.write_set.items():
            head = self._versions.get(key)
            if head is version:
                # Remove our version from the chain
                if version.prev_version is not None:
                    # Restore previous version (undelete if we replaced)
                    prev = version.prev_version
                    if prev.xmax == txn.txn_id:
                        prev.xmax = 0
                        prev.deleted_by = 0
                    self._versions[key] = prev
                else:
                    # We created this key fresh
                    del self._versions[key]
            else:
                # Our version is somewhere in the chain but not head
                # This shouldn't happen with proper locking, but handle it
                self._unlink_version(key, version)

            # Update indexes
            for idx in self._indexes.values():
                idx.remove(key, version.value)

        # Release write locks
        for key in txn.locks:
            if self._write_locks.get(key) == txn.txn_id:
                del self._write_locks[key]

        if txn.txn_id in self._active_txns:
            del self._active_txns[txn.txn_id]
        self._update_oldest_active()

    def _unlink_version(self, key: str, target: Version):
        """Remove a version from the middle of a chain."""
        head = self._versions.get(key)
        if head is None:
            return
        if head is target:
            if target.prev_version:
                self._versions[key] = target.prev_version
            else:
                del self._versions[key]
            return
        curr = head
        while curr.prev_version is not None:
            if curr.prev_version is target:
                curr.prev_version = target.prev_version
                return
            curr = curr.prev_version

    # -------------------------------------------------------------------
    # Savepoints
    # -------------------------------------------------------------------

    def savepoint(self, txn_id: int, name: str):
        """Create a savepoint within a transaction."""
        with self._lock:
            txn = self._get_active_txn(txn_id)
            sp = Savepoint(name, txn.write_set, txn.read_set, txn.locks)
            txn.savepoints.append(sp)

    def rollback_to_savepoint(self, txn_id: int, name: str):
        """Roll back to a named savepoint."""
        with self._lock:
            txn = self._get_active_txn(txn_id)

            # Find the savepoint
            sp_idx = None
            for i, sp in enumerate(txn.savepoints):
                if sp.name == name:
                    sp_idx = i
                    break
            if sp_idx is None:
                raise ValueError(f"Savepoint '{name}' not found")

            sp = txn.savepoints[sp_idx]

            # Collect all keys that need rollback:
            # 1. Keys written after savepoint that weren't in savepoint write_set
            # 2. Keys in savepoint write_set that were overwritten after savepoint
            all_current_keys = set(txn.write_set.keys())
            all_sp_keys = set(sp.write_set.keys())

            # Keys that are new after savepoint (not in sp.write_set)
            new_keys = all_current_keys - all_sp_keys

            # Keys that existed in both but were overwritten (new version object)
            overwritten_keys = {
                k for k in (all_current_keys & all_sp_keys)
                if txn.write_set[k] is not sp.write_set[k]
            }

            # Keys that are same version but were deleted after savepoint
            # (xmax was changed on same version object since savepoint)
            deleted_after_sp = {
                k for k in (all_current_keys & all_sp_keys)
                if (txn.write_set[k] is sp.write_set[k] and
                    txn.write_set[k].xmax != sp.version_xmax.get(k, 0))
            }

            # Undo new keys: remove our version, restore prev
            for key in new_keys:
                version = txn.write_set[key]
                head = self._versions.get(key)
                if head is version:
                    if version.prev_version is not None:
                        prev = version.prev_version
                        if prev.xmax == txn.txn_id:
                            prev.xmax = 0
                            prev.deleted_by = 0
                        self._versions[key] = prev
                    else:
                        del self._versions[key]

                # Update indexes
                for idx in self._indexes.values():
                    idx.remove(key, version.value)

            # Undo overwritten keys: restore to savepoint version
            for key in overwritten_keys:
                current_ver = txn.write_set[key]
                sp_ver = sp.write_set[key]
                # The chain is: current_ver -> ... -> sp_ver -> ...
                # We need to make sp_ver the head again
                self._versions[key] = sp_ver
                # Ensure sp_ver's xmax is cleared (it's alive again from our perspective)
                sp_ver.xmax = 0
                sp_ver.deleted_by = 0

                # Update indexes
                for idx in self._indexes.values():
                    idx.remove(key, current_ver.value)
                    idx.add(key, sp_ver.value)

            # Undo deletions made after savepoint (same version, xmax changed)
            for key in deleted_after_sp:
                ver = txn.write_set[key]
                old_xmax = sp.version_xmax.get(key, 0)
                ver.xmax = old_xmax
                ver.deleted_by = old_xmax
                # Re-add to indexes if it was alive at savepoint
                if old_xmax == 0:
                    for idx in self._indexes.values():
                        idx.add(key, ver.value)

            # Restore state
            txn.write_set = dict(sp.write_set)
            txn.read_set = dict(sp.read_set)

            # Release locks acquired after savepoint
            new_locks = txn.locks - sp.locks
            for key in new_locks:
                if self._write_locks.get(key) == txn.txn_id:
                    del self._write_locks[key]
            txn.locks = set(sp.locks)

            # Remove this savepoint and all after it
            txn.savepoints = txn.savepoints[:sp_idx]

    # -------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------

    def get(self, txn_id: int, key: str) -> Optional[Any]:
        """Read a value, respecting MVCC visibility."""
        with self._lock:
            txn = self._get_active_txn(txn_id)

            # For read-committed, refresh snapshot each read
            if txn.isolation == IsolationLevel.READ_COMMITTED:
                txn.snapshot = self._take_snapshot()

            version = self._find_visible_version(txn, key)
            if version is None:
                return None

            # Track reads for SSI
            txn.read_set[key] = version.xmin

            return version.value

    def scan(self, txn_id: int, prefix: str = "") -> dict[str, Any]:
        """Scan all visible key-value pairs, optionally filtered by prefix."""
        with self._lock:
            txn = self._get_active_txn(txn_id)

            if txn.isolation == IsolationLevel.READ_COMMITTED:
                txn.snapshot = self._take_snapshot()

            result = {}
            for key in self._versions:
                if prefix and not key.startswith(prefix):
                    continue
                version = self._find_visible_version(txn, key)
                if version is not None:
                    result[key] = version.value
                    txn.read_set[key] = version.xmin

            return result

    def range_scan(self, txn_id: int, lo: str, hi: str) -> dict[str, Any]:
        """Scan keys in range [lo, hi] inclusive."""
        with self._lock:
            txn = self._get_active_txn(txn_id)

            if txn.isolation == IsolationLevel.READ_COMMITTED:
                txn.snapshot = self._take_snapshot()

            result = {}
            for key in self._versions:
                if lo <= key <= hi:
                    version = self._find_visible_version(txn, key)
                    if version is not None:
                        result[key] = version.value
                        txn.read_set[key] = version.xmin

            return result

    def exists(self, txn_id: int, key: str) -> bool:
        """Check if a key exists (is visible) in this transaction."""
        return self.get(txn_id, key) is not None

    def count(self, txn_id: int, prefix: str = "") -> int:
        """Count visible keys, optionally filtered by prefix."""
        return len(self.scan(txn_id, prefix))

    # -------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------

    def put(self, txn_id: int, key: str, value: Any):
        """Insert or update a key-value pair."""
        with self._lock:
            txn = self._get_active_txn(txn_id)
            self._acquire_write_lock(txn, key)

            # First-updater-wins: check if a concurrent committed txn
            # wrote a version we can't see
            self._check_write_conflict(txn, key)

            old_version = self._find_visible_version(txn, key)

            # Create new version
            new_version = Version(key, value, txn.txn_id)

            if old_version is not None:
                # Mark old version as deleted by us
                old_version.xmax = txn.txn_id
                old_version.deleted_by = txn.txn_id
                new_version.prev_version = old_version
                # Update indexes: remove old, add new
                for idx in self._indexes.values():
                    idx.remove(key, old_version.value)
            elif key in self._versions:
                # Key exists but not visible to us -- chain behind head
                new_version.prev_version = self._versions[key]

            self._versions[key] = new_version

            # Update indexes
            for idx in self._indexes.values():
                idx.add(key, value)

            # Track writes
            txn.write_set[key] = new_version

    def delete(self, txn_id: int, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        with self._lock:
            txn = self._get_active_txn(txn_id)
            self._acquire_write_lock(txn, key)

            # First-updater-wins check
            self._check_write_conflict(txn, key)

            version = self._find_visible_version(txn, key)
            if version is None:
                return False

            # Mark as deleted
            version.xmax = txn.txn_id
            version.deleted_by = txn.txn_id

            # Update indexes
            for idx in self._indexes.values():
                idx.remove(key, version.value)

            # Track the deletion
            txn.write_set[key] = version
            return True

    # -------------------------------------------------------------------
    # Index Operations
    # -------------------------------------------------------------------

    def create_index(self, name: str, key_func) -> MVCCIndex:
        """Create a secondary index."""
        index = MVCCIndex(name, key_func)
        self._indexes[name] = index
        # Build index from current committed data
        for key, head in self._versions.items():
            ver = head
            while ver is not None:
                if ver.xmax == 0 or ver.xmax in self._aborted_txns:
                    if ver.xmin in self._committed_txns or ver.xmin == 0:
                        index.add(key, ver.value)
                        break
                ver = ver.prev_version
        return index

    def index_lookup(self, txn_id: int, index_name: str, index_key: Any) -> dict[str, Any]:
        """Look up values using a secondary index."""
        with self._lock:
            txn = self._get_active_txn(txn_id)
            index = self._indexes.get(index_name)
            if index is None:
                raise ValueError(f"Index '{index_name}' not found")

            primary_keys = index.lookup(index_key)
            result = {}
            for pk in primary_keys:
                version = self._find_visible_version(txn, pk)
                if version is not None:
                    # Verify the index key still matches for this visible version
                    if index.key_func(version.value) == index_key:
                        result[pk] = version.value
                        txn.read_set[pk] = version.xmin
            return result

    # -------------------------------------------------------------------
    # Garbage Collection
    # -------------------------------------------------------------------

    def gc(self) -> int:
        """
        Garbage collect old versions no longer visible to any active transaction.
        Returns number of versions collected.
        """
        with self._lock:
            if not self._active_txns:
                oldest = self._next_txn_id
            else:
                oldest = min(self._active_txns.keys())

            collected = 0
            keys_to_check = list(self._versions.keys())

            for key in keys_to_check:
                head = self._versions.get(key)
                if head is None:
                    continue

                # Walk the chain, remove versions invisible to all
                collected += self._gc_chain(key, head, oldest)

            # Clean up aborted txn tracking for old txns
            self._aborted_txns = {
                xid for xid in self._aborted_txns if xid >= oldest
            }

            return collected

    def _gc_chain(self, key: str, head: Version, oldest_active: int) -> int:
        """GC a version chain. Returns count of collected versions."""
        collected = 0

        # If head is deleted and the deleter committed before oldest_active,
        # the entire chain can be removed
        if head.xmax != 0 and head.xmax not in self._active_txns:
            if head.xmax < oldest_active and head.xmax not in self._aborted_txns:
                # Count versions in chain
                ver = head
                while ver is not None:
                    collected += 1
                    ver = ver.prev_version
                del self._versions[key]
                return collected

        # Otherwise, trim old versions from the tail
        curr = head
        while curr.prev_version is not None:
            prev = curr.prev_version
            # If prev was created before oldest_active and has been superseded,
            # it's invisible to everyone -- remove it and everything after
            if prev.xmin < oldest_active and prev.xmax != 0:
                if prev.xmax < oldest_active and prev.xmax not in self._aborted_txns:
                    # Count remaining chain
                    ver = prev
                    while ver is not None:
                        collected += 1
                        ver = ver.prev_version
                    curr.prev_version = None
                    break
            curr = curr.prev_version
            if curr is None:
                break

        return collected

    # -------------------------------------------------------------------
    # Visibility Rules
    # -------------------------------------------------------------------

    def _find_visible_version(self, txn: Transaction, key: str) -> Optional[Version]:
        """Find the version of key visible to the given transaction."""
        head = self._versions.get(key)
        if head is None:
            return None

        ver = head
        while ver is not None:
            if self._is_visible(txn, ver):
                return ver
            ver = ver.prev_version

        return None

    def _is_visible(self, txn: Transaction, ver: Version) -> bool:
        """
        MVCC visibility check.

        A version is visible if:
        1. It was created by the current transaction, OR
        2. It was created by a committed transaction visible in our snapshot
        AND it has NOT been deleted, or was deleted by a transaction
        not visible in our snapshot.
        """
        # Case 1: We created this version
        if ver.xmin == txn.txn_id:
            # But check if we also deleted it
            if ver.xmax == txn.txn_id:
                return False
            if ver.xmax != 0 and ver.xmax != txn.txn_id:
                # Someone else deleted it -- is that visible?
                if ver.xmax in self._committed_txns and txn.snapshot.is_visible(ver.xmax):
                    return False
            return True

        # Case 2: Created by an aborted transaction -- never visible
        if ver.xmin in self._aborted_txns:
            return False

        # Case 3: Created by another active transaction -- not visible
        if ver.xmin in self._active_txns and ver.xmin != txn.txn_id:
            return False

        # Case 4: Created by a committed transaction
        if not txn.snapshot.is_visible(ver.xmin):
            return False

        # Now check deletion
        if ver.xmax == 0:
            return True  # Not deleted

        if ver.xmax == txn.txn_id:
            return False  # We deleted it

        if ver.xmax in self._aborted_txns:
            return True  # Deleter aborted, so deletion doesn't count

        if ver.xmax in self._active_txns:
            return True  # Deleter still active, not committed yet

        # Deleter committed -- is it visible in our snapshot?
        if txn.snapshot.is_visible(ver.xmax):
            return False  # Deletion is visible

        return True  # Deletion happened after our snapshot

    # -------------------------------------------------------------------
    # Snapshot Management
    # -------------------------------------------------------------------

    def _take_snapshot(self) -> Snapshot:
        """Take a snapshot of the current transaction state."""
        active = frozenset(self._active_txns.keys())
        xmin = min(active) if active else self._next_txn_id
        xmax = self._next_txn_id
        return Snapshot(xmin, xmax, active)

    def _update_oldest_active(self):
        """Update the oldest active transaction ID."""
        if self._active_txns:
            self._oldest_active_xid = min(self._active_txns.keys())
        else:
            self._oldest_active_xid = self._next_txn_id

    # -------------------------------------------------------------------
    # Lock Management
    # -------------------------------------------------------------------

    def _acquire_write_lock(self, txn: Transaction, key: str):
        """Acquire a write lock on a key."""
        if key in txn.locks:
            return  # Already hold it

        current_holder = self._write_locks.get(key)
        if current_holder is not None and current_holder != txn.txn_id:
            if current_holder in self._active_txns:
                raise WriteConflictError(
                    f"Key '{key}' is locked by txn {current_holder}")
            # Holder committed or aborted, lock is stale
            del self._write_locks[key]

        self._write_locks[key] = txn.txn_id
        txn.locks.add(key)

    def _check_write_conflict(self, txn: Transaction, key: str):
        """
        First-updater-wins: detect if a concurrent committed transaction
        wrote a new version of this key that is not visible to our snapshot.
        """
        head = self._versions.get(key)
        if head is None:
            return

        # If the head version was created by a committed txn not in our snapshot,
        # that means someone else updated the key concurrently
        if (head.xmin != txn.txn_id and
            head.xmin in self._committed_txns and
            not txn.snapshot.is_visible(head.xmin)):
            raise WriteConflictError(
                f"Key '{key}' was modified by concurrent committed txn {head.xmin}")

    # -------------------------------------------------------------------
    # SSI (Serializable Snapshot Isolation) Validation
    # -------------------------------------------------------------------

    def _ssi_validate(self, txn: Transaction):
        """
        Validate transaction for serializable isolation.
        Detects write skew and other serialization anomalies.
        """
        # Check if any key we read was modified by a concurrent committed txn
        for key, read_xmin in txn.read_set.items():
            head = self._versions.get(key)
            if head is None:
                continue

            ver = head
            while ver is not None:
                # If a committed txn that's concurrent with us wrote to a key we read
                if (ver.xmin != txn.txn_id and
                    ver.xmin in self._committed_txns and
                    ver.xmin >= txn.snapshot.xmin and
                    ver.xmin in txn.snapshot.active_txns):
                    raise SerializationError(
                        f"Serialization failure: key '{key}' was modified by "
                        f"concurrent txn {ver.xmin}")

                # Also check if a concurrent committed txn deleted a version we read
                if (ver.xmin == read_xmin and ver.xmax != 0 and
                    ver.xmax != txn.txn_id and
                    ver.xmax in self._committed_txns and
                    ver.xmax >= txn.snapshot.xmin and
                    ver.xmax in txn.snapshot.active_txns):
                    raise SerializationError(
                        f"Serialization failure: key '{key}' version was deleted by "
                        f"concurrent txn {ver.xmax}")

                ver = ver.prev_version

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_active_txn(self, txn_id: int) -> Transaction:
        """Get an active transaction or raise."""
        txn = self._active_txns.get(txn_id)
        if txn is None:
            if txn_id in self._committed_txns:
                raise ValueError(f"Transaction {txn_id} already committed")
            if txn_id in self._aborted_txns:
                raise ValueError(f"Transaction {txn_id} already aborted")
            raise ValueError(f"Transaction {txn_id} not found")
        return txn

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def version_count(self, key: str) -> int:
        """Count versions in a key's chain (for testing/diagnostics)."""
        count = 0
        ver = self._versions.get(key)
        while ver is not None:
            count += 1
            ver = ver.prev_version
        return count

    def active_transactions(self) -> list[int]:
        """Return list of active transaction IDs."""
        return sorted(self._active_txns.keys())

    def stats(self) -> dict:
        """Return engine statistics."""
        total_versions = 0
        total_keys = len(self._versions)
        for head in self._versions.values():
            ver = head
            while ver is not None:
                total_versions += 1
                ver = ver.prev_version
        return {
            'total_keys': total_keys,
            'total_versions': total_versions,
            'active_txns': len(self._active_txns),
            'committed_txns': len(self._committed_txns),
            'aborted_txns': len(self._aborted_txns),
            'next_txn_id': self._next_txn_id,
            'indexes': len(self._indexes),
            'write_locks': len(self._write_locks),
        }
