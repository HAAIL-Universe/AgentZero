"""
C241: Write-Ahead Logging (WAL)

A complete WAL implementation for crash recovery:
- Log-structured write-ahead logging with LSN ordering
- ARIES-style recovery: Analysis, Redo, Undo phases
- Log record types: Begin, Commit, Abort, Insert, Update, Delete, Checkpoint, CLR
- Physiological logging (page-level redo, logical undo)
- Fuzzy checkpointing with dirty page table and active transaction table
- Log sequence numbers (LSN) for ordering and recovery
- Force-at-commit (WAL protocol) guarantees durability
- Group commit for throughput optimization
- Log truncation after checkpoint
- Compensation Log Records (CLR) for idempotent undo

Domain: Database Internals
Standalone implementation -- no external dependencies.
"""

from enum import Enum, auto
from typing import Any, Optional, Dict, List, Set, Tuple
import struct
import time
import threading
import hashlib


# ---------------------------------------------------------------------------
# Log Record Types
# ---------------------------------------------------------------------------

class LogRecordType(Enum):
    BEGIN = auto()
    COMMIT = auto()
    ABORT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    CHECKPOINT = auto()
    CLR = auto()           # Compensation Log Record (undo record)
    END = auto()           # Transaction fully completed (after commit/abort + undo)
    SAVEPOINT = auto()
    ROLLBACK_TO_SAVEPOINT = auto()


# ---------------------------------------------------------------------------
# Log Record
# ---------------------------------------------------------------------------

class LogRecord:
    """A single WAL log record."""

    __slots__ = ('lsn', 'prev_lsn', 'txn_id', 'record_type', 'page_id',
                 'key', 'old_value', 'new_value', 'timestamp',
                 'checkpoint_data', 'undo_next_lsn', 'savepoint_name')

    def __init__(self, lsn: int, txn_id: int, record_type: LogRecordType,
                 prev_lsn: int = 0, page_id: int = 0, key: str = '',
                 old_value: Any = None, new_value: Any = None,
                 checkpoint_data: Optional[dict] = None,
                 undo_next_lsn: int = 0, savepoint_name: str = ''):
        self.lsn = lsn
        self.prev_lsn = prev_lsn      # Previous LSN for this transaction
        self.txn_id = txn_id
        self.record_type = record_type
        self.page_id = page_id
        self.key = key
        self.old_value = old_value      # Before-image (for undo)
        self.new_value = new_value      # After-image (for redo)
        self.timestamp = time.time()
        self.checkpoint_data = checkpoint_data
        self.undo_next_lsn = undo_next_lsn  # For CLR: next LSN to undo
        self.savepoint_name = savepoint_name

    def is_undoable(self) -> bool:
        """Can this record be undone?"""
        return self.record_type in (
            LogRecordType.INSERT, LogRecordType.UPDATE, LogRecordType.DELETE
        )

    def is_redoable(self) -> bool:
        """Can this record be redone?"""
        return self.record_type in (
            LogRecordType.INSERT, LogRecordType.UPDATE, LogRecordType.DELETE,
            LogRecordType.CLR
        )

    def __repr__(self):
        return (f"LogRecord(lsn={self.lsn}, txn={self.txn_id}, "
                f"type={self.record_type.name}, page={self.page_id}, "
                f"key={self.key!r})")


# ---------------------------------------------------------------------------
# Page (simplified buffer pool page)
# ---------------------------------------------------------------------------

class Page:
    """A database page with LSN tracking."""

    __slots__ = ('page_id', 'data', 'page_lsn', 'rec_lsn', 'dirty')

    def __init__(self, page_id: int):
        self.page_id = page_id
        self.data: Dict[str, Any] = {}   # key -> value store within page
        self.page_lsn: int = 0           # LSN of last applied log record
        self.rec_lsn: int = 0            # First LSN that dirtied page since last flush
        self.dirty: bool = False

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def put(self, key: str, value: Any, lsn: int):
        self.data[key] = value
        self.page_lsn = lsn
        if not self.dirty:
            self.rec_lsn = lsn
        self.dirty = True

    def remove(self, key: str, lsn: int) -> Optional[Any]:
        old = self.data.pop(key, None)
        self.page_lsn = lsn
        if not self.dirty:
            self.rec_lsn = lsn
        self.dirty = True
        return old

    def flush(self):
        """Mark page as clean (written to disk)."""
        self.dirty = False
        self.rec_lsn = 0

    def __repr__(self):
        return f"Page(id={self.page_id}, lsn={self.page_lsn}, dirty={self.dirty})"


# ---------------------------------------------------------------------------
# Buffer Pool (simplified)
# ---------------------------------------------------------------------------

class BufferPool:
    """Simplified buffer pool for pages."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.pages: Dict[int, Page] = {}
        self.flushed_pages: Dict[int, Dict[str, Any]] = {}  # Simulates disk

    def get_page(self, page_id: int) -> Page:
        if page_id not in self.pages:
            # Load from disk or create new
            page = Page(page_id)
            if page_id in self.flushed_pages:
                page.data = dict(self.flushed_pages[page_id])
                page.dirty = False
            self.pages[page_id] = page
        return self.pages[page_id]

    def get_dirty_pages(self) -> Dict[int, int]:
        """Return dirty page table: page_id -> rec_lsn (first LSN that dirtied it)."""
        result = {}
        for pid, page in self.pages.items():
            if page.dirty:
                result[pid] = page.rec_lsn
        return result

    def flush_page(self, page_id: int):
        """Flush a page to 'disk'."""
        if page_id in self.pages:
            page = self.pages[page_id]
            self.flushed_pages[page_id] = dict(page.data)
            page.flush()

    def flush_all(self):
        """Flush all dirty pages."""
        for pid in list(self.pages.keys()):
            if self.pages[pid].dirty:
                self.flush_page(pid)

    def clear(self):
        """Simulate crash -- lose all unflushed pages."""
        self.pages.clear()

    def reload_from_disk(self):
        """Reload pages from disk after crash."""
        self.pages.clear()
        for pid, data in self.flushed_pages.items():
            page = Page(pid)
            page.data = dict(data)
            page.dirty = False
            self.pages[pid] = page


# ---------------------------------------------------------------------------
# Transaction Table Entry
# ---------------------------------------------------------------------------

class TransactionEntry:
    """Tracks an active transaction during recovery."""

    __slots__ = ('txn_id', 'status', 'last_lsn', 'undo_next_lsn', 'savepoints')

    def __init__(self, txn_id: int, status: str = 'active', last_lsn: int = 0):
        self.txn_id = txn_id
        self.status = status
        self.last_lsn = last_lsn
        self.undo_next_lsn = last_lsn
        self.savepoints: Dict[str, int] = {}  # name -> lsn at savepoint


# ---------------------------------------------------------------------------
# WAL (Write-Ahead Log)
# ---------------------------------------------------------------------------

class WAL:
    """
    Write-Ahead Log with ARIES-style recovery.

    Guarantees:
    - WAL protocol: log record written before data page
    - Force-at-commit: all log records flushed at commit
    - Steal: dirty pages can be written before commit
    - No-force: pages don't need to be written at commit
    """

    def __init__(self, buffer_pool: Optional[BufferPool] = None):
        self.buffer_pool = buffer_pool or BufferPool()
        self.log: List[LogRecord] = []
        self.flushed_lsn: int = 0         # Last LSN written to stable storage
        self.next_lsn: int = 1
        self.next_txn_id: int = 1
        self.active_txns: Dict[int, TransactionEntry] = {}
        self.committed_txns: Set[int] = set()
        self.aborted_txns: Set[int] = set()
        self.lock = threading.Lock()

        # Group commit
        self.group_commit_enabled: bool = False
        self.pending_commits: List[int] = []
        self.group_commit_interval: float = 0.01  # 10ms

        # Checkpoint
        self.last_checkpoint_lsn: int = 0

        # For recovery
        self._log_on_disk: List[LogRecord] = []  # Simulates log on stable storage

    # -------------------------------------------------------------------
    # LSN Management
    # -------------------------------------------------------------------

    def _next_lsn(self) -> int:
        lsn = self.next_lsn
        self.next_lsn += 1
        return lsn

    def _next_txn_id(self) -> int:
        txn_id = self.next_txn_id
        self.next_txn_id += 1
        return txn_id

    # -------------------------------------------------------------------
    # Log Record Writing
    # -------------------------------------------------------------------

    def _write_log(self, txn_id: int, record_type: LogRecordType,
                   page_id: int = 0, key: str = '',
                   old_value: Any = None, new_value: Any = None,
                   checkpoint_data: Optional[dict] = None,
                   undo_next_lsn: int = 0,
                   savepoint_name: str = '') -> LogRecord:
        """Write a log record and return it."""
        prev_lsn = 0
        if txn_id in self.active_txns:
            prev_lsn = self.active_txns[txn_id].last_lsn

        lsn = self._next_lsn()
        record = LogRecord(
            lsn=lsn, txn_id=txn_id, record_type=record_type,
            prev_lsn=prev_lsn, page_id=page_id, key=key,
            old_value=old_value, new_value=new_value,
            checkpoint_data=checkpoint_data,
            undo_next_lsn=undo_next_lsn,
            savepoint_name=savepoint_name
        )
        self.log.append(record)

        # Update transaction's last LSN
        if txn_id in self.active_txns:
            self.active_txns[txn_id].last_lsn = lsn
            self.active_txns[txn_id].undo_next_lsn = lsn

        return record

    def _flush_log(self, up_to_lsn: int = 0):
        """Flush log records to stable storage up to given LSN."""
        target = up_to_lsn or (self.next_lsn - 1)
        for record in self.log:
            if record.lsn > self.flushed_lsn and record.lsn <= target:
                self._log_on_disk.append(record)
        self.flushed_lsn = target

    # -------------------------------------------------------------------
    # Transaction Operations
    # -------------------------------------------------------------------

    def begin(self) -> int:
        """Begin a new transaction. Returns transaction ID."""
        with self.lock:
            txn_id = self._next_txn_id()
            self.active_txns[txn_id] = TransactionEntry(txn_id)
            self._write_log(txn_id, LogRecordType.BEGIN)
            return txn_id

    def commit(self, txn_id: int):
        """Commit a transaction. Force-flushes log."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            record = self._write_log(txn_id, LogRecordType.COMMIT)

            if self.group_commit_enabled:
                self.pending_commits.append(txn_id)
            else:
                # Force flush -- WAL protocol
                self._flush_log(record.lsn)

            # Write END record
            self._write_log(txn_id, LogRecordType.END)

            self.committed_txns.add(txn_id)
            del self.active_txns[txn_id]

            if not self.group_commit_enabled:
                self._flush_log()

    def abort(self, txn_id: int):
        """Abort a transaction. Undo all changes."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            # Undo all changes
            self._undo_transaction(txn_id)

            self._write_log(txn_id, LogRecordType.ABORT)
            self._write_log(txn_id, LogRecordType.END)
            self._flush_log()

            self.aborted_txns.add(txn_id)
            del self.active_txns[txn_id]

    def savepoint(self, txn_id: int, name: str):
        """Create a savepoint within a transaction."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            entry = self.active_txns[txn_id]
            record = self._write_log(txn_id, LogRecordType.SAVEPOINT,
                                     savepoint_name=name)
            entry.savepoints[name] = record.lsn

    def rollback_to_savepoint(self, txn_id: int, name: str):
        """Rollback to a named savepoint."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            entry = self.active_txns[txn_id]
            if name not in entry.savepoints:
                raise ValueError(f"Savepoint {name!r} not found")

            savepoint_lsn = entry.savepoints[name]
            self._undo_to_lsn(txn_id, savepoint_lsn)

            self._write_log(txn_id, LogRecordType.ROLLBACK_TO_SAVEPOINT,
                           savepoint_name=name)

            # Remove savepoints created after this one
            to_remove = [sp for sp, lsn in entry.savepoints.items()
                        if lsn > savepoint_lsn]
            for sp in to_remove:
                del entry.savepoints[sp]

    # -------------------------------------------------------------------
    # Data Operations (with WAL logging)
    # -------------------------------------------------------------------

    def insert(self, txn_id: int, page_id: int, key: str, value: Any):
        """Insert a key-value pair into a page."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            page = self.buffer_pool.get_page(page_id)
            if key in page.data:
                raise ValueError(f"Key {key!r} already exists on page {page_id}")

            # Write log FIRST (WAL protocol)
            record = self._write_log(
                txn_id, LogRecordType.INSERT,
                page_id=page_id, key=key, new_value=value
            )

            # Then modify the page
            page.put(key, value, record.lsn)

    def update(self, txn_id: int, page_id: int, key: str, new_value: Any):
        """Update an existing key-value pair."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            page = self.buffer_pool.get_page(page_id)
            old_value = page.get(key)
            if old_value is None and key not in page.data:
                raise ValueError(f"Key {key!r} not found on page {page_id}")

            # Write log FIRST
            record = self._write_log(
                txn_id, LogRecordType.UPDATE,
                page_id=page_id, key=key,
                old_value=old_value, new_value=new_value
            )

            # Then modify the page
            page.put(key, new_value, record.lsn)

    def delete(self, txn_id: int, page_id: int, key: str):
        """Delete a key from a page."""
        with self.lock:
            if txn_id not in self.active_txns:
                raise ValueError(f"Transaction {txn_id} not active")

            page = self.buffer_pool.get_page(page_id)
            old_value = page.get(key)
            if old_value is None and key not in page.data:
                raise ValueError(f"Key {key!r} not found on page {page_id}")

            # Write log FIRST
            record = self._write_log(
                txn_id, LogRecordType.DELETE,
                page_id=page_id, key=key, old_value=old_value
            )

            # Then modify the page
            page.remove(key, record.lsn)

    def read(self, txn_id: int, page_id: int, key: str) -> Optional[Any]:
        """Read a value (no logging needed for reads)."""
        page = self.buffer_pool.get_page(page_id)
        return page.get(key)

    # -------------------------------------------------------------------
    # Undo Operations
    # -------------------------------------------------------------------

    def _undo_record(self, record: LogRecord):
        """Undo a single log record, writing a CLR."""
        page = self.buffer_pool.get_page(record.page_id)

        if record.record_type == LogRecordType.INSERT:
            # Undo insert = delete
            page.remove(record.key, record.lsn)
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=record.new_value, new_value=None,
                undo_next_lsn=record.prev_lsn
            )

        elif record.record_type == LogRecordType.UPDATE:
            # Undo update = restore old value
            page.put(record.key, record.old_value, record.lsn)
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=record.new_value, new_value=record.old_value,
                undo_next_lsn=record.prev_lsn
            )

        elif record.record_type == LogRecordType.DELETE:
            # Undo delete = re-insert
            page.put(record.key, record.old_value, record.lsn)
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=None, new_value=record.old_value,
                undo_next_lsn=record.prev_lsn
            )

    def _undo_transaction(self, txn_id: int):
        """Undo all operations for a transaction (reverse order)."""
        # Collect undoable records for this transaction in reverse order
        records = [r for r in self.log
                   if r.txn_id == txn_id and r.is_undoable()]
        for record in reversed(records):
            self._undo_record(record)

    def _undo_to_lsn(self, txn_id: int, target_lsn: int):
        """Undo operations for a transaction back to target LSN."""
        records = [r for r in self.log
                   if r.txn_id == txn_id and r.is_undoable()
                   and r.lsn > target_lsn]
        for record in reversed(records):
            self._undo_record(record)

    # -------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------

    def checkpoint(self):
        """
        Write a fuzzy checkpoint.
        Records active transactions and dirty page table.
        """
        with self.lock:
            # Build active transaction table
            att = {}
            for txn_id, entry in self.active_txns.items():
                att[txn_id] = {
                    'status': entry.status,
                    'last_lsn': entry.last_lsn,
                    'undo_next_lsn': entry.undo_next_lsn
                }

            # Build dirty page table
            dpt = self.buffer_pool.get_dirty_pages()

            checkpoint_data = {
                'active_txns': att,
                'dirty_pages': dpt,
                'next_txn_id': self.next_txn_id
            }

            record = self._write_log(
                0, LogRecordType.CHECKPOINT,
                checkpoint_data=checkpoint_data
            )

            self._flush_log(record.lsn)
            self.last_checkpoint_lsn = record.lsn

            return record.lsn

    # -------------------------------------------------------------------
    # Group Commit
    # -------------------------------------------------------------------

    def enable_group_commit(self, interval: float = 0.01):
        """Enable group commit with given interval."""
        self.group_commit_enabled = True
        self.group_commit_interval = interval

    def flush_group_commit(self):
        """Flush all pending group commits."""
        with self.lock:
            if self.pending_commits:
                self._flush_log()
                self.pending_commits.clear()

    # -------------------------------------------------------------------
    # Log Truncation
    # -------------------------------------------------------------------

    def truncate_log(self, up_to_lsn: int = 0):
        """
        Truncate log records that are no longer needed.
        Safe to truncate before the min of:
        - Last checkpoint LSN
        - Oldest active transaction's first LSN
        """
        if up_to_lsn == 0:
            up_to_lsn = self._safe_truncation_point()

        if up_to_lsn <= 0:
            return 0

        before = len(self.log)
        self.log = [r for r in self.log if r.lsn >= up_to_lsn]
        self._log_on_disk = [r for r in self._log_on_disk if r.lsn >= up_to_lsn]
        return before - len(self.log)

    def _safe_truncation_point(self) -> int:
        """Calculate the safe truncation LSN."""
        candidates = []
        if self.last_checkpoint_lsn > 0:
            candidates.append(self.last_checkpoint_lsn)

        # Don't truncate past oldest active transaction
        for entry in self.active_txns.values():
            first_lsn = self._find_first_lsn(entry.txn_id)
            if first_lsn > 0:
                candidates.append(first_lsn)

        return min(candidates) if candidates else 0

    def _find_first_lsn(self, txn_id: int) -> int:
        """Find the first LSN for a given transaction."""
        for record in self.log:
            if record.txn_id == txn_id:
                return record.lsn
        return 0

    # -------------------------------------------------------------------
    # ARIES Recovery
    # -------------------------------------------------------------------

    def simulate_crash(self):
        """Simulate a crash: flush log, lose buffer pool state."""
        self._flush_log()
        self.buffer_pool.clear()
        # Lose in-memory state
        self.active_txns.clear()
        self.committed_txns.clear()
        self.aborted_txns.clear()
        # Keep log on disk
        self.log = list(self._log_on_disk)

    def recover(self) -> dict:
        """
        ARIES-style recovery with three phases:
        1. Analysis: determine what needs to be redone/undone
        2. Redo: replay all operations from appropriate point
        3. Undo: roll back incomplete transactions

        Returns recovery statistics.
        """
        # Restore log from disk
        self.log = list(self._log_on_disk)
        if not self.log:
            return {'analysis': {}, 'redo_count': 0, 'undo_count': 0}

        # Phase 1: Analysis
        analysis = self._analysis_phase()

        # Phase 2: Redo
        redo_count = self._redo_phase(analysis)

        # Phase 3: Undo
        undo_count = self._undo_phase(analysis)

        return {
            'analysis': {
                'active_txns': list(analysis['loser_txns']),
                'committed_txns': list(analysis['winner_txns']),
                'dirty_pages': dict(analysis['dirty_pages']),
            },
            'redo_count': redo_count,
            'undo_count': undo_count
        }

    def _analysis_phase(self) -> dict:
        """
        Analysis phase: scan log from last checkpoint.
        Build active transaction table and dirty page table.
        """
        # Start from checkpoint if available
        start_lsn = 0
        active_txns: Dict[int, TransactionEntry] = {}
        dirty_pages: Dict[int, int] = {}  # page_id -> rec_lsn (first LSN that dirtied it)

        # Find last checkpoint
        checkpoint_record = None
        for record in reversed(self.log):
            if record.record_type == LogRecordType.CHECKPOINT:
                checkpoint_record = record
                start_lsn = record.lsn
                break

        completed_committed: Set[int] = set()  # Tracks txns that committed + ended

        # Initialize from checkpoint data
        if checkpoint_record and checkpoint_record.checkpoint_data:
            cpd = checkpoint_record.checkpoint_data
            for txn_id_str, info in cpd.get('active_txns', {}).items():
                txn_id = int(txn_id_str) if isinstance(txn_id_str, str) else txn_id_str
                active_txns[txn_id] = TransactionEntry(
                    txn_id, info['status'], info['last_lsn']
                )
            for page_id_str, rec_lsn in cpd.get('dirty_pages', {}).items():
                page_id = int(page_id_str) if isinstance(page_id_str, str) else page_id_str
                dirty_pages[page_id] = rec_lsn

        # Scan forward from checkpoint
        for record in self.log:
            if record.lsn < start_lsn:
                continue

            txn_id = record.txn_id

            if record.record_type == LogRecordType.BEGIN:
                active_txns[txn_id] = TransactionEntry(txn_id)

            elif record.record_type == LogRecordType.COMMIT:
                if txn_id in active_txns:
                    active_txns[txn_id].status = 'committed'

            elif record.record_type == LogRecordType.ABORT:
                if txn_id in active_txns:
                    active_txns[txn_id].status = 'aborted'

            elif record.record_type == LogRecordType.END:
                if txn_id in active_txns:
                    if active_txns[txn_id].status == 'committed':
                        completed_committed.add(txn_id)
                    del active_txns[txn_id]

            elif record.is_redoable():
                # Track dirty pages (page_id 0 is valid)
                if record.page_id not in dirty_pages:
                    dirty_pages[record.page_id] = record.lsn

            # Update transaction's last LSN
            if txn_id in active_txns:
                active_txns[txn_id].last_lsn = record.lsn
                if record.record_type == LogRecordType.CLR:
                    active_txns[txn_id].undo_next_lsn = record.undo_next_lsn
                else:
                    active_txns[txn_id].undo_next_lsn = record.lsn

        # Classify transactions
        winner_txns = set()
        loser_txns = set()
        # Transactions still in active_txns at end of scan
        for txn_id, entry in active_txns.items():
            if entry.status == 'committed':
                winner_txns.add(txn_id)
            else:
                loser_txns.add(txn_id)
        # Also include fully-completed committed transactions
        winner_txns |= completed_committed

        return {
            'active_txns': active_txns,
            'dirty_pages': dirty_pages,
            'winner_txns': winner_txns,
            'loser_txns': loser_txns,
            'start_lsn': start_lsn
        }

    def _redo_phase(self, analysis: dict) -> int:
        """
        Redo phase: replay all redoable records from the minimum rec_lsn
        in the dirty page table.
        """
        dirty_pages = analysis['dirty_pages']
        if not dirty_pages:
            # Still need to redo from start to reconstruct pages
            min_lsn = analysis.get('start_lsn', 0)
        else:
            min_lsn = min(dirty_pages.values())

        redo_count = 0
        for record in self.log:
            if record.lsn < min_lsn:
                continue
            if not record.is_redoable():
                continue

            page = self.buffer_pool.get_page(record.page_id)

            # Skip if page LSN >= record LSN (already applied)
            if page.page_lsn >= record.lsn:
                continue

            # Redo the operation
            self._redo_record(record, page)
            redo_count += 1

        return redo_count

    def _redo_record(self, record: LogRecord, page: Page):
        """Apply a single log record during redo."""
        if record.record_type == LogRecordType.INSERT:
            page.put(record.key, record.new_value, record.lsn)
        elif record.record_type == LogRecordType.UPDATE:
            page.put(record.key, record.new_value, record.lsn)
        elif record.record_type == LogRecordType.DELETE:
            page.data.pop(record.key, None)
            page.page_lsn = record.lsn
            page.dirty = True
        elif record.record_type == LogRecordType.CLR:
            # CLR redo: apply the compensation action
            if record.new_value is not None:
                page.put(record.key, record.new_value, record.lsn)
            else:
                page.data.pop(record.key, None)
                page.page_lsn = record.lsn
                page.dirty = True

    def _undo_phase(self, analysis: dict) -> int:
        """
        Undo phase: roll back all loser transactions.
        Uses CLR records for idempotent undo.
        """
        loser_txns = analysis['loser_txns']
        if not loser_txns:
            return 0

        # Build set of LSNs to undo
        undo_count = 0

        # Collect undo_next_lsn for each loser
        to_undo: Dict[int, int] = {}  # txn_id -> next lsn to undo
        for txn_id in loser_txns:
            entry = analysis['active_txns'].get(txn_id)
            if entry:
                to_undo[txn_id] = entry.undo_next_lsn

        # Process in reverse LSN order (largest first)
        while to_undo:
            # Find the loser with the largest undo_next_lsn
            max_txn = max(to_undo.keys(), key=lambda t: to_undo[t])
            current_lsn = to_undo[max_txn]

            if current_lsn == 0:
                # Reached beginning -- write abort + end
                self._write_log(max_txn, LogRecordType.ABORT)
                self._write_log(max_txn, LogRecordType.END)
                del to_undo[max_txn]
                continue

            # Find the record at current_lsn
            record = self._find_record(current_lsn)
            if record is None:
                del to_undo[max_txn]
                continue

            if record.record_type == LogRecordType.CLR:
                # Follow the undo_next_lsn chain
                to_undo[max_txn] = record.undo_next_lsn
            elif record.is_undoable():
                # Undo this record
                self._undo_record_recovery(record)
                to_undo[max_txn] = record.prev_lsn
                undo_count += 1
            else:
                # Skip non-undoable (BEGIN, SAVEPOINT, etc.)
                to_undo[max_txn] = record.prev_lsn

        self._flush_log()
        return undo_count

    def _undo_record_recovery(self, record: LogRecord):
        """Undo a record during recovery, writing a CLR."""
        page = self.buffer_pool.get_page(record.page_id)

        if record.record_type == LogRecordType.INSERT:
            page.data.pop(record.key, None)
            page.page_lsn = record.lsn
            page.dirty = True
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=record.new_value, new_value=None,
                undo_next_lsn=record.prev_lsn
            )
        elif record.record_type == LogRecordType.UPDATE:
            page.put(record.key, record.old_value, record.lsn)
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=record.new_value, new_value=record.old_value,
                undo_next_lsn=record.prev_lsn
            )
        elif record.record_type == LogRecordType.DELETE:
            page.put(record.key, record.old_value, record.lsn)
            self._write_log(
                record.txn_id, LogRecordType.CLR,
                page_id=record.page_id, key=record.key,
                old_value=None, new_value=record.old_value,
                undo_next_lsn=record.prev_lsn
            )

    def _find_record(self, lsn: int) -> Optional[LogRecord]:
        """Find a log record by LSN."""
        for record in self.log:
            if record.lsn == lsn:
                return record
        return None

    # -------------------------------------------------------------------
    # Log Serialization (binary format)
    # -------------------------------------------------------------------

    def serialize_log(self) -> bytes:
        """Serialize the log to a binary format."""
        import json
        data = []
        for record in self.log:
            entry = {
                'lsn': record.lsn,
                'prev_lsn': record.prev_lsn,
                'txn_id': record.txn_id,
                'type': record.record_type.name,
                'page_id': record.page_id,
                'key': record.key,
                'old_value': record.old_value,
                'new_value': record.new_value,
                'undo_next_lsn': record.undo_next_lsn,
                'savepoint_name': record.savepoint_name,
            }
            if record.checkpoint_data:
                entry['checkpoint_data'] = record.checkpoint_data
            data.append(entry)

        payload = json.dumps(data).encode('utf-8')
        checksum = hashlib.md5(payload).digest()
        return checksum + payload

    def deserialize_log(self, data: bytes) -> List[LogRecord]:
        """Deserialize a binary log."""
        import json
        checksum = data[:16]
        payload = data[16:]
        if hashlib.md5(payload).digest() != checksum:
            raise ValueError("Log checksum mismatch -- log corruption detected")

        entries = json.loads(payload.decode('utf-8'))
        records = []
        for entry in entries:
            record = LogRecord(
                lsn=entry['lsn'],
                txn_id=entry['txn_id'],
                record_type=LogRecordType[entry['type']],
                prev_lsn=entry['prev_lsn'],
                page_id=entry['page_id'],
                key=entry['key'],
                old_value=entry.get('old_value'),
                new_value=entry.get('new_value'),
                undo_next_lsn=entry.get('undo_next_lsn', 0),
                savepoint_name=entry.get('savepoint_name', ''),
                checkpoint_data=entry.get('checkpoint_data')
            )
            records.append(record)
        return records

    # -------------------------------------------------------------------
    # Statistics & Inspection
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        """Return WAL statistics."""
        type_counts = {}
        for record in self.log:
            name = record.record_type.name
            type_counts[name] = type_counts.get(name, 0) + 1

        return {
            'total_records': len(self.log),
            'flushed_lsn': self.flushed_lsn,
            'next_lsn': self.next_lsn,
            'active_txns': len(self.active_txns),
            'committed_txns': len(self.committed_txns),
            'aborted_txns': len(self.aborted_txns),
            'last_checkpoint_lsn': self.last_checkpoint_lsn,
            'record_types': type_counts,
            'log_on_disk': len(self._log_on_disk)
        }

    def get_log_records(self, txn_id: Optional[int] = None,
                        record_type: Optional[LogRecordType] = None
                        ) -> List[LogRecord]:
        """Get log records with optional filtering."""
        records = self.log
        if txn_id is not None:
            records = [r for r in records if r.txn_id == txn_id]
        if record_type is not None:
            records = [r for r in records if r.record_type == record_type]
        return records

    def get_transaction_chain(self, txn_id: int) -> List[LogRecord]:
        """Follow the prev_lsn chain for a transaction."""
        chain = []
        # Find last record for this transaction
        last = None
        for record in reversed(self.log):
            if record.txn_id == txn_id:
                last = record
                break

        if last is None:
            return chain

        # Follow prev_lsn chain
        current = last
        while current:
            chain.append(current)
            if current.prev_lsn == 0:
                break
            current = self._find_record(current.prev_lsn)

        chain.reverse()
        return chain
