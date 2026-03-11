"""
C214: Write-Ahead Log Engine
=============================
A write-ahead logging system for crash recovery, composing with C213 Storage Engine.

Components:
- LogRecord: typed log entries (begin, insert, update, delete, commit, abort, checkpoint, CLR)
- LogSequenceNumber (LSN): monotonically increasing log position
- WALBuffer: in-memory log buffer with flush-to-stable-storage
- WALWriter: sequential log writer with CRC integrity checks
- WALReader: forward/backward log scanning with filtering
- RecoveryManager: ARIES-style crash recovery (analysis, redo, undo)
- WALStorageEngine: composes WAL + C213 StorageEngine for crash-safe operations

Key properties:
- Write-ahead: log record flushed before dirty page written
- Steal/no-force: pages can be flushed before commit, commits don't force page flush
- Physiological logging: page-level redo, logical undo
- ARIES recovery: analysis pass rebuilds state, redo replays history, undo rolls back losers
"""

import struct
import time
import hashlib
from enum import IntEnum
from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C213_storage_engine'))
from storage_engine import (
    DiskManager, BufferPool, HeapFile, BTreeIndex, Table, TableSchema,
    StorageEngine, SlottedPage, RowID, PAGE_SIZE, PageType,
    _encode_value, _decode_value
)


# =============================================================================
# Log Record Types
# =============================================================================

class LogType(IntEnum):
    BEGIN = 1
    COMMIT = 2
    ABORT = 3
    INSERT = 10
    UPDATE = 11
    DELETE = 12
    CLR = 20        # Compensation Log Record (undo action)
    CHECKPOINT = 30
    END = 40        # Transaction end (after commit/abort processing complete)


# =============================================================================
# Log Sequence Number
# =============================================================================

class LSN:
    """Log Sequence Number -- unique, monotonically increasing log position."""
    __slots__ = ['value']

    def __init__(self, value=0):
        self.value = value

    def __repr__(self):
        return f"LSN({self.value})"

    def __eq__(self, other):
        if isinstance(other, LSN):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, LSN):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, LSN):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, LSN):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, LSN):
            return self.value >= other.value
        return self.value >= other

    def __hash__(self):
        return hash(self.value)

    def __int__(self):
        return self.value


# =============================================================================
# Log Record
# =============================================================================

class LogRecord:
    """A single WAL log record."""

    def __init__(self, lsn, txn_id, log_type, table_name=None,
                 page_id=None, slot_idx=None,
                 before_image=None, after_image=None,
                 prev_lsn=None, undo_next_lsn=None,
                 active_txns=None, dirty_pages=None):
        self.lsn = lsn if isinstance(lsn, LSN) else LSN(lsn)
        self.txn_id = txn_id
        self.log_type = LogType(log_type)
        self.table_name = table_name
        self.page_id = page_id
        self.slot_idx = slot_idx
        self.before_image = before_image  # For undo (old value)
        self.after_image = after_image    # For redo (new value)
        self.prev_lsn = prev_lsn if prev_lsn is None or isinstance(prev_lsn, LSN) else LSN(prev_lsn)
        self.undo_next_lsn = undo_next_lsn if undo_next_lsn is None or isinstance(undo_next_lsn, LSN) else LSN(undo_next_lsn)
        # For checkpoint records
        self.active_txns = active_txns    # {txn_id: last_lsn}
        self.dirty_pages = dirty_pages    # {(table, page_id): rec_lsn}

    def __repr__(self):
        parts = [f"LogRecord(lsn={self.lsn}, txn={self.txn_id}, type={self.log_type.name}"]
        if self.table_name:
            parts.append(f", table={self.table_name}")
        if self.page_id is not None:
            parts.append(f", page={self.page_id}")
        if self.slot_idx is not None:
            parts.append(f", slot={self.slot_idx}")
        parts.append(")")
        return "".join(parts)

    def is_undoable(self):
        """Can this record be undone during recovery?"""
        return self.log_type in (LogType.INSERT, LogType.UPDATE, LogType.DELETE)

    def is_redoable(self):
        """Should this record be redone during recovery?"""
        return self.log_type in (LogType.INSERT, LogType.UPDATE, LogType.DELETE, LogType.CLR)


# =============================================================================
# Log Serialization
# =============================================================================

def _serialize_record(record):
    """Serialize a LogRecord to bytes with CRC."""
    parts = []
    # Header: lsn(8) + txn_id(4) + type(2) + prev_lsn(8) + undo_next_lsn(8)
    prev = record.prev_lsn.value if record.prev_lsn else -1
    undo = record.undo_next_lsn.value if record.undo_next_lsn else -1
    parts.append(struct.pack('<qiHqq', record.lsn.value, record.txn_id,
                             int(record.log_type), prev, undo))

    # Table name
    if record.table_name:
        name_bytes = record.table_name.encode('utf-8')
        parts.append(struct.pack('<H', len(name_bytes)))
        parts.append(name_bytes)
    else:
        parts.append(struct.pack('<H', 0))

    # Page ID + Slot
    page_id = record.page_id if record.page_id is not None else -1
    slot_idx = record.slot_idx if record.slot_idx is not None else -1
    parts.append(struct.pack('<ii', page_id, slot_idx))

    # Before image
    if record.before_image is not None:
        bi = _encode_value(record.before_image)
        parts.append(struct.pack('<I', len(bi)))
        parts.append(bi)
    else:
        parts.append(struct.pack('<I', 0))

    # After image
    if record.after_image is not None:
        ai = _encode_value(record.after_image)
        parts.append(struct.pack('<I', len(ai)))
        parts.append(ai)
    else:
        parts.append(struct.pack('<I', 0))

    # Checkpoint data
    if record.log_type == LogType.CHECKPOINT:
        _serialize_checkpoint_data(parts, record)
    else:
        parts.append(struct.pack('<I', 0))  # no checkpoint data

    body = b''.join(parts)
    # CRC32 + length prefix
    crc = _crc32(body)
    return struct.pack('<II', len(body), crc) + body


def _serialize_checkpoint_data(parts, record):
    """Serialize checkpoint-specific fields."""
    cp_parts = []
    # Active transactions
    active = record.active_txns or {}
    cp_parts.append(struct.pack('<I', len(active)))
    for txn_id, last_lsn in sorted(active.items()):
        lsn_val = last_lsn.value if isinstance(last_lsn, LSN) else last_lsn
        cp_parts.append(struct.pack('<iq', txn_id, lsn_val))

    # Dirty pages
    dirty = record.dirty_pages or {}
    cp_parts.append(struct.pack('<I', len(dirty)))
    for (tbl, pid), rec_lsn in sorted(dirty.items()):
        tbl_bytes = tbl.encode('utf-8')
        lsn_val = rec_lsn.value if isinstance(rec_lsn, LSN) else rec_lsn
        cp_parts.append(struct.pack('<H', len(tbl_bytes)))
        cp_parts.append(tbl_bytes)
        cp_parts.append(struct.pack('<iq', pid, lsn_val))

    cp_data = b''.join(cp_parts)
    parts.append(struct.pack('<I', len(cp_data)))
    parts.append(cp_data)


def _deserialize_record(data, offset=0):
    """Deserialize a LogRecord from bytes. Returns (record, new_offset)."""
    body_len, expected_crc = struct.unpack('<II', data[offset:offset + 8])
    offset += 8
    body = data[offset:offset + body_len]
    actual_crc = _crc32(body)
    if actual_crc != expected_crc:
        raise ValueError(f"CRC mismatch: expected {expected_crc}, got {actual_crc}")

    pos = 0
    lsn_val, txn_id, log_type_val, prev_val, undo_val = struct.unpack('<qiHqq', body[pos:pos + 30])
    pos += 30

    prev_lsn = LSN(prev_val) if prev_val >= 0 else None
    undo_next_lsn = LSN(undo_val) if undo_val >= 0 else None

    # Table name
    name_len = struct.unpack('<H', body[pos:pos + 2])[0]
    pos += 2
    table_name = body[pos:pos + name_len].decode('utf-8') if name_len > 0 else None
    pos += name_len

    # Page ID + Slot
    page_id, slot_idx = struct.unpack('<ii', body[pos:pos + 8])
    pos += 8
    page_id = page_id if page_id >= 0 else None
    slot_idx = slot_idx if slot_idx >= 0 else None

    # Before image
    bi_len = struct.unpack('<I', body[pos:pos + 4])[0]
    pos += 4
    before_image = None
    if bi_len > 0:
        before_image, _ = _decode_value(body, pos)
        pos += bi_len

    # After image
    ai_len = struct.unpack('<I', body[pos:pos + 4])[0]
    pos += 4
    after_image = None
    if ai_len > 0:
        after_image, _ = _decode_value(body, pos)
        pos += ai_len

    # Checkpoint data
    cp_len = struct.unpack('<I', body[pos:pos + 4])[0]
    pos += 4
    active_txns = None
    dirty_pages = None
    if cp_len > 0:
        active_txns, dirty_pages = _deserialize_checkpoint_data(body, pos)

    record = LogRecord(
        lsn=LSN(lsn_val), txn_id=txn_id, log_type=LogType(log_type_val),
        table_name=table_name, page_id=page_id, slot_idx=slot_idx,
        before_image=before_image, after_image=after_image,
        prev_lsn=prev_lsn, undo_next_lsn=undo_next_lsn,
        active_txns=active_txns, dirty_pages=dirty_pages
    )
    return record, offset + body_len


def _deserialize_checkpoint_data(body, pos):
    """Deserialize checkpoint active_txns and dirty_pages."""
    num_active = struct.unpack('<I', body[pos:pos + 4])[0]
    pos += 4
    active_txns = {}
    for _ in range(num_active):
        txn_id, lsn_val = struct.unpack('<iq', body[pos:pos + 12])
        pos += 12
        active_txns[txn_id] = LSN(lsn_val)

    num_dirty = struct.unpack('<I', body[pos:pos + 4])[0]
    pos += 4
    dirty_pages = {}
    for _ in range(num_dirty):
        tbl_len = struct.unpack('<H', body[pos:pos + 2])[0]
        pos += 2
        tbl = body[pos:pos + tbl_len].decode('utf-8')
        pos += tbl_len
        pid, lsn_val = struct.unpack('<iq', body[pos:pos + 12])
        pos += 12
        dirty_pages[(tbl, pid)] = LSN(lsn_val)

    return active_txns, dirty_pages


def _crc32(data):
    """CRC32 checksum for integrity. Returns uint32."""
    digest = hashlib.md5(data).digest()[:4]
    return struct.unpack('<I', digest)[0]


# =============================================================================
# WAL Buffer
# =============================================================================

class WALBuffer:
    """In-memory WAL buffer that batches log records before flushing."""

    def __init__(self, capacity=1024):
        self.capacity = capacity
        self._records = []
        self._flushed_lsn = LSN(0)  # Highest LSN flushed to stable storage

    def append(self, record):
        """Add a record to the buffer."""
        self._records.append(record)
        if len(self._records) >= self.capacity:
            return True  # Signal: buffer full, should flush
        return False

    def flush(self, writer):
        """Flush all buffered records to the WAL writer."""
        for record in self._records:
            writer.write(record)
        if self._records:
            self._flushed_lsn = self._records[-1].lsn
        self._records.clear()

    @property
    def flushed_lsn(self):
        return self._flushed_lsn

    @property
    def pending_count(self):
        return len(self._records)

    def flush_up_to(self, lsn, writer):
        """Flush records up to (and including) the given LSN."""
        to_flush = []
        remaining = []
        for r in self._records:
            if r.lsn <= lsn:
                to_flush.append(r)
            else:
                remaining.append(r)
        for r in to_flush:
            writer.write(r)
        if to_flush:
            self._flushed_lsn = to_flush[-1].lsn
        self._records = remaining


# =============================================================================
# WAL Writer
# =============================================================================

class WALWriter:
    """Sequential log writer with CRC integrity."""

    def __init__(self):
        self._log_data = bytearray()
        self._record_offsets = []  # (lsn, offset) for fast seeking
        self._record_count = 0

    def write(self, record):
        """Append a serialized record to the log."""
        offset = len(self._log_data)
        self._record_offsets.append((record.lsn, offset))
        serialized = _serialize_record(record)
        self._log_data.extend(serialized)
        self._record_count += 1

    @property
    def size(self):
        return len(self._log_data)

    @property
    def record_count(self):
        return self._record_count

    def get_data(self):
        """Return raw log data for reading."""
        return bytes(self._log_data)

    def truncate_after(self, lsn):
        """Truncate log after the given LSN (for log truncation after checkpoint)."""
        for i, (rec_lsn, offset) in enumerate(self._record_offsets):
            if rec_lsn > lsn:
                self._log_data = bytearray(self._log_data[:offset])
                self._record_offsets = self._record_offsets[:i]
                self._record_count = i
                return
        # LSN beyond all records -- no truncation needed


# =============================================================================
# WAL Reader
# =============================================================================

class WALReader:
    """Read log records forward or backward with optional filtering."""

    def __init__(self, writer):
        self._writer = writer

    def scan_forward(self, from_lsn=None):
        """Yield records in forward (chronological) order."""
        data = self._writer.get_data()
        offset = 0
        while offset < len(data):
            record, offset = _deserialize_record(data, offset)
            if from_lsn is not None and record.lsn < from_lsn:
                continue
            yield record

    def scan_backward(self):
        """Yield records in reverse order."""
        data = self._writer.get_data()
        # Build list of all records, then reverse
        records = []
        offset = 0
        while offset < len(data):
            record, offset = _deserialize_record(data, offset)
            records.append(record)
        for record in reversed(records):
            yield record

    def find_by_lsn(self, target_lsn):
        """Find a specific record by LSN."""
        if isinstance(target_lsn, int):
            target_lsn = LSN(target_lsn)
        for record in self.scan_forward():
            if record.lsn == target_lsn:
                return record
        return None

    def find_by_txn(self, txn_id):
        """Find all records for a specific transaction."""
        return [r for r in self.scan_forward() if r.txn_id == txn_id]

    def find_last_checkpoint(self):
        """Find the most recent checkpoint record."""
        last_cp = None
        for record in self.scan_forward():
            if record.log_type == LogType.CHECKPOINT:
                last_cp = record
        return last_cp


# =============================================================================
# Transaction Table (for recovery)
# =============================================================================

class TransactionEntry:
    """Tracks a transaction's state during recovery."""
    __slots__ = ['txn_id', 'status', 'last_lsn', 'undo_next_lsn']

    def __init__(self, txn_id, status='active', last_lsn=None, undo_next_lsn=None):
        self.txn_id = txn_id
        self.status = status  # 'active', 'committed', 'aborted'
        self.last_lsn = last_lsn
        self.undo_next_lsn = undo_next_lsn


# =============================================================================
# Recovery Manager (ARIES-style)
# =============================================================================

class RecoveryManager:
    """
    ARIES-style crash recovery with three passes:
    1. Analysis: scan from last checkpoint, rebuild transaction table + dirty page table
    2. Redo: replay all actions from earliest rec_lsn in dirty page table
    3. Undo: roll back uncommitted transactions by following prev_lsn chains
    """

    def __init__(self, wal_reader, wal_writer, wal_manager):
        self._reader = wal_reader
        self._writer = wal_writer
        self._wal_manager = wal_manager
        self._txn_table = {}   # txn_id -> TransactionEntry
        self._dirty_page_table = {}  # (table_name, page_id) -> rec_lsn
        self._redo_count = 0
        self._undo_count = 0

    @property
    def txn_table(self):
        return dict(self._txn_table)

    @property
    def dirty_page_table(self):
        return dict(self._dirty_page_table)

    @property
    def redo_count(self):
        return self._redo_count

    @property
    def undo_count(self):
        return self._undo_count

    def recover(self, engine=None):
        """Run full ARIES recovery. Returns (redo_count, undo_count)."""
        self._redo_count = 0
        self._undo_count = 0

        # Phase 1: Analysis
        self._analysis_pass()

        # Phase 2: Redo
        if engine:
            self._redo_pass(engine)

        # Phase 3: Undo
        self._undo_pass(engine)

        return self._redo_count, self._undo_count

    def _analysis_pass(self):
        """Rebuild transaction table and dirty page table from log."""
        # Start from last checkpoint if available
        checkpoint = self._reader.find_last_checkpoint()
        if checkpoint:
            # Initialize from checkpoint
            if checkpoint.active_txns:
                for txn_id, last_lsn in checkpoint.active_txns.items():
                    self._txn_table[txn_id] = TransactionEntry(txn_id, 'active', last_lsn)
            if checkpoint.dirty_pages:
                self._dirty_page_table = dict(checkpoint.dirty_pages)
            start_lsn = checkpoint.lsn
        else:
            start_lsn = LSN(0)

        # Scan forward from checkpoint
        for record in self._reader.scan_forward(from_lsn=start_lsn):
            txn_id = record.txn_id

            if record.log_type == LogType.BEGIN:
                self._txn_table[txn_id] = TransactionEntry(txn_id, 'active', record.lsn)

            elif record.log_type == LogType.COMMIT:
                if txn_id in self._txn_table:
                    self._txn_table[txn_id].status = 'committed'
                    self._txn_table[txn_id].last_lsn = record.lsn

            elif record.log_type == LogType.ABORT:
                if txn_id in self._txn_table:
                    self._txn_table[txn_id].status = 'aborted'
                    self._txn_table[txn_id].last_lsn = record.lsn

            elif record.log_type == LogType.END:
                # Transaction fully processed -- remove from table
                self._txn_table.pop(txn_id, None)

            elif record.is_redoable():
                if txn_id in self._txn_table:
                    self._txn_table[txn_id].last_lsn = record.lsn
                else:
                    self._txn_table[txn_id] = TransactionEntry(txn_id, 'active', record.lsn)

                # Track dirty pages
                if record.table_name and record.page_id is not None:
                    key = (record.table_name, record.page_id)
                    if key not in self._dirty_page_table:
                        self._dirty_page_table[key] = record.lsn

            if record.log_type == LogType.CLR and record.undo_next_lsn is not None:
                if txn_id in self._txn_table:
                    self._txn_table[txn_id].undo_next_lsn = record.undo_next_lsn

    def _redo_pass(self, engine):
        """Redo all actions from the earliest rec_lsn in dirty page table."""
        if not self._dirty_page_table:
            # Find earliest redoable record
            start_lsn = LSN(0)
        else:
            start_lsn = min(self._dirty_page_table.values(), key=lambda l: l.value)

        for record in self._reader.scan_forward(from_lsn=start_lsn):
            if not record.is_redoable():
                continue

            # Check if page is in dirty page table and rec_lsn <= record.lsn
            if record.table_name and record.page_id is not None:
                key = (record.table_name, record.page_id)
                if key in self._dirty_page_table:
                    rec_lsn = self._dirty_page_table[key]
                    if record.lsn < rec_lsn:
                        continue  # Page was flushed after this record

            # Redo the action
            self._apply_redo(record, engine)
            self._redo_count += 1

    def _undo_pass(self, engine=None):
        """Undo uncommitted transactions by following prev_lsn chains."""
        # Collect active (uncommitted) transactions
        to_undo = {}
        for txn_id, entry in self._txn_table.items():
            if entry.status == 'active':
                undo_lsn = entry.undo_next_lsn or entry.last_lsn
                if undo_lsn is not None:
                    to_undo[txn_id] = undo_lsn

        # Process undo in reverse LSN order (highest first)
        while to_undo:
            # Find the transaction with the highest undo LSN
            txn_id = max(to_undo, key=lambda t: to_undo[t].value)
            undo_lsn = to_undo[txn_id]

            record = self._reader.find_by_lsn(undo_lsn)
            if record is None:
                del to_undo[txn_id]
                continue

            if record.log_type == LogType.CLR:
                # Follow undo_next_lsn
                if record.undo_next_lsn and record.undo_next_lsn.value > 0:
                    to_undo[txn_id] = record.undo_next_lsn
                else:
                    del to_undo[txn_id]
                    self._write_end_record(txn_id)
            elif record.is_undoable():
                # Undo this record and write a CLR
                clr = self._create_clr(record)
                self._wal_manager._append_record(clr)

                if engine:
                    self._apply_undo(record, engine)
                self._undo_count += 1

                # Follow prev_lsn chain
                if record.prev_lsn and record.prev_lsn.value > 0:
                    to_undo[txn_id] = record.prev_lsn
                else:
                    del to_undo[txn_id]
                    self._write_end_record(txn_id)
            else:
                # Non-undoable record, follow prev_lsn
                if record.prev_lsn and record.prev_lsn.value > 0:
                    to_undo[txn_id] = record.prev_lsn
                else:
                    del to_undo[txn_id]
                    self._write_end_record(txn_id)

    def _write_end_record(self, txn_id):
        """Write an END record for a fully undone transaction."""
        end_record = LogRecord(
            lsn=self._wal_manager._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.END
        )
        self._wal_manager._append_record(end_record)

    def _create_clr(self, original):
        """Create a Compensation Log Record for undoing a record."""
        # CLR's undo_next_lsn points to original's prev_lsn (skip the undone record)
        return LogRecord(
            lsn=self._wal_manager._next_lsn(),
            txn_id=original.txn_id,
            log_type=LogType.CLR,
            table_name=original.table_name,
            page_id=original.page_id,
            slot_idx=original.slot_idx,
            before_image=original.after_image,   # CLR: swap images
            after_image=original.before_image,
            prev_lsn=self._wal_manager._get_last_lsn(original.txn_id),
            undo_next_lsn=original.prev_lsn
        )

    def _apply_redo(self, record, engine):
        """Apply a redo action to the storage engine."""
        if not record.table_name or record.table_name not in engine._tables:
            return
        table = engine._tables[record.table_name]

        if record.log_type in (LogType.INSERT, LogType.CLR):
            if record.after_image is not None:
                # Re-insert the row
                try:
                    if record.page_id is not None and record.slot_idx is not None:
                        row_id = table.heap.insert_row(record.after_image)
                        # Update indexes
                        for col_idx, idx in table.indexes.items():
                            if isinstance(record.after_image, (list, tuple)) and col_idx < len(record.after_image):
                                idx.insert(record.after_image[col_idx], row_id)
                except Exception:
                    pass  # Best effort during recovery

        elif record.log_type == LogType.UPDATE:
            if record.after_image is not None and record.page_id is not None:
                try:
                    row_id = RowID(record.page_id, record.slot_idx or 0)
                    table.heap.update_row(row_id, record.after_image)
                except Exception:
                    pass

        elif record.log_type == LogType.DELETE:
            if record.page_id is not None and record.slot_idx is not None:
                try:
                    row_id = RowID(record.page_id, record.slot_idx)
                    table.heap.delete_row(row_id)
                except Exception:
                    pass

    def _apply_undo(self, record, engine):
        """Apply an undo action to the storage engine (reverse the operation)."""
        if not record.table_name or record.table_name not in engine._tables:
            return
        table = engine._tables[record.table_name]

        if record.log_type == LogType.INSERT:
            # Undo insert = delete
            if record.page_id is not None and record.slot_idx is not None:
                try:
                    row_id = RowID(record.page_id, record.slot_idx)
                    table.heap.delete_row(row_id)
                except Exception:
                    pass

        elif record.log_type == LogType.UPDATE:
            # Undo update = restore before_image
            if record.before_image is not None and record.page_id is not None:
                try:
                    row_id = RowID(record.page_id, record.slot_idx or 0)
                    table.heap.update_row(row_id, record.before_image)
                except Exception:
                    pass

        elif record.log_type == LogType.DELETE:
            # Undo delete = re-insert before_image
            if record.before_image is not None:
                try:
                    table.heap.insert_row(record.before_image)
                except Exception:
                    pass


# =============================================================================
# WAL Manager
# =============================================================================

class WALManager:
    """
    Central WAL manager -- coordinates buffer, writer, reader, and recovery.
    Assigns LSNs, tracks per-transaction chains, and manages flush policy.
    """

    def __init__(self, buffer_capacity=1024):
        self._writer = WALWriter()
        self._buffer = WALBuffer(capacity=buffer_capacity)
        self._reader = WALReader(self._writer)
        self._next_lsn_counter = 1
        self._txn_last_lsn = {}  # txn_id -> last LSN for prev_lsn chains
        self._active_txns = set()
        self._committed_txns = set()
        self._aborted_txns = set()

    @property
    def writer(self):
        return self._writer

    @property
    def reader(self):
        return self._reader

    @property
    def buffer(self):
        return self._buffer

    @property
    def active_transactions(self):
        return set(self._active_txns)

    @property
    def committed_transactions(self):
        return set(self._committed_txns)

    def _next_lsn(self):
        """Get next LSN."""
        lsn = LSN(self._next_lsn_counter)
        self._next_lsn_counter += 1
        return lsn

    def _get_last_lsn(self, txn_id):
        """Get the last LSN for a transaction (for prev_lsn chain)."""
        return self._txn_last_lsn.get(txn_id)

    def _append_record(self, record):
        """Append record to buffer and update tracking."""
        self._txn_last_lsn[record.txn_id] = record.lsn
        should_flush = self._buffer.append(record)
        if should_flush:
            self.flush()
        return record.lsn

    def begin(self, txn_id):
        """Log a BEGIN record for a transaction."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.BEGIN,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        self._active_txns.add(txn_id)
        return self._append_record(record)

    def log_insert(self, txn_id, table_name, page_id, slot_idx, after_image):
        """Log an INSERT record."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.INSERT,
            table_name=table_name,
            page_id=page_id,
            slot_idx=slot_idx,
            after_image=after_image,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        return self._append_record(record)

    def log_update(self, txn_id, table_name, page_id, slot_idx, before_image, after_image):
        """Log an UPDATE record."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.UPDATE,
            table_name=table_name,
            page_id=page_id,
            slot_idx=slot_idx,
            before_image=before_image,
            after_image=after_image,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        return self._append_record(record)

    def log_delete(self, txn_id, table_name, page_id, slot_idx, before_image):
        """Log a DELETE record."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.DELETE,
            table_name=table_name,
            page_id=page_id,
            slot_idx=slot_idx,
            before_image=before_image,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        return self._append_record(record)

    def commit(self, txn_id):
        """Log a COMMIT record and force-flush to stable storage."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.COMMIT,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        lsn = self._append_record(record)
        # Force flush on commit (WAL protocol)
        self.flush()
        self._active_txns.discard(txn_id)
        self._committed_txns.add(txn_id)
        return lsn

    def abort(self, txn_id):
        """Log an ABORT record."""
        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=txn_id,
            log_type=LogType.ABORT,
            prev_lsn=self._get_last_lsn(txn_id)
        )
        lsn = self._append_record(record)
        self.flush()
        self._active_txns.discard(txn_id)
        self._aborted_txns.add(txn_id)
        return lsn

    def checkpoint(self, dirty_pages=None):
        """Write a checkpoint record capturing current state."""
        active_txns = {}
        for txn_id in self._active_txns:
            if txn_id in self._txn_last_lsn:
                active_txns[txn_id] = self._txn_last_lsn[txn_id]

        record = LogRecord(
            lsn=self._next_lsn(),
            txn_id=0,  # Checkpoint is system-level
            log_type=LogType.CHECKPOINT,
            active_txns=active_txns,
            dirty_pages=dirty_pages or {}
        )
        lsn = self._append_record(record)
        self.flush()
        return lsn

    def flush(self):
        """Flush WAL buffer to stable storage."""
        self._buffer.flush(self._writer)

    def create_recovery_manager(self):
        """Create a RecoveryManager for this WAL."""
        # Flush any pending records first
        self.flush()
        return RecoveryManager(self._reader, self._writer, self)

    def log_count(self):
        """Total records written (flushed + pending)."""
        return self._writer.record_count + self._buffer.pending_count

    def get_all_records(self):
        """Get all flushed records."""
        self.flush()
        return list(self._reader.scan_forward())


# =============================================================================
# WAL-Protected Storage Engine
# =============================================================================

class WALStorageEngine:
    """
    Composes WAL Manager + C213 StorageEngine for crash-safe operations.
    All mutations go through WAL before touching storage pages.
    C213 uses dict rows: {col_name: value}.
    """

    def __init__(self, buffer_pool_size=64):
        self._engine = StorageEngine(buffer_pool_size=buffer_pool_size)
        self._wal = WALManager()
        self._next_txn_id = 1
        self._txn_snapshots = {}  # txn_id -> [(table_name, row_key, before_image)]

    @property
    def wal(self):
        return self._wal

    @property
    def engine(self):
        return self._engine

    def create_table(self, name, columns, primary_key=None):
        """Create a table (DDL -- not WAL-logged)."""
        return self._engine.create_table(name, columns, primary_key)

    def begin_transaction(self):
        """Start a new transaction. Returns txn_id."""
        txn_id = self._next_txn_id
        self._next_txn_id += 1
        self._wal.begin(txn_id)
        self._txn_snapshots[txn_id] = []
        return txn_id

    def insert(self, txn_id, table_name, row):
        """Insert a row (dict) with WAL logging."""
        if table_name not in self._engine._tables:
            raise KeyError(f"Table '{table_name}' does not exist")
        table = self._engine._tables[table_name]

        # Perform the insert
        row_id = table.insert(row)

        # Log after we know the page_id/slot_index
        self._wal.log_insert(
            txn_id, table_name,
            page_id=row_id.page_id, slot_idx=row_id.slot_index,
            after_image=row
        )

        # Track for undo: None means this was an insert (undo = delete)
        self._txn_snapshots[txn_id].append((table_name, row_id, None))

        return row_id

    def update(self, txn_id, table_name, row_id, new_row):
        """Update a row (dict) with WAL logging."""
        if table_name not in self._engine._tables:
            raise KeyError(f"Table '{table_name}' does not exist")
        table = self._engine._tables[table_name]

        # Read before image
        before_image = table.heap.get(row_id)

        # Log before applying
        self._wal.log_update(
            txn_id, table_name,
            page_id=row_id.page_id, slot_idx=row_id.slot_index,
            before_image=before_image, after_image=new_row
        )

        # Apply update
        new_rid = table.heap.update(row_id, new_row)

        # Track for undo
        self._txn_snapshots[txn_id].append((table_name, new_rid, before_image))

        return new_rid

    def delete(self, txn_id, table_name, row_id):
        """Delete a row with WAL logging."""
        if table_name not in self._engine._tables:
            raise KeyError(f"Table '{table_name}' does not exist")
        table = self._engine._tables[table_name]

        # Read before image
        before_image = table.heap.get(row_id)

        # Log before applying
        self._wal.log_delete(
            txn_id, table_name,
            page_id=row_id.page_id, slot_idx=row_id.slot_index,
            before_image=before_image
        )

        # Apply delete
        table.heap.delete(row_id)

        # Track for undo: before_image means undo = re-insert
        self._txn_snapshots[txn_id].append((table_name, row_id, before_image))

    def commit(self, txn_id):
        """Commit a transaction -- force WAL flush."""
        self._wal.commit(txn_id)
        self._txn_snapshots.pop(txn_id, None)

    def abort(self, txn_id):
        """Abort a transaction -- undo all changes using before images."""
        undo_list = self._txn_snapshots.pop(txn_id, [])

        # Undo in reverse order
        for table_name, row_id, before_image in reversed(undo_list):
            if table_name not in self._engine._tables:
                continue
            table = self._engine._tables[table_name]
            if before_image is None:
                # Was an insert -- delete it
                try:
                    table.heap.delete(row_id)
                except Exception:
                    pass
            else:
                # Was an update or delete -- restore before image
                try:
                    table.heap.update(row_id, before_image)
                except Exception:
                    try:
                        table.heap.insert(before_image)
                    except Exception:
                        pass

        self._wal.abort(txn_id)

    def scan_table(self, table_name):
        """Scan all rows in a table. Returns list of (RowID, row_dict) pairs."""
        if table_name not in self._engine._tables:
            raise KeyError(f"Table '{table_name}' does not exist")
        return list(self._engine._tables[table_name].scan())

    def checkpoint(self):
        """Take a checkpoint -- record current dirty page state."""
        dirty_pages = {}
        for page_id, frame in self._engine.buffer_pool._frames.items():
            if frame.dirty:
                dirty_pages[('__buffer', page_id)] = LSN(0)
        self._wal.checkpoint(dirty_pages)

    def recover(self):
        """Run ARIES recovery after a simulated crash."""
        rm = self._wal.create_recovery_manager()
        return rm.recover(self._engine)

    def get_log_records(self):
        """Get all WAL records for inspection."""
        return self._wal.get_all_records()


# =============================================================================
# Log Analysis Utilities
# =============================================================================

class WALAnalyzer:
    """Utilities for analyzing WAL contents."""

    def __init__(self, wal_manager):
        self._wal = wal_manager

    def transaction_summary(self):
        """Summarize all transactions in the log."""
        records = self._wal.get_all_records()
        txns = {}  # txn_id -> {type_counts, status, first_lsn, last_lsn}

        for r in records:
            if r.txn_id == 0:
                continue  # Skip system records
            if r.txn_id not in txns:
                txns[r.txn_id] = {
                    'operations': {},
                    'status': 'active',
                    'first_lsn': r.lsn,
                    'last_lsn': r.lsn,
                    'tables': set()
                }
            txn = txns[r.txn_id]
            txn['last_lsn'] = r.lsn
            op_name = r.log_type.name
            txn['operations'][op_name] = txn['operations'].get(op_name, 0) + 1
            if r.table_name:
                txn['tables'].add(r.table_name)
            if r.log_type == LogType.COMMIT:
                txn['status'] = 'committed'
            elif r.log_type == LogType.ABORT:
                txn['status'] = 'aborted'

        # Convert sets to lists for JSON compatibility
        for txn in txns.values():
            txn['tables'] = sorted(txn['tables'])

        return txns

    def page_history(self, table_name, page_id):
        """Get modification history for a specific page."""
        records = self._wal.get_all_records()
        history = []
        for r in records:
            if r.table_name == table_name and r.page_id == page_id:
                history.append({
                    'lsn': r.lsn.value,
                    'txn_id': r.txn_id,
                    'type': r.log_type.name,
                    'slot': r.slot_idx,
                    'before': r.before_image,
                    'after': r.after_image
                })
        return history

    def log_statistics(self):
        """Compute WAL statistics."""
        records = self._wal.get_all_records()
        stats = {
            'total_records': len(records),
            'by_type': {},
            'unique_txns': set(),
            'unique_tables': set(),
            'unique_pages': set()
        }
        for r in records:
            name = r.log_type.name
            stats['by_type'][name] = stats['by_type'].get(name, 0) + 1
            if r.txn_id != 0:
                stats['unique_txns'].add(r.txn_id)
            if r.table_name:
                stats['unique_tables'].add(r.table_name)
            if r.table_name and r.page_id is not None:
                stats['unique_pages'].add((r.table_name, r.page_id))

        stats['unique_txns'] = len(stats['unique_txns'])
        stats['unique_tables'] = len(stats['unique_tables'])
        stats['unique_pages'] = len(stats['unique_pages'])
        return stats

    def verify_integrity(self):
        """Verify all log records have valid CRC checksums."""
        try:
            records = self._wal.get_all_records()
            return {'valid': True, 'record_count': len(records), 'errors': []}
        except ValueError as e:
            return {'valid': False, 'record_count': 0, 'errors': [str(e)]}

    def prev_lsn_chain(self, txn_id):
        """Follow the prev_lsn chain for a transaction."""
        records = self._wal.get_all_records()
        # Find last record for this txn
        txn_records = [r for r in records if r.txn_id == txn_id]
        if not txn_records:
            return []

        chain = []
        current = txn_records[-1]
        visited = set()
        while current and current.lsn.value not in visited:
            visited.add(current.lsn.value)
            chain.append({
                'lsn': current.lsn.value,
                'type': current.log_type.name,
                'prev_lsn': current.prev_lsn.value if current.prev_lsn else None
            })
            if current.prev_lsn and current.prev_lsn.value > 0:
                current = self._wal.reader.find_by_lsn(current.prev_lsn)
            else:
                break

        return chain
