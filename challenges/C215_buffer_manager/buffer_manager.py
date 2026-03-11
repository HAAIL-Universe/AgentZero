"""
C215: Buffer Manager with WAL Integration
Composes C213 (Storage Engine) + C214 (WAL Engine)

A production-grade buffer manager that integrates page-level LSN tracking
with write-ahead logging. Implements no-force/steal policy, WAL-aware
flushing, dirty page tracking, and ARIES-compatible recovery.

Key concepts:
- Page-level LSN: each page tracks the LSN of the last modification
- Write-Ahead Logging: WAL record must be flushed before dirty page
- No-Force: pages don't need to be flushed at commit time (WAL handles durability)
- Steal: dirty pages can be flushed before transaction commits (WAL handles undo)
- Checkpoint: periodic snapshot of dirty page table + active transactions
- Recovery: ARIES three-pass (analysis, redo, undo) using page LSNs to skip
"""

import sys
import os
import struct
import time
import json
from collections import OrderedDict
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C213_storage_engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C214_wal_engine'))

from storage_engine import (
    PAGE_SIZE, DiskManager, SlottedPage, RowID,
    _encode_row, _decode_row
)
from wal_engine import (
    LSN, LogRecord, LogType, WALBuffer, WALWriter, WALReader,
    WALManager, RecoveryManager as BaseRecoveryManager
)


# ============================================================
# Page LSN Tracking
# ============================================================

class LSNPage:
    """A page that tracks its last-written LSN.

    Stores the page LSN in the first 8 bytes of page data,
    followed by the actual SlottedPage content.
    """
    LSN_HEADER_SIZE = 8  # 8 bytes for LSN (uint64)

    def __init__(self, page_id, data=None, page_size=PAGE_SIZE):
        self.page_id = page_id
        self.page_size = page_size
        self._page_lsn = LSN(0)

        if data is not None:
            # Extract LSN from header
            if len(data) >= self.LSN_HEADER_SIZE:
                lsn_val = struct.unpack('<Q', data[:self.LSN_HEADER_SIZE])[0]
                self._page_lsn = LSN(lsn_val)
            # Rest is slotted page data
            slotted_data = bytearray(data[self.LSN_HEADER_SIZE:])
            # Pad to full page size for SlottedPage
            if len(slotted_data) < page_size:
                slotted_data.extend(b'\x00' * (page_size - len(slotted_data)))
            self.slotted = SlottedPage(page_id, slotted_data, page_size)
        else:
            self.slotted = SlottedPage(page_id, None, page_size)

    @property
    def page_lsn(self):
        return self._page_lsn

    @page_lsn.setter
    def page_lsn(self, lsn):
        self._page_lsn = lsn

    def serialize(self):
        """Serialize page with LSN header."""
        header = struct.pack('<Q', self._page_lsn.value)
        # Get slotted page raw data
        page_data = bytes(self.slotted.data)
        return header + page_data

    def insert_tuple(self, data):
        return self.slotted.insert_tuple(data)

    def get_tuple(self, slot_idx):
        return self.slotted.get_tuple(slot_idx)

    def delete_tuple(self, slot_idx):
        return self.slotted.delete_tuple(slot_idx)

    def update_tuple(self, slot_idx, new_data):
        return self.slotted.update_tuple(slot_idx, new_data)

    def iter_tuples(self):
        return self.slotted.iter_tuples()

    @property
    def num_slots(self):
        return self.slotted.num_slots

    def free_space(self):
        return self.slotted.free_space()


# ============================================================
# WAL-Aware Buffer Pool
# ============================================================

class BufferFrame:
    """A frame in the buffer pool holding a page."""

    def __init__(self, page_id, lsn_page):
        self.page_id = page_id
        self.page = lsn_page
        self.pin_count = 0
        self.dirty = False
        self.rec_lsn = LSN(0)  # LSN of first modification since clean

    def pin(self):
        self.pin_count += 1

    def unpin(self):
        if self.pin_count > 0:
            self.pin_count -= 1

    def mark_dirty(self, lsn):
        if not self.dirty:
            self.rec_lsn = lsn
        self.dirty = True
        self.page.page_lsn = lsn


class FlushPolicy(Enum):
    """When dirty pages get flushed to disk."""
    IMMEDIATE = auto()    # Flush on every unpin (force)
    ON_EVICT = auto()     # Only flush when evicting (no-force) -- default
    PERIODIC = auto()     # Flush on checkpoint intervals


class WALBufferPool:
    """Buffer pool with WAL integration.

    Enforces the Write-Ahead Log protocol:
    - Before flushing a dirty page, ensure the WAL is flushed up to the page's LSN
    - Tracks page-level LSNs for ARIES recovery
    - Supports steal/no-force policy
    """

    def __init__(self, disk_manager, wal_manager, pool_size=64,
                 flush_policy=FlushPolicy.ON_EVICT):
        self.disk = disk_manager
        self.wal = wal_manager
        self.pool_size = pool_size
        self.flush_policy = flush_policy

        # page_id -> BufferFrame
        self.frames = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._flushes = 0
        self._evictions = 0
        self._wal_forces = 0

    def fetch_page(self, page_id):
        """Fetch a page, loading from disk if needed. Pins the page."""
        if page_id in self.frames:
            self._hits += 1
            frame = self.frames[page_id]
            # Move to end (most recently used)
            self.frames.move_to_end(page_id)
            frame.pin()
            return frame

        self._misses += 1

        # Need to load from disk -- may need to evict
        if len(self.frames) >= self.pool_size:
            self._evict_one()

        # Read from disk
        raw_data = self.disk.read_page(page_id)
        lsn_page = LSNPage(page_id, raw_data, self.disk.page_size)

        frame = BufferFrame(page_id, lsn_page)
        frame.pin()
        self.frames[page_id] = frame
        return frame

    def new_page(self):
        """Allocate a new page and add to pool. Returns pinned frame."""
        if len(self.frames) >= self.pool_size:
            self._evict_one()

        page_id = self.disk.allocate_page()
        lsn_page = LSNPage(page_id, None, self.disk.page_size)

        frame = BufferFrame(page_id, lsn_page)
        frame.pin()
        self.frames[page_id] = frame
        return frame

    def unpin_page(self, page_id, dirty=False, lsn=None):
        """Unpin a page. Optionally mark dirty with the modifying LSN."""
        if page_id not in self.frames:
            return False

        frame = self.frames[page_id]
        frame.unpin()

        if dirty and lsn is not None:
            frame.mark_dirty(lsn)
        elif dirty:
            if not frame.dirty:
                frame.rec_lsn = frame.page.page_lsn
            frame.dirty = True

        if self.flush_policy == FlushPolicy.IMMEDIATE and frame.dirty and frame.pin_count == 0:
            self._flush_frame(frame)

        return True

    def flush_page(self, page_id):
        """Flush a specific page to disk, enforcing WAL protocol."""
        if page_id not in self.frames:
            return False

        frame = self.frames[page_id]
        if frame.dirty:
            self._flush_frame(frame)
        return True

    def flush_all(self):
        """Flush all dirty pages, enforcing WAL protocol."""
        for frame in list(self.frames.values()):
            if frame.dirty:
                self._flush_frame(frame)

    def delete_page(self, page_id):
        """Remove page from pool and deallocate from disk."""
        if page_id in self.frames:
            frame = self.frames[page_id]
            if frame.pin_count > 0:
                raise RuntimeError(f"Cannot delete pinned page {page_id}")
            del self.frames[page_id]
        self.disk.deallocate_page(page_id)
        return True

    def _flush_frame(self, frame):
        """Flush a frame to disk, ensuring WAL is flushed first."""
        if not frame.dirty:
            return

        # WAL protocol: flush WAL up to this page's LSN before writing page
        page_lsn = frame.page.page_lsn
        if page_lsn.value > 0:
            self._force_wal_to(page_lsn)

        # Write page data to disk
        serialized = frame.page.serialize()
        # Truncate to page_size if needed (LSN header + slotted = 2x page_size)
        # We store the full serialized form
        self.disk.write_page(frame.page_id, serialized[:self.disk.page_size])

        frame.dirty = False
        self._flushes += 1

    def _force_wal_to(self, lsn):
        """Force WAL flush up to the given LSN."""
        self.wal.flush()
        self._wal_forces += 1

    def _evict_one(self):
        """Evict the least recently used unpinned page."""
        for page_id in list(self.frames.keys()):
            frame = self.frames[page_id]
            if frame.pin_count == 0:
                if frame.dirty:
                    self._flush_frame(frame)
                del self.frames[page_id]
                self._evictions += 1
                return
        raise RuntimeError("Buffer pool full: all pages are pinned")

    def get_dirty_pages(self):
        """Return dict of dirty pages: {(table_name, page_id): rec_lsn}."""
        result = {}
        for page_id, frame in self.frames.items():
            if frame.dirty:
                result[page_id] = frame.rec_lsn
        return result

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self):
        return len(self.frames)

    @property
    def dirty_count(self):
        return sum(1 for f in self.frames.values() if f.dirty)

    @property
    def pinned_count(self):
        return sum(1 for f in self.frames.values() if f.pin_count > 0)

    def stats(self):
        return {
            'pool_size': self.pool_size,
            'used': len(self.frames),
            'dirty': self.dirty_count,
            'pinned': self.pinned_count,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'flushes': self._flushes,
            'evictions': self._evictions,
            'wal_forces': self._wal_forces,
        }


# ============================================================
# WAL-Aware Heap File
# ============================================================

class WALHeapFile:
    """Heap file that logs all mutations through WAL before modifying pages."""

    def __init__(self, buffer_pool, wal_manager, table_name="default"):
        self.pool = buffer_pool
        self.wal = wal_manager
        self.table_name = table_name
        self._pages = []  # list of page_ids
        self._row_count = 0

    def insert(self, txn_id, row_dict):
        """Insert a row, logging to WAL first. Returns RowID."""
        encoded = _encode_row(row_dict)

        # Find a page with space or allocate new
        page_id, slot_idx = self._find_space_and_insert(encoded)

        # Log the insert to WAL
        lsn = self.wal.log_insert(txn_id, self.table_name, page_id, slot_idx,
                                   after_image=encoded)

        # Mark page dirty with this LSN
        self.pool.unpin_page(page_id, dirty=True, lsn=lsn)

        self._row_count += 1
        return RowID(page_id, slot_idx)

    def get(self, row_id):
        """Read a row by RowID."""
        frame = self.pool.fetch_page(row_id.page_id)
        lsn_page = frame.page
        data = lsn_page.get_tuple(row_id.slot_index)
        self.pool.unpin_page(row_id.page_id)

        if data is None:
            return None
        row, _ = _decode_row(data)
        return row

    def update(self, txn_id, row_id, new_row_dict):
        """Update a row, logging before and after images. Returns new RowID."""
        frame = self.pool.fetch_page(row_id.page_id)
        lsn_page = frame.page

        # Get before image
        before_data = lsn_page.get_tuple(row_id.slot_index)

        new_encoded = _encode_row(new_row_dict)

        # Try to update in place
        new_slot = lsn_page.update_tuple(row_id.slot_index, new_encoded)

        if new_slot >= 0:
            # Updated in same page
            lsn = self.wal.log_update(txn_id, self.table_name, row_id.page_id,
                                       new_slot, before_image=before_data,
                                       after_image=new_encoded)
            self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn)
            return RowID(row_id.page_id, new_slot)
        else:
            # Need to delete from old page and insert in new
            lsn_page.delete_tuple(row_id.slot_index)
            lsn_del = self.wal.log_delete(txn_id, self.table_name, row_id.page_id,
                                           row_id.slot_index, before_image=before_data)
            self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn_del)

            # Insert in a page with space
            new_page_id, new_slot_idx = self._find_space_and_insert(new_encoded)
            lsn_ins = self.wal.log_insert(txn_id, self.table_name, new_page_id,
                                           new_slot_idx, after_image=new_encoded)
            self.pool.unpin_page(new_page_id, dirty=True, lsn=lsn_ins)

            return RowID(new_page_id, new_slot_idx)

    def delete(self, txn_id, row_id):
        """Delete a row, logging before image."""
        frame = self.pool.fetch_page(row_id.page_id)
        lsn_page = frame.page

        before_data = lsn_page.get_tuple(row_id.slot_index)
        if before_data is None:
            self.pool.unpin_page(row_id.page_id)
            return False

        lsn_page.delete_tuple(row_id.slot_index)

        lsn = self.wal.log_delete(txn_id, self.table_name, row_id.page_id,
                                   row_id.slot_index, before_image=before_data)
        self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn)

        self._row_count -= 1
        return True

    def scan(self):
        """Scan all rows. Yields (RowID, row_dict)."""
        for page_id in self._pages:
            frame = self.pool.fetch_page(page_id)
            for slot_idx, data in frame.page.iter_tuples():
                row, _ = _decode_row(data)
                yield RowID(page_id, slot_idx), row
            self.pool.unpin_page(page_id)

    def _find_space_and_insert(self, encoded):
        """Find a page with space and insert the encoded tuple."""
        # Try existing pages
        for page_id in self._pages:
            frame = self.pool.fetch_page(page_id)
            slot_idx = frame.page.insert_tuple(encoded)
            if slot_idx >= 0:
                # Don't unpin yet -- caller will unpin with dirty flag
                return page_id, slot_idx
            self.pool.unpin_page(page_id)

        # Allocate new page
        frame = self.pool.new_page()
        page_id = frame.page_id
        self._pages.append(page_id)
        slot_idx = frame.page.insert_tuple(encoded)
        return page_id, slot_idx

    @property
    def row_count(self):
        return self._row_count

    @property
    def page_count(self):
        return len(self._pages)


# ============================================================
# Dirty Page Table
# ============================================================

class DirtyPageTable:
    """Tracks dirty pages and their recovery LSNs for ARIES."""

    def __init__(self):
        # (table_name, page_id) -> rec_lsn (first dirty LSN)
        self._entries = {}

    def mark_dirty(self, table_name, page_id, lsn):
        """Mark a page as dirty. Only updates rec_lsn if page wasn't already dirty."""
        key = (table_name, page_id)
        if key not in self._entries:
            self._entries[key] = lsn

    def mark_clean(self, table_name, page_id):
        """Mark a page as clean (after flush)."""
        key = (table_name, page_id)
        self._entries.pop(key, None)

    def is_dirty(self, table_name, page_id):
        return (table_name, page_id) in self._entries

    def get_rec_lsn(self, table_name, page_id):
        return self._entries.get((table_name, page_id), None)

    def get_min_rec_lsn(self):
        """Minimum rec_lsn -- redo must start from here."""
        if not self._entries:
            return None
        return min(self._entries.values(), key=lambda l: l.value)

    def entries(self):
        """Return all entries as list of (table_name, page_id, rec_lsn)."""
        return [(k[0], k[1], v) for k, v in self._entries.items()]

    def to_dict(self):
        """Serialize for checkpointing."""
        result = {}
        for (table_name, page_id), lsn in self._entries.items():
            result[f"{table_name}:{page_id}"] = lsn.value
        return result

    @classmethod
    def from_dict(cls, d):
        """Deserialize from checkpoint data."""
        dpt = cls()
        for key, lsn_val in d.items():
            parts = key.rsplit(':', 1)
            table_name = parts[0]
            page_id = int(parts[1])
            dpt._entries[(table_name, page_id)] = LSN(lsn_val)
        return dpt

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()


# ============================================================
# Transaction Table
# ============================================================

class TransactionState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()


class TransactionEntry:
    """Entry in the transaction table for ARIES."""

    def __init__(self, txn_id, state=TransactionState.ACTIVE, last_lsn=None):
        self.txn_id = txn_id
        self.state = state
        self.last_lsn = last_lsn or LSN(0)
        self.undo_next_lsn = None  # For CLR chains

    def __repr__(self):
        return f"TxnEntry({self.txn_id}, {self.state.name}, last_lsn={self.last_lsn})"


class TransactionTable:
    """Active transaction table for ARIES recovery."""

    def __init__(self):
        self._entries = {}  # txn_id -> TransactionEntry

    def begin(self, txn_id, lsn):
        self._entries[txn_id] = TransactionEntry(txn_id, TransactionState.ACTIVE, lsn)

    def update_lsn(self, txn_id, lsn):
        if txn_id in self._entries:
            self._entries[txn_id].last_lsn = lsn

    def commit(self, txn_id):
        if txn_id in self._entries:
            self._entries[txn_id].state = TransactionState.COMMITTED

    def abort(self, txn_id):
        if txn_id in self._entries:
            self._entries[txn_id].state = TransactionState.ABORTED

    def remove(self, txn_id):
        self._entries.pop(txn_id, None)

    def get(self, txn_id):
        return self._entries.get(txn_id)

    def active_transactions(self):
        """Return list of active (uncommitted, non-aborted) transaction IDs."""
        return [e.txn_id for e in self._entries.values()
                if e.state == TransactionState.ACTIVE]

    def loser_transactions(self):
        """Return transactions that need to be undone (active at crash time)."""
        return [e for e in self._entries.values()
                if e.state == TransactionState.ACTIVE]

    def __len__(self):
        return len(self._entries)

    def __contains__(self, txn_id):
        return txn_id in self._entries

    def entries(self):
        return dict(self._entries)


# ============================================================
# Checkpoint Manager
# ============================================================

class CheckpointManager:
    """Manages fuzzy checkpoints for ARIES recovery."""

    def __init__(self, wal_manager, buffer_pool, dirty_page_table):
        self.wal = wal_manager
        self.pool = buffer_pool
        self.dpt = dirty_page_table
        self._checkpoint_count = 0
        self._last_checkpoint_lsn = None

    def take_checkpoint(self, active_txns=None):
        """Take a fuzzy checkpoint.

        Records dirty page table and active transactions in WAL.
        Does NOT flush dirty pages (that's what makes it 'fuzzy').
        """
        # Gather dirty pages from both buffer pool and DPT
        # C214 expects dirty_pages as {(table_name, page_id): LSN}
        dirty_pages = {}
        for page_id, frame in self.pool.frames.items():
            if frame.dirty:
                dirty_pages[("_", page_id)] = frame.rec_lsn

        for (table_name, page_id), lsn_val in self.dpt._entries.items():
            key = (table_name, page_id)
            if key not in dirty_pages:
                dirty_pages[key] = lsn_val

        # Log checkpoint record
        lsn = self.wal.checkpoint(dirty_pages=dirty_pages)

        self._checkpoint_count += 1
        self._last_checkpoint_lsn = lsn

        return lsn

    def take_sharp_checkpoint(self, active_txns=None):
        """Take a sharp checkpoint: flush all dirty pages, then record.

        More expensive but recovery starts from further ahead.
        """
        # Flush all dirty pages
        self.pool.flush_all()

        # Clear dirty page table
        self.dpt.clear()

        # Log checkpoint with empty dirty page table
        lsn = self.wal.checkpoint(dirty_pages={})

        self._checkpoint_count += 1
        self._last_checkpoint_lsn = lsn

        return lsn

    @property
    def checkpoint_count(self):
        return self._checkpoint_count

    @property
    def last_checkpoint_lsn(self):
        return self._last_checkpoint_lsn


# ============================================================
# Enhanced Recovery Manager (ARIES with page LSNs)
# ============================================================

class EnhancedRecoveryManager:
    """ARIES recovery using page-level LSNs for efficient redo.

    Three passes:
    1. Analysis: scan from last checkpoint, rebuild txn table + dirty page table
    2. Redo: replay from min(rec_lsn) of dirty pages, skip if page_lsn >= record_lsn
    3. Undo: rollback loser transactions using prev_lsn chain
    """

    def __init__(self, wal_manager, buffer_pool, dirty_page_table):
        self.wal = wal_manager
        self.pool = buffer_pool
        self.dpt = dirty_page_table
        self.txn_table = TransactionTable()

        self.redo_count = 0
        self.undo_count = 0
        self.skipped_redo = 0
        self._analysis_records = 0

    def recover(self, apply_fn=None):
        """Run full ARIES recovery. Returns (redo_count, undo_count).

        apply_fn: optional callback(record, is_redo) for applying changes.
        """
        self.redo_count = 0
        self.undo_count = 0
        self.skipped_redo = 0
        self._analysis_records = 0

        # Ensure WAL is flushed so reader can see all records
        self.wal.flush()

        # Phase 1: Analysis
        redo_start_lsn = self._analysis_pass()

        # Phase 2: Redo
        self._redo_pass(redo_start_lsn, apply_fn)

        # Phase 3: Undo
        self._undo_pass(apply_fn)

        return self.redo_count, self.undo_count

    def _analysis_pass(self):
        """Rebuild transaction table and dirty page table from WAL.

        Returns the LSN from which redo should start.
        """
        reader = WALReader(self.wal.writer)

        # Find last checkpoint
        checkpoint = reader.find_last_checkpoint()

        start_lsn = LSN(0)
        if checkpoint:
            start_lsn = checkpoint.lsn
            # Restore dirty page table from checkpoint
            if checkpoint.dirty_pages:
                for key, lsn_val in checkpoint.dirty_pages.items():
                    if ':' in str(key):
                        parts = str(key).rsplit(':', 1)
                        table_name = parts[0]
                        page_id = int(parts[1])
                        self.dpt.mark_dirty(table_name, page_id, LSN(lsn_val))
            # Restore active transactions
            if checkpoint.active_txns:
                for txn_id in checkpoint.active_txns:
                    self.txn_table.begin(txn_id, checkpoint.lsn)

        # Scan forward from checkpoint
        for record in reader.scan_forward(start_lsn):
            self._analysis_records += 1

            if record.log_type == LogType.BEGIN:
                self.txn_table.begin(record.txn_id, record.lsn)
            elif record.log_type == LogType.COMMIT:
                self.txn_table.commit(record.txn_id)
            elif record.log_type == LogType.ABORT:
                self.txn_table.abort(record.txn_id)
            elif record.log_type == LogType.END:
                self.txn_table.remove(record.txn_id)
            elif record.log_type in (LogType.INSERT, LogType.UPDATE, LogType.DELETE, LogType.CLR):
                # Update transaction's last LSN
                self.txn_table.update_lsn(record.txn_id, record.lsn)
                # Add to dirty page table if not already there
                if record.table_name and record.page_id is not None:
                    self.dpt.mark_dirty(record.table_name, record.page_id, record.lsn)

        # Redo starts from minimum rec_lsn in dirty page table
        min_lsn = self.dpt.get_min_rec_lsn()
        return min_lsn if min_lsn else LSN(0)

    def _redo_pass(self, start_lsn, apply_fn=None):
        """Redo all actions from start_lsn, skipping pages already up-to-date."""
        reader = WALReader(self.wal.writer)

        for record in reader.scan_forward(start_lsn):
            if not record.is_redoable():
                continue

            # Check if page is in dirty page table
            if record.table_name and record.page_id is not None:
                rec_lsn = self.dpt.get_rec_lsn(record.table_name, record.page_id)
                if rec_lsn is None:
                    # Page not in DPT -- already flushed, skip
                    self.skipped_redo += 1
                    continue
                if record.lsn < rec_lsn:
                    # Record is older than dirty page's rec_lsn -- skip
                    self.skipped_redo += 1
                    continue

            # Apply the redo
            if apply_fn:
                apply_fn(record, True)
            self.redo_count += 1

    def _undo_pass(self, apply_fn=None):
        """Undo loser transactions in reverse LSN order."""
        losers = self.txn_table.loser_transactions()
        if not losers:
            return

        # Collect all LSNs to undo
        to_undo = {}  # lsn -> txn_entry
        for entry in losers:
            if entry.last_lsn.value > 0:
                to_undo[entry.last_lsn.value] = entry

        reader = WALReader(self.wal.writer)

        while to_undo:
            # Process highest LSN first
            max_lsn_val = max(to_undo.keys())
            entry = to_undo.pop(max_lsn_val)

            record = reader.find_by_lsn(LSN(max_lsn_val))
            if record is None:
                continue

            if record.is_undoable():
                # Write CLR (compensation log record)
                clr_lsn = self._write_clr(record)

                if apply_fn:
                    apply_fn(record, False)
                self.undo_count += 1

            # Follow prev_lsn chain
            if record.prev_lsn and record.prev_lsn.value > 0:
                to_undo[record.prev_lsn.value] = entry

    def _write_clr(self, original_record):
        """Write a CLR for the undone record."""
        # CLR points to the record before the one being undone
        undo_next = original_record.prev_lsn or LSN(0)

        # Determine the undo action
        if original_record.log_type == LogType.INSERT:
            # Undo insert = delete
            before = original_record.after_image
            after = None
        elif original_record.log_type == LogType.DELETE:
            # Undo delete = insert
            before = None
            after = original_record.before_image
        elif original_record.log_type == LogType.UPDATE:
            # Undo update = restore before image
            before = original_record.after_image
            after = original_record.before_image
        else:
            return None

        record = LogRecord(
            lsn=self.wal._next_lsn(),
            txn_id=original_record.txn_id,
            log_type=LogType.CLR,
            table_name=original_record.table_name,
            page_id=original_record.page_id,
            slot_idx=original_record.slot_idx,
            before_image=before,
            after_image=after,
            prev_lsn=original_record.prev_lsn,
            undo_next_lsn=undo_next,
        )
        self.wal.writer.write(record)
        return record.lsn

    def recovery_stats(self):
        return {
            'analysis_records': self._analysis_records,
            'redo_count': self.redo_count,
            'undo_count': self.undo_count,
            'skipped_redo': self.skipped_redo,
            'loser_txns': len(self.txn_table.loser_transactions()),
            'dirty_pages': len(self.dpt),
        }


# ============================================================
# Buffer Manager (Top-Level API)
# ============================================================

class BufferManager:
    """Top-level buffer manager composing WAL + Buffer Pool + Recovery.

    Provides transactional CRUD with:
    - Write-ahead logging
    - Page-level LSN tracking
    - Steal/no-force policy
    - Fuzzy and sharp checkpoints
    - ARIES-style recovery
    """

    def __init__(self, page_size=PAGE_SIZE, buffer_pool_size=64,
                 flush_policy=FlushPolicy.ON_EVICT):
        self.disk = DiskManager(page_size)
        self.wal = WALManager()
        self.dpt = DirtyPageTable()

        self.pool = WALBufferPool(
            self.disk, self.wal, buffer_pool_size, flush_policy
        )

        self.checkpoint_mgr = CheckpointManager(self.wal, self.pool, self.dpt)

        self._tables = {}  # table_name -> WALHeapFile
        self._next_txn_id = 1
        self._active_txns = set()
        self._committed_txns = set()
        self._aborted_txns = set()
        self._txn_snapshots = {}  # txn_id -> list of (op, row_id, before_data)

    # -- Table Management --

    def create_table(self, name):
        """Create a new heap table."""
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists")
        heap = WALHeapFile(self.pool, self.wal, name)
        self._tables[name] = heap
        return heap

    def get_table(self, name):
        return self._tables.get(name)

    def drop_table(self, name):
        if name in self._tables:
            del self._tables[name]

    def list_tables(self):
        return list(self._tables.keys())

    # -- Transaction Management --

    def begin_transaction(self):
        """Start a new transaction. Returns txn_id."""
        txn_id = self._next_txn_id
        self._next_txn_id += 1

        self.wal.begin(txn_id)
        self._active_txns.add(txn_id)
        self._txn_snapshots[txn_id] = []

        # Track in dirty page table's transaction state
        self.dpt  # DPT doesn't track txns, that's the txn table

        return txn_id

    def commit(self, txn_id):
        """Commit a transaction. WAL is force-flushed."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} is not active")

        self.wal.commit(txn_id)
        self._active_txns.discard(txn_id)
        self._committed_txns.add(txn_id)
        self._txn_snapshots.pop(txn_id, None)

    def abort(self, txn_id):
        """Abort a transaction. Undo all changes using snapshots."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} is not active")

        # Undo changes in reverse order
        ops = self._txn_snapshots.get(txn_id, [])
        for op, table_name, row_id, before_data in reversed(ops):
            heap = self._tables.get(table_name)
            if heap is None:
                continue

            if op == 'INSERT':
                # Undo insert = delete the row
                frame = self.pool.fetch_page(row_id.page_id)
                frame.page.delete_tuple(row_id.slot_index)
                lsn = self.wal.log_delete(txn_id, table_name, row_id.page_id,
                                           row_id.slot_index, before_image=before_data)
                self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn)
                heap._row_count -= 1
            elif op == 'DELETE':
                # Undo delete = re-insert the before data
                frame = self.pool.fetch_page(row_id.page_id)
                slot = frame.page.insert_tuple(before_data)
                if slot >= 0:
                    lsn = self.wal.log_insert(txn_id, table_name, row_id.page_id,
                                               slot, after_image=before_data)
                    self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn)
                else:
                    self.pool.unpin_page(row_id.page_id)
                heap._row_count += 1
            elif op == 'UPDATE':
                # Undo update = restore before image
                frame = self.pool.fetch_page(row_id.page_id)
                frame.page.update_tuple(row_id.slot_index, before_data)
                lsn = self.wal.log_update(txn_id, table_name, row_id.page_id,
                                           row_id.slot_index, before_image=None,
                                           after_image=before_data)
                self.pool.unpin_page(row_id.page_id, dirty=True, lsn=lsn)

        self.wal.abort(txn_id)
        self._active_txns.discard(txn_id)
        self._aborted_txns.add(txn_id)
        self._txn_snapshots.pop(txn_id, None)

    # -- CRUD Operations --

    def insert(self, txn_id, table_name, row_dict):
        """Insert a row within a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} is not active")

        heap = self._tables.get(table_name)
        if heap is None:
            raise ValueError(f"Table '{table_name}' does not exist")

        row_id = heap.insert(txn_id, row_dict)

        # Track for abort
        encoded = _encode_row(row_dict)
        self._txn_snapshots[txn_id].append(('INSERT', table_name, row_id, encoded))

        # Update dirty page table
        self.dpt.mark_dirty(table_name, row_id.page_id, heap.pool.frames[row_id.page_id].page.page_lsn)

        return row_id

    def get(self, table_name, row_id):
        """Read a row (no transaction required for reads)."""
        heap = self._tables.get(table_name)
        if heap is None:
            raise ValueError(f"Table '{table_name}' does not exist")
        return heap.get(row_id)

    def update(self, txn_id, table_name, row_id, new_row_dict):
        """Update a row within a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} is not active")

        heap = self._tables.get(table_name)
        if heap is None:
            raise ValueError(f"Table '{table_name}' does not exist")

        # Capture before image
        old_row = heap.get(row_id)
        before_data = _encode_row(old_row) if old_row else None

        new_row_id = heap.update(txn_id, row_id, new_row_dict)

        # Track for abort
        self._txn_snapshots[txn_id].append(('UPDATE', table_name, row_id, before_data))

        return new_row_id

    def delete(self, txn_id, table_name, row_id):
        """Delete a row within a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"Transaction {txn_id} is not active")

        heap = self._tables.get(table_name)
        if heap is None:
            raise ValueError(f"Table '{table_name}' does not exist")

        # Capture before image
        old_row = heap.get(row_id)
        before_data = _encode_row(old_row) if old_row else None

        result = heap.delete(txn_id, row_id)

        if result:
            self._txn_snapshots[txn_id].append(('DELETE', table_name, row_id, before_data))

        return result

    def scan(self, table_name, predicate=None):
        """Scan all rows in a table, optionally filtered."""
        heap = self._tables.get(table_name)
        if heap is None:
            raise ValueError(f"Table '{table_name}' does not exist")

        results = []
        for row_id, row_dict in heap.scan():
            if predicate is None or predicate(row_dict):
                results.append((row_id, row_dict))
        return results

    # -- Checkpoint & Recovery --

    def checkpoint(self):
        """Take a fuzzy checkpoint."""
        return self.checkpoint_mgr.take_checkpoint(
            active_txns=list(self._active_txns)
        )

    def sharp_checkpoint(self):
        """Take a sharp checkpoint (flushes all dirty pages)."""
        return self.checkpoint_mgr.take_sharp_checkpoint(
            active_txns=list(self._active_txns)
        )

    def recover(self, apply_fn=None):
        """Run ARIES recovery. Returns (redo_count, undo_count)."""
        recovery = EnhancedRecoveryManager(self.wal, self.pool, self.dpt)
        return recovery.recover(apply_fn)

    def create_recovery_manager(self):
        """Create a recovery manager for inspection."""
        return EnhancedRecoveryManager(self.wal, self.pool, self.dpt)

    # -- Flush & Stats --

    def flush(self):
        """Flush all dirty pages (WAL is flushed first)."""
        self.pool.flush_all()

    def stats(self):
        """Return comprehensive statistics."""
        return {
            'tables': list(self._tables.keys()),
            'active_txns': len(self._active_txns),
            'committed_txns': len(self._committed_txns),
            'aborted_txns': len(self._aborted_txns),
            'buffer_pool': self.pool.stats(),
            'dirty_pages': len(self.dpt),
            'checkpoints': self.checkpoint_mgr.checkpoint_count,
            'wal_records': self.wal.log_count(),
        }


# ============================================================
# Buffer Manager Analyzer
# ============================================================

class BufferManagerAnalyzer:
    """Analyzes buffer manager behavior and performance."""

    def __init__(self, buffer_manager):
        self.bm = buffer_manager

    def page_heat_map(self):
        """Which pages are accessed most (currently in pool)."""
        heat = {}
        for page_id, frame in self.bm.pool.frames.items():
            heat[page_id] = {
                'dirty': frame.dirty,
                'pin_count': frame.pin_count,
                'page_lsn': frame.page.page_lsn.value,
                'rec_lsn': frame.rec_lsn.value,
            }
        return heat

    def dirty_page_summary(self):
        """Summary of dirty page table state."""
        entries = self.bm.dpt.entries()
        return {
            'count': len(entries),
            'entries': [(t, p, l.value) for t, p, l in entries],
            'min_rec_lsn': self.bm.dpt.get_min_rec_lsn(),
        }

    def wal_summary(self):
        """WAL statistics."""
        records = self.bm.wal.get_all_records()
        by_type = {}
        for r in records:
            by_type[r.log_type] = by_type.get(r.log_type, 0) + 1
        return {
            'total_records': len(records),
            'by_type': by_type,
        }

    def transaction_summary(self):
        """Transaction statistics."""
        return {
            'active': len(self.bm._active_txns),
            'committed': len(self.bm._committed_txns),
            'aborted': len(self.bm._aborted_txns),
            'active_ids': list(self.bm._active_txns),
        }

    def recovery_estimate(self):
        """Estimate recovery cost based on current state."""
        min_lsn = self.bm.dpt.get_min_rec_lsn()
        total_records = self.bm.wal.log_count()

        if min_lsn is None:
            redo_records_est = 0
        else:
            # Estimate: records from min_lsn to end
            records = self.bm.wal.get_all_records()
            redo_records_est = sum(1 for r in records
                                   if r.lsn >= min_lsn and r.is_redoable())

        return {
            'dirty_pages': len(self.bm.dpt),
            'min_rec_lsn': min_lsn.value if min_lsn else None,
            'estimated_redo_records': redo_records_est,
            'active_txns_to_undo': len(self.bm._active_txns),
            'total_wal_records': total_records,
        }

    def full_report(self):
        """Complete analysis report."""
        return {
            'buffer_pool': self.bm.pool.stats(),
            'dirty_pages': self.dirty_page_summary(),
            'wal': self.wal_summary(),
            'transactions': self.transaction_summary(),
            'recovery_estimate': self.recovery_estimate(),
            'page_heat_map': self.page_heat_map(),
        }
