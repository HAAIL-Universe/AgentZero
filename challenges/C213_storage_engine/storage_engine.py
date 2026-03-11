"""
C213: Storage Engine
====================
A disk-oriented storage engine with:
- DiskManager: page-level I/O (simulated with bytearray)
- Page: fixed-size 4KB pages (slotted data pages, B-tree index pages)
- BufferPool: LRU page cache with pin/unpin, dirty tracking, flush
- BTreeIndex: disk-backed B-tree using page IDs (not memory pointers)
- HeapFile: slotted-page row storage with free-space tracking
- StorageEngine: top-level API composing all components

Composes with C212 Transaction Manager for ACID storage.
"""

import struct
import hashlib
from enum import IntEnum
from collections import OrderedDict

# =============================================================================
# Constants
# =============================================================================

PAGE_SIZE = 4096  # 4KB pages
INVALID_PAGE_ID = -1

# Page layout constants
PAGE_HEADER_SIZE = 16  # page_id(4) + page_type(2) + num_slots(2) + free_offset(4) + next_page(4)
SLOT_SIZE = 8  # offset(4) + length(4)

# B-tree constants
BTREE_ORDER = 50  # max keys per node (tuned for 4KB pages)
BTREE_HEADER_SIZE = 14  # page_type(2) + num_keys(2) + is_leaf(2) + parent_page(4) + next_leaf(4)
BTREE_KEY_ENTRY_SIZE = 12  # key_hash(4) + key_len(2) + val_len(2) + child_page(4)


# =============================================================================
# Page Types
# =============================================================================

class PageType(IntEnum):
    FREE = 0
    HEAP_DATA = 1
    BTREE_INTERNAL = 2
    BTREE_LEAF = 3
    FREESPACE_MAP = 4
    OVERFLOW = 5


# =============================================================================
# Serialization helpers
# =============================================================================

def _encode_value(val):
    """Encode a Python value to bytes."""
    if val is None:
        return b'\x00'
    elif isinstance(val, bool):
        return b'\x01\x01' if val else b'\x01\x00'
    elif isinstance(val, int):
        return b'\x02' + struct.pack('<q', val)
    elif isinstance(val, float):
        return b'\x03' + struct.pack('<d', val)
    elif isinstance(val, str):
        encoded = val.encode('utf-8')
        return b'\x04' + struct.pack('<I', len(encoded)) + encoded
    elif isinstance(val, bytes):
        return b'\x05' + struct.pack('<I', len(val)) + val
    elif isinstance(val, (list, tuple)):
        parts = []
        for item in val:
            parts.append(_encode_value(item))
        joined = b''.join(parts)
        return b'\x06' + struct.pack('<I', len(val)) + joined
    else:
        # Fallback: encode as string
        s = str(val)
        encoded = s.encode('utf-8')
        return b'\x04' + struct.pack('<I', len(encoded)) + encoded


def _decode_value(data, offset=0):
    """Decode a Python value from bytes. Returns (value, new_offset)."""
    tag = data[offset]
    offset += 1
    if tag == 0:
        return None, offset
    elif tag == 1:
        val = data[offset] != 0
        return val, offset + 1
    elif tag == 2:
        val = struct.unpack('<q', data[offset:offset + 8])[0]
        return val, offset + 8
    elif tag == 3:
        val = struct.unpack('<d', data[offset:offset + 8])[0]
        return val, offset + 8
    elif tag == 4:
        length = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        val = data[offset:offset + length].decode('utf-8')
        return val, offset + length
    elif tag == 5:
        length = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        val = bytes(data[offset:offset + length])
        return val, offset + length
    elif tag == 6:
        count = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        items = []
        for _ in range(count):
            item, offset = _decode_value(data, offset)
            items.append(item)
        return items, offset
    else:
        raise ValueError(f"Unknown type tag: {tag}")


def _encode_row(row_dict):
    """Encode a dict row to bytes."""
    keys = sorted(row_dict.keys())
    parts = [struct.pack('<H', len(keys))]
    for key in keys:
        key_bytes = key.encode('utf-8')
        parts.append(struct.pack('<H', len(key_bytes)))
        parts.append(key_bytes)
        parts.append(_encode_value(row_dict[key]))
    return b''.join(parts)


def _decode_row(data, offset=0):
    """Decode a dict row from bytes. Returns (dict, new_offset)."""
    num_keys = struct.unpack('<H', data[offset:offset + 2])[0]
    offset += 2
    row = {}
    for _ in range(num_keys):
        key_len = struct.unpack('<H', data[offset:offset + 2])[0]
        offset += 2
        key = data[offset:offset + key_len].decode('utf-8')
        offset += key_len
        val, offset = _decode_value(data, offset)
        row[key] = val
    return row, offset


def _key_to_bytes(key):
    """Convert a key value to comparable bytes."""
    if isinstance(key, int):
        # Map signed int to unsigned for sortable bytes
        # Add offset to shift range: signed [-2^63, 2^63-1] -> unsigned [0, 2^64-1]
        mapped = (key + (1 << 63)) & ((1 << 64) - 1)
        return struct.pack('>Q', mapped)
    elif isinstance(key, float):
        # IEEE 754 trick for sortable float bytes
        bits = struct.pack('>d', key)
        n = int.from_bytes(bits, 'big')
        if key >= 0:
            n ^= (1 << 63)
        else:
            n = ~n & ((1 << 64) - 1)
        return n.to_bytes(8, 'big')
    elif isinstance(key, str):
        return key.encode('utf-8')
    elif isinstance(key, tuple):
        parts = []
        for k in key:
            kb = _key_to_bytes(k)
            parts.append(struct.pack('>H', len(kb)))
            parts.append(kb)
        return b''.join(parts)
    elif key is None:
        return b''
    else:
        return str(key).encode('utf-8')


def _normalize_key(k):
    """Normalize key: convert lists to tuples for consistent comparison."""
    if isinstance(k, list):
        return tuple(_normalize_key(x) for x in k)
    return k


def _compare_keys(a, b):
    """Compare two keys. Returns -1, 0, or 1."""
    a = _normalize_key(a)
    b = _normalize_key(b)
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    if type(a) == type(b):
        if a < b:
            return -1
        elif a > b:
            return 1
        return 0
    # Mixed types: compare by bytes
    ab = _key_to_bytes(a)
    bb = _key_to_bytes(b)
    if ab < bb:
        return -1
    elif ab > bb:
        return 1
    return 0


# =============================================================================
# DiskManager -- page-level I/O
# =============================================================================

class DiskManager:
    """Manages page-level I/O using in-memory bytearrays (simulating disk)."""

    def __init__(self, page_size=PAGE_SIZE):
        self.page_size = page_size
        self._storage = bytearray()  # Contiguous "disk"
        self._num_pages = 0
        self._free_pages = []  # Recycled page IDs

    def allocate_page(self):
        """Allocate a new page. Returns page_id."""
        if self._free_pages:
            page_id = self._free_pages.pop()
            # Zero out recycled page
            start = page_id * self.page_size
            self._storage[start:start + self.page_size] = bytes(self.page_size)
            return page_id
        page_id = self._num_pages
        self._num_pages += 1
        self._storage.extend(bytes(self.page_size))
        return page_id

    def deallocate_page(self, page_id):
        """Return a page to the free list."""
        if 0 <= page_id < self._num_pages and page_id not in self._free_pages:
            self._free_pages.append(page_id)
            start = page_id * self.page_size
            self._storage[start:start + self.page_size] = bytes(self.page_size)

    def read_page(self, page_id):
        """Read a page from disk. Returns bytearray copy."""
        if page_id < 0 or page_id >= self._num_pages:
            raise IOError(f"Invalid page_id: {page_id}")
        start = page_id * self.page_size
        return bytearray(self._storage[start:start + self.page_size])

    def write_page(self, page_id, data):
        """Write data to a page on disk."""
        if page_id < 0 or page_id >= self._num_pages:
            raise IOError(f"Invalid page_id: {page_id}")
        if len(data) != self.page_size:
            raise ValueError(f"Data size {len(data)} != page size {self.page_size}")
        start = page_id * self.page_size
        self._storage[start:start + self.page_size] = data

    @property
    def num_pages(self):
        return self._num_pages

    @property
    def num_free_pages(self):
        return len(self._free_pages)

    def flush(self):
        """Flush to disk (no-op for in-memory simulation)."""
        pass


# =============================================================================
# BufferPool -- LRU page cache
# =============================================================================

class BufferPoolPage:
    """A page frame in the buffer pool."""
    __slots__ = ['page_id', 'data', 'dirty', 'pin_count']

    def __init__(self, page_id, data):
        self.page_id = page_id
        self.data = data
        self.dirty = False
        self.pin_count = 0


class BufferPool:
    """LRU buffer pool with pin/unpin and dirty page tracking."""

    def __init__(self, disk_manager, pool_size=64):
        self.disk = disk_manager
        self.pool_size = pool_size
        self._frames = OrderedDict()  # page_id -> BufferPoolPage (LRU order)
        self._hit_count = 0
        self._miss_count = 0

    def fetch_page(self, page_id):
        """Fetch a page, pinning it. Returns BufferPoolPage."""
        if page_id in self._frames:
            frame = self._frames[page_id]
            # Move to end (most recently used)
            self._frames.move_to_end(page_id)
            frame.pin_count += 1
            self._hit_count += 1
            return frame

        # Cache miss
        self._miss_count += 1
        self._evict_if_needed()

        data = self.disk.read_page(page_id)
        frame = BufferPoolPage(page_id, data)
        frame.pin_count = 1
        self._frames[page_id] = frame
        return frame

    def new_page(self):
        """Allocate a new page and pin it. Returns BufferPoolPage."""
        self._evict_if_needed()
        page_id = self.disk.allocate_page()
        data = bytearray(self.disk.page_size)
        frame = BufferPoolPage(page_id, data)
        frame.pin_count = 1
        frame.dirty = True
        self._frames[page_id] = frame
        return frame

    def unpin_page(self, page_id, dirty=False):
        """Unpin a page. If dirty, mark for flush."""
        if page_id not in self._frames:
            return
        frame = self._frames[page_id]
        if frame.pin_count > 0:
            frame.pin_count -= 1
        if dirty:
            frame.dirty = True

    def flush_page(self, page_id):
        """Write a dirty page back to disk."""
        if page_id not in self._frames:
            return
        frame = self._frames[page_id]
        if frame.dirty:
            self.disk.write_page(page_id, frame.data)
            frame.dirty = False

    def flush_all(self):
        """Flush all dirty pages to disk."""
        for page_id, frame in self._frames.items():
            if frame.dirty:
                self.disk.write_page(page_id, frame.data)
                frame.dirty = False

    def delete_page(self, page_id):
        """Remove page from pool and deallocate on disk."""
        if page_id in self._frames:
            del self._frames[page_id]
        self.disk.deallocate_page(page_id)

    def _evict_if_needed(self):
        """Evict LRU unpinned page if pool is full."""
        while len(self._frames) >= self.pool_size:
            evicted = False
            # Find LRU unpinned page
            for pid in list(self._frames.keys()):
                frame = self._frames[pid]
                if frame.pin_count == 0:
                    if frame.dirty:
                        self.disk.write_page(pid, frame.data)
                    del self._frames[pid]
                    evicted = True
                    break
            if not evicted:
                raise RuntimeError("BufferPool full: all pages pinned")

    @property
    def hit_rate(self):
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    @property
    def size(self):
        return len(self._frames)

    @property
    def dirty_count(self):
        return sum(1 for f in self._frames.values() if f.dirty)


# =============================================================================
# HeapFile -- slotted page row storage
# =============================================================================

class RowID:
    """Identifies a row: (page_id, slot_index)."""
    __slots__ = ['page_id', 'slot_index']

    def __init__(self, page_id, slot_index):
        self.page_id = page_id
        self.slot_index = slot_index

    def __eq__(self, other):
        return (isinstance(other, RowID) and
                self.page_id == other.page_id and
                self.slot_index == other.slot_index)

    def __hash__(self):
        return hash((self.page_id, self.slot_index))

    def __repr__(self):
        return f"RowID({self.page_id}, {self.slot_index})"

    def to_tuple(self):
        return (self.page_id, self.slot_index)


class SlottedPage:
    """
    Slotted page layout:
    [Header: 16 bytes][Slot directory grows ->  ... free space ...  <- Tuple data grows]

    Header: page_id(4) + page_type(2) + num_slots(2) + free_start(4) + next_page(4)
    Each slot: offset(4) + length(4), where offset=0 means deleted
    Tuple data grows from end of page backward.
    """

    def __init__(self, page_id, data=None, page_size=PAGE_SIZE):
        self.page_id = page_id
        self.page_size = page_size
        if data is None:
            self.data = bytearray(page_size)
            self._write_header(page_id, PageType.HEAP_DATA, 0, PAGE_HEADER_SIZE, INVALID_PAGE_ID)
        else:
            self.data = data

    def _write_header(self, page_id, page_type, num_slots, free_start, next_page):
        struct.pack_into('<IHHiI', self.data, 0, page_id, page_type, num_slots, free_start, next_page & 0xFFFFFFFF)

    def _read_header(self):
        page_id, page_type, num_slots, free_start, next_page_raw = struct.unpack_from('<IHHiI', self.data, 0)
        next_page = next_page_raw if next_page_raw != 0xFFFFFFFF else INVALID_PAGE_ID
        return page_id, PageType(page_type), num_slots, free_start, next_page

    @property
    def num_slots(self):
        return struct.unpack_from('<H', self.data, 6)[0]

    @property
    def free_start(self):
        return struct.unpack_from('<i', self.data, 8)[0]

    @free_start.setter
    def free_start(self, val):
        struct.pack_into('<i', self.data, 8, val)

    @property
    def next_page(self):
        raw = struct.unpack_from('<I', self.data, 12)[0]
        return raw if raw != 0xFFFFFFFF else INVALID_PAGE_ID

    @next_page.setter
    def next_page(self, val):
        struct.pack_into('<I', self.data, 12, val & 0xFFFFFFFF if val != INVALID_PAGE_ID else 0xFFFFFFFF)

    def _slot_offset(self, slot_index):
        """Byte offset of slot entry in the directory."""
        return PAGE_HEADER_SIZE + slot_index * SLOT_SIZE

    def _read_slot(self, slot_index):
        """Read (offset, length) for a slot."""
        off = self._slot_offset(slot_index)
        return struct.unpack_from('<Ii', self.data, off)

    def _write_slot(self, slot_index, offset, length):
        """Write (offset, length) for a slot."""
        off = self._slot_offset(slot_index)
        struct.pack_into('<Ii', self.data, off, offset, length)

    def free_space(self):
        """Available space for new tuples."""
        _, _, num_slots, free_start, _ = self._read_header()
        slot_dir_end = PAGE_HEADER_SIZE + num_slots * SLOT_SIZE
        # Find lowest tuple offset
        lowest = self.page_size
        for i in range(num_slots):
            off, length = self._read_slot(i)
            if off > 0 and length > 0:
                lowest = min(lowest, off)
        # Free space = gap between slot directory end and lowest tuple
        # But also account for a new slot entry
        return lowest - slot_dir_end - SLOT_SIZE

    def insert_tuple(self, tuple_data):
        """Insert tuple data. Returns slot_index or -1 if no space."""
        data_len = len(tuple_data)
        _, _, num_slots, free_start, _ = self._read_header()

        # Calculate where new tuple would go
        slot_dir_end = PAGE_HEADER_SIZE + (num_slots + 1) * SLOT_SIZE

        # Find lowest existing tuple offset
        lowest = self.page_size
        for i in range(num_slots):
            off, length = self._read_slot(i)
            if off > 0 and length > 0:
                lowest = min(lowest, off)

        tuple_offset = lowest - data_len

        if tuple_offset < slot_dir_end:
            return -1  # No space

        # Check for deleted slot to reuse
        reuse_slot = -1
        for i in range(num_slots):
            off, length = self._read_slot(i)
            if off == 0 and length == 0:
                reuse_slot = i
                break

        # Write tuple data
        self.data[tuple_offset:tuple_offset + data_len] = tuple_data

        if reuse_slot >= 0:
            self._write_slot(reuse_slot, tuple_offset, data_len)
            return reuse_slot
        else:
            # New slot
            self._write_slot(num_slots, tuple_offset, data_len)
            struct.pack_into('<H', self.data, 6, num_slots + 1)
            return num_slots

    def get_tuple(self, slot_index):
        """Get tuple data by slot index. Returns bytes or None if deleted."""
        num_slots = self.num_slots
        if slot_index < 0 or slot_index >= num_slots:
            return None
        off, length = self._read_slot(slot_index)
        if off == 0 or length <= 0:
            return None
        return bytes(self.data[off:off + length])

    def delete_tuple(self, slot_index):
        """Mark a tuple as deleted. Returns True if deleted."""
        num_slots = self.num_slots
        if slot_index < 0 or slot_index >= num_slots:
            return False
        off, length = self._read_slot(slot_index)
        if off == 0:
            return False
        self._write_slot(slot_index, 0, 0)
        return True

    def update_tuple(self, slot_index, new_data):
        """Update a tuple in-place if it fits, otherwise delete + insert.
        Returns new slot_index or -1 if no space."""
        num_slots = self.num_slots
        if slot_index < 0 or slot_index >= num_slots:
            return -1
        old_off, old_len = self._read_slot(slot_index)
        if old_off == 0:
            return -1

        if len(new_data) <= old_len:
            # Fits in place
            self.data[old_off:old_off + len(new_data)] = new_data
            if len(new_data) < old_len:
                # Zero out remainder
                self.data[old_off + len(new_data):old_off + old_len] = bytes(old_len - len(new_data))
            self._write_slot(slot_index, old_off, len(new_data))
            return slot_index
        else:
            # Delete old, insert new
            self.delete_tuple(slot_index)
            return self.insert_tuple(new_data)

    def iter_tuples(self):
        """Iterate over (slot_index, tuple_data) for live tuples."""
        num_slots = self.num_slots
        for i in range(num_slots):
            off, length = self._read_slot(i)
            if off > 0 and length > 0:
                yield i, bytes(self.data[off:off + length])

    def compact(self):
        """Compact the page to reclaim fragmented space."""
        tuples = list(self.iter_tuples())
        num_slots = len(tuples)

        # Rebuild from scratch
        new_data = bytearray(self.page_size)
        page_id = struct.unpack_from('<I', self.data, 0)[0]
        next_page = self.next_page

        offset = self.page_size
        for i, (_, tdata) in enumerate(tuples):
            offset -= len(tdata)
            new_data[offset:offset + len(tdata)] = tdata
            slot_off = PAGE_HEADER_SIZE + i * SLOT_SIZE
            struct.pack_into('<Ii', new_data, slot_off, offset, len(tdata))

        struct.pack_into('<IHHiI', new_data, 0,
                         page_id, PageType.HEAP_DATA, num_slots,
                         PAGE_HEADER_SIZE + num_slots * SLOT_SIZE,
                         next_page & 0xFFFFFFFF if next_page != INVALID_PAGE_ID else 0xFFFFFFFF)

        self.data[:] = new_data
        return num_slots


class HeapFile:
    """Manages a collection of slotted pages for row storage."""

    def __init__(self, buffer_pool, table_name="default"):
        self.buffer_pool = buffer_pool
        self.table_name = table_name
        self._first_page_id = INVALID_PAGE_ID
        self._page_ids = []  # All page IDs in this heap
        self._row_count = 0

    def _ensure_page(self):
        """Get a page with space, or allocate a new one."""
        # Try existing pages
        for pid in self._page_ids:
            frame = self.buffer_pool.fetch_page(pid)
            page = SlottedPage(pid, frame.data, self.buffer_pool.disk.page_size)
            if page.free_space() > SLOT_SIZE + 16:  # At least some usable space
                return frame, page
            self.buffer_pool.unpin_page(pid)

        # Allocate new page
        frame = self.buffer_pool.new_page()
        pid = frame.page_id
        page = SlottedPage(pid, frame.data, self.buffer_pool.disk.page_size)

        if self._page_ids:
            # Link from last page
            last_frame = self.buffer_pool.fetch_page(self._page_ids[-1])
            last_page = SlottedPage(self._page_ids[-1], last_frame.data, self.buffer_pool.disk.page_size)
            last_page.next_page = pid
            self.buffer_pool.unpin_page(self._page_ids[-1], dirty=True)

        self._page_ids.append(pid)
        if self._first_page_id == INVALID_PAGE_ID:
            self._first_page_id = pid

        return frame, page

    def insert(self, row_dict):
        """Insert a row. Returns RowID."""
        row_bytes = _encode_row(row_dict)

        frame, page = self._ensure_page()
        slot = page.insert_tuple(row_bytes)

        if slot == -1:
            # Page was "big enough" but insert failed (fragmentation)
            page.compact()
            slot = page.insert_tuple(row_bytes)

        if slot == -1:
            # Still no room, try a brand new page
            self.buffer_pool.unpin_page(frame.page_id)
            new_frame = self.buffer_pool.new_page()
            pid = new_frame.page_id
            new_page = SlottedPage(pid, new_frame.data, self.buffer_pool.disk.page_size)

            if self._page_ids:
                last_frame = self.buffer_pool.fetch_page(self._page_ids[-1])
                last_page = SlottedPage(self._page_ids[-1], last_frame.data, self.buffer_pool.disk.page_size)
                last_page.next_page = pid
                self.buffer_pool.unpin_page(self._page_ids[-1], dirty=True)

            self._page_ids.append(pid)
            slot = new_page.insert_tuple(row_bytes)
            self.buffer_pool.unpin_page(pid, dirty=True)
            self._row_count += 1
            return RowID(pid, slot)

        self.buffer_pool.unpin_page(frame.page_id, dirty=True)
        self._row_count += 1
        return RowID(frame.page_id, slot)

    def get(self, row_id):
        """Get a row by RowID. Returns dict or None."""
        frame = self.buffer_pool.fetch_page(row_id.page_id)
        page = SlottedPage(row_id.page_id, frame.data, self.buffer_pool.disk.page_size)
        tdata = page.get_tuple(row_id.slot_index)
        self.buffer_pool.unpin_page(row_id.page_id)
        if tdata is None:
            return None
        row, _ = _decode_row(tdata)
        return row

    def update(self, row_id, new_row_dict):
        """Update a row. Returns new RowID (may change on page overflow)."""
        new_bytes = _encode_row(new_row_dict)
        frame = self.buffer_pool.fetch_page(row_id.page_id)
        page = SlottedPage(row_id.page_id, frame.data, self.buffer_pool.disk.page_size)
        new_slot = page.update_tuple(row_id.slot_index, new_bytes)

        if new_slot >= 0:
            self.buffer_pool.unpin_page(row_id.page_id, dirty=True)
            return RowID(row_id.page_id, new_slot)

        # Doesn't fit on this page -- delete and insert elsewhere
        page.delete_tuple(row_id.slot_index)
        self.buffer_pool.unpin_page(row_id.page_id, dirty=True)
        return self.insert(new_row_dict)

    def delete(self, row_id):
        """Delete a row. Returns True if deleted."""
        frame = self.buffer_pool.fetch_page(row_id.page_id)
        page = SlottedPage(row_id.page_id, frame.data, self.buffer_pool.disk.page_size)
        result = page.delete_tuple(row_id.slot_index)
        self.buffer_pool.unpin_page(row_id.page_id, dirty=True)
        if result:
            self._row_count -= 1
        return result

    def scan(self):
        """Full table scan. Yields (RowID, row_dict)."""
        for pid in self._page_ids:
            frame = self.buffer_pool.fetch_page(pid)
            page = SlottedPage(pid, frame.data, self.buffer_pool.disk.page_size)
            for slot_idx, tdata in page.iter_tuples():
                row, _ = _decode_row(tdata)
                yield RowID(pid, slot_idx), row
            self.buffer_pool.unpin_page(pid)

    @property
    def row_count(self):
        return self._row_count

    @property
    def page_count(self):
        return len(self._page_ids)


# =============================================================================
# BTreeIndex -- disk-backed B-tree using page IDs
# =============================================================================

class BTreeNode:
    """In-memory representation of a B-tree node (loaded from page)."""

    def __init__(self, page_id, is_leaf=True):
        self.page_id = page_id
        self.is_leaf = is_leaf
        self.keys = []      # List of key values
        self.values = []     # For leaves: list of RowID tuples; for internal: unused
        self.children = []   # For internal: list of child page_ids
        self.parent_page = INVALID_PAGE_ID
        self.next_leaf = INVALID_PAGE_ID  # For leaves: next leaf page
        self.prev_leaf = INVALID_PAGE_ID  # For leaves: prev leaf page

    def serialize(self, page_size=PAGE_SIZE):
        """Serialize node to page bytes."""
        data = bytearray(page_size)
        page_type = PageType.BTREE_LEAF if self.is_leaf else PageType.BTREE_INTERNAL

        # Header: type(2) + num_keys(2) + is_leaf(2) + parent(4) + next_leaf(4) + prev_leaf(4) = 18
        struct.pack_into('<HHHiii', data, 0,
                         page_type, len(self.keys), 1 if self.is_leaf else 0,
                         self.parent_page, self.next_leaf, self.prev_leaf)

        offset = 20  # After header

        # Encode keys
        for key in self.keys:
            kb = _encode_value(key)
            if offset + 2 + len(kb) > page_size:
                break
            struct.pack_into('<H', data, offset, len(kb))
            offset += 2
            data[offset:offset + len(kb)] = kb
            offset += len(kb)

        # Marker between keys and values/children
        struct.pack_into('<H', data, offset, 0xFFFF)
        offset += 2

        if self.is_leaf:
            # Encode values (RowID tuples)
            for val in self.values:
                if isinstance(val, RowID):
                    pid, si = val.page_id, val.slot_index
                elif isinstance(val, tuple):
                    pid, si = val
                else:
                    pid, si = -1, -1
                if offset + 8 > page_size:
                    break
                struct.pack_into('<ii', data, offset, pid, si)
                offset += 8
        else:
            # Encode children (page IDs)
            for child_pid in self.children:
                if offset + 4 > page_size:
                    break
                struct.pack_into('<i', data, offset, child_pid)
                offset += 4

        return data

    @staticmethod
    def deserialize(page_id, data):
        """Deserialize a node from page bytes."""
        page_type, num_keys, is_leaf_raw, parent_page, next_leaf, prev_leaf = \
            struct.unpack_from('<HHHiii', data, 0)

        node = BTreeNode(page_id, is_leaf=is_leaf_raw != 0)
        node.parent_page = parent_page
        node.next_leaf = next_leaf
        node.prev_leaf = prev_leaf

        offset = 20

        # Decode keys
        for _ in range(num_keys):
            key_len = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            key, _ = _decode_value(data, offset)
            offset += key_len
            node.keys.append(key)

        # Skip marker
        marker = struct.unpack_from('<H', data, offset)[0]
        offset += 2

        if node.is_leaf:
            for _ in range(num_keys):
                if offset + 8 > len(data):
                    break
                pid, si = struct.unpack_from('<ii', data, offset)
                offset += 8
                node.values.append(RowID(pid, si))
        else:
            # num_keys + 1 children
            for _ in range(num_keys + 1):
                if offset + 4 > len(data):
                    break
                child_pid = struct.unpack_from('<i', data, offset)[0]
                offset += 4
                node.children.append(child_pid)

        return node


class BTreeIndex:
    """Disk-backed B-tree index using page IDs."""

    def __init__(self, buffer_pool, name="index", order=None):
        self.buffer_pool = buffer_pool
        self.name = name
        self.order = order or self._compute_order()
        self._root_page_id = INVALID_PAGE_ID
        self._size = 0
        self._height = 0

        # Create root leaf
        self._create_root()

    def _compute_order(self):
        """Compute max keys per node based on page size."""
        # Conservative estimate
        return min(BTREE_ORDER, max(4, (self.buffer_pool.disk.page_size - 40) // 30))

    def _create_root(self):
        frame = self.buffer_pool.new_page()
        root = BTreeNode(frame.page_id, is_leaf=True)
        frame.data[:] = root.serialize(self.buffer_pool.disk.page_size)
        self.buffer_pool.unpin_page(frame.page_id, dirty=True)
        self._root_page_id = frame.page_id
        self._height = 1

    def _load_node(self, page_id):
        """Load a BTreeNode from a page."""
        frame = self.buffer_pool.fetch_page(page_id)
        node = BTreeNode.deserialize(page_id, frame.data)
        self.buffer_pool.unpin_page(page_id)
        return node

    def _save_node(self, node):
        """Save a BTreeNode to its page."""
        frame = self.buffer_pool.fetch_page(node.page_id)
        frame.data[:] = node.serialize(self.buffer_pool.disk.page_size)
        self.buffer_pool.unpin_page(node.page_id, dirty=True)

    def _find_leaf(self, key):
        """Find the leaf node that should contain key."""
        node = self._load_node(self._root_page_id)
        while not node.is_leaf:
            idx = self._find_child_index(node, key)
            if idx >= len(node.children):
                idx = len(node.children) - 1
            node = self._load_node(node.children[idx])
        return node

    def _find_child_index(self, node, key):
        """Find index of child to follow for key."""
        for i, k in enumerate(node.keys):
            if _compare_keys(key, k) < 0:
                return i
        return len(node.keys)

    def _find_key_index(self, node, key):
        """Find index of key in node, or insertion point."""
        for i, k in enumerate(node.keys):
            cmp = _compare_keys(key, k)
            if cmp == 0:
                return i, True
            elif cmp < 0:
                return i, False
        return len(node.keys), False

    def insert(self, key, row_id):
        """Insert a key -> RowID mapping."""
        if not isinstance(row_id, RowID):
            row_id = RowID(*row_id)

        leaf = self._find_leaf(key)
        idx, found = self._find_key_index(leaf, key)

        if found:
            # Update existing key
            leaf.values[idx] = row_id
            self._save_node(leaf)
            return

        # Insert into leaf
        leaf.keys.insert(idx, key)
        leaf.values.insert(idx, row_id)
        self._size += 1

        if len(leaf.keys) < self.order:
            self._save_node(leaf)
            return

        # Split leaf
        self._split_leaf(leaf)

    def _split_leaf(self, leaf):
        """Split a full leaf node."""
        mid = len(leaf.keys) // 2

        # Create new right leaf
        right_frame = self.buffer_pool.new_page()
        right = BTreeNode(right_frame.page_id, is_leaf=True)
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]
        right.parent_page = leaf.parent_page
        right.next_leaf = leaf.next_leaf
        right.prev_leaf = leaf.page_id

        # Update old next leaf's prev pointer
        if leaf.next_leaf != INVALID_PAGE_ID:
            old_next = self._load_node(leaf.next_leaf)
            old_next.prev_leaf = right.page_id
            self._save_node(old_next)

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.next_leaf = right.page_id

        self._save_node(leaf)
        self._save_node(right)
        self.buffer_pool.unpin_page(right_frame.page_id)

        # Promote first key of right to parent
        promote_key = right.keys[0]
        self._insert_into_parent(leaf, promote_key, right)

    def _split_internal(self, node):
        """Split a full internal node."""
        mid = len(node.keys) // 2
        promote_key = node.keys[mid]

        right_frame = self.buffer_pool.new_page()
        right = BTreeNode(right_frame.page_id, is_leaf=False)
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        right.parent_page = node.parent_page

        # Update children's parent pointers
        for child_pid in right.children:
            child = self._load_node(child_pid)
            child.parent_page = right.page_id
            self._save_node(child)

        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        self._save_node(node)
        self._save_node(right)
        self.buffer_pool.unpin_page(right_frame.page_id)

        self._insert_into_parent(node, promote_key, right)

    def _insert_into_parent(self, left, key, right):
        """Insert a key into the parent of left and right."""
        if left.parent_page == INVALID_PAGE_ID:
            # Create new root
            root_frame = self.buffer_pool.new_page()
            new_root = BTreeNode(root_frame.page_id, is_leaf=False)
            new_root.keys = [key]
            new_root.children = [left.page_id, right.page_id]

            left.parent_page = new_root.page_id
            right.parent_page = new_root.page_id

            self._save_node(left)
            self._save_node(right)
            self._save_node(new_root)
            self.buffer_pool.unpin_page(root_frame.page_id)

            self._root_page_id = new_root.page_id
            self._height += 1
            return

        parent = self._load_node(left.parent_page)
        idx = self._find_child_index(parent, key)
        parent.keys.insert(idx, key)
        parent.children.insert(idx + 1, right.page_id)
        right.parent_page = parent.page_id
        self._save_node(right)

        if len(parent.keys) < self.order:
            self._save_node(parent)
            return

        self._split_internal(parent)

    def search(self, key):
        """Search for a key. Returns RowID or None."""
        leaf = self._find_leaf(key)
        idx, found = self._find_key_index(leaf, key)
        if found:
            return leaf.values[idx]
        return None

    def delete(self, key):
        """Delete a key. Returns True if found and deleted."""
        leaf = self._find_leaf(key)
        idx, found = self._find_key_index(leaf, key)
        if not found:
            return False

        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        self._size -= 1
        self._save_node(leaf)

        # Simple approach: don't rebalance for now (common in practice)
        # Just handle empty root
        if leaf.page_id == self._root_page_id and not leaf.keys:
            pass  # Empty tree, that's fine

        return True

    def range_scan(self, low=None, high=None, include_low=True, include_high=True):
        """Range scan. Yields (key, RowID) pairs in order."""
        if low is not None:
            leaf = self._find_leaf(low)
        else:
            # Start from leftmost leaf
            leaf = self._load_node(self._root_page_id)
            while not leaf.is_leaf:
                leaf = self._load_node(leaf.children[0])

        while True:
            for i, key in enumerate(leaf.keys):
                if low is not None:
                    cmp = _compare_keys(key, low)
                    if cmp < 0 or (cmp == 0 and not include_low):
                        continue
                if high is not None:
                    cmp = _compare_keys(key, high)
                    if cmp > 0 or (cmp == 0 and not include_high):
                        return
                yield key, leaf.values[i]

            if leaf.next_leaf == INVALID_PAGE_ID:
                return
            leaf = self._load_node(leaf.next_leaf)

    def scan_all(self):
        """Scan all entries in order. Yields (key, RowID)."""
        return self.range_scan()

    def min_key(self):
        """Get minimum key."""
        node = self._load_node(self._root_page_id)
        while not node.is_leaf:
            node = self._load_node(node.children[0])
        if node.keys:
            return node.keys[0]
        return None

    def max_key(self):
        """Get maximum key."""
        node = self._load_node(self._root_page_id)
        while not node.is_leaf:
            node = self._load_node(node.children[-1])
        if node.keys:
            return node.keys[-1]
        return None

    @property
    def size(self):
        return self._size

    @property
    def height(self):
        return self._height

    @property
    def root_page_id(self):
        return self._root_page_id


# =============================================================================
# Table -- combines HeapFile + BTreeIndexes
# =============================================================================

class TableSchema:
    """Table schema definition."""

    def __init__(self, name, columns, primary_key=None, column_types=None):
        self.name = name
        self.columns = list(columns)
        self.primary_key = primary_key
        self.column_types = column_types or {}

    def validate_row(self, row):
        """Validate a row against the schema."""
        for col in self.columns:
            if col not in row and col != self.primary_key:
                row[col] = None
        return row


class Table:
    """A table with heap storage and B-tree indexes."""

    def __init__(self, schema, buffer_pool):
        self.schema = schema
        self.buffer_pool = buffer_pool
        self.heap = HeapFile(buffer_pool, schema.name)
        self.indexes = {}  # name -> BTreeIndex
        self._next_id = 1  # Auto-increment for primary key

        # Create primary key index if specified
        if schema.primary_key:
            self.indexes[f"pk_{schema.name}"] = BTreeIndex(
                buffer_pool, name=f"pk_{schema.name}")

    def insert(self, row):
        """Insert a row. Returns RowID."""
        row = dict(row)  # Copy
        self.schema.validate_row(row)

        # Auto-increment primary key
        if self.schema.primary_key:
            pk = self.schema.primary_key
            if pk not in row or row[pk] is None:
                row[pk] = self._next_id
                self._next_id += 1
            else:
                if isinstance(row[pk], int):
                    self._next_id = max(self._next_id, row[pk] + 1)

        row_id = self.heap.insert(row)

        # Update indexes
        for idx_name, idx in self.indexes.items():
            if idx_name.startswith("pk_"):
                pk_val = row.get(self.schema.primary_key)
                if pk_val is not None:
                    idx.insert(pk_val, row_id)
            else:
                # Secondary index -- composite key for uniqueness
                idx_cols = self._index_columns(idx_name)
                if idx_cols:
                    key = self._extract_key(row, idx_cols)
                    if key is not None:
                        composite = (key, row_id.page_id, row_id.slot_index)
                        idx.insert(composite, row_id)

        return row_id

    def get_by_pk(self, pk_value):
        """Get a row by primary key. Returns dict or None."""
        if not self.schema.primary_key:
            return None
        pk_idx_name = f"pk_{self.schema.name}"
        if pk_idx_name not in self.indexes:
            return None
        row_id = self.indexes[pk_idx_name].search(pk_value)
        if row_id is None:
            return None
        return self.heap.get(row_id)

    def get_by_rowid(self, row_id):
        """Get a row by RowID."""
        return self.heap.get(row_id)

    def update_by_pk(self, pk_value, new_data):
        """Update a row by primary key. Returns True if updated."""
        if not self.schema.primary_key:
            return False
        pk_idx_name = f"pk_{self.schema.name}"
        if pk_idx_name not in self.indexes:
            return False
        old_row_id = self.indexes[pk_idx_name].search(pk_value)
        if old_row_id is None:
            return False

        old_row = self.heap.get(old_row_id)
        if old_row is None:
            return False

        # Merge new data
        updated_row = dict(old_row)
        updated_row.update(new_data)
        # Preserve primary key
        updated_row[self.schema.primary_key] = pk_value

        new_row_id = self.heap.update(old_row_id, updated_row)

        # Update indexes if row moved
        if new_row_id != old_row_id:
            for idx_name, idx in self.indexes.items():
                if idx_name.startswith("pk_"):
                    idx.delete(pk_value)
                    idx.insert(pk_value, new_row_id)
                else:
                    idx_cols = self._index_columns(idx_name)
                    if idx_cols:
                        old_key = self._extract_key(old_row, idx_cols)
                        new_key = self._extract_key(updated_row, idx_cols)
                        if old_key is not None:
                            old_composite = (old_key, old_row_id.page_id, old_row_id.slot_index)
                            idx.delete(old_composite)
                        if new_key is not None:
                            new_composite = (new_key, new_row_id.page_id, new_row_id.slot_index)
                            idx.insert(new_composite, new_row_id)

        return True

    def delete_by_pk(self, pk_value):
        """Delete a row by primary key. Returns True if deleted."""
        if not self.schema.primary_key:
            return False
        pk_idx_name = f"pk_{self.schema.name}"
        if pk_idx_name not in self.indexes:
            return False
        row_id = self.indexes[pk_idx_name].search(pk_value)
        if row_id is None:
            return False

        row = self.heap.get(row_id)

        # Remove from all indexes
        for idx_name, idx in self.indexes.items():
            if idx_name.startswith("pk_"):
                idx.delete(pk_value)
            elif row:
                idx_cols = self._index_columns(idx_name)
                if idx_cols:
                    key = self._extract_key(row, idx_cols)
                    if key is not None:
                        composite = (key, row_id.page_id, row_id.slot_index)
                        idx.delete(composite)

        return self.heap.delete(row_id)

    def scan(self, predicate=None):
        """Full table scan. Yields row dicts. Optional predicate filter."""
        for row_id, row in self.heap.scan():
            if predicate is None or predicate(row):
                yield row

    def index_scan(self, index_name, low=None, high=None,
                   include_low=True, include_high=True):
        """Scan using an index range. Yields row dicts.
        For secondary indexes, keys are composite (value, page_id, slot_idx).
        User provides value-level bounds; we expand to composite bounds."""
        if index_name not in self.indexes:
            raise KeyError(f"No index: {index_name}")
        idx = self.indexes[index_name]
        is_secondary = not index_name.startswith("pk_")
        if is_secondary:
            # Expand value bounds to composite bounds
            comp_low = (low, -2**31, -2**31) if low is not None else None
            comp_high = (high, 2**31, 2**31) if high is not None else None
            for key, row_id in idx.range_scan(comp_low, comp_high, include_low, include_high):
                row = self.heap.get(row_id)
                if row is not None:
                    yield row
        else:
            for key, row_id in idx.range_scan(low, high, include_low, include_high):
                row = self.heap.get(row_id)
                if row is not None:
                    yield row

    def create_index(self, name, columns):
        """Create a secondary index on given columns.
        Secondary index keys are (value, page_id, slot_idx) to ensure uniqueness."""
        if name in self.indexes:
            raise ValueError(f"Index {name} already exists")

        idx = BTreeIndex(self.buffer_pool, name=name)
        self._index_column_map = getattr(self, '_index_column_map', {})
        self._index_column_map[name] = columns if isinstance(columns, (list, tuple)) else [columns]

        # Build index from existing data -- composite key for uniqueness
        for row_id, row in self.heap.scan():
            key = self._extract_key(row, self._index_column_map[name])
            if key is not None:
                composite = (key, row_id.page_id, row_id.slot_index)
                idx.insert(composite, row_id)

        self.indexes[name] = idx
        return idx

    def drop_index(self, name):
        """Drop a secondary index."""
        if name in self.indexes:
            del self.indexes[name]
            if hasattr(self, '_index_column_map') and name in self._index_column_map:
                del self._index_column_map[name]

    def _index_columns(self, idx_name):
        """Get columns for a named index."""
        if not hasattr(self, '_index_column_map'):
            self._index_column_map = {}
        return self._index_column_map.get(idx_name)

    def _extract_key(self, row, columns):
        """Extract index key from row."""
        if len(columns) == 1:
            return row.get(columns[0])
        vals = tuple(row.get(c) for c in columns)
        if all(v is None for v in vals):
            return None
        return vals

    @property
    def row_count(self):
        return self.heap.row_count

    @property
    def page_count(self):
        return self.heap.page_count


# =============================================================================
# Checkpoint / WAL integration
# =============================================================================

class CheckpointManager:
    """Manages checkpoints for crash recovery."""

    def __init__(self, disk_manager):
        self.disk = disk_manager
        self._checkpoints = []  # List of (checkpoint_id, page_snapshots)
        self._next_checkpoint_id = 1

    def create_checkpoint(self):
        """Create a checkpoint of current disk state."""
        cp_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        # Snapshot all page data
        snapshot = bytearray(self.disk._storage)
        self._checkpoints.append((cp_id, snapshot, self.disk._num_pages, list(self.disk._free_pages)))
        return cp_id

    def restore_checkpoint(self, checkpoint_id):
        """Restore disk to a checkpoint."""
        for cp_id, snapshot, num_pages, free_pages in self._checkpoints:
            if cp_id == checkpoint_id:
                self.disk._storage = bytearray(snapshot)
                self.disk._num_pages = num_pages
                self.disk._free_pages = list(free_pages)
                return True
        return False

    def list_checkpoints(self):
        """List all checkpoint IDs."""
        return [cp_id for cp_id, _, _, _ in self._checkpoints]

    def delete_checkpoint(self, checkpoint_id):
        """Delete a checkpoint."""
        self._checkpoints = [(c, s, n, f) for c, s, n, f in self._checkpoints
                             if c != checkpoint_id]


# =============================================================================
# StorageEngine -- top-level API
# =============================================================================

class StorageEngine:
    """
    Top-level storage engine composing:
    - DiskManager for page I/O
    - BufferPool for caching
    - HeapFiles for row storage
    - BTreeIndexes for fast lookups
    - CheckpointManager for crash recovery
    """

    def __init__(self, page_size=PAGE_SIZE, buffer_pool_size=128):
        self.disk = DiskManager(page_size)
        self.buffer_pool = BufferPool(self.disk, buffer_pool_size)
        self.checkpoint_mgr = CheckpointManager(self.disk)
        self._tables = {}  # name -> Table

    def create_table(self, name, columns, primary_key=None, column_types=None):
        """Create a new table."""
        if name in self._tables:
            raise ValueError(f"Table {name} already exists")
        schema = TableSchema(name, columns, primary_key, column_types)
        table = Table(schema, self.buffer_pool)
        self._tables[name] = table
        return table

    def drop_table(self, name):
        """Drop a table."""
        if name not in self._tables:
            raise KeyError(f"Table {name} not found")
        del self._tables[name]

    def get_table(self, name):
        """Get a table by name."""
        return self._tables.get(name)

    def table_names(self):
        """List all table names."""
        return list(self._tables.keys())

    def insert(self, table_name, row):
        """Insert a row into a table."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return table.insert(row)

    def get(self, table_name, pk_value):
        """Get a row by primary key."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return table.get_by_pk(pk_value)

    def update(self, table_name, pk_value, new_data):
        """Update a row by primary key."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return table.update_by_pk(pk_value, new_data)

    def delete(self, table_name, pk_value):
        """Delete a row by primary key."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return table.delete_by_pk(pk_value)

    def scan(self, table_name, predicate=None):
        """Full table scan."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return list(table.scan(predicate))

    def index_scan(self, table_name, index_name, low=None, high=None,
                   include_low=True, include_high=True):
        """Index range scan."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return list(table.index_scan(index_name, low, high, include_low, include_high))

    def create_index(self, table_name, index_name, columns):
        """Create an index on a table."""
        table = self._tables.get(table_name)
        if table is None:
            raise KeyError(f"Table {table_name} not found")
        return table.create_index(index_name, columns)

    def checkpoint(self):
        """Create a checkpoint."""
        self.buffer_pool.flush_all()
        return self.checkpoint_mgr.create_checkpoint()

    def restore(self, checkpoint_id):
        """Restore to a checkpoint. Invalidates buffer pool."""
        self.buffer_pool.flush_all()
        result = self.checkpoint_mgr.restore_checkpoint(checkpoint_id)
        if result:
            # Clear buffer pool (pages may have changed on disk)
            self.buffer_pool._frames.clear()
        return result

    def flush(self):
        """Flush all dirty pages to disk."""
        self.buffer_pool.flush_all()

    def stats(self):
        """Get storage statistics."""
        return {
            'num_tables': len(self._tables),
            'disk_pages': self.disk.num_pages,
            'free_pages': self.disk.num_free_pages,
            'buffer_pool_size': self.buffer_pool.size,
            'buffer_pool_dirty': self.buffer_pool.dirty_count,
            'buffer_pool_hit_rate': self.buffer_pool.hit_rate,
            'tables': {
                name: {
                    'rows': t.row_count,
                    'heap_pages': t.page_count,
                    'indexes': list(t.indexes.keys()),
                }
                for name, t in self._tables.items()
            }
        }


# =============================================================================
# TransactionalStorageEngine -- composes with C212
# =============================================================================

class TransactionalStorageEngine:
    """
    Storage engine with transaction support.
    Composes StorageEngine (C213) with TransactionManager patterns (C212).
    Provides begin/commit/rollback with undo logging.
    """

    def __init__(self, page_size=PAGE_SIZE, buffer_pool_size=128):
        self.engine = StorageEngine(page_size, buffer_pool_size)
        self._transactions = {}  # tx_id -> TxState
        self._next_tx_id = 1
        self._undo_log = {}  # tx_id -> list of undo operations

    def create_table(self, name, columns, primary_key=None):
        """Create table (DDL, outside transactions)."""
        return self.engine.create_table(name, columns, primary_key)

    def begin(self):
        """Begin a transaction. Returns tx_id."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._transactions[tx_id] = 'active'
        self._undo_log[tx_id] = []
        return tx_id

    def insert(self, tx_id, table_name, row):
        """Insert within a transaction."""
        self._check_active(tx_id)
        row_id = self.engine.insert(table_name, row)
        self._undo_log[tx_id].append(('delete_rowid', table_name, row_id))
        return row_id

    def get(self, tx_id, table_name, pk_value):
        """Read within a transaction."""
        self._check_active(tx_id)
        return self.engine.get(table_name, pk_value)

    def update(self, tx_id, table_name, pk_value, new_data):
        """Update within a transaction."""
        self._check_active(tx_id)
        old_row = self.engine.get(table_name, pk_value)
        result = self.engine.update(table_name, pk_value, new_data)
        if result and old_row:
            self._undo_log[tx_id].append(('update', table_name, pk_value, old_row))
        return result

    def delete(self, tx_id, table_name, pk_value):
        """Delete within a transaction."""
        self._check_active(tx_id)
        old_row = self.engine.get(table_name, pk_value)
        result = self.engine.delete(table_name, pk_value)
        if result and old_row:
            self._undo_log[tx_id].append(('insert', table_name, old_row))
        return result

    def scan(self, tx_id, table_name, predicate=None):
        """Scan within a transaction."""
        self._check_active(tx_id)
        return self.engine.scan(table_name, predicate)

    def commit(self, tx_id):
        """Commit a transaction."""
        self._check_active(tx_id)
        self._transactions[tx_id] = 'committed'
        # Flush dirty pages
        self.engine.flush()
        # Discard undo log
        if tx_id in self._undo_log:
            del self._undo_log[tx_id]

    def rollback(self, tx_id):
        """Rollback a transaction by applying undo log in reverse."""
        self._check_active(tx_id)
        self._transactions[tx_id] = 'aborted'

        # Apply undo operations in reverse
        for op in reversed(self._undo_log.get(tx_id, [])):
            if op[0] == 'delete_rowid':
                _, table_name, row_id = op
                table = self.engine.get_table(table_name)
                if table:
                    table.heap.delete(row_id)
                    # Also remove from indexes
                    row = table.heap.get(row_id)  # Already deleted, will be None
            elif op[0] == 'update':
                _, table_name, pk_value, old_row = op
                self.engine.update(table_name, pk_value, old_row)
            elif op[0] == 'insert':
                _, table_name, old_row = op
                self.engine.insert(table_name, old_row)

        if tx_id in self._undo_log:
            del self._undo_log[tx_id]

    def _check_active(self, tx_id):
        if tx_id not in self._transactions:
            raise ValueError(f"Unknown transaction: {tx_id}")
        if self._transactions[tx_id] != 'active':
            raise ValueError(f"Transaction {tx_id} is {self._transactions[tx_id]}")

    def get_table(self, name):
        return self.engine.get_table(name)

    def stats(self):
        s = self.engine.stats()
        s['active_transactions'] = sum(1 for v in self._transactions.values() if v == 'active')
        return s
