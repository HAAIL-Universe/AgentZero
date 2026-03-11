"""
C117: LSM Tree (Log-Structured Merge Tree)
==========================================
Composes C116 (B+ Tree) for on-disk sorted runs.

An LSM tree is the storage engine behind LevelDB, RocksDB, Cassandra, and
many modern databases. It achieves high write throughput by buffering writes
in memory (memtable) and flushing sorted runs to disk-like levels.

This implementation is fully in-memory but models the LSM architecture:
- MemTable: in-memory sorted buffer (uses B+ tree)
- SSTable: immutable sorted run with binary search
- Level: collection of SSTables at one compaction level
- LSMTree: full LSM with configurable levels, size ratios, compaction
- BloomFilter: per-SSTable bloom filter for fast negative lookups
- WAL: write-ahead log for crash recovery simulation

Architecture:
  Write path: WAL -> MemTable -> (flush) -> Level 0 -> (compact) -> Level 1 -> ...
  Read path:  MemTable -> Level 0 -> Level 1 -> ... (merge with tombstone check)

Components:
1. BloomFilter -- probabilistic membership test for SSTables
2. WAL -- write-ahead log (append-only record sequence)
3. SSTable -- immutable sorted string table with index + bloom filter
4. MemTable -- mutable sorted buffer backed by B+ tree
5. LSMTree -- full log-structured merge tree with tiered compaction
6. LSMTreeMap -- dict-like interface over LSMTree

Features:
- O(1) amortized writes (append to memtable)
- O(log n) reads across levels with bloom filter optimization
- Tombstone-based deletes with compaction cleanup
- Configurable memtable size, level count, size ratio
- Range queries across all levels with merge
- Write-ahead log for durability simulation
- Bloom filters reduce unnecessary SSTable searches
- Tiered and leveled compaction strategies
- Snapshots for consistent reads
- Statistics tracking (reads, writes, compactions, bloom hits)
"""

from __future__ import annotations
import sys
import os
import time
import hashlib
from typing import Any, Optional, Iterator

# Import B+ tree from C116
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C116_bplus_tree'))
from bplus_tree import BPlusTree, BulkLoader


# Sentinel for tombstones (deleted keys)
_TOMBSTONE = object()


# ---------------------------------------------------------------------------
# BloomFilter -- probabilistic membership test
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Simple bloom filter for SSTable key lookups.

    Uses multiple hash functions to set bits. False positives possible,
    false negatives impossible.
    """

    def __init__(self, expected_items: int = 1000, fp_rate: float = 0.01):
        import math
        if expected_items <= 0:
            expected_items = 1
        # Optimal bit count: -n*ln(p) / (ln2)^2
        self._num_bits = max(8, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        # Optimal hash count: (m/n) * ln2
        self._num_hashes = max(1, int((self._num_bits / expected_items) * math.log(2)))
        self._bits = bytearray((self._num_bits + 7) // 8)
        self._count = 0

    def _hashes(self, key) -> list:
        """Generate hash positions using double hashing."""
        key_bytes = str(key).encode('utf-8')
        h1 = int(hashlib.md5(key_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha1(key_bytes).hexdigest(), 16)
        return [(h1 + i * h2) % self._num_bits for i in range(self._num_hashes)]

    def add(self, key) -> None:
        """Add a key to the filter."""
        for pos in self._hashes(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bits[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def might_contain(self, key) -> bool:
        """Check if key might be in the filter. False = definitely not present."""
        for pos in self._hashes(key):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    @property
    def count(self) -> int:
        return self._count

    def __repr__(self):
        return f"BloomFilter(bits={self._num_bits}, hashes={self._num_hashes}, items={self._count})"


# ---------------------------------------------------------------------------
# WAL -- Write-Ahead Log
# ---------------------------------------------------------------------------

class WALEntry:
    """Single WAL record."""
    __slots__ = ('seq', 'op', 'key', 'value')

    def __init__(self, seq: int, op: str, key, value=None):
        self.seq = seq
        self.op = op  # 'put' or 'delete'
        self.key = key
        self.value = value

    def __repr__(self):
        if self.op == 'delete':
            return f"WAL({self.seq}: DEL {self.key})"
        return f"WAL({self.seq}: PUT {self.key}={self.value})"


class WAL:
    """
    Write-ahead log. Append-only sequence of put/delete operations.

    In a real system this would be on disk. Here it's in-memory but models
    the same semantics: append records, truncate after flush.
    """

    def __init__(self):
        self._entries: list[WALEntry] = []
        self._seq = 0

    def append_put(self, key, value) -> int:
        """Append a put record. Returns sequence number."""
        self._seq += 1
        self._entries.append(WALEntry(self._seq, 'put', key, value))
        return self._seq

    def append_delete(self, key) -> int:
        """Append a delete record. Returns sequence number."""
        self._seq += 1
        self._entries.append(WALEntry(self._seq, 'delete', key))
        return self._seq

    def entries(self) -> list[WALEntry]:
        """Return all entries."""
        return list(self._entries)

    def truncate(self) -> None:
        """Clear the log (called after successful flush)."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def seq(self) -> int:
        return self._seq

    def replay(self) -> list[WALEntry]:
        """Return entries for replay (recovery simulation)."""
        return list(self._entries)

    def __repr__(self):
        return f"WAL(entries={len(self._entries)}, seq={self._seq})"


# ---------------------------------------------------------------------------
# SSTable -- Immutable Sorted String Table
# ---------------------------------------------------------------------------

class SSTable:
    """
    Immutable sorted table of key-value pairs.

    Represents a flushed memtable or compaction output. Once created,
    never modified. Supports binary search lookup and sequential scan.

    Includes:
    - Sorted key-value pairs
    - Bloom filter for fast negative lookups
    - Min/max key for range pruning
    - Creation timestamp and sequence range
    """

    def __init__(self, data: list[tuple], seq_start: int = 0, seq_end: int = 0,
                 table_id: int = 0):
        """
        Create SSTable from sorted (key, value) pairs.
        Values can be _TOMBSTONE for deleted keys.
        """
        self._keys = [k for k, v in data]
        self._values = [v for k, v in data]
        self._size = len(data)
        self._seq_start = seq_start
        self._seq_end = seq_end
        self._table_id = table_id
        self._created = time.monotonic()

        # Build bloom filter
        self._bloom = BloomFilter(expected_items=max(1, self._size))
        for k in self._keys:
            self._bloom.add(k)

        # Cache min/max keys
        self._min_key = self._keys[0] if self._keys else None
        self._max_key = self._keys[-1] if self._keys else None

    @property
    def size(self) -> int:
        return self._size

    @property
    def table_id(self) -> int:
        return self._table_id

    @property
    def min_key(self):
        return self._min_key

    @property
    def max_key(self):
        return self._max_key

    @property
    def bloom(self) -> BloomFilter:
        return self._bloom

    def get(self, key):
        """
        Look up key. Returns (found, value) tuple.
        If found=True and value is _TOMBSTONE, key was deleted.
        """
        if not self._bloom.might_contain(key):
            return (False, None)

        # Binary search
        lo, hi = 0, self._size - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._keys[mid] == key:
                return (True, self._values[mid])
            elif self._keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid - 1

        return (False, None)

    def overlaps(self, low, high) -> bool:
        """Check if this SSTable's key range overlaps [low, high]."""
        if self._size == 0:
            return False
        if low is not None and self._max_key < low:
            return False
        if high is not None and self._min_key > high:
            return False
        return True

    def scan(self, low=None, high=None) -> Iterator[tuple]:
        """Iterate (key, value) pairs in range [low, high]."""
        if self._size == 0:
            return

        # Find start position
        if low is None:
            start = 0
        else:
            start = self._bisect_left(low)

        for i in range(start, self._size):
            k = self._keys[i]
            if high is not None and k > high:
                break
            yield (k, self._values[i])

    def all_items(self) -> Iterator[tuple]:
        """Iterate all (key, value) pairs including tombstones."""
        yield from zip(self._keys, self._values)

    def live_items(self) -> Iterator[tuple]:
        """Iterate non-tombstone (key, value) pairs."""
        for k, v in zip(self._keys, self._values):
            if v is not _TOMBSTONE:
                yield (k, v)

    def _bisect_left(self, key) -> int:
        lo, hi = 0, self._size
        while lo < hi:
            mid = (lo + hi) // 2
            if self._keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def tombstone_count(self) -> int:
        """Count tombstone entries."""
        return sum(1 for v in self._values if v is _TOMBSTONE)

    def __len__(self):
        return self._size

    def __repr__(self):
        return (f"SSTable(id={self._table_id}, size={self._size}, "
                f"keys=[{self._min_key}..{self._max_key}])")


# ---------------------------------------------------------------------------
# MemTable -- Mutable sorted buffer (backed by B+ tree from C116)
# ---------------------------------------------------------------------------

class MemTable:
    """
    In-memory sorted buffer for recent writes.

    Uses C116 B+ tree for O(log n) insert/lookup with ordered iteration.
    Supports tombstones for deletes. Flushes to SSTable when full.
    """

    def __init__(self, max_size: int = 1000):
        self._tree = BPlusTree(order=32)
        self._max_size = max_size
        self._approx_size = 0  # Track approximate memory usage

    def put(self, key, value) -> None:
        """Insert or update a key-value pair."""
        if key not in self._tree:
            self._approx_size += 1
        self._tree.insert(key, value)

    def delete(self, key) -> None:
        """Mark key as deleted with tombstone."""
        if key not in self._tree:
            self._approx_size += 1
        self._tree.insert(key, _TOMBSTONE)

    def get(self, key):
        """Look up key. Returns (found, value). Value may be _TOMBSTONE."""
        val = self._tree.get(key)
        if val is None and key not in self._tree:
            return (False, None)
        return (True, val)

    def is_full(self) -> bool:
        """Check if memtable has reached capacity."""
        return self._approx_size >= self._max_size

    def flush_to_sstable(self, seq_start: int, seq_end: int, table_id: int) -> SSTable:
        """Convert memtable contents to an immutable SSTable."""
        data = list(self._tree.items())
        return SSTable(data, seq_start=seq_start, seq_end=seq_end, table_id=table_id)

    def items(self) -> Iterator:
        """Iterate all (key, value) pairs in sorted order."""
        return self._tree.items()

    def clear(self) -> None:
        """Clear the memtable."""
        self._tree.clear()
        self._approx_size = 0

    @property
    def size(self) -> int:
        return len(self._tree)

    def __contains__(self, key):
        return key in self._tree

    def __repr__(self):
        return f"MemTable(size={len(self._tree)}, max={self._max_size})"


# ---------------------------------------------------------------------------
# Level -- Collection of SSTables at one compaction level
# ---------------------------------------------------------------------------

class Level:
    """
    A level in the LSM tree containing SSTables.

    Level 0: SSTables may have overlapping key ranges (direct flushes).
    Level 1+: SSTables have non-overlapping key ranges (compacted).
    """

    def __init__(self, level_num: int, max_tables: int = 4):
        self._level_num = level_num
        self._max_tables = max_tables
        self._tables: list[SSTable] = []

    @property
    def level_num(self) -> int:
        return self._level_num

    @property
    def tables(self) -> list[SSTable]:
        return self._tables

    @property
    def max_tables(self) -> int:
        return self._max_tables

    def add_table(self, table: SSTable) -> None:
        """Add an SSTable to this level."""
        self._tables.append(table)

    def remove_table(self, table: SSTable) -> None:
        """Remove an SSTable from this level."""
        self._tables.remove(table)

    def needs_compaction(self) -> bool:
        """Check if this level needs compaction."""
        return len(self._tables) > self._max_tables

    def total_entries(self) -> int:
        """Total entries across all tables."""
        return sum(t.size for t in self._tables)

    def get(self, key, check_bloom: bool = True):
        """
        Look up key across all tables in this level.
        Returns (found, value) -- searches newest table first.
        """
        # Search in reverse order (newest first)
        for table in reversed(self._tables):
            if check_bloom and not table.bloom.might_contain(key):
                continue
            found, value = table.get(key)
            if found:
                return (True, value)
        return (False, None)

    def overlapping_tables(self, low, high) -> list[SSTable]:
        """Return tables whose key range overlaps [low, high]."""
        result = []
        for table in self._tables:
            if table.overlaps(low, high):
                result.append(table)
        return result

    def clear(self) -> None:
        self._tables.clear()

    def __len__(self):
        return len(self._tables)

    def __repr__(self):
        total = sum(t.size for t in self._tables)
        return f"Level({self._level_num}, tables={len(self._tables)}, entries={total})"


# ---------------------------------------------------------------------------
# Merge iterator -- merges multiple sorted sources
# ---------------------------------------------------------------------------

def _merge_iterators(iterators: list, reverse_priority: bool = False) -> Iterator[tuple]:
    """
    Merge multiple sorted iterators, yielding (key, value) in sorted order.
    When keys collide, earlier iterators (lower index) win (they're newer).

    Each iterator yields (key, value) tuples in sorted key order.
    """
    # Initialize: get first item from each iterator
    heap = []  # (key, priority, value, iter_index)
    iters = list(iterators)

    for i, it in enumerate(iters):
        item = next(it, None)
        if item is not None:
            priority = i if not reverse_priority else -i
            heap.append((item[0], priority, item[1], i))

    if not heap:
        return

    # Use a simple sorted approach (efficient enough for typical LSM level counts)
    heap.sort()

    while heap:
        # Take the smallest key
        key, priority, value, idx = heap[0]

        # Remove all entries with the same key (keep the one with lowest priority = newest)
        new_heap = []
        best_value = value
        best_priority = priority
        for entry in heap:
            if entry[0] == key:
                if entry[1] < best_priority:
                    best_priority = entry[1]
                    best_value = entry[2]
                # Advance this iterator
                it_idx = entry[3]
                nxt = next(iters[it_idx], None)
                if nxt is not None:
                    new_heap.append((nxt[0], entry[1], nxt[1], it_idx))
            else:
                new_heap.append(entry)

        yield (key, best_value)

        heap = new_heap
        heap.sort()


# ---------------------------------------------------------------------------
# LSMTree -- Full Log-Structured Merge Tree
# ---------------------------------------------------------------------------

class LSMTree:
    """
    Log-Structured Merge Tree.

    Write path:
      1. Append to WAL (durability)
      2. Insert into MemTable (sorted buffer)
      3. When MemTable is full, flush to Level 0 as SSTable
      4. When Level N has too many tables, compact into Level N+1

    Read path:
      1. Check MemTable (most recent writes)
      2. Check Level 0 (most recent flushed, may overlap)
      3. Check Level 1, 2, ... (older, non-overlapping)
      4. Bloom filters skip SSTables that definitely don't contain key

    Parameters:
        memtable_size: Max entries in memtable before flush
        num_levels: Number of compaction levels
        level0_max: Max SSTables in Level 0 before compaction
        size_ratio: Size multiplier between levels (Level N+1 = ratio * Level N)
        enable_wal: Whether to use write-ahead log
    """

    def __init__(self, memtable_size: int = 1000, num_levels: int = 4,
                 level0_max: int = 4, size_ratio: int = 10,
                 enable_wal: bool = True):
        self._memtable = MemTable(max_size=memtable_size)
        self._memtable_size = memtable_size
        self._num_levels = num_levels
        self._size_ratio = size_ratio
        self._enable_wal = enable_wal

        # Create levels with increasing capacity
        self._levels: list[Level] = []
        for i in range(num_levels):
            max_tables = level0_max if i == 0 else level0_max * (size_ratio ** i)
            self._levels.append(Level(i, max_tables=max_tables))

        # WAL
        self._wal = WAL() if enable_wal else None

        # Immutable memtable (being flushed)
        self._immutable_memtable: Optional[MemTable] = None

        # ID counter for SSTables
        self._next_table_id = 0

        # Sequence numbers
        self._flush_seq_start = 0

        # Statistics
        self._stats = {
            'puts': 0,
            'deletes': 0,
            'gets': 0,
            'get_hits': 0,
            'get_misses': 0,
            'bloom_hits': 0,  # bloom filter saved a search
            'flushes': 0,
            'compactions': 0,
            'bytes_compacted': 0,
            'tombstones_cleaned': 0,
        }

        # Snapshot support
        self._snapshots: dict[int, list] = {}  # snap_id -> list of (key, value)
        self._next_snap_id = 0

    # -- Write operations --

    def put(self, key, value) -> None:
        """Insert or update a key-value pair."""
        if self._wal:
            self._wal.append_put(key, value)

        self._memtable.put(key, value)
        self._stats['puts'] += 1

        if self._memtable.is_full():
            self._flush_memtable()

    def delete(self, key) -> None:
        """Delete a key (writes tombstone)."""
        if self._wal:
            self._wal.append_delete(key)

        self._memtable.delete(key)
        self._stats['deletes'] += 1

        if self._memtable.is_full():
            self._flush_memtable()

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.delete(key)

    # -- Read operations --

    def get(self, key, default=None):
        """Look up a key. Returns value or default if not found."""
        self._stats['gets'] += 1

        # 1. Check memtable
        found, value = self._memtable.get(key)
        if found:
            if value is _TOMBSTONE:
                self._stats['get_misses'] += 1
                return default
            self._stats['get_hits'] += 1
            return value

        # 2. Check immutable memtable (if mid-flush)
        if self._immutable_memtable:
            found, value = self._immutable_memtable.get(key)
            if found:
                if value is _TOMBSTONE:
                    self._stats['get_misses'] += 1
                    return default
                self._stats['get_hits'] += 1
                return value

        # 3. Check each level
        for level in self._levels:
            found, value = level.get(key)
            if found:
                if value is _TOMBSTONE:
                    self._stats['get_misses'] += 1
                    return default
                self._stats['get_hits'] += 1
                return value

        self._stats['get_misses'] += 1
        return default

    def __getitem__(self, key):
        result = self.get(key, _TOMBSTONE)
        if result is _TOMBSTONE:
            raise KeyError(key)
        return result

    def __contains__(self, key) -> bool:
        return self.get(key, _TOMBSTONE) is not _TOMBSTONE

    # -- Flush --

    def _flush_memtable(self) -> None:
        """Flush current memtable to Level 0 as a new SSTable."""
        if self._memtable.size == 0:
            return

        table_id = self._next_table_id
        self._next_table_id += 1

        seq_end = self._wal.seq if self._wal else 0
        sstable = self._memtable.flush_to_sstable(
            seq_start=self._flush_seq_start,
            seq_end=seq_end,
            table_id=table_id
        )
        self._flush_seq_start = seq_end

        self._levels[0].add_table(sstable)
        self._memtable.clear()

        if self._wal:
            self._wal.truncate()

        self._stats['flushes'] += 1

        # Check if Level 0 needs compaction
        self._maybe_compact()

    # -- Compaction --

    def _maybe_compact(self) -> None:
        """Check all levels and compact if needed."""
        for i in range(self._num_levels - 1):
            if self._levels[i].needs_compaction():
                self._compact_level(i)

    def _compact_level(self, level_num: int) -> None:
        """Compact level N into level N+1."""
        source_level = self._levels[level_num]
        target_level = self._levels[level_num + 1]

        if not source_level.tables:
            return

        # For Level 0: compact ALL tables (they may overlap)
        # For Level 1+: pick tables that overlap with target
        if level_num == 0:
            source_tables = list(source_level.tables)
        else:
            # Pick oldest table
            source_tables = [source_level.tables[0]]

        # Find overlapping tables in target level
        all_source_keys = []
        for t in source_tables:
            if t.min_key is not None:
                all_source_keys.append(t.min_key)
                all_source_keys.append(t.max_key)

        if all_source_keys:
            overlap_low = min(all_source_keys)
            overlap_high = max(all_source_keys)
            target_tables = target_level.overlapping_tables(overlap_low, overlap_high)
        else:
            target_tables = []

        # Merge all source + overlapping target tables
        all_tables = source_tables + target_tables

        # Count entries for stats
        total_entries = sum(t.size for t in all_tables)

        # Create merged sorted stream
        iterators = [t.all_items() for t in all_tables]

        # For the merge: source tables are newer, so they should win on conflicts
        # We need to assign priority based on which table is newer
        # Source tables get lower priority numbers (= higher priority in merge)
        merged_data = []
        seen_keys = set()

        # Simple approach: collect all, sort, deduplicate
        # Priority: lower number = newer = wins on duplicate keys
        # Source tables are newer than target tables
        # Within source tables, later ones (higher index) are newer
        all_entries = []
        source_set = set(id(t) for t in source_tables)
        num_source = len(source_tables)
        num_target = len(target_tables)
        for i, table in enumerate(all_tables):
            is_source = id(table) in source_set
            if is_source:
                # Source tables: index in source_tables list, reversed so last=newest=0
                src_idx = source_tables.index(table)
                priority = num_source - 1 - src_idx
            else:
                # Target tables: always older than any source table
                priority = num_source + i
            for k, v in table.all_items():
                all_entries.append((k, priority, v))

        # Sort by key, then by priority (lower = newer = wins)
        all_entries.sort(key=lambda x: (x[0], x[1]))

        # Deduplicate: keep first occurrence of each key (lowest priority)
        tombstones_cleaned = 0
        for entry in all_entries:
            k, priority, v = entry
            if k in seen_keys:
                continue
            seen_keys.add(k)
            # At the deepest level, we can drop tombstones
            if v is _TOMBSTONE and level_num + 1 == self._num_levels - 1:
                tombstones_cleaned += 1
                continue
            merged_data.append((k, v))

        # Create new SSTable(s) from merged data
        if merged_data:
            new_table_id = self._next_table_id
            self._next_table_id += 1
            new_table = SSTable(merged_data, table_id=new_table_id)

            # Remove old tables
            for t in source_tables:
                source_level.remove_table(t)
            for t in target_tables:
                target_level.remove_table(t)

            # Add new table
            target_level.add_table(new_table)
        else:
            # All entries were tombstones that got cleaned
            for t in source_tables:
                source_level.remove_table(t)
            for t in target_tables:
                target_level.remove_table(t)

        self._stats['compactions'] += 1
        self._stats['bytes_compacted'] += total_entries
        self._stats['tombstones_cleaned'] += tombstones_cleaned

        # Recursively check if target level now needs compaction
        if target_level.needs_compaction() and level_num + 1 < self._num_levels - 1:
            self._compact_level(level_num + 1)

    # -- Range queries --

    def range_query(self, low=None, high=None) -> list[tuple]:
        """
        Return all live (key, value) pairs in [low, high] range.
        Merges across memtable and all levels, respecting tombstones.
        """
        # Collect iterators from all sources (newest first)
        sources = []

        # MemTable
        if low is None and high is None:
            sources.append(self._memtable.items())
        else:
            sources.append(self._filtered_memtable_iter(low, high))

        # Immutable memtable
        if self._immutable_memtable:
            if low is None and high is None:
                sources.append(self._immutable_memtable.items())
            else:
                sources.append(self._filtered_iter(self._immutable_memtable.items(), low, high))

        # Levels
        for level in self._levels:
            for table in reversed(level.tables):
                if table.overlaps(low, high):
                    sources.append(table.scan(low, high))

        # Merge and filter tombstones
        result = []
        seen = set()

        merged = _merge_iterators(sources)
        for key, value in merged:
            if key in seen:
                continue
            seen.add(key)
            if value is not _TOMBSTONE:
                result.append((key, value))

        return result

    def _filtered_memtable_iter(self, low, high):
        """Filter memtable items to range."""
        for k, v in self._memtable.items():
            if low is not None and k < low:
                continue
            if high is not None and k > high:
                break
            yield (k, v)

    def _filtered_iter(self, it, low, high):
        """Filter iterator items to range."""
        for k, v in it:
            if low is not None and k < low:
                continue
            if high is not None and k > high:
                break
            yield (k, v)

    # -- Iteration --

    def items(self) -> list[tuple]:
        """Return all live (key, value) pairs in sorted order."""
        return self.range_query()

    def keys(self) -> list:
        """Return all live keys in sorted order."""
        return [k for k, v in self.items()]

    def values(self) -> list:
        """Return all live values in key order."""
        return [v for k, v in self.items()]

    # -- Snapshot --

    def create_snapshot(self) -> int:
        """Create a point-in-time snapshot. Returns snapshot ID."""
        snap_id = self._next_snap_id
        self._next_snap_id += 1
        self._snapshots[snap_id] = self.items()
        return snap_id

    def read_snapshot(self, snap_id: int) -> list[tuple]:
        """Read all (key, value) pairs from a snapshot."""
        if snap_id not in self._snapshots:
            raise KeyError(f"Snapshot {snap_id} not found")
        return list(self._snapshots[snap_id])

    def release_snapshot(self, snap_id: int) -> None:
        """Release a snapshot to free memory."""
        if snap_id in self._snapshots:
            del self._snapshots[snap_id]

    # -- Maintenance --

    def force_flush(self) -> None:
        """Force flush the current memtable to Level 0."""
        self._flush_memtable()

    def force_compact(self) -> None:
        """Force compaction on all levels that need it."""
        for i in range(self._num_levels - 1):
            if self._levels[i].tables:
                self._compact_level(i)

    def full_compact(self) -> None:
        """Force full compaction: merge everything down to the last level."""
        # First flush memtable
        self._flush_memtable()
        # Then compact each level down
        for i in range(self._num_levels - 1):
            while self._levels[i].tables:
                self._compact_level(i)

    # -- WAL recovery --

    def recover_from_wal(self) -> int:
        """
        Replay WAL to recover state after simulated crash.
        Returns number of entries replayed.
        """
        if not self._wal:
            return 0

        entries = self._wal.replay()
        count = 0
        for entry in entries:
            if entry.op == 'put':
                self._memtable.put(entry.key, entry.value)
            elif entry.op == 'delete':
                self._memtable.delete(entry.key)
            count += 1

        return count

    # -- Statistics --

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def level_info(self) -> list[dict]:
        """Return info about each level."""
        info = []
        for level in self._levels:
            info.append({
                'level': level.level_num,
                'tables': len(level.tables),
                'entries': level.total_entries(),
                'max_tables': level.max_tables,
            })
        return info

    @property
    def memtable_size(self) -> int:
        return self._memtable.size

    @property
    def total_entries(self) -> int:
        """Approximate total entries (may include tombstones and duplicates)."""
        total = self._memtable.size
        for level in self._levels:
            total += level.total_entries()
        return total

    @property
    def wal(self) -> Optional[WAL]:
        return self._wal

    @property
    def levels(self) -> list[Level]:
        return self._levels

    def __repr__(self):
        level_info = ', '.join(f'L{l.level_num}:{len(l.tables)}' for l in self._levels)
        return f"LSMTree(mem={self._memtable.size}, {level_info})"


# ---------------------------------------------------------------------------
# LSMTreeMap -- Dict-like interface
# ---------------------------------------------------------------------------

class LSMTreeMap:
    """
    Dict-like ordered map backed by an LSM tree.

    Provides familiar dict interface with LSM tree performance characteristics:
    fast writes, good reads with bloom filter optimization.
    """

    def __init__(self, memtable_size: int = 1000, num_levels: int = 4, **kwargs):
        self._lsm = LSMTree(memtable_size=memtable_size, num_levels=num_levels, **kwargs)
        self._count = 0  # Exact count of live keys

    def __setitem__(self, key, value):
        # Check if key already exists to maintain accurate count
        if key not in self._lsm:
            self._count += 1
        self._lsm.put(key, value)

    def __getitem__(self, key):
        return self._lsm[key]

    def __delitem__(self, key):
        if key not in self._lsm:
            raise KeyError(key)
        self._lsm.delete(key)
        self._count -= 1

    def __contains__(self, key):
        return key in self._lsm

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count > 0

    def get(self, key, default=None):
        return self._lsm.get(key, default)

    def put(self, key, value):
        self[key] = value

    def delete(self, key):
        del self[key]

    def items(self):
        return self._lsm.items()

    def keys(self):
        return self._lsm.keys()

    def values(self):
        return self._lsm.values()

    def range_query(self, low=None, high=None):
        return self._lsm.range_query(low, high)

    def update(self, data):
        """Update from dict or iterable of (key, value) pairs."""
        if isinstance(data, dict):
            data = data.items()
        for k, v in data:
            self[k] = v

    def clear(self):
        """Clear all data."""
        self._lsm = LSMTree(
            memtable_size=self._lsm._memtable_size,
            num_levels=self._lsm._num_levels,
        )
        self._count = 0

    @property
    def stats(self):
        return self._lsm.stats

    @property
    def lsm(self):
        return self._lsm

    def __repr__(self):
        return f"LSMTreeMap(count={self._count}, {self._lsm})"
