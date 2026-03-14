"""
C244: Buffer Pool Manager

A complete buffer pool (page cache) for database storage:
- Fixed-size page frames with configurable pool size
- Page read/write with pin/unpin reference counting
- Multiple eviction policies: LRU, Clock (second-chance), LRU-K
- Dirty page tracking and flush management
- Write-back on eviction (lazy writes)
- Sequential scan prefetch (read-ahead)
- Page replacement with victim selection
- Buffer pool statistics (hit rate, evictions, flushes)
- Thread-safe concurrent access
- Free list management for empty frames
- Disk manager abstraction for I/O

Domain: Database Internals
Standalone implementation -- no external dependencies.
"""

from enum import Enum, auto
from typing import Any, Optional, Dict, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, deque
import threading
import time


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

PAGE_SIZE = 4096  # 4KB pages

@dataclass
class Page:
    """A database page."""
    page_id: int
    data: bytearray

    def __init__(self, page_id: int, size: int = PAGE_SIZE):
        self.page_id = page_id
        self.data = bytearray(size)

    def read(self, offset: int, length: int) -> bytes:
        return bytes(self.data[offset:offset + length])

    def write(self, offset: int, data: bytes):
        self.data[offset:offset + len(data)] = data

    def clear(self):
        self.data[:] = b'\x00' * len(self.data)

    def __repr__(self):
        return f"Page(id={self.page_id}, size={len(self.data)})"


# ---------------------------------------------------------------------------
# Disk Manager
# ---------------------------------------------------------------------------

class DiskManager:
    """Abstraction for disk I/O. Uses in-memory dict for testing."""

    def __init__(self, page_size: int = PAGE_SIZE):
        self.page_size = page_size
        self._pages: Dict[int, bytes] = {}
        self._next_page_id = 0
        self.read_count = 0
        self.write_count = 0
        self._lock = threading.Lock()

    def allocate_page(self) -> int:
        with self._lock:
            page_id = self._next_page_id
            self._next_page_id += 1
            self._pages[page_id] = b'\x00' * self.page_size
            return page_id

    def read_page(self, page_id: int) -> bytes:
        with self._lock:
            self.read_count += 1
            if page_id not in self._pages:
                raise ValueError(f"Page {page_id} does not exist on disk")
            return self._pages[page_id]

    def write_page(self, page_id: int, data: bytes):
        with self._lock:
            self.write_count += 1
            if len(data) != self.page_size:
                raise ValueError(f"Data size {len(data)} != page size {self.page_size}")
            self._pages[page_id] = bytes(data)

    def page_exists(self, page_id: int) -> bool:
        return page_id in self._pages

    def num_pages(self) -> int:
        return len(self._pages)

    def reset_counters(self):
        self.read_count = 0
        self.write_count = 0


# ---------------------------------------------------------------------------
# Frame Descriptor
# ---------------------------------------------------------------------------

@dataclass
class FrameDescriptor:
    """Metadata for a buffer frame."""
    frame_id: int
    page_id: int = -1
    pin_count: int = 0
    is_dirty: bool = False
    page: Optional[Page] = None

    # For Clock algorithm
    reference_bit: bool = False

    # For LRU-K
    access_history: List[float] = field(default_factory=list)

    # Timestamp for LRU
    last_access: float = 0.0

    def is_free(self) -> bool:
        return self.page_id == -1

    def is_pinned(self) -> bool:
        return self.pin_count > 0

    def reset(self):
        self.page_id = -1
        self.pin_count = 0
        self.is_dirty = False
        self.page = None
        self.reference_bit = False
        self.access_history = []
        self.last_access = 0.0


# ---------------------------------------------------------------------------
# Eviction Policies
# ---------------------------------------------------------------------------

class EvictionPolicy(Enum):
    LRU = 'LRU'
    CLOCK = 'Clock'
    LRU_K = 'LRU-K'


class Replacer:
    """Base class for page replacement algorithms."""

    def victim(self, frames: List[FrameDescriptor]) -> Optional[int]:
        """Select a victim frame for eviction. Returns frame_id or None."""
        raise NotImplementedError

    def pin(self, frame_id: int):
        """Mark frame as pinned (not evictable)."""
        pass

    def unpin(self, frame_id: int):
        """Mark frame as unpinned (evictable)."""
        pass

    def access(self, frame_id: int):
        """Record an access to a frame."""
        pass


class LRUReplacer(Replacer):
    """Least Recently Used page replacement."""

    def __init__(self):
        self._order: OrderedDict[int, bool] = OrderedDict()
        self._pinned: Set[int] = set()

    def victim(self, frames: List[FrameDescriptor]) -> Optional[int]:
        for frame_id in self._order:
            if frame_id not in self._pinned:
                del self._order[frame_id]
                return frame_id
        return None

    def pin(self, frame_id: int):
        self._pinned.add(frame_id)

    def unpin(self, frame_id: int):
        self._pinned.discard(frame_id)
        if frame_id not in self._order:
            self._order[frame_id] = True

    def access(self, frame_id: int):
        if frame_id in self._order:
            self._order.move_to_end(frame_id)
        else:
            self._order[frame_id] = True


class ClockReplacer(Replacer):
    """Clock (second-chance) page replacement."""

    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self._clock_hand = 0
        self._in_replacer: Set[int] = set()
        self._ref_bits: Dict[int, bool] = {}

    def victim(self, frames: List[FrameDescriptor]) -> Optional[int]:
        if not self._in_replacer:
            return None

        max_iterations = 2 * self.num_frames
        for _ in range(max_iterations):
            frame_id = self._clock_hand
            self._clock_hand = (self._clock_hand + 1) % self.num_frames

            if frame_id not in self._in_replacer:
                continue

            if self._ref_bits.get(frame_id, False):
                self._ref_bits[frame_id] = False
            else:
                self._in_replacer.discard(frame_id)
                return frame_id

        return None

    def pin(self, frame_id: int):
        self._in_replacer.discard(frame_id)

    def unpin(self, frame_id: int):
        self._in_replacer.add(frame_id)

    def access(self, frame_id: int):
        self._ref_bits[frame_id] = True


class LRUKReplacer(Replacer):
    """LRU-K page replacement.

    Evicts the page whose K-th most recent access is furthest in the past.
    Pages with fewer than K accesses are evicted first (using FIFO among them).
    """

    def __init__(self, k: int = 2):
        self.k = k
        self._history: Dict[int, List[float]] = {}  # frame_id -> access timestamps
        self._pinned: Set[int] = set()
        self._timestamp = 0

    def victim(self, frames: List[FrameDescriptor]) -> Optional[int]:
        # Separate into two groups: < K accesses and >= K accesses
        less_than_k = []
        at_least_k = []

        for frame_id, history in self._history.items():
            if frame_id in self._pinned:
                continue
            if len(history) < self.k:
                less_than_k.append((frame_id, history[0] if history else float('inf')))
            else:
                # K-th most recent access (from the end)
                kth_access = history[-self.k]
                at_least_k.append((frame_id, kth_access))

        # Evict pages with < K accesses first (FIFO: earliest first access)
        if less_than_k:
            less_than_k.sort(key=lambda x: x[1])
            victim_id = less_than_k[0][0]
            del self._history[victim_id]
            return victim_id

        # Among pages with >= K accesses, evict the one with oldest K-th access
        if at_least_k:
            at_least_k.sort(key=lambda x: x[1])
            victim_id = at_least_k[0][0]
            del self._history[victim_id]
            return victim_id

        return None

    def pin(self, frame_id: int):
        self._pinned.add(frame_id)

    def unpin(self, frame_id: int):
        self._pinned.discard(frame_id)

    def access(self, frame_id: int):
        self._timestamp += 1
        if frame_id not in self._history:
            self._history[frame_id] = []
        self._history[frame_id].append(self._timestamp)
        # Keep only last K+1 entries for memory efficiency
        if len(self._history[frame_id]) > self.k + 1:
            self._history[frame_id] = self._history[frame_id][-(self.k + 1):]

    def remove(self, frame_id: int):
        """Remove a frame from tracking."""
        self._history.pop(frame_id, None)
        self._pinned.discard(frame_id)


# ---------------------------------------------------------------------------
# Buffer Pool Statistics
# ---------------------------------------------------------------------------

@dataclass
class BufferPoolStats:
    """Statistics for buffer pool performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    flushes: int = 0
    dirty_writes: int = 0
    prefetch_requests: int = 0
    prefetch_hits: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.flushes = 0
        self.dirty_writes = 0
        self.prefetch_requests = 0
        self.prefetch_hits = 0

    def __repr__(self):
        return (f"Stats(reqs={self.total_requests}, hit_rate={self.hit_rate:.2%}, "
                f"evictions={self.evictions}, flushes={self.flushes})")


# ---------------------------------------------------------------------------
# Buffer Pool Manager
# ---------------------------------------------------------------------------

class BufferPoolManager:
    """Manages a fixed-size buffer pool of page frames.

    Supports:
    - Page fetch (pin) with automatic disk I/O
    - Page unpin with dirty marking
    - Multiple eviction policies (LRU, Clock, LRU-K)
    - Dirty page flush
    - Sequential prefetch
    - Thread-safe operations
    """

    def __init__(self, pool_size: int, disk_manager: DiskManager,
                 policy: EvictionPolicy = EvictionPolicy.LRU,
                 k: int = 2):
        if pool_size <= 0:
            raise ValueError("Pool size must be positive")

        self.pool_size = pool_size
        self.disk = disk_manager
        self.policy = policy
        self.stats = BufferPoolStats()

        # Initialize frames
        self._frames: List[FrameDescriptor] = [
            FrameDescriptor(frame_id=i) for i in range(pool_size)
        ]

        # Page table: page_id -> frame_id
        self._page_table: Dict[int, int] = {}

        # Free list
        self._free_list: deque = deque(range(pool_size))

        # Create replacer
        self._replacer = self._create_replacer(policy, k)

        # Lock for thread safety
        self._lock = threading.Lock()

        # Prefetch state
        self._prefetch_distance = 4  # pages ahead to prefetch

    def _create_replacer(self, policy: EvictionPolicy, k: int) -> Replacer:
        if policy == EvictionPolicy.LRU:
            return LRUReplacer()
        elif policy == EvictionPolicy.CLOCK:
            return ClockReplacer(self.pool_size)
        elif policy == EvictionPolicy.LRU_K:
            return LRUKReplacer(k)
        else:
            raise ValueError(f"Unknown eviction policy: {policy}")

    def fetch_page(self, page_id: int) -> Optional[Page]:
        """Fetch a page into the buffer pool and pin it.

        Returns the page, or None if no frames available.
        """
        with self._lock:
            return self._fetch_page_internal(page_id)

    def _fetch_page_internal(self, page_id: int) -> Optional[Page]:
        # Check if page is already in the buffer pool
        if page_id in self._page_table:
            frame_id = self._page_table[page_id]
            frame = self._frames[frame_id]
            frame.pin_count += 1
            frame.reference_bit = True
            self._replacer.pin(frame_id)
            self._replacer.access(frame_id)
            self.stats.hits += 1
            return frame.page

        # Page not in pool -- need to bring it in
        self.stats.misses += 1

        # Find a free frame or evict one
        frame_id = self._get_free_frame()
        if frame_id is None:
            return None  # No frames available

        frame = self._frames[frame_id]

        # Read page from disk
        if not self.disk.page_exists(page_id):
            raise ValueError(f"Page {page_id} does not exist")

        data = self.disk.read_page(page_id)
        page = Page(page_id)
        page.data[:] = data

        # Set up frame
        frame.page_id = page_id
        frame.page = page
        frame.pin_count = 1
        frame.is_dirty = False
        frame.reference_bit = True

        # Update page table
        self._page_table[page_id] = frame_id

        # Update replacer
        self._replacer.pin(frame_id)
        self._replacer.access(frame_id)

        return page

    def unpin_page(self, page_id: int, is_dirty: bool = False) -> bool:
        """Unpin a page. Returns False if page not in pool or not pinned."""
        with self._lock:
            if page_id not in self._page_table:
                return False

            frame_id = self._page_table[page_id]
            frame = self._frames[frame_id]

            if frame.pin_count <= 0:
                return False

            frame.pin_count -= 1
            if is_dirty:
                frame.is_dirty = True

            if frame.pin_count == 0:
                self._replacer.unpin(frame_id)

            return True

    def new_page(self) -> Optional[Tuple[int, Page]]:
        """Allocate a new page. Returns (page_id, page) or None."""
        with self._lock:
            # Get a frame
            frame_id = self._get_free_frame()
            if frame_id is None:
                return None

            # Allocate page on disk
            page_id = self.disk.allocate_page()
            page = Page(page_id)
            frame = self._frames[frame_id]

            frame.page_id = page_id
            frame.page = page
            frame.pin_count = 1
            frame.is_dirty = False
            frame.reference_bit = True

            self._page_table[page_id] = frame_id
            self._replacer.pin(frame_id)
            self._replacer.access(frame_id)

            return (page_id, page)

    def delete_page(self, page_id: int) -> bool:
        """Delete a page from the buffer pool. Returns False if page is pinned."""
        with self._lock:
            if page_id not in self._page_table:
                return True  # Not in pool, nothing to do

            frame_id = self._page_table[page_id]
            frame = self._frames[frame_id]

            if frame.pin_count > 0:
                return False  # Can't delete pinned page

            # Remove from page table
            del self._page_table[page_id]

            # Reset frame and add to free list
            if isinstance(self._replacer, LRUKReplacer):
                self._replacer.remove(frame_id)
            frame.reset()
            self._free_list.append(frame_id)

            return True

    def flush_page(self, page_id: int) -> bool:
        """Write a dirty page to disk. Returns False if page not in pool."""
        with self._lock:
            return self._flush_page_internal(page_id)

    def _flush_page_internal(self, page_id: int) -> bool:
        if page_id not in self._page_table:
            return False

        frame_id = self._page_table[page_id]
        frame = self._frames[frame_id]

        if frame.is_dirty:
            self.disk.write_page(page_id, bytes(frame.page.data))
            frame.is_dirty = False
            self.stats.flushes += 1
            self.stats.dirty_writes += 1

        return True

    def flush_all(self):
        """Flush all dirty pages to disk."""
        with self._lock:
            for page_id in list(self._page_table.keys()):
                self._flush_page_internal(page_id)

    def _get_free_frame(self) -> Optional[int]:
        """Get a free frame, evicting if necessary."""
        # Try free list first
        if self._free_list:
            return self._free_list.popleft()

        # Need to evict
        frame_id = self._replacer.victim(self._frames)
        if frame_id is None:
            return None  # All frames pinned

        frame = self._frames[frame_id]

        # Write back dirty page before evicting
        if frame.is_dirty:
            self.disk.write_page(frame.page_id, bytes(frame.page.data))
            self.stats.dirty_writes += 1

        # Remove old page from page table
        if frame.page_id in self._page_table:
            del self._page_table[frame.page_id]

        self.stats.evictions += 1
        frame.reset()

        return frame_id

    def prefetch(self, start_page_id: int, count: Optional[int] = None):
        """Prefetch sequential pages into the buffer pool.

        Pre-reads pages that are likely to be needed soon.
        Does not pin the prefetched pages.
        """
        if count is None:
            count = self._prefetch_distance

        with self._lock:
            for i in range(count):
                page_id = start_page_id + i
                self.stats.prefetch_requests += 1

                if page_id in self._page_table:
                    self.stats.prefetch_hits += 1
                    continue

                if not self.disk.page_exists(page_id):
                    break

                # Try to load into a free frame
                frame_id = None
                if self._free_list:
                    frame_id = self._free_list.popleft()

                if frame_id is None:
                    # Don't evict for prefetch -- only use free frames
                    break

                data = self.disk.read_page(page_id)
                page = Page(page_id)
                page.data[:] = data

                frame = self._frames[frame_id]
                frame.page_id = page_id
                frame.page = page
                frame.pin_count = 0  # Not pinned
                frame.is_dirty = False
                frame.reference_bit = False

                self._page_table[page_id] = frame_id
                self._replacer.unpin(frame_id)
                self._replacer.access(frame_id)

    def get_pin_count(self, page_id: int) -> int:
        """Get the pin count for a page."""
        with self._lock:
            if page_id not in self._page_table:
                return 0
            frame_id = self._page_table[page_id]
            return self._frames[frame_id].pin_count

    def is_dirty(self, page_id: int) -> bool:
        """Check if a page is dirty."""
        with self._lock:
            if page_id not in self._page_table:
                return False
            frame_id = self._page_table[page_id]
            return self._frames[frame_id].is_dirty

    def contains(self, page_id: int) -> bool:
        """Check if a page is in the buffer pool."""
        with self._lock:
            return page_id in self._page_table

    def num_free_frames(self) -> int:
        """Number of free (unused) frames."""
        with self._lock:
            return len(self._free_list)

    def num_pinned(self) -> int:
        """Number of currently pinned frames."""
        with self._lock:
            return sum(1 for f in self._frames if f.pin_count > 0)

    def get_dirty_pages(self) -> List[int]:
        """Return list of dirty page IDs."""
        with self._lock:
            return [f.page_id for f in self._frames
                    if f.page_id != -1 and f.is_dirty]

    def set_prefetch_distance(self, distance: int):
        """Configure prefetch distance."""
        self._prefetch_distance = max(1, distance)

    def get_stats(self) -> BufferPoolStats:
        """Return a copy of current statistics."""
        return BufferPoolStats(
            hits=self.stats.hits,
            misses=self.stats.misses,
            evictions=self.stats.evictions,
            flushes=self.stats.flushes,
            dirty_writes=self.stats.dirty_writes,
            prefetch_requests=self.stats.prefetch_requests,
            prefetch_hits=self.stats.prefetch_hits,
        )

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats.reset()


# ---------------------------------------------------------------------------
# Scan Buffer (optimized sequential access)
# ---------------------------------------------------------------------------

class ScanBuffer:
    """Optimized buffer for sequential table scans.

    Wraps BufferPool with automatic prefetching and double-buffering
    for sequential page access patterns.
    """

    def __init__(self, bpm: BufferPoolManager, start_page: int, end_page: int,
                 prefetch_distance: int = 4):
        self.bpm = bpm
        self.start_page = start_page
        self.end_page = end_page
        self.prefetch_distance = prefetch_distance
        self._current = start_page
        self._prefetched_up_to = start_page

    def next_page(self) -> Optional[Page]:
        """Get the next page in the sequential scan."""
        if self._current >= self.end_page:
            return None

        # Trigger prefetch if needed
        if self._current + self.prefetch_distance > self._prefetched_up_to:
            prefetch_start = self._prefetched_up_to
            prefetch_count = min(
                self.prefetch_distance,
                self.end_page - prefetch_start
            )
            if prefetch_count > 0:
                self.bpm.prefetch(prefetch_start, prefetch_count)
                self._prefetched_up_to = prefetch_start + prefetch_count

        page = self.bpm.fetch_page(self._current)
        self._current += 1
        return page

    def unpin_current(self):
        """Unpin the most recently fetched page."""
        if self._current > self.start_page:
            self.bpm.unpin_page(self._current - 1)

    def reset(self):
        """Reset scan to the beginning."""
        self._current = self.start_page
        self._prefetched_up_to = self.start_page

    @property
    def current_page_id(self) -> int:
        return self._current - 1 if self._current > self.start_page else -1

    @property
    def pages_remaining(self) -> int:
        return max(0, self.end_page - self._current)
