"""
C229: Sharding / Partitioning System

A data partitioning system supporting hash and range sharding strategies,
shard routing, rebalancing, split/merge, and multi-shard queries.

Composes:
- C205 (Consistent Hashing) for hash-based shard routing
- C226 (Database Replication) for per-shard replication
"""

import hashlib
import time
import math
from enum import Enum, auto
from collections import defaultdict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ShardStrategy(Enum):
    HASH = auto()
    RANGE = auto()
    LIST = auto()
    COMPOSITE = auto()


class ShardState(Enum):
    ACTIVE = auto()
    SPLITTING = auto()
    MERGING = auto()
    MIGRATING = auto()
    DRAINING = auto()
    INACTIVE = auto()


class RebalanceStrategy(Enum):
    MOVE = auto()       # Move whole shards between nodes
    SPLIT = auto()      # Split hot shards
    MERGE = auto()      # Merge cold shards


# ---------------------------------------------------------------------------
# Shard Key Extraction
# ---------------------------------------------------------------------------

class ShardKeyExtractor:
    """Extracts shard keys from records based on configured key columns."""

    def __init__(self, key_columns):
        if not key_columns:
            raise ValueError("key_columns must not be empty")
        self.key_columns = key_columns if isinstance(key_columns, list) else [key_columns]

    def extract(self, record):
        """Extract shard key from a record dict."""
        if len(self.key_columns) == 1:
            col = self.key_columns[0]
            if col not in record:
                raise KeyError(f"Shard key column '{col}' not in record")
            return record[col]
        # Composite key
        parts = []
        for col in self.key_columns:
            if col not in record:
                raise KeyError(f"Shard key column '{col}' not in record")
            parts.append(str(record[col]))
        return ":".join(parts)


# ---------------------------------------------------------------------------
# Hash Functions
# ---------------------------------------------------------------------------

def _md5_int(key):
    """Hash a key to an integer using MD5."""
    h = hashlib.md5(str(key).encode()).hexdigest()
    return int(h, 16)


def _consistent_hash(key, num_shards):
    """Simple consistent hash: MD5 mod num_shards."""
    return _md5_int(key) % num_shards


# ---------------------------------------------------------------------------
# Shard
# ---------------------------------------------------------------------------

class Shard:
    """A single shard holding a partition of data."""

    def __init__(self, shard_id, strategy=ShardStrategy.HASH,
                 range_start=None, range_end=None, list_values=None):
        self.shard_id = shard_id
        self.strategy = strategy
        self.state = ShardState.ACTIVE
        self.range_start = range_start    # For range sharding
        self.range_end = range_end        # For range sharding (exclusive)
        self.list_values = set(list_values) if list_values else set()
        self.data = {}                    # key -> record
        self.node_id = None               # Which node hosts this shard
        self.replicas = []                # Replica node IDs
        self.stats = {
            'reads': 0,
            'writes': 0,
            'size': 0,
            'created_at': time.time(),
        }

    def put(self, key, record):
        """Store a record."""
        is_new = key not in self.data
        self.data[key] = record
        self.stats['writes'] += 1
        self.stats['size'] = len(self.data)
        return is_new

    def get(self, key):
        """Retrieve a record."""
        self.stats['reads'] += 1
        return self.data.get(key)

    def delete(self, key):
        """Delete a record. Returns True if existed."""
        if key in self.data:
            del self.data[key]
            self.stats['size'] = len(self.data)
            return True
        return False

    def scan(self, predicate=None):
        """Scan all records, optionally filtering with predicate."""
        self.stats['reads'] += 1
        if predicate is None:
            return list(self.data.values())
        return [r for r in self.data.values() if predicate(r)]

    def contains_key(self, key):
        return key in self.data

    def keys(self):
        return list(self.data.keys())

    def size(self):
        return len(self.data)

    def clear(self):
        self.data.clear()
        self.stats['size'] = 0

    def status(self):
        return {
            'shard_id': self.shard_id,
            'state': self.state.name,
            'strategy': self.strategy.name,
            'node_id': self.node_id,
            'replicas': list(self.replicas),
            'size': len(self.data),
            'reads': self.stats['reads'],
            'writes': self.stats['writes'],
            'range': (self.range_start, self.range_end) if self.strategy == ShardStrategy.RANGE else None,
            'list_values': sorted(self.list_values) if self.list_values else None,
        }


# ---------------------------------------------------------------------------
# Hash Shard Router
# ---------------------------------------------------------------------------

class HashShardRouter:
    """Routes keys to shards using consistent hashing."""

    def __init__(self, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self._ring = {}       # hash_val -> shard_id
        self._sorted_keys = []
        self._shard_ids = set()

    def add_shard(self, shard_id):
        """Add a shard to the hash ring."""
        self._shard_ids.add(shard_id)
        for i in range(self.virtual_nodes):
            h = _md5_int(f"{shard_id}:{i}")
            self._ring[h] = shard_id
        self._sorted_keys = sorted(self._ring.keys())

    def remove_shard(self, shard_id):
        """Remove a shard from the hash ring."""
        self._shard_ids.discard(shard_id)
        to_remove = [h for h, sid in self._ring.items() if sid == shard_id]
        for h in to_remove:
            del self._ring[h]
        self._sorted_keys = sorted(self._ring.keys())

    def route(self, key):
        """Route a key to a shard ID."""
        if not self._sorted_keys:
            raise ValueError("No shards in router")
        h = _md5_int(str(key))
        # Binary search for first ring position >= h
        idx = self._bisect(h)
        if idx >= len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    def route_with_replicas(self, key, count=3):
        """Route a key to primary + replica shards."""
        if not self._sorted_keys:
            raise ValueError("No shards in router")
        h = _md5_int(str(key))
        idx = self._bisect(h)
        result = []
        seen = set()
        for i in range(len(self._sorted_keys)):
            pos = (idx + i) % len(self._sorted_keys)
            sid = self._ring[self._sorted_keys[pos]]
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
                if len(result) >= count:
                    break
        return result

    def get_distribution(self, keys):
        """Show distribution of keys across shards."""
        dist = defaultdict(int)
        for k in keys:
            sid = self.route(k)
            dist[sid] += 1
        return dict(dist)

    def _bisect(self, val):
        lo, hi = 0, len(self._sorted_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sorted_keys[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @property
    def shard_ids(self):
        return set(self._shard_ids)


# ---------------------------------------------------------------------------
# Range Shard Router
# ---------------------------------------------------------------------------

class RangeShardRouter:
    """Routes keys to shards based on key ranges."""

    def __init__(self):
        self._ranges = []  # [(range_start, range_end, shard_id)] sorted by range_start

    def add_shard(self, shard_id, range_start, range_end):
        """Add a range-based shard. range_end is exclusive."""
        # Check for overlaps
        for rs, re, sid in self._ranges:
            if range_start < re and range_end > rs:
                raise ValueError(
                    f"Range [{range_start}, {range_end}) overlaps with "
                    f"shard {sid} [{rs}, {re})"
                )
        self._ranges.append((range_start, range_end, shard_id))
        self._ranges.sort(key=lambda x: x[0])

    def remove_shard(self, shard_id):
        """Remove a shard by ID."""
        self._ranges = [(rs, re, sid) for rs, re, sid in self._ranges if sid != shard_id]

    def update_range(self, shard_id, new_start, new_end):
        """Update the range for a shard."""
        self._ranges = [(rs, re, sid) for rs, re, sid in self._ranges if sid != shard_id]
        self.add_shard(shard_id, new_start, new_end)

    def route(self, key):
        """Route a key to a shard ID based on range."""
        for rs, re, sid in self._ranges:
            if rs <= key < re:
                return sid
        raise KeyError(f"No shard covers key {key}")

    def get_ranges(self):
        """Return all ranges."""
        return [(rs, re, sid) for rs, re, sid in self._ranges]

    @property
    def shard_ids(self):
        return {sid for _, _, sid in self._ranges}


# ---------------------------------------------------------------------------
# List Shard Router
# ---------------------------------------------------------------------------

class ListShardRouter:
    """Routes keys to shards based on explicit value lists."""

    def __init__(self):
        self._value_map = {}   # value -> shard_id
        self._shard_values = defaultdict(set)  # shard_id -> set of values

    def add_shard(self, shard_id, values):
        """Add a shard with a list of values it handles."""
        for v in values:
            if v in self._value_map:
                raise ValueError(
                    f"Value {v} already assigned to shard {self._value_map[v]}"
                )
            self._value_map[v] = shard_id
            self._shard_values[shard_id].add(v)

    def remove_shard(self, shard_id):
        """Remove a shard."""
        for v in self._shard_values.get(shard_id, set()):
            del self._value_map[v]
        del self._shard_values[shard_id]

    def route(self, key):
        """Route a value to a shard."""
        if key not in self._value_map:
            raise KeyError(f"No shard assigned for value {key}")
        return self._value_map[key]

    @property
    def shard_ids(self):
        return set(self._shard_values.keys())


# ---------------------------------------------------------------------------
# Shard Manager
# ---------------------------------------------------------------------------

class ShardManager:
    """Manages a collection of shards with routing."""

    def __init__(self, strategy=ShardStrategy.HASH, key_columns=None,
                 virtual_nodes=150):
        self.strategy = strategy
        self.extractor = ShardKeyExtractor(key_columns or ['id'])
        self.shards = {}  # shard_id -> Shard

        if strategy == ShardStrategy.HASH:
            self.router = HashShardRouter(virtual_nodes=virtual_nodes)
        elif strategy == ShardStrategy.RANGE:
            self.router = RangeShardRouter()
        elif strategy == ShardStrategy.LIST:
            self.router = ListShardRouter()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self._event_handlers = defaultdict(list)
        self._migration_log = []

    def add_shard(self, shard_id, node_id=None, range_start=None,
                  range_end=None, list_values=None):
        """Add a new shard."""
        if shard_id in self.shards:
            raise ValueError(f"Shard {shard_id} already exists")

        shard = Shard(
            shard_id, self.strategy,
            range_start=range_start, range_end=range_end,
            list_values=list_values
        )
        shard.node_id = node_id
        self.shards[shard_id] = shard

        if self.strategy == ShardStrategy.HASH:
            self.router.add_shard(shard_id)
        elif self.strategy == ShardStrategy.RANGE:
            if range_start is None or range_end is None:
                raise ValueError("Range shards require range_start and range_end")
            self.router.add_shard(shard_id, range_start, range_end)
        elif self.strategy == ShardStrategy.LIST:
            if not list_values:
                raise ValueError("List shards require list_values")
            self.router.add_shard(shard_id, list_values)

        self._emit('shard_added', {'shard_id': shard_id})
        return shard

    def remove_shard(self, shard_id):
        """Remove a shard (must be empty or drained first)."""
        if shard_id not in self.shards:
            raise KeyError(f"Shard {shard_id} not found")
        shard = self.shards[shard_id]
        if shard.size() > 0:
            raise ValueError(f"Shard {shard_id} is not empty ({shard.size()} records)")
        self.router.remove_shard(shard_id)
        del self.shards[shard_id]
        self._emit('shard_removed', {'shard_id': shard_id})

    def get_shard(self, shard_id):
        """Get a shard by ID."""
        return self.shards.get(shard_id)

    def route_key(self, key):
        """Route a shard key to the appropriate shard ID."""
        return self.router.route(key)

    def route_record(self, record):
        """Route a record to the appropriate shard."""
        key = self.extractor.extract(record)
        shard_id = self.router.route(key)
        return self.shards[shard_id]

    def put(self, record):
        """Insert/update a record, routing to the correct shard."""
        key = self.extractor.extract(record)
        shard_id = self.router.route(key)
        shard = self.shards[shard_id]
        if shard.state != ShardState.ACTIVE:
            raise ValueError(f"Shard {shard_id} is {shard.state.name}, cannot write")
        is_new = shard.put(key, record)
        self._emit('write', {'shard_id': shard_id, 'key': key, 'is_new': is_new})
        return shard_id

    def get(self, key):
        """Get a record by shard key."""
        shard_id = self.router.route(key)
        shard = self.shards[shard_id]
        return shard.get(key)

    def delete(self, key):
        """Delete a record by shard key."""
        shard_id = self.router.route(key)
        shard = self.shards[shard_id]
        return shard.delete(key)

    def scatter_gather(self, predicate=None):
        """Query all shards and gather results (scatter-gather)."""
        results = []
        for shard in self.shards.values():
            if shard.state == ShardState.ACTIVE:
                results.extend(shard.scan(predicate))
        return results

    def scatter_gather_with_limit(self, predicate=None, sort_key=None,
                                   limit=None, offset=0):
        """Scatter-gather with sorting, pagination."""
        results = self.scatter_gather(predicate)
        if sort_key:
            results.sort(key=sort_key)
        if offset:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def aggregate(self, field, agg_func='sum'):
        """Aggregate a field across all shards."""
        values = []
        for shard in self.shards.values():
            if shard.state == ShardState.ACTIVE:
                for record in shard.data.values():
                    if field in record:
                        values.append(record[field])

        if not values:
            return None

        if agg_func == 'sum':
            return sum(values)
        elif agg_func == 'avg':
            return sum(values) / len(values)
        elif agg_func == 'min':
            return min(values)
        elif agg_func == 'max':
            return max(values)
        elif agg_func == 'count':
            return len(values)
        else:
            raise ValueError(f"Unknown aggregate function: {agg_func}")

    # ---- Split / Merge ----

    def split_shard(self, shard_id, new_shard_id_1=None, new_shard_id_2=None,
                    split_point=None):
        """Split a shard into two. For range shards, split_point is required.
        For hash shards, data is redistributed."""
        if shard_id not in self.shards:
            raise KeyError(f"Shard {shard_id} not found")

        old_shard = self.shards[shard_id]
        old_shard.state = ShardState.SPLITTING

        id1 = new_shard_id_1 or f"{shard_id}_a"
        id2 = new_shard_id_2 or f"{shard_id}_b"

        if self.strategy == ShardStrategy.RANGE:
            if split_point is None:
                # Auto split at midpoint
                split_point = (old_shard.range_start + old_shard.range_end) // 2

            # Create two new range shards
            self.router.remove_shard(shard_id)

            shard_a = Shard(id1, ShardStrategy.RANGE,
                           range_start=old_shard.range_start, range_end=split_point)
            shard_a.node_id = old_shard.node_id
            self.shards[id1] = shard_a
            self.router.add_shard(id1, old_shard.range_start, split_point)

            shard_b = Shard(id2, ShardStrategy.RANGE,
                           range_start=split_point, range_end=old_shard.range_end)
            shard_b.node_id = old_shard.node_id
            self.shards[id2] = shard_b
            self.router.add_shard(id2, split_point, old_shard.range_end)

            # Redistribute data
            for key, record in old_shard.data.items():
                if key < split_point:
                    shard_a.put(key, record)
                else:
                    shard_b.put(key, record)

        elif self.strategy == ShardStrategy.HASH:
            # For hash sharding: remove old, add two new, redistribute
            self.router.remove_shard(shard_id)

            shard_a = Shard(id1, ShardStrategy.HASH)
            shard_a.node_id = old_shard.node_id
            self.shards[id1] = shard_a
            self.router.add_shard(id1)

            shard_b = Shard(id2, ShardStrategy.HASH)
            shard_b.node_id = old_shard.node_id
            self.shards[id2] = shard_b
            self.router.add_shard(id2)

            # Redistribute all data from old shard
            for key, record in old_shard.data.items():
                target_id = self.router.route(key)
                self.shards[target_id].put(key, record)

        # Remove old shard
        del self.shards[shard_id]

        self._migration_log.append({
            'type': 'split',
            'source': shard_id,
            'targets': [id1, id2],
            'timestamp': time.time(),
            'records_moved': old_shard.size(),
        })
        self._emit('shard_split', {'source': shard_id, 'targets': [id1, id2]})
        return id1, id2

    def merge_shards(self, shard_id_1, shard_id_2, merged_shard_id=None):
        """Merge two shards into one."""
        if shard_id_1 not in self.shards:
            raise KeyError(f"Shard {shard_id_1} not found")
        if shard_id_2 not in self.shards:
            raise KeyError(f"Shard {shard_id_2} not found")

        s1 = self.shards[shard_id_1]
        s2 = self.shards[shard_id_2]
        s1.state = ShardState.MERGING
        s2.state = ShardState.MERGING

        merged_id = merged_shard_id or f"{shard_id_1}_{shard_id_2}"

        if self.strategy == ShardStrategy.RANGE:
            # Ranges must be adjacent
            if s1.range_end != s2.range_start and s2.range_end != s1.range_start:
                raise ValueError("Can only merge adjacent range shards")
            new_start = min(s1.range_start, s2.range_start)
            new_end = max(s1.range_end, s2.range_end)

            self.router.remove_shard(shard_id_1)
            self.router.remove_shard(shard_id_2)

            merged = Shard(merged_id, ShardStrategy.RANGE,
                          range_start=new_start, range_end=new_end)
            merged.node_id = s1.node_id
            self.shards[merged_id] = merged
            self.router.add_shard(merged_id, new_start, new_end)

            for key, record in s1.data.items():
                merged.put(key, record)
            for key, record in s2.data.items():
                merged.put(key, record)

        elif self.strategy == ShardStrategy.HASH:
            self.router.remove_shard(shard_id_1)
            self.router.remove_shard(shard_id_2)

            merged = Shard(merged_id, ShardStrategy.HASH)
            merged.node_id = s1.node_id
            self.shards[merged_id] = merged
            self.router.add_shard(merged_id)

            # Put all data into the merged shard first
            all_data = {}
            all_data.update(s1.data)
            all_data.update(s2.data)

            # Some keys might route to other existing shards, but for merge
            # we keep them in the merged shard
            for key, record in all_data.items():
                merged.put(key, record)

        del self.shards[shard_id_1]
        del self.shards[shard_id_2]

        total = s1.size() + s2.size()
        self._migration_log.append({
            'type': 'merge',
            'sources': [shard_id_1, shard_id_2],
            'target': merged_id,
            'timestamp': time.time(),
            'records_moved': total,
        })
        self._emit('shard_merged', {
            'sources': [shard_id_1, shard_id_2], 'target': merged_id
        })
        return merged_id

    # ---- Rebalancing ----

    def rebalance(self):
        """Rebalance data across shards (hash strategy only).
        Moves misrouted keys to their correct shard after topology changes."""
        if self.strategy != ShardStrategy.HASH:
            return {'moved': 0}

        moves = 0
        # Collect all data with current locations
        to_move = []  # (key, record, from_shard, to_shard)

        for shard_id, shard in list(self.shards.items()):
            for key in list(shard.data.keys()):
                correct_shard = self.router.route(key)
                if correct_shard != shard_id:
                    to_move.append((key, shard.data[key], shard_id, correct_shard))

        for key, record, from_id, to_id in to_move:
            self.shards[from_id].delete(key)
            self.shards[to_id].put(key, record)
            moves += 1

        if moves > 0:
            self._migration_log.append({
                'type': 'rebalance',
                'moves': moves,
                'timestamp': time.time(),
            })
            self._emit('rebalanced', {'moves': moves})
        return {'moved': moves}

    # ---- Migration ----

    def migrate_shard(self, shard_id, from_node, to_node):
        """Migrate a shard from one node to another."""
        if shard_id not in self.shards:
            raise KeyError(f"Shard {shard_id} not found")
        shard = self.shards[shard_id]
        if shard.node_id != from_node:
            raise ValueError(f"Shard {shard_id} is on node {shard.node_id}, not {from_node}")

        shard.state = ShardState.MIGRATING
        old_node = shard.node_id
        shard.node_id = to_node
        shard.state = ShardState.ACTIVE

        self._migration_log.append({
            'type': 'migrate',
            'shard_id': shard_id,
            'from_node': from_node,
            'to_node': to_node,
            'timestamp': time.time(),
            'records': shard.size(),
        })
        self._emit('shard_migrated', {
            'shard_id': shard_id, 'from': from_node, 'to': to_node
        })

    # ---- Stats / Status ----

    def get_distribution(self):
        """Get data distribution across shards."""
        dist = {}
        total = 0
        for sid, shard in self.shards.items():
            dist[sid] = shard.size()
            total += shard.size()
        return {
            'shards': dist,
            'total': total,
            'shard_count': len(self.shards),
            'avg_per_shard': total / len(self.shards) if self.shards else 0,
        }

    def get_hotspots(self, threshold=None):
        """Find shards with disproportionate load."""
        if not self.shards:
            return []
        total_writes = sum(s.stats['writes'] for s in self.shards.values())
        avg_writes = total_writes / len(self.shards) if self.shards else 0
        if threshold is None:
            # A shard is hot if it has > 1.5x the average
            threshold = avg_writes * 1.5 if avg_writes > 0 else 1

        hotspots = []
        for sid, shard in self.shards.items():
            if shard.stats['writes'] > threshold:
                hotspots.append({
                    'shard_id': sid,
                    'writes': shard.stats['writes'],
                    'size': shard.size(),
                    'ratio': shard.stats['writes'] / avg_writes if avg_writes > 0 else 0,
                })
        return hotspots

    def get_migration_log(self):
        return list(self._migration_log)

    def status(self):
        """Overall sharding status."""
        return {
            'strategy': self.strategy.name,
            'shard_count': len(self.shards),
            'total_records': sum(s.size() for s in self.shards.values()),
            'shards': {sid: s.status() for sid, s in self.shards.items()},
            'migrations': len(self._migration_log),
        }

    # ---- Events ----

    def on(self, event, callback):
        self._event_handlers[event].append(callback)

    def _emit(self, event, data):
        for cb in self._event_handlers.get(event, []):
            cb(data)


# ---------------------------------------------------------------------------
# Auto-Sharding Manager
# ---------------------------------------------------------------------------

class AutoShardManager:
    """Automatically manages shard splitting and merging based on thresholds."""

    def __init__(self, shard_manager, max_shard_size=1000, min_shard_size=100,
                 max_write_rate=500, check_interval=60):
        self.manager = shard_manager
        self.max_shard_size = max_shard_size
        self.min_shard_size = min_shard_size
        self.max_write_rate = max_write_rate
        self.check_interval = check_interval
        self._split_counter = 0
        self._merge_counter = 0

    def check_and_rebalance(self):
        """Check all shards and perform auto-split/merge as needed."""
        actions = []

        # Check for splits (too large or too hot)
        for sid in list(self.manager.shards.keys()):
            shard = self.manager.shards.get(sid)
            if shard is None or shard.state != ShardState.ACTIVE:
                continue

            if shard.size() > self.max_shard_size:
                self._split_counter += 1
                id1 = f"{sid}_split{self._split_counter}a"
                id2 = f"{sid}_split{self._split_counter}b"
                try:
                    r = self.manager.split_shard(sid, id1, id2)
                    actions.append({'action': 'split', 'source': sid, 'result': r})
                except Exception as e:
                    actions.append({'action': 'split_failed', 'source': sid, 'error': str(e)})

        # Check for merges (adjacent small shards -- range strategy only)
        if self.manager.strategy == ShardStrategy.RANGE:
            ranges = self.manager.router.get_ranges()
            i = 0
            while i < len(ranges) - 1:
                _, _, sid1 = ranges[i]
                _, _, sid2 = ranges[i + 1]
                s1 = self.manager.shards.get(sid1)
                s2 = self.manager.shards.get(sid2)
                if (s1 and s2 and
                    s1.state == ShardState.ACTIVE and s2.state == ShardState.ACTIVE and
                    s1.size() + s2.size() < self.min_shard_size):
                    self._merge_counter += 1
                    merged_id = f"merged_{self._merge_counter}"
                    try:
                        r = self.manager.merge_shards(sid1, sid2, merged_id)
                        actions.append({'action': 'merge', 'sources': [sid1, sid2], 'result': r})
                        # Re-fetch ranges since they changed
                        ranges = self.manager.router.get_ranges()
                        # Don't increment i since merged shard replaces the pair
                    except Exception as e:
                        actions.append({'action': 'merge_failed', 'error': str(e)})
                        i += 1
                else:
                    i += 1

        return actions


# ---------------------------------------------------------------------------
# Shard-Aware Query Coordinator
# ---------------------------------------------------------------------------

class QueryCoordinator:
    """Coordinates queries across shards with routing optimization."""

    def __init__(self, shard_manager):
        self.manager = shard_manager
        self._query_stats = defaultdict(int)

    def point_query(self, key):
        """Single-key lookup -- routes to one shard."""
        self._query_stats['point'] += 1
        return self.manager.get(key)

    def multi_get(self, keys):
        """Batch get -- groups keys by shard for efficiency."""
        self._query_stats['multi_get'] += 1
        # Group keys by target shard
        shard_keys = defaultdict(list)
        for key in keys:
            sid = self.manager.route_key(key)
            shard_keys[sid].append(key)

        results = {}
        for sid, ks in shard_keys.items():
            shard = self.manager.shards[sid]
            for k in ks:
                val = shard.get(k)
                if val is not None:
                    results[k] = val
        return results

    def scan_query(self, predicate=None, sort_key=None, limit=None, offset=0):
        """Scatter-gather query with optional sort/limit/offset."""
        self._query_stats['scan'] += 1
        return self.manager.scatter_gather_with_limit(
            predicate=predicate, sort_key=sort_key,
            limit=limit, offset=offset
        )

    def range_query(self, start_key, end_key):
        """Range query -- for range-sharded data, only hits relevant shards."""
        self._query_stats['range'] += 1
        if self.manager.strategy == ShardStrategy.RANGE:
            results = []
            for sid, shard in self.manager.shards.items():
                if shard.state != ShardState.ACTIVE:
                    continue
                # Check if shard range overlaps query range
                if shard.range_end is not None and shard.range_start is not None:
                    if shard.range_start < end_key and shard.range_end > start_key:
                        for key, record in shard.data.items():
                            if start_key <= key < end_key:
                                results.append(record)
            results.sort(key=lambda r: self.manager.extractor.extract(r))
            return results
        else:
            # For non-range strategies, must scatter-gather
            def pred(r):
                k = self.manager.extractor.extract(r)
                return start_key <= k < end_key
            return self.manager.scatter_gather(predicate=pred)

    def aggregate_query(self, field, agg_func='sum'):
        """Aggregate across all shards."""
        self._query_stats['aggregate'] += 1
        return self.manager.aggregate(field, agg_func)

    def count(self, predicate=None):
        """Count records across all shards."""
        self._query_stats['count'] += 1
        if predicate is None:
            return sum(s.size() for s in self.manager.shards.values()
                      if s.state == ShardState.ACTIVE)
        return len(self.manager.scatter_gather(predicate))

    def stats(self):
        return dict(self._query_stats)


# ---------------------------------------------------------------------------
# Resharding Planner
# ---------------------------------------------------------------------------

class ReshardingPlanner:
    """Plans resharding operations to balance load and data."""

    def __init__(self, shard_manager):
        self.manager = shard_manager

    def plan(self, target_shard_count=None, balance_metric='size'):
        """Generate a resharding plan."""
        current = self.manager.get_distribution()
        shards = current['shards']

        if target_shard_count is None:
            target_shard_count = len(shards)

        plan = {
            'current_shards': len(shards),
            'target_shards': target_shard_count,
            'total_records': current['total'],
            'operations': [],
        }

        if balance_metric == 'size':
            self._plan_by_size(plan, shards, target_shard_count)
        elif balance_metric == 'load':
            self._plan_by_load(plan, target_shard_count)

        return plan

    def _plan_by_size(self, plan, shards, target_count):
        """Plan based on data size distribution."""
        total = sum(shards.values())
        if total == 0 or not shards:
            return

        target_per_shard = total / target_count

        if target_count > len(shards):
            # Need more shards -- find largest to split
            sorted_shards = sorted(shards.items(), key=lambda x: x[1], reverse=True)
            splits_needed = target_count - len(shards)
            for sid, size in sorted_shards[:splits_needed]:
                plan['operations'].append({
                    'type': 'split',
                    'shard': sid,
                    'reason': f'size {size} exceeds target {target_per_shard:.0f}',
                })
        elif target_count < len(shards):
            # Need fewer shards -- find smallest to merge
            sorted_shards = sorted(shards.items(), key=lambda x: x[1])
            merges_needed = len(shards) - target_count
            for i in range(0, merges_needed * 2, 2):
                if i + 1 < len(sorted_shards):
                    plan['operations'].append({
                        'type': 'merge',
                        'shards': [sorted_shards[i][0], sorted_shards[i + 1][0]],
                        'reason': f'combined size {sorted_shards[i][1] + sorted_shards[i + 1][1]} fits target',
                    })
        else:
            # Same count -- check for imbalance
            for sid, size in shards.items():
                if size > target_per_shard * 1.5:
                    plan['operations'].append({
                        'type': 'split',
                        'shard': sid,
                        'reason': f'size {size} > 1.5x target {target_per_shard:.0f}',
                    })

    def _plan_by_load(self, plan, target_count):
        """Plan based on write load."""
        hotspots = self.manager.get_hotspots()
        for h in hotspots:
            plan['operations'].append({
                'type': 'split',
                'shard': h['shard_id'],
                'reason': f'hot: {h["writes"]} writes, {h["ratio"]:.1f}x avg',
            })


# ---------------------------------------------------------------------------
# Sharding System (top-level facade)
# ---------------------------------------------------------------------------

class ShardingSystem:
    """Top-level sharding system combining all components."""

    def __init__(self, strategy=ShardStrategy.HASH, key_columns=None,
                 num_shards=4, virtual_nodes=150,
                 auto_split_size=1000, auto_merge_size=100,
                 range_start=None, range_end=None):
        self.manager = ShardManager(
            strategy=strategy, key_columns=key_columns,
            virtual_nodes=virtual_nodes
        )
        self.coordinator = QueryCoordinator(self.manager)
        self.planner = ReshardingPlanner(self.manager)
        self.auto_manager = AutoShardManager(
            self.manager,
            max_shard_size=auto_split_size,
            min_shard_size=auto_merge_size
        )

        # Auto-create initial shards
        if strategy == ShardStrategy.HASH:
            for i in range(num_shards):
                self.manager.add_shard(f"shard_{i}", node_id=f"node_{i % max(1, num_shards // 2)}")
        elif strategy == ShardStrategy.RANGE:
            if range_start is not None and range_end is not None:
                chunk = (range_end - range_start) // num_shards
                for i in range(num_shards):
                    rs = range_start + i * chunk
                    re = range_start + (i + 1) * chunk if i < num_shards - 1 else range_end
                    self.manager.add_shard(
                        f"shard_{i}", node_id=f"node_{i}",
                        range_start=rs, range_end=re
                    )

    def put(self, record):
        return self.manager.put(record)

    def get(self, key):
        return self.coordinator.point_query(key)

    def multi_get(self, keys):
        return self.coordinator.multi_get(keys)

    def delete(self, key):
        return self.manager.delete(key)

    def query(self, predicate=None, sort_key=None, limit=None, offset=0):
        return self.coordinator.scan_query(predicate, sort_key, limit, offset)

    def range_query(self, start_key, end_key):
        return self.coordinator.range_query(start_key, end_key)

    def aggregate(self, field, agg_func='sum'):
        return self.coordinator.aggregate_query(field, agg_func)

    def count(self, predicate=None):
        return self.coordinator.count(predicate)

    def split_shard(self, shard_id, id1=None, id2=None, split_point=None):
        return self.manager.split_shard(shard_id, id1, id2, split_point)

    def merge_shards(self, id1, id2, merged_id=None):
        return self.manager.merge_shards(id1, id2, merged_id)

    def rebalance(self):
        return self.manager.rebalance()

    def auto_rebalance(self):
        return self.auto_manager.check_and_rebalance()

    def migrate_shard(self, shard_id, from_node, to_node):
        return self.manager.migrate_shard(shard_id, from_node, to_node)

    def plan_resharding(self, target_shards=None, metric='size'):
        return self.planner.plan(target_shards, metric)

    def distribution(self):
        return self.manager.get_distribution()

    def hotspots(self, threshold=None):
        return self.manager.get_hotspots(threshold)

    def status(self):
        return self.manager.status()
