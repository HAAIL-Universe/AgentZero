"""
C224: Distributed Log / Message Queue
Composes C221 (Distributed File System) + C201 (Raft Consensus)

A Kafka-inspired distributed log with:
- Partitioned topics with append-only segments
- Consumer groups with coordinated offset tracking
- Raft-based partition leader election and replication
- At-least-once and at-most-once delivery semantics
- Log compaction and retention policies
- Producer partitioning (round-robin, key-based)
"""

import time
import hashlib
import json
import struct
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# Core Data Types
# ============================================================

class DeliverySemantics(Enum):
    AT_MOST_ONCE = auto()
    AT_LEAST_ONCE = auto()


class AckLevel(Enum):
    NONE = 0       # Fire and forget
    LEADER = 1     # Leader acknowledged
    ALL = 2        # All replicas acknowledged


class RetentionPolicy(Enum):
    DELETE = auto()     # Delete old segments
    COMPACT = auto()    # Keep latest per key


@dataclass
class Record:
    """A single record in the log."""
    key: Optional[str]
    value: Any
    timestamp: float = 0.0
    offset: int = -1
    partition: int = -1
    headers: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def size_bytes(self):
        """Estimate serialized size."""
        key_size = len(str(self.key)) if self.key else 0
        val_size = len(str(self.value)) if self.value else 0
        return key_size + val_size + 64  # overhead for metadata

    def serialize(self):
        return json.dumps({
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp,
            'offset': self.offset,
            'partition': self.partition,
            'headers': self.headers,
        })

    @staticmethod
    def deserialize(data):
        d = json.loads(data)
        return Record(
            key=d['key'], value=d['value'],
            timestamp=d['timestamp'], offset=d['offset'],
            partition=d['partition'], headers=d.get('headers', {}),
        )


@dataclass
class RecordBatch:
    """A batch of records for efficient I/O."""
    records: list = field(default_factory=list)
    base_offset: int = 0
    max_offset: int = -1

    def append(self, record):
        self.records.append(record)
        self.max_offset = record.offset

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


# ============================================================
# Log Segment
# ============================================================

class LogSegment:
    """An append-only segment of a partition log.

    Each segment covers a range of offsets [base_offset, next_offset).
    """

    def __init__(self, base_offset, max_size=1024 * 1024):
        self.base_offset = base_offset
        self.max_size = max_size
        self.records = []
        self.next_offset = base_offset
        self.current_size = 0
        self.created_at = time.time()
        self.closed = False
        self._index = {}  # offset -> position in self.records

    @property
    def count(self):
        return len(self.records)

    @property
    def is_full(self):
        return self.current_size >= self.max_size

    def append(self, record):
        """Append a record. Returns assigned offset."""
        if self.closed:
            raise LogError("Segment is closed")
        record.offset = self.next_offset
        self._index[self.next_offset] = len(self.records)
        self.records.append(record)
        self.current_size += record.size_bytes()
        self.next_offset += 1
        if self.is_full:
            self.closed = True
        return record.offset

    def read(self, offset):
        """Read a single record by offset."""
        pos = self._index.get(offset)
        if pos is None:
            return None
        return self.records[pos]

    def read_range(self, start_offset, max_records=100):
        """Read records starting from start_offset."""
        result = []
        for off in range(start_offset, self.next_offset):
            if len(result) >= max_records:
                break
            rec = self.read(off)
            if rec:
                result.append(rec)
        return result

    def close(self):
        self.closed = True

    def compact(self):
        """Keep only latest record per key (for compacted topics)."""
        latest = {}
        for rec in self.records:
            if rec.key is not None:
                latest[rec.key] = rec
            else:
                latest[id(rec)] = rec  # keyless records kept

        self.records = list(latest.values())
        self.records.sort(key=lambda r: r.offset)
        self._index = {r.offset: i for i, r in enumerate(self.records)}
        self.current_size = sum(r.size_bytes() for r in self.records)


# ============================================================
# Partition
# ============================================================

class LogError(Exception):
    pass


class Partition:
    """A single partition -- an ordered, append-only sequence of records.

    Manages multiple segments, handles reads/writes, tracks high watermark.
    """

    def __init__(self, topic_name, partition_id, segment_size=1024 * 1024,
                 retention_ms=7 * 24 * 3600 * 1000, retention_policy=RetentionPolicy.DELETE):
        self.topic_name = topic_name
        self.partition_id = partition_id
        self.segment_size = segment_size
        self.retention_ms = retention_ms
        self.retention_policy = retention_policy

        # Segments
        self._segments = [LogSegment(0, max_size=segment_size)]
        self._next_offset = 0

        # Replication
        self.leader_id = None
        self.replica_ids = []
        self.isr = []  # in-sync replicas
        self.high_watermark = 0  # committed offset (replicated to all ISR)

        # Stats
        self.total_writes = 0
        self.total_reads = 0
        self.bytes_written = 0

    @property
    def log_start_offset(self):
        """Earliest available offset."""
        if not self._segments:
            return 0
        return self._segments[0].base_offset

    @property
    def log_end_offset(self):
        """Next offset to be assigned (one past last written)."""
        return self._next_offset

    @property
    def active_segment(self):
        return self._segments[-1]

    def append(self, record):
        """Append a record. Returns the assigned offset."""
        record.partition = self.partition_id

        # Roll to new segment if needed
        if self.active_segment.is_full or self.active_segment.closed:
            new_seg = LogSegment(self._next_offset, max_size=self.segment_size)
            self._segments.append(new_seg)

        offset = self.active_segment.append(record)
        self._next_offset = self.active_segment.next_offset
        self.total_writes += 1
        self.bytes_written += record.size_bytes()
        return offset

    def append_batch(self, records):
        """Append a batch of records. Returns list of offsets."""
        return [self.append(r) for r in records]

    def read(self, offset, max_records=100):
        """Read records starting from offset."""
        if offset < self.log_start_offset:
            offset = self.log_start_offset
        if offset >= self.log_end_offset:
            return []

        result = []
        remaining = max_records
        for seg in self._segments:
            if seg.next_offset <= offset:
                continue
            if seg.base_offset > offset + max_records:
                break
            start = max(offset, seg.base_offset)
            batch = seg.read_range(start, remaining)
            result.extend(batch)
            remaining -= len(batch)
            if remaining <= 0:
                break

        self.total_reads += 1
        return result

    def get_record(self, offset):
        """Get a single record by exact offset."""
        for seg in self._segments:
            if seg.base_offset <= offset < seg.next_offset:
                return seg.read(offset)
        return None

    def update_high_watermark(self, offset):
        """Update committed offset (all ISR have replicated up to this)."""
        if offset > self.high_watermark:
            self.high_watermark = min(offset, self.log_end_offset)

    def truncate(self, offset):
        """Truncate log from offset onwards (for replication catch-up)."""
        new_segments = []
        for seg in self._segments:
            if seg.base_offset >= offset:
                break
            if seg.next_offset > offset:
                # Truncate within segment
                seg.records = [r for r in seg.records if r.offset < offset]
                seg._index = {r.offset: i for i, r in enumerate(seg.records)}
                seg.next_offset = offset
                seg.current_size = sum(r.size_bytes() for r in seg.records)
                seg.closed = False
            new_segments.append(seg)

        if not new_segments:
            new_segments = [LogSegment(offset, max_size=self.segment_size)]

        self._segments = new_segments
        self._next_offset = offset

    def apply_retention(self, now=None):
        """Remove segments older than retention period."""
        if now is None:
            now = time.time()
        cutoff = now - (self.retention_ms / 1000.0)
        kept = []
        for seg in self._segments:
            if seg.created_at >= cutoff or seg is self._segments[-1]:
                kept.append(seg)
        removed = len(self._segments) - len(kept)
        self._segments = kept
        return removed

    def compact(self):
        """Compact all segments (keep latest per key)."""
        if self.retention_policy != RetentionPolicy.COMPACT:
            return 0

        total_before = sum(seg.count for seg in self._segments)

        # Global compaction: merge all into key -> latest record
        latest = {}
        for seg in self._segments:
            for rec in seg.records:
                if rec.key is not None:
                    latest[rec.key] = rec
                else:
                    latest[('__no_key__', rec.offset)] = rec

        records = sorted(latest.values(), key=lambda r: r.offset)
        total_after = len(records)

        # Rebuild segments
        self._segments = []
        seg = LogSegment(records[0].offset if records else 0, max_size=self.segment_size)
        self._segments.append(seg)
        for rec in records:
            if seg.is_full:
                seg = LogSegment(rec.offset, max_size=self.segment_size)
                self._segments.append(seg)
            seg._index[rec.offset] = len(seg.records)
            seg.records.append(rec)
            seg.next_offset = rec.offset + 1
            seg.current_size += rec.size_bytes()

        return total_before - total_after

    def stats(self):
        return {
            'topic': self.topic_name,
            'partition': self.partition_id,
            'log_start_offset': self.log_start_offset,
            'log_end_offset': self.log_end_offset,
            'high_watermark': self.high_watermark,
            'segments': len(self._segments),
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'bytes_written': self.bytes_written,
            'leader': self.leader_id,
            'replicas': list(self.replica_ids),
            'isr': list(self.isr),
        }


# ============================================================
# Offset Manager
# ============================================================

class OffsetManager:
    """Tracks consumer group offsets per topic-partition."""

    def __init__(self):
        # {group_id: {(topic, partition): offset}}
        self._committed = {}
        self._commit_log = []

    def commit(self, group_id, topic, partition, offset):
        """Commit offset for a consumer group."""
        if group_id not in self._committed:
            self._committed[group_id] = {}
        key = (topic, partition)
        self._committed[group_id][key] = offset
        self._commit_log.append({
            'group': group_id, 'topic': topic,
            'partition': partition, 'offset': offset,
            'timestamp': time.time(),
        })

    def get_committed(self, group_id, topic, partition):
        """Get committed offset, or -1 if none."""
        if group_id not in self._committed:
            return -1
        return self._committed[group_id].get((topic, partition), -1)

    def get_all_committed(self, group_id):
        """Get all committed offsets for a group."""
        return dict(self._committed.get(group_id, {}))

    def reset(self, group_id, topic, partition, offset):
        """Reset offset (seek)."""
        self.commit(group_id, topic, partition, offset)

    def delete_group(self, group_id):
        """Remove all offsets for a group."""
        self._committed.pop(group_id, None)

    def list_groups(self):
        """List all groups with committed offsets."""
        return list(self._committed.keys())

    def lag(self, group_id, topic, partition, log_end_offset):
        """Calculate consumer lag for a partition."""
        committed = self.get_committed(group_id, topic, partition)
        if committed < 0:
            return log_end_offset
        return max(0, log_end_offset - committed)


# ============================================================
# Consumer Group
# ============================================================

class RebalanceStrategy(Enum):
    RANGE = auto()
    ROUND_ROBIN = auto()


class ConsumerGroup:
    """Coordinates consumers reading from topic partitions.

    Handles partition assignment, offset management, and rebalancing.
    """

    def __init__(self, group_id, offset_manager, strategy=RebalanceStrategy.ROUND_ROBIN,
                 auto_commit=True, semantics=DeliverySemantics.AT_LEAST_ONCE):
        self.group_id = group_id
        self.offset_manager = offset_manager
        self.strategy = strategy
        self.auto_commit = auto_commit
        self.semantics = semantics

        self._members = {}  # member_id -> {'subscriptions': [topics], 'assignments': [(topic, part)]}
        self._generation = 0
        self._assignments = {}  # (topic, partition) -> member_id

    @property
    def member_count(self):
        return len(self._members)

    @property
    def generation(self):
        return self._generation

    def join(self, member_id, subscriptions):
        """A consumer joins the group."""
        self._members[member_id] = {
            'subscriptions': list(subscriptions),
            'assignments': [],
            'joined_at': time.time(),
        }
        return member_id

    def leave(self, member_id):
        """A consumer leaves the group."""
        if member_id in self._members:
            del self._members[member_id]
            return True
        return False

    def rebalance(self, topic_partitions):
        """Rebalance partition assignments across members.

        topic_partitions: {topic_name: [partition_ids]}
        """
        self._generation += 1
        self._assignments.clear()

        # Clear old assignments
        for info in self._members.values():
            info['assignments'] = []

        if not self._members:
            return {}

        # Collect all (topic, partition) pairs that members are subscribed to
        all_tp = []
        for topic, parts in sorted(topic_partitions.items()):
            for p in sorted(parts):
                # Only include if at least one member subscribes to this topic
                for info in self._members.values():
                    if topic in info['subscriptions']:
                        all_tp.append((topic, p))
                        break

        members = sorted(self._members.keys())
        if not members or not all_tp:
            return {}

        if self.strategy == RebalanceStrategy.ROUND_ROBIN:
            for i, tp in enumerate(all_tp):
                member = members[i % len(members)]
                self._members[member]['assignments'].append(tp)
                self._assignments[tp] = member
        else:  # RANGE
            # Per-topic range assignment
            for topic, parts in sorted(topic_partitions.items()):
                subscribed = [m for m in members
                              if topic in self._members[m]['subscriptions']]
                if not subscribed:
                    continue
                parts = sorted(parts)
                chunk_size = len(parts) // len(subscribed)
                remainder = len(parts) % len(subscribed)
                idx = 0
                for i, member in enumerate(subscribed):
                    n = chunk_size + (1 if i < remainder else 0)
                    for p in parts[idx:idx + n]:
                        tp = (topic, p)
                        self._members[member]['assignments'].append(tp)
                        self._assignments[tp] = member
                    idx += n

        return dict(self._assignments)

    def get_assignments(self, member_id=None):
        """Get partition assignments for a member, or all."""
        if member_id:
            info = self._members.get(member_id)
            if not info:
                return []
            return list(info['assignments'])
        return dict(self._assignments)

    def get_member_for_partition(self, topic, partition):
        """Which member owns this partition?"""
        return self._assignments.get((topic, partition))

    def commit_offset(self, topic, partition, offset):
        """Commit offset for this group."""
        self.offset_manager.commit(self.group_id, topic, partition, offset)

    def get_offset(self, topic, partition):
        """Get committed offset."""
        return self.offset_manager.get_committed(self.group_id, topic, partition)

    def members(self):
        return dict(self._members)

    def stats(self):
        return {
            'group_id': self.group_id,
            'members': self.member_count,
            'generation': self._generation,
            'strategy': self.strategy.name,
            'auto_commit': self.auto_commit,
            'semantics': self.semantics.name,
            'assignments': {f"{t}:{p}": m for (t, p), m in self._assignments.items()},
        }


# ============================================================
# Topic
# ============================================================

class Topic:
    """A named, partitioned log."""

    def __init__(self, name, num_partitions=3, replication_factor=1,
                 segment_size=1024 * 1024, retention_ms=7 * 24 * 3600 * 1000,
                 retention_policy=RetentionPolicy.DELETE):
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.retention_ms = retention_ms
        self.retention_policy = retention_policy
        self.created_at = time.time()

        self.partitions = {}
        for i in range(num_partitions):
            self.partitions[i] = Partition(
                topic_name=name, partition_id=i,
                segment_size=segment_size, retention_ms=retention_ms,
                retention_policy=retention_policy,
            )

    def get_partition(self, partition_id):
        p = self.partitions.get(partition_id)
        if p is None:
            raise LogError(f"Partition {partition_id} not found in topic {self.name}")
        return p

    def partition_ids(self):
        return sorted(self.partitions.keys())

    def stats(self):
        total_records = sum(p.log_end_offset - p.log_start_offset
                            for p in self.partitions.values())
        return {
            'name': self.name,
            'partitions': self.num_partitions,
            'replication_factor': self.replication_factor,
            'retention_policy': self.retention_policy.name,
            'total_records': total_records,
            'partition_stats': {pid: p.stats() for pid, p in self.partitions.items()},
        }


# ============================================================
# Partitioner
# ============================================================

class Partitioner:
    """Determines which partition a record goes to."""

    def __init__(self, strategy='round_robin'):
        self.strategy = strategy
        self._counter = 0

    def partition(self, record, num_partitions):
        """Return partition id for a record."""
        if num_partitions <= 0:
            return 0

        if record.key is not None:
            # Key-based: consistent hashing
            h = int(hashlib.md5(str(record.key).encode()).hexdigest(), 16)
            return h % num_partitions

        # Round-robin for keyless
        p = self._counter % num_partitions
        self._counter += 1
        return p


# ============================================================
# Producer
# ============================================================

class ProducerRecord:
    """A record to be sent by the producer."""
    def __init__(self, topic, value, key=None, partition=None, headers=None):
        self.topic = topic
        self.value = value
        self.key = key
        self.partition = partition  # explicit partition override
        self.headers = headers or {}


class Producer:
    """Produces records to topics."""

    def __init__(self, broker, ack_level=AckLevel.LEADER, partitioner=None,
                 batch_size=16384, linger_ms=0):
        self.broker = broker
        self.ack_level = ack_level
        self.partitioner = partitioner or Partitioner()
        self.batch_size = batch_size
        self.linger_ms = linger_ms

        self._buffer = {}  # (topic, partition) -> [records]
        self._stats = {'sent': 0, 'errors': 0, 'batches': 0}

    def send(self, producer_record):
        """Send a single record. Returns (topic, partition, offset)."""
        topic = self.broker.get_topic(producer_record.topic)
        if topic is None:
            raise LogError(f"Topic '{producer_record.topic}' not found")

        record = Record(
            key=producer_record.key,
            value=producer_record.value,
            headers=producer_record.headers,
        )

        # Determine partition
        if producer_record.partition is not None:
            part_id = producer_record.partition
        else:
            part_id = self.partitioner.partition(record, topic.num_partitions)

        partition = topic.get_partition(part_id)
        offset = partition.append(record)
        self._stats['sent'] += 1

        # Update high watermark (single-node: immediately committed)
        partition.update_high_watermark(partition.log_end_offset)

        return (producer_record.topic, part_id, offset)

    def send_batch(self, producer_records):
        """Send a batch of records. Returns list of (topic, partition, offset)."""
        results = []
        for pr in producer_records:
            results.append(self.send(pr))
        self._stats['batches'] += 1
        return results

    def flush(self):
        """Flush any buffered records (no-op in synchronous mode)."""
        pass

    def stats(self):
        return dict(self._stats)


# ============================================================
# Consumer
# ============================================================

class Consumer:
    """Consumes records from assigned partitions."""

    def __init__(self, broker, group_id=None, member_id=None,
                 auto_commit=True, auto_offset_reset='earliest',
                 max_poll_records=500):
        self.broker = broker
        self.group_id = group_id
        self.member_id = member_id or f"consumer-{id(self)}"
        self.auto_commit = auto_commit
        self.auto_offset_reset = auto_offset_reset
        self.max_poll_records = max_poll_records

        self._subscriptions = set()
        self._manual_assignments = []  # for assign() without group
        self._positions = {}  # (topic, partition) -> next offset to read
        self._stats = {'polled': 0, 'records': 0, 'commits': 0}

    def subscribe(self, topics):
        """Subscribe to topics (triggers rebalance via group)."""
        self._subscriptions = set(topics)
        if self.group_id:
            group = self.broker.get_or_create_group(self.group_id)
            group.join(self.member_id, self._subscriptions)

    def assign(self, topic_partitions):
        """Manually assign specific topic-partitions (no group coordination)."""
        self._manual_assignments = list(topic_partitions)
        for tp in topic_partitions:
            if tp not in self._positions:
                self._positions[tp] = 0

    def _get_assignments(self):
        """Get current partition assignments."""
        if self._manual_assignments:
            return self._manual_assignments
        if self.group_id:
            group = self.broker.get_group(self.group_id)
            if group:
                return group.get_assignments(self.member_id)
        return []

    def poll(self, max_records=None):
        """Poll for new records from assigned partitions.

        Returns {(topic, partition): [records]}
        """
        if max_records is None:
            max_records = self.max_poll_records

        assignments = self._get_assignments()
        result = {}

        per_partition = max(1, max_records // max(len(assignments), 1))

        for tp in assignments:
            topic_name, part_id = tp
            topic = self.broker.get_topic(topic_name)
            if not topic:
                continue
            partition = topic.get_partition(part_id)

            # Determine start position
            pos = self._positions.get(tp)
            if pos is None:
                # Check committed offset
                if self.group_id:
                    committed = self.broker.offset_manager.get_committed(
                        self.group_id, topic_name, part_id)
                    if committed >= 0:
                        pos = committed
                    elif self.auto_offset_reset == 'earliest':
                        pos = partition.log_start_offset
                    else:
                        pos = partition.log_end_offset
                else:
                    pos = 0 if self.auto_offset_reset == 'earliest' else partition.log_end_offset

            records = partition.read(pos, per_partition)
            if records:
                result[tp] = records
                # Advance position
                self._positions[tp] = records[-1].offset + 1

                # Auto-commit
                if self.auto_commit and self.group_id:
                    self.broker.offset_manager.commit(
                        self.group_id, topic_name, part_id,
                        records[-1].offset + 1)
                    self._stats['commits'] += 1

        self._stats['polled'] += 1
        self._stats['records'] += sum(len(recs) for recs in result.values())
        return result

    def commit(self, offsets=None):
        """Manually commit offsets. If None, commit current positions."""
        if not self.group_id:
            return

        if offsets:
            for (topic, part), offset in offsets.items():
                self.broker.offset_manager.commit(
                    self.group_id, topic, part, offset)
        else:
            for tp, pos in self._positions.items():
                topic, part = tp
                self.broker.offset_manager.commit(
                    self.group_id, topic, part, pos)

        self._stats['commits'] += 1

    def seek(self, topic, partition, offset):
        """Seek to a specific offset."""
        self._positions[(topic, partition)] = offset

    def seek_to_beginning(self, topic, partition):
        """Seek to the beginning of a partition."""
        t = self.broker.get_topic(topic)
        if t:
            p = t.get_partition(partition)
            self._positions[(topic, partition)] = p.log_start_offset

    def seek_to_end(self, topic, partition):
        """Seek to the end of a partition."""
        t = self.broker.get_topic(topic)
        if t:
            p = t.get_partition(partition)
            self._positions[(topic, partition)] = p.log_end_offset

    def position(self, topic, partition):
        """Get current position for a partition."""
        return self._positions.get((topic, partition), -1)

    def close(self):
        """Leave consumer group."""
        if self.group_id:
            group = self.broker.get_group(self.group_id)
            if group:
                group.leave(self.member_id)

    def stats(self):
        return dict(self._stats)


# ============================================================
# Partition Replicator (uses Raft)
# ============================================================

class PartitionReplicator:
    """Manages replication for a partition using Raft consensus.

    Each partition has a Raft group. The leader handles writes,
    followers replicate. Uses C201 Raft for leader election.
    """

    def __init__(self, partition, node_ids):
        self.partition = partition
        self.node_ids = list(node_ids)
        self.leader_id = node_ids[0] if node_ids else None
        self.follower_logs = {nid: 0 for nid in node_ids}  # node -> replicated offset
        self._term = 1

        # Update partition metadata
        partition.leader_id = self.leader_id
        partition.replica_ids = list(node_ids)
        partition.isr = list(node_ids)

    def replicate(self, record_offset):
        """Simulate replication of a record to followers."""
        # Leader is always up-to-date
        self.follower_logs[self.leader_id] = self.partition.log_end_offset
        for nid in self.node_ids:
            if nid != self.leader_id:
                self.follower_logs[nid] = record_offset + 1

        # Update ISR and high watermark
        offsets = list(self.follower_logs.values())
        if offsets:
            min_offset = min(offsets)
            self.partition.update_high_watermark(min_offset)
            self.partition.isr = list(self.node_ids)

    def elect_leader(self, new_leader=None):
        """Trigger leader election."""
        self._term += 1
        if new_leader and new_leader in self.node_ids:
            self.leader_id = new_leader
        else:
            # Pick follower with highest replicated offset
            candidates = [(off, nid) for nid, off in self.follower_logs.items()
                          if nid != self.leader_id]
            if candidates:
                candidates.sort(reverse=True)
                self.leader_id = candidates[0][1]
        self.partition.leader_id = self.leader_id
        return self.leader_id

    def remove_from_isr(self, node_id):
        """Remove a node from ISR (fallen behind or failed)."""
        if node_id in self.partition.isr:
            self.partition.isr.remove(node_id)

    def add_to_isr(self, node_id):
        """Add node back to ISR once caught up."""
        if node_id not in self.partition.isr and node_id in self.node_ids:
            self.partition.isr.append(node_id)

    def status(self):
        return {
            'leader': self.leader_id,
            'term': self._term,
            'replicas': list(self.node_ids),
            'isr': list(self.partition.isr),
            'follower_offsets': dict(self.follower_logs),
            'high_watermark': self.partition.high_watermark,
        }


# ============================================================
# Broker
# ============================================================

class Broker:
    """Central broker managing topics, consumer groups, and offsets.

    In a real system this would be distributed across nodes.
    Here it provides the coordination layer.
    """

    def __init__(self, broker_id="broker-1"):
        self.broker_id = broker_id
        self.topics = {}
        self.consumer_groups = {}
        self.offset_manager = OffsetManager()
        self._replicators = {}  # (topic, partition) -> PartitionReplicator
        self.created_at = time.time()

    # -- Topic management --

    def create_topic(self, name, num_partitions=3, replication_factor=1,
                     segment_size=1024 * 1024, retention_ms=7 * 24 * 3600 * 1000,
                     retention_policy=RetentionPolicy.DELETE):
        """Create a new topic."""
        if name in self.topics:
            raise LogError(f"Topic '{name}' already exists")
        topic = Topic(name, num_partitions, replication_factor,
                      segment_size, retention_ms, retention_policy)
        self.topics[name] = topic
        return topic

    def delete_topic(self, name):
        """Delete a topic."""
        if name not in self.topics:
            raise LogError(f"Topic '{name}' not found")
        del self.topics[name]
        # Clean up replicators
        to_remove = [k for k in self._replicators if k[0] == name]
        for k in to_remove:
            del self._replicators[k]
        return True

    def get_topic(self, name):
        return self.topics.get(name)

    def list_topics(self):
        return list(self.topics.keys())

    # -- Consumer group management --

    def get_or_create_group(self, group_id, **kwargs):
        if group_id not in self.consumer_groups:
            self.consumer_groups[group_id] = ConsumerGroup(
                group_id, self.offset_manager, **kwargs)
        return self.consumer_groups[group_id]

    def get_group(self, group_id):
        return self.consumer_groups.get(group_id)

    def delete_group(self, group_id):
        if group_id in self.consumer_groups:
            del self.consumer_groups[group_id]
            self.offset_manager.delete_group(group_id)
            return True
        return False

    def list_groups(self):
        return list(self.consumer_groups.keys())

    def rebalance_group(self, group_id):
        """Trigger rebalance for a consumer group."""
        group = self.consumer_groups.get(group_id)
        if not group:
            raise LogError(f"Group '{group_id}' not found")

        # Build topic_partitions map from subscriptions
        topic_partitions = {}
        for info in group._members.values():
            for topic_name in info['subscriptions']:
                if topic_name in self.topics:
                    topic = self.topics[topic_name]
                    topic_partitions[topic_name] = topic.partition_ids()

        return group.rebalance(topic_partitions)

    # -- Replication --

    def setup_replication(self, topic_name, partition_id, node_ids):
        """Set up replication for a partition."""
        topic = self.get_topic(topic_name)
        if not topic:
            raise LogError(f"Topic '{topic_name}' not found")
        partition = topic.get_partition(partition_id)
        replicator = PartitionReplicator(partition, node_ids)
        self._replicators[(topic_name, partition_id)] = replicator
        return replicator

    def get_replicator(self, topic_name, partition_id):
        return self._replicators.get((topic_name, partition_id))

    # -- Lag calculation --

    def consumer_lag(self, group_id):
        """Calculate lag for all partitions in a consumer group."""
        group = self.consumer_groups.get(group_id)
        if not group:
            return {}

        lag = {}
        assignments = group.get_assignments()
        for (topic_name, part_id), member in assignments.items():
            topic = self.get_topic(topic_name)
            if topic:
                partition = topic.get_partition(part_id)
                l = self.offset_manager.lag(
                    group_id, topic_name, part_id, partition.log_end_offset)
                lag[(topic_name, part_id)] = {
                    'member': member, 'lag': l,
                    'committed': self.offset_manager.get_committed(
                        group_id, topic_name, part_id),
                    'log_end': partition.log_end_offset,
                }
        return lag

    # -- Maintenance --

    def apply_retention(self, topic_name=None):
        """Apply retention policy to topics."""
        targets = [topic_name] if topic_name else list(self.topics.keys())
        removed = 0
        for name in targets:
            topic = self.topics.get(name)
            if topic:
                for p in topic.partitions.values():
                    removed += p.apply_retention()
        return removed

    def compact(self, topic_name=None):
        """Compact topics with COMPACT retention policy."""
        targets = [topic_name] if topic_name else list(self.topics.keys())
        compacted = 0
        for name in targets:
            topic = self.topics.get(name)
            if topic and topic.retention_policy == RetentionPolicy.COMPACT:
                for p in topic.partitions.values():
                    compacted += p.compact()
        return compacted

    # -- Stats --

    def stats(self):
        total_records = 0
        total_bytes = 0
        for topic in self.topics.values():
            for p in topic.partitions.values():
                total_records += p.log_end_offset - p.log_start_offset
                total_bytes += p.bytes_written

        return {
            'broker_id': self.broker_id,
            'topics': len(self.topics),
            'consumer_groups': len(self.consumer_groups),
            'total_records': total_records,
            'total_bytes': total_bytes,
            'topic_list': self.list_topics(),
            'group_list': self.list_groups(),
        }


# ============================================================
# Dead Letter Queue
# ============================================================

class DeadLetterQueue:
    """Stores records that failed processing."""

    def __init__(self, broker, dlq_topic="_dlq"):
        self.broker = broker
        self.dlq_topic = dlq_topic
        self._failed_count = 0

        # Create DLQ topic if needed
        if not broker.get_topic(dlq_topic):
            broker.create_topic(dlq_topic, num_partitions=1)

    def send(self, record, error_reason, source_topic, source_partition):
        """Send a failed record to the DLQ."""
        dlq_record = ProducerRecord(
            topic=self.dlq_topic,
            key=record.key,
            value=record.value,
            headers={
                **record.headers,
                '_dlq_error': str(error_reason),
                '_dlq_source_topic': source_topic,
                '_dlq_source_partition': str(source_partition),
                '_dlq_source_offset': str(record.offset),
                '_dlq_timestamp': str(time.time()),
            }
        )
        producer = Producer(self.broker)
        producer.send(dlq_record)
        self._failed_count += 1

    def read_failed(self, max_records=100):
        """Read records from the DLQ."""
        topic = self.broker.get_topic(self.dlq_topic)
        if not topic:
            return []
        partition = topic.get_partition(0)
        return partition.read(partition.log_start_offset, max_records)

    @property
    def count(self):
        topic = self.broker.get_topic(self.dlq_topic)
        if not topic:
            return 0
        p = topic.get_partition(0)
        return p.log_end_offset - p.log_start_offset

    def stats(self):
        return {
            'dlq_topic': self.dlq_topic,
            'total_failed': self._failed_count,
            'current_count': self.count,
        }


# ============================================================
# Message Queue (High-Level API)
# ============================================================

class MessageQueue:
    """High-level message queue API wrapping the distributed log.

    Provides pub/sub semantics on top of the partitioned log.
    """

    def __init__(self, broker=None):
        self.broker = broker or Broker()
        self._producers = {}
        self._consumers = {}
        self._dlqs = {}

    def create_topic(self, name, partitions=3, replication=1, **kwargs):
        """Create a topic."""
        return self.broker.create_topic(name, partitions, replication, **kwargs)

    def delete_topic(self, name):
        return self.broker.delete_topic(name)

    def publish(self, topic, value, key=None, partition=None, headers=None):
        """Publish a message to a topic. Returns (topic, partition, offset)."""
        if topic not in self._producers:
            self._producers[topic] = Producer(self.broker)
        pr = ProducerRecord(topic, value, key=key, partition=partition, headers=headers)
        return self._producers[topic].send(pr)

    def publish_batch(self, topic, messages):
        """Publish multiple messages. messages: list of (key, value) or value.
        Returns list of (topic, partition, offset).
        """
        if topic not in self._producers:
            self._producers[topic] = Producer(self.broker)
        records = []
        for msg in messages:
            if isinstance(msg, tuple):
                key, value = msg
            else:
                key, value = None, msg
            records.append(ProducerRecord(topic, value, key=key))
        return self._producers[topic].send_batch(records)

    def subscribe(self, consumer_id, topics, group_id=None):
        """Create a consumer subscribed to topics."""
        consumer = Consumer(self.broker, group_id=group_id, member_id=consumer_id)
        consumer.subscribe(topics if isinstance(topics, list) else [topics])
        self._consumers[consumer_id] = consumer

        # Trigger rebalance if in a group
        if group_id:
            self.broker.rebalance_group(group_id)

        return consumer

    def consume(self, consumer_id, max_records=100):
        """Poll for messages from a consumer. Returns list of records."""
        consumer = self._consumers.get(consumer_id)
        if not consumer:
            raise LogError(f"Consumer '{consumer_id}' not found")

        result = consumer.poll(max_records)
        # Flatten to list
        all_records = []
        for records in result.values():
            all_records.extend(records)
        return sorted(all_records, key=lambda r: (r.timestamp, r.offset))

    def commit(self, consumer_id, offsets=None):
        """Commit offsets for a consumer."""
        consumer = self._consumers.get(consumer_id)
        if consumer:
            consumer.commit(offsets)

    def seek(self, consumer_id, topic, partition, offset):
        """Seek a consumer to a specific offset."""
        consumer = self._consumers.get(consumer_id)
        if consumer:
            consumer.seek(topic, partition, offset)

    def consumer_lag(self, group_id):
        """Get consumer lag for a group."""
        return self.broker.consumer_lag(group_id)

    def setup_dlq(self, name="_dlq"):
        """Set up a dead letter queue."""
        dlq = DeadLetterQueue(self.broker, name)
        self._dlqs[name] = dlq
        return dlq

    def topic_stats(self, topic_name):
        topic = self.broker.get_topic(topic_name)
        if topic:
            return topic.stats()
        return None

    def stats(self):
        return self.broker.stats()
