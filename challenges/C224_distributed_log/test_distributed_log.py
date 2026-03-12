"""Tests for C224: Distributed Log / Message Queue."""

import time
import pytest
from distributed_log import (
    Record, RecordBatch, LogSegment, Partition, Topic, Partitioner,
    Producer, ProducerRecord, Consumer, ConsumerGroup, OffsetManager,
    PartitionReplicator, Broker, DeadLetterQueue, MessageQueue,
    LogError, DeliverySemantics, AckLevel, RetentionPolicy,
    RebalanceStrategy,
)


# ============================================================
# Record Tests
# ============================================================

class TestRecord:
    def test_create_record(self):
        r = Record(key="k1", value="hello")
        assert r.key == "k1"
        assert r.value == "hello"
        assert r.offset == -1
        assert r.timestamp > 0

    def test_record_no_key(self):
        r = Record(key=None, value=42)
        assert r.key is None
        assert r.value == 42

    def test_record_headers(self):
        r = Record(key="k", value="v", headers={"source": "test"})
        assert r.headers["source"] == "test"

    def test_record_serialize_deserialize(self):
        r = Record(key="k1", value={"data": [1, 2, 3]}, headers={"h": "v"})
        r.offset = 5
        r.partition = 2
        data = r.serialize()
        r2 = Record.deserialize(data)
        assert r2.key == "k1"
        assert r2.value == {"data": [1, 2, 3]}
        assert r2.offset == 5
        assert r2.partition == 2
        assert r2.headers == {"h": "v"}

    def test_record_size_bytes(self):
        r = Record(key="key", value="value")
        assert r.size_bytes() > 0

    def test_record_batch(self):
        batch = RecordBatch(base_offset=10)
        r1 = Record(key="a", value=1)
        r1.offset = 10
        r2 = Record(key="b", value=2)
        r2.offset = 11
        batch.append(r1)
        batch.append(r2)
        assert len(batch) == 2
        assert batch.max_offset == 11
        assert list(batch) == [r1, r2]


# ============================================================
# LogSegment Tests
# ============================================================

class TestLogSegment:
    def test_create_segment(self):
        seg = LogSegment(base_offset=0)
        assert seg.base_offset == 0
        assert seg.next_offset == 0
        assert seg.count == 0

    def test_append_and_read(self):
        seg = LogSegment(base_offset=0)
        r = Record(key="k", value="v")
        off = seg.append(r)
        assert off == 0
        assert seg.count == 1
        assert seg.next_offset == 1
        read_r = seg.read(0)
        assert read_r.key == "k"
        assert read_r.value == "v"

    def test_sequential_offsets(self):
        seg = LogSegment(base_offset=10, max_size=10000)
        for i in range(5):
            off = seg.append(Record(key=f"k{i}", value=i))
            assert off == 10 + i

    def test_read_nonexistent(self):
        seg = LogSegment(base_offset=0)
        assert seg.read(999) is None

    def test_read_range(self):
        seg = LogSegment(base_offset=0, max_size=10000)
        for i in range(10):
            seg.append(Record(key=f"k{i}", value=i))
        recs = seg.read_range(3, max_records=4)
        assert len(recs) == 4
        assert recs[0].offset == 3
        assert recs[-1].offset == 6

    def test_segment_full(self):
        seg = LogSegment(base_offset=0, max_size=200)
        count = 0
        while not seg.is_full:
            seg.append(Record(key="k", value="x" * 10))
            count += 1
            if count > 100:
                break
        assert seg.is_full
        assert seg.closed

    def test_closed_segment_rejects_write(self):
        seg = LogSegment(base_offset=0)
        seg.close()
        with pytest.raises(LogError):
            seg.append(Record(key="k", value="v"))

    def test_compact(self):
        seg = LogSegment(base_offset=0, max_size=10000)
        seg.append(Record(key="a", value=1))
        seg.append(Record(key="b", value=2))
        seg.append(Record(key="a", value=3))  # supersedes first
        seg.compact()
        assert seg.count == 2
        # "a" should have value 3 (latest)
        found_a = [r for r in seg.records if r.key == "a"]
        assert found_a[0].value == 3


# ============================================================
# Partition Tests
# ============================================================

class TestPartition:
    def test_create_partition(self):
        p = Partition("test-topic", 0)
        assert p.topic_name == "test-topic"
        assert p.partition_id == 0
        assert p.log_start_offset == 0
        assert p.log_end_offset == 0

    def test_append_and_read(self):
        p = Partition("t", 0)
        off = p.append(Record(key="k", value="v"))
        assert off == 0
        assert p.log_end_offset == 1
        recs = p.read(0)
        assert len(recs) == 1
        assert recs[0].value == "v"

    def test_many_records(self):
        p = Partition("t", 0)
        for i in range(100):
            p.append(Record(key=f"k{i}", value=i))
        assert p.log_end_offset == 100
        recs = p.read(50, max_records=10)
        assert len(recs) == 10
        assert recs[0].offset == 50

    def test_get_record(self):
        p = Partition("t", 0)
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        r = p.get_record(3)
        assert r.value == 3

    def test_high_watermark(self):
        p = Partition("t", 0)
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        p.update_high_watermark(3)
        assert p.high_watermark == 3
        p.update_high_watermark(10)  # capped at log_end
        assert p.high_watermark == 5

    def test_truncate(self):
        p = Partition("t", 0)
        for i in range(10):
            p.append(Record(key=f"k{i}", value=i))
        p.truncate(5)
        assert p.log_end_offset == 5
        assert p.get_record(4) is not None
        assert p.get_record(5) is None

    def test_append_batch(self):
        p = Partition("t", 0)
        records = [Record(key=f"k{i}", value=i) for i in range(5)]
        offsets = p.append_batch(records)
        assert offsets == [0, 1, 2, 3, 4]

    def test_segment_rollover(self):
        p = Partition("t", 0, segment_size=200)
        for i in range(50):
            p.append(Record(key=f"k{i}", value="x" * 10))
        assert len(p._segments) > 1
        # All records should still be readable
        recs = p.read(0, max_records=50)
        assert len(recs) == 50

    def test_compact(self):
        p = Partition("t", 0, retention_policy=RetentionPolicy.COMPACT)
        p.append(Record(key="a", value=1))
        p.append(Record(key="b", value=2))
        p.append(Record(key="a", value=3))
        p.append(Record(key="c", value=4))
        p.append(Record(key="b", value=5))
        removed = p.compact()
        assert removed == 2  # two old versions removed
        recs = p.read(0, 100)
        keys_values = {r.key: r.value for r in recs}
        assert keys_values["a"] == 3
        assert keys_values["b"] == 5
        assert keys_values["c"] == 4

    def test_retention(self):
        p = Partition("t", 0, retention_ms=1000)
        # Create segments with old timestamps
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        p._segments[0].created_at = time.time() - 100  # old
        removed = p.apply_retention(now=time.time())
        # Active segment should never be removed
        assert len(p._segments) >= 1

    def test_stats(self):
        p = Partition("t", 0)
        p.append(Record(key="k", value="v"))
        s = p.stats()
        assert s['topic'] == 't'
        assert s['partition'] == 0
        assert s['total_writes'] == 1

    def test_read_beyond_end(self):
        p = Partition("t", 0)
        p.append(Record(key="k", value="v"))
        recs = p.read(100)
        assert recs == []

    def test_read_before_start(self):
        p = Partition("t", 0)
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        recs = p.read(-5, max_records=3)
        assert len(recs) == 3
        assert recs[0].offset == 0


# ============================================================
# OffsetManager Tests
# ============================================================

class TestOffsetManager:
    def test_commit_and_get(self):
        om = OffsetManager()
        om.commit("g1", "topic", 0, 5)
        assert om.get_committed("g1", "topic", 0) == 5

    def test_uncommitted(self):
        om = OffsetManager()
        assert om.get_committed("g1", "topic", 0) == -1

    def test_get_all_committed(self):
        om = OffsetManager()
        om.commit("g1", "t1", 0, 10)
        om.commit("g1", "t1", 1, 20)
        om.commit("g1", "t2", 0, 5)
        all_c = om.get_all_committed("g1")
        assert len(all_c) == 3
        assert all_c[("t1", 0)] == 10

    def test_reset(self):
        om = OffsetManager()
        om.commit("g1", "t", 0, 10)
        om.reset("g1", "t", 0, 0)
        assert om.get_committed("g1", "t", 0) == 0

    def test_delete_group(self):
        om = OffsetManager()
        om.commit("g1", "t", 0, 5)
        om.delete_group("g1")
        assert om.get_committed("g1", "t", 0) == -1

    def test_list_groups(self):
        om = OffsetManager()
        om.commit("g1", "t", 0, 1)
        om.commit("g2", "t", 0, 2)
        assert sorted(om.list_groups()) == ["g1", "g2"]

    def test_lag(self):
        om = OffsetManager()
        om.commit("g1", "t", 0, 5)
        assert om.lag("g1", "t", 0, 10) == 5
        assert om.lag("g1", "t", 0, 5) == 0

    def test_lag_no_commit(self):
        om = OffsetManager()
        assert om.lag("g1", "t", 0, 10) == 10


# ============================================================
# ConsumerGroup Tests
# ============================================================

class TestConsumerGroup:
    def test_join_and_leave(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.join("c1", ["topic1"])
        assert cg.member_count == 1
        cg.leave("c1")
        assert cg.member_count == 0

    def test_round_robin_rebalance(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om, strategy=RebalanceStrategy.ROUND_ROBIN)
        cg.join("c1", ["t1"])
        cg.join("c2", ["t1"])
        assignments = cg.rebalance({"t1": [0, 1, 2, 3]})
        c1_parts = cg.get_assignments("c1")
        c2_parts = cg.get_assignments("c2")
        assert len(c1_parts) == 2
        assert len(c2_parts) == 2
        # All partitions assigned
        all_assigned = set(tp for tp in assignments.keys())
        assert len(all_assigned) == 4

    def test_range_rebalance(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om, strategy=RebalanceStrategy.RANGE)
        cg.join("c1", ["t1"])
        cg.join("c2", ["t1"])
        cg.rebalance({"t1": [0, 1, 2, 3, 4]})
        c1_parts = cg.get_assignments("c1")
        c2_parts = cg.get_assignments("c2")
        # Range: c1 gets 3, c2 gets 2 (5/2=2 remainder 1)
        assert len(c1_parts) == 3
        assert len(c2_parts) == 2

    def test_rebalance_generation(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.join("c1", ["t1"])
        assert cg.generation == 0
        cg.rebalance({"t1": [0, 1]})
        assert cg.generation == 1

    def test_get_member_for_partition(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.join("c1", ["t1"])
        cg.rebalance({"t1": [0]})
        assert cg.get_member_for_partition("t1", 0) == "c1"

    def test_commit_offset(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.commit_offset("t1", 0, 10)
        assert cg.get_offset("t1", 0) == 10

    def test_multi_topic_subscription(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.join("c1", ["t1", "t2"])
        cg.join("c2", ["t1"])
        cg.rebalance({"t1": [0, 1], "t2": [0, 1]})
        c1_parts = cg.get_assignments("c1")
        c2_parts = cg.get_assignments("c2")
        # c1 subscribed to both, c2 only t1
        # t2 partitions can only go to c1
        t2_assigned = [tp for tp in c1_parts if tp[0] == "t2"]
        assert len(t2_assigned) > 0

    def test_stats(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        cg.join("c1", ["t1"])
        s = cg.stats()
        assert s['group_id'] == "g1"
        assert s['members'] == 1

    def test_empty_rebalance(self):
        om = OffsetManager()
        cg = ConsumerGroup("g1", om)
        result = cg.rebalance({"t1": [0, 1]})
        assert result == {}


# ============================================================
# Topic Tests
# ============================================================

class TestTopic:
    def test_create_topic(self):
        t = Topic("events", num_partitions=4)
        assert t.name == "events"
        assert t.num_partitions == 4
        assert len(t.partitions) == 4

    def test_get_partition(self):
        t = Topic("events", num_partitions=3)
        p = t.get_partition(1)
        assert p.partition_id == 1

    def test_invalid_partition(self):
        t = Topic("events", num_partitions=3)
        with pytest.raises(LogError):
            t.get_partition(99)

    def test_partition_ids(self):
        t = Topic("events", num_partitions=3)
        assert t.partition_ids() == [0, 1, 2]

    def test_stats(self):
        t = Topic("events", num_partitions=2)
        t.get_partition(0).append(Record(key="k", value="v"))
        s = t.stats()
        assert s['name'] == "events"
        assert s['total_records'] == 1
        assert len(s['partition_stats']) == 2


# ============================================================
# Partitioner Tests
# ============================================================

class TestPartitioner:
    def test_round_robin(self):
        p = Partitioner(strategy='round_robin')
        parts = set()
        for _ in range(6):
            r = Record(key=None, value="v")
            parts.add(p.partition(r, 3))
        assert parts == {0, 1, 2}

    def test_key_based(self):
        p = Partitioner()
        r1 = Record(key="user-123", value="a")
        r2 = Record(key="user-123", value="b")
        assert p.partition(r1, 10) == p.partition(r2, 10)

    def test_different_keys_distribute(self):
        p = Partitioner()
        parts = set()
        for i in range(100):
            r = Record(key=f"key-{i}", value=i)
            parts.add(p.partition(r, 10))
        # With 100 different keys, should hit multiple partitions
        assert len(parts) > 1


# ============================================================
# Producer Tests
# ============================================================

class TestProducer:
    def setup_method(self):
        self.broker = Broker()
        self.broker.create_topic("events", num_partitions=3)

    def test_send(self):
        producer = Producer(self.broker)
        topic, part, offset = producer.send(ProducerRecord("events", "hello"))
        assert topic == "events"
        assert 0 <= part <= 2
        assert offset == 0

    def test_send_with_key(self):
        producer = Producer(self.broker)
        t1, p1, _ = producer.send(ProducerRecord("events", "v1", key="k1"))
        t2, p2, _ = producer.send(ProducerRecord("events", "v2", key="k1"))
        assert p1 == p2  # same key -> same partition

    def test_send_explicit_partition(self):
        producer = Producer(self.broker)
        _, part, _ = producer.send(ProducerRecord("events", "v", partition=2))
        assert part == 2

    def test_send_batch(self):
        producer = Producer(self.broker)
        records = [ProducerRecord("events", f"v{i}", key=f"k{i}") for i in range(10)]
        results = producer.send_batch(records)
        assert len(results) == 10

    def test_send_nonexistent_topic(self):
        producer = Producer(self.broker)
        with pytest.raises(LogError):
            producer.send(ProducerRecord("nonexistent", "v"))

    def test_stats(self):
        producer = Producer(self.broker)
        producer.send(ProducerRecord("events", "v"))
        s = producer.stats()
        assert s['sent'] == 1


# ============================================================
# Consumer Tests
# ============================================================

class TestConsumer:
    def setup_method(self):
        self.broker = Broker()
        self.broker.create_topic("events", num_partitions=3)
        self.producer = Producer(self.broker)
        for i in range(30):
            self.producer.send(ProducerRecord("events", f"msg-{i}", key=f"k{i}"))

    def test_manual_assign_and_poll(self):
        consumer = Consumer(self.broker, auto_offset_reset='earliest')
        consumer.assign([("events", 0)])
        result = consumer.poll()
        assert len(result) > 0
        records = result[("events", 0)]
        assert len(records) > 0

    def test_subscribe_with_group(self):
        consumer = Consumer(self.broker, group_id="g1", member_id="c1")
        consumer.subscribe(["events"])
        self.broker.rebalance_group("g1")
        result = consumer.poll()
        total = sum(len(recs) for recs in result.values())
        assert total > 0

    def test_seek(self):
        consumer = Consumer(self.broker)
        consumer.assign([("events", 0)])
        topic = self.broker.get_topic("events")
        end = topic.get_partition(0).log_end_offset
        consumer.seek("events", 0, end)
        result = consumer.poll()
        # Should get nothing after seeking to end
        recs = result.get(("events", 0), [])
        assert len(recs) == 0

    def test_seek_to_beginning(self):
        consumer = Consumer(self.broker, auto_offset_reset='latest')
        consumer.assign([("events", 0)])
        consumer.seek_to_beginning("events", 0)
        assert consumer.position("events", 0) == 0

    def test_seek_to_end(self):
        consumer = Consumer(self.broker)
        consumer.assign([("events", 0)])
        consumer.seek_to_end("events", 0)
        pos = consumer.position("events", 0)
        end = self.broker.get_topic("events").get_partition(0).log_end_offset
        assert pos == end

    def test_manual_commit(self):
        consumer = Consumer(self.broker, group_id="g1", member_id="c1",
                            auto_commit=False)
        consumer.subscribe(["events"])
        self.broker.rebalance_group("g1")
        consumer.poll()
        consumer.commit()
        # Verify offsets are committed
        for tp, pos in consumer._positions.items():
            committed = self.broker.offset_manager.get_committed("g1", tp[0], tp[1])
            assert committed == pos

    def test_auto_commit(self):
        consumer = Consumer(self.broker, group_id="g1", member_id="c1",
                            auto_commit=True)
        consumer.subscribe(["events"])
        self.broker.rebalance_group("g1")
        consumer.poll()
        # Auto-commit should have saved offsets
        for tp in consumer._positions:
            committed = self.broker.offset_manager.get_committed("g1", tp[0], tp[1])
            assert committed >= 0

    def test_close(self):
        consumer = Consumer(self.broker, group_id="g1", member_id="c1")
        consumer.subscribe(["events"])
        group = self.broker.get_group("g1")
        assert group.member_count == 1
        consumer.close()
        assert group.member_count == 0

    def test_stats(self):
        consumer = Consumer(self.broker)
        consumer.assign([("events", 0)])
        consumer.poll()
        s = consumer.stats()
        assert s['polled'] == 1
        assert s['records'] > 0


# ============================================================
# PartitionReplicator Tests
# ============================================================

class TestPartitionReplicator:
    def test_create_replicator(self):
        p = Partition("t", 0)
        rep = PartitionReplicator(p, ["n1", "n2", "n3"])
        assert rep.leader_id == "n1"
        assert p.leader_id == "n1"
        assert len(p.replica_ids) == 3

    def test_replicate(self):
        p = Partition("t", 0)
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        rep = PartitionReplicator(p, ["n1", "n2", "n3"])
        rep.replicate(4)  # replicate up to offset 4
        assert p.high_watermark == 5  # all followers at offset 5

    def test_elect_leader(self):
        p = Partition("t", 0)
        for i in range(5):
            p.append(Record(key=f"k{i}", value=i))
        rep = PartitionReplicator(p, ["n1", "n2", "n3"])
        rep.follower_logs["n2"] = 5
        rep.follower_logs["n3"] = 3
        new_leader = rep.elect_leader()
        assert new_leader == "n2"  # highest replicated

    def test_elect_specific_leader(self):
        p = Partition("t", 0)
        rep = PartitionReplicator(p, ["n1", "n2", "n3"])
        rep.elect_leader("n3")
        assert rep.leader_id == "n3"

    def test_isr_management(self):
        p = Partition("t", 0)
        rep = PartitionReplicator(p, ["n1", "n2", "n3"])
        assert len(p.isr) == 3
        rep.remove_from_isr("n3")
        assert "n3" not in p.isr
        rep.add_to_isr("n3")
        assert "n3" in p.isr

    def test_status(self):
        p = Partition("t", 0)
        rep = PartitionReplicator(p, ["n1", "n2"])
        s = rep.status()
        assert s['leader'] == "n1"
        assert len(s['replicas']) == 2


# ============================================================
# Broker Tests
# ============================================================

class TestBroker:
    def test_create_topic(self):
        b = Broker()
        t = b.create_topic("events", num_partitions=4)
        assert t.name == "events"
        assert t.num_partitions == 4

    def test_create_duplicate_topic(self):
        b = Broker()
        b.create_topic("events")
        with pytest.raises(LogError):
            b.create_topic("events")

    def test_delete_topic(self):
        b = Broker()
        b.create_topic("events")
        b.delete_topic("events")
        assert b.get_topic("events") is None

    def test_delete_nonexistent_topic(self):
        b = Broker()
        with pytest.raises(LogError):
            b.delete_topic("nonexistent")

    def test_list_topics(self):
        b = Broker()
        b.create_topic("t1")
        b.create_topic("t2")
        assert sorted(b.list_topics()) == ["t1", "t2"]

    def test_consumer_group_lifecycle(self):
        b = Broker()
        g = b.get_or_create_group("g1")
        assert g.group_id == "g1"
        assert b.get_group("g1") is g
        b.delete_group("g1")
        assert b.get_group("g1") is None

    def test_rebalance_group(self):
        b = Broker()
        b.create_topic("events", num_partitions=4)
        g = b.get_or_create_group("g1")
        g.join("c1", ["events"])
        g.join("c2", ["events"])
        assignments = b.rebalance_group("g1")
        assert len(assignments) == 4

    def test_rebalance_nonexistent_group(self):
        b = Broker()
        with pytest.raises(LogError):
            b.rebalance_group("nonexistent")

    def test_setup_replication(self):
        b = Broker()
        b.create_topic("events")
        rep = b.setup_replication("events", 0, ["n1", "n2", "n3"])
        assert rep.leader_id == "n1"
        assert b.get_replicator("events", 0) is rep

    def test_consumer_lag(self):
        b = Broker()
        b.create_topic("events", num_partitions=2)
        producer = Producer(b)
        for i in range(10):
            producer.send(ProducerRecord("events", f"v{i}", partition=0))

        g = b.get_or_create_group("g1")
        g.join("c1", ["events"])
        b.rebalance_group("g1")
        b.offset_manager.commit("g1", "events", 0, 5)

        lag = b.consumer_lag("g1")
        assert ("events", 0) in lag
        assert lag[("events", 0)]['lag'] == 5

    def test_apply_retention(self):
        b = Broker()
        b.create_topic("events")
        removed = b.apply_retention("events")
        assert removed >= 0

    def test_compact(self):
        b = Broker()
        b.create_topic("compact-topic", retention_policy=RetentionPolicy.COMPACT)
        producer = Producer(b)
        producer.send(ProducerRecord("compact-topic", "v1", key="k1"))
        producer.send(ProducerRecord("compact-topic", "v2", key="k1"))
        compacted = b.compact("compact-topic")
        assert compacted >= 0

    def test_stats(self):
        b = Broker()
        b.create_topic("events")
        producer = Producer(b)
        producer.send(ProducerRecord("events", "v"))
        s = b.stats()
        assert s['topics'] == 1
        assert s['total_records'] == 1

    def test_list_groups(self):
        b = Broker()
        b.get_or_create_group("g1")
        b.get_or_create_group("g2")
        assert sorted(b.list_groups()) == ["g1", "g2"]


# ============================================================
# DeadLetterQueue Tests
# ============================================================

class TestDeadLetterQueue:
    def test_create_dlq(self):
        b = Broker()
        dlq = DeadLetterQueue(b)
        assert b.get_topic("_dlq") is not None

    def test_send_to_dlq(self):
        b = Broker()
        dlq = DeadLetterQueue(b)
        r = Record(key="k", value="v", offset=5)
        dlq.send(r, "processing error", "events", 0)
        assert dlq.count == 1

    def test_read_failed(self):
        b = Broker()
        dlq = DeadLetterQueue(b)
        r = Record(key="k", value="v", offset=5)
        dlq.send(r, "error msg", "events", 2)
        failed = dlq.read_failed()
        assert len(failed) == 1
        assert failed[0].headers['_dlq_error'] == "error msg"
        assert failed[0].headers['_dlq_source_topic'] == "events"

    def test_dlq_stats(self):
        b = Broker()
        dlq = DeadLetterQueue(b)
        s = dlq.stats()
        assert s['total_failed'] == 0
        assert s['current_count'] == 0


# ============================================================
# MessageQueue (High-Level API) Tests
# ============================================================

class TestMessageQueue:
    def test_create_and_publish(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=3)
        topic, part, offset = mq.publish("events", "hello", key="k1")
        assert topic == "events"
        assert offset == 0

    def test_publish_batch(self):
        mq = MessageQueue()
        mq.create_topic("events")
        results = mq.publish_batch("events", [("k1", "v1"), ("k2", "v2"), "v3"])
        assert len(results) == 3

    def test_subscribe_and_consume(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=2)
        for i in range(10):
            mq.publish("events", f"msg-{i}", key=f"k{i}")

        mq.subscribe("c1", ["events"], group_id="g1")
        records = mq.consume("c1")
        assert len(records) > 0

    def test_multiple_consumers_in_group(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=4)
        for i in range(20):
            mq.publish("events", f"msg-{i}", key=f"k{i}")

        mq.subscribe("c1", ["events"], group_id="g1")
        mq.subscribe("c2", ["events"], group_id="g1")
        r1 = mq.consume("c1")
        r2 = mq.consume("c2")
        # Both should get records (different partitions)
        total = len(r1) + len(r2)
        assert total == 20

    def test_consumer_lag(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=1)
        for i in range(10):
            mq.publish("events", f"msg-{i}", partition=0)

        mq.subscribe("c1", ["events"], group_id="g1")
        mq.consume("c1", max_records=5)
        lag = mq.consumer_lag("g1")
        # Should have some lag (consumed 5 of 10)
        assert len(lag) > 0

    def test_seek(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=1)
        for i in range(10):
            mq.publish("events", f"msg-{i}", partition=0)

        mq.subscribe("c1", ["events"], group_id="g1")
        mq.consume("c1")  # consume all
        mq.seek("c1", "events", 0, 5)
        records = mq.consume("c1")
        assert len(records) == 5
        assert records[0].offset == 5

    def test_delete_topic(self):
        mq = MessageQueue()
        mq.create_topic("events")
        mq.delete_topic("events")
        assert mq.broker.get_topic("events") is None

    def test_topic_stats(self):
        mq = MessageQueue()
        mq.create_topic("events")
        mq.publish("events", "v")
        s = mq.topic_stats("events")
        assert s['name'] == "events"
        assert s['total_records'] == 1

    def test_overall_stats(self):
        mq = MessageQueue()
        mq.create_topic("t1")
        mq.create_topic("t2")
        s = mq.stats()
        assert s['topics'] == 2

    def test_dlq_setup(self):
        mq = MessageQueue()
        mq.create_topic("events")
        dlq = mq.setup_dlq()
        assert dlq is not None
        assert mq.broker.get_topic("_dlq") is not None

    def test_consume_nonexistent_consumer(self):
        mq = MessageQueue()
        with pytest.raises(LogError):
            mq.consume("nonexistent")

    def test_commit(self):
        mq = MessageQueue()
        mq.create_topic("events", partitions=1)
        for i in range(5):
            mq.publish("events", f"msg-{i}", partition=0)
        consumer = mq.subscribe("c1", ["events"], group_id="g1")
        consumer.auto_commit = False
        mq.consume("c1")
        mq.commit("c1")

    def test_publish_with_headers(self):
        mq = MessageQueue()
        mq.create_topic("events")
        t, p, o = mq.publish("events", "v", key="k", headers={"source": "test"})
        topic = mq.broker.get_topic("events")
        rec = topic.get_partition(p).get_record(o)
        assert rec.headers["source"] == "test"


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_multi_topic_consumer(self):
        """Consumer subscribing to multiple topics."""
        mq = MessageQueue()
        mq.create_topic("orders", partitions=2)
        mq.create_topic("payments", partitions=2)

        for i in range(5):
            mq.publish("orders", f"order-{i}")
            mq.publish("payments", f"payment-{i}")

        mq.subscribe("c1", ["orders", "payments"], group_id="g1")
        records = mq.consume("c1")
        assert len(records) == 10

    def test_consumer_group_rebalance_on_member_change(self):
        """Group rebalances when members join/leave."""
        mq = MessageQueue()
        mq.create_topic("events", partitions=4)

        # Start with 1 consumer
        mq.subscribe("c1", ["events"], group_id="g1")
        group = mq.broker.get_group("g1")
        assert len(group.get_assignments("c1")) == 4

        # Add second consumer and rebalance
        mq.subscribe("c2", ["events"], group_id="g1")
        assert len(group.get_assignments("c1")) == 2
        assert len(group.get_assignments("c2")) == 2

    def test_producer_consumer_roundtrip(self):
        """End-to-end: produce -> consume -> verify."""
        broker = Broker()
        broker.create_topic("test", num_partitions=1)

        producer = Producer(broker)
        for i in range(100):
            producer.send(ProducerRecord("test", f"value-{i}", key=f"key-{i}", partition=0))

        consumer = Consumer(broker, auto_offset_reset='earliest')
        consumer.assign([("test", 0)])
        result = consumer.poll(max_records=100)
        records = result[("test", 0)]
        assert len(records) == 100
        assert records[0].value == "value-0"
        assert records[99].value == "value-99"

    def test_compacted_topic_keeps_latest(self):
        """Compacted topic retains only latest value per key."""
        broker = Broker()
        broker.create_topic("state", num_partitions=1,
                            retention_policy=RetentionPolicy.COMPACT)
        producer = Producer(broker)
        # Write updates for same keys
        for _ in range(3):
            for k in ["user-1", "user-2", "user-3"]:
                producer.send(ProducerRecord("state", f"latest-{k}", key=k, partition=0))

        partition = broker.get_topic("state").get_partition(0)
        partition.compact()
        records = partition.read(0, 100)
        assert len(records) == 3
        values = {r.key: r.value for r in records}
        assert values["user-1"] == "latest-user-1"

    def test_multiple_consumer_groups_independent(self):
        """Different groups maintain independent offsets."""
        mq = MessageQueue()
        mq.create_topic("events", partitions=1)
        for i in range(10):
            mq.publish("events", f"msg-{i}", partition=0)

        mq.subscribe("c1", ["events"], group_id="fast-group")
        mq.subscribe("c2", ["events"], group_id="slow-group")

        # fast-group consumes all
        r1 = mq.consume("c1")
        assert len(r1) == 10

        # slow-group consumes only 5
        r2 = mq.consume("c2", max_records=5)
        assert len(r2) == 5

        # slow-group still has 5 remaining
        r3 = mq.consume("c2")
        assert len(r3) == 5

    def test_replication_and_leader_election(self):
        """Replication with leader failover."""
        broker = Broker()
        broker.create_topic("events", num_partitions=1)
        rep = broker.setup_replication("events", 0, ["n1", "n2", "n3"])

        producer = Producer(broker)
        for i in range(10):
            t, p, off = producer.send(ProducerRecord("events", f"v{i}", partition=0))
            rep.replicate(off)

        assert rep.leader_id == "n1"
        # Fail n1, elect new leader
        rep.remove_from_isr("n1")
        new_leader = rep.elect_leader()
        assert new_leader != "n1"
        assert rep.leader_id == new_leader

    def test_dlq_captures_failures(self):
        """Dead letter queue captures failed records."""
        broker = Broker()
        broker.create_topic("events")
        dlq = DeadLetterQueue(broker)

        producer = Producer(broker)
        t, p, o = producer.send(ProducerRecord("events", "bad-data", partition=0))
        record = broker.get_topic("events").get_partition(p).get_record(o)

        # Simulate processing failure
        dlq.send(record, "Deserialization error", "events", p)
        assert dlq.count == 1
        failed = dlq.read_failed()
        assert failed[0].value == "bad-data"

    def test_high_watermark_replication(self):
        """High watermark advances with replication."""
        broker = Broker()
        broker.create_topic("events", num_partitions=1)
        partition = broker.get_topic("events").get_partition(0)
        rep = broker.setup_replication("events", 0, ["n1", "n2", "n3"])

        producer = Producer(broker)
        for i in range(5):
            producer.send(ProducerRecord("events", f"v{i}", partition=0))

        # Before replication, watermark is at 5 (single-node auto-commit in Producer)
        # After explicit replication:
        rep.replicate(4)
        assert partition.high_watermark == 5

    def test_partition_key_ordering(self):
        """Records with same key go to same partition, in order."""
        mq = MessageQueue()
        mq.create_topic("orders", partitions=4)
        target_part = None
        for i in range(20):
            t, p, o = mq.publish("orders", f"order-{i}", key="user-42")
            if target_part is None:
                target_part = p
            assert p == target_part  # all to same partition

    def test_multi_partition_produce_consume(self):
        """Produce and consume across all partitions."""
        mq = MessageQueue()
        mq.create_topic("events", partitions=4)

        # Produce to all partitions explicitly
        for p in range(4):
            for i in range(5):
                mq.publish("events", f"p{p}-{i}", partition=p)

        # Consume all
        mq.subscribe("c1", ["events"], group_id="g1")
        records = mq.consume("c1", max_records=100)
        assert len(records) == 20

    def test_consumer_position_tracking(self):
        """Consumer remembers position across polls."""
        mq = MessageQueue()
        mq.create_topic("events", partitions=1)
        for i in range(20):
            mq.publish("events", f"msg-{i}", partition=0)

        consumer = mq.subscribe("c1", ["events"], group_id="g1")
        r1 = mq.consume("c1", max_records=5)
        assert len(r1) == 5
        r2 = mq.consume("c1", max_records=5)
        assert len(r2) == 5
        # Should be next 5, not duplicates
        assert r2[0].offset == 5

    def test_segment_rollover_during_writes(self):
        """Segments roll over correctly during heavy writes."""
        broker = Broker()
        broker.create_topic("events", num_partitions=1, segment_size=500)
        producer = Producer(broker)
        for i in range(100):
            producer.send(ProducerRecord("events", f"value-{i}", partition=0))

        partition = broker.get_topic("events").get_partition(0)
        assert len(partition._segments) > 1

        # All records readable across segments
        consumer = Consumer(broker, auto_offset_reset='earliest')
        consumer.assign([("events", 0)])
        result = consumer.poll(max_records=100)
        assert len(result[("events", 0)]) == 100

    def test_broker_stats_comprehensive(self):
        """Broker stats cover all aspects."""
        broker = Broker()
        broker.create_topic("t1", num_partitions=2)
        broker.create_topic("t2", num_partitions=3)
        producer = Producer(broker)
        for i in range(10):
            producer.send(ProducerRecord("t1", f"v{i}"))
            producer.send(ProducerRecord("t2", f"v{i}"))

        s = broker.stats()
        assert s['topics'] == 2
        assert s['total_records'] == 20
        assert s['total_bytes'] > 0
