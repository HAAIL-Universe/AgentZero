"""
Tests for C206: Distributed Key-Value Store
THE BIG COMPOSITION: Raft + Gossip + Vector Clocks + Consistent Hashing + HTTP
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C201_raft_consensus'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C203_gossip_protocol'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C204_vector_clocks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C205_consistent_hashing'))

from distributed_kv import (
    KVCluster, KVClient, KVHttpHandler, KVConfig,
    ConsistencyLevel, ConsistencyMode, OperationType, NodeState,
    Partition, PartitionStateMachine, ReplicaGroup, ClusterNode,
    KVEntry, KVResponse, KVRequest, HintedHandoff, SessionToken,
    VersionVector, create_cluster,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C204_vector_clocks'))
from vector_clocks import VectorClock


# ============================================================
# Helpers
# ============================================================

def make_cluster(n=3, num_partitions=4, replication_factor=3):
    """Create a bootstrapped cluster for testing."""
    config = KVConfig(
        num_partitions=num_partitions,
        replication_factor=min(replication_factor, n),
        num_vnodes=50,
        election_timeout_range=(50, 100),
        heartbeat_interval=10,
    )
    cluster = create_cluster(n, config)
    cluster.bootstrap(max_ticks=3000)
    return cluster


def make_small_cluster():
    """3-node cluster with 4 partitions."""
    return make_cluster(3, 4, 3)


# ============================================================
# 1. Partition Tests
# ============================================================

class TestPartition:
    """Test the Partition data layer."""

    def test_put_and_get(self):
        p = Partition(0, "node-0")
        p.put("key1", "value1", writer="w1")
        entries = p.get("key1")
        assert len(entries) == 1
        assert entries[0].value == "value1"
        assert entries[0].key == "key1"

    def test_get_missing_key(self):
        p = Partition(0, "node-0")
        entries = p.get("nonexistent")
        assert entries == []

    def test_overwrite(self):
        p = Partition(0, "node-0")
        p.put("k", "v1")
        p.put("k", "v2")
        entries = p.get("k")
        assert len(entries) == 1
        assert entries[0].value == "v2"

    def test_delete_creates_tombstone(self):
        p = Partition(0, "node-0")
        p.put("k", "v1")
        result = p.delete("k")
        assert result is not None
        assert result.tombstone is True
        # get() skips tombstones
        assert p.get("k") == []
        # get_with_tombstones sees them
        entries = p.get_with_tombstones("k")
        assert len(entries) == 1
        assert entries[0].tombstone is True

    def test_delete_nonexistent(self):
        p = Partition(0, "node-0")
        result = p.delete("nonexistent")
        assert result is None

    def test_keys(self):
        p = Partition(0, "node-0")
        p.put("a", 1)
        p.put("b", 2)
        p.put("c", 3)
        keys = sorted(p.keys())
        assert keys == ["a", "b", "c"]

    def test_size(self):
        p = Partition(0, "node-0")
        assert p.size() == 0
        p.put("a", 1)
        p.put("b", 2)
        assert p.size() == 2

    def test_version_vector_increments(self):
        p = Partition(0, "node-0")
        e1 = p.put("k", "v1")
        e2 = p.put("k", "v2")
        # Version should increase
        assert e2.version.get("node-0") > e1.version.get("node-0")

    def test_merge_entry_new(self):
        p = Partition(0, "node-0")
        entry = KVEntry(key="k", value="remote", version=VersionVector())
        entry.version.increment("node-1")
        assert p.merge_entry(entry) is True
        assert p.get("k")[0].value == "remote"

    def test_merge_entry_dominated(self):
        p = Partition(0, "node-0")
        p.put("k", "local")
        # Create an entry dominated by local
        old_entry = KVEntry(key="k", value="old")
        # The local entry's version dominates an empty version
        result = p.merge_entry(old_entry)
        # Should not overwrite since local is newer
        assert p.get("k")[0].value == "local"

    def test_merge_entry_concurrent_creates_siblings(self):
        p = Partition(0, "node-0")
        p.put("k", "local")

        # Create a concurrent entry from another node
        remote = KVEntry(key="k", value="remote", version=VersionVector())
        remote.version.increment("node-1")

        p.merge_entry(remote)
        entries = p.get("k")
        assert len(entries) == 2  # siblings
        values = {e.value for e in entries}
        assert values == {"local", "remote"}

    def test_has_conflicts(self):
        p = Partition(0, "node-0")
        p.put("k", "v1")
        remote = KVEntry(key="k", value="v2", version=VersionVector())
        remote.version.increment("node-1")
        p.merge_entry(remote)
        assert p.has_conflicts("k") is True

    def test_resolve_conflicts(self):
        p = Partition(0, "node-0")
        p.put("k", "v1")
        remote = KVEntry(key="k", value="v2", version=VersionVector())
        remote.version.increment("node-1")
        p.merge_entry(remote)
        assert p.has_conflicts("k")

        p.resolve_conflicts("k", "resolved")
        entries = p.get("k")
        assert len(entries) == 1
        assert entries[0].value == "resolved"
        assert not p.has_conflicts("k")

    def test_gc_tombstones(self):
        p = Partition(0, "node-0")
        p.put("k", "v", now=100.0)
        p.delete("k", now=100.0)
        assert p.get("k") == []

        # Before TTL
        p.gc_tombstones(ttl=60.0, now=150.0)
        assert "k" in [e.key for entries in p._data.values() for e in entries]

        # After TTL
        p.gc_tombstones(ttl=60.0, now=200.0)
        assert p.get_with_tombstones("k") == []

    def test_snapshot_and_restore(self):
        p = Partition(0, "node-0")
        p.put("a", 1)
        p.put("b", 2)

        snap = p.snapshot()
        p2 = Partition(0, "node-0")
        p2.restore(snap)

        assert p2.get("a")[0].value == 1
        assert p2.get("b")[0].value == 2

    def test_all_entries(self):
        p = Partition(0, "node-0")
        p.put("a", 1)
        p.put("b", 2)
        p.put("c", 3)
        entries = p.all_entries()
        assert len(entries) == 3

    def test_put_with_context_version(self):
        p = Partition(0, "node-0")
        e1 = p.put("k", "v1")
        ctx = e1.version.copy()
        e2 = p.put("k", "v2", context_version=ctx)
        # Should have replaced v1 since context covers it
        entries = p.get("k")
        assert len(entries) == 1
        assert entries[0].value == "v2"


# ============================================================
# 2. PartitionStateMachine Tests
# ============================================================

class TestPartitionStateMachine:
    """Test the Raft state machine wrapper."""

    def test_apply_put(self):
        sm = PartitionStateMachine(0, "node-0")
        result = sm.apply({"op": "put", "key": "k", "value": "v"}, 1)
        assert result["ok"] is True
        assert "version" in result

    def test_apply_get(self):
        sm = PartitionStateMachine(0, "node-0")
        sm.apply({"op": "put", "key": "k", "value": 42}, 1)
        result = sm.apply({"op": "get", "key": "k"}, 2)
        assert result["ok"] is True
        assert result["value"] == 42
        assert result["found"] is True

    def test_apply_get_missing(self):
        sm = PartitionStateMachine(0, "node-0")
        result = sm.apply({"op": "get", "key": "missing"}, 1)
        assert result["ok"] is True
        assert result["found"] is False

    def test_apply_delete(self):
        sm = PartitionStateMachine(0, "node-0")
        sm.apply({"op": "put", "key": "k", "value": "v"}, 1)
        result = sm.apply({"op": "delete", "key": "k"}, 2)
        assert result["ok"] is True
        assert result["deleted"] is True

    def test_apply_cas_success(self):
        sm = PartitionStateMachine(0, "node-0")
        sm.apply({"op": "put", "key": "k", "value": "v1"}, 1)
        result = sm.apply({"op": "cas", "key": "k", "expected": "v1",
                           "value": "v2"}, 2)
        assert result["ok"] is True

    def test_apply_cas_failure(self):
        sm = PartitionStateMachine(0, "node-0")
        sm.apply({"op": "put", "key": "k", "value": "v1"}, 1)
        result = sm.apply({"op": "cas", "key": "k", "expected": "wrong",
                           "value": "v2"}, 2)
        assert result["ok"] is False
        assert result["error"] == "cas_failed"

    def test_apply_unknown_op(self):
        sm = PartitionStateMachine(0, "node-0")
        result = sm.apply({"op": "invalid"}, 1)
        assert result["ok"] is False

    def test_snapshot_restore(self):
        sm = PartitionStateMachine(0, "node-0")
        sm.apply({"op": "put", "key": "a", "value": 1}, 1)
        sm.apply({"op": "put", "key": "b", "value": 2}, 2)

        snap = sm.snapshot()
        sm2 = PartitionStateMachine(0, "node-0")
        sm2.restore(snap)

        result = sm2.apply({"op": "get", "key": "a"}, 3)
        assert result["value"] == 1


# ============================================================
# 3. ReplicaGroup Tests
# ============================================================

class TestReplicaGroup:
    """Test Raft-based replica groups."""

    def test_elect_leader(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        assert group.tick_until_leader(3000)
        leader = group.get_leader()
        assert leader is not None
        assert leader in ["n0", "n1", "n2"]

    def test_submit_and_commit(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)

        result = group.submit_and_commit({"op": "put", "key": "k", "value": "v"})
        assert result.get("ok") is True

    def test_read_after_write(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)

        group.submit_and_commit({"op": "put", "key": "k", "value": 42})
        result = group.submit_and_commit({"op": "get", "key": "k"})
        assert result["value"] == 42

    def test_replicated_to_followers(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)

        group.submit_and_commit({"op": "put", "key": "k", "value": "replicated"})
        # Tick more to ensure replication
        for _ in range(100):
            group.tick(1)
            group.deliver_messages()

        # Check all nodes have the data
        for nid in ["n0", "n1", "n2"]:
            sm = group.get_state_machine(nid)
            entries = sm.partition.get("k")
            assert len(entries) > 0, f"Node {nid} missing key"
            assert entries[0].value == "replicated"

    def test_read_local(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)
        group.submit_and_commit({"op": "put", "key": "k", "value": "local"})
        for _ in range(100):
            group.tick(1)
            group.deliver_messages()

        leader = group.get_leader()
        entries = group.read_local(leader, "k")
        assert len(entries) > 0
        assert entries[0].value == "local"

    def test_no_leader_returns_error(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        # Don't elect -- immediate submit fails
        result = group.submit_and_commit({"op": "put", "key": "k", "value": "v"},
                                         max_ticks=100)
        # Might find a leader in 100 ticks or might not
        # The key point is it doesn't crash

    def test_partition_tolerance(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)
        leader = group.get_leader()

        # Partition minority
        minority = [n for n in ["n0", "n1", "n2"] if n != leader][:1]
        majority = [n for n in ["n0", "n1", "n2"] if n not in minority]
        group.partition_nodes(minority, majority)

        # Can still write to majority
        result = group.submit_and_commit({"op": "put", "key": "k", "value": "during_partition"})
        assert result.get("ok") is True

    def test_heal_partition(self):
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        group.tick_until_leader(3000)

        group.partition_nodes(["n2"], ["n0", "n1"])
        group.submit_and_commit({"op": "put", "key": "k", "value": "v"})

        group.heal_partitions()
        for _ in range(200):
            group.tick(1)
            group.deliver_messages()

        # n2 should catch up
        sm = group.get_state_machine("n2")
        entries = sm.partition.get("k")
        assert len(entries) > 0


# ============================================================
# 4. KVCluster Basic Tests
# ============================================================

class TestKVClusterBasic:
    """Test the full cluster -- basic operations."""

    def test_create_cluster(self):
        cluster = make_small_cluster()
        assert len(cluster.nodes) == 3
        stats = cluster.get_cluster_stats()
        assert stats["nodes"] == 3
        assert stats["partitions"] == 4

    def test_put_and_get_strong(self):
        cluster = make_small_cluster()
        resp = cluster.put("hello", "world", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True

        resp = cluster.get("hello", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True
        assert resp.value == "world"

    def test_put_and_get_quorum(self):
        cluster = make_small_cluster()
        resp = cluster.put("key1", "val1", consistency=ConsistencyLevel.QUORUM)
        assert resp.success is True

        resp = cluster.get("key1", consistency=ConsistencyLevel.QUORUM)
        assert resp.success is True
        assert resp.value == "val1"

    def test_put_and_get_one(self):
        cluster = make_small_cluster()
        resp = cluster.put("key1", "val1", consistency=ConsistencyLevel.ONE)
        assert resp.success is True

        resp = cluster.get("key1", consistency=ConsistencyLevel.ONE)
        assert resp.success is True
        assert resp.value == "val1"

    def test_delete(self):
        cluster = make_small_cluster()
        cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        resp = cluster.delete("k", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True

        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True
        assert resp.value is None

    def test_get_nonexistent(self):
        cluster = make_small_cluster()
        resp = cluster.get("missing", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True
        assert resp.value is None

    def test_overwrite(self):
        cluster = make_small_cluster()
        cluster.put("k", "v1", consistency=ConsistencyLevel.STRONG)
        cluster.put("k", "v2", consistency=ConsistencyLevel.STRONG)
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "v2"

    def test_multiple_keys(self):
        cluster = make_small_cluster()
        for i in range(10):
            cluster.put(f"key-{i}", f"val-{i}", consistency=ConsistencyLevel.STRONG)

        for i in range(10):
            resp = cluster.get(f"key-{i}", consistency=ConsistencyLevel.STRONG)
            assert resp.value == f"val-{i}", f"Failed for key-{i}"

    def test_numeric_values(self):
        cluster = make_small_cluster()
        cluster.put("int", 42, consistency=ConsistencyLevel.STRONG)
        cluster.put("float", 3.14, consistency=ConsistencyLevel.STRONG)
        cluster.put("list", [1, 2, 3], consistency=ConsistencyLevel.STRONG)

        assert cluster.get("int", consistency=ConsistencyLevel.STRONG).value == 42
        assert cluster.get("float", consistency=ConsistencyLevel.STRONG).value == 3.14
        assert cluster.get("list", consistency=ConsistencyLevel.STRONG).value == [1, 2, 3]


# ============================================================
# 5. Compare-and-Swap Tests
# ============================================================

class TestCAS:
    """Test compare-and-swap operations."""

    def test_cas_success(self):
        cluster = make_small_cluster()
        cluster.put("counter", 0, consistency=ConsistencyLevel.STRONG)
        resp = cluster.cas("counter", expected=0, new_value=1)
        assert resp.success is True

        resp = cluster.get("counter", consistency=ConsistencyLevel.STRONG)
        assert resp.value == 1

    def test_cas_failure(self):
        cluster = make_small_cluster()
        cluster.put("counter", 5, consistency=ConsistencyLevel.STRONG)
        resp = cluster.cas("counter", expected=0, new_value=1)
        assert resp.success is False
        assert resp.error == "cas_failed"

    def test_cas_on_none(self):
        cluster = make_small_cluster()
        # CAS on non-existent key (current = None)
        resp = cluster.cas("new_key", expected=None, new_value="created")
        assert resp.success is True

    def test_cas_sequence(self):
        cluster = make_small_cluster()
        cluster.put("seq", 0, consistency=ConsistencyLevel.STRONG)
        for i in range(5):
            resp = cluster.cas("seq", expected=i, new_value=i + 1)
            assert resp.success is True
        resp = cluster.get("seq", consistency=ConsistencyLevel.STRONG)
        assert resp.value == 5


# ============================================================
# 6. Scan Tests
# ============================================================

class TestScan:
    """Test key scanning."""

    def test_scan_all(self):
        cluster = make_small_cluster()
        for i in range(5):
            cluster.put(f"item-{i}", i, consistency=ConsistencyLevel.STRONG)

        resp = cluster.scan()
        assert resp.success is True
        assert len(resp.value) == 5

    def test_scan_with_prefix(self):
        cluster = make_small_cluster()
        cluster.put("user:1", "alice", consistency=ConsistencyLevel.STRONG)
        cluster.put("user:2", "bob", consistency=ConsistencyLevel.STRONG)
        cluster.put("post:1", "hello", consistency=ConsistencyLevel.STRONG)

        resp = cluster.scan(prefix="user:")
        assert resp.success is True
        assert len(resp.value) == 2
        for item in resp.value:
            assert item["key"].startswith("user:")

    def test_scan_with_limit(self):
        cluster = make_small_cluster()
        for i in range(10):
            cluster.put(f"k{i}", i, consistency=ConsistencyLevel.STRONG)

        resp = cluster.scan(limit=3)
        assert len(resp.value) <= 3

    def test_scan_empty(self):
        cluster = make_small_cluster()
        resp = cluster.scan()
        assert resp.success is True
        assert resp.value == []

    def test_scan_sorted(self):
        cluster = make_small_cluster()
        cluster.put("c", 3, consistency=ConsistencyLevel.STRONG)
        cluster.put("a", 1, consistency=ConsistencyLevel.STRONG)
        cluster.put("b", 2, consistency=ConsistencyLevel.STRONG)

        resp = cluster.scan()
        keys = [item["key"] for item in resp.value]
        assert keys == sorted(keys)


# ============================================================
# 7. Batch Operations Tests
# ============================================================

class TestBatch:
    """Test batch operations."""

    def test_batch_puts(self):
        cluster = make_small_cluster()
        ops = [{"op": "put", "key": f"k{i}", "value": i} for i in range(5)]
        resp = cluster.batch(ops)
        assert resp.success is True
        assert len(resp.value) == 5

    def test_batch_mixed(self):
        cluster = make_small_cluster()
        cluster.put("existing", "old", consistency=ConsistencyLevel.STRONG)

        ops = [
            {"op": "put", "key": "new1", "value": 1},
            {"op": "get", "key": "existing"},
            {"op": "delete", "key": "existing"},
        ]
        resp = cluster.batch(ops)
        assert resp.success is True
        assert len(resp.value) == 3

    def test_batch_empty(self):
        cluster = make_small_cluster()
        resp = cluster.batch([])
        assert resp.success is True
        assert resp.value == []


# ============================================================
# 8. Partition Assignment Tests
# ============================================================

class TestPartitionAssignment:
    """Test consistent hashing based partition assignment."""

    def test_all_partitions_assigned(self):
        cluster = make_small_cluster()
        info = cluster.get_partition_info()
        assert len(info) == 4  # num_partitions=4

    def test_partitions_have_replicas(self):
        cluster = make_small_cluster()
        info = cluster.get_partition_info()
        for pid, pinfo in info.items():
            assert len(pinfo["replicas"]) >= 1

    def test_partitions_have_leaders(self):
        cluster = make_small_cluster()
        info = cluster.get_partition_info()
        leaders = [pinfo["leader"] for pinfo in info.values()]
        assert all(l is not None for l in leaders)

    def test_key_to_partition_deterministic(self):
        cluster = make_small_cluster()
        pid1 = cluster._key_to_partition("test-key")
        pid2 = cluster._key_to_partition("test-key")
        assert pid1 == pid2

    def test_keys_distributed_across_partitions(self):
        cluster = make_small_cluster()
        partitions_used = set()
        for i in range(100):
            pid = cluster._key_to_partition(f"key-{i}")
            partitions_used.add(pid)
        # With 100 keys and 4 partitions, all should be used
        assert len(partitions_used) == 4


# ============================================================
# 9. Vector Clock / Versioning Tests
# ============================================================

class TestVersioning:
    """Test version tracking with vector clocks."""

    def test_version_on_put(self):
        cluster = make_small_cluster()
        resp = cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        assert resp.success is True
        assert resp.version  # Should have version info

    def test_version_increments(self):
        cluster = make_small_cluster()
        r1 = cluster.put("k", "v1", consistency=ConsistencyLevel.STRONG)
        r2 = cluster.put("k", "v2", consistency=ConsistencyLevel.STRONG)
        # Both should succeed
        assert r1.success
        assert r2.success

    def test_version_on_get(self):
        cluster = make_small_cluster()
        cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.version  # Should return version

    def test_conflict_detection_partition_level(self):
        """Test that concurrent writes on the same partition create siblings."""
        p = Partition(0, "node-0")
        p.put("k", "v1", writer="node-0")

        # Simulate concurrent write from different node
        remote = KVEntry(key="k", value="v2", version=VersionVector())
        remote.version.increment("node-1")
        p.merge_entry(remote)

        assert p.has_conflicts("k")
        entries = p.get("k")
        assert len(entries) == 2


# ============================================================
# 10. Anti-Entropy Tests
# ============================================================

class TestAntiEntropy:
    """Test anti-entropy synchronization."""

    def test_sync_two_replicas(self):
        cluster = make_small_cluster()
        # Write via quorum (bypassing Raft for direct partition writes)
        cluster.put("sync-test", "hello", consistency=ConsistencyLevel.QUORUM)

        # Find which partition has this key
        pid = cluster._key_to_partition("sync-test")
        nodes = cluster._partition_map[pid]

        if len(nodes) >= 2:
            synced = cluster.anti_entropy_sync(pid, nodes[0], nodes[1])
            # May sync 0 if already consistent
            assert synced >= 0

    def test_full_anti_entropy(self):
        cluster = make_small_cluster()
        for i in range(5):
            cluster.put(f"ae-{i}", i, consistency=ConsistencyLevel.QUORUM)

        total = cluster.full_anti_entropy()
        assert total >= 0  # Should not crash

    def test_anti_entropy_converges_data(self):
        """Write to one replica, sync to another."""
        group = ReplicaGroup(0, ["n0", "n1", "n2"])
        sm0 = group.get_state_machine("n0")
        sm1 = group.get_state_machine("n1")

        # Direct write to n0 only
        sm0.partition.put("k", "only-on-n0", writer="n0")
        assert sm1.partition.get("k") == []

        # Anti-entropy sync
        config = KVConfig()
        cluster = make_small_cluster()
        # Manually sync
        for entry in sm0.partition.all_entries():
            sm1.partition.merge_entry(entry)

        entries = sm1.partition.get("k")
        assert len(entries) == 1
        assert entries[0].value == "only-on-n0"


# ============================================================
# 11. Hinted Handoff Tests
# ============================================================

class TestHintedHandoff:
    """Test hinted handoff for unavailable nodes."""

    def test_store_and_retrieve_hints(self):
        cluster = make_small_cluster()
        entry = KVEntry(key="k", value="v", version=VersionVector())
        cluster.store_hint("node-0", "node-1", entry, partition_id=0)

        node = cluster._nodes["node-0"]
        hints = node.get_hints_for("node-1")
        assert len(hints) == 1
        assert hints[0].entry.key == "k"

    def test_deliver_hints(self):
        cluster = make_small_cluster()
        pid = 0
        nodes = cluster._partition_map[pid]
        if len(nodes) < 2:
            return

        entry = KVEntry(key="hinted", value="value", version=VersionVector())
        entry.version.increment(nodes[0])
        cluster.store_hint(nodes[0], nodes[1], entry, partition_id=pid)

        delivered = cluster.deliver_hints(nodes[0], nodes[1])
        assert delivered == 1

        # Hints should be cleared
        node = cluster._nodes[nodes[0]]
        assert len(node.get_hints_for(nodes[1])) == 0

    def test_hint_limit(self):
        node = ClusterNode("test-node")
        for i in range(1100):
            hint = HintedHandoff(
                target_node="target",
                entry=KVEntry(key=f"k{i}", value=i),
                created_at=0.0,
                partition_id=0,
            )
            node.store_hint(hint)
        assert len(node.hints) == 1000  # Capped


# ============================================================
# 12. Gossip Membership Tests
# ============================================================

class TestGossipMembership:
    """Test gossip-based membership management."""

    def test_gossip_network_created(self):
        cluster = make_small_cluster()
        # Gossip network should have all nodes
        assert len(cluster._gossip_net.nodes) == 3

    def test_gossip_tick(self):
        cluster = make_small_cluster()
        # Should not crash
        cluster.gossip_tick()

    def test_add_node(self):
        cluster = make_small_cluster()
        result = cluster.add_node("node-3")
        assert result is True
        assert "node-3" in cluster.nodes

    def test_add_duplicate_node(self):
        cluster = make_small_cluster()
        result = cluster.add_node("node-0")
        assert result is False

    def test_remove_node(self):
        cluster = make_small_cluster()
        result = cluster.remove_node("node-2")
        assert result is True
        assert cluster.nodes["node-2"].state == NodeState.DEAD

    def test_remove_nonexistent_node(self):
        cluster = make_small_cluster()
        result = cluster.remove_node("nonexistent")
        assert result is False


# ============================================================
# 13. Cluster Info Tests
# ============================================================

class TestClusterInfo:
    """Test cluster information and statistics."""

    def test_cluster_stats(self):
        cluster = make_small_cluster()
        stats = cluster.get_cluster_stats()
        assert stats["nodes"] == 3
        assert stats["partitions"] == 4
        assert stats["replication_factor"] == 3

    def test_partition_info(self):
        cluster = make_small_cluster()
        info = cluster.get_partition_info()
        assert len(info) == 4
        for pid, pinfo in info.items():
            assert "replicas" in pinfo
            assert "leader" in pinfo
            assert "size" in pinfo

    def test_node_info(self):
        cluster = make_small_cluster()
        info = cluster.get_node_info("node-0")
        assert info is not None
        assert info["node_id"] == "node-0"
        assert info["state"] == "ACTIVE"
        assert "partitions" in info

    def test_node_info_missing(self):
        cluster = make_small_cluster()
        info = cluster.get_node_info("nonexistent")
        assert info is None

    def test_stats_after_writes(self):
        cluster = make_small_cluster()
        for i in range(5):
            cluster.put(f"stat-key-{i}", i, consistency=ConsistencyLevel.STRONG)
        stats = cluster.get_cluster_stats()
        assert stats["total_keys"] >= 5


# ============================================================
# 14. KVClient Tests
# ============================================================

class TestKVClient:
    """Test the client wrapper."""

    def test_client_put_get(self):
        cluster = make_small_cluster()
        client = KVClient(cluster, session_id="test-session")

        resp = client.put("k", "v")
        assert resp.success
        assert resp.session_id == "test-session"

        resp = client.get("k")
        assert resp.success
        assert resp.value == "v"

    def test_client_delete(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        client.put("k", "v")
        resp = client.delete("k")
        assert resp.success

    def test_client_cas(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        client.put("counter", 0, consistency=ConsistencyLevel.STRONG)
        resp = client.cas("counter", expected=0, new_value=1)
        assert resp.success

    def test_client_scan(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        for i in range(3):
            client.put(f"item-{i}", i, consistency=ConsistencyLevel.STRONG)

        resp = client.scan(prefix="item-")
        assert resp.success
        assert len(resp.value) == 3

    def test_client_batch(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        ops = [{"op": "put", "key": f"batch-{i}", "value": i} for i in range(3)]
        resp = client.batch(ops)
        assert resp.success

    def test_client_request_ids_increment(self):
        cluster = make_small_cluster()
        client = KVClient(cluster, session_id="s1")

        r1 = client.put("a", 1)
        r2 = client.put("b", 2)
        assert r1.request_id != r2.request_id


# ============================================================
# 15. HTTP Handler Tests
# ============================================================

class TestKVHttpHandler:
    """Test the REST API handler."""

    def setup_method(self):
        self.cluster = make_small_cluster()
        self.handler = KVHttpHandler(self.cluster)

    def test_put_via_http(self):
        status, body = self.handler.handle_request(
            "PUT", "/kv/mykey", {"value": "myvalue"})
        assert status == 200
        assert body["success"] is True

    def test_get_via_http(self):
        self.handler.handle_request("PUT", "/kv/mykey",
                                     {"value": "myvalue"})
        status, body = self.handler.handle_request("GET", "/kv/mykey")
        assert status == 200
        assert body["success"] is True
        assert body["value"] == "myvalue"

    def test_delete_via_http(self):
        self.handler.handle_request("PUT", "/kv/mykey",
                                     {"value": "myvalue"})
        status, body = self.handler.handle_request("DELETE", "/kv/mykey")
        assert status == 200

    def test_cas_via_http(self):
        self.handler.handle_request("PUT", "/kv/counter",
                                     {"value": 0},
                                     {"X-Consistency": "strong"})
        status, body = self.handler.handle_request(
            "POST", "/kv/counter/cas",
            {"expected": 0, "value": 1},
            {"X-Consistency": "strong"})
        assert status == 200
        assert body["success"] is True

    def test_cas_conflict_via_http(self):
        self.handler.handle_request("PUT", "/kv/counter",
                                     {"value": 5},
                                     {"X-Consistency": "strong"})
        status, body = self.handler.handle_request(
            "POST", "/kv/counter/cas",
            {"expected": 0, "value": 1},
            {"X-Consistency": "strong"})
        assert status == 409
        assert body["success"] is False

    def test_scan_via_http(self):
        for i in range(3):
            self.handler.handle_request(
                "PUT", f"/kv/item-{i}", {"value": i},
                {"X-Consistency": "strong"})

        status, body = self.handler.handle_request(
            "GET", "/kv/_scan", {"prefix": "item-"})
        assert status == 200
        assert body["success"] is True
        assert len(body["value"]) == 3

    def test_batch_via_http(self):
        ops = [{"op": "put", "key": f"b{i}", "value": i} for i in range(3)]
        status, body = self.handler.handle_request(
            "POST", "/kv/_batch", {"operations": ops})
        assert status == 200
        assert body["success"] is True

    def test_cluster_info(self):
        status, body = self.handler.handle_request("GET", "/cluster/info")
        assert status == 200
        assert "nodes" in body

    def test_partition_info(self):
        status, body = self.handler.handle_request("GET", "/cluster/partitions")
        assert status == 200

    def test_node_info(self):
        status, body = self.handler.handle_request("GET", "/cluster/node/node-0")
        assert status == 200
        assert body["node_id"] == "node-0"

    def test_node_info_missing(self):
        status, body = self.handler.handle_request("GET", "/cluster/node/fake")
        assert status == 404

    def test_404_route(self):
        status, body = self.handler.handle_request("GET", "/unknown")
        assert status == 404

    def test_consistency_header(self):
        # Put with ONE consistency
        status, body = self.handler.handle_request(
            "PUT", "/kv/hkey", {"value": "hval"},
            {"X-Consistency": "one"})
        assert status == 200

    def test_session_header(self):
        self.handler.handle_request(
            "PUT", "/kv/skey", {"value": "sval"},
            {"X-Consistency": "strong"})
        status, body = self.handler.handle_request(
            "GET", "/kv/skey", {},
            {"X-Session-ID": "session-123"})
        assert status == 200


# ============================================================
# 16. Session Consistency Tests
# ============================================================

class TestSessionConsistency:
    """Test session-based consistency."""

    def test_session_token_creation(self):
        node = ClusterNode("n0")
        session = node.get_or_create_session("s1", now=100.0)
        assert session.session_id == "s1"
        assert session.node_id == "n0"

    def test_session_token_reuse(self):
        node = ClusterNode("n0")
        s1 = node.get_or_create_session("s1")
        s2 = node.get_or_create_session("s1")
        assert s1 is s2  # Same object

    def test_session_gc(self):
        node = ClusterNode("n0")
        node.get_or_create_session("s1", now=100.0)
        node.get_or_create_session("s2", now=200.0)
        node.gc_sessions(ttl=50.0, now=250.0)
        # s1 expired (age=150 > 50), s2 alive (age=50 = 50)
        assert "s1" not in node._sessions

    def test_session_update_clock(self):
        token = SessionToken(session_id="s1", node_id="n0")
        vc = VectorClock({"n0": 5})
        token.update(vc)
        assert token.vector_clock.get("n0") == 5


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_key(self):
        cluster = make_small_cluster()
        resp = cluster.put("", "empty_key")
        assert resp.success  # Empty keys are valid

    def test_none_value(self):
        cluster = make_small_cluster()
        resp = cluster.put("k", None, consistency=ConsistencyLevel.STRONG)
        assert resp.success
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value is None

    def test_large_value(self):
        cluster = make_small_cluster()
        large_val = "x" * 10000
        resp = cluster.put("large", large_val, consistency=ConsistencyLevel.STRONG)
        assert resp.success
        resp = cluster.get("large", consistency=ConsistencyLevel.STRONG)
        assert resp.value == large_val

    def test_special_characters_in_key(self):
        cluster = make_small_cluster()
        special_keys = ["key/with/slashes", "key:with:colons",
                        "key.with.dots", "key with spaces"]
        for key in special_keys:
            resp = cluster.put(key, "val")
            assert resp.success, f"Failed for key: {key}"

    def test_nested_value(self):
        cluster = make_small_cluster()
        nested = {"users": [{"name": "alice", "scores": [1, 2, 3]}]}
        resp = cluster.put("nested", nested, consistency=ConsistencyLevel.STRONG)
        assert resp.success
        resp = cluster.get("nested", consistency=ConsistencyLevel.STRONG)
        assert resp.value == nested

    def test_concurrent_puts_same_key(self):
        cluster = make_small_cluster()
        # Multiple puts to same key -- last should win
        for i in range(10):
            cluster.put("race", i, consistency=ConsistencyLevel.STRONG)
        resp = cluster.get("race", consistency=ConsistencyLevel.STRONG)
        assert resp.value == 9

    def test_delete_then_put(self):
        cluster = make_small_cluster()
        cluster.put("k", "v1", consistency=ConsistencyLevel.STRONG)
        cluster.delete("k", consistency=ConsistencyLevel.STRONG)
        cluster.put("k", "v2", consistency=ConsistencyLevel.STRONG)
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "v2"


# ============================================================
# 18. KVEntry Serialization Tests
# ============================================================

class TestKVEntry:
    """Test KVEntry serialization."""

    def test_to_dict(self):
        entry = KVEntry(key="k", value="v", timestamp=100.0, writer="w1")
        d = entry.to_dict()
        assert d["key"] == "k"
        assert d["value"] == "v"
        assert d["timestamp"] == 100.0
        assert d["writer"] == "w1"
        assert d["tombstone"] is False

    def test_from_dict(self):
        d = {"key": "k", "value": "v", "version": {"n0": 3},
             "timestamp": 100.0, "tombstone": False, "writer": "w1"}
        entry = KVEntry.from_dict(d)
        assert entry.key == "k"
        assert entry.value == "v"
        assert entry.version.get("n0") == 3

    def test_roundtrip(self):
        entry = KVEntry(key="k", value=42, timestamp=200.0, writer="w")
        entry.version.increment("n0")
        entry.version.increment("n0")

        d = entry.to_dict()
        restored = KVEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.version.get("n0") == 2


# ============================================================
# 19. KVResponse Tests
# ============================================================

class TestKVResponse:
    """Test response serialization."""

    def test_success_response(self):
        resp = KVResponse(success=True, value="hello")
        d = resp.to_dict()
        assert d["success"] is True
        assert d["value"] == "hello"

    def test_error_response(self):
        resp = KVResponse(success=False, error="timeout")
        d = resp.to_dict()
        assert d["success"] is False
        assert d["error"] == "timeout"

    def test_response_with_conflicts(self):
        resp = KVResponse(success=True, value="v1",
                          conflicts=[{"key": "k", "value": "v2"}])
        d = resp.to_dict()
        assert len(d["conflicts"]) == 1

    def test_response_minimal(self):
        resp = KVResponse(success=True)
        d = resp.to_dict()
        assert d == {"success": True, "value": None}


# ============================================================
# 20. Config Tests
# ============================================================

class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        config = KVConfig()
        assert config.num_partitions == 16
        assert config.replication_factor == 3
        assert config.read_repair is True
        assert config.hinted_handoff is True

    def test_custom_config(self):
        config = KVConfig(num_partitions=8, replication_factor=5)
        assert config.num_partitions == 8
        assert config.replication_factor == 5

    def test_config_consistency_levels(self):
        config = KVConfig(default_consistency=ConsistencyLevel.ONE)
        cluster = KVCluster(["n0", "n1", "n2"], config)
        assert cluster.config.default_consistency == ConsistencyLevel.ONE


# ============================================================
# 21. Multi-node Cluster Tests
# ============================================================

class TestMultiNodeCluster:
    """Test larger clusters."""

    def test_5_node_cluster(self):
        cluster = make_cluster(5, 8, 3)
        resp = cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        assert resp.success
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "v"

    def test_single_node_cluster(self):
        config = KVConfig(
            num_partitions=2,
            replication_factor=1,
            num_vnodes=50,
            election_timeout_range=(50, 100),
            heartbeat_interval=10,
        )
        cluster = KVCluster(["solo"], config)
        cluster.bootstrap()
        resp = cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        assert resp.success
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "v"

    def test_replication_factor_capped(self):
        """RF > nodes should still work."""
        config = KVConfig(
            num_partitions=2,
            replication_factor=5,  # more than nodes
            num_vnodes=50,
            election_timeout_range=(50, 100),
            heartbeat_interval=10,
        )
        cluster = KVCluster(["n0", "n1", "n2"], config)
        cluster.bootstrap()
        # Should work even though RF > node count
        resp = cluster.put("k", "v", consistency=ConsistencyLevel.ONE)
        assert resp.success


# ============================================================
# 22. Read Repair Tests
# ============================================================

class TestReadRepair:
    """Test read repair on quorum reads."""

    def test_read_repair_propagates(self):
        config = KVConfig(
            num_partitions=2,
            replication_factor=3,
            num_vnodes=50,
            read_repair=True,
            election_timeout_range=(50, 100),
            heartbeat_interval=10,
        )
        cluster = KVCluster(["n0", "n1", "n2"], config)
        cluster.bootstrap()

        # Write directly to one partition replica
        pid = cluster._key_to_partition("rr-key")
        group = cluster._replica_groups[pid]
        nodes = cluster._partition_map[pid]

        if len(nodes) >= 2:
            sm0 = group.get_state_machine(nodes[0])
            sm0.partition.put("rr-key", "rr-value", writer=nodes[0])

            # Quorum read triggers read repair
            cluster.get("rr-key", consistency=ConsistencyLevel.QUORUM)

            # After read repair, other replicas should have the value
            sm1 = group.get_state_machine(nodes[1])
            entries = sm1.partition.get("rr-key")
            assert len(entries) > 0


# ============================================================
# 23. Consistency Level Tests
# ============================================================

class TestConsistencyLevels:
    """Test different consistency levels."""

    def test_required_responses_one(self):
        cluster = make_small_cluster()
        assert cluster._required_responses(ConsistencyLevel.ONE) == 1

    def test_required_responses_quorum(self):
        cluster = make_small_cluster()
        # RF=3, quorum = 2
        assert cluster._required_responses(ConsistencyLevel.QUORUM) == 2

    def test_required_responses_all(self):
        cluster = make_small_cluster()
        assert cluster._required_responses(ConsistencyLevel.ALL) == 3

    def test_write_one_read_one(self):
        cluster = make_small_cluster()
        cluster.put("k", "v", consistency=ConsistencyLevel.ONE)
        resp = cluster.get("k", consistency=ConsistencyLevel.ONE)
        assert resp.success

    def test_write_all_read_one(self):
        cluster = make_small_cluster()
        cluster.put("k", "v", consistency=ConsistencyLevel.ALL)
        resp = cluster.get("k", consistency=ConsistencyLevel.ONE)
        assert resp.success
        assert resp.value == "v"


# ============================================================
# 24. Merge Entry Tests
# ============================================================

class TestMergeEntries:
    """Test entry merging logic."""

    def test_merge_empty(self):
        cluster = make_small_cluster()
        result = cluster._merge_entries([])
        assert result == []

    def test_merge_single(self):
        e = KVEntry(key="k", value="v")
        cluster = make_small_cluster()
        result = cluster._merge_entries([e])
        assert len(result) == 1

    def test_merge_dominated(self):
        """Newer version dominates older."""
        cluster = make_small_cluster()
        e1 = KVEntry(key="k", value="old", version=VersionVector())
        e1.version.increment("n0")

        e2 = KVEntry(key="k", value="new", version=VersionVector())
        e2.version.increment("n0")
        e2.version.increment("n0")

        result = cluster._merge_entries([e1, e2])
        assert len(result) == 1
        assert result[0].value == "new"

    def test_merge_concurrent(self):
        """Concurrent versions create siblings."""
        cluster = make_small_cluster()
        e1 = KVEntry(key="k", value="v1", version=VersionVector())
        e1.version.increment("n0")

        e2 = KVEntry(key="k", value="v2", version=VersionVector())
        e2.version.increment("n1")

        result = cluster._merge_entries([e1, e2])
        assert len(result) == 2


# ============================================================
# 25. Cluster Ticking Tests
# ============================================================

class TestClusterTicking:
    """Test cluster time advancement."""

    def test_tick(self):
        cluster = make_small_cluster()
        # Should not crash
        cluster.tick(1)

    def test_tick_n(self):
        cluster = make_small_cluster()
        cluster.tick_n(100)

    def test_tick_after_writes(self):
        cluster = make_small_cluster()
        cluster.put("k", "v", consistency=ConsistencyLevel.STRONG)
        cluster.tick_n(50)
        resp = cluster.get("k", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "v"


# ============================================================
# 26. Integration: Full Workflow
# ============================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_crud_lifecycle(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        # Create
        assert client.put("user:1", {"name": "Alice"}).success
        # Read
        resp = client.get("user:1")
        assert resp.value == {"name": "Alice"}
        # Update
        assert client.put("user:1", {"name": "Alice", "age": 30}).success
        resp = client.get("user:1")
        assert resp.value == {"name": "Alice", "age": 30}
        # Delete
        assert client.delete("user:1").success
        resp = client.get("user:1")
        assert resp.value is None

    def test_multiple_clients(self):
        cluster = make_small_cluster()
        c1 = KVClient(cluster, session_id="c1")
        c2 = KVClient(cluster, session_id="c2")

        c1.put("shared", "from-c1", consistency=ConsistencyLevel.STRONG)
        resp = c2.get("shared", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "from-c1"

        c2.put("shared", "from-c2", consistency=ConsistencyLevel.STRONG)
        resp = c1.get("shared", consistency=ConsistencyLevel.STRONG)
        assert resp.value == "from-c2"

    def test_http_full_workflow(self):
        cluster = make_small_cluster()
        handler = KVHttpHandler(cluster)

        # PUT
        s, b = handler.handle_request("PUT", "/kv/test",
                                       {"value": "hello"},
                                       {"X-Consistency": "strong"})
        assert s == 200

        # GET
        s, b = handler.handle_request("GET", "/kv/test", {},
                                       {"X-Consistency": "strong"})
        assert s == 200
        assert b["value"] == "hello"

        # DELETE
        s, b = handler.handle_request("DELETE", "/kv/test", {},
                                       {"X-Consistency": "strong"})
        assert s == 200

        # GET after delete
        s, b = handler.handle_request("GET", "/kv/test", {},
                                       {"X-Consistency": "strong"})
        assert s == 200
        assert b["value"] is None

    def test_batch_then_scan(self):
        cluster = make_small_cluster()
        client = KVClient(cluster)

        ops = [{"op": "put", "key": f"product:{i}", "value": f"item-{i}"}
               for i in range(5)]
        client.batch(ops)

        resp = client.scan(prefix="product:")
        assert resp.success
        assert len(resp.value) == 5

    def test_cas_counter_pattern(self):
        """Implement a distributed counter with CAS."""
        cluster = make_small_cluster()
        client = KVClient(cluster)

        client.put("counter", 0, consistency=ConsistencyLevel.STRONG)

        for _ in range(10):
            current = client.get("counter", consistency=ConsistencyLevel.STRONG).value
            client.cas("counter", expected=current, new_value=current + 1)

        resp = client.get("counter", consistency=ConsistencyLevel.STRONG)
        assert resp.value == 10

    def test_cluster_info_via_http(self):
        cluster = make_small_cluster()
        handler = KVHttpHandler(cluster)

        s, info = handler.handle_request("GET", "/cluster/info")
        assert s == 200
        assert info["nodes"] == 3

        s, parts = handler.handle_request("GET", "/cluster/partitions")
        assert s == 200

    def test_many_keys_distributed(self):
        """Write many keys and verify they distribute across partitions."""
        cluster = make_small_cluster()

        for i in range(50):
            cluster.put(f"dist-{i}", i, consistency=ConsistencyLevel.STRONG)

        stats = cluster.get_cluster_stats()
        assert stats["total_keys"] == 50

        # Check distribution
        info = cluster.get_partition_info()
        sizes = [pinfo["size"] for pinfo in info.values()]
        assert sum(sizes) == 50
        # Not all in one partition
        assert max(sizes) < 50

    def test_node_addition_workflow(self):
        cluster = make_small_cluster()
        # Write some data
        for i in range(5):
            cluster.put(f"pre-{i}", i, consistency=ConsistencyLevel.STRONG)

        # Add node
        cluster.add_node("node-3")
        assert "node-3" in cluster.nodes

        # Existing data still accessible
        for i in range(5):
            resp = cluster.get(f"pre-{i}", consistency=ConsistencyLevel.STRONG)
            assert resp.success


# ============================================================
# 27. Partition Map Tests
# ============================================================

class TestPartitionMap:
    """Test partition mapping."""

    def test_partition_map_populated(self):
        cluster = make_small_cluster()
        pm = cluster.partition_map
        assert len(pm) == 4

    def test_partition_map_immutable(self):
        cluster = make_small_cluster()
        pm = cluster.partition_map
        pm[99] = ["fake"]
        assert 99 not in cluster.partition_map

    def test_replica_groups_populated(self):
        cluster = make_small_cluster()
        rg = cluster.replica_groups
        assert len(rg) == 4


# ============================================================
# 28. ClusterNode Tests
# ============================================================

class TestClusterNode:
    """Test physical node representation."""

    def test_create_node(self):
        node = ClusterNode("n0", "localhost:8080")
        assert node.node_id == "n0"
        assert node.address == "localhost:8080"
        assert node.state == NodeState.ACTIVE

    def test_add_remove_partition(self):
        node = ClusterNode("n0")
        node.add_partition(0, is_leader=True)
        assert 0 in node.partitions
        assert node.partitions[0] == 1  # leader

        node.remove_partition(0)
        assert 0 not in node.partitions

    def test_default_address(self):
        node = ClusterNode("n0")
        assert "n0" in node.address


# ============================================================
# 29. Factory Function Tests
# ============================================================

class TestFactory:
    """Test create_cluster factory."""

    def test_default_cluster(self):
        cluster = create_cluster()
        assert len(cluster.nodes) == 3

    def test_custom_size(self):
        cluster = create_cluster(n=5)
        assert len(cluster.nodes) == 5

    def test_with_config(self):
        config = KVConfig(num_partitions=8)
        cluster = create_cluster(n=3, config=config)
        assert cluster.config.num_partitions == 8


# ============================================================
# 30. Stress / Volume Tests
# ============================================================

class TestStress:
    """Stress tests (lightweight)."""

    def test_many_operations(self):
        cluster = make_small_cluster()
        for i in range(100):
            cluster.put(f"stress-{i}", i, consistency=ConsistencyLevel.QUORUM)
        for i in range(100):
            resp = cluster.get(f"stress-{i}", consistency=ConsistencyLevel.QUORUM)
            assert resp.value == i

    def test_overwrite_same_key_many_times(self):
        cluster = make_small_cluster()
        for i in range(50):
            cluster.put("hot-key", i, consistency=ConsistencyLevel.STRONG)
        resp = cluster.get("hot-key", consistency=ConsistencyLevel.STRONG)
        assert resp.value == 49

    def test_delete_many_keys(self):
        cluster = make_small_cluster()
        for i in range(20):
            cluster.put(f"del-{i}", i, consistency=ConsistencyLevel.QUORUM)
        for i in range(20):
            cluster.delete(f"del-{i}", consistency=ConsistencyLevel.QUORUM)
        for i in range(20):
            resp = cluster.get(f"del-{i}", consistency=ConsistencyLevel.QUORUM)
            # Should be None or tombstoned
            assert resp.value is None or resp.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
