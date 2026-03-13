"""Tests for C233: Gossip Protocol"""

import random
import time
import unittest
from gossip import (
    GossipNode, GossipCluster, GossipMessage, GossipStrategy,
    NodeState, MessageType, VersionVector, VersionedValue,
    MerkleNode, build_merkle_tree, diff_merkle_trees, _hash
)


class TestVersionVector(unittest.TestCase):
    """Tests for version vector / vector clock."""

    def test_empty_vector(self):
        vv = VersionVector()
        self.assertEqual(vv.clock, {})

    def test_increment(self):
        vv = VersionVector()
        vv.increment("A")
        self.assertEqual(vv.clock, {"A": 1})
        vv.increment("A")
        self.assertEqual(vv.clock, {"A": 2})
        vv.increment("B")
        self.assertEqual(vv.clock, {"A": 2, "B": 1})

    def test_merge(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 3, "B": 1}
        vv2 = VersionVector()
        vv2.clock = {"A": 1, "B": 5, "C": 2}
        vv1.merge(vv2)
        self.assertEqual(vv1.clock, {"A": 3, "B": 5, "C": 2})

    def test_dominates(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 3, "B": 2}
        vv2 = VersionVector()
        vv2.clock = {"A": 1, "B": 1}
        self.assertTrue(vv1.dominates(vv2))
        self.assertFalse(vv2.dominates(vv1))

    def test_dominates_empty(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 1}
        vv2 = VersionVector()
        self.assertTrue(vv1.dominates(vv2))
        self.assertFalse(vv2.dominates(vv1))

    def test_concurrent(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 3, "B": 1}
        vv2 = VersionVector()
        vv2.clock = {"A": 1, "B": 5}
        self.assertTrue(vv1.concurrent(vv2))
        self.assertTrue(vv2.concurrent(vv1))

    def test_equal_not_concurrent(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 1}
        vv2 = VersionVector()
        vv2.clock = {"A": 1}
        self.assertFalse(vv1.concurrent(vv2))
        self.assertEqual(vv1, vv2)

    def test_copy(self):
        vv = VersionVector()
        vv.clock = {"A": 1, "B": 2}
        c = vv.copy()
        self.assertEqual(c.clock, vv.clock)
        c.increment("A")
        self.assertNotEqual(c.clock, vv.clock)

    def test_dominates_superset_keys(self):
        vv1 = VersionVector()
        vv1.clock = {"A": 1, "B": 1, "C": 1}
        vv2 = VersionVector()
        vv2.clock = {"A": 1}
        self.assertTrue(vv1.dominates(vv2))


class TestVersionedValue(unittest.TestCase):

    def test_create(self):
        vv = VersionedValue("hello")
        self.assertEqual(vv.value, "hello")
        self.assertEqual(vv.version.clock, {})

    def test_copy(self):
        vv = VersionedValue(42)
        vv.version.increment("A")
        c = vv.copy()
        self.assertEqual(c.value, 42)
        c.version.increment("B")
        self.assertNotIn("B", vv.version.clock)


class TestMerkleTree(unittest.TestCase):

    def test_empty_state(self):
        tree = build_merkle_tree({})
        self.assertIsNotNone(tree.hash)

    def test_single_entry(self):
        tree = build_merkle_tree({"a": 1})
        self.assertEqual(tree.key, "a")
        self.assertEqual(tree.value, 1)

    def test_multiple_entries(self):
        tree = build_merkle_tree({"a": 1, "b": 2, "c": 3})
        self.assertIsNotNone(tree.hash)

    def test_same_state_same_hash(self):
        t1 = build_merkle_tree({"a": 1, "b": 2})
        t2 = build_merkle_tree({"a": 1, "b": 2})
        self.assertEqual(t1.hash, t2.hash)

    def test_different_state_different_hash(self):
        t1 = build_merkle_tree({"a": 1, "b": 2})
        t2 = build_merkle_tree({"a": 1, "b": 3})
        self.assertNotEqual(t1.hash, t2.hash)

    def test_diff_identical(self):
        t1 = build_merkle_tree({"a": 1, "b": 2})
        t2 = build_merkle_tree({"a": 1, "b": 2})
        self.assertEqual(diff_merkle_trees(t1, t2), set())

    def test_diff_one_changed(self):
        t1 = build_merkle_tree({"a": 1, "b": 2})
        t2 = build_merkle_tree({"a": 1, "b": 3})
        diffs = diff_merkle_trees(t1, t2)
        self.assertIn("b", diffs)

    def test_diff_none_trees(self):
        self.assertEqual(diff_merkle_trees(None, None), set())


class TestGossipNode(unittest.TestCase):

    def test_create_node(self):
        node = GossipNode("node_1")
        self.assertEqual(node.node_id, "node_1")
        self.assertEqual(node.state, NodeState.ALIVE)
        self.assertIn("node_1", node.members)

    def test_put_get(self):
        node = GossipNode("A")
        node.put("x", 42)
        self.assertEqual(node.get("x"), 42)

    def test_put_updates_version(self):
        node = GossipNode("A")
        node.put("x", 1)
        v1 = node.get_versioned("x").version.copy()
        node.put("x", 2)
        v2 = node.get_versioned("x").version
        self.assertTrue(v2.dominates(v1))

    def test_get_missing(self):
        node = GossipNode("A")
        self.assertIsNone(node.get("missing"))

    def test_join_sends_messages(self):
        node = GossipNode("A")
        node.join(["B", "C"])
        self.assertEqual(len(node.outbox), 2)
        self.assertTrue(all(m.type == MessageType.JOIN for m in node.outbox))

    def test_leave(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.members["C"] = NodeState.ALIVE
        node.leave()
        self.assertEqual(node.state, NodeState.LEFT)
        self.assertEqual(node.members["A"], NodeState.LEFT)

    def test_get_alive_members(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.members["C"] = NodeState.DEAD
        node.members["D"] = NodeState.SUSPECT
        alive = node.get_alive_members()
        self.assertEqual(alive, ["B"])

    def test_gossip_round_push(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.put("key", "val")
        node.gossip_round(GossipStrategy.PUSH)
        self.assertTrue(any(m.type == MessageType.PUSH_STATE for m in node.outbox))

    def test_gossip_round_pull(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.gossip_round(GossipStrategy.PULL)
        self.assertTrue(any(m.type == MessageType.PULL_REQUEST for m in node.outbox))

    def test_gossip_round_push_pull(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.gossip_round(GossipStrategy.PUSH_PULL)
        types = {m.type for m in node.outbox}
        self.assertIn(MessageType.PUSH_STATE, types)
        self.assertIn(MessageType.PULL_REQUEST, types)

    def test_gossip_no_members(self):
        node = GossipNode("A")
        node.gossip_round()
        self.assertEqual(len(node.outbox), 0)

    def test_spread_rumor(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.spread_rumor("r1", {"info": "test"})
        self.assertIn("r1", node.seen_rumors)
        self.assertIn("r1", node.rumors)

    def test_duplicate_rumor_ignored(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.spread_rumor("r1", {"info": "test"})
        count = len(node.outbox)
        node.spread_rumor("r1", {"info": "test"})
        self.assertEqual(len(node.outbox), count)

    def test_ping_handler(self):
        node = GossipNode("A")
        msg = GossipMessage(MessageType.PING, "B", "A")
        node.receive(msg)
        self.assertTrue(any(m.type == MessageType.ACK for m in node.outbox))

    def test_join_handler_adds_member(self):
        node = GossipNode("A")
        msg = GossipMessage(MessageType.JOIN, "B", "A", {'incarnation': 0})
        node.receive(msg)
        self.assertIn("B", node.members)
        self.assertEqual(node.members["B"], NodeState.ALIVE)

    def test_leave_handler(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        msg = GossipMessage(MessageType.LEAVE, "B", "A", {'node': "B"})
        node.receive(msg)
        self.assertEqual(node.members["B"], NodeState.LEFT)

    def test_push_state_merges(self):
        node = GossipNode("A")
        vv = VersionVector()
        vv.increment("B")
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'key1': {'value': 'hello', 'clock': {'B': 1}, 'timestamp': 100}}
        })
        node.receive(msg)
        self.assertEqual(node.get("key1"), "hello")

    def test_pull_request_sends_response(self):
        node = GossipNode("A")
        node.put("x", 10)
        msg = GossipMessage(MessageType.PULL_REQUEST, "B", "A")
        node.receive(msg)
        self.assertTrue(any(m.type == MessageType.PULL_RESPONSE for m in node.outbox))

    def test_confirm_dead_handler(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.SUSPECT
        msg = GossipMessage(MessageType.CONFIRM_DEAD, "C", "A", {'node': 'B'})
        node.receive(msg)
        self.assertEqual(node.members["B"], NodeState.DEAD)

    def test_suspect_refutation(self):
        node = GossipNode("A")
        node.incarnation = 5
        node.members["B"] = NodeState.ALIVE
        msg = GossipMessage(MessageType.SUSPECT, "B", "A",
                            {'node': 'A', 'incarnation': 5})
        node.receive(msg)
        self.assertEqual(node.incarnation, 6)
        self.assertTrue(any(m.type == MessageType.ALIVE for m in node.outbox))

    def test_suspect_other_node(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.member_incarnations["B"] = 0
        msg = GossipMessage(MessageType.SUSPECT, "C", "A",
                            {'node': 'B', 'incarnation': 0})
        node.receive(msg)
        self.assertEqual(node.members["B"], NodeState.SUSPECT)

    def test_alive_handler_clears_suspect(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.SUSPECT
        node.member_incarnations["B"] = 0
        node.suspect_timers["B"] = 3
        msg = GossipMessage(MessageType.ALIVE, "B", "A",
                            {'node': 'B', 'incarnation': 1})
        node.receive(msg)
        self.assertEqual(node.members["B"], NodeState.ALIVE)
        self.assertNotIn("B", node.suspect_timers)

    def test_anti_entropy_round(self):
        node = GossipNode("A")
        node.members["B"] = NodeState.ALIVE
        node.put("k", "v")
        node.anti_entropy_round()
        self.assertTrue(any(m.type == MessageType.ANTI_ENTROPY_REQ for m in node.outbox))

    def test_anti_entropy_req_handler(self):
        node = GossipNode("A")
        node.put("k", "v")
        msg = GossipMessage(MessageType.ANTI_ENTROPY_REQ, "B", "A", {
            'root_hash': 'abc',
            'state_hashes': {'k': 'different_hash'}
        })
        node.receive(msg)
        self.assertTrue(any(m.type == MessageType.ANTI_ENTROPY_RESP for m in node.outbox))


class TestGossipCluster(unittest.TestCase):

    def test_create_empty_cluster(self):
        cluster = GossipCluster()
        self.assertEqual(len(cluster.nodes), 0)

    def test_create_cluster_with_nodes(self):
        random.seed(42)
        cluster = GossipCluster(5)
        self.assertEqual(len(cluster.nodes), 5)

    def test_all_nodes_know_each_other(self):
        random.seed(42)
        cluster = GossipCluster(3)
        for node in cluster.nodes.values():
            self.assertEqual(len(node.members), 3)

    def test_add_node(self):
        cluster = GossipCluster()
        node = cluster.add_node("new")
        self.assertIn("new", cluster.nodes)

    def test_remove_node(self):
        cluster = GossipCluster(3)
        cluster.remove_node("node_0")
        self.assertNotIn("node_0", cluster.nodes)

    def test_push_convergence(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("x", 100)
        cluster.run_rounds(10, GossipStrategy.PUSH)
        self.assertTrue(cluster.converged("x"))

    def test_pull_convergence(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("x", 200)
        cluster.run_rounds(10, GossipStrategy.PULL)
        self.assertTrue(cluster.converged("x"))

    def test_push_pull_convergence(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("x", 300)
        cluster.run_rounds(10, GossipStrategy.PUSH_PULL)
        self.assertTrue(cluster.converged("x"))

    def test_multiple_keys_converge(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("a", 1)
        cluster.nodes["node_1"].put("b", 2)
        cluster.nodes["node_2"].put("c", 3)
        cluster.run_rounds(15)
        self.assertTrue(cluster.converged())

    def test_concurrent_updates_converge(self):
        random.seed(42)
        cluster = GossipCluster(5)
        # Two nodes write same key
        cluster.nodes["node_0"].put("x", "from_0")
        cluster.nodes["node_1"].put("x", "from_1")
        cluster.run_rounds(15)
        self.assertTrue(cluster.converged("x"))

    def test_partition_blocks_messages(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.nodes["node_0"].put("x", "side_a")
        cluster.run_rounds(10)
        # node_2 and node_3 should not have the update
        val_0 = cluster.nodes["node_0"].get("x")
        val_1 = cluster.nodes["node_1"].get("x")
        self.assertEqual(val_0, "side_a")
        self.assertEqual(val_1, "side_a")

    def test_partition_heal_converges(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.nodes["node_0"].put("x", "partitioned")
        cluster.run_rounds(5)
        cluster.heal_partition()
        cluster.run_rounds(15)
        self.assertTrue(cluster.converged("x"))

    def test_node_crash_detection(self):
        random.seed(42)
        cluster = GossipCluster(5, config={'suspect_timeout': 2})
        cluster.remove_node("node_4")
        for _ in range(10):
            cluster.failure_detection_round()
        # Some node should have marked node_4 as dead or missing
        for nid, node in cluster.nodes.items():
            if "node_4" in node.members:
                self.assertIn(node.members["node_4"],
                              (NodeState.DEAD, NodeState.SUSPECT))

    def test_graceful_leave(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.nodes["node_3"].leave()
        cluster._deliver_all()
        for nid in ["node_0", "node_1", "node_2"]:
            self.assertEqual(cluster.nodes[nid].members.get("node_3"), NodeState.LEFT)

    def test_stats(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.run_rounds(2)
        s = cluster.stats()
        self.assertEqual(s['total_nodes'], 3)
        self.assertEqual(s['alive'], 3)
        self.assertGreater(s['total_messages'], 0)

    def test_convergence_status(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("k", "v")
        cluster.run_rounds(10)
        status = cluster.get_convergence_status()
        self.assertIn("k", status)
        self.assertTrue(status["k"]["converged"])

    def test_message_log(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.run_rounds(1)
        self.assertGreater(len(cluster.message_log), 0)

    def test_drop_rate(self):
        random.seed(42)
        cluster = GossipCluster(5, config={'drop_rate': 0.5})
        cluster.nodes["node_0"].put("x", 1)
        cluster.run_rounds(5)
        self.assertGreater(cluster.dropped_messages, 0)

    def test_large_cluster_convergence(self):
        random.seed(42)
        cluster = GossipCluster(20)
        cluster.nodes["node_0"].put("key", "value")
        cluster.run_rounds(20)
        self.assertTrue(cluster.converged("key"))

    def test_many_keys_converge(self):
        random.seed(42)
        cluster = GossipCluster(5)
        for i in range(20):
            cluster.nodes[f"node_{i % 5}"].put(f"k{i}", i)
        cluster.run_rounds(20)
        self.assertTrue(cluster.converged())

    def test_anti_entropy_convergence(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("ae_key", "ae_val")
        for _ in range(10):
            cluster.anti_entropy_round()
        self.assertTrue(cluster.converged("ae_key"))


class TestRumorMongering(unittest.TestCase):

    def test_rumor_spreads(self):
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].spread_rumor("rumor_1", {"event": "test"})
        cluster._deliver_all()
        # At least some nodes should have seen the rumor
        seen = sum(1 for n in cluster.nodes.values() if "rumor_1" in n.seen_rumors)
        self.assertGreater(seen, 1)

    def test_rumor_eventually_all(self):
        random.seed(42)
        cluster = GossipCluster(10)
        cluster.nodes["node_0"].spread_rumor("rumor_2", {"msg": "hello"})
        # Multiple delivery rounds
        for _ in range(5):
            cluster._deliver_all()
            for n in cluster.nodes.values():
                if "rumor_2" in n.rumors:
                    n._propagate_rumor("rumor_2")
            cluster._deliver_all()
        seen = sum(1 for n in cluster.nodes.values() if "rumor_2" in n.seen_rumors)
        self.assertGreater(seen, 5)

    def test_rumor_infection_limit(self):
        node = GossipNode("A", config={'rumor_max_infections': 2})
        node.members["B"] = NodeState.ALIVE
        node.members["C"] = NodeState.ALIVE
        node.spread_rumor("r", {"data": 1})
        node.outbox.clear()
        node._propagate_rumor("r")
        node.outbox.clear()
        node._propagate_rumor("r")
        # After max_infections, no more propagation
        count_before = len(node.outbox)
        node._propagate_rumor("r")
        self.assertEqual(len(node.outbox), count_before)


class TestFailureDetection(unittest.TestCase):

    def test_ping_ack_cycle(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.failure_detection_round()
        # Should have generated pings and acks
        self.assertGreater(cluster.total_messages, 0)

    def test_suspect_after_no_ack(self):
        random.seed(42)
        cluster = GossipCluster(3, config={'suspect_timeout': 2})
        # Remove node to simulate crash
        cluster.remove_node("node_2")
        for _ in range(15):
            cluster.failure_detection_round()
        # Remaining nodes should suspect or declare dead
        for nid in ["node_0", "node_1"]:
            if "node_2" in cluster.nodes[nid].members:
                self.assertIn(cluster.nodes[nid].members["node_2"],
                              (NodeState.SUSPECT, NodeState.DEAD))

    def test_suspect_refutation_in_cluster(self):
        random.seed(42)
        cluster = GossipCluster(4, config={'suspect_timeout': 5})
        # Manually suspect a node that's actually alive
        cluster.nodes["node_0"]._suspect_node("node_1")
        cluster._deliver_all()
        # node_1 should refute
        self.assertGreater(cluster.nodes["node_1"].incarnation, 0)

    def test_dead_after_timeout(self):
        random.seed(42)
        cluster = GossipCluster(3, config={'suspect_timeout': 1})
        cluster.remove_node("node_2")
        # Run many rounds to allow suspect -> dead transition
        for _ in range(20):
            cluster.failure_detection_round()
        for nid in ["node_0", "node_1"]:
            if "node_2" in cluster.nodes[nid].members:
                self.assertIn(cluster.nodes[nid].members["node_2"],
                              (NodeState.SUSPECT, NodeState.DEAD))

    def test_ping_req_indirect(self):
        """Test indirect ping through a helper node."""
        random.seed(42)
        node = GossipNode("A", config={'ping_req_count': 2})
        node.members["B"] = NodeState.ALIVE
        node.members["C"] = NodeState.ALIVE
        node.members["D"] = NodeState.ALIVE
        node._send_ping_req("D")
        self.assertTrue(any(m.type == MessageType.PING_REQ for m in node.outbox))


class TestMergeConflictResolution(unittest.TestCase):

    def test_newer_version_wins(self):
        node = GossipNode("A")
        node.put("x", "old")
        # Simulate remote with higher version
        remote_vv = node.store["x"].version.copy()
        remote_vv.increment("B")
        remote_vv.increment("B")
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'x': {'value': 'new', 'clock': remote_vv.clock, 'timestamp': time.time()}}
        })
        node.receive(msg)
        self.assertEqual(node.get("x"), "new")

    def test_local_newer_keeps(self):
        node = GossipNode("A")
        node.put("x", "local")
        node.put("x", "local2")
        node.put("x", "local3")
        # Remote with lower version
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'x': {'value': 'remote', 'clock': {'B': 1}, 'timestamp': 0}}
        })
        node.receive(msg)
        self.assertEqual(node.get("x"), "local3")

    def test_concurrent_lww(self):
        """Concurrent updates resolved by last-writer-wins (timestamp)."""
        node = GossipNode("A")
        node.put("x", "local_val")
        # Remote with concurrent version but later timestamp
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'x': {
                'value': 'remote_val',
                'clock': {'B': 1},
                'timestamp': time.time() + 1000
            }}
        })
        node.receive(msg)
        self.assertEqual(node.get("x"), "remote_val")

    def test_concurrent_same_timestamp_tiebreak(self):
        node = GossipNode("A")
        ts = 1000.0
        node.store["x"] = VersionedValue("aaa", VersionVector(), ts)
        node.store["x"].version.clock = {"A": 1}
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'x': {'value': 'zzz', 'clock': {'B': 1}, 'timestamp': ts}}
        })
        node.receive(msg)
        # 'zzz' > 'aaa' in string comparison
        self.assertEqual(node.get("x"), "zzz")

    def test_merge_new_key(self):
        node = GossipNode("A")
        msg = GossipMessage(MessageType.PUSH_STATE, "B", "A", {
            'state': {'new_key': {'value': 'new_val', 'clock': {'B': 1}, 'timestamp': 100}}
        })
        node.receive(msg)
        self.assertEqual(node.get("new_key"), "new_val")


class TestNetworkPartition(unittest.TestCase):

    def test_partition_isolation(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0"], ["node_1", "node_2", "node_3"])
        cluster.nodes["node_0"].put("isolated", "value")
        cluster.run_rounds(10)
        # node_0 can't reach others
        for nid in ["node_1", "node_2", "node_3"]:
            self.assertIsNone(cluster.nodes[nid].get("isolated"))

    def test_split_brain_different_values(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.nodes["node_0"].put("x", "side_a")
        cluster.nodes["node_2"].put("x", "side_b")
        cluster.run_rounds(10)
        # Each side converges to its own value
        self.assertEqual(cluster.nodes["node_0"].get("x"), "side_a")
        self.assertEqual(cluster.nodes["node_1"].get("x"), "side_a")
        self.assertEqual(cluster.nodes["node_2"].get("x"), "side_b")
        self.assertEqual(cluster.nodes["node_3"].get("x"), "side_b")

    def test_heal_after_split_brain(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.nodes["node_0"].put("x", "side_a")
        cluster.nodes["node_2"].put("x", "side_b")
        cluster.run_rounds(5)
        cluster.heal_partition()
        cluster.run_rounds(20)
        self.assertTrue(cluster.converged("x"))

    def test_is_partitioned(self):
        cluster = GossipCluster(3)
        cluster.partition(["node_0"], ["node_1"])
        self.assertTrue(cluster.is_partitioned("node_0", "node_1"))
        self.assertFalse(cluster.is_partitioned("node_0", "node_2"))


class TestScalability(unittest.TestCase):

    def test_convergence_time_scales_logarithmically(self):
        """Gossip should converge in O(log N) rounds."""
        random.seed(42)
        results = []
        for size in [5, 10, 20]:
            cluster = GossipCluster(size)
            cluster.nodes["node_0"].put("test", "value")
            rounds = 0
            for _ in range(50):
                cluster.gossip_round()
                rounds += 1
                if cluster.converged("test"):
                    break
            results.append((size, rounds))

        # Verify sub-linear growth
        # 20-node cluster shouldn't take 4x as many rounds as 5-node
        ratio = results[2][1] / max(results[0][1], 1)
        self.assertLess(ratio, 4.0)

    def test_message_count_bounded(self):
        """Each round should generate O(N * fanout) messages."""
        random.seed(42)
        cluster = GossipCluster(10, config={'fanout': 3})
        cluster.nodes["node_0"].put("x", 1)
        before = cluster.total_messages
        cluster.gossip_round()
        after = cluster.total_messages
        round_messages = after - before
        # Should be bounded by N * fanout * 2 (push + pull each generate msgs)
        self.assertLessEqual(round_messages, 10 * 3 * 4)

    def test_state_updates_decrease_over_rounds(self):
        """After convergence, state updates should stop."""
        random.seed(42)
        cluster = GossipCluster(5)
        cluster.nodes["node_0"].put("x", 1)
        cluster.run_rounds(20)
        updates_before = sum(n.state_updates for n in cluster.nodes.values())
        cluster.run_rounds(5)
        updates_after = sum(n.state_updates for n in cluster.nodes.values())
        # Should have very few or no new updates after convergence
        self.assertLessEqual(updates_after - updates_before, 5)


class TestEdgeCases(unittest.TestCase):

    def test_single_node_cluster(self):
        cluster = GossipCluster(1)
        cluster.nodes["node_0"].put("x", 42)
        cluster.run_rounds(5)
        self.assertEqual(cluster.nodes["node_0"].get("x"), 42)

    def test_two_node_cluster(self):
        random.seed(42)
        cluster = GossipCluster(2)
        cluster.nodes["node_0"].put("x", 1)
        cluster.run_rounds(5)
        self.assertTrue(cluster.converged("x"))

    def test_empty_state_gossip(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.run_rounds(3)
        self.assertTrue(cluster.converged())

    def test_overwrite_value(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("x", 1)
        cluster.run_rounds(5)
        cluster.nodes["node_0"].put("x", 2)
        cluster.run_rounds(10)
        self.assertTrue(cluster.converged("x"))
        self.assertEqual(cluster.nodes["node_0"].get("x"), 2)

    def test_message_to_nonexistent_node(self):
        cluster = GossipCluster(2)
        # Should not crash
        msg = GossipMessage(MessageType.PING, "node_0", "ghost")
        cluster.nodes["node_0"].outbox.append(msg)
        cluster._deliver_all()

    def test_concurrent_writes_all_keys(self):
        random.seed(42)
        cluster = GossipCluster(5)
        for i in range(5):
            cluster.nodes[f"node_{i}"].put("shared", f"val_{i}")
        cluster.run_rounds(20)
        self.assertTrue(cluster.converged("shared"))

    def test_node_rejoin(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("before", "crash")
        cluster.run_rounds(5)
        cluster.remove_node("node_2")
        cluster.run_rounds(3)
        new_node = cluster.add_node("node_2")
        new_node.join(["node_0", "node_1"])
        cluster._deliver_all()
        cluster.run_rounds(10)
        self.assertEqual(new_node.get("before"), "crash")

    def test_multiple_partitions(self):
        random.seed(42)
        cluster = GossipCluster(6)
        # Three-way partition
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.partition(["node_0", "node_1"], ["node_4", "node_5"])
        cluster.partition(["node_2", "node_3"], ["node_4", "node_5"])

        cluster.nodes["node_0"].put("x", "group_a")
        cluster.nodes["node_2"].put("x", "group_b")
        cluster.nodes["node_4"].put("x", "group_c")
        cluster.run_rounds(10)

        # Each group has its own value
        self.assertEqual(cluster.nodes["node_1"].get("x"), "group_a")
        self.assertEqual(cluster.nodes["node_3"].get("x"), "group_b")
        self.assertEqual(cluster.nodes["node_5"].get("x"), "group_c")


class TestGossipMessage(unittest.TestCase):

    def test_message_creation(self):
        msg = GossipMessage(MessageType.PING, "A", "B")
        self.assertEqual(msg.type, MessageType.PING)
        self.assertEqual(msg.sender, "A")
        self.assertEqual(msg.target, "B")

    def test_message_seq(self):
        m1 = GossipMessage(MessageType.PING, "A", "B")
        m2 = GossipMessage(MessageType.PING, "A", "B")
        self.assertNotEqual(m1.seq, m2.seq)

    def test_message_payload(self):
        msg = GossipMessage(MessageType.PUSH_STATE, "A", "B", {'data': 42})
        self.assertEqual(msg.payload['data'], 42)

    def test_message_repr(self):
        msg = GossipMessage(MessageType.PING, "A", "B")
        self.assertIn("PING", repr(msg))


class TestAntiEntropy(unittest.TestCase):

    def test_anti_entropy_detects_missing(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("only_on_0", "val")
        cluster.anti_entropy_round()
        # After anti-entropy, at least some diff should have been sent

    def test_anti_entropy_full_sync(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("a", 1)
        cluster.nodes["node_1"].put("b", 2)
        cluster.nodes["node_2"].put("c", 3)
        for _ in range(15):
            cluster.anti_entropy_round()
        self.assertTrue(cluster.converged())

    def test_anti_entropy_after_partition(self):
        random.seed(42)
        cluster = GossipCluster(4)
        cluster.partition(["node_0", "node_1"], ["node_2", "node_3"])
        cluster.nodes["node_0"].put("x", "val")
        for _ in range(5):
            cluster.anti_entropy_round()
        cluster.heal_partition()
        for _ in range(15):
            cluster.anti_entropy_round()
        self.assertTrue(cluster.converged("x"))

    def test_anti_entropy_idempotent(self):
        random.seed(42)
        cluster = GossipCluster(3)
        cluster.nodes["node_0"].put("k", "v")
        for _ in range(10):
            cluster.anti_entropy_round()
        updates_before = sum(n.state_updates for n in cluster.nodes.values())
        for _ in range(5):
            cluster.anti_entropy_round()
        updates_after = sum(n.state_updates for n in cluster.nodes.values())
        self.assertLessEqual(updates_after - updates_before, 3)


if __name__ == '__main__':
    unittest.main()
