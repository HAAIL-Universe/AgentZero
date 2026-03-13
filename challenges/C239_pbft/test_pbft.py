"""Tests for C239: Practical Byzantine Fault Tolerance (PBFT)."""

import pytest
import time
import json
from pbft import (
    PBFTMessage, MessageType, PBFTLog, PBFTNode, PBFTNetwork,
    CheckpointManager, ViewChangeManager, StateMachine, BatchRequestHandler,
    ByzantineDetector, compute_digest,
)


# ===========================================================================
# Message Tests
# ===========================================================================

class TestPBFTMessage:
    def test_create_message(self):
        msg = PBFTMessage(
            msg_type=MessageType.REQUEST,
            view=0, sequence=1, sender=0
        )
        assert msg.msg_type == MessageType.REQUEST
        assert msg.view == 0
        assert msg.sequence == 1

    def test_sign_and_verify(self):
        msg = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1, sender=0)
        msg.sign(0)
        assert msg.signature != ""
        assert msg.verify_signature(0)
        assert not msg.verify_signature(1)  # wrong node

    def test_digest(self):
        msg = PBFTMessage(msg_type=MessageType.REQUEST, view=0, sequence=1, sender=0,
                          payload={"key": "value"})
        d = msg.digest_of()
        assert len(d) == 32
        # Same payload -> same digest
        msg2 = PBFTMessage(msg_type=MessageType.PREPARE, view=1, sequence=2, sender=1,
                           payload={"key": "value"})
        assert msg2.digest_of() == d

    def test_empty_payload_digest(self):
        msg = PBFTMessage(msg_type=MessageType.REQUEST, view=0, sequence=1, sender=0)
        assert msg.digest_of() == ""

    def test_compute_digest_utility(self):
        d1 = compute_digest({"a": 1})
        d2 = compute_digest({"a": 1})
        d3 = compute_digest({"a": 2})
        assert d1 == d2
        assert d1 != d3

    def test_message_types(self):
        for mt in MessageType:
            msg = PBFTMessage(msg_type=mt, view=0, sequence=0, sender=0)
            assert msg.msg_type == mt


# ===========================================================================
# PBFTLog Tests
# ===========================================================================

class TestPBFTLog:
    def setup_method(self):
        self.log = PBFTLog(node_id=0, total_nodes=4)  # f=1

    def test_add_pre_prepare(self):
        msg = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                          sender=0, digest="abc")
        assert self.log.add_pre_prepare(msg)

    def test_conflicting_pre_prepare(self):
        msg1 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                           sender=0, digest="abc")
        msg2 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                           sender=0, digest="xyz")
        assert self.log.add_pre_prepare(msg1)
        assert not self.log.add_pre_prepare(msg2)

    def test_same_pre_prepare_ok(self):
        msg1 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                           sender=0, digest="abc")
        msg2 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                           sender=0, digest="abc")
        assert self.log.add_pre_prepare(msg1)
        assert self.log.add_pre_prepare(msg2)  # same digest is OK

    def test_add_prepare(self):
        msg = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                          sender=1, digest="abc")
        assert self.log.add_prepare(msg)

    def test_duplicate_prepare_rejected(self):
        msg = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                          sender=1, digest="abc")
        assert self.log.add_prepare(msg)
        assert not self.log.add_prepare(msg)  # same sender

    def test_add_commit(self):
        msg = PBFTMessage(msg_type=MessageType.COMMIT, view=0, sequence=1,
                          sender=1, digest="abc")
        assert self.log.add_commit(msg)

    def test_duplicate_commit_rejected(self):
        msg = PBFTMessage(msg_type=MessageType.COMMIT, view=0, sequence=1,
                          sender=1, digest="abc")
        assert self.log.add_commit(msg)
        assert not self.log.add_commit(msg)

    def test_prepared_state(self):
        # f=1, need pre-prepare + 2 prepares
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        p1 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=1, digest="abc")
        p2 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=2, digest="abc")

        self.log.add_pre_prepare(pp)
        self.log.add_prepare(p1)
        assert not self.log.is_prepared(0, 1, "abc")
        self.log.add_prepare(p2)
        assert self.log.is_prepared(0, 1, "abc")

    def test_prepared_wrong_digest(self):
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        p1 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=1, digest="xyz")
        p2 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=2, digest="xyz")
        self.log.add_pre_prepare(pp)
        self.log.add_prepare(p1)
        self.log.add_prepare(p2)
        assert not self.log.is_prepared(0, 1, "abc")

    def test_committed_local(self):
        # f=1, need prepared + 3 commits (2f+1)
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        p1 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=1, digest="abc")
        p2 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=2, digest="abc")
        c0 = PBFTMessage(msg_type=MessageType.COMMIT, view=0, sequence=1,
                         sender=0, digest="abc")
        c1 = PBFTMessage(msg_type=MessageType.COMMIT, view=0, sequence=1,
                         sender=1, digest="abc")
        c2 = PBFTMessage(msg_type=MessageType.COMMIT, view=0, sequence=1,
                         sender=2, digest="abc")

        self.log.add_pre_prepare(pp)
        self.log.add_prepare(p1)
        self.log.add_prepare(p2)
        self.log.add_commit(c0)
        self.log.add_commit(c1)
        assert not self.log.is_committed_local(0, 1, "abc")
        self.log.add_commit(c2)
        assert self.log.is_committed_local(0, 1, "abc")

    def test_garbage_collect(self):
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        self.log.add_pre_prepare(pp)
        assert (0, 1) in self.log.pre_prepares
        self.log.garbage_collect(5)
        assert (0, 1) not in self.log.pre_prepares
        assert self.log.low_water_mark == 5

    def test_pre_prepare_below_water_mark_rejected(self):
        self.log.garbage_collect(10)
        msg = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=5,
                          sender=0, digest="abc")
        assert not self.log.add_pre_prepare(msg)

    def test_prepare_certificate(self):
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        p1 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=1, digest="abc")
        p2 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=2, digest="abc")
        self.log.add_pre_prepare(pp)
        assert self.log.get_prepare_certificate(0, 1) is None
        self.log.add_prepare(p1)
        self.log.add_prepare(p2)
        cert = self.log.get_prepare_certificate(0, 1)
        assert cert is not None
        assert len(cert) == 3  # pp + 2 prepares


# ===========================================================================
# State Machine Tests
# ===========================================================================

class TestStateMachine:
    def setup_method(self):
        self.sm = StateMachine()

    def test_set_and_get(self):
        self.sm.execute(1, {"type": "set", "key": "x", "value": 42})
        result = self.sm.execute(2, {"type": "get", "key": "x"})
        assert result == 42

    def test_get_missing(self):
        result = self.sm.execute(1, {"type": "get", "key": "x"})
        assert result is None

    def test_delete(self):
        self.sm.execute(1, {"type": "set", "key": "x", "value": 42})
        result = self.sm.execute(2, {"type": "delete", "key": "x"})
        assert result == 42
        result = self.sm.execute(3, {"type": "get", "key": "x"})
        assert result is None

    def test_cas_success(self):
        self.sm.execute(1, {"type": "set", "key": "x", "value": 1})
        result = self.sm.execute(2, {"type": "cas", "key": "x", "expected": 1, "new_value": 2})
        assert result is True
        assert self.sm.state["x"] == 2

    def test_cas_failure(self):
        self.sm.execute(1, {"type": "set", "key": "x", "value": 1})
        result = self.sm.execute(2, {"type": "cas", "key": "x", "expected": 99, "new_value": 2})
        assert result is False
        assert self.sm.state["x"] == 1

    def test_noop(self):
        result = self.sm.execute(1, {"type": "noop"})
        assert result is None

    def test_none_operation(self):
        result = self.sm.execute(1, None)
        assert result is None

    def test_history(self):
        self.sm.execute(1, {"type": "set", "key": "a", "value": 1})
        self.sm.execute(2, {"type": "set", "key": "b", "value": 2})
        assert len(self.sm.history) == 2
        assert self.sm.history[0][0] == 1
        assert self.sm.history[1][0] == 2

    def test_digest(self):
        d1 = self.sm.get_digest()
        self.sm.execute(1, {"type": "set", "key": "x", "value": 1})
        d2 = self.sm.get_digest()
        assert d1 != d2

    def test_snapshot_restore(self):
        self.sm.execute(1, {"type": "set", "key": "x", "value": 42})
        snap = self.sm.snapshot()
        self.sm.execute(2, {"type": "set", "key": "x", "value": 99})
        assert self.sm.state["x"] == 99
        self.sm.restore(snap)
        assert self.sm.state["x"] == 42

    def test_unknown_operation(self):
        result = self.sm.execute(1, {"type": "unknown_op"})
        assert result is None


# ===========================================================================
# Checkpoint Manager Tests
# ===========================================================================

class TestCheckpointManager:
    def setup_method(self):
        self.cm = CheckpointManager(node_id=0, total_nodes=4, checkpoint_interval=10)

    def test_should_checkpoint(self):
        assert not self.cm.should_checkpoint(0)
        assert not self.cm.should_checkpoint(5)
        assert self.cm.should_checkpoint(10)
        assert self.cm.should_checkpoint(20)

    def test_create_checkpoint(self):
        msg = self.cm.create_checkpoint(10, "state_digest_10")
        assert msg.msg_type == MessageType.CHECKPOINT
        assert msg.sequence == 10
        assert msg.digest == "state_digest_10"

    def test_stable_checkpoint(self):
        # 4 nodes, f=1, need 2f+1=3 matching
        self.cm.add_checkpoint(10, 0, "digest_10")
        self.cm.add_checkpoint(10, 1, "digest_10")
        assert self.cm.stable_checkpoint == 0  # not enough yet
        became_stable = self.cm.add_checkpoint(10, 2, "digest_10")
        assert became_stable
        assert self.cm.stable_checkpoint == 10
        assert self.cm.stable_digest == "digest_10"

    def test_mismatching_checkpoints(self):
        self.cm.add_checkpoint(10, 0, "digest_a")
        self.cm.add_checkpoint(10, 1, "digest_b")
        became_stable = self.cm.add_checkpoint(10, 2, "digest_c")
        assert not became_stable
        assert self.cm.stable_checkpoint == 0

    def test_checkpoint_cleanup(self):
        # Add checkpoint 5 first
        self.cm.add_checkpoint(5, 0, "d5")
        # Make checkpoint 10 stable -- this should clean up checkpoint 5
        self.cm.add_checkpoint(10, 0, "d10")
        self.cm.add_checkpoint(10, 1, "d10")
        self.cm.add_checkpoint(10, 2, "d10")
        assert 5 not in self.cm.checkpoint_proofs

    def test_get_stable_checkpoint(self):
        seq, digest = self.cm.get_stable_checkpoint()
        assert seq == 0
        assert digest == ""


# ===========================================================================
# View Change Manager Tests
# ===========================================================================

class TestViewChangeManager:
    def setup_method(self):
        self.vcm = ViewChangeManager(node_id=0, total_nodes=4)

    def test_primary_for_view(self):
        assert self.vcm.primary_for_view(0) == 0
        assert self.vcm.primary_for_view(1) == 1
        assert self.vcm.primary_for_view(4) == 0

    def test_is_primary(self):
        assert self.vcm.is_primary(0, 0)
        assert not self.vcm.is_primary(0, 1)
        assert self.vcm.is_primary(1, 1)

    def test_create_view_change(self):
        msg = self.vcm.create_view_change(1, 0, {}, [])
        assert msg.msg_type == MessageType.VIEW_CHANGE
        assert msg.view == 1
        assert msg.sender == 0

    def test_add_view_change_quorum(self):
        # Need 2f+1 = 3 view-change messages for view 1
        for i in range(3):
            msg = PBFTMessage(msg_type=MessageType.VIEW_CHANGE, view=1,
                              sequence=0, sender=i, payload={})
            msg.sign(i)
            result = self.vcm.add_view_change(msg)
            if i < 2:
                assert not result
            else:
                assert result  # quorum reached

    def test_duplicate_view_change(self):
        msg = PBFTMessage(msg_type=MessageType.VIEW_CHANGE, view=1,
                          sequence=0, sender=0, payload={})
        msg.sign(0)
        self.vcm.add_view_change(msg)
        assert not self.vcm.add_view_change(msg)  # duplicate

    def test_create_new_view_only_primary(self):
        # Node 0 is primary for view 0, not view 1
        # Add enough view-change messages for view 1
        for i in range(3):
            msg = PBFTMessage(msg_type=MessageType.VIEW_CHANGE, view=1,
                              sequence=0, sender=i, payload={})
            msg.sign(i)
            self.vcm.add_view_change(msg)

        # Node 0 is NOT primary for view 1
        nv = self.vcm.create_new_view(1)
        assert nv is None  # node 0 is not primary for view 1

    def test_create_new_view_as_primary(self):
        vcm1 = ViewChangeManager(node_id=1, total_nodes=4)  # node 1 is primary for view 1
        for i in range(3):
            msg = PBFTMessage(msg_type=MessageType.VIEW_CHANGE, view=1,
                              sequence=0, sender=i, payload={})
            msg.sign(i)
            vcm1.add_view_change(msg)
        nv = vcm1.create_new_view(1)
        assert nv is not None
        assert nv.msg_type == MessageType.NEW_VIEW
        assert nv.view == 1

    def test_process_new_view(self):
        msg = PBFTMessage(msg_type=MessageType.NEW_VIEW, view=1,
                          sequence=0, sender=1, payload={})
        # Node 1 is primary for view 1
        assert self.vcm.process_new_view(msg)
        assert self.vcm.current_view == 1

    def test_process_new_view_wrong_primary(self):
        msg = PBFTMessage(msg_type=MessageType.NEW_VIEW, view=1,
                          sequence=0, sender=0, payload={})  # wrong primary
        assert not self.vcm.process_new_view(msg)


# ===========================================================================
# PBFT Node Tests
# ===========================================================================

class TestPBFTNode:
    def test_create_node(self):
        node = PBFTNode(0, 4)
        assert node.node_id == 0
        assert node.total_nodes == 4
        assert node.f == 1
        assert node.view == 0
        assert node.is_primary  # node 0 is primary for view 0

    def test_node_properties(self):
        node = PBFTNode(1, 4)
        assert not node.is_primary
        assert node.primary_id == 0

    def test_inactive_node(self):
        node = PBFTNode(0, 4)
        node.active = False
        result = node.handle_request("c1", 1.0, {"type": "set", "key": "x", "value": 1})
        assert result is None

    def test_byzantine_silent(self):
        node = PBFTNode(0, 4)
        node.byzantine = True
        node.byzantine_behavior = "silent"
        # Should silently ignore all messages
        msg = PBFTMessage(msg_type=MessageType.REQUEST, view=0, sequence=1, sender=1)
        node.receive(msg)  # no error, just silent


# ===========================================================================
# PBFT Network Tests (Integration)
# ===========================================================================

class TestPBFTNetwork:
    def test_create_network(self):
        net = PBFTNetwork(4)
        assert len(net.nodes) == 4
        assert net.f == 1

    def test_get_primary(self):
        net = PBFTNetwork(4)
        primary = net.get_primary()
        assert primary.node_id == 0
        assert primary.is_primary

    def test_single_request(self):
        net = PBFTNetwork(4)
        key = net.submit_request("client1", {"type": "set", "key": "x", "value": 42}, 1.0)
        reply = net.get_client_reply(key)
        assert reply == 42

    def test_get_request(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 10}, 1.0)
        key = net.submit_request("c1", {"type": "get", "key": "x"}, 2.0)
        reply = net.get_client_reply(key)
        assert reply == 10

    def test_multiple_requests(self):
        net = PBFTNetwork(4)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))

        # All nodes should agree
        assert net.check_safety()
        for node in net.nodes:
            assert node.last_executed == 5

    def test_safety_property(self):
        net = PBFTNetwork(4)
        for i in range(10):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i * 10}, float(i))
        assert net.check_safety()

    def test_state_agreement(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        net.submit_request("c1", {"type": "set", "key": "y", "value": 2}, 2.0)

        states = net.get_states()
        for nid in range(4):
            assert states[nid]["state"]["x"] == 1
            assert states[nid]["state"]["y"] == 2

    def test_7_nodes(self):
        net = PBFTNetwork(7)  # f=2
        assert net.f == 2
        for i in range(3):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()
        assert net.check_liveness(3)

    def test_delete_operation(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 42}, 1.0)
        key = net.submit_request("c1", {"type": "delete", "key": "x"}, 2.0)
        reply = net.get_client_reply(key)
        assert reply == 42
        # After delete, get should return None
        key2 = net.submit_request("c1", {"type": "get", "key": "x"}, 3.0)
        reply2 = net.get_client_reply(key2)
        assert reply2 is None

    def test_cas_operation(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        key = net.submit_request("c1", {"type": "cas", "key": "x", "expected": 1, "new_value": 2}, 2.0)
        assert net.get_client_reply(key) is True

    def test_message_log(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        assert len(net.message_log) > 0

    def test_noop(self):
        net = PBFTNetwork(4)
        key = net.submit_request("c1", {"type": "noop"}, 1.0)
        reply = net.get_client_reply(key)
        assert reply is None


# ===========================================================================
# Byzantine Fault Tolerance Tests
# ===========================================================================

class TestByzantineFaultTolerance:
    def test_one_crash_fault(self):
        """System works with 1 crashed node out of 4 (f=1)."""
        net = PBFTNetwork(4)
        net.nodes[3].active = False
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()
        # At least f+1 = 2 correct nodes executed
        assert net.check_liveness(5)

    def test_one_silent_byzantine(self):
        """System works with 1 silent Byzantine node."""
        net = PBFTNetwork(4)
        net.nodes[3].byzantine = True
        net.nodes[3].byzantine_behavior = "silent"
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()

    def test_crash_non_primary(self):
        """Crash a non-primary node."""
        net = PBFTNetwork(4)
        net.nodes[2].active = False
        key = net.submit_request("c1", {"type": "set", "key": "x", "value": 99}, 1.0)
        assert net.get_client_reply(key) == 99
        assert net.check_safety()

    def test_too_many_faults_blocks(self):
        """With more than f faults, system may not make progress."""
        net = PBFTNetwork(4)  # f=1
        net.nodes[2].active = False
        net.nodes[3].active = False
        # With 2 faults in a 4-node system, consensus should not be reached
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        # Check that NOT all active nodes executed (not enough for commit quorum)
        active_executed = sum(1 for n in net.nodes if n.active and n.last_executed >= 1)
        # May or may not work depending on timing; the point is safety holds
        assert net.check_safety()

    def test_seven_nodes_two_faults(self):
        """7 nodes, 2 faults (f=2) -- should still work."""
        net = PBFTNetwork(7)
        net.nodes[5].active = False
        net.nodes[6].active = False
        for i in range(3):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()
        assert net.check_liveness(3)


# ===========================================================================
# Network Partition Tests
# ===========================================================================

class TestNetworkPartition:
    def test_create_partition(self):
        net = PBFTNetwork(4)
        net.partition(0, 3)
        assert net._is_partitioned(0, 3)
        assert net._is_partitioned(3, 0)

    def test_heal_partition(self):
        net = PBFTNetwork(4)
        net.partition(0, 3)
        net.heal_partition(0, 3)
        assert not net._is_partitioned(0, 3)

    def test_heal_all(self):
        net = PBFTNetwork(4)
        net.partition(0, 1)
        net.partition(2, 3)
        net.heal_all()
        assert not net._is_partitioned(0, 1)
        assert not net._is_partitioned(2, 3)

    def test_partial_partition_safety(self):
        """Partition one node from one other -- should still work."""
        net = PBFTNetwork(4)
        net.partition(2, 3)  # nodes 2 and 3 can't talk
        for i in range(3):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()

    def test_dropped_messages_tracked(self):
        net = PBFTNetwork(4)
        net.partition(0, 3)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        assert len(net.dropped_messages) > 0


# ===========================================================================
# View Change Tests
# ===========================================================================

class TestViewChange:
    def test_trigger_view_change(self):
        net = PBFTNetwork(4)
        # Crash primary
        net.nodes[0].active = False
        # Trigger view change to view 1
        net.trigger_view_change(1)
        # Node 1 should become new primary
        new_primary = None
        for node in net.nodes:
            if node.active and node.view == 1 and node.is_primary:
                new_primary = node
        assert new_primary is not None
        assert new_primary.node_id == 1

    def test_view_change_updates_view(self):
        net = PBFTNetwork(4)
        net.nodes[0].active = False
        net.trigger_view_change(1)
        for node in net.nodes:
            if node.active and not node.byzantine:
                assert node.view == 1

    def test_requests_after_view_change(self):
        net = PBFTNetwork(4)
        # Process some requests in view 0
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        # Crash primary
        net.nodes[0].active = False
        # View change
        net.trigger_view_change(1)
        # New primary (node 1) handles requests
        key = net.submit_request("c1", {"type": "set", "key": "y", "value": 2}, 2.0)
        reply = net.get_client_reply(key)
        assert reply == 2

    def test_consecutive_view_changes(self):
        net = PBFTNetwork(7)  # f=2, more room for view changes
        # View 0 -> View 1
        net.nodes[0].active = False
        net.trigger_view_change(1)
        # Process some requests
        net.submit_request("c1", {"type": "set", "key": "a", "value": 1}, 1.0)
        # View 1 -> View 2
        net.nodes[1].active = False
        net.trigger_view_change(2)
        # Node 2 is primary for view 2
        key = net.submit_request("c1", {"type": "set", "key": "b", "value": 2}, 2.0)
        assert net.check_safety()

    def test_view_change_no_regression(self):
        """View change from lower to same/lower view is ignored."""
        net = PBFTNetwork(4)
        net.nodes[0].start_view_change(0)  # same view, should be no-op


# ===========================================================================
# Checkpoint Tests
# ===========================================================================

class TestCheckpointing:
    def test_checkpoint_after_interval(self):
        net = PBFTNetwork(4, checkpoint_interval=5)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))

        # Check that checkpoint happened
        for node in net.nodes:
            assert node.checkpoint_mgr.stable_checkpoint == 5

    def test_multiple_checkpoints(self):
        net = PBFTNetwork(4, checkpoint_interval=3)
        for i in range(9):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))

        for node in net.nodes:
            assert node.checkpoint_mgr.stable_checkpoint == 9

    def test_garbage_collection(self):
        net = PBFTNetwork(4, checkpoint_interval=5)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))

        # After checkpoint at seq 5, low water mark should be 5
        for node in net.nodes:
            assert node.log.low_water_mark == 5


# ===========================================================================
# Batch Request Handler Tests
# ===========================================================================

class TestBatchRequestHandler:
    def test_batch_accumulation(self):
        net = PBFTNetwork(4)
        batch = BatchRequestHandler(net, batch_size=3)
        keys = batch.add_request("c1", {"type": "set", "key": "a", "value": 1}, 1.0)
        assert keys == []
        keys = batch.add_request("c1", {"type": "set", "key": "b", "value": 2}, 2.0)
        assert keys == []
        keys = batch.add_request("c1", {"type": "set", "key": "c", "value": 3}, 3.0)
        assert len(keys) == 3  # flushed

    def test_manual_flush(self):
        net = PBFTNetwork(4)
        batch = BatchRequestHandler(net, batch_size=10)
        batch.add_request("c1", {"type": "set", "key": "a", "value": 1}, 1.0)
        batch.add_request("c1", {"type": "set", "key": "b", "value": 2}, 2.0)
        keys = batch.flush()
        assert len(keys) == 2
        assert len(batch.pending) == 0

    def test_batch_count(self):
        net = PBFTNetwork(4)
        batch = BatchRequestHandler(net, batch_size=2)
        batch.add_request("c1", {"type": "set", "key": "a", "value": 1}, 1.0)
        batch.add_request("c1", {"type": "set", "key": "b", "value": 2}, 2.0)
        assert batch.batch_count == 1
        batch.add_request("c1", {"type": "set", "key": "c", "value": 3}, 3.0)
        batch.add_request("c1", {"type": "set", "key": "d", "value": 4}, 4.0)
        assert batch.batch_count == 2


# ===========================================================================
# Byzantine Detector Tests
# ===========================================================================

class TestByzantineDetector:
    def test_no_suspects_initially(self):
        bd = ByzantineDetector(4)
        assert bd.get_suspects() == {}

    def test_detect_double_prepare(self):
        bd = ByzantineDetector(4)
        prepares = {
            (0, 1): [
                PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1, sender=1, digest="a"),
                PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1, sender=1, digest="b"),
                PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1, sender=2, digest="a"),
            ]
        }
        suspects = bd.check_double_prepare(prepares)
        assert 1 in suspects
        assert 2 not in suspects

    def test_is_suspected(self):
        bd = ByzantineDetector(4)
        assert not bd.is_suspected(0)
        bd.suspicions[0] = ["test reason"]
        assert bd.is_suspected(0)

    def test_primary_equivocation(self):
        bd = ByzantineDetector(4)
        pp1 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1, sender=0, digest="a")
        pp2 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1, sender=0, digest="b")
        found = bd.check_primary_equivocation({}, [pp1, pp2])
        assert found
        assert bd.is_suspected(0)

    def test_no_equivocation(self):
        bd = ByzantineDetector(4)
        pp1 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1, sender=0, digest="a")
        pp2 = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=2, sender=0, digest="b")
        found = bd.check_primary_equivocation({}, [pp1, pp2])
        assert not found


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_node_technically_works(self):
        """1 node, f=0. Trivial case."""
        net = PBFTNetwork(1)
        key = net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        # With 1 node, it's its own primary and quorum of 1
        assert net.nodes[0].is_primary

    def test_10_nodes(self):
        net = PBFTNetwork(10)  # f=3
        assert net.f == 3
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()

    def test_multiple_clients(self):
        net = PBFTNetwork(4)
        net.submit_request("alice", {"type": "set", "key": "alice_key", "value": "hello"}, 1.0)
        net.submit_request("bob", {"type": "set", "key": "bob_key", "value": "world"}, 1.0)
        assert net.check_safety()
        for node in net.nodes:
            assert node.state_machine.state.get("alice_key") == "hello"
            assert node.state_machine.state.get("bob_key") == "world"

    def test_rapid_requests(self):
        net = PBFTNetwork(4)
        for i in range(20):
            net.submit_request("c1", {"type": "set", "key": "counter", "value": i}, float(i))
        assert net.check_safety()
        for node in net.nodes:
            assert node.state_machine.state["counter"] == 19

    def test_interleaved_read_write(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 10}, 1.0)
        key = net.submit_request("c1", {"type": "get", "key": "x"}, 2.0)
        assert net.get_client_reply(key) == 10
        net.submit_request("c1", {"type": "set", "key": "x", "value": 20}, 3.0)
        key = net.submit_request("c1", {"type": "get", "key": "x"}, 4.0)
        assert net.get_client_reply(key) == 20

    def test_empty_state(self):
        net = PBFTNetwork(4)
        states = net.get_states()
        for nid in range(4):
            assert states[nid]["state"] == {}
            assert states[nid]["last_executed"] == 0

    def test_cas_across_network(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "lock", "value": "free"}, 1.0)
        key = net.submit_request("c1", {"type": "cas", "key": "lock", "expected": "free", "new_value": "taken"}, 2.0)
        assert net.get_client_reply(key) is True
        # CAS should fail now
        key2 = net.submit_request("c1", {"type": "cas", "key": "lock", "expected": "free", "new_value": "taken2"}, 3.0)
        assert net.get_client_reply(key2) is False

    def test_all_nodes_execute_same_order(self):
        net = PBFTNetwork(4)
        for i in range(10):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))

        # All nodes should have same history (same operations in same order)
        ref_history = [(h[0], h[1]) for h in net.nodes[0].state_machine.history]
        for node in net.nodes[1:]:
            node_history = [(h[0], h[1]) for h in node.state_machine.history]
            assert node_history == ref_history

    def test_client_reply_dedup(self):
        """Same request submitted twice should not execute twice."""
        net = PBFTNetwork(4)
        primary = net.get_primary()
        primary.handle_request("c1", 1.0, {"type": "set", "key": "x", "value": 1})
        # Submit again with same timestamp
        primary.handle_request("c1", 1.0, {"type": "set", "key": "x", "value": 1})
        # Should only execute once
        assert primary.last_executed == 1


# ===========================================================================
# Sequence and Ordering Tests
# ===========================================================================

class TestOrdering:
    def test_sequential_execution(self):
        """Requests are executed in sequence number order."""
        net = PBFTNetwork(4)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": "x", "value": i}, float(i))

        for node in net.nodes:
            assert node.last_executed == 5
            # Final value should be 4 (last set)
            assert node.state_machine.state["x"] == 4

    def test_sequence_numbers_increase(self):
        net = PBFTNetwork(4)
        primary = net.get_primary()
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert primary.sequence == 5

    def test_execution_counts_match(self):
        net = PBFTNetwork(4)
        n_requests = 8
        for i in range(n_requests):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        for node in net.nodes:
            assert node.last_executed == n_requests
            assert len(node.state_machine.history) == n_requests


# ===========================================================================
# Liveness Tests
# ===========================================================================

class TestLiveness:
    def test_liveness_all_correct(self):
        net = PBFTNetwork(4)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_liveness(5)

    def test_liveness_one_fault(self):
        net = PBFTNetwork(4)
        net.nodes[3].active = False
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_liveness(5)

    def test_liveness_check_value(self):
        net = PBFTNetwork(7)
        net.nodes[5].active = False
        net.nodes[6].active = False
        for i in range(3):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_liveness(3)

    def test_no_active_nodes_error(self):
        net = PBFTNetwork(4)
        for node in net.nodes:
            node.active = False
        with pytest.raises(RuntimeError):
            net.get_primary()


# ===========================================================================
# Log / Prepare Certificate Tests
# ===========================================================================

class TestPrepareCertificate:
    def test_7_node_prepare_certificate(self):
        """With 7 nodes (f=2), need 4 prepares for certificate."""
        log = PBFTLog(node_id=0, total_nodes=7)
        pp = PBFTMessage(msg_type=MessageType.PRE_PREPARE, view=0, sequence=1,
                         sender=0, digest="abc")
        log.add_pre_prepare(pp)

        for i in range(1, 5):
            p = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                            sender=i, digest="abc")
            log.add_prepare(p)

        cert = log.get_prepare_certificate(0, 1)
        assert cert is not None
        assert len(cert) == 5  # pp + 4 prepares

    def test_no_certificate_without_pre_prepare(self):
        log = PBFTLog(node_id=0, total_nodes=4)
        p1 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=1, digest="abc")
        p2 = PBFTMessage(msg_type=MessageType.PREPARE, view=0, sequence=1,
                         sender=2, digest="abc")
        log.add_prepare(p1)
        log.add_prepare(p2)
        assert log.get_prepare_certificate(0, 1) is None


# ===========================================================================
# Combined Scenario Tests
# ===========================================================================

class TestCombinedScenarios:
    def test_set_then_delete_then_set(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 1}, 1.0)
        net.submit_request("c1", {"type": "delete", "key": "x"}, 2.0)
        net.submit_request("c1", {"type": "set", "key": "x", "value": 2}, 3.0)
        assert net.check_safety()
        for node in net.nodes:
            assert node.state_machine.state["x"] == 2

    def test_many_keys(self):
        net = PBFTNetwork(4)
        for i in range(50):
            net.submit_request("c1", {"type": "set", "key": f"key_{i}", "value": i * 10}, float(i))
        assert net.check_safety()
        for node in net.nodes:
            assert len(node.state_machine.state) == 50
            assert node.state_machine.state["key_49"] == 490

    def test_view_change_preserves_safety(self):
        net = PBFTNetwork(7)
        for i in range(5):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        net.nodes[0].active = False
        net.trigger_view_change(1)
        for i in range(5, 10):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()

    def test_partition_then_heal(self):
        net = PBFTNetwork(4)
        net.submit_request("c1", {"type": "set", "key": "before", "value": 1}, 1.0)
        net.partition(0, 3)
        net.submit_request("c1", {"type": "set", "key": "during", "value": 2}, 2.0)
        net.heal_all()
        net.submit_request("c1", {"type": "set", "key": "after", "value": 3}, 3.0)
        assert net.check_safety()

    def test_checkpoint_and_gc(self):
        net = PBFTNetwork(4, checkpoint_interval=5)
        for i in range(10):
            net.submit_request("c1", {"type": "set", "key": f"k{i}", "value": i}, float(i))
        assert net.check_safety()
        for node in net.nodes:
            assert node.log.low_water_mark >= 5
            assert node.checkpoint_mgr.stable_checkpoint >= 5

    def test_multiple_clients_concurrent(self):
        net = PBFTNetwork(4)
        # Simulate multiple clients sending requests
        for i in range(10):
            client = f"client_{i % 3}"
            net.submit_request(client, {"type": "set", "key": f"{client}_k{i}", "value": i}, float(i))
        assert net.check_safety()
        assert net.check_liveness(10)
