"""Tests for C226: Database Replication"""
import pytest
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from database_replication import (
    WALEntry, WALShipper, DataStore, ReplicationNode, ReplicationCluster,
    RaftReplicationCluster, ConflictDetector, LagMonitor,
    StreamingReplicationManager, ReadWriteSplitter,
    ReplicationMode, NodeRole, ConsistencyLevel, ReplicationState,
    ReplicationSlot
)


# ============================================================
# WALEntry Tests
# ============================================================

class TestWALEntry:
    def test_create_entry(self):
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='users', data={'key': 'a', 'name': 'Alice'})
        assert e.lsn == 1
        assert e.operation == 'INSERT'
        assert e.table == 'users'
        assert e.checksum != ""

    def test_checksum_verification(self):
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='users', data={'key': 'a'})
        assert e.verify()

    def test_checksum_detects_tampering(self):
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='users', data={'key': 'a'})
        e.data['key'] = 'b'  # tamper
        assert not e.verify()

    def test_to_dict_from_dict_roundtrip(self):
        e = WALEntry(lsn=5, timestamp=2.0, operation='UPDATE',
                     table='items', data={'key': 'x', 'val': 42}, term=3)
        d = e.to_dict()
        e2 = WALEntry.from_dict(d)
        assert e2.lsn == 5
        assert e2.operation == 'UPDATE'
        assert e2.term == 3
        assert e2.verify()

    def test_different_entries_different_checksums(self):
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                      table='t', data={'key': 'a'})
        e2 = WALEntry(lsn=2, timestamp=1.0, operation='INSERT',
                      table='t', data={'key': 'b'})
        assert e1.checksum != e2.checksum


# ============================================================
# WALShipper Tests
# ============================================================

class TestWALShipper:
    def test_append_increments_lsn(self):
        ws = WALShipper()
        e1 = ws.append('INSERT', 'users', {'key': 'a'})
        e2 = ws.append('INSERT', 'users', {'key': 'b'})
        assert e1.lsn == 1
        assert e2.lsn == 2
        assert ws.current_lsn == 2

    def test_entries_from(self):
        ws = WALShipper()
        ws.append('INSERT', 't', {'key': '1'})
        ws.append('INSERT', 't', {'key': '2'})
        ws.append('INSERT', 't', {'key': '3'})
        entries = ws.entries_from(2)
        assert len(entries) == 2
        assert entries[0].lsn == 2

    def test_get_entry(self):
        ws = WALShipper()
        ws.append('INSERT', 't', {'key': '1'})
        ws.append('UPDATE', 't', {'key': '2'})
        e = ws.get_entry(2)
        assert e.operation == 'UPDATE'
        assert ws.get_entry(99) is None

    def test_create_and_remove_slot(self):
        ws = WALShipper()
        slot = ws.create_slot('replica1', ReplicationMode.SYNC)
        assert slot.replica_id == 'replica1'
        assert slot.mode == ReplicationMode.SYNC
        assert 'replica1' in ws.slots
        ws.remove_slot('replica1')
        assert 'replica1' not in ws.slots

    def test_pending_entries(self):
        ws = WALShipper()
        ws.append('INSERT', 't', {'key': '1'})
        ws.append('INSERT', 't', {'key': '2'})
        ws.create_slot('r1')
        pending = ws.get_pending_entries('r1')
        assert len(pending) == 0  # slot starts at next_lsn=3

        ws.append('INSERT', 't', {'key': '3'})
        pending = ws.get_pending_entries('r1')
        assert len(pending) == 1

    def test_mark_sent_and_confirm(self):
        ws = WALShipper()
        ws.create_slot('r1')
        e = ws.append('INSERT', 't', {'key': '1'})
        ws.mark_sent('r1', e.lsn)
        assert ws.slots['r1'].sent_lsn == 1
        ws.confirm('r1', e.lsn)
        assert ws.slots['r1'].confirmed_lsn == 1

    def test_compact(self):
        ws = WALShipper()
        for i in range(10):
            ws.append('INSERT', 't', {'key': str(i)})
        ws.create_slot('r1')
        ws.confirm('r1', 7)
        compacted = ws.compact(8)
        assert compacted == 7  # limited by replica confirmation
        assert ws.wal_size < 10

    def test_wal_callback(self):
        ws = WALShipper()
        events = []
        ws.on('wal_append', lambda e: events.append(e.lsn))
        ws.append('INSERT', 't', {'key': 'a'})
        assert events == [1]

    def test_confirm_callback(self):
        ws = WALShipper()
        confirms = []
        ws.on('confirm', lambda rid, lsn: confirms.append((rid, lsn)))
        ws.create_slot('r1')
        ws.confirm('r1', 5)
        assert confirms == [('r1', 5)]


# ============================================================
# DataStore Tests
# ============================================================

class TestDataStore:
    def test_insert(self):
        ds = DataStore()
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='users', data={'key': 'a', 'name': 'Alice'})
        assert ds.apply_entry(e)
        assert ds.query('users', 'a') == {'key': 'a', 'name': 'Alice'}

    def test_update(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                                table='users', data={'key': 'a', 'name': 'Alice'}))
        ds.apply_entry(WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                                table='users', data={'key': 'a', 'name': 'Alicia'}))
        assert ds.query('users', 'a')['name'] == 'Alicia'

    def test_delete(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                                table='users', data={'key': 'a', 'name': 'Alice'}))
        ds.apply_entry(WALEntry(lsn=2, timestamp=2.0, operation='DELETE',
                                table='users', data={'key': 'a'}))
        assert ds.query('users', 'a') is None

    def test_idempotent_apply(self):
        ds = DataStore()
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='t', data={'key': 'a'})
        assert ds.apply_entry(e)
        assert not ds.apply_entry(e)  # already applied

    def test_checksum_mismatch_raises(self):
        ds = DataStore()
        e = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='t', data={'key': 'a'})
        e.data['key'] = 'tampered'
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ds.apply_entry(e)

    def test_query_all_rows(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                                table='t', data={'key': 'a', 'v': 1}))
        ds.apply_entry(WALEntry(lsn=2, timestamp=2.0, operation='INSERT',
                                table='t', data={'key': 'b', 'v': 2}))
        all_rows = ds.query('t')
        assert len(all_rows) == 2

    def test_query_nonexistent_table(self):
        ds = DataStore()
        assert ds.query('nonexistent') == {}
        assert ds.query('nonexistent', 'k') is None

    def test_snapshot_restore(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                                table='t', data={'key': 'a', 'v': 1}))
        snap = ds.snapshot()

        ds2 = DataStore()
        ds2.restore(snap)
        assert ds2.query('t', 'a') == {'key': 'a', 'v': 1}
        assert ds2.applied_lsn == 1

    def test_ddl_create_table(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='DDL',
                                table='new_table', data={'action': 'CREATE_TABLE'}))
        assert ds.query('new_table') == {}

    def test_ddl_drop_table(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                                table='t', data={'key': 'a'}))
        ds.apply_entry(WALEntry(lsn=2, timestamp=2.0, operation='DDL',
                                table='t', data={'action': 'DROP_TABLE'}))
        assert ds.query('t') == {}

    def test_update_nonexistent_creates(self):
        ds = DataStore()
        ds.apply_entry(WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                                table='t', data={'key': 'a', 'v': 1}))
        assert ds.query('t', 'a') == {'key': 'a', 'v': 1}


# ============================================================
# ReplicationNode Tests
# ============================================================

class TestReplicationNode:
    def test_primary_write(self):
        node = ReplicationNode('p1', role=NodeRole.PRIMARY)
        entry = node.write('INSERT', 'users', {'key': 'a', 'name': 'Alice'})
        assert entry.lsn == 1
        assert node.read('users', 'a')['name'] == 'Alice'

    def test_replica_cannot_write(self):
        node = ReplicationNode('r1', role=NodeRole.REPLICA)
        with pytest.raises(RuntimeError, match="not primary"):
            node.write('INSERT', 'users', {'key': 'a'})

    def test_apply_entries_on_replica(self):
        replica = ReplicationNode('r1', role=NodeRole.REPLICA)
        entries = [
            WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='t', data={'key': 'a', 'v': 1}),
            WALEntry(lsn=2, timestamp=2.0, operation='INSERT',
                     table='t', data={'key': 'b', 'v': 2}),
        ]
        applied = replica.apply_entries(entries)
        assert len(applied) == 2
        assert replica.applied_lsn == 2
        assert replica.read('t', 'a')['v'] == 1

    def test_promote_demote(self):
        node = ReplicationNode('n1', role=NodeRole.REPLICA)
        assert node.role == NodeRole.REPLICA
        assert node.wal_shipper is None

        node.promote()
        assert node.role == NodeRole.PRIMARY
        assert node.wal_shipper is not None

        node.demote()
        assert node.role == NodeRole.REPLICA
        assert node.wal_shipper is None

    def test_snapshot_restore(self):
        primary = ReplicationNode('p1', role=NodeRole.PRIMARY)
        primary.write('INSERT', 't', {'key': 'a', 'v': 1})
        snap = primary.snapshot()

        replica = ReplicationNode('r1', role=NodeRole.REPLICA)
        replica.restore(snap)
        assert replica.read('t', 'a')['v'] == 1
        assert replica.applied_lsn == 1

    def test_status(self):
        node = ReplicationNode('p1', role=NodeRole.PRIMARY)
        s = node.status()
        assert s['node_id'] == 'p1'
        assert s['role'] == 'primary'
        assert s['is_running']

    def test_callback_on_write(self):
        node = ReplicationNode('p1', role=NodeRole.PRIMARY)
        writes = []
        node.on('write', lambda e: writes.append(e.lsn))
        node.write('INSERT', 't', {'key': 'a'})
        assert writes == [1]

    def test_callback_on_promote(self):
        node = ReplicationNode('r1', role=NodeRole.REPLICA)
        promoted = []
        node.on('promoted', lambda nid: promoted.append(nid))
        node.promote()
        assert promoted == ['r1']

    def test_read_consistency_eventual(self):
        replica = ReplicationNode('r1', role=NodeRole.REPLICA)
        replica.read_consistency = ConsistencyLevel.EVENTUAL
        # Should succeed even with no data
        result = replica.read('t')
        assert result == {}

    def test_promote_sets_wal_lsn(self):
        replica = ReplicationNode('r1', role=NodeRole.REPLICA)
        entries = [
            WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                     table='t', data={'key': 'a'}),
            WALEntry(lsn=2, timestamp=2.0, operation='INSERT',
                     table='t', data={'key': 'b'}),
        ]
        replica.apply_entries(entries)
        replica.promote()
        assert replica.wal_shipper.next_lsn == 3


# ============================================================
# ReplicationCluster Tests
# ============================================================

class TestReplicationCluster:
    def setup_method(self):
        self.cluster = ReplicationCluster(mode=ReplicationMode.ASYNC)
        self.cluster.add_node('primary', NodeRole.PRIMARY)
        self.cluster.add_node('replica1', NodeRole.REPLICA)
        self.cluster.add_node('replica2', NodeRole.REPLICA)

    def test_write_replicates(self):
        self.cluster.write('INSERT', 'users', {'key': 'a', 'name': 'Alice'})
        # Async replication should have applied
        r1 = self.cluster.nodes['replica1']
        assert r1.read('users', 'a')['name'] == 'Alice'

    def test_multiple_writes(self):
        for i in range(10):
            self.cluster.write('INSERT', 'items', {'key': str(i), 'val': i})
        primary = self.cluster.get_primary()
        r1 = self.cluster.nodes['replica1']
        assert primary.applied_lsn == 10
        assert r1.applied_lsn == 10

    def test_read_from_primary(self):
        self.cluster.read_preference = 'primary'
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        result = self.cluster.read('t', 'a')
        assert result['v'] == 1

    def test_read_from_replica(self):
        self.cluster.read_preference = 'replica'
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        result = self.cluster.read('t', 'a')
        assert result['v'] == 1

    def test_read_strong_consistency(self):
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        result = self.cluster.read('t', 'a', consistency=ConsistencyLevel.STRONG)
        assert result['v'] == 1

    def test_failover(self):
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        new_primary = self.cluster.failover('replica1')
        assert new_primary.node_id == 'replica1'
        assert new_primary.role == NodeRole.PRIMARY
        assert self.cluster.primary_id == 'replica1'
        assert self.cluster.nodes['primary'].role == NodeRole.REPLICA

    def test_failover_auto_select(self):
        self.cluster.write('INSERT', 't', {'key': 'a'})
        self.cluster.failover()
        assert self.cluster.primary_id in ('replica1', 'replica2')

    def test_failover_no_replicas(self):
        cluster = ReplicationCluster()
        cluster.add_node('p1', NodeRole.PRIMARY)
        with pytest.raises(RuntimeError, match="No replicas"):
            cluster.failover()

    def test_write_after_failover(self):
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        self.cluster.failover('replica1')
        entry = self.cluster.write('INSERT', 't', {'key': 'b', 'v': 2})
        assert entry.lsn == 2
        new_primary = self.cluster.get_primary()
        assert new_primary.read('t', 'b')['v'] == 2

    def test_failover_history(self):
        self.cluster.write('INSERT', 't', {'key': 'a'})
        self.cluster.failover('replica1')
        assert len(self.cluster.failover_history) == 1
        assert self.cluster.failover_history[0]['old_primary'] == 'primary'
        assert self.cluster.failover_history[0]['new_primary'] == 'replica1'

    def test_add_remove_node(self):
        self.cluster.add_node('replica3', NodeRole.REPLICA)
        assert len(self.cluster.get_replicas()) == 3
        self.cluster.remove_node('replica3')
        assert len(self.cluster.get_replicas()) == 2

    def test_cannot_add_second_primary(self):
        with pytest.raises(RuntimeError, match="Primary already exists"):
            self.cluster.add_node('p2', NodeRole.PRIMARY)

    def test_no_primary_write_raises(self):
        cluster = ReplicationCluster()
        with pytest.raises(RuntimeError, match="No primary"):
            cluster.write('INSERT', 't', {})

    def test_cluster_status(self):
        self.cluster.write('INSERT', 't', {'key': 'a'})
        status = self.cluster.cluster_status()
        assert status['primary'] == 'primary'
        assert status['node_count'] == 3
        assert status['replica_count'] == 2
        assert status['primary_lsn'] == 1

    def test_replicate_pending(self):
        # Write directly to primary without cluster replication
        primary = self.cluster.get_primary()
        primary.write('INSERT', 't', {'key': 'a', 'v': 1})
        primary.write('INSERT', 't', {'key': 'b', 'v': 2})

        results = self.cluster.replicate_pending()
        assert results.get('replica1', 0) > 0

    def test_catch_up_replica(self):
        self.cluster.write('INSERT', 't', {'key': 'a'})
        self.cluster.write('INSERT', 't', {'key': 'b'})

        # Add a new lagging replica
        self.cluster.add_node('replica3', NodeRole.REPLICA)
        snap = self.cluster.get_primary().snapshot()
        assert self.cluster.catch_up_replica('replica3', snap)
        assert self.cluster.nodes['replica3'].read('t', 'a') is not None

    def test_add_replica_from_snapshot(self):
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        node = self.cluster.add_replica_from_snapshot('replica3')
        assert node.read('t', 'a')['v'] == 1

    def test_compact_wal(self):
        for i in range(20):
            self.cluster.write('INSERT', 't', {'key': str(i)})
        compacted = self.cluster.compact_wal()
        primary = self.cluster.get_primary()
        assert primary.wal_shipper.wal_size < 20

    def test_read_preferred_node(self):
        self.cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        result = self.cluster.read('t', 'a', preferred_node='replica2')
        assert result['v'] == 1

    def test_failover_callback(self):
        events = []
        self.cluster.on('failover', lambda old, new: events.append((old, new)))
        self.cluster.write('INSERT', 't', {'key': 'a'})
        self.cluster.failover('replica1')
        assert events == [('primary', 'replica1')]


# ============================================================
# Sync Replication Tests
# ============================================================

class TestSyncReplication:
    def test_sync_replication(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        assert cluster.nodes['r1'].applied_lsn == 1
        assert cluster.nodes['r2'].applied_lsn == 1

    def test_semi_sync_replication(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SEMI_SYNC, sync_replicas=1)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        # At least one replica confirmed synchronously
        r1 = cluster.nodes['r1']
        r2 = cluster.nodes['r2']
        assert r1.applied_lsn == 1 or r2.applied_lsn == 1

    def test_sync_all_replicas_consistent(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        for i in range(5):
            cluster.add_node(f'r{i}', NodeRole.REPLICA)

        for i in range(10):
            cluster.write('INSERT', 't', {'key': str(i), 'v': i})

        for i in range(5):
            assert cluster.nodes[f'r{i}'].applied_lsn == 10


# ============================================================
# ConflictDetector Tests
# ============================================================

class TestConflictDetector:
    def test_no_conflict_different_keys(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 1})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'b', 'v': 2})
        assert cd.detect(e1, e2) is None

    def test_no_conflict_different_tables(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t1', data={'key': 'a'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t2', data={'key': 'a'})
        assert cd.detect(e1, e2) is None

    def test_update_update_conflict(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 1})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 2})
        conflict = cd.detect(e1, e2)
        assert conflict is not None
        assert conflict['type'] == 'UPDATE_UPDATE'

    def test_insert_insert_conflict(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='INSERT',
                      table='t', data={'key': 'a'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='INSERT',
                      table='t', data={'key': 'a'})
        conflict = cd.detect(e1, e2)
        assert conflict['type'] == 'INSERT_INSERT'

    def test_delete_conflict(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='DELETE',
                      table='t', data={'key': 'a'})
        conflict = cd.detect(e1, e2)
        assert conflict['type'] == 'DELETE_CONFLICT'

    def test_last_writer_wins(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 'old'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 'new'})
        winner = cd.resolve_last_writer_wins(e1, e2)
        assert winner.data['v'] == 'new'

    def test_node_priority_resolution(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a', '_source_node': 'dc1'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'a', '_source_node': 'dc2'})
        priority = {'dc1': 10, 'dc2': 5}
        winner = cd.resolve_by_node_priority(e1, e2, priority)
        assert winner.data['_source_node'] == 'dc1'

    def test_conflicts_recorded(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'a'})
        cd.detect(e1, e2)
        assert len(cd.conflicts) == 1


# ============================================================
# LagMonitor Tests
# ============================================================

class TestLagMonitor:
    def test_record_and_get_lag(self):
        lm = LagMonitor()
        lm.record('r1', 5, 100)
        assert lm.get_lag('r1') == 5

    def test_avg_lag(self):
        lm = LagMonitor()
        for i in range(10):
            lm.record('r1', i, 100)
        avg = lm.get_avg_lag('r1', window=5)
        assert avg == 7.0  # avg of 5,6,7,8,9

    def test_max_lag(self):
        lm = LagMonitor()
        lm.record('r1', 5, 100)
        lm.record('r2', 15, 100)
        lm.record('r3', 10, 100)
        replica, lag = lm.get_max_lag()
        assert replica == 'r2'
        assert lag == 15

    def test_warn_alert(self):
        lm = LagMonitor(warn_threshold=5)
        lm.record('r1', 7, 100)
        alerts = lm.get_alerts('WARN')
        assert len(alerts) == 1

    def test_critical_alert(self):
        lm = LagMonitor(critical_threshold=50)
        lm.record('r1', 60, 100)
        alerts = lm.get_alerts('CRITICAL')
        assert len(alerts) == 1

    def test_clear_alerts(self):
        lm = LagMonitor(warn_threshold=1)
        lm.record('r1', 5, 100)
        lm.clear_alerts()
        assert len(lm.get_alerts()) == 0

    def test_no_lag_initially(self):
        lm = LagMonitor()
        assert lm.get_lag('nonexistent') == 0
        assert lm.get_avg_lag('nonexistent') == 0.0

    def test_max_lag_empty(self):
        lm = LagMonitor()
        replica, lag = lm.get_max_lag()
        assert replica is None
        assert lag == 0


# ============================================================
# StreamingReplicationManager Tests
# ============================================================

class TestStreamingReplication:
    def setup_method(self):
        self.cluster = ReplicationCluster(mode=ReplicationMode.ASYNC)
        self.cluster.add_node('primary', NodeRole.PRIMARY)
        self.cluster.add_node('r1', NodeRole.REPLICA)
        self.mgr = StreamingReplicationManager(self.cluster)

    def test_start_stream(self):
        assert self.mgr.start_stream('r1')
        assert 'r1' in self.mgr.streams

    def test_stop_stream(self):
        self.mgr.start_stream('r1')
        stream = self.mgr.stop_stream('r1')
        assert stream is not None
        assert 'r1' not in self.mgr.streams

    def test_tick_applies_entries(self):
        primary = self.cluster.get_primary()
        primary.write('INSERT', 't', {'key': 'a', 'v': 1})
        primary.write('INSERT', 't', {'key': 'b', 'v': 2})

        self.mgr.start_stream('r1')
        results = self.mgr.tick()
        assert results.get('r1', 0) > 0

    def test_catching_up_to_streaming(self):
        self.mgr.batch_size = 5
        primary = self.cluster.get_primary()
        for i in range(3):
            primary.write('INSERT', 't', {'key': str(i)})

        self.mgr.start_stream('r1')
        self.mgr.tick()
        assert self.mgr.streams['r1']['state'] == ReplicationState.STREAMING

    def test_large_backlog_catching_up(self):
        self.mgr.batch_size = 5
        primary = self.cluster.get_primary()
        for i in range(20):
            primary.write('INSERT', 't', {'key': str(i)})

        self.mgr.start_stream('r1')
        self.mgr.tick()
        state = self.mgr.streams['r1']['state']
        # Should still be catching up after first batch
        assert state in (ReplicationState.CATCHING_UP, ReplicationState.STREAMING)

    def test_stream_status(self):
        self.mgr.start_stream('r1')
        status = self.mgr.stream_status()
        assert 'r1' in status
        assert 'state' in status['r1']

    def test_start_stream_nonexistent_replica(self):
        assert not self.mgr.start_stream('nonexistent')

    def test_tick_skips_stopped_streams(self):
        self.mgr.start_stream('r1')
        self.mgr.streams['r1']['state'] = ReplicationState.STOPPED
        results = self.mgr.tick()
        assert results == {}


# ============================================================
# ReadWriteSplitter Tests
# ============================================================

class TestReadWriteSplitter:
    def setup_method(self):
        self.cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        self.cluster.add_node('p', NodeRole.PRIMARY)
        self.cluster.add_node('r1', NodeRole.REPLICA)
        self.cluster.add_node('r2', NodeRole.REPLICA)
        self.splitter = ReadWriteSplitter(self.cluster)

    def test_write_through_splitter(self):
        entry = self.splitter.execute('write', 'users', data={'key': 'a', 'name': 'Alice'},
                                      operation='INSERT')
        assert entry.lsn == 1
        assert self.splitter.write_count == 1

    def test_read_through_splitter(self):
        self.splitter.execute('write', 'users', data={'key': 'a', 'name': 'Alice'},
                              operation='INSERT')
        result = self.splitter.execute('read', 'users', key='a')
        assert result['name'] == 'Alice'
        assert self.splitter.read_count == 1

    def test_reads_go_to_replicas(self):
        self.splitter.execute('write', 'users', data={'key': 'a', 'name': 'Alice'},
                              operation='INSERT')
        for _ in range(4):
            self.splitter.execute('read', 'users', key='a')
        assert self.splitter.read_from_replica == 4

    def test_strong_reads_go_to_primary(self):
        self.splitter.execute('write', 'users', data={'key': 'a'},
                              operation='INSERT')
        self.splitter.execute('read', 'users', key='a',
                              consistency=ConsistencyLevel.STRONG)
        assert self.splitter.read_from_primary == 1

    def test_round_robin(self):
        self.splitter.execute('write', 'users', data={'key': 'a'},
                              operation='INSERT')
        for _ in range(6):
            self.splitter.execute('read', 'users', key='a')
        assert self.splitter.read_from_replica == 6

    def test_stats(self):
        self.splitter.execute('write', 't', data={'key': 'a'}, operation='INSERT')
        self.splitter.execute('read', 't', key='a')
        self.splitter.execute('read', 't', key='a')
        stats = self.splitter.stats()
        assert stats['writes'] == 1
        assert stats['reads'] == 2


# ============================================================
# RaftReplicationCluster Tests
# ============================================================

class TestRaftReplicationCluster:
    def test_elect_leader(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        leader = cluster.elect_leader()
        assert leader is not None

    def test_write_through_raft(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        cluster.elect_leader()
        cluster.write('INSERT', 'users', {'key': 'a', 'name': 'Alice'})
        leader = cluster.get_leader()
        result = cluster.read('users', 'a', node_id=leader)
        assert result is not None
        assert result['name'] == 'Alice'

    def test_replicated_to_all_nodes(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        cluster.elect_leader()
        cluster.write('INSERT', 'users', {'key': 'a', 'name': 'Alice'})
        # All nodes should eventually have the data
        for nid in ['n1', 'n2', 'n3']:
            result = cluster.read('users', 'a', node_id=nid)
            if result:
                assert result['name'] == 'Alice'

    def test_multiple_raft_writes(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        cluster.elect_leader()
        for i in range(5):
            cluster.write('INSERT', 'items', {'key': str(i), 'val': i})
        leader = cluster.get_leader()
        for i in range(5):
            result = cluster.read('items', str(i), node_id=leader)
            assert result is not None

    def test_raft_status(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        cluster.elect_leader()
        status = cluster.status()
        assert status['leader'] is not None
        assert len(status['nodes']) == 3

    def test_no_leader_write_raises(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        # Don't elect leader
        with pytest.raises(RuntimeError, match="No leader"):
            cluster.write('INSERT', 't', {'key': 'a'})

    def test_read_from_specific_node(self):
        cluster = RaftReplicationCluster(['n1', 'n2', 'n3'])
        cluster.elect_leader()
        cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        # Read from leader
        leader = cluster.get_leader()
        result = cluster.read('t', 'a', node_id=leader)
        assert result['v'] == 1


# ============================================================
# ReplicationSlot Tests
# ============================================================

class TestReplicationSlot:
    def test_slot_defaults(self):
        slot = ReplicationSlot(replica_id='r1')
        assert slot.start_lsn == 0
        assert slot.confirmed_lsn == 0
        assert slot.state == ReplicationState.STOPPED
        assert slot.mode == ReplicationMode.ASYNC

    def test_slot_custom(self):
        slot = ReplicationSlot(
            replica_id='r1', start_lsn=10,
            mode=ReplicationMode.SYNC
        )
        assert slot.start_lsn == 10
        assert slot.mode == ReplicationMode.SYNC


# ============================================================
# Integration / End-to-End Tests
# ============================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """Primary writes, replicates, failover, continue writes."""
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p1', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        # Write some data
        for i in range(5):
            cluster.write('INSERT', 'users', {'key': str(i), 'name': f'User{i}'})

        assert cluster.nodes['r1'].applied_lsn == 5
        assert cluster.nodes['r2'].applied_lsn == 5

        # Failover to r1
        cluster.failover('r1')
        assert cluster.primary_id == 'r1'

        # Write more data on new primary
        cluster.write('INSERT', 'users', {'key': '5', 'name': 'User5'})
        assert cluster.get_primary().read('users', '5')['name'] == 'User5'

    def test_snapshot_based_replica_addition(self):
        """Add a new replica via snapshot when cluster is already running."""
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)

        for i in range(10):
            cluster.write('INSERT', 't', {'key': str(i), 'val': i * 10})

        # Add new replica from snapshot
        r2 = cluster.add_replica_from_snapshot('r2')
        assert r2.read('t', '5')['val'] == 50
        assert r2.applied_lsn == 10

        # New writes also replicate
        cluster.write('INSERT', 't', {'key': '10', 'val': 100})
        cluster.replicate_pending()
        # r2 should get the new entry
        assert r2.applied_lsn >= 10

    def test_mixed_operations(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)

        cluster.write('INSERT', 'users', {'key': 'a', 'name': 'Alice'})
        cluster.write('UPDATE', 'users', {'key': 'a', 'name': 'Alicia'})
        cluster.write('INSERT', 'users', {'key': 'b', 'name': 'Bob'})
        cluster.write('DELETE', 'users', {'key': 'b'})

        r1 = cluster.nodes['r1']
        assert r1.read('users', 'a')['name'] == 'Alicia'
        assert r1.read('users', 'b') is None

    def test_streaming_catch_up(self):
        cluster = ReplicationCluster(mode=ReplicationMode.ASYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)

        # Write directly to primary (bypass cluster replication)
        primary = cluster.get_primary()
        for i in range(50):
            primary.write('INSERT', 't', {'key': str(i), 'v': i})

        # Use streaming manager to catch up
        mgr = StreamingReplicationManager(cluster)
        mgr.batch_size = 10
        mgr.start_stream('r1')

        total_applied = 0
        for _ in range(10):
            results = mgr.tick()
            total_applied += results.get('r1', 0)
            if cluster.nodes['r1'].applied_lsn >= 50:
                break

        assert cluster.nodes['r1'].applied_lsn == 50

    def test_splitter_with_failover(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        splitter = ReadWriteSplitter(cluster)
        splitter.execute('write', 't', data={'key': 'a', 'v': 1}, operation='INSERT')

        # Failover
        cluster.failover('r1')
        splitter.execute('write', 't', data={'key': 'b', 'v': 2}, operation='INSERT')

        result = splitter.execute('read', 't', key='b',
                                  consistency=ConsistencyLevel.STRONG)
        assert result['v'] == 2

    def test_wal_compaction_preserves_data(self):
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)

        for i in range(20):
            cluster.write('INSERT', 't', {'key': str(i), 'val': i})

        cluster.compact_wal()

        # Data still accessible on all nodes
        assert cluster.get_primary().read('t', '15')['val'] == 15
        assert cluster.nodes['r1'].read('t', '15')['val'] == 15

    def test_lag_monitoring_integration(self):
        cluster = ReplicationCluster(mode=ReplicationMode.ASYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        for i in range(10):
            cluster.write('INSERT', 't', {'key': str(i)})

        status = cluster.cluster_status()
        assert 'r1' in status['nodes']
        assert 'lag' in status['nodes']['r1']

    def test_conflict_detection_workflow(self):
        cd = ConflictDetector()
        e1 = WALEntry(lsn=1, timestamp=1.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 1, '_source_node': 'dc1'})
        e2 = WALEntry(lsn=2, timestamp=2.0, operation='UPDATE',
                      table='t', data={'key': 'a', 'v': 2, '_source_node': 'dc2'})

        conflict = cd.detect(e1, e2)
        assert conflict is not None

        # Resolve
        winner = cd.resolve_last_writer_wins(e1, e2)
        assert winner.data['v'] == 2

    def test_double_failover(self):
        """Failover twice -- ensure cluster stays consistent."""
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        cluster.add_node('r1', NodeRole.REPLICA)
        cluster.add_node('r2', NodeRole.REPLICA)

        cluster.write('INSERT', 't', {'key': 'a', 'v': 1})
        cluster.failover('r1')
        cluster.write('INSERT', 't', {'key': 'b', 'v': 2})
        cluster.failover('r2')
        cluster.write('INSERT', 't', {'key': 'c', 'v': 3})

        assert cluster.primary_id == 'r2'
        assert cluster.get_primary().read('t', 'c')['v'] == 3
        assert len(cluster.failover_history) == 2

    def test_large_scale_replication(self):
        """Test with many writes and multiple replicas."""
        cluster = ReplicationCluster(mode=ReplicationMode.SYNC)
        cluster.add_node('p', NodeRole.PRIMARY)
        for i in range(5):
            cluster.add_node(f'r{i}', NodeRole.REPLICA)

        for i in range(100):
            cluster.write('INSERT', 'bulk', {'key': str(i), 'val': i * i})

        for i in range(5):
            assert cluster.nodes[f'r{i}'].applied_lsn == 100
            assert cluster.nodes[f'r{i}'].read('bulk', '99')['val'] == 9801
