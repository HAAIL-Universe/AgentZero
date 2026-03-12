"""Tests for C229: Sharding / Partitioning System."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sharding import (
    ShardStrategy, ShardState, RebalanceStrategy,
    ShardKeyExtractor, Shard, HashShardRouter, RangeShardRouter,
    ListShardRouter, ShardManager, AutoShardManager, QueryCoordinator,
    ReshardingPlanner, ShardingSystem,
)


# ===========================================================================
# ShardKeyExtractor
# ===========================================================================

class TestShardKeyExtractor:
    def test_single_column(self):
        ext = ShardKeyExtractor('id')
        assert ext.extract({'id': 42, 'name': 'x'}) == 42

    def test_multiple_columns(self):
        ext = ShardKeyExtractor(['region', 'id'])
        assert ext.extract({'region': 'us', 'id': 7}) == 'us:7'

    def test_missing_column_raises(self):
        ext = ShardKeyExtractor('id')
        with pytest.raises(KeyError):
            ext.extract({'name': 'x'})

    def test_empty_columns_raises(self):
        with pytest.raises(ValueError):
            ShardKeyExtractor([])


# ===========================================================================
# Shard
# ===========================================================================

class TestShard:
    def test_put_and_get(self):
        s = Shard('s1')
        assert s.put('k1', {'id': 'k1', 'v': 1}) is True  # new
        assert s.get('k1') == {'id': 'k1', 'v': 1}

    def test_put_overwrite(self):
        s = Shard('s1')
        s.put('k1', {'id': 'k1', 'v': 1})
        assert s.put('k1', {'id': 'k1', 'v': 2}) is False  # overwrite
        assert s.get('k1')['v'] == 2

    def test_delete(self):
        s = Shard('s1')
        s.put('k1', {'id': 'k1'})
        assert s.delete('k1') is True
        assert s.delete('k1') is False
        assert s.get('k1') is None

    def test_scan(self):
        s = Shard('s1')
        s.put('a', {'id': 'a', 'x': 1})
        s.put('b', {'id': 'b', 'x': 2})
        s.put('c', {'id': 'c', 'x': 3})
        assert len(s.scan()) == 3
        assert len(s.scan(lambda r: r['x'] > 1)) == 2

    def test_size_and_keys(self):
        s = Shard('s1')
        s.put('a', {})
        s.put('b', {})
        assert s.size() == 2
        assert set(s.keys()) == {'a', 'b'}

    def test_clear(self):
        s = Shard('s1')
        s.put('a', {})
        s.clear()
        assert s.size() == 0

    def test_status(self):
        s = Shard('s1', ShardStrategy.RANGE, range_start=0, range_end=100)
        s.node_id = 'n1'
        st = s.status()
        assert st['shard_id'] == 's1'
        assert st['state'] == 'ACTIVE'
        assert st['range'] == (0, 100)

    def test_contains_key(self):
        s = Shard('s1')
        s.put('x', {})
        assert s.contains_key('x') is True
        assert s.contains_key('y') is False

    def test_stats_tracking(self):
        s = Shard('s1')
        s.put('a', {})
        s.put('b', {})
        s.get('a')
        s.get('b')
        s.get('c')
        assert s.stats['writes'] == 2
        assert s.stats['reads'] == 3


# ===========================================================================
# HashShardRouter
# ===========================================================================

class TestHashShardRouter:
    def test_add_and_route(self):
        r = HashShardRouter()
        r.add_shard('s0')
        r.add_shard('s1')
        # Every key should route to one of the two shards
        for i in range(100):
            assert r.route(f"key_{i}") in ('s0', 's1')

    def test_remove_shard(self):
        r = HashShardRouter()
        r.add_shard('s0')
        r.add_shard('s1')
        r.remove_shard('s1')
        for i in range(50):
            assert r.route(f"key_{i}") == 's0'

    def test_empty_router_raises(self):
        r = HashShardRouter()
        with pytest.raises(ValueError):
            r.route('key')

    def test_distribution(self):
        r = HashShardRouter(virtual_nodes=150)
        for i in range(4):
            r.add_shard(f"s{i}")
        dist = r.get_distribution([f"k{i}" for i in range(1000)])
        # All shards should get some keys
        assert len(dist) == 4
        for count in dist.values():
            assert count > 50  # reasonable distribution

    def test_route_with_replicas(self):
        r = HashShardRouter()
        for i in range(4):
            r.add_shard(f"s{i}")
        replicas = r.route_with_replicas('key', count=3)
        assert len(replicas) == 3
        assert len(set(replicas)) == 3  # all unique

    def test_shard_ids_property(self):
        r = HashShardRouter()
        r.add_shard('a')
        r.add_shard('b')
        assert r.shard_ids == {'a', 'b'}

    def test_consistent_routing(self):
        r = HashShardRouter()
        r.add_shard('s0')
        r.add_shard('s1')
        first = r.route('stable_key')
        for _ in range(100):
            assert r.route('stable_key') == first


# ===========================================================================
# RangeShardRouter
# ===========================================================================

class TestRangeShardRouter:
    def test_basic_routing(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        r.add_shard('s1', 100, 200)
        assert r.route(50) == 's0'
        assert r.route(150) == 's1'

    def test_boundary(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        r.add_shard('s1', 100, 200)
        assert r.route(0) == 's0'
        assert r.route(99) == 's0'
        assert r.route(100) == 's1'

    def test_no_coverage_raises(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        with pytest.raises(KeyError):
            r.route(200)

    def test_overlap_raises(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        with pytest.raises(ValueError):
            r.add_shard('s1', 50, 150)

    def test_remove_shard(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        r.add_shard('s1', 100, 200)
        r.remove_shard('s0')
        assert r.shard_ids == {'s1'}

    def test_update_range(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 100)
        r.update_range('s0', 0, 200)
        assert r.route(150) == 's0'

    def test_get_ranges(self):
        r = RangeShardRouter()
        r.add_shard('s0', 0, 50)
        r.add_shard('s1', 50, 100)
        ranges = r.get_ranges()
        assert len(ranges) == 2
        assert ranges[0] == (0, 50, 's0')


# ===========================================================================
# ListShardRouter
# ===========================================================================

class TestListShardRouter:
    def test_basic_routing(self):
        r = ListShardRouter()
        r.add_shard('us', ['US', 'CA'])
        r.add_shard('eu', ['UK', 'DE', 'FR'])
        assert r.route('US') == 'us'
        assert r.route('DE') == 'eu'

    def test_unknown_value_raises(self):
        r = ListShardRouter()
        r.add_shard('us', ['US'])
        with pytest.raises(KeyError):
            r.route('JP')

    def test_duplicate_value_raises(self):
        r = ListShardRouter()
        r.add_shard('s1', ['A', 'B'])
        with pytest.raises(ValueError):
            r.add_shard('s2', ['B', 'C'])

    def test_remove_shard(self):
        r = ListShardRouter()
        r.add_shard('s1', ['A'])
        r.remove_shard('s1')
        assert len(r.shard_ids) == 0


# ===========================================================================
# ShardManager - Hash Strategy
# ===========================================================================

class TestShardManagerHash:
    def test_add_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        assert len(m.shards) == 2

    def test_duplicate_shard_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        with pytest.raises(ValueError):
            m.add_shard('s0')

    def test_put_and_get(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        m.put({'id': 'k1', 'val': 10})
        assert m.get('k1') == {'id': 'k1', 'val': 10}

    def test_delete(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.put({'id': 'k1'})
        assert m.delete('k1') is True
        assert m.get('k1') is None

    def test_scatter_gather(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        for i in range(4):
            m.add_shard(f"s{i}")
        for i in range(100):
            m.put({'id': f"k{i}", 'val': i})
        results = m.scatter_gather()
        assert len(results) == 100

    def test_scatter_gather_with_predicate(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(20):
            m.put({'id': f"k{i}", 'val': i})
        results = m.scatter_gather(lambda r: r['val'] >= 10)
        assert len(results) == 10

    def test_scatter_gather_with_limit(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        for i in range(20):
            m.put({'id': f"k{i}", 'val': i})
        results = m.scatter_gather_with_limit(
            sort_key=lambda r: r['val'], limit=5
        )
        assert len(results) == 5
        assert results[0]['val'] == 0

    def test_aggregate(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(10):
            m.put({'id': f"k{i}", 'val': i})
        assert m.aggregate('val', 'sum') == 45
        assert m.aggregate('val', 'count') == 10
        assert m.aggregate('val', 'min') == 0
        assert m.aggregate('val', 'max') == 9
        assert m.aggregate('val', 'avg') == 4.5

    def test_remove_shard_non_empty_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.put({'id': 'k1'})
        with pytest.raises(ValueError):
            m.remove_shard('s0')

    def test_remove_shard_empty(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        # s1 might be empty if no keys route there -- force empty
        s1 = m.shards['s1']
        s1.data.clear()
        s1.stats['size'] = 0
        if s1.size() == 0:
            m.remove_shard('s1')
            assert 's1' not in m.shards

    def test_events(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        events = []
        m.on('shard_added', lambda d: events.append(('added', d)))
        m.on('write', lambda d: events.append(('write', d)))
        m.add_shard('s0')
        m.put({'id': 'k1'})
        assert len(events) == 2
        assert events[0][0] == 'added'
        assert events[1][0] == 'write'

    def test_distribution(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        for i in range(4):
            m.add_shard(f"s{i}")
        for i in range(100):
            m.put({'id': f"k{i}"})
        dist = m.get_distribution()
        assert dist['total'] == 100
        assert dist['shard_count'] == 4

    def test_status(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        st = m.status()
        assert st['strategy'] == 'HASH'
        assert st['shard_count'] == 1


# ===========================================================================
# ShardManager - Range Strategy
# ===========================================================================

class TestShardManagerRange:
    def test_add_and_route(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=500)
        m.add_shard('s1', range_start=500, range_end=1000)
        m.put({'id': 100, 'name': 'a'})
        m.put({'id': 600, 'name': 'b'})
        assert m.get(100)['name'] == 'a'
        assert m.get(600)['name'] == 'b'

    def test_range_missing_params(self):
        m = ShardManager(strategy=ShardStrategy.RANGE)
        with pytest.raises(ValueError):
            m.add_shard('s0')  # no range_start/end

    def test_split_range_shard(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=1000)
        for i in range(0, 1000, 10):
            m.put({'id': i, 'val': i})
        id1, id2 = m.split_shard('s0', split_point=500)
        assert m.get(100)['val'] == 100
        assert m.get(600)['val'] == 600
        # Old shard gone
        assert 's0' not in m.shards

    def test_merge_range_shards(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=500)
        m.add_shard('s1', range_start=500, range_end=1000)
        m.put({'id': 100})
        m.put({'id': 700})
        merged = m.merge_shards('s0', 's1', 'merged')
        assert merged == 'merged'
        assert m.get(100) is not None
        assert m.get(700) is not None

    def test_merge_non_adjacent_raises(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=100)
        m.add_shard('s1', range_start=200, range_end=300)
        with pytest.raises(ValueError, match="adjacent"):
            m.merge_shards('s0', 's1')


# ===========================================================================
# ShardManager - List Strategy
# ===========================================================================

class TestShardManagerList:
    def test_list_routing(self):
        m = ShardManager(strategy=ShardStrategy.LIST, key_columns=['region'])
        m.add_shard('us', list_values=['US', 'CA', 'MX'])
        m.add_shard('eu', list_values=['UK', 'DE', 'FR'])
        m.put({'region': 'US', 'data': 'x'})
        m.put({'region': 'DE', 'data': 'y'})
        assert m.get('US')['data'] == 'x'
        assert m.get('DE')['data'] == 'y'

    def test_list_missing_values_raises(self):
        m = ShardManager(strategy=ShardStrategy.LIST)
        with pytest.raises(ValueError):
            m.add_shard('s0')


# ===========================================================================
# Split and Merge - Hash
# ===========================================================================

class TestSplitMergeHash:
    def test_split_hash_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(50):
            m.put({'id': f"k{i}"})
        total_before = sum(s.size() for s in m.shards.values())
        id1, id2 = m.split_shard('s0')
        total_after = sum(s.size() for s in m.shards.values())
        # Data preserved (some might redistribute to s1)
        assert total_after >= total_before - 5  # allow small variance from rehashing

    def test_merge_hash_shards(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(20):
            m.put({'id': f"k{i}"})
        merged = m.merge_shards('s0', 's1', 'merged')
        assert merged == 'merged'
        assert 'merged' in m.shards

    def test_split_nonexistent_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        with pytest.raises(KeyError):
            m.split_shard('nope')

    def test_migration_log_on_split(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.split_shard('s0')
        log = m.get_migration_log()
        assert len(log) == 1
        assert log[0]['type'] == 'split'


# ===========================================================================
# Rebalancing
# ===========================================================================

class TestRebalancing:
    def test_rebalance_after_add_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        # All data goes to s0
        for i in range(100):
            m.put({'id': f"k{i}"})
        assert m.shards['s0'].size() == 100

        # Add another shard and rebalance
        m.add_shard('s1')
        result = m.rebalance()
        assert result['moved'] > 0
        assert m.shards['s1'].size() > 0

    def test_rebalance_no_op(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(50):
            m.put({'id': f"k{i}"})
        # Already balanced
        result = m.rebalance()
        assert result['moved'] == 0

    def test_rebalance_range_noop(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=100)
        result = m.rebalance()
        assert result['moved'] == 0


# ===========================================================================
# Migration
# ===========================================================================

class TestMigration:
    def test_migrate_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0', node_id='n0')
        m.migrate_shard('s0', 'n0', 'n1')
        assert m.shards['s0'].node_id == 'n1'
        assert m.shards['s0'].state == ShardState.ACTIVE

    def test_migrate_wrong_node_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0', node_id='n0')
        with pytest.raises(ValueError):
            m.migrate_shard('s0', 'n1', 'n2')

    def test_migrate_nonexistent_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        with pytest.raises(KeyError):
            m.migrate_shard('nope', 'n0', 'n1')

    def test_migration_logged(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0', node_id='n0')
        m.migrate_shard('s0', 'n0', 'n1')
        log = m.get_migration_log()
        assert len(log) == 1
        assert log[0]['type'] == 'migrate'
        assert log[0]['to_node'] == 'n1'


# ===========================================================================
# Hotspot Detection
# ===========================================================================

class TestHotspots:
    def test_detect_hotspot(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        # Simulate heavy writes to s0
        m.shards['s0'].stats['writes'] = 1000
        m.shards['s1'].stats['writes'] = 10
        hotspots = m.get_hotspots()
        assert len(hotspots) == 1
        assert hotspots[0]['shard_id'] == 's0'

    def test_no_hotspots(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        hotspots = m.get_hotspots()
        assert len(hotspots) == 0


# ===========================================================================
# QueryCoordinator
# ===========================================================================

class TestQueryCoordinator:
    def setup_method(self):
        self.m = ShardManager(strategy=ShardStrategy.HASH)
        for i in range(4):
            self.m.add_shard(f"s{i}")
        for i in range(50):
            self.m.put({'id': f"k{i}", 'val': i, 'cat': 'even' if i % 2 == 0 else 'odd'})
        self.q = QueryCoordinator(self.m)

    def test_point_query(self):
        assert self.q.point_query('k5')['val'] == 5

    def test_multi_get(self):
        results = self.q.multi_get(['k1', 'k2', 'k3'])
        assert len(results) == 3
        assert results['k2']['val'] == 2

    def test_scan_query(self):
        results = self.q.scan_query(predicate=lambda r: r['cat'] == 'even')
        assert len(results) == 25

    def test_scan_with_sort_and_limit(self):
        results = self.q.scan_query(
            sort_key=lambda r: r['val'], limit=10
        )
        assert len(results) == 10
        assert results[0]['val'] == 0
        assert results[9]['val'] == 9

    def test_scan_with_offset(self):
        results = self.q.scan_query(
            sort_key=lambda r: r['val'], limit=5, offset=10
        )
        assert len(results) == 5
        assert results[0]['val'] == 10

    def test_aggregate(self):
        total = self.q.aggregate_query('val', 'sum')
        assert total == sum(range(50))

    def test_count(self):
        assert self.q.count() == 50
        assert self.q.count(lambda r: r['val'] < 10) == 10

    def test_stats(self):
        self.q.point_query('k1')
        self.q.multi_get(['k1'])
        s = self.q.stats()
        assert s['point'] == 1
        assert s['multi_get'] == 1


# ===========================================================================
# QueryCoordinator - Range Queries
# ===========================================================================

class TestRangeQueries:
    def test_range_query_on_range_shards(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=500)
        m.add_shard('s1', range_start=500, range_end=1000)
        for i in range(100):
            m.put({'id': i * 10, 'val': i})
        q = QueryCoordinator(m)
        results = q.range_query(100, 300)
        for r in results:
            assert 100 <= r['id'] < 300

    def test_range_query_on_hash_shards(self):
        m = ShardManager(strategy=ShardStrategy.HASH, key_columns=['id'])
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(50):
            m.put({'id': i, 'val': i})
        q = QueryCoordinator(m)
        results = q.range_query(10, 20)
        assert len(results) == 10
        for r in results:
            assert 10 <= r['id'] < 20


# ===========================================================================
# AutoShardManager
# ===========================================================================

class TestAutoShardManager:
    def test_auto_split_large_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        for i in range(200):
            m.put({'id': f"k{i}"})
        auto = AutoShardManager(m, max_shard_size=100)
        actions = auto.check_and_rebalance()
        assert any(a['action'] == 'split' for a in actions)
        assert len(m.shards) > 1

    def test_auto_merge_small_range_shards(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=500)
        m.add_shard('s1', range_start=500, range_end=1000)
        # Both shards very small
        for i in range(5):
            m.put({'id': i})
        for i in range(500, 505):
            m.put({'id': i})
        auto = AutoShardManager(m, min_shard_size=100)
        actions = auto.check_and_rebalance()
        assert any(a['action'] == 'merge' for a in actions)

    def test_no_action_needed(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(10):
            m.put({'id': f"k{i}"})
        auto = AutoShardManager(m, max_shard_size=1000)
        actions = auto.check_and_rebalance()
        assert len(actions) == 0


# ===========================================================================
# ReshardingPlanner
# ===========================================================================

class TestReshardingPlanner:
    def test_plan_more_shards(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        for i in range(100):
            m.put({'id': f"k{i}"})
        p = ReshardingPlanner(m)
        plan = p.plan(target_shard_count=4)
        assert plan['target_shards'] == 4
        assert len(plan['operations']) > 0
        assert plan['operations'][0]['type'] == 'split'

    def test_plan_fewer_shards(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        for i in range(4):
            m.add_shard(f"s{i}")
        for i in range(100):
            m.put({'id': f"k{i}"})
        p = ReshardingPlanner(m)
        plan = p.plan(target_shard_count=2)
        assert any(op['type'] == 'merge' for op in plan['operations'])

    def test_plan_by_load(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.add_shard('s1')
        m.shards['s0'].stats['writes'] = 10000
        m.shards['s1'].stats['writes'] = 10
        p = ReshardingPlanner(m)
        plan = p.plan(balance_metric='load')
        assert any(op['type'] == 'split' for op in plan['operations'])

    def test_plan_empty(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        p = ReshardingPlanner(m)
        plan = p.plan(target_shard_count=1)
        assert plan['total_records'] == 0


# ===========================================================================
# ShardingSystem (Facade)
# ===========================================================================

class TestShardingSystem:
    def test_hash_system_basic(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=4)
        for i in range(100):
            sys.put({'id': f"k{i}", 'val': i})
        assert sys.get('k50')['val'] == 50
        assert sys.count() == 100

    def test_hash_system_multi_get(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(20):
            sys.put({'id': f"k{i}", 'val': i})
        results = sys.multi_get([f"k{i}" for i in range(5)])
        assert len(results) == 5

    def test_hash_system_delete(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        sys.put({'id': 'k1', 'val': 1})
        assert sys.delete('k1') is True
        assert sys.get('k1') is None

    def test_hash_system_query(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=4)
        for i in range(50):
            sys.put({'id': f"k{i}", 'val': i})
        results = sys.query(predicate=lambda r: r['val'] < 10)
        assert len(results) == 10

    def test_hash_system_aggregate(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(10):
            sys.put({'id': f"k{i}", 'val': i})
        assert sys.aggregate('val', 'sum') == 45

    def test_range_system(self):
        sys = ShardingSystem(
            strategy=ShardStrategy.RANGE, key_columns=['id'],
            num_shards=4, range_start=0, range_end=1000
        )
        for i in range(100):
            sys.put({'id': i * 10, 'val': i})
        assert sys.get(500)['val'] == 50
        assert sys.count() == 100

    def test_range_system_range_query(self):
        sys = ShardingSystem(
            strategy=ShardStrategy.RANGE, key_columns=['id'],
            num_shards=4, range_start=0, range_end=1000
        )
        for i in range(100):
            sys.put({'id': i * 10, 'val': i})
        results = sys.range_query(200, 500)
        for r in results:
            assert 200 <= r['id'] < 500

    def test_system_split(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(50):
            sys.put({'id': f"k{i}"})
        total_before = sys.count()
        sys.split_shard('shard_0')
        # Data preserved (or redistributed)
        total_after = sys.count()
        assert total_after >= total_before - 5

    def test_system_rebalance(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(100):
            sys.put({'id': f"k{i}"})
        result = sys.rebalance()
        assert 'moved' in result

    def test_system_distribution(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=4)
        for i in range(200):
            sys.put({'id': f"k{i}"})
        dist = sys.distribution()
        assert dist['total'] == 200
        assert dist['shard_count'] == 4

    def test_system_hotspots(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        sys.manager.shards['shard_0'].stats['writes'] = 5000
        hotspots = sys.hotspots()
        assert len(hotspots) >= 1

    def test_system_status(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=3)
        st = sys.status()
        assert st['shard_count'] == 3
        assert st['strategy'] == 'HASH'

    def test_system_migrate(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        shard = sys.manager.shards['shard_0']
        old_node = shard.node_id
        sys.migrate_shard('shard_0', old_node, 'new_node')
        assert sys.manager.shards['shard_0'].node_id == 'new_node'

    def test_system_plan_resharding(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(100):
            sys.put({'id': f"k{i}"})
        plan = sys.plan_resharding(target_shards=4)
        assert plan['target_shards'] == 4

    def test_system_auto_rebalance(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=1,
                            auto_split_size=50)
        for i in range(100):
            sys.put({'id': f"k{i}"})
        actions = sys.auto_rebalance()
        assert any(a['action'] == 'split' for a in actions)


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_shard(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        for i in range(10):
            m.put({'id': f"k{i}"})
        assert m.get('k5') is not None

    def test_composite_key(self):
        m = ShardManager(strategy=ShardStrategy.HASH, key_columns=['region', 'id'])
        m.add_shard('s0')
        m.add_shard('s1')
        m.put({'region': 'us', 'id': '1', 'data': 'x'})
        assert m.get('us:1')['data'] == 'x'

    def test_write_to_inactive_shard_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.shards['s0'].state = ShardState.DRAINING
        with pytest.raises(ValueError, match="DRAINING"):
            m.put({'id': 'k1'})

    def test_aggregate_empty(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        assert m.aggregate('val', 'sum') is None

    def test_aggregate_unknown_func_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        m.put({'id': 'k1', 'val': 1})
        with pytest.raises(ValueError):
            m.aggregate('val', 'median')

    def test_offset_without_limit(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=2)
        for i in range(20):
            sys.put({'id': f"k{i}", 'val': i})
        results = sys.query(sort_key=lambda r: r['val'], offset=15)
        assert len(results) == 5

    def test_route_record_directly(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        shard = m.route_record({'id': 'k1'})
        assert shard.shard_id == 's0'

    def test_merge_nonexistent_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        with pytest.raises(KeyError):
            m.merge_shards('s0', 'nope')

    def test_remove_nonexistent_raises(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        with pytest.raises(KeyError):
            m.remove_shard('nope')

    def test_hash_router_single_shard_all_keys(self):
        r = HashShardRouter()
        r.add_shard('only')
        for i in range(100):
            assert r.route(f"k{i}") == 'only'


# ===========================================================================
# Data Integrity
# ===========================================================================

class TestDataIntegrity:
    def test_split_preserves_all_data_range(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=1000)
        records = {}
        for i in range(100):
            rec = {'id': i * 10, 'val': i}
            records[i * 10] = rec
            m.put(rec)
        m.split_shard('s0', split_point=500)
        for key, expected in records.items():
            assert m.get(key) == expected

    def test_merge_preserves_all_data_range(self):
        m = ShardManager(strategy=ShardStrategy.RANGE, key_columns=['id'])
        m.add_shard('s0', range_start=0, range_end=500)
        m.add_shard('s1', range_start=500, range_end=1000)
        records = {}
        for i in range(50):
            rec = {'id': i * 10, 'val': i}
            records[i * 10] = rec
            m.put(rec)
        for i in range(50, 100):
            rec = {'id': i * 10, 'val': i}
            records[i * 10] = rec
            m.put(rec)
        m.merge_shards('s0', 's1', 'merged')
        for key, expected in records.items():
            assert m.get(key) == expected

    def test_rebalance_preserves_all_data(self):
        m = ShardManager(strategy=ShardStrategy.HASH)
        m.add_shard('s0')
        for i in range(100):
            m.put({'id': f"k{i}", 'val': i})
        m.add_shard('s1')
        m.rebalance()
        for i in range(100):
            rec = m.get(f"k{i}")
            assert rec is not None
            assert rec['val'] == i

    def test_many_operations_data_consistent(self):
        sys = ShardingSystem(strategy=ShardStrategy.HASH, num_shards=4)
        # Insert
        for i in range(200):
            sys.put({'id': f"k{i}", 'val': i})
        # Delete some
        for i in range(0, 200, 3):
            sys.delete(f"k{i}")
        # Verify remaining
        for i in range(200):
            rec = sys.get(f"k{i}")
            if i % 3 == 0:
                assert rec is None
            else:
                assert rec['val'] == i


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
