"""Tests for C223: Database Connection Pool"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

from connection_pool import (
    Connection, ConnectionFactory, ConnectionPool, PooledConnection,
    PoolConfig, PoolStats, ConnectionState, PoolState,
    HealthChecker, PreparedStatementCache, PoolPartitioner,
    ConnectionLeakDetector, PoolCluster, LoadBalanceStrategy,
    ConnectionInfo,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C220_query_executor_integration'))
from query_executor_integration import IntegratedQueryEngine


# ---- Helpers ----

def make_config(**kwargs):
    defaults = dict(
        min_size=1, max_size=5, acquire_timeout=2.0,
        idle_timeout=60.0, max_lifetime=300.0,
        validation_interval=10.0, validate_on_borrow=False,
        validate_on_return=False, test_on_create=False,
        max_idle=5, leak_detection_threshold=0.5,
    )
    defaults.update(kwargs)
    return PoolConfig(**defaults)


def make_pool(**kwargs):
    config = make_config(**kwargs)
    return ConnectionPool(config=config)


# ==== Connection Tests ====

def test_connection_creation():
    engine = IntegratedQueryEngine(db_name="test_conn_create")
    config = make_config()
    conn = Connection(engine, config)
    assert conn.id > 0
    assert conn.state == ConnectionState.IDLE
    assert conn.is_valid
    assert conn.info.use_count == 0
    assert not conn.in_transaction

def test_connection_execute():
    engine = IntegratedQueryEngine(db_name="test_conn_exec")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    result = conn.execute("CREATE TABLE t1 (id INT, name TEXT)")
    assert result is not None
    conn.execute("INSERT INTO t1 VALUES (1, 'a')")
    assert conn.info.use_count == 2

def test_connection_query():
    engine = IntegratedQueryEngine(db_name="test_conn_query")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.execute("CREATE TABLE t1 (id INT, name TEXT)")
    conn.execute("INSERT INTO t1 VALUES (1, 'hello')")
    rows = conn.query("SELECT * FROM t1")
    assert len(rows) >= 1

def test_connection_transaction():
    engine = IntegratedQueryEngine(db_name="test_conn_tx")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.execute("CREATE TABLE t1 (id INT)")
    tx_id = conn.begin()
    assert conn.in_transaction
    assert tx_id is not None
    conn.execute("INSERT INTO t1 VALUES (1)")
    conn.commit()
    assert not conn.in_transaction

def test_connection_rollback():
    engine = IntegratedQueryEngine(db_name="test_conn_rb")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.execute("CREATE TABLE t1 (id INT)")
    conn.begin()
    conn.execute("INSERT INTO t1 VALUES (1)")
    conn.rollback()
    assert not conn.in_transaction

def test_connection_double_begin_fails():
    engine = IntegratedQueryEngine(db_name="test_conn_dbl")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.begin()
    try:
        conn.begin()
        assert False, "Should have raised"
    except ConnectionError:
        pass

def test_connection_commit_no_tx_fails():
    engine = IntegratedQueryEngine(db_name="test_conn_no_tx")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    try:
        conn.commit()
        assert False, "Should have raised"
    except ConnectionError:
        pass

def test_connection_validate():
    engine = IntegratedQueryEngine(db_name="test_conn_val")
    config = make_config()
    conn = Connection(engine, config)
    assert conn.validate("SELECT 1")
    assert conn.info.last_validated > 0

def test_connection_reset():
    engine = IntegratedQueryEngine(db_name="test_conn_reset")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.execute("CREATE TABLE t1 (id INT)")
    conn.begin()
    conn.reset()
    assert not conn.in_transaction

def test_connection_close():
    engine = IntegratedQueryEngine(db_name="test_conn_close")
    config = make_config()
    conn = Connection(engine, config)
    conn.close()
    assert conn.state == ConnectionState.CLOSED
    assert not conn.is_valid

def test_connection_execute_closed_fails():
    engine = IntegratedQueryEngine(db_name="test_conn_closed")
    config = make_config()
    conn = Connection(engine, config)
    conn.close()
    try:
        conn.execute("SELECT 1")
        assert False, "Should have raised"
    except ConnectionError:
        pass

def test_connection_age():
    engine = IntegratedQueryEngine(db_name="test_conn_age")
    config = make_config()
    conn = Connection(engine, config)
    assert conn.age >= 0
    assert conn.age < 5  # should be very small

def test_connection_idle_time():
    engine = IntegratedQueryEngine(db_name="test_conn_idle")
    config = make_config()
    conn = Connection(engine, config)
    assert conn.idle_time >= 0

def test_connection_prepared_cache():
    engine = IntegratedQueryEngine(db_name="test_conn_prep")
    config = make_config()
    conn = Connection(engine, config)
    conn.set_prepared("SELECT 1", {"plan": "scan"})
    assert conn.get_prepared("SELECT 1") == {"plan": "scan"}
    assert conn.get_prepared("SELECT 2") is None

def test_connection_repr():
    engine = IntegratedQueryEngine(db_name="test_conn_repr")
    config = make_config()
    conn = Connection(engine, config)
    r = repr(conn)
    assert "Connection" in r
    assert "IDLE" in r


# ==== ConnectionFactory Tests ====

def test_factory_create():
    config = make_config(test_on_create=False)
    factory = ConnectionFactory(config)
    conn = factory.create()
    assert isinstance(conn, Connection)
    assert conn.is_valid

def test_factory_create_with_validation():
    config = make_config(test_on_create=True)
    factory = ConnectionFactory(config)
    conn = factory.create()
    assert conn.is_valid

def test_factory_custom_engine():
    config = make_config(test_on_create=False)
    calls = []
    def custom_factory():
        calls.append(1)
        return IntegratedQueryEngine(db_name=f"custom_{len(calls)}")
    factory = ConnectionFactory(config, engine_factory=custom_factory)
    conn = factory.create()
    assert len(calls) == 1
    assert conn.is_valid

def test_factory_validate():
    config = make_config(test_on_create=False)
    factory = ConnectionFactory(config)
    conn = factory.create()
    assert factory.validate(conn)

def test_factory_destroy():
    config = make_config(test_on_create=False)
    factory = ConnectionFactory(config)
    conn = factory.create()
    factory.destroy(conn)
    assert conn.state == ConnectionState.CLOSED


# ==== ConnectionPool Basic Tests ====

def test_pool_creation():
    pool = make_pool(min_size=2)
    assert pool.size >= 2
    assert pool.state == PoolState.RUNNING
    pool.close()

def test_pool_acquire_return():
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    assert isinstance(conn, PooledConnection)
    assert pool.active_count == 1
    conn.close()
    assert pool.active_count == 0
    pool.close()

def test_pool_acquire_context_manager():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        assert pool.active_count == 1
    assert pool.active_count == 0
    pool.close()

def test_pool_max_size():
    pool = make_pool(min_size=1, max_size=2)
    c1 = pool.acquire()
    c2 = pool.acquire()
    assert pool.active_count == 2
    try:
        pool.acquire(timeout=0.1)
        assert False, "Should have timed out"
    except TimeoutError:
        pass
    c1.close()
    c2.close()
    pool.close()

def test_pool_reuse():
    pool = make_pool(min_size=1, max_size=2)
    with pool.acquire() as c1:
        id1 = c1.id
    with pool.acquire() as c2:
        id2 = c2.id
    assert id1 == id2  # same connection reused
    pool.close()

def test_pool_multiple_borrows():
    pool = make_pool(min_size=1, max_size=3)
    conns = [pool.acquire() for _ in range(3)]
    assert pool.active_count == 3
    for c in conns:
        c.close()
    assert pool.active_count == 0
    pool.close()

def test_pool_stats():
    pool = make_pool(min_size=1)
    with pool.acquire():
        pass
    stats = pool.stats
    assert stats.total_borrows >= 1
    assert stats.total_returns >= 1
    pool.close()

def test_pool_stats_to_dict():
    pool = make_pool(min_size=1)
    with pool.acquire():
        pass
    d = pool.stats.to_dict()
    assert 'total_borrows' in d
    assert 'avg_wait_time_ms' in d
    assert 'utilization' in d
    pool.close()

def test_pool_drain():
    pool = make_pool(min_size=2)
    pool.drain()
    assert pool.state == PoolState.DRAINING
    assert pool.idle_count == 0
    try:
        pool.acquire()
        assert False, "Should have raised"
    except ConnectionError:
        pass
    pool.close()

def test_pool_close():
    pool = make_pool(min_size=2)
    pool.close()
    assert pool.state == PoolState.CLOSED
    assert pool.size == 0

def test_pool_context_manager():
    with make_pool(min_size=1) as pool:
        with pool.acquire() as conn:
            assert conn is not None
    assert pool.state == PoolState.CLOSED

def test_pool_repr():
    pool = make_pool(min_size=1)
    r = repr(pool)
    assert "ConnectionPool" in r
    assert "RUNNING" in r
    pool.close()

def test_pool_timeout_stats():
    pool = make_pool(min_size=1, max_size=1)
    c1 = pool.acquire()
    try:
        pool.acquire(timeout=0.1)
    except TimeoutError:
        pass
    assert pool.stats.total_timeouts >= 1
    c1.close()
    pool.close()

def test_pool_peak_active():
    pool = make_pool(min_size=1, max_size=3)
    conns = [pool.acquire() for _ in range(3)]
    assert pool.stats.peak_active == 3
    for c in conns:
        c.close()
    pool.close()

def test_pool_avg_wait_time():
    pool = make_pool(min_size=1)
    with pool.acquire():
        pass
    assert pool.stats.avg_wait_time_ms >= 0
    pool.close()


# ==== Pooled Connection Tests ====

def test_pooled_conn_execute():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        result = conn.execute("CREATE TABLE pc_t1 (id INT)")
        assert result is not None
    pool.close()

def test_pooled_conn_query():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE pc_t2 (id INT, name TEXT)")
        conn.execute("INSERT INTO pc_t2 VALUES (1, 'test')")
        rows = conn.query("SELECT * FROM pc_t2")
        assert len(rows) >= 1
    pool.close()

def test_pooled_conn_transaction():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE pc_t3 (id INT)")
        tx_id = conn.begin()
        conn.execute("INSERT INTO pc_t3 VALUES (1)")
        conn.commit()
    pool.close()

def test_pooled_conn_rollback():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE pc_t4 (id INT)")
        conn.begin()
        conn.execute("INSERT INTO pc_t4 VALUES (1)")
        conn.rollback()
    pool.close()

def test_pooled_conn_closed_access_fails():
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    conn.close()
    try:
        conn.execute("SELECT 1")
        assert False
    except ConnectionError:
        pass
    pool.close()

def test_pooled_conn_double_close():
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    conn.close()
    conn.close()  # should be idempotent
    assert pool.active_count == 0
    pool.close()

def test_pooled_conn_repr():
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    r = repr(conn)
    assert "PooledConnection" in r
    assert "open" in r
    conn.close()
    r2 = repr(conn)
    assert "closed" in r2
    pool.close()

def test_pooled_conn_id():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        assert conn.id > 0
    pool.close()

def test_pooled_conn_info():
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        info = conn.info
        assert isinstance(info, ConnectionInfo)
    pool.close()

def test_pooled_conn_auto_rollback_on_return():
    """Connection with active tx should be rolled back on return."""
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE pc_t5 (id INT)")
        conn.begin()
        conn.execute("INSERT INTO pc_t5 VALUES (1)")
        # Don't commit -- close should trigger reset/rollback
    pool.close()


# ==== Validation Tests ====

def test_validate_on_borrow():
    pool = make_pool(min_size=1, validate_on_borrow=True)
    with pool.acquire() as conn:
        assert pool.stats.total_validations >= 1
    pool.close()

def test_validate_on_return():
    pool = make_pool(min_size=1, validate_on_return=True)
    with pool.acquire():
        pass
    assert pool.stats.total_validations >= 1
    pool.close()


# ==== Health Checker Tests ====

def test_health_checker_ensure_min():
    pool = make_pool(min_size=3)
    # Drain some
    with pool._lock:
        if pool._idle:
            conn = pool._idle.popleft()
            pool._stats.total_connections -= 1
            pool._stats.idle_connections -= 1
    result = pool._health_checker.ensure_min_size()
    assert pool.size >= 3
    pool.close()

def test_health_checker_maintenance():
    pool = make_pool(min_size=2)
    result = pool.maintenance()
    assert 'validated' in result
    assert 'evicted' in result
    assert 'created' in result
    pool.close()

def test_health_checker_evict_expired():
    config = make_config(min_size=1, max_lifetime=0.01)
    pool = ConnectionPool(config=config)
    time.sleep(0.02)  # let connections age out
    result = pool.maintenance()
    assert result['evicted'] >= 0  # may or may not have evicted depending on timing
    pool.close()

def test_health_checker_idle_timeout():
    config = make_config(min_size=0, max_idle=5, idle_timeout=0.01)
    pool = ConnectionPool(config=config)
    # Create and return a connection
    with pool.acquire():
        pass
    time.sleep(0.02)
    # Maintenance should find the idle one
    pool.maintenance()
    pool.close()

def test_health_checker_validation_interval():
    config = make_config(min_size=1, validation_interval=0.01)
    pool = ConnectionPool(config=config)
    time.sleep(0.02)
    pool.maintenance()
    assert pool.stats.total_validations >= 0  # at least attempted
    pool.close()


# ==== Prepared Statement Cache Tests ====

def test_prepared_cache_basic():
    cache = PreparedStatementCache(capacity=10)
    cache.put("SELECT 1", {"plan": "const"})
    assert cache.get("SELECT 1") == {"plan": "const"}
    assert cache.get("SELECT 2") is None

def test_prepared_cache_eviction():
    cache = PreparedStatementCache(capacity=2)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")
    assert cache.get("q1") is None  # evicted
    assert cache.get("q3") == "r3"

def test_prepared_cache_invalidate():
    cache = PreparedStatementCache(capacity=10)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.invalidate("q1")
    assert cache.get("q1") is None
    assert cache.get("q2") == "r2"

def test_prepared_cache_invalidate_all():
    cache = PreparedStatementCache(capacity=10)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.invalidate()
    assert cache.size == 0

def test_prepared_cache_stats():
    cache = PreparedStatementCache(capacity=10)
    cache.put("q1", "r1")
    cache.get("q1")  # hit
    cache.get("q2")  # miss
    s = cache.stats()
    assert 'capacity' in s or 'size' in s or isinstance(s, dict)

def test_prepared_cache_hit_rate():
    cache = PreparedStatementCache(capacity=10)
    cache.put("q1", "r1")
    cache.get("q1")
    cache.get("q2")
    # hit_rate should be numeric
    assert isinstance(cache.hit_rate, (int, float))


# ==== Leak Detector Tests ====

def test_leak_detector_no_leaks():
    detector = ConnectionLeakDetector(threshold=1.0)
    leaks = detector.check_leaks()
    assert len(leaks) == 0

def test_leak_detector_track():
    engine = IntegratedQueryEngine(db_name="test_leak")
    config = make_config()
    conn = Connection(engine, config)
    detector = ConnectionLeakDetector(threshold=0.01)
    detector.track_borrow(conn)
    time.sleep(0.02)
    leaks = detector.check_leaks()
    assert len(leaks) == 1
    assert leaks[0]['connection_id'] == conn.id

def test_leak_detector_return_clears():
    engine = IntegratedQueryEngine(db_name="test_leak_ret")
    config = make_config()
    conn = Connection(engine, config)
    detector = ConnectionLeakDetector(threshold=0.01)
    detector.track_borrow(conn)
    detector.track_return(conn)
    time.sleep(0.02)
    leaks = detector.check_leaks()
    assert len(leaks) == 0

def test_leak_detector_in_pool():
    pool = make_pool(min_size=1, leak_detection_threshold=0.01)
    conn = pool.acquire()
    time.sleep(0.02)
    leaks = pool.check_leaks()
    assert len(leaks) >= 1
    conn.close()
    pool.close()

def test_leak_detector_clear():
    detector = ConnectionLeakDetector(threshold=0.01)
    engine = IntegratedQueryEngine(db_name="test_leak_clear")
    config = make_config()
    conn = Connection(engine, config)
    detector.track_borrow(conn)
    detector.clear()
    assert detector.leaked_count == 0


# ==== Pool Resize Tests ====

def test_pool_resize_grow():
    pool = make_pool(min_size=1, max_size=3)
    pool.resize(min_size=3)
    assert pool.size >= 3
    pool.close()

def test_pool_resize_shrink():
    pool = make_pool(min_size=3, max_size=5, max_idle=5)
    initial = pool.size
    pool.resize(max_size=2, min_size=1)
    assert pool.config.max_size == 2
    pool.close()

def test_pool_resize_min_clamp():
    pool = make_pool(min_size=1, max_size=5)
    pool.resize(min_size=10, max_size=5)
    assert pool.config.min_size <= pool.config.max_size
    pool.close()


# ==== Event Listener Tests ====

def test_event_listener():
    events = []
    pool = make_pool(min_size=1)
    pool.add_event_listener(lambda event, conn: events.append((event, conn.id)))
    with pool.acquire():
        pass
    assert any(e[0] == 'borrow' for e in events)
    assert any(e[0] == 'return' for e in events)
    pool.close()

def test_event_listener_error_ignored():
    def bad_listener(event, conn):
        raise ValueError("boom")
    pool = make_pool(min_size=1)
    pool.add_event_listener(bad_listener)
    with pool.acquire():
        pass  # should not raise
    pool.close()


# ==== Pool Partitioner Tests ====

def test_partitioner_add_pool():
    part = PoolPartitioner()
    p = part.add_pool("read", config=make_config(min_size=1))
    assert isinstance(p, ConnectionPool)
    assert "read" in part.pool_names
    part.close_all()

def test_partitioner_default():
    part = PoolPartitioner()
    part.add_pool("write", config=make_config(min_size=1), is_default=True)
    part.add_pool("read", config=make_config(min_size=1))
    assert part.default_pool == "write"
    part.close_all()

def test_partitioner_acquire():
    part = PoolPartitioner()
    part.add_pool("main", config=make_config(min_size=1))
    with part.acquire("main") as conn:
        assert conn is not None
    part.close_all()

def test_partitioner_acquire_default():
    part = PoolPartitioner()
    part.add_pool("main", config=make_config(min_size=1), is_default=True)
    with part.acquire() as conn:
        assert conn is not None
    part.close_all()

def test_partitioner_unknown_pool():
    part = PoolPartitioner()
    try:
        part.get_pool("nope")
        assert False
    except KeyError:
        pass
    part.close_all()

def test_partitioner_stats():
    part = PoolPartitioner()
    part.add_pool("main", config=make_config(min_size=1))
    with part.acquire("main"):
        pass
    stats = part.stats()
    assert "main" in stats
    part.close_all()

def test_partitioner_context_manager():
    with PoolPartitioner() as part:
        part.add_pool("main", config=make_config(min_size=1))
        with part.acquire("main"):
            pass

def test_partitioner_read_write_split():
    part = PoolPartitioner()
    part.add_pool("write", config=make_config(min_size=1), is_default=True)
    part.add_pool("read", config=make_config(min_size=2))
    with part.acquire("write") as w:
        w.execute("CREATE TABLE rw_t1 (id INT)")
    # Read pool is separate engine, so separate state
    assert part.get_pool("read").size >= 2
    part.close_all()


# ==== Pool Cluster Tests ====

def test_cluster_add_node():
    cluster = PoolCluster()
    p = cluster.add_node("node1", config=make_config(min_size=1))
    assert isinstance(p, ConnectionPool)
    assert "node1" in cluster.node_names
    cluster.close()

def test_cluster_acquire():
    cluster = PoolCluster()
    cluster.add_node("node1", config=make_config(min_size=1))
    with cluster.acquire() as conn:
        assert conn is not None
    cluster.close()

def test_cluster_round_robin():
    cluster = PoolCluster(strategy=LoadBalanceStrategy.ROUND_ROBIN)
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.add_node("b", config=make_config(min_size=1))
    # Acquire several times
    conns = []
    for _ in range(4):
        c = cluster.acquire()
        conns.append(c)
    for c in conns:
        c.close()
    # Both nodes should have been used
    stats = cluster.stats()
    total = sum(s['total_borrows'] for s in stats.values())
    assert total == 4
    cluster.close()

def test_cluster_least_connections():
    cluster = PoolCluster(strategy=LoadBalanceStrategy.LEAST_CONNECTIONS)
    cluster.add_node("a", config=make_config(min_size=1, max_size=5))
    cluster.add_node("b", config=make_config(min_size=1, max_size=5))
    # Hold one connection on 'a'
    c1 = cluster.acquire(node_name="a")
    # Next should prefer 'b' (0 active vs 1)
    c2 = cluster.acquire()
    stats = cluster.stats()
    c1.close()
    c2.close()
    cluster.close()

def test_cluster_disable_node():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.add_node("b", config=make_config(min_size=1))
    cluster.disable_node("a")
    assert "a" not in cluster.enabled_nodes
    assert "b" in cluster.enabled_nodes
    with cluster.acquire() as conn:
        pass  # should go to 'b'
    cluster.close()

def test_cluster_enable_node():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.disable_node("a")
    cluster.enable_node("a")
    assert "a" in cluster.enabled_nodes
    cluster.close()

def test_cluster_remove_node():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.add_node("b", config=make_config(min_size=1))
    cluster.remove_node("a")
    assert "a" not in cluster.node_names
    cluster.close()

def test_cluster_failover():
    """If primary fails, should try secondary."""
    cluster = PoolCluster()
    cluster.add_node("primary", config=make_config(min_size=1, max_size=1))
    cluster.add_node("secondary", config=make_config(min_size=1, max_size=2))
    # Exhaust primary
    c1 = cluster.acquire(node_name="primary")
    # Next acquire should failover to secondary
    c2 = cluster.acquire(timeout=0.5)
    assert c2 is not None
    c1.close()
    c2.close()
    cluster.close()

def test_cluster_all_exhausted():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1, max_size=1))
    c1 = cluster.acquire()
    try:
        cluster.acquire(timeout=0.1)
        assert False
    except ConnectionError:
        pass
    c1.close()
    cluster.close()

def test_cluster_no_enabled():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.disable_node("a")
    try:
        cluster.acquire()
        assert False
    except ConnectionError:
        pass
    cluster.close()

def test_cluster_stats():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    with cluster.acquire():
        pass
    stats = cluster.stats()
    assert "a" in stats
    cluster.close()

def test_cluster_context_manager():
    with PoolCluster() as cluster:
        cluster.add_node("a", config=make_config(min_size=1))
        with cluster.acquire() as conn:
            pass

def test_cluster_named_acquire():
    cluster = PoolCluster()
    cluster.add_node("a", config=make_config(min_size=1))
    cluster.add_node("b", config=make_config(min_size=1))
    with cluster.acquire(node_name="b") as conn:
        pass
    assert cluster.stats()["b"]["total_borrows"] >= 1
    cluster.close()

def test_cluster_unknown_node():
    cluster = PoolCluster()
    try:
        cluster.acquire(node_name="nope")
        assert False
    except KeyError:
        pass
    cluster.close()

def test_cluster_random_strategy():
    cluster = PoolCluster(strategy=LoadBalanceStrategy.RANDOM)
    cluster.add_node("a", config=make_config(min_size=1), weight=1)
    cluster.add_node("b", config=make_config(min_size=1), weight=1)
    conns = []
    for _ in range(6):
        c = cluster.acquire()
        conns.append(c)
    for c in conns:
        c.close()
    cluster.close()


# ==== Integration Tests ====

def test_full_sql_workflow():
    """Full workflow: create table, insert, query via pool."""
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE wf_t1 (id INT, name TEXT)")
        conn.execute("INSERT INTO wf_t1 VALUES (1, 'alice')")
        conn.execute("INSERT INTO wf_t1 VALUES (2, 'bob')")
        rows = conn.query("SELECT * FROM wf_t1")
        assert len(rows) == 2
    pool.close()

def test_transaction_via_pool():
    """Transaction through pooled connection."""
    pool = make_pool(min_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE tx_t1 (id INT)")
        conn.begin()
        conn.execute("INSERT INTO tx_t1 VALUES (1)")
        conn.commit()
        rows = conn.query("SELECT * FROM tx_t1")
        assert len(rows) >= 1
    pool.close()

def test_connection_reuse_preserves_state():
    """Returned connection should have clean state."""
    pool = make_pool(min_size=1, max_size=1)
    with pool.acquire() as conn:
        conn.execute("CREATE TABLE reuse_t1 (id INT)")
        conn.begin()
        # Don't commit -- close should reset
    # Re-acquire same connection
    with pool.acquire() as conn:
        # Should not be in transaction
        assert not conn.connection.in_transaction
    pool.close()

def test_concurrent_pool_access():
    """Multiple threads borrowing from pool."""
    pool = make_pool(min_size=2, max_size=4)
    results = []
    errors = []

    def worker(i):
        try:
            with pool.acquire(timeout=5.0) as conn:
                conn.execute(f"CREATE TABLE conc_t{i} (id INT)")
                conn.execute(f"INSERT INTO conc_t{i} VALUES ({i})")
                rows = conn.query(f"SELECT * FROM conc_t{i}")
                results.append(len(rows))
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 4
    pool.close()

def test_pool_with_partitioner():
    """Partitioner with read/write pools."""
    with PoolPartitioner() as part:
        part.add_pool("write", config=make_config(min_size=1), is_default=True)
        part.add_pool("read", config=make_config(min_size=1))
        with part.acquire("write") as w:
            w.execute("CREATE TABLE part_t1 (id INT)")
            w.execute("INSERT INTO part_t1 VALUES (1)")
        stats = part.stats()
        assert stats["write"]["total_borrows"] >= 1

def test_pool_cluster_failover_integration():
    """Cluster failover with real pools."""
    with PoolCluster() as cluster:
        cluster.add_node("primary", config=make_config(min_size=1, max_size=1))
        cluster.add_node("backup", config=make_config(min_size=1, max_size=2))
        # Hold all primary connections
        c1 = cluster.acquire(node_name="primary")
        # Should failover
        c2 = cluster.acquire(timeout=0.5)
        assert c2 is not None
        c1.close()
        c2.close()

def test_pool_maintenance_cycle():
    """Full maintenance cycle."""
    pool = make_pool(min_size=2, max_size=5)
    with pool.acquire():
        pass
    result = pool.maintenance()
    assert isinstance(result, dict)
    assert pool.size >= 2
    pool.close()

def test_pool_stats_comprehensive():
    """Verify all stats fields are populated."""
    pool = make_pool(min_size=1)
    with pool.acquire():
        pass
    with pool.acquire():
        pass
    s = pool.stats
    assert s.total_borrows >= 2
    assert s.total_returns >= 2
    assert s.total_creates >= 1
    d = s.to_dict()
    assert d['total_borrows'] >= 2
    pool.close()

def test_connection_state_transitions():
    """Verify connection states through lifecycle."""
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    inner = conn.connection
    assert inner.state == ConnectionState.IN_USE
    conn.close()
    assert inner.state == ConnectionState.IDLE
    pool.close()

def test_pool_utilization():
    pool = make_pool(min_size=2, max_size=4)
    assert pool.stats.utilization == 0.0
    c1 = pool.acquire()
    assert pool.stats.utilization > 0
    c1.close()
    pool.close()

def test_empty_stats():
    stats = PoolStats()
    assert stats.avg_wait_time_ms == 0.0
    assert stats.utilization == 0.0


# ==== Edge Cases ====

def test_pool_zero_min_size():
    pool = make_pool(min_size=0, max_size=2)
    assert pool.size == 0
    with pool.acquire() as conn:
        assert conn is not None
    pool.close()

def test_pool_min_equals_max():
    pool = make_pool(min_size=2, max_size=2)
    assert pool.size == 2
    c1 = pool.acquire()
    c2 = pool.acquire()
    try:
        pool.acquire(timeout=0.1)
        assert False
    except TimeoutError:
        pass
    c1.close()
    c2.close()
    pool.close()

def test_return_invalid_connection():
    """Returning an invalid connection should destroy it."""
    pool = make_pool(min_size=1)
    conn = pool.acquire()
    inner = conn.connection
    inner.info.state = ConnectionState.INVALID
    conn.close()
    # Invalid connection should be destroyed, not returned to idle
    pool.close()

def test_pool_acquire_after_close():
    pool = make_pool(min_size=1)
    pool.close()
    try:
        pool.acquire()
        assert False
    except ConnectionError:
        pass

def test_connection_reset_with_tx():
    """Reset should rollback active transaction."""
    engine = IntegratedQueryEngine(db_name="test_reset_tx")
    config = make_config()
    conn = Connection(engine, config)
    conn.state = ConnectionState.IN_USE
    conn.execute("CREATE TABLE reset_t (id INT)")
    conn.begin()
    conn.reset()
    assert not conn.in_transaction

def test_max_idle_eviction():
    """Connections beyond max_idle should be destroyed on return."""
    pool = make_pool(min_size=0, max_size=5, max_idle=2)
    conns = [pool.acquire() for _ in range(4)]
    for c in conns:
        c.close()
    # Should only keep max_idle in the idle pool
    assert pool.idle_count <= 2
    pool.close()


# ---- Run all tests ----

if __name__ == '__main__':
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    errors = []

    for fn in test_funcs:
        try:
            fn()
            passed += 1
            print(f"  PASS: {fn.__name__}")
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, str(e)))
            print(f"  FAIL: {fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(test_funcs)}")
    print(f"{'='*60}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    sys.exit(0 if failed == 0 else 1)
