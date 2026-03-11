"""
C223: Database Connection Pool
Composes C220 (Query Executor Integration) + C118 (Cache Systems)

A production-grade connection pool with:
- ConnectionFactory: creates/validates/destroys connections
- Connection: wrapper with state tracking, auto-reset on return
- ConnectionPool: core pool with min/max sizing, borrow/return, timeouts
- PooledConnection: proxy that auto-returns on close/context-exit
- HealthChecker: background validation, idle timeout, connection aging
- ConnectionStats: hit rates, wait times, pool utilization
- PreparedStatementCache: per-connection LRU cache for prepared statements
- PoolPartitioner: named sub-pools for read/write splitting
- ConnectionLeakDetector: tracks unreturned connections with stack traces
- PoolCluster: multiple pools with failover and load balancing
"""

import sys
import os
import time
import threading
import traceback
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C220_query_executor_integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C118_cache'))

from query_executor_integration import IntegratedQueryEngine, ExecutionResult
from cache import LRUCache, TTLCache


# --- Enums ---

class ConnectionState(Enum):
    IDLE = auto()
    IN_USE = auto()
    VALIDATING = auto()
    CLOSED = auto()
    INVALID = auto()


class PoolState(Enum):
    RUNNING = auto()
    DRAINING = auto()
    CLOSED = auto()


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    RANDOM = auto()


# --- Data classes ---

@dataclass
class PoolConfig:
    min_size: int = 2
    max_size: int = 10
    acquire_timeout: float = 5.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0
    validation_interval: float = 30.0
    validation_query: str = "SELECT 1"
    validate_on_borrow: bool = True
    validate_on_return: bool = False
    test_on_create: bool = True
    max_idle: int = 5
    prepared_cache_size: int = 50
    leak_detection_threshold: float = 30.0
    db_name: str = "default"


@dataclass
class ConnectionInfo:
    id: int
    created_at: float
    last_used: float
    last_validated: float
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    borrowed_at: Optional[float] = None
    borrowed_stack: Optional[str] = None


@dataclass
class PoolStats:
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_borrows: int = 0
    total_returns: int = 0
    total_creates: int = 0
    total_destroys: int = 0
    total_validations: int = 0
    failed_validations: int = 0
    total_timeouts: int = 0
    total_wait_time_ms: float = 0.0
    peak_active: int = 0
    leaked_connections: int = 0

    @property
    def avg_wait_time_ms(self):
        if self.total_borrows == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_borrows

    @property
    def utilization(self):
        if self.total_connections == 0:
            return 0.0
        return self.active_connections / self.total_connections

    def to_dict(self):
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'idle_connections': self.idle_connections,
            'total_borrows': self.total_borrows,
            'total_returns': self.total_returns,
            'total_creates': self.total_creates,
            'total_destroys': self.total_destroys,
            'total_validations': self.total_validations,
            'failed_validations': self.failed_validations,
            'total_timeouts': self.total_timeouts,
            'avg_wait_time_ms': self.avg_wait_time_ms,
            'peak_active': self.peak_active,
            'utilization': self.utilization,
            'leaked_connections': self.leaked_connections,
        }


# --- Connection ---

class Connection:
    """Wraps an IntegratedQueryEngine with state tracking."""

    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, engine, config):
        with Connection._id_lock:
            Connection._next_id += 1
            self._id = Connection._next_id
        self._engine = engine
        self._config = config
        now = time.monotonic()
        self.info = ConnectionInfo(
            id=self._id,
            created_at=now,
            last_used=now,
            last_validated=now,
        )
        self._tx_id = None
        self._prepared_cache = LRUCache(config.prepared_cache_size)
        self._dirty = False

    @property
    def id(self):
        return self._id

    @property
    def engine(self):
        return self._engine

    @property
    def state(self):
        return self.info.state

    @state.setter
    def state(self, value):
        self.info.state = value

    @property
    def is_valid(self):
        return self.info.state not in (ConnectionState.CLOSED, ConnectionState.INVALID)

    @property
    def age(self):
        return time.monotonic() - self.info.created_at

    @property
    def idle_time(self):
        return time.monotonic() - self.info.last_used

    @property
    def in_transaction(self):
        return self._tx_id is not None

    def execute(self, sql, params=None):
        if not self.is_valid:
            raise ConnectionError(f"Connection {self._id} is {self.state.name}")
        self.info.last_used = time.monotonic()
        self.info.use_count += 1
        result = self._engine.execute(sql, tx_id=self._tx_id, params=params)
        if sql.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            self._dirty = True
        return result

    def query(self, sql, params=None):
        if not self.is_valid:
            raise ConnectionError(f"Connection {self._id} is {self.state.name}")
        self.info.last_used = time.monotonic()
        self.info.use_count += 1
        return self._engine.query(sql, tx_id=self._tx_id)

    def begin(self, isolation=None):
        if self._tx_id is not None:
            raise ConnectionError("Already in transaction")
        self._tx_id = self._engine.begin(isolation=isolation)
        self._dirty = False
        return self._tx_id

    def commit(self):
        if self._tx_id is None:
            raise ConnectionError("No active transaction")
        result = self._engine.commit(self._tx_id)
        self._tx_id = None
        self._dirty = False
        return result

    def rollback(self):
        if self._tx_id is None:
            raise ConnectionError("No active transaction")
        result = self._engine.rollback(self._tx_id)
        self._tx_id = None
        self._dirty = False
        return result

    def validate(self, query="SELECT 1"):
        """Run a validation query. Returns True if connection is healthy."""
        try:
            self.info.state = ConnectionState.VALIDATING
            self._engine.execute(query)
            self.info.last_validated = time.monotonic()
            return True
        except Exception:
            self.info.state = ConnectionState.INVALID
            return False

    def reset(self):
        """Reset connection state for return to pool."""
        if self._tx_id is not None:
            try:
                self._engine.rollback(self._tx_id)
            except Exception:
                pass
            self._tx_id = None
        self._dirty = False

    def close(self):
        """Mark connection as closed."""
        self.reset()
        self.info.state = ConnectionState.CLOSED

    def get_prepared(self, sql):
        """Get a cached prepared statement result."""
        return self._prepared_cache.get(sql)

    def set_prepared(self, sql, prepared):
        """Cache a prepared statement."""
        self._prepared_cache.put(sql, prepared)

    def __repr__(self):
        return f"Connection(id={self._id}, state={self.state.name}, uses={self.info.use_count})"


# --- Connection Factory ---

class ConnectionFactory:
    """Creates, validates, and destroys connections."""

    def __init__(self, config, engine_factory=None):
        self._config = config
        self._engine_factory = engine_factory

    def create(self):
        if self._engine_factory:
            engine = self._engine_factory()
        else:
            engine = IntegratedQueryEngine(db_name=self._config.db_name)
        conn = Connection(engine, self._config)
        if self._config.test_on_create:
            if not conn.validate(self._config.validation_query):
                conn.close()
                raise ConnectionError("New connection failed validation")
        return conn

    def validate(self, conn):
        return conn.validate(self._config.validation_query)

    def destroy(self, conn):
        conn.close()


# --- Pooled Connection (Proxy) ---

class PooledConnection:
    """Proxy that auto-returns to pool on close or context exit."""

    def __init__(self, connection, pool, on_close=None):
        self._connection = connection
        self._pool = pool
        self._on_close = on_close
        self._closed = False

    @property
    def id(self):
        return self._connection.id

    @property
    def connection(self):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection

    @property
    def info(self):
        return self._connection.info

    def execute(self, sql, params=None):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection.execute(sql, params=params)

    def query(self, sql, params=None):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection.query(sql, params=params)

    def begin(self, isolation=None):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection.begin(isolation=isolation)

    def commit(self):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection.commit()

    def rollback(self):
        if self._closed:
            raise ConnectionError("PooledConnection is closed")
        return self._connection.rollback()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._on_close:
            self._on_close(self._connection)
        else:
            self._pool.return_connection(self._connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        status = "closed" if self._closed else "open"
        return f"PooledConnection(id={self._connection.id}, {status})"


# --- Connection Leak Detector ---

class ConnectionLeakDetector:
    """Tracks borrowed connections and detects leaks."""

    def __init__(self, threshold=30.0):
        self._threshold = threshold
        self._borrowed = {}  # conn_id -> (borrowed_at, stack_trace)
        self._leaks = []
        self._lock = threading.Lock()

    def track_borrow(self, conn):
        with self._lock:
            self._borrowed[conn.id] = (
                time.monotonic(),
                traceback.format_stack()
            )

    def track_return(self, conn):
        with self._lock:
            self._borrowed.pop(conn.id, None)

    def check_leaks(self):
        """Check for connections held longer than threshold. Returns list of leak info."""
        now = time.monotonic()
        leaks = []
        with self._lock:
            for conn_id, (borrowed_at, stack) in list(self._borrowed.items()):
                held_time = now - borrowed_at
                if held_time > self._threshold:
                    leak_info = {
                        'connection_id': conn_id,
                        'held_seconds': held_time,
                        'stack': stack,
                    }
                    leaks.append(leak_info)
                    if conn_id not in [l['connection_id'] for l in self._leaks]:
                        self._leaks.append(leak_info)
        return leaks

    @property
    def leaked_count(self):
        return len(self.check_leaks())

    @property
    def all_leaks(self):
        return list(self._leaks)

    def clear(self):
        with self._lock:
            self._borrowed.clear()
            self._leaks.clear()


# --- Health Checker ---

class HealthChecker:
    """Validates pool connections periodically."""

    def __init__(self, pool, config):
        self._pool = pool
        self._config = config

    def check_idle_connections(self):
        """Validate idle connections, evict stale ones. Returns (validated, evicted)."""
        validated = 0
        evicted = 0
        now = time.monotonic()

        to_check = []
        with self._pool._lock:
            for conn in list(self._pool._idle):
                to_check.append(conn)

        for conn in to_check:
            # Check max lifetime
            if conn.age > self._config.max_lifetime:
                self._pool._evict_connection(conn)
                evicted += 1
                continue

            # Check idle timeout
            if conn.idle_time > self._config.idle_timeout:
                # Keep min_size connections
                with self._pool._lock:
                    total = len(self._pool._idle) + len(self._pool._active)
                if total > self._config.min_size:
                    self._pool._evict_connection(conn)
                    evicted += 1
                    continue

            # Validate if interval passed
            time_since_validation = now - conn.info.last_validated
            if time_since_validation > self._config.validation_interval:
                self._pool._stats.total_validations += 1
                if not self._pool._factory.validate(conn):
                    self._pool._stats.failed_validations += 1
                    self._pool._evict_connection(conn)
                    evicted += 1
                else:
                    conn.state = ConnectionState.IDLE
                    validated += 1

        return validated, evicted

    def ensure_min_size(self):
        """Ensure pool has at least min_size connections. Returns number created."""
        created = 0
        with self._pool._lock:
            total = len(self._pool._idle) + len(self._pool._active)
            needed = self._config.min_size - total

        for _ in range(needed):
            try:
                conn = self._pool._factory.create()
                conn.state = ConnectionState.IDLE
                with self._pool._lock:
                    self._pool._idle.append(conn)
                    self._pool._stats.total_connections += 1
                    self._pool._stats.idle_connections += 1
                    self._pool._stats.total_creates += 1
                created += 1
            except Exception:
                break

        return created

    def run_maintenance(self):
        """Full maintenance cycle: validate, evict, replenish."""
        validated, evicted = self.check_idle_connections()
        created = self.ensure_min_size()
        return {'validated': validated, 'evicted': evicted, 'created': created}


# --- Prepared Statement Cache ---

class PreparedStatementCache:
    """Per-connection prepared statement cache using LRU."""

    def __init__(self, capacity=50):
        self._cache = LRUCache(capacity)

    def get(self, sql):
        return self._cache.get(sql)

    def put(self, sql, result):
        self._cache.put(sql, result)

    def invalidate(self, sql=None):
        if sql:
            self._cache.delete(sql)
        else:
            self._cache.clear()

    @property
    def size(self):
        return len(self._cache)

    @property
    def hit_rate(self):
        return self._cache.hit_rate

    def stats(self):
        return self._cache.stats()


# --- Connection Pool ---

class ConnectionPool:
    """Core connection pool with borrow/return, sizing, health checking."""

    def __init__(self, config=None, engine_factory=None):
        self._config = config or PoolConfig()
        self._factory = ConnectionFactory(self._config, engine_factory=engine_factory)
        self._idle = deque()
        self._active = set()
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._state = PoolState.RUNNING
        self._stats = PoolStats()
        self._leak_detector = ConnectionLeakDetector(self._config.leak_detection_threshold)
        self._health_checker = HealthChecker(self, self._config)
        self._event_listeners = []

        # Pre-populate with min_size connections
        self._initialize()

    def _initialize(self):
        for _ in range(self._config.min_size):
            try:
                conn = self._factory.create()
                conn.state = ConnectionState.IDLE
                self._idle.append(conn)
                self._stats.total_connections += 1
                self._stats.idle_connections += 1
                self._stats.total_creates += 1
            except Exception:
                break

    def acquire(self, timeout=None):
        """Acquire a connection from the pool. Returns a PooledConnection."""
        if self._state != PoolState.RUNNING:
            raise ConnectionError(f"Pool is {self._state.name}")

        timeout = timeout if timeout is not None else self._config.acquire_timeout
        start = time.monotonic()

        with self._available:
            while True:
                # Try to get an idle connection
                conn = self._try_get_idle()
                if conn is not None:
                    break

                # Try to create a new one if under max
                if len(self._idle) + len(self._active) < self._config.max_size:
                    try:
                        conn = self._factory.create()
                        self._stats.total_creates += 1
                        self._stats.total_connections += 1
                        break
                    except Exception:
                        pass

                # Wait for a connection to be returned
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    self._stats.total_timeouts += 1
                    raise TimeoutError(f"Could not acquire connection within {timeout}s")

                self._available.wait(timeout=remaining)

                if self._state != PoolState.RUNNING:
                    raise ConnectionError(f"Pool is {self._state.name}")

            # Mark as active
            conn.state = ConnectionState.IN_USE
            conn.info.borrowed_at = time.monotonic()
            self._active.add(conn)
            self._stats.active_connections = len(self._active)
            self._stats.idle_connections = len(self._idle)
            self._stats.total_borrows += 1

            wait_ms = (time.monotonic() - start) * 1000
            self._stats.total_wait_time_ms += wait_ms

            if self._stats.active_connections > self._stats.peak_active:
                self._stats.peak_active = self._stats.active_connections

        # Validate on borrow (outside lock)
        if self._config.validate_on_borrow:
            self._stats.total_validations += 1
            if not self._factory.validate(conn):
                self._stats.failed_validations += 1
                with self._available:
                    self._active.discard(conn)
                    self._stats.active_connections = len(self._active)
                    self._stats.total_connections -= 1
                self._factory.destroy(conn)
                self._stats.total_destroys += 1
                # Retry once
                return self.acquire(timeout=max(0, timeout - (time.monotonic() - start)))

        self._leak_detector.track_borrow(conn)
        self._fire_event('borrow', conn)

        return PooledConnection(conn, self)

    def _try_get_idle(self):
        """Try to get a valid idle connection. Must hold lock."""
        while self._idle:
            conn = self._idle.popleft()
            if conn.is_valid and conn.age <= self._config.max_lifetime:
                return conn
            # Invalid or expired, destroy it
            self._stats.total_connections -= 1
            self._stats.total_destroys += 1
            self._factory.destroy(conn)
        return None

    def return_connection(self, conn):
        """Return a connection to the pool."""
        self._leak_detector.track_return(conn)
        self._fire_event('return', conn)

        # Reset connection state
        conn.reset()

        with self._available:
            self._active.discard(conn)
            self._stats.total_returns += 1

            # Validate on return
            if self._config.validate_on_return:
                self._stats.total_validations += 1
                if not self._factory.validate(conn):
                    self._stats.failed_validations += 1
                    self._stats.total_connections -= 1
                    self._stats.total_destroys += 1
                    self._factory.destroy(conn)
                    self._stats.active_connections = len(self._active)
                    self._stats.idle_connections = len(self._idle)
                    self._available.notify()
                    return

            # Check if pool is draining/closed or over max_idle
            if self._state != PoolState.RUNNING or len(self._idle) >= self._config.max_idle:
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)
            elif conn.is_valid and conn.age <= self._config.max_lifetime:
                conn.state = ConnectionState.IDLE
                conn.info.last_used = time.monotonic()
                conn.info.borrowed_at = None
                conn.info.borrowed_stack = None
                self._idle.append(conn)
            else:
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)

            self._stats.active_connections = len(self._active)
            self._stats.idle_connections = len(self._idle)
            self._available.notify()

    def _evict_connection(self, conn):
        """Remove a connection from the pool."""
        with self._available:
            if conn in self._idle:
                self._idle.remove(conn)
            self._active.discard(conn)
            self._stats.total_connections -= 1
            self._stats.idle_connections = len(self._idle)
            self._stats.active_connections = len(self._active)
            self._stats.total_destroys += 1
        self._factory.destroy(conn)

    def drain(self):
        """Stop accepting new borrows, wait for active to return, close all."""
        self._state = PoolState.DRAINING
        with self._available:
            # Close all idle
            while self._idle:
                conn = self._idle.popleft()
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)
            self._stats.idle_connections = 0
            self._available.notify_all()

    def close(self):
        """Force close all connections."""
        self._state = PoolState.CLOSED
        with self._available:
            while self._idle:
                conn = self._idle.popleft()
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)
            for conn in list(self._active):
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)
            self._active.clear()
            self._stats.idle_connections = 0
            self._stats.active_connections = 0
            self._available.notify_all()
        self._leak_detector.clear()

    def maintenance(self):
        """Run health check maintenance cycle."""
        return self._health_checker.run_maintenance()

    def check_leaks(self):
        """Check for leaked connections."""
        leaks = self._leak_detector.check_leaks()
        self._stats.leaked_connections = len(leaks)
        return leaks

    def resize(self, min_size=None, max_size=None):
        """Dynamically resize the pool."""
        if min_size is not None:
            self._config.min_size = min_size
        if max_size is not None:
            self._config.max_size = max_size
        # Ensure min <= max
        if self._config.min_size > self._config.max_size:
            self._config.min_size = self._config.max_size
        # Shrink if over max
        with self._available:
            while len(self._idle) + len(self._active) > self._config.max_size and self._idle:
                conn = self._idle.popleft()
                self._stats.total_connections -= 1
                self._stats.total_destroys += 1
                self._factory.destroy(conn)
            self._stats.idle_connections = len(self._idle)
        # Grow if under min
        self._health_checker.ensure_min_size()

    def add_event_listener(self, listener):
        """Add event listener. Called with (event_name, connection)."""
        self._event_listeners.append(listener)

    def _fire_event(self, event, conn):
        for listener in self._event_listeners:
            try:
                listener(event, conn)
            except Exception:
                pass

    @property
    def stats(self):
        return self._stats

    @property
    def size(self):
        with self._lock:
            return len(self._idle) + len(self._active)

    @property
    def idle_count(self):
        with self._lock:
            return len(self._idle)

    @property
    def active_count(self):
        with self._lock:
            return len(self._active)

    @property
    def state(self):
        return self._state

    @property
    def config(self):
        return self._config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        return (f"ConnectionPool(size={self.size}, active={self.active_count}, "
                f"idle={self.idle_count}, state={self._state.name})")


# --- Pool Partitioner ---

class PoolPartitioner:
    """Named sub-pools for read/write splitting or multi-tenant."""

    def __init__(self):
        self._pools = {}
        self._default = None

    def add_pool(self, name, config=None, engine_factory=None, is_default=False):
        """Add a named pool."""
        pool = ConnectionPool(config=config, engine_factory=engine_factory)
        self._pools[name] = pool
        if is_default or self._default is None:
            self._default = name
        return pool

    def get_pool(self, name=None):
        """Get a named pool, or the default."""
        name = name or self._default
        if name not in self._pools:
            raise KeyError(f"No pool named '{name}'")
        return self._pools[name]

    def acquire(self, pool_name=None, timeout=None):
        """Acquire from a named pool."""
        return self.get_pool(pool_name).acquire(timeout=timeout)

    def close_all(self):
        """Close all pools."""
        for pool in self._pools.values():
            pool.close()
        self._pools.clear()

    @property
    def pool_names(self):
        return list(self._pools.keys())

    @property
    def default_pool(self):
        return self._default

    def stats(self):
        """Get stats for all pools."""
        return {name: pool.stats.to_dict() for name, pool in self._pools.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False


# --- Pool Cluster ---

class PoolCluster:
    """Multiple pools with failover and load balancing."""

    def __init__(self, strategy=LoadBalanceStrategy.ROUND_ROBIN):
        self._pools = {}  # name -> (pool, weight, enabled)
        self._strategy = strategy
        self._rr_index = 0
        self._lock = threading.Lock()

    def add_node(self, name, config=None, engine_factory=None, weight=1):
        """Add a node to the cluster."""
        pool = ConnectionPool(config=config, engine_factory=engine_factory)
        self._pools[name] = {'pool': pool, 'weight': weight, 'enabled': True}
        return pool

    def remove_node(self, name):
        """Remove a node from the cluster."""
        if name in self._pools:
            self._pools[name]['pool'].close()
            del self._pools[name]

    def enable_node(self, name):
        if name in self._pools:
            self._pools[name]['enabled'] = True

    def disable_node(self, name):
        if name in self._pools:
            self._pools[name]['enabled'] = False

    def _get_enabled_nodes(self):
        return {n: info for n, info in self._pools.items() if info['enabled']}

    def _select_node(self):
        """Select a node based on strategy."""
        enabled = self._get_enabled_nodes()
        if not enabled:
            raise ConnectionError("No enabled nodes in cluster")

        names = sorted(enabled.keys())

        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            with self._lock:
                idx = self._rr_index % len(names)
                self._rr_index += 1
            return names[idx]

        elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            min_active = float('inf')
            best = names[0]
            for name in names:
                active = enabled[name]['pool'].active_count
                if active < min_active:
                    min_active = active
                    best = name
            return best

        elif self._strategy == LoadBalanceStrategy.RANDOM:
            import random
            # Weighted random
            weights = [enabled[n]['weight'] for n in names]
            total = sum(weights)
            r = random.random() * total
            cumulative = 0
            for i, name in enumerate(names):
                cumulative += weights[i]
                if r <= cumulative:
                    return name
            return names[-1]

        return names[0]

    def acquire(self, timeout=None, node_name=None):
        """Acquire a connection with optional failover."""
        if node_name:
            if node_name not in self._pools:
                raise KeyError(f"No node named '{node_name}'")
            return self._pools[node_name]['pool'].acquire(timeout=timeout)

        # Try selected node, then failover to others
        enabled = self._get_enabled_nodes()
        if not enabled:
            raise ConnectionError("No enabled nodes in cluster")

        selected = self._select_node()
        try:
            return enabled[selected]['pool'].acquire(timeout=timeout)
        except (ConnectionError, TimeoutError):
            # Failover to other nodes
            for name in sorted(enabled.keys()):
                if name == selected:
                    continue
                try:
                    return enabled[name]['pool'].acquire(timeout=timeout)
                except (ConnectionError, TimeoutError):
                    continue
            raise ConnectionError("All cluster nodes exhausted")

    def close(self):
        for info in self._pools.values():
            info['pool'].close()
        self._pools.clear()

    @property
    def node_names(self):
        return list(self._pools.keys())

    @property
    def enabled_nodes(self):
        return [n for n, info in self._pools.items() if info['enabled']]

    def stats(self):
        return {name: info['pool'].stats.to_dict() for name, info in self._pools.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
