"""
C228: Load Balancer
Composes C225 (Circuit Breaker & Resilience) + C222 (Service Discovery)

L4/L7 load balancer with:
- Multiple balancing algorithms (round-robin, weighted, least-connections, IP-hash, random, power-of-two)
- Health-based routing with active/passive health checks
- Sticky sessions (cookie-based and IP-based)
- Connection draining for graceful removal
- Request routing rules (path-based, header-based, method-based)
- Backend groups/pools with independent configs
- Metrics and monitoring
- Hot reconfiguration
"""

import time
import math
import hashlib
import random
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BalancingAlgorithm(Enum):
    ROUND_ROBIN = auto()
    WEIGHTED_ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    IP_HASH = auto()
    RANDOM = auto()
    POWER_OF_TWO = auto()  # Pick 2 random, choose least loaded


class BackendStatus(Enum):
    HEALTHY = auto()
    UNHEALTHY = auto()
    DRAINING = auto()  # Finishing existing connections, no new ones


class HealthCheckType(Enum):
    ACTIVE = auto()   # Periodically probe backend
    PASSIVE = auto()  # Track failures from real requests


class StickySessionType(Enum):
    NONE = auto()
    IP_HASH = auto()
    COOKIE = auto()


class RoutingMatchType(Enum):
    EXACT = auto()
    PREFIX = auto()
    REGEX = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    """A single backend server."""
    id: str
    address: str
    port: int
    weight: int = 1
    max_connections: int = 0  # 0 = unlimited
    status: BackendStatus = BackendStatus.HEALTHY
    metadata: dict = field(default_factory=dict)
    # Runtime state
    active_connections: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0
    last_health_check: float = 0.0
    last_failure: float = 0.0
    last_success: float = 0.0
    response_times: list = field(default_factory=list)  # Recent response times
    drain_started: float = 0.0

    @property
    def failure_rate(self):
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_failures / total

    @property
    def avg_response_time(self):
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def to_dict(self):
        return {
            'id': self.id, 'address': self.address, 'port': self.port,
            'weight': self.weight, 'status': self.status.name,
            'active_connections': self.active_connections,
            'total_requests': self.total_requests,
            'failure_rate': self.failure_rate,
            'avg_response_time': self.avg_response_time,
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checking a backend."""
    check_type: HealthCheckType = HealthCheckType.ACTIVE
    interval: float = 10.0          # Seconds between active checks
    timeout: float = 5.0            # Timeout for each check
    healthy_threshold: int = 2      # Consecutive successes to mark healthy
    unhealthy_threshold: int = 3    # Consecutive failures to mark unhealthy
    check_func: Optional[Callable] = None  # Custom check function
    # Passive check config
    passive_failure_threshold: int = 5     # Failures within window
    passive_window: float = 30.0           # Window for passive checks


@dataclass
class StickySessionConfig:
    """Configuration for sticky sessions."""
    session_type: StickySessionType = StickySessionType.NONE
    cookie_name: str = "LB_SESSION"
    ttl: float = 3600.0  # Session TTL in seconds


@dataclass
class RoutingRule:
    """A routing rule that maps requests to backend pools."""
    name: str
    pool_name: str
    match_type: RoutingMatchType = RoutingMatchType.PREFIX
    path_pattern: str = "/"
    methods: Optional[list] = None       # None = all methods
    headers: Optional[dict] = None       # Header conditions
    priority: int = 0                    # Higher = checked first
    metadata: dict = field(default_factory=dict)


@dataclass
class Request:
    """Represents an incoming request to the load balancer."""
    method: str = "GET"
    path: str = "/"
    headers: dict = field(default_factory=dict)
    body: Any = None
    client_ip: str = "127.0.0.1"
    metadata: dict = field(default_factory=dict)


@dataclass
class Response:
    """Response from a backend."""
    status: int = 200
    headers: dict = field(default_factory=dict)
    body: Any = None
    backend_id: str = ""
    response_time: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LoadBalancerError(Exception):
    pass

class NoHealthyBackendError(LoadBalancerError):
    pass

class BackendOverloadedError(LoadBalancerError):
    pass

class PoolNotFoundError(LoadBalancerError):
    pass

class DrainTimeoutError(LoadBalancerError):
    pass


# ---------------------------------------------------------------------------
# Backend Pool
# ---------------------------------------------------------------------------

class BackendPool:
    """A group of backends with shared balancing and health config."""

    def __init__(self, name, algorithm=BalancingAlgorithm.ROUND_ROBIN,
                 health_config=None, sticky_config=None,
                 max_response_times=100):
        self.name = name
        self.algorithm = algorithm
        self.health_config = health_config or HealthCheckConfig()
        self.sticky_config = sticky_config or StickySessionConfig()
        self.max_response_times = max_response_times
        self.backends = {}  # id -> Backend
        self._rr_index = 0
        self._wrr_state = {}  # id -> current_weight for smooth WRR
        self._sticky_map = {}  # session_key -> (backend_id, expiry)
        self._circuit_breakers = {}  # backend_id -> circuit state
        self._lock = threading.Lock()

    def add_backend(self, backend_id, address, port, weight=1,
                    max_connections=0, metadata=None):
        """Add a backend to this pool."""
        backend = Backend(
            id=backend_id, address=address, port=port,
            weight=weight, max_connections=max_connections,
            metadata=metadata or {}
        )
        self.backends[backend_id] = backend
        return backend

    def remove_backend(self, backend_id):
        """Remove a backend immediately."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            self._sticky_map = {
                k: v for k, v in self._sticky_map.items()
                if v[0] != backend_id
            }
            return True
        return False

    def drain_backend(self, backend_id, timeout=30.0):
        """Start draining a backend (no new connections)."""
        if backend_id not in self.backends:
            return False
        backend = self.backends[backend_id]
        backend.status = BackendStatus.DRAINING
        backend.drain_started = time.time()
        return True

    def get_healthy_backends(self):
        """Get all healthy backends."""
        return [b for b in self.backends.values()
                if b.status == BackendStatus.HEALTHY]

    def get_all_backends(self):
        """Get all backends."""
        return list(self.backends.values())

    def select_backend(self, request=None, now=None):
        """Select a backend using the configured algorithm."""
        if now is None:
            now = time.time()

        # Check sticky session first
        if self.sticky_config.session_type != StickySessionType.NONE and request:
            sticky_backend = self._check_sticky(request, now)
            if sticky_backend and sticky_backend.status == BackendStatus.HEALTHY:
                if sticky_backend.max_connections == 0 or \
                   sticky_backend.active_connections < sticky_backend.max_connections:
                    return sticky_backend

        healthy = self.get_healthy_backends()
        if not healthy:
            return None

        # Filter by max_connections
        available = [b for b in healthy
                     if b.max_connections == 0 or
                     b.active_connections < b.max_connections]
        if not available:
            return None

        backend = self._select_by_algorithm(available, request)

        # Record sticky session
        if backend and self.sticky_config.session_type != StickySessionType.NONE and request:
            self._record_sticky(request, backend, now)

        return backend

    def _select_by_algorithm(self, backends, request=None):
        """Apply the balancing algorithm."""
        if not backends:
            return None

        if self.algorithm == BalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin(backends)
        elif self.algorithm == BalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(backends)
        elif self.algorithm == BalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(backends)
        elif self.algorithm == BalancingAlgorithm.IP_HASH:
            return self._ip_hash(backends, request)
        elif self.algorithm == BalancingAlgorithm.RANDOM:
            return random.choice(backends)
        elif self.algorithm == BalancingAlgorithm.POWER_OF_TWO:
            return self._power_of_two(backends)
        return backends[0]

    def _round_robin(self, backends):
        """Simple round-robin selection."""
        idx = self._rr_index % len(backends)
        self._rr_index += 1
        return backends[idx]

    def _weighted_round_robin(self, backends):
        """Smooth weighted round-robin (Nginx algorithm)."""
        # Initialize weights
        for b in backends:
            if b.id not in self._wrr_state:
                self._wrr_state[b.id] = 0

        # Clean up stale entries
        active_ids = {b.id for b in backends}
        self._wrr_state = {k: v for k, v in self._wrr_state.items() if k in active_ids}

        total_weight = sum(b.weight for b in backends)

        # Add effective weight to current weight
        for b in backends:
            self._wrr_state[b.id] = self._wrr_state.get(b.id, 0) + b.weight

        # Select the one with highest current weight
        best = max(backends, key=lambda b: self._wrr_state[b.id])

        # Subtract total weight from selected
        self._wrr_state[best.id] -= total_weight

        return best

    def _least_connections(self, backends):
        """Select backend with fewest active connections."""
        return min(backends, key=lambda b: b.active_connections)

    def _ip_hash(self, backends, request):
        """Hash client IP to select backend."""
        ip = request.client_ip if request else "127.0.0.1"
        h = int(hashlib.md5(ip.encode()).hexdigest(), 16)
        return backends[h % len(backends)]

    def _power_of_two(self, backends):
        """Power of two random choices - pick 2 random, choose least loaded."""
        if len(backends) == 1:
            return backends[0]
        a, b = random.sample(backends, 2)
        if a.active_connections <= b.active_connections:
            return a
        return b

    def _get_sticky_key(self, request):
        """Get the sticky session key for a request."""
        if self.sticky_config.session_type == StickySessionType.IP_HASH:
            return f"ip:{request.client_ip}"
        elif self.sticky_config.session_type == StickySessionType.COOKIE:
            cookie = request.headers.get('Cookie', '')
            cookie_name = self.sticky_config.cookie_name
            for part in cookie.split(';'):
                part = part.strip()
                if part.startswith(f'{cookie_name}='):
                    return f"cookie:{part[len(cookie_name)+1:]}"
            return None
        return None

    def _check_sticky(self, request, now):
        """Check if request has a sticky session."""
        key = self._get_sticky_key(request)
        if key and key in self._sticky_map:
            backend_id, expiry = self._sticky_map[key]
            if now < expiry and backend_id in self.backends:
                return self.backends[backend_id]
            elif now >= expiry:
                del self._sticky_map[key]
        return None

    def _record_sticky(self, request, backend, now):
        """Record a sticky session binding."""
        if self.sticky_config.session_type == StickySessionType.IP_HASH:
            key = f"ip:{request.client_ip}"
        elif self.sticky_config.session_type == StickySessionType.COOKIE:
            # Generate a session ID for new cookies
            key = f"cookie:{backend.id}_{int(now)}"
        else:
            return
        expiry = now + self.sticky_config.ttl
        self._sticky_map[key] = (backend.id, expiry)

    def record_success(self, backend_id, response_time=0.0):
        """Record a successful request to a backend."""
        if backend_id not in self.backends:
            return
        b = self.backends[backend_id]
        b.total_successes += 1
        b.total_requests += 1
        b.consecutive_failures = 0
        b.last_success = time.time()
        if response_time > 0:
            b.response_times.append(response_time)
            if len(b.response_times) > self.max_response_times:
                b.response_times.pop(0)

        # Passive health: recover if enough successes
        if b.status == BackendStatus.UNHEALTHY:
            if b.total_successes >= self.health_config.healthy_threshold:
                b.status = BackendStatus.HEALTHY
                b.consecutive_failures = 0

    def record_failure(self, backend_id):
        """Record a failed request to a backend."""
        if backend_id not in self.backends:
            return
        b = self.backends[backend_id]
        b.total_failures += 1
        b.total_requests += 1
        b.consecutive_failures += 1
        b.last_failure = time.time()

        # Passive health: mark unhealthy after threshold
        if self.health_config.check_type == HealthCheckType.PASSIVE:
            if b.consecutive_failures >= self.health_config.unhealthy_threshold:
                b.status = BackendStatus.UNHEALTHY

    def run_health_checks(self, now=None):
        """Run active health checks on all backends."""
        if now is None:
            now = time.time()
        if self.health_config.check_type != HealthCheckType.ACTIVE:
            return []

        changes = []
        check_func = self.health_config.check_func
        if not check_func:
            return changes

        for b in list(self.backends.values()):
            if b.status == BackendStatus.DRAINING:
                continue
            if now - b.last_health_check < self.health_config.interval:
                continue

            b.last_health_check = now
            try:
                result = check_func(b)
                if result:
                    b.consecutive_failures = 0
                    if b.status == BackendStatus.UNHEALTHY:
                        b.total_successes += 1
                        if b.total_successes >= self.health_config.healthy_threshold:
                            b.status = BackendStatus.HEALTHY
                            changes.append((b.id, 'healthy'))
                else:
                    b.consecutive_failures += 1
                    if b.consecutive_failures >= self.health_config.unhealthy_threshold:
                        if b.status == BackendStatus.HEALTHY:
                            b.status = BackendStatus.UNHEALTHY
                            changes.append((b.id, 'unhealthy'))
            except Exception:
                b.consecutive_failures += 1
                if b.consecutive_failures >= self.health_config.unhealthy_threshold:
                    if b.status == BackendStatus.HEALTHY:
                        b.status = BackendStatus.UNHEALTHY
                        changes.append((b.id, 'unhealthy'))

        return changes

    def cleanup_draining(self, timeout=30.0, now=None):
        """Remove backends that have finished draining."""
        if now is None:
            now = time.time()
        removed = []
        for bid in list(self.backends.keys()):
            b = self.backends[bid]
            if b.status == BackendStatus.DRAINING:
                if b.active_connections == 0 or \
                   (now - b.drain_started) >= timeout:
                    del self.backends[bid]
                    removed.append(bid)
        return removed

    def get_metrics(self):
        """Get pool metrics."""
        backends = list(self.backends.values())
        healthy = [b for b in backends if b.status == BackendStatus.HEALTHY]
        return {
            'name': self.name,
            'algorithm': self.algorithm.name,
            'total_backends': len(backends),
            'healthy_backends': len(healthy),
            'total_connections': sum(b.active_connections for b in backends),
            'total_requests': sum(b.total_requests for b in backends),
            'backends': {b.id: b.to_dict() for b in backends},
            'sticky_sessions': len(self._sticky_map),
        }


# ---------------------------------------------------------------------------
# L7 Router
# ---------------------------------------------------------------------------

class L7Router:
    """Layer 7 request router based on path, headers, and method."""

    def __init__(self):
        self.rules = []  # Sorted by priority

    def add_rule(self, rule):
        """Add a routing rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_name):
        """Remove a routing rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]

    def route(self, request):
        """Route a request to a pool name. Returns pool name or None."""
        for rule in self.rules:
            if self._matches(rule, request):
                return rule.pool_name
        return None

    def _matches(self, rule, request):
        """Check if a request matches a rule."""
        # Method check
        if rule.methods and request.method not in rule.methods:
            return False

        # Path check
        if rule.match_type == RoutingMatchType.EXACT:
            if request.path != rule.path_pattern:
                return False
        elif rule.match_type == RoutingMatchType.PREFIX:
            if not request.path.startswith(rule.path_pattern):
                return False
        elif rule.match_type == RoutingMatchType.REGEX:
            import re
            if not re.match(rule.path_pattern, request.path):
                return False

        # Header check
        if rule.headers:
            for key, value in rule.headers.items():
                if request.headers.get(key) != value:
                    return False

        return True

    def get_rules(self):
        """Get all rules sorted by priority."""
        return list(self.rules)


# ---------------------------------------------------------------------------
# Load Balancer
# ---------------------------------------------------------------------------

class LoadBalancer:
    """
    Full L4/L7 load balancer composing:
    - Backend pools with multiple algorithms
    - Health checking (active + passive)
    - Sticky sessions
    - L7 routing rules
    - Connection draining
    - Circuit breaker integration (C225)
    - Service discovery integration (C222)
    """

    def __init__(self, default_pool_name="default",
                 default_algorithm=BalancingAlgorithm.ROUND_ROBIN):
        self.pools = {}
        self.router = L7Router()
        self.default_pool = default_pool_name
        self._request_log = []  # Recent request log for monitoring
        self._max_log = 1000
        self._callbacks = {
            'backend_healthy': [],
            'backend_unhealthy': [],
            'backend_added': [],
            'backend_removed': [],
            'request_completed': [],
            'request_failed': [],
            'no_backend': [],
        }
        self._circuit_breakers = {}  # backend_id -> CircuitBreaker state
        self._lock = threading.Lock()

        # Create default pool
        self.create_pool(default_pool_name, algorithm=default_algorithm)

    def create_pool(self, name, algorithm=BalancingAlgorithm.ROUND_ROBIN,
                    health_config=None, sticky_config=None):
        """Create a new backend pool."""
        pool = BackendPool(name, algorithm=algorithm,
                           health_config=health_config,
                           sticky_config=sticky_config)
        self.pools[name] = pool
        return pool

    def get_pool(self, name):
        """Get a pool by name."""
        if name not in self.pools:
            raise PoolNotFoundError(f"Pool '{name}' not found")
        return self.pools[name]

    def remove_pool(self, name):
        """Remove a pool."""
        if name in self.pools:
            del self.pools[name]
            # Remove routing rules for this pool
            self.router.rules = [r for r in self.router.rules
                                 if r.pool_name != name]
            return True
        return False

    def add_backend(self, backend_id, address, port, pool_name=None,
                    weight=1, max_connections=0, metadata=None):
        """Add a backend to a pool."""
        pool_name = pool_name or self.default_pool
        pool = self.get_pool(pool_name)
        backend = pool.add_backend(backend_id, address, port,
                                   weight=weight,
                                   max_connections=max_connections,
                                   metadata=metadata)
        self._emit('backend_added', backend_id=backend_id, pool=pool_name)
        return backend

    def remove_backend(self, backend_id, pool_name=None):
        """Remove a backend from a pool."""
        pool_name = pool_name or self.default_pool
        pool = self.get_pool(pool_name)
        result = pool.remove_backend(backend_id)
        if result:
            self._emit('backend_removed', backend_id=backend_id, pool=pool_name)
        return result

    def drain_backend(self, backend_id, pool_name=None, timeout=30.0):
        """Start draining a backend."""
        pool_name = pool_name or self.default_pool
        pool = self.get_pool(pool_name)
        return pool.drain_backend(backend_id, timeout)

    def add_routing_rule(self, name, pool_name, path_pattern="/",
                         match_type=RoutingMatchType.PREFIX,
                         methods=None, headers=None, priority=0):
        """Add an L7 routing rule."""
        rule = RoutingRule(
            name=name, pool_name=pool_name,
            match_type=match_type, path_pattern=path_pattern,
            methods=methods, headers=headers, priority=priority
        )
        self.router.add_rule(rule)
        return rule

    def remove_routing_rule(self, name):
        """Remove a routing rule."""
        self.router.remove_rule(name)

    def handle_request(self, request, handler=None, now=None):
        """
        Route and forward a request to a backend.

        Args:
            request: Request object
            handler: Callable(backend, request) -> Response
            now: Current time (for testing)

        Returns:
            Response object
        """
        if now is None:
            now = time.time()

        # L7 routing: find the right pool
        pool_name = self.router.route(request)
        if pool_name is None:
            pool_name = self.default_pool

        if pool_name not in self.pools:
            self._emit('no_backend', request=request, reason='pool_not_found')
            raise PoolNotFoundError(f"Pool '{pool_name}' not found")

        pool = self.pools[pool_name]

        # Select backend
        backend = pool.select_backend(request, now)
        if backend is None:
            self._emit('no_backend', request=request, pool=pool_name)
            raise NoHealthyBackendError(
                f"No healthy backend available in pool '{pool_name}'")

        # Check max connections
        if backend.max_connections > 0 and \
           backend.active_connections >= backend.max_connections:
            self._emit('no_backend', request=request, reason='overloaded')
            raise BackendOverloadedError(
                f"Backend '{backend.id}' at max connections")

        # Execute request
        backend.active_connections += 1
        start_time = now
        try:
            if handler:
                response = handler(backend, request)
            else:
                response = Response(status=200, backend_id=backend.id)

            elapsed = time.time() - start_time
            response.backend_id = backend.id
            response.response_time = elapsed

            pool.record_success(backend.id, elapsed)
            backend.active_connections = max(0, backend.active_connections - 1)

            # Add sticky session cookie to response if needed
            if pool.sticky_config.session_type == StickySessionType.COOKIE:
                response.headers['Set-Cookie'] = \
                    f"{pool.sticky_config.cookie_name}={backend.id}_{int(now)}"

            self._log_request(request, response, pool_name, backend.id, elapsed)
            self._emit('request_completed', request=request,
                       response=response, backend_id=backend.id)
            return response

        except Exception as e:
            backend.active_connections = max(0, backend.active_connections - 1)
            pool.record_failure(backend.id)
            self._emit('request_failed', request=request,
                       backend_id=backend.id, error=str(e))
            raise

    def handle_request_with_retry(self, request, handler=None,
                                  max_retries=2, now=None):
        """Handle request with automatic retry on failure."""
        if now is None:
            now = time.time()

        last_error = None
        tried = set()

        for attempt in range(max_retries + 1):
            pool_name = self.router.route(request) or self.default_pool
            if pool_name not in self.pools:
                raise PoolNotFoundError(f"Pool '{pool_name}' not found")

            pool = self.pools[pool_name]
            backend = pool.select_backend(request, now)

            # Skip already-tried backends if possible
            if backend and backend.id in tried:
                healthy = pool.get_healthy_backends()
                untried = [b for b in healthy if b.id not in tried]
                if untried:
                    backend = untried[0]

            if backend is None:
                raise NoHealthyBackendError(
                    f"No healthy backend in pool '{pool_name}'")

            tried.add(backend.id)
            backend.active_connections += 1
            start_time = time.time()

            try:
                if handler:
                    response = handler(backend, request)
                else:
                    response = Response(status=200, backend_id=backend.id)

                elapsed = time.time() - start_time
                response.backend_id = backend.id
                response.response_time = elapsed
                pool.record_success(backend.id, elapsed)
                backend.active_connections = max(0, backend.active_connections - 1)
                return response

            except Exception as e:
                backend.active_connections = max(0, backend.active_connections - 1)
                pool.record_failure(backend.id)
                last_error = e

        raise last_error

    def run_health_checks(self, now=None):
        """Run health checks across all pools."""
        all_changes = {}
        for name, pool in self.pools.items():
            changes = pool.run_health_checks(now)
            if changes:
                all_changes[name] = changes
                for backend_id, status in changes:
                    if status == 'healthy':
                        self._emit('backend_healthy',
                                   backend_id=backend_id, pool=name)
                    else:
                        self._emit('backend_unhealthy',
                                   backend_id=backend_id, pool=name)
        return all_changes

    def cleanup_draining(self, timeout=30.0, now=None):
        """Cleanup finished draining backends across all pools."""
        removed = {}
        for name, pool in self.pools.items():
            r = pool.cleanup_draining(timeout, now)
            if r:
                removed[name] = r
        return removed

    def on(self, event, callback):
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event, **kwargs):
        """Emit an event to callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                pass

    def _log_request(self, request, response, pool_name, backend_id, elapsed):
        """Log a request for monitoring."""
        entry = {
            'time': time.time(),
            'method': request.method,
            'path': request.path,
            'client_ip': request.client_ip,
            'pool': pool_name,
            'backend': backend_id,
            'status': response.status,
            'elapsed': elapsed,
        }
        self._request_log.append(entry)
        if len(self._request_log) > self._max_log:
            self._request_log = self._request_log[-self._max_log:]

    def get_metrics(self):
        """Get overall load balancer metrics."""
        pool_metrics = {}
        for name, pool in self.pools.items():
            pool_metrics[name] = pool.get_metrics()

        total_requests = sum(
            pm['total_requests'] for pm in pool_metrics.values())
        total_connections = sum(
            pm['total_connections'] for pm in pool_metrics.values())
        total_backends = sum(
            pm['total_backends'] for pm in pool_metrics.values())
        healthy_backends = sum(
            pm['healthy_backends'] for pm in pool_metrics.values())

        return {
            'pools': pool_metrics,
            'total_pools': len(self.pools),
            'total_backends': total_backends,
            'healthy_backends': healthy_backends,
            'total_connections': total_connections,
            'total_requests': total_requests,
            'routing_rules': len(self.router.rules),
            'request_log_size': len(self._request_log),
        }

    def get_request_log(self, limit=100):
        """Get recent request log entries."""
        return self._request_log[-limit:]


# ---------------------------------------------------------------------------
# Service Discovery Integration
# ---------------------------------------------------------------------------

class ServiceDiscoverySync:
    """
    Syncs service discovery (C222) instances with load balancer backend pools.
    Bridges ServiceRegistry -> BackendPool.
    """

    def __init__(self, load_balancer, registry=None):
        self.lb = load_balancer
        self.registry = registry
        self._service_pool_map = {}  # service_name -> pool_name
        self._synced_backends = {}   # service_name -> set of backend_ids

    def bind_service(self, service_name, pool_name=None, create_pool=True,
                     algorithm=BalancingAlgorithm.ROUND_ROBIN,
                     health_config=None, sticky_config=None):
        """Bind a service discovery service to a load balancer pool."""
        pool_name = pool_name or service_name
        if create_pool and pool_name not in self.lb.pools:
            self.lb.create_pool(pool_name, algorithm=algorithm,
                               health_config=health_config,
                               sticky_config=sticky_config)
        self._service_pool_map[service_name] = pool_name
        self._synced_backends[service_name] = set()

    def sync(self):
        """
        Sync all bound services from registry to load balancer pools.
        Returns dict of changes: {service_name: {added: [...], removed: [...]}}.
        """
        if not self.registry:
            return {}

        changes = {}
        for service_name, pool_name in self._service_pool_map.items():
            if pool_name not in self.lb.pools:
                continue

            pool = self.lb.pools[pool_name]

            # Get healthy instances from registry
            try:
                instances = self.registry.get_healthy_services(service_name)
            except Exception:
                instances = []

            current_ids = {inst.service_id for inst in instances}
            known_ids = self._synced_backends.get(service_name, set())

            added = []
            removed = []

            # Add new instances
            for inst in instances:
                if inst.service_id not in known_ids:
                    pool.add_backend(
                        inst.service_id, inst.address, inst.port,
                        weight=getattr(inst, 'weight', 1),
                        metadata=getattr(inst, 'metadata', {})
                    )
                    added.append(inst.service_id)

            # Remove disappeared instances
            for old_id in known_ids - current_ids:
                pool.remove_backend(old_id)
                removed.append(old_id)

            self._synced_backends[service_name] = current_ids

            if added or removed:
                changes[service_name] = {'added': added, 'removed': removed}

        return changes

    def get_binding_info(self):
        """Get current binding information."""
        return {
            svc: {
                'pool': pool,
                'synced_backends': list(self._synced_backends.get(svc, set()))
            }
            for svc, pool in self._service_pool_map.items()
        }


# ---------------------------------------------------------------------------
# Resilience Integration
# ---------------------------------------------------------------------------

class ResilientLoadBalancer:
    """
    Load balancer with circuit breaker integration (C225).
    Wraps each backend with a circuit breaker.
    """

    def __init__(self, load_balancer, circuit_breaker_factory=None):
        self.lb = load_balancer
        self._breakers = {}  # backend_id -> circuit breaker state
        self._factory = circuit_breaker_factory
        self._breaker_config = {
            'failure_threshold': 5,
            'recovery_timeout': 30.0,
            'half_open_max': 2,
        }

    def configure_breaker(self, failure_threshold=5, recovery_timeout=30.0,
                          half_open_max=2):
        """Configure default circuit breaker settings."""
        self._breaker_config = {
            'failure_threshold': failure_threshold,
            'recovery_timeout': recovery_timeout,
            'half_open_max': half_open_max,
        }

    def get_breaker_state(self, backend_id):
        """Get circuit breaker state for a backend."""
        if backend_id not in self._breakers:
            self._breakers[backend_id] = {
                'state': 'CLOSED',
                'failures': 0,
                'successes': 0,
                'last_failure': 0.0,
                'half_open_calls': 0,
            }
        return self._breakers[backend_id]

    def is_backend_available(self, backend_id, now=None):
        """Check if backend is available (circuit not open)."""
        if now is None:
            now = time.time()
        state = self.get_breaker_state(backend_id)

        if state['state'] == 'CLOSED':
            return True
        elif state['state'] == 'OPEN':
            # Check if recovery timeout has passed
            if now - state['last_failure'] >= self._breaker_config['recovery_timeout']:
                state['state'] = 'HALF_OPEN'
                state['half_open_calls'] = 0
                return True
            return False
        elif state['state'] == 'HALF_OPEN':
            return state['half_open_calls'] < self._breaker_config['half_open_max']
        return True

    def record_success(self, backend_id):
        """Record a success for circuit breaker."""
        state = self.get_breaker_state(backend_id)
        state['successes'] += 1
        if state['state'] == 'HALF_OPEN':
            state['half_open_calls'] += 1
            if state['successes'] >= self._breaker_config['half_open_max']:
                state['state'] = 'CLOSED'
                state['failures'] = 0
                state['successes'] = 0
        elif state['state'] == 'CLOSED':
            state['failures'] = 0

    def record_failure(self, backend_id, now=None):
        """Record a failure for circuit breaker."""
        if now is None:
            now = time.time()
        state = self.get_breaker_state(backend_id)
        state['failures'] += 1
        state['last_failure'] = now

        if state['state'] == 'HALF_OPEN':
            state['state'] = 'OPEN'
            state['successes'] = 0
        elif state['state'] == 'CLOSED':
            if state['failures'] >= self._breaker_config['failure_threshold']:
                state['state'] = 'OPEN'
                state['successes'] = 0

    def handle_request(self, request, handler=None, now=None):
        """Handle request with circuit breaker protection."""
        if now is None:
            now = time.time()

        pool_name = self.lb.router.route(request) or self.lb.default_pool
        if pool_name not in self.lb.pools:
            raise PoolNotFoundError(f"Pool '{pool_name}' not found")

        pool = self.lb.pools[pool_name]
        healthy = pool.get_healthy_backends()

        # Filter by circuit breaker state
        available = [b for b in healthy if self.is_backend_available(b.id, now)]

        if not available:
            raise NoHealthyBackendError(
                f"No available backend in pool '{pool_name}' "
                f"(all circuits open or unhealthy)")

        # Use pool's algorithm on available backends
        backend = pool._select_by_algorithm(available, request)
        if backend is None:
            raise NoHealthyBackendError("No backend selected")

        backend.active_connections += 1
        start_time = time.time()

        try:
            if handler:
                response = handler(backend, request)
            else:
                response = Response(status=200, backend_id=backend.id)

            elapsed = time.time() - start_time
            response.backend_id = backend.id
            response.response_time = elapsed

            pool.record_success(backend.id, elapsed)
            self.record_success(backend.id)
            backend.active_connections = max(0, backend.active_connections - 1)
            return response

        except Exception as e:
            backend.active_connections = max(0, backend.active_connections - 1)
            pool.record_failure(backend.id)
            self.record_failure(backend.id)
            raise

    def get_breaker_metrics(self):
        """Get all circuit breaker states."""
        return {bid: dict(state) for bid, state in self._breakers.items()}


# ---------------------------------------------------------------------------
# Weighted Health Score
# ---------------------------------------------------------------------------

class WeightedHealthScorer:
    """
    Computes a composite health score for backends using multiple signals:
    response time, error rate, active connections, custom metrics.
    """

    def __init__(self, response_time_weight=0.3, error_rate_weight=0.4,
                 connection_weight=0.2, custom_weight=0.1):
        self.weights = {
            'response_time': response_time_weight,
            'error_rate': error_rate_weight,
            'connections': connection_weight,
            'custom': custom_weight,
        }
        self._custom_scorers = {}  # backend_id -> callable returning 0-1

    def set_custom_scorer(self, backend_id, scorer):
        """Set a custom health scorer for a backend."""
        self._custom_scorers[backend_id] = scorer

    def score(self, backend, max_response_time=5.0, max_connections=100):
        """
        Score a backend 0.0 (worst) to 1.0 (best).
        """
        # Response time score (lower is better)
        if backend.avg_response_time <= 0:
            rt_score = 1.0
        else:
            rt_score = max(0, 1.0 - (backend.avg_response_time / max_response_time))

        # Error rate score (lower is better)
        err_score = 1.0 - backend.failure_rate

        # Connection score (lower is better)
        if max_connections > 0:
            conn_score = max(0, 1.0 - (backend.active_connections / max_connections))
        else:
            conn_score = 1.0

        # Custom score
        custom_fn = self._custom_scorers.get(backend.id)
        custom_score = custom_fn(backend) if custom_fn else 1.0

        total = (rt_score * self.weights['response_time'] +
                 err_score * self.weights['error_rate'] +
                 conn_score * self.weights['connections'] +
                 custom_score * self.weights['custom'])

        return round(total, 4)

    def rank_backends(self, backends, max_response_time=5.0, max_connections=100):
        """Rank backends by health score (best first)."""
        scored = [(b, self.score(b, max_response_time, max_connections))
                  for b in backends]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Load Balancer System (top-level facade)
# ---------------------------------------------------------------------------

class LoadBalancerSystem:
    """
    Top-level system combining all load balancer capabilities.
    """

    def __init__(self, default_algorithm=BalancingAlgorithm.ROUND_ROBIN):
        self.lb = LoadBalancer(default_algorithm=default_algorithm)
        self.resilient = ResilientLoadBalancer(self.lb)
        self.scorer = WeightedHealthScorer()
        self._discovery_sync = None

    def create_pool(self, name, **kwargs):
        return self.lb.create_pool(name, **kwargs)

    def add_backend(self, backend_id, address, port, **kwargs):
        return self.lb.add_backend(backend_id, address, port, **kwargs)

    def remove_backend(self, backend_id, **kwargs):
        return self.lb.remove_backend(backend_id, **kwargs)

    def drain_backend(self, backend_id, **kwargs):
        return self.lb.drain_backend(backend_id, **kwargs)

    def add_routing_rule(self, name, pool_name, **kwargs):
        return self.lb.add_routing_rule(name, pool_name, **kwargs)

    def handle_request(self, request, handler=None, use_circuit_breaker=False,
                       now=None):
        """Handle a request, optionally with circuit breaker protection."""
        if use_circuit_breaker:
            return self.resilient.handle_request(request, handler, now)
        return self.lb.handle_request(request, handler, now)

    def handle_request_with_retry(self, request, handler=None,
                                  max_retries=2, now=None):
        return self.lb.handle_request_with_retry(request, handler,
                                                 max_retries, now)

    def bind_service_discovery(self, registry):
        """Bind a service registry for auto-syncing backends."""
        self._discovery_sync = ServiceDiscoverySync(self.lb, registry)
        return self._discovery_sync

    def sync_services(self):
        """Sync backends from service discovery."""
        if self._discovery_sync:
            return self._discovery_sync.sync()
        return {}

    def run_health_checks(self, now=None):
        return self.lb.run_health_checks(now)

    def score_backends(self, pool_name=None):
        """Score and rank backends in a pool."""
        pool_name = pool_name or self.lb.default_pool
        pool = self.lb.get_pool(pool_name)
        backends = pool.get_healthy_backends()
        return self.scorer.rank_backends(backends)

    def configure_circuit_breaker(self, **kwargs):
        self.resilient.configure_breaker(**kwargs)

    def get_metrics(self):
        metrics = self.lb.get_metrics()
        metrics['circuit_breakers'] = self.resilient.get_breaker_metrics()
        if self._discovery_sync:
            metrics['service_bindings'] = self._discovery_sync.get_binding_info()
        return metrics

    def on(self, event, callback):
        self.lb.on(event, callback)
