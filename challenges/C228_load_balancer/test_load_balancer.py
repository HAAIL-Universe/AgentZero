"""Tests for C228: Load Balancer"""

import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(__file__))

from load_balancer import (
    Backend, BackendPool, BackendStatus, BalancingAlgorithm,
    HealthCheckConfig, HealthCheckType, StickySessionConfig, StickySessionType,
    RoutingRule, RoutingMatchType, Request, Response,
    L7Router, LoadBalancer, ServiceDiscoverySync, ResilientLoadBalancer,
    WeightedHealthScorer, LoadBalancerSystem,
    NoHealthyBackendError, BackendOverloadedError, PoolNotFoundError,
)


# ============================================================
# Backend basics
# ============================================================

def test_backend_creation():
    b = Backend(id="b1", address="10.0.0.1", port=8080)
    assert b.id == "b1"
    assert b.address == "10.0.0.1"
    assert b.port == 8080
    assert b.weight == 1
    assert b.status == BackendStatus.HEALTHY
    assert b.active_connections == 0

def test_backend_failure_rate():
    b = Backend(id="b1", address="10.0.0.1", port=8080)
    assert b.failure_rate == 0.0
    b.total_successes = 8
    b.total_failures = 2
    assert abs(b.failure_rate - 0.2) < 1e-9

def test_backend_avg_response_time():
    b = Backend(id="b1", address="10.0.0.1", port=8080)
    assert b.avg_response_time == 0.0
    b.response_times = [0.1, 0.2, 0.3]
    assert abs(b.avg_response_time - 0.2) < 1e-9

def test_backend_to_dict():
    b = Backend(id="b1", address="10.0.0.1", port=8080, weight=3)
    d = b.to_dict()
    assert d['id'] == "b1"
    assert d['weight'] == 3
    assert d['status'] == 'HEALTHY'


# ============================================================
# Backend Pool - basics
# ============================================================

def test_pool_create():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN)
    assert pool.name == "web"
    assert pool.algorithm == BalancingAlgorithm.ROUND_ROBIN
    assert len(pool.backends) == 0

def test_pool_add_remove_backend():
    pool = BackendPool("web")
    b = pool.add_backend("b1", "10.0.0.1", 8080)
    assert "b1" in pool.backends
    assert b.address == "10.0.0.1"
    pool.remove_backend("b1")
    assert "b1" not in pool.backends

def test_pool_healthy_backends():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.backends["b2"].status = BackendStatus.UNHEALTHY
    healthy = pool.get_healthy_backends()
    assert len(healthy) == 1
    assert healthy[0].id == "b1"


# ============================================================
# Round Robin
# ============================================================

def test_round_robin():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.add_backend("b3", "10.0.0.3", 8080)

    selections = [pool.select_backend().id for _ in range(6)]
    # Should cycle through all three
    assert selections[0] == selections[3]
    assert selections[1] == selections[4]
    assert selections[2] == selections[5]
    assert len(set(selections[:3])) == 3  # All three selected

def test_round_robin_skips_unhealthy():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.backends["b2"].status = BackendStatus.UNHEALTHY

    selections = [pool.select_backend().id for _ in range(4)]
    assert all(s == "b1" for s in selections)


# ============================================================
# Weighted Round Robin
# ============================================================

def test_weighted_round_robin():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.WEIGHTED_ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080, weight=5)
    pool.add_backend("b2", "10.0.0.2", 8080, weight=3)
    pool.add_backend("b3", "10.0.0.3", 8080, weight=2)

    counts = {"b1": 0, "b2": 0, "b3": 0}
    for _ in range(100):
        b = pool.select_backend()
        counts[b.id] += 1

    # Weight ratio should be approximately 5:3:2
    assert counts["b1"] > counts["b2"] > counts["b3"]
    assert counts["b1"] == 50  # Smooth WRR is exact over full cycles

def test_weighted_round_robin_smooth():
    """Smooth WRR should interleave, not batch."""
    pool = BackendPool("web", algorithm=BalancingAlgorithm.WEIGHTED_ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080, weight=3)
    pool.add_backend("b2", "10.0.0.2", 8080, weight=1)

    selections = [pool.select_backend().id for _ in range(8)]
    # Should not be all b1 then all b2; b2 should be interspersed
    # In 8 selections: 6 b1, 2 b2, and b2 should appear at different positions
    b2_positions = [i for i, s in enumerate(selections) if s == "b2"]
    assert len(b2_positions) == 2
    # b2 should not be consecutive (smooth distribution)
    assert b2_positions[1] - b2_positions[0] > 1


# ============================================================
# Least Connections
# ============================================================

def test_least_connections():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.LEAST_CONNECTIONS)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.backends["b1"].active_connections = 5
    pool.backends["b2"].active_connections = 2

    b = pool.select_backend()
    assert b.id == "b2"

def test_least_connections_tie_breaking():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.LEAST_CONNECTIONS)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    # Both at 0 connections - should pick first (min)
    b = pool.select_backend()
    assert b.id in ("b1", "b2")


# ============================================================
# IP Hash
# ============================================================

def test_ip_hash_consistent():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.IP_HASH)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    req = Request(client_ip="192.168.1.100")
    selections = [pool.select_backend(req).id for _ in range(10)]
    # Same IP should always go to same backend
    assert len(set(selections)) == 1

def test_ip_hash_distribution():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.IP_HASH)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    # Different IPs should distribute
    backends = set()
    for i in range(20):
        req = Request(client_ip=f"192.168.1.{i}")
        b = pool.select_backend(req)
        backends.add(b.id)
    assert len(backends) == 2  # Both backends should be used


# ============================================================
# Random
# ============================================================

def test_random_selection():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.RANDOM)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.add_backend("b3", "10.0.0.3", 8080)

    selections = set()
    for _ in range(100):
        b = pool.select_backend()
        selections.add(b.id)
    assert len(selections) == 3  # All backends should be selected


# ============================================================
# Power of Two
# ============================================================

def test_power_of_two():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.POWER_OF_TWO)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.add_backend("b3", "10.0.0.3", 8080)

    # Set different loads
    pool.backends["b1"].active_connections = 10
    pool.backends["b2"].active_connections = 1
    pool.backends["b3"].active_connections = 5

    counts = {"b1": 0, "b2": 0, "b3": 0}
    for _ in range(300):
        b = pool.select_backend()
        counts[b.id] += 1

    # b2 (least loaded) should be picked most often
    assert counts["b2"] > counts["b1"]
    assert counts["b2"] > counts["b3"]

def test_power_of_two_single():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.POWER_OF_TWO)
    pool.add_backend("b1", "10.0.0.1", 8080)
    b = pool.select_backend()
    assert b.id == "b1"


# ============================================================
# Max connections
# ============================================================

def test_max_connections_respected():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080, max_connections=2)
    pool.add_backend("b2", "10.0.0.2", 8080)

    pool.backends["b1"].active_connections = 2
    # b1 is at max, should only select b2
    for _ in range(5):
        b = pool.select_backend()
        assert b.id == "b2"

def test_all_at_max_connections():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080, max_connections=1)
    pool.backends["b1"].active_connections = 1
    b = pool.select_backend()
    assert b is None


# ============================================================
# Sticky sessions
# ============================================================

def test_sticky_ip_hash():
    sticky = StickySessionConfig(
        session_type=StickySessionType.IP_HASH, ttl=3600)
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN,
                       sticky_config=sticky)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    req = Request(client_ip="192.168.1.50")
    first = pool.select_backend(req, now=1000.0)

    # Same IP should get same backend
    for _ in range(10):
        b = pool.select_backend(req, now=1001.0)
        assert b.id == first.id

def test_sticky_ip_expires():
    sticky = StickySessionConfig(
        session_type=StickySessionType.IP_HASH, ttl=10)
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN,
                       sticky_config=sticky)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    req = Request(client_ip="192.168.1.50")
    first = pool.select_backend(req, now=1000.0)
    # After TTL, sticky binding should expire
    pool.select_backend(req, now=1020.0)
    # May or may not be same backend, but binding was cleared

def test_sticky_cookie():
    sticky = StickySessionConfig(
        session_type=StickySessionType.COOKIE,
        cookie_name="LBSESS", ttl=3600)
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN,
                       sticky_config=sticky)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    # First request without cookie
    req1 = Request(client_ip="192.168.1.50")
    first = pool.select_backend(req1, now=1000.0)
    assert first is not None

    # Request with cookie
    req2 = Request(
        client_ip="192.168.1.50",
        headers={'Cookie': f'LBSESS={first.id}_1000'}
    )
    second = pool.select_backend(req2, now=1001.0)
    assert second.id == first.id

def test_sticky_unhealthy_backend_fallback():
    sticky = StickySessionConfig(
        session_type=StickySessionType.IP_HASH, ttl=3600)
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN,
                       sticky_config=sticky)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    req = Request(client_ip="192.168.1.50")
    first = pool.select_backend(req, now=1000.0)

    # Make sticky backend unhealthy
    pool.backends[first.id].status = BackendStatus.UNHEALTHY

    # Should fall back to other backend
    second = pool.select_backend(req, now=1001.0)
    assert second is not None
    assert second.id != first.id


# ============================================================
# Health checks
# ============================================================

def test_passive_health_check():
    hc = HealthCheckConfig(check_type=HealthCheckType.PASSIVE,
                           unhealthy_threshold=3)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)

    # Record failures
    for _ in range(3):
        pool.record_failure("b1")

    assert pool.backends["b1"].status == BackendStatus.UNHEALTHY

def test_active_health_check():
    check_results = {"b1": True, "b2": False}
    def checker(backend):
        return check_results.get(backend.id, False)

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, unhealthy_threshold=1,
                           healthy_threshold=1, check_func=checker)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)

    changes = pool.run_health_checks(now=100.0)
    # b2 should become unhealthy
    assert any(bid == "b2" and status == "unhealthy"
               for bid, status in changes)

def test_active_health_recovery():
    call_count = [0]
    def checker(backend):
        call_count[0] += 1
        return True  # Always healthy

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, unhealthy_threshold=1,
                           healthy_threshold=1, check_func=checker)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.backends["b1"].status = BackendStatus.UNHEALTHY
    pool.backends["b1"].total_successes = 0

    changes = pool.run_health_checks(now=100.0)
    assert any(bid == "b1" and status == "healthy"
               for bid, status in changes)

def test_health_check_interval():
    """Health checks should respect interval."""
    def checker(backend):
        return True

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=10.0, check_func=checker)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)

    pool.run_health_checks(now=100.0)
    # Too soon - should not recheck
    pool.backends["b1"].status = BackendStatus.UNHEALTHY
    pool.run_health_checks(now=105.0)
    # Still unhealthy because interval not reached
    assert pool.backends["b1"].status == BackendStatus.UNHEALTHY

def test_draining_skips_health_check():
    call_count = [0]
    def checker(backend):
        call_count[0] += 1
        return True

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, check_func=checker)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.backends["b1"].status = BackendStatus.DRAINING

    pool.run_health_checks(now=100.0)
    assert call_count[0] == 0


# ============================================================
# Connection draining
# ============================================================

def test_drain_backend():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.drain_backend("b1")
    assert pool.backends["b1"].status == BackendStatus.DRAINING
    # Draining backend should not be selected
    b = pool.select_backend()
    assert b is None

def test_drain_cleanup():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.drain_backend("b1")
    pool.backends["b1"].drain_started = 1000.0
    pool.backends["b1"].active_connections = 0

    removed = pool.cleanup_draining(timeout=30.0, now=1001.0)
    assert "b1" in removed

def test_drain_timeout():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.drain_backend("b1")
    pool.backends["b1"].drain_started = 1000.0
    pool.backends["b1"].active_connections = 5  # Still has connections

    removed = pool.cleanup_draining(timeout=30.0, now=1035.0)
    assert "b1" in removed  # Force removed after timeout


# ============================================================
# Record success/failure
# ============================================================

def test_record_success():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.record_success("b1", response_time=0.05)
    b = pool.backends["b1"]
    assert b.total_successes == 1
    assert b.total_requests == 1
    assert len(b.response_times) == 1

def test_record_failure_passive():
    hc = HealthCheckConfig(check_type=HealthCheckType.PASSIVE,
                           unhealthy_threshold=2)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)

    pool.record_failure("b1")
    assert pool.backends["b1"].status == BackendStatus.HEALTHY
    pool.record_failure("b1")
    assert pool.backends["b1"].status == BackendStatus.UNHEALTHY

def test_response_time_capped():
    pool = BackendPool("web", max_response_times=5)
    pool.add_backend("b1", "10.0.0.1", 8080)
    for i in range(10):
        pool.record_success("b1", response_time=float(i))
    assert len(pool.backends["b1"].response_times) == 5

def test_pool_metrics():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.add_backend("b2", "10.0.0.2", 8080)
    pool.record_success("b1", 0.1)
    pool.record_failure("b2")

    m = pool.get_metrics()
    assert m['total_backends'] == 2
    assert m['total_requests'] == 2
    assert m['backends']['b1']['failure_rate'] == 0.0


# ============================================================
# L7 Router
# ============================================================

def test_router_prefix_match():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="api", pool_name="api-pool",
        match_type=RoutingMatchType.PREFIX, path_pattern="/api"))

    req = Request(path="/api/users")
    assert router.route(req) == "api-pool"

def test_router_exact_match():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="health", pool_name="health-pool",
        match_type=RoutingMatchType.EXACT, path_pattern="/health"))

    assert router.route(Request(path="/health")) == "health-pool"
    assert router.route(Request(path="/health/deep")) is None

def test_router_method_filter():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="post-api", pool_name="write-pool",
        match_type=RoutingMatchType.PREFIX, path_pattern="/api",
        methods=["POST", "PUT"]))

    assert router.route(Request(method="POST", path="/api/data")) == "write-pool"
    assert router.route(Request(method="GET", path="/api/data")) is None

def test_router_header_filter():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="v2", pool_name="v2-pool",
        match_type=RoutingMatchType.PREFIX, path_pattern="/api",
        headers={"X-API-Version": "2"}))

    req_v2 = Request(path="/api/data", headers={"X-API-Version": "2"})
    req_v1 = Request(path="/api/data", headers={"X-API-Version": "1"})
    assert router.route(req_v2) == "v2-pool"
    assert router.route(req_v1) is None

def test_router_priority():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="general", pool_name="general-pool",
        match_type=RoutingMatchType.PREFIX, path_pattern="/",
        priority=0))
    router.add_rule(RoutingRule(
        name="api", pool_name="api-pool",
        match_type=RoutingMatchType.PREFIX, path_pattern="/api",
        priority=10))

    req = Request(path="/api/data")
    assert router.route(req) == "api-pool"  # Higher priority wins

def test_router_regex_match():
    router = L7Router()
    router.add_rule(RoutingRule(
        name="user-id", pool_name="user-pool",
        match_type=RoutingMatchType.REGEX,
        path_pattern=r"/users/\d+"))

    assert router.route(Request(path="/users/123")) == "user-pool"
    assert router.route(Request(path="/users/abc")) is None

def test_router_remove_rule():
    router = L7Router()
    router.add_rule(RoutingRule(name="r1", pool_name="p1"))
    router.add_rule(RoutingRule(name="r2", pool_name="p2"))
    router.remove_rule("r1")
    assert len(router.get_rules()) == 1
    assert router.get_rules()[0].name == "r2"

def test_router_no_match():
    router = L7Router()
    req = Request(path="/unknown")
    assert router.route(req) is None


# ============================================================
# Load Balancer - core
# ============================================================

def test_lb_create():
    lb = LoadBalancer()
    assert "default" in lb.pools
    assert lb.default_pool == "default"

def test_lb_add_backend():
    lb = LoadBalancer()
    b = lb.add_backend("b1", "10.0.0.1", 8080)
    assert b.id == "b1"
    assert "b1" in lb.pools["default"].backends

def test_lb_handle_request():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    req = Request(method="GET", path="/")
    resp = lb.handle_request(req)
    assert resp.backend_id == "b1"
    assert resp.status == 200

def test_lb_handle_with_handler():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    def my_handler(backend, request):
        return Response(status=201, body=f"handled by {backend.id}")

    req = Request(method="GET", path="/")
    resp = lb.handle_request(req, handler=my_handler)
    assert resp.status == 201
    assert resp.body == "handled by b1"

def test_lb_no_backend_error():
    lb = LoadBalancer()
    req = Request()
    try:
        lb.handle_request(req)
        assert False, "Should raise"
    except NoHealthyBackendError:
        pass

def test_lb_multiple_pools():
    lb = LoadBalancer()
    lb.create_pool("api", algorithm=BalancingAlgorithm.LEAST_CONNECTIONS)
    lb.add_backend("web1", "10.0.0.1", 80)
    lb.add_backend("api1", "10.0.0.2", 8080, pool_name="api")

    lb.add_routing_rule("api-route", "api", path_pattern="/api")

    req_web = Request(path="/index.html")
    req_api = Request(path="/api/users")

    resp_web = lb.handle_request(req_web)
    resp_api = lb.handle_request(req_api)

    assert resp_web.backend_id == "web1"
    assert resp_api.backend_id == "api1"

def test_lb_pool_not_found():
    lb = LoadBalancer()
    try:
        lb.get_pool("nonexistent")
        assert False, "Should raise"
    except PoolNotFoundError:
        pass

def test_lb_remove_pool():
    lb = LoadBalancer()
    lb.create_pool("temp")
    lb.add_routing_rule("temp-rule", "temp")
    lb.remove_pool("temp")
    assert "temp" not in lb.pools
    assert len(lb.router.rules) == 0

def test_lb_remove_backend():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    assert lb.remove_backend("b1")
    assert "b1" not in lb.pools["default"].backends

def test_lb_drain_backend():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.add_backend("b2", "10.0.0.2", 8080)
    lb.drain_backend("b1")
    # Only b2 should serve
    for _ in range(5):
        resp = lb.handle_request(Request())
        assert resp.backend_id == "b2"

def test_lb_handler_failure():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    def failing_handler(backend, request):
        raise RuntimeError("connection refused")

    try:
        lb.handle_request(Request(), handler=failing_handler)
        assert False
    except RuntimeError:
        pass

    # Failure should be recorded
    b = lb.pools["default"].backends["b1"]
    assert b.total_failures == 1
    assert b.active_connections == 0  # Released on failure


# ============================================================
# Request with retry
# ============================================================

def test_retry_on_failure():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.add_backend("b2", "10.0.0.2", 8080)

    fail_count = [0]
    def flaky_handler(backend, request):
        fail_count[0] += 1
        if fail_count[0] <= 1:
            raise RuntimeError("fail")
        return Response(status=200)

    resp = lb.handle_request_with_retry(Request(), handler=flaky_handler,
                                        max_retries=2)
    assert resp.status == 200

def test_retry_exhausted():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    def always_fail(backend, request):
        raise RuntimeError("always fails")

    try:
        lb.handle_request_with_retry(Request(), handler=always_fail,
                                     max_retries=2)
        assert False
    except RuntimeError:
        pass


# ============================================================
# Event callbacks
# ============================================================

def test_callbacks():
    lb = LoadBalancer()
    events = []
    lb.on('backend_added', lambda **kw: events.append(('added', kw)))
    lb.on('backend_removed', lambda **kw: events.append(('removed', kw)))

    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.remove_backend("b1")

    assert len(events) == 2
    assert events[0][0] == 'added'
    assert events[1][0] == 'removed'

def test_request_callbacks():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    completed = []
    lb.on('request_completed', lambda **kw: completed.append(kw))

    lb.handle_request(Request())
    assert len(completed) == 1
    assert completed[0]['backend_id'] == "b1"


# ============================================================
# Request logging
# ============================================================

def test_request_log():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)

    for _ in range(5):
        lb.handle_request(Request(path="/test"))

    log = lb.get_request_log()
    assert len(log) == 5
    assert log[0]['path'] == "/test"
    assert log[0]['backend'] == "b1"

def test_request_log_limit():
    lb = LoadBalancer()
    lb._max_log = 5
    lb.add_backend("b1", "10.0.0.1", 8080)

    for _ in range(10):
        lb.handle_request(Request())

    log = lb.get_request_log()
    assert len(log) == 5


# ============================================================
# Metrics
# ============================================================

def test_lb_metrics():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.add_backend("b2", "10.0.0.2", 8080)
    lb.add_routing_rule("r1", "default")

    lb.handle_request(Request())

    m = lb.get_metrics()
    assert m['total_pools'] == 1
    assert m['total_backends'] == 2
    assert m['healthy_backends'] == 2
    assert m['total_requests'] == 1
    assert m['routing_rules'] == 1


# ============================================================
# Health checks via LB
# ============================================================

def test_lb_health_checks():
    results = {"b1": True, "b2": False}
    def checker(b):
        return results.get(b.id, False)

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, unhealthy_threshold=1,
                           check_func=checker)
    lb = LoadBalancer()
    lb.pools["default"].health_config = hc
    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.add_backend("b2", "10.0.0.2", 8080)

    changes = lb.run_health_checks(now=100.0)
    assert "default" in changes

def test_lb_cleanup_draining():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    lb.drain_backend("b1")
    lb.pools["default"].backends["b1"].drain_started = 1000.0
    lb.pools["default"].backends["b1"].active_connections = 0

    removed = lb.cleanup_draining(timeout=30.0, now=1001.0)
    assert "default" in removed


# ============================================================
# Resilient Load Balancer (Circuit Breaker)
# ============================================================

def test_resilient_basic():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)

    resp = rlb.handle_request(Request())
    assert resp.backend_id == "b1"
    assert rlb.get_breaker_state("b1")['state'] == 'CLOSED'

def test_resilient_circuit_opens():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=3)

    # Record failures
    for _ in range(3):
        rlb.record_failure("b1")

    assert rlb.get_breaker_state("b1")['state'] == 'OPEN'
    assert not rlb.is_backend_available("b1")

def test_resilient_circuit_half_open():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=3, recovery_timeout=10.0)

    for _ in range(3):
        rlb.record_failure("b1", now=1000.0)

    assert not rlb.is_backend_available("b1", now=1005.0)
    # After recovery timeout
    assert rlb.is_backend_available("b1", now=1015.0)
    assert rlb.get_breaker_state("b1")['state'] == 'HALF_OPEN'

def test_resilient_circuit_recovers():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=3, recovery_timeout=10.0,
                          half_open_max=2)

    for _ in range(3):
        rlb.record_failure("b1", now=1000.0)

    # Transition to half-open
    rlb.is_backend_available("b1", now=1015.0)

    # Succeed in half-open
    rlb.record_success("b1")
    rlb.record_success("b1")

    assert rlb.get_breaker_state("b1")['state'] == 'CLOSED'

def test_resilient_circuit_half_open_failure():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=3, recovery_timeout=10.0)

    for _ in range(3):
        rlb.record_failure("b1", now=1000.0)

    rlb.is_backend_available("b1", now=1015.0)
    assert rlb.get_breaker_state("b1")['state'] == 'HALF_OPEN'

    rlb.record_failure("b1")
    assert rlb.get_breaker_state("b1")['state'] == 'OPEN'

def test_resilient_all_circuits_open():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=2)

    rlb.record_failure("b1")
    rlb.record_failure("b1")

    try:
        rlb.handle_request(Request())
        assert False
    except NoHealthyBackendError:
        pass

def test_resilient_handler_failure_opens_circuit():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.configure_breaker(failure_threshold=2)

    def fail_handler(b, r):
        raise RuntimeError("fail")

    for _ in range(2):
        try:
            rlb.handle_request(Request(), handler=fail_handler)
        except RuntimeError:
            pass

    assert rlb.get_breaker_state("b1")['state'] == 'OPEN'

def test_resilient_metrics():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080)
    rlb = ResilientLoadBalancer(lb)
    rlb.record_failure("b1")

    m = rlb.get_breaker_metrics()
    assert "b1" in m
    assert m["b1"]["failures"] == 1


# ============================================================
# Weighted Health Scorer
# ============================================================

def test_scorer_perfect():
    scorer = WeightedHealthScorer()
    b = Backend(id="b1", address="10.0.0.1", port=8080)
    score = scorer.score(b)
    assert score == 1.0

def test_scorer_high_error_rate():
    scorer = WeightedHealthScorer()
    b = Backend(id="b1", address="10.0.0.1", port=8080,
                total_successes=5, total_failures=5)
    score = scorer.score(b)
    assert score < 0.9  # Penalty for 50% error rate

def test_scorer_high_response_time():
    scorer = WeightedHealthScorer()
    b = Backend(id="b1", address="10.0.0.1", port=8080,
                response_times=[4.0, 4.5, 5.0])
    score = scorer.score(b, max_response_time=5.0)
    assert score < 0.9  # Penalty for high response time

def test_scorer_rank():
    scorer = WeightedHealthScorer()
    b1 = Backend(id="b1", address="10.0.0.1", port=8080)  # Perfect
    b2 = Backend(id="b2", address="10.0.0.2", port=8080,
                 total_successes=5, total_failures=5)  # Bad error rate
    b3 = Backend(id="b3", address="10.0.0.3", port=8080,
                 active_connections=50)  # High load

    ranked = scorer.rank_backends([b1, b2, b3], max_connections=100)
    assert ranked[0][0].id == "b1"  # Best first

def test_scorer_custom():
    scorer = WeightedHealthScorer()
    scorer.set_custom_scorer("b1", lambda b: 0.5)
    b = Backend(id="b1", address="10.0.0.1", port=8080)
    score = scorer.score(b)
    assert score < 1.0  # Custom scorer drags it down


# ============================================================
# Service Discovery Sync
# ============================================================

class MockInstance:
    def __init__(self, service_id, address, port, weight=1, metadata=None):
        self.service_id = service_id
        self.address = address
        self.port = port
        self.weight = weight
        self.metadata = metadata or {}

class MockRegistry:
    def __init__(self):
        self.services = {}

    def get_healthy_services(self, service_name):
        return self.services.get(service_name, [])

def test_sd_sync_adds_backends():
    lb = LoadBalancer()
    registry = MockRegistry()
    registry.services["web"] = [
        MockInstance("web-1", "10.0.0.1", 8080),
        MockInstance("web-2", "10.0.0.2", 8080),
    ]

    sync = ServiceDiscoverySync(lb, registry)
    sync.bind_service("web", pool_name="web-pool")
    changes = sync.sync()

    assert "web" in changes
    assert len(changes["web"]["added"]) == 2
    assert "web-1" in lb.pools["web-pool"].backends
    assert "web-2" in lb.pools["web-pool"].backends

def test_sd_sync_removes_backends():
    lb = LoadBalancer()
    registry = MockRegistry()
    registry.services["web"] = [
        MockInstance("web-1", "10.0.0.1", 8080),
        MockInstance("web-2", "10.0.0.2", 8080),
    ]

    sync = ServiceDiscoverySync(lb, registry)
    sync.bind_service("web")
    sync.sync()

    # Remove web-2 from registry
    registry.services["web"] = [
        MockInstance("web-1", "10.0.0.1", 8080),
    ]

    changes = sync.sync()
    assert "web-2" in changes["web"]["removed"]
    assert "web-2" not in lb.pools["web"].backends

def test_sd_sync_no_changes():
    lb = LoadBalancer()
    registry = MockRegistry()
    registry.services["web"] = [
        MockInstance("web-1", "10.0.0.1", 8080),
    ]

    sync = ServiceDiscoverySync(lb, registry)
    sync.bind_service("web")
    sync.sync()

    changes = sync.sync()
    assert len(changes) == 0

def test_sd_binding_info():
    lb = LoadBalancer()
    registry = MockRegistry()
    sync = ServiceDiscoverySync(lb, registry)
    sync.bind_service("web", pool_name="web-pool")

    info = sync.get_binding_info()
    assert "web" in info
    assert info["web"]["pool"] == "web-pool"


# ============================================================
# LoadBalancerSystem (facade)
# ============================================================

def test_system_basic():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)

    resp = sys.handle_request(Request())
    assert resp.backend_id == "b1"

def test_system_with_circuit_breaker():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)

    resp = sys.handle_request(Request(), use_circuit_breaker=True)
    assert resp.backend_id == "b1"

def test_system_retry():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.add_backend("b2", "10.0.0.2", 8080)

    call_count = [0]
    def flaky(backend, request):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("fail")
        return Response(status=200)

    resp = sys.handle_request_with_retry(Request(), handler=flaky)
    assert resp.status == 200

def test_system_service_discovery():
    sys = LoadBalancerSystem()
    registry = MockRegistry()
    registry.services["api"] = [
        MockInstance("api-1", "10.0.0.1", 9090),
    ]

    sd = sys.bind_service_discovery(registry)
    sd.bind_service("api")
    changes = sys.sync_services()
    assert "api" in changes

def test_system_scoring():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.add_backend("b2", "10.0.0.2", 8080)

    ranked = sys.score_backends()
    assert len(ranked) == 2

def test_system_metrics():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.handle_request(Request())

    m = sys.get_metrics()
    assert m['total_backends'] == 1
    assert m['total_requests'] == 1
    assert 'circuit_breakers' in m

def test_system_pool_management():
    sys = LoadBalancerSystem()
    sys.create_pool("api", algorithm=BalancingAlgorithm.LEAST_CONNECTIONS)
    sys.add_backend("api1", "10.0.0.1", 8080, pool_name="api")
    sys.add_routing_rule("api-route", "api", path_pattern="/api")

    resp = sys.handle_request(Request(path="/api/test"))
    assert resp.backend_id == "api1"

def test_system_drain():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.add_backend("b2", "10.0.0.2", 8080)
    sys.drain_backend("b1")

    for _ in range(5):
        resp = sys.handle_request(Request())
        assert resp.backend_id == "b2"

def test_system_remove():
    sys = LoadBalancerSystem()
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.remove_backend("b1")
    assert "b1" not in sys.lb.pools["default"].backends

def test_system_configure_breaker():
    sys = LoadBalancerSystem()
    sys.configure_circuit_breaker(failure_threshold=10)
    assert sys.resilient._breaker_config['failure_threshold'] == 10

def test_system_event_callback():
    sys = LoadBalancerSystem()
    events = []
    sys.on('backend_added', lambda **kw: events.append(kw))
    sys.add_backend("b1", "10.0.0.1", 8080)
    assert len(events) == 1

def test_system_health_checks():
    results = {"b1": True}
    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, check_func=lambda b: results.get(b.id, True))
    sys = LoadBalancerSystem()
    sys.lb.pools["default"].health_config = hc
    sys.add_backend("b1", "10.0.0.1", 8080)
    sys.run_health_checks(now=100.0)


# ============================================================
# Edge cases
# ============================================================

def test_empty_pool_select():
    pool = BackendPool("empty")
    assert pool.select_backend() is None

def test_all_unhealthy():
    pool = BackendPool("web")
    pool.add_backend("b1", "10.0.0.1", 8080)
    pool.backends["b1"].status = BackendStatus.UNHEALTHY
    assert pool.select_backend() is None

def test_record_nonexistent_backend():
    pool = BackendPool("web")
    pool.record_success("nonexistent")  # Should not raise
    pool.record_failure("nonexistent")  # Should not raise

def test_remove_nonexistent_backend():
    pool = BackendPool("web")
    assert not pool.remove_backend("nonexistent")

def test_drain_nonexistent():
    pool = BackendPool("web")
    assert not pool.drain_backend("nonexistent")

def test_cookie_sticky_no_cookie():
    sticky = StickySessionConfig(
        session_type=StickySessionType.COOKIE, cookie_name="SID")
    pool = BackendPool("web", algorithm=BalancingAlgorithm.ROUND_ROBIN,
                       sticky_config=sticky)
    pool.add_backend("b1", "10.0.0.1", 8080)

    # Request without cookie should still work
    req = Request()
    b = pool.select_backend(req, now=1000.0)
    assert b is not None

def test_multiple_routing_rules():
    lb = LoadBalancer()
    lb.create_pool("api")
    lb.create_pool("static")
    lb.create_pool("ws")

    lb.add_backend("api1", "10.0.0.1", 8080, pool_name="api")
    lb.add_backend("static1", "10.0.0.2", 80, pool_name="static")
    lb.add_backend("ws1", "10.0.0.3", 9090, pool_name="ws")

    lb.add_routing_rule("ws", "ws", path_pattern="/ws", priority=20)
    lb.add_routing_rule("api", "api", path_pattern="/api", priority=10)
    lb.add_routing_rule("static", "static", path_pattern="/static", priority=5)

    assert lb.handle_request(Request(path="/ws/chat")).backend_id == "ws1"
    assert lb.handle_request(Request(path="/api/data")).backend_id == "api1"
    assert lb.handle_request(Request(path="/static/img.png")).backend_id == "static1"

def test_lb_cookie_in_response():
    sticky = StickySessionConfig(
        session_type=StickySessionType.COOKIE,
        cookie_name="SESS")
    lb = LoadBalancer()
    lb.pools["default"].sticky_config = sticky
    lb.add_backend("b1", "10.0.0.1", 8080)

    resp = lb.handle_request(Request())
    assert 'Set-Cookie' in resp.headers
    assert resp.headers['Set-Cookie'].startswith('SESS=')

def test_overloaded_backend_error():
    lb = LoadBalancer()
    lb.add_backend("b1", "10.0.0.1", 8080, max_connections=1)
    lb.pools["default"].backends["b1"].active_connections = 1

    try:
        lb.handle_request(Request())
        assert False
    except (NoHealthyBackendError, BackendOverloadedError):
        pass

def test_weighted_rr_single_backend():
    pool = BackendPool("web", algorithm=BalancingAlgorithm.WEIGHTED_ROUND_ROBIN)
    pool.add_backend("b1", "10.0.0.1", 8080, weight=5)
    for _ in range(10):
        assert pool.select_backend().id == "b1"

def test_health_check_exception():
    """Health check function that raises should mark unhealthy."""
    def bad_check(b):
        raise ConnectionError("refused")

    hc = HealthCheckConfig(check_type=HealthCheckType.ACTIVE,
                           interval=5.0, unhealthy_threshold=1,
                           check_func=bad_check)
    pool = BackendPool("web", health_config=hc)
    pool.add_backend("b1", "10.0.0.1", 8080)

    changes = pool.run_health_checks(now=100.0)
    assert any(bid == "b1" for bid, _ in changes)

def test_sd_sync_empty_registry():
    lb = LoadBalancer()
    registry = MockRegistry()
    sync = ServiceDiscoverySync(lb, registry)
    sync.bind_service("web")
    changes = sync.sync()
    assert len(changes) == 0

def test_sd_sync_no_registry():
    lb = LoadBalancer()
    sync = ServiceDiscoverySync(lb)
    changes = sync.sync()
    assert len(changes) == 0


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    test_functions = [v for k, v in sorted(globals().items())
                      if k.startswith("test_") and callable(v)]

    passed = 0
    failed = 0
    errors = []

    for test_fn in test_functions:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*60}")
    print(f"  C228 Load Balancer: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, e in errors:
            import traceback
            print(f"\n  {name}:")
            traceback.print_exception(type(e), e, e.__traceback__)

    assert failed == 0, f"{failed} test(s) failed"
