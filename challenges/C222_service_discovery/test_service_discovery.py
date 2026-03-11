"""
Tests for C222: Service Discovery
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from service_discovery import (
    ServiceRegistry, ServiceResolver, LeaderElection, TagFilter,
    ServiceCatalog, GossipDiscovery, ServiceMesh,
    HealthCheck, HealthStatus, CheckType, LoadBalanceStrategy,
    ServiceEvent, ServiceInstance, WatchEntry
)


# =============================================================================
# ServiceRegistry Tests
# =============================================================================

class TestServiceRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")

    def test_register_basic(self):
        inst = self.reg.register("web", address="10.0.0.1", port=8080)
        self.assertEqual(inst.service_name, "web")
        self.assertEqual(inst.address, "10.0.0.1")
        self.assertEqual(inst.port, 8080)
        self.assertEqual(inst.node_id, "node-1")

    def test_register_with_id(self):
        inst = self.reg.register("web", service_id="web-1")
        self.assertEqual(inst.service_id, "web-1")

    def test_register_with_tags(self):
        inst = self.reg.register("web", tags=["v1", "production"])
        self.assertIn("v1", inst.tags)
        self.assertIn("production", inst.tags)

    def test_register_with_metadata(self):
        inst = self.reg.register("web", metadata={"region": "us-east"})
        self.assertEqual(inst.metadata["region"], "us-east")

    def test_register_multiple(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("web", service_id="web-2")
        instances = self.reg.get_services("web")
        self.assertEqual(len(instances), 2)

    def test_deregister(self):
        self.reg.register("web", service_id="web-1")
        self.assertTrue(self.reg.deregister("web-1"))
        self.assertIsNone(self.reg.get_service("web-1"))

    def test_deregister_nonexistent(self):
        self.assertFalse(self.reg.deregister("nope"))

    def test_get_service(self):
        self.reg.register("web", service_id="web-1")
        inst = self.reg.get_service("web-1")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.service_id, "web-1")

    def test_get_services_by_name(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("api", service_id="api-1")
        webs = self.reg.get_services("web")
        self.assertEqual(len(webs), 1)
        self.assertEqual(webs[0].service_name, "web")

    def test_get_services_with_tags(self):
        self.reg.register("web", service_id="web-1", tags=["v1"])
        self.reg.register("web", service_id="web-2", tags=["v2"])
        v1 = self.reg.get_services("web", tags=["v1"])
        self.assertEqual(len(v1), 1)
        self.assertEqual(v1[0].service_id, "web-1")

    def test_get_healthy_services(self):
        inst = self.reg.register("web", service_id="web-1")
        inst.health_status = HealthStatus.CRITICAL
        self.reg.register("web", service_id="web-2")
        healthy = self.reg.get_healthy_services("web")
        self.assertEqual(len(healthy), 1)
        self.assertEqual(healthy[0].service_id, "web-2")

    def test_get_all_services(self):
        self.reg.register("web")
        self.reg.register("api")
        self.reg.register("db")
        names = self.reg.get_all_services()
        self.assertEqual(set(names), {"web", "api", "db"})

    def test_update_metadata(self):
        self.reg.register("web", service_id="web-1", metadata={"v": "1"})
        self.reg.update_metadata("web-1", {"v": "2", "region": "eu"})
        inst = self.reg.get_service("web-1")
        self.assertEqual(inst.metadata["v"], "2")
        self.assertEqual(inst.metadata["region"], "eu")

    def test_update_metadata_nonexistent(self):
        self.assertFalse(self.reg.update_metadata("nope", {}))

    def test_update_tags(self):
        self.reg.register("web", service_id="web-1", tags=["v1"])
        self.reg.update_tags("web-1", ["v2", "canary"])
        inst = self.reg.get_service("web-1")
        self.assertEqual(inst.tags, ["v2", "canary"])

    def test_update_tags_nonexistent(self):
        self.assertFalse(self.reg.update_tags("nope", []))

    def test_deregister_cleans_by_name(self):
        self.reg.register("web", service_id="web-1")
        self.reg.deregister("web-1")
        self.assertEqual(self.reg.get_services("web"), [])
        self.assertNotIn("web", self.reg.by_name)

    def test_register_custom_weight(self):
        inst = self.reg.register("web", weight=5)
        self.assertEqual(inst.weight, 5)

    def test_version_increments(self):
        i1 = self.reg.register("web", service_id="web-1")
        i2 = self.reg.register("web", service_id="web-2")
        self.assertGreater(i2.version, i1.version)

    def test_register_custom_node(self):
        inst = self.reg.register("web", node_id="node-2")
        self.assertEqual(inst.node_id, "node-2")


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthChecks(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")
        self.current_time = 1000.0
        self.reg._time = lambda: self.current_time

    def test_ttl_check_passing(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=30.0)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.current_time = 1010.0
        changed = self.reg.run_health_checks(self.current_time)
        self.assertEqual(len(changed), 0)
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_ttl_check_expired(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=10.0,
                         critical_threshold=1)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.current_time = 1020.0
        changed = self.reg.run_health_checks(self.current_time)
        self.assertEqual(len(changed), 1)
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_ttl_heartbeat_resets(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=10.0,
                         critical_threshold=1)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.current_time = 1015.0
        self.reg.ttl_heartbeat("web-1", now=self.current_time)
        changed = self.reg.run_health_checks(self.current_time)
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_ttl_heartbeat_all_checks(self):
        hc1 = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=10.0)
        hc2 = HealthCheck(check_id="ttl-2", check_type=CheckType.TTL, ttl=10.0)
        self.reg.register("web", service_id="web-1", health_checks=[hc1, hc2])
        self.current_time = 1015.0
        result = self.reg.ttl_heartbeat("web-1", now=self.current_time)
        self.assertTrue(result)

    def test_ttl_heartbeat_nonexistent(self):
        self.assertFalse(self.reg.ttl_heartbeat("nope"))

    def test_ttl_heartbeat_specific_check(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=10.0)
        self.reg.register("web", service_id="web-1", health_checks=[hc])
        result = self.reg.ttl_heartbeat("web-1", check_id="ttl-1")
        self.assertTrue(result)

    def test_ttl_heartbeat_wrong_check_id(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=10.0)
        self.reg.register("web", service_id="web-1", health_checks=[hc])
        result = self.reg.ttl_heartbeat("web-1", check_id="wrong")
        self.assertFalse(result)

    def test_http_check_passing(self):
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: 200)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_http_check_warning(self):
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: 429)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.WARNING)

    def test_http_check_critical(self):
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: 500)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_http_check_error(self):
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: (_ for _ in ()).throw(Exception("conn refused")))
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_http_check_no_callback(self):
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_tcp_check_passing(self):
        hc = HealthCheck(check_id="tcp-1", check_type=CheckType.TCP,
                         callback=lambda: True)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_tcp_check_refused(self):
        hc = HealthCheck(check_id="tcp-1", check_type=CheckType.TCP,
                         callback=lambda: False)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_tcp_check_error(self):
        def boom():
            raise OSError("network")
        hc = HealthCheck(check_id="tcp-1", check_type=CheckType.TCP, callback=boom)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_tcp_check_no_callback(self):
        hc = HealthCheck(check_id="tcp-1", check_type=CheckType.TCP)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_script_check_passing(self):
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT,
                         callback=lambda: 0)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_script_check_warning(self):
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT,
                         callback=lambda: 1)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.WARNING)

    def test_script_check_critical(self):
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT,
                         callback=lambda: 2)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_script_check_returns_health_status(self):
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT,
                         callback=lambda: HealthStatus.WARNING)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.WARNING)

    def test_script_check_error(self):
        def boom():
            raise RuntimeError("script failed")
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT, callback=boom)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)

    def test_script_check_no_callback(self):
        hc = HealthCheck(check_id="sc-1", check_type=CheckType.SCRIPT)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_warning_threshold(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=5.0,
                         warning_threshold=2, critical_threshold=5)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.current_time = 1010.0  # TTL expired
        self.reg.run_health_checks(self.current_time)
        self.assertEqual(hc.consecutive_failures, 1)
        self.reg.run_health_checks(self.current_time)
        self.assertEqual(hc.consecutive_failures, 2)
        self.assertEqual(inst.health_status, HealthStatus.WARNING)

    def test_auto_deregister(self):
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=5.0,
                         critical_threshold=1, deregister_after=10.0, interval=5.0)
        self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.current_time = 1010.0
        # Run enough checks to exceed deregister_after
        for _ in range(3):
            self.reg.run_health_checks(self.current_time)
        self.assertIsNone(self.reg.get_service("web-1"))

    def test_multiple_checks_worst_wins(self):
        hc1 = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                          callback=lambda: 200)
        hc2 = HealthCheck(check_id="tcp-1", check_type=CheckType.TCP,
                          callback=lambda: False)
        inst = self.reg.register("web", service_id="web-1", health_checks=[hc1, hc2])
        self.reg.run_health_checks()
        self.assertEqual(inst.health_status, HealthStatus.CRITICAL)


# =============================================================================
# Watch Tests
# =============================================================================

class TestWatches(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")
        self.events = []

    def test_watch_register(self):
        self.reg.watch("web", lambda inst: self.events.append(("reg", inst.service_id)))
        self.reg.register("web", service_id="web-1")
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0], ("reg", "web-1"))

    def test_watch_deregister(self):
        self.reg.watch("web", lambda inst: self.events.append(("dereg", inst.service_id)))
        self.reg.register("web", service_id="web-1")
        self.reg.deregister("web-1")
        self.assertEqual(len(self.events), 2)

    def test_watch_health_change(self):
        t = 1000.0
        self.reg._time = lambda: t
        hc = HealthCheck(check_id="ttl-1", check_type=CheckType.TTL, ttl=5.0,
                         critical_threshold=1)
        self.reg.register("web", service_id="web-1", health_checks=[hc])
        self.reg.watch("web", lambda inst: self.events.append(inst.health_status))
        t = 1010.0
        self.reg._time = lambda: t
        self.reg.run_health_checks(t)
        self.assertIn(HealthStatus.CRITICAL, self.events)

    def test_watch_tag_filter(self):
        self.reg.watch("web", lambda inst: self.events.append(inst.service_id),
                       tags=["v1"])
        self.reg.register("web", service_id="web-1", tags=["v1"])
        self.reg.register("web", service_id="web-2", tags=["v2"])
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0], "web-1")

    def test_unwatch(self):
        wid = self.reg.watch("web", lambda inst: self.events.append(1))
        self.reg.register("web", service_id="web-1")
        self.reg.unwatch(wid)
        self.reg.register("web", service_id="web-2")
        self.assertEqual(len(self.events), 1)

    def test_unwatch_nonexistent(self):
        self.assertFalse(self.reg.unwatch("nope"))

    def test_watch_callback_error_doesnt_crash(self):
        def bad_cb(inst):
            raise RuntimeError("boom")
        self.reg.watch("web", bad_cb)
        self.reg.register("web", service_id="web-1")  # Should not raise

    def test_watch_different_service(self):
        self.reg.watch("api", lambda inst: self.events.append(1))
        self.reg.register("web", service_id="web-1")
        self.assertEqual(len(self.events), 0)


# =============================================================================
# KV Store Tests
# =============================================================================

class TestKVStore(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")

    def test_put_get(self):
        self.reg.kv_put("config/db/host", "localhost")
        self.assertEqual(self.reg.kv_get("config/db/host"), "localhost")

    def test_get_default(self):
        self.assertEqual(self.reg.kv_get("missing", "default"), "default")

    def test_delete(self):
        self.reg.kv_put("key", "value")
        self.assertTrue(self.reg.kv_delete("key"))
        self.assertIsNone(self.reg.kv_get("key"))

    def test_delete_nonexistent(self):
        self.assertFalse(self.reg.kv_delete("nope"))

    def test_list_prefix(self):
        self.reg.kv_put("config/db/host", "localhost")
        self.reg.kv_put("config/db/port", "5432")
        self.reg.kv_put("config/app/name", "myapp")
        result = self.reg.kv_list("config/db/")
        self.assertEqual(len(result), 2)
        self.assertIn("config/db/host", result)

    def test_list_empty_prefix(self):
        self.reg.kv_put("a", 1)
        self.reg.kv_put("b", 2)
        result = self.reg.kv_list("")
        self.assertEqual(len(result), 2)


# =============================================================================
# Event Log Tests
# =============================================================================

class TestEventLog(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")

    def test_register_event(self):
        self.reg.register("web", service_id="web-1")
        events = self.reg.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1], ServiceEvent.REGISTERED)

    def test_deregister_event(self):
        self.reg.register("web", service_id="web-1")
        self.reg.deregister("web-1")
        events = self.reg.get_events(event_type=ServiceEvent.DEREGISTERED)
        self.assertEqual(len(events), 1)

    def test_filter_by_service_name(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("api", service_id="api-1")
        events = self.reg.get_events(service_name="web")
        self.assertEqual(len(events), 1)

    def test_event_limit(self):
        for i in range(10):
            self.reg.register("web", service_id=f"web-{i}")
        events = self.reg.get_events(limit=3)
        self.assertEqual(len(events), 3)


# =============================================================================
# ServiceResolver Tests
# =============================================================================

class TestServiceResolver(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")
        self.resolver = ServiceResolver(self.reg)

    def test_resolve_round_robin(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("web", service_id="web-2")
        ids = set()
        for _ in range(4):
            inst = self.resolver.resolve("web", LoadBalanceStrategy.ROUND_ROBIN)
            ids.add(inst.service_id)
        self.assertEqual(ids, {"web-1", "web-2"})

    def test_resolve_random(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("web", service_id="web-2")
        inst = self.resolver.resolve("web", LoadBalanceStrategy.RANDOM)
        self.assertIn(inst.service_id, ["web-1", "web-2"])

    def test_resolve_least_connections(self):
        i1 = self.reg.register("web", service_id="web-1")
        i2 = self.reg.register("web", service_id="web-2")
        i1.connections = 10
        i2.connections = 2
        inst = self.resolver.resolve("web", LoadBalanceStrategy.LEAST_CONNECTIONS)
        self.assertEqual(inst.service_id, "web-2")

    def test_resolve_weighted(self):
        self.reg.register("web", service_id="web-1", weight=10)
        self.reg.register("web", service_id="web-2", weight=1)
        # Over many runs, web-1 should be chosen much more often
        counts = {"web-1": 0, "web-2": 0}
        for _ in range(100):
            inst = self.resolver.resolve("web", LoadBalanceStrategy.WEIGHTED)
            counts[inst.service_id] += 1
        self.assertGreater(counts["web-1"], counts["web-2"])

    def test_resolve_consistent_hash(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("web", service_id="web-2")
        # Same key should always resolve to same instance
        inst1 = self.resolver.resolve("web", LoadBalanceStrategy.CONSISTENT_HASH,
                                      hash_key="user-123")
        inst2 = self.resolver.resolve("web", LoadBalanceStrategy.CONSISTENT_HASH,
                                      hash_key="user-123")
        self.assertEqual(inst1.service_id, inst2.service_id)

    def test_resolve_consistent_hash_different_keys(self):
        self.reg.register("web", service_id="web-1")
        self.reg.register("web", service_id="web-2")
        self.reg.register("web", service_id="web-3")
        # Different keys should distribute
        results = set()
        for i in range(50):
            inst = self.resolver.resolve("web", LoadBalanceStrategy.CONSISTENT_HASH,
                                         hash_key=f"key-{i}")
            results.add(inst.service_id)
        self.assertGreater(len(results), 1)

    def test_resolve_no_instances(self):
        inst = self.resolver.resolve("web")
        self.assertIsNone(inst)

    def test_resolve_skips_unhealthy(self):
        i1 = self.reg.register("web", service_id="web-1")
        i1.health_status = HealthStatus.CRITICAL
        self.reg.register("web", service_id="web-2")
        inst = self.resolver.resolve("web")
        self.assertEqual(inst.service_id, "web-2")

    def test_resolve_with_tags(self):
        self.reg.register("web", service_id="web-1", tags=["v1"])
        self.reg.register("web", service_id="web-2", tags=["v2"])
        inst = self.resolver.resolve("web", tags=["v2"])
        self.assertEqual(inst.service_id, "web-2")

    def test_resolve_all(self):
        self.reg.register("web", service_id="web-1")
        i2 = self.reg.register("web", service_id="web-2")
        i2.health_status = HealthStatus.CRITICAL
        healthy = self.resolver.resolve_all("web")
        self.assertEqual(len(healthy), 1)
        all_inst = self.resolver.resolve_all("web", include_unhealthy=True)
        self.assertEqual(len(all_inst), 2)

    def test_resolve_weighted_zero_weights(self):
        self.reg.register("web", service_id="web-1", weight=0)
        self.reg.register("web", service_id="web-2", weight=0)
        inst = self.resolver.resolve("web", LoadBalanceStrategy.WEIGHTED)
        self.assertIsNotNone(inst)


# =============================================================================
# LeaderElection Tests
# =============================================================================

class TestLeaderElection(unittest.TestCase):
    def setUp(self):
        self.le = LeaderElection()

    def test_first_candidate_wins(self):
        self.assertTrue(self.le.campaign("election-1", "node-1"))
        self.assertEqual(self.le.get_leader("election-1"), "node-1")

    def test_second_candidate_loses(self):
        self.le.campaign("election-1", "node-1")
        self.assertFalse(self.le.campaign("election-1", "node-2"))

    def test_leader_can_campaign_again(self):
        self.le.campaign("election-1", "node-1")
        self.assertTrue(self.le.campaign("election-1", "node-1"))

    def test_resign_promotes_next(self):
        self.le.campaign("election-1", "node-1")
        self.le.campaign("election-1", "node-2")
        self.le.resign("election-1", "node-1")
        self.assertEqual(self.le.get_leader("election-1"), "node-2")

    def test_resign_nonexistent(self):
        self.assertFalse(self.le.resign("nope", "node-1"))

    def test_resign_non_leader(self):
        self.le.campaign("election-1", "node-1")
        self.assertFalse(self.le.resign("election-1", "node-2"))

    def test_resign_last_candidate(self):
        self.le.campaign("election-1", "node-1")
        self.le.resign("election-1", "node-1")
        self.assertIsNone(self.le.get_leader("election-1"))

    def test_is_leader(self):
        self.le.campaign("election-1", "node-1")
        self.assertTrue(self.le.is_leader("election-1", "node-1"))
        self.assertFalse(self.le.is_leader("election-1", "node-2"))

    def test_get_term(self):
        self.assertEqual(self.le.get_term("election-1"), 0)
        self.le.campaign("election-1", "node-1")
        self.assertEqual(self.le.get_term("election-1"), 1)
        self.le.campaign("election-1", "node-2")
        self.le.resign("election-1", "node-1")
        self.assertEqual(self.le.get_term("election-1"), 2)

    def test_watch_election(self):
        events = []
        self.le.watch_election("election-1", lambda e, n, l: events.append((e, l)))
        self.le.campaign("election-1", "node-1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (ServiceEvent.LEADER_ELECTED, "node-1"))

    def test_watch_leader_lost(self):
        events = []
        self.le.watch_election("election-1", lambda e, n, l: events.append(e))
        self.le.campaign("election-1", "node-1")
        self.le.resign("election-1", "node-1")
        self.assertIn(ServiceEvent.LEADER_LOST, events)

    def test_multiple_elections(self):
        self.le.campaign("db-leader", "node-1")
        self.le.campaign("cache-leader", "node-2")
        self.assertEqual(self.le.get_leader("db-leader"), "node-1")
        self.assertEqual(self.le.get_leader("cache-leader"), "node-2")

    def test_create_election_explicit(self):
        self.le.create_election("my-election")
        self.assertIn("my-election", self.le._elections)


# =============================================================================
# TagFilter Tests
# =============================================================================

class TestTagFilter(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")
        self.tf = TagFilter(self.reg)
        self.reg.register("web", service_id="web-1", tags=["v1", "production"],
                          metadata={"region": "us-east"})
        self.reg.register("web", service_id="web-2", tags=["v2", "staging"],
                          metadata={"region": "eu-west"})
        self.reg.register("api", service_id="api-1", tags=["v1", "production"],
                          metadata={"region": "us-east"})

    def test_filter_by_service_name(self):
        result = self.tf.filter(service_name="web")
        self.assertEqual(len(result), 2)

    def test_filter_by_tags(self):
        result = self.tf.filter(tags=["production"])
        self.assertEqual(len(result), 2)

    def test_filter_by_exclude_tags(self):
        result = self.tf.filter(service_name="web", exclude_tags=["staging"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-1")

    def test_filter_by_metadata(self):
        result = self.tf.filter(metadata_match={"region": "eu-west"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-2")

    def test_filter_by_health(self):
        self.reg.get_service("web-2").health_status = HealthStatus.CRITICAL
        result = self.tf.filter(health_status=HealthStatus.PASSING)
        self.assertEqual(len(result), 2)  # web-1 and api-1

    def test_filter_by_node(self):
        self.reg.register("web", service_id="web-3", node_id="node-2")
        result = self.tf.filter(node_id="node-2")
        self.assertEqual(len(result), 1)

    def test_filter_by_weight(self):
        self.reg.register("web", service_id="web-3", weight=10)
        result = self.tf.filter(min_weight=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-3")

    def test_filter_combined(self):
        result = self.tf.filter(service_name="web", tags=["v1"],
                                metadata_match={"region": "us-east"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-1")

    def test_filter_all_services(self):
        result = self.tf.filter()
        self.assertEqual(len(result), 3)

    def test_tag_expression_simple(self):
        result = self.tf.filter_by_tag_expression("web", "v1")
        self.assertEqual(len(result), 1)

    def test_tag_expression_negation(self):
        result = self.tf.filter_by_tag_expression("web", "!staging")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-1")

    def test_tag_expression_and(self):
        result = self.tf.filter_by_tag_expression("web", "v1,production")
        self.assertEqual(len(result), 1)

    def test_tag_expression_or(self):
        result = self.tf.filter_by_tag_expression("web", "v1|v2")
        self.assertEqual(len(result), 2)

    def test_tag_expression_empty(self):
        result = self.tf.filter_by_tag_expression("web", "")
        self.assertEqual(len(result), 2)

    def test_filter_version_after(self):
        result = self.tf.filter(version_after=0)
        self.assertEqual(len(result), 3)
        # All have version > 0 because register bumps version
        v = max(r.version for r in result)
        result2 = self.tf.filter(version_after=v - 1)
        self.assertEqual(len(result2), 1)


# =============================================================================
# ServiceCatalog Tests
# =============================================================================

class TestServiceCatalog(unittest.TestCase):
    def setUp(self):
        self.reg = ServiceRegistry("node-1")
        self.cat = ServiceCatalog(self.reg)

    def test_summary_basic(self):
        self.reg.register("web", service_id="web-1", tags=["v1"])
        self.reg.register("web", service_id="web-2", tags=["v2"])
        i3 = self.reg.register("web", service_id="web-3")
        i3.health_status = HealthStatus.CRITICAL
        summary = self.cat.summary()
        self.assertEqual(summary["web"]["total"], 3)
        self.assertEqual(summary["web"]["passing"], 2)
        self.assertEqual(summary["web"]["critical"], 1)

    def test_summary_empty(self):
        summary = self.cat.summary()
        self.assertEqual(summary, {})

    def test_service_health_all_passing(self):
        self.reg.register("web", service_id="web-1")
        self.assertEqual(self.cat.service_health("web"), HealthStatus.PASSING)

    def test_service_health_all_critical(self):
        i1 = self.reg.register("web", service_id="web-1")
        i1.health_status = HealthStatus.CRITICAL
        self.assertEqual(self.cat.service_health("web"), HealthStatus.CRITICAL)

    def test_service_health_mixed(self):
        self.reg.register("web", service_id="web-1")
        i2 = self.reg.register("web", service_id="web-2")
        i2.health_status = HealthStatus.CRITICAL
        self.assertEqual(self.cat.service_health("web"), HealthStatus.WARNING)

    def test_service_health_unknown(self):
        self.assertEqual(self.cat.service_health("missing"), HealthStatus.UNKNOWN)

    def test_nodes(self):
        self.reg.register("web", service_id="web-1", node_id="node-1")
        self.reg.register("web", service_id="web-2", node_id="node-2")
        self.reg.register("api", service_id="api-1", node_id="node-1")
        nodes = self.cat.nodes()
        self.assertIn("node-1", nodes)
        self.assertIn("web", nodes["node-1"])
        self.assertIn("api", nodes["node-1"])

    def test_find_by_metadata_key(self):
        self.reg.register("web", service_id="web-1", metadata={"env": "prod"})
        self.reg.register("api", service_id="api-1", metadata={"env": "staging"})
        result = self.cat.find_by_metadata("env")
        self.assertEqual(len(result), 2)

    def test_find_by_metadata_key_value(self):
        self.reg.register("web", service_id="web-1", metadata={"env": "prod"})
        self.reg.register("api", service_id="api-1", metadata={"env": "staging"})
        result = self.cat.find_by_metadata("env", "prod")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].service_id, "web-1")

    def test_summary_tags_aggregated(self):
        self.reg.register("web", service_id="web-1", tags=["v1", "prod"])
        self.reg.register("web", service_id="web-2", tags=["v2", "prod"])
        summary = self.cat.summary()
        self.assertIn("v1", summary["web"]["tags"])
        self.assertIn("v2", summary["web"]["tags"])
        self.assertIn("prod", summary["web"]["tags"])


# =============================================================================
# ServiceInstance Tests
# =============================================================================

class TestServiceInstance(unittest.TestCase):
    def test_to_dict(self):
        inst = ServiceInstance(
            service_id="web-1", service_name="web", node_id="n1",
            address="10.0.0.1", port=8080, tags=["v1"],
            metadata={"env": "prod"}, health_status=HealthStatus.PASSING,
        )
        d = inst.to_dict()
        self.assertEqual(d["service_id"], "web-1")
        self.assertEqual(d["health_status"], "passing")

    def test_from_dict(self):
        d = {
            "service_id": "web-1", "service_name": "web", "node_id": "n1",
            "address": "10.0.0.1", "port": 8080, "tags": ["v1"],
            "metadata": {"env": "prod"}, "health_status": "passing",
        }
        inst = ServiceInstance.from_dict(d)
        self.assertEqual(inst.service_id, "web-1")
        self.assertEqual(inst.health_status, HealthStatus.PASSING)

    def test_copy(self):
        inst = ServiceInstance(
            service_id="web-1", service_name="web", node_id="n1",
            address="10.0.0.1", port=8080, tags=["v1"],
            metadata={"env": "prod"},
        )
        c = inst.copy()
        self.assertEqual(c.service_id, "web-1")
        c.tags.append("v2")
        self.assertNotIn("v2", inst.tags)

    def test_from_dict_defaults(self):
        d = {
            "service_id": "x", "service_name": "x", "node_id": "n",
            "address": "1.2.3.4", "port": 80,
        }
        inst = ServiceInstance.from_dict(d)
        self.assertEqual(inst.tags, [])
        self.assertEqual(inst.weight, 1)


# =============================================================================
# GossipDiscovery Tests
# =============================================================================

class TestGossipDiscovery(unittest.TestCase):
    def test_register_local(self):
        gd = GossipDiscovery("node-1")
        inst = gd.register_local("web", service_id="web-1", port=8080)
        self.assertEqual(inst.service_name, "web")
        self.assertIn("web-1", gd.registry.services)

    def test_register_propagates_to_gossip(self):
        gd = GossipDiscovery("node-1")
        gd.register_local("web", service_id="web-1")
        self.assertIn("svc:web-1", gd.gossip_state._entries)

    def test_deregister_local(self):
        gd = GossipDiscovery("node-1")
        gd.register_local("web", service_id="web-1")
        self.assertTrue(gd.deregister_local("web-1"))
        self.assertNotIn("web-1", gd.registry.services)

    def test_resolve_local(self):
        gd = GossipDiscovery("node-1")
        gd.register_local("web", service_id="web-1")
        inst = gd.resolve("web")
        self.assertEqual(inst.service_id, "web-1")

    def test_sync_from_gossip(self):
        gd = GossipDiscovery("node-1")
        # Manually put something in gossip state
        gd.gossip_state.set("svc:web-remote", {
            "service_id": "web-remote",
            "service_name": "web",
            "node_id": "node-2",
            "address": "10.0.0.2",
            "port": 9090,
            "tags": ["remote"],
            "metadata": {},
            "health_status": "passing",
            "version": 99,
        })
        changes = gd.sync_from_gossip()
        self.assertGreater(changes, 0)
        self.assertIn("web-remote", gd.registry.services)

    def test_sync_tombstone(self):
        gd = GossipDiscovery("node-1")
        gd.register_local("web", service_id="web-1")
        gd.gossip_state.set("svc:web-1", None)
        changes = gd.sync_from_gossip()
        self.assertNotIn("web-1", gd.registry.services)


# =============================================================================
# ServiceMesh Tests
# =============================================================================

class TestServiceMesh(unittest.TestCase):
    def test_create_mesh(self):
        mesh = ServiceMesh(["node-1", "node-2", "node-3"])
        self.assertEqual(len(mesh.nodes), 3)

    def test_add_node(self):
        mesh = ServiceMesh(["node-1"])
        mesh.add_node("node-2")
        self.assertEqual(len(mesh.nodes), 2)

    def test_remove_node(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.remove_node("node-2")
        self.assertEqual(len(mesh.nodes), 1)

    def test_remove_nonexistent(self):
        mesh = ServiceMesh(["node-1"])
        self.assertFalse(mesh.remove_node("node-99"))

    def test_register_service(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        inst = mesh.register_service("node-1", "web", service_id="web-1", port=8080)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.service_name, "web")

    def test_register_on_nonexistent_node(self):
        mesh = ServiceMesh(["node-1"])
        self.assertIsNone(mesh.register_service("node-99", "web"))

    def test_deregister_service(self):
        mesh = ServiceMesh(["node-1"])
        mesh.register_service("node-1", "web", service_id="web-1")
        self.assertTrue(mesh.deregister_service("node-1", "web-1"))

    def test_deregister_nonexistent_node(self):
        mesh = ServiceMesh(["node-1"])
        self.assertFalse(mesh.deregister_service("node-99", "web-1"))

    def test_sync_propagates_services(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web", service_id="web-1", port=8080)
        mesh.sync_all()
        # node-2 should now see web-1
        instances = mesh.get_service_instances("web", node_id="node-2")
        self.assertGreater(len(instances), 0)

    def test_resolve_after_sync(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web", service_id="web-1", port=8080)
        mesh.sync_all()
        inst = mesh.resolve("node-2", "web")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.service_id, "web-1")

    def test_resolve_nonexistent_node(self):
        mesh = ServiceMesh(["node-1"])
        self.assertIsNone(mesh.resolve("node-99", "web"))

    def test_leader_election(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        self.assertTrue(mesh.elect_leader("primary", "node-1"))
        self.assertEqual(mesh.get_leader("primary"), "node-1")

    def test_get_all_services(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web", service_id="web-1")
        mesh.register_service("node-2", "api", service_id="api-1")
        all_svc = mesh.get_all_services()
        self.assertIn("web", all_svc)
        self.assertIn("api", all_svc)

    def test_get_all_services_specific_node(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web")
        mesh.register_service("node-2", "api")
        n1_svc = mesh.get_all_services(node_id="node-1")
        self.assertIn("web", n1_svc)
        self.assertNotIn("api", n1_svc)

    def test_get_all_services_nonexistent_node(self):
        mesh = ServiceMesh(["node-1"])
        self.assertEqual(mesh.get_all_services(node_id="node-99"), [])

    def test_get_service_instances_from_specific_node(self):
        mesh = ServiceMesh(["node-1"])
        mesh.register_service("node-1", "web", service_id="web-1")
        instances = mesh.get_service_instances("web", node_id="node-1")
        self.assertEqual(len(instances), 1)

    def test_get_service_instances_nonexistent_node(self):
        mesh = ServiceMesh(["node-1"])
        self.assertEqual(mesh.get_service_instances("web", node_id="node-99"), [])

    def test_health_summary(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web", service_id="web-1")
        mesh.register_service("node-2", "web", service_id="web-2")
        summary = mesh.health_summary()
        self.assertIn("web", summary)
        self.assertEqual(summary["web"]["total"], 2)

    def test_multi_service_sync(self):
        mesh = ServiceMesh(["node-1", "node-2", "node-3"])
        mesh.register_service("node-1", "web", service_id="web-1")
        mesh.register_service("node-2", "api", service_id="api-1")
        mesh.register_service("node-3", "db", service_id="db-1")
        mesh.sync_all()
        # All nodes should see all services
        for nid in ["node-1", "node-2", "node-3"]:
            services = mesh.get_all_services(node_id=nid)
            self.assertEqual(len(services), 3, f"{nid} should see 3 services, got {len(services)}")

    def test_deregister_propagates(self):
        mesh = ServiceMesh(["node-1", "node-2"])
        mesh.register_service("node-1", "web", service_id="web-1")
        mesh.sync_all()
        # node-2 should see it
        self.assertGreater(len(mesh.get_service_instances("web", node_id="node-2")), 0)
        # Deregister on node-1
        mesh.deregister_service("node-1", "web-1")
        mesh.sync_all()
        # node-2 should no longer see it (tombstone)
        remaining = [i for i in mesh.get_service_instances("web", node_id="node-2")
                     if i.service_id == "web-1"]
        # Tombstone should have removed it
        self.assertEqual(len(remaining), 0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
    def test_full_lifecycle(self):
        """Register, health check, resolve, deregister."""
        reg = ServiceRegistry("node-1")
        resolver = ServiceResolver(reg)
        catalog = ServiceCatalog(reg)

        # Register
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: 200)
        reg.register("web", service_id="web-1", port=8080,
                     tags=["production"], health_checks=[hc])
        reg.register("web", service_id="web-2", port=8081,
                     tags=["production"])

        # Verify catalog
        summary = catalog.summary()
        self.assertEqual(summary["web"]["total"], 2)

        # Health check
        reg.run_health_checks()
        self.assertEqual(catalog.service_health("web"), HealthStatus.PASSING)

        # Resolve
        inst = resolver.resolve("web")
        self.assertIn(inst.service_id, ["web-1", "web-2"])

        # Deregister
        reg.deregister("web-1")
        inst = resolver.resolve("web")
        self.assertEqual(inst.service_id, "web-2")

    def test_mesh_with_health_checks(self):
        """Multi-node mesh with health checking."""
        mesh = ServiceMesh(["node-1", "node-2"])

        # Register healthy service on node-1
        node1 = mesh.nodes["node-1"]
        hc = HealthCheck(check_id="http-1", check_type=CheckType.HTTP,
                         callback=lambda: 200)
        node1.registry.register("web", service_id="web-1", port=8080,
                                health_checks=[hc], node_id="node-1")

        # Sync
        mesh.sync_all()

        # Verify visible on node-2
        instances = mesh.get_service_instances("web", node_id="node-2")
        self.assertGreater(len(instances), 0)

    def test_tag_filter_with_resolver(self):
        """Use tag filter + resolver together."""
        reg = ServiceRegistry("node-1")
        tf = TagFilter(reg)
        resolver = ServiceResolver(reg)

        reg.register("web", service_id="web-1", tags=["v1", "production"])
        reg.register("web", service_id="web-2", tags=["v2", "canary"])

        # Filter to production only
        prod = tf.filter(service_name="web", tags=["production"])
        self.assertEqual(len(prod), 1)

        # Resolve with tag filter
        inst = resolver.resolve("web", tags=["production"])
        self.assertEqual(inst.service_id, "web-1")

    def test_leader_with_watches(self):
        """Leader election with watch notifications."""
        le = LeaderElection()
        events = []
        le.watch_election("primary", lambda e, n, l: events.append((e, l)))

        le.campaign("primary", "node-1")
        le.campaign("primary", "node-2")
        le.resign("primary", "node-1")

        self.assertEqual(events[0], (ServiceEvent.LEADER_ELECTED, "node-1"))
        self.assertEqual(events[1], (ServiceEvent.LEADER_ELECTED, "node-2"))

    def test_kv_config_with_services(self):
        """Use KV store for service configuration."""
        reg = ServiceRegistry("node-1")

        reg.register("web", service_id="web-1")
        reg.kv_put("config/web/max_connections", "100")
        reg.kv_put("config/web/timeout", "30s")

        config = reg.kv_list("config/web/")
        self.assertEqual(len(config), 2)
        self.assertEqual(config["config/web/max_connections"], "100")

    def test_watch_then_filter(self):
        """Register a watch, then verify filter picks it up."""
        reg = ServiceRegistry("node-1")
        tf = TagFilter(reg)
        changes = []

        reg.watch("web", lambda inst: changes.append(inst.service_id))
        reg.register("web", service_id="web-1", tags=["v1"])
        reg.register("web", service_id="web-2", tags=["v2"])

        self.assertEqual(len(changes), 2)
        result = tf.filter(service_name="web", tags=["v1"])
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
