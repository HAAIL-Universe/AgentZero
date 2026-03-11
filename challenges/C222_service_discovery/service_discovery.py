"""
C222: Service Discovery
Composing C203 (Gossip Protocol) + C209 (Distributed Lock Service)

A Consul/Eureka-inspired service discovery system with:
- ServiceRegistry: register/deregister services with metadata
- HealthChecker: configurable health checks (TTL, HTTP-like, TCP-like, script)
- ServiceResolver: DNS-like resolution with load balancing strategies
- ServiceWatch: watch for service changes with callbacks
- GossipDiscovery: gossip-based service dissemination across nodes
- LeaderElection: leader election for consistent reads via distributed locks
- ServiceMesh: multi-node service discovery with anti-entropy sync
- TagFilter: service filtering by tags, metadata, health status
- ServiceCatalog: queryable catalog with KV store for service config
"""

import sys
import os
import time
import math
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C203_gossip_protocol'))
from gossip import (
    GossipNode, GossipNetwork, GossipState, NodeStatus as GossipNodeStatus,
    PhiAccrualDetector, HeartbeatDetector
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C209_distributed_lock_service'))
from distributed_lock import (
    LockServiceNode, LockServiceCluster, LockMode, SessionState
)


# =============================================================================
# Enums
# =============================================================================

class HealthStatus(Enum):
    PASSING = "passing"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CheckType(Enum):
    TTL = "ttl"           # Service must heartbeat within TTL
    HTTP = "http"         # Simulated HTTP check (callback returns status code)
    TCP = "tcp"           # Simulated TCP check (callback returns bool)
    SCRIPT = "script"     # Arbitrary script check (callback returns HealthStatus)


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


class ServiceEvent(Enum):
    REGISTERED = "registered"
    DEREGISTERED = "deregistered"
    HEALTH_CHANGED = "health_changed"
    METADATA_UPDATED = "metadata_updated"
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HealthCheck:
    """Configuration for a service health check."""
    check_id: str
    check_type: CheckType
    interval: float = 10.0    # Seconds between checks
    timeout: float = 5.0      # Timeout for check
    ttl: float = 30.0         # For TTL checks: max time between heartbeats
    deregister_after: float = 0.0  # Auto-deregister after critical for this long (0 = never)
    callback: Optional[Callable] = None  # For HTTP/TCP/SCRIPT checks
    last_check: float = 0.0
    last_output: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    warning_threshold: int = 3  # Consecutive failures before WARNING
    critical_threshold: int = 5  # Consecutive failures before CRITICAL
    last_heartbeat: float = 0.0  # For TTL checks


@dataclass
class ServiceInstance:
    """A single instance of a service."""
    service_id: str
    service_name: str
    node_id: str
    address: str
    port: int
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.PASSING
    health_checks: dict = field(default_factory=dict)  # check_id -> HealthCheck
    registered_at: float = 0.0
    last_updated: float = 0.0
    weight: int = 1  # For weighted load balancing
    connections: int = 0  # Current connection count (for least_connections)
    version: int = 0  # Monotonic version for conflict resolution

    def copy(self):
        inst = ServiceInstance(
            service_id=self.service_id,
            service_name=self.service_name,
            node_id=self.node_id,
            address=self.address,
            port=self.port,
            tags=list(self.tags),
            metadata=dict(self.metadata),
            health_status=self.health_status,
            health_checks={},
            registered_at=self.registered_at,
            last_updated=self.last_updated,
            weight=self.weight,
            connections=self.connections,
            version=self.version,
        )
        return inst

    def to_dict(self):
        return {
            'service_id': self.service_id,
            'service_name': self.service_name,
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'tags': self.tags,
            'metadata': self.metadata,
            'health_status': self.health_status.value,
            'registered_at': self.registered_at,
            'last_updated': self.last_updated,
            'weight': self.weight,
            'version': self.version,
        }

    @staticmethod
    def from_dict(d):
        return ServiceInstance(
            service_id=d['service_id'],
            service_name=d['service_name'],
            node_id=d['node_id'],
            address=d['address'],
            port=d['port'],
            tags=d.get('tags', []),
            metadata=d.get('metadata', {}),
            health_status=HealthStatus(d.get('health_status', 'passing')),
            registered_at=d.get('registered_at', 0.0),
            last_updated=d.get('last_updated', 0.0),
            weight=d.get('weight', 1),
            connections=d.get('connections', 0),
            version=d.get('version', 0),
        )


@dataclass
class WatchEntry:
    """A watch on a service for changes."""
    watch_id: str
    service_name: str
    callback: Callable
    tags: list = field(default_factory=list)
    health_filter: Optional[HealthStatus] = None
    last_index: int = 0  # For blocking queries


# =============================================================================
# ServiceRegistry
# =============================================================================

class ServiceRegistry:
    """
    Core service registry. Manages registration, deregistration,
    health checking, and querying of service instances.
    """

    def __init__(self, node_id: str = "node-1"):
        self.node_id = node_id
        self.services = {}       # service_id -> ServiceInstance
        self.by_name = defaultdict(set)  # service_name -> {service_id, ...}
        self.watches = {}        # watch_id -> WatchEntry
        self.event_log = []      # List of (timestamp, ServiceEvent, data)
        self.kv_store = {}       # Key-value config store
        self._version = 0
        self._watch_counter = 0
        self._time = time.time

    def register(self, service_name, service_id=None, address="127.0.0.1",
                 port=8080, tags=None, metadata=None, health_checks=None,
                 weight=1, node_id=None):
        """Register a service instance."""
        if service_id is None:
            service_id = f"{service_name}-{self._next_id()}"

        now = self._time()
        instance = ServiceInstance(
            service_id=service_id,
            service_name=service_name,
            node_id=node_id or self.node_id,
            address=address,
            port=port,
            tags=tags or [],
            metadata=metadata or {},
            health_status=HealthStatus.PASSING,
            registered_at=now,
            last_updated=now,
            weight=weight,
            version=self._bump_version(),
        )

        if health_checks:
            for hc in health_checks:
                hc.last_check = now
                hc.last_heartbeat = now
                hc.status = HealthStatus.PASSING
                instance.health_checks[hc.check_id] = hc

        self.services[service_id] = instance
        self.by_name[service_name].add(service_id)
        self._emit_event(ServiceEvent.REGISTERED, instance)
        return instance

    def deregister(self, service_id):
        """Deregister a service instance."""
        if service_id not in self.services:
            return False
        instance = self.services.pop(service_id)
        self.by_name[instance.service_name].discard(service_id)
        if not self.by_name[instance.service_name]:
            del self.by_name[instance.service_name]
        self._emit_event(ServiceEvent.DEREGISTERED, instance)
        return True

    def get_service(self, service_id):
        """Get a single service instance by ID."""
        return self.services.get(service_id)

    def get_services(self, service_name, tags=None, health_filter=None):
        """Get all instances of a service, optionally filtered."""
        ids = self.by_name.get(service_name, set())
        results = []
        for sid in ids:
            inst = self.services[sid]
            if tags and not all(t in inst.tags for t in tags):
                continue
            if health_filter and inst.health_status != health_filter:
                continue
            results.append(inst)
        return results

    def get_healthy_services(self, service_name, tags=None):
        """Get only healthy instances of a service."""
        return self.get_services(service_name, tags=tags,
                                health_filter=HealthStatus.PASSING)

    def get_all_services(self):
        """Get names of all registered services."""
        return list(self.by_name.keys())

    def update_metadata(self, service_id, metadata):
        """Update metadata for a service instance."""
        inst = self.services.get(service_id)
        if not inst:
            return False
        inst.metadata.update(metadata)
        inst.last_updated = self._time()
        inst.version = self._bump_version()
        self._emit_event(ServiceEvent.METADATA_UPDATED, inst)
        return True

    def update_tags(self, service_id, tags):
        """Replace tags for a service instance."""
        inst = self.services.get(service_id)
        if not inst:
            return False
        inst.tags = list(tags)
        inst.last_updated = self._time()
        inst.version = self._bump_version()
        return True

    # -- Health checking --

    def run_health_checks(self, now=None):
        """Run all pending health checks. Returns list of services whose health changed."""
        now = now or self._time()
        changed = []
        to_deregister = []

        for service_id, inst in list(self.services.items()):
            old_status = inst.health_status
            worst = HealthStatus.PASSING

            for check_id, hc in inst.health_checks.items():
                self._run_single_check(hc, now)
                if hc.status == HealthStatus.CRITICAL:
                    worst = HealthStatus.CRITICAL
                elif hc.status == HealthStatus.WARNING and worst != HealthStatus.CRITICAL:
                    worst = HealthStatus.WARNING

            if not inst.health_checks:
                continue

            inst.health_status = worst
            inst.last_updated = now

            if worst != old_status:
                changed.append(inst)
                self._emit_event(ServiceEvent.HEALTH_CHANGED, inst)
                self._notify_watches(inst)

            # Check auto-deregister
            for hc in inst.health_checks.values():
                if (hc.deregister_after > 0 and
                    hc.status == HealthStatus.CRITICAL and
                    hc.consecutive_failures * hc.interval >= hc.deregister_after):
                    to_deregister.append(service_id)
                    break

        for sid in to_deregister:
            self.deregister(sid)

        return changed

    def _run_single_check(self, hc, now):
        """Run a single health check."""
        if hc.check_type == CheckType.TTL:
            elapsed = now - hc.last_heartbeat
            if elapsed > hc.ttl:
                hc.consecutive_failures += 1
                if hc.consecutive_failures >= hc.critical_threshold:
                    hc.status = HealthStatus.CRITICAL
                elif hc.consecutive_failures >= hc.warning_threshold:
                    hc.status = HealthStatus.WARNING
                hc.last_output = f"TTL expired ({elapsed:.1f}s > {hc.ttl}s)"
            else:
                hc.consecutive_failures = 0
                hc.status = HealthStatus.PASSING
                hc.last_output = "TTL OK"
        elif hc.check_type == CheckType.HTTP:
            if hc.callback:
                try:
                    code = hc.callback()
                    if 200 <= code < 300:
                        hc.status = HealthStatus.PASSING
                        hc.consecutive_failures = 0
                    elif 400 <= code < 500:
                        hc.consecutive_failures += 1
                        hc.status = HealthStatus.WARNING
                    else:
                        hc.consecutive_failures += 1
                        hc.status = HealthStatus.CRITICAL
                    hc.last_output = f"HTTP {code}"
                except Exception as e:
                    hc.consecutive_failures += 1
                    hc.status = HealthStatus.CRITICAL
                    hc.last_output = f"HTTP error: {e}"
            else:
                hc.status = HealthStatus.PASSING
        elif hc.check_type == CheckType.TCP:
            if hc.callback:
                try:
                    ok = hc.callback()
                    if ok:
                        hc.status = HealthStatus.PASSING
                        hc.consecutive_failures = 0
                    else:
                        hc.consecutive_failures += 1
                        hc.status = HealthStatus.CRITICAL
                    hc.last_output = f"TCP {'open' if ok else 'refused'}"
                except Exception as e:
                    hc.consecutive_failures += 1
                    hc.status = HealthStatus.CRITICAL
                    hc.last_output = f"TCP error: {e}"
            else:
                hc.status = HealthStatus.PASSING
        elif hc.check_type == CheckType.SCRIPT:
            if hc.callback:
                try:
                    result = hc.callback()
                    if isinstance(result, HealthStatus):
                        hc.status = result
                    elif result == 0:
                        hc.status = HealthStatus.PASSING
                    elif result == 1:
                        hc.status = HealthStatus.WARNING
                    else:
                        hc.status = HealthStatus.CRITICAL
                    if hc.status == HealthStatus.PASSING:
                        hc.consecutive_failures = 0
                    else:
                        hc.consecutive_failures += 1
                    hc.last_output = f"Script result: {result}"
                except Exception as e:
                    hc.consecutive_failures += 1
                    hc.status = HealthStatus.CRITICAL
                    hc.last_output = f"Script error: {e}"
            else:
                hc.status = HealthStatus.PASSING
        hc.last_check = now

    def ttl_heartbeat(self, service_id, check_id=None, now=None):
        """Send a TTL heartbeat for a service."""
        now = now or self._time()
        inst = self.services.get(service_id)
        if not inst:
            return False
        if check_id:
            hc = inst.health_checks.get(check_id)
            if hc and hc.check_type == CheckType.TTL:
                hc.last_heartbeat = now
                hc.consecutive_failures = 0
                hc.status = HealthStatus.PASSING
                return True
            return False
        # Heartbeat all TTL checks
        found = False
        for hc in inst.health_checks.values():
            if hc.check_type == CheckType.TTL:
                hc.last_heartbeat = now
                hc.consecutive_failures = 0
                hc.status = HealthStatus.PASSING
                found = True
        return found

    # -- Watches --

    def watch(self, service_name, callback, tags=None, health_filter=None):
        """Watch for changes to a service."""
        self._watch_counter += 1
        watch_id = f"watch-{self._watch_counter}"
        entry = WatchEntry(
            watch_id=watch_id,
            service_name=service_name,
            callback=callback,
            tags=tags or [],
            health_filter=health_filter,
            last_index=self._version,
        )
        self.watches[watch_id] = entry
        return watch_id

    def unwatch(self, watch_id):
        """Remove a watch."""
        return self.watches.pop(watch_id, None) is not None

    def _notify_watches(self, instance):
        """Notify watches matching a changed service."""
        for watch in self.watches.values():
            if watch.service_name != instance.service_name:
                continue
            if watch.tags and not all(t in instance.tags for t in watch.tags):
                continue
            if watch.health_filter and instance.health_status != watch.health_filter:
                continue
            try:
                watch.callback(instance)
            except Exception:
                pass
            watch.last_index = self._version

    # -- KV Store --

    def kv_put(self, key, value):
        """Put a key-value pair in the config store."""
        self.kv_store[key] = value

    def kv_get(self, key, default=None):
        """Get a value from the config store."""
        return self.kv_store.get(key, default)

    def kv_delete(self, key):
        """Delete a key from the config store."""
        return self.kv_store.pop(key, None) is not None

    def kv_list(self, prefix=""):
        """List keys with given prefix."""
        return {k: v for k, v in self.kv_store.items() if k.startswith(prefix)}

    # -- Event Log --

    def get_events(self, service_name=None, event_type=None, limit=100):
        """Get events from the event log."""
        events = self.event_log
        if service_name:
            events = [e for e in events if e[2].get('service_name') == service_name]
        if event_type:
            events = [e for e in events if e[1] == event_type]
        return events[-limit:]

    def _emit_event(self, event_type, instance):
        data = {
            'service_id': instance.service_id,
            'service_name': instance.service_name,
            'health_status': instance.health_status.value,
            'node_id': instance.node_id,
        }
        self.event_log.append((self._time(), event_type, data))
        # Notify watches on register/deregister
        if event_type in (ServiceEvent.REGISTERED, ServiceEvent.DEREGISTERED):
            self._notify_watches(instance)

    # -- Internals --

    def _next_id(self):
        self._version += 1
        return f"{self.node_id}-{self._version}"

    def _bump_version(self):
        self._version += 1
        return self._version


# =============================================================================
# ServiceResolver
# =============================================================================

class ServiceResolver:
    """
    DNS-like service resolution with multiple load balancing strategies.
    """

    def __init__(self, registry):
        self.registry = registry
        self._rr_counters = defaultdict(int)  # service_name -> counter
        self._hash_ring = {}  # For consistent hashing

    def resolve(self, service_name, strategy=LoadBalanceStrategy.ROUND_ROBIN,
                tags=None, hash_key=None):
        """Resolve a service name to a single instance."""
        instances = self.registry.get_healthy_services(service_name, tags=tags)
        if not instances:
            return None

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, instances)
        elif strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(instances)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(instances)
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted(instances)
        elif strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._consistent_hash(service_name, instances, hash_key or "")
        return instances[0]

    def resolve_all(self, service_name, tags=None, include_unhealthy=False):
        """Resolve all instances of a service."""
        if include_unhealthy:
            return self.registry.get_services(service_name, tags=tags)
        return self.registry.get_healthy_services(service_name, tags=tags)

    def _round_robin(self, service_name, instances):
        idx = self._rr_counters[service_name] % len(instances)
        self._rr_counters[service_name] += 1
        return instances[idx]

    def _least_connections(self, instances):
        return min(instances, key=lambda i: i.connections)

    def _weighted(self, instances):
        total = sum(i.weight for i in instances)
        if total == 0:
            return random.choice(instances)
        r = random.uniform(0, total)
        cumulative = 0
        for inst in instances:
            cumulative += inst.weight
            if r <= cumulative:
                return inst
        return instances[-1]

    def _consistent_hash(self, service_name, instances, hash_key):
        """Simple consistent hash using virtual nodes."""
        vnodes = 100
        ring = []
        for inst in instances:
            for i in range(vnodes):
                h = hashlib.md5(f"{inst.service_id}:{i}".encode()).hexdigest()
                ring.append((int(h, 16), inst))
        ring.sort(key=lambda x: x[0])

        key_hash = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        for h, inst in ring:
            if h >= key_hash:
                return inst
        return ring[0][1] if ring else None


# =============================================================================
# LeaderElection
# =============================================================================

class LeaderElection:
    """
    Leader election using distributed locks.
    Provides leader/follower semantics for services that need a single leader.
    """

    def __init__(self, lock_service=None):
        self.lock_service = lock_service
        self.leaders = {}  # election_name -> leader_id
        self.candidates = defaultdict(list)  # election_name -> [candidate_ids]
        self.watchers = defaultdict(list)  # election_name -> [callbacks]
        self._elections = {}  # election_name -> election metadata

    def create_election(self, election_name):
        """Create a new election."""
        self._elections[election_name] = {
            'created_at': time.time(),
            'leader': None,
            'term': 0,
        }
        return True

    def campaign(self, election_name, candidate_id):
        """Enter a candidate into an election. Returns True if becomes leader."""
        if election_name not in self._elections:
            self.create_election(election_name)

        if candidate_id not in self.candidates[election_name]:
            self.candidates[election_name].append(candidate_id)

        election = self._elections[election_name]

        if election['leader'] is None:
            # No leader -- first candidate wins
            election['leader'] = candidate_id
            election['term'] += 1
            self.leaders[election_name] = candidate_id
            self._notify_leader_change(election_name, candidate_id)
            return True
        return election['leader'] == candidate_id

    def resign(self, election_name, candidate_id):
        """Resign from leadership."""
        if election_name not in self._elections:
            return False
        election = self._elections[election_name]
        if election['leader'] != candidate_id:
            return False

        election['leader'] = None
        self.leaders.pop(election_name, None)
        if candidate_id in self.candidates[election_name]:
            self.candidates[election_name].remove(candidate_id)

        # Promote next candidate
        remaining = self.candidates[election_name]
        if remaining:
            new_leader = remaining[0]
            election['leader'] = new_leader
            election['term'] += 1
            self.leaders[election_name] = new_leader
            self._notify_leader_change(election_name, new_leader)
        else:
            self._notify_leader_lost(election_name)

        return True

    def get_leader(self, election_name):
        """Get the current leader for an election."""
        return self.leaders.get(election_name)

    def is_leader(self, election_name, candidate_id):
        """Check if a candidate is the leader."""
        return self.leaders.get(election_name) == candidate_id

    def get_term(self, election_name):
        """Get the current election term."""
        if election_name not in self._elections:
            return 0
        return self._elections[election_name]['term']

    def watch_election(self, election_name, callback):
        """Watch for leader changes."""
        self.watchers[election_name].append(callback)

    def _notify_leader_change(self, election_name, leader_id):
        for cb in self.watchers[election_name]:
            try:
                cb(ServiceEvent.LEADER_ELECTED, election_name, leader_id)
            except Exception:
                pass

    def _notify_leader_lost(self, election_name):
        for cb in self.watchers[election_name]:
            try:
                cb(ServiceEvent.LEADER_LOST, election_name, None)
            except Exception:
                pass


# =============================================================================
# TagFilter
# =============================================================================

class TagFilter:
    """Advanced service filtering by tags, metadata, and health."""

    def __init__(self, registry):
        self.registry = registry

    def filter(self, service_name=None, tags=None, exclude_tags=None,
               metadata_match=None, health_status=None, node_id=None,
               min_weight=0, version_after=0):
        """
        Filter services with multiple criteria.
        All criteria are AND-combined.
        """
        if service_name:
            candidates = list(self.registry.by_name.get(service_name, set()))
        else:
            candidates = list(self.registry.services.keys())

        results = []
        for sid in candidates:
            inst = self.registry.services.get(sid)
            if not inst:
                continue
            if tags and not all(t in inst.tags for t in tags):
                continue
            if exclude_tags and any(t in inst.tags for t in exclude_tags):
                continue
            if metadata_match:
                if not all(inst.metadata.get(k) == v for k, v in metadata_match.items()):
                    continue
            if health_status and inst.health_status != health_status:
                continue
            if node_id and inst.node_id != node_id:
                continue
            if inst.weight < min_weight:
                continue
            if inst.version <= version_after:
                continue
            results.append(inst)
        return results

    def filter_by_tag_expression(self, service_name, expression):
        """
        Filter by tag expressions:
        - "tag1" -- must have tag1
        - "!tag1" -- must NOT have tag1
        - "tag1,tag2" -- must have BOTH (AND)
        - "tag1|tag2" -- must have EITHER (OR)
        """
        instances = self.registry.get_services(service_name)
        if not expression:
            return instances

        # Parse: OR groups separated by |, AND within groups by ,
        if '|' in expression:
            groups = expression.split('|')
            results = set()
            for group in groups:
                matching = self._filter_by_and_tags(instances, group.strip())
                results.update(m.service_id for m in matching)
            return [i for i in instances if i.service_id in results]
        else:
            return self._filter_by_and_tags(instances, expression)

    def _filter_by_and_tags(self, instances, and_expr):
        tags = [t.strip() for t in and_expr.split(',')]
        results = []
        for inst in instances:
            match = True
            for tag in tags:
                if tag.startswith('!'):
                    if tag[1:] in inst.tags:
                        match = False
                        break
                else:
                    if tag not in inst.tags:
                        match = False
                        break
            if match:
                results.append(inst)
        return results


# =============================================================================
# ServiceCatalog
# =============================================================================

class ServiceCatalog:
    """
    Queryable service catalog with aggregation and summary information.
    """

    def __init__(self, registry):
        self.registry = registry

    def summary(self):
        """Get a summary of all services."""
        result = {}
        for name in self.registry.get_all_services():
            instances = self.registry.get_services(name)
            passing = sum(1 for i in instances if i.health_status == HealthStatus.PASSING)
            warning = sum(1 for i in instances if i.health_status == HealthStatus.WARNING)
            critical = sum(1 for i in instances if i.health_status == HealthStatus.CRITICAL)
            result[name] = {
                'total': len(instances),
                'passing': passing,
                'warning': warning,
                'critical': critical,
                'tags': list(set(t for i in instances for t in i.tags)),
                'nodes': list(set(i.node_id for i in instances)),
            }
        return result

    def service_health(self, service_name):
        """Get aggregate health for a service."""
        instances = self.registry.get_services(service_name)
        if not instances:
            return HealthStatus.UNKNOWN
        statuses = [i.health_status for i in instances]
        if all(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.WARNING
        if all(s == HealthStatus.PASSING for s in statuses):
            return HealthStatus.PASSING
        return HealthStatus.WARNING

    def nodes(self):
        """Get all nodes with their services."""
        result = defaultdict(list)
        for inst in self.registry.services.values():
            result[inst.node_id].append(inst.service_name)
        return dict(result)

    def find_by_metadata(self, key, value=None):
        """Find services by metadata key (and optionally value)."""
        results = []
        for inst in self.registry.services.values():
            if key in inst.metadata:
                if value is None or inst.metadata[key] == value:
                    results.append(inst)
        return results


# =============================================================================
# GossipDiscovery
# =============================================================================

class GossipDiscovery:
    """
    Gossip-based service discovery across multiple nodes.
    Uses C203 GossipState for eventually-consistent service dissemination.
    """

    def __init__(self, node_id, network=None):
        self.node_id = node_id
        self.network = network or GossipNetwork()
        self.gossip_node = GossipNode(node_id, f"{node_id}:8301")
        self.gossip_state = GossipState(node_id)
        self.registry = ServiceRegistry(node_id)
        self.resolver = ServiceResolver(self.registry)
        self._sync_version = 0

    def register_local(self, service_name, service_id=None, address="127.0.0.1",
                       port=8080, tags=None, metadata=None, weight=1):
        """Register a service locally and propagate via gossip."""
        inst = self.registry.register(
            service_name, service_id=service_id, address=address,
            port=port, tags=tags, metadata=metadata, weight=weight,
            node_id=self.node_id,
        )
        # Propagate to gossip state
        key = f"svc:{inst.service_id}"
        self.gossip_state.set(key, inst.to_dict())
        self._sync_version += 1
        return inst

    def deregister_local(self, service_id):
        """Deregister a local service and propagate."""
        key = f"svc:{service_id}"
        self.gossip_state.set(key, None)  # Tombstone
        return self.registry.deregister(service_id)

    def sync_from_gossip(self):
        """
        Pull service state from gossip and update local registry.
        Returns number of services added/updated.
        """
        changes = 0
        for key, entry in self.gossip_state._entries.items():
            if not key.startswith("svc:"):
                continue
            service_id = key[4:]
            value = entry.value if hasattr(entry, 'value') else entry

            if value is None:
                # Tombstone -- deregister if we have it
                if service_id in self.registry.services:
                    self.registry.deregister(service_id)
                    changes += 1
                continue

            if isinstance(value, dict):
                existing = self.registry.services.get(service_id)
                remote_version = value.get('version', 0)
                if existing and existing.version >= remote_version:
                    continue
                # Register or update
                if existing:
                    # Update existing
                    existing.health_status = HealthStatus(value.get('health_status', 'passing'))
                    existing.tags = value.get('tags', [])
                    existing.metadata = value.get('metadata', {})
                    existing.weight = value.get('weight', 1)
                    existing.version = remote_version
                    existing.last_updated = value.get('last_updated', time.time())
                else:
                    # New registration
                    self.registry.register(
                        service_name=value['service_name'],
                        service_id=service_id,
                        address=value.get('address', '127.0.0.1'),
                        port=value.get('port', 8080),
                        tags=value.get('tags', []),
                        metadata=value.get('metadata', {}),
                        weight=value.get('weight', 1),
                        node_id=value.get('node_id', 'unknown'),
                    )
                changes += 1

        return changes

    def resolve(self, service_name, strategy=LoadBalanceStrategy.ROUND_ROBIN,
                tags=None, hash_key=None):
        """Resolve a service using the local registry (fed by gossip)."""
        return self.resolver.resolve(service_name, strategy=strategy,
                                     tags=tags, hash_key=hash_key)


# =============================================================================
# ServiceMesh
# =============================================================================

class ServiceMesh:
    """
    Multi-node service discovery mesh.
    Coordinates multiple GossipDiscovery nodes with anti-entropy sync.
    """

    def __init__(self, node_ids=None, network=None):
        self.network = network or GossipNetwork()
        self.nodes = {}  # node_id -> GossipDiscovery
        self.leader_election = LeaderElection()

        if node_ids:
            for nid in node_ids:
                self.add_node(nid)

    def add_node(self, node_id):
        """Add a node to the mesh."""
        node = GossipDiscovery(node_id, self.network)
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id):
        """Remove a node from the mesh."""
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        return True

    def register_service(self, node_id, service_name, **kwargs):
        """Register a service on a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        return node.register_local(service_name, **kwargs)

    def deregister_service(self, node_id, service_id):
        """Deregister a service from a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        return node.deregister_local(service_id)

    def sync_all(self):
        """Synchronize all nodes via gossip."""
        # Propagate gossip state between nodes
        total_changes = 0
        for node_id, node in self.nodes.items():
            # Push local state to gossip
            for sid, inst in list(node.registry.services.items()):
                if inst.node_id == node_id:
                    key = f"svc:{sid}"
                    node.gossip_state.set(key, inst.to_dict())

        # Exchange state between pairs
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                self._exchange_state(node_list[i], node_list[j])

        # Pull from gossip
        for node in self.nodes.values():
            changes = node.sync_from_gossip()
            total_changes += changes

        return total_changes

    def _exchange_state(self, node_a, node_b):
        """Exchange gossip state between two nodes."""
        # Copy state from A to B and B to A
        for key, entry in list(node_a.gossip_state._entries.items()):
            b_entry = node_b.gossip_state._entries.get(key)
            if b_entry is None:
                node_b.gossip_state.merge_entry(entry)
            elif entry.version > b_entry.version:
                node_b.gossip_state.merge_entry(entry)

        for key, entry in list(node_b.gossip_state._entries.items()):
            a_entry = node_a.gossip_state._entries.get(key)
            if a_entry is None:
                node_a.gossip_state.merge_entry(entry)
            elif entry.version > a_entry.version:
                node_a.gossip_state.merge_entry(entry)

    def resolve(self, node_id, service_name, strategy=LoadBalanceStrategy.ROUND_ROBIN,
                tags=None, hash_key=None):
        """Resolve a service from a specific node's perspective."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        return node.resolve(service_name, strategy=strategy, tags=tags, hash_key=hash_key)

    def elect_leader(self, election_name, candidate_id):
        """Run leader election for a service."""
        return self.leader_election.campaign(election_name, candidate_id)

    def get_leader(self, election_name):
        """Get the leader for an election."""
        return self.leader_election.get_leader(election_name)

    def get_all_services(self, node_id=None):
        """Get all services across the mesh or from a specific node."""
        if node_id:
            node = self.nodes.get(node_id)
            return node.registry.get_all_services() if node else []
        all_services = set()
        for node in self.nodes.values():
            all_services.update(node.registry.get_all_services())
        return list(all_services)

    def get_service_instances(self, service_name, node_id=None):
        """Get all instances of a service across the mesh."""
        if node_id:
            node = self.nodes.get(node_id)
            return node.registry.get_services(service_name) if node else []
        # Aggregate from all nodes, deduplicate by service_id
        seen = {}
        for node in self.nodes.values():
            for inst in node.registry.get_services(service_name):
                if inst.service_id not in seen or inst.version > seen[inst.service_id].version:
                    seen[inst.service_id] = inst
        return list(seen.values())

    def health_summary(self):
        """Get health summary across the mesh."""
        catalog = ServiceCatalog(ServiceRegistry("mesh-aggregate"))
        # Build aggregate
        agg = catalog.registry
        seen = {}
        for node in self.nodes.values():
            for inst in node.registry.services.values():
                if inst.service_id not in seen or inst.version > seen[inst.service_id].version:
                    seen[inst.service_id] = inst

        for inst in seen.values():
            agg.services[inst.service_id] = inst.copy()
            agg.by_name[inst.service_name].add(inst.service_id)

        return ServiceCatalog(agg).summary()
