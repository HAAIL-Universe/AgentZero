"""
C205: Consistent Hashing

A comprehensive consistent hashing library with virtual nodes, ring topology,
load balancing, and replication support. Key building block for distributed
key-value stores.

Components:
1. HashRing -- Core consistent hash ring with virtual nodes
2. WeightedHashRing -- Nodes with different weights (more vnodes for bigger nodes)
3. BoundedLoadHashRing -- Google's bounded-load consistent hashing
4. RendezvousHashing -- Highest Random Weight (HRW) hashing alternative
5. JumpConsistentHash -- Google's jump consistent hash (fixed bucket count)
6. MultiProbeHashRing -- Multi-probe consistent hashing (fewer vnodes, same balance)
7. MaglevHashRing -- Google Maglev hashing (lookup table, minimal disruption)
8. ReplicatedHashRing -- Hash ring with replication factor and preference lists
"""

import hashlib
import struct
import bisect
import math
from collections import defaultdict


# --- Hashing utilities ---

def md5_hash(key):
    """Hash a key to a 32-bit integer using MD5."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    digest = hashlib.md5(key).digest()
    return struct.unpack('<I', digest[:4])[0]


def sha256_hash(key):
    """Hash a key to a 32-bit integer using SHA-256."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    digest = hashlib.sha256(key).digest()
    return struct.unpack('<I', digest[:4])[0]


def xxhash_sim(key):
    """Simulate xxhash using MD5 with different seed for variety."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    digest = hashlib.md5(b'\x01' + key).digest()
    return struct.unpack('<I', digest[:4])[0]


RING_SIZE = 2**32


# --- 1. HashRing ---

class HashRing:
    """
    Consistent hash ring with virtual nodes.

    Each physical node maps to `num_replicas` virtual nodes on the ring.
    Keys are hashed and assigned to the next virtual node clockwise.
    """

    def __init__(self, nodes=None, num_replicas=150, hash_fn=None):
        self._hash_fn = hash_fn or md5_hash
        self._num_replicas = num_replicas
        self._ring = {}          # hash_value -> node_name
        self._sorted_keys = []   # sorted list of hash values
        self._nodes = set()      # physical node names

        if nodes:
            for node in nodes:
                self.add_node(node)

    @property
    def nodes(self):
        return frozenset(self._nodes)

    @property
    def num_replicas(self):
        return self._num_replicas

    def _hash_for_vnode(self, node, i):
        """Generate hash for virtual node i of a physical node."""
        return self._hash_fn(f"{node}:{i}")

    def add_node(self, node):
        """Add a physical node (with its virtual nodes) to the ring."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._num_replicas):
            h = self._hash_for_vnode(node, i)
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        """Remove a physical node (and all its virtual nodes) from the ring."""
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self._num_replicas):
            h = self._hash_for_vnode(node, i)
            if h in self._ring:
                del self._ring[h]
                idx = bisect.bisect_left(self._sorted_keys, h)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                    self._sorted_keys.pop(idx)

    def get_node(self, key):
        """Get the node responsible for a key."""
        if not self._sorted_keys:
            return None
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx >= len(self._sorted_keys):
            idx = 0  # wrap around
        return self._ring[self._sorted_keys[idx]]

    def get_nodes(self, key, count=1):
        """Get multiple distinct physical nodes for a key (for replication)."""
        if not self._sorted_keys or count <= 0:
            return []

        result = []
        seen = set()
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)

        for _ in range(len(self._sorted_keys)):
            if idx >= len(self._sorted_keys):
                idx = 0
            node = self._ring[self._sorted_keys[idx]]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) >= count:
                    break
            idx += 1

        return result

    def get_distribution(self, num_keys=10000):
        """Measure key distribution across nodes."""
        if not self._nodes:
            return {}
        counts = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"test_key_{i}")
            counts[node] += 1
        return dict(counts)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes


# --- 2. WeightedHashRing ---

class WeightedHashRing:
    """
    Consistent hash ring where nodes can have different weights.

    A node with weight 2 gets twice as many virtual nodes as weight 1,
    and therefore handles roughly twice the load.
    """

    def __init__(self, base_replicas=100, hash_fn=None):
        self._hash_fn = hash_fn or md5_hash
        self._base_replicas = base_replicas
        self._ring = {}
        self._sorted_keys = []
        self._nodes = {}  # node -> weight

    @property
    def nodes(self):
        return dict(self._nodes)

    def _vnode_count(self, weight):
        return max(1, int(self._base_replicas * weight))

    def add_node(self, node, weight=1.0):
        """Add a node with a given weight."""
        if node in self._nodes:
            self.remove_node(node)
        self._nodes[node] = weight
        count = self._vnode_count(weight)
        for i in range(count):
            h = self._hash_fn(f"{node}:{i}")
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        """Remove a node."""
        if node not in self._nodes:
            return
        weight = self._nodes.pop(node)
        count = self._vnode_count(weight)
        for i in range(count):
            h = self._hash_fn(f"{node}:{i}")
            if h in self._ring:
                del self._ring[h]
                idx = bisect.bisect_left(self._sorted_keys, h)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                    self._sorted_keys.pop(idx)

    def get_node(self, key):
        """Get the node responsible for a key."""
        if not self._sorted_keys:
            return None
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx >= len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    def get_distribution(self, num_keys=10000):
        """Measure key distribution across nodes."""
        if not self._nodes:
            return {}
        counts = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"test_key_{i}")
            counts[node] += 1
        return dict(counts)


# --- 3. BoundedLoadHashRing ---

class BoundedLoadHashRing:
    """
    Google's bounded-load consistent hashing.

    Each node has a capacity cap of ceil(average_load * (1 + epsilon)).
    If the target node is full, we walk clockwise until finding one with capacity.
    This gives O(1/epsilon^2) max load ratio.
    """

    def __init__(self, nodes=None, num_replicas=150, epsilon=0.25, hash_fn=None):
        self._ring = HashRing(nodes=nodes, num_replicas=num_replicas, hash_fn=hash_fn)
        self._epsilon = epsilon
        self._load = defaultdict(int)  # node -> current load
        self._total_load = 0
        self._assignments = {}  # key -> node

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def nodes(self):
        return self._ring.nodes

    def _capacity(self, node=None):
        """Max load per node."""
        n = len(self._ring)
        if n == 0:
            return 0
        avg = max(1, self._total_load / n)
        return math.ceil(avg * (1 + self._epsilon))

    def add_node(self, node):
        """Add a node to the ring."""
        self._ring.add_node(node)

    def remove_node(self, node):
        """Remove a node. Keys assigned to it become unassigned."""
        self._ring.remove_node(node)
        # Reassign keys that were on this node
        orphaned = [k for k, v in self._assignments.items() if v == node]
        load = self._load.pop(node, 0)
        self._total_load -= load
        for k in orphaned:
            del self._assignments[k]

    def assign(self, key):
        """Assign a key to a node respecting load bounds."""
        if key in self._assignments:
            return self._assignments[key]

        if not self._ring._sorted_keys:
            return None

        self._total_load += 1
        cap = self._capacity()

        h = self._ring._hash_fn(key)
        idx = bisect.bisect_right(self._ring._sorted_keys, h)

        for _ in range(len(self._ring._sorted_keys)):
            if idx >= len(self._ring._sorted_keys):
                idx = 0
            node = self._ring._ring[self._ring._sorted_keys[idx]]
            if self._load[node] < cap:
                self._load[node] += 1
                self._assignments[key] = node
                return node
            idx += 1

        # All nodes at capacity -- shouldn't happen but fallback
        node = self._ring.get_node(key)
        self._load[node] += 1
        self._assignments[key] = node
        return node

    def release(self, key):
        """Release a key assignment."""
        if key in self._assignments:
            node = self._assignments.pop(key)
            self._load[node] -= 1
            self._total_load -= 1

    def get_load(self, node):
        """Get current load on a node."""
        return self._load.get(node, 0)

    def get_all_loads(self):
        """Get loads for all nodes."""
        return {node: self._load.get(node, 0) for node in self._ring.nodes}


# --- 4. RendezvousHashing ---

class RendezvousHashing:
    """
    Highest Random Weight (HRW) / Rendezvous hashing.

    For each key, compute a score for every node. Pick the highest score.
    Minimal disruption: adding/removing a node only affects keys that
    map to/from that node.
    """

    def __init__(self, nodes=None, hash_fn=None):
        self._hash_fn = hash_fn or md5_hash
        self._nodes = set(nodes or [])

    @property
    def nodes(self):
        return frozenset(self._nodes)

    def add_node(self, node):
        self._nodes.add(node)

    def remove_node(self, node):
        self._nodes.discard(node)

    def _score(self, node, key):
        """Compute score for a (node, key) pair."""
        return self._hash_fn(f"{node}:{key}")

    def get_node(self, key):
        """Get the node with the highest score for this key."""
        if not self._nodes:
            return None
        return max(self._nodes, key=lambda n: self._score(n, key))

    def get_nodes(self, key, count=1):
        """Get top-N nodes by score for a key."""
        if not self._nodes or count <= 0:
            return []
        scored = sorted(self._nodes, key=lambda n: self._score(n, key), reverse=True)
        return scored[:min(count, len(scored))]

    def get_distribution(self, num_keys=10000):
        """Measure key distribution across nodes."""
        if not self._nodes:
            return {}
        counts = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"test_key_{i}")
            counts[node] += 1
        return dict(counts)


# --- 5. JumpConsistentHash ---

class JumpConsistentHash:
    """
    Google's Jump Consistent Hash.

    O(ln n) time, O(1) space. Returns a bucket number [0, num_buckets).
    Only supports adding/removing the highest-numbered bucket.
    Minimal movement: only 1/n keys move when adding the nth bucket.
    """

    @staticmethod
    def hash(key, num_buckets):
        """
        Jump consistent hash. Returns bucket in [0, num_buckets).

        Uses the algorithm from Lamping & Veach (2014).
        """
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")

        if isinstance(key, str):
            key = md5_hash(key)

        b = -1
        j = 0
        while j < num_buckets:
            b = j
            key = ((key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
            j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))

        return b

    @staticmethod
    def distribute(keys, num_buckets):
        """Distribute a list of keys across buckets."""
        dist = defaultdict(list)
        for key in keys:
            bucket = JumpConsistentHash.hash(key, num_buckets)
            dist[bucket].append(key)
        return dict(dist)


# --- 6. MultiProbeHashRing ---

class MultiProbeHashRing:
    """
    Multi-probe consistent hashing (Appleton & O'Reilly, 2015).

    Uses fewer virtual nodes but probes multiple positions on the ring
    for each lookup. Achieves good balance with O(1) memory per node.
    """

    def __init__(self, nodes=None, num_probes=21, hash_fn=None):
        self._hash_fn = hash_fn or md5_hash
        self._num_probes = num_probes
        self._ring = {}
        self._sorted_keys = []
        self._nodes = set()

        if nodes:
            for node in nodes:
                self.add_node(node)

    @property
    def nodes(self):
        return frozenset(self._nodes)

    def add_node(self, node):
        """Each node gets exactly one position on the ring."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        h = self._hash_fn(node)
        self._ring[h] = node
        bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        h = self._hash_fn(node)
        if h in self._ring:
            del self._ring[h]
            idx = bisect.bisect_left(self._sorted_keys, h)
            if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                self._sorted_keys.pop(idx)

    def get_node(self, key):
        """
        Multi-probe lookup: hash key with multiple seeds,
        find closest node for each probe, return the overall closest.
        """
        if not self._sorted_keys:
            return None

        best_node = None
        best_distance = RING_SIZE + 1

        for probe in range(self._num_probes):
            h = self._hash_fn(f"{key}:{probe}")
            idx = bisect.bisect_right(self._sorted_keys, h)
            if idx >= len(self._sorted_keys):
                idx = 0

            ring_pos = self._sorted_keys[idx]
            # Clockwise distance on the ring
            dist = (ring_pos - h) % RING_SIZE

            if dist < best_distance:
                best_distance = dist
                best_node = self._ring[ring_pos]

        return best_node

    def get_distribution(self, num_keys=10000):
        if not self._nodes:
            return {}
        counts = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"test_key_{i}")
            counts[node] += 1
        return dict(counts)


# --- 7. MaglevHashRing ---

class MaglevHashRing:
    """
    Google Maglev consistent hashing.

    Builds a lookup table of size M (prime) where each entry maps to a node.
    Provides uniform distribution and minimal disruption.
    Table size M should be >> number of nodes for good distribution.
    """

    def __init__(self, nodes=None, table_size=65537, hash_fn=None):
        self._hash_fn = hash_fn or md5_hash
        self._table_size = self._next_prime(table_size)
        self._nodes = list(nodes or [])
        self._node_set = set(self._nodes)
        self._table = None
        if self._nodes:
            self._build_table()

    @staticmethod
    def _next_prime(n):
        """Find the smallest prime >= n."""
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        while True:
            if MaglevHashRing._is_prime(n):
                return n
            n += 2

    @staticmethod
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _build_table(self):
        """Build the Maglev lookup table using permutation-based population."""
        n = len(self._nodes)
        M = self._table_size

        if n == 0:
            self._table = None
            return

        # Generate permutation for each node
        permutations = []
        for node in self._nodes:
            offset = self._hash_fn(f"{node}:offset") % M
            skip = (self._hash_fn(f"{node}:skip") % (M - 1)) + 1
            perm = [(offset + j * skip) % M for j in range(M)]
            permutations.append(perm)

        # Fill the table
        table = [None] * M
        next_idx = [0] * n  # next position in each node's permutation
        filled = 0

        while filled < M:
            for i in range(n):
                # Find next empty slot for this node
                while next_idx[i] < M:
                    slot = permutations[i][next_idx[i]]
                    next_idx[i] += 1
                    if table[slot] is None:
                        table[slot] = self._nodes[i]
                        filled += 1
                        break
                if filled >= M:
                    break

        self._table = table

    @property
    def nodes(self):
        return frozenset(self._node_set)

    def add_node(self, node):
        if node in self._node_set:
            return
        self._nodes.append(node)
        self._node_set.add(node)
        self._build_table()

    def remove_node(self, node):
        if node not in self._node_set:
            return
        self._nodes.remove(node)
        self._node_set.discard(node)
        self._build_table()

    def get_node(self, key):
        """O(1) lookup via the precomputed table."""
        if not self._table:
            return None
        h = self._hash_fn(key) % self._table_size
        return self._table[h]

    def get_distribution(self, num_keys=10000):
        if not self._table:
            return {}
        counts = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"test_key_{i}")
            counts[node] += 1
        return dict(counts)


# --- 8. ReplicatedHashRing ---

class ReplicatedHashRing:
    """
    Hash ring with replication factor and preference lists.

    Combines consistent hashing with replication for fault tolerance.
    Supports coordinator selection and preference list generation
    (like Amazon Dynamo).
    """

    def __init__(self, nodes=None, num_replicas=150, replication_factor=3, hash_fn=None):
        self._ring = HashRing(nodes=nodes, num_replicas=num_replicas, hash_fn=hash_fn)
        self._replication_factor = replication_factor
        self._failed_nodes = set()

    @property
    def nodes(self):
        return self._ring.nodes

    @property
    def replication_factor(self):
        return self._replication_factor

    @property
    def healthy_nodes(self):
        return self._ring.nodes - self._failed_nodes

    def add_node(self, node):
        self._ring.add_node(node)

    def remove_node(self, node):
        self._ring.remove_node(node)
        self._failed_nodes.discard(node)

    def mark_failed(self, node):
        """Mark a node as failed."""
        self._failed_nodes.add(node)

    def mark_healthy(self, node):
        """Mark a node as healthy again."""
        self._failed_nodes.discard(node)

    def get_preference_list(self, key):
        """
        Get the preference list for a key.

        Returns replication_factor distinct healthy nodes, walking clockwise.
        If not enough healthy nodes, returns what's available.
        """
        all_nodes = self._ring.get_nodes(key, count=len(self._ring))
        healthy = [n for n in all_nodes if n not in self._failed_nodes]
        return healthy[:self._replication_factor]

    def get_coordinator(self, key):
        """Get the coordinator (first healthy node) for a key."""
        pref = self.get_preference_list(key)
        return pref[0] if pref else None

    def get_handoff_node(self, key, failed_node):
        """
        Get a handoff node for hinted handoff when a node fails.

        Returns the next healthy node not already in the preference list.
        """
        all_nodes = self._ring.get_nodes(key, count=len(self._ring))
        pref = set(self.get_preference_list(key))
        pref.add(failed_node)

        for node in all_nodes:
            if node not in pref and node not in self._failed_nodes:
                return node
        return None

    def get_key_ranges(self, node):
        """
        Get the token ranges a node is responsible for.

        Returns list of (start, end) ranges on the ring.
        """
        if node not in self._ring._nodes:
            return []

        ranges = []
        for i in range(self._ring._num_replicas):
            h = self._ring._hash_for_vnode(node, i)
            # Find the predecessor
            idx = bisect.bisect_left(self._ring._sorted_keys, h)
            if idx == 0:
                prev = self._ring._sorted_keys[-1]
            else:
                prev = self._ring._sorted_keys[idx - 1]
            ranges.append((prev + 1, h))

        return ranges
