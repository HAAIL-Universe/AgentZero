"""
C234: Consistent Hashing

A comprehensive consistent hashing library for distributed systems.

Components:
1. HashRing -- core consistent hash ring with virtual nodes
2. WeightedHashRing -- nodes with different weights (more vnodes = more load)
3. BoundedLoadHashRing -- Google's bounded-load consistent hashing
4. JumpHash -- Jump Consistent Hash (Lamport/Thorup, no ring needed)
5. RendezvousHash -- Highest Random Weight (HRW) hashing
6. MultiProbeHash -- multi-probe consistent hashing (fewer vnodes, same uniformity)
7. MaglevHash -- Google Maglev lookup table hashing (O(1) lookup)
"""

import hashlib
import struct
import math
import bisect
from collections import defaultdict


# --- Utilities ---

def _md5_hash(key):
    """Hash a key to a 32-bit integer using MD5."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    digest = hashlib.md5(key).digest()
    return struct.unpack('<I', digest[:4])[0]


def _sha256_hash(key):
    """Hash a key to a 32-bit integer using SHA-256."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    digest = hashlib.sha256(key).digest()
    return struct.unpack('<I', digest[:4])[0]


def _multi_hash(key, count):
    """Generate multiple hash values from a single key."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    hashes = []
    for i in range(count):
        h = hashlib.md5(key + struct.pack('<I', i)).digest()
        hashes.append(struct.unpack('<I', h[:4])[0])
    return hashes


RING_SIZE = 2**32


# --- 1. HashRing ---

class HashRing:
    """
    Consistent hash ring with virtual nodes.

    Each physical node maps to `num_replicas` virtual nodes on the ring.
    Keys are assigned to the next virtual node clockwise on the ring.
    """

    def __init__(self, nodes=None, num_replicas=150, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._num_replicas = num_replicas
        self._ring = {}          # hash_value -> node_name
        self._sorted_keys = []   # sorted list of hash values
        self._nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash_for_vnode(self, node, i):
        return self._hash_fn(f"{node}#vnode{i}")

    def add_node(self, node):
        """Add a node with its virtual nodes to the ring."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._num_replicas):
            h = self._hash_for_vnode(node, i)
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        """Remove a node and all its virtual nodes from the ring."""
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
        """Get the node responsible for the given key."""
        if not self._sorted_keys:
            return None
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx >= len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    def get_nodes(self, key, count=1):
        """Get `count` distinct nodes for the key (for replication)."""
        if not self._sorted_keys or count <= 0:
            return []
        if count > len(self._nodes):
            count = len(self._nodes)

        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)

        result = []
        seen = set()
        n = len(self._sorted_keys)
        for i in range(n):
            pos = (idx + i) % n
            node = self._ring[self._sorted_keys[pos]]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) == count:
                    break
        return result

    @property
    def nodes(self):
        return set(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes

    def get_distribution(self, num_keys=10000):
        """Test key distribution across nodes. Returns {node: count}."""
        dist = defaultdict(int)
        for i in range(num_keys):
            node = self.get_node(f"key_{i}")
            if node:
                dist[node] += 1
        return dict(dist)


# --- 2. WeightedHashRing ---

class WeightedHashRing:
    """
    Consistent hash ring where nodes have weights.

    A node with weight 2 gets twice as many virtual nodes as weight 1,
    and thus handles roughly twice as many keys.
    """

    def __init__(self, base_replicas=100, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._base_replicas = base_replicas
        self._ring = {}
        self._sorted_keys = []
        self._nodes = {}  # node -> weight

    def _replicas_for(self, node):
        weight = self._nodes.get(node, 1)
        return max(1, int(self._base_replicas * weight))

    def add_node(self, node, weight=1.0):
        """Add a node with the given weight."""
        if node in self._nodes:
            self.remove_node(node)
        self._nodes[node] = weight
        num = self._replicas_for(node)
        for i in range(num):
            h = self._hash_fn(f"{node}#vnode{i}")
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        """Remove a node."""
        if node not in self._nodes:
            return
        num = self._replicas_for(node)
        del self._nodes[node]
        for i in range(num):
            h = self._hash_fn(f"{node}#vnode{i}")
            if h in self._ring:
                del self._ring[h]
                idx = bisect.bisect_left(self._sorted_keys, h)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                    self._sorted_keys.pop(idx)

    def get_node(self, key):
        if not self._sorted_keys:
            return None
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx >= len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]

    def get_nodes(self, key, count=1):
        if not self._sorted_keys or count <= 0:
            return []
        if count > len(self._nodes):
            count = len(self._nodes)
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        result = []
        seen = set()
        n = len(self._sorted_keys)
        for i in range(n):
            pos = (idx + i) % n
            node = self._ring[self._sorted_keys[pos]]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) == count:
                    break
        return result

    def get_weight(self, node):
        return self._nodes.get(node, 0)

    @property
    def nodes(self):
        return dict(self._nodes)

    def __len__(self):
        return len(self._nodes)


# --- 3. BoundedLoadHashRing ---

class BoundedLoadHashRing:
    """
    Bounded-load consistent hashing (Google, 2017).

    Each node has a capacity of ceil(avg_load * (1 + epsilon)).
    If a node is full, the key slides to the next node on the ring.
    This ensures no node is overloaded by more than (1+epsilon) factor.
    """

    def __init__(self, nodes=None, num_replicas=150, epsilon=0.25, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._num_replicas = num_replicas
        self._epsilon = epsilon
        self._ring = {}
        self._sorted_keys = []
        self._nodes = set()
        self._load = defaultdict(int)  # node -> current load
        self._total_load = 0
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash_for_vnode(self, node, i):
        return self._hash_fn(f"{node}#vnode{i}")

    def add_node(self, node):
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._num_replicas):
            h = self._hash_for_vnode(node, i)
            self._ring[h] = node
            bisect.insort(self._sorted_keys, h)

    def remove_node(self, node):
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        if node in self._load:
            self._total_load -= self._load[node]
            del self._load[node]
        for i in range(self._num_replicas):
            h = self._hash_for_vnode(node, i)
            if h in self._ring:
                del self._ring[h]
                idx = bisect.bisect_left(self._sorted_keys, h)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == h:
                    self._sorted_keys.pop(idx)

    @property
    def capacity_per_node(self):
        """Max load allowed per node."""
        if not self._nodes:
            return 0
        avg = max(1, self._total_load) / len(self._nodes)
        return math.ceil(avg * (1 + self._epsilon))

    def get_node(self, key):
        """Get node for key, respecting load bounds."""
        if not self._sorted_keys:
            return None
        h = self._hash_fn(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        cap = self.capacity_per_node
        n = len(self._sorted_keys)
        for i in range(n):
            pos = (idx + i) % n
            node = self._ring[self._sorted_keys[pos]]
            if self._load[node] < cap:
                self._load[node] += 1
                self._total_load += 1
                return node
        # All nodes at capacity -- assign to natural node anyway
        node = self._ring[self._sorted_keys[idx % n]]
        self._load[node] += 1
        self._total_load += 1
        return node

    def release(self, key, node):
        """Release a key's load from a node."""
        if self._load[node] > 0:
            self._load[node] -= 1
            self._total_load -= 1

    def reset_load(self):
        """Reset all load counters."""
        self._load.clear()
        self._total_load = 0

    def get_load(self, node):
        return self._load.get(node, 0)

    @property
    def nodes(self):
        return set(self._nodes)

    def __len__(self):
        return len(self._nodes)


# --- 4. JumpHash ---

class JumpHash:
    """
    Jump Consistent Hash (Lamport, Thorup 2014).

    O(ln n) time, O(1) space, no ring structure needed.
    Maps a key to one of `num_buckets` buckets with near-perfect uniformity.
    Minimal disruption: when adding bucket n+1, only ~1/(n+1) keys move.

    Limitation: only supports adding/removing the LAST bucket.
    """

    @staticmethod
    def hash(key, num_buckets):
        """Map key to a bucket in [0, num_buckets)."""
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        if isinstance(key, str):
            key = _md5_hash(key)

        b = -1
        j = 0
        while j < num_buckets:
            b = j
            key = ((key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
            j = int((b + 1) * (2**31 / ((key >> 33) + 1)))
        return b

    @staticmethod
    def hash_with_names(key, buckets):
        """Map key to a named bucket from the list."""
        if not buckets:
            return None
        idx = JumpHash.hash(key, len(buckets))
        return buckets[idx]


# --- 5. RendezvousHash ---

class RendezvousHash:
    """
    Highest Random Weight (HRW) / Rendezvous Hashing.

    For each key, compute hash(key, node) for all nodes.
    The node with the highest hash wins.

    O(n) per lookup but excellent uniformity and minimal disruption.
    Supports arbitrary node add/remove (unlike JumpHash).
    """

    def __init__(self, nodes=None, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._nodes = list(nodes) if nodes else []

    def add_node(self, node):
        if node not in self._nodes:
            self._nodes.append(node)

    def remove_node(self, node):
        if node in self._nodes:
            self._nodes.remove(node)

    def get_node(self, key):
        """Get the node with the highest hash for this key."""
        if not self._nodes:
            return None
        best_node = None
        best_hash = -1
        for node in self._nodes:
            h = self._hash_fn(f"{key}:{node}")
            if h > best_hash:
                best_hash = h
                best_node = node
        return best_node

    def get_nodes(self, key, count=1):
        """Get top-`count` nodes by hash weight."""
        if not self._nodes or count <= 0:
            return []
        count = min(count, len(self._nodes))
        scored = []
        for node in self._nodes:
            h = self._hash_fn(f"{key}:{node}")
            scored.append((h, node))
        scored.sort(reverse=True)
        return [node for _, node in scored[:count]]

    @property
    def nodes(self):
        return list(self._nodes)

    def __len__(self):
        return len(self._nodes)


# --- 6. MultiProbeHash ---

class MultiProbeHash:
    """
    Multi-Probe Consistent Hashing (Appleton & O'Reilly, 2015).

    Instead of many virtual nodes per physical node, each physical node
    gets ONE position on the ring, but each key lookup probes multiple
    hash positions and picks the closest node.

    Achieves similar uniformity to virtual nodes with much less memory.
    """

    def __init__(self, nodes=None, num_probes=21, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._num_probes = num_probes
        self._ring = {}          # hash_value -> node
        self._sorted_keys = []
        self._nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
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

    def _find_closest_node(self, h):
        """Find the closest node clockwise from hash h."""
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx >= len(self._sorted_keys):
            idx = 0
        ring_key = self._sorted_keys[idx]
        # Distance clockwise on the ring
        dist = (ring_key - h) % RING_SIZE
        return self._ring[ring_key], dist

    def get_node(self, key):
        """Get node using multi-probe lookup."""
        if not self._sorted_keys:
            return None

        best_node = None
        best_dist = RING_SIZE

        for i in range(self._num_probes):
            h = self._hash_fn(f"{key}#{i}")
            node, dist = self._find_closest_node(h)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    @property
    def nodes(self):
        return set(self._nodes)

    def __len__(self):
        return len(self._nodes)


# --- 7. MaglevHash ---

class MaglevHash:
    """
    Google Maglev Consistent Hashing (2016).

    Builds a fixed-size lookup table for O(1) key-to-node mapping.
    When a node is added/removed, the table is rebuilt but most entries
    stay the same (minimal disruption).

    Table size should be a prime number >= num_nodes for best results.
    """

    def __init__(self, nodes=None, table_size=65537, hash_fn=None):
        self._hash_fn = hash_fn or _md5_hash
        self._table_size = self._next_prime(table_size)
        self._nodes = list(nodes) if nodes else []
        self._table = []
        if self._nodes:
            self._build_table()

    @staticmethod
    def _next_prime(n):
        """Find the next prime >= n."""
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        while True:
            if MaglevHash._is_prime(n):
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
        """Build the Maglev permutation lookup table."""
        M = self._table_size
        n = len(self._nodes)
        if n == 0:
            self._table = []
            return

        # Generate offset and skip for each node
        permutation = []
        for node in self._nodes:
            offset = self._hash_fn(f"{node}:offset") % M
            skip = (self._hash_fn(f"{node}:skip") % (M - 1)) + 1
            permutation.append((offset, skip))

        # Build table using round-robin permutation filling
        table = [-1] * M
        next_idx = [0] * n  # current position in each node's permutation
        filled = 0

        while filled < M:
            for i in range(n):
                if filled >= M:
                    break
                offset, skip = permutation[i]
                # Find next empty slot in this node's permutation
                c = (offset + next_idx[i] * skip) % M
                while table[c] != -1:
                    next_idx[i] += 1
                    c = (offset + next_idx[i] * skip) % M
                table[c] = i
                next_idx[i] += 1
                filled += 1

        self._table = table

    def add_node(self, node):
        if node in self._nodes:
            return
        self._nodes.append(node)
        self._build_table()

    def remove_node(self, node):
        if node not in self._nodes:
            return
        self._nodes.remove(node)
        self._build_table()

    def get_node(self, key):
        """O(1) lookup after table is built."""
        if not self._table:
            return None
        h = self._hash_fn(key)
        idx = h % self._table_size
        node_idx = self._table[idx]
        return self._nodes[node_idx]

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def table_size(self):
        return self._table_size

    def __len__(self):
        return len(self._nodes)


# --- Utility: Migration calculator ---

def calculate_migration(old_mapping, new_mapping):
    """
    Calculate how many keys migrated between two mappings.

    Returns (moved_count, total_count, fraction_moved).
    """
    total = len(old_mapping)
    moved = sum(1 for k in old_mapping if k in new_mapping and old_mapping[k] != new_mapping[k])
    return moved, total, moved / total if total > 0 else 0.0


def measure_balance(distribution, expected=None):
    """
    Measure how balanced a distribution is.

    Returns (std_dev, coefficient_of_variation, max_deviation_from_mean).
    """
    if not distribution:
        return 0.0, 0.0, 0.0
    values = list(distribution.values())
    n = len(values)
    mean = sum(values) / n
    if mean == 0:
        return 0.0, 0.0, 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance)
    cv = std_dev / mean
    max_dev = max(abs(v - mean) for v in values) / mean
    return std_dev, cv, max_dev
