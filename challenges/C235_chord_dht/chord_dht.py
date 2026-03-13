"""
C235: Chord Distributed Hash Table

A comprehensive implementation of the Chord DHT protocol (Stoica et al., 2001).

Components:
1. ChordNode -- individual node with finger table, successor list, key storage
2. ChordRing -- simulated network of ChordNodes (local simulation, no sockets)
3. Stabilization -- periodic stabilize/fix_fingers/check_predecessor
4. Replication -- key replication across successor list
5. VirtualNodes -- multiple virtual nodes per physical machine
6. RangeQueries -- scan keys in an ID range
7. ConsistentMigration -- minimal key movement on join/leave
"""

import hashlib
import struct
import math
import random
from collections import defaultdict


# --- Configuration ---

DEFAULT_M = 8          # bits in identifier space (2^m = 256 for testing)
DEFAULT_SUCCESSORS = 3  # successor list size for replication
DEFAULT_REPLICAS = 3    # key replication factor


def _hash_to_m(key, m):
    """Hash a key to m-bit identifier space [0, 2^m)."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    elif isinstance(key, int):
        key = str(key).encode('utf-8')
    digest = hashlib.sha256(key).digest()
    value = struct.unpack('>I', digest[:4])[0]
    return value % (2 ** m)


def in_range(x, a, b, ring_size):
    """Check if x is in (a, b] on a circular ring of given size.

    Convention: (a, b] means a < x <= b, wrapping around.
    """
    if a == b:
        return True  # full ring
    if a < b:
        return a < x <= b
    else:
        return x > a or x <= b


def in_range_open(x, a, b, ring_size):
    """Check if x is in (a, b) on a circular ring -- open on both ends."""
    if a == b:
        return x != a
    if a < b:
        return a < x < b
    else:
        return x > a or x < b


def distance(a, b, ring_size):
    """Clockwise distance from a to b on the ring."""
    return (b - a) % ring_size


# --- ChordNode ---

class ChordNode:
    """
    A single node in the Chord ring.

    Each node maintains:
    - finger table: finger[i] = successor of (node_id + 2^i) mod 2^m
    - successor list: list of `r` immediate successors (for fault tolerance)
    - predecessor pointer
    - local key-value store
    """

    def __init__(self, node_id, m=DEFAULT_M, num_successors=DEFAULT_SUCCESSORS):
        self.node_id = node_id
        self.m = m
        self.ring_size = 2 ** m
        self.num_successors = num_successors

        # Finger table: finger[i].node is successor of (node_id + 2^i)
        self.finger = [None] * m
        self.predecessor = None
        self.successor_list = []

        # Key-value store -- keyed by key_name for collision safety
        self.store = {}       # key_name -> value
        self.replicas = {}    # key_name -> value -- replicated from predecessor

        # Network reference (set by ChordRing)
        self._network = None

        # Metadata
        self.alive = True
        self.transfer_log = []  # track key transfers

    @property
    def successor(self):
        """First entry in finger table is the immediate successor."""
        return self.finger[0]

    @successor.setter
    def successor(self, node):
        self.finger[0] = node

    def _lookup_node(self, node_id):
        """Look up a node in the network by ID."""
        if self._network is None:
            return None
        return self._network.get_node(node_id)

    def find_successor(self, id_val):
        """Find the successor node for the given identifier.

        Core Chord lookup: O(log N) hops via finger table.
        """
        # Check if id falls between us and our successor
        if self.successor is None:
            return self.node_id

        if in_range(id_val, self.node_id, self.successor, self.ring_size):
            return self.successor

        # Find closest preceding node and ask it
        n0 = self.closest_preceding_node(id_val)
        if n0 == self.node_id:
            return self.successor if self.successor is not None else self.node_id

        node = self._lookup_node(n0)
        if node is None or not node.alive:
            return self.successor if self.successor is not None else self.node_id
        return node.find_successor(id_val)

    def closest_preceding_node(self, id_val):
        """Find the closest preceding node for id in the finger table."""
        for i in range(self.m - 1, -1, -1):
            f = self.finger[i]
            if f is not None and in_range_open(f, self.node_id, id_val, self.ring_size):
                node = self._lookup_node(f)
                if node is not None and node.alive:
                    return f
        return self.node_id

    def join(self, existing_node_id=None):
        """Join the Chord ring.

        If existing_node_id is None, this node forms a new ring alone.
        Otherwise, join via the existing node.
        """
        if existing_node_id is None:
            # Create a new ring with just this node
            self.predecessor = None
            self.successor = self.node_id
            self.successor_list = [self.node_id]
            for i in range(self.m):
                self.finger[i] = self.node_id
        else:
            self.predecessor = None
            existing = self._lookup_node(existing_node_id)
            if existing is None:
                raise ValueError(f"Node {existing_node_id} not found in network")
            succ = existing.find_successor(self.node_id)
            self.successor = succ
            self.successor_list = [succ]
            # Initialize fingers to successor (will be fixed by fix_fingers)
            for i in range(self.m):
                self.finger[i] = succ

    def stabilize(self):
        """Periodic stabilization: verify successor and update successor list.

        Called periodically. Asks successor for its predecessor, and updates
        if there's a closer node between us and our successor.
        """
        if self.successor is None:
            return

        succ_node = self._lookup_node(self.successor)
        if succ_node is None or not succ_node.alive:
            # Try successor list
            self._recover_successor()
            return

        x = succ_node.predecessor
        if x is not None and in_range_open(x, self.node_id, self.successor, self.ring_size):
            x_node = self._lookup_node(x)
            if x_node is not None and x_node.alive:
                self.successor = x
                self.finger[0] = x

        # Notify successor
        succ_node = self._lookup_node(self.successor)
        if succ_node is not None and succ_node.alive:
            succ_node.notify(self.node_id)

        # Update successor list
        self._update_successor_list()

    def notify(self, candidate_id):
        """Called by a node that thinks it might be our predecessor."""
        if self.predecessor is None:
            self.predecessor = candidate_id
        elif in_range_open(candidate_id, self.predecessor, self.node_id, self.ring_size):
            pred_node = self._lookup_node(self.predecessor)
            if pred_node is None or not pred_node.alive:
                self.predecessor = candidate_id
            else:
                self.predecessor = candidate_id

    def fix_fingers(self, index=None):
        """Fix a single finger table entry (called periodically).

        If index is None, fix a random finger.
        """
        if index is None:
            index = random.randint(0, self.m - 1)

        start = (self.node_id + 2**index) % self.ring_size
        self.finger[index] = self.find_successor(start)

    def fix_all_fingers(self):
        """Fix all finger table entries at once."""
        for i in range(self.m):
            start = (self.node_id + 2**i) % self.ring_size
            self.finger[i] = self.find_successor(start)

    def check_predecessor(self):
        """Check if predecessor is still alive."""
        if self.predecessor is not None:
            pred = self._lookup_node(self.predecessor)
            if pred is None or not pred.alive:
                self.predecessor = None

    def _update_successor_list(self):
        """Update the successor list by querying successors."""
        new_list = []
        current = self.successor
        seen = {self.node_id}

        for _ in range(self.num_successors):
            if current is None or current in seen:
                break
            seen.add(current)
            new_list.append(current)
            node = self._lookup_node(current)
            if node is None or not node.alive:
                break
            current = node.successor

        if new_list:
            self.successor_list = new_list

    def _recover_successor(self):
        """Try to recover from a failed successor using successor list."""
        for s in self.successor_list:
            if s == self.successor:
                continue
            node = self._lookup_node(s)
            if node is not None and node.alive:
                self.successor = s
                self.finger[0] = s
                return
        # Last resort: point to self
        self.successor = self.node_id
        self.finger[0] = self.node_id

    # --- Key-Value Operations ---

    def put(self, key, value):
        """Store a key-value pair at the responsible node."""
        key_id = _hash_to_m(key, self.m)
        target_id = self.find_successor(key_id)
        target = self._lookup_node(target_id)
        if target is None:
            return False
        target.store[key] = value
        target.transfer_log.append(('put', key, key_id))
        return True

    def get(self, key):
        """Retrieve a value by key from the responsible node."""
        key_id = _hash_to_m(key, self.m)
        target_id = self.find_successor(key_id)
        target = self._lookup_node(target_id)
        if target is None:
            return None
        if key in target.store:
            return target.store[key]
        if key in target.replicas:
            return target.replicas[key]
        return None

    def delete(self, key):
        """Delete a key from the responsible node."""
        key_id = _hash_to_m(key, self.m)
        target_id = self.find_successor(key_id)
        target = self._lookup_node(target_id)
        if target is None:
            return False
        if key in target.store:
            del target.store[key]
            return True
        return False

    def local_keys(self):
        """Return all keys stored locally (primary + replicas)."""
        result = dict(self.store)
        return result

    def responsible_for(self, key_id):
        """Check if this node is responsible for the given key ID."""
        if self.predecessor is None:
            return True  # only node in ring
        return in_range(key_id, self.predecessor, self.node_id, self.ring_size)

    def transfer_keys_to(self, new_node_id):
        """Transfer keys that should belong to the new node."""
        new_node = self._lookup_node(new_node_id)
        if new_node is None:
            return 0

        transferred = 0
        to_transfer = []

        for key, value in list(self.store.items()):
            key_id = _hash_to_m(key, self.m)
            if in_range(key_id, new_node.predecessor if new_node.predecessor is not None else self.node_id,
                       new_node_id, self.ring_size):
                to_transfer.append((key, value, key_id))

        for key, value, key_id in to_transfer:
            new_node.store[key] = value
            del self.store[key]
            new_node.transfer_log.append(('transfer_in', key, key_id))
            self.transfer_log.append(('transfer_out', key, key_id))
            transferred += 1

        return transferred

    def __repr__(self):
        return f"ChordNode(id={self.node_id}, succ={self.successor}, pred={self.predecessor})"


# --- ChordRing (Simulated Network) ---

class ChordRing:
    """
    Simulated Chord ring -- manages a set of ChordNodes.

    This is a local simulation (no actual networking).
    Provides methods to add/remove nodes, run stabilization,
    and perform key-value operations.
    """

    def __init__(self, m=DEFAULT_M, num_successors=DEFAULT_SUCCESSORS, replicas=DEFAULT_REPLICAS):
        self.m = m
        self.ring_size = 2 ** m
        self.num_successors = num_successors
        self.replicas = replicas
        self.nodes = {}  # node_id -> ChordNode
        self._node_order = []  # sorted node IDs for convenience

    def get_node(self, node_id):
        """Get a node by ID (used by nodes for network lookups)."""
        return self.nodes.get(node_id)

    def add_node(self, node_id=None):
        """Add a new node to the ring. Returns the node."""
        if node_id is None:
            # Generate a random ID not already in use
            while True:
                node_id = random.randint(0, self.ring_size - 1)
                if node_id not in self.nodes:
                    break

        if node_id in self.nodes:
            return self.nodes[node_id]

        node = ChordNode(node_id, self.m, self.num_successors)
        node._network = self
        self.nodes[node_id] = node

        # Join via any existing node
        if len(self.nodes) == 1:
            node.join(None)  # first node
        else:
            existing = next(nid for nid in self.nodes if nid != node_id)
            node.join(existing)

        self._update_order()

        # Run stabilization to integrate the new node
        self._stabilize_rounds(3)

        # Transfer keys from successor
        if node.successor != node_id:
            succ = self.nodes.get(node.successor)
            if succ:
                succ.transfer_keys_to(node_id)

        return node

    def remove_node(self, node_id):
        """Gracefully remove a node from the ring.

        Transfers keys to successor before leaving.
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Transfer all keys to successor
        if node.successor != node_id and node.successor in self.nodes:
            succ = self.nodes[node.successor]
            for key, value in list(node.store.items()):
                succ.store[key] = value
            node.store.clear()

        # Mark as dead and remove
        node.alive = False
        del self.nodes[node_id]
        self._update_order()

        # Stabilize remaining nodes
        self._stabilize_rounds(3)

        return True

    def fail_node(self, node_id):
        """Simulate a node failure (crash, no key transfer)."""
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        node.alive = False
        del self.nodes[node_id]
        self._update_order()
        return True

    def _update_order(self):
        """Update sorted node order."""
        self._node_order = sorted(self.nodes.keys())

    def _stabilize_rounds(self, rounds=1):
        """Run stabilization on all nodes for the given number of rounds."""
        for _ in range(rounds):
            for node in list(self.nodes.values()):
                if node.alive:
                    node.stabilize()
            for node in list(self.nodes.values()):
                if node.alive:
                    node.fix_all_fingers()
            for node in list(self.nodes.values()):
                if node.alive:
                    node.check_predecessor()

    def stabilize(self, rounds=1):
        """Public stabilization -- run multiple rounds."""
        self._stabilize_rounds(rounds)

    # --- Key-Value Interface ---

    def put(self, key, value):
        """Put a key-value pair into the DHT."""
        if not self.nodes:
            return False
        entry_node = self._any_node()
        return entry_node.put(key, value)

    def get(self, key):
        """Get a value from the DHT."""
        if not self.nodes:
            return None
        entry_node = self._any_node()
        return entry_node.get(key)

    def delete(self, key):
        """Delete a key from the DHT."""
        if not self.nodes:
            return False
        entry_node = self._any_node()
        return entry_node.delete(key)

    def put_replicated(self, key, value):
        """Put a key-value pair with replication across successor list."""
        if not self.nodes:
            return False
        key_id = _hash_to_m(key, self.m)
        entry_node = self._any_node()
        target_id = entry_node.find_successor(key_id)
        target = self.nodes.get(target_id)
        if target is None:
            return False

        # Primary copy
        target.store[key] = value

        # Replicate to successors
        for succ_id in target.successor_list[:self.replicas - 1]:
            if succ_id != target_id:
                succ = self.nodes.get(succ_id)
                if succ is not None and succ.alive:
                    succ.replicas[key] = value

        return True

    def get_replicated(self, key):
        """Get a value, falling back to replicas if primary fails."""
        if not self.nodes:
            return None
        key_id = _hash_to_m(key, self.m)
        entry_node = self._any_node()
        target_id = entry_node.find_successor(key_id)
        target = self.nodes.get(target_id)

        if target is not None:
            if key in target.store:
                return target.store[key]
            if key in target.replicas:
                return target.replicas[key]

        # Try all nodes (primary failed)
        for node in self.nodes.values():
            if node.alive:
                if key in node.store:
                    return node.store[key]
                if key in node.replicas:
                    return node.replicas[key]

        return None

    def _any_node(self):
        """Get any alive node as an entry point."""
        for node in self.nodes.values():
            if node.alive:
                return node
        return None

    # --- Diagnostics ---

    def get_ring_order(self):
        """Get nodes in ring order (by ID)."""
        return list(self._node_order)

    def get_all_keys(self):
        """Get all keys stored across the DHT."""
        result = {}
        for node in self.nodes.values():
            for key, value in node.store.items():
                result[key] = value
        return result

    def get_key_distribution(self):
        """Get the distribution of keys across nodes."""
        dist = {}
        for nid, node in self.nodes.items():
            dist[nid] = len(node.store)
        return dist

    def verify_ring_integrity(self):
        """Verify the ring is properly connected.

        Returns (is_valid, issues_list).
        """
        issues = []

        if not self.nodes:
            return True, []

        # Check each node's successor and predecessor
        for nid, node in self.nodes.items():
            if node.successor is None:
                issues.append(f"Node {nid} has no successor")
            elif node.successor not in self.nodes:
                issues.append(f"Node {nid}'s successor {node.successor} not in ring")

            if node.predecessor is not None and node.predecessor not in self.nodes:
                issues.append(f"Node {nid}'s predecessor {node.predecessor} not in ring")

        # Check ring connectivity: traversing successors should visit all nodes
        if self.nodes:
            start = self._node_order[0]
            visited = set()
            current = start
            for _ in range(len(self.nodes) + 1):
                if current in visited:
                    break
                visited.add(current)
                node = self.nodes[current]
                current = node.successor

            if visited != set(self.nodes.keys()):
                missing = set(self.nodes.keys()) - visited
                issues.append(f"Ring not fully connected. Missing: {missing}")

        return len(issues) == 0, issues

    def get_finger_table(self, node_id):
        """Get a node's finger table as readable data."""
        node = self.nodes.get(node_id)
        if node is None:
            return None

        table = []
        for i in range(node.m):
            start = (node.node_id + 2**i) % self.ring_size
            table.append({
                'index': i,
                'start': start,
                'node': node.finger[i]
            })
        return table

    def get_lookup_path(self, key):
        """Trace the lookup path for a key (for debugging)."""
        if not self.nodes:
            return []

        key_id = _hash_to_m(key, self.m)
        entry = self._any_node()
        path = [entry.node_id]

        current = entry
        visited = set()

        while True:
            if current.node_id in visited:
                break
            visited.add(current.node_id)

            if current.successor is not None and \
               in_range(key_id, current.node_id, current.successor, self.ring_size):
                if current.successor != current.node_id and current.successor not in visited:
                    path.append(current.successor)
                break

            next_id = current.closest_preceding_node(key_id)
            if next_id == current.node_id:
                if current.successor is not None and current.successor != current.node_id:
                    path.append(current.successor)
                break

            path.append(next_id)
            next_node = self.nodes.get(next_id)
            if next_node is None or not next_node.alive:
                break
            current = next_node

        return path

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node_id):
        return node_id in self.nodes


# --- VirtualNodeManager ---

class VirtualNodeManager:
    """
    Manage virtual nodes -- multiple Chord nodes per physical machine.

    Each physical node maps to `num_vnodes` virtual nodes in the ring,
    giving better load distribution (similar to virtual nodes in consistent hashing).
    """

    def __init__(self, ring, num_vnodes=3):
        self.ring = ring
        self.num_vnodes = num_vnodes
        self.physical_to_virtual = defaultdict(list)  # phys_name -> [vnode_ids]
        self.virtual_to_physical = {}                  # vnode_id -> phys_name

    def add_physical_node(self, name):
        """Add a physical node, creating multiple virtual nodes."""
        vnode_ids = []
        for i in range(self.num_vnodes):
            vnode_id = _hash_to_m(f"{name}#v{i}", self.ring.m)
            # Avoid collisions
            while vnode_id in self.ring.nodes:
                vnode_id = (vnode_id + 1) % self.ring.ring_size

            self.ring.add_node(vnode_id)
            self.physical_to_virtual[name].append(vnode_id)
            self.virtual_to_physical[vnode_id] = name
            vnode_ids.append(vnode_id)

        return vnode_ids

    def remove_physical_node(self, name):
        """Remove a physical node and all its virtual nodes."""
        if name not in self.physical_to_virtual:
            return False

        for vnode_id in self.physical_to_virtual[name]:
            self.ring.remove_node(vnode_id)
            if vnode_id in self.virtual_to_physical:
                del self.virtual_to_physical[vnode_id]

        del self.physical_to_virtual[name]
        return True

    def get_physical_node(self, key):
        """Get the physical node responsible for a key."""
        if not self.ring.nodes:
            return None
        key_id = _hash_to_m(key, self.ring.m)
        entry = self.ring._any_node()
        if entry is None:
            return None
        vnode_id = entry.find_successor(key_id)
        return self.virtual_to_physical.get(vnode_id, vnode_id)

    def get_physical_distribution(self):
        """Get key distribution by physical node."""
        dist = defaultdict(int)
        for nid, node in self.ring.nodes.items():
            phys = self.virtual_to_physical.get(nid, nid)
            dist[phys] += len(node.store)
        return dict(dist)

    @property
    def physical_nodes(self):
        return list(self.physical_to_virtual.keys())


# --- RangeQuery Support ---

class RangeQueryMixin:
    """Range query support for ChordRing."""

    @staticmethod
    def range_query(ring, start_key, end_key):
        """Query all key-value pairs with key IDs in [start, end].

        Since Chord only supports point queries natively, this scans
        the relevant nodes.
        """
        start_id = _hash_to_m(start_key, ring.m) if isinstance(start_key, str) else start_key
        end_id = _hash_to_m(end_key, ring.m) if isinstance(end_key, str) else end_key

        results = {}
        for node in ring.nodes.values():
            for key, value in node.store.items():
                key_id = _hash_to_m(key, ring.m)
                if _in_id_range(key_id, start_id, end_id, ring.ring_size):
                    results[key] = value
        return results

    @staticmethod
    def scan_all(ring):
        """Scan all keys in the DHT."""
        return ring.get_all_keys()

    @staticmethod
    def count_in_range(ring, start_id, end_id):
        """Count keys in the given ID range."""
        count = 0
        for node in ring.nodes.values():
            for key in node.store:
                key_id = _hash_to_m(key, ring.m)
                if _in_id_range(key_id, start_id, end_id, ring.ring_size):
                    count += 1
        return count


def _in_id_range(x, start, end, ring_size):
    """Check if x is in [start, end] on circular ring."""
    if start <= end:
        return start <= x <= end
    else:
        return x >= start or x <= end


# --- Lookup Statistics ---

class LookupStats:
    """Track lookup performance statistics."""

    def __init__(self):
        self.total_lookups = 0
        self.total_hops = 0
        self.max_hops = 0
        self.hop_histogram = defaultdict(int)

    def record(self, hops):
        self.total_lookups += 1
        self.total_hops += hops
        self.max_hops = max(self.max_hops, hops)
        self.hop_histogram[hops] += 1

    @property
    def avg_hops(self):
        if self.total_lookups == 0:
            return 0.0
        return self.total_hops / self.total_lookups

    def reset(self):
        self.total_lookups = 0
        self.total_hops = 0
        self.max_hops = 0
        self.hop_histogram.clear()


def measure_lookup_performance(ring, num_queries=100):
    """Measure the average number of hops for lookups."""
    stats = LookupStats()

    for i in range(num_queries):
        key = f"perf_test_{i}"
        path = ring.get_lookup_path(key)
        stats.record(len(path))

    return stats


# --- Migration Calculator ---

def calculate_key_migration(ring, new_node_id):
    """Calculate which keys would move if a new node were added.

    Returns (keys_that_move, total_keys, fraction).
    Does NOT actually add the node.
    """
    total_keys = 0
    would_move = 0

    # Find predecessor of new_node_id (the node just before it in ring order)
    pred_of_new = None
    for nid in reversed(ring._node_order):
        if nid < new_node_id:
            pred_of_new = nid
            break
    if pred_of_new is None and ring._node_order:
        pred_of_new = ring._node_order[-1]  # wrap around

    for node in ring.nodes.values():
        for key in node.store:
            total_keys += 1
            key_id = _hash_to_m(key, ring.m)
            # New node would be responsible for (pred_of_new, new_node_id]
            if pred_of_new is not None:
                if in_range(key_id, pred_of_new, new_node_id, ring.ring_size):
                    would_move += 1

    fraction = would_move / total_keys if total_keys > 0 else 0.0
    return would_move, total_keys, fraction


# --- Batch Operations ---

def batch_put(ring, items):
    """Put multiple key-value pairs. items is list of (key, value)."""
    results = []
    for key, value in items:
        results.append(ring.put(key, value))
    return results


def batch_get(ring, keys):
    """Get multiple values. Returns dict of key -> value."""
    results = {}
    for key in keys:
        results[key] = ring.get(key)
    return results
