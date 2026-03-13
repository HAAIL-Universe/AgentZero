"""
C236: Kademlia Distributed Hash Table

A comprehensive implementation of the Kademlia DHT protocol (Maymounkov & Mazieres, 2002).

Key differences from Chord (C235):
- XOR distance metric (symmetric, unidirectional)
- k-buckets for routing (one per bit of ID space)
- Iterative parallel lookup (alpha concurrent queries)
- No ring structure -- topology emerges from XOR distance
- Republishing and expiration for data persistence

Components:
1. KademliaNode -- individual node with k-buckets, key-value storage
2. KademliaNetwork -- simulated network of KademliaNodes (local simulation)
3. RoutingTable -- k-bucket based routing with LRU eviction
4. IterativeLookup -- iterative node/value lookup with alpha concurrency
5. Republishing -- periodic republish and expire of stored values
6. Caching -- along-path caching for hot keys
7. KeyPublishing -- original publisher republish protocol
"""

import hashlib
import struct
import time
import random
from collections import OrderedDict


# --- Configuration ---

DEFAULT_ID_BITS = 8       # bits in identifier space (2^8 = 256 for testing)
DEFAULT_K = 3             # k-bucket size (replication parameter)
DEFAULT_ALPHA = 2         # concurrency parameter for iterative lookups
DEFAULT_REPUBLISH = 3600  # republish interval (seconds)
DEFAULT_EXPIRE = 86400    # key expiration time (seconds)
DEFAULT_REFRESH = 3600    # bucket refresh interval (seconds)


def _hash_to_bits(key, id_bits):
    """Hash a key to id_bits-bit identifier space [0, 2^id_bits)."""
    if isinstance(key, str):
        key = key.encode('utf-8')
    elif isinstance(key, int):
        key = str(key).encode('utf-8')
    digest = hashlib.sha256(key).digest()
    value = struct.unpack('>I', digest[:4])[0]
    return value % (2 ** id_bits)


def xor_distance(a, b):
    """XOR distance between two node/key IDs."""
    return a ^ b


def bit_length(n):
    """Number of bits needed to represent n (0 -> 0)."""
    if n == 0:
        return 0
    count = 0
    while n > 0:
        count += 1
        n >>= 1
    return count


def bucket_index(own_id, other_id, id_bits):
    """Determine which k-bucket an ID belongs to.

    Bucket i covers distances [2^i, 2^(i+1)).
    Returns index in range [0, id_bits-1], or -1 if own_id == other_id.
    """
    dist = xor_distance(own_id, other_id)
    if dist == 0:
        return -1
    return bit_length(dist) - 1


# --- Contact ---

class Contact:
    """Represents a known node (id, address, last_seen)."""

    __slots__ = ('node_id', 'address', 'last_seen')

    def __init__(self, node_id, address=None, last_seen=None):
        self.node_id = node_id
        self.address = address or f"addr_{node_id}"
        self.last_seen = last_seen or 0.0

    def __eq__(self, other):
        if not isinstance(other, Contact):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self):
        return hash(self.node_id)

    def __repr__(self):
        return f"Contact({self.node_id})"


# --- KBucket ---

class KBucket:
    """A k-bucket holding up to k contacts, ordered by last-seen (LRU at front)."""

    def __init__(self, k, range_lower, range_upper):
        self.k = k
        self.range_lower = range_lower
        self.range_upper = range_upper
        self.contacts = OrderedDict()  # node_id -> Contact, LRU order
        self.last_updated = 0.0

    def covers(self, node_id):
        """Check if a node_id falls in this bucket's range."""
        return self.range_lower <= node_id < self.range_upper

    def is_full(self):
        return len(self.contacts) >= self.k

    def add_contact(self, contact, now=None):
        """Add or update a contact. Returns eviction candidate if full, else None."""
        now = now or time.time()
        if contact.node_id in self.contacts:
            # Move to end (most recently seen)
            del self.contacts[contact.node_id]
            contact.last_seen = now
            self.contacts[contact.node_id] = contact
            self.last_updated = now
            return None

        if not self.is_full():
            contact.last_seen = now
            self.contacts[contact.node_id] = contact
            self.last_updated = now
            return None

        # Bucket full -- return LRU contact for ping check
        lru_id = next(iter(self.contacts))
        return self.contacts[lru_id]

    def remove_contact(self, node_id):
        """Remove a contact by ID."""
        if node_id in self.contacts:
            del self.contacts[node_id]
            return True
        return False

    def get_contacts(self):
        """Return list of contacts (most recently seen last)."""
        return list(self.contacts.values())

    def head(self):
        """Return LRU (least recently seen) contact."""
        if self.contacts:
            return self.contacts[next(iter(self.contacts))]
        return None

    def depth(self):
        """Shared prefix depth of contacts in this bucket."""
        if len(self.contacts) == 0:
            return 0
        # Simple: return count of contacts
        return len(self.contacts)

    def __len__(self):
        return len(self.contacts)

    def __repr__(self):
        return f"KBucket([{self.range_lower}, {self.range_upper}), {len(self)} contacts)"


# --- RoutingTable ---

class RoutingTable:
    """Routing table with k-buckets indexed by XOR distance prefix length."""

    def __init__(self, own_id, id_bits=DEFAULT_ID_BITS, k=DEFAULT_K):
        self.own_id = own_id
        self.id_bits = id_bits
        self.k = k
        self.id_space = 2 ** id_bits
        # One bucket per bit position
        self.buckets = []
        for i in range(id_bits):
            lower = 2 ** i
            upper = 2 ** (i + 1)
            self.buckets.append(KBucket(k, lower, upper))

    def _bucket_for(self, node_id):
        """Get the bucket index for a given node_id."""
        idx = bucket_index(self.own_id, node_id, self.id_bits)
        if idx < 0:
            return None  # same as own_id
        return idx

    def add_contact(self, contact, now=None):
        """Add a contact to the appropriate bucket.

        Returns (added: bool, eviction_candidate: Contact or None).
        """
        idx = self._bucket_for(contact.node_id)
        if idx is None:
            return (False, None)
        bucket = self.buckets[idx]
        evict = bucket.add_contact(contact, now)
        if evict is None:
            return (True, None)
        return (False, evict)

    def remove_contact(self, node_id):
        """Remove a contact from the routing table."""
        idx = self._bucket_for(node_id)
        if idx is None:
            return False
        return self.buckets[idx].remove_contact(node_id)

    def find_closest(self, target_id, count=None, exclude=None):
        """Find the count closest contacts to target_id by XOR distance."""
        if count is None:
            count = self.k
        exclude = exclude or set()

        all_contacts = []
        for bucket in self.buckets:
            for c in bucket.get_contacts():
                if c.node_id not in exclude:
                    all_contacts.append(c)

        all_contacts.sort(key=lambda c: xor_distance(c.node_id, target_id))
        return all_contacts[:count]

    def get_bucket(self, index):
        """Get bucket by index."""
        if 0 <= index < len(self.buckets):
            return self.buckets[index]
        return None

    def get_all_contacts(self):
        """Return all contacts in the routing table."""
        result = []
        for bucket in self.buckets:
            result.extend(bucket.get_contacts())
        return result

    def total_contacts(self):
        """Total number of contacts across all buckets."""
        return sum(len(b) for b in self.buckets)

    def stale_buckets(self, now=None, threshold=DEFAULT_REFRESH):
        """Return indices of buckets that haven't been updated recently."""
        now = now or time.time()
        stale = []
        for i, b in enumerate(self.buckets):
            if len(b) > 0 and now - b.last_updated > threshold:
                stale.append(i)
        return stale


# --- StoredValue ---

class StoredValue:
    """A value stored in the DHT with metadata."""

    __slots__ = ('value', 'publisher_id', 'store_time', 'last_republish', 'expire_time')

    def __init__(self, value, publisher_id, store_time=None, expire_time=DEFAULT_EXPIRE):
        self.value = value
        self.publisher_id = publisher_id
        self.store_time = time.time() if store_time is None else store_time
        self.last_republish = self.store_time
        self.expire_time = expire_time

    def is_expired(self, now=None):
        now = now or time.time()
        return (now - self.store_time) >= self.expire_time


# --- KademliaNode ---

class KademliaNode:
    """A single Kademlia node with routing table and key-value storage."""

    def __init__(self, node_id=None, id_bits=DEFAULT_ID_BITS, k=DEFAULT_K,
                 alpha=DEFAULT_ALPHA):
        self.id_bits = id_bits
        self.k = k
        self.alpha = alpha
        self.id_space = 2 ** id_bits
        self.node_id = node_id if node_id is not None else random.randint(0, self.id_space - 1)
        self.routing_table = RoutingTable(self.node_id, id_bits, k)
        self.storage = {}       # key_id -> StoredValue
        self.cache = {}         # key_id -> StoredValue (along-path cache)
        self.network = None     # set by KademliaNetwork
        self.is_alive = True
        self.rpc_count = 0      # track RPCs for testing

    def _contact(self):
        """Create a Contact for this node."""
        return Contact(self.node_id)

    # --- RPCs (simulated) ---

    def rpc_ping(self, sender_contact):
        """PING RPC -- respond if alive, update routing table."""
        if not self.is_alive:
            return False
        self._update_routing(sender_contact)
        self.rpc_count += 1
        return True

    def rpc_store(self, sender_contact, key_id, value, publisher_id=None):
        """STORE RPC -- store a key-value pair."""
        if not self.is_alive:
            return False
        self._update_routing(sender_contact)
        self.rpc_count += 1
        pub = publisher_id or sender_contact.node_id
        self.storage[key_id] = StoredValue(value, pub)
        return True

    def rpc_find_node(self, sender_contact, target_id):
        """FIND_NODE RPC -- return k closest contacts to target_id."""
        if not self.is_alive:
            return None
        self._update_routing(sender_contact)
        self.rpc_count += 1
        closest = self.routing_table.find_closest(
            target_id, self.k,
            exclude={sender_contact.node_id}
        )
        return closest

    def rpc_find_value(self, sender_contact, key_id):
        """FIND_VALUE RPC -- return value if stored, else k closest contacts."""
        if not self.is_alive:
            return None
        self._update_routing(sender_contact)
        self.rpc_count += 1
        # Check storage first
        if key_id in self.storage:
            sv = self.storage[key_id]
            if not sv.is_expired():
                return {'value': sv.value, 'publisher': sv.publisher_id}
        # Check cache
        if key_id in self.cache:
            sv = self.cache[key_id]
            if not sv.is_expired():
                return {'value': sv.value, 'publisher': sv.publisher_id}
        # Return closest contacts
        closest = self.routing_table.find_closest(
            key_id, self.k,
            exclude={sender_contact.node_id}
        )
        return {'contacts': closest}

    def _update_routing(self, contact):
        """Update routing table with a newly seen contact."""
        if contact.node_id == self.node_id:
            return
        added, evict = self.routing_table.add_contact(Contact(contact.node_id), time.time())
        if not added and evict is not None:
            # In real Kademlia, we'd ping the LRU contact
            # If it responds, keep it; if not, replace with new
            if self.network:
                evict_node = self.network.get_node(evict.node_id)
                if evict_node and evict_node.is_alive:
                    # LRU is alive, keep it (move to tail)
                    bucket_idx = self.routing_table._bucket_for(evict.node_id)
                    if bucket_idx is not None:
                        bucket = self.routing_table.buckets[bucket_idx]
                        bucket.add_contact(evict, time.time())
                else:
                    # LRU is dead, replace
                    bucket_idx = self.routing_table._bucket_for(evict.node_id)
                    if bucket_idx is not None:
                        bucket = self.routing_table.buckets[bucket_idx]
                        bucket.remove_contact(evict.node_id)
                        bucket.add_contact(Contact(contact.node_id), time.time())

    # --- High-level operations ---

    def store(self, key, value):
        """Store a key-value pair in the DHT."""
        key_id = _hash_to_bits(key, self.id_bits)
        closest = self._iterative_find_node(key_id)
        stored_count = 0
        for contact in closest[:self.k]:
            node = self.network.get_node(contact.node_id) if self.network else None
            if node and node.is_alive:
                result = node.rpc_store(self._contact(), key_id, value, self.node_id)
                if result:
                    stored_count += 1
        # Also store locally if we're among the closest
        self.storage[key_id] = StoredValue(value, self.node_id)
        stored_count += 1
        return key_id, stored_count

    def lookup(self, key):
        """Look up a value in the DHT by key."""
        key_id = _hash_to_bits(key, self.id_bits)
        return self._iterative_find_value(key_id)

    def delete(self, key):
        """Delete a key from local storage and closest nodes."""
        key_id = _hash_to_bits(key, self.id_bits)
        deleted = 0
        if key_id in self.storage:
            del self.storage[key_id]
            deleted += 1
        # Also delete from closest nodes
        closest = self._iterative_find_node(key_id)
        for contact in closest[:self.k]:
            node = self.network.get_node(contact.node_id) if self.network else None
            if node and node.is_alive:
                if key_id in node.storage:
                    del node.storage[key_id]
                    deleted += 1
        return deleted

    def _iterative_find_node(self, target_id):
        """Iterative FIND_NODE lookup."""
        # Start with alpha closest contacts from own routing table
        shortlist = self.routing_table.find_closest(target_id, self.k)
        if not shortlist:
            return [self._contact()]

        queried = {self.node_id}
        closest_seen = {}
        for c in shortlist:
            closest_seen[c.node_id] = c

        while True:
            # Pick alpha unqueried contacts closest to target
            unqueried = [
                c for c in sorted(closest_seen.values(),
                                  key=lambda c: xor_distance(c.node_id, target_id))
                if c.node_id not in queried
            ][:self.alpha]

            if not unqueried:
                break

            made_progress = False
            for contact in unqueried:
                queried.add(contact.node_id)
                node = self.network.get_node(contact.node_id) if self.network else None
                if node and node.is_alive:
                    result = node.rpc_find_node(self._contact(), target_id)
                    if result:
                        for new_c in result:
                            if new_c.node_id not in closest_seen and new_c.node_id != self.node_id:
                                closest_seen[new_c.node_id] = new_c
                                made_progress = True

            if not made_progress:
                # Try remaining unqueried in the k closest
                k_closest = sorted(closest_seen.values(),
                                   key=lambda c: xor_distance(c.node_id, target_id))[:self.k]
                remaining = [c for c in k_closest if c.node_id not in queried]
                if not remaining:
                    break
                for contact in remaining:
                    queried.add(contact.node_id)
                    node = self.network.get_node(contact.node_id) if self.network else None
                    if node and node.is_alive:
                        result = node.rpc_find_node(self._contact(), target_id)
                        if result:
                            for new_c in result:
                                if new_c.node_id not in closest_seen and new_c.node_id != self.node_id:
                                    closest_seen[new_c.node_id] = new_c
                break

        result = sorted(closest_seen.values(),
                        key=lambda c: xor_distance(c.node_id, target_id))
        return result[:self.k]

    def _iterative_find_value(self, key_id):
        """Iterative FIND_VALUE lookup -- returns value or None."""
        # Check local storage first
        if key_id in self.storage:
            sv = self.storage[key_id]
            if not sv.is_expired():
                return sv.value
        if key_id in self.cache:
            sv = self.cache[key_id]
            if not sv.is_expired():
                return sv.value

        shortlist = self.routing_table.find_closest(key_id, self.k)
        if not shortlist:
            return None

        queried = {self.node_id}
        closest_seen = {}
        for c in shortlist:
            closest_seen[c.node_id] = c

        last_node_without_value = None

        while True:
            unqueried = [
                c for c in sorted(closest_seen.values(),
                                  key=lambda c: xor_distance(c.node_id, key_id))
                if c.node_id not in queried
            ][:self.alpha]

            if not unqueried:
                break

            found_value = False
            for contact in unqueried:
                queried.add(contact.node_id)
                node = self.network.get_node(contact.node_id) if self.network else None
                if node and node.is_alive:
                    result = node.rpc_find_value(self._contact(), key_id)
                    if result is None:
                        continue
                    if 'value' in result:
                        # Cache on closest node that didn't have the value
                        if last_node_without_value and self.network:
                            cache_node = self.network.get_node(last_node_without_value.node_id)
                            if cache_node and cache_node.is_alive:
                                cache_node.cache[key_id] = StoredValue(
                                    result['value'], result['publisher']
                                )
                        return result['value']
                    if 'contacts' in result:
                        last_node_without_value = contact
                        for new_c in result['contacts']:
                            if new_c.node_id not in closest_seen and new_c.node_id != self.node_id:
                                closest_seen[new_c.node_id] = new_c
                                found_value = True

            if not found_value:
                # Try all remaining unqueried in k closest
                k_closest = sorted(closest_seen.values(),
                                   key=lambda c: xor_distance(c.node_id, key_id))[:self.k]
                remaining = [c for c in k_closest if c.node_id not in queried]
                if not remaining:
                    break
                for contact in remaining:
                    queried.add(contact.node_id)
                    node = self.network.get_node(contact.node_id) if self.network else None
                    if node and node.is_alive:
                        result = node.rpc_find_value(self._contact(), key_id)
                        if result and 'value' in result:
                            return result['value']
                break

        return None

    def bootstrap(self, bootstrap_contact):
        """Bootstrap into the network by contacting a known node."""
        self._update_routing(bootstrap_contact)
        # Look up own ID to populate routing table
        self._iterative_find_node(self.node_id)

    def republish(self, now=None):
        """Republish all stored values to k closest nodes."""
        now = now or time.time()
        republished = 0
        for key_id, sv in list(self.storage.items()):
            if sv.is_expired(now):
                del self.storage[key_id]
                continue
            if sv.publisher_id == self.node_id:
                # Original publisher: republish
                closest = self._iterative_find_node(key_id)
                for contact in closest[:self.k]:
                    node = self.network.get_node(contact.node_id) if self.network else None
                    if node and node.is_alive and node.node_id != self.node_id:
                        node.rpc_store(self._contact(), key_id, sv.value, self.node_id)
                sv.last_republish = now
                republished += 1
        return republished

    def expire_keys(self, now=None):
        """Remove expired keys from storage and cache."""
        now = now or time.time()
        expired = 0
        for key_id in list(self.storage.keys()):
            if self.storage[key_id].is_expired(now):
                del self.storage[key_id]
                expired += 1
        for key_id in list(self.cache.keys()):
            if self.cache[key_id].is_expired(now):
                del self.cache[key_id]
                expired += 1
        return expired

    def refresh_buckets(self, now=None, threshold=DEFAULT_REFRESH):
        """Refresh stale buckets by performing a lookup of a random ID in each."""
        now = now or time.time()
        refreshed = 0
        stale = self.routing_table.stale_buckets(now, threshold)
        for idx in stale:
            bucket = self.routing_table.buckets[idx]
            # Generate random ID in bucket's distance range
            random_dist = random.randint(bucket.range_lower, bucket.range_upper - 1)
            random_id = self.node_id ^ random_dist
            random_id %= self.id_space
            self._iterative_find_node(random_id)
            bucket.last_updated = now
            refreshed += 1
        return refreshed


# --- KademliaNetwork ---

class KademliaNetwork:
    """Simulated Kademlia network for testing (no real networking)."""

    def __init__(self, id_bits=DEFAULT_ID_BITS, k=DEFAULT_K, alpha=DEFAULT_ALPHA):
        self.id_bits = id_bits
        self.k = k
        self.alpha = alpha
        self.id_space = 2 ** id_bits
        self.nodes = {}  # node_id -> KademliaNode
        self.stats = {
            'stores': 0,
            'lookups': 0,
            'rpcs': 0,
        }

    def create_node(self, node_id=None):
        """Create and add a new node to the network."""
        node = KademliaNode(node_id=node_id, id_bits=self.id_bits,
                            k=self.k, alpha=self.alpha)
        node.network = self
        self.nodes[node.node_id] = node
        return node

    def get_node(self, node_id):
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def bootstrap_network(self, count=10, seed=None):
        """Create a network of `count` nodes with routing tables populated."""
        if seed is not None:
            random.seed(seed)

        # Create nodes with unique IDs
        ids = random.sample(range(self.id_space), min(count, self.id_space))
        nodes = []
        for nid in ids:
            nodes.append(self.create_node(nid))

        if len(nodes) < 2:
            return nodes

        # Bootstrap: each node contacts the first node
        bootstrap_node = nodes[0]
        for node in nodes[1:]:
            node.bootstrap(bootstrap_node._contact())

        # Second pass: bootstrap from random nodes for better connectivity
        for node in nodes:
            other = random.choice(nodes)
            if other.node_id != node.node_id:
                node.bootstrap(other._contact())

        return nodes

    def store(self, source_node, key, value):
        """Store a key-value pair from a specific node."""
        self.stats['stores'] += 1
        return source_node.store(key, value)

    def lookup(self, source_node, key):
        """Lookup a value from a specific node."""
        self.stats['lookups'] += 1
        return source_node.lookup(key)

    def remove_node(self, node_id):
        """Remove a node from the network (simulate failure)."""
        if node_id in self.nodes:
            self.nodes[node_id].is_alive = False
            return True
        return False

    def restore_node(self, node_id):
        """Restore a failed node."""
        if node_id in self.nodes:
            self.nodes[node_id].is_alive = True
            return True
        return False

    def get_alive_nodes(self):
        """Return list of alive nodes."""
        return [n for n in self.nodes.values() if n.is_alive]

    def total_stored_keys(self):
        """Total number of key-value pairs across all nodes."""
        total = 0
        seen = set()
        for node in self.nodes.values():
            if node.is_alive:
                for key_id in node.storage:
                    seen.add(key_id)
        return len(seen)

    def key_redundancy(self, key_id):
        """Count how many alive nodes store a given key_id."""
        count = 0
        for node in self.nodes.values():
            if node.is_alive and key_id in node.storage:
                count += 1
        return count

    def network_health(self):
        """Return health metrics for the network."""
        alive = self.get_alive_nodes()
        if not alive:
            return {'alive': 0, 'total': len(self.nodes), 'avg_contacts': 0,
                    'avg_storage': 0, 'health': 0.0}

        total_contacts = sum(n.routing_table.total_contacts() for n in alive)
        total_storage = sum(len(n.storage) for n in alive)
        return {
            'alive': len(alive),
            'total': len(self.nodes),
            'avg_contacts': total_contacts / len(alive),
            'avg_storage': total_storage / len(alive),
            'health': len(alive) / len(self.nodes) if self.nodes else 1.0,
        }

    def republish_all(self, now=None):
        """Run republish on all alive nodes."""
        total = 0
        for node in self.get_alive_nodes():
            total += node.republish(now)
        return total

    def expire_all(self, now=None):
        """Run expire on all alive nodes."""
        total = 0
        for node in self.get_alive_nodes():
            total += node.expire_keys(now)
        return total


# --- Visualization / Debugging ---

def visualize_routing_table(node):
    """Return a human-readable view of a node's routing table."""
    lines = [f"Routing table for node {node.node_id}:"]
    for i, bucket in enumerate(node.routing_table.buckets):
        if len(bucket) > 0:
            contacts = [str(c.node_id) for c in bucket.get_contacts()]
            lines.append(f"  Bucket {i} [{bucket.range_lower}-{bucket.range_upper}): "
                         f"{', '.join(contacts)}")
    return '\n'.join(lines)


def visualize_network(network):
    """Return a summary view of the network."""
    health = network.network_health()
    lines = [
        f"Kademlia Network ({health['alive']}/{health['total']} alive)",
        f"  Avg contacts/node: {health['avg_contacts']:.1f}",
        f"  Avg storage/node: {health['avg_storage']:.1f}",
    ]
    for node in sorted(network.get_alive_nodes(), key=lambda n: n.node_id):
        storage_count = len(node.storage)
        contact_count = node.routing_table.total_contacts()
        lines.append(f"  Node {node.node_id}: {contact_count} contacts, "
                     f"{storage_count} keys")
    return '\n'.join(lines)
