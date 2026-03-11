"""
C106: Disjoint Set / Union-Find

A comprehensive Union-Find data structure with multiple variants:
1. UnionFind -- core with path compression + union by rank
2. WeightedUnionFind -- tracks relative weights/distances between elements
3. PersistentUnionFind -- version history with rollback
4. DynamicUnionFind -- supports element deletion
5. UnionFindMap -- generic keys (not just integers)

Applications: Kruskal's MST, connected components, cycle detection,
equivalence classes, dynamic connectivity.
"""


class UnionFind:
    """Disjoint set with path compression and union by rank.

    O(alpha(n)) amortized per operation where alpha is the
    inverse Ackermann function (effectively constant).
    """

    def __init__(self, n=0):
        self._parent = list(range(n))
        self._rank = [0] * n
        self._size = [1] * n
        self._count = n  # number of disjoint sets

    @property
    def count(self):
        """Number of disjoint sets."""
        return self._count

    @property
    def n(self):
        """Number of elements."""
        return len(self._parent)

    def make_set(self):
        """Add a new element and return its id."""
        idx = len(self._parent)
        self._parent.append(idx)
        self._rank.append(0)
        self._size.append(1)
        self._count += 1
        return idx

    def find(self, x):
        """Find root of x with path compression."""
        if x < 0 or x >= len(self._parent):
            raise ValueError(f"Element {x} not in set")
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

    def union(self, x, y):
        """Union sets containing x and y. Returns True if they were separate."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._count -= 1
        return True

    def connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def set_size(self, x):
        """Size of the set containing x."""
        return self._size[self.find(x)]

    def sets(self):
        """Return all sets as a dict of root -> list of members."""
        groups = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

    def roots(self):
        """Return set of all root elements."""
        return {self.find(i) for i in range(len(self._parent))}

    def __repr__(self):
        return f"UnionFind(n={self.n}, sets={self.count})"


class WeightedUnionFind:
    """Union-Find tracking relative weights between elements.

    Each element has a weight relative to its root.
    weight(x) - weight(y) gives the relative weight between x and y
    when they're in the same set.

    Useful for: potentials, relative distances, offset tracking.
    """

    def __init__(self, n=0):
        self._parent = list(range(n))
        self._rank = [0] * n
        self._weight = [0] * n  # weight relative to parent
        self._count = n

    @property
    def count(self):
        return self._count

    def make_set(self):
        idx = len(self._parent)
        self._parent.append(idx)
        self._rank.append(0)
        self._weight.append(0)
        self._count += 1
        return idx

    def find(self, x):
        """Find root, compressing path and accumulating weights."""
        if x < 0 or x >= len(self._parent):
            raise ValueError(f"Element {x} not in set")
        if self._parent[x] == x:
            return x
        # Path compression with weight accumulation
        path = []
        while self._parent[x] != x:
            path.append(x)
            x = self._parent[x]
        root = x
        for node in path:
            # Accumulate weight to root
            w = 0
            curr = node
            while curr != root:
                w += self._weight[curr]
                curr = self._parent[curr]
            self._weight[node] = w
            self._parent[node] = root
        return root

    def weight(self, x):
        """Get weight of x relative to its root."""
        self.find(x)  # ensure path is compressed
        return self._weight[x]

    def diff(self, x, y):
        """Get weight(x) - weight(y). Raises if not connected."""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            raise ValueError(f"Elements {x} and {y} not connected")
        return self._weight[x] - self._weight[y]

    def union(self, x, y, w=0):
        """Union x and y with weight(x) - weight(y) = w.
        Returns True if they were separate."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # weight(x) relative to rx = self._weight[x]
        # weight(y) relative to ry = self._weight[y]
        # We want weight(x) - weight(y) = w after union
        # If ry becomes child of rx:
        #   weight(y) = weight_ry_to_rx + self._weight[y] (through old path)
        #   weight(x) - (weight_ry_to_rx + self._weight[y]) = w
        #   weight_ry_to_rx = self._weight[x] - self._weight[y] - w
        if self._rank[rx] < self._rank[ry]:
            # ry is new root, rx becomes child
            # weight(rx) relative to ry: weight(x) - w - weight(y)...
            # weight_rx_to_ry: weight(y) + w - weight(x)
            # because weight(x) = weight_rx_to_ry + self._weight[x]
            # and we want weight(x) - weight(y) = w
            # weight(x) through rx->ry = weight_rx_to_ry + self._weight[x]
            # weight(y) through ry = self._weight[y]
            # (weight_rx_to_ry + self._weight[x]) - self._weight[y] = w
            # weight_rx_to_ry = w + self._weight[y] - self._weight[x]
            self._weight[rx] = w + self._weight[y] - self._weight[x]
            self._parent[rx] = ry
            self._count -= 1
            return True

        # rx is new root, ry becomes child
        self._weight[ry] = self._weight[x] - self._weight[y] - w
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class PersistentUnionFind:
    """Union-Find with version history and rollback.

    Supports save/restore to named checkpoints and full undo.
    Uses union by rank WITHOUT path compression to enable rollback.
    """

    def __init__(self, n=0):
        self._parent = list(range(n))
        self._rank = [0] * n
        self._size = [1] * n
        self._count = n
        self._history = []  # stack of (type, data) operations
        self._checkpoints = {}  # name -> history length

    @property
    def count(self):
        return self._count

    def find(self, x):
        """Find root WITHOUT path compression (to support rollback)."""
        if x < 0 or x >= len(self._parent):
            raise ValueError(f"Element {x} not in set")
        while self._parent[x] != x:
            x = self._parent[x]
        return x

    def union(self, x, y):
        """Union with history tracking."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        # Record for undo: (child, old_parent, old_rank_of_root, old_size_of_root, old_count)
        self._history.append(('union', ry, rx, self._rank[rx], self._size[rx], self._count))
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._count -= 1
        return True

    def undo(self):
        """Undo the last union operation. Returns True if undone."""
        if not self._history:
            return False
        op, child, parent, old_rank, old_size, old_count = self._history.pop()
        self._parent[child] = child
        self._rank[parent] = old_rank
        self._size[parent] = old_size
        self._count = old_count
        return True

    def save(self, name):
        """Save a checkpoint."""
        self._checkpoints[name] = len(self._history)

    def restore(self, name):
        """Restore to a named checkpoint."""
        if name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found")
        target = self._checkpoints[name]
        while len(self._history) > target:
            self.undo()
        # Remove checkpoints created after this one
        self._checkpoints = {k: v for k, v in self._checkpoints.items() if v <= target}

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def history_size(self):
        return len(self._history)


class DynamicUnionFind:
    """Union-Find supporting element deletion via indirection.

    Elements are mapped to internal IDs. Deletion creates a new
    internal ID for the element, effectively disconnecting it.
    """

    def __init__(self, n=0):
        self._uf = UnionFind(n)
        self._elem_to_id = {i: i for i in range(n)}
        self._next_id = n
        self._active = set(range(n))

    @property
    def count(self):
        """Number of disjoint sets among active elements."""
        roots = set()
        for elem in self._active:
            roots.add(self._uf.find(self._elem_to_id[elem]))
        return len(roots)

    def _ensure_capacity(self, idx):
        """Grow internal UF to accommodate idx."""
        while idx >= self._uf.n:
            self._uf.make_set()

    def add(self, elem):
        """Add or re-add an element."""
        if elem in self._active:
            return
        internal_id = self._next_id
        self._next_id += 1
        self._ensure_capacity(internal_id)
        self._elem_to_id[elem] = internal_id
        self._active.add(elem)

    def delete(self, elem):
        """Delete an element from all sets."""
        if elem not in self._active:
            raise ValueError(f"Element {elem} not active")
        self._active.discard(elem)
        # Don't remap -- just remove from active. Old internal ID becomes orphaned.

    def find(self, elem):
        """Find root element."""
        if elem not in self._active:
            raise ValueError(f"Element {elem} not active")
        root_id = self._uf.find(self._elem_to_id[elem])
        # Find which active element maps to this root
        for e in self._active:
            if self._uf.find(self._elem_to_id[e]) == root_id:
                return e
        return elem

    def union(self, x, y):
        """Union sets containing x and y."""
        if x not in self._active or y not in self._active:
            raise ValueError("Both elements must be active")
        return self._uf.union(self._elem_to_id[x], self._elem_to_id[y])

    def connected(self, x, y):
        if x not in self._active or y not in self._active:
            return False
        return self._uf.connected(self._elem_to_id[x], self._elem_to_id[y])

    @property
    def active_elements(self):
        return frozenset(self._active)


class UnionFindMap:
    """Union-Find with arbitrary hashable keys.

    Unlike the integer-based UnionFind, this supports any hashable
    type as elements (strings, tuples, etc.). Elements are created
    on first use.
    """

    def __init__(self, elements=None):
        self._parent = {}
        self._rank = {}
        self._size = {}
        self._count = 0
        if elements:
            for e in elements:
                self.make_set(e)

    @property
    def count(self):
        return self._count

    @property
    def n(self):
        return len(self._parent)

    def make_set(self, x):
        """Create a new singleton set for x. No-op if exists."""
        if x in self._parent:
            return False
        self._parent[x] = x
        self._rank[x] = 0
        self._size[x] = 1
        self._count += 1
        return True

    def find(self, x):
        """Find with path compression."""
        if x not in self._parent:
            raise ValueError(f"Element {x} not in set")
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

    def union(self, x, y):
        """Union sets. Auto-creates elements if not present."""
        if x not in self._parent:
            self.make_set(x)
        if y not in self._parent:
            self.make_set(y)
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._count -= 1
        return True

    def connected(self, x, y):
        if x not in self._parent or y not in self._parent:
            return False
        return self.find(x) == self.find(y)

    def set_size(self, x):
        return self._size[self.find(x)]

    def sets(self):
        groups = {}
        for elem in self._parent:
            root = self.find(elem)
            if root not in groups:
                groups[root] = []
            groups[root].append(elem)
        return groups

    def __contains__(self, x):
        return x in self._parent


# --- Applications ---

def kruskal_mst(n, edges):
    """Kruskal's MST using Union-Find.

    Args:
        n: number of vertices
        edges: list of (weight, u, v)

    Returns:
        (total_weight, mst_edges) where mst_edges is list of (weight, u, v)
    """
    uf = UnionFind(n)
    sorted_edges = sorted(edges)
    mst = []
    total = 0
    for w, u, v in sorted_edges:
        if uf.union(u, v):
            mst.append((w, u, v))
            total += w
            if len(mst) == n - 1:
                break
    return total, mst


def connected_components(n, edges):
    """Find connected components using Union-Find.

    Args:
        n: number of vertices
        edges: list of (u, v) pairs

    Returns:
        dict of component_id -> list of vertices
    """
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.sets()


def has_cycle(n, edges):
    """Detect if an undirected graph has a cycle.

    Args:
        n: number of vertices
        edges: list of (u, v) pairs

    Returns:
        True if cycle exists
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False


def equivalence_classes(n, pairs):
    """Compute equivalence classes from pairs of equivalent elements.

    Args:
        n: number of elements
        pairs: list of (a, b) pairs that are equivalent

    Returns:
        list of sets, each set is an equivalence class
    """
    uf = UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)
    return [set(members) for members in uf.sets().values()]


def earliest_connection(n, connections):
    """Find earliest time all nodes are connected.

    Args:
        n: number of nodes
        connections: list of (time, u, v) sorted by time

    Returns:
        time when all connected, or -1 if never
    """
    uf = UnionFind(n)
    for t, u, v in sorted(connections):
        uf.union(u, v)
        if uf.count == 1:
            return t
    return -1


def redundant_connections(n, edges):
    """Find all redundant edges (those that don't connect new components).

    Args:
        n: number of vertices
        edges: list of (u, v) pairs

    Returns:
        list of redundant (u, v) edges
    """
    uf = UnionFind(n)
    redundant = []
    for u, v in edges:
        if not uf.union(u, v):
            redundant.append((u, v))
    return redundant


def accounts_merge(accounts):
    """Merge accounts that share emails.

    Args:
        accounts: list of [name, email1, email2, ...]

    Returns:
        list of [name, sorted_emails...] merged accounts
    """
    uf = UnionFindMap()
    email_to_name = {}

    for account in accounts:
        name = account[0]
        first_email = account[1]
        for email in account[1:]:
            email_to_name[email] = name
            uf.union(first_email, email)

    groups = uf.sets()
    result = []
    for root, emails in groups.items():
        name = email_to_name[root]
        result.append([name] + sorted(emails))
    return result
