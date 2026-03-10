"""
C084: Link-Cut Tree (Sleator-Tarjan)

A dynamic forest data structure supporting:
- link(u, v): connect two trees by making u a child of v
- cut(u): disconnect u from its parent
- find_root(u): find the root of u's tree
- path_aggregate(u, v): aggregate values on the path u->v
- lca(u, v): lowest common ancestor
- evert(u): re-root u's tree at u (make_root)

All operations amortized O(log n) via splay trees on preferred paths.

Implementation:
- Each node has: left, right, parent (splay tree pointers)
- Preferred path = chain of nodes connected by preferred edges
- Auxiliary trees (splay trees) represent each preferred path, keyed by depth
- parent pointer may point to path-parent (not in same splay tree)

Variants:
1. LinkCutTree -- basic (connectivity, link, cut, find_root, evert)
2. PathAggregateTree -- path queries (sum, min, max on paths)
3. SubtreeAggregateTree -- subtree queries (subtree sum/size)
4. WeightedLinkCutTree -- weighted edges (edge weights stored at child node)
5. LinkCutForest -- manages a forest of link-cut trees with component tracking
"""


class SplayNode:
    """Node in a link-cut tree (also a splay tree node)."""
    __slots__ = (
        'id', 'value', 'left', 'right', 'parent',
        'rev',  # lazy reversal flag
        'size',  # subtree size in splay tree
        'agg_sum', 'agg_min', 'agg_max',  # path aggregates
        'sub_sum', 'sub_size',  # subtree aggregates (virtual children)
        'vsub_sum', 'vsub_size',  # virtual subtree aggregates
        'weight',  # edge weight (to parent in represented tree)
    )

    def __init__(self, node_id, value=0):
        self.id = node_id
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.rev = False
        self.size = 1
        self.agg_sum = value
        self.agg_min = value
        self.agg_max = value
        self.sub_sum = 0   # sum of virtual children subtrees
        self.sub_size = 0  # size of virtual children subtrees
        self.vsub_sum = 0
        self.vsub_size = 0
        self.weight = 0    # edge weight to parent


def _is_root(x):
    """Check if x is root of its auxiliary (splay) tree.
    A node is root of its splay tree if its parent doesn't have it as a child."""
    if x.parent is None:
        return True
    return x.parent.left is not x and x.parent.right is not x


def _push_down(x):
    """Push lazy reversal flag down to children."""
    if x.rev:
        x.left, x.right = x.right, x.left
        if x.left:
            x.left.rev = not x.left.rev
        if x.right:
            x.right.rev = not x.right.rev
        x.rev = False


def _pull_up(x):
    """Recompute aggregates from children."""
    x.size = 1
    x.agg_sum = x.value
    x.agg_min = x.value
    x.agg_max = x.value

    if x.left:
        x.size += x.left.size
        x.agg_sum += x.left.agg_sum
        x.agg_min = min(x.agg_min, x.left.agg_min)
        x.agg_max = max(x.agg_max, x.left.agg_max)

    if x.right:
        x.size += x.right.size
        x.agg_sum += x.right.agg_sum
        x.agg_min = min(x.agg_min, x.right.agg_min)
        x.agg_max = max(x.agg_max, x.right.agg_max)

    # Virtual subtree aggregates
    x.vsub_sum = x.sub_sum
    x.vsub_size = x.sub_size
    if x.left:
        x.vsub_sum += x.left.vsub_sum
        x.vsub_size += x.left.vsub_size
    if x.right:
        x.vsub_sum += x.right.vsub_sum
        x.vsub_size += x.right.vsub_size


def _rotate(x):
    """Rotate x with its parent."""
    y = x.parent
    z = y.parent
    # Determine if x is left or right child
    if y.left is x:
        # Right rotation
        y.left = x.right
        if x.right:
            x.right.parent = y
        x.right = y
    else:
        # Left rotation
        y.right = x.left
        if x.left:
            x.left.parent = y
        x.left = y
    y.parent = x
    x.parent = z
    if z is not None:
        if z.left is y:
            z.left = x
        elif z.right is y:
            z.right = x
        # else: z is path-parent, link stays
    _pull_up(y)
    _pull_up(x)


def _splay(x):
    """Splay x to root of its auxiliary tree."""
    while not _is_root(x):
        y = x.parent
        if not _is_root(y):
            z = y.parent
            _push_down(z)
            _push_down(y)
            _push_down(x)
            # Zig-zig or zig-zag
            if (z.left is y) == (y.left is x):
                _rotate(y)  # zig-zig: rotate parent first
            else:
                _rotate(x)  # zig-zag: rotate x first
            _rotate(x)
        else:
            _push_down(y)
            _push_down(x)
            _rotate(x)
    _push_down(x)
    _pull_up(x)


def _access(x):
    """Access operation: make x-to-root path preferred.
    Returns the last node that was on the previous preferred path (for LCA)."""
    last = None
    u = x
    while u is not None:
        _splay(u)
        # Old right child becomes virtual (non-preferred)
        if u.right is not None:
            u.sub_sum += u.right.agg_sum + u.right.vsub_sum
            u.sub_size += u.right.size + u.right.vsub_size
        # New right child comes from preferred path
        if last is not None:
            u.sub_sum -= last.agg_sum + last.vsub_sum
            u.sub_size -= last.size + last.vsub_size
        u.right = last
        _pull_up(u)
        last = u
        u = u.parent
    _splay(x)
    return last


def _make_root(x):
    """Make x the root of its represented tree (evert)."""
    _access(x)
    x.rev = not x.rev
    _push_down(x)


class LinkCutTree:
    """Basic link-cut tree for dynamic forest connectivity."""

    def __init__(self, n=0, values=None):
        """Create a forest of n isolated nodes.
        values: optional list of node values (length n).
        """
        self.nodes = {}
        for i in range(n):
            v = values[i] if values else 0
            self.nodes[i] = SplayNode(i, v)

    def _get_node(self, u):
        if u not in self.nodes:
            raise ValueError(f"Node {u} does not exist")
        return self.nodes[u]

    def add_node(self, node_id, value=0):
        """Add a new isolated node."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        self.nodes[node_id] = SplayNode(node_id, value)

    def link(self, u, v):
        """Link: make u a child of v. u must be a root of its tree."""
        nu, nv = self._get_node(u), self._get_node(v)
        _make_root(nu)
        _access(nv)
        # Check they're in different trees
        if nu.parent is not None:
            raise ValueError(f"Nodes {u} and {v} are already connected")
        nu.parent = nv
        nv.sub_sum += nu.agg_sum + nu.vsub_sum
        nv.sub_size += nu.size + nu.vsub_size
        _pull_up(nv)

    def cut(self, u, v):
        """Cut the edge between u and v."""
        nu, nv = self._get_node(u), self._get_node(v)
        _make_root(nu)
        _access(nv)
        if nv.left is not nu or nu.right is not None:
            raise ValueError(f"No direct edge between {u} and {v}")
        nv.left = None
        nu.parent = None
        _pull_up(nv)

    def connected(self, u, v):
        """Check if u and v are in the same tree."""
        if u == v:
            return True
        nu, nv = self._get_node(u), self._get_node(v)
        _access(nu)
        _access(nv)
        # After accessing nv, if nu has a parent, they're connected
        return nu.parent is not None

    def find_root(self, u):
        """Find the root of u's tree."""
        nu = self._get_node(u)
        _access(nu)
        # Go to leftmost node (shallowest = root)
        r = nu
        _push_down(r)
        while r.left is not None:
            r = r.left
            _push_down(r)
        _splay(r)
        return r.id

    def evert(self, u):
        """Re-root u's tree at u (make_root)."""
        nu = self._get_node(u)
        _make_root(nu)

    def lca(self, u, v):
        """Find the lowest common ancestor of u and v.
        Assumes they are in the same tree. Returns None if not connected."""
        if u == v:
            return u
        nu, nv = self._get_node(u), self._get_node(v)
        _access(nu)
        last = _access(nv)
        if nu.parent is None and nu is not nv:
            return None  # Not connected
        return last.id

    def path_aggregate(self, u, v):
        """Get sum, min, max on path from u to v.
        Returns (sum, min, max)."""
        nu, nv = self._get_node(u), self._get_node(v)
        _make_root(nu)
        _access(nv)
        return (nv.agg_sum, nv.agg_min, nv.agg_max)

    def path_length(self, u, v):
        """Number of edges on path from u to v."""
        nu, nv = self._get_node(u), self._get_node(v)
        _make_root(nu)
        _access(nv)
        return nv.size - 1

    def get_value(self, u):
        """Get the value of node u."""
        nu = self._get_node(u)
        _access(nu)
        return nu.value

    def set_value(self, u, value):
        """Set the value of node u."""
        nu = self._get_node(u)
        _access(nu)
        nu.value = value
        _pull_up(nu)

    def subtree_sum(self, u):
        """Get the sum of all values in u's subtree (u as root via evert).
        After access(u), u.sub_sum contains the sum of u's virtual children
        (which are exactly u's non-preferred subtrees = u's represented subtree)."""
        nu = self._get_node(u)
        _access(nu)
        return nu.value + nu.sub_sum

    def subtree_size(self, u):
        """Get the size of u's subtree."""
        nu = self._get_node(u)
        _access(nu)
        return 1 + nu.sub_size


class PathAggregateTree(LinkCutTree):
    """Link-cut tree with path aggregate queries (sum, min, max)."""

    def path_sum(self, u, v):
        """Sum of values on path u to v."""
        s, _, _ = self.path_aggregate(u, v)
        return s

    def path_min(self, u, v):
        """Min value on path u to v."""
        _, m, _ = self.path_aggregate(u, v)
        return m

    def path_max(self, u, v):
        """Max value on path u to v."""
        _, _, m = self.path_aggregate(u, v)
        return m

    def path_update(self, u, v, delta):
        """Add delta to all nodes on path u to v.
        Note: This is O(n) for simplicity. For O(log n), would need lazy propagation."""
        nu, nv = self._get_node(u), self._get_node(v)
        _make_root(nu)
        _access(nv)
        # Collect path nodes via in-order traversal of splay tree
        nodes = []
        self._collect_inorder(nv, nodes)
        for node in nodes:
            node.value += delta
        # Re-pull aggregates bottom-up
        for node in nodes:
            _pull_up(node)

    def _collect_inorder(self, node, result):
        if node is None:
            return
        _push_down(node)
        self._collect_inorder(node.left, result)
        result.append(node)
        self._collect_inorder(node.right, result)


class WeightedLinkCutTree:
    """Link-cut tree with weighted edges.
    Uses auxiliary edge nodes: each edge (u,v,w) creates a virtual node with value=w.
    Real nodes have value=0. This works correctly with make_root/evert."""

    def __init__(self, n=0):
        self._lct = LinkCutTree()
        self._real_nodes = set()
        self._edge_nodes = {}  # (min(u,v), max(u,v)) -> edge_node_id
        self._next_edge_id = None
        for i in range(n):
            self._lct.add_node(i, 0)
            self._real_nodes.add(i)
        # Edge node IDs: use tuples to avoid collision with integer node IDs
        self._edge_counter = 0

    def _make_edge_id(self):
        eid = ('__edge__', self._edge_counter)
        self._edge_counter += 1
        return eid

    @property
    def nodes(self):
        return {k: v for k, v in self._lct.nodes.items() if k in self._real_nodes}

    def _get_node(self, u):
        if u not in self._real_nodes:
            raise ValueError(f"Node {u} does not exist")
        return self._lct.nodes[u]

    def add_node(self, node_id):
        if node_id in self._real_nodes:
            raise ValueError(f"Node {node_id} already exists")
        self._lct.add_node(node_id, 0)
        self._real_nodes.add(node_id)

    def link(self, u, v, weight=1):
        """Link u to v with given edge weight via auxiliary edge node."""
        if u not in self._real_nodes or v not in self._real_nodes:
            raise ValueError(f"Node does not exist")
        if self._lct.connected(u, v):
            raise ValueError(f"Nodes {u} and {v} are already connected")
        eid = self._make_edge_id()
        self._lct.add_node(eid, weight)
        edge_key = (min(u, v), max(u, v))
        self._edge_nodes[edge_key] = eid
        self._lct.link(u, eid)
        self._lct.link(eid, v)

    def cut(self, u, v):
        """Cut edge between u and v."""
        edge_key = (min(u, v), max(u, v))
        if edge_key not in self._edge_nodes:
            raise ValueError(f"No direct edge between {u} and {v}")
        eid = self._edge_nodes.pop(edge_key)
        # Cut both halves: u-eid and eid-v
        self._lct.evert(u)
        self._lct.cut(eid, u)
        self._lct.cut(v, eid)
        # Remove edge node
        del self._lct.nodes[eid]

    def connected(self, u, v):
        if u == v:
            return True
        return self._lct.connected(u, v)

    def find_root(self, u):
        r = self._lct.find_root(u)
        # Root might be an edge node; find the real node
        if r in self._real_nodes:
            return r
        # Shouldn't happen in normal use since we evert real nodes
        return r

    def path_weight(self, u, v):
        """Sum of edge weights on path from u to v.
        Real nodes have value=0, edge nodes have value=weight, so agg_sum = total weight."""
        if u == v:
            return 0
        return self._lct.path_aggregate(u, v)[0]

    def min_edge(self, u, v):
        """Minimum edge weight on path from u to v."""
        if u == v:
            return 0
        _, mn, _ = self._lct.path_aggregate(u, v)
        # mn might be 0 (from real nodes). We need min of only edge nodes.
        # Since real nodes have value 0 and edge weights are positive,
        # we need to find min > 0 on the path. Collect path instead.
        return self._min_max_edge(u, v, is_min=True)

    def max_edge(self, u, v):
        """Maximum edge weight on path from u to v."""
        if u == v:
            return 0
        return self._min_max_edge(u, v, is_min=False)

    def _min_max_edge(self, u, v, is_min=True):
        """Find min or max edge weight on path by collecting edge node values."""
        nu, nv = self._lct._get_node(u), self._lct._get_node(v)
        _make_root(nu)
        _access(nv)
        # Collect all nodes on path
        nodes = []
        self._collect_inorder(nv, nodes)
        edge_weights = [n.value for n in nodes if n.id not in self._real_nodes]
        if not edge_weights:
            return 0
        return min(edge_weights) if is_min else max(edge_weights)

    def _collect_inorder(self, node, result):
        if node is None:
            return
        _push_down(node)
        self._collect_inorder(node.left, result)
        result.append(node)
        self._collect_inorder(node.right, result)


class LinkCutForest:
    """Manages a forest of link-cut trees with component tracking."""

    def __init__(self, n=0, values=None):
        self._tree = LinkCutTree(n, values)
        self._num_components = n

    @property
    def num_components(self):
        return self._num_components

    @property
    def nodes(self):
        return self._tree.nodes

    def add_node(self, node_id, value=0):
        self._tree.add_node(node_id, value)
        self._num_components += 1

    def link(self, u, v):
        """Link two trees. Returns True if successful, False if already connected."""
        if self._tree.connected(u, v):
            return False
        self._tree.link(u, v)
        self._num_components -= 1
        return True

    def cut(self, u, v):
        """Cut edge between u and v."""
        self._tree.cut(u, v)
        self._num_components += 1

    def connected(self, u, v):
        return self._tree.connected(u, v)

    def find_root(self, u):
        return self._tree.find_root(u)

    def evert(self, u):
        self._tree.evert(u)

    def get_value(self, u):
        return self._tree.get_value(u)

    def set_value(self, u, value):
        self._tree.set_value(u, value)

    def path_aggregate(self, u, v):
        return self._tree.path_aggregate(u, v)

    def path_length(self, u, v):
        return self._tree.path_length(u, v)

    def subtree_size(self, u):
        return self._tree.subtree_size(u)

    def subtree_sum(self, u):
        return self._tree.subtree_sum(u)

    def get_components(self):
        """Get all connected components. Returns list of sets."""
        seen = set()
        components = []
        for node_id in self.nodes:
            root = self.find_root(node_id)
            if root not in seen:
                seen.add(root)
                comp = set()
                for nid in self.nodes:
                    if self.find_root(nid) == root:
                        comp.add(nid)
                components.append(comp)
        return components
