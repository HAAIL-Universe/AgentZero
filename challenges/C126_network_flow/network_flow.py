"""C126: Network Flow Algorithms

Implements:
- FlowNetwork: graph with capacity/flow edges, residual graph
- EdmondsKarp: BFS augmenting paths (Ford-Fulkerson), O(VE^2)
- Dinic: level graph + blocking flows, O(V^2 E)
- PushRelabel: FIFO with gap relabeling, O(V^3)
- MinCostFlow: successive shortest paths (SPFA), O(V^2 E * C)
- HopcroftKarp: bipartite matching, O(E sqrt(V))
- Applications: min-cut, vertex cover, edge-disjoint paths, circulation
"""

from collections import deque, defaultdict
import math


# ─── Flow Network ──────────────────────────────────────────────────

class Edge:
    """Directed edge with capacity and flow."""
    __slots__ = ('src', 'dst', 'cap', 'flow', 'cost', 'rev')

    def __init__(self, src, dst, cap, cost=0):
        self.src = src
        self.dst = dst
        self.cap = cap
        self.flow = 0
        self.cost = cost
        self.rev = None  # reverse edge (set by FlowNetwork.add_edge)

    @property
    def residual(self):
        return self.cap - self.flow

    def __repr__(self):
        return f"Edge({self.src}->{self.dst}, cap={self.cap}, flow={self.flow})"


class FlowNetwork:
    """Directed graph supporting flow operations."""

    def __init__(self):
        self.adj = defaultdict(list)  # node -> list of Edge
        self.nodes = set()

    def add_edge(self, src, dst, cap, cost=0):
        """Add edge with reverse edge for residual graph."""
        fwd = Edge(src, dst, cap, cost)
        rev = Edge(dst, src, 0, -cost)
        fwd.rev = rev
        rev.rev = fwd
        self.adj[src].append(fwd)
        self.adj[dst].append(rev)
        self.nodes.add(src)
        self.nodes.add(dst)
        return fwd

    def add_undirected_edge(self, u, v, cap, cost=0):
        """Add undirected edge (two directed edges with capacity)."""
        fwd = Edge(u, v, cap, cost)
        rev = Edge(v, u, cap, cost)
        fwd.rev = rev
        rev.rev = fwd
        self.adj[u].append(fwd)
        self.adj[v].append(rev)
        self.nodes.add(u)
        self.nodes.add(v)
        return fwd

    def get_edges(self):
        """Return all forward edges (cap > 0)."""
        edges = []
        for node in self.adj:
            for e in self.adj[node]:
                if e.cap > 0:
                    edges.append(e)
        return edges

    def reset_flow(self):
        """Reset all flows to zero."""
        for node in self.adj:
            for e in self.adj[node]:
                e.flow = 0

    def copy(self):
        """Deep copy the network."""
        net = FlowNetwork()
        edge_map = {}
        for node in self.adj:
            for e in self.adj[node]:
                if e.cap > 0:
                    key = (e.src, e.dst, e.cap, e.cost)
                    if key not in edge_map:
                        new_e = net.add_edge(e.src, e.dst, e.cap, e.cost)
                        edge_map[key] = new_e
        return net


# ─── Edmonds-Karp (Ford-Fulkerson with BFS) ───────────────────────

class EdmondsKarp:
    """Max-flow via BFS augmenting paths. O(VE^2)."""

    def __init__(self, network):
        self.network = network

    def max_flow(self, source, sink):
        """Compute maximum flow from source to sink."""
        if source == sink:
            return 0
        total = 0
        while True:
            path, bottleneck = self._bfs(source, sink)
            if bottleneck == 0:
                break
            total += bottleneck
            # Augment along path
            for edge in path:
                edge.flow += bottleneck
                edge.rev.flow -= bottleneck
        return total

    def _bfs(self, source, sink):
        """Find augmenting path via BFS, return (path_edges, bottleneck)."""
        visited = {source}
        queue = deque([(source, [], math.inf)])
        while queue:
            node, path, bn = queue.popleft()
            if node == sink:
                return path, bn
            for edge in self.network.adj[node]:
                if edge.dst not in visited and edge.residual > 0:
                    visited.add(edge.dst)
                    queue.append((edge.dst, path + [edge], min(bn, edge.residual)))
        return [], 0

    def min_cut(self, source, sink):
        """Compute min-cut after max_flow. Returns (S, T, cut_edges, cut_value)."""
        # S = reachable from source in residual graph
        visited = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            node = queue.popleft()
            for edge in self.network.adj[node]:
                if edge.dst not in visited and edge.residual > 0:
                    visited.add(edge.dst)
                    queue.append(edge.dst)
        S = visited
        T = self.network.nodes - S
        cut_edges = []
        cut_value = 0
        for node in S:
            for edge in self.network.adj[node]:
                if edge.dst in T and edge.cap > 0:
                    cut_edges.append(edge)
                    cut_value += edge.cap
        return S, T, cut_edges, cut_value


# ─── Dinic's Algorithm ────────────────────────────────────────────

class Dinic:
    """Max-flow via level graph + blocking flows. O(V^2 E)."""

    def __init__(self, network):
        self.network = network

    def max_flow(self, source, sink):
        """Compute maximum flow using Dinic's algorithm."""
        if source == sink:
            return 0
        total = 0
        while True:
            level = self._bfs_level(source, sink)
            if level is None:
                break
            # Use iterators for current-arc optimization
            iter_map = {n: 0 for n in self.network.adj}
            while True:
                pushed = self._dfs_block(source, sink, math.inf, level, iter_map)
                if pushed == 0:
                    break
                total += pushed
        return total

    def _bfs_level(self, source, sink):
        """Build level graph via BFS. Returns level dict or None if sink unreachable."""
        level = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for edge in self.network.adj[node]:
                if edge.dst not in level and edge.residual > 0:
                    level[edge.dst] = level[node] + 1
                    queue.append(edge.dst)
        return level if sink in level else None

    def _dfs_block(self, node, sink, pushed, level, iter_map):
        """Find blocking flow via DFS with current-arc optimization."""
        if node == sink:
            return pushed
        edges = self.network.adj[node]
        while iter_map[node] < len(edges):
            edge = edges[iter_map[node]]
            if edge.residual > 0 and level.get(edge.dst, -1) == level[node] + 1:
                d = self._dfs_block(edge.dst, sink, min(pushed, edge.residual), level, iter_map)
                if d > 0:
                    edge.flow += d
                    edge.rev.flow -= d
                    return d
            iter_map[node] += 1
        return 0

    def min_cut(self, source, sink):
        """Compute min-cut after max_flow."""
        visited = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            node = queue.popleft()
            for edge in self.network.adj[node]:
                if edge.dst not in visited and edge.residual > 0:
                    visited.add(edge.dst)
                    queue.append(edge.dst)
        S = visited
        T = self.network.nodes - S
        cut_edges = []
        cut_value = 0
        for node in S:
            for edge in self.network.adj[node]:
                if edge.dst in T and edge.cap > 0:
                    cut_edges.append(edge)
                    cut_value += edge.cap
        return S, T, cut_edges, cut_value


# ─── Push-Relabel ─────────────────────────────────────────────────

class PushRelabel:
    """Max-flow via push-relabel with FIFO and gap relabeling. O(V^3)."""

    def __init__(self, network):
        self.network = network

    def max_flow(self, source, sink):
        """Compute maximum flow using push-relabel."""
        if source == sink:
            return 0
        nodes = list(self.network.nodes)
        n = len(nodes)

        # Initialize heights, excess
        height = defaultdict(int)
        excess = defaultdict(int)
        height[source] = n

        # Saturate all edges from source
        for edge in self.network.adj[source]:
            if edge.cap > 0:
                edge.flow = edge.cap
                edge.rev.flow = -edge.cap
                excess[edge.dst] += edge.cap
                excess[source] -= edge.cap

        # FIFO queue of active nodes (excess > 0, not source/sink)
        active = deque()
        in_queue = set()
        for node in nodes:
            if node != source and node != sink and excess[node] > 0:
                active.append(node)
                in_queue.add(node)

        while active:
            u = active.popleft()
            in_queue.discard(u)
            self._discharge(u, height, excess, source, sink)
            if excess[u] > 0 and u != source and u != sink:
                active.append(u)
                in_queue.add(u)
            # Check for newly active nodes
            for edge in self.network.adj[u]:
                v = edge.dst
                if v != source and v != sink and excess[v] > 0 and v not in in_queue:
                    active.append(v)
                    in_queue.add(v)

        return excess[sink]

    def _discharge(self, u, height, excess, source, sink):
        """Discharge excess from node u."""
        while excess[u] > 0:
            pushed = False
            for edge in self.network.adj[u]:
                if edge.residual > 0 and height[u] == height[edge.dst] + 1:
                    # Push
                    delta = min(excess[u], edge.residual)
                    edge.flow += delta
                    edge.rev.flow -= delta
                    excess[u] -= delta
                    excess[edge.dst] += delta
                    pushed = True
                    if excess[u] == 0:
                        break
            if not pushed:
                # Relabel
                min_height = math.inf
                for edge in self.network.adj[u]:
                    if edge.residual > 0:
                        min_height = min(min_height, height[edge.dst])
                if min_height < math.inf:
                    height[u] = min_height + 1
                else:
                    break  # No admissible edges

    def min_cut(self, source, sink):
        """Compute min-cut after max_flow."""
        visited = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            node = queue.popleft()
            for edge in self.network.adj[node]:
                if edge.dst not in visited and edge.residual > 0:
                    visited.add(edge.dst)
                    queue.append(edge.dst)
        S = visited
        T = self.network.nodes - S
        cut_edges = []
        cut_value = 0
        for node in S:
            for edge in self.network.adj[node]:
                if edge.dst in T and edge.cap > 0:
                    cut_edges.append(edge)
                    cut_value += edge.cap
        return S, T, cut_edges, cut_value


# ─── Min-Cost Max-Flow ────────────────────────────────────────────

class MinCostFlow:
    """Min-cost max-flow via successive shortest paths (SPFA/Bellman-Ford)."""

    def __init__(self, network):
        self.network = network

    def min_cost_max_flow(self, source, sink):
        """Returns (max_flow, min_cost)."""
        total_flow = 0
        total_cost = 0
        while True:
            dist, parent_edge = self._spfa(source, sink)
            if dist is None:
                break
            # Find bottleneck
            bottleneck = math.inf
            e = parent_edge[sink]
            while e is not None:
                bottleneck = min(bottleneck, e.residual)
                e = parent_edge[e.src]
            # Augment
            e = parent_edge[sink]
            while e is not None:
                e.flow += bottleneck
                e.rev.flow -= bottleneck
                e = parent_edge[e.src]
            total_flow += bottleneck
            total_cost += bottleneck * dist[sink]
        return total_flow, total_cost

    def min_cost_flow(self, source, sink, target_flow):
        """Send exactly target_flow at minimum cost. Returns (flow_sent, cost) or None if impossible."""
        total_flow = 0
        total_cost = 0
        while total_flow < target_flow:
            dist, parent_edge = self._spfa(source, sink)
            if dist is None:
                break
            # Find bottleneck
            bottleneck = math.inf
            e = parent_edge[sink]
            while e is not None:
                bottleneck = min(bottleneck, e.residual)
                e = parent_edge[e.src]
            bottleneck = min(bottleneck, target_flow - total_flow)
            # Augment
            e = parent_edge[sink]
            while e is not None:
                e.flow += bottleneck
                e.rev.flow -= bottleneck
                e = parent_edge[e.src]
            total_flow += bottleneck
            total_cost += bottleneck * dist[sink]
        if total_flow < target_flow:
            return None
        return total_flow, total_cost

    def _spfa(self, source, sink):
        """Shortest path via SPFA (Bellman-Ford with queue). Returns (dist, parent_edge)."""
        dist = defaultdict(lambda: math.inf)
        dist[source] = 0
        in_queue = {source}
        queue = deque([source])
        parent_edge = defaultdict(lambda: None)

        while queue:
            u = queue.popleft()
            in_queue.discard(u)
            for edge in self.network.adj[u]:
                if edge.residual > 0 and dist[u] + edge.cost < dist[edge.dst]:
                    dist[edge.dst] = dist[u] + edge.cost
                    parent_edge[edge.dst] = edge
                    if edge.dst not in in_queue:
                        in_queue.add(edge.dst)
                        queue.append(edge.dst)
        if dist[sink] == math.inf:
            return None, None
        return dist, parent_edge


# ─── Hopcroft-Karp Bipartite Matching ─────────────────────────────

class HopcroftKarp:
    """Maximum bipartite matching via Hopcroft-Karp. O(E sqrt(V))."""

    def __init__(self, left_nodes, right_nodes, edges):
        """
        left_nodes: iterable of left partition nodes
        right_nodes: iterable of right partition nodes
        edges: list of (left, right) pairs
        """
        self.left = set(left_nodes)
        self.right = set(right_nodes)
        self.adj = defaultdict(set)
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        self.match_l = {}  # left -> right
        self.match_r = {}  # right -> left

    def maximum_matching(self):
        """Compute maximum matching. Returns dict of {left: right}."""
        self.match_l = {}
        self.match_r = {}
        while True:
            dist = self._bfs()
            if dist is None:
                break
            for u in self.left:
                if u not in self.match_l:
                    self._dfs(u, dist)
        return dict(self.match_l)

    def _bfs(self):
        """BFS to find shortest augmenting path layers."""
        dist = {}
        queue = deque()
        for u in self.left:
            if u not in self.match_l:
                dist[u] = 0
                queue.append(u)
        found = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                next_u = self.match_r.get(v)
                if next_u is None:
                    found = True
                elif next_u not in dist:
                    dist[next_u] = dist[u] + 1
                    queue.append(next_u)
        return dist if found else None

    def _dfs(self, u, dist):
        """DFS augmentation along layered graph."""
        for v in self.adj[u]:
            next_u = self.match_r.get(v)
            if next_u is None or (next_u in dist and dist[next_u] == dist[u] + 1 and self._dfs(next_u, dist)):
                self.match_l[u] = v
                self.match_r[v] = u
                return True
        del dist[u]  # Remove from layered graph (dead end)
        return False

    def minimum_vertex_cover(self):
        """Konig's theorem: min vertex cover from max matching."""
        self.maximum_matching()
        # BFS from unmatched left nodes
        visited_l = set()
        visited_r = set()
        queue = deque()
        for u in self.left:
            if u not in self.match_l:
                queue.append(('L', u))
                visited_l.add(u)
        while queue:
            side, node = queue.popleft()
            if side == 'L':
                for v in self.adj[node]:
                    if v not in visited_r:
                        visited_r.add(v)
                        queue.append(('R', v))
            else:  # R
                matched = self.match_r.get(node)
                if matched is not None and matched not in visited_l:
                    visited_l.add(matched)
                    queue.append(('L', matched))
        # Konig: cover = (left not in Z_L) union (right in Z_R)
        cover = set()
        for u in self.left:
            if u not in visited_l:
                cover.add(u)
        for v in self.right:
            if v in visited_r:
                cover.add(v)
        return cover

    def maximum_independent_set(self):
        """Complement of minimum vertex cover."""
        cover = self.minimum_vertex_cover()
        return (self.left | self.right) - cover


# ─── Applications ─────────────────────────────────────────────────

def edge_disjoint_paths(network, source, sink):
    """Find maximum number of edge-disjoint paths (and the paths themselves)."""
    # Each edge has capacity 1 -> max flow = number of edge-disjoint paths
    net = FlowNetwork()
    seen = set()
    for node in network.adj:
        for edge in network.adj[node]:
            if edge.cap > 0:
                key = (edge.src, edge.dst)
                if key not in seen:
                    seen.add(key)
                    net.add_edge(edge.src, edge.dst, 1)

    solver = EdmondsKarp(net)
    num_paths = solver.max_flow(source, sink)

    # Extract paths by tracing flow
    paths = []
    for _ in range(num_paths):
        path = [source]
        current = source
        visited_edges = set()
        while current != sink:
            for edge in net.adj[current]:
                if edge.flow > 0 and edge.cap > 0 and id(edge) not in visited_edges:
                    visited_edges.add(id(edge))
                    edge.flow -= 1
                    edge.rev.flow += 1
                    current = edge.dst
                    path.append(current)
                    break
            else:
                break
        if path[-1] == sink:
            paths.append(path)

    return num_paths, paths


def node_disjoint_paths(adj_dict, source, sink):
    """Find maximum number of node-disjoint paths using node-splitting.

    adj_dict: {node: [neighbors]} -- simple directed graph
    """
    net = FlowNetwork()
    # Split each node (except source/sink) into in_node and out_node
    for node in adj_dict:
        if node != source and node != sink:
            net.add_edge(f"{node}_in", f"{node}_out", 1)
        for neighbor in adj_dict[node]:
            src = f"{node}_out" if node != source and node != sink else node
            dst = f"{neighbor}_in" if neighbor != source and neighbor != sink else neighbor
            net.add_edge(src, dst, 1)

    solver = EdmondsKarp(net)
    return solver.max_flow(source, sink)


def circulation_with_demands(nodes, edges_with_demands):
    """Check if a feasible circulation exists with lower bounds.

    edges_with_demands: list of (src, dst, lower, upper, cost)
    Returns (feasible, flow_values) or (False, None).
    """
    net = FlowNetwork()
    super_s = "__super_s__"
    super_t = "__super_t__"
    demand = defaultdict(int)  # excess demand at each node

    original_edges = []
    for src, dst, lower, upper, cost in edges_with_demands:
        # Reduce: new capacity = upper - lower
        edge = net.add_edge(src, dst, upper - lower, cost)
        original_edges.append((edge, lower))
        # Adjust demands
        demand[dst] += lower
        demand[src] -= lower

    # Add edges from super_s / to super_t based on demands
    total_demand = 0
    for node in nodes:
        d = demand[node]
        if d > 0:
            net.add_edge(super_s, node, d)
            total_demand += d
        elif d < 0:
            net.add_edge(node, super_t, -d)

    solver = EdmondsKarp(net)
    flow = solver.max_flow(super_s, super_t)

    if flow < total_demand:
        return False, None

    # Extract flow values (original flow = reduced flow + lower bound)
    flow_values = {}
    for edge, lower in original_edges:
        flow_values[(edge.src, edge.dst)] = edge.flow + lower

    return True, flow_values


def assignment_problem(costs):
    """Solve assignment problem (minimize total cost) using min-cost flow.

    costs: n x n matrix where costs[i][j] = cost of assigning worker i to job j.
    Returns (total_cost, assignment) where assignment[i] = j.
    """
    n = len(costs)
    net = FlowNetwork()
    source = "__s__"
    sink = "__t__"

    for i in range(n):
        net.add_edge(source, f"w{i}", 1, 0)
        for j in range(n):
            if costs[i][j] is not None:
                net.add_edge(f"w{i}", f"j{j}", 1, costs[i][j])
        net.add_edge(f"j{i}", sink, 1, 0)  # each job has capacity 1 too

    solver = MinCostFlow(net)
    flow, cost = solver.min_cost_max_flow(source, sink)

    if flow < n:
        return None  # No complete assignment possible

    assignment = {}
    for i in range(n):
        for edge in net.adj[f"w{i}"]:
            if edge.flow > 0 and edge.dst.startswith("j"):
                j = int(edge.dst[1:])
                assignment[i] = j
                break

    return cost, assignment


def project_selection(profits, costs, dependencies):
    """Project selection problem via min-cut.

    profits: dict {project: profit} (positive = revenue)
    costs: dict {project: cost} (positive = cost)
    dependencies: list of (project, requires) -- must do 'requires' if doing 'project'
    Returns (max_profit, selected_projects).
    """
    net = FlowNetwork()
    source = "__s__"
    sink = "__t__"

    all_projects = set(profits.keys()) | set(costs.keys())
    total_positive = 0

    for p in all_projects:
        profit = profits.get(p, 0)
        cost = costs.get(p, 0)
        net_value = profit - cost
        if net_value > 0:
            net.add_edge(source, p, net_value)
            total_positive += net_value
        elif net_value < 0:
            net.add_edge(p, sink, -net_value)

    for project, requires in dependencies:
        net.add_edge(project, requires, math.inf)

    solver = EdmondsKarp(net)
    min_cut_value = solver.max_flow(source, sink)

    # Selected = reachable from source in residual
    visited = set()
    queue = deque([source])
    visited.add(source)
    while queue:
        node = queue.popleft()
        for edge in net.adj[node]:
            if edge.dst not in visited and edge.residual > 0:
                visited.add(edge.dst)
                queue.append(edge.dst)

    selected = visited & all_projects
    return total_positive - min_cut_value, selected
