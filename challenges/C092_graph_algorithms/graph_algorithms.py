"""
C092: Graph Algorithms
Shortest paths (Dijkstra, A*, Bellman-Ford, Floyd-Warshall),
MST (Kruskal, Prim), Network flow (Edmonds-Karp),
Topological sort, SCC (Tarjan), Bipartite check, Cycle detection.
"""

import heapq
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Graph data structure
# ---------------------------------------------------------------------------

class Graph:
    """Weighted graph supporting directed and undirected modes."""

    def __init__(self, directed=False):
        self.directed = directed
        self._adj = defaultdict(list)   # node -> [(neighbor, weight)]
        self._nodes = set()

    def add_node(self, node):
        self._nodes.add(node)

    def add_edge(self, u, v, weight=1.0):
        self._nodes.add(u)
        self._nodes.add(v)
        self._adj[u].append((v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))

    def neighbors(self, node):
        """Return list of (neighbor, weight) pairs."""
        return list(self._adj[node])

    def nodes(self):
        return set(self._nodes)

    def edges(self):
        """Return list of (u, v, weight) triples."""
        seen = set()
        result = []
        for u in self._adj:
            for v, w in self._adj[u]:
                if self.directed:
                    result.append((u, v, w))
                else:
                    key = (min(u, v), max(u, v)) if isinstance(u, (int, float, str)) else (u, v)
                    if key not in seen:
                        seen.add(key)
                        result.append((u, v, w))
        return result

    def has_node(self, node):
        return node in self._nodes

    def has_edge(self, u, v):
        return any(n == v for n, _ in self._adj[u])

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self.edges())

    def degree(self, node):
        """Number of edges incident to node."""
        if self.directed:
            out_deg = len(self._adj[node])
            in_deg = sum(1 for u in self._adj for v, _ in self._adj[u] if v == node)
            return out_deg + in_deg
        return len(self._adj[node])

    def out_degree(self, node):
        return len(self._adj[node])

    def in_degree(self, node):
        return sum(1 for u in self._adj for v, _ in self._adj[u] if v == node)

    def reverse(self):
        """Return a new graph with all edges reversed (directed only)."""
        if not self.directed:
            raise ValueError("Cannot reverse an undirected graph")
        g = Graph(directed=True)
        for node in self._nodes:
            g.add_node(node)
        for u in self._adj:
            for v, w in self._adj[u]:
                g.add_edge(v, u, w)
        return g

    def subgraph(self, nodes):
        """Return induced subgraph on given node set."""
        node_set = set(nodes)
        g = Graph(directed=self.directed)
        for n in node_set:
            if n in self._nodes:
                g.add_node(n)
        for u in node_set:
            for v, w in self._adj[u]:
                if v in node_set:
                    g._adj[u].append((v, w))
                    g._nodes.add(u)
                    g._nodes.add(v)
        return g


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

def dijkstra(graph, source, target=None):
    """
    Dijkstra's algorithm for single-source shortest paths.
    Returns (distances, predecessors).
    If target is given, stops early when target is reached.
    """
    dist = {source: 0.0}
    pred = {source: None}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if target is not None and u == target:
            break
        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("Dijkstra does not support negative weights")
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, pred


def reconstruct_path(pred, target):
    """Reconstruct path from predecessors dict."""
    if target not in pred:
        return None
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()
    return path


def bellman_ford(graph, source):
    """
    Bellman-Ford single-source shortest paths.
    Supports negative weights. Detects negative cycles.
    Returns (distances, predecessors) or raises ValueError for negative cycles.
    """
    nodes = graph.nodes()
    dist = {n: float('inf') for n in nodes}
    pred = {n: None for n in nodes}
    dist[source] = 0.0

    # Collect all directed edges
    all_edges = []
    for u in graph._adj:
        for v, w in graph._adj[u]:
            all_edges.append((u, v, w))

    # Relax |V|-1 times
    for _ in range(len(nodes) - 1):
        changed = False
        for u, v, w in all_edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                changed = True
        if not changed:
            break

    # Check for negative cycles
    for u, v, w in all_edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            raise ValueError("Graph contains a negative weight cycle")

    return dist, pred


def floyd_warshall(graph):
    """
    Floyd-Warshall all-pairs shortest paths.
    Returns (dist_matrix, next_matrix) as dicts of dicts.
    """
    nodes = list(graph.nodes())
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    nxt = {u: {v: None for v in nodes} for u in nodes}

    for u in nodes:
        dist[u][u] = 0.0

    for u in graph._adj:
        for v, w in graph._adj[u]:
            if w < dist[u][v]:
                dist[u][v] = w
                nxt[u][v] = v

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]

    return dist, nxt


def floyd_warshall_path(nxt, u, v):
    """Reconstruct path from Floyd-Warshall next matrix."""
    if nxt[u][v] is None:
        return None
    path = [u]
    while u != v:
        u = nxt[u][v]
        path.append(u)
    return path


def astar(graph, source, target, heuristic):
    """
    A* search algorithm.
    heuristic(node, target) -> estimated cost to target.
    Returns (distance, path) or (inf, None) if unreachable.
    """
    g_score = {source: 0.0}
    f_score = {source: heuristic(source, target)}
    pred = {source: None}
    pq = [(f_score[source], id(source), source)]
    closed = set()

    while pq:
        f, _, u = heapq.heappop(pq)
        if u == target:
            return g_score[u], reconstruct_path(pred, target)
        if u in closed:
            continue
        closed.add(u)
        for v, w in graph.neighbors(u):
            if v in closed:
                continue
            tentative = g_score[u] + w
            if v not in g_score or tentative < g_score[v]:
                g_score[v] = tentative
                f_score[v] = tentative + heuristic(v, target)
                pred[v] = u
                heapq.heappush(pq, (f_score[v], id(v), v))

    return float('inf'), None


# ---------------------------------------------------------------------------
# Minimum Spanning Tree
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint set union with path compression and union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(graph):
    """
    Kruskal's MST algorithm.
    Returns list of (u, v, weight) edges in the MST and total weight.
    """
    edges = sorted(graph.edges(), key=lambda e: e[2])
    uf = UnionFind()
    for node in graph.nodes():
        uf.make_set(node)

    mst = []
    total = 0.0
    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if len(mst) == graph.node_count() - 1:
                break

    return mst, total


def prim(graph, start=None):
    """
    Prim's MST algorithm.
    Returns list of (u, v, weight) edges in the MST and total weight.
    """
    nodes = graph.nodes()
    if not nodes:
        return [], 0.0
    if start is None:
        start = next(iter(nodes))

    in_mst = {start}
    mst = []
    total = 0.0
    # Push all edges from start
    pq = []
    for v, w in graph.neighbors(start):
        heapq.heappush(pq, (w, start, v))

    while pq and len(in_mst) < len(nodes):
        w, u, v = heapq.heappop(pq)
        if v in in_mst:
            continue
        in_mst.add(v)
        mst.append((u, v, w))
        total += w
        for nv, nw in graph.neighbors(v):
            if nv not in in_mst:
                heapq.heappush(pq, (nw, v, nv))

    return mst, total


# ---------------------------------------------------------------------------
# Network Flow
# ---------------------------------------------------------------------------

def edmonds_karp(graph, source, sink):
    """
    Edmonds-Karp max flow algorithm (BFS-based Ford-Fulkerson).
    Returns (max_flow_value, flow_dict).
    flow_dict[u][v] = flow on edge (u, v).
    """
    # Build capacity and flow dicts
    capacity = defaultdict(lambda: defaultdict(float))
    flow = defaultdict(lambda: defaultdict(float))
    all_nodes = set()

    for u in graph._adj:
        for v, w in graph._adj[u]:
            capacity[u][v] += w  # handle parallel edges
            all_nodes.add(u)
            all_nodes.add(v)

    def bfs_path():
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            u, path = queue.popleft()
            for v in all_nodes:
                if v not in visited and capacity[u][v] - flow[u][v] > 1e-12:
                    visited.add(v)
                    new_path = path + [v]
                    if v == sink:
                        return new_path
                    queue.append((v, new_path))
        return None

    max_flow = 0.0
    while True:
        path = bfs_path()
        if path is None:
            break
        # Find bottleneck
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            bottleneck = min(bottleneck, capacity[u][v] - flow[u][v])
        # Update flow
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow[u][v] += bottleneck
            flow[v][u] -= bottleneck
        max_flow += bottleneck

    return max_flow, dict(flow)


def min_cut(graph, source, sink):
    """
    Find minimum cut using max flow.
    Returns (max_flow_value, S_set, T_set) where S contains source.
    """
    max_flow_val, flow_dict = edmonds_karp(graph, source, sink)

    # Build residual capacity
    capacity = defaultdict(lambda: defaultdict(float))
    for u in graph._adj:
        for v, w in graph._adj[u]:
            capacity[u][v] += w

    flow = defaultdict(lambda: defaultdict(float))
    for u in flow_dict:
        for v in flow_dict[u]:
            flow[u][v] = flow_dict[u][v]

    # BFS on residual graph from source
    visited = {source}
    queue = deque([source])
    all_nodes = graph.nodes()
    while queue:
        u = queue.popleft()
        for v in all_nodes:
            if v not in visited and capacity[u][v] - flow[u][v] > 1e-12:
                visited.add(v)
                queue.append(v)

    s_set = visited
    t_set = all_nodes - s_set
    return max_flow_val, s_set, t_set


# ---------------------------------------------------------------------------
# Topological Sort
# ---------------------------------------------------------------------------

def topological_sort(graph):
    """
    Kahn's algorithm for topological sort.
    Returns ordered list or raises ValueError if graph has a cycle.
    Requires directed graph.
    """
    if not graph.directed:
        raise ValueError("Topological sort requires a directed graph")

    in_deg = defaultdict(int)
    for node in graph.nodes():
        in_deg[node] = in_deg.get(node, 0)
    for u in graph._adj:
        for v, _ in graph._adj[u]:
            in_deg[v] = in_deg.get(v, 0) + 1

    queue = deque(sorted(n for n in graph.nodes() if in_deg[n] == 0))
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v, _ in graph.neighbors(u):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(result) != graph.node_count():
        raise ValueError("Graph contains a cycle")

    return result


# ---------------------------------------------------------------------------
# Strongly Connected Components (Tarjan)
# ---------------------------------------------------------------------------

def tarjan_scc(graph):
    """
    Tarjan's algorithm for strongly connected components.
    Returns list of SCCs (each SCC is a list of nodes).
    Requires directed graph.
    """
    if not graph.directed:
        raise ValueError("SCC requires a directed graph")

    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w, _ in graph.neighbors(v):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    for node in sorted(graph.nodes(), key=str):
        if node not in index:
            strongconnect(node)

    return result


# ---------------------------------------------------------------------------
# Cycle Detection
# ---------------------------------------------------------------------------

def has_cycle(graph):
    """Detect if graph contains a cycle."""
    if graph.directed:
        # DFS with coloring: white=0, gray=1, black=2
        color = {n: 0 for n in graph.nodes()}

        def dfs(u):
            color[u] = 1
            for v, _ in graph.neighbors(u):
                if color[v] == 1:
                    return True
                if color[v] == 0 and dfs(v):
                    return True
            color[u] = 2
            return False

        return any(color[n] == 0 and dfs(n) for n in graph.nodes())
    else:
        # Undirected: DFS with parent tracking
        visited = set()

        def dfs(u, parent):
            visited.add(u)
            for v, _ in graph.neighbors(u):
                if v not in visited:
                    if dfs(v, u):
                        return True
                elif v != parent:
                    return True
            return False

        for n in graph.nodes():
            if n not in visited:
                if dfs(n, None):
                    return True
        return False


def find_cycle(graph):
    """Find and return a cycle if one exists, or None."""
    if graph.directed:
        color = {n: 0 for n in graph.nodes()}
        pred = {}

        def dfs(u):
            color[u] = 1
            for v, _ in graph.neighbors(u):
                if color[v] == 1:
                    # Found cycle, reconstruct
                    cycle = [v, u]
                    node = u
                    while node != v:
                        node = pred.get(node)
                        if node is None:
                            break
                        cycle.append(node)
                    cycle.reverse()
                    return cycle
                if color[v] == 0:
                    pred[v] = u
                    result = dfs(v)
                    if result:
                        return result
            color[u] = 2
            return None

        for n in sorted(graph.nodes(), key=str):
            if color[n] == 0:
                result = dfs(n)
                if result:
                    return result
        return None
    else:
        visited = set()

        def dfs(u, parent):
            visited.add(u)
            for v, _ in graph.neighbors(u):
                if v not in visited:
                    result = dfs(v, u)
                    if result is not None:
                        return result
                elif v != parent:
                    return [v, u]  # simple cycle indication
            return None

        for n in graph.nodes():
            if n not in visited:
                result = dfs(n, None)
                if result:
                    return result
        return None


# ---------------------------------------------------------------------------
# Bipartite Check
# ---------------------------------------------------------------------------

def is_bipartite(graph):
    """
    Check if graph is bipartite using BFS coloring.
    Returns (True, partition) or (False, None).
    partition is a tuple of two sets.
    """
    color = {}
    for start in graph.nodes():
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v, _ in graph.neighbors(u):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, None

    set_a = {n for n, c in color.items() if c == 0}
    set_b = {n for n, c in color.items() if c == 1}
    return True, (set_a, set_b)


# ---------------------------------------------------------------------------
# Connected Components
# ---------------------------------------------------------------------------

def connected_components(graph):
    """Find connected components (undirected) or weakly connected (directed)."""
    visited = set()
    components = []

    # For directed graphs, ignore edge direction
    adj = defaultdict(set)
    for u in graph._adj:
        for v, _ in graph._adj[u]:
            adj[u].add(v)
            adj[v].add(u)

    for start in sorted(graph.nodes(), key=str):
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        components.append(component)

    return components


# ---------------------------------------------------------------------------
# BFS / DFS Traversals
# ---------------------------------------------------------------------------

def bfs(graph, source):
    """BFS traversal. Returns list of nodes in BFS order and distances."""
    visited = {source}
    queue = deque([(source, 0)])
    order = []
    distances = {source: 0}

    while queue:
        u, d = queue.popleft()
        order.append(u)
        for v, _ in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                distances[v] = d + 1
                queue.append((v, d + 1))

    return order, distances


def dfs(graph, source):
    """DFS traversal. Returns list of nodes in DFS order."""
    visited = set()
    order = []

    def visit(u):
        visited.add(u)
        order.append(u)
        for v, _ in sorted(graph.neighbors(u)):
            if v not in visited:
                visit(v)

    visit(source)
    return order
