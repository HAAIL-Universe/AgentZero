"""
C093: Network Analysis -- composing C092 Graph Algorithms.

PageRank, centrality measures (betweenness, closeness, degree, eigenvector),
HITS (hubs/authorities), community detection (Louvain, label propagation),
k-core decomposition, clustering coefficients.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C092_graph_algorithms'))

from graph_algorithms import Graph
from collections import defaultdict, deque
import heapq
import random
import math


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank for a directed graph using power iteration.

    Returns dict mapping node -> rank (sums to 1.0).
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    rank = {node: 1.0 / n for node in nodes}

    # Build adjacency: out-neighbors for each node
    out_neighbors = defaultdict(list)
    out_degree = defaultdict(int)
    for node in nodes:
        for neighbor, _ in graph.neighbors(node):
            out_neighbors[node].append(neighbor)
            out_degree[node] += 1

    for _ in range(max_iter):
        new_rank = {}
        # Dangling node mass (nodes with no outgoing edges)
        dangling_sum = sum(rank[node] for node in nodes if out_degree[node] == 0)

        for node in nodes:
            # Sum of rank flowing in from predecessors
            incoming = 0.0
            for other in nodes:
                if node in out_neighbors[other]:
                    incoming += rank[other] / out_degree[other]

            new_rank[node] = (1.0 - damping) / n + damping * (incoming + dangling_sum / n)

        # Check convergence
        diff = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if diff < tol:
            break

    return rank


# ---------------------------------------------------------------------------
# Degree Centrality
# ---------------------------------------------------------------------------

def degree_centrality(graph):
    """Normalized degree centrality for each node.

    For undirected: deg(v) / (n-1).
    For directed: returns (in_centrality, out_centrality) dicts.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        if graph.directed:
            return ({node: 0.0 for node in nodes}, {node: 0.0 for node in nodes})
        return {node: 0.0 for node in nodes}

    denom = n - 1

    if graph.directed:
        in_cent = {}
        out_cent = {}
        # Count in-degree and out-degree
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for node in nodes:
            for neighbor, _ in graph.neighbors(node):
                out_deg[node] += 1
                in_deg[neighbor] += 1
        for node in nodes:
            in_cent[node] = in_deg[node] / denom
            out_cent[node] = out_deg[node] / denom
        return (in_cent, out_cent)
    else:
        result = {}
        for node in nodes:
            result[node] = len(graph.neighbors(node)) / denom
        return result


# ---------------------------------------------------------------------------
# Closeness Centrality
# ---------------------------------------------------------------------------

def _bfs_distances(graph, source):
    """BFS shortest path distances (unweighted) from source."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, _ in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def closeness_centrality(graph):
    """Closeness centrality: 1 / avg_distance for reachable nodes.

    For disconnected graphs, uses Wasserman-Faust formula:
    (reachable-1)/(n-1) * (reachable-1)/sum_distances.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}

    result = {}
    for node in nodes:
        dist = _bfs_distances(graph, node)
        reachable = len(dist) - 1  # exclude self
        if reachable == 0:
            result[node] = 0.0
        else:
            total_dist = sum(d for v, d in dist.items() if v != node)
            if total_dist == 0:
                result[node] = 0.0
            else:
                # Wasserman-Faust normalization for disconnected graphs
                result[node] = (reachable / (n - 1)) * (reachable / total_dist)
    return result


# ---------------------------------------------------------------------------
# Betweenness Centrality (Brandes' algorithm)
# ---------------------------------------------------------------------------

def betweenness_centrality(graph, normalized=True):
    """Brandes' algorithm for betweenness centrality. O(VE) for unweighted."""
    nodes = list(graph.nodes())
    n = len(nodes)
    betweenness = {v: 0.0 for v in nodes}

    for s in nodes:
        # BFS from s
        stack = []
        predecessors = {v: [] for v in nodes}
        sigma = {v: 0 for v in nodes}
        sigma[s] = 1
        dist = {v: -1 for v in nodes}
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w, _ in graph.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        # Back-propagation
        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # For undirected graphs, each shortest path is counted twice
    if not graph.directed:
        for v in nodes:
            betweenness[v] /= 2.0

    if normalized and n > 2:
        if graph.directed:
            norm = 1.0 / ((n - 1) * (n - 2))
        else:
            norm = 1.0 / ((n - 1) * (n - 2) / 2.0)
        # Standard normalization: divide by (n-1)(n-2) for directed
        # or (n-1)(n-2)/2 for undirected
        norm_factor = (n - 1) * (n - 2)
        if not graph.directed:
            norm_factor /= 2.0
        for v in nodes:
            betweenness[v] /= norm_factor

    return betweenness


# ---------------------------------------------------------------------------
# Eigenvector Centrality (Power iteration)
# ---------------------------------------------------------------------------

def eigenvector_centrality(graph, max_iter=100, tol=1e-6):
    """Eigenvector centrality via power iteration on adjacency matrix."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # Start with uniform vector
    x = {node: 1.0 / n for node in nodes}

    # Build adjacency for undirected, or in-neighbors for directed
    adj = defaultdict(list)
    for node in nodes:
        for neighbor, _ in graph.neighbors(node):
            if graph.directed:
                # For eigenvector centrality on directed graph, use in-neighbors
                adj[neighbor].append(node)
            else:
                adj[node].append(neighbor)

    for _ in range(max_iter):
        new_x = {}
        for node in nodes:
            new_x[node] = sum(x.get(neighbor, 0.0) for neighbor in adj[node])

        # Normalize
        norm = math.sqrt(sum(v * v for v in new_x.values()))
        if norm == 0:
            return {node: 0.0 for node in nodes}
        for node in nodes:
            new_x[node] /= norm

        # Check convergence
        diff = sum(abs(new_x[node] - x[node]) for node in nodes)
        x = new_x
        if diff < tol:
            break

    return x


# ---------------------------------------------------------------------------
# HITS (Hyperlink-Induced Topic Search) -- Kleinberg
# ---------------------------------------------------------------------------

def hits(graph, max_iter=100, tol=1e-6):
    """Compute hub and authority scores for a directed graph.

    Returns (hubs, authorities) dicts.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}, {}

    hub = {node: 1.0 for node in nodes}
    auth = {node: 1.0 for node in nodes}

    # Build out-neighbors and in-neighbors
    out_nbrs = defaultdict(list)
    in_nbrs = defaultdict(list)
    for node in nodes:
        for neighbor, _ in graph.neighbors(node):
            out_nbrs[node].append(neighbor)
            in_nbrs[neighbor].append(node)

    for _ in range(max_iter):
        # Authority update: auth(v) = sum of hub scores of in-neighbors
        new_auth = {}
        for node in nodes:
            new_auth[node] = sum(hub[u] for u in in_nbrs[node])

        # Hub update: hub(v) = sum of authority scores of out-neighbors
        new_hub = {}
        for node in nodes:
            new_hub[node] = sum(new_auth[v] for v in out_nbrs[node])

        # Normalize
        auth_norm = math.sqrt(sum(v * v for v in new_auth.values()))
        hub_norm = math.sqrt(sum(v * v for v in new_hub.values()))

        if auth_norm > 0:
            for node in nodes:
                new_auth[node] /= auth_norm
        if hub_norm > 0:
            for node in nodes:
                new_hub[node] /= hub_norm

        # Check convergence
        diff = sum(abs(new_auth[node] - auth[node]) for node in nodes)
        diff += sum(abs(new_hub[node] - hub[node]) for node in nodes)
        auth = new_auth
        hub = new_hub
        if diff < tol:
            break

    return hub, auth


# ---------------------------------------------------------------------------
# Clustering Coefficient
# ---------------------------------------------------------------------------

def clustering_coefficient(graph, node=None):
    """Local clustering coefficient for a node, or all nodes if node is None.

    For undirected graphs only. Returns dict or single float.
    """
    if node is not None:
        return _local_clustering(graph, node)

    nodes = list(graph.nodes())
    return {v: _local_clustering(graph, v) for v in nodes}


def _local_clustering(graph, v):
    """Clustering coefficient for a single node."""
    neighbors = [n for n, _ in graph.neighbors(v)]
    k = len(neighbors)
    if k < 2:
        return 0.0

    # Count edges between neighbors
    neighbor_set = set(neighbors)
    triangles = 0
    for i, u in enumerate(neighbors):
        for w, _ in graph.neighbors(u):
            if w in neighbor_set and w != u:
                triangles += 1

    # Each triangle counted twice in undirected (u->w and w->u)
    if not graph.directed:
        triangles //= 2

    return triangles / (k * (k - 1) / 2.0) if not graph.directed else triangles / (k * (k - 1))


def average_clustering(graph):
    """Average clustering coefficient over all nodes."""
    cc = clustering_coefficient(graph)
    if not cc:
        return 0.0
    return sum(cc.values()) / len(cc)


def transitivity(graph):
    """Global clustering coefficient (transitivity): 3*triangles / triads."""
    nodes = list(graph.nodes())
    triangles = 0
    triads = 0

    for v in nodes:
        neighbors = [n for n, _ in graph.neighbors(v)]
        k = len(neighbors)
        triads += k * (k - 1)  # ordered pairs of neighbors

        neighbor_set = set(neighbors)
        for u in neighbors:
            for w, _ in graph.neighbors(u):
                if w in neighbor_set and w != v:
                    triangles += 1

    if triads == 0:
        return 0.0
    return triangles / triads


# ---------------------------------------------------------------------------
# K-Core Decomposition
# ---------------------------------------------------------------------------

def k_core_decomposition(graph):
    """Compute coreness of each node using Batagelj-Zaversnik peeling.

    Returns dict node -> core_number.
    """
    nodes = list(graph.nodes())
    if not nodes:
        return {}

    # Build neighbor sets
    adj = defaultdict(set)
    for v in nodes:
        for u, _ in graph.neighbors(v):
            adj[v].add(u)

    # Compute degrees
    deg = {v: len(adj[v]) for v in nodes}
    core = {}
    remaining = set(nodes)

    while remaining:
        # Pick minimum-degree node
        v = min(remaining, key=lambda x: deg[x])
        k = deg[v]
        core[v] = k
        remaining.remove(v)

        # Reduce degree of neighbors still remaining
        for u in adj[v]:
            if u in remaining and deg[u] > k:
                deg[u] -= 1

    return core


def k_core(graph, k):
    """Return subgraph induced by the k-core (nodes with coreness >= k)."""
    cores = k_core_decomposition(graph)
    core_nodes = {v for v, c in cores.items() if c >= k}

    sub = Graph(directed=graph.directed)
    for v in core_nodes:
        sub.add_node(v)
    for v in core_nodes:
        for u, w in graph.neighbors(v):
            if u in core_nodes:
                if graph.directed:
                    sub.add_edge(v, u, w)
                elif v < u or (not isinstance(v, (int, float, str))):
                    sub.add_edge(v, u, w)
    return sub


# ---------------------------------------------------------------------------
# Label Propagation Community Detection
# ---------------------------------------------------------------------------

def label_propagation(graph, max_iter=100, seed=None):
    """Label propagation algorithm for community detection.

    Returns dict node -> community_label.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    labels = {node: i for i, node in enumerate(nodes)}

    for _ in range(max_iter):
        order = list(nodes)
        rng.shuffle(order)
        changed = False
        for node in order:
            neighbors = [n for n, _ in graph.neighbors(node)]
            if not neighbors:
                continue

            # Count neighbor labels
            label_counts = defaultdict(int)
            for n in neighbors:
                label_counts[labels[n]] += 1

            max_count = max(label_counts.values())
            best_labels = [l for l, c in label_counts.items() if c == max_count]
            new_label = rng.choice(best_labels)

            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    return labels


def communities_from_labels(labels):
    """Convert label dict to list of sets (communities)."""
    communities = defaultdict(set)
    for node, label in labels.items():
        communities[label].add(node)
    return list(communities.values())


# ---------------------------------------------------------------------------
# Louvain Community Detection (Modularity optimization)
# ---------------------------------------------------------------------------

def modularity(graph, communities):
    """Compute modularity Q for a given partition.

    communities: list of sets of nodes.
    Works for undirected graphs.
    """
    m = 0.0  # total edge weight
    for u in graph.nodes():
        for v, w in graph.neighbors(u):
            m += w
    if graph.directed:
        pass  # m is already total weight of all directed edges
    else:
        m /= 2.0  # each edge counted twice in undirected

    if m == 0:
        return 0.0

    # Build community mapping
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    # Compute degree sums per community
    q = 0.0
    for u in graph.nodes():
        for v, w in graph.neighbors(u):
            if node_to_comm.get(u) == node_to_comm.get(v):
                q += w
    if not graph.directed:
        q /= 2.0  # each internal edge counted twice

    # Subtract expected
    # Sum of (degree of community / 2m)^2
    comm_degree = defaultdict(float)
    for node in graph.nodes():
        deg = sum(w for _, w in graph.neighbors(node))
        comm_degree[node_to_comm.get(node, -1)] += deg

    expected = 0.0
    for c, d in comm_degree.items():
        if not graph.directed:
            expected += (d / (2.0 * m)) ** 2
        else:
            expected += (d / m) ** 2

    if not graph.directed:
        return q / m - expected
    else:
        return q / m - expected


def louvain(graph, seed=None):
    """Louvain method for community detection (modularity optimization).

    Returns list of sets (communities).
    Works on undirected graphs.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    if not nodes:
        return []

    # Phase 1: Each node in its own community
    node_to_comm = {node: node for node in nodes}

    # Build weighted adjacency
    adj_weight = defaultdict(lambda: defaultdict(float))
    for u in nodes:
        for v, w in graph.neighbors(u):
            adj_weight[u][v] += w

    # Total edge weight
    m = 0.0
    for u in nodes:
        for v, w in graph.neighbors(u):
            m += w
    if not graph.directed:
        m /= 2.0

    if m == 0:
        return [set(nodes)]

    # Degree of each node (sum of edge weights)
    deg = {}
    for node in nodes:
        deg[node] = sum(w for _, w in graph.neighbors(node))

    improved = True
    while improved:
        improved = False
        order = list(nodes)
        rng.shuffle(order)

        for node in order:
            current_comm = node_to_comm[node]

            # Calculate weight to each neighboring community
            comm_weights = defaultdict(float)
            for neighbor, w in adj_weight[node].items():
                c = node_to_comm[neighbor]
                comm_weights[c] += w

            # Try moving node to each neighboring community
            best_comm = current_comm
            best_gain = 0.0

            # Sum of degrees in current community (excluding node)
            sum_in_current = sum(deg[v] for v in nodes if node_to_comm[v] == current_comm and v != node)
            ki_in_current = comm_weights.get(current_comm, 0.0)

            for c, ki_in in comm_weights.items():
                if c == current_comm:
                    continue
                sum_c = sum(deg[v] for v in nodes if node_to_comm[v] == c)

                # Modularity gain of moving node to community c
                gain = (ki_in - ki_in_current) / m
                gain -= deg[node] * (sum_c - sum_in_current) / (2.0 * m * m)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            if best_comm != current_comm:
                node_to_comm[node] = best_comm
                improved = True

    # Collect communities
    communities = defaultdict(set)
    for node, comm in node_to_comm.items():
        communities[comm].add(node)
    return list(communities.values())


# ---------------------------------------------------------------------------
# Connected Components (for analysis)
# ---------------------------------------------------------------------------

def connected_components(graph):
    """Find connected components. Returns list of sets."""
    visited = set()
    components = []
    for node in graph.nodes():
        if node not in visited:
            component = set()
            queue = deque([node])
            while queue:
                v = queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                component.add(v)
                for neighbor, _ in graph.neighbors(v):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
    return components


def strongly_connected_components(graph):
    """Tarjan's SCC algorithm for directed graphs. Returns list of sets."""
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

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
            scc = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in graph.nodes():
        if v not in index:
            strongconnect(v)

    return sccs


# ---------------------------------------------------------------------------
# Graph Density
# ---------------------------------------------------------------------------

def density(graph):
    """Graph density: edges / max_possible_edges."""
    n = graph.node_count()
    if n <= 1:
        return 0.0
    e = graph.edge_count()
    if graph.directed:
        return e / (n * (n - 1))
    return 2.0 * e / (n * (n - 1))


# ---------------------------------------------------------------------------
# Diameter and Eccentricity
# ---------------------------------------------------------------------------

def eccentricity(graph, node=None):
    """Eccentricity of a node (max distance to any reachable node).

    If node is None, returns dict for all nodes.
    """
    if node is not None:
        dist = _bfs_distances(graph, node)
        if len(dist) <= 1:
            return 0
        return max(dist.values())

    return {v: eccentricity(graph, v) for v in graph.nodes()}


def diameter(graph):
    """Diameter: max eccentricity over all nodes."""
    ecc = eccentricity(graph)
    if not ecc:
        return 0
    return max(ecc.values())


def radius(graph):
    """Radius: min eccentricity over all nodes."""
    ecc = eccentricity(graph)
    if not ecc:
        return 0
    return min(ecc.values())


def center(graph):
    """Center: set of nodes whose eccentricity equals the radius."""
    ecc = eccentricity(graph)
    if not ecc:
        return set()
    r = min(ecc.values())
    return {v for v, e in ecc.items() if e == r}


# ---------------------------------------------------------------------------
# Bridges and Articulation Points
# ---------------------------------------------------------------------------

def bridges(graph):
    """Find all bridges in an undirected graph.

    Returns list of (u, v) edge pairs.
    """
    disc = {}
    low = {}
    parent = {}
    timer = [0]
    result = []

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v, _ in graph.neighbors(u):
            if v not in disc:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    result.append((u, v))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in graph.nodes():
        if node not in disc:
            parent[node] = None
            dfs(node)

    return result


def articulation_points(graph):
    """Find all articulation points (cut vertices) in an undirected graph."""
    disc = {}
    low = {}
    parent = {}
    timer = [0]
    result = set()

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0

        for v, _ in graph.neighbors(u):
            if v not in disc:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # u is articulation point if:
                # 1) u is root and has 2+ children
                # 2) u is not root and low[v] >= disc[u]
                if parent[u] is None and children > 1:
                    result.add(u)
                if parent[u] is not None and low[v] >= disc[u]:
                    result.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in graph.nodes():
        if node not in disc:
            parent[node] = None
            dfs(node)

    return result


# ---------------------------------------------------------------------------
# Katz Centrality
# ---------------------------------------------------------------------------

def katz_centrality(graph, alpha=0.1, beta=1.0, max_iter=100, tol=1e-6):
    """Katz centrality: weighted sum of paths of all lengths.

    x_i = alpha * sum(A_ji * x_j) + beta
    alpha should be < 1/spectral_radius for convergence.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    x = {node: 0.0 for node in nodes}

    # Build in-neighbor adjacency
    in_nbrs = defaultdict(list)
    for node in nodes:
        for neighbor, _ in graph.neighbors(node):
            in_nbrs[neighbor].append(node)

    for _ in range(max_iter):
        new_x = {}
        for node in nodes:
            new_x[node] = alpha * sum(x.get(u, 0.0) for u in in_nbrs[node]) + beta

        # Check convergence
        diff = sum(abs(new_x[node] - x[node]) for node in nodes)
        x = new_x
        if diff < tol:
            break

    # Normalize
    norm = math.sqrt(sum(v * v for v in x.values()))
    if norm > 0:
        for node in nodes:
            x[node] /= norm

    return x


# ---------------------------------------------------------------------------
# Assortativity
# ---------------------------------------------------------------------------

def degree_assortativity(graph):
    """Degree assortativity coefficient (Pearson correlation of degrees at edge endpoints).

    Returns value in [-1, 1]. Positive means assortative (high-degree nodes
    connect to high-degree), negative means disassortative.
    """
    edges = graph.edges()
    if not edges:
        return 0.0

    # Get degree for each node
    deg = {}
    for node in graph.nodes():
        deg[node] = len(graph.neighbors(node))

    x_vals = []
    y_vals = []
    for u, v, _ in edges:
        x_vals.append(deg[u])
        y_vals.append(deg[v])
        if not graph.directed:
            x_vals.append(deg[v])
            y_vals.append(deg[u])

    n = len(x_vals)
    if n == 0:
        return 0.0

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    cov = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in x_vals)
    var_y = sum((y - mean_y) ** 2 for y in y_vals)

    if var_x == 0 or var_y == 0:
        return 0.0

    return cov / math.sqrt(var_x * var_y)


# ---------------------------------------------------------------------------
# Rich Club Coefficient
# ---------------------------------------------------------------------------

def rich_club_coefficient(graph, k=None):
    """Rich club coefficient phi(k) for each degree k.

    phi(k) = 2 * E_k / (N_k * (N_k - 1))
    where E_k = edges among nodes with degree > k, N_k = count of such nodes.

    If k is given, returns single float. Otherwise returns dict k -> phi(k).
    """
    deg = {}
    for node in graph.nodes():
        deg[node] = len(graph.neighbors(node))

    if k is not None:
        return _rich_club_at_k(graph, deg, k)

    max_deg = max(deg.values()) if deg else 0
    result = {}
    for ki in range(max_deg + 1):
        phi = _rich_club_at_k(graph, deg, ki)
        if phi is not None:
            result[ki] = phi
    return result


def _rich_club_at_k(graph, deg, k):
    """Compute rich club coefficient at degree threshold k."""
    rich_nodes = {v for v, d in deg.items() if d > k}
    nk = len(rich_nodes)
    if nk < 2:
        return 0.0

    # Count edges among rich nodes
    ek = 0
    for u in rich_nodes:
        for v, _ in graph.neighbors(u):
            if v in rich_nodes:
                ek += 1
    if not graph.directed:
        ek //= 2

    return 2.0 * ek / (nk * (nk - 1))


# ---------------------------------------------------------------------------
# Neighborhood Overlap (for link analysis)
# ---------------------------------------------------------------------------

def neighborhood_overlap(graph, u, v):
    """Jaccard similarity of neighborhoods of u and v."""
    nbrs_u = {n for n, _ in graph.neighbors(u)}
    nbrs_v = {n for n, _ in graph.neighbors(v)}
    # Remove u and v themselves
    nbrs_u.discard(v)
    nbrs_v.discard(u)

    union = nbrs_u | nbrs_v
    if not union:
        return 0.0
    return len(nbrs_u & nbrs_v) / len(union)


def common_neighbors(graph, u, v):
    """Count of common neighbors between u and v."""
    nbrs_u = {n for n, _ in graph.neighbors(u)}
    nbrs_v = {n for n, _ in graph.neighbors(v)}
    return len(nbrs_u & nbrs_v)


def adamic_adar(graph, u, v):
    """Adamic-Adar index: sum of 1/log(deg(w)) for common neighbors w."""
    nbrs_u = {n for n, _ in graph.neighbors(u)}
    nbrs_v = {n for n, _ in graph.neighbors(v)}
    common = nbrs_u & nbrs_v
    score = 0.0
    for w in common:
        deg_w = len(graph.neighbors(w))
        if deg_w > 1:
            score += 1.0 / math.log(deg_w)
    return score


# ---------------------------------------------------------------------------
# Reciprocity (directed graphs)
# ---------------------------------------------------------------------------

def reciprocity(graph):
    """Fraction of edges that are reciprocated in a directed graph."""
    if not graph.directed:
        return 1.0  # all edges are bidirectional in undirected

    edges_set = set()
    for u in graph.nodes():
        for v, _ in graph.neighbors(u):
            edges_set.add((u, v))

    if not edges_set:
        return 0.0

    reciprocated = sum(1 for u, v in edges_set if (v, u) in edges_set)
    return reciprocated / len(edges_set)
