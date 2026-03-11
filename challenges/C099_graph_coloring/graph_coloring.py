"""
C099: Graph Coloring

Algorithms for vertex coloring of graphs:
- Greedy coloring (with vertex ordering strategies)
- DSatur (saturation-degree based)
- Backtracking exact chromatic number
- k-colorability checking
- Color class analysis
- Graph operations (complement, subgraph, product)
- Special graph recognition (bipartite, chordal, perfect)
- Edge coloring via Vizing's theorem bound
- Fractional chromatic number approximation
"""


class Graph:
    """Adjacency-list undirected graph."""

    def __init__(self, n=0):
        self.n = n
        self.adj = [set() for _ in range(n)]

    def add_vertex(self):
        v = self.n
        self.n += 1
        self.adj.append(set())
        return v

    def add_edge(self, u, v):
        if u == v:
            raise ValueError("Self-loops not allowed")
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise ValueError(f"Vertex out of range: {u}, {v}")
        self.adj[u].add(v)
        self.adj[v].add(u)

    def has_edge(self, u, v):
        return v in self.adj[u]

    def degree(self, v):
        return len(self.adj[v])

    def max_degree(self):
        if self.n == 0:
            return 0
        return max(len(self.adj[v]) for v in range(self.n))

    def neighbors(self, v):
        return self.adj[v]

    def vertices(self):
        return range(self.n)

    def edge_count(self):
        return sum(len(self.adj[v]) for v in range(self.n)) // 2

    def complement(self):
        g = Graph(self.n)
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if v not in self.adj[u]:
                    g.add_edge(u, v)
        return g

    def subgraph(self, vertices):
        """Induced subgraph on given vertex set."""
        vset = set(vertices)
        vlist = sorted(vset)
        mapping = {v: i for i, v in enumerate(vlist)}
        g = Graph(len(vlist))
        for u in vlist:
            for v in self.adj[u]:
                if v in vset and v > u:
                    g.add_edge(mapping[u], mapping[v])
        return g, mapping

    def copy(self):
        g = Graph(self.n)
        for u in range(self.n):
            g.adj[u] = set(self.adj[u])
        return g


# --- Vertex Ordering Strategies ---

def order_natural(g):
    """Natural ordering: 0, 1, 2, ..."""
    return list(range(g.n))


def order_largest_first(g):
    """Largest-first: descending degree."""
    return sorted(range(g.n), key=lambda v: g.degree(v), reverse=True)


def order_smallest_last(g):
    """Smallest-last: repeatedly remove min-degree vertex."""
    degrees = [g.degree(v) for v in range(g.n)]
    remaining = set(range(g.n))
    order = []
    for _ in range(g.n):
        v = min(remaining, key=lambda x: degrees[x])
        remaining.remove(v)
        order.append(v)
        for u in g.adj[v]:
            if u in remaining:
                degrees[u] -= 1
    order.reverse()
    return order


def order_random(g, rng=None):
    """Random ordering."""
    import random
    r = rng or random
    verts = list(range(g.n))
    r.shuffle(verts)
    return verts


# --- Greedy Coloring ---

def greedy_color(g, order=None):
    """
    Greedy coloring with given vertex order.
    Returns dict mapping vertex -> color (0-indexed).
    """
    if g.n == 0:
        return {}
    if order is None:
        order = order_natural(g)

    coloring = {}
    for v in order:
        used = set()
        for u in g.adj[v]:
            if u in coloring:
                used.add(coloring[u])
        # Find smallest available color
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


# --- DSatur (Saturation Degree) ---

def dsatur_color(g):
    """
    DSatur algorithm: always color the vertex with highest saturation degree
    (number of distinct colors among neighbors). Ties broken by highest degree.
    Returns dict mapping vertex -> color.
    """
    if g.n == 0:
        return {}

    coloring = {}
    # saturation[v] = set of colors used by colored neighbors
    saturation = [set() for _ in range(g.n)]
    uncolored = set(range(g.n))

    for _ in range(g.n):
        # Pick vertex with max saturation, break ties by degree
        v = max(uncolored, key=lambda x: (len(saturation[x]), g.degree(x)))

        # Find smallest available color
        used = saturation[v]
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
        uncolored.remove(v)

        # Update saturation of neighbors
        for u in g.adj[v]:
            if u in uncolored:
                saturation[u].add(c)

    return coloring


# --- Exact Chromatic Number (Backtracking) ---

def chromatic_number(g, timeout=None):
    """
    Compute exact chromatic number using backtracking with pruning.
    Returns (chi, coloring) where chi is the chromatic number.
    Optional timeout in seconds.
    """
    if g.n == 0:
        return 0, {}

    # Check for isolated vertices / edges
    if g.edge_count() == 0:
        if g.n > 0:
            return 1, {v: 0 for v in range(g.n)}
        return 0, {}

    # Upper bound from DSatur
    best_coloring = dsatur_color(g)
    upper = num_colors_used(best_coloring)

    # Lower bound from clique
    lower = max_clique_greedy(g)

    if lower == upper:
        return upper, best_coloring

    import time
    start = time.time() if timeout else None

    # Try to color with k colors, for k from lower to upper-1
    for k in range(lower, upper):
        if timeout and time.time() - start > timeout:
            break
        result = _try_k_color(g, k, timeout, start)
        if result is not None:
            return k, result

    return upper, best_coloring


def _try_k_color(g, k, timeout=None, start_time=None):
    """Try to k-color the graph. Returns coloring or None."""
    import time

    # Order vertices by degree (high first) for better pruning
    order = order_largest_first(g)
    coloring = {}

    def backtrack(idx):
        if timeout and start_time:
            if time.time() - start_time > timeout:
                return False

        if idx == g.n:
            return True

        v = order[idx]
        used = set()
        for u in g.adj[v]:
            if u in coloring:
                used.add(coloring[u])

        for c in range(k):
            if c not in used:
                coloring[v] = c
                if backtrack(idx + 1):
                    return True
                del coloring[v]

        return False

    if backtrack(0):
        return dict(coloring)
    return None


def is_k_colorable(g, k):
    """Check if graph is k-colorable."""
    if k >= g.n:
        return True
    if k <= 0:
        return g.n == 0
    return _try_k_color(g, k) is not None


# --- Clique Finding (for lower bound) ---

def max_clique_greedy(g):
    """Greedy lower bound on clique number (and thus chromatic number)."""
    if g.n == 0:
        return 0

    best = 1
    # Try starting from each vertex
    for start in range(g.n):
        clique = {start}
        candidates = set(g.adj[start])
        while candidates:
            # Pick vertex with most connections to current clique
            v = max(candidates, key=lambda x: sum(1 for c in clique if x in g.adj[c]))
            if all(v in g.adj[c] for c in clique):
                clique.add(v)
                candidates = candidates & g.adj[v]
            else:
                candidates.discard(v)
        best = max(best, len(clique))
    return best


# --- Analysis Functions ---

def num_colors_used(coloring):
    """Number of distinct colors in a coloring."""
    if not coloring:
        return 0
    return len(set(coloring.values()))


def is_valid_coloring(g, coloring):
    """Check if coloring is proper (no adjacent vertices share color)."""
    for v in range(g.n):
        if v not in coloring:
            return False
        for u in g.adj[v]:
            if u in coloring and coloring[u] == coloring[v]:
                return False
    return True


def color_classes(coloring):
    """Return dict mapping color -> set of vertices."""
    classes = {}
    for v, c in coloring.items():
        if c not in classes:
            classes[c] = set()
        classes[c].add(v)
    return classes


def coloring_stats(g, coloring):
    """Return statistics about a coloring."""
    classes = color_classes(coloring)
    sizes = [len(s) for s in classes.values()]
    return {
        'num_colors': len(classes),
        'valid': is_valid_coloring(g, coloring),
        'class_sizes': sorted(sizes, reverse=True),
        'balance': min(sizes) / max(sizes) if sizes else 0,
    }


# --- Special Graph Recognition ---

def is_bipartite(g):
    """Check if graph is bipartite (2-colorable). Returns (bool, partition_or_odd_cycle)."""
    if g.n == 0:
        return True, (set(), set())

    color = [-1] * g.n
    part_a = set()
    part_b = set()

    for start in range(g.n):
        if color[start] != -1:
            continue
        color[start] = 0
        part_a.add(start)
        queue = [start]
        while queue:
            v = queue.pop(0)
            for u in g.adj[v]:
                if color[u] == -1:
                    color[u] = 1 - color[v]
                    if color[u] == 0:
                        part_a.add(u)
                    else:
                        part_b.add(u)
                    queue.append(u)
                elif color[u] == color[v]:
                    return False, None

    return True, (part_a, part_b)


def is_chordal(g):
    """
    Check if graph is chordal using perfect elimination ordering (PEO).
    A graph is chordal iff it has a PEO.
    Uses maximum cardinality search (MCS).
    """
    if g.n <= 3:
        return True

    # MCS: repeatedly pick vertex with most numbered neighbors
    order = []
    weight = [0] * g.n
    numbered = [False] * g.n

    for _ in range(g.n):
        # Pick unnumbered vertex with max weight
        v = -1
        best_w = -1
        for u in range(g.n):
            if not numbered[u] and weight[u] > best_w:
                best_w = weight[u]
                v = u
        order.append(v)
        numbered[v] = True
        for u in g.adj[v]:
            if not numbered[u]:
                weight[u] += 1

    order.reverse()
    pos = [0] * g.n
    for i, v in enumerate(order):
        pos[v] = i

    # Check PEO: for each v, its neighbors later in order must form a clique
    for v in order:
        later_nbrs = [u for u in g.adj[v] if pos[u] > pos[v]]
        for i in range(len(later_nbrs)):
            for j in range(i + 1, len(later_nbrs)):
                if later_nbrs[j] not in g.adj[later_nbrs[i]]:
                    return False
    return True


def is_complete(g):
    """Check if graph is complete (all pairs connected)."""
    if g.n <= 1:
        return True
    return all(g.degree(v) == g.n - 1 for v in range(g.n))


def is_cycle(g):
    """Check if graph is a single cycle."""
    if g.n < 3:
        return False
    if not all(g.degree(v) == 2 for v in range(g.n)):
        return False
    # Check connectivity
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        v = queue.pop()
        for u in g.adj[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)
    return len(visited) == g.n


# --- Graph Constructors ---

def complete_graph(n):
    """K_n complete graph."""
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            g.add_edge(u, v)
    return g


def cycle_graph(n):
    """C_n cycle graph."""
    if n < 3:
        raise ValueError("Cycle needs at least 3 vertices")
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def complete_bipartite(m, n):
    """K_{m,n} complete bipartite graph."""
    g = Graph(m + n)
    for u in range(m):
        for v in range(m, m + n):
            g.add_edge(u, v)
    return g


def petersen_graph():
    """Petersen graph (chi=3, 10 vertices, 15 edges)."""
    g = Graph(10)
    # Outer cycle
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    # Inner pentagram
    for i in range(5):
        g.add_edge(5 + i, 5 + (i + 2) % 5)
    # Spokes
    for i in range(5):
        g.add_edge(i, 5 + i)
    return g


def wheel_graph(n):
    """W_n wheel graph (hub + n-cycle)."""
    if n < 3:
        raise ValueError("Wheel needs at least 3 rim vertices")
    g = Graph(n + 1)
    # Hub is vertex 0
    for i in range(1, n + 1):
        g.add_edge(0, i)
    # Rim cycle
    for i in range(1, n + 1):
        g.add_edge(i, (i % n) + 1)
    return g


def crown_graph(n):
    """Crown graph S_n^0: complete bipartite K_{n,n} minus perfect matching."""
    if n < 2:
        raise ValueError("Crown needs n >= 2")
    g = Graph(2 * n)
    for u in range(n):
        for v in range(n, 2 * n):
            if v - n != u:  # Skip the matching edge
                g.add_edge(u, v)
    return g


def grid_graph(rows, cols):
    """Grid graph (rows x cols)."""
    g = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1)
            if r + 1 < rows:
                g.add_edge(v, v + cols)
    return g


# --- Edge Coloring ---

def edge_chromatic_number_bounds(g):
    """
    Vizing's theorem: chi'(G) is either Delta or Delta+1.
    Returns (lower, upper) = (Delta, Delta+1).
    For bipartite graphs, chi'(G) = Delta exactly (Konig's theorem).
    """
    delta = g.max_degree()
    bip, _ = is_bipartite(g)
    if bip:
        return delta, delta
    return delta, delta + 1


def edge_color_greedy(g):
    """
    Greedy edge coloring. Returns dict mapping (u,v) -> color where u < v.
    """
    coloring = {}
    for u in range(g.n):
        for v in g.adj[u]:
            if v > u:
                used_u = set()
                used_v = set()
                for w in g.adj[u]:
                    key = (min(u, w), max(u, w))
                    if key in coloring:
                        used_u.add(coloring[key])
                for w in g.adj[v]:
                    key = (min(v, w), max(v, w))
                    if key in coloring:
                        used_v.add(coloring[key])
                used = used_u | used_v
                c = 0
                while c in used:
                    c += 1
                coloring[(u, v)] = c
    return coloring


# --- Fractional Chromatic Number ---

def fractional_chromatic_bound(g):
    """
    Bounds on fractional chromatic number.
    Lower: n / alpha(G) where alpha is independence number (approximated).
    Upper: chi(G) from DSatur.
    Returns (lower_bound, upper_bound).
    """
    if g.n == 0:
        return 0, 0

    chi_upper = num_colors_used(dsatur_color(g))

    # Approximate independence number
    alpha = _max_independent_set_greedy(g)

    lower = g.n / alpha if alpha > 0 else g.n
    return lower, chi_upper


def _max_independent_set_greedy(g):
    """Greedy max independent set size."""
    if g.n == 0:
        return 0

    remaining = set(range(g.n))
    indep = set()

    while remaining:
        # Pick vertex with min degree in remaining subgraph
        v = min(remaining, key=lambda x: len(g.adj[x] & remaining))
        indep.add(v)
        remaining -= {v} | (g.adj[v] & remaining)

    return len(indep)


# --- Interval Graph Coloring (Optimal for interval graphs) ---

def interval_graph_color(intervals):
    """
    Color intervals optimally. Each interval is (start, end).
    Returns (coloring, num_colors) where coloring maps index -> color.
    Uses sweep-line algorithm.
    """
    if not intervals:
        return {}, 0

    events = []
    for i, (s, e) in enumerate(intervals):
        events.append((s, 0, i))   # 0 = start
        events.append((e, 1, i))   # 1 = end
    events.sort()

    coloring = {}
    available = []  # min-heap of freed colors
    next_color = 0

    import heapq
    for _, etype, idx in events:
        if etype == 0:  # interval starts
            if available:
                c = heapq.heappop(available)
            else:
                c = next_color
                next_color += 1
            coloring[idx] = c
        else:  # interval ends
            heapq.heappush(available, coloring[idx])

    return coloring, next_color


# --- Map Coloring (Planar graph 4-coloring heuristic) ---

def map_color_greedy(g):
    """
    Four-color heuristic for planar graphs using smallest-last ordering.
    Not guaranteed to use 4 colors for non-planar graphs.
    """
    order = order_smallest_last(g)
    return greedy_color(g, order)


# --- Register Allocation (Interval Graph Coloring Application) ---

def register_allocate(live_ranges):
    """
    Allocate registers given live ranges of variables.
    live_ranges: dict mapping var_name -> (start, end)
    Returns (allocation, num_registers) mapping var_name -> register_id.
    """
    if not live_ranges:
        return {}, 0

    names = list(live_ranges.keys())
    intervals = [live_ranges[n] for n in names]
    coloring, num_colors = interval_graph_color(intervals)

    allocation = {names[i]: coloring[i] for i in range(len(names))}
    return allocation, num_colors


# --- Welsh-Powell Algorithm ---

def welsh_powell_color(g):
    """
    Welsh-Powell algorithm: sort by descending degree, greedy color.
    Equivalent to greedy with largest-first ordering.
    """
    return greedy_color(g, order_largest_first(g))


# --- Connected Components ---

def connected_components(g):
    """Return list of vertex sets, one per component."""
    visited = set()
    components = []
    for v in range(g.n):
        if v not in visited:
            comp = set()
            queue = [v]
            visited.add(v)
            while queue:
                u = queue.pop()
                comp.add(u)
                for w in g.adj[u]:
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
            components.append(comp)
    return components


def chromatic_polynomial_tree(n):
    """
    Chromatic polynomial of a tree with n vertices: k * (k-1)^(n-1).
    Returns coefficients as callable.
    """
    def P(k):
        if n == 0:
            return 1
        return k * (k - 1) ** (n - 1)
    return P


def chromatic_polynomial_cycle(n):
    """
    Chromatic polynomial of C_n: (k-1)^n + (-1)^n * (k-1).
    """
    def P(k):
        return (k - 1) ** n + ((-1) ** n) * (k - 1)
    return P


def chromatic_polynomial_complete(n):
    """
    Chromatic polynomial of K_n: k * (k-1) * (k-2) * ... * (k-n+1).
    """
    def P(k):
        result = 1
        for i in range(n):
            result *= (k - i)
        return result
    return P
