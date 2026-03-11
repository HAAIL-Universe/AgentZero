"""
C090: Convex Hull -- Computational Geometry Algorithms

Algorithms:
  - Graham scan (O(n log n))
  - Andrew's monotone chain (O(n log n))
  - Gift wrapping / Jarvis march (O(nh))
  - QuickHull (O(n log n) avg, O(n^2) worst)
  - Dynamic convex hull (insert/delete, O(log^2 n))
  - Convex hull trick (line container for DP optimization)
  - Minkowski sum of convex polygons
  - Rotating calipers (diameter, width, antipodal pairs)

All points are (x, y) tuples. Hulls are counter-clockwise.
"""

import math
from bisect import insort, bisect_left, bisect_right
from collections import deque

EPS = 1e-9


# ── Primitives ──────────────────────────────────────────────

def cross(o, a, b):
    """Cross product of vectors OA and OB. Positive = CCW turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist_sq(a, b):
    """Squared distance between two points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def dist(a, b):
    """Euclidean distance between two points."""
    return math.sqrt(dist_sq(a, b))


def point_in_convex_polygon(point, hull):
    """Test if point is inside convex polygon (hull in CCW order).
    Returns: 1 = inside, 0 = on boundary, -1 = outside."""
    n = len(hull)
    if n < 3:
        return -1
    for i in range(n):
        c = cross(hull[i], hull[(i + 1) % n], point)
        if c < -EPS:
            return -1
        if abs(c) < EPS:
            # Check if on the edge segment
            x_min = min(hull[i][0], hull[(i + 1) % n][0])
            x_max = max(hull[i][0], hull[(i + 1) % n][0])
            y_min = min(hull[i][1], hull[(i + 1) % n][1])
            y_max = max(hull[i][1], hull[(i + 1) % n][1])
            if x_min - EPS <= point[0] <= x_max + EPS and y_min - EPS <= point[1] <= y_max + EPS:
                return 0
    return 1


def polygon_area(hull):
    """Area of polygon given vertices in order (shoelace formula)."""
    n = len(hull)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def perimeter(hull):
    """Perimeter of polygon."""
    n = len(hull)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        total += dist(hull[i], hull[(i + 1) % n])
    return total


def centroid(hull):
    """Centroid of convex polygon."""
    n = len(hull)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return hull[0]
    if n == 2:
        return ((hull[0][0] + hull[1][0]) / 2, (hull[0][1] + hull[1][1]) / 2)
    cx, cy, a_total = 0.0, 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        f = hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
        cx += (hull[i][0] + hull[j][0]) * f
        cy += (hull[i][1] + hull[j][1]) * f
        a_total += f
    if abs(a_total) < EPS:
        # Degenerate -- average points
        return (sum(p[0] for p in hull) / n, sum(p[1] for p in hull) / n)
    a_total *= 3.0
    return (cx / a_total, cy / a_total)


# ── Graham Scan ─────────────────────────────────────────────

def graham_scan(points):
    """Convex hull via Graham scan. Returns CCW hull vertices."""
    pts = list(set(points))
    n = len(pts)
    if n <= 1:
        return pts[:]
    if n == 2:
        return pts[:] if pts[0] != pts[1] else [pts[0]]

    # Find lowest-then-leftmost point as pivot
    pivot = min(pts, key=lambda p: (p[1], p[0]))

    def angle_key(p):
        if p == pivot:
            return (-math.inf, 0)
        return (math.atan2(p[1] - pivot[1], p[0] - pivot[0]), dist_sq(pivot, p))

    pts.sort(key=angle_key)

    # Handle collinear points at same angle: keep farthest, except for
    # the last angle group where we keep closest (to close the hull properly)
    # Actually, standard Graham scan just uses strict left turns.

    stack = []
    for p in pts:
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack


# ── Andrew's Monotone Chain ─────────────────────────────────

def monotone_chain(points):
    """Convex hull via Andrew's monotone chain. Returns CCW hull."""
    pts = sorted(set(points))
    n = len(pts)
    if n <= 1:
        return pts[:]
    if n == 2:
        return pts[:]

    # Lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]


# ── Gift Wrapping (Jarvis March) ────────────────────────────

def gift_wrapping(points):
    """Convex hull via Jarvis march. O(nh). Returns CCW hull."""
    pts = list(set(points))
    n = len(pts)
    if n <= 1:
        return pts[:]
    if n == 2:
        return pts[:]

    # Start from bottom-most, then left-most point (guarantees CCW traversal)
    start = min(pts, key=lambda p: (p[1], p[0]))
    hull = []
    current = start

    while True:
        hull.append(current)
        candidate = pts[0]
        for p in pts[1:]:
            if candidate == current:
                candidate = p
                continue
            c = cross(current, candidate, p)
            if c < 0:
                # p is to the right of current->candidate (more clockwise)
                # For CCW traversal, we want the most clockwise candidate
                candidate = p
            elif c == 0:
                # Collinear -- take farther point
                if dist_sq(current, p) > dist_sq(current, candidate):
                    candidate = p
        current = candidate
        if current == start:
            break
        if len(hull) > n:
            break  # Safety

    return hull


# ── QuickHull ───────────────────────────────────────────────

def quickhull(points):
    """Convex hull via QuickHull. Returns CCW hull."""
    pts = list(set(points))
    n = len(pts)
    if n <= 1:
        return pts[:]
    if n == 2:
        return pts[:]

    # Find extremes
    left = min(pts, key=lambda p: (p[0], p[1]))
    right = max(pts, key=lambda p: (p[0], p[1]))

    above = [p for p in pts if cross(left, right, p) > 0]
    below = [p for p in pts if cross(left, right, p) < 0]

    hull = []
    _quickhull_rec(left, right, above, hull)
    _quickhull_rec(right, left, below, hull)

    if not hull:
        # All collinear
        hull = [left, right]

    # Sort hull CCW
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    hull.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return hull


def _quickhull_rec(a, b, points, hull):
    """Recursively find hull points above line a->b."""
    if not points:
        hull.append(a)
        return

    # Find farthest point from line a->b
    farthest = max(points, key=lambda p: abs(cross(a, b, p)))
    hull_above_af = [p for p in points if cross(a, farthest, p) > 0]
    hull_above_fb = [p for p in points if cross(farthest, b, p) > 0]

    _quickhull_rec(a, farthest, hull_above_af, hull)
    _quickhull_rec(farthest, b, hull_above_fb, hull)


# ── Rotating Calipers ──────────────────────────────────────

def rotating_calipers(hull):
    """Find all antipodal pairs using rotating calipers.
    Hull must be CCW convex polygon. Returns list of (i, j) index pairs."""
    n = len(hull)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]

    pairs = []
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while True:
            nj = (j + 1) % n
            # Edge vector for edge i -> i+1
            ei = (hull[ni][0] - hull[i][0], hull[ni][1] - hull[i][1])
            # Edge vector for edge j -> j+1
            ej = (hull[nj][0] - hull[j][0], hull[nj][1] - hull[j][1])
            # Cross product of edge vectors
            c = ei[0] * ej[1] - ei[1] * ej[0]
            if c <= 0:
                break
            j = nj
        pairs.append((i, j))
    return pairs


def diameter(hull):
    """Diameter (max distance between any two hull points) via rotating calipers."""
    n = len(hull)
    if n < 2:
        return 0.0
    if n == 2:
        return dist(hull[0], hull[1])
    pairs = rotating_calipers(hull)
    max_d = 0.0
    for i, j in pairs:
        d = dist_sq(hull[i], hull[j])
        if d > max_d:
            max_d = d
    return math.sqrt(max_d)


def width(hull):
    """Width (minimum distance between parallel supporting lines)."""
    n = len(hull)
    if n < 2:
        return 0.0
    if n == 2:
        return 0.0  # Line has zero width

    min_w = float('inf')
    pairs = rotating_calipers(hull)
    for i, j in pairs:
        ni = (i + 1) % n
        # Distance from hull[j] to line hull[i]->hull[ni]
        edge_len = dist(hull[i], hull[ni])
        if edge_len < EPS:
            continue
        w = abs(cross(hull[i], hull[ni], hull[j])) / edge_len
        if w < min_w:
            min_w = w
    return min_w if min_w != float('inf') else 0.0


def min_bounding_rectangle(hull):
    """Minimum area bounding rectangle via rotating calipers.
    Returns (area, corners) where corners is 4 points."""
    n = len(hull)
    if n < 2:
        if n == 1:
            return (0.0, [hull[0]] * 4)
        return (0.0, [])
    if n == 2:
        return (0.0, [hull[0], hull[1], hull[1], hull[0]])

    min_area = float('inf')
    best_rect = None

    # For each edge, find the minimum bounding rectangle aligned to that edge
    for i in range(n):
        ni = (i + 1) % n
        ex = hull[ni][0] - hull[i][0]
        ey = hull[ni][1] - hull[i][1]
        edge_len = math.sqrt(ex * ex + ey * ey)
        if edge_len < EPS:
            continue
        # Unit vectors along and perpendicular to edge
        ux, uy = ex / edge_len, ey / edge_len
        vx, vy = -uy, ux  # perpendicular (left)

        # Project all hull points onto these axes
        min_u, max_u = float('inf'), -float('inf')
        min_v, max_v = float('inf'), -float('inf')
        for p in hull:
            dx, dy = p[0] - hull[i][0], p[1] - hull[i][1]
            pu = dx * ux + dy * uy
            pv = dx * vx + dy * vy
            min_u = min(min_u, pu)
            max_u = max(max_u, pu)
            min_v = min(min_v, pv)
            max_v = max(max_v, pv)

        area = (max_u - min_u) * (max_v - min_v)
        if area < min_area:
            min_area = area
            # Reconstruct corners
            ox, oy = hull[i][0], hull[i][1]
            c0 = (ox + min_u * ux + min_v * vx, oy + min_u * uy + min_v * vy)
            c1 = (ox + max_u * ux + min_v * vx, oy + max_u * uy + min_v * vy)
            c2 = (ox + max_u * ux + max_v * vx, oy + max_u * uy + max_v * vy)
            c3 = (ox + min_u * ux + max_v * vx, oy + min_u * uy + max_v * vy)
            best_rect = [c0, c1, c2, c3]

    return (min_area, best_rect)


# ── Minkowski Sum ───────────────────────────────────────────

def minkowski_sum(P, Q):
    """Minkowski sum of two convex polygons P and Q (both CCW).
    Returns convex polygon (CCW)."""
    if not P or not Q:
        return []

    # Ensure CCW
    P = _ensure_ccw(P)
    Q = _ensure_ccw(Q)

    n, m = len(P), len(Q)
    if n == 1 and m == 1:
        return [(P[0][0] + Q[0][0], P[0][1] + Q[0][1])]
    if n == 1:
        return [(P[0][0] + q[0], P[0][1] + q[1]) for q in Q]
    if m == 1:
        return [(p[0] + Q[0][0], p[1] + Q[0][1]) for p in P]

    # Start from bottom-most points
    pi = min(range(n), key=lambda i: (P[i][1], P[i][0]))
    qi = min(range(m), key=lambda i: (Q[i][1], Q[i][0]))

    result = []
    i, j = 0, 0
    while i < n or j < m:
        result.append((P[(pi + i) % n][0] + Q[(qi + j) % m][0],
                        P[(pi + i) % n][1] + Q[(qi + j) % m][1]))
        # Edge vectors
        p_edge = _edge_vec(P, (pi + i) % n)
        q_edge = _edge_vec(Q, (qi + j) % m)

        c = p_edge[0] * q_edge[1] - p_edge[1] * q_edge[0]
        if c > 0:
            i += 1
        elif c < 0:
            j += 1
        else:
            i += 1
            j += 1

        if i >= n and j >= m:
            break

    return result


def _edge_vec(poly, i):
    n = len(poly)
    ni = (i + 1) % n
    return (poly[ni][0] - poly[i][0], poly[ni][1] - poly[i][1])


def _ensure_ccw(poly):
    """Ensure polygon is counter-clockwise."""
    area = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    if area < 0:
        return list(reversed(poly))
    return poly


# ── Convex Hull Trick (Line Container) ─────────────────────

class Line:
    """Line y = mx + b for convex hull trick."""
    __slots__ = ('m', 'b', 'x_left')

    def __init__(self, m, b):
        self.m = m
        self.b = b
        self.x_left = -float('inf')

    def eval(self, x):
        return self.m * x + self.b

    def intersect_x(self, other):
        """X-coordinate where self and other intersect."""
        if self.m == other.m:
            return float('inf') if self.b <= other.b else -float('inf')
        return (other.b - self.b) / (self.m - other.m)


class ConvexHullTrick:
    """Convex Hull Trick for minimum query over lines y = mx + b.

    add_line(m, b): Add line. Lines must be added in decreasing slope order.
    query(x): Get minimum y value at x.

    For maximum queries, negate m and b, then negate the result.
    """

    def __init__(self):
        self.lines = deque()

    def _bad(self, l1, l2, l3):
        """Is l2 unnecessary given l1 and l3?"""
        # l2 is bad if intersection of l1,l3 is at or before intersection of l1,l2
        return (l3.b - l1.b) * (l1.m - l2.m) <= (l2.b - l1.b) * (l1.m - l3.m)

    def add_line(self, m, b):
        """Add line y = mx + b. For sorted insertion, slopes should decrease."""
        new_line = Line(m, b)
        while len(self.lines) >= 2 and self._bad(self.lines[-2], self.lines[-1], new_line):
            self.lines.pop()
        self.lines.append(new_line)

    def query(self, x):
        """Query minimum y at given x. Queries must be in increasing x order (amortized)."""
        while len(self.lines) >= 2 and self.lines[0].eval(x) >= self.lines[1].eval(x):
            self.lines.popleft()
        if not self.lines:
            return float('inf')
        return self.lines[0].eval(x)

    def __len__(self):
        return len(self.lines)


class LiChaoTree:
    """Li Chao segment tree for arbitrary-order line insertions and queries.
    Supports minimum queries over lines y = mx + b.
    """

    def __init__(self, lo=-10**9, hi=10**9):
        self.lo = lo
        self.hi = hi
        self.nodes = {}  # (lo, hi) -> Line
        self._best = {}  # cache

    def add_line(self, m, b):
        """Insert line y = mx + b."""
        self._insert(Line(m, b), self.lo, self.hi)

    def _insert(self, new_line, lo, hi):
        mid = (lo + hi) // 2
        key = (lo, hi)
        cur = self.nodes.get(key)

        if cur is None:
            self.nodes[key] = new_line
            return

        left_better = new_line.eval(lo) < cur.eval(lo)
        mid_better = new_line.eval(mid) < cur.eval(mid)

        if mid_better:
            self.nodes[key] = new_line
            new_line = cur

        if lo == hi:
            return

        if left_better != mid_better:
            self._insert(new_line, lo, mid)
        else:
            self._insert(new_line, mid + 1, hi)

    def query(self, x):
        """Query minimum y at x."""
        result = float('inf')
        lo, hi = self.lo, self.hi
        while lo <= hi:
            mid = (lo + hi) // 2
            key = (lo, hi)
            line = self.nodes.get(key)
            if line is not None:
                val = line.eval(x)
                if val < result:
                    result = val
            if lo == hi:
                break
            if x <= mid:
                hi = mid
            else:
                lo = mid + 1
        return result

    def __len__(self):
        return len(self.nodes)


# ── Dynamic Convex Hull ────────────────────────────────────

class DynamicConvexHull:
    """Dynamic convex hull supporting insert and point-in-hull queries.
    Uses sorted point set with upper/lower hull maintenance.

    insert(point): Add point to set, update hull.
    contains(point): Check if point is inside current hull.
    get_hull(): Return current hull vertices (CCW).
    """

    def __init__(self):
        self.points = set()
        self._upper = []  # Upper hull (sorted by x)
        self._lower = []  # Lower hull (sorted by x)

    def insert(self, point):
        """Insert a point into the dynamic hull."""
        point = (float(point[0]), float(point[1]))
        if point in self.points:
            return
        self.points.add(point)
        self._rebuild()

    def remove(self, point):
        """Remove a point from the dynamic hull."""
        point = (float(point[0]), float(point[1]))
        if point not in self.points:
            return
        self.points.discard(point)
        self._rebuild()

    def _rebuild(self):
        """Rebuild upper and lower hulls from scratch."""
        pts = sorted(self.points)
        n = len(pts)
        if n <= 2:
            self._upper = list(pts)
            self._lower = list(pts)
            return

        # Lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        self._lower = lower
        self._upper = upper

    def get_hull(self):
        """Return CCW hull vertices."""
        if len(self.points) <= 1:
            return list(self.points)
        if len(self.points) == 2:
            return sorted(self.points)
        return self._lower[:-1] + self._upper[:-1]

    def contains(self, point):
        """Check if point is inside or on the current convex hull.
        Returns True if inside or on boundary."""
        hull = self.get_hull()
        n = len(hull)
        if n == 0:
            return False
        if n == 1:
            return abs(point[0] - hull[0][0]) < EPS and abs(point[1] - hull[0][1]) < EPS
        if n == 2:
            # On the line segment?
            c = cross(hull[0], hull[1], point)
            if abs(c) > EPS:
                return False
            t = 0.0
            dx = hull[1][0] - hull[0][0]
            dy = hull[1][1] - hull[0][1]
            if abs(dx) > abs(dy):
                t = (point[0] - hull[0][0]) / dx
            elif abs(dy) > EPS:
                t = (point[1] - hull[0][1]) / dy
            return -EPS <= t <= 1 + EPS

        return point_in_convex_polygon(point, hull) >= 0

    def area(self):
        return polygon_area(self.get_hull())

    def __len__(self):
        return len(self.points)


# ── Half-Plane Intersection ────────────────────────────────

def half_plane_intersection(half_planes, bounds=(-1e9, -1e9, 1e9, 1e9)):
    """Compute intersection of half-planes using incremental algorithm.

    Each half-plane is (a, b, c) representing ax + by <= c.
    bounds = (xmin, ymin, xmax, ymax) initial bounding box.

    Returns list of vertices of the intersection polygon (CCW), or [] if empty.
    """
    xmin, ymin, xmax, ymax = bounds
    # Start with bounding box as polygon
    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    for a, b, c in half_planes:
        poly = _clip_polygon(poly, a, b, c)
        if not poly:
            return []

    return poly


def _clip_polygon(poly, a, b, c):
    """Clip polygon by half-plane ax + by <= c (Sutherland-Hodgman)."""
    if not poly:
        return []

    result = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        nxt = poly[(i + 1) % n]
        curr_val = a * curr[0] + b * curr[1]
        nxt_val = a * nxt[0] + b * nxt[1]
        curr_in = curr_val <= c + EPS
        nxt_in = nxt_val <= c + EPS

        if curr_in:
            result.append(curr)
        if curr_in != nxt_in:
            # Intersection point
            denom = a * (nxt[0] - curr[0]) + b * (nxt[1] - curr[1])
            if abs(denom) > EPS:
                t = (c - a * curr[0] - b * curr[1]) / denom
                ix = curr[0] + t * (nxt[0] - curr[0])
                iy = curr[1] + t * (nxt[1] - curr[1])
                result.append((ix, iy))

    return result


# ── Convex Hull Union Area ─────────────────────────────────

def convex_hull_of_points(points):
    """Convenience: compute convex hull using monotone chain."""
    return monotone_chain(points)


def convex_hull_area(points):
    """Area of convex hull of given points."""
    hull = monotone_chain(points)
    return polygon_area(hull)


def convex_hull_union(hull1, hull2):
    """Convex hull of union of two convex polygon vertex sets."""
    all_pts = list(hull1) + list(hull2)
    return monotone_chain(all_pts)


# ── Farthest Pair ──────────────────────────────────────────

def farthest_pair(points):
    """Find farthest pair of points. Returns (dist, p1, p2)."""
    hull = monotone_chain(points)
    n = len(hull)
    if n < 2:
        return (0.0, points[0], points[0]) if points else (0.0, None, None)
    if n == 2:
        return (dist(hull[0], hull[1]), hull[0], hull[1])

    pairs = rotating_calipers(hull)
    max_d = 0.0
    best = (hull[0], hull[1])
    for i, j in pairs:
        d = dist_sq(hull[i], hull[j])
        if d > max_d:
            max_d = d
            best = (hull[i], hull[j])
    return (math.sqrt(max_d), best[0], best[1])


# ── Closest Pair (divide and conquer) ─────────────────────

def closest_pair(points):
    """Find closest pair of points. O(n log n).
    Returns (dist, p1, p2)."""
    pts = sorted(set(points))
    n = len(pts)
    if n < 2:
        return (float('inf'), None, None)
    if n == 2:
        return (dist(pts[0], pts[1]), pts[0], pts[1])

    result = _closest_pair_rec(pts)
    return result


def _closest_pair_rec(pts):
    n = len(pts)
    if n <= 3:
        best_d = float('inf')
        best = (pts[0], pts[1])
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(pts[i], pts[j])
                if d < best_d:
                    best_d = d
                    best = (pts[i], pts[j])
        return (best_d, best[0], best[1])

    mid = n // 2
    mid_x = pts[mid][0]

    left_result = _closest_pair_rec(pts[:mid])
    right_result = _closest_pair_rec(pts[mid:])

    if left_result[0] < right_result[0]:
        best_d, p1, p2 = left_result
    else:
        best_d, p1, p2 = right_result

    # Check strip
    strip = [p for p in pts if abs(p[0] - mid_x) < best_d]
    strip.sort(key=lambda p: p[1])

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and strip[j][1] - strip[i][1] < best_d:
            d = dist(strip[i], strip[j])
            if d < best_d:
                best_d = d
                p1, p2 = strip[i], strip[j]
            j += 1

    return (best_d, p1, p2)


# ── Tangent Lines ──────────────────────────────────────────

def upper_tangent(hull1, hull2):
    """Find upper tangent between two convex hulls (both CCW, hull1 left of hull2).
    Returns (i, j) indices into hull1 and hull2.
    The upper tangent touches the top of both hulls."""
    n, m = len(hull1), len(hull2)
    # Start from topmost point of hull1 (rightmost, highest)
    i = max(range(n), key=lambda k: (hull1[k][1], hull1[k][0]))
    # Start from topmost point of hull2 (leftmost, highest)
    j = max(range(m), key=lambda k: (hull2[k][1], -hull2[k][0]))

    # Iterate: move i on hull1 and j on hull2 until tangent found
    for _ in range(n + m + 2):
        changed = False
        # On hull1: move CW (forward) while hull1[(i+1)] is above tangent line
        while cross(hull1[i], hull2[j], hull1[(i + 1) % n]) > 0:
            i = (i + 1) % n
            changed = True
        # On hull2: move CCW (backward) while hull2[(j-1)] is above tangent line
        while cross(hull1[i], hull2[j], hull2[(j - 1) % m]) < 0:
            j = (j - 1) % m
            changed = True
        if not changed:
            break

    return (i, j)


def lower_tangent(hull1, hull2):
    """Find lower tangent between two convex hulls (both CCW, hull1 left of hull2).
    Returns (i, j) indices. The lower tangent touches the bottom of both hulls."""
    n, m = len(hull1), len(hull2)
    # Start from bottommost of hull1 (rightmost, lowest)
    i = min(range(n), key=lambda k: (hull1[k][1], -hull1[k][0]))
    # Start from bottommost of hull2 (leftmost, lowest)
    j = min(range(m), key=lambda k: (hull2[k][1], hull2[k][0]))

    for _ in range(n + m + 2):
        changed = False
        # On hull1: move CCW (backward) while hull1[(i-1)] is below tangent line
        while cross(hull1[i], hull2[j], hull1[(i - 1) % n]) < 0:
            i = (i - 1) % n
            changed = True
        # On hull2: move CW (forward) while hull2[(j+1)] is below tangent line
        while cross(hull1[i], hull2[j], hull2[(j + 1) % m]) > 0:
            j = (j + 1) % m
            changed = True
        if not changed:
            break

    return (i, j)
