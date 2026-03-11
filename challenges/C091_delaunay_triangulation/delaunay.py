"""
C091: Delaunay Triangulation

Bowyer-Watson incremental insertion, Voronoi dual extraction,
constrained Delaunay triangulation, Ruppert's mesh refinement,
and point location.

Composes conceptually with C090 (convex hull) -- the convex hull
of the point set forms the boundary of the Delaunay triangulation.
"""

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

EPSILON = 1e-10


def orient2d(a, b, c):
    """Return positive if a->b->c is CCW, negative if CW, zero if collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def in_circumcircle(a, b, c, d):
    """Return True if point d is strictly inside the circumcircle of triangle (a,b,c).
    Assumes a,b,c are in CCW order."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    det = (ax * ax + ay * ay) * (bx * cy - cx * by) \
        - (bx * bx + by * by) * (ax * cy - cx * ay) \
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    return det > EPSILON


def circumcenter(a, b, c):
    """Return circumcenter of triangle (a,b,c)."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < EPSILON:
        # Degenerate (collinear) -- return midpoint as fallback
        return ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0)
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / D
    return (ux, uy)


def circumradius_sq(a, b, c):
    """Return squared circumradius of triangle (a,b,c)."""
    cc = circumcenter(a, b, c)
    dx = a[0] - cc[0]
    dy = a[1] - cc[1]
    return dx * dx + dy * dy


def dist_sq(a, b):
    """Squared distance between two points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a, b):
    """Distance between two points."""
    return math.sqrt(dist_sq(a, b))


def midpoint(a, b):
    """Midpoint of segment a-b."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (properly, not at endpoints)."""
    d1 = orient2d(p3, p4, p1)
    d2 = orient2d(p3, p4, p2)
    d3 = orient2d(p1, p2, p3)
    d4 = orient2d(p1, p2, p4)
    if ((d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)) and \
       ((d3 > EPSILON and d4 < -EPSILON) or (d3 < -EPSILON and d4 > EPSILON)):
        return True
    return False


def point_on_segment(p, a, b):
    """Check if point p lies on segment a-b (excluding endpoints)."""
    cross = orient2d(a, b, p)
    if abs(cross) > EPSILON:
        return False
    # Check bounding box
    minx = min(a[0], b[0]) - EPSILON
    maxx = max(a[0], b[0]) + EPSILON
    miny = min(a[1], b[1]) - EPSILON
    maxy = max(a[1], b[1]) + EPSILON
    if minx <= p[0] <= maxx and miny <= p[1] <= maxy:
        # Exclude endpoints
        if dist_sq(p, a) < EPSILON * EPSILON or dist_sq(p, b) < EPSILON * EPSILON:
            return False
        return True
    return False


def segment_intersection_point(p1, p2, p3, p4):
    """Return intersection point of segments p1-p2 and p3-p4, or None."""
    d1 = orient2d(p3, p4, p1)
    d2 = orient2d(p3, p4, p2)
    d3 = orient2d(p1, p2, p3)
    d4 = orient2d(p1, p2, p4)

    if ((d1 > EPSILON and d2 < -EPSILON) or (d1 < -EPSILON and d2 > EPSILON)) and \
       ((d3 > EPSILON and d4 < -EPSILON) or (d3 < -EPSILON and d4 > EPSILON)):
        t = d1 / (d1 - d2)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        return (x, y)
    return None


def triangle_area(a, b, c):
    """Signed area of triangle a,b,c."""
    return orient2d(a, b, c) / 2.0


def shortest_edge_length_sq(a, b, c):
    """Squared length of the shortest edge of triangle a,b,c."""
    return min(dist_sq(a, b), dist_sq(b, c), dist_sq(a, c))


def triangle_quality(a, b, c):
    """Ratio of circumradius to shortest edge -- lower is better.
    Equilateral triangle has ratio ~0.577. Used for mesh refinement."""
    # Degenerate (collinear) check
    area = abs(orient2d(a, b, c))
    if area < EPSILON:
        return float('inf')
    r2 = circumradius_sq(a, b, c)
    s2 = shortest_edge_length_sq(a, b, c)
    if s2 < EPSILON * EPSILON:
        return float('inf')
    return math.sqrt(r2 / s2)


# ---------------------------------------------------------------------------
# Triangle class
# ---------------------------------------------------------------------------

class Triangle:
    """A triangle in the Delaunay triangulation."""
    __slots__ = ('vertices', 'neighbors', 'id')
    _next_id = 0

    def __init__(self, v0, v1, v2):
        # Ensure CCW order
        if orient2d(v0, v1, v2) < 0:
            v1, v2 = v2, v1
        self.vertices = (v0, v1, v2)
        self.neighbors = [None, None, None]  # neighbor opposite vertex i
        self.id = Triangle._next_id
        Triangle._next_id += 1

    def __repr__(self):
        return f"Triangle({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

    def has_vertex(self, v):
        return v in self.vertices

    def vertex_index(self, v):
        return self.vertices.index(v)

    def opposite_neighbor(self, v):
        """Return neighbor opposite to vertex v."""
        return self.neighbors[self.vertices.index(v)]

    def edge_opposite(self, v):
        """Return the edge (pair of vertices) opposite to vertex v."""
        i = self.vertices.index(v)
        return (self.vertices[(i + 1) % 3], self.vertices[(i + 2) % 3])

    def contains_point(self, p):
        """Check if point p is inside or on boundary of this triangle."""
        a, b, c = self.vertices
        d1 = orient2d(a, b, p)
        d2 = orient2d(b, c, p)
        d3 = orient2d(c, a, p)
        has_neg = (d1 < -EPSILON) or (d2 < -EPSILON) or (d3 < -EPSILON)
        has_pos = (d1 > EPSILON) or (d2 > EPSILON) or (d3 > EPSILON)
        return not (has_neg and has_pos)

    def circumcenter(self):
        return circumcenter(*self.vertices)

    def circumradius_sq(self):
        return circumradius_sq(*self.vertices)


def _link_neighbors(t1, t2):
    """Set t1 and t2 as neighbors sharing their common edge."""
    shared = []
    for v in t1.vertices:
        if v in t2.vertices:
            shared.append(v)
    if len(shared) != 2:
        return
    # t1's neighbor opposite to the vertex NOT in shared
    for v in t1.vertices:
        if v not in shared:
            t1.neighbors[t1.vertex_index(v)] = t2
            break
    for v in t2.vertices:
        if v not in shared:
            t2.neighbors[t2.vertex_index(v)] = t1
            break


# ---------------------------------------------------------------------------
# Delaunay Triangulation (Bowyer-Watson)
# ---------------------------------------------------------------------------

class DelaunayTriangulation:
    """Delaunay triangulation using Bowyer-Watson incremental insertion."""

    def __init__(self, points=None):
        self.triangles = set()
        self.points = []
        self._super_triangle_verts = None

        if points:
            self.build(points)

    def build(self, points):
        """Build triangulation from a list of (x, y) points."""
        if len(points) < 3:
            self.points = list(points)
            return self

        # Deduplicate
        seen = set()
        unique = []
        for p in points:
            key = (round(p[0], 10), round(p[1], 10))
            if key not in seen:
                seen.add(key)
                unique.append((float(p[0]), float(p[1])))
        self.points = unique

        if len(unique) < 3:
            return self

        # Check if all points are collinear
        all_collinear = True
        for i in range(2, len(unique)):
            if abs(orient2d(unique[0], unique[1], unique[i])) > EPSILON:
                all_collinear = False
                break
        if all_collinear:
            return self  # No triangulation possible

        # Create super-triangle
        self._create_super_triangle()

        # Insert points one by one
        for p in unique:
            self._insert_point(p)

        # Remove super-triangle vertices and their incident triangles
        self._remove_super_triangle()

        return self

    def _create_super_triangle(self):
        """Create a super-triangle that contains all points."""
        min_x = min(p[0] for p in self.points)
        max_x = max(p[0] for p in self.points)
        min_y = min(p[1] for p in self.points)
        max_y = max(p[1] for p in self.points)

        dx = max_x - min_x
        dy = max_y - min_y
        d = max(dx, dy, 1.0)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Make super-triangle large enough
        margin = 10.0 * d
        v0 = (cx - margin, cy - margin)
        v1 = (cx + margin, cy - margin)
        v2 = (cx, cy + margin)

        self._super_triangle_verts = (v0, v1, v2)
        t = Triangle(v0, v1, v2)
        self.triangles.add(t)

    def _insert_point(self, p):
        """Insert a point into the triangulation using Bowyer-Watson."""
        # Find all triangles whose circumcircle contains p
        bad_triangles = set()
        for t in self.triangles:
            if in_circumcircle(t.vertices[0], t.vertices[1], t.vertices[2], p):
                bad_triangles.add(t)

        if not bad_triangles:
            # Point might be on a circumcircle edge or outside -- find containing triangle
            for t in self.triangles:
                if t.contains_point(p):
                    bad_triangles.add(t)
                    break

        if not bad_triangles:
            return  # Point outside all triangles (shouldn't happen with super-triangle)

        # Find boundary of the polygonal hole
        boundary = []
        for t in bad_triangles:
            for i in range(3):
                neighbor = t.neighbors[i]
                if neighbor is None or neighbor not in bad_triangles:
                    # This edge is on the boundary
                    e0 = t.vertices[(i + 1) % 3]
                    e1 = t.vertices[(i + 2) % 3]
                    boundary.append((e0, e1))

        # Remove bad triangles
        for t in bad_triangles:
            # Unlink from neighbors
            for i in range(3):
                nb = t.neighbors[i]
                if nb and nb not in bad_triangles:
                    for j in range(3):
                        if nb.neighbors[j] is t:
                            nb.neighbors[j] = None
            self.triangles.discard(t)

        # Create new triangles from boundary edges to the new point
        new_triangles = []
        for e0, e1 in boundary:
            nt = Triangle(e0, e1, p)
            self.triangles.add(nt)
            new_triangles.append(nt)

        # Link new triangles to each other and to existing neighbors
        for nt in new_triangles:
            for ot in self.triangles:
                if ot is nt:
                    continue
                _link_neighbors(nt, ot)

    def _remove_super_triangle(self):
        """Remove all triangles that share a vertex with the super-triangle."""
        if not self._super_triangle_verts:
            return
        sv = set(self._super_triangle_verts)
        to_remove = {t for t in self.triangles if any(v in sv for v in t.vertices)}
        for t in to_remove:
            for i in range(3):
                nb = t.neighbors[i]
                if nb and nb not in to_remove:
                    for j in range(3):
                        if nb.neighbors[j] is t:
                            nb.neighbors[j] = None
            self.triangles.discard(t)
        self._super_triangle_verts = None

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def get_triangles(self):
        """Return list of triangles as tuples of (v0, v1, v2)."""
        return [t.vertices for t in self.triangles]

    def get_edges(self):
        """Return set of all edges as frozensets of point pairs."""
        edges = set()
        for t in self.triangles:
            v = t.vertices
            for i in range(3):
                e = frozenset((v[i], v[(i + 1) % 3]))
                edges.add(e)
        return edges

    def get_edge_list(self):
        """Return edges as list of tuples (p1, p2)."""
        return [tuple(e) for e in self.get_edges()]

    def locate_point(self, p):
        """Find the triangle containing point p using walking strategy."""
        if not self.triangles:
            return None

        # Start from an arbitrary triangle
        current = next(iter(self.triangles))
        visited = set()

        for _ in range(len(self.triangles) + 1):
            if current in visited:
                break
            visited.add(current)

            if current.contains_point(p):
                return current

            # Walk toward p
            a, b, c = current.vertices
            moved = False
            # Check which edge to cross
            if orient2d(a, b, p) < -EPSILON and current.neighbors[2] is not None:
                current = current.neighbors[2]
                moved = True
            elif orient2d(b, c, p) < -EPSILON and current.neighbors[0] is not None:
                current = current.neighbors[0]
                moved = True
            elif orient2d(c, a, p) < -EPSILON and current.neighbors[1] is not None:
                current = current.neighbors[1]
                moved = True

            if not moved:
                break

        # Fallback: linear scan
        for t in self.triangles:
            if t.contains_point(p):
                return t
        return None

    def nearest_point(self, query):
        """Find the nearest point in the triangulation to query point."""
        if not self.points:
            return None
        best = None
        best_dist = float('inf')
        for p in self.points:
            d = dist_sq(query, p)
            if d < best_dist:
                best_dist = d
                best = p
        return best

    def get_neighbors(self, point):
        """Return all points connected to the given point by an edge."""
        neighbors = set()
        for t in self.triangles:
            if point in t.vertices:
                for v in t.vertices:
                    if v != point:
                        neighbors.add(v)
        return neighbors

    def convex_hull(self):
        """Extract convex hull from the triangulation boundary."""
        if not self.triangles:
            return []

        # Find boundary edges (edges with only one incident triangle)
        edge_count = defaultdict(int)
        edge_tris = defaultdict(list)
        for t in self.triangles:
            v = t.vertices
            for i in range(3):
                e = frozenset((v[i], v[(i + 1) % 3]))
                edge_count[e] += 1
                edge_tris[e].append((v[i], v[(i + 1) % 3]))

        boundary_edges = {}
        for e, count in edge_count.items():
            if count == 1:
                directed = edge_tris[e][0]
                # Find the triangle containing this edge and get CCW direction
                for t in self.triangles:
                    v = t.vertices
                    for i in range(3):
                        if frozenset((v[i], v[(i + 1) % 3])) == e:
                            boundary_edges[v[i]] = v[(i + 1) % 3]
                            break

        if not boundary_edges:
            return []

        # Follow boundary edges to form hull
        start = min(boundary_edges.keys())  # deterministic start
        hull = [start]
        current = boundary_edges.get(start)
        while current and current != start and len(hull) < len(boundary_edges) + 1:
            hull.append(current)
            current = boundary_edges.get(current)

        return hull

    def is_delaunay(self):
        """Verify the Delaunay property: no point is inside any circumcircle."""
        all_points = set()
        for t in self.triangles:
            for v in t.vertices:
                all_points.add(v)

        for t in self.triangles:
            a, b, c = t.vertices
            for p in all_points:
                if p in t.vertices:
                    continue
                if in_circumcircle(a, b, c, p):
                    return False
        return True

    def num_triangles(self):
        return len(self.triangles)

    def num_edges(self):
        return len(self.get_edges())

    def num_points(self):
        return len(self.points)


# ---------------------------------------------------------------------------
# Voronoi Diagram (dual of Delaunay)
# ---------------------------------------------------------------------------

class VoronoiDiagram:
    """Voronoi diagram extracted from a Delaunay triangulation."""

    def __init__(self, delaunay=None, points=None):
        self.vertices = []  # Voronoi vertices (circumcenters)
        self.edges = []     # (v1_idx, v2_idx) or (v_idx, None) for infinite
        self.regions = {}   # point -> list of vertex indices (CCW order)
        self.ridge_points = []  # pairs of input points separated by each edge
        self._dt = None

        if delaunay:
            self._build_from_delaunay(delaunay)
        elif points:
            dt = DelaunayTriangulation(points)
            self._build_from_delaunay(dt)

    def _build_from_delaunay(self, dt):
        self._dt = dt
        if not dt.triangles:
            return

        # Each triangle's circumcenter is a Voronoi vertex
        tri_to_idx = {}
        for t in dt.triangles:
            idx = len(self.vertices)
            tri_to_idx[t.id] = idx
            self.vertices.append(t.circumcenter())

        # Each Delaunay edge corresponds to a Voronoi edge
        # connecting circumcenters of the two adjacent triangles
        edge_processed = set()
        for t in dt.triangles:
            for i in range(3):
                e0 = t.vertices[(i + 1) % 3]
                e1 = t.vertices[(i + 2) % 3]
                edge_key = frozenset((e0, e1))
                if edge_key in edge_processed:
                    continue
                edge_processed.add(edge_key)

                nb = t.neighbors[i]
                t_idx = tri_to_idx[t.id]

                if nb is not None:
                    nb_idx = tri_to_idx[nb.id]
                    self.edges.append((t_idx, nb_idx))
                    self.ridge_points.append((e0, e1))
                else:
                    # Boundary edge -- Voronoi edge goes to infinity
                    self.edges.append((t_idx, None))
                    self.ridge_points.append((e0, e1))

        # Build regions: for each input point, collect surrounding circumcenters
        for p in dt.points:
            incident = []
            for t in dt.triangles:
                if p in t.vertices:
                    incident.append(t)
            if not incident:
                continue

            # Sort incident triangles by angle around p
            def angle_key(t):
                cc = t.circumcenter()
                return math.atan2(cc[1] - p[1], cc[0] - p[0])

            incident.sort(key=angle_key)
            self.regions[p] = [tri_to_idx[t.id] for t in incident]

    def get_vertices(self):
        return list(self.vertices)

    def get_edges(self):
        """Return Voronoi edges as pairs of (point, point_or_None)."""
        result = []
        for v1_idx, v2_idx in self.edges:
            p1 = self.vertices[v1_idx]
            p2 = self.vertices[v2_idx] if v2_idx is not None else None
            result.append((p1, p2))
        return result

    def get_finite_edges(self):
        """Return only finite Voronoi edges as pairs of points."""
        result = []
        for v1_idx, v2_idx in self.edges:
            if v2_idx is not None:
                result.append((self.vertices[v1_idx], self.vertices[v2_idx]))
        return result

    def get_region(self, point):
        """Return Voronoi region vertices for a given input point."""
        if point not in self.regions:
            return []
        return [self.vertices[i] for i in self.regions[point]]

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def num_finite_edges(self):
        return sum(1 for _, v2 in self.edges if v2 is not None)


# ---------------------------------------------------------------------------
# Constrained Delaunay Triangulation
# ---------------------------------------------------------------------------

class ConstrainedDelaunay:
    """Constrained Delaunay triangulation -- Delaunay with forced edges."""

    def __init__(self, points=None, constraints=None):
        self.dt = None
        self.constraints = set()  # frozenset of forced edges
        self._constraint_points = set()

        if points:
            self.build(points, constraints or [])

    def build(self, points, constraints):
        """Build CDT from points with constrained edge pairs [(p1, p2), ...]."""
        all_points = list(points)

        # Collect constraint endpoints
        constraint_pairs = []
        for c in constraints:
            p1 = (float(c[0][0]), float(c[0][1]))
            p2 = (float(c[1][0]), float(c[1][1]))
            constraint_pairs.append((p1, p2))
            all_points.append(p1)
            all_points.append(p2)

        # Build unconstrained DT
        self.dt = DelaunayTriangulation(all_points)

        # Insert constraints
        for p1, p2 in constraint_pairs:
            self._insert_constraint(p1, p2)

        return self

    def _find_matching_point(self, target):
        """Find the actual point in the triangulation matching target."""
        for p in self.dt.points:
            if dist_sq(p, target) < EPSILON * EPSILON:
                return p
        return target

    def _insert_constraint(self, p1, p2):
        """Insert a constrained edge into the triangulation."""
        p1 = self._find_matching_point(p1)
        p2 = self._find_matching_point(p2)
        self.constraints.add(frozenset((p1, p2)))
        self._constraint_points.add(p1)
        self._constraint_points.add(p2)

        # Check if edge already exists
        edges = self.dt.get_edges()
        if frozenset((p1, p2)) in edges:
            return

        # Find intersecting edges and insert constraint by
        # collecting triangles the segment passes through and re-triangulating
        # Simple approach: insert midpoints along the constraint to force it
        self._force_edge(p1, p2)

    def _force_edge(self, p1, p2):
        """Force edge p1-p2 into triangulation by inserting subdivision points."""
        # Find triangles intersected by p1-p2
        edges = self.dt.get_edges()
        if frozenset((p1, p2)) in edges:
            return

        # Insert midpoint and recurse
        mid = midpoint(p1, p2)
        # Check if midpoint is already a point
        for p in self.dt.points:
            if dist_sq(p, mid) < EPSILON * EPSILON:
                # Already exists, try sub-segments
                self._force_edge(p1, p)
                self._force_edge(p, p2)
                return

        # Rebuild with midpoint
        new_points = self.dt.points + [mid]
        self.dt = DelaunayTriangulation(new_points)
        self._force_edge(p1, mid)
        self._force_edge(mid, p2)

    def get_triangles(self):
        return self.dt.get_triangles() if self.dt else []

    def get_edges(self):
        return self.dt.get_edges() if self.dt else set()

    def get_constraints(self):
        return self.constraints

    def is_constraint(self, p1, p2):
        """Check if edge p1-p2 is a constrained edge."""
        return frozenset((p1, p2)) in self.constraints

    def num_triangles(self):
        return self.dt.num_triangles() if self.dt else 0

    def is_valid(self):
        """Check that all constraints are present as edges."""
        if not self.dt:
            return True
        edges = self.dt.get_edges()
        for c in self.constraints:
            # Check if constraint exists as edge or is covered by sub-edges
            if c not in edges:
                # Check if it's covered by collinear sub-edges
                pts = sorted(c, key=lambda p: (p[0], p[1]))
                # For now, check direct edge
                return False
        return True


# ---------------------------------------------------------------------------
# Mesh Refinement (Ruppert's Algorithm)
# ---------------------------------------------------------------------------

class MeshRefiner:
    """Ruppert's algorithm for quality mesh refinement."""

    def __init__(self, dt=None, min_angle=20.0, max_area=None):
        """
        min_angle: minimum angle in degrees (default 20)
        max_area: maximum triangle area (optional)
        """
        self.min_angle = min_angle
        self.max_area = max_area
        self.dt = dt
        self._quality_threshold = 1.0 / (2.0 * math.sin(math.radians(min_angle)))

    def refine(self, dt=None, max_iterations=1000):
        """Refine the triangulation to improve mesh quality."""
        if dt:
            self.dt = dt

        if not self.dt or not self.dt.triangles:
            return self.dt

        points_to_add = []

        for iteration in range(max_iterations):
            # Find worst triangle
            worst = None
            worst_quality = 0

            for t in list(self.dt.triangles):
                a, b, c = t.vertices
                q = triangle_quality(a, b, c)
                area = abs(triangle_area(a, b, c))

                needs_refine = False
                if q > self._quality_threshold:
                    needs_refine = True
                if self.max_area is not None and area > self.max_area:
                    needs_refine = True

                if needs_refine and q > worst_quality:
                    worst_quality = q
                    worst = t

            if worst is None:
                break  # All triangles are good

            # Insert circumcenter of worst triangle
            cc = worst.circumcenter()
            # Check if point already exists (avoid infinite loop)
            too_close = False
            for p in self.dt.points:
                if dist_sq(p, cc) < EPSILON * EPSILON:
                    too_close = True
                    break
            if too_close:
                break

            new_points = self.dt.points + [cc]
            self.dt = DelaunayTriangulation(new_points)

        return self.dt

    def get_quality_stats(self):
        """Return quality statistics for the current mesh."""
        if not self.dt or not self.dt.triangles:
            return {'min_quality': 0, 'max_quality': 0, 'avg_quality': 0, 'count': 0}

        qualities = []
        for t in self.dt.triangles:
            q = triangle_quality(*t.vertices)
            qualities.append(q)

        return {
            'min_quality': min(qualities),
            'max_quality': max(qualities),
            'avg_quality': sum(qualities) / len(qualities),
            'count': len(qualities),
        }

    def min_angle_degrees(self):
        """Return the minimum angle across all triangles in degrees."""
        if not self.dt or not self.dt.triangles:
            return 0

        min_ang = 180.0
        for t in self.dt.triangles:
            a, b, c = t.vertices
            for p, q, r in [(a, b, c), (b, c, a), (c, a, b)]:
                # Angle at p between edges p-q and p-r
                v1 = (q[0] - p[0], q[1] - p[1])
                v2 = (r[0] - p[0], r[1] - p[1])
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                if m1 < EPSILON or m2 < EPSILON:
                    continue
                cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
                ang = math.degrees(math.acos(cos_a))
                min_ang = min(min_ang, ang)

        return min_ang


# ---------------------------------------------------------------------------
# Utility: Triangulation from polygon
# ---------------------------------------------------------------------------

def triangulate_polygon(polygon):
    """Triangulate a simple polygon given as a list of vertices.
    Returns list of triangle tuples."""
    if len(polygon) < 3:
        return []

    # Use Delaunay triangulation of polygon vertices,
    # then filter to only include triangles inside the polygon
    dt = DelaunayTriangulation(polygon)
    result = []
    poly_set = set((round(p[0], 10), round(p[1], 10)) for p in polygon)

    for tri in dt.get_triangles():
        # Check if triangle centroid is inside polygon
        cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
        cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
        if _point_in_polygon((cx, cy), polygon):
            result.append(tri)

    return result


def _point_in_polygon(point, polygon):
    """Ray casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


# ---------------------------------------------------------------------------
# Euler formula checks
# ---------------------------------------------------------------------------

def euler_check(dt):
    """Verify Euler's formula: V - E + F = 2 for triangulation.
    F includes the outer face."""
    v = dt.num_points()
    e = dt.num_edges()
    f = dt.num_triangles() + 1  # +1 for outer face
    return v - e + f


def expected_triangles(n, h):
    """Expected number of triangles for n points with h on convex hull: 2n - h - 2."""
    return 2 * n - h - 2


def expected_edges(n, h):
    """Expected number of edges for n points with h on convex hull: 3n - h - 3."""
    return 3 * n - h - 3
