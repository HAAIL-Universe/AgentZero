"""
C088: KD-Tree -- Spatial Partitioning for Multi-dimensional Data

Components:
1. KDNode / KDTree -- balanced k-d tree construction and operations
2. NearestNeighborSearch -- 1-NN and k-NN with branch-and-bound pruning
3. RangeSearch -- orthogonal range queries (box) and radius queries (sphere)
4. BallTree -- bounding-sphere spatial index (better for high dimensions)
5. SpatialIndex -- unified interface with bulk operations
"""

import math
import heapq
from collections import namedtuple


# ---------------------------------------------------------------------------
# KDNode
# ---------------------------------------------------------------------------

class KDNode:
    """A node in a KD-tree."""
    __slots__ = ('point', 'data', 'left', 'right', 'axis', 'deleted')

    def __init__(self, point, data=None, axis=0, left=None, right=None):
        self.point = tuple(point)
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right
        self.deleted = False


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def euclidean_distance_sq(a, b):
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def euclidean_distance(a, b):
    return math.sqrt(euclidean_distance_sq(a, b))


def manhattan_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def chebyshev_distance(a, b):
    return max(abs(ai - bi) for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# KDTree
# ---------------------------------------------------------------------------

class KDTree:
    """
    K-dimensional tree for spatial partitioning.

    Supports:
    - Balanced construction from point set: O(n log n)
    - Nearest neighbor: O(log n) average
    - k-nearest neighbors: O(k log n) average
    - Range search (box): O(sqrt(n) + k) average
    - Radius search (sphere): O(sqrt(n) + k) average
    - Insert / lazy delete
    """

    def __init__(self, points=None, dimensions=None, distance_fn=None):
        """
        Build a KD-tree.

        Args:
            points: list of (point, data) tuples or just points (tuples/lists).
                    If just points, data is set to None.
            dimensions: number of dimensions (inferred from first point if not given)
            distance_fn: custom distance function (default: euclidean)
        """
        self.root = None
        self.dimensions = dimensions
        self.size = 0
        self._distance_fn = distance_fn or euclidean_distance
        self._distance_sq_fn = euclidean_distance_sq if distance_fn is None else None

        if points:
            normalized = []
            for p in points:
                if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (tuple, list)):
                    normalized.append((tuple(p[0]), p[1]))
                else:
                    normalized.append((tuple(p), None))

            if self.dimensions is None:
                self.dimensions = len(normalized[0][0])

            self.root = self._build(normalized, depth=0)
            self.size = len(normalized)

    def _build(self, points, depth):
        """Build balanced KD-tree using median-of-axis splitting."""
        if not points:
            return None

        axis = depth % self.dimensions
        points.sort(key=lambda p: p[0][axis])
        mid = len(points) // 2

        node = KDNode(
            point=points[mid][0],
            data=points[mid][1],
            axis=axis,
            left=self._build(points[:mid], depth + 1),
            right=self._build(points[mid + 1:], depth + 1),
        )
        return node

    def insert(self, point, data=None):
        """Insert a point into the tree."""
        point = tuple(point)
        if self.dimensions is None:
            self.dimensions = len(point)
        self.root = self._insert(self.root, point, data, 0)
        self.size += 1

    def _insert(self, node, point, data, depth):
        if node is None:
            return KDNode(point, data, axis=depth % self.dimensions)

        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, data, depth + 1)
        else:
            node.right = self._insert(node.right, point, data, depth + 1)
        return node

    def delete(self, point):
        """Lazy delete: mark node as deleted."""
        point = tuple(point)
        node = self._find_node(self.root, point)
        if node is not None and not node.deleted:
            node.deleted = True
            self.size -= 1
            return True
        return False

    def _find_node(self, node, point):
        """Find exact node matching point."""
        if node is None:
            return None
        if node.point == point and not node.deleted:
            return node

        axis = node.axis
        if point[axis] < node.point[axis]:
            return self._find_node(node.left, point)
        else:
            # Could be in right subtree, but also check left for duplicates
            result = self._find_node(node.right, point)
            if result is None and point[axis] == node.point[axis]:
                result = self._find_node(node.left, point)
            return result

    def contains(self, point):
        """Check if point exists (not deleted) in tree."""
        point = tuple(point)
        return self._find_node(self.root, point) is not None

    def nearest(self, query, return_distance=False):
        """
        Find the nearest neighbor to query point.

        Returns:
            (point, data) or (point, data, distance) if return_distance=True.
            None if tree is empty.
        """
        query = tuple(query)
        if self.root is None:
            return None

        best = [None, float('inf')]  # [node, best_dist]
        self._nearest_search(self.root, query, best)

        if best[0] is None:
            return None

        node = best[0]
        if return_distance:
            return (node.point, node.data, best[1])
        return (node.point, node.data)

    def _nearest_search(self, node, query, best):
        if node is None:
            return

        if not node.deleted:
            dist = self._distance_fn(query, node.point)
            if dist < best[1]:
                best[0] = node
                best[1] = dist

        axis = node.axis
        diff = query[axis] - node.point[axis]

        # Search the side containing the query first
        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        self._nearest_search(near, query, best)

        # Prune: only search far side if the splitting plane is closer than best
        if abs(diff) < best[1]:
            self._nearest_search(far, query, best)

    def k_nearest(self, query, k, return_distances=False):
        """
        Find k nearest neighbors.

        Returns list of (point, data) or (point, data, distance) sorted by distance.
        """
        query = tuple(query)
        if self.root is None:
            return []

        # Max-heap of (-distance, point, data) -- keep k closest
        heap = []
        self._knn_search(self.root, query, k, heap)

        results = []
        while heap:
            neg_dist, _, node = heapq.heappop(heap)
            if return_distances:
                results.append((node.point, node.data, -neg_dist))
            else:
                results.append((node.point, node.data))

        results.reverse()  # Sort ascending by distance
        return results

    def _knn_search(self, node, query, k, heap):
        if node is None:
            return

        if not node.deleted:
            dist = self._distance_fn(query, node.point)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, id(node), node))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, id(node), node))

        axis = node.axis
        diff = query[axis] - node.point[axis]

        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        self._knn_search(near, query, k, heap)

        # Prune: only search far side if splitting plane is closer than k-th best
        worst_dist = -heap[0][0] if len(heap) == k else float('inf')
        if abs(diff) < worst_dist:
            self._knn_search(far, query, k, heap)

    def range_search(self, min_point, max_point):
        """
        Orthogonal range query: find all points within axis-aligned bounding box.

        Args:
            min_point: lower corner of box (inclusive)
            max_point: upper corner of box (inclusive)

        Returns:
            list of (point, data) within the box.
        """
        min_point = tuple(min_point)
        max_point = tuple(max_point)
        results = []
        self._range_search(self.root, min_point, max_point, results)
        return results

    def _range_search(self, node, min_pt, max_pt, results):
        if node is None:
            return

        if not node.deleted:
            # Check if point is inside box
            inside = all(lo <= p <= hi for p, lo, hi in zip(node.point, min_pt, max_pt))
            if inside:
                results.append((node.point, node.data))

        axis = node.axis
        # Prune subtrees that can't contain points in range
        if min_pt[axis] <= node.point[axis]:
            self._range_search(node.left, min_pt, max_pt, results)
        if max_pt[axis] >= node.point[axis]:
            self._range_search(node.right, min_pt, max_pt, results)

    def radius_search(self, center, radius, return_distances=False):
        """
        Find all points within given radius of center.

        Returns:
            list of (point, data) or (point, data, distance) within radius.
        """
        center = tuple(center)
        results = []
        self._radius_search(self.root, center, radius, results, return_distances)
        return results

    def _radius_search(self, node, center, radius, results, return_distances):
        if node is None:
            return

        if not node.deleted:
            dist = self._distance_fn(center, node.point)
            if dist <= radius:
                if return_distances:
                    results.append((node.point, node.data, dist))
                else:
                    results.append((node.point, node.data))

        axis = node.axis
        diff = center[axis] - node.point[axis]

        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        self._radius_search(near, center, radius, results, return_distances)

        # Prune: axis distance must be within radius
        if abs(diff) <= radius:
            self._radius_search(far, center, radius, results, return_distances)

    def points(self):
        """Return all non-deleted points as (point, data) list."""
        result = []
        self._collect(self.root, result)
        return result

    def _collect(self, node, result):
        if node is None:
            return
        if not node.deleted:
            result.append((node.point, node.data))
        self._collect(node.left, result)
        self._collect(node.right, result)

    def depth(self):
        """Return tree depth."""
        return self._depth(self.root)

    def _depth(self, node):
        if node is None:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def rebalance(self):
        """Rebuild tree from non-deleted points for optimal balance."""
        pts = self.points()
        self.root = self._build(pts, 0)
        self.size = len(pts)


# ---------------------------------------------------------------------------
# BallTree -- bounding sphere spatial index
# ---------------------------------------------------------------------------

class BallNode:
    """A node in a Ball Tree."""
    __slots__ = ('center', 'radius', 'point', 'data', 'left', 'right', 'is_leaf')

    def __init__(self, center, radius, point=None, data=None,
                 left=None, right=None, is_leaf=False):
        self.center = center
        self.radius = radius
        self.point = point
        self.data = data
        self.left = left
        self.right = right
        self.is_leaf = is_leaf


class BallTree:
    """
    Ball Tree for spatial queries using bounding spheres.

    Better than KD-tree for high-dimensional data where axis-aligned
    splits become inefficient.
    """

    def __init__(self, points=None, leaf_size=10, distance_fn=None):
        self.root = None
        self.leaf_size = leaf_size
        self.dimensions = None
        self.size = 0
        self._distance_fn = distance_fn or euclidean_distance

        if points:
            normalized = []
            for p in points:
                if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (tuple, list)):
                    normalized.append((tuple(p[0]), p[1]))
                else:
                    normalized.append((tuple(p), None))

            self.dimensions = len(normalized[0][0])
            self.size = len(normalized)
            self.root = self._build(normalized)

    def _centroid(self, points):
        """Compute centroid of point set."""
        n = len(points)
        dims = len(points[0][0])
        center = [0.0] * dims
        for pt, _ in points:
            for i in range(dims):
                center[i] += pt[i]
        return tuple(c / n for c in center)

    def _bounding_radius(self, center, points):
        """Max distance from center to any point."""
        return max(self._distance_fn(center, pt) for pt, _ in points)

    def _build(self, points):
        center = self._centroid(points)

        if len(points) <= self.leaf_size:
            # Leaf: store all points, create one leaf per point for simplicity
            if len(points) == 1:
                return BallNode(
                    center=points[0][0],
                    radius=0.0,
                    point=points[0][0],
                    data=points[0][1],
                    is_leaf=True
                )
            radius = self._bounding_radius(center, points)
            # Create a balanced subtree for small clusters
            return self._build_leaf_cluster(points, center, radius)

        # Find dimension with max spread
        dims = len(points[0][0])
        best_axis = 0
        best_spread = 0
        for d in range(dims):
            vals = [pt[d] for pt, _ in points]
            spread = max(vals) - min(vals)
            if spread > best_spread:
                best_spread = spread
                best_axis = d

        # Split on median of best axis
        points.sort(key=lambda p: p[0][best_axis])
        mid = len(points) // 2

        left = self._build(points[:mid])
        right = self._build(points[mid:])
        radius = self._bounding_radius(center, points)

        return BallNode(center=center, radius=radius, left=left, right=right)

    def _build_leaf_cluster(self, points, center, radius):
        """Build a balanced subtree for a small cluster of points."""
        if len(points) == 1:
            return BallNode(
                center=points[0][0], radius=0.0,
                point=points[0][0], data=points[0][1], is_leaf=True
            )
        if center is None:
            center = self._centroid(points)
        if radius is None:
            radius = self._bounding_radius(center, points)
        mid = len(points) // 2
        left = self._build_leaf_cluster(points[:mid], None, None)
        right = self._build_leaf_cluster(points[mid:], None, None)
        return BallNode(center=center, radius=radius, left=left, right=right)

    def nearest(self, query, return_distance=False):
        """Find nearest neighbor."""
        query = tuple(query)
        if self.root is None:
            return None

        best = [None, float('inf')]
        self._nearest_search(self.root, query, best)

        if best[0] is None:
            return None
        node = best[0]
        if return_distance:
            return (node.point, node.data, best[1])
        return (node.point, node.data)

    def _nearest_search(self, node, query, best):
        if node is None:
            return

        # Prune: if ball is farther than best, skip
        dist_to_center = self._distance_fn(query, node.center)
        if dist_to_center - node.radius >= best[1]:
            return

        if node.is_leaf:
            dist = self._distance_fn(query, node.point)
            if dist < best[1]:
                best[0] = node
                best[1] = dist
            return

        # Search closer child first
        if node.left and node.right:
            dl = self._distance_fn(query, node.left.center)
            dr = self._distance_fn(query, node.right.center)
            if dl <= dr:
                self._nearest_search(node.left, query, best)
                self._nearest_search(node.right, query, best)
            else:
                self._nearest_search(node.right, query, best)
                self._nearest_search(node.left, query, best)
        elif node.left:
            self._nearest_search(node.left, query, best)
        elif node.right:
            self._nearest_search(node.right, query, best)

    def k_nearest(self, query, k, return_distances=False):
        """Find k nearest neighbors."""
        query = tuple(query)
        if self.root is None:
            return []

        heap = []
        self._knn_search(self.root, query, k, heap)

        results = []
        while heap:
            neg_dist, _, node = heapq.heappop(heap)
            if return_distances:
                results.append((node.point, node.data, -neg_dist))
            else:
                results.append((node.point, node.data))

        results.reverse()
        return results

    def _knn_search(self, node, query, k, heap):
        if node is None:
            return

        dist_to_center = self._distance_fn(query, node.center)
        worst = -heap[0][0] if len(heap) == k else float('inf')
        if dist_to_center - node.radius >= worst:
            return

        if node.is_leaf:
            dist = self._distance_fn(query, node.point)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, id(node), node))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, id(node), node))
            return

        if node.left and node.right:
            dl = self._distance_fn(query, node.left.center)
            dr = self._distance_fn(query, node.right.center)
            if dl <= dr:
                self._knn_search(node.left, query, k, heap)
                self._knn_search(node.right, query, k, heap)
            else:
                self._knn_search(node.right, query, k, heap)
                self._knn_search(node.left, query, k, heap)
        elif node.left:
            self._knn_search(node.left, query, k, heap)
        elif node.right:
            self._knn_search(node.right, query, k, heap)

    def radius_search(self, center, radius, return_distances=False):
        """Find all points within radius of center."""
        center = tuple(center)
        results = []
        self._radius_search(self.root, center, radius, results, return_distances)
        return results

    def _radius_search(self, node, center, radius, results, return_distances):
        if node is None:
            return

        dist_to_center = self._distance_fn(center, node.center)
        if dist_to_center - node.radius > radius:
            return

        if node.is_leaf:
            dist = self._distance_fn(center, node.point)
            if dist <= radius:
                if return_distances:
                    results.append((node.point, node.data, dist))
                else:
                    results.append((node.point, node.data))
            return

        self._radius_search(node.left, center, radius, results, return_distances)
        self._radius_search(node.right, center, radius, results, return_distances)

    def points(self):
        """Return all points as (point, data) list."""
        result = []
        self._collect(self.root, result)
        return result

    def _collect(self, node, result):
        if node is None:
            return
        if node.is_leaf:
            result.append((node.point, node.data))
        else:
            self._collect(node.left, result)
            self._collect(node.right, result)


# ---------------------------------------------------------------------------
# SpatialIndex -- unified interface
# ---------------------------------------------------------------------------

class SpatialIndex:
    """
    Unified spatial index supporting multiple backends and bulk operations.

    Provides:
    - Backend selection (KDTree or BallTree)
    - Bulk insert / delete
    - All-pairs nearest neighbors
    - Convex hull (2D)
    - Bounding box computation
    """

    def __init__(self, points=None, dimensions=None, backend='kdtree',
                 distance_fn=None, leaf_size=10):
        self.backend_type = backend
        self._distance_fn = distance_fn

        if backend == 'balltree':
            self._tree = BallTree(points, leaf_size=leaf_size, distance_fn=distance_fn)
            self.dimensions = self._tree.dimensions
        else:
            self._tree = KDTree(points, dimensions=dimensions, distance_fn=distance_fn)
            self.dimensions = self._tree.dimensions

    @property
    def size(self):
        return self._tree.size

    def insert(self, point, data=None):
        """Insert a point (KDTree backend only)."""
        if self.backend_type == 'kdtree':
            self._tree.insert(point, data)
        else:
            raise NotImplementedError("BallTree does not support incremental insert")

    def delete(self, point):
        """Delete a point (KDTree backend only)."""
        if self.backend_type == 'kdtree':
            return self._tree.delete(point)
        raise NotImplementedError("BallTree does not support delete")

    def nearest(self, query, return_distance=False):
        return self._tree.nearest(query, return_distance=return_distance)

    def k_nearest(self, query, k, return_distances=False):
        return self._tree.k_nearest(query, k, return_distances=return_distances)

    def range_search(self, min_point, max_point):
        """Orthogonal range query (KDTree only)."""
        if self.backend_type == 'kdtree':
            return self._tree.range_search(min_point, max_point)
        raise NotImplementedError("Use radius_search for BallTree")

    def radius_search(self, center, radius, return_distances=False):
        return self._tree.radius_search(center, radius, return_distances=return_distances)

    def contains(self, point):
        """Check if point exists (KDTree only)."""
        if self.backend_type == 'kdtree':
            return self._tree.contains(point)
        # For BallTree, do a radius-0 search
        results = self._tree.radius_search(point, 0.0)
        return len(results) > 0

    def points(self):
        return self._tree.points()

    def rebalance(self):
        """Rebalance tree (KDTree only)."""
        if self.backend_type == 'kdtree':
            self._tree.rebalance()

    def bounding_box(self):
        """Compute axis-aligned bounding box of all points."""
        pts = self.points()
        if not pts:
            return None
        dims = len(pts[0][0])
        min_pt = list(pts[0][0])
        max_pt = list(pts[0][0])
        for pt, _ in pts:
            for i in range(dims):
                if pt[i] < min_pt[i]:
                    min_pt[i] = pt[i]
                if pt[i] > max_pt[i]:
                    max_pt[i] = pt[i]
        return (tuple(min_pt), tuple(max_pt))

    def all_pairs_nearest(self):
        """
        For each point, find its nearest neighbor among all other points.

        Returns: list of (point, nearest_point, distance)
        """
        pts = self.points()
        results = []
        for pt, data in pts:
            neighbors = self._tree.k_nearest(pt, 2, return_distances=True)
            # First result is the point itself (distance 0), second is nearest
            for n_pt, n_data, n_dist in neighbors:
                if n_pt != pt or n_dist > 0:
                    results.append((pt, n_pt, n_dist))
                    break
        return results

    def convex_hull_2d(self):
        """
        Compute 2D convex hull using Andrew's monotone chain algorithm.

        Returns: list of points forming the convex hull in counter-clockwise order.
        Requires 2D points.
        """
        pts = self.points()
        if not pts:
            return []

        coords = sorted(set(pt for pt, _ in pts))
        if len(coords) <= 1:
            return list(coords)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in coords:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(coords):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove last point of each half (it's repeated)
        return lower[:-1] + upper[:-1]
