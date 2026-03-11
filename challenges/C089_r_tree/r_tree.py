"""
C089: R-Tree -- Spatial indexing for rectangles and regions.

Extends spatial indexing beyond points (C088 KD-tree) to axis-aligned
bounding boxes. Supports insert, delete, search (window query, containment,
intersection), nearest neighbor, and bulk loading.

Components:
  - BoundingBox: N-dimensional axis-aligned bounding box
  - RTreeNode: Internal/leaf node with entries
  - RTree: Classic R-tree with quadratic split (Guttman 1984)
  - RStarTree: R*-tree with forced reinsert + overlap-minimizing split
  - STRBulkLoader: Sort-Tile-Recursive bulk loading for static datasets
  - SpatialIndex: Unified interface
"""

import math
from collections import deque


# ============================================================
# BoundingBox
# ============================================================

class BoundingBox:
    """N-dimensional axis-aligned bounding box."""

    __slots__ = ('mins', 'maxs', 'dims')

    def __init__(self, mins, maxs):
        if len(mins) != len(maxs):
            raise ValueError("mins and maxs must have same length")
        for i in range(len(mins)):
            if mins[i] > maxs[i]:
                raise ValueError(f"min > max on axis {i}: {mins[i]} > {maxs[i]}")
        self.mins = tuple(mins)
        self.maxs = tuple(maxs)
        self.dims = len(mins)

    @staticmethod
    def from_point(point):
        """Create a zero-volume bbox from a point."""
        return BoundingBox(point, point)

    def area(self):
        """Compute volume/area (product of extents)."""
        result = 1.0
        for i in range(self.dims):
            result *= (self.maxs[i] - self.mins[i])
        return result

    def margin(self):
        """Sum of edge lengths (perimeter in 2D)."""
        return sum(self.maxs[i] - self.mins[i] for i in range(self.dims))

    def center(self):
        """Center point of the bbox."""
        return tuple((self.mins[i] + self.maxs[i]) / 2.0 for i in range(self.dims))

    def intersects(self, other):
        """Check if two bboxes overlap (including touching)."""
        for i in range(self.dims):
            if self.mins[i] > other.maxs[i] or self.maxs[i] < other.mins[i]:
                return False
        return True

    def contains(self, other):
        """Check if self fully contains other."""
        for i in range(self.dims):
            if self.mins[i] > other.mins[i] or self.maxs[i] < other.maxs[i]:
                return False
        return True

    def contains_point(self, point):
        """Check if a point is inside (or on boundary of) this bbox."""
        for i in range(self.dims):
            if point[i] < self.mins[i] or point[i] > self.maxs[i]:
                return False
        return True

    def union(self, other):
        """Return the smallest bbox containing both."""
        new_mins = tuple(min(self.mins[i], other.mins[i]) for i in range(self.dims))
        new_maxs = tuple(max(self.maxs[i], other.maxs[i]) for i in range(self.dims))
        return BoundingBox(new_mins, new_maxs)

    def intersection(self, other):
        """Return the intersection bbox, or None if disjoint."""
        new_mins = tuple(max(self.mins[i], other.mins[i]) for i in range(self.dims))
        new_maxs = tuple(min(self.maxs[i], other.maxs[i]) for i in range(self.dims))
        for i in range(self.dims):
            if new_mins[i] > new_maxs[i]:
                return None
        return BoundingBox(new_mins, new_maxs)

    def enlargement(self, other):
        """Area increase needed to include other."""
        merged = self.union(other)
        return merged.area() - self.area()

    def overlap_area(self, other):
        """Area of overlap between two bboxes."""
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        return inter.area()

    def min_distance_to_point(self, point):
        """Minimum distance from point to this bbox."""
        dist_sq = 0.0
        for i in range(self.dims):
            if point[i] < self.mins[i]:
                dist_sq += (self.mins[i] - point[i]) ** 2
            elif point[i] > self.maxs[i]:
                dist_sq += (point[i] - self.maxs[i]) ** 2
        return math.sqrt(dist_sq)

    def min_distance_to_bbox(self, other):
        """Minimum distance between two bboxes."""
        dist_sq = 0.0
        for i in range(self.dims):
            if self.maxs[i] < other.mins[i]:
                dist_sq += (other.mins[i] - self.maxs[i]) ** 2
            elif other.maxs[i] < self.mins[i]:
                dist_sq += (self.mins[i] - other.maxs[i]) ** 2
        return math.sqrt(dist_sq)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return False
        return self.mins == other.mins and self.maxs == other.maxs

    def __hash__(self):
        return hash((self.mins, self.maxs))

    def __repr__(self):
        return f"BBox({list(self.mins)}, {list(self.maxs)})"


def union_all(bboxes):
    """Compute the bounding box of a list of bboxes."""
    it = iter(bboxes)
    result = next(it)
    for bb in it:
        result = result.union(bb)
    return result


# ============================================================
# R-Tree Node
# ============================================================

class RTreeEntry:
    """An entry in an R-tree node."""
    __slots__ = ('bbox', 'child', 'data_id')

    def __init__(self, bbox, child=None, data_id=None):
        self.bbox = bbox
        self.child = child  # RTreeNode for internal entries
        self.data_id = data_id  # identifier for leaf entries

    def is_leaf_entry(self):
        return self.child is None

    def __repr__(self):
        if self.is_leaf_entry():
            return f"Entry(id={self.data_id}, bbox={self.bbox})"
        return f"Entry(child={id(self.child)}, bbox={self.bbox})"


class RTreeNode:
    """A node in the R-tree."""
    __slots__ = ('entries', 'is_leaf', 'parent')

    def __init__(self, is_leaf=True):
        self.entries = []
        self.is_leaf = is_leaf
        self.parent = None  # weak uplink for deletion

    def bbox(self):
        """Compute bounding box of all entries."""
        if not self.entries:
            return None
        return union_all(e.bbox for e in self.entries)

    def __repr__(self):
        kind = "Leaf" if self.is_leaf else "Internal"
        return f"{kind}Node({len(self.entries)} entries)"


# ============================================================
# R-Tree (Classic Guttman)
# ============================================================

class RTree:
    """
    Classic R-tree (Guttman 1984).

    Parameters:
      max_entries: Maximum entries per node (M)
      min_entries: Minimum entries per node (m), default M//2
      dims: Number of dimensions
    """

    def __init__(self, max_entries=8, min_entries=None, dims=2):
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max(2, max_entries // 2)
        self.dims = dims
        self.root = RTreeNode(is_leaf=True)
        self._size = 0
        self._height = 1

    def __len__(self):
        return self._size

    @property
    def height(self):
        return self._height

    # ----------------------------------------------------------
    # Insert
    # ----------------------------------------------------------

    def insert(self, bbox, data_id=None):
        """Insert a bounding box with optional data identifier."""
        if isinstance(bbox, (list, tuple)) and not isinstance(bbox[0], (list, tuple)):
            bbox = BoundingBox.from_point(bbox)
        elif not isinstance(bbox, BoundingBox):
            bbox = BoundingBox(bbox[0], bbox[1])

        entry = RTreeEntry(bbox, data_id=data_id)
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append(entry)
        self._size += 1

        split_node = None
        if len(leaf.entries) > self.max_entries:
            split_node = self._split_node(leaf)

        self._adjust_tree(leaf, split_node)

    def _choose_leaf(self, node, bbox):
        """Choose the best leaf to insert into (least enlargement)."""
        if node.is_leaf:
            return node
        best = None
        best_enlarge = float('inf')
        best_area = float('inf')
        for entry in node.entries:
            enlarge = entry.bbox.enlargement(bbox)
            area = entry.bbox.area()
            if enlarge < best_enlarge or (enlarge == best_enlarge and area < best_area):
                best_enlarge = enlarge
                best_area = area
                best = entry
        return self._choose_leaf(best.child, bbox)

    def _split_node(self, node):
        """Quadratic split (Guttman)."""
        entries = node.entries
        # Pick seeds: maximize wasted area
        seed1, seed2 = self._pick_seeds(entries)

        group1 = [entries[seed1]]
        group2 = [entries[seed2]]
        remaining = [e for i, e in enumerate(entries) if i != seed1 and i != seed2]

        bb1 = group1[0].bbox
        bb2 = group2[0].bbox

        while remaining:
            # Check if one group needs all remaining
            if len(group1) + len(remaining) == self.min_entries:
                group1.extend(remaining)
                break
            if len(group2) + len(remaining) == self.min_entries:
                group2.extend(remaining)
                break

            # Pick next: maximize difference in enlargement
            best_idx = 0
            best_diff = -1
            for i, e in enumerate(remaining):
                d1 = bb1.enlargement(e.bbox)
                d2 = bb2.enlargement(e.bbox)
                diff = abs(d1 - d2)
                if diff > best_diff:
                    best_diff = diff
                    best_idx = i

            entry = remaining.pop(best_idx)
            d1 = bb1.enlargement(entry.bbox)
            d2 = bb2.enlargement(entry.bbox)
            if d1 < d2:
                group1.append(entry)
                bb1 = bb1.union(entry.bbox)
            elif d2 < d1:
                group2.append(entry)
                bb2 = bb2.union(entry.bbox)
            elif bb1.area() <= bb2.area():
                group1.append(entry)
                bb1 = bb1.union(entry.bbox)
            else:
                group2.append(entry)
                bb2 = bb2.union(entry.bbox)

        node.entries = group1
        new_node = RTreeNode(is_leaf=node.is_leaf)
        new_node.entries = group2

        # Update parent pointers for internal nodes
        if not node.is_leaf:
            for e in new_node.entries:
                if e.child is not None:
                    e.child.parent = new_node

        return new_node

    def _pick_seeds(self, entries):
        """Pick two entries that waste the most area together."""
        worst = float('-inf')
        s1, s2 = 0, 1
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                merged = entries[i].bbox.union(entries[j].bbox)
                waste = merged.area() - entries[i].bbox.area() - entries[j].bbox.area()
                if waste > worst:
                    worst = waste
                    s1, s2 = i, j
        return s1, s2

    def _adjust_tree(self, node, split_node):
        """Walk up, adjusting bboxes and propagating splits."""
        if node is self.root:
            if split_node is not None:
                new_root = RTreeNode(is_leaf=False)
                e1 = RTreeEntry(node.bbox(), child=node)
                e2 = RTreeEntry(split_node.bbox(), child=split_node)
                new_root.entries = [e1, e2]
                node.parent = new_root
                split_node.parent = new_root
                self.root = new_root
                self._height += 1
            return

        parent = node.parent
        # Update bbox in parent
        for e in parent.entries:
            if e.child is node:
                e.bbox = node.bbox()
                break

        new_split = None
        if split_node is not None:
            entry = RTreeEntry(split_node.bbox(), child=split_node)
            split_node.parent = parent
            parent.entries.append(entry)
            if len(parent.entries) > self.max_entries:
                new_split = self._split_node(parent)

        self._adjust_tree(parent, new_split)

    # ----------------------------------------------------------
    # Search
    # ----------------------------------------------------------

    def search(self, query_bbox):
        """Find all entries whose bbox intersects the query bbox."""
        if not isinstance(query_bbox, BoundingBox):
            query_bbox = BoundingBox(query_bbox[0], query_bbox[1])
        results = []
        self._search_node(self.root, query_bbox, results)
        return results

    def _search_node(self, node, query_bbox, results):
        if node.is_leaf:
            for entry in node.entries:
                if entry.bbox.intersects(query_bbox):
                    results.append((entry.bbox, entry.data_id))
        else:
            for entry in node.entries:
                if entry.bbox.intersects(query_bbox):
                    self._search_node(entry.child, query_bbox, results)

    def contains(self, query_bbox):
        """Find all entries fully contained within the query bbox."""
        if not isinstance(query_bbox, BoundingBox):
            query_bbox = BoundingBox(query_bbox[0], query_bbox[1])
        results = []
        self._contains_node(self.root, query_bbox, results)
        return results

    def _contains_node(self, node, query_bbox, results):
        if node.is_leaf:
            for entry in node.entries:
                if query_bbox.contains(entry.bbox):
                    results.append((entry.bbox, entry.data_id))
        else:
            for entry in node.entries:
                if entry.bbox.intersects(query_bbox):
                    self._contains_node(entry.child, query_bbox, results)

    def point_query(self, point):
        """Find all entries containing a point."""
        bb = BoundingBox.from_point(point)
        results = []
        self._point_query_node(self.root, point, bb, results)
        return results

    def _point_query_node(self, node, point, bb, results):
        if node.is_leaf:
            for entry in node.entries:
                if entry.bbox.contains_point(point):
                    results.append((entry.bbox, entry.data_id))
        else:
            for entry in node.entries:
                if entry.bbox.contains_point(point):
                    self._point_query_node(entry.child, point, bb, results)

    # ----------------------------------------------------------
    # Nearest Neighbor
    # ----------------------------------------------------------

    def nearest(self, point, k=1):
        """Find k nearest entries to a point."""
        import heapq
        best = []
        self._knn_search(self.root, point, k, best)
        result = [(-d, bb, did) for d, _, bb, did in best]
        result.sort()
        return [(bb, did, dist) for dist, bb, did in result]

    def _knn_search(self, node, point, k, best):
        import heapq
        if node.is_leaf:
            for entry in node.entries:
                dist = entry.bbox.min_distance_to_point(point)
                if len(best) < k:
                    heapq.heappush(best, (-dist, id(entry), entry.bbox, entry.data_id))
                elif dist < -best[0][0]:
                    heapq.heapreplace(best, (-dist, id(entry), entry.bbox, entry.data_id))
        else:
            # Sort children by min distance
            children = []
            for entry in node.entries:
                dist = entry.bbox.min_distance_to_point(point)
                children.append((dist, id(entry), entry))
            children.sort()

            for dist, _, entry in children:
                if len(best) >= k and dist >= -best[0][0]:
                    break
                self._knn_search(entry.child, point, k, best)

    # ----------------------------------------------------------
    # Delete
    # ----------------------------------------------------------

    def delete(self, bbox, data_id=None):
        """Delete an entry matching bbox and optionally data_id."""
        if not isinstance(bbox, BoundingBox):
            if isinstance(bbox[0], (list, tuple)):
                bbox = BoundingBox(bbox[0], bbox[1])
            else:
                bbox = BoundingBox.from_point(bbox)

        leaf, idx = self._find_leaf(self.root, bbox, data_id)
        if leaf is None:
            return False

        del leaf.entries[idx]
        self._size -= 1
        self._condense_tree(leaf)

        # Shrink root if needed
        if not self.root.is_leaf and len(self.root.entries) == 1:
            self.root = self.root.entries[0].child
            self.root.parent = None
            self._height -= 1

        return True

    def _find_leaf(self, node, bbox, data_id):
        """Find the leaf node containing the entry."""
        if node.is_leaf:
            for i, entry in enumerate(node.entries):
                if entry.bbox == bbox:
                    if data_id is None or entry.data_id == data_id:
                        return node, i
            return None, -1
        else:
            for entry in node.entries:
                if entry.bbox.intersects(bbox):
                    result = self._find_leaf(entry.child, bbox, data_id)
                    if result[0] is not None:
                        return result
            return None, -1

    def _condense_tree(self, node):
        """Walk up, removing underfull nodes and reinserting orphans."""
        orphans = []
        while node is not self.root:
            parent = node.parent
            if len(node.entries) < self.min_entries:
                # Remove from parent
                parent.entries = [e for e in parent.entries if e.child is not node]
                # Collect orphaned entries
                orphans.extend(self._collect_entries(node))
            else:
                # Update bbox in parent
                for e in parent.entries:
                    if e.child is node:
                        e.bbox = node.bbox()
                        break
            node = parent

        # Reinsert orphans
        for entry, level in orphans:
            self._reinsert_entry(entry, level)

    def _collect_entries(self, node, level=0):
        """Collect all leaf entries from a subtree with their level."""
        if node.is_leaf:
            return [(e, level) for e in node.entries]
        result = []
        for entry in node.entries:
            result.extend(self._collect_entries(entry.child, level + 1))
        return result

    def _reinsert_entry(self, entry, level):
        """Reinsert a leaf entry."""
        # For simplicity, just insert normally
        leaf = self._choose_leaf(self.root, entry.bbox)
        leaf.entries.append(entry)

        split_node = None
        if len(leaf.entries) > self.max_entries:
            split_node = self._split_node(leaf)

        self._adjust_tree(leaf, split_node)

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def all_entries(self):
        """Return all leaf entries."""
        results = []
        self._collect_all(self.root, results)
        return results

    def _collect_all(self, node, results):
        if node.is_leaf:
            for entry in node.entries:
                results.append((entry.bbox, entry.data_id))
        else:
            for entry in node.entries:
                self._collect_all(entry.child, results)

    def clear(self):
        """Remove all entries."""
        self.root = RTreeNode(is_leaf=True)
        self._size = 0
        self._height = 1

    def bbox(self):
        """Return the bounding box of the entire tree."""
        return self.root.bbox()

    def depth_stats(self):
        """Return tree statistics: height, node count, fill factor."""
        nodes = 0
        leaves = 0
        total_entries = 0
        total_capacity = 0

        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes += 1
            total_entries += len(node.entries)
            total_capacity += self.max_entries
            if node.is_leaf:
                leaves += 1
            else:
                for e in node.entries:
                    stack.append(e.child)

        fill = total_entries / total_capacity if total_capacity > 0 else 0
        return {
            'height': self._height,
            'nodes': nodes,
            'leaves': leaves,
            'entries': self._size,
            'fill_factor': round(fill, 3),
        }


# ============================================================
# R*-Tree
# ============================================================

class RStarTree(RTree):
    """
    R*-tree variant with:
    - Overlap-minimizing split (Beckmann et al. 1990)
    - Forced reinsert on overflow (30% of entries)
    - Choose subtree considers overlap for near-leaf levels
    """

    def __init__(self, max_entries=8, min_entries=None, dims=2):
        super().__init__(max_entries, min_entries, dims)
        self._reinsert_levels = set()  # track levels being reinserted this insert

    def insert(self, bbox, data_id=None):
        """Insert with R*-tree overflow handling."""
        if isinstance(bbox, (list, tuple)) and not isinstance(bbox[0], (list, tuple)):
            bbox = BoundingBox.from_point(bbox)
        elif not isinstance(bbox, BoundingBox):
            bbox = BoundingBox(bbox[0], bbox[1])

        self._reinsert_levels = set()
        entry = RTreeEntry(bbox, data_id=data_id)
        self._insert_entry(entry, target_level=0)

    def _insert_entry(self, entry, target_level=0):
        """Insert an entry at a specific level (0 = leaf)."""
        node = self._choose_subtree(self.root, entry.bbox, self._height - 1, target_level)

        if entry.child is not None:
            entry.child.parent = node

        node.entries.append(entry)
        if entry.is_leaf_entry():
            self._size += 1

        if len(node.entries) > self.max_entries:
            self._overflow_treatment(node, self._node_level(node))
        else:
            self._adjust_path(node)

    def _node_level(self, node):
        """Compute level of a node (0 = leaf)."""
        level = 0
        n = node
        while n.is_leaf is False and n.entries and n.entries[0].child:
            n = n.entries[0].child
            level += 1
        if node.is_leaf:
            return 0
        return level

    def _choose_subtree(self, node, bbox, current_level, target_level):
        """Choose subtree with overlap minimization for near-leaf levels."""
        if current_level == target_level:
            return node

        if current_level == 1:
            # Near leaf: minimize overlap
            best = None
            best_overlap_increase = float('inf')
            best_area_increase = float('inf')
            best_area = float('inf')

            for entry in node.entries:
                enlarged = entry.bbox.union(bbox)
                # Calculate overlap increase
                overlap_before = sum(
                    entry.bbox.overlap_area(other.bbox)
                    for other in node.entries if other is not entry
                )
                overlap_after = sum(
                    enlarged.overlap_area(other.bbox)
                    for other in node.entries if other is not entry
                )
                overlap_increase = overlap_after - overlap_before
                area_increase = enlarged.area() - entry.bbox.area()
                area = entry.bbox.area()

                if (overlap_increase < best_overlap_increase or
                    (overlap_increase == best_overlap_increase and
                     area_increase < best_area_increase) or
                    (overlap_increase == best_overlap_increase and
                     area_increase == best_area_increase and
                     area < best_area)):
                    best_overlap_increase = overlap_increase
                    best_area_increase = area_increase
                    best_area = area
                    best = entry
        else:
            # Higher levels: least area enlargement
            best = None
            best_enlarge = float('inf')
            best_area = float('inf')
            for entry in node.entries:
                enlarge = entry.bbox.enlargement(bbox)
                area = entry.bbox.area()
                if enlarge < best_enlarge or (enlarge == best_enlarge and area < best_area):
                    best_enlarge = enlarge
                    best_area = area
                    best = entry

        return self._choose_subtree(best.child, bbox, current_level - 1, target_level)

    def _overflow_treatment(self, node, level):
        """Handle overflow: forced reinsert or split."""
        if level > 0 and level not in self._reinsert_levels:
            self._reinsert_levels.add(level)
            self._forced_reinsert(node, level)
        else:
            split_node = self._rstar_split(node)
            self._handle_split(node, split_node)

    def _forced_reinsert(self, node, level):
        """Remove 30% farthest entries and reinsert them."""
        center = node.bbox().center()
        # Sort entries by distance from center (descending)
        dists = []
        for entry in node.entries:
            ec = entry.bbox.center()
            d = sum((ec[i] - center[i]) ** 2 for i in range(self.dims))
            dists.append((d, entry))
        dists.sort(reverse=True)

        p = max(1, int(len(node.entries) * 0.3))
        to_reinsert = [e for _, e in dists[:p]]
        node.entries = [e for _, e in dists[p:]]

        # Update parent pointers
        if not node.is_leaf:
            for e in to_reinsert:
                if e.child is not None:
                    e.child.parent = None

        # Adjust bboxes up the path
        self._adjust_path(node)

        # Reinsert removed entries
        for entry in to_reinsert:
            if entry.is_leaf_entry():
                self._size -= 1  # will be re-incremented in _insert_entry
            self._insert_entry(entry, target_level=0 if entry.is_leaf_entry() else level)

    def _rstar_split(self, node):
        """R*-tree split: choose axis by margin sum, then minimize overlap."""
        entries = node.entries
        n = len(entries)
        m = self.min_entries

        best_axis = 0
        best_margin = float('inf')

        # Choose split axis: minimize sum of margins
        for axis in range(self.dims):
            margin_sum = 0
            sorted_lo = sorted(entries, key=lambda e: e.bbox.mins[axis])
            sorted_hi = sorted(entries, key=lambda e: e.bbox.maxs[axis])

            for sort_list in [sorted_lo, sorted_hi]:
                for k in range(m, n - m + 1):
                    g1 = sort_list[:k]
                    g2 = sort_list[k:]
                    bb1 = union_all(e.bbox for e in g1)
                    bb2 = union_all(e.bbox for e in g2)
                    margin_sum += bb1.margin() + bb2.margin()

            if margin_sum < best_margin:
                best_margin = margin_sum
                best_axis = axis

        # Choose split index on best axis: minimize overlap, then area
        best_overlap = float('inf')
        best_area = float('inf')
        best_split = None

        sorted_lo = sorted(entries, key=lambda e: e.bbox.mins[best_axis])
        sorted_hi = sorted(entries, key=lambda e: e.bbox.maxs[best_axis])

        for sort_list in [sorted_lo, sorted_hi]:
            for k in range(m, n - m + 1):
                g1 = sort_list[:k]
                g2 = sort_list[k:]
                bb1 = union_all(e.bbox for e in g1)
                bb2 = union_all(e.bbox for e in g2)
                overlap = bb1.overlap_area(bb2)
                area = bb1.area() + bb2.area()
                if (overlap < best_overlap or
                    (overlap == best_overlap and area < best_area)):
                    best_overlap = overlap
                    best_area = area
                    best_split = (g1, g2)

        node.entries = list(best_split[0])
        new_node = RTreeNode(is_leaf=node.is_leaf)
        new_node.entries = list(best_split[1])

        if not node.is_leaf:
            for e in new_node.entries:
                if e.child is not None:
                    e.child.parent = new_node

        return new_node

    def _handle_split(self, node, split_node):
        """Propagate split up the tree."""
        if node is self.root:
            new_root = RTreeNode(is_leaf=False)
            e1 = RTreeEntry(node.bbox(), child=node)
            e2 = RTreeEntry(split_node.bbox(), child=split_node)
            new_root.entries = [e1, e2]
            node.parent = new_root
            split_node.parent = new_root
            self.root = new_root
            self._height += 1
            return

        parent = node.parent
        # Update bbox
        for e in parent.entries:
            if e.child is node:
                e.bbox = node.bbox()
                break

        split_entry = RTreeEntry(split_node.bbox(), child=split_node)
        split_node.parent = parent
        parent.entries.append(split_entry)

        if len(parent.entries) > self.max_entries:
            self._overflow_treatment(parent, self._node_level(parent))
        else:
            self._adjust_path(parent)

    def _adjust_path(self, node):
        """Adjust bounding boxes up to root."""
        while node is not self.root:
            parent = node.parent
            for e in parent.entries:
                if e.child is node:
                    e.bbox = node.bbox()
                    break
            node = parent


# ============================================================
# Bulk Loading: Sort-Tile-Recursive (STR)
# ============================================================

def str_bulk_load(items, max_entries=8, min_entries=None, dims=2):
    """
    Build an R-tree using Sort-Tile-Recursive bulk loading.

    items: list of (bbox, data_id) or (mins, maxs, data_id)
    Returns a populated RTree.
    """
    if not items:
        return RTree(max_entries=max_entries, min_entries=min_entries, dims=dims)

    # Normalize items
    entries = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            bbox, data_id = item
            if not isinstance(bbox, BoundingBox):
                bbox = BoundingBox.from_point(bbox) if not isinstance(bbox[0], (list, tuple)) else BoundingBox(bbox[0], bbox[1])
        elif isinstance(item, tuple) and len(item) == 3:
            bbox = BoundingBox(item[0], item[1])
            data_id = item[2]
        else:
            raise ValueError(f"Invalid item format: {item}")
        entries.append(RTreeEntry(bbox, data_id=data_id))

    tree = RTree(max_entries=max_entries, min_entries=min_entries, dims=dims)

    if len(entries) <= max_entries:
        tree.root = RTreeNode(is_leaf=True)
        tree.root.entries = entries
        tree._size = len(entries)
        tree._height = 1
        return tree

    # Build bottom-up
    leaf_nodes = _str_partition(entries, max_entries, dims)
    tree._size = len(entries)

    # Build internal levels
    current_level = leaf_nodes
    height = 1
    while len(current_level) > 1:
        height += 1
        parent_entries = [RTreeEntry(node.bbox(), child=node) for node in current_level]
        if len(parent_entries) <= max_entries:
            parent = RTreeNode(is_leaf=False)
            parent.entries = parent_entries
            for e in parent_entries:
                e.child.parent = parent
            current_level = [parent]
        else:
            parent_nodes = _str_partition_internal(parent_entries, max_entries, dims)
            current_level = parent_nodes

    tree.root = current_level[0]
    tree._height = height
    _set_parents(tree.root)
    return tree


def _str_partition(entries, max_entries, dims):
    """Partition leaf entries into nodes using STR."""
    n = len(entries)
    leaves_needed = math.ceil(n / max_entries)
    slices_per_dim = max(1, int(math.ceil(leaves_needed ** (1.0 / dims))))

    nodes = _str_recursive(entries, max_entries, dims, 0, slices_per_dim)
    return nodes


def _str_recursive(entries, max_entries, dims, axis, slices_per_dim):
    """Recursively partition entries along each axis."""
    if len(entries) <= max_entries:
        node = RTreeNode(is_leaf=True)
        node.entries = entries
        return [node]

    if axis >= dims:
        # Base case: create leaf nodes from chunks
        nodes = []
        for i in range(0, len(entries), max_entries):
            node = RTreeNode(is_leaf=True)
            node.entries = entries[i:i + max_entries]
            nodes.append(node)
        return nodes

    # Sort by center on current axis
    entries.sort(key=lambda e: e.bbox.center()[axis])

    # Divide into slices
    slice_size = max(1, math.ceil(len(entries) / slices_per_dim))
    nodes = []
    for i in range(0, len(entries), slice_size):
        slice_entries = entries[i:i + slice_size]
        nodes.extend(_str_recursive(slice_entries, max_entries, dims, axis + 1, slices_per_dim))

    return nodes


def _str_partition_internal(entries, max_entries, dims):
    """Partition internal entries into parent nodes."""
    entries.sort(key=lambda e: e.bbox.center()[0])
    nodes = []
    for i in range(0, len(entries), max_entries):
        node = RTreeNode(is_leaf=False)
        node.entries = entries[i:i + max_entries]
        for e in node.entries:
            if e.child is not None:
                e.child.parent = node
        nodes.append(node)
    return nodes


def _set_parents(node):
    """Set parent pointers recursively."""
    if not node.is_leaf:
        for e in node.entries:
            if e.child is not None:
                e.child.parent = node
                _set_parents(e.child)


# ============================================================
# Spatial Join
# ============================================================

def spatial_join(tree1, tree2):
    """
    Find all pairs of entries from tree1 and tree2 whose bboxes intersect.
    Returns list of ((bbox1, id1), (bbox2, id2)).
    """
    results = []
    if tree1.root.bbox() is None or tree2.root.bbox() is None:
        return results
    if not tree1.root.bbox().intersects(tree2.root.bbox()):
        return results
    _join_nodes(tree1.root, tree2.root, results)
    return results


def _join_nodes(node1, node2, results):
    """Recursively join two R-tree nodes."""
    if node1.is_leaf and node2.is_leaf:
        for e1 in node1.entries:
            for e2 in node2.entries:
                if e1.bbox.intersects(e2.bbox):
                    results.append(((e1.bbox, e1.data_id), (e2.bbox, e2.data_id)))
    elif node1.is_leaf:
        for e2 in node2.entries:
            if node1.bbox() is not None and e2.bbox.intersects(node1.bbox()):
                _join_nodes(node1, e2.child, results)
    elif node2.is_leaf:
        for e1 in node1.entries:
            if node2.bbox() is not None and e1.bbox.intersects(node2.bbox()):
                _join_nodes(e1.child, node2, results)
    else:
        for e1 in node1.entries:
            for e2 in node2.entries:
                if e1.bbox.intersects(e2.bbox):
                    _join_nodes(e1.child, e2.child, results)


# ============================================================
# SpatialIndex -- Unified interface
# ============================================================

class SpatialIndex:
    """
    Unified spatial index wrapping RTree or RStarTree.
    Provides a clean API for common operations.
    """

    def __init__(self, variant='rstar', max_entries=8, dims=2):
        if variant == 'rstar':
            self.tree = RStarTree(max_entries=max_entries, dims=dims)
        elif variant == 'classic':
            self.tree = RTree(max_entries=max_entries, dims=dims)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self._data = {}  # data_id -> user_data

    def insert(self, bbox, data_id, data=None):
        """Insert a bbox with an identifier and optional data."""
        if not isinstance(bbox, BoundingBox):
            if isinstance(bbox[0], (list, tuple)):
                bbox = BoundingBox(bbox[0], bbox[1])
            else:
                bbox = BoundingBox.from_point(bbox)
        self.tree.insert(bbox, data_id=data_id)
        if data is not None:
            self._data[data_id] = data

    def query(self, bbox):
        """Find all entries intersecting the query bbox."""
        return self.tree.search(bbox)

    def query_point(self, point):
        """Find all entries containing a point."""
        return self.tree.point_query(point)

    def nearest(self, point, k=1):
        """Find k nearest entries."""
        return self.tree.nearest(point, k)

    def delete(self, bbox, data_id=None):
        """Delete an entry."""
        result = self.tree.delete(bbox, data_id)
        if result and data_id is not None and data_id in self._data:
            del self._data[data_id]
        return result

    def get_data(self, data_id):
        """Get user data for an entry."""
        return self._data.get(data_id)

    def __len__(self):
        return len(self.tree)

    def stats(self):
        return self.tree.depth_stats()

    def all_entries(self):
        return self.tree.all_entries()
