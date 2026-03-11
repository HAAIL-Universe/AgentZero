"""V106: Convex Hull Computation for Polyhedra

Precise convex hull via H-V representation conversion (Double Description method).

The key insight: V105's join is approximate because it works only in H-representation
(constraints). The true convex hull requires converting to V-representation (vertices
and rays), taking the union of generators, then converting back to H-representation.

Components:
1. HPolyhedron: constraint-based representation (from V105)
2. VPolyhedron: vertex/ray-based representation (generators)
3. H-to-V conversion: vertex enumeration via constraint intersection
4. V-to-H conversion: facet enumeration via generator processing
5. Exact convex hull: V(P1) ∪ V(P2) -> H(convex_hull)
6. Comparison API: approximate vs exact join

Uses Fraction arithmetic throughout for exact computation (no floating-point).

Composes: V105 (polyhedral domain) + C010 (parser)

Author: A2 (AgentZero verification agent)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any, FrozenSet
from fractions import Fraction
from itertools import combinations
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from polyhedral_domain import (
    LinearConstraint, PolyhedralDomain, PolyhedralInterpreter,
    ZERO, ONE, INF, frac,
    polyhedral_analyze
)
from stack_vm import (
    lex, Parser, FnDecl
)

# ---------------------------------------------------------------------------
# V-Representation: Vertices and Rays
# ---------------------------------------------------------------------------

@dataclass
class Generator:
    """A generator (vertex or ray) in V-representation.

    vertex: point p such that p is in the polyhedron
    ray: direction r such that p + t*r is in the polyhedron for all t >= 0
    """
    coords: Dict[str, Fraction]  # {var_name: value}
    is_ray: bool = False         # False = vertex, True = ray

    def __hash__(self):
        return hash((tuple(sorted(self.coords.items())), self.is_ray))

    def __eq__(self, other):
        if not isinstance(other, Generator):
            return False
        return self.is_ray == other.is_ray and self.coords == other.coords

    def __repr__(self):
        kind = "ray" if self.is_ray else "vertex"
        coords_str = ", ".join(f"{v}={c}" for v, c in sorted(self.coords.items()))
        return f"{kind}({coords_str})"


@dataclass
class VPolyhedron:
    """V-representation of a polyhedron: set of vertices and rays.

    The polyhedron is the convex hull of vertices plus the conic hull of rays:
    P = {sum(lambda_i * v_i) + sum(mu_j * r_j) : lambda_i >= 0, sum(lambda_i) = 1, mu_j >= 0}
    """
    var_names: List[str]
    vertices: List[Generator]
    rays: List[Generator]

    def is_empty(self) -> bool:
        return len(self.vertices) == 0

    def is_bounded(self) -> bool:
        return len(self.rays) == 0

    def dim(self) -> int:
        return len(self.var_names)

    def add_vertex(self, coords: Dict[str, Fraction]):
        """Add a vertex."""
        full = {v: coords.get(v, ZERO) for v in self.var_names}
        g = Generator(coords=full, is_ray=False)
        if g not in self.vertices:
            self.vertices.append(g)

    def add_ray(self, coords: Dict[str, Fraction]):
        """Add a ray direction."""
        full = {v: coords.get(v, ZERO) for v in self.var_names}
        g = Generator(coords=full, is_ray=True)
        if g not in self.rays:
            self.rays.append(g)

    def __repr__(self):
        return f"VPolyhedron(vertices={self.vertices}, rays={self.rays})"


# ---------------------------------------------------------------------------
# H-to-V Conversion: Vertex Enumeration
# ---------------------------------------------------------------------------

def _solve_linear_system(equations: List[Tuple[Dict[str, Fraction], Fraction]],
                         var_names: List[str]) -> Optional[Dict[str, Fraction]]:
    """Solve a system of linear equations using Gaussian elimination.

    equations: list of (coeffs_dict, rhs) representing sum(c_i * x_i) = rhs
    Returns solution dict or None if no unique solution.
    """
    n = len(var_names)
    m = len(equations)
    if m < n:
        return None

    # Build augmented matrix [A | b] using Fractions
    matrix = []
    for coeffs, rhs in equations:
        row = [Fraction(coeffs.get(v, 0)) for v in var_names] + [Fraction(rhs)]
        matrix.append(row)

    # Gaussian elimination with partial pivoting
    pivot_row = 0
    for col in range(n):
        # Find pivot
        found = -1
        for row in range(pivot_row, m):
            if matrix[row][col] != ZERO:
                found = row
                break
        if found == -1:
            return None  # singular

        # Swap
        matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]

        # Eliminate
        pivot_val = matrix[pivot_row][col]
        for row in range(m):
            if row == pivot_row:
                continue
            factor = matrix[row][col] / pivot_val
            for j in range(n + 1):
                matrix[row][j] -= factor * matrix[pivot_row][j]

        pivot_row += 1

    # Back-substitute
    solution = {}
    for i in range(n):
        pivot_val = matrix[i][i]
        if pivot_val == ZERO:
            return None
        solution[var_names[i]] = matrix[i][n] / pivot_val

    return solution


def h_to_v(poly: PolyhedralDomain) -> VPolyhedron:
    """Convert H-representation to V-representation via vertex enumeration.

    For bounded polyhedra in d dimensions:
    - Each vertex is the intersection of exactly d tight constraints
    - Enumerate all d-subsets of constraints, solve, check feasibility

    For unbounded polyhedra:
    - Also find rays by solving homogeneous system of d-1 constraints
    """
    var_names = list(poly.var_names)
    n = len(var_names)
    vpoly = VPolyhedron(var_names=var_names, vertices=[], rays=[])

    if poly.is_bot():
        return vpoly

    # Expand equalities into pairs of inequalities for uniform treatment
    inequalities = []
    for c in poly.constraints:
        cd = c.coeffs_dict
        if c.is_equality:
            inequalities.append((cd, c.bound))          # sum <= b
            neg = {v: -coeff for v, coeff in cd.items()}
            inequalities.append((neg, -c.bound))         # -sum <= -b  (i.e. sum >= b)
        else:
            inequalities.append((cd, c.bound))

    if n == 0:
        # 0-dimensional: single point (origin)
        vpoly.add_vertex({})
        return vpoly

    # Add implicit bounds for ray detection
    # We need to handle the case where there are fewer constraints than dimensions

    # --- Vertex enumeration: intersect n constraints at equality ---
    if len(inequalities) >= n:
        for subset in combinations(range(len(inequalities)), n):
            equations = [(inequalities[i][0], inequalities[i][1]) for i in subset]
            sol = _solve_linear_system(equations, var_names)
            if sol is None:
                continue

            # Check if solution satisfies ALL inequalities
            feasible = True
            for coeffs, bound in inequalities:
                val = sum(Fraction(coeffs.get(v, 0)) * sol[v] for v in var_names)
                if val > bound + Fraction(1, 10**10):  # tiny tolerance for exact arithmetic
                    feasible = False
                    break

            if feasible:
                vpoly.add_vertex(sol)

    # --- Ray enumeration: solve n-1 constraints at equality, find direction ---
    if n >= 1 and len(inequalities) >= n - 1:
        for subset in combinations(range(len(inequalities)), max(n - 1, 0)):
            if n == 1:
                # 1D: check if unbounded above or below
                has_upper = any(inequalities[i][0].get(var_names[0], ZERO) > ZERO for i in range(len(inequalities)))
                has_lower = any(inequalities[i][0].get(var_names[0], ZERO) < ZERO for i in range(len(inequalities)))
                if not has_upper:
                    vpoly.add_ray({var_names[0]: ONE})
                if not has_lower:
                    vpoly.add_ray({var_names[0]: -ONE})
                break

            # Build n-1 equations
            equations = [(inequalities[i][0], ZERO) for i in subset]  # homogeneous

            # Find nullspace direction: solve with one var free
            for free_idx in range(n):
                # Set free var to 1, solve rest
                remaining_vars = [v for j, v in enumerate(var_names) if j != free_idx]
                modified_eqs = []
                for coeffs, _ in equations:
                    new_rhs = -Fraction(coeffs.get(var_names[free_idx], 0))
                    new_coeffs = {v: Fraction(coeffs.get(v, 0)) for v in remaining_vars}
                    modified_eqs.append((new_coeffs, new_rhs))

                sol = _solve_linear_system(modified_eqs, remaining_vars)
                if sol is not None:
                    ray_dir = {v: sol.get(v, ZERO) for v in var_names}
                    ray_dir[var_names[free_idx]] = ONE

                    # Check if ray direction satisfies all constraints (a*r <= 0)
                    valid = True
                    for coeffs, bound in inequalities:
                        dot = sum(Fraction(coeffs.get(v, 0)) * ray_dir[v] for v in var_names)
                        if dot > ZERO:
                            # Try negated direction
                            break
                    else:
                        if any(ray_dir[v] != ZERO for v in var_names):
                            vpoly.add_ray(ray_dir)
                        continue

                    # Try negated
                    neg_dir = {v: -ray_dir[v] for v in var_names}
                    valid_neg = True
                    for coeffs, bound in inequalities:
                        dot = sum(Fraction(coeffs.get(v, 0)) * neg_dir[v] for v in var_names)
                        if dot > ZERO:
                            valid_neg = False
                            break
                    if valid_neg and any(neg_dir[v] != ZERO for v in var_names):
                        vpoly.add_ray(neg_dir)

                    break

    # Deduplicate vertices (exact arithmetic means exact comparison)
    _deduplicate_generators(vpoly)

    return vpoly


def _deduplicate_generators(vpoly: VPolyhedron):
    """Remove duplicate vertices and normalize rays."""
    # Deduplicate vertices
    seen = set()
    unique_verts = []
    for v in vpoly.vertices:
        key = tuple(sorted(v.coords.items()))
        if key not in seen:
            seen.add(key)
            unique_verts.append(v)
    vpoly.vertices = unique_verts

    # Normalize and deduplicate rays (scale so GCD of coeffs = 1)
    seen_rays = set()
    unique_rays = []
    for r in vpoly.rays:
        coeffs = [r.coords[v] for v in vpoly.var_names]
        nonzero = [abs(c) for c in coeffs if c != ZERO]
        if not nonzero:
            continue
        # Normalize: divide by GCD, make first nonzero positive
        from math import gcd
        from functools import reduce
        nums = [abs(c.numerator) for c in coeffs if c != ZERO]
        dens = [c.denominator for c in coeffs if c != ZERO]
        g = reduce(gcd, nums)
        # Scale so first nonzero is positive
        first_nonzero = next(c for c in coeffs if c != ZERO)
        sign = 1 if first_nonzero > 0 else -1
        normalized = {vpoly.var_names[i]: Fraction(sign * coeffs[i].numerator, coeffs[i].denominator)
                      for i in range(len(vpoly.var_names))}
        key = tuple(sorted(normalized.items()))
        if key not in seen_rays:
            seen_rays.add(key)
            unique_rays.append(Generator(coords=normalized, is_ray=True))
    vpoly.rays = unique_rays


# ---------------------------------------------------------------------------
# V-to-H Conversion: Facet Enumeration
# ---------------------------------------------------------------------------

def v_to_h(vpoly: VPolyhedron) -> PolyhedralDomain:
    """Convert V-representation to H-representation.

    Given vertices and rays, find the minimal set of linear constraints
    that define the same polyhedron.

    Uses the approach:
    1. For bounded polyhedra: compute supporting hyperplanes from vertex subsets
    2. For each candidate hyperplane, verify all vertices/rays satisfy it
    3. Keep only irredundant constraints
    """
    var_names = list(vpoly.var_names)
    n = len(var_names)

    if vpoly.is_empty():
        return PolyhedralDomain.bot(var_names)

    poly = PolyhedralDomain(var_names=list(var_names))

    if n == 0:
        return poly

    all_vertices = vpoly.vertices
    all_rays = vpoly.rays

    if n == 1:
        return _v_to_h_1d(vpoly)

    # General case: find facet-defining hyperplanes
    # A hyperplane a*x = b is facet-defining if:
    # 1. All vertices satisfy a*x <= b
    # 2. All rays satisfy a*r <= 0
    # 3. At least n-1 affinely independent vertices lie ON it (tight)

    constraints = _find_facets(vpoly)

    for coeffs_dict, bound, is_eq in constraints:
        poly.add_constraint(coeffs_dict, bound, is_equality=is_eq)

    return poly


def _v_to_h_1d(vpoly: VPolyhedron) -> PolyhedralDomain:
    """Convert 1D V-representation to H."""
    var = vpoly.var_names[0]
    poly = PolyhedralDomain(var_names=[var])

    if not vpoly.vertices:
        return PolyhedralDomain.bot([var])

    values = [v.coords[var] for v in vpoly.vertices]
    lo, hi = min(values), max(values)

    has_pos_ray = any(r.coords[var] > ZERO for r in vpoly.rays)
    has_neg_ray = any(r.coords[var] < ZERO for r in vpoly.rays)

    if not has_neg_ray:
        poly.set_lower(var, lo)
    if not has_pos_ray:
        poly.set_upper(var, hi)

    return poly


def _find_facets(vpoly: VPolyhedron) -> List[Tuple[Dict[str, Fraction], Fraction, bool]]:
    """Find facet-defining hyperplanes for a V-representation polyhedron.

    For n dimensions, a facet is defined by n-1 affinely independent points
    (or n-1 generators including rays). We enumerate all (n-1)-subsets of
    generators, compute the normal to their affine hull, then check which
    side all other generators fall on.

    Returns list of (coeffs_dict, bound, is_equality) tuples.
    """
    var_names = vpoly.var_names
    n = len(var_names)
    all_verts = vpoly.vertices
    all_rays = vpoly.rays

    if not all_verts:
        return []

    facets = []
    found_normals = set()

    all_generators = all_verts + all_rays
    num_gens = len(all_generators)

    if num_gens <= n:
        return _bounding_constraints(vpoly)

    # A facet in n-D is defined by n generators (1 anchor + n-1 direction vectors).
    # Enumerate all n-subsets of generators to find candidate hyperplanes.
    for subset_indices in combinations(range(num_gens), n):
        subset = [all_generators[i] for i in subset_indices]

        # Need a vertex as anchor (for computing the bound b in a*x = b)
        anchor = None
        for g in subset:
            if not g.is_ray:
                anchor = g
                break
        if anchor is None:
            # All rays -- use any vertex as anchor
            anchor = all_verts[0]

        # Build n-1 direction vectors relative to anchor
        diff_vecs = []
        for g in subset:
            if g is anchor:
                continue
            if g.is_ray:
                diff = {v: g.coords.get(v, ZERO) for v in var_names}
            else:
                diff = {v: g.coords.get(v, ZERO) - anchor.coords.get(v, ZERO) for v in var_names}
            diff_vecs.append(diff)

        if len(diff_vecs) < n - 1:
            continue

        # Compute normal orthogonal to all direction vectors
        normal = _compute_normal(diff_vecs[:n-1], var_names)
        if normal is None:
            continue

        # Compute bound from anchor: a * anchor = b
        bound = sum(normal.get(v, ZERO) * anchor.coords.get(v, ZERO) for v in var_names)

        # Check which side all generators fall on
        all_leq = True
        all_geq = True
        for g in all_generators:
            if g.is_ray:
                dot = sum(normal.get(v, ZERO) * g.coords.get(v, ZERO) for v in var_names)
                if dot > ZERO:
                    all_leq = False
                if dot < ZERO:
                    all_geq = False
            else:
                val = sum(normal.get(v, ZERO) * g.coords.get(v, ZERO) for v in var_names)
                if val > bound:
                    all_leq = False
                if val < bound:
                    all_geq = False

            if not all_leq and not all_geq:
                break

        if all_leq:
            norm_key = _normalize_normal(normal, var_names)
            if norm_key not in found_normals:
                found_normals.add(norm_key)
                facets.append((dict(normal), bound, False))
        elif all_geq:
            neg_normal = {v: -c for v, c in normal.items()}
            norm_key = _normalize_normal(neg_normal, var_names)
            if norm_key not in found_normals:
                found_normals.add(norm_key)
                facets.append((neg_normal, -bound, False))

    return facets


def _compute_normal(diff_vecs: List[Dict[str, Fraction]], var_names: List[str]) -> Optional[Dict[str, Fraction]]:
    """Compute normal vector orthogonal to all difference vectors.

    Solves: dot(normal, diff_i) = 0 for all i.
    Uses nullspace computation.
    """
    n = len(var_names)
    m = len(diff_vecs)

    if m < n - 1:
        return None

    # Build matrix A where each row is a diff vector
    A = []
    for dv in diff_vecs[:n-1]:
        row = [Fraction(dv.get(v, 0)) for v in var_names]
        A.append(row)

    # Find nullspace of A (dimension should be 1 for a hyperplane)
    # Use Gaussian elimination to row-reduce, then read off nullspace

    # Row reduce
    matrix = [row[:] for row in A]
    pivot_cols = []
    pivot_row = 0

    for col in range(n):
        if pivot_row >= len(matrix):
            break
        # Find pivot
        found = -1
        for row in range(pivot_row, len(matrix)):
            if matrix[row][col] != ZERO:
                found = row
                break
        if found == -1:
            continue

        matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
        pivot_cols.append(col)

        # Scale pivot row
        scale = matrix[pivot_row][col]
        for j in range(n):
            matrix[pivot_row][j] /= scale

        # Eliminate
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][col]
            for j in range(n):
                matrix[row][j] -= factor * matrix[pivot_row][j]

        pivot_row += 1

    # Free variables: columns not in pivot_cols
    free_cols = [c for c in range(n) if c not in pivot_cols]

    if len(free_cols) == 0:
        return None  # No nullspace (overdetermined)

    # Set first free variable to 1, others to 0
    free_col = free_cols[0]
    normal_vec = [ZERO] * n
    normal_vec[free_col] = ONE

    # Back-substitute: for each pivot column, compute value
    for i, pc in enumerate(pivot_cols):
        val = ZERO
        for fc in free_cols:
            if fc == free_col:
                val -= matrix[i][fc] * ONE
            # else: other free vars are 0
        normal_vec[pc] = val

    result = {var_names[i]: normal_vec[i] for i in range(n) if normal_vec[i] != ZERO}

    if not result:
        return None

    return result


def _normalize_normal(normal: Dict[str, Fraction], var_names: List[str]) -> tuple:
    """Normalize a normal vector for deduplication.

    Preserves sign (direction matters for <= vs >=).
    Scales so GCD of absolute numerators = 1 for canonical form.
    """
    coeffs = [normal.get(v, ZERO) for v in var_names]
    nonzero = [(i, c) for i, c in enumerate(coeffs) if c != ZERO]
    if not nonzero:
        return ()

    from math import gcd
    from functools import reduce
    nums = [abs(c.numerator) for _, c in nonzero]
    g = reduce(gcd, nums) if nums else 1

    # Scale down by GCD but preserve sign
    scaled = tuple(Fraction(c.numerator, c.denominator * g) if c != ZERO else ZERO for c in coeffs)
    return scaled


def _bounding_constraints(vpoly: VPolyhedron) -> List[Tuple[Dict[str, Fraction], Fraction, bool]]:
    """Generate bounding box constraints from vertices/rays."""
    constraints = []
    for v in vpoly.var_names:
        values = [vert.coords[v] for vert in vpoly.vertices]
        if not values:
            continue
        lo, hi = min(values), max(values)

        has_pos_ray = any(r.coords.get(v, ZERO) > ZERO for r in vpoly.rays)
        has_neg_ray = any(r.coords.get(v, ZERO) < ZERO for r in vpoly.rays)

        if not has_pos_ray:
            constraints.append(({v: ONE}, hi, False))
        if not has_neg_ray:
            constraints.append(({v: -ONE}, -lo, False))
    return constraints


# ---------------------------------------------------------------------------
# Convex Hull: The Core Operation
# ---------------------------------------------------------------------------

def convex_hull_v(vp1: VPolyhedron, vp2: VPolyhedron) -> VPolyhedron:
    """Compute convex hull of two V-polyhedra by taking union of generators."""
    all_vars = sorted(set(vp1.var_names) | set(vp2.var_names))
    result = VPolyhedron(var_names=all_vars, vertices=[], rays=[])

    for v in vp1.vertices:
        coords = {var: v.coords.get(var, ZERO) for var in all_vars}
        result.add_vertex(coords)
    for v in vp2.vertices:
        coords = {var: v.coords.get(var, ZERO) for var in all_vars}
        result.add_vertex(coords)

    for r in vp1.rays:
        coords = {var: r.coords.get(var, ZERO) for var in all_vars}
        result.add_ray(coords)
    for r in vp2.rays:
        coords = {var: r.coords.get(var, ZERO) for var in all_vars}
        result.add_ray(coords)

    return result


def convex_hull(p1: PolyhedralDomain, p2: PolyhedralDomain) -> PolyhedralDomain:
    """Compute exact convex hull of two H-polyhedra.

    Pipeline: H1, H2 -> V1, V2 -> V_union -> H_result
    """
    if p1.is_bot():
        return p2.copy()
    if p2.is_bot():
        return p1.copy()

    # Convert to V-representation
    v1 = h_to_v(p1)
    v2 = h_to_v(p2)

    # Union of generators
    v_union = convex_hull_v(v1, v2)

    # Convert back to H-representation
    return v_to_h(v_union)


# ---------------------------------------------------------------------------
# Minkowski Sum
# ---------------------------------------------------------------------------

def minkowski_sum(p1: PolyhedralDomain, p2: PolyhedralDomain) -> PolyhedralDomain:
    """Compute Minkowski sum: P1 + P2 = {a + b : a in P1, b in P2}.

    In V-representation: vertices = {v1 + v2}, rays = union of rays.
    """
    if p1.is_bot() or p2.is_bot():
        all_vars = sorted(set(p1.var_names) | set(p2.var_names))
        return PolyhedralDomain.bot(all_vars)

    v1 = h_to_v(p1)
    v2 = h_to_v(p2)

    all_vars = sorted(set(v1.var_names) | set(v2.var_names))
    result = VPolyhedron(var_names=all_vars, vertices=[], rays=[])

    # Sum all vertex pairs
    for va in v1.vertices:
        for vb in v2.vertices:
            coords = {v: va.coords.get(v, ZERO) + vb.coords.get(v, ZERO) for v in all_vars}
            result.add_vertex(coords)

    # Union of rays
    for r in v1.rays:
        coords = {v: r.coords.get(v, ZERO) for v in all_vars}
        result.add_ray(coords)
    for r in v2.rays:
        coords = {v: r.coords.get(v, ZERO) for v in all_vars}
        result.add_ray(coords)

    return v_to_h(result)


# ---------------------------------------------------------------------------
# Intersection (dual of convex hull -- in H it's just conjunction)
# ---------------------------------------------------------------------------

def intersection(p1: PolyhedralDomain, p2: PolyhedralDomain) -> PolyhedralDomain:
    """Compute intersection of two polyhedra (equivalent to p1.meet(p2))."""
    return p1.meet(p2)


# ---------------------------------------------------------------------------
# Projection (existential quantification)
# ---------------------------------------------------------------------------

def project(poly: PolyhedralDomain, keep_vars: List[str]) -> PolyhedralDomain:
    """Project polyhedron onto a subset of variables (forget the rest)."""
    result = poly.copy()
    to_remove = [v for v in poly.var_names if v not in keep_vars]
    for v in to_remove:
        result.forget(v)
    return result


# ---------------------------------------------------------------------------
# Inclusion Check
# ---------------------------------------------------------------------------

def is_subset(p1: PolyhedralDomain, p2: PolyhedralDomain) -> bool:
    """Check if P1 ⊆ P2 (every point in P1 is also in P2)."""
    return p1.leq(p2)


# ---------------------------------------------------------------------------
# Exact Join for PolyhedralInterpreter
# ---------------------------------------------------------------------------

class ExactJoinPolyhedralDomain(PolyhedralDomain):
    """PolyhedralDomain with exact convex hull join instead of approximate."""

    def _wrap(self, poly):
        """Wrap a PolyhedralDomain result as ExactJoinPolyhedralDomain."""
        result = ExactJoinPolyhedralDomain(list(poly.var_names))
        result.constraints = list(poly.constraints)
        result._is_bot = poly._is_bot
        return result

    def join(self, other):
        """Exact convex hull join."""
        if self.is_bot():
            return self._wrap(other)
        if other.is_bot():
            return self._wrap(self)

        hull = convex_hull(self, other)
        return self._wrap(hull)

    def copy(self):
        result = ExactJoinPolyhedralDomain(list(self.var_names))
        result.constraints = [c for c in self.constraints]
        result._is_bot = self._is_bot
        return result

    def widen(self, other):
        """Widening that preserves ExactJoinPolyhedralDomain type."""
        result = super().widen(other)
        return self._wrap(result)

    def meet(self, other):
        """Meet that preserves ExactJoinPolyhedralDomain type."""
        result = super().meet(other)
        return self._wrap(result)


# ---------------------------------------------------------------------------
# Comparison: Approximate vs Exact Join
# ---------------------------------------------------------------------------

def compare_joins(p1: PolyhedralDomain, p2: PolyhedralDomain) -> Dict[str, Any]:
    """Compare V105's approximate join with V106's exact convex hull.

    Returns metrics showing precision improvement.
    """
    approx = p1.join(p2)
    exact = convex_hull(p1, p2)

    var_names = sorted(set(p1.var_names) | set(p2.var_names))

    result = {
        'approximate': {},
        'exact': {},
        'precision_gains': {},
        'approx_constraints': len(approx.constraints),
        'exact_constraints': len(exact.constraints),
    }

    for v in var_names:
        a_lo, a_hi = _safe_interval(approx, v)
        e_lo, e_hi = _safe_interval(exact, v)
        result['approximate'][v] = (a_lo, a_hi)
        result['exact'][v] = (e_lo, e_hi)

        # Precision gain: how much tighter is the exact interval?
        a_width = a_hi - a_lo if a_hi != float('inf') and a_lo != float('-inf') else float('inf')
        e_width = e_hi - e_lo if e_hi != float('inf') and e_lo != float('-inf') else float('inf')

        if a_width == float('inf') and e_width != float('inf'):
            result['precision_gains'][v] = 'inf->bounded'
        elif a_width > 0 and e_width < a_width:
            reduction = float((a_width - e_width) / a_width * 100) if a_width > 0 else 0
            result['precision_gains'][v] = f'{reduction:.1f}% tighter'
        else:
            result['precision_gains'][v] = 'same'

    # Check if exact is strictly tighter (by variable bounds)
    exact_is_tighter = False
    for v in var_names:
        a_lo, a_hi = result['approximate'].get(v, (float('-inf'), float('inf')))
        e_lo, e_hi = result['exact'].get(v, (float('-inf'), float('inf')))
        if e_lo > a_lo or e_hi < a_hi:
            exact_is_tighter = True
            break
    result['exact_is_tighter'] = exact_is_tighter

    # Soundness check: verify all vertices of inputs are contained in both joins
    def _vertices_contained(poly, hull):
        vp = h_to_v(poly)
        for v in vp.vertices:
            for c in hull.constraints:
                val = c.evaluate({vn: v.coords.get(vn, ZERO) for vn in hull.var_names})
                if c.is_equality:
                    if abs(val - c.bound) > Fraction(1, 100):
                        return False
                else:
                    if val > c.bound + Fraction(1, 100):
                        return False
        return True

    result['both_sound'] = (_vertices_contained(p1, approx) and
                            _vertices_contained(p2, approx) and
                            _vertices_contained(p1, exact) and
                            _vertices_contained(p2, exact))

    return result


def _safe_interval(poly, var):
    """Get interval for a variable, handling missing vars."""
    if var not in poly.var_names:
        return (float('-inf'), float('inf'))
    return (poly.get_lower(var), poly.get_upper(var))


# ---------------------------------------------------------------------------
# Widening Variants
# ---------------------------------------------------------------------------

def widening_with_thresholds(p1: PolyhedralDomain, p2: PolyhedralDomain,
                              thresholds: List[Fraction] = None) -> PolyhedralDomain:
    """Widening with thresholds: instead of dropping constraints entirely,
    relax them to the nearest threshold value.

    Standard widening drops any constraint from p1 not satisfied by p2.
    Threshold widening relaxes to the next threshold instead of dropping.
    """
    if thresholds is None:
        thresholds = [Fraction(v) for v in [-100, -10, -1, 0, 1, 10, 100]]
    thresholds = sorted(thresholds)

    if p1.is_bot():
        return p2.copy()
    if p2.is_bot():
        return p1.copy()

    result = PolyhedralDomain(var_names=list(p1.var_names))

    # Expand equalities
    p1_ineqs = _expand_equalities(p1)
    p2_ineqs = _expand_equalities(p2)

    for c in p1_ineqs:
        # Check if p2 satisfies this constraint
        satisfied = _constraint_satisfied_by(c, p2_ineqs)

        if satisfied:
            result.constraints.append(c)
        else:
            # Instead of dropping, find the next threshold
            # Evaluate constraint on p2's extreme points
            max_violation = _max_constraint_value(c, p2)
            if max_violation is not None and max_violation != float('inf'):
                # Find next threshold >= max_violation
                next_t = None
                for t in thresholds:
                    if t >= max_violation:
                        next_t = t
                        break
                if next_t is not None:
                    relaxed = LinearConstraint.from_dict(c.coeffs_dict, next_t)
                    result.constraints.append(relaxed)
                # else: drop (no threshold large enough)

    return result


def delayed_widening(p1: PolyhedralDomain, p2: PolyhedralDomain,
                     iteration: int, delay: int = 3) -> PolyhedralDomain:
    """Delayed widening: use exact join for the first 'delay' iterations,
    then switch to standard widening.

    This allows the analysis to discover constraints before widening drops them.
    """
    if iteration < delay:
        return convex_hull(p1, p2)
    else:
        return p1.widen(p2)


def _expand_equalities(poly: PolyhedralDomain) -> List[LinearConstraint]:
    """Expand equality constraints into pairs of inequalities."""
    result = []
    for c in poly.constraints:
        if c.is_equality:
            result.append(LinearConstraint.from_dict(c.coeffs_dict, c.bound))
            neg = {v: -coeff for v, coeff in c.coeffs_dict.items()}
            result.append(LinearConstraint.from_dict(neg, -c.bound))
        else:
            result.append(c)
    return result


def _constraint_satisfied_by(c: LinearConstraint, constraints: List[LinearConstraint]) -> bool:
    """Check if constraint c is implied by the given constraint set (heuristic)."""
    cd = c.coeffs_dict
    for other in constraints:
        if other.coeffs_dict == cd and other.bound <= c.bound:
            return True
    return False


def _max_constraint_value(c: LinearConstraint, poly: PolyhedralDomain) -> Optional[float]:
    """Compute the maximum value of the constraint's LHS over the polyhedron."""
    # Use V-representation to evaluate
    vpoly = h_to_v(poly)
    if vpoly.is_empty():
        return None

    max_val = float('-inf')
    for v in vpoly.vertices:
        val = sum(c.coeffs_dict.get(vn, ZERO) * v.coords.get(vn, ZERO)
                  for vn in poly.var_names)
        max_val = max(max_val, float(val))

    # If there are rays in the constraint direction, the max is unbounded
    for r in vpoly.rays:
        dot = sum(c.coeffs_dict.get(vn, ZERO) * r.coords.get(vn, ZERO)
                  for vn in poly.var_names)
        if dot > ZERO:
            return float('inf')

    return max_val


# ---------------------------------------------------------------------------
# C10 Integration: Exact-Join Interpreter
# ---------------------------------------------------------------------------

class ExactJoinInterpreter(PolyhedralInterpreter):
    """PolyhedralInterpreter that uses exact convex hull for joins.

    At every if-else join and loop widening point, uses V106's exact
    convex hull instead of V105's approximate constraint intersection.

    Works by overriding analyze() to start with ExactJoinPolyhedralDomain,
    which preserves itself through copy/join/widen/meet operations.
    """

    def analyze(self, source: str) -> Dict[str, Any]:
        """Analyze with exact joins."""
        tokens = lex(source)
        ast = Parser(tokens).parse()

        env = ExactJoinPolyhedralDomain([])

        for stmt in ast.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt
            else:
                env = self._interpret_stmt(stmt, env, self.functions)
                if env.is_bot():
                    break

        # Build ranges dict for compatibility
        ranges = {}
        for v in env.var_names:
            ranges[v] = (env.get_lower(v), env.get_upper(v))

        return {
            'ranges': ranges,
            'constraints': env.get_constraints(),
            'relational': env.get_relational_constraints(),
            'env': env,
        }


# ---------------------------------------------------------------------------
# Polyhedral Operations: Volume Estimation
# ---------------------------------------------------------------------------

def estimate_volume(poly: PolyhedralDomain) -> Optional[float]:
    """Estimate the volume of a bounded polyhedron using its vertices.

    For a bounded polyhedron, decomposes into simplices from a central point
    and sums their volumes.

    Returns None for unbounded polyhedra.
    """
    vpoly = h_to_v(poly)

    if vpoly.is_empty():
        return 0.0

    if not vpoly.is_bounded():
        return None

    n = len(vpoly.var_names)
    verts = vpoly.vertices

    if n == 0:
        return 1.0  # A point has "volume" 1

    if n == 1:
        values = [float(v.coords[vpoly.var_names[0]]) for v in verts]
        return max(values) - min(values)

    if len(verts) < n + 1:
        return 0.0  # Degenerate (lower-dimensional)

    # Compute centroid
    centroid = {}
    for var in vpoly.var_names:
        centroid[var] = sum(v.coords[var] for v in verts) / len(verts)

    # Triangulate from centroid using numpy for determinant computation
    total_volume = Fraction(0)

    # For 2D: sum triangle areas from centroid to each edge
    if n == 2:
        v0, v1 = vpoly.var_names
        # Sort vertices by angle from centroid
        import math
        def angle(vert):
            dx = float(vert.coords[v0] - centroid[v0])
            dy = float(vert.coords[v1] - centroid[v1])
            return math.atan2(dy, dx)

        sorted_verts = sorted(verts, key=angle)

        for i in range(len(sorted_verts)):
            a = sorted_verts[i]
            b = sorted_verts[(i + 1) % len(sorted_verts)]

            # Triangle area = 0.5 * |det([[ax-cx, bx-cx], [ay-cy, by-cy]])|
            dx1 = a.coords[v0] - centroid[v0]
            dy1 = a.coords[v1] - centroid[v1]
            dx2 = b.coords[v0] - centroid[v0]
            dy2 = b.coords[v1] - centroid[v1]

            area = abs(dx1 * dy2 - dx2 * dy1) / 2
            total_volume += area

        return float(total_volume)

    # General n-D: use simplex decomposition
    # Volume of simplex = |det(M)| / n! where M has edge vectors as rows
    from math import factorial

    for subset in combinations(range(len(verts)), n):
        # Form simplex with centroid + n vertices
        matrix = []
        for idx in subset:
            row = [float(verts[idx].coords[v] - centroid[v]) for v in vpoly.var_names]
            matrix.append(row)

        try:
            det = abs(np.linalg.det(np.array(matrix, dtype=float)))
            total_volume += Fraction(det) / factorial(n)
        except (np.linalg.LinAlgError, ValueError):
            pass

    return float(total_volume)


# ---------------------------------------------------------------------------
# Polyhedral Operations: Affine Image/Pre-image
# ---------------------------------------------------------------------------

def affine_image(poly: PolyhedralDomain, target: str,
                 coeffs: Dict[str, Fraction], constant: Fraction) -> PolyhedralDomain:
    """Compute affine image: target := sum(c_i * x_i) + constant.

    Equivalent to assign_expr but exposed as a standalone operation.
    """
    result = poly.copy()
    result.assign_expr(target, {v: Fraction(c) for v, c in coeffs.items()}, Fraction(constant))
    return result


def affine_preimage(poly: PolyhedralDomain, target: str,
                    coeffs: Dict[str, Fraction], constant: Fraction) -> PolyhedralDomain:
    """Compute affine pre-image: find points that map INTO poly under target := expr.

    If target := a*x + b*y + c, then pre-image replaces target in each constraint
    with (a*x + b*y + c).
    """
    result = PolyhedralDomain(var_names=list(poly.var_names))

    for con in poly.constraints:
        cd = con.coeffs_dict
        if target not in cd:
            result.constraints.append(con)
            continue

        # Replace target with the expression
        target_coeff = cd[target]
        new_coeffs = {v: c for v, c in cd.items() if v != target}

        for v, c in coeffs.items():
            new_coeffs[v] = new_coeffs.get(v, ZERO) + target_coeff * Fraction(c)

        new_bound = con.bound - target_coeff * Fraction(constant)

        result.constraints.append(
            LinearConstraint.from_dict(new_coeffs, new_bound, con.is_equality)
        )

    return result


# ---------------------------------------------------------------------------
# Top-Level APIs
# ---------------------------------------------------------------------------

def exact_convex_hull(p1: PolyhedralDomain, p2: PolyhedralDomain) -> PolyhedralDomain:
    """Compute exact convex hull of two polyhedra (main API)."""
    return convex_hull(p1, p2)


def convert_to_vertices(poly: PolyhedralDomain) -> VPolyhedron:
    """Convert H-representation to V-representation."""
    return h_to_v(poly)


def convert_to_constraints(vpoly: VPolyhedron) -> PolyhedralDomain:
    """Convert V-representation to H-representation."""
    return v_to_h(vpoly)


def exact_analyze(source: str) -> Dict[str, Any]:
    """Analyze a C10 program using exact-join polyhedral analysis."""
    interp = ExactJoinInterpreter()
    return interp.analyze(source)


def compare_analyses(source: str) -> Dict[str, Any]:
    """Compare approximate vs exact polyhedral analysis on a C10 program."""
    approx_result = polyhedral_analyze(source)
    exact_result = exact_analyze(source)

    all_vars = sorted(set(list(approx_result.get('ranges', {}).keys()) +
                         list(exact_result.get('ranges', {}).keys())))

    comparison = {}
    for v in all_vars:
        a_range = approx_result.get('ranges', {}).get(v, (float('-inf'), float('inf')))
        e_range = exact_result.get('ranges', {}).get(v, (float('-inf'), float('inf')))
        comparison[v] = {
            'approximate': a_range,
            'exact': e_range,
            'improved': e_range != a_range
        }

    return {
        'approximate': approx_result,
        'exact': exact_result,
        'comparison': comparison,
        'any_improvement': any(c['improved'] for c in comparison.values())
    }


def convex_hull_summary(p1: PolyhedralDomain, p2: PolyhedralDomain) -> str:
    """Human-readable summary of convex hull computation."""
    v1 = h_to_v(p1)
    v2 = h_to_v(p2)
    hull = convex_hull(p1, p2)
    comp = compare_joins(p1, p2)

    lines = ["=== V106 Convex Hull Summary ==="]
    lines.append(f"P1: {len(p1.constraints)} constraints, {len(v1.vertices)} vertices, {len(v1.rays)} rays")
    lines.append(f"P2: {len(p2.constraints)} constraints, {len(v2.vertices)} vertices, {len(v2.rays)} rays")
    lines.append(f"Hull: {len(hull.constraints)} constraints")
    lines.append(f"Approximate join: {comp['approx_constraints']} constraints")
    lines.append(f"Exact is tighter: {comp['exact_is_tighter']}")
    lines.append(f"Both sound: {comp['both_sound']}")

    if comp['precision_gains']:
        lines.append("\nPrecision gains:")
        for v, gain in sorted(comp['precision_gains'].items()):
            lines.append(f"  {v}: {gain}")

    return "\n".join(lines)
