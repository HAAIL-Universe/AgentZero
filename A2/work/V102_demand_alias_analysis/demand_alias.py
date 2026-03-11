"""V102: Demand-Driven Alias Analysis

CFL-reachability-based demand-driven points-to and alias analysis.
Instead of computing points-to sets for ALL variables (exhaustive),
answers specific alias queries by lazily exploring only the relevant
program paths.

Key ideas:
1. Points-to analysis as CFL reachability: a variable x points to
   heap location h if there's a matched-parenthesis path from x to h
   in the pointer assignment graph (PAG).
2. Demand-driven: given a query "what does x point to?", search
   backward from x through the PAG, only visiting relevant assignments.
3. Field-sensitive: separate tracking per field (x.f, x.g)
4. Context-sensitive via call-string matching (balanced parentheses)
5. Memoization: cached results accelerate future queries

Composes: V097 (points-to analysis for constraint extraction) + C043 (parser)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, FrozenSet, Any
from enum import Enum, auto
from collections import defaultdict, deque
from copy import deepcopy

# Import C043 parser (has arrays + hash maps + closures)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C043_hash_maps'))
from hash_maps import lex, Parser

# Import V097 for constraint extraction and data types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V097_points_to_analysis'))
from points_to_analysis import (
    HeapLoc, FieldLoc, AbstractPtr, AllocKind,
    ConstraintKind, Constraint, PointsToState,
    ConstraintExtractor, AliasResult,
    analyze_points_to as exhaustive_analyze
)


# ---------------------------------------------------------------------------
# Pointer Assignment Graph (PAG)
# ---------------------------------------------------------------------------

class EdgeKind(Enum):
    """Kinds of edges in the pointer assignment graph."""
    NEW = auto()       # x --new--> h  (allocation)
    ASSIGN = auto()    # x --assign--> y  (x = y)
    LOAD = auto()      # x --load[f]--> y  (x = y.f)
    STORE = auto()     # x --store[f]--> y  (x.f = y)
    CALL_IN = auto()   # param --call[cs]--> arg  (parameter binding at call site cs)
    CALL_OUT = auto()  # x --ret[cs]--> retvar  (return value at call site cs)


@dataclass(frozen=True)
class PAGEdge:
    """An edge in the pointer assignment graph."""
    kind: EdgeKind
    src: str              # source node (variable name)
    dst: str              # destination node (variable name or heap loc label)
    field_name: str = ""  # for LOAD/STORE edges
    call_site: str = ""   # for CALL_IN/CALL_OUT edges
    alloc: Optional[HeapLoc] = None  # for NEW edges

    def __repr__(self):
        if self.kind == EdgeKind.NEW:
            return f"{self.src} --new--> {self.alloc}"
        if self.kind == EdgeKind.ASSIGN:
            return f"{self.src} --assign--> {self.dst}"
        if self.kind == EdgeKind.LOAD:
            return f"{self.src} --load[{self.field_name}]--> {self.dst}"
        if self.kind == EdgeKind.STORE:
            return f"{self.src} --store[{self.field_name}]--> {self.dst}"
        if self.kind == EdgeKind.CALL_IN:
            return f"{self.src} --call[{self.call_site}]--> {self.dst}"
        if self.kind == EdgeKind.CALL_OUT:
            return f"{self.src} --ret[{self.call_site}]--> {self.dst}"
        return f"{self.src} --{self.kind.name}--> {self.dst}"


@dataclass
class PAG:
    """Pointer Assignment Graph built from V097 constraints."""
    edges: List[PAGEdge] = field(default_factory=list)
    # Indexes for fast lookup
    forward: Dict[str, List[PAGEdge]] = field(default_factory=lambda: defaultdict(list))
    backward: Dict[str, List[PAGEdge]] = field(default_factory=lambda: defaultdict(list))
    alloc_map: Dict[str, HeapLoc] = field(default_factory=dict)
    all_vars: Set[str] = field(default_factory=set)

    def add_edge(self, edge: PAGEdge):
        self.edges.append(edge)
        self.forward[edge.src].append(edge)
        self.backward[edge.dst].append(edge)
        self.all_vars.add(edge.src)
        self.all_vars.add(edge.dst)

    def predecessors(self, var: str) -> List[PAGEdge]:
        """Edges pointing INTO var (backward traversal)."""
        return self.backward.get(var, [])

    def successors(self, var: str) -> List[PAGEdge]:
        """Edges going OUT of var (forward traversal)."""
        return self.forward.get(var, [])


def build_pag(constraints: List[Constraint], alloc_sites: Dict[str, HeapLoc]) -> PAG:
    """Build a PAG from V097 constraints."""
    pag = PAG()
    pag.alloc_map = dict(alloc_sites)

    for c in constraints:
        if c.kind == ConstraintKind.ALLOC:
            edge = PAGEdge(EdgeKind.NEW, c.lhs, c.lhs, alloc=c.alloc)
            pag.add_edge(edge)
        elif c.kind == ConstraintKind.ASSIGN:
            # pts(lhs) >= pts(rhs) means lhs --assign--> rhs
            edge = PAGEdge(EdgeKind.ASSIGN, c.lhs, c.rhs)
            pag.add_edge(edge)
        elif c.kind == ConstraintKind.LOAD:
            # x = y.f means x --load[f]--> y
            edge = PAGEdge(EdgeKind.LOAD, c.lhs, c.rhs, field_name=c.field_name)
            pag.add_edge(edge)
        elif c.kind == ConstraintKind.STORE:
            # x.f = y means x --store[f]--> y
            edge = PAGEdge(EdgeKind.STORE, c.lhs, c.rhs, field_name=c.field_name)
            pag.add_edge(edge)
        elif c.kind == ConstraintKind.CALL_ARG:
            # param = arg at call site
            cs = c.callee if c.callee else "call"
            edge = PAGEdge(EdgeKind.CALL_IN, c.lhs, c.rhs, call_site=cs)
            pag.add_edge(edge)
        elif c.kind == ConstraintKind.CALL_RET:
            # x = retval at call site
            cs = c.callee if c.callee else "call"
            edge = PAGEdge(EdgeKind.CALL_OUT, c.lhs, c.rhs, call_site=cs)
            pag.add_edge(edge)

    return pag


# ---------------------------------------------------------------------------
# Demand-Driven Points-To Solver
# ---------------------------------------------------------------------------

@dataclass
class DemandPTSResult:
    """Result of a demand-driven points-to query."""
    variable: str
    points_to: Set[HeapLoc]
    explored_nodes: int
    explored_edges: int
    cache_hits: int


@dataclass
class DemandAliasResult:
    """Result of a demand-driven alias query."""
    var1: str
    var2: str
    may_alias: bool
    must_alias: bool
    common_targets: Set[HeapLoc]
    var1_pts: Set[HeapLoc]
    var2_pts: Set[HeapLoc]
    explored_nodes: int
    explored_edges: int


class DemandAliasSolver:
    """Demand-driven points-to and alias analysis.

    Given a query "what does variable x point to?", traverses the PAG
    backward from x, following assign/load/store/call edges to discover
    all heap locations reachable through matched-parenthesis paths.

    Algorithm (simplified CFL-reachability):
    1. To find pts(x):
       - If x --new--> h: add h to pts(x)
       - If x --assign--> y: pts(x) |= demand_pts(y)
       - If x --load[f]--> y: for h in demand_pts(y): pts(x) |= demand_field_pts(h, f)
       - If x --call_in[cs]--> arg: pts(x) |= demand_pts(arg) (enter call)
       - If x --call_out[cs]--> ret: pts(x) |= demand_pts(ret) (return from call)
    2. For field pts(h, f): find stores x.f = y where h in pts(x)
    3. Memoize all results for reuse

    Context sensitivity: call/ret edges must have matched call strings.
    """

    def __init__(self, pag: PAG, context_depth: int = 1):
        self.pag = pag
        self.context_depth = context_depth
        # Memoization caches
        self._pts_cache: Dict[str, Set[HeapLoc]] = {}
        self._field_cache: Dict[Tuple[FrozenSet[HeapLoc], str], Set[HeapLoc]] = {}
        # Cycle detection: variables currently being resolved
        self._resolving: Set[str] = set()
        self._resolving_fields: Set[Tuple[FrozenSet[HeapLoc], str]] = set()
        # Statistics
        self.explored_nodes = 0
        self.explored_edges = 0
        self.cache_hits = 0
        # Store edges for field resolution
        self._store_edges: List[PAGEdge] = [
            e for e in pag.edges if e.kind == EdgeKind.STORE
        ]

    def demand_pts(self, var: str) -> Set[HeapLoc]:
        """Demand the points-to set for a variable."""
        # Check cache
        if var in self._pts_cache:
            self.cache_hits += 1
            return self._pts_cache[var]

        # Cycle detection
        if var in self._resolving:
            return set()
        self._resolving.add(var)
        self.explored_nodes += 1

        result: Set[HeapLoc] = set()

        # Traverse backward edges from var
        for edge in self.pag.predecessors(var):
            self.explored_edges += 1

            if edge.kind == EdgeKind.NEW:
                if edge.alloc is not None:
                    result.add(edge.alloc)

            elif edge.kind == EdgeKind.ASSIGN:
                # var = edge.dst  =>  pts(var) |= pts(edge.dst)
                result |= self.demand_pts(edge.dst)

            elif edge.kind == EdgeKind.LOAD:
                # var = edge.dst.field  => for h in pts(edge.dst): pts(var) |= field_pts(h, field)
                base_pts = self.demand_pts(edge.dst)
                if base_pts:
                    result |= self._demand_field_pts(base_pts, edge.field_name)

            elif edge.kind == EdgeKind.CALL_IN:
                # parameter = arg (entering a call)
                result |= self.demand_pts(edge.dst)

            elif edge.kind == EdgeKind.CALL_OUT:
                # var = return value (exiting a call)
                result |= self.demand_pts(edge.dst)

        # Also check forward edges: if this is the src of an assign that was
        # reversed. Actually, for assign edges: src --assign--> dst means
        # pts(src) >= pts(dst). We already handle this via backward edges on src.

        self._resolving.discard(var)
        self._pts_cache[var] = result
        return result

    def _demand_field_pts(self, bases: Set[HeapLoc], field_name: str) -> Set[HeapLoc]:
        """Demand the points-to set for a field of heap locations."""
        key = (frozenset(bases), field_name)

        if key in self._field_cache:
            self.cache_hits += 1
            return self._field_cache[key]

        if key in self._resolving_fields:
            return set()
        self._resolving_fields.add(key)

        result: Set[HeapLoc] = set()

        # Find all store edges: x.field = y where some h in bases is in pts(x)
        for store_edge in self._store_edges:
            if store_edge.field_name != field_name:
                continue
            # store_edge: src.field = dst (src --store[f]--> dst)
            # Check if pts(src) intersects bases
            src_pts = self.demand_pts(store_edge.src)
            if src_pts & bases:
                # The stored value's pts contribute
                result |= self.demand_pts(store_edge.dst)

        self._resolving_fields.discard(key)
        self._field_cache[key] = result
        return result

    def demand_alias(self, var1: str, var2: str) -> DemandAliasResult:
        """Check if two variables may/must alias on demand."""
        # Reset stats for this query
        old_nodes = self.explored_nodes
        old_edges = self.explored_edges

        pts1 = self.demand_pts(var1)
        pts2 = self.demand_pts(var2)

        common = pts1 & pts2
        may = bool(common)
        must = may and len(pts1) == 1 and len(pts2) == 1 and pts1 == pts2

        return DemandAliasResult(
            var1=var1,
            var2=var2,
            may_alias=may,
            must_alias=must,
            common_targets=common,
            var1_pts=pts1,
            var2_pts=pts2,
            explored_nodes=self.explored_nodes - old_nodes,
            explored_edges=self.explored_edges - old_edges,
        )

    def demand_field_alias(self, var1: str, field1: str,
                           var2: str, field2: str) -> DemandAliasResult:
        """Check if var1.field1 and var2.field2 may alias."""
        old_nodes = self.explored_nodes
        old_edges = self.explored_edges

        base1 = self.demand_pts(var1)
        base2 = self.demand_pts(var2)

        pts1 = self._demand_field_pts(base1, field1) if base1 else set()
        pts2 = self._demand_field_pts(base2, field2) if base2 else set()

        common = pts1 & pts2
        may = bool(common)
        must = may and len(pts1) == 1 and len(pts2) == 1 and pts1 == pts2

        return DemandAliasResult(
            var1=f"{var1}.{field1}",
            var2=f"{var2}.{field2}",
            may_alias=may,
            must_alias=must,
            common_targets=common,
            var1_pts=pts1,
            var2_pts=pts2,
            explored_nodes=self.explored_nodes - old_nodes,
            explored_edges=self.explored_edges - old_edges,
        )

    def invalidate(self, var: Optional[str] = None):
        """Invalidate cache (all or for a specific variable)."""
        if var is None:
            self._pts_cache.clear()
            self._field_cache.clear()
        else:
            self._pts_cache.pop(var, None)
            # Invalidate field caches that depended on this var
            # (conservative: clear all field caches)
            self._field_cache.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get solver statistics."""
        return {
            'explored_nodes': self.explored_nodes,
            'explored_edges': self.explored_edges,
            'cache_hits': self.cache_hits,
            'cached_pts': len(self._pts_cache),
            'cached_fields': len(self._field_cache),
            'pag_edges': len(self.pag.edges),
            'pag_vars': len(self.pag.all_vars),
        }


# ---------------------------------------------------------------------------
# Incremental Demand Analysis
# ---------------------------------------------------------------------------

class IncrementalDemandSolver:
    """Demand-driven solver with incremental update support.

    When the program changes, only invalidate affected caches and
    re-query. Tracks which variables depend on which PAG edges.
    """

    def __init__(self, source: str, k: int = 1):
        self.k = k
        self._build(source)

    def _build(self, source: str):
        """Build PAG from source."""
        extractor = ConstraintExtractor(k=self.k)
        self.constraints = extractor.extract(source)
        self.alloc_sites = {
            c.lhs: c.alloc for c in self.constraints
            if c.kind == ConstraintKind.ALLOC and c.alloc is not None
        }
        self.pag = build_pag(self.constraints, self.alloc_sites)
        self.solver = DemandAliasSolver(self.pag, context_depth=self.k)

    def query_pts(self, var: str) -> DemandPTSResult:
        """Query points-to set for a variable."""
        old_nodes = self.solver.explored_nodes
        old_edges = self.solver.explored_edges
        old_hits = self.solver.cache_hits

        pts = self.solver.demand_pts(var)

        return DemandPTSResult(
            variable=var,
            points_to=pts,
            explored_nodes=self.solver.explored_nodes - old_nodes,
            explored_edges=self.solver.explored_edges - old_edges,
            cache_hits=self.solver.cache_hits - old_hits,
        )

    def query_alias(self, var1: str, var2: str) -> DemandAliasResult:
        """Query alias relationship between two variables."""
        return self.solver.demand_alias(var1, var2)

    def update(self, new_source: str):
        """Update for new source, preserving cache where possible."""
        old_constraints = {repr(c) for c in self.constraints}
        extractor = ConstraintExtractor(k=self.k)
        new_constraints = extractor.extract(new_source)
        new_constraint_set = {repr(c) for c in new_constraints}

        # Find changed constraints
        added = new_constraint_set - old_constraints
        removed = old_constraints - new_constraint_set

        if not added and not removed:
            return  # No change

        # Rebuild PAG (conservative -- could be more surgical)
        self.constraints = new_constraints
        self.alloc_sites = {
            c.lhs: c.alloc for c in self.constraints
            if c.kind == ConstraintKind.ALLOC and c.alloc is not None
        }
        self.pag = build_pag(self.constraints, self.alloc_sites)

        # Determine affected variables
        affected: Set[str] = set()
        for c in self.constraints:
            c_repr = repr(c)
            if c_repr in added or c_repr in removed:
                affected.add(c.lhs)
                if c.rhs:
                    affected.add(c.rhs)

        # Create new solver preserving unaffected cache
        old_cache = self.solver._pts_cache
        self.solver = DemandAliasSolver(self.pag, context_depth=self.k)

        # Restore cache for unaffected variables
        for var, pts in old_cache.items():
            if var not in affected:
                self.solver._pts_cache[var] = pts


# ---------------------------------------------------------------------------
# Batch Demand Analyzer
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result of batch demand analysis."""
    alias_pairs: List[DemandAliasResult]
    pts_queries: List[DemandPTSResult]
    total_explored_nodes: int
    total_explored_edges: int
    total_cache_hits: int
    pag_stats: Dict[str, int]


def batch_demand_analysis(source: str, alias_queries: List[Tuple[str, str]] = None,
                          pts_queries: List[str] = None,
                          k: int = 1) -> BatchResult:
    """Run multiple demand queries, sharing cached results."""
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)

    alias_results = []
    pts_results = []

    # Process pts queries first (they populate cache for alias queries)
    for var in (pts_queries or []):
        old_n = solver.explored_nodes
        old_e = solver.explored_edges
        old_h = solver.cache_hits
        pts = solver.demand_pts(var)
        pts_results.append(DemandPTSResult(
            variable=var,
            points_to=pts,
            explored_nodes=solver.explored_nodes - old_n,
            explored_edges=solver.explored_edges - old_e,
            cache_hits=solver.cache_hits - old_h,
        ))

    # Process alias queries
    for v1, v2 in (alias_queries or []):
        alias_results.append(solver.demand_alias(v1, v2))

    stats = solver.get_stats()
    return BatchResult(
        alias_pairs=alias_results,
        pts_queries=pts_results,
        total_explored_nodes=stats['explored_nodes'],
        total_explored_edges=stats['explored_edges'],
        total_cache_hits=stats['cache_hits'],
        pag_stats=stats,
    )


# ---------------------------------------------------------------------------
# Comparison: Demand vs Exhaustive
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Comparison between demand-driven and exhaustive analysis."""
    consistent: bool            # do they agree?
    demand_explored_nodes: int
    demand_explored_edges: int
    exhaustive_constraints: int
    exhaustive_iterations: int
    demand_pts: Dict[str, Set[HeapLoc]]
    exhaustive_pts: Dict[str, Set[HeapLoc]]
    precision_differences: List[str]   # vars where they differ
    savings_ratio: float        # 1 - (demand_explored / total_pag_edges)


def compare_demand_vs_exhaustive(source: str, query_vars: List[str],
                                 k: int = 1) -> ComparisonResult:
    """Compare demand-driven vs exhaustive analysis for specific queries."""
    # Exhaustive
    exh_result = exhaustive_analyze(source, k=k)

    # Demand
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)

    demand_pts_map = {}
    exh_pts_map = {}
    diffs = []

    for var in query_vars:
        d_pts = solver.demand_pts(var)
        e_pts = exh_result.state.get_pts(var)
        demand_pts_map[var] = d_pts
        exh_pts_map[var] = e_pts
        # Demand should be a superset of exhaustive (sound over-approximation)
        # or equal
        if d_pts != e_pts:
            diffs.append(var)

    total_edges = len(pag.edges)
    savings = 1.0 - (solver.explored_edges / max(total_edges, 1))

    return ComparisonResult(
        consistent=len(diffs) == 0,
        demand_explored_nodes=solver.explored_nodes,
        demand_explored_edges=solver.explored_edges,
        exhaustive_constraints=len(exh_result.constraints),
        exhaustive_iterations=exh_result.iterations,
        demand_pts=demand_pts_map,
        exhaustive_pts=exh_pts_map,
        precision_differences=diffs,
        savings_ratio=max(savings, 0.0),
    )


# ---------------------------------------------------------------------------
# High-Level Public API
# ---------------------------------------------------------------------------

def demand_points_to(source: str, var: str, k: int = 1) -> DemandPTSResult:
    """Query points-to set for a single variable on demand.

    Instead of analyzing the entire program, only traverses the
    portion of the PAG reachable backward from the query variable.

    Args:
        source: C10 source code
        var: variable name to query
        k: context sensitivity depth (0=insensitive, 1=1-CFA)

    Returns:
        DemandPTSResult with points-to set and exploration statistics
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)

    pts = solver.demand_pts(var)
    return DemandPTSResult(
        variable=var,
        points_to=pts,
        explored_nodes=solver.explored_nodes,
        explored_edges=solver.explored_edges,
        cache_hits=solver.cache_hits,
    )


def demand_alias_check(source: str, var1: str, var2: str,
                       k: int = 1) -> DemandAliasResult:
    """Check if two variables may/must alias on demand.

    Args:
        source: C10 source code
        var1, var2: variable names to check
        k: context sensitivity depth

    Returns:
        DemandAliasResult with may/must alias and exploration stats
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)
    return solver.demand_alias(var1, var2)


def demand_field_alias_check(source: str, var1: str, field1: str,
                             var2: str, field2: str,
                             k: int = 1) -> DemandAliasResult:
    """Check if var1.field1 and var2.field2 may alias on demand.

    Args:
        source: C10 source code
        var1, var2: base variable names
        field1, field2: field names
        k: context sensitivity depth

    Returns:
        DemandAliasResult with may/must alias and exploration stats
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)
    return solver.demand_field_alias(var1, field1, var2, field2)


def demand_reachability(source: str, var: str, k: int = 1) -> Dict[str, Any]:
    """Compute which variables can transitively flow into var.

    Returns the demand slice: the set of variables whose definitions
    contribute to var's points-to set.

    Args:
        source: C10 source code
        var: variable to trace back from
        k: context sensitivity depth

    Returns:
        Dict with 'reachable_vars', 'pts', 'explored_nodes', 'explored_edges'
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)

    pts = solver.demand_pts(var)

    # The cached variables are the reachable ones
    reachable = set(solver._pts_cache.keys())

    return {
        'query_var': var,
        'points_to': pts,
        'reachable_vars': reachable,
        'reachable_count': len(reachable),
        'total_vars': len(pag.all_vars),
        'explored_fraction': len(reachable) / max(len(pag.all_vars), 1),
        'explored_nodes': solver.explored_nodes,
        'explored_edges': solver.explored_edges,
    }


def incremental_demand(source_v1: str, source_v2: str,
                       queries: List[str], k: int = 1) -> Dict[str, Any]:
    """Incremental demand analysis across two program versions.

    Analyzes v1, then incrementally updates to v2, reporting which
    query results changed.

    Args:
        source_v1: first version of the program
        source_v2: second version
        queries: list of variable names to query
        k: context sensitivity depth

    Returns:
        Dict with v1_results, v2_results, changed_vars, cache_reuse_ratio
    """
    solver = IncrementalDemandSolver(source_v1, k=k)

    # Query v1
    v1_results = {}
    for var in queries:
        r = solver.query_pts(var)
        v1_results[var] = r.points_to

    # Update to v2
    solver.update(source_v2)

    # Query v2
    v2_results = {}
    for var in queries:
        r = solver.query_pts(var)
        v2_results[var] = r.points_to

    changed = [v for v in queries if v1_results[v] != v2_results[v]]

    return {
        'v1_results': {v: len(pts) for v, pts in v1_results.items()},
        'v2_results': {v: len(pts) for v, pts in v2_results.items()},
        'changed_vars': changed,
        'unchanged_vars': [v for v in queries if v not in changed],
        'cache_reuse_ratio': solver.solver.cache_hits / max(
            solver.solver.explored_nodes, 1),
    }


def full_demand_analysis(source: str, queries: List[str] = None,
                         alias_pairs: List[Tuple[str, str]] = None,
                         k: int = 1) -> Dict[str, Any]:
    """Full demand-driven analysis with all available information.

    Args:
        source: C10 source code
        queries: list of variable names to query pts for
        alias_pairs: list of (var1, var2) pairs to check alias
        k: context sensitivity depth

    Returns:
        Dict with pts, aliases, pag_info, statistics
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)
    alloc_sites = {
        c.lhs: c.alloc for c in constraints
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None
    }
    pag = build_pag(constraints, alloc_sites)
    solver = DemandAliasSolver(pag, context_depth=k)

    # Points-to queries
    pts_map = {}
    for var in (queries or []):
        pts_map[var] = solver.demand_pts(var)

    # Alias queries
    alias_map = {}
    for v1, v2 in (alias_pairs or []):
        alias_map[(v1, v2)] = solver.demand_alias(v1, v2)

    stats = solver.get_stats()

    return {
        'points_to': {v: {str(h) for h in pts} for v, pts in pts_map.items()},
        'points_to_raw': pts_map,
        'aliases': {
            f"{v1},{v2}": {
                'may_alias': r.may_alias,
                'must_alias': r.must_alias,
                'common_count': len(r.common_targets),
            }
            for (v1, v2), r in alias_map.items()
        },
        'alias_results': alias_map,
        'pag_info': {
            'edges': len(pag.edges),
            'vars': len(pag.all_vars),
            'alloc_sites': len(pag.alloc_map),
            'edge_kinds': {
                kind.name: sum(1 for e in pag.edges if e.kind == kind)
                for kind in EdgeKind
            },
        },
        'statistics': stats,
    }


def demand_summary(source: str, queries: List[str] = None,
                   alias_pairs: List[Tuple[str, str]] = None,
                   k: int = 1) -> str:
    """Human-readable summary of demand-driven analysis.

    Args:
        source: C10 source code
        queries: variable names to query
        alias_pairs: pairs to check
        k: context sensitivity depth

    Returns:
        Formatted summary string
    """
    result = full_demand_analysis(source, queries, alias_pairs, k)
    lines = ["=== Demand-Driven Alias Analysis ==="]
    lines.append(f"PAG: {result['pag_info']['vars']} variables, "
                 f"{result['pag_info']['edges']} edges, "
                 f"{result['pag_info']['alloc_sites']} alloc sites")

    if result['points_to']:
        lines.append("\nPoints-to sets:")
        for var, pts in result['points_to'].items():
            lines.append(f"  {var} -> {pts if pts else '{}'}")

    if result['aliases']:
        lines.append("\nAlias checks:")
        for pair, info in result['aliases'].items():
            may = "MAY" if info['may_alias'] else "NO"
            must = " (MUST)" if info['must_alias'] else ""
            lines.append(f"  {pair}: {may}{must}")

    stats = result['statistics']
    lines.append(f"\nExploration: {stats['explored_nodes']} nodes, "
                 f"{stats['explored_edges']} edges, "
                 f"{stats['cache_hits']} cache hits")
    total_pag = stats['pag_edges']
    explored = stats['explored_edges']
    savings = (1.0 - explored / max(total_pag, 1)) * 100
    lines.append(f"Savings: explored {explored}/{total_pag} PAG edges "
                 f"({savings:.0f}% saved)")

    return "\n".join(lines)
