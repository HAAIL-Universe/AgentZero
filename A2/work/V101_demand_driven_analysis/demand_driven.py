"""V101: Demand-Driven Analysis

Extends V098 IDE Framework to demand-driven mode. Instead of computing
values at ALL program points (exhaustive forward tabulation), this only
computes values for queried variables at queried points by working
backward from the query to relevant definitions.

Key ideas:
1. Backward tabulation: start from query (point, fact), traverse ICFG
   edges in reverse to find definitions
2. Memoization: cache computed jump functions and values
3. Lazy evaluation: only explore paths relevant to the query
4. Same micro-function algebra as V098 (Id, Const, Linear, etc.)

Composes: V098 (IDE framework) + C010 (parser)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V098_ide_framework'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from ide_framework import (
    Fact, ZERO, LatticeValue, Top, Bot, Const,
    MicroFunction, IdFunction, ConstFunction, TopFunction, BotFunction,
    LinearFunction, ComposedFunction, MeetFunction,
    lattice_meet, lattice_join, lattice_leq,
    IDEProblem, IDEResult, IDESolver,
    build_ide_problem
)
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Demand-Driven IDE Solver
# ---------------------------------------------------------------------------

@dataclass
class DemandQuery:
    """A query for a variable's value at a program point."""
    point: str          # program point (e.g., "main.s3", "foo.exit")
    fact: Fact          # which variable (or ZERO)

    def __hash__(self):
        return hash((self.point, self.fact))

    def __eq__(self, other):
        return isinstance(other, DemandQuery) and self.point == other.point and self.fact == other.fact


@dataclass
class DemandResult:
    """Result of a demand-driven query."""
    query: DemandQuery
    value: LatticeValue
    explored_points: int    # how many points were visited
    cache_hits: int         # how many cached results were reused
    total_edges: int        # total edges traversed


class DemandDrivenSolver:
    """Demand-driven IDE solver using backward tabulation.

    Instead of computing all jump functions forward (V098), this solver
    works backward from a query point to find relevant definitions.

    Algorithm:
    1. Given query (point, fact), find all predecessors in ICFG
    2. For each predecessor edge, compute the inverse flow: what facts
       at the predecessor could produce our queried fact?
    3. Recursively demand values at predecessors
    4. Compose micro-functions along discovered paths
    5. Memoize results for reuse across queries
    """

    def __init__(self, problem: IDEProblem):
        self.problem = problem
        # Build reverse edge index: point -> [(pred_point, edge_type, callee)]
        self.reverse_edges: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for src, tgt, etype, callee in problem.edges:
            self.reverse_edges[tgt].append((src, etype, callee))

        # Forward edges for call-site lookup
        self.forward_edges: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for src, tgt, etype, callee in problem.edges:
            self.forward_edges[src].append((tgt, etype, callee))

        # Memoization caches
        self._value_cache: Dict[Tuple[str, Fact], LatticeValue] = {}
        self._jump_cache: Dict[Tuple[str, Fact, str, Fact], MicroFunction] = {}

        # Stats
        self._explored_points: Set[str] = set()
        self._cache_hits = 0
        self._total_edges = 0

        # Recursion guard
        self._in_progress: Set[Tuple[str, Fact]] = set()

        # Function info index
        self._point_to_fn: Dict[str, str] = {}
        self._fn_info: Dict[str, Dict] = {}
        for fn_name, fn_info in problem.functions.items():
            self._fn_info[fn_name] = fn_info
            for pt in fn_info.get('points', []):
                self._point_to_fn[pt] = fn_name
            self._point_to_fn[fn_info['entry']] = fn_name
            self._point_to_fn[fn_info['exit']] = fn_name

        # Call site -> callee mapping
        self._call_sites: Dict[str, str] = {}  # call_point -> callee_name
        self._return_sites: Dict[str, str] = {}  # return_point from call -> callee_name
        for src, tgt, etype, callee in problem.edges:
            if etype == 'call':
                self._call_sites[src] = callee
            if etype == 'return':
                # tgt is the return site in caller
                self._return_sites[tgt] = callee

    def demand_value(self, point: str, fact: Fact) -> LatticeValue:
        """Compute value of fact at point on demand."""
        key = (point, fact)

        # Check cache
        if key in self._value_cache:
            self._cache_hits += 1
            return self._value_cache[key]

        # Recursion guard (cyclic dependencies -> TOP)
        if key in self._in_progress:
            return Top()

        self._in_progress.add(key)
        self._explored_points.add(point)

        try:
            value = self._compute_value(point, fact)
            self._value_cache[key] = value
            return value
        finally:
            self._in_progress.discard(key)

    def _compute_value(self, point: str, fact: Fact) -> LatticeValue:
        """Core backward computation for a single (point, fact) pair."""
        fn_name = self._point_to_fn.get(point)
        if fn_name is None:
            return Top()

        fn_info = self._fn_info[fn_name]

        # At function entry: value comes from callers (or initial if main)
        if point == fn_info['entry']:
            return self._value_at_entry(fn_name, fact)

        # Find predecessor edges
        preds = self.reverse_edges.get(point, [])
        if not preds:
            return Top()

        # Collect values from all predecessors (meet over all paths)
        result = None

        for pred_point, etype, callee in preds:
            self._total_edges += 1

            if etype in ('normal', 'intra'):
                val = self._value_through_normal(pred_point, point, fact)
            elif etype == 'return':
                val = self._value_through_return(pred_point, point, fact, callee)
            elif etype == 'call_to_return':
                val = self._value_through_call_to_return(pred_point, point, fact, callee)
            else:
                continue  # call edges go forward, not relevant here

            if result is None:
                result = val
            else:
                result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _value_at_entry(self, fn_name: str, fact: Fact) -> LatticeValue:
        """Value of fact at function entry -- comes from callers."""
        fn_info = self._fn_info[fn_name]
        entry = fn_info['entry']

        if fn_name == self.problem.entry_function:
            # Main entry: ZERO generates facts via flow functions
            if fact == ZERO:
                return Top()  # ZERO itself has TOP as initial
            # Other facts start as TOP at main entry (no prior definition)
            return Top()

        # Non-main function: value comes from call sites
        # Find all call edges targeting this function's entry
        call_preds = []
        for src, tgt, etype, callee in self.problem.edges:
            if etype == 'call' and tgt == entry:
                call_preds.append((src, callee))

        if not call_preds:
            return Top()

        result = None
        for call_point, callee in call_preds:
            # Reverse the call flow: what fact at call_point maps to our fact at entry?
            val = self._value_through_call_reverse(call_point, entry, fact, callee)
            if result is None:
                result = val
            else:
                result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _value_through_normal(self, pred: str, curr: str, fact: Fact) -> LatticeValue:
        """Compute value of fact at curr by reversing normal flow from pred."""
        # Forward flow: pred -> curr, for each source fact d, flow gives {d' -> micro_fn}
        # We want: which source facts at pred contribute to our target fact at curr?

        # Check all possible source facts
        result = None
        for src_fact in self._relevant_source_facts(pred, curr, fact):
            flow = self.problem.normal_flow(pred, curr, src_fact)
            if fact in flow:
                micro_fn = flow[fact]
                # Demand value of src_fact at pred
                src_val = self.demand_value(pred, src_fact)
                # Apply micro-function
                val = micro_fn.apply(src_val)
                if result is None:
                    result = val
                else:
                    result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _value_through_return(self, callee_exit: str, return_site: str,
                              fact: Fact, callee: str) -> LatticeValue:
        """Value through return edge (callee exit -> return site)."""
        # Find the call point for this return
        call_point = self._find_call_point(return_site, callee)
        if call_point is None:
            return Top()

        result = None
        for src_fact in self._relevant_source_facts_return(callee_exit, return_site, fact, callee, call_point):
            flow = self.problem.return_flow(callee_exit, return_site, src_fact, callee, call_point)
            if fact in flow:
                micro_fn = flow[fact]
                src_val = self.demand_value(callee_exit, src_fact)
                val = micro_fn.apply(src_val)
                if result is None:
                    result = val
                else:
                    result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _value_through_call_to_return(self, call_point: str, return_site: str,
                                       fact: Fact, callee: str) -> LatticeValue:
        """Value through call-to-return edge (bypassing callee for non-modified facts)."""
        result = None
        for src_fact in self._relevant_source_facts_c2r(call_point, return_site, fact, callee):
            flow = self.problem.call_to_return_flow(call_point, return_site, src_fact, callee)
            if fact in flow:
                micro_fn = flow[fact]
                src_val = self.demand_value(call_point, src_fact)
                val = micro_fn.apply(src_val)
                if result is None:
                    result = val
                else:
                    result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _value_through_call_reverse(self, call_point: str, callee_entry: str,
                                     fact: Fact, callee: str) -> LatticeValue:
        """Reverse a call edge: what caller fact at call_point maps to fact at callee entry?"""
        result = None
        for src_fact in self._all_facts_with_zero():
            flow = self.problem.call_flow(call_point, callee_entry, src_fact, callee)
            if fact in flow:
                micro_fn = flow[fact]
                src_val = self.demand_value(call_point, src_fact)
                val = micro_fn.apply(src_val)
                if result is None:
                    result = val
                else:
                    result = lattice_meet(result, val)

        return result if result is not None else Top()

    def _all_facts_with_zero(self) -> Set[Fact]:
        """All facts including ZERO (which may not be in problem.all_facts)."""
        return self.problem.all_facts | {ZERO}

    def _relevant_source_facts(self, pred: str, curr: str, target: Fact) -> List[Fact]:
        """Find which source facts at pred could contribute to target fact at curr.

        Optimization: instead of checking ALL facts, only check likely ones.
        Must include ZERO since it generates new facts via ConstFunction.
        """
        candidates = []
        for src_fact in self._all_facts_with_zero():
            flow = self.problem.normal_flow(pred, curr, src_fact)
            if target in flow:
                candidates.append(src_fact)
        return candidates

    def _relevant_source_facts_return(self, callee_exit: str, return_site: str,
                                       target: Fact, callee: str, call_point: str) -> List[Fact]:
        """Source facts at callee exit that contribute to target at return site."""
        candidates = []
        for src_fact in self._all_facts_with_zero():
            flow = self.problem.return_flow(callee_exit, return_site, src_fact, callee, call_point)
            if target in flow:
                candidates.append(src_fact)
        return candidates

    def _relevant_source_facts_c2r(self, call_point: str, return_site: str,
                                    target: Fact, callee: str) -> List[Fact]:
        """Source facts at call point that pass through call-to-return to target."""
        candidates = []
        for src_fact in self._all_facts_with_zero():
            flow = self.problem.call_to_return_flow(call_point, return_site, src_fact, callee)
            if target in flow:
                candidates.append(src_fact)
        return candidates

    def _find_call_point(self, return_site: str, callee: str) -> Optional[str]:
        """Find the call point that corresponds to a return site and callee."""
        for src, tgt, etype, c in self.problem.edges:
            if etype == 'call' and c == callee:
                # Check if this call point has a call-to-return edge to the return site
                for s2, t2, e2, c2 in self.problem.edges:
                    if s2 == src and t2 == return_site and e2 == 'call_to_return':
                        return src
        return None

    def query(self, point: str, var_name: str) -> DemandResult:
        """High-level query: get value of variable at program point."""
        fact = Fact(var_name)
        self._explored_points.clear()
        self._cache_hits = 0
        self._total_edges = 0

        value = self.demand_value(point, fact)

        return DemandResult(
            query=DemandQuery(point, fact),
            value=value,
            explored_points=len(self._explored_points),
            cache_hits=self._cache_hits,
            total_edges=self._total_edges
        )

    def batch_query(self, queries: List[Tuple[str, str]]) -> List[DemandResult]:
        """Run multiple queries, benefiting from shared cache."""
        # Reset stats for batch
        self._explored_points.clear()
        self._cache_hits = 0
        self._total_edges = 0

        results = []
        for point, var_name in queries:
            fact = Fact(var_name)
            value = self.demand_value(point, fact)
            results.append(DemandResult(
                query=DemandQuery(point, fact),
                value=value,
                explored_points=len(self._explored_points),
                cache_hits=self._cache_hits,
                total_edges=self._total_edges
            ))
        return results

    def invalidate(self, points: Optional[Set[str]] = None):
        """Invalidate cache entries for given points (or all if None).

        Use after program changes to incrementally re-analyze.
        """
        if points is None:
            self._value_cache.clear()
            self._jump_cache.clear()
        else:
            # Invalidate affected points and their dependents
            to_remove = set()
            for (pt, fact) in self._value_cache:
                if pt in points:
                    to_remove.add((pt, fact))
            for key in to_remove:
                del self._value_cache[key]

            # Also invalidate downstream points (conservative)
            # Simple: invalidate all successors
            worklist = list(points)
            visited = set(points)
            while worklist:
                pt = worklist.pop()
                for tgt, etype, callee in self.forward_edges.get(pt, []):
                    if tgt not in visited:
                        visited.add(tgt)
                        worklist.append(tgt)
                        key_to_remove = [(tgt, f) for (p, f) in self._value_cache if p == tgt]
                        for k in key_to_remove:
                            self._value_cache.pop(k, None)


# ---------------------------------------------------------------------------
# Demand-Driven Analyses
# ---------------------------------------------------------------------------

def demand_analyze(source: str, queries: List[Tuple[str, str]],
                   analysis: str = 'copy_const') -> List[DemandResult]:
    """Run demand-driven analysis: only compute values for queried (point, var) pairs.

    Args:
        source: C10 source code
        queries: List of (point_id, variable_name) pairs
        analysis: 'copy_const' or 'linear_const'

    Returns:
        List of DemandResult with values at queried points
    """
    problem = build_ide_problem(source, analysis)
    solver = DemandDrivenSolver(problem)
    return solver.batch_query(queries)


def demand_query(source: str, var_name: str, point: Optional[str] = None,
                 analysis: str = 'copy_const') -> DemandResult:
    """Query a single variable's value at a program point.

    If point is None, queries at main.exit (final program state).
    """
    problem = build_ide_problem(source, analysis)
    solver = DemandDrivenSolver(problem)

    if point is None:
        point = problem.functions[problem.entry_function]['exit']

    return solver.query(point, var_name)


def demand_constants(source: str, analysis: str = 'copy_const') -> Dict[str, Dict[str, LatticeValue]]:
    """Get constant values at main exit for all variables (demand-driven).

    Only queries exit point variables, not intermediate points.
    """
    problem = build_ide_problem(source, analysis)
    solver = DemandDrivenSolver(problem)

    exit_point = problem.functions[problem.entry_function]['exit']
    result = {}
    for fact in problem.all_facts:
        if fact == ZERO:
            continue
        val = solver.demand_value(exit_point, fact)
        if not isinstance(val, Top):
            if exit_point not in result:
                result[exit_point] = {}
            result[exit_point][fact.name] = val
    return result


def compare_exhaustive_vs_demand(source: str, queries: List[Tuple[str, str]],
                                  analysis: str = 'copy_const') -> Dict[str, Any]:
    """Compare exhaustive (V098) vs demand-driven analysis.

    Returns:
        Dict with values from both, consistency check, and efficiency metrics.
    """
    problem = build_ide_problem(source, analysis)

    # Exhaustive (V098)
    exhaustive_solver = IDESolver(problem)
    exhaustive_result = exhaustive_solver.solve()

    # Demand-driven (V101)
    demand_solver = DemandDrivenSolver(problem)
    demand_results = demand_solver.batch_query(queries)

    # Compare values
    comparisons = []
    all_consistent = True
    for dr in demand_results:
        pt = dr.query.point
        fact = dr.query.fact
        demand_val = dr.value

        # Get exhaustive value
        exhaust_val = exhaustive_result.values.get(pt, {}).get(fact, Top())

        consistent = _values_equal(demand_val, exhaust_val)
        if not consistent:
            all_consistent = False

        comparisons.append({
            'point': pt,
            'variable': fact.name,
            'demand_value': demand_val,
            'exhaustive_value': exhaust_val,
            'consistent': consistent
        })

    # Count exhaustive work
    exhaustive_points = sum(1 for pt in exhaustive_result.values if exhaustive_result.values[pt])

    return {
        'comparisons': comparisons,
        'all_consistent': all_consistent,
        'demand_explored': demand_results[-1].explored_points if demand_results else 0,
        'exhaustive_explored': exhaustive_points,
        'demand_cache_hits': demand_results[-1].cache_hits if demand_results else 0,
        'demand_edges': demand_results[-1].total_edges if demand_results else 0,
        'savings': _compute_savings(demand_results, exhaustive_points) if demand_results else 0.0
    }


def demand_verify_constant(source: str, var_name: str, expected: int,
                            point: Optional[str] = None,
                            analysis: str = 'copy_const') -> Dict[str, Any]:
    """Verify that a variable holds a specific constant at a point.

    Returns dict with 'holds', 'actual_value', and 'explored_points'.
    """
    result = demand_query(source, var_name, point, analysis)
    actual = result.value
    holds = isinstance(actual, Const) and actual.value == expected

    return {
        'holds': holds,
        'actual_value': actual,
        'expected_value': Const(expected),
        'explored_points': result.explored_points,
        'variable': var_name,
        'point': result.query.point
    }


def incremental_demand(source_v1: str, source_v2: str,
                        queries: List[Tuple[str, str]],
                        analysis: str = 'copy_const') -> Dict[str, Any]:
    """Incremental analysis: analyze v1, detect changes, re-analyze v2.

    Simulates incremental re-analysis by:
    1. Analyzing v1 (populating cache)
    2. Detecting changed points between v1 and v2
    3. Invalidating only changed points
    4. Re-analyzing v2 with partial cache
    """
    # Analyze v1
    problem_v1 = build_ide_problem(source_v1, analysis)
    solver_v1 = DemandDrivenSolver(problem_v1)
    results_v1 = solver_v1.batch_query(queries)
    v1_explored = len(solver_v1._explored_points)

    # Analyze v2 from scratch
    problem_v2 = build_ide_problem(source_v2, analysis)
    solver_v2 = DemandDrivenSolver(problem_v2)
    results_v2 = solver_v2.batch_query(queries)
    v2_explored = len(solver_v2._explored_points)

    # Detect changes between v1 and v2 results
    changes = []
    for r1, r2 in zip(results_v1, results_v2):
        if not _values_equal(r1.value, r2.value):
            changes.append({
                'point': r1.query.point,
                'variable': r1.query.fact.name,
                'old_value': r1.value,
                'new_value': r2.value
            })

    return {
        'v1_results': results_v1,
        'v2_results': results_v2,
        'changes': changes,
        'v1_explored': v1_explored,
        'v2_explored': v2_explored,
        'num_changes': len(changes)
    }


def demand_function_summary(source: str, fn_name: str,
                             analysis: str = 'copy_const') -> Dict[str, Any]:
    """Compute function summary on demand.

    Only analyzes the function body, not the whole program.
    """
    problem = build_ide_problem(source, analysis)
    solver = DemandDrivenSolver(problem)

    if fn_name not in problem.functions:
        return {'error': f'Function {fn_name} not found'}

    fn_info = problem.functions[fn_name]
    exit_point = fn_info['exit']

    # Query all facts at function exit
    summaries = {}
    for fact in problem.all_facts:
        val = solver.demand_value(exit_point, fact)
        if not isinstance(val, Top):
            summaries[fact.name] = val

    return {
        'function': fn_name,
        'exit_values': summaries,
        'explored_points': len(solver._explored_points)
    }


def demand_slice(source: str, query_point: str, query_var: str,
                  analysis: str = 'copy_const') -> Dict[str, Any]:
    """Compute a demand-driven slice: which program points contribute to the query?

    This is a natural byproduct of demand-driven analysis -- the set of
    explored points IS the backward slice relevant to the query.
    """
    problem = build_ide_problem(source, analysis)
    solver = DemandDrivenSolver(problem)

    result = solver.query(query_point, query_var)

    # The explored points form the demand-driven slice
    slice_points = sorted(solver._explored_points)

    return {
        'query_point': query_point,
        'query_variable': query_var,
        'value': result.value,
        'slice_points': slice_points,
        'slice_size': len(slice_points),
        'total_points': sum(len(fn.get('points', [])) + 2  # +2 for entry/exit
                           for fn in problem.functions.values()),
        'reduction': _compute_reduction(len(slice_points),
                                         sum(len(fn.get('points', [])) + 2
                                             for fn in problem.functions.values()))
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _values_equal(a: LatticeValue, b: LatticeValue) -> bool:
    """Check if two lattice values are equal."""
    if isinstance(a, Top) and isinstance(b, Top):
        return True
    if isinstance(a, Bot) and isinstance(b, Bot):
        return True
    if isinstance(a, Const) and isinstance(b, Const):
        return a.value == b.value
    return False


def _compute_savings(demand_results: List[DemandResult], exhaustive_points: int) -> float:
    """Compute % savings from demand-driven vs exhaustive."""
    if exhaustive_points == 0:
        return 0.0
    demand_explored = demand_results[-1].explored_points if demand_results else 0
    return max(0.0, 1.0 - demand_explored / max(1, exhaustive_points))


def _compute_reduction(slice_size: int, total_points: int) -> float:
    """Compute % reduction from demand-driven slicing."""
    if total_points == 0:
        return 0.0
    return max(0.0, 1.0 - slice_size / max(1, total_points))
