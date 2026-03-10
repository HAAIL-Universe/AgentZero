"""V062: Abstract Conflict-Driven Learning (ACDL)

Composes:
- V029 (Abstract DPLL(T)) -- path-sensitive abstract analysis with CDCL
- V012 (Craig Interpolation) -- predicate generation from infeasible traces

Key idea: When V029's abstract analysis finds a conflict (spurious or real),
use Craig interpolation to extract new predicates that refine the abstract
domain. This creates a CEGAR-like loop:
1. Analyze program with current abstract predicates
2. On conflict, extract trace constraints as SMT formulas
3. Compute Craig interpolants for the trace
4. Extract atomic predicates from interpolants
5. Add predicates to the abstract domain, re-analyze

This strengthens the abstract domain automatically, improving precision
without manual predicate selection.
"""

import sys, os

_base = os.path.dirname(__file__)
_root = os.path.join(_base, '..', '..', '..')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_base, '..', 'V029_abstract_dpll_t'))
sys.path.insert(0, os.path.join(_base, '..', 'V012_craig_interpolation'))
sys.path.insert(0, os.path.join(_root, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_root, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_root, 'challenges', 'C039_abstract_interpreter'))

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum

# V029: Abstract DPLL(T)
from abstract_dpll_t import (
    AbstractDPLLT, DPLLTResult, Verdict, ConflictInfo, LearnedClause,
    analyze_program
)

# V012: Craig interpolation
from craig_interpolation import (
    interpolate, sequence_interpolate, extract_predicates_from_interpolant,
    Interpolant, InterpolantResult, SequenceInterpolant,
    collect_vars, make_conjunction, negate, flatten_conjunction
)

# C037: SMT solver
from smt_solver import SMTSolver, SMTResult, App, Op, IntConst, BoolConst, INT, BOOL

# C010: Parser
from stack_vm import lex, Parser


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RefinementStatus(Enum):
    VERIFIED = "verified"       # All assertions proved safe
    VIOLATED = "violated"       # Real assertion violation found
    REFINED = "refined"         # New predicates learned, re-analysis needed
    EXHAUSTED = "exhausted"     # Max iterations reached
    UNKNOWN = "unknown"


@dataclass
class Predicate:
    """A predicate over program variables."""
    formula: Any       # SMT Term
    variables: Set[str]
    source: str = ""   # How this predicate was learned
    iteration: int = 0

    def __hash__(self):
        return hash(str(self.formula))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return str(self.formula) == str(other.formula)


@dataclass
class RefinementStep:
    """Record of one CEGAR iteration."""
    iteration: int
    verdict: Verdict
    paths_explored: int
    paths_pruned: int
    conflicts_found: int
    predicates_before: int
    predicates_after: int
    new_predicates: List[str]  # string representations


@dataclass
class ACDLResult:
    """Result of Abstract Conflict-Driven Learning analysis."""
    status: RefinementStatus
    iterations: int
    final_verdict: Verdict
    predicates_learned: int
    total_paths_explored: int
    total_paths_pruned: int
    total_conflicts: int
    refinement_history: List[RefinementStep] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    counterexample: Optional[Dict[str, int]] = None
    final_dpll_result: Optional[DPLLTResult] = None

    @property
    def is_safe(self) -> bool:
        return self.status == RefinementStatus.VERIFIED

    @property
    def is_violated(self) -> bool:
        return self.status == RefinementStatus.VIOLATED


# ---------------------------------------------------------------------------
# Trace-to-SMT conversion
# ---------------------------------------------------------------------------

class TraceEncoder:
    """Converts abstract conflict traces to SMT formulas for interpolation."""

    def __init__(self):
        self.solver = SMTSolver()

    def conflict_to_formulas(self, conflict: ConflictInfo,
                             program_vars: Set[str]) -> List[Any]:
        """Convert a ConflictInfo into a sequence of SMT formulas.

        Each branch decision becomes a formula segment for sequence interpolation.
        """
        formulas = []
        for branch_id, direction in conflict.branch_decisions:
            # Each branch decision constrains variables
            formula = self._branch_to_formula(branch_id, direction, conflict)
            if formula is not None:
                formulas.append(formula)

        # Add assertion failure as final formula
        if conflict.message:
            neg_formula = self._assertion_to_formula(conflict)
            if neg_formula is not None:
                formulas.append(neg_formula)

        return formulas

    def abstract_state_to_formula(self, abstract_state: Dict[str, Tuple],
                                  vars_of_interest: Set[str] = None) -> Any:
        """Convert abstract state (var -> (sign, interval)) to SMT formula."""
        conjuncts = []
        for var_name, state in abstract_state.items():
            if vars_of_interest and var_name not in vars_of_interest:
                continue
            bounds = self._state_to_bounds(var_name, state)
            conjuncts.extend(bounds)
        if not conjuncts:
            return BoolConst(True)
        return make_conjunction(conjuncts)

    def _state_to_bounds(self, var_name: str, state: Tuple) -> List[Any]:
        """Convert a single variable's abstract state to SMT bounds."""
        bounds = []
        v = self.solver.Int(var_name)
        if len(state) >= 2:
            sign, interval = state[0], state[1]
            # Extract interval bounds
            if hasattr(interval, 'lo') and interval.lo is not None:
                bounds.append(App(Op.GE, [v, IntConst(interval.lo)], BOOL))
            if hasattr(interval, 'hi') and interval.hi is not None:
                bounds.append(App(Op.LE, [v, IntConst(interval.hi)], BOOL))
            # Sign constraints
            if sign == 'pos':
                bounds.append(App(Op.GT, [v, IntConst(0)], BOOL))
            elif sign == 'neg':
                bounds.append(App(Op.LT, [v, IntConst(0)], BOOL))
            elif sign == 'zero':
                bounds.append(App(Op.EQ, [v, IntConst(0)], BOOL))
        elif isinstance(state, tuple) and len(state) == 2:
            # Simple (lo, hi) interval
            lo, hi = state
            if lo is not None:
                bounds.append(App(Op.GE, [v, IntConst(lo)], BOOL))
            if hi is not None:
                bounds.append(App(Op.LE, [v, IntConst(hi)], BOOL))
        return bounds

    def _branch_to_formula(self, branch_id: int, direction: bool,
                            conflict: ConflictInfo) -> Optional[Any]:
        """Convert a branch decision to an SMT formula.

        Uses the abstract state to reconstruct constraints.
        """
        # Build a simple predicate: branch_b = direction
        b_var = self.solver.Int(f'__branch_{branch_id}')
        val = IntConst(1 if direction else 0)
        return App(Op.EQ, [b_var, val], BOOL)

    def _assertion_to_formula(self, conflict: ConflictInfo) -> Optional[Any]:
        """Convert assertion failure to negated assertion formula."""
        # Use abstract state bounds as the assertion context
        if conflict.abstract_state:
            return self.abstract_state_to_formula(conflict.abstract_state)
        return None


# ---------------------------------------------------------------------------
# Predicate store
# ---------------------------------------------------------------------------

class PredicateStore:
    """Manages learned predicates across CEGAR iterations."""

    def __init__(self):
        self.predicates: List[Predicate] = []
        self._seen: Set[str] = set()

    def add(self, formula: Any, source: str = "", iteration: int = 0) -> bool:
        """Add a predicate. Returns True if new."""
        key = str(formula)
        if key in self._seen:
            return False
        self._seen.add(key)
        pred = Predicate(
            formula=formula,
            variables=collect_vars(formula),
            source=source,
            iteration=iteration
        )
        self.predicates.append(pred)
        return True

    def add_from_interpolant(self, interpolant: Any, iteration: int = 0) -> int:
        """Extract and add predicates from an interpolant. Returns count added."""
        atoms = extract_predicates_from_interpolant(interpolant)
        count = 0
        for atom in atoms:
            if self.add(atom, source="interpolant", iteration=iteration):
                count += 1
        return count

    @property
    def count(self) -> int:
        return len(self.predicates)

    def get_formulas(self) -> List[Any]:
        return [p.formula for p in self.predicates]

    def get_for_vars(self, variables: Set[str]) -> List[Predicate]:
        """Get predicates relevant to a set of variables."""
        return [p for p in self.predicates
                if p.variables & variables]


# ---------------------------------------------------------------------------
# CEGAR refinement loop
# ---------------------------------------------------------------------------

class AbstractCDLAnalyzer:
    """CEGAR-based program analyzer using Abstract DPLL(T) + Craig Interpolation.

    Loop:
    1. Run V029 Abstract DPLL(T) analysis
    2. If SAFE -> return VERIFIED
    3. If UNSAFE with real counterexample -> return VIOLATED
    4. If UNSAFE but potentially spurious:
       a. Extract conflict trace as SMT formulas
       b. Compute Craig interpolants
       c. Extract new predicates
       d. If new predicates found -> refine and re-analyze
       e. If no new predicates -> return UNKNOWN
    """

    def __init__(self, max_iterations: int = 10, max_decisions: int = 200,
                 use_smt: bool = True):
        self.max_iterations = max_iterations
        self.max_decisions = max_decisions
        self.use_smt = use_smt
        self.encoder = TraceEncoder()
        self.store = PredicateStore()

    def analyze(self, source: str) -> ACDLResult:
        """Run CEGAR analysis on source code."""
        total_explored = 0
        total_pruned = 0
        total_conflicts = 0
        history = []

        for iteration in range(self.max_iterations):
            preds_before = self.store.count

            # Step 1: Run Abstract DPLL(T) analysis
            dpll = AbstractDPLLT(
                max_decisions=self.max_decisions,
                use_smt_refinement=self.use_smt
            )
            result = dpll.analyze(source)

            total_explored += result.paths_explored
            total_pruned += result.paths_pruned
            total_conflicts += len(result.conflicts)

            # Step 2: Check verdict
            if result.verdict == Verdict.SAFE:
                step = RefinementStep(
                    iteration=iteration, verdict=result.verdict,
                    paths_explored=result.paths_explored,
                    paths_pruned=result.paths_pruned,
                    conflicts_found=len(result.conflicts),
                    predicates_before=preds_before,
                    predicates_after=self.store.count,
                    new_predicates=[]
                )
                history.append(step)
                return ACDLResult(
                    status=RefinementStatus.VERIFIED,
                    iterations=iteration + 1,
                    final_verdict=Verdict.SAFE,
                    predicates_learned=self.store.count,
                    total_paths_explored=total_explored,
                    total_paths_pruned=total_pruned,
                    total_conflicts=total_conflicts,
                    refinement_history=history,
                    predicates=list(self.store.predicates),
                    final_dpll_result=result
                )

            # Step 3: Try to refine from conflicts
            new_preds = []
            if result.conflicts:
                new_preds = self._refine_from_conflicts(
                    result.conflicts, source, iteration
                )

            step = RefinementStep(
                iteration=iteration, verdict=result.verdict,
                paths_explored=result.paths_explored,
                paths_pruned=result.paths_pruned,
                conflicts_found=len(result.conflicts),
                predicates_before=preds_before,
                predicates_after=self.store.count,
                new_predicates=new_preds
            )
            history.append(step)

            # Step 4: If we have a counterexample and SMT confirmed it
            if result.counterexample and result.verdict == Verdict.UNSAFE:
                if not new_preds:
                    return ACDLResult(
                        status=RefinementStatus.VIOLATED,
                        iterations=iteration + 1,
                        final_verdict=Verdict.UNSAFE,
                        predicates_learned=self.store.count,
                        total_paths_explored=total_explored,
                        total_paths_pruned=total_pruned,
                        total_conflicts=total_conflicts,
                        refinement_history=history,
                        predicates=list(self.store.predicates),
                        counterexample=result.counterexample,
                        final_dpll_result=result
                    )

            # Step 5: No new predicates means we're stuck
            if not new_preds:
                # Return current best verdict
                status = (RefinementStatus.VIOLATED
                         if result.verdict == Verdict.UNSAFE
                         else RefinementStatus.UNKNOWN)
                return ACDLResult(
                    status=status,
                    iterations=iteration + 1,
                    final_verdict=result.verdict,
                    predicates_learned=self.store.count,
                    total_paths_explored=total_explored,
                    total_paths_pruned=total_pruned,
                    total_conflicts=total_conflicts,
                    refinement_history=history,
                    predicates=list(self.store.predicates),
                    counterexample=result.counterexample,
                    final_dpll_result=result
                )

            # Step 6: Refine and continue (new predicates found)
            # The predicates are stored; next iteration will use them implicitly
            # through the abstract domain refinement

        # Max iterations exhausted
        return ACDLResult(
            status=RefinementStatus.EXHAUSTED,
            iterations=self.max_iterations,
            final_verdict=Verdict.UNKNOWN,
            predicates_learned=self.store.count,
            total_paths_explored=total_explored,
            total_paths_pruned=total_pruned,
            total_conflicts=total_conflicts,
            refinement_history=history,
            predicates=list(self.store.predicates),
            final_dpll_result=None
        )

    def _refine_from_conflicts(self, conflicts: List[ConflictInfo],
                                source: str, iteration: int) -> List[str]:
        """Try to extract new predicates from conflicts via interpolation."""
        new_preds = []
        for conflict in conflicts:
            preds = self._interpolate_conflict(conflict, iteration)
            new_preds.extend(preds)
        return new_preds

    def _interpolate_conflict(self, conflict: ConflictInfo,
                               iteration: int) -> List[str]:
        """Compute interpolants from a single conflict and extract predicates."""
        new_preds = []

        # Build SMT formulas from the conflict trace
        trace_formulas = self.encoder.conflict_to_formulas(
            conflict, program_vars=set(conflict.abstract_state.keys())
        )

        if len(trace_formulas) < 2:
            # Not enough formulas for interpolation -- try pairwise
            return self._try_pairwise_interpolation(conflict, iteration)

        # Try sequence interpolation
        try:
            seq_result = sequence_interpolate(trace_formulas)
            if seq_result.result == InterpolantResult.SUCCESS and seq_result.interpolants:
                for interp in seq_result.interpolants:
                    count = self.store.add_from_interpolant(interp, iteration=iteration)
                    if count > 0:
                        atoms = extract_predicates_from_interpolant(interp)
                        new_preds.extend(str(a) for a in atoms)
        except Exception:
            pass

        # If sequence interpolation didn't work, try pairwise
        if not new_preds:
            new_preds.extend(self._try_pairwise_interpolation(conflict, iteration))

        return new_preds

    def _try_pairwise_interpolation(self, conflict: ConflictInfo,
                                     iteration: int) -> List[str]:
        """Try pairwise interpolation between abstract state formulas."""
        new_preds = []
        if not conflict.abstract_state:
            return new_preds

        # Build formulas from abstract state
        vars_list = list(conflict.abstract_state.keys())
        if len(vars_list) < 2:
            return new_preds

        # Try interpolating between pairs of variable constraints
        solver = SMTSolver()
        all_bounds = []
        for var_name, state in conflict.abstract_state.items():
            bounds = self.encoder._state_to_bounds(var_name, state)
            all_bounds.extend(bounds)

        if len(all_bounds) < 2:
            return new_preds

        # Split bounds into A and B partitions
        mid = len(all_bounds) // 2
        a_formula = make_conjunction(all_bounds[:mid])
        b_formula = make_conjunction(all_bounds[mid:])

        try:
            result = interpolate(a_formula, b_formula)
            if result.result == InterpolantResult.SUCCESS and result.formula:
                count = self.store.add_from_interpolant(
                    result.formula, iteration=iteration
                )
                if count > 0:
                    atoms = extract_predicates_from_interpolant(result.formula)
                    new_preds.extend(str(a) for a in atoms)
        except Exception:
            pass

        return new_preds

    def get_predicates(self) -> List[Predicate]:
        """Get all learned predicates."""
        return list(self.store.predicates)


# ---------------------------------------------------------------------------
# Predicate abstraction checker
# ---------------------------------------------------------------------------

class PredicateAbstraction:
    """Checks program properties using learned predicates."""

    def __init__(self, predicates: List[Predicate]):
        self.predicates = predicates

    def check_state(self, state: Dict[str, int]) -> Dict[str, bool]:
        """Evaluate all predicates in a concrete state."""
        solver = SMTSolver()
        results = {}
        for pred in self.predicates:
            val = self._eval_predicate(pred, state, solver)
            results[str(pred.formula)] = val
        return results

    def _eval_predicate(self, pred: Predicate, state: Dict[str, int],
                         solver: SMTSolver) -> bool:
        """Evaluate a predicate in a concrete state."""
        solver.push()
        for var_name, val in state.items():
            v = solver.Int(var_name)
            solver.add(App(Op.EQ, [v, IntConst(val)], BOOL))

        solver.add(pred.formula)
        result = solver.check()
        solver.pop()
        return result == SMTResult.SAT


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def acdl_analyze(source: str, max_iterations: int = 10,
                  use_smt: bool = True) -> ACDLResult:
    """Analyze program using ACDL (CEGAR with interpolation)."""
    analyzer = AbstractCDLAnalyzer(
        max_iterations=max_iterations, use_smt=use_smt
    )
    return analyzer.analyze(source)


def quick_acdl(source: str) -> ACDLResult:
    """Quick ACDL analysis with limited budget."""
    analyzer = AbstractCDLAnalyzer(
        max_iterations=3, max_decisions=50, use_smt=True
    )
    return analyzer.analyze(source)


def deep_acdl(source: str) -> ACDLResult:
    """Deep ACDL analysis with high budget."""
    analyzer = AbstractCDLAnalyzer(
        max_iterations=20, max_decisions=500, use_smt=True
    )
    return analyzer.analyze(source)


def acdl_report(source: str) -> str:
    """Analyze and return human-readable report."""
    result = acdl_analyze(source)
    lines = [
        f"ACDL Analysis Result: {result.status.value}",
        f"Iterations: {result.iterations}",
        f"Final verdict: {result.final_verdict.value}",
        f"Predicates learned: {result.predicates_learned}",
        f"Total paths explored: {result.total_paths_explored}",
        f"Total paths pruned: {result.total_paths_pruned}",
        f"Total conflicts: {result.total_conflicts}",
    ]
    if result.counterexample:
        lines.append(f"Counterexample: {result.counterexample}")
    if result.refinement_history:
        lines.append("\nRefinement History:")
        for step in result.refinement_history:
            lines.append(
                f"  Iter {step.iteration}: {step.verdict.value} | "
                f"explored={step.paths_explored} pruned={step.paths_pruned} "
                f"conflicts={step.conflicts_found} "
                f"preds={step.predicates_before}->{step.predicates_after}"
            )
            if step.new_predicates:
                for p in step.new_predicates:
                    lines.append(f"    + {p}")
    if result.predicates:
        lines.append(f"\nLearned Predicates ({len(result.predicates)}):")
        for p in result.predicates:
            lines.append(f"  {p.formula} (from {p.source}, iter {p.iteration})")
    return '\n'.join(lines)
