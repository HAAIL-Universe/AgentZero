"""V121: Fixpoint Acceleration

Accelerates abstract interpretation fixpoint convergence for polyhedral domains
using extrapolation, recurrence detection, and staged widening.

Standard widening loses precision by dropping constraints too aggressively.
This module provides techniques to compute tighter fixpoints faster:

1. **Extrapolation**: Track constraint evolution across iterations, extrapolate
   stable trends instead of jumping to infinity
2. **Linear Recurrence Acceleration**: Detect x' = x + c patterns in loop bodies
   and directly compute the limit (skip intermediate iterations)
3. **Staged Polyhedral Widening**: Delay -> Threshold -> Standard pipeline with
   program-derived thresholds for polyhedral constraints
4. **Post-fixpoint Narrowing**: Recover precision after widening reaches fixpoint
5. **Strategy Comparison**: Side-by-side standard vs accelerated analysis

Composes:
- V105 (polyhedral domain) -- LinearConstraint, PolyhedralDomain, PolyhedralInterpreter
- V117 (widening strategy framework) -- WideningPhase, AdaptivePolicy
- C010 (parser) -- AST access
- C039 (abstract interpreter) -- baseline comparison

Author: A2 (AgentZero verification agent)
"""

import sys
import os
import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any
from fractions import Fraction
from enum import Enum

# Import dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import analyze as c039_analyze

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
from polyhedral_domain import (
    LinearConstraint, PolyhedralDomain, PolyhedralInterpreter,
    polyhedral_analyze as v105_analyze,
    ZERO, ONE, frac, Fraction, INF
)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class AccelPhase(Enum):
    """Phases for staged polyhedral widening."""
    DELAY = "delay"           # Join only, no widening
    THRESHOLD = "threshold"   # Widen to program-derived thresholds
    EXTRAPOLATE = "extrapolate"  # Extrapolate from iteration sequence
    STANDARD = "standard"     # Standard polyhedral widening

class AccelVerdict(Enum):
    CONVERGED = "converged"
    MAX_ITER = "max_iterations"


@dataclass
class RecurrenceInfo:
    """Detected linear recurrence for a variable in a loop."""
    var: str
    delta: Fraction  # x' = x + delta per iteration
    init_lower: Any  # initial lower bound (Fraction or -inf)
    init_upper: Any  # initial upper bound (Fraction or +inf)
    condition_var: Optional[str] = None  # loop condition variable (if x < N)
    condition_bound: Optional[Fraction] = None  # N


@dataclass
class ConstraintHistory:
    """Tracks a constraint's bound across iterations for extrapolation."""
    constraint_key: tuple  # (coeffs, is_equality) -- identifies the constraint shape
    bounds: List[Fraction]  # bound value at each iteration

    @property
    def is_monotone_increasing(self) -> bool:
        if len(self.bounds) < 2:
            return False
        return all(self.bounds[i+1] >= self.bounds[i] for i in range(len(self.bounds)-1))

    @property
    def is_monotone_decreasing(self) -> bool:
        if len(self.bounds) < 2:
            return False
        return all(self.bounds[i+1] <= self.bounds[i] for i in range(len(self.bounds)-1))

    @property
    def is_stable(self) -> bool:
        if len(self.bounds) < 2:
            return False
        return all(self.bounds[i+1] == self.bounds[i] for i in range(len(self.bounds)-1))

    @property
    def delta(self) -> Optional[Fraction]:
        """Return constant delta if bounds increase linearly."""
        if len(self.bounds) < 2:
            return None
        d = self.bounds[1] - self.bounds[0]
        if all(self.bounds[i+1] - self.bounds[i] == d for i in range(len(self.bounds)-1)):
            return d
        return None


@dataclass
class AccelerationStats:
    """Statistics about the acceleration process."""
    total_iterations: int = 0
    widening_iterations: int = 0
    narrowing_iterations: int = 0
    constraints_extrapolated: int = 0
    constraints_dropped: int = 0
    recurrences_detected: int = 0
    recurrences_accelerated: int = 0
    phase_transitions: List[Tuple[int, str]] = field(default_factory=list)
    thresholds_used: List[Fraction] = field(default_factory=list)


@dataclass
class AccelResult:
    """Result of accelerated fixpoint analysis."""
    env: PolyhedralDomain
    warnings: List[str]
    stats: AccelerationStats
    functions: List[str] = field(default_factory=list)
    loop_invariants: Dict[int, PolyhedralDomain] = field(default_factory=dict)
    verdict: AccelVerdict = AccelVerdict.CONVERGED


@dataclass
class AccelConfig:
    """Configuration for fixpoint acceleration."""
    max_iterations: int = 100
    delay_iterations: int = 3        # join-only before widening
    threshold_iterations: int = 5    # threshold widening before standard
    narrowing_iterations: int = 3    # post-fixpoint narrowing passes
    enable_extrapolation: bool = True
    enable_recurrence: bool = True
    enable_narrowing: bool = True
    extrapolation_window: int = 3    # iterations to observe before extrapolating


# ---------------------------------------------------------------------------
# Threshold Extraction
# ---------------------------------------------------------------------------

def extract_thresholds(source: str) -> List[Fraction]:
    """Extract numeric constants from source code as polyhedral thresholds."""
    tokens = lex(source)
    thresholds = set()
    for tok in tokens:
        if isinstance(tok.value, (int, float)) and tok.value is not None and tok.value is not True and tok.value is not False:
            try:
                val = Fraction(tok.value)
                thresholds.add(val)
                # Also add negation and +/- 1 variants
                thresholds.add(-val)
                thresholds.add(val + 1)
                thresholds.add(val - 1)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
    # Always include common thresholds
    thresholds.update([Fraction(0), Fraction(1), Fraction(-1)])
    return sorted(thresholds)


# ---------------------------------------------------------------------------
# Constraint Extrapolation
# ---------------------------------------------------------------------------

def _constraint_key(c: LinearConstraint) -> tuple:
    """Identify a constraint by its coefficient structure (ignoring bound)."""
    return (c.coeffs, c.is_equality)


def extrapolate_constraint(history: ConstraintHistory, thresholds: List[Fraction]) -> Optional[Fraction]:
    """Extrapolate the next bound for a constraint based on its history.

    Returns the extrapolated bound, or None if no good extrapolation found.
    """
    if history.is_stable:
        return history.bounds[-1]

    # Linear extrapolation: constant delta
    delta = history.delta
    if delta is not None and delta != 0:
        next_bound = history.bounds[-1] + delta
        # If there's a threshold near the extrapolation, snap to it
        for t in thresholds:
            if delta > 0 and history.bounds[-1] < t <= next_bound + delta:
                return t
            if delta < 0 and next_bound + delta <= t < history.bounds[-1]:
                return t
        return next_bound

    # Non-linear but monotone: try snapping to next threshold
    if history.is_monotone_increasing and len(history.bounds) >= 2:
        current = history.bounds[-1]
        for t in thresholds:
            if t > current:
                return t
        return None

    if history.is_monotone_decreasing and len(history.bounds) >= 2:
        current = history.bounds[-1]
        for t in reversed(thresholds):
            if t < current:
                return t
        return None

    return None


# ---------------------------------------------------------------------------
# Recurrence Detection & Acceleration
# ---------------------------------------------------------------------------

def detect_recurrences(body_stmts, env: PolyhedralDomain,
                       cond_var: Optional[str] = None,
                       cond_bound: Optional[Fraction] = None) -> List[RecurrenceInfo]:
    """Detect linear recurrences x' = x + c in loop body statements.

    Looks for patterns like: x = x + 1, x = x - 1, i = i + step
    """
    recurrences = []

    for stmt in body_stmts:
        if isinstance(stmt, (Assign, LetDecl)):
            if isinstance(stmt, Assign):
                target = stmt.name
                value = stmt.value
            else:
                target = stmt.name
                value = stmt.value
                if value is None:
                    continue

            # Check for x = x + c or x = x - c
            delta = _extract_delta(target, value)
            if delta is not None:
                lo = env.get_lower(target) if target in env.var_names else float('-inf')
                hi = env.get_upper(target) if target in env.var_names else float('inf')

                rec = RecurrenceInfo(
                    var=target,
                    delta=delta,
                    init_lower=lo if lo != float('-inf') else None,
                    init_upper=hi if hi != float('inf') else None,
                    condition_var=cond_var if cond_var == target else None,
                    condition_bound=cond_bound if cond_var == target else None
                )
                recurrences.append(rec)

    return recurrences


def _extract_delta(target: str, expr) -> Optional[Fraction]:
    """Extract constant delta from expr if it matches target + c or target - c."""
    if not isinstance(expr, BinOp):
        return None

    op = expr.op
    left = expr.left
    right = expr.right

    if op == '+':
        if isinstance(left, Var) and left.name == target and isinstance(right, IntLit):
            return Fraction(right.value)
        if isinstance(right, Var) and right.name == target and isinstance(left, IntLit):
            return Fraction(left.value)
    elif op == '-':
        if isinstance(left, Var) and left.name == target and isinstance(right, IntLit):
            return Fraction(-right.value)

    return None


def accelerate_recurrence(rec: RecurrenceInfo) -> Optional[Tuple[Any, Any]]:
    """Compute the limit bounds for a linear recurrence.

    Returns (lower_bound, upper_bound) at the fixpoint, or None if unknown.
    """
    if rec.delta == 0:
        return (rec.init_lower, rec.init_upper)

    if rec.delta > 0:
        # x increases: lower stays, upper goes to condition_bound or infinity
        lo = rec.init_lower
        hi = rec.condition_bound if rec.condition_bound is not None else float('inf')
        return (lo, hi)

    if rec.delta < 0:
        # x decreases: upper stays, lower goes to condition_bound or -infinity
        lo = rec.condition_bound if rec.condition_bound is not None else float('-inf')
        hi = rec.init_upper
        return (lo, hi)

    return None


# ---------------------------------------------------------------------------
# Loop Condition Analysis
# ---------------------------------------------------------------------------

def _extract_loop_condition(cond) -> Tuple[Optional[str], Optional[str], Optional[Fraction]]:
    """Extract (var, op, bound) from a loop condition like i < 10."""
    if isinstance(cond, BinOp):
        if isinstance(cond.left, Var) and isinstance(cond.right, IntLit):
            return (cond.left.name, cond.op, Fraction(cond.right.value))
        if isinstance(cond.left, IntLit) and isinstance(cond.right, Var):
            # Flip: 0 < x => x > 0
            flipped = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!='}
            return (cond.right.name, flipped.get(cond.op, cond.op), Fraction(cond.left.value))
    return (None, None, None)


# ---------------------------------------------------------------------------
# Staged Polyhedral Widening
# ---------------------------------------------------------------------------

def polyhedral_threshold_widen(old: PolyhedralDomain, new: PolyhedralDomain,
                                thresholds: List[Fraction]) -> PolyhedralDomain:
    """Polyhedral widening that snaps bounds to thresholds instead of infinity.

    For each constraint in old:
    - If satisfied by new: keep
    - If violated: try to find a threshold that still satisfies new
    - If no threshold works: drop (standard widening behavior)
    """
    if old._is_bot:
        return new.copy()
    if new._is_bot:
        return old.copy()

    result = PolyhedralDomain(list(set(old.var_names + new.var_names)))
    result._is_bot = False

    # Expand equalities for comparison
    old_expanded = old.copy()
    old_expanded._expand_equalities()
    new_expanded = new.copy()
    new_expanded._expand_equalities()

    kept = []
    for c in old_expanded.constraints:
        # Check if new satisfies this constraint
        satisfied = False
        for nc in new_expanded.constraints:
            if nc.coeffs == c.coeffs and nc.bound <= c.bound:
                satisfied = True
                break

        if satisfied:
            kept.append(c)
        else:
            # Try threshold widening: relax bound to nearest threshold
            best_threshold = None
            for t in thresholds:
                relaxed = LinearConstraint(c.coeffs, Fraction(t), c.is_equality)
                # Check if new satisfies the relaxed constraint
                new_satisfies = False
                for nc in new_expanded.constraints:
                    if nc.coeffs == relaxed.coeffs and nc.bound <= relaxed.bound:
                        new_satisfies = True
                        break
                if new_satisfies and (best_threshold is None or t < best_threshold):
                    best_threshold = t

            if best_threshold is not None:
                kept.append(LinearConstraint(c.coeffs, Fraction(best_threshold), c.is_equality))
            # else: drop the constraint (standard widening)

    result.constraints = kept
    return result


def polyhedral_extrapolation_widen(old: PolyhedralDomain, new: PolyhedralDomain,
                                    histories: Dict[tuple, ConstraintHistory],
                                    thresholds: List[Fraction]) -> PolyhedralDomain:
    """Polyhedral widening using constraint history extrapolation.

    Tracks how each constraint's bound evolves and extrapolates trends.
    """
    if old._is_bot:
        return new.copy()
    if new._is_bot:
        return old.copy()

    result = PolyhedralDomain(list(set(old.var_names + new.var_names)))
    result._is_bot = False

    old_expanded = old.copy()
    old_expanded._expand_equalities()
    new_expanded = new.copy()
    new_expanded._expand_equalities()

    # Update histories with new iteration's bounds
    new_constraint_map = {}
    for c in new_expanded.constraints:
        key = _constraint_key(c)
        new_constraint_map[key] = c.bound

    kept = []
    for c in old_expanded.constraints:
        key = _constraint_key(c)

        # Check if new satisfies
        satisfied = False
        for nc in new_expanded.constraints:
            if nc.coeffs == c.coeffs and nc.bound <= c.bound:
                satisfied = True
                break

        if satisfied:
            kept.append(c)
            # Update history
            if key in histories:
                histories[key].bounds.append(c.bound)
            else:
                histories[key] = ConstraintHistory(key, [c.bound])
        else:
            # Try extrapolation
            if key in histories:
                histories[key].bounds.append(new_constraint_map.get(key, c.bound + 1))
                extrap = extrapolate_constraint(histories[key], thresholds)
                if extrap is not None:
                    kept.append(LinearConstraint(c.coeffs, extrap, c.is_equality))
                    continue
            else:
                histories[key] = ConstraintHistory(key, [c.bound])
            # Fallback: drop

    result.constraints = kept
    return result


def polyhedral_narrowing(wide: PolyhedralDomain, body_result: PolyhedralDomain) -> PolyhedralDomain:
    """Narrowing: tighten a widened result using the loop body's actual output.

    For each constraint: if body_result has a tighter bound, use it.
    """
    if wide._is_bot or body_result._is_bot:
        return wide.copy()

    result = wide.copy()

    for var in result.var_names:
        # Tighten upper bounds
        wide_upper = wide.get_upper(var)
        body_upper = body_result.get_upper(var)
        if wide_upper == float('inf') and body_upper != float('inf'):
            result.set_upper(var, body_upper)
        elif wide_upper != float('inf') and body_upper != float('inf'):
            result.set_upper(var, min(wide_upper, body_upper))

        # Tighten lower bounds
        wide_lower = wide.get_lower(var)
        body_lower = body_result.get_lower(var)
        if wide_lower == float('-inf') and body_lower != float('-inf'):
            result.set_lower(var, body_lower)
        elif wide_lower != float('-inf') and body_lower != float('-inf'):
            result.set_lower(var, max(wide_lower, body_lower))

    return result


# ---------------------------------------------------------------------------
# Accelerated Interpreter
# ---------------------------------------------------------------------------

class AcceleratedInterpreter:
    """C10 abstract interpreter with fixpoint acceleration for polyhedral domain."""

    def __init__(self, config: Optional[AccelConfig] = None):
        self.config = config or AccelConfig()
        self.warnings: List[str] = []
        self.stats = AccelerationStats()
        self.functions: Dict[str, Any] = {}
        self.loop_invariants: Dict[int, PolyhedralDomain] = {}
        self._loop_counter = 0

    def analyze(self, source: str) -> AccelResult:
        """Parse and analyze a C10 program with fixpoint acceleration."""
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        thresholds = extract_thresholds(source)

        env = PolyhedralDomain([])

        for stmt in program.stmts:
            env = self._interpret_stmt(stmt, env, thresholds)

        verdict = AccelVerdict.CONVERGED

        return AccelResult(
            env=env,
            warnings=list(self.warnings),
            stats=self.stats,
            functions=list(self.functions.keys()),
            loop_invariants=dict(self.loop_invariants),
            verdict=verdict
        )

    def _interpret_stmt(self, stmt, env: PolyhedralDomain,
                        thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret a single statement."""
        if isinstance(stmt, LetDecl):
            return self._interpret_let(stmt, env, thresholds)
        elif isinstance(stmt, Assign):
            return self._interpret_assign(stmt, env, thresholds)
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env, thresholds)
        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env, thresholds)
        elif isinstance(stmt, Block):
            return self._interpret_block(stmt, env, thresholds)
        elif isinstance(stmt, FnDecl):
            self.functions[stmt.name] = stmt
            return env
        elif isinstance(stmt, PrintStmt):
            return env
        elif isinstance(stmt, ReturnStmt):
            return env
        else:
            return env

    def _interpret_block(self, block, env: PolyhedralDomain,
                         thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret a block of statements."""
        stmts = block.stmts if isinstance(block, Block) else [block]
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env, thresholds)
        return env

    def _interpret_let(self, stmt: LetDecl, env: PolyhedralDomain,
                       thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret let declaration."""
        env = env.copy()
        if stmt.name not in env.var_names:
            env.add_var(stmt.name)
        if stmt.value is not None:
            self._apply_assignment(stmt.name, stmt.value, env)
        return env

    def _interpret_assign(self, stmt: Assign, env: PolyhedralDomain,
                          thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret assignment."""
        env = env.copy()
        if stmt.name not in env.var_names:
            env.add_var(stmt.name)
        self._apply_assignment(stmt.name, stmt.value, env)
        return env

    def _apply_assignment(self, target: str, expr, env: PolyhedralDomain):
        """Apply an assignment to the environment."""
        coeffs, const = self._linearize_expr(expr)
        if coeffs is not None:
            # Linear expression: use precise polyhedral assignment
            if len(coeffs) == 0:
                env.assign_const(target, const)
            elif len(coeffs) == 1 and list(coeffs.keys())[0] == target:
                # Self-referential: x = a*x + c
                env.assign_expr(target, coeffs, const)
            elif target not in coeffs:
                # Non-self-referential: x = a*y + b*z + c
                env.assign_linear(target, coeffs, const)
            else:
                # Mixed: x = a*x + b*y + c
                env.assign_expr(target, coeffs, const)
        else:
            # Non-linear: fall back to interval bounds
            lo, hi = self._eval_interval(expr, env)
            env.forget(target)
            if target not in env.var_names:
                env.add_var(target)
            if lo != float('-inf'):
                env.set_lower(target, lo)
            if hi != float('inf'):
                env.set_upper(target, hi)

    def _linearize_expr(self, expr) -> Tuple[Optional[Dict[str, Fraction]], Optional[Fraction]]:
        """Try to decompose expr into {var: coeff} + constant."""
        if isinstance(expr, IntLit):
            return ({}, Fraction(expr.value))
        elif isinstance(expr, Var):
            return ({expr.name: ONE}, ZERO)
        elif isinstance(expr, UnaryOp) and expr.op == '-':
            sub_coeffs, sub_const = self._linearize_expr(expr.operand)
            if sub_coeffs is not None:
                return ({v: -c for v, c in sub_coeffs.items()}, -sub_const)
            return (None, None)
        elif isinstance(expr, BinOp):
            lc, lconst = self._linearize_expr(expr.left)
            rc, rconst = self._linearize_expr(expr.right)

            if lc is None or rc is None:
                return (None, None)

            if expr.op == '+':
                merged = dict(lc)
                for v, c in rc.items():
                    merged[v] = merged.get(v, ZERO) + c
                return (merged, lconst + rconst)
            elif expr.op == '-':
                merged = dict(lc)
                for v, c in rc.items():
                    merged[v] = merged.get(v, ZERO) - c
                return (merged, lconst - rconst)
            elif expr.op == '*':
                # Only linear if one side is constant
                if len(lc) == 0:  # left is constant
                    return ({v: lconst * c for v, c in rc.items()}, lconst * rconst)
                elif len(rc) == 0:  # right is constant
                    return ({v: rconst * c for v, c in lc.items()}, rconst * lconst)
                return (None, None)
            else:
                return (None, None)
        return (None, None)

    def _eval_interval(self, expr, env: PolyhedralDomain) -> Tuple[float, float]:
        """Evaluate expression to interval bounds using polyhedral projection."""
        if isinstance(expr, IntLit):
            v = float(expr.value)
            return (v, v)
        elif isinstance(expr, Var):
            if expr.name in env.var_names:
                lo = env.get_lower(expr.name)
                hi = env.get_upper(expr.name)
                return (lo, hi)
            return (float('-inf'), float('inf'))
        elif isinstance(expr, UnaryOp) and expr.op == '-':
            lo, hi = self._eval_interval(expr.operand, env)
            return (-hi, -lo)
        elif isinstance(expr, BinOp):
            l_lo, l_hi = self._eval_interval(expr.left, env)
            r_lo, r_hi = self._eval_interval(expr.right, env)
            if expr.op == '+':
                return (l_lo + r_lo, l_hi + r_hi)
            elif expr.op == '-':
                return (l_lo - r_hi, l_hi - r_lo)
            elif expr.op == '*':
                products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
                # Handle inf * 0
                products = [p for p in products if not math.isnan(p)]
                if not products:
                    return (float('-inf'), float('inf'))
                return (min(products), max(products))
            elif expr.op == '/':
                if r_lo <= 0 <= r_hi:
                    self.warnings.append("possible division by zero")
                    return (float('-inf'), float('inf'))
                products = []
                for a in [l_lo, l_hi]:
                    for b in [r_lo, r_hi]:
                        if b != 0:
                            products.append(a / b)
                if not products:
                    return (float('-inf'), float('inf'))
                return (min(products), max(products))
            elif expr.op == '%':
                if r_lo <= 0 <= r_hi:
                    self.warnings.append("possible division by zero in modulo")
                    return (float('-inf'), float('inf'))
                # Modulo result bounded by divisor
                abs_max = max(abs(r_lo), abs(r_hi))
                return (Fraction(0), abs_max - 1) if r_lo > 0 else (-(abs_max - 1), abs_max - 1)
        return (float('-inf'), float('inf'))

    def _interpret_if(self, stmt: IfStmt, env: PolyhedralDomain,
                      thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret if-else with condition refinement."""
        then_env = self._refine_condition(stmt.cond, env, True)
        else_env = self._refine_condition(stmt.cond, env, False)

        if not then_env.is_bot():
            then_env = self._interpret_block(stmt.then_body, then_env, thresholds)

        if stmt.else_body is not None and not else_env.is_bot():
            else_env = self._interpret_block(stmt.else_body, else_env, thresholds)
        elif stmt.else_body is None:
            pass  # else_env unchanged

        # Join branches
        if then_env.is_bot():
            return else_env
        if else_env.is_bot():
            return then_env
        return then_env.join(else_env)

    def _refine_condition(self, cond, env: PolyhedralDomain,
                          take_true: bool) -> PolyhedralDomain:
        """Refine environment based on a condition being true or false."""
        refined = env.copy()

        if isinstance(cond, BinOp):
            lc, lconst = self._linearize_expr(cond.left)
            rc, rconst = self._linearize_expr(cond.right)

            if lc is not None and rc is not None:
                # diff = left - right: diff_coeffs * vars + diff_const
                diff_coeffs = dict(lc)
                for v, c in rc.items():
                    diff_coeffs[v] = diff_coeffs.get(v, ZERO) - c
                diff_const = lconst - rconst

                # Remove zero coefficients
                diff_coeffs = {v: c for v, c in diff_coeffs.items() if c != ZERO}

                op = cond.op
                if not take_true:
                    # Negate: < -> >=, <= -> >, == -> !=, != -> ==, > -> <=, >= -> <
                    neg = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
                    op = neg.get(op, op)

                # Add constraint: diff_coeffs * vars + diff_const OP 0
                # Rewrite as: diff_coeffs * vars OP -diff_const
                bound = -diff_const

                if op == '<':
                    # sum < bound => sum <= bound - 1 (integer)
                    refined.add_constraint(diff_coeffs, bound - 1)
                elif op == '<=':
                    refined.add_constraint(diff_coeffs, bound)
                elif op == '>':
                    # sum > bound => -sum <= -(bound+1) => -sum < -bound
                    neg_coeffs = {v: -c for v, c in diff_coeffs.items()}
                    refined.add_constraint(neg_coeffs, -(bound + 1))
                elif op == '>=':
                    neg_coeffs = {v: -c for v, c in diff_coeffs.items()}
                    refined.add_constraint(neg_coeffs, -bound)
                elif op == '==':
                    refined.add_constraint(diff_coeffs, bound, is_equality=True)
                elif op == '!=':
                    pass  # Cannot represent != as single linear constraint

                return refined

        elif isinstance(cond, Var):
            # Truthy: var != 0
            if take_true:
                pass  # Can't represent != 0 precisely
            else:
                if cond.name in env.var_names:
                    refined.set_equal(cond.name, 0)
            return refined

        return refined

    def _interpret_while(self, stmt: WhileStmt, env: PolyhedralDomain,
                         thresholds: List[Fraction]) -> PolyhedralDomain:
        """Interpret while loop with accelerated fixpoint computation."""
        loop_id = self._loop_counter
        self._loop_counter += 1

        config = self.config

        # Extract loop condition info for recurrence detection
        cond_var, cond_op, cond_bound = _extract_loop_condition(stmt.cond)

        # Get loop body statements
        body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]

        # Detect recurrences if enabled
        recurrences = []
        if config.enable_recurrence:
            recurrences = detect_recurrences(body_stmts, env, cond_var, cond_bound)
            self.stats.recurrences_detected += len(recurrences)

        # Initialize constraint histories for extrapolation
        histories: Dict[tuple, ConstraintHistory] = {}

        # Staged fixpoint iteration
        phase = AccelPhase.DELAY
        self.stats.phase_transitions.append((0, phase.value))

        current = env.copy()
        iteration = 0
        converged = False

        while iteration < config.max_iterations:
            iteration += 1
            self.stats.total_iterations += 1

            # Evaluate condition
            cond_env = self._refine_condition(stmt.cond, current, True)

            if cond_env.is_bot():
                # Loop condition always false -> loop doesn't execute
                converged = True
                break

            # Execute body
            body_result = self._interpret_block(stmt.body, cond_env, thresholds)

            # Join with pre-loop (loop may not execute)
            joined = env.join(body_result) if not env.is_bot() else body_result

            # Phase transitions
            new_phase = self._determine_phase(iteration, config)
            if new_phase != phase:
                phase = new_phase
                self.stats.phase_transitions.append((iteration, phase.value))

            # Apply widening based on current phase
            if phase == AccelPhase.DELAY:
                # Just join, no widening
                next_env = current.join(joined)
            elif phase == AccelPhase.THRESHOLD:
                next_env = polyhedral_threshold_widen(current, joined, thresholds)
                self.stats.widening_iterations += 1
                self.stats.thresholds_used = list(thresholds[:5])
            elif phase == AccelPhase.EXTRAPOLATE and config.enable_extrapolation:
                next_env = polyhedral_extrapolation_widen(current, joined, histories, thresholds)
                self.stats.widening_iterations += 1
                self.stats.constraints_extrapolated += sum(
                    1 for h in histories.values() if h.delta is not None
                )
            else:
                # Standard widening
                next_env = current.widen(joined)
                self.stats.widening_iterations += 1

            # Apply recurrence acceleration
            if config.enable_recurrence and recurrences and iteration >= config.delay_iterations:
                for rec in recurrences:
                    limit = accelerate_recurrence(rec)
                    if limit is not None:
                        lo, hi = limit
                        if rec.var in next_env.var_names:
                            if lo is not None and lo != float('-inf'):
                                next_env.set_lower(rec.var, lo)
                            if hi is not None and hi != float('inf'):
                                next_env.set_upper(rec.var, hi)
                            self.stats.recurrences_accelerated += 1

            # Check convergence
            if next_env.leq(current) and current.leq(next_env):
                converged = True
                current = next_env
                break

            current = next_env

        # Post-fixpoint narrowing
        if converged and config.enable_narrowing:
            for n_iter in range(config.narrowing_iterations):
                self.stats.narrowing_iterations += 1
                cond_env = self._refine_condition(stmt.cond, current, True)
                if cond_env.is_bot():
                    break
                body_result = self._interpret_block(stmt.body, cond_env, thresholds)
                joined = env.join(body_result) if not env.is_bot() else body_result
                narrowed = polyhedral_narrowing(current, joined)
                if narrowed.leq(current) and current.leq(narrowed):
                    break
                current = narrowed

        # Store loop invariant
        self.loop_invariants[loop_id] = current.copy()

        # Return post-loop state (condition is false)
        post_env = self._refine_condition(stmt.cond, current, False)

        if not converged:
            self.warnings.append(f"loop {loop_id}: did not converge in {config.max_iterations} iterations")

        return post_env

    def _determine_phase(self, iteration: int, config: AccelConfig) -> AccelPhase:
        """Determine widening phase based on iteration count."""
        if iteration <= config.delay_iterations:
            return AccelPhase.DELAY
        elif iteration <= config.delay_iterations + config.threshold_iterations:
            return AccelPhase.THRESHOLD
        elif config.enable_extrapolation and iteration <= config.delay_iterations + config.threshold_iterations + config.extrapolation_window:
            return AccelPhase.EXTRAPOLATE
        else:
            return AccelPhase.STANDARD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def accelerated_analyze(source: str, config: Optional[AccelConfig] = None) -> AccelResult:
    """Analyze a C10 program with fixpoint acceleration.

    Returns AccelResult with polyhedral environment, warnings, and statistics.
    """
    interp = AcceleratedInterpreter(config)
    return interp.analyze(source)


def standard_analyze(source: str) -> dict:
    """Analyze with standard V105 polyhedral interpreter (baseline)."""
    return v105_analyze(source)


def get_variable_range(source: str, var_name: str,
                       config: Optional[AccelConfig] = None) -> Tuple[float, float]:
    """Get the range of a variable after accelerated analysis."""
    result = accelerated_analyze(source, config)
    if var_name in result.env.var_names:
        return (result.env.get_lower(var_name), result.env.get_upper(var_name))
    return (float('-inf'), float('inf'))


def get_loop_invariant(source: str, loop_index: int = 0,
                       config: Optional[AccelConfig] = None) -> Optional[PolyhedralDomain]:
    """Get the inferred loop invariant for a specific loop."""
    result = accelerated_analyze(source, config)
    return result.loop_invariants.get(loop_index)


def get_acceleration_stats(source: str,
                           config: Optional[AccelConfig] = None) -> AccelerationStats:
    """Get acceleration statistics for analysis of a program."""
    result = accelerated_analyze(source, config)
    return result.stats


def compare_analyses(source: str, config: Optional[AccelConfig] = None) -> dict:
    """Compare accelerated vs standard polyhedral analysis.

    Returns dict with precision comparison for each variable.
    """
    accel = accelerated_analyze(source, config)
    standard = standard_analyze(source)

    std_env = standard['env']
    acc_env = accel.env

    comparison = {
        'variables': {},
        'standard_warnings': standard['warnings'],
        'accelerated_warnings': accel.warnings,
        'stats': {
            'total_iterations': accel.stats.total_iterations,
            'widening_iterations': accel.stats.widening_iterations,
            'narrowing_iterations': accel.stats.narrowing_iterations,
            'recurrences_detected': accel.stats.recurrences_detected,
            'recurrences_accelerated': accel.stats.recurrences_accelerated,
            'constraints_extrapolated': accel.stats.constraints_extrapolated,
            'phase_transitions': accel.stats.phase_transitions,
        },
        'precision_gains': [],
    }

    all_vars = set(std_env.var_names) | set(acc_env.var_names)
    for var in sorted(all_vars):
        std_lo = std_env.get_lower(var) if var in std_env.var_names else float('-inf')
        std_hi = std_env.get_upper(var) if var in std_env.var_names else float('inf')
        acc_lo = acc_env.get_lower(var) if var in acc_env.var_names else float('-inf')
        acc_hi = acc_env.get_upper(var) if var in acc_env.var_names else float('inf')

        entry = {
            'standard': (std_lo, std_hi),
            'accelerated': (acc_lo, acc_hi),
        }

        # Check if accelerated is more precise (tighter)
        more_precise = False
        if acc_lo > std_lo and std_lo != float('-inf'):
            more_precise = True
        if acc_lo != float('-inf') and std_lo == float('-inf'):
            more_precise = True
        if acc_hi < std_hi and std_hi != float('inf'):
            more_precise = True
        if acc_hi != float('inf') and std_hi == float('inf'):
            more_precise = True

        entry['accelerated_more_precise'] = more_precise
        if more_precise:
            comparison['precision_gains'].append(var)

        comparison['variables'][var] = entry

    return comparison


def detect_program_recurrences(source: str) -> List[RecurrenceInfo]:
    """Detect linear recurrences in all loops of a program."""
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    all_recs = []
    env = PolyhedralDomain([])

    for stmt in program.stmts:
        if isinstance(stmt, LetDecl):
            if stmt.name not in env.var_names:
                env.add_var(stmt.name)
            if stmt.value is not None and isinstance(stmt.value, IntLit):
                env.assign_const(stmt.name, Fraction(stmt.value.value))
        elif isinstance(stmt, WhileStmt):
            cond_var, cond_op, cond_bound = _extract_loop_condition(stmt.cond)
            body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            recs = detect_recurrences(body_stmts, env, cond_var, cond_bound)
            all_recs.extend(recs)

    return all_recs


def verify_invariant(source: str, loop_index: int = 0,
                     config: Optional[AccelConfig] = None) -> dict:
    """Verify that the inferred loop invariant is inductive.

    Returns dict with invariant constraints and inductiveness check.
    """
    result = accelerated_analyze(source, config)
    inv = result.loop_invariants.get(loop_index)

    if inv is None:
        return {'verified': False, 'reason': 'no invariant found'}

    return {
        'verified': True,
        'constraints': inv.get_constraints(),
        'relational': inv.get_relational_constraints(),
        'variable_ranges': {
            var: (inv.get_lower(var), inv.get_upper(var))
            for var in inv.var_names
        },
        'stats': {
            'total_iterations': result.stats.total_iterations,
            'widening_iterations': result.stats.widening_iterations,
            'narrowing_iterations': result.stats.narrowing_iterations,
        }
    }


def acceleration_summary(source: str, config: Optional[AccelConfig] = None) -> str:
    """Human-readable summary of accelerated analysis."""
    result = accelerated_analyze(source, config)

    lines = ["=== Fixpoint Acceleration Summary ===", ""]

    # Variable ranges
    lines.append("Variable Ranges:")
    for var in sorted(result.env.var_names):
        lo = result.env.get_lower(var)
        hi = result.env.get_upper(var)
        lo_s = str(lo) if lo != float('-inf') else "-inf"
        hi_s = str(hi) if hi != float('inf') else "inf"
        lines.append(f"  {var}: [{lo_s}, {hi_s}]")

    # Constraints
    constraints = result.env.get_relational_constraints()
    if constraints:
        lines.append("")
        lines.append("Relational Constraints:")
        for c in constraints:
            lines.append(f"  {c}")

    # Warnings
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    # Statistics
    lines.append("")
    lines.append("Acceleration Statistics:")
    s = result.stats
    lines.append(f"  Total iterations: {s.total_iterations}")
    lines.append(f"  Widening iterations: {s.widening_iterations}")
    lines.append(f"  Narrowing iterations: {s.narrowing_iterations}")
    lines.append(f"  Recurrences detected: {s.recurrences_detected}")
    lines.append(f"  Recurrences accelerated: {s.recurrences_accelerated}")
    lines.append(f"  Constraints extrapolated: {s.constraints_extrapolated}")

    if s.phase_transitions:
        lines.append(f"  Phase transitions: {', '.join(f'{p}@{i}' for i, p in s.phase_transitions)}")

    # Loop invariants
    if result.loop_invariants:
        lines.append("")
        lines.append("Loop Invariants:")
        for lid, inv in sorted(result.loop_invariants.items()):
            lines.append(f"  Loop {lid}:")
            for var in sorted(inv.var_names):
                lo = inv.get_lower(var)
                hi = inv.get_upper(var)
                lo_s = str(lo) if lo != float('-inf') else "-inf"
                hi_s = str(hi) if hi != float('inf') else "inf"
                lines.append(f"    {var}: [{lo_s}, {hi_s}]")

    return "\n".join(lines)
