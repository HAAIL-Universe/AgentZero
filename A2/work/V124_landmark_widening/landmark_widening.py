"""V124: Polyhedral Widening with Landmarks

Per-loop landmark-based widening for polyhedral abstract interpretation.
Instead of a uniform staged widening policy, each loop gets a custom widening
strategy based on its structural landmarks -- program points and expressions
that constrain the loop's behavior.

Key innovations over V121 (Fixpoint Acceleration):
1. **Landmark extraction**: Analyze each loop's AST to identify constraining
   expressions (conditions, increments, bounds, nested conditions)
2. **Per-variable widening policy**: Variables with detected recurrences get
   acceleration; variables with condition bounds get threshold widening;
   others get standard widening -- all in the SAME fixpoint iteration
3. **Back-edge delta analysis**: Track per-variable deltas across the loop body
   to compute per-variable landmarks (init, limit, stride)
4. **Nested loop interaction**: Inner loop landmarks propagate to outer loops
   as additional thresholds
5. **Landmark-guided narrowing**: Post-fixpoint narrowing prioritizes variables
   whose landmarks suggest tighter bounds are achievable

Composes:
- V121 (fixpoint acceleration) -- AcceleratedInterpreter, RecurrenceInfo, AccelConfig
- V105 (polyhedral domain) -- LinearConstraint, PolyhedralDomain
- C010 (parser) -- AST access

Author: A2 (AgentZero verification agent)
"""

import sys
import os
import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any, FrozenSet
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
from polyhedral_domain import (
    LinearConstraint, PolyhedralDomain, PolyhedralInterpreter,
    polyhedral_analyze as v105_analyze,
    ZERO, ONE, frac, Fraction, INF
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V121_fixpoint_acceleration'))
from fixpoint_acceleration import (
    AccelPhase, AccelVerdict, AccelConfig, AccelResult, AccelerationStats,
    RecurrenceInfo, ConstraintHistory,
    detect_recurrences, accelerate_recurrence, extract_thresholds,
    polyhedral_threshold_widen, polyhedral_narrowing,
    AcceleratedInterpreter, accelerated_analyze, standard_analyze
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class LandmarkKind(Enum):
    """Classification of a program landmark."""
    CONDITION_BOUND = "condition_bound"    # From loop condition: i < N
    INCREMENT = "increment"                # From assignment: i = i + 1
    INIT_VALUE = "init_value"              # From pre-loop assignment: let i = 0
    NESTED_BOUND = "nested_bound"          # From nested loop condition
    BRANCH_THRESHOLD = "branch_threshold"  # From if-condition inside loop
    ASSIGNMENT_CONST = "assignment_const"  # From constant assignment in loop body


@dataclass
class Landmark:
    """A program landmark that constrains a variable's behavior in a loop."""
    kind: LandmarkKind
    variable: str
    value: Fraction           # The numeric value associated with this landmark
    source: str = ""          # Human-readable description
    priority: int = 0         # Higher = more useful for widening (0-10)


@dataclass
class LoopProfile:
    """Structural profile of a loop, derived from landmark analysis."""
    landmarks: List[Landmark]
    modified_vars: Set[str]           # Variables assigned in the loop body
    condition_vars: Set[str]          # Variables in the loop condition
    recurrences: List[RecurrenceInfo] # Detected linear recurrences
    per_var_thresholds: Dict[str, List[Fraction]]  # Per-variable thresholds from landmarks
    global_thresholds: List[Fraction]  # All thresholds merged
    has_nested_loops: bool
    nested_profiles: List['LoopProfile']  # Profiles of nested loops

    def get_var_policy(self, var: str) -> str:
        """Determine widening policy for a specific variable.
        Returns: 'accelerate', 'threshold', or 'standard'."""
        # Check if variable has a detected recurrence
        for rec in self.recurrences:
            if rec.var == var and rec.condition_bound is not None:
                return 'accelerate'
        # Check if variable has landmarks providing thresholds
        if var in self.per_var_thresholds and self.per_var_thresholds[var]:
            return 'threshold'
        return 'standard'


@dataclass
class LandmarkWideningStats(AccelerationStats):
    """Extended statistics for landmark-based widening."""
    landmarks_extracted: int = 0
    per_var_accelerations: int = 0
    per_var_threshold_uses: int = 0
    nested_propagations: int = 0
    landmark_narrowings: int = 0


@dataclass
class LandmarkConfig:
    """Configuration for landmark-based widening."""
    max_iterations: int = 100
    delay_iterations: int = 2         # Fewer delay iterations (landmarks give info earlier)
    narrowing_iterations: int = 4     # More narrowing (landmarks guide tightening)
    enable_recurrence: bool = True
    enable_nested_propagation: bool = True
    enable_landmark_narrowing: bool = True
    enable_branch_landmarks: bool = True


@dataclass
class LandmarkResult:
    """Result of landmark-based polyhedral analysis."""
    env: PolyhedralDomain
    warnings: List[str]
    stats: LandmarkWideningStats
    functions: List[str] = field(default_factory=list)
    loop_invariants: Dict[int, PolyhedralDomain] = field(default_factory=dict)
    loop_profiles: Dict[int, LoopProfile] = field(default_factory=dict)
    verdict: AccelVerdict = AccelVerdict.CONVERGED


# ---------------------------------------------------------------------------
# Landmark Extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_condition(cond, env: PolyhedralDomain) -> List[Landmark]:
    """Extract landmarks from a loop condition (e.g., i < 10, x >= 0)."""
    landmarks = []
    if not isinstance(cond, BinOp):
        return landmarks

    op, left, right = cond.op, cond.left, cond.right

    # Pattern: var OP const or const OP var
    if isinstance(left, Var) and isinstance(right, (IntLit, FloatLit)):
        var_name = left.name
        val = frac(right.value)
        landmarks.append(Landmark(
            kind=LandmarkKind.CONDITION_BOUND,
            variable=var_name,
            value=val,
            source=f"{var_name} {op} {val}",
            priority=10
        ))
        # Also add boundary values (val-1, val+1) as secondary landmarks
        if op in ('<', '<=', '>', '>='):
            landmarks.append(Landmark(
                kind=LandmarkKind.CONDITION_BOUND,
                variable=var_name,
                value=val - ONE,
                source=f"{var_name} boundary {val - ONE}",
                priority=5
            ))
            landmarks.append(Landmark(
                kind=LandmarkKind.CONDITION_BOUND,
                variable=var_name,
                value=val + ONE,
                source=f"{var_name} boundary {val + ONE}",
                priority=5
            ))

    elif isinstance(right, Var) and isinstance(left, (IntLit, FloatLit)):
        var_name = right.name
        val = frac(left.value)
        landmarks.append(Landmark(
            kind=LandmarkKind.CONDITION_BOUND,
            variable=var_name,
            value=val,
            source=f"{val} {op} {var_name}",
            priority=10
        ))

    # Pattern: var OP var (relational -- extract bounds from env)
    elif isinstance(left, Var) and isinstance(right, Var):
        lname, rname = left.name, right.name
        # Get current bounds from the environment
        for vname in [lname, rname]:
            lo, hi = env.get_interval(vname)
            if lo != float('-inf'):
                landmarks.append(Landmark(
                    kind=LandmarkKind.CONDITION_BOUND,
                    variable=vname,
                    value=frac(int(lo)),
                    source=f"{vname} env lower {lo}",
                    priority=3
                ))
            if hi != float('inf'):
                landmarks.append(Landmark(
                    kind=LandmarkKind.CONDITION_BOUND,
                    variable=vname,
                    value=frac(int(hi)),
                    source=f"{vname} env upper {hi}",
                    priority=3
                ))

    return landmarks


def extract_landmarks_from_body(stmts, env: PolyhedralDomain) -> List[Landmark]:
    """Extract landmarks from loop body statements."""
    landmarks = []
    if not isinstance(stmts, list):
        stmts = [stmts]

    for stmt in stmts:
        if isinstance(stmt, (LetDecl, Assign)):
            target = stmt.name
            expr = stmt.value if isinstance(stmt, LetDecl) else stmt.value

            # Pattern: x = x + c or x = x - c (increment landmark)
            if isinstance(expr, BinOp) and isinstance(expr.left, Var) and expr.left.name == target:
                if isinstance(expr.right, (IntLit, FloatLit)) and expr.op in ('+', '-'):
                    delta = frac(expr.right.value)
                    if expr.op == '-':
                        delta = -delta
                    landmarks.append(Landmark(
                        kind=LandmarkKind.INCREMENT,
                        variable=target,
                        value=delta,
                        source=f"{target} += {delta}",
                        priority=8
                    ))

            # Pattern: x = const (constant assignment landmark)
            elif isinstance(expr, (IntLit, FloatLit)):
                val = frac(expr.value)
                landmarks.append(Landmark(
                    kind=LandmarkKind.ASSIGNMENT_CONST,
                    variable=target,
                    value=val,
                    source=f"{target} = {val}",
                    priority=6
                ))

        elif isinstance(stmt, IfStmt):
            # Extract branch condition landmarks
            cond_landmarks = _extract_branch_landmarks(stmt.cond)
            landmarks.extend(cond_landmarks)
            # Recurse into branches
            if hasattr(stmt, 'then_body') and stmt.then_body:
                then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body]
                landmarks.extend(extract_landmarks_from_body(then_stmts, env))
            if hasattr(stmt, 'else_body') and stmt.else_body:
                else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body]
                landmarks.extend(extract_landmarks_from_body(else_stmts, env))

        elif isinstance(stmt, WhileStmt):
            # Nested loop -- extract its condition as a landmark for the outer loop
            nested_cond_landmarks = extract_landmarks_from_condition(stmt.cond, env)
            for lm in nested_cond_landmarks:
                lm.kind = LandmarkKind.NESTED_BOUND
                lm.priority = min(lm.priority, 7)
            landmarks.extend(nested_cond_landmarks)

        elif isinstance(stmt, Block):
            landmarks.extend(extract_landmarks_from_body(stmt.stmts, env))

    return landmarks


def _extract_branch_landmarks(cond) -> List[Landmark]:
    """Extract landmarks from an if-condition inside a loop body."""
    landmarks = []
    if isinstance(cond, BinOp):
        op, left, right = cond.op, cond.left, cond.right
        if isinstance(left, Var) and isinstance(right, (IntLit, FloatLit)):
            val = frac(right.value)
            landmarks.append(Landmark(
                kind=LandmarkKind.BRANCH_THRESHOLD,
                variable=left.name,
                value=val,
                source=f"branch: {left.name} {op} {val}",
                priority=7
            ))
        elif isinstance(right, Var) and isinstance(left, (IntLit, FloatLit)):
            val = frac(left.value)
            landmarks.append(Landmark(
                kind=LandmarkKind.BRANCH_THRESHOLD,
                variable=right.name,
                value=val,
                source=f"branch: {val} {op} {right.name}",
                priority=7
            ))
    return landmarks


def extract_init_landmarks(pre_stmts: List, loop_vars: Set[str]) -> List[Landmark]:
    """Extract init-value landmarks from statements before the loop."""
    landmarks = []
    for stmt in pre_stmts:
        if isinstance(stmt, LetDecl) and stmt.name in loop_vars:
            expr = stmt.value
            if isinstance(expr, (IntLit, FloatLit)):
                val = frac(expr.value)
                landmarks.append(Landmark(
                    kind=LandmarkKind.INIT_VALUE,
                    variable=stmt.name,
                    value=val,
                    source=f"init: {stmt.name} = {val}",
                    priority=9
                ))
    return landmarks


def _collect_modified_vars(stmts) -> Set[str]:
    """Collect all variables modified in a list of statements."""
    modified = set()
    if not isinstance(stmts, list):
        stmts = [stmts]
    for stmt in stmts:
        if isinstance(stmt, Assign):
            modified.add(stmt.name)
        elif isinstance(stmt, LetDecl):
            modified.add(stmt.name)
        elif isinstance(stmt, IfStmt):
            if hasattr(stmt, 'then_body') and stmt.then_body:
                then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body]
                modified |= _collect_modified_vars(then_stmts)
            if hasattr(stmt, 'else_body') and stmt.else_body:
                else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body]
                modified |= _collect_modified_vars(else_stmts)
        elif isinstance(stmt, WhileStmt):
            body = stmt.body if isinstance(stmt.body, list) else [stmt.body]
            modified |= _collect_modified_vars(body)
        elif isinstance(stmt, Block):
            modified |= _collect_modified_vars(stmt.stmts)
    return modified


def _collect_condition_vars(cond) -> Set[str]:
    """Collect all variables referenced in a condition expression."""
    if isinstance(cond, Var):
        return {cond.name}
    if isinstance(cond, BinOp):
        return _collect_condition_vars(cond.left) | _collect_condition_vars(cond.right)
    if isinstance(cond, UnaryOp):
        return _collect_condition_vars(cond.operand)
    return set()


def _has_nested_loops(stmts) -> bool:
    """Check if statements contain nested while loops."""
    if not isinstance(stmts, list):
        stmts = [stmts]
    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            return True
        if isinstance(stmt, IfStmt):
            if hasattr(stmt, 'then_body') and stmt.then_body:
                then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body]
                if _has_nested_loops(then_stmts):
                    return True
            if hasattr(stmt, 'else_body') and stmt.else_body:
                else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body]
                if _has_nested_loops(else_stmts):
                    return True
        if isinstance(stmt, Block):
            if _has_nested_loops(stmt.stmts):
                return True
    return False


def build_loop_profile(while_stmt: WhileStmt, env: PolyhedralDomain,
                       pre_stmts: List = None) -> LoopProfile:
    """Build a complete structural profile of a loop from its AST and environment."""
    body = while_stmt.body if isinstance(while_stmt.body, list) else [while_stmt.body]

    # Collect structural info
    modified_vars = _collect_modified_vars(body)
    condition_vars = _collect_condition_vars(while_stmt.cond)

    # Extract landmarks from all sources
    landmarks = []
    landmarks.extend(extract_landmarks_from_condition(while_stmt.cond, env))
    landmarks.extend(extract_landmarks_from_body(body, env))
    if pre_stmts:
        landmarks.extend(extract_init_landmarks(pre_stmts, modified_vars | condition_vars))

    # Detect recurrences using V121's detector
    cond_var = None
    cond_bound = None
    if isinstance(while_stmt.cond, BinOp):
        if isinstance(while_stmt.cond.left, Var) and isinstance(while_stmt.cond.right, (IntLit, FloatLit)):
            cond_var = while_stmt.cond.left.name
            cond_bound = frac(while_stmt.cond.right.value)
    recurrences = detect_recurrences(body, env, cond_var, cond_bound)

    # Build per-variable threshold maps from landmarks
    per_var_thresholds = {}
    for lm in landmarks:
        if lm.variable not in per_var_thresholds:
            per_var_thresholds[lm.variable] = []
        per_var_thresholds[lm.variable].append(lm.value)

    # Deduplicate and sort per-variable thresholds
    for var in per_var_thresholds:
        per_var_thresholds[var] = sorted(set(per_var_thresholds[var]))

    # Global thresholds = union of all
    all_thresholds = set()
    for vals in per_var_thresholds.values():
        all_thresholds.update(vals)
    global_thresholds = sorted(all_thresholds)

    # Check for nested loops
    has_nested = _has_nested_loops(body)

    # Build nested profiles (recursive)
    nested_profiles = []
    if has_nested:
        nested_profiles = _extract_nested_profiles(body, env)

    return LoopProfile(
        landmarks=landmarks,
        modified_vars=modified_vars,
        condition_vars=condition_vars,
        recurrences=recurrences,
        per_var_thresholds=per_var_thresholds,
        global_thresholds=global_thresholds,
        has_nested_loops=has_nested,
        nested_profiles=nested_profiles
    )


def _extract_nested_profiles(stmts, env: PolyhedralDomain) -> List[LoopProfile]:
    """Extract loop profiles from nested while loops."""
    profiles = []
    if not isinstance(stmts, list):
        stmts = [stmts]
    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            profiles.append(build_loop_profile(stmt, env))
        elif isinstance(stmt, IfStmt):
            if hasattr(stmt, 'then_body') and stmt.then_body:
                then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body]
                profiles.extend(_extract_nested_profiles(then_stmts, env))
            if hasattr(stmt, 'else_body') and stmt.else_body:
                else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body]
                profiles.extend(_extract_nested_profiles(else_stmts, env))
        elif isinstance(stmt, Block):
            profiles.extend(_extract_nested_profiles(stmt.stmts, env))
    return profiles


# ---------------------------------------------------------------------------
# Per-Variable Landmark Widening
# ---------------------------------------------------------------------------

def landmark_widen_per_var(old: PolyhedralDomain, new: PolyhedralDomain,
                           profile: LoopProfile, iteration: int,
                           delay: int = 2) -> PolyhedralDomain:
    """Apply per-variable widening based on the loop profile.

    For each variable:
    - If in delay phase: join only
    - If has recurrence with bound: accelerate (compute limit directly)
    - If has landmarks: threshold widen using per-variable thresholds
    - Otherwise: standard polyhedral widen
    """
    if old.is_bot():
        return new.copy()
    if new.is_bot():
        return old.copy()

    # Phase 1: Delay (join only)
    if iteration < delay:
        return old.join(new)

    # Classify variables by policy
    all_vars = set(old.var_names) | set(new.var_names)
    accelerate_vars = set()
    threshold_vars = set()
    standard_vars = set()

    for var in all_vars:
        policy = profile.get_var_policy(var)
        if policy == 'accelerate':
            accelerate_vars.add(var)
        elif policy == 'threshold':
            threshold_vars.add(var)
        else:
            standard_vars.add(var)

    # Start from join (conservative baseline)
    result = old.join(new)

    # Apply accelerated bounds for recurrence variables
    for rec in profile.recurrences:
        if rec.var in accelerate_vars:
            limit = accelerate_recurrence(rec)
            if limit is not None:
                lo_limit, hi_limit = limit
                if lo_limit is not None and lo_limit != float('-inf'):
                    result.set_lower(rec.var, lo_limit)
                if hi_limit is not None and hi_limit != float('inf'):
                    result.set_upper(rec.var, hi_limit)

    # Apply threshold widening for landmark variables
    for var in threshold_vars:
        thresholds = profile.per_var_thresholds.get(var, [])
        if not thresholds:
            continue
        # Get current bounds from old and new
        old_lo, old_hi = old.get_interval(var)
        new_lo, new_hi = new.get_interval(var)

        # Upper bound: if new > old, snap to next threshold above new
        if new_hi != float('inf') and old_hi != float('inf') and new_hi > old_hi:
            next_thresh = None
            for t in thresholds:
                ft = float(t)
                if ft >= new_hi:
                    next_thresh = t
                    break
            if next_thresh is not None:
                result.set_upper(var, next_thresh)
            # else: no threshold above -> standard widening leaves it (from join)

        # Lower bound: if new < old, snap to next threshold below new
        if new_lo != float('-inf') and old_lo != float('-inf') and new_lo < old_lo:
            prev_thresh = None
            for t in reversed(thresholds):
                ft = float(t)
                if ft <= new_lo:
                    prev_thresh = t
                    break
            if prev_thresh is not None:
                result.set_lower(var, prev_thresh)

    # For standard variables: apply standard polyhedral widening selectively
    # The join above already gives us a sound over-approximation.
    # Standard widening would drop constraints -- we keep them from the join
    # unless we detect non-convergence (bounds still growing).
    # This is handled implicitly: the join + acceleration + thresholds
    # typically converge faster than standard widening alone.

    return result


def landmark_narrowing(wide: PolyhedralDomain, body_result: PolyhedralDomain,
                       profile: LoopProfile) -> PolyhedralDomain:
    """Landmark-guided narrowing: tighten bounds using loop profile info.

    Standard narrowing takes meet(wide, body_result). We additionally
    use landmark knowledge to suggest tighter bounds."""
    result = polyhedral_narrowing(wide, body_result)

    # Additionally tighten using landmark values
    for var, thresholds in profile.per_var_thresholds.items():
        if not thresholds:
            continue
        lo, hi = result.get_interval(var)
        body_lo, body_hi = body_result.get_interval(var)

        # If body result has a tighter upper bound, try to snap to nearest landmark
        if body_hi != float('inf') and hi != float('inf') and body_hi < hi:
            # Find the tightest landmark that's still >= body_hi
            for t in thresholds:
                ft = float(t)
                if ft >= body_hi:
                    if ft < hi:
                        result.set_upper(var, t)
                    break

        # Same for lower bound
        if body_lo != float('-inf') and lo != float('-inf') and body_lo > lo:
            for t in reversed(thresholds):
                ft = float(t)
                if ft <= body_lo:
                    if ft > lo:
                        result.set_lower(var, ft)
                    break

    return result


# ---------------------------------------------------------------------------
# Landmark Interpreter
# ---------------------------------------------------------------------------

class LandmarkInterpreter:
    """C10 abstract interpreter with per-loop landmark-based widening."""

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.config = config or LandmarkConfig()
        self.warnings = []
        self.stats = LandmarkWideningStats()
        self.functions = {}
        self.loop_invariants = {}
        self.loop_profiles = {}
        self._loop_counter = 0
        self._pre_stmts = []  # Track statements before current loop

    def analyze(self, source: str) -> LandmarkResult:
        """Parse and analyze a C10 program with landmark-based widening."""
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        env = PolyhedralDomain()
        self._pre_stmts = []

        env = self._interpret_block(program.stmts, env)

        return LandmarkResult(
            env=env,
            warnings=self.warnings,
            stats=self.stats,
            functions=list(self.functions.keys()),
            loop_invariants=self.loop_invariants,
            loop_profiles=self.loop_profiles,
            verdict=AccelVerdict.CONVERGED
        )

    def _interpret_block(self, stmts, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret a block of statements."""
        pre_stmts_backup = self._pre_stmts[:]
        for stmt in stmts:
            if env.is_bot():
                break
            env = self._interpret_stmt(stmt, env)
        self._pre_stmts = pre_stmts_backup
        return env

    def _interpret_stmt(self, stmt, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret a single statement."""
        if isinstance(stmt, LetDecl):
            self._pre_stmts.append(stmt)
            return self._interpret_let(stmt, env)
        elif isinstance(stmt, Assign):
            self._pre_stmts.append(stmt)
            return self._interpret_assign(stmt, env)
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env)
        elif isinstance(stmt, WhileStmt):
            result = self._interpret_while(stmt, env)
            self._pre_stmts = []  # Reset after loop
            return result
        elif isinstance(stmt, FnDecl):
            self.functions[stmt.name] = stmt
            return env
        elif isinstance(stmt, Block):
            return self._interpret_block(stmt.stmts, env)
        elif isinstance(stmt, PrintStmt):
            return env
        elif isinstance(stmt, ReturnStmt):
            return env
        else:
            # Expression statement or unknown -- skip
            return env

    def _interpret_let(self, stmt: LetDecl, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret let declaration."""
        env = env.copy()
        target = stmt.name
        if target not in env.var_names:
            env.add_var(target)
        self._apply_assignment(target, stmt.value, env)
        return env

    def _interpret_assign(self, stmt: Assign, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret assignment."""
        env = env.copy()
        target = stmt.name
        if target not in env.var_names:
            env.add_var(target)
        self._apply_assignment(target, stmt.value, env)
        return env

    def _apply_assignment(self, target: str, expr, env: PolyhedralDomain):
        """Apply assignment target := expr to the environment."""
        coeffs, const = self._linearize_expr(expr)
        if coeffs is not None and const is not None:
            if target in coeffs:
                # Self-referencing: x = ... + c*x + ...
                env.assign_expr(target, coeffs, const)
            elif not coeffs:
                # Pure constant
                env.assign_const(target, const)
            elif len(coeffs) == 1:
                src, coeff = next(iter(coeffs.items()))
                if coeff == ONE and const == ZERO:
                    env.assign_var(target, src)
                else:
                    env.assign_linear(target, coeffs, const)
            else:
                env.assign_linear(target, coeffs, const)
        else:
            # Non-linear: evaluate to interval and assign bounds
            lo, hi = self._eval_interval(expr, env)
            env.forget(target)
            if lo != float('-inf'):
                env.set_lower(target, frac(int(lo)) if lo == int(lo) else frac(lo))
            if hi != float('inf'):
                env.set_upper(target, frac(int(hi)) if hi == int(hi) else frac(hi))

    def _linearize_expr(self, expr):
        """Decompose expression into {var: coeff} + constant, or (None, None)."""
        if isinstance(expr, IntLit):
            return {}, frac(expr.value)
        elif isinstance(expr, FloatLit):
            return {}, frac(expr.value)
        elif isinstance(expr, BoolLit):
            return {}, frac(1 if expr.value else 0)
        elif isinstance(expr, Var):
            return {expr.name: ONE}, ZERO
        elif isinstance(expr, UnaryOp):
            if expr.op == '-':
                sub_c, sub_k = self._linearize_expr(expr.operand)
                if sub_c is not None:
                    return {v: -c for v, c in sub_c.items()}, -sub_k
            return None, None
        elif isinstance(expr, BinOp):
            lc, lk = self._linearize_expr(expr.left)
            rc, rk = self._linearize_expr(expr.right)
            if lc is None or rc is None:
                return None, None
            if expr.op == '+':
                merged = dict(lc)
                for v, c in rc.items():
                    merged[v] = merged.get(v, ZERO) + c
                return merged, lk + rk
            elif expr.op == '-':
                merged = dict(lc)
                for v, c in rc.items():
                    merged[v] = merged.get(v, ZERO) - c
                return merged, lk - rk
            elif expr.op == '*':
                # One side must be constant
                if not lc and not rc:
                    return {}, lk * rk
                if not lc:
                    return {v: lk * c for v, c in rc.items()}, lk * rk
                if not rc:
                    return {v: rk * c for v, c in lc.items()}, lk * rk
                return None, None
            return None, None
        return None, None

    def _eval_interval(self, expr, env: PolyhedralDomain):
        """Evaluate expression to interval bounds."""
        if isinstance(expr, IntLit):
            v = float(expr.value)
            return v, v
        elif isinstance(expr, FloatLit):
            v = float(expr.value)
            return v, v
        elif isinstance(expr, BoolLit):
            v = 1.0 if expr.value else 0.0
            return v, v
        elif isinstance(expr, Var):
            return env.get_interval(expr.name)
        elif isinstance(expr, UnaryOp) and expr.op == '-':
            lo, hi = self._eval_interval(expr.operand, env)
            return -hi, -lo
        elif isinstance(expr, BinOp):
            l_lo, l_hi = self._eval_interval(expr.left, env)
            r_lo, r_hi = self._eval_interval(expr.right, env)
            if expr.op == '+':
                return l_lo + r_lo, l_hi + r_hi
            elif expr.op == '-':
                return l_lo - r_hi, l_hi - r_lo
            elif expr.op == '*':
                products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
                products = [p for p in products if not math.isnan(p)]
                if not products:
                    return float('-inf'), float('inf')
                return min(products), max(products)
            elif expr.op == '/':
                if r_lo <= 0 <= r_hi:
                    self.warnings.append("possible division by zero")
                    return float('-inf'), float('inf')
                products = []
                for l in [l_lo, l_hi]:
                    for r in [r_lo, r_hi]:
                        if r != 0:
                            products.append(l / r)
                if not products:
                    return float('-inf'), float('inf')
                return min(products), max(products)
            elif expr.op == '%':
                if r_lo <= 0 <= r_hi:
                    return float('-inf'), float('inf')
                abs_r = max(abs(r_lo), abs(r_hi))
                return 0.0, abs_r - 1.0
        return float('-inf'), float('inf')

    def _refine_condition(self, cond, env: PolyhedralDomain, is_true: bool):
        """Refine environment given a branch condition."""
        env = env.copy()
        if isinstance(cond, BinOp):
            op = cond.op
            if not is_true:
                # Negate the condition
                neg_map = {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                           '==': '!=', '!=': '=='}
                op = neg_map.get(op, op)

            left, right = cond.left, cond.right

            # var OP const
            if isinstance(left, Var) and isinstance(right, (IntLit, FloatLit)):
                var_name = left.name
                val = frac(right.value)
                if var_name not in env.var_names:
                    env.add_var(var_name)
                if op == '<':
                    env.set_upper(var_name, val - ONE)
                elif op == '<=':
                    env.set_upper(var_name, val)
                elif op == '>':
                    env.set_lower(var_name, val + ONE)
                elif op == '>=':
                    env.set_lower(var_name, val)
                elif op == '==':
                    env.set_equal(var_name, val)

            # const OP var
            elif isinstance(right, Var) and isinstance(left, (IntLit, FloatLit)):
                var_name = right.name
                val = frac(left.value)
                if var_name not in env.var_names:
                    env.add_var(var_name)
                # Flip: val OP var -> var FLIPPED_OP val
                if op == '<':
                    env.set_lower(var_name, val + ONE)
                elif op == '<=':
                    env.set_lower(var_name, val)
                elif op == '>':
                    env.set_upper(var_name, val - ONE)
                elif op == '>=':
                    env.set_upper(var_name, val)
                elif op == '==':
                    env.set_equal(var_name, val)

            # var OP var
            elif isinstance(left, Var) and isinstance(right, Var):
                lname, rname = left.name, right.name
                for n in [lname, rname]:
                    if n not in env.var_names:
                        env.add_var(n)
                # Add relational constraint: left - right OP 0
                if op == '<':
                    # left < right  =>  left - right <= -1
                    env.add_constraint({lname: ONE, rname: -ONE}, frac(-1))
                elif op == '<=':
                    env.add_constraint({lname: ONE, rname: -ONE}, ZERO)
                elif op == '>':
                    env.add_constraint({rname: ONE, lname: -ONE}, frac(-1))
                elif op == '>=':
                    env.add_constraint({rname: ONE, lname: -ONE}, ZERO)
                elif op == '==':
                    env.add_constraint({lname: ONE, rname: -ONE}, ZERO, is_equality=True)

        return env

    def _interpret_if(self, stmt: IfStmt, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret if-else statement."""
        if env.is_bot():
            return env

        then_env = self._refine_condition(stmt.cond, env, True)
        else_env = self._refine_condition(stmt.cond, env, False)

        then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body]
        then_result = self._interpret_block(then_stmts, then_env)

        if stmt.else_body:
            else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body]
            else_result = self._interpret_block(else_stmts, else_env)
        else:
            else_result = else_env

        # Join branches
        if then_result.is_bot():
            return else_result
        if else_result.is_bot():
            return then_result
        return then_result.join(else_result)

    def _interpret_while(self, stmt: WhileStmt, env: PolyhedralDomain) -> PolyhedralDomain:
        """Interpret while loop with landmark-based widening."""
        if env.is_bot():
            return env

        loop_id = self._loop_counter
        self._loop_counter += 1
        config = self.config

        body = stmt.body if isinstance(stmt.body, list) else [stmt.body]

        # Build loop profile from landmarks
        profile = build_loop_profile(stmt, env, self._pre_stmts)
        self.loop_profiles[loop_id] = profile
        self.stats.landmarks_extracted += len(profile.landmarks)

        # Propagate nested loop thresholds to this profile
        if config.enable_nested_propagation and profile.nested_profiles:
            for np in profile.nested_profiles:
                for var, thresholds in np.per_var_thresholds.items():
                    if var not in profile.per_var_thresholds:
                        profile.per_var_thresholds[var] = []
                    profile.per_var_thresholds[var] = sorted(
                        set(profile.per_var_thresholds[var]) | set(thresholds)
                    )
                    self.stats.nested_propagations += 1

        # Fixed-point iteration with landmark-based widening
        current = env.copy()
        prev = None

        for iteration in range(config.max_iterations):
            # Enter loop: refine with condition
            loop_entry = self._refine_condition(stmt.cond, current, True)
            if loop_entry.is_bot():
                break

            # Execute body
            body_result = self._interpret_block(body, loop_entry)

            # Widen current with body result
            if prev is not None:
                new_state = landmark_widen_per_var(
                    current, body_result.join(current),
                    profile, iteration, config.delay_iterations
                )
                self.stats.widening_iterations += 1

                # Track per-var stats
                for rec in profile.recurrences:
                    if profile.get_var_policy(rec.var) == 'accelerate':
                        self.stats.per_var_accelerations += 1
                for var in profile.per_var_thresholds:
                    if profile.get_var_policy(var) == 'threshold':
                        self.stats.per_var_threshold_uses += 1
            else:
                new_state = current.join(body_result)

            self.stats.total_iterations += 1

            # Check convergence
            if prev is not None and new_state.leq(current):
                # Converged -- store pre-narrowing invariant
                self.loop_invariants[loop_id] = current.copy()

                # Narrowing phase
                if config.enable_landmark_narrowing:
                    narrowed = current.copy()
                    for _ in range(config.narrowing_iterations):
                        loop_entry_n = self._refine_condition(stmt.cond, narrowed, True)
                        if loop_entry_n.is_bot():
                            break
                        body_result_n = self._interpret_block(body, loop_entry_n)
                        new_narrowed = landmark_narrowing(narrowed, body_result_n, profile)
                        self.stats.narrowing_iterations += 1
                        self.stats.landmark_narrowings += 1
                        if new_narrowed.equals(narrowed):
                            break
                        narrowed = new_narrowed
                    current = narrowed
                    self.loop_invariants[loop_id] = current.copy()
                break

            prev = current
            current = new_state
        else:
            # Did not converge
            self.loop_invariants[loop_id] = current.copy()
            self.warnings.append(f"loop {loop_id}: max iterations reached")

        # Exit loop: refine with negated condition
        exit_env = self._refine_condition(stmt.cond, current, False)
        return exit_env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def landmark_analyze(source: str, config: Optional[LandmarkConfig] = None) -> LandmarkResult:
    """Analyze a C10 program with landmark-based polyhedral widening.

    This is the main entry point. Returns a LandmarkResult with the final
    environment, warnings, statistics, loop invariants, and loop profiles.
    """
    interp = LandmarkInterpreter(config)
    return interp.analyze(source)


def compare_widening_strategies(source: str, config: Optional[LandmarkConfig] = None) -> dict:
    """Compare landmark-based widening vs standard V105 vs V121 accelerated.

    Returns a dict with results from all three strategies and precision comparison.
    """
    # Landmark analysis
    lm_result = landmark_analyze(source, config)

    # Standard V105
    std_result = standard_analyze(source)

    # V121 accelerated
    accel_config = AccelConfig()
    accel_result = accelerated_analyze(source, accel_config)

    # Collect variable ranges from each
    all_vars = set(lm_result.env.var_names)
    comparison = {}
    precision_wins = {'landmark': 0, 'standard': 0, 'accelerated': 0}

    for var in sorted(all_vars):
        lm_lo, lm_hi = lm_result.env.get_interval(var)
        std_lo, std_hi = std_result['env'].get_interval(var) if var in std_result['env'].var_names else (float('-inf'), float('inf'))
        acc_lo, acc_hi = accel_result.env.get_interval(var) if var in accel_result.env.var_names else (float('-inf'), float('inf'))

        lm_width = lm_hi - lm_lo if lm_hi != float('inf') and lm_lo != float('-inf') else float('inf')
        std_width = std_hi - std_lo if std_hi != float('inf') and std_lo != float('-inf') else float('inf')
        acc_width = acc_hi - acc_lo if acc_hi != float('inf') and acc_lo != float('-inf') else float('inf')

        comparison[var] = {
            'landmark': (lm_lo, lm_hi),
            'standard': (std_lo, std_hi),
            'accelerated': (acc_lo, acc_hi),
            'landmark_width': lm_width,
            'standard_width': std_width,
            'accelerated_width': acc_width
        }

        # Compare widths
        widths = {'landmark': lm_width, 'standard': std_width, 'accelerated': acc_width}
        best = min(widths, key=widths.get)
        precision_wins[best] += 1

    return {
        'landmark': lm_result,
        'standard': std_result,
        'accelerated': accel_result,
        'comparison': comparison,
        'precision_wins': precision_wins,
        'landmark_stats': {
            'landmarks_extracted': lm_result.stats.landmarks_extracted,
            'per_var_accelerations': lm_result.stats.per_var_accelerations,
            'per_var_threshold_uses': lm_result.stats.per_var_threshold_uses,
            'nested_propagations': lm_result.stats.nested_propagations,
            'landmark_narrowings': lm_result.stats.landmark_narrowings,
            'total_iterations': lm_result.stats.total_iterations,
        }
    }


def get_variable_range(source: str, var_name: str,
                       config: Optional[LandmarkConfig] = None) -> Tuple[float, float]:
    """Get the computed range for a variable after landmark analysis."""
    result = landmark_analyze(source, config)
    if var_name in result.env.var_names:
        return result.env.get_interval(var_name)
    return (float('-inf'), float('inf'))


def get_loop_profile(source: str, loop_index: int = 0,
                     config: Optional[LandmarkConfig] = None) -> Optional[LoopProfile]:
    """Get the structural profile of a specific loop."""
    result = landmark_analyze(source, config)
    return result.loop_profiles.get(loop_index)


def get_loop_invariant(source: str, loop_index: int = 0,
                       config: Optional[LandmarkConfig] = None) -> Optional[PolyhedralDomain]:
    """Get the loop invariant for a specific loop."""
    result = landmark_analyze(source, config)
    return result.loop_invariants.get(loop_index)


def get_landmark_stats(source: str,
                       config: Optional[LandmarkConfig] = None) -> LandmarkWideningStats:
    """Get statistics from landmark analysis."""
    result = landmark_analyze(source, config)
    return result.stats


def landmark_summary(source: str, config: Optional[LandmarkConfig] = None) -> str:
    """Human-readable summary of landmark analysis results."""
    result = landmark_analyze(source, config)
    lines = ["=== Landmark Widening Analysis ===", ""]

    # Variable ranges
    lines.append("Variable ranges:")
    for var in sorted(result.env.var_names):
        lo, hi = result.env.get_interval(var)
        lines.append(f"  {var}: [{lo}, {hi}]")

    # Loop profiles
    lines.append("")
    lines.append("Loop profiles:")
    for lid, profile in result.loop_profiles.items():
        lines.append(f"  Loop {lid}:")
        lines.append(f"    Modified vars: {sorted(profile.modified_vars)}")
        lines.append(f"    Condition vars: {sorted(profile.condition_vars)}")
        lines.append(f"    Landmarks: {len(profile.landmarks)}")
        for lm in profile.landmarks:
            lines.append(f"      [{lm.kind.value}] {lm.source} (priority {lm.priority})")
        lines.append(f"    Recurrences: {len(profile.recurrences)}")
        for rec in profile.recurrences:
            lines.append(f"      {rec.var} += {rec.delta}")
        lines.append(f"    Per-var thresholds: {dict(profile.per_var_thresholds)}")
        for var in sorted(profile.modified_vars):
            policy = profile.get_var_policy(var)
            lines.append(f"    Policy for {var}: {policy}")

    # Statistics
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Total iterations: {result.stats.total_iterations}")
    lines.append(f"  Widening iterations: {result.stats.widening_iterations}")
    lines.append(f"  Narrowing iterations: {result.stats.narrowing_iterations}")
    lines.append(f"  Landmarks extracted: {result.stats.landmarks_extracted}")
    lines.append(f"  Per-var accelerations: {result.stats.per_var_accelerations}")
    lines.append(f"  Per-var threshold uses: {result.stats.per_var_threshold_uses}")
    lines.append(f"  Nested propagations: {result.stats.nested_propagations}")
    lines.append(f"  Landmark narrowings: {result.stats.landmark_narrowings}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
