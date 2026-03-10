"""
V027: Quantitative Abstract Interpretation -- Resource Bound Analysis

Composes:
- C039 (abstract interpreter) for variable range analysis
- V025 (termination analysis) for loop extraction and ranking functions
- V019 (widening thresholds) for precise interval bounds
- C010 (parser) for AST traversal
- C037 (SMT solver) for bound verification

Given a program, computes:
- Loop iteration bounds (upper bound on iteration count per loop)
- Nested loop complexity (product of inner/outer bounds)
- Operation counts (assignments, comparisons, calls per execution)
- Complexity classification (O(1), O(n), O(n^2), etc.)
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_work, 'V025_termination_analysis'))
sys.path.insert(0, os.path.join(_work, 'V019_widening_thresholds'))

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

# C010
from stack_vm import (
    lex, Parser, Program, Block,
    IntLit, BoolLit, Var as ASTVar, BinOp, UnaryOp,
    Assign, LetDecl, IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt,
    PrintStmt,
)

# C037
from smt_solver import (
    SMTSolver, SMTResult, Op, BOOL, INT,
    Var as SMTVar, IntConst, BoolConst, App,
)

# C039
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, AbstractValue,
    Interval, INTERVAL_TOP, INTERVAL_BOT,
    Sign, INF, NEG_INF,
    analyze as ai_analyze,
)

# V025
from termination import (
    extract_loop_info, find_ranking_function, generate_candidates,
    verify_ranking_function, RankingFunction, TermResult,
)

# V019
from widening_thresholds import threshold_analyze


# ---------- Data types ----------

class ComplexityClass(Enum):
    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n^2)"
    O_N3 = "O(n^3)"
    O_POLY = "O(n^k)"
    O_UNKNOWN = "O(?)"


@dataclass
class BoundExpr:
    """Represents a symbolic bound expression."""
    kind: str  # "constant", "linear", "product", "symbolic"
    value: object = None  # int for constant, dict for linear, etc.
    text: str = ""

    def __str__(self):
        return self.text

    @staticmethod
    def constant(n):
        return BoundExpr("constant", n, str(n))

    @staticmethod
    def linear(coeffs, params):
        """coeffs: {param: coeff, '_const': c}. bound = sum(coeff*param) + c."""
        parts = []
        c = coeffs.get('_const', 0)
        for p in sorted(params):
            co = coeffs.get(p, 0)
            if co == 0:
                continue
            if co == 1:
                parts.append(p)
            elif co == -1:
                parts.append(f"-{p}")
            else:
                parts.append(f"{co}*{p}")
        if c != 0 or not parts:
            parts.append(str(c))
        text = " + ".join(parts).replace("+ -", "- ")
        return BoundExpr("linear", coeffs, text)

    @staticmethod
    def product(a, b):
        ta = str(a) if a.kind != "product" else f"({a})"
        tb = str(b) if b.kind != "product" else f"({b})"
        return BoundExpr("product", (a, b), f"{ta} * {tb}")

    @staticmethod
    def symbolic(text):
        return BoundExpr("symbolic", None, text)

    def evaluate(self, env):
        """Evaluate bound with concrete variable values."""
        if self.kind == "constant":
            return self.value
        elif self.kind == "linear":
            total = self.value.get('_const', 0)
            for var, coeff in self.value.items():
                if var == '_const':
                    continue
                if var in env:
                    total += coeff * env[var]
                else:
                    return None
            return max(0, total)
        elif self.kind == "product":
            a, b = self.value
            va = a.evaluate(env)
            vb = b.evaluate(env)
            if va is not None and vb is not None:
                return va * vb
            return None
        return None

    def get_params(self):
        """Get symbolic parameters in this bound."""
        if self.kind == "constant":
            return set()
        elif self.kind == "linear":
            return {k for k in self.value if k != '_const'}
        elif self.kind == "product":
            a, b = self.value
            return a.get_params() | b.get_params()
        return set()


@dataclass
class LoopBoundResult:
    """Result of analyzing a single loop's iteration bound."""
    loop_index: int
    bound: Optional[BoundExpr]
    ranking_function: Optional[RankingFunction]
    initial_value: Optional[int]  # Concrete initial value of ranking fn (if known)
    initial_interval: Optional[Interval]  # Interval of initial ranking fn value
    is_tight: bool  # Whether the bound is exact (not just upper bound)
    message: str = ""
    nesting_depth: int = 0


@dataclass
class ResourceCount:
    """Counts of operations in a program."""
    assignments: int = 0
    comparisons: int = 0
    arithmetic_ops: int = 0
    function_calls: int = 0
    loop_iterations: Optional[BoundExpr] = None  # Total across all loops
    per_loop: Dict[int, BoundExpr] = field(default_factory=dict)


@dataclass
class BoundResult:
    """Full result of quantitative analysis."""
    loop_bounds: List[LoopBoundResult]
    total_bound: Optional[BoundExpr]
    complexity: ComplexityClass
    complexity_text: str
    resource_counts: ResourceCount
    params: set  # Symbolic parameters (uninitialized variables)
    message: str = ""


# ---------- AST utilities ----------

def parse(source):
    tokens = lex(source)
    return Parser(tokens).parse()


def find_all_loops(node, depth=0):
    """Find all while loops in AST with nesting depth."""
    results = []
    if isinstance(node, Program):
        for s in node.stmts:
            results.extend(find_all_loops(s, depth))
    elif isinstance(node, Block):
        for s in node.stmts:
            results.extend(find_all_loops(s, depth))
    elif isinstance(node, WhileStmt):
        results.append((node, depth))
        # Recurse into body for nested loops
        results.extend(find_all_loops(node.body, depth + 1))
    elif isinstance(node, IfStmt):
        results.extend(find_all_loops(node.then_body, depth))
        if node.else_body:
            results.extend(find_all_loops(node.else_body, depth))
    elif isinstance(node, FnDecl):
        for s in node.body:
            results.extend(find_all_loops(s, depth))
    return results


def count_operations(node):
    """Count operations in an AST node."""
    counts = ResourceCount()
    _count_ops_rec(node, counts)
    return counts


def _count_ops_rec(node, counts):
    if node is None:
        return
    if isinstance(node, Program):
        for s in node.stmts:
            _count_ops_rec(s, counts)
    elif isinstance(node, Block):
        for s in node.stmts:
            _count_ops_rec(s, counts)
    elif isinstance(node, (LetDecl, Assign)):
        counts.assignments += 1
        _count_ops_rec(node.value, counts)
    elif isinstance(node, BinOp):
        if node.op in ('<', '>', '<=', '>=', '==', '!='):
            counts.comparisons += 1
        else:
            counts.arithmetic_ops += 1
        _count_ops_rec(node.left, counts)
        _count_ops_rec(node.right, counts)
    elif isinstance(node, UnaryOp):
        counts.arithmetic_ops += 1
        _count_ops_rec(node.operand, counts)
    elif isinstance(node, CallExpr):
        counts.function_calls += 1
        for a in node.args:
            _count_ops_rec(a, counts)
    elif isinstance(node, IfStmt):
        counts.comparisons += 1
        _count_ops_rec(node.cond, counts)
        _count_ops_rec(node.then_body, counts)
        if node.else_body:
            _count_ops_rec(node.else_body, counts)
    elif isinstance(node, WhileStmt):
        # The condition comparison is counted per iteration (handled in loop bound)
        _count_ops_rec(node.cond, counts)
        _count_ops_rec(node.body, counts)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _count_ops_rec(node.value, counts)
    elif isinstance(node, PrintStmt):
        _count_ops_rec(node.value, counts)
    elif isinstance(node, FnDecl):
        body = node.body
        if isinstance(body, Block):
            body = body.stmts
        for s in body:
            _count_ops_rec(s, counts)


def collect_pre_assignments(stmts, target_loop_index):
    """Collect LetDecl assignments before the target loop."""
    assignments = {}
    loop_count = 0
    for s in stmts:
        if isinstance(s, WhileStmt):
            if loop_count == target_loop_index:
                return assignments
            loop_count += 1
        elif isinstance(s, LetDecl):
            if isinstance(s.value, IntLit):
                assignments[s.name] = s.value.value
            elif isinstance(s.value, BoolLit):
                assignments[s.name] = 1 if s.value.value else 0
            else:
                assignments[s.name] = None  # Non-constant init
    return assignments


def find_symbolic_params(source):
    """Find variables used before any LetDecl assigns them (symbolic parameters).

    A variable like `n` in `while (n > 0) { n = n - 1; }` is a parameter
    because it's used (in condition) before any LetDecl initializes it.
    Assign (n = n - 1) is a reassignment, not a declaration.
    """
    prog = parse(source)
    let_declared = set()
    used_before_decl = set()
    _collect_params(prog, let_declared, used_before_decl)
    return used_before_decl


def _collect_params(node, let_declared, used_before_decl):
    """Track variables used before LetDecl."""
    if node is None:
        return
    if isinstance(node, Program):
        for s in node.stmts:
            _collect_params(s, let_declared, used_before_decl)
    elif isinstance(node, Block):
        for s in node.stmts:
            _collect_params(s, let_declared, used_before_decl)
    elif isinstance(node, LetDecl):
        # RHS is evaluated before the name is declared
        _collect_params(node.value, let_declared, used_before_decl)
        let_declared.add(node.name)
    elif isinstance(node, Assign):
        # RHS uses are recorded, but Assign doesn't introduce a new variable
        _collect_params(node.value, let_declared, used_before_decl)
    elif isinstance(node, ASTVar):
        if node.name not in let_declared:
            used_before_decl.add(node.name)
    elif isinstance(node, BinOp):
        _collect_params(node.left, let_declared, used_before_decl)
        _collect_params(node.right, let_declared, used_before_decl)
    elif isinstance(node, UnaryOp):
        _collect_params(node.operand, let_declared, used_before_decl)
    elif isinstance(node, IfStmt):
        _collect_params(node.cond, let_declared, used_before_decl)
        _collect_params(node.then_body, let_declared, used_before_decl)
        if node.else_body:
            _collect_params(node.else_body, let_declared, used_before_decl)
    elif isinstance(node, WhileStmt):
        _collect_params(node.cond, let_declared, used_before_decl)
        _collect_params(node.body, let_declared, used_before_decl)
    elif isinstance(node, CallExpr):
        for a in node.args:
            _collect_params(a, let_declared, used_before_decl)
    elif isinstance(node, FnDecl):
        # Function params are declared
        fn_declared = let_declared | set(node.params)
        body = node.body
        if isinstance(body, Block):
            body = body.stmts
        for s in body:
            _collect_params(s, fn_declared, used_before_decl)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _collect_params(node.value, let_declared, used_before_decl)
    elif isinstance(node, PrintStmt):
        _collect_params(node.value, let_declared, used_before_decl)


# ---------- Bound computation ----------

def compute_ranking_initial_value(ranking, pre_assignments, ai_result=None):
    """Compute the initial value of a ranking function from pre-loop state.

    Returns (concrete_value_or_None, interval).
    """
    coeffs = ranking.coefficients
    # Try concrete evaluation
    concrete = coeffs.get('_const', 0)
    all_concrete = True
    for var, coeff in coeffs.items():
        if var == '_const':
            continue
        if var in pre_assignments and pre_assignments[var] is not None:
            concrete += coeff * pre_assignments[var]
        else:
            all_concrete = False

    if all_concrete:
        return concrete, Interval(concrete, concrete)

    # Use abstract interpretation for interval bound
    if ai_result:
        env = ai_result.get('env') if isinstance(ai_result, dict) else ai_result
        lo_total = coeffs.get('_const', 0)
        hi_total = coeffs.get('_const', 0)
        all_bounded = True
        for var, coeff in coeffs.items():
            if var == '_const':
                continue
            # Try pre-assignments first
            if var in pre_assignments and pre_assignments[var] is not None:
                v = pre_assignments[var]
                if coeff > 0:
                    lo_total += coeff * v
                    hi_total += coeff * v
                else:
                    lo_total += coeff * v
                    hi_total += coeff * v
            elif hasattr(env, 'get_interval'):
                iv = env.get_interval(var)
                if iv.is_bot() or iv.is_top():
                    all_bounded = False
                    break
                if coeff > 0:
                    lo_total += coeff * iv.lo
                    hi_total += coeff * iv.hi
                else:
                    lo_total += coeff * iv.hi
                    hi_total += coeff * iv.lo
            else:
                all_bounded = False
                break

        if all_bounded:
            return None, Interval(lo_total, hi_total)

    return None, INTERVAL_TOP


def ranking_to_bound(ranking, pre_assignments, ai_result=None, params=None):
    """Convert a ranking function to a bound expression.

    The bound is the initial value of the ranking function (upper bound on iterations).
    """
    if params is None:
        params = set()

    coeffs = ranking.coefficients
    concrete_val, interval = compute_ranking_initial_value(
        ranking, pre_assignments, ai_result
    )

    # If fully concrete, return constant bound
    if concrete_val is not None:
        return BoundExpr.constant(max(0, concrete_val)), concrete_val, interval, True

    # Check if bound is linear in parameters
    has_params = False
    bound_coeffs = {}
    const_part = coeffs.get('_const', 0)

    for var, coeff in coeffs.items():
        if var == '_const':
            continue
        if var in pre_assignments and pre_assignments[var] is not None:
            const_part += coeff * pre_assignments[var]
        elif var in params:
            bound_coeffs[var] = coeff
            has_params = True
        else:
            # Unknown variable with known pre-assignment
            bound_coeffs[var] = coeff
            has_params = True

    if has_params:
        bound_coeffs['_const'] = const_part
        param_names = {k for k in bound_coeffs if k != '_const'}
        bound = BoundExpr.linear(bound_coeffs, param_names)
        # Use interval upper bound if available
        init_val = int(interval.hi) if not interval.is_top() and interval.hi != INF else None
        return bound, init_val, interval, False
    else:
        # All concrete
        return BoundExpr.constant(max(0, int(const_part))), int(const_part), Interval(const_part, const_part), True


def analyze_loop_bound(source, loop_index=0):
    """Analyze the iteration bound of a single loop.

    Returns LoopBoundResult with the computed bound.
    """
    # Find ranking function via V025
    ranking = find_ranking_function(source, loop_index)

    if ranking is None:
        # Try harder: use threshold analysis for tighter intervals
        return LoopBoundResult(
            loop_index=loop_index,
            bound=None,
            ranking_function=None,
            initial_value=None,
            initial_interval=None,
            is_tight=False,
            message="Could not find ranking function"
        )

    # Get pre-loop variable values
    prog = parse(source)
    pre = collect_pre_assignments(prog.stmts, loop_index)

    # Run abstract interpretation for interval bounds
    ai_result = None
    try:
        ai_result = threshold_analyze(source)
    except Exception:
        try:
            ai_result = ai_analyze(source)
        except Exception:
            pass

    # Find symbolic parameters
    params = find_symbolic_params(source)

    # Compute bound from ranking function
    bound, init_val, interval, is_tight = ranking_to_bound(
        ranking, pre, ai_result, params
    )

    # Find nesting depth
    loops = find_all_loops(prog)
    depth = 0
    if loop_index < len(loops):
        depth = loops[loop_index][1]

    return LoopBoundResult(
        loop_index=loop_index,
        bound=bound,
        ranking_function=ranking,
        initial_value=init_val,
        initial_interval=interval,
        is_tight=is_tight,
        nesting_depth=depth,
        message=f"Bound: {bound} (ranking: {ranking.expression})"
    )


# ---------- Complexity classification ----------

def classify_complexity(loop_bounds, nesting_info):
    """Classify the overall complexity from loop bounds and nesting.

    nesting_info: list of (loop_index, nesting_depth, bound)
    """
    if not loop_bounds or all(lb.bound is None for lb in loop_bounds):
        # No loops or no bounds found
        return ComplexityClass.O_1, "O(1)"

    # Group loops by nesting depth
    top_level = []
    nested_groups = {}  # parent_depth -> list of bounds

    for lb in loop_bounds:
        if lb.bound is None:
            return ComplexityClass.O_UNKNOWN, "O(?)"

        if lb.nesting_depth == 0:
            top_level.append(lb)
        else:
            # Nested loops contribute multiplicatively
            parent_depth = lb.nesting_depth - 1
            if parent_depth not in nested_groups:
                nested_groups[parent_depth] = []
            nested_groups[parent_depth].append(lb)

    # Compute total complexity
    max_param_count = 0
    has_params = False

    for lb in loop_bounds:
        p = lb.bound.get_params()
        if p:
            has_params = True
            max_param_count = max(max_param_count, len(p))

    if not has_params:
        # All bounds are constants
        return ComplexityClass.O_1, "O(1)"

    # Count nesting depth of parametric loops
    max_nesting = 0
    for lb in loop_bounds:
        if lb.bound.get_params():
            max_nesting = max(max_nesting, lb.nesting_depth + 1)

    # Check for nested parametric loops (quadratic, cubic, etc.)
    if max_nesting >= 3:
        return ComplexityClass.O_N3, "O(n^3)"
    elif max_nesting >= 2:
        return ComplexityClass.O_N2, "O(n^2)"

    # Check for multiple independent parametric loops at same level
    param_top_level = [lb for lb in top_level if lb.bound.get_params()]
    if len(param_top_level) > 1:
        # Multiple independent O(n) loops are still O(n)
        all_params = set()
        for lb in param_top_level:
            all_params |= lb.bound.get_params()
        if len(all_params) > 1:
            param_str = ", ".join(sorted(all_params))
            return ComplexityClass.O_N, f"O({param_str})"

    # Single parametric loop
    params = set()
    for lb in loop_bounds:
        params |= lb.bound.get_params()

    if len(params) == 1:
        p = next(iter(params))
        return ComplexityClass.O_N, f"O({p})"
    elif params:
        param_str = " + ".join(sorted(params))
        return ComplexityClass.O_N, f"O({param_str})"

    return ComplexityClass.O_UNKNOWN, "O(?)"


def compute_total_bound(loop_bounds):
    """Compute total iteration bound across all loops, respecting nesting."""
    if not loop_bounds:
        return BoundExpr.constant(0)

    valid = [lb for lb in loop_bounds if lb.bound is not None]
    if not valid:
        return None

    # Group by nesting: nested loops multiply, top-level loops add
    # Simple approach: multiply nested, sum top-level
    by_depth = {}
    for lb in valid:
        d = lb.nesting_depth
        if d not in by_depth:
            by_depth[d] = []
        by_depth[d].append(lb)

    # Build from deepest nesting outward
    max_depth = max(by_depth.keys()) if by_depth else 0

    if max_depth == 0:
        # No nesting: sum all bounds
        if len(valid) == 1:
            return valid[0].bound

        # Sum constants
        total_const = 0
        symbolic_parts = []
        for lb in valid:
            if lb.bound.kind == "constant":
                total_const += lb.bound.value
            else:
                symbolic_parts.append(lb.bound)

        if not symbolic_parts:
            return BoundExpr.constant(total_const)
        # Return the largest symbolic bound (simplified)
        return symbolic_parts[0] if len(symbolic_parts) == 1 else symbolic_parts[0]

    # With nesting: multiply outer * inner
    # Find the deepest chain
    result = None
    for depth in range(max_depth, -1, -1):
        if depth in by_depth:
            for lb in by_depth[depth]:
                if result is None:
                    result = lb.bound
                else:
                    result = BoundExpr.product(lb.bound, result)
    return result


# ---------- Main API ----------

def analyze_bounds(source):
    """Analyze resource bounds for a program.

    Returns BoundResult with loop bounds, complexity, and resource counts.
    """
    prog = parse(source)
    loops = find_all_loops(prog)
    params = find_symbolic_params(source)

    # Analyze each loop
    loop_bounds = []
    for i, (loop_node, depth) in enumerate(loops):
        try:
            lb = analyze_loop_bound(source, i)
            lb.nesting_depth = depth
            loop_bounds.append(lb)
        except Exception as e:
            loop_bounds.append(LoopBoundResult(
                loop_index=i, bound=None, ranking_function=None,
                initial_value=None, initial_interval=None,
                is_tight=False, nesting_depth=depth,
                message=f"Error: {e}"
            ))

    # Compute total bound
    total_bound = compute_total_bound(loop_bounds)

    # Classify complexity
    nesting_info = [(lb.loop_index, lb.nesting_depth, lb.bound) for lb in loop_bounds]
    complexity, complexity_text = classify_complexity(loop_bounds, nesting_info)

    # Count operations
    resource_counts = count_operations(prog)
    resource_counts.per_loop = {
        lb.loop_index: lb.bound for lb in loop_bounds if lb.bound
    }
    resource_counts.loop_iterations = total_bound

    return BoundResult(
        loop_bounds=loop_bounds,
        total_bound=total_bound,
        complexity=complexity,
        complexity_text=complexity_text,
        resource_counts=resource_counts,
        params=params,
        message=f"Complexity: {complexity_text}"
    )


def loop_bound(source, loop_index=0):
    """Analyze the iteration bound of a specific loop.

    Convenience wrapper around analyze_loop_bound.
    """
    return analyze_loop_bound(source, loop_index)


def complexity_class(source):
    """Get the complexity classification for a program.

    Returns (ComplexityClass, str).
    """
    result = analyze_bounds(source)
    return result.complexity, result.complexity_text


def resource_count(source):
    """Count operations in a program.

    Returns ResourceCount.
    """
    prog = parse(source)
    return count_operations(prog)


def verify_bound(source, loop_index, proposed_bound, params_env=None):
    """Verify that a proposed bound is correct for a loop.

    proposed_bound: int (concrete) or dict {param: coeff, '_const': c}
    params_env: dict of parameter values for concrete verification

    Returns (valid, message).
    """
    lb = analyze_loop_bound(source, loop_index)

    if lb.bound is None:
        return False, "Could not determine loop bound"

    if isinstance(proposed_bound, int):
        # Verify concrete bound
        if lb.bound.kind == "constant":
            if lb.bound.value <= proposed_bound:
                return True, f"Verified: actual bound {lb.bound.value} <= proposed {proposed_bound}"
            else:
                return False, f"Bound too tight: actual {lb.bound.value} > proposed {proposed_bound}"

        # Parametric bound -- need env
        if params_env:
            actual = lb.bound.evaluate(params_env)
            if actual is not None and actual <= proposed_bound:
                return True, f"Verified for given params: actual {actual} <= proposed {proposed_bound}"
            elif actual is not None:
                return False, f"Bound too tight for given params: actual {actual} > proposed {proposed_bound}"

        return False, "Cannot verify concrete bound against parametric actual bound"

    elif isinstance(proposed_bound, dict):
        # Verify parametric bound via SMT
        # The proposed bound should be >= the ranking function's initial value
        ranking = lb.ranking_function
        if ranking is None:
            return False, "No ranking function found"

        prog = parse(source)
        pre = collect_pre_assignments(prog.stmts, loop_index)

        # Build SMT check: forall params, proposed >= ranking(init)
        s = SMTSolver()
        params = find_symbolic_params(source)
        var_map = {}
        for p in params:
            var_map[p] = s.Int(p)

        # Build ranking(init) expression
        rank_init = IntConst(ranking.coefficients.get('_const', 0))
        for var, coeff in ranking.coefficients.items():
            if var == '_const':
                continue
            if var in pre and pre[var] is not None:
                term = IntConst(coeff * pre[var])
            elif var in var_map:
                if coeff == 1:
                    term = var_map[var]
                else:
                    term = App(Op.MUL, [IntConst(coeff), var_map[var]], INT)
            else:
                return False, f"Variable {var} not found in params or pre-assignments"
            rank_init = App(Op.ADD, [rank_init, term], INT)

        # Build proposed expression
        proposed_expr = IntConst(proposed_bound.get('_const', 0))
        for var, coeff in proposed_bound.items():
            if var == '_const':
                continue
            if var not in var_map:
                var_map[var] = s.Int(var)
            if coeff == 1:
                term = var_map[var]
            else:
                term = App(Op.MUL, [IntConst(coeff), var_map[var]], INT)
            proposed_expr = App(Op.ADD, [proposed_expr, term], INT)

        # Check: NOT(proposed >= rank_init) is UNSAT?
        # i.e., rank_init > proposed is UNSAT
        s.add(App(Op.GT, [rank_init, proposed_expr], BOOL))
        result = s.check()

        if result == SMTResult.UNSAT:
            return True, "Verified: proposed bound >= ranking function initial value"
        elif result == SMTResult.SAT:
            model = s.model()
            return False, f"Counterexample: {model}"
        else:
            return False, "SMT returned UNKNOWN"

    return False, "Invalid proposed_bound type"


def compare_bounds(source1, source2, params=None):
    """Compare resource bounds of two programs.

    Returns dict with comparison results.
    """
    r1 = analyze_bounds(source1)
    r2 = analyze_bounds(source2)

    result = {
        'program1': {
            'complexity': r1.complexity_text,
            'loop_count': len(r1.loop_bounds),
            'bounds': [str(lb.bound) for lb in r1.loop_bounds if lb.bound],
        },
        'program2': {
            'complexity': r2.complexity_text,
            'loop_count': len(r2.loop_bounds),
            'bounds': [str(lb.bound) for lb in r2.loop_bounds if lb.bound],
        },
        'same_complexity': r1.complexity == r2.complexity,
        'complexity_comparison': _compare_classes(r1.complexity, r2.complexity),
    }

    # Concrete comparison with params
    if params:
        for key, r in [('program1', r1), ('program2', r2)]:
            if r.total_bound:
                val = r.total_bound.evaluate(params)
                result[key]['concrete_bound'] = val

    return result


def _compare_classes(c1, c2):
    order = [
        ComplexityClass.O_1, ComplexityClass.O_LOG_N, ComplexityClass.O_N,
        ComplexityClass.O_N_LOG_N, ComplexityClass.O_N2, ComplexityClass.O_N3,
        ComplexityClass.O_POLY, ComplexityClass.O_UNKNOWN,
    ]
    i1 = order.index(c1) if c1 in order else len(order)
    i2 = order.index(c2) if c2 in order else len(order)
    if i1 < i2:
        return "program1 is faster"
    elif i1 > i2:
        return "program2 is faster"
    else:
        return "same complexity class"


def bound_summary(source):
    """Get a concise human-readable summary of program bounds."""
    result = analyze_bounds(source)
    lines = [f"Complexity: {result.complexity_text}"]
    for lb in result.loop_bounds:
        if lb.bound:
            tight = " (tight)" if lb.is_tight else ""
            depth = f" [depth={lb.nesting_depth}]" if lb.nesting_depth > 0 else ""
            lines.append(f"  Loop {lb.loop_index}: <= {lb.bound} iterations{tight}{depth}")
        else:
            lines.append(f"  Loop {lb.loop_index}: bound unknown")
    if result.total_bound:
        lines.append(f"Total iterations: <= {result.total_bound}")
    return "\n".join(lines)
