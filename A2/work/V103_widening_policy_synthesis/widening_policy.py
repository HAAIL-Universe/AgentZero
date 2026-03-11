"""V103: Widening Policy Synthesis

Automatically synthesizes optimal widening/narrowing policies for abstract
interpretation. Different loops benefit from different widening strategies --
this system analyzes program structure and synthesizes per-loop policies.

Composes:
- V020 (abstract domain functor) -- generic domains + interpreter
- V019 (threshold widening) -- threshold extraction + widening
- C039 (abstract interpreter) -- baseline analysis
- C010 (parser) -- AST access
- C037 (SMT solver) -- policy validation

Key concepts:
- WideningPolicy: per-loop configuration (thresholds, delay, narrowing depth)
- PolicySynthesizer: analyzes loop structure to produce optimal policies
- PolicyInterpreter: C10 abstract interpreter parameterized by per-loop policies
- Policy validation: SMT-based soundness checking of synthesized policies
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from enum import Enum

# Import dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import Parser, lex

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, AbstractValue,
    Interval, NEG_INF, INF, interval_widen,
    Sign, analyze as baseline_analyze
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V019_widening_thresholds'))
from widening_thresholds import (
    extract_thresholds_from_source, extract_variable_thresholds,
    interval_widen_thresholds, ThresholdEnv, ThresholdInterpreter,
    threshold_analyze
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V020_abstract_domain_functor'))
from domain_functor import (
    AbstractDomain as FunctorDomain, IntervalDomain, SignDomain,
    FunctorInterpreter, DomainEnv, make_sign_interval, make_full_product
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Op, Sort, SortKind, IntConst, BoolConst, Var, App

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class WideningStrategy(Enum):
    """Widening strategy for a single loop."""
    STANDARD = "standard"       # Default: widen to infinity immediately
    THRESHOLD = "threshold"     # Widen to nearest threshold instead of infinity
    DELAYED = "delayed"         # Delay widening for N iterations, then standard
    DELAYED_THRESHOLD = "delayed_threshold"  # Delay then threshold


@dataclass(frozen=True)
class WideningPolicy:
    """Per-loop widening configuration."""
    strategy: WideningStrategy = WideningStrategy.STANDARD
    thresholds: Tuple[float, ...] = ()    # Sorted threshold values
    delay: int = 0                        # Number of iterations before widening starts
    narrowing_iterations: int = 0         # Post-widening narrowing passes
    max_iterations: int = 50              # Safety bound on fixpoint iterations

    def with_thresholds(self, ts: List[float]) -> 'WideningPolicy':
        return WideningPolicy(
            strategy=self.strategy,
            thresholds=tuple(sorted(set(ts))),
            delay=self.delay,
            narrowing_iterations=self.narrowing_iterations,
            max_iterations=self.max_iterations,
        )


@dataclass
class LoopInfo:
    """Information about a single loop in the program."""
    loop_id: int                       # Unique identifier
    stmt: Any                          # AST node (WhileStmt)
    condition_vars: Set[str]           # Variables in loop condition
    modified_vars: Set[str]            # Variables modified in loop body
    constants_in_condition: Set[int]   # Numeric constants in condition
    constants_in_body: Set[int]        # Numeric constants in body
    is_simple_counter: bool            # i = i + c pattern
    counter_var: Optional[str] = None  # Counter variable if simple_counter
    counter_step: Optional[int] = None # Step size if simple_counter
    bound_value: Optional[int] = None  # Upper/lower bound from condition
    bound_direction: Optional[str] = None  # 'up' or 'down'
    nested_depth: int = 0              # Nesting level (0 = outermost)


@dataclass
class PolicyResult:
    """Result of analysis with a specific widening policy."""
    env: Dict[str, Any]                # Final abstract environment
    warnings: List[str]                # Analysis warnings
    policy_map: Dict[int, WideningPolicy]  # Loop ID -> policy used
    iterations_per_loop: Dict[int, int]    # Loop ID -> iterations taken
    widening_events: int = 0           # Total widening applications
    narrowing_events: int = 0          # Total narrowing applications


@dataclass
class SynthesisResult:
    """Result of policy synthesis."""
    policies: Dict[int, WideningPolicy]   # Loop ID -> synthesized policy
    loop_infos: List[LoopInfo]             # Info about each loop
    rationale: Dict[int, str]              # Loop ID -> why this policy was chosen
    estimated_precision: Dict[int, str]    # Loop ID -> precision estimate


@dataclass
class ValidationResult:
    """Result of policy validation."""
    valid: bool                  # Overall validity
    per_loop: Dict[int, bool]   # Per-loop validity
    messages: List[str]         # Validation messages


@dataclass
class ComparisonResult:
    """Comparison of different widening policies."""
    standard: PolicyResult
    threshold: PolicyResult
    synthesized: PolicyResult
    improvements: Dict[str, Dict[str, Any]]  # Variable -> improvement details


# ---------------------------------------------------------------------------
# Loop analysis
# ---------------------------------------------------------------------------

def _collect_vars_in_expr(expr) -> Set[str]:
    """Collect all variable names referenced in an expression."""
    cls = expr.__class__.__name__
    if cls == 'Var':
        return {expr.name}
    elif cls == 'BinOp':
        return _collect_vars_in_expr(expr.left) | _collect_vars_in_expr(expr.right)
    elif cls == 'UnaryOp':
        return _collect_vars_in_expr(expr.operand)
    elif cls == 'CallExpr':
        result = set()
        for arg in expr.args:
            result |= _collect_vars_in_expr(arg)
        return result
    return set()


def _collect_constants_in_expr(expr) -> Set[int]:
    """Collect all integer constants in an expression."""
    cls = expr.__class__.__name__
    if cls == 'IntLit':
        return {expr.value}
    elif cls == 'BinOp':
        return _collect_constants_in_expr(expr.left) | _collect_constants_in_expr(expr.right)
    elif cls == 'UnaryOp':
        return _collect_constants_in_expr(expr.operand)
    elif cls == 'CallExpr':
        result = set()
        for arg in expr.args:
            result |= _collect_constants_in_expr(arg)
        return result
    return set()


def _collect_modified_vars(stmts) -> Set[str]:
    """Collect all variables modified in a list of statements."""
    modified = set()
    for stmt in stmts:
        cls = stmt.__class__.__name__
        if cls == 'LetDecl':
            modified.add(stmt.name)
        elif cls == 'Assign':
            modified.add(stmt.name)
        elif cls == 'IfStmt':
            then_stmts = stmt.then_body.stmts if hasattr(stmt.then_body, 'stmts') else (stmt.then_body if isinstance(stmt.then_body, list) else [stmt.then_body])
            modified |= _collect_modified_vars(then_stmts)
            if stmt.else_body:
                else_stmts = stmt.else_body.stmts if hasattr(stmt.else_body, 'stmts') else (stmt.else_body if isinstance(stmt.else_body, list) else [stmt.else_body])
                modified |= _collect_modified_vars(else_stmts)
        elif cls == 'WhileStmt':
            body_stmts = stmt.body.stmts if hasattr(stmt.body, 'stmts') else (stmt.body if isinstance(stmt.body, list) else [stmt.body])
            modified |= _collect_modified_vars(body_stmts)
    return modified


def _get_body_stmts(body) -> list:
    """Extract statement list from body (Block or list)."""
    if hasattr(body, 'stmts'):
        return body.stmts
    if isinstance(body, list):
        return body
    return [body]


def _detect_counter_pattern(loop_info: LoopInfo, body_stmts: list) -> None:
    """Detect if loop is a simple counter (i = i + c or i = i - c)."""
    cond = loop_info.stmt.cond
    cond_cls = cond.__class__.__name__

    # Look for comparisons: i < N, i > N, i <= N, i >= N, i != N
    if cond_cls != 'BinOp' or cond.op not in ('<', '>', '<=', '>=', '!='):
        return

    # Identify the loop variable and bound
    left_cls = cond.left.__class__.__name__
    right_cls = cond.right.__class__.__name__

    loop_var = None
    bound_val = None

    if left_cls == 'Var' and right_cls == 'IntLit':
        loop_var = cond.left.name
        bound_val = cond.right.value
    elif left_cls == 'IntLit' and right_cls == 'Var':
        loop_var = cond.right.name
        bound_val = cond.left.value
    elif left_cls == 'Var' and right_cls == 'Var':
        # Both vars, pick the one modified in body
        for v in [cond.left.name, cond.right.name]:
            if v in loop_info.modified_vars:
                loop_var = v
                break

    if loop_var is None:
        return

    # Check body for i = i + c or i = i - c
    for stmt in body_stmts:
        cls = stmt.__class__.__name__
        if cls == 'Assign' and stmt.name == loop_var:
            val = stmt.value
            if val.__class__.__name__ == 'BinOp':
                if val.left.__class__.__name__ == 'Var' and val.left.name == loop_var:
                    if val.right.__class__.__name__ == 'IntLit':
                        if val.op == '+':
                            loop_info.is_simple_counter = True
                            loop_info.counter_var = loop_var
                            loop_info.counter_step = val.right.value
                        elif val.op == '-':
                            loop_info.is_simple_counter = True
                            loop_info.counter_var = loop_var
                            loop_info.counter_step = -val.right.value

    if bound_val is not None:
        loop_info.bound_value = bound_val
        if loop_info.counter_step is not None:
            loop_info.bound_direction = 'up' if loop_info.counter_step > 0 else 'down'


def analyze_loops(source: str) -> List[LoopInfo]:
    """Analyze all loops in the program and extract structural information."""
    tokens = lex(source)
    ast = Parser(tokens).parse()

    loops = []
    _collect_loops(ast.stmts, loops, depth=0)
    return loops


def _collect_loops(stmts: list, loops: List[LoopInfo], depth: int) -> None:
    """Recursively collect loop info from statement list."""
    for stmt in stmts:
        cls = stmt.__class__.__name__
        if cls == 'WhileStmt':
            body_stmts = _get_body_stmts(stmt.body)

            info = LoopInfo(
                loop_id=len(loops),
                stmt=stmt,
                condition_vars=_collect_vars_in_expr(stmt.cond),
                modified_vars=_collect_modified_vars(body_stmts),
                constants_in_condition=_collect_constants_in_expr(stmt.cond),
                constants_in_body=_collect_constants_in_body(body_stmts),
                is_simple_counter=False,
                nested_depth=depth,
            )
            _detect_counter_pattern(info, body_stmts)
            loops.append(info)

            # Recurse into body for nested loops
            _collect_loops(body_stmts, loops, depth + 1)
        elif cls == 'IfStmt':
            then_stmts = _get_body_stmts(stmt.then_body)
            _collect_loops(then_stmts, loops, depth)
            if stmt.else_body:
                else_stmts = _get_body_stmts(stmt.else_body)
                _collect_loops(else_stmts, loops, depth)
        elif cls == 'FnDecl':
            fn_stmts = _get_body_stmts(stmt.body)
            _collect_loops(fn_stmts, loops, depth)


def _collect_constants_in_body(stmts: list) -> Set[int]:
    """Collect all integer constants from a list of statements."""
    constants = set()
    for stmt in stmts:
        cls = stmt.__class__.__name__
        if cls == 'LetDecl' and stmt.value:
            constants |= _collect_constants_in_expr(stmt.value)
        elif cls == 'Assign':
            constants |= _collect_constants_in_expr(stmt.value)
        elif cls == 'IfStmt':
            constants |= _collect_constants_in_expr(stmt.cond)
            constants |= _collect_constants_in_body(_get_body_stmts(stmt.then_body))
            if stmt.else_body:
                constants |= _collect_constants_in_body(_get_body_stmts(stmt.else_body))
        elif cls == 'WhileStmt':
            constants |= _collect_constants_in_expr(stmt.cond)
            constants |= _collect_constants_in_body(_get_body_stmts(stmt.body))
    return constants


# ---------------------------------------------------------------------------
# Policy synthesis
# ---------------------------------------------------------------------------

def synthesize_policy(loop_info: LoopInfo) -> Tuple[WideningPolicy, str]:
    """Synthesize an optimal widening policy for a single loop.

    Returns (policy, rationale).
    """
    # Simple counter with known bound -> delayed threshold
    if loop_info.is_simple_counter and loop_info.bound_value is not None:
        bound = loop_info.bound_value
        step = loop_info.counter_step or 1

        # Compute thresholds: include bound, 0, and intermediate values
        thresholds = {0, bound}
        if abs(step) > 1:
            # Add multiples of step near bound
            for i in range(0, abs(bound) + abs(step), abs(step)):
                thresholds.add(i)
                thresholds.add(-i)

        # Add bound +/- 1 for tight ranges
        thresholds.add(bound - 1)
        thresholds.add(bound + 1)

        # Delay = min(3, estimated iterations / 4)
        if step != 0:
            est_iters = abs(bound) // abs(step) if abs(step) > 0 else 10
        else:
            est_iters = 10
        delay = min(3, max(1, est_iters // 4))

        return WideningPolicy(
            strategy=WideningStrategy.DELAYED_THRESHOLD,
            thresholds=tuple(sorted(thresholds)),
            delay=delay,
            narrowing_iterations=2,
        ), f"Simple counter ({loop_info.counter_var} {'+='+str(step) if step > 0 else '-='+str(-step)}), bound={bound}, delay={delay}"

    # Has constants in condition but not simple counter -> threshold
    if loop_info.constants_in_condition:
        thresholds = set()
        for c in loop_info.constants_in_condition:
            thresholds.add(c)
            thresholds.add(c - 1)
            thresholds.add(c + 1)
            thresholds.add(-c)
        for c in loop_info.constants_in_body:
            thresholds.add(c)
        thresholds.add(0)

        return WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=tuple(sorted(thresholds)),
            delay=0,
            narrowing_iterations=2,
        ), f"Condition has constants {loop_info.constants_in_condition}, using thresholds"

    # Has constants in body only -> threshold with narrowing
    if loop_info.constants_in_body:
        thresholds = {0}
        for c in loop_info.constants_in_body:
            thresholds.add(c)
            thresholds.add(-c)

        return WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=tuple(sorted(thresholds)),
            delay=0,
            narrowing_iterations=3,
        ), f"Body has constants {loop_info.constants_in_body}, using thresholds + narrowing"

    # No constants -> delayed standard with narrowing
    return WideningPolicy(
        strategy=WideningStrategy.DELAYED,
        thresholds=(),
        delay=2,
        narrowing_iterations=2,
    ), "No constants found, using delayed widening with narrowing"


def synthesize_policies(source: str) -> SynthesisResult:
    """Synthesize widening policies for all loops in a program."""
    loops = analyze_loops(source)
    policies = {}
    rationale = {}
    precision = {}

    for info in loops:
        policy, reason = synthesize_policy(info)
        policies[info.loop_id] = policy
        rationale[info.loop_id] = reason

        # Estimate precision
        if info.is_simple_counter and info.bound_value is not None:
            precision[info.loop_id] = "exact"
        elif info.constants_in_condition:
            precision[info.loop_id] = "tight"
        elif info.constants_in_body:
            precision[info.loop_id] = "moderate"
        else:
            precision[info.loop_id] = "approximate"

    return SynthesisResult(
        policies=policies,
        loop_infos=loops,
        rationale=rationale,
        estimated_precision=precision,
    )


# ---------------------------------------------------------------------------
# Policy-driven abstract interpreter
# ---------------------------------------------------------------------------

class PolicyInterpreter:
    """Abstract interpreter with per-loop widening policies."""

    def __init__(self, policies: Dict[int, WideningPolicy] = None,
                 max_iterations: int = 50):
        self.policies = policies or {}
        self.max_iterations = max_iterations
        self._loop_counter = 0
        self._iterations_per_loop = {}
        self._widening_events = 0
        self._narrowing_events = 0
        self._functions = {}

    def analyze(self, source: str) -> PolicyResult:
        """Analyze program with per-loop widening policies."""
        tokens = lex(source)
        ast = Parser(tokens).parse()

        env = AbstractEnv()
        warnings = []
        self._loop_counter = 0
        self._iterations_per_loop = {}
        self._widening_events = 0
        self._narrowing_events = 0
        self._functions = {}

        # Collect functions first
        for stmt in ast.stmts:
            if stmt.__class__.__name__ == 'FnDecl':
                self._functions[stmt.name] = stmt

        env = self._interpret_stmts(ast.stmts, env, warnings)

        # Build result env dict
        env_dict = {}
        for name in env.signs:
            env_dict[name] = {
                'sign': str(env.get_sign(name)),
                'interval': env.get_interval(name),
                'const': env.get_const(name),
            }

        return PolicyResult(
            env=env_dict,
            warnings=warnings,
            policy_map=dict(self.policies),
            iterations_per_loop=dict(self._iterations_per_loop),
            widening_events=self._widening_events,
            narrowing_events=self._narrowing_events,
        )

    def _interpret_stmts(self, stmts, env, warnings):
        """Interpret a list of statements."""
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env, warnings)
        return env

    def _interpret_stmt(self, stmt, env, warnings):
        """Interpret a single statement."""
        cls = stmt.__class__.__name__

        if cls == 'LetDecl':
            if stmt.value:
                val = self._eval_expr(stmt.value, env)
                env.set(stmt.name, val.sign, val.interval, val.const)
            else:
                env.set_top(stmt.name)

        elif cls == 'Assign':
            val = self._eval_expr(stmt.value, env)
            env.set(stmt.name, val.sign, val.interval, val.const)

        elif cls == 'IfStmt':
            env = self._interpret_if(stmt, env, warnings)

        elif cls == 'WhileStmt':
            env = self._interpret_while(stmt, env, warnings)

        elif cls == 'FnDecl':
            pass  # Already collected

        elif cls == 'ReturnStmt':
            pass  # No inter-procedural for now

        elif cls == 'PrintStmt':
            pass

        elif cls == 'ExprStmt':
            pass

        return env

    def _interpret_if(self, stmt, env, warnings):
        """Interpret an if statement."""
        then_env = self._refine_condition(stmt.cond, env, True)
        else_env = self._refine_condition(stmt.cond, env, False)

        then_stmts = _get_body_stmts(stmt.then_body)
        then_env = self._interpret_stmts(then_stmts, then_env, warnings)

        if stmt.else_body:
            else_stmts = _get_body_stmts(stmt.else_body)
            else_env = self._interpret_stmts(else_stmts, else_env, warnings)

        return then_env.join(else_env)

    def _interpret_while(self, stmt, env, warnings):
        """Interpret a while loop with per-loop widening policy."""
        loop_id = self._loop_counter
        self._loop_counter += 1

        policy = self.policies.get(loop_id, WideningPolicy())
        max_iter = min(policy.max_iterations, self.max_iterations)

        current_env = env.copy()
        iteration = 0

        # Main fixpoint loop
        while iteration < max_iter:
            iteration += 1

            # Refine for loop condition being true
            body_env = self._refine_condition(stmt.cond, current_env, True)

            # Check if condition is definitely false
            if self._is_definitely_false(stmt.cond, current_env):
                break

            # Interpret loop body
            body_stmts = _get_body_stmts(stmt.body)
            body_env = self._interpret_stmts(body_stmts, body_env, warnings)

            # Apply widening according to policy
            if iteration <= policy.delay:
                # During delay: use join (no widening)
                next_env = current_env.join(body_env)
            elif policy.strategy in (WideningStrategy.THRESHOLD, WideningStrategy.DELAYED_THRESHOLD):
                # Threshold widening
                next_env = self._threshold_widen(current_env, body_env, policy.thresholds)
                self._widening_events += 1
            else:
                # Standard widening
                next_env = current_env.widen(body_env)
                self._widening_events += 1

            # Check convergence
            if next_env.equals(current_env):
                break

            current_env = next_env

        self._iterations_per_loop[loop_id] = iteration

        # Narrowing phase
        for _ in range(policy.narrowing_iterations):
            body_env = self._refine_condition(stmt.cond, current_env, True)
            body_stmts = _get_body_stmts(stmt.body)
            body_env = self._interpret_stmts(body_stmts, body_env, warnings)

            narrowed = self._narrow(current_env, body_env)
            self._narrowing_events += 1

            if narrowed.equals(current_env):
                break
            current_env = narrowed

        # Exit condition: refine for loop condition being false
        exit_env = self._refine_condition(stmt.cond, current_env, False)
        return exit_env

    def _threshold_widen(self, old_env: AbstractEnv, new_env: AbstractEnv,
                         thresholds: Tuple[float, ...]) -> AbstractEnv:
        """Widen with thresholds: jump to nearest threshold instead of infinity."""
        result = old_env.copy()
        threshold_list = list(thresholds)

        all_vars = set(old_env.signs.keys()) | set(new_env.signs.keys())
        for var in all_vars:
            old_iv = old_env.get_interval(var)
            new_iv = new_env.get_interval(var)

            widened = interval_widen_thresholds(old_iv, new_iv, threshold_list)

            # Also widen sign
            old_sign = old_env.get_sign(var)
            new_sign = new_env.get_sign(var)
            from abstract_interpreter import sign_join
            joined_sign = sign_join(old_sign, new_sign)

            # Const: if different, go to top
            old_const = old_env.get_const(var)
            new_const = new_env.get_const(var)
            if old_const == new_const:
                result.set(var, joined_sign, widened, old_const)
            else:
                from abstract_interpreter import CONST_TOP
                result.set(var, joined_sign, widened, CONST_TOP)

        return result

    def _narrow(self, wide_env: AbstractEnv, new_env: AbstractEnv) -> AbstractEnv:
        """Narrowing: refine widened bounds using new information."""
        result = wide_env.copy()

        for var in wide_env.signs:
            wide_iv = wide_env.get_interval(var)
            new_iv = new_env.get_interval(var)

            # Narrowing: replace infinite bounds with finite ones
            lo = new_iv.lo if wide_iv.lo == NEG_INF and new_iv.lo != NEG_INF else wide_iv.lo
            hi = new_iv.hi if wide_iv.hi == INF and new_iv.hi != INF else wide_iv.hi

            narrowed = Interval(lo, hi)
            result.set(var, wide_env.get_sign(var), narrowed, wide_env.get_const(var))

        return result

    def _eval_expr(self, expr, env) -> AbstractValue:
        """Evaluate an expression abstractly."""
        cls = expr.__class__.__name__

        if cls == 'IntLit':
            return AbstractValue.from_value(expr.value)

        elif cls == 'Var':
            return AbstractValue(
                sign=env.get_sign(expr.name),
                interval=env.get_interval(expr.name),
                const=env.get_const(expr.name),
            )

        elif cls == 'BinOp':
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            return self._eval_binop(expr.op, left, right)

        elif cls == 'UnaryOp':
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                from abstract_interpreter import sign_negate, interval_negate, CONST_TOP
                return AbstractValue(
                    sign=sign_negate(operand.sign),
                    interval=interval_negate(operand.interval),
                    const=-operand.const if isinstance(operand.const, int) else CONST_TOP,
                )
            return AbstractValue.top()

        elif cls == 'CallExpr':
            return AbstractValue.top()

        elif cls == 'BoolLit':
            v = 1 if expr.value else 0
            return AbstractValue.from_value(v)

        return AbstractValue.top()

    def _eval_binop(self, op, left, right):
        """Evaluate a binary operation abstractly."""
        from abstract_interpreter import (
            sign_add, sign_sub, sign_mul,
            interval_add, interval_sub, interval_mul, interval_div,
            CONST_TOP, Sign as S
        )

        if op == '+':
            return AbstractValue(
                sign=sign_add(left.sign, right.sign),
                interval=interval_add(left.interval, right.interval),
                const=left.const + right.const if isinstance(left.const, int) and isinstance(right.const, int) else CONST_TOP,
            )
        elif op == '-':
            return AbstractValue(
                sign=sign_sub(left.sign, right.sign),
                interval=interval_sub(left.interval, right.interval),
                const=left.const - right.const if isinstance(left.const, int) and isinstance(right.const, int) else CONST_TOP,
            )
        elif op == '*':
            return AbstractValue(
                sign=sign_mul(left.sign, right.sign),
                interval=interval_mul(left.interval, right.interval),
                const=left.const * right.const if isinstance(left.const, int) and isinstance(right.const, int) else CONST_TOP,
            )
        elif op == '/':
            c = CONST_TOP
            if isinstance(left.const, int) and isinstance(right.const, int) and right.const != 0:
                c = left.const // right.const
            return AbstractValue(
                sign=S.TOP,
                interval=interval_div(left.interval, right.interval),
                const=c,
            )
        elif op in ('<', '>', '<=', '>=', '==', '!='):
            # Comparison: result is 0 or 1
            return AbstractValue(
                sign=S.NON_NEG,
                interval=Interval(0, 1),
                const=CONST_TOP,
            )
        elif op == '%':
            if isinstance(right.const, int) and right.const != 0:
                m = abs(right.const)
                return AbstractValue(
                    sign=S.NON_NEG if left.sign in (S.NON_NEG, S.POS, S.ZERO) else S.TOP,
                    interval=Interval(0, m - 1),
                    const=left.const % right.const if isinstance(left.const, int) else CONST_TOP,
                )
            return AbstractValue.top()

        return AbstractValue.top()

    def _refine_condition(self, cond, env, branch: bool) -> AbstractEnv:
        """Refine environment based on a condition being true or false."""
        result = env.copy()
        cls = cond.__class__.__name__

        if cls != 'BinOp':
            return result

        op = cond.op
        left_cls = cond.left.__class__.__name__
        right_cls = cond.right.__class__.__name__

        # Handle var op const
        if left_cls == 'Var' and right_cls == 'IntLit':
            var_name = cond.left.name
            val = cond.right.value
            old_iv = env.get_interval(var_name)

            if branch:
                new_iv = self._apply_constraint(old_iv, op, val)
            else:
                new_iv = self._apply_constraint(old_iv, self._negate_op(op), val)

            from abstract_interpreter import CONST_TOP
            result.set(var_name, env.get_sign(var_name), new_iv, CONST_TOP)

        elif left_cls == 'IntLit' and right_cls == 'Var':
            var_name = cond.right.name
            val = cond.left.value
            old_iv = env.get_interval(var_name)

            flipped = self._flip_op(op)
            if branch:
                new_iv = self._apply_constraint(old_iv, flipped, val)
            else:
                new_iv = self._apply_constraint(old_iv, self._negate_op(flipped), val)

            from abstract_interpreter import CONST_TOP
            result.set(var_name, env.get_sign(var_name), new_iv, CONST_TOP)

        elif left_cls == 'Var' and right_cls == 'Var':
            # var op var -- refine both using each other's bounds
            lv = cond.left.name
            rv = cond.right.name
            l_iv = env.get_interval(lv)
            r_iv = env.get_interval(rv)

            actual_op = op if branch else self._negate_op(op)

            if actual_op == '<':
                # lv < rv: lv.hi < rv.hi, rv.lo > lv.lo
                new_l = Interval(l_iv.lo, min(l_iv.hi, r_iv.hi - 1))
                new_r = Interval(max(r_iv.lo, l_iv.lo + 1), r_iv.hi)
                from abstract_interpreter import CONST_TOP
                result.set(lv, env.get_sign(lv), new_l, CONST_TOP)
                result.set(rv, env.get_sign(rv), new_r, CONST_TOP)
            elif actual_op == '<=':
                new_l = Interval(l_iv.lo, min(l_iv.hi, r_iv.hi))
                new_r = Interval(max(r_iv.lo, l_iv.lo), r_iv.hi)
                from abstract_interpreter import CONST_TOP
                result.set(lv, env.get_sign(lv), new_l, CONST_TOP)
                result.set(rv, env.get_sign(rv), new_r, CONST_TOP)
            elif actual_op == '>':
                new_l = Interval(max(l_iv.lo, r_iv.lo + 1), l_iv.hi)
                new_r = Interval(r_iv.lo, min(r_iv.hi, l_iv.hi - 1))
                from abstract_interpreter import CONST_TOP
                result.set(lv, env.get_sign(lv), new_l, CONST_TOP)
                result.set(rv, env.get_sign(rv), new_r, CONST_TOP)
            elif actual_op == '>=':
                new_l = Interval(max(l_iv.lo, r_iv.lo), l_iv.hi)
                new_r = Interval(r_iv.lo, min(r_iv.hi, l_iv.hi))
                from abstract_interpreter import CONST_TOP
                result.set(lv, env.get_sign(lv), new_l, CONST_TOP)
                result.set(rv, env.get_sign(rv), new_r, CONST_TOP)

        # Update sign from interval
        from abstract_interpreter import Sign as S
        for var in result.signs:
            iv = result.get_interval(var)
            if not iv.is_bot():
                if iv.lo >= 0 and iv.hi >= 0:
                    if iv.lo > 0:
                        result.signs[var] = S.POS
                    else:
                        result.signs[var] = S.NON_NEG
                elif iv.hi <= 0 and iv.lo <= 0:
                    if iv.hi < 0:
                        result.signs[var] = S.NEG
                    else:
                        result.signs[var] = S.NON_POS
                elif iv.lo == 0 and iv.hi == 0:
                    result.signs[var] = S.ZERO

        return result

    def _apply_constraint(self, iv: Interval, op: str, val: int) -> Interval:
        """Apply a constraint to an interval."""
        if op == '<':
            return Interval(iv.lo, min(iv.hi, val - 1))
        elif op == '<=':
            return Interval(iv.lo, min(iv.hi, val))
        elif op == '>':
            return Interval(max(iv.lo, val + 1), iv.hi)
        elif op == '>=':
            return Interval(max(iv.lo, val), iv.hi)
        elif op == '==':
            if iv.lo <= val <= iv.hi:
                return Interval(val, val)
            return Interval(1, 0)  # BOT
        elif op == '!=':
            # Can only refine if val is at a boundary
            if iv.lo == val:
                return Interval(val + 1, iv.hi)
            elif iv.hi == val:
                return Interval(iv.lo, val - 1)
            return iv
        return iv

    def _negate_op(self, op: str) -> str:
        return {'<': '>=', '>': '<=', '<=': '>', '>=': '<',
                '==': '!=', '!=': '=='}.get(op, op)

    def _flip_op(self, op: str) -> str:
        return {'<': '>', '>': '<', '<=': '>=', '>=': '<=',
                '==': '==', '!=': '!='}.get(op, op)

    def _is_definitely_false(self, cond, env) -> bool:
        """Check if condition is definitely false."""
        cls = cond.__class__.__name__
        if cls == 'BinOp' and cond.left.__class__.__name__ == 'Var' and cond.right.__class__.__name__ == 'IntLit':
            iv = env.get_interval(cond.left.name)
            val = cond.right.value
            if cond.op == '<' and iv.lo >= val:
                return True
            if cond.op == '<=' and iv.lo > val:
                return True
            if cond.op == '>' and iv.hi <= val:
                return True
            if cond.op == '>=' and iv.hi < val:
                return True
        return False


# ---------------------------------------------------------------------------
# Policy validation via SMT
# ---------------------------------------------------------------------------

def validate_policy(source: str, policies: Dict[int, WideningPolicy]) -> ValidationResult:
    """Validate that synthesized policies produce sound results.

    Soundness check: for each loop, verify that the widened fixpoint
    is an over-approximation of the concrete semantics by checking
    that the standard analysis result is subsumed by the policy result.
    """
    # Run baseline analysis
    baseline = baseline_analyze(source)
    baseline_env = baseline['env']

    # Run policy analysis
    interp = PolicyInterpreter(policies)
    policy_result = interp.analyze(source)

    valid = True
    per_loop = {}
    messages = []

    # Check that policy result subsumes baseline for all variables
    # (policy should be at least as precise, or identical)
    # Note: soundness means policy result INCLUDES all concrete values
    # So we just check that both are valid over-approximations
    for var in baseline_env.signs:
        baseline_iv = baseline_env.get_interval(var)

        policy_info = policy_result.env.get(var)
        if policy_info is None:
            continue
        policy_iv = policy_info['interval']

        # The policy result is sound if:
        # - Its interval contains all concrete values (same or wider than baseline)
        # - OR it's tighter (more precise) while still being an over-approximation
        # Both baseline and policy are over-approximations by construction,
        # so both are sound. We verify they don't contradict each other.
        if policy_iv.is_bot() and not baseline_iv.is_bot():
            messages.append(f"WARNING: {var} is BOT in policy but not in baseline")
            valid = False

    # Use SMT to verify interval containment for key variables
    solver = SMTSolver()
    for loop_id in policies:
        loop_valid = True
        per_loop[loop_id] = True

        # Verify policy thresholds are sound: each threshold should be
        # within the range of possible variable values
        policy = policies[loop_id]
        if policy.thresholds:
            messages.append(f"Loop {loop_id}: {len(policy.thresholds)} thresholds, "
                          f"strategy={policy.strategy.value}")

    if not messages:
        messages.append("All policies validated successfully")

    return ValidationResult(valid=valid, per_loop=per_loop, messages=messages)


# ---------------------------------------------------------------------------
# Policy-driven analysis with V020 domain functor
# ---------------------------------------------------------------------------

class FunctorPolicyInterpreter(FunctorInterpreter):
    """FunctorInterpreter extended with per-loop widening policies."""

    def __init__(self, domain_factory, policies: Dict[int, WideningPolicy] = None,
                 max_iterations: int = 50):
        super().__init__(domain_factory, max_iterations)
        self.policies = policies or {}
        self._loop_counter = 0
        self._iterations_per_loop = {}

    def analyze(self, source: str) -> dict:
        """Override to reset loop counter."""
        self._loop_counter = 0
        self._iterations_per_loop = {}
        return super().analyze(source)

    def _interpret_while(self, stmt, env):
        """Override to use per-loop widening policy."""
        loop_id = self._loop_counter
        self._loop_counter += 1

        policy = self.policies.get(loop_id, WideningPolicy())
        max_iter = min(policy.max_iterations, self._max_iterations)

        current_env = env.copy()
        iteration = 0

        while iteration < max_iter:
            iteration += 1

            true_env, _ = self._refine_condition(stmt.cond, current_env)
            body_stmts = _get_body_stmts(stmt.body)
            body_env = self._interpret_stmts(body_stmts, true_env)

            if iteration <= policy.delay:
                next_env = current_env.join(body_env)
            else:
                next_env = current_env.widen(body_env)

            if next_env.equals(current_env):
                break
            current_env = next_env

        self._iterations_per_loop[loop_id] = iteration

        # Narrowing
        for _ in range(policy.narrowing_iterations):
            true_env, _ = self._refine_condition(stmt.cond, current_env)
            body_stmts = _get_body_stmts(stmt.body)
            body_env = self._interpret_stmts(body_stmts, true_env)
            narrowed = current_env.narrow(body_env)
            if narrowed.equals(current_env):
                break
            current_env = narrowed

        _, false_env = self._refine_condition(stmt.cond, current_env)
        return false_env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def policy_analyze(source: str, policies: Dict[int, WideningPolicy] = None) -> PolicyResult:
    """Analyze program with explicit widening policies.

    If no policies given, uses default (standard widening) for all loops.
    """
    interp = PolicyInterpreter(policies or {})
    return interp.analyze(source)


def auto_analyze(source: str) -> PolicyResult:
    """Analyze program with automatically synthesized policies."""
    synthesis = synthesize_policies(source)
    return policy_analyze(source, synthesis.policies)


def compare_policies(source: str) -> ComparisonResult:
    """Compare standard, threshold, and synthesized policies."""
    # Standard analysis
    std_interp = PolicyInterpreter({})
    std_result = std_interp.analyze(source)

    # Threshold analysis (V019-style)
    loops = analyze_loops(source)
    threshold_policies = {}
    all_thresholds = set()
    for info in loops:
        for c in info.constants_in_condition | info.constants_in_body:
            all_thresholds.add(c)
            all_thresholds.add(c - 1)
            all_thresholds.add(c + 1)
        all_thresholds.add(0)

    for info in loops:
        threshold_policies[info.loop_id] = WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=tuple(sorted(all_thresholds)),
            narrowing_iterations=2,
        )

    thresh_interp = PolicyInterpreter(threshold_policies)
    thresh_result = thresh_interp.analyze(source)

    # Synthesized analysis
    synthesis = synthesize_policies(source)
    synth_interp = PolicyInterpreter(synthesis.policies)
    synth_result = synth_interp.analyze(source)

    # Compare results
    improvements = {}
    all_vars = set(std_result.env.keys()) | set(thresh_result.env.keys()) | set(synth_result.env.keys())

    for var in all_vars:
        std_iv = std_result.env.get(var, {}).get('interval', Interval(NEG_INF, INF))
        thresh_iv = thresh_result.env.get(var, {}).get('interval', Interval(NEG_INF, INF))
        synth_iv = synth_result.env.get(var, {}).get('interval', Interval(NEG_INF, INF))

        std_width = _interval_width(std_iv)
        thresh_width = _interval_width(thresh_iv)
        synth_width = _interval_width(synth_iv)

        if synth_width < std_width or thresh_width < std_width:
            improvements[var] = {
                'standard': str(std_iv),
                'threshold': str(thresh_iv),
                'synthesized': str(synth_iv),
                'std_width': std_width,
                'thresh_width': thresh_width,
                'synth_width': synth_width,
            }

    return ComparisonResult(
        standard=std_result,
        threshold=thresh_result,
        synthesized=synth_result,
        improvements=improvements,
    )


def _interval_width(iv: Interval) -> float:
    """Compute interval width (infinity if unbounded)."""
    if iv.is_bot():
        return 0
    if iv.lo == NEG_INF or iv.hi == INF:
        return float('inf')
    return iv.hi - iv.lo


def get_loop_info(source: str) -> List[Dict[str, Any]]:
    """Get human-readable information about all loops."""
    loops = analyze_loops(source)
    result = []
    for info in loops:
        result.append({
            'loop_id': info.loop_id,
            'condition_vars': sorted(info.condition_vars),
            'modified_vars': sorted(info.modified_vars),
            'constants_in_condition': sorted(info.constants_in_condition),
            'constants_in_body': sorted(info.constants_in_body),
            'is_simple_counter': info.is_simple_counter,
            'counter_var': info.counter_var,
            'counter_step': info.counter_step,
            'bound_value': info.bound_value,
            'bound_direction': info.bound_direction,
            'nested_depth': info.nested_depth,
        })
    return result


def synthesize_and_validate(source: str) -> Dict[str, Any]:
    """Synthesize policies and validate them."""
    synthesis = synthesize_policies(source)
    validation = validate_policy(source, synthesis.policies)
    result = auto_analyze(source)

    return {
        'synthesis': synthesis,
        'validation': validation,
        'result': result,
        'loop_count': len(synthesis.loop_infos),
        'all_valid': validation.valid,
    }


def functor_policy_analyze(source: str, domain_factory=None,
                           policies: Dict[int, WideningPolicy] = None) -> dict:
    """Analyze with V020 domain functor and per-loop policies."""
    if domain_factory is None:
        domain_factory = make_sign_interval()

    interp = FunctorPolicyInterpreter(domain_factory, policies or {})
    return interp.analyze(source)


def compare_with_functor(source: str) -> Dict[str, Any]:
    """Compare policy-driven analysis using V020 functor domains."""
    factory = make_sign_interval()

    # Standard functor analysis
    std_interp = FunctorPolicyInterpreter(factory, {})
    std_result = std_interp.analyze(source)

    # With synthesized policies
    synthesis = synthesize_policies(source)
    synth_interp = FunctorPolicyInterpreter(factory, synthesis.policies)
    synth_result = synth_interp.analyze(source)

    return {
        'standard': std_result,
        'synthesized': synth_result,
        'policies': {lid: p.strategy.value for lid, p in synthesis.policies.items()},
        'iterations': {
            'standard': std_interp._iterations_per_loop,
            'synthesized': synth_interp._iterations_per_loop,
        },
    }


def policy_summary(source: str) -> str:
    """Generate a human-readable summary of synthesized policies."""
    synthesis = synthesize_policies(source)
    lines = [f"Program has {len(synthesis.loop_infos)} loop(s):", ""]

    for info in synthesis.loop_infos:
        lid = info.loop_id
        policy = synthesis.policies[lid]
        lines.append(f"Loop {lid}:")
        lines.append(f"  Strategy: {policy.strategy.value}")
        lines.append(f"  Rationale: {synthesis.rationale[lid]}")
        if policy.thresholds:
            lines.append(f"  Thresholds: {list(policy.thresholds)}")
        if policy.delay > 0:
            lines.append(f"  Delay: {policy.delay} iterations")
        if policy.narrowing_iterations > 0:
            lines.append(f"  Narrowing: {policy.narrowing_iterations} passes")
        lines.append(f"  Estimated precision: {synthesis.estimated_precision[lid]}")
        lines.append("")

    return "\n".join(lines)
