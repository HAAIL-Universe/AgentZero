"""V055: Modular Abstract Interpretation

Composes V039 (modular verification/contracts) + C039 (abstract interpreter).

Key idea: analyze each function in isolation using its contract as a summary.
- Extract contracts (requires/ensures) from annotated source
- Use ensures clauses to build abstract summaries (interval/sign bounds)
- At call sites, apply the callee's summary instead of re-analyzing the body
- Use contract-derived thresholds for widening in loops
- Topological ordering ensures callees are summarized before callers

This gives modular abstract interpretation: each function is analyzed once,
producing a FunctionSummary that maps abstract input domains to abstract output domains.
"""

import os, sys

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, "challenges", "C010_stack_vm"))
sys.path.insert(0, os.path.join(_az, "challenges", "C037_smt_solver"))
sys.path.insert(0, os.path.join(_az, "challenges", "C039_abstract_interpreter"))
sys.path.insert(0, os.path.join(_work, "V039_modular_verification"))
sys.path.insert(0, os.path.join(_work, "V004_verification_conditions"))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from stack_vm import (
    lex, Parser, Program, Var as ASTVar, IntLit, BinOp, UnaryOp,
    IfStmt, WhileStmt, LetDecl, Assign, FnDecl, CallExpr, Block,
    ReturnStmt, PrintStmt
)
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, AbstractValue,
    Sign, Interval, Warning, WarningKind,
    sign_join, interval_join, interval_widen,
    INTERVAL_BOT, INTERVAL_TOP,
    analyze as ai_analyze,
)
from modular_verification import (
    extract_all_contracts, extract_contract, ContractStore, Contract,
    verify_program_modular,
)
# SExpr types from V004 (used in contract clauses)
from vc_gen import SBinOp, SVar, SInt, SExpr


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AbstractBound:
    """An abstract bound extracted from a contract clause."""
    variable: str
    lower: Optional[int] = None   # None means -inf
    upper: Optional[int] = None   # None means +inf
    sign: Optional[Sign] = None


@dataclass
class FunctionSummary:
    """Abstract summary of a function's behavior."""
    fn_name: str
    params: List[str]
    # Maps param names to their precondition bounds
    param_bounds: Dict[str, AbstractBound] = field(default_factory=dict)
    # Maps result/output variable to postcondition bounds
    result_bounds: Dict[str, AbstractBound] = field(default_factory=dict)
    # Thresholds extracted from contract for widening
    thresholds: List[int] = field(default_factory=list)
    # Abstract environment after analyzing the function body
    body_env: Optional[AbstractEnv] = None
    # Warnings from body analysis
    warnings: List[Warning] = field(default_factory=list)
    # Whether analysis was successful
    analyzed: bool = False


@dataclass
class ModularAIResult:
    """Result of modular abstract interpretation."""
    summaries: Dict[str, FunctionSummary] = field(default_factory=dict)
    global_env: Optional[AbstractEnv] = None
    global_warnings: List[Warning] = field(default_factory=list)
    analysis_order: List[str] = field(default_factory=list)
    contracts: Optional[ContractStore] = None

    @property
    def total_warnings(self) -> int:
        count = len(self.global_warnings)
        for s in self.summaries.values():
            count += len(s.warnings)
        return count

    @property
    def functions_analyzed(self) -> int:
        return sum(1 for s in self.summaries.values() if s.analyzed)

    def get_summary(self, fn_name: str) -> Optional[FunctionSummary]:
        return self.summaries.get(fn_name)

    def get_result_bounds(self, fn_name: str) -> Dict[str, AbstractBound]:
        s = self.summaries.get(fn_name)
        return s.result_bounds if s else {}

    def summary_report(self) -> str:
        lines = [
            f"Modular Abstract Interpretation Report",
            f"  Functions analyzed: {self.functions_analyzed}/{len(self.summaries)}",
            f"  Analysis order: {' -> '.join(self.analysis_order)}",
            f"  Total warnings: {self.total_warnings}",
        ]
        for name, s in self.summaries.items():
            status = "OK" if s.analyzed else "FAILED"
            warns = len(s.warnings)
            lines.append(f"  {name}: {status}, {warns} warnings")
            for var, bound in s.result_bounds.items():
                parts = []
                if bound.lower is not None:
                    parts.append(f"{bound.lower} <=")
                parts.append(var)
                if bound.upper is not None:
                    parts.append(f"<= {bound.upper}")
                if bound.sign is not None:
                    parts.append(f"(sign: {bound.sign.name})")
                lines.append(f"    result: {' '.join(parts)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contract analysis: extract abstract bounds from contract clauses
# ---------------------------------------------------------------------------

def _parse_source(source: str) -> Program:
    return Parser(lex(source)).parse()


def _try_eval_sexpr_int(expr) -> Optional[int]:
    """Try to evaluate an SExpr to a concrete integer."""
    if isinstance(expr, SInt):
        return expr.value
    if isinstance(expr, SBinOp):
        l = _try_eval_sexpr_int(expr.left)
        r = _try_eval_sexpr_int(expr.right)
        if l is not None and r is not None:
            if expr.op == '+': return l + r
            if expr.op == '-': return l - r
            if expr.op == '*': return l * r
    return None


def _extract_bounds_from_expr(expr, var_name: str) -> Optional[AbstractBound]:
    """Try to extract a bound from a comparison SExpr like x >= 0, x < 100.

    Contract clauses use SExpr types (SBinOp, SVar, SInt) from V004.
    """
    if not isinstance(expr, SBinOp):
        return None

    left_var = None
    right_var = None
    left_val = None
    right_val = None

    if isinstance(expr.left, SVar):
        left_var = expr.left.name
    if isinstance(expr.right, SVar):
        right_var = expr.right.name

    left_val = _try_eval_sexpr_int(expr.left)
    right_val = _try_eval_sexpr_int(expr.right)

    # Handle: x >= N, x > N, x <= N, x < N, x == N
    if left_var == var_name and right_val is not None:
        op = expr.op
        if op == '>=':
            return AbstractBound(var_name, lower=right_val)
        elif op == '>':
            return AbstractBound(var_name, lower=right_val + 1)
        elif op == '<=':
            return AbstractBound(var_name, upper=right_val)
        elif op == '<':
            return AbstractBound(var_name, upper=right_val - 1)
        elif op == '==':
            return AbstractBound(var_name, lower=right_val, upper=right_val)

    # Handle: N <= x, N < x, N >= x, N > x
    if right_var == var_name and left_val is not None:
        op = expr.op
        if op == '<=':
            return AbstractBound(var_name, lower=left_val)
        elif op == '<':
            return AbstractBound(var_name, lower=left_val + 1)
        elif op == '>=':
            return AbstractBound(var_name, upper=left_val)
        elif op == '>':
            return AbstractBound(var_name, upper=left_val - 1)
        elif op == '==':
            return AbstractBound(var_name, lower=left_val, upper=left_val)

    return None


def _extract_bounds_from_contract(contract: Contract) -> Tuple[Dict[str, AbstractBound], Dict[str, AbstractBound], List[int]]:
    """Extract abstract bounds from a contract's pre/postconditions.

    Returns (param_bounds, result_bounds, thresholds).
    """
    param_bounds: Dict[str, AbstractBound] = {}
    result_bounds: Dict[str, AbstractBound] = {}
    thresholds: List[int] = []

    all_vars = set(contract.params) | {"result", "__result__"}

    for clause_list, target_bounds in [
        (contract.preconditions, param_bounds),
        (contract.postconditions, result_bounds),
    ]:
        for clause in clause_list:
            for var_name in all_vars:
                bound = _extract_bounds_from_expr(clause, var_name)
                if bound:
                    # Merge with existing bound
                    if var_name in target_bounds:
                        existing = target_bounds[var_name]
                        if bound.lower is not None:
                            if existing.lower is None or bound.lower > existing.lower:
                                existing.lower = bound.lower
                        if bound.upper is not None:
                            if existing.upper is None or bound.upper < existing.upper:
                                existing.upper = bound.upper
                    else:
                        target_bounds[var_name] = bound

                    # Extract thresholds
                    if bound.lower is not None:
                        thresholds.extend([bound.lower - 1, bound.lower, bound.lower + 1])
                    if bound.upper is not None:
                        thresholds.extend([bound.upper - 1, bound.upper, bound.upper + 1])

    return param_bounds, result_bounds, sorted(set(thresholds))


# ---------------------------------------------------------------------------
# Modular abstract interpreter
# ---------------------------------------------------------------------------

class ModularAbstractInterpreter:
    """Abstract interpreter that uses contracts for modular function analysis.

    Instead of re-analyzing callee bodies, uses contract-derived summaries.
    """

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.summaries: Dict[str, FunctionSummary] = {}
        self.contracts: Optional[ContractStore] = None

    def analyze(self, source: str) -> ModularAIResult:
        """Analyze a C10 program modularly.

        1. Extract contracts from annotated source
        2. Build call graph, topological order
        3. Analyze each function in order, building summaries
        4. Analyze global (non-function) code using summaries at call sites
        """
        program = _parse_source(source)
        self.contracts = extract_all_contracts(source)

        # Separate functions and global statements
        functions: Dict[str, FnDecl] = {}
        global_stmts = []
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                functions[stmt.name] = stmt
            else:
                global_stmts.append(stmt)

        # Build call graph and get topological order
        call_graph = _build_call_graph(functions)
        order = _topological_order(call_graph, list(functions.keys()))

        result = ModularAIResult()
        result.contracts = self.contracts
        result.analysis_order = order

        # Analyze each function in dependency order
        for fn_name in order:
            if fn_name not in functions:
                continue
            fn = functions[fn_name]
            contract = self.contracts.get(fn_name)
            summary = self._analyze_function(fn, contract)
            self.summaries[fn_name] = summary
            result.summaries[fn_name] = summary

        # Analyze global code with function summaries available
        if global_stmts:
            global_env, global_warnings = self._analyze_stmts(
                global_stmts, AbstractEnv()
            )
            result.global_env = global_env
            result.global_warnings = global_warnings

        return result

    def _analyze_function(self, fn: FnDecl, contract: Optional[Contract]) -> FunctionSummary:
        """Analyze a single function, producing a FunctionSummary."""
        summary = FunctionSummary(fn_name=fn.name, params=fn.params)

        # Extract bounds from contract
        if contract:
            param_bounds, result_bounds, thresholds = _extract_bounds_from_contract(contract)
            summary.param_bounds = param_bounds
            summary.thresholds = thresholds
            # result_bounds from ensures are the output spec
            summary.result_bounds = result_bounds

        # Set up initial environment from preconditions
        env = AbstractEnv()
        for param in fn.params:
            if param in summary.param_bounds:
                bound = summary.param_bounds[param]
                lo = bound.lower if bound.lower is not None else -float('inf')
                hi = bound.upper if bound.upper is not None else float('inf')
                env.set(param, interval=Interval(lo, hi))
                # Derive sign from interval
                if lo >= 0:
                    env.set(param, sign=Sign.NON_NEG)
                elif hi <= 0:
                    env.set(param, sign=Sign.NON_POS)
                elif lo > 0:
                    env.set(param, sign=Sign.POS)
                elif hi < 0:
                    env.set(param, sign=Sign.NEG)
            else:
                env.set_top(param)

        # Filter out annotation calls from body
        body_stmts = _filter_annotations(fn.body)

        try:
            final_env, warnings = self._analyze_stmts(body_stmts, env)
            summary.body_env = final_env
            summary.warnings = warnings
            summary.analyzed = True

            # Infer result bounds from body analysis if not from contract
            if not summary.result_bounds and final_env:
                # Check for return-like patterns (last assignment or result var)
                for var_name in _get_result_candidates(body_stmts):
                    interval = final_env.get_interval(var_name)
                    sign = final_env.get_sign(var_name)
                    if interval != INTERVAL_TOP or sign != Sign.TOP:
                        bound = AbstractBound(var_name)
                        if interval.lo != -float('inf'):
                            bound.lower = int(interval.lo)
                        if interval.hi != float('inf'):
                            bound.upper = int(interval.hi)
                        if sign != Sign.TOP:
                            bound.sign = sign
                        summary.result_bounds[var_name] = bound
        except Exception as e:
            summary.warnings.append(Warning(
                kind=WarningKind.UNREACHABLE_CODE,
                message=f"Analysis error in {fn.name}: {e}"
            ))

        return summary

    def _analyze_stmts(self, stmts, env: AbstractEnv) -> Tuple[AbstractEnv, List[Warning]]:
        """Analyze a list of statements."""
        warnings = []
        for stmt in stmts:
            env, ws = self._analyze_stmt(stmt, env)
            warnings.extend(ws)
        return env, warnings

    def _analyze_stmt(self, stmt, env: AbstractEnv) -> Tuple[AbstractEnv, List[Warning]]:
        """Analyze a single statement."""
        warnings = []

        if isinstance(stmt, LetDecl):
            val = _eval_abstract(stmt.value, env, self.summaries)
            env.set(stmt.name,
                    sign=val.sign,
                    interval=val.interval,
                    const=val.const)

        elif isinstance(stmt, Assign):
            val = _eval_abstract(stmt.value, env, self.summaries)
            env.set(stmt.name,
                    sign=val.sign,
                    interval=val.interval,
                    const=val.const)

        elif isinstance(stmt, IfStmt):
            env, ws = self._analyze_if(stmt, env)
            warnings.extend(ws)

        elif isinstance(stmt, WhileStmt):
            env, ws = self._analyze_while(stmt, env)
            warnings.extend(ws)

        elif isinstance(stmt, PrintStmt):
            pass  # no effect on abstract state

        elif isinstance(stmt, ReturnStmt):
            pass  # handled by caller

        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                env, ws = self._analyze_stmt(s, env)
                warnings.extend(ws)

        elif isinstance(stmt, CallExpr):
            # Statement-level call (e.g., print, requires, ensures)
            # Apply summary if available
            _apply_call_summary(stmt, env, self.summaries)

        return env, warnings

    def _analyze_if(self, stmt: IfStmt, env: AbstractEnv) -> Tuple[AbstractEnv, List[Warning]]:
        """Analyze if statement with condition refinement."""
        warnings = []
        then_env = env.copy()
        else_env = env.copy()

        # Refine environments based on condition
        _refine_for_condition(stmt.cond, then_env, True)
        _refine_for_condition(stmt.cond, else_env, False)

        # Analyze branches
        then_stmts = stmt.then_body.stmts if hasattr(stmt.then_body, 'stmts') else [stmt.then_body]
        then_env, ws = self._analyze_stmts(then_stmts, then_env)
        warnings.extend(ws)

        if stmt.else_body:
            if hasattr(stmt.else_body, 'stmts'):
                else_stmts = stmt.else_body.stmts
            else:
                else_stmts = [stmt.else_body]
            else_env, ws = self._analyze_stmts(else_stmts, else_env)
            warnings.extend(ws)

        # Join branches
        result = then_env.join(else_env)
        return result, warnings

    def _analyze_while(self, stmt: WhileStmt, env: AbstractEnv) -> Tuple[AbstractEnv, List[Warning]]:
        """Analyze while loop with widening for convergence."""
        warnings = []
        loop_env = env.copy()

        for iteration in range(self.max_iterations):
            # Refine for condition being true (inside loop)
            body_env = loop_env.copy()
            _refine_for_condition(stmt.cond, body_env, True)

            # Analyze body
            body_stmts = stmt.body.stmts if hasattr(stmt.body, 'stmts') else [stmt.body]
            post_body, ws = self._analyze_stmts(body_stmts, body_env)
            if iteration == 0:
                warnings.extend(ws)

            # Join with loop entry
            new_env = loop_env.join(post_body)

            # Check convergence
            if new_env.equals(loop_env):
                break

            # Widen to ensure convergence
            loop_env = loop_env.widen(new_env)

        # After loop: condition is false
        result = loop_env.copy()
        _refine_for_condition(stmt.cond, result, False)
        return result, warnings


# ---------------------------------------------------------------------------
# Abstract expression evaluation with modular summaries
# ---------------------------------------------------------------------------

def _eval_abstract(expr, env: AbstractEnv, summaries: Dict[str, FunctionSummary]) -> AbstractValue:
    """Evaluate expression abstractly, using function summaries for calls."""
    if isinstance(expr, IntLit):
        return AbstractValue.from_value(expr.value)

    elif isinstance(expr, ASTVar):
        sign = env.get_sign(expr.name)
        interval = env.get_interval(expr.name)
        const = env.get_const(expr.name)
        return AbstractValue(sign=sign, interval=interval, const=const)

    elif isinstance(expr, BinOp):
        left = _eval_abstract(expr.left, env, summaries)
        right = _eval_abstract(expr.right, env, summaries)
        return _abstract_binop(expr.op, left, right)

    elif isinstance(expr, UnaryOp):
        operand = _eval_abstract(expr.operand, env, summaries)
        if expr.op == '-':
            from abstract_interpreter import sign_neg, interval_neg
            return AbstractValue(
                sign=sign_neg(operand.sign),
                interval=interval_neg(operand.interval),
                const=operand.const
            )
        return operand

    elif isinstance(expr, CallExpr):
        callee = expr.callee if isinstance(expr.callee, str) else expr.callee.name
        # Skip annotation calls
        if callee in ('requires', 'ensures', 'invariant', 'modifies', 'assert'):
            return AbstractValue.top()
        return _eval_call_abstract(callee, expr.args, env, summaries)

    return AbstractValue.top()


def _eval_call_abstract(callee: str, args, env: AbstractEnv,
                        summaries: Dict[str, FunctionSummary]) -> AbstractValue:
    """Evaluate a function call using its abstract summary."""
    if callee not in summaries:
        return AbstractValue.top()

    summary = summaries[callee]
    if not summary.analyzed or not summary.body_env:
        return AbstractValue.top()

    # Try to use result bounds from the summary
    # If the summary has explicit result bounds (from ensures or inference),
    # return them. Otherwise return TOP.
    if summary.result_bounds:
        # Use the first result bound as the return value abstraction
        for var_name, bound in summary.result_bounds.items():
            lo = bound.lower if bound.lower is not None else -float('inf')
            hi = bound.upper if bound.upper is not None else float('inf')
            sign = bound.sign if bound.sign is not None else Sign.TOP
            return AbstractValue(
                sign=sign,
                interval=Interval(lo, hi),
                const=None
            )

    return AbstractValue.top()


def _apply_call_summary(call: CallExpr, env: AbstractEnv,
                        summaries: Dict[str, FunctionSummary]):
    """Apply a function call's summary to the environment (for statement-level calls)."""
    callee = call.callee if isinstance(call.callee, str) else call.callee.name
    if callee in ('requires', 'ensures', 'invariant', 'modifies', 'assert', 'print'):
        return
    # Statement-level calls don't produce values, but may modify state
    # For now, we don't model side effects on the caller's env


def _abstract_binop(op: str, left: AbstractValue, right: AbstractValue) -> AbstractValue:
    """Compute abstract binary operation."""
    from abstract_interpreter import (
        sign_add, sign_sub, sign_mul, sign_div,
        interval_add, interval_sub, interval_mul, interval_div,
    )

    if op == '+':
        return AbstractValue(
            sign=sign_add(left.sign, right.sign),
            interval=interval_add(left.interval, right.interval),
            const=None
        )
    elif op == '-':
        return AbstractValue(
            sign=sign_sub(left.sign, right.sign),
            interval=interval_sub(left.interval, right.interval),
            const=None
        )
    elif op == '*':
        return AbstractValue(
            sign=sign_mul(left.sign, right.sign),
            interval=interval_mul(left.interval, right.interval),
            const=None
        )
    elif op == '/':
        return AbstractValue(
            sign=sign_div(left.sign, right.sign),
            interval=interval_div(left.interval, right.interval),
            const=None
        )
    elif op in ('==', '!=', '<', '<=', '>', '>='):
        # Comparison: result is 0 or 1
        return AbstractValue(
            sign=Sign.NON_NEG,
            interval=Interval(0, 1),
            const=None
        )
    elif op == 'and':
        return AbstractValue(sign=Sign.NON_NEG, interval=Interval(0, 1), const=None)
    elif op == 'or':
        return AbstractValue(sign=Sign.NON_NEG, interval=Interval(0, 1), const=None)

    return AbstractValue.top()


# ---------------------------------------------------------------------------
# Condition refinement
# ---------------------------------------------------------------------------

def _refine_for_condition(cond, env: AbstractEnv, is_true: bool):
    """Refine environment based on condition truth value."""
    if not isinstance(cond, BinOp):
        return

    # Handle: var op literal
    var_name = None
    literal_val = None
    swapped = False

    if isinstance(cond.left, ASTVar) and isinstance(cond.right, IntLit):
        var_name = cond.left.name
        literal_val = cond.right.value
    elif isinstance(cond.left, IntLit) and isinstance(cond.right, ASTVar):
        var_name = cond.right.name
        literal_val = cond.left.value
        swapped = True

    if var_name is None or literal_val is None:
        return

    op = cond.op
    if swapped:
        # Flip the operator
        flip = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!='}
        op = flip.get(op, op)

    if not is_true:
        # Negate the operator
        negate = {'<': '>=', '>': '<=', '<=': '>', '>=': '<', '==': '!=', '!=': '=='}
        op = negate.get(op, op)

    current = env.get_interval(var_name)

    if op == '<':
        refined = Interval(current.lo, min(current.hi, literal_val - 1))
    elif op == '<=':
        refined = Interval(current.lo, min(current.hi, literal_val))
    elif op == '>':
        refined = Interval(max(current.lo, literal_val + 1), current.hi)
    elif op == '>=':
        refined = Interval(max(current.lo, literal_val), current.hi)
    elif op == '==':
        refined = Interval(literal_val, literal_val)
    elif op == '!=':
        # Can't precisely represent != in intervals, keep as-is
        return
    else:
        return

    env.set(var_name, interval=refined)

    # Update sign based on refined interval
    if refined.lo > 0:
        env.set(var_name, sign=Sign.POS)
    elif refined.hi < 0:
        env.set(var_name, sign=Sign.NEG)
    elif refined.lo == 0 and refined.hi == 0:
        env.set(var_name, sign=Sign.ZERO)
    elif refined.lo >= 0:
        env.set(var_name, sign=Sign.NON_NEG)
    elif refined.hi <= 0:
        env.set(var_name, sign=Sign.NON_POS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_annotations(body) -> List:
    """Filter out requires/ensures/invariant/modifies annotation calls from body."""
    if hasattr(body, 'stmts'):
        stmts = body.stmts
    elif isinstance(body, list):
        stmts = body
    else:
        return [body]

    result = []
    for stmt in stmts:
        if isinstance(stmt, CallExpr):
            callee = stmt.callee if isinstance(stmt.callee, str) else getattr(stmt.callee, 'name', '')
            if callee in ('requires', 'ensures', 'invariant', 'modifies', 'assert'):
                continue
        result.append(stmt)
    return result


def _get_result_candidates(stmts) -> List[str]:
    """Get candidate result variable names from function body."""
    candidates = []
    for stmt in reversed(stmts):
        if isinstance(stmt, ReturnStmt):
            if isinstance(stmt.value, ASTVar):
                candidates.append(stmt.value.name)
            break
        elif isinstance(stmt, Assign):
            candidates.append(stmt.name)
            break
        elif isinstance(stmt, LetDecl):
            candidates.append(stmt.name)
            break
    # Also check for common result variable names
    for stmt in stmts:
        if isinstance(stmt, (LetDecl, Assign)):
            name = stmt.name
            if name in ('result', 'r', 'ret', 'res', 'out', 'output'):
                if name not in candidates:
                    candidates.append(name)
    return candidates


def _build_call_graph(functions: Dict[str, FnDecl]) -> Dict[str, Set[str]]:
    """Build call graph: fn_name -> set of called function names."""
    graph = {name: set() for name in functions}
    for name, fn in functions.items():
        _collect_calls(fn.body, graph[name], set(functions.keys()))
    return graph


def _collect_calls(node, calls: Set[str], known_fns: Set[str]):
    """Collect function calls from an AST node."""
    if isinstance(node, CallExpr):
        callee = node.callee if isinstance(node.callee, str) else getattr(node.callee, 'name', '')
        if callee in known_fns:
            calls.add(callee)
        for arg in node.args:
            _collect_calls(arg, calls, known_fns)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_calls(item, calls, known_fns)
    elif hasattr(node, '__dict__'):
        for val in node.__dict__.values():
            if hasattr(val, '__dict__') or isinstance(val, (list, tuple)):
                _collect_calls(val, calls, known_fns)


def _topological_order(graph: Dict[str, Set[str]], all_nodes: List[str]) -> List[str]:
    """Topological sort (callees before callers). Breaks cycles."""
    visited = set()
    in_stack = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        if node in in_stack:
            return  # cycle, break
        in_stack.add(node)
        for dep in graph.get(node, set()):
            dfs(dep)
        in_stack.discard(node)
        visited.add(node)
        order.append(node)

    for node in all_nodes:
        dfs(node)

    return order  # callees first


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def modular_analyze(source: str, max_iterations: int = 50) -> ModularAIResult:
    """Analyze a C10 program using modular abstract interpretation.

    Each function is analyzed in isolation using its contract as summary.
    """
    mai = ModularAbstractInterpreter(max_iterations=max_iterations)
    return mai.analyze(source)


def analyze_function(source: str, fn_name: str) -> Optional[FunctionSummary]:
    """Analyze a single function and return its summary."""
    result = modular_analyze(source)
    return result.get_summary(fn_name)


def compare_modular_vs_monolithic(source: str) -> Dict:
    """Compare modular analysis results with monolithic (C039) analysis."""
    modular_result = modular_analyze(source)
    monolithic_result = ai_analyze(source)

    comparison = {
        'modular_warnings': modular_result.total_warnings,
        'monolithic_warnings': len(monolithic_result.get('warnings', [])),
        'functions_analyzed': modular_result.functions_analyzed,
        'analysis_order': modular_result.analysis_order,
        'modular_summaries': {},
    }

    mono_env = monolithic_result.get('env')
    for fn_name, summary in modular_result.summaries.items():
        entry = {
            'analyzed': summary.analyzed,
            'warnings': len(summary.warnings),
            'result_bounds': {
                var: {
                    'lower': b.lower,
                    'upper': b.upper,
                    'sign': b.sign.name if b.sign else None,
                }
                for var, b in summary.result_bounds.items()
            }
        }
        comparison['modular_summaries'][fn_name] = entry

    return comparison


def get_function_thresholds(source: str, fn_name: str) -> List[int]:
    """Get widening thresholds extracted from a function's contract."""
    result = modular_analyze(source)
    summary = result.get_summary(fn_name)
    return summary.thresholds if summary else []


def get_all_summaries(source: str) -> Dict[str, FunctionSummary]:
    """Get all function summaries from modular analysis."""
    result = modular_analyze(source)
    return result.summaries
