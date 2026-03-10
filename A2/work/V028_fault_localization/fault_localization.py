"""
V028: Fault Localization
========================
Given a buggy program and test cases (some passing, some failing),
identify the most likely buggy statement(s).

Composes:
  - C038 (symbolic execution) -- path exploration, coverage, test generation
  - C010 (parser/AST) -- statement identification
  - C037 (SMT solver) -- constraint solving for symbolic analysis

Techniques:
  1. Spectrum-Based Fault Localization (SBFL) -- Ochiai, Tarantula, DStar
  2. Backward dependency slicing from failure point
  3. Combined ranking with auto test generation pipeline
"""

import sys
import os
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any

# --- Path setup ---
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C038_symbolic_execution'))

from stack_vm import lex, Parser, Program, LetDecl, Assign, IfStmt, WhileStmt, \
    FnDecl, CallExpr, ReturnStmt, PrintStmt, Block, BinOp, UnaryOp, Var, IntLit
from symbolic_execution import SymbolicExecutor, PathStatus


# ============================================================
# Data Structures
# ============================================================

class Metric(Enum):
    OCHIAI = "ochiai"
    TARANTULA = "tarantula"
    DSTAR = "dstar"


class TestVerdict(Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class TestCase:
    """A test case with concrete inputs and expected behavior."""
    inputs: Dict[str, int]
    verdict: TestVerdict
    description: str = ""


@dataclass
class StatementInfo:
    """Information about a single statement in the program."""
    index: int          # 0-based statement index in flattened list
    line: int           # 1-based line number in source
    kind: str           # e.g. "let", "assign", "if", "while", "print", "return", "call"
    description: str    # human-readable description
    ast_node: Any = field(default=None, repr=False)


@dataclass
class SuspiciousnessScore:
    """Suspiciousness score for a statement under a given metric."""
    statement: StatementInfo
    score: float
    metric: Metric
    # Coverage counts
    executed_by_failing: int = 0
    executed_by_passing: int = 0
    not_executed_by_failing: int = 0
    not_executed_by_passing: int = 0


@dataclass
class SpectrumResult:
    """Result of spectrum-based fault localization."""
    rankings: Dict[Metric, List[SuspiciousnessScore]]
    statements: List[StatementInfo]
    total_passing: int
    total_failing: int
    coverage_matrix: Dict[int, Dict[int, bool]]  # test_idx -> stmt_idx -> covered


@dataclass
class SliceResult:
    """Result of backward dependency slicing."""
    relevant_statements: List[StatementInfo]
    dependency_chain: List[Tuple[int, str]]  # (stmt_idx, reason)


@dataclass
class FaultResult:
    """Combined fault localization result."""
    ranked_statements: List[SuspiciousnessScore]  # Combined ranking
    spectrum: Optional[SpectrumResult]
    slice_info: Optional[SliceResult]
    auto_generated_tests: int
    total_tests: int
    top_suspect: Optional[StatementInfo]


# ============================================================
# AST Utilities
# ============================================================

def _parse(source: str) -> Program:
    tokens = lex(source)
    return Parser(tokens).parse()


def _flatten_statements(program: Program) -> List[Tuple[Any, int]]:
    """Flatten all statements in program to a list with nesting depth."""
    result = []

    def walk(stmts, depth=0):
        for stmt in stmts:
            result.append((stmt, depth))
            if isinstance(stmt, IfStmt):
                walk(stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body], depth + 1)
                if stmt.else_body:
                    else_stmts = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                    walk(else_stmts, depth + 1)
            elif isinstance(stmt, WhileStmt):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                walk(body, depth + 1)
            elif isinstance(stmt, FnDecl):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                walk(body, depth + 1)

    walk(program.stmts)
    return result


def _stmt_kind(stmt) -> str:
    if isinstance(stmt, LetDecl):
        return "let"
    elif isinstance(stmt, Assign):
        return "assign"
    elif isinstance(stmt, IfStmt):
        return "if"
    elif isinstance(stmt, WhileStmt):
        return "while"
    elif isinstance(stmt, FnDecl):
        return "fn"
    elif isinstance(stmt, ReturnStmt):
        return "return"
    elif isinstance(stmt, PrintStmt):
        return "print"
    elif isinstance(stmt, CallExpr):
        return "call"
    else:
        return type(stmt).__name__.lower()


def _stmt_description(stmt) -> str:
    kind = _stmt_kind(stmt)
    if isinstance(stmt, LetDecl):
        return f"let {stmt.name} = ..."
    elif isinstance(stmt, Assign):
        return f"{stmt.name} = ..."
    elif isinstance(stmt, IfStmt):
        return f"if ..."
    elif isinstance(stmt, WhileStmt):
        return f"while ..."
    elif isinstance(stmt, FnDecl):
        return f"fn {stmt.name}(...)"
    elif isinstance(stmt, ReturnStmt):
        return f"return ..."
    elif isinstance(stmt, PrintStmt):
        return f"print ..."
    elif isinstance(stmt, CallExpr):
        return f"{stmt.callee}(...)"
    return kind


def _extract_statements(source: str) -> List[StatementInfo]:
    """Extract all statements with line info from source."""
    program = _parse(source)
    flat = _flatten_statements(program)
    statements = []
    for idx, (stmt, depth) in enumerate(flat):
        # Estimate line from position in flattened list
        # Use source line counting heuristic
        line = _estimate_line(source, stmt, idx)
        statements.append(StatementInfo(
            index=idx,
            line=line,
            kind=_stmt_kind(stmt),
            description=_stmt_description(stmt),
            ast_node=stmt
        ))
    return statements


def _estimate_line(source: str, stmt, index: int) -> int:
    """Estimate line number for a statement. Heuristic based on index."""
    lines = source.strip().split('\n')
    # Simple heuristic: map statement index to non-empty, non-brace lines
    code_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped and stripped not in ('{', '}', ''):
            code_lines.append(i)
    if index < len(code_lines):
        return code_lines[index]
    return index + 1


# ============================================================
# Concrete Execution for Coverage Tracking
# ============================================================

class CoverageTracer:
    """Execute a program concretely and track which statements are covered.

    Uses the AST nodes from _extract_statements directly (same object identity)
    so that coverage tracking maps correctly to statement indices.
    """

    def __init__(self, source: str, statements: List[StatementInfo]):
        self.source = source
        self.statements = statements
        # Build a flat list of AST nodes from statements (same objects)
        self._stmt_nodes = {id(s.ast_node): s.index for s in statements if s.ast_node is not None}
        # We need the program's top-level stmts -- extract from statements
        # Re-parse to get the program structure, but then map via structural matching
        self.program = _parse(source)
        # Build mapping from program's AST nodes to statement indices
        self._node_map = {}
        self._build_node_map(self.program.stmts, statements, 0)

    def _build_node_map(self, ast_stmts, info_stmts, start_idx):
        """Map AST nodes from parsed program to statement info indices.
        Uses sequential order matching since both come from the same source."""
        flat_ast = []
        self._flatten_ast(ast_stmts, flat_ast)
        for i, node in enumerate(flat_ast):
            if start_idx + i < len(info_stmts):
                self._node_map[id(node)] = info_stmts[start_idx + i].index

    def _flatten_ast(self, stmts, result):
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, IfStmt):
                then_body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
                self._flatten_ast(then_body, result)
                if stmt.else_body:
                    else_body = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                    self._flatten_ast(else_body, result)
            elif isinstance(stmt, WhileStmt):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                self._flatten_ast(body, result)
            elif isinstance(stmt, FnDecl):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                self._flatten_ast(body, result)

    def execute(self, inputs: Dict[str, int]) -> Tuple[Set[int], bool, Any]:
        """Execute with given inputs. Returns (covered_stmt_indices, passed, output)."""
        covered = set()
        env = dict(inputs)
        output = []
        passed = True
        error = None

        try:
            self._exec_stmts(self.program.stmts, env, covered, output)
        except AssertionError as e:
            passed = False
            error = str(e)
        except Exception as e:
            passed = False
            error = str(e)

        return covered, passed, output

    def _exec_stmts(self, stmts, env, covered, output):
        for stmt in stmts:
            self._exec_stmt(stmt, env, covered, output)

    def _exec_stmt(self, stmt, env, covered, output):
        node_id = id(stmt)
        if node_id in self._node_map:
            covered.add(self._node_map[node_id])

        if isinstance(stmt, LetDecl):
            env[stmt.name] = self._eval_expr(stmt.value, env)
        elif isinstance(stmt, Assign):
            env[stmt.name] = self._eval_expr(stmt.value, env)
        elif isinstance(stmt, IfStmt):
            cond = self._eval_expr(stmt.cond, env)
            if cond:
                body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
                self._exec_stmts(body, env, covered, output)
            elif stmt.else_body:
                body = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                self._exec_stmts(body, env, covered, output)
        elif isinstance(stmt, WhileStmt):
            max_iter = 1000
            i = 0
            while self._eval_expr(stmt.cond, env) and i < max_iter:
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                self._exec_stmts(body, env, covered, output)
                i += 1
                # Re-mark the while as covered each iteration
                if node_id in self._node_map:
                    covered.add(self._node_map[node_id])
        elif isinstance(stmt, PrintStmt):
            val = self._eval_expr(stmt.value, env)
            output.append(val)
        elif isinstance(stmt, ReturnStmt):
            pass  # Simplified -- no function calls in tracer
        elif isinstance(stmt, FnDecl):
            env[stmt.name] = stmt  # Store function
        elif isinstance(stmt, CallExpr):
            self._eval_expr(stmt, env)

    def _eval_expr(self, expr, env):
        if isinstance(expr, IntLit):
            return expr.value
        elif isinstance(expr, Var):
            return env.get(expr.name, 0)
        elif isinstance(expr, BinOp):
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            op = expr.op
            if op == '+': return left + right
            elif op == '-': return left - right
            elif op == '*': return left * right
            elif op == '/':
                if right == 0:
                    raise Exception("division by zero")
                return left // right
            elif op == '%':
                if right == 0:
                    raise Exception("modulo by zero")
                return left % right
            elif op == '<': return int(left < right)
            elif op == '>': return int(left > right)
            elif op == '<=': return int(left <= right)
            elif op == '>=': return int(left >= right)
            elif op == '==': return int(left == right)
            elif op == '!=': return int(left != right)
            elif op == '&&': return int(bool(left) and bool(right))
            elif op == '||': return int(bool(left) or bool(right))
            return 0
        elif isinstance(expr, UnaryOp):
            val = self._eval_expr(expr.operand, env)
            if expr.op == '-': return -val
            if expr.op == '!': return int(not val)
            return val
        elif isinstance(expr, CallExpr):
            fn = env.get(expr.callee)
            if isinstance(fn, FnDecl):
                args = [self._eval_expr(a, env) for a in expr.args]
                local_env = dict(env)
                for param, arg in zip(fn.params, args):
                    local_env[param] = arg
                body = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
                out = []
                self._exec_stmts(body, local_env, set(), out)
                return local_env.get('__return__', 0)
            elif callable(fn):
                args = [self._eval_expr(a, env) for a in expr.args]
                return fn(*args)
            # Built-in assert
            if expr.callee == 'assert':
                args = [self._eval_expr(a, env) for a in expr.args]
                if not args[0]:
                    raise AssertionError(f"assertion failed")
                return 0
            return 0
        return 0


# ============================================================
# Spectrum-Based Fault Localization (SBFL)
# ============================================================

def _compute_suspiciousness(ef: int, ep: int, nf: int, np_: int,
                            total_f: int, total_p: int,
                            metric: Metric) -> float:
    """Compute suspiciousness score for a statement.

    ef = executed by failing tests
    ep = executed by passing tests
    nf = NOT executed by failing tests
    np_ = NOT executed by passing tests
    """
    if metric == Metric.OCHIAI:
        denom = math.sqrt(total_f * (ef + ep))
        if denom == 0:
            return 0.0
        return ef / denom

    elif metric == Metric.TARANTULA:
        if total_f == 0:
            return 0.0
        fail_ratio = ef / total_f if total_f > 0 else 0
        pass_ratio = ep / total_p if total_p > 0 else 0
        denom = fail_ratio + pass_ratio
        if denom == 0:
            return 0.0
        return fail_ratio / denom

    elif metric == Metric.DSTAR:
        # DStar with * = 2
        denom = ep + nf
        if denom == 0:
            return float('inf') if ef > 0 else 0.0
        return (ef * ef) / denom

    return 0.0


def spectrum_localize(source: str, tests: List[TestCase],
                      metrics: Optional[List[Metric]] = None) -> SpectrumResult:
    """Run spectrum-based fault localization on a set of test cases.

    Args:
        source: Program source code
        tests: List of test cases with inputs and verdicts
        metrics: Which metrics to compute (default: all three)

    Returns:
        SpectrumResult with rankings per metric
    """
    if metrics is None:
        metrics = [Metric.OCHIAI, Metric.TARANTULA, Metric.DSTAR]

    statements = _extract_statements(source)
    tracer = CoverageTracer(source, statements)

    # Build coverage matrix
    coverage_matrix = {}  # test_idx -> stmt_idx -> covered
    test_verdicts = []

    for t_idx, test in enumerate(tests):
        covered, passed, output = tracer.execute(test.inputs)
        # Use the test's declared verdict (it may declare fail for expected-fail)
        coverage_matrix[t_idx] = {s.index: (s.index in covered) for s in statements}
        test_verdicts.append(test.verdict)

    total_f = sum(1 for v in test_verdicts if v == TestVerdict.FAIL)
    total_p = sum(1 for v in test_verdicts if v == TestVerdict.PASS)

    # Compute per-statement counts
    rankings = {}
    for metric in metrics:
        scores = []
        for stmt in statements:
            ef = sum(1 for t_idx, v in enumerate(test_verdicts)
                     if v == TestVerdict.FAIL and coverage_matrix[t_idx].get(stmt.index, False))
            ep = sum(1 for t_idx, v in enumerate(test_verdicts)
                     if v == TestVerdict.PASS and coverage_matrix[t_idx].get(stmt.index, False))
            nf = total_f - ef
            np_ = total_p - ep

            score = _compute_suspiciousness(ef, ep, nf, np_, total_f, total_p, metric)
            scores.append(SuspiciousnessScore(
                statement=stmt,
                score=score,
                metric=metric,
                executed_by_failing=ef,
                executed_by_passing=ep,
                not_executed_by_failing=nf,
                not_executed_by_passing=np_
            ))

        # Sort by score descending, then by statement index ascending (tie-break)
        scores.sort(key=lambda s: (-s.score, s.statement.index))
        rankings[metric] = scores

    return SpectrumResult(
        rankings=rankings,
        statements=statements,
        total_passing=total_p,
        total_failing=total_f,
        coverage_matrix=coverage_matrix
    )


# ============================================================
# Backward Dependency Slicing
# ============================================================

def _collect_vars_in_expr(expr) -> Set[str]:
    """Collect all variable names referenced in an expression."""
    result = set()
    if isinstance(expr, Var):
        result.add(expr.name)
    elif isinstance(expr, BinOp):
        result |= _collect_vars_in_expr(expr.left)
        result |= _collect_vars_in_expr(expr.right)
    elif isinstance(expr, UnaryOp):
        result |= _collect_vars_in_expr(expr.operand)
    elif isinstance(expr, CallExpr):
        for arg in expr.args:
            result |= _collect_vars_in_expr(arg)
    return result


def _defined_var(stmt) -> Optional[str]:
    """Return the variable defined by this statement, if any."""
    if isinstance(stmt, LetDecl):
        return stmt.name
    elif isinstance(stmt, Assign):
        return stmt.name
    return None


def _used_vars(stmt) -> Set[str]:
    """Return variables used (read) by this statement."""
    result = set()
    if isinstance(stmt, LetDecl):
        result = _collect_vars_in_expr(stmt.value)
    elif isinstance(stmt, Assign):
        result = _collect_vars_in_expr(stmt.value)
    elif isinstance(stmt, IfStmt):
        result = _collect_vars_in_expr(stmt.cond)
    elif isinstance(stmt, WhileStmt):
        result = _collect_vars_in_expr(stmt.cond)
    elif isinstance(stmt, PrintStmt):
        result = _collect_vars_in_expr(stmt.value)
    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            result = _collect_vars_in_expr(stmt.value)
    elif isinstance(stmt, CallExpr):
        for arg in stmt.args:
            result |= _collect_vars_in_expr(arg)
    return result


def backward_slice(source: str, target_var: Optional[str] = None,
                   target_stmt_index: Optional[int] = None) -> SliceResult:
    """Compute backward dependency slice from a target variable or statement.

    If target_var is given, find all statements that influence that variable.
    If target_stmt_index is given, slice from that statement.
    If neither, slice from the last statement.
    """
    statements = _extract_statements(source)
    if not statements:
        return SliceResult(relevant_statements=[], dependency_chain=[])

    # Determine starting point
    if target_stmt_index is not None:
        start_idx = target_stmt_index
    elif target_var is not None:
        # Find last assignment to target_var
        start_idx = len(statements) - 1
        for s in reversed(statements):
            if _defined_var(s.ast_node) == target_var:
                start_idx = s.index
                break
    else:
        start_idx = len(statements) - 1

    # Build nesting structure: which control-flow statement contains each statement
    program = _parse(source)
    parent_map = {}  # stmt_index -> parent_stmt_index (for control flow)
    _build_parent_map(program.stmts, statements, None, parent_map)

    # Backward slice via worklist
    relevant = set()
    chain = []
    worklist = set()

    start_stmt = statements[start_idx]
    relevant.add(start_idx)
    chain.append((start_idx, "target"))
    worklist = _used_vars(start_stmt.ast_node)

    # Also include control-flow parents of the target
    _add_control_parents(start_idx, parent_map, relevant, chain, statements, worklist)

    # Walk backward through statements
    for idx in range(start_idx - 1, -1, -1):
        stmt = statements[idx]
        defined = _defined_var(stmt.ast_node)
        if defined and defined in worklist:
            relevant.add(idx)
            chain.append((idx, f"defines {defined}"))
            worklist.discard(defined)
            worklist |= _used_vars(stmt.ast_node)
            # Include control-flow parents
            _add_control_parents(idx, parent_map, relevant, chain, statements, worklist)
        elif isinstance(stmt.ast_node, (IfStmt, WhileStmt)):
            # Control flow statements that might affect execution
            cond_vars = _used_vars(stmt.ast_node)
            if cond_vars & worklist:
                relevant.add(idx)
                chain.append((idx, "control dependency"))
                worklist |= cond_vars

    relevant_stmts = [s for s in statements if s.index in relevant]
    return SliceResult(
        relevant_statements=relevant_stmts,
        dependency_chain=chain
    )


def _build_parent_map(ast_stmts, statements, parent_idx, parent_map):
    """Build a map from statement index to its enclosing control-flow statement index."""
    flat_iter = iter(range(len(statements)))

    def walk(stmts, parent):
        for stmt in stmts:
            try:
                idx = next(flat_iter)
            except StopIteration:
                return
            if parent is not None:
                parent_map[idx] = parent
            if isinstance(stmt, IfStmt):
                then_body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
                walk(then_body, idx)
                if stmt.else_body:
                    else_body = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                    walk(else_body, idx)
            elif isinstance(stmt, WhileStmt):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                walk(body, idx)
            elif isinstance(stmt, FnDecl):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                walk(body, idx)

    walk(ast_stmts, parent_idx)


def _add_control_parents(idx, parent_map, relevant, chain, statements, worklist):
    """Add enclosing control-flow statements to the slice."""
    current = idx
    while current in parent_map:
        parent = parent_map[current]
        if parent not in relevant and parent < len(statements):
            stmt = statements[parent]
            if isinstance(stmt.ast_node, (IfStmt, WhileStmt)):
                relevant.add(parent)
                chain.append((parent, "control parent"))
                worklist |= _used_vars(stmt.ast_node)
        current = parent


# ============================================================
# Symbolic Fault Localization
# ============================================================

def symbolic_localize(source: str, failing_inputs: Dict[str, int],
                      symbolic_vars: Dict[str, str],
                      max_paths: int = 64) -> SliceResult:
    """Use symbolic execution to identify statements contributing to failure.

    Runs symbolic execution, finds the path matching the failing inputs,
    and performs backward slicing on the constraints of that path.
    """
    executor = SymbolicExecutor(max_paths=max_paths)
    result = executor.execute(source, symbolic_vars)

    # Find the path that matches the failing inputs (by constraint satisfaction)
    # Use backward slice as fallback since exact path matching needs SMT
    statements = _extract_statements(source)

    # Identify which statements the failing path covers
    tracer = CoverageTracer(source, statements)
    covered, passed, output = tracer.execute(failing_inputs)

    # Find assertion or error point
    target_idx = None
    for idx in sorted(covered, reverse=True):
        stmt = statements[idx] if idx < len(statements) else None
        if stmt and stmt.kind in ('call', 'print', 'return'):
            # Check if it's an assert call
            if isinstance(stmt.ast_node, CallExpr) and stmt.ast_node.callee == 'assert':
                target_idx = idx
                break

    # If no assert found, use the last covered statement
    if target_idx is None and covered:
        target_idx = max(covered)

    if target_idx is None:
        return SliceResult(relevant_statements=[], dependency_chain=[])

    # Backward slice from the failure point, restricted to covered statements
    slice_result = backward_slice(source, target_stmt_index=target_idx)

    # Filter to only statements that were actually executed
    relevant = [s for s in slice_result.relevant_statements if s.index in covered]
    chain = [(idx, reason) for idx, reason in slice_result.dependency_chain if idx in covered]

    return SliceResult(relevant_statements=relevant, dependency_chain=chain)


# ============================================================
# Auto Test Generation
# ============================================================

def generate_test_suite(source: str, symbolic_vars: Dict[str, str],
                        oracle_fn=None,
                        max_paths: int = 64) -> List[TestCase]:
    """Generate a test suite using symbolic execution.

    Args:
        source: Program source code
        symbolic_vars: Variables to make symbolic {'name': 'int'}
        oracle_fn: Function(inputs) -> TestVerdict to classify tests.
                   If None, uses assertion-based: assertion failure = FAIL.

    Returns:
        List of TestCase objects with pass/fail verdicts
    """
    executor = SymbolicExecutor(max_paths=max_paths)
    result = executor.execute(source, symbolic_vars)

    tests = []
    seen_inputs = set()

    for tc in result.test_cases:
        if tc.path_status == PathStatus.INFEASIBLE:
            continue

        # Normalize inputs to int
        inputs = {}
        for name, val in tc.inputs.items():
            if name in symbolic_vars:
                inputs[name] = int(val) if val is not None else 0

        # Dedup
        key = tuple(sorted(inputs.items()))
        if key in seen_inputs:
            continue
        seen_inputs.add(key)

        # Determine verdict
        if oracle_fn:
            verdict = oracle_fn(inputs)
        else:
            # Execute concretely to check for assertion failures
            statements = _extract_statements(source)
            tracer = CoverageTracer(source, statements)
            _, passed, _ = tracer.execute(inputs)
            verdict = TestVerdict.PASS if passed else TestVerdict.FAIL

        tests.append(TestCase(inputs=inputs, verdict=verdict))

    return tests


# ============================================================
# Combined Fault Localization
# ============================================================

def localize_fault(source: str,
                   failing_tests: List[Dict[str, int]],
                   passing_tests: List[Dict[str, int]],
                   metric: Metric = Metric.OCHIAI) -> FaultResult:
    """Localize fault given explicit passing and failing test inputs.

    Args:
        source: Program source
        failing_tests: List of input dicts that trigger the bug
        passing_tests: List of input dicts that pass correctly
        metric: SBFL metric to use for primary ranking

    Returns:
        FaultResult with combined rankings
    """
    tests = []
    for inputs in failing_tests:
        tests.append(TestCase(inputs=inputs, verdict=TestVerdict.FAIL))
    for inputs in passing_tests:
        tests.append(TestCase(inputs=inputs, verdict=TestVerdict.PASS))

    # Spectrum-based
    spectrum = spectrum_localize(source, tests, metrics=[metric])

    # Backward slice from first failing test
    slice_info = None
    if failing_tests:
        slice_info = backward_slice(source)

    # Primary ranking from spectrum
    ranked = spectrum.rankings[metric]

    top = ranked[0].statement if ranked and ranked[0].score > 0 else None

    return FaultResult(
        ranked_statements=ranked,
        spectrum=spectrum,
        slice_info=slice_info,
        auto_generated_tests=0,
        total_tests=len(tests),
        top_suspect=top
    )


def auto_localize(source: str,
                  symbolic_vars: Dict[str, str],
                  oracle_fn=None,
                  metric: Metric = Metric.OCHIAI,
                  max_paths: int = 64,
                  extra_tests: Optional[List[TestCase]] = None) -> FaultResult:
    """Fully automatic fault localization pipeline.

    1. Generate tests via symbolic execution
    2. Classify as pass/fail (oracle or assertion-based)
    3. Run SBFL + backward slicing
    4. Combine and rank

    Args:
        source: Buggy program source
        symbolic_vars: Variables to explore symbolically
        oracle_fn: Optional oracle(inputs) -> TestVerdict
        metric: SBFL metric
        max_paths: Max symbolic execution paths
        extra_tests: Additional test cases to include

    Returns:
        FaultResult with full analysis
    """
    # Generate tests
    tests = generate_test_suite(source, symbolic_vars, oracle_fn, max_paths)

    # Add extra tests
    if extra_tests:
        tests.extend(extra_tests)

    if not tests:
        statements = _extract_statements(source)
        return FaultResult(
            ranked_statements=[],
            spectrum=None,
            slice_info=None,
            auto_generated_tests=0,
            total_tests=0,
            top_suspect=None
        )

    auto_count = len(tests) - (len(extra_tests) if extra_tests else 0)

    # Need at least one failing test
    failing = [t for t in tests if t.verdict == TestVerdict.FAIL]
    passing = [t for t in tests if t.verdict == TestVerdict.PASS]

    if not failing:
        # No failures found -- program may not be buggy for explored paths
        statements = _extract_statements(source)
        return FaultResult(
            ranked_statements=[],
            spectrum=None,
            slice_info=None,
            auto_generated_tests=auto_count,
            total_tests=len(tests),
            top_suspect=None
        )

    # SBFL
    all_metrics = [Metric.OCHIAI, Metric.TARANTULA, Metric.DSTAR]
    spectrum = spectrum_localize(source, tests, metrics=all_metrics)

    # Backward slice from first failing input
    slice_info = backward_slice(source)

    # Combined ranking: boost statements that appear in both SBFL top and slice
    primary_ranking = list(spectrum.rankings[metric])
    slice_indices = set()
    if slice_info:
        slice_indices = {s.index for s in slice_info.relevant_statements}

    # Boost: multiply score by 1.5 for statements in the backward slice
    for score_obj in primary_ranking:
        if score_obj.statement.index in slice_indices:
            score_obj.score *= 1.5

    # Re-sort
    primary_ranking.sort(key=lambda s: (-s.score, s.statement.index))

    top = primary_ranking[0].statement if primary_ranking and primary_ranking[0].score > 0 else None

    return FaultResult(
        ranked_statements=primary_ranking,
        spectrum=spectrum,
        slice_info=slice_info,
        auto_generated_tests=auto_count,
        total_tests=len(tests),
        top_suspect=top
    )


# ============================================================
# Convenience: Rank-at-N and Exam Score
# ============================================================

def rank_at(result: FaultResult, buggy_stmt_index: int) -> Optional[int]:
    """Find the rank (1-based) of the buggy statement in the result."""
    for rank, score in enumerate(result.ranked_statements, 1):
        if score.statement.index == buggy_stmt_index:
            return rank
    return None


def exam_score(result: FaultResult, buggy_stmt_index: int) -> Optional[float]:
    """Compute EXAM score: fraction of statements examined before finding the bug.
    Lower is better. Range [0, 1].
    """
    r = rank_at(result, buggy_stmt_index)
    if r is None:
        return None
    total = len(result.ranked_statements)
    if total == 0:
        return None
    return r / total
