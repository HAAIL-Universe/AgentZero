"""
V041: Symbolic Debugging
=========================
Given a buggy program and a failing assertion, produce a minimal counterexample
trace and rank statements by likelihood of being the bug.

Composes:
  - C038 (symbolic execution) -- path exploration, assertion checking
  - V028 (fault localization) -- SBFL ranking (Ochiai, Tarantula, DStar)
  - V004 (VCGen) -- WP calculus, SExpr layer
  - C010 (parser) -- AST
  - C037 (SMT solver) -- constraint solving

Pipeline:
  1. Symbolic execution to find ALL assertion-failing paths
  2. Extract minimal counterexample (shortest path to failure)
  3. Spectrum-based fault localization on passing vs failing paths
  4. Backward slice from failure point for dependency analysis
  5. Combined ranking: SBFL score + slice membership + path distance
"""

from __future__ import annotations
import sys, os, math
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto

# --- Path setup ---
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C038_symbolic_execution'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, App, INT, BOOL
from symbolic_execution import (
    SymbolicExecutor, PathStatus, PathState, ExecutionResult,
    AssertionResult, AssertionViolation, symbolic_execute,
)
from vc_gen import parse


# ============================================================
# Debug Trace
# ============================================================

@dataclass
class TraceStep:
    """One step in a counterexample trace."""
    line: int
    stmt_type: str      # "let", "assign", "if_true", "if_false", "while_enter", "while_exit", "return", "assert_fail"
    description: str    # Human-readable description
    env_snapshot: dict = field(default_factory=dict)  # Variable values at this point


@dataclass
class CounterexampleTrace:
    """A concrete counterexample showing how the program reaches a failing assertion."""
    inputs: dict[str, int]           # Concrete input values
    steps: list[TraceStep]           # Execution trace
    assertion_line: int              # Line of the failing assertion
    assertion_expr: str              # The assertion expression
    path_length: int                 # Number of steps (shorter = simpler)

    @property
    def is_minimal(self) -> bool:
        """Whether this is the shortest path to the failure."""
        return self._minimal

    def __repr__(self):
        lines = [f"Counterexample (inputs: {self.inputs}, {self.path_length} steps):"]
        for step in self.steps:
            lines.append(f"  L{step.line}: [{step.stmt_type}] {step.description}")
        return "\n".join(lines)


# ============================================================
# Suspiciousness Ranking
# ============================================================

class SBFLMetric(Enum):
    OCHIAI = "ochiai"
    TARANTULA = "tarantula"
    DSTAR = "dstar"


@dataclass
class SuspiciousStatement:
    """A statement ranked by suspiciousness."""
    line: int
    stmt_type: str
    score: float           # 0.0 (innocent) to 1.0 (very suspicious)
    in_slice: bool         # Whether it's in the backward slice from failure
    metric: SBFLMetric
    description: str = ""

    def __repr__(self):
        slice_tag = " [in slice]" if self.in_slice else ""
        return f"L{self.line}: {self.score:.3f} ({self.metric.value}){slice_tag} -- {self.description}"


# ============================================================
# Debug Result
# ============================================================

@dataclass
class DebugResult:
    """Complete debugging result."""
    has_bug: bool
    counterexamples: list[CounterexampleTrace]
    suspicious: list[SuspiciousStatement]    # Ranked by score descending
    paths_explored: int
    failing_paths: int
    passing_paths: int

    @property
    def minimal_counterexample(self) -> Optional[CounterexampleTrace]:
        if not self.counterexamples:
            return None
        return min(self.counterexamples, key=lambda c: c.path_length)

    @property
    def top_suspects(self) -> list[SuspiciousStatement]:
        """Top 5 most suspicious statements."""
        return self.suspicious[:5]


# ============================================================
# Statement Collector (AST -> line mapping)
# ============================================================

def _collect_stmts(stmts, result=None) -> list[tuple[int, str, str]]:
    """Collect (line, type, description) for all statements in an AST."""
    if result is None:
        result = []
    for stmt in stmts:
        line = getattr(stmt, 'line', 0) or 0
        if isinstance(stmt, LetDecl):
            result.append((line, "let", f"let {stmt.name} = ..."))
        elif isinstance(stmt, Assign):
            result.append((line, "assign", f"{stmt.name} = ..."))
        elif isinstance(stmt, IfStmt):
            result.append((line, "if", f"if (...)"))
            body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            _collect_stmts(body, result)
            if stmt.else_body:
                ebody = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                _collect_stmts(ebody, result)
        elif isinstance(stmt, WhileStmt):
            result.append((line, "while", f"while (...)"))
            body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            _collect_stmts(body, result)
        elif isinstance(stmt, ReturnStmt):
            result.append((line, "return", f"return ..."))
        elif isinstance(stmt, PrintStmt):
            result.append((line, "print", f"print(...)"))
        elif isinstance(stmt, FnDecl):
            result.append((line, "fn", f"fn {stmt.name}(...)"))
            body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            _collect_stmts(body, result)
        elif isinstance(stmt, CallExpr):
            callee = stmt.callee if isinstance(stmt.callee, str) else (
                stmt.callee.name if isinstance(stmt.callee, ASTVar) else str(stmt.callee))
            result.append((line, "call", f"{callee}(...)"))
    return result


# ============================================================
# Backward Slice from Failure
# ============================================================

def _compute_backward_slice(stmts, failure_line: int) -> set[int]:
    """Compute a backward slice: all statements that could affect the failure line.

    Simple approach: collect all lines that define variables used at the failure,
    transitively. This is a lightweight data-dependency slice.
    """
    # Build def-use chains
    defs = {}   # var -> set of lines that define it
    uses = {}   # line -> set of vars used

    def _walk(stmts_list):
        for stmt in stmts_list:
            line = getattr(stmt, 'line', 0) or 0
            if isinstance(stmt, LetDecl):
                defs.setdefault(stmt.name, set()).add(line)
                _collect_uses(stmt.value, line)
            elif isinstance(stmt, Assign):
                defs.setdefault(stmt.name, set()).add(line)
                _collect_uses(stmt.value, line)
            elif isinstance(stmt, IfStmt):
                _collect_uses(stmt.cond, line)
                body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
                _walk(body)
                if stmt.else_body:
                    ebody = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                    _walk(ebody)
            elif isinstance(stmt, WhileStmt):
                _collect_uses(stmt.cond, line)
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                _walk(body)
            elif isinstance(stmt, ReturnStmt):
                if hasattr(stmt, 'value') and stmt.value:
                    _collect_uses(stmt.value, line)
            elif isinstance(stmt, CallExpr):
                for arg in stmt.args:
                    _collect_uses(arg, line)
            elif isinstance(stmt, FnDecl):
                body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                _walk(body)

    def _collect_uses(expr, line):
        if expr is None:
            return
        if isinstance(expr, ASTVar):
            uses.setdefault(line, set()).add(expr.name)
        elif isinstance(expr, BinOp):
            _collect_uses(expr.left, line)
            _collect_uses(expr.right, line)
        elif isinstance(expr, UnaryOp):
            _collect_uses(expr.operand, line)
        elif isinstance(expr, CallExpr):
            for arg in expr.args:
                _collect_uses(arg, line)

    _walk(stmts)

    # Backward slice: start from failure_line, add all defs of used vars
    slice_lines = {failure_line}
    worklist = [failure_line]
    while worklist:
        current = worklist.pop()
        for var in uses.get(current, set()):
            for def_line in defs.get(var, set()):
                if def_line not in slice_lines:
                    slice_lines.add(def_line)
                    worklist.append(def_line)

    return slice_lines


# ============================================================
# SBFL Computation
# ============================================================

def _ochiai(ef: int, ep: int, nf: int, np_: int) -> float:
    """Ochiai suspiciousness: ef / sqrt((ef+ep) * (ef+nf))"""
    denom = math.sqrt((ef + ep) * (ef + nf))
    return ef / denom if denom > 0 else 0.0


def _tarantula(ef: int, ep: int, nf: int, np_: int) -> float:
    """Tarantula suspiciousness."""
    total_f = ef + nf
    total_p = ep + np_
    if total_f == 0:
        return 0.0
    fail_ratio = ef / total_f
    pass_ratio = ep / total_p if total_p > 0 else 0.0
    denom = fail_ratio + pass_ratio
    return fail_ratio / denom if denom > 0 else 0.0


def _dstar(ef: int, ep: int, nf: int, np_: int, star: int = 2) -> float:
    """DStar suspiciousness: ef^* / (ep + nf)"""
    denom = ep + nf
    return (ef ** star) / denom if denom > 0 else float(ef ** star) if ef > 0 else 0.0


# ============================================================
# Symbolic Debugger
# ============================================================

class SymbolicDebugger:
    """Symbolic debugging engine.

    Given a program with assert() calls and symbolic inputs, finds:
    1. Minimal counterexample traces to assertion failures
    2. Suspiciousness ranking of statements (SBFL + slice)
    """

    def __init__(self, max_paths: int = 64, max_loop_unroll: int = 5,
                 metric: SBFLMetric = SBFLMetric.OCHIAI):
        self.max_paths = max_paths
        self.max_loop_unroll = max_loop_unroll
        self.metric = metric

    def debug(self, source: str,
              symbolic_inputs: dict[str, str]) -> DebugResult:
        """Debug a program by symbolic execution + fault localization.

        Args:
            source: C10 source code with assert() calls
            symbolic_inputs: {var_name: 'int'|'bool'} for symbolic inputs

        Returns:
            DebugResult with counterexamples and suspiciousness ranking
        """
        # 1. Symbolic execution
        engine = SymbolicExecutor(
            max_paths=self.max_paths,
            max_loop_unroll=self.max_loop_unroll,
        )
        exec_result = engine.execute(source, symbolic_inputs)

        # 2. Classify paths
        failing_paths = []
        passing_paths = []
        for path in exec_result.paths:
            if path.status == PathStatus.INFEASIBLE:
                continue
            if path.status == PathStatus.ASSERTION_FAILED:
                failing_paths.append(path)
            else:
                passing_paths.append(path)

        if not failing_paths:
            return DebugResult(
                has_bug=False,
                counterexamples=[],
                suspicious=[],
                paths_explored=len(exec_result.paths),
                failing_paths=0,
                passing_paths=len(passing_paths),
            )

        # 3. Generate counterexamples
        counterexamples = []
        for path in failing_paths:
            ce = self._extract_counterexample(path, symbolic_inputs, source)
            if ce is not None:
                counterexamples.append(ce)

        # 4. SBFL ranking
        program = parse(source)
        all_stmts = _collect_stmts(program.stmts)

        # Compute coverage spectrum
        suspicious = self._compute_sbfl(
            all_stmts, failing_paths, passing_paths, program.stmts,
            failing_paths[0].assertion_line if failing_paths else 0,
        )

        return DebugResult(
            has_bug=True,
            counterexamples=counterexamples,
            suspicious=suspicious,
            paths_explored=len(exec_result.paths),
            failing_paths=len(failing_paths),
            passing_paths=len(passing_paths),
        )

    def _extract_counterexample(self, path: PathState,
                                symbolic_inputs: dict[str, str],
                                source: str) -> Optional[CounterexampleTrace]:
        """Extract a concrete counterexample from a failing path."""
        # Solve constraints to get concrete inputs
        solver = SMTSolver()
        # Declare symbolic input vars
        for name, sort in symbolic_inputs.items():
            if sort == 'int':
                solver.Int(name)
            elif sort == 'bool':
                solver.Bool(name)
        # Declare any other vars in constraints
        self._declare_constraint_vars(solver, path.constraints)

        for c in path.constraints:
            solver.add(c)

        result = solver.check()
        if result != SMTResult.SAT:
            return None

        model = solver.model()
        inputs = {name: model.get(name, 0) for name in symbolic_inputs}

        # Build trace from covered lines
        steps = []
        for line in sorted(path.covered_lines):
            steps.append(TraceStep(
                line=line,
                stmt_type="exec",
                description=f"line {line}",
            ))

        # Add assertion failure step
        steps.append(TraceStep(
            line=path.assertion_line,
            stmt_type="assert_fail",
            description=f"Assertion failed at line {path.assertion_line}",
        ))

        return CounterexampleTrace(
            inputs=inputs,
            steps=steps,
            assertion_line=path.assertion_line,
            assertion_expr=path.error_msg or "assert(...)",
            path_length=len(steps),
        )

    def _declare_constraint_vars(self, solver: SMTSolver, constraints: list):
        """Declare all variables that appear in constraints."""
        seen = set()
        for c in constraints:
            self._collect_vars(c, seen)
        for name, sort in seen:
            if sort == 'int':
                solver.Int(name)
            elif sort == 'bool':
                solver.Bool(name)

    def _collect_vars(self, term, seen: set):
        if isinstance(term, SMTVar):
            sort = 'bool' if term.sort == BOOL else 'int'
            seen.add((term.name, sort))
        elif isinstance(term, App):
            for child in term.args:
                self._collect_vars(child, seen)

    def _compute_sbfl(self, all_stmts: list[tuple[int, str, str]],
                      failing_paths: list, passing_paths: list,
                      ast_stmts: list, failure_line: int) -> list[SuspiciousStatement]:
        """Compute SBFL suspiciousness for all statements."""
        # Compute backward slice
        slice_lines = _compute_backward_slice(ast_stmts, failure_line)

        total_fail = len(failing_paths)
        total_pass = len(passing_paths)

        results = []
        seen_lines = set()
        for line, stype, desc in all_stmts:
            if line in seen_lines or line == 0:
                continue
            seen_lines.add(line)

            # Count coverage
            ef = sum(1 for p in failing_paths if line in p.covered_lines)
            ep = sum(1 for p in passing_paths if line in p.covered_lines)
            nf = total_fail - ef
            np_ = total_pass - ep

            # Compute score
            if self.metric == SBFLMetric.OCHIAI:
                score = _ochiai(ef, ep, nf, np_)
            elif self.metric == SBFLMetric.TARANTULA:
                score = _tarantula(ef, ep, nf, np_)
            else:
                score = _dstar(ef, ep, nf, np_)

            in_slice = line in slice_lines

            # Boost score for slice membership
            if in_slice:
                score = min(1.0, score * 1.2)

            results.append(SuspiciousStatement(
                line=line,
                stmt_type=stype,
                score=score,
                in_slice=in_slice,
                metric=self.metric,
                description=desc,
            ))

        # Sort by score descending, slice membership as tiebreaker
        results.sort(key=lambda s: (-s.score, not s.in_slice, s.line))
        return results


# ============================================================
# High-Level API
# ============================================================

def symbolic_debug(source: str, symbolic_inputs: dict[str, str],
                   metric: SBFLMetric = SBFLMetric.OCHIAI,
                   max_paths: int = 64) -> DebugResult:
    """Debug a C10 program using symbolic execution + fault localization.

    Args:
        source: C10 source with assert() calls
        symbolic_inputs: {var_name: 'int'|'bool'}
        metric: SBFL metric to use
        max_paths: Maximum paths to explore

    Returns:
        DebugResult with counterexamples and suspiciousness ranking
    """
    debugger = SymbolicDebugger(max_paths=max_paths, metric=metric)
    return debugger.debug(source, symbolic_inputs)


def find_minimal_counterexample(source: str, symbolic_inputs: dict[str, str],
                                max_paths: int = 64) -> Optional[CounterexampleTrace]:
    """Find the shortest counterexample trace for a failing assertion."""
    result = symbolic_debug(source, symbolic_inputs, max_paths=max_paths)
    return result.minimal_counterexample
