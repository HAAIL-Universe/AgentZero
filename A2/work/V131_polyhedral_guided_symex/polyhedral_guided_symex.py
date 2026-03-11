"""V131: Polyhedral-Guided Symbolic Execution.

Composes V105 (polyhedral abstract interpretation) + C038 (symbolic execution).

Like V001 (interval-guided), but uses the polyhedral domain to prune infeasible
symbolic execution paths. Polyhedral constraints capture RELATIONAL invariants
(e.g., x + y <= 10) that interval analysis cannot express, enabling more precise
pruning when variables are correlated.

Pipeline:
  Phase 1: Polyhedral pre-analysis
    - Run PolyhedralInterpreter on source
    - At each branch point, record polyhedral feasibility
  Phase 2: Guided symbolic execution
    - Override _check_feasible to consult polyhedral info first
    - If polyhedral analysis says infeasible -> skip SMT (prune)
    - Otherwise -> fall back to SMT solver
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))

from polyhedral_domain import PolyhedralInterpreter, PolyhedralDomain, LinearConstraint
from symbolic_execution import (
    SymbolicExecutor, SymValue, PathState, PathStatus, SymType,
    ExecutionResult, TestCase, AssertionResult, AssertionViolation,
    CoverageResult,
    symbolic_execute, generate_tests, check_assertions,
    smt_not, smt_and, smt_or,
)
from stack_vm import (
    lex, Parser, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)
from smt_solver import SMTSolver, SMTResult


# ---------------------------------------------------------------------------
# Branch feasibility info from polyhedral analysis
# ---------------------------------------------------------------------------

@dataclass
class PolyBranchInfo:
    """Polyhedral feasibility at a branch point."""
    line: int
    true_feasible: bool = True
    false_feasible: bool = True


# ---------------------------------------------------------------------------
# Polyhedral pre-analyzer (captures branch info)
# ---------------------------------------------------------------------------

class BranchCapturingInterpreter(PolyhedralInterpreter):
    """PolyhedralInterpreter subclass that records branch feasibility."""

    def __init__(self, max_iterations=50, branch_info=None, symbolic_inputs=None):
        super().__init__(max_iterations)
        self.branch_info = branch_info if branch_info is not None else {}
        self.symbolic_inputs = symbolic_inputs or {}

    def _interpret_let(self, stmt, env, functions):
        result = super()._interpret_let(stmt, env, functions)
        if hasattr(stmt, 'name') and stmt.name in self.symbolic_inputs:
            result.forget(stmt.name)
        return result

    def _interpret_if(self, stmt, env, functions):
        then_env, else_env = self._refine_condition(stmt.cond, env)

        line = getattr(stmt, 'line', None) or getattr(stmt.cond, 'line', 0)
        self.branch_info[line] = PolyBranchInfo(
            line=line,
            true_feasible=not then_env.is_bot(),
            false_feasible=not else_env.is_bot(),
        )

        then_result = self._interpret_block(stmt.then_body, then_env, functions)
        if stmt.else_body:
            else_result = self._interpret_block(stmt.else_body, else_env, functions)
        else:
            else_result = else_env

        if then_result.is_bot():
            return else_result
        if else_result.is_bot():
            return then_result
        return then_result.join(else_result)

    def _interpret_while(self, stmt, env, functions):
        current = env.copy()

        for i in range(self.max_iterations):
            then_env, _ = self._refine_condition(stmt.cond, current)
            body_env = self._interpret_block(stmt.body, then_env, functions)

            if body_env.is_bot():
                break

            joined = current.join(body_env)
            next_env = current.widen(joined)
            if next_env.equals(current):
                break
            current = next_env

        # Record loop-condition branch info
        then_env, exit_env = self._refine_condition(stmt.cond, current)
        line = getattr(stmt, 'line', None) or getattr(stmt.cond, 'line', 0)
        if line not in self.branch_info:
            self.branch_info[line] = PolyBranchInfo(
                line=line,
                true_feasible=not then_env.is_bot(),
                false_feasible=not exit_env.is_bot(),
            )

        return exit_env


class PolyhedralPreAnalyzer:
    """Runs polyhedral pre-analysis for branch feasibility."""

    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations

    def analyze(self, source: str, symbolic_inputs: dict = None) -> dict:
        branch_info = {}
        interp = BranchCapturingInterpreter(
            max_iterations=self.max_iterations,
            branch_info=branch_info,
            symbolic_inputs=symbolic_inputs or {},
        )
        analysis = interp.analyze(source)
        return {
            "env": analysis["env"],
            "warnings": analysis.get("warnings", []),
            "branch_info": branch_info,
        }


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PolyGuidedResult:
    """Result from polyhedral-guided symbolic execution."""
    execution: ExecutionResult
    polyhedral_env: Optional[PolyhedralDomain]
    polyhedral_warnings: list
    pruned_by_polyhedral: int
    smt_checks_performed: int
    smt_checks_saved: int

    @property
    def paths(self):
        return self.execution.paths

    @property
    def test_cases(self):
        return self.execution.test_cases

    @property
    def pruning_ratio(self) -> float:
        total = self.smt_checks_performed + self.smt_checks_saved
        return self.smt_checks_saved / total if total > 0 else 0.0

    def __repr__(self):
        return (
            f"PolyGuidedResult(paths={len(self.paths)}, "
            f"pruned={self.pruned_by_polyhedral}, "
            f"smt_saved={self.smt_checks_saved})"
        )


# ---------------------------------------------------------------------------
# Guided executor: overrides _check_feasible to use polyhedral info
# ---------------------------------------------------------------------------

class PolyGuidedExecutor(SymbolicExecutor):
    """Symbolic executor guided by polyhedral abstract interpretation.

    Before each SMT feasibility check, consults polyhedral branch info.
    If polyhedral analysis proved a branch infeasible, skips the SMT call.
    """

    def __init__(self, max_paths=64, max_loop_unroll=5):
        super().__init__(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
        self.branch_info: Dict[int, PolyBranchInfo] = {}
        self.polyhedral_env = None
        self.polyhedral_warnings = []
        self._pruned_count = 0
        self._smt_checks = 0
        self._smt_saved = 0
        self._current_branch_line = None
        self._is_true_branch = None

    def guided_execute(self, source: str, symbolic_inputs: dict = None) -> PolyGuidedResult:
        """Execute with polyhedral guidance."""
        # Phase 1: Polyhedral pre-analysis
        pre = PolyhedralPreAnalyzer()
        analysis = pre.analyze(source, symbolic_inputs)
        self.branch_info = analysis["branch_info"]
        self.polyhedral_env = analysis["env"]
        self.polyhedral_warnings = analysis["warnings"]
        self._pruned_count = 0
        self._smt_checks = 0
        self._smt_saved = 0

        # Phase 2: Symbolic execution
        exec_result = self.execute(source, symbolic_inputs)

        return PolyGuidedResult(
            execution=exec_result,
            polyhedral_env=self.polyhedral_env,
            polyhedral_warnings=self.polyhedral_warnings,
            pruned_by_polyhedral=self._pruned_count,
            smt_checks_performed=self._smt_checks,
            smt_checks_saved=self._smt_saved,
        )

    def _execute_if(self, state: PathState, stmt) -> Optional[list]:
        """Override to track which branch we're checking."""
        # Record the line so _check_feasible can consult branch_info
        line = getattr(stmt, 'line', None) or getattr(stmt.cond, 'line', 0)
        old_line = self._current_branch_line
        self._current_branch_line = line
        self._is_true_branch = True  # Will alternate in _check_feasible

        result = super()._execute_if(state, stmt)

        self._current_branch_line = old_line
        return result

    def _check_feasible(self, state: PathState) -> bool:
        """Override to consult polyhedral branch info before SMT."""
        line = self._current_branch_line
        if line is not None and line in self.branch_info:
            info = self.branch_info[line]

            # Determine which branch this is by looking at recent constraint
            # (true branch adds cond, false branch adds not(cond))
            # We check both directions based on the info
            if self._is_true_branch is not None:
                if self._is_true_branch and not info.true_feasible:
                    self._pruned_count += 1
                    self._smt_saved += 1
                    self._is_true_branch = False  # Next call is false branch
                    return False
                if not self._is_true_branch and not info.false_feasible:
                    self._pruned_count += 1
                    self._smt_saved += 1
                    return False

                # Toggle for next call (true checked first, then false)
                if self._is_true_branch:
                    self._is_true_branch = False

        # Fall back to SMT
        self._smt_checks += 1
        return super()._check_feasible(state)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def poly_guided_execute(
    source: str, symbolic_inputs: dict = None,
    max_paths: int = 64, max_loop_unroll: int = 5,
) -> PolyGuidedResult:
    """Run polyhedral-guided symbolic execution."""
    executor = PolyGuidedExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    return executor.guided_execute(source, symbolic_inputs)


def poly_guided_check_assertions(source: str, symbolic_inputs: dict = None) -> Tuple:
    """Check assertions with polyhedral guidance."""
    guided = poly_guided_execute(source, symbolic_inputs)
    assertion_result = check_assertions(source, symbolic_inputs)
    return (assertion_result, guided)


def poly_guided_generate_tests(source: str, symbolic_inputs: dict = None) -> Tuple:
    """Generate tests with polyhedral guidance."""
    guided = poly_guided_execute(source, symbolic_inputs)
    return (guided.test_cases, guided)


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_guided_vs_plain(source: str, symbolic_inputs: dict = None) -> dict:
    """Compare plain vs polyhedral-guided symbolic execution."""
    results = {}

    t0 = time.time()
    plain = symbolic_execute(source, symbolic_inputs)
    results["plain"] = {
        "paths": len(plain.paths),
        "test_cases": len(plain.test_cases),
        "time": time.time() - t0,
    }

    t0 = time.time()
    guided = poly_guided_execute(source, symbolic_inputs)
    results["polyhedral_guided"] = {
        "paths": len(guided.paths),
        "test_cases": len(guided.test_cases),
        "pruned": guided.pruned_by_polyhedral,
        "smt_saved": guided.smt_checks_saved,
        "smt_performed": guided.smt_checks_performed,
        "pruning_ratio": guided.pruning_ratio,
        "time": time.time() - t0,
    }

    return results


def compare_all_strategies(source: str, symbolic_inputs: dict = None) -> dict:
    """Compare plain vs V001 interval-guided vs V131 polyhedral-guided."""
    results = compare_guided_vs_plain(source, symbolic_inputs)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V001_guided_symbolic_execution'))
        from guided_symbolic import guided_execute as v001_execute
        t0 = time.time()
        v001 = v001_execute(source, symbolic_inputs)
        results["interval_guided"] = {
            "paths": len(v001.paths),
            "test_cases": len(v001.test_cases),
            "pruned": v001.pruned_by_abstract,
            "smt_saved": v001.smt_checks_saved,
            "smt_performed": v001.smt_checks_performed,
            "pruning_ratio": v001.smt_checks_saved / max(1, v001.smt_checks_saved + v001.smt_checks_performed),
            "time": time.time() - t0,
        }
    except Exception:
        results["interval_guided"] = {"error": "V001 not available"}

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def poly_guided_summary(source: str, symbolic_inputs: dict = None) -> str:
    """Human-readable summary of polyhedral-guided symbolic execution."""
    result = poly_guided_execute(source, symbolic_inputs)
    lines = [
        "Polyhedral-Guided Symbolic Execution",
        "=" * 40,
        f"Paths explored: {len(result.paths)}",
        f"Test cases generated: {len(result.test_cases)}",
        f"Branches pruned by polyhedral: {result.pruned_by_polyhedral}",
        f"SMT checks saved: {result.smt_checks_saved}",
        f"SMT checks performed: {result.smt_checks_performed}",
        f"Pruning ratio: {result.pruning_ratio:.1%}",
    ]
    if result.polyhedral_warnings:
        lines.append(f"Polyhedral warnings: {len(result.polyhedral_warnings)}")
    return "\n".join(lines)
