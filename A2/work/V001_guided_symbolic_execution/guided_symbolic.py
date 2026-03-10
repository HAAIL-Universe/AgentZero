"""
V001: Abstract-Interpretation-Guided Symbolic Execution
A2 (Verification Agent) -- Session 001

Composes: C039 (Abstract Interpreter) + C038 (Symbolic Execution)

Idea: Run cheap abstract interpretation FIRST to get variable bounds,
then use those bounds to prune infeasible paths BEFORE expensive SMT
feasibility checks. This reduces path explosion in symbolic execution.

Architecture:
  Source -> C039 Abstract Interpreter -> Variable bounds (intervals, signs, consts)
         -> C038 Symbolic Executor (with guided pruning)
         -> Pruned path tree + test cases

Key insight: Interval checks are O(1) per branch. SMT checks are
potentially exponential. By eliminating obviously-infeasible branches
with intervals first, we avoid calling the SMT solver entirely on
those paths.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# Import C039 abstract interpreter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, AbstractValue,
    Sign, Interval, INTERVAL_BOT, INTERVAL_TOP,
    interval_meet, interval_join,
    ConstVal, ConstTop, ConstBot, CONST_TOP, CONST_BOT,
    WarningKind, Warning as AIWarning,
    analyze as ai_analyze
)

# Import C038 symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, SymValue, PathState, PathStatus, SymType,
    ExecutionResult, TestCase, AssertionResult, AssertionViolation,
    CoverageResult,
    symbolic_execute, generate_tests, check_assertions,
    smt_not, smt_and, smt_or
)

# Import C010 parser (shared AST)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

# Import C037 SMT solver types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Term, Var as SMTVar, App, IntConst, BoolConst, Op as SMTOp, BOOL, INT


# ============================================================
# Guided Analysis Results
# ============================================================

@dataclass
class GuidedResult:
    """Result from guided symbolic execution."""
    execution: ExecutionResult           # Full symbolic execution result
    abstract_env: AbstractEnv            # Abstract interpretation environment
    abstract_warnings: list              # Warnings from abstract analysis
    pruned_by_abstract: int              # Branches pruned by abstract interp (no SMT needed)
    smt_checks_performed: int            # SMT checks still needed
    smt_checks_saved: int                # SMT checks avoided due to abstract pruning

    @property
    def paths(self):
        return self.execution.paths

    @property
    def test_cases(self):
        return self.execution.test_cases

    @property
    def feasible_paths(self):
        return self.execution.feasible_paths

    @property
    def total_paths(self):
        return self.execution.total_paths

    @property
    def pruning_ratio(self) -> float:
        """Fraction of SMT checks saved by abstract pre-filtering."""
        total = self.smt_checks_performed + self.smt_checks_saved
        if total == 0:
            return 0.0
        return self.smt_checks_saved / total


@dataclass
class BranchInfo:
    """Information about a branch point from abstract analysis."""
    line: int
    true_feasible: bool = True      # True branch possibly feasible per abstract interp
    false_feasible: bool = True     # False branch possibly feasible per abstract interp
    true_interval: Optional[Interval] = None   # Refined interval if true
    false_interval: Optional[Interval] = None  # Refined interval if false
    variable: Optional[str] = None             # Variable being compared


# ============================================================
# Abstract Pre-Analysis
# ============================================================

class AbstractPreAnalyzer:
    """Runs abstract interpretation to gather branch feasibility info."""

    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.branch_info = {}  # line -> BranchInfo
        self.symbolic_inputs = {}  # var_name -> type

    def analyze(self, source: str, symbolic_inputs: dict = None) -> dict:
        """Run abstract interpretation, treating symbolic inputs as TOP."""
        self.symbolic_inputs = symbolic_inputs or {}
        interp = AbstractInterpreter(max_iterations=self.max_iterations)

        # Parse to get AST
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        # Set up environment: symbolic inputs are TOP (unknown)
        env = AbstractEnv()
        interp.warnings = []
        interp.functions = {}
        interp.var_reads = set()
        interp.var_writes = {}

        # Collect function declarations first
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                interp.functions[stmt.name] = stmt

        # For symbolic inputs, set variables to TOP before interpretation
        if symbolic_inputs:
            for var_name, var_type in symbolic_inputs.items():
                env.set_top(var_name)

        # Interpret statements and collect branch info
        self._analyze_stmts(interp, program.stmts, env)

        return {
            'env': env,
            'warnings': list(interp.warnings),
            'branch_info': dict(self.branch_info),
        }

    def _analyze_stmts(self, interp: AbstractInterpreter, stmts, env: AbstractEnv) -> AbstractEnv:
        """Analyze statements, recording branch info at if/while."""
        for stmt in stmts:
            env = self._analyze_stmt(interp, stmt, env)
        return env

    def _analyze_stmt(self, interp: AbstractInterpreter, stmt, env: AbstractEnv) -> AbstractEnv:
        """Analyze one statement, wrapping the interpreter's logic to capture branch info."""
        if isinstance(stmt, IfStmt):
            return self._analyze_if(interp, stmt, env)
        elif isinstance(stmt, WhileStmt):
            return self._analyze_while(interp, stmt, env)
        elif isinstance(stmt, Block):
            return self._analyze_stmts(interp, stmt.stmts, env)
        else:
            # Delegate to the standard interpreter for non-branching stmts
            env = interp._interpret_stmt(stmt, env)
            # If this was a let/assign for a symbolic input, force it back to TOP
            # because the symbolic executor treats it as unconstrained
            if isinstance(stmt, (LetDecl, Assign)):
                name = stmt.name
                if name in self.symbolic_inputs:
                    env.set_top(name)
            return env

    def _analyze_if(self, interp: AbstractInterpreter, stmt: IfStmt, env: AbstractEnv) -> AbstractEnv:
        """Analyze if-statement and record branch feasibility."""
        cond = interp._eval_expr(stmt.cond, env)

        info = BranchInfo(line=getattr(stmt, 'line', 0))

        # Extract variable and comparison info for richer reporting
        if isinstance(stmt.cond, BinOp):
            if isinstance(stmt.cond.left, ASTVar):
                info.variable = stmt.cond.left.name
            elif isinstance(stmt.cond.right, ASTVar):
                info.variable = stmt.cond.right.name

        # Check if condition is statically determined
        if isinstance(cond.const, ConstVal):
            if cond.const.value:
                info.true_feasible = True
                info.false_feasible = False
            else:
                info.true_feasible = False
                info.false_feasible = True
        else:
            # Both branches possible in general, but check intervals
            info.true_feasible = True
            info.false_feasible = True

            # Refine environments for each branch
            then_env = interp._refine_env_for_condition(stmt.cond, env, True)
            else_env = interp._refine_env_for_condition(stmt.cond, env, False)

            # Check if refined interval is BOT (empty) for either branch
            if info.variable:
                true_iv = then_env.get_interval(info.variable)
                false_iv = else_env.get_interval(info.variable)
                info.true_interval = true_iv
                info.false_interval = false_iv

                if true_iv.is_bot():
                    info.true_feasible = False
                if false_iv.is_bot():
                    info.false_feasible = False

        self.branch_info[info.line] = info

        # Now do the actual abstract interpretation for both branches
        if isinstance(cond.const, ConstVal):
            if cond.const.value:
                return self._analyze_stmt(interp, stmt.then_body, env)
            else:
                if stmt.else_body:
                    return self._analyze_stmt(interp, stmt.else_body, env)
                return env

        then_env = interp._refine_env_for_condition(stmt.cond, env, True)
        else_env = interp._refine_env_for_condition(stmt.cond, env, False)

        then_env = self._analyze_stmt(interp, stmt.then_body, then_env)
        if stmt.else_body:
            else_env = self._analyze_stmt(interp, stmt.else_body, else_env)

        return then_env.join(else_env)

    def _analyze_while(self, interp: AbstractInterpreter, stmt: WhileStmt, env: AbstractEnv) -> AbstractEnv:
        """Delegate while to interpreter but record branch info."""
        # Record that this is a branching point
        info = BranchInfo(line=getattr(stmt, 'line', 0))
        if isinstance(stmt.cond, BinOp):
            if isinstance(stmt.cond.left, ASTVar):
                info.variable = stmt.cond.left.name
            elif isinstance(stmt.cond.right, ASTVar):
                info.variable = stmt.cond.right.name

        # Loops are always potentially feasible both ways (enter/skip)
        # unless we can prove the condition is always true/false
        cond = interp._eval_expr(stmt.cond, env)
        if isinstance(cond.const, ConstVal):
            info.true_feasible = bool(cond.const.value)
            info.false_feasible = not bool(cond.const.value)
        else:
            info.true_feasible = True
            info.false_feasible = True

        self.branch_info[info.line] = info

        # Delegate the actual analysis to the interpreter
        return interp._interpret_while(stmt, env)


# ============================================================
# Guided Symbolic Executor
# ============================================================

class GuidedSymbolicExecutor(SymbolicExecutor):
    """Symbolic executor that uses abstract interpretation to prune paths.

    Overrides the feasibility check to first consult abstract interpretation
    results. If abstract analysis says a branch is infeasible, we skip the
    expensive SMT check entirely.
    """

    def __init__(self, max_paths=64, max_loop_unroll=5):
        super().__init__(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
        self.branch_info = {}        # line -> BranchInfo (from abstract pre-analysis)
        self.abstract_env = None     # Abstract environment from pre-analysis
        self.abstract_warnings = []  # Warnings from abstract analysis
        self._pruned_count = 0       # Branches pruned by abstract interp
        self._smt_checks = 0         # SMT feasibility checks performed
        self._smt_saved = 0          # SMT checks avoided

    def guided_execute(self, source: str, symbolic_inputs: dict = None) -> GuidedResult:
        """Execute with abstract-interpretation guidance.

        1. Run abstract interpretation to get branch feasibility info
        2. Run symbolic execution using that info to prune paths
        3. Return combined results
        """
        # Phase 1: Abstract pre-analysis
        pre = AbstractPreAnalyzer()
        analysis = pre.analyze(source, symbolic_inputs)
        self.branch_info = analysis['branch_info']
        self.abstract_env = analysis['env']
        self.abstract_warnings = analysis['warnings']
        self._pruned_count = 0
        self._smt_checks = 0
        self._smt_saved = 0

        # Phase 2: Symbolic execution with guided pruning
        result = self.execute(source, symbolic_inputs)

        return GuidedResult(
            execution=result,
            abstract_env=self.abstract_env,
            abstract_warnings=self.abstract_warnings,
            pruned_by_abstract=self._pruned_count,
            smt_checks_performed=self._smt_checks,
            smt_checks_saved=self._smt_saved,
        )

    def _execute_if(self, state: PathState, stmt: IfStmt) -> Optional[list]:
        """Override if-execution to use abstract branch info for early pruning."""
        cond_val = self._eval_expr(state, stmt.cond)

        if cond_val.is_concrete():
            truth = bool(cond_val.concrete)
            state.covered_branches.append((stmt.line, truth))
            if truth:
                return self._execute_block_fork(state, stmt.then_body)
            elif stmt.else_body is not None:
                if isinstance(stmt.else_body, IfStmt):
                    return self._execute_if(state, stmt.else_body)
                return self._execute_block_fork(state, stmt.else_body)
            return None

        # Symbolic -- check abstract guidance first
        cond_term = self._ensure_bool_term(cond_val)
        if cond_term is None:
            state.status = PathStatus.ERROR
            state.error_msg = "Cannot create boolean constraint"
            self.completed_paths.append(state)
            return []

        true_state = state.copy()
        true_state.path_id = self._new_path_id()
        true_state.constraints.append(cond_term)
        true_state.covered_branches.append((stmt.line, True))

        false_state = state.copy()
        false_state.path_id = self._new_path_id()
        false_state.constraints.append(smt_not(cond_term))
        false_state.covered_branches.append((stmt.line, False))

        # Consult abstract analysis for this branch point
        info = self.branch_info.get(stmt.line)

        # Determine feasibility with abstract guidance
        true_feasible = self._guided_check_feasible(true_state, info, True)
        false_feasible = self._guided_check_feasible(false_state, info, False)

        forks = []

        if true_feasible:
            try:
                true_forks = self._execute_block_fork(true_state, stmt.then_body)
            except ReturnSignal as rs:
                true_state.return_value = rs.value
                true_state.status = PathStatus.COMPLETED
                self.completed_paths.append(true_state)
                true_forks = None
            if true_forks:
                forks.extend(true_forks)
            elif true_state.status == PathStatus.ACTIVE:
                forks.append((true_state, []))
            else:
                if true_state not in self.completed_paths:
                    self.completed_paths.append(true_state)
        else:
            true_state.status = PathStatus.INFEASIBLE
            self.completed_paths.append(true_state)

        if false_feasible:
            if stmt.else_body is not None:
                if isinstance(stmt.else_body, IfStmt):
                    try:
                        false_forks = self._execute_if(false_state, stmt.else_body)
                    except ReturnSignal as rs:
                        false_state.return_value = rs.value
                        false_state.status = PathStatus.COMPLETED
                        self.completed_paths.append(false_state)
                        false_forks = None
                    if false_forks:
                        forks.extend(false_forks)
                    elif false_state.status == PathStatus.ACTIVE:
                        forks.append((false_state, []))
                    elif false_state not in self.completed_paths:
                        self.completed_paths.append(false_state)
                else:
                    try:
                        false_forks = self._execute_block_fork(false_state, stmt.else_body)
                    except ReturnSignal as rs:
                        false_state.return_value = rs.value
                        false_state.status = PathStatus.COMPLETED
                        self.completed_paths.append(false_state)
                        false_forks = None
                    if false_forks:
                        forks.extend(false_forks)
                    elif false_state.status == PathStatus.ACTIVE:
                        forks.append((false_state, []))
                    elif false_state not in self.completed_paths:
                        self.completed_paths.append(false_state)
            else:
                forks.append((false_state, []))
        else:
            false_state.status = PathStatus.INFEASIBLE
            self.completed_paths.append(false_state)

        state.status = PathStatus.COMPLETED
        return forks if forks else []

    def _guided_check_feasible(self, state: PathState, info: Optional[BranchInfo], is_true_branch: bool) -> bool:
        """Check feasibility with abstract guidance.

        If abstract analysis says a branch is infeasible, skip SMT.
        Otherwise fall back to the expensive SMT check.
        """
        if info is not None:
            if is_true_branch and not info.true_feasible:
                self._pruned_count += 1
                self._smt_saved += 1
                return False
            if not is_true_branch and not info.false_feasible:
                self._pruned_count += 1
                self._smt_saved += 1
                return False

        # Abstract analysis couldn't rule it out -- fall back to SMT
        self._smt_checks += 1
        return self._check_feasible(state)


# We need ReturnSignal from the parent module
try:
    from symbolic_execution import ReturnSignal
except ImportError:
    # Define it locally if not exported
    class ReturnSignal(Exception):
        def __init__(self, value):
            self.value = value


# ============================================================
# Convenience API
# ============================================================

def guided_execute(source: str, symbolic_inputs: dict = None,
                   max_paths=64, max_loop_unroll=5) -> GuidedResult:
    """Run abstract-interpretation-guided symbolic execution."""
    executor = GuidedSymbolicExecutor(
        max_paths=max_paths,
        max_loop_unroll=max_loop_unroll,
    )
    return executor.guided_execute(source, symbolic_inputs)


def guided_check_assertions(source: str, symbolic_inputs: dict = None) -> tuple:
    """Check assertions with guidance. Returns (AssertionResult, GuidedResult)."""
    executor = GuidedSymbolicExecutor()
    guided = executor.guided_execute(source, symbolic_inputs)

    # Also run assertion checking
    assertion_result = check_assertions(source, symbolic_inputs)
    return assertion_result, guided


def compare_guided_vs_plain(source: str, symbolic_inputs: dict = None) -> dict:
    """Compare guided vs plain symbolic execution for benchmarking."""
    # Plain execution
    plain_executor = SymbolicExecutor()
    plain_executor._smt_check_count = 0
    original_check = plain_executor._check_feasible

    plain_smt_calls = [0]
    def counting_check(state):
        plain_smt_calls[0] += 1
        return original_check(state)
    plain_executor._check_feasible = counting_check

    plain_result = plain_executor.execute(source, symbolic_inputs)

    # Guided execution
    guided_result = guided_execute(source, symbolic_inputs)

    return {
        'plain_paths': plain_result.total_paths,
        'guided_paths': guided_result.total_paths,
        'plain_feasible': len(plain_result.feasible_paths),
        'guided_feasible': len(guided_result.feasible_paths),
        'plain_smt_checks': plain_smt_calls[0],
        'guided_smt_checks': guided_result.smt_checks_performed,
        'guided_smt_saved': guided_result.smt_checks_saved,
        'pruning_ratio': guided_result.pruning_ratio,
        'abstract_warnings': guided_result.abstract_warnings,
    }
