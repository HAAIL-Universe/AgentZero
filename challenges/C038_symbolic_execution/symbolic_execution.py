"""
C038: Symbolic Execution Engine
Composes C037 (SMT Solver) + C010 (Stack VM Parser)

Architecture:
  Source -> C010 Parser -> AST -> Symbolic Interpreter -> Path Constraints -> C037 SMT -> Test Inputs

Features:
  - Symbolic integer/boolean values
  - Path forking at branches (if/while)
  - Constraint collection along each path
  - SMT-based feasibility checking and model generation
  - Automatic test input generation for path coverage
  - Coverage analysis (statement, branch, path)
  - Loop unrolling with configurable bound
  - Function support with symbolic arguments
  - Dead path detection (infeasible branches)
  - Assertion checking (prove assertions hold on all paths)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var as SMTVar, App,
    IntConst, BoolConst, Op as SMTOp, BOOL, INT
)


# ============================================================
# Symbolic Values
# ============================================================

class SymType(Enum):
    CONCRETE = "concrete"
    SYMBOLIC = "symbolic"


@dataclass
class SymValue:
    """A value that is either concrete or symbolic (an SMT term)."""
    kind: SymType
    concrete: Any = None
    term: Optional[Term] = None
    name: Optional[str] = None

    @staticmethod
    def from_concrete(val):
        return SymValue(kind=SymType.CONCRETE, concrete=val)

    @staticmethod
    def from_symbolic(term, name=None):
        return SymValue(kind=SymType.SYMBOLIC, term=term, name=name)

    def is_concrete(self):
        return self.kind == SymType.CONCRETE

    def is_symbolic(self):
        return self.kind == SymType.SYMBOLIC

    def __repr__(self):
        if self.is_concrete():
            return f"Concrete({self.concrete})"
        return f"Symbolic({self.term})"


# ============================================================
# Execution State
# ============================================================

class PathStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INFEASIBLE = "infeasible"
    ERROR = "error"
    ASSERTION_FAILED = "assertion_failed"


@dataclass
class PathState:
    """State of one execution path."""
    env: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)
    status: PathStatus = PathStatus.ACTIVE
    output: list = field(default_factory=list)
    covered_lines: set = field(default_factory=set)
    covered_branches: list = field(default_factory=list)
    path_id: int = 0
    error_msg: str = ""
    assertion_line: int = 0
    return_value: Optional[SymValue] = None

    def copy(self):
        return PathState(
            env=dict(self.env),
            constraints=list(self.constraints),
            status=self.status,
            output=list(self.output),
            covered_lines=set(self.covered_lines),
            covered_branches=list(self.covered_branches),
            path_id=self.path_id,
            error_msg=self.error_msg,
            return_value=self.return_value,
        )


# ============================================================
# Return signal
# ============================================================

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class ForkSignal(Exception):
    """Signal that execution needs to fork from within a function call."""
    def __init__(self, fn_results):
        self.fn_results = fn_results  # list of (PathState, return_value)


# ============================================================
# SMT term helpers
# ============================================================

def smt_not(t: Term) -> Term:
    """Negate an SMT term. Uses direct complement ops where possible
    to avoid NOT(EQ) which the DPLL(T) solver doesn't handle well."""
    if isinstance(t, App):
        # Direct complement for comparison operators
        complement = {
            SMTOp.EQ: SMTOp.NEQ,
            SMTOp.NEQ: SMTOp.EQ,
            SMTOp.LT: SMTOp.GE,
            SMTOp.GE: SMTOp.LT,
            SMTOp.GT: SMTOp.LE,
            SMTOp.LE: SMTOp.GT,
        }
        if t.op in complement:
            return App(complement[t.op], t.args, BOOL)
        # Double negation elimination
        if t.op == SMTOp.NOT:
            return t.args[0]
    return App(SMTOp.NOT, [t], BOOL)

def smt_and(a: Term, b: Term) -> Term:
    return App(SMTOp.AND, [a, b], BOOL)

def smt_or(a: Term, b: Term) -> Term:
    return App(SMTOp.OR, [a, b], BOOL)


# ============================================================
# Symbolic Execution Engine
# ============================================================

class SymbolicExecutor:
    """
    Symbolically executes programs parsed by C010's parser.
    Collects path constraints and uses C037 SMT solver to
    generate concrete test inputs.
    """

    def __init__(self, max_paths=64, max_loop_unroll=5):
        self.max_paths = max_paths
        self.max_loop_unroll = max_loop_unroll
        self._path_counter = 0
        self._sym_counter = 0
        self.completed_paths: list = []
        self.functions: dict = {}
        # Shared symbolic variable registry (name -> SMTVar term)
        self._sym_vars: dict = {}

    def _fresh_sym(self, prefix="sym"):
        self._sym_counter += 1
        return f"{prefix}_{self._sym_counter}"

    def _new_path_id(self):
        self._path_counter += 1
        return self._path_counter

    def _get_or_create_int_var(self, name: str) -> SMTVar:
        """Get or create a shared symbolic integer variable."""
        if name not in self._sym_vars:
            self._sym_vars[name] = SMTVar(name, INT)
        return self._sym_vars[name]

    def _get_or_create_bool_var(self, name: str) -> SMTVar:
        """Get or create a shared symbolic boolean variable."""
        if name not in self._sym_vars:
            self._sym_vars[name] = SMTVar(name, BOOL)
        return self._sym_vars[name]

    # --- Public API ---

    def execute(self, source: str, symbolic_inputs: dict = None) -> 'ExecutionResult':
        """
        Symbolically execute a program.

        Args:
            source: C010 source code
            symbolic_inputs: dict of {var_name: 'int'|'bool'} for symbolic inputs.

        Returns:
            ExecutionResult with all explored paths and generated test cases.
        """
        self.completed_paths = []
        self.functions = {}
        self._path_counter = 0
        self._sym_counter = 0
        self._sym_vars = {}

        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()

        # Collect function declarations first
        for stmt in ast.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt

        # Create initial state
        initial = PathState(path_id=self._new_path_id())

        # Set up symbolic inputs
        if symbolic_inputs:
            for name, typ in symbolic_inputs.items():
                if typ == 'int':
                    term = self._get_or_create_int_var(name)
                    initial.env[name] = SymValue.from_symbolic(term, name=name)
                elif typ == 'bool':
                    term = self._get_or_create_bool_var(name)
                    initial.env[name] = SymValue.from_symbolic(term, name=name)

        # Execute all paths via worklist
        self._run_worklist(initial, ast.stmts)

        # Generate test cases
        test_cases = self._generate_tests(symbolic_inputs or {})

        return ExecutionResult(
            paths=self.completed_paths,
            test_cases=test_cases,
            functions=list(self.functions.keys()),
        )

    def check_assertions(self, source: str, symbolic_inputs: dict = None) -> 'AssertionResult':
        """Check if all assert() calls hold on all feasible paths."""
        result = self.execute(source, symbolic_inputs)
        violations = []
        for path in result.paths:
            if path.status == PathStatus.ASSERTION_FAILED:
                solver = SMTSolver()
                self._declare_vars(solver, symbolic_inputs or {})
                for c in path.constraints:
                    solver.add(c)
                r = solver.check()
                model = solver.model() if r == SMTResult.SAT else None
                violations.append(AssertionViolation(
                    line=path.assertion_line,
                    path_constraints=path.constraints,
                    counterexample=model,
                ))
        return AssertionResult(
            holds=len(violations) == 0,
            violations=violations,
            paths_explored=len(result.paths),
        )

    def find_inputs(self, source: str, symbolic_inputs: dict,
                    target_line: int) -> Optional[dict]:
        """Find concrete inputs that cause execution to reach a specific line."""
        result = self.execute(source, symbolic_inputs)
        for path in result.paths:
            if target_line in path.covered_lines and path.status in (
                PathStatus.COMPLETED, PathStatus.ACTIVE
            ):
                solver = SMTSolver()
                self._declare_vars(solver, symbolic_inputs)
                for c in path.constraints:
                    solver.add(c)
                r = solver.check()
                if r == SMTResult.SAT:
                    return solver.model()
        return None

    def get_coverage(self, source: str, symbolic_inputs: dict = None) -> 'CoverageResult':
        """Analyze statement and branch coverage across all paths."""
        result = self.execute(source, symbolic_inputs)
        covered_lines = set()
        all_branches = set()
        covered_branches = set()

        for path in result.paths:
            if path.status != PathStatus.INFEASIBLE:
                covered_lines |= path.covered_lines
                for b in path.covered_branches:
                    covered_branches.add(b)

        # Collect all branch sites
        for path in result.paths:
            for b in path.covered_branches:
                all_branches.add((b[0], True))
                all_branches.add((b[0], False))

        all_lines = set()
        for path in result.paths:
            all_lines |= path.covered_lines

        return CoverageResult(
            total_lines=len(all_lines),
            covered_lines=len(covered_lines),
            line_coverage=len(covered_lines) / max(len(all_lines), 1),
            total_branches=len(all_branches),
            covered_branches=len(covered_branches),
            branch_coverage=len(covered_branches) / max(len(all_branches), 1),
            paths_explored=len(result.paths),
            feasible_paths=sum(1 for p in result.paths if p.status != PathStatus.INFEASIBLE),
            dead_branches=[(b[0], b[1]) for b in all_branches - covered_branches],
        )

    # --- Worklist Engine ---

    def _run_worklist(self, initial: PathState, stmts: list):
        """Execute all paths via BFS worklist."""
        worklist = [(initial, stmts)]
        while worklist and len(self.completed_paths) < self.max_paths:
            state, remaining = worklist.pop(0)
            try:
                forks = self._execute_stmts(state, remaining)
                for fork_state, fork_stmts in forks:
                    if fork_state.status == PathStatus.ACTIVE:
                        if len(self.completed_paths) < self.max_paths:
                            worklist.append((fork_state, fork_stmts))
                        else:
                            fork_state.status = PathStatus.COMPLETED
                            self.completed_paths.append(fork_state)
            except ReturnSignal as rs:
                state.return_value = rs.value
                state.status = PathStatus.COMPLETED
                self.completed_paths.append(state)

    # --- Statement Execution ---

    def _execute_stmts(self, state: PathState, stmts: list) -> list:
        """Execute statements, returning forks for continued execution."""
        forks = []
        i = 0
        while i < len(stmts):
            if state.status != PathStatus.ACTIVE:
                break
            stmt = stmts[i]
            if hasattr(stmt, 'line') and stmt.line > 0:
                state.covered_lines.add(stmt.line)

            result = self._execute_one(state, stmt)
            if result is not None:
                remaining = stmts[i+1:]
                for fork_state, fork_stmts in result:
                    forks.append((fork_state, fork_stmts + remaining))
                return forks
            i += 1

        if state.status == PathStatus.ACTIVE:
            state.status = PathStatus.COMPLETED
        self.completed_paths.append(state)
        return forks

    def _execute_one(self, state: PathState, stmt) -> Optional[list]:
        """Execute one statement. Returns None for normal flow, list of forks otherwise."""
        if isinstance(stmt, FnDecl):
            self.functions[stmt.name] = stmt
            return None

        if isinstance(stmt, LetDecl):
            # If variable is already symbolic (from inputs), keep symbolic value
            if stmt.name in state.env and state.env[stmt.name].is_symbolic():
                return None
            try:
                val = self._eval_expr(state, stmt.value)
            except ForkSignal as fs:
                return self._apply_fn_forks(fs.fn_results, stmt.name)
            state.env[stmt.name] = val
            return None

        if isinstance(stmt, Assign):
            try:
                val = self._eval_expr(state, stmt.value)
            except ForkSignal as fs:
                return self._apply_fn_forks(fs.fn_results, stmt.name)
            state.env[stmt.name] = val
            return None

        if isinstance(stmt, PrintStmt):
            try:
                val = self._eval_expr(state, stmt.value)
            except ForkSignal as fs:
                # Each fork path prints its own return value
                forks = []
                for fn_state, ret_val in fs.fn_results:
                    fn_state.output.append(ret_val)
                    forks.append((fn_state, []))
                return forks
            state.output.append(val)
            return None

        if isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                try:
                    val = self._eval_expr(state, stmt.value)
                except ForkSignal as fs:
                    # Each fork raises its own ReturnSignal -- just take first
                    # and add rest to completed
                    for fn_state, ret_val in fs.fn_results[1:]:
                        fn_state.return_value = ret_val
                        fn_state.status = PathStatus.COMPLETED
                        self.completed_paths.append(fn_state)
                    if fs.fn_results:
                        raise ReturnSignal(fs.fn_results[0][1])
                    val = SymValue.from_concrete(None)
            else:
                val = SymValue.from_concrete(None)
            raise ReturnSignal(val)

        if isinstance(stmt, Block):
            return self._execute_block_fork(state, stmt)

        if isinstance(stmt, IfStmt):
            return self._execute_if(state, stmt)

        if isinstance(stmt, WhileStmt):
            return self._execute_while(state, stmt)

        # Expression statement
        try:
            self._eval_expr(state, stmt)
        except ForkSignal as fs:
            return [(fn_state, []) for fn_state, _ in fs.fn_results]
        return None

    def _execute_block_fork(self, state: PathState, block: Block) -> Optional[list]:
        """Execute block, returning forks if any."""
        return self._execute_stmts_inline(state, block.stmts)

    def _execute_stmts_inline(self, state: PathState, stmts: list) -> Optional[list]:
        """Execute statements inline, returning forks if branching occurs."""
        for i, stmt in enumerate(stmts):
            if state.status != PathStatus.ACTIVE:
                return None
            if hasattr(stmt, 'line') and stmt.line > 0:
                state.covered_lines.add(stmt.line)
            result = self._execute_one(state, stmt)
            if result is not None:
                remaining = stmts[i+1:]
                augmented = []
                for fork_state, fork_stmts in result:
                    augmented.append((fork_state, fork_stmts + remaining))
                return augmented
        return None

    def _execute_if(self, state: PathState, stmt: IfStmt) -> Optional[list]:
        """Fork execution at an if statement."""
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

        # Symbolic -- fork
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

        true_feasible = self._check_feasible(true_state)
        false_feasible = self._check_feasible(false_state)

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

        # Original state consumed by fork
        state.status = PathStatus.COMPLETED
        return forks if forks else []

    def _execute_while(self, state: PathState, stmt: WhileStmt) -> Optional[list]:
        """Unroll a while loop up to max_loop_unroll times."""
        forks = []
        current = state

        for iteration in range(self.max_loop_unroll):
            cond_val = self._eval_expr(current, stmt.cond)

            if cond_val.is_concrete():
                if not bool(cond_val.concrete):
                    current.covered_branches.append((stmt.line, False))
                    if forks:
                        return forks
                    return None
                current.covered_branches.append((stmt.line, True))
                result = self._execute_block_fork(current, stmt.body)
                if result:
                    forks.extend(result)
                    return forks
                if current.status != PathStatus.ACTIVE:
                    self.completed_paths.append(current)
                    return forks if forks else []
                continue

            # Symbolic -- fork exit vs continue
            cond_term = self._ensure_bool_term(cond_val)
            if cond_term is None:
                current.status = PathStatus.ERROR
                current.error_msg = "Cannot create boolean constraint for while"
                self.completed_paths.append(current)
                return forks if forks else []

            # Exit path
            exit_state = current.copy()
            exit_state.path_id = self._new_path_id()
            exit_state.constraints.append(smt_not(cond_term))
            exit_state.covered_branches.append((stmt.line, False))

            if self._check_feasible(exit_state):
                forks.append((exit_state, []))
            else:
                exit_state.status = PathStatus.INFEASIBLE
                self.completed_paths.append(exit_state)

            # Continue path
            current.constraints.append(cond_term)
            current.covered_branches.append((stmt.line, True))
            current.path_id = self._new_path_id()

            if not self._check_feasible(current):
                current.status = PathStatus.INFEASIBLE
                self.completed_paths.append(current)
                return forks if forks else []

            result = self._execute_block_fork(current, stmt.body)
            if result:
                forks.extend(result)
                return forks
            if current.status != PathStatus.ACTIVE:
                self.completed_paths.append(current)
                return forks if forks else []

        # Loop bound reached
        if current.status == PathStatus.ACTIVE:
            forks.append((current, []))
        return forks if forks else None

    # --- Expression Evaluation ---

    def _eval_expr(self, state: PathState, expr) -> SymValue:
        if isinstance(expr, IntLit):
            return SymValue.from_concrete(expr.value)
        if isinstance(expr, FloatLit):
            return SymValue.from_concrete(expr.value)
        if isinstance(expr, StringLit):
            return SymValue.from_concrete(expr.value)
        if isinstance(expr, BoolLit):
            return SymValue.from_concrete(expr.value)

        if isinstance(expr, ASTVar):
            if expr.name in state.env:
                return state.env[expr.name]
            state.status = PathStatus.ERROR
            state.error_msg = f"Undefined variable: {expr.name}"
            return SymValue.from_concrete(0)

        if isinstance(expr, UnaryOp):
            return self._eval_unary(state, expr)
        if isinstance(expr, BinOp):
            return self._eval_binop(state, expr)
        if isinstance(expr, Assign):
            val = self._eval_expr(state, expr.value)
            state.env[expr.name] = val
            return val
        if isinstance(expr, CallExpr):
            return self._eval_call(state, expr)

        return SymValue.from_concrete(None)

    def _eval_unary(self, state: PathState, expr: UnaryOp) -> SymValue:
        operand = self._eval_expr(state, expr.operand)
        if operand.is_concrete():
            if expr.op == '-':
                return SymValue.from_concrete(-operand.concrete)
            if expr.op == 'not':
                return SymValue.from_concrete(not operand.concrete)
            return SymValue.from_concrete(None)

        if expr.op == '-':
            return SymValue.from_symbolic(-operand.term)
        if expr.op == 'not':
            bool_term = self._ensure_bool_term(operand)
            if bool_term:
                return SymValue.from_symbolic(smt_not(bool_term))
        return SymValue.from_concrete(None)

    def _eval_binop(self, state: PathState, expr: BinOp) -> SymValue:
        left = self._eval_expr(state, expr.left)
        right = self._eval_expr(state, expr.right)

        if left.is_concrete() and right.is_concrete():
            return self._eval_concrete_binop(expr.op, left.concrete, right.concrete)

        # Lift to SMT
        left_term = self._to_int_term(left)
        right_term = self._to_int_term(right)

        # Boolean operations
        if expr.op in ('and', 'or'):
            lt = self._ensure_bool_term(left)
            rt = self._ensure_bool_term(right)
            if lt is not None and rt is not None:
                if expr.op == 'and':
                    return SymValue.from_symbolic(smt_and(lt, rt))
                else:
                    return SymValue.from_symbolic(smt_or(lt, rt))
            return SymValue.from_concrete(None)

        if left_term is None or right_term is None:
            return SymValue.from_concrete(None)

        # Arithmetic
        if expr.op == '+':
            return SymValue.from_symbolic(left_term + right_term)
        if expr.op == '-':
            return SymValue.from_symbolic(left_term - right_term)
        if expr.op == '*':
            return SymValue.from_symbolic(left_term * right_term)

        # Comparison
        if expr.op == '<':
            return SymValue.from_symbolic(left_term < right_term)
        if expr.op == '>':
            return SymValue.from_symbolic(left_term > right_term)
        if expr.op == '<=':
            return SymValue.from_symbolic(left_term <= right_term)
        if expr.op == '>=':
            return SymValue.from_symbolic(left_term >= right_term)
        if expr.op == '==':
            return SymValue.from_symbolic(App(SMTOp.EQ, [left_term, right_term], BOOL))
        if expr.op == '!=':
            return SymValue.from_symbolic(App(SMTOp.NEQ, [left_term, right_term], BOOL))

        return SymValue.from_concrete(None)

    def _eval_concrete_binop(self, op: str, left, right) -> SymValue:
        try:
            if op == '+': return SymValue.from_concrete(left + right)
            if op == '-': return SymValue.from_concrete(left - right)
            if op == '*': return SymValue.from_concrete(left * right)
            if op == '/':
                if right == 0: return SymValue.from_concrete(None)
                if isinstance(left, int) and isinstance(right, int):
                    return SymValue.from_concrete(left // right)
                return SymValue.from_concrete(left / right)
            if op == '%':
                if right == 0: return SymValue.from_concrete(None)
                return SymValue.from_concrete(left % right)
            if op == '<': return SymValue.from_concrete(left < right)
            if op == '>': return SymValue.from_concrete(left > right)
            if op == '<=': return SymValue.from_concrete(left <= right)
            if op == '>=': return SymValue.from_concrete(left >= right)
            if op == '==': return SymValue.from_concrete(left == right)
            if op == '!=': return SymValue.from_concrete(left != right)
            if op == 'and': return SymValue.from_concrete(left and right)
            if op == 'or': return SymValue.from_concrete(left or right)
        except Exception:
            pass
        return SymValue.from_concrete(None)

    def _eval_call(self, state: PathState, expr: CallExpr) -> SymValue:
        # Built-in assert
        if expr.callee == 'assert':
            if len(expr.args) > 0:
                val = self._eval_expr(state, expr.args[0])
                if val.is_concrete():
                    if not bool(val.concrete):
                        state.status = PathStatus.ASSERTION_FAILED
                        state.assertion_line = expr.line
                else:
                    cond_term = self._ensure_bool_term(val)
                    if cond_term is not None:
                        # Check if negation is feasible
                        neg_state = state.copy()
                        neg_state.constraints.append(smt_not(cond_term))
                        if self._check_feasible(neg_state):
                            fail_state = neg_state
                            fail_state.status = PathStatus.ASSERTION_FAILED
                            fail_state.assertion_line = expr.line
                            fail_state.path_id = self._new_path_id()
                            self.completed_paths.append(fail_state)
                        # Assume assertion holds on current path
                        state.constraints.append(cond_term)
            return SymValue.from_concrete(None)

        # User-defined function
        if expr.callee in self.functions:
            fn = self.functions[expr.callee]
            arg_vals = [self._eval_expr(state, a) for a in expr.args]

            # Run the function body, collecting all completed paths
            fn_results = self._run_function(state, fn, arg_vals)

            if len(fn_results) == 1:
                # Single path -- continue normally
                result_state, ret_val = fn_results[0]
                # Update state to match result_state
                state.env = result_state.env
                state.constraints = result_state.constraints
                state.covered_lines = result_state.covered_lines
                state.covered_branches = result_state.covered_branches
                state.output = result_state.output
                state.status = PathStatus.ACTIVE  # Keep active for caller
                return ret_val

            if len(fn_results) == 0:
                return SymValue.from_concrete(None)

            # Multiple paths -- fork via signal
            raise ForkSignal(fn_results)

        return SymValue.from_concrete(None)

    def _apply_fn_forks(self, fn_results: list, var_name: str) -> list:
        """Apply assignment to each fork from a function call."""
        forks = []
        for fn_state, ret_val in fn_results:
            fn_state.env[var_name] = ret_val
            fn_state.status = PathStatus.ACTIVE  # Reset for continued execution
            forks.append((fn_state, []))
        return forks

    def _run_function(self, caller_state: PathState, fn: FnDecl,
                       arg_vals: list) -> list:
        """
        Run a function body to completion, exploring all internal paths.
        Returns list of (state, return_value) pairs.
        """
        old_env = dict(caller_state.env)

        # Set up function params
        fn_state = caller_state.copy()
        fn_state.path_id = self._new_path_id()
        for param, arg_val in zip(fn.params, arg_vals):
            fn_state.env[param] = arg_val

        # Temporarily redirect completed_paths during function execution
        saved_completed = self.completed_paths
        self.completed_paths = []

        fn_completed = []
        fn_worklist = [(fn_state, fn.body.stmts if isinstance(fn.body, Block) else [fn.body])]

        while fn_worklist:
            state, stmts = fn_worklist.pop(0)
            try:
                forks = self._execute_fn_stmts(state, stmts)
                for fork_state, fork_stmts in forks:
                    if fork_state.status == PathStatus.ACTIVE:
                        fn_worklist.append((fork_state, fork_stmts))
            except ReturnSignal as rs:
                state.return_value = rs.value
                fn_completed.append((state, rs.value))

        # Paths that completed without explicit return (fell off end)
        for p in self.completed_paths:
            if p.status == PathStatus.COMPLETED:
                fn_completed.append((p, p.return_value or SymValue.from_concrete(None)))

        # Restore completed_paths
        self.completed_paths = saved_completed

        # Restore params for each completed state
        results = []
        for state, ret_val in fn_completed:
            self._restore_params(state, fn.params, old_env)
            results.append((state, ret_val))

        return results

    def _execute_fn_stmts(self, state: PathState, stmts: list) -> list:
        """Execute statements in function body context. Like _execute_stmts but
        doesn't add to completed_paths."""
        forks = []
        i = 0
        while i < len(stmts):
            if state.status != PathStatus.ACTIVE:
                break
            stmt = stmts[i]
            if hasattr(stmt, 'line') and stmt.line > 0:
                state.covered_lines.add(stmt.line)
            result = self._execute_one(state, stmt)
            if result is not None:
                remaining = stmts[i+1:]
                for fork_state, fork_stmts in result:
                    forks.append((fork_state, fork_stmts + remaining))
                return forks
            i += 1
        return forks

    def _restore_params(self, state: PathState, params: list, old_env: dict):
        """Restore function parameters after call."""
        for param in params:
            if param in old_env:
                state.env[param] = old_env[param]
            elif param in state.env:
                del state.env[param]

    # --- Term conversion helpers ---

    def _to_int_term(self, val: SymValue) -> Optional[Term]:
        """Convert SymValue to an integer SMT term."""
        if val.is_symbolic():
            return val.term
        if val.is_concrete():
            if isinstance(val.concrete, bool):
                return IntConst(1 if val.concrete else 0)
            if isinstance(val.concrete, (int, float)):
                return IntConst(int(val.concrete))
        return None

    def _ensure_bool_term(self, val: SymValue) -> Optional[Term]:
        """Convert SymValue to a boolean SMT term."""
        if val.is_symbolic():
            term = val.term
            # If term is already boolean (comparison/logical), use directly
            if isinstance(term, App) and term.sort == BOOL:
                return term
            if isinstance(term, SMTVar) and term.sort == BOOL:
                return term
            # Integer variable used as boolean: treat as != 0
            if isinstance(term, (SMTVar, App)) and term.sort == INT:
                return App(SMTOp.NEQ, [term, IntConst(0)], BOOL)
            return term  # hope for the best
        return None

    # --- Feasibility checking ---

    def _check_feasible(self, state: PathState) -> bool:
        if not state.constraints:
            return True
        solver = SMTSolver()
        self._declare_constraint_vars(solver, state.constraints)
        for c in state.constraints:
            solver.add(c)
        return solver.check() == SMTResult.SAT

    def _declare_constraint_vars(self, solver: SMTSolver, constraints: list):
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

    def _declare_vars(self, solver: SMTSolver, symbolic_inputs: dict):
        for name, typ in symbolic_inputs.items():
            if typ == 'int':
                solver.Int(name)
            elif typ == 'bool':
                solver.Bool(name)

    # --- Test Generation ---

    def _generate_tests(self, symbolic_inputs: dict) -> list:
        tests = []
        for path in self.completed_paths:
            if path.status == PathStatus.INFEASIBLE:
                continue
            if not symbolic_inputs:
                tests.append(TestCase(
                    path_id=path.path_id,
                    inputs={},
                    path_status=path.status,
                    output=path.output,
                    covered_lines=path.covered_lines,
                    branches=path.covered_branches,
                ))
                continue

            if not path.constraints:
                inputs = {}
                for name, typ in symbolic_inputs.items():
                    inputs[name] = 0 if typ == 'int' else False
                tests.append(TestCase(
                    path_id=path.path_id,
                    inputs=inputs,
                    path_status=path.status,
                    output=path.output,
                    covered_lines=path.covered_lines,
                    branches=path.covered_branches,
                ))
                continue

            solver = SMTSolver()
            self._declare_vars(solver, symbolic_inputs)
            self._declare_constraint_vars(solver, path.constraints)
            for c in path.constraints:
                solver.add(c)
            result = solver.check()
            if result == SMTResult.SAT:
                model = solver.model()
                inputs = {}
                for name in symbolic_inputs:
                    if model and name in model:
                        inputs[name] = model[name]
                    else:
                        inputs[name] = 0 if symbolic_inputs[name] == 'int' else False
                tests.append(TestCase(
                    path_id=path.path_id,
                    inputs=inputs,
                    path_status=path.status,
                    output=path.output,
                    covered_lines=path.covered_lines,
                    branches=path.covered_branches,
                ))
        return tests


# ============================================================
# Result Types
# ============================================================

@dataclass
class TestCase:
    path_id: int
    inputs: dict
    path_status: PathStatus
    output: list = field(default_factory=list)
    covered_lines: set = field(default_factory=set)
    branches: list = field(default_factory=list)


@dataclass
class ExecutionResult:
    paths: list
    test_cases: list
    functions: list = field(default_factory=list)

    @property
    def feasible_paths(self):
        return [p for p in self.paths if p.status != PathStatus.INFEASIBLE]

    @property
    def total_paths(self):
        return len(self.paths)

    @property
    def total_test_cases(self):
        return len(self.test_cases)


@dataclass
class AssertionViolation:
    line: int
    path_constraints: list
    counterexample: Optional[dict] = None


@dataclass
class AssertionResult:
    holds: bool
    violations: list
    paths_explored: int = 0


@dataclass
class CoverageResult:
    total_lines: int
    covered_lines: int
    line_coverage: float
    total_branches: int
    covered_branches: int
    branch_coverage: float
    paths_explored: int
    feasible_paths: int
    dead_branches: list = field(default_factory=list)


# ============================================================
# Convenience Functions
# ============================================================

def symbolic_execute(source: str, symbolic_inputs: dict = None,
                     max_paths=64, max_loop_unroll=5) -> ExecutionResult:
    engine = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    return engine.execute(source, symbolic_inputs)

def generate_tests(source: str, symbolic_inputs: dict,
                   max_paths=64) -> list:
    engine = SymbolicExecutor(max_paths=max_paths)
    result = engine.execute(source, symbolic_inputs)
    return result.test_cases

def check_assertions(source: str, symbolic_inputs: dict = None) -> AssertionResult:
    engine = SymbolicExecutor()
    return engine.check_assertions(source, symbolic_inputs)

def find_inputs_for_line(source: str, symbolic_inputs: dict,
                         target_line: int) -> Optional[dict]:
    engine = SymbolicExecutor()
    return engine.find_inputs(source, symbolic_inputs, target_line)

def get_coverage(source: str, symbolic_inputs: dict = None) -> CoverageResult:
    engine = SymbolicExecutor()
    return engine.get_coverage(source, symbolic_inputs)
