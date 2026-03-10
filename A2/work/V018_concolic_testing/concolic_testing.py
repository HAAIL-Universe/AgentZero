"""
V018: Concolic Testing Engine
Composes C038 (Symbolic Execution) + C037 (SMT Solver) + C010 (Parser)

Concolic = CONCrete + symbOLIC execution.

Architecture:
  1. Start with concrete input values
  2. Execute program concretely, collecting symbolic path constraints along the way
  3. Negate a path constraint to explore a new branch
  4. Solve with SMT to get a new concrete input
  5. Repeat until coverage saturates or budget exhausted

Key advantage over pure symbolic execution (C038):
  - Concrete execution handles complex operations (nonlinear, string ops) as fallback
  - Directed path exploration via constraint negation (not exhaustive forking)
  - Coverage-guided prioritization: prefers inputs reaching uncovered branches
  - Scales better for programs with many paths

Composes:
  - C010: Parser (source -> AST)
  - C037: SMT solver (constraint solving for new inputs)
  - C038: Symbolic execution engine (path constraint collection)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var as SMTVar, App,
    IntConst, BoolConst, Op as SMTOp, BOOL, INT
)

# Import C038 symbolic execution helpers
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymValue, SymType, PathState, PathStatus,
    smt_not, smt_and, smt_or, SymbolicExecutor
)


# ============================================================
# Parse helper
# ============================================================

def parse(source: str) -> Program:
    """Parse source code into AST."""
    tokens = lex(source)
    return Parser(tokens).parse()


# ============================================================
# Concrete Interpreter (with symbolic shadow)
# ============================================================

class ConcreteInterpreter:
    """
    Executes a program concretely while maintaining a symbolic shadow state.
    At each branch point, records the symbolic condition and the concrete direction taken.
    """

    def __init__(self, inputs: Dict[str, int]):
        self.inputs = inputs
        self.env: Dict[str, int] = {}           # concrete environment
        self.sym_env: Dict[str, Term] = {}      # symbolic shadow
        self.path_constraints: List[Term] = []  # collected branch conditions
        self.branch_decisions: List[bool] = []  # True = then-branch taken
        self.branch_locations: List[int] = []   # which branch point (index)
        self.output: List[Any] = []
        self.covered_branches: Set[Tuple[int, bool]] = set()  # (branch_id, direction)
        self.functions: Dict[str, FnDecl] = {}
        self._branch_counter = 0
        self._sym_vars: Dict[str, SMTVar] = {}

    def _get_sym_var(self, name: str) -> SMTVar:
        if name not in self._sym_vars:
            self._sym_vars[name] = SMTVar(name, INT)
        return self._sym_vars[name]

    def execute(self, program: Program):
        """Execute a parsed program concretely."""
        for stmt in program.stmts:
            self._exec_stmt(stmt)

    def _exec_stmt(self, stmt):
        if isinstance(stmt, LetDecl):
            val = self._eval_expr(stmt.value)
            sym = self._eval_sym(stmt.value)
            self.env[stmt.name] = val
            self.sym_env[stmt.name] = sym

        elif isinstance(stmt, Assign):
            val = self._eval_expr(stmt.value)
            sym = self._eval_sym(stmt.value)
            self.env[stmt.name] = val
            self.sym_env[stmt.name] = sym

        elif isinstance(stmt, IfStmt):
            cond_val = self._eval_expr(stmt.cond)
            cond_sym = self._eval_sym(stmt.cond)
            branch_id = self._branch_counter
            self._branch_counter += 1

            took_then = bool(cond_val)

            # Record constraint
            if cond_sym is not None:
                constraint = cond_sym if took_then else smt_not(cond_sym)
                self.path_constraints.append(constraint)
                self.branch_decisions.append(took_then)
                self.branch_locations.append(branch_id)

            self.covered_branches.add((branch_id, took_then))

            if took_then:
                body = stmt.then_body
                if isinstance(body, Block):
                    for s in body.stmts:
                        self._exec_stmt(s)
                elif isinstance(body, list):
                    for s in body:
                        self._exec_stmt(s)
                else:
                    self._exec_stmt(body)
            else:
                body = stmt.else_body
                if body is not None:
                    if isinstance(body, Block):
                        for s in body.stmts:
                            self._exec_stmt(s)
                    elif isinstance(body, list):
                        for s in body:
                            self._exec_stmt(s)
                    else:
                        self._exec_stmt(body)

        elif isinstance(stmt, WhileStmt):
            loop_limit = 50
            iterations = 0
            while iterations < loop_limit:
                cond_val = self._eval_expr(stmt.cond)
                cond_sym = self._eval_sym(stmt.cond)
                branch_id = self._branch_counter
                self._branch_counter += 1

                took_then = bool(cond_val)
                if cond_sym is not None:
                    constraint = cond_sym if took_then else smt_not(cond_sym)
                    self.path_constraints.append(constraint)
                    self.branch_decisions.append(took_then)
                    self.branch_locations.append(branch_id)
                self.covered_branches.add((branch_id, took_then))

                if not took_then:
                    break
                body = stmt.body
                if isinstance(body, Block):
                    for s in body.stmts:
                        self._exec_stmt(s)
                elif isinstance(body, list):
                    for s in body:
                        self._exec_stmt(s)
                else:
                    self._exec_stmt(body)
                iterations += 1

        elif isinstance(stmt, PrintStmt):
            val = self._eval_expr(stmt.value)
            self.output.append(val)

        elif isinstance(stmt, FnDecl):
            self.functions[stmt.name] = stmt

        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self._exec_stmt(s)

        elif isinstance(stmt, ReturnStmt):
            pass  # top-level return, ignore

        elif isinstance(stmt, CallExpr):
            self._eval_expr(stmt)

    def _eval_expr(self, expr) -> Any:
        """Evaluate expression concretely."""
        if isinstance(expr, IntLit):
            return expr.value
        elif isinstance(expr, FloatLit):
            return expr.value
        elif isinstance(expr, BoolLit):
            return 1 if expr.value else 0
        elif isinstance(expr, StringLit):
            return expr.value
        elif isinstance(expr, ASTVar):
            if expr.name in self.env:
                return self.env[expr.name]
            elif expr.name in self.inputs:
                return self.inputs[expr.name]
            return 0

        elif isinstance(expr, UnaryOp):
            val = self._eval_expr(expr.operand)
            if expr.op == '-':
                return -val
            elif expr.op == 'not':
                return 0 if val else 1
            return val

        elif isinstance(expr, BinOp):
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            op = expr.op
            if op == '+': return left + right
            elif op == '-': return left - right
            elif op == '*': return left * right
            elif op == '/': return left // right if right != 0 else 0
            elif op == '%': return left % right if right != 0 else 0
            elif op == '==': return 1 if left == right else 0
            elif op == '!=': return 1 if left != right else 0
            elif op == '<': return 1 if left < right else 0
            elif op == '>': return 1 if left > right else 0
            elif op == '<=': return 1 if left <= right else 0
            elif op == '>=': return 1 if left >= right else 0
            elif op == 'and': return 1 if (left and right) else 0
            elif op == 'or': return 1 if (left or right) else 0
            return 0

        elif isinstance(expr, CallExpr):
            fn_name = expr.callee
            if fn_name in self.functions:
                fn = self.functions[fn_name]
                args = [self._eval_expr(a) for a in expr.args]
                # Save env, execute function body, restore
                saved_env = dict(self.env)
                saved_sym = dict(self.sym_env)
                for param, arg_val in zip(fn.params, args):
                    self.env[param] = arg_val
                    sym_args = [self._eval_sym(a) for a in expr.args]
                    self.sym_env[param] = sym_args[fn.params.index(param)]
                result = 0
                for s in fn.body.stmts if isinstance(fn.body, Block) else [fn.body]:
                    if isinstance(s, ReturnStmt):
                        result = self._eval_expr(s.value)
                        break
                    self._exec_stmt(s)
                self.env = saved_env
                self.sym_env = saved_sym
                return result
            return 0

        return 0

    def _eval_sym(self, expr) -> Optional[Term]:
        """Evaluate expression symbolically (shadow execution)."""
        if isinstance(expr, IntLit):
            return IntConst(expr.value)
        elif isinstance(expr, FloatLit):
            return IntConst(int(expr.value))
        elif isinstance(expr, BoolLit):
            return IntConst(1 if expr.value else 0)
        elif isinstance(expr, StringLit):
            return None  # Can't symbolically reason about strings

        elif isinstance(expr, ASTVar):
            if expr.name in self.sym_env:
                return self.sym_env[expr.name]
            elif expr.name in self.inputs:
                return self._get_sym_var(expr.name)
            return None

        elif isinstance(expr, UnaryOp):
            val = self._eval_sym(expr.operand)
            if val is None:
                return None
            if expr.op == '-':
                return App(SMTOp.SUB, [IntConst(0), val], INT)
            elif expr.op == 'not':
                return smt_not(val)
            return val

        elif isinstance(expr, BinOp):
            left = self._eval_sym(expr.left)
            right = self._eval_sym(expr.right)
            if left is None or right is None:
                return None
            op = expr.op
            op_map = {
                '+': (SMTOp.ADD, INT),
                '-': (SMTOp.SUB, INT),
                '*': (SMTOp.MUL, INT),
                '==': (SMTOp.EQ, BOOL),
                '!=': (SMTOp.NEQ, BOOL),
                '<': (SMTOp.LT, BOOL),
                '>': (SMTOp.GT, BOOL),
                '<=': (SMTOp.LE, BOOL),
                '>=': (SMTOp.GE, BOOL),
            }
            if op in op_map:
                smt_op, sort = op_map[op]
                return App(smt_op, [left, right], sort)
            # and/or
            if op == 'and':
                return smt_and(left, right)
            if op == 'or':
                return smt_or(left, right)
            return None

        elif isinstance(expr, CallExpr):
            # For function calls, try to get symbolic return value
            fn_name = expr.callee
            if fn_name in self.functions:
                fn = self.functions[fn_name]
                sym_args = [self._eval_sym(a) for a in expr.args]
                if any(a is None for a in sym_args):
                    return None
                # Inline symbolically
                saved_sym = dict(self.sym_env)
                for param, sym_val in zip(fn.params, sym_args):
                    self.sym_env[param] = sym_val
                result = None
                for s in fn.body.stmts if isinstance(fn.body, Block) else [fn.body]:
                    if isinstance(s, ReturnStmt):
                        result = self._eval_sym(s.value)
                        break
                    elif isinstance(s, LetDecl):
                        sym = self._eval_sym(s.value)
                        if sym is not None:
                            self.sym_env[s.name] = sym
                    elif isinstance(s, Assign):
                        sym = self._eval_sym(s.value)
                        if sym is not None:
                            self.sym_env[s.name] = sym
                    elif isinstance(s, IfStmt):
                        # For function-level if, we can't easily track symbolically
                        return None
                self.sym_env = saved_sym
                return result
            return None

        return None


# ============================================================
# Path Constraint for Negation
# ============================================================

@dataclass
class BranchPoint:
    """A branch point with its constraint and direction."""
    constraint: Term          # symbolic condition
    took_then: bool           # direction taken in concrete execution
    branch_id: int            # unique branch identifier
    index: int                # index in path_constraints list


# ============================================================
# Concolic Result
# ============================================================

class ConcolicStatus(Enum):
    COMPLETE = "complete"           # All reachable paths explored
    BUDGET_EXHAUSTED = "budget"     # Hit iteration limit
    COVERAGE_SATURATED = "saturated"  # No new coverage in N iterations


@dataclass
class ConcolicTestCase:
    """A test case generated by concolic testing."""
    inputs: Dict[str, int]
    output: List[Any]
    path_constraints: List[Term]
    branches_covered: Set[Tuple[int, bool]]
    is_new_coverage: bool = False    # Did this test cover new branches?


@dataclass
class ConcolicResult:
    """Result of concolic testing."""
    test_cases: List[ConcolicTestCase]
    total_branches: int
    covered_branches: Set[Tuple[int, bool]]
    branch_coverage: float
    iterations: int
    status: ConcolicStatus
    all_outputs: List[List[Any]]     # output per test case

    @property
    def num_tests(self):
        return len(self.test_cases)


# ============================================================
# Concolic Testing Engine
# ============================================================

class ConcolicEngine:
    """
    Concolic testing: concrete execution with symbolic shadow.

    Strategy:
    1. Start with initial concrete inputs
    2. Execute concretely, collecting path constraints
    3. Negate one constraint to explore a new path
    4. Use SMT to solve for new inputs
    5. Repeat with coverage-guided prioritization
    """

    def __init__(self, max_iterations=50, max_loop_iter=20,
                 saturation_limit=5):
        self.max_iterations = max_iterations
        self.max_loop_iter = max_loop_iter
        self.saturation_limit = saturation_limit  # stop after N iterations with no new coverage

    def run(self, source: str, input_vars: Dict[str, str],
            initial_inputs: Optional[Dict[str, int]] = None) -> ConcolicResult:
        """
        Run concolic testing on a program.

        Args:
            source: Program source code
            input_vars: Map of input variable names to types (e.g. {'x': 'int', 'y': 'int'})
            initial_inputs: Starting concrete inputs (default: all zeros)

        Returns:
            ConcolicResult with test cases and coverage
        """
        program = parse(source)
        var_names = list(input_vars.keys())

        # Initialize with default inputs if not provided
        if initial_inputs is None:
            initial_inputs = {name: 0 for name in var_names}

        test_cases: List[ConcolicTestCase] = []
        all_covered: Set[Tuple[int, bool]] = set()
        total_branches = 0

        # Work queue: (inputs, constraint_prefix_to_negate_index)
        # Each entry is a concrete input to try
        work_queue: List[Dict[str, int]] = [initial_inputs]
        seen_inputs: Set[tuple] = set()
        # Track which (prefix, negated_index) pairs we've tried
        tried_negations: Set[tuple] = set()

        iterations = 0
        no_progress_count = 0

        while work_queue and iterations < self.max_iterations:
            current_inputs = work_queue.pop(0)

            # Deduplicate inputs
            input_key = tuple(sorted(current_inputs.items()))
            if input_key in seen_inputs:
                continue
            seen_inputs.add(input_key)

            iterations += 1

            # Execute concretely with symbolic shadow
            interp = ConcreteInterpreter(current_inputs)
            interp.execute(program)

            # Record test case
            prev_coverage = len(all_covered)
            new_branches = interp.covered_branches - all_covered
            all_covered |= interp.covered_branches

            tc = ConcolicTestCase(
                inputs=dict(current_inputs),
                output=list(interp.output),
                path_constraints=list(interp.path_constraints),
                branches_covered=set(interp.covered_branches),
                is_new_coverage=len(new_branches) > 0
            )
            test_cases.append(tc)

            # Track total branches seen
            for bid, _ in interp.covered_branches:
                total_branches = max(total_branches, bid + 1)

            # Check progress
            if len(all_covered) > prev_coverage:
                no_progress_count = 0
            else:
                no_progress_count += 1
                if no_progress_count >= self.saturation_limit:
                    break

            # Generate new inputs by negating path constraints
            constraints = interp.path_constraints
            decisions = interp.branch_decisions

            for i in range(len(constraints)):
                # Try negating constraint i while keeping 0..i-1
                neg_key = tuple(str(c) for c in constraints[:i]) + (str(i),)
                if neg_key in tried_negations:
                    continue
                tried_negations.add(neg_key)

                # Check if the negated direction would be new coverage
                branch_id = interp.branch_locations[i] if i < len(interp.branch_locations) else -1
                negated_dir = not decisions[i]
                # Prioritize uncovered branches
                if (branch_id, negated_dir) in all_covered:
                    # Already covered, lower priority -- add to end
                    new_inputs = self._solve_negated(
                        constraints[:i], decisions[:i], constraints[i],
                        decisions[i], var_names
                    )
                    if new_inputs is not None:
                        work_queue.append(new_inputs)
                else:
                    # Uncovered -- high priority, add to front
                    new_inputs = self._solve_negated(
                        constraints[:i], decisions[:i], constraints[i],
                        decisions[i], var_names
                    )
                    if new_inputs is not None:
                        work_queue.insert(0, new_inputs)

        # Determine status
        if no_progress_count >= self.saturation_limit:
            status = ConcolicStatus.COVERAGE_SATURATED
        elif not work_queue:
            status = ConcolicStatus.COMPLETE
        else:
            status = ConcolicStatus.BUDGET_EXHAUSTED

        total_possible = total_branches * 2 if total_branches > 0 else 1
        coverage = len(all_covered) / total_possible if total_possible > 0 else 1.0

        return ConcolicResult(
            test_cases=test_cases,
            total_branches=total_branches,
            covered_branches=all_covered,
            branch_coverage=coverage,
            iterations=iterations,
            status=status,
            all_outputs=[tc.output for tc in test_cases]
        )

    def _solve_negated(self, prefix_constraints: List[Term],
                       prefix_decisions: List[bool],
                       target_constraint: Term,
                       target_decision: bool,
                       var_names: List[str]) -> Optional[Dict[str, int]]:
        """
        Solve for inputs that follow the prefix path but negate the target constraint.

        prefix_constraints[0..n-1] + NOT(target_constraint)
        """
        solver = SMTSolver()
        sym_vars = {}
        for name in var_names:
            sym_vars[name] = solver.Int(name)

        # Assert prefix constraints (same direction as concrete execution)
        for constraint in prefix_constraints:
            registered = self._register_terms(solver, constraint, sym_vars)
            solver.add(registered)

        # Negate the target constraint
        negated = smt_not(target_constraint)
        registered = self._register_terms(solver, negated, sym_vars)
        solver.add(registered)

        result = solver.check()
        if result == SMTResult.SAT:
            model = solver.model()
            inputs = {}
            for name in var_names:
                if name in model:
                    inputs[name] = model[name]
                else:
                    inputs[name] = 0
            return inputs
        return None

    def _register_terms(self, solver: SMTSolver, term: Term,
                        sym_vars: Dict[str, SMTVar]) -> Term:
        """
        Replace free variables in a term with solver-registered variables.
        """
        if isinstance(term, SMTVar):
            if term.name in sym_vars:
                return sym_vars[term.name]
            # Create new var
            sym_vars[term.name] = solver.Int(term.name)
            return sym_vars[term.name]
        elif isinstance(term, (IntConst, BoolConst)):
            return term
        elif isinstance(term, App):
            new_args = [self._register_terms(solver, a, sym_vars) for a in term.args]
            return App(term.op, new_args, term.sort)
        return term


# ============================================================
# Coverage-Guided Concolic Testing
# ============================================================

class CoverageGuidedConcolic:
    """
    Enhanced concolic testing with coverage-guided input generation.
    Uses a fitness function to prioritize inputs that are closest to
    uncovered branches.
    """

    def __init__(self, max_iterations=100, seeds=None):
        self.max_iterations = max_iterations
        self.seeds = seeds or []

    def run(self, source: str, input_vars: Dict[str, str],
            initial_inputs: Optional[Dict[str, int]] = None) -> ConcolicResult:
        """Run coverage-guided concolic testing with multiple seeds."""
        engine = ConcolicEngine(
            max_iterations=self.max_iterations,
            saturation_limit=10
        )

        # Start with provided initial inputs
        result = engine.run(source, input_vars, initial_inputs)

        # If we have seeds, try them too
        if self.seeds:
            for seed in self.seeds:
                if result.branch_coverage >= 1.0:
                    break
                seed_result = engine.run(source, input_vars, seed)
                # Merge results
                result = self._merge_results(result, seed_result)

        return result

    def _merge_results(self, r1: ConcolicResult, r2: ConcolicResult) -> ConcolicResult:
        """Merge two concolic results."""
        all_tests = r1.test_cases + r2.test_cases
        all_covered = r1.covered_branches | r2.covered_branches
        total_branches = max(r1.total_branches, r2.total_branches)
        total_possible = total_branches * 2 if total_branches > 0 else 1
        coverage = len(all_covered) / total_possible if total_possible > 0 else 1.0

        return ConcolicResult(
            test_cases=all_tests,
            total_branches=total_branches,
            covered_branches=all_covered,
            branch_coverage=coverage,
            iterations=r1.iterations + r2.iterations,
            status=r1.status,
            all_outputs=[tc.output for tc in all_tests]
        )


# ============================================================
# Concolic-based Bug Finding
# ============================================================

@dataclass
class BugReport:
    """A bug found by concolic testing."""
    kind: str               # "division_by_zero", "assertion_failure", etc.
    inputs: Dict[str, int]
    description: str
    path_constraints: List[Term]


@dataclass
class BugFindingResult:
    """Result of concolic bug finding."""
    bugs: List[BugReport]
    test_cases: List[ConcolicTestCase]
    iterations: int
    coverage: float


class ConcolicBugFinder:
    """
    Uses concolic testing to find bugs (division by zero, assertion failures,
    out-of-bound values).
    """

    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations

    def find_bugs(self, source: str, input_vars: Dict[str, str],
                  initial_inputs: Optional[Dict[str, int]] = None,
                  assertions: Optional[Dict[str, Term]] = None) -> BugFindingResult:
        """
        Find bugs using concolic testing.

        assertions: optional map of assertion names to SMT conditions that should hold
        """
        engine = ConcolicEngine(max_iterations=self.max_iterations)
        result = engine.run(source, input_vars, initial_inputs)

        bugs = []

        # Check each test case for bugs
        for tc in result.test_cases:
            # Check division by zero
            program = parse(source)
            div_zero_inputs = self._check_division_by_zero(program, tc.inputs)
            if div_zero_inputs:
                bugs.append(BugReport(
                    kind="division_by_zero",
                    inputs=tc.inputs,
                    description=f"Division by zero with inputs {tc.inputs}",
                    path_constraints=tc.path_constraints
                ))

            # Check assertions
            if assertions:
                for name, condition in assertions.items():
                    if not self._check_assertion(condition, tc.inputs, input_vars):
                        bugs.append(BugReport(
                            kind="assertion_failure",
                            inputs=tc.inputs,
                            description=f"Assertion '{name}' failed with inputs {tc.inputs}",
                            path_constraints=tc.path_constraints
                        ))

        # Deduplicate bugs
        seen = set()
        unique_bugs = []
        for bug in bugs:
            key = (bug.kind, tuple(sorted(bug.inputs.items())))
            if key not in seen:
                seen.add(key)
                unique_bugs.append(bug)

        return BugFindingResult(
            bugs=unique_bugs,
            test_cases=result.test_cases,
            iterations=result.iterations,
            coverage=result.branch_coverage
        )

    def _check_division_by_zero(self, program: Program, inputs: Dict[str, int]) -> bool:
        """Check if executing with these inputs causes division by zero."""
        return self._has_div_by_zero(program.stmts, inputs)

    def _has_div_by_zero(self, stmts, env: Dict[str, int]) -> bool:
        """Recursively check for division by zero."""
        local_env = dict(env)
        for stmt in stmts:
            if isinstance(stmt, LetDecl):
                if self._expr_div_zero(stmt.value, local_env):
                    return True
                local_env[stmt.name] = self._concrete_eval(stmt.value, local_env)
            elif isinstance(stmt, Assign):
                if self._expr_div_zero(stmt.value, local_env):
                    return True
                local_env[stmt.name] = self._concrete_eval(stmt.value, local_env)
            elif isinstance(stmt, IfStmt):
                cond = self._concrete_eval(stmt.cond, local_env)
                if cond:
                    body = stmt.then_body
                    if isinstance(body, Block):
                        if self._has_div_by_zero(body.stmts, local_env):
                            return True
                else:
                    if stmt.else_body:
                        body = stmt.else_body
                        if isinstance(body, Block):
                            if self._has_div_by_zero(body.stmts, local_env):
                                return True
        return False

    def _expr_div_zero(self, expr, env: Dict[str, int]) -> bool:
        """Check if expression performs division by zero."""
        if isinstance(expr, BinOp):
            if expr.op in ('/', '%'):
                right = self._concrete_eval(expr.right, env)
                if right == 0:
                    return True
            return self._expr_div_zero(expr.left, env) or self._expr_div_zero(expr.right, env)
        return False

    def _concrete_eval(self, expr, env: Dict[str, int]) -> int:
        """Simple concrete expression evaluation."""
        if isinstance(expr, IntLit):
            return expr.value
        elif isinstance(expr, BoolLit):
            return 1 if expr.value else 0
        elif isinstance(expr, ASTVar):
            return env.get(expr.name, 0)
        elif isinstance(expr, UnaryOp):
            val = self._concrete_eval(expr.operand, env)
            if expr.op == '-': return -val
            if expr.op == 'not': return 0 if val else 1
            return val
        elif isinstance(expr, BinOp):
            left = self._concrete_eval(expr.left, env)
            right = self._concrete_eval(expr.right, env)
            op = expr.op
            if op == '+': return left + right
            if op == '-': return left - right
            if op == '*': return left * right
            if op == '/': return left // right if right != 0 else 0
            if op == '%': return left % right if right != 0 else 0
            if op == '==': return 1 if left == right else 0
            if op == '!=': return 1 if left != right else 0
            if op == '<': return 1 if left < right else 0
            if op == '>': return 1 if left > right else 0
            if op == '<=': return 1 if left <= right else 0
            if op == '>=': return 1 if left >= right else 0
            if op == 'and': return 1 if (left and right) else 0
            if op == 'or': return 1 if (left or right) else 0
        return 0

    def _check_assertion(self, condition: Term, inputs: Dict[str, int],
                         input_vars: Dict[str, str]) -> bool:
        """Check if an assertion holds for given inputs."""
        solver = SMTSolver()
        sym_vars = {}
        for name in input_vars:
            sym_vars[name] = solver.Int(name)

        # Assert the concrete values
        for name, val in inputs.items():
            if name in sym_vars:
                solver.add(App(SMTOp.EQ, [sym_vars[name], IntConst(val)], BOOL))

        # Check if NOT(condition) is satisfiable
        neg_cond = smt_not(self._register_terms_static(condition, sym_vars))
        solver.add(neg_cond)
        result = solver.check()
        return result != SMTResult.SAT  # assertion holds if negation is UNSAT

    def _register_terms_static(self, term: Term, sym_vars: Dict[str, SMTVar]) -> Term:
        """Replace vars with registered vars."""
        if isinstance(term, SMTVar):
            if term.name in sym_vars:
                return sym_vars[term.name]
            return term
        elif isinstance(term, App):
            new_args = [self._register_terms_static(a, sym_vars) for a in term.args]
            return App(term.op, new_args, term.sort)
        return term


# ============================================================
# Comparison: Concolic vs Pure Symbolic
# ============================================================

def compare_concolic_vs_symbolic(source: str, input_vars: Dict[str, str],
                                  initial_inputs: Optional[Dict[str, int]] = None) -> dict:
    """
    Compare concolic testing against pure symbolic execution (C038).

    Returns dict with coverage and test count comparisons.
    """
    # Concolic
    engine = ConcolicEngine(max_iterations=30)
    concolic_result = engine.run(source, input_vars, initial_inputs)

    # Pure symbolic
    sym_engine = SymbolicExecutor(max_paths=64)
    sym_result = sym_engine.execute(source, input_vars)

    return {
        'concolic': {
            'tests': concolic_result.num_tests,
            'coverage': concolic_result.branch_coverage,
            'iterations': concolic_result.iterations,
            'branches_covered': len(concolic_result.covered_branches),
        },
        'symbolic': {
            'tests': sym_result.total_test_cases,
            'paths': sym_result.total_paths,
            'feasible_paths': len(sym_result.feasible_paths),
        },
        'concolic_tests': concolic_result.test_cases,
        'symbolic_tests': sym_result.test_cases,
    }


# ============================================================
# Convenience Functions
# ============================================================

def concolic_test(source: str, input_vars: Dict[str, str],
                  initial_inputs: Optional[Dict[str, int]] = None,
                  max_iterations: int = 50) -> ConcolicResult:
    """Run concolic testing on a program."""
    engine = ConcolicEngine(max_iterations=max_iterations)
    return engine.run(source, input_vars, initial_inputs)


def concolic_find_bugs(source: str, input_vars: Dict[str, str],
                       initial_inputs: Optional[Dict[str, int]] = None) -> BugFindingResult:
    """Find bugs using concolic testing."""
    finder = ConcolicBugFinder()
    return finder.find_bugs(source, input_vars, initial_inputs)


def concolic_with_seeds(source: str, input_vars: Dict[str, str],
                        seeds: List[Dict[str, int]],
                        max_iterations: int = 100) -> ConcolicResult:
    """Run coverage-guided concolic testing with seed inputs."""
    guided = CoverageGuidedConcolic(max_iterations=max_iterations, seeds=seeds)
    return guided.run(source, input_vars)


def concolic_reach_branch(source: str, input_vars: Dict[str, str],
                          target_branch: int, target_direction: bool,
                          initial_inputs: Optional[Dict[str, int]] = None,
                          max_iterations: int = 50) -> Optional[Dict[str, int]]:
    """
    Try to find inputs that reach a specific branch in a specific direction.

    Returns inputs if found, None otherwise.
    """
    result = concolic_test(source, input_vars, initial_inputs, max_iterations)
    for tc in result.test_cases:
        if (target_branch, target_direction) in tc.branches_covered:
            return tc.inputs
    return None
