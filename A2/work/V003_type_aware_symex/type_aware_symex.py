"""
V003: Type-Aware Symbolic Execution
Composes C013 (Type Checker) + C038 (Symbolic Execution)

Uses static type information to improve symbolic execution:
  1. Type-guided symbolic variable detection: auto-discover which vars need
     symbolic treatment based on function signatures
  2. Type invariant injection: add constraints that enforce type domains
     (e.g., bool vars are 0 or 1)
  3. Type-based path pruning: skip paths that would cause type errors
  4. Type-informed test generation: produce type-correct test values
  5. Type error as reachability: find inputs that trigger type-unsafe operations

Architecture:
  Source -> C010 Parser -> AST -> C013 Type Check -> Type Env
                                                        |
                                                        v
  Source -> C038 Symbolic Exec (+ type constraints) -> Paths + Typed Tests
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

# Import C013 type checker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C013_type_checker'))
from type_checker import (
    TypeChecker, TypeEnv, parse as tc_parse, check_program,
    TInt, TFloat, TString, TBool, TVoid, TFunc, TVar as TypeVariable,
    INT as T_INT, FLOAT as T_FLOAT, STRING as T_STRING,
    BOOL as T_BOOL, VOID as T_VOID,
    resolve as resolve_type, TypeError_
)

# Import C038 symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, SymValue, SymType, PathState, PathStatus,
    ExecutionResult, TestCase, AssertionResult, CoverageResult,
    symbolic_execute, check_assertions,
    smt_not, smt_and, smt_or
)

# Import SMT solver types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var as SMTVar, App,
    IntConst, BoolConst, Op as SMTOp, BOOL, INT as SMT_INT
)


# ============================================================
# Type Analysis Result
# ============================================================

@dataclass
class VarTypeInfo:
    """Type information for a variable discovered by static analysis."""
    name: str
    inferred_type: Any  # A C013 type (TInt, TBool, etc.)
    source: str  # 'param', 'let', 'assign'
    line: int = 0


@dataclass
class FunctionTypeInfo:
    """Type signature of a function from static analysis."""
    name: str
    param_types: list  # list of (param_name, C013_type)
    return_type: Any
    line: int = 0


@dataclass
class TypeAnalysis:
    """Result of static type analysis on a program."""
    variables: dict  # name -> VarTypeInfo
    functions: dict  # name -> FunctionTypeInfo
    errors: list  # list of TypeError_
    has_errors: bool = False


def analyze_types(source: str) -> TypeAnalysis:
    """Run C013 type checker and extract type information."""
    program = tc_parse(source)
    checker = TypeChecker()
    errors = checker.check(program)

    variables = {}
    functions = {}

    # Extract variable types from the type environment
    _extract_env(checker.env, variables)

    # Extract function signatures
    for stmt in program.stmts:
        if isinstance(stmt, FnDecl):
            fn_type = checker.env.lookup(stmt.name)
            if fn_type is not None:
                fn_type = resolve_type(fn_type)
                if isinstance(fn_type, TFunc):
                    param_types = []
                    for i, pname in enumerate(stmt.params):
                        if i < len(fn_type.params):
                            pt = resolve_type(fn_type.params[i])
                            param_types.append((pname, pt))
                        else:
                            param_types.append((pname, None))
                    ret = resolve_type(fn_type.ret)
                    functions[stmt.name] = FunctionTypeInfo(
                        name=stmt.name,
                        param_types=param_types,
                        return_type=ret,
                        line=getattr(stmt, 'line', 0),
                    )

    return TypeAnalysis(
        variables=variables,
        functions=functions,
        errors=errors,
        has_errors=len(errors) > 0,
    )


def _extract_env(env: TypeEnv, variables: dict):
    """Recursively extract variable types from a TypeEnv."""
    for name, typ in env.bindings.items():
        resolved = resolve_type(typ)
        if not isinstance(resolved, TFunc):
            variables[name] = VarTypeInfo(
                name=name,
                inferred_type=resolved,
                source='env',
            )
    if env.parent:
        _extract_env(env.parent, variables)


# ============================================================
# Type-to-SMT mapping
# ============================================================

def c013_type_to_smt_sort(t) -> Optional[str]:
    """Map a C013 type to an SMT sort name ('int' or 'bool')."""
    t = resolve_type(t)
    if isinstance(t, TInt):
        return 'int'
    if isinstance(t, TFloat):
        return 'int'  # approximate floats as ints for SMT
    if isinstance(t, TBool):
        return 'bool'
    if isinstance(t, TypeVariable):
        return 'int'  # default unresolved type vars to int
    return None


def c013_type_to_symbolic_input(t) -> Optional[str]:
    """Map a C013 type to a symbolic_inputs type string for C038."""
    t = resolve_type(t)
    if isinstance(t, TInt):
        return 'int'
    if isinstance(t, TFloat):
        return 'int'
    if isinstance(t, TBool):
        return 'bool'
    return None


# ============================================================
# Type Constraints
# ============================================================

def make_bool_invariant(var_name: str) -> Term:
    """Create constraint: var == 0 OR var == 1 (bool domain)."""
    v = SMTVar(var_name, SMT_INT)
    eq0 = App(SMTOp.EQ, [v, IntConst(0)], BOOL)
    eq1 = App(SMTOp.EQ, [v, IntConst(1)], BOOL)
    return App(SMTOp.OR, [eq0, eq1], BOOL)


def make_nonneg_invariant(var_name: str) -> Term:
    """Create constraint: var >= 0."""
    v = SMTVar(var_name, SMT_INT)
    return App(SMTOp.GE, [v, IntConst(0)], BOOL)


def make_range_invariant(var_name: str, lo: int, hi: int) -> Term:
    """Create constraint: lo <= var AND var <= hi."""
    v = SMTVar(var_name, SMT_INT)
    ge_lo = App(SMTOp.GE, [v, IntConst(lo)], BOOL)
    le_hi = App(SMTOp.LE, [v, IntConst(hi)], BOOL)
    return App(SMTOp.AND, [ge_lo, le_hi], BOOL)


# ============================================================
# Type-Aware Symbolic Executor
# ============================================================

class TypeAwareExecutor:
    """
    Composes C013 type checking with C038 symbolic execution.

    Workflow:
      1. Parse source with C010
      2. Run C013 type checker to get type environment
      3. Use type info to:
         a. Auto-detect symbolic inputs for function parameters
         b. Inject type invariants as constraints
         c. Generate type-correct test values
      4. Run C038 symbolic execution with enhanced constraints
      5. Post-process results with type information
    """

    def __init__(self, max_paths=64, max_loop_unroll=5,
                 inject_bool_invariants=True,
                 inject_range_constraints=False,
                 range_bound=100):
        self.max_paths = max_paths
        self.max_loop_unroll = max_loop_unroll
        self.inject_bool_invariants = inject_bool_invariants
        self.inject_range_constraints = inject_range_constraints
        self.range_bound = range_bound

    def execute(self, source: str,
                symbolic_inputs: dict = None,
                target_function: str = None) -> 'TypeAwareResult':
        """
        Execute with type awareness.

        Args:
            source: C010 source code
            symbolic_inputs: explicit symbolic inputs (overrides auto-detection)
            target_function: if set, auto-detect symbolic inputs from this
                           function's parameter types

        Returns:
            TypeAwareResult with type analysis, paths, typed test cases, and stats.
        """
        # Phase 1: Type analysis
        type_info = analyze_types(source)

        # Phase 2: Determine symbolic inputs
        if symbolic_inputs is None and target_function is not None:
            symbolic_inputs = self._infer_symbolic_inputs(
                type_info, target_function)
        elif symbolic_inputs is None:
            symbolic_inputs = self._infer_toplevel_symbolic_inputs(
                source, type_info)

        # Phase 3: Compute type invariants
        type_constraints = []
        if symbolic_inputs:
            type_constraints = self._build_type_constraints(
                symbolic_inputs, type_info)

        # Phase 4: Run symbolic execution
        engine = SymbolicExecutor(
            max_paths=self.max_paths,
            max_loop_unroll=self.max_loop_unroll,
        )
        exec_result = engine.execute(source, symbolic_inputs)

        # Phase 5: Filter paths by type constraints
        if type_constraints:
            exec_result = self._filter_by_type_constraints(
                exec_result, type_constraints, symbolic_inputs)

        # Phase 6: Build typed test cases
        typed_tests = self._build_typed_tests(
            exec_result, type_info, symbolic_inputs or {})

        # Phase 7: Detect type-unsafe paths
        type_warnings = self._detect_type_warnings(
            exec_result, type_info, symbolic_inputs or {})

        stats = TypeAwareStats(
            total_paths=len(exec_result.paths),
            feasible_paths=len(exec_result.feasible_paths),
            typed_tests=len(typed_tests),
            type_constraints_injected=len(type_constraints),
            type_errors_found=len(type_info.errors),
            type_warnings=len(type_warnings),
            paths_pruned_by_types=getattr(self, '_pruned_count', 0),
        )

        return TypeAwareResult(
            type_analysis=type_info,
            execution=exec_result,
            typed_tests=typed_tests,
            type_warnings=type_warnings,
            symbolic_inputs=symbolic_inputs or {},
            type_constraints=type_constraints,
            stats=stats,
        )

    def analyze_function(self, source: str,
                         function_name: str) -> 'TypeAwareResult':
        """
        Analyze a specific function by auto-detecting parameter types
        and generating symbolic inputs accordingly.
        """
        return self.execute(source, target_function=function_name)

    def find_type_errors(self, source: str,
                         symbolic_inputs: dict = None) -> 'TypeErrorResult':
        """
        Use symbolic execution to find inputs that trigger operations
        on wrong types (e.g., adding a bool to an int in a way the
        type checker flags).
        """
        type_info = analyze_types(source)
        result = self.execute(source, symbolic_inputs)

        type_error_paths = []
        for path in result.execution.paths:
            if path.status == PathStatus.ERROR:
                type_error_paths.append(path)

        return TypeErrorResult(
            static_errors=type_info.errors,
            dynamic_error_paths=type_error_paths,
            total_paths=len(result.execution.paths),
        )

    # --- Internal methods ---

    def _infer_symbolic_inputs(self, type_info: TypeAnalysis,
                               fn_name: str) -> dict:
        """Infer symbolic inputs from a function's parameter types."""
        if fn_name not in type_info.functions:
            return {}
        fn = type_info.functions[fn_name]
        inputs = {}
        for pname, ptype in fn.param_types:
            smt_type = c013_type_to_symbolic_input(ptype)
            if smt_type:
                inputs[pname] = smt_type
        return inputs

    def _infer_toplevel_symbolic_inputs(self, source: str,
                                        type_info: TypeAnalysis) -> dict:
        """
        Infer symbolic inputs for top-level variables that are declared
        but not initialized with a concrete value (e.g., function params
        of the first function, or variables used before assignment).
        """
        # Parse to find the first function's params
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()

        for stmt in ast.stmts:
            if isinstance(stmt, FnDecl) and stmt.name in type_info.functions:
                return self._infer_symbolic_inputs(type_info, stmt.name)
        return {}

    def _build_type_constraints(self, symbolic_inputs: dict,
                                type_info: TypeAnalysis) -> list:
        """Build SMT constraints from type information."""
        constraints = []

        for var_name, smt_type in symbolic_inputs.items():
            # Check if type checker knows this is a bool
            if var_name in type_info.variables:
                var_info = type_info.variables[var_name]
                resolved = resolve_type(var_info.inferred_type)
                if isinstance(resolved, TBool) and self.inject_bool_invariants:
                    # Bool vars must be 0 or 1
                    constraints.append(make_bool_invariant(var_name))
                elif isinstance(resolved, TInt) and self.inject_range_constraints:
                    constraints.append(make_range_invariant(
                        var_name, -self.range_bound, self.range_bound))
            elif smt_type == 'bool' and self.inject_bool_invariants:
                constraints.append(make_bool_invariant(var_name))

        return constraints

    def _filter_by_type_constraints(self, result: ExecutionResult,
                                     type_constraints: list,
                                     symbolic_inputs: dict) -> ExecutionResult:
        """Filter out paths that violate type constraints."""
        self._pruned_count = 0
        filtered_paths = []

        for path in result.paths:
            if path.status == PathStatus.INFEASIBLE:
                filtered_paths.append(path)
                continue

            # Check if path + type constraints is feasible
            all_constraints = path.constraints + type_constraints
            if all_constraints:
                solver = SMTSolver()
                self._declare_vars(solver, symbolic_inputs, all_constraints)
                for c in all_constraints:
                    solver.add(c)
                r = solver.check()
                if r != SMTResult.SAT:
                    path.status = PathStatus.INFEASIBLE
                    self._pruned_count += 1

            filtered_paths.append(path)

        # Regenerate test cases for surviving paths
        new_tests = []
        for path in filtered_paths:
            if path.status == PathStatus.INFEASIBLE:
                continue
            all_constraints = path.constraints + type_constraints
            if not symbolic_inputs:
                new_tests.append(TestCase(
                    path_id=path.path_id,
                    inputs={},
                    path_status=path.status,
                    output=path.output,
                    covered_lines=path.covered_lines,
                    branches=path.covered_branches,
                ))
                continue

            solver = SMTSolver()
            self._declare_vars(solver, symbolic_inputs, all_constraints)
            for c in all_constraints:
                solver.add(c)
            r = solver.check()
            if r == SMTResult.SAT:
                model = solver.model()
                inputs = {}
                for name in symbolic_inputs:
                    if model and name in model:
                        inputs[name] = model[name]
                    else:
                        inputs[name] = 0 if symbolic_inputs[name] == 'int' else False
                new_tests.append(TestCase(
                    path_id=path.path_id,
                    inputs=inputs,
                    path_status=path.status,
                    output=path.output,
                    covered_lines=path.covered_lines,
                    branches=path.covered_branches,
                ))

        return ExecutionResult(
            paths=filtered_paths,
            test_cases=new_tests,
            functions=result.functions,
        )

    def _build_typed_tests(self, result: ExecutionResult,
                           type_info: TypeAnalysis,
                           symbolic_inputs: dict) -> list:
        """Build test cases with type annotations."""
        typed_tests = []
        for tc in result.test_cases:
            typed_inputs = {}
            for name, val in tc.inputs.items():
                c013_type = None
                if name in type_info.variables:
                    c013_type = resolve_type(type_info.variables[name].inferred_type)
                elif name in symbolic_inputs:
                    smt_type = symbolic_inputs[name]
                    if smt_type == 'int':
                        c013_type = T_INT
                    elif smt_type == 'bool':
                        c013_type = T_BOOL

                # Coerce value to match type
                coerced = self._coerce_to_type(val, c013_type)
                typed_inputs[name] = TypedValue(
                    value=coerced,
                    c013_type=c013_type,
                    raw_smt_value=val,
                )

            typed_tests.append(TypedTestCase(
                path_id=tc.path_id,
                inputs=typed_inputs,
                path_status=tc.path_status,
                output=tc.output,
                covered_lines=tc.covered_lines,
                branches=tc.branches,
            ))
        return typed_tests

    def _coerce_to_type(self, val, c013_type) -> Any:
        """Coerce an SMT model value to the expected C013 type."""
        if c013_type is None:
            return val
        if isinstance(c013_type, TBool):
            return bool(val) if val is not None else False
        if isinstance(c013_type, TInt):
            return int(val) if val is not None else 0
        if isinstance(c013_type, TFloat):
            return float(val) if val is not None else 0.0
        if isinstance(c013_type, TString):
            return str(val) if val is not None else ""
        return val

    def _detect_type_warnings(self, result: ExecutionResult,
                              type_info: TypeAnalysis,
                              symbolic_inputs: dict) -> list:
        """Detect potential type-related issues from symbolic execution."""
        warnings = []

        # Static type errors
        for err in type_info.errors:
            warnings.append(TypeWarning(
                kind='static_type_error',
                message=err.message,
                line=err.line,
            ))

        # Paths that errored out (possibly type-related)
        for path in result.paths:
            if path.status == PathStatus.ERROR and path.error_msg:
                warnings.append(TypeWarning(
                    kind='runtime_error',
                    message=path.error_msg,
                    line=0,
                    path_id=path.path_id,
                ))

        return warnings

    def _declare_vars(self, solver: SMTSolver, symbolic_inputs: dict,
                      constraints: list):
        """Declare SMT variables from symbolic inputs and constraints."""
        for name, typ in symbolic_inputs.items():
            if typ == 'int':
                solver.Int(name)
            elif typ == 'bool':
                solver.Bool(name)
        # Also declare any vars in constraints not in symbolic_inputs
        seen = set()
        for c in constraints:
            self._collect_vars(c, seen)
        for name, sort in seen:
            if name not in symbolic_inputs:
                if sort == 'int':
                    solver.Int(name)
                elif sort == 'bool':
                    solver.Bool(name)

    def _collect_vars(self, term, seen: set):
        """Collect variable names from SMT terms."""
        if isinstance(term, SMTVar):
            sort = 'bool' if term.sort == BOOL else 'int'
            seen.add((term.name, sort))
        elif isinstance(term, App):
            for child in term.args:
                self._collect_vars(child, seen)


# ============================================================
# Result Types
# ============================================================

@dataclass
class TypedValue:
    """A test value with type annotation."""
    value: Any
    c013_type: Any = None  # C013 type
    raw_smt_value: Any = None

    def __repr__(self):
        type_str = repr(self.c013_type) if self.c013_type else '?'
        return f"{self.value}: {type_str}"


@dataclass
class TypedTestCase:
    """A test case with typed inputs."""
    path_id: int
    inputs: dict  # name -> TypedValue
    path_status: PathStatus = PathStatus.COMPLETED
    output: list = field(default_factory=list)
    covered_lines: set = field(default_factory=set)
    branches: list = field(default_factory=list)

    @property
    def input_values(self) -> dict:
        """Get plain input values (no type wrappers)."""
        return {k: v.value for k, v in self.inputs.items()}


@dataclass
class TypeWarning:
    """A type-related warning from analysis."""
    kind: str  # 'static_type_error', 'runtime_error', 'type_mismatch'
    message: str
    line: int = 0
    path_id: int = 0


@dataclass
class TypeAwareStats:
    """Statistics about the type-aware execution."""
    total_paths: int = 0
    feasible_paths: int = 0
    typed_tests: int = 0
    type_constraints_injected: int = 0
    type_errors_found: int = 0
    type_warnings: int = 0
    paths_pruned_by_types: int = 0


@dataclass
class TypeAwareResult:
    """Complete result of type-aware symbolic execution."""
    type_analysis: TypeAnalysis
    execution: ExecutionResult
    typed_tests: list  # list of TypedTestCase
    type_warnings: list  # list of TypeWarning
    symbolic_inputs: dict
    type_constraints: list
    stats: TypeAwareStats

    @property
    def has_type_errors(self) -> bool:
        return self.type_analysis.has_errors

    @property
    def all_tests_pass_types(self) -> bool:
        """Check that all test inputs match their declared types."""
        for tc in self.typed_tests:
            for name, tv in tc.inputs.items():
                if not self._value_matches_type(tv.value, tv.c013_type):
                    return False
        return True

    def _value_matches_type(self, val, c013_type) -> bool:
        if c013_type is None:
            return True
        if isinstance(c013_type, TBool):
            return isinstance(val, bool)
        if isinstance(c013_type, TInt):
            return isinstance(val, int) and not isinstance(val, bool)
        if isinstance(c013_type, TFloat):
            return isinstance(val, (int, float))
        return True


@dataclass
class TypeErrorResult:
    """Result of searching for type-error-triggering inputs."""
    static_errors: list
    dynamic_error_paths: list
    total_paths: int = 0


# ============================================================
# Convenience Functions
# ============================================================

def type_aware_execute(source: str, symbolic_inputs: dict = None,
                       target_function: str = None,
                       max_paths=64, max_loop_unroll=5,
                       inject_bool_invariants=True) -> TypeAwareResult:
    """Run type-aware symbolic execution."""
    engine = TypeAwareExecutor(
        max_paths=max_paths,
        max_loop_unroll=max_loop_unroll,
        inject_bool_invariants=inject_bool_invariants,
    )
    return engine.execute(source, symbolic_inputs, target_function)


def analyze_function(source: str, function_name: str,
                     max_paths=64) -> TypeAwareResult:
    """Analyze a function with auto-detected symbolic inputs from types."""
    engine = TypeAwareExecutor(max_paths=max_paths)
    return engine.analyze_function(source, function_name)


def find_type_errors(source: str,
                     symbolic_inputs: dict = None) -> TypeErrorResult:
    """Find inputs that trigger type-related errors."""
    engine = TypeAwareExecutor()
    return engine.find_type_errors(source, symbolic_inputs)


def get_typed_tests(source: str, symbolic_inputs: dict,
                    max_paths=64) -> list:
    """Generate test cases with type annotations."""
    result = type_aware_execute(source, symbolic_inputs, max_paths=max_paths)
    return result.typed_tests
