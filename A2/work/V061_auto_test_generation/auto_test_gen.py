"""V061: Automatic Test Generation from Specifications

Composes:
- V004 (VCGen) -- extract requires/ensures, WP calculus, SMT checking
- V001 (Guided Symbolic Execution) -- path-covering input generation
- V054 (Verification-Driven Fuzzing) -- mutation/boundary inputs

Given source code with requires/ensures annotations, generates comprehensive
test suites covering:
1. Specification boundary values (from requires constraints)
2. Path-covering inputs (via symbolic execution)
3. Mutation-based inputs (via fuzzing)
4. Counterexample-derived tests (from failed VCs)
"""

import sys, os
_base = os.path.dirname(__file__)
_root = os.path.join(_base, '..', '..', '..')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_base, '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(_base, '..', 'V001_guided_symbolic_execution'))
sys.path.insert(0, os.path.join(_base, '..', 'V054_verification_driven_fuzzing'))
sys.path.insert(0, os.path.join(_root, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_root, 'challenges', 'C037_smt_solver'))

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
import random
import copy

# V004: Verification conditions
from vc_gen import (
    verify_function, verify_program, VerificationResult, VCResult, VCStatus,
    FnSpec, extract_fn_spec, WPCalculus, SExpr, SVar, SInt, SBool,
    SBinOp, SUnaryOp, SAnd, SOr, SNot, SImplies, SIte,
    s_and, s_or, s_not, s_implies, ast_to_sexpr, lower_to_smt
)

# V001: Guided symbolic execution
from guided_symbolic import GuidedSymbolicExecutor, GuidedResult

# V054: Fuzzing
from verification_driven_fuzzing import (
    MutationEngine, FuzzInput, FuzzFinding, CoverageInfo
)

# C010: Parser
from stack_vm import lex, Parser

# C037: SMT solver
from smt_solver import SMTSolver, SMTResult, App, Op, IntConst, BOOL


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TestSource(Enum):
    """How a test case was generated."""
    SPEC_BOUNDARY = "spec_boundary"
    SYMBOLIC = "symbolic"
    COUNTEREXAMPLE = "counterexample"
    MUTATION = "mutation"
    RANDOM = "random"
    BOUNDARY = "boundary"
    MINIMAL = "minimal"


class TestVerdict(Enum):
    """Test outcome."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class TestCase:
    """A single generated test case."""
    inputs: Dict[str, int]
    expected_output: Optional[Any] = None
    actual_output: Optional[Any] = None
    verdict: TestVerdict = TestVerdict.SKIP
    source: TestSource = TestSource.RANDOM
    description: str = ""
    precondition_met: bool = True
    postcondition_checked: bool = False
    postcondition_held: bool = True

    @property
    def key(self) -> tuple:
        return tuple(sorted(self.inputs.items()))


@dataclass
class TestSuite:
    """Complete generated test suite for a function."""
    function_name: str
    tests: List[TestCase] = field(default_factory=list)
    spec: Optional[FnSpec] = None
    coverage: Optional[CoverageInfo] = None
    verification_result: Optional[VerificationResult] = None

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.verdict == TestVerdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.verdict == TestVerdict.FAIL)

    @property
    def errors(self) -> int:
        return sum(1 for t in self.tests if t.verdict == TestVerdict.ERROR)

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.verdict == TestVerdict.SKIP)

    @property
    def unique_inputs(self) -> int:
        return len(set(t.key for t in self.tests))

    def tests_by_source(self) -> Dict[TestSource, List[TestCase]]:
        result = {}
        for t in self.tests:
            result.setdefault(t.source, []).append(t)
        return result


@dataclass
class GenerationResult:
    """Result of test generation for one or more functions."""
    suites: List[TestSuite] = field(default_factory=list)
    total_generated: int = 0
    total_unique: int = 0

    @property
    def all_passed(self) -> bool:
        return all(s.failed == 0 and s.errors == 0 for s in self.suites)


# ---------------------------------------------------------------------------
# Spec analysis -- extract constraints from requires/ensures
# ---------------------------------------------------------------------------

class SpecAnalyzer:
    """Analyzes function specifications to extract testable constraints."""

    def extract_boundaries(self, spec: FnSpec) -> List[Dict[str, int]]:
        """Extract boundary values from precondition constraints."""
        boundaries = []
        for pre in spec.preconditions:
            bounds = self._extract_bounds(pre, spec.params)
            boundaries.extend(bounds)
        return boundaries

    def _extract_bounds(self, expr: SExpr, params: List[str]) -> List[Dict[str, int]]:
        """Extract concrete boundary values from a symbolic expression."""
        results = []
        if isinstance(expr, SBinOp):
            var_name, const_val = self._extract_var_const(expr)
            if var_name and const_val is not None:
                # For x > 5: test at 6 (just inside), 5 (boundary), 4 (just outside)
                if expr.op in ('>', 'gt'):
                    results.append({var_name: const_val + 1})  # just inside
                    results.append({var_name: const_val})       # boundary
                    results.append({var_name: const_val - 1})   # just outside
                elif expr.op in ('>=', 'gte', 'ge'):
                    results.append({var_name: const_val})
                    results.append({var_name: const_val - 1})
                    results.append({var_name: const_val + 1})
                elif expr.op in ('<', 'lt'):
                    results.append({var_name: const_val - 1})
                    results.append({var_name: const_val})
                    results.append({var_name: const_val + 1})
                elif expr.op in ('<=', 'lte', 'le'):
                    results.append({var_name: const_val})
                    results.append({var_name: const_val + 1})
                    results.append({var_name: const_val - 1})
                elif expr.op in ('==', 'eq'):
                    results.append({var_name: const_val})
                    results.append({var_name: const_val + 1})
                    results.append({var_name: const_val - 1})
                elif expr.op in ('!=', 'neq', 'ne'):
                    results.append({var_name: const_val + 1})
                    results.append({var_name: const_val - 1})
                    results.append({var_name: const_val})
            # Recurse into operands
            results.extend(self._extract_bounds(expr.left, params))
            results.extend(self._extract_bounds(expr.right, params))
        elif isinstance(expr, SAnd):
            for c in expr.conjuncts:
                results.extend(self._extract_bounds(c, params))
        elif isinstance(expr, SOr):
            for d in expr.disjuncts:
                results.extend(self._extract_bounds(d, params))
        elif isinstance(expr, SNot):
            results.extend(self._extract_bounds(expr.operand, params))
        elif isinstance(expr, SUnaryOp):
            results.extend(self._extract_bounds(expr.operand, params))
        return results

    def _extract_var_const(self, binop: SBinOp) -> Tuple[Optional[str], Optional[int]]:
        """Extract (variable_name, constant_value) from a comparison."""
        left_var = self._as_var(binop.left)
        right_const = self._as_const(binop.right)
        if left_var is not None and right_const is not None:
            return left_var, right_const
        right_var = self._as_var(binop.right)
        left_const = self._as_const(binop.left)
        if right_var is not None and left_const is not None:
            return right_var, left_const
        return None, None

    def _as_var(self, expr: SExpr) -> Optional[str]:
        if isinstance(expr, SVar):
            return expr.name
        return None

    def _as_const(self, expr: SExpr) -> Optional[int]:
        if isinstance(expr, SInt):
            return expr.value
        if isinstance(expr, SUnaryOp) and expr.op == '-' and isinstance(expr.operand, SInt):
            return -expr.operand.value
        return None

    def check_precondition(self, spec: FnSpec, inputs: Dict[str, int]) -> bool:
        """Check if inputs satisfy the precondition."""
        if not spec.preconditions:
            return True
        for pre in spec.preconditions:
            if not self._eval_sexpr(pre, inputs):
                return False
        return True

    def check_postcondition(self, spec: FnSpec, inputs: Dict[str, int],
                            result_val: Any) -> bool:
        """Check if result satisfies the postcondition."""
        if not spec.postconditions:
            return True
        env = dict(inputs)
        env['result'] = result_val
        for post in spec.postconditions:
            if not self._eval_sexpr(post, env):
                return False
        return True

    def _eval_sexpr(self, expr: SExpr, env: Dict[str, Any]) -> Any:
        """Evaluate a symbolic expression with concrete values."""
        if isinstance(expr, SVar):
            return env.get(expr.name, 0)
        if isinstance(expr, SInt):
            return expr.value
        if isinstance(expr, SBool):
            return expr.value
        if isinstance(expr, SBinOp):
            left = self._eval_sexpr(expr.left, env)
            right = self._eval_sexpr(expr.right, env)
            return self._eval_binop(expr.op, left, right)
        if isinstance(expr, SUnaryOp):
            val = self._eval_sexpr(expr.operand, env)
            if expr.op in ('-', 'neg'):
                return -val
            if expr.op in ('not', '!'):
                return not val
            return val
        if isinstance(expr, SAnd):
            return all(self._eval_sexpr(c, env) for c in expr.conjuncts)
        if isinstance(expr, SOr):
            return any(self._eval_sexpr(d, env) for d in expr.disjuncts)
        if isinstance(expr, SNot):
            return not self._eval_sexpr(expr.operand, env)
        if isinstance(expr, SImplies):
            a = self._eval_sexpr(expr.antecedent, env)
            b = self._eval_sexpr(expr.consequent, env)
            return (not a) or b
        if isinstance(expr, SIte):
            c = self._eval_sexpr(expr.cond, env)
            if c:
                return self._eval_sexpr(expr.then_val, env)
            else:
                return self._eval_sexpr(expr.else_val, env)
        return 0

    def _eval_binop(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate a binary operation."""
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a // b if b != 0 else 0,
            '%': lambda a, b: a % b if b != 0 else 0,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
        }
        fn = ops.get(op)
        if fn:
            return fn(left, right)
        return 0

    def generate_smt_inputs(self, spec: FnSpec, count: int = 10,
                            seed: int = 42) -> List[Dict[str, int]]:
        """Use SMT to generate inputs satisfying the precondition."""
        if not spec.preconditions:
            return []
        results = []
        solver = SMTSolver()
        solver.push()
        # Declare params
        smt_vars = {}
        for p in spec.params:
            smt_vars[p] = solver.Int(p)
        # Add preconditions
        for pre in spec.preconditions:
            smt_formula = lower_to_smt(solver, pre, smt_vars)
            solver.add(smt_formula)
        # Generate diverse inputs by excluding previous solutions
        for i in range(count):
            status = solver.check()
            if status == SMTResult.SAT:
                model = solver.model()
                inputs = {}
                for p in spec.params:
                    val = model.get(p, 0)
                    if isinstance(val, bool):
                        inputs[p] = 1 if val else 0
                    else:
                        inputs[p] = int(val) if val is not None else 0
                results.append(inputs)
                # Exclude this exact solution
                exclusion_parts = []
                for p in spec.params:
                    exclusion_parts.append(
                        App(Op.NEQ, [smt_vars[p], IntConst(inputs[p])], BOOL)
                    )
                if exclusion_parts:
                    solver.add(solver.Or(*exclusion_parts) if len(exclusion_parts) > 1
                              else exclusion_parts[0])
            else:
                break
        solver.pop()
        return results


# ---------------------------------------------------------------------------
# Test executor -- runs function with inputs
# ---------------------------------------------------------------------------

class TestExecutor:
    """Executes test inputs against source code and captures results."""

    def execute(self, source: str, fn_name: str, inputs: Dict[str, int]) -> Tuple[Any, Optional[str]]:
        """Execute function with inputs. Returns (result, error_or_None)."""
        # Strip annotation calls (requires/ensures/invariant) before execution
        clean_source = self._strip_annotations(source)
        # Build a call expression
        args = ', '.join(str(inputs.get(p, 0)) for p in self._get_params(source, fn_name))
        call_source = clean_source + f'\nlet __r = {fn_name}({args});\nprint(__r);'
        try:
            from stack_vm import lex as lex_fn, Parser as P, Compiler, VM
            tokens = lex_fn(call_source)
            ast = P(tokens).parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            vm = VM(chunk)
            vm.run()
            # Get printed output
            if vm.output:
                result_str = vm.output[-1]
                return self._parse_result(result_str), None
            return None, None
        except Exception as e:
            return None, str(e)

    def _strip_annotations(self, source: str) -> str:
        """Remove requires/ensures/invariant/assert calls from source."""
        import re
        lines = source.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('requires(', 'ensures(', 'invariant(')):
                # Skip annotation lines
                continue
            result.append(line)
        return '\n'.join(result)

    def _get_params(self, source: str, fn_name: str) -> List[str]:
        """Extract parameter names for a function."""
        try:
            tokens = lex(source)
            ast = Parser(tokens).parse()
            for stmt in ast.stmts:
                if hasattr(stmt, 'name') and stmt.name == fn_name:
                    return list(stmt.params)
        except Exception:
            pass
        return []

    def _parse_result(self, s: str) -> Any:
        """Parse a printed result back to a value."""
        s = s.strip()
        if s == 'true':
            return True
        if s == 'false':
            return False
        if s == 'null' or s == 'None':
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s


# ---------------------------------------------------------------------------
# Input combiner -- merge partial inputs into full test inputs
# ---------------------------------------------------------------------------

class InputCombiner:
    """Combines partial boundary inputs into full test inputs."""

    def __init__(self, params: List[str], seed: int = 42):
        self.params = params
        self.rng = random.Random(seed)

    def combine(self, partial_inputs: List[Dict[str, int]],
                defaults: Optional[Dict[str, int]] = None) -> List[Dict[str, int]]:
        """Fill in missing parameters with defaults or zero."""
        if defaults is None:
            defaults = {p: 0 for p in self.params}
        results = []
        seen = set()
        for partial in partial_inputs:
            full = dict(defaults)
            full.update(partial)
            # Only include params that exist
            full = {k: v for k, v in full.items() if k in self.params}
            key = tuple(sorted(full.items()))
            if key not in seen:
                seen.add(key)
                results.append(full)
        return results

    def cross_product(self, boundary_values: Dict[str, List[int]],
                      max_combos: int = 100) -> List[Dict[str, int]]:
        """Generate cross-product of boundary values, capped."""
        if not boundary_values:
            return []
        params_with_vals = [(p, vals) for p, vals in boundary_values.items()
                           if p in self.params and vals]
        if not params_with_vals:
            return []

        results = [{}]
        for param, vals in params_with_vals:
            new_results = []
            for existing in results:
                for v in vals:
                    combo = dict(existing)
                    combo[param] = v
                    new_results.append(combo)
            results = new_results
            if len(results) > max_combos * 2:
                self.rng.shuffle(results)
                results = results[:max_combos]

        # Fill missing params
        for r in results:
            for p in self.params:
                if p not in r:
                    r[p] = 0

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:max_combos]


# ---------------------------------------------------------------------------
# Test minimizer -- reduce inputs to minimal failing/covering set
# ---------------------------------------------------------------------------

class TestMinimizer:
    """Minimizes test inputs while preserving the property of interest."""

    def minimize_input(self, inputs: Dict[str, int],
                       check_fn, max_steps: int = 50) -> Dict[str, int]:
        """Try to reduce input values while check_fn still returns True."""
        current = dict(inputs)
        for _ in range(max_steps):
            improved = False
            for param in list(current.keys()):
                val = current[param]
                if val == 0:
                    continue
                # Try zero
                candidate = dict(current)
                candidate[param] = 0
                if check_fn(candidate):
                    current = candidate
                    improved = True
                    continue
                # Try halving
                half = val // 2
                if half != val:
                    candidate[param] = half
                    if check_fn(candidate):
                        current = candidate
                        improved = True
                        continue
                # Try decrementing toward zero
                toward_zero = val - (1 if val > 0 else -1)
                if toward_zero != val:
                    candidate[param] = toward_zero
                    if check_fn(candidate):
                        current = candidate
                        improved = True
            if not improved:
                break
        return current


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class AutoTestGenerator:
    """Automatic test generation from function specifications.

    Pipeline:
    1. Parse source, extract function specs (V004)
    2. Verify specs via VCGen -- extract counterexamples as test cases
    3. Analyze spec boundaries -- generate boundary value tests
    4. Run guided symbolic execution (V001) -- path-covering tests
    5. Mutate best inputs (V054) -- mutation-based tests
    6. Generate random valid inputs via SMT
    7. Execute all tests, check postconditions
    8. Minimize failing tests
    """

    def __init__(self, max_tests: int = 100, max_symbolic_paths: int = 32,
                 mutation_rounds: int = 2, random_count: int = 10,
                 seed: int = 42):
        self.max_tests = max_tests
        self.max_symbolic_paths = max_symbolic_paths
        self.mutation_rounds = mutation_rounds
        self.random_count = random_count
        self.seed = seed
        self.spec_analyzer = SpecAnalyzer()
        self.executor = TestExecutor()
        self.minimizer = TestMinimizer()

    def generate(self, source: str, fn_name: str = None) -> GenerationResult:
        """Generate tests for function(s) in source."""
        result = GenerationResult()
        # Parse and find functions
        try:
            tokens = lex(source)
            ast = Parser(tokens).parse()
        except Exception as e:
            return result

        functions = []
        for stmt in ast.stmts:
            if hasattr(stmt, 'name') and hasattr(stmt, 'params') and hasattr(stmt, 'body'):
                if fn_name is None or stmt.name == fn_name:
                    functions.append(stmt)

        for fn in functions:
            suite = self._generate_for_function(source, fn)
            result.suites.append(suite)
            result.total_generated += suite.total
            result.total_unique += suite.unique_inputs

        return result

    def _generate_for_function(self, source: str, fn) -> TestSuite:
        """Generate test suite for a single function."""
        suite = TestSuite(function_name=fn.name)

        # Step 1: Extract spec
        try:
            spec = extract_fn_spec(fn)
            suite.spec = spec
        except Exception:
            spec = FnSpec(name=fn.name, params=list(fn.params),
                         preconditions=[], postconditions=[],
                         body_stmts=[])
            suite.spec = spec

        params = list(fn.params)
        combiner = InputCombiner(params, seed=self.seed)
        seen_keys = set()

        # Step 2: Verify and extract counterexamples
        try:
            vr = verify_function(source, fn.name)
            suite.verification_result = vr
            for vc in vr.vcs:
                if vc.status == VCStatus.INVALID and vc.counterexample:
                    inputs = {k: int(v) if not isinstance(v, bool) else (1 if v else 0)
                              for k, v in vc.counterexample.items()
                              if k in params}
                    if inputs:
                        # Fill missing params
                        for p in params:
                            if p not in inputs:
                                inputs[p] = 0
                        key = tuple(sorted(inputs.items()))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            suite.tests.append(TestCase(
                                inputs=inputs,
                                source=TestSource.COUNTEREXAMPLE,
                                description=f"Counterexample for: {vc.name}"
                            ))
        except Exception:
            pass

        # Step 3: Spec boundary values
        if spec.preconditions:
            boundary_partials = self.spec_analyzer.extract_boundaries(spec)
            boundary_inputs = combiner.combine(boundary_partials)
            # Also try cross-product of per-param boundaries
            per_param = {}
            for partial in boundary_partials:
                for p, v in partial.items():
                    per_param.setdefault(p, set()).add(v)
            cross = combiner.cross_product(
                {p: list(vs) for p, vs in per_param.items()}, max_combos=30
            )
            boundary_inputs.extend(cross)
            for inp in boundary_inputs:
                key = tuple(sorted(inp.items()))
                if key not in seen_keys and len(suite.tests) < self.max_tests:
                    seen_keys.add(key)
                    suite.tests.append(TestCase(
                        inputs=inp,
                        source=TestSource.SPEC_BOUNDARY,
                        description="Boundary value from spec"
                    ))

        # Step 4: SMT-generated valid inputs
        if spec.preconditions:
            smt_inputs = self.spec_analyzer.generate_smt_inputs(
                spec, count=self.random_count, seed=self.seed
            )
            for inp in smt_inputs:
                # Fill missing params
                for p in params:
                    if p not in inp:
                        inp[p] = 0
                key = tuple(sorted(inp.items()))
                if key not in seen_keys and len(suite.tests) < self.max_tests:
                    seen_keys.add(key)
                    suite.tests.append(TestCase(
                        inputs=inp,
                        source=TestSource.RANDOM,
                        description="SMT-generated valid input"
                    ))

        # Step 5: Symbolic execution for path coverage
        try:
            sym_inputs = {p: 'int' for p in params}
            guided = GuidedSymbolicExecutor(
                max_paths=self.max_symbolic_paths, max_loop_unroll=5
            )
            # Build source that calls the function with symbolic inputs
            guided_result = guided.guided_execute(source, symbolic_inputs=sym_inputs)
            if guided_result and guided_result.test_cases:
                for tc in guided_result.test_cases:
                    if isinstance(tc, dict):
                        inp = {k: int(v) if not isinstance(v, bool) else (1 if v else 0)
                               for k, v in tc.items() if k in params}
                        for p in params:
                            if p not in inp:
                                inp[p] = 0
                        key = tuple(sorted(inp.items()))
                        if key not in seen_keys and len(suite.tests) < self.max_tests:
                            seen_keys.add(key)
                            suite.tests.append(TestCase(
                                inputs=inp,
                                source=TestSource.SYMBOLIC,
                                description="Path-covering symbolic input"
                            ))
        except Exception:
            pass

        # Step 6: Mutation-based inputs
        if suite.tests:
            mutator = MutationEngine(seed=self.seed)
            seeds = [t.inputs for t in suite.tests[:10]]
            for round_num in range(self.mutation_rounds):
                strength = round_num + 1
                for seed_input in seeds:
                    mutated = mutator.mutate_batch(
                        seed_input, count=5, strength=strength
                    )
                    for m in mutated:
                        # Filter to known params
                        m = {k: v for k, v in m.items() if k in params}
                        for p in params:
                            if p not in m:
                                m[p] = 0
                        key = tuple(sorted(m.items()))
                        if key not in seen_keys and len(suite.tests) < self.max_tests:
                            seen_keys.add(key)
                            suite.tests.append(TestCase(
                                inputs=m,
                                source=TestSource.MUTATION,
                                description=f"Mutation (strength {strength})"
                            ))

        # Step 7: Random fill if under budget
        rng = random.Random(self.seed)
        while len(suite.tests) < min(self.max_tests, self.random_count + len(suite.tests)):
            inp = {p: rng.randint(-100, 100) for p in params}
            key = tuple(sorted(inp.items()))
            if key not in seen_keys:
                seen_keys.add(key)
                suite.tests.append(TestCase(
                    inputs=inp,
                    source=TestSource.RANDOM,
                    description="Random input"
                ))
            if len(suite.tests) >= self.max_tests:
                break

        # Step 8: Execute all tests and check specs
        self._execute_suite(source, suite)

        # Step 9: Minimize failing tests
        self._minimize_failures(source, suite)

        return suite

    def _execute_suite(self, source: str, suite: TestSuite):
        """Execute all tests in a suite and check postconditions."""
        spec = suite.spec
        for test in suite.tests:
            # Check precondition
            if spec and spec.preconditions:
                test.precondition_met = self.spec_analyzer.check_precondition(
                    spec, test.inputs
                )
                if not test.precondition_met:
                    test.verdict = TestVerdict.SKIP
                    test.description += " [precond not met]"
                    continue

            # Execute
            result, error = self.executor.execute(
                source, suite.function_name, test.inputs
            )
            test.actual_output = result

            if error:
                test.verdict = TestVerdict.ERROR
                test.description += f" [error: {error[:80]}]"
                continue

            # Check postcondition
            if spec and spec.postconditions and result is not None:
                test.postcondition_checked = True
                post_ok = self.spec_analyzer.check_postcondition(
                    spec, test.inputs, result
                )
                test.postcondition_held = post_ok
                if post_ok:
                    test.verdict = TestVerdict.PASS
                else:
                    test.verdict = TestVerdict.FAIL
                    test.description += " [postcond violated]"
            else:
                # No postcondition to check -- pass if no error
                test.verdict = TestVerdict.PASS

    def _minimize_failures(self, source: str, suite: TestSuite):
        """Try to minimize failing test inputs."""
        spec = suite.spec
        for test in suite.tests:
            if test.verdict != TestVerdict.FAIL:
                continue

            def still_fails(inputs):
                if spec and spec.preconditions:
                    if not self.spec_analyzer.check_precondition(spec, inputs):
                        return False
                result, error = self.executor.execute(
                    source, suite.function_name, inputs
                )
                if error:
                    return True
                if spec and spec.postconditions and result is not None:
                    return not self.spec_analyzer.check_postcondition(
                        spec, inputs, result
                    )
                return False

            minimized = self.minimizer.minimize_input(test.inputs, still_fails)
            if minimized != test.inputs:
                # Add minimized version
                suite.tests.append(TestCase(
                    inputs=minimized,
                    actual_output=test.actual_output,
                    verdict=TestVerdict.FAIL,
                    source=TestSource.MINIMAL,
                    description=f"Minimized from {test.inputs}"
                ))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def generate_tests(source: str, fn_name: str = None,
                   max_tests: int = 100, seed: int = 42) -> GenerationResult:
    """Generate tests for function(s) in source code with specs."""
    gen = AutoTestGenerator(max_tests=max_tests, seed=seed)
    return gen.generate(source, fn_name)


def generate_and_report(source: str, fn_name: str = None) -> str:
    """Generate tests and return a human-readable report."""
    result = generate_tests(source, fn_name)
    lines = []
    for suite in result.suites:
        lines.append(f"=== {suite.function_name} ===")
        lines.append(f"Total: {suite.total} | Pass: {suite.passed} | "
                     f"Fail: {suite.failed} | Error: {suite.errors} | "
                     f"Skip: {suite.skipped}")
        by_source = suite.tests_by_source()
        for src, tests in sorted(by_source.items(), key=lambda x: x[0].value):
            lines.append(f"  {src.value}: {len(tests)} tests")
        if suite.failed > 0:
            lines.append("  FAILURES:")
            for t in suite.tests:
                if t.verdict == TestVerdict.FAIL:
                    lines.append(f"    inputs={t.inputs} -> {t.actual_output} "
                                f"({t.description})")
        lines.append("")
    return '\n'.join(lines)


def quick_generate(source: str, fn_name: str = None) -> GenerationResult:
    """Quick generation with minimal effort."""
    gen = AutoTestGenerator(max_tests=30, max_symbolic_paths=8,
                           mutation_rounds=1, random_count=5, seed=42)
    return gen.generate(source, fn_name)


def deep_generate(source: str, fn_name: str = None) -> GenerationResult:
    """Deep generation with maximum effort."""
    gen = AutoTestGenerator(max_tests=200, max_symbolic_paths=64,
                           mutation_rounds=3, random_count=20, seed=42)
    return gen.generate(source, fn_name)
