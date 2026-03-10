"""V054: Verification-Driven Fuzzing

Composes V001 (guided symbolic execution) + V028 (fault localization) +
V018 (concolic testing) to create an intelligent fuzzing pipeline.

Key idea: use verification analysis to GUIDE fuzz input generation.
- Symbolic execution discovers paths and boundary conditions
- Fault localization ranks suspicious statements
- Concolic testing generates targeted inputs around those boundaries
- Mutation-based fuzzing explores neighborhoods of interesting inputs

Pipeline:
1. Guided symbolic execution -> paths, test cases, branch conditions
2. Extract boundary inputs (near branch flips) and interesting regions
3. Fault localization on generated tests -> suspicious statement ranking
4. Concolic testing targeting suspicious branches
5. Mutation fuzzing around boundary inputs
6. Combine all findings into a unified FuzzResult
"""

import os, sys

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, "challenges", "C010_stack_vm"))
sys.path.insert(0, os.path.join(_az, "challenges", "C037_smt_solver"))
sys.path.insert(0, os.path.join(_az, "challenges", "C038_symbolic_execution"))
sys.path.insert(0, os.path.join(_az, "challenges", "C039_abstract_interpreter"))
sys.path.insert(0, os.path.join(_work, "V001_guided_symbolic_execution"))
sys.path.insert(0, os.path.join(_work, "V018_concolic_testing"))
sys.path.insert(0, os.path.join(_work, "V028_fault_localization"))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import random
import copy

from stack_vm import lex, Parser, Program, Var, IntLit, BinOp, IfStmt, WhileStmt, LetDecl, Assign, PrintStmt
from guided_symbolic import guided_execute, GuidedResult, compare_guided_vs_plain
from concolic_testing import (
    concolic_test, concolic_find_bugs, concolic_reach_branch,
    ConcolicResult, ConcolicTestCase, BugFindingResult, BugReport
)
from fault_localization import (
    auto_localize, spectrum_localize, generate_test_suite,
    FaultResult, SuspiciousnessScore, StatementInfo, TestCase, TestVerdict, Metric
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FuzzStatus(Enum):
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget"
    BUG_FOUND = "bug_found"


@dataclass
class FuzzInput:
    """A single fuzz input with provenance."""
    values: Dict[str, int]
    source: str  # "symbolic", "concolic", "mutation", "boundary", "random"
    generation: int = 0  # mutation generation


@dataclass
class FuzzFinding:
    """A bug or interesting behavior found during fuzzing."""
    kind: str  # "crash", "assertion_failure", "divergence", "boundary"
    inputs: Dict[str, int]
    description: str
    source: str  # which phase found it
    suspicious_stmt: Optional[StatementInfo] = None


@dataclass
class CoverageInfo:
    """Branch coverage tracking."""
    covered_branches: Set[Tuple[int, bool]] = field(default_factory=set)
    total_branches: int = 0
    inputs_by_branch: Dict[Tuple[int, bool], List[Dict[str, int]]] = field(
        default_factory=dict
    )

    @property
    def coverage(self) -> float:
        if self.total_branches == 0:
            return 0.0
        return len(self.covered_branches) / self.total_branches

    def add(self, branch: Tuple[int, bool], inputs: Dict[str, int]):
        self.covered_branches.add(branch)
        if branch not in self.inputs_by_branch:
            self.inputs_by_branch[branch] = []
        self.inputs_by_branch[branch].append(inputs)


@dataclass
class FuzzResult:
    """Complete fuzzing result."""
    findings: List[FuzzFinding] = field(default_factory=list)
    coverage: Optional[CoverageInfo] = None
    total_inputs_tested: int = 0
    symbolic_inputs: int = 0
    concolic_inputs: int = 0
    mutation_inputs: int = 0
    random_inputs: int = 0
    boundary_inputs: int = 0
    status: FuzzStatus = FuzzStatus.COMPLETE
    suspicious_stmts: List[SuspiciousnessScore] = field(default_factory=list)
    all_test_inputs: List[FuzzInput] = field(default_factory=list)

    @property
    def has_bugs(self) -> bool:
        return any(f.kind in ("crash", "assertion_failure") for f in self.findings)

    @property
    def bug_count(self) -> int:
        return sum(1 for f in self.findings if f.kind in ("crash", "assertion_failure"))

    @property
    def unique_findings(self) -> int:
        return len(self.findings)

    def summary(self) -> str:
        lines = [
            f"Fuzz Result: {self.status.value}",
            f"  Total inputs tested: {self.total_inputs_tested}",
            f"    Symbolic: {self.symbolic_inputs}, Concolic: {self.concolic_inputs}",
            f"    Mutation: {self.mutation_inputs}, Random: {self.random_inputs}",
            f"    Boundary: {self.boundary_inputs}",
            f"  Findings: {len(self.findings)} ({self.bug_count} bugs)",
        ]
        if self.coverage:
            lines.append(f"  Branch coverage: {self.coverage.coverage:.1%}")
        if self.suspicious_stmts:
            top = self.suspicious_stmts[0]
            lines.append(f"  Top suspect: line {top.statement.line} ({top.statement.kind}) score={top.score:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _parse(source: str) -> Program:
    return Parser(lex(source)).parse()


class _FuzzInterpreter:
    """Simple C10 interpreter that injects fuzz inputs into let-declarations."""

    def __init__(self, inputs: Dict[str, int]):
        self.inputs = inputs
        self.env: Dict[str, int] = {}
        self.output: List = []

    def execute(self, program: Program):
        for stmt in program.stmts:
            self._exec(stmt)

    def _exec(self, stmt):
        if isinstance(stmt, LetDecl):
            # Use fuzz input if available, otherwise evaluate normally
            if stmt.name in self.inputs:
                self.env[stmt.name] = self.inputs[stmt.name]
            else:
                self.env[stmt.name] = self._eval(stmt.value)
        elif isinstance(stmt, Assign):
            self.env[stmt.name] = self._eval(stmt.value)
        elif isinstance(stmt, IfStmt):
            if self._eval(stmt.cond):
                for s in stmt.then_body.stmts:
                    self._exec(s)
            elif stmt.else_body:
                body = stmt.else_body
                if hasattr(body, 'stmts'):
                    for s in body.stmts:
                        self._exec(s)
                else:
                    self._exec(body)
        elif isinstance(stmt, WhileStmt):
            limit = 1000
            while self._eval(stmt.cond) and limit > 0:
                for s in stmt.body.stmts:
                    self._exec(s)
                limit -= 1
        elif isinstance(stmt, PrintStmt):
            val = self._eval(stmt.value)
            self.output.append(val)

    def _eval(self, expr) -> int:
        if isinstance(expr, IntLit):
            return expr.value
        elif isinstance(expr, Var):
            if expr.name in self.env:
                return self.env[expr.name]
            if expr.name in self.inputs:
                return self.inputs[expr.name]
            return 0
        elif isinstance(expr, BinOp):
            l = self._eval(expr.left)
            r = self._eval(expr.right)
            op = expr.op
            if op == '+': return l + r
            if op == '-': return l - r
            if op == '*': return l * r
            if op == '/':
                if r == 0:
                    raise ZeroDivisionError("division by zero")
                return l // r if (l >= 0 and r > 0) or (l < 0 and r < 0) else -(abs(l) // abs(r))
            if op == '%': return l % r if r != 0 else 0
            if op == '==': return 1 if l == r else 0
            if op == '!=': return 1 if l != r else 0
            if op == '<': return 1 if l < r else 0
            if op == '<=': return 1 if l <= r else 0
            if op == '>': return 1 if l > r else 0
            if op == '>=': return 1 if l >= r else 0
            if op == 'and': return 1 if (l and r) else 0
            if op == 'or': return 1 if (l or r) else 0
            return 0
        elif hasattr(expr, 'op') and hasattr(expr, 'operand'):
            # UnaryOp
            val = self._eval(expr.operand)
            if expr.op == '-': return -val
            if expr.op == 'not': return 0 if val else 1
            return val
        elif hasattr(expr, 'value') and not hasattr(expr, 'name'):
            return expr.value
        return 0


def _safe_execute(source: str, inputs: Dict[str, int]) -> Tuple[Optional[List], bool, str]:
    """Execute source with inputs, catching errors.
    Returns (output, crashed, error_message).

    Inputs override `let x = 0;` initializations so that fuzz values
    are actually used by the program.
    """
    try:
        program = _parse(source)
        interp = _FuzzInterpreter(inputs)
        interp.execute(program)
        return interp.output, False, ""
    except ZeroDivisionError:
        return None, True, "division_by_zero"
    except Exception as e:
        err = str(e)
        if "assertion" in err.lower():
            return None, True, "assertion_failure"
        return None, True, f"runtime_error: {err}"


def _extract_branch_count(source: str, inputs: Dict[str, int]) -> Set[Tuple[int, bool]]:
    """Execute and extract covered branches as (branch_id, direction) tuples."""
    try:
        program = _parse(source)
        branches = set()
        _trace_branches(program.stmts, inputs, {}, branches, 0)
        return branches
    except Exception:
        return set()


def _trace_branches(stmts, inputs, env, branches, counter):
    """Trace execution to collect branch decisions."""
    env = dict(env)
    for stmt in stmts:
        if isinstance(stmt, LetDecl):
            if stmt.name in inputs:
                env[stmt.name] = inputs[stmt.name]
            else:
                try:
                    interp = _FuzzInterpreter(inputs)
                    interp.env = dict(env)
                    env[stmt.name] = interp._eval(stmt.value)
                except Exception:
                    env[stmt.name] = 0
        elif isinstance(stmt, Assign):
            try:
                interp = _FuzzInterpreter(inputs)
                interp.env = dict(env)
                env[stmt.name] = interp._eval(stmt.value)
            except Exception:
                pass
        elif isinstance(stmt, IfStmt):
            try:
                interp = _FuzzInterpreter(inputs)
                interp.env = dict(env)
                cond = interp._eval(stmt.cond)
                took_then = bool(cond)
                branches.add((counter, took_then))
                counter += 1
                if took_then:
                    _trace_branches(stmt.then_body.stmts, inputs, env, branches, counter)
                elif stmt.else_body:
                    body = stmt.else_body
                    if hasattr(body, 'stmts'):
                        _trace_branches(body.stmts, inputs, env, branches, counter)
            except Exception:
                counter += 1
        elif isinstance(stmt, WhileStmt):
            try:
                interp = _FuzzInterpreter(inputs)
                interp.env = dict(env)
                cond = interp._eval(stmt.cond)
                branches.add((counter, bool(cond)))
                counter += 1
            except Exception:
                counter += 1


# ---------------------------------------------------------------------------
# Mutation engine
# ---------------------------------------------------------------------------

class MutationEngine:
    """Generate mutated inputs from seed inputs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate(self, inputs: Dict[str, int], strength: int = 1) -> Dict[str, int]:
        """Mutate inputs with given strength (1=small, 2=medium, 3=large)."""
        result = dict(inputs)
        if not result:
            return result

        var = self.rng.choice(list(result.keys()))

        if strength == 1:
            # Small: +/- 1..3
            delta = self.rng.randint(1, 3) * self.rng.choice([-1, 1])
            result[var] = result[var] + delta
        elif strength == 2:
            # Medium: bit flip, multiply, or larger delta
            op = self.rng.randint(0, 2)
            if op == 0:
                bit = self.rng.randint(0, 7)
                result[var] = result[var] ^ (1 << bit)
            elif op == 1:
                result[var] = result[var] * self.rng.choice([-1, 2, -2, 0])
            else:
                delta = self.rng.randint(1, 100) * self.rng.choice([-1, 1])
                result[var] = result[var] + delta
        else:
            # Large: extreme values, random
            result[var] = self.rng.choice([
                0, 1, -1, 127, -128, 255, 256, 1000, -1000,
                2**15 - 1, -(2**15), self.rng.randint(-10000, 10000)
            ])

        return result

    def mutate_batch(self, inputs: Dict[str, int], count: int,
                     strength: int = 1) -> List[Dict[str, int]]:
        """Generate multiple mutations."""
        seen = set()
        results = []
        for _ in range(count * 3):  # oversample to get unique
            m = self.mutate(inputs, strength)
            key = tuple(sorted(m.items()))
            if key not in seen:
                seen.add(key)
                results.append(m)
            if len(results) >= count:
                break
        return results

    def boundary_mutate(self, inputs: Dict[str, int],
                        boundary_values: List[int] = None) -> List[Dict[str, int]]:
        """Generate inputs near common boundary values."""
        if boundary_values is None:
            boundary_values = [0, 1, -1, 2, -2, 10, -10, 100, -100]

        results = []
        seen = set()
        for var in inputs:
            for bv in boundary_values:
                for delta in [-1, 0, 1]:
                    m = dict(inputs)
                    m[var] = bv + delta
                    key = tuple(sorted(m.items()))
                    if key not in seen:
                        seen.add(key)
                        results.append(m)
        return results


# ---------------------------------------------------------------------------
# Verification-Driven Fuzzer
# ---------------------------------------------------------------------------

class VerificationDrivenFuzzer:
    """Main fuzzing engine that uses verification to guide input generation.

    Pipeline:
    1. Symbolic execution -> discover paths, extract test cases
    2. Fault localization -> rank suspicious statements
    3. Concolic testing -> target uncovered branches
    4. Mutation fuzzing -> explore neighborhoods of interesting inputs
    5. Combine all findings
    """

    def __init__(self, max_total_inputs: int = 200, mutation_rounds: int = 3,
                 mutation_per_seed: int = 10, random_count: int = 20,
                 seed: int = 42):
        self.max_total_inputs = max_total_inputs
        self.mutation_rounds = mutation_rounds
        self.mutation_per_seed = mutation_per_seed
        self.random_count = random_count
        self.mutator = MutationEngine(seed=seed)
        self.rng = random.Random(seed)

    def fuzz(self, source: str, input_vars: Dict[str, str],
             oracle_fn: Optional[Callable] = None,
             initial_inputs: Optional[Dict[str, int]] = None) -> FuzzResult:
        """Run full verification-driven fuzzing pipeline."""
        result = FuzzResult()
        coverage = CoverageInfo()
        tested = set()  # dedup by input tuple

        def _record(inputs: Dict[str, int], source_tag: str) -> bool:
            """Test an input, record coverage. Returns True if new coverage."""
            key = tuple(sorted(inputs.items()))
            if key in tested:
                return False
            tested.add(key)
            result.total_inputs_tested += 1
            result.all_test_inputs.append(FuzzInput(values=inputs, source=source_tag))

            # Execute and check for crashes
            output, crashed, err_msg = _safe_execute(source, inputs)
            if crashed:
                result.findings.append(FuzzFinding(
                    kind="assertion_failure" if "assertion" in err_msg else "crash",
                    inputs=inputs,
                    description=err_msg,
                    source=source_tag,
                ))

            # Track coverage
            branches = _extract_branch_count(source, inputs)
            new_cov = False
            for br in branches:
                if br not in coverage.covered_branches:
                    new_cov = True
                coverage.add(br, inputs)

            # Oracle check
            if oracle_fn and output is not None:
                try:
                    if not oracle_fn(inputs, output):
                        result.findings.append(FuzzFinding(
                            kind="assertion_failure",
                            inputs=inputs,
                            description="oracle check failed",
                            source=source_tag,
                        ))
                except Exception:
                    pass

            return new_cov

        # Budget check helper
        def _budget_ok() -> bool:
            return result.total_inputs_tested < self.max_total_inputs

        # ---------------------------------------------------------------
        # Phase 1: Guided symbolic execution
        # ---------------------------------------------------------------
        symbolic_inputs_dict = {v: "int" for v in input_vars}
        sym_inputs_for_v001 = {v: None for v in input_vars}

        try:
            guided = guided_execute(source, sym_inputs_for_v001)
            for tc in guided.test_cases:
                if not _budget_ok():
                    break
                inp = {}
                for v in input_vars:
                    if v in tc:
                        inp[v] = tc[v]
                    else:
                        inp[v] = 0
                if _record(inp, "symbolic"):
                    result.symbolic_inputs += 1
                else:
                    result.symbolic_inputs += 1
            result.symbolic_inputs = sum(
                1 for fi in result.all_test_inputs if fi.source == "symbolic"
            )
        except Exception:
            pass

        if not _budget_ok():
            result.status = FuzzStatus.BUDGET_EXHAUSTED
            result.coverage = coverage
            return result

        # ---------------------------------------------------------------
        # Phase 2: Concolic testing for additional coverage
        # ---------------------------------------------------------------
        if initial_inputs is None:
            initial_inputs = {v: 0 for v in input_vars}

        try:
            concolic_result = concolic_test(
                source, symbolic_inputs_dict, initial_inputs, max_iterations=30
            )
            for tc in concolic_result.test_cases:
                if not _budget_ok():
                    break
                _record(tc.inputs, "concolic")
            result.concolic_inputs = sum(
                1 for fi in result.all_test_inputs if fi.source == "concolic"
            )
        except Exception:
            pass

        if not _budget_ok():
            result.status = FuzzStatus.BUDGET_EXHAUSTED
            result.coverage = coverage
            return result

        # ---------------------------------------------------------------
        # Phase 3: Fault localization to identify suspicious statements
        # ---------------------------------------------------------------
        try:
            fault_result = auto_localize(
                source, symbolic_inputs_dict, oracle_fn, max_paths=32
            )
            result.suspicious_stmts = fault_result.ranked_statements[:10]
        except Exception:
            pass

        # ---------------------------------------------------------------
        # Phase 4: Boundary mutation (around branch-critical values)
        # ---------------------------------------------------------------
        # Collect seed inputs from phases 1+2
        seed_inputs = []
        for fi in result.all_test_inputs:
            seed_inputs.append(fi.values)

        # Extract boundary values from the source
        boundary_vals = _extract_boundary_values(source)

        for seed in seed_inputs[:10]:  # top 10 seeds
            if not _budget_ok():
                break
            boundary_mutations = self.mutator.boundary_mutate(seed, boundary_vals)
            for m in boundary_mutations:
                if not _budget_ok():
                    break
                _record(m, "boundary")

        result.boundary_inputs = sum(
            1 for fi in result.all_test_inputs if fi.source == "boundary"
        )

        if not _budget_ok():
            result.status = FuzzStatus.BUDGET_EXHAUSTED
            result.coverage = coverage
            return result

        # ---------------------------------------------------------------
        # Phase 5: Mutation fuzzing
        # ---------------------------------------------------------------
        # Prioritize seeds that found new coverage or are near bugs
        priority_seeds = []
        for fi in result.all_test_inputs:
            if any(f.inputs == fi.values for f in result.findings):
                priority_seeds.append(fi.values)

        # Also add seeds that reached new branches
        if not priority_seeds:
            priority_seeds = seed_inputs[:5]

        for round_num in range(self.mutation_rounds):
            if not _budget_ok():
                break
            strength = min(round_num + 1, 3)
            for seed in priority_seeds[:5]:
                if not _budget_ok():
                    break
                mutations = self.mutator.mutate_batch(
                    seed, self.mutation_per_seed, strength
                )
                for m in mutations:
                    if not _budget_ok():
                        break
                    _record(m, "mutation")

        result.mutation_inputs = sum(
            1 for fi in result.all_test_inputs if fi.source == "mutation"
        )

        # ---------------------------------------------------------------
        # Phase 6: Random inputs to fill budget
        # ---------------------------------------------------------------
        for _ in range(self.random_count):
            if not _budget_ok():
                break
            rand_inp = {v: self.rng.randint(-100, 100) for v in input_vars}
            _record(rand_inp, "random")

        result.random_inputs = sum(
            1 for fi in result.all_test_inputs if fi.source == "random"
        )

        # ---------------------------------------------------------------
        # Finalize
        # ---------------------------------------------------------------
        # Estimate total branches from coverage info
        if coverage.covered_branches:
            branch_ids = set(br[0] for br in coverage.covered_branches)
            coverage.total_branches = len(branch_ids) * 2  # each branch has true/false

        result.coverage = coverage
        if result.has_bugs:
            result.status = FuzzStatus.BUG_FOUND
        elif result.total_inputs_tested >= self.max_total_inputs:
            result.status = FuzzStatus.BUDGET_EXHAUSTED
        else:
            result.status = FuzzStatus.COMPLETE

        return result


# ---------------------------------------------------------------------------
# Targeted fuzzer: focus on specific branches or statements
# ---------------------------------------------------------------------------

class TargetedFuzzer:
    """Fuzz targeting specific branches or suspicious statements."""

    def __init__(self, max_inputs: int = 50, seed: int = 42):
        self.max_inputs = max_inputs
        self.mutator = MutationEngine(seed=seed)

    def fuzz_branch(self, source: str, input_vars: Dict[str, str],
                    target_branch: int, target_direction: bool,
                    initial_inputs: Optional[Dict[str, int]] = None) -> FuzzResult:
        """Fuzz targeting a specific branch direction."""
        result = FuzzResult()

        if initial_inputs is None:
            initial_inputs = {v: 0 for v in input_vars}

        # Try concolic first to reach the target branch
        try:
            reaching_inputs = concolic_reach_branch(
                source, input_vars, target_branch, target_direction,
                initial_inputs, max_iterations=20
            )
            if reaching_inputs:
                result.findings.append(FuzzFinding(
                    kind="boundary",
                    inputs=reaching_inputs,
                    description=f"reached branch {target_branch} dir={target_direction}",
                    source="concolic",
                ))
                result.concolic_inputs = 1
                result.total_inputs_tested = 1
                result.all_test_inputs.append(
                    FuzzInput(values=reaching_inputs, source="concolic")
                )

                # Mutate around the reaching input
                for strength in [1, 2]:
                    mutations = self.mutator.mutate_batch(
                        reaching_inputs, self.max_inputs // 4, strength
                    )
                    for m in mutations:
                        if result.total_inputs_tested >= self.max_inputs:
                            break
                        output, crashed, err = _safe_execute(source, m)
                        result.total_inputs_tested += 1
                        result.all_test_inputs.append(
                            FuzzInput(values=m, source="mutation")
                        )
                        if crashed:
                            result.findings.append(FuzzFinding(
                                kind="crash" if "assertion" not in err else "assertion_failure",
                                inputs=m,
                                description=err,
                                source="mutation",
                            ))
        except Exception:
            pass

        result.mutation_inputs = sum(
            1 for fi in result.all_test_inputs if fi.source == "mutation"
        )
        result.status = FuzzStatus.BUG_FOUND if result.has_bugs else FuzzStatus.COMPLETE
        return result

    def fuzz_suspicious(self, source: str, input_vars: Dict[str, str],
                        suspicious: List[SuspiciousnessScore],
                        oracle_fn: Optional[Callable] = None) -> FuzzResult:
        """Fuzz targeting suspicious statements from fault localization."""
        result = FuzzResult()

        if not suspicious:
            return result

        # Generate inputs via concolic testing
        try:
            concolic_result = concolic_test(
                source, input_vars, max_iterations=20
            )
            seeds = [tc.inputs for tc in concolic_result.test_cases]
        except Exception:
            seeds = [{v: 0 for v in input_vars}]

        tested = set()
        for seed in seeds[:5]:
            for strength in [1, 2, 3]:
                mutations = self.mutator.mutate_batch(
                    seed, self.max_inputs // 10, strength
                )
                for m in mutations:
                    key = tuple(sorted(m.items()))
                    if key in tested:
                        continue
                    tested.add(key)
                    if result.total_inputs_tested >= self.max_inputs:
                        break

                    output, crashed, err = _safe_execute(source, m)
                    result.total_inputs_tested += 1
                    result.all_test_inputs.append(
                        FuzzInput(values=m, source="mutation")
                    )

                    if crashed:
                        # Check if crash is near a suspicious statement
                        sus_stmt = suspicious[0].statement if suspicious else None
                        result.findings.append(FuzzFinding(
                            kind="crash" if "assertion" not in err else "assertion_failure",
                            inputs=m,
                            description=err,
                            source="mutation",
                            suspicious_stmt=sus_stmt,
                        ))

                    if oracle_fn and output is not None:
                        try:
                            if not oracle_fn(m, output):
                                result.findings.append(FuzzFinding(
                                    kind="assertion_failure",
                                    inputs=m,
                                    description="oracle check failed",
                                    source="mutation",
                                    suspicious_stmt=suspicious[0].statement if suspicious else None,
                                ))
                        except Exception:
                            pass

        result.mutation_inputs = result.total_inputs_tested
        result.suspicious_stmts = suspicious
        result.status = FuzzStatus.BUG_FOUND if result.has_bugs else FuzzStatus.COMPLETE
        return result


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def detect_divergence(source: str, input_vars: Dict[str, str],
                      reference_fn: Callable,
                      max_inputs: int = 100,
                      seed: int = 42) -> FuzzResult:
    """Fuzz to find inputs where program diverges from a reference function.

    reference_fn: takes Dict[str, int] -> expected output list
    """
    result = FuzzResult()
    fuzzer = VerificationDrivenFuzzer(
        max_total_inputs=max_inputs, seed=seed
    )

    # Use guided symex to get initial inputs
    sym_inputs = {v: None for v in input_vars}
    try:
        guided = guided_execute(source, sym_inputs)
        seeds = []
        for tc in guided.test_cases:
            inp = {v: tc.get(v, 0) for v in input_vars}
            seeds.append(inp)
    except Exception:
        seeds = [{v: 0 for v in input_vars}]

    mutator = MutationEngine(seed=seed)
    tested = set()

    # Test seeds and their mutations
    all_inputs = list(seeds)
    for seed_inp in seeds[:10]:
        all_inputs.extend(mutator.mutate_batch(seed_inp, 10, strength=1))
        all_inputs.extend(mutator.boundary_mutate(seed_inp))

    for inp in all_inputs:
        key = tuple(sorted(inp.items()))
        if key in tested:
            continue
        tested.add(key)
        if result.total_inputs_tested >= max_inputs:
            break

        result.total_inputs_tested += 1
        result.all_test_inputs.append(FuzzInput(values=inp, source="mutation"))

        actual_output, crashed, err = _safe_execute(source, inp)
        if crashed:
            result.findings.append(FuzzFinding(
                kind="crash",
                inputs=inp,
                description=err,
                source="divergence_check",
            ))
            continue

        try:
            expected = reference_fn(inp)
            if actual_output != expected:
                result.findings.append(FuzzFinding(
                    kind="divergence",
                    inputs=inp,
                    description=f"expected {expected}, got {actual_output}",
                    source="divergence_check",
                ))
        except Exception:
            pass

    result.mutation_inputs = result.total_inputs_tested
    result.status = FuzzStatus.BUG_FOUND if result.findings else FuzzStatus.COMPLETE
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_boundary_values(source: str) -> List[int]:
    """Extract integer constants from source code as boundary values."""
    values = set()
    try:
        program = _parse(source)
        _walk_for_ints(program.stmts, values)
    except Exception:
        pass
    # Always include common boundaries
    values.update([0, 1, -1])
    # Add +/- 1 neighbors
    extras = set()
    for v in list(values):
        extras.add(v - 1)
        extras.add(v + 1)
    values.update(extras)
    return sorted(values)


def _walk_for_ints(stmts, values: set):
    """Walk AST to collect integer literals."""
    for stmt in stmts:
        _walk_node(stmt, values)


def _walk_node(node, values: set):
    """Recursively walk AST node to find IntLit values."""
    if isinstance(node, IntLit):
        values.add(node.value)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _walk_node(item, values)
        return
    if hasattr(node, '__dict__'):
        for attr_val in node.__dict__.values():
            if isinstance(attr_val, int) and not isinstance(attr_val, bool):
                # Only small ints are likely boundaries
                if -10000 <= attr_val <= 10000:
                    values.add(attr_val)
            elif hasattr(attr_val, '__dict__') or isinstance(attr_val, (list, tuple)):
                _walk_node(attr_val, values)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def verification_fuzz(source: str, input_vars: Dict[str, str],
                      oracle_fn: Optional[Callable] = None,
                      max_inputs: int = 200,
                      seed: int = 42) -> FuzzResult:
    """Run full verification-driven fuzzing on a C10 program.

    Args:
        source: C10 source code
        input_vars: map of input variable names to types (e.g., {'x': 'int'})
        oracle_fn: optional function(inputs, output) -> bool for correctness
        max_inputs: maximum number of inputs to test
        seed: random seed for reproducibility

    Returns:
        FuzzResult with findings, coverage, and test inputs
    """
    fuzzer = VerificationDrivenFuzzer(
        max_total_inputs=max_inputs, seed=seed
    )
    return fuzzer.fuzz(source, input_vars, oracle_fn)


def quick_fuzz(source: str, input_vars: Dict[str, str],
               max_inputs: int = 50) -> FuzzResult:
    """Quick fuzzing with smaller budget."""
    fuzzer = VerificationDrivenFuzzer(
        max_total_inputs=max_inputs, mutation_rounds=1,
        mutation_per_seed=5, random_count=10
    )
    return fuzzer.fuzz(source, input_vars)


def deep_fuzz(source: str, input_vars: Dict[str, str],
              oracle_fn: Optional[Callable] = None,
              max_inputs: int = 500) -> FuzzResult:
    """Deep fuzzing with larger budget and more mutation rounds."""
    fuzzer = VerificationDrivenFuzzer(
        max_total_inputs=max_inputs, mutation_rounds=5,
        mutation_per_seed=20, random_count=50
    )
    return fuzzer.fuzz(source, input_vars, oracle_fn)


def fuzz_with_localization(source: str, input_vars: Dict[str, str],
                           oracle_fn: Optional[Callable] = None) -> FuzzResult:
    """Fuzz guided by fault localization results."""
    # First run fault localization
    try:
        fault_result = auto_localize(source, input_vars, oracle_fn, max_paths=32)
        suspicious = fault_result.ranked_statements[:10]
    except Exception:
        suspicious = []

    targeted = TargetedFuzzer(max_inputs=100)
    return targeted.fuzz_suspicious(source, input_vars, suspicious, oracle_fn)
