"""
V006: Equivalence Checking
Prove two programs compute the same function.

Composes C038 (Symbolic Execution) + C037 (SMT Solver) + C010 (Parser)

Architecture:
  Source1 -> Symbolic Execution -> Paths1 (constraints + outputs)
  Source2 -> Symbolic Execution -> Paths2 (constraints + outputs)
  For each (p1, p2) pair: check constraints(p1) AND constraints(p2) AND output(p1) != output(p2)
  If any pair is SAT -> NOT equivalent (counterexample found)
  If all pairs are UNSAT -> EQUIVALENT

Features:
  - Function equivalence: compare return values of two functions
  - Program equivalence: compare outputs of two programs
  - Counterexample generation when programs differ
  - Partial equivalence: equivalent under restricted input domains
  - Regression checking: verify refactored code matches original
  - Output sequence equivalence (print statements)
  - Variable mapping: compare programs with different variable names
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from enum import Enum

# Import C038 symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, ExecutionResult, PathState, PathStatus,
    SymValue, SymType, smt_not, smt_and, smt_or,
    TestCase
)

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var as SMTVar, App,
    IntConst, BoolConst, Op as SMTOp, BOOL, INT
)

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, FnDecl, Block, ReturnStmt
)


# ============================================================
# Result Types
# ============================================================

class EquivResult(Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    UNKNOWN = "unknown"


@dataclass
class Counterexample:
    """A concrete input where two programs produce different outputs."""
    inputs: Dict[str, any]
    output1: any  # Output of program 1
    output2: any  # Output of program 2
    path1_id: int = 0
    path2_id: int = 0


@dataclass
class EquivCheckResult:
    """Result of an equivalence check."""
    result: EquivResult
    counterexample: Optional[Counterexample] = None
    paths_checked: int = 0
    path_pairs_checked: int = 0
    equivalent_pairs: int = 0
    stats: Dict[str, any] = field(default_factory=dict)

    @property
    def is_equivalent(self) -> bool:
        return self.result == EquivResult.EQUIVALENT


@dataclass
class PartialEquivResult:
    """Result of a partial equivalence check (under domain constraints)."""
    result: EquivResult
    domain_description: str = ""
    counterexample: Optional[Counterexample] = None
    pairs_checked: int = 0


# ============================================================
# Core: SMT-based output comparison
# ============================================================

def _terms_structurally_equal(t1: Term, t2: Term) -> bool:
    """Check if two SMT terms are structurally identical."""
    if t1 is t2:
        return True
    if type(t1) != type(t2):
        return False
    if isinstance(t1, IntConst) and isinstance(t2, IntConst):
        return t1.value == t2.value
    if isinstance(t1, BoolConst) and isinstance(t2, BoolConst):
        return t1.value == t2.value
    if isinstance(t1, SMTVar) and isinstance(t2, SMTVar):
        return t1.name == t2.name and t1.sort == t2.sort
    if isinstance(t1, App) and isinstance(t2, App):
        if t1.op != t2.op or t1.sort != t2.sort:
            return False
        if len(t1.args) != len(t2.args):
            return False
        return all(_terms_structurally_equal(a, b) for a, b in zip(t1.args, t2.args))
    return False


def _collect_vars_from_term(term, seen: set):
    """Collect all SMT variables from a term."""
    if isinstance(term, SMTVar):
        sort = 'bool' if term.sort == BOOL else 'int'
        seen.add((term.name, sort))
    elif isinstance(term, App):
        for child in term.args:
            _collect_vars_from_term(child, seen)


def _declare_vars_in_solver(solver: SMTSolver, constraints: list, extra_vars: dict = None):
    """Declare all variables appearing in constraints."""
    seen = set()
    for c in constraints:
        _collect_vars_from_term(c, seen)
    for name, sort in seen:
        if sort == 'int':
            solver.Int(name)
        elif sort == 'bool':
            solver.Bool(name)
    if extra_vars:
        for name, typ in extra_vars.items():
            if typ == 'int':
                solver.Int(name)
            elif typ == 'bool':
                solver.Bool(name)


def _symval_to_term(val: SymValue) -> Optional[Term]:
    """Convert a SymValue to an SMT term for comparison."""
    if val is None:
        return None
    if val.is_symbolic():
        return val.term
    if val.is_concrete():
        v = val.concrete
        if isinstance(v, bool):
            return IntConst(1 if v else 0)
        if isinstance(v, int):
            return IntConst(v)
        if isinstance(v, float):
            return IntConst(int(v))
    return None


def _check_output_equivalence(
    path1: PathState, path2: PathState,
    symbolic_inputs: dict,
    output_mode: str = "return"
) -> Tuple[bool, Optional[dict]]:
    """
    Check if two paths produce the same output when both are feasible.

    Returns (equivalent, counterexample_model).
    equivalent=True means outputs match under shared constraints.
    If not equivalent, counterexample_model gives a distinguishing input.
    """
    # Gather all constraints
    all_constraints = list(path1.constraints) + list(path2.constraints)

    # Get the output terms to compare
    if output_mode == "return":
        term1 = _symval_to_term(path1.return_value)
        term2 = _symval_to_term(path2.return_value)
    elif output_mode == "env":
        # Compare all shared env variables (for programs without explicit return)
        return _check_env_equivalence(path1, path2, symbolic_inputs)
    elif output_mode == "output":
        return _check_print_equivalence(path1, path2, symbolic_inputs)
    else:
        return True, None

    if term1 is None and term2 is None:
        # Both return None -- equivalent
        return True, None

    if term1 is None or term2 is None:
        # One returns a value, the other doesn't -- check if constraint region is feasible
        solver = SMTSolver()
        _declare_vars_in_solver(solver, all_constraints, symbolic_inputs)
        for c in all_constraints:
            solver.add(c)
        r = solver.check()
        if r == SMTResult.SAT:
            return False, solver.model()
        return True, None  # Infeasible overlap

    # Structural equality shortcut
    if _terms_structurally_equal(term1, term2):
        return True, None

    # Build: constraints(p1) AND constraints(p2) AND output1 != output2
    neq = App(SMTOp.NEQ, [term1, term2], BOOL)
    check_constraints = all_constraints + [neq]

    solver = SMTSolver()
    _declare_vars_in_solver(solver, check_constraints, symbolic_inputs)
    for c in check_constraints:
        solver.add(c)

    r = solver.check()
    if r == SMTResult.SAT:
        return False, solver.model()
    return True, None


def _check_env_equivalence(
    path1: PathState, path2: PathState,
    symbolic_inputs: dict
) -> Tuple[bool, Optional[dict]]:
    """Check if two paths produce the same environment for non-input variables."""
    all_constraints = list(path1.constraints) + list(path2.constraints)

    # Find non-input variables present in both envs
    input_names = set(symbolic_inputs.keys()) if symbolic_inputs else set()
    shared_vars = (set(path1.env.keys()) & set(path2.env.keys())) - input_names

    if not shared_vars:
        return True, None

    for var_name in shared_vars:
        term1 = _symval_to_term(path1.env[var_name])
        term2 = _symval_to_term(path2.env[var_name])

        if term1 is None and term2 is None:
            continue
        if term1 is None or term2 is None:
            solver = SMTSolver()
            _declare_vars_in_solver(solver, all_constraints, symbolic_inputs)
            for c in all_constraints:
                solver.add(c)
            r = solver.check()
            if r == SMTResult.SAT:
                return False, solver.model()
            continue

        if _terms_structurally_equal(term1, term2):
            continue

        neq = App(SMTOp.NEQ, [term1, term2], BOOL)
        check_constraints = all_constraints + [neq]
        solver = SMTSolver()
        _declare_vars_in_solver(solver, check_constraints, symbolic_inputs)
        for c in check_constraints:
            solver.add(c)
        r = solver.check()
        if r == SMTResult.SAT:
            return False, solver.model()

    return True, None


def _check_print_equivalence(
    path1: PathState, path2: PathState,
    symbolic_inputs: dict
) -> Tuple[bool, Optional[dict]]:
    """Check if two paths produce the same print output sequence."""
    all_constraints = list(path1.constraints) + list(path2.constraints)

    out1 = path1.output
    out2 = path2.output

    if len(out1) != len(out2):
        # Different number of outputs -- check if constraint region feasible
        solver = SMTSolver()
        _declare_vars_in_solver(solver, all_constraints, symbolic_inputs)
        for c in all_constraints:
            solver.add(c)
        r = solver.check()
        if r == SMTResult.SAT:
            return False, solver.model()
        return True, None

    for v1, v2 in zip(out1, out2):
        term1 = _symval_to_term(v1) if isinstance(v1, SymValue) else (
            IntConst(v1) if isinstance(v1, int) else None
        )
        term2 = _symval_to_term(v2) if isinstance(v2, SymValue) else (
            IntConst(v2) if isinstance(v2, int) else None
        )

        if term1 is None and term2 is None:
            continue
        if term1 is None or term2 is None:
            solver = SMTSolver()
            _declare_vars_in_solver(solver, all_constraints, symbolic_inputs)
            for c in all_constraints:
                solver.add(c)
            r = solver.check()
            if r == SMTResult.SAT:
                return False, solver.model()
            continue

        if _terms_structurally_equal(term1, term2):
            continue

        neq = App(SMTOp.NEQ, [term1, term2], BOOL)
        check_constraints = all_constraints + [neq]
        solver = SMTSolver()
        _declare_vars_in_solver(solver, check_constraints, symbolic_inputs)
        for c in check_constraints:
            solver.add(c)
        r = solver.check()
        if r == SMTResult.SAT:
            return False, solver.model()

    return True, None


# ============================================================
# Path coverage check: ensure both programs cover the same input space
# ============================================================

def _check_coverage_equivalence(
    result1: ExecutionResult, result2: ExecutionResult,
    symbolic_inputs: dict
) -> Tuple[bool, Optional[dict]]:
    """
    Check that both programs are defined on the same input domain.
    i.e., every feasible input of prog1 is also feasible in prog2 and vice versa.

    For symbolic execution with full path exploration, this is typically satisfied
    since both programs execute on all inputs. However, if one program errors on
    some path, we detect that.
    """
    # Check if prog1 has error paths that prog2 doesn't, or vice versa
    errors1 = [p for p in result1.paths if p.status == PathStatus.ERROR]
    errors2 = [p for p in result2.paths if p.status == PathStatus.ERROR]

    if errors1 and not errors2:
        # prog1 has error paths, prog2 doesn't
        for ep in errors1:
            if ep.constraints:
                solver = SMTSolver()
                _declare_vars_in_solver(solver, ep.constraints, symbolic_inputs)
                for c in ep.constraints:
                    solver.add(c)
                if solver.check() == SMTResult.SAT:
                    return False, solver.model()
        return True, None

    return True, None


# ============================================================
# Variable renaming support
# ============================================================

def _rename_source(source: str, var_map: Dict[str, str]) -> str:
    """
    Rename variables in source code according to var_map.
    Simple token-level renaming (not a full AST rewrite).
    """
    # Tokenize, rename, reconstruct
    tokens = lex(source)
    result_parts = []
    # This is a simple approach -- just replace in the source string
    result = source
    for old_name, new_name in var_map.items():
        # Replace whole-word occurrences
        import re
        result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, result)
    return result


# ============================================================
# Public API
# ============================================================

def check_function_equivalence(
    source1: str, fn_name1: str,
    source2: str, fn_name2: str,
    param_types: Dict[str, str],
    max_paths: int = 64,
    max_loop_unroll: int = 5,
) -> EquivCheckResult:
    """
    Check if two functions are equivalent (same return value for all inputs).

    Args:
        source1: Source code containing function 1
        fn_name1: Name of function 1
        source2: Source code containing function 2
        fn_name2: Name of function 2
        param_types: Dict of {param_name: 'int'|'bool'} for symbolic parameters
        max_paths: Max paths to explore per program
        max_loop_unroll: Max loop iterations

    Returns:
        EquivCheckResult with equivalence verdict and optional counterexample
    """
    # Build wrapper programs that call each function with symbolic inputs
    # and return the result
    param_decls = "\n".join(f"let {name} = 0;" for name in param_types)
    args = ", ".join(param_types.keys())

    wrapper1 = f"{source1}\n{param_decls}\nlet __result1 = {fn_name1}({args});"
    wrapper2 = f"{source2}\n{param_decls}\nlet __result2 = {fn_name2}({args});"

    # Run symbolic execution on both
    engine1 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result1 = engine1.execute(wrapper1, param_types)

    engine2 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result2 = engine2.execute(wrapper2, param_types)

    # Get feasible completed paths
    paths1 = [p for p in result1.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]
    paths2 = [p for p in result2.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]

    if not paths1 or not paths2:
        return EquivCheckResult(
            result=EquivResult.UNKNOWN,
            stats={'reason': 'no feasible paths', 'paths1': len(paths1), 'paths2': len(paths2)}
        )

    # Check all path pairs
    pairs_checked = 0
    equivalent_pairs = 0

    for p1 in paths1:
        for p2 in paths2:
            pairs_checked += 1

            # First check if the path constraints overlap (both can be active)
            overlap_constraints = list(p1.constraints) + list(p2.constraints)
            solver = SMTSolver()
            _declare_vars_in_solver(solver, overlap_constraints, param_types)
            for c in overlap_constraints:
                solver.add(c)
            if solver.check() != SMTResult.SAT:
                equivalent_pairs += 1
                continue  # Paths don't overlap -- trivially equivalent here

            # Get return values from env (__result1 / __result2)
            ret1 = p1.env.get('__result1')
            ret2 = p2.env.get('__result2')

            # Also check return_value
            if ret1 is None:
                ret1 = p1.return_value
            if ret2 is None:
                ret2 = p2.return_value

            term1 = _symval_to_term(ret1)
            term2 = _symval_to_term(ret2)

            if term1 is None and term2 is None:
                equivalent_pairs += 1
                continue

            if term1 is None or term2 is None:
                # One returns a value, the other doesn't
                model = solver.model()
                return EquivCheckResult(
                    result=EquivResult.NOT_EQUIVALENT,
                    counterexample=_build_counterexample(
                        model, param_types, ret1, ret2, p1, p2
                    ),
                    paths_checked=len(paths1) + len(paths2),
                    path_pairs_checked=pairs_checked,
                    equivalent_pairs=equivalent_pairs,
                )

            # Structural equality shortcut (handles nonlinear terms)
            if _terms_structurally_equal(term1, term2):
                equivalent_pairs += 1
                continue

            # Check: overlap_constraints AND term1 != term2
            neq = App(SMTOp.NEQ, [term1, term2], BOOL)
            neq_constraints = overlap_constraints + [neq]

            neq_solver = SMTSolver()
            _declare_vars_in_solver(neq_solver, neq_constraints, param_types)
            for c in neq_constraints:
                neq_solver.add(c)

            r = neq_solver.check()
            if r == SMTResult.SAT:
                model = neq_solver.model()
                return EquivCheckResult(
                    result=EquivResult.NOT_EQUIVALENT,
                    counterexample=_build_counterexample(
                        model, param_types, ret1, ret2, p1, p2
                    ),
                    paths_checked=len(paths1) + len(paths2),
                    path_pairs_checked=pairs_checked,
                    equivalent_pairs=equivalent_pairs,
                )
            equivalent_pairs += 1

    return EquivCheckResult(
        result=EquivResult.EQUIVALENT,
        paths_checked=len(paths1) + len(paths2),
        path_pairs_checked=pairs_checked,
        equivalent_pairs=equivalent_pairs,
    )


def check_program_equivalence(
    source1: str, source2: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    mode: str = "auto",
    max_paths: int = 64,
    max_loop_unroll: int = 5,
) -> EquivCheckResult:
    """
    Check if two programs are equivalent.

    Args:
        source1, source2: Source code of the two programs
        symbolic_inputs: Dict of {var_name: 'int'|'bool'}
        output_var: Variable name to compare (if mode='env')
        mode: 'auto', 'env', 'output', or 'return'
            - 'auto': try env (output_var) first, then output, then return
            - 'env': compare specific variable in environment
            - 'output': compare print output sequences
            - 'return': compare return values
        max_paths: Max paths to explore per program
        max_loop_unroll: Max loop iterations

    Returns:
        EquivCheckResult
    """
    engine1 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result1 = engine1.execute(source1, symbolic_inputs)

    engine2 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result2 = engine2.execute(source2, symbolic_inputs)

    paths1 = [p for p in result1.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]
    paths2 = [p for p in result2.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]

    if not paths1 or not paths2:
        return EquivCheckResult(
            result=EquivResult.UNKNOWN,
            stats={'reason': 'no feasible paths'}
        )

    # Determine output mode
    if mode == "auto":
        if output_var:
            mode = "env"
        elif any(p.output for p in paths1) or any(p.output for p in paths2):
            mode = "output"
        else:
            mode = "env"

    pairs_checked = 0
    equivalent_pairs = 0

    for p1 in paths1:
        for p2 in paths2:
            pairs_checked += 1

            # Check constraint overlap
            overlap_constraints = list(p1.constraints) + list(p2.constraints)
            solver = SMTSolver()
            _declare_vars_in_solver(solver, overlap_constraints, symbolic_inputs)
            for c in overlap_constraints:
                solver.add(c)
            if solver.check() != SMTResult.SAT:
                equivalent_pairs += 1
                continue

            if mode == "env":
                if output_var:
                    # Compare a specific variable
                    val1 = p1.env.get(output_var)
                    val2 = p2.env.get(output_var)
                    term1 = _symval_to_term(val1)
                    term2 = _symval_to_term(val2)

                    if term1 is None and term2 is None:
                        equivalent_pairs += 1
                        continue
                    if term1 is None or term2 is None:
                        model = solver.model()
                        return EquivCheckResult(
                            result=EquivResult.NOT_EQUIVALENT,
                            counterexample=_build_counterexample(
                                model, symbolic_inputs, val1, val2, p1, p2
                            ),
                            paths_checked=len(paths1) + len(paths2),
                            path_pairs_checked=pairs_checked,
                            equivalent_pairs=equivalent_pairs,
                        )

                    if _terms_structurally_equal(term1, term2):
                        equivalent_pairs += 1
                        continue

                    neq = App(SMTOp.NEQ, [term1, term2], BOOL)
                    neq_solver = SMTSolver()
                    neq_constraints = overlap_constraints + [neq]
                    _declare_vars_in_solver(neq_solver, neq_constraints, symbolic_inputs)
                    for c in neq_constraints:
                        neq_solver.add(c)
                    r = neq_solver.check()
                    if r == SMTResult.SAT:
                        model = neq_solver.model()
                        return EquivCheckResult(
                            result=EquivResult.NOT_EQUIVALENT,
                            counterexample=_build_counterexample(
                                model, symbolic_inputs, val1, val2, p1, p2
                            ),
                            paths_checked=len(paths1) + len(paths2),
                            path_pairs_checked=pairs_checked,
                            equivalent_pairs=equivalent_pairs,
                        )
                    equivalent_pairs += 1
                else:
                    # Compare all non-input env vars
                    eq, model = _check_env_equivalence(p1, p2, symbolic_inputs)
                    if not eq:
                        return EquivCheckResult(
                            result=EquivResult.NOT_EQUIVALENT,
                            counterexample=Counterexample(
                                inputs=_extract_inputs(model, symbolic_inputs),
                                output1=_env_snapshot(p1, symbolic_inputs),
                                output2=_env_snapshot(p2, symbolic_inputs),
                                path1_id=p1.path_id,
                                path2_id=p2.path_id,
                            ),
                            paths_checked=len(paths1) + len(paths2),
                            path_pairs_checked=pairs_checked,
                            equivalent_pairs=equivalent_pairs,
                        )
                    equivalent_pairs += 1

            elif mode == "output":
                eq, model = _check_print_equivalence(p1, p2, symbolic_inputs)
                if not eq:
                    return EquivCheckResult(
                        result=EquivResult.NOT_EQUIVALENT,
                        counterexample=Counterexample(
                            inputs=_extract_inputs(model, symbolic_inputs),
                            output1=p1.output,
                            output2=p2.output,
                            path1_id=p1.path_id,
                            path2_id=p2.path_id,
                        ),
                        paths_checked=len(paths1) + len(paths2),
                        path_pairs_checked=pairs_checked,
                        equivalent_pairs=equivalent_pairs,
                    )
                equivalent_pairs += 1

            elif mode == "return":
                eq, model = _check_output_equivalence(p1, p2, symbolic_inputs, "return")
                if not eq:
                    return EquivCheckResult(
                        result=EquivResult.NOT_EQUIVALENT,
                        counterexample=_build_counterexample(
                            model, symbolic_inputs,
                            p1.return_value, p2.return_value, p1, p2
                        ),
                        paths_checked=len(paths1) + len(paths2),
                        path_pairs_checked=pairs_checked,
                        equivalent_pairs=equivalent_pairs,
                    )
                equivalent_pairs += 1

    return EquivCheckResult(
        result=EquivResult.EQUIVALENT,
        paths_checked=len(paths1) + len(paths2),
        path_pairs_checked=pairs_checked,
        equivalent_pairs=equivalent_pairs,
    )


def check_equivalence_with_mapping(
    source1: str, source2: str,
    symbolic_inputs: Dict[str, str],
    var_map: Dict[str, str],
    output_var1: str = None,
    output_var2: str = None,
    max_paths: int = 64,
) -> EquivCheckResult:
    """
    Check equivalence with variable name mapping between programs.

    Args:
        source1, source2: Source code
        symbolic_inputs: Symbolic inputs for source1
        var_map: Mapping from source1 var names to source2 var names
        output_var1: Output variable in source1
        output_var2: Output variable in source2
        max_paths: Max paths

    Returns:
        EquivCheckResult
    """
    # Rename source2's variables to match source1
    inverse_map = {v: k for k, v in var_map.items()}
    renamed_source2 = _rename_source(source2, inverse_map)

    output_var = output_var1
    if output_var2 and output_var2 in inverse_map:
        output_var = inverse_map[output_var2]

    return check_program_equivalence(
        source1, renamed_source2, symbolic_inputs,
        output_var=output_var, max_paths=max_paths
    )


def check_partial_equivalence(
    source1: str, source2: str,
    symbolic_inputs: Dict[str, str],
    domain_constraints: List[Term],
    output_var: str = None,
    max_paths: int = 64,
) -> EquivCheckResult:
    """
    Check equivalence under restricted input domain.

    Args:
        source1, source2: Source code
        symbolic_inputs: Symbolic inputs
        domain_constraints: SMT terms restricting the input domain
        output_var: Variable to compare
        max_paths: Max paths

    Returns:
        EquivCheckResult
    """
    engine1 = SymbolicExecutor(max_paths=max_paths)
    result1 = engine1.execute(source1, symbolic_inputs)

    engine2 = SymbolicExecutor(max_paths=max_paths)
    result2 = engine2.execute(source2, symbolic_inputs)

    paths1 = [p for p in result1.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]
    paths2 = [p for p in result2.paths
               if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]

    if not paths1 or not paths2:
        return EquivCheckResult(result=EquivResult.UNKNOWN)

    pairs_checked = 0
    equivalent_pairs = 0

    for p1 in paths1:
        for p2 in paths2:
            pairs_checked += 1

            # Include domain constraints
            overlap_constraints = (
                list(p1.constraints) + list(p2.constraints) + list(domain_constraints)
            )
            solver = SMTSolver()
            _declare_vars_in_solver(solver, overlap_constraints, symbolic_inputs)
            for c in overlap_constraints:
                solver.add(c)
            if solver.check() != SMTResult.SAT:
                equivalent_pairs += 1
                continue

            # Compare output variable
            if output_var:
                val1 = p1.env.get(output_var)
                val2 = p2.env.get(output_var)
                term1 = _symval_to_term(val1)
                term2 = _symval_to_term(val2)

                if term1 is None and term2 is None:
                    equivalent_pairs += 1
                    continue
                if term1 is None or term2 is None:
                    model = solver.model()
                    return EquivCheckResult(
                        result=EquivResult.NOT_EQUIVALENT,
                        counterexample=_build_counterexample(
                            model, symbolic_inputs, val1, val2, p1, p2
                        ),
                        path_pairs_checked=pairs_checked,
                        equivalent_pairs=equivalent_pairs,
                    )

                if _terms_structurally_equal(term1, term2):
                    equivalent_pairs += 1
                    continue

                neq = App(SMTOp.NEQ, [term1, term2], BOOL)
                neq_constraints = overlap_constraints + [neq]
                neq_solver = SMTSolver()
                _declare_vars_in_solver(neq_solver, neq_constraints, symbolic_inputs)
                for c in neq_constraints:
                    neq_solver.add(c)
                r = neq_solver.check()
                if r == SMTResult.SAT:
                    model = neq_solver.model()
                    return EquivCheckResult(
                        result=EquivResult.NOT_EQUIVALENT,
                        counterexample=_build_counterexample(
                            model, symbolic_inputs, val1, val2, p1, p2
                        ),
                        path_pairs_checked=pairs_checked,
                        equivalent_pairs=equivalent_pairs,
                    )
                equivalent_pairs += 1
            else:
                # Compare full env
                eq, model = _check_env_equivalence(p1, p2, symbolic_inputs)
                if not eq:
                    return EquivCheckResult(
                        result=EquivResult.NOT_EQUIVALENT,
                        counterexample=Counterexample(
                            inputs=_extract_inputs(model, symbolic_inputs),
                            output1=_env_snapshot(p1, symbolic_inputs),
                            output2=_env_snapshot(p2, symbolic_inputs),
                        ),
                        path_pairs_checked=pairs_checked,
                        equivalent_pairs=equivalent_pairs,
                    )
                equivalent_pairs += 1

    return EquivCheckResult(
        result=EquivResult.EQUIVALENT,
        path_pairs_checked=pairs_checked,
        equivalent_pairs=equivalent_pairs,
    )


def check_regression(
    original_source: str, refactored_source: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    fn_name: str = None,
    param_types: Dict[str, str] = None,
    max_paths: int = 64,
) -> EquivCheckResult:
    """
    Verify that refactored code matches original behavior.
    Convenience wrapper that picks the right comparison mode.

    Args:
        original_source: Original program
        refactored_source: Refactored program
        symbolic_inputs: Symbolic inputs (used if fn_name is None)
        output_var: Variable to compare (optional)
        fn_name: If set, compare this function in both sources
        param_types: Parameter types for function comparison
        max_paths: Max paths

    Returns:
        EquivCheckResult
    """
    if fn_name:
        return check_function_equivalence(
            original_source, fn_name,
            refactored_source, fn_name,
            param_types or symbolic_inputs,
            max_paths=max_paths,
        )
    return check_program_equivalence(
        original_source, refactored_source,
        symbolic_inputs,
        output_var=output_var,
        max_paths=max_paths,
    )


# ============================================================
# Helpers
# ============================================================

def _build_counterexample(
    model: Optional[dict],
    symbolic_inputs: dict,
    ret1: Optional[SymValue],
    ret2: Optional[SymValue],
    p1: PathState,
    p2: PathState,
) -> Counterexample:
    """Build a Counterexample from an SMT model."""
    inputs = _extract_inputs(model, symbolic_inputs)

    out1 = _eval_symval(ret1, model) if ret1 else None
    out2 = _eval_symval(ret2, model) if ret2 else None

    return Counterexample(
        inputs=inputs,
        output1=out1,
        output2=out2,
        path1_id=p1.path_id,
        path2_id=p2.path_id,
    )


def _extract_inputs(model: Optional[dict], symbolic_inputs: dict) -> dict:
    """Extract input values from an SMT model."""
    inputs = {}
    for name in symbolic_inputs:
        if model and name in model:
            inputs[name] = model[name]
        else:
            inputs[name] = 0
    return inputs


def _eval_symval(val: SymValue, model: Optional[dict]) -> any:
    """Try to evaluate a SymValue given a model."""
    if val is None:
        return None
    if val.is_concrete():
        return val.concrete
    # For symbolic values, we'd need to evaluate the term under the model
    # For now, just return the term representation
    if val.name and model and val.name in model:
        return model[val.name]
    return str(val.term) if val.term else None


def _env_snapshot(path: PathState, symbolic_inputs: dict) -> dict:
    """Get a snapshot of non-input variables in a path's environment."""
    input_names = set(symbolic_inputs.keys()) if symbolic_inputs else set()
    snap = {}
    for k, v in path.env.items():
        if k not in input_names:
            if v.is_concrete():
                snap[k] = v.concrete
            else:
                snap[k] = str(v.term)
    return snap
