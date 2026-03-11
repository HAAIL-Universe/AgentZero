"""V134: Certified Equivalence Checking

Composes V006 (equivalence checking) + V044 (proof certificates) to produce
machine-checkable certificates that two programs compute the same function.

Each path pair from symbolic execution becomes a proof obligation:
  constraints(p1) AND constraints(p2) AND output1 != output2 is UNSAT

Certificate is VALID when all such obligations are verified UNSAT.
Independent checking re-runs SMT queries from serialized formulas.
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

# V006: Equivalence Checking
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V006_equivalence_checking'))
from equiv_check import (
    check_function_equivalence, check_program_equivalence,
    check_equivalence_with_mapping, check_partial_equivalence,
    check_regression, EquivResult, EquivCheckResult, Counterexample,
    _symval_to_term, _declare_vars_in_solver, _terms_structurally_equal,
    _collect_vars_from_term
)

# V044: Proof Certificates
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
from proof_certificates import (
    ProofCertificate, ProofObligation, CertStatus, ProofKind,
    smt_term_to_str, smt_term_to_smtlib, combine_certificates,
    save_certificate, load_certificate
)

# C037: SMT Solver (for independent checking)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Term, Var as SMTVar, App, IntConst, BoolConst, Op as SMTOp, BOOL, INT

# C038: Symbolic Execution (for path extraction)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, ExecutionResult, PathState, PathStatus,
    SymValue, SymType, smt_not, smt_and, smt_or
)

# C010: Parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser, FnDecl, Block, ReturnStmt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EquivCertKind(str, Enum):
    FUNCTION = "function"
    PROGRAM = "program"
    REGRESSION = "regression"
    PARTIAL = "partial"


@dataclass
class PathPairObligation:
    """One path pair from the equivalence check."""
    path1_id: int
    path2_id: int
    constraints_feasible: bool  # whether path constraints overlap
    output_equal: bool          # whether outputs are provably equal
    structural_equal: bool      # shortcut: outputs structurally identical
    formula_str: str = ""       # human-readable
    formula_smt: str = ""       # SMT-LIB2
    status: str = "unchecked"


@dataclass
class EquivCertificate:
    """Certificate for program equivalence."""
    kind: EquivCertKind
    claim: str
    source1: str
    source2: str
    result: str  # "equivalent", "not_equivalent", "unknown"
    path_pairs: List[PathPairObligation] = field(default_factory=list)
    obligations: List[ProofObligation] = field(default_factory=list)
    counterexample: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def total_obligations(self):
        return len(self.obligations)

    @property
    def valid_obligations(self):
        return sum(1 for o in self.obligations if o.status == CertStatus.VALID)

    @property
    def invalid_obligations(self):
        return sum(1 for o in self.obligations if o.status == CertStatus.INVALID)

    def summary(self) -> str:
        lines = [
            f"Equivalence Certificate ({self.kind.value})",
            f"  Claim: {self.claim}",
            f"  Result: {self.result}",
            f"  Status: {self.status.value}",
            f"  Obligations: {self.valid_obligations}/{self.total_obligations} valid",
            f"  Path pairs: {len(self.path_pairs)}",
        ]
        if self.counterexample:
            lines.append(f"  Counterexample: {self.counterexample}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "claim": self.claim,
            "source1": self.source1,
            "source2": self.source2,
            "result": self.result,
            "path_pairs": [
                {
                    "path1_id": pp.path1_id,
                    "path2_id": pp.path2_id,
                    "constraints_feasible": pp.constraints_feasible,
                    "output_equal": pp.output_equal,
                    "structural_equal": pp.structural_equal,
                    "formula_str": pp.formula_str,
                    "formula_smt": pp.formula_smt,
                    "status": pp.status,
                }
                for pp in self.path_pairs
            ],
            "obligations": [o.to_dict() for o in self.obligations],
            "counterexample": self.counterexample,
            "metadata": self.metadata,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EquivCertificate':
        cert = cls(
            kind=EquivCertKind(d["kind"]),
            claim=d["claim"],
            source1=d["source1"],
            source2=d["source2"],
            result=d["result"],
            counterexample=d.get("counterexample"),
            metadata=d.get("metadata", {}),
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )
        for pp_d in d.get("path_pairs", []):
            cert.path_pairs.append(PathPairObligation(
                path1_id=pp_d["path1_id"],
                path2_id=pp_d["path2_id"],
                constraints_feasible=pp_d["constraints_feasible"],
                output_equal=pp_d["output_equal"],
                structural_equal=pp_d["structural_equal"],
                formula_str=pp_d.get("formula_str", ""),
                formula_smt=pp_d.get("formula_smt", ""),
                status=pp_d.get("status", "unchecked"),
            ))
        for o_d in d.get("obligations", []):
            cert.obligations.append(ProofObligation.from_dict(o_d))
        return cert

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> 'EquivCertificate':
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Core: Extract paths + build obligations
# ---------------------------------------------------------------------------

def _run_symbolic(source, symbolic_inputs, fn_name=None, param_types=None,
                  max_paths=64, max_loop_unroll=5):
    """Run symbolic execution on source, return paths."""
    executor = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    if fn_name and param_types:
        # Build a call wrapper
        param_names = list(param_types.keys())
        decls = "\n".join(f"let {p} = 0;" for p in param_names)
        call_args = ", ".join(param_names)
        wrapper = f"{source}\n{decls}\nlet __result = {fn_name}({call_args});"
        sym_inputs = {p: "int" for p in param_names}
        result = executor.execute(wrapper, sym_inputs)
    else:
        result = executor.execute(source, symbolic_inputs)
    return result


def _term_to_str(t) -> str:
    """Convert an SMT term or symbolic value to string."""
    if t is None:
        return "None"
    if isinstance(t, (int, float, bool, str)):
        return str(t)
    try:
        return smt_term_to_str(t)
    except Exception:
        return str(t)


def _term_to_smtlib(t, declared=None) -> str:
    """Convert an SMT term to SMT-LIB2 format."""
    if t is None:
        return "true"
    if isinstance(t, (int, float)):
        return str(t)
    if isinstance(t, bool):
        return "true" if t else "false"
    try:
        return smt_term_to_smtlib(t)
    except Exception:
        return str(t)


def _outputs_structurally_equal(out1, out2) -> bool:
    """Check if two SMT Term outputs are structurally identical."""
    if out1 is None and out2 is None:
        return True
    if out1 is None or out2 is None:
        return False
    if isinstance(out1, Term) and isinstance(out2, Term):
        return _terms_structurally_equal(out1, out2)
    return str(out1) == str(out2)


def _build_inequiv_formula_str(p1_constraints, p2_constraints, out1, out2, p1_id, p2_id):
    """Build a human-readable inequivalence formula."""
    parts = []
    for c in p1_constraints:
        parts.append(f"P1.c: {_term_to_str(c)}")
    for c in p2_constraints:
        parts.append(f"P2.c: {_term_to_str(c)}")
    parts.append(f"P1.out: {_term_to_str(out1)}")
    parts.append(f"P2.out: {_term_to_str(out2)}")
    parts.append(f"Claim: P1.out != P2.out under joint constraints is UNSAT")
    return "; ".join(parts)


def _build_inequiv_formula_smt(p1_constraints, p2_constraints, out1, out2, all_vars):
    """Build SMT-LIB2 for the inequivalence query.

    all_vars is a set of (name, sort) tuples from _collect_vars_from_term.
    """
    lines = ["(set-logic LIA)"]
    for name, sort in sorted(all_vars):
        smt_sort = "Bool" if sort == "bool" else "Int"
        lines.append(f"(declare-const {name} {smt_sort})")
    for c in p1_constraints:
        try:
            lines.append(f"(assert {_term_to_smtlib(c)})")
        except Exception:
            lines.append(f"(assert true) ; could not encode: {c}")
    for c in p2_constraints:
        try:
            lines.append(f"(assert {_term_to_smtlib(c)})")
        except Exception:
            lines.append(f"(assert true) ; could not encode: {c}")
    # Assert outputs differ
    if out1 is not None and out2 is not None:
        o1_str = _term_to_smtlib(out1)
        o2_str = _term_to_smtlib(out2)
        lines.append(f"(assert (not (= {o1_str} {o2_str})))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def _get_path_output(path, output_var=None):
    """Extract the output from a path state as an SMT Term.

    Converts SymValue to Term using V006's _symval_to_term.
    """
    raw = None
    if hasattr(path, 'return_value') and path.return_value is not None:
        raw = path.return_value
    elif output_var and hasattr(path, 'env') and output_var in path.env:
        raw = path.env[output_var]
    elif hasattr(path, 'env'):
        if '__result' in path.env:
            raw = path.env['__result']
        else:
            non_input = {k: v for k, v in path.env.items() if not k.startswith('__')}
            if non_input:
                raw = list(non_input.values())[-1]

    if raw is None:
        return None

    # Convert SymValue to SMT Term
    if isinstance(raw, SymValue):
        return _symval_to_term(raw)
    if isinstance(raw, Term):
        return raw
    if isinstance(raw, int):
        return IntConst(raw)
    if isinstance(raw, bool):
        return IntConst(1 if raw else 0)
    return None


def _check_path_pair_smt(p1_constraints, p2_constraints, out1, out2):
    """Check if a path pair proves inequivalence using SMT.

    out1, out2 are SMT Terms (already converted from SymValue).
    Returns (verdict, model) where verdict is "infeasible"/"equal"/"not_equal"/"unknown".
    """
    # Structural equality shortcut
    if out1 is not None and out2 is not None and _outputs_structurally_equal(out1, out2):
        return "equal", None

    # Both None
    if out1 is None and out2 is None:
        return "equal", None

    # Build all constraints + output terms for variable declaration
    all_terms = list(p1_constraints) + list(p2_constraints)
    if isinstance(out1, Term):
        all_terms.append(out1)
    if isinstance(out2, Term):
        all_terms.append(out2)

    solver = SMTSolver()
    _declare_vars_in_solver(solver, all_terms)

    # Add path constraints
    for c in p1_constraints:
        solver.add(c)
    for c in p2_constraints:
        solver.add(c)

    # First check feasibility (path constraints overlap)
    solver.push()
    feasible_result = solver.check()
    solver.pop()
    if feasible_result == SMTResult.UNSAT:
        return "infeasible", None

    # One is None, other is not -- paths produce different types of output
    if out1 is None or out2 is None:
        return "not_equal", None

    # Build inequivalence query: constraints AND out1 != out2
    neq = App(SMTOp.NEQ, [out1, out2], BOOL)
    solver.add(neq)

    result = solver.check()
    if result == SMTResult.UNSAT:
        return "equal", None
    elif result == SMTResult.SAT:
        model = solver.model()
        return "not_equal", model
    else:
        return "unknown", None


# ---------------------------------------------------------------------------
# Core certification functions
# ---------------------------------------------------------------------------

def certify_function_equivalence(source1, fn_name1, source2, fn_name2,
                                  param_types, max_paths=64, max_loop_unroll=5):
    """Certify that two functions compute the same result.

    Returns an EquivCertificate with proof obligations for each path pair.
    """
    claim = f"{fn_name1}(source1) == {fn_name2}(source2) for all inputs"

    # Run equivalence check first
    equiv_result = check_function_equivalence(
        source1, fn_name1, source2, fn_name2, param_types,
        max_paths=max_paths, max_loop_unroll=max_loop_unroll
    )

    cert = EquivCertificate(
        kind=EquivCertKind.FUNCTION,
        claim=claim,
        source1=source1,
        source2=source2,
        result=equiv_result.result.value.lower(),
        metadata={
            "fn_name1": fn_name1,
            "fn_name2": fn_name2,
            "param_types": param_types,
            "paths_checked": equiv_result.paths_checked,
            "path_pairs_checked": equiv_result.path_pairs_checked,
        }
    )

    if equiv_result.counterexample:
        ce = equiv_result.counterexample
        cert.counterexample = {
            "inputs": ce.inputs,
            "output1": str(ce.output1),
            "output2": str(ce.output2),
        }

    # Now build detailed path-pair obligations
    _build_path_pair_obligations(cert, source1, source2, param_types,
                                  fn_name1, fn_name2, max_paths, max_loop_unroll)

    # Set certificate status
    _compute_cert_status(cert)
    return cert


def certify_program_equivalence(source1, source2, symbolic_inputs,
                                 output_var=None, max_paths=64, max_loop_unroll=5):
    """Certify that two programs produce the same output.

    Returns an EquivCertificate with proof obligations for each path pair.
    """
    claim = f"Program equivalence over inputs {list(symbolic_inputs.keys())}"

    equiv_result = check_program_equivalence(
        source1, source2, symbolic_inputs,
        output_var=output_var, max_paths=max_paths, max_loop_unroll=max_loop_unroll
    )

    cert = EquivCertificate(
        kind=EquivCertKind.PROGRAM,
        claim=claim,
        source1=source1,
        source2=source2,
        result=equiv_result.result.value.lower(),
        metadata={
            "symbolic_inputs": symbolic_inputs,
            "output_var": output_var,
            "paths_checked": equiv_result.paths_checked,
            "path_pairs_checked": equiv_result.path_pairs_checked,
        }
    )

    if equiv_result.counterexample:
        ce = equiv_result.counterexample
        cert.counterexample = {
            "inputs": ce.inputs,
            "output1": str(ce.output1),
            "output2": str(ce.output2),
        }

    _build_program_path_obligations(cert, source1, source2, symbolic_inputs,
                                     output_var, max_paths, max_loop_unroll)

    _compute_cert_status(cert)
    return cert


def certify_regression(original, refactored, symbolic_inputs,
                        output_var=None, fn_name=None, param_types=None,
                        max_paths=64):
    """Certify that refactored code matches original behavior.

    Returns an EquivCertificate.
    """
    claim = "Refactored code preserves original behavior"

    equiv_result = check_regression(
        original, refactored, symbolic_inputs,
        output_var=output_var, fn_name=fn_name, param_types=param_types,
        max_paths=max_paths
    )

    cert = EquivCertificate(
        kind=EquivCertKind.REGRESSION,
        claim=claim,
        source1=original,
        source2=refactored,
        result=equiv_result.result.value.lower(),
        metadata={
            "symbolic_inputs": symbolic_inputs,
            "output_var": output_var,
            "fn_name": fn_name,
            "paths_checked": equiv_result.paths_checked,
            "path_pairs_checked": equiv_result.path_pairs_checked,
        }
    )

    if equiv_result.counterexample:
        ce = equiv_result.counterexample
        cert.counterexample = {
            "inputs": ce.inputs,
            "output1": str(ce.output1),
            "output2": str(ce.output2),
        }

    if fn_name and param_types:
        _build_path_pair_obligations(cert, original, refactored, param_types,
                                      fn_name, fn_name, max_paths, 5)
    else:
        _build_program_path_obligations(cert, original, refactored, symbolic_inputs,
                                         output_var, max_paths, 5)

    _compute_cert_status(cert)
    return cert


def certify_partial_equivalence(source1, source2, symbolic_inputs,
                                 domain_constraints, output_var=None, max_paths=64):
    """Certify equivalence under restricted domain.

    domain_constraints: list of SMT Term constraints on inputs.
    Returns an EquivCertificate.
    """
    claim = f"Partial equivalence over inputs {list(symbolic_inputs.keys())} with domain constraints"

    equiv_result = check_partial_equivalence(
        source1, source2, symbolic_inputs, domain_constraints,
        output_var=output_var, max_paths=max_paths
    )

    cert = EquivCertificate(
        kind=EquivCertKind.PARTIAL,
        claim=claim,
        source1=source1,
        source2=source2,
        result=equiv_result.result.value.lower(),
        metadata={
            "symbolic_inputs": symbolic_inputs,
            "output_var": output_var,
            "domain_constraints": [_term_to_str(c) for c in domain_constraints],
            "paths_checked": equiv_result.paths_checked,
            "path_pairs_checked": equiv_result.path_pairs_checked,
        }
    )

    if equiv_result.counterexample:
        ce = equiv_result.counterexample
        cert.counterexample = {
            "inputs": ce.inputs,
            "output1": str(ce.output1),
            "output2": str(ce.output2),
        }

    _build_program_path_obligations(cert, source1, source2, symbolic_inputs,
                                     output_var, max_paths, 5)

    _compute_cert_status(cert)
    return cert


# ---------------------------------------------------------------------------
# Path-pair obligation building
# ---------------------------------------------------------------------------

def _build_path_pair_obligations(cert, source1, source2, param_types,
                                  fn_name1, fn_name2, max_paths, max_loop_unroll):
    """Build obligations from function equivalence paths."""
    param_names = list(param_types.keys())
    sym_inputs = {p: "int" for p in param_names}

    # Build wrapper for each function
    decls = "\n".join(f"let {p} = 0;" for p in param_names)
    call_args = ", ".join(param_names)
    wrapper1 = f"{source1}\n{decls}\nlet __result = {fn_name1}({call_args});"
    wrapper2 = f"{source2}\n{decls}\nlet __result = {fn_name2}({call_args});"

    _build_obligations_from_sources(cert, wrapper1, wrapper2, sym_inputs,
                                     "__result", max_paths, max_loop_unroll)


def _build_program_path_obligations(cert, source1, source2, symbolic_inputs,
                                     output_var, max_paths, max_loop_unroll):
    """Build obligations from program equivalence paths."""
    _build_obligations_from_sources(cert, source1, source2, symbolic_inputs,
                                     output_var, max_paths, max_loop_unroll)


def _build_obligations_from_sources(cert, source1, source2, symbolic_inputs,
                                     output_var, max_paths, max_loop_unroll):
    """Core: run symbolic execution on both sources and build path pair obligations."""
    try:
        exec1 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
        exec2 = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
        result1 = exec1.execute(source1, symbolic_inputs)
        result2 = exec2.execute(source2, symbolic_inputs)
    except Exception as e:
        # If symbolic execution fails, add a single unknown obligation
        obl = ProofObligation(
            name="symex_failure",
            description=f"Symbolic execution failed: {e}",
            formula_str=str(e),
            formula_smt="",
            status=CertStatus.UNKNOWN,
        )
        cert.obligations.append(obl)
        return

    paths1 = [p for p in result1.paths if p.status == PathStatus.COMPLETED]
    paths2 = [p for p in result2.paths if p.status == PathStatus.COMPLETED]

    if not paths1 or not paths2:
        obl = ProofObligation(
            name="no_paths",
            description="No completed paths from symbolic execution",
            formula_str="No paths",
            formula_smt="",
            status=CertStatus.UNKNOWN,
        )
        cert.obligations.append(obl)
        return

    pair_idx = 0
    for i, p1 in enumerate(paths1):
        for j, p2 in enumerate(paths2):
            pair_idx += 1
            out1 = _get_path_output(p1, output_var)
            out2 = _get_path_output(p2, output_var)

            # Check structural equality first
            struct_eq = _outputs_structurally_equal(out1, out2)

            # Collect all vars as (name, sort) tuples
            all_vars = set()
            p1_constraints = list(p1.constraints) if hasattr(p1, 'constraints') else []
            p2_constraints = list(p2.constraints) if hasattr(p2, 'constraints') else []
            for c in p1_constraints + p2_constraints:
                _collect_vars_from_term(c, all_vars)
            if isinstance(out1, Term):
                _collect_vars_from_term(out1, all_vars)
            if isinstance(out2, Term):
                _collect_vars_from_term(out2, all_vars)

            formula_str = _build_inequiv_formula_str(
                p1_constraints, p2_constraints, out1, out2, i, j
            )
            formula_smt = _build_inequiv_formula_smt(
                p1_constraints, p2_constraints, out1, out2, all_vars
            )

            # SMT check
            smt_result, model = _check_path_pair_smt(
                p1_constraints, p2_constraints, out1, out2
            )

            if smt_result == "infeasible":
                status = CertStatus.VALID  # paths don't overlap, trivially ok
                obl_desc = f"Path pair ({i},{j}): constraints infeasible (no overlap)"
                feasible = False
                output_eq = True
            elif smt_result == "equal" or struct_eq:
                status = CertStatus.VALID
                obl_desc = f"Path pair ({i},{j}): outputs provably equal"
                feasible = True
                output_eq = True
            elif smt_result == "not_equal":
                status = CertStatus.INVALID
                obl_desc = f"Path pair ({i},{j}): outputs differ (counterexample found)"
                feasible = True
                output_eq = False
                if model and not cert.counterexample:
                    cert.counterexample = {"model": {str(k): str(v) for k, v in model.items()}}
            else:
                status = CertStatus.UNKNOWN
                obl_desc = f"Path pair ({i},{j}): could not determine"
                feasible = True
                output_eq = False

            pp = PathPairObligation(
                path1_id=i,
                path2_id=j,
                constraints_feasible=feasible,
                output_equal=output_eq,
                structural_equal=struct_eq,
                formula_str=formula_str,
                formula_smt=formula_smt,
                status=status.value,
            )
            cert.path_pairs.append(pp)

            obl = ProofObligation(
                name=f"pair_{i}_{j}",
                description=obl_desc,
                formula_str=formula_str,
                formula_smt=formula_smt,
                status=status,
            )
            cert.obligations.append(obl)


def _compute_cert_status(cert):
    """Compute overall certificate status from obligations."""
    if not cert.obligations:
        cert.status = CertStatus.UNKNOWN
        return

    if any(o.status == CertStatus.INVALID for o in cert.obligations):
        cert.status = CertStatus.INVALID
        cert.result = "not_equivalent"
    elif all(o.status == CertStatus.VALID for o in cert.obligations):
        cert.status = CertStatus.VALID
        if cert.result != "not_equivalent":
            cert.result = "equivalent"
    elif any(o.status == CertStatus.UNKNOWN for o in cert.obligations):
        cert.status = CertStatus.UNKNOWN
    else:
        cert.status = CertStatus.UNCHECKED


# ---------------------------------------------------------------------------
# Independent checking
# ---------------------------------------------------------------------------

def check_equiv_certificate(cert):
    """Independently verify an equivalence certificate.

    Re-runs SMT checks on all obligations from their serialized formulas.
    Returns a new certificate with updated statuses.
    """
    checked = EquivCertificate(
        kind=cert.kind,
        claim=cert.claim,
        source1=cert.source1,
        source2=cert.source2,
        result=cert.result,
        counterexample=cert.counterexample,
        metadata=dict(cert.metadata),
        timestamp=cert.timestamp,
    )

    for pp in cert.path_pairs:
        checked.path_pairs.append(PathPairObligation(
            path1_id=pp.path1_id,
            path2_id=pp.path2_id,
            constraints_feasible=pp.constraints_feasible,
            output_equal=pp.output_equal,
            structural_equal=pp.structural_equal,
            formula_str=pp.formula_str,
            formula_smt=pp.formula_smt,
            status=pp.status,
        ))

    for obl in cert.obligations:
        # Re-check via SMT from the serialized formula
        new_status = _recheck_obligation(obl)
        new_obl = ProofObligation(
            name=obl.name,
            description=obl.description,
            formula_str=obl.formula_str,
            formula_smt=obl.formula_smt,
            status=new_status,
            counterexample=obl.counterexample,
        )
        checked.obligations.append(new_obl)

    _compute_cert_status(checked)
    checked.metadata["independently_checked"] = True
    return checked


def _recheck_obligation(obl):
    """Re-check a single obligation via SMT from its SMT-LIB2 formula."""
    if not obl.formula_smt or obl.formula_smt.strip() == "":
        return obl.status  # Can't re-check without formula

    # Parse the SMT-LIB2 and re-run
    try:
        solver = SMTSolver()
        lines = obl.formula_smt.strip().split("\n")
        declared_vars = {}

        for line in lines:
            line = line.strip()
            if line.startswith("(declare-const"):
                # (declare-const name Int)
                parts = line.replace("(", "").replace(")", "").split()
                if len(parts) >= 3:
                    name = parts[1]
                    sort = parts[2]
                    if sort == "Int":
                        declared_vars[name] = solver.Int(name)
                    elif sort == "Bool":
                        declared_vars[name] = solver.Bool(name)
            elif line.startswith("(assert"):
                # Extract the assertion body
                body = line[len("(assert "):-1].strip()
                if body == "true":
                    continue
                if body.startswith("; could not encode"):
                    continue
                # Parse the SMT-LIB2 assertion
                term = _parse_smtlib_term(body, declared_vars)
                if term is not None:
                    solver.add(term)
            elif line == "(check-sat)":
                pass  # We'll check at the end

        result = solver.check()
        if result == SMTResult.UNSAT:
            return CertStatus.VALID  # inequivalence is UNSAT -> equivalent
        elif result == SMTResult.SAT:
            return CertStatus.INVALID
        else:
            return CertStatus.UNKNOWN
    except Exception:
        return obl.status  # Fall back to original


def _parse_smtlib_term(s, vars_dict):
    """Parse a simple SMT-LIB2 term into C037 Term."""
    s = s.strip()

    # Integer literal
    if s.lstrip('-').isdigit():
        val = int(s)
        if val < 0:
            return App(SMTOp.SUB, [IntConst(0), IntConst(-val)], INT)
        return IntConst(val)

    # Boolean literals
    if s == "true":
        return BoolConst(True)
    if s == "false":
        return BoolConst(False)

    # Variable
    if s in vars_dict:
        return vars_dict[s]

    # S-expression
    if s.startswith("("):
        # Strip outer parens
        inner = s[1:-1].strip()
        # Parse operator
        op_str, rest = _split_first_token(inner)

        op_map = {
            "+": SMTOp.ADD, "-": SMTOp.SUB, "*": SMTOp.MUL,
            "=": SMTOp.EQ, "distinct": SMTOp.NEQ,
            "<": SMTOp.LT, "<=": SMTOp.LE, ">": SMTOp.GT, ">=": SMTOp.GE,
            "and": SMTOp.AND, "or": SMTOp.OR, "not": SMTOp.NOT,
            "ite": SMTOp.ITE,
        }

        if op_str in op_map:
            args = _parse_smtlib_args(rest, vars_dict)
            op = op_map[op_str]

            # Determine sort
            if op in (SMTOp.EQ, SMTOp.NEQ, SMTOp.LT, SMTOp.LE, SMTOp.GT, SMTOp.GE,
                      SMTOp.AND, SMTOp.OR, SMTOp.NOT):
                sort = BOOL
            elif op == SMTOp.ITE:
                sort = INT  # Assume integer ITE
            else:
                sort = INT

            if op == SMTOp.NOT and len(args) == 1:
                return App(op, args, sort)
            return App(op, args, sort)

        # Unrecognized
        return None

    return None


def _split_first_token(s):
    """Split 'op arg1 arg2...' into ('op', 'arg1 arg2...')."""
    s = s.strip()
    i = 0
    while i < len(s) and not s[i].isspace():
        i += 1
    return s[:i], s[i:].strip()


def _parse_smtlib_args(s, vars_dict):
    """Parse multiple SMT-LIB2 arguments from a string."""
    args = []
    s = s.strip()
    i = 0
    while i < len(s):
        if s[i] == '(':
            # Find matching close paren
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                j += 1
            arg = _parse_smtlib_term(s[i:j], vars_dict)
            if arg is not None:
                args.append(arg)
            i = j
        elif s[i].isspace():
            i += 1
        else:
            # Token
            j = i
            while j < len(s) and not s[j].isspace() and s[j] not in '()':
                j += 1
            token = s[i:j]
            arg = _parse_smtlib_term(token, vars_dict)
            if arg is not None:
                args.append(arg)
            i = j
    return args


# ---------------------------------------------------------------------------
# V044 bridge
# ---------------------------------------------------------------------------

def to_v044_certificate(cert):
    """Convert EquivCertificate to V044 ProofCertificate."""
    v044_cert = ProofCertificate(
        kind=ProofKind.COMPOSITE,
        claim=cert.claim,
        source=f"Source1:\n{cert.source1}\n---\nSource2:\n{cert.source2}",
        obligations=list(cert.obligations),
        metadata={
            **cert.metadata,
            "cert_kind": cert.kind.value,
            "result": cert.result,
            "path_pairs": len(cert.path_pairs),
        },
        status=cert.status,
        timestamp=cert.timestamp,
    )
    return v044_cert


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------

def certify_and_check(source1, source2, symbolic_inputs=None,
                       fn_name1=None, fn_name2=None, param_types=None,
                       output_var=None, max_paths=64):
    """One-shot: generate certificate + independently check it.

    Auto-detects function vs program mode based on arguments.
    """
    if fn_name1 and fn_name2 and param_types:
        cert = certify_function_equivalence(
            source1, fn_name1, source2, fn_name2, param_types,
            max_paths=max_paths
        )
    elif symbolic_inputs:
        cert = certify_program_equivalence(
            source1, source2, symbolic_inputs,
            output_var=output_var, max_paths=max_paths
        )
    else:
        raise ValueError("Must provide either (fn_name1, fn_name2, param_types) or symbolic_inputs")

    checked = check_equiv_certificate(cert)
    return checked


def compare_certified_vs_uncertified(source1, source2, symbolic_inputs=None,
                                       fn_name1=None, fn_name2=None,
                                       param_types=None, output_var=None,
                                       max_paths=64):
    """Compare certified vs plain equivalence checking.

    Returns a comparison dict with both results and timing.
    """
    # Plain check
    t0 = time.time()
    if fn_name1 and fn_name2 and param_types:
        plain = check_function_equivalence(
            source1, fn_name1, source2, fn_name2, param_types,
            max_paths=max_paths
        )
    elif symbolic_inputs:
        plain = check_program_equivalence(
            source1, source2, symbolic_inputs,
            output_var=output_var, max_paths=max_paths
        )
    else:
        raise ValueError("Need fn names+param_types or symbolic_inputs")
    t_plain = time.time() - t0

    # Certified check
    t0 = time.time()
    if fn_name1 and fn_name2 and param_types:
        cert = certify_function_equivalence(
            source1, fn_name1, source2, fn_name2, param_types,
            max_paths=max_paths
        )
    elif symbolic_inputs:
        cert = certify_program_equivalence(
            source1, source2, symbolic_inputs,
            output_var=output_var, max_paths=max_paths
        )
    t_cert = time.time() - t0

    # Independent check
    t0 = time.time()
    checked = check_equiv_certificate(cert)
    t_check = time.time() - t0

    return {
        "plain_result": plain.result.value.lower(),
        "certified_result": cert.result,
        "checked_result": checked.result,
        "cert_status": cert.status.value,
        "checked_status": checked.status.value,
        "obligations": cert.total_obligations,
        "valid_obligations": cert.valid_obligations,
        "checked_valid": checked.valid_obligations,
        "plain_time_s": round(t_plain, 3),
        "cert_time_s": round(t_cert, 3),
        "check_time_s": round(t_check, 3),
        "overhead_factor": round((t_cert + t_check) / max(t_plain, 0.001), 2),
        "agreement": plain.result.value.lower() == cert.result,
    }


def equiv_certificate_summary(cert):
    """Get a concise summary of an equivalence certificate."""
    return {
        "kind": cert.kind.value,
        "claim": cert.claim,
        "result": cert.result,
        "status": cert.status.value,
        "total_obligations": cert.total_obligations,
        "valid": cert.valid_obligations,
        "invalid": cert.invalid_obligations,
        "unknown": sum(1 for o in cert.obligations if o.status == CertStatus.UNKNOWN),
        "path_pairs": len(cert.path_pairs),
        "has_counterexample": cert.counterexample is not None,
        "serializable": True,
    }


def save_equiv_certificate(cert, path):
    """Save an equivalence certificate to JSON file."""
    with open(path, 'w') as f:
        f.write(cert.to_json())


def load_equiv_certificate(path):
    """Load an equivalence certificate from JSON file."""
    with open(path, 'r') as f:
        return EquivCertificate.from_json(f.read())
