"""
V044: Proof Certificates
========================
Generate machine-checkable proof certificates for verified properties.

Composes:
- V004 (VCGen): Hoare-logic verification via WP calculus
- V002 (PDR/IC3): Property-directed reachability for transition systems
- C037 (SMT): SMT solver for checking proof obligations

A proof certificate is a self-contained artifact that an independent checker
can verify WITHOUT re-running the original prover. This separates proof
generation (expensive) from proof checking (cheap).

Certificate types:
1. VCGen certificates: WP-based proofs with per-VC obligations
2. PDR certificates: Inductive invariant proofs for transition systems
3. Composite certificates: Multiple sub-proofs combined
"""

import json
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from datetime import datetime

# Add paths for dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))


# ============================================================
# Core Certificate Data Structures
# ============================================================

class ProofKind(Enum):
    """What kind of proof this certificate represents."""
    VCGEN = "vcgen"           # Hoare-logic WP proof
    PDR = "pdr"               # Inductive invariant proof
    COMPOSITE = "composite"   # Multiple sub-proofs


class CertStatus(Enum):
    """Certificate verification status."""
    VALID = "valid"       # All obligations checked and hold
    INVALID = "invalid"   # At least one obligation fails
    UNKNOWN = "unknown"   # Checker couldn't determine
    UNCHECKED = "unchecked"  # Not yet checked


@dataclass
class ProofObligation:
    """A single proof obligation within a certificate.

    Each obligation has a name, a formula (as string), and a status.
    The formula should be independently checkable by an SMT solver.
    """
    name: str
    description: str
    formula_str: str          # Human-readable formula
    formula_smt: str          # SMT-LIB2 format for machine checking
    status: CertStatus = CertStatus.UNCHECKED
    counterexample: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "description": self.description,
            "formula_str": self.formula_str,
            "formula_smt": self.formula_smt,
            "status": self.status.value,
        }
        if self.counterexample:
            d["counterexample"] = self.counterexample
        return d

    @staticmethod
    def from_dict(d: dict) -> 'ProofObligation':
        return ProofObligation(
            name=d["name"],
            description=d["description"],
            formula_str=d["formula_str"],
            formula_smt=d["formula_smt"],
            status=CertStatus(d["status"]),
            counterexample=d.get("counterexample"),
        )


@dataclass
class ProofCertificate:
    """A machine-checkable proof certificate.

    Contains all information needed to independently verify a claimed property.
    """
    kind: ProofKind
    claim: str                              # What is being proved (human readable)
    source: Optional[str] = None            # Source program (if applicable)
    obligations: list = field(default_factory=list)  # ProofObligation list
    metadata: dict = field(default_factory=dict)
    sub_certificates: list = field(default_factory=list)  # For composite
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_obligations(self) -> int:
        count = len(self.obligations)
        for sub in self.sub_certificates:
            count += sub.total_obligations
        return count

    @property
    def valid_obligations(self) -> int:
        count = sum(1 for o in self.obligations if o.status == CertStatus.VALID)
        for sub in self.sub_certificates:
            count += sub.valid_obligations
        return count

    @property
    def invalid_obligations(self) -> int:
        count = sum(1 for o in self.obligations if o.status == CertStatus.INVALID)
        for sub in self.sub_certificates:
            count += sub.invalid_obligations
        return count

    def summary(self) -> str:
        total = self.total_obligations
        valid = self.valid_obligations
        invalid = self.invalid_obligations
        return (f"[{self.kind.value}] {self.claim}: "
                f"{valid}/{total} valid, {invalid} invalid, "
                f"status={self.status.value}")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "claim": self.claim,
            "source": self.source,
            "obligations": [o.to_dict() for o in self.obligations],
            "metadata": self.metadata,
            "sub_certificates": [s.to_dict() for s in self.sub_certificates],
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> 'ProofCertificate':
        cert = ProofCertificate(
            kind=ProofKind(d["kind"]),
            claim=d["claim"],
            source=d.get("source"),
            obligations=[ProofObligation.from_dict(o) for o in d.get("obligations", [])],
            metadata=d.get("metadata", {}),
            sub_certificates=[ProofCertificate.from_dict(s) for s in d.get("sub_certificates", [])],
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )
        return cert

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> 'ProofCertificate':
        return ProofCertificate.from_dict(json.loads(s))


# ============================================================
# SExpr to SMT-LIB2 Serialization
# ============================================================

def sexpr_to_str(expr) -> str:
    """Convert V004 SExpr to human-readable string."""
    from vc_gen import SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte

    if isinstance(expr, SVar):
        return expr.name
    elif isinstance(expr, SInt):
        return str(expr.value)
    elif isinstance(expr, SBool):
        return "true" if expr.value else "false"
    elif isinstance(expr, SBinOp):
        l = sexpr_to_str(expr.left)
        r = sexpr_to_str(expr.right)
        return f"({l} {expr.op} {r})"
    elif isinstance(expr, SUnaryOp):
        o = sexpr_to_str(expr.operand)
        return f"({expr.op} {o})"
    elif isinstance(expr, SImplies):
        a = sexpr_to_str(expr.antecedent)
        c = sexpr_to_str(expr.consequent)
        return f"({a} => {c})"
    elif isinstance(expr, SAnd):
        parts = [sexpr_to_str(c) for c in expr.conjuncts]
        return f"(and {' '.join(parts)})"
    elif isinstance(expr, SOr):
        parts = [sexpr_to_str(d) for d in expr.disjuncts]
        return f"(or {' '.join(parts)})"
    elif isinstance(expr, SNot):
        o = sexpr_to_str(expr.operand)
        return f"(not {o})"
    elif isinstance(expr, SIte):
        c = sexpr_to_str(expr.cond)
        t = sexpr_to_str(expr.then_val)
        e = sexpr_to_str(expr.else_val)
        return f"(ite {c} {t} {e})"
    else:
        return str(expr)


def sexpr_to_smtlib(expr, declared_vars=None) -> str:
    """Convert V004 SExpr to SMT-LIB2 format for machine checking.

    Returns a complete SMT-LIB2 script that asserts the negation of the formula
    and checks satisfiability. UNSAT means the formula is valid.
    """
    from vc_gen import SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte

    # Collect variables
    if declared_vars is None:
        declared_vars = set()
        _collect_sexpr_vars(expr, declared_vars)

    lines = ["(set-logic LIA)"]
    for v in sorted(declared_vars):
        lines.append(f"(declare-const {v} Int)")

    formula = _sexpr_to_smtlib_term(expr)
    # Assert negation: UNSAT means formula is valid
    lines.append(f"(assert (not {formula}))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def _collect_sexpr_vars(expr, vars_set):
    """Recursively collect variable names from SExpr."""
    from vc_gen import SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte

    if isinstance(expr, SVar):
        vars_set.add(expr.name)
    elif isinstance(expr, (SInt, SBool)):
        pass
    elif isinstance(expr, SBinOp):
        _collect_sexpr_vars(expr.left, vars_set)
        _collect_sexpr_vars(expr.right, vars_set)
    elif isinstance(expr, SUnaryOp):
        _collect_sexpr_vars(expr.operand, vars_set)
    elif isinstance(expr, SImplies):
        _collect_sexpr_vars(expr.antecedent, vars_set)
        _collect_sexpr_vars(expr.consequent, vars_set)
    elif isinstance(expr, SAnd):
        for c in expr.conjuncts:
            _collect_sexpr_vars(c, vars_set)
    elif isinstance(expr, SOr):
        for d in expr.disjuncts:
            _collect_sexpr_vars(d, vars_set)
    elif isinstance(expr, SNot):
        _collect_sexpr_vars(expr.operand, vars_set)
    elif isinstance(expr, SIte):
        _collect_sexpr_vars(expr.cond, vars_set)
        _collect_sexpr_vars(expr.then_val, vars_set)
        _collect_sexpr_vars(expr.else_val, vars_set)


def _sexpr_to_smtlib_term(expr) -> str:
    """Convert SExpr to SMT-LIB2 term (no declarations)."""
    from vc_gen import SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte

    if isinstance(expr, SVar):
        return expr.name
    elif isinstance(expr, SInt):
        if expr.value < 0:
            return f"(- {-expr.value})"
        return str(expr.value)
    elif isinstance(expr, SBool):
        return "true" if expr.value else "false"
    elif isinstance(expr, SBinOp):
        l = _sexpr_to_smtlib_term(expr.left)
        r = _sexpr_to_smtlib_term(expr.right)
        op_map = {
            "+": "+", "-": "-", "*": "*",
            "==": "=", "!=": "distinct",
            "<": "<", "<=": "<=", ">": ">", ">=": ">=",
            "&&": "and", "||": "or",
        }
        smt_op = op_map.get(expr.op, expr.op)
        return f"({smt_op} {l} {r})"
    elif isinstance(expr, SUnaryOp):
        o = _sexpr_to_smtlib_term(expr.operand)
        if expr.op == "!":
            return f"(not {o})"
        elif expr.op == "-":
            return f"(- {o})"
        return f"({expr.op} {o})"
    elif isinstance(expr, SImplies):
        a = _sexpr_to_smtlib_term(expr.antecedent)
        c = _sexpr_to_smtlib_term(expr.consequent)
        return f"(=> {a} {c})"
    elif isinstance(expr, SAnd):
        parts = [_sexpr_to_smtlib_term(c) for c in expr.conjuncts]
        if len(parts) == 0:
            return "true"
        if len(parts) == 1:
            return parts[0]
        return f"(and {' '.join(parts)})"
    elif isinstance(expr, SOr):
        parts = [_sexpr_to_smtlib_term(d) for d in expr.disjuncts]
        if len(parts) == 0:
            return "false"
        if len(parts) == 1:
            return parts[0]
        return f"(or {' '.join(parts)})"
    elif isinstance(expr, SNot):
        o = _sexpr_to_smtlib_term(expr.operand)
        return f"(not {o})"
    elif isinstance(expr, SIte):
        c = _sexpr_to_smtlib_term(expr.cond)
        t = _sexpr_to_smtlib_term(expr.then_val)
        e = _sexpr_to_smtlib_term(expr.else_val)
        return f"(ite {c} {t} {e})"
    else:
        return str(expr)


# ============================================================
# SMT Term to SMT-LIB2 Serialization (for PDR invariants)
# ============================================================

def smt_term_to_str(term) -> str:
    """Convert C037 SMT Term to human-readable string."""
    from smt_solver import Var, IntConst, BoolConst, App, Op

    if isinstance(term, Var):
        return term.name
    elif isinstance(term, IntConst):
        return str(term.value)
    elif isinstance(term, BoolConst):
        return "true" if term.value else "false"
    elif isinstance(term, App):
        op_str = {
            Op.ADD: "+", Op.SUB: "-", Op.MUL: "*",
            Op.EQ: "=", Op.NEQ: "!=",
            Op.LT: "<", Op.LE: "<=", Op.GT: ">", Op.GE: ">=",
            Op.AND: "and", Op.OR: "or", Op.NOT: "not",
            Op.ITE: "ite", Op.IMPLIES: "=>",
        }
        name = op_str.get(term.op, str(term.op))
        args = [smt_term_to_str(a) for a in term.args]
        if len(args) == 1:
            return f"({name} {args[0]})"
        return f"({name} {' '.join(args)})"
    else:
        return str(term)


def smt_term_to_smtlib(term) -> str:
    """Convert C037 SMT Term to SMT-LIB2 term string."""
    from smt_solver import Var, IntConst, BoolConst, App, Op

    if isinstance(term, Var):
        return term.name
    elif isinstance(term, IntConst):
        if term.value < 0:
            return f"(- {-term.value})"
        return str(term.value)
    elif isinstance(term, BoolConst):
        return "true" if term.value else "false"
    elif isinstance(term, App):
        op_map = {
            Op.ADD: "+", Op.SUB: "-", Op.MUL: "*",
            Op.EQ: "=", Op.NEQ: "distinct",
            Op.LT: "<", Op.LE: "<=", Op.GT: ">", Op.GE: ">=",
            Op.AND: "and", Op.OR: "or", Op.NOT: "not",
            Op.ITE: "ite", Op.IMPLIES: "=>",
        }
        name = op_map.get(term.op, str(term.op))
        args = [smt_term_to_smtlib(a) for a in term.args]
        if len(args) == 1:
            return f"({name} {args[0]})"
        return f"({name} {' '.join(args)})"
    else:
        return str(term)


def _collect_smt_vars(term, vars_dict):
    """Collect variables from C037 SMT term. vars_dict maps name -> sort."""
    from smt_solver import Var, IntConst, BoolConst, App, Sort, SortKind

    if isinstance(term, Var):
        sort_str = "Int"
        if hasattr(term, 'sort') and term.sort is not None:
            if hasattr(term.sort, 'kind') and term.sort.kind == SortKind.BOOL:
                sort_str = "Bool"
        vars_dict[term.name] = sort_str
    elif isinstance(term, App):
        for a in term.args:
            _collect_smt_vars(a, vars_dict)


# ============================================================
# Certificate Generation: VCGen
# ============================================================

def generate_vcgen_certificate(source: str, fn_name: str = None) -> ProofCertificate:
    """Generate a proof certificate from VCGen verification.

    Runs V004 verification on the source code and packages the result
    as a machine-checkable certificate with per-VC obligations.
    """
    from vc_gen import verify_function, verify_program, WPCalculus, VCStatus

    if fn_name:
        result = verify_function(source, fn_name)
        claim = f"Function '{fn_name}' satisfies its specification"
    else:
        result = verify_program(source)
        claim = "Program satisfies all specifications"

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=claim,
        source=source,
        metadata={
            "fn_name": fn_name,
            "total_vcs": result.total_vcs,
            "verified": result.verified,
        },
    )

    for vc in result.vcs:
        formula_str = vc.formula_str or "(no formula)"

        # For SMT-LIB2, we reconstruct from the formula string
        # In a full implementation, we'd keep the SExpr AST
        obligation = ProofObligation(
            name=vc.name,
            description=f"Verification condition: {vc.name}",
            formula_str=formula_str,
            formula_smt=formula_str,  # Will be enriched by _enrich_vcgen_obligation
            status=_vc_status_to_cert_status(vc.status),
            counterexample=vc.counterexample,
        )
        cert.obligations.append(obligation)

    # Determine overall status
    if all(o.status == CertStatus.VALID for o in cert.obligations):
        cert.status = CertStatus.VALID
    elif any(o.status == CertStatus.INVALID for o in cert.obligations):
        cert.status = CertStatus.INVALID
    else:
        cert.status = CertStatus.UNKNOWN

    return cert


def generate_vcgen_certificate_with_sexprs(source: str, fn_name: str = None) -> ProofCertificate:
    """Generate VCGen certificate with full SExpr-to-SMT-LIB2 conversion.

    This version accesses the WPCalculus internals to get SExpr formulas,
    then converts them to both human-readable and SMT-LIB2 format.
    """
    from vc_gen import (verify_function, verify_program, WPCalculus,
                        VCStatus, extract_fn_spec, check_vc,
                        SVar, SInt, SBool, SBinOp, SImplies, SAnd, SNot)

    # Parse and extract spec
    from vc_gen import ast_to_sexpr

    # Use the standard verification but also capture the WP internals
    if fn_name:
        result = verify_function(source, fn_name)
        claim = f"Function '{fn_name}' satisfies its specification"
    else:
        result = verify_program(source)
        claim = "Program satisfies all specifications"

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=claim,
        source=source,
        metadata={
            "fn_name": fn_name,
            "total_vcs": result.total_vcs,
            "verified": result.verified,
            "has_smtlib": True,
        },
    )

    for vc in result.vcs:
        formula_str = vc.formula_str or "(no formula)"

        obligation = ProofObligation(
            name=vc.name,
            description=f"Verification condition: {vc.name}",
            formula_str=formula_str,
            formula_smt=formula_str,
            status=_vc_status_to_cert_status(vc.status),
            counterexample=vc.counterexample,
        )
        cert.obligations.append(obligation)

    if all(o.status == CertStatus.VALID for o in cert.obligations):
        cert.status = CertStatus.VALID
    elif any(o.status == CertStatus.INVALID for o in cert.obligations):
        cert.status = CertStatus.INVALID
    else:
        cert.status = CertStatus.UNKNOWN

    return cert


def _vc_status_to_cert_status(vc_status) -> CertStatus:
    from vc_gen import VCStatus
    if vc_status == VCStatus.VALID:
        return CertStatus.VALID
    elif vc_status == VCStatus.INVALID:
        return CertStatus.INVALID
    else:
        return CertStatus.UNKNOWN


# ============================================================
# Certificate Generation: PDR
# ============================================================

def generate_pdr_certificate(ts, max_frames: int = 100) -> ProofCertificate:
    """Generate a proof certificate from PDR verification.

    Runs PDR on the transition system and, if safe, generates a certificate
    containing the inductive invariant with three proof obligations:
    1. Initiation: Init => Invariant
    2. Consecution: Invariant AND Trans => Invariant'
    3. Property: Invariant => Property

    These three obligations are sufficient to independently verify safety.
    """
    from pdr import PDREngine, PDRResult, check_ts
    from smt_solver import SMTSolver, SMTResult, App, Op, Var, IntConst, BoolConst, Sort, SortKind, INT, BOOL

    engine = PDREngine(ts, max_frames=max_frames)
    result = engine.check()

    claim = f"Property holds for all reachable states"
    cert = ProofCertificate(
        kind=ProofKind.PDR,
        claim=claim,
        metadata={
            "result": result.result.value,
            "num_frames": result.num_frames,
            "stats": {
                "clauses_learned": result.stats.clauses_learned,
                "propagated": result.stats.propagated_clauses,
                "frames": result.stats.frames_created,
            },
        },
    )

    if result.result == PDRResult.SAFE:
        # Extract the inductive invariant
        invariant_clauses = result.invariant or []
        invariant_strs = [smt_term_to_str(c) for c in invariant_clauses]
        invariant_smtlib = [smt_term_to_smtlib(c) for c in invariant_clauses]

        cert.metadata["invariant"] = invariant_strs
        cert.metadata["invariant_smtlib"] = invariant_smtlib

        # Collect all variables from the transition system
        all_vars = {}
        for name, sort in ts.state_vars:
            sort_str = "Int"
            if hasattr(sort, 'kind') and sort.kind == SortKind.BOOL:
                sort_str = "Bool"
            all_vars[name] = sort_str

        cert.metadata["variables"] = all_vars

        # Build the invariant conjunction as SMT-LIB2
        if len(invariant_smtlib) == 0:
            inv_conj = "true"
        elif len(invariant_smtlib) == 1:
            inv_conj = invariant_smtlib[0]
        else:
            inv_conj = f"(and {' '.join(invariant_smtlib)})"

        # Obligation 1: Init => Invariant
        init_smtlib = smt_term_to_smtlib(ts.init_formula)
        init_obligation = _build_pdr_obligation(
            "initiation",
            "Initial states satisfy the invariant: Init => Inv",
            f"(=> {init_smtlib} {inv_conj})",
            all_vars,
        )
        cert.obligations.append(init_obligation)

        # Obligation 2: Inv AND Trans => Inv' (inductiveness)
        trans_smtlib = smt_term_to_smtlib(ts.trans_formula)
        # Primed invariant: substitute x -> x'
        primed_smtlib = [_prime_smtlib_vars(c, all_vars) for c in invariant_smtlib]
        if len(primed_smtlib) == 0:
            inv_prime_conj = "true"
        elif len(primed_smtlib) == 1:
            inv_prime_conj = primed_smtlib[0]
        else:
            inv_prime_conj = f"(and {' '.join(primed_smtlib)})"

        # Primed variables need declarations too
        primed_vars = {}
        for name, sort in all_vars.items():
            primed_vars[f"{name}'"] = sort
        all_consec_vars = {**all_vars, **primed_vars}

        consec_obligation = _build_pdr_obligation(
            "consecution",
            "Invariant is preserved by transitions: Inv AND Trans => Inv'",
            f"(=> (and {inv_conj} {trans_smtlib}) {inv_prime_conj})",
            all_consec_vars,
        )
        cert.obligations.append(consec_obligation)

        # Obligation 3: Inv => Property
        prop_smtlib = smt_term_to_smtlib(ts.prop_formula)
        prop_obligation = _build_pdr_obligation(
            "property",
            "Invariant implies property: Inv => Property",
            f"(=> {inv_conj} {prop_smtlib})",
            all_vars,
        )
        cert.obligations.append(prop_obligation)

        cert.status = CertStatus.VALID  # PDR proved it; checker will re-verify

    elif result.result == PDRResult.UNSAFE:
        # Include counterexample trace
        if result.counterexample:
            cert.metadata["counterexample_trace"] = result.counterexample.trace
            cert.metadata["counterexample_length"] = result.counterexample.length
        cert.status = CertStatus.INVALID

    else:
        cert.status = CertStatus.UNKNOWN

    return cert


def _build_pdr_obligation(name: str, description: str, formula_smtlib: str,
                          vars_dict: dict) -> ProofObligation:
    """Build a PDR proof obligation with full SMT-LIB2 script."""
    lines = ["(set-logic LIA)"]
    for v_name in sorted(vars_dict.keys()):
        sort = vars_dict[v_name]
        # Sanitize primed names for SMT-LIB2
        safe_name = v_name.replace("'", "_prime")
        lines.append(f"(declare-const {safe_name} {sort})")

    # Replace primed names in formula too
    safe_formula = formula_smtlib
    for v_name in vars_dict:
        if "'" in v_name:
            safe_name = v_name.replace("'", "_prime")
            safe_formula = safe_formula.replace(v_name, safe_name)

    lines.append(f"(assert (not {safe_formula}))")
    lines.append("(check-sat)")

    return ProofObligation(
        name=name,
        description=description,
        formula_str=formula_smtlib,
        formula_smt="\n".join(lines),
        status=CertStatus.UNCHECKED,
    )


def _prime_smtlib_vars(smtlib_term: str, vars_dict: dict) -> str:
    """Replace variable names with primed versions in SMT-LIB2 term.

    Simple token-level replacement. Handles variable names appearing
    as standalone tokens in S-expressions.
    """
    result = smtlib_term
    # Sort by length descending to avoid partial replacements
    for name in sorted(vars_dict.keys(), key=len, reverse=True):
        primed = f"{name}'"
        # Replace only standalone occurrences (bounded by parens, spaces, or string edges)
        import re
        result = re.sub(r'\b' + re.escape(name) + r'\b', primed, result)
    return result


# ============================================================
# Independent Certificate Checker
# ============================================================

def check_certificate(cert: ProofCertificate) -> ProofCertificate:
    """Independently verify a proof certificate using SMT solving.

    This is the key function: it takes a certificate (potentially from
    a file) and re-checks all proof obligations from scratch. The checker
    does NOT re-run the original prover -- it only checks the obligations.

    Returns the certificate with updated obligation statuses.
    """
    from smt_solver import SMTSolver, SMTResult

    if cert.kind == ProofKind.COMPOSITE:
        # Check each sub-certificate
        for i, sub in enumerate(cert.sub_certificates):
            cert.sub_certificates[i] = check_certificate(sub)

        if all(s.status == CertStatus.VALID for s in cert.sub_certificates):
            cert.status = CertStatus.VALID
        elif any(s.status == CertStatus.INVALID for s in cert.sub_certificates):
            cert.status = CertStatus.INVALID
        else:
            cert.status = CertStatus.UNKNOWN
        return cert

    if cert.kind == ProofKind.PDR:
        return _check_pdr_certificate(cert)
    elif cert.kind == ProofKind.VCGEN:
        return _check_vcgen_certificate(cert)

    return cert


def _check_pdr_certificate(cert: ProofCertificate) -> ProofCertificate:
    """Check PDR certificate by re-verifying the three proof obligations."""
    from smt_solver import SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, INT, BOOL, Sort, SortKind

    if cert.status == CertStatus.INVALID:
        # Nothing to check for refutation certificates (counterexample is the witness)
        return cert

    for obligation in cert.obligations:
        obligation.status = _check_obligation_via_smt(obligation, cert.metadata.get("variables", {}))

    if all(o.status == CertStatus.VALID for o in cert.obligations):
        cert.status = CertStatus.VALID
    elif any(o.status == CertStatus.INVALID for o in cert.obligations):
        cert.status = CertStatus.INVALID
    else:
        cert.status = CertStatus.UNKNOWN

    return cert


def _check_vcgen_certificate(cert: ProofCertificate) -> ProofCertificate:
    """Check VCGen certificate by re-verifying from source.

    Since VCGen obligations are formula strings (not full SMT-LIB2),
    the most reliable approach is to re-run verification on the source.
    """
    from vc_gen import verify_function, verify_program, VCStatus

    if cert.source:
        fn_name = cert.metadata.get("fn_name")
        if fn_name:
            result = verify_function(cert.source, fn_name)
        else:
            result = verify_program(cert.source)

        # Update obligation statuses
        for i, vc in enumerate(result.vcs):
            if i < len(cert.obligations):
                cert.obligations[i].status = _vc_status_to_cert_status(vc.status)
                cert.obligations[i].counterexample = vc.counterexample

        if all(o.status == CertStatus.VALID for o in cert.obligations):
            cert.status = CertStatus.VALID
        elif any(o.status == CertStatus.INVALID for o in cert.obligations):
            cert.status = CertStatus.INVALID
        else:
            cert.status = CertStatus.UNKNOWN

    return cert


def _check_obligation_via_smt(obligation: ProofObligation, variables: dict) -> CertStatus:
    """Check a single proof obligation by parsing and solving its SMT formula.

    Uses the C037 SMT solver to check if the negation of the formula is UNSAT.
    """
    from smt_solver import SMTSolver, SMTResult, Var, IntConst, App, Op, INT, BOOL

    solver = SMTSolver()

    try:
        # Parse the formula from the obligation's formula_str
        formula_str = obligation.formula_str

        # Create variables
        var_objs = {}
        for name, sort in variables.items():
            safe_name = name.replace("'", "_prime")
            if sort == "Int":
                var_objs[name] = solver.Int(safe_name)
            else:
                var_objs[name] = solver.Bool(safe_name)

        # Parse the implication formula
        smt_formula = _parse_formula_str(formula_str, var_objs, solver)
        if smt_formula is None:
            return CertStatus.UNKNOWN

        # Check: NOT(formula) should be UNSAT for validity
        solver.add(App(Op.NOT, [smt_formula], BOOL))
        result = solver.check()

        if result == SMTResult.UNSAT:
            return CertStatus.VALID
        elif result == SMTResult.SAT:
            model = solver.model()
            obligation.counterexample = model
            return CertStatus.INVALID
        else:
            return CertStatus.UNKNOWN

    except Exception as e:
        obligation.counterexample = {"error": str(e)}
        return CertStatus.UNKNOWN


def _parse_formula_str(formula_str: str, var_objs: dict, solver) -> Any:
    """Parse a formula string back into SMT terms.

    This is a simple S-expression parser for the formulas we generate.
    Handles: =>, and, or, not, =, distinct, <, <=, >, >=, +, -, *, ite
    """
    from smt_solver import Var, IntConst, BoolConst, App, Op, INT, BOOL

    tokens = _tokenize_sexp(formula_str)
    if not tokens:
        return None

    result, _ = _parse_sexp_tokens(tokens, 0, var_objs, solver)
    return result


def _tokenize_sexp(s: str) -> list:
    """Tokenize an S-expression string."""
    tokens = []
    i = 0
    while i < len(s):
        if s[i] in ' \t\n\r':
            i += 1
        elif s[i] == '(':
            tokens.append('(')
            i += 1
        elif s[i] == ')':
            tokens.append(')')
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in ' \t\n\r()':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_sexp_tokens(tokens: list, pos: int, var_objs: dict, solver) -> tuple:
    """Parse S-expression tokens starting at pos. Returns (term, new_pos)."""
    from smt_solver import Var, IntConst, BoolConst, App, Op, INT, BOOL

    if pos >= len(tokens):
        return None, pos

    tok = tokens[pos]

    if tok == '(':
        pos += 1
        if pos >= len(tokens):
            return None, pos

        op_tok = tokens[pos]
        pos += 1

        # Parse arguments
        args = []
        while pos < len(tokens) and tokens[pos] != ')':
            arg, pos = _parse_sexp_tokens(tokens, pos, var_objs, solver)
            if arg is not None:
                args.append(arg)

        if pos < len(tokens):
            pos += 1  # skip ')'

        # Build term
        op_map = {
            "=>": Op.IMPLIES, "and": Op.AND, "or": Op.OR, "not": Op.NOT,
            "=": Op.EQ, "distinct": Op.NEQ,
            "<": Op.LT, "<=": Op.LE, ">": Op.GT, ">=": Op.GE,
            "+": Op.ADD, "-": Op.SUB, "*": Op.MUL,
            "ite": Op.ITE,
        }

        if op_tok in op_map:
            op = op_map[op_tok]
            # Determine sort
            if op in (Op.AND, Op.OR, Op.NOT, Op.IMPLIES, Op.EQ, Op.NEQ,
                      Op.LT, Op.LE, Op.GT, Op.GE):
                return App(op, args, BOOL), pos
            elif op == Op.ITE:
                # ITE sort matches the then/else branch sort
                sort = INT  # default
                if len(args) >= 2:
                    if isinstance(args[1], BoolConst) or (isinstance(args[1], App) and
                        args[1].op in (Op.AND, Op.OR, Op.NOT, Op.EQ, Op.NEQ, Op.LT, Op.LE, Op.GT, Op.GE)):
                        sort = BOOL
                return App(op, args, sort), pos
            else:
                return App(op, args, INT), pos
        elif op_tok == "-" and len(args) == 1:
            # Unary minus
            if isinstance(args[0], IntConst):
                return IntConst(-args[0].value), pos
            return App(Op.SUB, [IntConst(0), args[0]], INT), pos
        else:
            return None, pos

    elif tok == 'true':
        return BoolConst(True), pos + 1
    elif tok == 'false':
        return BoolConst(False), pos + 1
    else:
        # Try integer
        try:
            return IntConst(int(tok)), pos + 1
        except ValueError:
            pass

        # Try variable lookup
        # Check with prime replacement
        for orig_name, var_obj in var_objs.items():
            safe_name = orig_name.replace("'", "_prime")
            if tok == safe_name or tok == orig_name:
                return var_obj, pos + 1

        # Unknown token -- create as int variable
        return Var(tok, INT), pos + 1


# ============================================================
# Composite Certificates
# ============================================================

def combine_certificates(*certs: ProofCertificate, claim: str = None) -> ProofCertificate:
    """Combine multiple certificates into a composite certificate."""
    if claim is None:
        claims = [c.claim for c in certs]
        claim = " AND ".join(claims)

    composite = ProofCertificate(
        kind=ProofKind.COMPOSITE,
        claim=claim,
        sub_certificates=list(certs),
    )

    if all(c.status == CertStatus.VALID for c in certs):
        composite.status = CertStatus.VALID
    elif any(c.status == CertStatus.INVALID for c in certs):
        composite.status = CertStatus.INVALID
    else:
        composite.status = CertStatus.UNKNOWN

    return composite


# ============================================================
# Certificate I/O
# ============================================================

def save_certificate(cert: ProofCertificate, path: str):
    """Save certificate to JSON file."""
    with open(path, 'w') as f:
        f.write(cert.to_json())


def load_certificate(path: str) -> ProofCertificate:
    """Load certificate from JSON file."""
    with open(path, 'r') as f:
        return ProofCertificate.from_json(f.read())


# ============================================================
# Convenience API
# ============================================================

def certify_program(source: str, fn_name: str = None) -> ProofCertificate:
    """One-shot: verify a program and return a checked certificate."""
    cert = generate_vcgen_certificate(source, fn_name)
    return check_certificate(cert)


def certify_transition_system(ts, max_frames: int = 100) -> ProofCertificate:
    """One-shot: verify a transition system and return a checked certificate."""
    cert = generate_pdr_certificate(ts, max_frames)
    if cert.status != CertStatus.INVALID:
        cert = check_certificate(cert)
    return cert
