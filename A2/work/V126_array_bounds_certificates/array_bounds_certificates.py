"""V126: Array Bounds Certificates

Composes V123 (array bounds verification) + V044 (proof certificates) to produce
machine-checkable certificates proving array accesses are safe.

Pipeline:
  1. V123 verifies all array accesses (AI + SMT)
  2. For each safe access, encode the bounds proof as a V044 ProofObligation
  3. Bundle into a ProofCertificate with full SMT-LIB2 formulas
  4. Independent checker re-verifies obligations without re-running the analysis
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V123_array_bounds_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json

from array_bounds_verify import (
    verify_bounds, certify_bounds, find_unsafe_accesses, check_access_safe,
    verify_with_context, bounds_summary, check_certificate as v123_check_cert,
    VerificationResult, BoundsObligation, BoundsCertificate, Verdict,
    AccessInfo, ArrayBoundsVerifier, BoundsTrackingInterpreter, SMTEncoder,
    eval_index_interval, expr_to_str
)
from proof_certificates import (
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    check_certificate as v044_check_cert, combine_certificates,
    save_certificate, load_certificate
)
from smt_solver import SMTSolver, SMTResult, Var, IntConst, App, Op, INT, BOOL


# ---- Data structures ----

class ArrayCertKind(Enum):
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"


@dataclass
class ArrayBoundsCertificate:
    """Certificate proving all array accesses in a program are safe."""
    source: str
    obligations: List[ProofObligation]
    access_count: int
    safe_count: int
    unsafe_count: int
    unknown_count: int
    ai_safe_count: int
    smt_safe_count: int
    all_safe: bool
    status: CertStatus
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    @property
    def total_obligations(self) -> int:
        return len(self.obligations)

    @property
    def valid_obligations(self) -> int:
        return sum(1 for o in self.obligations if o.status == CertStatus.VALID)

    @property
    def invalid_obligations(self) -> int:
        return sum(1 for o in self.obligations if o.status == CertStatus.INVALID)

    def summary(self) -> str:
        lines = [
            f"Array Bounds Certificate: {self.status.value}",
            f"  Accesses: {self.access_count}",
            f"  Obligations: {self.total_obligations} "
            f"(valid={self.valid_obligations}, invalid={self.invalid_obligations})",
            f"  AI-safe: {self.ai_safe_count}, SMT-safe: {self.smt_safe_count}, "
            f"Unsafe: {self.unsafe_count}, Unknown: {self.unknown_count}",
        ]
        if self.all_safe:
            lines.append("  All accesses proven safe.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": "array_bounds",
            "source": self.source,
            "obligations": [o.to_dict() for o in self.obligations],
            "access_count": self.access_count,
            "safe_count": self.safe_count,
            "unsafe_count": self.unsafe_count,
            "unknown_count": self.unknown_count,
            "ai_safe_count": self.ai_safe_count,
            "smt_safe_count": self.smt_safe_count,
            "all_safe": self.all_safe,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> 'ArrayBoundsCertificate':
        return ArrayBoundsCertificate(
            source=d["source"],
            obligations=[ProofObligation.from_dict(o) for o in d["obligations"]],
            access_count=d["access_count"],
            safe_count=d["safe_count"],
            unsafe_count=d["unsafe_count"],
            unknown_count=d["unknown_count"],
            ai_safe_count=d["ai_safe_count"],
            smt_safe_count=d["smt_safe_count"],
            all_safe=d["all_safe"],
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> 'ArrayBoundsCertificate':
        return ArrayBoundsCertificate.from_dict(json.loads(s))


# ---- SMT-LIB2 encoding for obligations ----

def _encode_lower_bound_smtlib(index_name: str, context_vars: List[Tuple[str, int, int]]) -> str:
    """Encode lower bound obligation: given context, prove index >= 0."""
    lines = ["(set-logic LIA)"]
    all_vars = set()
    all_vars.add(index_name)
    for vname, _, _ in context_vars:
        all_vars.add(vname)
    for v in sorted(all_vars):
        lines.append(f"(declare-const {v} Int)")
    # Context constraints
    for vname, lo, hi in context_vars:
        if lo is not None:
            lines.append(f"(assert (>= {vname} {lo}))")
        if hi is not None:
            lines.append(f"(assert (<= {vname} {hi}))")
    # Negate the property: index >= 0  ->  assert index < 0
    lines.append(f"(assert (< {index_name} 0))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def _encode_upper_bound_smtlib(index_name: str, length_name: str,
                                context_vars: List[Tuple[str, int, int]]) -> str:
    """Encode upper bound obligation: given context, prove index < length."""
    lines = ["(set-logic LIA)"]
    all_vars = set()
    all_vars.add(index_name)
    all_vars.add(length_name)
    for vname, _, _ in context_vars:
        all_vars.add(vname)
    for v in sorted(all_vars):
        lines.append(f"(declare-const {v} Int)")
    for vname, lo, hi in context_vars:
        if lo is not None:
            lines.append(f"(assert (>= {vname} {lo}))")
        if hi is not None:
            lines.append(f"(assert (<= {vname} {hi}))")
    # Negate: index < length  ->  assert index >= length
    lines.append(f"(assert (>= {index_name} {length_name}))")
    lines.append("(check-sat)")
    return "\n".join(lines)


def _encode_ai_lower_smtlib(lo: int) -> str:
    """Encode AI-safe lower bound: lo >= 0 is trivially true."""
    return f"(set-logic LIA)\n; AI-safe: index lower bound = {lo} >= 0\n(check-sat)"


def _encode_ai_upper_smtlib(index_hi: int, length_lo: int) -> str:
    """Encode AI-safe upper bound: index.hi < length.lo."""
    return (f"(set-logic LIA)\n"
            f"; AI-safe: index upper bound = {index_hi} < length lower bound = {length_lo}\n"
            f"(check-sat)")


# ---- Core: obligation generation from V123 results ----

def _make_obligation_name(line: int, array: str, check: str, idx: int) -> str:
    return f"bounds_{idx}_{array}_L{line}_{check}"


def _bounds_obligation_to_proof(obl: BoundsObligation, idx: int) -> ProofObligation:
    """Convert a V123 BoundsObligation to a V044 ProofObligation."""
    name = _make_obligation_name(obl.access_line, obl.array_name, obl.check_type, idx)

    # Build human-readable formula
    if obl.check_type == "lower":
        formula_str = f"{obl.index_expr} >= 0"
        desc = f"Lower bound: {obl.array_name}[{obl.index_expr}] at line {obl.access_line}"
    else:
        formula_str = f"{obl.index_expr} < len({obl.array_name})"
        desc = f"Upper bound: {obl.array_name}[{obl.index_expr}] at line {obl.access_line}"

    # Build SMT-LIB2 formula
    if obl.verdict == Verdict.AI_SAFE:
        if obl.check_type == "lower":
            lo = obl.abstract_index[0] if obl.abstract_index else 0
            formula_smt = _encode_ai_lower_smtlib(lo)
        else:
            hi = obl.abstract_index[1] if obl.abstract_index else 0
            length_lo = obl.abstract_length[0] if obl.abstract_length else 1
            formula_smt = _encode_ai_upper_smtlib(hi, length_lo)
    elif obl.verdict == Verdict.SAFE:
        # Build full SMT encoding with context
        index_name = f"idx_{obl.array_name}_L{obl.access_line}"
        context_vars = []
        if obl.abstract_index:
            lo, hi = obl.abstract_index
            if lo is not None and lo > -1000:
                context_vars.append((index_name, lo, None))
            if hi is not None and hi < 1000:
                context_vars.append((index_name, None, hi))
        if obl.check_type == "lower":
            formula_smt = _encode_lower_bound_smtlib(index_name, context_vars)
        else:
            length_name = f"len_{obl.array_name}_L{obl.access_line}"
            if obl.abstract_length:
                lo_l, hi_l = obl.abstract_length
                if lo_l is not None and lo_l > 0:
                    context_vars.append((length_name, lo_l, None))
                if hi_l is not None and hi_l < 1000:
                    context_vars.append((length_name, None, hi_l))
            formula_smt = _encode_upper_bound_smtlib(index_name, length_name, context_vars)
    else:
        # UNSAFE or UNKNOWN -- record but can't prove
        formula_smt = f"; {obl.verdict.value}: no proof obligation"

    # Map V123 Verdict -> V044 CertStatus
    if obl.verdict in (Verdict.SAFE, Verdict.AI_SAFE):
        status = CertStatus.VALID
    elif obl.verdict == Verdict.UNSAFE:
        status = CertStatus.INVALID
    else:
        status = CertStatus.UNKNOWN

    return ProofObligation(
        name=name,
        description=desc,
        formula_str=formula_str,
        formula_smt=formula_smt,
        status=status,
        counterexample=obl.counterexample,
    )


# ---- Certificate generation ----

def _generate_certificate(source: str, result: VerificationResult) -> ArrayBoundsCertificate:
    """Generate certificate from V123 verification result."""
    obligations = []
    for i, obl in enumerate(result.obligations):
        obligations.append(_bounds_obligation_to_proof(obl, i))

    # Count by category
    ai_safe = sum(1 for o in result.obligations if o.verdict == Verdict.AI_SAFE)
    smt_safe = sum(1 for o in result.obligations if o.verdict == Verdict.SAFE)

    if result.all_safe:
        status = CertStatus.VALID
    elif result.unsafe_count > 0:
        status = CertStatus.INVALID
    else:
        status = CertStatus.UNKNOWN

    return ArrayBoundsCertificate(
        source=source,
        obligations=obligations,
        access_count=len(result.accesses),
        safe_count=result.safe_count,
        unsafe_count=result.unsafe_count,
        unknown_count=result.unknown_count,
        ai_safe_count=ai_safe,
        smt_safe_count=smt_safe,
        all_safe=result.all_safe,
        status=status,
        metadata={
            "total_accesses": len(result.accesses),
            "total_obligations": len(result.obligations),
            "method": "V123_ai_smt",
        },
    )


# ---- Independent certificate checking ----

def _check_ai_safe_obligation(obl: ProofObligation) -> CertStatus:
    """Check an AI-safe obligation: the abstract bounds directly prove safety."""
    # AI-safe obligations have their proof in the formula_smt comment
    if "; AI-safe:" in obl.formula_smt:
        # Parse the AI bounds from the comment
        line = obl.formula_smt.split("\n")[1]  # The comment line
        if "index lower bound" in line:
            # Extract: "index lower bound = N >= 0"
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    val = int(parts[1].strip().split()[0])
                    return CertStatus.VALID if val >= 0 else CertStatus.INVALID
                except (ValueError, IndexError):
                    pass
        elif "index upper bound" in line:
            # Extract: "index upper bound = N < length lower bound = M"
            parts = line.split("=")
            if len(parts) >= 3:
                try:
                    idx_hi = int(parts[1].strip().split()[0])
                    len_lo = int(parts[2].strip().split()[0])
                    return CertStatus.VALID if idx_hi < len_lo else CertStatus.INVALID
                except (ValueError, IndexError):
                    pass
    return CertStatus.UNKNOWN


def _check_smt_obligation(obl: ProofObligation) -> CertStatus:
    """Check an SMT obligation by re-running the query."""
    if obl.formula_smt.startswith(";"):
        return CertStatus.UNKNOWN

    # Parse the SMT-LIB2 and re-verify
    solver = SMTSolver()
    declared = {}
    constraints = []
    negated_prop = None

    for line in obl.formula_smt.split("\n"):
        line = line.strip()
        if line.startswith("(declare-const"):
            parts = line.replace("(", " ").replace(")", " ").split()
            if len(parts) >= 3:
                vname = parts[1]
                declared[vname] = solver.Int(vname)
        elif line.startswith("(assert"):
            # Parse the assertion
            inner = line[len("(assert "):-1].strip()
            term = _parse_smtlib_term(inner, declared, solver)
            if term is not None:
                constraints.append(term)

    # All assertions together should be UNSAT for the property to hold
    for c in constraints:
        solver.add(c)

    result = solver.check()
    if result == SMTResult.UNSAT:
        return CertStatus.VALID
    elif result == SMTResult.SAT:
        return CertStatus.INVALID
    else:
        return CertStatus.UNKNOWN


def _parse_smtlib_term(s: str, declared: dict, solver: SMTSolver):
    """Parse a simple SMT-LIB2 term into C037 objects."""
    s = s.strip()

    # Integer literal
    if s.lstrip('-').isdigit():
        val = int(s)
        if val < 0:
            return App(Op.SUB, [IntConst(0), IntConst(-val)], INT)
        return IntConst(val)

    # Variable
    if s in declared:
        return declared[s]

    # Parenthesized expression
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        # Tokenize respecting nested parens
        tokens = _tokenize_sexp(inner)
        if not tokens:
            return None

        op_str = tokens[0]
        args = [_parse_smtlib_term(t, declared, solver) for t in tokens[1:]]
        if any(a is None for a in args):
            return None

        op_map = {
            ">=": Op.GE, "<=": Op.LE, ">": Op.GT, "<": Op.LT,
            "=": Op.EQ, "+": Op.ADD, "-": Op.SUB, "*": Op.MUL,
        }

        if op_str in op_map:
            op = op_map[op_str]
            if op in (Op.GE, Op.LE, Op.GT, Op.LT, Op.EQ):
                return App(op, args, BOOL)
            else:
                return App(op, args, INT)
        elif op_str == "and":
            result = args[0]
            for a in args[1:]:
                result = App(Op.AND, [result, a], BOOL)
            return result
        elif op_str == "or":
            result = args[0]
            for a in args[1:]:
                result = App(Op.OR, [result, a], BOOL)
            return result
        elif op_str == "not":
            return App(Op.NOT, [args[0]], BOOL)

    return None


def _tokenize_sexp(s: str) -> List[str]:
    """Tokenize an S-expression into top-level tokens."""
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] == '(':
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            j = i
            while j < len(s) and not s[j].isspace() and s[j] not in '()':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def check_array_certificate(cert: ArrayBoundsCertificate) -> ArrayBoundsCertificate:
    """Independently re-verify all obligations in an array bounds certificate.

    Does NOT re-run V123 analysis. Instead:
    - AI-safe obligations: checks the abstract bound arithmetic
    - SMT obligations: re-runs the SMT query from the encoded formula
    """
    checked_obligations = []
    for obl in cert.obligations:
        new_obl = ProofObligation(
            name=obl.name,
            description=obl.description,
            formula_str=obl.formula_str,
            formula_smt=obl.formula_smt,
            status=obl.status,
            counterexample=obl.counterexample,
        )

        if "; AI-safe:" in obl.formula_smt:
            new_obl.status = _check_ai_safe_obligation(obl)
        elif obl.formula_smt.startswith(";"):
            # UNSAFE or UNKNOWN -- no proof to check
            new_obl.status = obl.status
        else:
            new_obl.status = _check_smt_obligation(obl)

        checked_obligations.append(new_obl)

    # Recompute overall status
    all_valid = all(o.status == CertStatus.VALID for o in checked_obligations)
    any_invalid = any(o.status == CertStatus.INVALID for o in checked_obligations)

    new_cert = ArrayBoundsCertificate(
        source=cert.source,
        obligations=checked_obligations,
        access_count=cert.access_count,
        safe_count=cert.safe_count,
        unsafe_count=cert.unsafe_count,
        unknown_count=cert.unknown_count,
        ai_safe_count=cert.ai_safe_count,
        smt_safe_count=cert.smt_safe_count,
        all_safe=cert.all_safe,
        status=CertStatus.VALID if all_valid else (CertStatus.INVALID if any_invalid else CertStatus.UNKNOWN),
        timestamp=cert.timestamp,
        metadata=cert.metadata,
    )
    return new_cert


# ---- V044 bridge: convert to standard ProofCertificate ----

def to_v044_certificate(cert: ArrayBoundsCertificate) -> ProofCertificate:
    """Convert ArrayBoundsCertificate to a standard V044 ProofCertificate."""
    return ProofCertificate(
        kind=ProofKind.VCGEN,  # Closest match -- verification condition style
        claim=f"All {cert.access_count} array accesses are within bounds",
        source=cert.source,
        obligations=list(cert.obligations),
        metadata={
            "array_bounds": True,
            "access_count": cert.access_count,
            "ai_safe_count": cert.ai_safe_count,
            "smt_safe_count": cert.smt_safe_count,
            "unsafe_count": cert.unsafe_count,
            "unknown_count": cert.unknown_count,
        },
        status=cert.status,
    )


def from_v044_certificate(cert: ProofCertificate, source: str = None) -> ArrayBoundsCertificate:
    """Convert a V044 ProofCertificate back to ArrayBoundsCertificate."""
    meta = cert.metadata or {}
    return ArrayBoundsCertificate(
        source=source or cert.source or "",
        obligations=list(cert.obligations),
        access_count=meta.get("access_count", len(cert.obligations) // 2),
        safe_count=meta.get("ai_safe_count", 0) + meta.get("smt_safe_count", 0),
        unsafe_count=meta.get("unsafe_count", 0),
        unknown_count=meta.get("unknown_count", 0),
        ai_safe_count=meta.get("ai_safe_count", 0),
        smt_safe_count=meta.get("smt_safe_count", 0),
        all_safe=cert.status == CertStatus.VALID,
        status=cert.status,
    )


# ---- Public APIs ----

def certify_array_bounds(source: str) -> ArrayBoundsCertificate:
    """Main API: Verify all array accesses and produce a certificate.

    Pipeline: V123 bounds verification -> obligation encoding -> certificate
    """
    result = verify_bounds(source)
    return _generate_certificate(source, result)


def certify_and_check(source: str) -> ArrayBoundsCertificate:
    """Generate certificate and independently verify it."""
    cert = certify_array_bounds(source)
    return check_array_certificate(cert)


def certify_with_context(source: str, constraints: Dict[str, Tuple[int, int]]) -> ArrayBoundsCertificate:
    """Certify with user-provided variable constraints."""
    result = verify_with_context(source, constraints)
    return _generate_certificate(source, result)


def save_array_certificate(cert: ArrayBoundsCertificate, path: str):
    """Save certificate to JSON file."""
    with open(path, 'w') as f:
        json.dump(cert.to_dict(), f, indent=2)


def load_array_certificate(path: str) -> ArrayBoundsCertificate:
    """Load certificate from JSON file."""
    with open(path, 'r') as f:
        return ArrayBoundsCertificate.from_dict(json.load(f))


def combine_array_certificates(*certs: ArrayBoundsCertificate) -> ArrayBoundsCertificate:
    """Combine multiple array bounds certificates (e.g., from different modules)."""
    all_obligations = []
    total_accesses = 0
    total_safe = 0
    total_unsafe = 0
    total_unknown = 0
    total_ai_safe = 0
    total_smt_safe = 0
    sources = []

    for cert in certs:
        all_obligations.extend(cert.obligations)
        total_accesses += cert.access_count
        total_safe += cert.safe_count
        total_unsafe += cert.unsafe_count
        total_unknown += cert.unknown_count
        total_ai_safe += cert.ai_safe_count
        total_smt_safe += cert.smt_safe_count
        sources.append(cert.source)

    all_safe = total_unsafe == 0 and total_unknown == 0
    if all_safe:
        status = CertStatus.VALID
    elif total_unsafe > 0:
        status = CertStatus.INVALID
    else:
        status = CertStatus.UNKNOWN

    return ArrayBoundsCertificate(
        source="\n---\n".join(sources),
        obligations=all_obligations,
        access_count=total_accesses,
        safe_count=total_safe,
        unsafe_count=total_unsafe,
        unknown_count=total_unknown,
        ai_safe_count=total_ai_safe,
        smt_safe_count=total_smt_safe,
        all_safe=all_safe,
        status=status,
        metadata={"combined": True, "num_modules": len(certs)},
    )


def compare_certification_strength(source: str) -> dict:
    """Compare AI-only vs AI+SMT certification strength."""
    result = verify_bounds(source)

    ai_only_safe = sum(1 for o in result.obligations if o.verdict == Verdict.AI_SAFE)
    smt_safe = sum(1 for o in result.obligations if o.verdict == Verdict.SAFE)
    total = len(result.obligations)

    return {
        "total_obligations": total,
        "ai_safe": ai_only_safe,
        "smt_safe": smt_safe,
        "total_safe": ai_only_safe + smt_safe,
        "unsafe": result.unsafe_count,
        "unknown": result.unknown_count,
        "ai_coverage": ai_only_safe / total if total > 0 else 0,
        "smt_additional_coverage": smt_safe / total if total > 0 else 0,
        "total_coverage": (ai_only_safe + smt_safe) / total if total > 0 else 0,
        "smt_lift": (f"SMT proved {smt_safe} additional obligations "
                     f"beyond AI's {ai_only_safe}") if smt_safe > 0
                    else "AI alone was sufficient",
    }


def certificate_summary(source: str) -> str:
    """Human-readable certificate summary."""
    cert = certify_and_check(source)
    lines = [cert.summary(), ""]

    for obl in cert.obligations:
        marker = {
            CertStatus.VALID: "[VALID]",
            CertStatus.INVALID: "[FAIL]",
            CertStatus.UNKNOWN: "[????]",
            CertStatus.UNCHECKED: "[----]",
        }.get(obl.status, "[????]")
        lines.append(f"  {marker} {obl.description}")
        if obl.counterexample:
            lines.append(f"         Counterexample: {obl.counterexample}")

    return "\n".join(lines)
