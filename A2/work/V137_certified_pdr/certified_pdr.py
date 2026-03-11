"""V137: Certified PDR -- Machine-checkable certificates for PDR/IC3 proofs.

Composes V002 (PDR/IC3) + V044 (proof certificates) + V136 (certified k-induction).

Extends V044's existing PDR certificate with:
- Richer certificate data structure with invariant clauses
- Source-level API for loop verification
- JSON round-trip serialization
- Comparison with k-induction (V136)
- Combined certification strategy (try both, pick winner)
"""

import sys, os, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V136_certified_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from pdr import TransitionSystem, PDREngine, PDRResult
from proof_certificates import (
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    generate_pdr_certificate, check_certificate,
    smt_term_to_str, smt_term_to_smtlib,
)
from smt_solver import SMTSolver, Var, App, Op, IntConst, BoolConst, INT, BOOL, SMTResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PDRCertKind(Enum):
    PDR = "pdr"
    COMBINED = "combined"  # PDR + k-induction comparison


@dataclass
class PDRCertificate:
    """Certificate for a PDR proof."""
    kind: PDRCertKind
    claim: str
    result: str  # "safe", "unsafe", "unknown"
    invariant_clauses: List[str] = field(default_factory=list)
    obligations: List[ProofObligation] = field(default_factory=list)
    counterexample: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        valid = sum(1 for o in self.obligations if o.status == CertStatus.VALID)
        invalid = sum(1 for o in self.obligations if o.status == CertStatus.INVALID)
        lines = [
            f"PDRCertificate: {self.claim}",
            f"  Kind: {self.kind.value}, Result: {self.result}",
            f"  Status: {self.status.value}",
            f"  Obligations: {len(self.obligations)} total",
            f"  Valid: {valid}, Invalid: {invalid}",
        ]
        if self.invariant_clauses:
            lines.append(f"  Invariant clauses: {len(self.invariant_clauses)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "claim": self.claim,
            "result": self.result,
            "invariant_clauses": self.invariant_clauses,
            "obligations": [_ob_to_dict(o) for o in self.obligations],
            "counterexample": self.counterexample,
            "metadata": self.metadata,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> 'PDRCertificate':
        return PDRCertificate(
            kind=PDRCertKind(d["kind"]),
            claim=d["claim"],
            result=d["result"],
            invariant_clauses=d.get("invariant_clauses", []),
            obligations=[_ob_from_dict(o) for o in d["obligations"]],
            counterexample=d.get("counterexample"),
            metadata=d.get("metadata", {}),
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(s: str) -> 'PDRCertificate':
        import json
        return PDRCertificate.from_dict(json.loads(s))


def _ob_to_dict(o: ProofObligation) -> dict:
    d = {
        "name": o.name,
        "description": o.description,
        "formula_str": o.formula_str,
        "formula_smt": o.formula_smt,
        "status": o.status.value,
    }
    if o.counterexample:
        d["counterexample"] = o.counterexample
    return d


def _ob_from_dict(d: dict) -> ProofObligation:
    return ProofObligation(
        name=d["name"],
        description=d["description"],
        formula_str=d["formula_str"],
        formula_smt=d["formula_smt"],
        status=CertStatus(d["status"]),
        counterexample=d.get("counterexample"),
    )


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

def certify_pdr(ts, max_frames=100):
    """Run PDR and generate a certificate with independent checking."""
    t0 = time.time()

    # Use V044's generate_pdr_certificate (already does the heavy lifting)
    v044_cert = generate_pdr_certificate(ts, max_frames)
    duration = time.time() - t0

    # Extract invariant clauses
    inv_clauses = v044_cert.metadata.get("invariant_clauses", [])
    pdr_result = v044_cert.metadata.get("result", "unknown")

    if pdr_result == "safe":
        result_str = "safe"
        claim = "Property holds for all reachable states (PDR)"
    elif pdr_result == "unsafe":
        result_str = "unsafe"
        claim = "Property violated (counterexample found by PDR)"
    else:
        result_str = "unknown"
        claim = "PDR could not determine safety"

    cert = PDRCertificate(
        kind=PDRCertKind.PDR,
        claim=claim,
        result=result_str,
        invariant_clauses=inv_clauses,
        obligations=list(v044_cert.obligations),
        counterexample=v044_cert.metadata.get("counterexample_trace"),
        metadata={
            "method": "pdr",
            "num_frames": v044_cert.metadata.get("num_frames", 0),
            "duration": duration,
        },
        status=v044_cert.status,
        timestamp=v044_cert.timestamp,
    )

    return cert


def certify_and_check_pdr(ts, max_frames=100):
    """Generate PDR certificate and independently verify it."""
    cert = certify_pdr(ts, max_frames)

    if cert.result == "safe":
        check_pdr_certificate(cert)
    return cert


def check_pdr_certificate(cert):
    """Independently verify all obligations in a PDR certificate."""
    # Build a V044 certificate and use its checker
    v044_cert = to_v044_certificate(cert)
    checked = check_certificate(v044_cert)

    # Copy status back
    for i, ob in enumerate(cert.obligations):
        if i < len(checked.obligations):
            ob.status = checked.obligations[i].status
            ob.counterexample = checked.obligations[i].counterexample

    cert.status = checked.status
    return cert


# ---------------------------------------------------------------------------
# Source-level API
# ---------------------------------------------------------------------------

def certify_pdr_loop(source, property_source, max_frames=100):
    """Source-level certified PDR for while loops in C10 source."""
    from k_induction import _extract_loop_ts

    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_expr_to_smt(property_source, ts_vars)
    ts.set_property(prop_smt)
    return certify_and_check_pdr(ts, max_frames)


def _parse_expr_to_smt(expr_str, ts_vars):
    """Parse a C10 expression string to SMT term."""
    from stack_vm import lex, Parser, IntLit, Var as ASTVar, BinOp

    tokens = lex(f"let __p = ({expr_str});")
    stmts = Parser(tokens).parse().stmts
    expr = stmts[0].value

    def convert(e):
        if isinstance(e, IntLit):
            return IntConst(e.value)
        elif isinstance(e, ASTVar):
            if e.name in ts_vars:
                return ts_vars[e.name]
            return IntConst(0)
        elif isinstance(e, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = convert(e.left)
            r = convert(e.right)
            op = op_map.get(e.op)
            if op is None:
                raise ValueError(f"Unknown op: {e.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        return IntConst(0)

    return convert(expr)


# ---------------------------------------------------------------------------
# V044 bridge
# ---------------------------------------------------------------------------

def to_v044_certificate(cert):
    """Convert PDRCertificate to V044 ProofCertificate."""
    return ProofCertificate(
        kind=ProofKind.PDR,
        claim=cert.claim,
        source=None,
        obligations=list(cert.obligations),
        metadata={
            **cert.metadata,
            "cert_kind": cert.kind.value,
            "result": cert.result,
            "invariant_clauses": cert.invariant_clauses,
        },
        status=cert.status,
        timestamp=cert.timestamp,
    )


# ---------------------------------------------------------------------------
# Comparison and combined strategies
# ---------------------------------------------------------------------------

def compare_pdr_vs_kind(ts, max_frames=100, max_k=20):
    """Compare PDR certification with k-induction certification."""
    from certified_k_induction import certify_and_check as kind_certify

    t0 = time.time()
    pdr_cert = certify_and_check_pdr(ts, max_frames)
    pdr_time = time.time() - t0

    t0 = time.time()
    kind_cert = kind_certify(ts, max_k=max_k)
    kind_time = time.time() - t0

    return {
        "pdr_result": pdr_cert.result,
        "pdr_status": pdr_cert.status.value,
        "pdr_obligations": len(pdr_cert.obligations),
        "pdr_time": pdr_time,
        "kind_result": kind_cert.result,
        "kind_status": kind_cert.status.value,
        "kind_k": kind_cert.k,
        "kind_obligations": len(kind_cert.obligations),
        "kind_time": kind_time,
    }


def certify_combined(ts, max_frames=100, max_k=20):
    """Try both PDR and k-induction, return the best certified result."""
    from certified_k_induction import certify_and_check as kind_certify

    # Try k-induction first (often faster for simple systems)
    kind_cert = kind_certify(ts, max_k=max_k)
    if kind_cert.result == "safe" and kind_cert.status == CertStatus.VALID:
        return PDRCertificate(
            kind=PDRCertKind.COMBINED,
            claim=f"Property holds (k-induction, k={kind_cert.k})",
            result="safe",
            obligations=kind_cert.obligations,
            metadata={
                "method": "combined",
                "winner": "k-induction",
                "k": kind_cert.k,
            },
            status=CertStatus.VALID,
        )

    # Try PDR
    pdr_cert = certify_and_check_pdr(ts, max_frames)
    if pdr_cert.result == "safe" and pdr_cert.status == CertStatus.VALID:
        pdr_cert.kind = PDRCertKind.COMBINED
        pdr_cert.metadata["method"] = "combined"
        pdr_cert.metadata["winner"] = "pdr"
        return pdr_cert

    # Both failed or unsafe
    if kind_cert.result == "unsafe":
        return PDRCertificate(
            kind=PDRCertKind.COMBINED,
            claim="Property violated",
            result="unsafe",
            counterexample=kind_cert.counterexample,
            metadata={"method": "combined", "winner": "k-induction"},
        )
    if pdr_cert.result == "unsafe":
        return pdr_cert

    return PDRCertificate(
        kind=PDRCertKind.COMBINED,
        claim="Verification inconclusive",
        result="unknown",
        metadata={"method": "combined"},
    )


def compare_certified_vs_uncertified(ts, max_frames=100):
    """Compare certified PDR with plain PDR."""
    t0 = time.time()
    engine = PDREngine(ts, max_frames)
    plain_result = engine.check()
    plain_time = time.time() - t0

    t0 = time.time()
    cert = certify_and_check_pdr(ts, max_frames)
    cert_time = time.time() - t0

    return {
        "plain_result": plain_result.result.value,
        "plain_time": plain_time,
        "certified_result": cert.result,
        "certified_status": cert.status.value,
        "certified_obligations": len(cert.obligations),
        "certified_time": cert_time,
        "overhead_ratio": cert_time / plain_time if plain_time > 0 else 0,
    }


def pdr_certificate_summary(cert):
    """Get human-readable summary."""
    return cert.summary()
