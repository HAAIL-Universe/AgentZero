"""
V139: Certified Regression Verification
=========================================
Compose V134 (certified equivalence) + V136 (certified k-induction) for
certified regression verification with proof certificates.

Strategy:
  1. Try certified equivalence (V134): prove new version matches old
  2. If equivalent -> certificate proves property preserved trivially
  3. If not equivalent -> fall back to certified k-induction (V136) on new version
  4. Combined certificate: equivalence-based OR k-induction-based proof

Composes:
  - V134 (certified equivalence) -- prove old == new
  - V136 (certified k-induction) -- prove property on new independently
  - V044 (proof certificates) -- ProofObligation, CertStatus
  - V002 (PDR) -- TransitionSystem
  - C037 (SMT solver)
  - C010 (parser)
"""

from __future__ import annotations
import sys, os, json, time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto
from datetime import datetime

# --- Path setup ---
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, '..', 'V134_certified_equivalence'))
sys.path.insert(0, os.path.join(_here, '..', 'V136_certified_k_induction'))
sys.path.insert(0, os.path.join(_here, '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(_here, '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(_here, '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(_here, '..', 'V006_equivalence_checking'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C038_symbolic_execution'))

from certified_equivalence import (
    certify_regression as v134_certify_regression,
    certify_function_equivalence as v134_certify_fn_equiv,
    certify_and_check as v134_certify_and_check,
    check_equiv_certificate,
    EquivCertificate, EquivCertKind, PathPairObligation,
)
from certified_k_induction import (
    certify_k_induction, certify_strengthened_k_induction,
    certify_loop, certify_loop_with_invariants,
    check_kind_certificate, KIndCertificate, KIndCertKind,
)
from proof_certificates import (
    ProofCertificate, ProofObligation, CertStatus, ProofKind,
    combine_certificates,
)
from pdr import TransitionSystem
from smt_solver import SMTSolver, Var, App, Op, IntConst, BoolConst, INT, BOOL, SMTResult


# ============================================================
# Result types
# ============================================================

class RegressionMethod(Enum):
    """How the regression was verified."""
    EQUIVALENCE = auto()    # Old == New proven via V134
    K_INDUCTION = auto()    # Property re-proven on new version via V136
    COMBINED = auto()       # Both attempted, best used
    NONE = auto()           # Neither succeeded


class RegressionVerdict(Enum):
    """Regression verification result."""
    SAFE = auto()           # Regression proven safe
    UNSAFE = auto()         # Regression introduces violation
    UNKNOWN = auto()        # Could not determine


@dataclass
class RegressionCertificate:
    """Combined regression verification certificate."""
    verdict: RegressionVerdict
    method: RegressionMethod
    claim: str
    source_old: str
    source_new: str
    property_desc: Optional[str] = None
    equiv_cert: Optional[EquivCertificate] = None
    kind_cert: Optional[KIndCertificate] = None
    obligations: List[ProofObligation] = field(default_factory=list)
    counterexample: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

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
            f"Regression Certificate: {self.verdict.name}",
            f"  Method: {self.method.name}",
            f"  Claim: {self.claim}",
            f"  Obligations: {self.valid_obligations}/{self.total_obligations} valid",
            f"  Status: {self.status.value}",
        ]
        if self.property_desc:
            lines.insert(2, f"  Property: {self.property_desc}")
        if self.counterexample:
            lines.append(f"  Counterexample: {self.counterexample}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "verdict": self.verdict.name,
            "method": self.method.name,
            "claim": self.claim,
            "source_old": self.source_old,
            "source_new": self.source_new,
            "property_desc": self.property_desc,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "obligations": [
                {
                    "name": o.name,
                    "description": o.description,
                    "formula_str": o.formula_str,
                    "formula_smt": getattr(o, 'formula_smt', ''),
                    "status": o.status.value,
                }
                for o in self.obligations
            ],
        }
        if self.counterexample:
            d["counterexample"] = self.counterexample
        if self.equiv_cert:
            d["equiv_cert_result"] = self.equiv_cert.result
        if self.kind_cert:
            d["kind_cert_result"] = self.kind_cert.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(d: dict) -> RegressionCertificate:
        obligations = []
        for od in d.get("obligations", []):
            obligations.append(ProofObligation(
                name=od["name"],
                description=od["description"],
                formula_str=od.get("formula_str", ""),
                formula_smt=od.get("formula_smt", ""),
                status=CertStatus(od["status"]),
            ))
        return RegressionCertificate(
            verdict=RegressionVerdict[d["verdict"]],
            method=RegressionMethod[d["method"]],
            claim=d["claim"],
            source_old=d["source_old"],
            source_new=d["source_new"],
            property_desc=d.get("property_desc"),
            obligations=obligations,
            counterexample=d.get("counterexample"),
            metadata=d.get("metadata", {}),
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )

    @staticmethod
    def from_json(s: str) -> RegressionCertificate:
        return RegressionCertificate.from_dict(json.loads(s))


# ============================================================
# Core: Equivalence-based regression verification
# ============================================================

def _try_equivalence(source_old: str, source_new: str,
                     symbolic_inputs: Optional[Dict] = None,
                     fn_name: Optional[str] = None,
                     param_types: Optional[Dict] = None,
                     output_var: Optional[str] = None,
                     max_paths: int = 64) -> Optional[EquivCertificate]:
    """Try to prove old == new via certified equivalence."""
    try:
        if fn_name and param_types:
            cert = v134_certify_fn_equiv(
                source_old, fn_name,
                source_new, fn_name,
                param_types=param_types,
                max_paths=max_paths,
            )
        elif symbolic_inputs:
            cert = v134_certify_regression(
                source_old, source_new,
                symbolic_inputs=symbolic_inputs,
                output_var=output_var,
                fn_name=fn_name,
                param_types=param_types,
                max_paths=max_paths,
            )
        else:
            return None

        return cert
    except Exception:
        return None


# ============================================================
# Core: k-Induction-based regression verification
# ============================================================

def _try_k_induction(source_new: str, property_source: str,
                     invariant_sources: Optional[List[str]] = None,
                     max_k: int = 20) -> Optional[KIndCertificate]:
    """Try to prove property on new version via certified k-induction."""
    try:
        if invariant_sources:
            cert = certify_loop_with_invariants(
                source_new, property_source, invariant_sources, max_k=max_k
            )
        else:
            cert = certify_loop(source_new, property_source, max_k=max_k)
        return cert
    except Exception:
        return None


# ============================================================
# Main API: Certified regression verification
# ============================================================

def verify_regression(source_old: str, source_new: str,
                      symbolic_inputs: Optional[Dict] = None,
                      property_source: Optional[str] = None,
                      fn_name: Optional[str] = None,
                      param_types: Optional[Dict] = None,
                      output_var: Optional[str] = None,
                      invariant_sources: Optional[List[str]] = None,
                      max_paths: int = 64,
                      max_k: int = 20) -> RegressionCertificate:
    """Certified regression verification.

    Strategy:
      1. Try certified equivalence (fast path)
      2. If not equivalent and property given, try certified k-induction
      3. Return combined certificate

    Args:
        source_old: Original C10 source code
        source_new: Modified C10 source code
        symbolic_inputs: Dict of {var: (lo, hi)} for equivalence checking
        property_source: Source-level property for k-induction (e.g., "i >= 0")
        fn_name: Function name for function-level checking
        param_types: Dict of {param: (lo, hi)} for function equivalence
        output_var: Variable to compare in program equivalence
        invariant_sources: Loop invariants for k-induction strengthening
        max_paths: Max symbolic paths for equivalence
        max_k: Max k for k-induction

    Returns:
        RegressionCertificate with proof obligations and verdict
    """
    t0 = time.time()
    claim = "Regression verification: new version preserves properties of old version"

    # Phase 1: Try equivalence
    equiv_cert = _try_equivalence(
        source_old, source_new,
        symbolic_inputs=symbolic_inputs,
        fn_name=fn_name,
        param_types=param_types,
        output_var=output_var,
        max_paths=max_paths,
    )

    if equiv_cert is not None and equiv_cert.result == "equivalent":
        # Equivalence proven: property preserved trivially
        obligations = list(equiv_cert.obligations)
        return RegressionCertificate(
            verdict=RegressionVerdict.SAFE,
            method=RegressionMethod.EQUIVALENCE,
            claim=claim,
            source_old=source_old,
            source_new=source_new,
            property_desc=property_source,
            equiv_cert=equiv_cert,
            obligations=obligations,
            metadata={
                "method": "equivalence",
                "equiv_result": equiv_cert.result,
                "duration": time.time() - t0,
                "path_pairs_checked": equiv_cert.metadata.get("path_pairs_checked", 0),
            },
            status=equiv_cert.status,
        )

    # Phase 2: Try k-induction on new version (if property given)
    kind_cert = None
    if property_source:
        kind_cert = _try_k_induction(
            source_new, property_source,
            invariant_sources=invariant_sources,
            max_k=max_k,
        )

    if kind_cert is not None and kind_cert.result == "safe":
        obligations = list(kind_cert.obligations)
        return RegressionCertificate(
            verdict=RegressionVerdict.SAFE,
            method=RegressionMethod.K_INDUCTION,
            claim=claim,
            source_old=source_old,
            source_new=source_new,
            property_desc=property_source,
            kind_cert=kind_cert,
            obligations=obligations,
            metadata={
                "method": "k_induction",
                "kind_result": kind_cert.result,
                "k": kind_cert.k,
                "duration": time.time() - t0,
                "equiv_attempted": equiv_cert is not None,
                "equiv_result": equiv_cert.result if equiv_cert else None,
            },
            status=kind_cert.status,
        )

    # Phase 3: Neither worked
    # Check if equivalence found a counterexample
    counterexample = None
    if equiv_cert and equiv_cert.result == "not_equivalent":
        counterexample = equiv_cert.counterexample
    elif kind_cert and kind_cert.result == "unsafe":
        counterexample = kind_cert.counterexample

    if counterexample:
        verdict = RegressionVerdict.UNSAFE
    else:
        verdict = RegressionVerdict.UNKNOWN

    all_obligations = []
    if equiv_cert:
        all_obligations.extend(equiv_cert.obligations)
    if kind_cert:
        all_obligations.extend(kind_cert.obligations)

    return RegressionCertificate(
        verdict=verdict,
        method=RegressionMethod.COMBINED if (equiv_cert and kind_cert) else RegressionMethod.NONE,
        claim=claim,
        source_old=source_old,
        source_new=source_new,
        property_desc=property_source,
        equiv_cert=equiv_cert,
        kind_cert=kind_cert,
        obligations=all_obligations,
        counterexample=counterexample,
        metadata={
            "method": "combined" if (equiv_cert and kind_cert) else "none",
            "equiv_result": equiv_cert.result if equiv_cert else None,
            "kind_result": kind_cert.result if kind_cert else None,
            "duration": time.time() - t0,
        },
    )


def verify_function_regression(source_old: str, source_new: str,
                               fn_name: str, param_types: Dict,
                               property_source: Optional[str] = None,
                               max_paths: int = 64,
                               max_k: int = 20) -> RegressionCertificate:
    """Verify regression for a specific function.

    Args:
        source_old: Old source
        source_new: New source
        fn_name: Function to check
        param_types: {param_name: (lo, hi)} ranges
        property_source: Optional property to verify if not equivalent
    """
    return verify_regression(
        source_old, source_new,
        fn_name=fn_name,
        param_types=param_types,
        property_source=property_source,
        max_paths=max_paths,
        max_k=max_k,
    )


def verify_program_regression(source_old: str, source_new: str,
                              symbolic_inputs: Dict,
                              output_var: Optional[str] = None,
                              property_source: Optional[str] = None,
                              max_paths: int = 64) -> RegressionCertificate:
    """Verify regression for whole programs.

    Args:
        source_old: Old program source
        source_new: New program source
        symbolic_inputs: {var: (lo, hi)} input ranges
        output_var: Variable to compare
        property_source: Optional property for fallback
    """
    return verify_regression(
        source_old, source_new,
        symbolic_inputs=symbolic_inputs,
        output_var=output_var,
        property_source=property_source,
        max_paths=max_paths,
    )


# ============================================================
# Certificate checking
# ============================================================

def check_regression_certificate(cert: RegressionCertificate) -> RegressionCertificate:
    """Independently verify a regression certificate.

    Re-checks all proof obligations via SMT.
    """
    if cert.equiv_cert:
        checked_equiv = check_equiv_certificate(cert.equiv_cert)
        cert.equiv_cert = checked_equiv

    if cert.kind_cert:
        checked_kind = check_kind_certificate(cert.kind_cert)
        cert.kind_cert = checked_kind

    # Re-check obligations
    all_valid = True
    any_invalid = False

    for obl in cert.obligations:
        if obl.status == CertStatus.VALID:
            continue
        elif obl.status == CertStatus.INVALID:
            any_invalid = True
            all_valid = False
        else:
            all_valid = False

    if any_invalid:
        cert.status = CertStatus.INVALID
    elif all_valid and len(cert.obligations) > 0:
        cert.status = CertStatus.VALID
    else:
        cert.status = CertStatus.UNCHECKED

    return cert


# ============================================================
# Serialization
# ============================================================

def save_regression_certificate(cert: RegressionCertificate, path: str):
    """Save certificate to JSON file."""
    with open(path, 'w') as f:
        f.write(cert.to_json())


def load_regression_certificate(path: str) -> RegressionCertificate:
    """Load certificate from JSON file."""
    with open(path, 'r') as f:
        return RegressionCertificate.from_json(f.read())


# ============================================================
# Comparison API
# ============================================================

def compare_equiv_vs_kind(source_old: str, source_new: str,
                          symbolic_inputs: Dict,
                          property_source: str,
                          output_var: Optional[str] = None,
                          max_paths: int = 64,
                          max_k: int = 20) -> dict:
    """Compare equivalence-based vs k-induction-based regression verification.

    Returns comparison dict with results from both approaches.
    """
    # Approach 1: Equivalence
    t0 = time.time()
    equiv_cert = _try_equivalence(
        source_old, source_new,
        symbolic_inputs=symbolic_inputs,
        output_var=output_var,
        max_paths=max_paths,
    )
    equiv_time = time.time() - t0

    # Approach 2: k-Induction
    t0 = time.time()
    kind_cert = _try_k_induction(
        source_new, property_source,
        max_k=max_k,
    )
    kind_time = time.time() - t0

    return {
        "equivalence": {
            "result": equiv_cert.result if equiv_cert else "error",
            "obligations": len(equiv_cert.obligations) if equiv_cert else 0,
            "time": equiv_time,
        },
        "k_induction": {
            "result": kind_cert.result if kind_cert else "error",
            "k": kind_cert.k if kind_cert else None,
            "obligations": len(kind_cert.obligations) if kind_cert else 0,
            "time": kind_time,
        },
        "both_safe": (
            (equiv_cert is not None and equiv_cert.result == "equivalent") or
            (kind_cert is not None and kind_cert.result == "safe")
        ),
    }


def regression_summary(cert: RegressionCertificate) -> dict:
    """Get a human-readable summary of regression verification."""
    return {
        "verdict": cert.verdict.name,
        "method": cert.method.name,
        "claim": cert.claim,
        "property": cert.property_desc,
        "total_obligations": cert.total_obligations,
        "valid_obligations": cert.valid_obligations,
        "invalid_obligations": cert.invalid_obligations,
        "status": cert.status.value,
        "counterexample": cert.counterexample,
        "metadata": cert.metadata,
    }
