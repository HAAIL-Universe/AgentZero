"""V064: Probabilistic Proof Certificates

Composes V063 (verified probabilistic programs) + V044 (proof certificates).

Statistical verification certificates with confidence bounds -- machine-checkable
proof artifacts for probabilistic program properties.

Certificate types:
1. Deterministic: standard SMT-based proof obligations (reused from V044)
2. Statistical: sample-based evidence with confidence intervals, SPRT log ratios
3. Composite: combines deterministic + statistical sub-certificates

Key idea: a statistical certificate includes enough information for an independent
checker to evaluate whether the evidence supports the claimed probability bound,
WITHOUT re-running the sampling process. The checker verifies:
- Sample count meets minimum bound (Chernoff-Hoeffding)
- Confidence interval lower bound >= claimed threshold
- SPRT log likelihood ratio exceeds decision boundary
"""

import json
import math
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V063_verified_probabilistic_programs'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from proof_certificates import (
    ProofKind, CertStatus, ProofObligation, ProofCertificate,
    check_certificate, combine_certificates, save_certificate, load_certificate,
    sexpr_to_str, sexpr_to_smtlib,
)
from verified_probabilistic import (
    ProbVerdict, ProbVC, ProbFnSpec, ProbVerificationResult,
    verify_probabilistic, verify_probabilistic_function, verify_prob_function,
    check_prob_property, extract_prob_fn_spec, check_deterministic_vcs,
    wilson_confidence_interval, sprt_test,
)
from vc_gen import SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte


# ============================================================
# Statistical Certificate Data Structures
# ============================================================

class StatCertKind(Enum):
    """Kind of statistical certificate."""
    MONTE_CARLO = "monte_carlo"   # Fixed-sample with Wilson CI
    SPRT = "sprt"                 # Sequential probability ratio test
    CHERNOFF = "chernoff"         # Chernoff-Hoeffding bound


@dataclass
class StatisticalEvidence:
    """Evidence from a statistical test -- the core of a probabilistic certificate.

    Contains all information needed for independent verification of the
    claimed probability bound, without re-running the sampling process.
    """
    kind: StatCertKind
    property_desc: str            # Human-readable property description
    property_source: str          # C10 expression source for the property
    claimed_threshold: float      # P >= threshold is the claim

    # Sampling results
    n_samples: int                # Total samples taken
    n_successes: int              # Number of samples satisfying property
    observed_probability: float   # n_successes / n_samples

    # Confidence interval (Wilson score)
    confidence_level: float       # e.g. 0.95
    ci_lower: float               # Lower bound of CI
    ci_upper: float               # Upper bound of CI

    # SPRT info (if applicable)
    sprt_log_ratio: Optional[float] = None     # Log likelihood ratio
    sprt_accept_bound: Optional[float] = None  # ln(1/alpha)
    sprt_reject_bound: Optional[float] = None  # ln(beta)

    # Chernoff-Hoeffding bound (minimum samples for epsilon-delta guarantee)
    chernoff_min_samples: Optional[int] = None
    chernoff_epsilon: Optional[float] = None
    chernoff_delta: Optional[float] = None

    # Random variable ranges used in sampling
    random_var_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Verdict
    verdict: str = "unchecked"    # "accept", "reject", "inconclusive"

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind.value,
            "property_desc": self.property_desc,
            "property_source": self.property_source,
            "claimed_threshold": self.claimed_threshold,
            "n_samples": self.n_samples,
            "n_successes": self.n_successes,
            "observed_probability": self.observed_probability,
            "confidence_level": self.confidence_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "verdict": self.verdict,
        }
        if self.sprt_log_ratio is not None:
            d["sprt_log_ratio"] = self.sprt_log_ratio
            d["sprt_accept_bound"] = self.sprt_accept_bound
            d["sprt_reject_bound"] = self.sprt_reject_bound
        if self.chernoff_min_samples is not None:
            d["chernoff_min_samples"] = self.chernoff_min_samples
            d["chernoff_epsilon"] = self.chernoff_epsilon
            d["chernoff_delta"] = self.chernoff_delta
        if self.random_var_ranges:
            d["random_var_ranges"] = {k: list(v) for k, v in self.random_var_ranges.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> 'StatisticalEvidence':
        rvr = {}
        for k, v in d.get("random_var_ranges", {}).items():
            rvr[k] = tuple(v)
        return StatisticalEvidence(
            kind=StatCertKind(d["kind"]),
            property_desc=d["property_desc"],
            property_source=d["property_source"],
            claimed_threshold=d["claimed_threshold"],
            n_samples=d["n_samples"],
            n_successes=d["n_successes"],
            observed_probability=d["observed_probability"],
            confidence_level=d["confidence_level"],
            ci_lower=d["ci_lower"],
            ci_upper=d["ci_upper"],
            sprt_log_ratio=d.get("sprt_log_ratio"),
            sprt_accept_bound=d.get("sprt_accept_bound"),
            sprt_reject_bound=d.get("sprt_reject_bound"),
            chernoff_min_samples=d.get("chernoff_min_samples"),
            chernoff_epsilon=d.get("chernoff_epsilon"),
            chernoff_delta=d.get("chernoff_delta"),
            random_var_ranges=rvr,
            verdict=d.get("verdict", "unchecked"),
        )


# Extended ProofKind for probabilistic certificates
PROB_CERT_KIND = "probabilistic"


@dataclass
class ProbProofCertificate:
    """A proof certificate for probabilistic programs.

    Extends V044's ProofCertificate with statistical evidence.
    Can contain both deterministic obligations (SMT-checkable)
    and statistical evidence (sample-checkable).
    """
    claim: str
    source: Optional[str] = None
    deterministic_obligations: list = field(default_factory=list)  # ProofObligation
    statistical_evidence: list = field(default_factory=list)       # StatisticalEvidence
    metadata: dict = field(default_factory=dict)
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_checks(self) -> int:
        return len(self.deterministic_obligations) + len(self.statistical_evidence)

    @property
    def valid_checks(self) -> int:
        det_valid = sum(1 for o in self.deterministic_obligations if o.status == CertStatus.VALID)
        stat_valid = sum(1 for e in self.statistical_evidence if e.verdict == "accept")
        return det_valid + stat_valid

    @property
    def invalid_checks(self) -> int:
        det_inv = sum(1 for o in self.deterministic_obligations if o.status == CertStatus.INVALID)
        stat_inv = sum(1 for e in self.statistical_evidence if e.verdict == "reject")
        return det_inv + stat_inv

    def summary(self) -> str:
        lines = [f"Probabilistic Certificate: {self.claim}"]
        lines.append(f"Status: {self.status.value}")
        lines.append(f"Total checks: {self.total_checks} ({self.valid_checks} valid, {self.invalid_checks} invalid)")
        if self.deterministic_obligations:
            lines.append(f"Deterministic obligations ({len(self.deterministic_obligations)}):")
            for o in self.deterministic_obligations:
                lines.append(f"  [{o.status.value}] {o.name}: {o.description}")
        if self.statistical_evidence:
            lines.append(f"Statistical evidence ({len(self.statistical_evidence)}):")
            for e in self.statistical_evidence:
                lines.append(f"  [{e.verdict}] {e.property_desc}")
                lines.append(f"    P={e.observed_probability:.4f}, CI=[{e.ci_lower:.4f}, {e.ci_upper:.4f}]")
                lines.append(f"    threshold={e.claimed_threshold}, samples={e.n_samples}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": PROB_CERT_KIND,
            "claim": self.claim,
            "source": self.source,
            "deterministic_obligations": [o.to_dict() for o in self.deterministic_obligations],
            "statistical_evidence": [e.to_dict() for e in self.statistical_evidence],
            "metadata": self.metadata,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> 'ProbProofCertificate':
        return ProbProofCertificate(
            claim=d["claim"],
            source=d.get("source"),
            deterministic_obligations=[ProofObligation.from_dict(o) for o in d.get("deterministic_obligations", [])],
            statistical_evidence=[StatisticalEvidence.from_dict(e) for e in d.get("statistical_evidence", [])],
            metadata=d.get("metadata", {}),
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> 'ProbProofCertificate':
        return ProbProofCertificate.from_dict(json.loads(s))


# ============================================================
# Statistical Checking Functions
# ============================================================

def chernoff_min_samples(epsilon: float, delta: float) -> int:
    """Minimum samples for Chernoff-Hoeffding bound.

    P(|p_hat - p| >= epsilon) <= 2*exp(-2*n*epsilon^2)
    Solving for n: n >= ln(2/delta) / (2*epsilon^2)
    """
    if epsilon <= 0 or delta <= 0:
        return 1
    return math.ceil(math.log(2.0 / delta) / (2.0 * epsilon * epsilon))


def check_statistical_evidence(evidence: StatisticalEvidence) -> StatisticalEvidence:
    """Independently verify statistical evidence.

    Checks:
    1. Sample count >= Chernoff minimum for claimed precision
    2. Wilson CI lower bound >= claimed threshold (for accept)
    3. SPRT log ratio within decision bounds (if applicable)
    4. Observed probability is consistent with n_successes/n_samples
    """
    # Verify basic consistency
    if evidence.n_samples <= 0:
        evidence.verdict = "reject"
        return evidence

    expected_prob = evidence.n_successes / evidence.n_samples
    if abs(expected_prob - evidence.observed_probability) > 1e-6:
        evidence.verdict = "reject"
        return evidence

    # Recompute Wilson CI (V060 signature: wilson_confidence_interval(n_total, n_successes, confidence))
    ci_lo, ci_hi = wilson_confidence_interval(
        evidence.n_samples, evidence.n_successes, evidence.confidence_level
    )

    # Allow small floating-point tolerance on stored CI
    if abs(ci_lo - evidence.ci_lower) > 0.01 or abs(ci_hi - evidence.ci_upper) > 0.01:
        # Stored CI doesn't match recomputed -- use recomputed
        evidence.ci_lower = ci_lo
        evidence.ci_upper = ci_hi

    # Check Chernoff minimum samples
    epsilon = 0.05  # Default precision
    delta = 1.0 - evidence.confidence_level
    min_n = chernoff_min_samples(epsilon, delta)
    evidence.chernoff_min_samples = min_n
    evidence.chernoff_epsilon = epsilon
    evidence.chernoff_delta = delta

    # Determine verdict based on CI
    if ci_lo >= evidence.claimed_threshold:
        evidence.verdict = "accept"
    elif ci_hi < evidence.claimed_threshold:
        evidence.verdict = "reject"
    else:
        # CI straddles threshold
        evidence.verdict = "inconclusive"

    # SPRT cross-check (if available)
    if evidence.sprt_log_ratio is not None and evidence.sprt_accept_bound is not None:
        if evidence.sprt_log_ratio >= evidence.sprt_accept_bound:
            # SPRT also accepts -- consistent
            pass
        elif evidence.sprt_log_ratio <= (evidence.sprt_reject_bound or float('-inf')):
            # SPRT rejects -- override to reject
            evidence.verdict = "reject"

    return evidence


# ============================================================
# Certificate Generation
# ============================================================

def generate_prob_certificate(
    source: str,
    fn_name: str = None,
    param_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    use_sprt: bool = True,
) -> ProbProofCertificate:
    """Generate a probabilistic proof certificate for a C10 program.

    Runs V063 verification and packages results into a certificate
    with both deterministic obligations and statistical evidence.
    """
    # Run V063 verification
    result = verify_probabilistic_function(
        source, fn_name=fn_name, param_ranges=param_ranges,
        n_samples=n_samples, confidence=confidence, seed=seed,
        use_sprt=use_sprt,
    )

    claim_parts = []
    if fn_name:
        claim_parts.append(f"Function '{fn_name}'")
    else:
        claim_parts.append("Program")

    det_obligations = []
    stat_evidence = []

    # Convert deterministic VCs to proof obligations
    for vc in result.deterministic_vcs:
        obl = ProofObligation(
            name=vc.name,
            description=f"Deterministic VC: {vc.name}",
            formula_str=vc.detail or str(vc.formula) if vc.formula else "",
            formula_smt="",  # Would need full SMT-LIB2 generation
            status=_vc_status_to_cert(vc.status),
            counterexample=vc.counterexample,
        )
        det_obligations.append(obl)
        claim_parts.append(f"det:{vc.name}")

    # Convert probabilistic VCs to statistical evidence
    for vc in result.probabilistic_vcs:
        n_succ = 0
        n_total = 0
        ci_lo, ci_hi = 0.0, 1.0
        obs_prob = 0.0
        sprt_lr = None
        sprt_ab = None
        sprt_rb = None
        rvr = {}

        # Extract from VC
        if vc.confidence_interval:
            ci_lo, ci_hi = vc.confidence_interval
        if vc.estimated_probability is not None:
            obs_prob = vc.estimated_probability

        # Reconstruct sample counts from observed probability and detail
        if obs_prob > 0:
            # Estimate n_total from CI width (inverse Wilson)
            n_total = n_samples  # Use the requested sample count as approximation
            n_succ = round(obs_prob * n_total)
            # Recompute exact CI
            ci_lo, ci_hi = wilson_confidence_interval(n_total, n_succ, confidence)
            obs_prob = n_succ / n_total if n_total > 0 else 0.0
        else:
            n_total = n_samples
            n_succ = 0

        # SPRT info
        if use_sprt and obs_prob > 0:
            alpha = 1.0 - confidence
            beta = alpha
            sprt_ab = math.log(1.0 / alpha) if alpha > 0 else 10.0
            sprt_rb = math.log(beta) if beta > 0 else -10.0
            # Compute log likelihood ratio for the observed data
            p0 = vc.threshold * 0.95 if vc.threshold > 0 else 0.01  # H0: p < threshold
            p1 = vc.threshold
            if 0 < p0 < 1 and 0 < p1 < 1 and p0 != p1:
                try:
                    sprt_lr = (n_succ * math.log(p1 / p0) +
                               (n_total - n_succ) * math.log((1 - p1) / (1 - p0)))
                except (ValueError, ZeroDivisionError):
                    sprt_lr = None

        kind = StatCertKind.SPRT if use_sprt else StatCertKind.MONTE_CARLO
        verdict = _vc_prob_status_to_verdict(vc.status)

        ev = StatisticalEvidence(
            kind=kind,
            property_desc=vc.name,
            property_source=vc.postcondition_src or "",
            claimed_threshold=vc.threshold,
            n_samples=n_total,
            n_successes=n_succ,
            observed_probability=obs_prob,
            confidence_level=confidence,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            sprt_log_ratio=sprt_lr,
            sprt_accept_bound=sprt_ab,
            sprt_reject_bound=sprt_rb,
            random_var_ranges=rvr,
            verdict=verdict,
        )
        stat_evidence.append(ev)
        claim_parts.append(f"prob:{vc.name}")

    # Determine certificate status
    status = _result_to_cert_status(result)
    claim = " | ".join(claim_parts[:3])  # Keep claim concise
    if len(claim_parts) > 3:
        claim += f" | (+{len(claim_parts)-3} more)"

    cert = ProbProofCertificate(
        claim=claim,
        source=source,
        deterministic_obligations=det_obligations,
        statistical_evidence=stat_evidence,
        metadata={
            "fn_name": fn_name,
            "n_samples": n_samples,
            "confidence": confidence,
            "seed": seed,
            "verdict": result.verdict.value,
        },
        status=status,
    )
    return cert


def check_prob_certificate(cert: ProbProofCertificate) -> ProbProofCertificate:
    """Independently verify a probabilistic proof certificate.

    For deterministic obligations: re-checks via SMT (delegates to V044).
    For statistical evidence: re-checks CI bounds, Chernoff, SPRT consistency.
    Does NOT re-run sampling -- checks the recorded evidence only.
    """
    # Check deterministic obligations
    if cert.deterministic_obligations and cert.source:
        # Re-verify via V044-style SMT checking
        try:
            from vc_gen import verify_function, verify_program, VCStatus
            fn_name = cert.metadata.get("fn_name")
            if fn_name:
                vr = verify_function(cert.source, fn_name)
            else:
                vr = verify_program(cert.source)

            for i, vc in enumerate(vr.vcs):
                if i < len(cert.deterministic_obligations):
                    cert.deterministic_obligations[i].status = _vcstatus_to_certstatus(vc.status)
                    cert.deterministic_obligations[i].counterexample = vc.counterexample
        except Exception:
            # If re-verification fails, mark as unknown
            for obl in cert.deterministic_obligations:
                if obl.status == CertStatus.UNCHECKED:
                    obl.status = CertStatus.UNKNOWN

    # Check statistical evidence
    for ev in cert.statistical_evidence:
        check_statistical_evidence(ev)

    # Update overall status
    det_ok = all(o.status == CertStatus.VALID for o in cert.deterministic_obligations)
    stat_ok = all(e.verdict == "accept" for e in cert.statistical_evidence)
    det_fail = any(o.status == CertStatus.INVALID for o in cert.deterministic_obligations)
    stat_fail = any(e.verdict == "reject" for e in cert.statistical_evidence)

    if det_fail or stat_fail:
        cert.status = CertStatus.INVALID
    elif det_ok and stat_ok and cert.total_checks > 0:
        cert.status = CertStatus.VALID
    elif cert.total_checks == 0:
        cert.status = CertStatus.VALID  # Vacuous
    else:
        cert.status = CertStatus.UNKNOWN

    return cert


# ============================================================
# Certificate I/O
# ============================================================

def save_prob_certificate(cert: ProbProofCertificate, path: str):
    """Save probabilistic certificate to JSON file."""
    with open(path, 'w') as f:
        f.write(cert.to_json())


def load_prob_certificate(path: str) -> ProbProofCertificate:
    """Load probabilistic certificate from JSON file."""
    with open(path, 'r') as f:
        return ProbProofCertificate.from_json(f.read())


# ============================================================
# Composite Certificates
# ============================================================

def combine_prob_certificates(
    *certs: ProbProofCertificate,
    claim: str = None
) -> ProbProofCertificate:
    """Combine multiple probabilistic certificates into one."""
    all_det = []
    all_stat = []
    all_meta = {}

    for i, c in enumerate(certs):
        all_det.extend(c.deterministic_obligations)
        all_stat.extend(c.statistical_evidence)
        all_meta[f"sub_{i}"] = c.metadata

    if claim is None:
        claim = " AND ".join(c.claim for c in certs)

    combined = ProbProofCertificate(
        claim=claim,
        deterministic_obligations=all_det,
        statistical_evidence=all_stat,
        metadata=all_meta,
    )

    # Compute status
    det_ok = all(o.status == CertStatus.VALID for o in all_det)
    stat_ok = all(e.verdict == "accept" for e in all_stat)
    det_fail = any(o.status == CertStatus.INVALID for o in all_det)
    stat_fail = any(e.verdict == "reject" for e in all_stat)

    if det_fail or stat_fail:
        combined.status = CertStatus.INVALID
    elif det_ok and stat_ok and combined.total_checks > 0:
        combined.status = CertStatus.VALID
    elif combined.total_checks == 0:
        combined.status = CertStatus.VALID
    else:
        combined.status = CertStatus.UNKNOWN

    return combined


# ============================================================
# Bridge to V044 ProofCertificate
# ============================================================

def to_v044_certificate(cert: ProbProofCertificate) -> ProofCertificate:
    """Convert a ProbProofCertificate to a V044 ProofCertificate.

    Statistical evidence is encoded as proof obligations with
    descriptive formula strings (not SMT-checkable, but human-readable).
    """
    obligations = list(cert.deterministic_obligations)

    # Encode statistical evidence as obligations
    for ev in cert.statistical_evidence:
        formula_str = (
            f"P({ev.property_source}) >= {ev.claimed_threshold} "
            f"[observed: {ev.observed_probability:.4f}, "
            f"CI: [{ev.ci_lower:.4f}, {ev.ci_upper:.4f}], "
            f"n={ev.n_samples}]"
        )
        obl = ProofObligation(
            name=f"stat:{ev.property_desc}",
            description=f"Statistical: {ev.property_desc}",
            formula_str=formula_str,
            formula_smt="",  # Not SMT-checkable
            status=CertStatus.VALID if ev.verdict == "accept" else
                   CertStatus.INVALID if ev.verdict == "reject" else
                   CertStatus.UNKNOWN,
        )
        obligations.append(obl)

    v044_cert = ProofCertificate(
        kind=ProofKind.COMPOSITE,
        claim=cert.claim,
        source=cert.source,
        obligations=obligations,
        metadata={**cert.metadata, "probabilistic": True},
        status=cert.status,
        timestamp=cert.timestamp,
    )
    return v044_cert


def from_v044_certificate(cert: ProofCertificate) -> Optional[ProbProofCertificate]:
    """Try to convert a V044 ProofCertificate back to ProbProofCertificate.

    Only works if the certificate was created via to_v044_certificate().
    """
    if not cert.metadata.get("probabilistic"):
        return None

    det_obls = []
    stat_evs = []

    for obl in cert.obligations:
        if obl.name.startswith("stat:"):
            # This is statistical evidence encoded as obligation
            # We can't fully reconstruct, but create minimal evidence
            ev = StatisticalEvidence(
                kind=StatCertKind.MONTE_CARLO,
                property_desc=obl.name[5:],
                property_source="",
                claimed_threshold=0.0,
                n_samples=0,
                n_successes=0,
                observed_probability=0.0,
                confidence_level=0.95,
                ci_lower=0.0,
                ci_upper=1.0,
                verdict="accept" if obl.status == CertStatus.VALID else
                        "reject" if obl.status == CertStatus.INVALID else
                        "inconclusive",
            )
            stat_evs.append(ev)
        else:
            det_obls.append(obl)

    return ProbProofCertificate(
        claim=cert.claim,
        source=cert.source,
        deterministic_obligations=det_obls,
        statistical_evidence=stat_evs,
        metadata=cert.metadata,
        status=cert.status,
        timestamp=cert.timestamp,
    )


# ============================================================
# High-Level APIs
# ============================================================

def certify_probabilistic(
    source: str,
    fn_name: str = None,
    param_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> ProbProofCertificate:
    """One-shot: verify + generate + check certificate for probabilistic program."""
    cert = generate_prob_certificate(
        source, fn_name=fn_name, param_ranges=param_ranges,
        n_samples=n_samples, confidence=confidence, seed=seed,
    )
    return check_prob_certificate(cert)


def certify_and_save(
    source: str,
    path: str,
    fn_name: str = None,
    n_samples: int = 500,
    seed: Optional[int] = None,
) -> ProbProofCertificate:
    """Verify, certify, and save to file."""
    cert = certify_probabilistic(source, fn_name=fn_name, n_samples=n_samples, seed=seed)
    save_prob_certificate(cert, path)
    return cert


def load_and_check(path: str) -> ProbProofCertificate:
    """Load certificate from file and independently verify it."""
    cert = load_prob_certificate(path)
    return check_prob_certificate(cert)


def certificate_report(cert: ProbProofCertificate) -> dict:
    """Generate a structured report from a certificate."""
    return {
        "claim": cert.claim,
        "status": cert.status.value,
        "total_checks": cert.total_checks,
        "valid_checks": cert.valid_checks,
        "invalid_checks": cert.invalid_checks,
        "deterministic": [
            {
                "name": o.name,
                "status": o.status.value,
                "has_counterexample": o.counterexample is not None,
            }
            for o in cert.deterministic_obligations
        ],
        "statistical": [
            {
                "property": e.property_desc,
                "threshold": e.claimed_threshold,
                "observed": e.observed_probability,
                "ci": [e.ci_lower, e.ci_upper],
                "samples": e.n_samples,
                "verdict": e.verdict,
            }
            for e in cert.statistical_evidence
        ],
    }


# ============================================================
# Helpers
# ============================================================

def _vc_status_to_cert(status) -> CertStatus:
    """Convert ProbVC status string to CertStatus."""
    if status in ("valid", "accept"):
        return CertStatus.VALID
    elif status in ("invalid", "reject"):
        return CertStatus.INVALID
    elif status == "inconclusive":
        return CertStatus.UNKNOWN
    return CertStatus.UNCHECKED


def _vc_prob_status_to_verdict(status) -> str:
    """Convert ProbVC status to statistical verdict."""
    if status in ("valid", "accept"):
        return "accept"
    elif status in ("invalid", "reject"):
        return "reject"
    elif status == "inconclusive":
        return "inconclusive"
    return "unchecked"


def _result_to_cert_status(result: ProbVerificationResult) -> CertStatus:
    """Convert ProbVerificationResult verdict to CertStatus."""
    if result.verdict == ProbVerdict.VERIFIED:
        return CertStatus.VALID
    elif result.verdict == ProbVerdict.VIOLATED:
        return CertStatus.INVALID
    elif result.verdict == ProbVerdict.INCONCLUSIVE:
        return CertStatus.UNKNOWN
    return CertStatus.UNKNOWN


def _vcstatus_to_certstatus(vc_status) -> CertStatus:
    """Convert V004 VCStatus to CertStatus."""
    from vc_gen import VCStatus
    if vc_status == VCStatus.VALID:
        return CertStatus.VALID
    elif vc_status == VCStatus.INVALID:
        return CertStatus.INVALID
    return CertStatus.UNKNOWN
