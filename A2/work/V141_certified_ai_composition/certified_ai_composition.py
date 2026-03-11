"""V141: Certified AI-Strengthened k-Induction

Composes V046 (certified abstract interpretation) + V136 (certified k-induction)
to use abstract-interpretation-derived invariants to strengthen k-induction proofs.

Key insight: k-induction often fails because the property alone isn't inductive.
Abstract interpretation can cheaply compute invariants (interval bounds, signs) that
strengthen the induction hypothesis, making proofs succeed that would otherwise fail.

Pipeline:
1. Run V046 certified abstract interpretation on the program
2. Extract variable bounds as candidate invariants
3. Feed invariants into V136 strengthened k-induction
4. Combine certificates: AI soundness + k-induction validity = full proof
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
_challenges = os.path.join(_az, "challenges")

for p in [
    os.path.join(_work, "V046_certified_abstract_interpretation"),
    os.path.join(_work, "V136_certified_k_induction"),
    os.path.join(_work, "V044_proof_certificates"),
    os.path.join(_work, "V015_k_induction"),
    os.path.join(_work, "V004_verification_conditions"),
    os.path.join(_work, "V002_pdr"),
    os.path.join(_challenges, "C010_stack_vm"),
    os.path.join(_challenges, "C037_smt_solver"),
    os.path.join(_challenges, "C039_abstract_interpreter"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from certified_abstract_interpretation import (
    AIAnalysisResult, VerifiedAIResult,
    traced_analyze, generate_ai_certificate, check_ai_certificate,
    verified_analyze, certify_abstract_interpretation,
    certify_variable_bound, certify_sign,
)
from certified_k_induction import (
    KIndCertificate, KIndCertKind,
    certify_k_induction, certify_strengthened_k_induction,
    certify_and_check, certify_loop, certify_loop_with_invariants,
    check_kind_certificate, to_v044_certificate,
    compare_certified_vs_uncertified, kind_certificate_summary,
)
from proof_certificates import (
    ProofKind, CertStatus, ProofObligation, ProofCertificate,
    combine_certificates,
)
from abstract_interpreter import (
    Interval, INTERVAL_TOP, INTERVAL_BOT, Sign,
    analyze as ai_analyze, get_variable_range, get_variable_sign,
)
from stack_vm import lex, Parser, LetDecl, WhileStmt, FnDecl


class AIKIndVerdict(Enum):
    SAFE = "safe"           # Property proven with AI-strengthened k-induction
    UNSAFE = "unsafe"       # Counterexample found
    UNKNOWN = "unknown"     # Could not prove or disprove
    AI_ONLY = "ai_only"     # AI analysis completed but k-induction not attempted


class AIKIndMethod(Enum):
    BASIC_KIND = "basic_kind"               # Plain k-induction (no AI)
    AI_STRENGTHENED = "ai_strengthened"      # AI invariants + k-induction
    AI_ONLY = "ai_only"                     # Only AI analysis (no k-induction)
    COMBINED = "combined"                    # Both attempted, best result used


@dataclass
class AIInvariant:
    """An invariant derived from abstract interpretation."""
    variable: str
    expression: str     # e.g., "x >= 0", "x <= 10"
    source: str         # "interval_lower", "interval_upper", "sign"
    interval: Optional[Interval] = None
    sign: Optional[Sign] = None


@dataclass
class AIKIndResult:
    """Result of AI-strengthened k-induction."""
    verdict: AIKIndVerdict
    method: AIKIndMethod
    source: str
    property_desc: str
    ai_invariants: List[AIInvariant]
    ai_result: Optional[VerifiedAIResult]
    ai_certificate: Optional[ProofCertificate]
    kind_certificate: Optional[KIndCertificate]
    combined_certificate: Optional[ProofCertificate]
    k_value: Optional[int]
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        if self.combined_certificate:
            return self.combined_certificate.status == CertStatus.VALID
        if self.kind_certificate:
            return self.kind_certificate.status == CertStatus.VALID
        return False

    @property
    def total_obligations(self) -> int:
        total = 0
        if self.ai_certificate:
            total += len(self.ai_certificate.obligations)
        if self.kind_certificate:
            total += len(self.kind_certificate.obligations)
        return total

    @property
    def valid_obligations(self) -> int:
        count = 0
        if self.ai_certificate:
            count += sum(1 for o in self.ai_certificate.obligations if o.status == CertStatus.VALID)
        if self.kind_certificate:
            count += sum(1 for o in self.kind_certificate.obligations if o.status == CertStatus.VALID)
        return count

    def summary(self) -> str:
        lines = [
            f"AI-Strengthened k-Induction: {self.verdict.value}",
            f"  Method: {self.method.value}",
            f"  AI invariants: {len(self.ai_invariants)}",
        ]
        if self.k_value is not None:
            lines.append(f"  k value: {self.k_value}")
        if self.kind_certificate:
            lines.append(f"  k-induction: {self.kind_certificate.result}")
        lines.append(f"  Obligations: {self.valid_obligations}/{self.total_obligations} valid")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "method": self.method.value,
            "property": self.property_desc,
            "ai_invariants": [
                {"variable": inv.variable, "expression": inv.expression, "source": inv.source}
                for inv in self.ai_invariants
            ],
            "k_value": self.k_value,
            "certified": self.certified,
            "total_obligations": self.total_obligations,
            "valid_obligations": self.valid_obligations,
            "errors": self.errors,
        }


def _extract_loop_variables(source: str) -> List[str]:
    """Extract variables that appear in while loops."""
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        variables = set()
        _collect_vars_from_stmts(program.stmts, variables)
        return sorted(variables)
    except Exception:
        return []


def _collect_vars_from_stmts(stmts, variables):
    """Recursively collect variable names from statements."""
    from stack_vm import Assign, LetDecl, IfStmt, WhileStmt, Block
    for stmt in stmts:
        if isinstance(stmt, LetDecl):
            variables.add(stmt.name)
        elif isinstance(stmt, Assign):
            variables.add(stmt.name)
        elif isinstance(stmt, IfStmt):
            body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else stmt.then_body
            _collect_vars_from_stmts(body, variables)
            if stmt.else_body:
                ebody = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else stmt.else_body
                _collect_vars_from_stmts(ebody, variables)
        elif isinstance(stmt, WhileStmt):
            body = stmt.body.stmts if isinstance(stmt.body, Block) else stmt.body
            _collect_vars_from_stmts(body, variables)


def _extract_ai_invariants(source: str, max_iterations: int = 50) -> List[AIInvariant]:
    """Run abstract interpretation and extract invariants from results."""
    invariants = []
    try:
        result = ai_analyze(source)
        if not result:
            return invariants

        # ai_analyze returns a dict with 'env' key
        env = result['env'] if isinstance(result, dict) else getattr(result, 'env', None)
        if not env:
            return invariants

        # Extract interval bounds (env.intervals is a dict: var_name -> Interval)
        if hasattr(env, 'intervals'):
            for var_name in sorted(env.intervals.keys()):
                iv = env.intervals[var_name]
                if iv and iv != INTERVAL_TOP and iv != INTERVAL_BOT:
                    if iv.lo is not None:
                        lo = int(iv.lo) if iv.lo == int(iv.lo) else iv.lo
                        invariants.append(AIInvariant(
                            variable=var_name,
                            expression=f"{var_name} >= {lo}",
                            source="interval_lower",
                            interval=iv,
                        ))
                    if iv.hi is not None:
                        hi = int(iv.hi) if iv.hi == int(iv.hi) else iv.hi
                        invariants.append(AIInvariant(
                            variable=var_name,
                            expression=f"{var_name} <= {hi}",
                            source="interval_upper",
                            interval=iv,
                        ))

        # Extract sign info (env.signs is a dict: var_name -> Sign)
        if hasattr(env, 'signs'):
            for var_name in sorted(env.signs.keys()):
                sign = env.signs[var_name]
                if sign == Sign.POS:
                    invariants.append(AIInvariant(
                        variable=var_name,
                        expression=f"{var_name} > 0",
                        source="sign",
                        sign=sign,
                    ))
                elif sign == Sign.NEG:
                    invariants.append(AIInvariant(
                        variable=var_name,
                        expression=f"{var_name} < 0",
                        source="sign",
                        sign=sign,
                    ))
                elif sign == Sign.ZERO:
                    invariants.append(AIInvariant(
                        variable=var_name,
                        expression=f"{var_name} == 0",
                        source="sign",
                        sign=sign,
                    ))
                elif sign == Sign.NON_NEG:
                    invariants.append(AIInvariant(
                        variable=var_name,
                        expression=f"{var_name} >= 0",
                        source="sign",
                        sign=sign,
                    ))
                elif sign == Sign.NON_POS:
                    invariants.append(AIInvariant(
                        variable=var_name,
                        expression=f"{var_name} <= 0",
                        source="sign",
                        sign=sign,
                    ))
    except Exception:
        pass

    return invariants


def _invariants_to_sources(invariants: List[AIInvariant]) -> List[str]:
    """Convert AI invariants to source-level property strings for k-induction."""
    return [inv.expression for inv in invariants]


def certify_ai_kind(source: str, property_source: str,
                     max_iterations: int = 50, max_k: int = 20) -> AIKIndResult:
    """Full AI-strengthened k-induction pipeline.

    1. Run abstract interpretation to extract invariants
    2. Use invariants to strengthen k-induction
    3. Combine certificates
    """
    t0 = time.time()
    errors = []

    # Phase 1: Abstract interpretation for invariants
    ai_result = None
    ai_cert = None
    invariants = []
    try:
        ai_result_tuple = certify_abstract_interpretation(source, max_iterations=max_iterations)
        ai_result, ai_cert = ai_result_tuple
        invariants = _extract_ai_invariants(source, max_iterations)
    except Exception as e:
        errors.append(f"AI analysis failed: {e}")

    # Phase 2: k-Induction with AI invariants
    kind_cert = None
    inv_sources = _invariants_to_sources(invariants)

    if inv_sources:
        # Try strengthened k-induction with AI invariants
        try:
            kind_cert = certify_loop_with_invariants(
                source, property_source, inv_sources, max_k=max_k
            )
        except Exception as e:
            errors.append(f"Strengthened k-induction failed: {e}")
            # Fallback to plain k-induction
            try:
                kind_cert = certify_loop(source, property_source, max_k=max_k)
            except Exception as e2:
                errors.append(f"Plain k-induction also failed: {e2}")
    else:
        # No invariants from AI, try plain k-induction
        try:
            kind_cert = certify_loop(source, property_source, max_k=max_k)
        except Exception as e:
            errors.append(f"Plain k-induction failed: {e}")

    # Phase 3: Combine certificates
    combined_cert = None
    if ai_cert and kind_cert:
        try:
            kind_v044 = to_v044_certificate(kind_cert)
            combined_cert = combine_certificates([ai_cert, kind_v044])
        except Exception as e:
            errors.append(f"Certificate combination failed: {e}")

    # Determine verdict and method
    if kind_cert and kind_cert.result == "safe":
        verdict = AIKIndVerdict.SAFE
        method = AIKIndMethod.AI_STRENGTHENED if inv_sources else AIKIndMethod.BASIC_KIND
    elif kind_cert and kind_cert.result == "unsafe":
        verdict = AIKIndVerdict.UNSAFE
        method = AIKIndMethod.AI_STRENGTHENED if inv_sources else AIKIndMethod.BASIC_KIND
    elif ai_result and not kind_cert:
        verdict = AIKIndVerdict.AI_ONLY
        method = AIKIndMethod.AI_ONLY
    else:
        verdict = AIKIndVerdict.UNKNOWN
        method = AIKIndMethod.COMBINED if inv_sources else AIKIndMethod.BASIC_KIND

    elapsed = time.time() - t0

    return AIKIndResult(
        verdict=verdict,
        method=method,
        source=source,
        property_desc=property_source,
        ai_invariants=invariants,
        ai_result=ai_result,
        ai_certificate=ai_cert,
        kind_certificate=kind_cert,
        combined_certificate=combined_cert,
        k_value=kind_cert.k if kind_cert else None,
        errors=errors,
        metadata={"elapsed": elapsed, "invariant_count": len(invariants)},
    )


def certify_ai_kind_basic(source: str, property_source: str,
                            max_k: int = 20) -> AIKIndResult:
    """Simplified version: skip AI, just run k-induction for comparison."""
    t0 = time.time()
    errors = []
    kind_cert = None

    try:
        kind_cert = certify_loop(source, property_source, max_k=max_k)
    except Exception as e:
        errors.append(f"k-induction failed: {e}")

    if kind_cert and kind_cert.result == "safe":
        verdict = AIKIndVerdict.SAFE
    elif kind_cert and kind_cert.result == "unsafe":
        verdict = AIKIndVerdict.UNSAFE
    else:
        verdict = AIKIndVerdict.UNKNOWN

    return AIKIndResult(
        verdict=verdict,
        method=AIKIndMethod.BASIC_KIND,
        source=source,
        property_desc=property_source,
        ai_invariants=[],
        ai_result=None,
        ai_certificate=None,
        kind_certificate=kind_cert,
        combined_certificate=None,
        k_value=kind_cert.k if kind_cert else None,
        errors=errors,
        metadata={"elapsed": time.time() - t0},
    )


def analyze_ai_invariants(source: str, max_iterations: int = 50) -> Dict:
    """Just run AI analysis and return invariants without k-induction."""
    invariants = _extract_ai_invariants(source, max_iterations)
    return {
        "source": source,
        "invariant_count": len(invariants),
        "invariants": [
            {"variable": inv.variable, "expression": inv.expression, "source": inv.source}
            for inv in invariants
        ],
        "variables": sorted(set(inv.variable for inv in invariants)),
    }


def compare_basic_vs_ai(source: str, property_source: str,
                          max_iterations: int = 50,
                          max_k: int = 20) -> Dict:
    """Compare plain k-induction vs AI-strengthened k-induction."""
    t0 = time.time()
    basic = certify_ai_kind_basic(source, property_source, max_k=max_k)
    t_basic = time.time() - t0

    t1 = time.time()
    ai_str = certify_ai_kind(source, property_source,
                              max_iterations=max_iterations, max_k=max_k)
    t_ai = time.time() - t1

    return {
        "basic": {
            "verdict": basic.verdict.value,
            "k_value": basic.k_value,
            "time": t_basic,
            "obligations": basic.total_obligations,
        },
        "ai_strengthened": {
            "verdict": ai_str.verdict.value,
            "k_value": ai_str.k_value,
            "invariants_used": len(ai_str.ai_invariants),
            "time": t_ai,
            "obligations": ai_str.total_obligations,
        },
        "ai_helped": (basic.verdict != AIKIndVerdict.SAFE and
                      ai_str.verdict == AIKIndVerdict.SAFE),
    }


def ai_kind_summary(result: AIKIndResult) -> Dict:
    """Generate summary dict for an AI-strengthened k-induction result."""
    return {
        "verdict": result.verdict.value,
        "method": result.method.value,
        "property": result.property_desc,
        "ai_invariants": len(result.ai_invariants),
        "k_value": result.k_value,
        "certified": result.certified,
        "total_obligations": result.total_obligations,
        "valid_obligations": result.valid_obligations,
        "errors": result.errors,
    }
