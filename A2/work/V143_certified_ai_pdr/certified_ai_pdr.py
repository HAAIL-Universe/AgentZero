"""V143: Certified AI-Strengthened PDR

Composes V046 (certified abstract interpretation) + V137 (certified PDR)
to use abstract-interpretation-derived invariants to strengthen PDR proofs.

Key insight: PDR can struggle with convergence on complex transition systems.
Abstract interpretation cheaply computes invariants (interval bounds, signs) that
can be conjoined with the property, giving PDR tighter constraints to work with
and helping it find inductive invariants faster.

Pipeline:
1. Run V046 certified abstract interpretation on the program
2. Extract variable bounds as candidate invariants
3. Conjoin invariants with property for strengthened PDR
4. Combine certificates: AI soundness + PDR validity = full proof
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
    os.path.join(_work, "V137_certified_pdr"),
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
from certified_pdr import (
    PDRCertificate, PDRCertKind,
    certify_pdr, certify_and_check_pdr, certify_pdr_loop,
    check_pdr_certificate, to_v044_certificate,
    compare_pdr_vs_kind, certify_combined,
    pdr_certificate_summary,
    _parse_expr_to_smt,
)
from proof_certificates import (
    ProofKind, CertStatus, ProofObligation, ProofCertificate,
    combine_certificates,
)
from abstract_interpreter import (
    Interval, INTERVAL_TOP, INTERVAL_BOT, Sign,
    analyze as ai_analyze, get_variable_range, get_variable_sign,
)
from smt_solver import Op, App, IntConst, BoolConst, Var as SMTVar, INT, BOOL
from stack_vm import lex, Parser, LetDecl, WhileStmt, FnDecl


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class AIPDRVerdict(Enum):
    SAFE = "safe"           # Property proven with AI-strengthened PDR
    UNSAFE = "unsafe"       # Counterexample found
    UNKNOWN = "unknown"     # Could not prove or disprove
    AI_ONLY = "ai_only"     # AI analysis completed but PDR not attempted


class AIPDRMethod(Enum):
    BASIC_PDR = "basic_pdr"             # Plain PDR (no AI)
    AI_STRENGTHENED = "ai_strengthened"  # AI invariants + PDR
    AI_ONLY = "ai_only"                 # Only AI analysis (no PDR)
    COMBINED = "combined"                # Both attempted, best result used


@dataclass
class AIInvariant:
    """An invariant derived from abstract interpretation."""
    variable: str
    expression: str     # e.g., "x >= 0", "x <= 10"
    source: str         # "interval_lower", "interval_upper", "sign"
    interval: Optional[Interval] = None
    sign: Optional[Sign] = None


@dataclass
class AIPDRResult:
    """Result of AI-strengthened PDR verification."""
    verdict: AIPDRVerdict
    method: AIPDRMethod
    source: str
    property_desc: str
    ai_invariants: List[AIInvariant]
    ai_result: Optional[VerifiedAIResult]
    ai_certificate: Optional[ProofCertificate]
    pdr_certificate: Optional[PDRCertificate]
    combined_certificate: Optional[ProofCertificate]
    num_frames: Optional[int]
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        if self.combined_certificate:
            return self.combined_certificate.status == CertStatus.VALID
        if self.pdr_certificate:
            return self.pdr_certificate.status == CertStatus.VALID
        return False

    @property
    def total_obligations(self) -> int:
        total = 0
        if self.ai_certificate:
            total += len(self.ai_certificate.obligations)
        if self.pdr_certificate:
            total += len(self.pdr_certificate.obligations)
        return total

    @property
    def valid_obligations(self) -> int:
        count = 0
        if self.ai_certificate:
            count += sum(1 for o in self.ai_certificate.obligations if o.status == CertStatus.VALID)
        if self.pdr_certificate:
            count += sum(1 for o in self.pdr_certificate.obligations if o.status == CertStatus.VALID)
        return count

    def summary(self) -> str:
        lines = [
            f"AI-Strengthened PDR: {self.verdict.value}",
            f"  Method: {self.method.value}",
            f"  AI invariants: {len(self.ai_invariants)}",
        ]
        if self.num_frames is not None:
            lines.append(f"  Frames: {self.num_frames}")
        if self.pdr_certificate:
            lines.append(f"  PDR result: {self.pdr_certificate.result}")
            lines.append(f"  Invariant clauses: {len(self.pdr_certificate.invariant_clauses)}")
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
            "num_frames": self.num_frames,
            "certified": self.certified,
            "total_obligations": self.total_obligations,
            "valid_obligations": self.valid_obligations,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# AI invariant extraction (shared with V141)
# ---------------------------------------------------------------------------

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

        env = result['env'] if isinstance(result, dict) else getattr(result, 'env', None)
        if not env:
            return invariants

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

        if hasattr(env, 'signs'):
            for var_name in sorted(env.signs.keys()):
                sign = env.signs[var_name]
                if sign == Sign.POS:
                    invariants.append(AIInvariant(
                        variable=var_name, expression=f"{var_name} > 0",
                        source="sign", sign=sign))
                elif sign == Sign.NEG:
                    invariants.append(AIInvariant(
                        variable=var_name, expression=f"{var_name} < 0",
                        source="sign", sign=sign))
                elif sign == Sign.ZERO:
                    invariants.append(AIInvariant(
                        variable=var_name, expression=f"{var_name} == 0",
                        source="sign", sign=sign))
                elif sign == Sign.NON_NEG:
                    invariants.append(AIInvariant(
                        variable=var_name, expression=f"{var_name} >= 0",
                        source="sign", sign=sign))
                elif sign == Sign.NON_POS:
                    invariants.append(AIInvariant(
                        variable=var_name, expression=f"{var_name} <= 0",
                        source="sign", sign=sign))
    except Exception:
        pass

    return invariants


def _invariants_to_sources(invariants: List[AIInvariant]) -> List[str]:
    """Convert AI invariants to source-level property strings."""
    return [inv.expression for inv in invariants]


# ---------------------------------------------------------------------------
# Strengthened PDR (conjoins AI invariants with property)
# ---------------------------------------------------------------------------

def _extract_init_values(source: str) -> Dict[str, int]:
    """Extract initial variable values from let declarations before while loops."""
    try:
        tokens = lex(source)
        program = Parser(tokens).parse()
        init_vals = {}
        for stmt in program.stmts:
            if isinstance(stmt, LetDecl):
                # Try to evaluate simple integer literals
                from stack_vm import IntLit, BinOp as ASTBinOp
                if hasattr(stmt, 'value') and isinstance(stmt.value, IntLit):
                    init_vals[stmt.name] = stmt.value.value
            elif isinstance(stmt, WhileStmt):
                break  # Stop at first while loop
        return init_vals
    except Exception:
        return {}


def _eval_invariant_at_init(expression: str, init_vals: Dict[str, int]) -> bool:
    """Check if an invariant expression holds given initial variable values.

    Simple evaluation of expressions like 'x >= 0', 'i <= 10'.
    """
    import re
    # Parse simple comparison: var op value
    m = re.match(r'(\w+)\s*(>=|<=|>|<|==|!=)\s*(-?\d+)', expression)
    if not m:
        return False  # Can't evaluate, skip it

    var_name, op, val_str = m.groups()
    val = int(val_str)

    if var_name not in init_vals:
        return False  # Unknown variable, skip

    init_val = init_vals[var_name]
    ops = {
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '<': lambda a, b: a < b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }
    return ops[op](init_val, val)


def _filter_init_safe_invariants(source: str,
                                   invariants: List[AIInvariant]) -> List[AIInvariant]:
    """Filter AI invariants to only those holding at the initial state.

    Post-loop invariants (e.g., i >= 5 after while(i<5) with i=0) are discarded
    since they would cause PDR to find false counterexamples at the initial state.
    """
    init_vals = _extract_init_values(source)
    if not init_vals:
        return invariants  # Can't evaluate, keep all

    safe = []
    for inv in invariants:
        if _eval_invariant_at_init(inv.expression, init_vals):
            safe.append(inv)
    return safe


def certify_pdr_loop_with_invariants(source, property_source, invariant_sources,
                                      max_frames=100):
    """Source-level certified PDR with AI invariant strengthening.

    Only conjoins invariants that hold at the initial state (init-safe invariants).
    Post-loop invariants (like i >= 5 after while(i<5)) are filtered out since
    they would cause false counterexamples in PDR.
    """
    from k_induction import _extract_loop_ts

    ts, ts_vars = _extract_loop_ts(source)

    # Parse the main property
    prop_smt = _parse_expr_to_smt(property_source, ts_vars)

    # Parse each invariant and filter to init-safe ones
    init_vals = _extract_init_values(source)
    init_safe_terms = []
    for inv_src in invariant_sources:
        try:
            # Check if invariant holds at initial state
            if init_vals and not _eval_invariant_at_init(inv_src, init_vals):
                continue  # Skip post-loop invariants

            inv_term = _parse_expr_to_smt(inv_src, ts_vars)
            init_safe_terms.append(inv_term)
        except Exception:
            continue  # Skip unparseable invariants

    # Strengthened property: prop AND inv1 AND inv2 AND ... (init-safe only)
    if init_safe_terms:
        strengthened = prop_smt
        for inv_term in init_safe_terms:
            strengthened = App(Op.AND, [strengthened, inv_term], BOOL)
        ts.set_property(strengthened)
    else:
        ts.set_property(prop_smt)

    return certify_and_check_pdr(ts, max_frames)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def certify_ai_pdr(source: str, property_source: str,
                    max_iterations: int = 50, max_frames: int = 100) -> AIPDRResult:
    """Full AI-strengthened PDR pipeline.

    1. Run abstract interpretation to extract invariants
    2. Conjoin invariants with property for strengthened PDR
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

    # Phase 2: PDR with AI invariants
    pdr_cert = None
    inv_sources = _invariants_to_sources(invariants)

    if inv_sources:
        # Try strengthened PDR with AI invariants
        try:
            pdr_cert = certify_pdr_loop_with_invariants(
                source, property_source, inv_sources, max_frames=max_frames
            )
        except Exception as e:
            errors.append(f"Strengthened PDR failed: {e}")
            # Fallback to plain PDR
            try:
                pdr_cert = certify_pdr_loop(source, property_source, max_frames=max_frames)
            except Exception as e2:
                errors.append(f"Plain PDR also failed: {e2}")
    else:
        # No invariants from AI, try plain PDR
        try:
            pdr_cert = certify_pdr_loop(source, property_source, max_frames=max_frames)
        except Exception as e:
            errors.append(f"Plain PDR failed: {e}")

    # Phase 3: Combine certificates
    combined_cert = None
    if ai_cert and pdr_cert:
        try:
            pdr_v044 = to_v044_certificate(pdr_cert)
            combined_cert = combine_certificates([ai_cert, pdr_v044])
        except Exception as e:
            errors.append(f"Certificate combination failed: {e}")

    # Determine verdict and method
    if pdr_cert and pdr_cert.result == "safe":
        verdict = AIPDRVerdict.SAFE
        method = AIPDRMethod.AI_STRENGTHENED if inv_sources else AIPDRMethod.BASIC_PDR
    elif pdr_cert and pdr_cert.result == "unsafe":
        verdict = AIPDRVerdict.UNSAFE
        method = AIPDRMethod.AI_STRENGTHENED if inv_sources else AIPDRMethod.BASIC_PDR
    elif ai_result and not pdr_cert:
        verdict = AIPDRVerdict.AI_ONLY
        method = AIPDRMethod.AI_ONLY
    else:
        verdict = AIPDRVerdict.UNKNOWN
        method = AIPDRMethod.COMBINED if inv_sources else AIPDRMethod.BASIC_PDR

    elapsed = time.time() - t0

    return AIPDRResult(
        verdict=verdict,
        method=method,
        source=source,
        property_desc=property_source,
        ai_invariants=invariants,
        ai_result=ai_result,
        ai_certificate=ai_cert,
        pdr_certificate=pdr_cert,
        combined_certificate=combined_cert,
        num_frames=pdr_cert.metadata.get("num_frames") if pdr_cert else None,
        errors=errors,
        metadata={"elapsed": elapsed, "invariant_count": len(invariants)},
    )


def certify_ai_pdr_basic(source: str, property_source: str,
                           max_frames: int = 100) -> AIPDRResult:
    """Plain PDR without AI (baseline for comparison)."""
    t0 = time.time()
    errors = []
    pdr_cert = None

    try:
        pdr_cert = certify_pdr_loop(source, property_source, max_frames=max_frames)
    except Exception as e:
        errors.append(f"PDR failed: {e}")

    if pdr_cert and pdr_cert.result == "safe":
        verdict = AIPDRVerdict.SAFE
    elif pdr_cert and pdr_cert.result == "unsafe":
        verdict = AIPDRVerdict.UNSAFE
    else:
        verdict = AIPDRVerdict.UNKNOWN

    return AIPDRResult(
        verdict=verdict,
        method=AIPDRMethod.BASIC_PDR,
        source=source,
        property_desc=property_source,
        ai_invariants=[],
        ai_result=None,
        ai_certificate=None,
        pdr_certificate=pdr_cert,
        combined_certificate=None,
        num_frames=pdr_cert.metadata.get("num_frames") if pdr_cert else None,
        errors=errors,
        metadata={"elapsed": time.time() - t0},
    )


def analyze_ai_invariants(source: str, max_iterations: int = 50) -> Dict:
    """Run AI analysis and return invariants without PDR."""
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
                          max_frames: int = 100) -> Dict:
    """Compare plain PDR vs AI-strengthened PDR."""
    t0 = time.time()
    basic = certify_ai_pdr_basic(source, property_source, max_frames=max_frames)
    t_basic = time.time() - t0

    t1 = time.time()
    ai_str = certify_ai_pdr(source, property_source,
                              max_iterations=max_iterations, max_frames=max_frames)
    t_ai = time.time() - t1

    return {
        "basic": {
            "verdict": basic.verdict.value,
            "num_frames": basic.num_frames,
            "time": t_basic,
            "obligations": basic.total_obligations,
        },
        "ai_strengthened": {
            "verdict": ai_str.verdict.value,
            "num_frames": ai_str.num_frames,
            "invariants_used": len(ai_str.ai_invariants),
            "time": t_ai,
            "obligations": ai_str.total_obligations,
        },
        "ai_helped": (basic.verdict != AIPDRVerdict.SAFE and
                      ai_str.verdict == AIPDRVerdict.SAFE),
    }


def compare_pdr_vs_kind_ai(source: str, property_source: str,
                              max_iterations: int = 50,
                              max_frames: int = 100, max_k: int = 20) -> Dict:
    """Compare AI-strengthened PDR vs AI-strengthened k-induction.

    Both methods use the same AI invariants but different proof backends.
    """
    # Import V141 for k-induction comparison
    try:
        v141_dir = os.path.join(_work, "V141_certified_ai_composition")
        if v141_dir not in sys.path:
            sys.path.insert(0, v141_dir)
        from certified_ai_composition import certify_ai_kind
    except ImportError:
        return {"error": "V141 not available for comparison"}

    t0 = time.time()
    pdr_result = certify_ai_pdr(source, property_source,
                                  max_iterations=max_iterations, max_frames=max_frames)
    pdr_time = time.time() - t0

    t1 = time.time()
    kind_result = certify_ai_kind(source, property_source,
                                    max_iterations=max_iterations, max_k=max_k)
    kind_time = time.time() - t1

    return {
        "ai_pdr": {
            "verdict": pdr_result.verdict.value,
            "num_frames": pdr_result.num_frames,
            "invariants_used": len(pdr_result.ai_invariants),
            "time": pdr_time,
            "obligations": pdr_result.total_obligations,
            "certified": pdr_result.certified,
        },
        "ai_kind": {
            "verdict": kind_result.verdict.value,
            "k_value": kind_result.k_value,
            "invariants_used": len(kind_result.ai_invariants),
            "time": kind_time,
            "obligations": kind_result.total_obligations,
            "certified": kind_result.certified,
        },
        "same_verdict": pdr_result.verdict.value == kind_result.verdict.value,
        "same_invariants": len(pdr_result.ai_invariants) == len(kind_result.ai_invariants),
    }


def ai_pdr_summary(result: AIPDRResult) -> Dict:
    """Generate summary dict for an AI-strengthened PDR result."""
    return {
        "verdict": result.verdict.value,
        "method": result.method.value,
        "property": result.property_desc,
        "ai_invariants": len(result.ai_invariants),
        "num_frames": result.num_frames,
        "certified": result.certified,
        "total_obligations": result.total_obligations,
        "valid_obligations": result.valid_obligations,
        "errors": result.errors,
    }
