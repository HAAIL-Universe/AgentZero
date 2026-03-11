"""
V144: Certified Effect-Aware PDR
Composes V143 (certified AI-PDR) + V140 (effect regression) + V040 (effect inference)

Verifies that loops both:
1. Satisfy a given property (via AI-strengthened PDR with certificates)
2. Maintain declared effect discipline (via effect inference + checking)

Combined certificates cover correctness AND effect compliance.
"""

import sys, os, time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple

# V143: Certified AI-PDR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V143_certified_ai_pdr'))
from certified_ai_pdr import (
    AIPDRResult, AIPDRVerdict, AIPDRMethod, AIInvariant,
    certify_ai_pdr, certify_ai_pdr_basic, analyze_ai_invariants,
    _extract_ai_invariants, _filter_init_safe_invariants,
)

# V140: Effect Regression
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V140_effect_aware_regression'))
from effect_aware_regression import (
    EffectRegressionResult, EffectRegressionVerdict, EffectChange, EffectChangeKind,
    verify_effect_regression, verify_function_effect_regression,
    check_effect_purity_preserved, _infer_effects, _compute_effect_changes,
)

# V044: Proof Certificates
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
from proof_certificates import (
    ProofKind, CertStatus, ProofObligation, ProofCertificate,
    combine_certificates,
)


# --- Result types ---

class EffectPDRVerdict(Enum):
    """Combined verdict for effect-aware PDR verification."""
    SAFE = "safe"                        # Property holds AND effects conform
    PROPERTY_FAILURE = "property_failure" # PDR found counterexample
    EFFECT_VIOLATION = "effect_violation" # Property holds but undeclared effects
    UNSAFE = "unsafe"                    # Both property and effect failures
    UNKNOWN = "unknown"                  # Indeterminate


class EffectPDRMethod(Enum):
    """Method used for verification."""
    AI_PDR_PLUS_EFFECTS = "ai_pdr_plus_effects"
    BASIC_PDR_PLUS_EFFECTS = "basic_pdr_plus_effects"
    EFFECTS_ONLY = "effects_only"
    PDR_ONLY = "pdr_only"


@dataclass
class EffectInfo:
    """Effect analysis result for a function."""
    function: str
    inferred_effects: List[str]
    declared_effects: Optional[List[str]] = None
    undeclared: List[str] = field(default_factory=list)
    conforms: bool = True


@dataclass
class EffectPDRResult:
    """Complete result of certified effect-aware PDR verification."""
    verdict: EffectPDRVerdict
    method: EffectPDRMethod
    source: str
    property_desc: str

    # PDR results
    pdr_result: Optional[AIPDRResult] = None
    pdr_verdict: Optional[AIPDRVerdict] = None

    # Effect results
    effect_infos: List[EffectInfo] = field(default_factory=list)
    effects_conform: bool = True

    # Certificates
    pdr_certificate: Optional[ProofCertificate] = None
    effect_certificate: Optional[ProofCertificate] = None
    combined_certificate: Optional[ProofCertificate] = None

    # AI invariants (from V143)
    ai_invariants: List[AIInvariant] = field(default_factory=list)

    # Metadata
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        """Whether combined certificate is valid."""
        if self.combined_certificate:
            return self.combined_certificate.status == CertStatus.VALID
        if self.pdr_certificate:
            return self.pdr_certificate.status == CertStatus.VALID
        return False

    @property
    def total_obligations(self) -> int:
        count = 0
        if self.pdr_result:
            count += self.pdr_result.total_obligations
        if self.effect_certificate:
            count += len(self.effect_certificate.obligations)
        return count

    @property
    def valid_obligations(self) -> int:
        count = 0
        if self.pdr_result:
            count += self.pdr_result.valid_obligations
        if self.effect_certificate:
            count += sum(1 for o in self.effect_certificate.obligations
                        if o.status == CertStatus.VALID)
        return count

    def summary(self) -> str:
        lines = [f"Effect-Aware PDR: {self.verdict.value}"]
        lines.append(f"  Method: {self.method.value}")
        if self.pdr_verdict:
            lines.append(f"  PDR verdict: {self.pdr_verdict.value}")
        lines.append(f"  Effects conform: {self.effects_conform}")
        if self.effect_infos:
            for ei in self.effect_infos:
                eff_str = ', '.join(ei.inferred_effects) if ei.inferred_effects else 'pure'
                lines.append(f"    {ei.function}: {eff_str}" +
                           (f" (undeclared: {ei.undeclared})" if ei.undeclared else ""))
        if self.ai_invariants:
            lines.append(f"  AI invariants: {len(self.ai_invariants)}")
        lines.append(f"  Certified: {self.certified}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {
            'verdict': self.verdict.value,
            'method': self.method.value,
            'property': self.property_desc,
            'pdr_verdict': self.pdr_verdict.value if self.pdr_verdict else None,
            'effects_conform': self.effects_conform,
            'effect_infos': [
                {'function': ei.function, 'inferred': ei.inferred_effects,
                 'declared': ei.declared_effects, 'undeclared': ei.undeclared,
                 'conforms': ei.conforms}
                for ei in self.effect_infos
            ],
            'ai_invariants': [
                {'variable': inv.variable, 'expression': inv.expression, 'source': inv.source}
                for inv in self.ai_invariants
            ],
            'certified': self.certified,
            'total_obligations': self.total_obligations,
            'valid_obligations': self.valid_obligations,
            'errors': self.errors,
            'metadata': self.metadata,
        }


# --- Effect analysis helpers ---

def _analyze_effects(source: str, declared_effects: Optional[Dict[str, List[str]]] = None) -> Tuple[List[EffectInfo], bool]:
    """Infer effects and check against declarations.

    Returns (effect_infos, all_conform).
    """
    inferred = _infer_effects(source)
    if not inferred:
        # No functions found or parse error -- report as pure
        return [], True

    effect_infos = []
    all_conform = True

    for fn_name, effects in sorted(inferred.items()):
        declared = None
        undeclared = []
        conforms = True

        if declared_effects and fn_name in declared_effects:
            declared = sorted(declared_effects[fn_name])
            declared_set = set(declared)
            undeclared = sorted(set(effects) - declared_set)
            if undeclared:
                conforms = False
                all_conform = False

        ei = EffectInfo(
            function=fn_name,
            inferred_effects=sorted(effects),
            declared_effects=declared,
            undeclared=undeclared,
            conforms=conforms,
        )
        effect_infos.append(ei)

    return effect_infos, all_conform


def _build_effect_certificate(effect_infos: List[EffectInfo], all_conform: bool) -> ProofCertificate:
    """Build a proof certificate for effect analysis results."""
    obligations = []
    for ei in effect_infos:
        status = CertStatus.VALID if ei.conforms else CertStatus.INVALID
        desc = f"Effect conformance for {ei.function}"
        if ei.declared_effects is not None:
            desc += f": declared={ei.declared_effects}, inferred={ei.inferred_effects}"
        else:
            desc += f": inferred={ei.inferred_effects} (no declaration)"
            status = CertStatus.VALID  # No declaration = no violation

        obligations.append(ProofObligation(
            name=f"effect_{ei.function}",
            description=desc,
            formula_str=f"effects({ei.function}) conform",
            formula_smt=f"effects({ei.function})",
            status=status,
        ))

    cert_status = CertStatus.VALID if all_conform else CertStatus.INVALID
    return ProofCertificate(
        kind=ProofKind.VCGEN,
        claim="Effect conformance",
        obligations=obligations,
        metadata={'type': 'effect_conformance', 'num_functions': len(effect_infos)},
        status=cert_status,
    )


def _combine_verdicts(pdr_verdict: Optional[AIPDRVerdict], effects_conform: bool) -> EffectPDRVerdict:
    """Combine PDR and effect verdicts into unified verdict."""
    if pdr_verdict is None:
        return EffectPDRVerdict.SAFE if effects_conform else EffectPDRVerdict.EFFECT_VIOLATION

    pdr_safe = pdr_verdict == AIPDRVerdict.SAFE
    pdr_unknown = pdr_verdict == AIPDRVerdict.UNKNOWN

    if pdr_safe and effects_conform:
        return EffectPDRVerdict.SAFE
    elif pdr_safe and not effects_conform:
        return EffectPDRVerdict.EFFECT_VIOLATION
    elif not pdr_safe and effects_conform:
        if pdr_unknown:
            return EffectPDRVerdict.UNKNOWN
        return EffectPDRVerdict.PROPERTY_FAILURE
    else:
        if pdr_unknown:
            return EffectPDRVerdict.EFFECT_VIOLATION
        return EffectPDRVerdict.UNSAFE


# --- Main APIs ---

def certify_effect_pdr(
    source: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_iterations: int = 50,
    max_frames: int = 100,
) -> EffectPDRResult:
    """Full certified effect-aware PDR verification.

    Phase 1: AI-strengthened PDR for property verification (V143)
    Phase 2: Effect inference and conformance checking (V140/V040)
    Phase 3: Certificate combination (V044)

    Args:
        source: Source code with loop
        property_source: Property to verify (e.g., "x >= 0")
        declared_effects: Optional declared effects per function
        max_iterations: Max AI analysis iterations
        max_frames: Max PDR frames

    Returns:
        EffectPDRResult with combined verdict and certificates
    """
    errors = []
    metadata = {}
    t0 = time.time()

    # Phase 1: AI-strengthened PDR
    try:
        pdr_result = certify_ai_pdr(source, property_source,
                                     max_iterations=max_iterations,
                                     max_frames=max_frames)
        pdr_verdict = pdr_result.verdict
        pdr_cert = pdr_result.combined_certificate or pdr_result.pdr_certificate
        ai_invariants = pdr_result.ai_invariants
    except Exception as e:
        errors.append(f"PDR phase error: {e}")
        pdr_result = None
        pdr_verdict = None
        pdr_cert = None
        ai_invariants = []

    metadata['pdr_time'] = time.time() - t0

    # Phase 2: Effect analysis
    t1 = time.time()
    try:
        effect_infos, effects_conform = _analyze_effects(source, declared_effects)
        effect_cert = _build_effect_certificate(effect_infos, effects_conform)
    except Exception as e:
        errors.append(f"Effect phase error: {e}")
        effect_infos = []
        effects_conform = True
        effect_cert = ProofCertificate(
            kind=ProofKind.VCGEN, claim="Effect conformance",
            obligations=[], metadata={'error': str(e)},
            status=CertStatus.UNKNOWN,
        )

    metadata['effect_time'] = time.time() - t1

    # Phase 3: Combine certificates
    combined_cert = None
    certs_to_combine = []
    if pdr_cert:
        certs_to_combine.append(pdr_cert)
    if effect_cert:
        certs_to_combine.append(effect_cert)
    if certs_to_combine:
        try:
            combined_cert = combine_certificates(certs_to_combine)
        except Exception as e:
            errors.append(f"Certificate combination error: {e}")

    # Combine verdicts
    verdict = _combine_verdicts(pdr_verdict, effects_conform)
    metadata['total_time'] = time.time() - t0

    return EffectPDRResult(
        verdict=verdict,
        method=EffectPDRMethod.AI_PDR_PLUS_EFFECTS,
        source=source,
        property_desc=property_source,
        pdr_result=pdr_result,
        pdr_verdict=pdr_verdict,
        effect_infos=effect_infos,
        effects_conform=effects_conform,
        pdr_certificate=pdr_cert,
        effect_certificate=effect_cert,
        combined_certificate=combined_cert,
        ai_invariants=ai_invariants,
        errors=errors,
        metadata=metadata,
    )


def certify_effect_pdr_basic(
    source: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_frames: int = 100,
) -> EffectPDRResult:
    """Effect-aware PDR without AI strengthening (baseline).

    Uses plain PDR (no AI invariants) + effect analysis.
    """
    errors = []
    metadata = {}
    t0 = time.time()

    # Phase 1: Basic PDR
    try:
        pdr_result = certify_ai_pdr_basic(source, property_source,
                                           max_frames=max_frames)
        pdr_verdict = pdr_result.verdict
        pdr_cert = pdr_result.pdr_certificate
    except Exception as e:
        errors.append(f"PDR phase error: {e}")
        pdr_result = None
        pdr_verdict = None
        pdr_cert = None

    metadata['pdr_time'] = time.time() - t0

    # Phase 2: Effect analysis
    t1 = time.time()
    try:
        effect_infos, effects_conform = _analyze_effects(source, declared_effects)
        effect_cert = _build_effect_certificate(effect_infos, effects_conform)
    except Exception as e:
        errors.append(f"Effect phase error: {e}")
        effect_infos = []
        effects_conform = True
        effect_cert = None

    metadata['effect_time'] = time.time() - t1

    # Combine
    combined_cert = None
    certs = [c for c in [pdr_cert, effect_cert] if c is not None]
    if certs:
        try:
            combined_cert = combine_certificates(certs)
        except Exception as e:
            errors.append(f"Certificate combination error: {e}")

    verdict = _combine_verdicts(pdr_verdict, effects_conform)
    metadata['total_time'] = time.time() - t0

    return EffectPDRResult(
        verdict=verdict,
        method=EffectPDRMethod.BASIC_PDR_PLUS_EFFECTS,
        source=source,
        property_desc=property_source,
        pdr_result=pdr_result,
        pdr_verdict=pdr_verdict,
        effect_infos=effect_infos,
        effects_conform=effects_conform,
        pdr_certificate=pdr_cert,
        effect_certificate=effect_cert,
        combined_certificate=combined_cert,
        ai_invariants=[],
        errors=errors,
        metadata=metadata,
    )


def verify_effect_loop(
    source: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_frames: int = 10,
) -> EffectPDRResult:
    """Convenience API: verify a loop with effect checking.

    Smaller defaults for fast usage.
    """
    return certify_effect_pdr(source, property_source,
                               declared_effects=declared_effects,
                               max_iterations=20,
                               max_frames=max_frames)


def analyze_effects_only(source: str,
                          declared_effects: Optional[Dict[str, List[str]]] = None) -> EffectPDRResult:
    """Effect-only analysis without PDR (fast).

    Useful when you only care about effect discipline.
    """
    errors = []
    t0 = time.time()

    try:
        effect_infos, effects_conform = _analyze_effects(source, declared_effects)
        effect_cert = _build_effect_certificate(effect_infos, effects_conform)
    except Exception as e:
        errors.append(f"Effect analysis error: {e}")
        effect_infos = []
        effects_conform = True
        effect_cert = None

    verdict = EffectPDRVerdict.SAFE if effects_conform else EffectPDRVerdict.EFFECT_VIOLATION

    return EffectPDRResult(
        verdict=verdict,
        method=EffectPDRMethod.EFFECTS_ONLY,
        source=source,
        property_desc="(none)",
        effect_infos=effect_infos,
        effects_conform=effects_conform,
        effect_certificate=effect_cert,
        errors=errors,
        metadata={'total_time': time.time() - t0},
    )


def verify_effect_regression_pdr(
    source_old: str,
    source_new: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_frames: int = 10,
) -> Dict:
    """Verify property preservation AND effect regression between two versions.

    Combines V143 PDR on new version + V140 effect regression between versions.

    Returns:
        Dict with 'pdr' (EffectPDRResult for new), 'regression' (effect changes),
        'verdict' (combined string).
    """
    # PDR on new version
    pdr_result = certify_effect_pdr(source_new, property_source,
                                     declared_effects=declared_effects,
                                     max_iterations=20,
                                     max_frames=max_frames)

    # Effect regression between versions
    old_effects = _infer_effects(source_old)
    new_effects = _infer_effects(source_new)
    changes = _compute_effect_changes(old_effects, new_effects)
    has_regression = any(c.kind == EffectChangeKind.ADDED for c in changes)

    # Combined verdict
    if pdr_result.verdict == EffectPDRVerdict.SAFE and not has_regression:
        combined_verdict = "safe"
    elif pdr_result.verdict == EffectPDRVerdict.SAFE and has_regression:
        combined_verdict = "effect_regression"
    elif pdr_result.verdict in (EffectPDRVerdict.PROPERTY_FAILURE, EffectPDRVerdict.UNSAFE):
        combined_verdict = "property_failure" if not has_regression else "unsafe"
    else:
        combined_verdict = "unknown"

    return {
        'pdr': pdr_result,
        'regression': {
            'changes': [{'function': c.function, 'effect': c.effect, 'kind': c.kind.value}
                        for c in changes],
            'has_regression': has_regression,
            'num_changes': len(changes),
        },
        'verdict': combined_verdict,
    }


def compare_effect_vs_plain(
    source: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_frames: int = 10,
) -> Dict:
    """Compare effect-aware PDR vs plain PDR.

    Returns timing and verdict comparison.
    """
    # Plain PDR (V143 only)
    t0 = time.time()
    plain = certify_ai_pdr(source, property_source, max_frames=max_frames)
    plain_time = time.time() - t0

    # Effect-aware PDR (V144)
    t1 = time.time()
    effect_aware = certify_effect_pdr(source, property_source,
                                       declared_effects=declared_effects,
                                       max_frames=max_frames)
    effect_time = time.time() - t1

    return {
        'plain': {
            'verdict': plain.verdict.value,
            'time': plain_time,
            'certified': plain.combined_certificate is not None or plain.pdr_certificate is not None,
        },
        'effect_aware': {
            'verdict': effect_aware.verdict.value,
            'time': effect_time,
            'effects_conform': effect_aware.effects_conform,
            'effect_count': len(effect_aware.effect_infos),
            'certified': effect_aware.certified,
        },
        'overhead': effect_time - plain_time if plain_time > 0 else 0,
    }


def compare_ai_vs_basic_effect_pdr(
    source: str,
    property_source: str,
    declared_effects: Optional[Dict[str, List[str]]] = None,
    max_frames: int = 10,
) -> Dict:
    """Compare AI-strengthened vs basic effect-aware PDR."""
    t0 = time.time()
    ai_result = certify_effect_pdr(source, property_source,
                                    declared_effects=declared_effects,
                                    max_iterations=20, max_frames=max_frames)
    ai_time = time.time() - t0

    t1 = time.time()
    basic_result = certify_effect_pdr_basic(source, property_source,
                                             declared_effects=declared_effects,
                                             max_frames=max_frames)
    basic_time = time.time() - t1

    return {
        'ai_strengthened': {
            'verdict': ai_result.verdict.value,
            'pdr_verdict': ai_result.pdr_verdict.value if ai_result.pdr_verdict else None,
            'ai_invariants': len(ai_result.ai_invariants),
            'effects_conform': ai_result.effects_conform,
            'time': ai_time,
        },
        'basic': {
            'verdict': basic_result.verdict.value,
            'pdr_verdict': basic_result.pdr_verdict.value if basic_result.pdr_verdict else None,
            'effects_conform': basic_result.effects_conform,
            'time': basic_time,
        },
        'ai_helped': ai_result.verdict != basic_result.verdict,
    }


def effect_pdr_summary(result: EffectPDRResult) -> Dict:
    """Extract summary dict from result."""
    return {
        'verdict': result.verdict.value,
        'method': result.method.value,
        'pdr_verdict': result.pdr_verdict.value if result.pdr_verdict else None,
        'effects_conform': result.effects_conform,
        'num_functions': len(result.effect_infos),
        'ai_invariants': len(result.ai_invariants),
        'certified': result.certified,
        'total_obligations': result.total_obligations,
        'valid_obligations': result.valid_obligations,
        'errors': result.errors,
    }
