"""V140: Effect-Aware Regression Verification

Composes V138 (effect-aware verification) + V139 (certified regression)
to verify that code changes preserve effect properties.

Two-phase approach:
1. Run V138 effect analysis on both old and new versions
2. Use V139 certified regression to verify property preservation
3. Compare effect signatures: detect effect regressions (new effects introduced)

Key insight: a regression isn't just "does it compute the same thing?" but also
"does it maintain the same effect discipline?" (e.g., old version was pure,
new version introduces IO -- that's an effect regression even if outputs match).
"""

import sys, os, time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V138_effect_aware_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V139_certified_regression'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))

from effect_aware_verification import (
    EffectAwareVerifier, EffectAwareResult, EffectVC, EffectVCGenerator,
    EAVStatus, verify_effects, infer_and_verify, effect_verification_summary
)
from certified_regression import (
    RegressionCertificate, RegressionVerdict, RegressionMethod,
    verify_regression, verify_function_regression, verify_program_regression,
    check_regression_certificate, regression_summary
)
from proof_certificates import ProofCertificate, ProofObligation, CertStatus, ProofKind, combine_certificates
from effect_systems import EffectInferrer, EffectSet, Effect, FnEffectSig


class EffectRegressionVerdict(Enum):
    SAFE = "safe"           # No effect regression, properties preserved
    EFFECT_REGRESSION = "effect_regression"  # New effects introduced
    PROPERTY_FAILURE = "property_failure"    # Same effects but property broken
    UNSAFE = "unsafe"       # Both effect regression and property failure
    UNKNOWN = "unknown"


class EffectChangeKind(Enum):
    ADDED = "added"         # New effect in new version
    REMOVED = "removed"     # Effect removed in new version
    UNCHANGED = "unchanged"


@dataclass
class EffectChange:
    """Describes a single effect change between versions."""
    function: str
    effect: str
    kind: EffectChangeKind
    old_effects: List[str]
    new_effects: List[str]


@dataclass
class EffectRegressionResult:
    """Full result of effect-aware regression verification."""
    verdict: EffectRegressionVerdict
    source_old: str
    source_new: str
    effect_changes: List[EffectChange]
    old_effects: Dict[str, EffectAwareResult]
    new_effects: Dict[str, EffectAwareResult]
    regression_cert: Optional[RegressionCertificate]
    old_verification: Optional[EffectAwareResult]
    new_verification: Optional[EffectAwareResult]
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def has_effect_regression(self) -> bool:
        return any(c.kind == EffectChangeKind.ADDED for c in self.effect_changes)

    @property
    def has_effect_improvement(self) -> bool:
        return any(c.kind == EffectChangeKind.REMOVED for c in self.effect_changes)

    @property
    def added_effects(self) -> List[EffectChange]:
        return [c for c in self.effect_changes if c.kind == EffectChangeKind.ADDED]

    @property
    def removed_effects(self) -> List[EffectChange]:
        return [c for c in self.effect_changes if c.kind == EffectChangeKind.REMOVED]

    def summary(self) -> str:
        lines = [f"Effect-Aware Regression: {self.verdict.value}"]
        if self.effect_changes:
            lines.append(f"  Effect changes: {len(self.effect_changes)}")
            for c in self.effect_changes:
                lines.append(f"    {c.function}: {c.effect} ({c.kind.value})")
        if self.regression_cert:
            lines.append(f"  Regression cert: {self.regression_cert.verdict.value}")
        if self.old_verification:
            lines.append(f"  Old verification: {self.old_verification.status.value}")
        if self.new_verification:
            lines.append(f"  New verification: {self.new_verification.status.value}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "effect_changes": [
                {"function": c.function, "effect": c.effect, "kind": c.kind.value,
                 "old_effects": c.old_effects, "new_effects": c.new_effects}
                for c in self.effect_changes
            ],
            "has_effect_regression": self.has_effect_regression,
            "has_effect_improvement": self.has_effect_improvement,
            "added_effects_count": len(self.added_effects),
            "removed_effects_count": len(self.removed_effects),
            "regression_verdict": self.regression_cert.verdict.value if self.regression_cert else None,
            "old_verification_status": self.old_verification.status.value if self.old_verification else None,
            "new_verification_status": self.new_verification.status.value if self.new_verification else None,
            "errors": self.errors,
        }


def _extract_effect_names(sig: FnEffectSig) -> List[str]:
    """Extract effect names from a FnEffectSig."""
    names = []
    if sig.effects:
        for eff in sig.effects.effects:
            if isinstance(eff, Effect):
                names.append(str(eff))
            else:
                names.append(str(eff))
    return sorted(names)


def _infer_effects(source: str) -> Dict[str, List[str]]:
    """Run effect inference on source, return {fn_name: [effect_names]}."""
    try:
        inferrer = EffectInferrer()
        sigs = inferrer.infer_program(source)
        result = {}
        for fn_name, sig in sigs.items():
            result[fn_name] = _extract_effect_names(sig)
        return result
    except Exception:
        return {}


def _compute_effect_changes(old_effects: Dict[str, List[str]],
                             new_effects: Dict[str, List[str]]) -> List[EffectChange]:
    """Compare effect signatures between versions."""
    changes = []
    all_fns = sorted(set(list(old_effects.keys()) + list(new_effects.keys())))

    for fn in all_fns:
        old_effs = set(old_effects.get(fn, []))
        new_effs = set(new_effects.get(fn, []))

        # Effects added in new version
        for eff in sorted(new_effs - old_effs):
            changes.append(EffectChange(
                function=fn, effect=eff, kind=EffectChangeKind.ADDED,
                old_effects=sorted(old_effs), new_effects=sorted(new_effs)
            ))

        # Effects removed in new version
        for eff in sorted(old_effs - new_effs):
            changes.append(EffectChange(
                function=fn, effect=eff, kind=EffectChangeKind.REMOVED,
                old_effects=sorted(old_effs), new_effects=sorted(new_effs)
            ))

    return changes


def verify_effect_regression(source_old: str, source_new: str,
                              symbolic_inputs: Optional[Dict] = None,
                              property_source: Optional[str] = None,
                              output_var: Optional[str] = None,
                              declared_effects: Optional[Dict[str, EffectSet]] = None,
                              max_paths: int = 64,
                              max_k: int = 20) -> EffectRegressionResult:
    """Full effect-aware regression verification.

    Phase 1: Infer effects on both versions, compute changes
    Phase 2: Run V138 verification on both versions
    Phase 3: If property given, run V139 certified regression
    Phase 4: Combine into final verdict
    """
    errors = []
    t0 = time.time()

    # Phase 1: Effect inference and comparison
    old_inferred = _infer_effects(source_old)
    new_inferred = _infer_effects(source_new)
    effect_changes = _compute_effect_changes(old_inferred, new_inferred)

    # Phase 2: Effect verification on both versions
    old_ver = None
    new_ver = None
    try:
        old_ver = verify_effects(source_old, declared=declared_effects)
    except Exception as e:
        errors.append(f"Old version effect verification failed: {e}")
    try:
        new_ver = verify_effects(source_new, declared=declared_effects)
    except Exception as e:
        errors.append(f"New version effect verification failed: {e}")

    # Phase 3: Certified regression (if property given)
    reg_cert = None
    if property_source or symbolic_inputs:
        try:
            reg_cert = verify_regression(
                source_old, source_new,
                symbolic_inputs=symbolic_inputs,
                property_source=property_source,
                output_var=output_var,
                max_paths=max_paths,
                max_k=max_k
            )
        except Exception as e:
            errors.append(f"Regression verification failed: {e}")

    # Phase 4: Compute verdict
    has_effect_reg = any(c.kind == EffectChangeKind.ADDED for c in effect_changes)
    has_property_fail = (reg_cert is not None and reg_cert.verdict == RegressionVerdict.UNSAFE)

    if has_effect_reg and has_property_fail:
        verdict = EffectRegressionVerdict.UNSAFE
    elif has_effect_reg:
        verdict = EffectRegressionVerdict.EFFECT_REGRESSION
    elif has_property_fail:
        verdict = EffectRegressionVerdict.PROPERTY_FAILURE
    elif reg_cert is not None and reg_cert.verdict == RegressionVerdict.SAFE:
        verdict = EffectRegressionVerdict.SAFE
    elif not has_effect_reg and not effect_changes:
        verdict = EffectRegressionVerdict.SAFE
    elif not has_effect_reg and all(c.kind == EffectChangeKind.REMOVED for c in effect_changes):
        # Only removed effects = improvement
        verdict = EffectRegressionVerdict.SAFE
    else:
        verdict = EffectRegressionVerdict.UNKNOWN

    elapsed = time.time() - t0

    return EffectRegressionResult(
        verdict=verdict,
        source_old=source_old,
        source_new=source_new,
        effect_changes=effect_changes,
        old_effects={fn: None for fn in old_inferred},
        new_effects={fn: None for fn in new_inferred},
        regression_cert=reg_cert,
        old_verification=old_ver,
        new_verification=new_ver,
        errors=errors,
        metadata={"elapsed": elapsed, "old_inferred": old_inferred, "new_inferred": new_inferred}
    )


def verify_function_effect_regression(source_old: str, source_new: str,
                                       fn_name: str,
                                       param_types: Optional[Dict] = None,
                                       declared_effects: Optional[Dict[str, EffectSet]] = None,
                                       max_paths: int = 64) -> EffectRegressionResult:
    """Verify effect regression for a specific function."""
    # Infer effects
    old_inferred = _infer_effects(source_old)
    new_inferred = _infer_effects(source_new)

    # Filter to the target function
    old_fn = {fn_name: old_inferred.get(fn_name, [])}
    new_fn = {fn_name: new_inferred.get(fn_name, [])}
    effect_changes = _compute_effect_changes(old_fn, new_fn)

    # Effect verification
    errors = []
    old_ver = None
    new_ver = None
    try:
        old_ver = verify_effects(source_old, declared=declared_effects)
    except Exception as e:
        errors.append(f"Old version effect verification failed: {e}")
    try:
        new_ver = verify_effects(source_new, declared=declared_effects)
    except Exception as e:
        errors.append(f"New version effect verification failed: {e}")

    # Function regression
    reg_cert = None
    if param_types is not None:
        try:
            reg_cert = verify_function_regression(
                source_old, source_new, fn_name, param_types,
                max_paths=max_paths
            )
        except Exception as e:
            errors.append(f"Function regression failed: {e}")

    has_effect_reg = any(c.kind == EffectChangeKind.ADDED for c in effect_changes)
    has_property_fail = (reg_cert is not None and reg_cert.verdict == RegressionVerdict.UNSAFE)

    if has_effect_reg and has_property_fail:
        verdict = EffectRegressionVerdict.UNSAFE
    elif has_effect_reg:
        verdict = EffectRegressionVerdict.EFFECT_REGRESSION
    elif has_property_fail:
        verdict = EffectRegressionVerdict.PROPERTY_FAILURE
    elif reg_cert and reg_cert.verdict == RegressionVerdict.SAFE:
        verdict = EffectRegressionVerdict.SAFE
    elif not has_effect_reg:
        verdict = EffectRegressionVerdict.SAFE
    else:
        verdict = EffectRegressionVerdict.UNKNOWN

    return EffectRegressionResult(
        verdict=verdict,
        source_old=source_old,
        source_new=source_new,
        effect_changes=effect_changes,
        old_effects={fn_name: None},
        new_effects={fn_name: None},
        regression_cert=reg_cert,
        old_verification=old_ver,
        new_verification=new_ver,
        errors=errors,
        metadata={"fn_name": fn_name, "old_inferred": old_inferred, "new_inferred": new_inferred}
    )


def check_effect_purity_preserved(source_old: str, source_new: str,
                                    fn_name: str) -> Dict:
    """Check whether a function that was pure in old version is still pure in new version."""
    old_inferred = _infer_effects(source_old)
    new_inferred = _infer_effects(source_new)

    old_effs = set(old_inferred.get(fn_name, []))
    new_effs = set(new_inferred.get(fn_name, []))

    was_pure = len(old_effs) == 0
    is_pure = len(new_effs) == 0

    return {
        "function": fn_name,
        "was_pure": was_pure,
        "is_pure": is_pure,
        "purity_preserved": not was_pure or is_pure,  # Only fails if was pure and now isn't
        "old_effects": sorted(old_effs),
        "new_effects": sorted(new_effs),
        "added_effects": sorted(new_effs - old_effs),
        "removed_effects": sorted(old_effs - new_effs),
    }


def compare_effect_regression_methods(source_old: str, source_new: str,
                                        symbolic_inputs: Dict,
                                        property_source: str,
                                        output_var: Optional[str] = None,
                                        max_paths: int = 64,
                                        max_k: int = 20) -> Dict:
    """Compare effect-only vs full regression vs combined."""
    t0 = time.time()

    # Effect-only analysis
    old_inferred = _infer_effects(source_old)
    new_inferred = _infer_effects(source_new)
    changes = _compute_effect_changes(old_inferred, new_inferred)
    t_effect = time.time() - t0

    # Full regression
    t1 = time.time()
    try:
        reg_cert = verify_regression(
            source_old, source_new,
            symbolic_inputs=symbolic_inputs,
            property_source=property_source,
            output_var=output_var,
            max_paths=max_paths,
            max_k=max_k
        )
        reg_verdict = reg_cert.verdict.value
    except Exception as e:
        reg_cert = None
        reg_verdict = f"error: {e}"
    t_reg = time.time() - t1

    # Combined
    t2 = time.time()
    combined = verify_effect_regression(
        source_old, source_new,
        symbolic_inputs=symbolic_inputs,
        property_source=property_source,
        output_var=output_var,
        max_paths=max_paths,
        max_k=max_k
    )
    t_combined = time.time() - t2

    return {
        "effect_only": {
            "changes": len(changes),
            "has_regression": any(c.kind == EffectChangeKind.ADDED for c in changes),
            "time": t_effect,
        },
        "regression_only": {
            "verdict": reg_verdict,
            "time": t_reg,
        },
        "combined": {
            "verdict": combined.verdict.value,
            "effect_changes": len(combined.effect_changes),
            "has_effect_regression": combined.has_effect_regression,
            "time": t_combined,
        },
    }


def effect_regression_summary(result: EffectRegressionResult) -> Dict:
    """Generate a summary dict for an effect regression result."""
    return {
        "verdict": result.verdict.value,
        "effect_changes": len(result.effect_changes),
        "added_effects": len(result.added_effects),
        "removed_effects": len(result.removed_effects),
        "has_effect_regression": result.has_effect_regression,
        "has_effect_improvement": result.has_effect_improvement,
        "regression_cert_verdict": result.regression_cert.verdict.value if result.regression_cert else None,
        "old_verification_status": result.old_verification.status.value if result.old_verification else None,
        "new_verification_status": result.new_verification.status.value if result.new_verification else None,
        "errors": result.errors,
    }
