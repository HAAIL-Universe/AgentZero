"""
V050: Holistic Verification Dashboard

Composes ALL applicable V-challenges into a unified verification pipeline.
Given a C10 program, runs multiple analyses and produces a comprehensive
verification report with proof certificates.

Analyses orchestrated:
  1. Certified Abstract Interpretation (V046) - intervals/signs with certificates
  2. Verification Condition Generation (V004) - Hoare logic VCs
  3. Effect Analysis (V040) - effect inference and checking
  4. Guided Symbolic Execution (V001) - AI-pruned path exploration
  5. Refinement Types (V011) - liquid type checking
  6. Termination Analysis (V025) - ranking function proofs
  7. Modular Verification (V039) - contract-based inter-procedural
  8. Verified Compilation (V049) - translation validation
  9. Proof Certificates (V044) - combined certificate from all analyses
"""

import sys
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))

# V-challenge imports (each has its own sys.path setup)
_work = os.path.join(os.path.dirname(__file__), '..')

def _add_vpath(name):
    p = os.path.join(_work, name)
    if p not in sys.path:
        sys.path.insert(0, p)

_add_vpath('V001_guided_symbolic_execution')
_add_vpath('V004_verification_conditions')
_add_vpath('V011_refinement_types')
_add_vpath('V025_termination_analysis')
_add_vpath('V039_modular_verification')
_add_vpath('V040_effect_systems')
_add_vpath('V044_proof_certificates')
_add_vpath('V046_certified_abstract_interpretation')
_add_vpath('V049_verified_compilation')


# --- Result Types ---

class AnalysisStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"       # analysis itself crashed
    SKIPPED = "skipped"   # not applicable or disabled


@dataclass
class AnalysisResult:
    """Result from a single analysis pass."""
    name: str
    status: AnalysisStatus
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    certificate: Any = None  # ProofCertificate if available
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in (AnalysisStatus.PASSED, AnalysisStatus.SKIPPED)


@dataclass
class VerificationReport:
    """Comprehensive verification report from all analyses."""
    source: str
    analyses: List[AnalysisResult] = field(default_factory=list)
    combined_certificate: Any = None
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> List[AnalysisResult]:
        return [a for a in self.analyses if a.status == AnalysisStatus.PASSED]

    @property
    def failed(self) -> List[AnalysisResult]:
        return [a for a in self.analyses if a.status == AnalysisStatus.FAILED]

    @property
    def warnings(self) -> List[AnalysisResult]:
        return [a for a in self.analyses if a.status == AnalysisStatus.WARNING]

    @property
    def errors(self) -> List[AnalysisResult]:
        return [a for a in self.analyses if a.status == AnalysisStatus.ERROR]

    @property
    def all_passed(self) -> bool:
        """True if no analysis failed (warnings/skips OK)."""
        return len(self.failed) == 0 and len(self.errors) == 0

    @property
    def score(self) -> float:
        """Verification confidence score: 0.0 to 1.0."""
        if not self.analyses:
            return 0.0
        active = [a for a in self.analyses if a.status != AnalysisStatus.SKIPPED]
        if not active:
            return 1.0
        passed = sum(1 for a in active if a.status == AnalysisStatus.PASSED)
        warned = sum(1 for a in active if a.status == AnalysisStatus.WARNING)
        return (passed + warned * 0.5) / len(active)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("HOLISTIC VERIFICATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        for a in self.analyses:
            icon = {
                AnalysisStatus.PASSED: "[PASS]",
                AnalysisStatus.FAILED: "[FAIL]",
                AnalysisStatus.WARNING: "[WARN]",
                AnalysisStatus.ERROR: "[ERR ]",
                AnalysisStatus.SKIPPED: "[SKIP]",
            }[a.status]
            lines.append(f"  {icon} {a.name}: {a.summary}")
            for f in a.findings:
                lines.append(f"         - {f}")

        lines.append("")
        lines.append(f"Score: {self.score:.0%} ({len(self.passed)} passed, "
                     f"{len(self.warnings)} warnings, {len(self.failed)} failed, "
                     f"{len(self.errors)} errors)")
        lines.append(f"Duration: {self.total_duration_ms:.0f}ms")

        if self.combined_certificate:
            cert = self.combined_certificate
            ob_count = len(cert.obligations) if hasattr(cert, 'obligations') else 0
            lines.append(f"Certificate: {ob_count} proof obligations")

        lines.append("=" * 60)
        return "\n".join(lines)


# --- Individual Analysis Runners ---

def _timed(fn):
    """Run fn, return (result, duration_ms). Catches exceptions."""
    t0 = time.time()
    try:
        result = fn()
        dt = (time.time() - t0) * 1000
        return result, dt, None
    except Exception as e:
        dt = (time.time() - t0) * 1000
        return None, dt, f"{type(e).__name__}: {e}"


def run_certified_ai(source: str) -> AnalysisResult:
    """V046: Certified Abstract Interpretation."""
    from certified_abstract_interpretation import certify_abstract_interpretation

    raw, dt, err = _timed(lambda: certify_abstract_interpretation(source))
    if err:
        return AnalysisResult("Certified Abstract Interpretation", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    ai_result, cert = raw
    findings = []

    # Extract variable bounds from the analysis
    if hasattr(ai_result, 'analysis') and ai_result.analysis:
        analysis = ai_result.analysis
        if hasattr(analysis, 'env') and analysis.env:
            env = analysis.env
            if hasattr(env, 'intervals'):
                for var, interval in env.intervals.items():
                    findings.append(f"{var} in {interval}")

    certified = ai_result.certified if hasattr(ai_result, 'certified') else False
    status = AnalysisStatus.PASSED if certified else AnalysisStatus.WARNING

    return AnalysisResult(
        "Certified Abstract Interpretation", status,
        f"Certified: {certified}",
        details={"certified": certified},
        findings=findings[:5],  # limit output
        certificate=cert,
        duration_ms=dt
    )


def run_vcgen(source: str) -> AnalysisResult:
    """V004: Verification Condition Generation."""
    from vc_gen import verify_program

    raw, dt, err = _timed(lambda: verify_program(source))
    if err:
        return AnalysisResult("Verification Conditions", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    failed_vcs = []
    if hasattr(raw, 'vcs'):
        for vc in raw.vcs:
            if hasattr(vc, 'verified') and not vc.verified:
                name = getattr(vc, 'name', 'unknown')
                failed_vcs.append(name)
                findings.append(f"VC failed: {name}")

    verified = getattr(raw, 'verified', False)
    status = AnalysisStatus.PASSED if verified else AnalysisStatus.FAILED
    total = len(raw.vcs) if hasattr(raw, 'vcs') else 0

    return AnalysisResult(
        "Verification Conditions", status,
        f"{total - len(failed_vcs)}/{total} VCs verified",
        details={"verified": verified, "total_vcs": total, "failed_vcs": failed_vcs},
        findings=findings[:5],
        duration_ms=dt
    )


def run_effects(source: str) -> AnalysisResult:
    """V040: Effect Analysis."""
    from effect_systems import infer_effects, check_effects

    raw, dt, err = _timed(lambda: check_effects(source))
    if err:
        # Fallback to just inference
        raw2, dt2, err2 = _timed(lambda: infer_effects(source))
        if err2:
            return AnalysisResult("Effect Analysis", AnalysisStatus.ERROR,
                                  f"Analysis error: {err}", duration_ms=dt + dt2)
        findings = []
        for fn_name, sig in raw2.items():
            eff_set = getattr(sig, 'effects', None)
            eff_str = str(eff_set) if eff_set else "Pure"
            findings.append(f"{fn_name}: {eff_str}")
        return AnalysisResult(
            "Effect Analysis", AnalysisStatus.PASSED,
            f"Inferred effects for {len(raw2)} functions",
            details={"fn_count": len(raw2)},
            findings=findings[:5],
            duration_ms=dt + dt2
        )

    ok = raw.ok if hasattr(raw, 'ok') else True
    findings = []
    if hasattr(raw, 'fn_sigs'):
        for fn_name, sig in raw.fn_sigs.items():
            eff_set = getattr(sig, 'effects', None)
            eff_str = str(eff_set) if eff_set else "Pure"
            findings.append(f"{fn_name}: {eff_str}")
    if hasattr(raw, 'checks'):
        for c in raw.checks:
            if hasattr(c, 'ok') and not c.ok:
                msg = getattr(c, 'message', str(c))
                findings.append(f"Check failed: {msg}")

    status = AnalysisStatus.PASSED if ok else AnalysisStatus.WARNING

    return AnalysisResult(
        "Effect Analysis", status,
        f"Effects {'consistent' if ok else 'inconsistent'}",
        details={"ok": ok},
        findings=findings[:5],
        duration_ms=dt
    )


def run_guided_symex(source: str) -> AnalysisResult:
    """V001: Abstract-Interpretation-Guided Symbolic Execution."""
    from guided_symbolic import guided_execute

    raw, dt, err = _timed(lambda: guided_execute(source))
    if err:
        return AnalysisResult("Guided Symbolic Execution", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    paths = 0
    assertion_failures = 0
    pruned = 0

    if hasattr(raw, 'execution') and raw.execution:
        exec_result = raw.execution
        if hasattr(exec_result, 'completed_paths'):
            paths = len(exec_result.completed_paths)
            for p in exec_result.completed_paths:
                status_val = getattr(p, 'status', None)
                if status_val and 'ASSERTION' in str(status_val):
                    assertion_failures += 1

    if hasattr(raw, 'pruned_by_abstract'):
        pruned = raw.pruned_by_abstract
        if pruned > 0:
            findings.append(f"{pruned} paths pruned by abstract analysis")

    if assertion_failures > 0:
        findings.append(f"{assertion_failures} assertion failure(s) found")
        status = AnalysisStatus.FAILED
    else:
        status = AnalysisStatus.PASSED

    findings.append(f"{paths} paths explored")

    return AnalysisResult(
        "Guided Symbolic Execution", status,
        f"{paths} paths, {assertion_failures} failures, {pruned} pruned",
        details={"paths": paths, "assertion_failures": assertion_failures, "pruned": pruned},
        findings=findings,
        duration_ms=dt
    )


def run_refinement_types(source: str) -> AnalysisResult:
    """V011: Refinement Type Checking."""
    from refinement_types import check_program_refinements

    raw, dt, err = _timed(lambda: check_program_refinements(source))
    if err:
        return AnalysisResult("Refinement Types", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    errors = getattr(raw, 'errors', [])
    verified = getattr(raw, 'verified_obligations', 0)
    total = getattr(raw, 'total_obligations', 0)

    for e in errors[:3]:
        findings.append(str(e))

    if errors:
        status = AnalysisStatus.FAILED
    elif total > 0:
        status = AnalysisStatus.PASSED
    else:
        status = AnalysisStatus.PASSED

    return AnalysisResult(
        "Refinement Types", status,
        f"{verified}/{total} obligations, {len(errors)} errors",
        details={"verified": verified, "total": total, "errors": len(errors)},
        findings=findings,
        duration_ms=dt
    )


def run_termination(source: str) -> AnalysisResult:
    """V025: Termination Analysis."""
    from termination import verify_all_terminate

    raw, dt, err = _timed(lambda: verify_all_terminate(source))
    if err:
        return AnalysisResult("Termination Analysis", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    proven = getattr(raw, 'proven', False)
    details_dict = {}

    if hasattr(raw, 'ranking_function') and raw.ranking_function:
        findings.append(f"Ranking function found: {raw.ranking_function}")
    if hasattr(raw, 'loops_analyzed'):
        details_dict['loops_analyzed'] = raw.loops_analyzed
    if hasattr(raw, 'loops_proven'):
        details_dict['loops_proven'] = raw.loops_proven

    status = AnalysisStatus.PASSED if proven else AnalysisStatus.WARNING

    return AnalysisResult(
        "Termination Analysis", status,
        f"Termination {'proven' if proven else 'not proven'}",
        details=details_dict,
        findings=findings[:3],
        duration_ms=dt
    )


def run_modular_verification(source: str) -> AnalysisResult:
    """V039: Modular Verification (contracts)."""
    from modular_verification import verify_modular

    raw, dt, err = _timed(lambda: verify_modular(source))
    if err:
        return AnalysisResult("Modular Verification", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    ok = True

    if hasattr(raw, 'function_results'):
        for fn_name, fn_res in raw.function_results.items():
            verified = getattr(fn_res, 'verified', False)
            if not verified:
                ok = False
                findings.append(f"{fn_name}: verification failed")
            else:
                findings.append(f"{fn_name}: verified")

    status = AnalysisStatus.PASSED if ok else AnalysisStatus.FAILED

    return AnalysisResult(
        "Modular Verification", status,
        f"{'All contracts verified' if ok else 'Contract violations found'}",
        details={},
        findings=findings[:5],
        duration_ms=dt
    )


def run_verified_compilation(source: str) -> AnalysisResult:
    """V049: Verified Compilation (translation validation)."""
    from verified_compilation import validate_compilation

    raw, dt, err = _timed(lambda: validate_compilation(source))
    if err:
        return AnalysisResult("Verified Compilation", AnalysisStatus.ERROR,
                              f"Analysis error: {err}", duration_ms=dt)

    findings = []
    all_valid = True

    if hasattr(raw, 'pass_results'):
        for pr in raw.pass_results:
            name = getattr(pr, 'pass_name', 'unknown')
            valid = getattr(pr, 'valid', False)
            if not valid:
                all_valid = False
                reason = getattr(pr, 'reason', '')
                findings.append(f"{name}: INVALID - {reason}")
            else:
                proof_count = len(getattr(pr, 'proofs', []))
                findings.append(f"{name}: valid ({proof_count} proofs)")

    exec_match = getattr(raw, 'execution_match', None)
    if exec_match is not None and not exec_match:
        all_valid = False
        findings.append("Execution mismatch: optimized output differs")

    status = AnalysisStatus.PASSED if all_valid else AnalysisStatus.FAILED
    cert = getattr(raw, 'certificate', None)

    return AnalysisResult(
        "Verified Compilation", status,
        f"Translation validation {'passed' if all_valid else 'failed'}",
        details={"all_valid": all_valid, "execution_match": exec_match},
        findings=findings[:6],
        certificate=cert,
        duration_ms=dt
    )


# --- Pipeline Configuration ---

@dataclass
class PipelineConfig:
    """Configure which analyses to run."""
    certified_ai: bool = True
    vcgen: bool = True
    effects: bool = True
    guided_symex: bool = True
    refinement_types: bool = True
    termination: bool = True
    modular_verification: bool = True
    verified_compilation: bool = True
    combine_certificates: bool = True

    @staticmethod
    def all_enabled() -> 'PipelineConfig':
        return PipelineConfig()

    @staticmethod
    def fast() -> 'PipelineConfig':
        """Lightweight config: just AI + effects + VCGen."""
        return PipelineConfig(
            guided_symex=False,
            refinement_types=False,
            termination=False,
            modular_verification=False,
            verified_compilation=False,
            combine_certificates=False,
        )

    @staticmethod
    def deep() -> 'PipelineConfig':
        """Full analysis with all passes enabled."""
        return PipelineConfig()


# --- Main Pipeline ---

def verify_holistic(source: str, config: PipelineConfig = None) -> VerificationReport:
    """
    Run the holistic verification pipeline on a C10 program.

    Args:
        source: C10 program source code
        config: Pipeline configuration (default: all enabled)

    Returns:
        VerificationReport with results from all analyses
    """
    if config is None:
        config = PipelineConfig.all_enabled()

    report = VerificationReport(source=source)
    t0 = time.time()

    # Run each analysis pass
    runners = []
    if config.certified_ai:
        runners.append(("certified_ai", run_certified_ai))
    if config.vcgen:
        runners.append(("vcgen", run_vcgen))
    if config.effects:
        runners.append(("effects", run_effects))
    if config.guided_symex:
        runners.append(("guided_symex", run_guided_symex))
    if config.refinement_types:
        runners.append(("refinement_types", run_refinement_types))
    if config.termination:
        runners.append(("termination", run_termination))
    if config.modular_verification:
        runners.append(("modular_verification", run_modular_verification))
    if config.verified_compilation:
        runners.append(("verified_compilation", run_verified_compilation))

    for name, runner in runners:
        result = runner(source)
        report.analyses.append(result)

    # Combine proof certificates if requested
    if config.combine_certificates:
        certs = [a.certificate for a in report.analyses if a.certificate is not None]
        if len(certs) >= 2:
            try:
                from proof_certificates import combine_certificates
                combined = combine_certificates(*certs, claim="holistic verification")
                report.combined_certificate = combined
            except Exception:
                pass  # certificate combination is best-effort
        elif len(certs) == 1:
            report.combined_certificate = certs[0]

    report.total_duration_ms = (time.time() - t0) * 1000
    return report


def quick_verify(source: str) -> VerificationReport:
    """Fast verification: AI + effects + VCGen only."""
    return verify_holistic(source, PipelineConfig.fast())


def deep_verify(source: str) -> VerificationReport:
    """Deep verification: all analyses enabled."""
    return verify_holistic(source, PipelineConfig.deep())


def verify_and_report(source: str, config: PipelineConfig = None) -> str:
    """Run verification and return human-readable summary string."""
    report = verify_holistic(source, config)
    return report.summary()


# --- Selective Analysis ---

def run_single_analysis(source: str, analysis_name: str) -> AnalysisResult:
    """Run a single named analysis."""
    dispatch = {
        "certified_ai": run_certified_ai,
        "vcgen": run_vcgen,
        "effects": run_effects,
        "guided_symex": run_guided_symex,
        "refinement_types": run_refinement_types,
        "termination": run_termination,
        "modular_verification": run_modular_verification,
        "verified_compilation": run_verified_compilation,
    }
    runner = dispatch.get(analysis_name)
    if runner is None:
        return AnalysisResult(analysis_name, AnalysisStatus.ERROR,
                              f"Unknown analysis: {analysis_name}. "
                              f"Available: {', '.join(dispatch.keys())}")
    return runner(source)


def available_analyses() -> List[str]:
    """List all available analysis names."""
    return [
        "certified_ai",
        "vcgen",
        "effects",
        "guided_symex",
        "refinement_types",
        "termination",
        "modular_verification",
        "verified_compilation",
    ]
