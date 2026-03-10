"""
V053: Cross-Analysis Correlation

Mines correlations across V050's analyses to understand which analyses
are complementary, redundant, or conflicting. Given a set of programs,
runs all analyses and computes:

1. Agreement matrix: how often do analyses agree on pass/fail?
2. Complementarity: which analyses catch bugs the others miss?
3. Redundancy: which analyses always agree (one could be skipped)?
4. Conflict detection: when one says safe but another says unsafe
5. Recommendation engine: given program features, suggest which analyses to run
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any
from collections import defaultdict

_work = os.path.join(os.path.dirname(__file__), '..')

def _add_vpath(name):
    p = os.path.join(_work, name)
    if p not in sys.path:
        sys.path.insert(0, p)

_add_vpath('V050_holistic_verification_dashboard')

from holistic_verification import (
    verify_holistic, PipelineConfig, VerificationReport,
    AnalysisResult, AnalysisStatus, available_analyses,
    run_single_analysis,
)


# --- Data Types ---

@dataclass
class AnalysisPair:
    """Statistics about a pair of analyses."""
    a: str
    b: str
    both_pass: int = 0
    both_fail: int = 0
    a_pass_b_fail: int = 0
    a_fail_b_pass: int = 0
    total: int = 0

    @property
    def agreement_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.both_pass + self.both_fail) / self.total

    @property
    def conflict_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.a_pass_b_fail + self.a_fail_b_pass) / self.total

    @property
    def is_redundant(self) -> bool:
        """True if analyses always agree (one is redundant)."""
        return self.total > 0 and self.agreement_rate == 1.0

    @property
    def is_complementary(self) -> bool:
        """True if each catches bugs the other misses."""
        return self.a_pass_b_fail > 0 and self.a_fail_b_pass > 0


@dataclass
class AnalysisProfile:
    """Profile of a single analysis across multiple programs."""
    name: str
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    error_count: int = 0
    skip_count: int = 0
    total: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.pass_count / self.total

    @property
    def fail_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.fail_count / self.total

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.error_count / self.total

    @property
    def effectiveness(self) -> float:
        """How often this analysis produces a definitive result (pass or fail, not error/skip)."""
        if self.total == 0:
            return 0.0
        return (self.pass_count + self.fail_count) / self.total


@dataclass
class ProgramFeatures:
    """Detected features of a C10 program for recommendation."""
    has_loops: bool = False
    has_functions: bool = False
    has_specs: bool = False          # requires/ensures annotations
    has_conditionals: bool = False
    has_io: bool = False             # print statements
    has_recursion: bool = False      # function calls within function bodies
    function_count: int = 0
    statement_count: int = 0


@dataclass
class AnalysisRecommendation:
    """Recommendation for which analyses to run."""
    recommended: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    skip: List[str] = field(default_factory=list)
    reason: Dict[str, str] = field(default_factory=dict)


@dataclass
class CorrelationReport:
    """Complete cross-analysis correlation report."""
    programs_analyzed: int = 0
    profiles: Dict[str, AnalysisProfile] = field(default_factory=dict)
    pairs: Dict[Tuple[str, str], AnalysisPair] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def redundant_pairs(self) -> List[AnalysisPair]:
        return [p for p in self.pairs.values() if p.is_redundant]

    @property
    def complementary_pairs(self) -> List[AnalysisPair]:
        return [p for p in self.pairs.values() if p.is_complementary]

    def summary(self) -> str:
        lines = []
        lines.append("Cross-Analysis Correlation Report")
        lines.append("=" * 50)
        lines.append(f"Programs analyzed: {self.programs_analyzed}")
        lines.append("")

        # Per-analysis profiles
        lines.append("Analysis Effectiveness:")
        for name in sorted(self.profiles.keys()):
            p = self.profiles[name]
            lines.append(f"  {name}: pass={p.pass_rate:.0%} fail={p.fail_rate:.0%} "
                        f"error={p.error_rate:.0%} effectiveness={p.effectiveness:.0%}")

        # Redundant pairs
        redundant = self.redundant_pairs
        if redundant:
            lines.append("")
            lines.append(f"Redundant pairs ({len(redundant)}):")
            for p in redundant:
                lines.append(f"  {p.a} <-> {p.b}: always agree ({p.agreement_rate:.0%})")

        # Complementary pairs
        comp = self.complementary_pairs
        if comp:
            lines.append("")
            lines.append(f"Complementary pairs ({len(comp)}):")
            for p in comp:
                lines.append(f"  {p.a} <-> {p.b}: each catches unique bugs")

        # Conflicts
        if self.conflicts:
            lines.append("")
            lines.append(f"Conflicts ({len(self.conflicts)}):")
            for c in self.conflicts[:5]:
                lines.append(f"  {c.get('description', str(c))}")

        return "\n".join(lines)


# --- Feature Detection ---

def detect_features(source: str) -> ProgramFeatures:
    """Detect structural features of a C10 program."""
    features = ProgramFeatures()

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                         'challenges', 'C010_stack_vm'))
        from stack_vm import lex, Parser
        tokens = lex(source)
        program = Parser(tokens).parse()

        fn_names = set()
        call_names = set()

        def walk_stmts(stmts):
            for stmt in stmts:
                features.statement_count += 1
                cls = type(stmt).__name__

                if cls == 'FnDecl':
                    features.has_functions = True
                    features.function_count += 1
                    fn_names.add(stmt.name)
                    if hasattr(stmt, 'body') and hasattr(stmt.body, 'stmts'):
                        walk_stmts(stmt.body.stmts)

                if cls == 'WhileStmt':
                    features.has_loops = True
                    if hasattr(stmt, 'body') and hasattr(stmt.body, 'stmts'):
                        walk_stmts(stmt.body.stmts)

                if cls == 'IfStmt':
                    features.has_conditionals = True
                    if hasattr(stmt, 'then_body') and hasattr(stmt.then_body, 'stmts'):
                        walk_stmts(stmt.then_body.stmts)
                    if hasattr(stmt, 'else_body') and stmt.else_body:
                        if hasattr(stmt.else_body, 'stmts'):
                            walk_stmts(stmt.else_body.stmts)

                if cls == 'PrintStmt':
                    features.has_io = True

                # Check for annotations (requires/ensures are CallExpr at stmt level)
                if cls == 'CallExpr':
                    callee = getattr(stmt, 'callee', '')
                    if callee in ('requires', 'ensures', 'invariant'):
                        features.has_specs = True

                # Track calls for recursion detection
                walk_expr(getattr(stmt, 'value', None))
                walk_expr(getattr(stmt, 'cond', None))

        def walk_expr(expr):
            if expr is None:
                return
            if hasattr(expr, 'callee'):
                call_names.add(expr.callee)
            if hasattr(expr, 'left'):
                walk_expr(expr.left)
            if hasattr(expr, 'right'):
                walk_expr(expr.right)
            if hasattr(expr, 'args'):
                for a in expr.args:
                    walk_expr(a)

        walk_stmts(program.stmts)
        features.has_recursion = bool(fn_names & call_names)

    except Exception:
        pass

    return features


# --- Recommendation Engine ---

def recommend_analyses(features: ProgramFeatures) -> AnalysisRecommendation:
    """Recommend analyses based on program features."""
    rec = AnalysisRecommendation()

    # Always recommend: certified AI and effects (fast, always informative)
    rec.recommended.append("certified_ai")
    rec.reason["certified_ai"] = "Fast, always provides value bounds"

    rec.recommended.append("effects")
    rec.reason["effects"] = "Fast, identifies side effects"

    # VCGen: only useful with specs
    if features.has_specs:
        rec.recommended.append("vcgen")
        rec.reason["vcgen"] = "Program has requires/ensures annotations"
        rec.recommended.append("modular_verification")
        rec.reason["modular_verification"] = "Specs enable contract checking"
    else:
        rec.skip.append("vcgen")
        rec.reason["vcgen"] = "No specs; VCGen has nothing to verify"
        rec.skip.append("modular_verification")
        rec.reason["modular_verification"] = "No contracts to check"

    # Symbolic execution: useful with conditionals/functions
    if features.has_conditionals or features.has_functions:
        rec.recommended.append("guided_symex")
        rec.reason["guided_symex"] = "Conditionals/functions benefit from path exploration"
    else:
        rec.optional.append("guided_symex")
        rec.reason["guided_symex"] = "Linear code; symex adds little value"

    # Termination: only useful with loops
    if features.has_loops:
        rec.recommended.append("termination")
        rec.reason["termination"] = "Program has loops to analyze"
    else:
        rec.skip.append("termination")
        rec.reason["termination"] = "No loops; termination is trivial"

    # Refinement types: useful with functions
    if features.has_functions:
        rec.optional.append("refinement_types")
        rec.reason["refinement_types"] = "Functions present for type refinement"
    else:
        rec.skip.append("refinement_types")
        rec.reason["refinement_types"] = "No functions to type-check"

    # Verified compilation: always available but expensive
    if features.has_functions:
        rec.optional.append("verified_compilation")
        rec.reason["verified_compilation"] = "Can validate optimization passes"
    else:
        rec.skip.append("verified_compilation")
        rec.reason["verified_compilation"] = "Trivial program; optimization irrelevant"

    return rec


def recommendation_to_config(rec: AnalysisRecommendation) -> PipelineConfig:
    """Convert an analysis recommendation to a PipelineConfig."""
    enabled = set(rec.recommended + rec.optional)
    return PipelineConfig(
        certified_ai="certified_ai" in enabled,
        vcgen="vcgen" in enabled,
        effects="effects" in enabled,
        guided_symex="guided_symex" in enabled,
        refinement_types="refinement_types" in enabled,
        termination="termination" in enabled,
        modular_verification="modular_verification" in enabled,
        verified_compilation="verified_compilation" in enabled,
    )


# --- Correlation Analysis ---

def _is_pass(status: AnalysisStatus) -> bool:
    return status in (AnalysisStatus.PASSED, AnalysisStatus.SKIPPED)


def _is_fail(status: AnalysisStatus) -> bool:
    return status == AnalysisStatus.FAILED


def correlate_analyses(programs: List[str], config: PipelineConfig = None) -> CorrelationReport:
    """
    Run all analyses on multiple programs and compute correlations.

    Args:
        programs: List of C10 program source strings
        config: Pipeline configuration (default: all enabled)

    Returns:
        CorrelationReport with profiles, pairs, and conflicts
    """
    if config is None:
        config = PipelineConfig.all_enabled()

    report = CorrelationReport()

    # Collect results per program
    all_results: List[VerificationReport] = []
    for source in programs:
        vr = verify_holistic(source, config)
        all_results.append(vr)
        report.programs_analyzed += 1

    if not all_results:
        return report

    # Build per-analysis profiles
    analysis_names = set()
    for vr in all_results:
        for a in vr.analyses:
            analysis_names.add(a.name)

    for name in analysis_names:
        profile = AnalysisProfile(name=name)
        for vr in all_results:
            for a in vr.analyses:
                if a.name == name:
                    profile.total += 1
                    if a.status == AnalysisStatus.PASSED:
                        profile.pass_count += 1
                    elif a.status == AnalysisStatus.FAILED:
                        profile.fail_count += 1
                    elif a.status == AnalysisStatus.WARNING:
                        profile.warn_count += 1
                    elif a.status == AnalysisStatus.ERROR:
                        profile.error_count += 1
                    elif a.status == AnalysisStatus.SKIPPED:
                        profile.skip_count += 1
        report.profiles[name] = profile

    # Build pairwise correlations
    name_list = sorted(analysis_names)
    for i, a_name in enumerate(name_list):
        for j, b_name in enumerate(name_list):
            if i >= j:
                continue

            pair = AnalysisPair(a=a_name, b=b_name)
            for vr in all_results:
                a_result = None
                b_result = None
                for a in vr.analyses:
                    if a.name == a_name:
                        a_result = a
                    if a.name == b_name:
                        b_result = a

                if a_result is None or b_result is None:
                    continue

                pair.total += 1
                a_ok = _is_pass(a_result.status)
                b_ok = _is_pass(b_result.status)

                if a_ok and b_ok:
                    pair.both_pass += 1
                elif not a_ok and not b_ok:
                    pair.both_fail += 1
                elif a_ok and not b_ok:
                    pair.a_pass_b_fail += 1

                    # Record conflict
                    if _is_fail(b_result.status):
                        report.conflicts.append({
                            "description": f"{a_name} PASS but {b_name} FAIL",
                            "program_index": all_results.index(vr),
                            "a": a_name, "b": b_name,
                        })
                else:
                    pair.a_fail_b_pass += 1

                    if _is_fail(a_result.status):
                        report.conflicts.append({
                            "description": f"{a_name} FAIL but {b_name} PASS",
                            "program_index": all_results.index(vr),
                            "a": a_name, "b": b_name,
                        })

            report.pairs[(a_name, b_name)] = pair

    return report


def smart_verify(source: str) -> VerificationReport:
    """
    Verify a program using feature-guided analysis selection.

    Detects program features, recommends analyses, and runs only
    the recommended ones.
    """
    features = detect_features(source)
    rec = recommend_analyses(features)
    config = recommendation_to_config(rec)
    return verify_holistic(source, config)
