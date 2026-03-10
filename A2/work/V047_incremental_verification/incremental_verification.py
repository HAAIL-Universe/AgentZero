"""V047: Incremental Verification

Composes:
- V004 (VCGen) - verification condition generation
- V044 (Proof Certificates) - machine-checkable certificates
- V009 (Differential Testing/AST Diff concepts)

When a program changes, avoid re-verifying the entire program.
Instead:
1. Parse old and new versions
2. Determine which functions changed (AST-level diff)
3. Re-verify only changed functions
4. Reuse valid certificates for unchanged functions
5. Combine into a composite certificate

This gives O(changed) verification cost instead of O(total).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple
from enum import Enum

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
_challenges = os.path.join(_az, "challenges")

for p in [
    os.path.join(_work, "V044_proof_certificates"),
    os.path.join(_work, "V004_verification_conditions"),
    os.path.join(_work, "V002_pdr"),
    os.path.join(_challenges, "C010_stack_vm"),
    os.path.join(_challenges, "C037_smt_solver"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# V044 imports
from proof_certificates import (
    ProofKind, CertStatus, ProofObligation, ProofCertificate,
    check_certificate, combine_certificates,
    generate_vcgen_certificate, certify_program,
    sexpr_to_str, sexpr_to_smtlib,
)

# V004 imports
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SAnd,
    s_and, s_implies, lower_to_smt,
)

# C010 imports
from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr, Var, IntLit, BinOp,
)


# ---------------------------------------------------------------------------
# AST-Level Diff for C10 Programs
# ---------------------------------------------------------------------------

class ChangeKind(Enum):
    """Kind of change between two program versions."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class FunctionChange:
    """Description of a change to a function."""
    name: str
    kind: ChangeKind
    old_source: Optional[str] = None
    new_source: Optional[str] = None


@dataclass
class ProgramDiff:
    """Diff between two versions of a C10 program."""
    old_source: str
    new_source: str
    function_changes: List[FunctionChange] = field(default_factory=list)
    # Top-level statement changes
    toplevel_changed: bool = False
    # Functions present in each version
    old_functions: Set[str] = field(default_factory=set)
    new_functions: Set[str] = field(default_factory=set)

    @property
    def changed_functions(self) -> Set[str]:
        return {fc.name for fc in self.function_changes
                if fc.kind != ChangeKind.UNCHANGED}

    @property
    def unchanged_functions(self) -> Set[str]:
        return {fc.name for fc in self.function_changes
                if fc.kind == ChangeKind.UNCHANGED}

    @property
    def has_changes(self) -> bool:
        return self.toplevel_changed or len(self.changed_functions) > 0


def _parse_c10(source: str) -> List:
    """Parse C10 source into statement list."""
    tokens = lex(source)
    return Parser(tokens).parse().stmts


def _extract_functions(stmts) -> Dict[str, FnDecl]:
    """Extract function declarations from statement list."""
    fns = {}
    for stmt in stmts:
        if isinstance(stmt, FnDecl):
            fns[stmt.name] = stmt
    return fns


def _stmt_signature(stmt) -> str:
    """Generate a structural signature for a statement (for diff comparison)."""
    if isinstance(stmt, LetDecl):
        return f"let:{stmt.name}={_expr_sig(stmt.value)}"
    elif isinstance(stmt, Assign):
        return f"assign:{stmt.name}={_expr_sig(stmt.value)}"
    elif isinstance(stmt, IfStmt):
        then_sigs = [_stmt_signature(s) for s in _block_stmts(stmt.then_body)]
        else_sigs = [_stmt_signature(s) for s in _block_stmts(stmt.else_body)] if stmt.else_body else []
        return f"if:{_expr_sig(stmt.cond)}:{';'.join(then_sigs)}:{';'.join(else_sigs)}"
    elif isinstance(stmt, WhileStmt):
        body_sigs = [_stmt_signature(s) for s in _block_stmts(stmt.body)]
        return f"while:{_expr_sig(stmt.cond)}:{';'.join(body_sigs)}"
    elif isinstance(stmt, ReturnStmt):
        return f"return:{_expr_sig(stmt.value)}"
    elif isinstance(stmt, PrintStmt):
        return f"print:{_expr_sig(stmt.value)}"
    elif isinstance(stmt, FnDecl):
        params = ",".join(stmt.params)
        body_sigs = [_stmt_signature(s) for s in _block_stmts(stmt.body)]
        return f"fn:{stmt.name}({params}):{';'.join(body_sigs)}"
    else:
        return f"unknown:{type(stmt).__name__}"


def _expr_sig(expr) -> str:
    """Generate a structural signature for an expression."""
    if expr is None:
        return "none"
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, BinOp):
        return f"({_expr_sig(expr.left)}{expr.op}{_expr_sig(expr.right)})"
    if isinstance(expr, CallExpr):
        args = ",".join(_expr_sig(a) for a in expr.args)
        return f"{expr.callee}({args})"
    return f"?{type(expr).__name__}"


def _block_stmts(body) -> list:
    """Extract statements from a block (handle Block object or list)."""
    if body is None:
        return []
    if isinstance(body, Block):
        return body.stmts
    if isinstance(body, list):
        return body
    return [body]


def _fn_signature(fn: FnDecl) -> str:
    """Full structural signature of a function."""
    params = ",".join(fn.params)
    body_sigs = [_stmt_signature(s) for s in _block_stmts(fn.body)]
    return f"fn:{fn.name}({params}):{';'.join(body_sigs)}"


def diff_programs(old_source: str, new_source: str) -> ProgramDiff:
    """Compute the diff between two C10 program versions.

    Identifies which functions were added, removed, modified, or unchanged.
    """
    old_stmts = _parse_c10(old_source)
    new_stmts = _parse_c10(new_source)

    old_fns = _extract_functions(old_stmts)
    new_fns = _extract_functions(new_stmts)

    changes = []

    # Check each function in the old version
    for name, old_fn in old_fns.items():
        if name not in new_fns:
            changes.append(FunctionChange(
                name=name,
                kind=ChangeKind.REMOVED,
                old_source=_fn_signature(old_fn),
            ))
        else:
            new_fn = new_fns[name]
            old_sig = _fn_signature(old_fn)
            new_sig = _fn_signature(new_fn)
            if old_sig == new_sig:
                changes.append(FunctionChange(
                    name=name,
                    kind=ChangeKind.UNCHANGED,
                ))
            else:
                changes.append(FunctionChange(
                    name=name,
                    kind=ChangeKind.MODIFIED,
                    old_source=old_sig,
                    new_source=new_sig,
                ))

    # Check for added functions
    for name in new_fns:
        if name not in old_fns:
            changes.append(FunctionChange(
                name=name,
                kind=ChangeKind.ADDED,
                new_source=_fn_signature(new_fns[name]),
            ))

    # Check top-level statements (non-function)
    old_toplevel = [_stmt_signature(s) for s in old_stmts if not isinstance(s, FnDecl)]
    new_toplevel = [_stmt_signature(s) for s in new_stmts if not isinstance(s, FnDecl)]
    toplevel_changed = old_toplevel != new_toplevel

    return ProgramDiff(
        old_source=old_source,
        new_source=new_source,
        function_changes=changes,
        toplevel_changed=toplevel_changed,
        old_functions=set(old_fns.keys()),
        new_functions=set(new_fns.keys()),
    )


# ---------------------------------------------------------------------------
# Certificate Cache
# ---------------------------------------------------------------------------

@dataclass
class CertificateCache:
    """Cache of proof certificates for verified functions."""
    # function_name -> (source_signature, certificate)
    entries: Dict[str, Tuple[str, ProofCertificate]] = field(default_factory=dict)

    def get(self, fn_name: str) -> Optional[ProofCertificate]:
        """Get cached certificate for a function."""
        entry = self.entries.get(fn_name)
        return entry[1] if entry else None

    def put(self, fn_name: str, signature: str, cert: ProofCertificate):
        """Cache a certificate for a function."""
        self.entries[fn_name] = (signature, cert)

    def invalidate(self, fn_name: str):
        """Remove a cached certificate."""
        self.entries.pop(fn_name, None)

    def has_valid(self, fn_name: str, signature: str) -> bool:
        """Check if cache has a valid certificate matching the signature."""
        entry = self.entries.get(fn_name)
        if entry is None:
            return False
        cached_sig, cert = entry
        return cached_sig == signature and cert.status == CertStatus.VALID

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def valid_count(self) -> int:
        return sum(1 for _, (_, c) in self.entries.items()
                   if c.status == CertStatus.VALID)


# ---------------------------------------------------------------------------
# Incremental Verification Engine
# ---------------------------------------------------------------------------

@dataclass
class IncrementalResult:
    """Result of incremental verification."""
    certificate: ProofCertificate
    diff: Optional[ProgramDiff] = None
    # Statistics
    functions_reverified: Set[str] = field(default_factory=set)
    functions_reused: Set[str] = field(default_factory=set)
    functions_skipped: Set[str] = field(default_factory=set)  # removed functions
    total_functions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def is_valid(self) -> bool:
        return self.certificate.status == CertStatus.VALID

    @property
    def summary(self) -> str:
        parts = [
            f"Status: {self.certificate.status.value}",
            f"Functions: {self.total_functions}",
            f"Re-verified: {len(self.functions_reverified)}",
            f"Reused: {len(self.functions_reused)}",
            f"Cache: {self.cache_hits} hits, {self.cache_misses} misses",
            f"Obligations: {self.certificate.valid_obligations}/{self.certificate.total_obligations}",
        ]
        return " | ".join(parts)


class IncrementalVerifier:
    """Incremental verification engine.

    Maintains a certificate cache and uses AST-level diffing to
    determine which functions need re-verification.
    """

    def __init__(self):
        self.cache = CertificateCache()
        self._last_source: Optional[str] = None

    def verify(self, source: str) -> IncrementalResult:
        """Verify a program, using cached certificates where possible.

        If this is the first verification, verify everything.
        If a previous version was verified, diff and re-verify only changes.
        """
        if self._last_source is None:
            result = self._verify_full(source)
        else:
            result = self._verify_incremental(self._last_source, source)

        self._last_source = source
        return result

    def verify_against(self, old_source: str, new_source: str) -> IncrementalResult:
        """Verify new_source incrementally against old_source.

        Does not update internal state (cache, last_source).
        """
        # First ensure old version is in cache
        self._ensure_cached(old_source)
        return self._verify_incremental(old_source, new_source)

    def _verify_full(self, source: str) -> IncrementalResult:
        """Full verification of all functions."""
        stmts = _parse_c10(source)
        fns = _extract_functions(stmts)

        sub_certs = []
        reverified = set()

        for fn_name, fn_decl in fns.items():
            cert = self._verify_function(source, fn_name)
            sub_certs.append(cert)
            reverified.add(fn_name)

            # Cache the result
            sig = _fn_signature(fn_decl)
            self.cache.put(fn_name, sig, cert)

        # If no functions, create a trivial certificate
        if not sub_certs:
            composite = ProofCertificate(
                kind=ProofKind.VCGEN,
                claim="No functions to verify",
                source=source,
                status=CertStatus.VALID,
            )
        elif len(sub_certs) == 1:
            composite = sub_certs[0]
        else:
            composite = combine_certificates(
                *sub_certs,
                claim=f"Full verification of {len(fns)} functions",
            )
            # Compute composite status
            if all(c.status == CertStatus.VALID for c in sub_certs):
                composite.status = CertStatus.VALID
            elif any(c.status == CertStatus.INVALID for c in sub_certs):
                composite.status = CertStatus.INVALID
            else:
                composite.status = CertStatus.UNKNOWN

        return IncrementalResult(
            certificate=composite,
            functions_reverified=reverified,
            total_functions=len(fns),
            cache_misses=len(fns),
        )

    def _verify_incremental(
        self, old_source: str, new_source: str,
    ) -> IncrementalResult:
        """Incremental verification using diff."""
        diff = diff_programs(old_source, new_source)
        new_stmts = _parse_c10(new_source)
        new_fns = _extract_functions(new_stmts)

        sub_certs = []
        reverified = set()
        reused = set()
        skipped = set()
        cache_hits = 0
        cache_misses = 0

        for change in diff.function_changes:
            if change.kind == ChangeKind.UNCHANGED:
                # Try to reuse cached certificate
                cached = self.cache.get(change.name)
                if cached and cached.status == CertStatus.VALID:
                    sub_certs.append(cached)
                    reused.add(change.name)
                    cache_hits += 1
                else:
                    # Cache miss -- re-verify
                    cert = self._verify_function(new_source, change.name)
                    sub_certs.append(cert)
                    reverified.add(change.name)
                    cache_misses += 1
                    if change.name in new_fns:
                        self.cache.put(change.name, _fn_signature(new_fns[change.name]), cert)

            elif change.kind == ChangeKind.MODIFIED:
                # Must re-verify
                cert = self._verify_function(new_source, change.name)
                sub_certs.append(cert)
                reverified.add(change.name)
                cache_misses += 1
                if change.name in new_fns:
                    self.cache.put(change.name, _fn_signature(new_fns[change.name]), cert)

            elif change.kind == ChangeKind.ADDED:
                # New function -- verify
                cert = self._verify_function(new_source, change.name)
                sub_certs.append(cert)
                reverified.add(change.name)
                cache_misses += 1
                if change.name in new_fns:
                    self.cache.put(change.name, _fn_signature(new_fns[change.name]), cert)

            elif change.kind == ChangeKind.REMOVED:
                # Invalidate cache
                self.cache.invalidate(change.name)
                skipped.add(change.name)

        # Build composite
        if not sub_certs:
            composite = ProofCertificate(
                kind=ProofKind.VCGEN,
                claim="No functions to verify",
                source=new_source,
                status=CertStatus.VALID,
            )
        elif len(sub_certs) == 1:
            composite = sub_certs[0]
        else:
            composite = combine_certificates(
                *sub_certs,
                claim=f"Incremental verification: {len(reverified)} re-verified, {len(reused)} reused",
            )
            if all(c.status == CertStatus.VALID for c in sub_certs):
                composite.status = CertStatus.VALID
            elif any(c.status == CertStatus.INVALID for c in sub_certs):
                composite.status = CertStatus.INVALID
            else:
                composite.status = CertStatus.UNKNOWN

        return IncrementalResult(
            certificate=composite,
            diff=diff,
            functions_reverified=reverified,
            functions_reused=reused,
            functions_skipped=skipped,
            total_functions=len(new_fns),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    def _verify_function(self, source: str, fn_name: str) -> ProofCertificate:
        """Verify a single function and return its certificate."""
        try:
            cert = generate_vcgen_certificate(source, fn_name=fn_name)
            cert = check_certificate(cert)
            return cert
        except Exception as e:
            return ProofCertificate(
                kind=ProofKind.VCGEN,
                claim=f"Verification of {fn_name} failed: {e}",
                source=source,
                status=CertStatus.UNKNOWN,
                metadata={"error": str(e), "function": fn_name},
            )

    def _ensure_cached(self, source: str):
        """Ensure all functions in source are cached."""
        stmts = _parse_c10(source)
        fns = _extract_functions(stmts)
        for fn_name, fn_decl in fns.items():
            sig = _fn_signature(fn_decl)
            if not self.cache.has_valid(fn_name, sig):
                cert = self._verify_function(source, fn_name)
                self.cache.put(fn_name, sig, cert)


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------

def incremental_verify(
    old_source: str,
    new_source: str,
) -> IncrementalResult:
    """One-shot incremental verification of two program versions.

    Verifies old_source fully, then incrementally verifies new_source.
    """
    verifier = IncrementalVerifier()
    verifier._verify_full(old_source)
    verifier._last_source = old_source
    return verifier.verify(new_source)


def diff_and_report(old_source: str, new_source: str) -> str:
    """Generate a human-readable diff report."""
    diff = diff_programs(old_source, new_source)
    lines = [f"Program diff: {len(diff.function_changes)} function(s) analyzed"]
    for change in diff.function_changes:
        lines.append(f"  {change.name}: {change.kind.value}")
    if diff.toplevel_changed:
        lines.append("  Top-level statements: changed")
    lines.append(f"Changed: {diff.changed_functions}")
    lines.append(f"Unchanged: {diff.unchanged_functions}")
    return "\n".join(lines)


def verify_with_cache(
    sources: List[str],
) -> List[IncrementalResult]:
    """Verify a sequence of program versions incrementally.

    Each version is verified against the previous one, building up
    a cache of certificates that are reused across versions.
    """
    verifier = IncrementalVerifier()
    results = []
    for source in sources:
        result = verifier.verify(source)
        results.append(result)
    return results
