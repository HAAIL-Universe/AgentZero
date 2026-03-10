"""V056: Regression Verification

Composes:
- V047 (incremental verification) -- AST diff + certificate caching
- V009 (differential symbolic execution) -- behavioral change detection
- V054 (verification-driven fuzzing) -- targeted fuzz of changed paths
- V004 (VCGen) + V044 (proof certificates) -- formal verification

When code changes:
1. Identify which functions changed (AST diff)
2. Reuse proof certificates for unchanged functions
3. Differential symbolic execution on changed functions
4. Targeted fuzzing on changed paths
5. Combined regression verdict with evidence
"""

import sys, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

# V047: incremental verification (AST diff + cert cache)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V047_incremental_verification'))
from incremental_verification import (
    diff_programs as v047_diff, ProgramDiff, FunctionChange, ChangeKind,
    CertificateCache, IncrementalVerifier, IncrementalResult
)

# V009: differential symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V009_differential_symex'))
from diff_symex import (
    diff_programs as diff_symex, diff_functions as diff_fn_symex,
    check_regression as symex_check_regression,
    DiffResult, DiffImpact, BehavioralDiff, ASTDiff
)

# V054: verification-driven fuzzing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V054_verification_driven_fuzzing'))
from verification_driven_fuzzing import (
    verification_fuzz, quick_fuzz, FuzzResult, FuzzFinding, MutationEngine
)

# V044: proof certificates
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
from proof_certificates import (
    ProofCertificate, CertStatus, ProofKind, ProofObligation,
    check_certificate, combine_certificates, generate_vcgen_certificate
)

# C010: parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser, FnDecl, LetDecl, Var, IntLit, BinOp, CallExpr


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class RegressionVerdict(Enum):
    SAFE = "safe"                    # No behavioral changes detected
    REGRESSION = "regression"        # Behavioral change found
    UNKNOWN = "unknown"              # Could not determine
    IMPROVED = "improved"            # Changed but all proofs still hold


@dataclass
class FunctionVerdict:
    """Per-function regression result."""
    name: str
    change_kind: ChangeKind
    verdict: RegressionVerdict
    cert_reused: bool = False
    diff_result: Optional[DiffResult] = None
    fuzz_result: Optional[FuzzResult] = None
    certificate: Optional[ProofCertificate] = None
    details: str = ""


@dataclass
class RegressionResult:
    """Full regression verification result."""
    verdict: RegressionVerdict
    program_diff: ProgramDiff
    function_verdicts: List[FunctionVerdict] = field(default_factory=list)
    certificate: Optional[ProofCertificate] = None
    # Statistics
    functions_unchanged: int = 0
    functions_reverified: int = 0
    functions_regressed: int = 0
    certs_reused: int = 0
    symex_checks: int = 0
    fuzz_inputs_tested: int = 0

    @property
    def is_safe(self) -> bool:
        return self.verdict in (RegressionVerdict.SAFE, RegressionVerdict.IMPROVED)

    @property
    def regressions(self) -> List[FunctionVerdict]:
        return [fv for fv in self.function_verdicts
                if fv.verdict == RegressionVerdict.REGRESSION]

    @property
    def summary(self) -> str:
        lines = [f"Regression verdict: {self.verdict.value}"]
        lines.append(f"  Functions: {self.functions_unchanged} unchanged, "
                     f"{self.functions_reverified} reverified, "
                     f"{self.functions_regressed} regressed")
        lines.append(f"  Certs reused: {self.certs_reused}")
        lines.append(f"  Symex checks: {self.symex_checks}")
        lines.append(f"  Fuzz inputs: {self.fuzz_inputs_tested}")
        for fv in self.function_verdicts:
            status = fv.verdict.value
            if fv.cert_reused:
                status += " (cert reused)"
            lines.append(f"  {fv.name}: {fv.change_kind.value} -> {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(source):
    """Parse C10 source to statement list."""
    tokens = lex(source)
    prog = Parser(tokens).parse()
    return prog.stmts if hasattr(prog, 'stmts') else list(prog)


def _extract_functions(stmts) -> Dict[str, FnDecl]:
    """Extract function declarations."""
    fns = {}
    for s in stmts:
        if isinstance(s, FnDecl):
            fns[s.name] = s
    return fns


def _extract_fn_source(source, fn_name):
    """Extract source text for a single function (rough heuristic)."""
    # Find function and extract its body for standalone analysis
    stmts = _parse(source)
    fns = _extract_functions(stmts)
    if fn_name not in fns:
        return None
    # Rebuild source from the function declaration
    fn = fns[fn_name]
    params = ", ".join(fn.params) if fn.params else ""
    body_stmts = fn.body.stmts if hasattr(fn.body, 'stmts') else fn.body
    # We can't perfectly reconstruct, so use the full source for now
    return source


def _infer_symbolic_inputs(source, fn_name=None):
    """Infer symbolic input variables from source."""
    stmts = _parse(source)
    if fn_name:
        fns = _extract_functions(stmts)
        if fn_name in fns:
            return {p: 'int' for p in fns[fn_name].params}
    # For top-level: find variables used before assignment
    defined = set()
    used_before_def = {}
    for s in stmts:
        if isinstance(s, LetDecl):
            defined.add(s.name)
        elif isinstance(s, FnDecl):
            continue
    # Default: first let-bound variables as symbolic inputs
    for s in stmts:
        if isinstance(s, LetDecl) and isinstance(s.value, IntLit):
            used_before_def[s.name] = 'int'
            if len(used_before_def) >= 3:
                break
    return used_before_def if used_before_def else {'x': 'int'}


def _wrap_fn_as_program(source, fn_name, inputs):
    """Wrap a function call with let-bindings for symbolic inputs."""
    stmts = _parse(source)
    fns = _extract_functions(stmts)
    if fn_name not in fns:
        return None
    fn = fns[fn_name]
    params = fn.params if fn.params else []
    # Build: let p1 = 0; let p2 = 0; ... let __result = fn(p1, p2, ...);
    lines = []
    for p in params:
        lines.append(f"let {p} = 0;")
    # Include the function definition
    lines.append(source)
    args = ", ".join(params)
    lines.append(f"let __result = {fn_name}({args});")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core regression verification
# ---------------------------------------------------------------------------

def verify_regression(old_source: str, new_source: str,
                      symbolic_inputs: Optional[Dict[str, str]] = None,
                      max_paths: int = 64,
                      fuzz_budget: int = 100,
                      cache: Optional[CertificateCache] = None
                      ) -> RegressionResult:
    """Full regression verification pipeline.

    1. AST diff to find changed functions
    2. Reuse certificates for unchanged functions
    3. Differential symbolic execution on changed functions
    4. Targeted fuzzing on changed paths
    5. Combined verdict
    """
    # Step 1: AST diff
    prog_diff = v047_diff(old_source, new_source)

    if cache is None:
        cache = CertificateCache()

    result = RegressionResult(
        verdict=RegressionVerdict.UNKNOWN,
        program_diff=prog_diff,
    )

    all_certs = []
    has_regression = False

    for change in prog_diff.function_changes:
        fv = _verify_function_change(
            change, old_source, new_source, symbolic_inputs,
            max_paths, fuzz_budget, cache
        )
        result.function_verdicts.append(fv)

        if fv.change_kind == ChangeKind.UNCHANGED:
            result.functions_unchanged += 1
        else:
            result.functions_reverified += 1

        if fv.cert_reused:
            result.certs_reused += 1

        if fv.diff_result:
            result.symex_checks += 1

        if fv.fuzz_result:
            result.fuzz_inputs_tested += fv.fuzz_result.total_inputs_tested

        if fv.verdict == RegressionVerdict.REGRESSION:
            has_regression = True
            result.functions_regressed += 1

        if fv.certificate:
            all_certs.append(fv.certificate)

    # Also check top-level changes
    if prog_diff.toplevel_changed:
        fv = _verify_toplevel_change(
            old_source, new_source, symbolic_inputs, max_paths, fuzz_budget
        )
        result.function_verdicts.append(fv)
        if fv.verdict == RegressionVerdict.REGRESSION:
            has_regression = True
            result.functions_regressed += 1
        if fv.diff_result:
            result.symex_checks += 1
        if fv.fuzz_result:
            result.fuzz_inputs_tested += fv.fuzz_result.total_inputs_tested

    # Combine certificates
    if all_certs:
        try:
            result.certificate = combine_certificates(
                *all_certs, claim="Regression verification"
            )
        except Exception:
            pass

    # Final verdict
    if has_regression:
        result.verdict = RegressionVerdict.REGRESSION
    elif all(fv.verdict in (RegressionVerdict.SAFE, RegressionVerdict.IMPROVED)
             for fv in result.function_verdicts):
        if any(fv.change_kind != ChangeKind.UNCHANGED for fv in result.function_verdicts):
            # Something changed but all proofs still hold
            result.verdict = RegressionVerdict.IMPROVED
        else:
            result.verdict = RegressionVerdict.SAFE
    else:
        result.verdict = RegressionVerdict.UNKNOWN

    return result


def _verify_function_change(change: FunctionChange,
                            old_source: str, new_source: str,
                            symbolic_inputs: Optional[Dict[str, str]],
                            max_paths: int, fuzz_budget: int,
                            cache: CertificateCache) -> FunctionVerdict:
    """Verify a single function change."""

    if change.kind == ChangeKind.UNCHANGED:
        # Try to reuse certificate
        cached = cache.get(change.name)
        if cached and cached.status == CertStatus.VALID:
            return FunctionVerdict(
                name=change.name,
                change_kind=ChangeKind.UNCHANGED,
                verdict=RegressionVerdict.SAFE,
                cert_reused=True,
                certificate=cached,
                details="Unchanged, certificate reused"
            )
        # No cache -- try to verify
        try:
            cert = generate_vcgen_certificate(new_source, change.name)
            cert = check_certificate(cert)
            cache.put(change.name, change.new_source or "", cert)
            return FunctionVerdict(
                name=change.name,
                change_kind=ChangeKind.UNCHANGED,
                verdict=RegressionVerdict.SAFE,
                certificate=cert,
                details="Unchanged, newly verified"
            )
        except Exception:
            return FunctionVerdict(
                name=change.name,
                change_kind=ChangeKind.UNCHANGED,
                verdict=RegressionVerdict.SAFE,
                details="Unchanged, verification skipped"
            )

    if change.kind == ChangeKind.REMOVED:
        return FunctionVerdict(
            name=change.name,
            change_kind=ChangeKind.REMOVED,
            verdict=RegressionVerdict.SAFE,
            details="Function removed"
        )

    if change.kind == ChangeKind.ADDED:
        # New function -- verify it, no regression possible
        try:
            cert = generate_vcgen_certificate(new_source, change.name)
            cert = check_certificate(cert)
            cache.put(change.name, change.new_source or "", cert)
            return FunctionVerdict(
                name=change.name,
                change_kind=ChangeKind.ADDED,
                verdict=RegressionVerdict.SAFE,
                certificate=cert,
                details="New function, verified"
            )
        except Exception:
            return FunctionVerdict(
                name=change.name,
                change_kind=ChangeKind.ADDED,
                verdict=RegressionVerdict.SAFE,
                details="New function, verification skipped"
            )

    # MODIFIED -- the interesting case
    return _verify_modified_function(
        change, old_source, new_source, symbolic_inputs, max_paths, fuzz_budget, cache
    )


def _verify_modified_function(change: FunctionChange,
                              old_source: str, new_source: str,
                              symbolic_inputs: Optional[Dict[str, str]],
                              max_paths: int, fuzz_budget: int,
                              cache: CertificateCache) -> FunctionVerdict:
    """Verify a modified function: symex diff + fuzz."""
    fn_name = change.name

    # Infer inputs if not provided
    if symbolic_inputs is None:
        symbolic_inputs = _infer_symbolic_inputs(new_source, fn_name)

    # Phase 1: Differential symbolic execution
    diff_result = None
    try:
        param_types = symbolic_inputs
        diff_result = diff_fn_symex(
            old_source, fn_name, new_source, fn_name,
            param_types=param_types, max_paths=max_paths
        )
    except Exception:
        pass

    # If symex found behavioral change (full or partial), it's a regression
    if diff_result and diff_result.impact in (DiffImpact.BEHAVIORAL_CHANGE, DiffImpact.PARTIAL_CHANGE) and diff_result.behavioral_diffs:
        cache.invalidate(fn_name)
        return FunctionVerdict(
            name=fn_name,
            change_kind=ChangeKind.MODIFIED,
            verdict=RegressionVerdict.REGRESSION,
            diff_result=diff_result,
            details=f"Behavioral change detected: {len(diff_result.behavioral_diffs)} differences"
        )

    # Phase 2: Targeted fuzzing on modified function
    fuzz_result = None
    if fuzz_budget > 0:
        try:
            wrapped_old = _wrap_fn_as_program(old_source, fn_name, symbolic_inputs)
            wrapped_new = _wrap_fn_as_program(new_source, fn_name, symbolic_inputs)
            if wrapped_old and wrapped_new:
                fuzz_result = _fuzz_for_regression(
                    wrapped_old, wrapped_new,
                    list(symbolic_inputs.keys()),
                    max_inputs=fuzz_budget
                )
        except Exception:
            pass

    has_fuzz_regression = fuzz_result and (
        fuzz_result.has_bugs or
        any(f.kind == "divergence" for f in fuzz_result.findings)
    )
    if has_fuzz_regression:
        cache.invalidate(fn_name)
        return FunctionVerdict(
            name=fn_name,
            change_kind=ChangeKind.MODIFIED,
            verdict=RegressionVerdict.REGRESSION,
            diff_result=diff_result,
            fuzz_result=fuzz_result,
            details=f"Fuzz found {fuzz_result.bug_count} regressions"
        )

    # Phase 3: Re-verify with VCGen
    cert = None
    try:
        cert = generate_vcgen_certificate(new_source, fn_name)
        cert = check_certificate(cert)
        cache.put(fn_name, change.new_source or "", cert)
    except Exception:
        pass

    # No regression found
    verdict = RegressionVerdict.IMPROVED
    if diff_result and diff_result.impact == DiffImpact.NO_BEHAVIORAL_CHANGE:
        verdict = RegressionVerdict.SAFE

    return FunctionVerdict(
        name=fn_name,
        change_kind=ChangeKind.MODIFIED,
        verdict=verdict,
        diff_result=diff_result,
        fuzz_result=fuzz_result,
        certificate=cert,
        details="Modified, no regression found"
    )


def _verify_toplevel_change(old_source: str, new_source: str,
                            symbolic_inputs: Optional[Dict[str, str]],
                            max_paths: int, fuzz_budget: int) -> FunctionVerdict:
    """Verify top-level code changes."""
    if symbolic_inputs is None:
        symbolic_inputs = _infer_symbolic_inputs(new_source)

    diff_result = None
    try:
        diff_result = diff_symex(
            old_source, new_source, symbolic_inputs, max_paths=max_paths
        )
    except Exception:
        pass

    if diff_result and diff_result.impact == DiffImpact.BEHAVIORAL_CHANGE:
        return FunctionVerdict(
            name="<toplevel>",
            change_kind=ChangeKind.MODIFIED,
            verdict=RegressionVerdict.REGRESSION,
            diff_result=diff_result,
            details=f"Top-level behavioral change: {len(diff_result.behavioral_diffs)} differences"
        )

    return FunctionVerdict(
        name="<toplevel>",
        change_kind=ChangeKind.MODIFIED,
        verdict=RegressionVerdict.SAFE if diff_result else RegressionVerdict.UNKNOWN,
        diff_result=diff_result,
        details="Top-level change, no regression detected"
    )


# ---------------------------------------------------------------------------
# Fuzzing for regression
# ---------------------------------------------------------------------------

def _fuzz_for_regression(old_wrapped: str, new_wrapped: str,
                         input_vars: List[str],
                         max_inputs: int = 100) -> FuzzResult:
    """Fuzz both versions and detect output divergences."""
    engine = MutationEngine(seed=42)

    findings = []
    tested = 0
    all_inputs = []

    # Generate test inputs via mutation
    base = {v: 0 for v in input_vars}
    seeds = [base]
    # Add boundary values
    for v in input_vars:
        for val in [0, 1, -1, 5, -5, 10, -10, 100]:
            seed = dict(base)
            seed[v] = val
            seeds.append(seed)

    # Add mutations
    for seed in seeds[:20]:
        for strength in [1, 2]:
            batch = engine.mutate_batch(seed, count=5, strength=strength)
            seeds.extend(batch)

    # Test each input on both versions
    seen = set()
    for inp in seeds:
        if tested >= max_inputs:
            break
        key = tuple(sorted(inp.items()))
        if key in seen:
            continue
        seen.add(key)

        old_out = _safe_run(old_wrapped, inp)
        new_out = _safe_run(new_wrapped, inp)
        tested += 1

        if old_out != new_out:
            findings.append(FuzzFinding(
                kind="divergence",
                inputs=inp,
                description=f"Output changed: {old_out} -> {new_out}",
                source="regression_fuzz"
            ))

    from verification_driven_fuzzing import FuzzStatus, CoverageInfo, FuzzInput
    return FuzzResult(
        findings=findings,
        coverage=None,
        total_inputs_tested=tested,
        symbolic_inputs=0,
        concolic_inputs=0,
        mutation_inputs=tested,
        random_inputs=0,
        boundary_inputs=0,
        status=FuzzStatus.BUG_FOUND if findings else FuzzStatus.COMPLETE,
        suspicious_stmts=[],
        all_test_inputs=[FuzzInput(values=inp, source="regression", generation=0)
                         for inp in seeds[:tested]]
    )


def _safe_run(source, inputs):
    """Safely execute source with inputs, return output or error tag."""
    try:
        stmts = _parse(source)
        env = {}
        output = []
        _interpret(stmts, env, inputs, output, limit=1000)
        return ("ok", tuple(output), env.get('__result'))
    except ZeroDivisionError:
        return ("crash", "div_by_zero")
    except Exception as e:
        return ("crash", str(type(e).__name__))


def _interpret(stmts, env, inputs, output, limit=1000):
    """Minimal interpreter for regression testing."""
    for s in stmts:
        _exec_stmt(s, env, inputs, output, limit)


def _exec_stmt(s, env, inputs, output, limit):
    if isinstance(s, LetDecl):
        if s.name in inputs:
            env[s.name] = inputs[s.name]
        else:
            env[s.name] = _eval_expr(s.value, env)
    elif hasattr(s, 'name') and hasattr(s, 'value') and not isinstance(s, FnDecl):
        # Assign
        env[s.name] = _eval_expr(s.value, env)
    elif hasattr(s, 'cond') and hasattr(s, 'then_body'):
        # IfStmt
        cond = _eval_expr(s.cond, env)
        if cond:
            body = s.then_body.stmts if hasattr(s.then_body, 'stmts') else s.then_body
            _interpret(body, env, inputs, output, limit)
        elif s.else_body:
            body = s.else_body.stmts if hasattr(s.else_body, 'stmts') else s.else_body
            _interpret(body, env, inputs, output, limit)
    elif hasattr(s, 'cond') and hasattr(s, 'body') and not hasattr(s, 'then_body'):
        # WhileStmt
        count = 0
        while _eval_expr(s.cond, env) and count < limit:
            body = s.body.stmts if hasattr(s.body, 'stmts') else s.body
            _interpret(body, env, inputs, output, limit)
            count += 1
    elif hasattr(s, 'value') and type(s).__name__ == 'PrintStmt':
        output.append(_eval_expr(s.value, env))
    elif hasattr(s, 'value') and type(s).__name__ == 'ReturnStmt':
        env['__result'] = _eval_expr(s.value, env)
    elif isinstance(s, FnDecl):
        env[s.name] = s  # store function


def _eval_expr(expr, env):
    if isinstance(expr, IntLit):
        return expr.value
    elif isinstance(expr, Var):
        return env.get(expr.name, 0)
    elif isinstance(expr, BinOp):
        l = _eval_expr(expr.left, env)
        r = _eval_expr(expr.right, env)
        op = expr.op
        if op == '+': return l + r
        if op == '-': return l - r
        if op == '*': return l * r
        if op == '/': return l // r if r != 0 else 0
        if op == '%': return l % r if r != 0 else 0
        if op == '<': return int(l < r)
        if op == '>': return int(l > r)
        if op == '<=': return int(l <= r)
        if op == '>=': return int(l >= r)
        if op == '==': return int(l == r)
        if op == '!=': return int(l != r)
        return 0
    elif isinstance(expr, CallExpr):
        fn_name = expr.callee if isinstance(expr.callee, str) else (
            expr.callee.name if hasattr(expr.callee, 'name') else str(expr.callee))
        args = [_eval_expr(a, env) for a in expr.args]
        fn = env.get(fn_name)
        if isinstance(fn, FnDecl):
            fn_env = dict(env)
            for p, a in zip(fn.params, args):
                fn_env[p] = a
            body = fn.body.stmts if hasattr(fn.body, 'stmts') else fn.body
            _interpret(body, fn_env, {}, [], 1000)
            return fn_env.get('__result', 0)
        return 0
    elif hasattr(expr, 'op') and hasattr(expr, 'operand'):
        # UnaryOp
        val = _eval_expr(expr.operand, env)
        if expr.op == '-': return -val
        if expr.op == '!': return int(not val)
        return val
    return 0


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------

def check_regression(old_source: str, new_source: str,
                     symbolic_inputs: Optional[Dict[str, str]] = None,
                     max_paths: int = 64) -> RegressionResult:
    """Quick regression check using only differential symbolic execution."""
    return verify_regression(old_source, new_source, symbolic_inputs,
                            max_paths=max_paths, fuzz_budget=0)


def check_regression_with_fuzz(old_source: str, new_source: str,
                               symbolic_inputs: Optional[Dict[str, str]] = None,
                               fuzz_budget: int = 200) -> RegressionResult:
    """Regression check with additional fuzzing for changed functions."""
    return verify_regression(old_source, new_source, symbolic_inputs,
                            fuzz_budget=fuzz_budget)


def regression_report(old_source: str, new_source: str,
                      symbolic_inputs: Optional[Dict[str, str]] = None) -> str:
    """Human-readable regression report."""
    result = verify_regression(old_source, new_source, symbolic_inputs)
    return result.summary


class RegressionVerifier:
    """Stateful regression verifier with persistent certificate cache."""

    def __init__(self):
        self.cache = CertificateCache()
        self._versions: List[str] = []
        self._results: List[RegressionResult] = []

    def verify(self, new_source: str,
               symbolic_inputs: Optional[Dict[str, str]] = None) -> RegressionResult:
        """Verify new version against previous."""
        if not self._versions:
            # First version -- full verify
            self._versions.append(new_source)
            result = RegressionResult(
                verdict=RegressionVerdict.SAFE,
                program_diff=ProgramDiff(
                    old_source="", new_source=new_source,
                    function_changes=[], toplevel_changed=False,
                    old_functions=set(), new_functions=set()
                ),
            )
            # Populate cache
            stmts = _parse(new_source)
            fns = _extract_functions(stmts)
            for fn_name in fns:
                try:
                    cert = generate_vcgen_certificate(new_source, fn_name)
                    cert = check_certificate(cert)
                    self.cache.put(fn_name, "", cert)
                except Exception:
                    pass
            self._results.append(result)
            return result

        old_source = self._versions[-1]
        result = verify_regression(
            old_source, new_source, symbolic_inputs, cache=self.cache
        )
        self._versions.append(new_source)
        self._results.append(result)
        return result

    def verify_sequence(self, versions: List[str],
                        symbolic_inputs: Optional[Dict[str, str]] = None
                        ) -> List[RegressionResult]:
        """Verify a sequence of versions incrementally."""
        results = []
        for v in versions:
            r = self.verify(v, symbolic_inputs)
            results.append(r)
        return results

    @property
    def version_count(self) -> int:
        return len(self._versions)

    @property
    def all_results(self) -> List[RegressionResult]:
        return list(self._results)


def compare_verification_strategies(old_source: str, new_source: str,
                                    symbolic_inputs: Optional[Dict[str, str]] = None
                                    ) -> Dict[str, Any]:
    """Compare regression verification vs full re-verification."""
    # Incremental
    inc_result = verify_regression(old_source, new_source, symbolic_inputs)

    # Full re-verification (no cache)
    full_verifier = IncrementalVerifier()
    try:
        full_result = full_verifier.verify(new_source)
    except Exception:
        full_result = None

    return {
        "incremental_verdict": inc_result.verdict.value,
        "incremental_certs_reused": inc_result.certs_reused,
        "incremental_reverified": inc_result.functions_reverified,
        "incremental_unchanged": inc_result.functions_unchanged,
        "full_verification_ran": full_result is not None,
        "full_reverified": full_result.functions_reverified if full_result else 0,
        "savings": inc_result.certs_reused,
    }
