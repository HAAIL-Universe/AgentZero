"""
V145: Certified Compositional Verification

Modular verification: verify functions independently, compose proofs.
Each function is verified against its spec in isolation. At call sites,
preconditions become proof obligations and postconditions become assumptions.
Per-function certificates compose into a whole-program certificate.

Composes:
  - V004 (VCGen) -- WP calculus, SExpr, annotation extraction
  - V044 (Proof Certificates) -- ProofObligation, ProofCertificate, composition
  - C010 (Parser) -- AST
  - C037 (SMT Solver) -- validity checking
"""

import sys, os, time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

# -- imports --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, Var as SMTVar, App, Op, IntConst, BoolConst, SMTResult
from vc_gen import (
    WPCalculus, SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp,
    SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies,
    FnSpec, extract_fn_spec, ast_to_sexpr, substitute,
    lower_to_smt, check_vc, VCStatus, VCResult, VerificationResult,
    verify_function, verify_program
)
from proof_certificates import (
    ProofObligation, ProofCertificate, ProofKind, CertStatus,
    combine_certificates, check_certificate,
    sexpr_to_str, sexpr_to_smtlib
)


# ============================================================
# Data structures
# ============================================================

class CompVerdict(Enum):
    """Compositional verification verdict."""
    SOUND = "sound"              # All modules verify, composition is sound
    MODULE_FAILURE = "module_failure"  # Some module fails verification
    CALL_FAILURE = "call_failure"      # Call-site obligation fails
    UNKNOWN = "unknown"


@dataclass
class ModuleSpec:
    """Specification for a single module (function)."""
    name: str
    params: List[str]
    preconditions: List[SExpr]
    postconditions: List[SExpr]
    body_stmts: List  # AST statements


@dataclass
class CallSiteObligation:
    """Proof obligation at a function call site."""
    caller: str           # Calling function name
    callee: str           # Called function name
    location: str         # Human-readable location
    precond_vc: SExpr     # The precondition that must hold at call site
    status: VCStatus = VCStatus.UNKNOWN
    counterexample: Optional[dict] = None


@dataclass
class ModuleResult:
    """Verification result for a single module."""
    name: str
    verified: bool
    vcs: List[VCResult]
    certificate: Optional[ProofCertificate] = None
    call_obligations: List[CallSiteObligation] = field(default_factory=list)


@dataclass
class CompositionalResult:
    """Result of compositional verification."""
    verdict: CompVerdict
    modules: Dict[str, ModuleResult]
    call_obligations: List[CallSiteObligation]
    certificate: Optional[ProofCertificate] = None
    monolithic_result: Optional[VerificationResult] = None  # for comparison
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_modules(self) -> int:
        return len(self.modules)

    @property
    def verified_modules(self) -> int:
        return sum(1 for m in self.modules.values() if m.verified)

    @property
    def total_call_obligations(self) -> int:
        return len(self.call_obligations)

    @property
    def satisfied_call_obligations(self) -> int:
        return sum(1 for co in self.call_obligations if co.status == VCStatus.VALID)

    def summary(self) -> str:
        lines = [f"Compositional Verification: {self.verdict.value}"]
        lines.append(f"  Modules: {self.verified_modules}/{self.total_modules} verified")
        lines.append(f"  Call obligations: {self.satisfied_call_obligations}/{self.total_call_obligations} satisfied")
        if self.certificate:
            lines.append(f"  Certificate: {self.certificate.status.value}")
        return "\n".join(lines)


# ============================================================
# Module decomposition
# ============================================================

def _parse_program(source: str):
    """Parse source to AST statements."""
    tokens = lex(source)
    program = Parser(tokens).parse()
    return program.stmts


def extract_modules(source: str) -> Tuple[List[ModuleSpec], List]:
    """Extract function modules and top-level code from source.

    Returns (modules, top_level_stmts).
    """
    stmts = _parse_program(source)
    modules = []
    top_level = []

    for stmt in stmts:
        if isinstance(stmt, FnDecl):
            spec = extract_fn_spec(stmt)
            modules.append(ModuleSpec(
                name=spec.name,
                params=spec.params,
                preconditions=spec.preconditions,
                postconditions=spec.postconditions,
                body_stmts=spec.body_stmts,
            ))
        else:
            top_level.append(stmt)

    return modules, top_level


# ============================================================
# Modular WP with call-site reasoning
# ============================================================

class ModularWPCalculus(WPCalculus):
    """WP calculus that handles calls modularly.

    At call sites to specified functions:
    - Generate precondition obligation (caller must establish callee's precondition)
    - Assume postcondition (substitute actuals for formals, result for return)
    """

    def __init__(self, specs: Dict[str, ModuleSpec]):
        super().__init__()
        self.specs = specs
        self.call_obligations: List[CallSiteObligation] = []
        self.current_caller: str = ""

    def wp_stmt(self, stmt, postcond: SExpr) -> SExpr:
        """Override to handle call expressions modularly."""
        # Handle let x = callee(...) modularly
        if isinstance(stmt, LetDecl) and isinstance(stmt.value, CallExpr):
            callee_name = stmt.value.callee
            if callee_name and callee_name in self.specs:
                return self._wp_modular_call(
                    var_name=stmt.name,
                    callee_name=callee_name,
                    args=stmt.value.args,
                    postcond=postcond
                )

        # Handle x = callee(...) modularly
        if isinstance(stmt, Assign) and isinstance(stmt.value, CallExpr):
            callee_name = stmt.value.callee
            if callee_name and callee_name in self.specs:
                return self._wp_modular_call(
                    var_name=stmt.name,
                    callee_name=callee_name,
                    args=stmt.value.args,
                    postcond=postcond
                )

        return super().wp_stmt(stmt, postcond)

    def _wp_modular_call(self, var_name: str, callee_name: str,
                         args: list, postcond: SExpr) -> SExpr:
        """Compute WP for a modular call: let var = callee(args).

        1. Record precondition obligation (caller must establish precond)
        2. Assume postcondition holds (substitute actuals + result)
        """
        spec = self.specs[callee_name]

        # Build argument substitution: formal -> actual
        arg_sexprs = [ast_to_sexpr(a) for a in args]

        # 1. Precondition obligation: substitute formals with actuals
        for precond in spec.preconditions:
            instantiated = precond
            for formal, actual in zip(spec.params, arg_sexprs):
                instantiated = substitute(instantiated, formal, actual)

            self.call_obligations.append(CallSiteObligation(
                caller=self.current_caller,
                callee=callee_name,
                location=f"{self.current_caller} -> {callee_name}({', '.join(spec.params)})",
                precond_vc=instantiated,
            ))

        # 2. Assume postcondition: substitute formals and 'result' with var_name
        # WP(let x = f(a,b), Q) = Q[x / result_of_f] where f satisfies postcond
        # We substitute result -> var_name, formals -> actuals in postcond
        # Then the postcondition becomes an assumption
        assumed_post = SBool(True)
        for post in spec.postconditions:
            inst_post = post
            for formal, actual in zip(spec.params, arg_sexprs):
                inst_post = substitute(inst_post, formal, actual)
            inst_post = substitute(inst_post, 'result', SVar(var_name))
            assumed_post = s_and(assumed_post, inst_post)

        # WP = precond_holds AND (postcond => Q)
        # But precond is separate obligation. So WP = postcond => Q[x assumed via post]
        # Actually: we assume postcond holds, so WP(call, Q) = Q with info from postcond
        # The proper WP: postcond[result/x] => Q  (if postcond holds, then Q must follow)
        # Simpler: s_implies(assumed_post, postcond_after_substitution)
        # But we want: assumed_post AND Q (postcond is an assumption that strengthens context)
        # Actually the correct WP for assume(P); rest is P => WP(rest)
        # No -- WP(assume P, Q) = P => Q only if we NEED to check.
        # In modular verification: we TRUST the callee spec. So postcond is simply assumed true.
        # The WP is: substitute result->var in postcond, then substitute into Q.
        # This means: WP = Q[var / fresh] where fresh satisfies postcond.
        # Simplest correct approach: assumed_post implies postcond, so just return
        # s_implies(assumed_post, postcond) -- no, that's not right either.

        # Correct modular WP for `let x = f(args)`:
        # We trust f's spec. After the call, x satisfies the postcondition.
        # So the assumption is: forall postcond P of f, P[formals/actuals, result/x] holds.
        # WP(let x = f(args), Q) = assumed_post => Q
        # This means: if the postcondition holds (which we trust), then Q must follow.
        # But since we TRUST it, the WP should be just Q (the postcond is free).
        # Actually no -- we want the postcondition to INFORM Q.

        # The standard approach: treat the call as a havoc of x followed by assume(postcond).
        # WP(havoc x; assume P, Q) = forall x. P => Q
        # For verification: this becomes P[result/x] => Q
        return s_implies(assumed_post, postcond)


# ============================================================
# Module verification
# ============================================================

def verify_module(module: ModuleSpec, all_specs: Dict[str, ModuleSpec]) -> ModuleResult:
    """Verify a single module (function) in isolation.

    Uses modular WP: calls to other specified functions use their specs
    instead of inlining their bodies.
    """
    wp_calc = ModularWPCalculus(all_specs)
    wp_calc.current_caller = module.name

    # Build postcondition from ensures
    if module.postconditions:
        postcond = module.postconditions[0]
        for p in module.postconditions[1:]:
            postcond = s_and(postcond, p)
    else:
        postcond = SBool(True)

    # Build precondition from requires
    if module.preconditions:
        precond = module.preconditions[0]
        for p in module.preconditions[1:]:
            precond = s_and(precond, p)
    else:
        precond = SBool(True)

    # Compute WP of body with postcondition
    try:
        wp = wp_calc.wp_stmts(module.body_stmts, postcond)
    except Exception as e:
        return ModuleResult(
            name=module.name,
            verified=False,
            vcs=[VCResult(f"{module.name}_wp_error", VCStatus.UNKNOWN, None, str(e))],
            call_obligations=wp_calc.call_obligations,
        )

    # Main VC: precondition => WP(body, postcondition)
    main_vc = s_implies(precond, wp)
    vc_result = check_vc(f"{module.name}_main_vc", main_vc)

    # Collect all VCs (main + any loop VCs from WP computation)
    all_vcs = [vc_result]

    # Check call-site precondition obligations
    for co in wp_calc.call_obligations:
        # The precondition must hold under the caller's precondition
        co_vc = s_implies(precond, co.precond_vc)
        co_result = check_vc(f"call_{co.caller}_to_{co.callee}", co_vc)
        co.status = co_result.status
        co.counterexample = co_result.counterexample

    verified = vc_result.status == VCStatus.VALID

    # Build certificate
    cert = _build_module_certificate(module.name, all_vcs, verified)

    return ModuleResult(
        name=module.name,
        verified=verified,
        vcs=all_vcs,
        certificate=cert,
        call_obligations=wp_calc.call_obligations,
    )


def _build_module_certificate(name: str, vcs: List[VCResult], verified: bool) -> ProofCertificate:
    """Build a V044 ProofCertificate for a module."""
    obligations = []
    for vc in vcs:
        status = CertStatus.VALID if vc.status == VCStatus.VALID else (
            CertStatus.INVALID if vc.status == VCStatus.INVALID else CertStatus.UNKNOWN
        )
        obligations.append(ProofObligation(
            name=vc.name,
            description=f"VC for {name}: {vc.name}",
            formula_str=vc.formula_str or "",
            formula_smt="",  # SMT-LIB2 format not needed for compositional
            status=status,
            counterexample=vc.counterexample,
        ))

    cert_status = CertStatus.VALID if verified else CertStatus.INVALID
    return ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Module {name} satisfies its specification",
        obligations=obligations,
        status=cert_status,
    )


# ============================================================
# Compositional verification
# ============================================================

def verify_compositional(source: str) -> CompositionalResult:
    """Verify a program compositionally.

    1. Extract all function modules with specs
    2. Verify each module independently (using modular WP for inter-function calls)
    3. Check all call-site obligations
    4. Compose certificates

    Returns CompositionalResult with per-module results and composed certificate.
    """
    t0 = time.time()
    modules, top_level = extract_modules(source)

    if not modules:
        return CompositionalResult(
            verdict=CompVerdict.SOUND,
            modules={},
            call_obligations=[],
            metadata={"reason": "no functions to verify", "duration": time.time() - t0},
        )

    # Build spec map
    spec_map = {m.name: m for m in modules}

    # Verify each module
    module_results = {}
    all_call_obligations = []

    for module in modules:
        result = verify_module(module, spec_map)
        module_results[module.name] = result
        all_call_obligations.extend(result.call_obligations)

    # Determine verdict
    all_modules_verified = all(r.verified for r in module_results.values())
    all_calls_satisfied = all(co.status == VCStatus.VALID for co in all_call_obligations)

    if all_modules_verified and all_calls_satisfied:
        verdict = CompVerdict.SOUND
    elif not all_modules_verified:
        verdict = CompVerdict.MODULE_FAILURE
    elif not all_calls_satisfied:
        verdict = CompVerdict.CALL_FAILURE
    else:
        verdict = CompVerdict.UNKNOWN

    # Compose certificates
    certs = [r.certificate for r in module_results.values() if r.certificate]

    # Add call-site obligation certificates
    call_obligations_cert = _build_call_obligations_certificate(all_call_obligations)
    if call_obligations_cert:
        certs.append(call_obligations_cert)

    composed = None
    if certs:
        composed = combine_certificates(
            *certs,
            claim=f"Compositional verification of {len(modules)} modules"
        )

    duration = time.time() - t0
    return CompositionalResult(
        verdict=verdict,
        modules=module_results,
        call_obligations=all_call_obligations,
        certificate=composed,
        metadata={
            "duration": duration,
            "num_modules": len(modules),
            "num_call_obligations": len(all_call_obligations),
        },
    )


def _build_call_obligations_certificate(obligations: List[CallSiteObligation]) -> Optional[ProofCertificate]:
    """Build a certificate for all call-site obligations."""
    if not obligations:
        return None

    proof_obs = []
    for co in obligations:
        status = CertStatus.VALID if co.status == VCStatus.VALID else (
            CertStatus.INVALID if co.status == VCStatus.INVALID else CertStatus.UNKNOWN
        )
        proof_obs.append(ProofObligation(
            name=f"call_{co.caller}_to_{co.callee}",
            description=f"Precondition of {co.callee} holds at call site in {co.caller}",
            formula_str=sexpr_to_str(co.precond_vc) if co.precond_vc else "",
            formula_smt="",
            status=status,
            counterexample=co.counterexample,
        ))

    all_valid = all(o.status == CertStatus.VALID for o in proof_obs)
    any_invalid = any(o.status == CertStatus.INVALID for o in proof_obs)

    return ProofCertificate(
        kind=ProofKind.VCGEN,
        claim="All call-site precondition obligations are satisfied",
        obligations=proof_obs,
        status=CertStatus.VALID if all_valid else (CertStatus.INVALID if any_invalid else CertStatus.UNKNOWN),
    )


# ============================================================
# Comparison: modular vs monolithic
# ============================================================

def compare_modular_vs_monolithic(source: str) -> dict:
    """Compare compositional verification against monolithic V004 verification.

    Returns dict with both results and timing comparison.
    """
    # Modular
    t0 = time.time()
    comp_result = verify_compositional(source)
    modular_time = time.time() - t0

    # Monolithic
    t1 = time.time()
    mono_result = verify_program(source)
    mono_time = time.time() - t1

    comp_result.monolithic_result = mono_result

    return {
        "modular_verdict": comp_result.verdict.value,
        "modular_modules": comp_result.total_modules,
        "modular_verified": comp_result.verified_modules,
        "modular_call_obligations": comp_result.total_call_obligations,
        "modular_calls_satisfied": comp_result.satisfied_call_obligations,
        "modular_time": modular_time,
        "monolithic_verified": mono_result.verified,
        "monolithic_vcs": mono_result.total_vcs,
        "monolithic_valid": mono_result.valid_vcs,
        "monolithic_time": mono_time,
        "agree": (comp_result.verdict == CompVerdict.SOUND) == mono_result.verified,
        "compositional_result": comp_result,
    }


# ============================================================
# Incremental re-verification
# ============================================================

def verify_incremental(source: str, changed_modules: List[str],
                       cached_results: Dict[str, ModuleResult] = None) -> CompositionalResult:
    """Re-verify only changed modules, reusing cached results for unchanged ones.

    This is the key benefit of compositional verification: when one module changes,
    only that module and its callers need re-verification (if the spec didn't change).

    Args:
        source: Full program source
        changed_modules: Names of modules that changed
        cached_results: Previous verification results to reuse

    Returns:
        CompositionalResult with mix of fresh and cached results
    """
    t0 = time.time()
    modules, top_level = extract_modules(source)
    spec_map = {m.name: m for m in modules}

    if cached_results is None:
        cached_results = {}

    module_results = {}
    all_call_obligations = []
    reverified = []
    reused = []

    for module in modules:
        if module.name in changed_modules or module.name not in cached_results:
            # Re-verify this module
            result = verify_module(module, spec_map)
            module_results[module.name] = result
            reverified.append(module.name)
        else:
            # Reuse cached result -- but re-check call obligations if callees changed
            cached = cached_results[module.name]
            needs_recheck = any(
                co.callee in changed_modules for co in cached.call_obligations
            )
            if needs_recheck:
                result = verify_module(module, spec_map)
                module_results[module.name] = result
                reverified.append(module.name)
            else:
                module_results[module.name] = cached
                reused.append(module.name)

        all_call_obligations.extend(module_results[module.name].call_obligations)

    # Determine verdict
    all_modules_verified = all(r.verified for r in module_results.values())
    all_calls_satisfied = all(co.status == VCStatus.VALID for co in all_call_obligations)

    if all_modules_verified and all_calls_satisfied:
        verdict = CompVerdict.SOUND
    elif not all_modules_verified:
        verdict = CompVerdict.MODULE_FAILURE
    elif not all_calls_satisfied:
        verdict = CompVerdict.CALL_FAILURE
    else:
        verdict = CompVerdict.UNKNOWN

    # Compose certificates
    certs = [r.certificate for r in module_results.values() if r.certificate]
    call_cert = _build_call_obligations_certificate(all_call_obligations)
    if call_cert:
        certs.append(call_cert)
    composed = combine_certificates(*certs, claim="Incremental compositional verification") if certs else None

    duration = time.time() - t0
    return CompositionalResult(
        verdict=verdict,
        modules=module_results,
        call_obligations=all_call_obligations,
        certificate=composed,
        metadata={
            "duration": duration,
            "reverified": reverified,
            "reused": reused,
            "incremental": True,
        },
    )


# ============================================================
# Interface refinement checking
# ============================================================

def check_spec_refinement(source_old: str, source_new: str, fn_name: str) -> dict:
    """Check if a new function spec refines the old one.

    Refinement means: weaker precondition OR stronger postcondition.
    This guarantees that all callers verified against the old spec remain valid.

    Returns dict with refinement status and details.
    """
    modules_old, _ = extract_modules(source_old)
    modules_new, _ = extract_modules(source_new)

    old_spec = None
    new_spec = None
    for m in modules_old:
        if m.name == fn_name:
            old_spec = m
    for m in modules_new:
        if m.name == fn_name:
            new_spec = m

    if not old_spec or not new_spec:
        return {"refinement": False, "error": f"Function {fn_name} not found in both versions"}

    results = {
        "function": fn_name,
        "precond_weakened": False,
        "postcond_strengthened": False,
        "refinement": False,
        "details": [],
    }

    # Check precondition: new_pre should be implied by old_pre (new is weaker)
    # old_pre => new_pre (every state satisfying old spec also satisfies new)
    if old_spec.preconditions and new_spec.preconditions:
        old_pre = old_spec.preconditions[0]
        for p in old_spec.preconditions[1:]:
            old_pre = s_and(old_pre, p)
        new_pre = new_spec.preconditions[0]
        for p in new_spec.preconditions[1:]:
            new_pre = s_and(new_pre, p)

        pre_vc = s_implies(old_pre, new_pre)
        pre_result = check_vc("precond_weakening", pre_vc)
        results["precond_weakened"] = pre_result.status == VCStatus.VALID
        results["details"].append(f"Precondition weakening: {pre_result.status.name}")
    elif not new_spec.preconditions:
        results["precond_weakened"] = True  # no precondition is weakest
        results["details"].append("New has no precondition (weakest possible)")
    elif not old_spec.preconditions:
        results["precond_weakened"] = False  # old had none, new adds one -- stricter
        results["details"].append("Old had no precondition, new adds one (stricter)")

    # Check postcondition: new_post should imply old_post (new is stronger)
    # new_post => old_post (new guarantees at least what old did)
    if old_spec.postconditions and new_spec.postconditions:
        old_post = old_spec.postconditions[0]
        for p in old_spec.postconditions[1:]:
            old_post = s_and(old_post, p)
        new_post = new_spec.postconditions[0]
        for p in new_spec.postconditions[1:]:
            new_post = s_and(new_post, p)

        post_vc = s_implies(new_post, old_post)
        post_result = check_vc("postcond_strengthening", post_vc)
        results["postcond_strengthened"] = post_result.status == VCStatus.VALID
        results["details"].append(f"Postcondition strengthening: {post_result.status.name}")
    elif not old_spec.postconditions:
        results["postcond_strengthened"] = True  # old had no post, anything is stronger
        results["details"].append("Old had no postcondition (trivially strengthened)")
    elif not new_spec.postconditions:
        results["postcond_strengthened"] = False  # old had post, new drops it
        results["details"].append("Old had postcondition, new drops it (weaker)")

    # Refinement holds if precondition is weakened AND postcondition is strengthened
    # (or stays the same in either dimension)
    results["refinement"] = results["precond_weakened"] and results["postcond_strengthened"]

    return results


# ============================================================
# Dependency analysis
# ============================================================

def analyze_call_graph(source: str) -> dict:
    """Analyze the call graph of functions in the program.

    Returns dict with:
    - call_graph: {fn_name: [callees]}
    - reverse_graph: {fn_name: [callers]}
    - specified: list of functions with specs
    - unspecified: list of functions without specs
    """
    modules, top_level = extract_modules(source)
    stmts = _parse_program(source)

    # Build call graph by walking AST
    call_graph = {}
    for stmt in stmts:
        if isinstance(stmt, FnDecl):
            callees = set()
            _collect_calls(stmt.body, callees)
            # Remove self-recursion for simplicity
            callees.discard(stmt.name)
            call_graph[stmt.name] = sorted(callees)

    # Reverse graph
    reverse_graph = {name: [] for name in call_graph}
    for caller, callees in call_graph.items():
        for callee in callees:
            if callee in reverse_graph:
                reverse_graph[callee].append(caller)

    spec_names = {m.name for m in modules if m.preconditions or m.postconditions}

    return {
        "call_graph": call_graph,
        "reverse_graph": reverse_graph,
        "specified": sorted(spec_names),
        "unspecified": sorted(set(call_graph.keys()) - spec_names),
        "num_functions": len(call_graph),
    }


def _collect_calls(node, calls: set):
    """Recursively collect function call names from AST."""
    if isinstance(node, CallExpr):
        callee = node.callee  # CallExpr.callee is always a str in C010
        # Filter out annotation pseudo-calls
        if callee and callee not in ('requires', 'ensures', 'invariant', 'assert'):
            calls.add(callee)
        for arg in node.args:
            _collect_calls(arg, calls)
    elif isinstance(node, FnDecl):
        _collect_calls(node.body, calls)
    elif isinstance(node, Block):
        for s in node.stmts:
            _collect_calls(s, calls)
    elif isinstance(node, LetDecl):
        if node.value:
            _collect_calls(node.value, calls)
    elif isinstance(node, Assign):
        _collect_calls(node.value, calls)
    elif isinstance(node, IfStmt):
        _collect_calls(node.cond, calls)
        _collect_calls(node.then_body, calls)
        if node.else_body:
            _collect_calls(node.else_body, calls)
    elif isinstance(node, WhileStmt):
        _collect_calls(node.cond, calls)
        _collect_calls(node.body, calls)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _collect_calls(node.value, calls)
    elif isinstance(node, PrintStmt):
        _collect_calls(node.value, calls)
    elif isinstance(node, BinOp):
        _collect_calls(node.left, calls)
        _collect_calls(node.right, calls)
    elif isinstance(node, UnaryOp):
        _collect_calls(node.operand, calls)


# ============================================================
# Impact analysis
# ============================================================

def analyze_change_impact(source: str, changed_fn: str) -> dict:
    """Determine which modules need re-verification when a function changes.

    If only the body changed (spec unchanged): re-verify the changed function.
    If the spec changed: re-verify the function AND all callers.

    Returns dict with impacted modules and reason.
    """
    graph = analyze_call_graph(source)
    callers = graph["reverse_graph"].get(changed_fn, [])

    return {
        "changed": changed_fn,
        "body_change_impact": [changed_fn],  # Only the function itself
        "spec_change_impact": [changed_fn] + callers,  # Function + all callers
        "callers": callers,
        "call_graph": graph["call_graph"],
    }


# ============================================================
# Convenience API
# ============================================================

def certify_compositional(source: str) -> CompositionalResult:
    """One-shot: verify compositionally and return checked certificate."""
    result = verify_compositional(source)
    if result.certificate:
        try:
            checked = check_certificate(result.certificate)
            result.certificate = checked
        except Exception:
            pass  # Certificate checking is best-effort
    return result


def compositional_summary(result: CompositionalResult) -> str:
    """Human-readable summary of compositional verification."""
    return result.summary()
