"""V063: Verified Probabilistic Programs

Hoare-logic style verification for programs with random inputs.
Composes V004 (VCGen/WP/SExpr) + V060 (statistical model checking) + C010 (parser/VM) + C037 (SMT).

Key idea: {P} S {Q @ p} means "if P holds, then after S, Q holds with probability >= p".
- Deterministic VCs checked exactly via SMT (V004)
- Probabilistic VCs checked statistically via Monte Carlo/SPRT (V060)
- random(lo, hi) introduces uniform integer randomness

Annotations in C10 source:
  requires(P)            -- precondition (deterministic)
  ensures(Q)             -- postcondition must hold always (deterministic, checked via SMT)
  prob_ensures(Q, p)     -- postcondition holds with probability >= p (statistical)
"""

import sys, os, math, random as py_random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V060_probabilistic_verification'))

from stack_vm import lex, Parser, Compiler, VM
from smt_solver import SMTSolver, SMTResult
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies,
    WPCalculus, VCResult, VCStatus, VerificationResult, FnSpec,
    lower_to_smt, check_vc, extract_fn_spec, verify_function as vc_verify_function
)
from probabilistic_verification import (
    ProbabilisticExecutor, StatVerdict, StatProperty, PropertyKind,
    StatCheckResult, SampleResult, MonteCarloResult,
    stat_check, stat_check_sprt, monte_carlo_estimate, expected_value_check,
    wilson_confidence_interval, sprt_test
)


# ---------- Data Model ----------

class ProbVerdict(Enum):
    """Verdict for probabilistic verification."""
    VERIFIED = "verified"           # All VCs pass (deterministic + probabilistic)
    VIOLATED = "violated"           # Some VC fails
    INCONCLUSIVE = "inconclusive"   # Statistical test inconclusive
    ERROR = "error"                 # Parse/execution error


@dataclass
class ProbVC:
    """A verification condition -- either deterministic or probabilistic."""
    name: str
    kind: str  # "deterministic" or "probabilistic"
    # For deterministic VCs:
    formula: Optional[SExpr] = None
    # For probabilistic VCs:
    postcondition_src: Optional[str] = None   # C10 expression source
    threshold: float = 0.95
    # Result (filled after checking):
    status: Optional[str] = None              # "valid", "invalid", "accept", "reject", "inconclusive", "unknown"
    counterexample: Optional[dict] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    estimated_probability: Optional[float] = None
    detail: str = ""


@dataclass
class ProbFnSpec:
    """Function spec with both deterministic and probabilistic postconditions."""
    name: str
    params: List[str]
    preconditions: List[SExpr]           # requires(...)
    postconditions: List[SExpr]          # ensures(...)  -- deterministic
    prob_postconditions: List[Tuple[SExpr, float, str]]  # (expr, threshold, src) from prob_ensures(...)
    body_stmts: list                     # Executable statements (no annotations)
    random_vars: Dict[str, Tuple[int, int]]  # var -> (lo, hi) from random() calls


@dataclass
class ProbVerificationResult:
    """Result of probabilistic program verification."""
    verdict: ProbVerdict
    deterministic_vcs: List[ProbVC]
    probabilistic_vcs: List[ProbVC]
    errors: List[str] = field(default_factory=list)

    @property
    def total_vcs(self) -> int:
        return len(self.deterministic_vcs) + len(self.probabilistic_vcs)

    @property
    def all_vcs(self) -> List[ProbVC]:
        return self.deterministic_vcs + self.probabilistic_vcs

    def summary(self) -> str:
        lines = [f"Verdict: {self.verdict.value}"]
        lines.append(f"Deterministic VCs: {len(self.deterministic_vcs)}")
        for vc in self.deterministic_vcs:
            lines.append(f"  [{vc.status}] {vc.name}")
        lines.append(f"Probabilistic VCs: {len(self.probabilistic_vcs)}")
        for vc in self.probabilistic_vcs:
            prob_str = f" (P={vc.estimated_probability:.3f})" if vc.estimated_probability is not None else ""
            ci_str = f" CI={vc.confidence_interval}" if vc.confidence_interval else ""
            lines.append(f"  [{vc.status}] {vc.name}{prob_str}{ci_str}")
        if self.errors:
            lines.append(f"Errors: {self.errors}")
        return "\n".join(lines)


# ---------- AST Analysis: Extract Random Variables and Probabilistic Specs ----------

def _is_random_call(node) -> Optional[Tuple[int, int]]:
    """Check if AST node is random(lo, hi). Returns (lo, hi) or None.

    If arguments are variables (not int literals), returns default range.
    """
    if not hasattr(node, '__class__'):
        return None
    cls = node.__class__.__name__
    if cls == 'CallExpr':
        callee = node.callee if hasattr(node, 'callee') else None
        if callee == 'random':
            args = node.args if hasattr(node, 'args') else []
            if len(args) == 2:
                lo = _extract_int(args[0])
                hi = _extract_int(args[1])
                if lo is not None and hi is not None:
                    return (lo, hi)
                # Variable args -- use default range
                return (0, 100)
            elif len(args) == 0:
                return (0, 100)
            elif len(args) == 1:
                hi = _extract_int(args[0])
                if hi is not None:
                    return (0, hi)
                return (0, 100)
    return None


def _extract_int(node) -> Optional[int]:
    """Extract integer value from AST node."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return node.value
    if cls == 'UnaryOp' and hasattr(node, 'op') and node.op == '-':
        inner = _extract_int(node.operand)
        if inner is not None:
            return -inner
    return None


def _is_prob_ensures(node) -> Optional[Tuple[Any, float]]:
    """Check if AST node is prob_ensures(expr, threshold). Returns (expr_node, threshold) or None."""
    cls = node.__class__.__name__
    if cls == 'ExprStmt' and hasattr(node, 'expr'):
        node = node.expr
        cls = node.__class__.__name__
    if cls == 'CallExpr':
        callee = node.callee if hasattr(node, 'callee') else None
        if callee == 'prob_ensures':
            args = node.args if hasattr(node, 'args') else []
            if len(args) == 2:
                threshold = _extract_float(args[1])
                if threshold is not None:
                    return (args[0], threshold)
    return None


def _extract_float(node) -> Optional[float]:
    """Extract float from AST. Supports int literals and simple division expressions."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return float(node.value)
    if cls == 'BinOp' and hasattr(node, 'op') and node.op == '/':
        l = _extract_int(node.left)
        r = _extract_int(node.right)
        if l is not None and r is not None and r != 0:
            return l / r
    return None


def _ast_to_sexpr(node) -> SExpr:
    """Convert C10 AST expression to SExpr for WP calculus."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return SInt(node.value)
    if cls == 'Var':
        name = node.name if hasattr(node, 'name') else str(node)
        return SVar(name)
    if cls == 'BinOp':
        left = _ast_to_sexpr(node.left)
        right = _ast_to_sexpr(node.right)
        return SBinOp(node.op, left, right)
    if cls == 'UnaryOp':
        operand = _ast_to_sexpr(node.operand)
        return SUnaryOp(node.op, operand)
    if cls == 'CallExpr':
        callee = node.callee if hasattr(node, 'callee') else ''
        if callee == 'result':
            return SVar('result')
    # Fallback
    return SVar(str(node))


def _sexpr_to_source(expr: SExpr) -> str:
    """Convert SExpr back to C10-compatible expression string for runtime evaluation."""
    if isinstance(expr, SVar):
        return expr.name
    if isinstance(expr, SInt):
        if expr.value < 0:
            return f"(0 - {-expr.value})"
        return str(expr.value)
    if isinstance(expr, SBool):
        return "1" if expr.value else "0"
    if isinstance(expr, SBinOp):
        left = _sexpr_to_source(expr.left)
        right = _sexpr_to_source(expr.right)
        op = expr.op
        if op == 'and':
            op = '&&'
        elif op == 'or':
            op = '||'
        elif op == '==':
            op = '=='
        elif op == '!=':
            op = '!='
        return f"({left} {op} {right})"
    if isinstance(expr, SUnaryOp):
        operand = _sexpr_to_source(expr.operand)
        op = expr.op
        if op == 'not':
            op = '!'
        return f"({op}{operand})"
    if isinstance(expr, SAnd):
        parts = [_sexpr_to_source(c) for c in expr.conjuncts]
        return "(" + " && ".join(parts) + ")"
    if isinstance(expr, SOr):
        parts = [_sexpr_to_source(d) for d in expr.disjuncts]
        return "(" + " || ".join(parts) + ")"
    if isinstance(expr, SNot):
        return f"(!{_sexpr_to_source(expr.operand)})"
    if isinstance(expr, SImplies):
        # P => Q  ===  !P || Q
        a = _sexpr_to_source(expr.antecedent)
        b = _sexpr_to_source(expr.consequent)
        return f"(!({a}) || ({b}))"
    if isinstance(expr, SIte):
        c = _sexpr_to_source(expr.cond)
        t = _sexpr_to_source(expr.then_val)
        e = _sexpr_to_source(expr.else_val)
        return f"(if ({c}) {t} else {e})"
    return str(expr)


def _expr_source_from_ast(node) -> str:
    """Convert AST expression to source string for runtime evaluation."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return str(node.value)
    if cls == 'Var':
        return node.name if hasattr(node, 'name') else str(node)
    if cls == 'BinOp':
        left = _expr_source_from_ast(node.left)
        right = _expr_source_from_ast(node.right)
        op = node.op
        if op == 'and':
            op = '&&'
        elif op == 'or':
            op = '||'
        return f"({left} {op} {right})"
    if cls == 'UnaryOp':
        operand = _expr_source_from_ast(node.operand)
        op = node.op
        if op == 'not':
            op = '!'
        return f"({op}{operand})"
    if cls == 'CallExpr':
        callee = node.callee if hasattr(node, 'callee') else ''
        args = node.args if hasattr(node, 'args') else []
        arg_strs = [_expr_source_from_ast(a) for a in args]
        return f"{callee}({', '.join(arg_strs)})"
    return str(node)


def extract_prob_fn_spec(stmts, fn_name: str = None, params: List[str] = None) -> ProbFnSpec:
    """Extract probabilistic function spec from AST statements.

    Recognizes:
    - requires(P) -> deterministic precondition
    - ensures(Q) -> deterministic postcondition
    - prob_ensures(Q, threshold) -> probabilistic postcondition
    - let x = random(lo, hi) -> random variable
    """
    preconditions = []
    postconditions = []
    prob_postconditions = []
    body_stmts = []
    random_vars = {}

    for stmt in stmts:
        cls = stmt.__class__.__name__

        # Check for annotation calls at statement level
        expr_node = None
        if cls == 'ExprStmt' and hasattr(stmt, 'expr'):
            expr_node = stmt.expr
        elif cls == 'CallExpr':
            expr_node = stmt

        if expr_node and hasattr(expr_node, 'callee'):
            callee = expr_node.callee
            args = expr_node.args if hasattr(expr_node, 'args') else []

            if callee == 'requires' and len(args) >= 1:
                preconditions.append(_ast_to_sexpr(args[0]))
                continue
            elif callee == 'ensures' and len(args) >= 1:
                postconditions.append(_ast_to_sexpr(args[0]))
                continue
            elif callee == 'prob_ensures' and len(args) >= 2:
                expr_sexpr = _ast_to_sexpr(args[0])
                threshold = _extract_float(args[1])
                if threshold is None:
                    threshold = 0.95
                src = _expr_source_from_ast(args[0])
                prob_postconditions.append((expr_sexpr, threshold, src))
                continue

        # Check for random() in let declarations
        if cls == 'LetDecl' and hasattr(stmt, 'value'):
            rng = _is_random_call(stmt.value)
            if rng is not None:
                var_name = stmt.name if hasattr(stmt, 'name') else str(stmt)
                random_vars[var_name] = rng
                body_stmts.append(stmt)
                continue

        body_stmts.append(stmt)

    return ProbFnSpec(
        name=fn_name or "<top>",
        params=params or [],
        preconditions=preconditions,
        postconditions=postconditions,
        prob_postconditions=prob_postconditions,
        body_stmts=body_stmts,
        random_vars=random_vars
    )


# ---------- Deterministic VC Checking (via V004) ----------

def check_deterministic_vcs(source: str, fn_name: str = None) -> List[ProbVC]:
    """Check deterministic VCs using V004's exact SMT verification."""
    results = []
    try:
        vr = vc_verify_function(source, fn_name)
        for vc in vr.vcs:
            pvc = ProbVC(
                name=vc.name,
                kind="deterministic",
                formula=None,
                status=vc.status.value if hasattr(vc.status, 'value') else str(vc.status),
                counterexample=vc.counterexample,
                detail=vc.formula_str or ""
            )
            results.append(pvc)
    except Exception as e:
        results.append(ProbVC(
            name="deterministic_check",
            kind="deterministic",
            status="error",
            detail=str(e)
        ))
    return results


# ---------- Probabilistic VC Checking (via V060) ----------

def _strip_annotations(source: str) -> str:
    """Remove requires/ensures/prob_ensures annotations from source for execution."""
    lines = source.split('\n')
    stripped = []
    for line in lines:
        s = line.strip()
        if s.startswith('requires(') or s.startswith('ensures(') or s.startswith('prob_ensures('):
            continue
        stripped.append(line)
    return '\n'.join(stripped)


def _build_postcond_checker(postcond_src: str) -> str:
    """Build a C10 expression that checks postcondition at runtime.

    The postcondition source may reference 'result' (function return value)
    or program variables directly.
    """
    return postcond_src


def _run_with_inputs(source: str, inputs: Dict[str, int], postcond_src: str) -> bool:
    """Run program with concrete inputs and evaluate postcondition.

    Replaces random() calls with concrete values, strips annotations,
    appends postcondition check, returns True if postcond holds.
    """
    clean = _strip_annotations(source)
    replaced = _replace_random_with_input(clean, inputs)

    # Build full source: input declarations + body + postcondition check
    parts = []
    for var, val in inputs.items():
        if val < 0:
            parts.append(f"let {var} = 0 - {-val};")
        else:
            parts.append(f"let {var} = {val};")
    parts.append(replaced)
    parts.append(f"let __check = {postcond_src};")
    full = "\n".join(parts)

    try:
        tokens = lex(full)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        chunk = compiler.compile(ast)
        vm = VM(chunk)
        vm.run()
        check_val = vm.env.get('__check', None)
        return bool(check_val) and check_val != 0
    except Exception:
        return False


def check_probabilistic_vc(
    source: str,
    postcond_src: str,
    threshold: float,
    random_vars: Dict[str, Tuple[int, int]],
    param_inputs: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    use_sprt: bool = True,
    vc_name: str = "prob_vc"
) -> ProbVC:
    """Check a single probabilistic VC using statistical sampling.

    Runs the program many times with random inputs, evaluates postcondition,
    checks if P(postcondition) >= threshold.

    Does NOT use V060's ProbabilisticExecutor (which can't handle random() calls).
    Instead, samples directly and uses V060's statistical functions.
    """
    # All random variables (from random() calls + params with ranges)
    all_ranges = dict(random_vars)
    if param_inputs:
        for k, v in param_inputs.items():
            all_ranges[k] = v

    if not all_ranges:
        return ProbVC(
            name=vc_name,
            kind="probabilistic",
            postcondition_src=postcond_src,
            threshold=threshold,
            status="error",
            detail="No random variables found -- use deterministic verification"
        )

    # Sample loop
    rng = py_random.Random(seed)
    samples_bool = []   # True/False per sample
    failing_inputs = None
    passing = 0
    total = 0

    for _ in range(n_samples):
        inputs = {}
        for var, (lo, hi) in all_ranges.items():
            inputs[var] = rng.randint(lo, hi)

        passed = _run_with_inputs(source, inputs, postcond_src)
        samples_bool.append(passed)
        total += 1
        if passed:
            passing += 1
        elif failing_inputs is None:
            failing_inputs = dict(inputs)

        # SPRT early termination check
        if use_sprt and threshold > 0 and total >= 10:
            p0 = threshold
            p1 = max(0.01, threshold - 0.10)
            verdict, log_ratio = sprt_test(samples_bool, p0, p1, alpha=0.01, beta=0.01)
            if verdict != StatVerdict.INCONCLUSIVE:
                break

    # Compute final statistics
    est_prob = passing / max(total, 1)
    ci = wilson_confidence_interval(total, passing, confidence)

    # Determine verdict
    if use_sprt and threshold > 0:
        verdict, _ = sprt_test(samples_bool, threshold, max(0.01, threshold - 0.10))
        if verdict == StatVerdict.ACCEPT:
            status = "accept"
        elif verdict == StatVerdict.REJECT:
            status = "reject"
        else:
            status = "inconclusive"
    else:
        # Fixed-sample: use CI
        if ci[0] >= threshold:
            status = "accept"
        elif ci[1] < threshold:
            status = "reject"
        else:
            status = "inconclusive"

    return ProbVC(
        name=vc_name,
        kind="probabilistic",
        postcondition_src=postcond_src,
        threshold=threshold,
        status=status,
        counterexample=failing_inputs,
        confidence_interval=ci,
        estimated_probability=est_prob,
        detail=f"samples={total}, passing={passing}"
    )


def _evaluate_postcondition(
    clean_source: str,
    inputs: Dict[str, int],
    result: Any,
    postcond_src: str,
    input_vars: List[str]
) -> bool:
    """Evaluate a postcondition expression against program state.

    Delegates to _run_with_inputs which handles random() replacement.
    """
    return _run_with_inputs(clean_source, inputs, postcond_src)


def _replace_random_with_input(source: str, inputs: Dict[str, int]) -> str:
    """Replace let x = random(...) with concrete assignment from inputs.

    Also removes duplicate let declarations for variables already in inputs
    (since _run_with_inputs prepends let x = val; for each input).
    """
    import re
    lines = source.split('\n')
    result_lines = []
    for line in lines:
        stripped = line.strip()
        skip = False
        for var in inputs:
            # Match: let var = random(...)
            pattern = rf'let\s+{re.escape(var)}\s*=\s*random\s*\([^)]*\)\s*;'
            if re.search(pattern, stripped):
                skip = True
                break
            # Also skip plain let var = ...; if we're providing that var as input
            # (to avoid double declaration)
            pattern2 = rf'^let\s+{re.escape(var)}\s*='
            if re.match(pattern2, stripped) and 'random(' in stripped:
                skip = True
                break
        if not skip:
            result_lines.append(line)
    return '\n'.join(result_lines)


def _build_result_capture_source(
    inputs: Dict[str, int],
    clean_source: str,
    check_src: str
) -> str:
    """Build source that captures function return value and checks postcondition."""
    parts = []
    for var, val in inputs.items():
        if val < 0:
            parts.append(f"let {var} = 0 - {-val};")
        else:
            parts.append(f"let {var} = {val};")

    body = _replace_random_with_input(clean_source, inputs)
    parts.append(body)

    # If there's a function, capture its return
    # For top-level, __result__ is the last computed value
    parts.append(f"let __check = {check_src};")
    return "\n".join(parts)


# ---------- Main Verification Pipeline ----------

def verify_probabilistic_function(
    source: str,
    fn_name: str = None,
    param_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    use_sprt: bool = True
) -> ProbVerificationResult:
    """Verify a function with both deterministic and probabilistic specs.

    Args:
        source: C10 source code with annotations
        fn_name: Function to verify (None for top-level)
        param_ranges: Input ranges for function parameters
        n_samples: Max samples for probabilistic checks
        confidence: Confidence level for statistical tests
        seed: Random seed for reproducibility
        use_sprt: Use SPRT (early termination) vs fixed-sample MC

    Returns:
        ProbVerificationResult with all VC results
    """
    errors = []
    det_vcs = []
    prob_vcs = []

    # Parse source
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()
    except Exception as e:
        return ProbVerificationResult(
            verdict=ProbVerdict.ERROR,
            deterministic_vcs=[],
            probabilistic_vcs=[],
            errors=[f"Parse error: {e}"]
        )

    # Extract spec
    stmts = ast.stmts if hasattr(ast, 'stmts') else []

    # If fn_name specified, find that function
    if fn_name:
        fn_stmts = None
        fn_params = []
        for s in stmts:
            if s.__class__.__name__ == 'FnDecl' and hasattr(s, 'name') and s.name == fn_name:
                fn_stmts = s.body.stmts if hasattr(s.body, 'stmts') else [s.body]
                fn_params = s.params if hasattr(s, 'params') else []
                break
        if fn_stmts is None:
            return ProbVerificationResult(
                verdict=ProbVerdict.ERROR,
                deterministic_vcs=[],
                probabilistic_vcs=[],
                errors=[f"Function '{fn_name}' not found"]
            )
        spec = extract_prob_fn_spec(fn_stmts, fn_name, fn_params)
    else:
        spec = extract_prob_fn_spec(stmts)

    # Check deterministic postconditions via V004 (if any)
    if spec.postconditions:
        det_vcs = check_deterministic_vcs(source, fn_name)

    # Check probabilistic postconditions via V060
    for i, (expr, threshold, src) in enumerate(spec.prob_postconditions):
        pvc = check_probabilistic_vc(
            source=source,
            postcond_src=src,
            threshold=threshold,
            random_vars=spec.random_vars,
            param_inputs=param_ranges,
            n_samples=n_samples,
            confidence=confidence,
            seed=seed,
            use_sprt=use_sprt,
            vc_name=f"{spec.name}: prob_ensures({src}, {threshold})"
        )
        prob_vcs.append(pvc)

    # Determine overall verdict
    verdict = _compute_verdict(det_vcs, prob_vcs, errors)

    return ProbVerificationResult(
        verdict=verdict,
        deterministic_vcs=det_vcs,
        probabilistic_vcs=prob_vcs,
        errors=errors
    )


def _compute_verdict(
    det_vcs: List[ProbVC],
    prob_vcs: List[ProbVC],
    errors: List[str]
) -> ProbVerdict:
    """Compute overall verdict from individual VC results."""
    if errors:
        return ProbVerdict.ERROR

    all_vcs = det_vcs + prob_vcs
    if not all_vcs:
        return ProbVerdict.VERIFIED  # Vacuously true

    has_failure = False
    has_inconclusive = False
    has_error = False

    for vc in all_vcs:
        if vc.status in ("invalid", "reject"):
            has_failure = True
        elif vc.status == "inconclusive":
            has_inconclusive = True
        elif vc.status == "error":
            has_error = True
        elif vc.status == "unknown":
            has_inconclusive = True

    if has_failure:
        return ProbVerdict.VIOLATED
    if has_error:
        return ProbVerdict.ERROR
    if has_inconclusive:
        return ProbVerdict.INCONCLUSIVE
    return ProbVerdict.VERIFIED


# ---------- High-Level APIs ----------

def verify_probabilistic(
    source: str,
    input_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> ProbVerificationResult:
    """Verify a probabilistic program (top-level statements).

    Example:
        source = '''
        let x = random(1, 10);
        let y = x * 2;
        prob_ensures(y >= 2, 99/100);
        prob_ensures(y <= 20, 99/100);
        '''
        result = verify_probabilistic(source)
    """
    return verify_probabilistic_function(
        source, fn_name=None, param_ranges=input_ranges,
        n_samples=n_samples, confidence=confidence, seed=seed
    )


def verify_prob_function(
    source: str,
    fn_name: str,
    param_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> ProbVerificationResult:
    """Verify a named function with probabilistic specs.

    Example:
        source = '''
        fn fair_coin(n) {
            let x = random(0, 1);
            prob_ensures(x == 0, 49/100);
            return x;
        }
        '''
        result = verify_prob_function(source, 'fair_coin', {'n': (1, 10)})
    """
    return verify_probabilistic_function(
        source, fn_name=fn_name, param_ranges=param_ranges,
        n_samples=n_samples, seed=seed
    )


def check_prob_property(
    source: str,
    property_expr: str,
    threshold: float,
    random_vars: Dict[str, Tuple[int, int]],
    n_samples: int = 500,
    seed: Optional[int] = None
) -> ProbVC:
    """Check a single probabilistic property about a program.

    Args:
        source: C10 source (without annotations)
        property_expr: C10 boolean expression to check
        threshold: Minimum probability required
        random_vars: Variables to randomize with ranges
        n_samples: Sample budget
        seed: Random seed

    Returns:
        ProbVC with status and statistics
    """
    return check_probabilistic_vc(
        source=source,
        postcond_src=property_expr,
        threshold=threshold,
        random_vars=random_vars,
        n_samples=n_samples,
        seed=seed,
        vc_name=f"P({property_expr}) >= {threshold}"
    )


def expected_value_analysis(
    source: str,
    value_expr: str,
    random_vars: Dict[str, Tuple[int, int]],
    expected_lo: float = float('-inf'),
    expected_hi: float = float('inf'),
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze expected value of an expression in a probabilistic program.

    Args:
        source: C10 source (without annotations)
        value_expr: C10 expression whose expected value to analyze
        random_vars: Variables to randomize
        expected_lo/hi: Expected bounds on E[value_expr]
        n_samples: Sample budget
        seed: Random seed

    Returns:
        Dict with mean, CI, verdict, samples
    """
    clean_source = _strip_annotations(source)
    input_vars = list(random_vars.keys())
    ranges = dict(random_vars)

    def value_fn(inputs: Dict[str, int], result: Any) -> float:
        # Execute and extract value
        src = _replace_random_with_input(clean_source, inputs)
        parts = []
        for var, val in inputs.items():
            if val < 0:
                parts.append(f"let {var} = 0 - {-val};")
            else:
                parts.append(f"let {var} = {val};")
        full = "\n".join(parts) + "\n" + src + f"\nlet __val = {value_expr};"
        try:
            tokens = lex(full)
            parser = Parser(tokens)
            ast = parser.parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            vm = VM(chunk)
            vm.run()
            return float(vm.env.get('__val', 0))
        except Exception:
            return 0.0

    ev_result = expected_value_check(
        source=clean_source,
        input_vars=input_vars,
        value_fn=value_fn,
        bound_lo=expected_lo,
        bound_hi=expected_hi,
        n_samples=n_samples,
        confidence=0.95,
        input_ranges=ranges,
        seed=seed
    )

    mean = ev_result.metadata.get('mean', 0.0)
    mean_ci = ev_result.metadata.get('mean_ci', (0, 0))

    return {
        'mean': mean,
        'mean_ci': mean_ci,
        'expected_lo': expected_lo,
        'expected_hi': expected_hi,
        'verdict': ev_result.verdict.value,
        'samples': ev_result.total_samples,
        'in_bounds': expected_lo <= mean <= expected_hi
    }


def compare_deterministic_vs_probabilistic(
    source: str,
    fn_name: str = None,
    param_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Compare deterministic (V004) vs probabilistic (V060) verification.

    Runs both analysis modes and shows where they agree/disagree.
    """
    # Deterministic
    det_result = None
    try:
        det_result = vc_verify_function(source, fn_name)
    except Exception as e:
        det_result = None

    # Probabilistic
    prob_result = verify_probabilistic_function(
        source, fn_name=fn_name, param_ranges=param_ranges,
        n_samples=n_samples, seed=seed
    )

    return {
        'deterministic': {
            'available': det_result is not None,
            'verified': det_result.verified if det_result else None,
            'total_vcs': det_result.total_vcs if det_result else 0,
            'valid_vcs': det_result.valid_vcs if det_result else 0,
        },
        'probabilistic': {
            'verdict': prob_result.verdict.value,
            'det_vcs': len(prob_result.deterministic_vcs),
            'prob_vcs': len(prob_result.probabilistic_vcs),
        },
        'summary': prob_result.summary()
    }


def prob_hoare_triple(
    precondition: str,
    program: str,
    postcondition: str,
    threshold: float,
    random_vars: Dict[str, Tuple[int, int]],
    n_samples: int = 500,
    seed: Optional[int] = None
) -> ProbVerificationResult:
    """Verify a probabilistic Hoare triple: {P} S {Q @ threshold}.

    "If P holds, then after running S, Q holds with probability >= threshold."

    Args:
        precondition: C10 boolean expression (or "" for True)
        program: C10 program body
        postcondition: C10 boolean expression
        threshold: Minimum probability
        random_vars: Random input variables with ranges
        n_samples: Sample budget
        seed: Random seed
    """
    # Build annotated source
    parts = []
    if precondition:
        parts.append(f"requires({precondition});")
    parts.append(program)
    parts.append(f"prob_ensures({postcondition}, {_float_to_c10(threshold)});")
    source = "\n".join(parts)

    return verify_probabilistic(
        source=source,
        input_ranges=None,
        n_samples=n_samples,
        seed=seed
    )


def _float_to_c10(f: float) -> str:
    """Convert float to C10-compatible fraction representation."""
    # Express as integer fraction for C10 (no float literals)
    if f == int(f):
        return str(int(f))
    # Find simple fraction
    num = int(f * 100)
    return f"{num}/100"


def concentration_bound(
    source: str,
    value_expr: str,
    random_vars: Dict[str, Tuple[int, int]],
    epsilon: float = 0.1,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Estimate concentration bounds: P(|X - E[X]| > epsilon * E[X]).

    Uses Chebyshev's inequality and empirical estimation.
    """
    # First get expected value
    ev = expected_value_analysis(
        source, value_expr, random_vars,
        n_samples=n_samples, seed=seed
    )
    mean = ev['mean']

    if abs(mean) < 1e-10:
        return {
            'mean': mean,
            'epsilon': epsilon,
            'deviation_bound': 0.0,
            'chebyshev_bound': 1.0,
            'empirical_deviation_prob': 0.0,
            'samples': n_samples
        }

    # Collect values to estimate variance
    clean_source = _strip_annotations(source)
    input_vars = list(random_vars.keys())
    values = []

    rng = py_random.Random(seed)
    for _ in range(n_samples):
        inputs = {}
        for var in input_vars:
            lo, hi = random_vars[var]
            inputs[var] = rng.randint(lo, hi)

        src = _replace_random_with_input(clean_source, inputs)
        parts = []
        for var, val in inputs.items():
            if val < 0:
                parts.append(f"let {var} = 0 - {-val};")
            else:
                parts.append(f"let {var} = {val};")
        full = "\n".join(parts) + "\n" + src + f"\nlet __val = {value_expr};"
        try:
            tokens = lex(full)
            parser = Parser(tokens)
            ast = parser.parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            vm = VM(chunk)
            vm.run()
            values.append(float(vm.env.get('__val', 0)))
        except Exception:
            values.append(0.0)

    # Compute variance
    variance = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
    std = math.sqrt(variance)

    # Chebyshev bound: P(|X - mu| >= k*sigma) <= 1/k^2
    deviation = epsilon * abs(mean)
    if std > 0 and deviation > 0:
        k = deviation / std
        chebyshev_bound = min(1.0, 1.0 / (k * k))
    else:
        chebyshev_bound = 0.0

    # Empirical deviation probability
    deviations = sum(1 for v in values if abs(v - mean) > deviation)
    empirical_prob = deviations / max(len(values), 1)

    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'epsilon': epsilon,
        'deviation_bound': deviation,
        'chebyshev_bound': chebyshev_bound,
        'empirical_deviation_prob': empirical_prob,
        'samples': len(values)
    }


def verify_randomized_algorithm(
    source: str,
    correctness_expr: str,
    random_vars: Dict[str, Tuple[int, int]],
    min_success_prob: float = 0.5,
    n_trials: int = 100,
    amplification_rounds: int = 1,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Verify a randomized algorithm (Las Vegas / Monte Carlo style).

    For Monte Carlo algorithms: checks P(correct) >= min_success_prob.
    With amplification: repeats and takes majority vote.

    Args:
        source: C10 source with random() calls
        correctness_expr: Boolean expression for "correct output"
        random_vars: Random variables with ranges
        min_success_prob: Minimum success probability to verify
        n_trials: Number of trials per amplification round
        amplification_rounds: Number of independent repetitions
        seed: Random seed
    """
    clean_source = _strip_annotations(source)

    # Single-round success probability
    single_vc = check_probabilistic_vc(
        source=clean_source,
        postcond_src=correctness_expr,
        threshold=min_success_prob,
        random_vars=random_vars,
        n_samples=n_trials * 3,
        seed=seed,
        vc_name=f"P({correctness_expr}) >= {min_success_prob}"
    )

    single_prob = single_vc.estimated_probability or 0.0

    # Amplification analysis (mathematical, not empirical)
    if amplification_rounds > 1 and single_prob > 0.5:
        # Majority vote amplification: P(fail) = sum(C(n,k) * p^k * (1-p)^(n-k)) for k < n/2
        n = amplification_rounds
        p = single_prob
        fail_prob = 0.0
        threshold_k = n // 2  # Need majority
        for k in range(threshold_k + 1):
            fail_prob += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        amplified_prob = 1.0 - fail_prob
    else:
        amplified_prob = single_prob

    return {
        'single_round': {
            'estimated_probability': single_prob,
            'threshold': min_success_prob,
            'status': single_vc.status,
            'confidence_interval': single_vc.confidence_interval,
            'samples': single_vc.detail
        },
        'amplification': {
            'rounds': amplification_rounds,
            'amplified_probability': amplified_prob,
            'error_probability': 1.0 - amplified_prob
        },
        'verdict': 'verified' if single_vc.status == 'accept' else single_vc.status,
        'algorithm_type': 'monte_carlo' if single_prob < 1.0 else 'deterministic'
    }


def independence_test(
    source: str,
    expr_a: str,
    expr_b: str,
    random_vars: Dict[str, Tuple[int, int]],
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Test statistical independence of two program expressions.

    Checks if P(A and B) ~= P(A) * P(B) (approximately independent).
    """
    clean_source = _strip_annotations(source)
    input_vars = list(random_vars.keys())
    ranges = dict(random_vars)

    rng = py_random.Random(seed)
    count_a = 0
    count_b = 0
    count_ab = 0
    total = 0

    for _ in range(n_samples):
        inputs = {}
        for var in input_vars:
            lo, hi = ranges[var]
            inputs[var] = rng.randint(lo, hi)

        # Evaluate both expressions
        src = _replace_random_with_input(clean_source, inputs)
        parts = []
        for var, val in inputs.items():
            if val < 0:
                parts.append(f"let {var} = 0 - {-val};")
            else:
                parts.append(f"let {var} = {val};")
        check = f"\nlet __a = {expr_a};\nlet __b = {expr_b};"
        full = "\n".join(parts) + "\n" + src + check

        try:
            tokens = lex(full)
            parser = Parser(tokens)
            ast = parser.parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            vm = VM(chunk)
            vm.run()
            a_val = bool(vm.env.get('__a', 0)) and vm.env.get('__a', 0) != 0
            b_val = bool(vm.env.get('__b', 0)) and vm.env.get('__b', 0) != 0
            if a_val:
                count_a += 1
            if b_val:
                count_b += 1
            if a_val and b_val:
                count_ab += 1
            total += 1
        except Exception:
            pass

    if total == 0:
        return {'error': 'No successful samples'}

    p_a = count_a / total
    p_b = count_b / total
    p_ab = count_ab / total
    p_a_times_b = p_a * p_b

    # Measure of dependence
    if p_a_times_b > 0:
        dependence_ratio = p_ab / p_a_times_b
    else:
        dependence_ratio = float('inf') if p_ab > 0 else 1.0

    # Chi-squared test for independence
    expected_ab = p_a * p_b * total
    expected_anb = p_a * (1 - p_b) * total
    expected_nab = (1 - p_a) * p_b * total
    expected_nanb = (1 - p_a) * (1 - p_b) * total

    observed = [count_ab, count_a - count_ab, count_b - count_ab, total - count_a - count_b + count_ab]
    expected = [expected_ab, expected_anb, expected_nab, expected_nanb]

    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e) ** 2 / e

    # At 1 df, chi2 > 3.841 -> reject independence at 95%
    independent = chi2 < 3.841

    return {
        'p_a': p_a,
        'p_b': p_b,
        'p_ab': p_ab,
        'p_a_times_p_b': p_a_times_b,
        'dependence_ratio': dependence_ratio,
        'chi_squared': chi2,
        'independent': independent,
        'confidence': '95%' if independent else 'dependent at 95%',
        'samples': total
    }
