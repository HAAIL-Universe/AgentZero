"""V123: Array Bounds Verification

Composes V120 (Array Domain Abstract Interpretation) + C037 (SMT Solver)
to generate SMT-verified proofs that array accesses are within bounds.

Pipeline:
1. V120 abstract interpretation identifies array accesses + abstract bounds
2. For each access, generate SMT verification conditions:
   - index >= 0  (lower bound)
   - index < len(array)  (upper bound)
3. SMT solver checks if bounds always hold under abstract constraints
4. Produces proof obligations with SAFE/UNSAFE/UNKNOWN verdicts
5. Counterexample generation for UNSAFE accesses

Key insight: V120 provides abstract intervals for indices and array lengths.
We encode these as SMT constraints and check if OOB is possible.
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from copy import deepcopy
import math

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V120_array_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from array_domain import (
    parse_source, Program, IntLit, VarExpr, BinExpr, UnaryExpr,
    ArrayLit, ArrayNew, ArrayRead, ArrayLen, ArrayWriteStmt,
    LetStmt, AssignStmt, IfStmt, WhileStmt, AssertStmt,
    ArrayAbstractValue, ArrayEnv, ArrayInterpreter,
    Warning, WarningKind, IntervalDomain, INF, NEG_INF,
    array_analyze, check_bounds as v120_check_bounds,
)
from smt_solver import (
    SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op,
    Sort, SortKind,
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ===========================================================================
# Verdicts and Results
# ===========================================================================

class Verdict(Enum):
    SAFE = "safe"           # Proven safe by SMT
    UNSAFE = "unsafe"       # Proven unsafe (counterexample found)
    UNKNOWN = "unknown"     # SMT couldn't decide
    AI_SAFE = "ai_safe"     # Proven safe by abstract interpretation alone


@dataclass
class BoundsObligation:
    """A single array bounds proof obligation."""
    access_line: int
    array_name: str
    index_expr: str          # Human-readable index expression
    check_type: str          # "lower" or "upper"
    verdict: Verdict
    counterexample: Optional[Dict[str, int]] = None
    abstract_index: Optional[Tuple] = None   # (lo, hi) from AI
    abstract_length: Optional[Tuple] = None  # (lo, hi) from AI
    message: str = ""


@dataclass
class AccessInfo:
    """Information about a single array access extracted during analysis."""
    line: int
    array_name: str
    index_expr: Any          # AST node
    is_read: bool            # True for read, False for write
    context_vars: Dict[str, IntervalDomain] = field(default_factory=dict)
    context_arrays: Dict[str, ArrayAbstractValue] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Full verification result for a program."""
    obligations: List[BoundsObligation]
    accesses: List[AccessInfo]
    safe_count: int
    unsafe_count: int
    unknown_count: int
    ai_safe_count: int
    all_safe: bool
    summary: str


# ===========================================================================
# Access Extractor -- walks AST to find all array accesses
# ===========================================================================

class AccessExtractor:
    """Extract all array read/write accesses from an AST."""

    def __init__(self):
        self.accesses: List[Tuple[int, str, Any, bool]] = []  # (line, array, index, is_read)

    def extract(self, program: Program) -> List[Tuple[int, str, Any, bool]]:
        self.accesses = []
        for stmt in program.stmts:
            self._visit_stmt(stmt)
        return self.accesses

    def _visit_stmt(self, stmt):
        if isinstance(stmt, LetStmt):
            self._visit_expr(stmt.value)
        elif isinstance(stmt, AssignStmt):
            self._visit_expr(stmt.value)
        elif isinstance(stmt, ArrayWriteStmt):
            self._visit_expr(stmt.index)
            self._visit_expr(stmt.value)
            self.accesses.append((stmt.line, stmt.array, stmt.index, False))
        elif isinstance(stmt, IfStmt):
            self._visit_expr(stmt.cond)
            for s in stmt.then_body:
                self._visit_stmt(s)
            if stmt.else_body:
                for s in stmt.else_body:
                    self._visit_stmt(s)
        elif isinstance(stmt, WhileStmt):
            self._visit_expr(stmt.cond)
            for s in stmt.body:
                self._visit_stmt(s)
        elif isinstance(stmt, AssertStmt):
            self._visit_expr(stmt.cond)

    def _visit_expr(self, expr):
        if isinstance(expr, ArrayRead):
            self._visit_expr(expr.array)
            self._visit_expr(expr.index)
            # Extract array name
            arr_name = self._get_name(expr.array)
            if arr_name:
                self.accesses.append((expr.line, arr_name, expr.index, True))
        elif isinstance(expr, BinExpr):
            self._visit_expr(expr.left)
            self._visit_expr(expr.right)
        elif isinstance(expr, UnaryExpr):
            self._visit_expr(expr.operand)
        elif isinstance(expr, ArrayLit):
            for e in expr.elements:
                self._visit_expr(e)
        elif isinstance(expr, ArrayNew):
            self._visit_expr(expr.size)
            self._visit_expr(expr.init_value)
        elif isinstance(expr, ArrayLen):
            self._visit_expr(expr.array)

    def _get_name(self, expr) -> Optional[str]:
        if isinstance(expr, VarExpr):
            return expr.name
        return None


# ===========================================================================
# SMT Encoder -- encodes abstract bounds as SMT constraints
# ===========================================================================

class SMTEncoder:
    """Encode array bounds verification conditions as SMT problems."""

    def __init__(self):
        self.solver = SMTSolver()

    def check_lower_bound(
        self,
        index_interval: IntervalDomain,
        context_constraints: List[Tuple[str, IntervalDomain]],
    ) -> Tuple[Verdict, Optional[Dict[str, int]]]:
        """Check if index >= 0 always holds given abstract constraints.

        Returns (verdict, counterexample_or_None).
        """
        # Quick abstract check
        if not math.isinf(index_interval.lo) and index_interval.lo >= 0:
            return Verdict.AI_SAFE, None

        # SMT check: is there an assignment where index < 0?
        s = SMTSolver()
        idx = s.Int("__idx")  # Register to get in model
        self._add_interval_constraint(s, "__idx", index_interval)
        for name, interval in context_constraints:
            s.Int(name)  # Register
            self._add_interval_constraint(s, name, interval)

        # Check: idx < 0
        s.add(App(Op.LT, [idx, IntConst(0)], BOOL))
        result = s.check()

        if result == SMTResult.UNSAT:
            return Verdict.SAFE, None
        elif result == SMTResult.SAT:
            model = s.model()
            ce = {}
            for k, v in model.items():
                if k == "__idx":
                    ce["index"] = v
                elif not k.startswith("__"):
                    ce[k] = v
            return Verdict.UNSAFE, ce
        else:
            return Verdict.UNKNOWN, None

    def check_upper_bound(
        self,
        index_interval: IntervalDomain,
        length_interval: IntervalDomain,
        context_constraints: List[Tuple[str, IntervalDomain]],
    ) -> Tuple[Verdict, Optional[Dict[str, int]]]:
        """Check if index < length always holds given abstract constraints.

        Returns (verdict, counterexample_or_None).
        """
        # Quick abstract check
        if (not math.isinf(index_interval.hi) and
            not math.isinf(length_interval.lo) and
            index_interval.hi < length_interval.lo):
            return Verdict.AI_SAFE, None

        # SMT check: is there an assignment where index >= length?
        s = SMTSolver()
        idx = s.Int("__idx")
        length = s.Int("__len")
        self._add_interval_constraint(s, "__idx", index_interval)
        self._add_interval_constraint(s, "__len", length_interval)
        for name, interval in context_constraints:
            s.Int(name)
            self._add_interval_constraint(s, name, interval)

        # Check: idx >= length
        s.add(App(Op.GE, [idx, length], BOOL))
        result = s.check()

        if result == SMTResult.UNSAT:
            return Verdict.SAFE, None
        elif result == SMTResult.SAT:
            model = s.model()
            ce = {}
            for k, v in model.items():
                if k == "__idx":
                    ce["index"] = v
                elif k == "__len":
                    ce["length"] = v
                else:
                    ce[k] = v
            return Verdict.UNSAFE, ce
        else:
            return Verdict.UNKNOWN, None

    def check_bounds_combined(
        self,
        index_interval: IntervalDomain,
        length_interval: IntervalDomain,
        context_constraints: List[Tuple[str, IntervalDomain]],
    ) -> Tuple[Verdict, Verdict, Optional[Dict], Optional[Dict]]:
        """Check both lower and upper bounds.

        Returns (lower_verdict, upper_verdict, lower_ce, upper_ce).
        """
        lv, lce = self.check_lower_bound(index_interval, context_constraints)
        uv, uce = self.check_upper_bound(index_interval, length_interval, context_constraints)
        return lv, uv, lce, uce

    def _add_interval_constraint(self, solver: SMTSolver, name: str, interval: IntervalDomain):
        """Add constraints: lo <= var <= hi."""
        var = Var(name, INT)
        if not math.isinf(interval.lo):
            solver.add(App(Op.GE, [var, IntConst(int(interval.lo))], BOOL))
        if not math.isinf(interval.hi):
            solver.add(App(Op.LE, [var, IntConst(int(interval.hi))], BOOL))


# ===========================================================================
# Bounds-Tracking Interpreter -- extends V120 to record access contexts
# ===========================================================================

class BoundsTrackingInterpreter(ArrayInterpreter):
    """Extends ArrayInterpreter to record abstract state at each array access."""

    def __init__(self, max_iterations: int = 50):
        super().__init__(max_iterations=max_iterations)
        self.access_contexts: List[AccessInfo] = []

    def analyze(self, source: str) -> dict:
        self.access_contexts = []
        result = super().analyze(source)
        result['access_contexts'] = self.access_contexts
        return result

    def _eval_expr(self, expr, env: ArrayEnv) -> IntervalDomain:
        if isinstance(expr, ArrayRead):
            arr_name = None
            if isinstance(expr.array, VarExpr):
                arr_name = expr.array.name
            if arr_name and arr_name in env.arrays:
                # Record access context before parent eval
                self.access_contexts.append(AccessInfo(
                    line=expr.line,
                    array_name=arr_name,
                    index_expr=expr.index,
                    is_read=True,
                    context_vars={k: v for k, v in env.scalars.items()},
                    context_arrays={k: v.copy() for k, v in env.arrays.items()},
                ))
        return super()._eval_expr(expr, env)

    def _interpret_array_write(self, stmt: ArrayWriteStmt, env: ArrayEnv) -> ArrayEnv:
        if stmt.array in env.arrays:
            self.access_contexts.append(AccessInfo(
                line=stmt.line,
                array_name=stmt.array,
                index_expr=stmt.index,
                is_read=False,
                context_vars={k: v for k, v in env.scalars.items()},
                context_arrays={k: v.copy() for k, v in env.arrays.items()},
            ))
        return super()._interpret_array_write(stmt, env)


# ===========================================================================
# Expression to interval evaluator (from abstract state)
# ===========================================================================

def eval_index_interval(expr, env: ArrayEnv) -> IntervalDomain:
    """Evaluate an expression to an interval given an abstract environment."""
    if isinstance(expr, IntLit):
        return IntervalDomain(expr.value, expr.value)
    elif isinstance(expr, VarExpr):
        return env.get_scalar(expr.name)
    elif isinstance(expr, BinExpr):
        left = eval_index_interval(expr.left, env)
        right = eval_index_interval(expr.right, env)
        op = expr.op
        if op == '+':
            return left.add(right)
        elif op == '-':
            return left.sub(right)
        elif op == '*':
            return left.mul(right)
        elif op == '/':
            if right.contains(0):
                return IntervalDomain(NEG_INF, INF)
            return left.div(right)
        else:
            return IntervalDomain(NEG_INF, INF)
    elif isinstance(expr, UnaryExpr):
        if expr.op == '-':
            operand = eval_index_interval(expr.operand, env)
            return IntervalDomain(-operand.hi, -operand.lo)
        return IntervalDomain(NEG_INF, INF)
    elif isinstance(expr, ArrayLen):
        if isinstance(expr.array, VarExpr):
            arr = env.get_array(expr.array.name)
            return arr.length
        return IntervalDomain(0, INF)
    else:
        return IntervalDomain(NEG_INF, INF)


def expr_to_str(expr) -> str:
    """Convert an expression AST to a human-readable string."""
    if isinstance(expr, IntLit):
        return str(expr.value)
    elif isinstance(expr, VarExpr):
        return expr.name
    elif isinstance(expr, BinExpr):
        return f"({expr_to_str(expr.left)} {expr.op} {expr_to_str(expr.right)})"
    elif isinstance(expr, UnaryExpr):
        return f"(-{expr_to_str(expr.operand)})"
    elif isinstance(expr, ArrayLen):
        if isinstance(expr.array, VarExpr):
            return f"len({expr.array.name})"
        return "len(?)"
    else:
        return "?"


# ===========================================================================
# Main Verification Engine
# ===========================================================================

class ArrayBoundsVerifier:
    """Verify array bounds safety by composing V120 + C037."""

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.encoder = SMTEncoder()

    def verify(self, source: str) -> VerificationResult:
        """Full verification of all array accesses in a program.

        Pipeline:
        1. Run V120 abstract interpretation to get abstract state
        2. Extract all array accesses with their abstract contexts
        3. For each access, generate and check SMT proof obligations
        4. Produce verification result with verdicts
        """
        # Phase 1: Abstract interpretation
        interp = BoundsTrackingInterpreter(max_iterations=self.max_iterations)
        ai_result = interp.analyze(source)
        final_env = ai_result['env']

        # Phase 2: Extract accesses from AST
        program = parse_source(source)
        extractor = AccessExtractor()
        raw_accesses = extractor.extract(program)

        # Phase 3: Match accesses with contexts from tracking interpreter
        # Use the access contexts recorded during interpretation
        access_contexts = ai_result.get('access_contexts', [])

        # Phase 4: Generate and check proof obligations
        obligations: List[BoundsObligation] = []
        access_infos: List[AccessInfo] = []

        # Deduplicate accesses by (line, array_name, is_read).
        # For loop accesses seen multiple times, join the index contexts
        # to get the union of all possible values at that access point.
        seen: Dict[Tuple, int] = {}  # key -> index in access_infos

        for ctx in access_contexts:
            key = (ctx.line, ctx.array_name, ctx.is_read)
            if key in seen:
                # Join contexts: union of variable ranges (covers all iterations)
                idx = seen[key]
                old = access_infos[idx]
                merged_vars = {}
                all_keys = set(old.context_vars.keys()) | set(ctx.context_vars.keys())
                for k in all_keys:
                    v1 = old.context_vars.get(k, IntervalDomain(NEG_INF, INF))
                    v2 = ctx.context_vars.get(k, IntervalDomain(NEG_INF, INF))
                    merged_vars[k] = v1.join(v2)
                merged_arrays = {}
                arr_keys = set(old.context_arrays.keys()) | set(ctx.context_arrays.keys())
                for k in arr_keys:
                    v1 = old.context_arrays.get(k, ArrayAbstractValue.top())
                    v2 = ctx.context_arrays.get(k, ArrayAbstractValue.top())
                    merged_arrays[k] = v1.join(v2)
                access_infos[idx] = AccessInfo(
                    line=ctx.line,
                    array_name=ctx.array_name,
                    index_expr=ctx.index_expr,
                    is_read=ctx.is_read,
                    context_vars=merged_vars,
                    context_arrays=merged_arrays,
                )
                continue
            seen[key] = len(access_infos)
            access_infos.append(ctx)

        # Also check raw accesses not captured by the interpreter
        # (e.g., in dead branches)
        for line, arr_name, index_expr, is_read in raw_accesses:
            key = (line, arr_name, is_read)
            if key not in seen:
                seen.add(key)
                access_infos.append(AccessInfo(
                    line=line,
                    array_name=arr_name,
                    index_expr=index_expr,
                    is_read=is_read,
                    context_vars=dict(final_env.scalars),
                    context_arrays={k: v.copy() for k, v in final_env.arrays.items()},
                ))

        # Generate obligations for each unique access
        for access in access_infos:
            env = ArrayEnv()
            env.scalars = dict(access.context_vars)
            env.arrays = {k: v.copy() for k, v in access.context_arrays.items()}

            index_interval = eval_index_interval(access.index_expr, env)
            arr_val = env.get_array(access.array_name)
            length_interval = arr_val.length

            index_str = expr_to_str(access.index_expr)

            # Build context constraints for SMT
            context = []
            for var_name, interval in env.scalars.items():
                if not interval.is_top():
                    context.append((var_name, interval))

            abs_idx = (
                index_interval.lo if not math.isinf(index_interval.lo) else None,
                index_interval.hi if not math.isinf(index_interval.hi) else None,
            )
            abs_len = (
                length_interval.lo if not math.isinf(length_interval.lo) else None,
                length_interval.hi if not math.isinf(length_interval.hi) else None,
            )

            # Lower bound: index >= 0
            lv, lce = self.encoder.check_lower_bound(index_interval, context)
            obligations.append(BoundsObligation(
                access_line=access.line,
                array_name=access.array_name,
                index_expr=index_str,
                check_type="lower",
                verdict=lv,
                counterexample=lce,
                abstract_index=abs_idx,
                abstract_length=abs_len,
                message=self._make_message("lower", access.array_name, index_str, lv, lce),
            ))

            # Upper bound: index < len(array)
            uv, uce = self.encoder.check_upper_bound(index_interval, length_interval, context)
            obligations.append(BoundsObligation(
                access_line=access.line,
                array_name=access.array_name,
                index_expr=index_str,
                check_type="upper",
                verdict=uv,
                counterexample=uce,
                abstract_index=abs_idx,
                abstract_length=abs_len,
                message=self._make_message("upper", access.array_name, index_str, uv, uce),
            ))

        # Summarize
        safe = sum(1 for o in obligations if o.verdict == Verdict.SAFE)
        unsafe = sum(1 for o in obligations if o.verdict == Verdict.UNSAFE)
        unknown = sum(1 for o in obligations if o.verdict == Verdict.UNKNOWN)
        ai_safe = sum(1 for o in obligations if o.verdict == Verdict.AI_SAFE)

        return VerificationResult(
            obligations=obligations,
            accesses=access_infos,
            safe_count=safe,
            unsafe_count=unsafe,
            unknown_count=unknown,
            ai_safe_count=ai_safe,
            all_safe=(unsafe == 0 and unknown == 0),
            summary=self._make_summary(obligations, safe, unsafe, unknown, ai_safe),
        )

    def _make_message(self, check_type, arr_name, index_str, verdict, ce):
        if verdict == Verdict.AI_SAFE:
            return f"{arr_name}[{index_str}] {check_type} bound: SAFE (abstract interpretation)"
        elif verdict == Verdict.SAFE:
            return f"{arr_name}[{index_str}] {check_type} bound: SAFE (SMT verified)"
        elif verdict == Verdict.UNSAFE:
            ce_str = ", ".join(f"{k}={v}" for k, v in (ce or {}).items())
            return f"{arr_name}[{index_str}] {check_type} bound: UNSAFE (counterexample: {ce_str})"
        else:
            return f"{arr_name}[{index_str}] {check_type} bound: UNKNOWN"

    def _make_summary(self, obligations, safe, unsafe, unknown, ai_safe):
        total = len(obligations)
        lines = [
            f"Array Bounds Verification: {total} obligations",
            f"  SAFE (SMT): {safe}",
            f"  SAFE (AI):  {ai_safe}",
            f"  UNSAFE:     {unsafe}",
            f"  UNKNOWN:    {unknown}",
        ]
        if unsafe == 0 and unknown == 0:
            lines.append("  Result: ALL BOUNDS PROVEN SAFE")
        elif unsafe > 0:
            lines.append("  Result: BOUNDS VIOLATIONS FOUND")
        else:
            lines.append("  Result: SOME BOUNDS UNRESOLVED")
        return "\n".join(lines)


# ===========================================================================
# Proof Certificate Generation
# ===========================================================================

@dataclass
class BoundsCertificate:
    """A proof certificate for array bounds safety."""
    program_source: str
    obligations: List[Dict]
    all_safe: bool
    method: str  # "ai+smt"

    def to_dict(self) -> dict:
        return {
            'program': self.program_source,
            'obligations': self.obligations,
            'all_safe': self.all_safe,
            'method': self.method,
        }

    @staticmethod
    def from_result(source: str, result: VerificationResult) -> BoundsCertificate:
        obs = []
        for o in result.obligations:
            obs.append({
                'line': o.access_line,
                'array': o.array_name,
                'index': o.index_expr,
                'check': o.check_type,
                'verdict': o.verdict.value,
                'counterexample': o.counterexample,
                'abstract_index': o.abstract_index,
                'abstract_length': o.abstract_length,
            })
        return BoundsCertificate(
            program_source=source,
            obligations=obs,
            all_safe=result.all_safe,
            method="ai+smt",
        )


def check_certificate(cert: BoundsCertificate) -> Tuple[bool, List[str]]:
    """Independently verify a bounds certificate by re-running analysis.

    Returns (valid, issues).
    """
    issues = []
    verifier = ArrayBoundsVerifier()
    result = verifier.verify(cert.program_source)

    # Check that all_safe matches
    if result.all_safe != cert.all_safe:
        issues.append(f"all_safe mismatch: cert={cert.all_safe}, recheck={result.all_safe}")

    # Check each obligation
    cert_obs = {(o['line'], o['array'], o['check']): o for o in cert.obligations}
    recheck_obs = {(o.access_line, o.array_name, o.check_type): o for o in result.obligations}

    for key, co in cert_obs.items():
        if key not in recheck_obs:
            issues.append(f"Certificate obligation {key} not found in recheck")
            continue
        ro = recheck_obs[key]
        # A certificate claiming SAFE should still be SAFE on recheck
        if co['verdict'] in ('safe', 'ai_safe') and ro.verdict == Verdict.UNSAFE:
            issues.append(f"Obligation {key} claimed safe but recheck found unsafe")

    return len(issues) == 0, issues


# ===========================================================================
# High-level APIs
# ===========================================================================

def verify_bounds(source: str) -> VerificationResult:
    """Verify all array bounds in a program. Main API."""
    verifier = ArrayBoundsVerifier()
    return verifier.verify(source)


def check_access_safe(source: str, line: int) -> Optional[BoundsObligation]:
    """Check if a specific array access (by line number) is safe."""
    result = verify_bounds(source)
    for o in result.obligations:
        if o.access_line == line and o.verdict == Verdict.UNSAFE:
            return o
    # Check if any obligation at this line exists
    at_line = [o for o in result.obligations if o.access_line == line]
    if at_line:
        # All safe
        return None
    return None


def find_unsafe_accesses(source: str) -> List[BoundsObligation]:
    """Find all array accesses that may be out of bounds."""
    result = verify_bounds(source)
    return [o for o in result.obligations if o.verdict == Verdict.UNSAFE]


def certify_bounds(source: str) -> BoundsCertificate:
    """Generate a proof certificate for array bounds."""
    result = verify_bounds(source)
    return BoundsCertificate.from_result(source, result)


def compare_ai_vs_smt(source: str) -> dict:
    """Compare AI-only vs AI+SMT verification.

    Shows how many additional obligations SMT can prove beyond AI alone.
    """
    result = verify_bounds(source)

    ai_only_safe = result.ai_safe_count
    smt_safe = result.safe_count
    total = len(result.obligations)
    unsafe = result.unsafe_count
    unknown = result.unknown_count

    return {
        'total_obligations': total,
        'ai_safe': ai_only_safe,
        'smt_safe': smt_safe,
        'total_safe': ai_only_safe + smt_safe,
        'unsafe': unsafe,
        'unknown': unknown,
        'all_safe': result.all_safe,
        'smt_additional': smt_safe,  # These are the ones AI couldn't prove but SMT could
        'summary': result.summary,
    }


def bounds_summary(source: str) -> str:
    """Human-readable summary of array bounds verification."""
    result = verify_bounds(source)
    lines = ["=== Array Bounds Verification Summary ===", ""]
    lines.append(result.summary)
    lines.append("")

    if result.unsafe_count > 0:
        lines.append("Unsafe Accesses:")
        for o in result.obligations:
            if o.verdict == Verdict.UNSAFE:
                lines.append(f"  Line {o.access_line}: {o.message}")
        lines.append("")

    if result.ai_safe_count > 0:
        lines.append(f"AI-proven safe: {result.ai_safe_count} obligations")
    if result.safe_count > 0:
        lines.append(f"SMT-proven safe: {result.safe_count} obligations")

    return "\n".join(lines)


def verify_with_context(
    source: str,
    extra_constraints: Optional[Dict[str, Tuple[int, int]]] = None,
) -> VerificationResult:
    """Verify bounds with extra user-provided variable constraints.

    extra_constraints: dict mapping var_name -> (lo, hi) intervals
    """
    # Parse and run AI
    verifier = ArrayBoundsVerifier()
    result = verifier.verify(source)

    if extra_constraints is None or not extra_constraints:
        return result

    # Re-check obligations with extra constraints
    encoder = SMTEncoder()
    new_obligations = []

    for o in result.obligations:
        if o.verdict in (Verdict.AI_SAFE, Verdict.SAFE):
            new_obligations.append(o)
            continue

        # Build constraint list from abstract info + extra
        context = []
        for name, (lo, hi) in extra_constraints.items():
            context.append((name, IntervalDomain(lo, hi)))

        # Reconstruct index and length intervals
        idx_lo = o.abstract_index[0] if o.abstract_index and o.abstract_index[0] is not None else NEG_INF
        idx_hi = o.abstract_index[1] if o.abstract_index and o.abstract_index[1] is not None else INF
        index_interval = IntervalDomain(idx_lo, idx_hi)

        len_lo = o.abstract_length[0] if o.abstract_length and o.abstract_length[0] is not None else 0
        len_hi = o.abstract_length[1] if o.abstract_length and o.abstract_length[1] is not None else INF
        length_interval = IntervalDomain(len_lo, len_hi)

        if o.check_type == "lower":
            v, ce = encoder.check_lower_bound(index_interval, context)
        else:
            v, ce = encoder.check_upper_bound(index_interval, length_interval, context)

        new_obligations.append(BoundsObligation(
            access_line=o.access_line,
            array_name=o.array_name,
            index_expr=o.index_expr,
            check_type=o.check_type,
            verdict=v,
            counterexample=ce,
            abstract_index=o.abstract_index,
            abstract_length=o.abstract_length,
            message=o.message if v == o.verdict else
                verifier._make_message(o.check_type, o.array_name, o.index_expr, v, ce),
        ))

    safe = sum(1 for o in new_obligations if o.verdict == Verdict.SAFE)
    unsafe = sum(1 for o in new_obligations if o.verdict == Verdict.UNSAFE)
    unknown = sum(1 for o in new_obligations if o.verdict == Verdict.UNKNOWN)
    ai_safe = sum(1 for o in new_obligations if o.verdict == Verdict.AI_SAFE)

    return VerificationResult(
        obligations=new_obligations,
        accesses=result.accesses,
        safe_count=safe,
        unsafe_count=unsafe,
        unknown_count=unknown,
        ai_safe_count=ai_safe,
        all_safe=(unsafe == 0 and unknown == 0),
        summary=verifier._make_summary(new_obligations, safe, unsafe, unknown, ai_safe),
    )
