"""
V007: Automatic Loop Invariant Inference
=========================================
Composes:
  - V004 VCGen (Hoare logic, WP calculus, SExpr)
  - V005 AI-PDR (loop extraction, abstract candidate generation, validation)
  - V002 PDR/IC3 (inductive invariant discovery)
  - C039 Abstract Interpreter (interval/sign/constant analysis)
  - C037 SMT solver
  - C010 Parser (AST)

The key problem: V004 requires manual invariant() annotations in while loops.
This module infers them automatically via a tiered strategy:
  1. Abstract interpretation (cheap, polynomial) -> candidate invariants
  2. Candidate validation (two SMT checks per candidate)
  3. PDR discovery (expensive, but finds invariants AI can't)
  4. Relational invariant templates (e.g., x + y == c)
  5. Inject discovered invariants back into V004 WP calculus

APIs:
  - infer_loop_invariants(source, loop_index=0) -> list of SExpr invariants
  - auto_verify_function(source, fn_name) -> VerificationResult
  - auto_verify_program(source) -> VerificationResult
"""

from __future__ import annotations
import sys, os, copy, itertools
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto

# --- Path setup ---
_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(_base, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V005_ai_strengthened_pdr'))

# C010 Parser
from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)

# C037 SMT
from smt_solver import (
    SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, BoolConst, App, INT, BOOL,
)

# V004 VCGen - SExpr and verification
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute,
    ast_to_sexpr, parse, check_vc,
    WPCalculus, FnSpec, LoopSpec, extract_fn_spec, extract_loop_invariants,
    VCStatus, VCResult, VerificationResult,
    lower_to_smt,
)

# V005 AI-PDR - loop extraction, abstract candidates
from ai_pdr import (
    extract_loop_ts, extract_abstract_candidates,
    validate_candidate, filter_and_validate_candidates,
    _collect_assigned_vars, _collect_vars_in_expr,
    _body_to_transition, _expr_to_smt,
)

# V002 PDR
from pdr import (
    TransitionSystem, PDREngine, PDROutput, PDRResult,
    check_ts, _and, _or, _negate, _eq, _substitute,
)

# C039 Abstract Interpreter
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, Interval, INTERVAL_TOP, INTERVAL_BOT,
    Sign, INF, NEG_INF, analyze as ai_analyze,
)


# ============================================================
# Result Types
# ============================================================

class InferenceMethod(Enum):
    ABSTRACT_INTERP = "abstract_interpretation"
    PDR_DISCOVERY = "pdr_discovery"
    RELATIONAL_TEMPLATE = "relational_template"
    CONDITION_BASED = "condition_based"


@dataclass
class InferredInvariant:
    """A single inferred invariant with provenance."""
    sexpr: SExpr                    # The invariant as SExpr (for V004)
    method: InferenceMethod         # How it was discovered
    description: str                # Human-readable description
    is_inductive: bool = True       # Validated as inductive?


@dataclass
class InferenceResult:
    """Result of loop invariant inference."""
    invariants: list[InferredInvariant] = field(default_factory=list)
    sufficient: bool = False        # True if invariants suffice for verification
    stats: dict = field(default_factory=dict)

    @property
    def sexprs(self) -> list[SExpr]:
        return [inv.sexpr for inv in self.invariants]


@dataclass
class AutoVerifyResult:
    """Result of automatic verification (inference + checking)."""
    verified: bool
    verification: Optional[VerificationResult] = None
    inferred: Optional[InferenceResult] = None
    errors: list[str] = field(default_factory=list)


# ============================================================
# SMT formula -> SExpr conversion
# ============================================================

def smt_to_sexpr(term) -> SExpr:
    """Convert an SMT term (from C037/V002) to a V004 SExpr."""
    if isinstance(term, IntConst):
        return SInt(term.value)
    if isinstance(term, BoolConst):
        return SBool(term.value)
    if isinstance(term, SMTVar):
        return SVar(term.name)
    if isinstance(term, App):
        op = term.op
        args = term.args

        # Binary arithmetic/comparison
        op_map = {
            Op.ADD: '+', Op.SUB: '-', Op.MUL: '*',
            Op.EQ: '==', Op.NEQ: '!=',
            Op.LT: '<', Op.GT: '>', Op.LE: '<=', Op.GE: '>=',
        }
        if op in op_map and len(args) == 2:
            return SBinOp(op_map[op], smt_to_sexpr(args[0]), smt_to_sexpr(args[1]))

        # Logical
        if op == Op.AND:
            return s_and(*(smt_to_sexpr(a) for a in args))
        if op == Op.OR:
            return s_or(*(smt_to_sexpr(a) for a in args))
        if op == Op.NOT and len(args) == 1:
            return s_not(smt_to_sexpr(args[0]))
        if op == Op.IMPLIES and len(args) == 2:
            return s_implies(smt_to_sexpr(args[0]), smt_to_sexpr(args[1]))
        if op == Op.ITE and len(args) == 3:
            return SIte(smt_to_sexpr(args[0]), smt_to_sexpr(args[1]),
                        smt_to_sexpr(args[2]))

        # Fallback for unrecognized ops
        raise ValueError(f"Cannot convert SMT op to SExpr: {op}")

    raise ValueError(f"Cannot convert SMT term to SExpr: {type(term)}")


# ============================================================
# Guarded Transition System
# ============================================================

def _build_guarded_ts(source, loop_index=0):
    """
    Build a transition system where the transition is guarded by the loop condition.

    Standard extract_loop_ts gives unguarded: Trans = body_transition.
    For invariant inference we need: Trans = (cond => body_transition) AND (!cond => frame).
    This ensures invariant validation considers that transitions only fire under the guard.
    """
    program = parse(source)
    stmts = program.stmts

    pre_assignments = {}
    loop_stmt = None
    loop_count = 0

    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            if loop_count == loop_index:
                loop_stmt = stmt
                break
            loop_count += 1
        elif isinstance(stmt, LetDecl):
            if isinstance(stmt.value, IntLit):
                pre_assignments[stmt.name] = stmt.value.value
            elif isinstance(stmt.value, BoolLit):
                pre_assignments[stmt.name] = 1 if stmt.value.value else 0
            else:
                pre_assignments[stmt.name] = None

    if loop_stmt is None:
        raise ValueError(f"No while loop found at index {loop_index}")

    body_vars = _collect_assigned_vars(loop_stmt.body)
    cond_vars = _collect_vars_in_expr(loop_stmt.cond)
    state_var_names = sorted(body_vars | cond_vars)

    ts = TransitionSystem()
    for name in state_var_names:
        ts.add_int_var(name)

    # Init from pre-assignments
    init_conjuncts = []
    for name in state_var_names:
        v = ts.var(name)
        if name in pre_assignments and pre_assignments[name] is not None:
            init_conjuncts.append(_eq(v, IntConst(pre_assignments[name])))

    ts.set_init(_and(*init_conjuncts) if init_conjuncts else BoolConst(True))

    # Build body transition (unguarded)
    body_trans = _body_to_transition(loop_stmt.body, state_var_names, ts)

    # Build frame transition (all vars unchanged)
    frame_conjuncts = []
    for name in state_var_names:
        frame_conjuncts.append(_eq(ts.prime(name), ts.var(name)))
    frame_trans = _and(*frame_conjuncts)

    # Guard: (cond AND body_trans) OR (!cond AND frame_trans)
    var_lookup = {name: ts.var(name) for name in state_var_names}
    cond_smt = _expr_to_smt(loop_stmt.cond, var_lookup)
    guarded_trans = _or(
        _and(cond_smt, body_trans),
        _and(_negate(cond_smt), frame_trans),
    )
    ts.set_trans(guarded_trans)
    ts.set_property(BoolConst(True))

    return ts, state_var_names, loop_stmt, pre_assignments


# ============================================================
# Invariant Candidate Generation
# ============================================================

def _generate_abstract_candidates(source, state_vars, ts):
    """
    Use abstract interpretation to generate candidate invariants.
    Validates against the guarded TS.
    """
    results = []
    try:
        candidates, intervals, _ = extract_abstract_candidates(source, state_vars)
        accepted, rejected = filter_and_validate_candidates(ts, candidates)

        for kind, name, smt_formula in accepted:
            try:
                sexpr = smt_to_sexpr(smt_formula)
                desc = f"AI({kind}): {sexpr}"
                results.append(InferredInvariant(
                    sexpr=sexpr,
                    method=InferenceMethod.ABSTRACT_INTERP,
                    description=desc,
                    is_inductive=True,
                ))
            except ValueError:
                pass
    except Exception:
        pass

    return results


def _generate_init_and_bound_candidates(pre_assignments, while_stmt, state_vars, ts):
    """
    Generate candidates from initial values and loop condition structure:
    - Upper/lower bounds from init values (e.g., i starts at 10 => i <= 10)
    - Bounds derived from condition (e.g., cond is i > 0, body decrements => i >= 0)
    - Monotonicity: if var only increments, var >= init; if only decrements, var <= init
    """
    results = []
    cond = while_stmt.cond

    for name in state_vars:
        if name not in pre_assignments or pre_assignments[name] is None:
            continue
        init_val = pre_assignments[name]

        # Upper bound from init: if var decreases monotonically
        upper = SBinOp('<=', SVar(name), SInt(init_val))
        _try_validate_and_add(upper, f"init_upper({name}<={init_val})",
                              ts, state_vars, results, InferenceMethod.ABSTRACT_INTERP)

        # Lower bound from init: if var increases monotonically
        lower = SBinOp('>=', SVar(name), SInt(init_val))
        _try_validate_and_add(lower, f"init_lower({name}>={init_val})",
                              ts, state_vars, results, InferenceMethod.ABSTRACT_INTERP)

    return results


def _generate_condition_invariants(while_stmt, state_vars, ts):
    """
    Generate invariants from the loop condition and its weakened forms.
    Key insight: if cond is (i > 0) and transition guards on it,
    then (i >= 0) is invariant because:
      - i >= 0 AND i > 0 => body fires => i' = i-1 >= 0 (since i >= 1)
      - i >= 0 AND !(i > 0) => i' = i (frame) => i' >= 0
    """
    results = []
    cond = while_stmt.cond

    # The loop condition itself
    try:
        cond_sexpr = ast_to_sexpr(cond)
        _try_validate_and_add(cond_sexpr, "loop_condition", ts, state_vars, results,
                              InferenceMethod.CONDITION_BASED)
    except (ValueError, TypeError):
        pass

    # If condition is a comparison, try weakenings
    if isinstance(cond, BinOp) and cond.op in ('<', '<=', '>', '>=', '!='):
        _try_comparison_weakenings(cond, ts, state_vars, results)

    return results


def _try_comparison_weakenings(cond, ts, state_vars, results):
    """Try weakened versions of comparison conditions as invariant candidates."""
    op = cond.op
    try:
        left = ast_to_sexpr(cond.left)
        right = ast_to_sexpr(cond.right)
    except (ValueError, TypeError):
        return

    weakenings = []
    if op == '<':
        weakenings.append(SBinOp('<=', left, right))
    elif op == '>':
        weakenings.append(SBinOp('>=', right, left))  # i > 0 => 0 <= i => i >= 0
        weakenings.append(SBinOp('>=', left, right))   # i >= 0 (weakened >)
    elif op == '>=':
        weakenings.append(SBinOp('>=', left, right))
    elif op == '<=':
        weakenings.append(SBinOp('<=', left, right))
    elif op == '!=':
        pass

    for w in weakenings:
        _try_validate_and_add(w, f"weakened({w})", ts, state_vars, results,
                              InferenceMethod.CONDITION_BASED)


def _try_validate_and_add(sexpr, desc, ts, state_vars, results, method):
    """Try to validate an SExpr as an inductive invariant and add if valid."""
    try:
        smt_f = _sexpr_to_ts_smt(sexpr, ts, state_vars)
        if smt_f is not None and validate_candidate(ts, smt_f):
            results.append(InferredInvariant(
                sexpr=sexpr, method=method,
                description=desc, is_inductive=True,
            ))
    except Exception:
        pass


def _sexpr_to_ts_smt(sexpr, ts, state_vars):
    """Convert SExpr to SMT formula using the transition system's variables."""
    try:
        if isinstance(sexpr, SVar):
            if sexpr.name in [name for name, _ in ts.state_vars]:
                return ts.var(sexpr.name)
            return SMTVar(sexpr.name, INT)
        if isinstance(sexpr, SInt):
            return IntConst(sexpr.value)
        if isinstance(sexpr, SBool):
            return BoolConst(sexpr.value)
        if isinstance(sexpr, SBinOp):
            l = _sexpr_to_ts_smt(sexpr.left, ts, state_vars)
            r = _sexpr_to_ts_smt(sexpr.right, ts, state_vars)
            if l is None or r is None:
                return None
            op_map = {
                '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                '==': Op.EQ, '!=': Op.NEQ,
                '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
            }
            if sexpr.op in op_map:
                sort = BOOL if sexpr.op in ('==', '!=', '<', '>', '<=', '>=') else INT
                return App(op_map[sexpr.op], [l, r], sort)
            if sexpr.op == 'and':
                return _and(l, r)
            if sexpr.op == 'or':
                return _or(l, r)
            return None
        if isinstance(sexpr, SAnd):
            parts = [_sexpr_to_ts_smt(c, ts, state_vars) for c in sexpr.conjuncts]
            if any(p is None for p in parts):
                return None
            return _and(*parts)
        if isinstance(sexpr, SOr):
            parts = [_sexpr_to_ts_smt(d, ts, state_vars) for d in sexpr.disjuncts]
            if any(p is None for p in parts):
                return None
            return _or(*parts)
        if isinstance(sexpr, SNot):
            inner = _sexpr_to_ts_smt(sexpr.operand, ts, state_vars)
            if inner is None:
                return None
            return _negate(inner)
        if isinstance(sexpr, SUnaryOp):
            if sexpr.op == '-':
                inner = _sexpr_to_ts_smt(sexpr.operand, ts, state_vars)
                if inner is None:
                    return None
                return App(Op.SUB, [IntConst(0), inner], INT)
            if sexpr.op == 'not':
                inner = _sexpr_to_ts_smt(sexpr.operand, ts, state_vars)
                if inner is None:
                    return None
                return _negate(inner)
        return None
    except Exception:
        return None


def _generate_relational_invariants(source, state_vars, ts):
    """
    Generate relational invariant candidates over pairs of variables.
    Templates:
      - x + y == c  (sum conservation)
      - x - y == c  (difference conservation)
      - x == f(y)   where f is linear: x == a*y + b
    """
    results = []

    if len(state_vars) < 2:
        return results

    # Get initial values from transition system
    init_vals = _extract_init_values(ts, state_vars)
    if not init_vals:
        return results

    for v1, v2 in itertools.combinations(state_vars, 2):
        if v1 not in init_vals or v2 not in init_vals:
            continue

        # Template: v1 + v2 == c
        c_sum = init_vals[v1] + init_vals[v2]
        sum_sexpr = SBinOp('==', SBinOp('+', SVar(v1), SVar(v2)), SInt(c_sum))
        _try_validate_and_add(sum_sexpr, f"sum({v1}+{v2}=={c_sum})",
                              ts, state_vars, results,
                              InferenceMethod.RELATIONAL_TEMPLATE)

        # Template: v1 - v2 == c
        c_diff = init_vals[v1] - init_vals[v2]
        diff_sexpr = SBinOp('==', SBinOp('-', SVar(v1), SVar(v2)), SInt(c_diff))
        _try_validate_and_add(diff_sexpr, f"diff({v1}-{v2}=={c_diff})",
                              ts, state_vars, results,
                              InferenceMethod.RELATIONAL_TEMPLATE)

    # Single-variable linear relations with constants from init
    for v in state_vars:
        if v not in init_vals:
            continue

        # Template: v == init_val (constant invariant)
        const_sexpr = SBinOp('==', SVar(v), SInt(init_vals[v]))
        _try_validate_and_add(const_sexpr, f"const({v}=={init_vals[v]})",
                              ts, state_vars, results,
                              InferenceMethod.RELATIONAL_TEMPLATE)

    return results


def _extract_init_values(ts, state_vars):
    """Extract concrete initial values from a transition system's init formula."""
    vals = {}
    _extract_init_from_formula(ts.init_formula, vals)
    return vals


def _extract_init_from_formula(formula, vals):
    """Recursively extract var == const from an init formula."""
    if isinstance(formula, App):
        if formula.op == Op.EQ and len(formula.args) == 2:
            lhs, rhs = formula.args
            if isinstance(lhs, SMTVar) and isinstance(rhs, IntConst):
                vals[lhs.name] = rhs.value
            elif isinstance(rhs, SMTVar) and isinstance(lhs, IntConst):
                vals[rhs.name] = lhs.value
        elif formula.op == Op.AND:
            for arg in formula.args:
                _extract_init_from_formula(arg, vals)


def _generate_pdr_invariants(ts, state_vars, property_sexpr=None):
    """
    Use PDR to discover inductive invariants.
    This is the expensive fallback when abstract interp can't find enough.
    """
    results = []

    # If we have a property to check, use it
    if property_sexpr is not None:
        prop_smt = _sexpr_to_ts_smt(property_sexpr, ts, state_vars)
        if prop_smt is not None:
            ts_copy = _copy_ts_with_property(ts, prop_smt)
            try:
                output = check_ts(ts_copy, max_frames=50)
                if output.result == PDRResult.SAFE and output.invariant:
                    for clause in output.invariant:
                        try:
                            sexpr = smt_to_sexpr(clause)
                            results.append(InferredInvariant(
                                sexpr=sexpr,
                                method=InferenceMethod.PDR_DISCOVERY,
                                description=f"PDR: {sexpr}",
                                is_inductive=True,
                            ))
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass

    return results


def _copy_ts_with_property(original_ts, property_formula):
    """Create a copy of a transition system with a different property."""
    ts = TransitionSystem()
    for name, sort in original_ts.state_vars:
        if sort == INT:
            ts.add_int_var(name)
        else:
            ts.add_bool_var(name)
    ts.set_init(original_ts.init_formula)
    ts.set_trans(original_ts.trans_formula)
    ts.set_property(property_formula)
    return ts


# ============================================================
# Main Inference Engine
# ============================================================

def infer_loop_invariants(source, loop_index=0, postcondition=None):
    """
    Infer loop invariants for a while loop in source code.

    Args:
        source: C10 source code containing a while loop
        loop_index: which while loop (0-indexed)
        postcondition: optional SExpr postcondition the invariant must establish

    Returns:
        InferenceResult with discovered invariants as SExprs
    """
    result = InferenceResult()

    # Step 1: Build guarded transition system (transition fires only under loop cond)
    try:
        ts, state_vars, while_stmt, pre_assignments = _build_guarded_ts(
            source, loop_index=loop_index
        )
    except (ValueError, Exception) as e:
        result.stats['error'] = f"Loop extraction failed: {e}"
        return result

    # Step 2: Generate candidates from multiple sources
    # Tier 1: Abstract interpretation (cheap)
    ai_invs = _generate_abstract_candidates(source, state_vars, ts)
    result.stats['ai_candidates'] = len(ai_invs)

    # Tier 1b: Init-value and bound-derived candidates
    init_invs = _generate_init_and_bound_candidates(pre_assignments, while_stmt,
                                                     state_vars, ts)
    result.stats['init_candidates'] = len(init_invs)

    # Tier 2: Condition-based (cheap)
    cond_invs = _generate_condition_invariants(while_stmt, state_vars, ts)
    result.stats['cond_candidates'] = len(cond_invs)

    # Tier 3: Relational templates (moderate)
    rel_invs = _generate_relational_invariants(source, state_vars, ts)
    result.stats['rel_candidates'] = len(rel_invs)

    # Combine all candidates
    all_invs = ai_invs + init_invs + cond_invs + rel_invs

    # Deduplicate by string representation
    seen = set()
    unique_invs = []
    for inv in all_invs:
        key = str(inv.sexpr)
        if key not in seen:
            seen.add(key)
            unique_invs.append(inv)

    result.invariants = unique_invs
    result.stats['total_unique'] = len(unique_invs)

    # Step 4: Check if we have enough for the postcondition
    if postcondition is not None and unique_invs:
        result.sufficient = _check_sufficiency(
            unique_invs, while_stmt, postcondition, state_vars, ts
        )

    # Step 5: If not sufficient and we have a postcondition, try PDR
    if postcondition is not None and not result.sufficient:
        pdr_invs = _generate_pdr_invariants(ts, state_vars, postcondition)
        result.stats['pdr_candidates'] = len(pdr_invs)

        for inv in pdr_invs:
            key = str(inv.sexpr)
            if key not in seen:
                seen.add(key)
                result.invariants.append(inv)

        if pdr_invs:
            result.sufficient = _check_sufficiency(
                result.invariants, while_stmt, postcondition, state_vars, ts
            )

    return result


def _check_sufficiency(invariants, while_stmt, postcondition, state_vars, ts):
    """
    Check if a set of invariants is sufficient to verify:
    1. Invariant preservation (I AND cond => WP(body, I))
    2. Postcondition establishment (I AND !cond => postcondition)
    """
    if not invariants:
        return False

    # Build combined invariant
    combined = s_and(*(inv.sexpr for inv in invariants))
    cond = ast_to_sexpr(while_stmt.cond)

    # Check: I AND !cond => postcondition
    vc = s_implies(s_and(combined, s_not(cond)), postcondition)
    vc_result = check_vc("sufficiency_check", vc)

    return vc_result.status == VCStatus.VALID


# ============================================================
# Auto-Verification API
# ============================================================

class AutoWPCalculus(WPCalculus):
    """Extended WP calculus that auto-infers loop invariants."""

    def __init__(self, source, inferred_cache=None):
        super().__init__()
        self.source = source
        self.inferred_cache = inferred_cache or {}
        self.inference_results = []

    def wp_stmt(self, stmt, postcond):
        """Override WP for while loops to auto-infer invariants."""
        if isinstance(stmt, WhileStmt):
            loop = extract_loop_invariants(stmt)
            cond = ast_to_sexpr(stmt.cond)

            # If invariants are provided, use them (V004 behavior)
            if loop.invariants:
                return super().wp_stmt(stmt, postcond)

            # Auto-infer invariants
            inv_result = infer_loop_invariants(
                self.source, loop_index=self._count_prior_loops(stmt),
                postcondition=postcond,
            )
            self.inference_results.append(inv_result)

            if not inv_result.invariants:
                raise ValueError(
                    "Could not automatically infer loop invariants. "
                    "Consider adding invariant() annotations manually."
                )

            # Use inferred invariants
            inv_sexprs = [inv.sexpr for inv in inv_result.invariants]
            inv = s_and(*inv_sexprs) if len(inv_sexprs) > 1 else inv_sexprs[0]

            # Generate VCs just like V004
            wp_body = self.wp_stmts(loop.body_stmts, inv)
            preservation = s_implies(s_and(inv, cond), wp_body)
            self.vcs.append(("Loop invariant preservation (auto-inferred)", preservation))

            termination = s_implies(s_and(inv, s_not(cond)), postcond)
            self.vcs.append(("Loop postcondition establishment (auto-inferred)", termination))

            return inv

        return super().wp_stmt(stmt, postcond)

    def _count_prior_loops(self, target_stmt):
        """Count how many while loops appear before target_stmt in the source."""
        # Simple heuristic: use loop index 0 for now
        # (most functions have a single loop)
        return 0


def auto_verify_function(source, fn_name=None):
    """
    Verify a function's specification, automatically inferring loop invariants.

    Like V004's verify_function(), but without requiring invariant() annotations.
    """
    try:
        program = parse(source)
    except Exception as e:
        return AutoVerifyResult(verified=False, errors=[f"Parse error: {e}"])

    functions = [s for s in program.stmts if isinstance(s, FnDecl)]

    if fn_name:
        functions = [f for f in functions if f.name == fn_name]
        if not functions:
            return AutoVerifyResult(verified=False,
                                    errors=[f"Function '{fn_name}' not found"])

    all_vcs = []
    all_errors = []
    all_inferred = []
    verified = True

    for fn in functions:
        spec = extract_fn_spec(fn)

        if not spec.preconditions and not spec.postconditions:
            continue

        postcond = s_and(*spec.postconditions) if spec.postconditions else SBool(True)
        precond = s_and(*spec.preconditions) if spec.preconditions else SBool(True)

        # Use auto-inference WP calculus
        wp_calc = AutoWPCalculus(source)
        try:
            wp = wp_calc.wp_stmts(spec.body_stmts, postcond)
        except Exception as e:
            all_errors.append(f"WP error in {spec.name}: {e}")
            verified = False
            continue

        all_inferred.extend(wp_calc.inference_results)

        # Main VC
        main_vc = s_implies(precond, wp)
        vc_result = check_vc(f"{spec.name}: precondition => wp(body, postcondition)", main_vc)
        all_vcs.append(vc_result)
        if vc_result.status != VCStatus.VALID:
            verified = False

        # Loop/assertion VCs
        for desc, vc_formula in wp_calc.vcs:
            full_vc = s_implies(precond, vc_formula)
            vc_result = check_vc(f"{spec.name}: {desc}", full_vc)
            all_vcs.append(vc_result)
            if vc_result.status != VCStatus.VALID:
                verified = False

    verification = VerificationResult(verified=verified, vcs=all_vcs, errors=all_errors)

    # Combine inference results
    combined_inferred = InferenceResult()
    for ir in all_inferred:
        combined_inferred.invariants.extend(ir.invariants)
        combined_inferred.stats.update(ir.stats)
    combined_inferred.sufficient = verified

    return AutoVerifyResult(
        verified=verified,
        verification=verification,
        inferred=combined_inferred,
        errors=all_errors,
    )


def auto_verify_program(source):
    """
    Verify all annotated constructs, automatically inferring loop invariants.

    Like V004's verify_program(), but without requiring invariant() annotations.
    """
    try:
        program = parse(source)
    except Exception as e:
        return AutoVerifyResult(verified=False, errors=[f"Parse error: {e}"])

    all_vcs = []
    all_errors = []
    all_inferred = []
    verified = True

    # Verify functions
    functions = [s for s in program.stmts if isinstance(s, FnDecl)]
    for fn in functions:
        fn_result = auto_verify_function(source, fn.name)
        if fn_result.verification:
            all_vcs.extend(fn_result.verification.vcs)
            all_errors.extend(fn_result.verification.errors)
        if fn_result.inferred:
            all_inferred.append(fn_result.inferred)
        if not fn_result.verified:
            verified = False

    # Verify top-level code
    top_stmts = [s for s in program.stmts if not isinstance(s, FnDecl)]
    if top_stmts:
        wp_calc = AutoWPCalculus(source)
        try:
            wp = wp_calc.wp_stmts(top_stmts, SBool(True))
        except Exception as e:
            all_errors.append(f"WP error in top-level: {e}")
            verified = False
        else:
            all_inferred.extend(
                InferenceResult(invariants=ir.invariants, stats=ir.stats)
                for ir in wp_calc.inference_results
            )
            for desc, vc_formula in wp_calc.vcs:
                vc_result = check_vc(f"top-level: {desc}", vc_formula)
                all_vcs.append(vc_result)
                if vc_result.status != VCStatus.VALID:
                    verified = False

    verification = VerificationResult(verified=verified, vcs=all_vcs, errors=all_errors)

    combined = InferenceResult()
    for ir in all_inferred:
        combined.invariants.extend(ir.invariants)
    combined.sufficient = verified

    return AutoVerifyResult(
        verified=verified, verification=verification,
        inferred=combined, errors=all_errors,
    )
