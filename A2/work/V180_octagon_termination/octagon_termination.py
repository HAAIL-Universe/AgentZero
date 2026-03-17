"""
V180: Octagon-Based Termination Analysis

Proves program termination using relational ranking functions discovered via
octagon abstract interpretation. Standard termination analysis (V025) finds
single-variable or simple two-variable ranking functions. This module handles
cases where termination depends on relational properties between variables
(e.g., x - y decreasing, x + y bounded).

Composes:
  - V025 (termination analysis) -- ranking function verification, loop extraction
  - V173 (octagon domain) -- relational constraint discovery
  - C010 (parser) -- AST
  - C037 (SMT solver) -- verification

Key capabilities:
  1. Octagon-guided candidate generation: abstract interpretation discovers
     relational invariants that seed better ranking function candidates
  2. Relational ranking functions: R = x - y, R = x + y + c, etc.
  3. Octagon-bounded termination: use relational bounds to prove ranking >= 0
  4. Invariant-strengthened verification: octagon invariants narrow the SMT search
  5. Adaptive analysis: tries V025 first, falls back to octagon-guided if needed
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set
from fractions import Fraction
from enum import Enum

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V025_termination_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from termination import (
    TermResult, RankingFunction, LexRankingFunction, TerminationResult,
    LoopTermResult, parse, extract_loop_info, verify_ranking_function,
    generate_candidates, find_ranking_function, prove_termination,
    analyze_termination, _build_ranking_expr, _coefficients_to_str,
    _expr_to_smt, _find_all_loops, _register_vars, _rebuild_in_solver,
    find_lexicographic_ranking, _verify_lex_pair,
)
from octagon import (
    Octagon, OctConstraint, OctExpr, OctagonInterpreter, OctAnalysisResult,
    INF, _bar,
)
from stack_vm import lex, Parser
from smt_solver import (
    SMTSolver, SMTResult, Op, BOOL, INT,
    IntConst, BoolConst, App,
    Var as SMTVar,
)


# ---------------------------------------------------------------------------
# AST -> Octagon Program Translation
# ---------------------------------------------------------------------------

def _ast_expr_to_oct_expr(expr):
    """Convert C010 AST expression to octagon tuple-based expression for OctagonInterpreter."""
    t = type(expr).__name__
    if t == 'IntLit':
        return ('const', expr.value)
    if t == 'BoolLit':
        return ('const', 1 if expr.value else 0)
    if t == 'Var':
        return ('var', expr.name)
    if t == 'BinOp':
        left = _ast_expr_to_oct_expr(expr.left)
        right = _ast_expr_to_oct_expr(expr.right)
        op_map = {
            '+': 'add', '-': 'sub', '*': 'mul',
            '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge',
            '==': 'eq', '!=': 'ne',
            'and': 'and', 'or': 'or',
        }
        oct_op = op_map.get(expr.op, expr.op)
        return (oct_op, left, right)
    if t == 'UnaryOp':
        operand = _ast_expr_to_oct_expr(expr.operand)
        if expr.op == '-':
            return ('sub', ('const', 0), operand)
        if expr.op == 'not':
            return ('not', operand)
    return ('const', 0)  # fallback


def _block_to_list(block):
    """Extract statement list from a Block or return as-is if already a list."""
    t = type(block).__name__
    if t == 'Block':
        return block.stmts
    if isinstance(block, list):
        return block
    return [block]


def _stmts_to_oct(stmts_list):
    """Convert a list of AST statements to a single octagon tuple-statement.

    Returns a single statement if only one, or ('seq', s1, s2, ...) if multiple.
    """
    oct_stmts = [_ast_stmt_to_oct_stmt(s) for s in stmts_list]
    if len(oct_stmts) == 0:
        return ('skip',)
    if len(oct_stmts) == 1:
        return oct_stmts[0]
    return ('seq',) + tuple(oct_stmts)


def _ast_stmt_to_oct_stmt(stmt):
    """Convert C010 AST statement to octagon tuple-based statement."""
    t = type(stmt).__name__
    if t == 'LetDecl':
        expr = _ast_expr_to_oct_expr(stmt.value)
        return ('assign', stmt.name, expr)
    if t == 'Assign':
        expr = _ast_expr_to_oct_expr(stmt.value)
        return ('assign', stmt.name, expr)
    if t == 'WhileStmt':
        cond = _ast_expr_to_oct_expr(stmt.cond)
        body = _stmts_to_oct(_block_to_list(stmt.body))
        return ('while', cond, body)
    if t == 'IfStmt':
        cond = _ast_expr_to_oct_expr(stmt.cond)
        then_body = _stmts_to_oct(_block_to_list(stmt.then_body))
        else_body = _stmts_to_oct(_block_to_list(stmt.else_body)) if stmt.else_body else ('skip',)
        return ('if', cond, then_body, else_body)
    if t == 'Block':
        return _stmts_to_oct(stmt.stmts)
    if t == 'ExprStmt':
        return ('skip',)
    return ('skip',)  # fallback


def _ast_to_oct_program(stmts):
    """Convert list of C010 AST statements to octagon tuple-program."""
    return _stmts_to_oct(stmts)


# ---------------------------------------------------------------------------
# Octagon Invariant Extraction
# ---------------------------------------------------------------------------

def extract_octagon_invariants(source, loop_index=0):
    """Run octagon abstract interpretation on source and extract loop invariants.

    Returns dict with:
      - 'pre_loop': Octagon state just before the loop
      - 'loop_body': Octagon state at loop entry (fixpoint)
      - 'post_loop': Octagon state after the loop
      - 'constraints': List of OctConstraint from loop body fixpoint
      - 'intervals': Dict of var -> (lo, hi) intervals
      - 'difference_bounds': Dict of (v1,v2) -> (lo, hi) for v1-v2
      - 'sum_bounds': Dict of (v1,v2) -> (lo, hi) for v1+v2
    """
    ast = parse(source)
    stmts = ast.stmts

    # Find the target loop
    all_loops = _find_all_loops(stmts)
    if loop_index >= len(all_loops):
        return None

    target_loop = all_loops[loop_index]

    # Build octagon program from AST
    oct_program = _ast_to_oct_program(stmts)

    # Run octagon interpreter
    interp = OctagonInterpreter(max_iterations=100, widen_delay=3)
    result = interp.analyze(oct_program)

    final_state = result.final_state

    # Also run just the pre-loop portion to get pre-loop state
    pre_stmts = []
    for s in stmts:
        if s is target_loop:
            break
        pre_stmts.append(s)

    if pre_stmts:
        pre_program = _ast_to_oct_program(pre_stmts)
        pre_result = interp.analyze(pre_program)
        pre_state = pre_result.final_state
    else:
        pre_state = Octagon.top()

    # Extract relational information
    variables = list(pre_state.variables() | final_state.variables())
    variables.sort()

    intervals = {}
    difference_bounds = {}
    sum_bounds = {}

    # Use pre_state as the loop entry approximation
    state = pre_state

    for v in variables:
        lo, hi = state.get_bounds(v)
        if lo is not None or hi is not None:
            intervals[v] = (lo, hi)

    for i, v1 in enumerate(variables):
        for v2 in variables[i+1:]:
            lo, hi = state.get_difference_bound(v1, v2)
            if lo is not None or hi is not None:
                difference_bounds[(v1, v2)] = (lo, hi)
            lo, hi = state.get_sum_bound(v1, v2)
            if lo is not None or hi is not None:
                sum_bounds[(v1, v2)] = (lo, hi)

    constraints = state.extract_constraints() if not state.is_bot() else []

    return {
        'pre_loop': pre_state,
        'loop_body': state,
        'post_loop': final_state,
        'constraints': constraints,
        'intervals': intervals,
        'difference_bounds': difference_bounds,
        'sum_bounds': sum_bounds,
        'variables': variables,
    }


# ---------------------------------------------------------------------------
# Octagon-Guided Candidate Generation
# ---------------------------------------------------------------------------

def _oct_constraint_to_coefficients(c):
    """Convert an OctConstraint to ranking function coefficient dicts.

    Returns list of candidate coefficient dicts derived from this constraint.
    E.g., if x - y <= 5, candidates include {x:1, y:-1, _const:0} and
    {x:-1, y:1, _const:5} (the slack).
    """
    candidates = []

    if c.var2 is None:
        # Unary: coeff1 * var1 <= bound
        if c.coeff1 == 1:
            # var1 <= bound -> candidate: bound - var1 (decreasing as var1 grows)
            candidates.append({c.var1: -1, '_const': int(c.bound)})
            # Also: var1 itself
            candidates.append({c.var1: 1, '_const': 0})
        elif c.coeff1 == -1:
            # -var1 <= bound -> var1 >= -bound -> candidate: var1 + bound
            candidates.append({c.var1: 1, '_const': int(c.bound)})
            candidates.append({c.var1: -1, '_const': 0})
    else:
        # Binary: coeff1 * var1 + coeff2 * var2 <= bound
        # The expression coeff1*var1 + coeff2*var2 is bounded above by bound
        # Candidate: bound - (coeff1*var1 + coeff2*var2)
        candidates.append({
            c.var1: -c.coeff1,
            c.var2: -c.coeff2,
            '_const': int(c.bound),
        })
        # Also the expression itself as a candidate
        candidates.append({
            c.var1: c.coeff1,
            c.var2: c.coeff2,
            '_const': 0,
        })

    return candidates


def generate_octagon_candidates(source, loop_index=0):
    """Generate ranking function candidates using octagon invariants.

    Returns list of coefficient dicts ordered by likelihood.
    """
    oct_info = extract_octagon_invariants(source, loop_index)
    if oct_info is None:
        return []

    loop_info = extract_loop_info(source, loop_index)
    state_vars = loop_info['state_vars']
    pre_assignments = loop_info.get('pre_assignments', {})

    candidates = []
    seen = set()

    def add_candidate(c):
        # Normalize: remove zero coefficients
        c = {k: v for k, v in c.items() if v != 0 or k == '_const'}
        if '_const' not in c:
            c['_const'] = 0
        # Only keep candidates involving state vars
        var_keys = [k for k in c if k != '_const']
        if not var_keys:
            return
        if not any(v in state_vars for v in var_keys):
            return
        key = _coefficients_to_str(c)
        if key not in seen:
            seen.add(key)
            candidates.append(c)

    # Tier 1: Difference-bound derived candidates (relational)
    for (v1, v2), (lo, hi) in oct_info.get('difference_bounds', {}).items():
        # v1 - v2 bounded
        add_candidate({v1: 1, v2: -1, '_const': 0})
        add_candidate({v1: -1, v2: 1, '_const': 0})
        if hi is not None:
            add_candidate({v1: -1, v2: 1, '_const': int(hi)})
        if lo is not None:
            add_candidate({v1: 1, v2: -1, '_const': -int(lo)})

    # Tier 2: Sum-bound derived candidates
    for (v1, v2), (lo, hi) in oct_info.get('sum_bounds', {}).items():
        add_candidate({v1: 1, v2: 1, '_const': 0})
        add_candidate({v1: -1, v2: -1, '_const': 0})
        if hi is not None:
            add_candidate({v1: -1, v2: -1, '_const': int(hi)})
        if lo is not None:
            add_candidate({v1: 1, v2: 1, '_const': -int(lo)})

    # Tier 3: Interval-derived candidates (same as V025 but with octagon precision)
    for v, (lo, hi) in oct_info.get('intervals', {}).items():
        add_candidate({v: 1, '_const': 0})
        add_candidate({v: -1, '_const': 0})
        if hi is not None:
            add_candidate({v: -1, '_const': int(hi)})
        if lo is not None:
            add_candidate({v: 1, '_const': -int(lo)})

    # Tier 4: Constraint-derived candidates
    for c in oct_info.get('constraints', []):
        for cand in _oct_constraint_to_coefficients(c):
            add_candidate(cand)

    # Tier 5: Pre-assignment derived (constants from init values)
    for v in state_vars:
        if v in pre_assignments:
            c_val = pre_assignments[v]
            add_candidate({v: -1, '_const': c_val})
            add_candidate({v: 1, '_const': -c_val})

    # Tier 6: All pairs of state vars with small coefficients
    for i, v1 in enumerate(state_vars):
        for v2 in state_vars[i+1:]:
            for c1 in [-1, 1]:
                for c2 in [-1, 1]:
                    add_candidate({v1: c1, v2: c2, '_const': 0})
                    # With constant offsets from pre-assignments
                    for v in [v1, v2]:
                        if v in pre_assignments:
                            add_candidate({v1: c1, v2: c2, '_const': pre_assignments[v]})

    return candidates


# ---------------------------------------------------------------------------
# Octagon-Strengthened Ranking Verification
# ---------------------------------------------------------------------------

def verify_ranking_with_octagon(loop_info, coefficients, oct_invariants):
    """Verify a ranking function using octagon invariants to strengthen the proof.

    The octagon invariants provide relational bounds that can help prove:
    1. Boundedness: R(s) >= 0 under octagon constraints (not just loop condition)
    2. Decrease: R(s) - R(s') >= 1 under octagon constraints

    Returns (bounded_ok, decreasing_ok).
    """
    # First try standard verification
    bounded, decreasing = verify_ranking_function(loop_info, coefficients)

    if bounded and decreasing:
        return True, True

    # If standard fails, try with octagon strengthening
    state_vars = loop_info['state_vars']
    var_lookup = loop_info['var_lookup']
    next_state = loop_info['next_state']
    cond_smt = loop_info['cond_smt']

    # Build ranking expression
    ranking = _build_ranking_expr(coefficients, var_lookup)
    if ranking is None:
        return False, False

    # Extract octagon constraints as SMT assertions
    oct_state = oct_invariants.get('loop_body') or oct_invariants.get('pre_loop')
    if oct_state is None or oct_state.is_bot() or oct_state.is_top():
        return bounded, decreasing

    oct_constraints = oct_state.extract_constraints()

    # Try strengthened boundedness check
    if not bounded:
        s = SMTSolver()
        for v in state_vars:
            s.Int(v)

        cond_rebuilt = _rebuild_in_solver(cond_smt, s)
        s.add(cond_rebuilt)

        # Assert octagon invariants
        for c in oct_constraints:
            smt_c = _oct_constraint_to_smt(c, s)
            if smt_c is not None:
                s.add(smt_c)

        # Assert ranking < 0 (negate bounded condition)
        ranking_rebuilt = _rebuild_in_solver(ranking, s)
        s.add(App(Op.LT, [ranking_rebuilt, IntConst(0)], BOOL))

        bounded = s.check() == SMTResult.UNSAT

    # Try strengthened decrease check
    if not decreasing:
        s = SMTSolver()
        for v in state_vars:
            s.Int(v)

        cond_rebuilt = _rebuild_in_solver(cond_smt, s)
        s.add(cond_rebuilt)

        # Assert octagon invariants
        for c in oct_constraints:
            smt_c = _oct_constraint_to_smt(c, s)
            if smt_c is not None:
                s.add(smt_c)

        # Assert transition relation
        for v in state_vars:
            if v in next_state:
                nv = s.Int(f"{v}_next")
                ns_expr = _rebuild_in_solver(next_state[v], s)
                s.add(App(Op.EQ, [nv, ns_expr], BOOL))

        # Build R(next) using _next variables
        next_var_lookup = {v: s.Int(f"{v}_next") for v in state_vars}
        r_next = _build_ranking_expr(coefficients, next_var_lookup)
        r_next_rebuilt = _rebuild_in_solver(r_next, s)
        r_curr_rebuilt = _rebuild_in_solver(ranking, s)

        diff = App(Op.SUB, [r_curr_rebuilt, r_next_rebuilt], INT)
        s.add(App(Op.LE, [diff, IntConst(0)], BOOL))

        decreasing = s.check() == SMTResult.UNSAT

    return bounded, decreasing


def _oct_constraint_to_smt(c, solver):
    """Convert OctConstraint to SMT assertion using C037 API."""
    bound_val = int(c.bound)
    bound = IntConst(bound_val)

    try:
        if c.var2 is None:
            # Unary: coeff1 * var1 <= bound
            v1 = solver.Int(c.var1)
            if c.coeff1 == 1:
                return App(Op.LE, [v1, bound], BOOL)
            elif c.coeff1 == -1:
                neg = App(Op.SUB, [IntConst(0), v1], INT)
                return App(Op.LE, [neg, bound], BOOL)
        else:
            # Binary: coeff1 * var1 + coeff2 * var2 <= bound
            v1 = solver.Int(c.var1)
            v2 = solver.Int(c.var2)
            t1 = v1 if c.coeff1 == 1 else App(Op.SUB, [IntConst(0), v1], INT)
            t2 = v2 if c.coeff2 == 1 else App(Op.SUB, [IntConst(0), v2], INT)
            lhs = App(Op.ADD, [t1, t2], INT)
            return App(Op.LE, [lhs, bound], BOOL)
    except Exception:
        return None

    return None


# ---------------------------------------------------------------------------
# Relational Ranking Function Discovery
# ---------------------------------------------------------------------------

@dataclass
class RelationalRankingFunction:
    """A ranking function derived from octagon analysis.

    May involve relational expressions like x - y, x + y.
    """
    expression: str
    coefficients: dict
    kind: str = "relational"
    octagon_invariant: str = ""  # The octagon constraint that inspired it


def find_relational_ranking(source, loop_index=0):
    """Find a relational ranking function using octagon analysis.

    This discovers ranking functions involving difference/sum of variables
    that V025's standard approach might miss.

    Returns RelationalRankingFunction or None.
    """
    loop_info = extract_loop_info(source, loop_index)

    # Generate octagon-guided candidates
    candidates = generate_octagon_candidates(source, loop_index)

    # Also get standard candidates as fallback
    std_candidates = generate_candidates(loop_info)

    # Try octagon candidates first (they're the novel ones)
    for coeffs in candidates:
        bounded, decreasing = verify_ranking_function(loop_info, coeffs)
        if bounded and decreasing:
            return RelationalRankingFunction(
                expression=_coefficients_to_str(coeffs),
                coefficients=coeffs,
                kind="relational",
            )

    # Try standard candidates
    for coeffs in std_candidates:
        if _coefficients_to_str(coeffs) not in {_coefficients_to_str(c) for c in candidates}:
            bounded, decreasing = verify_ranking_function(loop_info, coeffs)
            if bounded and decreasing:
                return RelationalRankingFunction(
                    expression=_coefficients_to_str(coeffs),
                    coefficients=coeffs,
                    kind="linear",
                )

    return None


def find_octagon_strengthened_ranking(source, loop_index=0):
    """Find ranking function with octagon-strengthened verification.

    Uses octagon invariants to strengthen the SMT proof, potentially
    proving ranking functions that fail under standard verification.

    Returns RelationalRankingFunction or None.
    """
    loop_info = extract_loop_info(source, loop_index)
    oct_info = extract_octagon_invariants(source, loop_index)

    if oct_info is None:
        return None

    # Generate all candidates
    candidates = generate_octagon_candidates(source, loop_index)
    std_candidates = generate_candidates(loop_info)

    all_candidates = list(candidates)
    seen = {_coefficients_to_str(c) for c in candidates}
    for c in std_candidates:
        key = _coefficients_to_str(c)
        if key not in seen:
            seen.add(key)
            all_candidates.append(c)

    for coeffs in all_candidates:
        bounded, decreasing = verify_ranking_with_octagon(
            loop_info, coeffs, oct_info
        )
        if bounded and decreasing:
            return RelationalRankingFunction(
                expression=_coefficients_to_str(coeffs),
                coefficients=coeffs,
                kind="octagon-strengthened",
            )

    return None


# ---------------------------------------------------------------------------
# Relational Lexicographic Ranking
# ---------------------------------------------------------------------------

def find_relational_lex_ranking(source, loop_index=0, max_components=2):
    """Find lexicographic ranking using relational candidates.

    Tries pairs (R1, R2) where R1 and/or R2 are relational (octagon-derived).

    Returns LexRankingFunction or None.
    """
    loop_info = extract_loop_info(source, loop_index)

    oct_candidates = generate_octagon_candidates(source, loop_index)
    std_candidates = generate_candidates(loop_info)

    all_candidates = list(oct_candidates)
    seen = {_coefficients_to_str(c) for c in oct_candidates}
    for c in std_candidates:
        key = _coefficients_to_str(c)
        if key not in seen:
            seen.add(key)
            all_candidates.append(c)

    # Try all pairs
    for i, c1 in enumerate(all_candidates):
        for c2 in all_candidates:
            if c1 is c2:
                continue
            if _verify_lex_pair(loop_info, c1, c2):
                rf1 = RankingFunction(
                    expression=_coefficients_to_str(c1),
                    coefficients=c1,
                )
                rf2 = RankingFunction(
                    expression=_coefficients_to_str(c2),
                    coefficients=c2,
                )
                return LexRankingFunction(components=[rf1, rf2])

    return None


# ---------------------------------------------------------------------------
# Comprehensive Octagon Termination Analysis
# ---------------------------------------------------------------------------

@dataclass
class OctTerminationResult:
    """Result of octagon-based termination analysis."""
    result: TermResult
    ranking_function: object = None  # RankingFunction, LexRankingFunction, or RelationalRankingFunction
    strategy: str = ""  # Which strategy found the proof
    octagon_invariants: dict = field(default_factory=dict)
    loops_analyzed: int = 0
    loops_proved: int = 0
    loop_results: list = field(default_factory=list)
    message: str = ""


@dataclass
class OctLoopResult:
    """Per-loop result for octagon termination."""
    loop_index: int
    result: TermResult
    ranking_function: object = None
    strategy: str = ""
    candidates_tried: int = 0
    message: str = ""


def prove_termination_with_octagon(source, loop_index=0):
    """Comprehensive termination proof using octagon analysis.

    Strategy (in order):
    1. Standard V025 (fast, handles simple cases)
    2. Octagon-guided relational ranking (novel candidates from octagon AI)
    3. Octagon-strengthened verification (use invariants to strengthen SMT)
    4. Relational lexicographic ranking (pairs of relational candidates)

    Returns OctLoopResult.
    """
    candidates_tried = 0

    # Strategy 1: Standard V025
    std_result = prove_termination(source, loop_index)
    candidates_tried += std_result.candidates_tried
    if std_result.result == TermResult.TERMINATES:
        return OctLoopResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=std_result.ranking_function,
            strategy="standard",
            candidates_tried=candidates_tried,
            message=f"Standard ranking: {std_result.ranking_function.expression}",
        )

    # Strategy 2: Relational ranking from octagon candidates
    rel_rf = find_relational_ranking(source, loop_index)
    if rel_rf is not None:
        return OctLoopResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=rel_rf,
            strategy="relational",
            candidates_tried=candidates_tried,
            message=f"Relational ranking: {rel_rf.expression}",
        )

    # Strategy 3: Octagon-strengthened verification
    str_rf = find_octagon_strengthened_ranking(source, loop_index)
    if str_rf is not None:
        return OctLoopResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=str_rf,
            strategy="octagon-strengthened",
            candidates_tried=candidates_tried,
            message=f"Octagon-strengthened ranking: {str_rf.expression}",
        )

    # Strategy 4: Relational lexicographic
    lex_rf = find_relational_lex_ranking(source, loop_index)
    if lex_rf is not None:
        return OctLoopResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=lex_rf,
            strategy="relational-lexicographic",
            candidates_tried=candidates_tried,
            message=f"Relational lex ranking: {lex_rf.expression}",
        )

    return OctLoopResult(
        loop_index=loop_index,
        result=TermResult.UNKNOWN,
        strategy="exhausted",
        candidates_tried=candidates_tried,
        message="All octagon strategies exhausted",
    )


def analyze_termination_with_octagon(source):
    """Analyze all loops in a program using octagon-based termination.

    Returns OctTerminationResult.
    """
    ast = parse(source)
    all_loops = _find_all_loops(ast.stmts)

    if not all_loops:
        return OctTerminationResult(
            result=TermResult.TERMINATES,
            loops_analyzed=0,
            loops_proved=0,
            message="No loops found",
        )

    loop_results = []
    loops_proved = 0

    for i in range(len(all_loops)):
        try:
            lr = prove_termination_with_octagon(source, i)
        except Exception as e:
            lr = OctLoopResult(
                loop_index=i,
                result=TermResult.UNKNOWN,
                message=f"Error: {e}",
            )
        loop_results.append(lr)
        if lr.result == TermResult.TERMINATES:
            loops_proved += 1

    overall = TermResult.TERMINATES if loops_proved == len(all_loops) else TermResult.UNKNOWN

    return OctTerminationResult(
        result=overall,
        loops_analyzed=len(all_loops),
        loops_proved=loops_proved,
        loop_results=loop_results,
        message=f"Proved {loops_proved}/{len(all_loops)} loops",
    )


# ---------------------------------------------------------------------------
# Octagon Invariant Verification for Termination
# ---------------------------------------------------------------------------

def verify_termination_invariant(source, ranking_coeffs, loop_index=0):
    """Verify that octagon invariants support the given ranking function.

    Returns dict with:
      - 'standard': (bounded, decreasing) from V025
      - 'octagon_strengthened': (bounded, decreasing) with octagon help
      - 'invariants_used': list of octagon constraints used
    """
    loop_info = extract_loop_info(source, loop_index)
    oct_info = extract_octagon_invariants(source, loop_index)

    std_b, std_d = verify_ranking_function(loop_info, ranking_coeffs)

    if oct_info:
        oct_b, oct_d = verify_ranking_with_octagon(loop_info, ranking_coeffs, oct_info)
        constraints = oct_info.get('constraints', [])
    else:
        oct_b, oct_d = std_b, std_d
        constraints = []

    return {
        'standard': (std_b, std_d),
        'octagon_strengthened': (oct_b, oct_d),
        'invariants_used': [str(c) for c in constraints],
    }


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------

def compare_strategies(source, loop_index=0):
    """Compare standard vs octagon-based termination strategies.

    Returns dict with results from each strategy.
    """
    results = {}

    # Standard V025
    std = prove_termination(source, loop_index)
    results['standard'] = {
        'result': std.result.value,
        'ranking': std.ranking_function.expression if std.ranking_function else None,
        'candidates': std.candidates_tried,
    }

    # Relational
    rel = find_relational_ranking(source, loop_index)
    results['relational'] = {
        'result': 'terminates' if rel else 'unknown',
        'ranking': rel.expression if rel else None,
    }

    # Octagon-strengthened
    ostr = find_octagon_strengthened_ranking(source, loop_index)
    results['octagon_strengthened'] = {
        'result': 'terminates' if ostr else 'unknown',
        'ranking': ostr.expression if ostr else None,
    }

    # Full octagon analysis
    full = prove_termination_with_octagon(source, loop_index)
    results['full_octagon'] = {
        'result': full.result.value,
        'ranking': full.ranking_function.expression if full.ranking_function else None,
        'strategy': full.strategy,
    }

    return results


# ---------------------------------------------------------------------------
# Convenience: check specific relational ranking
# ---------------------------------------------------------------------------

def check_relational_ranking(source, var1, coeff1, var2, coeff2, const=0, loop_index=0):
    """Check if coeff1*var1 + coeff2*var2 + const is a valid ranking function.

    Example: check_relational_ranking(src, 'x', 1, 'y', -1) checks R = x - y.

    Returns (bounded, decreasing, octagon_bounded, octagon_decreasing).
    """
    coeffs = {var1: coeff1, var2: coeff2, '_const': const}
    loop_info = extract_loop_info(source, loop_index)
    oct_info = extract_octagon_invariants(source, loop_index)

    std_b, std_d = verify_ranking_function(loop_info, coeffs)

    if oct_info:
        oct_b, oct_d = verify_ranking_with_octagon(loop_info, coeffs, oct_info)
    else:
        oct_b, oct_d = std_b, std_d

    return std_b, std_d, oct_b, oct_d
