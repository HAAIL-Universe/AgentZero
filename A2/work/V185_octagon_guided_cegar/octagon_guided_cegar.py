"""
V185: Octagon-Guided CEGAR
===========================
Composes: V173 (octagon abstract domain) + V010 (predicate abstraction + CEGAR)

Uses octagon abstract interpretation as a pre-pass to generate high-quality
relational predicates for CEGAR verification:

1. Octagon pre-analysis: run octagon interpreter on program AST to discover
   relational invariants (x-y <= c, x+y <= c, x <= c)
2. Predicate seeding: convert octagon constraints to SMT predicates
3. Octagon-guided refinement: when standard CEGAR refinement stalls, use
   octagon analysis to suggest relational predicates at the infeasible step
4. Comparison framework: benchmark standard vs octagon-guided CEGAR

Key insight: octagon discovers relational predicates (difference/sum bounds)
that standard auto_predicates_from_ts misses, enabling faster convergence
on systems with relational invariants.
"""

import sys, os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from fractions import Fraction

# Path setup
_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V010_predicate_abstraction_cegar'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))

from smt_solver import Var, IntConst, BoolConst, App, Op, Sort, SortKind, Term
from octagon import Octagon, OctConstraint, OctExpr, OctagonInterpreter, OctAnalysisResult
from pred_abs_cegar import (
    Predicate, CEGARVerdict, CEGARStats, CEGARResult, ConcreteTS,
    cegar_check, verify_with_cegar, auto_predicates_from_ts,
    extract_loop_ts, _parse_property, _ast_to_smt, _and, _or, _not, _eq,
    cartesian_abstraction, check_counterexample_feasibility, refine_predicates,
    _extract_atomic_predicates, _smt_check
)
from pdr import TransitionSystem, check_ts, PDRResult, PDROutput

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)

INF = Fraction(10**9)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OctCEGARStats:
    """Extended stats for octagon-guided CEGAR."""
    oct_constraints_found: int = 0
    oct_predicates_generated: int = 0
    oct_predicates_useful: int = 0  # actually used by CEGAR
    cegar_stats: CEGARStats = field(default_factory=CEGARStats)
    oct_refinements: int = 0  # times octagon suggested refinement preds


@dataclass
class OctCEGARResult:
    """Result of octagon-guided CEGAR verification."""
    verdict: CEGARVerdict
    invariant: Optional[list] = None
    counterexample: Optional[list] = None
    predicates: list = field(default_factory=list)
    oct_predicates: list = field(default_factory=list)
    stats: OctCEGARStats = field(default_factory=OctCEGARStats)


@dataclass
class ComparisonResult:
    """Comparison between standard and octagon-guided CEGAR."""
    standard: CEGARResult = None
    octagon_guided: OctCEGARResult = None
    speedup_iterations: float = 0.0  # ratio of iterations saved
    extra_predicates: int = 0  # how many octagon added
    both_agree: bool = True


# ---------------------------------------------------------------------------
# AST conversion: C10 AST -> octagon tuple AST
# ---------------------------------------------------------------------------

def _ast_to_oct_program(ast):
    """Convert C10 parsed AST (Program) to octagon interpreter tuple format."""
    stmts = []
    for s in ast.stmts:
        converted = _convert_stmt(s)
        if converted is not None:
            stmts.append(converted)
    if len(stmts) == 0:
        return ('skip',)
    if len(stmts) == 1:
        return stmts[0]
    return ('seq',) + tuple(stmts)


def _convert_stmt(stmt):
    """Convert a single C10 AST statement to octagon tuple."""
    stype = type(stmt).__name__
    if stype == 'LetDecl':
        return ('assign', stmt.name, _convert_expr(stmt.value))
    if stype == 'Assign':
        return ('assign', stmt.name, _convert_expr(stmt.value))
    if stype == 'IfStmt':
        cond = _convert_cond(stmt.cond)
        then_body = _block_to_single(stmt.then_body)
        else_body = _block_to_single(stmt.else_body) if stmt.else_body else ('skip',)
        return ('if', cond, then_body, else_body)
    if stype == 'WhileStmt':
        cond = _convert_cond(stmt.cond)
        body = _block_to_single(stmt.body)
        return ('while', cond, body)
    if stype == 'Block':
        inner = _convert_block(stmt)
        if len(inner) == 1:
            return inner[0]
        return ('seq',) + tuple(inner)
    return ('skip',)


def _convert_block(block):
    """Convert a block (with .stmts) to list of octagon tuples."""
    if block is None:
        return [('skip',)]
    stmts_attr = block.stmts if hasattr(block, 'stmts') else (block if isinstance(block, list) else [block])
    result = []
    for s in stmts_attr:
        converted = _convert_stmt(s)
        if converted is not None:
            result.append(converted)
    return result if result else [('skip',)]


def _block_to_single(block):
    """Convert a block to a single octagon statement tuple.

    The octagon interpreter expects if/while bodies to be a single
    statement (which can be a seq tuple), not a list.
    """
    stmts = _convert_block(block)
    if len(stmts) == 1:
        return stmts[0]
    return ('seq',) + tuple(stmts)


def _convert_expr(expr):
    """Convert C10 expression to octagon expression tuple."""
    etype = type(expr).__name__
    if etype == 'IntLit':
        return ('const', expr.value)
    if etype in ('ASTVar', 'Var'):
        return ('var', expr.name)
    if etype == 'BinOp':
        left = _convert_expr(expr.left)
        right = _convert_expr(expr.right)
        op_map = {'+': 'add', '-': 'sub', '*': 'mul'}
        op = op_map.get(expr.op)
        if op:
            return (op, left, right)
        # Comparison operators are conditions, not expressions
        return ('const', 0)
    if etype == 'UnaryOp' and expr.op == '-':
        return ('neg', _convert_expr(expr.operand))
    return ('const', 0)


def _convert_cond(expr):
    """Convert C10 condition expression to octagon condition tuple."""
    etype = type(expr).__name__
    if etype == 'BinOp':
        if expr.op in ('<', '<=', '>', '>=', '==', '!='):
            left = _convert_expr(expr.left)
            right = _convert_expr(expr.right)
            op_map = {'<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne'}
            return (op_map[expr.op], left, right)
        if expr.op == '&&':
            return ('and', _convert_cond(expr.left), _convert_cond(expr.right))
        if expr.op == '||':
            return ('or', _convert_cond(expr.left), _convert_cond(expr.right))
    if etype == 'UnaryOp' and expr.op == '!':
        return ('not', _convert_cond(expr.operand))
    if etype == 'IntLit':
        return ('le', ('const', 1), ('const', 0)) if expr.value == 0 else ('le', ('const', 0), ('const', 1))
    return ('le', ('const', 0), ('const', 1))  # default: true


# ---------------------------------------------------------------------------
# Octagon constraint -> SMT predicate conversion
# ---------------------------------------------------------------------------

def _oct_constraint_to_smt(c, cts):
    """Convert an OctConstraint to an SMT Term suitable for CEGAR predicates.

    OctConstraint has: var1, coeff1, var2, coeff2, bound
    Represents: coeff1*var1 + coeff2*var2 <= bound
    """
    bound_val = int(c.bound) if c.bound == int(c.bound) else int(c.bound)

    if c.var2 is None:
        # Unary: coeff1 * var1 <= bound
        v1 = cts.var(c.var1)
        if c.coeff1 == 1:
            # var1 <= bound
            return App(Op.LE, [v1, IntConst(bound_val)], BOOL)
        else:
            # -var1 <= bound  =>  var1 >= -bound
            return App(Op.GE, [v1, IntConst(-bound_val)], BOOL)
    else:
        # Binary: coeff1*var1 + coeff2*var2 <= bound
        v1 = cts.var(c.var1)
        v2 = cts.var(c.var2)
        if c.coeff1 == 1 and c.coeff2 == -1:
            # var1 - var2 <= bound
            diff = App(Op.SUB, [v1, v2], INT)
            return App(Op.LE, [diff, IntConst(bound_val)], BOOL)
        elif c.coeff1 == -1 and c.coeff2 == 1:
            # -var1 + var2 <= bound  =>  var2 - var1 <= bound
            diff = App(Op.SUB, [v2, v1], INT)
            return App(Op.LE, [diff, IntConst(bound_val)], BOOL)
        elif c.coeff1 == 1 and c.coeff2 == 1:
            # var1 + var2 <= bound
            s = App(Op.ADD, [v1, v2], INT)
            return App(Op.LE, [s, IntConst(bound_val)], BOOL)
        elif c.coeff1 == -1 and c.coeff2 == -1:
            # -var1 - var2 <= bound  =>  var1 + var2 >= -bound
            s = App(Op.ADD, [v1, v2], INT)
            return App(Op.GE, [s, IntConst(-bound_val)], BOOL)
        else:
            # Shouldn't happen for octagon, but handle gracefully
            return None


def _oct_constraint_name(c):
    """Generate a readable name for an octagon-derived predicate."""
    if c.var2 is None:
        if c.coeff1 == 1:
            return f"oct_{c.var1}_le_{int(c.bound)}"
        else:
            return f"oct_{c.var1}_ge_{int(-c.bound)}"
    else:
        bound_int = int(c.bound)
        if c.coeff1 == 1 and c.coeff2 == -1:
            return f"oct_{c.var1}_minus_{c.var2}_le_{bound_int}"
        elif c.coeff1 == -1 and c.coeff2 == 1:
            return f"oct_{c.var2}_minus_{c.var1}_le_{bound_int}"
        elif c.coeff1 == 1 and c.coeff2 == 1:
            return f"oct_{c.var1}_plus_{c.var2}_le_{bound_int}"
        elif c.coeff1 == -1 and c.coeff2 == -1:
            return f"oct_{c.var1}_plus_{c.var2}_ge_{int(-bound_int)}"
        return f"oct_constraint"


# ---------------------------------------------------------------------------
# Octagon pre-analysis
# ---------------------------------------------------------------------------

def octagon_pre_analyze(source):
    """Run octagon abstract interpretation on C10 source code.

    Returns:
        (OctAnalysisResult, list of OctConstraint) -- the analysis result
        and extracted constraints from the final state.
    """
    from stack_vm import lex, Parser
    tokens = lex(source)
    ast = Parser(tokens).parse()

    oct_ast = _ast_to_oct_program(ast)
    interp = OctagonInterpreter(max_iterations=50, widen_delay=2)
    result = interp.analyze(oct_ast)

    constraints = []
    if result.final_state and not result.final_state.is_bot():
        constraints = result.final_state.extract_constraints()

    return result, constraints


def octagon_predicates_from_source(source, cts):
    """Generate SMT predicates from octagon analysis of source code.

    Args:
        source: C10 source code
        cts: ConcreteTS (needed for variable references)

    Returns:
        list of Predicate objects derived from octagon analysis
    """
    result, constraints = octagon_pre_analyze(source)
    return octagon_predicates_from_constraints(constraints, cts)


def octagon_predicates_from_constraints(constraints, cts):
    """Convert octagon constraints to CEGAR predicates.

    Filters out trivial constraints (huge bounds) and ensures all
    referenced variables exist in the ConcreteTS.
    """
    predicates = []
    seen = set()
    cts_vars = set(cts.all_vars())

    for c in constraints:
        # Skip constraints with huge bounds (not informative)
        if abs(c.bound) >= INF:
            continue

        # Skip constraints referencing variables not in the system
        if c.var1 not in cts_vars:
            continue
        if c.var2 is not None and c.var2 not in cts_vars:
            continue

        smt_formula = _oct_constraint_to_smt(c, cts)
        if smt_formula is None:
            continue

        name = _oct_constraint_name(c)
        fstr = str(smt_formula)
        if fstr not in seen:
            seen.add(fstr)
            predicates.append(Predicate(name, smt_formula))

    return predicates


# ---------------------------------------------------------------------------
# Octagon-guided CEGAR loop
# ---------------------------------------------------------------------------

def octagon_guided_cegar(concrete_ts, source=None, initial_predicates=None,
                         max_iterations=10, max_pdr_frames=50,
                         use_oct_refinement=True):
    """Run CEGAR with octagon-derived initial predicates.

    Pipeline:
    1. Run octagon pre-analysis on source (if provided)
    2. Generate relational predicates from octagon constraints
    3. Combine with standard auto-predicates
    4. Run CEGAR with enriched predicate set
    5. On refinement failure, try octagon-guided refinement

    Args:
        concrete_ts: ConcreteTS
        source: Optional C10 source code for octagon analysis
        initial_predicates: Optional user-supplied predicates
        max_iterations: Max CEGAR iterations
        max_pdr_frames: Max PDR frames per iteration
        use_oct_refinement: Whether to use octagon for refinement guidance

    Returns:
        OctCEGARResult
    """
    stats = OctCEGARStats()

    # Step 1: Gather predicates from multiple sources
    all_predicates = []
    seen_formulas = set()

    def add_preds(preds, source_name=""):
        for p in preds:
            fstr = str(p.formula)
            if fstr not in seen_formulas:
                seen_formulas.add(fstr)
                all_predicates.append(p)

    # Standard auto-predicates
    auto_preds = auto_predicates_from_ts(concrete_ts)
    add_preds(auto_preds, "auto")

    # User-supplied predicates
    if initial_predicates:
        add_preds(initial_predicates, "user")

    # Octagon-derived predicates
    oct_preds = []
    if source is not None:
        oct_preds = octagon_predicates_from_source(source, concrete_ts)
        stats.oct_constraints_found = len(oct_preds)
        before = len(all_predicates)
        add_preds(oct_preds, "octagon")
        stats.oct_predicates_generated = len(all_predicates) - before

    # Step 2: Run CEGAR
    if use_oct_refinement:
        result = _octagon_cegar_loop(concrete_ts, all_predicates, source,
                                     max_iterations, max_pdr_frames, stats)
    else:
        cegar_result = cegar_check(concrete_ts, all_predicates,
                                   max_iterations, max_pdr_frames)
        result = OctCEGARResult(
            verdict=cegar_result.verdict,
            invariant=cegar_result.invariant,
            counterexample=cegar_result.counterexample,
            predicates=cegar_result.predicates,
            oct_predicates=oct_preds,
            stats=stats
        )
        stats.cegar_stats = cegar_result.stats

    return result


def _octagon_cegar_loop(concrete_ts, predicates, source,
                        max_iterations, max_pdr_frames, stats):
    """CEGAR loop with octagon-guided refinement fallback.

    When standard WP-based refinement fails to produce new predicates,
    attempts to generate relational predicates from octagon analysis
    of the transition relation.
    """
    from pdr import check_ts, PDRResult

    predicates = list(predicates)
    oct_preds_used = []

    for iteration in range(max_iterations):
        stats.cegar_stats.iterations = iteration + 1

        # Abstract
        abs_ts = cartesian_abstraction(concrete_ts, predicates)

        # Model check
        stats.cegar_stats.pdr_calls += 1
        pdr_result = check_ts(abs_ts, max_frames=max_pdr_frames)

        if pdr_result.result == PDRResult.SAFE:
            # Extract concrete invariant
            from pred_abs_cegar import _concretize_invariant
            invariant = _concretize_invariant(pdr_result.invariant, predicates)
            stats.cegar_stats.predicates_final = len(predicates)
            return OctCEGARResult(
                verdict=CEGARVerdict.SAFE,
                invariant=invariant,
                predicates=predicates,
                oct_predicates=oct_preds_used,
                stats=stats
            )

        if pdr_result.result == PDRResult.UNKNOWN:
            stats.cegar_stats.predicates_final = len(predicates)
            return OctCEGARResult(
                verdict=CEGARVerdict.UNKNOWN,
                predicates=predicates,
                oct_predicates=oct_preds_used,
                stats=stats
            )

        # UNSAFE -- check feasibility
        abstract_trace = pdr_result.counterexample.trace

        is_feasible, infeasible_step, concrete_trace = check_counterexample_feasibility(
            abstract_trace, concrete_ts, predicates
        )

        if is_feasible:
            stats.cegar_stats.predicates_final = len(predicates)
            return OctCEGARResult(
                verdict=CEGARVerdict.UNSAFE,
                counterexample=concrete_trace,
                predicates=predicates,
                oct_predicates=oct_preds_used,
                stats=stats
            )

        # Spurious -- try standard refinement first
        stats.cegar_stats.spurious_traces += 1
        new_preds = refine_predicates(predicates, concrete_ts, infeasible_step, abstract_trace)

        if new_preds:
            predicates.extend(new_preds)
            continue

        # Standard refinement failed -- try octagon-guided refinement
        oct_new = _octagon_refine(concrete_ts, predicates, source)
        if oct_new:
            stats.oct_refinements += 1
            predicates.extend(oct_new)
            oct_preds_used.extend(oct_new)
            continue

        # No refinement possible
        stats.cegar_stats.predicates_final = len(predicates)
        return OctCEGARResult(
            verdict=CEGARVerdict.UNKNOWN,
            predicates=predicates,
            oct_predicates=oct_preds_used,
            stats=stats
        )

    stats.cegar_stats.predicates_final = len(predicates)
    return OctCEGARResult(
        verdict=CEGARVerdict.UNKNOWN,
        predicates=predicates,
        oct_predicates=oct_preds_used,
        stats=stats
    )


def _octagon_refine(concrete_ts, existing_predicates, source):
    """Use octagon analysis to suggest relational predicates not yet in the set.

    Strategy: run octagon analysis, extract all constraints, filter to those
    not already represented, and return as new predicates.
    """
    if source is None:
        return []

    oct_preds = octagon_predicates_from_source(source, concrete_ts)
    existing_formulas = {str(p.formula) for p in existing_predicates}

    new_preds = []
    for p in oct_preds:
        if str(p.formula) not in existing_formulas:
            new_preds.append(p)

    return new_preds


# ---------------------------------------------------------------------------
# Source-level verification
# ---------------------------------------------------------------------------

def verify_loop_with_octagon_cegar(source, property_str, predicates=None, **kwargs):
    """Verify a loop property using octagon-guided CEGAR.

    Args:
        source: C10 source with a while loop
        property_str: Property like "x >= 0" or "x - y <= 5"
        predicates: Optional initial predicates
        **kwargs: passed to octagon_guided_cegar

    Returns: OctCEGARResult
    """
    cts = extract_loop_ts(source)
    prop = _parse_property(property_str, cts)
    cts.prop_formula = prop

    return octagon_guided_cegar(cts, source=source, initial_predicates=predicates, **kwargs)


def verify_ts_with_octagon_cegar(concrete_ts, source=None, predicates=None, **kwargs):
    """Verify a transition system using octagon-guided CEGAR.

    Convenience wrapper that accepts a ConcreteTS directly.
    """
    return octagon_guided_cegar(concrete_ts, source=source,
                                initial_predicates=predicates, **kwargs)


# ---------------------------------------------------------------------------
# Comparison framework
# ---------------------------------------------------------------------------

def compare_cegar_approaches(concrete_ts, source=None, predicates=None,
                             max_iterations=10, max_pdr_frames=50):
    """Compare standard CEGAR vs octagon-guided CEGAR.

    Returns a ComparisonResult showing iterations, predicates, and verdicts.
    """
    # Standard CEGAR
    std_preds = list(predicates) if predicates else None
    std_result = verify_with_cegar(concrete_ts, std_preds,
                                   max_iterations=max_iterations,
                                   max_pdr_frames=max_pdr_frames)

    # Octagon-guided CEGAR
    oct_result = octagon_guided_cegar(concrete_ts, source=source,
                                      initial_predicates=predicates,
                                      max_iterations=max_iterations,
                                      max_pdr_frames=max_pdr_frames)

    # Compare
    std_iters = std_result.stats.iterations
    oct_iters = oct_result.stats.cegar_stats.iterations

    speedup = (std_iters / oct_iters) if oct_iters > 0 else 0.0
    extra = oct_result.stats.oct_predicates_generated

    both_agree = (std_result.verdict == oct_result.verdict)

    return ComparisonResult(
        standard=std_result,
        octagon_guided=oct_result,
        speedup_iterations=speedup,
        extra_predicates=extra,
        both_agree=both_agree
    )


def compare_loop(source, property_str, predicates=None, **kwargs):
    """Compare CEGAR approaches on a loop + property.

    Convenience wrapper for compare_cegar_approaches.
    """
    cts = extract_loop_ts(source)
    prop = _parse_property(property_str, cts)
    cts.prop_formula = prop
    return compare_cegar_approaches(cts, source=source, predicates=predicates, **kwargs)


# ---------------------------------------------------------------------------
# Predicate analysis utilities
# ---------------------------------------------------------------------------

def classify_predicates(predicates):
    """Classify predicates by type: unary, difference, sum.

    Returns dict with keys 'unary', 'difference', 'sum', 'other'.
    """
    result = {'unary': [], 'difference': [], 'sum': [], 'other': []}
    for p in predicates:
        name = p.name
        if 'minus' in name:
            result['difference'].append(p)
        elif 'plus' in name:
            result['sum'].append(p)
        elif name.startswith('oct_') and ('le_' in name or 'ge_' in name):
            result['unary'].append(p)
        else:
            result['other'].append(p)
    return result


def predicate_strength_analysis(predicates, concrete_ts):
    """Analyze how constraining each predicate is.

    For each predicate, checks whether it's satisfiable and
    whether its negation is satisfiable (tautology vs contradiction check).

    Returns list of (predicate, is_tautology, is_contradiction).
    """
    results = []
    for p in predicates:
        # Check satisfiability of predicate
        from smt_solver import SMTResult
        pos_result, _ = _smt_check(p.formula)
        pos_sat = (pos_result == SMTResult.SAT)
        # Check satisfiability of negation
        neg_result, _ = _smt_check(_not(p.formula))
        neg_sat = (neg_result == SMTResult.SAT)

        is_tautology = (pos_sat and not neg_sat)
        is_contradiction = (not pos_sat and neg_sat)

        results.append((p, is_tautology, is_contradiction))
    return results


def octagon_invariant_candidates(source, cts):
    """Generate candidate invariant formulas from octagon analysis.

    These are stronger than individual predicates -- they represent
    the full octagon invariant at the loop fixpoint.

    Returns list of SMT formulas representing the octagon invariant.
    """
    result, constraints = octagon_pre_analyze(source)
    candidates = []
    cts_vars = set(cts.all_vars())

    for c in constraints:
        if abs(c.bound) >= INF:
            continue
        if c.var1 not in cts_vars:
            continue
        if c.var2 is not None and c.var2 not in cts_vars:
            continue

        smt = _oct_constraint_to_smt(c, cts)
        if smt is not None:
            candidates.append(smt)

    return candidates


def verify_octagon_invariant(source, property_str):
    """Quick check: does octagon analysis alone prove the property?

    Returns (proved, oct_result, constraints) where proved is True if
    octagon invariant implies the property.
    """
    from stack_vm import lex, Parser

    cts = extract_loop_ts(source)
    prop = _parse_property(property_str, cts)
    cts.prop_formula = prop

    result, constraints = octagon_pre_analyze(source)

    # Check if octagon constraints imply the property
    # Build: AND(all_constraints) => property
    # Equivalent to: NOT(AND(all_constraints) AND NOT(property)) is UNSAT
    if not constraints:
        return False, result, constraints

    smt_constraints = []
    cts_vars = set(cts.all_vars())
    for c in constraints:
        if abs(c.bound) >= INF:
            continue
        if c.var1 not in cts_vars:
            continue
        if c.var2 is not None and c.var2 not in cts_vars:
            continue
        smt = _oct_constraint_to_smt(c, cts)
        if smt is not None:
            smt_constraints.append(smt)

    if not smt_constraints:
        return False, result, constraints

    # Check: constraints AND NOT(property) is UNSAT => proved
    conj = _and(*smt_constraints, _not(prop))
    from smt_solver import SMTResult
    smt_result, _ = _smt_check(conj)

    return (smt_result == SMTResult.UNSAT), result, constraints


# ---------------------------------------------------------------------------
# Direct transition system construction with octagon hints
# ---------------------------------------------------------------------------

def build_ts_with_octagon_hints(int_vars, init_map, trans_map, prop_str,
                                 oct_hints=None):
    """Build a ConcreteTS from simple specs with optional octagon hints.

    Args:
        int_vars: list of variable names
        init_map: {var: initial_value}
        trans_map: {var: SMT expression for next value}
        prop_str: property as SMT term or string
        oct_hints: optional list of OctConstraint to seed predicates

    Returns: (ConcreteTS, list of Predicate)
    """
    cts = ConcreteTS(int_vars=list(int_vars))

    # Init formula
    init_clauses = []
    for v in int_vars:
        if v in init_map:
            init_clauses.append(_eq(cts.var(v), IntConst(init_map[v])))
    cts.init_formula = _and(*init_clauses) if init_clauses else BoolConst(True)

    # Transition formula
    trans_clauses = []
    for v in int_vars:
        vp = cts.prime(v)
        if v in trans_map:
            trans_clauses.append(_eq(vp, trans_map[v]))
        else:
            trans_clauses.append(_eq(vp, cts.var(v)))  # frame
    cts.trans_formula = _and(*trans_clauses) if trans_clauses else BoolConst(True)

    # Property
    if isinstance(prop_str, str):
        cts.prop_formula = _parse_property(prop_str, cts)
    else:
        cts.prop_formula = prop_str

    # Generate predicates from octagon hints
    preds = []
    if oct_hints:
        preds = octagon_predicates_from_constraints(oct_hints, cts)

    return cts, preds
