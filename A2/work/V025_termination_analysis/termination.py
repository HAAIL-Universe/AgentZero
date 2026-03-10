"""
V025: Termination Analysis

Proves program termination by discovering ranking functions -- expressions
that strictly decrease on every loop iteration and are bounded below by 0.

Composes:
  - C010 Parser (AST)
  - C037 SMT Solver (verification of ranking conditions)
  - V005/V002 helpers (loop extraction, transition systems)
  - C039 Abstract Interpreter (bound estimation for candidates)

Approach:
  1. Extract loop structure: condition, body transition, state variables
  2. Generate candidate ranking functions (templates):
     - Single variable: R = x, R = -x, R = c - x
     - Linear combinations: R = c0 + c1*x1 + ... + cn*xn
     - Condition-derived: upper bound minus variable
  3. Verify via SMT:
     - Bounded: cond(s) => R(s) >= 0
     - Decreasing: cond(s) AND Trans(s,s') => R(s) - R(s') >= 1
  4. For lexicographic ranking: try tuples (R1, R2, ...) where R1 decreases
     or stays same while R2 decreases, etc.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

# Import C010 parser
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program,
    IntLit, BoolLit, Var as ASTVar,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt
)

# Import C037 SMT solver
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Op, BOOL, INT,
    Var as SMTVar, IntConst, BoolConst, App
)

# Import V002 TransitionSystem and helpers
sys.path.insert(0, os.path.join(_work, 'V002_pdr_ic3'))
from pdr import (
    TransitionSystem,
    _and, _or, _negate, _eq, _implies,
)

# Import C039 abstract interpreter
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, Interval, INTERVAL_TOP, INTERVAL_BOT,
    Sign, INF, NEG_INF, analyze as ai_analyze
)


# ============================================================
# Result Types
# ============================================================

class TermResult(Enum):
    TERMINATES = "terminates"
    UNKNOWN = "unknown"
    NONTERMINATING = "nonterminating"


@dataclass
class RankingFunction:
    """A ranking function that proves termination."""
    expression: str              # Human-readable expression
    coefficients: dict           # var_name -> coefficient, plus '_const' -> constant term
    kind: str = "linear"         # "linear", "lexicographic", "conditional"

    def evaluate_str(self):
        """String representation of the ranking function."""
        return self.expression


@dataclass
class LexRankingFunction:
    """Lexicographic ranking function (tuple of ranking functions)."""
    components: List[RankingFunction]
    kind: str = "lexicographic"

    @property
    def expression(self):
        return "(" + ", ".join(c.expression for c in self.components) + ")"


@dataclass
class TerminationResult:
    """Result of termination analysis."""
    result: TermResult
    ranking_function: Optional[object] = None   # RankingFunction or LexRankingFunction
    loops_analyzed: int = 0
    loops_proved: int = 0
    loop_results: list = field(default_factory=list)  # per-loop results
    message: str = ""


@dataclass
class LoopTermResult:
    """Termination result for a single loop."""
    loop_index: int
    result: TermResult
    ranking_function: Optional[object] = None
    candidates_tried: int = 0
    message: str = ""


# ============================================================
# AST Helpers (from V005, adapted)
# ============================================================

def parse(source):
    """Parse C010 source into AST."""
    tokens = lex(source)
    return Parser(tokens).parse()


def _collect_vars_in_expr(expr):
    """Collect all variable names referenced in an expression."""
    if isinstance(expr, ASTVar):
        return {expr.name}
    elif isinstance(expr, BinOp):
        return _collect_vars_in_expr(expr.left) | _collect_vars_in_expr(expr.right)
    elif isinstance(expr, UnaryOp):
        return _collect_vars_in_expr(expr.operand)
    elif isinstance(expr, CallExpr):
        result = set()
        for arg in expr.args:
            result |= _collect_vars_in_expr(arg)
        return result
    return set()


def _collect_assigned_vars(stmt):
    """Collect variables that are assigned in a statement."""
    if isinstance(stmt, (LetDecl, Assign)):
        return {stmt.name}
    elif isinstance(stmt, IfStmt):
        result = _collect_assigned_vars(stmt.then_body)
        if stmt.else_body:
            result |= _collect_assigned_vars(stmt.else_body)
        return result
    elif isinstance(stmt, WhileStmt):
        return _collect_assigned_vars(stmt.body)
    elif isinstance(stmt, Block):
        result = set()
        for s in stmt.stmts:
            result |= _collect_assigned_vars(s)
        return result
    return set()


def _expr_to_smt(expr, var_lookup):
    """Convert C010 AST expression to SMT term."""
    if isinstance(expr, IntLit):
        return IntConst(expr.value)
    elif isinstance(expr, BoolLit):
        return BoolConst(expr.value)
    elif isinstance(expr, ASTVar):
        if expr.name in var_lookup:
            return var_lookup[expr.name]
        return SMTVar(expr.name, INT)
    elif isinstance(expr, UnaryOp):
        operand = _expr_to_smt(expr.operand, var_lookup)
        if expr.op == '-':
            return App(Op.SUB, [IntConst(0), operand], INT)
        elif expr.op == 'not':
            return _negate(operand)
        return operand
    elif isinstance(expr, BinOp):
        left = _expr_to_smt(expr.left, var_lookup)
        right = _expr_to_smt(expr.right, var_lookup)
        op_map = {
            '+': (Op.ADD, INT), '-': (Op.SUB, INT),
            '*': (Op.MUL, INT),
            '<': (Op.LT, BOOL), '>': (Op.GT, BOOL),
            '<=': (Op.LE, BOOL), '>=': (Op.GE, BOOL),
            '==': (Op.EQ, BOOL), '!=': (Op.NEQ, BOOL),
            'and': (Op.AND, BOOL), 'or': (Op.OR, BOOL),
        }
        if expr.op in op_map:
            op, sort = op_map[expr.op]
            return App(op, [left, right], sort)
        return IntConst(0)
    return IntConst(0)


def _process_body_stmts(body, var_lookup, next_state, state_vars):
    """Process body statements to build next-state expressions."""
    if isinstance(body, Block):
        stmts = body.stmts
    elif isinstance(body, list):
        stmts = body
    else:
        stmts = [body]

    for stmt in stmts:
        if isinstance(stmt, (LetDecl, Assign)):
            if stmt.name in state_vars:
                expr_smt = _expr_to_smt(stmt.value, var_lookup)
                next_state[stmt.name] = expr_smt
                var_lookup[stmt.name] = expr_smt
        elif isinstance(stmt, IfStmt):
            cond_smt = _expr_to_smt(stmt.cond, var_lookup)
            saved_lookup = dict(var_lookup)
            saved_next = dict(next_state)

            then_lookup = dict(var_lookup)
            then_next = dict(next_state)
            _process_body_stmts(stmt.then_body, then_lookup, then_next, state_vars)

            else_lookup = dict(saved_lookup)
            else_next = dict(saved_next)
            if stmt.else_body:
                _process_body_stmts(stmt.else_body, else_lookup, else_next, state_vars)

            for name in state_vars:
                if str(then_next.get(name)) != str(else_next.get(name)):
                    ite = App(Op.ITE, [cond_smt, then_next[name], else_next[name]], INT)
                    next_state[name] = ite
                    var_lookup[name] = ite
                else:
                    next_state[name] = then_next[name]
                    var_lookup[name] = then_next[name]


# ============================================================
# Loop Extraction
# ============================================================

def _collect_pre_assignments(stmts, target_loop, pre_assignments, found):
    """Collect LetDecl assignments that appear before the target loop."""
    for stmt in stmts:
        if found[0]:
            return
        if stmt is target_loop:
            found[0] = True
            return
        if isinstance(stmt, LetDecl):
            if isinstance(stmt.value, IntLit):
                pre_assignments[stmt.name] = stmt.value.value
            elif isinstance(stmt.value, BoolLit):
                pre_assignments[stmt.name] = 1 if stmt.value.value else 0
            elif isinstance(stmt.value, UnaryOp) and stmt.value.op == '-' and isinstance(stmt.value.operand, IntLit):
                pre_assignments[stmt.name] = -stmt.value.operand.value
            else:
                pre_assignments[stmt.name] = None
        elif isinstance(stmt, WhileStmt):
            # Recurse into while body to find nested loops
            if isinstance(stmt.body, Block):
                _collect_pre_assignments(stmt.body.stmts, target_loop, pre_assignments, found)
        elif isinstance(stmt, IfStmt):
            if isinstance(stmt.then_body, Block):
                _collect_pre_assignments(stmt.then_body.stmts, target_loop, pre_assignments, found)
            if not found[0] and stmt.else_body and isinstance(stmt.else_body, Block):
                _collect_pre_assignments(stmt.else_body.stmts, target_loop, pre_assignments, found)
        elif isinstance(stmt, Block):
            _collect_pre_assignments(stmt.stmts, target_loop, pre_assignments, found)

def extract_loop_info(source, loop_index=0):
    """
    Extract loop information needed for termination analysis.

    Returns: dict with keys:
      - state_vars: list of variable names
      - pre_assignments: dict var_name -> init value (int or None)
      - cond_ast: the loop condition AST node
      - body_ast: the loop body AST node
      - cond_smt: condition as SMT formula (using current-state vars)
      - next_state: dict var_name -> SMT expression for next state
      - var_lookup: dict var_name -> current-state SMT Var
    """
    program = parse(source)

    # Find all loops (including nested) and collect pre-assignments along the way
    all_loops = _find_all_loops(program.stmts)
    if loop_index >= len(all_loops):
        raise ValueError(f"No while loop found at index {loop_index}")
    loop_stmt = all_loops[loop_index]

    # Collect pre-assignments: all LetDecl statements before the target loop
    # at any nesting level
    pre_assignments = {}
    _collect_pre_assignments(program.stmts, loop_stmt, pre_assignments, found=[False])

    body_vars = _collect_assigned_vars(loop_stmt.body)
    cond_vars = _collect_vars_in_expr(loop_stmt.cond)
    state_vars = sorted(body_vars | cond_vars)

    # Build current-state var lookup
    var_lookup = {}
    for name in state_vars:
        var_lookup[name] = SMTVar(name, INT)

    # Build condition SMT
    cond_smt = _expr_to_smt(loop_stmt.cond, var_lookup)

    # Build next-state expressions
    next_state = {name: SMTVar(name, INT) for name in state_vars}
    body_lookup = dict(var_lookup)
    _process_body_stmts(loop_stmt.body, body_lookup, next_state, state_vars)

    return {
        'state_vars': state_vars,
        'pre_assignments': pre_assignments,
        'cond_ast': loop_stmt.cond,
        'body_ast': loop_stmt.body,
        'cond_smt': cond_smt,
        'next_state': next_state,
        'var_lookup': var_lookup,
    }


# ============================================================
# Ranking Function Construction and Verification
# ============================================================

def _build_ranking_expr(coefficients, var_lookup):
    """Build SMT expression for ranking function: c0 + c1*x1 + ... + cn*xn."""
    terms = []
    const = coefficients.get('_const', 0)
    if const != 0:
        terms.append(IntConst(const))

    for name, coeff in coefficients.items():
        if name == '_const':
            continue
        if coeff == 0:
            continue
        var = var_lookup[name]
        if coeff == 1:
            terms.append(var)
        elif coeff == -1:
            terms.append(App(Op.SUB, [IntConst(0), var], INT))
        else:
            terms.append(App(Op.MUL, [IntConst(coeff), var], INT))

    if not terms:
        return IntConst(0)
    result = terms[0]
    for t in terms[1:]:
        result = App(Op.ADD, [result, t], INT)
    return result


def _coefficients_to_str(coefficients):
    """Convert coefficients to human-readable expression string."""
    parts = []
    const = coefficients.get('_const', 0)

    for name, coeff in sorted(coefficients.items()):
        if name == '_const':
            continue
        if coeff == 0:
            continue
        if coeff == 1:
            parts.append(name)
        elif coeff == -1:
            parts.append(f"-{name}")
        else:
            parts.append(f"{coeff}*{name}")

    if const != 0:
        parts.append(str(const))

    if not parts:
        return "0"
    return " + ".join(parts).replace("+ -", "- ")


def verify_ranking_function(loop_info, coefficients):
    """
    Verify that a ranking function with given coefficients proves termination.

    Checks two conditions:
    1. Bounded: cond(s) => R(s) >= 0
    2. Decreasing: cond(s) AND body transition => R(s) - R(s') >= 1

    Returns: (bounded_ok, decreasing_ok)
    """
    state_vars = loop_info['state_vars']
    cond_smt = loop_info['cond_smt']
    next_state = loop_info['next_state']
    var_lookup = loop_info['var_lookup']

    # Build R(current state)
    r_current = _build_ranking_expr(coefficients, var_lookup)

    # Build R(next state) by substituting next-state expressions
    next_lookup = {}
    for name in state_vars:
        next_lookup[name] = next_state[name]
    r_next = _build_ranking_expr(coefficients, next_lookup)

    # Check 1: Bounded -- cond => R >= 0
    # Negate: cond AND R < 0 is SAT?
    s1 = SMTSolver()
    # Register variables
    for name in state_vars:
        s1.Int(name)
    r_curr_term = _rebuild_in_solver(r_current, s1)
    cond_term = _rebuild_in_solver(cond_smt, s1)
    s1.add(cond_term)
    s1.add(App(Op.LT, [r_curr_term, IntConst(0)], BOOL))
    bounded_ok = s1.check() == SMTResult.UNSAT

    # Check 2: Decreasing -- cond AND trans => R(s) - R(s') >= 1
    # Negate: cond AND trans AND R(s) - R(s') < 1 is SAT?
    # Equivalently: cond AND trans AND R(s) - R(s') <= 0
    s2 = SMTSolver()
    # Register current and next-state variables
    for name in state_vars:
        s2.Int(name)
    # We need to register variables that appear in next_state expressions
    _register_vars(next_state, s2, state_vars)

    cond_term2 = _rebuild_in_solver(cond_smt, s2)
    r_curr_term2 = _rebuild_in_solver(r_current, s2)
    r_next_term2 = _rebuild_in_solver(r_next, s2)

    # Assert condition holds
    s2.add(cond_term2)

    # Assert transition relation: for each var, next_var == next_state_expr
    for name in state_vars:
        nv = s2.Int(f"{name}_next")
        ns_expr = _rebuild_in_solver(next_state[name], s2)
        s2.add(App(Op.EQ, [nv, ns_expr], BOOL))

    # Build R(next) using _next variables
    next_var_lookup = {name: s2.Int(f"{name}_next") for name in state_vars}
    r_next_via_next_vars = _build_ranking_expr(coefficients, next_var_lookup)
    r_next_final = _rebuild_in_solver(r_next_via_next_vars, s2)
    r_curr_final = _rebuild_in_solver(r_current, s2)

    # R(s) - R(s') <= 0 means NOT decreasing by at least 1
    diff = App(Op.SUB, [r_curr_final, r_next_final], INT)
    s2.add(App(Op.LE, [diff, IntConst(0)], BOOL))

    decreasing_ok = s2.check() == SMTResult.UNSAT

    return bounded_ok, decreasing_ok


def _register_vars(next_state, solver, state_vars):
    """Register any additional variables needed for next_state expressions."""
    # The solver already has the state vars registered via s.Int(name)
    # Next-state expressions reference state vars, which are already registered
    pass


def _rebuild_in_solver(term, solver):
    """
    Rebuild an SMT term using solver-registered variables.
    This ensures variable identity matches within the solver.
    """
    if isinstance(term, IntConst):
        return term
    elif isinstance(term, BoolConst):
        return term
    elif hasattr(term, 'name') and hasattr(term, 'sort'):
        # It's a Var
        return solver.Int(term.name) if term.sort == INT else solver.Bool(term.name)
    elif isinstance(term, App):
        rebuilt_args = [_rebuild_in_solver(a, solver) for a in term.args]
        return App(term.op, rebuilt_args, term.sort)
    return term


# ============================================================
# Candidate Generation
# ============================================================

def _generate_single_var_candidates(state_vars, pre_assignments, cond_ast):
    """
    Generate single-variable ranking function candidates.

    For each variable:
      - R = x (var decreases to 0)
      - R = -x (var increases from negative)
      - R = bound - x (var approaches a bound from below)
      - R = x - bound (var approaches a bound from above)
    """
    candidates = []

    # Extract constants from condition for bound estimation
    cond_constants = _extract_constants(cond_ast)

    for var in state_vars:
        # R = x (simple countdown)
        candidates.append({'_const': 0, var: 1})
        # R = -x (count-up to 0)
        candidates.append({'_const': 0, var: -1})

        # R = bound - x for each constant in condition
        for c in cond_constants:
            candidates.append({'_const': c, var: -1})  # c - x
            candidates.append({'_const': -c, var: 1})   # x - c
            candidates.append({'_const': c, var: 1})     # c + x
            candidates.append({'_const': -c, var: -1})   # -c - x

        # Use pre-assignment values as bounds
        if var in pre_assignments and pre_assignments[var] is not None:
            init_val = pre_assignments[var]
            candidates.append({'_const': init_val, var: -1})  # init - x
            candidates.append({'_const': -init_val, var: 1})  # x - init

    return candidates


def _generate_two_var_candidates(state_vars, pre_assignments, cond_ast):
    """Generate two-variable linear ranking function candidates."""
    candidates = []
    cond_constants = _extract_constants(cond_ast)
    consts = list(set([0, 1, -1] + list(cond_constants)))

    for i, v1 in enumerate(state_vars):
        for v2 in state_vars[i+1:]:
            for c1 in [1, -1]:
                for c2 in [1, -1]:
                    for c0 in consts:
                        candidates.append({'_const': c0, v1: c1, v2: c2})

    return candidates


def _extract_constants(expr):
    """Extract integer constants from an AST expression."""
    constants = set()
    if isinstance(expr, IntLit):
        constants.add(expr.value)
    elif isinstance(expr, BinOp):
        constants |= _extract_constants(expr.left)
        constants |= _extract_constants(expr.right)
    elif isinstance(expr, UnaryOp):
        constants |= _extract_constants(expr.operand)
        if expr.op == '-' and isinstance(expr.operand, IntLit):
            constants.add(-expr.operand.value)
    return constants


def _extract_constants_from_body(body):
    """Extract integer constants from loop body."""
    constants = set()
    if isinstance(body, Block):
        stmts = body.stmts
    elif isinstance(body, list):
        stmts = body
    else:
        stmts = [body]

    for stmt in stmts:
        if isinstance(stmt, (LetDecl, Assign)):
            constants |= _extract_constants(stmt.value)
        elif isinstance(stmt, IfStmt):
            constants |= _extract_constants(stmt.cond)
            constants |= _extract_constants_from_body(stmt.then_body)
            if stmt.else_body:
                constants |= _extract_constants_from_body(stmt.else_body)
    return constants


def _generate_condition_derived_candidates(state_vars, cond_ast, pre_assignments):
    """
    Generate candidates derived from the loop condition structure.

    For `while (x < n)`, the ranking function is `n - x`.
    For `while (x > 0)`, the ranking function is `x`.
    For `while (x != 0)`, try `x` and `-x`.
    """
    candidates = []
    if isinstance(cond_ast, BinOp):
        left_vars = _collect_vars_in_expr(cond_ast.left)
        right_vars = _collect_vars_in_expr(cond_ast.right)

        if cond_ast.op in ('<', '<='):
            # x < n or x <= n: ranking = n - x
            if len(left_vars) == 1 and len(right_vars) <= 1:
                lv = list(left_vars)[0]
                if right_vars:
                    rv = list(right_vars)[0]
                    # n - x
                    candidates.append({rv: 1, lv: -1, '_const': 0})
                    candidates.append({rv: 1, lv: -1, '_const': 1})  # n - x + 1
                elif isinstance(cond_ast.right, IntLit):
                    c = cond_ast.right.value
                    candidates.append({lv: -1, '_const': c})
                    candidates.append({lv: -1, '_const': c + 1})

        elif cond_ast.op in ('>', '>='):
            # x > 0 or x >= 1: ranking = x
            if len(left_vars) == 1:
                lv = list(left_vars)[0]
                candidates.append({lv: 1, '_const': 0})
                if isinstance(cond_ast.right, IntLit):
                    c = cond_ast.right.value
                    candidates.append({lv: 1, '_const': -c})

        elif cond_ast.op == '!=':
            # x != 0: could go either way
            if len(left_vars) == 1:
                lv = list(left_vars)[0]
                candidates.append({lv: 1, '_const': 0})
                candidates.append({lv: -1, '_const': 0})

        elif cond_ast.op == 'and':
            # Conjunction: recurse on each side
            candidates += _generate_condition_derived_candidates(
                state_vars, cond_ast.left, pre_assignments)
            candidates += _generate_condition_derived_candidates(
                state_vars, cond_ast.right, pre_assignments)

    return candidates


def generate_candidates(loop_info):
    """Generate all ranking function candidates for a loop."""
    state_vars = loop_info['state_vars']
    pre_assignments = loop_info['pre_assignments']
    cond_ast = loop_info['cond_ast']

    candidates = []

    # Tier 1: Condition-derived (most likely to work)
    candidates += _generate_condition_derived_candidates(
        state_vars, cond_ast, pre_assignments)

    # Tier 2: Single-variable candidates
    candidates += _generate_single_var_candidates(
        state_vars, pre_assignments, cond_ast)

    # Tier 3: Two-variable candidates
    if len(state_vars) >= 2:
        candidates += _generate_two_var_candidates(
            state_vars, pre_assignments, cond_ast)

    # Deduplicate by string representation
    seen = set()
    unique = []
    for c in candidates:
        key = str(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ============================================================
# Main Termination Analysis
# ============================================================

def find_ranking_function(source, loop_index=0):
    """
    Find a ranking function for the specified loop.

    Returns: RankingFunction if found, None otherwise.
    """
    loop_info = extract_loop_info(source, loop_index)
    candidates = generate_candidates(loop_info)

    for coefficients in candidates:
        bounded, decreasing = verify_ranking_function(loop_info, coefficients)
        if bounded and decreasing:
            expr_str = _coefficients_to_str(coefficients)
            return RankingFunction(
                expression=expr_str,
                coefficients=coefficients,
                kind="linear"
            )

    return None


def find_lexicographic_ranking(source, loop_index=0, max_components=3):
    """
    Find a lexicographic ranking function for the specified loop.

    A lexicographic ranking (R1, R2, ..., Rk) proves termination if:
      - At each step, R1 decreases, OR
      - R1 stays the same and R2 decreases, OR
      - R1 and R2 stay the same and R3 decreases, etc.

    This handles loops where no single expression decreases monotonically.
    """
    loop_info = extract_loop_info(source, loop_index)
    candidates = generate_candidates(loop_info)

    # First try single ranking function
    for coefficients in candidates:
        bounded, decreasing = verify_ranking_function(loop_info, coefficients)
        if bounded and decreasing:
            rf = RankingFunction(
                expression=_coefficients_to_str(coefficients),
                coefficients=coefficients,
                kind="linear"
            )
            return LexRankingFunction(components=[rf])

    # Try lexicographic pairs
    if max_components >= 2:
        valid_bounded = []
        for coefficients in candidates:
            bounded, _ = verify_ranking_function(loop_info, coefficients)
            if bounded:
                valid_bounded.append(coefficients)

        for i, c1 in enumerate(valid_bounded):
            for c2 in valid_bounded:
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


def _verify_lex_pair(loop_info, c1, c2):
    """
    Verify lexicographic pair (R1, R2) proves termination.

    Condition: cond AND trans => (R1 decreases) OR (R1 same AND R2 decreases)
    Equivalently:
      cond AND trans AND R1'>=R1 AND (R1'>R1 OR R2'>=R2) is UNSAT

    That is: cond AND trans AND NOT((R1 decreases) OR (R1 same AND R2 decreases)) is UNSAT
    NOT(...) = R1' >= R1 AND NOT(R1 same AND R2 decreases)
             = R1' >= R1 AND (R1' != R1 OR R2' >= R2)
    But R1' >= R1 AND R1' != R1 means R1' > R1 which contradicts nothing...

    Simpler: check cond AND trans =>
       (R1(s) - R1(s') >= 1) OR (R1(s) == R1(s') AND R2(s) - R2(s') >= 1)

    Negate: cond AND trans AND R1(s)-R1(s') <= 0 AND (R1(s) != R1(s') OR R2(s)-R2(s') <= 0)
    """
    state_vars = loop_info['state_vars']
    cond_smt = loop_info['cond_smt']
    next_state = loop_info['next_state']
    var_lookup = loop_info['var_lookup']

    s = SMTSolver()
    for name in state_vars:
        s.Int(name)

    # Register next-state variables
    for name in state_vars:
        nv = s.Int(f"{name}_next")
        ns_expr = _rebuild_in_solver(next_state[name], s)
        s.add(App(Op.EQ, [nv, ns_expr], BOOL))

    next_var_lookup = {name: s.Int(f"{name}_next") for name in state_vars}

    # Assert condition
    s.add(_rebuild_in_solver(cond_smt, s))

    # Build ranking expressions
    r1_curr = _rebuild_in_solver(_build_ranking_expr(c1, var_lookup), s)
    r1_next = _rebuild_in_solver(_build_ranking_expr(c1, next_var_lookup), s)
    r2_curr = _rebuild_in_solver(_build_ranking_expr(c2, var_lookup), s)
    r2_next = _rebuild_in_solver(_build_ranking_expr(c2, next_var_lookup), s)

    # R1 diff <= 0 (R1 does NOT decrease by at least 1)
    diff1 = App(Op.SUB, [r1_curr, r1_next], INT)
    s.add(App(Op.LE, [diff1, IntConst(0)], BOOL))

    # R1 != R1' OR R2 diff <= 0
    diff2 = App(Op.SUB, [r2_curr, r2_next], INT)
    r1_neq = App(Op.NEQ, [r1_curr, r1_next], BOOL)
    r2_no_decrease = App(Op.LE, [diff2, IntConst(0)], BOOL)
    s.add(App(Op.OR, [r1_neq, r2_no_decrease], BOOL))

    return s.check() == SMTResult.UNSAT


def prove_termination(source, loop_index=0):
    """
    Prove termination of the specified loop.

    Tries:
    1. Single linear ranking function
    2. Lexicographic ranking function

    Returns: LoopTermResult
    """
    loop_info = extract_loop_info(source, loop_index)
    candidates = generate_candidates(loop_info)
    tried = 0

    # Try single ranking functions
    for coefficients in candidates:
        tried += 1
        bounded, decreasing = verify_ranking_function(loop_info, coefficients)
        if bounded and decreasing:
            expr_str = _coefficients_to_str(coefficients)
            rf = RankingFunction(
                expression=expr_str,
                coefficients=coefficients,
                kind="linear"
            )
            return LoopTermResult(
                loop_index=loop_index,
                result=TermResult.TERMINATES,
                ranking_function=rf,
                candidates_tried=tried,
                message=f"Terminates with ranking function: {expr_str}"
            )

    # Try lexicographic
    lex_rf = find_lexicographic_ranking(source, loop_index)
    if lex_rf:
        return LoopTermResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=lex_rf,
            candidates_tried=tried,
            message=f"Terminates with lexicographic ranking: {lex_rf.expression}"
        )

    return LoopTermResult(
        loop_index=loop_index,
        result=TermResult.UNKNOWN,
        candidates_tried=tried,
        message=f"Could not find ranking function after {tried} candidates"
    )


def check_ranking_function(source, ranking_expr, loop_index=0):
    """
    Verify a user-provided ranking function expression.

    ranking_expr: dict of coefficients, e.g. {'n': 1, 'i': -1, '_const': 0}

    Returns: (bounded: bool, decreasing: bool)
    """
    loop_info = extract_loop_info(source, loop_index)
    return verify_ranking_function(loop_info, ranking_expr)


def analyze_termination(source):
    """
    Analyze termination of ALL loops in a program.

    Returns: TerminationResult with per-loop results.
    """
    program = parse(source)
    loops = _find_all_loops(program.stmts)

    if not loops:
        return TerminationResult(
            result=TermResult.TERMINATES,
            loops_analyzed=0,
            loops_proved=0,
            message="No loops found -- program trivially terminates"
        )

    loop_results = []
    for idx in range(len(loops)):
        try:
            lr = prove_termination(source, loop_index=idx)
            loop_results.append(lr)
        except Exception as e:
            loop_results.append(LoopTermResult(
                loop_index=idx,
                result=TermResult.UNKNOWN,
                message=f"Error analyzing loop {idx}: {str(e)}"
            ))

    proved = sum(1 for r in loop_results if r.result == TermResult.TERMINATES)
    overall = TermResult.TERMINATES if proved == len(loops) else TermResult.UNKNOWN

    return TerminationResult(
        result=overall,
        loops_analyzed=len(loops),
        loops_proved=proved,
        loop_results=loop_results,
        message=f"Proved {proved}/{len(loops)} loops terminate"
    )


def _find_all_loops(stmts):
    """Find all while loops in a statement list."""
    loops = []
    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            loops.append(stmt)
            # Also recurse into while body to find nested loops
            if isinstance(stmt.body, Block):
                loops += _find_all_loops(stmt.body.stmts)
        elif isinstance(stmt, IfStmt):
            if isinstance(stmt.then_body, Block):
                loops += _find_all_loops(stmt.then_body.stmts)
            if stmt.else_body and isinstance(stmt.else_body, Block):
                loops += _find_all_loops(stmt.else_body.stmts)
        elif isinstance(stmt, Block):
            loops += _find_all_loops(stmt.stmts)
        elif isinstance(stmt, FnDecl):
            if isinstance(stmt.body, Block):
                loops += _find_all_loops(stmt.body.stmts)
    return loops


# ============================================================
# Conditional Ranking Functions
# ============================================================

def find_conditional_ranking(source, loop_index=0):
    """
    Find a conditional ranking function: different ranking expressions
    for different branches of the loop body.

    For loops with if-then-else:
      - R = R1 when condition holds
      - R = R2 when condition doesn't hold
      Both must decrease and be bounded.

    Returns: RankingFunction with kind="conditional" if found, None otherwise.
    """
    loop_info = extract_loop_info(source, loop_index)
    body = loop_info['body_ast']

    # Check if the body is a conditional
    if isinstance(body, Block) and len(body.stmts) >= 1:
        for stmt in body.stmts:
            if isinstance(stmt, IfStmt):
                # Try to find ranking functions for each branch
                result = _try_conditional_ranking(loop_info, stmt)
                if result:
                    return result

    return None


def _try_conditional_ranking(loop_info, if_stmt):
    """Try to find conditional ranking for an if-then-else in loop body."""
    state_vars = loop_info['state_vars']
    var_lookup = loop_info['var_lookup']
    cond_smt = loop_info['cond_smt']

    # Build branch conditions
    branch_cond = _expr_to_smt(if_stmt.cond, var_lookup)

    # Build next-state for then branch
    then_next = {name: SMTVar(name, INT) for name in state_vars}
    then_lookup = dict(var_lookup)
    _process_body_stmts(if_stmt.then_body, then_lookup, then_next, state_vars)

    # Build next-state for else branch
    else_next = {name: SMTVar(name, INT) for name in state_vars}
    else_lookup = dict(var_lookup)
    if if_stmt.else_body:
        _process_body_stmts(if_stmt.else_body, else_lookup, else_next, state_vars)

    candidates = generate_candidates(loop_info)

    # Try each pair (R_then, R_else)
    for c_then in candidates:
        for c_else in candidates:
            ok = _verify_conditional_ranking(
                loop_info, branch_cond, c_then, then_next, c_else, else_next)
            if ok:
                expr_then = _coefficients_to_str(c_then)
                expr_else = _coefficients_to_str(c_else)
                return RankingFunction(
                    expression=f"if branch: {expr_then}, else: {expr_else}",
                    coefficients={'then': c_then, 'else': c_else},
                    kind="conditional"
                )

    return None


def _verify_conditional_ranking(loop_info, branch_cond, c_then, then_next,
                                 c_else, else_next):
    """Verify conditional ranking function."""
    state_vars = loop_info['state_vars']
    cond_smt = loop_info['cond_smt']
    var_lookup = loop_info['var_lookup']

    s = SMTSolver()
    for name in state_vars:
        s.Int(name)

    cond_term = _rebuild_in_solver(cond_smt, s)
    branch_term = _rebuild_in_solver(branch_cond, s)

    # Build R for current state (use the max of both to get a single measure)
    r_then_curr = _rebuild_in_solver(_build_ranking_expr(c_then, var_lookup), s)
    r_else_curr = _rebuild_in_solver(_build_ranking_expr(c_else, var_lookup), s)

    # Next-state vars
    for name in state_vars:
        nv = s.Int(f"{name}_next_then")
        s.add(App(Op.EQ, [nv, _rebuild_in_solver(then_next[name], s)], BOOL))
        nv2 = s.Int(f"{name}_next_else")
        s.add(App(Op.EQ, [nv2, _rebuild_in_solver(else_next[name], s)], BOOL))

    then_next_lookup = {name: s.Int(f"{name}_next_then") for name in state_vars}
    else_next_lookup = {name: s.Int(f"{name}_next_else") for name in state_vars}

    r_then_next = _rebuild_in_solver(_build_ranking_expr(c_then, then_next_lookup), s)
    r_else_next = _rebuild_in_solver(_build_ranking_expr(c_else, else_next_lookup), s)

    # Check then-branch: loop_cond AND branch_cond => R_then >= 0 AND R_then decreases
    s.push()
    s.add(cond_term)
    s.add(branch_term)
    diff_then = App(Op.SUB, [r_then_curr, r_then_next], INT)
    # NOT(R >= 0 AND diff >= 1)
    s.add(App(Op.OR, [
        App(Op.LT, [r_then_curr, IntConst(0)], BOOL),
        App(Op.LE, [diff_then, IntConst(0)], BOOL)
    ], BOOL))
    then_ok = s.check() == SMTResult.UNSAT
    s.pop()

    if not then_ok:
        return False

    # Check else-branch: loop_cond AND NOT(branch_cond) => R_else >= 0 AND R_else decreases
    s.push()
    s.add(cond_term)
    s.add(_negate(branch_term))
    diff_else = App(Op.SUB, [r_else_curr, r_else_next], INT)
    s.add(App(Op.OR, [
        App(Op.LT, [r_else_curr, IntConst(0)], BOOL),
        App(Op.LE, [diff_else, IntConst(0)], BOOL)
    ], BOOL))
    else_ok = s.check() == SMTResult.UNSAT
    s.pop()

    return else_ok


# ============================================================
# Abstract Interpretation Enhanced Candidate Generation
# ============================================================

def find_ranking_with_ai(source, loop_index=0):
    """
    Use abstract interpretation to estimate variable ranges and generate
    better ranking function candidates.
    """
    loop_info = extract_loop_info(source, loop_index)

    # Run abstract interpretation
    try:
        ai_result = ai_analyze(source)
        env = ai_result['env']
    except Exception:
        env = None

    # Generate base candidates
    candidates = generate_candidates(loop_info)

    # Add AI-derived candidates
    if env:
        state_vars = loop_info['state_vars']
        for var in state_vars:
            interval = env.get_interval(var)
            if interval and not interval.is_top() and not interval.is_bot():
                if hasattr(interval, 'hi') and interval.hi != INF:
                    hi = int(interval.hi)
                    candidates.append({var: -1, '_const': hi})  # hi - x
                if hasattr(interval, 'lo') and interval.lo != NEG_INF:
                    lo = int(interval.lo)
                    candidates.append({var: 1, '_const': -lo})  # x - lo

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = str(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Try each candidate
    for coefficients in unique:
        bounded, decreasing = verify_ranking_function(loop_info, coefficients)
        if bounded and decreasing:
            expr_str = _coefficients_to_str(coefficients)
            return RankingFunction(
                expression=expr_str,
                coefficients=coefficients,
                kind="linear"
            )

    return None


# ============================================================
# Nontermination Detection
# ============================================================

def detect_nontermination(source, loop_index=0, max_depth=10):
    """
    Try to detect nontermination by finding a reachable state from which
    the loop never exits (recurrent set).

    Approach: Find a state s such that:
      1. s is reachable from init (via BMC unrolling)
      2. cond(s) holds
      3. After one transition, cond(s') still holds
      4. No variable changes in a way that would eventually falsify cond

    Simplified: find a cycle -- a state s such that after transition, s' == s.
    """
    loop_info = extract_loop_info(source, loop_index)
    state_vars = loop_info['state_vars']
    cond_smt = loop_info['cond_smt']
    next_state = loop_info['next_state']
    pre_assignments = loop_info['pre_assignments']
    var_lookup = loop_info['var_lookup']

    # Method 1: Check for reachable fixed point via BMC + fixed-point constraint
    # Unroll from init for up to max_depth steps, then check if state is a fixed point
    for depth in range(max_depth + 1):
        s = SMTSolver()
        # Step-indexed variables
        for step in range(depth + 1):
            for name in state_vars:
                s.Int(f"{name}_{step}")

        # Init constraints
        for name in state_vars:
            v0 = s.Int(f"{name}_0")
            if name in pre_assignments and pre_assignments[name] is not None:
                s.add(App(Op.EQ, [v0, IntConst(pre_assignments[name])], BOOL))

        # Transitions for each step
        for step in range(depth):
            # Condition holds at this step
            cond_at_step = _rebuild_in_solver(
                _expr_to_smt(loop_info['cond_ast'],
                             {name: SMTVar(f"{name}_{step}", INT) for name in state_vars}),
                s
            )
            s.add(cond_at_step)
            # Transition
            for name in state_vars:
                ns = _rebuild_in_solver(
                    _substitute_vars(next_state[name], state_vars, step),
                    s
                )
                s.add(App(Op.EQ, [s.Int(f"{name}_{step+1}"), ns], BOOL))

        # At final step: cond holds AND state is a fixed point (next == current)
        final_cond = _rebuild_in_solver(
            _expr_to_smt(loop_info['cond_ast'],
                         {name: SMTVar(f"{name}_{depth}", INT) for name in state_vars}),
            s
        )
        s.add(final_cond)

        for name in state_vars:
            ns = _rebuild_in_solver(
                _substitute_vars(next_state[name], state_vars, depth),
                s
            )
            s.add(App(Op.EQ, [s.Int(f"{name}_{depth}"), ns], BOOL))

        if s.check() == SMTResult.SAT:
            model = s.model()
            return LoopTermResult(
                loop_index=loop_index,
                result=TermResult.NONTERMINATING,
                message=f"Found reachable fixed point at depth {depth}: {model}"
            )

    # Method 2: Check for 2-cycle (s -> s' -> s)
    s2 = SMTSolver()
    for name in state_vars:
        s2.Int(name)
        s2.Int(f"{name}_mid")

    s2.add(_rebuild_in_solver(cond_smt, s2))

    # First step: s -> s_mid
    for name in state_vars:
        ns_expr = _rebuild_in_solver(next_state[name], s2)
        mid = s2.Int(f"{name}_mid")
        s2.add(App(Op.EQ, [mid, ns_expr], BOOL))

    # Cond holds at mid
    mid_lookup = {name: s2.Int(f"{name}_mid") for name in state_vars}
    mid_cond = _rebuild_in_solver(
        _expr_to_smt(loop_info['cond_ast'], {name: SMTVar(name, INT) for name in state_vars}),
        s2
    )
    # Rebuild cond with mid vars
    mid_cond_expr = _expr_to_smt(
        loop_info['cond_ast'],
        {name: SMTVar(f"{name}_mid", INT) for name in state_vars}
    )
    s2.add(_rebuild_in_solver(mid_cond_expr, s2))

    # Second step: s_mid -> s (cycle back)
    mid_next = {name: SMTVar(f"{name}_mid", INT) for name in state_vars}
    body_lookup_mid = dict(mid_lookup)
    mid_next_state = dict(mid_next)
    # We need the transition using mid vars -- rebuild manually
    for name in state_vars:
        # Replace var references with mid vars in next_state expressions
        pass  # Complex -- skip for now

    # Instead, just check: mid vars map back to original vars
    for name in state_vars:
        curr = s2.Int(name)
        mid = s2.Int(f"{name}_mid")
        # For simple loops: next(mid) should equal curr
        # This is complex for general transitions; keep method 1 as primary

    return LoopTermResult(
        loop_index=loop_index,
        result=TermResult.UNKNOWN,
        message="Could not determine termination or nontermination"
    )


def _check_reachable(loop_info, target_state, max_depth):
    """Check if target_state is reachable from init within max_depth steps."""
    state_vars = loop_info['state_vars']
    pre_assignments = loop_info['pre_assignments']
    next_state = loop_info['next_state']
    cond_smt = loop_info['cond_smt']

    for depth in range(max_depth + 1):
        s = SMTSolver()

        # Step-indexed variables
        for step in range(depth + 1):
            for name in state_vars:
                s.Int(f"{name}_{step}")

        # Init
        for name in state_vars:
            v0 = s.Int(f"{name}_0")
            if name in pre_assignments and pre_assignments[name] is not None:
                s.add(App(Op.EQ, [v0, IntConst(pre_assignments[name])], BOOL))

        # Transitions
        for step in range(depth):
            step_lookup = {name: s.Int(f"{name}_{step}") for name in state_vars}
            next_step_lookup = {name: s.Int(f"{name}_{step+1}") for name in state_vars}

            # Condition holds at this step
            cond_at_step = _rebuild_in_solver(
                _expr_to_smt(loop_info['cond_ast'],
                             {name: SMTVar(f"{name}_{step}", INT) for name in state_vars}),
                s
            )
            s.add(cond_at_step)

            # Transition
            for name in state_vars:
                ns = _rebuild_in_solver(
                    _substitute_vars(next_state[name], state_vars, step),
                    s
                )
                s.add(App(Op.EQ, [s.Int(f"{name}_{step+1}"), ns], BOOL))

        # Target at final step
        for name in state_vars:
            vf = s.Int(f"{name}_{depth}")
            if name in target_state:
                s.add(App(Op.EQ, [vf, IntConst(target_state[name])], BOOL))

        if s.check() == SMTResult.SAT:
            return True

    return False


def _substitute_vars(term, state_vars, step):
    """Substitute state vars with step-indexed versions."""
    if isinstance(term, IntConst) or isinstance(term, BoolConst):
        return term
    elif hasattr(term, 'name') and hasattr(term, 'sort'):
        # Var
        if term.name in state_vars:
            return SMTVar(f"{term.name}_{step}", term.sort)
        return term
    elif isinstance(term, App):
        new_args = [_substitute_vars(a, state_vars, step) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


# ============================================================
# High-Level API
# ============================================================

def verify_terminates(source, loop_index=0):
    """
    Full termination analysis: try ranking functions, then nontermination detection.

    Returns: LoopTermResult
    """
    # First try to prove termination
    result = prove_termination(source, loop_index)
    if result.result == TermResult.TERMINATES:
        return result

    # Try AI-enhanced search
    rf = find_ranking_with_ai(source, loop_index)
    if rf:
        return LoopTermResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=rf,
            message=f"Terminates with AI-guided ranking function: {rf.expression}"
        )

    # Try conditional ranking
    crf = find_conditional_ranking(source, loop_index)
    if crf:
        return LoopTermResult(
            loop_index=loop_index,
            result=TermResult.TERMINATES,
            ranking_function=crf,
            message=f"Terminates with conditional ranking: {crf.expression}"
        )

    # Try nontermination detection
    nonterm = detect_nontermination(source, loop_index)
    if nonterm.result == TermResult.NONTERMINATING:
        return nonterm

    return LoopTermResult(
        loop_index=loop_index,
        result=TermResult.UNKNOWN,
        candidates_tried=result.candidates_tried,
        message="Could not determine termination"
    )


def verify_all_terminate(source):
    """
    Verify termination of all loops in a program.

    Returns: TerminationResult
    """
    return analyze_termination(source)


def compare_ranking_strategies(source, loop_index=0):
    """
    Compare different ranking function discovery strategies.

    Returns: dict with strategy names -> results
    """
    results = {}

    # Strategy 1: Condition-derived only
    loop_info = extract_loop_info(source, loop_index)
    cond_candidates = _generate_condition_derived_candidates(
        loop_info['state_vars'], loop_info['cond_ast'], loop_info['pre_assignments'])
    for c in cond_candidates:
        b, d = verify_ranking_function(loop_info, c)
        if b and d:
            results['condition_derived'] = _coefficients_to_str(c)
            break
    else:
        results['condition_derived'] = None

    # Strategy 2: Single variable
    single_candidates = _generate_single_var_candidates(
        loop_info['state_vars'], loop_info['pre_assignments'], loop_info['cond_ast'])
    for c in single_candidates:
        b, d = verify_ranking_function(loop_info, c)
        if b and d:
            results['single_variable'] = _coefficients_to_str(c)
            break
    else:
        results['single_variable'] = None

    # Strategy 3: Two variable
    if len(loop_info['state_vars']) >= 2:
        two_candidates = _generate_two_var_candidates(
            loop_info['state_vars'], loop_info['pre_assignments'], loop_info['cond_ast'])
        for c in two_candidates:
            b, d = verify_ranking_function(loop_info, c)
            if b and d:
                results['two_variable'] = _coefficients_to_str(c)
                break
        else:
            results['two_variable'] = None

    # Strategy 4: Full search
    rf = find_ranking_function(source, loop_index)
    results['full_search'] = rf.expression if rf else None

    # Strategy 5: AI-enhanced
    ai_rf = find_ranking_with_ai(source, loop_index)
    results['ai_enhanced'] = ai_rf.expression if ai_rf else None

    return results
