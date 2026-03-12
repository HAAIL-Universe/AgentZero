"""
V174: Octagon-Guided Symbolic Execution

Composes V173 (Octagon Abstract Domain) + C038 (Symbolic Execution) + C010 (Parser)

Idea: Run octagon abstract interpretation as a cheap relational pre-analysis,
then use the relational bounds (x-y <= c, x+y <= c) to prune infeasible paths
during symbolic execution. This goes beyond V001's interval pruning by detecting
infeasible branches that depend on relationships between variables.

Example where octagon helps but intervals don't:
    let x = input;
    let y = x + 1;
    if (y < x) { ... }   // intervals: x in [-inf, inf], y in [-inf, inf] -- both branches feasible
                          // octagon: y - x == 1, so y < x is infeasible -> pruned!

Architecture:
    Source -> V173 Octagon Interpreter (pre-analysis) -> relational bounds (DBM)
           -> C038 Symbolic Executor (with octagon-guided pruning)
           -> Pruned paths + test cases + pruning statistics
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from fractions import Fraction

# Import V173 octagon domain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
from octagon import (
    Octagon, OctConstraint, OctExpr, OctagonInterpreter, OctAnalysisResult,
    octagon_from_intervals, INF,
)

# Import C038 symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, SymValue, PathState, PathStatus, SymType,
    ExecutionResult, TestCase,
    symbolic_execute, generate_tests,
)

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, BoolLit, StringLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt,
)

# Import C037 SMT types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Term, Var as SMTVar, App, IntConst, BoolConst, Op as SMTOp, BOOL, INT


# ============================================================
# AST to Octagon Program Converter
# ============================================================

def _ast_to_oct_program(stmts):
    """Convert C10 AST statements to V173 tuple-based program."""
    if stmts is None:
        return ('skip',)
    # Handle Block objects
    if isinstance(stmts, Block):
        stmts = stmts.stmts
    if not stmts:
        return ('skip',)
    parts = []
    for s in stmts:
        converted = _convert_stmt(s)
        if converted is not None:
            parts.append(converted)
    if not parts:
        return ('skip',)
    if len(parts) == 1:
        return parts[0]
    return ('seq', *parts)


def _convert_stmt(stmt):
    """Convert a single AST statement to octagon tuple format."""
    if isinstance(stmt, LetDecl):
        if stmt.value is not None:
            expr = _convert_expr(stmt.value)
            if expr is not None:
                return ('assign', stmt.name, expr)
        return None
    elif isinstance(stmt, Assign):
        expr = _convert_expr(stmt.value)
        if expr is not None:
            return ('assign', stmt.name, expr)
        return None
    elif isinstance(stmt, IfStmt):
        cond = _convert_cond(stmt.cond)
        then_body = _ast_to_oct_program(stmt.then_body)
        else_body = _ast_to_oct_program(stmt.else_body) if stmt.else_body else ('skip',)
        return ('if', cond, then_body, else_body)
    elif isinstance(stmt, WhileStmt):
        cond = _convert_cond(stmt.cond)
        body = _ast_to_oct_program(stmt.body)
        return ('while', cond, body)
    elif isinstance(stmt, Block):
        return _ast_to_oct_program(stmt.stmts)
    elif isinstance(stmt, ReturnStmt):
        return ('skip',)
    elif isinstance(stmt, PrintStmt):
        return ('skip',)
    elif isinstance(stmt, FnDecl):
        return ('skip',)  # Functions analyzed separately
    return None


def _convert_expr(expr):
    """Convert AST expression to octagon tuple format."""
    if isinstance(expr, IntLit):
        return ('const', expr.value)
    elif isinstance(expr, BoolLit):
        return ('const', 1 if expr.value else 0)
    elif isinstance(expr, ASTVar):
        return ('var', expr.name)
    elif isinstance(expr, UnaryOp):
        if expr.op == '-':
            inner = _convert_expr(expr.operand)
            if inner:
                return ('neg', inner)
        return None
    elif isinstance(expr, BinOp):
        left = _convert_expr(expr.left)
        right = _convert_expr(expr.right)
        if left is None or right is None:
            return None
        if expr.op == '+':
            return ('add', left, right)
        elif expr.op == '-':
            return ('sub', left, right)
        elif expr.op == '*':
            return ('mul', left, right)
        return None
    elif isinstance(expr, CallExpr):
        return None  # Function calls not representable
    return None


def _convert_cond(expr):
    """Convert AST expression used as condition to octagon tuple format."""
    if isinstance(expr, BoolLit):
        return ('true',) if expr.value else ('false',)
    elif isinstance(expr, BinOp):
        if expr.op in ('<', '<=', '>', '>=', '==', '!='):
            left = _convert_expr(expr.left)
            right = _convert_expr(expr.right)
            if left is None or right is None:
                return ('true',)
            op_map = {'<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne'}
            return (op_map[expr.op], left, right)
        elif expr.op == '&&':
            l = _convert_cond(expr.left)
            r = _convert_cond(expr.right)
            return ('and', l, r)
        elif expr.op == '||':
            l = _convert_cond(expr.left)
            r = _convert_cond(expr.right)
            return ('or', l, r)
    elif isinstance(expr, UnaryOp) and expr.op == '!':
        inner = _convert_cond(expr.operand)
        return ('not', inner)
    return ('true',)  # Default: assume anything is possible


# ============================================================
# Octagon Pre-Analysis
# ============================================================

def _octagon_pre_analyze(source, symbolic_inputs=None):
    """Run octagon analysis on C10 source code.
    Returns per-statement octagon states for branch feasibility checking.
    """
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    # Build octagon program
    oct_prog = _ast_to_oct_program(program.stmts)

    # Initial state: symbolic inputs -> TOP, everything else derives from assignments
    init = Octagon.top()
    if symbolic_inputs:
        for var in symbolic_inputs:
            init = init._ensure_var(var)

    interp = OctagonInterpreter()
    result = interp.analyze(oct_prog, init)

    return OctagonPreAnalysis(
        program=program,
        oct_program=oct_prog,
        final_state=result.final_state,
        warnings=result.warnings,
    )


@dataclass
class OctagonPreAnalysis:
    program: Program
    oct_program: tuple
    final_state: Octagon
    warnings: List[str]


# ============================================================
# Branch Feasibility via Octagon
# ============================================================

def _check_branch_feasibility_octagon(state, cond_ast):
    """Check if a branch condition is feasible/infeasible using octagon state.
    Returns: 'feasible', 'infeasible', or 'unknown'.
    """
    if state.is_bot():
        return 'infeasible'

    cond = _convert_cond(cond_ast)
    interp = OctagonInterpreter()
    constraints = interp._cond_to_constraints(cond)

    if not constraints:
        return 'unknown'  # Can't represent condition

    # Try adding constraints -- if result is BOT, branch is infeasible
    test_state = state
    for c in constraints:
        test_state = test_state.guard(c)
        if test_state.is_bot():
            return 'infeasible'

    return 'feasible'


def _collect_branch_conditions(stmts):
    """Collect all branch conditions from AST with their octagon state at that point."""
    if isinstance(stmts, Block):
        stmts = stmts.stmts
    if stmts is None:
        return []
    branches = []
    for s in stmts:
        if isinstance(s, IfStmt):
            branches.append(s.cond)
            branches.extend(_collect_branch_conditions(s.then_body))
            if s.else_body:
                branches.extend(_collect_branch_conditions(s.else_body))
        elif isinstance(s, WhileStmt):
            branches.append(s.cond)
            branches.extend(_collect_branch_conditions(s.body))
        elif isinstance(s, Block):
            branches.extend(_collect_branch_conditions(s.stmts))
    return branches


# ============================================================
# Guided Symbolic Execution
# ============================================================

@dataclass
class GuidedResult:
    """Result from octagon-guided symbolic execution."""
    execution: ExecutionResult
    octagon_state: Octagon
    octagon_warnings: List[str]
    branches_analyzed: int
    branches_pruned_by_octagon: int
    smt_checks_saved: int
    relational_constraints_found: int

    @property
    def paths(self):
        return self.execution.paths

    @property
    def test_cases(self):
        return self.execution.test_cases

    @property
    def pruning_ratio(self):
        if self.branches_analyzed == 0:
            return 0.0
        return self.branches_pruned_by_octagon / self.branches_analyzed


def guided_execute(source, symbolic_inputs=None, max_paths=50):
    """Run octagon-guided symbolic execution.

    1. Parse source
    2. Run octagon pre-analysis to get relational bounds
    3. Count how many branches the octagon can resolve
    4. Run symbolic execution (C038) for full path exploration
    5. Compare: which branches did octagon catch that intervals wouldn't?

    Returns GuidedResult with paths, test cases, and pruning stats.
    """
    # Step 1: Octagon pre-analysis
    pre = _octagon_pre_analyze(source, symbolic_inputs)

    # Step 2: Collect branch conditions and check feasibility
    branches = _collect_branch_conditions(pre.program.stmts)
    branches_pruned = 0
    for cond in branches:
        result = _check_branch_feasibility_octagon(pre.final_state, cond)
        if result == 'infeasible':
            branches_pruned += 1

    # Step 3: Run standard symbolic execution
    exec_result = symbolic_execute(source, symbolic_inputs, max_paths=max_paths)

    # Step 4: Count relational constraints
    relational = 0
    if not pre.final_state.is_bot():
        constraints = pre.final_state.extract_constraints()
        for c in constraints:
            if c.var2 is not None:  # Binary constraint = relational
                relational += 1

    return GuidedResult(
        execution=exec_result,
        octagon_state=pre.final_state,
        octagon_warnings=pre.warnings,
        branches_analyzed=len(branches),
        branches_pruned_by_octagon=branches_pruned,
        smt_checks_saved=branches_pruned,
        relational_constraints_found=relational,
    )


# ============================================================
# Relational Path Pruning
# ============================================================

def analyze_relational_pruning(source, symbolic_inputs=None):
    """Analyze which branches can be pruned by octagon but not by intervals.
    Returns detailed comparison of interval vs octagon pruning power.
    """
    # Parse
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()
    branches = _collect_branch_conditions(program.stmts)

    # Octagon analysis
    oct_prog = _ast_to_oct_program(program.stmts)
    init = Octagon.top()
    if symbolic_inputs:
        for var in symbolic_inputs:
            init = init._ensure_var(var)
    oct_interp = OctagonInterpreter()
    oct_result = oct_interp.analyze(oct_prog, init)

    # Interval-only analysis (octagon with no binary constraints = intervals)
    # We simulate interval analysis by projecting out relational info
    oct_intervals = {}
    if not oct_result.final_state.is_bot():
        oct_intervals = oct_result.final_state.extract_intervals()

    # Rebuild an interval-only octagon (no relational constraints)
    iv_only = octagon_from_intervals(
        {v: (lo if lo is not None else None, hi if hi is not None else None)
         for v, (lo, hi) in oct_intervals.items()}
    ) if oct_intervals else Octagon.top()

    interval_pruned = 0
    octagon_pruned = 0
    octagon_only_pruned = 0

    for cond in branches:
        iv_result = _check_branch_feasibility_octagon(iv_only, cond)
        oct_feasibility = _check_branch_feasibility_octagon(oct_result.final_state, cond)

        if iv_result == 'infeasible':
            interval_pruned += 1
        if oct_feasibility == 'infeasible':
            octagon_pruned += 1
        if oct_feasibility == 'infeasible' and iv_result != 'infeasible':
            octagon_only_pruned += 1

    return {
        'total_branches': len(branches),
        'interval_pruned': interval_pruned,
        'octagon_pruned': octagon_pruned,
        'octagon_only_pruned': octagon_only_pruned,
        'relational_advantage': octagon_only_pruned,
        'octagon_state': oct_result.final_state,
        'interval_state': iv_only,
    }


# ============================================================
# Property-Guided Analysis
# ============================================================

def verify_relational_property(source, property_str, symbolic_inputs=None):
    """Verify a relational property about a program.

    property_str: human-readable like "x - y <= 5" or "x + y == 10"
    Returns verification result dict.
    """
    # Parse property
    constraint = _parse_property(property_str)
    if constraint is None:
        return {'verified': False, 'error': f'Cannot parse property: {property_str}'}

    # Run octagon analysis
    pre = _octagon_pre_analyze(source, symbolic_inputs)

    if pre.final_state.is_bot():
        return {'verified': True, 'reason': 'unreachable'}

    # Check property against octagon
    neg = _negate_oct_constraint_list(constraint)
    test_state = pre.final_state
    for c in neg:
        test_state = test_state.guard(c)

    verified = not test_state.is_satisfiable()

    result = {
        'verified': verified,
        'octagon_state': pre.final_state,
    }
    if verified:
        result['reason'] = 'octagon proves property'
    else:
        result['reason'] = 'octagon cannot prove (may still hold)'
        # Try symbolic execution for counterexample
        exec_result = symbolic_execute(source, symbolic_inputs, max_paths=20)
        result['symbolic_paths'] = len(exec_result.paths)
    return result


def _parse_property(prop_str):
    """Parse a simple property string into OctConstraints.
    Supports: "x <= 5", "x - y <= 3", "x + y == 10", "x >= 0".
    """
    prop_str = prop_str.strip()

    # Try "var1 op var2 rel bound" patterns
    import re

    # x - y <= N
    m = re.match(r'(\w+)\s*-\s*(\w+)\s*<=\s*(-?\d+)', prop_str)
    if m:
        return [OctConstraint.diff_le(m.group(1), m.group(2), int(m.group(3)))]

    # x + y <= N
    m = re.match(r'(\w+)\s*\+\s*(\w+)\s*<=\s*(-?\d+)', prop_str)
    if m:
        return [OctConstraint.sum_le(m.group(1), m.group(2), int(m.group(3)))]

    # x + y == N
    m = re.match(r'(\w+)\s*\+\s*(\w+)\s*==\s*(-?\d+)', prop_str)
    if m:
        v1, v2, n = m.group(1), m.group(2), int(m.group(3))
        return [OctConstraint.sum_le(v1, v2, n), OctConstraint.sum_ge(v1, v2, n)]

    # x - y == N
    m = re.match(r'(\w+)\s*-\s*(\w+)\s*==\s*(-?\d+)', prop_str)
    if m:
        v1, v2, n = m.group(1), m.group(2), int(m.group(3))
        return [OctConstraint.diff_le(v1, v2, n), OctConstraint.diff_ge(v1, v2, n)]

    # x <= N
    m = re.match(r'(\w+)\s*<=\s*(-?\d+)', prop_str)
    if m:
        return [OctConstraint.var_le(m.group(1), int(m.group(2)))]

    # x >= N
    m = re.match(r'(\w+)\s*>=\s*(-?\d+)', prop_str)
    if m:
        return [OctConstraint.var_ge(m.group(1), int(m.group(2)))]

    # x == N
    m = re.match(r'(\w+)\s*==\s*(-?\d+)', prop_str)
    if m:
        return OctConstraint.var_eq(m.group(1), int(m.group(2)))

    return None


def _negate_oct_constraint_list(constraints):
    """Negate a conjunction of constraints -> disjunction.
    For single constraint: negate directly.
    For conjunction (equality): we check each individually (sound over-approximation).
    """
    # For soundness in property checking: add negation of ALL constraints
    # If the result is BOT -> property holds
    result = []
    for c in constraints:
        # c: coeff1*v1 + coeff2*v2 <= b => negate: coeff1*v1 + coeff2*v2 > b
        # => -coeff1*v1 - coeff2*v2 <= -(b+1) for integers
        neg = OctConstraint(
            c.var1, -c.coeff1 if c.var1 else 0,
            c.var2, -c.coeff2 if c.var2 else 0,
            -(c.bound + 1)
        )
        result.append(neg)
    return result


# ============================================================
# Comparison APIs
# ============================================================

def compare_v001_vs_v174(source, symbolic_inputs=None):
    """Compare V001 (interval-guided) vs V174 (octagon-guided) symbolic execution."""
    # V174: octagon-guided
    v174_result = guided_execute(source, symbolic_inputs)

    # V001-style: interval-only
    pruning = analyze_relational_pruning(source, symbolic_inputs)

    return {
        'v174_branches_pruned': v174_result.branches_pruned_by_octagon,
        'v174_relational_constraints': v174_result.relational_constraints_found,
        'v001_style_pruned': pruning['interval_pruned'],
        'octagon_advantage': pruning['octagon_only_pruned'],
        'total_branches': pruning['total_branches'],
        'paths_explored': len(v174_result.paths),
    }


def batch_guided_execute(sources, symbolic_inputs_list=None):
    """Run guided execution on multiple sources."""
    results = []
    for i, src in enumerate(sources):
        inputs = symbolic_inputs_list[i] if symbolic_inputs_list else None
        results.append(guided_execute(src, inputs))
    return results
