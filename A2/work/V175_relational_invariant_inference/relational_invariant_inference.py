"""
V175: Relational Invariant Inference

Composes V173 (octagon abstract domain) + V007 (loop invariant inference) +
V002 (transition systems + PDR) + C037 (SMT solver) + C010 (parser).

Key contribution: V007's relational templates only find sum/diff conservation.
V173's octagon fixpoint discovers arbitrary octagonal relations from loops,
which we then validate as inductive invariants using SMT.

Pipeline:
1. Parse source -> extract while loops -> build transition systems
2. Run V173 octagon analysis on loop body -> extract relational constraints
3. Convert octagonal constraints to SMT formulas
4. Validate each as inductive: Init => Inv AND (Inv AND cond AND Trans => Inv')
5. Optionally compose with V007's non-relational candidates
6. Check sufficiency for postcondition if provided
"""

import sys, os
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Dict, Tuple, Set, Any

# --- Imports from composed systems ---

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import Parser, lex, LetDecl, Assign, WhileStmt, IfStmt, Block, BinOp, IntLit, Var as ASTVar, CallExpr, ReturnStmt, FnDecl, Program, UnaryOp, BoolLit

def parse(source):
    """Parse C10 source code into AST."""
    tokens = lex(source)
    return Parser(tokens).parse()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
from octagon import (
    Octagon, OctConstraint, OctExpr, OctagonInterpreter,
    analyze_program as octagon_analyze, OctAnalysisResult,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult,
    Var as SMTVar, IntConst, BoolConst, App, Op,
    BOOL, INT, Sort, Term,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
from pdr import TransitionSystem


# ============================================================
# Data structures
# ============================================================

class InvariantMethod(Enum):
    OCTAGON_BOUND = auto()       # Unary bound from octagon: x <= c, x >= c
    OCTAGON_DIFF = auto()        # Difference bound: x - y <= c
    OCTAGON_SUM = auto()         # Sum bound: x + y <= c
    OCTAGON_EQUALITY = auto()    # Exact relation: x - y == c, x + y == c, x == c
    V007_ABSTRACT_INTERP = auto()  # From V007's abstract interpretation tier
    V007_INIT_BOUND = auto()     # From V007's init-value bounds
    V007_CONDITION = auto()      # From V007's condition weakening
    V007_RELATIONAL = auto()     # From V007's relational templates
    V007_PDR = auto()            # From V007's PDR discovery


@dataclass
class RelationalInvariant:
    """A discovered invariant with provenance."""
    description: str           # Human-readable
    smt_formula: Term          # SMT formula
    method: InvariantMethod    # How it was discovered
    is_inductive: bool = False # Validated as inductive?
    octagon_constraint: Optional[OctConstraint] = None  # Source constraint if from octagon


@dataclass
class RelationalInferenceResult:
    """Result of relational invariant inference."""
    invariants: List[RelationalInvariant]
    octagon_candidates: int = 0   # How many candidates octagon produced
    validated_count: int = 0      # How many passed inductive check
    v007_candidates: int = 0      # How many V007 contributed
    sufficient: bool = False      # Do they suffice for postcondition?
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def inductive_invariants(self) -> List[RelationalInvariant]:
        return [inv for inv in self.invariants if inv.is_inductive]

    @property
    def relational_invariants(self) -> List[RelationalInvariant]:
        """Only the truly relational ones (two variables)."""
        rel_methods = {InvariantMethod.OCTAGON_DIFF, InvariantMethod.OCTAGON_SUM,
                       InvariantMethod.OCTAGON_EQUALITY, InvariantMethod.V007_RELATIONAL}
        return [inv for inv in self.invariants if inv.is_inductive and inv.method in rel_methods]


# ============================================================
# AST -> Octagon program translation
# ============================================================

def _ast_to_oct_expr(node) -> Optional[tuple]:
    """Convert C10 AST expression to octagon program expression tuple."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return ('const', node.value)
    if cls == 'Var':
        return ('var', node.name)
    if cls == 'UnaryOp':
        if node.op == '-':
            inner = _ast_to_oct_expr(node.operand)
            if inner:
                return ('neg', inner)
        return None
    if cls == 'BinOp':
        left = _ast_to_oct_expr(node.left)
        right = _ast_to_oct_expr(node.right)
        if left and right:
            op_map = {'+': 'add', '-': 'sub', '*': 'mul'}
            if node.op in op_map:
                return (op_map[node.op], left, right)
    return None


def _ast_to_oct_cond(node) -> Optional[tuple]:
    """Convert C10 AST condition to octagon condition tuple.

    C10 uses BinOp for both arithmetic and comparisons.
    Comparison ops: <, <=, >, >=, ==, !=
    Logical ops: &&, ||
    """
    cls = node.__class__.__name__
    if cls == 'BoolLit':
        return ('true',) if node.value else ('false',)
    if cls == 'BinOp':
        cmp_ops = {'<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge', '==': 'eq', '!=': 'ne'}
        if node.op in cmp_ops:
            left = _ast_to_oct_expr(node.left)
            right = _ast_to_oct_expr(node.right)
            if left and right:
                return (cmp_ops[node.op], left, right)
        if node.op == '&&':
            l = _ast_to_oct_cond(node.left)
            r = _ast_to_oct_cond(node.right)
            if l and r:
                return ('and', l, r)
        if node.op == '||':
            l = _ast_to_oct_cond(node.left)
            r = _ast_to_oct_cond(node.right)
            if l and r:
                return ('or', l, r)
    if cls == 'UnaryOp' and node.op == '!':
        inner = _ast_to_oct_cond(node.operand)
        if inner:
            return ('not', inner)
    return None


def _ast_to_oct_stmt(node) -> Optional[tuple]:
    """Convert C10 AST statement to octagon program statement tuple."""
    cls = node.__class__.__name__
    if cls in ('LetDecl', 'Assign'):
        name = node.name if hasattr(node, 'name') else None
        if name and hasattr(node, 'value') and node.value is not None:
            expr = _ast_to_oct_expr(node.value)
            if expr:
                return ('assign', name, expr)
            # Non-octagonal expression: havoc (assign TOP)
            return ('assign', name, ('var', '__top__'))
        return ('skip',)
    if cls == 'WhileStmt':
        cond = _ast_to_oct_cond(node.cond)
        body = _ast_to_oct_body(node.body)
        if cond and body:
            return ('while', cond, body)
        return ('skip',)
    if cls == 'IfStmt':
        cond = _ast_to_oct_cond(node.cond)
        then_body = _ast_to_oct_body(node.then_body)
        else_body = _ast_to_oct_body(node.else_body) if node.else_body else ('skip',)
        if cond and then_body:
            return ('if', cond, then_body, else_body)
        return ('skip',)
    if cls == 'Block':
        return _ast_to_oct_stmts(node.stmts)
    if cls == 'ExprStmt':
        return ('skip',)
    if cls == 'ReturnStmt':
        return ('skip',)
    return ('skip',)


def _ast_to_oct_body(node) -> Optional[tuple]:
    """Convert Block or statement list to octagon program."""
    if isinstance(node, Block):
        return _ast_to_oct_stmts(node.stmts)
    if isinstance(node, list):
        return _ast_to_oct_stmts(node)
    return _ast_to_oct_stmt(node)


def _ast_to_oct_stmts(stmts) -> tuple:
    """Convert statement list to octagon seq tuple."""
    parts = []
    for s in stmts:
        p = _ast_to_oct_stmt(s)
        if p and p != ('skip',):
            parts.append(p)
    if not parts:
        return ('skip',)
    if len(parts) == 1:
        return parts[0]
    return ('seq',) + tuple(parts)


# ============================================================
# Loop extraction
# ============================================================

def _find_while_loops(stmts) -> List[Tuple[int, Any, List[Any]]]:
    """Find while loops in a statement list. Returns [(index, while_node, pre_stmts)]."""
    results = []
    pre = []
    for i, s in enumerate(stmts):
        cls = s.__class__.__name__
        if cls == 'WhileStmt':
            results.append((i, s, list(pre)))
        pre.append(s)
    return results


def _extract_loop_vars(while_node) -> Set[str]:
    """Extract variables modified in a while loop body."""
    modified = set()
    _collect_modified(while_node.body, modified)
    return modified


def _collect_modified(node, modified: set):
    """Collect all variables assigned in a subtree."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls in ('LetDecl', 'Assign'):
        if hasattr(node, 'name') and node.name:
            modified.add(node.name)
    if cls == 'Block':
        for s in node.stmts:
            _collect_modified(s, modified)
    if cls == 'IfStmt':
        _collect_modified(node.then_body, modified)
        if node.else_body:
            _collect_modified(node.else_body, modified)
    if cls == 'WhileStmt':
        _collect_modified(node.body, modified)


def _extract_condition_vars(node) -> Set[str]:
    """Extract variable names from a condition AST."""
    result = set()
    cls = node.__class__.__name__
    if cls == 'Var':
        result.add(node.name)
    for attr in ('left', 'right', 'operand', 'cond'):
        child = getattr(node, attr, None)
        if child:
            result.update(_extract_condition_vars(child))
    return result


def _extract_pre_assignments(pre_stmts) -> Dict[str, int]:
    """Extract initial variable values from pre-loop statements."""
    init_vals = {}
    for s in pre_stmts:
        cls = s.__class__.__name__
        if cls in ('LetDecl', 'Assign'):
            if hasattr(s, 'value') and s.value is not None:
                val_cls = s.value.__class__.__name__
                if val_cls == 'IntLit':
                    init_vals[s.name] = s.value.value
                elif val_cls == 'UnaryOp' and s.value.op == '-':
                    inner = s.value.operand
                    if inner.__class__.__name__ == 'IntLit':
                        init_vals[s.name] = -inner.value
    return init_vals


# ============================================================
# Octagon -> SMT conversion
# ============================================================

def _oct_constraint_to_smt(c: OctConstraint, var_map: Dict[str, Term]) -> Optional[Term]:
    """Convert an OctConstraint to an SMT formula."""
    bound = int(c.bound) if c.bound == int(c.bound) else None
    if bound is None:
        return None  # Can't represent non-integer bounds in LIA

    if c.var2 is None:
        # Unary: coeff1 * var1 <= bound
        v = var_map.get(c.var1)
        if v is None:
            return None
        if c.coeff1 == 1:
            # var1 <= bound
            return App(Op.LE, [v, IntConst(bound)], BOOL)
        else:
            # -var1 <= bound  =>  var1 >= -bound
            return App(Op.GE, [v, IntConst(-bound)], BOOL)
    else:
        # Binary: coeff1 * var1 + coeff2 * var2 <= bound
        v1 = var_map.get(c.var1)
        v2 = var_map.get(c.var2)
        if v1 is None or v2 is None:
            return None
        # Build LHS expression
        if c.coeff1 == 1 and c.coeff2 == 1:
            # var1 + var2 <= bound
            lhs = App(Op.ADD, [v1, v2], INT)
        elif c.coeff1 == 1 and c.coeff2 == -1:
            # var1 - var2 <= bound
            lhs = App(Op.SUB, [v1, v2], INT)
        elif c.coeff1 == -1 and c.coeff2 == 1:
            # -var1 + var2 <= bound  =>  var2 - var1 <= bound
            lhs = App(Op.SUB, [v2, v1], INT)
        elif c.coeff1 == -1 and c.coeff2 == -1:
            # -(var1 + var2) <= bound  =>  var1 + var2 >= -bound
            lhs = App(Op.ADD, [v1, v2], INT)
            return App(Op.GE, [lhs, IntConst(-bound)], BOOL)
        else:
            return None
        return App(Op.LE, [lhs, IntConst(bound)], BOOL)


def _oct_constraint_to_description(c: OctConstraint) -> str:
    """Human-readable description of an octagonal constraint."""
    bound = int(c.bound) if c.bound == int(c.bound) else float(c.bound)
    if c.var2 is None:
        if c.coeff1 == 1:
            return f"{c.var1} <= {bound}"
        else:
            return f"{c.var1} >= {-bound}"
    else:
        if c.coeff1 == 1 and c.coeff2 == -1:
            return f"{c.var1} - {c.var2} <= {bound}"
        elif c.coeff1 == -1 and c.coeff2 == 1:
            return f"{c.var2} - {c.var1} <= {bound}"
        elif c.coeff1 == 1 and c.coeff2 == 1:
            return f"{c.var1} + {c.var2} <= {bound}"
        elif c.coeff1 == -1 and c.coeff2 == -1:
            return f"{c.var1} + {c.var2} >= {-bound}"
        return f"{c.coeff1}*{c.var1} + {c.coeff2}*{c.var2} <= {bound}"


def _equality_description(c: OctConstraint) -> str:
    """Generate equality description from a constraint (for complementary pair)."""
    bound = int(c.bound) if c.bound == int(c.bound) else float(c.bound)
    if c.var2 is None:
        if c.coeff1 == 1:
            return f"{c.var1} == {bound}"
        else:
            return f"{c.var1} == {-bound}"
    else:
        if c.coeff1 == 1 and c.coeff2 == -1:
            return f"{c.var1} - {c.var2} == {bound}"
        elif c.coeff1 == -1 and c.coeff2 == 1:
            return f"{c.var2} - {c.var1} == {bound}"
        elif c.coeff1 == 1 and c.coeff2 == 1:
            return f"{c.var1} + {c.var2} == {bound}"
        elif c.coeff1 == -1 and c.coeff2 == -1:
            return f"{c.var1} + {c.var2} == {-bound}"
        return f"{c.coeff1}*{c.var1} + {c.coeff2}*{c.var2} == {bound}"


def _classify_method(c: OctConstraint) -> InvariantMethod:
    """Classify an octagonal constraint by type."""
    if c.var2 is None:
        return InvariantMethod.OCTAGON_BOUND
    elif c.coeff1 == 1 and c.coeff2 == -1:
        return InvariantMethod.OCTAGON_DIFF
    elif c.coeff1 == -1 and c.coeff2 == 1:
        return InvariantMethod.OCTAGON_DIFF
    elif abs(c.coeff1) == 1 and abs(c.coeff2) == 1 and c.coeff1 == c.coeff2:
        return InvariantMethod.OCTAGON_SUM
    return InvariantMethod.OCTAGON_DIFF


# ============================================================
# Transition system construction for validation
# ============================================================

def _build_transition_system(while_node, pre_stmts, state_vars: Set[str]) -> Optional[TransitionSystem]:
    """Build a TransitionSystem from a while loop for invariant validation."""
    init_vals = _extract_pre_assignments(pre_stmts)

    ts = TransitionSystem()
    var_map = {}
    for v in sorted(state_vars):
        var_map[v] = ts.add_int_var(v)

    # Init: from pre-assignments
    init_parts = []
    for v in sorted(state_vars):
        if v in init_vals:
            init_parts.append(App(Op.EQ, [ts.var(v), IntConst(init_vals[v])], BOOL))
    if init_parts:
        init_f = init_parts[0]
        for p in init_parts[1:]:
            init_f = App(Op.AND, [init_f, p], BOOL)
        ts.set_init(init_f)
    else:
        ts.set_init(BoolConst(True))

    # Transition: from loop body
    trans = _build_body_transition(while_node.body, ts, state_vars)
    cond = _build_cond_smt(while_node.cond, ts)
    if trans and cond:
        # Guarded transition: (cond AND body_trans) OR (!cond AND frame)
        frame_parts = []
        for v in sorted(state_vars):
            frame_parts.append(App(Op.EQ, [ts.prime(v), ts.var(v)], BOOL))
        frame = frame_parts[0]
        for p in frame_parts[1:]:
            frame = App(Op.AND, [frame, p], BOOL)
        neg_cond = _smt_negate(cond)
        guarded_trans = App(Op.OR, [
            App(Op.AND, [cond, trans], BOOL),
            App(Op.AND, [neg_cond, frame], BOOL)
        ], BOOL)
        ts.set_trans(guarded_trans)
    else:
        return None

    return ts, var_map


def _build_cond_smt(node, ts) -> Optional[Term]:
    """Convert C10 condition AST to SMT formula over TS variables.
    C10 uses BinOp for comparisons and logical operators."""
    cls = node.__class__.__name__
    if cls == 'BinOp':
        cmp_ops = {'<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE, '==': Op.EQ, '!=': Op.NEQ}
        if node.op in cmp_ops:
            left = _build_expr_smt(node.left, ts)
            right = _build_expr_smt(node.right, ts)
            if left is None or right is None:
                return None
            return App(cmp_ops[node.op], [left, right], BOOL)
        if node.op == '&&':
            l = _build_cond_smt(node.left, ts)
            r = _build_cond_smt(node.right, ts)
            if l and r:
                return App(Op.AND, [l, r], BOOL)
        if node.op == '||':
            l = _build_cond_smt(node.left, ts)
            r = _build_cond_smt(node.right, ts)
            if l and r:
                return App(Op.OR, [l, r], BOOL)
    if cls == 'UnaryOp' and node.op == '!':
        inner = _build_cond_smt(node.operand, ts)
        if inner:
            return _smt_negate(inner)
    if cls == 'BoolLit':
        return BoolConst(node.value)
    if cls == 'Var':
        try:
            v = ts.var(node.name)
            return App(Op.NEQ, [v, IntConst(0)], BOOL)
        except:
            return None
    return None


def _build_expr_smt(node, ts) -> Optional[Term]:
    """Convert C10 expression AST to SMT term over TS variables."""
    cls = node.__class__.__name__
    if cls == 'IntLit':
        return IntConst(node.value)
    if cls == 'Var':
        try:
            return ts.var(node.name)
        except:
            return None
    if cls == 'BinOp':
        left = _build_expr_smt(node.left, ts)
        right = _build_expr_smt(node.right, ts)
        if left is None or right is None:
            return None
        op_map = {'+': Op.ADD, '-': Op.SUB}
        if node.op in op_map:
            return App(op_map[node.op], [left, right], INT)
        if node.op == '*':
            # Only handle constant multiplication
            if cls == 'IntLit':
                pass  # handled above
            return App(Op.MUL, [left, right], INT) if hasattr(Op, 'MUL') else None
    if cls == 'UnaryOp' and node.op == '-':
        inner = _build_expr_smt(node.operand, ts)
        if inner:
            return App(Op.SUB, [IntConst(0), inner], INT)
    return None


def _build_body_transition(body_node, ts, state_vars: Set[str]) -> Optional[Term]:
    """Build transition relation from loop body."""
    # Collect assignments: last assignment to each variable wins
    assignments = {}
    _collect_assignments(body_node, assignments, ts)

    parts = []
    for v in sorted(state_vars):
        if v in assignments:
            parts.append(App(Op.EQ, [ts.prime(v), assignments[v]], BOOL))
        else:
            # Frame: v' = v
            parts.append(App(Op.EQ, [ts.prime(v), ts.var(v)], BOOL))

    if not parts:
        return None
    result = parts[0]
    for p in parts[1:]:
        result = App(Op.AND, [result, p], BOOL)
    return result


def _collect_assignments(node, assignments: dict, ts):
    """Collect last assignments from body (simple: no branching within body for now)."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls == 'Block':
        for s in node.stmts:
            _collect_assignments(s, assignments, ts)
    elif cls in ('LetDecl', 'Assign'):
        name = node.name if hasattr(node, 'name') else None
        if name and hasattr(node, 'value') and node.value is not None:
            expr = _build_expr_smt(node.value, ts)
            if expr:
                assignments[name] = expr
    elif cls == 'IfStmt':
        # For if-then-else, we build ITE expressions for assigned vars
        then_assigns = {}
        _collect_assignments(node.then_body, then_assigns, ts)
        else_assigns = {}
        if node.else_body:
            _collect_assignments(node.else_body, else_assigns, ts)
        cond = _build_cond_smt(node.cond, ts)
        if cond:
            all_vars = set(then_assigns.keys()) | set(else_assigns.keys())
            for v in all_vars:
                then_val = then_assigns.get(v, ts.var(v))
                else_val = else_assigns.get(v, ts.var(v))
                # ITE: if cond then then_val else else_val
                # Encode as: (cond AND v'=then) OR (!cond AND v'=else)
                # Since we're building a single expression for v', use conditional
                # We store the ITE as a special construct
                assignments[v] = _make_ite(cond, then_val, else_val)


def _make_ite(cond, then_val, else_val) -> Term:
    """Build ITE as: (cond => v=then) AND (!cond => v=else) factored into expression."""
    # For SMT, we just store the conditional assignment
    # The validation will use: v' = ITE(cond, then, else)
    # Encode using ITE if available, otherwise use conjunction form
    return _ITEExpr(cond, then_val, else_val)


class _ITEExpr:
    """Helper to represent ITE in transition."""
    def __init__(self, cond, then_val, else_val):
        self.cond = cond
        self.then_val = then_val
        self.else_val = else_val


def _smt_negate(term: Term) -> Term:
    """Negate an SMT boolean term using complement operators."""
    if isinstance(term, BoolConst):
        return BoolConst(not term.value)
    if isinstance(term, App):
        # Use complement operators (V002 bug: NOT(EQ) doesn't work)
        complements = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if term.op in complements:
            return App(complements[term.op], term.args, BOOL)
        if term.op == Op.AND:
            return App(Op.OR, [_smt_negate(a) for a in term.args], BOOL)
        if term.op == Op.OR:
            return App(Op.AND, [_smt_negate(a) for a in term.args], BOOL)
        if term.op == Op.NOT:
            return term.args[0]
    return App(Op.NOT, [term], BOOL)


# ============================================================
# Inductiveness validation
# ============================================================

def _substitute_prime(formula: Term, var_map: Dict[str, Term], ts) -> Term:
    """Replace each variable v in formula with its primed version v'."""
    if isinstance(formula, SMTVar):
        if formula.name in var_map:
            return ts.prime(formula.name)
        return formula
    if isinstance(formula, (IntConst, BoolConst)):
        return formula
    if isinstance(formula, App):
        new_args = [_substitute_prime(a, var_map, ts) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    return formula


def _validate_inductive(inv_formula: Term, ts_result, var_map: Dict[str, Term]) -> bool:
    """Check if inv_formula is inductive for the transition system.

    Checks:
    1. Init => Inv  (invariant holds initially)
    2. Inv AND Trans => Inv'  (invariant is preserved)
    """
    ts, vm = ts_result

    # Check 1: Init => Inv
    # Negate: Init AND NOT(Inv) should be UNSAT
    solver = SMTSolver()
    for name in sorted(var_map.keys()):
        solver.Int(name)
    neg_inv = _smt_negate(inv_formula)
    solver.add(ts.init_formula)
    solver.add(neg_inv)
    result = solver.check()
    if result != SMTResult.UNSAT:
        return False

    # Check 2: Inv AND Trans => Inv'
    # We need to handle ITE in transitions specially
    solver2 = SMTSolver()
    for name in sorted(var_map.keys()):
        solver2.Int(name)
        solver2.Int(name + "'")

    solver2.add(inv_formula)
    # Add transition - handle ITE expressions
    trans = ts.trans_formula
    if trans:
        trans_clean = _expand_ite_in_formula(trans, ts)
        solver2.add(trans_clean)

    # Negate Inv' (primed)
    inv_prime = _substitute_prime(inv_formula, var_map, ts)
    neg_inv_prime = _smt_negate(inv_prime)
    solver2.add(neg_inv_prime)

    result2 = solver2.check()
    return result2 == SMTResult.UNSAT


def _expand_ite_in_formula(formula, ts) -> Term:
    """Expand ITE expressions in transition formula to proper SMT."""
    if isinstance(formula, App):
        if formula.op == Op.EQ and len(formula.args) == 2:
            lhs, rhs = formula.args
            if isinstance(rhs, _ITEExpr):
                # v' = ITE(cond, then, else) =>
                # (cond AND v'=then) OR (!cond AND v'=else)
                then_eq = App(Op.EQ, [lhs, rhs.then_val], BOOL)
                else_eq = App(Op.EQ, [lhs, rhs.else_val], BOOL)
                return App(Op.OR, [
                    App(Op.AND, [rhs.cond, then_eq], BOOL),
                    App(Op.AND, [_smt_negate(rhs.cond), else_eq], BOOL)
                ], BOOL)
        new_args = [_expand_ite_in_formula(a, ts) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    return formula


# ============================================================
# Equality detection from constraint pairs
# ============================================================

def _normalize_constraint(c: OctConstraint) -> Tuple:
    """Normalize a constraint to a canonical form: (var1, coeff1, var2, coeff2, bound)
    where var1 <= var2 lexically (for binary), or coeff1 = 1 (for unary)."""
    if c.var2 is None:
        return (c.var1, c.coeff1, None, 0, c.bound)
    return (c.var1, c.coeff1, c.var2, c.coeff2, c.bound)


def _constraints_are_complementary(ci: OctConstraint, cj: OctConstraint) -> bool:
    """Check if ci and cj form complementary pair: P <= b AND -P <= -b => P == b."""
    # Same vars, negated coefficients, negated bound
    if ci.var2 is None and cj.var2 is None:
        if ci.var1 == cj.var1:
            return ci.coeff1 == -cj.coeff1 and ci.bound == -cj.bound
        return False
    if ci.var2 is not None and cj.var2 is not None:
        # Case 1: same var order, negated coefficients
        if (ci.var1 == cj.var1 and ci.var2 == cj.var2 and
            ci.coeff1 == -cj.coeff1 and ci.coeff2 == -cj.coeff2 and
            ci.bound == -cj.bound):
            return True
        # Case 2: swapped var order with matching negation
        if (ci.var1 == cj.var2 and ci.var2 == cj.var1 and
            ci.coeff1 == -cj.coeff2 and ci.coeff2 == -cj.coeff1 and
            ci.bound == -cj.bound):
            return True
    return False


def _detect_equalities(constraints: List[OctConstraint]) -> List[Tuple[OctConstraint, OctConstraint, str]]:
    """Detect pairs of constraints that form equalities."""
    equalities = []
    used = set()

    seen_descs = set()
    for i in range(len(constraints)):
        if i in used:
            continue
        for j in range(i + 1, len(constraints)):
            if j in used:
                continue
            ci, cj = constraints[i], constraints[j]
            if _constraints_are_complementary(ci, cj):
                used.add(i)
                used.add(j)
                # Build equality description
                if ci.var2 is None:
                    val = int(ci.bound) if ci.coeff1 == 1 else int(-ci.bound)
                    desc = f"{ci.var1} == {val}"
                else:
                    # Use whichever constraint has positive leading coeff
                    if ci.coeff1 == 1 or (ci.coeff1 == -1 and ci.coeff2 == 1):
                        ref = ci
                    else:
                        ref = cj
                    desc = _equality_description(ref)
                # Deduplicate: normalize "x + y == 10" and "y + x == 10"
                norm = _normalize_equality_desc(desc)
                if norm not in seen_descs:
                    seen_descs.add(norm)
                    equalities.append((ci, cj, desc))
                else:
                    # Still mark as used so the raw constraints are excluded
                    pass
                break

    return equalities


def _normalize_equality_desc(desc: str) -> str:
    """Normalize equality description for deduplication.
    'x + y == 10' and 'y + x == 10' -> same normalized form.
    'x - y == 5' and 'y - x == -5' -> same normalized form."""
    import re
    m = re.match(r'(\w+)\s*([+\-])\s*(\w+)\s*==\s*(-?\d+)', desc)
    if m:
        v1, op, v2, val = m.groups()
        val = int(val)
        if op == '+':
            # Normalize: sorted vars
            vars_sorted = tuple(sorted([v1, v2]))
            return f"{vars_sorted[0]} + {vars_sorted[1]} == {val}"
        else:  # '-'
            # Normalize: always v1 - v2 with v1 < v2 lexically
            if v1 > v2:
                return f"{v2} - {v1} == {-val}"
            return desc
    # Unary
    return desc


# ============================================================
# Main inference pipeline
# ============================================================

def _run_octagon_on_loop(while_node, pre_stmts, state_vars: Set[str]) -> Optional[Octagon]:
    """Run octagon analysis on a while loop, returning the FIXPOINT state (loop invariant).

    The fixpoint state is the widened state at the loop head BEFORE the exit
    condition is applied. This represents what holds at every loop entry,
    which is exactly a loop invariant.
    """
    # Build octagon program parts
    pre_oct_parts = []
    for s in pre_stmts:
        p = _ast_to_oct_stmt(s)
        if p and p != ('skip',):
            pre_oct_parts.append(p)

    while_oct = _ast_to_oct_stmt(while_node)
    if while_oct == ('skip',):
        return None

    # Run pre-assignments to get initial state
    if pre_oct_parts:
        if len(pre_oct_parts) == 1:
            pre_prog = pre_oct_parts[0]
        else:
            pre_prog = ('seq',) + tuple(pre_oct_parts)
        try:
            pre_result = octagon_analyze(pre_prog)
            init_state = pre_result.final_state
        except Exception:
            init_state = Octagon.top()
    else:
        init_state = Octagon.top()

    # Now manually run while-loop fixpoint to capture the INVARIANT state
    # (before exit condition), not the post-loop state
    interp = OctagonInterpreter(max_iterations=50, widen_delay=2)

    cond_oct = _ast_to_oct_cond(while_node.cond)
    body_oct = _ast_to_oct_body(while_node.body)
    if cond_oct is None or body_oct is None:
        return None

    try:
        guard_cs = interp._cond_to_constraints(cond_oct)
        current = init_state

        for iteration in range(50):
            body_entry = current
            for c in guard_cs:
                body_entry = body_entry.guard(c)
            body_exit = interp._interpret(body_oct, body_entry)

            next_state = current.join(body_exit)
            if iteration >= 2:  # widen_delay
                next_state = current.widen(next_state)

            if next_state.includes(current) and current.includes(next_state):
                break
            current = next_state

        # Return the fixpoint state (the loop invariant)
        return current
    except Exception:
        return None


def _suppress_implied(eq_desc: str, seen: set):
    """Given an equality like 'x + y == 10', suppress all implied LE/GE forms."""
    import re
    m = re.match(r'(\w+)\s*([+\-])\s*(\w+)\s*==\s*(-?\d+)', eq_desc)
    if not m:
        return
    v1, op, v2, val = m.groups()
    val = int(val)
    if op == '+':
        # x + y == val implies x + y <= val, x + y >= val
        seen.add(f"{v1} + {v2} <= {val}")
        seen.add(f"{v1} + {v2} >= {val}")
        seen.add(f"{v2} + {v1} <= {val}")
        seen.add(f"{v2} + {v1} >= {val}")
    elif op == '-':
        # x - y == val implies x - y <= val, x - y >= val, y - x <= -val, y - x >= -val
        seen.add(f"{v1} - {v2} <= {val}")
        seen.add(f"{v1} - {v2} >= {val}")
        seen.add(f"{v2} - {v1} <= {-val}")
        seen.add(f"{v2} - {v1} >= {-val}")


def _extract_octagon_candidates(octagon_state: Octagon, state_vars: Set[str],
                                 var_map: Dict[str, Term]) -> List[RelationalInvariant]:
    """Extract invariant candidates from octagon fixpoint state."""
    if octagon_state is None or octagon_state.is_bot():
        return []

    candidates = []
    constraints = octagon_state.extract_constraints()
    seen_descriptions = set()

    # Detect equalities first
    equalities = _detect_equalities(constraints)
    equality_constraints = set()
    for ci, cj, desc in equalities:
        equality_constraints.add(id(ci))
        equality_constraints.add(id(cj))
        # Build SMT for equality
        if ci.var2 is None:
            v = var_map.get(ci.var1)
            if v:
                val = int(ci.bound) if ci.coeff1 == 1 else int(-ci.bound)
                smt_f = App(Op.EQ, [v, IntConst(val)], BOOL)
                candidates.append(RelationalInvariant(
                    description=desc,
                    smt_formula=smt_f,
                    method=InvariantMethod.OCTAGON_EQUALITY,
                    octagon_constraint=ci,
                ))
                seen_descriptions.add(desc)
        else:
            # Build relational equality SMT -- use ref constraint with positive coeff
            # Pick ref from (ci, cj) with positive leading coeff
            ref = ci if ci.coeff1 == 1 else cj
            v1 = var_map.get(ref.var1)
            v2 = var_map.get(ref.var2)
            if v1 and v2:
                bound = int(ref.bound)
                if ref.coeff1 == 1 and ref.coeff2 == -1:
                    lhs = App(Op.SUB, [v1, v2], INT)
                elif ref.coeff1 == 1 and ref.coeff2 == 1:
                    lhs = App(Op.ADD, [v1, v2], INT)
                elif ref.coeff1 == -1 and ref.coeff2 == 1:
                    lhs = App(Op.SUB, [v2, v1], INT)
                elif ref.coeff1 == -1 and ref.coeff2 == -1:
                    lhs = App(Op.ADD, [v1, v2], INT)
                    bound = -bound
                else:
                    continue
                smt_f = App(Op.EQ, [lhs, IntConst(bound)], BOOL)
                if desc not in seen_descriptions:
                    candidates.append(RelationalInvariant(
                        description=desc,
                        smt_formula=smt_f,
                        method=InvariantMethod.OCTAGON_EQUALITY,
                        octagon_constraint=ref,
                    ))
                    seen_descriptions.add(desc)
                    # Suppress all implied <= and >= forms (both var orderings)
                    for ci_inner in (ci, cj):
                        seen_descriptions.add(_oct_constraint_to_description(ci_inner))
                    # Also suppress with swapped var ordering
                    _suppress_implied(desc, seen_descriptions)

    # Then add non-equality constraints
    for c in constraints:
        if id(c) in equality_constraints:
            continue
        # Filter: only keep constraints involving state vars
        if c.var1 not in state_vars:
            continue
        if c.var2 is not None and c.var2 not in state_vars:
            continue

        desc = _oct_constraint_to_description(c)
        if desc in seen_descriptions:
            continue
        seen_descriptions.add(desc)

        smt_f = _oct_constraint_to_smt(c, var_map)
        if smt_f is None:
            continue

        method = _classify_method(c)
        candidates.append(RelationalInvariant(
            description=desc,
            smt_formula=smt_f,
            method=method,
            octagon_constraint=c,
        ))

    return candidates


def _check_postcondition_sufficiency(invariants: List[RelationalInvariant],
                                      while_node, var_map: Dict[str, Term],
                                      postcondition: Term) -> bool:
    """Check if invariants + !loop_cond imply postcondition."""
    if not invariants:
        return False

    solver = SMTSolver()
    for name in sorted(var_map.keys()):
        solver.Int(name)

    # Add all inductive invariants
    for inv in invariants:
        if inv.is_inductive:
            solver.add(inv.smt_formula)

    # Add negated loop condition (we exited the loop)
    # Build SMT from loop condition
    class _FakeTS:
        def __init__(self, vm):
            self._vm = vm
        def var(self, name):
            return self._vm.get(name)

    fake_ts = _FakeTS(var_map)
    loop_cond_smt = _build_cond_smt(while_node.cond, fake_ts)
    if loop_cond_smt:
        solver.add(_smt_negate(loop_cond_smt))

    # Check: invariants AND !cond => postcondition
    # Negate: invariants AND !cond AND !postcondition should be UNSAT
    solver.add(_smt_negate(postcondition))
    result = solver.check()
    return result == SMTResult.UNSAT


# ============================================================
# Public API
# ============================================================

def infer_relational_invariants(source: str, loop_index: int = 0,
                                 postcondition: Optional[Term] = None) -> RelationalInferenceResult:
    """Infer relational loop invariants using octagon abstract domain.

    Args:
        source: C10 source code containing a while loop
        loop_index: Which while loop to analyze (0-based)
        postcondition: Optional SMT formula to check sufficiency against

    Returns:
        RelationalInferenceResult with discovered invariants
    """
    ast = parse(source)
    stmts = ast.stmts if hasattr(ast, 'stmts') else ast

    # Find while loops
    loops = _find_while_loops(stmts)
    if loop_index >= len(loops):
        return RelationalInferenceResult(invariants=[], stats={'error': 'no loop found'})

    idx, while_node, pre_stmts = loops[loop_index]
    state_vars = _extract_loop_vars(while_node)
    cond_vars = _extract_condition_vars(while_node.cond)
    state_vars |= cond_vars

    # Also include pre-assigned vars that appear in loop
    for s in pre_stmts:
        cls = s.__class__.__name__
        if cls in ('LetDecl', 'Assign') and hasattr(s, 'name') and s.name in state_vars:
            pass  # already there
        elif cls in ('LetDecl',) and hasattr(s, 'name'):
            # Include if used in loop body
            state_vars.add(s.name)

    # Build SMT variable map
    var_map = {}
    for v in sorted(state_vars):
        var_map[v] = SMTVar(v, INT)

    # Step 1: Run octagon analysis
    octagon_state = _run_octagon_on_loop(while_node, pre_stmts, state_vars)

    # Step 2: Extract candidates
    oct_candidates = _extract_octagon_candidates(octagon_state, state_vars, var_map) if octagon_state else []

    # Step 3: Build transition system for validation
    ts_result = _build_transition_system(while_node, pre_stmts, state_vars)

    # Step 4: Validate each candidate
    validated = 0
    if ts_result:
        for cand in oct_candidates:
            try:
                if _validate_inductive(cand.smt_formula, ts_result, var_map):
                    cand.is_inductive = True
                    validated += 1
            except Exception:
                pass  # SMT solver may not handle all formulas

    # Step 5: Check postcondition sufficiency
    sufficient = False
    if postcondition and oct_candidates:
        sufficient = _check_postcondition_sufficiency(oct_candidates, while_node, var_map, postcondition)

    result = RelationalInferenceResult(
        invariants=oct_candidates,
        octagon_candidates=len(oct_candidates),
        validated_count=validated,
        sufficient=sufficient,
        stats={
            'state_vars': sorted(state_vars),
            'octagon_constraints': len(octagon_state.extract_constraints()) if octagon_state else 0,
            'equalities_detected': len([c for c in oct_candidates if c.method == InvariantMethod.OCTAGON_EQUALITY]),
            'relational_candidates': len([c for c in oct_candidates if c.method in
                                          {InvariantMethod.OCTAGON_DIFF, InvariantMethod.OCTAGON_SUM, InvariantMethod.OCTAGON_EQUALITY}]),
        }
    )
    return result


def infer_with_v007(source: str, loop_index: int = 0,
                     postcondition: Optional[Term] = None) -> RelationalInferenceResult:
    """Infer invariants combining octagon analysis with V007's tiered approach.

    Runs octagon first for relational invariants, then V007 for non-relational,
    deduplicating across both sources.
    """
    # Get octagon-based invariants
    result = infer_relational_invariants(source, loop_index, postcondition)

    # Try V007 for additional non-relational invariants
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V007_loop_invariant_inference'))
        from loop_invariant_inference import infer_loop_invariants as v007_infer
        v007_result = v007_infer(source, loop_index=loop_index)

        # Convert V007 invariants and add non-duplicates
        oct_descriptions = {inv.description for inv in result.invariants}
        v007_count = 0
        for v007_inv in v007_result.invariants:
            desc = v007_inv.description
            if desc not in oct_descriptions:
                method_map = {
                    'ABSTRACT_INTERP': InvariantMethod.V007_ABSTRACT_INTERP,
                    'INIT_BOUND': InvariantMethod.V007_INIT_BOUND,
                    'CONDITION_BASED': InvariantMethod.V007_CONDITION,
                    'RELATIONAL_TEMPLATE': InvariantMethod.V007_RELATIONAL,
                    'PDR_DISCOVERY': InvariantMethod.V007_PDR,
                }
                m = method_map.get(v007_inv.method.name, InvariantMethod.V007_ABSTRACT_INTERP)
                ri = RelationalInvariant(
                    description=desc,
                    smt_formula=v007_inv.sexpr,  # V007 uses SExpr, not SMT Term
                    method=m,
                    is_inductive=v007_inv.is_inductive,
                )
                result.invariants.append(ri)
                v007_count += 1
        result.v007_candidates = v007_count
    except Exception:
        pass  # V007 not available or failed

    return result


def verify_relational_property(source: str, property_str: str,
                                loop_index: int = 0) -> Dict[str, Any]:
    """Verify a relational property holds after a loop.

    Args:
        source: C10 source code
        property_str: Property like "x + y == 10", "x - y <= 5", "x >= 0"
        loop_index: Which while loop

    Returns:
        Dict with verified, invariants_found, property, etc.
    """
    ast = parse(source)
    stmts = ast.stmts if hasattr(ast, 'stmts') else ast
    loops = _find_while_loops(stmts)
    if loop_index >= len(loops):
        return {'verified': False, 'error': 'no loop found'}

    idx, while_node, pre_stmts = loops[loop_index]
    state_vars = _extract_loop_vars(while_node)
    state_vars |= _extract_condition_vars(while_node.cond)
    for s in pre_stmts:
        if s.__class__.__name__ == 'LetDecl' and hasattr(s, 'name'):
            state_vars.add(s.name)

    var_map = {}
    for v in sorted(state_vars):
        var_map[v] = SMTVar(v, INT)

    # Parse property string into SMT
    post = _parse_property(property_str, var_map)
    if post is None:
        return {'verified': False, 'error': f'cannot parse property: {property_str}'}

    result = infer_relational_invariants(source, loop_index, postcondition=post)

    return {
        'verified': result.sufficient,
        'property': property_str,
        'invariants_found': len(result.inductive_invariants),
        'relational_invariants': len(result.relational_invariants),
        'invariant_descriptions': [inv.description for inv in result.inductive_invariants],
        'total_candidates': result.octagon_candidates,
    }


def _parse_property(prop_str: str, var_map: Dict[str, Term]) -> Optional[Term]:
    """Parse a simple property string into SMT formula."""
    prop_str = prop_str.strip()

    # Try: "expr op val" patterns
    import re
    # Pattern: var1 op var2 op2 val  (e.g., "x + y == 10", "x - y <= 5")
    m = re.match(r'(\w+)\s*([+\-])\s*(\w+)\s*(==|!=|<=|>=|<|>)\s*(-?\d+)', prop_str)
    if m:
        v1, arith_op, v2, cmp_op, val = m.groups()
        val = int(val)
        lv1, lv2 = var_map.get(v1), var_map.get(v2)
        if lv1 is None or lv2 is None:
            return None
        if arith_op == '+':
            lhs = App(Op.ADD, [lv1, lv2], INT)
        else:
            lhs = App(Op.SUB, [lv1, lv2], INT)
        op_map = {'==': Op.EQ, '!=': Op.NEQ, '<=': Op.LE, '>=': Op.GE, '<': Op.LT, '>': Op.GT}
        return App(op_map[cmp_op], [lhs, IntConst(val)], BOOL)

    # Pattern: var op val  (e.g., "x >= 0", "i == 0")
    m = re.match(r'(\w+)\s*(==|!=|<=|>=|<|>)\s*(-?\d+)', prop_str)
    if m:
        v, cmp_op, val = m.groups()
        val = int(val)
        lv = var_map.get(v)
        if lv is None:
            return None
        op_map = {'==': Op.EQ, '!=': Op.NEQ, '<=': Op.LE, '>=': Op.GE, '<': Op.LT, '>': Op.GT}
        return App(op_map[cmp_op], [lv, IntConst(val)], BOOL)

    return None


def compare_with_v007(source: str, loop_index: int = 0) -> Dict[str, Any]:
    """Compare octagon-based inference with V007's approach.

    Returns dict with counts and unique contributions from each.
    """
    oct_result = infer_relational_invariants(source, loop_index)

    v007_result = None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V007_loop_invariant_inference'))
        from loop_invariant_inference import infer_loop_invariants as v007_infer
        v007_result = v007_infer(source, loop_index=loop_index)
    except Exception:
        pass

    oct_descs = {inv.description for inv in oct_result.inductive_invariants}
    v007_descs = set()
    if v007_result:
        v007_descs = {inv.description for inv in v007_result.invariants if inv.is_inductive}

    return {
        'octagon_total': oct_result.octagon_candidates,
        'octagon_inductive': len(oct_result.inductive_invariants),
        'octagon_relational': len(oct_result.relational_invariants),
        'octagon_unique': oct_descs - v007_descs,
        'v007_total': len(v007_result.invariants) if v007_result else 0,
        'v007_inductive': len(v007_descs),
        'v007_unique': v007_descs - oct_descs,
        'shared': oct_descs & v007_descs,
        'octagon_descriptions': sorted(oct_descs),
        'v007_descriptions': sorted(v007_descs),
    }


def batch_infer(sources: List[str]) -> List[RelationalInferenceResult]:
    """Run relational invariant inference on multiple programs."""
    return [infer_relational_invariants(src) for src in sources]


def invariant_summary(source: str, loop_index: int = 0) -> str:
    """Human-readable summary of discovered invariants."""
    result = infer_relational_invariants(source, loop_index)
    lines = [f"Relational Invariant Inference Summary",
             f"  Candidates from octagon: {result.octagon_candidates}",
             f"  Validated as inductive: {result.validated_count}",
             f"  Relational invariants: {len(result.relational_invariants)}"]

    if result.inductive_invariants:
        lines.append("  Inductive invariants:")
        for inv in result.inductive_invariants:
            lines.append(f"    - {inv.description} [{inv.method.name}]")

    return "\n".join(lines)
