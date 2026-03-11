"""V114: Recursive Predicate Discovery

Automatically discovers predicates for CEGAR-based program verification.
Instead of relying solely on interpolation from spurious counterexamples,
this module proactively generates candidate predicates using multiple strategies:

1. Template instantiation: parameterized predicate templates over program variables
2. Abstract domain projection: extract predicates from interval/zone analysis
3. Interpolation mining: systematic interpolation across program paths
4. Recursive predicate learning: discover inductive predicates for loops
5. Predicate clustering and ranking: score and select the most useful predicates

Composes: V110 (ART/CFG) + V107 (Craig interpolation) + V113 (CPA) +
          V104 (zone domain) + C037 (SMT solver) + C010 (parser)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum
from collections import defaultdict

# --- Path setup for compositions ---
CHALLENGES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges')
A2_WORK = os.path.join(os.path.dirname(__file__), '..')

for p in [
    os.path.join(CHALLENGES, 'C010_stack_vm'),
    os.path.join(CHALLENGES, 'C037_smt_solver'),
    os.path.join(A2_WORK, 'V104_relational_abstract_domains'),
    os.path.join(A2_WORK, 'V107_craig_interpolation'),
    os.path.join(A2_WORK, 'V110_abstract_reachability_tree'),
]:
    ap = os.path.abspath(p)
    if ap not in sys.path:
        sys.path.insert(0, ap)

from stack_vm import lex, Parser
from smt_solver import (
    SMTSolver, Var, IntConst, BoolConst, App, Op, Sort, SortKind,
    SMTResult
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ============================================================
# Data Structures
# ============================================================

class PredicateSource(Enum):
    """How a predicate was discovered."""
    TEMPLATE = "template"
    INTERVAL = "interval"
    ZONE = "zone"
    INTERPOLATION = "interpolation"
    CONDITION = "condition"
    ASSERTION = "assertion"
    INDUCTIVE = "inductive"
    USER = "user"


@dataclass(frozen=True)
class Predicate:
    """A predicate with metadata about its origin and usefulness."""
    term: object       # SMT term
    source: PredicateSource
    location: Optional[int] = None  # CFG node id (None = global)
    score: float = 0.0
    description: str = ""

    def __hash__(self):
        return hash((str(self.term), self.source, self.location))

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return (str(self.term) == str(other.term) and
                self.source == other.source and
                self.location == other.location)


@dataclass
class DiscoveryResult:
    """Result from predicate discovery."""
    predicates: List[Predicate]
    ranked_predicates: List[Predicate]  # sorted by score descending
    source_counts: Dict[str, int]       # count per source
    total_candidates: int
    selected_count: int
    discovery_stats: Dict[str, object]


@dataclass
class InductivePredicate:
    """A predicate proven inductive for a loop."""
    term: object            # SMT formula
    init_holds: bool        # holds at loop entry
    preserved: bool         # preserved by loop body
    strengthening: List[object]  # supporting predicates needed


@dataclass
class TemplateInstance:
    """An instantiated predicate template."""
    template_name: str
    variables: List[str]
    constants: List[int]
    term: object  # SMT term


# ============================================================
# CFG Construction (from V110, simplified)
# ============================================================

class CFGNodeType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    ASSIGN = "assign"
    ASSUME = "assume"
    ASSUME_NOT = "assume_not"
    ASSERT = "assert"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class CFGNode:
    id: int
    ntype: CFGNodeType
    data: object = None
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    line: int = 0


@dataclass
class CFGEdge:
    source: int
    target: int
    edge_type: CFGNodeType
    data: object = None


class CFG:
    def __init__(self):
        self.nodes: Dict[int, CFGNode] = {}
        self.entry: Optional[int] = None
        self.exit: Optional[int] = None
        self._next_id = 0

    def add_node(self, ntype, data=None, line=0):
        nid = self._next_id
        self._next_id += 1
        node = CFGNode(id=nid, ntype=ntype, data=data, line=line)
        self.nodes[nid] = node
        return nid

    def add_edge(self, src, dst):
        if dst not in self.nodes[src].successors:
            self.nodes[src].successors.append(dst)
        if src not in self.nodes[dst].predecessors:
            self.nodes[dst].predecessors.append(src)

    def get_edges(self) -> List[CFGEdge]:
        edges = []
        for nid, node in self.nodes.items():
            for succ in node.successors:
                edges.append(CFGEdge(nid, succ, node.ntype, node.data))
        return edges

    def get_variables(self) -> Set[str]:
        """Extract all variable names mentioned in CFG."""
        variables = set()
        for node in self.nodes.values():
            if node.data:
                variables.update(_extract_vars_from_data(node.data))
        return variables

    def get_loop_headers(self) -> List[int]:
        """Find loop headers (nodes that are targets of back edges)."""
        headers = []
        visited = set()
        in_stack = set()

        def dfs(nid):
            visited.add(nid)
            in_stack.add(nid)
            for succ in self.nodes[nid].successors:
                if succ in in_stack:
                    if succ not in headers:
                        headers.append(succ)
                elif succ not in visited:
                    dfs(succ)
            in_stack.discard(nid)

        if self.entry is not None:
            dfs(self.entry)
        return headers


def _extract_vars_from_data(data) -> Set[str]:
    """Extract variable names from CFG node data."""
    if data is None:
        return set()
    if isinstance(data, dict):
        vs = set()
        if 'var' in data:
            vs.add(data['var'])
        if 'expr' in data:
            vs.update(_extract_vars_from_ast(data['expr']))
        if 'cond' in data:
            vs.update(_extract_vars_from_ast(data['cond']))
        return vs
    return set()


def _extract_vars_from_ast(node) -> Set[str]:
    """Recursively extract variable names from C10 AST."""
    if node is None:
        return set()
    vs = set()
    cls = node.__class__.__name__
    if cls == 'Var':
        vs.add(node.name)
    elif cls == 'BinOp':
        vs.update(_extract_vars_from_ast(node.left))
        vs.update(_extract_vars_from_ast(node.right))
    elif cls == 'UnaryOp':
        vs.update(_extract_vars_from_ast(node.operand))
    return vs


def _get_stmts(body):
    """Extract statement list from a Block or list."""
    if body is None:
        return []
    if isinstance(body, list):
        return body
    if hasattr(body, 'stmts'):
        return body.stmts
    return [body]


def build_cfg(source: str) -> CFG:
    """Build CFG from C10 source."""
    program = Parser(lex(source)).parse()
    cfg = CFG()
    entry = cfg.add_node(CFGNodeType.ENTRY)
    cfg.entry = entry
    exit_node = cfg.add_node(CFGNodeType.EXIT)
    cfg.exit = exit_node
    last = _build_cfg_stmts(cfg, program.stmts, entry)
    if last is not None:
        cfg.add_edge(last, exit_node)
    return cfg


def _build_cfg_stmts(cfg, stmts, prev):
    """Build CFG nodes for a list of statements. Returns last node id."""
    current = prev
    for stmt in stmts:
        current = _build_cfg_stmt(cfg, stmt, current)
        if current is None:
            return None
    return current


def _build_cfg_stmt(cfg, stmt, prev):
    """Build CFG node for a single statement. Returns node id."""
    cls = stmt.__class__.__name__

    if cls == 'LetDecl':
        nid = cfg.add_node(CFGNodeType.ASSIGN,
                           data={'var': stmt.name, 'expr': stmt.value},
                           line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, nid)
        return nid

    elif cls == 'Assign':
        nid = cfg.add_node(CFGNodeType.ASSIGN,
                           data={'var': stmt.name, 'expr': stmt.value},
                           line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, nid)
        return nid

    elif cls == 'IfStmt':
        assume_t = cfg.add_node(CFGNodeType.ASSUME,
                                data={'cond': stmt.cond},
                                line=getattr(stmt, 'line', 0))
        assume_f = cfg.add_node(CFGNodeType.ASSUME_NOT,
                                data={'cond': stmt.cond},
                                line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, assume_t)
        cfg.add_edge(prev, assume_f)

        join = cfg.add_node(CFGNodeType.SKIP)

        then_end = _build_cfg_stmts(cfg, _get_stmts(stmt.then_body), assume_t)
        if then_end is not None:
            cfg.add_edge(then_end, join)

        if stmt.else_body:
            else_end = _build_cfg_stmts(cfg, _get_stmts(stmt.else_body), assume_f)
            if else_end is not None:
                cfg.add_edge(else_end, join)
        else:
            cfg.add_edge(assume_f, join)

        return join

    elif cls == 'WhileStmt':
        header = cfg.add_node(CFGNodeType.SKIP, line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, header)

        assume_t = cfg.add_node(CFGNodeType.ASSUME,
                                data={'cond': stmt.cond},
                                line=getattr(stmt, 'line', 0))
        assume_f = cfg.add_node(CFGNodeType.ASSUME_NOT,
                                data={'cond': stmt.cond},
                                line=getattr(stmt, 'line', 0))
        cfg.add_edge(header, assume_t)
        cfg.add_edge(header, assume_f)

        body_end = _build_cfg_stmts(cfg, _get_stmts(stmt.body), assume_t)
        if body_end is not None:
            cfg.add_edge(body_end, header)

        return assume_f

    elif cls == 'CallExpr':
        callee = stmt.callee
        callee_name = callee if isinstance(callee, str) else getattr(callee, 'name', None)
        if callee_name == 'assert':
            nid = cfg.add_node(CFGNodeType.ASSERT,
                               data={'cond': stmt.args[0] if stmt.args else None},
                               line=getattr(stmt, 'line', 0))
            cfg.add_edge(prev, nid)
            return nid
        nid = cfg.add_node(CFGNodeType.SKIP, data={'expr': stmt},
                           line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, nid)
        return nid

    elif cls == 'ExprStmt':
        # Check for assert inside ExprStmt
        expr = stmt.expr
        if hasattr(expr, '__class__') and expr.__class__.__name__ == 'CallExpr':
            callee = expr.callee
            callee_name = callee if isinstance(callee, str) else getattr(callee, 'name', None)
            if callee_name == 'assert':
                nid = cfg.add_node(CFGNodeType.ASSERT,
                                   data={'cond': expr.args[0] if expr.args else None},
                                   line=getattr(stmt, 'line', 0))
                cfg.add_edge(prev, nid)
                return nid
        nid = cfg.add_node(CFGNodeType.SKIP, data={'expr': stmt.expr},
                           line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, nid)
        return nid

    else:
        nid = cfg.add_node(CFGNodeType.SKIP, line=getattr(stmt, 'line', 0))
        cfg.add_edge(prev, nid)
        return nid


# ============================================================
# AST to SMT Conversion
# ============================================================

def _ast_to_smt(expr, variables: Dict[str, object]) -> Optional[object]:
    """Convert C10 AST expression to SMT term."""
    if expr is None:
        return None
    cls = expr.__class__.__name__

    if cls == 'IntLit':
        return IntConst(expr.value)

    if cls == 'Var':
        name = expr.name
        if name == 'true':
            return BoolConst(True)
        if name == 'false':
            return BoolConst(False)
        if name not in variables:
            variables[name] = Var(name, INT)
        return variables[name]

    if cls == 'BinOp':
        left = _ast_to_smt(expr.left, variables)
        right = _ast_to_smt(expr.right, variables)
        if left is None or right is None:
            return None
        op_map = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
            '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
            '==': Op.EQ, '!=': Op.NEQ,
            '&&': Op.AND, '||': Op.OR,
        }
        op = op_map.get(expr.op)
        if op is None:
            return None
        if op in (Op.AND, Op.OR):
            return App(op, [left, right], BOOL)
        if op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ):
            return App(op, [left, right], BOOL)
        return App(op, [left, right], INT)

    if cls == 'UnaryOp':
        operand = _ast_to_smt(expr.operand, variables)
        if operand is None:
            return None
        if expr.op == '-':
            return App(Op.SUB, [IntConst(0), operand], INT)
        if expr.op == '!':
            return App(Op.NOT, [operand], BOOL)
        return None

    return None


def _collect_smt_vars(term) -> Set[str]:
    """Collect variable names from an SMT term."""
    if term is None:
        return set()
    if isinstance(term, Var):
        return {term.name}
    if isinstance(term, (IntConst, BoolConst)):
        return set()
    if isinstance(term, App):
        result = set()
        for arg in term.args:
            result.update(_collect_smt_vars(arg))
        return result
    return set()


# ============================================================
# Template-Based Predicate Generation
# ============================================================

class PredicateTemplate:
    """A parameterized predicate schema."""

    def __init__(self, name: str, arity: int, gen_func):
        self.name = name
        self.arity = arity  # number of variable slots
        self.gen_func = gen_func  # (vars, consts) -> List[term]


def _make_comparison_templates() -> List[PredicateTemplate]:
    """Standard comparison templates: x op c, x op y."""
    templates = []

    # x >= 0 (non-negativity)
    def gen_nonneg(variables, constants):
        results = []
        for v in variables:
            var = Var(v, INT)
            results.append(App(Op.GE, [var, IntConst(0)], BOOL))
        return results
    templates.append(PredicateTemplate("nonneg", 1, gen_nonneg))

    # x > 0 (positivity)
    def gen_pos(variables, constants):
        results = []
        for v in variables:
            var = Var(v, INT)
            results.append(App(Op.GT, [var, IntConst(0)], BOOL))
        return results
    templates.append(PredicateTemplate("positive", 1, gen_pos))

    # x <= c for constants from the program
    def gen_upper(variables, constants):
        results = []
        for v in variables:
            var = Var(v, INT)
            for c in constants:
                results.append(App(Op.LE, [var, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("upper_bound", 1, gen_upper))

    # x >= c
    def gen_lower(variables, constants):
        results = []
        for v in variables:
            var = Var(v, INT)
            for c in constants:
                results.append(App(Op.GE, [var, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("lower_bound", 1, gen_lower))

    # x == c
    def gen_equal(variables, constants):
        results = []
        for v in variables:
            var = Var(v, INT)
            for c in constants:
                results.append(App(Op.EQ, [var, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("equality", 1, gen_equal))

    # x < y
    def gen_lt(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                results.append(App(Op.LT, [a, b], BOOL))
                results.append(App(Op.LT, [b, a], BOOL))
        return results
    templates.append(PredicateTemplate("var_order", 2, gen_lt))

    # x <= y
    def gen_le(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                results.append(App(Op.LE, [a, b], BOOL))
                results.append(App(Op.LE, [b, a], BOOL))
        return results
    templates.append(PredicateTemplate("var_leq", 2, gen_le))

    # x == y
    def gen_eq(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                results.append(App(Op.EQ, [a, b], BOOL))
        return results
    templates.append(PredicateTemplate("var_equality", 2, gen_eq))

    return templates


def _make_linear_templates() -> List[PredicateTemplate]:
    """Linear combination templates: a*x + b*y op c."""
    templates = []

    # x + y == c (sum conservation)
    def gen_sum_eq(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                s = App(Op.ADD, [a, b], INT)
                for c in constants:
                    results.append(App(Op.EQ, [s, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("sum_conservation", 2, gen_sum_eq))

    # x - y == c (difference conservation)
    def gen_diff_eq(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                d = App(Op.SUB, [a, b], INT)
                for c in constants:
                    results.append(App(Op.EQ, [d, IntConst(c)], BOOL))
                d2 = App(Op.SUB, [b, a], INT)
                for c in constants:
                    results.append(App(Op.EQ, [d2, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("diff_conservation", 2, gen_diff_eq))

    # x + y <= c (sum bound)
    def gen_sum_le(variables, constants):
        results = []
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                a = Var(v1, INT)
                b = Var(v2, INT)
                s = App(Op.ADD, [a, b], INT)
                for c in constants:
                    results.append(App(Op.LE, [s, IntConst(c)], BOOL))
        return results
    templates.append(PredicateTemplate("sum_bound", 2, gen_sum_le))

    return templates


# ============================================================
# Constant and Variable Extraction from Source
# ============================================================

def _ast_children(node):
    """Get all child AST nodes from a node."""
    if node is None:
        return []
    children = []
    for attr_name in ['stmts', 'body', 'then_body', 'else_body', 'args']:
        attr = getattr(node, attr_name, None)
        if attr is not None:
            if attr_name in ('body', 'then_body', 'else_body'):
                children.extend(_get_stmts(attr))
            elif isinstance(attr, list):
                children.extend(attr)
    for attr_name in ['value', 'left', 'right', 'operand', 'cond', 'expr',
                      'callee']:
        attr = getattr(node, attr_name, None)
        if attr is not None and hasattr(attr, '__class__') and not isinstance(attr, (int, float, str, bool, list, dict, tuple)):
            children.append(attr)
    return children


def _extract_program_constants(source: str) -> Set[int]:
    """Extract integer constants from C10 source."""
    constants = {0, 1}
    try:
        program = Parser(lex(source)).parse()
        _walk_ast_constants(program, constants)
    except Exception:
        pass
    return constants


def _walk_ast_constants(node, constants: Set[int]):
    """Walk AST collecting integer constants."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls == 'IntLit':
        constants.add(node.value)
        if node.value > 0:
            constants.add(node.value - 1)
            constants.add(node.value + 1)
        return
    for child in _ast_children(node):
        _walk_ast_constants(child, constants)


def _extract_program_variables(source: str) -> List[str]:
    """Extract variable names from C10 source."""
    variables = []
    try:
        program = Parser(lex(source)).parse()
        seen = set()
        _walk_ast_variables(program, variables, seen)
    except Exception:
        pass
    return variables


def _walk_ast_variables(node, variables: List[str], seen: Set[str]):
    """Walk AST collecting variable names."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls == 'LetDecl':
        if node.name not in seen:
            seen.add(node.name)
            variables.append(node.name)
    elif cls == 'Assign':
        if node.name not in seen:
            seen.add(node.name)
            variables.append(node.name)
    for child in _ast_children(node):
        _walk_ast_variables(child, variables, seen)


def _extract_conditions(source: str) -> List[object]:
    """Extract branch/loop conditions as SMT terms."""
    conditions = []
    try:
        program = Parser(lex(source)).parse()
        variables = {}
        _walk_ast_conditions(program, conditions, variables)
    except Exception:
        pass
    return conditions


def _walk_ast_conditions(node, conditions, variables):
    """Walk AST collecting conditions from if/while."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls in ('IfStmt', 'WhileStmt'):
        term = _ast_to_smt(node.cond, variables)
        if term is not None:
            conditions.append(term)
    for child in _ast_children(node):
        _walk_ast_conditions(child, conditions, variables)


def _extract_assertions(source: str) -> List[object]:
    """Extract assertion conditions as SMT terms."""
    assertions = []
    try:
        program = Parser(lex(source)).parse()
        variables = {}
        _walk_ast_assertions(program, assertions, variables)
    except Exception:
        pass
    return assertions


def _walk_ast_assertions(node, assertions, variables):
    """Walk AST collecting assert(...) call arguments."""
    if node is None:
        return
    cls = node.__class__.__name__
    if cls == 'CallExpr':
        callee = node.callee
        callee_name = callee if isinstance(callee, str) else getattr(callee, 'name', None)
        if callee_name == 'assert' and node.args:
            term = _ast_to_smt(node.args[0], variables)
            if term is not None:
                assertions.append(term)
    elif cls == 'ExprStmt':
        expr = node.expr
        if hasattr(expr, '__class__') and expr.__class__.__name__ == 'CallExpr':
            callee = expr.callee
            callee_name = callee if isinstance(callee, str) else getattr(callee, 'name', None)
            if callee_name == 'assert' and expr.args:
                term = _ast_to_smt(expr.args[0], variables)
                if term is not None:
                    assertions.append(term)
    for child in _ast_children(node):
        _walk_ast_assertions(child, assertions, variables)


# ============================================================
# Interval Analysis (lightweight, for predicate extraction)
# ============================================================

def _run_interval_analysis(source: str) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """Run lightweight interval analysis on C10 source.
    Returns dict of variable -> (lower, upper) bounds."""
    intervals = {}
    try:
        program = Parser(lex(source)).parse()
        env = {}
        _interval_stmts(program.stmts, env)
        for var, (lo, hi) in env.items():
            intervals[var] = (lo, hi)
    except Exception:
        pass
    return intervals


def _interval_stmts(stmts, env):
    for stmt in stmts:
        _interval_stmt(stmt, env)


def _interval_stmt(stmt, env):
    cls = stmt.__class__.__name__
    if cls in ('LetDecl', 'Assign'):
        name = stmt.name
        val = _interval_eval(stmt.value, env)
        env[name] = val
    elif cls == 'IfStmt':
        env_t = dict(env)
        env_f = dict(env)
        _refine_condition(stmt.cond, env_t, True)
        _refine_condition(stmt.cond, env_f, False)
        _interval_stmts(_get_stmts(stmt.then_body), env_t)
        if stmt.else_body:
            _interval_stmts(_get_stmts(stmt.else_body), env_f)
        # Join
        for var in set(list(env_t.keys()) + list(env_f.keys())):
            lo_t, hi_t = env_t.get(var, (None, None))
            lo_f, hi_f = env_f.get(var, (None, None))
            lo = _min_none(lo_t, lo_f)
            hi = _max_none(hi_t, hi_f)
            env[var] = (lo, hi)
    elif cls == 'WhileStmt':
        # Simple: widen after a few iterations
        for _ in range(3):
            env_body = dict(env)
            _refine_condition(stmt.cond, env_body, True)
            _interval_stmts(_get_stmts(stmt.body), env_body)
            for var in env_body:
                if var in env:
                    old_lo, old_hi = env[var]
                    new_lo, new_hi = env_body[var]
                    lo = _min_none(old_lo, new_lo)
                    hi = _max_none(old_hi, new_hi)
                    env[var] = (lo, hi)
                else:
                    env[var] = env_body[var]
        # After loop: apply negated condition
        _refine_condition(stmt.cond, env, False)


def _interval_eval(expr, env) -> Tuple[Optional[int], Optional[int]]:
    if expr is None:
        return (None, None)
    cls = expr.__class__.__name__
    if cls == 'IntLit':
        return (expr.value, expr.value)
    if cls == 'Var':
        return env.get(expr.name, (None, None))
    if cls == 'BinOp':
        l = _interval_eval(expr.left, env)
        r = _interval_eval(expr.right, env)
        if expr.op == '+':
            return (_add_none(l[0], r[0]), _add_none(l[1], r[1]))
        if expr.op == '-':
            return (_sub_none(l[0], r[1]), _sub_none(l[1], r[0]))
        if expr.op == '*':
            if l[0] is not None and l[1] is not None and r[0] is not None and r[1] is not None:
                products = [l[0]*r[0], l[0]*r[1], l[1]*r[0], l[1]*r[1]]
                return (min(products), max(products))
    if cls == 'UnaryOp' and expr.op == '-':
        inner = _interval_eval(expr.operand, env)
        lo = -inner[1] if inner[1] is not None else None
        hi = -inner[0] if inner[0] is not None else None
        return (lo, hi)
    return (None, None)


def _refine_condition(cond, env, take_true):
    """Refine interval env based on condition truth value."""
    if cond is None:
        return
    cls = cond.__class__.__name__
    if cls != 'BinOp':
        return
    if cond.left.__class__.__name__ == 'Var' and cond.right.__class__.__name__ == 'IntLit':
        var_name = cond.left.name
        c = cond.right.value
        lo, hi = env.get(var_name, (None, None))
        op = cond.op
        if not take_true:
            op = {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                  '==': '!=', '!=': '=='}.get(op, op)
        if op == '<':
            hi = min(hi, c - 1) if hi is not None else c - 1
        elif op == '<=':
            hi = min(hi, c) if hi is not None else c
        elif op == '>':
            lo = max(lo, c + 1) if lo is not None else c + 1
        elif op == '>=':
            lo = max(lo, c) if lo is not None else c
        elif op == '==':
            lo, hi = c, c
        env[var_name] = (lo, hi)


def _min_none(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _max_none(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _add_none(a, b):
    if a is None or b is None:
        return None
    return a + b


def _sub_none(a, b):
    if a is None or b is None:
        return None
    return a - b


# ============================================================
# Predicate Generators
# ============================================================

def _generate_template_predicates(variables: List[str],
                                  constants: Set[int]) -> List[Predicate]:
    """Generate predicates from templates."""
    preds = []
    const_list = sorted(constants)
    # Limit constants to avoid explosion
    if len(const_list) > 10:
        const_list = sorted(constants, key=lambda c: abs(c))[:10]

    for tmpl in _make_comparison_templates():
        terms = tmpl.gen_func(variables, const_list)
        for term in terms:
            preds.append(Predicate(
                term=term,
                source=PredicateSource.TEMPLATE,
                description=f"template:{tmpl.name}"
            ))

    for tmpl in _make_linear_templates():
        terms = tmpl.gen_func(variables, const_list)
        for term in terms:
            preds.append(Predicate(
                term=term,
                source=PredicateSource.TEMPLATE,
                description=f"template:{tmpl.name}"
            ))

    return preds


def _generate_interval_predicates(source: str) -> List[Predicate]:
    """Generate predicates from interval analysis results."""
    preds = []
    intervals = _run_interval_analysis(source)
    for var, (lo, hi) in intervals.items():
        v = Var(var, INT)
        if lo is not None:
            preds.append(Predicate(
                term=App(Op.GE, [v, IntConst(lo)], BOOL),
                source=PredicateSource.INTERVAL,
                description=f"interval:lower({var}>={lo})"
            ))
        if hi is not None:
            preds.append(Predicate(
                term=App(Op.LE, [v, IntConst(hi)], BOOL),
                source=PredicateSource.INTERVAL,
                description=f"interval:upper({var}<={hi})"
            ))
        if lo is not None and hi is not None and lo == hi:
            preds.append(Predicate(
                term=App(Op.EQ, [v, IntConst(lo)], BOOL),
                source=PredicateSource.INTERVAL,
                description=f"interval:exact({var}=={lo})"
            ))
    return preds


def _generate_condition_predicates(source: str) -> List[Predicate]:
    """Generate predicates from branch/loop conditions."""
    preds = []
    conditions = _extract_conditions(source)
    for cond in conditions:
        preds.append(Predicate(
            term=cond,
            source=PredicateSource.CONDITION,
            description="condition:branch"
        ))
    return preds


def _generate_assertion_predicates(source: str) -> List[Predicate]:
    """Generate predicates from assertions (the goal we want to prove)."""
    preds = []
    assertions = _extract_assertions(source)
    for assertion in assertions:
        preds.append(Predicate(
            term=assertion,
            source=PredicateSource.ASSERTION,
            description="assertion"
        ))
    return preds


# ============================================================
# Inductiveness Checking
# ============================================================

def _check_inductive(predicate_term, cfg: CFG, variables: List[str]) -> InductivePredicate:
    """Check if a predicate is inductive for loops in the CFG."""
    loop_headers = cfg.get_loop_headers()
    if not loop_headers:
        return InductivePredicate(
            term=predicate_term, init_holds=True, preserved=True, strengthening=[]
        )

    # For each loop header, check init and preservation
    # We encode the loop as: if predicate holds before iteration, does it hold after?
    # Use SMT to check

    # Simple check: does executing any assignment preserve the predicate?
    # This is a sound overapproximation
    assign_nodes = [n for n in cfg.nodes.values()
                    if n.ntype == CFGNodeType.ASSIGN and n.data]

    preserved = True
    for node in assign_nodes:
        var_name = node.data.get('var')
        expr = node.data.get('expr')
        if var_name is None or expr is None:
            continue

        # Check: predicate(pre) AND var' = expr(pre) => predicate(post)
        s = SMTSolver()
        smt_vars = {}
        for v in variables:
            smt_vars[v] = s.Int(v)
        smt_vars_post = {}
        for v in variables:
            smt_vars_post[v] = s.Int(v + "_post")

        # predicate holds in pre-state
        pre_pred = _substitute_vars(predicate_term, smt_vars)
        if pre_pred is None:
            continue

        # post-state: all vars same except var_name
        post_pred = _substitute_vars(predicate_term, smt_vars_post)
        if post_pred is None:
            continue

        # Frame: unchanged vars
        frame_constraints = []
        for v in variables:
            if v != var_name:
                frame_constraints.append(
                    App(Op.EQ, [smt_vars_post[v], smt_vars[v]], BOOL)
                )

        # var_name_post = expr(pre_state)
        expr_smt = _ast_to_smt(expr, smt_vars)
        if expr_smt is None:
            continue
        assign_constraint = App(Op.EQ, [smt_vars_post[var_name], expr_smt], BOOL)

        # Check: pre_pred AND frame AND assign => post_pred
        # Equivalently: pre_pred AND frame AND assign AND NOT(post_pred) is UNSAT
        s.add(pre_pred)
        for fc in frame_constraints:
            s.add(fc)
        s.add(assign_constraint)
        neg_post = App(Op.NOT, [post_pred], BOOL)
        s.add(neg_post)

        result = s.check()
        if result == SMTResult.SAT:
            preserved = False
            break

    return InductivePredicate(
        term=predicate_term,
        init_holds=True,  # We don't check init here (done separately)
        preserved=preserved,
        strengthening=[]
    )


def _substitute_vars(term, var_map):
    """Replace Var nodes in term with mapped versions."""
    if term is None:
        return None
    if isinstance(term, Var):
        return var_map.get(term.name, term)
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = []
        for arg in term.args:
            new_arg = _substitute_vars(arg, var_map)
            if new_arg is None:
                return None
            new_args.append(new_arg)
        return App(term.op, new_args, term.sort)
    return None


# ============================================================
# Interpolation-Based Predicate Mining
# ============================================================

def _mine_interpolation_predicates(cfg: CFG, variables: List[str],
                                   max_paths: int = 10) -> List[Predicate]:
    """Mine predicates by generating interpolants along CFG paths."""
    preds = []

    # Find paths from entry to assertion/error nodes
    target_nodes = [n.id for n in cfg.nodes.values()
                    if n.ntype in (CFGNodeType.ASSERT, CFGNodeType.ERROR)]
    if not target_nodes:
        return preds

    paths = _enumerate_paths(cfg, cfg.entry, target_nodes, max_paths)

    for path in paths:
        path_preds = _interpolate_along_path(cfg, path, variables)
        preds.extend(path_preds)

    return preds


def _enumerate_paths(cfg: CFG, start: int, targets: List[int],
                     max_paths: int) -> List[List[int]]:
    """BFS enumerate paths from start to targets."""
    if start is None:
        return []
    paths = []
    queue = [(start, [start])]
    visited_paths = set()

    while queue and len(paths) < max_paths:
        current, path = queue.pop(0)
        if current in targets:
            path_key = tuple(path)
            if path_key not in visited_paths:
                visited_paths.add(path_key)
                paths.append(path)
            continue
        if len(path) > 50:
            continue
        for succ in cfg.nodes[current].successors:
            # Allow revisiting for loops (but limit path length)
            if path.count(succ) < 2:
                queue.append((succ, path + [succ]))

    return paths


def _interpolate_along_path(cfg: CFG, path: List[int],
                            variables: List[str]) -> List[Predicate]:
    """Generate interpolants at each cut point along a path."""
    preds = []
    if len(path) < 3:
        return preds

    # Build formulas for each step
    step_formulas = []
    smt_vars = {v: Var(v, INT) for v in variables}

    for i in range(len(path) - 1):
        node = cfg.nodes[path[i]]
        formula = _node_to_formula(node, smt_vars)
        if formula is not None:
            step_formulas.append(formula)
        else:
            step_formulas.append(BoolConst(True))

    # Try interpolating at each midpoint
    for mid in range(1, len(step_formulas)):
        a_formulas = step_formulas[:mid]
        b_formulas = step_formulas[mid:]

        # Conjoin A-formulas and B-formulas
        a_conj = _conjoin(a_formulas)
        b_conj = _conjoin(b_formulas)

        if a_conj is None or b_conj is None:
            continue

        # Check if A AND B is UNSAT (necessary for interpolation)
        s = SMTSolver()
        for v in variables:
            s.Int(v)
        s.add(a_conj)
        s.add(b_conj)
        result = s.check()

        if result == SMTResult.UNSAT:
            # Extract a simple interpolant: try A's implied bounds
            interp = _extract_simple_interpolant(a_conj, b_conj, variables)
            if interp is not None:
                preds.append(Predicate(
                    term=interp,
                    source=PredicateSource.INTERPOLATION,
                    description="interpolation:path_cut"
                ))

    return preds


def _node_to_formula(node: CFGNode, smt_vars: Dict) -> Optional[object]:
    """Convert a CFG node to an SMT formula."""
    if node.data is None:
        return None

    if node.ntype == CFGNodeType.ASSUME:
        cond = node.data.get('cond')
        if cond:
            return _ast_to_smt(cond, smt_vars)

    if node.ntype == CFGNodeType.ASSUME_NOT:
        cond = node.data.get('cond')
        if cond:
            inner = _ast_to_smt(cond, smt_vars)
            if inner:
                return App(Op.NOT, [inner], BOOL)

    if node.ntype == CFGNodeType.ASSERT:
        cond = node.data.get('cond')
        if cond:
            # For the path to error: negate the assertion
            inner = _ast_to_smt(cond, smt_vars)
            if inner:
                return App(Op.NOT, [inner], BOOL)

    return None


def _conjoin(formulas: List[object]) -> Optional[object]:
    """Conjoin a list of formulas."""
    non_trivial = [f for f in formulas
                   if f is not None and not (isinstance(f, BoolConst) and f.value)]
    if not non_trivial:
        return BoolConst(True)
    if len(non_trivial) == 1:
        return non_trivial[0]
    result = non_trivial[0]
    for f in non_trivial[1:]:
        result = App(Op.AND, [result, f], BOOL)
    return result


def _extract_simple_interpolant(a_conj, b_conj, variables) -> Optional[object]:
    """Extract a simple interpolant by testing variable bounds implied by A."""
    a_vars = _collect_smt_vars(a_conj)
    b_vars = _collect_smt_vars(b_conj)
    shared = a_vars & b_vars

    if not shared:
        return None

    # Try each shared variable's bounds
    for var_name in shared:
        if var_name not in variables:
            continue
        var = Var(var_name, INT)

        # Binary search for upper bound implied by A
        for bound in [0, 1, -1, 5, 10, 100]:
            candidate = App(Op.LE, [var, IntConst(bound)], BOOL)

            # Check: A => candidate (i.e., A AND NOT(candidate) is UNSAT)
            s1 = SMTSolver()
            for v in variables:
                s1.Int(v)
            s1.add(a_conj)
            s1.add(App(Op.NOT, [candidate], BOOL))
            if s1.check() == SMTResult.UNSAT:
                # Check: candidate AND B is UNSAT
                s2 = SMTSolver()
                for v in variables:
                    s2.Int(v)
                s2.add(candidate)
                s2.add(b_conj)
                if s2.check() == SMTResult.UNSAT:
                    return candidate

        # Try lower bound
        for bound in [0, 1, -1, 5, 10]:
            candidate = App(Op.GE, [var, IntConst(bound)], BOOL)
            s1 = SMTSolver()
            for v in variables:
                s1.Int(v)
            s1.add(a_conj)
            s1.add(App(Op.NOT, [candidate], BOOL))
            if s1.check() == SMTResult.UNSAT:
                s2 = SMTSolver()
                for v in variables:
                    s2.Int(v)
                s2.add(candidate)
                s2.add(b_conj)
                if s2.check() == SMTResult.UNSAT:
                    return candidate

    return None


# ============================================================
# Predicate Scoring and Ranking
# ============================================================

def _score_predicates(predicates: List[Predicate], cfg: CFG,
                      variables: List[str]) -> List[Predicate]:
    """Score and rank predicates by estimated usefulness."""
    # Dedup: keep the version with the best source
    source_priority = {
        PredicateSource.ASSERTION: 10,
        PredicateSource.INDUCTIVE: 9,
        PredicateSource.CONDITION: 8,
        PredicateSource.INTERPOLATION: 7,
        PredicateSource.INTERVAL: 5,
        PredicateSource.ZONE: 5,
        PredicateSource.USER: 10,
        PredicateSource.TEMPLATE: 2,
    }
    best_by_term = {}
    for pred in predicates:
        term_str = str(pred.term)
        if term_str not in best_by_term:
            best_by_term[term_str] = pred
        else:
            existing = best_by_term[term_str]
            if source_priority.get(pred.source, 0) > source_priority.get(existing.source, 0):
                best_by_term[term_str] = pred

    scored = []
    seen_terms = set()

    for pred in best_by_term.values():
        term_str = str(pred.term)
        if term_str in seen_terms:
            continue
        seen_terms.add(term_str)

        score = 0.0

        # Source-based scoring
        source_scores = {
            PredicateSource.ASSERTION: 10.0,
            PredicateSource.CONDITION: 8.0,
            PredicateSource.INTERPOLATION: 7.0,
            PredicateSource.INDUCTIVE: 9.0,
            PredicateSource.INTERVAL: 5.0,
            PredicateSource.ZONE: 5.0,
            PredicateSource.TEMPLATE: 2.0,
            PredicateSource.USER: 10.0,
        }
        score += source_scores.get(pred.source, 1.0)

        # Variable relevance: predicates over more program variables score higher
        pred_vars = _collect_smt_vars(pred.term)
        relevant_vars = pred_vars & set(variables)
        score += len(relevant_vars) * 1.5

        # Inductiveness bonus
        if cfg and variables:
            ind = _check_inductive(pred.term, cfg, variables)
            if ind.preserved:
                score += 5.0

        scored.append(Predicate(
            term=pred.term,
            source=pred.source,
            location=pred.location,
            score=score,
            description=pred.description
        ))

    scored.sort(key=lambda p: p.score, reverse=True)
    return scored


# ============================================================
# Recursive Predicate Learning
# ============================================================

def _learn_recursive_predicates(source: str, cfg: CFG,
                                variables: List[str]) -> List[Predicate]:
    """Learn predicates that are inductive for loops.
    Uses template instantiation + inductiveness checking."""
    preds = []
    constants = _extract_program_constants(source)

    # Generate candidate predicates
    candidates = _generate_template_predicates(variables, constants)

    # Check each for inductiveness
    for cand in candidates:
        ind = _check_inductive(cand.term, cfg, variables)
        if ind.preserved:
            preds.append(Predicate(
                term=cand.term,
                source=PredicateSource.INDUCTIVE,
                location=cand.location,
                score=9.0,
                description=f"inductive:{cand.description}"
            ))

    return preds


# ============================================================
# Main Discovery Engine
# ============================================================

class PredicateDiscoveryEngine:
    """Main engine for recursive predicate discovery."""

    def __init__(self, source: str, max_predicates: int = 50,
                 use_templates: bool = True,
                 use_intervals: bool = True,
                 use_conditions: bool = True,
                 use_assertions: bool = True,
                 use_interpolation: bool = True,
                 use_inductive: bool = True):
        self.source = source
        self.max_predicates = max_predicates
        self.use_templates = use_templates
        self.use_intervals = use_intervals
        self.use_conditions = use_conditions
        self.use_assertions = use_assertions
        self.use_interpolation = use_interpolation
        self.use_inductive = use_inductive

        self.variables = _extract_program_variables(source)
        self.constants = _extract_program_constants(source)
        self.cfg = build_cfg(source)

    def discover(self) -> DiscoveryResult:
        """Run full predicate discovery pipeline."""
        all_preds = []
        stats = {}

        # Phase 1: condition and assertion predicates (cheapest, highest value)
        if self.use_assertions:
            assertion_preds = _generate_assertion_predicates(self.source)
            all_preds.extend(assertion_preds)
            stats['assertions'] = len(assertion_preds)

        if self.use_conditions:
            condition_preds = _generate_condition_predicates(self.source)
            all_preds.extend(condition_preds)
            stats['conditions'] = len(condition_preds)

        # Phase 2: interval-based predicates
        if self.use_intervals:
            interval_preds = _generate_interval_predicates(self.source)
            all_preds.extend(interval_preds)
            stats['intervals'] = len(interval_preds)

        # Phase 3: template predicates
        if self.use_templates:
            template_preds = _generate_template_predicates(
                self.variables, self.constants)
            all_preds.extend(template_preds)
            stats['templates'] = len(template_preds)

        # Phase 4: inductive predicate learning
        if self.use_inductive:
            inductive_preds = _learn_recursive_predicates(
                self.source, self.cfg, self.variables)
            all_preds.extend(inductive_preds)
            stats['inductive'] = len(inductive_preds)

        # Phase 5: interpolation mining
        if self.use_interpolation:
            interp_preds = _mine_interpolation_predicates(
                self.cfg, self.variables)
            all_preds.extend(interp_preds)
            stats['interpolation'] = len(interp_preds)

        total = len(all_preds)

        # Score and rank
        ranked = _score_predicates(all_preds, self.cfg, self.variables)

        # Select top predicates
        selected = ranked[:self.max_predicates]

        # Count by source
        source_counts = defaultdict(int)
        for p in selected:
            source_counts[p.source.value] += 1

        return DiscoveryResult(
            predicates=selected,
            ranked_predicates=ranked,
            source_counts=dict(source_counts),
            total_candidates=total,
            selected_count=len(selected),
            discovery_stats=stats
        )


# ============================================================
# CEGAR Integration: Discovery-Guided Refinement
# ============================================================

def _check_predicate_sufficient(source: str, pred_terms: List[object],
                                property_term: object = None) -> Dict:
    """Check if discovered predicates are sufficient to prove a property.
    Uses predicate abstraction + forward reachability."""
    cfg = build_cfg(source)
    variables = _extract_program_variables(source)

    # Build abstract reachability using predicate set
    # For each CFG node, compute which predicates hold
    abstract_states = {}  # node_id -> set of predicate indices that hold
    worklist = []

    if cfg.entry is not None:
        # Init: check which predicates hold at entry
        init_preds = set()
        for i, pt in enumerate(pred_terms):
            # Check if predicate is implied by initial state (trivially true)
            s = SMTSolver()
            for v in variables:
                s.Int(v)
            s.add(App(Op.NOT, [pt], BOOL))
            # If NOT(pred) is UNSAT, pred always holds
            if s.check() == SMTResult.UNSAT:
                init_preds.add(i)
        abstract_states[cfg.entry] = init_preds
        worklist.append(cfg.entry)

    visited = set()
    while worklist:
        nid = worklist.pop(0)
        if nid in visited:
            continue
        visited.add(nid)

        current_preds = abstract_states.get(nid, set())
        node = cfg.nodes[nid]

        for succ_id in node.successors:
            # Compute abstract post
            succ_preds = set()
            for i, pt in enumerate(pred_terms):
                # Check if pred is preserved/implied
                if i in current_preds:
                    succ_preds.add(i)  # Conservative: keep if held before
                # Check if node action establishes it
                if node.ntype == CFGNodeType.ASSUME:
                    cond = node.data.get('cond') if node.data else None
                    if cond:
                        smt_vars = {}
                        cond_smt = _ast_to_smt(cond, smt_vars)
                        if cond_smt:
                            # Check: condition => pred
                            s = SMTSolver()
                            for v in variables:
                                s.Int(v)
                            s.add(cond_smt)
                            s.add(App(Op.NOT, [pt], BOOL))
                            if s.check() == SMTResult.UNSAT:
                                succ_preds.add(i)

            old = abstract_states.get(succ_id, set())
            new = old | succ_preds
            if new != old:
                abstract_states[succ_id] = new
                worklist.append(succ_id)

    # Check if property holds at all assertion nodes
    assertion_nodes = [n for n in cfg.nodes.values()
                       if n.ntype == CFGNodeType.ASSERT]
    all_proved = True
    proved_assertions = []
    for node in assertion_nodes:
        preds_at = abstract_states.get(node.id, set())
        # Check if any predicate at this node implies the assertion
        if property_term:
            for i in preds_at:
                s = SMTSolver()
                for v in variables:
                    s.Int(v)
                s.add(pred_terms[i])
                s.add(App(Op.NOT, [property_term], BOOL))
                if s.check() == SMTResult.UNSAT:
                    proved_assertions.append(node.id)
                    break
            else:
                all_proved = False
        else:
            proved_assertions.append(node.id)

    return {
        'sufficient': all_proved,
        'proved_assertions': proved_assertions,
        'total_assertions': len(assertion_nodes),
        'abstract_states': {k: len(v) for k, v in abstract_states.items()},
        'predicate_count': len(pred_terms),
    }


# ============================================================
# Public API
# ============================================================

def discover_predicates(source: str, max_predicates: int = 50,
                        **kwargs) -> DiscoveryResult:
    """Discover predicates for a C10 program.

    Args:
        source: C10 source code
        max_predicates: maximum number of predicates to return
        **kwargs: flags for use_templates, use_intervals, etc.

    Returns:
        DiscoveryResult with ranked predicates
    """
    engine = PredicateDiscoveryEngine(source, max_predicates, **kwargs)
    return engine.discover()


def discover_inductive_predicates(source: str,
                                  max_predicates: int = 30) -> List[Predicate]:
    """Discover only inductive predicates for loops.

    Args:
        source: C10 source code
        max_predicates: maximum number of predicates

    Returns:
        List of inductive predicates sorted by score
    """
    engine = PredicateDiscoveryEngine(
        source, max_predicates,
        use_templates=True, use_intervals=True,
        use_conditions=True, use_assertions=True,
        use_interpolation=False, use_inductive=True
    )
    result = engine.discover()
    return [p for p in result.ranked_predicates
            if p.source == PredicateSource.INDUCTIVE][:max_predicates]


def discover_and_verify(source: str, property_term=None,
                        max_predicates: int = 50) -> Dict:
    """Discover predicates and check if they suffice for verification.

    Args:
        source: C10 source code
        property_term: optional SMT property to verify
        max_predicates: maximum predicates to discover

    Returns:
        Dict with discovery result and verification check
    """
    result = discover_predicates(source, max_predicates)
    pred_terms = [p.term for p in result.predicates]
    verification = _check_predicate_sufficient(source, pred_terms, property_term)
    return {
        'discovery': result,
        'verification': verification,
        'sufficient': verification['sufficient'],
        'predicate_count': result.selected_count,
        'total_candidates': result.total_candidates,
    }


def get_cfg(source: str) -> CFG:
    """Build CFG from C10 source.

    Args:
        source: C10 source code

    Returns:
        CFG object
    """
    return build_cfg(source)


def get_program_info(source: str) -> Dict:
    """Extract program information: variables, constants, conditions.

    Args:
        source: C10 source code

    Returns:
        Dict with variables, constants, conditions, assertions
    """
    return {
        'variables': _extract_program_variables(source),
        'constants': sorted(_extract_program_constants(source)),
        'conditions': [str(c) for c in _extract_conditions(source)],
        'assertions': [str(a) for a in _extract_assertions(source)],
    }


def check_inductiveness(source: str, predicate_term) -> InductivePredicate:
    """Check if a predicate is inductive for loops in the program.

    Args:
        source: C10 source code
        predicate_term: SMT term to check

    Returns:
        InductivePredicate with init_holds and preserved flags
    """
    cfg = build_cfg(source)
    variables = _extract_program_variables(source)
    return _check_inductive(predicate_term, cfg, variables)


def compare_discovery_strategies(source: str) -> Dict:
    """Compare different predicate discovery strategies.

    Args:
        source: C10 source code

    Returns:
        Dict with per-strategy results and comparison
    """
    strategies = {
        'templates_only': {'use_templates': True, 'use_intervals': False,
                           'use_conditions': False, 'use_assertions': False,
                           'use_interpolation': False, 'use_inductive': False},
        'conditions_only': {'use_templates': False, 'use_intervals': False,
                            'use_conditions': True, 'use_assertions': True,
                            'use_interpolation': False, 'use_inductive': False},
        'intervals_only': {'use_templates': False, 'use_intervals': True,
                           'use_conditions': False, 'use_assertions': False,
                           'use_interpolation': False, 'use_inductive': False},
        'inductive_only': {'use_templates': False, 'use_intervals': False,
                           'use_conditions': False, 'use_assertions': False,
                           'use_interpolation': False, 'use_inductive': True},
        'full': {'use_templates': True, 'use_intervals': True,
                 'use_conditions': True, 'use_assertions': True,
                 'use_interpolation': True, 'use_inductive': True},
    }

    results = {}
    for name, kwargs in strategies.items():
        result = discover_predicates(source, max_predicates=50, **kwargs)
        results[name] = {
            'total_candidates': result.total_candidates,
            'selected': result.selected_count,
            'source_counts': result.source_counts,
            'top_scores': [p.score for p in result.ranked_predicates[:5]],
        }

    return results


def predicate_summary(source: str) -> str:
    """Human-readable summary of predicate discovery.

    Args:
        source: C10 source code

    Returns:
        Formatted summary string
    """
    result = discover_predicates(source)
    lines = ["=== Predicate Discovery Summary ==="]
    lines.append(f"Total candidates: {result.total_candidates}")
    lines.append(f"Selected: {result.selected_count}")
    lines.append(f"Sources: {result.source_counts}")
    lines.append(f"Discovery stats: {result.discovery_stats}")
    lines.append("")
    lines.append("Top 10 predicates:")
    for i, p in enumerate(result.ranked_predicates[:10]):
        lines.append(f"  {i+1}. [{p.source.value}] score={p.score:.1f} "
                     f"{p.description}: {p.term}")
    return "\n".join(lines)
