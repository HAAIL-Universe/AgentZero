"""
V032: Combined Numeric + Shape Analysis

Composes V030 shape analysis with numeric abstract interpretation to analyze
programs that have both heap structures and integer data.

Tracks:
- Heap shape (pointers, sharing, cycles) via 3-valued shape graphs
- Numeric data stored in heap nodes (integer intervals + signs)
- List length bounds (via instrumentation predicates)
- Sortedness properties (data(n) <= data(n.next))
- Program variables with both pointer and integer types

Language extends V030's heap language with:
- x.data = e       (store integer to node field)
- x = y.data       (load integer from node field)
- x = len(y)       (get length of list from y)
- Integer arithmetic and comparisons on program variables
- assert_sorted(x) (check ascending order)
- assert_length(x, op, n) (check list length property)
"""

import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import math

# Import V030 shape analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V030_shape_analysis'))
from shape_analysis import (
    TV, Node, ShapeGraph, ShapeWarning, AnalysisVerdict,
)


# =============================================================================
# Numeric Domain (simplified from C039, tailored for heap node data)
# =============================================================================

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def is_bot(self):
        return self.lo > self.hi

    def is_top(self):
        return self.lo == float('-inf') and self.hi == float('inf')

    def contains(self, v):
        return not self.is_bot() and self.lo <= v <= self.hi

    def contains_zero(self):
        return not self.is_bot() and self.lo <= 0 <= self.hi

    def __repr__(self):
        if self.is_bot():
            return "BOT"
        if self.is_top():
            return "[-inf, +inf]"
        lo_s = "-inf" if self.lo == float('-inf') else str(int(self.lo) if self.lo == int(self.lo) else self.lo)
        hi_s = "+inf" if self.hi == float('inf') else str(int(self.hi) if self.hi == int(self.hi) else self.hi)
        return f"[{lo_s}, {hi_s}]"


INTERVAL_BOT = Interval(1, 0)
INTERVAL_TOP = Interval(float('-inf'), float('inf'))
INTERVAL_ZERO = Interval(0, 0)
INTERVAL_NON_NEG = Interval(0, float('inf'))


def interval_join(a, b):
    if a.is_bot():
        return b
    if b.is_bot():
        return a
    return Interval(min(a.lo, b.lo), max(a.hi, b.hi))


def interval_meet(a, b):
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    return Interval(max(a.lo, b.lo), min(a.hi, b.hi))


def interval_widen(old, new):
    if old.is_bot():
        return new
    if new.is_bot():
        return old
    lo = float('-inf') if new.lo < old.lo else old.lo
    hi = float('inf') if new.hi > old.hi else old.hi
    return Interval(lo, hi)


def interval_add(a, b):
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    return Interval(a.lo + b.lo, a.hi + b.hi)


def interval_sub(a, b):
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    return Interval(a.lo - b.hi, a.hi - b.lo)


def interval_neg(a):
    if a.is_bot():
        return INTERVAL_BOT
    return Interval(-a.hi, -a.lo)


def interval_mul(a, b):
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    prods = []
    for x in [a.lo, a.hi]:
        for y in [b.lo, b.hi]:
            p = x * y
            if math.isnan(p):
                p = 0.0
            prods.append(p)
    return Interval(min(prods), max(prods))


# =============================================================================
# Extended AST for Combined Language
# =============================================================================

class StmtKind(Enum):
    # Pointer operations (from V030)
    NEW = "new"
    ASSIGN = "assign"          # x = y (pointer or int)
    LOAD_NEXT = "load_next"    # x = y.next
    STORE_NEXT = "store_next"  # x.next = y
    NULL = "null"              # x = null
    # Integer data operations
    STORE_DATA = "store_data"  # x.data = expr
    LOAD_DATA = "load_data"    # x = y.data
    INT_ASSIGN = "int_assign"  # x = integer_expr
    # Control flow
    IF = "if"
    WHILE = "while"
    ASSUME = "assume"
    # Assertions
    ASSERT_NOT_NULL = "assert_not_null"
    ASSERT_ACYCLIC = "assert_acyclic"
    ASSERT_REACHABLE = "assert_reachable"
    ASSERT_DISJOINT = "assert_disjoint"
    ASSERT_SORTED = "assert_sorted"
    ASSERT_LENGTH = "assert_length"
    ASSERT_DATA_RANGE = "assert_data_range"


class ExprKind(Enum):
    VAR = "var"
    INT_LIT = "int_lit"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    NEG = "neg"
    LEN = "len"          # len(x)  -- list length from pointer var


@dataclass
class Expr:
    kind: ExprKind
    value: object = None      # for INT_LIT
    name: str = ""            # for VAR
    left: 'Expr' = None
    right: 'Expr' = None
    var_name: str = ""        # for LEN


@dataclass
class Stmt:
    kind: StmtKind
    lhs: str = ""             # target variable
    rhs: str = ""             # source variable
    expr: Expr = None         # integer expression
    body: list = field(default_factory=list)
    else_body: list = field(default_factory=list)
    cond_var: str = ""        # condition variable
    cond_null: bool = True    # True = x==null, False = x!=null
    cond_op: str = ""         # for integer conditions: <, <=, >, >=, ==, !=
    cond_expr: Expr = None    # RHS of integer condition
    assert_op: str = ""       # for assert_length: <, <=, >, >=, ==
    assert_val: int = 0       # for assert_length: the bound
    assert_lo: int = 0        # for assert_data_range: low bound
    assert_hi: int = 0        # for assert_data_range: high bound


# =============================================================================
# Parser for Combined Language
# =============================================================================

class ParseError(Exception):
    pass


def parse_combined_program(source):
    """Parse a combined heap+numeric program into a list of Stmts."""
    tokens = _tokenize(source)
    parser = _CombinedParser(tokens)
    return parser.parse_stmts()


def _tokenize(source):
    """Simple tokenizer for the combined language."""
    tokens = []
    i = 0
    keywords = {
        'new', 'null', 'if', 'else', 'while', 'assume', 'next', 'data',
        'assert_not_null', 'assert_acyclic', 'assert_reachable',
        'assert_disjoint', 'assert_sorted', 'assert_length',
        'assert_data_range', 'len',
    }
    while i < len(source):
        c = source[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue
        if c in '(){}=;.,+-*':
            if c == '=' and i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(('==', '=='))
                i += 2
                continue
            if c == '!' and i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(('!=', '!='))
                i += 2
                continue
            tokens.append((c, c))
            i += 1
            continue
        if c == '!':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(('!=', '!='))
                i += 2
            else:
                tokens.append(('!', '!'))
                i += 1
            continue
        if c == '<':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(('<=', '<='))
                i += 2
            else:
                tokens.append(('<', '<'))
                i += 1
            continue
        if c == '>':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(('>=', '>='))
                i += 2
            else:
                tokens.append(('>', '>'))
                i += 1
            continue
        if c.isdigit() or (c == '-' and i + 1 < len(source) and source[i + 1].isdigit()):
            j = i
            if c == '-':
                j += 1
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(('NUM', int(source[i:j])))
            i = j
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            if word in keywords:
                tokens.append((word, word))
            else:
                tokens.append(('ID', word))
            i = j
            continue
        raise ParseError(f"Unexpected character: {c!r}")
    tokens.append(('EOF', None))
    return tokens


class _CombinedParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, kind):
        t = self.advance()
        if t[0] != kind:
            raise ParseError(f"Expected {kind}, got {t}")
        return t

    def match(self, kind):
        if self.peek()[0] == kind:
            return self.advance()
        return None

    def parse_stmts(self):
        stmts = []
        while self.peek()[0] not in ('EOF', '}'):
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        tok = self.peek()

        if tok[0] == 'if':
            return self.parse_if()
        if tok[0] == 'while':
            return self.parse_while()
        if tok[0] == 'assume':
            return self.parse_assume()
        if tok[0] in ('assert_not_null', 'assert_acyclic', 'assert_sorted'):
            return self.parse_assert_unary()
        if tok[0] in ('assert_reachable', 'assert_disjoint'):
            return self.parse_assert_binary()
        if tok[0] == 'assert_length':
            return self.parse_assert_length()
        if tok[0] == 'assert_data_range':
            return self.parse_assert_data_range()

        # Assignment forms
        if tok[0] == 'ID':
            return self.parse_assignment()

        raise ParseError(f"Unexpected token: {tok}")

    def parse_assignment(self):
        name = self.expect('ID')[1]

        # x.next = y  or  x.data = expr
        if self.match('.'):
            field_tok = self.peek()
            if field_tok[0] == 'next':
                self.advance()
                self.expect('=')
                rhs = self.expect('ID')[1]
                self.expect(';')
                return Stmt(kind=StmtKind.STORE_NEXT, lhs=name, rhs=rhs)
            elif field_tok[0] == 'data':
                self.advance()
                self.expect('=')
                expr = self.parse_expr()
                self.expect(';')
                return Stmt(kind=StmtKind.STORE_DATA, lhs=name, expr=expr)
            else:
                raise ParseError(f"Unknown field: {field_tok}")

        self.expect('=')
        rhs_tok = self.peek()

        # x = new()
        if rhs_tok[0] == 'new':
            self.advance()
            self.expect('(')
            self.expect(')')
            self.expect(';')
            return Stmt(kind=StmtKind.NEW, lhs=name)

        # x = null
        if rhs_tok[0] == 'null':
            self.advance()
            self.expect(';')
            return Stmt(kind=StmtKind.NULL, lhs=name)

        # x = y.next  or  x = y.data
        if rhs_tok[0] == 'ID':
            saved = self.pos
            rhs_name = self.advance()[1]
            if self.match('.'):
                field_tok = self.peek()
                if field_tok[0] == 'next':
                    self.advance()
                    self.expect(';')
                    return Stmt(kind=StmtKind.LOAD_NEXT, lhs=name, rhs=rhs_name)
                elif field_tok[0] == 'data':
                    self.advance()
                    self.expect(';')
                    return Stmt(kind=StmtKind.LOAD_DATA, lhs=name, rhs=rhs_name)
                else:
                    raise ParseError(f"Unknown field: {field_tok}")
            # x = y (pointer assign) -- only if followed by ;
            if self.peek()[0] == ';':
                self.advance()
                return Stmt(kind=StmtKind.ASSIGN, lhs=name, rhs=rhs_name)
            # Otherwise, backtrack and parse as integer expression
            self.pos = saved

        # x = len(y)
        if rhs_tok[0] == 'len':
            self.advance()
            self.expect('(')
            var_name = self.expect('ID')[1]
            self.expect(')')
            self.expect(';')
            return Stmt(kind=StmtKind.INT_ASSIGN, lhs=name,
                        expr=Expr(kind=ExprKind.LEN, var_name=var_name))

        # x = integer_expr
        expr = self.parse_expr()
        self.expect(';')
        return Stmt(kind=StmtKind.INT_ASSIGN, lhs=name, expr=expr)

    def parse_expr(self):
        return self.parse_add()

    def parse_add(self):
        left = self.parse_mul()
        while self.peek()[0] in ('+', '-'):
            op = self.advance()[0]
            right = self.parse_mul()
            if op == '+':
                left = Expr(kind=ExprKind.ADD, left=left, right=right)
            else:
                left = Expr(kind=ExprKind.SUB, left=left, right=right)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek()[0] == '*':
            self.advance()
            right = self.parse_unary()
            left = Expr(kind=ExprKind.MUL, left=left, right=right)
        return left

    def parse_unary(self):
        if self.peek()[0] == '-':
            self.advance()
            operand = self.parse_atom()
            return Expr(kind=ExprKind.NEG, left=operand)
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()
        if tok[0] == 'NUM':
            self.advance()
            return Expr(kind=ExprKind.INT_LIT, value=tok[1])
        if tok[0] == 'len':
            self.advance()
            self.expect('(')
            var_name = self.expect('ID')[1]
            self.expect(')')
            return Expr(kind=ExprKind.LEN, var_name=var_name)
        if tok[0] == 'ID':
            self.advance()
            return Expr(kind=ExprKind.VAR, name=tok[1])
        if tok[0] == '(':
            self.advance()
            e = self.parse_expr()
            self.expect(')')
            return e
        raise ParseError(f"Expected expression, got {tok}")

    def parse_if(self):
        self.expect('if')
        self.expect('(')
        cond_var, cond_null, cond_op, cond_expr = self.parse_condition()
        self.expect(')')
        self.expect('{')
        body = self.parse_stmts()
        self.expect('}')
        else_body = []
        if self.match('else'):
            self.expect('{')
            else_body = self.parse_stmts()
            self.expect('}')
        return Stmt(kind=StmtKind.IF, cond_var=cond_var, cond_null=cond_null,
                    cond_op=cond_op, cond_expr=cond_expr,
                    body=body, else_body=else_body)

    def parse_while(self):
        self.expect('while')
        self.expect('(')
        cond_var, cond_null, cond_op, cond_expr = self.parse_condition()
        self.expect(')')
        self.expect('{')
        body = self.parse_stmts()
        self.expect('}')
        return Stmt(kind=StmtKind.WHILE, cond_var=cond_var, cond_null=cond_null,
                    cond_op=cond_op, cond_expr=cond_expr, body=body)

    def parse_condition(self):
        """Parse condition. Returns (var, is_null, cond_op, cond_expr).
        For null checks: cond_op="" and cond_expr=None.
        For integer comparisons: cond_null is ignored, cond_op and cond_expr set."""
        var_name = self.expect('ID')[1]
        tok = self.peek()
        # x == null / x != null
        if tok[0] == '==' and self._lookahead_null():
            self.advance()
            self.expect('null')
            return (var_name, True, "", None)
        if tok[0] == '!=' and self._lookahead_null():
            self.advance()
            self.expect('null')
            return (var_name, False, "", None)
        # Integer comparison: x op expr
        if tok[0] in ('<', '<=', '>', '>=', '==', '!='):
            op = self.advance()[0]
            expr = self.parse_expr()
            return (var_name, True, op, expr)  # cond_null irrelevant for int cond
        raise ParseError(f"Expected condition operator, got {tok}")

    def _lookahead_null(self):
        return self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == 'null'

    def parse_assume(self):
        self.expect('assume')
        self.expect('(')
        cond_var, cond_null, cond_op, cond_expr = self.parse_condition()
        self.expect(')')
        self.expect(';')
        return Stmt(kind=StmtKind.ASSUME, cond_var=cond_var, cond_null=cond_null,
                    cond_op=cond_op, cond_expr=cond_expr)

    def parse_assert_unary(self):
        kind_map = {
            'assert_not_null': StmtKind.ASSERT_NOT_NULL,
            'assert_acyclic': StmtKind.ASSERT_ACYCLIC,
            'assert_sorted': StmtKind.ASSERT_SORTED,
        }
        tok = self.advance()
        kind = kind_map[tok[0]]
        self.expect('(')
        var_name = self.expect('ID')[1]
        self.expect(')')
        self.expect(';')
        return Stmt(kind=kind, lhs=var_name)

    def parse_assert_binary(self):
        kind_map = {
            'assert_reachable': StmtKind.ASSERT_REACHABLE,
            'assert_disjoint': StmtKind.ASSERT_DISJOINT,
        }
        tok = self.advance()
        kind = kind_map[tok[0]]
        self.expect('(')
        v1 = self.expect('ID')[1]
        self.expect(',')
        v2 = self.expect('ID')[1]
        self.expect(')')
        self.expect(';')
        return Stmt(kind=kind, lhs=v1, rhs=v2)

    def parse_assert_length(self):
        self.expect('assert_length')
        self.expect('(')
        var_name = self.expect('ID')[1]
        self.expect(',')
        op = self.advance()
        if op[0] not in ('<', '<=', '>', '>=', '=='):
            raise ParseError(f"Expected comparison op, got {op}")
        self.expect(',')
        val = self.expect('NUM')[1]
        self.expect(')')
        self.expect(';')
        return Stmt(kind=StmtKind.ASSERT_LENGTH, lhs=var_name,
                    assert_op=op[0], assert_val=val)

    def parse_assert_data_range(self):
        self.expect('assert_data_range')
        self.expect('(')
        var_name = self.expect('ID')[1]
        self.expect(',')
        lo = self.expect('NUM')[1]
        self.expect(',')
        hi = self.expect('NUM')[1]
        self.expect(')')
        self.expect(';')
        return Stmt(kind=StmtKind.ASSERT_DATA_RANGE, lhs=var_name,
                    assert_lo=lo, assert_hi=hi)


# =============================================================================
# Combined State: Shape + Numeric
# =============================================================================

class CombinedState:
    """Combined abstract state: shape graph + numeric info per node + per variable."""

    def __init__(self):
        self.graph = ShapeGraph()
        # Numeric data stored in heap nodes: node_id -> Interval
        self.node_data = {}
        # Numeric values for program variables: var_name -> Interval
        self.var_values = {}
        # Track which variables are pointer-type vs int-type
        self.ptr_vars = set()
        self.int_vars = set()

    def copy(self):
        s = CombinedState()
        s.graph = self.graph.copy()
        s.node_data = dict(self.node_data)
        s.var_values = dict(self.var_values)
        s.ptr_vars = set(self.ptr_vars)
        s.int_vars = set(self.int_vars)
        return s

    def get_node_data(self, node):
        """Get the data interval for a node."""
        return self.node_data.get(node.id, INTERVAL_TOP)

    def set_node_data(self, node, interval):
        """Set the data interval for a node."""
        self.node_data[node.id] = interval

    def get_var_value(self, name):
        """Get the numeric interval for a variable."""
        return self.var_values.get(name, INTERVAL_TOP)

    def set_var_value(self, name, interval):
        """Set the numeric interval for a variable."""
        self.var_values[name] = interval
        self.int_vars.add(name)

    def join(self, other):
        """Join two combined states."""
        result = CombinedState()
        result.graph = self.graph.join(other.graph)
        # Join node data
        all_node_ids = set(self.node_data.keys()) | set(other.node_data.keys())
        for nid in all_node_ids:
            a = self.node_data.get(nid, INTERVAL_BOT)
            b = other.node_data.get(nid, INTERVAL_BOT)
            result.node_data[nid] = interval_join(a, b)
        # Join variable values
        all_vars = set(self.var_values.keys()) | set(other.var_values.keys())
        for v in all_vars:
            a = self.var_values.get(v, INTERVAL_BOT)
            b = other.var_values.get(v, INTERVAL_BOT)
            result.var_values[v] = interval_join(a, b)
        result.ptr_vars = self.ptr_vars | other.ptr_vars
        result.int_vars = self.int_vars | other.int_vars
        return result

    def widen(self, other):
        """Widen for loop convergence."""
        result = CombinedState()
        result.graph = self.graph.join(other.graph)
        # Widen node data
        all_node_ids = set(self.node_data.keys()) | set(other.node_data.keys())
        for nid in all_node_ids:
            a = self.node_data.get(nid, INTERVAL_BOT)
            b = other.node_data.get(nid, INTERVAL_BOT)
            if a.is_bot():
                result.node_data[nid] = b
            elif b.is_bot():
                result.node_data[nid] = a
            else:
                result.node_data[nid] = interval_widen(a, b)
        # Widen variable values
        all_vars = set(self.var_values.keys()) | set(other.var_values.keys())
        for v in all_vars:
            a = self.var_values.get(v, INTERVAL_BOT)
            b = other.var_values.get(v, INTERVAL_BOT)
            if a.is_bot():
                result.var_values[v] = b
            elif b.is_bot():
                result.var_values[v] = a
            else:
                result.var_values[v] = interval_widen(a, b)
        result.ptr_vars = self.ptr_vars | other.ptr_vars
        result.int_vars = self.int_vars | other.int_vars
        return result

    def equals(self, other):
        """Check structural equality."""
        if not self.graph.equals(other.graph):
            return False
        if self.node_data != other.node_data:
            return False
        if self.var_values != other.var_values:
            return False
        return True


# =============================================================================
# Instrumentation Predicates (Numeric)
# =============================================================================

def compute_list_length(state, var_name):
    """Compute an interval bounding the list length from var_name.

    Returns an Interval representing possible lengths.
    Concrete nodes contribute 1, summary nodes contribute [1, +inf).
    """
    graph = state.graph
    targets = graph.get_var_targets(var_name)
    if not targets:
        return INTERVAL_ZERO  # null pointer -> length 0

    # BFS/DFS to count reachable nodes
    # We track min/max possible lengths
    visited = set()
    min_len = 0
    max_len = 0
    has_summary = False

    worklist = []
    for node, tv in targets.items():
        if tv != TV.FALSE:
            worklist.append(node)

    if not worklist:
        return INTERVAL_ZERO

    # Traverse the list structure
    current_nodes = worklist
    while current_nodes:
        next_nodes = []
        for node in current_nodes:
            if node.id in visited:
                continue
            visited.add(node.id)
            if node.summary:
                has_summary = True
                min_len += 1
            else:
                min_len += 1
            max_len += 1
            # Follow next edges
            nexts = graph.get_next_targets(node)
            for nxt, tv in nexts.items():
                if tv != TV.FALSE and nxt.id not in visited:
                    next_nodes.append(nxt)
        current_nodes = next_nodes

    if has_summary:
        return Interval(1, float('inf'))
    return Interval(min_len, max_len)


def check_sorted_property(state, var_name):
    """Check if the list from var_name is sorted (ascending data values).

    Returns TV.TRUE if definitely sorted, TV.FALSE if definitely unsorted,
    TV.MAYBE if uncertain.
    """
    graph = state.graph
    targets = graph.get_var_targets(var_name)
    if not targets:
        return TV.TRUE  # empty list is sorted

    result = TV.TRUE
    visited = set()

    def traverse(node):
        nonlocal result
        if node.id in visited:
            return
        visited.add(node.id)
        node_interval = state.get_node_data(node)
        nexts = graph.get_next_targets(node)
        for nxt, tv in nexts.items():
            if tv == TV.FALSE:
                continue
            nxt_interval = state.get_node_data(nxt)
            # Check: node_interval.hi <= nxt_interval.lo means definitely sorted
            if node_interval.is_bot() or nxt_interval.is_bot():
                pass  # skip BOT
            elif node_interval.hi <= nxt_interval.lo:
                pass  # definitely ordered at this edge
            elif nxt_interval.hi < node_interval.lo:
                # Definitely NOT sorted
                if tv == TV.TRUE:
                    result = TV.FALSE
                    return
                else:
                    result = result & TV.MAYBE
            else:
                # Overlapping ranges -- maybe sorted
                if tv == TV.TRUE:
                    result = result & TV.MAYBE
                else:
                    result = result & TV.MAYBE
            traverse(nxt)

    for node, tv in targets.items():
        if tv != TV.FALSE:
            traverse(node)

    return result


def check_data_range(state, var_name, lo, hi):
    """Check if all data values in the list from var_name are in [lo, hi].

    Returns TV.TRUE/FALSE/MAYBE.
    """
    graph = state.graph
    targets = graph.get_var_targets(var_name)
    if not targets:
        return TV.TRUE  # empty list trivially satisfies

    constraint = Interval(lo, hi)
    result = TV.TRUE
    visited = set()

    def traverse(node):
        nonlocal result
        if node.id in visited:
            return
        visited.add(node.id)
        data = state.get_node_data(node)
        if data.is_bot():
            pass
        else:
            contained = interval_meet(data, constraint)
            if contained.is_bot():
                # data completely outside range
                result = TV.FALSE
                return
            elif data.lo >= lo and data.hi <= hi:
                pass  # completely inside range
            else:
                result = result & TV.MAYBE
        nexts = graph.get_next_targets(node)
        for nxt, tv in nexts.items():
            if tv != TV.FALSE:
                traverse(nxt)

    for node, tv in targets.items():
        if tv != TV.FALSE:
            traverse(node)

    return result


# =============================================================================
# Combined Analyzer
# =============================================================================

@dataclass
class CombinedWarning:
    kind: str
    message: str
    stmt: Optional[Stmt] = None


@dataclass
class CombinedResult:
    verdict: AnalysisVerdict
    final_state: CombinedState
    warnings: list
    properties: dict


class CombinedAnalyzer:
    """Analyzer that tracks both shape and numeric properties."""

    def __init__(self, max_iterations=20):
        self.max_iterations = max_iterations
        self.warnings = []

    def analyze(self, stmts):
        """Analyze a list of statements, return final CombinedState."""
        state = CombinedState()
        state = self._interpret_stmts(stmts, state)
        state.graph.canonicalize()
        return state

    def _interpret_stmts(self, stmts, state):
        for stmt in stmts:
            state = self._interpret_stmt(stmt, state)
        return state

    def _interpret_stmt(self, stmt, state):
        kind = stmt.kind

        if kind == StmtKind.NEW:
            return self._do_new(stmt, state)
        elif kind == StmtKind.ASSIGN:
            return self._do_assign(stmt, state)
        elif kind == StmtKind.LOAD_NEXT:
            return self._do_load_next(stmt, state)
        elif kind == StmtKind.STORE_NEXT:
            return self._do_store_next(stmt, state)
        elif kind == StmtKind.NULL:
            return self._do_null(stmt, state)
        elif kind == StmtKind.STORE_DATA:
            return self._do_store_data(stmt, state)
        elif kind == StmtKind.LOAD_DATA:
            return self._do_load_data(stmt, state)
        elif kind == StmtKind.INT_ASSIGN:
            return self._do_int_assign(stmt, state)
        elif kind == StmtKind.IF:
            return self._do_if(stmt, state)
        elif kind == StmtKind.WHILE:
            return self._do_while(stmt, state)
        elif kind == StmtKind.ASSUME:
            return self._do_assume(stmt, state)
        elif kind in (StmtKind.ASSERT_NOT_NULL, StmtKind.ASSERT_ACYCLIC,
                      StmtKind.ASSERT_SORTED, StmtKind.ASSERT_REACHABLE,
                      StmtKind.ASSERT_DISJOINT, StmtKind.ASSERT_LENGTH,
                      StmtKind.ASSERT_DATA_RANGE):
            return self._do_assert(stmt, state)
        else:
            raise ValueError(f"Unknown statement kind: {kind}")

    # -- Pointer operations --

    def _do_new(self, stmt, state):
        state = state.copy()
        node = state.graph.fresh_node(summary=False)
        state.graph.clear_var(stmt.lhs)
        state.graph.set_var(stmt.lhs, node, TV.TRUE)
        state.set_node_data(node, INTERVAL_TOP)  # fresh node has unknown data
        state.ptr_vars.add(stmt.lhs)
        return state

    def _do_assign(self, stmt, state):
        state = state.copy()
        # Pointer assignment: x = y
        if stmt.rhs in state.ptr_vars or stmt.rhs not in state.int_vars:
            targets = state.graph.get_var_targets(stmt.rhs)
            state.graph.clear_var(stmt.lhs)
            for node, tv in targets.items():
                if tv != TV.FALSE:
                    state.graph.set_var(stmt.lhs, node, tv)
            state.ptr_vars.add(stmt.lhs)
        else:
            # Integer assignment: x = y
            state.set_var_value(stmt.lhs, state.get_var_value(stmt.rhs))
        return state

    def _do_load_next(self, stmt, state):
        state = state.copy()
        self._focus_var(state, stmt.rhs)
        targets = state.graph.get_var_targets(stmt.rhs)
        state.graph.clear_var(stmt.lhs)
        if not targets:
            self._warn('NULL_DEREF', f"Null dereference: {stmt.rhs}.next", stmt)
            return state
        for node, tv in targets.items():
            if tv == TV.FALSE:
                continue
            nexts = state.graph.get_next_targets(node)
            for nxt, ntv in nexts.items():
                if ntv == TV.FALSE:
                    continue
                combined = tv & ntv
                old = state.graph.get_var_targets(stmt.lhs).get(nxt, TV.FALSE)
                state.graph.set_var(stmt.lhs, nxt, old | combined)
        state.ptr_vars.add(stmt.lhs)
        return state

    def _do_store_next(self, stmt, state):
        state = state.copy()
        self._focus_var(state, stmt.lhs)
        lhs_targets = state.graph.get_var_targets(stmt.lhs)
        rhs_targets = state.graph.get_var_targets(stmt.rhs)
        if not lhs_targets:
            self._warn('NULL_DEREF', f"Null dereference: {stmt.lhs}.next = {stmt.rhs}", stmt)
            return state
        for node, tv in lhs_targets.items():
            if tv == TV.FALSE:
                continue
            if tv == TV.TRUE:
                # Strong update
                state.graph.clear_next(node)
                for rhs_node, rtv in rhs_targets.items():
                    if rtv != TV.FALSE:
                        state.graph.set_next(node, rhs_node, rtv)
            else:
                # Weak update (MAYBE)
                for rhs_node, rtv in rhs_targets.items():
                    if rtv != TV.FALSE:
                        state.graph.set_next(node, rhs_node, TV.MAYBE)
                # Existing edges become MAYBE
                for nxt, ntv in state.graph.get_next_targets(node).items():
                    state.graph.set_next(node, nxt, TV.MAYBE)
        self._blur(state)
        return state

    def _do_null(self, stmt, state):
        state = state.copy()
        state.graph.clear_var(stmt.lhs)
        state.ptr_vars.add(stmt.lhs)
        return state

    # -- Numeric data operations --

    def _do_store_data(self, stmt, state):
        """x.data = expr"""
        state = state.copy()
        val = self._eval_expr(stmt.expr, state)
        self._focus_var(state, stmt.lhs)
        targets = state.graph.get_var_targets(stmt.lhs)
        if not targets:
            self._warn('NULL_DEREF', f"Null dereference: {stmt.lhs}.data = ...", stmt)
            return state
        for node, tv in targets.items():
            if tv == TV.FALSE:
                continue
            if tv == TV.TRUE:
                # Strong update
                state.set_node_data(node, val)
            else:
                # Weak update: join old and new
                old = state.get_node_data(node)
                state.set_node_data(node, interval_join(old, val))
        return state

    def _do_load_data(self, stmt, state):
        """x = y.data"""
        state = state.copy()
        self._focus_var(state, stmt.rhs)
        targets = state.graph.get_var_targets(stmt.rhs)
        if not targets:
            self._warn('NULL_DEREF', f"Null dereference: {stmt.rhs}.data", stmt)
            state.set_var_value(stmt.lhs, INTERVAL_BOT)
            return state
        result = INTERVAL_BOT
        for node, tv in targets.items():
            if tv == TV.FALSE:
                continue
            result = interval_join(result, state.get_node_data(node))
        state.set_var_value(stmt.lhs, result)
        return state

    def _do_int_assign(self, stmt, state):
        """x = integer_expr"""
        state = state.copy()
        val = self._eval_expr(stmt.expr, state)
        state.set_var_value(stmt.lhs, val)
        return state

    # -- Expression evaluation --

    def _eval_expr(self, expr, state):
        """Evaluate expression to an Interval."""
        if expr.kind == ExprKind.INT_LIT:
            v = expr.value
            return Interval(v, v)
        elif expr.kind == ExprKind.VAR:
            return state.get_var_value(expr.name)
        elif expr.kind == ExprKind.ADD:
            l = self._eval_expr(expr.left, state)
            r = self._eval_expr(expr.right, state)
            return interval_add(l, r)
        elif expr.kind == ExprKind.SUB:
            l = self._eval_expr(expr.left, state)
            r = self._eval_expr(expr.right, state)
            return interval_sub(l, r)
        elif expr.kind == ExprKind.MUL:
            l = self._eval_expr(expr.left, state)
            r = self._eval_expr(expr.right, state)
            return interval_mul(l, r)
        elif expr.kind == ExprKind.NEG:
            v = self._eval_expr(expr.left, state)
            return interval_neg(v)
        elif expr.kind == ExprKind.LEN:
            return compute_list_length(state, expr.var_name)
        else:
            return INTERVAL_TOP

    # -- Control flow --

    def _do_if(self, stmt, state):
        if stmt.cond_op:
            return self._do_if_int(stmt, state)
        return self._do_if_null(stmt, state)

    def _do_if_null(self, stmt, state):
        """if (x == null) / if (x != null)"""
        then_state = state.copy()
        else_state = state.copy()
        self._assume_null(then_state, stmt.cond_var, stmt.cond_null)
        self._assume_null(else_state, stmt.cond_var, not stmt.cond_null)
        then_state = self._interpret_stmts(stmt.body, then_state)
        else_state = self._interpret_stmts(stmt.else_body, else_state)
        return then_state.join(else_state)

    def _do_if_int(self, stmt, state):
        """if (x < expr) / if (x >= expr) etc."""
        val = self._eval_expr(stmt.cond_expr, state)
        var_val = state.get_var_value(stmt.cond_var)
        # Check if condition is definitely true or definitely false
        cond_result = self._eval_int_cond(var_val, stmt.cond_op, val)
        if cond_result == TV.TRUE:
            # Only then-branch is feasible
            then_state = state.copy()
            self._refine_int_var(then_state, stmt.cond_var, stmt.cond_op, val)
            return self._interpret_stmts(stmt.body, then_state)
        elif cond_result == TV.FALSE:
            # Only else-branch is feasible
            else_state = state.copy()
            neg_op = self._negate_op(stmt.cond_op)
            self._refine_int_var(else_state, stmt.cond_var, neg_op, val)
            return self._interpret_stmts(stmt.else_body, else_state)
        else:
            # Both branches feasible
            then_state = state.copy()
            else_state = state.copy()
            self._refine_int_var(then_state, stmt.cond_var, stmt.cond_op, val)
            neg_op = self._negate_op(stmt.cond_op)
            self._refine_int_var(else_state, stmt.cond_var, neg_op, val)
            then_state = self._interpret_stmts(stmt.body, then_state)
            else_state = self._interpret_stmts(stmt.else_body, else_state)
            return then_state.join(else_state)

    def _do_while(self, stmt, state):
        for _ in range(self.max_iterations):
            loop_state = state.copy()
            if stmt.cond_op:
                val = self._eval_expr(stmt.cond_expr, loop_state)
                self._refine_int_var(loop_state, stmt.cond_var, stmt.cond_op, val)
            else:
                self._assume_null(loop_state, stmt.cond_var, stmt.cond_null)
            loop_state = self._interpret_stmts(stmt.body, loop_state)
            self._blur(loop_state)
            next_state = state.widen(loop_state)
            if next_state.equals(state):
                break
            state = next_state

        # Exit condition
        exit_state = state.copy()
        if stmt.cond_op:
            val = self._eval_expr(stmt.cond_expr, exit_state)
            neg_op = self._negate_op(stmt.cond_op)
            self._refine_int_var(exit_state, stmt.cond_var, neg_op, val)
        else:
            self._assume_null(exit_state, stmt.cond_var, not stmt.cond_null)
        return exit_state

    def _do_assume(self, stmt, state):
        state = state.copy()
        if stmt.cond_op:
            val = self._eval_expr(stmt.cond_expr, state)
            self._refine_int_var(state, stmt.cond_var, stmt.cond_op, val)
        else:
            self._assume_null(state, stmt.cond_var, stmt.cond_null)
        return state

    def _assume_null(self, state, var, is_null):
        """Refine state assuming var is/isn't null."""
        if is_null:
            state.graph.clear_var(var)
        else:
            targets = state.graph.get_var_targets(var)
            if not targets:
                # Must be non-null but points to nothing known -- create summary
                node = state.graph.fresh_node(summary=True)
                state.graph.set_var(var, node, TV.TRUE)
                state.set_node_data(node, INTERVAL_TOP)

    def _refine_int_var(self, state, var_name, op, bound):
        """Refine integer variable interval based on comparison."""
        current = state.get_var_value(var_name)
        if bound.is_bot() or current.is_bot():
            return
        if op == '<':
            constraint = Interval(float('-inf'), bound.hi - 1)
        elif op == '<=':
            constraint = Interval(float('-inf'), bound.hi)
        elif op == '>':
            constraint = Interval(bound.lo + 1, float('inf'))
        elif op == '>=':
            constraint = Interval(bound.lo, float('inf'))
        elif op == '==':
            constraint = bound
        elif op == '!=':
            return  # can't represent with intervals
        else:
            return
        refined = interval_meet(current, constraint)
        state.set_var_value(var_name, refined)

    def _negate_op(self, op):
        neg_map = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
        return neg_map.get(op, op)

    def _eval_int_cond(self, var_interval, op, rhs_interval):
        """Evaluate if var op rhs is definitely TRUE/FALSE/MAYBE."""
        if var_interval.is_bot() or rhs_interval.is_bot():
            return TV.MAYBE
        if op == '<':
            if var_interval.hi < rhs_interval.lo:
                return TV.TRUE
            if var_interval.lo >= rhs_interval.hi:
                return TV.FALSE
        elif op == '<=':
            if var_interval.hi <= rhs_interval.lo:
                return TV.TRUE
            if var_interval.lo > rhs_interval.hi:
                return TV.FALSE
        elif op == '>':
            if var_interval.lo > rhs_interval.hi:
                return TV.TRUE
            if var_interval.hi <= rhs_interval.lo:
                return TV.FALSE
        elif op == '>=':
            if var_interval.lo >= rhs_interval.hi:
                return TV.TRUE
            if var_interval.hi < rhs_interval.lo:
                return TV.FALSE
        elif op == '==':
            if var_interval.lo == var_interval.hi == rhs_interval.lo == rhs_interval.hi:
                return TV.TRUE
            m = interval_meet(var_interval, rhs_interval)
            if m.is_bot():
                return TV.FALSE
        elif op == '!=':
            m = interval_meet(var_interval, rhs_interval)
            if m.is_bot():
                return TV.TRUE
            if var_interval.lo == var_interval.hi == rhs_interval.lo == rhs_interval.hi:
                return TV.FALSE
        return TV.MAYBE

    # -- Assertions --

    def _do_assert(self, stmt, state):
        kind = stmt.kind

        if kind == StmtKind.ASSERT_NOT_NULL:
            is_null = state.graph.is_null(stmt.lhs)
            if is_null == TV.TRUE:
                self._warn('VIOLATION', f"Assertion failed: {stmt.lhs} is null", stmt)
            elif is_null == TV.MAYBE:
                self._warn('MAYBE_VIOLATION', f"Assertion may fail: {stmt.lhs} might be null", stmt)

        elif kind == StmtKind.ASSERT_ACYCLIC:
            targets = state.graph.get_var_targets(stmt.lhs)
            for node, tv in targets.items():
                if tv == TV.FALSE:
                    continue
                on_cycle = state.graph.is_on_cycle(node)
                if on_cycle == TV.TRUE:
                    self._warn('VIOLATION', f"Assertion failed: {stmt.lhs} has cycle", stmt)
                elif on_cycle == TV.MAYBE:
                    self._warn('MAYBE_VIOLATION', f"Assertion may fail: {stmt.lhs} might have cycle", stmt)

        elif kind == StmtKind.ASSERT_SORTED:
            result = check_sorted_property(state, stmt.lhs)
            if result == TV.FALSE:
                self._warn('VIOLATION', f"Assertion failed: {stmt.lhs} is not sorted", stmt)
            elif result == TV.MAYBE:
                self._warn('MAYBE_VIOLATION', f"Assertion may fail: {stmt.lhs} might not be sorted", stmt)

        elif kind == StmtKind.ASSERT_REACHABLE:
            # Check if any node pointed to by rhs is reachable from lhs
            rhs_targets = state.graph.get_var_targets(stmt.rhs)
            reach = TV.FALSE
            for target_node, rtv in rhs_targets.items():
                if rtv == TV.FALSE:
                    continue
                r = state.graph.reachable_from_var_general(stmt.lhs, target_node)
                reach = reach | r
            if reach == TV.FALSE:
                self._warn('VIOLATION', f"Assertion failed: {stmt.rhs} not reachable from {stmt.lhs}", stmt)
            elif reach == TV.MAYBE:
                self._warn('MAYBE_VIOLATION', f"Assertion may fail: {stmt.rhs} might not be reachable from {stmt.lhs}", stmt)

        elif kind == StmtKind.ASSERT_DISJOINT:
            # Check if any node is shared between the two lists
            targets_a = state.graph.get_var_targets(stmt.lhs)
            targets_b = state.graph.get_var_targets(stmt.rhs)
            shared = set(targets_a.keys()) & set(targets_b.keys())
            definitely_shared = any(
                targets_a.get(n, TV.FALSE) == TV.TRUE and targets_b.get(n, TV.FALSE) == TV.TRUE
                for n in shared
            )
            maybe_shared = any(
                targets_a.get(n, TV.FALSE) != TV.FALSE and targets_b.get(n, TV.FALSE) != TV.FALSE
                for n in shared
            )
            if definitely_shared:
                self._warn('VIOLATION', f"Assertion failed: {stmt.lhs} and {stmt.rhs} not disjoint", stmt)
            elif maybe_shared:
                self._warn('MAYBE_VIOLATION', f"Assertion may fail: {stmt.lhs} and {stmt.rhs} might not be disjoint", stmt)

        elif kind == StmtKind.ASSERT_LENGTH:
            length = compute_list_length(state, stmt.lhs)
            holds = self._check_interval_op(length, stmt.assert_op, stmt.assert_val)
            if holds == TV.FALSE:
                self._warn('VIOLATION',
                           f"Assertion failed: len({stmt.lhs}) {stmt.assert_op} {stmt.assert_val}", stmt)
            elif holds == TV.MAYBE:
                self._warn('MAYBE_VIOLATION',
                           f"Assertion may fail: len({stmt.lhs}) {stmt.assert_op} {stmt.assert_val}", stmt)

        elif kind == StmtKind.ASSERT_DATA_RANGE:
            result = check_data_range(state, stmt.lhs, stmt.assert_lo, stmt.assert_hi)
            if result == TV.FALSE:
                self._warn('VIOLATION',
                           f"Assertion failed: data in {stmt.lhs} not in [{stmt.assert_lo}, {stmt.assert_hi}]", stmt)
            elif result == TV.MAYBE:
                self._warn('MAYBE_VIOLATION',
                           f"Assertion may fail: data in {stmt.lhs} might not be in [{stmt.assert_lo}, {stmt.assert_hi}]", stmt)

        return state

    def _check_interval_op(self, interval, op, val):
        """Check if interval satisfies op val. Returns TV."""
        if interval.is_bot():
            return TV.TRUE  # vacuously true
        if op == '==':
            if interval.lo == interval.hi == val:
                return TV.TRUE
            elif not interval.contains(val):
                return TV.FALSE
            else:
                return TV.MAYBE
        elif op == '<':
            if interval.hi < val:
                return TV.TRUE
            elif interval.lo >= val:
                return TV.FALSE
            else:
                return TV.MAYBE
        elif op == '<=':
            if interval.hi <= val:
                return TV.TRUE
            elif interval.lo > val:
                return TV.FALSE
            else:
                return TV.MAYBE
        elif op == '>':
            if interval.lo > val:
                return TV.TRUE
            elif interval.hi <= val:
                return TV.FALSE
            else:
                return TV.MAYBE
        elif op == '>=':
            if interval.lo >= val:
                return TV.TRUE
            elif interval.hi < val:
                return TV.FALSE
            else:
                return TV.MAYBE
        return TV.MAYBE

    # -- Focus/Blur (delegated to shape graph with numeric propagation) --

    def _focus_var(self, state, var_name):
        """Materialize summary nodes that var points to."""
        targets = state.graph.get_var_targets(var_name)
        for node, tv in list(targets.items()):
            if tv != TV.FALSE and node.summary:
                self._materialize(state, var_name, node)

    def _materialize(self, state, var_name, summary_node):
        """Materialize a summary node into concrete + remaining summary.

        Also propagates numeric data: both concrete and remaining inherit
        the summary's data interval.
        """
        graph = state.graph
        data = state.get_node_data(summary_node)

        concrete = graph.fresh_node(summary=False)
        remaining = graph.fresh_node(summary=True)

        # Propagate numeric data to both halves
        state.set_node_data(concrete, data)
        state.set_node_data(remaining, data)

        # Variable that triggered focus points definitely to concrete
        graph.set_var(var_name, concrete, TV.TRUE)

        # Other variables that pointed to summary -> split
        for v in list(graph.var_points.keys()):
            if v == var_name:
                continue
            targets = graph.get_var_targets(v)
            if summary_node in targets and targets[summary_node] != TV.FALSE:
                graph.set_var(v, concrete, TV.MAYBE)
                graph.set_var(v, remaining, TV.MAYBE)

        # Transfer incoming next edges
        for n in list(graph.nodes):
            if n == summary_node:
                continue
            nexts = graph.get_next_targets(n)
            if summary_node in nexts and nexts[summary_node] != TV.FALSE:
                old_tv = nexts[summary_node]
                graph.set_next(n, concrete, old_tv)
                graph.set_next(n, remaining, old_tv)

        # Transfer outgoing edges from summary
        self_loop = TV.FALSE
        other_edges = {}
        for nxt, tv in graph.get_next_targets(summary_node).items():
            if nxt == summary_node:
                self_loop = tv
            else:
                other_edges[nxt] = tv

        # Concrete node: gets other edges + maybe-edge to remaining (for self-loop)
        for nxt, tv in other_edges.items():
            graph.set_next(concrete, nxt, tv)
        if self_loop != TV.FALSE:
            graph.set_next(concrete, remaining, TV.MAYBE)

        # Remaining: gets other edges + self-loop stays
        for nxt, tv in other_edges.items():
            graph.set_next(remaining, nxt, tv)
        if self_loop != TV.FALSE:
            graph.set_next(remaining, remaining, TV.MAYBE)

        # Remove summary
        graph.remove_node(summary_node)
        if summary_node.id in state.node_data:
            del state.node_data[summary_node.id]

    def _blur(self, state):
        """Merge indistinguishable non-pointed-to nodes (with numeric join)."""
        graph = state.graph

        # Find nodes definitely pointed to by variables
        definite = set()
        for v in graph.var_points:
            for node, tv in graph.get_var_targets(v).items():
                if tv == TV.TRUE:
                    definite.add(node)

        candidates = [n for n in graph.nodes if n not in definite]
        if len(candidates) < 2:
            return

        # Group by out-degree
        groups = {}
        for n in candidates:
            nexts = graph.get_next_targets(n)
            deg = sum(1 for tv in nexts.values() if tv != TV.FALSE)
            groups.setdefault(deg, []).append(n)

        for group in groups.values():
            if len(group) < 2:
                continue
            self._merge_nodes(state, group)

    def _merge_nodes(self, state, nodes):
        """Merge a set of nodes into a summary node, joining numeric data."""
        graph = state.graph
        summary = graph.fresh_node(summary=True)

        # Join numeric data from all merged nodes
        merged_data = INTERVAL_BOT
        for n in nodes:
            merged_data = interval_join(merged_data, state.get_node_data(n))
        state.set_node_data(summary, merged_data)

        node_set = set(n.id for n in nodes)

        # Transfer variable pointers
        for v in list(graph.var_points.keys()):
            combined = TV.FALSE
            for n in nodes:
                targets = graph.get_var_targets(v)
                if n in targets:
                    combined = combined | targets[n]
            if combined != TV.FALSE:
                graph.set_var(v, summary, combined)

        # Transfer outgoing edges
        for n in nodes:
            for nxt, tv in graph.get_next_targets(n).items():
                if tv == TV.FALSE:
                    continue
                if nxt.id in node_set:
                    # Self-referencing -> summary self-loop
                    graph.set_next(summary, summary, tv)
                else:
                    old = graph.get_next_targets(summary).get(nxt, TV.FALSE)
                    graph.set_next(summary, nxt, old | tv)

        # Transfer incoming edges
        for n in list(graph.nodes):
            if n.id in node_set:
                continue
            for tgt in nodes:
                nexts = graph.get_next_targets(n)
                if tgt in nexts and nexts[tgt] != TV.FALSE:
                    old = graph.get_next_targets(n).get(summary, TV.FALSE)
                    graph.set_next(n, summary, old | nexts[tgt])

        # Remove old nodes
        for n in nodes:
            graph.remove_node(n)
            if n.id in state.node_data:
                del state.node_data[n.id]

    def _warn(self, kind, message, stmt=None):
        self.warnings.append(CombinedWarning(kind=kind, message=message, stmt=stmt))


# =============================================================================
# Public API
# =============================================================================

def analyze_combined(source, max_iterations=20):
    """Analyze a combined heap+numeric program.

    Returns CombinedResult with verdict, final state, warnings, and properties.
    """
    stmts = parse_combined_program(source)
    analyzer = CombinedAnalyzer(max_iterations=max_iterations)
    final_state = analyzer.analyze(stmts)

    # Determine verdict
    has_violation = any(w.kind == 'VIOLATION' for w in analyzer.warnings)
    has_maybe = any(w.kind == 'MAYBE_VIOLATION' for w in analyzer.warnings)

    if has_violation:
        verdict = AnalysisVerdict.UNSAFE
    elif has_maybe:
        verdict = AnalysisVerdict.MAYBE
    else:
        verdict = AnalysisVerdict.SAFE

    # Collect properties
    properties = {}
    for v in final_state.ptr_vars:
        props = {}
        props['is_null'] = str(final_state.graph.is_null(v))
        props['length'] = str(compute_list_length(final_state, v))
        props['sorted'] = str(check_sorted_property(final_state, v))
        properties[v] = props
    for v in final_state.int_vars:
        if v not in properties:
            properties[v] = {}
        properties[v]['value'] = str(final_state.get_var_value(v))

    return CombinedResult(
        verdict=verdict,
        final_state=final_state,
        warnings=analyzer.warnings,
        properties=properties,
    )


def get_list_length(source, var_name):
    """Get the list length interval for a pointer variable."""
    stmts = parse_combined_program(source)
    analyzer = CombinedAnalyzer()
    state = analyzer.analyze(stmts)
    return compute_list_length(state, var_name)


def get_node_data_range(source, var_name):
    """Get the data interval for nodes reachable from var_name."""
    stmts = parse_combined_program(source)
    analyzer = CombinedAnalyzer()
    state = analyzer.analyze(stmts)
    targets = state.graph.get_var_targets(var_name)
    result = INTERVAL_BOT
    visited = set()

    def collect(node):
        nonlocal result
        if node.id in visited:
            return
        visited.add(node.id)
        result = interval_join(result, state.get_node_data(node))
        for nxt, tv in state.graph.get_next_targets(node).items():
            if tv != TV.FALSE:
                collect(nxt)

    for node, tv in targets.items():
        if tv != TV.FALSE:
            collect(node)
    return result


def check_sorted(source, var_name):
    """Check if list from var_name is sorted. Returns TV value."""
    stmts = parse_combined_program(source)
    analyzer = CombinedAnalyzer()
    state = analyzer.analyze(stmts)
    return check_sorted_property(state, var_name)


def verify_combined(source, max_iterations=20):
    """Verify all assertions in a combined program. Returns CombinedResult."""
    return analyze_combined(source, max_iterations)
