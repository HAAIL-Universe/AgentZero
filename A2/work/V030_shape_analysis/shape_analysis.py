"""
V030: Shape Analysis -- TVLA-style heap abstraction with 3-valued logic.

Implements shape analysis for a simple pointer language:
- Variables point to heap nodes or null
- Each node has a single 'next' pointer field
- Analysis tracks list shapes: acyclic, cyclic, shared, etc.

Core concepts:
- 3-valued logic: 1 (true), 0 (false), 1/2 (maybe)
- Shape graphs: nodes (concrete or summary), predicate valuations
- Core predicates: x(v) for pointer vars, n(v1,v2) for next field
- Instrumentation predicates: reachable, shared, cyclic, etc.
- Focus, Coerce, Blur operations for precision
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


# ---------------------------------------------------------------------------
# 3-valued logic
# ---------------------------------------------------------------------------

class TV(Enum):
    """Three-valued logic: TRUE (1), FALSE (0), MAYBE (1/2)."""
    TRUE = 1
    FALSE = 0
    MAYBE = 0.5

    def __and__(self, other):
        if self == TV.FALSE or other == TV.FALSE:
            return TV.FALSE
        if self == TV.TRUE and other == TV.TRUE:
            return TV.TRUE
        return TV.MAYBE

    def __or__(self, other):
        if self == TV.TRUE or other == TV.TRUE:
            return TV.TRUE
        if self == TV.FALSE and other == TV.FALSE:
            return TV.FALSE
        return TV.MAYBE

    def __invert__(self):
        if self == TV.TRUE:
            return TV.FALSE
        if self == TV.FALSE:
            return TV.TRUE
        return TV.MAYBE

    def join(self, other):
        """Least upper bound in the information order: 1/2 < {0, 1}."""
        if self == other:
            return self
        return TV.MAYBE

    def __repr__(self):
        return {TV.TRUE: '1', TV.FALSE: '0', TV.MAYBE: '1/2'}[self]


def tv_join(a, b):
    return a.join(b)


# ---------------------------------------------------------------------------
# Heap language AST
# ---------------------------------------------------------------------------

class StmtKind(Enum):
    NEW = 'new'           # x = new()
    ASSIGN = 'assign'     # x = y
    LOAD = 'load'         # x = y.next
    STORE = 'store'       # x.next = y
    NULL = 'null'          # x = null
    IF = 'if'
    WHILE = 'while'
    ASSERT_ACYCLIC = 'assert_acyclic'
    ASSERT_REACHABLE = 'assert_reachable'
    ASSERT_NOT_NULL = 'assert_not_null'
    ASSERT_DISJOINT = 'assert_disjoint'
    ASSUME = 'assume'     # assume x != null, assume x == null


@dataclass
class Stmt:
    kind: StmtKind
    lhs: str = ''        # target variable
    rhs: str = ''        # source variable
    body: list = field(default_factory=list)       # for if/while
    else_body: list = field(default_factory=list)   # for if
    cond_var: str = ''   # condition variable for if/while
    cond_null: bool = True  # True: "x == null", False: "x != null"


def parse_heap_program(source: str) -> list:
    """Parse a simple heap language into Stmt list.

    Syntax:
        x = new()
        x = y
        x = y.next
        x.next = y
        x = null
        if (x == null) { ... } else { ... }
        if (x != null) { ... } else { ... }
        while (x != null) { ... }
        assert_acyclic(x)
        assert_reachable(x, y)
        assert_not_null(x)
        assert_disjoint(x, y)
        assume(x == null)
        assume(x != null)
    """
    tokens = _tokenize(source)
    stmts, _ = _parse_stmts(tokens, 0)
    return stmts


def _tokenize(source):
    token_re = re.compile(
        r'(\bif\b|\belse\b|\bwhile\b|\bassert_acyclic\b|\bassert_reachable\b|'
        r'\bassert_not_null\b|\bassert_disjoint\b|\bassume\b|'
        r'\bnew\b|\bnull\b|'
        r'[a-zA-Z_]\w*|!=|==|[=(){}.;,])'
    )
    return token_re.findall(source)


def _parse_stmts(tokens, pos):
    stmts = []
    while pos < len(tokens) and tokens[pos] != '}':
        stmt, pos = _parse_stmt(tokens, pos)
        if stmt:
            stmts.append(stmt)
    return stmts, pos


def _parse_stmt(tokens, pos):
    if pos >= len(tokens):
        return None, pos
    tok = tokens[pos]

    if tok == 'if':
        return _parse_if(tokens, pos)
    if tok == 'while':
        return _parse_while(tokens, pos)
    if tok.startswith('assert_') or tok == 'assume':
        return _parse_assert(tokens, pos)

    # Assignment forms: x = new(), x = null, x = y.next, x = y, x.next = y
    lhs = tokens[pos]
    pos += 1

    if pos < len(tokens) and tokens[pos] == '.':
        # x.next = y  or  x.next = null
        pos += 1  # skip '.'
        assert tokens[pos] == 'next', f"Expected 'next', got '{tokens[pos]}'"
        pos += 1  # skip 'next'
        assert tokens[pos] == '=', f"Expected '=', got '{tokens[pos]}'"
        pos += 1  # skip '='
        rhs = tokens[pos]
        pos += 1
        if rhs == 'null':
            if pos < len(tokens) and tokens[pos] == ';':
                pos += 1
            return Stmt(StmtKind.STORE, lhs=lhs, rhs='null'), pos
        if rhs == 'new':
            # x.next = new() -- not directly supported, rewrite internally
            assert tokens[pos] == '('
            pos += 1
            assert tokens[pos] == ')'
            pos += 1
            if pos < len(tokens) and tokens[pos] == ';':
                pos += 1
            # Return a pair: new node + store. Use special marker.
            return Stmt(StmtKind.STORE, lhs=lhs, rhs='__new__'), pos
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.STORE, lhs=lhs, rhs=rhs), pos

    assert tokens[pos] == '=', f"Expected '=', got '{tokens[pos]}'"
    pos += 1

    if tokens[pos] == 'new':
        pos += 1  # 'new'
        assert tokens[pos] == '('
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.NEW, lhs=lhs), pos

    if tokens[pos] == 'null':
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.NULL, lhs=lhs), pos

    rhs = tokens[pos]
    pos += 1
    if pos < len(tokens) and tokens[pos] == '.':
        pos += 1  # '.'
        assert tokens[pos] == 'next', f"Expected 'next', got '{tokens[pos]}'"
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.LOAD, lhs=lhs, rhs=rhs), pos

    if pos < len(tokens) and tokens[pos] == ';':
        pos += 1
    return Stmt(StmtKind.ASSIGN, lhs=lhs, rhs=rhs), pos


def _parse_if(tokens, pos):
    pos += 1  # 'if'
    assert tokens[pos] == '('
    pos += 1
    var = tokens[pos]
    pos += 1
    op = tokens[pos]
    pos += 1
    assert tokens[pos] == 'null'
    pos += 1
    assert tokens[pos] == ')'
    pos += 1
    cond_null = (op == '==')

    assert tokens[pos] == '{'
    pos += 1
    body, pos = _parse_stmts(tokens, pos)
    assert tokens[pos] == '}'
    pos += 1

    else_body = []
    if pos < len(tokens) and tokens[pos] == 'else':
        pos += 1
        assert tokens[pos] == '{'
        pos += 1
        else_body, pos = _parse_stmts(tokens, pos)
        assert tokens[pos] == '}'
        pos += 1

    return Stmt(StmtKind.IF, cond_var=var, cond_null=cond_null,
                body=body, else_body=else_body), pos


def _parse_while(tokens, pos):
    pos += 1  # 'while'
    assert tokens[pos] == '('
    pos += 1
    var = tokens[pos]
    pos += 1
    op = tokens[pos]
    pos += 1
    assert tokens[pos] == 'null'
    pos += 1
    assert tokens[pos] == ')'
    pos += 1
    cond_null = (op == '==')

    assert tokens[pos] == '{'
    pos += 1
    body, pos = _parse_stmts(tokens, pos)
    assert tokens[pos] == '}'
    pos += 1

    return Stmt(StmtKind.WHILE, cond_var=var, cond_null=cond_null, body=body), pos


def _parse_assert(tokens, pos):
    kind_name = tokens[pos]
    pos += 1
    assert tokens[pos] == '('
    pos += 1

    if kind_name == 'assume':
        var = tokens[pos]
        pos += 1
        op = tokens[pos]
        pos += 1
        assert tokens[pos] == 'null'
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.ASSUME, cond_var=var, cond_null=(op == '==')), pos

    if kind_name == 'assert_acyclic':
        var = tokens[pos]
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.ASSERT_ACYCLIC, lhs=var), pos

    if kind_name == 'assert_not_null':
        var = tokens[pos]
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.ASSERT_NOT_NULL, lhs=var), pos

    if kind_name == 'assert_reachable':
        v1 = tokens[pos]
        pos += 1
        assert tokens[pos] == ','
        pos += 1
        v2 = tokens[pos]
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.ASSERT_REACHABLE, lhs=v1, rhs=v2), pos

    if kind_name == 'assert_disjoint':
        v1 = tokens[pos]
        pos += 1
        assert tokens[pos] == ','
        pos += 1
        v2 = tokens[pos]
        pos += 1
        assert tokens[pos] == ')'
        pos += 1
        if pos < len(tokens) and tokens[pos] == ';':
            pos += 1
        return Stmt(StmtKind.ASSERT_DISJOINT, lhs=v1, rhs=v2), pos

    raise ValueError(f"Unknown assert: {kind_name}")


# ---------------------------------------------------------------------------
# Shape Graph
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A node in the shape graph."""
    id: int
    summary: bool = False  # True = represents multiple concrete nodes

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"n{self.id}{'*' if self.summary else ''}"


class ShapeGraph:
    """3-valued shape graph with core and instrumentation predicates.

    Core predicates:
      - var_points[var_name] -> {node: TV}  (variable x points to node v)
      - next_edge[node] -> {node: TV}       (v1.next = v2)

    Instrumentation predicates (derived, cached):
      - reachable[var_name] -> {node: TV}   (node reachable from var via next*)
      - shared[node] -> TV                  (node has >1 incoming next edges)
      - cycle[node] -> TV                   (node is on a cycle)
      - is_null[var_name] -> TV             (variable is null)
    """

    _next_id = 0

    def __init__(self):
        self.nodes: set = set()
        self.var_points: dict = {}   # var -> {node: TV}
        self.next_edge: dict = {}    # node -> {node: TV}
        self._dirty = True

    def copy(self):
        g = ShapeGraph()
        g.nodes = set(self.nodes)
        g.var_points = {v: dict(m) for v, m in self.var_points.items()}
        g.next_edge = {n: dict(m) for n, m in self.next_edge.items()}
        g._dirty = True
        return g

    def fresh_node(self, summary=False) -> Node:
        ShapeGraph._next_id += 1
        n = Node(ShapeGraph._next_id, summary)
        self.nodes.add(n)
        self.next_edge[n] = {}
        self._dirty = True
        return n

    def set_var(self, var: str, node: Optional[Node], val: TV = TV.TRUE):
        """Set variable pointing predicate."""
        if var not in self.var_points:
            self.var_points[var] = {}
        if node is None:
            return
        self.var_points[var][node] = val
        self._dirty = True

    def clear_var(self, var: str):
        """Variable no longer points to anything (set to null)."""
        self.var_points[var] = {}
        self._dirty = True

    def set_next(self, src: Node, dst: Node, val: TV = TV.TRUE):
        """Set next-pointer edge."""
        if src not in self.next_edge:
            self.next_edge[src] = {}
        self.next_edge[src][dst] = val
        self._dirty = True

    def clear_next(self, src: Node):
        """Remove all next edges from src."""
        self.next_edge[src] = {}
        self._dirty = True

    def get_var_targets(self, var: str) -> dict:
        """Return {node: TV} for variable."""
        return dict(self.var_points.get(var, {}))

    def get_next_targets(self, node: Node) -> dict:
        """Return {node: TV} for next edges from node."""
        return dict(self.next_edge.get(node, {}))

    def remove_node(self, node: Node):
        """Remove a node and all its edges."""
        self.nodes.discard(node)
        self.next_edge.pop(node, None)
        for src in list(self.next_edge):
            self.next_edge[src].pop(node, None)
        for var in self.var_points:
            self.var_points[var].pop(node, None)
        self._dirty = True

    def garbage_collect(self):
        """Remove nodes not reachable from any variable."""
        reachable = set()
        worklist = []
        for var, targets in self.var_points.items():
            for n, tv in targets.items():
                if tv != TV.FALSE and n not in reachable:
                    reachable.add(n)
                    worklist.append(n)
        while worklist:
            n = worklist.pop()
            for dst, tv in self.next_edge.get(n, {}).items():
                if tv != TV.FALSE and dst not in reachable:
                    reachable.add(dst)
                    worklist.append(dst)
        unreachable = self.nodes - reachable
        for n in unreachable:
            self.remove_node(n)

    # -- Instrumentation predicates --

    def is_null(self, var: str) -> TV:
        """Is variable null?"""
        targets = self.var_points.get(var, {})
        has_true = any(v == TV.TRUE for v in targets.values())
        has_maybe = any(v == TV.MAYBE for v in targets.values())
        if has_true:
            return TV.FALSE  # definitely points to something
        if has_maybe:
            return TV.MAYBE
        return TV.TRUE  # no targets = null

    def is_shared(self, node: Node) -> TV:
        """Does node have >1 incoming next edges?"""
        incoming_count = 0
        has_maybe = False
        for src in self.nodes:
            edges = self.next_edge.get(src, {})
            tv = edges.get(node, TV.FALSE)
            if tv == TV.TRUE:
                incoming_count += 1
            elif tv == TV.MAYBE:
                has_maybe = True
        if incoming_count >= 2:
            return TV.TRUE
        if incoming_count == 1 and has_maybe:
            return TV.MAYBE
        if incoming_count == 0 and has_maybe:
            return TV.MAYBE
        return TV.FALSE

    def is_on_cycle(self, node: Node) -> TV:
        """Is node on a cycle via next edges?"""
        # Check if node can reach itself
        return self._reachable_from_node(node, node)

    def reachable_from_var(self, var: str, target: Node) -> TV:
        """Is target reachable from var via next*?"""
        targets = self.var_points.get(var, {})
        result = TV.FALSE
        for start, tv_start in targets.items():
            if tv_start == TV.FALSE:
                continue
            reach = self._reachable_from_node(start, target)
            combined = tv_start & reach
            result = result | combined
        return result

    def _reachable_from_node(self, start: Node, target: Node) -> TV:
        """Is target reachable from start via 1+ next steps?

        For cycle detection (start == target), checks if there's a path
        of length >= 1 from start back to start.
        """
        # BFS from start's successors
        visited = {}  # node -> TV
        worklist = []

        # Seed with start's direct successors (1 step)
        for dst, edge_tv in self.next_edge.get(start, {}).items():
            if edge_tv != TV.FALSE:
                worklist.append((dst, edge_tv))

        while worklist:
            curr, curr_tv = worklist.pop(0)
            if curr_tv == TV.FALSE:
                continue

            if curr == target:
                return curr_tv

            if curr in visited:
                old = visited[curr]
                new_tv = old | curr_tv
                if new_tv == old:
                    continue
                visited[curr] = new_tv
            else:
                visited[curr] = curr_tv

            for dst, edge_tv in self.next_edge.get(curr, {}).items():
                if edge_tv == TV.FALSE:
                    continue
                reach_tv = curr_tv & edge_tv
                if reach_tv != TV.FALSE:
                    worklist.append((dst, reach_tv))

        return TV.FALSE

    def reachable_from_var_general(self, var: str, target: Node) -> TV:
        """Reachable including 0 steps (var points directly to target)."""
        targets = self.var_points.get(var, {})
        tv_direct = targets.get(target, TV.FALSE)

        result = tv_direct
        for start, tv_start in targets.items():
            if tv_start == TV.FALSE or start == target:
                continue
            reach = self._reachable_from_node_general(start, target)
            combined = tv_start & reach
            result = result | combined
        return result

    def _reachable_from_node_general(self, start: Node, target: Node) -> TV:
        """General reachability (1+ next steps)."""
        visited = {}
        worklist = []

        for dst, edge_tv in self.next_edge.get(start, {}).items():
            if edge_tv != TV.FALSE:
                worklist.append((dst, edge_tv))

        while worklist:
            curr, curr_tv = worklist.pop(0)
            if curr_tv == TV.FALSE:
                continue
            if curr == target:
                return curr_tv if curr_tv == TV.TRUE else TV.MAYBE

            if curr in visited:
                old = visited[curr]
                new = old | curr_tv
                if new == old:
                    continue
                visited[curr] = new
            else:
                visited[curr] = curr_tv

            for dst, edge_tv in self.next_edge.get(curr, {}).items():
                if edge_tv == TV.FALSE:
                    continue
                reach_tv = curr_tv & edge_tv
                if reach_tv != TV.FALSE:
                    worklist.append((dst, reach_tv))

        return TV.FALSE

    # -- Join for loop convergence --

    def join(self, other: 'ShapeGraph') -> 'ShapeGraph':
        """Join two shape graphs (least upper bound)."""
        result = ShapeGraph()

        # Collect all nodes from both graphs
        all_nodes = self.nodes | other.nodes
        result.nodes = set(all_nodes)

        # Join variable predicates
        all_vars = set(self.var_points) | set(other.var_points)
        for var in all_vars:
            result.var_points[var] = {}
            m1 = self.var_points.get(var, {})
            m2 = other.var_points.get(var, {})
            all_targets = set(m1) | set(m2)
            for n in all_targets:
                tv1 = m1.get(n, TV.FALSE)
                tv2 = m2.get(n, TV.FALSE)
                result.var_points[var][n] = tv_join(tv1, tv2)

        # Join next predicates
        for n in all_nodes:
            result.next_edge[n] = {}
            m1 = self.next_edge.get(n, {})
            m2 = other.next_edge.get(n, {})
            all_targets = set(m1) | set(m2)
            for dst in all_targets:
                tv1 = m1.get(dst, TV.FALSE)
                tv2 = m2.get(dst, TV.FALSE)
                result.next_edge[n][dst] = tv_join(tv1, tv2)

        return result

    def equals(self, other: 'ShapeGraph') -> bool:
        """Check structural equality."""
        if self.nodes != other.nodes:
            return False
        if set(self.var_points) != set(other.var_points):
            return False
        for var in self.var_points:
            if self.var_points[var] != other.var_points.get(var, {}):
                return False
        for n in self.nodes:
            if self.next_edge.get(n, {}) != other.next_edge.get(n, {}):
                return False
        return True

    # -- Canonicalization --

    def canonicalize(self):
        """Remove FALSE-valued edges and unreachable nodes."""
        for var in list(self.var_points):
            self.var_points[var] = {
                n: tv for n, tv in self.var_points[var].items() if tv != TV.FALSE
            }
        for src in list(self.next_edge):
            self.next_edge[src] = {
                dst: tv for dst, tv in self.next_edge[src].items() if tv != TV.FALSE
            }
        self.garbage_collect()

    def __repr__(self):
        lines = []
        for n in sorted(self.nodes, key=lambda x: x.id):
            lines.append(f"  {n}")
        for var, targets in sorted(self.var_points.items()):
            for n, tv in targets.items():
                if tv != TV.FALSE:
                    lines.append(f"  {var} -> {n} [{tv}]")
        for src in sorted(self.next_edge, key=lambda x: x.id):
            for dst, tv in self.next_edge[src].items():
                if tv != TV.FALSE:
                    lines.append(f"  {src}.next -> {dst} [{tv}]")
        return "ShapeGraph{\n" + "\n".join(lines) + "\n}"


# ---------------------------------------------------------------------------
# Shape Analysis Engine
# ---------------------------------------------------------------------------

class AnalysisVerdict(Enum):
    SAFE = 'safe'
    UNSAFE = 'unsafe'
    MAYBE = 'maybe'


@dataclass
class ShapeWarning:
    kind: str
    message: str
    stmt: Optional[Stmt] = None


@dataclass
class ShapeResult:
    verdict: AnalysisVerdict
    final_graph: ShapeGraph
    warnings: list = field(default_factory=list)
    properties: dict = field(default_factory=dict)


class ShapeAnalyzer:
    """TVLA-style shape analysis with focus/coerce/blur.

    Analyzes programs in the heap language to verify shape properties
    of linked data structures.
    """

    def __init__(self, max_iterations=20):
        self.max_iterations = max_iterations
        self.warnings = []

    def analyze(self, source: str) -> ShapeResult:
        """Analyze a heap program and return shape properties."""
        stmts = parse_heap_program(source)
        graph = ShapeGraph()
        self.warnings = []

        graph = self._interpret_stmts(stmts, graph)
        graph.canonicalize()

        verdict = AnalysisVerdict.SAFE
        for w in self.warnings:
            if 'VIOLATION' in w.kind:
                verdict = AnalysisVerdict.UNSAFE
                break
            if 'MAYBE' in w.kind:
                verdict = AnalysisVerdict.MAYBE

        props = self._compute_properties(graph)

        return ShapeResult(
            verdict=verdict,
            final_graph=graph,
            warnings=list(self.warnings),
            properties=props,
        )

    def _compute_properties(self, graph: ShapeGraph) -> dict:
        """Compute high-level shape properties."""
        props = {}

        # Per-variable properties
        for var in graph.var_points:
            var_props = {}
            var_props['is_null'] = graph.is_null(var)

            # Check if the list from var is acyclic
            has_cycle = TV.FALSE
            for node, tv in graph.var_points.get(var, {}).items():
                if tv == TV.FALSE:
                    continue
                cycle_tv = graph.is_on_cycle(node)
                has_cycle = has_cycle | (tv & cycle_tv)
                # Also check reachable nodes
                for n2 in graph.nodes:
                    reach = graph.reachable_from_var(var, n2)
                    if reach != TV.FALSE:
                        c2 = graph.is_on_cycle(n2)
                        has_cycle = has_cycle | (reach & c2)

            var_props['acyclic'] = ~has_cycle
            var_props['shared'] = TV.FALSE
            for n in graph.nodes:
                reach = graph.reachable_from_var_general(var, n)
                if reach != TV.FALSE:
                    s = graph.is_shared(n)
                    var_props['shared'] = var_props['shared'] | (reach & s)

            props[var] = var_props

        return props

    def _interpret_stmts(self, stmts: list, graph: ShapeGraph) -> ShapeGraph:
        for stmt in stmts:
            graph = self._interpret_stmt(stmt, graph)
        return graph

    def _interpret_stmt(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        if stmt.kind == StmtKind.NEW:
            return self._do_new(stmt, graph)
        if stmt.kind == StmtKind.ASSIGN:
            return self._do_assign(stmt, graph)
        if stmt.kind == StmtKind.LOAD:
            return self._do_load(stmt, graph)
        if stmt.kind == StmtKind.STORE:
            return self._do_store(stmt, graph)
        if stmt.kind == StmtKind.NULL:
            return self._do_null(stmt, graph)
        if stmt.kind == StmtKind.IF:
            return self._do_if(stmt, graph)
        if stmt.kind == StmtKind.WHILE:
            return self._do_while(stmt, graph)
        if stmt.kind == StmtKind.ASSUME:
            return self._do_assume(stmt, graph)
        if stmt.kind in (StmtKind.ASSERT_ACYCLIC, StmtKind.ASSERT_REACHABLE,
                         StmtKind.ASSERT_NOT_NULL, StmtKind.ASSERT_DISJOINT):
            return self._do_assert(stmt, graph)
        return graph

    def _do_new(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """x = new(): create fresh node, point x to it, next = null."""
        g = graph.copy()
        node = g.fresh_node()
        g.clear_var(stmt.lhs)
        g.set_var(stmt.lhs, node, TV.TRUE)
        # next is null (no edges from new node) - already default
        return g

    def _do_assign(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """x = y: x points to whatever y points to."""
        g = graph.copy()
        g.clear_var(stmt.lhs)
        targets = g.get_var_targets(stmt.rhs)
        for node, tv in targets.items():
            g.set_var(stmt.lhs, node, tv)
        return g

    def _do_null(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """x = null: clear x."""
        g = graph.copy()
        g.clear_var(stmt.lhs)
        return g

    def _do_load(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """x = y.next: x points to successors of y's targets.

        With focus: if y points to a summary node, materialize it first.
        """
        g = graph.copy()
        g.clear_var(stmt.lhs)

        y_targets = g.get_var_targets(stmt.rhs)
        if not y_targets:
            # y is null -> x = null.next is null dereference
            self.warnings.append(ShapeWarning(
                'NULL_DEREF', f"Null dereference: {stmt.rhs}.next", stmt))
            return g

        # Focus: materialize summary nodes that y points to
        g = self._focus_var(g, stmt.rhs)
        y_targets = g.get_var_targets(stmt.rhs)

        for y_node, y_tv in y_targets.items():
            if y_tv == TV.FALSE:
                continue
            next_targets = g.get_next_targets(y_node)
            if not next_targets:
                # y_node.next is null -> x gets null contribution
                # (no node to point to, which represents null)
                pass
            for dst, edge_tv in next_targets.items():
                combined = y_tv & edge_tv
                if combined != TV.FALSE:
                    old = g.var_points.get(stmt.lhs, {}).get(dst, TV.FALSE)
                    g.set_var(stmt.lhs, dst, old | combined)

        return g

    def _do_store(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """x.next = y: for each node x points to, set its next to y's targets.

        With focus: if x points to a summary node, materialize first.
        """
        g = graph.copy()

        x_targets = g.get_var_targets(stmt.lhs)
        if not x_targets:
            self.warnings.append(ShapeWarning(
                'NULL_DEREF', f"Null dereference: {stmt.lhs}.next = ...", stmt))
            return g

        # Focus: materialize summary nodes that x points to
        g = self._focus_var(g, stmt.lhs)
        x_targets = g.get_var_targets(stmt.lhs)

        if stmt.rhs == '__new__':
            # x.next = new(): create fresh node, store it
            new_node = g.fresh_node()
            y_targets = {new_node: TV.TRUE}
        elif stmt.rhs == 'null':
            y_targets = {}
        else:
            y_targets = g.get_var_targets(stmt.rhs)

        for x_node, x_tv in x_targets.items():
            if x_tv == TV.FALSE:
                continue
            if x_tv == TV.TRUE:
                # Definitely this node: strong update
                g.clear_next(x_node)
                for y_node, y_tv in y_targets.items():
                    if y_tv != TV.FALSE:
                        g.set_next(x_node, y_node, y_tv)
            else:
                # Maybe this node: weak update (keep old edges as MAYBE)
                old_edges = g.get_next_targets(x_node)
                for dst, tv in old_edges.items():
                    g.next_edge[x_node][dst] = TV.MAYBE
                for y_node, y_tv in y_targets.items():
                    if y_tv != TV.FALSE:
                        old = g.next_edge.get(x_node, {}).get(y_node, TV.FALSE)
                        g.set_next(x_node, y_node, old | TV.MAYBE)

        # Blur: merge nodes that are indistinguishable
        g = self._blur(g)
        return g

    def _do_if(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """if (x == null) {...} else {...}"""
        # Then branch: assume condition
        g_then = self._assume(graph, stmt.cond_var, stmt.cond_null)
        g_then = self._interpret_stmts(stmt.body, g_then)

        # Else branch: assume negation
        g_else = self._assume(graph, stmt.cond_var, not stmt.cond_null)
        if stmt.else_body:
            g_else = self._interpret_stmts(stmt.else_body, g_else)

        return g_then.join(g_else)

    def _do_while(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """while (x != null) {...} -- fixpoint iteration with widening."""
        g = graph.copy()

        for _ in range(self.max_iterations):
            # Assume condition true -> execute body
            g_in = self._assume(g, stmt.cond_var, stmt.cond_null)
            g_body = self._interpret_stmts(stmt.body, g_in)

            # Join with pre-loop state
            g_new = g.join(g_body)

            # Blur for convergence
            g_new = self._blur(g_new)

            if g_new.equals(g):
                break
            g = g_new

        # Exit: assume condition false
        g_exit = self._assume(g, stmt.cond_var, not stmt.cond_null)
        return g_exit

    def _do_assume(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """assume(x == null) or assume(x != null)."""
        return self._assume(graph, stmt.cond_var, stmt.cond_null)

    def _assume(self, graph: ShapeGraph, var: str, is_null: bool) -> ShapeGraph:
        """Refine graph under assumption that var is/isn't null."""
        g = graph.copy()
        targets = g.get_var_targets(var)

        if is_null:
            # var == null: remove all definite pointings
            g.clear_var(var)
        else:
            # var != null: must point to something
            if not targets:
                # Contradiction: var is null but we assume not null
                # Create a summary node representing unknown heap
                node = g.fresh_node(summary=True)
                g.set_var(var, node, TV.MAYBE)
            else:
                # Strengthen MAYBE to stay as-is (can't strengthen without more info)
                # But ensure at least one definite target if possible
                pass

        return g

    def _do_assert(self, stmt: Stmt, graph: ShapeGraph) -> ShapeGraph:
        """Check assertions."""
        if stmt.kind == StmtKind.ASSERT_NOT_NULL:
            null_tv = graph.is_null(stmt.lhs)
            if null_tv == TV.TRUE:
                self.warnings.append(ShapeWarning(
                    'VIOLATION', f"assert_not_null({stmt.lhs}) fails: definitely null", stmt))
            elif null_tv == TV.MAYBE:
                self.warnings.append(ShapeWarning(
                    'MAYBE_VIOLATION', f"assert_not_null({stmt.lhs}) may fail", stmt))

        elif stmt.kind == StmtKind.ASSERT_ACYCLIC:
            var = stmt.lhs
            has_cycle = TV.FALSE
            for n in graph.nodes:
                reach = graph.reachable_from_var_general(var, n)
                if reach != TV.FALSE:
                    c = graph.is_on_cycle(n)
                    has_cycle = has_cycle | (reach & c)
            if has_cycle == TV.TRUE:
                self.warnings.append(ShapeWarning(
                    'VIOLATION', f"assert_acyclic({var}) fails: cycle detected", stmt))
            elif has_cycle == TV.MAYBE:
                self.warnings.append(ShapeWarning(
                    'MAYBE_VIOLATION', f"assert_acyclic({var}) may fail", stmt))

        elif stmt.kind == StmtKind.ASSERT_REACHABLE:
            from_var = stmt.lhs
            to_var = stmt.rhs
            # Check if any node reachable from to_var is reachable from from_var
            to_targets = graph.get_var_targets(to_var)
            reachable_tv = TV.FALSE
            for n, tv in to_targets.items():
                if tv != TV.FALSE:
                    r = graph.reachable_from_var_general(from_var, n)
                    reachable_tv = reachable_tv | (tv & r)
            if reachable_tv == TV.FALSE:
                self.warnings.append(ShapeWarning(
                    'VIOLATION', f"assert_reachable({from_var}, {to_var}) fails", stmt))
            elif reachable_tv == TV.MAYBE:
                self.warnings.append(ShapeWarning(
                    'MAYBE_VIOLATION', f"assert_reachable({from_var}, {to_var}) may fail", stmt))

        elif stmt.kind == StmtKind.ASSERT_DISJOINT:
            v1, v2 = stmt.lhs, stmt.rhs
            # Check if any node is reachable from both
            overlap = TV.FALSE
            for n in graph.nodes:
                r1 = graph.reachable_from_var_general(v1, n)
                r2 = graph.reachable_from_var_general(v2, n)
                overlap = overlap | (r1 & r2)
            if overlap == TV.TRUE:
                self.warnings.append(ShapeWarning(
                    'VIOLATION', f"assert_disjoint({v1}, {v2}) fails: shared nodes", stmt))
            elif overlap == TV.MAYBE:
                self.warnings.append(ShapeWarning(
                    'MAYBE_VIOLATION', f"assert_disjoint({v1}, {v2}) may fail", stmt))

        return graph

    # -- Focus: materialize summary nodes --

    def _focus_var(self, graph: ShapeGraph, var: str) -> ShapeGraph:
        """Focus operation: materialize summary nodes pointed to by var.

        When a variable definitely points to a summary node, split it into:
        - A concrete node (the one var points to)
        - A (possibly empty) summary node (the rest)
        This enables strong updates on the concrete node.
        """
        g = graph.copy()
        targets = g.get_var_targets(var)

        for node, tv in list(targets.items()):
            if node.summary and tv != TV.FALSE:
                g = self._materialize(g, var, node)

        return g

    def _materialize(self, graph: ShapeGraph, var: str, summary: Node) -> ShapeGraph:
        """Materialize a summary node into a concrete node + remaining summary.

        The concrete node inherits the variable pointer and gets MAYBE edges
        to the remaining summary (which may not exist if list has length 1).
        """
        g = graph.copy()

        # Create concrete node
        concrete = g.fresh_node(summary=False)

        # Create remaining summary (may represent 0 or more nodes)
        remaining = g.fresh_node(summary=True)

        # Variable now points to concrete node
        g.var_points[var].pop(summary, None)
        g.set_var(var, concrete, TV.TRUE)

        # Transfer other variable pointers from summary
        for v in list(g.var_points):
            if v == var:
                continue
            ptrs = g.var_points[v]
            if summary in ptrs:
                tv = ptrs[summary]
                # Other vars might point to concrete or remaining
                ptrs[concrete] = tv_join(ptrs.get(concrete, TV.FALSE), TV.MAYBE)
                ptrs[remaining] = tv_join(ptrs.get(remaining, TV.FALSE), TV.MAYBE)

        # Transfer incoming next edges to summary -> split to concrete/remaining
        for src in list(g.nodes):
            if src == summary:
                continue
            edges = g.next_edge.get(src, {})
            if summary in edges:
                tv = edges[summary]
                edges[concrete] = tv_join(edges.get(concrete, TV.FALSE), TV.MAYBE)
                edges[remaining] = tv_join(edges.get(remaining, TV.FALSE), TV.MAYBE)

        # Transfer outgoing next edges from summary
        old_edges = g.next_edge.get(summary, {})
        g.next_edge[concrete] = {}
        for dst, tv in old_edges.items():
            if dst == summary:
                # Self-loop on summary -> concrete points to remaining (MAYBE)
                # and remaining self-loops (MAYBE)
                g.set_next(concrete, remaining, TV.MAYBE)
                g.set_next(remaining, remaining, TV.MAYBE)
            else:
                g.set_next(concrete, dst, tv)
                g.set_next(remaining, dst, tv_join(
                    g.next_edge.get(remaining, {}).get(dst, TV.FALSE), tv))

        # Concrete also might point to remaining
        g.set_next(concrete, remaining, tv_join(
            g.next_edge.get(concrete, {}).get(remaining, TV.FALSE), TV.MAYBE))

        # Remove original summary
        g.nodes.discard(summary)
        g.next_edge.pop(summary, None)
        for v in g.var_points:
            g.var_points[v].pop(summary, None)
        for src in list(g.next_edge):
            g.next_edge[src].pop(summary, None)

        return g

    # -- Blur: merge indistinguishable nodes --

    def _blur(self, graph: ShapeGraph) -> ShapeGraph:
        """Blur operation: merge nodes that no variable points to definitively.

        Nodes not pointed to by any variable (or only pointed to with MAYBE)
        and not distinguished by any instrumentation predicate can be merged
        into a summary node.
        """
        g = graph.copy()

        # Find nodes with no definite variable pointer
        definite_targets = set()
        for var, targets in g.var_points.items():
            for n, tv in targets.items():
                if tv == TV.TRUE:
                    definite_targets.add(n)

        # Nodes that could be blurred
        blur_candidates = [n for n in g.nodes if n not in definite_targets]

        if len(blur_candidates) <= 1:
            return g

        # Group by structural similarity (same in/out edge pattern type)
        # For simplicity: merge all non-pointed-to nodes into one summary
        # More sophisticated: group by predicate valuations

        # Only merge if there are multiple non-distinguished nodes
        to_merge = [n for n in blur_candidates if not n.summary]
        if len(to_merge) < 2:
            return g

        # Check if they're structurally similar enough to merge
        # (conservative: only merge if they have similar edge patterns)
        groups = self._group_for_blur(g, to_merge)

        for group in groups:
            if len(group) < 2:
                continue
            g = self._merge_nodes(g, group)

        return g

    def _group_for_blur(self, graph: ShapeGraph, nodes: list) -> list:
        """Group nodes that can be merged (similar edge signatures)."""
        # Simple grouping: nodes with similar out-degree patterns
        groups = {}
        for n in nodes:
            out_count = len([d for d, tv in graph.next_edge.get(n, {}).items()
                           if tv != TV.FALSE])
            key = out_count
            if key not in groups:
                groups[key] = []
            groups[key].append(n)
        return list(groups.values())

    def _merge_nodes(self, graph: ShapeGraph, nodes: list) -> ShapeGraph:
        """Merge a set of nodes into a single summary node."""
        g = graph.copy()
        summary = g.fresh_node(summary=True)

        node_set = set(nodes)

        # Collect all edges
        for n in nodes:
            # Transfer variable pointers
            for var in list(g.var_points):
                ptrs = g.var_points[var]
                if n in ptrs:
                    old_tv = ptrs.pop(n)
                    ptrs[summary] = tv_join(ptrs.get(summary, TV.FALSE), old_tv)

            # Transfer outgoing edges
            for dst, tv in g.next_edge.get(n, {}).items():
                actual_dst = summary if dst in node_set else dst
                g.set_next(summary, actual_dst, tv_join(
                    g.next_edge.get(summary, {}).get(actual_dst, TV.FALSE), tv))

            # Transfer incoming edges
            for src in list(g.nodes):
                if src in node_set or src == summary:
                    continue
                edges = g.next_edge.get(src, {})
                if n in edges:
                    tv = edges.pop(n)
                    edges[summary] = tv_join(edges.get(summary, TV.FALSE), tv)

            # Remove the original
            g.nodes.discard(n)
            g.next_edge.pop(n, None)

        return g

    # -- Coerce: tighten predicate values --

    def _coerce(self, graph: ShapeGraph) -> ShapeGraph:
        """Coerce operation: tighten MAYBE values using compatibility constraints.

        If a node is not summary and has a definite (TRUE/FALSE) next edge,
        then MAYBE edges that contradict functionality constraints can be
        resolved.
        """
        g = graph.copy()
        changed = True

        while changed:
            changed = False

            for n in list(g.nodes):
                if n.summary:
                    continue

                # Functionality constraint: non-summary node has at most one
                # next successor
                edges = g.next_edge.get(n, {})
                true_targets = [d for d, tv in edges.items() if tv == TV.TRUE]

                if len(true_targets) == 1:
                    # Definite successor: all other edges must be FALSE
                    for dst, tv in list(edges.items()):
                        if dst != true_targets[0] and tv == TV.MAYBE:
                            edges[dst] = TV.FALSE
                            changed = True

                elif len(true_targets) == 0:
                    # Could have 0 or 1 successor
                    maybe_targets = [d for d, tv in edges.items() if tv == TV.MAYBE]
                    if len(maybe_targets) == 0:
                        pass  # null next, fine
                    # Can't resolve further without more info

        g.canonicalize()
        return g


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def analyze_shape(source: str, max_iterations=20) -> ShapeResult:
    """Analyze a heap program for shape properties."""
    analyzer = ShapeAnalyzer(max_iterations=max_iterations)
    return analyzer.analyze(source)


def check_acyclic(source: str, var: str) -> TV:
    """Check if the list rooted at var is acyclic."""
    result = analyze_shape(source)
    props = result.properties.get(var, {})
    return props.get('acyclic', TV.MAYBE)


def check_not_null(source: str, var: str) -> TV:
    """Check if var is definitely not null."""
    result = analyze_shape(source)
    props = result.properties.get(var, {})
    null_tv = props.get('is_null', TV.MAYBE)
    return ~null_tv


def check_shared(source: str, var: str) -> TV:
    """Check if any node reachable from var is shared (multiple incoming)."""
    result = analyze_shape(source)
    props = result.properties.get(var, {})
    return props.get('shared', TV.MAYBE)


def check_reachable(source: str, from_var: str, to_var: str) -> TV:
    """Check if to_var is reachable from from_var."""
    result = analyze_shape(source)
    graph = result.final_graph
    to_targets = graph.get_var_targets(to_var)
    reach = TV.FALSE
    for n, tv in to_targets.items():
        if tv != TV.FALSE:
            r = graph.reachable_from_var_general(from_var, n)
            reach = reach | (tv & r)
    return reach


def verify_shape(source: str) -> ShapeResult:
    """Verify all shape assertions in the program."""
    return analyze_shape(source)


def get_shape_info(source: str) -> dict:
    """Get detailed shape information for all variables."""
    result = analyze_shape(source)
    info = {}
    for var, props in result.properties.items():
        info[var] = {
            'is_null': str(props.get('is_null', TV.MAYBE)),
            'acyclic': str(props.get('acyclic', TV.MAYBE)),
            'shared': str(props.get('shared', TV.MAYBE)),
        }
    return info


def compare_shapes(source1: str, source2: str) -> dict:
    """Compare shape analysis results of two programs."""
    r1 = analyze_shape(source1)
    r2 = analyze_shape(source2)
    return {
        'program1': {
            'verdict': r1.verdict.value,
            'warnings': len(r1.warnings),
            'nodes': len(r1.final_graph.nodes),
            'properties': {v: {k: str(vv) for k, vv in p.items()}
                          for v, p in r1.properties.items()},
        },
        'program2': {
            'verdict': r2.verdict.value,
            'warnings': len(r2.warnings),
            'nodes': len(r2.final_graph.nodes),
            'properties': {v: {k: str(vv) for k, vv in p.items()}
                          for v, p in r2.properties.items()},
        },
    }
