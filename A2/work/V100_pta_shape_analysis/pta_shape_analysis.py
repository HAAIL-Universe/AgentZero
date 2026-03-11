"""
V100: Points-To-Guided Shape Analysis

Composes V097 (context-sensitive points-to analysis) with V030 (shape analysis)
for more precise heap reasoning on C10 programs with arrays, hashes, and closures.

Key insight: V097 provides alias information (which pointers may/must point to the
same heap location). V030 provides shape properties (acyclicity, sharing, reachability).
The composition uses V097's alias info to guide V030's strong/weak update decisions,
producing strictly more precise results than either analysis alone.

Architecture:
1. Parse C10 source using C043 parser (arrays + hashes)
2. Run V097 points-to analysis -> HeapLoc allocation sites, alias info
3. Extract heap operations from AST -> V030-style statements
4. Run shape analysis guided by PTA: must-alias -> strong update, may-alias -> weak update
5. Combined results: shape properties + alias-refined warnings

Dependencies:
- V097 (context_sensitive_pta) - points-to analysis for C10
- V030 (shape_analysis) - TVLA-style shape analysis
- C043 parser (arrays, hashes, closures)
"""

import sys
import os
from dataclasses import dataclass, field as dc_field
from enum import Enum, auto
from typing import (
    Dict, Set, List, Optional, Tuple, FrozenSet, Any
)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V097_points_to_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V030_shape_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C043_hash_maps'))

from points_to_analysis import (
    analyze_points_to, analyze_flow_sensitive, check_alias, check_may_alias,
    analyze_escapes, analyze_mod_ref, HeapLoc, AllocKind, PointsToState,
    PointsToResult, AliasResult, AbstractPtr, Constraint, ConstraintKind,
)
from shape_analysis import TV, Node, ShapeGraph, ShapeResult, AnalysisVerdict

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HeapOpKind(Enum):
    """Kinds of heap operations extracted from C10 AST."""
    ALLOC = auto()       # x = []; x = {}; x = new obj
    ASSIGN = auto()      # x = y
    FIELD_LOAD = auto()  # x = y.f  or  x = y[i]
    FIELD_STORE = auto() # x.f = y  or  x[i] = y
    NULL_ASSIGN = auto() # x = null
    CALL = auto()        # x = f(...)
    RETURN = auto()      # return x


@dataclass
class HeapOp:
    """A heap operation extracted from C10 AST."""
    kind: HeapOpKind
    lhs: str = ""
    rhs: str = ""
    field: str = ""
    alloc_kind: str = ""
    line: int = 0
    args: List[str] = dc_field(default_factory=list)


class ShapeProperty(Enum):
    """Shape properties that can be checked."""
    ACYCLIC = auto()
    NOT_NULL = auto()
    SHARED = auto()
    REACHABLE = auto()
    DISJOINT = auto()
    LIST_SEGMENT = auto()
    TREE = auto()


@dataclass
class ShapeWarning:
    """A warning from shape analysis."""
    kind: str  # 'NULL_DEREF', 'CYCLE', 'SHARING', 'LEAK', 'ALIAS_CONFLICT'
    message: str
    line: int = 0
    severity: str = "warning"  # 'warning' or 'error'


@dataclass
class PTAShapeNode:
    """A node in the PTA-guided shape graph.
    Maps V097 HeapLoc to V030 Node with enriched metadata."""
    node: Node
    heap_loc: Optional[HeapLoc] = None
    alloc_kind: str = ""
    variables: Set[str] = dc_field(default_factory=set)


@dataclass
class PTAShapeGraph:
    """Shape graph enriched with points-to information."""
    nodes: Dict[int, PTAShapeNode]  # node_id -> PTAShapeNode
    var_points: Dict[str, Dict[int, TV]]  # var -> {node_id: TV}
    field_edges: Dict[int, Dict[str, Dict[int, TV]]]  # src_id -> {field -> {dst_id: TV}}
    _next_id: int = 0

    def fresh_node(self, summary: bool = False,
                   heap_loc: Optional[HeapLoc] = None,
                   alloc_kind: str = "") -> PTAShapeNode:
        nid = self._next_id
        self._next_id += 1
        node = Node(nid, summary=summary)
        ps_node = PTAShapeNode(node=node, heap_loc=heap_loc, alloc_kind=alloc_kind)
        self.nodes[nid] = ps_node
        self.field_edges[nid] = {}
        return ps_node

    def set_var(self, var: str, node_id: int, val: TV = TV.TRUE):
        if var not in self.var_points:
            self.var_points[var] = {}
        self.var_points[var][node_id] = val

    def clear_var(self, var: str):
        self.var_points[var] = {}

    def set_field_edge(self, src_id: int, field: str, dst_id: int, val: TV = TV.TRUE):
        if src_id not in self.field_edges:
            self.field_edges[src_id] = {}
        if field not in self.field_edges[src_id]:
            self.field_edges[src_id][field] = {}
        self.field_edges[src_id][field][dst_id] = val

    def get_var_targets(self, var: str) -> Dict[int, TV]:
        return dict(self.var_points.get(var, {}))

    def get_field_targets(self, node_id: int, fname: str) -> Dict[int, TV]:
        return dict(self.field_edges.get(node_id, {}).get(fname, {}))

    def get_all_field_targets(self, node_id: int) -> Dict[str, Dict[int, TV]]:
        return {f: dict(targets) for f, targets in self.field_edges.get(node_id, {}).items()}

    def remove_node(self, node_id: int):
        self.nodes.pop(node_id, None)
        self.field_edges.pop(node_id, None)
        for var in list(self.var_points):
            self.var_points[var].pop(node_id, None)
        for src in list(self.field_edges):
            for fname in list(self.field_edges[src]):
                self.field_edges[src][fname].pop(node_id, None)

    def is_null(self, var: str) -> TV:
        targets = self.get_var_targets(var)
        if not targets:
            return TV.TRUE
        if all(v == TV.FALSE for v in targets.values()):
            return TV.TRUE
        if any(v == TV.TRUE for v in targets.values()):
            return TV.FALSE
        return TV.MAYBE

    def is_shared(self, node_id: int) -> TV:
        """Check if node has >1 incoming field edge."""
        incoming = 0
        maybe_incoming = 0
        for src in self.field_edges:
            for fname in self.field_edges[src]:
                v = self.field_edges[src][fname].get(node_id)
                if v == TV.TRUE:
                    incoming += 1
                elif v == TV.MAYBE:
                    maybe_incoming += 1
        # Also count variable pointings
        for var in self.var_points:
            v = self.var_points[var].get(node_id)
            if v == TV.TRUE:
                incoming += 1
            elif v == TV.MAYBE:
                maybe_incoming += 1
        if incoming > 1:
            return TV.TRUE
        if incoming + maybe_incoming > 1:
            return TV.MAYBE
        return TV.FALSE

    def reachable(self, from_id: int, to_id: int, field: str = "next") -> TV:
        """Check reachability via field edges (BFS with 3-valued logic)."""
        if from_id == to_id:
            return TV.TRUE
        visited = set()
        queue = [(from_id, TV.TRUE)]
        best = TV.FALSE
        while queue:
            cur, cur_val = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            targets = self.get_field_targets(cur, field)
            for t, tv in targets.items():
                if tv == TV.FALSE:
                    continue
                edge_val = cur_val & tv
                if t == to_id:
                    best = best | edge_val
                    if best == TV.TRUE:
                        return TV.TRUE
                else:
                    queue.append((t, edge_val))
        return best

    def is_acyclic_from(self, var: str, field: str = "next") -> TV:
        """Check if structure from var is acyclic along field edges."""
        targets = self.get_var_targets(var)
        if not targets:
            return TV.TRUE  # null is acyclic
        result = TV.TRUE
        for nid, tv in targets.items():
            if tv == TV.FALSE:
                continue
            # Check for self-loops and cycles
            cycle_val = self._check_cycle(nid, field)
            if cycle_val == TV.TRUE:
                result = result & ~tv  # definite cycle
            elif cycle_val == TV.MAYBE:
                result = result & TV.MAYBE
        return result

    def _check_cycle(self, start_id: int, field: str) -> TV:
        """Check if there's a cycle starting from start_id."""
        visited = set()
        stack = [(start_id, TV.TRUE)]
        while stack:
            cur, val = stack.pop()
            if cur in visited:
                if cur == start_id:
                    return val
                continue
            visited.add(cur)
            targets = self.get_field_targets(cur, field)
            for t, tv in targets.items():
                if tv == TV.FALSE:
                    continue
                edge_val = val & tv
                if t == start_id:
                    return edge_val
                if t not in visited:
                    stack.append((t, edge_val))
        return TV.FALSE

    def copy(self) -> 'PTAShapeGraph':
        g = PTAShapeGraph(
            nodes={nid: PTAShapeNode(
                node=Node(n.node.id, summary=n.node.summary),
                heap_loc=n.heap_loc,
                alloc_kind=n.alloc_kind,
                variables=set(n.variables),
            ) for nid, n in self.nodes.items()},
            var_points={v: dict(pts) for v, pts in self.var_points.items()},
            field_edges={
                src: {f: dict(tgts) for f, tgts in fields.items()}
                for src, fields in self.field_edges.items()
            },
            _next_id=self._next_id,
        )
        return g

    def join(self, other: 'PTAShapeGraph') -> 'PTAShapeGraph':
        """Least upper bound of two shape graphs."""
        result = self.copy()
        # Add nodes from other
        for nid, on in other.nodes.items():
            if nid not in result.nodes:
                result.nodes[nid] = PTAShapeNode(
                    node=Node(on.node.id, summary=on.node.summary),
                    heap_loc=on.heap_loc,
                    alloc_kind=on.alloc_kind,
                    variables=set(on.variables),
                )
                result.field_edges[nid] = {}
        # Join var points
        for var, pts in other.var_points.items():
            if var not in result.var_points:
                result.var_points[var] = {}
            for nid, tv in pts.items():
                old = result.var_points[var].get(nid, TV.FALSE)
                result.var_points[var][nid] = old | tv
        # Join field edges
        for src, fields in other.field_edges.items():
            if src not in result.field_edges:
                result.field_edges[src] = {}
            for fname, tgts in fields.items():
                if fname not in result.field_edges[src]:
                    result.field_edges[src][fname] = {}
                for dst, tv in tgts.items():
                    old = result.field_edges[src][fname].get(dst, TV.FALSE)
                    result.field_edges[src][fname][dst] = old | tv
        result._next_id = max(result._next_id, other._next_id)
        return result

    def equals(self, other: 'PTAShapeGraph') -> bool:
        if set(self.nodes.keys()) != set(other.nodes.keys()):
            return False
        if self.var_points != other.var_points:
            return False
        if self.field_edges != other.field_edges:
            return False
        return True

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        count = 0
        for src in self.field_edges:
            for fname in self.field_edges[src]:
                count += len([v for v in self.field_edges[src][fname].values() if v != TV.FALSE])
        return count


@dataclass
class PTAShapeResult:
    """Combined result of PTA-guided shape analysis."""
    verdict: AnalysisVerdict
    shape_graph: PTAShapeGraph
    warnings: List[ShapeWarning]
    properties: Dict[str, Dict[str, TV]]  # var -> {property -> TV}
    pta_result: Optional[PointsToResult] = None
    alias_info: Dict[Tuple[str, str], AliasResult] = dc_field(default_factory=dict)
    stats: Dict[str, Any] = dc_field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return self.verdict == AnalysisVerdict.SAFE

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ---------------------------------------------------------------------------
# AST to HeapOp extraction (from C10/C043 AST)
# ---------------------------------------------------------------------------

def _extract_heap_ops(stmts, ops=None) -> List[HeapOp]:
    """Extract heap operations from C10 AST statements."""
    if ops is None:
        ops = []
    # Handle Block objects
    if hasattr(stmts, 'stmts'):
        stmts = stmts.stmts
    for stmt in stmts:
        cls = stmt.__class__.__name__
        line = getattr(stmt, 'line', 0)

        if cls == 'LetDecl':
            _extract_let(stmt, ops, line)
        elif cls == 'Assign':
            _extract_assign(stmt, ops, line)
        elif cls == 'IndexAssign':
            # x.f = val or x[i] = val
            obj_name = _var_name(stmt.obj)
            idx = _expr_str(stmt.index)
            val_name = _var_name(stmt.value)
            if obj_name and val_name:
                ops.append(HeapOp(
                    kind=HeapOpKind.FIELD_STORE,
                    lhs=obj_name, field=idx, rhs=val_name, line=line
                ))
            elif obj_name:
                ops.append(HeapOp(
                    kind=HeapOpKind.FIELD_STORE,
                    lhs=obj_name, field=idx, rhs="__val", line=line
                ))
        elif cls == 'IfStmt':
            body_ops = _extract_heap_ops(stmt.then_body)
            else_ops = _extract_heap_ops(getattr(stmt, 'else_body', None) or [])
            ops.extend(body_ops)
            ops.extend(else_ops)
        elif cls == 'WhileStmt':
            body_ops = _extract_heap_ops(stmt.body)
            ops.extend(body_ops)
        elif cls == 'FnDecl':
            body_ops = _extract_heap_ops(stmt.body)
            ops.extend(body_ops)
        elif cls == 'ReturnStmt':
            val_name = _var_name(stmt.value) if stmt.value else ""
            ops.append(HeapOp(kind=HeapOpKind.RETURN, lhs=val_name, line=line))
        elif cls == 'ExprStmt':
            _extract_expr_heap_op(stmt.expr, ops, line)

    return ops


def _extract_let(stmt, ops, line):
    cls = stmt.value.__class__.__name__ if stmt.value else ""
    if cls == 'ArrayLit':
        ops.append(HeapOp(kind=HeapOpKind.ALLOC, lhs=stmt.name,
                          alloc_kind='array', line=line))
        # Extract element stores
        for i, elem in enumerate(stmt.value.elements):
            ename = _var_name(elem)
            if ename:
                ops.append(HeapOp(kind=HeapOpKind.FIELD_STORE,
                                  lhs=stmt.name, field=str(i), rhs=ename, line=line))
    elif cls == 'HashLit':
        ops.append(HeapOp(kind=HeapOpKind.ALLOC, lhs=stmt.name,
                          alloc_kind='hash', line=line))
        for pair in stmt.value.pairs:
            if isinstance(pair, tuple) and len(pair) == 2:
                k, v = pair
                fname = _expr_str(k)
                vname = _var_name(v)
                if vname:
                    ops.append(HeapOp(kind=HeapOpKind.FIELD_STORE,
                                      lhs=stmt.name, field=fname, rhs=vname, line=line))
    elif cls == 'CallExpr':
        callee = _var_name(stmt.value.callee)
        arg_names = [_var_name(a) or "__arg" for a in stmt.value.args]
        ops.append(HeapOp(kind=HeapOpKind.CALL, lhs=stmt.name,
                          rhs=callee or "__fn", args=arg_names, line=line))
    elif cls == 'IndexExpr':
        # x = y.f or x = y[i]
        obj_name = _var_name(stmt.value.obj)
        idx = _expr_str(stmt.value.index)
        if obj_name:
            ops.append(HeapOp(kind=HeapOpKind.FIELD_LOAD,
                              lhs=stmt.name, rhs=obj_name, field=idx, line=line))
        else:
            ops.append(HeapOp(kind=HeapOpKind.ASSIGN, lhs=stmt.name,
                              rhs="__unknown", line=line))
    elif cls == 'NullLit':
        ops.append(HeapOp(kind=HeapOpKind.NULL_ASSIGN, lhs=stmt.name, line=line))
    elif cls == 'Var':
        if stmt.value.name == 'null':
            ops.append(HeapOp(kind=HeapOpKind.NULL_ASSIGN, lhs=stmt.name, line=line))
        else:
            ops.append(HeapOp(kind=HeapOpKind.ASSIGN, lhs=stmt.name,
                              rhs=stmt.value.name, line=line))
    elif cls in ('FnDecl', 'LambdaExpr'):
        ops.append(HeapOp(kind=HeapOpKind.ALLOC, lhs=stmt.name,
                          alloc_kind='closure', line=line))
    else:
        # Scalar or computed -- treat as opaque assignment
        ops.append(HeapOp(kind=HeapOpKind.ASSIGN, lhs=stmt.name,
                          rhs="__scalar", line=line))
    return ops


def _extract_assign(stmt, ops, line):
    cls = stmt.value.__class__.__name__ if stmt.value else ""
    if cls == 'ArrayLit':
        ops.append(HeapOp(kind=HeapOpKind.ALLOC, lhs=stmt.name,
                          alloc_kind='array', line=line))
    elif cls == 'HashLit':
        ops.append(HeapOp(kind=HeapOpKind.ALLOC, lhs=stmt.name,
                          alloc_kind='hash', line=line))
    elif cls == 'IndexExpr':
        obj_name = _var_name(stmt.value.obj)
        idx = _expr_str(stmt.value.index)
        if obj_name:
            ops.append(HeapOp(kind=HeapOpKind.FIELD_LOAD,
                              lhs=stmt.name, rhs=obj_name, field=idx, line=line))
    elif cls == 'NullLit':
        ops.append(HeapOp(kind=HeapOpKind.NULL_ASSIGN, lhs=stmt.name, line=line))
    elif cls == 'Var':
        if stmt.value.name == 'null':
            ops.append(HeapOp(kind=HeapOpKind.NULL_ASSIGN, lhs=stmt.name, line=line))
        else:
            ops.append(HeapOp(kind=HeapOpKind.ASSIGN, lhs=stmt.name,
                              rhs=stmt.value.name, line=line))
    elif cls == 'CallExpr':
        callee = _var_name(stmt.value.callee)
        arg_names = [_var_name(a) or "__arg" for a in stmt.value.args]
        ops.append(HeapOp(kind=HeapOpKind.CALL, lhs=stmt.name,
                          rhs=callee or "__fn", args=arg_names, line=line))
    else:
        ops.append(HeapOp(kind=HeapOpKind.ASSIGN, lhs=stmt.name,
                          rhs="__scalar", line=line))


def _extract_expr_heap_op(expr, ops, line):
    """Extract heap ops from expression statements (e.g., function calls)."""
    cls = expr.__class__.__name__
    if cls == 'CallExpr':
        callee = _var_name(expr.callee)
        arg_names = [_var_name(a) or "__arg" for a in expr.args]
        ops.append(HeapOp(kind=HeapOpKind.CALL, lhs="__discard",
                          rhs=callee or "__fn", args=arg_names, line=line))


def _var_name(node) -> Optional[str]:
    if node is None:
        return None
    cls = node.__class__.__name__
    if cls == 'Var':
        return node.name
    if cls == 'StringLit':
        return None  # string literal, not a variable
    return None


def _expr_str(node) -> str:
    if node is None:
        return "?"
    cls = node.__class__.__name__
    if cls == 'Var':
        return node.name
    if cls == 'IntLit':
        return str(node.value)
    if cls == 'StringLit':
        return node.value
    return "?"


# ---------------------------------------------------------------------------
# PTA-Guided Shape Analyzer
# ---------------------------------------------------------------------------

class PTAShapeAnalyzer:
    """Shape analysis guided by points-to analysis results."""

    def __init__(self, pta_result: Optional[PointsToResult] = None,
                 max_iterations: int = 20):
        self.pta = pta_result
        self.max_iterations = max_iterations
        self.warnings: List[ShapeWarning] = []
        self.alias_cache: Dict[Tuple[str, str], AliasResult] = {}
        self.graph = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        # Map HeapLoc -> node_id
        self._loc_to_node: Dict[str, int] = {}

    def _must_alias(self, var1: str, var2: str) -> bool:
        """Check if var1 and var2 must alias (via PTA)."""
        if var1 == var2:
            return True
        if self.pta is None:
            return False
        key = (min(var1, var2), max(var1, var2))
        if key not in self.alias_cache:
            result = self.pta.alias(var1, var2)
            self.alias_cache[key] = result
        return self.alias_cache[key].must_alias

    def _may_alias(self, var1: str, var2: str) -> bool:
        """Check if var1 and var2 may alias (via PTA)."""
        if var1 == var2:
            return True
        if self.pta is None:
            return True  # conservative
        key = (min(var1, var2), max(var1, var2))
        if key not in self.alias_cache:
            result = self.pta.alias(var1, var2)
            self.alias_cache[key] = result
        return self.alias_cache[key].may_alias

    def _get_pta_targets(self, var: str) -> Set[str]:
        """Get PTA allocation sites for variable."""
        if self.pta is None:
            return set()
        pts = self.pta.points_to(var)
        return {loc.site_id for loc in pts}

    def _ensure_node_for_loc(self, loc_id: str, alloc_kind: str = "") -> int:
        """Get or create a shape node for a PTA allocation site."""
        if loc_id not in self._loc_to_node:
            ps_node = self.graph.fresh_node(summary=False, alloc_kind=alloc_kind)
            self._loc_to_node[loc_id] = ps_node.node.id
        return self._loc_to_node[loc_id]

    def analyze_ops(self, ops: List[HeapOp]) -> PTAShapeGraph:
        """Analyze a sequence of heap operations."""
        for op in ops:
            self._process_op(op)
        return self.graph

    def _process_op(self, op: HeapOp):
        if op.kind == HeapOpKind.ALLOC:
            self._do_alloc(op)
        elif op.kind == HeapOpKind.ASSIGN:
            self._do_assign(op)
        elif op.kind == HeapOpKind.FIELD_LOAD:
            self._do_field_load(op)
        elif op.kind == HeapOpKind.FIELD_STORE:
            self._do_field_store(op)
        elif op.kind == HeapOpKind.NULL_ASSIGN:
            self._do_null(op)
        elif op.kind == HeapOpKind.CALL:
            self._do_call(op)

    def _do_alloc(self, op: HeapOp):
        """x = new allocation (array, hash, closure)."""
        # Check if PTA knows about this allocation
        pta_targets = self._get_pta_targets(op.lhs)
        if pta_targets:
            # Use PTA allocation sites
            self.graph.clear_var(op.lhs)
            for loc_id in pta_targets:
                nid = self._ensure_node_for_loc(loc_id, op.alloc_kind)
                self.graph.set_var(op.lhs, nid, TV.TRUE)
                self.graph.nodes[nid].variables.add(op.lhs)
        else:
            # No PTA info -- create fresh node
            ps_node = self.graph.fresh_node(summary=False, alloc_kind=op.alloc_kind)
            self.graph.clear_var(op.lhs)
            self.graph.set_var(op.lhs, ps_node.node.id, TV.TRUE)
            ps_node.variables.add(op.lhs)

    def _do_assign(self, op: HeapOp):
        """x = y (pointer copy)."""
        if op.rhs.startswith("__"):
            return  # skip opaque assignments
        rhs_targets = self.graph.get_var_targets(op.rhs)
        self.graph.clear_var(op.lhs)
        for nid, tv in rhs_targets.items():
            self.graph.set_var(op.lhs, nid, tv)
            if nid in self.graph.nodes:
                self.graph.nodes[nid].variables.add(op.lhs)

    def _do_field_load(self, op: HeapOp):
        """x = y.f (field read)."""
        rhs_targets = self.graph.get_var_targets(op.rhs)
        if not rhs_targets:
            # Only warn if variable was explicitly set to null (exists in var_points)
            if op.rhs in self.graph.var_points:
                self.warnings.append(ShapeWarning(
                    kind='NULL_DEREF',
                    message=f"Possible null dereference: {op.rhs}.{op.field}",
                    line=op.line, severity='error',
                ))
            return

        self.graph.clear_var(op.lhs)
        for nid, tv in rhs_targets.items():
            if tv == TV.FALSE:
                continue
            # Check null deref
            if tv == TV.MAYBE:
                self.warnings.append(ShapeWarning(
                    kind='NULL_DEREF',
                    message=f"Possible null dereference: {op.rhs}.{op.field} (maybe-null pointer)",
                    line=op.line, severity='warning',
                ))
            field_targets = self.graph.get_field_targets(nid, op.field)
            for fnid, ftv in field_targets.items():
                combined = tv & ftv
                if combined != TV.FALSE:
                    old = self.graph.var_points.get(op.lhs, {}).get(fnid, TV.FALSE)
                    self.graph.set_var(op.lhs, fnid, old | combined)

    def _do_field_store(self, op: HeapOp):
        """x.f = y (field write). PTA guides strong vs weak update."""
        lhs_targets = self.graph.get_var_targets(op.lhs)
        if not lhs_targets:
            # Only warn if variable was explicitly set to null
            if op.lhs in self.graph.var_points:
                self.warnings.append(ShapeWarning(
                    kind='NULL_DEREF',
                    message=f"Possible null dereference: {op.lhs}.{op.field} = ...",
                    line=op.line, severity='error',
                ))
            return

        rhs_targets = self.graph.get_var_targets(op.rhs) if not op.rhs.startswith("__") else {}
        if not rhs_targets and not op.rhs.startswith("__"):
            # RHS is not a pointer, might be a scalar -- create a value node
            val_node = self.graph.fresh_node(summary=False, alloc_kind='value')
            rhs_targets = {val_node.node.id: TV.TRUE}

        # Determine strong vs weak update using PTA
        definite_targets = [nid for nid, tv in lhs_targets.items() if tv == TV.TRUE]
        can_strong_update = len(definite_targets) == 1 and len(lhs_targets) == 1

        # Additional PTA check: if other vars must-alias lhs, still strong update
        if not can_strong_update and self.pta is not None:
            # If PTA says lhs points to exactly one location, strong update
            pta_targets = self._get_pta_targets(op.lhs)
            if len(pta_targets) == 1:
                can_strong_update = len(definite_targets) == 1

        for nid, tv in lhs_targets.items():
            if tv == TV.FALSE:
                continue
            if can_strong_update and tv == TV.TRUE:
                # Strong update: overwrite field
                self.graph.field_edges[nid][op.field] = {}
                for rnid, rtv in rhs_targets.items():
                    self.graph.set_field_edge(nid, op.field, rnid, rtv)
            else:
                # Weak update: add without removing existing
                for rnid, rtv in rhs_targets.items():
                    combined = tv & rtv
                    if combined != TV.FALSE:
                        old = self.graph.field_edges.get(nid, {}).get(op.field, {}).get(rnid, TV.FALSE)
                        self.graph.set_field_edge(nid, op.field, rnid, old | combined)

    def _do_null(self, op: HeapOp):
        """x = null."""
        self.graph.clear_var(op.lhs)

    def _do_call(self, op: HeapOp):
        """x = f(...). Use PTA to determine what the return value points to."""
        pta_targets = self._get_pta_targets(op.lhs)
        if pta_targets:
            self.graph.clear_var(op.lhs)
            for loc_id in pta_targets:
                nid = self._ensure_node_for_loc(loc_id, 'object')
                self.graph.set_var(op.lhs, nid, TV.TRUE)
        # Otherwise lhs has unknown shape -- leave as-is

    def get_properties(self, variables: List[str]) -> Dict[str, Dict[str, TV]]:
        """Compute shape properties for given variables."""
        props = {}
        for var in variables:
            var_props = {}
            var_props['is_null'] = self.graph.is_null(var)
            var_props['acyclic'] = self.graph.is_acyclic_from(var)

            targets = self.graph.get_var_targets(var)
            shared = TV.FALSE
            for nid in targets:
                s = self.graph.is_shared(nid)
                shared = shared | s
            var_props['shared'] = shared

            props[var] = var_props
        return props


# ---------------------------------------------------------------------------
# Conservative (no PTA) Shape Analyzer for comparison
# ---------------------------------------------------------------------------

class ConservativeShapeAnalyzer(PTAShapeAnalyzer):
    """Shape analysis without PTA guidance (everything is may-alias)."""
    def __init__(self, max_iterations=20):
        super().__init__(pta_result=None, max_iterations=max_iterations)

    def _must_alias(self, var1, var2):
        return var1 == var2

    def _may_alias(self, var1, var2):
        return True  # always conservative


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_c10(source: str):
    """Parse C10 source using C043 parser."""
    from hash_maps import parse
    ast = parse(source)
    return ast


def analyze_pta_shape(source: str, k: int = 1,
                      flow_sensitive: bool = False) -> PTAShapeResult:
    """Main API: PTA-guided shape analysis on C10 source.

    Args:
        source: C10 source code
        k: context sensitivity depth for PTA (0=insensitive, 1+=sensitive)
        flow_sensitive: use flow-sensitive PTA

    Returns:
        PTAShapeResult with shape graph, warnings, and properties
    """
    # Step 1: Parse
    ast = _parse_c10(source)

    # Step 2: Run PTA
    if flow_sensitive:
        pta_result = analyze_flow_sensitive(source, k=k)
    else:
        pta_result = analyze_points_to(source, k=k)

    # Step 3: Extract heap operations
    ops = _extract_heap_ops(ast.stmts)

    # Step 4: Run PTA-guided shape analysis
    analyzer = PTAShapeAnalyzer(pta_result=pta_result)
    graph = analyzer.analyze_ops(ops)

    # Step 5: Compute properties for all variables
    all_vars = list(graph.var_points.keys())
    properties = analyzer.get_properties(all_vars)

    # Step 6: Determine verdict
    errors = [w for w in analyzer.warnings if w.severity == 'error']
    maybes = [w for w in analyzer.warnings if w.severity == 'warning']
    if errors:
        verdict = AnalysisVerdict.UNSAFE
    elif maybes:
        verdict = AnalysisVerdict.MAYBE
    else:
        verdict = AnalysisVerdict.SAFE

    return PTAShapeResult(
        verdict=verdict,
        shape_graph=graph,
        warnings=analyzer.warnings,
        properties=properties,
        pta_result=pta_result,
        alias_info=analyzer.alias_cache,
        stats={
            'nodes': graph.node_count(),
            'edges': graph.edge_count(),
            'variables': len(all_vars),
            'warnings': len(analyzer.warnings),
            'pta_alloc_sites': len(pta_result.alloc_sites),
            'pta_iterations': pta_result.iterations,
            'pta_k': k,
        },
    )


def analyze_conservative(source: str) -> PTAShapeResult:
    """Shape analysis without PTA guidance (for comparison)."""
    ast = _parse_c10(source)
    ops = _extract_heap_ops(ast.stmts)

    analyzer = ConservativeShapeAnalyzer()
    graph = analyzer.analyze_ops(ops)

    all_vars = list(graph.var_points.keys())
    properties = analyzer.get_properties(all_vars)

    errors = [w for w in analyzer.warnings if w.severity == 'error']
    maybes = [w for w in analyzer.warnings if w.severity == 'warning']
    if errors:
        verdict = AnalysisVerdict.UNSAFE
    elif maybes:
        verdict = AnalysisVerdict.MAYBE
    else:
        verdict = AnalysisVerdict.SAFE

    return PTAShapeResult(
        verdict=verdict,
        shape_graph=graph,
        warnings=analyzer.warnings,
        properties=properties,
        stats={
            'nodes': graph.node_count(),
            'edges': graph.edge_count(),
            'variables': len(all_vars),
            'warnings': len(analyzer.warnings),
        },
    )


def check_acyclic(source: str, var: str, k: int = 1) -> TV:
    """Check if the structure rooted at var is acyclic."""
    result = analyze_pta_shape(source, k=k)
    return result.properties.get(var, {}).get('acyclic', TV.MAYBE)


def check_not_null(source: str, var: str, k: int = 1) -> TV:
    """Check if var is definitely not null."""
    result = analyze_pta_shape(source, k=k)
    null_val = result.properties.get(var, {}).get('is_null', TV.MAYBE)
    return ~null_val  # NOT null


def check_shared(source: str, var: str, k: int = 1) -> TV:
    """Check if structure at var has sharing (nodes pointed to by >1 pointer)."""
    result = analyze_pta_shape(source, k=k)
    return result.properties.get(var, {}).get('shared', TV.MAYBE)


def check_disjoint(source: str, var1: str, var2: str, k: int = 1) -> TV:
    """Check if structures at var1 and var2 are disjoint (no shared nodes)."""
    result = analyze_pta_shape(source, k=k)
    g = result.shape_graph
    targets1 = set(nid for nid, tv in g.get_var_targets(var1).items() if tv != TV.FALSE)
    targets2 = set(nid for nid, tv in g.get_var_targets(var2).items() if tv != TV.FALSE)

    # Check for overlap including reachable nodes
    reach1 = _reachable_set(g, targets1)
    reach2 = _reachable_set(g, targets2)

    overlap = reach1 & reach2
    if not overlap:
        return TV.TRUE

    # Check if overlap is definite or maybe
    all_definite = True
    for nid in overlap:
        # Is this node definitely in both?
        in1 = any(g.get_var_targets(var1).get(nid) == TV.TRUE for _ in [1]) or nid in targets1
        in2 = any(g.get_var_targets(var2).get(nid) == TV.TRUE for _ in [1]) or nid in targets2
        if not (in1 and in2):
            all_definite = False
    if all_definite and overlap:
        return TV.FALSE  # definitely not disjoint
    return TV.MAYBE


def _reachable_set(g: PTAShapeGraph, start_ids: Set[int]) -> Set[int]:
    """Compute all nodes reachable from start_ids via any field edges."""
    visited = set()
    queue = list(start_ids)
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        for fname, targets in g.get_all_field_targets(nid).items():
            for tnid, tv in targets.items():
                if tv != TV.FALSE and tnid not in visited:
                    queue.append(tnid)
    return visited


def check_reachable(source: str, from_var: str, to_var: str,
                    field: str = "next", k: int = 1) -> TV:
    """Check if to_var is reachable from from_var via field edges."""
    result = analyze_pta_shape(source, k=k)
    g = result.shape_graph
    from_targets = g.get_var_targets(from_var)
    to_targets = g.get_var_targets(to_var)

    if not from_targets or not to_targets:
        return TV.FALSE

    best = TV.FALSE
    for fnid, ftv in from_targets.items():
        if ftv == TV.FALSE:
            continue
        for tnid, ttv in to_targets.items():
            if ttv == TV.FALSE:
                continue
            r = g.reachable(fnid, tnid, field)
            combined = ftv & ttv & r
            best = best | combined
    return best


def compare_precision(source: str, k: int = 1) -> Dict[str, Any]:
    """Compare PTA-guided vs conservative shape analysis precision."""
    pta_result = analyze_pta_shape(source, k=k)
    cons_result = analyze_conservative(source)

    # Compare properties
    gains = []
    for var in pta_result.properties:
        pta_props = pta_result.properties[var]
        cons_props = cons_result.properties.get(var, {})
        for prop_name in pta_props:
            pta_val = pta_props[prop_name]
            cons_val = cons_props.get(prop_name, TV.MAYBE)
            if pta_val != cons_val:
                gains.append({
                    'variable': var,
                    'property': prop_name,
                    'pta_value': pta_val.name,
                    'conservative_value': cons_val.name,
                    'improved': pta_val != TV.MAYBE and cons_val == TV.MAYBE,
                })

    return {
        'pta_verdict': pta_result.verdict.name,
        'conservative_verdict': cons_result.verdict.name,
        'pta_warnings': len(pta_result.warnings),
        'conservative_warnings': len(cons_result.warnings),
        'pta_nodes': pta_result.stats.get('nodes', 0),
        'conservative_nodes': cons_result.stats.get('nodes', 0),
        'precision_gains': gains,
        'gain_count': sum(1 for g in gains if g['improved']),
    }


def alias_query(source: str, var1: str, var2: str, k: int = 1) -> AliasResult:
    """Query alias relationship between two variables."""
    pta_result = analyze_points_to(source, k=k)
    return pta_result.alias(var1, var2)


def full_pta_shape_analysis(source: str, k: int = 1) -> Dict[str, Any]:
    """Full analysis: PTA + shape + escape + comparison."""
    pta_shape = analyze_pta_shape(source, k=k)
    conservative = analyze_conservative(source)
    comparison = compare_precision(source, k=k)

    # Escape analysis
    escape_result = analyze_escapes(source, k=k)

    return {
        'pta_shape': pta_shape,
        'conservative': conservative,
        'comparison': comparison,
        'escape_info': {
            'escaped': {fn: len(locs) for fn, locs in escape_result.escaped.items()},
            'local': {fn: len(locs) for fn, locs in escape_result.local.items()},
        },
        'verdict': pta_shape.verdict.name,
        'warnings': len(pta_shape.warnings),
        'properties': {
            var: {p: v.name for p, v in props.items()}
            for var, props in pta_shape.properties.items()
        },
    }


def pta_shape_summary(source: str, k: int = 1) -> str:
    """Human-readable summary of PTA-guided shape analysis."""
    result = analyze_pta_shape(source, k=k)
    lines = []
    lines.append(f"PTA-Guided Shape Analysis (k={k})")
    lines.append(f"  Verdict: {result.verdict.name}")
    lines.append(f"  Nodes: {result.stats.get('nodes', 0)}")
    lines.append(f"  Edges: {result.stats.get('edges', 0)}")
    lines.append(f"  Warnings: {len(result.warnings)}")
    lines.append(f"  PTA alloc sites: {result.stats.get('pta_alloc_sites', 0)}")

    if result.warnings:
        lines.append("  Warnings:")
        for w in result.warnings:
            lines.append(f"    [{w.severity}] {w.kind}: {w.message}")

    if result.properties:
        lines.append("  Properties:")
        for var, props in result.properties.items():
            prop_strs = [f"{p}={v.name}" for p, v in props.items()]
            lines.append(f"    {var}: {', '.join(prop_strs)}")

    return '\n'.join(lines)
