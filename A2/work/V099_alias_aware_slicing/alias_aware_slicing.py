"""V099: Alias-Aware Program Slicing for C10 Programs.

Composes V097 (points-to analysis) + C10 parser (C043) to build precise
program slices that understand heap aliasing.

Standard slicing assumes all heap accesses may alias (conservative).
With points-to information, we can determine that x.f and y.f only
affect each other when x and y may-alias -- producing smaller, more
precise slices.

Key features:
- CFG construction from C10 AST (not Python ast like V037)
- Def-use analysis with heap-awareness (field reads/writes)
- Points-to-guided alias resolution for heap dependences
- Backward/forward/thin/chop slicing with alias precision
- Comparison API: alias-aware vs conservative slicing
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, Set, List, Tuple, Optional, Any, FrozenSet
)
from collections import defaultdict

# --- Imports from composed systems ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V097_points_to_analysis'))
from points_to_analysis import (
    analyze_points_to, check_may_alias, analyze_flow_sensitive,
    PointsToResult, AliasResult, HeapLoc, PointsToState,
)

# C043 parser (arrays + hash maps + closures)
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', 'challenges', 'C043_hash_maps'))
from hash_maps import lex, Parser


# =========================================================================
# Data Structures
# =========================================================================

class DepKind(Enum):
    """Dependence edge kinds."""
    DATA = auto()       # variable def-use
    CONTROL = auto()    # control flow (branch condition -> body)
    HEAP_DATA = auto()  # heap field def-use (through alias)
    CALL = auto()       # caller -> callee entry
    PARAM_IN = auto()   # actual arg -> formal param
    PARAM_OUT = auto()  # return value -> call site result


@dataclass(frozen=True)
class CfgNode:
    """A node in the control flow graph."""
    func: str           # function name or "__main__"
    index: int          # sequential statement index within function
    kind: str           # "stmt", "entry", "exit", "branch", "call", "return"
    line: int = 0       # source line estimate
    label: str = ""     # human-readable label

    def __repr__(self):
        return f"CfgNode({self.func}:{self.index}:{self.kind})"


@dataclass(frozen=True)
class DepEdge:
    """A dependence edge in the PDG/SDG."""
    src: CfgNode
    dst: CfgNode
    kind: DepKind
    var: str = ""       # variable or field name
    heap: bool = False  # True if this is a heap-mediated dependence

    def __repr__(self):
        h = " [heap]" if self.heap else ""
        return f"DepEdge({self.src}->{self.dst} {self.kind.name} '{self.var}'{h})"


@dataclass
class SliceCriterion:
    """Slicing criterion: a program point and variables of interest."""
    node_index: int             # statement index in function
    variables: Set[str] = field(default_factory=set)
    func: str = "__main__"      # function containing the criterion


@dataclass
class SliceResult:
    """Result of a slicing operation."""
    criterion: SliceCriterion
    direction: str              # "backward", "forward", "thin_backward", "chop"
    nodes: Set[CfgNode]         # included nodes
    edges: List[DepEdge]        # relevant dependence edges
    functions_involved: Set[str]
    alias_aware: bool           # whether alias info was used
    conservative_size: int = 0  # size without alias info (for comparison)

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def indices(self) -> Set[int]:
        return {n.index for n in self.nodes}

    @property
    def precision_gain(self) -> float:
        """How much smaller this slice is vs conservative (0.0-1.0)."""
        if self.conservative_size == 0:
            return 0.0
        return 1.0 - (self.size / self.conservative_size)


@dataclass
class PDG:
    """Program Dependence Graph for a single function."""
    func_name: str
    nodes: List[CfgNode]
    edges: List[DepEdge]
    defs: Dict[int, Set[str]]          # index -> variables defined
    uses: Dict[int, Set[str]]          # index -> variables used
    heap_defs: Dict[int, Set[Tuple[str, str]]]  # index -> (base_var, field) written
    heap_uses: Dict[int, Set[Tuple[str, str]]]  # index -> (base_var, field) read

    @property
    def data_edges(self) -> List[DepEdge]:
        return [e for e in self.edges if e.kind in (DepKind.DATA, DepKind.HEAP_DATA)]

    @property
    def control_edges(self) -> List[DepEdge]:
        return [e for e in self.edges if e.kind == DepKind.CONTROL]


@dataclass
class SDG:
    """System Dependence Graph (interprocedural)."""
    pdgs: Dict[str, PDG]
    inter_edges: List[DepEdge]      # CALL, PARAM_IN, PARAM_OUT
    all_nodes: List[CfgNode]
    all_edges: List[DepEdge]

    def get_node(self, func: str, index: int) -> Optional[CfgNode]:
        if func in self.pdgs:
            for n in self.pdgs[func].nodes:
                if n.index == index:
                    return n
        return None


# =========================================================================
# AST Traversal Helpers
# =========================================================================

def _class_name(node):
    return node.__class__.__name__


def _collect_vars_used(node) -> Set[str]:
    """Collect all variables read in an expression."""
    result = set()
    cls = _class_name(node)
    if cls == 'Var':
        result.add(node.name)
    elif cls == 'BinOp':
        result |= _collect_vars_used(node.left)
        result |= _collect_vars_used(node.right)
    elif cls == 'UnaryOp':
        result |= _collect_vars_used(node.operand)
    elif cls == 'CallExpr':
        if hasattr(node, 'callee'):
            if isinstance(node.callee, str):
                result.add(node.callee)
            else:
                result |= _collect_vars_used(node.callee)
        for arg in node.args:
            result |= _collect_vars_used(arg)
    elif cls == 'IndexExpr':
        result |= _collect_vars_used(node.obj)
        result |= _collect_vars_used(node.index)
    elif cls == 'ArrayLit':
        for elem in node.elements:
            result |= _collect_vars_used(elem)
    elif cls == 'HashLit':
        for pair in node.pairs:
            if isinstance(pair, tuple):
                result |= _collect_vars_used(pair[0])
                result |= _collect_vars_used(pair[1])
    elif cls == 'Assign':
        result |= _collect_vars_used(node.value)
    elif cls == 'IfExpr':
        result |= _collect_vars_used(node.cond)
        result |= _collect_vars_used(node.then_expr)
        result |= _collect_vars_used(node.else_expr)
    return result


def _collect_vars_defined(stmt) -> Set[str]:
    """Collect variables defined by a statement."""
    cls = _class_name(stmt)
    if cls == 'LetDecl':
        return {stmt.name}
    elif cls == 'Assign':
        return {stmt.name}
    elif cls == 'IndexAssign':
        return set()  # heap write, not variable def
    return set()


def _collect_heap_reads(node) -> Set[Tuple[str, str]]:
    """Collect (base_var, field) pairs read in an expression."""
    result = set()
    cls = _class_name(node)
    if cls == 'IndexExpr':
        base_cls = _class_name(node.obj)
        if base_cls == 'Var':
            idx_cls = _class_name(node.index)
            if idx_cls == 'StringLit':
                result.add((node.obj.name, node.index.value))
            elif idx_cls == 'IntLit':
                result.add((node.obj.name, str(node.index.value)))
            else:
                result.add((node.obj.name, '*'))
        result |= _collect_heap_reads(node.obj)
        result |= _collect_heap_reads(node.index)
    elif cls == 'BinOp':
        result |= _collect_heap_reads(node.left)
        result |= _collect_heap_reads(node.right)
    elif cls == 'UnaryOp':
        result |= _collect_heap_reads(node.operand)
    elif cls == 'CallExpr':
        for arg in node.args:
            result |= _collect_heap_reads(arg)
    elif cls == 'ArrayLit':
        for elem in node.elements:
            result |= _collect_heap_reads(elem)
    return result


def _collect_heap_writes(stmt) -> Set[Tuple[str, str]]:
    """Collect (base_var, field) pairs written by a statement."""
    cls = _class_name(stmt)
    if cls == 'IndexAssign':
        base_cls = _class_name(stmt.obj)
        if base_cls == 'Var':
            idx_cls = _class_name(stmt.index)
            if idx_cls == 'StringLit':
                return {(stmt.obj.name, stmt.index.value)}
            elif idx_cls == 'IntLit':
                return {(stmt.obj.name, str(stmt.index.value))}
            else:
                return {(stmt.obj.name, '*')}
    return set()


def _stmt_uses(stmt) -> Set[str]:
    """All variables used (read) by a statement."""
    cls = _class_name(stmt)
    if cls == 'LetDecl':
        return _collect_vars_used(stmt.value)
    elif cls == 'Assign':
        return _collect_vars_used(stmt.value)
    elif cls == 'IndexAssign':
        result = _collect_vars_used(stmt.obj)
        result |= _collect_vars_used(stmt.index)
        result |= _collect_vars_used(stmt.value)
        return result
    elif cls == 'ReturnStmt':
        if stmt.value:
            return _collect_vars_used(stmt.value)
        return set()
    elif cls == 'IfStmt':
        return _collect_vars_used(stmt.cond)
    elif cls == 'WhileStmt':
        return _collect_vars_used(stmt.cond)
    elif cls == 'PrintStmt':
        return _collect_vars_used(stmt.value)
    elif cls == 'CallExpr':
        result = set()
        if isinstance(stmt.callee, str):
            result.add(stmt.callee)
        else:
            result |= _collect_vars_used(stmt.callee)
        for arg in stmt.args:
            result |= _collect_vars_used(arg)
        return result
    elif cls in ('FnDecl',):
        return set()
    # Expression statement
    return _collect_vars_used(stmt)


# =========================================================================
# C10 CFG / PDG / SDG Construction
# =========================================================================

def _flatten_stmts(stmts, func_name="__main__"):
    """Flatten C10 AST statements into a list of (index, stmt, nesting_conds).

    Returns list of (index, stmt, parent_cond_indices).
    Recursively enters if/while bodies, tracking which conditions control each stmt.
    """
    flat = []
    idx = [0]  # mutable counter

    def _walk(stmts_list, cond_indices):
        for s in stmts_list:
            cls = _class_name(s)
            my_idx = idx[0]
            idx[0] += 1

            if cls == 'IfStmt':
                flat.append((my_idx, s, list(cond_indices), 'branch'))
                cond_with_me = cond_indices + [my_idx]
                if hasattr(s, 'then_body') and s.then_body:
                    body = s.then_body.stmts if hasattr(s.then_body, 'stmts') else s.then_body
                    _walk(body, cond_with_me)
                if hasattr(s, 'else_body') and s.else_body:
                    body = s.else_body.stmts if hasattr(s.else_body, 'stmts') else s.else_body
                    _walk(body, cond_with_me)
            elif cls == 'WhileStmt':
                flat.append((my_idx, s, list(cond_indices), 'branch'))
                cond_with_me = cond_indices + [my_idx]
                if hasattr(s, 'body') and s.body:
                    body = s.body.stmts if hasattr(s.body, 'stmts') else s.body
                    _walk(body, cond_with_me)
            elif cls == 'FnDecl':
                flat.append((my_idx, s, list(cond_indices), 'fn_decl'))
            else:
                flat.append((my_idx, s, list(cond_indices), 'stmt'))

    _walk(stmts, [])
    return flat


def _extract_functions(program):
    """Extract function declarations from program."""
    fns = {}
    for s in program.stmts:
        if _class_name(s) == 'FnDecl':
            fns[s.name] = s
    return fns


def build_pdg(stmts, func_name="__main__", params=None):
    """Build a Program Dependence Graph for a list of statements."""
    flat = _flatten_stmts(stmts, func_name)

    # Create nodes
    nodes = []
    entry = CfgNode(func=func_name, index=-1, kind="entry", label=f"{func_name}:entry")
    exit_node = CfgNode(func=func_name, index=-2, kind="exit", label=f"{func_name}:exit")
    nodes.append(entry)

    node_map = {}  # index -> CfgNode
    for idx, stmt, cond_idxs, kind in flat:
        cls = _class_name(stmt)
        lbl = f"{func_name}:s{idx}:{cls}"
        n = CfgNode(func=func_name, index=idx, kind=kind, label=lbl)
        nodes.append(n)
        node_map[idx] = n

    nodes.append(exit_node)

    # Compute defs/uses/heap_defs/heap_uses per index
    defs = {}
    uses = {}
    heap_defs = {}
    heap_uses = {}

    # Entry defines parameters
    if params:
        defs[-1] = set(params)
    else:
        defs[-1] = set()
    uses[-1] = set()
    heap_defs[-1] = set()
    heap_uses[-1] = set()

    for idx, stmt, cond_idxs, kind in flat:
        defs[idx] = _collect_vars_defined(stmt)
        uses[idx] = _stmt_uses(stmt)
        heap_defs[idx] = _collect_heap_writes(stmt)
        heap_uses[idx] = _collect_heap_reads(stmt)
        # Also collect heap reads from value of definitions
        cls = _class_name(stmt)
        if cls in ('LetDecl', 'Assign'):
            heap_uses[idx] |= _collect_heap_reads(stmt.value)

    defs[-2] = set()
    uses[-2] = set()
    heap_defs[-2] = set()
    heap_uses[-2] = set()

    edges = []

    # --- Control dependence ---
    for idx, stmt, cond_idxs, kind in flat:
        for c_idx in cond_idxs:
            if c_idx in node_map:
                edges.append(DepEdge(
                    src=node_map[c_idx], dst=node_map[idx],
                    kind=DepKind.CONTROL, var=""
                ))
        if not cond_idxs:
            # Top-level statements are control-dependent on entry
            edges.append(DepEdge(
                src=entry, dst=node_map[idx],
                kind=DepKind.CONTROL, var=""
            ))

    # --- Data dependence (reaching definitions) ---
    # Simple: for each use of variable v at index j, find the nearest
    # prior def of v at index i (same nesting level or enclosing)
    all_indices = [idx for idx, _, _, _ in flat]

    # Build reaching defs: for each index, which previous index defines each var
    # Walk forward, maintaining last def for each variable
    last_def = {}  # var -> index
    if params:
        for p in params:
            last_def[p] = -1  # entry node

    # Simple forward pass (conservative for loops/branches)
    reaching = {}  # (index, var) -> set of defining indices
    for idx, stmt, cond_idxs, kind in flat:
        for v in uses[idx]:
            if v in last_def:
                reaching[(idx, v)] = {last_def[v]}
            else:
                reaching[(idx, v)] = set()
        for v in defs[idx]:
            last_def[v] = idx

    # Build data edges
    for idx, stmt, cond_idxs, kind in flat:
        for v in uses[idx]:
            for def_idx in reaching.get((idx, v), set()):
                src_node = node_map.get(def_idx, entry)
                edges.append(DepEdge(
                    src=src_node, dst=node_map[idx],
                    kind=DepKind.DATA, var=v
                ))

    # --- Heap data dependence (will be refined by alias analysis) ---
    # For now, create HEAP_DATA edges conservatively (all writes to field f
    # potentially reach all reads of field f). Alias analysis prunes these.
    heap_write_indices = {}  # field -> list of (index, base_var)
    for idx, stmt, cond_idxs, kind in flat:
        for base, fld in heap_defs[idx]:
            heap_write_indices.setdefault(fld, []).append((idx, base))

    for idx, stmt, cond_idxs, kind in flat:
        for base_r, fld_r in heap_uses[idx]:
            # Find all writes to this field (or wildcard)
            for fld_pattern in [fld_r, '*']:
                for w_idx, base_w in heap_write_indices.get(fld_pattern, []):
                    if w_idx < idx or (fld_r == '*' or fld_pattern == '*'):
                        edges.append(DepEdge(
                            src=node_map[w_idx], dst=node_map[idx],
                            kind=DepKind.HEAP_DATA, var=fld_r,
                            heap=True
                        ))
            # Also writes with wildcard field
            if fld_r != '*':
                for w_idx, base_w in heap_write_indices.get('*', []):
                    if w_idx < idx:
                        edges.append(DepEdge(
                            src=node_map[w_idx], dst=node_map[idx],
                            kind=DepKind.HEAP_DATA, var=fld_r,
                            heap=True
                        ))

    # Return edge from last statement to exit (if return exists)
    for idx, stmt, cond_idxs, kind in flat:
        cls = _class_name(stmt)
        if cls == 'ReturnStmt':
            edges.append(DepEdge(
                src=node_map[idx], dst=exit_node,
                kind=DepKind.DATA, var="__return__"
            ))

    return PDG(
        func_name=func_name,
        nodes=nodes,
        edges=edges,
        defs=defs,
        uses=uses,
        heap_defs=heap_defs,
        heap_uses=heap_uses,
    )


def build_sdg(source: str) -> SDG:
    """Build a System Dependence Graph from C10 source."""
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    # Extract functions
    fns = _extract_functions(program)

    # Build PDG for main (top-level)
    main_stmts = [s for s in program.stmts if _class_name(s) != 'FnDecl']
    pdgs = {}
    pdgs["__main__"] = build_pdg(main_stmts, "__main__")

    # Build PDG for each function
    for name, fn in fns.items():
        body_stmts = fn.body.stmts if hasattr(fn.body, 'stmts') else fn.body
        pdgs[name] = build_pdg(body_stmts, name, params=fn.params)

    # Build inter-procedural edges
    inter_edges = []
    # For each function, flatten the SAME stmts used to build its PDG
    stmt_sources = {"__main__": main_stmts}
    for name, fn in fns.items():
        stmt_sources[name] = fn.body.stmts if hasattr(fn.body, 'stmts') else fn.body

    for fname, pdg in pdgs.items():
        flat = _flatten_stmts(stmt_sources[fname], fname)
        flat_map = {idx: stmt for idx, stmt, _, _ in flat}
        for node in pdg.nodes:
            if node.index < 0:
                continue
            if node.index in flat_map:
                _add_call_edges(flat_map[node.index], node, fns, pdgs, inter_edges)

    all_nodes = []
    all_edges = []
    for pdg in pdgs.values():
        all_nodes.extend(pdg.nodes)
        all_edges.extend(pdg.edges)
    all_edges.extend(inter_edges)

    return SDG(pdgs=pdgs, inter_edges=inter_edges,
               all_nodes=all_nodes, all_edges=all_edges)


def _add_call_edges(stmt, caller_node, fns, pdgs, inter_edges):
    """Add CALL/PARAM_IN/PARAM_OUT edges for a call expression."""
    cls = _class_name(stmt)
    call_expr = None

    if cls == 'CallExpr':
        call_expr = stmt
    elif cls == 'LetDecl' and _class_name(stmt.value) == 'CallExpr':
        call_expr = stmt.value
    elif cls == 'Assign' and _class_name(stmt.value) == 'CallExpr':
        call_expr = stmt.value

    if call_expr is None:
        return

    if isinstance(call_expr.callee, str):
        callee_name = call_expr.callee
    elif hasattr(call_expr.callee, 'name'):
        callee_name = call_expr.callee.name
    else:
        callee_name = None
    if callee_name and callee_name in fns and callee_name in pdgs:
        callee_pdg = pdgs[callee_name]
        callee_entry = callee_pdg.nodes[0]  # entry node
        callee_exit = callee_pdg.nodes[-1]  # exit node

        # CALL edge
        inter_edges.append(DepEdge(
            src=caller_node, dst=callee_entry,
            kind=DepKind.CALL, var=callee_name
        ))

        # PARAM_IN edges
        fn_decl = fns[callee_name]
        for i, param in enumerate(fn_decl.params):
            if i < len(call_expr.args):
                inter_edges.append(DepEdge(
                    src=caller_node, dst=callee_entry,
                    kind=DepKind.PARAM_IN, var=param
                ))

        # PARAM_OUT edge (return -> caller)
        inter_edges.append(DepEdge(
            src=callee_exit, dst=caller_node,
            kind=DepKind.PARAM_OUT, var="__return__"
        ))


# =========================================================================
# Alias-Aware Edge Pruning
# =========================================================================

def _prune_heap_edges_with_alias(pdg: PDG, pta_result: PointsToResult) -> PDG:
    """Remove HEAP_DATA edges where points-to analysis proves no aliasing."""
    pruned_edges = []
    for edge in pdg.edges:
        if edge.kind == DepKind.HEAP_DATA:
            # Get base variables for source (writer) and destination (reader)
            src_idx = edge.src.index
            dst_idx = edge.dst.index

            # Find base variables at writer and reader
            writer_bases = {b for b, f in pdg.heap_defs.get(src_idx, set())
                          if f == edge.var or f == '*' or edge.var == '*'}
            reader_bases = {b for b, f in pdg.heap_uses.get(dst_idx, set())
                          if f == edge.var or f == '*' or edge.var == '*'}

            # Check if any writer base may-alias any reader base
            keep = False
            for wb in writer_bases:
                for rb in reader_bases:
                    if wb == rb:
                        keep = True
                        break
                    # Check alias via points-to
                    alias = pta_result.alias(wb, rb)
                    if alias.may_alias:
                        keep = True
                        break
                if keep:
                    break

            if not writer_bases or not reader_bases:
                keep = True  # Conservative: unknown bases

            if keep:
                pruned_edges.append(edge)
        else:
            pruned_edges.append(edge)

    return PDG(
        func_name=pdg.func_name,
        nodes=pdg.nodes,
        edges=pruned_edges,
        defs=pdg.defs,
        uses=pdg.uses,
        heap_defs=pdg.heap_defs,
        heap_uses=pdg.heap_uses,
    )


def _prune_sdg_with_alias(sdg: SDG, pta_result: PointsToResult) -> SDG:
    """Prune all PDGs in the SDG using alias information."""
    new_pdgs = {}
    for name, pdg in sdg.pdgs.items():
        new_pdgs[name] = _prune_heap_edges_with_alias(pdg, pta_result)

    all_nodes = []
    all_edges = []
    for pdg in new_pdgs.values():
        all_nodes.extend(pdg.nodes)
        all_edges.extend(pdg.edges)
    all_edges.extend(sdg.inter_edges)

    return SDG(pdgs=new_pdgs, inter_edges=sdg.inter_edges,
               all_nodes=all_nodes, all_edges=all_edges)


# =========================================================================
# Slicing Algorithms
# =========================================================================

def _find_node(sdg: SDG, criterion: SliceCriterion) -> Optional[CfgNode]:
    """Find the CFG node for a slicing criterion."""
    if criterion.func in sdg.pdgs:
        pdg = sdg.pdgs[criterion.func]
        for n in pdg.nodes:
            if n.index == criterion.node_index:
                return n
    return None


def _backward_slice_impl(sdg: SDG, criterion: SliceCriterion,
                         include_control: bool = True,
                         interprocedural: bool = True) -> SliceResult:
    """Backward slicing: find all nodes that affect the criterion."""
    start_node = _find_node(sdg, criterion)
    if start_node is None:
        return SliceResult(
            criterion=criterion, direction="backward",
            nodes=set(), edges=[], functions_involved=set(),
            alias_aware=False
        )

    # Build reverse adjacency
    rev_adj = defaultdict(list)  # node -> list of (source_node, edge)
    for edge in sdg.all_edges:
        if not include_control and edge.kind == DepKind.CONTROL:
            continue
        rev_adj[edge.dst].append((edge.src, edge))

    # BFS backward
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    slice_edges = []
    funcs = set()

    # If criterion has variables, only follow edges for those variables initially
    criterion_vars = criterion.variables if criterion.variables else None

    while queue:
        current = queue.pop(0)
        funcs.add(current.func)

        for src, edge in rev_adj.get(current, []):
            # Variable filtering on first hop
            if current == start_node and criterion_vars:
                if edge.kind in (DepKind.DATA, DepKind.HEAP_DATA) and edge.var:
                    if edge.var not in criterion_vars:
                        continue

            # Interprocedural control
            if not interprocedural:
                if edge.kind in (DepKind.CALL, DepKind.PARAM_IN, DepKind.PARAM_OUT):
                    continue
                if src.func != current.func:
                    continue

            if src not in visited:
                visited.add(src)
                queue.append(src)
                slice_edges.append(edge)

    return SliceResult(
        criterion=criterion, direction="backward",
        nodes=visited, edges=slice_edges,
        functions_involved=funcs, alias_aware=False
    )


def _forward_slice_impl(sdg: SDG, criterion: SliceCriterion,
                        include_control: bool = True,
                        interprocedural: bool = True) -> SliceResult:
    """Forward slicing: find all nodes affected by the criterion."""
    start_node = _find_node(sdg, criterion)
    if start_node is None:
        return SliceResult(
            criterion=criterion, direction="forward",
            nodes=set(), edges=[], functions_involved=set(),
            alias_aware=False
        )

    # Build forward adjacency
    fwd_adj = defaultdict(list)
    for edge in sdg.all_edges:
        if not include_control and edge.kind == DepKind.CONTROL:
            continue
        fwd_adj[edge.src].append((edge.dst, edge))

    # BFS forward
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    slice_edges = []
    funcs = set()

    while queue:
        current = queue.pop(0)
        funcs.add(current.func)

        for dst, edge in fwd_adj.get(current, []):
            if not interprocedural:
                if edge.kind in (DepKind.CALL, DepKind.PARAM_IN, DepKind.PARAM_OUT):
                    continue
                if dst.func != current.func:
                    continue

            if dst not in visited:
                visited.add(dst)
                queue.append(dst)
                slice_edges.append(edge)

    return SliceResult(
        criterion=criterion, direction="forward",
        nodes=visited, edges=slice_edges,
        functions_involved=funcs, alias_aware=False
    )


# =========================================================================
# Public API
# =========================================================================

def backward_slice(source: str, criterion: SliceCriterion,
                   alias_aware: bool = True,
                   interprocedural: bool = True,
                   k: int = 1) -> SliceResult:
    """Compute a backward slice from a criterion.

    Args:
        source: C10 source code
        criterion: SliceCriterion(node_index, variables, func)
        alias_aware: Use points-to analysis for heap precision
        interprocedural: Include inter-procedural edges
        k: Context sensitivity depth for points-to analysis

    Returns:
        SliceResult with included nodes, edges, precision info
    """
    sdg = build_sdg(source)

    if alias_aware:
        pta = analyze_points_to(source, k=k)
        sdg_pruned = _prune_sdg_with_alias(sdg, pta)
        result = _backward_slice_impl(sdg_pruned, criterion,
                                      interprocedural=interprocedural)
        result.alias_aware = True

        # Compute conservative size for comparison
        conservative = _backward_slice_impl(sdg, criterion,
                                           interprocedural=interprocedural)
        result.conservative_size = conservative.size
    else:
        result = _backward_slice_impl(sdg, criterion,
                                      interprocedural=interprocedural)
        result.conservative_size = result.size

    return result


def forward_slice(source: str, criterion: SliceCriterion,
                  alias_aware: bool = True,
                  interprocedural: bool = True,
                  k: int = 1) -> SliceResult:
    """Compute a forward slice from a criterion.

    Args:
        source: C10 source code
        criterion: SliceCriterion(node_index, variables, func)
        alias_aware: Use points-to analysis for heap precision
        interprocedural: Include inter-procedural edges
        k: Context sensitivity depth for points-to analysis

    Returns:
        SliceResult with included nodes, edges, precision info
    """
    sdg = build_sdg(source)

    if alias_aware:
        pta = analyze_points_to(source, k=k)
        sdg_pruned = _prune_sdg_with_alias(sdg, pta)
        result = _forward_slice_impl(sdg_pruned, criterion,
                                     interprocedural=interprocedural)
        result.alias_aware = True

        conservative = _forward_slice_impl(sdg, criterion,
                                          interprocedural=interprocedural)
        result.conservative_size = conservative.size
    else:
        result = _forward_slice_impl(sdg, criterion,
                                     interprocedural=interprocedural)
        result.conservative_size = result.size

    return result


def thin_backward_slice(source: str, criterion: SliceCriterion,
                        alias_aware: bool = True,
                        k: int = 1) -> SliceResult:
    """Thin backward slice: data dependencies only (no control deps).

    Produces smaller slices by ignoring control flow dependencies.
    """
    sdg = build_sdg(source)

    if alias_aware:
        pta = analyze_points_to(source, k=k)
        sdg_pruned = _prune_sdg_with_alias(sdg, pta)
        result = _backward_slice_impl(sdg_pruned, criterion,
                                      include_control=False)
        result.alias_aware = True
        result.direction = "thin_backward"

        conservative = _backward_slice_impl(sdg, criterion,
                                           include_control=False)
        result.conservative_size = conservative.size
    else:
        result = _backward_slice_impl(sdg, criterion, include_control=False)
        result.direction = "thin_backward"
        result.conservative_size = result.size

    return result


def chop(source: str, source_criterion: SliceCriterion,
         target_criterion: SliceCriterion,
         alias_aware: bool = True,
         k: int = 1) -> SliceResult:
    """Compute a chop: intersection of forward from source and backward from target.

    Finds statements on paths from source to target.
    """
    sdg = build_sdg(source)

    if alias_aware:
        pta = analyze_points_to(source, k=k)
        sdg_used = _prune_sdg_with_alias(sdg, pta)
    else:
        sdg_used = sdg

    fwd = _forward_slice_impl(sdg_used, source_criterion)
    bwd = _backward_slice_impl(sdg_used, target_criterion)

    chop_nodes = fwd.nodes & bwd.nodes
    chop_edges = [e for e in fwd.edges + bwd.edges
                  if e.src in chop_nodes and e.dst in chop_nodes]
    funcs = {n.func for n in chop_nodes}

    result = SliceResult(
        criterion=target_criterion, direction="chop",
        nodes=chop_nodes, edges=chop_edges,
        functions_involved=funcs, alias_aware=alias_aware
    )

    if alias_aware:
        fwd_cons = _forward_slice_impl(sdg, source_criterion)
        bwd_cons = _backward_slice_impl(sdg, target_criterion)
        result.conservative_size = len(fwd_cons.nodes & bwd_cons.nodes)

    return result


def alias_query(source: str, var1: str, var2: str, k: int = 1) -> AliasResult:
    """Check if two variables may alias in a C10 program."""
    return check_may_alias(source, var1, var2, k=k)


def slice_with_pta(source: str, criterion: SliceCriterion,
                   direction: str = "backward",
                   k: int = 1) -> SliceResult:
    """Convenience: slice with alias awareness using points-to analysis.

    Args:
        direction: "backward", "forward", "thin_backward"
    """
    if direction == "forward":
        return forward_slice(source, criterion, alias_aware=True, k=k)
    elif direction == "thin_backward":
        return thin_backward_slice(source, criterion, alias_aware=True, k=k)
    else:
        return backward_slice(source, criterion, alias_aware=True, k=k)


def compare_slices(source: str, criterion: SliceCriterion,
                   direction: str = "backward",
                   k: int = 1) -> Dict[str, Any]:
    """Compare alias-aware vs conservative slicing.

    Returns dict with both results, sizes, and precision metrics.
    """
    if direction == "forward":
        aware = forward_slice(source, criterion, alias_aware=True, k=k)
        conservative = forward_slice(source, criterion, alias_aware=False, k=k)
    elif direction == "thin_backward":
        aware = thin_backward_slice(source, criterion, alias_aware=True, k=k)
        conservative = thin_backward_slice(source, criterion, alias_aware=False, k=k)
    else:
        aware = backward_slice(source, criterion, alias_aware=True, k=k)
        conservative = backward_slice(source, criterion, alias_aware=False, k=k)

    return {
        'alias_aware': aware,
        'conservative': conservative,
        'aware_size': aware.size,
        'conservative_size': conservative.size,
        'precision_gain': aware.precision_gain,
        'edges_removed': len(conservative.edges) - len(aware.edges),
        'nodes_removed': conservative.size - aware.size,
        'direction': direction,
    }


def get_heap_deps(source: str, func: str = "__main__",
                  alias_aware: bool = True,
                  k: int = 1) -> Dict[str, Any]:
    """Get all heap dependence edges for a function.

    Returns info about heap reads, writes, and dependencies.
    """
    sdg = build_sdg(source)
    if func not in sdg.pdgs:
        return {'error': f'Function {func} not found'}

    pdg = sdg.pdgs[func]

    heap_edges_conservative = [e for e in pdg.edges if e.kind == DepKind.HEAP_DATA]

    if alias_aware:
        pta = analyze_points_to(source, k=k)
        pruned_pdg = _prune_heap_edges_with_alias(pdg, pta)
        heap_edges_aware = [e for e in pruned_pdg.edges if e.kind == DepKind.HEAP_DATA]
    else:
        heap_edges_aware = heap_edges_conservative

    return {
        'heap_writes': dict(pdg.heap_defs),
        'heap_reads': dict(pdg.heap_uses),
        'conservative_heap_edges': len(heap_edges_conservative),
        'alias_aware_heap_edges': len(heap_edges_aware),
        'edges_pruned': len(heap_edges_conservative) - len(heap_edges_aware),
        'func': func,
    }


def full_slicing_analysis(source: str, criterion: SliceCriterion,
                          k: int = 1) -> Dict[str, Any]:
    """Run all four slicing modes and compare.

    Returns backward, forward, thin, and chop results with comparison.
    """
    bwd_aware = backward_slice(source, criterion, alias_aware=True, k=k)
    bwd_cons = backward_slice(source, criterion, alias_aware=False, k=k)
    fwd_aware = forward_slice(source, criterion, alias_aware=True, k=k)
    thin_aware = thin_backward_slice(source, criterion, alias_aware=True, k=k)

    return {
        'backward_aware': bwd_aware,
        'backward_conservative': bwd_cons,
        'forward_aware': fwd_aware,
        'thin_backward_aware': thin_aware,
        'backward_precision_gain': bwd_aware.precision_gain,
        'backward_aware_size': bwd_aware.size,
        'backward_conservative_size': bwd_cons.size,
        'forward_aware_size': fwd_aware.size,
        'thin_backward_size': thin_aware.size,
    }


def slice_summary(source: str, criterion: SliceCriterion,
                  k: int = 1) -> str:
    """Human-readable summary of alias-aware slicing."""
    result = full_slicing_analysis(source, criterion, k=k)
    lines = []
    lines.append(f"=== Alias-Aware Slicing Summary ===")
    lines.append(f"Criterion: index={criterion.node_index}, "
                 f"vars={criterion.variables}, func={criterion.func}")
    lines.append(f"")
    lines.append(f"Backward slice (alias-aware): {result['backward_aware_size']} nodes")
    lines.append(f"Backward slice (conservative): {result['backward_conservative_size']} nodes")
    gain = result['backward_precision_gain']
    lines.append(f"Precision gain: {gain:.1%}")
    lines.append(f"Forward slice (alias-aware): {result['forward_aware_size']} nodes")
    lines.append(f"Thin backward (alias-aware): {result['thin_backward_size']} nodes")
    return "\n".join(lines)
