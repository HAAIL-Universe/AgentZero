"""V037: Program Slicing -- Forward/backward slicing on Python ASTs.

Composes V035 (call graph) + V033 (def-use) for inter-procedural slicing.
Builds CFG, PDG (data + control dependence), and SDG (system dependence graph).

Slicing criterion: (line, variable) or (line, set_of_variables).
Backward slice: all statements that could affect the criterion.
Forward slice: all statements that could be affected by the criterion.
"""

import ast
import sys
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Tuple, Optional, FrozenSet, Sequence, Union
)
from collections import defaultdict, deque

# --- Import V035 call graph ---
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_v035 = os.path.join(_work, "V035_call_graph_analysis")
if _v035 not in sys.path:
    sys.path.insert(0, _v035)
from call_graph_analysis import (
    build_call_graph, CallGraph, CallSite, FuncNode, analyze as cg_analyze
)


# ============================================================
# Data structures
# ============================================================

class DepKind(Enum):
    """Kinds of dependence edges."""
    DATA = auto()        # def-use data dependence
    CONTROL = auto()     # control dependence
    CALL = auto()        # call site -> callee entry
    PARAM_IN = auto()    # actual param -> formal param
    PARAM_OUT = auto()   # formal return -> actual return
    SUMMARY = auto()     # transitive callee effect (for 2-pass SDG slicing)


@dataclass(frozen=True)
class CfgNode:
    """A node in the control flow graph."""
    func: str          # qualified function name (or "<module>")
    line: int          # source line
    kind: str          # "stmt", "entry", "exit", "branch", "call"
    ast_node: object = field(compare=False, hash=False, repr=False, default=None)

    def __repr__(self):
        return f"CFG({self.func}:{self.line}:{self.kind})"


@dataclass(frozen=True)
class DepEdge:
    """A dependence edge in the PDG/SDG."""
    src: CfgNode
    dst: CfgNode
    kind: DepKind
    var: str = ""  # variable name for data deps

    def __repr__(self):
        v = f"({self.var})" if self.var else ""
        return f"{self.src} --{self.kind.name}{v}--> {self.dst}"


@dataclass
class SliceCriterion:
    """Slicing criterion: line + optional variable set."""
    line: int
    variables: Set[str] = field(default_factory=set)

    def __init__(self, line: int, variables: Optional[Set[str]] = None):
        self.line = line
        self.variables = variables if variables is not None else set()


@dataclass
class SliceResult:
    """Result of a slicing operation."""
    criterion: SliceCriterion
    direction: str              # "backward" or "forward"
    lines: Set[int]             # included lines
    statements: List[CfgNode]   # included CFG nodes (ordered by line)
    dep_edges: List[DepEdge]    # relevant dependence edges
    functions_involved: Set[str]  # functions touched by the slice
    is_interprocedural: bool


# ============================================================
# CFG Builder
# ============================================================

class CFGBuilder(ast.NodeVisitor):
    """Builds a control flow graph from a Python function AST."""

    def __init__(self, func_name: str = "<module>"):
        self.func_name = func_name
        self.nodes: List[CfgNode] = []
        self.edges: List[Tuple[CfgNode, CfgNode]] = []  # (from, to)
        self.entry: Optional[CfgNode] = None
        self.exit_node: Optional[CfgNode] = None
        self._node_map: Dict[int, CfgNode] = {}  # line -> node

    def _make_node(self, line: int, kind: str, ast_node=None) -> CfgNode:
        n = CfgNode(func=self.func_name, line=line, kind=kind, ast_node=ast_node)
        self.nodes.append(n)
        if line not in self._node_map or kind == "stmt":
            self._node_map[line] = n
        return n

    def build(self, body: List[ast.stmt], entry_line: int = 0) -> Tuple[CfgNode, CfgNode]:
        """Build CFG for a list of statements. Returns (entry, exit)."""
        self.entry = self._make_node(entry_line, "entry")
        self.exit_node = self._make_node(-1, "exit")

        last_nodes = self._build_block(body, {self.entry})
        for n in last_nodes:
            self.edges.append((n, self.exit_node))

        return self.entry, self.exit_node

    def _build_block(self, stmts: List[ast.stmt], preds: Set[CfgNode]) -> Set[CfgNode]:
        """Process a block of statements. Returns set of exit nodes."""
        current = preds
        for stmt in stmts:
            current = self._build_stmt(stmt, current)
            if not current:  # unreachable (after return/break/continue)
                break
        return current

    def _build_stmt(self, stmt: ast.stmt, preds: Set[CfgNode]) -> Set[CfgNode]:
        """Process a single statement. Returns set of successor predecessor nodes."""
        if isinstance(stmt, (ast.If, ast.IfExp)):
            return self._build_if(stmt, preds)
        elif isinstance(stmt, (ast.While, ast.For)):
            return self._build_loop(stmt, preds)
        elif isinstance(stmt, ast.Return):
            node = self._make_node(stmt.lineno, "stmt", stmt)
            for p in preds:
                self.edges.append((p, node))
            self.edges.append((node, self.exit_node))
            return set()  # no successors
        elif isinstance(stmt, ast.Try):
            return self._build_try(stmt, preds)
        elif isinstance(stmt, ast.With):
            return self._build_with(stmt, preds)
        elif isinstance(stmt, ast.FunctionDef) or isinstance(stmt, ast.AsyncFunctionDef):
            # Function def is a single statement at module/class level
            node = self._make_node(stmt.lineno, "stmt", stmt)
            for p in preds:
                self.edges.append((p, node))
            return {node}
        else:
            # Simple statement (assign, expr, import, etc.)
            node = self._make_node(stmt.lineno, "stmt", stmt)
            for p in preds:
                self.edges.append((p, node))
            return {node}

    def _build_if(self, stmt, preds: Set[CfgNode]) -> Set[CfgNode]:
        branch = self._make_node(stmt.lineno, "branch", stmt)
        for p in preds:
            self.edges.append((p, branch))

        then_exits = self._build_block(stmt.body, {branch})
        else_exits = self._build_block(stmt.orelse, {branch}) if stmt.orelse else {branch}

        return then_exits | else_exits

    def _build_loop(self, stmt, preds: Set[CfgNode]) -> Set[CfgNode]:
        header = self._make_node(stmt.lineno, "branch", stmt)
        for p in preds:
            self.edges.append((p, header))

        body_exits = self._build_block(stmt.body, {header})
        for b in body_exits:
            self.edges.append((b, header))  # back edge

        exits = {header}  # loop may not execute (while condition false)
        if stmt.orelse:
            else_exits = self._build_block(stmt.orelse, {header})
            exits = exits | else_exits
        return exits

    def _build_try(self, stmt: ast.Try, preds: Set[CfgNode]) -> Set[CfgNode]:
        exits = set()
        try_exits = self._build_block(stmt.body, preds)
        exits |= try_exits

        for handler in stmt.handlers:
            h_node = self._make_node(handler.lineno, "branch", handler)
            for p in preds:
                self.edges.append((p, h_node))
            h_exits = self._build_block(handler.body, {h_node})
            exits |= h_exits

        if stmt.finalbody:
            final_exits = self._build_block(stmt.finalbody, exits)
            return final_exits
        return exits

    def _build_with(self, stmt: ast.With, preds: Set[CfgNode]) -> Set[CfgNode]:
        node = self._make_node(stmt.lineno, "stmt", stmt)
        for p in preds:
            self.edges.append((p, node))
        return self._build_block(stmt.body, {node})

    def get_node_at_line(self, line: int) -> Optional[CfgNode]:
        return self._node_map.get(line)

    def successors(self, node: CfgNode) -> Set[CfgNode]:
        return {dst for src, dst in self.edges if src == node}

    def predecessors(self, node: CfgNode) -> Set[CfgNode]:
        return {src for src, dst in self.edges if dst == node}


# ============================================================
# Dominance and Control Dependence
# ============================================================

def compute_dominators(
    entry: CfgNode,
    nodes: List[CfgNode],
    succs_fn,
    preds_fn
) -> Dict[CfgNode, Set[CfgNode]]:
    """Compute dominators using iterative algorithm."""
    dom: Dict[CfgNode, Set[CfgNode]] = {}
    all_nodes = set(nodes)
    dom[entry] = {entry}
    for n in nodes:
        if n != entry:
            dom[n] = set(all_nodes)

    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == entry:
                continue
            preds = preds_fn(n)
            if not preds:
                new_dom = {n}
            else:
                new_dom = set.intersection(*(dom[p] for p in preds if p in dom))
                new_dom = new_dom | {n}
            if new_dom != dom.get(n):
                dom[n] = new_dom
                changed = True
    return dom


def compute_post_dominators(
    exit_node: CfgNode,
    nodes: List[CfgNode],
    succs_fn,
    preds_fn
) -> Dict[CfgNode, Set[CfgNode]]:
    """Compute post-dominators (dominators on reversed CFG)."""
    return compute_dominators(exit_node, nodes, preds_fn, succs_fn)


def compute_idom(
    entry: CfgNode,
    nodes: List[CfgNode],
    dom: Dict[CfgNode, Set[CfgNode]]
) -> Dict[CfgNode, Optional[CfgNode]]:
    """Compute immediate dominator from dominator sets."""
    idom: Dict[CfgNode, Optional[CfgNode]] = {entry: None}
    for n in nodes:
        if n == entry:
            continue
        doms_n = dom.get(n, set()) - {n}
        # idom(n) = the dominator d of n such that d dominates all other dominators of n
        for d in doms_n:
            # d is idom(n) if d dominates all other strict dominators
            d_doms = dom.get(d, set())
            if doms_n <= d_doms:
                continue
            # Check if all others in doms_n are dominated by d
            if all(d in dom.get(other, set()) for other in doms_n if other != d):
                idom[n] = d
                break
        else:
            # Fallback: pick the strict dominator with largest dom set
            if doms_n:
                idom[n] = max(doms_n, key=lambda d: len(dom.get(d, set())))
            else:
                idom[n] = None
    return idom


def compute_control_dependence(cfg: CFGBuilder) -> List[Tuple[CfgNode, CfgNode]]:
    """Compute control dependence edges.

    Node Y is control-dependent on X if:
    1. There exists a path from X to Y in the CFG
    2. Y post-dominates every node on the path from X to Y (exclusive), but
    3. Y does not strictly post-dominate X

    In practice: for each CFG edge (A,B), walk up B's post-dominator tree
    from A until we hit B's immediate post-dominator. All nodes on this walk
    (including A) are control-dependent on the nodes they dominate.
    """
    pdom = compute_post_dominators(
        cfg.exit_node, cfg.nodes, cfg.successors, cfg.predecessors
    )
    ipdom = compute_idom(cfg.exit_node, cfg.nodes, pdom)

    # Reverse ipdom: for correct algorithm
    # For each edge (A, B) in CFG where B doesn't post-dominate A,
    # all nodes from A up to (but not including) ipdom(A) are CD on the branch at A
    cd_edges: List[Tuple[CfgNode, CfgNode]] = []
    seen = set()

    for src, dst in cfg.edges:
        # dst is control-dependent on src if src is a branch
        if src.kind != "branch":
            continue
        # Walk from dst up post-dom tree to ipdom(src)
        ipd_src = ipdom.get(src)
        runner = dst
        visited = set()
        while runner is not None and runner != ipd_src and runner not in visited:
            visited.add(runner)
            pair = (src, runner)
            if pair not in seen:
                seen.add(pair)
                cd_edges.append(pair)
            runner = ipdom.get(runner)

    return cd_edges


# ============================================================
# Def-Use Analysis (for data dependence)
# ============================================================

class DefUseCollector(ast.NodeVisitor):
    """Collect variable definitions and uses per line."""

    def __init__(self):
        self.defs: Dict[int, Set[str]] = defaultdict(set)  # line -> defined vars
        self.uses: Dict[int, Set[str]] = defaultdict(set)  # line -> used vars
        self._in_target = False

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.defs[node.lineno].add(node.id)
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            self.uses[node.lineno].add(node.id)

    def visit_Assign(self, node: ast.Assign):
        # Uses first (RHS), then defs (LHS)
        if node.value:
            self.visit(node.value)
        for target in node.targets:
            self._visit_target(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.value)
        # AugAssign both uses and defines the target
        if isinstance(node.target, ast.Name):
            self.uses[node.target.lineno].add(node.target.id)
            self.defs[node.target.lineno].add(node.target.id)
        else:
            self.visit(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value:
            self.visit(node.value)
        if node.target and isinstance(node.target, ast.Name):
            self.defs[node.target.lineno].add(node.target.id)

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self._visit_target(node.target)
        for s in node.body:
            self.visit(s)
        for s in node.orelse:
            self.visit(s)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Function name is defined at this line
        self.defs[node.lineno].add(node.name)
        # Parameters are definitions
        for arg in node.args.args:
            self.defs[node.lineno].add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.defs[node.lineno].add(arg.arg)
        if node.args.vararg:
            self.defs[node.lineno].add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defs[node.lineno].add(node.args.kwarg.arg)
        # Don't recurse into function body (handled separately per function)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.defs[node.lineno].add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.defs[node.lineno].add(name)

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.defs[node.lineno].add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self.defs[node.lineno].add(name)

    def visit_comprehension(self, node: ast.comprehension):
        self.visit(node.iter)
        self._visit_target(node.target)
        for if_ in node.ifs:
            self.visit(if_)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            self.defs[node.lineno].add(node.name)
        if node.type:
            self.visit(node.type)
        for s in node.body:
            self.visit(s)

    def _visit_target(self, target):
        """Visit assignment target (handles tuple unpacking)."""
        if isinstance(target, ast.Name):
            self.defs[target.lineno].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_target(elt)
        elif isinstance(target, ast.Starred):
            self._visit_target(target.value)
        else:
            self.visit(target)

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def collect_def_use(stmts: List[ast.stmt]) -> DefUseCollector:
    """Collect def-use info for a block of statements."""
    collector = DefUseCollector()
    for stmt in stmts:
        collector.visit(stmt)
    return collector


# ============================================================
# Call Site Analysis
# ============================================================

class CallCollector(ast.NodeVisitor):
    """Collect call sites and their argument mappings."""

    def __init__(self):
        self.calls: List[Tuple[int, str, List[str]]] = []  # (line, callee, arg_names)

    def visit_Call(self, node: ast.Call):
        callee = self._resolve_name(node.func)
        arg_names = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                arg_names.append(arg.id)
            else:
                arg_names.append(None)
        self.calls.append((node.lineno, callee, arg_names))
        self.generic_visit(node)

    def _resolve_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._resolve_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return "<unknown>"

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)


# ============================================================
# PDG (Program Dependence Graph) Builder
# ============================================================

@dataclass
class PDG:
    """Program Dependence Graph for a single function."""
    func_name: str
    cfg: CFGBuilder
    nodes: List[CfgNode]
    edges: List[DepEdge]
    data_edges: List[DepEdge]
    control_edges: List[DepEdge]
    defs: Dict[int, Set[str]]
    uses: Dict[int, Set[str]]

    def deps_on(self, node: CfgNode) -> List[DepEdge]:
        """Edges where this node is the destination (depends ON something)."""
        return [e for e in self.edges if e.dst == node]

    def dependents(self, node: CfgNode) -> List[DepEdge]:
        """Edges where this node is the source (something depends on this)."""
        return [e for e in self.edges if e.src == node]


def build_pdg(func_ast: ast.FunctionDef, func_name: str = "<module>") -> PDG:
    """Build PDG for a function."""
    # 1. Build CFG
    cfg = CFGBuilder(func_name)
    body = func_ast.body if hasattr(func_ast, 'body') else [func_ast]
    entry_line = func_ast.lineno if hasattr(func_ast, 'lineno') else 0
    cfg.build(body, entry_line)

    # 2. Collect def-use
    du = collect_def_use(body)

    # 3. Compute data dependence (reaching definitions)
    data_edges = _compute_data_deps(cfg, du)

    # 4. Compute control dependence
    cd_pairs = compute_control_dependence(cfg)
    control_edges = [
        DepEdge(src=src, dst=dst, kind=DepKind.CONTROL)
        for src, dst in cd_pairs
    ]

    all_edges = data_edges + control_edges

    return PDG(
        func_name=func_name,
        cfg=cfg,
        nodes=cfg.nodes,
        edges=all_edges,
        data_edges=data_edges,
        control_edges=control_edges,
        defs=dict(du.defs),
        uses=dict(du.uses),
    )


def _compute_data_deps(cfg: CFGBuilder, du: DefUseCollector) -> List[DepEdge]:
    """Compute data dependence via reaching definitions analysis."""
    # reaching defs: for each CFG node, which (var, line) pairs reach it
    # A def at line L of var V reaches node N if there's a path from L to N
    # without another def of V.

    # Map lines to cfg nodes
    line_to_node: Dict[int, CfgNode] = {}
    for n in cfg.nodes:
        if n.line > 0 and n.kind in ("stmt", "branch"):
            line_to_node[n.line] = n

    # Reaching definitions: iterative dataflow
    # gen[n] = {(var, line) for var in defs[line]}
    # kill[n] = {(var, L) for all L where var in defs[L]} for each var in defs[line]
    gen: Dict[CfgNode, Set[Tuple[str, int]]] = defaultdict(set)
    kill: Dict[CfgNode, Set[Tuple[str, int]]] = defaultdict(set)

    # All defs across all lines
    all_defs: Dict[str, Set[int]] = defaultdict(set)
    for line, vars_ in du.defs.items():
        for v in vars_:
            all_defs[v].add(line)

    for n in cfg.nodes:
        if n.line in du.defs:
            for v in du.defs[n.line]:
                gen[n].add((v, n.line))
                for other_line in all_defs[v]:
                    if other_line != n.line:
                        kill[n].add((v, other_line))

    # Iterative reaching definitions
    reach_in: Dict[CfgNode, Set[Tuple[str, int]]] = defaultdict(set)
    reach_out: Dict[CfgNode, Set[Tuple[str, int]]] = defaultdict(set)

    changed = True
    while changed:
        changed = False
        for n in cfg.nodes:
            new_in = set()
            for p in cfg.predecessors(n):
                new_in |= reach_out[p]

            new_out = gen.get(n, set()) | (new_in - kill.get(n, set()))

            if new_in != reach_in[n] or new_out != reach_out[n]:
                reach_in[n] = new_in
                reach_out[n] = new_out
                changed = True

    # Build data dep edges: if node N uses var V, and (V, L) reaches N,
    # then there's a data dep from L to N on V
    data_edges: List[DepEdge] = []
    seen = set()
    for n in cfg.nodes:
        if n.line not in du.uses:
            continue
        for v in du.uses[n.line]:
            for def_var, def_line in reach_in[n]:
                if def_var == v:
                    src = line_to_node.get(def_line)
                    if src and (src, n, v) not in seen:
                        seen.add((src, n, v))
                        data_edges.append(DepEdge(
                            src=src, dst=n, kind=DepKind.DATA, var=v
                        ))

    return data_edges


# ============================================================
# SDG (System Dependence Graph) -- Inter-procedural
# ============================================================

@dataclass
class SDG:
    """System Dependence Graph for inter-procedural slicing."""
    pdgs: Dict[str, PDG]                # func_name -> PDG
    call_edges: List[DepEdge]           # CALL edges
    param_in_edges: List[DepEdge]       # actual -> formal
    param_out_edges: List[DepEdge]      # formal return -> actual
    summary_edges: List[DepEdge]        # transitive effects
    all_nodes: List[CfgNode]
    all_edges: List[DepEdge]
    call_graph: Optional[CallGraph]

    def get_node_at_line(self, line: int, func: Optional[str] = None) -> Optional[CfgNode]:
        """Find a CFG node at the given line, optionally in a specific function."""
        for n in self.all_nodes:
            if n.line == line and n.kind in ("stmt", "branch"):
                if func is None or n.func == func:
                    return n
        return None


def build_sdg(source: str) -> SDG:
    """Build System Dependence Graph from Python source."""
    tree = ast.parse(source)

    # Collect all functions + module-level code
    functions: Dict[str, ast.FunctionDef] = {}
    module_stmts: List[ast.stmt] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Use qualified name for methods
            functions[node.name] = node
        if isinstance(node, ast.Module):
            module_stmts = node.body

    # Build PDG for each function
    pdgs: Dict[str, PDG] = {}
    for name, func_node in functions.items():
        pdgs[name] = build_pdg(func_node, name)

    # Build PDG for module-level code
    if module_stmts:
        module_func = ast.Module(body=module_stmts, type_ignores=[])
        module_func.lineno = 1
        pdgs["<module>"] = build_pdg(module_func, "<module>")

    # Build call graph for inter-procedural edges
    cg = build_call_graph(source)

    # Build inter-procedural edges
    call_edges: List[DepEdge] = []
    param_in_edges: List[DepEdge] = []
    param_out_edges: List[DepEdge] = []

    for site in cg.edges:
        caller_pdg = pdgs.get(site.caller) or pdgs.get("<module>")
        callee_pdg = pdgs.get(site.callee)
        if not caller_pdg or not callee_pdg:
            continue

        # Find call site node in caller
        call_node = caller_pdg.cfg.get_node_at_line(site.line)
        if not call_node:
            continue

        # CALL edge: call site -> callee entry
        callee_entry = callee_pdg.cfg.entry
        if callee_entry:
            call_edges.append(DepEdge(
                src=call_node, dst=callee_entry, kind=DepKind.CALL
            ))

        # PARAM_IN edges: actual args -> formal params
        # Find call AST in caller to get actual args
        if call_node.ast_node:
            calls = CallCollector()
            calls.visit(call_node.ast_node)
            for _, callee_name, arg_names in calls.calls:
                if callee_name == site.callee and callee_pdg:
                    func_node = functions.get(site.callee)
                    if func_node:
                        params = [a.arg for a in func_node.args.args]
                        for i, param in enumerate(params):
                            if i < len(arg_names) and arg_names[i]:
                                param_in_edges.append(DepEdge(
                                    src=call_node, dst=callee_entry,
                                    kind=DepKind.PARAM_IN, var=param
                                ))

        # PARAM_OUT: callee exit -> call site (for return values)
        callee_exit = callee_pdg.cfg.exit_node
        if callee_exit:
            param_out_edges.append(DepEdge(
                src=callee_exit, dst=call_node, kind=DepKind.PARAM_OUT
            ))

    # Compute summary edges (transitive effects through callees)
    summary_edges = _compute_summary_edges(pdgs, call_edges, param_in_edges, param_out_edges)

    # Collect all nodes and edges
    all_nodes = []
    all_edges = []
    for pdg in pdgs.values():
        all_nodes.extend(pdg.nodes)
        all_edges.extend(pdg.edges)
    all_edges.extend(call_edges)
    all_edges.extend(param_in_edges)
    all_edges.extend(param_out_edges)
    all_edges.extend(summary_edges)

    return SDG(
        pdgs=pdgs,
        call_edges=call_edges,
        param_in_edges=param_in_edges,
        param_out_edges=param_out_edges,
        summary_edges=summary_edges,
        all_nodes=all_nodes,
        all_edges=all_edges,
        call_graph=cg,
    )


def _compute_summary_edges(
    pdgs: Dict[str, PDG],
    call_edges: List[DepEdge],
    param_in: List[DepEdge],
    param_out: List[DepEdge],
) -> List[DepEdge]:
    """Compute summary edges: transitive effects of calls.

    A summary edge connects actual-in to actual-out if there exists
    a path in the callee from formal-in to formal-out.
    """
    summary: List[DepEdge] = []

    # For each call edge, check if callee has data flow from entry to exit
    for ce in call_edges:
        callee_func = ce.dst.func
        callee_pdg = pdgs.get(callee_func)
        if not callee_pdg:
            continue

        # Check if there's a path from entry to exit in the callee PDG
        has_path = _has_dep_path(callee_pdg, callee_pdg.cfg.entry, callee_pdg.cfg.exit_node)
        if has_path:
            # Summary: call site depends on itself through callee
            summary.append(DepEdge(
                src=ce.src, dst=ce.src, kind=DepKind.SUMMARY
            ))

    return summary


def _has_dep_path(pdg: PDG, src: CfgNode, dst: CfgNode) -> bool:
    """Check if there's a dependence path from src to dst in the PDG."""
    visited = set()
    queue = deque([src])
    while queue:
        n = queue.popleft()
        if n == dst:
            return True
        if n in visited:
            continue
        visited.add(n)
        for e in pdg.edges:
            if e.src == n and e.dst not in visited:
                queue.append(e.dst)
    return False


# ============================================================
# Slicing Algorithms
# ============================================================

def backward_slice(
    source: str,
    criterion: SliceCriterion,
    interprocedural: bool = True
) -> SliceResult:
    """Compute backward slice: all statements that could affect the criterion.

    Uses Weiser's algorithm for intraprocedural, Horwitz-Reps-Binkley (HRB)
    two-pass algorithm for interprocedural.
    """
    sdg = build_sdg(source)
    return _backward_slice_sdg(sdg, criterion, interprocedural)


def _backward_slice_sdg(
    sdg: SDG,
    criterion: SliceCriterion,
    interprocedural: bool = True
) -> SliceResult:
    """Backward slice on a pre-built SDG."""
    # Find the criterion node
    seed = sdg.get_node_at_line(criterion.line)
    if not seed:
        # Try to find any node near this line
        for n in sdg.all_nodes:
            if n.line == criterion.line:
                seed = n
                break
    if not seed:
        return SliceResult(
            criterion=criterion, direction="backward",
            lines=set(), statements=[], dep_edges=[],
            functions_involved=set(), is_interprocedural=False
        )

    if not interprocedural:
        # Simple intraprocedural backward slice via BFS on reverse dep edges
        return _intra_backward(sdg, seed, criterion)

    # HRB two-pass inter-procedural backward slice
    return _hrb_backward(sdg, seed, criterion)


def _intra_backward(sdg: SDG, seed: CfgNode, criterion: SliceCriterion) -> SliceResult:
    """Intraprocedural backward slice."""
    func = seed.func
    pdg = sdg.pdgs.get(func)
    if not pdg:
        return SliceResult(
            criterion=criterion, direction="backward",
            lines=set(), statements=[], dep_edges=[],
            functions_involved=set(), is_interprocedural=False
        )

    visited: Set[CfgNode] = set()
    queue = deque([seed])
    slice_edges: List[DepEdge] = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        for e in pdg.edges:
            if e.dst == node:
                # If criterion has specific variables, only follow relevant data deps
                if criterion.variables and e.kind == DepKind.DATA:
                    if e.var not in criterion.variables:
                        # Check if this var contributes to criterion vars
                        if not _var_contributes(pdg, e.var, criterion.variables, node):
                            continue
                slice_edges.append(e)
                if e.src not in visited:
                    queue.append(e.src)

    lines = {n.line for n in visited if n.line > 0}
    stmts = sorted([n for n in visited if n.line > 0 and n.kind in ("stmt", "branch")],
                    key=lambda n: n.line)

    return SliceResult(
        criterion=criterion, direction="backward",
        lines=lines, statements=stmts, dep_edges=slice_edges,
        functions_involved={func}, is_interprocedural=False
    )


def _var_contributes(pdg: PDG, var: str, target_vars: Set[str], at_node: CfgNode) -> bool:
    """Check if var contributes to any target variable via transitive data deps."""
    visited = set()
    queue = deque([var])
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        if v in target_vars:
            return True
        # Find what v flows into
        for e in pdg.data_edges:
            if e.var == v and e.src.line <= at_node.line:
                for e2 in pdg.data_edges:
                    if e2.src == e.dst and e2.var not in visited:
                        queue.append(e2.var)
    return False


def _hrb_backward(sdg: SDG, seed: CfgNode, criterion: SliceCriterion) -> SliceResult:
    """HRB two-pass inter-procedural backward slice.

    Pass 1: Traverse up from criterion, following DATA, CONTROL, CALL, PARAM_IN,
            SUMMARY edges. Don't descend into callees (no PARAM_OUT edges up).
    Pass 2: From nodes found in Pass 1, traverse DATA, CONTROL, PARAM_IN,
            SUMMARY edges. Don't ascend to callers (no CALL edges up).
    """
    # Build adjacency for reverse traversal
    reverse_adj: Dict[CfgNode, List[DepEdge]] = defaultdict(list)
    for e in sdg.all_edges:
        reverse_adj[e.dst].append(e)

    # Pass 1: ascend from criterion (don't follow PARAM_OUT backward = don't descend)
    pass1_visited: Set[CfgNode] = set()
    pass1_edges: List[DepEdge] = []
    queue = deque([seed])

    while queue:
        node = queue.popleft()
        if node in pass1_visited:
            continue
        pass1_visited.add(node)
        for e in reverse_adj[node]:
            if e.kind == DepKind.PARAM_OUT:
                continue  # Don't descend into callees in pass 1
            pass1_edges.append(e)
            if e.src not in pass1_visited:
                queue.append(e.src)

    # Pass 2: from pass1 nodes, descend into callees (follow PARAM_OUT backward)
    # but don't ascend to callers (no CALL backward)
    pass2_visited: Set[CfgNode] = set()
    pass2_edges: List[DepEdge] = []
    queue = deque(list(pass1_visited))

    while queue:
        node = queue.popleft()
        if node in pass2_visited:
            continue
        pass2_visited.add(node)
        for e in reverse_adj[node]:
            if e.kind == DepKind.CALL:
                continue  # Don't ascend to callers in pass 2
            pass2_edges.append(e)
            if e.src not in pass2_visited:
                queue.append(e.src)

    all_visited = pass1_visited | pass2_visited
    lines = {n.line for n in all_visited if n.line > 0}
    stmts = sorted([n for n in all_visited if n.line > 0 and n.kind in ("stmt", "branch")],
                    key=lambda n: n.line)
    funcs = {n.func for n in all_visited}
    all_edges = pass1_edges + pass2_edges

    return SliceResult(
        criterion=criterion, direction="backward",
        lines=lines, statements=stmts, dep_edges=all_edges,
        functions_involved=funcs, is_interprocedural=len(funcs) > 1
    )


def forward_slice(
    source: str,
    criterion: SliceCriterion,
    interprocedural: bool = True
) -> SliceResult:
    """Compute forward slice: all statements that could be affected by the criterion."""
    sdg = build_sdg(source)
    return _forward_slice_sdg(sdg, criterion, interprocedural)


def _forward_slice_sdg(
    sdg: SDG,
    criterion: SliceCriterion,
    interprocedural: bool = True
) -> SliceResult:
    """Forward slice on a pre-built SDG."""
    seed = sdg.get_node_at_line(criterion.line)
    if not seed:
        for n in sdg.all_nodes:
            if n.line == criterion.line:
                seed = n
                break
    if not seed:
        return SliceResult(
            criterion=criterion, direction="forward",
            lines=set(), statements=[], dep_edges=[],
            functions_involved=set(), is_interprocedural=False
        )

    # Build forward adjacency
    forward_adj: Dict[CfgNode, List[DepEdge]] = defaultdict(list)
    for e in sdg.all_edges:
        forward_adj[e.src].append(e)

    visited: Set[CfgNode] = set()
    slice_edges: List[DepEdge] = []
    queue = deque([seed])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for e in forward_adj[node]:
            if not interprocedural and e.kind in (DepKind.CALL, DepKind.PARAM_IN,
                                                    DepKind.PARAM_OUT, DepKind.SUMMARY):
                continue
            # Filter by criterion variables if specified
            if criterion.variables and e.kind == DepKind.DATA:
                if node == seed and e.var not in criterion.variables:
                    continue
            slice_edges.append(e)
            if e.dst not in visited:
                queue.append(e.dst)

    lines = {n.line for n in visited if n.line > 0}
    stmts = sorted([n for n in visited if n.line > 0 and n.kind in ("stmt", "branch")],
                    key=lambda n: n.line)
    funcs = {n.func for n in visited}

    return SliceResult(
        criterion=criterion, direction="forward",
        lines=lines, statements=stmts, dep_edges=slice_edges,
        functions_involved=funcs,
        is_interprocedural=interprocedural and len(funcs) > 1
    )


# ============================================================
# Chopping (intersection of forward and backward slices)
# ============================================================

def chop(
    source: str,
    source_criterion: SliceCriterion,
    target_criterion: SliceCriterion,
    interprocedural: bool = True
) -> SliceResult:
    """Compute a chop: statements on any path from source to target.

    A chop is the intersection of:
    - Forward slice from source criterion
    - Backward slice from target criterion
    """
    sdg = build_sdg(source)
    fwd = _forward_slice_sdg(sdg, source_criterion, interprocedural)
    bwd = _backward_slice_sdg(sdg, target_criterion, interprocedural)

    common_lines = fwd.lines & bwd.lines
    common_stmts = [s for s in fwd.statements if s.line in common_lines]
    common_funcs = fwd.functions_involved & bwd.functions_involved
    common_edges = [e for e in fwd.dep_edges
                    if e.src.line in common_lines and e.dst.line in common_lines]

    return SliceResult(
        criterion=source_criterion,  # Use source as the primary criterion
        direction="chop",
        lines=common_lines,
        statements=sorted(common_stmts, key=lambda n: n.line),
        dep_edges=common_edges,
        functions_involved=common_funcs,
        is_interprocedural=interprocedural and len(common_funcs) > 1
    )


# ============================================================
# Thin Slicing (data-only, no control deps)
# ============================================================

def thin_backward_slice(
    source: str,
    criterion: SliceCriterion,
) -> SliceResult:
    """Compute thin backward slice (data dependencies only, no control deps).

    Thin slices are smaller and focus on the direct data flow to the criterion.
    """
    sdg = build_sdg(source)
    seed = sdg.get_node_at_line(criterion.line)
    if not seed:
        for n in sdg.all_nodes:
            if n.line == criterion.line:
                seed = n
                break
    if not seed:
        return SliceResult(
            criterion=criterion, direction="thin_backward",
            lines=set(), statements=[], dep_edges=[],
            functions_involved=set(), is_interprocedural=False
        )

    # Only follow DATA edges backward
    reverse_data: Dict[CfgNode, List[DepEdge]] = defaultdict(list)
    for e in sdg.all_edges:
        if e.kind == DepKind.DATA:
            reverse_data[e.dst].append(e)

    visited: Set[CfgNode] = set()
    slice_edges: List[DepEdge] = []
    queue = deque([seed])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for e in reverse_data[node]:
            # At the seed, criterion vars name what's DEFINED, not what's USED.
            # Filter: only skip edges whose var isn't used in the seed's defs.
            if criterion.variables and node == seed:
                # Get what's defined at the seed line
                mod_pdg = sdg.pdgs.get(seed.func)
                if mod_pdg:
                    seed_defs = mod_pdg.defs.get(seed.line, set())
                    # If criterion vars overlap with seed defs, follow all uses
                    if seed_defs & criterion.variables:
                        pass  # follow this edge
                    elif e.var not in criterion.variables:
                        continue
            slice_edges.append(e)
            if e.src not in visited:
                queue.append(e.src)

    lines = {n.line for n in visited if n.line > 0}
    stmts = sorted([n for n in visited if n.line > 0 and n.kind in ("stmt", "branch")],
                    key=lambda n: n.line)

    return SliceResult(
        criterion=criterion, direction="thin_backward",
        lines=lines, statements=stmts, dep_edges=slice_edges,
        functions_involved={n.func for n in visited},
        is_interprocedural=False
    )


# ============================================================
# Diff Slicing (impact analysis)
# ============================================================

def diff_slice(
    source: str,
    changed_lines: Set[int],
) -> SliceResult:
    """Compute the impact of changed lines via forward slicing.

    Given a set of lines that changed, compute all lines potentially affected.
    Useful for change impact analysis and regression test selection.
    """
    sdg = build_sdg(source)

    # Forward slice from all changed lines
    all_visited: Set[CfgNode] = set()
    all_edges: List[DepEdge] = []
    all_funcs: Set[str] = set()

    forward_adj: Dict[CfgNode, List[DepEdge]] = defaultdict(list)
    for e in sdg.all_edges:
        forward_adj[e.src].append(e)

    seeds: Set[CfgNode] = set()
    for line in changed_lines:
        node = sdg.get_node_at_line(line)
        if node:
            seeds.add(node)

    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if node in all_visited:
            continue
        all_visited.add(node)
        all_funcs.add(node.func)
        for e in forward_adj[node]:
            all_edges.append(e)
            if e.dst not in all_visited:
                queue.append(e.dst)

    lines = {n.line for n in all_visited if n.line > 0}
    stmts = sorted([n for n in all_visited if n.line > 0 and n.kind in ("stmt", "branch")],
                    key=lambda n: n.line)

    return SliceResult(
        criterion=SliceCriterion(line=min(changed_lines) if changed_lines else 0),
        direction="diff_forward",
        lines=lines, statements=stmts, dep_edges=all_edges,
        functions_involved=all_funcs,
        is_interprocedural=len(all_funcs) > 1
    )


# ============================================================
# Utility: extract slice as source code
# ============================================================

def extract_slice_source(source: str, result: SliceResult) -> str:
    """Extract the source lines included in the slice."""
    source_lines = source.splitlines()
    output = []
    for line_no in sorted(result.lines):
        if 1 <= line_no <= len(source_lines):
            output.append(f"{line_no:4d}: {source_lines[line_no - 1]}")
    return "\n".join(output)


def slice_report(source: str, result: SliceResult) -> str:
    """Generate a human-readable slice report."""
    parts = [
        f"=== {result.direction.upper()} SLICE ===",
        f"Criterion: line {result.criterion.line}"
    ]
    if result.criterion.variables:
        parts.append(f"Variables: {', '.join(sorted(result.criterion.variables))}")
    parts.append(f"Lines in slice: {len(result.lines)}")
    parts.append(f"Functions: {', '.join(sorted(result.functions_involved))}")
    parts.append(f"Inter-procedural: {result.is_interprocedural}")
    parts.append(f"Dependence edges: {len(result.dep_edges)}")
    parts.append("")
    parts.append("--- Slice Source ---")
    parts.append(extract_slice_source(source, result))
    return "\n".join(parts)
