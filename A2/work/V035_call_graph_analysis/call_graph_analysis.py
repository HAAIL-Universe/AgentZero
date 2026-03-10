"""
V035: Call Graph Analysis + Dead Code Detection

Builds static call graphs from Python source code, identifies unreachable
functions (dead code), detects circular dependencies, computes SCCs, and
provides various graph-theoretic analyses over function relationships.

Composes with V033's per-function call tracking but builds its own
higher-fidelity AST walker for qualified name resolution (methods,
module-level calls, nested functions).
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, FrozenSet
from enum import Enum
from collections import defaultdict, deque


# ============================================================
# Core Data Structures
# ============================================================

class NodeKind(Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASSMETHOD = "classmethod"
    STATICMETHOD = "staticmethod"
    PROPERTY = "property"
    MODULE_LEVEL = "module_level"
    LAMBDA = "lambda"
    NESTED = "nested"


class EdgeKind(Enum):
    DIRECT_CALL = "direct_call"
    METHOD_CALL = "method_call"
    SUPER_CALL = "super_call"
    DECORATOR_CALL = "decorator_call"
    CALLBACK = "callback"        # passed as argument
    CONDITIONAL = "conditional"  # called inside if/try


@dataclass
class CallSite:
    """A specific location where a call occurs."""
    caller: str
    callee: str
    line: int
    col: int
    edge_kind: EdgeKind
    in_conditional: bool = False
    in_try: bool = False


@dataclass
class FuncNode:
    """A node in the call graph representing a callable."""
    name: str                      # qualified name (Class.method or func)
    kind: NodeKind
    file: str
    line: int
    end_line: int
    num_lines: int
    decorators: List[str] = field(default_factory=list)
    is_entry_point: bool = False   # main, __init__, test_, etc.
    is_dunder: bool = False
    is_callback_target: bool = False
    parent_class: str = ""


@dataclass
class CallGraph:
    """Complete call graph for a codebase."""
    nodes: Dict[str, FuncNode] = field(default_factory=dict)
    edges: List[CallSite] = field(default_factory=list)
    # Adjacency lists
    callers: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    callees: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node: FuncNode):
        self.nodes[node.name] = node

    def add_edge(self, site: CallSite):
        self.edges.append(site)
        self.callees[site.caller].add(site.callee)
        self.callers[site.callee].add(site.caller)

    def all_callees(self, name: str) -> Set[str]:
        """Transitively reachable callees from name."""
        visited = set()
        queue = deque([name])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            for c in self.callees.get(n, set()):
                if c not in visited:
                    queue.append(c)
        visited.discard(name)
        return visited

    def all_callers(self, name: str) -> Set[str]:
        """Transitively reachable callers of name."""
        visited = set()
        queue = deque([name])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            for c in self.callers.get(n, set()):
                if c not in visited:
                    queue.append(c)
        visited.discard(name)
        return visited


@dataclass
class DeadCodeResult:
    """Results of dead code analysis."""
    dead_functions: List[FuncNode]
    entry_points: List[str]
    reachable_count: int
    total_count: int
    dead_lines: int  # total lines in dead functions


@dataclass
class CycleInfo:
    """Information about a cycle in the call graph."""
    members: List[str]
    is_direct_recursion: bool
    is_mutual_recursion: bool
    length: int


@dataclass
class SCCInfo:
    """Strongly connected component."""
    members: FrozenSet[str]
    size: int
    has_cycle: bool  # size > 1 or self-loop


@dataclass
class CallGraphAnalysis:
    """Full analysis results."""
    graph: CallGraph
    dead_code: DeadCodeResult
    cycles: List[CycleInfo]
    sccs: List[SCCInfo]
    # Metrics
    max_fan_in: Tuple[str, int]   # most-called function
    max_fan_out: Tuple[str, int]  # function calling most others
    max_depth: Tuple[str, int]    # deepest call chain
    orphan_functions: List[str]   # neither call nor are called


# ============================================================
# AST-based Call Graph Builder
# ============================================================

ENTRY_PATTERNS = {
    "main", "__main__", "__init__", "__new__", "__del__",
    "__enter__", "__exit__", "__call__",
    "setup", "teardown", "setUp", "tearDown",
    "setUpClass", "tearDownClass", "setUpModule", "tearDownModule",
}

DUNDER_PROTOCOL = {
    "__str__", "__repr__", "__len__", "__getitem__", "__setitem__",
    "__delitem__", "__contains__", "__iter__", "__next__", "__hash__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
    "__mod__", "__pow__", "__neg__", "__pos__", "__abs__",
    "__bool__", "__int__", "__float__", "__index__",
    "__getattr__", "__setattr__", "__delattr__",
    "__get__", "__set__", "__delete__",
    "__init_subclass__", "__class_getitem__",
}


class CallGraphVisitor(ast.NodeVisitor):
    """Walks Python AST to extract function definitions and call sites."""

    def __init__(self, file_path: str = "<string>"):
        self.file = file_path
        self.nodes: List[FuncNode] = []
        self.call_sites: List[CallSite] = []
        self._scope_stack: List[str] = []  # current scope chain
        self._class_stack: List[str] = []  # current class chain
        self._in_conditional = False
        self._in_try = False
        self._callback_names: Set[str] = set()

    def _qualified_name(self, name: str) -> str:
        if self._scope_stack:
            return ".".join(self._scope_stack) + "." + name
        return name

    def _current_scope(self) -> str:
        return ".".join(self._scope_stack) if self._scope_stack else "<module>"

    def _get_decorators(self, node) -> List[str]:
        decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                decorators.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorators.append(d.attr)
            elif isinstance(d, ast.Call):
                if isinstance(d.func, ast.Name):
                    decorators.append(d.func.id)
                elif isinstance(d.func, ast.Attribute):
                    decorators.append(d.func.attr)
        return decorators

    def _determine_kind(self, node: ast.FunctionDef, decorators: List[str]) -> NodeKind:
        if self._class_stack:
            if "classmethod" in decorators:
                return NodeKind.CLASSMETHOD
            if "staticmethod" in decorators:
                return NodeKind.STATICMETHOD
            if "property" in decorators:
                return NodeKind.PROPERTY
            return NodeKind.METHOD
        if self._scope_stack:
            return NodeKind.NESTED
        return NodeKind.FUNCTION

    def _is_entry_point(self, name: str, decorators: List[str]) -> bool:
        if name in ENTRY_PATTERNS:
            return True
        if name.startswith("test_") or name.startswith("test"):
            return True
        for d in decorators:
            if d in ("app.route", "route", "pytest.fixture", "fixture",
                     "click.command", "command", "register"):
                return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef):
        self._scope_stack.append(node.name)
        self._class_stack.append(node.name)
        # Decorators can be calls
        for d in node.decorator_list:
            self._visit_decorator_call(d)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

    def _visit_decorator_call(self, decorator):
        if isinstance(decorator, ast.Call):
            call_name = self._resolve_call_name(decorator)
            if call_name:
                scope = self._current_scope()
                self.call_sites.append(CallSite(
                    caller=scope, callee=call_name,
                    line=decorator.lineno, col=decorator.col_offset,
                    edge_kind=EdgeKind.DECORATOR_CALL
                ))
        elif isinstance(decorator, ast.Name):
            scope = self._current_scope()
            self.call_sites.append(CallSite(
                caller=scope, callee=decorator.id,
                line=decorator.lineno, col=decorator.col_offset,
                edge_kind=EdgeKind.DECORATOR_CALL
            ))
        elif isinstance(decorator, ast.Attribute):
            scope = self._current_scope()
            self.call_sites.append(CallSite(
                caller=scope, callee=decorator.attr,
                line=decorator.lineno, col=decorator.col_offset,
                edge_kind=EdgeKind.DECORATOR_CALL
            ))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        decorators = self._get_decorators(node)
        kind = self._determine_kind(node, decorators)
        name = node.name
        qualified = self._qualified_name(name)
        is_dunder = name.startswith("__") and name.endswith("__")

        fn_node = FuncNode(
            name=qualified,
            kind=kind,
            file=self.file,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            num_lines=(node.end_lineno or node.lineno) - node.lineno + 1,
            decorators=decorators,
            is_entry_point=self._is_entry_point(name, decorators),
            is_dunder=is_dunder,
            parent_class=self._class_stack[-1] if self._class_stack else "",
        )
        self.nodes.append(fn_node)

        # Visit decorators for calls
        for d in node.decorator_list:
            self._visit_decorator_call(d)

        self._scope_stack.append(name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda):
        qualified = self._qualified_name(f"<lambda>@{node.lineno}")
        fn_node = FuncNode(
            name=qualified,
            kind=NodeKind.LAMBDA,
            file=self.file,
            line=node.lineno,
            end_line=node.lineno,
            num_lines=1,
        )
        self.nodes.append(fn_node)
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.Call) -> Optional[str]:
        """Resolve a Call node to a function name string."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # Try to resolve the chain: obj.method
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)
            # e.g., func().method -- just use the method name
            return func.attr
        return None

    def visit_Call(self, node: ast.Call):
        call_name = self._resolve_call_name(node)
        scope = self._current_scope()

        if call_name:
            # Determine edge kind
            edge_kind = EdgeKind.DIRECT_CALL
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Call) and \
                   isinstance(node.func.value.func, ast.Name) and \
                   node.func.value.func.id == "super":
                    edge_kind = EdgeKind.SUPER_CALL
                else:
                    edge_kind = EdgeKind.METHOD_CALL

            self.call_sites.append(CallSite(
                caller=scope, callee=call_name,
                line=node.lineno, col=node.col_offset,
                edge_kind=edge_kind,
                in_conditional=self._in_conditional,
                in_try=self._in_try,
            ))

        # Check for callback arguments (functions passed as args)
        for arg in node.args:
            if isinstance(arg, ast.Name):
                # Could be a callback -- track it
                self._callback_names.add(arg.id)

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        old = self._in_conditional
        self._in_conditional = True
        self.generic_visit(node)
        self._in_conditional = old

    def visit_Try(self, node):
        old = self._in_try
        self._in_try = True
        self.generic_visit(node)
        self._in_try = old

    visit_TryStar = visit_Try


def _build_graph_from_visitor(visitor: CallGraphVisitor) -> CallGraph:
    """Convert visitor results to a CallGraph."""
    graph = CallGraph()

    for fn in visitor.nodes:
        graph.add_node(fn)

    # Build name resolution map: short name -> qualified names
    name_map: Dict[str, List[str]] = defaultdict(list)
    for qname in graph.nodes:
        # Map last component to qualified name
        short = qname.rsplit(".", 1)[-1]
        name_map[short].append(qname)
        name_map[qname].append(qname)

    # Also map Class.method patterns
    for qname, node in graph.nodes.items():
        if node.parent_class:
            class_method = f"{node.parent_class}.{qname.rsplit('.', 1)[-1]}"
            name_map[class_method].append(qname)

    # Mark callback targets
    for cb_name in visitor._callback_names:
        for qname in name_map.get(cb_name, []):
            graph.nodes[qname].is_callback_target = True

    # Resolve call sites
    for site in visitor.call_sites:
        callee = site.callee
        # Try exact match first
        if callee in graph.nodes:
            graph.add_edge(site)
            continue

        # Try short-name resolution
        parts = callee.split(".")
        short = parts[-1]
        candidates = name_map.get(callee, [])
        if not candidates:
            candidates = name_map.get(short, [])

        if len(candidates) == 1:
            resolved_site = CallSite(
                caller=site.caller, callee=candidates[0],
                line=site.line, col=site.col,
                edge_kind=site.edge_kind,
                in_conditional=site.in_conditional,
                in_try=site.in_try,
            )
            graph.add_edge(resolved_site)
        elif len(candidates) > 1:
            # Prefer same-class method for self.method() calls
            caller_class = ""
            if site.caller in graph.nodes:
                caller_class = graph.nodes[site.caller].parent_class

            best = None
            for c in candidates:
                cn = graph.nodes[c]
                if cn.parent_class and cn.parent_class == caller_class:
                    best = c
                    break
            if best is None:
                best = candidates[0]

            resolved_site = CallSite(
                caller=site.caller, callee=best,
                line=site.line, col=site.col,
                edge_kind=site.edge_kind,
                in_conditional=site.in_conditional,
                in_try=site.in_try,
            )
            graph.add_edge(resolved_site)
        # else: external call, not in our graph -- skip

    return graph


# ============================================================
# Analysis Algorithms
# ============================================================

def find_dead_code(graph: CallGraph, extra_entry_points: Optional[Set[str]] = None) -> DeadCodeResult:
    """Find functions unreachable from any entry point."""
    entry_points = set()
    for name, node in graph.nodes.items():
        if node.is_entry_point:
            entry_points.add(name)
        if node.is_dunder and node.name.rsplit(".", 1)[-1] in DUNDER_PROTOCOL:
            entry_points.add(name)
        if node.is_callback_target:
            entry_points.add(name)
        if node.kind == NodeKind.PROPERTY:
            entry_points.add(name)

    if extra_entry_points:
        entry_points |= extra_entry_points

    # If no entry points found, treat all top-level functions as entry
    if not entry_points:
        for name, node in graph.nodes.items():
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                entry_points.add(name)

    # BFS from all entry points
    reachable = set()
    queue = deque(entry_points)
    while queue:
        n = queue.popleft()
        if n in reachable:
            continue
        reachable.add(n)
        for c in graph.callees.get(n, set()):
            if c not in reachable:
                queue.append(c)

    # Find dead functions
    dead = []
    dead_lines = 0
    for name, node in graph.nodes.items():
        if name not in reachable and node.kind != NodeKind.LAMBDA:
            dead.append(node)
            dead_lines += node.num_lines

    dead.sort(key=lambda n: (n.file, n.line))

    return DeadCodeResult(
        dead_functions=dead,
        entry_points=sorted(entry_points),
        reachable_count=len(reachable),
        total_count=len(graph.nodes),
        dead_lines=dead_lines,
    )


def find_cycles(graph: CallGraph) -> List[CycleInfo]:
    """Find all elementary cycles using Johnson's algorithm (simplified)."""
    cycles = []

    # Check direct recursion first
    for name in graph.nodes:
        if name in graph.callees.get(name, set()):
            cycles.append(CycleInfo(
                members=[name],
                is_direct_recursion=True,
                is_mutual_recursion=False,
                length=1,
            ))

    # Find mutual recursion via SCC analysis
    sccs = tarjan_scc(graph)
    for scc in sccs:
        if len(scc.members) > 1:
            members = sorted(scc.members)
            cycles.append(CycleInfo(
                members=members,
                is_direct_recursion=False,
                is_mutual_recursion=True,
                length=len(members),
            ))

    return cycles


def tarjan_scc(graph: CallGraph) -> List[SCCInfo]:
    """Compute strongly connected components using Tarjan's algorithm."""
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.callees.get(v, set()):
            if w not in graph.nodes:
                continue  # external call
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc_members = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc_members.add(w)
                if w == v:
                    break

            has_cycle = len(scc_members) > 1
            if not has_cycle:
                # Check self-loop
                member = next(iter(scc_members))
                has_cycle = member in graph.callees.get(member, set())

            result.append(SCCInfo(
                members=frozenset(scc_members),
                size=len(scc_members),
                has_cycle=has_cycle,
            ))

    for v in graph.nodes:
        if v not in index:
            strongconnect(v)

    return result


def compute_max_depth(graph: CallGraph) -> Tuple[str, int, List[str]]:
    """Find the longest acyclic call chain. Returns (root, depth, path)."""
    # Memoize with cycle detection
    cache: Dict[str, Tuple[int, List[str]]] = {}

    def dfs(name: str, visited: Set[str]) -> Tuple[int, List[str]]:
        if name in cache:
            return cache[name]
        if name in visited:
            return (0, [name])  # cycle -- stop

        visited.add(name)
        max_d = 0
        max_path = [name]

        for c in graph.callees.get(name, set()):
            if c in graph.nodes:
                d, p = dfs(c, visited)
                if d + 1 > max_d:
                    max_d = d + 1
                    max_path = [name] + p

        visited.discard(name)
        cache[name] = (max_d, max_path)
        return (max_d, max_path)

    best_root = ""
    best_depth = 0
    best_path: List[str] = []

    for name in graph.nodes:
        d, p = dfs(name, set())
        if d > best_depth:
            best_depth = d
            best_root = name
            best_path = p

    return (best_root, best_depth, best_path)


def compute_fan_metrics(graph: CallGraph) -> Tuple[Tuple[str, int], Tuple[str, int]]:
    """Compute max fan-in (most called) and max fan-out (calls most)."""
    max_fan_in = ("", 0)
    max_fan_out = ("", 0)

    for name in graph.nodes:
        fan_in = len(graph.callers.get(name, set()))
        fan_out = len(graph.callees.get(name, set()))
        if fan_in > max_fan_in[1]:
            max_fan_in = (name, fan_in)
        if fan_out > max_fan_out[1]:
            max_fan_out = (name, fan_out)

    return max_fan_in, max_fan_out


def find_orphans(graph: CallGraph) -> List[str]:
    """Find functions that neither call anything nor are called by anything."""
    orphans = []
    for name, node in graph.nodes.items():
        if node.kind == NodeKind.LAMBDA:
            continue
        has_callers = len(graph.callers.get(name, set())) > 0
        has_callees = len(graph.callees.get(name, set())) > 0
        if not has_callers and not has_callees and not node.is_entry_point:
            orphans.append(name)
    return sorted(orphans)


def dependency_layers(graph: CallGraph) -> List[List[str]]:
    """Compute dependency layers (topological ordering by levels).

    Layer 0 = leaf functions (call nothing), Layer 1 = calls only layer 0, etc.
    Functions in cycles are placed in the layer of their SCC's deepest dependency + 1.
    """
    # First compute SCCs
    sccs = tarjan_scc(graph)
    # Map each node to its SCC index
    node_to_scc: Dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for m in scc.members:
            node_to_scc[m] = i

    # Build SCC-level DAG
    scc_edges: Dict[int, Set[int]] = defaultdict(set)
    for name in graph.nodes:
        s1 = node_to_scc.get(name)
        if s1 is None:
            continue
        for c in graph.callees.get(name, set()):
            s2 = node_to_scc.get(c)
            if s2 is not None and s2 != s1:
                scc_edges[s1].add(s2)

    # Topological level assignment on SCC DAG
    scc_level: Dict[int, int] = {}

    def get_level(si: int, visiting: Set[int]) -> int:
        if si in scc_level:
            return scc_level[si]
        if si in visiting:
            return 0  # cycle in SCC DAG shouldn't happen, but guard
        visiting.add(si)
        max_dep = -1
        for dep in scc_edges.get(si, set()):
            max_dep = max(max_dep, get_level(dep, visiting))
        visiting.discard(si)
        scc_level[si] = max_dep + 1
        return max_dep + 1

    for i in range(len(sccs)):
        get_level(i, set())

    # Group nodes by level
    level_groups: Dict[int, List[str]] = defaultdict(list)
    for name in sorted(graph.nodes.keys()):
        si = node_to_scc.get(name)
        if si is not None:
            level_groups[scc_level[si]].append(name)

    max_level = max(level_groups.keys()) if level_groups else 0
    return [level_groups.get(i, []) for i in range(max_level + 1)]


# ============================================================
# High-Level API
# ============================================================

def build_call_graph(source: str, file_path: str = "<string>") -> CallGraph:
    """Build a call graph from Python source code."""
    tree = ast.parse(source)
    visitor = CallGraphVisitor(file_path)
    visitor.visit(tree)
    return _build_graph_from_visitor(visitor)


def build_call_graph_from_file(file_path: str) -> CallGraph:
    """Build a call graph from a Python file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    return build_call_graph(source, file_path)


def build_call_graph_from_directory(dir_path: str,
                                     exclude_tests: bool = False) -> CallGraph:
    """Build a combined call graph from all Python files in a directory."""
    combined = CallGraph()

    for root, dirs, files in os.walk(dir_path):
        # Skip hidden dirs and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            if exclude_tests and (fname.startswith("test_") or fname.endswith("_test.py")):
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                tree = ast.parse(source)
                visitor = CallGraphVisitor(fpath)
                visitor.visit(tree)

                for fn in visitor.nodes:
                    combined.add_node(fn)
            except SyntaxError:
                continue

    # Now resolve all call sites across all files
    # Rebuild with combined graph for cross-file resolution
    name_map: Dict[str, List[str]] = defaultdict(list)
    for qname in combined.nodes:
        short = qname.rsplit(".", 1)[-1]
        name_map[short].append(qname)
        name_map[qname].append(qname)
        node = combined.nodes[qname]
        if node.parent_class:
            class_method = f"{node.parent_class}.{qname.rsplit('.', 1)[-1]}"
            name_map[class_method].append(qname)

    # Re-walk all files for call sites with the combined name map
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            if exclude_tests and (fname.startswith("test_") or fname.endswith("_test.py")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                tree = ast.parse(source)
                visitor = CallGraphVisitor(fpath)
                visitor.visit(tree)

                for site in visitor.call_sites:
                    callee = site.callee
                    candidates = name_map.get(callee, [])
                    if not candidates:
                        short = callee.rsplit(".", 1)[-1]
                        candidates = name_map.get(short, [])

                    if len(candidates) == 1:
                        site_resolved = CallSite(
                            caller=site.caller, callee=candidates[0],
                            line=site.line, col=site.col,
                            edge_kind=site.edge_kind,
                            in_conditional=site.in_conditional,
                            in_try=site.in_try,
                        )
                        combined.add_edge(site_resolved)
                    elif len(candidates) > 1:
                        # Prefer same-class
                        caller_class = ""
                        if site.caller in combined.nodes:
                            caller_class = combined.nodes[site.caller].parent_class
                        best = candidates[0]
                        for c in candidates:
                            cn = combined.nodes[c]
                            if cn.parent_class and cn.parent_class == caller_class:
                                best = c
                                break
                        site_resolved = CallSite(
                            caller=site.caller, callee=best,
                            line=site.line, col=site.col,
                            edge_kind=site.edge_kind,
                            in_conditional=site.in_conditional,
                            in_try=site.in_try,
                        )
                        combined.add_edge(site_resolved)
            except SyntaxError:
                continue

    return combined


def analyze(source: str, file_path: str = "<string>",
            extra_entry_points: Optional[Set[str]] = None) -> CallGraphAnalysis:
    """Full call graph analysis of Python source."""
    graph = build_call_graph(source, file_path)
    return analyze_graph(graph, extra_entry_points)


def analyze_graph(graph: CallGraph,
                  extra_entry_points: Optional[Set[str]] = None) -> CallGraphAnalysis:
    """Full analysis of an existing call graph."""
    dead = find_dead_code(graph, extra_entry_points)
    cycles = find_cycles(graph)
    sccs = tarjan_scc(graph)
    fan_in, fan_out = compute_fan_metrics(graph)
    root, depth, path = compute_max_depth(graph)
    orphans = find_orphans(graph)

    return CallGraphAnalysis(
        graph=graph,
        dead_code=dead,
        cycles=cycles,
        sccs=sccs,
        max_fan_in=fan_in,
        max_fan_out=fan_out,
        max_depth=(root, depth),
        orphan_functions=orphans,
    )


def report(analysis: CallGraphAnalysis) -> str:
    """Generate a human-readable report."""
    lines = []
    g = analysis.graph
    dc = analysis.dead_code

    lines.append(f"=== Call Graph Analysis ===")
    lines.append(f"Functions: {len(g.nodes)}")
    lines.append(f"Call edges: {len(g.edges)}")
    lines.append(f"Entry points: {len(dc.entry_points)}")
    lines.append(f"Reachable: {dc.reachable_count}/{dc.total_count}")
    lines.append("")

    if dc.dead_functions:
        lines.append(f"--- Dead Code ({len(dc.dead_functions)} functions, {dc.dead_lines} lines) ---")
        for fn in dc.dead_functions:
            lines.append(f"  {fn.name} ({fn.file}:{fn.line}, {fn.num_lines} lines)")
        lines.append("")

    if analysis.cycles:
        lines.append(f"--- Cycles ({len(analysis.cycles)}) ---")
        for c in analysis.cycles:
            if c.is_direct_recursion:
                lines.append(f"  Self-recursive: {c.members[0]}")
            else:
                lines.append(f"  Mutual recursion ({c.length}): {' -> '.join(c.members)}")
        lines.append("")

    non_trivial_sccs = [s for s in analysis.sccs if s.has_cycle]
    if non_trivial_sccs:
        lines.append(f"--- SCCs with cycles ({len(non_trivial_sccs)}) ---")
        for scc in non_trivial_sccs:
            lines.append(f"  Size {scc.size}: {', '.join(sorted(scc.members))}")
        lines.append("")

    if analysis.max_fan_in[0]:
        lines.append(f"Most called: {analysis.max_fan_in[0]} ({analysis.max_fan_in[1]} callers)")
    if analysis.max_fan_out[0]:
        lines.append(f"Most calling: {analysis.max_fan_out[0]} ({analysis.max_fan_out[1]} callees)")
    if analysis.max_depth[0]:
        lines.append(f"Deepest chain: {analysis.max_depth[1]} levels from {analysis.max_depth[0]}")

    if analysis.orphan_functions:
        lines.append(f"\n--- Orphans ({len(analysis.orphan_functions)}) ---")
        for o in analysis.orphan_functions:
            lines.append(f"  {o}")

    return "\n".join(lines)
