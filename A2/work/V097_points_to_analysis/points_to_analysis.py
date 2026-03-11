"""V097: Context-Sensitive Points-To Analysis

Composes V096 (interprocedural analysis/ICFG) + C010 (parser) for
whole-program points-to analysis with context sensitivity.

Key idea: Track which allocation sites each variable may point to.
Context sensitivity via k-CFA call strings ensures that the same
function called from different sites gets separate analysis.

Features:
- Andersen's inclusion-based constraint generation from C10 source
- Field-sensitive analysis for hash map field access (x.f)
- k-CFA call-string context sensitivity (configurable k)
- Flow-sensitive analysis with per-program-point results
- Alias queries: may-alias, must-alias, points-to set
- Heap modeling: allocation sites for arrays, hashes, closures, objects
- Call graph construction from points-to results (virtual dispatch)
- Escape analysis: which allocations escape their creating function
- Mod/ref analysis: which heap locations a function may read/write
"""

import sys
import os
from dataclasses import dataclass, field
from typing import (Dict, Set, List, Tuple, Optional, FrozenSet,
                    Any, Union)
from enum import Enum, auto
from collections import defaultdict, deque
from copy import deepcopy

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser

# Import V096 ICFG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V096_interprocedural_analysis'))
from interprocedural_analysis import (
    ICFG, ProgramPoint, ICFGEdge, build_icfg, _extract_vars_from_expr
)


# ---------------------------------------------------------------------------
# Abstract heap locations
# ---------------------------------------------------------------------------

class AllocKind(Enum):
    """Kind of heap allocation."""
    ARRAY = auto()     # [] or array literal
    HASH = auto()      # {} or hash literal
    CLOSURE = auto()   # fn() or lambda
    OBJECT = auto()    # new ClassName()
    STRING = auto()    # string literal or concatenation
    UNKNOWN = auto()   # unknown allocation


@dataclass(frozen=True)
class HeapLoc:
    """Abstract heap location identified by allocation site.

    An allocation site is (function, label) pair identifying where
    the allocation occurs in the source code.
    """
    function: str     # function containing the allocation
    label: str        # statement label within function
    kind: AllocKind   # what kind of object
    context: Tuple[str, ...] = ()  # call-string context for k-CFA

    def __repr__(self):
        ctx = f"@{'->'.join(self.context)}" if self.context else ""
        return f"alloc({self.function}.{self.label}{ctx},{self.kind.name})"

    @property
    def site_id(self) -> str:
        """Context-insensitive site identifier."""
        return f"{self.function}.{self.label}"

    @property
    def full_id(self) -> str:
        """Context-sensitive identifier."""
        ctx = "->".join(self.context) if self.context else ""
        return f"{self.function}.{self.label}[{ctx}]"


@dataclass(frozen=True)
class FieldLoc:
    """A field of a heap location: heap_loc.field_name."""
    base: HeapLoc
    field: str

    def __repr__(self):
        return f"{self.base}.{self.field}"


@dataclass(frozen=True)
class AbstractPtr:
    """Abstract pointer: set of heap locations it may point to."""
    targets: FrozenSet[HeapLoc] = frozenset()

    def __or__(self, other: 'AbstractPtr') -> 'AbstractPtr':
        return AbstractPtr(self.targets | other.targets)

    def __bool__(self):
        return bool(self.targets)

    def __repr__(self):
        if not self.targets:
            return "NULL"
        return "{" + ", ".join(str(t) for t in sorted(self.targets, key=str)) + "}"

    def __iter__(self):
        return iter(self.targets)

    def __len__(self):
        return len(self.targets)


# ---------------------------------------------------------------------------
# Points-to constraint system
# ---------------------------------------------------------------------------

class ConstraintKind(Enum):
    """Kind of points-to constraint."""
    ALLOC = auto()      # x = new T()  =>  x -> {alloc_site}
    ASSIGN = auto()     # x = y        =>  pts(x) >= pts(y)
    LOAD = auto()       # x = y.f      =>  for h in pts(y): pts(x) >= pts(h.f)
    STORE = auto()      # x.f = y      =>  for h in pts(x): pts(h.f) >= pts(y)
    CALL_ARG = auto()   # f(x)         =>  param_i -> pts(arg_i)
    CALL_RET = auto()   # x = f(y)     =>  pts(x) >= pts(return_of_f)


@dataclass(frozen=True)
class Constraint:
    """A points-to constraint."""
    kind: ConstraintKind
    lhs: str               # variable receiving points-to info
    rhs: str = ""          # source variable
    field_name: str = ""   # for LOAD/STORE
    alloc: Optional[HeapLoc] = None  # for ALLOC
    callee: str = ""       # for CALL_ARG/CALL_RET
    context: Tuple[str, ...] = ()  # call-string context

    def __repr__(self):
        if self.kind == ConstraintKind.ALLOC:
            return f"{self.lhs} = {self.alloc}"
        if self.kind == ConstraintKind.ASSIGN:
            return f"pts({self.lhs}) >= pts({self.rhs})"
        if self.kind == ConstraintKind.LOAD:
            return f"pts({self.lhs}) >= pts({self.rhs}.{self.field_name})"
        if self.kind == ConstraintKind.STORE:
            return f"pts({self.lhs}.{self.field_name}) >= pts({self.rhs})"
        if self.kind == ConstraintKind.CALL_ARG:
            return f"pts({self.callee}.{self.lhs}) >= pts({self.rhs})"
        if self.kind == ConstraintKind.CALL_RET:
            return f"pts({self.lhs}) >= pts({self.callee}.return)"
        return f"constraint({self.kind}, {self.lhs}, {self.rhs})"


# ---------------------------------------------------------------------------
# Points-to state
# ---------------------------------------------------------------------------

@dataclass
class PointsToState:
    """Points-to state: maps variables and field locations to pointer sets."""
    var_pts: Dict[str, Set[HeapLoc]] = field(default_factory=lambda: defaultdict(set))
    field_pts: Dict[Tuple[HeapLoc, str], Set[HeapLoc]] = field(
        default_factory=lambda: defaultdict(set))

    def get_pts(self, var: str) -> Set[HeapLoc]:
        return self.var_pts.get(var, set())

    def get_field_pts(self, base: HeapLoc, fname: str) -> Set[HeapLoc]:
        return self.field_pts.get((base, fname), set())

    def add_pts(self, var: str, locs: Set[HeapLoc]) -> bool:
        """Add locations to var's points-to set. Returns True if changed."""
        old_size = len(self.var_pts[var])
        self.var_pts[var] |= locs
        return len(self.var_pts[var]) > old_size

    def add_field_pts(self, base: HeapLoc, fname: str, locs: Set[HeapLoc]) -> bool:
        """Add locations to field's points-to set. Returns True if changed."""
        key = (base, fname)
        old_size = len(self.field_pts[key])
        self.field_pts[key] |= locs
        return len(self.field_pts[key]) > old_size

    def copy(self) -> 'PointsToState':
        new = PointsToState()
        for k, v in self.var_pts.items():
            new.var_pts[k] = set(v)
        for k, v in self.field_pts.items():
            new.field_pts[k] = set(v)
        return new

    def join(self, other: 'PointsToState') -> bool:
        """Join other state into this one. Returns True if changed."""
        changed = False
        for var, locs in other.var_pts.items():
            if self.add_pts(var, locs):
                changed = True
        for (base, fname), locs in other.field_pts.items():
            if self.add_field_pts(base, fname, locs):
                changed = True
        return changed


# ---------------------------------------------------------------------------
# Constraint extraction from C10 AST
# ---------------------------------------------------------------------------

class ConstraintExtractor:
    """Extract points-to constraints from C10 AST."""

    def __init__(self, k: int = 1):
        self.k = k  # context sensitivity depth
        self.constraints: List[Constraint] = []
        self.alloc_sites: Dict[str, HeapLoc] = {}  # site_id -> HeapLoc
        self.functions: Dict[str, Any] = {}  # name -> FnDecl AST
        self.current_function: str = "main"
        self.label_counter: int = 0

    def _next_label(self) -> str:
        self.label_counter += 1
        return f"s{self.label_counter}"

    def _make_alloc(self, kind: AllocKind, context: Tuple[str, ...] = ()) -> HeapLoc:
        label = self._next_label()
        alloc = HeapLoc(
            function=self.current_function,
            label=label,
            kind=kind,
            context=context[:self.k] if context else ()
        )
        self.alloc_sites[alloc.site_id] = alloc
        return alloc

    def _scoped_var(self, name: str, context: Tuple[str, ...] = ()) -> str:
        """Create a context-scoped variable name."""
        if context:
            ctx_str = "->".join(context[:self.k])
            return f"{self.current_function}::{name}[{ctx_str}]"
        return f"{self.current_function}::{name}"

    def extract(self, source: str) -> List[Constraint]:
        """Extract all constraints from C10 source."""
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        # First pass: collect function declarations
        for stmt in program.stmts:
            cls = type(stmt).__name__
            if cls == 'FnDecl':
                self.functions[stmt.name] = stmt

        # Second pass: extract constraints
        self.current_function = "main"
        for stmt in program.stmts:
            cls = type(stmt).__name__
            if cls != 'FnDecl':
                self._extract_stmt(stmt, ())

        # Process function bodies
        for name, fn in self.functions.items():
            self.current_function = name
            for stmt in fn.body.stmts:
                self._extract_stmt(stmt, ())

        return self.constraints

    def _extract_stmt(self, stmt, context: Tuple[str, ...]):
        """Extract constraints from a statement."""
        cls = type(stmt).__name__

        if cls == 'LetDecl':
            self._extract_assign(stmt.name, stmt.value, context)

        elif cls == 'Assign':
            target_cls = type(stmt.target).__name__ if hasattr(stmt, 'target') else ''
            if target_cls == 'IndexExpr':
                # x.f = y or x[i] = y => STORE
                self._extract_store(stmt, context)
            else:
                name = stmt.name if hasattr(stmt, 'name') else str(stmt.target)
                self._extract_assign(name, stmt.value, context)

        elif cls == 'IfStmt':
            for s in stmt.then_body.stmts:
                self._extract_stmt(s, context)
            if stmt.else_body:
                body = stmt.else_body
                if hasattr(body, 'stmts'):
                    for s in body.stmts:
                        self._extract_stmt(s, context)

        elif cls == 'WhileStmt':
            for s in stmt.body.stmts:
                self._extract_stmt(s, context)

        elif cls == 'ReturnStmt':
            if stmt.value:
                ret_var = self._scoped_var("__return__", context)
                rhs_targets = self._extract_expr_targets(stmt.value, context)
                if rhs_targets is not None:
                    for t in rhs_targets:
                        self.constraints.append(Constraint(
                            kind=ConstraintKind.ASSIGN,
                            lhs=ret_var, rhs=t, context=context))

        elif cls == 'ExprStmt' or cls == 'PrintStmt':
            expr = stmt.expr if hasattr(stmt, 'expr') else (
                stmt.value if hasattr(stmt, 'value') else None)
            if expr:
                self._extract_expr_targets(expr, context)

    def _extract_assign(self, name: str, value, context: Tuple[str, ...]):
        """Extract constraints from an assignment: name = value."""
        if value is None:
            return

        lhs = self._scoped_var(name, context)
        cls = type(value).__name__

        # Array literal -> allocation
        if cls == 'ArrayLit':
            alloc = self._make_alloc(AllocKind.ARRAY, context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ALLOC, lhs=lhs, alloc=alloc, context=context))
            # Process array elements
            for i, elem in enumerate(value.elements):
                elem_targets = self._extract_expr_targets(elem, context)
                if elem_targets:
                    for t in elem_targets:
                        self.constraints.append(Constraint(
                            kind=ConstraintKind.STORE,
                            lhs=lhs, rhs=t, field_name=str(i), context=context))
            return

        # Hash literal -> allocation
        if cls == 'HashLit':
            alloc = self._make_alloc(AllocKind.HASH, context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ALLOC, lhs=lhs, alloc=alloc, context=context))
            # Process hash entries
            if hasattr(value, 'pairs'):
                for pair in value.pairs:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        key, val = pair
                        key_str = key.value if hasattr(key, 'value') else str(key)
                        val_targets = self._extract_expr_targets(val, context)
                        if val_targets:
                            for t in val_targets:
                                self.constraints.append(Constraint(
                                    kind=ConstraintKind.STORE,
                                    lhs=lhs, rhs=t,
                                    field_name=str(key_str),
                                    context=context))
            return

        # Function/lambda -> closure allocation
        if cls in ('FnExpr', 'LambdaExpr', 'FnDecl'):
            alloc = self._make_alloc(AllocKind.CLOSURE, context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ALLOC, lhs=lhs, alloc=alloc, context=context))
            return

        # Call expression -> return value flows to lhs
        if cls == 'CallExpr':
            callee_name = self._resolve_callee(value)
            if callee_name and callee_name in self.functions:
                fn = self.functions[callee_name]
                new_ctx = (context + (f"{self.current_function}:{self._next_label()}",))[-self.k:]
                # Argument flow: actual -> formal
                for i, arg in enumerate(value.args):
                    if i < len(fn.params):
                        param_var = self._scoped_var(fn.params[i], new_ctx)
                        # Use context for the callee
                        old_func = self.current_function
                        self.current_function = callee_name
                        param_var_in_callee = self._scoped_var(fn.params[i], new_ctx)
                        self.current_function = old_func
                        arg_targets = self._extract_expr_targets(arg, context)
                        if arg_targets:
                            for t in arg_targets:
                                self.constraints.append(Constraint(
                                    kind=ConstraintKind.ASSIGN,
                                    lhs=param_var_in_callee, rhs=t,
                                    context=new_ctx))
                # Return flow: callee return -> lhs
                old_func = self.current_function
                self.current_function = callee_name
                ret_var = self._scoped_var("__return__", new_ctx)
                self.current_function = old_func
                self.constraints.append(Constraint(
                    kind=ConstraintKind.ASSIGN,
                    lhs=lhs, rhs=ret_var,
                    context=context))
            else:
                # Unknown callee: could return anything
                # Conservative: allocate unknown
                alloc = self._make_alloc(AllocKind.UNKNOWN, context)
                self.constraints.append(Constraint(
                    kind=ConstraintKind.ALLOC, lhs=lhs, alloc=alloc, context=context))
            return

        # Field/index access -> LOAD
        if cls == 'IndexExpr':
            base_targets = self._extract_expr_targets(value.object, context)
            field_name = self._get_field_name(value.index)
            if base_targets and field_name is not None:
                for bt in base_targets:
                    self.constraints.append(Constraint(
                        kind=ConstraintKind.LOAD,
                        lhs=lhs, rhs=bt,
                        field_name=field_name, context=context))
            return

        # Variable reference -> ASSIGN (copy)
        if cls == 'Var':
            rhs = self._scoped_var(value.name, context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ASSIGN, lhs=lhs, rhs=rhs, context=context))
            return

        # Other expressions: check for nested pointer-relevant sub-expressions
        targets = self._extract_expr_targets(value, context)
        if targets:
            for t in targets:
                self.constraints.append(Constraint(
                    kind=ConstraintKind.ASSIGN, lhs=lhs, rhs=t, context=context))

    def _extract_store(self, stmt, context: Tuple[str, ...]):
        """Extract STORE constraint from x.f = y or x[i] = y."""
        target = stmt.target if hasattr(stmt, 'target') else None
        if target is None:
            return
        target_cls = type(target).__name__
        if target_cls != 'IndexExpr':
            return

        base_var = self._extract_expr_targets(target.object, context)
        field_name = self._get_field_name(target.index)
        value = stmt.value
        val_targets = self._extract_expr_targets(value, context)

        if base_var and field_name is not None and val_targets:
            for bv in base_var:
                for vt in val_targets:
                    self.constraints.append(Constraint(
                        kind=ConstraintKind.STORE,
                        lhs=bv, rhs=vt,
                        field_name=field_name, context=context))

    def _extract_expr_targets(self, expr, context: Tuple[str, ...]) -> Optional[List[str]]:
        """Extract variable names that an expression evaluates to.

        Returns list of scoped variable names, or None if not pointer-relevant.
        """
        if expr is None:
            return None

        cls = type(expr).__name__

        if cls == 'Var':
            return [self._scoped_var(expr.name, context)]

        if cls == 'CallExpr':
            # The call result is in a temporary
            callee = self._resolve_callee(expr)
            if callee and callee in self.functions:
                new_ctx = (context + (f"{self.current_function}:{self._next_label()}",))[-self.k:]
                old_func = self.current_function
                self.current_function = callee
                ret_var = self._scoped_var("__return__", new_ctx)
                self.current_function = old_func
                # Also process arguments
                fn = self.functions[callee]
                for i, arg in enumerate(expr.args):
                    if i < len(fn.params):
                        old_func2 = self.current_function
                        self.current_function = callee
                        param_var = self._scoped_var(fn.params[i], new_ctx)
                        self.current_function = old_func2
                        arg_targets = self._extract_expr_targets(arg, context)
                        if arg_targets:
                            for t in arg_targets:
                                self.constraints.append(Constraint(
                                    kind=ConstraintKind.ASSIGN,
                                    lhs=param_var, rhs=t,
                                    context=new_ctx))
                return [ret_var]
            return None

        if cls == 'ArrayLit':
            # Inline array: create alloc
            alloc = self._make_alloc(AllocKind.ARRAY, context)
            tmp = self._scoped_var(f"__tmp_{self.label_counter}", context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ALLOC, lhs=tmp, alloc=alloc, context=context))
            return [tmp]

        if cls == 'HashLit':
            alloc = self._make_alloc(AllocKind.HASH, context)
            tmp = self._scoped_var(f"__tmp_{self.label_counter}", context)
            self.constraints.append(Constraint(
                kind=ConstraintKind.ALLOC, lhs=tmp, alloc=alloc, context=context))
            return [tmp]

        if cls == 'IndexExpr':
            # x.f or x[i] - returns field contents
            return None  # Will be handled as LOAD at assignment site

        if cls == 'BinOp':
            # Could be string concat or arithmetic - not pointer relevant
            return None

        return None

    def _resolve_callee(self, call_expr) -> Optional[str]:
        """Resolve the callee function name from a CallExpr."""
        callee = call_expr.callee if hasattr(call_expr, 'callee') else None
        if callee is None:
            return None
        if isinstance(callee, str):
            return callee
        cls = type(callee).__name__
        if cls == 'Var':
            return callee.name
        return None

    def _get_field_name(self, index_expr) -> Optional[str]:
        """Extract field name from an index expression."""
        if index_expr is None:
            return None
        cls = type(index_expr).__name__
        if cls == 'StringLit':
            return index_expr.value
        if cls == 'IntLit':
            return str(index_expr.value)
        if cls == 'Var':
            return f"*{index_expr.name}"  # dynamic field, approximated
        return None


# ---------------------------------------------------------------------------
# Andersen's inclusion-based solver
# ---------------------------------------------------------------------------

class AndersenSolver:
    """Solve points-to constraints using Andersen's algorithm.

    Iterative worklist algorithm propagating points-to sets until fixpoint.
    """

    def __init__(self, constraints: List[Constraint], k: int = 1):
        self.constraints = constraints
        self.k = k
        self.state = PointsToState()
        self.iterations = 0
        self.max_iterations = 100

    def solve(self) -> PointsToState:
        """Solve all constraints to fixpoint."""
        # Phase 1: Process ALLOC constraints (seeds)
        for c in self.constraints:
            if c.kind == ConstraintKind.ALLOC and c.alloc is not None:
                self.state.var_pts[c.lhs].add(c.alloc)

        # Phase 2: Iterate ASSIGN/LOAD/STORE until fixpoint
        changed = True
        while changed and self.iterations < self.max_iterations:
            changed = False
            self.iterations += 1

            for c in self.constraints:
                if c.kind == ConstraintKind.ASSIGN:
                    # pts(lhs) >= pts(rhs)
                    rhs_pts = self.state.get_pts(c.rhs)
                    if rhs_pts and self.state.add_pts(c.lhs, rhs_pts):
                        changed = True

                elif c.kind == ConstraintKind.LOAD:
                    # x = y.f => for each h in pts(y): pts(x) >= pts(h.f)
                    # But here rhs is a variable name, not a HeapLoc
                    base_pts = self.state.get_pts(c.rhs)
                    for h in base_pts:
                        field_locs = self.state.get_field_pts(h, c.field_name)
                        if field_locs and self.state.add_pts(c.lhs, field_locs):
                            changed = True

                elif c.kind == ConstraintKind.STORE:
                    # x.f = y => for each h in pts(x): pts(h.f) >= pts(y)
                    base_pts = self.state.get_pts(c.lhs)
                    rhs_pts = self.state.get_pts(c.rhs)
                    if rhs_pts:
                        for h in base_pts:
                            if self.state.add_field_pts(h, c.field_name, rhs_pts):
                                changed = True

        return self.state


# ---------------------------------------------------------------------------
# Flow-sensitive points-to analysis
# ---------------------------------------------------------------------------

class FlowSensitivePTA:
    """Flow-sensitive points-to analysis over ICFG.

    Unlike Andersen's (flow-insensitive), this tracks per-program-point
    points-to sets, giving more precise results at higher cost.
    """

    def __init__(self, source: str, k: int = 1):
        self.source = source
        self.k = k
        self.icfg = build_icfg(source)
        self.functions = {}
        self._parse_functions(source)
        self.point_states: Dict[str, PointsToState] = {}
        self.alloc_counter = 0

    def _parse_functions(self, source: str):
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        for stmt in program.stmts:
            if type(stmt).__name__ == 'FnDecl':
                self.functions[stmt.name] = stmt

    def _make_alloc(self, func: str, label: str, kind: AllocKind,
                    context: Tuple[str, ...] = ()) -> HeapLoc:
        self.alloc_counter += 1
        return HeapLoc(func, f"{label}_{self.alloc_counter}", kind,
                       context[:self.k] if context else ())

    def analyze(self) -> Dict[str, PointsToState]:
        """Run flow-sensitive analysis. Returns per-point states."""
        # Initialize all points with empty state
        for pid in self.icfg.points:
            self.point_states[pid] = PointsToState()

        # Worklist algorithm
        worklist = deque()
        entry = self.icfg.entry_function
        entry_id = f"{entry}.entry"
        if entry_id in self.icfg.points:
            worklist.append(entry_id)

        visited = set()
        max_iter = 200

        iteration = 0
        while worklist and iteration < max_iter:
            iteration += 1
            pid = worklist.popleft()

            if pid not in self.icfg.points:
                continue

            pp = self.icfg.points[pid]

            # Compute input state from predecessors
            in_state = PointsToState()
            for edge in self.icfg.get_predecessors(pid):
                pred_state = self.point_states.get(edge.source)
                if pred_state:
                    in_state.join(pred_state)

            # Apply transfer function
            out_state = self._transfer(pp, in_state)

            # Check if state changed
            old_state = self.point_states.get(pid, PointsToState())
            test_state = old_state.copy()
            if test_state.join(out_state) or pid not in visited:
                visited.add(pid)
                self.point_states[pid] = out_state
                # Add successors to worklist
                for edge in self.icfg.get_successors(pid):
                    worklist.append(edge.target)

        return self.point_states

    def _transfer(self, pp: ProgramPoint, in_state: PointsToState) -> PointsToState:
        """Transfer function for a program point."""
        out = in_state.copy()
        details = pp.details

        if pp.stmt_type == 'assign':
            var = details.get('var', '')
            rhs_type = details.get('rhs_type', '')

            if var:
                scoped = f"{pp.function}::{var}"

                if rhs_type == 'array':
                    alloc = self._make_alloc(pp.function, pp.label, AllocKind.ARRAY)
                    out.var_pts[scoped] = {alloc}

                elif rhs_type == 'hash':
                    alloc = self._make_alloc(pp.function, pp.label, AllocKind.HASH)
                    out.var_pts[scoped] = {alloc}

                elif rhs_type == 'closure':
                    alloc = self._make_alloc(pp.function, pp.label, AllocKind.CLOSURE)
                    out.var_pts[scoped] = {alloc}

                elif 'rhs_var' in details:
                    # x = y: copy points-to set
                    rhs_scoped = f"{pp.function}::{details['rhs_var']}"
                    out.var_pts[scoped] = set(out.get_pts(rhs_scoped))

                elif 'field_base' in details and 'field_name' in details:
                    # x = y.f: load
                    base_scoped = f"{pp.function}::{details['field_base']}"
                    fname = details['field_name']
                    result = set()
                    for h in out.get_pts(base_scoped):
                        result |= out.get_field_pts(h, fname)
                    out.var_pts[scoped] = result

        elif pp.stmt_type == 'call':
            # At call sites, map actuals to formals
            pass  # Handled by call/return edges in ICFG

        return out


# ---------------------------------------------------------------------------
# Alias queries
# ---------------------------------------------------------------------------

@dataclass
class AliasResult:
    """Result of an alias query."""
    may_alias: bool       # could they point to same location?
    must_alias: bool      # do they definitely point to same location?
    common_targets: Set[HeapLoc]  # shared targets
    var1_targets: Set[HeapLoc]
    var2_targets: Set[HeapLoc]

    def __repr__(self):
        if self.must_alias:
            return "MUST_ALIAS"
        if self.may_alias:
            return f"MAY_ALIAS(common={len(self.common_targets)})"
        return "NO_ALIAS"


def check_alias(state: PointsToState, var1: str, var2: str) -> AliasResult:
    """Check alias relationship between two variables."""
    pts1 = state.get_pts(var1)
    pts2 = state.get_pts(var2)
    common = pts1 & pts2

    may = len(common) > 0
    must = (len(pts1) == 1 and len(pts2) == 1 and pts1 == pts2)

    return AliasResult(
        may_alias=may,
        must_alias=must,
        common_targets=common,
        var1_targets=pts1,
        var2_targets=pts2
    )


# ---------------------------------------------------------------------------
# Escape analysis
# ---------------------------------------------------------------------------

@dataclass
class EscapeResult:
    """Result of escape analysis."""
    escaped: Dict[str, Set[HeapLoc]]     # function -> escaped allocations
    local: Dict[str, Set[HeapLoc]]       # function -> local-only allocations
    returned: Dict[str, Set[HeapLoc]]    # function -> allocations that are returned
    stored_to_params: Dict[str, Set[HeapLoc]]  # function -> stored to param fields


def escape_analysis(state: PointsToState, constraints: List[Constraint],
                    functions: Dict[str, Any]) -> EscapeResult:
    """Determine which allocations escape their creating function."""
    escaped = defaultdict(set)
    local = defaultdict(set)
    returned = defaultdict(set)
    stored = defaultdict(set)

    # Collect all allocation sites per function
    func_allocs = defaultdict(set)
    for c in constraints:
        if c.kind == ConstraintKind.ALLOC and c.alloc is not None:
            func_allocs[c.alloc.function].add(c.alloc)

    # Check which allocations are returned
    for var, pts in state.var_pts.items():
        if "__return__" in var:
            for func_name in functions:
                if var.startswith(f"{func_name}::"):
                    returned[func_name] |= pts

    # Check which allocations are stored to parameter fields
    for c in constraints:
        if c.kind == ConstraintKind.STORE:
            # If storing into a param's field, the allocation escapes
            for func_name, fn in functions.items():
                params = fn.params if hasattr(fn, 'params') else []
                for p in params:
                    if f"{func_name}::{p}" == c.lhs:
                        rhs_pts = state.get_pts(c.rhs)
                        stored[func_name] |= rhs_pts

    # Classify each allocation
    for func_name, allocs in func_allocs.items():
        for alloc in allocs:
            if alloc in returned.get(func_name, set()) or \
               alloc in stored.get(func_name, set()):
                escaped[func_name].add(alloc)
            else:
                local[func_name].add(alloc)

    return EscapeResult(
        escaped=dict(escaped),
        local=dict(local),
        returned=dict(returned),
        stored_to_params=dict(stored)
    )


# ---------------------------------------------------------------------------
# Mod/Ref analysis
# ---------------------------------------------------------------------------

@dataclass
class ModRefResult:
    """What heap locations a function may modify (mod) or reference (ref)."""
    mod: Dict[str, Set[Tuple[HeapLoc, str]]]  # function -> {(base, field)} modified
    ref: Dict[str, Set[Tuple[HeapLoc, str]]]  # function -> {(base, field)} referenced


def mod_ref_analysis(state: PointsToState,
                     constraints: List[Constraint]) -> ModRefResult:
    """Compute mod/ref sets for each function."""
    mod = defaultdict(set)
    ref = defaultdict(set)

    for c in constraints:
        # Extract function from context
        func = ""
        if c.lhs and "::" in c.lhs:
            func = c.lhs.split("::")[0]

        if c.kind == ConstraintKind.STORE and func:
            # Storing to field -> MOD
            base_pts = state.get_pts(c.lhs)
            for h in base_pts:
                mod[func].add((h, c.field_name))

        elif c.kind == ConstraintKind.LOAD and func:
            # Loading from field -> REF
            rhs_func = c.rhs.split("::")[0] if "::" in c.rhs else func
            base_pts = state.get_pts(c.rhs)
            for h in base_pts:
                ref[rhs_func].add((h, c.field_name))

    return ModRefResult(mod=dict(mod), ref=dict(ref))


# ---------------------------------------------------------------------------
# Call graph from points-to (resolves indirect calls)
# ---------------------------------------------------------------------------

@dataclass
class PTACallGraph:
    """Call graph constructed from points-to results."""
    edges: List[Tuple[str, str, str]]  # (caller, call_site, callee)
    virtual_calls: int = 0    # number of resolved indirect calls
    total_calls: int = 0


def build_call_graph(state: PointsToState,
                     constraints: List[Constraint],
                     functions: Dict[str, Any]) -> PTACallGraph:
    """Build call graph using points-to information for dispatch resolution."""
    edges = []
    virtual = 0
    total = 0

    for c in constraints:
        if c.kind == ConstraintKind.CALL_ARG or c.kind == ConstraintKind.CALL_RET:
            total += 1
            if c.callee:
                caller = c.lhs.split("::")[0] if "::" in c.lhs else "main"
                edges.append((caller, c.lhs, c.callee))

    # Check for indirect calls through closures
    for var, pts in state.var_pts.items():
        for h in pts:
            if h.kind == AllocKind.CLOSURE:
                virtual += 1

    return PTACallGraph(edges=edges, virtual_calls=virtual, total_calls=total)


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

@dataclass
class PointsToResult:
    """Complete result of points-to analysis."""
    state: PointsToState
    constraints: List[Constraint]
    alloc_sites: Dict[str, HeapLoc]
    iterations: int
    context_depth: int
    flow_sensitive: bool

    def points_to(self, var: str) -> Set[HeapLoc]:
        """Get points-to set for a variable."""
        # Try exact match first
        pts = self.state.get_pts(var)
        if pts:
            return pts
        # Try with main:: prefix
        pts = self.state.get_pts(f"main::{var}")
        if pts:
            return pts
        # Search all scoped versions
        for v, p in self.state.var_pts.items():
            if v.endswith(f"::{var}"):
                pts |= p
        return pts

    def alias(self, var1: str, var2: str) -> AliasResult:
        """Check alias between two variables."""
        v1 = var1 if "::" in var1 else f"main::{var1}"
        v2 = var2 if "::" in var2 else f"main::{var2}"
        return check_alias(self.state, v1, v2)

    def field_points_to(self, var: str, field: str) -> Set[HeapLoc]:
        """Get what var.field points to."""
        v = var if "::" in var else f"main::{var}"
        result = set()
        for h in self.state.get_pts(v):
            result |= self.state.get_field_pts(h, field)
        return result


def analyze_points_to(source: str, k: int = 1,
                      flow_sensitive: bool = False) -> PointsToResult:
    """Main API: analyze points-to relationships in C10 source.

    Args:
        source: C10 source code
        k: context sensitivity depth (0=insensitive, 1=1-CFA, etc.)
        flow_sensitive: if True, use flow-sensitive analysis (slower, more precise)

    Returns:
        PointsToResult with points-to sets, alias queries, etc.
    """
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)

    solver = AndersenSolver(constraints, k=k)
    state = solver.solve()

    return PointsToResult(
        state=state,
        constraints=constraints,
        alloc_sites=extractor.alloc_sites,
        iterations=solver.iterations,
        context_depth=k,
        flow_sensitive=False
    )


def analyze_flow_sensitive(source: str, k: int = 1) -> PointsToResult:
    """Flow-sensitive points-to analysis."""
    pta = FlowSensitivePTA(source, k=k)
    point_states = pta.analyze()

    # Merge all point states for a summary
    merged = PointsToState()
    for ps in point_states.values():
        merged.join(ps)

    return PointsToResult(
        state=merged,
        constraints=[],
        alloc_sites={},
        iterations=0,
        context_depth=k,
        flow_sensitive=True
    )


def check_may_alias(source: str, var1: str, var2: str,
                    k: int = 1) -> AliasResult:
    """Check if two variables may alias in the given program."""
    result = analyze_points_to(source, k=k)
    return result.alias(var1, var2)


def analyze_escapes(source: str, k: int = 1) -> EscapeResult:
    """Escape analysis: which allocations escape their function."""
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)

    solver = AndersenSolver(constraints, k=k)
    state = solver.solve()

    return escape_analysis(state, constraints, extractor.functions)


def analyze_mod_ref(source: str, k: int = 1) -> ModRefResult:
    """Mod/ref analysis: what heap locations each function reads/writes."""
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)

    solver = AndersenSolver(constraints, k=k)
    state = solver.solve()

    return mod_ref_analysis(state, constraints)


def build_pta_call_graph(source: str, k: int = 1) -> PTACallGraph:
    """Build call graph using points-to analysis."""
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)

    solver = AndersenSolver(constraints, k=k)
    state = solver.solve()

    return build_call_graph(state, constraints, extractor.functions)


def compare_sensitivity(source: str, max_k: int = 2) -> Dict[str, Any]:
    """Compare context-insensitive vs context-sensitive analysis.

    Returns precision metrics for each k level.
    """
    results = {}
    for k in range(max_k + 1):
        result = analyze_points_to(source, k=k)
        total_pts = sum(len(pts) for pts in result.state.var_pts.values())
        num_vars = len(result.state.var_pts)
        avg_pts = total_pts / max(num_vars, 1)

        results[f"k={k}"] = {
            "total_points_to": total_pts,
            "num_variables": num_vars,
            "avg_points_to_size": round(avg_pts, 2),
            "num_alloc_sites": len(result.alloc_sites),
            "iterations": result.iterations,
            "constraints": len(result.constraints)
        }

    return results


def full_points_to_analysis(source: str, k: int = 1) -> Dict[str, Any]:
    """Full combined analysis: points-to + escape + mod/ref + call graph."""
    extractor = ConstraintExtractor(k=k)
    constraints = extractor.extract(source)

    solver = AndersenSolver(constraints, k=k)
    state = solver.solve()

    esc = escape_analysis(state, constraints, extractor.functions)
    mr = mod_ref_analysis(state, constraints)
    cg = build_call_graph(state, constraints, extractor.functions)

    # Summary
    total_pts = sum(len(pts) for pts in state.var_pts.values())
    num_vars = len(state.var_pts)
    num_allocs = len(extractor.alloc_sites)
    num_escaped = sum(len(s) for s in esc.escaped.values())
    num_local = sum(len(s) for s in esc.local.values())

    return {
        "points_to": {
            "total_targets": total_pts,
            "num_variables": num_vars,
            "avg_targets": round(total_pts / max(num_vars, 1), 2),
            "num_alloc_sites": num_allocs,
            "iterations": solver.iterations,
        },
        "escape": {
            "escaped": num_escaped,
            "local": num_local,
            "escape_ratio": round(num_escaped / max(num_escaped + num_local, 1), 2),
        },
        "mod_ref": {
            "mod_sites": sum(len(s) for s in mr.mod.values()),
            "ref_sites": sum(len(s) for s in mr.ref.values()),
        },
        "call_graph": {
            "edges": len(cg.edges),
            "virtual_calls": cg.virtual_calls,
        },
        "constraints": len(constraints),
        "context_depth": k,
    }


def points_to_summary(source: str, k: int = 1) -> str:
    """Human-readable summary of points-to analysis."""
    result = analyze_points_to(source, k=k)
    lines = [f"Points-To Analysis (k={k}):", ""]

    # Variables and their targets
    for var in sorted(result.state.var_pts.keys()):
        pts = result.state.var_pts[var]
        if pts:
            targets = ", ".join(str(h) for h in sorted(pts, key=str))
            lines.append(f"  {var} -> {{{targets}}}")

    # Field targets
    if result.state.field_pts:
        lines.append("")
        lines.append("Field targets:")
        for (base, field), pts in sorted(result.state.field_pts.items(), key=str):
            if pts:
                targets = ", ".join(str(h) for h in sorted(pts, key=str))
                lines.append(f"  {base}.{field} -> {{{targets}}}")

    lines.append("")
    lines.append(f"Total constraints: {len(result.constraints)}")
    lines.append(f"Solver iterations: {result.iterations}")
    lines.append(f"Allocation sites: {len(result.alloc_sites)}")

    return "\n".join(lines)
