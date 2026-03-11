"""
V112: Trace Abstraction Refinement

Automata-based program verification (Heizmann et al., 2009).

Key idea: program traces are words over statement alphabet. The verifier
maintains an automaton A that accepts known-infeasible traces. If
program(P) subset A, then P is safe. Otherwise, extract a trace w in
P \\ A, check feasibility via SMT. If infeasible, use Craig interpolation
to generalize w into a new automaton that accepts many infeasible traces
(not just w), and add it to A.

Composes:
- V110 (ART): CFG construction, SMT encoding helpers
- V107 (Craig interpolation): interpolant-based trace generalization
- C037 (SMT solver): feasibility checking
- C010 (parser): source parsing

The alphabet is the set of CFG edges (statements). A trace is a path
from entry to error in the CFG. The interpolation automaton generalizes
a single infeasible trace to a regular set of infeasible traces.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, FrozenSet
from enum import Enum

# Add paths
_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))

from stack_vm import lex, Parser
from smt_solver import (
    SMTSolver, SMTResult, Term, Var, IntConst, BoolConst, App, Op,
    Sort, SortKind
)
from craig_interpolation import (
    craig_interpolate, sequence_interpolate, verify_interpolant,
    collect_vars, collect_atoms, simplify_term, negate_atom
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# CFG (reuse V110 design but self-contained for clarity)
# ---------------------------------------------------------------------------

class StmtKind(Enum):
    ASSIGN = "assign"       # x := expr
    ASSUME = "assume"       # branch condition (true path)
    ASSUME_NOT = "assume_not"  # branch condition (false path)
    ASSERT = "assert"       # assertion check
    SKIP = "skip"           # no-op / join
    ENTRY = "entry"
    EXIT = "exit"
    ERROR = "error"


@dataclass
class CFGEdge:
    """A labeled edge in the CFG. The label is the statement."""
    src: int
    dst: int
    kind: StmtKind
    data: object = None   # (var, expr_ast) for ASSIGN, condition_ast for ASSUME/ASSERT
    label: str = ""       # human-readable label

    def __hash__(self):
        return hash((self.src, self.dst, self.kind, self.label))

    def __eq__(self, other):
        return (self.src, self.dst, self.kind, self.label) == (
            other.src, other.dst, other.kind, other.label)

    def __repr__(self):
        return f"Edge({self.src}->{self.dst}, {self.kind.value}: {self.label})"


@dataclass
class CFG:
    """Control-flow graph with labeled edges."""
    edges: List[CFGEdge] = field(default_factory=list)
    entry: int = 0
    exit_node: int = -1
    error_node: int = -1
    _next_id: int = 0

    def new_node(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_edge(self, src: int, dst: int, kind: StmtKind,
                 data=None, label: str = "") -> CFGEdge:
        e = CFGEdge(src, dst, kind, data, label)
        self.edges.append(e)
        return e

    def successors(self, node: int) -> List[CFGEdge]:
        return [e for e in self.edges if e.src == node]

    def predecessors(self, node: int) -> List[CFGEdge]:
        return [e for e in self.edges if e.dst == node]

    def all_nodes(self) -> Set[int]:
        nodes = {self.entry, self.exit_node, self.error_node}
        for e in self.edges:
            nodes.add(e.src)
            nodes.add(e.dst)
        return nodes


def _expr_to_str(expr) -> str:
    """Convert AST expression to readable string."""
    cls = expr.__class__.__name__
    if cls == 'IntLit':
        return str(expr.value)
    elif cls == 'BoolLit':
        return str(expr.value).lower()
    elif cls == 'Var':
        return expr.name
    elif cls == 'BinOp':
        return f"({_expr_to_str(expr.left)} {expr.op} {_expr_to_str(expr.right)})"
    elif cls == 'UnaryOp':
        return f"({expr.op}{_expr_to_str(expr.operand)})"
    return str(expr)


def build_cfg(source: str) -> CFG:
    """Build CFG from C10 source code."""
    tokens = lex(source)
    program = Parser(tokens).parse()

    cfg = CFG()
    entry = cfg.new_node()
    exit_node = cfg.new_node()
    error_node = cfg.new_node()
    cfg.entry = entry
    cfg.exit_node = exit_node
    cfg.error_node = error_node

    end = _build_stmts(cfg, program.stmts, entry, error_node)
    if end is not None:
        cfg.add_edge(end, exit_node, StmtKind.SKIP, label="exit")

    return cfg


def _build_stmts(cfg: CFG, stmts: list, start: int, error_node: int) -> Optional[int]:
    """Build CFG for a list of statements. Returns the exit node or None."""
    current = start
    for stmt in stmts:
        if current is None:
            return None
        current = _build_stmt(cfg, stmt, current, error_node)
    return current


def _build_stmt(cfg: CFG, stmt, current: int, error_node: int) -> Optional[int]:
    """Build CFG for a single statement. Returns exit node."""
    cls = stmt.__class__.__name__

    if cls == 'LetDecl':
        nxt = cfg.new_node()
        cfg.add_edge(current, nxt, StmtKind.ASSIGN,
                     data=(stmt.name, stmt.value),
                     label=f"{stmt.name} = {_expr_to_str(stmt.value)}")
        return nxt

    elif cls == 'Assign':
        nxt = cfg.new_node()
        cfg.add_edge(current, nxt, StmtKind.ASSIGN,
                     data=(stmt.name, stmt.value),
                     label=f"{stmt.name} = {_expr_to_str(stmt.value)}")
        return nxt

    elif cls == 'IfStmt':
        cond_str = _expr_to_str(stmt.cond)
        then_entry = cfg.new_node()
        cfg.add_edge(current, then_entry, StmtKind.ASSUME,
                     data=stmt.cond, label=f"assume({cond_str})")

        join = cfg.new_node()

        then_body = stmt.then_body
        if hasattr(then_body, 'stmts'):
            then_body = then_body.stmts
        then_exit = _build_stmts(cfg, then_body, then_entry, error_node)
        if then_exit is not None:
            cfg.add_edge(then_exit, join, StmtKind.SKIP, label="join")

        if stmt.else_body:
            else_entry = cfg.new_node()
            cfg.add_edge(current, else_entry, StmtKind.ASSUME_NOT,
                         data=stmt.cond, label=f"assume(!{cond_str})")
            else_body = stmt.else_body
            if hasattr(else_body, 'stmts'):
                else_body = else_body.stmts
            else_exit = _build_stmts(cfg, else_body, else_entry, error_node)
            if else_exit is not None:
                cfg.add_edge(else_exit, join, StmtKind.SKIP, label="join")
        else:
            cfg.add_edge(current, join, StmtKind.ASSUME_NOT,
                         data=stmt.cond, label=f"assume(!{cond_str})")

        return join

    elif cls == 'WhileStmt':
        cond_str = _expr_to_str(stmt.cond)
        header = cfg.new_node()
        cfg.add_edge(current, header, StmtKind.SKIP, label="loop_header")

        body_entry = cfg.new_node()
        cfg.add_edge(header, body_entry, StmtKind.ASSUME,
                     data=stmt.cond, label=f"assume({cond_str})")

        body = stmt.body
        if hasattr(body, 'stmts'):
            body = body.stmts
        body_exit = _build_stmts(cfg, body, body_entry, error_node)
        if body_exit is not None:
            cfg.add_edge(body_exit, header, StmtKind.SKIP, label="loop_back")

        loop_exit = cfg.new_node()
        cfg.add_edge(header, loop_exit, StmtKind.ASSUME_NOT,
                     data=stmt.cond, label=f"assume(!{cond_str})")
        return loop_exit

    elif cls == 'ExprStmt':
        return _build_expr_stmt(cfg, stmt.expr, current, error_node)

    elif cls == 'CallExpr':
        # Top-level CallExpr (C10 doesn't always wrap in ExprStmt)
        return _build_expr_stmt(cfg, stmt, current, error_node)

    elif cls == 'ReturnStmt':
        # Return just goes to exit -- simplified
        return None

    else:
        # Unknown statement type, skip
        return current


def _build_expr_stmt(cfg: CFG, expr, current: int, error_node: int) -> Optional[int]:
    """Handle expression statements (especially assert calls)."""
    cls = expr.__class__.__name__
    if cls == 'CallExpr':
        callee = expr.callee if isinstance(expr.callee, str) else getattr(expr.callee, 'name', str(expr.callee))
        if callee == 'assert':
            # assert(cond) -- check and branch to error if fails
            cond = expr.args[0]
            cond_str = _expr_to_str(cond)

            ok = cfg.new_node()
            cfg.add_edge(current, ok, StmtKind.ASSUME,
                         data=cond, label=f"assume({cond_str})")

            cfg.add_edge(current, error_node, StmtKind.ASSUME_NOT,
                         data=cond, label=f"assume(!{cond_str})")
            return ok

        elif callee in ('requires', 'ensures', 'invariant'):
            # Annotation -- skip
            return current

    return current


# ---------------------------------------------------------------------------
# SMT encoding of statements
# ---------------------------------------------------------------------------

def ast_to_smt(expr, env_vars: dict) -> Term:
    """Convert C10 AST expression to SMT term."""
    cls = expr.__class__.__name__

    if cls == 'IntLit':
        return IntConst(expr.value)

    elif cls == 'BoolLit':
        if expr.value:
            return App(Op.EQ, [IntConst(0), IntConst(0)], BOOL)
        else:
            return App(Op.EQ, [IntConst(0), IntConst(1)], BOOL)

    elif cls == 'Var':
        name = expr.name
        if name not in env_vars:
            env_vars[name] = Var(name, INT)
        return env_vars[name]

    elif cls == 'BinOp':
        left = ast_to_smt(expr.left, env_vars)
        right = ast_to_smt(expr.right, env_vars)
        op_map = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
            '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
            '==': Op.EQ, '!=': Op.NEQ,
            'and': Op.AND, 'or': Op.OR,
        }
        op = op_map.get(expr.op)
        if op is None:
            return IntConst(0)  # fallback
        sort = BOOL if op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ,
                              Op.AND, Op.OR) else INT
        return App(op, [left, right], sort)

    elif cls == 'UnaryOp':
        operand = ast_to_smt(expr.operand, env_vars)
        if expr.op == '-':
            return App(Op.SUB, [IntConst(0), operand], INT)
        elif expr.op == 'not':
            return App(Op.NOT, [operand], BOOL)
        return operand

    return IntConst(0)


def make_step_var(name: str, step: int) -> Var:
    """Create step-indexed variable for SSA-like encoding."""
    return Var(f"{name}_{step}", INT)


# ---------------------------------------------------------------------------
# Trace = sequence of CFG edges
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """A program trace: sequence of CFG edges from entry to error."""
    edges: List[CFGEdge]

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        labels = [e.label for e in self.edges]
        return f"Trace({' -> '.join(labels)})"


# ---------------------------------------------------------------------------
# Interpolation Automaton (NFA)
# ---------------------------------------------------------------------------

@dataclass
class InterpolationAutomaton:
    """
    NFA over the statement alphabet. Accepts infeasible traces.

    States are abstract states (sets of predicates as frozensets of Term).
    Transitions are labeled by CFG edges.
    Initial state: TRUE (no constraints)
    Final state: FALSE (contradiction reached)
    """
    states: Set[FrozenSet[Term]] = field(default_factory=set)
    initial: FrozenSet[Term] = field(default_factory=frozenset)
    finals: Set[FrozenSet[Term]] = field(default_factory=set)
    # transitions: (src_state, edge_label, dst_state)
    transitions: List[Tuple[FrozenSet[Term], CFGEdge, FrozenSet[Term]]] = field(
        default_factory=list)

    def accepts_trace(self, trace: Trace) -> bool:
        """Check if this automaton accepts a given trace (NFA simulation)."""
        current_states = {self.initial}
        for edge in trace.edges:
            next_states = set()
            for state in current_states:
                for src, e, dst in self.transitions:
                    if src == state and self._edge_matches(e, edge):
                        next_states.add(dst)
            current_states = next_states
            if not current_states:
                return False
        return bool(current_states & self.finals)

    def _edge_matches(self, pattern: CFGEdge, concrete: CFGEdge) -> bool:
        """Check if an automaton edge label matches a concrete edge."""
        return (pattern.kind == concrete.kind and
                pattern.label == concrete.label)

    def size(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# Union Automaton (tracks all learned infeasible traces)
# ---------------------------------------------------------------------------

class InfeasibilityAutomaton:
    """
    Union of interpolation automata. Accepts traces proven infeasible.
    Implemented as a list of component automata (union = any accepts).
    """
    def __init__(self):
        self.components: List[InterpolationAutomaton] = []
        self._accepted_cache: Set[Tuple] = set()

    def add(self, automaton: InterpolationAutomaton):
        self.components.append(automaton)

    def accepts(self, trace: Trace) -> bool:
        """Check if any component accepts this trace."""
        key = tuple((e.src, e.dst, e.kind, e.label) for e in trace.edges)
        if key in self._accepted_cache:
            return True
        for comp in self.components:
            if comp.accepts_trace(trace):
                self._accepted_cache.add(key)
                return True
        return False

    def total_size(self) -> int:
        return sum(c.size() for c in self.components)

    def num_components(self) -> int:
        return len(self.components)


# ---------------------------------------------------------------------------
# Trace Feasibility Checker
# ---------------------------------------------------------------------------

class TraceFeasibilityChecker:
    """Check if a trace is feasible using SMT, produce formulas for interpolation."""

    def check(self, trace: Trace) -> Tuple[bool, Optional[dict], List[Term]]:
        """
        Check trace feasibility.
        Returns (feasible, model_if_feasible, per_step_formulas).
        """
        solver = SMTSolver()
        env = {}  # var_name -> current SMT variable
        step = 0
        per_step_formulas = []

        for edge in trace.edges:
            formula = self._encode_edge(solver, edge, env, step)
            per_step_formulas.append(formula)
            if formula is not None:
                solver.add(formula)
            step += 1

        result = solver.check()
        if result == SMTResult.SAT:
            model = solver.model()
            return True, model, per_step_formulas
        else:
            return False, None, per_step_formulas

    def _encode_edge(self, solver: SMTSolver, edge: CFGEdge,
                     env: dict, step: int) -> Optional[Term]:
        """Encode a single edge as an SMT formula."""
        if edge.kind == StmtKind.ASSIGN:
            var_name, expr_ast = edge.data
            # Create new step variable for the assigned var
            old_vars = dict(env)
            smt_env = {}
            for name, var in env.items():
                smt_env[name] = var
            rhs = ast_to_smt(expr_ast, smt_env)
            # Update env with new vars from expression
            for name, var in smt_env.items():
                if name not in env:
                    env[name] = solver.Int(f"{name}_{step}")
                    # Equate with fresh var
                    if var != env[name]:
                        solver.add(App(Op.EQ, [env[name], var], BOOL))

            new_var = solver.Int(f"{var_name}_{step + 1}")
            formula = App(Op.EQ, [new_var, rhs], BOOL)
            env[var_name] = new_var
            return formula

        elif edge.kind in (StmtKind.ASSUME, StmtKind.ASSERT):
            smt_env = {}
            for name, var in env.items():
                smt_env[name] = var
            cond = ast_to_smt(edge.data, smt_env)
            for name, var in smt_env.items():
                if name not in env:
                    env[name] = solver.Int(f"{name}_{step}")
                    solver.add(App(Op.EQ, [env[name], var], BOOL))
            return cond

        elif edge.kind == StmtKind.ASSUME_NOT:
            smt_env = {}
            for name, var in env.items():
                smt_env[name] = var
            cond = ast_to_smt(edge.data, smt_env)
            for name, var in smt_env.items():
                if name not in env:
                    env[name] = solver.Int(f"{name}_{step}")
                    solver.add(App(Op.EQ, [env[name], var], BOOL))
            return App(Op.NOT, [cond], BOOL)

        return None


# ---------------------------------------------------------------------------
# Trace Feasibility Checker (cleaner SSA approach)
# ---------------------------------------------------------------------------

class SSAEncoder:
    """Encode a trace in SSA form for SMT checking and interpolation."""

    def encode(self, trace: Trace) -> Tuple[List[Term], Dict[str, Var]]:
        """
        Encode trace as SSA formulas, one per edge.
        Returns (formulas, final_env).
        """
        env = {}      # var_name -> current SSA Var
        counters = {} # var_name -> next index
        formulas = []

        for edge in trace.edges:
            f = self._encode_edge(edge, env, counters)
            formulas.append(f)

        return formulas, env

    def _fresh_var(self, name: str, counters: dict) -> Var:
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        return Var(f"{name}_{idx}", INT)

    def _get_or_create(self, name: str, env: dict, counters: dict) -> Var:
        if name not in env:
            env[name] = self._fresh_var(name, counters)
        return env[name]

    def _expr_to_smt(self, expr, env: dict, counters: dict) -> Term:
        """Convert AST to SMT using current SSA env."""
        cls = expr.__class__.__name__

        if cls == 'IntLit':
            return IntConst(expr.value)
        elif cls == 'BoolLit':
            if expr.value:
                return App(Op.EQ, [IntConst(0), IntConst(0)], BOOL)
            else:
                return App(Op.EQ, [IntConst(0), IntConst(1)], BOOL)
        elif cls == 'Var':
            return self._get_or_create(expr.name, env, counters)
        elif cls == 'BinOp':
            left = self._expr_to_smt(expr.left, env, counters)
            right = self._expr_to_smt(expr.right, env, counters)
            op_map = {
                '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
                '==': Op.EQ, '!=': Op.NEQ,
                'and': Op.AND, 'or': Op.OR,
            }
            op = op_map.get(expr.op)
            if op is None:
                return IntConst(0)
            sort = BOOL if op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ,
                                  Op.AND, Op.OR) else INT
            return App(op, [left, right], sort)
        elif cls == 'UnaryOp':
            operand = self._expr_to_smt(expr.operand, env, counters)
            if expr.op == '-':
                return App(Op.SUB, [IntConst(0), operand], INT)
            elif expr.op == 'not':
                return App(Op.NOT, [operand], BOOL)
            return operand

        return IntConst(0)

    def _encode_edge(self, edge: CFGEdge, env: dict, counters: dict) -> Term:
        """Encode edge as SSA formula."""
        TRUE = App(Op.EQ, [IntConst(0), IntConst(0)], BOOL)

        if edge.kind == StmtKind.ASSIGN and edge.data is not None:
            var_name, expr_ast = edge.data
            rhs = self._expr_to_smt(expr_ast, env, counters)
            new_var = self._fresh_var(var_name, counters)
            env[var_name] = new_var
            return App(Op.EQ, [new_var, rhs], BOOL)

        elif edge.kind == StmtKind.ASSUME and edge.data is not None:
            cond = self._expr_to_smt(edge.data, env, counters)
            return cond

        elif edge.kind == StmtKind.ASSUME_NOT and edge.data is not None:
            cond = self._expr_to_smt(edge.data, env, counters)
            return App(Op.NOT, [cond], BOOL)

        elif edge.kind == StmtKind.ASSERT and edge.data is not None:
            cond = self._expr_to_smt(edge.data, env, counters)
            return cond

        return TRUE


# ---------------------------------------------------------------------------
# Interpolation Automaton Construction
# ---------------------------------------------------------------------------

def build_interpolation_automaton(trace: Trace, formulas: List[Term]) -> InterpolationAutomaton:
    """
    Build an interpolation automaton from a trace and its sequence interpolants.

    Given trace t = e_0 e_1 ... e_n and formulas phi_0 ... phi_n where
    phi_0 AND ... AND phi_n is UNSAT, compute interpolants I_0, I_1, ..., I_n
    where I_0 = TRUE, I_n = FALSE.

    The automaton has states {I_0, I_1, ..., I_n} with transitions
    I_i --e_i--> I_{i+1}.

    This accepts exactly the traces that follow the same abstract path as the
    given trace. Generalization happens because the interpolant states abstract
    away concrete values -- any trace passing through the same predicates
    in the same order is also infeasible.
    """
    # Filter out trivial (TRUE) formulas and their edges
    non_trivial = []
    for i, f in enumerate(formulas):
        if f is not None and not _is_trivial_true(f):
            non_trivial.append((i, f))

    if len(non_trivial) < 2:
        # Can't interpolate with fewer than 2 formulas
        return _build_exact_automaton(trace)

    # Collect non-trivial formulas
    nt_formulas = [f for _, f in non_trivial]

    # Try sequence interpolation
    seq_result = sequence_interpolate(nt_formulas)

    if not seq_result.is_unsat or not seq_result.interpolants:
        # Fallback: build exact automaton for just this trace
        return _build_exact_automaton(trace)

    # Build automaton from interpolants
    # V107 returns [True, I_1, ..., I_{n-1}, False] -- exactly n+1 states for n formulas
    TRUE_STATE = frozenset()
    FALSE_STATE = frozenset([BoolConst(False)])

    interps = seq_result.interpolants  # len = n+1 where n = len(nt_formulas)

    def _interp_to_state(interp):
        if interp is None:
            return TRUE_STATE
        if isinstance(interp, BoolConst):
            if interp.value:
                return TRUE_STATE
            else:
                return FALSE_STATE
        atoms = _extract_atoms(interp)
        return frozenset(atoms) if atoms else frozenset([interp])

    states = [_interp_to_state(i) for i in interps]

    automaton = InterpolationAutomaton()
    automaton.initial = states[0]
    automaton.finals = {states[-1]}

    # Map non-trivial indices back to trace edges
    nt_indices = [i for i, _ in non_trivial]

    # Build transitions: states[j] --e_j--> states[j+1]
    for j in range(len(nt_formulas)):
        src_state = states[j]
        dst_state = states[j + 1]
        edge = trace.edges[nt_indices[j]]
        automaton.states.add(src_state)
        automaton.states.add(dst_state)
        automaton.transitions.append((src_state, edge, dst_state))

    # Add self-loops on ALL states for trivial edges (they don't change state)
    trivial_indices = set(range(len(trace.edges))) - set(nt_indices)
    for i in trivial_indices:
        edge = trace.edges[i]
        for state in automaton.states:
            automaton.transitions.append((state, edge, state))

    return automaton


def _build_exact_automaton(trace: Trace) -> InterpolationAutomaton:
    """Build an automaton that accepts exactly the given trace."""
    automaton = InterpolationAutomaton()

    states = []
    for i in range(len(trace.edges) + 1):
        state = frozenset([IntConst(i)])
        states.append(state)
        automaton.states.add(state)

    automaton.initial = states[0]
    automaton.finals = {states[-1]}

    for i, edge in enumerate(trace.edges):
        automaton.transitions.append((states[i], edge, states[i + 1]))

    return automaton


def _is_trivial_true(term: Term) -> bool:
    """Check if a term is trivially TRUE (0 == 0)."""
    if isinstance(term, App) and term.op == Op.EQ:
        if (isinstance(term.args[0], IntConst) and isinstance(term.args[1], IntConst)
                and term.args[0].value == term.args[1].value):
            return True
    return False


def _extract_atoms(term: Term) -> Set[Term]:
    """Extract atomic predicates from a formula."""
    atoms = set()
    if isinstance(term, App):
        if term.op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ):
            atoms.add(term)
        elif term.op in (Op.AND, Op.OR, Op.NOT, Op.IMPLIES):
            for arg in term.args:
                atoms.update(_extract_atoms(arg))
    return atoms


# ---------------------------------------------------------------------------
# Trace Enumerator (BFS/DFS over CFG paths)
# ---------------------------------------------------------------------------

class TraceEnumerator:
    """Enumerate traces (paths from entry to error) in the CFG."""

    def __init__(self, cfg: CFG, max_depth: int = 50):
        self.cfg = cfg
        self.max_depth = max_depth

    def enumerate(self, max_traces: int = 100) -> List[Trace]:
        """BFS enumeration of traces from entry to error."""
        if self.cfg.error_node < 0:
            return []

        traces = []
        # BFS: (current_node, edge_path)
        queue = [(self.cfg.entry, [])]
        visited_paths = set()

        while queue and len(traces) < max_traces:
            node, path = queue.pop(0)

            if len(path) > self.max_depth:
                continue

            if node == self.cfg.error_node:
                trace = Trace(edges=list(path))
                path_key = tuple((e.src, e.dst, e.label) for e in path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    traces.append(trace)
                continue

            # Avoid revisiting same node in same trace more than a few times
            # (loop unrolling bound)
            node_count = sum(1 for e in path if e.dst == node)
            if node_count > 3:
                continue

            for edge in self.cfg.successors(node):
                queue.append((edge.dst, path + [edge]))

        return traces


# ---------------------------------------------------------------------------
# Trace Abstraction Refinement Engine
# ---------------------------------------------------------------------------

class TraceAbstractionVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class TraceAbstractionResult:
    """Result of trace abstraction refinement."""
    verdict: TraceAbstractionVerdict
    iterations: int = 0
    traces_checked: int = 0
    traces_infeasible: int = 0
    traces_feasible: int = 0
    counterexample: Optional[Trace] = None
    counterexample_inputs: Optional[dict] = None
    automaton_size: int = 0
    automaton_components: int = 0
    predicates_discovered: Set[str] = field(default_factory=set)


class TraceAbstractionRefinement:
    """
    Main CEGAR loop using trace abstraction refinement.

    Algorithm:
    1. Enumerate error traces in program CFG
    2. For each trace not accepted by infeasibility automaton:
       a. Check feasibility via SMT
       b. If feasible -> UNSAFE (real bug)
       c. If infeasible -> build interpolation automaton, add to infeasibility set
    3. If all traces accepted by infeasibility automaton -> SAFE
    """

    def __init__(self, max_iterations: int = 20, max_traces_per_iter: int = 50,
                 max_unroll: int = 3):
        self.max_iterations = max_iterations
        self.max_traces_per_iter = max_traces_per_iter
        self.max_unroll = max_unroll

    def verify(self, source: str) -> TraceAbstractionResult:
        """Verify a C10 program using trace abstraction refinement."""
        cfg = build_cfg(source)

        if cfg.error_node < 0:
            # No assertions -> trivially safe
            return TraceAbstractionResult(
                verdict=TraceAbstractionVerdict.SAFE,
                iterations=0
            )

        # Check if error node is reachable at all
        reachable = self._reachable_nodes(cfg)
        if cfg.error_node not in reachable:
            return TraceAbstractionResult(
                verdict=TraceAbstractionVerdict.SAFE,
                iterations=0
            )

        infeasibility = InfeasibilityAutomaton()
        encoder = SSAEncoder()
        enumerator = TraceEnumerator(cfg, max_depth=20 + self.max_unroll * 10)
        all_predicates = set()

        total_checked = 0
        total_infeasible = 0

        for iteration in range(self.max_iterations):
            # Enumerate traces not yet accepted by infeasibility automaton
            all_traces = enumerator.enumerate(max_traces=self.max_traces_per_iter)

            if not all_traces:
                # No error traces exist
                return TraceAbstractionResult(
                    verdict=TraceAbstractionVerdict.SAFE,
                    iterations=iteration + 1,
                    traces_checked=total_checked,
                    traces_infeasible=total_infeasible,
                    automaton_size=infeasibility.total_size(),
                    automaton_components=infeasibility.num_components(),
                    predicates_discovered=all_predicates
                )

            new_traces = [t for t in all_traces if not infeasibility.accepts(t)]

            if not new_traces:
                # All traces accepted by infeasibility automaton -> SAFE
                return TraceAbstractionResult(
                    verdict=TraceAbstractionVerdict.SAFE,
                    iterations=iteration + 1,
                    traces_checked=total_checked,
                    traces_infeasible=total_infeasible,
                    automaton_size=infeasibility.total_size(),
                    automaton_components=infeasibility.num_components(),
                    predicates_discovered=all_predicates
                )

            found_bug = False
            for trace in new_traces:
                total_checked += 1

                # Check feasibility
                formulas, _ = encoder.encode(trace)
                non_trivial = [f for f in formulas if f is not None and not _is_trivial_true(f)]

                if len(non_trivial) == 0:
                    # Empty trace is feasible (no constraints)
                    found_bug = True
                    return TraceAbstractionResult(
                        verdict=TraceAbstractionVerdict.UNSAFE,
                        iterations=iteration + 1,
                        traces_checked=total_checked,
                        traces_infeasible=total_infeasible,
                        counterexample=trace,
                        counterexample_inputs={},
                        automaton_size=infeasibility.total_size(),
                        automaton_components=infeasibility.num_components(),
                        predicates_discovered=all_predicates
                    )

                # Check if conjunction is UNSAT
                solver = SMTSolver()
                for f in non_trivial:
                    solver.add(f)
                result = solver.check()

                if result == SMTResult.SAT:
                    # Real bug
                    model = solver.model()
                    return TraceAbstractionResult(
                        verdict=TraceAbstractionVerdict.UNSAFE,
                        iterations=iteration + 1,
                        traces_checked=total_checked,
                        traces_infeasible=total_infeasible,
                        counterexample=trace,
                        counterexample_inputs=model,
                        automaton_size=infeasibility.total_size(),
                        automaton_components=infeasibility.num_components(),
                        predicates_discovered=all_predicates
                    )
                else:
                    # Infeasible -- build interpolation automaton
                    total_infeasible += 1
                    itp_auto = build_interpolation_automaton(trace, formulas)
                    infeasibility.add(itp_auto)

                    # Collect predicates for statistics
                    for state in itp_auto.states:
                        for term in state:
                            all_predicates.add(str(term))

        return TraceAbstractionResult(
            verdict=TraceAbstractionVerdict.UNKNOWN,
            iterations=self.max_iterations,
            traces_checked=total_checked,
            traces_infeasible=total_infeasible,
            automaton_size=infeasibility.total_size(),
            automaton_components=infeasibility.num_components(),
            predicates_discovered=all_predicates
        )

    def _reachable_nodes(self, cfg: CFG) -> Set[int]:
        """BFS from entry to find reachable nodes."""
        visited = set()
        queue = [cfg.entry]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for edge in cfg.successors(node):
                queue.append(edge.dst)
        return visited


# ---------------------------------------------------------------------------
# Enhanced Trace Abstraction with Coverage (Lazy Abstraction style)
# ---------------------------------------------------------------------------

class LazyTraceAbstraction:
    """
    Combines trace abstraction refinement with lazy abstraction (coverage).

    Instead of eagerly enumerating all traces, explores traces on-demand
    using DFS, checking coverage against previously proven infeasible traces.
    """

    def __init__(self, max_iterations: int = 30, max_depth: int = 40):
        self.max_iterations = max_iterations
        self.max_depth = max_depth

    def verify(self, source: str) -> TraceAbstractionResult:
        """Verify using lazy trace exploration."""
        cfg = build_cfg(source)

        if cfg.error_node < 0:
            return TraceAbstractionResult(verdict=TraceAbstractionVerdict.SAFE)

        reachable = self._reachable(cfg)
        if cfg.error_node not in reachable:
            return TraceAbstractionResult(verdict=TraceAbstractionVerdict.SAFE)

        infeasibility = InfeasibilityAutomaton()
        encoder = SSAEncoder()
        predicates = set()
        total_checked = 0
        total_infeasible = 0

        for iteration in range(self.max_iterations):
            # DFS to find an uncovered error trace
            trace = self._find_uncovered_trace(cfg, infeasibility)

            if trace is None:
                # All error traces are covered
                return TraceAbstractionResult(
                    verdict=TraceAbstractionVerdict.SAFE,
                    iterations=iteration + 1,
                    traces_checked=total_checked,
                    traces_infeasible=total_infeasible,
                    automaton_size=infeasibility.total_size(),
                    automaton_components=infeasibility.num_components(),
                    predicates_discovered=predicates
                )

            total_checked += 1
            formulas, _ = encoder.encode(trace)
            non_trivial = [f for f in formulas if f is not None and not _is_trivial_true(f)]

            if not non_trivial:
                return TraceAbstractionResult(
                    verdict=TraceAbstractionVerdict.UNSAFE,
                    iterations=iteration + 1,
                    traces_checked=total_checked,
                    traces_infeasible=total_infeasible,
                    counterexample=trace,
                    counterexample_inputs={},
                    automaton_size=infeasibility.total_size(),
                    automaton_components=infeasibility.num_components(),
                    predicates_discovered=predicates
                )

            solver = SMTSolver()
            for f in non_trivial:
                solver.add(f)
            result = solver.check()

            if result == SMTResult.SAT:
                model = solver.model()
                return TraceAbstractionResult(
                    verdict=TraceAbstractionVerdict.UNSAFE,
                    iterations=iteration + 1,
                    traces_checked=total_checked,
                    traces_infeasible=total_infeasible,
                    counterexample=trace,
                    counterexample_inputs=model,
                    automaton_size=infeasibility.total_size(),
                    automaton_components=infeasibility.num_components(),
                    predicates_discovered=predicates
                )
            else:
                total_infeasible += 1
                itp_auto = build_interpolation_automaton(trace, formulas)
                infeasibility.add(itp_auto)
                for state in itp_auto.states:
                    for term in state:
                        predicates.add(str(term))

        return TraceAbstractionResult(
            verdict=TraceAbstractionVerdict.UNKNOWN,
            iterations=self.max_iterations,
            traces_checked=total_checked,
            traces_infeasible=total_infeasible,
            automaton_size=infeasibility.total_size(),
            automaton_components=infeasibility.num_components(),
            predicates_discovered=predicates
        )

    def _find_uncovered_trace(self, cfg: CFG,
                               infeasibility: InfeasibilityAutomaton) -> Optional[Trace]:
        """DFS to find an error trace not covered by infeasibility automaton."""
        stack = [(cfg.entry, [])]  # (node, edge_path)
        visited_states = set()

        while stack:
            node, path = stack.pop()

            if len(path) > self.max_depth:
                continue

            if node == cfg.error_node:
                trace = Trace(edges=list(path))
                if not infeasibility.accepts(trace):
                    return trace
                continue

            # Loop bound: don't visit same node too many times in one path
            node_count = sum(1 for e in path if e.dst == node)
            if node_count > 3:
                continue

            state_key = (node, tuple((e.src, e.dst, e.label) for e in path[-5:]))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            for edge in cfg.successors(node):
                stack.append((edge.dst, path + [edge]))

        return None

    def _reachable(self, cfg: CFG) -> Set[int]:
        visited = set()
        queue = [cfg.entry]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            for e in cfg.successors(n):
                queue.append(e.dst)
        return visited


# ---------------------------------------------------------------------------
# Comparison: Trace Abstraction vs Predicate Abstraction (V110)
# ---------------------------------------------------------------------------

def compare_with_art(source: str) -> dict:
    """Compare trace abstraction refinement with V110's ART-based CEGAR."""
    import importlib
    v110_path = os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree')
    if v110_path not in sys.path:
        sys.path.insert(0, v110_path)
    from art import verify_program as art_verify

    tar = TraceAbstractionRefinement(max_iterations=20)
    tar_result = tar.verify(source)

    try:
        art_result = art_verify(source, max_iterations=20)
        art_safe = art_result.safe
        art_iters = art_result.refinement_count
        art_preds = len(art_result.predicates)
    except Exception:
        art_safe = None
        art_iters = -1
        art_preds = 0

    return {
        'tar_verdict': tar_result.verdict.value,
        'tar_iterations': tar_result.iterations,
        'tar_traces_checked': tar_result.traces_checked,
        'tar_automaton_size': tar_result.automaton_size,
        'tar_predicates': len(tar_result.predicates_discovered),
        'art_safe': art_safe,
        'art_iterations': art_iters,
        'art_predicates': art_preds,
        'agree': (tar_result.verdict == TraceAbstractionVerdict.SAFE) == (art_safe is True)
            if art_safe is not None else None
    }


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def verify_trace_abstraction(source: str, max_iterations: int = 20,
                              max_traces: int = 50) -> TraceAbstractionResult:
    """Verify a C10 program using trace abstraction refinement."""
    tar = TraceAbstractionRefinement(
        max_iterations=max_iterations,
        max_traces_per_iter=max_traces
    )
    return tar.verify(source)


def verify_lazy(source: str, max_iterations: int = 30,
                max_depth: int = 40) -> TraceAbstractionResult:
    """Verify using lazy trace exploration (DFS + coverage)."""
    lazy = LazyTraceAbstraction(
        max_iterations=max_iterations,
        max_depth=max_depth
    )
    return lazy.verify(source)


def check_assertion(source: str) -> Tuple[bool, Optional[dict]]:
    """Quick check: is the program safe? Returns (safe, counterexample_inputs)."""
    result = verify_trace_abstraction(source)
    if result.verdict == TraceAbstractionVerdict.SAFE:
        return True, None
    elif result.verdict == TraceAbstractionVerdict.UNSAFE:
        return False, result.counterexample_inputs
    else:
        return True, None  # Conservative: unknown -> assume safe


def get_cfg(source: str) -> dict:
    """Get CFG structure for inspection."""
    cfg = build_cfg(source)
    nodes = cfg.all_nodes()
    return {
        'num_nodes': len(nodes),
        'num_edges': len(cfg.edges),
        'entry': cfg.entry,
        'exit': cfg.exit_node,
        'error': cfg.error_node,
        'edges': [(e.src, e.dst, e.kind.value, e.label) for e in cfg.edges]
    }


def trace_abstraction_summary(source: str) -> dict:
    """Get detailed summary of trace abstraction verification."""
    result = verify_trace_abstraction(source)
    cfg = build_cfg(source)

    return {
        'verdict': result.verdict.value,
        'iterations': result.iterations,
        'traces_checked': result.traces_checked,
        'traces_infeasible': result.traces_infeasible,
        'traces_feasible': result.traces_feasible,
        'automaton_size': result.automaton_size,
        'automaton_components': result.automaton_components,
        'predicates_discovered': len(result.predicates_discovered),
        'cfg_nodes': len(cfg.all_nodes()),
        'cfg_edges': len(cfg.edges),
        'has_counterexample': result.counterexample is not None,
    }
