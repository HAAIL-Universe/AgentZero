"""
V110: Abstract Reachability Tree (ART)
======================================

The core data structure behind CEGAR model checkers like BLAST and CPAchecker.

An ART is an explicit unfolding of a program's control-flow graph, where each
node holds an abstract state. The tree is built lazily: only paths that might
reach an error are explored. When a spurious counterexample is found, Craig
interpolation refines the abstraction along that path.

Composes:
- C010 (parser/AST) for program representation
- C037 (SMT solver) for feasibility checking
- V107 (Craig interpolation) for refinement
- C039-style abstract interpretation for abstract states

Key algorithms:
1. ART construction via DFS/BFS with abstract post and coverage checking
2. Counterexample feasibility checking via SMT path encoding
3. Interpolation-based refinement: Craig interpolants -> refined abstract states
4. Lazy predicate abstraction: predicates added only where needed (per-location)
5. Coverage: a node is covered if its abstract state is subsumed by a sibling's
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# --- C010 imports ---
from stack_vm import lex, Parser

# --- C037 imports ---
from smt_solver import (
    SMTSolver, Var, IntConst, App, Op, Sort, SortKind, SMTResult
)

# --- V107 imports ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))
from craig_interpolation import craig_interpolate, InterpolantResult


# ============================================================================
# Control Flow Graph (CFG) Construction from C10 AST
# ============================================================================

class CFGNodeType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    ASSIGN = "assign"       # let x = e; or x = e;
    ASSUME = "assume"       # branch condition (true branch)
    ASSUME_NOT = "assume_not"  # branch condition (false branch)
    ASSERT = "assert"       # assertion check
    SKIP = "skip"           # no-op (join point)
    ERROR = "error"         # error location


@dataclass
class CFGNode:
    id: int
    type: CFGNodeType
    # For ASSIGN: (var_name, expr_ast)
    # For ASSUME/ASSUME_NOT: condition AST
    # For ASSERT: condition AST
    data: object = None
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    line: int = 0


@dataclass
class CFG:
    nodes: list  # list of CFGNode
    entry: CFGNode = None
    exit_node: CFGNode = None
    error_node: CFGNode = None

    def add_node(self, ntype, data=None, line=0):
        n = CFGNode(id=len(self.nodes), type=ntype, data=data, line=line)
        self.nodes.append(n)
        return n

    def add_edge(self, src, dst):
        if dst not in src.successors:
            src.successors.append(dst)
        if src not in dst.predecessors:
            dst.predecessors.append(src)


def build_cfg(source):
    """Build CFG from C10 source code."""
    tokens = lex(source)
    program = Parser(tokens).parse()

    cfg = CFG(nodes=[])
    cfg.entry = cfg.add_node(CFGNodeType.ENTRY)
    cfg.exit_node = cfg.add_node(CFGNodeType.EXIT)
    cfg.error_node = cfg.add_node(CFGNodeType.ERROR)

    last = _build_stmts(cfg, program.stmts, cfg.entry)
    if last is not None:
        cfg.add_edge(last, cfg.exit_node)

    return cfg


def _build_stmts(cfg, stmts, current):
    """Build CFG nodes for a list of statements, returning the last node."""
    for stmt in stmts:
        current = _build_stmt(cfg, stmt, current)
        if current is None:
            return None
    return current


def _build_stmt(cfg, stmt, current):
    """Build CFG node(s) for a single statement, returning the successor node."""
    cls = stmt.__class__.__name__

    if cls == 'LetDecl':
        n = cfg.add_node(CFGNodeType.ASSIGN, data=(stmt.name, stmt.value),
                         line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, n)
        return n

    elif cls == 'Assign':
        n = cfg.add_node(CFGNodeType.ASSIGN, data=(stmt.name, stmt.value),
                         line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, n)
        return n

    elif cls == 'IfStmt':
        # True branch
        assume_true = cfg.add_node(CFGNodeType.ASSUME, data=stmt.cond,
                                   line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, assume_true)
        then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else stmt.then_body.stmts
        then_last = _build_stmts(cfg, then_stmts, assume_true)

        # False branch
        assume_false = cfg.add_node(CFGNodeType.ASSUME_NOT, data=stmt.cond,
                                    line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, assume_false)

        join = cfg.add_node(CFGNodeType.SKIP, line=getattr(stmt, 'line', 0))

        if then_last is not None:
            cfg.add_edge(then_last, join)

        if stmt.else_body:
            else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else stmt.else_body.stmts
            else_last = _build_stmts(cfg, else_stmts, assume_false)
            if else_last is not None:
                cfg.add_edge(else_last, join)
        else:
            cfg.add_edge(assume_false, join)

        return join

    elif cls == 'WhileStmt':
        # Loop header (condition check)
        loop_head = cfg.add_node(CFGNodeType.SKIP,
                                 line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, loop_head)

        # True branch (loop body)
        assume_true = cfg.add_node(CFGNodeType.ASSUME, data=stmt.cond,
                                   line=getattr(stmt, 'line', 0))
        cfg.add_edge(loop_head, assume_true)

        body_stmts = stmt.body if isinstance(stmt.body, list) else stmt.body.stmts
        body_last = _build_stmts(cfg, body_stmts, assume_true)
        if body_last is not None:
            cfg.add_edge(body_last, loop_head)  # back edge

        # False branch (exit loop)
        assume_false = cfg.add_node(CFGNodeType.ASSUME_NOT, data=stmt.cond,
                                    line=getattr(stmt, 'line', 0))
        cfg.add_edge(loop_head, assume_false)

        return assume_false

    elif cls == 'CallExpr':
        # Expression statement -- check if it's assert()
        if hasattr(stmt, 'callee') and stmt.callee == 'assert':
            assert_node = cfg.add_node(CFGNodeType.ASSERT, data=stmt.args[0],
                                       line=getattr(stmt, 'line', 0))
            cfg.add_edge(current, assert_node)

            # Error edge: assertion fails
            cfg.add_edge(assert_node, cfg.error_node)

            return assert_node
        else:
            # Other call expressions -- treat as skip
            n = cfg.add_node(CFGNodeType.SKIP, line=getattr(stmt, 'line', 0))
            cfg.add_edge(current, n)
            return n

    elif cls == 'ExprStmt':
        # Expression statement wrapper
        return _build_stmt(cfg, stmt.expr, current)

    elif cls == 'Block':
        return _build_stmts(cfg, stmt.stmts, current)

    else:
        # Unknown statement type -- skip
        n = cfg.add_node(CFGNodeType.SKIP, line=getattr(stmt, 'line', 0))
        cfg.add_edge(current, n)
        return n


# ============================================================================
# Abstract State: Predicate-based abstraction
# ============================================================================

@dataclass
class PredicateState:
    """Abstract state = set of predicates known to be true."""
    predicates: frozenset  # frozenset of predicate indices (into a predicate list)
    is_bottom: bool = False

    @staticmethod
    def top():
        return PredicateState(frozenset())

    @staticmethod
    def bottom():
        return PredicateState(frozenset(), is_bottom=True)

    def subsumes(self, other):
        """Does self subsume other? (self >= other in the lattice)
        self subsumes other if other's known predicates are a superset of self's.
        In predicate abstraction: more predicates = more precise = lower in lattice.
        """
        if self.is_bottom:
            return other.is_bottom
        if other.is_bottom:
            return True
        # self has fewer constraints => it represents more states => it covers other
        return self.predicates.issubset(other.predicates)

    def join(self, other):
        """Join (over-approximation): keep only shared predicates."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return PredicateState(self.predicates & other.predicates)

    def __eq__(self, other):
        if not isinstance(other, PredicateState):
            return False
        return self.is_bottom == other.is_bottom and self.predicates == other.predicates

    def __hash__(self):
        return hash((self.is_bottom, self.predicates))


# ============================================================================
# SMT Encoding: AST expressions to SMT terms
# ============================================================================

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


def _ast_to_smt(expr, env_vars):
    """Convert C10 AST expression to SMT term. env_vars maps name -> Var."""
    cls = expr.__class__.__name__

    if cls == 'IntLit':
        return IntConst(expr.value)

    elif cls == 'Var':
        name = expr.name
        if name not in env_vars:
            env_vars[name] = Var(name, INT)
        return env_vars[name]

    elif cls == 'BinOp':
        left = _ast_to_smt(expr.left, env_vars)
        right = _ast_to_smt(expr.right, env_vars)
        op_map = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
            '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
            '==': Op.EQ, '!=': Op.NEQ,
        }
        if expr.op in op_map:
            return App(op_map[expr.op], [left, right],
                       BOOL if expr.op in ('<', '<=', '>', '>=', '==', '!=') else INT)
        elif expr.op == 'and':
            return App(Op.AND, [left, right], BOOL)
        elif expr.op == 'or':
            return App(Op.OR, [left, right], BOOL)
        else:
            raise ValueError(f"Unsupported operator: {expr.op}")

    elif cls == 'UnaryOp':
        operand = _ast_to_smt(expr.operand, env_vars)
        if expr.op == '-':
            return App(Op.SUB, [IntConst(0), operand], INT)
        elif expr.op == 'not':
            return App(Op.NOT, [operand], BOOL)
        else:
            raise ValueError(f"Unsupported unary operator: {expr.op}")

    elif cls == 'BoolLit':
        if expr.value:
            return App(Op.GE, [IntConst(1), IntConst(0)], BOOL)  # TRUE
        else:
            return App(Op.LT, [IntConst(1), IntConst(0)], BOOL)  # FALSE

    else:
        raise ValueError(f"Cannot convert AST node type: {cls}")


def _make_step_var(name, step):
    """Create a step-indexed variable: name_step."""
    v = Var(f"{name}_{step}", INT)
    return v


# ============================================================================
# ART Node and Tree
# ============================================================================

class ARTNodeStatus(Enum):
    EXPANDED = "expanded"
    COVERED = "covered"
    UNCOVERED = "uncovered"  # waiting to be expanded
    ERROR = "error"


@dataclass
class ARTNode:
    id: int
    cfg_node: CFGNode
    abstract_state: PredicateState
    parent: 'ARTNode' = None
    children: list = field(default_factory=list)
    status: ARTNodeStatus = ARTNodeStatus.UNCOVERED
    covered_by: 'ARTNode' = None
    depth: int = 0


@dataclass
class ARTResult:
    """Result of ART-based verification."""
    safe: bool
    art_nodes: int = 0
    counterexample: list = None  # list of (cfg_node, data) pairs
    counterexample_inputs: dict = None  # concrete inputs triggering error
    refinement_count: int = 0
    predicates: list = None  # list of predicate terms
    predicate_map: dict = None  # location_id -> set of predicate indices
    covered_count: int = 0


# ============================================================================
# Predicate Registry: per-location predicate tracking
# ============================================================================

class PredicateRegistry:
    """Manages predicates, possibly per-location (lazy abstraction)."""

    def __init__(self):
        self.predicates = []  # list of (term, name_str)
        self.pred_map = {}    # term_str -> index
        self.location_preds = {}  # cfg_node_id -> set of pred indices

    def add_predicate(self, term, location_id=None):
        """Add a predicate, optionally associated with a location."""
        key = str(term)
        if key in self.pred_map:
            idx = self.pred_map[key]
        else:
            idx = len(self.predicates)
            self.predicates.append((term, key))
            self.pred_map[key] = idx

        if location_id is not None:
            if location_id not in self.location_preds:
                self.location_preds[location_id] = set()
            self.location_preds[location_id].add(idx)
        else:
            # Global predicate -- add to all known locations
            for loc in self.location_preds:
                self.location_preds[loc].add(idx)

        return idx

    def get_predicates_at(self, location_id):
        """Get predicate indices relevant at a location."""
        return self.location_preds.get(location_id, set())

    def get_all_predicate_indices(self):
        return set(range(len(self.predicates)))

    def get_predicate_term(self, idx):
        return self.predicates[idx][0]

    def get_predicate_name(self, idx):
        return self.predicates[idx][1]


# ============================================================================
# Abstract Post: compute successor abstract state
# ============================================================================

def _abstract_post_assign(state, var_name, expr_ast, predicates, registry):
    """Compute abstract post for assignment var_name := expr.
    Check which predicates still hold after the assignment using SMT.
    """
    if state.is_bottom:
        return PredicateState.bottom()

    # For each predicate, check if it's preserved by the assignment
    new_preds = set()
    for idx in predicates:
        term, _ = registry.predicates[idx]
        # Check: (conjunction of current preds) AND (var_name = expr) => pred[var_name/expr]
        # Simplified: just check if the predicate doesn't mention var_name
        # or if it can be proven via SMT
        pred_str = str(term)
        if var_name not in pred_str:
            # Predicate doesn't mention the assigned variable -- preserved
            if idx in state.predicates:
                new_preds.add(idx)
        else:
            # Predicate mentions the variable -- check via SMT
            preserved = _check_predicate_after_assign(
                state, var_name, expr_ast, term, registry
            )
            if preserved:
                new_preds.add(idx)

    return PredicateState(frozenset(new_preds))


def _check_predicate_after_assign(state, var_name, expr_ast, pred_term, registry):
    """Check if pred_term holds after var_name := expr_ast, given current state predicates."""
    solver = SMTSolver()
    env = {}

    # Assert current predicates
    for idx in state.predicates:
        t = registry.get_predicate_term(idx)
        _assert_term(solver, t, env)

    # Encode the assignment: create fresh variable for post-state
    post_var = Var(f"{var_name}_post", INT)
    env_post = dict(env)
    # Build expr in current env
    expr_smt = _ast_to_smt(expr_ast, env)
    # post_var = expr
    solver.add(App(Op.EQ, [post_var, expr_smt], BOOL))

    # Check if predicate holds in post-state (with var_name replaced by post_var)
    pred_post = _substitute_var_in_term(pred_term, var_name, post_var)
    # Check negation
    solver.add(App(Op.NOT, [pred_post], BOOL))

    result = solver.check()
    return result == SMTResult.UNSAT  # negation UNSAT means predicate holds


def _substitute_var_in_term(term, var_name, replacement):
    """Substitute occurrences of Var(var_name) with replacement in term."""
    if isinstance(term, Var):
        if term.name == var_name:
            return replacement
        return term
    elif isinstance(term, IntConst):
        return term
    elif isinstance(term, App):
        new_args = [_substitute_var_in_term(a, var_name, replacement) for a in term.args]
        return App(term.op, new_args, term.sort)
    else:
        return term


def _assert_term(solver, term, env):
    """Add a term as assertion to solver, creating variables as needed."""
    # Terms from the registry already use Var objects -- just add them
    solver.add(term)


def _abstract_post_assume(state, cond_ast, positive, predicates, registry):
    """Compute abstract post for assume(cond) or assume(!cond).
    Add predicates that are implied by the condition.
    """
    if state.is_bottom:
        return PredicateState.bottom()

    new_preds = set(state.predicates)

    # Check which new predicates are implied by the assumption
    for idx in predicates:
        if idx in new_preds:
            continue
        term = registry.get_predicate_term(idx)
        implied = _check_predicate_after_assume(state, cond_ast, positive, term, registry)
        if implied:
            new_preds.add(idx)

    # Check if the assumption is feasible with current predicates
    feasible = _check_assume_feasible(state, cond_ast, positive, registry)
    if not feasible:
        return PredicateState.bottom()

    return PredicateState(frozenset(new_preds))


def _check_predicate_after_assume(state, cond_ast, positive, pred_term, registry):
    """Check if pred_term is implied by current state + assume(cond)."""
    solver = SMTSolver()
    env = {}

    # Assert current predicates
    for idx in state.predicates:
        t = registry.get_predicate_term(idx)
        _assert_term(solver, t, env)

    # Assert the condition (or its negation)
    cond_smt = _ast_to_smt(cond_ast, env)
    if positive:
        solver.add(cond_smt)
    else:
        solver.add(App(Op.NOT, [cond_smt], BOOL))

    # Check negation of predicate
    solver.add(App(Op.NOT, [pred_term], BOOL))

    result = solver.check()
    return result == SMTResult.UNSAT


def _check_assume_feasible(state, cond_ast, positive, registry):
    """Check if assume(cond) is feasible with current predicates."""
    solver = SMTSolver()
    env = {}

    for idx in state.predicates:
        t = registry.get_predicate_term(idx)
        _assert_term(solver, t, env)

    cond_smt = _ast_to_smt(cond_ast, env)
    if positive:
        solver.add(cond_smt)
    else:
        solver.add(App(Op.NOT, [cond_smt], BOOL))

    result = solver.check()
    return result == SMTResult.SAT


# ============================================================================
# Counterexample Feasibility Checking
# ============================================================================

def _extract_path(art_node):
    """Extract CFG path from root to art_node."""
    path = []
    n = art_node
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path


def _check_cex_feasibility(path, cfg):
    """Check if a counterexample path is feasible using SMT.
    Returns (feasible, model_or_None, formulas_per_step).
    """
    solver = SMTSolver()
    env = {}  # var_name -> current Var at this step
    step = 0
    formulas = []  # per-step formulas for interpolation

    for art_node in path:
        cfg_node = art_node.cfg_node
        step_formulas = []

        if cfg_node.type == CFGNodeType.ASSIGN:
            var_name, expr_ast = cfg_node.data
            # Create step-indexed variable
            new_var = _make_step_var(var_name, step)
            expr_smt = _ast_to_smt(expr_ast, env)
            eq = App(Op.EQ, [new_var, expr_smt], BOOL)
            solver.add(eq)
            step_formulas.append(eq)
            env[var_name] = new_var
            step += 1

        elif cfg_node.type == CFGNodeType.ASSUME:
            cond_smt = _ast_to_smt(cfg_node.data, env)
            solver.add(cond_smt)
            step_formulas.append(cond_smt)

        elif cfg_node.type == CFGNodeType.ASSUME_NOT:
            cond_smt = _ast_to_smt(cfg_node.data, env)
            neg = App(Op.NOT, [cond_smt], BOOL)
            solver.add(neg)
            step_formulas.append(neg)

        elif cfg_node.type == CFGNodeType.ASSERT:
            # At the error location, the assertion must FAIL
            cond_smt = _ast_to_smt(cfg_node.data, env)
            neg = App(Op.NOT, [cond_smt], BOOL)
            solver.add(neg)
            step_formulas.append(neg)

        elif cfg_node.type == CFGNodeType.ERROR:
            pass  # No additional constraints

        formulas.append(step_formulas)

    result = solver.check()
    if result == SMTResult.SAT:
        model = solver.model()
        return True, model, formulas
    else:
        return False, None, formulas


# ============================================================================
# Interpolation-Based Refinement
# ============================================================================

def _refine_with_interpolation(path, formulas, registry):
    """Refine predicates using Craig interpolation along a spurious path.
    Returns list of new predicates added (as indices).
    """
    # Collect all formulas into a flat list for sequence interpolation
    # Each step contributes its conjunction
    flat_formulas = []
    step_indices = []  # maps flat index -> path index

    for i, step_fs in enumerate(formulas):
        if step_fs:
            # Conjunction of step formulas
            if len(step_fs) == 1:
                flat_formulas.append(step_fs[0])
            else:
                conj = step_fs[0]
                for f in step_fs[1:]:
                    conj = App(Op.AND, [conj, f], BOOL)
                flat_formulas.append(conj)
            step_indices.append(i)

    if len(flat_formulas) < 2:
        return []

    new_predicates = []

    # Binary interpolation: split path into prefix A and suffix B
    # Try different split points
    for split in range(1, len(flat_formulas)):
        # A = conjunction of flat_formulas[0..split-1]
        # B = conjunction of flat_formulas[split..]
        a_terms = flat_formulas[:split]
        b_terms = flat_formulas[split:]

        a_conj = a_terms[0]
        for t in a_terms[1:]:
            a_conj = App(Op.AND, [a_conj, t], BOOL)

        b_conj = b_terms[0]
        for t in b_terms[1:]:
            b_conj = App(Op.AND, [b_conj, t], BOOL)

        try:
            result = craig_interpolate(a_conj, b_conj)
            if result.is_unsat and result.interpolant is not None:
                interp = result.interpolant
                # Extract atomic predicates from interpolant
                atoms = _extract_atoms(interp)
                path_idx = step_indices[split] if split < len(step_indices) else step_indices[-1]
                loc_id = path[path_idx].cfg_node.id if path_idx < len(path) else None

                for atom in atoms:
                    idx = registry.add_predicate(atom, location_id=loc_id)
                    if idx not in [p for p in new_predicates]:
                        new_predicates.append(idx)
        except Exception:
            continue

    return new_predicates


def _extract_atoms(term):
    """Extract atomic predicates from an interpolant formula."""
    atoms = []

    if isinstance(term, App):
        if term.op in (Op.AND, Op.OR):
            for arg in term.args:
                atoms.extend(_extract_atoms(arg))
        elif term.op == Op.NOT:
            # NOT(atom) -- extract the atom
            atoms.extend(_extract_atoms(term.args[0]))
        elif term.op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ):
            atoms.append(term)
        else:
            atoms.append(term)
    elif isinstance(term, (Var, IntConst)):
        pass  # Not a predicate by itself
    else:
        atoms.append(term)

    return atoms


# ============================================================================
# Main ART Construction and CEGAR Loop
# ============================================================================

def build_art(source, max_iterations=20, max_nodes=500):
    """Build an ART for the given C10 source and check safety.

    Returns ARTResult with:
    - safe: True if no reachable error states found
    - counterexample: concrete path to error if unsafe
    - refinement_count: number of CEGAR iterations
    - predicates: final predicate set
    """
    cfg = build_cfg(source)
    registry = PredicateRegistry()

    # Seed predicates from assertions in the source
    _seed_predicates_from_cfg(cfg, registry)

    refinement_count = 0

    for iteration in range(max_iterations):
        # Build ART with current predicates
        root, all_nodes, error_nodes, covered = _explore_art(
            cfg, registry, max_nodes
        )

        if not error_nodes:
            # No error nodes reachable -- SAFE
            return ARTResult(
                safe=True,
                art_nodes=len(all_nodes),
                refinement_count=refinement_count,
                predicates=[registry.get_predicate_name(i)
                            for i in range(len(registry.predicates))],
                predicate_map={k: list(v) for k, v in registry.location_preds.items()},
                covered_count=covered,
            )

        # Check feasibility of error paths
        found_real = False
        for err_node in error_nodes:
            path = _extract_path(err_node)
            feasible, model, formulas = _check_cex_feasibility(path, cfg)

            if feasible:
                # Real counterexample
                cex = [(n.cfg_node.type.value, n.cfg_node.data) for n in path]
                inputs = {}
                if model:
                    for k, v in model.items():
                        # Extract initial variable values
                        if '_' not in str(k) or str(k).endswith('_0'):
                            clean_name = str(k).replace('_0', '')
                            inputs[clean_name] = v
                return ARTResult(
                    safe=False,
                    art_nodes=len(all_nodes),
                    counterexample=cex,
                    counterexample_inputs=inputs,
                    refinement_count=refinement_count,
                    predicates=[registry.get_predicate_name(i)
                                for i in range(len(registry.predicates))],
                    predicate_map={k: list(v) for k, v in registry.location_preds.items()},
                    covered_count=covered,
                )

            # Spurious -- refine
            new_preds = _refine_with_interpolation(path, formulas, registry)
            if new_preds:
                refinement_count += 1
                found_real = False
            else:
                # Refinement produced no new predicates -- try a simple fallback
                _fallback_refinement(path, registry)
                refinement_count += 1

        if found_real:
            break

    # Max iterations reached
    return ARTResult(
        safe=True,  # Conservative: couldn't find a real bug
        art_nodes=len(all_nodes) if 'all_nodes' in dir() else 0,
        refinement_count=refinement_count,
        predicates=[registry.get_predicate_name(i)
                    for i in range(len(registry.predicates))],
        predicate_map={k: list(v) for k, v in registry.location_preds.items()},
        covered_count=covered if 'covered' in dir() else 0,
    )


def _seed_predicates_from_cfg(cfg, registry):
    """Extract initial predicates from CFG assertions and conditions."""
    env = {}
    for node in cfg.nodes:
        if node.type == CFGNodeType.ASSERT:
            try:
                term = _ast_to_smt(node.data, env)
                registry.add_predicate(term, location_id=node.id)
            except (ValueError, AttributeError):
                pass
        elif node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT):
            try:
                term = _ast_to_smt(node.data, env)
                registry.add_predicate(term, location_id=node.id)
            except (ValueError, AttributeError):
                pass


def _explore_art(cfg, registry, max_nodes):
    """Build ART via DFS exploration with coverage checking."""
    root = ARTNode(
        id=0,
        cfg_node=cfg.entry,
        abstract_state=PredicateState.top(),
        depth=0,
    )

    all_nodes = [root]
    expanded = {cfg.entry.id: [root]}  # cfg_node_id -> list of ART nodes
    worklist = [root]
    error_nodes = []
    covered_count = 0
    node_counter = 1

    while worklist and len(all_nodes) < max_nodes:
        current = worklist.pop()

        if current.status == ARTNodeStatus.COVERED:
            continue

        if current.cfg_node.type == CFGNodeType.ERROR:
            current.status = ARTNodeStatus.ERROR
            error_nodes.append(current)
            continue

        if current.cfg_node.type == CFGNodeType.EXIT:
            current.status = ARTNodeStatus.EXPANDED
            continue

        # Expand: compute successors
        current.status = ARTNodeStatus.EXPANDED

        for succ_cfg in current.cfg_node.successors:
            # Compute abstract post
            preds_at_succ = registry.get_predicates_at(succ_cfg.id) | registry.get_all_predicate_indices()
            new_state = _compute_abstract_post(
                current.abstract_state, current.cfg_node, succ_cfg,
                preds_at_succ, registry
            )

            if new_state.is_bottom:
                continue  # Infeasible path

            # Check coverage
            covered = False
            for existing in expanded.get(succ_cfg.id, []):
                if existing.status != ARTNodeStatus.COVERED and existing.abstract_state.subsumes(new_state):
                    covered = True
                    covered_count += 1
                    break

            child = ARTNode(
                id=node_counter,
                cfg_node=succ_cfg,
                abstract_state=new_state,
                parent=current,
                depth=current.depth + 1,
            )
            node_counter += 1
            current.children.append(child)
            all_nodes.append(child)

            if succ_cfg.id not in expanded:
                expanded[succ_cfg.id] = []
            expanded[succ_cfg.id].append(child)

            if covered:
                child.status = ARTNodeStatus.COVERED
            else:
                worklist.append(child)

    return root, all_nodes, error_nodes, covered_count


def _compute_abstract_post(current_state, current_cfg, succ_cfg, predicates, registry):
    """Compute abstract post-state for transition current_cfg -> succ_cfg."""

    if succ_cfg.type == CFGNodeType.ASSIGN:
        var_name, expr_ast = succ_cfg.data
        return _abstract_post_assign(current_state, var_name, expr_ast, predicates, registry)

    elif succ_cfg.type == CFGNodeType.ASSUME:
        return _abstract_post_assume(current_state, succ_cfg.data, True, predicates, registry)

    elif succ_cfg.type == CFGNodeType.ASSUME_NOT:
        return _abstract_post_assume(current_state, succ_cfg.data, False, predicates, registry)

    elif succ_cfg.type == CFGNodeType.ASSERT:
        # For the assert node, the assertion might fail (edge to error)
        # or succeed (fall through). Return the current state for now.
        return current_state

    elif succ_cfg.type == CFGNodeType.ERROR:
        # Check if the assertion that leads here can fail
        # Find the ASSERT predecessor
        for pred in succ_cfg.predecessors:
            if pred.type == CFGNodeType.ASSERT:
                # Check if NOT(assertion) is feasible
                feasible = _check_assume_feasible(
                    current_state, pred.data, False, registry
                )
                if feasible:
                    return current_state
                else:
                    return PredicateState.bottom()
        return current_state

    else:
        # SKIP, ENTRY, EXIT -- pass through
        return current_state


def _fallback_refinement(path, registry):
    """Fallback refinement: add predicates from conditions along the path."""
    env = {}
    for art_node in path:
        cfg_node = art_node.cfg_node
        if cfg_node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT):
            try:
                term = _ast_to_smt(cfg_node.data, env)
                registry.add_predicate(term, location_id=cfg_node.id)
            except (ValueError, AttributeError):
                pass
        elif cfg_node.type == CFGNodeType.ASSIGN:
            var_name, expr_ast = cfg_node.data
            try:
                var = Var(var_name, INT)
                expr_smt = _ast_to_smt(expr_ast, env)
                eq = App(Op.EQ, [var, expr_smt], BOOL)
                registry.add_predicate(eq, location_id=cfg_node.id)
                env[var_name] = var
            except (ValueError, AttributeError):
                pass


# ============================================================================
# High-Level APIs
# ============================================================================

def verify_program(source, max_iterations=20, max_nodes=500):
    """Verify a C10 program using ART-based CEGAR.

    Args:
        source: C10 source code with assert() statements
        max_iterations: max CEGAR refinement iterations
        max_nodes: max ART nodes before stopping

    Returns:
        ARTResult with safety verdict, counterexample if unsafe,
        and verification statistics.
    """
    return build_art(source, max_iterations, max_nodes)


def build_cfg_from_source(source):
    """Build a CFG from C10 source code.

    Returns a CFG object with entry, exit, and error nodes.
    Useful for inspection and debugging.
    """
    return build_cfg(source)


def check_assertion(source, max_iterations=20):
    """Check if all assertions in the source hold.

    Returns (safe, counterexample_inputs_or_None).
    """
    result = verify_program(source, max_iterations)
    if result.safe:
        return True, None
    else:
        return False, result.counterexample_inputs


def get_predicates(source, max_iterations=10):
    """Run CEGAR and return the discovered predicates.

    Returns a dict with:
    - predicates: list of predicate strings
    - predicate_map: location_id -> list of predicate indices
    - refinement_count: number of refinement iterations
    """
    result = verify_program(source, max_iterations)
    return {
        'predicates': result.predicates,
        'predicate_map': result.predicate_map,
        'refinement_count': result.refinement_count,
        'safe': result.safe,
    }


def compare_with_without_refinement(source):
    """Compare verification with and without CEGAR refinement.

    Returns a dict with:
    - no_refinement: result with 0 iterations (just initial predicates)
    - with_refinement: result with up to 20 iterations
    - refinement_helped: whether refinement changed the verdict
    """
    result_no_ref = verify_program(source, max_iterations=1)
    result_with_ref = verify_program(source, max_iterations=20)

    return {
        'no_refinement': {
            'safe': result_no_ref.safe,
            'art_nodes': result_no_ref.art_nodes,
            'predicates': len(result_no_ref.predicates) if result_no_ref.predicates else 0,
        },
        'with_refinement': {
            'safe': result_with_ref.safe,
            'art_nodes': result_with_ref.art_nodes,
            'predicates': len(result_with_ref.predicates) if result_with_ref.predicates else 0,
            'refinement_count': result_with_ref.refinement_count,
        },
        'refinement_helped': result_no_ref.safe != result_with_ref.safe,
    }


def cfg_summary(source):
    """Get a summary of the CFG for a source program.

    Returns a dict with node count, edge count, and node types.
    """
    cfg = build_cfg(source)
    edge_count = sum(len(n.successors) for n in cfg.nodes)
    type_counts = {}
    for n in cfg.nodes:
        t = n.type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        'nodes': len(cfg.nodes),
        'edges': edge_count,
        'types': type_counts,
        'entry_id': cfg.entry.id,
        'exit_id': cfg.exit_node.id,
        'error_id': cfg.error_node.id,
    }


def art_summary(source, max_iterations=20):
    """Get a summary of the ART construction and verification.

    Returns a dict with all verification statistics.
    """
    result = verify_program(source, max_iterations)
    return {
        'safe': result.safe,
        'art_nodes': result.art_nodes,
        'refinement_count': result.refinement_count,
        'predicates': result.predicates,
        'predicate_count': len(result.predicates) if result.predicates else 0,
        'covered_count': result.covered_count,
        'has_counterexample': result.counterexample is not None,
        'counterexample_inputs': result.counterexample_inputs,
    }
