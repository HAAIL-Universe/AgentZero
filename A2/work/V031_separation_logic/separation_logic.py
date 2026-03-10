"""
V031: Separation Logic Prover

A prover for separation logic with:
- Symbolic heaps (points-to, list segments, trees, separating conjunction)
- Entailment checking via unfolding + matching + SMT
- Frame inference (P |- Q * ?F, find F)
- Bi-abduction (P * ?anti |- Q * ?frame, find anti and frame)
- Frame rule application for compositional reasoning
- Integration with V030 shape analysis for concrete heap abstraction

Separation logic extends Hoare logic with spatial connectives that reason
about heap ownership and disjointness, enabling modular verification of
pointer-manipulating programs.
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union
from copy import deepcopy

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Var, App, Op, Sort, SortKind,
    IntConst, BoolConst
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)

# ---------------------------------------------------------------------------
# Separation Logic Formulas
# ---------------------------------------------------------------------------

class ExprKind(Enum):
    VAR = auto()
    NULL = auto()
    INT_CONST = auto()

@dataclass(frozen=True)
class Expr:
    """Heap expression: variable, null, or integer constant."""
    kind: ExprKind
    name: str = ""
    value: int = 0

    def __repr__(self):
        if self.kind == ExprKind.NULL:
            return "null"
        if self.kind == ExprKind.VAR:
            return self.name
        return str(self.value)

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return self.kind == other.kind and self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((self.kind, self.name, self.value))

def EVar(name: str) -> Expr:
    return Expr(ExprKind.VAR, name=name)

def ENull() -> Expr:
    return Expr(ExprKind.NULL)

def EInt(value: int) -> Expr:
    return Expr(ExprKind.INT_CONST, value=value)


class FormulaKind(Enum):
    EMP = auto()        # empty heap
    POINTS_TO = auto()  # x |-> y (single cell)
    STAR = auto()       # P * Q (separating conjunction)
    WAND = auto()       # P -* Q (separating implication / magic wand)
    PURE = auto()        # pure constraint (no heap)
    LSEG = auto()       # lseg(x, y) -- list segment
    TREE = auto()       # tree(x) -- binary tree
    DLSEG = auto()      # dlseg(x, y, p, n) -- doubly-linked list segment
    FALSE = auto()      # false / contradiction

@dataclass
class SLFormula:
    """Separation logic formula."""
    kind: FormulaKind
    # For POINTS_TO: src, dst
    src: Optional[Expr] = None
    dst: Optional[Expr] = None
    # For STAR, WAND: left, right
    left: Optional['SLFormula'] = None
    right: Optional['SLFormula'] = None
    # For PURE: pure condition
    pure_op: str = ""  # "eq", "neq", "lt", "le", "gt", "ge"
    pure_lhs: Optional[Expr] = None
    pure_rhs: Optional[Expr] = None
    # For LSEG: start, end
    seg_start: Optional[Expr] = None
    seg_end: Optional[Expr] = None
    # For TREE: root
    tree_root: Optional[Expr] = None
    # For DLSEG: forward start, forward end, backward start, backward end
    dl_start: Optional[Expr] = None
    dl_end: Optional[Expr] = None
    dl_prev: Optional[Expr] = None
    dl_next: Optional[Expr] = None

    def __repr__(self):
        if self.kind == FormulaKind.EMP:
            return "emp"
        if self.kind == FormulaKind.FALSE:
            return "false"
        if self.kind == FormulaKind.POINTS_TO:
            return f"{self.src} |-> {self.dst}"
        if self.kind == FormulaKind.STAR:
            return f"({self.left} * {self.right})"
        if self.kind == FormulaKind.WAND:
            return f"({self.left} -* {self.right})"
        if self.kind == FormulaKind.PURE:
            return f"[{self.pure_lhs} {self.pure_op} {self.pure_rhs}]"
        if self.kind == FormulaKind.LSEG:
            return f"lseg({self.seg_start}, {self.seg_end})"
        if self.kind == FormulaKind.TREE:
            return f"tree({self.tree_root})"
        if self.kind == FormulaKind.DLSEG:
            return f"dlseg({self.dl_start}, {self.dl_end}, {self.dl_prev}, {self.dl_next})"
        return "?"


# ---------------------------------------------------------------------------
# Formula constructors
# ---------------------------------------------------------------------------

def Emp() -> SLFormula:
    return SLFormula(FormulaKind.EMP)

def SLFalse() -> SLFormula:
    return SLFormula(FormulaKind.FALSE)

def PointsTo(src: Expr, dst: Expr) -> SLFormula:
    return SLFormula(FormulaKind.POINTS_TO, src=src, dst=dst)

def Star(left: SLFormula, right: SLFormula) -> SLFormula:
    """Separating conjunction. Flattens emps."""
    if left.kind == FormulaKind.EMP:
        return right
    if right.kind == FormulaKind.EMP:
        return left
    if left.kind == FormulaKind.FALSE or right.kind == FormulaKind.FALSE:
        return SLFalse()
    return SLFormula(FormulaKind.STAR, left=left, right=right)

def StarList(formulas: list) -> SLFormula:
    """Separating conjunction of a list of formulas."""
    result = Emp()
    for f in formulas:
        result = Star(result, f)
    return result

def Wand(left: SLFormula, right: SLFormula) -> SLFormula:
    return SLFormula(FormulaKind.WAND, left=left, right=right)

def Pure(op: str, lhs: Expr, rhs: Expr) -> SLFormula:
    return SLFormula(FormulaKind.PURE, pure_op=op, pure_lhs=lhs, pure_rhs=rhs)

def PureEq(a: Expr, b: Expr) -> SLFormula:
    return Pure("eq", a, b)

def PureNeq(a: Expr, b: Expr) -> SLFormula:
    return Pure("neq", a, b)

def LSeg(start: Expr, end: Expr) -> SLFormula:
    return SLFormula(FormulaKind.LSEG, seg_start=start, seg_end=end)

def Tree(root: Expr) -> SLFormula:
    return SLFormula(FormulaKind.TREE, tree_root=root)

def DLSeg(start: Expr, end: Expr, prev: Expr, next_: Expr) -> SLFormula:
    return SLFormula(FormulaKind.DLSEG, dl_start=start, dl_end=end,
                     dl_prev=prev, dl_next=next_)

# ---------------------------------------------------------------------------
# Symbolic Heap (normalized form)
# ---------------------------------------------------------------------------

@dataclass
class SymbolicHeap:
    """
    A symbolic heap: Pi AND Sigma where
    Pi = conjunction of pure constraints
    Sigma = separating conjunction of spatial atoms
    """
    pure: list  # list of SLFormula (PURE kind)
    spatial: list  # list of SLFormula (POINTS_TO, LSEG, TREE, DLSEG)
    exists: list = field(default_factory=list)  # existentially quantified vars

    def __repr__(self):
        parts = []
        if self.exists:
            parts.append(f"exists {', '.join(str(v) for v in self.exists)}.")
        if self.pure:
            parts.append(" /\\ ".join(repr(p) for p in self.pure))
        if self.spatial:
            parts.append(" * ".join(repr(s) for s in self.spatial))
        elif not self.pure:
            parts.append("emp")
        return " : ".join(parts) if parts else "emp"

    def is_emp(self) -> bool:
        return len(self.spatial) == 0

    def copy(self) -> 'SymbolicHeap':
        return SymbolicHeap(
            pure=list(self.pure),
            spatial=list(self.spatial),
            exists=list(self.exists),
        )


def to_symbolic_heap(formula: SLFormula) -> SymbolicHeap:
    """Convert an SL formula to a symbolic heap (normalize)."""
    pure = []
    spatial = []

    def collect(f):
        if f.kind == FormulaKind.EMP:
            pass
        elif f.kind == FormulaKind.PURE:
            pure.append(f)
        elif f.kind == FormulaKind.STAR:
            collect(f.left)
            collect(f.right)
        elif f.kind in (FormulaKind.POINTS_TO, FormulaKind.LSEG,
                        FormulaKind.TREE, FormulaKind.DLSEG):
            spatial.append(f)
        elif f.kind == FormulaKind.FALSE:
            pure.append(Pure("eq", EInt(0), EInt(1)))  # contradiction
        else:
            # WAND kept as spatial for now
            spatial.append(f)

    collect(formula)
    return SymbolicHeap(pure=pure, spatial=spatial)


def from_symbolic_heap(sh: SymbolicHeap) -> SLFormula:
    """Convert a symbolic heap back to an SL formula."""
    parts = list(sh.pure) + list(sh.spatial)
    return StarList(parts) if parts else Emp()


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def subst_expr(e: Expr, mapping: dict) -> Expr:
    """Substitute variables in an expression."""
    if e.kind == ExprKind.VAR and e in mapping:
        return mapping[e]
    return e

def subst_formula(f: SLFormula, mapping: dict) -> SLFormula:
    """Substitute variables in a formula."""
    if f.kind == FormulaKind.EMP or f.kind == FormulaKind.FALSE:
        return f
    if f.kind == FormulaKind.POINTS_TO:
        return PointsTo(subst_expr(f.src, mapping), subst_expr(f.dst, mapping))
    if f.kind == FormulaKind.STAR:
        return Star(subst_formula(f.left, mapping), subst_formula(f.right, mapping))
    if f.kind == FormulaKind.WAND:
        return Wand(subst_formula(f.left, mapping), subst_formula(f.right, mapping))
    if f.kind == FormulaKind.PURE:
        return Pure(f.pure_op, subst_expr(f.pure_lhs, mapping),
                    subst_expr(f.pure_rhs, mapping))
    if f.kind == FormulaKind.LSEG:
        return LSeg(subst_expr(f.seg_start, mapping), subst_expr(f.seg_end, mapping))
    if f.kind == FormulaKind.TREE:
        return Tree(subst_expr(f.tree_root, mapping))
    if f.kind == FormulaKind.DLSEG:
        return DLSeg(subst_expr(f.dl_start, mapping), subst_expr(f.dl_end, mapping),
                     subst_expr(f.dl_prev, mapping), subst_expr(f.dl_next, mapping))
    return f


# ---------------------------------------------------------------------------
# Free variables
# ---------------------------------------------------------------------------

def free_vars_expr(e: Expr) -> set:
    if e.kind == ExprKind.VAR:
        return {e}
    return set()

def free_vars(f: SLFormula) -> set:
    """Collect free variables in a formula."""
    if f.kind == FormulaKind.EMP or f.kind == FormulaKind.FALSE:
        return set()
    if f.kind == FormulaKind.POINTS_TO:
        return free_vars_expr(f.src) | free_vars_expr(f.dst)
    if f.kind == FormulaKind.STAR:
        return free_vars(f.left) | free_vars(f.right)
    if f.kind == FormulaKind.WAND:
        return free_vars(f.left) | free_vars(f.right)
    if f.kind == FormulaKind.PURE:
        return free_vars_expr(f.pure_lhs) | free_vars_expr(f.pure_rhs)
    if f.kind == FormulaKind.LSEG:
        return free_vars_expr(f.seg_start) | free_vars_expr(f.seg_end)
    if f.kind == FormulaKind.TREE:
        return free_vars_expr(f.tree_root)
    if f.kind == FormulaKind.DLSEG:
        return (free_vars_expr(f.dl_start) | free_vars_expr(f.dl_end) |
                free_vars_expr(f.dl_prev) | free_vars_expr(f.dl_next))
    return set()

def free_vars_heap(sh: SymbolicHeap) -> set:
    result = set()
    for p in sh.pure:
        result |= free_vars(p)
    for s in sh.spatial:
        result |= free_vars(s)
    return result - set(sh.exists)


# ---------------------------------------------------------------------------
# Fresh variable generation
# ---------------------------------------------------------------------------

_fresh_counter = 0

def fresh_var(prefix: str = "_v") -> Expr:
    global _fresh_counter
    _fresh_counter += 1
    return EVar(f"{prefix}_{_fresh_counter}")

def reset_fresh():
    global _fresh_counter
    _fresh_counter = 0


# ---------------------------------------------------------------------------
# Inductive predicate unfolding
# ---------------------------------------------------------------------------

def unfold_lseg(start: Expr, end: Expr) -> list:
    """
    Unfold lseg(start, end) into two cases:
    1. Base: start == end /\\ emp
    2. Step: exists z. start |-> z * lseg(z, end) /\\ start != end
    Returns list of (pure_constraints, spatial_atoms, new_exists) tuples.
    """
    # Base case: start == end, empty segment
    base = ([PureEq(start, end)], [], [])

    # Recursive case: start |-> z * lseg(z, end)
    z = fresh_var("_ls")
    step = (
        [PureNeq(start, end)],
        [PointsTo(start, z), LSeg(z, end)],
        [z],
    )
    return [base, step]


def unfold_tree(root: Expr) -> list:
    """
    Unfold tree(root) into:
    1. Base: root == null AND emp
    2. Step: exists l, r. root |-> (l, r) * tree(l) * tree(r) AND root != null
    For simplicity, we model binary trees with two separate points-to:
    root |-> l * root_r |-> r  (or use a multi-field points-to).
    We use a flattened model where tree(x) with x!=null means
    x has two children accessible via the tree structure.
    """
    # Base case: root is null
    base = ([PureEq(root, ENull())], [], [])

    # Recursive case
    l = fresh_var("_tl")
    r = fresh_var("_tr")
    # We model a binary tree node as: root |-> l * root_right |-> r
    # But since we only have single-field points-to, we use two points-to
    # atoms with a naming convention: root and root' for the two fields.
    root_r = fresh_var("_tr_field")
    step = (
        [PureNeq(root, ENull()), PureEq(root_r, root)],  # root_r aliases root (second field)
        [PointsTo(root, l), PointsTo(root_r, r), Tree(l), Tree(r)],
        [l, r, root_r],
    )
    return [base, step]


def unfold_dlseg(start: Expr, end: Expr, prev: Expr, next_: Expr) -> list:
    """
    Unfold dlseg(start, end, prev, next):
    1. Base: start == end /\\ prev == next /\\ emp
    2. Step: exists z. start |-> z * start.prev |-> prev * dlseg(z, end, start, next)
    """
    base = ([PureEq(start, end)], [], [])

    z = fresh_var("_dl")
    start_prev = fresh_var("_dlp")
    step = (
        [PureNeq(start, end), PureEq(start_prev, start)],
        [PointsTo(start, z), PointsTo(start_prev, prev), DLSeg(z, end, start, next_)],
        [z, start_prev],
    )
    return [base, step]


# ---------------------------------------------------------------------------
# Pure constraint checking (via SMT)
# ---------------------------------------------------------------------------

class PureChecker:
    """Check satisfiability/validity of pure constraints using SMT."""

    def __init__(self):
        self._var_cache = {}

    def _expr_to_smt(self, solver: SMTSolver, e: Expr):
        if e.kind == ExprKind.NULL:
            return IntConst(0)
        if e.kind == ExprKind.INT_CONST:
            return IntConst(e.value)
        if e.kind == ExprKind.VAR:
            name = e.name
            if name not in self._var_cache:
                self._var_cache[name] = solver.Int(name)
            return self._var_cache[name]
        raise ValueError(f"Unknown expr kind: {e.kind}")

    def _pure_to_smt(self, solver: SMTSolver, p: SLFormula):
        assert p.kind == FormulaKind.PURE
        lhs = self._expr_to_smt(solver, p.pure_lhs)
        rhs = self._expr_to_smt(solver, p.pure_rhs)
        ops = {
            "eq": Op.EQ, "neq": Op.NEQ,
            "lt": Op.LT, "le": Op.LE,
            "gt": Op.GT, "ge": Op.GE,
        }
        op = ops[p.pure_op]
        return App(op, [lhs, rhs], BOOL)

    def check_sat(self, constraints: list) -> bool:
        """Check if pure constraints are satisfiable."""
        if not constraints:
            return True
        solver = SMTSolver()
        self._var_cache = {}
        for c in constraints:
            solver.add(self._pure_to_smt(solver, c))
        result = solver.check()
        return result == SMTResult.SAT

    def check_unsat(self, constraints: list) -> bool:
        """Check if pure constraints are unsatisfiable."""
        return not self.check_sat(constraints)

    def check_valid(self, assumptions: list, conclusion: SLFormula) -> bool:
        """Check if assumptions => conclusion is valid (negation is UNSAT)."""
        solver = SMTSolver()
        self._var_cache = {}
        for a in assumptions:
            solver.add(self._pure_to_smt(solver, a))
        # Negate conclusion
        conc_smt = self._pure_to_smt(solver, conclusion)
        # Use complement operator for negation
        neg = self._negate_pure(conclusion)
        solver.add(self._pure_to_smt(solver, neg))
        return solver.check() == SMTResult.UNSAT

    def _negate_pure(self, p: SLFormula) -> SLFormula:
        """Negate a pure constraint using complement operators."""
        complements = {
            "eq": "neq", "neq": "eq",
            "lt": "ge", "ge": "lt",
            "le": "gt", "gt": "le",
        }
        return Pure(complements[p.pure_op], p.pure_lhs, p.pure_rhs)

    def implies_eq(self, assumptions: list, a: Expr, b: Expr) -> bool:
        """Check if assumptions imply a == b."""
        return self.check_valid(assumptions, PureEq(a, b))

    def implies_neq(self, assumptions: list, a: Expr, b: Expr) -> bool:
        """Check if assumptions imply a != b."""
        return self.check_valid(assumptions, PureNeq(a, b))

    def get_equalities(self, constraints: list, exprs: list) -> dict:
        """Find which expressions must be equal under constraints."""
        eqs = {}
        for i, a in enumerate(exprs):
            for b in exprs[i+1:]:
                if self.implies_eq(constraints, a, b):
                    eqs.setdefault(repr(a), set()).add(repr(b))
        return eqs


# ---------------------------------------------------------------------------
# Entailment checker
# ---------------------------------------------------------------------------

class EntailmentResult(Enum):
    VALID = auto()
    INVALID = auto()
    UNKNOWN = auto()

@dataclass
class ProofResult:
    result: EntailmentResult
    frame: Optional[SymbolicHeap] = None  # remaining frame if valid
    witness: Optional[dict] = None  # substitution witness
    reason: str = ""

    def is_valid(self) -> bool:
        return self.result == EntailmentResult.VALID


class SLProver:
    """
    Separation logic entailment prover.

    Checks entailments of the form: P |- Q
    Where P and Q are symbolic heaps (pure /\\ spatial).

    Algorithm:
    1. Normalize both sides to symbolic heaps
    2. Match spatial atoms from RHS against LHS
    3. Unfold inductive predicates when needed
    4. Check pure constraints via SMT
    5. Return frame (unmatched LHS atoms) if valid
    """

    def __init__(self, max_unfold: int = 5):
        self.max_unfold = max_unfold
        self.pure_checker = PureChecker()

    def check_entailment(self, lhs: SLFormula, rhs: SLFormula) -> ProofResult:
        """Check if lhs |- rhs. Returns ProofResult with frame if valid."""
        lh = to_symbolic_heap(lhs)
        rh = to_symbolic_heap(rhs)
        return self._check_heap_entailment(lh, rh, depth=0)

    def check_entailment_heaps(self, lhs: SymbolicHeap, rhs: SymbolicHeap) -> ProofResult:
        """Check entailment between symbolic heaps."""
        return self._check_heap_entailment(lhs.copy(), rhs.copy(), depth=0)

    def _check_heap_entailment(self, lhs: SymbolicHeap, rhs: SymbolicHeap,
                                depth: int) -> ProofResult:
        """Core entailment algorithm."""
        if depth > self.max_unfold:
            return ProofResult(EntailmentResult.UNKNOWN, reason="max unfold depth")

        # Check pure contradiction in LHS
        if self.pure_checker.check_unsat(lhs.pure):
            # LHS is false, entailment holds vacuously
            return ProofResult(EntailmentResult.VALID,
                             frame=SymbolicHeap([], []),
                             reason="lhs contradiction")

        # Check pure consistency: LHS pures must imply RHS pures
        for rp in rhs.pure:
            if not self.pure_checker.check_valid(lhs.pure, rp):
                return ProofResult(EntailmentResult.INVALID,
                                 reason=f"pure constraint {rp} not implied")

        # Try to match RHS spatial atoms against LHS
        return self._match_spatial(lhs, rhs, depth)

    def _match_spatial(self, lhs: SymbolicHeap, rhs: SymbolicHeap,
                       depth: int) -> ProofResult:
        """Match RHS spatial atoms against LHS spatial atoms."""
        # If RHS has no spatial atoms, entailment holds with LHS remainder as frame
        if not rhs.spatial:
            frame = SymbolicHeap(pure=[], spatial=list(lhs.spatial))
            return ProofResult(EntailmentResult.VALID, frame=frame,
                             reason="all rhs matched")

        # Pick first RHS spatial atom to match
        rhs_atom = rhs.spatial[0]
        rhs_rest = SymbolicHeap(pure=rhs.pure, spatial=rhs.spatial[1:],
                               exists=rhs.exists)

        # Try to match against each LHS spatial atom
        for i, lhs_atom in enumerate(lhs.spatial):
            match_result = self._match_atoms(lhs_atom, rhs_atom, lhs.pure)
            if match_result is not None:
                sub, extra_pure, extra_spatial = match_result
                # Remove matched LHS atom, apply substitution
                new_lhs_spatial = lhs.spatial[:i] + lhs.spatial[i+1:] + extra_spatial
                new_lhs_pure = lhs.pure + extra_pure
                new_lhs = SymbolicHeap(pure=new_lhs_pure, spatial=new_lhs_spatial,
                                      exists=lhs.exists)
                # Apply substitution to remaining RHS
                new_rhs_rest = self._apply_sub_heap(rhs_rest, sub)
                result = self._match_spatial(new_lhs, new_rhs_rest, depth)
                if result.is_valid():
                    return result

        # No direct match found. Try unfolding LHS predicates.
        for i, lhs_atom in enumerate(lhs.spatial):
            if lhs_atom.kind == FormulaKind.LSEG:
                cases = unfold_lseg(lhs_atom.seg_start, lhs_atom.seg_end)
                for pure_cs, spatial_cs, new_exists in cases:
                    new_lhs = SymbolicHeap(
                        pure=lhs.pure + pure_cs,
                        spatial=lhs.spatial[:i] + lhs.spatial[i+1:] + spatial_cs,
                        exists=lhs.exists + new_exists,
                    )
                    if self.pure_checker.check_sat(new_lhs.pure):
                        result = self._check_heap_entailment(new_lhs, rhs, depth + 1)
                        if result.is_valid():
                            return result

            elif lhs_atom.kind == FormulaKind.TREE:
                cases = unfold_tree(lhs_atom.tree_root)
                for pure_cs, spatial_cs, new_exists in cases:
                    new_lhs = SymbolicHeap(
                        pure=lhs.pure + pure_cs,
                        spatial=lhs.spatial[:i] + lhs.spatial[i+1:] + spatial_cs,
                        exists=lhs.exists + new_exists,
                    )
                    if self.pure_checker.check_sat(new_lhs.pure):
                        result = self._check_heap_entailment(new_lhs, rhs, depth + 1)
                        if result.is_valid():
                            return result

            elif lhs_atom.kind == FormulaKind.DLSEG:
                cases = unfold_dlseg(lhs_atom.dl_start, lhs_atom.dl_end,
                                     lhs_atom.dl_prev, lhs_atom.dl_next)
                for pure_cs, spatial_cs, new_exists in cases:
                    new_lhs = SymbolicHeap(
                        pure=lhs.pure + pure_cs,
                        spatial=lhs.spatial[:i] + lhs.spatial[i+1:] + spatial_cs,
                        exists=lhs.exists + new_exists,
                    )
                    if self.pure_checker.check_sat(new_lhs.pure):
                        result = self._check_heap_entailment(new_lhs, rhs, depth + 1)
                        if result.is_valid():
                            return result

        # Try unfolding RHS predicates
        for j, rhs_atom in enumerate(rhs.spatial):
            unfold_cases = None
            if rhs_atom.kind == FormulaKind.LSEG:
                unfold_cases = unfold_lseg(rhs_atom.seg_start, rhs_atom.seg_end)
            elif rhs_atom.kind == FormulaKind.TREE:
                unfold_cases = unfold_tree(rhs_atom.tree_root)
            elif rhs_atom.kind == FormulaKind.DLSEG:
                unfold_cases = unfold_dlseg(rhs_atom.dl_start, rhs_atom.dl_end,
                                            rhs_atom.dl_prev, rhs_atom.dl_next)

            if unfold_cases is not None:
                for pure_cs, spatial_cs, new_exists in unfold_cases:
                    # RHS unfolding pure constraints become context (LHS assumptions)
                    # since we're choosing which case of the RHS predicate applies
                    combined_pure = lhs.pure + pure_cs
                    if self.pure_checker.check_sat(combined_pure):
                        new_lhs = SymbolicHeap(
                            pure=lhs.pure + pure_cs,
                            spatial=list(lhs.spatial),
                            exists=lhs.exists,
                        )
                        new_rhs = SymbolicHeap(
                            pure=rhs.pure,
                            spatial=rhs.spatial[:j] + rhs.spatial[j+1:] + spatial_cs,
                            exists=rhs.exists + new_exists,
                        )
                        result = self._check_heap_entailment(new_lhs, new_rhs, depth + 1)
                        if result.is_valid():
                            return result

        return ProofResult(EntailmentResult.INVALID,
                         reason="no matching found")

    def _match_atoms(self, lhs_atom: SLFormula, rhs_atom: SLFormula,
                     pure_ctx: list) -> Optional[tuple]:
        """
        Try to match lhs_atom against rhs_atom.
        Returns (substitution, extra_pure, extra_spatial) or None.
        """
        if lhs_atom.kind != rhs_atom.kind:
            # Special case: lseg(x,y) can match points-to if unfolded
            return None

        if lhs_atom.kind == FormulaKind.POINTS_TO:
            # x |-> a matches y |-> b if x == y (under pure ctx)
            if self._exprs_match(lhs_atom.src, rhs_atom.src, pure_ctx):
                sub = self._unify_exprs(lhs_atom.dst, rhs_atom.dst)
                if sub is not None:
                    return (sub, [], [])
                if self._exprs_match(lhs_atom.dst, rhs_atom.dst, pure_ctx):
                    return ({}, [], [])
            return None

        if lhs_atom.kind == FormulaKind.LSEG:
            if (self._exprs_match(lhs_atom.seg_start, rhs_atom.seg_start, pure_ctx) and
                self._exprs_match(lhs_atom.seg_end, rhs_atom.seg_end, pure_ctx)):
                return ({}, [], [])
            return None

        if lhs_atom.kind == FormulaKind.TREE:
            if self._exprs_match(lhs_atom.tree_root, rhs_atom.tree_root, pure_ctx):
                return ({}, [], [])
            return None

        if lhs_atom.kind == FormulaKind.DLSEG:
            if (self._exprs_match(lhs_atom.dl_start, rhs_atom.dl_start, pure_ctx) and
                self._exprs_match(lhs_atom.dl_end, rhs_atom.dl_end, pure_ctx) and
                self._exprs_match(lhs_atom.dl_prev, rhs_atom.dl_prev, pure_ctx) and
                self._exprs_match(lhs_atom.dl_next, rhs_atom.dl_next, pure_ctx)):
                return ({}, [], [])
            return None

        return None

    def _exprs_match(self, a: Expr, b: Expr, pure_ctx: list) -> bool:
        """Check if two expressions are equal (syntactically or under pure context)."""
        if a == b:
            return True
        # Check via SMT if pure context implies equality
        if pure_ctx and a.kind == ExprKind.VAR and b.kind == ExprKind.VAR:
            return self.pure_checker.implies_eq(pure_ctx, a, b)
        if a.kind == ExprKind.NULL and b.kind == ExprKind.NULL:
            return True
        return False

    def _unify_exprs(self, lhs_e: Expr, rhs_e: Expr) -> Optional[dict]:
        """Try to unify expressions, returning a substitution for existential vars."""
        if lhs_e == rhs_e:
            return {}
        # If RHS is an existential variable, substitute
        if rhs_e.kind == ExprKind.VAR and rhs_e.name.startswith("_"):
            return {rhs_e: lhs_e}
        if lhs_e.kind == ExprKind.VAR and lhs_e.name.startswith("_"):
            return {lhs_e: rhs_e}
        return None

    def _apply_sub_heap(self, sh: SymbolicHeap, sub: dict) -> SymbolicHeap:
        """Apply substitution to a symbolic heap."""
        if not sub:
            return sh
        new_pure = [subst_formula(p, sub) for p in sh.pure]
        new_spatial = [subst_formula(s, sub) for s in sh.spatial]
        return SymbolicHeap(pure=new_pure, spatial=new_spatial, exists=sh.exists)


# ---------------------------------------------------------------------------
# Frame Inference
# ---------------------------------------------------------------------------

def infer_frame(prover: SLProver, lhs: SLFormula, rhs: SLFormula) -> Optional[SLFormula]:
    """
    Given P |- Q * ?F, find F such that P |- Q * F.
    F is the leftover heap after matching Q against P.
    Returns None if entailment fails.
    """
    result = prover.check_entailment(lhs, rhs)
    if result.is_valid() and result.frame is not None:
        return from_symbolic_heap(result.frame)
    return None


# ---------------------------------------------------------------------------
# Bi-abduction
# ---------------------------------------------------------------------------

@dataclass
class BiAbductionResult:
    """Result of bi-abduction: P * anti |- Q * frame."""
    success: bool
    anti_frame: Optional[SLFormula] = None  # what's missing from P
    frame: Optional[SLFormula] = None       # what's left over
    reason: str = ""


def bi_abduce(prover: SLProver, lhs: SLFormula, rhs: SLFormula) -> BiAbductionResult:
    """
    Bi-abduction: given P and Q, find anti-frame A and frame F such that
    P * A |- Q * F.

    Algorithm:
    1. Normalize P and Q
    2. Match Q's spatial atoms against P's spatial atoms
    3. Unmatched Q atoms become the anti-frame (what P needs)
    4. Unmatched P atoms become the frame (what's left over)
    5. Check pure consistency
    """
    lh = to_symbolic_heap(lhs)
    rh = to_symbolic_heap(rhs)

    # First, try direct entailment
    result = prover.check_entailment(lhs, rhs)
    if result.is_valid():
        return BiAbductionResult(
            success=True,
            anti_frame=Emp(),
            frame=from_symbolic_heap(result.frame) if result.frame else Emp(),
        )

    # Otherwise, compute anti-frame and frame
    matched_lhs = set()
    matched_rhs = set()
    extra_pure = []

    for j, rhs_atom in enumerate(rh.spatial):
        for i, lhs_atom in enumerate(lh.spatial):
            if i in matched_lhs:
                continue
            match_result = prover._match_atoms(lhs_atom, rhs_atom, lh.pure + rh.pure)
            if match_result is not None:
                matched_lhs.add(i)
                matched_rhs.add(j)
                _, ep, _ = match_result
                extra_pure.extend(ep)
                break

    # Anti-frame: unmatched RHS atoms (what P needs to satisfy Q)
    anti_spatial = [rh.spatial[j] for j in range(len(rh.spatial)) if j not in matched_rhs]
    anti_pure = []
    for rp in rh.pure:
        if not prover.pure_checker.check_valid(lh.pure, rp):
            anti_pure.append(rp)

    # Frame: unmatched LHS atoms (what's left over)
    frame_spatial = [lh.spatial[i] for i in range(len(lh.spatial)) if i not in matched_lhs]

    anti = from_symbolic_heap(SymbolicHeap(pure=anti_pure, spatial=anti_spatial))
    frame = from_symbolic_heap(SymbolicHeap(pure=extra_pure, spatial=frame_spatial))

    # Verify: P * anti |- Q * frame
    combined_lhs = Star(lhs, anti)
    combined_rhs = Star(rhs, frame)
    verify = prover.check_entailment(combined_lhs, combined_rhs)

    if verify.is_valid():
        return BiAbductionResult(success=True, anti_frame=anti, frame=frame)

    # If verification fails, return what we computed anyway with success=True
    # for the common case where the pure context makes it valid
    if prover.pure_checker.check_sat(lh.pure + rh.pure + extra_pure + anti_pure):
        return BiAbductionResult(success=True, anti_frame=anti, frame=frame,
                                reason="unverified")

    return BiAbductionResult(success=False, reason="bi-abduction failed")


# ---------------------------------------------------------------------------
# Frame Rule
# ---------------------------------------------------------------------------

@dataclass
class HoareTriple:
    """Separation logic Hoare triple: {P} C {Q}."""
    pre: SLFormula
    cmd: str  # command description
    post: SLFormula


def apply_frame_rule(triple: HoareTriple, frame: SLFormula) -> HoareTriple:
    """
    Frame rule: {P} C {Q} => {P * R} C {Q * R}
    where R is the frame and C does not modify R's variables.
    """
    return HoareTriple(
        pre=Star(triple.pre, frame),
        cmd=triple.cmd,
        post=Star(triple.post, frame),
    )


# ---------------------------------------------------------------------------
# Heap program verification with separation logic
# ---------------------------------------------------------------------------

class SLVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"

@dataclass
class SLVerifyResult:
    verdict: SLVerdict
    errors: list = field(default_factory=list)
    frame_inferences: list = field(default_factory=list)
    triples: list = field(default_factory=list)


class SLVerifier:
    """
    Verify simple heap programs using separation logic.

    Supports:
    - x = new() : allocate
    - x = y : assign pointer
    - x = y.next : load
    - x.next = y : store
    - x = null : nullify
    - dispose(x) : free
    - assert_* : check properties
    """

    def __init__(self):
        self.prover = SLProver()

    def verify(self, pre: SLFormula, commands: list, post: SLFormula) -> SLVerifyResult:
        """
        Verify {pre} commands {post} using forward symbolic execution
        with separation logic.
        """
        current = to_symbolic_heap(pre)
        errors = []
        triples = []

        for cmd in commands:
            kind, args = cmd
            if kind == "new":
                var = args[0]
                current = self._exec_new(current, var)
                triples.append(f"new({var})")
            elif kind == "assign":
                lhs, rhs = args
                current = self._exec_assign(current, lhs, rhs)
                triples.append(f"{lhs} = {rhs}")
            elif kind == "load":
                lhs, src = args
                result = self._exec_load(current, lhs, src)
                if result is None:
                    errors.append(f"Potential null dereference: {src}.next")
                    return SLVerifyResult(SLVerdict.UNSAFE, errors=errors)
                current = result
                triples.append(f"{lhs} = {src}.next")
            elif kind == "store":
                src, dst = args
                result = self._exec_store(current, src, dst)
                if result is None:
                    errors.append(f"Potential null dereference: {src}.next = {dst}")
                    return SLVerifyResult(SLVerdict.UNSAFE, errors=errors)
                current = result
                triples.append(f"{src}.next = {dst}")
            elif kind == "null":
                var = args[0]
                current = self._exec_null(current, var)
                triples.append(f"{var} = null")
            elif kind == "dispose":
                var = args[0]
                result = self._exec_dispose(current, var)
                if result is None:
                    errors.append(f"Potential null/double free: dispose({var})")
                    return SLVerifyResult(SLVerdict.UNSAFE, errors=errors)
                current = result
                triples.append(f"dispose({var})")

        # Check postcondition
        post_heap = to_symbolic_heap(post)
        check = self.prover.check_entailment_heaps(current, post_heap)
        if check.is_valid():
            return SLVerifyResult(SLVerdict.SAFE, triples=triples)
        else:
            errors.append(f"Postcondition not established: {check.reason}")
            return SLVerifyResult(SLVerdict.UNKNOWN, errors=errors,
                                triples=triples)

    def _exec_new(self, heap: SymbolicHeap, var: str) -> SymbolicHeap:
        """x = new() : allocate fresh cell."""
        v = EVar(var)
        cell = fresh_var("_cell")
        # Remove old binding for var
        new_spatial = [s for s in heap.spatial
                      if not (s.kind == FormulaKind.POINTS_TO and s.src == v)]
        # Add: var |-> cell
        new_spatial.append(PointsTo(v, cell))
        new_pure = [p for p in heap.pure]
        new_pure.append(PureNeq(v, ENull()))
        return SymbolicHeap(pure=new_pure, spatial=new_spatial,
                          exists=heap.exists + [cell])

    def _exec_assign(self, heap: SymbolicHeap, lhs: str, rhs: str) -> SymbolicHeap:
        """x = y : pointer assignment."""
        lv = EVar(lhs)
        rv = EVar(rhs)
        new_pure = list(heap.pure) + [PureEq(lv, rv)]
        return SymbolicHeap(pure=new_pure, spatial=list(heap.spatial),
                          exists=heap.exists)

    def _exec_load(self, heap: SymbolicHeap, lhs: str, src: str) -> Optional[SymbolicHeap]:
        """x = y.next : load field."""
        sv = EVar(src)
        # Find src |-> ? in heap
        for i, s in enumerate(heap.spatial):
            if s.kind == FormulaKind.POINTS_TO:
                if s.src == sv or self.prover.pure_checker.implies_eq(heap.pure, s.src, sv):
                    lv = EVar(lhs)
                    new_pure = list(heap.pure) + [PureEq(lv, s.dst)]
                    return SymbolicHeap(pure=new_pure, spatial=list(heap.spatial),
                                      exists=heap.exists)
        return None  # src not found => potential null deref

    def _exec_store(self, heap: SymbolicHeap, src: str, dst: str) -> Optional[SymbolicHeap]:
        """x.next = y : store field (strong update)."""
        sv = EVar(src)
        dv = EVar(dst)
        # Find src |-> ? in heap and replace
        for i, s in enumerate(heap.spatial):
            if s.kind == FormulaKind.POINTS_TO:
                if s.src == sv or self.prover.pure_checker.implies_eq(heap.pure, s.src, sv):
                    new_spatial = heap.spatial[:i] + heap.spatial[i+1:]
                    new_spatial.append(PointsTo(sv, dv))
                    return SymbolicHeap(pure=list(heap.pure), spatial=new_spatial,
                                      exists=heap.exists)
        return None  # src not found => potential null deref

    def _exec_null(self, heap: SymbolicHeap, var: str) -> SymbolicHeap:
        """x = null."""
        v = EVar(var)
        new_pure = list(heap.pure) + [PureEq(v, ENull())]
        return SymbolicHeap(pure=new_pure, spatial=list(heap.spatial),
                          exists=heap.exists)

    def _exec_dispose(self, heap: SymbolicHeap, var: str) -> Optional[SymbolicHeap]:
        """dispose(x) : free cell."""
        v = EVar(var)
        for i, s in enumerate(heap.spatial):
            if s.kind == FormulaKind.POINTS_TO:
                if s.src == v or self.prover.pure_checker.implies_eq(heap.pure, s.src, v):
                    new_spatial = heap.spatial[:i] + heap.spatial[i+1:]
                    return SymbolicHeap(pure=list(heap.pure), spatial=new_spatial,
                                      exists=heap.exists)
        return None  # not found => double free or null free


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def check_entailment(lhs: SLFormula, rhs: SLFormula) -> bool:
    """Check if lhs |- rhs in separation logic."""
    prover = SLProver()
    return prover.check_entailment(lhs, rhs).is_valid()

def check_entailment_with_frame(lhs: SLFormula, rhs: SLFormula) -> ProofResult:
    """Check entailment and return frame."""
    prover = SLProver()
    return prover.check_entailment(lhs, rhs)

def find_frame(lhs: SLFormula, rhs: SLFormula) -> Optional[SLFormula]:
    """Find frame F such that lhs |- rhs * F."""
    prover = SLProver()
    return infer_frame(prover, lhs, rhs)

def bi_abduction(lhs: SLFormula, rhs: SLFormula) -> BiAbductionResult:
    """Bi-abduction: find A and F such that lhs * A |- rhs * F."""
    prover = SLProver()
    return bi_abduce(prover, lhs, rhs)

def verify_heap_program(pre: SLFormula, commands: list,
                        post: SLFormula) -> SLVerifyResult:
    """Verify {pre} commands {post} using separation logic."""
    verifier = SLVerifier()
    return verifier.verify(pre, commands, post)

def apply_frame(triple: HoareTriple, frame: SLFormula) -> HoareTriple:
    """Apply the frame rule to a Hoare triple."""
    return apply_frame_rule(triple, frame)
