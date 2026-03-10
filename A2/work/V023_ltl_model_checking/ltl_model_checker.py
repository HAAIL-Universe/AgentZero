"""
V023: LTL Model Checking
========================
Linear Temporal Logic model checking via Buchi automata product construction.

Composes V021 (BDD-based model checking) for the underlying state space manipulation.

Approach:
1. Parse LTL formula
2. Negate formula (we check for counterexamples)
3. Convert negated formula to Buchi automaton (tableau construction)
4. Product construction: system x Buchi automaton
5. Fair cycle detection: is there an accepting run?
6. If yes -> property VIOLATED (with counterexample)
7. If no -> property HOLDS

LTL Operators:
- Atomic propositions: variable names (boolean)
- Boolean: AND, OR, NOT, IMPLIES, IFF, TRUE, FALSE
- Temporal: X (next), F (finally/eventually), G (globally/always),
            U (until), R (release), W (weak until)

Fairness constraints:
- Strong fairness (compassion): if enabled infinitely often, taken infinitely often
- Weak fairness (justice): if continuously enabled, eventually taken
- Implemented as additional Buchi acceptance sets
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, FrozenSet
from enum import Enum, auto
import math

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

sys.path.insert(0, os.path.join(_a2, 'work', 'V021_bdd_model_checking'))
from bdd_model_checker import (
    BDD, BooleanTS, SymbolicModelChecker, MCOutput, MCResult,
    make_boolean_ts, check_boolean_system
)


# ============================================================
# LTL Formula AST
# ============================================================

class LTLOp(Enum):
    # Atomic
    ATOM = auto()
    TRUE = auto()
    FALSE = auto()
    # Boolean
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    IFF = auto()
    # Temporal
    X = auto()      # neXt
    F = auto()      # Finally (eventually)
    G = auto()      # Globally (always)
    U = auto()      # Until (strong)
    R = auto()      # Release
    W = auto()      # Weak until


@dataclass(frozen=True)
class LTL:
    """LTL formula node."""
    op: LTLOp
    name: str = ""              # for ATOM
    left: Optional['LTL'] = None
    right: Optional['LTL'] = None

    def __repr__(self):
        if self.op == LTLOp.ATOM:
            return self.name
        if self.op == LTLOp.TRUE:
            return "true"
        if self.op == LTLOp.FALSE:
            return "false"
        if self.op == LTLOp.NOT:
            return f"!{self.left}"
        if self.op == LTLOp.AND:
            return f"({self.left} & {self.right})"
        if self.op == LTLOp.OR:
            return f"({self.left} | {self.right})"
        if self.op == LTLOp.IMPLIES:
            return f"({self.left} -> {self.right})"
        if self.op == LTLOp.IFF:
            return f"({self.left} <-> {self.right})"
        if self.op == LTLOp.X:
            return f"X({self.left})"
        if self.op == LTLOp.F:
            return f"F({self.left})"
        if self.op == LTLOp.G:
            return f"G({self.left})"
        if self.op == LTLOp.U:
            return f"({self.left} U {self.right})"
        if self.op == LTLOp.R:
            return f"({self.left} R {self.right})"
        if self.op == LTLOp.W:
            return f"({self.left} W {self.right})"
        return f"LTL({self.op})"


# Convenience constructors
def Atom(name: str) -> LTL:
    return LTL(LTLOp.ATOM, name=name)

def LTLTrue() -> LTL:
    return LTL(LTLOp.TRUE)

def LTLFalse() -> LTL:
    return LTL(LTLOp.FALSE)

def Not(f: LTL) -> LTL:
    # Simplifications
    if f.op == LTLOp.TRUE:
        return LTLFalse()
    if f.op == LTLOp.FALSE:
        return LTLTrue()
    if f.op == LTLOp.NOT:
        return f.left
    return LTL(LTLOp.NOT, left=f)

def And(a: LTL, b: LTL) -> LTL:
    if a.op == LTLOp.TRUE:
        return b
    if b.op == LTLOp.TRUE:
        return a
    if a.op == LTLOp.FALSE or b.op == LTLOp.FALSE:
        return LTLFalse()
    return LTL(LTLOp.AND, left=a, right=b)

def Or(a: LTL, b: LTL) -> LTL:
    if a.op == LTLOp.FALSE:
        return b
    if b.op == LTLOp.FALSE:
        return a
    if a.op == LTLOp.TRUE or b.op == LTLOp.TRUE:
        return LTLTrue()
    return LTL(LTLOp.OR, left=a, right=b)

def Implies(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.IMPLIES, left=a, right=b)

def Iff(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.IFF, left=a, right=b)

def Next(f: LTL) -> LTL:
    return LTL(LTLOp.X, left=f)

def Finally(f: LTL) -> LTL:
    """F(f) = true U f"""
    return LTL(LTLOp.F, left=f)

def Globally(f: LTL) -> LTL:
    """G(f) = false R f"""
    return LTL(LTLOp.G, left=f)

def Until(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.U, left=a, right=b)

def Release(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.R, left=a, right=b)

def WeakUntil(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.W, left=a, right=b)


# ============================================================
# LTL Formula Utilities
# ============================================================

def atoms(f: LTL) -> Set[str]:
    """Extract all atomic propositions from an LTL formula."""
    if f.op == LTLOp.ATOM:
        return {f.name}
    if f.op in (LTLOp.TRUE, LTLOp.FALSE):
        return set()
    result = set()
    if f.left:
        result |= atoms(f.left)
    if f.right:
        result |= atoms(f.right)
    return result


def nnf(f: LTL) -> LTL:
    """Convert LTL formula to Negation Normal Form (push NOT inward)."""
    if f.op == LTLOp.ATOM or f.op in (LTLOp.TRUE, LTLOp.FALSE):
        return f
    if f.op == LTLOp.NOT:
        return _nnf_neg(f.left)
    if f.op == LTLOp.AND:
        return And(nnf(f.left), nnf(f.right))
    if f.op == LTLOp.OR:
        return Or(nnf(f.left), nnf(f.right))
    if f.op == LTLOp.IMPLIES:
        return Or(_nnf_neg(f.left), nnf(f.right))
    if f.op == LTLOp.IFF:
        return And(Or(_nnf_neg(f.left), nnf(f.right)),
                   Or(nnf(f.left), _nnf_neg(f.right)))
    if f.op == LTLOp.X:
        return Next(nnf(f.left))
    if f.op == LTLOp.F:
        # F(a) = true U a
        return Until(LTLTrue(), nnf(f.left))
    if f.op == LTLOp.G:
        # G(a) = false R a
        return Release(LTLFalse(), nnf(f.left))
    if f.op == LTLOp.U:
        return Until(nnf(f.left), nnf(f.right))
    if f.op == LTLOp.R:
        return Release(nnf(f.left), nnf(f.right))
    if f.op == LTLOp.W:
        # a W b = (a U b) | G(a) = (a U b) | (false R a)
        return Or(Until(nnf(f.left), nnf(f.right)),
                  Release(LTLFalse(), nnf(f.left)))
    return f


def _nnf_neg(f: LTL) -> LTL:
    """NNF of NOT(f), pushing negation inward."""
    if f.op == LTLOp.ATOM:
        return Not(f)
    if f.op == LTLOp.TRUE:
        return LTLFalse()
    if f.op == LTLOp.FALSE:
        return LTLTrue()
    if f.op == LTLOp.NOT:
        return nnf(f.left)
    if f.op == LTLOp.AND:
        # !(a & b) = !a | !b
        return Or(_nnf_neg(f.left), _nnf_neg(f.right))
    if f.op == LTLOp.OR:
        # !(a | b) = !a & !b
        return And(_nnf_neg(f.left), _nnf_neg(f.right))
    if f.op == LTLOp.IMPLIES:
        # !(a -> b) = a & !b
        return And(nnf(f.left), _nnf_neg(f.right))
    if f.op == LTLOp.IFF:
        # !(a <-> b) = (a & !b) | (!a & b)
        return Or(And(nnf(f.left), _nnf_neg(f.right)),
                  And(_nnf_neg(f.left), nnf(f.right)))
    if f.op == LTLOp.X:
        # !X(a) = X(!a)
        return Next(_nnf_neg(f.left))
    if f.op == LTLOp.F:
        # !F(a) = G(!a) = false R !a
        return Release(LTLFalse(), _nnf_neg(f.left))
    if f.op == LTLOp.G:
        # !G(a) = F(!a) = true U !a
        return Until(LTLTrue(), _nnf_neg(f.left))
    if f.op == LTLOp.U:
        # !(a U b) = (!a R !b)
        return Release(_nnf_neg(f.left), _nnf_neg(f.right))
    if f.op == LTLOp.R:
        # !(a R b) = (!a U !b)
        return Until(_nnf_neg(f.left), _nnf_neg(f.right))
    if f.op == LTLOp.W:
        # !(a W b) = !((a U b) | G(a)) = !(a U b) & !G(a) = (!a R !b) & F(!a)
        # = (!a R !b) & (true U !a)
        return And(Release(_nnf_neg(f.left), _nnf_neg(f.right)),
                   Until(LTLTrue(), _nnf_neg(f.left)))
    return Not(f)


def subformulas(f: LTL) -> Set[LTL]:
    """Collect all subformulas of an LTL formula."""
    result = {f}
    if f.left:
        result |= subformulas(f.left)
    if f.right:
        result |= subformulas(f.right)
    return result


def until_subformulas(f: LTL) -> List[LTL]:
    """Extract all Until subformulas (these generate acceptance sets)."""
    result = []
    for sf in subformulas(f):
        if sf.op == LTLOp.U:
            result.append(sf)
    return result


# ============================================================
# Generalized Buchi Automaton
# ============================================================

@dataclass
class Label:
    """Transition label: (positive atoms, negative atoms)."""
    pos: FrozenSet[str]
    neg: FrozenSet[str]

    def __hash__(self):
        return hash((self.pos, self.neg))

    def __eq__(self, other):
        return isinstance(other, Label) and self.pos == other.pos and self.neg == other.neg


@dataclass
class GBA:
    """Generalized Buchi Automaton (multiple acceptance sets)."""
    states: Set[FrozenSet[LTL]]              # Set of states (each is a set of formulas)
    initial: Set[FrozenSet[LTL]]             # Initial states
    transitions: Dict[FrozenSet[LTL], List[Tuple[Label, FrozenSet[LTL]]]]
                                             # state -> [(label, next_state)]
    acceptance: List[Set[FrozenSet[LTL]]]    # List of acceptance sets (GBA)
    ap: Set[str]                              # Atomic propositions


@dataclass
class NBA:
    """Non-deterministic Buchi Automaton (single acceptance set)."""
    states: Set[int]
    initial: Set[int]
    transitions: Dict[int, List[Tuple[Label, int]]]  # state -> [(label, next)]
    accepting: Set[int]
    ap: Set[str]


# ============================================================
# LTL to GBA (Tableau Construction)
# ============================================================

def _expand_formula(f: LTL) -> List[List[LTL]]:
    """
    Expand an LTL formula into disjunctive normal form of obligations.
    Each inner list is a conjunction of current-state and next-state requirements.
    Returns list of (current_atoms, next_obligations) pairs.
    """
    if f.op == LTLOp.ATOM:
        return [([f], [])]
    if f.op == LTLOp.TRUE:
        return [([], [])]
    if f.op == LTLOp.FALSE:
        return []  # unsatisfiable
    if f.op == LTLOp.NOT:
        if f.left.op == LTLOp.ATOM:
            return [([f], [])]
        return [([f], [])]  # negated atom
    if f.op == LTLOp.AND:
        # Both must hold
        return [([f], [])]
    if f.op == LTLOp.OR:
        # Either can hold
        return [([f], [])]
    if f.op == LTLOp.X:
        # Obligation on next state
        return [([], [f.left])]
    if f.op == LTLOp.U:
        # a U b = b | (a & X(a U b))
        return [([f], [])]
    if f.op == LTLOp.R:
        # a R b = b & (a | X(a R b))
        return [([f], [])]
    return [([f], [])]


def ltl_to_gba(formula: LTL) -> GBA:
    """
    Convert an LTL formula (in NNF) to a Generalized Buchi Automaton
    using the tableau/expansion method.

    The states are sets of LTL subformulas that must hold at that point.
    Transitions are labeled with sets of atomic propositions that must be true.
    """
    f_nnf = nnf(formula)
    ap = atoms(f_nnf)

    # States are sets of formulas (obligations)
    # Use frozen sets for hashability
    initial_state = frozenset({f_nnf})

    states = set()
    transitions = {}
    queue = [initial_state]
    visited = set()

    while queue:
        state = queue.pop(0)
        if state in visited:
            continue
        visited.add(state)
        states.add(state)
        transitions[state] = []

        # Expand state: determine which atomic propositions must be true/false
        # and what next-state obligations arise
        successors = _expand_state(state, ap)

        for (pos, neg), next_obligations in successors:
            next_state = frozenset(next_obligations)
            lbl = Label(frozenset(pos), frozenset(neg))
            transitions[state].append((lbl, next_state))
            if next_state not in visited:
                queue.append(next_state)
            states.add(next_state)
            if next_state not in transitions:
                transitions[next_state] = []

    # Acceptance sets: one per Until subformula
    # For (a U b), accepting states are those where b holds OR (a U b) is not an obligation
    until_sfs = until_subformulas(f_nnf)
    acceptance = []
    for u_sf in until_sfs:
        acc_set = set()
        for s in states:
            # State is accepting for this Until if:
            # - u_sf is NOT in the obligation set (already satisfied), OR
            # - u_sf.right IS in the obligation set (b holds now)
            if u_sf not in s or u_sf.right in s:
                acc_set.add(s)
        acceptance.append(acc_set)

    # If no Until subformulas, all states are accepting
    if not acceptance:
        acceptance = [states.copy()]

    return GBA(
        states=states,
        initial={initial_state},
        transitions=transitions,
        acceptance=acceptance,
        ap=ap
    )


def _expand_state(state: FrozenSet[LTL], ap: Set[str]) -> List[Tuple[Tuple[Set[str], Set[str]], Set[LTL]]]:
    """
    Expand a state (set of obligations) into possible transitions.
    Returns list of ((pos_atoms, neg_atoms), next_obligations).

    Uses a worklist to process each obligation and build combinations.
    """
    # Start with one empty option: no atoms required, no next obligations
    options = [(set(), set(), set())]  # (pos_atoms, neg_atoms, next_obligations)

    for formula in state:
        new_options = []
        for pos, neg, nxt in options:
            expansions = _expand_single(formula)
            for req_pos, req_neg, req_nxt in expansions:
                # Check consistency of atom requirements
                if req_pos & neg or req_neg & pos:
                    continue  # Contradictory
                new_options.append((
                    pos | req_pos,
                    neg | req_neg,
                    nxt | req_nxt
                ))
        options = new_options

    # Convert to ((pos, neg), next_obligations) format
    result = []
    for pos, neg, nxt in options:
        result.append(((pos, neg), nxt))
    return result


def _expand_single(f: LTL) -> List[Tuple[Set[str], Set[str], Set[LTL]]]:
    """
    Expand a single formula into (pos_atoms, neg_atoms, next_obligations).
    Returns a list of alternatives (disjuncts).
    """
    if f.op == LTLOp.TRUE:
        return [(set(), set(), set())]
    if f.op == LTLOp.FALSE:
        return []  # No way to satisfy
    if f.op == LTLOp.ATOM:
        return [({f.name}, set(), set())]
    if f.op == LTLOp.NOT:
        if f.left.op == LTLOp.ATOM:
            return [(set(), {f.left.name}, set())]
        # For negated non-atoms in NNF, this shouldn't happen
        # but handle gracefully
        return [(set(), set(), set())]
    if f.op == LTLOp.AND:
        # Both must hold - combine expansions
        left_exps = _expand_single(f.left)
        right_exps = _expand_single(f.right)
        result = []
        for lp, ln, lnxt in left_exps:
            for rp, rn, rnxt in right_exps:
                if lp & rn or ln & rp:
                    continue  # Contradictory
                result.append((lp | rp, ln | rn, lnxt | rnxt))
        return result
    if f.op == LTLOp.OR:
        # Either can hold
        return _expand_single(f.left) + _expand_single(f.right)
    if f.op == LTLOp.X:
        # Next-state obligation
        return [(set(), set(), {f.left})]
    if f.op == LTLOp.U:
        # a U b: either b holds now, or (a holds now AND X(a U b))
        b_exps = _expand_single(f.right)
        a_exps = _expand_single(f.left)
        # Option 1: b holds now (Until satisfied)
        result = list(b_exps)
        # Option 2: a holds now, carry obligation to next step
        for ap_set, an_set, a_nxt in a_exps:
            result.append((ap_set, an_set, a_nxt | {f}))
        return result
    if f.op == LTLOp.R:
        # a R b: b holds now AND (a holds now OR X(a R b))
        # = (a & b) | (b & X(a R b))
        b_exps = _expand_single(f.right)
        a_exps = _expand_single(f.left)
        result = []
        # Option 1: both a and b hold (Release discharged)
        for bp, bn, bnxt in b_exps:
            for ap_set, an_set, a_nxt in a_exps:
                if bp & an_set or bn & ap_set:
                    continue
                result.append((bp | ap_set, bn | an_set, bnxt | a_nxt))
        # Option 2: b holds and carry obligation
        for bp, bn, bnxt in b_exps:
            result.append((bp, bn, bnxt | {f}))
        return result

    # Fallback: treat as opaque (shouldn't reach here for NNF formulas)
    return [(set(), set(), set())]


# ============================================================
# GBA to NBA (degeneralization)
# ============================================================

def gba_to_nba(gba: GBA) -> NBA:
    """
    Convert a Generalized Buchi Automaton (multiple acceptance sets)
    to a standard Buchi Automaton (single acceptance set).

    Uses the standard product construction with a counter tracking
    which acceptance set to target next.
    """
    k = len(gba.acceptance)
    if k == 0:
        # No acceptance condition - all states accepting
        state_map = {}
        idx = 0
        for s in gba.states:
            state_map[(s, 0)] = idx
            idx += 1
        nba_states = set(range(idx))
        nba_initial = set()
        for s in gba.initial:
            if (s, 0) in state_map:
                nba_initial.add(state_map[(s, 0)])
        nba_trans = {}
        for s in gba.states:
            src = state_map[(s, 0)]
            nba_trans[src] = []
            for label, dst in gba.transitions.get(s, []):
                if (dst, 0) in state_map:
                    nba_trans[src].append((label, state_map[(dst, 0)]))
        return NBA(nba_states, nba_initial, nba_trans, nba_states, gba.ap)

    if k == 1:
        # Already a standard Buchi
        state_map = {}
        idx = 0
        for s in gba.states:
            state_map[s] = idx
            idx += 1
        nba_states = set(range(idx))
        nba_initial = {state_map[s] for s in gba.initial if s in state_map}
        nba_trans = {}
        for s in gba.states:
            src = state_map[s]
            nba_trans[src] = []
            for label, dst in gba.transitions.get(s, []):
                if dst in state_map:
                    nba_trans[src].append((label, state_map[dst]))
        accepting = {state_map[s] for s in gba.acceptance[0] if s in state_map}
        return NBA(nba_states, nba_initial, nba_trans, accepting, gba.ap)

    # General case: product with counter
    # State = (gba_state, counter) where counter in [0, k)
    # Transition: (s, i) -> (s', i') where:
    #   - s -> s' is a GBA transition
    #   - If s in acceptance[i], then i' = (i+1) % k
    #   - Otherwise i' = i
    # Accepting: states with counter = 0 (just visited acceptance[k-1])

    state_map = {}
    idx = 0
    for s in gba.states:
        for i in range(k):
            state_map[(s, i)] = idx
            idx += 1

    nba_states = set(range(idx))
    nba_initial = set()
    for s in gba.initial:
        if (s, 0) in state_map:
            nba_initial.add(state_map[(s, 0)])

    nba_trans = {}
    for s in gba.states:
        for i in range(k):
            src = state_map[(s, i)]
            nba_trans[src] = []
            # Determine next counter value
            if s in gba.acceptance[i]:
                next_i = (i + 1) % k
            else:
                next_i = i

            for label, dst in gba.transitions.get(s, []):
                if (dst, next_i) in state_map:
                    nba_trans[src].append((label, state_map[(dst, next_i)]))

    # Accepting states: counter = 0
    accepting = set()
    for s in gba.states:
        if (s, 0) in state_map:
            accepting.add(state_map[(s, 0)])

    return NBA(nba_states, nba_initial, nba_trans, accepting, gba.ap)


# ============================================================
# BDD-based LTL Model Checking
# ============================================================

@dataclass
class LTLResult:
    """Result of LTL model checking."""
    holds: bool                                    # Does the property hold?
    counterexample: Optional[Tuple[List[Dict[str, bool]], List[Dict[str, bool]]]] = None
                                                   # (prefix, cycle) if violated
    automaton_states: int = 0                      # Buchi automaton size
    product_vars: int = 0                          # Product system variables
    fixpoint_iterations: int = 0
    method: str = "ltl"


class LTLModelChecker:
    """
    BDD-based LTL model checker.

    Takes a BooleanTS (from V021) and checks LTL properties.
    """

    def __init__(self, ts: BooleanTS):
        self.ts = ts
        self.bdd = ts.bdd

    def check(self, formula: LTL, max_steps: int = 500) -> LTLResult:
        """
        Check if an LTL formula holds on the transition system.

        Returns LTLResult with holds=True if property holds everywhere,
        or holds=False with counterexample (prefix, cycle) if violated.
        """
        # Step 1: Negate the formula (we look for counterexamples)
        neg_formula = Not(formula)

        # Step 2: Convert negated formula to NNF
        neg_nnf = nnf(neg_formula)

        # Step 3: Build Buchi automaton for negated formula
        gba = ltl_to_gba(neg_nnf)
        nba = gba_to_nba(gba)

        # Step 4: Build product system (TS x NBA)
        product_bdd, product_ts, aut_var_info = self._build_product(nba)

        # Step 5: Check for accepting cycles (fair cycle detection)
        mc = SymbolicModelChecker(product_ts)
        accepting_bdd = aut_var_info['accepting_bdd']

        has_fair_cycle, iterations = self._check_fair_cycle(
            mc, accepting_bdd, max_steps
        )

        if not has_fair_cycle:
            return LTLResult(
                holds=True,
                automaton_states=len(nba.states),
                product_vars=len(product_ts.state_vars),
                fixpoint_iterations=iterations
            )

        # Step 6: Extract counterexample
        cex = self._extract_counterexample(
            mc, accepting_bdd, aut_var_info, max_steps
        )

        return LTLResult(
            holds=False,
            counterexample=cex,
            automaton_states=len(nba.states),
            product_vars=len(product_ts.state_vars),
            fixpoint_iterations=iterations
        )

    def _build_product(self, nba: NBA) -> Tuple[BDD, BooleanTS, dict]:
        """
        Build product transition system: original TS x NBA.

        The product state = (system_state, automaton_state).
        Automaton state is encoded in log2(|Q|) boolean variables.
        """
        n_aut_states = len(nba.states)
        if n_aut_states == 0:
            # Empty automaton - property trivially holds
            raise ValueError("Empty automaton")

        # Number of bits for automaton state
        n_bits = max(1, math.ceil(math.log2(max(n_aut_states, 2))))

        # Create new BDD manager with enough variables
        # Variables: system vars + automaton bits (current + next)
        sys_vars = self.ts.state_vars
        n_sys = len(sys_vars)

        # New BDD with space for all variables
        bdd = BDD(2 * (n_sys + n_bits))

        # Assign variable indices
        # Current: sys_0, ..., sys_{n-1}, aut_0, ..., aut_{b-1}
        # Next: sys_0', ..., sys_{n-1}', aut_0', ..., aut_{b-1}'
        all_current = []
        all_next = []
        sys_var_map = {}
        sys_next_map = {}
        aut_var_indices = []
        aut_next_indices = []

        idx = 0
        for v in sys_vars:
            bdd.named_var(v)
            sys_var_map[v] = idx
            all_current.append(v)
            idx += 1

        for i in range(n_bits):
            name = f"__aut_{i}"
            bdd.named_var(name)
            aut_var_indices.append(idx)
            all_current.append(name)
            idx += 1

        for v in sys_vars:
            nv = f"{v}'"
            bdd.named_var(nv)
            sys_next_map[v] = idx
            all_next.append(nv)
            idx += 1

        for i in range(n_bits):
            name = f"__aut_{i}'"
            bdd.named_var(name)
            aut_next_indices.append(idx)
            all_next.append(name)
            idx += 1

        # Encode automaton state as bit pattern
        def encode_aut_state(state_id, use_next=False):
            """BDD representing automaton being in state_id."""
            indices = aut_next_indices if use_next else aut_var_indices
            result = bdd.TRUE
            for bit in range(n_bits):
                v = bdd.var(indices[bit])
                if (state_id >> bit) & 1:
                    result = bdd.AND(result, v)
                else:
                    result = bdd.AND(result, bdd.NOT(v))
            return result

        # Encode label (positive + negative atoms) as BDD
        def encode_label(label: Label):
            """BDD representing the label constraint on system variables."""
            result = bdd.TRUE
            for atom in label.pos:
                if atom in sys_var_map:
                    v = bdd.var(sys_var_map[atom])
                    result = bdd.AND(result, v)
            for atom in label.neg:
                if atom in sys_var_map:
                    v = bdd.var(sys_var_map[atom])
                    result = bdd.AND(result, bdd.NOT(v))
            return result

        # Build initial states: sys_init AND aut_initial
        # Re-encode system init in new BDD
        sys_init = self._re_encode_bdd(bdd, self.ts.init, sys_var_map)
        product_init = bdd.FALSE
        for q0 in nba.initial:
            if q0 < (1 << n_bits):
                product_init = bdd.OR(product_init, bdd.AND(sys_init, encode_aut_state(q0)))

        # Build transition relation
        # For each automaton transition (q, label, q'):
        #   product_trans includes: aut_state=q AND label(sys_vars) AND sys_trans AND aut_state'=q'
        sys_trans = self._re_encode_bdd_with_next(
            bdd, self.ts.trans, sys_var_map, sys_next_map
        )

        product_trans = bdd.FALSE
        for q in nba.states:
            if q >= (1 << n_bits):
                continue
            q_bdd = encode_aut_state(q)
            for label, q_next in nba.transitions.get(q, []):
                if q_next >= (1 << n_bits):
                    continue
                label_bdd = encode_label(label)
                q_next_bdd = encode_aut_state(q_next, use_next=True)
                trans_part = bdd.and_all([q_bdd, label_bdd, sys_trans, q_next_bdd])
                product_trans = bdd.OR(product_trans, trans_part)

        # Build accepting states BDD
        accepting_bdd = bdd.FALSE
        for q in nba.accepting:
            if q < (1 << n_bits):
                accepting_bdd = bdd.OR(accepting_bdd, encode_aut_state(q))

        # Build product transition system
        product_ts = BooleanTS(
            bdd=bdd,
            state_vars=all_current,
            next_vars=all_next,
            init=product_init,
            trans=product_trans,
            var_indices={v: bdd.var_index(v) for v in all_current},
            next_indices={v: bdd.var_index(f"{v}'") for v in all_current}
        )

        aut_var_info = {
            'accepting_bdd': accepting_bdd,
            'n_bits': n_bits,
            'aut_var_indices': aut_var_indices,
            'aut_next_indices': aut_next_indices,
            'sys_vars': sys_vars,
            'sys_var_map': sys_var_map,
            'encode_aut_state': encode_aut_state,
        }

        return bdd, product_ts, aut_var_info

    def _re_encode_bdd(self, new_bdd: BDD, old_bdd_node, var_map: Dict[str, int]):
        """Re-encode a BDD from old manager to new manager using variable mapping."""
        old_bdd = self.ts.bdd

        # Simple approach: enumerate satisfying assignments and reconstruct
        if old_bdd_node == old_bdd.TRUE:
            return new_bdd.TRUE
        if old_bdd_node == old_bdd.FALSE:
            return new_bdd.FALSE

        # Get all satisfying assignments
        n_vars = len(self.ts.state_vars)
        assignments = old_bdd.all_sat(old_bdd_node, n_vars)

        result = new_bdd.FALSE
        for assignment in assignments:
            clause = new_bdd.TRUE
            for var_idx, val in assignment.items():
                # Map old var index to name, then to new index
                old_name = old_bdd.var_name(var_idx)
                if old_name in var_map:
                    new_idx = var_map[old_name]
                    v = new_bdd.var(new_idx)
                    if val:
                        clause = new_bdd.AND(clause, v)
                    else:
                        clause = new_bdd.AND(clause, new_bdd.NOT(v))
            result = new_bdd.OR(result, clause)

        return result

    def _re_encode_bdd_with_next(self, new_bdd: BDD, old_bdd_node,
                                  sys_var_map: Dict[str, int],
                                  sys_next_map: Dict[str, int]):
        """Re-encode transition BDD, mapping both current and next vars."""
        old_bdd = self.ts.bdd

        if old_bdd_node == old_bdd.TRUE:
            return new_bdd.TRUE
        if old_bdd_node == old_bdd.FALSE:
            return new_bdd.FALSE

        n_vars = len(self.ts.state_vars) + len(self.ts.next_vars)
        assignments = old_bdd.all_sat(old_bdd_node, n_vars)

        result = new_bdd.FALSE
        for assignment in assignments:
            clause = new_bdd.TRUE
            for var_idx, val in assignment.items():
                old_name = old_bdd.var_name(var_idx)
                new_idx = None
                if old_name in sys_var_map:
                    new_idx = sys_var_map[old_name]
                elif old_name.endswith("'"):
                    base = old_name[:-1]
                    if base in sys_next_map:
                        new_idx = sys_next_map[base]
                if new_idx is not None:
                    v = new_bdd.var(new_idx)
                    if val:
                        clause = new_bdd.AND(clause, v)
                    else:
                        clause = new_bdd.AND(clause, new_bdd.NOT(v))
            result = new_bdd.OR(result, clause)

        return result

    def _check_fair_cycle(self, mc: SymbolicModelChecker,
                          accepting: int, max_steps: int) -> Tuple[bool, int]:
        """
        Check for fair accepting cycles using Emerson-Lei nested fixpoint.

        Fair cycle exists iff: EG(true) intersected with accepting is reachable.

        More precisely, compute the set of states on a fair cycle:
        nu Z. (accepting AND EX(EU(true, Z)))
        = greatest fixpoint of states that are accepting and can reach Z again.
        """
        bdd = mc.ts.bdd
        total_iters = 0

        # Compute: nu Z. accepting AND EX(E[true U Z])
        z = bdd.TRUE  # Start with all states
        for i in range(max_steps):
            total_iters += 1
            # E[true U Z] = states that can reach Z
            eu_z = mc.EU(bdd.TRUE, z, max_steps)
            # EX(eu_z) = states with a successor in eu_z
            ex_eu = mc.EX(eu_z)
            # New Z = accepting states with successor path to accepting
            new_z = bdd.AND(accepting, ex_eu)

            if new_z == z:
                break
            z = new_z

        # Check if any initial state is in the fair cycle set or can reach it
        if z == bdd.FALSE:
            return False, total_iters

        # Can we reach z from initial states?
        reachable_to_z = mc.EU(bdd.TRUE, z, max_steps)
        init_can_reach = bdd.AND(mc.ts.init, reachable_to_z)

        return init_can_reach != bdd.FALSE, total_iters

    def _extract_counterexample(self, mc: SymbolicModelChecker,
                                 accepting: int, aut_info: dict,
                                 max_steps: int) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Extract a lasso-shaped counterexample: (prefix, cycle).
        prefix: path from initial state to accepting state
        cycle: path from accepting state back to itself
        """
        bdd = mc.ts.bdd
        sys_vars = aut_info['sys_vars']

        # Find reachable accepting states
        reached = mc.ts.init
        prefix_states = [reached]

        for step in range(max_steps):
            acc_reached = bdd.AND(reached, accepting)
            if acc_reached != bdd.FALSE:
                break
            img = mc.image(reached)
            reached = bdd.OR(reached, img)
            prefix_states.append(reached)

        # Extract prefix trace (system variables only)
        prefix = []
        state = mc.ts.init
        for step in range(min(len(prefix_states), max_steps)):
            assignment = bdd.any_sat(state) if state != bdd.FALSE else {}
            sys_assignment = {}
            for v in sys_vars:
                idx = bdd.var_index(v)
                sys_assignment[v] = assignment.get(idx, False)
            prefix.append(sys_assignment)

            acc_check = bdd.AND(state, accepting)
            if acc_check != bdd.FALSE:
                break

            img = mc.image(state)
            state = img

        # For cycle, just report a single-state self-loop indication
        cycle = [prefix[-1]] if prefix else [{}]

        return (prefix, cycle)


# ============================================================
# Fairness Constraints
# ============================================================

@dataclass
class FairnessConstraint:
    """A fairness constraint: if 'assumption' holds infinitely often,
    then 'guarantee' must hold infinitely often."""
    assumption: Optional[int] = None   # BDD (None = unconditional/justice)
    guarantee: int = 0                 # BDD


class FairModelChecker:
    """
    Model checker with fairness constraints.

    Supports:
    - Justice (weak fairness): GF(p) -- p must hold infinitely often
    - Compassion (strong fairness): GF(p) -> GF(q) -- if p inf often, then q inf often
    """

    def __init__(self, ts: BooleanTS):
        self.ts = ts
        self.bdd = ts.bdd
        self.mc = SymbolicModelChecker(ts)
        self.justice: List[int] = []      # List of BDDs (must visit each inf often)
        self.compassion: List[Tuple[int, int]] = []  # (assumption, guarantee) BDD pairs

    def add_justice(self, condition: int):
        """Add justice constraint: GF(condition)."""
        self.justice.append(condition)

    def add_compassion(self, assumption: int, guarantee: int):
        """Add compassion constraint: GF(assumption) -> GF(guarantee)."""
        self.compassion.append((assumption, guarantee))

    def check_fair_ag(self, prop: int, max_steps: int = 500) -> LTLResult:
        """
        Check AG(prop) under fairness constraints.

        A state satisfies AG(prop) fairly if every fair path from it satisfies G(prop).
        """
        bdd = self.bdd

        if not self.justice and not self.compassion:
            # No fairness - use standard AG
            output = self.mc.check_safety(prop, max_steps)
            return LTLResult(
                holds=(output.result == MCResult.SAFE),
                fixpoint_iterations=output.fixpoint_iterations,
                method="ag_no_fairness"
            )

        # Fair EG(!prop) = states with a fair path staying in !prop
        not_prop = bdd.NOT(prop)
        fair_bad = self._fair_eg(not_prop, max_steps)

        # Check if any initial state can reach fair_bad
        reachable = self.mc.EU(bdd.TRUE, fair_bad, max_steps)
        init_bad = bdd.AND(self.ts.init, reachable)

        return LTLResult(
            holds=(init_bad == bdd.FALSE),
            method="fair_ag"
        )

    def check_fair_ef(self, target: int, max_steps: int = 500) -> LTLResult:
        """Check EF(target) under fairness: can we fairly reach target?"""
        bdd = self.bdd

        # Fair EF = can reach target on a fair path
        # EF(target) = E[true U target] (unaffected by fairness for reachability)
        # But for AF(target) we'd need fairness
        eu = self.mc.EU(bdd.TRUE, target, max_steps)
        init_reach = bdd.AND(self.ts.init, eu)

        return LTLResult(
            holds=(init_reach != bdd.FALSE),
            method="fair_ef"
        )

    def check_fair_af(self, target: int, max_steps: int = 500) -> LTLResult:
        """
        Check AF(target) under fairness: all fair paths eventually reach target.
        AF(target) = !EG(!target) under fairness.
        """
        bdd = self.bdd
        not_target = bdd.NOT(target)

        # Fair EG(!target) = states with a fair path always avoiding target
        fair_avoid = self._fair_eg(not_target, max_steps)

        # If initial state is in fair_avoid, then there exists a fair path
        # that never reaches target -> AF(target) fails
        init_bad = bdd.AND(self.ts.init, fair_avoid)

        return LTLResult(
            holds=(init_bad == bdd.FALSE),
            method="fair_af"
        )

    def _fair_eg(self, phi: int, max_steps: int = 500) -> int:
        """
        Compute EG(phi) under fairness constraints using Emerson-Lei.

        For justice {J_1, ..., J_n} and compassion {(P_1,Q_1), ..., (P_m,Q_m)}:

        fair_EG(phi) = nu Z. phi AND
            AND_i EX(E[phi U (phi AND J_i AND Z)]) AND
            AND_j EX(E[phi U (phi AND (NOT P_j OR Q_j) AND Z)])
        """
        bdd = self.bdd

        # Build acceptance conditions
        acceptance_sets = []

        # Justice: must visit each J_i
        for j in self.justice:
            acceptance_sets.append(j)

        # Compassion: if P_j visited, must visit Q_j
        # Encoded as: NOT P_j OR Q_j (either don't enable, or do take)
        for p, q in self.compassion:
            acceptance_sets.append(bdd.OR(bdd.NOT(p), q))

        if not acceptance_sets:
            return self.mc.EG(phi, max_steps)

        # Nested fixpoint
        z = phi
        for iteration in range(max_steps):
            new_z = phi
            for acc in acceptance_sets:
                target = bdd.and_all([phi, acc, z])
                eu = self.mc.EU(phi, target, max_steps)
                ex_eu = self.mc.EX(eu)
                new_z = bdd.AND(new_z, ex_eu)

            if new_z == z:
                break
            z = new_z

        return z


# ============================================================
# High-Level API
# ============================================================

def check_ltl(state_vars: List[str],
              init_fn, trans_fn, formula: LTL,
              max_steps: int = 500) -> LTLResult:
    """
    Check an LTL property on a boolean system.

    Args:
        state_vars: List of state variable names
        init_fn: Lambda(bdd) -> BDD for initial states
        trans_fn: Lambda(bdd, current_vars, next_vars) -> BDD for transitions
        formula: LTL formula to check
        max_steps: Maximum fixpoint iterations

    Returns:
        LTLResult with holds=True/False and optional counterexample
    """
    bdd, ts = _build_ts(state_vars, init_fn, trans_fn)
    checker = LTLModelChecker(ts)
    return checker.check(formula, max_steps)


def check_ltl_fair(state_vars: List[str],
                   init_fn, trans_fn, formula: LTL,
                   justice: List = None,
                   compassion: List = None,
                   max_steps: int = 500) -> LTLResult:
    """
    Check an LTL property with fairness constraints.

    Currently supports checking G(prop) and F(prop) formulas with fairness.

    Args:
        state_vars, init_fn, trans_fn: System description
        formula: LTL formula (G(atom) or F(atom) for fairness-aware checking)
        justice: List of lambda(bdd) -> BDD for justice constraints
        compassion: List of (lambda_p, lambda_q) for compassion constraints
        max_steps: Maximum iterations

    Returns:
        LTLResult
    """
    bdd, ts = _build_ts(state_vars, init_fn, trans_fn)

    if justice is None and compassion is None:
        # No fairness, use standard LTL
        checker = LTLModelChecker(ts)
        return checker.check(formula, max_steps)

    fair_mc = FairModelChecker(ts)

    if justice:
        for j_fn in justice:
            fair_mc.add_justice(j_fn(bdd))

    if compassion:
        for p_fn, q_fn in compassion:
            fair_mc.add_compassion(p_fn(bdd), q_fn(bdd))

    # Dispatch based on formula structure
    if formula.op == LTLOp.G:
        prop = _formula_to_bdd(bdd, formula.left, ts.var_indices)
        return fair_mc.check_fair_ag(prop, max_steps)
    elif formula.op == LTLOp.F:
        target = _formula_to_bdd(bdd, formula.left, ts.var_indices)
        return fair_mc.check_fair_af(target, max_steps)
    else:
        # For general LTL + fairness, use the full product approach
        checker = LTLModelChecker(ts)
        return checker.check(formula, max_steps)


def check_ltl_boolean(ts: BooleanTS, formula: LTL,
                      max_steps: int = 500) -> LTLResult:
    """Check LTL property on an existing BooleanTS."""
    checker = LTLModelChecker(ts)
    return checker.check(formula, max_steps)


def check_fair_cycle(ts: BooleanTS,
                     justice: List[int] = None,
                     compassion: List[Tuple[int, int]] = None,
                     max_steps: int = 500) -> LTLResult:
    """
    Check if a fair cycle exists in the system.

    Returns holds=True if a fair cycle exists (reachable from init).
    """
    fair_mc = FairModelChecker(ts)
    if justice:
        for j in justice:
            fair_mc.add_justice(j)
    if compassion:
        for p, q in compassion:
            fair_mc.add_compassion(p, q)

    bdd = ts.bdd
    fair_states = fair_mc._fair_eg(bdd.TRUE, max_steps)

    # Check if reachable
    mc = SymbolicModelChecker(ts)
    reachable = mc.EU(bdd.TRUE, fair_states, max_steps)
    init_reach = bdd.AND(ts.init, reachable)

    return LTLResult(
        holds=(init_reach != bdd.FALSE),
        method="fair_cycle"
    )


def compare_ltl_ctl(state_vars: List[str],
                    init_fn, trans_fn,
                    ltl_formula: LTL,
                    ctl_fn=None,
                    max_steps: int = 500) -> dict:
    """
    Compare LTL and CTL model checking on the same system.

    Args:
        state_vars, init_fn, trans_fn: System description
        ltl_formula: LTL property
        ctl_fn: Optional CTL checking function (lambda mc -> result)

    Returns:
        dict with ltl_result, ctl_result, comparison
    """
    bdd, ts = _build_ts(state_vars, init_fn, trans_fn)

    # LTL check
    ltl_checker = LTLModelChecker(ts)
    ltl_result = ltl_checker.check(ltl_formula, max_steps)

    result = {
        'ltl_result': ltl_result,
        'ltl_holds': ltl_result.holds,
        'automaton_states': ltl_result.automaton_states,
    }

    # CTL check if provided
    if ctl_fn:
        mc = SymbolicModelChecker(ts)
        ctl_result = ctl_fn(mc)
        result['ctl_result'] = ctl_result
        result['agree'] = ltl_result.holds == ctl_result

    return result


# ============================================================
# LTL Formula Parser
# ============================================================

def parse_ltl(text: str) -> LTL:
    """
    Parse an LTL formula from text.

    Syntax:
        atom ::= [a-zA-Z_][a-zA-Z0-9_]*
        formula ::= atom | 'true' | 'false'
                  | '!' formula
                  | 'X' formula | 'F' formula | 'G' formula
                  | formula '&' formula | formula '|' formula
                  | formula '->' formula | formula '<->' formula
                  | formula 'U' formula | formula 'R' formula | formula 'W' formula
                  | '(' formula ')'

    Precedence (low to high):
        <->, ->, |, &, U/R/W, !/X/F/G, atom
    """
    tokens = _tokenize_ltl(text)
    result, pos = _parse_iff(tokens, 0)
    return result


def _tokenize_ltl(text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in '()!&|':
            tokens.append(c)
            i += 1
        elif c == '-' and i + 1 < len(text) and text[i+1] == '>':
            tokens.append('->')
            i += 2
        elif c == '<' and i + 2 < len(text) and text[i+1:i+3] == '->':
            tokens.append('<->')
            i += 3
        elif c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1
    return tokens


def _parse_iff(tokens, pos):
    left, pos = _parse_implies(tokens, pos)
    while pos < len(tokens) and tokens[pos] == '<->':
        pos += 1
        right, pos = _parse_implies(tokens, pos)
        left = Iff(left, right)
    return left, pos


def _parse_implies(tokens, pos):
    left, pos = _parse_or(tokens, pos)
    if pos < len(tokens) and tokens[pos] == '->':
        pos += 1
        right, pos = _parse_implies(tokens, pos)  # right-associative
        left = Implies(left, right)
    return left, pos


def _parse_or(tokens, pos):
    left, pos = _parse_and(tokens, pos)
    while pos < len(tokens) and tokens[pos] == '|':
        pos += 1
        right, pos = _parse_and(tokens, pos)
        left = Or(left, right)
    return left, pos


def _parse_and(tokens, pos):
    left, pos = _parse_temporal(tokens, pos)
    while pos < len(tokens) and tokens[pos] == '&':
        pos += 1
        right, pos = _parse_temporal(tokens, pos)
        left = And(left, right)
    return left, pos


def _parse_temporal(tokens, pos):
    left, pos = _parse_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ('U', 'R', 'W'):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_unary(tokens, pos)
        if op == 'U':
            left = Until(left, right)
        elif op == 'R':
            left = Release(left, right)
        elif op == 'W':
            left = WeakUntil(left, right)
    return left, pos


def _parse_unary(tokens, pos):
    if pos >= len(tokens):
        return LTLTrue(), pos

    t = tokens[pos]
    if t == '!':
        inner, pos = _parse_unary(tokens, pos + 1)
        return Not(inner), pos
    if t == 'X':
        inner, pos = _parse_unary(tokens, pos + 1)
        return Next(inner), pos
    if t == 'F':
        inner, pos = _parse_unary(tokens, pos + 1)
        return Finally(inner), pos
    if t == 'G':
        inner, pos = _parse_unary(tokens, pos + 1)
        return Globally(inner), pos

    return _parse_atom(tokens, pos)


def _parse_atom(tokens, pos):
    if pos >= len(tokens):
        return LTLTrue(), pos

    t = tokens[pos]
    if t == '(':
        inner, pos = _parse_iff(tokens, pos + 1)
        if pos < len(tokens) and tokens[pos] == ')':
            pos += 1
        return inner, pos
    if t == 'true':
        return LTLTrue(), pos + 1
    if t == 'false':
        return LTLFalse(), pos + 1
    # Must be a variable name
    return Atom(t), pos + 1


# ============================================================
# Internal Helpers
# ============================================================

def _build_ts(state_vars, init_fn, trans_fn):
    """Build BooleanTS from lambdas."""
    bdd = BDD()
    for v in state_vars:
        bdd.named_var(v)
    for v in state_vars:
        bdd.named_var(f"{v}'")

    current_vars = {v: bdd.var_index(v) for v in state_vars}
    next_var_names = {f"{v}'": bdd.var_index(f"{v}'") for v in state_vars}

    init = init_fn(bdd)
    trans = trans_fn(bdd, current_vars, next_var_names)

    # next_indices keyed by state var names (not primed), per SymbolicModelChecker convention
    next_indices = {v: bdd.var_index(f"{v}'") for v in state_vars}

    ts = BooleanTS(
        bdd=bdd,
        state_vars=state_vars,
        next_vars=[f"{v}'" for v in state_vars],
        init=init,
        trans=trans,
        var_indices=current_vars,
        next_indices=next_indices
    )
    return bdd, ts


def _formula_to_bdd(bdd: BDD, formula: LTL, var_indices: Dict[str, int]) -> int:
    """Convert a propositional (non-temporal) LTL formula to a BDD."""
    if formula.op == LTLOp.TRUE:
        return bdd.TRUE
    if formula.op == LTLOp.FALSE:
        return bdd.FALSE
    if formula.op == LTLOp.ATOM:
        if formula.name in var_indices:
            return bdd.var(var_indices[formula.name])
        return bdd.FALSE
    if formula.op == LTLOp.NOT:
        return bdd.NOT(_formula_to_bdd(bdd, formula.left, var_indices))
    if formula.op == LTLOp.AND:
        return bdd.AND(
            _formula_to_bdd(bdd, formula.left, var_indices),
            _formula_to_bdd(bdd, formula.right, var_indices)
        )
    if formula.op == LTLOp.OR:
        return bdd.OR(
            _formula_to_bdd(bdd, formula.left, var_indices),
            _formula_to_bdd(bdd, formula.right, var_indices)
        )
    return bdd.TRUE
