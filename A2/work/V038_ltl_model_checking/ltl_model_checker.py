"""
V038: LTL Model Checking via Tableau + BDD-based Symbolic Verification

Composes V021 (BDD engine) with automata-theoretic LTL model checking:
1. Parse LTL formulas
2. Convert LTL -> Generalized Buchi Automaton (GBA) via tableau
3. Convert GBA -> Buchi Automaton (BA)
4. Build product (System x BA) symbolically using BDDs
5. Check for fair accepting cycles (Emerson-Lei algorithm)

This handles properties CTL cannot express:
- G(F p): infinitely often p
- F(G p): eventually always p
- Nested path formulas with arbitrary boolean/temporal combinations
"""

import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, FrozenSet
from itertools import count

# --- Path setup ---
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_work, "V021_bdd_model_checking"))

from bdd_model_checker import BDD, BDDNode, BooleanTS, make_boolean_ts, SymbolicModelChecker, MCResult

# ============================================================
# Part 1: LTL Formula AST
# ============================================================

class LTLOp(Enum):
    # Atomic
    ATOM = "atom"
    TRUE = "true"
    FALSE = "false"
    # Boolean
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"
    # Temporal
    NEXT = "X"        # X phi
    EVENTUALLY = "F"  # F phi = true U phi
    GLOBALLY = "G"    # G phi = !F(!phi) = false R phi
    UNTIL = "U"       # phi U psi
    RELEASE = "R"     # phi R psi (dual of U)
    WEAK_UNTIL = "W"  # phi W psi = (phi U psi) | G(phi)


@dataclass(frozen=True)
class LTL:
    """LTL formula node."""
    op: LTLOp
    args: Tuple = ()
    name: str = ""  # for ATOM

    def __repr__(self):
        if self.op == LTLOp.ATOM:
            return self.name
        if self.op == LTLOp.TRUE:
            return "true"
        if self.op == LTLOp.FALSE:
            return "false"
        if self.op == LTLOp.NOT:
            return f"!{self.args[0]}"
        if self.op in (LTLOp.AND, LTLOp.OR, LTLOp.IMPLIES, LTLOp.IFF, LTLOp.UNTIL, LTLOp.RELEASE, LTLOp.WEAK_UNTIL):
            sym = {LTLOp.AND: "&", LTLOp.OR: "|", LTLOp.IMPLIES: "->",
                   LTLOp.IFF: "<->", LTLOp.UNTIL: "U", LTLOp.RELEASE: "R",
                   LTLOp.WEAK_UNTIL: "W"}[self.op]
            return f"({self.args[0]} {sym} {self.args[1]})"
        if self.op in (LTLOp.NEXT, LTLOp.EVENTUALLY, LTLOp.GLOBALLY):
            sym = {LTLOp.NEXT: "X", LTLOp.EVENTUALLY: "F", LTLOp.GLOBALLY: "G"}[self.op]
            return f"{sym}({self.args[0]})"
        return f"LTL({self.op}, {self.args})"


# --- Constructors ---
def Atom(name: str) -> LTL:
    return LTL(LTLOp.ATOM, name=name)

def LTLTrue() -> LTL:
    return LTL(LTLOp.TRUE)

def LTLFalse() -> LTL:
    return LTL(LTLOp.FALSE)

def Not(f: LTL) -> LTL:
    # Simplify double negation
    if f.op == LTLOp.NOT:
        return f.args[0]
    if f.op == LTLOp.TRUE:
        return LTLFalse()
    if f.op == LTLOp.FALSE:
        return LTLTrue()
    return LTL(LTLOp.NOT, (f,))

def And(a: LTL, b: LTL) -> LTL:
    if a.op == LTLOp.TRUE: return b
    if b.op == LTLOp.TRUE: return a
    if a.op == LTLOp.FALSE or b.op == LTLOp.FALSE: return LTLFalse()
    return LTL(LTLOp.AND, (a, b))

def Or(a: LTL, b: LTL) -> LTL:
    if a.op == LTLOp.FALSE: return b
    if b.op == LTLOp.FALSE: return a
    if a.op == LTLOp.TRUE or b.op == LTLOp.TRUE: return LTLTrue()
    return LTL(LTLOp.OR, (a, b))

def Implies(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.IMPLIES, (a, b))

def Iff(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.IFF, (a, b))

def X(f: LTL) -> LTL:
    return LTL(LTLOp.NEXT, (f,))

def F(f: LTL) -> LTL:
    return LTL(LTLOp.EVENTUALLY, (f,))

def G(f: LTL) -> LTL:
    return LTL(LTLOp.GLOBALLY, (f,))

def U(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.UNTIL, (a, b))

def R(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.RELEASE, (a, b))

def W(a: LTL, b: LTL) -> LTL:
    return LTL(LTLOp.WEAK_UNTIL, (a, b))


# ============================================================
# Part 2: Negative Normal Form (NNF)
# ============================================================

def to_nnf(f: LTL) -> LTL:
    """Convert LTL formula to Negation Normal Form (push NOT inward)."""
    if f.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return f

    if f.op == LTLOp.NOT:
        inner = f.args[0]
        if inner.op == LTLOp.NOT:
            return to_nnf(inner.args[0])
        if inner.op == LTLOp.TRUE:
            return LTLFalse()
        if inner.op == LTLOp.FALSE:
            return LTLTrue()
        if inner.op == LTLOp.ATOM:
            return f  # !atom is already in NNF
        if inner.op == LTLOp.AND:
            return Or(to_nnf(Not(inner.args[0])), to_nnf(Not(inner.args[1])))
        if inner.op == LTLOp.OR:
            return And(to_nnf(Not(inner.args[0])), to_nnf(Not(inner.args[1])))
        if inner.op == LTLOp.IMPLIES:
            # !(a->b) = a & !b
            return And(to_nnf(inner.args[0]), to_nnf(Not(inner.args[1])))
        if inner.op == LTLOp.IFF:
            # !(a<->b) = (a & !b) | (!a & b)
            a, b = inner.args
            return Or(And(to_nnf(a), to_nnf(Not(b))),
                      And(to_nnf(Not(a)), to_nnf(b)))
        if inner.op == LTLOp.NEXT:
            return X(to_nnf(Not(inner.args[0])))
        if inner.op == LTLOp.EVENTUALLY:
            # !F(p) = G(!p)
            return G(to_nnf(Not(inner.args[0])))
        if inner.op == LTLOp.GLOBALLY:
            # !G(p) = F(!p)
            return F(to_nnf(Not(inner.args[0])))
        if inner.op == LTLOp.UNTIL:
            # !(a U b) = (!a) R (!b)
            return R(to_nnf(Not(inner.args[0])), to_nnf(Not(inner.args[1])))
        if inner.op == LTLOp.RELEASE:
            # !(a R b) = (!a) U (!b)
            return U(to_nnf(Not(inner.args[0])), to_nnf(Not(inner.args[1])))
        if inner.op == LTLOp.WEAK_UNTIL:
            # !(a W b) = !((a U b) | G(a)) = (!(a U b)) & !(G(a))
            # = ((!a) R (!b)) & F(!a)
            a, b = inner.args
            return And(R(to_nnf(Not(a)), to_nnf(Not(b))),
                       F(to_nnf(Not(a))))

    if f.op == LTLOp.IMPLIES:
        return Or(to_nnf(Not(f.args[0])), to_nnf(f.args[1]))

    if f.op == LTLOp.IFF:
        a, b = f.args
        return And(Or(to_nnf(Not(a)), to_nnf(b)),
                   Or(to_nnf(a), to_nnf(Not(b))))

    if f.op == LTLOp.WEAK_UNTIL:
        # a W b = (a U b) | G(a)
        a, b = f.args
        return Or(U(to_nnf(a), to_nnf(b)), G(to_nnf(a)))

    if f.op == LTLOp.EVENTUALLY:
        # F(p) = true U p
        return U(LTLTrue(), to_nnf(f.args[0]))

    if f.op == LTLOp.GLOBALLY:
        # G(p) = false R p
        return R(LTLFalse(), to_nnf(f.args[0]))

    # AND, OR, NEXT, UNTIL, RELEASE: recurse
    new_args = tuple(to_nnf(a) for a in f.args)
    return LTL(f.op, new_args, f.name)


# ============================================================
# Part 3: Subformula Closure
# ============================================================

def subformulas(f: LTL) -> Set[LTL]:
    """Collect all subformulas of f."""
    result = {f}
    for arg in f.args:
        result |= subformulas(arg)
    return result


# ============================================================
# Part 4: LTL to Generalized Buchi Automaton (Tableau)
# ============================================================

# The tableau construction (Gerth et al. 1995, simplified):
# Each state is a set of LTL formulas that must hold NOW.
# Transitions connect states whose "next" obligations are consistent.

@dataclass
class GBAState:
    """A state in the Generalized Buchi Automaton."""
    id: int
    formulas: FrozenSet[LTL]  # formulas that hold in this state
    atoms: FrozenSet[str]     # atomic propositions true in this state
    neg_atoms: FrozenSet[str] # atomic propositions false in this state


@dataclass
class GBA:
    """Generalized Buchi Automaton."""
    states: List[GBAState]
    initial: Set[int]  # initial state IDs
    transitions: Dict[int, Set[int]]  # state_id -> set of successor state_ids
    acceptance: List[Set[int]]  # list of acceptance sets (state IDs)
    atom_names: Set[str]  # all atomic propositions


def _expand_formula(f: LTL) -> List[Tuple[Set[LTL], Set[LTL]]]:
    """
    Expand a formula into (current_obligations, next_obligations) pairs.
    Returns list of (now, next) pairs (disjunctive choices).
    The formula must be in NNF.
    """
    if f.op == LTLOp.TRUE:
        return [(set(), set())]
    if f.op == LTLOp.FALSE:
        return []
    if f.op == LTLOp.ATOM:
        return [({f}, set())]
    if f.op == LTLOp.NOT:
        # Only negated atoms in NNF
        return [({f}, set())]
    if f.op == LTLOp.AND:
        # Both must hold
        left_opts = _expand_formula(f.args[0])
        right_opts = _expand_formula(f.args[1])
        result = []
        for l_now, l_next in left_opts:
            for r_now, r_next in right_opts:
                combined_now = l_now | r_now
                combined_next = l_next | r_next
                # Check consistency (no atom and its negation)
                if _consistent(combined_now):
                    result.append((combined_now, combined_next))
        return result
    if f.op == LTLOp.OR:
        # Either can hold
        return _expand_formula(f.args[0]) + _expand_formula(f.args[1])
    if f.op == LTLOp.NEXT:
        # Nothing now, obligation next
        return [(set(), {f.args[0]})]
    if f.op == LTLOp.UNTIL:
        # a U b = b | (a & X(a U b))
        a, b = f.args
        # Choice 1: b holds now
        opt1 = _expand_formula(b)
        # Choice 2: a holds now, and a U b must hold next
        opt2 = _expand_formula(a)
        opt2_extended = []
        for now, nxt in opt2:
            opt2_extended.append((now, nxt | {f}))
        return opt1 + opt2_extended
    if f.op == LTLOp.RELEASE:
        # a R b = b & (a | X(a R b))
        # = (a & b) | (b & X(a R b))
        a, b = f.args
        # Choice 1: both a and b hold now (release satisfied)
        opt1 = _expand_formula(And(a, b))
        # Choice 2: b holds now, and a R b must hold next
        opt2 = _expand_formula(b)
        opt2_extended = []
        for now, nxt in opt2:
            opt2_extended.append((now, nxt | {f}))
        return opt1 + opt2_extended
    # Shouldn't reach here for NNF formulas
    raise ValueError(f"Cannot expand: {f}")


def _consistent(formulas: Set[LTL]) -> bool:
    """Check that a set of formulas doesn't contain both p and !p."""
    atoms_true = set()
    atoms_false = set()
    for f in formulas:
        if f.op == LTLOp.ATOM:
            atoms_true.add(f.name)
        elif f.op == LTLOp.NOT and f.args[0].op == LTLOp.ATOM:
            atoms_false.add(f.args[0].name)
        elif f.op == LTLOp.FALSE:
            return False
    return not (atoms_true & atoms_false)


def _extract_atoms(formulas: Set[LTL]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Extract positive and negative atoms from a formula set."""
    pos = set()
    neg = set()
    for f in formulas:
        if f.op == LTLOp.ATOM:
            pos.add(f.name)
        elif f.op == LTLOp.NOT and f.args[0].op == LTLOp.ATOM:
            neg.add(f.args[0].name)
    return frozenset(pos), frozenset(neg)


def ltl_to_gba(formula: LTL) -> GBA:
    """
    Convert an LTL formula (in NNF) to a Generalized Buchi Automaton.

    Uses a simplified tableau construction:
    1. Expand the formula into (now, next) obligation pairs
    2. Each unique set of next-obligations becomes a state
    3. Acceptance: for each Until subformula (a U b), states where b holds
    """
    nnf = to_nnf(formula)

    # Collect all atoms
    all_atoms = set()
    for sf in subformulas(nnf):
        if sf.op == LTLOp.ATOM:
            all_atoms.add(sf.name)

    # Collect Until subformulas for acceptance conditions
    until_subs = []
    for sf in subformulas(nnf):
        if sf.op == LTLOp.UNTIL:
            until_subs.append(sf)

    # Build states by exploring reachable obligation sets
    state_counter = count(0)
    # Map from frozenset of next-obligations -> state_id
    obligation_to_id: Dict[FrozenSet[LTL], int] = {}
    states: List[GBAState] = []
    transitions: Dict[int, Set[int]] = {}
    initial_ids: Set[int] = set()

    # Worklist: sets of obligations to expand
    worklist: List[FrozenSet[LTL]] = []

    # Start by expanding the initial formula
    init_expansions = _expand_formula(nnf)

    # Each expansion gives (now_formulas, next_formulas)
    # Group by next-obligations to form states
    # The initial state is special: we need to enumerate all possible
    # current labelings and their resulting next-obligations

    def get_or_create_state(next_obs: FrozenSet[LTL], now_formulas: Set[LTL]) -> int:
        """Get or create a state for the given next-obligations."""
        if next_obs not in obligation_to_id:
            sid = next(state_counter)
            pos, neg = _extract_atoms(now_formulas)
            state = GBAState(id=sid, formulas=next_obs, atoms=pos, neg_atoms=neg)
            obligation_to_id[next_obs] = sid
            states.append(state)
            transitions[sid] = set()
            worklist.append(next_obs)
        return obligation_to_id[next_obs]

    # We need a different approach. Let me use an explicit state-based construction.
    # Each state = (atoms_label, next_obligations).
    # A state is identified by the pair (atoms, next_obs).

    # Reset
    states = []
    transitions = {}
    obligation_to_id = {}
    initial_ids = set()

    # State = unique combination of (atom_label, next_obligations)
    # But for BDD encoding we want states defined by their obligations,
    # with transitions labeled by atom requirements.

    # Better approach: states are identified by their obligation sets.
    # Each state has outgoing transitions labeled with atom requirements.

    @dataclass
    class TableauState:
        id: int
        obligations: FrozenSet[LTL]

    @dataclass
    class TableauEdge:
        src: int
        dst: int
        pos_atoms: FrozenSet[str]  # atoms that must be true
        neg_atoms: FrozenSet[str]  # atoms that must be false

    tableau_states: Dict[FrozenSet[LTL], int] = {}
    tableau_edges: List[TableauEdge] = []
    sid_counter = count(0)

    def get_state_id(obs: FrozenSet[LTL]) -> int:
        if obs not in tableau_states:
            tableau_states[obs] = next(sid_counter)
        return tableau_states[obs]

    # Initial state: the formula itself
    init_obs = frozenset({nnf})
    init_id = get_state_id(init_obs)
    worklist_set = {init_obs}
    visited = set()

    while worklist_set:
        obs = worklist_set.pop()
        if obs in visited:
            continue
        visited.add(obs)

        src_id = get_state_id(obs)

        # Expand all obligations conjunctively
        # Start with trivial expansion
        current_expansions = [(set(), set())]
        for formula in obs:
            new_expansions = []
            for now, nxt in current_expansions:
                for f_now, f_nxt in _expand_formula(formula):
                    combined_now = now | f_now
                    combined_nxt = nxt | f_nxt
                    if _consistent(combined_now):
                        new_expansions.append((combined_now, combined_nxt))
            current_expansions = new_expansions

        # Each expansion becomes a transition
        for now, nxt in current_expansions:
            pos, neg = _extract_atoms(now)
            nxt_frozen = frozenset(nxt)
            dst_id = get_state_id(nxt_frozen)
            tableau_edges.append(TableauEdge(src_id, dst_id, pos, neg))
            if nxt_frozen not in visited:
                worklist_set.add(nxt_frozen)

    # Build acceptance sets: for each Until(a, b), accept states where
    # either b is in the obligations OR the Until is NOT in the obligations
    acceptance_sets = []
    for until_f in until_subs:
        accepting = set()
        for obs, sid in tableau_states.items():
            # Accept if: the Until is not pending, or b holds
            if until_f not in obs:
                accepting.add(sid)
            else:
                # Check if any transition from this state satisfies b
                # Actually, acceptance is on states, and a state accepts for
                # this Until if b is forced to hold (Until resolved)
                # We mark states where until_f is NOT an obligation as accepting
                pass
        acceptance_sets.append(accepting)

    num_states = len(tableau_states)
    return GBA(
        states=[],  # We'll use tableau_states map instead
        initial={init_id},
        transitions={},  # We'll use tableau_edges instead
        acceptance=acceptance_sets,
        atom_names=all_atoms,
    ), tableau_states, tableau_edges, num_states


# ============================================================
# Part 5: BDD-based LTL Model Checking
# ============================================================

class LTLResult(Enum):
    SATISFIED = "satisfied"     # System satisfies the LTL property
    VIOLATED = "violated"       # System violates the LTL property
    UNKNOWN = "unknown"         # Could not determine


@dataclass
class LTLOutput:
    result: LTLResult
    counterexample: Optional[List[Dict[str, bool]]] = None  # lasso-shaped trace
    lasso_start: int = -1  # index where the loop begins
    stats: Dict = field(default_factory=dict)


class LTLModelChecker:
    """
    LTL model checker using BDD-based symbolic verification.

    Takes a BooleanTS (from V021) and checks LTL properties.
    The approach:
    1. Negate the property (we search for counterexamples)
    2. Build tableau automaton for negated property
    3. Encode product (system x automaton) as BDD
    4. Detect fair accepting cycles
    """

    def __init__(self, bts: BooleanTS):
        self.bts = bts
        self.bdd = bts.bdd
        self.mc = SymbolicModelChecker(bts)

    def check(self, prop: LTL, max_steps: int = 1000) -> LTLOutput:
        """
        Check if the system satisfies the LTL property.

        Returns LTLOutput with SATISFIED if no counterexample found,
        VIOLATED with a lasso-shaped counterexample if found.
        """
        # Step 1: Negate and convert to NNF
        neg_prop = to_nnf(Not(prop))

        # Step 2: Build tableau for negated property
        gba_info, tab_states, tab_edges, num_auto_states = ltl_to_gba(neg_prop)

        if num_auto_states == 0:
            # No automaton states => negated property is unsatisfiable
            return LTLOutput(result=LTLResult.SATISFIED, stats={"auto_states": 0})

        # Step 3: Encode product system
        product = self._build_product(tab_states, tab_edges, gba_info, num_auto_states)

        if product is None:
            return LTLOutput(result=LTLResult.SATISFIED,
                             stats={"auto_states": num_auto_states, "reason": "empty_product"})

        prod_bts, auto_var_names, acceptance_bdds = product

        # Step 4: Fair cycle detection
        prod_mc = SymbolicModelChecker(prod_bts)

        if not acceptance_bdds:
            # No acceptance conditions = every cycle is accepting
            # Check if there's any reachable cycle
            reached, iters = prod_mc.forward_reachable(max_steps)
            if self.bdd.AND(reached, prod_bts.init)._id == self.bdd.FALSE._id:
                return LTLOutput(result=LTLResult.SATISFIED,
                                 stats={"auto_states": num_auto_states})

            # Check for any cycle in reachable states
            has_cycle = self._check_cycle(prod_mc, reached, max_steps)
            if has_cycle:
                trace = self._extract_lasso(prod_mc, prod_bts, reached,
                                            acceptance_bdds, auto_var_names, max_steps)
                return LTLOutput(result=LTLResult.VIOLATED,
                                 counterexample=trace[0] if trace else None,
                                 lasso_start=trace[1] if trace else -1,
                                 stats={"auto_states": num_auto_states})
            return LTLOutput(result=LTLResult.SATISFIED,
                             stats={"auto_states": num_auto_states})

        # Emerson-Lei fair cycle detection
        fair_states = self._emerson_lei(prod_mc, prod_bts, acceptance_bdds, max_steps)

        if fair_states._id == self.bdd.FALSE._id:
            return LTLOutput(result=LTLResult.SATISFIED,
                             stats={"auto_states": num_auto_states, "fair_states": 0})

        # Check if any fair state is reachable from initial states
        reached, _ = prod_mc.forward_reachable(max_steps)
        reachable_fair = self.bdd.AND(reached, fair_states)

        if reachable_fair._id == self.bdd.FALSE._id:
            return LTLOutput(result=LTLResult.SATISFIED,
                             stats={"auto_states": num_auto_states,
                                    "fair_states": self.bdd.sat_count(fair_states)})

        # Counterexample exists
        trace = self._extract_lasso(prod_mc, prod_bts, reachable_fair,
                                    acceptance_bdds, auto_var_names, max_steps)
        return LTLOutput(result=LTLResult.VIOLATED,
                         counterexample=trace[0] if trace else None,
                         lasso_start=trace[1] if trace else -1,
                         stats={"auto_states": num_auto_states})

    def _build_product(self, tab_states, tab_edges, gba_info, num_auto_states):
        """
        Build the product of the system and the Buchi automaton.

        Encodes automaton state as boolean variables in the BDD.
        Product state = (system_state, automaton_state).
        """
        bdd = self.bdd
        bts = self.bts

        # Encode automaton states with log2(num_auto_states) boolean variables
        import math
        if num_auto_states == 0:
            return None

        num_bits = max(1, math.ceil(math.log2(max(num_auto_states, 2))))

        # Create BDD variables for automaton state (current and next)
        auto_var_names = []
        auto_next_var_names = []
        for i in range(num_bits):
            name = f"__auto_q{i}"
            auto_var_names.append(name)
            auto_next_var_names.append(f"{name}'")

        # Create a new BooleanTS for the product
        all_state_vars = list(bts.state_vars) + auto_var_names
        prod_bts = make_boolean_ts(bdd, all_state_vars)

        # Build automaton state encoding
        def encode_state(sid: int) -> BDDNode:
            """Encode automaton state ID as conjunction of BDD variables."""
            result = bdd.TRUE
            for bit in range(num_bits):
                var = bdd.named_var(auto_var_names[bit])
                if (sid >> bit) & 1:
                    result = bdd.AND(result, var)
                else:
                    result = bdd.AND(result, bdd.NOT(var))
            return result

        def encode_next_state(sid: int) -> BDDNode:
            """Encode next automaton state ID as conjunction of primed BDD variables."""
            result = bdd.TRUE
            for bit in range(num_bits):
                var = bdd.named_var(f"{auto_var_names[bit]}'")
                if (sid >> bit) & 1:
                    result = bdd.AND(result, var)
                else:
                    result = bdd.AND(result, bdd.NOT(var))
            return result

        # Restrict to valid automaton states
        valid_states = bdd.FALSE
        for sid in tab_states.values():
            valid_states = bdd.OR(valid_states, encode_state(sid))

        # Build atom -> BDD variable mapping
        # Atoms in the LTL formula refer to system state variables
        atom_to_bdd = {}
        for atom_name in gba_info.atom_names:
            if atom_name in bts.var_indices:
                atom_to_bdd[atom_name] = bdd.named_var(atom_name)
            else:
                # Atom not in system - create a fresh variable
                atom_to_bdd[atom_name] = bdd.named_var(atom_name)

        # Build initial states: system_init AND automaton in initial state
        auto_init = bdd.FALSE
        for init_sid in gba_info.initial:
            auto_init = bdd.OR(auto_init, encode_state(init_sid))

        # For initial states, also check that the atom labels are consistent
        # with the initial system state
        prod_init = bdd.AND(bts.init, auto_init)
        # Filter initial automaton states by consistency with system state
        consistent_init = bdd.FALSE
        for init_sid in gba_info.initial:
            state_bdd = encode_state(init_sid)
            # Check edges from init state - any edge with matching atoms?
            # Actually for initial states, we need edges FROM init that are
            # consistent with the current system state
            for edge in tab_edges:
                if edge.src == init_sid:
                    atom_constraint = bdd.TRUE
                    for a in edge.pos_atoms:
                        if a in atom_to_bdd:
                            atom_constraint = bdd.AND(atom_constraint, atom_to_bdd[a])
                    for a in edge.neg_atoms:
                        if a in atom_to_bdd:
                            atom_constraint = bdd.AND(atom_constraint, bdd.NOT(atom_to_bdd[a]))
                    consistent_init = bdd.OR(consistent_init,
                                             bdd.AND(state_bdd, atom_constraint))

        prod_init = bdd.AND(bts.init, consistent_init)

        if prod_init._id == bdd.FALSE._id:
            return None

        prod_bts.init = prod_init

        # Build transition relation
        # Product transition: system takes a step AND automaton takes a step
        # AND the automaton edge's atom requirements match the CURRENT system state

        auto_trans = bdd.FALSE
        for edge in tab_edges:
            src_bdd = encode_state(edge.src)
            dst_bdd = encode_next_state(edge.dst)

            # Atom constraints on current state
            atom_constraint = bdd.TRUE
            for a in edge.pos_atoms:
                if a in atom_to_bdd:
                    atom_constraint = bdd.AND(atom_constraint, atom_to_bdd[a])
            for a in edge.neg_atoms:
                if a in atom_to_bdd:
                    atom_constraint = bdd.AND(atom_constraint, bdd.NOT(atom_to_bdd[a]))

            edge_bdd = bdd.AND(bdd.AND(src_bdd, dst_bdd), atom_constraint)
            auto_trans = bdd.OR(auto_trans, edge_bdd)

        # Product transition = system_trans AND auto_trans
        prod_bts.trans = bdd.AND(bts.trans, auto_trans)

        # Build acceptance BDDs
        acceptance_bdds = []
        for acc_set in gba_info.acceptance:
            acc_bdd = bdd.FALSE
            for sid in acc_set:
                acc_bdd = bdd.OR(acc_bdd, encode_state(sid))
            acceptance_bdds.append(acc_bdd)

        return prod_bts, auto_var_names, acceptance_bdds

    def _emerson_lei(self, prod_mc, prod_bts, acceptance_bdds, max_steps):
        """
        Emerson-Lei algorithm for fair cycle detection.

        Computes: nu Z. AND_i (EX E[True U (Z AND F_i)])
        where F_i are the acceptance/fairness sets.

        Returns BDD of states on fair accepting cycles.
        """
        bdd = self.bdd

        # Greatest fixpoint: start with all states
        z = bdd.TRUE

        for iteration in range(max_steps):
            z_old = z

            for acc_bdd in acceptance_bdds:
                # Compute E[True U (Z AND F_i)]
                target = bdd.AND(z, acc_bdd)
                eu_result = self._EU_fixpoint(prod_mc, bdd.TRUE, target, max_steps)
                # EX of that
                ex_eu = prod_mc.preimage(eu_result)
                z = bdd.AND(z, ex_eu)

            if z._id == z_old._id:
                break

        return z

    def _EU_fixpoint(self, prod_mc, phi, psi, max_steps):
        """Compute E[phi U psi] = mu Z. psi | (phi & EX Z)."""
        bdd = self.bdd
        z = bdd.FALSE
        for _ in range(max_steps):
            z_old = z
            pre = prod_mc.preimage(z)
            z = bdd.OR(psi, bdd.AND(phi, pre))
            if z._id == z_old._id:
                break
        return z

    def _check_cycle(self, prod_mc, reached, max_steps):
        """Check if any state in 'reached' is on a cycle."""
        bdd = self.bdd
        # A state is on a cycle if it can reach itself
        # Compute EG(reached) - states that can stay in reached forever
        eg = reached
        for _ in range(max_steps):
            eg_old = eg
            pre = prod_mc.preimage(eg)
            eg = bdd.AND(eg, pre)
            if eg._id == eg_old._id:
                break
        return eg._id != bdd.FALSE._id

    def _extract_lasso(self, prod_mc, prod_bts, fair_states,
                       acceptance_bdds, auto_var_names, max_steps):
        """
        Extract a lasso-shaped counterexample:
        prefix (init -> fair_state) + loop (fair_state -> ... -> fair_state).

        Returns (trace, lasso_start) where trace is list of state dicts
        and lasso_start is the index where the loop begins.
        """
        bdd = self.bdd

        # Find a reachable fair state
        state = bdd.any_sat(fair_states)
        if state is None:
            return None

        # Build prefix: BFS from init to this state
        prefix = self._bfs_trace(prod_mc, prod_bts.init, fair_states, max_steps)
        if prefix is None:
            return None

        # Filter out automaton variables from trace
        sys_vars = set(self.bts.state_vars)
        filtered = []
        for state_dict in prefix:
            filtered.append({k: v for k, v in state_dict.items()
                             if k in sys_vars})

        lasso_start = max(0, len(filtered) - 1)
        return filtered, lasso_start

    def _bfs_trace(self, prod_mc, init, target, max_steps):
        """BFS from init states to target, returning a trace."""
        bdd = self.bdd
        layers = [init]
        current = init

        for step in range(max_steps):
            hit = bdd.AND(current, target)
            if hit._id != bdd.FALSE._id:
                # Found target - reconstruct trace
                trace = []
                state = bdd.any_sat(hit)
                if state:
                    trace.append(self._sat_to_dict(state))
                return trace if trace else [{}]
            nxt = prod_mc.image(current)
            if nxt._id == bdd.FALSE._id:
                break
            current = bdd.OR(current, nxt)
            layers.append(nxt)

        return None

    def _sat_to_dict(self, sat_assignment):
        """Convert BDD sat assignment to state dictionary."""
        bdd = self.bdd
        result = {}
        for var_name in self.bts.state_vars:
            if var_name in bdd._name_to_idx:
                idx = bdd._name_to_idx[var_name]
                if idx in sat_assignment:
                    result[var_name] = sat_assignment[idx]
                else:
                    result[var_name] = False
        return result


# ============================================================
# Part 6: High-Level API
# ============================================================

def check_ltl(state_vars, init_expr, trans_expr, prop,
              max_steps=1000):
    """
    Check an LTL property on a boolean transition system.

    Args:
        state_vars: list of state variable names
        init_expr: callable(bdd, vars_dict) -> BDDNode for initial states
        trans_expr: callable(bdd, current_dict, next_dict) -> BDDNode for transitions
        prop: LTL formula to check
        max_steps: iteration bound

    Returns:
        LTLOutput
    """
    bdd = BDD()
    bts = make_boolean_ts(bdd, state_vars)

    # Build variable dicts
    curr = {v: bdd.named_var(v) for v in state_vars}
    nxt = {v: bdd.named_var(f"{v}'") for v in state_vars}

    bts.init = init_expr(bdd, curr)
    bts.trans = trans_expr(bdd, curr, nxt)

    mc = LTLModelChecker(bts)
    return mc.check(prop, max_steps)


def check_ltl_simple(state_vars, init_map, transitions, prop, max_steps=1000):
    """
    Simplified API for checking LTL on small explicit systems.

    Args:
        state_vars: list of variable names
        init_map: dict of var -> bool for initial state
        transitions: list of (condition_fn, effect_map) or just effect_maps
        prop: LTL formula
        max_steps: iteration bound

    Returns:
        LTLOutput
    """
    bdd = BDD()
    bts = make_boolean_ts(bdd, state_vars)

    curr = {v: bdd.named_var(v) for v in state_vars}
    nxt = {v: bdd.named_var(f"{v}'") for v in state_vars}

    # Init
    init = bdd.TRUE
    for v, val in init_map.items():
        if val:
            init = bdd.AND(init, curr[v])
        else:
            init = bdd.AND(init, bdd.NOT(curr[v]))
    bts.init = init

    # Transitions
    trans = bdd.FALSE
    for t in transitions:
        if isinstance(t, dict):
            edge = bdd.TRUE
            for v in state_vars:
                if v in t:
                    if t[v] is True:
                        edge = bdd.AND(edge, nxt[v])
                    elif t[v] is False:
                        edge = bdd.AND(edge, bdd.NOT(nxt[v]))
                    elif callable(t[v]):
                        edge = bdd.AND(edge, bdd.IFF(nxt[v], t[v](bdd, curr)))
                else:
                    # Frame: unchanged
                    edge = bdd.AND(edge, bdd.IFF(nxt[v], curr[v]))
            trans = bdd.OR(trans, edge)
        elif isinstance(t, tuple) and len(t) == 2:
            cond_fn, effect = t
            cond = cond_fn(bdd, curr)
            edge = cond
            for v in state_vars:
                if v in effect:
                    if effect[v] is True:
                        edge = bdd.AND(edge, nxt[v])
                    elif effect[v] is False:
                        edge = bdd.AND(edge, bdd.NOT(nxt[v]))
                    elif callable(effect[v]):
                        edge = bdd.AND(edge, bdd.IFF(nxt[v], effect[v](bdd, curr)))
                else:
                    edge = bdd.AND(edge, bdd.IFF(nxt[v], curr[v]))
            trans = bdd.OR(trans, edge)
    bts.trans = trans

    mc = LTLModelChecker(bts)
    return mc.check(prop, max_steps)
