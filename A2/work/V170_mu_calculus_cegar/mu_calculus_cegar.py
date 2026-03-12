"""V170: Symbolic Mu-Calculus Model Checker with CEGAR.

Combines mu-calculus model checking (Emerson-Lei fixpoint evaluation on BDDs)
with counterexample-guided abstraction refinement.

Key components:
1. Mu-calculus formula AST: propositions, boolean ops, modal ops (EX/AX/EU/AU),
   least/greatest fixpoints (mu/nu)
2. BDD-based model checker: evaluates formulas over Kripke structures symbolically
3. Predicate abstraction: abstract concrete systems using predicate sets
4. CEGAR loop: abstract model check -> counterexample -> check spurious ->
   refine predicates -> repeat

Composes: V021 (BDD engine) for symbolic state set manipulation.

References:
  - Emerson & Clarke (1982): mu-calculus model checking
  - Clarke, Grumberg, Long (1994): CEGAR for model checking
  - Graf & Saidi (1997): predicate abstraction
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet, Callable, Any
from enum import Enum, auto
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
from bdd_model_checker import BDD, BDDNode


# ============================================================
# Mu-Calculus Formula AST
# ============================================================

class FormulaKind(Enum):
    TRUE = auto()
    FALSE = auto()
    PROP = auto()       # atomic proposition
    VAR = auto()        # bound variable (fixpoint)
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    EX = auto()         # exists next
    AX = auto()         # forall next
    EU = auto()         # exists until
    AU = auto()         # forall until
    EF = auto()         # exists finally
    AF = auto()         # forall finally
    EG = auto()         # exists globally
    AG = auto()         # forall globally
    MU = auto()         # least fixpoint
    NU = auto()         # greatest fixpoint


@dataclass(frozen=True)
class Formula:
    kind: FormulaKind
    prop: Optional[str] = None          # for PROP
    var: Optional[str] = None           # for VAR, MU, NU
    children: Tuple = ()                # sub-formulas
    body: Optional['Formula'] = None    # for MU, NU

    def __repr__(self):
        if self.kind == FormulaKind.TRUE:
            return "tt"
        if self.kind == FormulaKind.FALSE:
            return "ff"
        if self.kind == FormulaKind.PROP:
            return self.prop
        if self.kind == FormulaKind.VAR:
            return self.var
        if self.kind == FormulaKind.NOT:
            return f"~{self.children[0]}"
        if self.kind in (FormulaKind.AND, FormulaKind.OR, FormulaKind.IMPLIES):
            op = {FormulaKind.AND: '/\\', FormulaKind.OR: '\\/',
                  FormulaKind.IMPLIES: '=>'}[self.kind]
            return f"({self.children[0]} {op} {self.children[1]})"
        if self.kind == FormulaKind.EX:
            return f"EX({self.children[0]})"
        if self.kind == FormulaKind.AX:
            return f"AX({self.children[0]})"
        if self.kind == FormulaKind.EU:
            return f"E[{self.children[0]} U {self.children[1]}]"
        if self.kind == FormulaKind.AU:
            return f"A[{self.children[0]} U {self.children[1]}]"
        if self.kind == FormulaKind.EF:
            return f"EF({self.children[0]})"
        if self.kind == FormulaKind.AF:
            return f"AF({self.children[0]})"
        if self.kind == FormulaKind.EG:
            return f"EG({self.children[0]})"
        if self.kind == FormulaKind.AG:
            return f"AG({self.children[0]})"
        if self.kind == FormulaKind.MU:
            return f"mu {self.var}.{self.body}"
        if self.kind == FormulaKind.NU:
            return f"nu {self.var}.{self.body}"
        return f"Formula({self.kind})"


# --- Formula constructors ---

def tt():
    return Formula(FormulaKind.TRUE)

def ff():
    return Formula(FormulaKind.FALSE)

def prop(name):
    return Formula(FormulaKind.PROP, prop=name)

def var(name):
    return Formula(FormulaKind.VAR, var=name)

def neg(f):
    return Formula(FormulaKind.NOT, children=(f,))

def conj(f, g):
    return Formula(FormulaKind.AND, children=(f, g))

def disj(f, g):
    return Formula(FormulaKind.OR, children=(f, g))

def implies(f, g):
    return Formula(FormulaKind.IMPLIES, children=(f, g))

def ex(f):
    return Formula(FormulaKind.EX, children=(f,))

def ax(f):
    return Formula(FormulaKind.AX, children=(f,))

def eu(f, g):
    return Formula(FormulaKind.EU, children=(f, g))

def au(f, g):
    return Formula(FormulaKind.AU, children=(f, g))

def ef(f):
    return Formula(FormulaKind.EF, children=(f,))

def af(f):
    return Formula(FormulaKind.AF, children=(f,))

def eg(f):
    return Formula(FormulaKind.EG, children=(f,))

def ag(f):
    return Formula(FormulaKind.AG, children=(f,))

def mu(varname, body):
    return Formula(FormulaKind.MU, var=varname, body=body)

def nu(varname, body):
    return Formula(FormulaKind.NU, var=varname, body=body)


# ============================================================
# Kripke Structure
# ============================================================

@dataclass
class KripkeStructure:
    """Explicit Kripke structure for model checking.

    states: set of state IDs (integers)
    initial: set of initial state IDs
    transitions: dict mapping state -> set of successor states
    labeling: dict mapping state -> set of atomic propositions true at that state
    """
    states: Set[int]
    initial: Set[int]
    transitions: Dict[int, Set[int]]
    labeling: Dict[int, Set[str]]

    def props(self):
        """All atomic propositions."""
        result = set()
        for s in self.states:
            result.update(self.labeling.get(s, set()))
        return result


# ============================================================
# BDD-based Mu-Calculus Model Checker
# ============================================================

@dataclass
class SymbolicKripke:
    """BDD-encoded Kripke structure."""
    bdd: BDD
    n_bits: int
    state_vars: List[str]
    next_vars: List[str]
    var_indices: Dict[str, int]
    next_indices: Dict[str, int]
    states: BDDNode          # valid states
    initial: BDDNode         # initial states
    transitions: BDDNode     # transition relation
    prop_bdds: Dict[str, BDDNode]  # proposition -> BDD of states where it holds


def kripke_to_symbolic(ks: KripkeStructure) -> SymbolicKripke:
    """Convert explicit Kripke structure to BDD-encoded form."""
    import math
    n = len(ks.states)
    n_bits = max(1, math.ceil(math.log2(n + 1))) if n > 0 else 1

    bdd = BDD(num_vars=2 * n_bits)

    state_vars = [f"s{i}" for i in range(n_bits)]
    next_vars = [f"s{i}'" for i in range(n_bits)]
    var_indices = {}
    next_indices = {}

    # Interleave curr/next vars for better BDD ordering
    for i in range(n_bits):
        bdd.named_var(state_vars[i])
        var_indices[state_vars[i]] = bdd.var_index(state_vars[i])
        bdd.named_var(next_vars[i])
        next_indices[next_vars[i]] = bdd.var_index(next_vars[i])

    state_list = sorted(ks.states)
    state_map = {s: i for i, s in enumerate(state_list)}

    def encode_state(sid, use_next=False):
        idx = state_map[sid]
        indices = next_indices if use_next else var_indices
        names = next_vars if use_next else state_vars
        result = bdd.TRUE
        for bit in range(n_bits):
            v = bdd.var(indices[names[bit]])
            if (idx >> bit) & 1:
                result = bdd.AND(result, v)
            else:
                result = bdd.AND(result, bdd.NOT(v))
        return result

    # States BDD
    states_bdd = bdd.FALSE
    for s in state_list:
        states_bdd = bdd.OR(states_bdd, encode_state(s))

    # Initial states
    init_bdd = bdd.FALSE
    for s in ks.initial:
        init_bdd = bdd.OR(init_bdd, encode_state(s))

    # Transitions
    trans_bdd = bdd.FALSE
    for s, succs in ks.transitions.items():
        if s not in state_map:
            continue
        s_bdd = encode_state(s)
        for t in succs:
            if t not in state_map:
                continue
            t_bdd = encode_state(t, use_next=True)
            trans_bdd = bdd.OR(trans_bdd, bdd.AND(s_bdd, t_bdd))

    # Propositions
    prop_bdds = {}
    for p in ks.props():
        p_bdd = bdd.FALSE
        for s in state_list:
            if p in ks.labeling.get(s, set()):
                p_bdd = bdd.OR(p_bdd, encode_state(s))
        prop_bdds[p] = p_bdd

    return SymbolicKripke(
        bdd=bdd, n_bits=n_bits,
        state_vars=state_vars, next_vars=next_vars,
        var_indices=var_indices, next_indices=next_indices,
        states=states_bdd, initial=init_bdd,
        transitions=trans_bdd, prop_bdds=prop_bdds
    )


def _sk_preimage(sk: SymbolicKripke, target: BDDNode) -> BDDNode:
    """Pre-image: states with at least one successor in target."""
    var_map = {}
    for i, sv in enumerate(sk.state_vars):
        var_map[sk.var_indices[sv]] = sk.next_indices[sk.next_vars[i]]
    target_next = sk.bdd.rename(target, var_map)
    conj = sk.bdd.AND(sk.transitions, target_next)
    next_idxs = [sk.next_indices[nv] for nv in sk.next_vars]
    return sk.bdd.exists_multi(next_idxs, conj)


def _sk_preimage_universal(sk: SymbolicKripke, target: BDDNode) -> BDDNode:
    """Universal pre-image: states where ALL successors are in target.
    = states AND NOT(pre(NOT target))
    """
    has_succ_outside = _sk_preimage(sk, sk.bdd.AND(sk.states, sk.bdd.NOT(target)))
    return sk.bdd.AND(sk.states, sk.bdd.NOT(has_succ_outside))


def _bdd_eval(bdd: BDD, node: BDDNode, assignment: Dict[int, bool]) -> bool:
    """Evaluate a BDD under an assignment."""
    while not node.is_terminal():
        if assignment.get(node.var, False):
            node = node.hi
        else:
            node = node.lo
    return node.value


def _bdd_states(sk: SymbolicKripke, bdd_node: BDDNode) -> Set[int]:
    """Extract concrete state IDs from a BDD (for debugging/counterexamples)."""
    result = set()
    n = 2 ** sk.n_bits
    for i in range(n):
        assignment = {}
        for bit in range(sk.n_bits):
            assignment[sk.var_indices[sk.state_vars[bit]]] = bool((i >> bit) & 1)
        if _bdd_eval(sk.bdd, bdd_node, assignment):
            if _bdd_eval(sk.bdd, sk.states, assignment):
                result.add(i)
    return result


@dataclass
class MCResult:
    """Result of mu-calculus model checking."""
    satisfied: bool          # does formula hold at all initial states?
    sat_states: BDDNode      # BDD of states satisfying the formula
    initial_states: BDDNode  # BDD of initial states
    counterexample: Optional[List[int]] = None  # witness path (if not satisfied)


def check_mu_calculus(sk: SymbolicKripke, formula: Formula,
                      max_iter: int = 1000) -> MCResult:
    """Evaluate a mu-calculus formula over a symbolic Kripke structure.

    Uses Emerson-Lei fixpoint evaluation for mu/nu operators.
    CTL operators (EX, AX, EU, AU, EF, AF, EG, AG) are desugared to mu-calculus.

    Returns MCResult with satisfaction set and initial-state check.
    """
    env = {}  # variable name -> BDD

    def eval_formula(f: Formula) -> BDDNode:
        bdd = sk.bdd
        kind = f.kind

        if kind == FormulaKind.TRUE:
            return sk.states
        if kind == FormulaKind.FALSE:
            return bdd.FALSE
        if kind == FormulaKind.PROP:
            return bdd.AND(sk.states, sk.prop_bdds.get(f.prop, bdd.FALSE))
        if kind == FormulaKind.VAR:
            return env.get(f.var, bdd.FALSE)
        if kind == FormulaKind.NOT:
            inner = eval_formula(f.children[0])
            return bdd.AND(sk.states, bdd.NOT(inner))
        if kind == FormulaKind.AND:
            return bdd.AND(eval_formula(f.children[0]), eval_formula(f.children[1]))
        if kind == FormulaKind.OR:
            return bdd.OR(eval_formula(f.children[0]), eval_formula(f.children[1]))
        if kind == FormulaKind.IMPLIES:
            left = eval_formula(f.children[0])
            right = eval_formula(f.children[1])
            return bdd.OR(bdd.AND(sk.states, bdd.NOT(left)), right)

        # Modal operators
        if kind == FormulaKind.EX:
            inner = eval_formula(f.children[0])
            return bdd.AND(sk.states, _sk_preimage(sk, inner))
        if kind == FormulaKind.AX:
            inner = eval_formula(f.children[0])
            return bdd.AND(sk.states, _sk_preimage_universal(sk, inner))

        # CTL fixpoints
        if kind == FormulaKind.EF:
            # EF(phi) = mu X. phi \/ EX(X)
            phi = eval_formula(f.children[0])
            x = bdd.FALSE
            for _ in range(max_iter):
                x_new = bdd.OR(phi, bdd.AND(sk.states, _sk_preimage(sk, x)))
                if x_new == x:
                    return x
                x = x_new
            return x

        if kind == FormulaKind.AF:
            # AF(phi) = mu X. phi \/ AX(X)
            phi = eval_formula(f.children[0])
            x = bdd.FALSE
            for _ in range(max_iter):
                ax_x = _sk_preimage_universal(sk, x)
                x_new = bdd.OR(phi, bdd.AND(sk.states, ax_x))
                if x_new == x:
                    return x
                x = x_new
            return x

        if kind == FormulaKind.EG:
            # EG(phi) = nu X. phi /\ EX(X)
            phi = eval_formula(f.children[0])
            x = sk.states
            for _ in range(max_iter):
                pre_x = _sk_preimage(sk, x)
                x_new = bdd.AND(phi, bdd.AND(sk.states, pre_x))
                if x_new == x:
                    return x
                x = x_new
            return x

        if kind == FormulaKind.AG:
            # AG(phi) = nu X. phi /\ AX(X)
            phi = eval_formula(f.children[0])
            x = sk.states
            for _ in range(max_iter):
                ax_x = _sk_preimage_universal(sk, x)
                x_new = bdd.AND(phi, bdd.AND(sk.states, ax_x))
                if x_new == x:
                    return x
                x = x_new
            return x

        if kind == FormulaKind.EU:
            # E[phi U psi] = mu X. psi \/ (phi /\ EX(X))
            phi = eval_formula(f.children[0])
            psi = eval_formula(f.children[1])
            x = bdd.FALSE
            for _ in range(max_iter):
                pre_x = _sk_preimage(sk, x)
                x_new = bdd.OR(psi, bdd.AND(phi, bdd.AND(sk.states, pre_x)))
                if x_new == x:
                    return x
                x = x_new
            return x

        if kind == FormulaKind.AU:
            # A[phi U psi] = mu X. psi \/ (phi /\ AX(X))
            phi = eval_formula(f.children[0])
            psi = eval_formula(f.children[1])
            x = bdd.FALSE
            for _ in range(max_iter):
                ax_x = _sk_preimage_universal(sk, x)
                x_new = bdd.OR(psi, bdd.AND(phi, bdd.AND(sk.states, ax_x)))
                if x_new == x:
                    return x
                x = x_new
            return x

        # Mu/Nu fixpoints
        if kind == FormulaKind.MU:
            # Least fixpoint: start from FALSE, iterate up
            x = bdd.FALSE
            for _ in range(max_iter):
                env[f.var] = x
                x_new = eval_formula(f.body)
                if x_new == x:
                    del env[f.var]
                    return x
                x = x_new
            if f.var in env:
                del env[f.var]
            return x

        if kind == FormulaKind.NU:
            # Greatest fixpoint: start from all states, iterate down
            x = sk.states
            for _ in range(max_iter):
                env[f.var] = x
                x_new = eval_formula(f.body)
                if x_new == x:
                    del env[f.var]
                    return x
                x = x_new
            if f.var in env:
                del env[f.var]
            return x

        raise ValueError(f"Unknown formula kind: {kind}")

    sat_bdd = eval_formula(formula)
    # Check: do all initial states satisfy?
    init_not_sat = sk.bdd.AND(sk.initial, sk.bdd.NOT(sat_bdd))
    satisfied = (init_not_sat == sk.bdd.FALSE)

    cex = None
    if not satisfied:
        cex = _extract_counterexample(sk, formula, sat_bdd)

    return MCResult(
        satisfied=satisfied,
        sat_states=sat_bdd,
        initial_states=sk.initial,
        counterexample=cex
    )


def _extract_counterexample(sk: SymbolicKripke, formula: Formula,
                             sat_bdd: BDDNode) -> Optional[List[int]]:
    """Extract a counterexample path from an unsatisfied initial state."""
    # Find an initial state not satisfying the formula
    bad_init = sk.bdd.AND(sk.initial, sk.bdd.NOT(sat_bdd))
    bad_states = _bdd_states(sk, bad_init)
    if not bad_states:
        return None

    start = min(bad_states)
    path = [start]

    # Try to extend the path for a few steps to show behavior
    current = start
    visited = {current}
    for _ in range(10):
        # Find a successor
        curr_bdd = _encode_single_state(sk, current)
        succ_bdd = _image_single(sk, curr_bdd)
        succs = _bdd_states(sk, succ_bdd)
        succs -= visited
        if not succs:
            break
        nxt = min(succs)
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


def _encode_single_state(sk: SymbolicKripke, state_id: int) -> BDDNode:
    """Encode a single state ID as a BDD."""
    result = sk.bdd.TRUE
    for bit in range(sk.n_bits):
        v = sk.bdd.var(sk.var_indices[sk.state_vars[bit]])
        if (state_id >> bit) & 1:
            result = sk.bdd.AND(result, v)
        else:
            result = sk.bdd.AND(result, sk.bdd.NOT(v))
    return result


def _image_single(sk: SymbolicKripke, source: BDDNode) -> BDDNode:
    """Compute successors of source states."""
    conj = sk.bdd.AND(sk.transitions, source)
    curr_idxs = [sk.var_indices[sv] for sv in sk.state_vars]
    projected = sk.bdd.exists_multi(curr_idxs, conj)
    # Rename next vars to curr
    var_map = {}
    for i, nv in enumerate(sk.next_vars):
        var_map[sk.next_indices[nv]] = sk.var_indices[sk.state_vars[i]]
    return sk.bdd.rename(projected, var_map)


# ============================================================
# Predicate Abstraction
# ============================================================

@dataclass
class ConcreteSystem:
    """A concrete (possibly infinite-state) transition system.

    States are represented as variable assignments (dicts).
    Transitions are given as a function.
    """
    variables: List[str]
    init_states: List[Dict[str, int]]
    transition_fn: Callable[[Dict[str, int]], List[Dict[str, int]]]
    prop_fn: Callable[[Dict[str, int]], Set[str]]  # state -> set of props true


@dataclass
class Predicate:
    """A predicate over concrete state variables.

    name: human-readable name (used as proposition in abstract system)
    test: function from state dict -> bool
    """
    name: str
    test: Callable[[Dict[str, int]], bool]


def predicate_abstract(system: ConcreteSystem,
                       predicates: List[Predicate],
                       max_states: int = 10000) -> KripkeStructure:
    """Construct abstract Kripke structure via predicate abstraction.

    Each abstract state is a tuple of predicate truth values.
    Transitions are over-approximated: if any concrete state in abstract
    state s can reach any concrete state in abstract state t, then s->t
    in the abstract system.

    For finite concrete systems (up to max_states), we enumerate all
    concrete states and compute exact abstraction.
    """
    # Enumerate concrete states via BFS
    concrete_states = []
    state_set = set()
    queue = deque()

    for init in system.init_states:
        key = _state_key(init)
        if key not in state_set:
            state_set.add(key)
            concrete_states.append(init)
            queue.append(init)

    while queue and len(concrete_states) < max_states:
        s = queue.popleft()
        for t in system.transition_fn(s):
            key = _state_key(t)
            if key not in state_set:
                state_set.add(key)
                concrete_states.append(t)
                queue.append(t)

    # Compute abstract state for each concrete state
    def abstract_state(s):
        return tuple(p.test(s) for p in predicates)

    abs_states = {}  # abstract_tuple -> abstract_id
    abs_id_counter = 0
    concrete_to_abs = {}

    for cs in concrete_states:
        at = abstract_state(cs)
        if at not in abs_states:
            abs_states[at] = abs_id_counter
            abs_id_counter += 1
        concrete_to_abs[_state_key(cs)] = abs_states[at]

    # Abstract transitions (over-approximation)
    abs_transitions = {i: set() for i in range(abs_id_counter)}
    for cs in concrete_states:
        src_abs = concrete_to_abs[_state_key(cs)]
        for ct in system.transition_fn(cs):
            key = _state_key(ct)
            if key in concrete_to_abs:
                tgt_abs = concrete_to_abs[key]
                abs_transitions[src_abs].add(tgt_abs)

    # Abstract labeling: proposition p holds at abstract state a
    # if ANY concrete state in a satisfies p (over-approximation for EX/EF)
    abs_labeling = {i: set() for i in range(abs_id_counter)}
    for cs in concrete_states:
        aid = concrete_to_abs[_state_key(cs)]
        # Predicate-based propositions
        at = abstract_state(cs)
        for j, p in enumerate(predicates):
            if at[j]:
                abs_labeling[aid].add(p.name)
        # System propositions
        for prop_name in system.prop_fn(cs):
            abs_labeling[aid].add(prop_name)

    # Initial abstract states
    abs_initial = set()
    for init in system.init_states:
        key = _state_key(init)
        if key in concrete_to_abs:
            abs_initial.add(concrete_to_abs[key])

    return KripkeStructure(
        states=set(range(abs_id_counter)),
        initial=abs_initial,
        transitions=abs_transitions,
        labeling=abs_labeling
    )


def _state_key(state_dict):
    """Hashable key for a state dictionary."""
    return tuple(sorted(state_dict.items()))


# ============================================================
# CEGAR Loop
# ============================================================

class CEGARVerdict(Enum):
    SATISFIED = auto()      # property holds (proven on abstraction)
    VIOLATED = auto()       # property violated (real counterexample)
    UNKNOWN = auto()        # max refinements reached without conclusion
    SPURIOUS_LIMIT = auto() # too many spurious counterexamples


@dataclass
class CEGARResult:
    """Result of CEGAR verification."""
    verdict: CEGARVerdict
    iterations: int
    predicates: List[Predicate]     # final predicate set
    counterexample: Optional[List[int]] = None  # concrete cex if violated
    concrete_trace: Optional[List[Dict[str, int]]] = None
    abstract_cex: Optional[List[int]] = None
    refinement_history: List[str] = field(default_factory=list)


def cegar_verify(system: ConcreteSystem,
                 formula: Formula,
                 initial_predicates: List[Predicate],
                 refine_fn: Optional[Callable] = None,
                 max_iterations: int = 20,
                 max_abstract_states: int = 10000) -> CEGARResult:
    """Counterexample-guided abstraction refinement.

    1. Abstract the system using current predicates
    2. Model-check the formula on the abstract system
    3. If satisfied -> SATISFIED (abstraction preserves universal properties)
    4. If violated:
       a. Extract abstract counterexample
       b. Check if counterexample is feasible in concrete system
       c. If feasible -> VIOLATED (real counterexample)
       d. If spurious -> refine predicates and repeat

    Args:
        system: concrete system to verify
        formula: mu-calculus formula to check
        initial_predicates: starting set of predicates
        refine_fn: function(system, spurious_cex, predicates) -> new_predicates
                   If None, uses default interpolation-based refinement
        max_iterations: max CEGAR iterations
        max_abstract_states: max states for abstract system
    """
    predicates = list(initial_predicates)
    history = []

    for iteration in range(max_iterations):
        # Step 1: Abstract
        abstract_ks = predicate_abstract(system, predicates, max_abstract_states)
        history.append(f"Iter {iteration}: {len(abstract_ks.states)} abstract states, "
                       f"{len(predicates)} predicates")

        # Step 2: Model check on abstraction
        sk = kripke_to_symbolic(abstract_ks)
        result = check_mu_calculus(sk, formula)

        if result.satisfied:
            # Property holds on abstraction => holds on concrete
            # (for ACTL* / universal properties over over-approximation)
            return CEGARResult(
                verdict=CEGARVerdict.SATISFIED,
                iterations=iteration + 1,
                predicates=predicates,
                refinement_history=history
            )

        # Step 3: Extract and validate counterexample
        abstract_cex = result.counterexample
        if abstract_cex is None:
            abstract_cex = _extract_abstract_cex_states(sk, result)

        if abstract_cex is None:
            return CEGARResult(
                verdict=CEGARVerdict.UNKNOWN,
                iterations=iteration + 1,
                predicates=predicates,
                refinement_history=history
            )

        # Step 4: Check feasibility
        concrete_trace = _check_feasibility(system, abstract_ks, abstract_cex, predicates)

        if concrete_trace is not None:
            # Real counterexample
            return CEGARResult(
                verdict=CEGARVerdict.VIOLATED,
                iterations=iteration + 1,
                predicates=predicates,
                counterexample=abstract_cex,
                concrete_trace=concrete_trace,
                abstract_cex=abstract_cex,
                refinement_history=history
            )

        # Step 5: Spurious -- refine
        if refine_fn:
            new_preds = refine_fn(system, abstract_cex, predicates)
        else:
            new_preds = _default_refine(system, abstract_cex, predicates)

        if len(new_preds) == len(predicates):
            # No new predicates -- stuck
            history.append(f"Iter {iteration}: refinement produced no new predicates")
            return CEGARResult(
                verdict=CEGARVerdict.SPURIOUS_LIMIT,
                iterations=iteration + 1,
                predicates=predicates,
                abstract_cex=abstract_cex,
                refinement_history=history
            )

        predicates = new_preds
        history.append(f"Iter {iteration}: refined to {len(predicates)} predicates")

    return CEGARResult(
        verdict=CEGARVerdict.UNKNOWN,
        iterations=max_iterations,
        predicates=predicates,
        refinement_history=history
    )


def _extract_abstract_cex_states(sk: SymbolicKripke, result: MCResult) -> Optional[List[int]]:
    """Extract abstract counterexample as list of state IDs."""
    bad_init = sk.bdd.AND(sk.initial, sk.bdd.NOT(result.sat_states))
    bad = _bdd_states(sk, bad_init)
    if not bad:
        return None
    start = min(bad)
    return [start]


def _check_feasibility(system: ConcreteSystem,
                       abstract_ks: KripkeStructure,
                       abstract_cex: List[int],
                       predicates: List[Predicate]) -> Optional[List[Dict[str, int]]]:
    """Check if abstract counterexample is feasible in concrete system.

    For each abstract state in cex, try to find concrete states and transitions.
    Returns concrete trace if feasible, None if spurious.
    """
    # Enumerate concrete states
    concrete_states = []
    state_set = set()
    queue = deque()

    for init in system.init_states:
        key = _state_key(init)
        if key not in state_set:
            state_set.add(key)
            concrete_states.append(init)
            queue.append(init)

    while queue and len(concrete_states) < 10000:
        s = queue.popleft()
        for t in system.transition_fn(s):
            key = _state_key(t)
            if key not in state_set:
                state_set.add(key)
                concrete_states.append(t)
                queue.append(t)

    # Build mapping: abstract state tuple -> abstract state ID
    # The abstract state is defined by predicate values
    tuple_to_abs_id = {}
    for cs in concrete_states:
        at = tuple(p.test(cs) for p in predicates)
        if at not in tuple_to_abs_id:
            # Find which abstract state ID this tuple maps to
            # by checking labeling consistency
            for sid in abstract_ks.states:
                labels = abstract_ks.labeling.get(sid, set())
                match = True
                for j, p in enumerate(predicates):
                    if at[j] and p.name not in labels:
                        match = False
                        break
                if match:
                    tuple_to_abs_id[at] = sid
                    break

    def concrete_to_abs(cs):
        at = tuple(p.test(cs) for p in predicates)
        return tuple_to_abs_id.get(at)

    # For single-state cex
    if len(abstract_cex) == 1:
        target_abs = abstract_cex[0]
        for init in system.init_states:
            if concrete_to_abs(init) == target_abs:
                return [init]
        return None

    # For multi-state cex: BFS to find concrete path matching abstract path
    first_abs = abstract_cex[0]
    candidates = [init for init in system.init_states
                  if concrete_to_abs(init) == first_abs]

    if not candidates:
        return None

    for start in candidates:
        trace = [start]
        current = start
        feasible = True
        for step in range(1, len(abstract_cex)):
            target_abs = abstract_cex[step]
            succs = system.transition_fn(current)
            found = False
            for s in succs:
                if concrete_to_abs(s) == target_abs:
                    trace.append(s)
                    current = s
                    found = True
                    break
            if not found:
                feasible = False
                break
        if feasible:
            return trace

    return None


def _default_refine(system: ConcreteSystem,
                    abstract_cex: List[int],
                    predicates: List[Predicate]) -> List[Predicate]:
    """Default refinement: add predicates to distinguish states.

    Heuristic: explore reachable states and add value-equality predicates
    for each observed variable value, plus threshold predicates.
    """
    new_preds = list(predicates)
    pred_names = {p.name for p in predicates}

    # Explore a few steps from initial states to find relevant thresholds
    observed = {}  # var -> set of observed values
    state_set = set()
    queue = deque()
    for init in system.init_states:
        key = _state_key(init)
        if key not in state_set:
            state_set.add(key)
            queue.append(init)
    # Only explore a small number of states to keep predicates manageable
    while queue and len(state_set) < 30:
        s = queue.popleft()
        for var in system.variables:
            val = s.get(var, 0)
            if var not in observed:
                observed[var] = set()
            observed[var].add(val)
        for t in system.transition_fn(s):
            key = _state_key(t)
            if key not in state_set:
                state_set.add(key)
                queue.append(t)

    # Add at most a few predicates per variable to avoid abstraction explosion
    added = 0
    max_new = 4  # at most 4 new predicates per refinement
    for var in system.variables:
        vals = sorted(observed.get(var, set()))
        for val in vals:
            if added >= max_new:
                break
            # Equality predicate
            name = f"{var}_eq_{val}"
            if name not in pred_names:
                v, t = var, val
                new_preds.append(Predicate(name, lambda s, v=v, t=t: s.get(v, 0) == t))
                pred_names.add(name)
                added += 1
            if added >= max_new:
                break
            # Threshold predicate
            name = f"{var}_ge_{val}"
            if name not in pred_names:
                v, t = var, val
                new_preds.append(Predicate(name, lambda s, v=v, t=t: s.get(v, 0) >= t))
                pred_names.add(name)
                added += 1

    return new_preds


# ============================================================
# Formula Parser
# ============================================================

def parse_formula(text: str) -> Formula:
    """Parse a mu-calculus formula from string.

    Grammar:
        formula := 'tt' | 'ff' | prop | var
                 | '~' formula
                 | '(' formula '/\\' formula ')'
                 | '(' formula '\\/' formula ')'
                 | '(' formula '=>' formula ')'
                 | 'EX' '(' formula ')'
                 | 'AX' '(' formula ')'
                 | 'EF' '(' formula ')'
                 | 'AF' '(' formula ')'
                 | 'EG' '(' formula ')'
                 | 'AG' '(' formula ')'
                 | 'E[' formula 'U' formula ']'
                 | 'A[' formula 'U' formula ']'
                 | 'mu' var '.' formula
                 | 'nu' var '.' formula
    """
    tokens = _tokenize(text)
    pos = [0]
    bound_vars = set()  # track mu/nu bound variable names

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def advance():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(tok):
        t = advance()
        if t != tok:
            raise ValueError(f"Expected '{tok}', got '{t}'")
        return t

    def parse_primary():
        t = peek()
        if t == 'tt':
            advance()
            return tt()
        if t == 'ff':
            advance()
            return ff()
        if t == '~':
            advance()
            return neg(parse_primary())
        if t == '(':
            advance()
            left = parse_expr()
            op = advance()
            if op == '/\\':
                right = parse_expr()
                expect(')')
                return conj(left, right)
            elif op == '\\/':
                right = parse_expr()
                expect(')')
                return disj(left, right)
            elif op == '=>':
                right = parse_expr()
                expect(')')
                return implies(left, right)
            else:
                raise ValueError(f"Unexpected binary op: {op}")
        if t in ('EX', 'AX', 'EF', 'AF', 'EG', 'AG'):
            advance()
            expect('(')
            inner = parse_expr()
            expect(')')
            return {'EX': ex, 'AX': ax, 'EF': ef, 'AF': af,
                    'EG': eg, 'AG': ag}[t](inner)
        if t == 'E[':
            advance()
            left = parse_expr()
            expect('U')
            right = parse_expr()
            expect(']')
            return eu(left, right)
        if t == 'A[':
            advance()
            left = parse_expr()
            expect('U')
            right = parse_expr()
            expect(']')
            return au(left, right)
        if t in ('mu', 'nu'):
            advance()
            v = advance()
            expect('.')
            bound_vars.add(v)
            body = parse_expr()
            result = mu(v, body) if t == 'mu' else nu(v, body)
            return result
        # Bound variable reference or proposition
        advance()
        if t in bound_vars:
            return var(t)
        return prop(t)

    def parse_expr():
        return parse_primary()

    result = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError(f"Unexpected tokens after formula: {tokens[pos[0]:]}")
    return result


def _tokenize(text):
    """Tokenize mu-calculus formula string."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text[i:i+2] == '/\\':
            tokens.append('/\\')
            i += 2
            continue
        if text[i:i+2] == '\\/':
            tokens.append('\\/')
            i += 2
            continue
        if text[i:i+2] == '=>':
            tokens.append('=>')
            i += 2
            continue
        if text[i:i+2] in ('E[', 'A['):
            tokens.append(text[i:i+2])
            i += 2
            continue
        if text[i] in '()[]~.':
            tokens.append(text[i])
            i += 1
            continue
        # Word token
        j = i
        while j < len(text) and (text[j].isalnum() or text[j] == '_'):
            j += 1
        if j > i:
            tokens.append(text[i:j])
            i = j
        else:
            raise ValueError(f"Unexpected character: {text[i]}")
    return tokens


# ============================================================
# Convenience: Direct model checking on explicit Kripke structures
# ============================================================

def model_check(ks: KripkeStructure, formula: Formula) -> MCResult:
    """Model check a formula on an explicit Kripke structure."""
    sk = kripke_to_symbolic(ks)
    return check_mu_calculus(sk, formula)


def model_check_states(ks: KripkeStructure, formula: Formula) -> Set[int]:
    """Return set of state IDs satisfying formula."""
    sk = kripke_to_symbolic(ks)
    result = check_mu_calculus(sk, formula)
    return _bdd_states(sk, result.sat_states)


# ============================================================
# Game Helpers / Example Systems
# ============================================================

def make_counter_system(max_val: int) -> ConcreteSystem:
    """Counter that increments from 0 to max_val then resets."""
    def trans(s):
        x = s['x']
        if x >= max_val:
            return [{'x': 0}]
        return [{'x': x + 1}]

    def props(s):
        result = set()
        if s['x'] == 0:
            result.add('zero')
        if s['x'] == max_val:
            result.add('max')
        if s['x'] % 2 == 0:
            result.add('even')
        return result

    return ConcreteSystem(
        variables=['x'],
        init_states=[{'x': 0}],
        transition_fn=trans,
        prop_fn=props
    )


def make_traffic_light() -> ConcreteSystem:
    """Traffic light: red -> green -> yellow -> red."""
    def trans(s):
        c = s['color']
        if c == 0:    # red
            return [{'color': 1}]
        elif c == 1:  # green
            return [{'color': 2}]
        else:         # yellow
            return [{'color': 0}]

    def props(s):
        c = s['color']
        if c == 0: return {'red'}
        if c == 1: return {'green'}
        return {'yellow'}

    return ConcreteSystem(
        variables=['color'],
        init_states=[{'color': 0}],
        transition_fn=trans,
        prop_fn=props
    )


def make_mutex_system() -> ConcreteSystem:
    """Two-process mutual exclusion attempt.

    States: (p1, p2) where each is 0=idle, 1=trying, 2=critical
    Transitions: nondeterministic (either process can step)
    Bug: both can enter critical simultaneously (no real mutex)
    """
    def trans(s):
        p1, p2 = s['p1'], s['p2']
        succs = []
        # p1 transitions
        if p1 == 0:
            succs.append({'p1': 1, 'p2': p2})  # idle -> trying
        elif p1 == 1:
            succs.append({'p1': 2, 'p2': p2})  # trying -> critical
        else:
            succs.append({'p1': 0, 'p2': p2})  # critical -> idle
        # p2 transitions
        if p2 == 0:
            succs.append({'p1': p1, 'p2': 1})
        elif p2 == 1:
            succs.append({'p1': p1, 'p2': 2})
        else:
            succs.append({'p1': p1, 'p2': 0})
        return succs

    def props(s):
        result = set()
        if s['p1'] == 2 and s['p2'] == 2:
            result.add('both_critical')
        if s['p1'] == 2 or s['p2'] == 2:
            result.add('some_critical')
        if s['p1'] == 0 and s['p2'] == 0:
            result.add('both_idle')
        return result

    return ConcreteSystem(
        variables=['p1', 'p2'],
        init_states=[{'p1': 0, 'p2': 0}],
        transition_fn=trans,
        prop_fn=props
    )


def make_bounded_counter(bound: int) -> ConcreteSystem:
    """Counter with nondeterministic increment/reset. Checks bound."""
    def trans(s):
        x = s['x']
        succs = [{'x': x + 1}]  # always can increment
        if x > 0:
            succs.append({'x': 0})  # can reset from nonzero
        return succs

    def props(s):
        result = set()
        if s['x'] == 0:
            result.add('zero')
        if s['x'] >= bound:
            result.add('overflow')
        if s['x'] < bound:
            result.add('safe')
        return result

    return ConcreteSystem(
        variables=['x'],
        init_states=[{'x': 0}],
        transition_fn=trans,
        prop_fn=props
    )


# ============================================================
# Analysis Helpers
# ============================================================

def formula_size(f: Formula) -> int:
    """Count nodes in formula AST."""
    count = 1
    for c in f.children:
        count += formula_size(c)
    if f.body:
        count += formula_size(f.body)
    return count


def formula_depth(f: Formula) -> int:
    """Nesting depth of formula."""
    child_depth = 0
    for c in f.children:
        child_depth = max(child_depth, formula_depth(c))
    if f.body:
        child_depth = max(child_depth, formula_depth(f.body))
    return 1 + child_depth


def alternation_depth(f: Formula) -> int:
    """Fixpoint alternation depth (mu-nu nesting depth)."""
    if f.kind == FormulaKind.MU:
        return 1 + _nu_depth_inside(f.body)
    if f.kind == FormulaKind.NU:
        return 1 + _mu_depth_inside(f.body)
    d = 0
    for c in f.children:
        d = max(d, alternation_depth(c))
    if f.body:
        d = max(d, alternation_depth(f.body))
    return d


def _mu_depth_inside(f: Formula) -> int:
    if f.kind == FormulaKind.MU:
        return 1 + _nu_depth_inside(f.body)
    d = 0
    for c in f.children:
        d = max(d, _mu_depth_inside(c))
    if f.body:
        d = max(d, _mu_depth_inside(f.body))
    return d


def _nu_depth_inside(f: Formula) -> int:
    if f.kind == FormulaKind.NU:
        return 1 + _mu_depth_inside(f.body)
    d = 0
    for c in f.children:
        d = max(d, _nu_depth_inside(c))
    if f.body:
        d = max(d, _nu_depth_inside(f.body))
    return d


def verify_model_check(ks: KripkeStructure, formula: Formula,
                        expected: bool) -> Dict[str, Any]:
    """Verify model checking result against expected value."""
    result = model_check(ks, formula)
    return {
        'formula': str(formula),
        'expected': expected,
        'actual': result.satisfied,
        'correct': result.satisfied == expected,
        'sat_states': _bdd_states(kripke_to_symbolic(ks), result.sat_states),
        'counterexample': result.counterexample
    }


def compare_mc_methods(ks: KripkeStructure, formula: Formula) -> Dict[str, Any]:
    """Compare direct symbolic MC with CEGAR approach."""
    # Direct
    sk = kripke_to_symbolic(ks)
    direct = check_mu_calculus(sk, formula)

    return {
        'formula': str(formula),
        'direct_satisfied': direct.satisfied,
        'direct_sat_states': _bdd_states(sk, direct.sat_states),
        'num_states': len(ks.states),
    }


def batch_check(ks: KripkeStructure, formulas: List[Tuple[str, Formula]]) -> List[Dict]:
    """Check multiple formulas on the same Kripke structure."""
    sk = kripke_to_symbolic(ks)
    results = []
    for name, f in formulas:
        r = check_mu_calculus(sk, f)
        results.append({
            'name': name,
            'formula': str(f),
            'satisfied': r.satisfied,
            'sat_states': _bdd_states(sk, r.sat_states),
        })
    return results


def cegar_statistics(result: CEGARResult) -> Dict[str, Any]:
    """Extract statistics from a CEGAR result."""
    return {
        'verdict': result.verdict.name,
        'iterations': result.iterations,
        'num_predicates': len(result.predicates),
        'predicate_names': [p.name for p in result.predicates],
        'has_counterexample': result.concrete_trace is not None,
        'history': result.refinement_history,
    }
