"""V158: Symbolic Mu-Calculus Model Checking

BDD-based evaluation of mu-calculus formulas, composing:
- V021 (BDD model checking) for BDD operations and symbolic state representation
- V157 (mu-calculus) for formula AST, parser, and explicit-state reference

Instead of representing state sets explicitly (Set[int]), formulas evaluate
to BDD nodes. This enables model checking systems with 2^N states using
polynomial-size BDD representations.
"""

import sys, os, math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Callable
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V157_mu_calculus'))

from bdd_model_checker import BDD, BDDNode, BooleanTS, make_boolean_ts, SymbolicModelChecker
from mu_calculus import (
    Formula, Prop, Var, TT, FF, Not, And, Or, Diamond, Box, Mu, Nu,
    LTS, make_lts, parse_mu, to_pnf, eval_formula, model_check as explicit_model_check,
    subformulas, free_vars, is_closed, alternation_depth, fixpoint_nesting_depth,
    formula_info,
    ctl_EF, ctl_AG, ctl_AF, ctl_EG, ctl_EU, ctl_AU, ctl_EX, ctl_AX,
)


# ---------------------------------------------------------------------------
# Symbolic Labeled Transition System
# ---------------------------------------------------------------------------

@dataclass
class SymbolicLTS:
    """LTS encoded symbolically with BDDs.

    State variables are BDD variables. Transitions are BDD relations
    over current-state and next-state variables. Labels (propositions)
    are BDDs over current-state variables.
    """
    bdd: BDD
    state_vars: List[str]       # current-state variable names
    next_vars: List[str]        # next-state (primed) variable names
    n_bits: int                 # number of bits per state encoding
    # BDD encodings
    init: BDDNode               # initial states (optional, for reachability)
    trans: Dict[str, BDDNode]   # action -> transition relation BDD(curr, next)
    labels: Dict[str, BDDNode]  # proposition -> BDD over current-state vars
    # Variable index maps
    var_indices: Dict[str, int]     # curr var name -> BDD variable index
    next_indices: Dict[str, int]    # next var name -> BDD variable index
    # For enumeration
    valid_states: Optional[BDDNode] = None  # mask for non-power-of-2 state counts


# ---------------------------------------------------------------------------
# Conversion: V157 LTS -> SymbolicLTS
# ---------------------------------------------------------------------------

def lts_to_symbolic(lts: LTS) -> SymbolicLTS:
    """Convert an explicit LTS to a symbolic (BDD-based) representation."""
    n_states = len(lts.states)
    if n_states == 0:
        raise ValueError("LTS has no states")

    n_bits = max(1, math.ceil(math.log2(n_states))) if n_states > 1 else 1
    total_vars = 2 * n_bits  # current + next state bits

    bdd = BDD(total_vars)

    # Variable names and indices
    state_vars = [f"s{i}" for i in range(n_bits)]
    next_vars = [f"s{i}'" for i in range(n_bits)]
    var_indices = {}
    next_indices = {}
    for i in range(n_bits):
        bdd.named_var(state_vars[i])
        var_indices[state_vars[i]] = bdd.var_index(state_vars[i])
        bdd.named_var(next_vars[i])
        next_indices[next_vars[i]] = bdd.var_index(next_vars[i])

    # State encoding: map state IDs to bit patterns
    sorted_states = sorted(lts.states)
    state_to_int = {s: i for i, s in enumerate(sorted_states)}

    def state_bdd(state_id, use_next=False):
        """Encode a single state as a BDD (conjunction of variable assignments)."""
        idx = state_to_int[state_id]
        indices = next_indices if use_next else var_indices
        names = next_vars if use_next else state_vars
        result = bdd.TRUE
        for bit in range(n_bits):
            var_node = bdd.var(indices[names[bit]])
            if (idx >> bit) & 1:
                result = bdd.AND(result, var_node)
            else:
                result = bdd.AND(result, bdd.NOT(var_node))
        return result

    # Valid states mask (for non-power-of-2 state counts)
    valid = bdd.FALSE
    for s in sorted_states:
        valid = bdd.OR(valid, state_bdd(s, use_next=False))

    # Transition relations per action
    actions = lts.actions()
    if not actions:
        actions = {"tau"}  # default action if none

    trans = {}
    for act in actions:
        rel = bdd.FALSE
        for s in sorted_states:
            s_bdd = state_bdd(s, use_next=False)
            for a, t in lts.transitions.get(s, []):
                if a == act:
                    t_bdd = state_bdd(t, use_next=True)
                    rel = bdd.OR(rel, bdd.AND(s_bdd, t_bdd))
        trans[act] = rel

    # Label BDDs (propositions)
    label_bdds = {}
    all_props = set()
    for s in sorted_states:
        for p in lts.labels.get(s, set()):
            all_props.add(p)

    for p in all_props:
        p_bdd = bdd.FALSE
        for s in sorted_states:
            if p in lts.labels.get(s, set()):
                p_bdd = bdd.OR(p_bdd, state_bdd(s, use_next=False))
        label_bdds[p] = p_bdd

    # Init = all states (no distinguished initial state in V157 LTS)
    init = valid

    return SymbolicLTS(
        bdd=bdd, state_vars=state_vars, next_vars=next_vars, n_bits=n_bits,
        init=init, trans=trans, labels=label_bdds,
        var_indices=var_indices, next_indices=next_indices,
        valid_states=valid,
    )


def make_symbolic_lts(
    state_var_names: List[str],
    init_fn: Callable,
    trans_fns: Dict[str, Callable],
    label_fns: Dict[str, Callable],
) -> SymbolicLTS:
    """Create a SymbolicLTS directly from BDD-building functions.

    Args:
        state_var_names: Names of state variables (each is one BDD variable)
        init_fn: fn(bdd, curr_vars) -> BDDNode for initial states
        trans_fns: {action: fn(bdd, curr_vars, next_vars) -> BDDNode}
        label_fns: {prop: fn(bdd, curr_vars) -> BDDNode}
    """
    n_bits = len(state_var_names)
    total_vars = 2 * n_bits
    bdd = BDD(total_vars)

    state_vars = list(state_var_names)
    next_vars = [f"{v}'" for v in state_var_names]

    var_indices = {}
    next_indices = {}
    curr_dict = {}
    next_dict = {}

    for v in state_vars:
        bdd.named_var(v)
        var_indices[v] = bdd.var_index(v)
        curr_dict[v] = bdd.var(var_indices[v])

    # next_dict uses UNPRIMED keys (same names as curr) for user convenience
    for i, v in enumerate(next_vars):
        bdd.named_var(v)
        next_indices[v] = bdd.var_index(v)
        next_dict[state_var_names[i]] = bdd.var(next_indices[v])

    init = init_fn(bdd, curr_dict)
    trans = {act: fn(bdd, curr_dict, next_dict) for act, fn in trans_fns.items()}
    labels = {prop: fn(bdd, curr_dict) for prop, fn in label_fns.items()}

    return SymbolicLTS(
        bdd=bdd, state_vars=state_vars, next_vars=next_vars, n_bits=n_bits,
        init=init, trans=trans, labels=labels,
        var_indices=var_indices, next_indices=next_indices,
    )


def boolean_ts_to_symbolic_lts(bts: BooleanTS, labels: Dict[str, BDDNode] = None) -> SymbolicLTS:
    """Convert a V021 BooleanTS to a SymbolicLTS.

    BooleanTS has a single monolithic transition relation. We store it
    under action None (wildcard action).

    Note: V021 BooleanTS uses unprimed keys in next_indices (e.g., "x" -> idx),
    while SymbolicLTS uses primed keys (e.g., "x'" -> idx). Convert here.
    """
    trans = {None: bts.trans}  # None = wildcard action (matches any)
    # Convert next_indices from unprimed to primed keys
    primed_next = {}
    for sv, nv in zip(bts.state_vars, bts.next_vars):
        if sv in bts.next_indices:
            primed_next[nv] = bts.next_indices[sv]
        elif nv in bts.next_indices:
            primed_next[nv] = bts.next_indices[nv]
    return SymbolicLTS(
        bdd=bts.bdd, state_vars=bts.state_vars, next_vars=bts.next_vars,
        n_bits=len(bts.state_vars),
        init=bts.init, trans=trans, labels=labels or {},
        var_indices=bts.var_indices, next_indices=primed_next,
    )


# ---------------------------------------------------------------------------
# Symbolic Mu-Calculus Evaluator
# ---------------------------------------------------------------------------

class SymbolicMuChecker:
    """Evaluate mu-calculus formulas on a SymbolicLTS using BDDs.

    Each formula evaluates to a BDD representing the set of states
    satisfying the formula.
    """

    def __init__(self, slts: SymbolicLTS, max_iterations: int = 1000):
        self.slts = slts
        self.bdd = slts.bdd
        self.max_iterations = max_iterations
        self._stats = {"fixpoint_iterations": 0, "diamond_ops": 0, "box_ops": 0}

    @property
    def stats(self):
        return dict(self._stats)

    def check(self, formula: Formula, env: Optional[Dict[str, BDDNode]] = None) -> BDDNode:
        """Evaluate formula, returning BDD of satisfying states."""
        if env is None:
            env = {}
        self._stats = {"fixpoint_iterations": 0, "diamond_ops": 0, "box_ops": 0}
        return self._eval(formula, env)

    def _eval(self, f: Formula, env: Dict[str, BDDNode]) -> BDDNode:
        if isinstance(f, TT):
            if self.slts.valid_states is not None:
                return self.slts.valid_states
            return self.bdd.TRUE

        if isinstance(f, FF):
            return self.bdd.FALSE

        if isinstance(f, Prop):
            base = self.slts.labels.get(f.name, self.bdd.FALSE)
            if self.slts.valid_states is not None:
                return self.bdd.AND(base, self.slts.valid_states)
            return base

        if isinstance(f, Var):
            if f.name in env:
                return env[f.name]
            raise ValueError(f"Unbound variable: {f.name}")

        if isinstance(f, Not):
            sub = self._eval(f.sub, env)
            result = self.bdd.NOT(sub)
            if self.slts.valid_states is not None:
                result = self.bdd.AND(result, self.slts.valid_states)
            return result

        if isinstance(f, And):
            left = self._eval(f.left, env)
            right = self._eval(f.right, env)
            return self.bdd.AND(left, right)

        if isinstance(f, Or):
            left = self._eval(f.left, env)
            right = self._eval(f.right, env)
            return self.bdd.OR(left, right)

        if isinstance(f, Diamond):
            return self._diamond(f.action, self._eval(f.sub, env))

        if isinstance(f, Box):
            return self._box(f.action, self._eval(f.sub, env))

        if isinstance(f, Mu):
            return self._mu(f.var, f.body, env)

        if isinstance(f, Nu):
            return self._nu(f.var, f.body, env)

        raise ValueError(f"Unknown formula type: {type(f)}")

    def _diamond(self, action: Optional[str], phi: BDDNode) -> BDDNode:
        """<action>phi: exists successor via action satisfying phi.

        Pre_action(phi) = exists next_vars. (trans_action(curr, next) AND phi[curr->next])
        """
        self._stats["diamond_ops"] += 1

        # Rename phi from current vars to next vars
        phi_next = self._rename_to_next(phi)

        # Get relevant transition relations
        trans_bdds = self._get_trans(action)

        result = self.bdd.FALSE
        for t_bdd in trans_bdds:
            conj = self.bdd.AND(t_bdd, phi_next)
            # Existentially quantify out next-state variables
            projected = self._exists_next(conj)
            result = self.bdd.OR(result, projected)

        if self.slts.valid_states is not None:
            result = self.bdd.AND(result, self.slts.valid_states)
        return result

    def _box(self, action: Optional[str], phi: BDDNode) -> BDDNode:
        """[action]phi: all successors via action satisfy phi.

        [a]phi = NOT <a>(NOT phi)
        But we must handle states with no successors (vacuously true).
        """
        self._stats["box_ops"] += 1

        not_phi = self.bdd.NOT(phi)
        if self.slts.valid_states is not None:
            not_phi = self.bdd.AND(not_phi, self.slts.valid_states)

        diamond_not_phi = self._diamond(action, not_phi)
        result = self.bdd.NOT(diamond_not_phi)

        if self.slts.valid_states is not None:
            result = self.bdd.AND(result, self.slts.valid_states)
        return result

    def _mu(self, var: str, body: Formula, env: Dict[str, BDDNode]) -> BDDNode:
        """Least fixpoint: start from FALSE, iterate body until stable."""
        current = self.bdd.FALSE
        for _ in range(self.max_iterations):
            new_env = dict(env)
            new_env[var] = current
            next_val = self._eval(body, new_env)
            self._stats["fixpoint_iterations"] += 1
            if next_val._id == current._id:
                return current
            current = next_val
        return current  # may not have converged

    def _nu(self, var: str, body: Formula, env: Dict[str, BDDNode]) -> BDDNode:
        """Greatest fixpoint: start from TRUE (or valid_states), iterate body until stable."""
        current = self.slts.valid_states if self.slts.valid_states is not None else self.bdd.TRUE
        for _ in range(self.max_iterations):
            new_env = dict(env)
            new_env[var] = current
            next_val = self._eval(body, new_env)
            self._stats["fixpoint_iterations"] += 1
            if next_val._id == current._id:
                return current
            current = next_val
        return current

    def _get_trans(self, action: Optional[str]) -> List[BDDNode]:
        """Get transition relation BDDs for an action (None = all actions)."""
        if action is None:
            return list(self.slts.trans.values())
        if action in self.slts.trans:
            return [self.slts.trans[action]]
        return []  # no transitions for this action

    def _rename_to_next(self, phi: BDDNode) -> BDDNode:
        """Rename current-state variables to next-state variables in a BDD."""
        var_map = {}
        for i, sv in enumerate(self.slts.state_vars):
            curr_idx = self.slts.var_indices[sv]
            nxt_idx = self.slts.next_indices[self.slts.next_vars[i]]
            var_map[curr_idx] = nxt_idx
        return self.bdd.rename(phi, var_map)

    def _rename_to_curr(self, phi: BDDNode) -> BDDNode:
        """Rename next-state variables to current-state variables in a BDD."""
        var_map = {}
        for i, nv in enumerate(self.slts.next_vars):
            nxt_idx = self.slts.next_indices[nv]
            curr_idx = self.slts.var_indices[self.slts.state_vars[i]]
            var_map[nxt_idx] = curr_idx
        return self.bdd.rename(phi, var_map)

    def _exists_next(self, phi: BDDNode) -> BDDNode:
        """Existentially quantify out all next-state variables."""
        next_idxs = [self.slts.next_indices[nv] for nv in self.slts.next_vars]
        return self.bdd.exists_multi(next_idxs, phi)

    def _forall_next(self, phi: BDDNode) -> BDDNode:
        """Universally quantify out all next-state variables."""
        next_idxs = [self.slts.next_indices[nv] for nv in self.slts.next_vars]
        return self.bdd.forall_multi(next_idxs, phi)

    def sat_states(self, result_bdd: BDDNode) -> Set[int]:
        """Extract explicit state set from a BDD result (for small systems)."""
        assignments = self.bdd.all_sat(result_bdd)
        states = set()
        for assignment in assignments:
            # Decode current-state bits to state number
            state_num = 0
            valid = True
            for i, sv in enumerate(self.slts.state_vars):
                idx = self.slts.var_indices[sv]
                if idx in assignment:
                    if assignment[idx]:
                        state_num |= (1 << i)
                # If variable not in assignment, it's don't-care
                # Need to expand don't-cares
            states.add(state_num)

        # For proper expansion, use a more careful approach
        return self._expand_sat(result_bdd)

    def _expand_sat(self, result_bdd: BDDNode) -> Set[int]:
        """Properly expand BDD satisfying assignments to state numbers.

        BDD all_sat may return partial assignments (don't-care variables).
        We need to expand those into all concrete assignments.
        """
        if result_bdd._id == self.bdd.FALSE._id:
            return set()

        # Project away next-state vars first
        next_idxs = [self.slts.next_indices[nv] for nv in self.slts.next_vars]
        projected = result_bdd
        for idx in next_idxs:
            projected = self.bdd.exists(idx, projected)

        curr_idxs = [self.slts.var_indices[sv] for sv in self.slts.state_vars]
        assignments = self.bdd.all_sat(projected, num_vars=max(curr_idxs) + 1 if curr_idxs else 0)

        states = set()
        for asgn in assignments:
            # Find which state bits are don't-care
            fixed_bits = {}
            free_bits = []
            for i, sv in enumerate(self.slts.state_vars):
                idx = self.slts.var_indices[sv]
                if idx in asgn:
                    fixed_bits[i] = 1 if asgn[idx] else 0
                else:
                    free_bits.append(i)

            # Expand don't-care bits
            for mask in range(1 << len(free_bits)):
                state_num = 0
                for i, val in fixed_bits.items():
                    state_num |= (val << i)
                for j, bit_pos in enumerate(free_bits):
                    if (mask >> j) & 1:
                        state_num |= (1 << bit_pos)
                states.add(state_num)

        return states

    def sat_count(self, result_bdd: BDDNode) -> int:
        """Count number of states satisfying the BDD."""
        # Project away next-state variables
        projected = result_bdd
        for idx in [self.slts.next_indices[nv] for nv in self.slts.next_vars]:
            projected = self.bdd.exists(idx, projected)
        return self.bdd.sat_count(projected, self.slts.n_bits)


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

@dataclass
class SymMCResult:
    """Result of symbolic mu-calculus model checking."""
    formula: Formula
    sat_bdd: object             # BDDNode
    sat_states: Set[int]        # explicit state set (for small systems)
    sat_count: int
    total_states: int           # 2^n_bits (or valid state count)
    stats: Dict

    @property
    def holds_everywhere(self) -> bool:
        return self.sat_count == self.total_states

    @property
    def holds_nowhere(self) -> bool:
        return self.sat_count == 0


def symbolic_check(slts: SymbolicLTS, formula: Formula) -> SymMCResult:
    """Check a mu-calculus formula on a SymbolicLTS."""
    mc = SymbolicMuChecker(slts)
    result_bdd = mc.check(formula)
    sat = mc._expand_sat(result_bdd)

    # Total valid states
    if slts.valid_states is not None:
        total = len(mc._expand_sat(slts.valid_states))
    else:
        total = 2 ** slts.n_bits

    return SymMCResult(
        formula=formula,
        sat_bdd=result_bdd,
        sat_states=sat,
        sat_count=len(sat),
        total_states=total,
        stats=mc.stats,
    )


def symbolic_check_lts(lts: LTS, formula: Formula) -> SymMCResult:
    """Check a mu-calculus formula on an explicit LTS using BDD-based evaluation."""
    slts = lts_to_symbolic(lts)
    return symbolic_check(slts, formula)


def check_state_symbolic(slts: SymbolicLTS, formula: Formula, state: int) -> bool:
    """Check if a specific state satisfies a formula."""
    mc = SymbolicMuChecker(slts)
    result_bdd = mc.check(formula)

    # Build BDD for the single state
    state_bdd = _encode_state(slts, state)
    # Check intersection
    conj = slts.bdd.AND(result_bdd, state_bdd)
    return conj._id != slts.bdd.FALSE._id


def _encode_state(slts: SymbolicLTS, state_num: int) -> BDDNode:
    """Encode a state number as a BDD."""
    result = slts.bdd.TRUE
    for i, sv in enumerate(slts.state_vars):
        idx = slts.var_indices[sv]
        var_node = slts.bdd.var(idx)
        if (state_num >> i) & 1:
            result = slts.bdd.AND(result, var_node)
        else:
            result = slts.bdd.AND(result, slts.bdd.NOT(var_node))
    return result


# ---------------------------------------------------------------------------
# Comparison: Symbolic vs Explicit
# ---------------------------------------------------------------------------

def compare_with_explicit(lts: LTS, formula: Formula) -> Dict:
    """Compare symbolic (BDD) vs explicit (V157) model checking results."""
    # Explicit
    explicit_sat = explicit_model_check(lts, formula, method="direct")

    # Symbolic
    sym_result = symbolic_check_lts(lts, formula)

    # Map state numbers back to original state IDs
    sorted_states = sorted(lts.states)
    sym_original = set()
    for snum in sym_result.sat_states:
        if snum < len(sorted_states):
            sym_original.add(sorted_states[snum])

    agree = explicit_sat == sym_original

    return {
        "explicit_sat": explicit_sat,
        "symbolic_sat": sym_original,
        "agree": agree,
        "explicit_count": len(explicit_sat),
        "symbolic_count": len(sym_original),
        "total_states": len(lts.states),
        "symbolic_stats": sym_result.stats,
    }


# ---------------------------------------------------------------------------
# CTL via Symbolic Mu-Calculus
# ---------------------------------------------------------------------------

def check_ctl_symbolic(slts: SymbolicLTS, ctl_formula: Formula) -> SymMCResult:
    """Check a CTL formula expressed as mu-calculus on a SymbolicLTS."""
    return symbolic_check(slts, ctl_formula)


def check_ctl_on_lts(lts: LTS, ctl_formula: Formula) -> SymMCResult:
    """Check a CTL formula on an explicit LTS using symbolic evaluation."""
    slts = lts_to_symbolic(lts)
    return symbolic_check(slts, ctl_formula)


# ---------------------------------------------------------------------------
# Batch and Analysis
# ---------------------------------------------------------------------------

def batch_symbolic_check(slts: SymbolicLTS, formulas: List[Formula]) -> List[SymMCResult]:
    """Check multiple formulas on the same SymbolicLTS."""
    return [symbolic_check(slts, f) for f in formulas]


def symbolic_reachable(slts: SymbolicLTS, max_steps: int = 1000) -> Tuple[BDDNode, int]:
    """Compute forward-reachable states from init using BDDs."""
    bdd = slts.bdd
    reached = slts.init

    for step in range(max_steps):
        # Image: exists curr. (reached(curr) AND trans(curr, next))[next->curr]
        new = bdd.FALSE
        for t_bdd in slts.trans.values():
            conj = bdd.AND(reached, t_bdd)
            # Quantify out current-state variables
            curr_idxs = [slts.var_indices[sv] for sv in slts.state_vars]
            projected = bdd.exists_multi(curr_idxs, conj)
            # Rename next -> curr
            var_map = {}
            for i, nv in enumerate(slts.next_vars):
                nxt_idx = slts.next_indices[nv]
                curr_idx = slts.var_indices[slts.state_vars[i]]
                var_map[nxt_idx] = curr_idx
            renamed = bdd.rename(projected, var_map)
            new = bdd.OR(new, renamed)

        combined = bdd.OR(reached, new)
        if combined._id == reached._id:
            return reached, step + 1
        reached = combined

    return reached, max_steps


def check_safety_symbolic(slts: SymbolicLTS, prop_bdd: BDDNode, max_steps: int = 1000) -> Dict:
    """Check AG(prop) on a SymbolicLTS.

    Returns dict with result ('safe'/'unsafe'), reachable states, iterations.
    """
    reached, iters = symbolic_reachable(slts, max_steps)
    bdd = slts.bdd

    # Check if all reachable states satisfy prop
    violation = bdd.AND(reached, bdd.NOT(prop_bdd))
    if slts.valid_states is not None:
        violation = bdd.AND(violation, slts.valid_states)

    if violation._id == bdd.FALSE._id:
        return {"result": "safe", "iterations": iters}
    else:
        return {"result": "unsafe", "iterations": iters}


# ---------------------------------------------------------------------------
# Parametric Constructors for Common Systems
# ---------------------------------------------------------------------------

def make_counter_lts(n_bits: int, labels: Dict[str, Callable] = None) -> SymbolicLTS:
    """Create a counter system (0, 1, 2, ..., 2^n-1, 0, ...) with labels.

    labels: {prop_name: fn(state_value) -> bool}
    """
    total_vars = 2 * n_bits
    bdd = BDD(total_vars)

    state_vars = [f"b{i}" for i in range(n_bits)]
    next_vars = [f"b{i}'" for i in range(n_bits)]

    var_indices = {}
    next_indices = {}
    for v in state_vars:
        bdd.named_var(v)
        var_indices[v] = bdd.var_index(v)
    for v in next_vars:
        bdd.named_var(v)
        next_indices[v] = bdd.var_index(v)

    # Init: state 0 (all bits false)
    init = bdd.TRUE
    for sv in state_vars:
        init = bdd.AND(init, bdd.NOT(bdd.var(var_indices[sv])))

    # Transition: increment by 1 (modular)
    # next = curr + 1 mod 2^n
    # Encode as: for each state value i, transition from i to (i+1) mod 2^n
    trans_bdd = bdd.FALSE
    for i in range(2 ** n_bits):
        j = (i + 1) % (2 ** n_bits)
        src = bdd.TRUE
        dst = bdd.TRUE
        for bit in range(n_bits):
            sv = bdd.var(var_indices[state_vars[bit]])
            nv = bdd.var(next_indices[next_vars[bit]])
            if (i >> bit) & 1:
                src = bdd.AND(src, sv)
            else:
                src = bdd.AND(src, bdd.NOT(sv))
            if (j >> bit) & 1:
                dst = bdd.AND(dst, nv)
            else:
                dst = bdd.AND(dst, bdd.NOT(nv))
        trans_bdd = bdd.OR(trans_bdd, bdd.AND(src, dst))

    # Labels
    label_bdds = {}
    if labels:
        for prop, fn in labels.items():
            p_bdd = bdd.FALSE
            for i in range(2 ** n_bits):
                if fn(i):
                    s_bdd = bdd.TRUE
                    for bit in range(n_bits):
                        sv = bdd.var(var_indices[state_vars[bit]])
                        if (i >> bit) & 1:
                            s_bdd = bdd.AND(s_bdd, sv)
                        else:
                            s_bdd = bdd.AND(s_bdd, bdd.NOT(sv))
                    p_bdd = bdd.OR(p_bdd, s_bdd)
            label_bdds[prop] = p_bdd

    return SymbolicLTS(
        bdd=bdd, state_vars=state_vars, next_vars=next_vars, n_bits=n_bits,
        init=init, trans={"tick": trans_bdd}, labels=label_bdds,
        var_indices=var_indices, next_indices=next_indices,
    )


def make_mutex_lts(n_processes: int = 2) -> SymbolicLTS:
    """Create a mutual exclusion protocol LTS.

    Each process has states: idle(0), trying(1), critical(2).
    Encoded with 2 bits per process.
    """
    bits_per_proc = 2  # 0=idle, 1=trying, 2=critical
    n_bits = bits_per_proc * n_processes
    total_vars = 2 * n_bits
    bdd = BDD(total_vars)

    state_vars = []
    next_vars = []
    var_indices = {}
    next_indices = {}

    for p in range(n_processes):
        for b in range(bits_per_proc):
            sv = f"p{p}b{b}"
            nv = f"p{p}b{b}'"
            state_vars.append(sv)
            next_vars.append(nv)
            bdd.named_var(sv)
            var_indices[sv] = bdd.var_index(sv)
            bdd.named_var(nv)
            next_indices[nv] = bdd.var_index(nv)

    def proc_state_bdd(proc, val, use_next=False):
        """BDD for process proc being in state val."""
        result = bdd.TRUE
        for b in range(bits_per_proc):
            if use_next:
                name = f"p{proc}b{b}'"
                idx = next_indices[name]
            else:
                name = f"p{proc}b{b}"
                idx = var_indices[name]
            v = bdd.var(idx)
            if (val >> b) & 1:
                result = bdd.AND(result, v)
            else:
                result = bdd.AND(result, bdd.NOT(v))
        return result

    def others_unchanged(changing_proc, use_next=False):
        """BDD: all processes except changing_proc stay the same."""
        result = bdd.TRUE
        for p in range(n_processes):
            if p == changing_proc:
                continue
            for b in range(bits_per_proc):
                cv = bdd.var(var_indices[f"p{p}b{b}"])
                nv_node = bdd.var(next_indices[f"p{p}b{b}'"])
                result = bdd.AND(result, bdd.IFF(cv, nv_node))
        return result

    # Init: all idle
    init = bdd.TRUE
    for p in range(n_processes):
        init = bdd.AND(init, proc_state_bdd(p, 0))

    # Transitions (per process): idle->trying, trying->critical (if no other in critical), critical->idle
    trans_bdd = bdd.FALSE
    for p in range(n_processes):
        unch = others_unchanged(p)

        # idle -> trying
        t1 = bdd.AND(proc_state_bdd(p, 0), proc_state_bdd(p, 1, True))
        t1 = bdd.AND(t1, unch)
        trans_bdd = bdd.OR(trans_bdd, t1)

        # trying -> critical (only if no other process is in critical)
        no_other_crit = bdd.TRUE
        for q in range(n_processes):
            if q != p:
                no_other_crit = bdd.AND(no_other_crit, bdd.NOT(proc_state_bdd(q, 2)))
        t2 = bdd.AND(proc_state_bdd(p, 1), proc_state_bdd(p, 2, True))
        t2 = bdd.AND(t2, no_other_crit)
        t2 = bdd.AND(t2, unch)
        trans_bdd = bdd.OR(trans_bdd, t2)

        # critical -> idle
        t3 = bdd.AND(proc_state_bdd(p, 2), proc_state_bdd(p, 0, True))
        t3 = bdd.AND(t3, unch)
        trans_bdd = bdd.OR(trans_bdd, t3)

    # Labels
    label_bdds = {}
    for p in range(n_processes):
        label_bdds[f"idle{p}"] = proc_state_bdd(p, 0)
        label_bdds[f"trying{p}"] = proc_state_bdd(p, 1)
        label_bdds[f"critical{p}"] = proc_state_bdd(p, 2)

    # mutual exclusion label: at most one in critical
    mutex_bdd = bdd.TRUE
    for p in range(n_processes):
        for q in range(p + 1, n_processes):
            not_both = bdd.NOT(bdd.AND(proc_state_bdd(p, 2), proc_state_bdd(q, 2)))
            mutex_bdd = bdd.AND(mutex_bdd, not_both)
    label_bdds["mutex"] = mutex_bdd

    return SymbolicLTS(
        bdd=bdd, state_vars=state_vars, next_vars=next_vars, n_bits=n_bits,
        init=init, trans={None: trans_bdd}, labels=label_bdds,
        var_indices=var_indices, next_indices=next_indices,
    )


# ---------------------------------------------------------------------------
# Summary and Reporting
# ---------------------------------------------------------------------------

def symbolic_mu_summary(slts: SymbolicLTS, formula: Formula) -> str:
    """Human-readable summary of symbolic model checking."""
    result = symbolic_check(slts, formula)
    info = formula_info(formula)

    lines = [
        f"Formula: {formula}",
        f"  Alternation depth: {info['alternation_depth']}",
        f"  Fixpoint nesting: {info['fixpoint_nesting']}",
        f"  Closed: {info['is_closed']}",
        f"State bits: {slts.n_bits}",
        f"Actions: {list(slts.trans.keys())}",
        f"Propositions: {list(slts.labels.keys())}",
        f"Satisfying states: {result.sat_count} / {result.total_states}",
        f"  States: {sorted(result.sat_states) if result.sat_count <= 20 else '(too many)'}",
        f"Stats: {result.stats}",
    ]
    return "\n".join(lines)


def full_analysis(slts: SymbolicLTS, formulas: List[Tuple[str, Formula]]) -> Dict:
    """Run multiple named formulas and return structured results."""
    results = {}
    for name, formula in formulas:
        r = symbolic_check(slts, formula)
        results[name] = {
            "sat_count": r.sat_count,
            "total": r.total_states,
            "holds_everywhere": r.holds_everywhere,
            "holds_nowhere": r.holds_nowhere,
            "stats": r.stats,
        }
    return results
