"""
V021: BDD-based Symbolic Model Checking

Binary Decision Diagrams (BDDs) for boolean/finite-state model checking.
Complement to SMT-based approaches (V002 PDR, V015 k-induction).

BDDs provide:
  - Canonical representation of boolean functions
  - Efficient set operations (union, intersection, complement)
  - Image/preimage computation for reachability analysis
  - CTL model checking (EX, EG, EU, AF, AG, etc.)

For finite-state systems, BDDs can be exponentially more efficient than
SMT-based approaches because they represent state sets symbolically and
compute fixpoints via set operations rather than individual SMT queries.

References:
  - Bryant (1986) "Graph-Based Algorithms for Boolean Function Manipulation"
  - McMillan (1993) "Symbolic Model Checking"
  - Clarke, Grumberg, Peled (1999) "Model Checking" Ch. 6
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, FrozenSet
from enum import Enum

# Import V002 TransitionSystem for composition
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'V002_pdr_ic3'))
from pdr import TransitionSystem, PDRResult, PDROutput

# Import C037 SMT types for TS compatibility
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import Var, IntConst, BoolConst, App, Op, BOOL, INT, Term


# ============================================================
# BDD Core Library
# ============================================================

class BDDNode:
    """A node in a Binary Decision Diagram.

    Terminal nodes: var == -1, lo/hi are None.
      TRUE node:  var=-1, value=True
      FALSE node: var=-1, value=False

    Internal nodes: var >= 0, lo = low child (var=0), hi = high child (var=1).
    """
    __slots__ = ('var', 'lo', 'hi', 'value', '_id')

    def __init__(self, var, lo, hi, value=None, _id=0):
        self.var = var
        self.lo = lo
        self.hi = hi
        self.value = value
        self._id = _id

    def is_terminal(self):
        return self.var == -1

    def __repr__(self):
        if self.is_terminal():
            return f"BDD({self.value})"
        return f"BDD(x{self.var}, lo={self.lo._id}, hi={self.hi._id})"

    def __eq__(self, other):
        return isinstance(other, BDDNode) and self._id == other._id

    def __hash__(self):
        return hash(self._id)


class BDD:
    """BDD Manager -- handles node creation, caching, and operations.

    All operations return canonical (reduced, ordered) BDD nodes.
    Variable ordering: lower index = closer to root (higher in decision order).
    """

    def __init__(self, num_vars=0):
        self.num_vars = num_vars
        self._next_id = 2  # 0=FALSE, 1=TRUE
        self._unique_table = {}  # (var, lo_id, hi_id) -> node
        self._op_cache = {}  # (op, a_id, b_id) -> node

        # Terminal nodes
        self.FALSE = BDDNode(-1, None, None, value=False, _id=0)
        self.TRUE = BDDNode(-1, None, None, value=True, _id=1)

        # Variable name mapping
        self._var_names = {}  # index -> name
        self._name_to_idx = {}  # name -> index

    def _make_node(self, var, lo, hi):
        """Create or retrieve a unique BDD node (reduction rule)."""
        # Reduction: if lo == hi, skip this variable
        if lo._id == hi._id:
            return lo

        key = (var, lo._id, hi._id)
        if key in self._unique_table:
            return self._unique_table[key]

        node = BDDNode(var, lo, hi, _id=self._next_id)
        self._next_id += 1
        self._unique_table[key] = node
        return node

    def var(self, index):
        """Create a BDD for a single variable x_index."""
        if index >= self.num_vars:
            self.num_vars = index + 1
        return self._make_node(index, self.FALSE, self.TRUE)

    def named_var(self, name):
        """Create/get a named variable."""
        if name not in self._name_to_idx:
            idx = len(self._name_to_idx)
            self._name_to_idx[name] = idx
            self._var_names[idx] = name
            if idx >= self.num_vars:
                self.num_vars = idx + 1
        return self.var(self._name_to_idx[name])

    def var_index(self, name):
        """Get index for a named variable."""
        return self._name_to_idx[name]

    def var_name(self, index):
        """Get name for a variable index."""
        return self._var_names.get(index, f"x{index}")

    # --- Core BDD Operations ---

    def apply(self, op, a, b=None):
        """Apply a boolean operation to BDD(s).

        op: 'and', 'or', 'xor', 'nand', 'nor', 'iff', 'imp', 'not'
        """
        if op == 'not':
            return self._apply_not(a)
        return self._apply_binary(op, a, b)

    def _apply_not(self, a):
        """Complement a BDD."""
        if a.is_terminal():
            return self.TRUE if a.value is False else self.FALSE
        cache_key = ('not', a._id, 0)
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]
        lo = self._apply_not(a.lo)
        hi = self._apply_not(a.hi)
        result = self._make_node(a.var, lo, hi)
        self._op_cache[cache_key] = result
        return result

    def _apply_binary(self, op, a, b):
        """Apply binary boolean operation."""
        # Terminal cases
        if a.is_terminal() and b.is_terminal():
            av, bv = a.value, b.value
            if op == 'and': rv = av and bv
            elif op == 'or': rv = av or bv
            elif op == 'xor': rv = av ^ bv
            elif op == 'nand': rv = not (av and bv)
            elif op == 'nor': rv = not (av or bv)
            elif op == 'iff': rv = av == bv
            elif op == 'imp': rv = (not av) or bv
            else: raise ValueError(f"Unknown op: {op}")
            return self.TRUE if rv else self.FALSE

        # Short-circuit for common cases
        if op == 'and':
            if a._id == 0 or b._id == 0: return self.FALSE
            if a._id == 1: return b
            if b._id == 1: return a
            if a._id == b._id: return a
        elif op == 'or':
            if a._id == 1 or b._id == 1: return self.TRUE
            if a._id == 0: return b
            if b._id == 0: return a
            if a._id == b._id: return a

        # Cache lookup
        cache_key = (op, a._id, b._id)
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        # Recursive case: Shannon expansion on top variable
        if a.is_terminal():
            top_var = b.var
        elif b.is_terminal():
            top_var = a.var
        else:
            top_var = min(a.var, b.var)

        a_lo = a.lo if (not a.is_terminal() and a.var == top_var) else a
        a_hi = a.hi if (not a.is_terminal() and a.var == top_var) else a
        b_lo = b.lo if (not b.is_terminal() and b.var == top_var) else b
        b_hi = b.hi if (not b.is_terminal() and b.var == top_var) else b

        lo = self._apply_binary(op, a_lo, b_lo)
        hi = self._apply_binary(op, a_hi, b_hi)
        result = self._make_node(top_var, lo, hi)

        self._op_cache[cache_key] = result
        return result

    # --- Convenience operations ---

    def AND(self, a, b):
        return self.apply('and', a, b)

    def OR(self, a, b):
        return self.apply('or', a, b)

    def NOT(self, a):
        return self.apply('not', a)

    def IMP(self, a, b):
        return self.apply('imp', a, b)

    def IFF(self, a, b):
        return self.apply('iff', a, b)

    def XOR(self, a, b):
        return self.apply('xor', a, b)

    def ITE(self, cond, then_bdd, else_bdd):
        """If-then-else: (cond AND then) OR (NOT cond AND else)."""
        return self.OR(self.AND(cond, then_bdd), self.AND(self.NOT(cond), else_bdd))

    def and_all(self, bdds):
        """Conjunction of multiple BDDs."""
        result = self.TRUE
        for b in bdds:
            result = self.AND(result, b)
        return result

    def or_all(self, bdds):
        """Disjunction of multiple BDDs."""
        result = self.FALSE
        for b in bdds:
            result = self.OR(result, b)
        return result

    # --- Quantification ---

    def exists(self, var_idx, bdd):
        """Existential quantification: exists x_var. bdd
        = bdd[x=0] OR bdd[x=1]
        """
        lo = self.restrict(bdd, var_idx, False)
        hi = self.restrict(bdd, var_idx, True)
        return self.OR(lo, hi)

    def forall(self, var_idx, bdd):
        """Universal quantification: forall x_var. bdd
        = bdd[x=0] AND bdd[x=1]
        """
        lo = self.restrict(bdd, var_idx, False)
        hi = self.restrict(bdd, var_idx, True)
        return self.AND(lo, hi)

    def exists_multi(self, var_indices, bdd):
        """Existential quantification over multiple variables."""
        result = bdd
        for idx in var_indices:
            result = self.exists(idx, result)
        return result

    def forall_multi(self, var_indices, bdd):
        """Universal quantification over multiple variables."""
        result = bdd
        for idx in var_indices:
            result = self.forall(idx, result)
        return result

    def restrict(self, bdd, var_idx, value):
        """Restrict: substitute var_idx = value (True/False) in bdd."""
        if bdd.is_terminal():
            return bdd

        cache_key = ('restrict', bdd._id, var_idx, value)
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        if bdd.var == var_idx:
            result = bdd.hi if value else bdd.lo
        elif bdd.var > var_idx:
            # Variable not in this subtree
            result = bdd
        else:
            lo = self.restrict(bdd.lo, var_idx, value)
            hi = self.restrict(bdd.hi, var_idx, value)
            result = self._make_node(bdd.var, lo, hi)

        self._op_cache[cache_key] = result
        return result

    # --- Variable Substitution ---

    def compose(self, bdd, var_idx, replacement):
        """Replace variable var_idx with BDD 'replacement'.
        bdd[x := replacement] = ITE(replacement, bdd[x=1], bdd[x=0])
        """
        if bdd.is_terminal():
            return bdd

        cache_key = ('compose', bdd._id, var_idx, replacement._id)
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        if bdd.var == var_idx:
            result = self.ITE(replacement, bdd.hi, bdd.lo)
        elif bdd.var > var_idx:
            result = bdd
        else:
            lo = self.compose(bdd.lo, var_idx, replacement)
            hi = self.compose(bdd.hi, var_idx, replacement)
            result = self._make_node(bdd.var, lo, hi)

        self._op_cache[cache_key] = result
        return result

    def rename(self, bdd, var_map):
        """Rename variables: var_map is {old_idx: new_idx}.
        Builds a new BDD with variables renamed.
        """
        if bdd.is_terminal():
            return bdd

        cache_key = ('rename', bdd._id, tuple(sorted(var_map.items())))
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        lo = self.rename(bdd.lo, var_map)
        hi = self.rename(bdd.hi, var_map)
        new_var = var_map.get(bdd.var, bdd.var)
        # Can't just _make_node because ordering might change
        # Use ITE with the new variable
        new_var_bdd = self.var(new_var)
        result = self.ITE(new_var_bdd, hi, lo)

        self._op_cache[cache_key] = result
        return result

    # --- Counting and Enumeration ---

    def sat_count(self, bdd, num_vars=None):
        """Count the number of satisfying assignments."""
        if num_vars is None:
            num_vars = self.num_vars
        return self._sat_count_rec(bdd, 0, num_vars)

    def _sat_count_rec(self, bdd, level, num_vars):
        if bdd.is_terminal():
            if bdd.value:
                return 2 ** (num_vars - level)
            return 0
        # Account for skipped variables
        skip = 2 ** (bdd.var - level)
        lo_count = self._sat_count_rec(bdd.lo, bdd.var + 1, num_vars)
        hi_count = self._sat_count_rec(bdd.hi, bdd.var + 1, num_vars)
        return skip * (lo_count + hi_count)

    def any_sat(self, bdd):
        """Find one satisfying assignment (dict of var_idx -> bool)."""
        if bdd == self.FALSE:
            return None
        if bdd == self.TRUE:
            return {}
        assignment = {}
        self._any_sat_rec(bdd, assignment)
        return assignment

    def _any_sat_rec(self, bdd, assignment):
        if bdd.is_terminal():
            return bdd.value
        # Try high branch first
        if bdd.hi != self.FALSE:
            assignment[bdd.var] = True
            if self._any_sat_rec(bdd.hi, assignment):
                return True
        assignment[bdd.var] = False
        return self._any_sat_rec(bdd.lo, assignment)

    def all_sat(self, bdd, num_vars=None):
        """Enumerate all satisfying assignments."""
        if num_vars is None:
            num_vars = self.num_vars
        results = []
        self._all_sat_rec(bdd, {}, 0, num_vars, results)
        return results

    def _all_sat_rec(self, bdd, assignment, level, num_vars, results):
        if bdd == self.FALSE:
            return
        if bdd == self.TRUE:
            # Fill in don't-care variables
            for v in range(level, num_vars):
                if v not in assignment:
                    # Both values work; enumerate both
                    a0 = dict(assignment)
                    a0[v] = False
                    a1 = dict(assignment)
                    a1[v] = True
                    self._all_sat_rec(self.TRUE, a0, v + 1, num_vars, results)
                    self._all_sat_rec(self.TRUE, a1, v + 1, num_vars, results)
                    return
            results.append(dict(assignment))
            return
        # Process this node
        a_lo = dict(assignment)
        a_lo[bdd.var] = False
        self._all_sat_rec(bdd.lo, a_lo, bdd.var + 1, num_vars, results)
        a_hi = dict(assignment)
        a_hi[bdd.var] = True
        self._all_sat_rec(bdd.hi, a_hi, bdd.var + 1, num_vars, results)

    def node_count(self, bdd):
        """Count the number of nodes in a BDD."""
        visited = set()
        self._count_nodes(bdd, visited)
        return len(visited)

    def _count_nodes(self, bdd, visited):
        if bdd._id in visited:
            return
        visited.add(bdd._id)
        if not bdd.is_terminal():
            self._count_nodes(bdd.lo, visited)
            self._count_nodes(bdd.hi, visited)

    def to_expr(self, bdd, var_names=None):
        """Convert BDD to a readable boolean expression string."""
        if bdd == self.TRUE:
            return "True"
        if bdd == self.FALSE:
            return "False"
        if var_names is None:
            var_names = self._var_names

        def name(idx):
            return var_names.get(idx, f"x{idx}")

        if bdd.lo == self.FALSE and bdd.hi == self.TRUE:
            return name(bdd.var)
        if bdd.lo == self.TRUE and bdd.hi == self.FALSE:
            return f"!{name(bdd.var)}"

        parts = []
        if bdd.hi != self.FALSE:
            hi_str = self.to_expr(bdd.hi, var_names)
            if bdd.hi == self.TRUE:
                parts.append(name(bdd.var))
            else:
                parts.append(f"({name(bdd.var)} & {hi_str})")
        if bdd.lo != self.FALSE:
            lo_str = self.to_expr(bdd.lo, var_names)
            if bdd.lo == self.TRUE:
                parts.append(f"!{name(bdd.var)}")
            else:
                parts.append(f"(!{name(bdd.var)} & {lo_str})")

        if len(parts) == 1:
            return parts[0]
        return f"({' | '.join(parts)})"


# ============================================================
# Boolean Transition System (for BDD model checking)
# ============================================================

@dataclass
class BooleanTS:
    """A boolean (finite-state) transition system for BDD model checking.

    state_vars: list of variable names (each is a boolean bit)
    init: BDD over state_vars representing initial states
    trans: BDD over state_vars + next_state_vars representing transitions
    """
    bdd: BDD
    state_vars: List[str]  # names of current-state variables
    next_vars: List[str]   # names of next-state variables (primed)
    init: object  # BDD node
    trans: object  # BDD node
    var_indices: Dict[str, int] = field(default_factory=dict)
    next_indices: Dict[str, int] = field(default_factory=dict)


def make_boolean_ts(bdd, state_var_names):
    """Create a BooleanTS with named current and next-state variables."""
    state_vars = []
    next_vars = []
    var_indices = {}
    next_indices = {}

    for name in state_var_names:
        idx = bdd.named_var(name).var
        state_vars.append(name)
        var_indices[name] = idx

    for name in state_var_names:
        nname = name + "'"
        idx = bdd.named_var(nname).var
        next_vars.append(nname)
        next_indices[name] = idx

    return BooleanTS(
        bdd=bdd,
        state_vars=state_vars,
        next_vars=next_vars,
        init=bdd.TRUE,
        trans=bdd.TRUE,
        var_indices=var_indices,
        next_indices=next_indices,
    )


# ============================================================
# Symbolic Model Checker
# ============================================================

class MCResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class MCOutput:
    """Result of a model checking query."""
    result: MCResult
    witness: Optional[List[Dict[str, bool]]] = None  # Counterexample/witness trace
    states_explored: int = 0
    fixpoint_iterations: int = 0
    invariant: object = None  # BDD of reachable states (if computed)


class SymbolicModelChecker:
    """BDD-based symbolic model checker.

    Supports:
      - Forward/backward reachability
      - CTL model checking (EX, EU, EG, AF, AG, AX, AU, AR, EF, ER)
      - Image/preimage computation
      - Fixpoint iteration
    """

    def __init__(self, ts: BooleanTS):
        self.ts = ts
        self.bdd = ts.bdd
        self._curr_indices = list(ts.var_indices.values())
        self._next_indices = list(ts.next_indices.values())
        # Map from next-var index to current-var index
        self._next_to_curr = {}
        for name in ts.state_vars:
            self._next_to_curr[ts.next_indices[name]] = ts.var_indices[name]
        self._curr_to_next = {v: k for k, v in self._next_to_curr.items()}

    # --- Image and Preimage ---

    def image(self, states):
        """Compute successor states: Image(S) = exists curr. (S(curr) AND T(curr, next))[next->curr]

        Given a set of current states, compute the set of next states.
        """
        # Conjoin states with transition relation
        conjoined = self.bdd.AND(states, self.ts.trans)
        # Existentially quantify out current-state variables
        projected = self.bdd.exists_multi(self._curr_indices, conjoined)
        # Rename next-state variables to current-state variables
        result = self.bdd.rename(projected, self._next_to_curr)
        return result

    def preimage(self, states):
        """Compute predecessor states: Pre(S) = exists next. (S[curr->next] AND T(curr, next))

        Given a set of states, compute which states can reach them in one step.
        """
        # Rename state vars to next-state vars in the target set
        renamed = self.bdd.rename(states, self._curr_to_next)
        # Conjoin with transition relation
        conjoined = self.bdd.AND(renamed, self.ts.trans)
        # Existentially quantify out next-state variables
        result = self.bdd.exists_multi(self._next_indices, conjoined)
        return result

    # --- Reachability ---

    def forward_reachable(self, max_steps=1000):
        """Compute all forward-reachable states from init via BFS fixpoint."""
        reached = self.ts.init
        iterations = 0
        for i in range(max_steps):
            iterations += 1
            new_states = self.image(reached)
            next_reached = self.bdd.OR(reached, new_states)
            if next_reached._id == reached._id:
                break  # Fixpoint
            reached = next_reached
        return reached, iterations

    def backward_reachable(self, target, max_steps=1000):
        """Compute all states that can reach 'target' via backward BFS."""
        reached = target
        iterations = 0
        for i in range(max_steps):
            iterations += 1
            pre = self.preimage(reached)
            next_reached = self.bdd.OR(reached, pre)
            if next_reached._id == reached._id:
                break
            reached = next_reached
        return reached, iterations

    # --- Safety Checking ---

    def check_safety(self, prop, max_steps=1000):
        """Check AG prop: does 'prop' hold in all reachable states?

        Returns MCOutput with SAFE/UNSAFE and optional counterexample.
        """
        # Check init satisfies property
        init_violates = self.bdd.AND(self.ts.init, self.bdd.NOT(prop))
        if init_violates != self.bdd.FALSE:
            # Initial state violates property
            witness = self._extract_trace_forward([], init_violates)
            return MCOutput(
                result=MCResult.UNSAFE,
                witness=witness,
                states_explored=0,
                fixpoint_iterations=0,
            )

        # Forward reachability with safety check at each step
        reached = self.ts.init
        frontier = self.ts.init
        iterations = 0

        for step in range(max_steps):
            iterations += 1
            new_states = self.image(frontier)
            # Check if any new state violates property
            new_only = self.bdd.AND(new_states, self.bdd.NOT(reached))
            violators = self.bdd.AND(new_only, self.bdd.NOT(prop))

            if violators != self.bdd.FALSE:
                # Found a violation -- extract counterexample
                trace = self._extract_counterexample(prop, step + 1)
                count = self.bdd.sat_count(reached, len(self._curr_indices))
                return MCOutput(
                    result=MCResult.UNSAFE,
                    witness=trace,
                    states_explored=count,
                    fixpoint_iterations=iterations,
                )

            next_reached = self.bdd.OR(reached, new_only)
            if next_reached._id == reached._id:
                # Fixpoint -- all reachable states checked
                count = self.bdd.sat_count(reached, len(self._curr_indices))
                return MCOutput(
                    result=MCResult.SAFE,
                    states_explored=count,
                    fixpoint_iterations=iterations,
                    invariant=reached,
                )
            frontier = new_only
            reached = next_reached

        # Exceeded max steps
        count = self.bdd.sat_count(reached, len(self._curr_indices))
        return MCOutput(
            result=MCResult.UNKNOWN,
            states_explored=count,
            fixpoint_iterations=iterations,
        )

    def _extract_counterexample(self, prop, depth):
        """Extract a counterexample trace by backward search from violation."""
        # Forward: compute reachable sets at each step
        layers = [self.ts.init]
        reached = self.ts.init
        for i in range(depth):
            succ = self.image(layers[-1])
            new = self.bdd.AND(succ, self.bdd.NOT(reached))
            layers.append(self.bdd.OR(new, succ))
            reached = self.bdd.OR(reached, succ)

        # Find a violating state at the last layer
        bad = self.bdd.AND(layers[-1], self.bdd.NOT(prop))
        if bad == self.bdd.FALSE:
            return None

        # Backward: trace from bad state
        trace = []
        current_state = bad
        for i in range(len(layers) - 1, -1, -1):
            # Pick a concrete state
            state_bdd = self.bdd.AND(current_state, layers[i])
            if state_bdd == self.bdd.FALSE:
                state_bdd = current_state
            assignment = self.bdd.any_sat(state_bdd)
            if assignment is None:
                break
            state_dict = self._assignment_to_state(assignment)
            trace.append(state_dict)
            if i > 0:
                current_state = self.preimage(state_bdd)
                current_state = self.bdd.AND(current_state, layers[i - 1])

        trace.reverse()
        return trace

    def _extract_trace_forward(self, prefix_states, target_bdd):
        """Extract a single state from target_bdd."""
        assignment = self.bdd.any_sat(target_bdd)
        if assignment is None:
            return prefix_states
        state = self._assignment_to_state(assignment)
        return prefix_states + [state]

    def _assignment_to_state(self, assignment):
        """Convert BDD variable assignment to state dict."""
        state = {}
        for name in self.ts.state_vars:
            idx = self.ts.var_indices[name]
            state[name] = assignment.get(idx, False)
        return state

    # --- CTL Model Checking ---
    # EX, EG, EU, EF  (existential)
    # AX, AG, AU, AF  (universal)

    def EX(self, phi):
        """EX phi: states that have a successor satisfying phi."""
        return self.preimage(phi)

    def AX(self, phi):
        """AX phi: states where ALL successors satisfy phi.
        AX phi = NOT EX (NOT phi)
        """
        return self.bdd.NOT(self.EX(self.bdd.NOT(phi)))

    def EF(self, phi, max_steps=1000):
        """EF phi: states that can eventually reach phi.
        Least fixpoint: Z = phi OR EX(Z)
        """
        z = self.bdd.FALSE
        for _ in range(max_steps):
            z_new = self.bdd.OR(phi, self.EX(z))
            if z_new._id == z._id:
                return z
            z = z_new
        return z

    def AG(self, phi, max_steps=1000):
        """AG phi: phi holds on all paths forever.
        AG phi = NOT EF (NOT phi)
        """
        return self.bdd.NOT(self.EF(self.bdd.NOT(phi), max_steps))

    def AF(self, phi, max_steps=1000):
        """AF phi: on all paths, phi eventually holds.
        Least fixpoint: Z = phi OR AX(Z)
        """
        z = self.bdd.FALSE
        for _ in range(max_steps):
            z_new = self.bdd.OR(phi, self.AX(z))
            if z_new._id == z._id:
                return z
            z = z_new
        return z

    def EG(self, phi, max_steps=1000):
        """EG phi: there exists a path where phi holds forever.
        Greatest fixpoint: Z = phi AND EX(Z)
        """
        z = phi
        for _ in range(max_steps):
            z_new = self.bdd.AND(phi, self.EX(z))
            if z_new._id == z._id:
                return z
            z = z_new
        return z

    def EU(self, phi, psi, max_steps=1000):
        """E[phi U psi]: there exists a path where phi holds until psi.
        Least fixpoint: Z = psi OR (phi AND EX(Z))
        """
        z = self.bdd.FALSE
        for _ in range(max_steps):
            z_new = self.bdd.OR(psi, self.bdd.AND(phi, self.EX(z)))
            if z_new._id == z._id:
                return z
            z = z_new
        return z

    def AU(self, phi, psi, max_steps=1000):
        """A[phi U psi]: on all paths, phi holds until psi.
        Least fixpoint: Z = psi OR (phi AND AX(Z))
        """
        z = self.bdd.FALSE
        for _ in range(max_steps):
            z_new = self.bdd.OR(psi, self.bdd.AND(phi, self.AX(z)))
            if z_new._id == z._id:
                return z
            z = z_new
        return z

    def ER(self, phi, psi, max_steps=1000):
        """E[phi R psi]: exists path where psi holds until phi AND psi (or psi forever).
        Greatest fixpoint: Z = (phi AND psi) OR (psi AND EX(Z))
        ER = NOT AU(NOT phi, NOT psi)
        """
        return self.bdd.NOT(self.AU(self.bdd.NOT(phi), self.bdd.NOT(psi), max_steps))

    def AR(self, phi, psi, max_steps=1000):
        """A[phi R psi]: on all paths, psi holds until phi AND psi (or psi forever).
        AR = NOT EU(NOT phi, NOT psi)
        """
        return self.bdd.NOT(self.EU(self.bdd.NOT(phi), self.bdd.NOT(psi), max_steps))


# ============================================================
# V002 TransitionSystem -> BooleanTS Conversion
# ============================================================

def ts_to_boolean(v002_ts, bit_width=4):
    """Convert a V002 (integer) TransitionSystem to a BooleanTS.

    Encodes each integer variable using 'bit_width' boolean bits
    (unsigned, 0 to 2^bit_width - 1).

    This enables BDD-based model checking of finite-state projections
    of integer systems.
    """
    bdd = BDD()
    var_names = [name for name, sort in v002_ts.state_vars]

    # Create boolean variables for each bit of each integer var
    bit_vars = {}  # (var_name, bit_index) -> bdd variable name
    all_state_var_names = []

    for vname in var_names:
        for bit in range(bit_width):
            bname = f"{vname}_b{bit}"
            all_state_var_names.append(bname)
            bit_vars[(vname, bit)] = bname

    bts = make_boolean_ts(bdd, all_state_var_names)

    def int_to_bdd(vname, value, use_next=False):
        """Encode integer value as conjunction of bit constraints."""
        result = bdd.TRUE
        for bit in range(bit_width):
            bname = bit_vars[(vname, bit)]
            if use_next:
                bname = bname + "'"
            var_bdd = bdd.named_var(bname)
            if (value >> bit) & 1:
                result = bdd.AND(result, var_bdd)
            else:
                result = bdd.AND(result, bdd.NOT(var_bdd))
        return result

    def var_sum_bdd(vname, use_next=False):
        """Get BDD representation for the integer value of a variable.
        Returns list of BDD nodes for each bit.
        """
        bits = []
        for bit in range(bit_width):
            bname = bit_vars[(vname, bit)]
            if use_next:
                bname = bname + "'"
            bits.append(bdd.named_var(bname))
        return bits

    # Encode init formula
    if v002_ts.init_formula is not None:
        init_bdd = _encode_smt_formula(bdd, v002_ts.init_formula, bit_vars,
                                       bit_width, False)
        bts.init = init_bdd
    else:
        bts.init = bdd.TRUE

    # Encode transition formula
    if v002_ts.trans_formula is not None:
        trans_bdd = _encode_smt_formula(bdd, v002_ts.trans_formula, bit_vars,
                                        bit_width, False, allow_primed=True)
        bts.trans = trans_bdd
    else:
        bts.trans = bdd.TRUE

    return bts, bit_vars, bit_width


def _encode_smt_formula(bdd, formula, bit_vars, bit_width, use_next,
                         allow_primed=False):
    """Encode an SMT formula as a BDD.

    Handles: AND, OR, NOT, EQ, NEQ, LT, LE, GE, GT, ADD, SUB, constants, variables.
    For integer comparisons, uses unsigned bit-vector encoding.
    """
    if isinstance(formula, BoolConst):
        return bdd.TRUE if formula.value else bdd.FALSE

    if isinstance(formula, IntConst):
        # Return the integer value (will be used by parent)
        return ('int', formula.value)

    if isinstance(formula, Var):
        name = formula.name
        is_primed = name.endswith("'")
        if is_primed:
            base_name = name[:-1]
            # Return bit representation for next-state variable
            bits = []
            for bit in range(bit_width):
                bname = bit_vars.get((base_name, bit))
                if bname:
                    bits.append(bdd.named_var(bname + "'"))
                else:
                    bits.append(bdd.FALSE)
            return ('bits', bits)
        else:
            bits = []
            for bit in range(bit_width):
                bname = bit_vars.get((name, bit))
                if bname:
                    bits.append(bdd.named_var(bname))
                else:
                    bits.append(bdd.FALSE)
            return ('bits', bits)

    if isinstance(formula, App):
        op = formula.op
        args = formula.args

        if op == Op.AND:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            # Both should be boolean BDDs
            left = _to_bool_bdd(bdd, left, bit_width)
            right = _to_bool_bdd(bdd, right, bit_width)
            return bdd.AND(left, right)

        if op == Op.OR:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            left = _to_bool_bdd(bdd, left, bit_width)
            right = _to_bool_bdd(bdd, right, bit_width)
            return bdd.OR(left, right)

        if op == Op.NOT:
            inner = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            inner = _to_bool_bdd(bdd, inner, bit_width)
            return bdd.NOT(inner)

        if op == Op.EQ:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return _bdd_eq(bdd, left, right, bit_width)

        if op == Op.NEQ:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return bdd.NOT(_bdd_eq(bdd, left, right, bit_width))

        if op in (Op.LE, Op.LT, Op.GE, Op.GT):
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return _bdd_compare(bdd, op, left, right, bit_width)

        if op == Op.ADD:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return _bdd_add(bdd, left, right, bit_width)

        if op == Op.SUB:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return _bdd_sub(bdd, left, right, bit_width)

        if op == Op.MUL:
            left = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            right = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            return _bdd_mul(bdd, left, right, bit_width)

        if op == Op.ITE:
            cond = _encode_smt_formula(bdd, args[0], bit_vars, bit_width, use_next, allow_primed)
            then_v = _encode_smt_formula(bdd, args[1], bit_vars, bit_width, use_next, allow_primed)
            else_v = _encode_smt_formula(bdd, args[2], bit_vars, bit_width, use_next, allow_primed)
            cond = _to_bool_bdd(bdd, cond, bit_width)
            return _bdd_ite_arith(bdd, cond, then_v, else_v, bit_width)

        raise ValueError(f"Unsupported SMT op for BDD encoding: {op}")

    raise ValueError(f"Unsupported SMT term for BDD encoding: {type(formula)}")


def _to_bits(bdd, val, bit_width):
    """Convert a value representation to a list of bit BDDs."""
    if isinstance(val, tuple):
        if val[0] == 'int':
            n = val[1]
            bits = []
            for bit in range(bit_width):
                bits.append(bdd.TRUE if ((n >> bit) & 1) else bdd.FALSE)
            return bits
        elif val[0] == 'bits':
            return val[1]
    # Assume it's a boolean BDD -- treat as 1-bit
    bits = [val] + [bdd.FALSE] * (bit_width - 1)
    return bits


def _to_bool_bdd(bdd, val, bit_width):
    """Convert a value to a boolean BDD."""
    if isinstance(val, tuple):
        if val[0] == 'int':
            return bdd.TRUE if val[1] != 0 else bdd.FALSE
        elif val[0] == 'bits':
            # Non-zero check: OR all bits
            result = bdd.FALSE
            for b in val[1]:
                result = bdd.OR(result, b)
            return result
    return val  # Already a BDD


def _bdd_eq(bdd, left, right, bit_width):
    """Bit-vector equality."""
    left_bits = _to_bits(bdd, left, bit_width)
    right_bits = _to_bits(bdd, right, bit_width)
    result = bdd.TRUE
    for lb, rb in zip(left_bits, right_bits):
        result = bdd.AND(result, bdd.IFF(lb, rb))
    return result


def _bdd_compare(bdd, op, left, right, bit_width):
    """Bit-vector comparison (unsigned)."""
    left_bits = _to_bits(bdd, left, bit_width)
    right_bits = _to_bits(bdd, right, bit_width)

    # Build comparator from MSB to LSB
    # We compute LT and EQ simultaneously
    eq = bdd.TRUE
    lt = bdd.FALSE

    for bit in range(bit_width - 1, -1, -1):
        lb, rb = left_bits[bit], right_bits[bit]
        # left < right at this bit: !lb AND rb AND (all higher bits equal)
        lt = bdd.OR(lt, bdd.AND(eq, bdd.AND(bdd.NOT(lb), rb)))
        eq = bdd.AND(eq, bdd.IFF(lb, rb))

    if op == Op.LT:
        return lt
    elif op == Op.LE:
        return bdd.OR(lt, eq)
    elif op == Op.GT:
        return bdd.AND(bdd.NOT(lt), bdd.NOT(eq))
    elif op == Op.GE:
        return bdd.NOT(lt)
    else:
        raise ValueError(f"Unknown comparison op: {op}")


def _bdd_add(bdd, left, right, bit_width):
    """Bit-vector addition (unsigned, modular)."""
    left_bits = _to_bits(bdd, left, bit_width)
    right_bits = _to_bits(bdd, right, bit_width)
    result_bits = []
    carry = bdd.FALSE

    for bit in range(bit_width):
        lb, rb = left_bits[bit], right_bits[bit]
        # sum = lb XOR rb XOR carry
        s = bdd.XOR(bdd.XOR(lb, rb), carry)
        # carry = (lb AND rb) OR (lb AND carry) OR (rb AND carry)
        carry = bdd.OR(bdd.OR(bdd.AND(lb, rb), bdd.AND(lb, carry)), bdd.AND(rb, carry))
        result_bits.append(s)

    return ('bits', result_bits)


def _bdd_sub(bdd, left, right, bit_width):
    """Bit-vector subtraction (unsigned, modular via 2's complement)."""
    left_bits = _to_bits(bdd, left, bit_width)
    right_bits = _to_bits(bdd, right, bit_width)
    # Negate right: flip bits and add 1
    neg_bits = [bdd.NOT(b) for b in right_bits]
    # Add 1 via carry-in
    result_bits = []
    carry = bdd.TRUE  # carry-in = 1 for 2's complement

    for bit in range(bit_width):
        lb = left_bits[bit]
        rb = neg_bits[bit]
        s = bdd.XOR(bdd.XOR(lb, rb), carry)
        carry = bdd.OR(bdd.OR(bdd.AND(lb, rb), bdd.AND(lb, carry)), bdd.AND(rb, carry))
        result_bits.append(s)

    return ('bits', result_bits)


def _bdd_mul(bdd, left, right, bit_width):
    """Bit-vector multiplication (unsigned, modular)."""
    left_bits = _to_bits(bdd, left, bit_width)
    right_bits = _to_bits(bdd, right, bit_width)

    # Standard shift-and-add multiplication
    result_bits = [bdd.FALSE] * bit_width

    for i in range(bit_width):
        # Partial product: left_bits AND right_bits[i], shifted left by i
        partial = [bdd.FALSE] * bit_width
        for j in range(bit_width):
            if i + j < bit_width:
                partial[i + j] = bdd.AND(left_bits[j], right_bits[i])

        # Add partial to result
        carry = bdd.FALSE
        new_result = []
        for bit in range(bit_width):
            s = bdd.XOR(bdd.XOR(result_bits[bit], partial[bit]), carry)
            carry = bdd.OR(
                bdd.OR(bdd.AND(result_bits[bit], partial[bit]),
                       bdd.AND(result_bits[bit], carry)),
                bdd.AND(partial[bit], carry))
            new_result.append(s)
        result_bits = new_result

    return ('bits', result_bits)


def _bdd_ite_arith(bdd_mgr, cond, then_v, else_v, bit_width):
    """BDD ITE for arithmetic (bit-vector) values."""
    then_bits = _to_bits(bdd_mgr, then_v, bit_width)
    else_bits = _to_bits(bdd_mgr, else_v, bit_width)
    result_bits = []
    for tb, eb in zip(then_bits, else_bits):
        result_bits.append(bdd_mgr.ITE(cond, tb, eb))
    return ('bits', result_bits)


# ============================================================
# High-level API
# ============================================================

def check_boolean_system(state_vars, init_expr, trans_expr, prop_expr):
    """Check safety of a boolean transition system.

    state_vars: list of variable names
    init_expr: function(bdd, vars_dict) -> BDD for initial states
    trans_expr: function(bdd, curr_dict, next_dict) -> BDD for transitions
    prop_expr: function(bdd, vars_dict) -> BDD for property

    Returns MCOutput.
    """
    bdd = BDD()
    bts = make_boolean_ts(bdd, state_vars)

    curr = {name: bdd.named_var(name) for name in state_vars}
    nxt = {name: bdd.named_var(name + "'") for name in state_vars}

    bts.init = init_expr(bdd, curr)
    bts.trans = trans_expr(bdd, curr, nxt)
    prop = prop_expr(bdd, curr)

    mc = SymbolicModelChecker(bts)
    return mc.check_safety(prop)


def check_ctl(state_vars, init_expr, trans_expr, ctl_expr, max_steps=1000):
    """Check a CTL formula on a boolean transition system.

    ctl_expr: function(mc, bdd, vars_dict) -> BDD of states satisfying formula

    Returns dict with:
      - sat_in_init: whether all initial states satisfy the formula
      - sat_states: BDD of satisfying states
      - sat_count: number of satisfying states
    """
    bdd = BDD()
    bts = make_boolean_ts(bdd, state_vars)

    curr = {name: bdd.named_var(name) for name in state_vars}
    nxt = {name: bdd.named_var(name + "'") for name in state_vars}

    bts.init = init_expr(bdd, curr)
    bts.trans = trans_expr(bdd, curr, nxt)

    mc = SymbolicModelChecker(bts)
    sat_bdd = ctl_expr(mc, bdd, curr)

    # Check if all initial states satisfy
    init_and_not_sat = bdd.AND(bts.init, bdd.NOT(sat_bdd))
    sat_in_init = (init_and_not_sat == bdd.FALSE)

    num_vars = len(state_vars)
    return {
        'sat_in_init': sat_in_init,
        'sat_states': sat_bdd,
        'sat_count': bdd.sat_count(sat_bdd, num_vars),
        'total_states': 2 ** num_vars,
    }


def check_v002_system(v002_ts, bit_width=4):
    """Check a V002 TransitionSystem using BDD-based model checking.

    Converts integer variables to bit-vectors and performs BDD reachability.
    Only works for systems with small integer ranges (due to bit-width encoding).

    Returns MCOutput.
    """
    bts, bit_vars, bw = ts_to_boolean(v002_ts, bit_width)

    # Encode property
    if v002_ts.prop_formula is not None:
        prop_bdd = _encode_smt_formula(bts.bdd, v002_ts.prop_formula, bit_vars,
                                        bw, False)
        prop_bdd = _to_bool_bdd(bts.bdd, prop_bdd, bw)
    else:
        prop_bdd = bts.bdd.TRUE

    mc = SymbolicModelChecker(bts)
    return mc.check_safety(prop_bdd)


def compare_with_pdr(v002_ts, bit_width=4):
    """Compare BDD-based model checking with V002 PDR on the same system.

    Returns dict with results from both approaches.
    """
    # BDD approach
    bdd_result = check_v002_system(v002_ts, bit_width)

    # PDR approach
    from pdr import check_ts as pdr_check_ts
    pdr_output = pdr_check_ts(v002_ts)

    return {
        'bdd_result': bdd_result.result.value,
        'bdd_iterations': bdd_result.fixpoint_iterations,
        'bdd_states': bdd_result.states_explored,
        'pdr_result': pdr_output.result.value,
        'pdr_frames': pdr_output.num_frames,
        'pdr_queries': pdr_output.stats.smt_queries,
        'agree': bdd_result.result.value == pdr_output.result.value,
    }
