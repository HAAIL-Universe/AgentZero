"""V089: Tree Automata

Bottom-up and top-down finite tree automata over ranked alphabets.

Features:
- RankedAlphabet: symbols with fixed arities (constants, unary, binary, etc.)
- Tree: ranked tree data structure with symbol + children
- BottomUpTreeAutomaton (BUTA): nondeterministic bottom-up FTA
  - Transitions: f(q1, ..., qn) -> q
  - Determinization via subset construction
  - Boolean closure: union, intersection, complement
  - Emptiness, membership, language inclusion
  - Minimization via partition refinement
- TopDownTreeAutomaton (TDTA): nondeterministic top-down FTA
  - Transitions: q -> f(q1, ..., qn)
  - Equivalent to BUTA (conversion both ways)
- Tree generation: enumerate trees in the language
- Product construction for intersection
- Pumping: find witness/counterexample trees

Standalone -- new domain (tree-structured data).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Any
from collections import deque
import itertools


# --- Ranked Alphabet ---

@dataclass(frozen=True)
class Symbol:
    """A ranked symbol with a name and arity (number of children)."""
    name: str
    arity: int

    def __repr__(self):
        return f"{self.name}/{self.arity}"


class RankedAlphabet:
    """A set of ranked symbols."""

    def __init__(self, symbols: List[Symbol] = None):
        self.symbols: Dict[str, Symbol] = {}
        if symbols:
            for s in symbols:
                self.symbols[s.name] = s

    def add(self, name: str, arity: int) -> Symbol:
        s = Symbol(name, arity)
        self.symbols[name] = s
        return s

    def get(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def constants(self) -> List[Symbol]:
        return [s for s in self.symbols.values() if s.arity == 0]

    def by_arity(self, arity: int) -> List[Symbol]:
        return [s for s in self.symbols.values() if s.arity == arity]

    def max_arity(self) -> int:
        return max((s.arity for s in self.symbols.values()), default=0)

    def __iter__(self):
        return iter(self.symbols.values())

    def __len__(self):
        return len(self.symbols)

    def __repr__(self):
        return f"RankedAlphabet({list(self.symbols.values())})"


# --- Tree ---

class Tree:
    """A ranked tree: a symbol applied to children."""

    def __init__(self, symbol: str, children: List['Tree'] = None):
        self.symbol = symbol
        self.children = children or []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def height(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.height() for c in self.children)

    def subtrees(self) -> List['Tree']:
        """All subtrees including self."""
        result = [self]
        for c in self.children:
            result.extend(c.subtrees())
        return result

    def __eq__(self, other):
        if not isinstance(other, Tree):
            return False
        return self.symbol == other.symbol and self.children == other.children

    def __hash__(self):
        return hash((self.symbol, tuple(self.children)))

    def __repr__(self):
        if not self.children:
            return self.symbol
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.symbol}({args})"


def tree(symbol: str, *children) -> Tree:
    """Convenience constructor."""
    return Tree(symbol, list(children))


# --- Bottom-Up Tree Automaton ---

# A transition: (symbol, (q1, ..., qn)) -> set of states
# For constants: (symbol, ()) -> set of states

class BottomUpTreeAutomaton:
    """Nondeterministic bottom-up finite tree automaton.

    States are hashable objects (typically strings or ints).
    Transitions: f(q1, ..., qn) -> q  (bottom-up: children states determine parent state)
    """

    def __init__(self, alphabet: RankedAlphabet = None):
        self.alphabet = alphabet or RankedAlphabet()
        self.states: Set[str] = set()
        self.final_states: Set[str] = set()
        # transitions[(symbol_name, (q1, ..., qn))] = {q, ...}
        self.transitions: Dict[Tuple[str, Tuple], Set[str]] = {}

    def add_state(self, state: str, final: bool = False):
        self.states.add(state)
        if final:
            self.final_states.add(state)

    def add_transition(self, symbol: str, children_states: Tuple, target_state: str):
        """Add transition: symbol(children_states) -> target_state."""
        self.states.add(target_state)
        for s in children_states:
            self.states.add(s)
        key = (symbol, children_states)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(target_state)

    def get_targets(self, symbol: str, children_states: Tuple) -> Set[str]:
        """Get target states for a transition."""
        return self.transitions.get((symbol, children_states), set())

    def run(self, t: Tree) -> Set[str]:
        """Run the automaton bottom-up on a tree. Returns set of states at root."""
        if t.is_leaf():
            return self.get_targets(t.symbol, ())

        # Recursively compute children state sets
        children_state_sets = [self.run(c) for c in t.children]

        # For each combination of children states, collect target states
        result = set()
        for combo in itertools.product(*children_state_sets):
            result.update(self.get_targets(t.symbol, combo))
        return result

    def accepts(self, t: Tree) -> bool:
        """Check if tree is in the language."""
        return bool(self.run(t) & self.final_states)

    def is_empty(self) -> bool:
        """Check if the language is empty (no accepted tree exists).

        Bottom-up reachability: find which states are reachable,
        then check if any final state is reachable.
        """
        reachable = set()
        changed = True
        while changed:
            changed = False
            for (sym, children), targets in self.transitions.items():
                if all(c in reachable for c in children):
                    for t in targets:
                        if t not in reachable:
                            reachable.add(t)
                            changed = True
        return not bool(reachable & self.final_states)

    def witness(self, max_size: int = 100) -> Optional[Tree]:
        """Find a witness tree accepted by the automaton (BFS by size)."""
        if self.is_empty():
            return None

        # Build reachable states with witness trees
        # state -> tree that reaches that state
        witnesses: Dict[str, Tree] = {}
        changed = True
        while changed:
            changed = False
            for (sym, children), targets in self.transitions.items():
                if all(c in witnesses for c in children):
                    child_trees = [witnesses[c] for c in children]
                    t = Tree(sym, child_trees)
                    if t.size() <= max_size:
                        for tgt in targets:
                            if tgt not in witnesses:
                                witnesses[tgt] = t
                                changed = True
                                if tgt in self.final_states:
                                    return t
        # Check if we found any final state
        for f in self.final_states:
            if f in witnesses:
                return witnesses[f]
        return None

    def is_deterministic(self) -> bool:
        """Check if at most one target per transition."""
        return all(len(targets) <= 1 for targets in self.transitions.values())

    def determinize(self) -> 'BottomUpTreeAutomaton':
        """Determinize via subset construction."""
        det = BottomUpTreeAutomaton(self.alphabet)

        # State of deterministic automaton = frozenset of NFA states
        # Start with constants
        state_map: Dict[FrozenSet[str], str] = {}
        state_counter = [0]

        def get_det_state(state_set: FrozenSet[str]) -> str:
            if state_set in state_map:
                return state_map[state_set]
            name = f"d{state_counter[0]}"
            state_counter[0] += 1
            state_map[state_set] = name
            is_final = bool(state_set & self.final_states)
            det.add_state(name, final=is_final)
            return name

        # BFS over reachable deterministic states
        worklist = deque()

        # Process constants first
        const_states = {}
        for sym in self.alphabet.constants():
            targets = self.get_targets(sym.name, ())
            fs = frozenset(targets)
            if fs:
                det_state = get_det_state(fs)
                const_states[sym.name] = fs
                det.add_transition(sym.name, (), det_state)
                worklist.append(fs)

        processed = set()
        # Process higher-arity symbols
        while worklist:
            current = worklist.popleft()
            if current in processed:
                continue
            processed.add(current)

            # For each symbol with arity > 0, try all combinations of known state sets
            known_sets = list(state_map.keys())
            for sym in self.alphabet:
                if sym.arity == 0:
                    continue
                for combo in itertools.product(known_sets, repeat=sym.arity):
                    # Check if at least one element of combo is current or involves current
                    # Actually, we need all reachable combos, not just those involving current
                    targets = set()
                    for nfa_combo in itertools.product(*(set(c) for c in combo)):
                        targets.update(self.get_targets(sym.name, nfa_combo))
                    if targets:
                        fs = frozenset(targets)
                        det_children = tuple(state_map[c] for c in combo)
                        det_state = get_det_state(fs)
                        det.add_transition(sym.name, det_children, det_state)
                        if fs not in processed:
                            worklist.append(fs)

        return det

    def complete(self) -> 'BottomUpTreeAutomaton':
        """Make the automaton complete (total transition function).

        Add a sink state and transitions for all missing (symbol, children) combos.
        """
        result = BottomUpTreeAutomaton(self.alphabet)
        result.states = set(self.states)
        result.final_states = set(self.final_states)
        result.transitions = {k: set(v) for k, v in self.transitions.items()}

        sink = "__sink__"
        sink_needed = False
        all_states = list(self.states) + [sink]

        for sym in self.alphabet:
            if sym.arity == 0:
                if not result.get_targets(sym.name, ()):
                    result.add_transition(sym.name, (), sink)
                    sink_needed = True
            else:
                for combo in itertools.product(all_states, repeat=sym.arity):
                    if not result.get_targets(sym.name, combo):
                        result.add_transition(sym.name, combo, sink)
                        sink_needed = True

        if sink_needed:
            result.add_state(sink)

        return result

    def complement(self) -> 'BottomUpTreeAutomaton':
        """Complement: determinize, complete, then flip final states."""
        det = self.determinize()
        comp_det = det.complete()
        result = BottomUpTreeAutomaton(comp_det.alphabet)
        result.states = set(comp_det.states)
        result.final_states = comp_det.states - comp_det.final_states
        result.transitions = dict(comp_det.transitions)
        return result

    def enumerate_trees(self, max_size: int = 10, max_count: int = 100) -> List[Tree]:
        """Enumerate accepted trees up to a given size."""
        results = []
        self._enum_trees(max_size, max_count, results)
        return results

    def _enum_trees(self, max_size: int, max_count: int, results: List[Tree]):
        """Generate trees by size and check acceptance."""
        # Generate all trees up to max_size and filter
        for size in range(1, max_size + 1):
            if len(results) >= max_count:
                break
            for t in self._trees_of_size(size):
                if len(results) >= max_count:
                    break
                if self.accepts(t):
                    results.append(t)

    def _trees_of_size(self, size: int) -> List[Tree]:
        """Generate all trees of exactly the given size over the alphabet."""
        if size == 1:
            return [Tree(s.name) for s in self.alphabet.constants()]
        results = []
        for sym in self.alphabet:
            if sym.arity == 0:
                continue
            # Partition size-1 among sym.arity children (each >= 1)
            for partition in _partitions(size - 1, sym.arity):
                child_lists = [self._trees_of_size(p) for p in partition]
                for combo in itertools.product(*child_lists):
                    results.append(Tree(sym.name, list(combo)))
        return results

    def transition_count(self) -> int:
        return sum(len(targets) for targets in self.transitions.values())

    def __repr__(self):
        return (f"BUTA(states={len(self.states)}, final={len(self.final_states)}, "
                f"transitions={self.transition_count()})")


def _partitions(n: int, k: int) -> List[Tuple[int, ...]]:
    """Partition n into k parts, each >= 1."""
    if k == 1:
        return [(n,)]
    result = []
    for i in range(1, n - k + 2):
        for rest in _partitions(n - i, k - 1):
            result.append((i,) + rest)
    return result


# --- Top-Down Tree Automaton ---

class TopDownTreeAutomaton:
    """Nondeterministic top-down finite tree automaton.

    Transitions: (q, symbol) -> set of (q1, ..., qn) tuples
    """

    def __init__(self, alphabet: RankedAlphabet = None):
        self.alphabet = alphabet or RankedAlphabet()
        self.states: Set[str] = set()
        self.initial_states: Set[str] = set()
        # transitions[(state, symbol_name)] = {(q1, ..., qn), ...}
        self.transitions: Dict[Tuple[str, str], Set[Tuple]] = {}

    def add_state(self, state: str, initial: bool = False):
        self.states.add(state)
        if initial:
            self.initial_states.add(state)

    def add_transition(self, state: str, symbol: str, children_states: Tuple):
        """Add transition: state -> symbol(children_states)."""
        self.states.add(state)
        for s in children_states:
            self.states.add(s)
        key = (state, symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(children_states)

    def get_children_states(self, state: str, symbol: str) -> Set[Tuple]:
        return self.transitions.get((state, symbol), set())

    def accepts(self, t: Tree) -> bool:
        """Check if tree is accepted (exists initial state that processes entire tree)."""
        for init in self.initial_states:
            if self._run_from(init, t):
                return True
        return False

    def _run_from(self, state: str, t: Tree) -> bool:
        """Check if tree is accepted starting from given state."""
        children_options = self.get_children_states(state, t.symbol)
        for child_states in children_options:
            if len(child_states) != len(t.children):
                continue
            if all(self._run_from(cs, c) for cs, c in zip(child_states, t.children)):
                return True
        return False

    def to_bottom_up(self) -> BottomUpTreeAutomaton:
        """Convert to equivalent bottom-up automaton.

        TDTA transition: (q, f) -> (q1, ..., qn)
        becomes BUTA transition: f(q1, ..., qn) -> q
        Final states of BUTA = initial states of TDTA
        """
        buta = BottomUpTreeAutomaton(self.alphabet)
        for s in self.states:
            buta.add_state(s, final=(s in self.initial_states))
        for (state, sym), children_set in self.transitions.items():
            for children in children_set:
                buta.add_transition(sym, children, state)
        return buta

    def __repr__(self):
        trans_count = sum(len(v) for v in self.transitions.values())
        return (f"TDTA(states={len(self.states)}, initial={len(self.initial_states)}, "
                f"transitions={trans_count})")


# --- BUTA to TDTA conversion ---

def buta_to_tdta(buta: BottomUpTreeAutomaton) -> TopDownTreeAutomaton:
    """Convert bottom-up to top-down automaton.

    BUTA transition: f(q1, ..., qn) -> q
    becomes TDTA transition: (q, f) -> (q1, ..., qn)
    Initial states of TDTA = final states of BUTA
    """
    tdta = TopDownTreeAutomaton(buta.alphabet)
    for s in buta.states:
        tdta.add_state(s, initial=(s in buta.final_states))
    for (sym, children), targets in buta.transitions.items():
        for t in targets:
            tdta.add_transition(t, sym, children)
    return tdta


# --- Boolean Operations ---

def buta_union(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Union of two BUTAs (disjoint state rename + merge)."""
    result = BottomUpTreeAutomaton(a1.alphabet)

    # Rename states to avoid collision
    for s in a1.states:
        result.add_state(f"a_{s}", final=(s in a1.final_states))
    for s in a2.states:
        result.add_state(f"b_{s}", final=(s in a2.final_states))

    for (sym, children), targets in a1.transitions.items():
        new_children = tuple(f"a_{c}" for c in children)
        for t in targets:
            result.add_transition(sym, new_children, f"a_{t}")

    for (sym, children), targets in a2.transitions.items():
        new_children = tuple(f"b_{c}" for c in children)
        for t in targets:
            result.add_transition(sym, new_children, f"b_{t}")

    return result


def buta_intersection(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Intersection via product construction."""
    result = BottomUpTreeAutomaton(a1.alphabet)

    # Product states
    for s1 in a1.states:
        for s2 in a2.states:
            is_final = (s1 in a1.final_states) and (s2 in a2.final_states)
            result.add_state(f"{s1}_{s2}", final=is_final)

    # Product transitions
    for (sym1, children1), targets1 in a1.transitions.items():
        for (sym2, children2), targets2 in a2.transitions.items():
            if sym1 != sym2:
                continue
            if len(children1) != len(children2):
                continue
            # Product of children
            prod_children = tuple(f"{c1}_{c2}" for c1, c2 in zip(children1, children2))
            for t1 in targets1:
                for t2 in targets2:
                    result.add_transition(sym1, prod_children, f"{t1}_{t2}")

    return result


def buta_complement(a: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Complement (determinize + complete + flip finals)."""
    return a.complement()


def buta_difference(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Difference: L(a1) - L(a2) = L(a1) intersect L(complement(a2))."""
    comp = buta_complement(a2)
    return buta_intersection(a1, comp)


# --- Language Inclusion ---

def buta_is_subset(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> Tuple[bool, Optional[Tree]]:
    """Check if L(a1) subset L(a2). Returns (result, counterexample if False)."""
    diff = buta_difference(a1, a2)
    if diff.is_empty():
        return (True, None)
    witness = diff.witness()
    return (False, witness)


def buta_is_equivalent(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> Tuple[bool, Optional[Tree]]:
    """Check if L(a1) = L(a2). Returns (result, counterexample if not equal)."""
    sub1, w1 = buta_is_subset(a1, a2)
    if not sub1:
        return (False, w1)
    sub2, w2 = buta_is_subset(a2, a1)
    if not sub2:
        return (False, w2)
    return (True, None)


# --- Minimization ---

def buta_minimize(a: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Minimize a deterministic BUTA via partition refinement.

    Myhill-Nerode style: merge states with identical future behavior.
    """
    det = a if a.is_deterministic() else a.determinize()

    # Initial partition: final vs non-final
    final = frozenset(det.final_states)
    non_final = frozenset(det.states - det.final_states)
    partition = []
    if final:
        partition.append(final)
    if non_final:
        partition.append(non_final)

    if len(partition) <= 1:
        return det

    # Build reverse map: state -> partition index
    def state_to_class(state, part):
        for i, cls in enumerate(part):
            if state in cls:
                return i
        return -1

    changed = True
    while changed:
        changed = False
        new_partition = []
        for cls in partition:
            # Split this class based on transition signatures
            sig_groups: Dict[Any, Set[str]] = {}
            for state in cls:
                sig = _transition_signature(det, state, partition)
                sig_key = _hashable_sig(sig)
                if sig_key not in sig_groups:
                    sig_groups[sig_key] = set()
                sig_groups[sig_key].add(state)
            groups = list(sig_groups.values())
            if len(groups) > 1:
                changed = True
            new_partition.extend(frozenset(g) for g in groups)
        partition = new_partition

    # Build minimized automaton
    result = BottomUpTreeAutomaton(det.alphabet)
    class_rep = {}
    for cls in partition:
        rep = min(cls)  # canonical representative
        for s in cls:
            class_rep[s] = rep
        result.add_state(rep, final=(rep in det.final_states))

    for (sym, children), targets in det.transitions.items():
        new_children = tuple(class_rep.get(c, c) for c in children)
        for t in targets:
            new_t = class_rep.get(t, t)
            result.add_transition(sym, new_children, new_t)

    return result


def _transition_signature(det, state, partition):
    """Compute the transition signature of a state wrt current partition."""
    sig = {}
    for (sym, children), targets in det.transitions.items():
        if not targets:
            continue
        target = next(iter(targets))  # deterministic
        if target == state or state in children:
            # Find class indices for children and target
            child_classes = tuple(_find_class(c, partition) for c in children)
            target_class = _find_class(target, partition)
            key = (sym, child_classes)
            sig[key] = target_class
    return sig


def _find_class(state, partition):
    for i, cls in enumerate(partition):
        if state in cls:
            return i
    return -1


def _hashable_sig(sig):
    return tuple(sorted(sig.items()))


# --- Construction Helpers ---

def make_alphabet(*specs) -> RankedAlphabet:
    """Create alphabet from (name, arity) pairs."""
    alpha = RankedAlphabet()
    for name, arity in specs:
        alpha.add(name, arity)
    return alpha


def make_buta(alphabet: RankedAlphabet, states: List[str],
              final: List[str], transitions: List[Tuple]) -> BottomUpTreeAutomaton:
    """Convenience constructor.

    transitions: list of (symbol, (child_states...), target_state)
    """
    buta = BottomUpTreeAutomaton(alphabet)
    for s in states:
        buta.add_state(s, final=(s in final))
    for sym, children, target in transitions:
        buta.add_transition(sym, children, target)
    return buta


def make_tdta(alphabet: RankedAlphabet, states: List[str],
              initial: List[str], transitions: List[Tuple]) -> TopDownTreeAutomaton:
    """Convenience constructor.

    transitions: list of (state, symbol, (child_states...))
    """
    tdta = TopDownTreeAutomaton(alphabet)
    for s in states:
        tdta.add_state(s, initial=(s in initial))
    for state, sym, children in transitions:
        tdta.add_transition(state, sym, children)
    return tdta


# --- Tree Pattern Matching ---

class TreePattern:
    """A pattern for matching trees. Supports:
    - Exact symbol match: TreePattern("f", [p1, p2])
    - Wildcard: TreePattern(None)  -- matches any tree
    - Variable capture: TreePattern(None, var="x")
    """

    def __init__(self, symbol: str = None, children: List['TreePattern'] = None,
                 var: str = None):
        self.symbol = symbol
        self.children = children or []
        self.var = var

    def match(self, t: Tree) -> Optional[Dict[str, Tree]]:
        """Try to match tree against pattern. Returns bindings or None."""
        bindings = {}
        if self._match(t, bindings):
            return bindings
        return None

    def _match(self, t: Tree, bindings: Dict[str, Tree]) -> bool:
        if self.var is not None:
            if self.var in bindings:
                return bindings[self.var] == t
            bindings[self.var] = t
            # If symbol is also specified, check it
            if self.symbol is not None and self.symbol != t.symbol:
                del bindings[self.var]
                return False
            return True
        if self.symbol is None:
            return True  # wildcard
        if self.symbol != t.symbol:
            return False
        if len(self.children) != len(t.children):
            return False
        return all(cp._match(ct, bindings) for cp, ct in zip(self.children, t.children))

    def __repr__(self):
        if self.var:
            return f"?{self.var}"
        if self.symbol is None:
            return "_"
        if not self.children:
            return self.symbol
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.symbol}({args})"


def pat(symbol: str = None, *children, var: str = None) -> TreePattern:
    """Convenience pattern constructor."""
    return TreePattern(symbol, list(children), var=var)


# --- Term Rewriting ---

@dataclass
class RewriteRule:
    """A term rewrite rule: lhs -> rhs (with variable substitution)."""
    lhs: TreePattern
    rhs_builder: Any  # callable: bindings -> Tree

    def apply(self, t: Tree) -> Optional[Tree]:
        """Try to apply rule at root of tree."""
        bindings = self.lhs.match(t)
        if bindings is None:
            return None
        return self.rhs_builder(bindings)


class TermRewriteSystem:
    """A set of rewrite rules with strategy."""

    def __init__(self, rules: List[RewriteRule] = None):
        self.rules = rules or []

    def add_rule(self, lhs: TreePattern, rhs_builder):
        self.rules.append(RewriteRule(lhs, rhs_builder))

    def rewrite_step(self, t: Tree) -> Optional[Tree]:
        """Apply one rewrite step (leftmost-outermost)."""
        # Try at root first
        for rule in self.rules:
            result = rule.apply(t)
            if result is not None:
                return result
        # Try in children
        for i, child in enumerate(t.children):
            result = self.rewrite_step(child)
            if result is not None:
                new_children = list(t.children)
                new_children[i] = result
                return Tree(t.symbol, new_children)
        return None

    def normalize(self, t: Tree, max_steps: int = 1000) -> Tree:
        """Rewrite to normal form."""
        current = t
        for _ in range(max_steps):
            next_t = self.rewrite_step(current)
            if next_t is None:
                return current
            current = next_t
        return current

    def is_normal_form(self, t: Tree) -> bool:
        return self.rewrite_step(t) is None


# --- BUTA from Patterns ---

def buta_from_patterns(alphabet: RankedAlphabet,
                       accept_patterns: List[TreePattern],
                       max_depth: int = 5) -> BottomUpTreeAutomaton:
    """Build a BUTA that accepts trees matching any of the given patterns.

    This is an approximation: builds automaton recognizing trees up to max_depth
    that match any pattern.
    """
    buta = BottomUpTreeAutomaton(alphabet)

    # State "match" = a subtree that participates in an accepted tree
    # State "any" = any subtree (wildcard acceptance)
    buta.add_state("accept", final=True)
    buta.add_state("any")

    # "any" accepts everything
    for sym in alphabet:
        if sym.arity == 0:
            buta.add_transition(sym.name, (), "any")
        else:
            for combo in itertools.product(["any"], repeat=sym.arity):
                buta.add_transition(sym.name, combo, "any")

    # For each pattern, add transitions recognizing it
    for i, pattern in enumerate(accept_patterns):
        _add_pattern_transitions(buta, pattern, "accept", alphabet, f"p{i}")

    return buta


def _add_pattern_transitions(buta, pattern, target_state, alphabet, prefix):
    """Add transitions to recognize a pattern."""
    if pattern.symbol is None:
        # Wildcard or variable -- accept anything into target state
        for sym in alphabet:
            if sym.arity == 0:
                buta.add_transition(sym.name, (), target_state)
            else:
                for combo in itertools.product(["any", target_state], repeat=sym.arity):
                    buta.add_transition(sym.name, combo, target_state)
        return

    if not pattern.children:
        # Leaf pattern
        buta.add_transition(pattern.symbol, (), target_state)
        return

    # Create intermediate states for children
    child_states = []
    for j, child_pat in enumerate(pattern.children):
        child_state = f"{prefix}_c{j}"
        buta.add_state(child_state)
        child_states.append(child_state)
        _add_pattern_transitions(buta, child_pat, child_state, alphabet, f"{prefix}_c{j}")

    buta.add_transition(pattern.symbol, tuple(child_states), target_state)


# --- Analysis Utilities ---

def tree_language_size(buta: BottomUpTreeAutomaton, max_size: int = 10) -> int:
    """Count accepted trees up to given size."""
    return len(buta.enumerate_trees(max_size=max_size, max_count=10000))


def buta_stats(buta: BottomUpTreeAutomaton) -> Dict:
    """Statistics about a BUTA."""
    return {
        "states": len(buta.states),
        "final_states": len(buta.final_states),
        "transitions": buta.transition_count(),
        "deterministic": buta.is_deterministic(),
        "empty": buta.is_empty(),
    }


def compare_butas(a1: BottomUpTreeAutomaton, a2: BottomUpTreeAutomaton) -> Dict:
    """Compare two BUTAs: subset/equivalence/witnesses."""
    sub12, w12 = buta_is_subset(a1, a2)
    sub21, w21 = buta_is_subset(a2, a1)
    return {
        "a1_subset_a2": sub12,
        "a2_subset_a1": sub21,
        "equivalent": sub12 and sub21,
        "a1_not_in_a2_witness": w12,
        "a2_not_in_a1_witness": w21,
        "a1_stats": buta_stats(a1),
        "a2_stats": buta_stats(a2),
    }


# --- XML/Schema-like validation ---

def schema_automaton(element_specs: Dict[str, List[List[str]]]) -> BottomUpTreeAutomaton:
    """Build a BUTA from a simplified XML-like schema.

    element_specs: maps element name -> list of valid children configurations.
    Each configuration is a list of child element names (order matters).

    Example:
      {"doc": [["para", "para"], ["para"]],
       "para": [["text"], []],
       "text": [[]]}
    means doc can have 1 or 2 paras, para can have text or be empty, text is leaf.
    """
    # Determine arities
    alpha = RankedAlphabet()
    arities = {}
    for name, configs in element_specs.items():
        for config in configs:
            arity = len(config)
            if name not in arities:
                arities[name] = set()
            arities[name].add(arity)

    # Add symbols for each (name, arity) pair
    sym_names = {}
    for name, arity_set in arities.items():
        for arity in arity_set:
            sym_name = f"{name}_{arity}" if len(arity_set) > 1 else name
            alpha.add(sym_name, arity)
            sym_names[(name, arity)] = sym_name

    buta = BottomUpTreeAutomaton(alpha)

    # Each element name is a state
    for name in element_specs:
        buta.add_state(name)

    # Root elements are final
    # Convention: first element in specs is the root
    root = list(element_specs.keys())[0]
    buta.final_states.add(root)

    # Add transitions
    for name, configs in element_specs.items():
        for config in configs:
            arity = len(config)
            sym_name = sym_names.get((name, arity), name)
            children = tuple(config)
            buta.add_transition(sym_name, children, name)

    return buta


# --- High-Level APIs ---

def check_tree_membership(buta: BottomUpTreeAutomaton, t: Tree) -> Dict:
    """Check if tree is in language with details."""
    states = buta.run(t)
    accepted = bool(states & buta.final_states)
    return {
        "accepted": accepted,
        "root_states": states,
        "final_intersection": states & buta.final_states,
    }


def check_language_emptiness(buta: BottomUpTreeAutomaton) -> Dict:
    """Check emptiness with witness."""
    empty = buta.is_empty()
    witness = None if empty else buta.witness()
    return {
        "empty": empty,
        "witness": witness,
    }


def check_language_inclusion(a1: BottomUpTreeAutomaton,
                              a2: BottomUpTreeAutomaton) -> Dict:
    """Check L(a1) subset L(a2) with counterexample."""
    result, counter = buta_is_subset(a1, a2)
    return {
        "included": result,
        "counterexample": counter,
    }


def check_language_equivalence(a1: BottomUpTreeAutomaton,
                                a2: BottomUpTreeAutomaton) -> Dict:
    """Check L(a1) = L(a2) with counterexample."""
    result, counter = buta_is_equivalent(a1, a2)
    return {
        "equivalent": result,
        "counterexample": counter,
    }
