"""
V081: Symbolic Automata
======================

Automata with symbolic transitions -- predicates on characters instead of
concrete character labels. Supports infinite alphabets efficiently.

Key concepts:
- Effective Boolean Algebra (EBA): predicates with conjunction, disjunction,
  negation, satisfiability checking, and enumeration
- Symbolic Finite Automaton (SFA): NFA/DFA with predicate-labeled transitions
- Boolean closure: intersection, union, complement, difference
- Minimization via symbolic partition refinement
- Equivalence checking via product construction

This connects to A1's string-processing stack (C087 suffix array, C101 suffix
automaton, C113 suffix tree, C114 Aho-Corasick) and extends formal methods to
string constraint reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum
import itertools


# ============================================================================
# Effective Boolean Algebra (EBA)
# ============================================================================

class PredKind(Enum):
    TRUE = "true"
    FALSE = "false"
    CHAR = "char"          # exact character match
    RANGE = "range"        # character range [lo, hi]
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True)
class Pred:
    """Predicate over characters (elements of the alphabet)."""
    kind: PredKind
    char: Optional[str] = None       # for CHAR
    lo: Optional[str] = None         # for RANGE
    hi: Optional[str] = None         # for RANGE
    left: Optional['Pred'] = None    # for AND, OR
    right: Optional['Pred'] = None   # for AND, OR
    child: Optional['Pred'] = None   # for NOT

    def __repr__(self):
        if self.kind == PredKind.TRUE:
            return "T"
        if self.kind == PredKind.FALSE:
            return "F"
        if self.kind == PredKind.CHAR:
            return repr(self.char)
        if self.kind == PredKind.RANGE:
            return f"[{self.lo}-{self.hi}]"
        if self.kind == PredKind.AND:
            return f"({self.left} & {self.right})"
        if self.kind == PredKind.OR:
            return f"({self.left} | {self.right})"
        if self.kind == PredKind.NOT:
            return f"~{self.child}"
        return f"Pred({self.kind})"


# Predicate constructors
def PTrue():
    return Pred(PredKind.TRUE)

def PFalse():
    return Pred(PredKind.FALSE)

def PChar(c):
    return Pred(PredKind.CHAR, char=c)

def PRange(lo, hi):
    if lo > hi:
        return PFalse()
    if lo == hi:
        return PChar(lo)
    return Pred(PredKind.RANGE, lo=lo, hi=hi)

def PAnd(a, b):
    if a.kind == PredKind.FALSE or b.kind == PredKind.FALSE:
        return PFalse()
    if a.kind == PredKind.TRUE:
        return b
    if b.kind == PredKind.TRUE:
        return a
    if a == b:
        return a
    return Pred(PredKind.AND, left=a, right=b)

def POr(a, b):
    if a.kind == PredKind.TRUE or b.kind == PredKind.TRUE:
        return PTrue()
    if a.kind == PredKind.FALSE:
        return b
    if b.kind == PredKind.FALSE:
        return a
    if a == b:
        return a
    return Pred(PredKind.OR, left=a, right=b)

def PNot(p):
    if p.kind == PredKind.TRUE:
        return PFalse()
    if p.kind == PredKind.FALSE:
        return PTrue()
    if p.kind == PredKind.NOT:
        return p.child
    return Pred(PredKind.NOT, child=p)


class CharAlgebra:
    """Effective Boolean Algebra over characters (single chars, printable ASCII)."""

    def __init__(self, alphabet=None):
        """alphabet: string of characters forming the universe. Default: printable ASCII."""
        if alphabet is None:
            self.alphabet = [chr(c) for c in range(32, 127)]
        else:
            self.alphabet = sorted(set(alphabet))

    def evaluate(self, pred, char):
        """Evaluate predicate on a concrete character."""
        if pred.kind == PredKind.TRUE:
            return True
        if pred.kind == PredKind.FALSE:
            return False
        if pred.kind == PredKind.CHAR:
            return char == pred.char
        if pred.kind == PredKind.RANGE:
            return pred.lo <= char <= pred.hi
        if pred.kind == PredKind.AND:
            return self.evaluate(pred.left, char) and self.evaluate(pred.right, char)
        if pred.kind == PredKind.OR:
            return self.evaluate(pred.left, char) or self.evaluate(pred.right, char)
        if pred.kind == PredKind.NOT:
            return not self.evaluate(pred.child, char)
        raise ValueError(f"Unknown predicate kind: {pred.kind}")

    def is_satisfiable(self, pred):
        """Check if any character satisfies the predicate."""
        if pred.kind == PredKind.TRUE:
            return True
        if pred.kind == PredKind.FALSE:
            return False
        if pred.kind == PredKind.CHAR:
            return pred.char in self.alphabet
        if pred.kind == PredKind.RANGE:
            return any(pred.lo <= c <= pred.hi for c in self.alphabet)
        # General case: enumerate
        return any(self.evaluate(pred, c) for c in self.alphabet)

    def witness(self, pred):
        """Find a character satisfying the predicate, or None."""
        if pred.kind == PredKind.FALSE:
            return None
        if pred.kind == PredKind.TRUE:
            return self.alphabet[0] if self.alphabet else None
        if pred.kind == PredKind.CHAR:
            return pred.char if pred.char in self.alphabet else None
        for c in self.alphabet:
            if self.evaluate(pred, c):
                return c
        return None

    def enumerate(self, pred):
        """List all characters satisfying the predicate."""
        return [c for c in self.alphabet if self.evaluate(pred, c)]

    def is_equivalent(self, a, b):
        """Check if two predicates accept the same set of characters."""
        diff = PAnd(a, PNot(b))
        if self.is_satisfiable(diff):
            return False
        diff2 = PAnd(b, PNot(a))
        return not self.is_satisfiable(diff2)

    def simplify(self, pred):
        """Basic predicate simplification."""
        if pred.kind in (PredKind.TRUE, PredKind.FALSE, PredKind.CHAR, PredKind.RANGE):
            return pred
        if pred.kind == PredKind.NOT:
            child = self.simplify(pred.child)
            return PNot(child)
        if pred.kind == PredKind.AND:
            left = self.simplify(pred.left)
            right = self.simplify(pred.right)
            return PAnd(left, right)
        if pred.kind == PredKind.OR:
            left = self.simplify(pred.left)
            right = self.simplify(pred.right)
            return POr(left, right)
        return pred


# ============================================================================
# Integer Algebra (for testing with integer alphabets)
# ============================================================================

class IntAlgebra:
    """Effective Boolean Algebra over integers in a finite range."""

    def __init__(self, lo=0, hi=255):
        self.lo = lo
        self.hi = hi
        self.alphabet = list(range(lo, hi + 1))

    def evaluate(self, pred, val):
        if pred.kind == PredKind.TRUE:
            return True
        if pred.kind == PredKind.FALSE:
            return False
        if pred.kind == PredKind.CHAR:
            return val == pred.char
        if pred.kind == PredKind.RANGE:
            return pred.lo <= val <= pred.hi
        if pred.kind == PredKind.AND:
            return self.evaluate(pred.left, val) and self.evaluate(pred.right, val)
        if pred.kind == PredKind.OR:
            return self.evaluate(pred.left, val) or self.evaluate(pred.right, val)
        if pred.kind == PredKind.NOT:
            return not self.evaluate(pred.child, val)
        raise ValueError(f"Unknown predicate kind: {pred.kind}")

    def is_satisfiable(self, pred):
        if pred.kind == PredKind.TRUE:
            return True
        if pred.kind == PredKind.FALSE:
            return False
        if pred.kind == PredKind.CHAR:
            return self.lo <= pred.char <= self.hi
        if pred.kind == PredKind.RANGE:
            return pred.lo <= self.hi and pred.hi >= self.lo
        return any(self.evaluate(pred, v) for v in self.alphabet)

    def witness(self, pred):
        if pred.kind == PredKind.FALSE:
            return None
        for v in self.alphabet:
            if self.evaluate(pred, v):
                return v
        return None

    def enumerate(self, pred):
        return [v for v in self.alphabet if self.evaluate(pred, v)]

    def is_equivalent(self, a, b):
        diff = PAnd(a, PNot(b))
        if self.is_satisfiable(diff):
            return False
        return not self.is_satisfiable(PAnd(b, PNot(a)))


# ============================================================================
# Symbolic Finite Automaton (SFA)
# ============================================================================

@dataclass
class SFATransition:
    """A symbolic transition: from state, predicate guard, to state."""
    src: int
    pred: Pred
    dst: int

    def __repr__(self):
        return f"{self.src} --{self.pred}--> {self.dst}"


@dataclass
class SFA:
    """
    Symbolic Finite Automaton.

    States are integers. Transitions are predicate-guarded.
    Can be deterministic or nondeterministic.
    """
    states: set
    initial: int
    accepting: set
    transitions: list  # list of SFATransition
    algebra: Any       # EBA instance (CharAlgebra or IntAlgebra)

    def __post_init__(self):
        self._trans_from = {}
        for t in self.transitions:
            self._trans_from.setdefault(t.src, []).append(t)

    def _rebuild_index(self):
        self._trans_from = {}
        for t in self.transitions:
            self._trans_from.setdefault(t.src, []).append(t)

    def accepts(self, word):
        """Check if the SFA accepts the given word (list/string of symbols)."""
        current = frozenset([self.initial])
        for sym in word:
            next_states = set()
            for s in current:
                for t in self._trans_from.get(s, []):
                    if self.algebra.evaluate(t.pred, sym):
                        next_states.add(t.dst)
            if not next_states:
                return False
            current = frozenset(next_states)
        return bool(current & self.accepting)

    def is_deterministic(self):
        """Check if the SFA is deterministic: at most one enabled transition per state+symbol."""
        for state in self.states:
            trans = self._trans_from.get(state, [])
            for i, t1 in enumerate(trans):
                for t2 in trans[i+1:]:
                    overlap = PAnd(t1.pred, t2.pred)
                    if self.algebra.is_satisfiable(overlap):
                        return False
        return True

    def is_empty(self):
        """Check if the SFA accepts no strings (language is empty)."""
        # BFS from initial to any accepting state
        visited = set()
        queue = [self.initial]
        visited.add(self.initial)
        while queue:
            s = queue.pop(0)
            if s in self.accepting:
                return False
            for t in self._trans_from.get(s, []):
                if self.algebra.is_satisfiable(t.pred) and t.dst not in visited:
                    visited.add(t.dst)
                    queue.append(t.dst)
        return True

    def accepted_word(self):
        """Find a word accepted by the SFA, or None if empty."""
        # BFS, tracking path
        visited = set()
        queue = [(self.initial, [])]
        visited.add(self.initial)
        while queue:
            s, path = queue.pop(0)
            if s in self.accepting:
                return path
            for t in self._trans_from.get(s, []):
                w = self.algebra.witness(t.pred)
                if w is not None and t.dst not in visited:
                    visited.add(t.dst)
                    queue.append((t.dst, path + [w]))
        return None

    def reachable_states(self):
        """Find all states reachable from initial."""
        visited = set()
        queue = [self.initial]
        visited.add(self.initial)
        while queue:
            s = queue.pop(0)
            for t in self._trans_from.get(s, []):
                if self.algebra.is_satisfiable(t.pred) and t.dst not in visited:
                    visited.add(t.dst)
                    queue.append(t.dst)
        return visited

    def trim(self):
        """Remove unreachable states and dead-end states."""
        reachable = self.reachable_states()
        # Reverse reachability from accepting states
        rev_trans = {}
        for t in self.transitions:
            if self.algebra.is_satisfiable(t.pred):
                rev_trans.setdefault(t.dst, []).append(t.src)
        productive = set()
        queue = [s for s in self.accepting if s in reachable]
        productive.update(queue)
        while queue:
            s = queue.pop(0)
            for src in rev_trans.get(s, []):
                if src not in productive and src in reachable:
                    productive.add(src)
                    queue.append(src)
        keep = reachable & productive
        if self.initial not in keep:
            keep.add(self.initial)
        new_trans = [t for t in self.transitions
                     if t.src in keep and t.dst in keep and self.algebra.is_satisfiable(t.pred)]
        return SFA(
            states=keep,
            initial=self.initial,
            accepting=self.accepting & keep,
            transitions=new_trans,
            algebra=self.algebra
        )

    def determinize(self):
        """Convert NFA to DFA via subset construction with predicate partitioning."""
        algebra = self.algebra

        # State mapping: frozenset -> int
        state_map = {}
        next_id = 0

        def get_id(ss):
            nonlocal next_id
            if ss not in state_map:
                state_map[ss] = next_id
                next_id += 1
            return state_map[ss]

        init_set = frozenset([self.initial])
        init_id = get_id(init_set)

        dfa_states = set()
        dfa_accepting = set()
        dfa_trans = []
        queue = [init_set]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            current_id = get_id(current)
            dfa_states.add(current_id)

            if current & self.accepting:
                dfa_accepting.add(current_id)

            # Collect all outgoing predicates from this state set
            all_preds = []
            for s in current:
                for t in self._trans_from.get(s, []):
                    all_preds.append(t)

            if not all_preds:
                continue

            # Compute minterms (maximal satisfiable conjunctions of pred/~pred)
            minterms = self._compute_minterms([t.pred for t in all_preds])

            for minterm in minterms:
                if not algebra.is_satisfiable(minterm):
                    continue
                # Compute target state set for this minterm
                target = set()
                for s in current:
                    for t in self._trans_from.get(s, []):
                        if algebra.is_satisfiable(PAnd(minterm, t.pred)):
                            target.add(t.dst)
                if target:
                    target_frozen = frozenset(target)
                    target_id = get_id(target_frozen)
                    dfa_trans.append(SFATransition(current_id, minterm, target_id))
                    if target_frozen not in visited:
                        queue.append(target_frozen)

        return SFA(
            states=dfa_states,
            initial=init_id,
            accepting=dfa_accepting,
            transitions=dfa_trans,
            algebra=algebra
        )

    def _compute_minterms(self, predicates):
        """Compute minterms: maximal satisfiable conjunctions of pred_i/~pred_i."""
        if not predicates:
            return [PTrue()]

        # Remove duplicates
        unique_preds = list(dict.fromkeys(predicates))

        # For small pred sets, enumerate all combinations
        if len(unique_preds) > 10:
            # Fallback: just use individual predicates and their negations
            return unique_preds + [PNot(p) for p in unique_preds]

        minterms = []
        for bits in range(1 << len(unique_preds)):
            conj = PTrue()
            for i, p in enumerate(unique_preds):
                if bits & (1 << i):
                    conj = PAnd(conj, p)
                else:
                    conj = PAnd(conj, PNot(p))
            if self.algebra.is_satisfiable(conj):
                minterms.append(conj)
        return minterms

    def minimize(self):
        """Minimize a deterministic SFA via symbolic partition refinement."""
        if not self.is_deterministic():
            return self.determinize().minimize()

        # Start: trim first
        sfa = self.trim()

        # Initial partition: accepting vs non-accepting
        accepting = sfa.accepting & sfa.states
        non_accepting = sfa.states - accepting
        partition = []
        if accepting:
            partition.append(frozenset(accepting))
        if non_accepting:
            partition.append(frozenset(non_accepting))
        if not partition:
            return sfa

        # State -> block index
        def state_block(s, part):
            for i, block in enumerate(part):
                if s in block:
                    return i
            return -1

        changed = True
        while changed:
            changed = False
            new_partition = []
            for block in partition:
                if len(block) <= 1:
                    new_partition.append(block)
                    continue
                # Try to split this block
                # Pick a representative
                rep = next(iter(block))
                same = {rep}
                diff_groups = {}
                for s in block:
                    if s == rep:
                        continue
                    # Check if s and rep have same transition behavior
                    # (go to same block for same predicates)
                    if self._same_transition_behavior(s, rep, sfa, partition):
                        same.add(s)
                    else:
                        # Group by signature
                        sig = self._transition_signature(s, sfa, partition)
                        sig_key = str(sig)
                        if sig_key not in diff_groups:
                            diff_groups[sig_key] = set()
                        diff_groups[sig_key].add(s)
                if diff_groups:
                    changed = True
                    new_partition.append(frozenset(same))
                    for group in diff_groups.values():
                        new_partition.append(frozenset(group))
                else:
                    new_partition.append(block)
            partition = new_partition

        # Build minimized SFA
        block_map = {}
        for i, block in enumerate(partition):
            for s in block:
                block_map[s] = i

        min_states = set(range(len(partition)))
        min_initial = block_map.get(sfa.initial, 0)
        min_accepting = set()
        for i, block in enumerate(partition):
            if block & sfa.accepting:
                min_accepting.add(i)

        # Merge transitions
        seen_trans = set()
        min_trans = []
        for t in sfa.transitions:
            src_block = block_map.get(t.src)
            dst_block = block_map.get(t.dst)
            if src_block is None or dst_block is None:
                continue
            key = (src_block, dst_block)
            if key not in seen_trans:
                seen_trans.add(key)
                # Collect all predicates for this block pair
                preds = []
                for t2 in sfa.transitions:
                    if block_map.get(t2.src) == src_block and block_map.get(t2.dst) == dst_block:
                        preds.append(t2.pred)
                merged = preds[0]
                for p in preds[1:]:
                    merged = POr(merged, p)
                min_trans.append(SFATransition(src_block, merged, dst_block))

        return SFA(
            states=min_states,
            initial=min_initial,
            accepting=min_accepting,
            transitions=min_trans,
            algebra=sfa.algebra
        )

    def _same_transition_behavior(self, s1, s2, sfa, partition):
        """Check if two states have the same transition behavior w.r.t. partition."""
        sig1 = self._transition_signature(s1, sfa, partition)
        sig2 = self._transition_signature(s2, sfa, partition)
        return sig1 == sig2

    def _transition_signature(self, state, sfa, partition):
        """Compute transition signature: (target_block, pred) pairs."""
        block_map = {}
        for i, block in enumerate(partition):
            for s in block:
                block_map[s] = i
        sig = []
        for t in sfa._trans_from.get(state, []):
            target_block = block_map.get(t.dst, -1)
            sig.append((target_block, repr(t.pred)))
        sig.sort()
        return tuple(sig)

    def count_states(self):
        return len(self.states)

    def count_transitions(self):
        return len(self.transitions)


# ============================================================================
# SFA Boolean Operations
# ============================================================================

def sfa_intersection(a, b):
    """Product construction: intersection of two SFAs."""
    assert type(a.algebra) == type(b.algebra), "Algebras must be same type"
    algebra = a.algebra

    state_map = {}
    next_id = 0

    def get_id(pair):
        nonlocal next_id
        if pair not in state_map:
            state_map[pair] = next_id
            next_id += 1
        return state_map[pair]

    init_pair = (a.initial, b.initial)
    init_id = get_id(init_pair)

    states = set()
    accepting = set()
    transitions = []
    queue = [init_pair]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        pid = get_id(pair)
        states.add(pid)

        sa, sb = pair
        if sa in a.accepting and sb in b.accepting:
            accepting.add(pid)

        for ta in a._trans_from.get(sa, []):
            for tb in b._trans_from.get(sb, []):
                conj = PAnd(ta.pred, tb.pred)
                if algebra.is_satisfiable(conj):
                    target = (ta.dst, tb.dst)
                    tid = get_id(target)
                    transitions.append(SFATransition(pid, conj, tid))
                    if target not in visited:
                        queue.append(target)

    return SFA(
        states=states,
        initial=init_id,
        accepting=accepting,
        transitions=transitions,
        algebra=algebra
    )


def sfa_union(a, b):
    """Union of two SFAs via product construction on completed automata."""
    assert type(a.algebra) == type(b.algebra)
    algebra = a.algebra

    # Make both complete so product construction works correctly
    ac = _make_complete(a if a.is_deterministic() else a.determinize())
    bc = _make_complete(b if b.is_deterministic() else b.determinize())

    state_map = {}
    next_id = 0

    def get_id(pair):
        nonlocal next_id
        if pair not in state_map:
            state_map[pair] = next_id
            next_id += 1
        return state_map[pair]

    init_pair = (ac.initial, bc.initial)
    init_id = get_id(init_pair)

    states = set()
    accepting = set()
    transitions = []
    queue = [init_pair]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        pid = get_id(pair)
        states.add(pid)

        sa, sb = pair
        if sa in ac.accepting or sb in bc.accepting:
            accepting.add(pid)

        for ta in ac._trans_from.get(sa, []):
            for tb in bc._trans_from.get(sb, []):
                conj = PAnd(ta.pred, tb.pred)
                if algebra.is_satisfiable(conj):
                    target = (ta.dst, tb.dst)
                    tid = get_id(target)
                    transitions.append(SFATransition(pid, conj, tid))
                    if target not in visited:
                        queue.append(target)

    return SFA(
        states=states,
        initial=init_id,
        accepting=accepting,
        transitions=transitions,
        algebra=algebra
    )


def sfa_complement(sfa):
    """Complement of an SFA. Requires deterministic + complete."""
    dfa = sfa if sfa.is_deterministic() else sfa.determinize()
    # Make complete (add sink state for missing transitions)
    dfa = _make_complete(dfa)
    return SFA(
        states=dfa.states.copy(),
        initial=dfa.initial,
        accepting=dfa.states - dfa.accepting,
        transitions=list(dfa.transitions),
        algebra=dfa.algebra
    )


def _make_complete(sfa):
    """Add a sink state for any missing transitions."""
    algebra = sfa.algebra
    sink = max(sfa.states) + 1 if sfa.states else 0
    need_sink = False
    new_trans = list(sfa.transitions)

    for state in sfa.states:
        # Compute union of all outgoing predicates
        trans = sfa._trans_from.get(state, [])
        if not trans:
            # No transitions at all -> sink on everything
            new_trans.append(SFATransition(state, PTrue(), sink))
            need_sink = True
        else:
            covered = trans[0].pred
            for t in trans[1:]:
                covered = POr(covered, t.pred)
            uncovered = PAnd(PTrue(), PNot(covered))
            if algebra.is_satisfiable(uncovered):
                new_trans.append(SFATransition(state, uncovered, sink))
                need_sink = True

    if need_sink:
        new_states = sfa.states | {sink}
        new_trans.append(SFATransition(sink, PTrue(), sink))
    else:
        new_states = sfa.states.copy()

    return SFA(
        states=new_states,
        initial=sfa.initial,
        accepting=sfa.accepting.copy(),
        transitions=new_trans,
        algebra=algebra
    )


def sfa_difference(a, b):
    """Difference: L(a) - L(b) = L(a) & ~L(b)."""
    comp_b = sfa_complement(b)
    return sfa_intersection(a, comp_b)


def sfa_is_equivalent(a, b):
    """Check if two SFAs accept the same language."""
    diff1 = sfa_difference(a, b)
    if not diff1.is_empty():
        return False
    diff2 = sfa_difference(b, a)
    return diff2.is_empty()


def sfa_is_subset(a, b):
    """Check if L(a) is a subset of L(b)."""
    diff = sfa_difference(a, b)
    return diff.is_empty()


# ============================================================================
# SFA Construction Helpers
# ============================================================================

def sfa_from_string(s, algebra=None):
    """Build an SFA that accepts exactly the given string."""
    if algebra is None:
        algebra = CharAlgebra()
    states = set(range(len(s) + 1))
    transitions = []
    for i, c in enumerate(s):
        transitions.append(SFATransition(i, PChar(c), i + 1))
    return SFA(
        states=states,
        initial=0,
        accepting={len(s)},
        transitions=transitions,
        algebra=algebra
    )


def sfa_from_char_class(chars, algebra=None):
    """Build an SFA that accepts any single character from the given set."""
    if algebra is None:
        algebra = CharAlgebra()
    pred = PFalse()
    for c in chars:
        pred = POr(pred, PChar(c))
    return SFA(
        states={0, 1},
        initial=0,
        accepting={1},
        transitions=[SFATransition(0, pred, 1)],
        algebra=algebra
    )


def sfa_from_range(lo, hi, algebra=None):
    """Build an SFA that accepts any single character in [lo, hi]."""
    if algebra is None:
        algebra = CharAlgebra()
    return SFA(
        states={0, 1},
        initial=0,
        accepting={1},
        transitions=[SFATransition(0, PRange(lo, hi), 1)],
        algebra=algebra
    )


def sfa_any_char(algebra=None):
    """Build an SFA accepting any single character."""
    if algebra is None:
        algebra = CharAlgebra()
    return SFA(
        states={0, 1},
        initial=0,
        accepting={1},
        transitions=[SFATransition(0, PTrue(), 1)],
        algebra=algebra
    )


def sfa_epsilon(algebra=None):
    """Build an SFA accepting only the empty string."""
    if algebra is None:
        algebra = CharAlgebra()
    return SFA(
        states={0},
        initial=0,
        accepting={0},
        transitions=[],
        algebra=algebra
    )


def sfa_empty(algebra=None):
    """Build an SFA accepting nothing."""
    if algebra is None:
        algebra = CharAlgebra()
    return SFA(
        states={0},
        initial=0,
        accepting=set(),
        transitions=[],
        algebra=algebra
    )


def sfa_concat(a, b):
    """Concatenation of two SFAs."""
    assert type(a.algebra) == type(b.algebra)
    algebra = a.algebra

    # Offset b's states
    offset = max(a.states) + 1 if a.states else 0
    b_state_map = {s: s + offset for s in b.states}

    states = a.states | {b_state_map[s] for s in b.states}
    transitions = list(a.transitions)

    # Add b's transitions with offset
    for t in b.transitions:
        transitions.append(SFATransition(b_state_map[t.src], t.pred, b_state_map[t.dst]))

    # Epsilon transitions from a's accepting to b's initial
    # (implemented as copying b's initial transitions to each a-accepting state)
    b_init_mapped = b_state_map[b.initial]
    for acc in a.accepting:
        for t in b.transitions:
            if t.src == b.initial:
                transitions.append(SFATransition(acc, t.pred, b_state_map[t.dst]))

    # Accepting states: b's accepting (mapped)
    # Plus a's accepting if b accepts epsilon
    accepting = {b_state_map[s] for s in b.accepting}
    if b.initial in b.accepting:
        accepting |= a.accepting

    return SFA(
        states=states,
        initial=a.initial,
        accepting=accepting,
        transitions=transitions,
        algebra=algebra
    )


def sfa_star(sfa):
    """Kleene star: zero or more repetitions."""
    algebra = sfa.algebra

    # Add new initial state that is accepting (for epsilon)
    new_init = max(sfa.states) + 1 if sfa.states else 0
    states = sfa.states | {new_init}
    transitions = list(sfa.transitions)

    # Copy initial transitions to new initial
    for t in sfa._trans_from.get(sfa.initial, []):
        transitions.append(SFATransition(new_init, t.pred, t.dst))

    # From each accepting state, add transitions from initial (loop back)
    for acc in sfa.accepting:
        for t in sfa._trans_from.get(sfa.initial, []):
            transitions.append(SFATransition(acc, t.pred, t.dst))

    return SFA(
        states=states,
        initial=new_init,
        accepting=sfa.accepting | {new_init},
        transitions=transitions,
        algebra=algebra
    )


def sfa_plus(sfa):
    """One or more repetitions."""
    return sfa_concat(sfa, sfa_star(sfa))


def sfa_optional(sfa):
    """Zero or one occurrence."""
    return sfa_union(sfa_epsilon(sfa.algebra), sfa)


# ============================================================================
# Symbolic Transducer (SFT) -- Bonus: SFA with output
# ============================================================================

@dataclass
class SFTTransition:
    """Symbolic transducer transition: (src, guard, output_fn, dst)."""
    src: int
    pred: Pred
    output: list    # list of output symbols (can include input-dependent functions)
    dst: int


@dataclass
class SFT:
    """
    Symbolic Finite Transducer -- SFA with output.
    Each transition produces a list of output symbols.
    """
    states: set
    initial: int
    accepting: set
    transitions: list  # list of SFTTransition
    algebra: Any

    def __post_init__(self):
        self._trans_from = {}
        for t in self.transitions:
            self._trans_from.setdefault(t.src, []).append(t)

    def transduce(self, word):
        """Apply the transducer to a word. Returns output list or None."""
        current = [(self.initial, [])]
        for sym in word:
            next_states = []
            for s, out_so_far in current:
                for t in self._trans_from.get(s, []):
                    if self.algebra.evaluate(t.pred, sym):
                        new_out = list(out_so_far)
                        for o in t.output:
                            if callable(o):
                                new_out.append(o(sym))
                            else:
                                new_out.append(o)
                        next_states.append((t.dst, new_out))
            if not next_states:
                return None
            current = next_states
        # Return first accepting path's output
        for s, out in current:
            if s in self.accepting:
                return out
        return None

    def domain(self):
        """Return the SFA representing the domain (input language) of this SFT."""
        sfa_trans = [SFATransition(t.src, t.pred, t.dst) for t in self.transitions]
        return SFA(
            states=self.states.copy(),
            initial=self.initial,
            accepting=self.accepting.copy(),
            transitions=sfa_trans,
            algebra=self.algebra
        )


# ============================================================================
# Analysis and Utilities
# ============================================================================

def count_accepting_paths(sfa, max_length):
    """Count the number of words of length <= max_length accepted by the SFA."""
    count = 0
    # BFS with length tracking
    queue = [(sfa.initial, 0)]
    for _ in range(100000):  # Safety limit
        if not queue:
            break
        s, length = queue.pop(0)
        if s in sfa.accepting:
            count += 1
        if length >= max_length:
            continue
        for t in sfa._trans_from.get(s, []):
            if sfa.algebra.is_satisfiable(t.pred):
                queue.append((t.dst, length + 1))
    return count


def shortest_accepted(sfa):
    """Find the shortest word accepted by the SFA."""
    return sfa.accepted_word()


def sfa_stats(sfa):
    """Return statistics about an SFA."""
    return {
        'states': len(sfa.states),
        'transitions': len(sfa.transitions),
        'accepting': len(sfa.accepting),
        'deterministic': sfa.is_deterministic(),
        'empty': sfa.is_empty(),
    }


def compare_sfas(a, b):
    """Compare two SFAs for equivalence and report differences."""
    eq = sfa_is_equivalent(a, b)
    a_subset_b = sfa_is_subset(a, b)
    b_subset_a = sfa_is_subset(b, a)

    result = {
        'equivalent': eq,
        'a_subset_b': a_subset_b,
        'b_subset_a': b_subset_a,
        'a_stats': sfa_stats(a),
        'b_stats': sfa_stats(b),
    }

    if not eq:
        diff_ab = sfa_difference(a, b)
        w = diff_ab.accepted_word()
        if w is not None:
            result['witness_in_a_not_b'] = w
        diff_ba = sfa_difference(b, a)
        w2 = diff_ba.accepted_word()
        if w2 is not None:
            result['witness_in_b_not_a'] = w2

    return result
