"""
V083: Weighted Automata over Semirings

Automata with semiring-valued transitions for quantitative language analysis.
Composes with V081 (symbolic automata) concepts but standalone implementation.

Semiring framework:
  (S, +, *, 0, 1) where + is "combine" and * is "extend"
  - Boolean: (or, and, False, True) -- standard NFA
  - Tropical: (min, +, inf, 0) -- shortest paths
  - MaxPlus: (max, +, -inf, 0) -- longest paths
  - Probability: (+, *, 0, 1) -- stochastic systems
  - Counting: (+, *, 0, 1) over integers -- count accepting runs
  - Viterbi: (max, *, 0, 1) -- most probable path
  - MinMax: (min, max, inf, -inf) -- bottleneck paths
  - Log: (log-sum-exp, +, -inf, 0) -- log-probability (numerically stable)

Weighted Finite Automaton (WFA):
  - States, initial weights, final weights, transitions with weights
  - Weight of run = extend (product) of transition weights
  - Weight of word = combine (sum) of all accepting run weights

Operations:
  - Union, concatenation, Kleene star (closure)
  - Intersection (Hadamard product) for commutative semirings
  - Determinization for certain semirings (e.g., tropical)
  - Shortest-distance (generalized Dijkstra)
  - N-best paths
  - Weight pushing (for tropical/probability semirings)
  - Equivalence checking
  - Trim (remove non-productive/non-accessible states)
"""

from dataclasses import dataclass, field
from typing import (
    Dict, Set, List, Tuple, Optional, Any, TypeVar, Generic,
    Callable, Iterator
)
from enum import Enum
import math
from collections import defaultdict
import heapq


# ============================================================
# Semiring Framework
# ============================================================

class Semiring:
    """Abstract semiring (S, +, *, 0, 1)."""

    def zero(self):
        raise NotImplementedError

    def one(self):
        raise NotImplementedError

    def plus(self, a, b):
        """Combine operation (additive)."""
        raise NotImplementedError

    def times(self, a, b):
        """Extend operation (multiplicative)."""
        raise NotImplementedError

    def is_zero(self, a) -> bool:
        """Check if value is the additive identity."""
        return a == self.zero()

    def is_one(self, a) -> bool:
        """Check if value is the multiplicative identity."""
        return a == self.one()

    def star(self, a):
        """Kleene star: 1 + a + a*a + a*a*a + ...
        Default: iterate until convergence or raise."""
        result = self.one()
        power = self.one()
        for _ in range(1000):
            prev = result
            power = self.times(power, a)
            result = self.plus(result, power)
            if result == prev:
                return result
        raise ValueError(f"Kleene star did not converge for {a}")

    def plus_n(self, values):
        """Combine multiple values."""
        result = self.zero()
        for v in values:
            result = self.plus(result, v)
        return result

    def times_n(self, values):
        """Extend multiple values."""
        result = self.one()
        for v in values:
            result = self.times(result, v)
        return result


class BooleanSemiring(Semiring):
    """(or, and, False, True) -- standard NFA acceptance."""

    def zero(self): return False
    def one(self): return True
    def plus(self, a, b): return a or b
    def times(self, a, b): return a and b
    def star(self, a): return True
    def is_zero(self, a): return not a


class TropicalSemiring(Semiring):
    """(min, +, inf, 0) -- shortest paths."""

    def zero(self): return float('inf')
    def one(self): return 0.0
    def plus(self, a, b): return min(a, b)
    def times(self, a, b):
        if a == float('inf') or b == float('inf'):
            return float('inf')
        return a + b
    def is_zero(self, a): return a == float('inf')

    def star(self, a):
        if a >= 0:
            return 0.0
        raise ValueError("Tropical star undefined for negative weights (negative cycles)")


class MaxPlusSemiring(Semiring):
    """(max, +, -inf, 0) -- longest paths."""

    def zero(self): return float('-inf')
    def one(self): return 0.0
    def plus(self, a, b): return max(a, b)
    def times(self, a, b):
        if a == float('-inf') or b == float('-inf'):
            return float('-inf')
        return a + b
    def is_zero(self, a): return a == float('-inf')

    def star(self, a):
        if a <= 0:
            return 0.0
        raise ValueError("MaxPlus star undefined for positive weights (unbounded)")


class ProbabilitySemiring(Semiring):
    """(+, *, 0, 1) -- stochastic systems."""

    def zero(self): return 0.0
    def one(self): return 1.0
    def plus(self, a, b): return a + b
    def times(self, a, b): return a * b
    def is_zero(self, a): return a == 0.0

    def star(self, a):
        if abs(a) < 1.0:
            return 1.0 / (1.0 - a)
        raise ValueError(f"Probability star undefined for |a| >= 1: {a}")


class CountingSemiring(Semiring):
    """(+, *, 0, 1) over integers -- count accepting runs."""

    def zero(self): return 0
    def one(self): return 1
    def plus(self, a, b): return a + b
    def times(self, a, b): return a * b
    def is_zero(self, a): return a == 0


class ViterbiSemiring(Semiring):
    """(max, *, 0, 1) -- most probable path."""

    def zero(self): return 0.0
    def one(self): return 1.0
    def plus(self, a, b): return max(a, b)
    def times(self, a, b): return a * b
    def is_zero(self, a): return a == 0.0

    def star(self, a):
        if a <= 1.0:
            return 1.0
        raise ValueError(f"Viterbi star undefined for a > 1: {a}")


class MinMaxSemiring(Semiring):
    """(min, max, inf, -inf) -- bottleneck paths."""

    def zero(self): return float('inf')
    def one(self): return float('-inf')
    def plus(self, a, b): return min(a, b)
    def times(self, a, b): return max(a, b)
    def is_zero(self, a): return a == float('inf')

    def star(self, a):
        return max(float('-inf'), a)


class LogSemiring(Semiring):
    """(log-sum-exp, +, -inf, 0) -- log-probability (numerically stable)."""

    def zero(self): return float('-inf')
    def one(self): return 0.0

    def plus(self, a, b):
        if a == float('-inf'):
            return b
        if b == float('-inf'):
            return a
        mx = max(a, b)
        return mx + math.log1p(math.exp(-abs(a - b)))

    def times(self, a, b):
        if a == float('-inf') or b == float('-inf'):
            return float('-inf')
        return a + b

    def is_zero(self, a): return a == float('-inf')

    def star(self, a):
        if a < 0:
            return -math.log1p(-math.exp(a))
        raise ValueError(f"Log star undefined for a >= 0: {a}")


# Semiring registry
SEMIRINGS = {
    'boolean': BooleanSemiring,
    'tropical': TropicalSemiring,
    'maxplus': MaxPlusSemiring,
    'probability': ProbabilitySemiring,
    'counting': CountingSemiring,
    'viterbi': ViterbiSemiring,
    'minmax': MinMaxSemiring,
    'log': LogSemiring,
}


def make_semiring(name: str) -> Semiring:
    """Create a semiring by name."""
    if name not in SEMIRINGS:
        raise ValueError(f"Unknown semiring: {name}. Available: {list(SEMIRINGS.keys())}")
    return SEMIRINGS[name]()


# ============================================================
# Weighted Finite Automaton
# ============================================================

@dataclass
class WFATransition:
    """A weighted transition: source --label/weight--> target."""
    src: int
    label: str  # alphabet symbol (single character or string)
    dst: int
    weight: Any  # semiring element


@dataclass
class WFA:
    """Weighted Finite Automaton over a semiring.

    States are integers. Alphabet symbols are strings.
    initial_weight[state] = weight of starting in that state.
    final_weight[state] = weight of accepting in that state.
    Transitions: state -> list of (label, dst, weight).
    """
    semiring: Semiring
    states: Set[int] = field(default_factory=set)
    alphabet: Set[str] = field(default_factory=set)
    initial_weight: Dict[int, Any] = field(default_factory=dict)
    final_weight: Dict[int, Any] = field(default_factory=dict)
    transitions: Dict[int, List[Tuple[str, int, Any]]] = field(default_factory=lambda: defaultdict(list))
    _next_state: int = 0

    def add_state(self, state: Optional[int] = None,
                  initial=None, final=None) -> int:
        """Add a state with optional initial/final weights."""
        if state is None:
            state = self._next_state
        if state >= self._next_state:
            self._next_state = state + 1
        self.states.add(state)
        if initial is not None and not self.semiring.is_zero(initial):
            self.initial_weight[state] = initial
        if final is not None and not self.semiring.is_zero(final):
            self.final_weight[state] = final
        return state

    def add_transition(self, src: int, label: str, dst: int, weight=None):
        """Add a weighted transition."""
        if weight is None:
            weight = self.semiring.one()
        self.states.add(src)
        self.states.add(dst)
        self.alphabet.add(label)
        self.transitions[src].append((label, dst, weight))

    def get_transitions(self, state: int, label: Optional[str] = None) -> List[Tuple[str, int, Any]]:
        """Get transitions from a state, optionally filtered by label."""
        trans = self.transitions.get(state, [])
        if label is not None:
            return [(l, d, w) for l, d, w in trans if l == label]
        return trans

    def initial_states(self) -> Set[int]:
        """States with non-zero initial weight."""
        return set(self.initial_weight.keys())

    def final_states(self) -> Set[int]:
        """States with non-zero final weight."""
        return set(self.final_weight.keys())

    def get_initial_weight(self, state: int):
        """Get initial weight (zero if not initial)."""
        return self.initial_weight.get(state, self.semiring.zero())

    def get_final_weight(self, state: int):
        """Get final weight (zero if not final)."""
        return self.final_weight.get(state, self.semiring.zero())

    def n_transitions(self) -> int:
        """Total number of transitions."""
        return sum(len(t) for t in self.transitions.values())

    def copy(self) -> 'WFA':
        """Deep copy of this WFA."""
        w = WFA(self.semiring)
        w.states = set(self.states)
        w.alphabet = set(self.alphabet)
        w.initial_weight = dict(self.initial_weight)
        w.final_weight = dict(self.final_weight)
        w.transitions = defaultdict(list)
        for s, tlist in self.transitions.items():
            w.transitions[s] = list(tlist)
        w._next_state = self._next_state
        return w


# ============================================================
# WFA Construction Helpers
# ============================================================

def wfa_from_word(word: str, semiring: Semiring, weight=None) -> WFA:
    """Create a WFA accepting exactly one word with given weight."""
    if weight is None:
        weight = semiring.one()
    wfa = WFA(semiring)
    states = [wfa.add_state() for _ in range(len(word) + 1)]
    wfa.initial_weight[states[0]] = semiring.one()
    wfa.final_weight[states[-1]] = weight
    for i, ch in enumerate(word):
        wfa.add_transition(states[i], ch, states[i + 1], semiring.one())
    return wfa


def wfa_from_symbol(symbol: str, semiring: Semiring, weight=None) -> WFA:
    """Create a WFA accepting a single symbol."""
    return wfa_from_word(symbol, semiring, weight)


def wfa_epsilon(semiring: Semiring, weight=None) -> WFA:
    """Create a WFA accepting only the empty string."""
    if weight is None:
        weight = semiring.one()
    wfa = WFA(semiring)
    s = wfa.add_state(initial=semiring.one(), final=weight)
    return wfa


def wfa_empty(semiring: Semiring) -> WFA:
    """Create a WFA accepting nothing (empty language)."""
    wfa = WFA(semiring)
    wfa.add_state()
    return wfa


# ============================================================
# WFA Operations
# ============================================================

def _remap_states(wfa: WFA, offset: int) -> WFA:
    """Create a copy with all state IDs shifted by offset."""
    result = WFA(wfa.semiring)
    for s in wfa.states:
        result.states.add(s + offset)
    result.alphabet = set(wfa.alphabet)
    for s, w in wfa.initial_weight.items():
        result.initial_weight[s + offset] = w
    for s, w in wfa.final_weight.items():
        result.final_weight[s + offset] = w
    for s, tlist in wfa.transitions.items():
        for label, dst, w in tlist:
            result.transitions[s + offset].append((label, dst + offset, w))
    result._next_state = max(s + offset for s in wfa.states) + 1 if wfa.states else offset
    return result


def wfa_union(a: WFA, b: WFA) -> WFA:
    """Union of two WFAs. Weight of word = a(word) + b(word)."""
    sr = a.semiring
    offset = max(a.states) + 1 if a.states else 0
    result = WFA(sr)
    result.alphabet = a.alphabet | b.alphabet

    # Copy a
    for s in a.states:
        result.states.add(s)
    for s, w in a.initial_weight.items():
        result.initial_weight[s] = w
    for s, w in a.final_weight.items():
        result.final_weight[s] = w
    for s, tlist in a.transitions.items():
        result.transitions[s] = list(tlist)

    # Copy b with offset
    for s in b.states:
        result.states.add(s + offset)
    for s, w in b.initial_weight.items():
        result.initial_weight[s + offset] = w
    for s, w in b.final_weight.items():
        result.final_weight[s + offset] = w
    for s, tlist in b.transitions.items():
        for label, dst, w in tlist:
            result.transitions[s + offset].append((label, dst + offset, w))

    result._next_state = max(result.states) + 1 if result.states else 0
    return result


def wfa_concat(a: WFA, b: WFA) -> WFA:
    """Concatenation of two WFAs. Uses epsilon-free construction."""
    sr = a.semiring
    offset = max(a.states) + 1 if a.states else 0
    result = WFA(sr)
    result.alphabet = a.alphabet | b.alphabet

    # Copy a states
    for s in a.states:
        result.states.add(s)
    for s, w in a.initial_weight.items():
        result.initial_weight[s] = w
    for s, tlist in a.transitions.items():
        result.transitions[s] = list(tlist)

    # Copy b states with offset
    for s in b.states:
        result.states.add(s + offset)
    for s, w in b.final_weight.items():
        result.final_weight[s + offset] = w
    for s, tlist in b.transitions.items():
        for label, dst, w in tlist:
            result.transitions[s + offset].append((label, dst + offset, w))

    # Connect: for each final state of a and each initial state of b,
    # add transitions from a-final through b-initial's outgoing edges
    for fa, fw in a.final_weight.items():
        for ib, iw in b.initial_weight.items():
            bridge_w = sr.times(fw, iw)
            for label, dst, tw in b.transitions.get(ib, []):
                result.transitions[fa].append((label, dst + offset, sr.times(bridge_w, tw)))
            # If b-initial is also b-final, a-final becomes final
            bf = b.final_weight.get(ib)
            if bf is not None:
                combined = sr.times(bridge_w, bf)
                existing = result.final_weight.get(fa, sr.zero())
                result.final_weight[fa] = sr.plus(existing, combined)

    result._next_state = max(result.states) + 1 if result.states else 0
    return result


def wfa_star(a: WFA) -> WFA:
    """Kleene star (closure) of a WFA."""
    sr = a.semiring
    result = a.copy()

    # Add epsilon acceptance: make initial states also final
    for s, iw in list(a.initial_weight.items()):
        existing = result.final_weight.get(s, sr.zero())
        result.final_weight[s] = sr.plus(existing, iw)

    # Add loopback: for each final state, connect to initial state's successors
    for fa, fw in a.final_weight.items():
        for ib, iw in a.initial_weight.items():
            bridge_w = sr.times(fw, iw)
            for label, dst, tw in a.transitions.get(ib, []):
                result.transitions[fa].append((label, dst, sr.times(bridge_w, tw)))

    return result


def wfa_intersect(a: WFA, b: WFA) -> WFA:
    """Hadamard product (intersection) of two WFAs.
    Weight of word = a(word) * b(word). Requires commutative semiring."""
    sr = a.semiring
    result = WFA(sr)
    result.alphabet = a.alphabet & b.alphabet

    # Product construction
    state_map = {}
    counter = [0]

    def get_state(sa, sb):
        key = (sa, sb)
        if key not in state_map:
            state_map[key] = counter[0]
            counter[0] += 1
            result.states.add(state_map[key])
        return state_map[key]

    # BFS from initial state pairs
    queue = []
    visited = set()

    for sa, wa in a.initial_weight.items():
        for sb, wb in b.initial_weight.items():
            s = get_state(sa, sb)
            result.initial_weight[s] = sr.times(wa, wb)
            queue.append((sa, sb))
            visited.add((sa, sb))

    while queue:
        sa, sb = queue.pop(0)
        s = get_state(sa, sb)

        # Final weight
        fa = a.final_weight.get(sa)
        fb = b.final_weight.get(sb)
        if fa is not None and fb is not None:
            result.final_weight[s] = sr.times(fa, fb)

        # Transitions on matching labels
        a_trans = defaultdict(list)
        for label, dst, w in a.transitions.get(sa, []):
            a_trans[label].append((dst, w))

        for label, dst_b, wb in b.transitions.get(sb, []):
            for dst_a, wa in a_trans.get(label, []):
                dst = get_state(dst_a, dst_b)
                result.add_transition(s, label, dst, sr.times(wa, wb))
                if (dst_a, dst_b) not in visited:
                    visited.add((dst_a, dst_b))
                    queue.append((dst_a, dst_b))

    result._next_state = counter[0]
    return result


# ============================================================
# Run Weight Computation
# ============================================================

def wfa_run_weight(wfa: WFA, word: str) -> Any:
    """Compute the weight of a word in the WFA.
    This is the semiring-sum over all accepting runs of the
    semiring-product of transition weights."""
    sr = wfa.semiring
    # Dynamic programming: current[state] = combined weight of reaching state
    current = {}
    for s, w in wfa.initial_weight.items():
        current[s] = w

    for ch in word:
        nxt = {}
        for s, sw in current.items():
            for label, dst, tw in wfa.transitions.get(s, []):
                if label == ch:
                    w = sr.times(sw, tw)
                    if dst in nxt:
                        nxt[dst] = sr.plus(nxt[dst], w)
                    else:
                        nxt[dst] = w
        current = nxt

    # Combine final weights
    result = sr.zero()
    for s, sw in current.items():
        fw = wfa.final_weight.get(s)
        if fw is not None:
            result = sr.plus(result, sr.times(sw, fw))

    return result


def wfa_accepts(wfa: WFA, word: str) -> bool:
    """Check if a word is accepted (has non-zero weight)."""
    return not wfa.semiring.is_zero(wfa_run_weight(wfa, word))


# ============================================================
# Shortest Distance (Generalized Dijkstra)
# ============================================================

def shortest_distance(wfa: WFA, source: Optional[int] = None) -> Dict[int, Any]:
    """Compute shortest distance from source (or initial states) to all states.
    Works for tropical and similar semirings with monotonic weights."""
    sr = wfa.semiring

    dist = {s: sr.zero() for s in wfa.states}

    if source is not None:
        dist[source] = sr.one()
    else:
        for s, w in wfa.initial_weight.items():
            dist[s] = w

    # Bellman-Ford style relaxation (works for all semirings)
    changed = True
    iterations = 0
    max_iter = len(wfa.states) + 1
    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        for s in wfa.states:
            if sr.is_zero(dist[s]):
                continue
            for label, dst, w in wfa.transitions.get(s, []):
                new_w = sr.times(dist[s], w)
                old = dist[dst]
                combined = sr.plus(old, new_w)
                if combined != old:
                    dist[dst] = combined
                    changed = True

    return dist


def all_pairs_distance(wfa: WFA) -> Dict[Tuple[int, int], Any]:
    """All-pairs shortest distances (Floyd-Warshall style)."""
    sr = wfa.semiring
    states = sorted(wfa.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Initialize
    dist = [[sr.zero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = sr.one()

    for s, tlist in wfa.transitions.items():
        si = idx[s]
        for label, dst, w in tlist:
            di = idx[dst]
            dist[si][di] = sr.plus(dist[si][di], w)

    # Relaxation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via_k = sr.times(dist[i][k], sr.times(sr.star(dist[k][k]), dist[k][j]))
                dist[i][j] = sr.plus(dist[i][j], via_k)

    # Build result
    result = {}
    for i in range(n):
        for j in range(n):
            result[(states[i], states[j])] = dist[i][j]
    return result


# ============================================================
# N-Best Paths
# ============================================================

def n_best_paths(wfa: WFA, word: str, n: int) -> List[Tuple[Any, List[int]]]:
    """Find the n best (lowest weight for tropical, highest for viterbi) accepting
    runs for a given word. Returns list of (weight, state_sequence) pairs."""
    sr = wfa.semiring

    # Use Viterbi-style search with beam
    # Each entry: (accumulated_weight, state, position, path)
    if isinstance(sr, TropicalSemiring):
        # Min-heap for tropical
        heap = []
        for s, iw in wfa.initial_weight.items():
            heapq.heappush(heap, (iw, s, 0, [s]))
    elif isinstance(sr, ViterbiSemiring) or isinstance(sr, MaxPlusSemiring):
        # Max-heap (negate for heapq)
        heap = []
        for s, iw in wfa.initial_weight.items():
            heapq.heappush(heap, (-iw, s, 0, [s]))
    else:
        # Default: collect all and sort
        return _n_best_exhaustive(wfa, word, n)

    results = []
    use_negate = isinstance(sr, (ViterbiSemiring, MaxPlusSemiring))

    while heap and len(results) < n:
        if use_negate:
            neg_w, state, pos, path = heapq.heappop(heap)
            w = -neg_w
        else:
            w, state, pos, path = heapq.heappop(heap)

        if pos == len(word):
            fw = wfa.final_weight.get(state)
            if fw is not None:
                final_w = sr.times(w, fw)
                results.append((final_w, path))
            continue

        ch = word[pos]
        for label, dst, tw in wfa.transitions.get(state, []):
            if label == ch:
                nw = sr.times(w, tw)
                npath = path + [dst]
                if use_negate:
                    heapq.heappush(heap, (-nw, dst, pos + 1, npath))
                else:
                    heapq.heappush(heap, (nw, dst, pos + 1, npath))

    return results


def _n_best_exhaustive(wfa: WFA, word: str, n: int) -> List[Tuple[Any, List[int]]]:
    """Exhaustive search for n-best paths (for general semirings)."""
    sr = wfa.semiring
    results = []

    def dfs(state, pos, weight, path):
        if pos == len(word):
            fw = wfa.final_weight.get(state)
            if fw is not None:
                results.append((sr.times(weight, fw), list(path)))
            return
        ch = word[pos]
        for label, dst, tw in wfa.transitions.get(state, []):
            if label == ch:
                path.append(dst)
                dfs(dst, pos + 1, sr.times(weight, tw), path)
                path.pop()

    for s, iw in wfa.initial_weight.items():
        dfs(s, 0, iw, [s])

    # Sort: for tropical/minmax use min, for viterbi/probability use max
    if isinstance(sr, (TropicalSemiring, MinMaxSemiring)):
        results.sort(key=lambda x: x[0])
    else:
        results.sort(key=lambda x: -x[0] if isinstance(x[0], (int, float)) else 0)

    return results[:n]


# ============================================================
# Trim (remove unreachable / non-productive states)
# ============================================================

def wfa_trim(wfa: WFA) -> WFA:
    """Remove states that are not accessible from initial or not co-accessible to final."""
    sr = wfa.semiring

    # Forward reachability from initial states
    accessible = set()
    queue = list(wfa.initial_weight.keys())
    accessible.update(queue)
    while queue:
        s = queue.pop()
        for label, dst, w in wfa.transitions.get(s, []):
            if dst not in accessible:
                accessible.add(dst)
                queue.append(dst)

    # Backward reachability from final states
    # Build reverse adjacency
    rev = defaultdict(set)
    for s, tlist in wfa.transitions.items():
        for label, dst, w in tlist:
            rev[dst].add(s)

    co_accessible = set()
    queue = list(wfa.final_weight.keys())
    co_accessible.update(queue)
    while queue:
        s = queue.pop()
        for pred in rev.get(s, set()):
            if pred not in co_accessible:
                co_accessible.add(pred)
                queue.append(pred)

    # Keep states in both sets
    keep = accessible & co_accessible

    result = WFA(sr)
    for s in keep:
        result.states.add(s)
    for s, w in wfa.initial_weight.items():
        if s in keep:
            result.initial_weight[s] = w
    for s, w in wfa.final_weight.items():
        if s in keep:
            result.final_weight[s] = w
    for s, tlist in wfa.transitions.items():
        if s in keep:
            for label, dst, w in tlist:
                if dst in keep:
                    result.transitions[s].append((label, dst, w))
    result.alphabet = set(wfa.alphabet)
    result._next_state = wfa._next_state
    return result


# ============================================================
# Weight Pushing
# ============================================================

def wfa_push_weights(wfa: WFA) -> WFA:
    """Push weights toward initial states (tropical/probability semirings).
    After pushing, all states have 'balanced' outgoing weight distributions.
    For tropical: shortest distance from each state to any final state."""
    sr = wfa.semiring

    # Compute shortest distance to final states (reverse graph)
    rev_wfa = WFA(sr)
    for s in wfa.states:
        rev_wfa.states.add(s)
    for s, w in wfa.final_weight.items():
        rev_wfa.initial_weight[s] = w
    for s, tlist in wfa.transitions.items():
        for label, dst, w in tlist:
            rev_wfa.transitions[dst].append((label, s, w))

    dist_to_final = shortest_distance(rev_wfa)

    # Reweight
    result = WFA(sr)
    result.states = set(wfa.states)
    result.alphabet = set(wfa.alphabet)
    result._next_state = wfa._next_state

    for s, iw in wfa.initial_weight.items():
        d = dist_to_final.get(s, sr.zero())
        if not sr.is_zero(d):
            result.initial_weight[s] = sr.times(iw, d)

    for s, fw in wfa.final_weight.items():
        d = dist_to_final.get(s, sr.zero())
        if not sr.is_zero(d):
            if isinstance(sr, TropicalSemiring):
                result.final_weight[s] = 0.0  # Pushed to zero
            elif isinstance(sr, ProbabilitySemiring):
                result.final_weight[s] = fw / d if d != 0 else fw
            else:
                result.final_weight[s] = fw

    for s, tlist in wfa.transitions.items():
        ds = dist_to_final.get(s, sr.zero())
        for label, dst, w in tlist:
            dd = dist_to_final.get(dst, sr.zero())
            if not sr.is_zero(ds) and not sr.is_zero(dd):
                if isinstance(sr, TropicalSemiring):
                    new_w = w + dd - ds
                elif isinstance(sr, ProbabilitySemiring):
                    new_w = w * dd / ds if ds != 0 else w
                else:
                    new_w = w
                result.transitions[s].append((label, dst, new_w))
            else:
                result.transitions[s].append((label, dst, w))

    return result


# ============================================================
# Determinization (for tropical semiring)
# ============================================================

def wfa_determinize(wfa: WFA) -> WFA:
    """Determinize a WFA. Works correctly for tropical semiring
    (uses weight-residual construction). For other semirings,
    produces a deterministic approximation."""
    sr = wfa.semiring
    result = WFA(sr)
    result.alphabet = set(wfa.alphabet)

    # A deterministic state is a frozenset of (original_state, residual_weight)
    init_config = frozenset(
        (s, w) for s, w in wfa.initial_weight.items()
    )

    state_map = {init_config: 0}
    result.add_state(0)
    # Initial weight: minimum (tropical) or sum (probability) of initial weights
    result.initial_weight[0] = sr.plus_n(w for _, w in init_config)

    queue = [init_config]
    counter = 1

    while queue:
        config = queue.pop(0)
        src = state_map[config]
        combined_initial = sr.plus_n(w for _, w in config)

        # Check final
        final_vals = []
        for s, rw in config:
            fw = wfa.final_weight.get(s)
            if fw is not None:
                if isinstance(sr, TropicalSemiring):
                    final_vals.append(rw + fw)
                else:
                    final_vals.append(sr.times(rw, fw))
        if final_vals:
            result.final_weight[src] = sr.plus_n(final_vals)

        # Group transitions by label
        label_targets = defaultdict(list)
        for s, rw in config:
            for label, dst, tw in wfa.transitions.get(s, []):
                if isinstance(sr, TropicalSemiring):
                    label_targets[label].append((dst, rw + tw))
                else:
                    label_targets[label].append((dst, sr.times(rw, tw)))

        for label, targets in label_targets.items():
            # Merge targets by destination
            merged = defaultdict(lambda: sr.zero())
            for dst, w in targets:
                merged[dst] = sr.plus(merged[dst], w)

            # For tropical: normalize by subtracting minimum residual
            if isinstance(sr, TropicalSemiring):
                min_w = min(merged.values())
                new_config = frozenset((d, w - min_w) for d, w in merged.items())
                edge_w = min_w
            else:
                total = sr.plus_n(merged.values())
                if isinstance(sr, ProbabilitySemiring) and total != 0:
                    new_config = frozenset((d, w / total) for d, w in merged.items())
                    edge_w = total
                else:
                    new_config = frozenset(merged.items())
                    edge_w = sr.one()

            if new_config not in state_map:
                state_map[new_config] = counter
                result.add_state(counter)
                counter += 1
                queue.append(new_config)

            result.add_transition(src, label, state_map[new_config], edge_w)

    result._next_state = counter
    return result


# ============================================================
# Equivalence Checking
# ============================================================

def wfa_equivalent(a: WFA, b: WFA, test_words: Optional[List[str]] = None,
                   max_length: int = 6) -> Tuple[bool, Optional[str]]:
    """Check if two WFAs are equivalent (assign same weight to all words).
    Uses exhaustive testing up to max_length, or provided test words.
    Returns (is_equivalent, counterexample_or_None)."""
    sr = a.semiring
    if test_words is None:
        alpha = sorted(a.alphabet | b.alphabet)
        if not alpha:
            wa = wfa_run_weight(a, "")
            wb = wfa_run_weight(b, "")
            return (wa == wb, None if wa == wb else "")
        test_words = _generate_words(alpha, max_length)

    for word in test_words:
        wa = wfa_run_weight(a, word)
        wb = wfa_run_weight(b, word)
        if isinstance(wa, float) and isinstance(wb, float):
            if abs(wa - wb) > 1e-9:
                return (False, word)
        elif wa != wb:
            return (False, word)

    return (True, None)


def _generate_words(alphabet: List[str], max_length: int) -> Iterator[str]:
    """Generate all words up to max_length over alphabet."""
    yield ""
    if max_length == 0:
        return
    queue = list(alphabet)
    for word in queue:
        yield word
        if len(word) < max_length:
            for ch in alphabet:
                queue.append(word + ch)


# ============================================================
# WFA to/from NFA conversion
# ============================================================

def nfa_to_wfa(states: Set[int], initial: Set[int], final: Set[int],
               transitions: Dict[int, List[Tuple[str, int]]],
               semiring: Optional[Semiring] = None) -> WFA:
    """Convert a classical NFA to a WFA over boolean (or given) semiring."""
    if semiring is None:
        semiring = BooleanSemiring()
    wfa = WFA(semiring)
    for s in states:
        init_w = semiring.one() if s in initial else None
        fin_w = semiring.one() if s in final else None
        wfa.add_state(s, initial=init_w, final=fin_w)
    for s, tlist in transitions.items():
        for label, dst in tlist:
            wfa.add_transition(s, label, dst, semiring.one())
    return wfa


def wfa_to_nfa(wfa: WFA) -> Tuple[Set[int], Set[int], Set[int], Dict[int, List[Tuple[str, int]]]]:
    """Convert a WFA to an NFA (dropping weights). Returns (states, initial, final, transitions)."""
    transitions = defaultdict(list)
    for s, tlist in wfa.transitions.items():
        for label, dst, w in tlist:
            if not wfa.semiring.is_zero(w):
                transitions[s].append((label, dst))
    return (wfa.states, set(wfa.initial_weight.keys()),
            set(wfa.final_weight.keys()), dict(transitions))


# ============================================================
# Analysis & Statistics
# ============================================================

@dataclass
class WFAStats:
    """Statistics about a WFA."""
    n_states: int
    n_transitions: int
    n_initial: int
    n_final: int
    alphabet_size: int
    is_deterministic: bool
    is_trim: bool
    density: float  # transitions / (states * alphabet_size)


def wfa_stats(wfa: WFA) -> WFAStats:
    """Compute statistics about a WFA."""
    n_states = len(wfa.states)
    n_trans = wfa.n_transitions()
    n_init = len(wfa.initial_weight)
    n_final = len(wfa.final_weight)
    alpha_size = len(wfa.alphabet)

    # Check deterministic: at most one transition per (state, label)
    is_det = True
    for s in wfa.states:
        seen_labels = set()
        for label, dst, w in wfa.transitions.get(s, []):
            if label in seen_labels:
                is_det = False
                break
            seen_labels.add(label)
        if not is_det:
            break
    is_det = is_det and n_init <= 1

    # Check trim
    trimmed = wfa_trim(wfa)
    is_trim = len(trimmed.states) == n_states

    density = n_trans / (n_states * alpha_size) if n_states > 0 and alpha_size > 0 else 0.0

    return WFAStats(
        n_states=n_states,
        n_transitions=n_trans,
        n_initial=n_init,
        n_final=n_final,
        alphabet_size=alpha_size,
        is_deterministic=is_det,
        is_trim=is_trim,
        density=density,
    )


def wfa_summary(wfa: WFA) -> str:
    """Human-readable summary of a WFA."""
    stats = wfa_stats(wfa)
    lines = [
        f"WFA: {stats.n_states} states, {stats.n_transitions} transitions",
        f"  Alphabet: {sorted(wfa.alphabet)} ({stats.alphabet_size} symbols)",
        f"  Initial: {stats.n_initial}, Final: {stats.n_final}",
        f"  Deterministic: {stats.is_deterministic}, Trim: {stats.is_trim}",
        f"  Density: {stats.density:.2f}",
    ]
    return "\n".join(lines)


# ============================================================
# Semiring-Parametric Algorithms
# ============================================================

def wfa_language_weight(wfa: WFA, max_length: int = 10) -> Any:
    """Total weight of all words up to max_length."""
    sr = wfa.semiring
    alpha = sorted(wfa.alphabet)
    if not alpha:
        return wfa_run_weight(wfa, "")

    total = sr.zero()
    for word in _generate_words(alpha, max_length):
        w = wfa_run_weight(wfa, word)
        total = sr.plus(total, w)
    return total


def wfa_compose(a: WFA, b: WFA) -> WFA:
    """Compose two WFAs as weighted transducers (treat label as input:output).
    For simple automata, this is the same as intersection."""
    return wfa_intersect(a, b)


# ============================================================
# Comparison & Conversion Between Semirings
# ============================================================

def convert_semiring(wfa: WFA, new_semiring: Semiring,
                     weight_map: Callable = None) -> WFA:
    """Convert a WFA to a different semiring using a weight mapping function."""
    if weight_map is None:
        weight_map = lambda x: x
    result = WFA(new_semiring)
    result.states = set(wfa.states)
    result.alphabet = set(wfa.alphabet)
    result._next_state = wfa._next_state
    for s, w in wfa.initial_weight.items():
        result.initial_weight[s] = weight_map(w)
    for s, w in wfa.final_weight.items():
        result.final_weight[s] = weight_map(w)
    for s, tlist in wfa.transitions.items():
        for label, dst, w in tlist:
            result.transitions[s].append((label, dst, weight_map(w)))
    return result


def compare_wfas(a: WFA, b: WFA, words: List[str]) -> Dict:
    """Compare two WFAs on a set of words."""
    results = {}
    for word in words:
        wa = wfa_run_weight(a, word)
        wb = wfa_run_weight(b, word)
        results[word] = {'a': wa, 'b': wb, 'equal': wa == wb}
    return {
        'per_word': results,
        'all_equal': all(r['equal'] for r in results.values()),
        'n_words': len(words),
        'n_agree': sum(1 for r in results.values() if r['equal']),
    }
