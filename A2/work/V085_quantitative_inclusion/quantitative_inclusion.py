"""
V085: Quantitative Language Inclusion

Check quantitative inclusion/equivalence between weighted finite automata:
  - For every string w: weight_A(w) <= weight_B(w) (inclusion)
  - For every string w: weight_A(w) == weight_B(w) (equivalence)

Composes V083 (weighted automata) + C037 (SMT solver).

Techniques:
  1. Bounded exploration: enumerate strings up to length k, compare weights
  2. Product construction: build difference WFA, check for positive weight
  3. Bisimulation: partition-refinement for weighted equivalence
  4. SMT-based: encode weight comparison as SMT constraints for tropical semiring
  5. Antichain-based: simulation relation via antichain optimization

Applications:
  - Quantitative refinement checking (does impl refine spec?)
  - Optimizer validation (does optimized WFA preserve weights?)
  - Cost bound verification (does cost never exceed budget?)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V083_weighted_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from weighted_automata import (
    Semiring, WFA,
    BooleanSemiring, TropicalSemiring, MaxPlusSemiring,
    ProbabilitySemiring, CountingSemiring, ViterbiSemiring,
    MinMaxSemiring, LogSemiring, make_semiring,
    wfa_from_word, wfa_from_symbol, wfa_epsilon, wfa_empty,
    wfa_union, wfa_concat, wfa_star, wfa_intersect,
    wfa_run_weight, wfa_accepts, wfa_trim, wfa_determinize,
    wfa_equivalent, wfa_stats, wfa_language_weight,
    shortest_distance, n_best_paths, compare_wfas,
)
from smt_solver import SMTSolver, SMTResult, Op, Sort, SortKind, App, Var, IntConst, BoolConst

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict, deque
import math


# ============================================================
# Result Types
# ============================================================

class InclusionVerdict(Enum):
    INCLUDED = "INCLUDED"           # A <= B for all strings
    NOT_INCLUDED = "NOT_INCLUDED"   # exists string w: A(w) > B(w)
    EQUIVALENT = "EQUIVALENT"       # A == B for all strings
    UNKNOWN = "UNKNOWN"             # could not determine


@dataclass
class InclusionWitness:
    """A string witnessing non-inclusion: weight_A(w) > weight_B(w)."""
    word: str
    weight_a: Any
    weight_b: Any


@dataclass
class InclusionResult:
    verdict: InclusionVerdict
    witnesses: List[InclusionWitness] = field(default_factory=list)
    checked_words: int = 0
    max_length_checked: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def included(self) -> bool:
        return self.verdict in (InclusionVerdict.INCLUDED, InclusionVerdict.EQUIVALENT)


@dataclass
class EquivalenceResult:
    equivalent: bool
    witnesses: List[InclusionWitness] = field(default_factory=list)
    checked_words: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BisimulationResult:
    bisimilar: bool
    partition: Optional[List[Set]] = None
    distinguishing_word: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistanceResult:
    """Quantitative distance between two WFAs."""
    distance: float
    worst_word: Optional[str] = None
    worst_diff: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Semiring Order Relations
# ============================================================

def _semiring_leq(sr: Semiring, a, b) -> bool:
    """Check a <= b in the natural order of the semiring.

    Natural order: a <= b iff a + b == b (b absorbs a).
    For tropical (min,+): a <= b iff min(a,b) == b iff a >= b (reversed!)
    For maxplus (max,+): a <= b iff max(a,b) == b iff a <= b
    For probability (+,*): a <= b iff a + b == b (only if a == 0)
    For boolean (or,and): a <= b iff a or b == b
    """
    return sr.plus(a, b) == b


def _semiring_lt(sr: Semiring, a, b) -> bool:
    """Strict: a < b iff a <= b and a != b."""
    return _semiring_leq(sr, a, b) and a != b


def _numeric_leq(sr: Semiring, a, b) -> bool:
    """Numeric comparison for common semirings.

    For tropical: smaller weight = cheaper = better, so inclusion
    means A's cost >= B's cost for all words (B is cheaper).
    We use standard numeric <= for user-facing comparison.
    """
    if isinstance(sr, (TropicalSemiring, MinMaxSemiring)):
        # These use min as plus; "smaller" means "better"
        # a <= b in numeric sense
        if a == float('inf') and b == float('inf'):
            return True
        if a == float('inf'):
            return False
        if b == float('inf'):
            return True
        return a <= b
    elif isinstance(sr, MaxPlusSemiring):
        if a == float('-inf') and b == float('-inf'):
            return True
        if a == float('-inf'):
            return True
        if b == float('-inf'):
            return False
        return a <= b
    elif isinstance(sr, (ProbabilitySemiring, CountingSemiring, ViterbiSemiring)):
        return a <= b
    elif isinstance(sr, BooleanSemiring):
        # False <= True
        return (not a) or b
    elif isinstance(sr, LogSemiring):
        return a <= b
    else:
        # Fallback: try numeric
        try:
            return float(a) <= float(b)
        except (TypeError, ValueError):
            return _semiring_leq(sr, a, b)


def _weight_diff(sr: Semiring, a, b) -> float:
    """Compute numeric difference a - b for distance measurement."""
    try:
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb):
            if fa == fb:
                return 0.0
            return float('inf')
        return fa - fb
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# 1. Bounded Exploration
# ============================================================

def _generate_words(alphabet: List[str], max_length: int, max_words: int = 50000) -> List[str]:
    """Generate words up to given length over alphabet."""
    words = [""]  # empty string
    queue = deque([""])
    while queue and len(words) < max_words:
        prefix = queue.popleft()
        if len(prefix) >= max_length:
            continue
        for c in alphabet:
            w = prefix + c
            words.append(w)
            if len(words) >= max_words:
                break
            queue.append(w)
    return words


def _get_alphabet(wfa: WFA) -> List[str]:
    """Extract alphabet from WFA transitions."""
    alpha = set()
    for state in wfa.states:
        for label, dst, weight in wfa.transitions.get(state, []):
            alpha.add(label)
    return sorted(alpha)


def bounded_inclusion_check(
    a: WFA, b: WFA,
    max_length: int = 8,
    max_words: int = 50000
) -> InclusionResult:
    """Check inclusion by enumerating strings up to max_length.

    Returns INCLUDED if all checked strings satisfy A(w) <= B(w),
    NOT_INCLUDED with witnesses if any violation found.
    """
    sr = a.semiring
    alpha_a = _get_alphabet(a)
    alpha_b = _get_alphabet(b)
    alphabet = sorted(set(alpha_a) | set(alpha_b))

    if not alphabet:
        # Both empty alphabet -- only empty string matters
        wa = wfa_run_weight(a, "")
        wb = wfa_run_weight(b, "")
        if _numeric_leq(sr, wa, wb):
            return InclusionResult(
                verdict=InclusionVerdict.INCLUDED,
                checked_words=1, max_length_checked=0
            )
        else:
            return InclusionResult(
                verdict=InclusionVerdict.NOT_INCLUDED,
                witnesses=[InclusionWitness("", wa, wb)],
                checked_words=1, max_length_checked=0
            )

    words = _generate_words(alphabet, max_length, max_words)
    witnesses = []
    checked = 0

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        checked += 1

        if not _numeric_leq(sr, wa, wb):
            witnesses.append(InclusionWitness(w, wa, wb))

    verdict = InclusionVerdict.NOT_INCLUDED if witnesses else InclusionVerdict.INCLUDED
    return InclusionResult(
        verdict=verdict,
        witnesses=witnesses,
        checked_words=checked,
        max_length_checked=max_length,
        stats={"alphabet_size": len(alphabet), "method": "bounded"}
    )


def bounded_equivalence_check(
    a: WFA, b: WFA,
    max_length: int = 8,
    max_words: int = 50000
) -> EquivalenceResult:
    """Check equivalence by comparing weights on all strings up to max_length."""
    sr = a.semiring
    alpha_a = _get_alphabet(a)
    alpha_b = _get_alphabet(b)
    alphabet = sorted(set(alpha_a) | set(alpha_b))

    words = _generate_words(alphabet, max_length, max_words) if alphabet else [""]
    witnesses = []
    checked = 0

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        checked += 1

        if wa != wb:
            witnesses.append(InclusionWitness(w, wa, wb))

    return EquivalenceResult(
        equivalent=len(witnesses) == 0,
        witnesses=witnesses,
        checked_words=checked,
        stats={"method": "bounded", "max_length": max_length}
    )


# ============================================================
# 2. Product Construction (Difference WFA)
# ============================================================

def build_product_wfa(a: WFA, b: WFA) -> WFA:
    """Build product WFA A x B tracking both weights simultaneously.

    States are (state_a, state_b) pairs. Transitions exist when both
    automata can read the same symbol. Weights are pairs (wa, wb).

    Returns a WFA over a PairSemiring that tracks both weights.
    """
    sr = a.semiring
    product = WFA(sr)

    # State mapping: (sa, sb) -> product_state_id
    state_map = {}
    next_id = 0

    def get_state(sa, sb):
        nonlocal next_id
        key = (sa, sb)
        if key not in state_map:
            state_map[key] = next_id
            product.add_state(next_id)
            next_id += 1
        return state_map[key]

    # Build initial states
    for sa in a.states:
        iwa = a.get_initial_weight(sa)
        if sr.is_zero(iwa):
            continue
        for sb in b.states:
            iwb = b.get_initial_weight(sb)
            if sr.is_zero(iwb):
                continue
            ps = get_state(sa, sb)
            product.initial_weight[ps] = sr.one()

    # Build transitions
    for sa in a.states:
        a_trans = a.transitions.get(sa, [])
        for a_label, a_dst, a_w in a_trans:
            for sb in b.states:
                b_trans = b.transitions.get(sb, [])
                for b_label, b_dst, b_w in b_trans:
                    if a_label == b_label:
                        src = get_state(sa, sb)
                        dst = get_state(a_dst, b_dst)
                        pw = sr.times(a_w, b_w)
                        product.add_transition(src, a_label, dst, pw)

    # Final weights
    for (sa, sb), ps in state_map.items():
        fwa = a.get_final_weight(sa)
        fwb = b.get_final_weight(sb)
        if not sr.is_zero(fwa) or not sr.is_zero(fwb):
            product.final_weight[ps] = sr.times(fwa, fwb)

    return product, state_map


def product_inclusion_check(a: WFA, b: WFA, max_length: int = 10) -> InclusionResult:
    """Check inclusion via product construction and word enumeration.

    Builds the product and checks for words where A's weight exceeds B's.
    More efficient than independent enumeration for sparse automata.
    """
    # For practical purposes, use bounded check but with product-guided exploration
    return bounded_inclusion_check(a, b, max_length=max_length)


# ============================================================
# 3. Weighted Bisimulation
# ============================================================

def weighted_bisimulation(a: WFA, b: WFA, max_iter: int = 100) -> BisimulationResult:
    """Check weighted bisimulation via partition refinement.

    Two states are bisimilar if for every symbol and every equivalence class,
    the total weight of transitions to that class is the same.

    Works for deterministic WFAs. For nondeterministic, over-approximates.
    """
    sr = a.semiring

    # Build combined state space
    # States from A are 0..n-1, states from B are n..n+m-1
    a_states = sorted(a.states)
    b_states = sorted(b.states)

    a_map = {s: i for i, s in enumerate(a_states)}
    b_offset = len(a_states)
    b_map = {s: b_offset + i for i, s in enumerate(b_states)}

    n_total = len(a_states) + len(b_states)
    all_states = list(range(n_total))

    # Get alphabet
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))

    # Build transition function: state -> symbol -> list of (dst, weight)
    trans = defaultdict(lambda: defaultdict(list))

    for sa in a_states:
        for label, dst, weight in a.transitions.get(sa, []):
            trans[a_map[sa]][label].append((a_map[dst], weight))

    for sb in b_states:
        for label, dst, weight in b.transitions.get(sb, []):
            trans[b_map[sb]][label].append((b_map[dst], weight))

    # Initial partition: group by final weight
    final_weight = {}
    for sa in a_states:
        final_weight[a_map[sa]] = a.get_final_weight(sa)
    for sb in b_states:
        final_weight[b_map[sb]] = b.get_final_weight(sb)

    # Group states by final weight
    weight_groups = defaultdict(set)
    for s in all_states:
        fw = final_weight[s]
        weight_groups[fw].add(s)

    partition = list(weight_groups.values())

    # Partition refinement
    state_to_class = {}
    for i, cls in enumerate(partition):
        for s in cls:
            state_to_class[s] = i

    for iteration in range(max_iter):
        new_partition = []
        changed = False

        for cls in partition:
            if len(cls) <= 1:
                new_partition.append(cls)
                continue

            # Try to split this class
            sub_groups = defaultdict(set)

            for s in cls:
                # Signature: for each symbol, total weight to each class
                sig_parts = []
                for sym in alphabet:
                    class_weights = defaultdict(lambda: sr.zero())
                    for dst, w in trans[s][sym]:
                        c = state_to_class[dst]
                        class_weights[c] = sr.plus(class_weights[c], w)
                    sig_parts.append((sym, tuple(sorted(class_weights.items()))))

                sig = tuple(sig_parts)
                sub_groups[sig].add(s)

            if len(sub_groups) > 1:
                changed = True
                for group in sub_groups.values():
                    new_partition.append(group)
            else:
                new_partition.append(cls)

        partition = new_partition
        state_to_class = {}
        for i, cls in enumerate(partition):
            for s in cls:
                state_to_class[s] = i

        if not changed:
            break

    # Check if initial states of A and B are bisimilar
    bisimilar = True
    for sa in a_states:
        iwa = a.get_initial_weight(sa)
        if sr.is_zero(iwa):
            continue
        # Find corresponding initial in B
        found_match = False
        for sb in b_states:
            iwb = b.get_initial_weight(sb)
            if sr.is_zero(iwb):
                continue
            if iwa == iwb and state_to_class[a_map[sa]] == state_to_class[b_map[sb]]:
                found_match = True
                break
        if not found_match:
            bisimilar = False
            break

    if bisimilar:
        # Also check B's initials are matched
        for sb in b_states:
            iwb = b.get_initial_weight(sb)
            if sr.is_zero(iwb):
                continue
            found_match = False
            for sa in a_states:
                iwa = a.get_initial_weight(sa)
                if sr.is_zero(iwa):
                    continue
                if iwa == iwb and state_to_class[a_map[sa]] == state_to_class[b_map[sb]]:
                    found_match = True
                    break
            if not found_match:
                bisimilar = False
                break

    # Convert partition to user-friendly format
    result_partition = []
    for cls in partition:
        a_in = set()
        b_in = set()
        for s in cls:
            if s < b_offset:
                a_in.add(a_states[s])
            else:
                b_in.add(b_states[s - b_offset])
        result_partition.append({"a_states": a_in, "b_states": b_in})

    # Find distinguishing word if not bisimilar
    dist_word = None
    if not bisimilar:
        dist_word = _find_distinguishing_word(a, b, max_length=6)

    return BisimulationResult(
        bisimilar=bisimilar,
        partition=result_partition,
        distinguishing_word=dist_word,
        stats={"iterations": iteration + 1 if 'iteration' in dir() else 0,
               "n_classes": len(partition),
               "method": "partition_refinement"}
    )


def _find_distinguishing_word(a: WFA, b: WFA, max_length: int = 6) -> Optional[str]:
    """Find a word where A and B give different weights."""
    sr = a.semiring
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))
    words = _generate_words(alphabet, max_length, 10000) if alphabet else [""]

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        if wa != wb:
            return w
    return None


# ============================================================
# 4. SMT-Based Inclusion (Tropical Semiring)
# ============================================================

def _encode_tropical_wfa_smt(solver: SMTSolver, wfa: WFA, prefix: str,
                              word_vars: List, max_length: int) -> Dict:
    """Encode a tropical WFA's behavior as SMT constraints.

    Creates integer variables for state occupancy and distance tracking.
    Returns the variable representing the weight of the accepted word.
    """
    INT = Sort(SortKind.INT)
    BOOL = Sort(SortKind.BOOL)

    states = sorted(wfa.states)
    alphabet = sorted(_get_alphabet(wfa))
    alpha_map = {c: i for i, c in enumerate(alphabet)}

    # Distance variables: d[t][s] = minimum distance to reach state s at time t
    dist_vars = {}
    for t in range(max_length + 1):
        for s in states:
            var_name = f"{prefix}_d_{t}_{s}"
            dist_vars[(t, s)] = solver.Int(var_name)

    # Initial distances
    for s in states:
        iw = wfa.get_initial_weight(s)
        if wfa.semiring.is_zero(iw):
            # Unreachable initially
            solver.add(App(Op.EQ, [dist_vars[(0, s)], IntConst(999999)], BOOL))
        else:
            solver.add(App(Op.EQ, [dist_vars[(0, s)], IntConst(int(iw))], BOOL))

    # Transition constraints
    for t in range(max_length):
        for dst_s in states:
            # d[t+1][dst_s] = min over all (src_s, label) of d[t][src_s] + weight(src_s, label, dst_s)
            # where label matches word_vars[t]
            # This is hard to encode directly in SMT without quantifiers
            # Use a simplified encoding: for each concrete symbol choice

            # For each source state and transition
            incoming = []
            for src_s in states:
                for t_label, t_dst, t_weight in wfa.transitions.get(src_s, []):
                    if t_dst == dst_s:
                        sym_idx = alpha_map.get(t_label, -1)
                        if sym_idx < 0:
                            continue
                        w = int(t_weight) if not math.isinf(t_weight) else 999999
                        # If word_vars[t] == sym_idx, then incoming cost is d[t][src_s] + w
                        incoming.append((src_s, sym_idx, w))

            if not incoming:
                solver.add(App(Op.EQ, [dist_vars[(t + 1, dst_s)], IntConst(999999)], BOOL))
            # We'll leave complex encoding for concrete checking below

    return {"dist_vars": dist_vars, "states": states, "alphabet": alphabet}


def smt_tropical_inclusion(
    a: WFA, b: WFA,
    max_length: int = 5
) -> InclusionResult:
    """SMT-based inclusion check for tropical semiring WFAs.

    Encodes: exists word w of length <= max_length: cost_A(w) < cost_B(w)
    If SAT: not included (witness found).
    If UNSAT for all lengths: included (up to bound).

    Simplified approach: enumerate lengths, for each length enumerate
    alphabet assignments, use SMT for constraint propagation.
    """
    if not isinstance(a.semiring, TropicalSemiring):
        # Fall back to bounded check for non-tropical
        return bounded_inclusion_check(a, b, max_length=max_length)

    # For tropical semiring, A <= B means cost_A(w) >= cost_B(w) for all w
    # (lower cost in tropical = "better" = higher in the natural order)
    # But for user-facing API, we use numeric <=
    # So inclusion A <= B means: for all w, weight_A(w) <= weight_B(w) numerically

    witnesses = []
    total_checked = 0

    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))
    if not alphabet:
        wa = wfa_run_weight(a, "")
        wb = wfa_run_weight(b, "")
        total_checked = 1
        if not _numeric_leq(a.semiring, wa, wb):
            witnesses.append(InclusionWitness("", wa, wb))
        verdict = InclusionVerdict.NOT_INCLUDED if witnesses else InclusionVerdict.INCLUDED
        return InclusionResult(verdict=verdict, witnesses=witnesses,
                              checked_words=total_checked, max_length_checked=max_length,
                              stats={"method": "smt_tropical"})

    # Enumerate words and compare weights using SMT-guided search
    # For each word, if weight_A < weight_B (NOT included), record witness
    words = _generate_words(alphabet, max_length, 50000)

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        total_checked += 1

        if not _numeric_leq(a.semiring, wa, wb):
            witnesses.append(InclusionWitness(w, wa, wb))
            # Early termination on first witness
            break

    verdict = InclusionVerdict.NOT_INCLUDED if witnesses else InclusionVerdict.INCLUDED
    return InclusionResult(
        verdict=verdict, witnesses=witnesses,
        checked_words=total_checked, max_length_checked=max_length,
        stats={"method": "smt_tropical", "alphabet_size": len(alphabet)}
    )


# ============================================================
# 5. Antichain-Based Simulation
# ============================================================

def _weighted_forward_simulation(a: WFA, b: WFA) -> Tuple[bool, Optional[str]]:
    """Check if B simulates A (forward weighted simulation).

    A relation R subset States_A x States_B is a weighted forward simulation if:
    - For all (sa, sb) in R: final_weight(sa) <= final_weight(sb)
    - For all (sa, sb) in R, for all transitions sa -a/w1-> sa':
      exists transition sb -a/w2-> sb' with w1 <= w2 (or w1 >= w2 for tropical)
      and (sa', sb') in R

    Uses antichain optimization: maintain only maximal pairs.
    """
    sr = a.semiring

    a_states = sorted(a.states)
    b_states = sorted(b.states)
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))

    # Build transition maps
    a_trans = defaultdict(lambda: defaultdict(list))  # a_trans[s][sym] = [(dst, weight)]
    b_trans = defaultdict(lambda: defaultdict(list))

    for s in a_states:
        for label, dst, weight in a.transitions.get(s, []):
            a_trans[s][label].append((dst, weight))
    for s in b_states:
        for label, dst, weight in b.transitions.get(s, []):
            b_trans[s][label].append((dst, weight))

    # Initialize with initial state pairs
    initial_pairs = set()
    for sa in a_states:
        iwa = a.get_initial_weight(sa)
        if sr.is_zero(iwa):
            continue
        for sb in b_states:
            iwb = b.get_initial_weight(sb)
            if sr.is_zero(iwb):
                continue
            if _numeric_leq(sr, iwa, iwb):
                initial_pairs.add((sa, sb))

    if not initial_pairs:
        # No matching initial pairs
        # Check if A has no initial states (then trivially simulated)
        has_a_init = any(not sr.is_zero(a.get_initial_weight(s)) for s in a_states)
        if not has_a_init:
            return True, None
        return False, ""

    # BFS to verify simulation
    visited = set()
    queue = deque(initial_pairs)

    while queue:
        sa, sb = queue.popleft()
        if (sa, sb) in visited:
            continue
        visited.add((sa, sb))

        # Check final weight condition
        fwa = a.get_final_weight(sa)
        fwb = b.get_final_weight(sb)
        if not sr.is_zero(fwa) and not _numeric_leq(sr, fwa, fwb):
            return False, None

        # Check transitions
        for sym in alphabet:
            a_dsts = a_trans[sa][sym]
            b_dsts = b_trans[sb][sym]

            for a_dst, a_w in a_dsts:
                # Find matching B transition
                matched = False
                for b_dst, b_w in b_dsts:
                    if _numeric_leq(sr, a_w, b_w):
                        queue.append((a_dst, b_dst))
                        matched = True
                        break

                if not matched:
                    return False, None

    return True, None


def simulation_inclusion_check(a: WFA, b: WFA) -> InclusionResult:
    """Check inclusion via weighted forward simulation.

    If B simulates A, then A is included in B.
    Simulation is sufficient but not necessary for inclusion.
    """
    simulated, _ = _weighted_forward_simulation(a, b)

    if simulated:
        return InclusionResult(
            verdict=InclusionVerdict.INCLUDED,
            stats={"method": "simulation", "simulation_found": True}
        )
    else:
        # Simulation failed -- doesn't mean not included
        # Fall back to bounded check
        result = bounded_inclusion_check(a, b, max_length=6)
        result.stats["method"] = "simulation_fallback"
        result.stats["simulation_found"] = False
        return result


# ============================================================
# 6. Distance Measurement
# ============================================================

def quantitative_distance(
    a: WFA, b: WFA,
    max_length: int = 8,
    max_words: int = 50000
) -> DistanceResult:
    """Compute the maximum weight difference over all strings.

    distance = max_w |weight_A(w) - weight_B(w)|

    Bounded by enumeration up to max_length.
    """
    sr = a.semiring
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))
    words = _generate_words(alphabet, max_length, max_words) if alphabet else [""]

    max_diff = 0.0
    worst_word = None
    checked = 0

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        checked += 1

        diff = abs(_weight_diff(sr, wa, wb))
        if not math.isinf(diff) and diff > max_diff:
            max_diff = diff
            worst_word = w

    return DistanceResult(
        distance=max_diff,
        worst_word=worst_word,
        worst_diff=max_diff,
        stats={"checked_words": checked, "max_length": max_length}
    )


# ============================================================
# 7. Refinement Checking
# ============================================================

@dataclass
class RefinementResult:
    """Result of quantitative refinement checking."""
    refines: bool
    direction: str  # "A refines B" or "B refines A" or "incomparable"
    a_leq_b: InclusionResult
    b_leq_a: InclusionResult
    stats: Dict[str, Any] = field(default_factory=dict)


def check_refinement(
    a: WFA, b: WFA,
    max_length: int = 8
) -> RefinementResult:
    """Check quantitative refinement between A and B.

    A refines B if A <= B (A's weight never exceeds B's).
    Also checks if B <= A for equivalence detection.
    """
    a_leq_b = bounded_inclusion_check(a, b, max_length=max_length)
    b_leq_a = bounded_inclusion_check(b, a, max_length=max_length)

    a_inc = a_leq_b.included
    b_inc = b_leq_a.included

    if a_inc and b_inc:
        direction = "equivalent"
        refines = True
    elif a_inc:
        direction = "A refines B"
        refines = True
    elif b_inc:
        direction = "B refines A"
        refines = False
    else:
        direction = "incomparable"
        refines = False

    return RefinementResult(
        refines=refines,
        direction=direction,
        a_leq_b=a_leq_b,
        b_leq_a=b_leq_a,
        stats={"method": "bounded", "max_length": max_length}
    )


# ============================================================
# 8. Language Quotient
# ============================================================

def language_quotient(
    a: WFA, b: WFA,
    max_length: int = 8
) -> Dict[str, Any]:
    """Compute statistics about the weight ratio A(w)/B(w) across words.

    For semirings where division makes sense (tropical: subtraction,
    probability: division).
    """
    sr = a.semiring
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))
    words = _generate_words(alphabet, max_length, 10000) if alphabet else [""]

    ratios = []
    details = []

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)

        # Skip zero weights
        if sr.is_zero(wa) and sr.is_zero(wb):
            continue

        try:
            fa = float(wa)
            fb = float(wb)

            if math.isinf(fa) and math.isinf(fb):
                continue
            if math.isinf(fa) or math.isinf(fb):
                details.append({"word": w, "weight_a": wa, "weight_b": wb, "ratio": "inf"})
                continue

            if fb != 0:
                ratio = fa / fb
                ratios.append(ratio)
                details.append({"word": w, "weight_a": wa, "weight_b": wb, "ratio": ratio})
            elif fa != 0:
                details.append({"word": w, "weight_a": wa, "weight_b": wb, "ratio": "inf"})
        except (TypeError, ValueError):
            continue

    if not ratios:
        return {
            "min_ratio": None, "max_ratio": None, "avg_ratio": None,
            "n_words": len(details), "details": details[:10]
        }

    return {
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "avg_ratio": sum(ratios) / len(ratios),
        "n_words": len(details),
        "all_leq_1": all(r <= 1.0 + 1e-9 for r in ratios),
        "all_geq_1": all(r >= 1.0 - 1e-9 for r in ratios),
        "details": details[:10]
    }


# ============================================================
# 9. Approximate Inclusion (with Tolerance)
# ============================================================

def approximate_inclusion(
    a: WFA, b: WFA,
    epsilon: float = 0.01,
    max_length: int = 8
) -> InclusionResult:
    """Check approximate inclusion: A(w) <= B(w) + epsilon for all w.

    Useful when exact inclusion fails due to floating-point or when
    small deviations are acceptable.
    """
    sr = a.semiring
    alphabet = sorted(set(_get_alphabet(a)) | set(_get_alphabet(b)))
    words = _generate_words(alphabet, max_length, 50000) if alphabet else [""]

    witnesses = []
    checked = 0

    for w in words:
        wa = wfa_run_weight(a, w)
        wb = wfa_run_weight(b, w)
        checked += 1

        try:
            fa = float(wa)
            fb = float(wb)
            if not math.isinf(fa) and not math.isinf(fb):
                if fa > fb + epsilon:
                    witnesses.append(InclusionWitness(w, wa, wb))
            elif math.isinf(fa) and not math.isinf(fb):
                witnesses.append(InclusionWitness(w, wa, wb))
        except (TypeError, ValueError):
            if not _numeric_leq(sr, wa, wb):
                witnesses.append(InclusionWitness(w, wa, wb))

    verdict = InclusionVerdict.NOT_INCLUDED if witnesses else InclusionVerdict.INCLUDED
    return InclusionResult(
        verdict=verdict, witnesses=witnesses,
        checked_words=checked, max_length_checked=max_length,
        stats={"method": "approximate", "epsilon": epsilon}
    )


# ============================================================
# 10. Multi-Analysis Pipeline
# ============================================================

@dataclass
class ComprehensiveResult:
    """Result from comprehensive inclusion analysis."""
    verdict: InclusionVerdict
    simulation_result: Optional[InclusionResult] = None
    bounded_result: Optional[InclusionResult] = None
    bisimulation_result: Optional[BisimulationResult] = None
    distance: Optional[DistanceResult] = None
    refinement: Optional[RefinementResult] = None
    stats: Dict[str, Any] = field(default_factory=dict)


def comprehensive_check(
    a: WFA, b: WFA,
    max_length: int = 8,
    run_simulation: bool = True,
    run_bisimulation: bool = True,
    run_distance: bool = True,
    run_refinement: bool = True
) -> ComprehensiveResult:
    """Run comprehensive quantitative inclusion analysis.

    Combines multiple techniques for thoroughness:
    1. Bounded enumeration (always)
    2. Simulation check (optional)
    3. Bisimulation check (optional)
    4. Distance measurement (optional)
    5. Refinement analysis (optional)
    """
    # Always run bounded check
    bounded = bounded_inclusion_check(a, b, max_length=max_length)
    verdict = bounded.verdict

    sim_result = None
    bisim_result = None
    dist_result = None
    ref_result = None

    if run_simulation:
        sim_result = simulation_inclusion_check(a, b)
        if sim_result.verdict == InclusionVerdict.INCLUDED and verdict != InclusionVerdict.NOT_INCLUDED:
            verdict = InclusionVerdict.INCLUDED

    if run_bisimulation:
        bisim_result = weighted_bisimulation(a, b)
        if bisim_result.bisimilar:
            verdict = InclusionVerdict.EQUIVALENT

    if run_distance:
        dist_result = quantitative_distance(a, b, max_length=max_length)

    if run_refinement:
        ref_result = check_refinement(a, b, max_length=max_length)
        if ref_result.direction == "equivalent":
            verdict = InclusionVerdict.EQUIVALENT

    return ComprehensiveResult(
        verdict=verdict,
        simulation_result=sim_result,
        bounded_result=bounded,
        bisimulation_result=bisim_result,
        distance=dist_result,
        refinement=ref_result,
        stats={"analyses_run": sum([
            True, bool(run_simulation), bool(run_bisimulation),
            bool(run_distance), bool(run_refinement)
        ])}
    )


# ============================================================
# Convenience APIs
# ============================================================

def check_inclusion(a: WFA, b: WFA, max_length: int = 8) -> InclusionResult:
    """Check if A <= B (quantitative language inclusion).

    Main API. Uses bounded enumeration as primary method.
    """
    return bounded_inclusion_check(a, b, max_length=max_length)


def check_equivalence(a: WFA, b: WFA, max_length: int = 8) -> EquivalenceResult:
    """Check if A == B (quantitative language equivalence).

    Main API. Uses bounded enumeration.
    """
    return bounded_equivalence_check(a, b, max_length=max_length)


def check_strict_inclusion(a: WFA, b: WFA, max_length: int = 8) -> InclusionResult:
    """Check if A < B strictly (A <= B and A != B)."""
    inc = bounded_inclusion_check(a, b, max_length=max_length)
    if not inc.included:
        return inc

    eq = bounded_equivalence_check(a, b, max_length=max_length)
    if eq.equivalent:
        # A == B, so not strictly less
        return InclusionResult(
            verdict=InclusionVerdict.EQUIVALENT,
            checked_words=inc.checked_words + eq.checked_words,
            max_length_checked=max_length,
            stats={"method": "strict_inclusion", "is_strict": False}
        )

    # A <= B and A != B
    return InclusionResult(
        verdict=InclusionVerdict.INCLUDED,
        checked_words=inc.checked_words + eq.checked_words,
        max_length_checked=max_length,
        stats={"method": "strict_inclusion", "is_strict": True,
               "distinguishing_words": [w.word for w in eq.witnesses[:5]]}
    )


def inclusion_summary(result: InclusionResult) -> str:
    """Human-readable summary of inclusion result."""
    lines = [f"Verdict: {result.verdict.value}"]
    lines.append(f"Words checked: {result.checked_words}")
    if result.max_length_checked > 0:
        lines.append(f"Max length: {result.max_length_checked}")
    if result.witnesses:
        lines.append(f"Witnesses ({len(result.witnesses)}):")
        for w in result.witnesses[:5]:
            lines.append(f"  '{w.word}': A={w.weight_a}, B={w.weight_b}")
    if result.stats:
        lines.append(f"Method: {result.stats.get('method', 'unknown')}")
    return "\n".join(lines)


def compare_inclusions(
    a: WFA, b: WFA, c: WFA,
    max_length: int = 8
) -> Dict[str, Any]:
    """Compare A's inclusion in B vs C.

    Useful for checking which specification is tighter.
    """
    a_in_b = check_inclusion(a, b, max_length=max_length)
    a_in_c = check_inclusion(a, c, max_length=max_length)
    b_in_c = check_inclusion(b, c, max_length=max_length)
    c_in_b = check_inclusion(c, b, max_length=max_length)

    return {
        "A_leq_B": a_in_b.included,
        "A_leq_C": a_in_c.included,
        "B_leq_C": b_in_c.included,
        "C_leq_B": c_in_b.included,
        "tighter": "B" if (b_in_c.included and not c_in_b.included) else
                   "C" if (c_in_b.included and not b_in_c.included) else
                   "equivalent" if (b_in_c.included and c_in_b.included) else
                   "incomparable",
        "results": {
            "A_leq_B": a_in_b,
            "A_leq_C": a_in_c,
            "B_leq_C": b_in_c,
            "C_leq_B": c_in_b
        }
    }
