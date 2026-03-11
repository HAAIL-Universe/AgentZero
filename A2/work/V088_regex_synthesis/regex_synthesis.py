"""V088: Regex Synthesis from Examples

Synthesizes regular expressions from positive and negative string examples.

Composes:
- V084 (Symbolic Regex): regex parsing, compilation, equivalence checking
- V081 (Symbolic Automata): SFA operations, membership, determinization

Approaches:
1. Prefix Tree Generalization: Build prefix tree from positives, merge states
   that don't accept negatives (RPNI-style state merging)
2. Enumerative Synthesis: enumerate regexes by AST size, check against examples
3. Compositional: combine character classes, quantifiers from observed patterns
4. CEGIS: candidate + oracle refinement loop
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))

from symbolic_automata import (
    SFA, SFATransition, CharAlgebra, Pred, PChar, PRange, PTrue, PFalse,
    PAnd, POr, PNot, sfa_union, sfa_intersection, sfa_complement,
    sfa_is_equivalent, sfa_is_subset, sfa_from_string, sfa_concat,
    sfa_star, sfa_empty, sfa_epsilon, sfa_any_char, sfa_from_char_class
)
from symbolic_regex import (
    Regex, RegexKind, RLit, RDot, RClass, RNegClass, RConcat, RAlt,
    RStar, RPlus, ROptional, REpsilon, REmpty,
    compile_regex, compile_regex_dfa, regex_equivalent, regex_accepts_epsilon,
    regex_to_string, regex_size, RegexCompiler, parse_regex
)
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, FrozenSet
from collections import defaultdict, deque
from itertools import combinations


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Result of regex synthesis."""
    success: bool
    regex: Optional[Regex] = None          # Synthesized regex AST
    pattern: Optional[str] = None          # Regex as string
    method: str = ""                       # Which method succeeded
    stats: Dict = field(default_factory=dict)

    def accepts_all(self, positives, algebra=None):
        """Check synthesized regex accepts all positives."""
        if not self.success or self.regex is None:
            return False
        alg = algebra or CharAlgebra()
        compiler = RegexCompiler(alg)
        sfa = compiler.compile(self.regex).determinize()
        return all(sfa.accepts(s) for s in positives)

    def rejects_all(self, negatives, algebra=None):
        """Check synthesized regex rejects all negatives."""
        if not self.success or self.regex is None:
            return True
        alg = algebra or CharAlgebra()
        compiler = RegexCompiler(alg)
        sfa = compiler.compile(self.regex).determinize()
        return all(not sfa.accepts(s) for s in negatives)


# ---------------------------------------------------------------------------
# Helper: SFA to Regex conversion (state elimination)
# ---------------------------------------------------------------------------

def _sfa_to_regex(sfa):
    """Convert a deterministic SFA to a Regex AST via state elimination.

    Uses the classical state elimination algorithm:
    1. Add new start and accept states
    2. Eliminate states one by one, updating transition regexes
    3. Final regex is on the single remaining edge
    """
    sfa = sfa.determinize().trim()

    if sfa.is_empty():
        return REmpty()

    # Check for epsilon-only language
    if sfa.initial in sfa.accepting and not sfa._trans_from.get(sfa.initial, []):
        return REpsilon()

    # Build regex-labeled graph: state -> state -> regex
    # States are integers; we add new_start and new_accept
    states = set(sfa.states)
    new_start = max(states) + 1
    new_accept = new_start + 1

    # graph[s1][s2] = regex for edge s1 -> s2
    graph = defaultdict(lambda: defaultdict(lambda: None))

    # Original transitions: convert predicates to regex
    for s in states:
        for t in sfa._trans_from.get(s, []):
            r = _pred_to_regex(t.pred, sfa.algebra)
            if r is not None:
                existing = graph[t.src][t.dst]
                if existing is None:
                    graph[t.src][t.dst] = r
                else:
                    graph[t.src][t.dst] = RAlt(existing, r)

    # New start -> original start (epsilon)
    graph[new_start][sfa.initial] = REpsilon()

    # Original accepting -> new accept (epsilon)
    for acc in sfa.accepting:
        existing = graph[acc][new_accept]
        if existing is None:
            graph[acc][new_accept] = REpsilon()
        else:
            graph[acc][new_accept] = RAlt(existing, REpsilon())

    # State elimination: remove each original state
    eliminate_order = sorted(states)

    for q in eliminate_order:
        # Self-loop on q
        loop = graph[q][q]

        # For each pair (p, r) where p -> q and q -> r
        predecessors = []
        successors = []

        for p in list(graph.keys()):
            if p != q and graph[p][q] is not None:
                predecessors.append(p)

        for r in list(graph[q].keys()):
            if r != q and graph[q][r] is not None:
                successors.append(r)

        for p in predecessors:
            for r in successors:
                # New edge: p -> r via p->q (loop*)? q->r
                pq = graph[p][q]
                qr = graph[q][r]

                if loop is not None:
                    middle = _simplify_concat(pq, _simplify_concat(RStar(loop), qr))
                else:
                    middle = _simplify_concat(pq, qr)

                existing = graph[p][r]
                if existing is None:
                    graph[p][r] = middle
                else:
                    graph[p][r] = _simplify_alt(existing, middle)

        # Remove q from graph
        del graph[q]
        for p in list(graph.keys()):
            if q in graph[p]:
                del graph[p][q]

    result = graph[new_start][new_accept]
    if result is None:
        return REmpty()

    return result


def _pred_to_regex(pred, algebra):
    """Convert a predicate to a regex."""
    if pred.kind.name == 'TRUE':
        return RDot()
    elif pred.kind.name == 'FALSE':
        return None
    elif pred.kind.name == 'CHAR':
        return RLit(pred.char)
    elif pred.kind.name == 'RANGE':
        return RClass(((pred.lo, pred.hi),))
    elif pred.kind.name == 'NOT':
        # Negated predicate -> enumerate chars
        chars = list(algebra.enumerate(pred))
        if not chars:
            return None
        if len(chars) == 1:
            return RLit(chars[0])
        # Build character class from ranges
        return _chars_to_regex(chars)
    elif pred.kind.name == 'OR':
        left = _pred_to_regex(pred.left, algebra)
        right = _pred_to_regex(pred.right, algebra)
        if left is None:
            return right
        if right is None:
            return left
        return RAlt(left, right)
    elif pred.kind.name == 'AND':
        # Enumerate satisfying chars
        chars = list(algebra.enumerate(pred))
        if not chars:
            return None
        return _chars_to_regex(chars)
    else:
        # Fallback: enumerate
        chars = list(algebra.enumerate(pred))
        if not chars:
            return None
        return _chars_to_regex(chars)


def _chars_to_regex(chars):
    """Convert a list of characters to a compact regex (char class or alternation)."""
    if not chars:
        return None
    if len(chars) == 1:
        return RLit(chars[0])

    # Try to form ranges
    sorted_chars = sorted(chars)
    ranges = []
    i = 0
    while i < len(sorted_chars):
        start = sorted_chars[i]
        end = start
        while i + 1 < len(sorted_chars) and ord(sorted_chars[i + 1]) == ord(sorted_chars[i]) + 1:
            i += 1
            end = sorted_chars[i]
        ranges.append((start, end))
        i += 1

    if len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
        return RLit(ranges[0][0])

    return RClass(tuple(ranges))


def _simplify_concat(a, b):
    """Simplify concatenation."""
    if a.kind == RegexKind.EPSILON:
        return b
    if b.kind == RegexKind.EPSILON:
        return a
    if a.kind == RegexKind.EMPTY or b.kind == RegexKind.EMPTY:
        return REmpty()
    return RConcat(a, b)


def _simplify_alt(a, b):
    """Simplify alternation."""
    if a.kind == RegexKind.EMPTY:
        return b
    if b.kind == RegexKind.EMPTY:
        return a
    return RAlt(a, b)


# ---------------------------------------------------------------------------
# Approach 1: Prefix Tree (Trie) + State Merging (RPNI-style)
# ---------------------------------------------------------------------------

def _build_prefix_tree(positives, algebra):
    """Build a prefix tree acceptor (PTA) from positive examples.

    Returns an SFA that accepts exactly the positive strings.
    """
    if not positives:
        return sfa_empty(algebra)

    # Build trie as state machine
    next_state = [0]  # mutable counter
    transitions = []
    accepting = set()
    states = {0}

    # Trie structure: (state, char) -> state
    trie = {}

    for s in positives:
        current = 0
        for ch in s:
            key = (current, ch)
            if key not in trie:
                next_state[0] += 1
                new_state = next_state[0]
                trie[key] = new_state
                states.add(new_state)
                transitions.append(SFATransition(current, PChar(ch), new_state))
            current = trie[key]
        accepting.add(current)

    return SFA(states, 0, accepting, transitions, algebra)


def _merge_states(sfa, s1, s2, negatives):
    """Try to merge states s1 and s2. Return merged SFA if valid, None otherwise."""
    if s1 == s2:
        return sfa

    # Merging: replace all references to s2 with s1
    keep, remove = (s1, s2) if s1 <= s2 else (s1, s2)

    new_states = {s for s in sfa.states if s != remove}
    new_accepting = set()
    for s in sfa.accepting:
        new_accepting.add(keep if s == remove else s)

    new_initial = keep if sfa.initial == remove else sfa.initial

    # Remap transitions
    new_trans = []
    seen = set()
    for t in sfa._trans_from.get(t.src, []) if False else []:
        pass

    for s in sfa.states:
        for t in sfa._trans_from.get(s, []):
            src = keep if t.src == remove else t.src
            dst = keep if t.dst == remove else t.dst
            key = (src, t.pred, dst)
            if key not in seen:
                seen.add(key)
                new_trans.append(SFATransition(src, t.pred, dst))

    merged = SFA(new_states, new_initial, new_accepting, new_trans, sfa.algebra)

    # Validate: must reject all negatives
    for neg in negatives:
        if merged.accepts(neg):
            return None

    return merged


def rpni_synthesize(positives, negatives, algebra=None):
    """RPNI (Regular Positive and Negative Inference) algorithm.

    Builds a prefix tree from positives, then greedily merges states
    in order, keeping only merges that don't accept any negative example.

    Returns SynthesisResult with the generalized regex.
    """
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    # Build prefix tree acceptor
    pta = _build_prefix_tree(positives, alg)

    # Get states in BFS order (lexicographic/canonical)
    bfs_order = []
    visited = set()
    queue = deque([pta.initial])
    visited.add(pta.initial)
    while queue:
        s = queue.popleft()
        bfs_order.append(s)
        for t in sorted(pta._trans_from.get(s, []), key=lambda t: t.pred.char if t.pred.kind.name == 'CHAR' else ''):
            if t.dst not in visited:
                visited.add(t.dst)
                queue.append(t.dst)

    current = pta
    merges = 0

    # Try to merge each state pair (in BFS order, earlier states first)
    for i in range(len(bfs_order)):
        for j in range(i + 1, len(bfs_order)):
            s1 = bfs_order[i]
            s2 = bfs_order[j]

            # Check both states still exist in current automaton
            if s1 not in current.states or s2 not in current.states:
                continue

            merged = _merge_states(current, s1, s2, negatives)
            if merged is not None:
                current = merged
                merges += 1

    # Convert SFA to regex
    regex = _sfa_to_regex(current)
    pattern = regex_to_string(regex)

    return SynthesisResult(
        success=True,
        regex=regex,
        pattern=pattern,
        method="rpni",
        stats={"merges": merges, "pta_states": len(pta.states), "final_states": len(current.states)}
    )


# ---------------------------------------------------------------------------
# Approach 2: Enumerative Synthesis
# ---------------------------------------------------------------------------

def _enumerate_regexes(max_size, alphabet_chars):
    """Generate regexes by increasing AST size."""
    # Size 1: epsilon, dot, each literal
    if max_size >= 1:
        yield REpsilon()
        yield RDot()
        for c in alphabet_chars:
            yield RLit(c)

    # Build useful character classes from the alphabet
    digit_chars = [c for c in alphabet_chars if c.isdigit()]
    lower_chars = [c for c in alphabet_chars if c.islower()]
    upper_chars = [c for c in alphabet_chars if c.isupper()]

    char_classes = []
    if len(digit_chars) >= 2:
        char_classes.append(RClass((('0', '9'),)))
    if len(lower_chars) >= 2:
        char_classes.append(RClass((('a', 'z'),)))
    if len(upper_chars) >= 2:
        char_classes.append(RClass((('A', 'Z'),)))
    # Also try a class of exactly the observed chars (if > 1)
    if len(alphabet_chars) > 1:
        char_classes.append(_chars_to_regex(alphabet_chars))

    if max_size >= 2:
        for cc in char_classes:
            yield cc

    # Quantifiers on base regexes
    bases = []
    for c in alphabet_chars:
        bases.append(RLit(c))
    bases.append(RDot())
    bases.extend(char_classes)

    if max_size >= 2:
        for b in bases:
            yield RStar(b)
            yield RPlus(b)
            yield ROptional(b)

    if max_size >= 3:
        # Concatenation of two base elements
        for b1 in bases:
            for b2 in bases:
                yield RConcat(b1, b2)

        # Alternation of two base elements
        for i, b1 in enumerate(bases):
            for b2 in bases[i+1:]:
                yield RAlt(b1, b2)

        # Quantified concatenation
        for b1 in bases:
            for b2 in bases:
                yield RConcat(RStar(b1), b2)
                yield RConcat(b1, RStar(b2))

    if max_size >= 4:
        # Three-element concatenation
        for b1 in bases[:5]:  # limit to avoid explosion
            for b2 in bases[:5]:
                for b3 in bases[:5]:
                    yield RConcat(b1, RConcat(b2, b3))

        # Quantified alternation
        for i, b1 in enumerate(bases[:5]):
            for b2 in bases[i+1:5]:
                yield RStar(RAlt(b1, b2))
                yield RPlus(RAlt(b1, b2))


def _check_regex(regex, positives, negatives, algebra):
    """Check if regex accepts all positives and rejects all negatives."""
    compiler = RegexCompiler(algebra)
    try:
        sfa = compiler.compile(regex).determinize()
    except Exception:
        return False

    for p in positives:
        if not sfa.accepts(p):
            return False
    for n in negatives:
        if sfa.accepts(n):
            return False
    return True


def enumerative_synthesize(positives, negatives, max_size=4, algebra=None):
    """Enumerate regexes by size, return first consistent one.

    Prioritizes smaller (simpler) regexes.
    """
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    # Collect alphabet from examples
    alpha_chars = set()
    for s in positives:
        alpha_chars.update(s)
    for s in negatives:
        alpha_chars.update(s)
    alpha_chars = sorted(alpha_chars)

    checked = 0
    for regex in _enumerate_regexes(max_size, alpha_chars):
        checked += 1
        if _check_regex(regex, positives, negatives, alg):
            return SynthesisResult(
                success=True,
                regex=regex,
                pattern=regex_to_string(regex),
                method="enumerative",
                stats={"checked": checked, "size": regex_size(regex)}
            )

    return SynthesisResult(False, method="enumerative", stats={"checked": checked})


# ---------------------------------------------------------------------------
# Approach 3: Pattern-based synthesis (structural analysis of examples)
# ---------------------------------------------------------------------------

def _common_prefix(strings):
    """Find longest common prefix."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _common_suffix(strings):
    """Find longest common suffix."""
    if not strings:
        return ""
    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""
    return suffix


def _char_class_at(strings, pos):
    """Determine what characters appear at position pos across all strings."""
    chars = set()
    for s in strings:
        if pos < len(s):
            chars.add(s[pos])
    return chars


def _is_digit_set(chars):
    return chars.issubset(set('0123456789'))


def _is_lower_set(chars):
    return chars.issubset(set('abcdefghijklmnopqrstuvwxyz'))


def _is_upper_set(chars):
    return chars.issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))


def _is_alpha_set(chars):
    return _is_lower_set(chars) or _is_upper_set(chars) or chars.issubset(
        set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))


def _char_set_to_regex(chars):
    """Convert a set of characters to a regex element."""
    if len(chars) == 1:
        return RLit(next(iter(chars)))
    if _is_digit_set(chars):
        return RClass((('0', '9'),))
    if _is_lower_set(chars):
        return RClass((('a', 'z'),))
    if _is_upper_set(chars):
        return RClass((('A', 'Z'),))
    if _is_alpha_set(chars):
        return RClass((('a', 'z'), ('A', 'Z')))
    # Literal alternation
    sorted_chars = sorted(chars)
    return _chars_to_regex(sorted_chars)


def pattern_synthesize(positives, negatives, algebra=None):
    """Pattern-based synthesis: analyze structure of positive examples.

    Looks for common prefixes, suffixes, fixed-length patterns,
    character classes at each position.
    """
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    candidates = []

    # Strategy 1: Fixed-length pattern (all same length)
    lengths = set(len(s) for s in positives)
    if len(lengths) == 1:
        length = lengths.pop()
        # Build per-position character class
        parts = []
        for i in range(length):
            chars = _char_class_at(positives, i)
            parts.append(_char_set_to_regex(chars))

        if length == 0:
            regex = REpsilon()
        elif length == 1:
            regex = parts[0]
        else:
            regex = parts[0]
            for p in parts[1:]:
                regex = RConcat(regex, p)

        if _check_regex(regex, positives, negatives, alg):
            candidates.append(("fixed_length", regex))

    # Strategy 2: Common prefix + suffix + variable middle
    prefix = _common_prefix(positives)
    suffix = _common_suffix(positives)

    if prefix or suffix:
        # Build: prefix . middle . suffix
        # middle: figure out what varies
        middles = []
        for s in positives:
            mid = s[len(prefix):len(s) - len(suffix) if suffix else len(s)]
            middles.append(mid)

        mid_lengths = set(len(m) for m in middles)
        mid_chars = set()
        for m in middles:
            mid_chars.update(m)

        # Build middle regex
        if all(m == "" for m in middles):
            mid_regex = REpsilon()
        elif len(mid_lengths) == 1:
            # Fixed length middle
            ml = mid_lengths.pop()
            if ml == 1:
                mid_regex = _char_set_to_regex(mid_chars)
            else:
                mid_regex = _char_set_to_regex(mid_chars)
                for _ in range(ml - 1):
                    mid_regex = RConcat(mid_regex, _char_set_to_regex(mid_chars))
        else:
            # Variable length middle
            if mid_chars:
                base = _char_set_to_regex(mid_chars)
                min_len = min(mid_lengths)
                max_len = max(mid_lengths)
                if min_len == 0:
                    mid_regex = RStar(base)
                else:
                    mid_regex = RPlus(base)
            else:
                mid_regex = RStar(RDot())

        # Assemble
        parts = []
        for c in prefix:
            parts.append(RLit(c))
        parts.append(mid_regex)
        for c in suffix:
            parts.append(RLit(c))

        regex = parts[0]
        for p in parts[1:]:
            regex = _simplify_concat(regex, p)

        if _check_regex(regex, positives, negatives, alg):
            candidates.append(("prefix_suffix", regex))

    # Strategy 3: Alternation of exact strings
    if len(positives) <= 8:
        alts = [_string_to_regex(s) for s in sorted(set(positives))]
        if len(alts) == 1:
            regex = alts[0]
        else:
            regex = alts[0]
            for a in alts[1:]:
                regex = RAlt(regex, a)
        if _check_regex(regex, positives, negatives, alg):
            candidates.append(("alternation", regex))

    # Strategy 4: Character class star/plus
    all_chars = set()
    for s in positives:
        all_chars.update(s)

    if all_chars:
        base = _char_set_to_regex(all_chars)

        if any(len(s) == 0 for s in positives):
            regex = RStar(base)
        else:
            regex = RPlus(base)

        if _check_regex(regex, positives, negatives, alg):
            candidates.append(("char_class_repeat", regex))

    # Pick smallest candidate
    if candidates:
        candidates.sort(key=lambda x: regex_size(x[1]))
        method, regex = candidates[0]
        return SynthesisResult(
            success=True,
            regex=regex,
            pattern=regex_to_string(regex),
            method=f"pattern:{method}",
            stats={"candidates": len(candidates), "chosen": method}
        )

    return SynthesisResult(False, method="pattern", stats={"candidates": 0})


def _string_to_regex(s):
    """Convert a string to a regex (concatenation of literals)."""
    if not s:
        return REpsilon()
    if len(s) == 1:
        return RLit(s[0])
    parts = [RLit(c) for c in s]
    result = parts[0]
    for p in parts[1:]:
        result = RConcat(result, p)
    return result


# ---------------------------------------------------------------------------
# Approach 4: CEGIS (Counterexample-Guided Inductive Synthesis)
# ---------------------------------------------------------------------------

def cegis_synthesize(positives, negatives, max_rounds=10, max_size=4, algebra=None):
    """CEGIS loop: generate candidate, check, refine.

    Uses pattern synthesis to generate initial candidate, then
    checks against full example set and refines.
    """
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    # Start with subset of examples
    pos_subset = [positives[0]]
    neg_subset = list(negatives[:2])

    for round_num in range(max_rounds):
        # Try pattern synthesis on subset
        result = pattern_synthesize(pos_subset, neg_subset, alg)

        if not result.success:
            # Try enumerative on subset
            result = enumerative_synthesize(pos_subset, neg_subset, max_size, alg)

        if not result.success:
            # Expand subset and retry
            if len(pos_subset) < len(positives):
                pos_subset.append(positives[len(pos_subset)])
            continue

        # Check against full example set
        if result.accepts_all(positives, alg) and result.rejects_all(negatives, alg):
            result.method = f"cegis:{result.method}"
            result.stats["cegis_rounds"] = round_num + 1
            return result

        # Find counterexample and add to subset
        compiler = RegexCompiler(alg)
        sfa = compiler.compile(result.regex).determinize()

        # Find false negative (positive not accepted)
        for p in positives:
            if not sfa.accepts(p) and p not in pos_subset:
                pos_subset.append(p)
                break

        # Find false positive (negative accepted)
        for n in negatives:
            if sfa.accepts(n) and n not in neg_subset:
                neg_subset.append(n)
                break

    # Final attempt with all examples
    result = pattern_synthesize(positives, negatives, alg)
    if result.success:
        result.method = f"cegis:{result.method}"
        result.stats["cegis_rounds"] = max_rounds
        return result

    return SynthesisResult(False, method="cegis", stats={"rounds": max_rounds})


# ---------------------------------------------------------------------------
# Approach 5: DFA Learning (L*-style with examples as oracle)
# ---------------------------------------------------------------------------

def _membership_oracle(word, positives, negatives):
    """Membership oracle based on examples.
    Returns True if in positives, False if in negatives, None if unknown.
    """
    if word in positives:
        return True
    if word in negatives:
        return False
    return None


def _equivalence_oracle_examples(sfa, positives, negatives):
    """Check SFA against examples. Return counterexample or None."""
    for p in positives:
        if not sfa.accepts(p):
            return (p, True)  # should accept
    for n in negatives:
        if sfa.accepts(n):
            return (n, False)  # should reject
    return None


def lstar_synthesize(positives, negatives, algebra=None):
    """Simplified L*-style learning using examples as oracle.

    Builds an observation table and constructs a DFA.
    Counterexamples from pos/neg sets drive table expansion.
    """
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    pos_set = set(positives)
    neg_set = set(negatives)

    # Collect alphabet
    alpha = set()
    for s in positives:
        alpha.update(s)
    for s in negatives:
        alpha.update(s)
    alpha = sorted(alpha)

    if not alpha:
        # All empty strings
        if "" in pos_set:
            return SynthesisResult(True, REpsilon(), "", "lstar", {})
        return SynthesisResult(False, method="lstar")

    # Observation table: S (prefixes), E (suffixes), T (membership)
    S = [""]  # prefix-closed set of states
    E = [""]  # suffix-closed set of experiments
    T = {}    # (s, e) -> bool

    def query(word):
        """Query membership."""
        if word in pos_set:
            return True
        if word in neg_set:
            return False
        # Heuristic: unknown -> False (conservative)
        return False

    def fill_table():
        for s in S:
            for a in alpha:
                sa = s + a
                for e in E:
                    T[(sa, e)] = query(sa + e)
            for e in E:
                T[(s, e)] = query(s + e)

    def row(s):
        return tuple(T.get((s, e), query(s + e)) for e in E)

    def is_closed():
        """Check if for every s.a, there's an equivalent row in S."""
        for s in S:
            for a in alpha:
                sa = s + a
                sa_row = row(sa)
                if not any(row(s2) == sa_row for s2 in S):
                    return (False, sa)
        return (True, None)

    def is_consistent():
        """Check if same-row states have same-row successors."""
        for i, s1 in enumerate(S):
            for s2 in S[i+1:]:
                if row(s1) == row(s2):
                    for a in alpha:
                        if row(s1 + a) != row(s2 + a):
                            # Find distinguishing suffix
                            for e in E:
                                if T.get((s1 + a, e), query(s1 + a + e)) != T.get((s2 + a, e), query(s2 + a + e)):
                                    return (False, a + e)
                            return (False, a)
        return (True, None)

    def build_hypothesis():
        """Build DFA from observation table."""
        # States: distinct rows of S
        row_to_state = {}
        states = set()
        state_counter = 0

        for s in S:
            r = row(s)
            if r not in row_to_state:
                row_to_state[r] = state_counter
                states.add(state_counter)
                state_counter += 1

        initial = row_to_state[row("")]
        accepting = set()
        for s in S:
            r = row(s)
            sid = row_to_state[r]
            if T.get((s, ""), query(s)):
                accepting.add(sid)

        transitions = []
        seen_trans = set()
        for s in S:
            src = row_to_state[row(s)]
            for a in alpha:
                sa_row = row(s + a)
                if sa_row in row_to_state:
                    dst = row_to_state[sa_row]
                    key = (src, a, dst)
                    if key not in seen_trans:
                        seen_trans.add(key)
                        transitions.append(SFATransition(src, PChar(a), dst))

        return SFA(states, initial, accepting, transitions, alg)

    fill_table()

    max_iterations = 50
    for iteration in range(max_iterations):
        # Close
        closed, witness = is_closed()
        if not closed:
            S.append(witness)
            fill_table()
            continue

        # Make consistent
        consistent, new_suffix = is_consistent()
        if not consistent:
            if new_suffix not in E:
                E.append(new_suffix)
            fill_table()
            continue

        # Build hypothesis and check
        hyp = build_hypothesis()
        ce = _equivalence_oracle_examples(hyp, positives, negatives)

        if ce is None:
            # Found consistent DFA
            regex = _sfa_to_regex(hyp)
            return SynthesisResult(
                success=True,
                regex=regex,
                pattern=regex_to_string(regex),
                method="lstar",
                stats={"iterations": iteration + 1, "states": len(S), "experiments": len(E)}
            )

        # Add counterexample prefixes
        ce_word, _should_accept = ce
        for i in range(len(ce_word) + 1):
            prefix = ce_word[:i]
            if prefix not in S:
                S.append(prefix)
        fill_table()

    # Return best effort
    hyp = build_hypothesis()
    regex = _sfa_to_regex(hyp)
    return SynthesisResult(
        success=True,
        regex=regex,
        pattern=regex_to_string(regex),
        method="lstar",
        stats={"iterations": max_iterations, "converged": False}
    )


# ---------------------------------------------------------------------------
# Main API: Multi-strategy synthesis
# ---------------------------------------------------------------------------

def synthesize_regex(positives, negatives=None, algebra=None, strategy="auto",
                     max_size=4, max_rounds=10):
    """Synthesize a regex from positive and negative examples.

    Args:
        positives: List of strings that must be accepted
        negatives: List of strings that must be rejected (default: [])
        algebra: CharAlgebra instance (default: CharAlgebra())
        strategy: "auto", "pattern", "enumerative", "rpni", "lstar", "cegis"
        max_size: Maximum AST size for enumerative search
        max_rounds: Maximum CEGIS rounds

    Returns:
        SynthesisResult with synthesized regex (or failure)
    """
    if negatives is None:
        negatives = []
    alg = algebra or CharAlgebra()

    if not positives:
        return SynthesisResult(False, stats={"reason": "no positive examples"})

    if strategy == "pattern":
        return pattern_synthesize(positives, negatives, alg)
    elif strategy == "enumerative":
        return enumerative_synthesize(positives, negatives, max_size, alg)
    elif strategy == "rpni":
        return rpni_synthesize(positives, negatives, alg)
    elif strategy == "lstar":
        return lstar_synthesize(positives, negatives, alg)
    elif strategy == "cegis":
        return cegis_synthesize(positives, negatives, max_rounds, max_size, alg)

    # Auto: try strategies in order of preference
    # 1. Pattern (fast, structural)
    result = pattern_synthesize(positives, negatives, alg)
    if result.success:
        return result

    # 2. Enumerative (small regexes)
    result = enumerative_synthesize(positives, negatives, max_size, alg)
    if result.success:
        return result

    # 3. L* learning
    result = lstar_synthesize(positives, negatives, alg)
    if result.success:
        return result

    # 4. RPNI
    result = rpni_synthesize(positives, negatives, alg)
    if result.success:
        return result

    # 5. CEGIS (last resort)
    return cegis_synthesize(positives, negatives, max_rounds, max_size, alg)


def verify_synthesis(regex, positives, negatives, algebra=None):
    """Verify that a regex is consistent with examples.

    Returns dict with:
        - valid: True if accepts all pos and rejects all neg
        - false_negatives: positive examples not accepted
        - false_positives: negative examples accepted
    """
    alg = algebra or CharAlgebra()
    compiler = RegexCompiler(alg)

    if isinstance(regex, str):
        regex = parse_regex(regex)

    sfa = compiler.compile(regex).determinize()

    false_neg = [p for p in positives if not sfa.accepts(p)]
    false_pos = [n for n in negatives if sfa.accepts(n)]

    return {
        "valid": len(false_neg) == 0 and len(false_pos) == 0,
        "false_negatives": false_neg,
        "false_positives": false_pos,
        "accepts_count": sum(1 for p in positives if sfa.accepts(p)),
        "rejects_count": sum(1 for n in negatives if not sfa.accepts(n)),
    }


def compare_strategies(positives, negatives=None, algebra=None):
    """Compare all synthesis strategies on the same examples.

    Returns dict with per-strategy results.
    """
    if negatives is None:
        negatives = []
    alg = algebra or CharAlgebra()

    strategies = ["pattern", "enumerative", "rpni", "lstar", "cegis"]
    results = {}

    for strat in strategies:
        try:
            r = synthesize_regex(positives, negatives, alg, strategy=strat)
            valid = verify_synthesis(r.regex, positives, negatives, alg) if r.success else None
            results[strat] = {
                "success": r.success,
                "pattern": r.pattern,
                "method": r.method,
                "size": regex_size(r.regex) if r.regex else None,
                "valid": valid["valid"] if valid else False,
                "stats": r.stats,
            }
        except Exception as e:
            results[strat] = {"success": False, "error": str(e)}

    return results


def synthesize_from_language(target_pattern, num_pos=10, num_neg=10, algebra=None):
    """Synthesize a regex equivalent to target_pattern using generated examples.

    Generates positive and negative examples from the target regex,
    then synthesizes a (possibly different but equivalent) regex.
    """
    alg = algebra or CharAlgebra()

    # Compile target to SFA
    target_sfa = compile_regex(target_pattern, alg).determinize()
    complement_sfa = sfa_complement(target_sfa)

    # Generate positive examples (shortest words accepted)
    positives = []
    _bfs_words(target_sfa, positives, num_pos, alg)

    # Generate negative examples
    negatives = []
    _bfs_words(complement_sfa, negatives, num_neg, alg)

    if not positives:
        return SynthesisResult(False, stats={"reason": "target language empty"})

    result = synthesize_regex(positives, negatives, alg)

    if result.success:
        # Check equivalence with target
        result_sfa = RegexCompiler(alg).compile(result.regex).determinize()
        is_equiv = sfa_is_equivalent(result_sfa, target_sfa)
        result.stats["equivalent_to_target"] = is_equiv
        result.stats["positives_used"] = len(positives)
        result.stats["negatives_used"] = len(negatives)

    return result


def _bfs_words(sfa, out, max_count, algebra):
    """BFS to find shortest accepted words."""
    if sfa.is_empty():
        return

    # BFS: (state, word_so_far)
    queue = deque([(sfa.initial, "")])
    visited = {sfa.initial}

    # Get a few alphabet chars for BFS
    sample_chars = []
    for t in sfa._trans_from.get(sfa.initial, []):
        w = algebra.witness(t.pred)
        if w is not None:
            sample_chars.append(w)

    # Also get chars from all transitions
    all_trans_chars = set()
    for s in sfa.states:
        for t in sfa._trans_from.get(s, []):
            w = algebra.witness(t.pred)
            if w is not None:
                all_trans_chars.add(w)
    sample_chars = sorted(set(sample_chars) | all_trans_chars)

    if not sample_chars:
        sample_chars = [chr(c) for c in range(32, 127)]

    while queue and len(out) < max_count:
        state, word = queue.popleft()

        if state in sfa.accepting:
            if word not in out:
                out.append(word)
                if len(out) >= max_count:
                    return

        for t in sfa._trans_from.get(state, []):
            w = algebra.witness(t.pred)
            if w is not None and t.dst not in visited:
                visited.add(t.dst)
                queue.append((t.dst, word + (w if isinstance(w, str) else chr(w))))

    # If BFS didn't find enough, try longer words
    if len(out) < max_count:
        for t in sfa._trans_from.get(sfa.initial, []):
            w = algebra.witness(t.pred)
            if w is not None:
                for t2 in sfa._trans_from.get(t.dst, []):
                    w2 = algebra.witness(t2.pred)
                    if w2 is not None:
                        word = (w if isinstance(w, str) else chr(w)) + (w2 if isinstance(w2, str) else chr(w2))
                        if sfa.accepts(word) and word not in out:
                            out.append(word)
                            if len(out) >= max_count:
                                return
