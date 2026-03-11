"""V093: Tree Regular Language Learning

Active learning of tree regular languages via an L*-style algorithm adapted
for bottom-up tree automata over ranked alphabets.

Key concepts:
- Tree context: a tree with a single hole [] that can be filled with a subtree
- Observation table: maps (tree, context) pairs to membership results
- Teacher: answers membership queries (is tree in language?) and
  equivalence queries (is hypothesis correct? if not, give counterexample)
- Learner: builds hypothesis BUTA from observation table

Composes V089 (tree automata) for hypothesis construction and operations.

References: Drewes & Hogberg "Learning Tree Languages from Text" (2007),
Habrard et al. "Learning Multiplicity Tree Automata" (2006).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_automata import (
    Symbol, RankedAlphabet, Tree, BottomUpTreeAutomaton,
    buta_union, buta_intersection, buta_complement,
    check_language_equivalence, check_language_emptiness,
)
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, FrozenSet, Any
from collections import deque
import itertools


# --- Tree Contexts ---

class Context:
    """A tree context: a tree with exactly one hole [].

    Represented as a tree where the hole is a special marker node.
    Filling the hole with a tree t produces a complete tree.
    """
    HOLE = "__HOLE__"

    def __init__(self, tree: Tree):
        """Create context from a tree containing a HOLE node."""
        self.tree = tree

    @staticmethod
    def trivial() -> 'Context':
        """The trivial context [] (just a hole)."""
        return Context(Tree(Context.HOLE))

    @staticmethod
    def make(symbol: str, children_before: List[Tree],
             inner: 'Context', children_after: List[Tree]) -> 'Context':
        """Build context: symbol(children_before..., inner_context, children_after...)."""
        all_children = list(children_before) + [inner.tree] + list(children_after)
        return Context(Tree(symbol, all_children))

    def fill(self, t: Tree) -> Tree:
        """Fill the hole with tree t."""
        return self._fill_rec(self.tree, t)

    def _fill_rec(self, node: Tree, t: Tree) -> Tree:
        if node.symbol == Context.HOLE:
            return t
        new_children = [self._fill_rec(c, t) for c in node.children]
        return Tree(node.symbol, new_children)

    def is_trivial(self) -> bool:
        return self.tree.symbol == Context.HOLE and not self.tree.children

    def depth(self) -> int:
        """Depth of the hole in the context."""
        return self._hole_depth(self.tree)

    def _hole_depth(self, node: Tree) -> int:
        if node.symbol == Context.HOLE:
            return 0
        for c in node.children:
            d = self._hole_depth(c)
            if d >= 0:
                return 1 + d
        return -1

    def __repr__(self):
        return f"Context({self._repr_tree(self.tree)})"

    def _repr_tree(self, node: Tree) -> str:
        if node.symbol == Context.HOLE:
            return "[]"
        if not node.children:
            return node.symbol
        children_str = ", ".join(self._repr_tree(c) for c in node.children)
        return f"{node.symbol}({children_str})"

    def _to_tuple(self, node: Tree) -> tuple:
        if node.symbol == Context.HOLE:
            return (Context.HOLE,)
        return (node.symbol, tuple(self._to_tuple(c) for c in node.children))

    def __eq__(self, other):
        if not isinstance(other, Context):
            return False
        return self._to_tuple(self.tree) == self._to_tuple(other.tree)

    def __hash__(self):
        return hash(self._to_tuple(self.tree))


def tree_to_tuple(t: Tree) -> tuple:
    """Convert tree to hashable tuple for use as dict key."""
    if not t.children:
        return (t.symbol,)
    return (t.symbol, tuple(tree_to_tuple(c) for c in t.children))


def tuple_to_tree(tup: tuple) -> Tree:
    """Convert tuple back to tree."""
    if len(tup) == 1:
        return Tree(tup[0])
    return Tree(tup[0], [tuple_to_tree(c) for c in tup[1]])


# --- Teachers ---

class Teacher:
    """Abstract teacher interface for L* learning."""

    def membership(self, t: Tree) -> bool:
        """Is tree t in the target language?"""
        raise NotImplementedError

    def equivalence(self, hypothesis: BottomUpTreeAutomaton) -> Optional[Tree]:
        """Is hypothesis correct? Returns None if yes, counterexample tree if no."""
        raise NotImplementedError


class AutomatonTeacher(Teacher):
    """Teacher backed by a BUTA (for testing/benchmarking)."""

    def __init__(self, target: BottomUpTreeAutomaton):
        self.target = target
        self.membership_count = 0
        self.equivalence_count = 0

    def membership(self, t: Tree) -> bool:
        self.membership_count += 1
        return self.target.accepts(t)

    def equivalence(self, hypothesis: BottomUpTreeAutomaton) -> Optional[Tree]:
        self.equivalence_count += 1
        # Check both directions: target \ hypothesis and hypothesis \ target
        equiv = check_language_equivalence(self.target, hypothesis)
        if equiv["equivalent"]:
            return None
        return equiv.get("counterexample")


class PredicateTeacher(Teacher):
    """Teacher backed by a predicate function over trees.

    Equivalence queries use bounded enumeration to find counterexamples.
    """

    def __init__(self, alphabet: RankedAlphabet, predicate: Callable[[Tree], bool],
                 max_ce_size: int = 8):
        self.alphabet = alphabet
        self.predicate = predicate
        self.max_ce_size = max_ce_size
        self.membership_count = 0
        self.equivalence_count = 0

    def membership(self, t: Tree) -> bool:
        self.membership_count += 1
        return self.predicate(t)

    def equivalence(self, hypothesis: BottomUpTreeAutomaton) -> Optional[Tree]:
        self.equivalence_count += 1
        for t in self._enumerate_trees(self.max_ce_size):
            pred_result = self.predicate(t)
            hyp_result = hypothesis.accepts(t)
            if pred_result != hyp_result:
                return t
        return None

    def _enumerate_trees(self, max_size: int) -> List[Tree]:
        """Enumerate trees up to max_size nodes."""
        by_size: Dict[int, List[Tree]] = {0: [], 1: []}
        # Size 1: constants
        for s in self.alphabet:
            if s.arity == 0:
                by_size[1].append(Tree(s.name))

        for size in range(2, max_size + 1):
            by_size[size] = []
            for s in self.alphabet:
                if s.arity == 0:
                    continue
                # Distribute size-1 among s.arity children (each >= 1)
                for partition in self._partitions(size - 1, s.arity):
                    child_options = [by_size.get(p, []) for p in partition]
                    for combo in itertools.product(*child_options):
                        by_size[size].append(Tree(s.name, list(combo)))

        result = []
        for size in range(1, max_size + 1):
            result.extend(by_size[size])
        return result

    def _partitions(self, n: int, k: int) -> List[Tuple[int, ...]]:
        """Partition n into k parts, each >= 1."""
        if k == 1:
            return [(n,)]
        result = []
        for first in range(1, n - k + 2):
            for rest in self._partitions(n - first, k - 1):
                result.append((first,) + rest)
        return result


# --- Observation Table ---

class ObservationTable:
    """Observation table for tree L* learning.

    The table maps (tree_term, context) -> membership result.

    Rows are indexed by tree terms (representative trees for states).
    Columns are indexed by contexts (distinguishing contexts).

    For a tree term t and context c, table[t][c] = membership(c[t])
    where c[t] means filling the hole in c with t.
    """

    def __init__(self, alphabet: RankedAlphabet, teacher: Teacher):
        self.alphabet = alphabet
        self.teacher = teacher

        # S: state representatives (tree tuples)
        self.S: List[tuple] = []
        # R: boundary (extensions of S by one symbol application)
        self.R: List[tuple] = []
        # E: distinguishing contexts
        self.E: List[Context] = [Context.trivial()]
        # Table entries: (tree_tuple, context_index) -> bool
        self.table: Dict[Tuple[tuple, int], bool] = {}

    def row(self, t_tuple: tuple) -> Tuple[bool, ...]:
        """Get the row vector for a tree term."""
        return tuple(self.table.get((t_tuple, i), False)
                     for i in range(len(self.E)))

    def _query(self, t_tuple: tuple, ctx_idx: int) -> bool:
        """Query membership for tree t in context E[ctx_idx]."""
        key = (t_tuple, ctx_idx)
        if key not in self.table:
            t = tuple_to_tree(t_tuple)
            ctx = self.E[ctx_idx]
            filled = ctx.fill(t)
            self.table[key] = self.teacher.membership(filled)
        return self.table[key]

    def fill_row(self, t_tuple: tuple):
        """Ensure all entries for a row are computed."""
        for i in range(len(self.E)):
            self._query(t_tuple, i)

    def fill_all(self):
        """Fill all table entries."""
        for t in self.S + self.R:
            self.fill_row(t)

    def initialize(self):
        """Initialize table with constants as S, one-symbol extensions as R."""
        # S = constants (trees of size 1)
        for s in self.alphabet:
            if s.arity == 0:
                t_tuple = tree_to_tuple(Tree(s.name))
                if t_tuple not in self.S:
                    self.S.append(t_tuple)

        # R = one-symbol extensions
        self._compute_extensions()
        self.fill_all()

    def _compute_extensions(self):
        """Compute R: apply each symbol to combinations from S."""
        existing_R = set(tuple(r) if isinstance(r, list) else r for r in self.R)
        for s in self.alphabet:
            if s.arity == 0:
                continue
            # Apply s to all combinations of S members
            all_terms = self.S  # Only use S members as children
            for combo in itertools.product(all_terms, repeat=s.arity):
                children = [tuple_to_tree(c) for c in combo]
                new_tree = Tree(s.name, children)
                new_tuple = tree_to_tuple(new_tree)
                if new_tuple not in self.S and new_tuple not in existing_R:
                    self.R.append(new_tuple)
                    existing_R.add(new_tuple)

    def is_closed(self) -> Optional[tuple]:
        """Check if table is closed: every row in R has a matching row in S.

        Returns an R element that needs promotion, or None if closed.
        """
        s_rows = {self.row(s) for s in self.S}
        for r in self.R:
            if self.row(r) not in s_rows:
                return r
        return None

    def is_consistent(self) -> Optional[Tuple[tuple, tuple, Context]]:
        """Check consistency: if two S rows are equal, their extensions must agree.

        Returns (s1, s2, new_context) if inconsistent, None if consistent.
        """
        for i, s1 in enumerate(self.S):
            for s2 in self.S[i+1:]:
                if self.row(s1) != self.row(s2):
                    continue
                # s1 and s2 have the same row -- check their extensions
                for sym in self.alphabet:
                    if sym.arity == 0:
                        continue
                    for pos in range(sym.arity):
                        # Build extension: sym(..., s1_or_s2 at pos, ...)
                        for other_combos in itertools.product(self.S, repeat=sym.arity - 1):
                            others = list(other_combos)
                            args1 = others[:pos] + [s1] + others[pos:]
                            args2 = others[:pos] + [s2] + others[pos:]

                            t1 = Tree(sym.name, [tuple_to_tree(a) for a in args1])
                            t2 = Tree(sym.name, [tuple_to_tree(a) for a in args2])
                            tt1 = tree_to_tuple(t1)
                            tt2 = tree_to_tuple(t2)

                            # Check all existing contexts
                            for ctx_idx, ctx in enumerate(self.E):
                                self._query(tt1, ctx_idx)
                                self._query(tt2, ctx_idx)
                                if self.table.get((tt1, ctx_idx)) != self.table.get((tt2, ctx_idx)):
                                    # Found inconsistency: new context =
                                    # ctx composed with sym(..., [], ...)
                                    before = [tuple_to_tree(a) for a in others[:pos]]
                                    after = [tuple_to_tree(a) for a in others[pos:]]
                                    new_ctx = Context.make(sym.name, before, ctx, after)
                                    return (s1, s2, new_ctx)
        return None

    def close(self):
        """Make table closed by promoting R elements to S."""
        while True:
            unclosed = self.is_closed()
            if unclosed is None:
                break
            # Promote unclosed element from R to S
            self.R.remove(unclosed)
            self.S.append(unclosed)
            # Recompute extensions
            self._compute_extensions()
            self.fill_all()

    def make_consistent(self):
        """Make table consistent by adding distinguishing contexts."""
        while True:
            incon = self.is_consistent()
            if incon is None:
                break
            s1, s2, new_ctx = incon
            if new_ctx not in self.E:
                self.E.append(new_ctx)
                self.fill_all()

    def build_hypothesis(self) -> BottomUpTreeAutomaton:
        """Build a hypothesis BUTA from the closed, consistent table.

        States correspond to distinct rows in S.
        """
        hyp = BottomUpTreeAutomaton(self.alphabet)

        # Map each distinct row to a state
        row_to_state: Dict[Tuple[bool, ...], str] = {}
        state_counter = 0
        for s in self.S:
            r = self.row(s)
            if r not in row_to_state:
                state_name = f"q{state_counter}"
                row_to_state[r] = state_name
                state_counter += 1
                # Final if the trivial context column is True
                is_final = r[0] if len(r) > 0 else False
                hyp.add_state(state_name, final=is_final)

        # Add transitions for each symbol
        for sym in self.alphabet:
            if sym.arity == 0:
                # Constant: () -> state
                t = Tree(sym.name)
                tt = tree_to_tuple(t)
                if tt in self.S:
                    r = self.row(tt)
                    if r in row_to_state:
                        hyp.add_transition(sym.name, (), row_to_state[r])
            else:
                # For each combination of S-rows as children states
                s_rows = list(row_to_state.keys())
                for combo_rows in itertools.product(s_rows, repeat=sym.arity):
                    # Find representative S terms for each child row
                    child_terms = []
                    valid = True
                    for cr in combo_rows:
                        found = False
                        for s in self.S:
                            if self.row(s) == cr:
                                child_terms.append(s)
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Build the composite tree
                    children = [tuple_to_tree(ct) for ct in child_terms]
                    composite = Tree(sym.name, children)
                    ct = tree_to_tuple(composite)

                    # Find what row this composite maps to
                    # It might be in S or R
                    self.fill_row(ct)
                    r = self.row(ct)
                    if r in row_to_state:
                        child_states = tuple(row_to_state[cr] for cr in combo_rows)
                        hyp.add_transition(sym.name, child_states, row_to_state[r])

        return hyp

    def process_counterexample(self, ce: Tree):
        """Process counterexample by adding all subtrees to S and generating contexts.

        Standard approach: promote all CE subtrees to S (forcing the table to
        distinguish them), then add alphabet-derived contexts for discrimination.
        """
        # Add all subtrees of CE to S
        subtrees = ce.subtrees()
        for st in subtrees:
            st_tuple = tree_to_tuple(st)
            if st_tuple not in self.S:
                if st_tuple in self.R:
                    self.R.remove(st_tuple)
                self.S.append(st_tuple)

        # Generate single-step contexts from alphabet: f(s1,...,[],...,sn)
        # These are critical for distinguishing height/depth properties
        self._add_alphabet_contexts()

        # Recompute extensions and fill
        self._compute_extensions()
        self.fill_all()

    def _add_alphabet_contexts(self):
        """Add single-step contexts: for each symbol and position, f(s,...,[],...,s)."""
        for sym in self.alphabet:
            if sym.arity == 0:
                continue
            for pos in range(sym.arity):
                # Use each S representative for the other positions
                other_positions = sym.arity - 1
                if other_positions == 0:
                    # Unary: just sym([])
                    ctx = Context.make(sym.name, [], Context.trivial(), [])
                    if ctx not in self.E:
                        self.E.append(ctx)
                else:
                    for combo in itertools.product(self.S, repeat=other_positions):
                        others = [tuple_to_tree(c) for c in combo]
                        before = others[:pos]
                        after = others[pos:]
                        ctx = Context.make(sym.name, before, Context.trivial(), after)
                        if ctx not in self.E:
                            self.E.append(ctx)


# --- L* Learner ---

@dataclass
class LearningResult:
    """Result of the L* learning algorithm."""
    automaton: BottomUpTreeAutomaton
    membership_queries: int
    equivalence_queries: int
    rounds: int
    states: int
    transitions: int
    converged: bool


def learn_tree_language(alphabet: RankedAlphabet, teacher: Teacher,
                        max_rounds: int = 50) -> LearningResult:
    """Learn a tree regular language using L* algorithm.

    Args:
        alphabet: The ranked alphabet
        teacher: Answers membership and equivalence queries
        max_rounds: Maximum learning rounds

    Returns:
        LearningResult with the learned automaton and statistics
    """
    table = ObservationTable(alphabet, teacher)
    table.initialize()

    for round_num in range(max_rounds):
        # Close and make consistent
        table.close()
        table.make_consistent()

        # Build hypothesis
        hyp = table.build_hypothesis()

        # Equivalence query
        ce = teacher.equivalence(hyp)
        if ce is None:
            # Correct!
            mq = teacher.membership_count if hasattr(teacher, 'membership_count') else 0
            eq = teacher.equivalence_count if hasattr(teacher, 'equivalence_count') else 0
            return LearningResult(
                automaton=hyp,
                membership_queries=mq,
                equivalence_queries=eq,
                rounds=round_num + 1,
                states=len(hyp.states),
                transitions=sum(len(v) for v in hyp.transitions.values()),
                converged=True,
            )

        # Process counterexample
        table.process_counterexample(ce)

    # Did not converge
    hyp = table.build_hypothesis()
    mq = teacher.membership_count if hasattr(teacher, 'membership_count') else 0
    eq = teacher.equivalence_count if hasattr(teacher, 'equivalence_count') else 0
    return LearningResult(
        automaton=hyp,
        membership_queries=mq,
        equivalence_queries=eq,
        rounds=max_rounds,
        states=len(hyp.states),
        transitions=sum(len(v) for v in hyp.transitions.values()),
        converged=False,
    )


# --- Convenience APIs ---

def learn_from_automaton(target: BottomUpTreeAutomaton,
                         max_rounds: int = 50) -> LearningResult:
    """Learn a tree language given a target automaton as teacher.

    This is the standard testing setup: the target is known, and we
    verify that L* recovers an equivalent automaton.
    """
    teacher = AutomatonTeacher(target)
    return learn_tree_language(target.alphabet, teacher, max_rounds)


def learn_from_predicate(alphabet: RankedAlphabet,
                         predicate: Callable[[Tree], bool],
                         max_ce_size: int = 8,
                         max_rounds: int = 50) -> LearningResult:
    """Learn a tree language given a predicate function.

    Uses bounded enumeration for equivalence queries.
    """
    teacher = PredicateTeacher(alphabet, predicate, max_ce_size)
    return learn_tree_language(alphabet, teacher, max_rounds)


def learn_from_examples(alphabet: RankedAlphabet,
                        positive: List[Tree],
                        negative: List[Tree],
                        max_rounds: int = 50) -> LearningResult:
    """Learn a tree language from positive and negative examples.

    Builds a teacher that classifies known examples and uses
    bounded enumeration to find counterexamples beyond the examples.
    """
    pos_set = {tree_to_tuple(t) for t in positive}
    neg_set = {tree_to_tuple(t) for t in negative}

    class ExampleTeacher(Teacher):
        def __init__(self):
            self.membership_count = 0
            self.equivalence_count = 0

        def membership(self, t: Tree) -> bool:
            self.membership_count += 1
            tt = tree_to_tuple(t)
            if tt in pos_set:
                return True
            if tt in neg_set:
                return False
            # Unknown: default to False (conservative)
            return False

        def equivalence(self, hypothesis: BottomUpTreeAutomaton) -> Optional[Tree]:
            self.equivalence_count += 1
            # Check all known examples
            for t in positive:
                if not hypothesis.accepts(t):
                    return t
            for t in negative:
                if hypothesis.accepts(t):
                    return t
            return None

    teacher = ExampleTeacher()
    return learn_tree_language(alphabet, teacher, max_rounds)


def learn_and_compare(target: BottomUpTreeAutomaton,
                      max_rounds: int = 50) -> Dict:
    """Learn a language and compare the learned automaton with the target.

    Returns comparison statistics.
    """
    result = learn_from_automaton(target, max_rounds)

    equiv = check_language_equivalence(target, result.automaton)

    return {
        "converged": result.converged,
        "equivalent": equiv["equivalent"],
        "rounds": result.rounds,
        "membership_queries": result.membership_queries,
        "equivalence_queries": result.equivalence_queries,
        "target_states": len(target.states),
        "learned_states": result.states,
        "target_transitions": sum(len(v) for v in target.transitions.values()),
        "learned_transitions": result.transitions,
    }


def learn_boolean_tree_language(alphabet: RankedAlphabet,
                                predicate: Callable[[Tree], bool],
                                max_ce_size: int = 6,
                                max_rounds: int = 50) -> Dict:
    """Learn and return comprehensive results for a boolean tree language.

    Returns dict with automaton, statistics, and sample accepted/rejected trees.
    """
    result = learn_from_predicate(alphabet, predicate, max_ce_size, max_rounds)

    # Find some accepted and rejected trees
    accepted = result.automaton.enumerate_trees(max_size=6, max_count=10)
    # Find rejected by enumerating and filtering
    teacher = PredicateTeacher(alphabet, lambda t: True, max_ce_size=4)
    all_trees = teacher._enumerate_trees(4)
    rejected = [t for t in all_trees if not result.automaton.accepts(t)][:10]

    return {
        "automaton": result.automaton,
        "converged": result.converged,
        "rounds": result.rounds,
        "states": result.states,
        "transitions": result.transitions,
        "membership_queries": result.membership_queries,
        "equivalence_queries": result.equivalence_queries,
        "sample_accepted": accepted[:5],
        "sample_rejected": rejected[:5],
    }


# --- Benchmark Suite ---

def make_all_trees_language(alphabet: RankedAlphabet) -> BottomUpTreeAutomaton:
    """Language of ALL trees over the alphabet (universal language)."""
    buta = BottomUpTreeAutomaton(alphabet)
    buta.add_state("q0", final=True)
    for s in alphabet:
        buta.add_transition(s.name, tuple(["q0"] * s.arity), "q0")
    return buta


def make_height_bounded_language(alphabet: RankedAlphabet,
                                 max_height: int) -> BottomUpTreeAutomaton:
    """Language of trees with height <= max_height."""
    buta = BottomUpTreeAutomaton(alphabet)
    for h in range(max_height + 1):
        buta.add_state(f"h{h}", final=True)

    for s in alphabet:
        if s.arity == 0:
            buta.add_transition(s.name, (), "h0")
        else:
            for combo in itertools.product(range(max_height), repeat=s.arity):
                max_child = max(combo)
                if max_child + 1 <= max_height:
                    child_states = tuple(f"h{c}" for c in combo)
                    buta.add_transition(s.name, child_states, f"h{max_child + 1}")
    return buta


def make_symbol_count_language(alphabet: RankedAlphabet,
                               target_symbol: str,
                               count_mod: int,
                               target_remainder: int) -> BottomUpTreeAutomaton:
    """Language where count of target_symbol mod count_mod == target_remainder."""
    buta = BottomUpTreeAutomaton(alphabet)

    for i in range(count_mod):
        buta.add_state(f"r{i}", final=(i == target_remainder))

    for s in alphabet:
        sym_count = 1 if s.name == target_symbol else 0
        if s.arity == 0:
            buta.add_transition(s.name, (), f"r{sym_count % count_mod}")
        else:
            for combo in itertools.product(range(count_mod), repeat=s.arity):
                total = (sum(combo) + sym_count) % count_mod
                child_states = tuple(f"r{c}" for c in combo)
                buta.add_transition(s.name, child_states, f"r{total}")
    return buta


def benchmark_learning(target: BottomUpTreeAutomaton, name: str = "") -> Dict:
    """Run learning on a target and collect benchmark data."""
    result = learn_and_compare(target)
    result["name"] = name
    return result


def run_benchmark_suite(alphabet: RankedAlphabet) -> List[Dict]:
    """Run a suite of benchmarks on standard target languages."""
    results = []

    # 1. All trees
    target = make_all_trees_language(alphabet)
    results.append(benchmark_learning(target, "all_trees"))

    # 2. Height bounded
    for h in [1, 2]:
        target = make_height_bounded_language(alphabet, h)
        results.append(benchmark_learning(target, f"height_le_{h}"))

    # 3. Symbol counting
    constants = alphabet.constants()
    if constants:
        sym = constants[0].name
        target = make_symbol_count_language(alphabet, sym, 2, 0)
        results.append(benchmark_learning(target, f"even_{sym}"))

    return results
