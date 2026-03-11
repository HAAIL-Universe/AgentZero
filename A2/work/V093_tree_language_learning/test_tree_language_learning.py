"""Tests for V093: Tree Regular Language Learning"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_language_learning import (
    Context, Tree, tree_to_tuple, tuple_to_tree,
    Symbol, RankedAlphabet, BottomUpTreeAutomaton,
    ObservationTable, AutomatonTeacher, PredicateTeacher,
    LearningResult, learn_tree_language, learn_from_automaton,
    learn_from_predicate, learn_from_examples, learn_and_compare,
    learn_boolean_tree_language, make_all_trees_language,
    make_height_bounded_language, make_symbol_count_language,
    benchmark_learning, run_benchmark_suite, Teacher,
)
from tree_automata import check_language_equivalence


# --- Helpers ---

def binary_alphabet():
    """Alphabet: a (const), b (const), f (binary)."""
    alpha = RankedAlphabet()
    alpha.add("a", 0)
    alpha.add("b", 0)
    alpha.add("f", 2)
    return alpha

def unary_binary_alphabet():
    """Alphabet: a (const), g (unary), f (binary)."""
    alpha = RankedAlphabet()
    alpha.add("a", 0)
    alpha.add("g", 1)
    alpha.add("f", 2)
    return alpha

def simple_alphabet():
    """Alphabet: a (const), b (const), s (unary)."""
    alpha = RankedAlphabet()
    alpha.add("a", 0)
    alpha.add("b", 0)
    alpha.add("s", 1)
    return alpha


# ============================================================
# Section 1: Context basics
# ============================================================

class TestContextBasics:
    def test_trivial_context(self):
        ctx = Context.trivial()
        assert ctx.is_trivial()
        t = Tree("a")
        filled = ctx.fill(t)
        assert filled.symbol == "a"
        assert filled.children == []

    def test_single_level_context(self):
        """Context: f([], a)"""
        ctx = Context.make("f", [], Context.trivial(), [Tree("a")])
        assert not ctx.is_trivial()
        filled = ctx.fill(Tree("b"))
        assert filled.symbol == "f"
        assert filled.children[0].symbol == "b"
        assert filled.children[1].symbol == "a"

    def test_nested_context(self):
        """Context: f(g([]), a)"""
        inner = Context.make("g", [], Context.trivial(), [])
        outer = Context.make("f", [], inner, [Tree("a")])
        filled = outer.fill(Tree("b"))
        assert filled.symbol == "f"
        assert filled.children[0].symbol == "g"
        assert filled.children[0].children[0].symbol == "b"

    def test_context_depth(self):
        ctx0 = Context.trivial()
        assert ctx0.depth() == 0
        ctx1 = Context.make("f", [], Context.trivial(), [Tree("a")])
        assert ctx1.depth() == 1
        ctx2 = Context.make("f", [], ctx1, [Tree("b")])
        assert ctx2.depth() == 2

    def test_context_equality(self):
        ctx1 = Context.make("f", [], Context.trivial(), [Tree("a")])
        ctx2 = Context.make("f", [], Context.trivial(), [Tree("a")])
        assert ctx1 == ctx2
        assert hash(ctx1) == hash(ctx2)

    def test_context_inequality(self):
        ctx1 = Context.make("f", [], Context.trivial(), [Tree("a")])
        ctx2 = Context.make("f", [Tree("a")], Context.trivial(), [])
        assert ctx1 != ctx2


# ============================================================
# Section 2: Tree tuple conversion
# ============================================================

class TestTreeTuples:
    def test_leaf_roundtrip(self):
        t = Tree("a")
        tt = tree_to_tuple(t)
        assert tt == ("a",)
        t2 = tuple_to_tree(tt)
        assert t2.symbol == "a"
        assert t2.children == []

    def test_binary_roundtrip(self):
        t = Tree("f", [Tree("a"), Tree("b")])
        tt = tree_to_tuple(t)
        t2 = tuple_to_tree(tt)
        assert t2.symbol == "f"
        assert len(t2.children) == 2
        assert t2.children[0].symbol == "a"
        assert t2.children[1].symbol == "b"

    def test_nested_roundtrip(self):
        t = Tree("f", [Tree("f", [Tree("a"), Tree("b")]), Tree("a")])
        tt = tree_to_tuple(t)
        t2 = tuple_to_tree(tt)
        assert t2.symbol == "f"
        assert t2.children[0].symbol == "f"
        assert t2.children[0].children[0].symbol == "a"


# ============================================================
# Section 3: AutomatonTeacher
# ============================================================

class TestAutomatonTeacher:
    def test_membership_positive(self):
        alpha = binary_alphabet()
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q", final=True)
        buta.add_transition("a", (), "q")
        buta.add_transition("b", (), "q")
        buta.add_transition("f", ("q", "q"), "q")

        teacher = AutomatonTeacher(buta)
        assert teacher.membership(Tree("a"))
        assert teacher.membership(Tree("f", [Tree("a"), Tree("b")]))

    def test_membership_negative(self):
        alpha = binary_alphabet()
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("qa", final=True)
        buta.add_state("qb", final=False)
        buta.add_transition("a", (), "qa")
        buta.add_transition("b", (), "qb")

        teacher = AutomatonTeacher(buta)
        assert teacher.membership(Tree("a"))
        assert not teacher.membership(Tree("b"))

    def test_equivalence_exact(self):
        alpha = binary_alphabet()
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q", final=True)
        buta.add_transition("a", (), "q")
        buta.add_transition("b", (), "q")
        buta.add_transition("f", ("q", "q"), "q")

        teacher = AutomatonTeacher(buta)
        # Same automaton should be equivalent
        assert teacher.equivalence(buta) is None

    def test_equivalence_counterexample(self):
        alpha = binary_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("q", final=True)
        target.add_transition("a", (), "q")
        target.add_transition("b", (), "q")
        target.add_transition("f", ("q", "q"), "q")

        # Hypothesis only accepts "a"
        hyp = BottomUpTreeAutomaton(alpha)
        hyp.add_state("q", final=True)
        hyp.add_transition("a", (), "q")

        teacher = AutomatonTeacher(target)
        ce = teacher.equivalence(hyp)
        assert ce is not None
        # Counterexample should be in target but not in hyp
        assert target.accepts(ce) != hyp.accepts(ce)

    def test_query_counts(self):
        alpha = simple_alphabet()
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q", final=True)
        buta.add_transition("a", (), "q")
        teacher = AutomatonTeacher(buta)
        teacher.membership(Tree("a"))
        teacher.membership(Tree("b"))
        assert teacher.membership_count == 2
        teacher.equivalence(buta)
        assert teacher.equivalence_count == 1


# ============================================================
# Section 4: PredicateTeacher
# ============================================================

class TestPredicateTeacher:
    def test_membership(self):
        alpha = simple_alphabet()
        teacher = PredicateTeacher(alpha, lambda t: t.symbol == "a")
        assert teacher.membership(Tree("a"))
        assert not teacher.membership(Tree("b"))

    def test_equivalence_finds_ce(self):
        alpha = simple_alphabet()
        # Target: only "a"
        teacher = PredicateTeacher(alpha, lambda t: t.symbol == "a", max_ce_size=3)

        # Hypothesis accepts everything
        hyp = BottomUpTreeAutomaton(alpha)
        hyp.add_state("q", final=True)
        hyp.add_transition("a", (), "q")
        hyp.add_transition("b", (), "q")
        hyp.add_transition("s", ("q",), "q")

        ce = teacher.equivalence(hyp)
        assert ce is not None
        # ce should be accepted by hyp but not by predicate
        assert hyp.accepts(ce) and not (ce.symbol == "a" and not ce.children)

    def test_equivalence_correct(self):
        alpha = simple_alphabet()
        # Target: only constants
        teacher = PredicateTeacher(alpha, lambda t: t.is_leaf(), max_ce_size=3)

        hyp = BottomUpTreeAutomaton(alpha)
        hyp.add_state("q", final=True)
        hyp.add_state("qn", final=False)
        hyp.add_transition("a", (), "q")
        hyp.add_transition("b", (), "q")
        hyp.add_transition("s", ("q",), "qn")
        hyp.add_transition("s", ("qn",), "qn")

        ce = teacher.equivalence(hyp)
        assert ce is None


# ============================================================
# Section 5: ObservationTable
# ============================================================

class TestObservationTable:
    def test_initialization(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        teacher = AutomatonTeacher(target)
        table = ObservationTable(alpha, teacher)
        table.initialize()
        # Should have constants in S
        assert len(table.S) >= 2  # a, b

    def test_row_vector(self):
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("q", final=True)
        target.add_transition("a", (), "q")
        teacher = AutomatonTeacher(target)
        table = ObservationTable(alpha, teacher)
        table.initialize()
        # Row for "a" should have True in trivial context
        a_tuple = tree_to_tuple(Tree("a"))
        row_a = table.row(a_tuple)
        assert row_a[0] == True
        # Row for "b" should have False
        b_tuple = tree_to_tuple(Tree("b"))
        row_b = table.row(b_tuple)
        assert row_b[0] == False

    def test_closure_check(self):
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("qa", final=True)
        target.add_state("qb", final=False)
        target.add_transition("a", (), "qa")
        target.add_transition("b", (), "qb")
        target.add_transition("s", ("qa",), "qa")
        target.add_transition("s", ("qb",), "qb")
        teacher = AutomatonTeacher(target)
        table = ObservationTable(alpha, teacher)
        table.initialize()
        # Table should start closed or become closable
        table.close()
        assert table.is_closed() is None


# ============================================================
# Section 6: Learn all-trees language
# ============================================================

class TestLearnAllTrees:
    def test_binary_all_trees(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        result = learn_from_automaton(target)
        assert result.converged
        assert result.states >= 1
        # Verify learned automaton accepts same language
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_unary_binary_all_trees(self):
        alpha = unary_binary_alphabet()
        target = make_all_trees_language(alpha)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_simple_all_trees(self):
        alpha = simple_alphabet()
        target = make_all_trees_language(alpha)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 7: Learn constant-only language
# ============================================================

class TestLearnConstants:
    def test_only_a(self):
        """Learn language containing only the tree 'a'."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("qa", final=True)
        target.add_state("qr", final=False)
        target.add_transition("a", (), "qa")
        target.add_transition("b", (), "qr")
        target.add_transition("s", ("qa",), "qr")
        target.add_transition("s", ("qr",), "qr")

        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))

    def test_only_leaves(self):
        """Learn language of all leaf trees (constants only)."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("ql", final=True)
        target.add_state("qn", final=False)
        target.add_transition("a", (), "ql")
        target.add_transition("b", (), "ql")
        target.add_transition("s", ("ql",), "qn")
        target.add_transition("s", ("qn",), "qn")

        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))


# ============================================================
# Section 8: Learn height-bounded languages
# ============================================================

class TestLearnHeightBounded:
    def test_height_0(self):
        """Learn language of trees with height 0 (leaves only)."""
        alpha = binary_alphabet()
        target = make_height_bounded_language(alpha, 0)
        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("f", [Tree("a"), Tree("b")]))

    def test_height_1(self):
        """Learn language of trees with height <= 1."""
        alpha = binary_alphabet()
        target = make_height_bounded_language(alpha, 1)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_height_2(self):
        """Learn language of trees with height <= 2."""
        alpha = binary_alphabet()
        target = make_height_bounded_language(alpha, 2)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 9: Learn symbol-counting languages
# ============================================================

class TestLearnSymbolCounting:
    def test_even_a(self):
        """Learn: even number of 'a' symbols."""
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 2, 0)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_odd_b(self):
        """Learn: odd number of 'b' symbols."""
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "b", 2, 1)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_a_mod_3(self):
        """Learn: count of 'a' mod 3 == 0."""
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 3, 0)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 10: Learn with two-state automata
# ============================================================

class TestLearnTwoState:
    def test_left_child_a(self):
        """Learn: f(x,y) accepted iff left subtree is 'a'."""
        alpha = binary_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("qa", final=True)
        target.add_state("qb", final=False)
        target.add_state("qf_good", final=True)
        target.add_state("qf_bad", final=False)
        target.add_transition("a", (), "qa")
        target.add_transition("b", (), "qb")
        # f(qa, _) -> qf_good, f(qb, _) -> qf_bad
        for q in ["qa", "qb", "qf_good", "qf_bad"]:
            target.add_transition("f", ("qa", q), "qf_good")
            target.add_transition("f", ("qb", q), "qf_bad")
            target.add_transition("f", ("qf_good", q), "qf_bad")
            target.add_transition("f", ("qf_bad", q), "qf_bad")

        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("f", [Tree("a"), Tree("b")]))
        assert not result.automaton.accepts(Tree("f", [Tree("b"), Tree("a")]))

    def test_balanced_binary(self):
        """Learn: balanced binary trees (both subtrees same height)."""
        alpha = binary_alphabet()
        # States: h0, h1, h2, bad
        target = BottomUpTreeAutomaton(alpha)
        for h in range(3):
            target.add_state(f"h{h}", final=True)
        target.add_state("bad", final=False)
        target.add_transition("a", (), "h0")
        target.add_transition("b", (), "h0")
        for h in range(2):
            target.add_transition("f", (f"h{h}", f"h{h}"), f"h{h+1}")
        # Unbalanced -> bad
        for h1 in range(3):
            for h2 in range(3):
                if h1 != h2:
                    target.add_transition("f", (f"h{h1}", f"h{h2}"), "bad")
            target.add_transition("f", (f"h{h1}", "bad"), "bad")
            target.add_transition("f", ("bad", f"h{h1}"), "bad")
        target.add_transition("f", ("bad", "bad"), "bad")

        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 11: Learn from predicate
# ============================================================

class TestLearnFromPredicate:
    def test_leaf_predicate(self):
        """Learn 'is leaf' from predicate."""
        alpha = simple_alphabet()
        result = learn_from_predicate(alpha, lambda t: t.is_leaf(), max_ce_size=4)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))

    def test_size_predicate(self):
        """Learn 'size <= 3' from predicate."""
        alpha = simple_alphabet()
        result = learn_from_predicate(alpha, lambda t: t.size() <= 3, max_ce_size=5)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("s", [Tree("a")]))
        assert result.automaton.accepts(Tree("s", [Tree("s", [Tree("a")])]))
        assert not result.automaton.accepts(
            Tree("s", [Tree("s", [Tree("s", [Tree("a")])])]))

    def test_root_symbol_predicate(self):
        """Learn 'root symbol is a' from predicate."""
        alpha = simple_alphabet()
        result = learn_from_predicate(alpha, lambda t: t.symbol == "a", max_ce_size=4)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))


# ============================================================
# Section 12: Learn from examples
# ============================================================

class TestLearnFromExamples:
    def test_simple_examples(self):
        alpha = simple_alphabet()
        positive = [Tree("a")]
        negative = [Tree("b"), Tree("s", [Tree("a")]), Tree("s", [Tree("b")])]
        result = learn_from_examples(alpha, positive, negative)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("b"))

    def test_leaf_examples(self):
        alpha = simple_alphabet()
        positive = [Tree("a"), Tree("b")]
        negative = [Tree("s", [Tree("a")]), Tree("s", [Tree("b")])]
        result = learn_from_examples(alpha, positive, negative)
        assert result.converged
        assert result.automaton.accepts(Tree("a"))
        assert result.automaton.accepts(Tree("b"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))


# ============================================================
# Section 13: Learn and compare
# ============================================================

class TestLearnAndCompare:
    def test_compare_all_trees(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        result = learn_and_compare(target)
        assert result["converged"]
        assert result["equivalent"]
        assert result["rounds"] >= 1
        assert result["membership_queries"] > 0
        assert result["equivalence_queries"] >= 1

    def test_compare_symbol_counting(self):
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 2, 0)
        result = learn_and_compare(target)
        assert result["converged"]
        assert result["equivalent"]
        assert result["target_states"] > 0
        assert result["learned_states"] > 0


# ============================================================
# Section 14: Boolean tree language learning
# ============================================================

class TestBooleanTreeLanguage:
    def test_learn_boolean(self):
        alpha = simple_alphabet()
        result = learn_boolean_tree_language(alpha, lambda t: t.is_leaf())
        assert result["converged"]
        assert result["states"] > 0
        assert len(result["sample_accepted"]) > 0

    def test_boolean_with_unary(self):
        alpha = simple_alphabet()
        # Accept trees of height exactly 1 (s(a) or s(b))
        def height_one(t):
            return t.height() == 1
        result = learn_boolean_tree_language(alpha, height_one, max_ce_size=4)
        assert result["converged"]
        assert result["automaton"].accepts(Tree("s", [Tree("a")]))
        assert not result["automaton"].accepts(Tree("a"))


# ============================================================
# Section 15: Benchmark targets
# ============================================================

class TestBenchmarkTargets:
    def test_all_trees_language(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        assert target.accepts(Tree("a"))
        assert target.accepts(Tree("f", [Tree("a"), Tree("b")]))

    def test_height_bounded(self):
        alpha = binary_alphabet()
        target = make_height_bounded_language(alpha, 1)
        assert target.accepts(Tree("a"))
        assert target.accepts(Tree("f", [Tree("a"), Tree("b")]))
        assert not target.accepts(Tree("f", [Tree("f", [Tree("a"), Tree("b")]), Tree("a")]))

    def test_symbol_count(self):
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 2, 0)
        # 0 a's: b
        assert target.accepts(Tree("b"))
        # 1 a: not accepted
        assert not target.accepts(Tree("a"))
        # 2 a's: f(a, a) has 2 a's + f doesn't count as a -> 2 a's (even)
        assert target.accepts(Tree("f", [Tree("a"), Tree("a")]))


# ============================================================
# Section 16: Benchmark suite
# ============================================================

class TestBenchmarkSuite:
    def test_run_suite(self):
        alpha = binary_alphabet()
        results = run_benchmark_suite(alpha)
        assert len(results) >= 3
        for r in results:
            assert r["converged"]
            assert r["equivalent"]
            assert "name" in r


# ============================================================
# Section 17: Unary chain languages
# ============================================================

class TestUnaryChain:
    def test_exact_depth(self):
        """Learn language of trees with exactly depth 2: s(s(a)) or s(s(b))."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("d0", final=False)
        target.add_state("d1", final=False)
        target.add_state("d2", final=True)
        target.add_state("bad", final=False)
        target.add_transition("a", (), "d0")
        target.add_transition("b", (), "d0")
        target.add_transition("s", ("d0",), "d1")
        target.add_transition("s", ("d1",), "d2")
        target.add_transition("s", ("d2",), "bad")
        target.add_transition("s", ("bad",), "bad")

        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("s", [Tree("s", [Tree("a")])]))
        assert result.automaton.accepts(Tree("s", [Tree("s", [Tree("b")])]))
        assert not result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("s", [Tree("a")]))

    def test_even_depth(self):
        """Learn: trees with even height (mod 2 == 0)."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("even", final=True)
        target.add_state("odd", final=False)
        target.add_transition("a", (), "even")
        target.add_transition("b", (), "even")
        target.add_transition("s", ("even",), "odd")
        target.add_transition("s", ("odd",), "even")

        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 18: Multi-state learning
# ============================================================

class TestMultiState:
    def test_three_state(self):
        """Learn a 3-state language: count symbols mod 3."""
        alpha = simple_alphabet()
        target = make_symbol_count_language(alpha, "a", 3, 0)
        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]

    def test_product_language(self):
        """Learn intersection of two properties (via product automaton)."""
        alpha = binary_alphabet()
        # Even 'a' count AND height <= 1
        even_a = make_symbol_count_language(alpha, "a", 2, 0)
        height_1 = make_height_bounded_language(alpha, 1)
        target = BottomUpTreeAutomaton(alpha)
        # Manual product
        from tree_automata import buta_intersection
        target = buta_intersection(even_a, height_1)

        result = learn_from_automaton(target)
        assert result.converged
        equiv = check_language_equivalence(target, result.automaton)
        assert equiv["equivalent"]


# ============================================================
# Section 19: Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_language(self):
        """Learn the empty language (no trees accepted)."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("q", final=False)
        target.add_transition("a", (), "q")
        target.add_transition("b", (), "q")
        target.add_transition("s", ("q",), "q")

        result = learn_from_automaton(target)
        assert result.converged
        assert not result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("b"))

    def test_single_tree_language(self):
        """Learn language containing exactly one tree."""
        alpha = simple_alphabet()
        target = BottomUpTreeAutomaton(alpha)
        target.add_state("qa", final=False)
        target.add_state("qb", final=False)
        target.add_state("qsa", final=True)
        target.add_state("qother", final=False)
        target.add_transition("a", (), "qa")
        target.add_transition("b", (), "qb")
        target.add_transition("s", ("qa",), "qsa")
        target.add_transition("s", ("qb",), "qother")
        target.add_transition("s", ("qsa",), "qother")
        target.add_transition("s", ("qother",), "qother")

        result = learn_from_automaton(target)
        assert result.converged
        assert result.automaton.accepts(Tree("s", [Tree("a")]))
        assert not result.automaton.accepts(Tree("a"))
        assert not result.automaton.accepts(Tree("s", [Tree("b")]))
        assert not result.automaton.accepts(Tree("s", [Tree("s", [Tree("a")])]))

    def test_max_rounds_exceeded(self):
        """Learning with very few rounds may not converge."""
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 3, 0)
        result = learn_from_automaton(target, max_rounds=1)
        # May or may not converge in 1 round -- just ensure no crash
        assert isinstance(result, LearningResult)


# ============================================================
# Section 20: Query complexity
# ============================================================

class TestQueryComplexity:
    def test_membership_count_bounded(self):
        """Membership queries should be reasonable for small targets."""
        alpha = simple_alphabet()
        target = make_all_trees_language(alpha)
        teacher = AutomatonTeacher(target)
        result = learn_tree_language(alpha, teacher)
        assert teacher.membership_count < 500
        assert teacher.equivalence_count < 10

    def test_equivalence_count_bounded(self):
        """Equivalence queries should be small for simple languages."""
        alpha = binary_alphabet()
        target = make_symbol_count_language(alpha, "a", 2, 0)
        teacher = AutomatonTeacher(target)
        result = learn_tree_language(alpha, teacher)
        assert result.converged
        assert teacher.equivalence_count <= 10


# ============================================================
# Section 21: Counterexample processing
# ============================================================

class TestCounterexampleProcessing:
    def test_process_adds_subtrees(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        teacher = AutomatonTeacher(target)
        table = ObservationTable(alpha, teacher)
        table.initialize()

        ce = Tree("f", [Tree("a"), Tree("b")])
        old_count = len(table.S) + len(table.R)
        table.process_counterexample(ce)
        new_count = len(table.S) + len(table.R)
        # Should have added the subtree f(a,b) and possibly new contexts
        assert new_count >= old_count

    def test_process_adds_contexts(self):
        alpha = binary_alphabet()
        target = make_all_trees_language(alpha)
        teacher = AutomatonTeacher(target)
        table = ObservationTable(alpha, teacher)
        table.initialize()

        ce = Tree("f", [Tree("a"), Tree("b")])
        old_contexts = len(table.E)
        table.process_counterexample(ce)
        # Should add contexts from the counterexample
        assert len(table.E) >= old_contexts


# ============================================================
# Section 22: Learning statistics
# ============================================================

class TestLearningStatistics:
    def test_result_fields(self):
        alpha = simple_alphabet()
        target = make_all_trees_language(alpha)
        result = learn_from_automaton(target)
        assert hasattr(result, 'automaton')
        assert hasattr(result, 'membership_queries')
        assert hasattr(result, 'equivalence_queries')
        assert hasattr(result, 'rounds')
        assert hasattr(result, 'states')
        assert hasattr(result, 'transitions')
        assert hasattr(result, 'converged')
        assert result.states > 0
        assert result.transitions > 0

    def test_benchmark_learning(self):
        alpha = simple_alphabet()
        target = make_all_trees_language(alpha)
        result = benchmark_learning(target, "test")
        assert result["name"] == "test"
        assert result["converged"]
        assert result["equivalent"]
