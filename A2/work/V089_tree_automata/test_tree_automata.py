"""Tests for V089: Tree Automata"""

import pytest
from tree_automata import (
    Symbol, RankedAlphabet, Tree, tree,
    BottomUpTreeAutomaton, TopDownTreeAutomaton,
    buta_to_tdta, buta_union, buta_intersection, buta_complement,
    buta_difference, buta_is_subset, buta_is_equivalent, buta_minimize,
    make_alphabet, make_buta, make_tdta,
    TreePattern, pat, RewriteRule, TermRewriteSystem,
    buta_from_patterns, tree_language_size, buta_stats, compare_butas,
    schema_automaton, check_tree_membership, check_language_emptiness,
    check_language_inclusion, check_language_equivalence,
)


# ─── Ranked Alphabet ───────────────────────────────────────────────

class TestRankedAlphabet:
    def test_create_alphabet(self):
        alpha = make_alphabet(("a", 0), ("f", 1), ("g", 2))
        assert len(alpha) == 3
        assert alpha.get("a").arity == 0
        assert alpha.get("f").arity == 1
        assert alpha.get("g").arity == 2

    def test_constants(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 1))
        consts = alpha.constants()
        assert len(consts) == 2
        assert all(c.arity == 0 for c in consts)

    def test_by_arity(self):
        alpha = make_alphabet(("a", 0), ("f", 1), ("g", 2), ("h", 2))
        assert len(alpha.by_arity(2)) == 2
        assert len(alpha.by_arity(1)) == 1
        assert len(alpha.by_arity(3)) == 0

    def test_max_arity(self):
        alpha = make_alphabet(("a", 0), ("g", 2), ("h", 3))
        assert alpha.max_arity() == 3

    def test_symbol_repr(self):
        s = Symbol("f", 2)
        assert repr(s) == "f/2"


# ─── Tree ──────────────────────────────────────────────────────────

class TestTree:
    def test_leaf(self):
        t = tree("a")
        assert t.is_leaf()
        assert t.size() == 1
        assert t.height() == 0

    def test_unary(self):
        t = tree("f", tree("a"))
        assert not t.is_leaf()
        assert t.size() == 2
        assert t.height() == 1

    def test_binary(self):
        t = tree("g", tree("a"), tree("b"))
        assert t.size() == 3
        assert t.height() == 1

    def test_nested(self):
        t = tree("g", tree("f", tree("a")), tree("b"))
        assert t.size() == 4
        assert t.height() == 2

    def test_equality(self):
        t1 = tree("g", tree("a"), tree("b"))
        t2 = tree("g", tree("a"), tree("b"))
        t3 = tree("g", tree("b"), tree("a"))
        assert t1 == t2
        assert t1 != t3

    def test_hash(self):
        t1 = tree("g", tree("a"), tree("b"))
        t2 = tree("g", tree("a"), tree("b"))
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1

    def test_subtrees(self):
        t = tree("g", tree("f", tree("a")), tree("b"))
        subs = t.subtrees()
        assert len(subs) == 4

    def test_repr_leaf(self):
        assert repr(tree("a")) == "a"

    def test_repr_nested(self):
        t = tree("f", tree("a"))
        assert repr(t) == "f(a)"


# ─── Bottom-Up Tree Automaton: Basic ──────────────────────────────

class TestBUTABasic:
    def _nat_automaton(self):
        """Automaton recognizing natural numbers: z | s(z) | s(s(z)) | ..."""
        alpha = make_alphabet(("z", 0), ("s", 1))
        return make_buta(alpha,
                         states=["q"],
                         final=["q"],
                         transitions=[
                             ("z", (), "q"),
                             ("s", ("q",), "q"),
                         ])

    def test_accepts_zero(self):
        buta = self._nat_automaton()
        assert buta.accepts(tree("z"))

    def test_accepts_succ(self):
        buta = self._nat_automaton()
        assert buta.accepts(tree("s", tree("z")))
        assert buta.accepts(tree("s", tree("s", tree("z"))))

    def test_rejects(self):
        buta = self._nat_automaton()
        # Unknown symbol
        assert not buta.accepts(tree("a"))

    def test_run_states(self):
        buta = self._nat_automaton()
        states = buta.run(tree("s", tree("z")))
        assert "q" in states

    def test_is_deterministic(self):
        buta = self._nat_automaton()
        assert buta.is_deterministic()

    def test_transition_count(self):
        buta = self._nat_automaton()
        assert buta.transition_count() == 2


class TestBUTABinaryTrees:
    def _bin_tree_automaton(self):
        """Automaton recognizing all binary trees over {leaf, node}."""
        alpha = make_alphabet(("leaf", 0), ("node", 2))
        return make_buta(alpha,
                         states=["q"],
                         final=["q"],
                         transitions=[
                             ("leaf", (), "q"),
                             ("node", ("q", "q"), "q"),
                         ])

    def test_accepts_leaf(self):
        assert self._bin_tree_automaton().accepts(tree("leaf"))

    def test_accepts_simple_node(self):
        t = tree("node", tree("leaf"), tree("leaf"))
        assert self._bin_tree_automaton().accepts(t)

    def test_accepts_deep_tree(self):
        t = tree("node",
                 tree("node", tree("leaf"), tree("leaf")),
                 tree("leaf"))
        assert self._bin_tree_automaton().accepts(t)

    def test_rejects_wrong_arity(self):
        # node with 1 child should be rejected (no matching transition)
        t = tree("node", tree("leaf"))
        buta = self._bin_tree_automaton()
        # This won't match because node expects 2 children but we gave 1
        assert not buta.accepts(t)


# ─── BUTA: Nondeterminism ────────────────────────────────────────

class TestBUTANondeterministic:
    def test_nondeterministic_acceptance(self):
        """NFA that accepts trees with an 'a' leaf somewhere."""
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 2))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("qa", final=True)  # saw 'a'
        buta.add_state("qn")              # no 'a' yet

        # a -> qa
        buta.add_transition("a", (), "qa")
        # b -> qn
        buta.add_transition("b", (), "qn")
        # f(x, y) -> qa if either x or y is qa
        buta.add_transition("f", ("qa", "qa"), "qa")
        buta.add_transition("f", ("qa", "qn"), "qa")
        buta.add_transition("f", ("qn", "qa"), "qa")
        buta.add_transition("f", ("qn", "qn"), "qn")

        # f(a, b) should accept
        assert buta.accepts(tree("f", tree("a"), tree("b")))
        # f(b, a) should accept
        assert buta.accepts(tree("f", tree("b"), tree("a")))
        # f(b, b) should NOT accept
        assert not buta.accepts(tree("f", tree("b"), tree("b")))
        # Deep: f(f(b, a), b) should accept
        assert buta.accepts(tree("f", tree("f", tree("b"), tree("a")), tree("b")))

    def test_nondeterministic_multiple_targets(self):
        """Transition with multiple targets."""
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q1", final=True)
        buta.add_state("q2", final=True)
        buta.add_transition("a", (), "q1")
        buta.add_transition("a", (), "q2")  # nondeterministic
        buta.add_transition("f", ("q1",), "q1")
        buta.add_transition("f", ("q2",), "q2")

        assert not buta.is_deterministic()
        assert buta.accepts(tree("f", tree("a")))


# ─── Emptiness and Witness ───────────────────────────────────────

class TestEmptinessWitness:
    def test_non_empty(self):
        alpha = make_alphabet(("a", 0))
        buta = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        assert not buta.is_empty()

    def test_empty_no_final(self):
        alpha = make_alphabet(("a", 0))
        buta = make_buta(alpha, ["q"], [], [("a", (), "q")])
        assert buta.is_empty()

    def test_empty_unreachable_final(self):
        alpha = make_alphabet(("a", 0))
        buta = make_buta(alpha, ["q", "qf"], ["qf"], [("a", (), "q")])
        assert buta.is_empty()

    def test_witness_found(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])
        w = buta.witness()
        assert w is not None
        assert buta.accepts(w)

    def test_witness_none_when_empty(self):
        alpha = make_alphabet(("a", 0))
        buta = make_buta(alpha, ["q"], [], [("a", (), "q")])
        assert buta.witness() is None


# ─── Enumeration ─────────────────────────────────────────────────

class TestEnumeration:
    def test_enumerate_finite_language(self):
        """Language = {a, b}"""
        alpha = make_alphabet(("a", 0), ("b", 0))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("b", (), "q"),
        ])
        trees = buta.enumerate_trees(max_size=1)
        assert len(trees) == 2

    def test_enumerate_with_unary(self):
        """Language = {a, f(a), f(f(a)), ...} up to size 3"""
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])
        trees = buta.enumerate_trees(max_size=3)
        assert len(trees) == 3
        assert tree("a") in trees
        assert tree("f", tree("a")) in trees
        assert tree("f", tree("f", tree("a"))) in trees


# ─── Determinization ─────────────────────────────────────────────

class TestDeterminization:
    def test_determinize_nfa(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q1", final=True)
        buta.add_state("q2")
        buta.add_transition("a", (), "q1")
        buta.add_transition("a", (), "q2")
        buta.add_transition("f", ("q1",), "q1")
        buta.add_transition("f", ("q2",), "q1")

        assert not buta.is_deterministic()
        det = buta.determinize()
        assert det.is_deterministic()
        # Should still accept same trees
        assert det.accepts(tree("a"))
        assert det.accepts(tree("f", tree("a")))

    def test_determinize_preserves_language(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 2))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("qa", final=True)
        buta.add_state("qb")
        buta.add_transition("a", (), "qa")
        buta.add_transition("b", (), "qb")
        buta.add_transition("f", ("qa", "qb"), "qa")
        buta.add_transition("f", ("qb", "qa"), "qa")

        det = buta.determinize()
        # f(a, b) accepted
        assert det.accepts(tree("f", tree("a"), tree("b")))
        # f(b, a) accepted
        assert det.accepts(tree("f", tree("b"), tree("a")))
        # f(a, a) -- not accepted (both children in same state is fine but produces qa)
        # Wait, f(qa, qa) is not defined, so this should NOT be accepted
        assert not det.accepts(tree("f", tree("a"), tree("a")))


# ─── Top-Down Tree Automaton ─────────────────────────────────────

class TestTDTA:
    def test_basic_acceptance(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        tdta = make_tdta(alpha, ["q"], ["q"], [
            ("q", "a", ()),
            ("q", "f", ("q",)),
        ])
        assert tdta.accepts(tree("a"))
        assert tdta.accepts(tree("f", tree("a")))
        assert not tdta.accepts(tree("b"))

    def test_binary_tree(self):
        alpha = make_alphabet(("leaf", 0), ("node", 2))
        tdta = make_tdta(alpha, ["q"], ["q"], [
            ("q", "leaf", ()),
            ("q", "node", ("q", "q")),
        ])
        assert tdta.accepts(tree("leaf"))
        assert tdta.accepts(tree("node", tree("leaf"), tree("leaf")))

    def test_convert_to_buta(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        tdta = make_tdta(alpha, ["q"], ["q"], [
            ("q", "a", ()),
            ("q", "f", ("q",)),
        ])
        buta = tdta.to_bottom_up()
        assert buta.accepts(tree("a"))
        assert buta.accepts(tree("f", tree("a")))
        assert buta.accepts(tree("f", tree("f", tree("a"))))

    def test_buta_to_tdta_roundtrip(self):
        alpha = make_alphabet(("a", 0), ("s", 1))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("s", ("q",), "q"),
        ])
        tdta = buta_to_tdta(buta)
        # Should accept same language
        assert tdta.accepts(tree("a"))
        assert tdta.accepts(tree("s", tree("a")))
        assert tdta.accepts(tree("s", tree("s", tree("a"))))


# ─── Union ───────────────────────────────────────────────────────

class TestUnion:
    def test_union_disjoint(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        u = buta_union(a1, a2)
        assert u.accepts(tree("a"))
        assert u.accepts(tree("b"))

    def test_union_overlapping(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        u = buta_union(a1, a2)
        assert u.accepts(tree("a"))
        assert u.accepts(tree("b"))


# ─── Intersection ────────────────────────────────────────────────

class TestIntersection:
    def test_intersection_common(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        inter = buta_intersection(a1, a2)
        assert inter.accepts(tree("b"))
        assert not inter.accepts(tree("a"))

    def test_intersection_with_structure(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 1))
        # a1 accepts: a, f(a), f(f(a)), ...
        a1 = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])
        # a2 accepts: a, b, f(a), f(b), ...
        a2 = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("b", (), "q"),
            ("f", ("q",), "q"),
        ])
        inter = buta_intersection(a1, a2)
        assert inter.accepts(tree("a"))
        assert inter.accepts(tree("f", tree("a")))
        # b is only in a2, not a1
        assert not inter.accepts(tree("b"))


# ─── Complement ──────────────────────────────────────────────────

class TestComplement:
    def test_complement_accepts_rejected(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        comp = buta_complement(a)
        assert not comp.accepts(tree("a"))
        assert comp.accepts(tree("b"))

    def test_double_complement(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        comp2 = buta_complement(buta_complement(a))
        assert comp2.accepts(tree("a"))
        assert not comp2.accepts(tree("b"))


# ─── Difference ──────────────────────────────────────────────────

class TestDifference:
    def test_difference_basic(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        diff = buta_difference(a1, a2)
        assert diff.accepts(tree("a"))
        assert not diff.accepts(tree("b"))


# ─── Language Inclusion and Equivalence ──────────────────────────

class TestInclusion:
    def test_subset_true(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        result, counter = buta_is_subset(a1, a2)
        assert result
        assert counter is None

    def test_subset_false(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        result, counter = buta_is_subset(a1, a2)
        assert not result
        assert counter is not None
        # Counter should be in a1 but not a2
        assert a1.accepts(counter)
        assert not a2.accepts(counter)

    def test_equivalence_true(self):
        alpha = make_alphabet(("a", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        result, counter = buta_is_equivalent(a1, a2)
        assert result
        assert counter is None

    def test_equivalence_false(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        result, counter = buta_is_equivalent(a1, a2)
        assert not result


# ─── Minimization ────────────────────────────────────────────────

class TestMinimization:
    def test_minimize_redundant_states(self):
        """Two states with identical behavior should be merged."""
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("q1", final=True)
        buta.add_state("q2", final=True)
        buta.add_transition("a", (), "q1")
        buta.add_transition("a", (), "q2")
        buta.add_transition("f", ("q1",), "q1")
        buta.add_transition("f", ("q2",), "q2")

        mini = buta_minimize(buta)
        # Should have fewer states (q1 and q2 merged)
        assert len(mini.states) <= len(buta.states)
        # Language preserved
        assert mini.accepts(tree("a"))
        assert mini.accepts(tree("f", tree("a")))

    def test_minimize_already_minimal(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        buta = make_buta(alpha, ["qa", "qb"], ["qa"], [
            ("a", (), "qa"),
            ("b", (), "qb"),
        ])
        mini = buta_minimize(buta)
        assert len(mini.states) == 2  # Already minimal


# ─── Tree Pattern Matching ───────────────────────────────────────

class TestTreePattern:
    def test_exact_leaf_match(self):
        p = pat("a")
        assert p.match(tree("a")) == {}
        assert p.match(tree("b")) is None

    def test_exact_nested_match(self):
        p = pat("f", pat("a"))
        assert p.match(tree("f", tree("a"))) == {}
        assert p.match(tree("f", tree("b"))) is None

    def test_wildcard_match(self):
        p = pat(None)  # wildcard
        assert p.match(tree("anything")) == {}
        assert p.match(tree("f", tree("a"))) == {}

    def test_variable_capture(self):
        p = pat("f", pat(var="x"))
        result = p.match(tree("f", tree("a")))
        assert result is not None
        assert result["x"] == tree("a")

    def test_variable_consistency(self):
        """Same variable must match same subtree."""
        p = pat("g", pat(var="x"), pat(var="x"))
        assert p.match(tree("g", tree("a"), tree("a"))) is not None
        assert p.match(tree("g", tree("a"), tree("b"))) is None

    def test_nested_variable(self):
        p = pat("g", pat("f", pat(var="x")), pat(var="y"))
        result = p.match(tree("g", tree("f", tree("a")), tree("b")))
        assert result == {"x": tree("a"), "y": tree("b")}


# ─── Term Rewriting ──────────────────────────────────────────────

class TestTermRewriting:
    def test_simple_rewrite(self):
        trs = TermRewriteSystem()
        # Rule: f(a) -> b
        trs.add_rule(
            pat("f", pat("a")),
            lambda bindings: tree("b")
        )
        result = trs.rewrite_step(tree("f", tree("a")))
        assert result == tree("b")

    def test_no_match(self):
        trs = TermRewriteSystem()
        trs.add_rule(pat("f", pat("a")), lambda b: tree("b"))
        result = trs.rewrite_step(tree("g", tree("a")))
        assert result is None

    def test_deep_rewrite(self):
        trs = TermRewriteSystem()
        # Rule: f(a) -> b  (applied inside g)
        trs.add_rule(pat("f", pat("a")), lambda b: tree("b"))
        result = trs.rewrite_step(tree("g", tree("f", tree("a")), tree("c")))
        assert result == tree("g", tree("b"), tree("c"))

    def test_normalize(self):
        trs = TermRewriteSystem()
        # s(z) -> 1, s(1) -> 2
        trs.add_rule(pat("s", pat("z")), lambda b: tree("1"))
        trs.add_rule(pat("s", pat("1")), lambda b: tree("2"))
        result = trs.normalize(tree("s", tree("s", tree("z"))))
        assert result == tree("2")

    def test_normalize_with_variables(self):
        trs = TermRewriteSystem()
        # add(x, z) -> x
        trs.add_rule(
            pat("add", pat(var="x"), pat("z")),
            lambda b: b["x"]
        )
        # add(x, s(y)) -> s(add(x, y))
        trs.add_rule(
            pat("add", pat(var="x"), pat("s", pat(var="y"))),
            lambda b: tree("s", tree("add", b["x"], b["y"]))
        )
        # add(s(z), s(z)) = s(add(s(z), z)) = s(s(z))
        result = trs.normalize(tree("add", tree("s", tree("z")), tree("s", tree("z"))))
        assert result == tree("s", tree("s", tree("z")))

    def test_is_normal_form(self):
        trs = TermRewriteSystem()
        trs.add_rule(pat("f", pat("a")), lambda b: tree("b"))
        assert trs.is_normal_form(tree("b"))
        assert not trs.is_normal_form(tree("f", tree("a")))


# ─── Schema Automaton ────────────────────────────────────────────

class TestSchemaAutomaton:
    def test_simple_schema(self):
        buta = schema_automaton({
            "doc": [["para"]],
            "para": [[]],
        })
        # doc(para) should be accepted
        assert buta.accepts(tree("doc", tree("para")))
        # plain para should not (doc is root)
        assert not buta.accepts(tree("para"))

    def test_schema_multiple_children(self):
        buta = schema_automaton({
            "doc": [["para", "para"]],
            "para": [[]],
        })
        assert buta.accepts(tree("doc", tree("para"), tree("para")))

    def test_schema_alternatives(self):
        """doc can have 1 or 2 paras (different arities get different symbol names)."""
        buta = schema_automaton({
            "doc": [["para"], ["para", "para"]],
            "para": [[]],
        })
        # With multiple arities, schema creates doc_1 and doc_2
        assert buta.accepts(tree("doc_1", tree("para")))
        assert buta.accepts(tree("doc_2", tree("para"), tree("para")))


# ─── High-Level APIs ────────────────────────────────────────────

class TestHighLevelAPIs:
    def _simple_buta(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 1))
        return make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])

    def test_check_tree_membership_accepted(self):
        buta = self._simple_buta()
        result = check_tree_membership(buta, tree("a"))
        assert result["accepted"]

    def test_check_tree_membership_rejected(self):
        buta = self._simple_buta()
        result = check_tree_membership(buta, tree("b"))
        assert not result["accepted"]

    def test_check_language_emptiness_non_empty(self):
        buta = self._simple_buta()
        result = check_language_emptiness(buta)
        assert not result["empty"]
        assert result["witness"] is not None

    def test_check_language_emptiness_empty(self):
        alpha = make_alphabet(("a", 0))
        buta = make_buta(alpha, ["q"], [], [("a", (), "q")])
        result = check_language_emptiness(buta)
        assert result["empty"]
        assert result["witness"] is None

    def test_check_language_inclusion(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("a", (), "q"), ("b", (), "q")])
        result = check_language_inclusion(a1, a2)
        assert result["included"]

    def test_check_language_equivalence(self):
        alpha = make_alphabet(("a", 0))
        a1 = make_buta(alpha, ["q1"], ["q1"], [("a", (), "q1")])
        a2 = make_buta(alpha, ["q2"], ["q2"], [("a", (), "q2")])
        result = check_language_equivalence(a1, a2)
        assert result["equivalent"]


# ─── Stats and Comparison ────────────────────────────────────────

class TestStatsComparison:
    def test_buta_stats(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])
        stats = buta_stats(buta)
        assert stats["states"] == 1
        assert stats["final_states"] == 1
        assert stats["transitions"] == 2
        assert stats["deterministic"]
        assert not stats["empty"]

    def test_compare_butas_equivalent(self):
        alpha = make_alphabet(("a", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["p"], ["p"], [("a", (), "p")])
        result = compare_butas(a1, a2)
        assert result["equivalent"]
        assert result["a1_subset_a2"]
        assert result["a2_subset_a1"]

    def test_compare_butas_different(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        a1 = make_buta(alpha, ["q"], ["q"], [("a", (), "q")])
        a2 = make_buta(alpha, ["q"], ["q"], [("b", (), "q")])
        result = compare_butas(a1, a2)
        assert not result["equivalent"]


# ─── Complex Examples ────────────────────────────────────────────

class TestComplexExamples:
    def test_balanced_binary_trees(self):
        """Automaton accepting only balanced binary trees.

        A tree is balanced if left and right subtrees have the same height.
        We encode heights as states: q0 (leaf), q1 (height 1), q2 (height 2).
        """
        alpha = make_alphabet(("leaf", 0), ("node", 2))
        buta = BottomUpTreeAutomaton(alpha)
        max_h = 3
        for h in range(max_h + 1):
            buta.add_state(f"q{h}", final=True)
        buta.add_transition("leaf", (), "q0")
        for h in range(max_h):
            buta.add_transition("node", (f"q{h}", f"q{h}"), f"q{h+1}")

        # Balanced: node(leaf, leaf)
        assert buta.accepts(tree("node", tree("leaf"), tree("leaf")))
        # Balanced: node(node(leaf,leaf), node(leaf,leaf))
        assert buta.accepts(tree("node",
            tree("node", tree("leaf"), tree("leaf")),
            tree("node", tree("leaf"), tree("leaf"))))
        # Unbalanced: node(node(leaf,leaf), leaf)
        assert not buta.accepts(tree("node",
            tree("node", tree("leaf"), tree("leaf")),
            tree("leaf")))

    def test_even_depth_leaves(self):
        """Accept trees where all leaves are at even depth."""
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("even", final=True)
        buta.add_state("odd")
        # Leaf at depth 0 (even)
        buta.add_transition("a", (), "even")
        # f flips parity
        buta.add_transition("f", ("even",), "odd")
        buta.add_transition("f", ("odd",), "even")

        assert buta.accepts(tree("a"))           # depth 0 (even)
        assert not buta.accepts(tree("f", tree("a")))  # depth 1 (odd)
        assert buta.accepts(tree("f", tree("f", tree("a"))))  # depth 2 (even)

    def test_boolean_formula_evaluation(self):
        """Tree automaton that evaluates boolean formulas.

        true, false are constants; and(x,y), or(x,y), not(x) are operations.
        Accept iff formula evaluates to true.
        """
        alpha = make_alphabet(
            ("true", 0), ("false", 0),
            ("and", 2), ("or", 2), ("not", 1)
        )
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("T", final=True)
        buta.add_state("F")

        buta.add_transition("true", (), "T")
        buta.add_transition("false", (), "F")
        buta.add_transition("and", ("T", "T"), "T")
        buta.add_transition("and", ("T", "F"), "F")
        buta.add_transition("and", ("F", "T"), "F")
        buta.add_transition("and", ("F", "F"), "F")
        buta.add_transition("or", ("T", "T"), "T")
        buta.add_transition("or", ("T", "F"), "T")
        buta.add_transition("or", ("F", "T"), "T")
        buta.add_transition("or", ("F", "F"), "F")
        buta.add_transition("not", ("T",), "F")
        buta.add_transition("not", ("F",), "T")

        # and(true, or(false, true)) = and(true, true) = true
        assert buta.accepts(tree("and", tree("true"), tree("or", tree("false"), tree("true"))))
        # and(true, false) = false
        assert not buta.accepts(tree("and", tree("true"), tree("false")))
        # not(false) = true
        assert buta.accepts(tree("not", tree("false")))
        # not(true) = false
        assert not buta.accepts(tree("not", tree("true")))

    def test_sorted_list_automaton(self):
        """Accept lists (cons/nil) where values are sorted (a < b < c)."""
        alpha = make_alphabet(
            ("nil", 0),
            ("cons_a", 1), ("cons_b", 1), ("cons_c", 1)
        )
        buta = BottomUpTreeAutomaton(alpha)
        # States represent "minimum expected value"
        # after_c: tail contained only >= c values (or nil)
        # after_b: tail contained only >= b values
        # after_a: tail contained only >= a values
        buta.add_state("after_c")
        buta.add_state("after_b")
        buta.add_state("after_a", final=True)
        buta.add_state("nil_state")

        buta.add_transition("nil", (), "nil_state")
        # Also nil counts as any "after_x"
        buta.add_transition("nil", (), "after_a")
        buta.add_transition("nil", (), "after_b")
        buta.add_transition("nil", (), "after_c")

        # cons_c(tail) -> after_c  if tail is nil or after_c
        buta.add_transition("cons_c", ("nil_state",), "after_c")
        buta.add_transition("cons_c", ("after_c",), "after_c")

        # cons_b(tail) -> after_b  if tail is after_c or nil
        buta.add_transition("cons_b", ("nil_state",), "after_b")
        buta.add_transition("cons_b", ("after_c",), "after_b")

        # cons_a(tail) -> after_a  if tail is after_b or after_c or nil
        buta.add_transition("cons_a", ("nil_state",), "after_a")
        buta.add_transition("cons_a", ("after_b",), "after_a")
        buta.add_transition("cons_a", ("after_c",), "after_a")

        # sorted: cons_a(cons_b(cons_c(nil)))
        assert buta.accepts(tree("cons_a", tree("cons_b", tree("cons_c", tree("nil")))))
        # sorted: cons_a(nil)
        assert buta.accepts(tree("cons_a", tree("nil")))
        # unsorted: cons_b(cons_a(nil)) -- b before a
        assert not buta.accepts(tree("cons_b", tree("cons_a", tree("nil"))))

    def test_arithmetic_expression_type_check(self):
        """Type check arithmetic: int+int=int, int>int=bool, if(bool,t,t)=t."""
        alpha = make_alphabet(
            ("num", 0), ("tt", 0), ("ff", 0),
            ("add", 2), ("gt", 2), ("ite", 3)
        )
        buta = BottomUpTreeAutomaton(alpha)
        buta.add_state("int")
        buta.add_state("bool")
        buta.add_state("well_typed_int", final=True)
        buta.add_state("well_typed_bool", final=True)

        buta.add_transition("num", (), "int")
        buta.add_transition("num", (), "well_typed_int")
        buta.add_transition("tt", (), "bool")
        buta.add_transition("tt", (), "well_typed_bool")
        buta.add_transition("ff", (), "bool")
        buta.add_transition("ff", (), "well_typed_bool")

        # add: int * int -> int
        buta.add_transition("add", ("int", "int"), "int")
        buta.add_transition("add", ("int", "int"), "well_typed_int")
        buta.add_transition("add", ("well_typed_int", "int"), "int")
        buta.add_transition("add", ("well_typed_int", "int"), "well_typed_int")
        buta.add_transition("add", ("int", "well_typed_int"), "int")
        buta.add_transition("add", ("int", "well_typed_int"), "well_typed_int")
        buta.add_transition("add", ("well_typed_int", "well_typed_int"), "int")
        buta.add_transition("add", ("well_typed_int", "well_typed_int"), "well_typed_int")

        # gt: int * int -> bool
        for q1 in ["int", "well_typed_int"]:
            for q2 in ["int", "well_typed_int"]:
                buta.add_transition("gt", (q1, q2), "bool")
                buta.add_transition("gt", (q1, q2), "well_typed_bool")

        # ite: bool * T * T -> T (both branches same type)
        for cond in ["bool", "well_typed_bool"]:
            for t in ["int", "well_typed_int"]:
                buta.add_transition("ite", (cond, t, t), t)
                buta.add_transition("ite", (cond, t, t), "well_typed_int")

        # Well-typed: add(num, num)
        assert buta.accepts(tree("add", tree("num"), tree("num")))
        # Well-typed: gt(num, num)
        assert buta.accepts(tree("gt", tree("num"), tree("num")))
        # Well-typed: ite(gt(num,num), num, num)
        assert buta.accepts(tree("ite",
            tree("gt", tree("num"), tree("num")),
            tree("num"), tree("num")))


# ─── BUTA from Patterns ─────────────────────────────────────────

class TestBUTAFromPatterns:
    def test_simple_pattern(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 1))
        buta = buta_from_patterns(alpha, [pat("f", pat("a"))])
        assert buta.accepts(tree("f", tree("a")))

    def test_wildcard_pattern(self):
        alpha = make_alphabet(("a", 0), ("b", 0), ("f", 1))
        buta = buta_from_patterns(alpha, [pat("f", pat(None))])
        assert buta.accepts(tree("f", tree("a")))
        assert buta.accepts(tree("f", tree("b")))


# ─── Tree Language Size ─────────────────────────────────────────

class TestTreeLanguageSize:
    def test_finite_language_size(self):
        alpha = make_alphabet(("a", 0), ("b", 0))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("b", (), "q"),
        ])
        size = tree_language_size(buta, max_size=1)
        assert size == 2

    def test_infinite_language_bounded(self):
        alpha = make_alphabet(("a", 0), ("f", 1))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q",), "q"),
        ])
        size = tree_language_size(buta, max_size=3)
        assert size == 3  # a, f(a), f(f(a))


# ─── Edge Cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_automaton(self):
        buta = BottomUpTreeAutomaton()
        assert buta.is_empty()
        assert buta.witness() is None

    def test_single_constant(self):
        alpha = make_alphabet(("x", 0))
        buta = make_buta(alpha, ["q"], ["q"], [("x", (), "q")])
        assert buta.accepts(tree("x"))
        assert not buta.is_empty()

    def test_no_constants_reachable(self):
        """Automaton with only unary transitions but no constants."""
        alpha = make_alphabet(("f", 1))
        buta = make_buta(alpha, ["q"], ["q"], [("f", ("q",), "q")])
        assert buta.is_empty()  # Can't build any tree without constants

    def test_large_arity(self):
        alpha = make_alphabet(("a", 0), ("f", 3))
        buta = make_buta(alpha, ["q"], ["q"], [
            ("a", (), "q"),
            ("f", ("q", "q", "q"), "q"),
        ])
        assert buta.accepts(tree("f", tree("a"), tree("a"), tree("a")))

    def test_many_symbols(self):
        specs = [(f"s{i}", 0) for i in range(10)]
        alpha = make_alphabet(*specs)
        transitions = [(f"s{i}", (), "q") for i in range(10)]
        buta = make_buta(alpha, ["q"], ["q"], transitions)
        for i in range(10):
            assert buta.accepts(tree(f"s{i}"))
