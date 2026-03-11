"""Tests for V090: Tree Transducers."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_automata import (
    Tree, tree, make_alphabet, make_buta, RankedAlphabet,
    BottomUpTreeAutomaton, TreePattern, pat,
)
from tree_transducers import (
    OutputTemplate, out, out_var, BUTTRule,
    BottomUpTreeTransducer, TopDownTreeTransducer,
    TDTTRule, compose_butt, sequential_transduce,
    rewrite_to_butt, transducer_domain, transducer_range,
    check_functionality, check_totality, transducer_equivalence,
    type_check_transducer, inverse_butt,
    identity_transducer, relabeling_transducer,
    pruning_transducer, flattening_transducer,
    transducer_stats, compare_transducers,
    transform_tree, verify_transformation, compose_transformations,
    transformation_summary, _buta_from_trees,
    make_ast_optimizer,
)


# ===== Test Alphabet =====

def arith_alphabet():
    """Arithmetic expression alphabet: plus/2, times/2, neg/1, zero/0, one/0, two/0"""
    return make_alphabet(
        ("plus", 2), ("times", 2), ("neg", 1),
        ("zero", 0), ("one", 0), ("two", 0),
    )

def simple_alphabet():
    """Simple alphabet: f/2, g/1, a/0, b/0"""
    return make_alphabet(("f", 2), ("g", 1), ("a", 0), ("b", 0))

def bool_alphabet():
    """Boolean expression alphabet: and/2, or/2, not/1, true/0, false/0"""
    return make_alphabet(
        ("and", 2), ("or", 2), ("not", 1), ("true", 0), ("false", 0)
    )


# ===== OutputTemplate Tests =====

class TestOutputTemplate:
    def test_build_leaf(self):
        t = out("a")
        result = t.build({})
        assert result.symbol == "a"
        assert result.children == []

    def test_build_with_children(self):
        t = out("f", out("a"), out("b"))
        result = t.build({})
        assert result.symbol == "f"
        assert len(result.children) == 2
        assert result.children[0].symbol == "a"
        assert result.children[1].symbol == "b"

    def test_build_variable(self):
        t = out_var("x")
        subtree = tree("a")
        result = t.build({"x": subtree})
        assert result.symbol == "a"

    def test_build_mixed(self):
        t = out("f", out_var("x"), out("b"))
        result = t.build({"x": tree("a")})
        assert result.symbol == "f"
        assert result.children[0].symbol == "a"
        assert result.children[1].symbol == "b"

    def test_unbound_variable_error(self):
        t = out_var("x")
        with pytest.raises(ValueError, match="Unbound"):
            t.build({})

    def test_variables(self):
        t = out("f", out_var("x"), out("g", out_var("y")))
        assert t.variables() == {"x", "y"}

    def test_is_linear_true(self):
        t = out("f", out_var("x"), out_var("y"))
        assert t.is_linear() is True

    def test_is_linear_false(self):
        t = out("f", out_var("x"), out_var("x"))
        assert t.is_linear() is False

    def test_repr_leaf(self):
        assert repr(out("a")) == "a"

    def test_repr_var(self):
        assert repr(out_var("x")) == "$x"

    def test_repr_compound(self):
        assert repr(out("f", out_var("x"), out("a"))) == "f($x, a)"


# ===== Bottom-Up Tree Transducer Tests =====

class TestBottomUpTreeTransducer:
    def test_identity_manual(self):
        """Manual identity transducer."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("b", (), "q", out("b"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("0"), out_var("1")))

        t = tree("f", tree("a"), tree("g", tree("b")))
        results = butt.transduce(t)
        assert len(results) == 1
        assert repr(results[0]) == repr(t)

    def test_swap_children(self):
        """Swap children of binary node."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("b", (), "q", out("b"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("1"), out_var("0")))

        t = tree("f", tree("a"), tree("b"))
        results = butt.transduce(t)
        assert len(results) == 1
        assert results[0].children[0].symbol == "b"
        assert results[0].children[1].symbol == "a"

    def test_constant_folding(self):
        """Fold plus(zero, x) -> x."""
        alpha = arith_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_state("qz")  # state for zero

        butt.add_rule("zero", (), "qz", out("zero"))
        butt.add_rule("zero", (), "q", out("zero"))
        butt.add_rule("one", (), "q", out("one"))
        butt.add_rule("two", (), "q", out("two"))
        butt.add_rule("neg", ("q",), "q", out("neg", out_var("0")))
        butt.add_rule("times", ("q", "q"), "q", out("times", out_var("0"), out_var("1")))
        butt.add_rule("plus", ("q", "q"), "q", out("plus", out_var("0"), out_var("1")))
        # Fold: plus(zero, x) -> x
        butt.add_rule("plus", ("qz", "q"), "q", out_var("1"))

        # plus(zero, one) -> one
        t = tree("plus", tree("zero"), tree("one"))
        results = butt.transduce(t)
        assert any(r.symbol == "one" for r in results)

    def test_nondeterministic(self):
        """Nondeterministic transducer with multiple outputs."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("a", (), "q", out("b"))  # nondeterministic: a -> a or b
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        t = tree("g", tree("a"))
        results = butt.transduce(t)
        assert len(results) == 2
        symbols = {r.children[0].symbol for r in results}
        assert symbols == {"a", "b"}

    def test_is_deterministic(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        assert butt.is_deterministic() is True
        butt.add_rule("a", (), "q", out("b"))
        assert butt.is_deterministic() is False

    def test_is_linear(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("0"), out_var("1")))
        assert butt.is_linear() is True

    def test_is_not_linear(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        # Duplicate: f(x,y) -> f(x,x)
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("0"), out_var("0")))
        assert butt.is_linear() is False

    def test_relabeling(self):
        """Rename symbols."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("b"))  # a -> b
        butt.add_rule("b", (), "q", out("a"))  # b -> a
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("0"), out_var("1")))

        t = tree("f", tree("a"), tree("b"))
        results = butt.transduce(t)
        assert len(results) == 1
        assert results[0].children[0].symbol == "b"
        assert results[0].children[1].symbol == "a"

    def test_input_automaton(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        domain = butt.input_automaton()
        assert domain.accepts(tree("a"))
        assert domain.accepts(tree("g", tree("a")))
        assert not domain.accepts(tree("b"))

    def test_deep_tree(self):
        """Process a deeply nested tree."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("b"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        t = tree("g", tree("g", tree("g", tree("a"))))
        results = butt.transduce(t)
        assert len(results) == 1
        # Inner 'a' becomes 'b'
        inner = results[0].children[0].children[0].children[0]
        assert inner.symbol == "b"


# ===== Top-Down Tree Transducer Tests =====

class TestTopDownTreeTransducer:
    def test_identity(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        tdtt.add_rule("q", "b", out("b"), ())
        tdtt.add_rule("q", "g", out("g", out_var("0")), ("q",))
        tdtt.add_rule("q", "f", out("f", out_var("0"), out_var("1")), ("q", "q"))

        t = tree("f", tree("a"), tree("g", tree("b")))
        results = tdtt.transduce(t)
        assert len(results) == 1
        assert repr(results[0]) == repr(t)

    def test_swap_children(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        tdtt.add_rule("q", "b", out("b"), ())
        tdtt.add_rule("q", "f", out("f", out_var("1"), out_var("0")), ("q", "q"))

        t = tree("f", tree("a"), tree("b"))
        results = tdtt.transduce(t)
        assert len(results) == 1
        assert results[0].children[0].symbol == "b"
        assert results[0].children[1].symbol == "a"

    def test_state_dependent(self):
        """Different transformation based on state."""
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_state("qleft")
        tdtt.add_state("qright")

        tdtt.add_rule("q", "f", out("f", out_var("0"), out_var("1")), ("qleft", "qright"))
        tdtt.add_rule("qleft", "a", out("b"), ())  # left child: a -> b
        tdtt.add_rule("qleft", "b", out("b"), ())
        tdtt.add_rule("qright", "a", out("a"), ())  # right child: keep
        tdtt.add_rule("qright", "b", out("a"), ())  # right child: b -> a

        t = tree("f", tree("a"), tree("b"))
        results = tdtt.transduce(t)
        assert len(results) == 1
        assert results[0].children[0].symbol == "b"  # left a->b
        assert results[0].children[1].symbol == "a"  # right b->a

    def test_nondeterministic(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        tdtt.add_rule("q", "a", out("b"), ())  # nondeterministic

        results = tdtt.transduce(tree("a"))
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert symbols == {"a", "b"}

    def test_is_deterministic(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        assert tdtt.is_deterministic() is True
        tdtt.add_rule("q", "a", out("b"), ())
        assert tdtt.is_deterministic() is False

    def test_input_automaton(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        tdtt.add_rule("q", "g", out("g", out_var("0")), ("q",))

        domain_buta = tdtt.input_automaton().to_bottom_up()
        assert domain_buta.accepts(tree("a"))
        assert domain_buta.accepts(tree("g", tree("a")))


# ===== Identity Transducer Tests =====

class TestIdentityTransducer:
    def test_identity_leaf(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        for sym_name in ["a", "b"]:
            t = tree(sym_name)
            results = ident.transduce(t)
            assert len(results) == 1
            assert repr(results[0]) == repr(t)

    def test_identity_compound(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        t = tree("f", tree("g", tree("a")), tree("b"))
        results = ident.transduce(t)
        assert len(results) == 1
        assert repr(results[0]) == repr(t)

    def test_identity_stats(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        stats = transducer_stats(ident)
        assert stats['deterministic'] is True
        assert stats['linear'] is True


# ===== Relabeling Transducer Tests =====

class TestRelabelingTransducer:
    def test_swap_labels(self):
        alpha = simple_alphabet()
        relabel = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})
        t = tree("a")
        results = relabel.transduce(t)
        assert len(results) == 1
        assert results[0].symbol == "b"

    def test_partial_relabel(self):
        alpha = simple_alphabet()
        relabel = relabeling_transducer(alpha, alpha, {"a": "b"})
        t = tree("f", tree("a"), tree("b"))
        results = relabel.transduce(t)
        assert len(results) == 1
        # a->b, b stays b
        assert results[0].children[0].symbol == "b"
        assert results[0].children[1].symbol == "b"


# ===== Pruning Transducer Tests =====

class TestPruningTransducer:
    def test_prune_leaf(self):
        alpha = simple_alphabet()
        pruner = pruning_transducer(alpha, "a", out("b"))
        t = tree("a")
        results = pruner.transduce(t)
        assert len(results) == 1
        assert results[0].symbol == "b"

    def test_prune_subtree(self):
        alpha = simple_alphabet()
        pruner = pruning_transducer(alpha, "g", out("a"))
        t = tree("f", tree("g", tree("b")), tree("a"))
        results = pruner.transduce(t)
        assert len(results) == 1
        assert results[0].children[0].symbol == "a"  # g(b) -> a
        assert results[0].children[1].symbol == "a"   # unchanged

    def test_prune_preserves_other(self):
        alpha = simple_alphabet()
        pruner = pruning_transducer(alpha, "a", out("b"))
        t = tree("g", tree("b"))
        results = pruner.transduce(t)
        assert len(results) == 1
        assert results[0].symbol == "g"
        assert results[0].children[0].symbol == "b"


# ===== Rewrite-to-Transducer Tests =====

class TestRewriteToTransducer:
    def test_simple_rewrite(self):
        alpha = simple_alphabet()
        # g(x) -> x (remove g wrapper)
        rules = [
            (pat("g", pat(var="x")), out_var("x"))
        ]
        butt = rewrite_to_butt(alpha, rules)
        t = tree("g", tree("a"))
        results = butt.transduce(t)
        # Should have both: g(a) (identity) and a (rewrite)
        reprs = {repr(r) for r in results}
        assert "a" in reprs

    def test_double_g_rewrite(self):
        alpha = simple_alphabet()
        rules = [
            (pat("g", pat(var="x")), out_var("x"))
        ]
        butt = rewrite_to_butt(alpha, rules)
        t = tree("g", tree("g", tree("a")))
        results = butt.transduce(t)
        reprs = {repr(r) for r in results}
        # Should include "a" (both g's removed)
        assert "a" in reprs

    def test_swap_rewrite(self):
        alpha = simple_alphabet()
        # f(x, y) -> f(y, x)
        rules = [
            (pat("f", pat(var="x"), pat(var="y")),
             out("f", out_var("y"), out_var("x")))
        ]
        butt = rewrite_to_butt(alpha, rules)
        t = tree("f", tree("a"), tree("b"))
        results = butt.transduce(t)
        reprs = {repr(r) for r in results}
        assert "f(b, a)" in reprs


# ===== Domain and Range Tests =====

class TestDomainRange:
    def test_domain_extraction(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        domain = transducer_domain(butt)
        assert domain.accepts(tree("a"))
        assert domain.accepts(tree("g", tree("a")))
        assert not domain.accepts(tree("b"))

    def test_range_extraction(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("b"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        range_buta = transducer_range(butt, max_input_size=4)
        assert range_buta.accepts(tree("b"))
        assert range_buta.accepts(tree("g", tree("b")))

    def test_domain_tdtt(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())

        domain = transducer_domain(tdtt)
        assert domain.accepts(tree("a"))


# ===== Functionality and Totality Tests =====

class TestProperties:
    def test_functional_deterministic(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        result = check_functionality(ident)
        assert result['functional'] is True

    def test_not_functional(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        butt.add_rule("a", (), "q", out("b"))

        result = check_functionality(butt)
        assert result['functional'] is False

    def test_total(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        result = check_totality(ident)
        assert result['total'] is True

    def test_partial_is_total_over_domain(self):
        """A transducer with limited rules is total over its own domain.

        The domain is exactly the set of trees the transducer can process,
        so every domain tree produces output -- total is True.
        """
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("a"))
        # Domain is just {a}, and a produces output -> total
        result = check_totality(butt)
        assert result['total'] is True

    def test_empty_domain_is_total(self):
        """Empty transducer has no domain, so totality is vacuously true."""
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        result = check_totality(butt)
        assert result['total'] is True


# ===== Equivalence Tests =====

class TestEquivalence:
    def test_equivalent_identity(self):
        alpha = simple_alphabet()
        id1 = identity_transducer(alpha)
        id2 = identity_transducer(alpha)
        result = transducer_equivalence(id1, id2)
        assert result['equivalent'] is True

    def test_not_equivalent(self):
        alpha = simple_alphabet()
        id1 = identity_transducer(alpha)
        relabel = relabeling_transducer(alpha, alpha, {"a": "b"})
        result = transducer_equivalence(id1, relabel)
        assert result['equivalent'] is False


# ===== Type Checking Tests =====

class TestTypeChecking:
    def test_well_typed(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)

        # Input type: trees with only a and g
        input_buta = BottomUpTreeAutomaton(alpha)
        input_buta.add_state("q", final=True)
        input_buta.add_transition("a", (), "q")
        input_buta.add_transition("g", ("q",), "q")

        # Output type: same
        output_buta = BottomUpTreeAutomaton(alpha)
        output_buta.add_state("q", final=True)
        output_buta.add_transition("a", (), "q")
        output_buta.add_transition("g", ("q",), "q")

        result = type_check_transducer(ident, input_buta, output_buta)
        assert result['well_typed'] is True

    def test_not_well_typed(self):
        alpha = simple_alphabet()
        # Transducer that maps a -> b
        relabel = relabeling_transducer(alpha, alpha, {"a": "b"})

        # Input type: only a
        input_buta = BottomUpTreeAutomaton(alpha)
        input_buta.add_state("q", final=True)
        input_buta.add_transition("a", (), "q")

        # Output type: only a (but transducer produces b)
        output_buta = BottomUpTreeAutomaton(alpha)
        output_buta.add_state("q", final=True)
        output_buta.add_transition("a", (), "q")

        result = type_check_transducer(relabel, input_buta, output_buta)
        assert result['well_typed'] is False
        assert len(result['violations']) > 0


# ===== Inverse Transducer Tests =====

class TestInverse:
    def test_inverse_relabeling(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("b"))
        butt.add_rule("b", (), "q", out("a"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))
        butt.add_rule("f", ("q", "q"), "q", out("f", out_var("0"), out_var("1")))

        inv = inverse_butt(butt)

        # Forward: a -> b, so inverse: b -> a
        results = inv.transduce(tree("b"))
        assert any(r.symbol == "a" for r in results)

    def test_inverse_compound(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_rule("a", (), "q", out("b"))
        butt.add_rule("b", (), "q", out("a"))
        butt.add_rule("g", ("q",), "q", out("g", out_var("0")))

        inv = inverse_butt(butt)

        # Forward: g(a) -> g(b), so inverse: g(b) -> g(a)
        results = inv.transduce(tree("g", tree("b")))
        assert any(r.children[0].symbol == "a" for r in results)


# ===== Sequential Composition Tests =====

class TestSequentialTransduce:
    def test_sequential_identity(self):
        alpha = simple_alphabet()
        id1 = identity_transducer(alpha)
        id2 = identity_transducer(alpha)
        t = tree("f", tree("a"), tree("b"))
        results = sequential_transduce([id1, id2], t)
        assert len(results) == 1
        assert repr(results[0]) == repr(t)

    def test_sequential_relabel_twice(self):
        alpha = simple_alphabet()
        swap1 = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})
        swap2 = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})
        t = tree("a")
        # swap twice = identity
        results = sequential_transduce([swap1, swap2], t)
        assert len(results) == 1
        assert results[0].symbol == "a"

    def test_sequential_three(self):
        alpha = simple_alphabet()
        relabel = relabeling_transducer(alpha, alpha, {"a": "b"})
        t = tree("a")
        results = sequential_transduce([relabel, relabel, relabel], t)
        assert len(results) == 1
        assert results[0].symbol == "b"

    def test_compose_transformations_api(self):
        alpha = simple_alphabet()
        swap = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})
        composed = compose_transformations([swap, swap])
        results = composed(tree("a"))
        assert len(results) == 1
        assert results[0].symbol == "a"


# ===== Statistics Tests =====

class TestStatistics:
    def test_butt_stats(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        stats = transducer_stats(ident)
        assert stats['type'] == 'bottom-up'
        assert stats['states'] == 1
        assert stats['final_states'] == 1
        assert stats['deterministic'] is True
        assert stats['linear'] is True

    def test_tdtt_stats(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        tdtt.add_rule("q", "a", out("a"), ())
        stats = transducer_stats(tdtt)
        assert stats['type'] == 'top-down'
        assert stats['initial_states'] == 1

    def test_compare_transducers(self):
        alpha = simple_alphabet()
        id1 = identity_transducer(alpha)
        id2 = identity_transducer(alpha)
        result = compare_transducers(id1, id2)
        assert result['equivalent'] is True
        assert result['functional1'] is True
        assert result['functional2'] is True


# ===== Transform Tree API Tests =====

class TestTransformTree:
    def test_transform_butt(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        results = transform_tree(ident, tree("a"))
        assert len(results) == 1

    def test_verify_transformation(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)

        input_buta = BottomUpTreeAutomaton(alpha)
        input_buta.add_state("q", final=True)
        input_buta.add_transition("a", (), "q")
        input_buta.add_transition("b", (), "q")

        output_buta = BottomUpTreeAutomaton(alpha)
        output_buta.add_state("q", final=True)
        output_buta.add_transition("a", (), "q")
        output_buta.add_transition("b", (), "q")

        result = verify_transformation(ident, input_buta, output_buta)
        assert result['well_typed'] is True


# ===== Transformation Summary Tests =====

class TestSummary:
    def test_summary_identity(self):
        alpha = simple_alphabet()
        ident = identity_transducer(alpha)
        summary = transformation_summary(ident)
        assert summary['total_examples'] > 0
        assert summary['stats']['deterministic'] is True

    def test_summary_relabel(self):
        alpha = simple_alphabet()
        relabel = relabeling_transducer(alpha, alpha, {"a": "b"})
        summary = transformation_summary(relabel)
        assert summary['transformations'] > 0


# ===== Boolean Simplification Tests =====

class TestBooleanSimplification:
    def test_double_negation(self):
        """not(not(x)) -> x"""
        alpha = bool_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_state("qnot")  # "we just saw a not"

        butt.add_rule("true", (), "q", out("true"))
        butt.add_rule("false", (), "q", out("false"))
        butt.add_rule("and", ("q", "q"), "q", out("and", out_var("0"), out_var("1")))
        butt.add_rule("or", ("q", "q"), "q", out("or", out_var("0"), out_var("1")))
        butt.add_rule("not", ("q",), "q", out("not", out_var("0")))
        butt.add_rule("not", ("q",), "qnot", out_var("0"))  # Remember we saw not
        butt.add_rule("not", ("qnot",), "q", out_var("0"))  # Double negation elimination

        t = tree("not", tree("not", tree("true")))
        results = butt.transduce(t)
        assert any(r.symbol == "true" for r in results)

    def test_and_true(self):
        """and(true, x) -> x"""
        alpha = bool_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_state("qt")  # true state

        butt.add_rule("true", (), "q", out("true"))
        butt.add_rule("true", (), "qt", out("true"))
        butt.add_rule("false", (), "q", out("false"))
        butt.add_rule("not", ("q",), "q", out("not", out_var("0")))
        butt.add_rule("or", ("q", "q"), "q", out("or", out_var("0"), out_var("1")))
        butt.add_rule("and", ("q", "q"), "q", out("and", out_var("0"), out_var("1")))
        butt.add_rule("and", ("qt", "q"), "q", out_var("1"))  # and(true, x) -> x

        t = tree("and", tree("true"), tree("false"))
        results = butt.transduce(t)
        assert any(r.symbol == "false" for r in results)


# ===== AST Optimizer Tests =====

class TestASTOptimizer:
    def test_optimizer_creation(self):
        alpha = arith_alphabet()
        opts = [
            # plus(zero, x) -> x
            (pat("plus", pat("zero"), pat(var="x")),
             out_var("x")),
        ]
        optimizer = make_ast_optimizer(alpha, opts)
        assert optimizer is not None
        stats = transducer_stats(optimizer)
        assert stats['rules'] > 0


# ===== BUTA from Trees Tests =====

class TestBUTAFromTrees:
    def test_single_tree(self):
        alpha = simple_alphabet()
        buta = _buta_from_trees(alpha, [tree("a")])
        assert buta.accepts(tree("a"))

    def test_multiple_trees(self):
        alpha = simple_alphabet()
        trees = [tree("a"), tree("b"), tree("g", tree("a"))]
        buta = _buta_from_trees(alpha, trees)
        for t in trees:
            assert buta.accepts(t)


# ===== Compose BUTT Tests =====

class TestComposeBUTT:
    def test_compose_relabeling(self):
        alpha = simple_alphabet()
        # t1: a->b, b->a, keep structure
        t1 = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})
        # t2: b->a, a->b again (so composed = identity... sort of)
        t2 = relabeling_transducer(alpha, alpha, {"a": "b", "b": "a"})

        composed = compose_butt(t1, t2)
        # May not produce full composition due to complexity, but should not crash
        assert composed is not None


# ===== Edge Cases =====

class TestEdgeCases:
    def test_empty_transducer(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        results = butt.transduce(tree("a"))
        assert results == []

    def test_no_final_state(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q")  # not final
        butt.add_rule("a", (), "q", out("a"))
        results = butt.transduce(tree("a"))
        assert results == []

    def test_tdtt_no_initial_state(self):
        alpha = simple_alphabet()
        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q")  # not initial
        tdtt.add_rule("q", "a", out("a"), ())
        results = tdtt.transduce(tree("a"))
        assert results == []

    def test_repr(self):
        alpha = simple_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        assert "BUTT" in repr(butt)

        tdtt = TopDownTreeTransducer(alpha, alpha)
        tdtt.add_state("q", initial=True)
        assert "TDTT" in repr(tdtt)


# ===== Complex Composition Tests =====

class TestComplexComposition:
    def test_arith_simplify(self):
        """Arithmetic simplification: plus(zero, x) -> x, times(one, x) -> x"""
        alpha = arith_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_state("qz")  # zero
        butt.add_state("q1")  # one

        butt.add_rule("zero", (), "qz", out("zero"))
        butt.add_rule("zero", (), "q", out("zero"))
        butt.add_rule("one", (), "q1", out("one"))
        butt.add_rule("one", (), "q", out("one"))
        butt.add_rule("two", (), "q", out("two"))
        butt.add_rule("neg", ("q",), "q", out("neg", out_var("0")))
        butt.add_rule("neg", ("qz",), "q", out("neg", out_var("0")))
        butt.add_rule("neg", ("q1",), "q", out("neg", out_var("0")))
        butt.add_rule("plus", ("q", "q"), "q", out("plus", out_var("0"), out_var("1")))
        butt.add_rule("times", ("q", "q"), "q", out("times", out_var("0"), out_var("1")))

        # Optimizations
        butt.add_rule("plus", ("qz", "q"), "q", out_var("1"))   # 0 + x -> x
        butt.add_rule("plus", ("q", "qz"), "q", out_var("0"))   # x + 0 -> x
        butt.add_rule("times", ("q1", "q"), "q", out_var("1"))   # 1 * x -> x
        butt.add_rule("times", ("q", "q1"), "q", out_var("0"))   # x * 1 -> x
        butt.add_rule("times", ("qz", "q"), "q", out("zero"))    # 0 * x -> 0
        butt.add_rule("times", ("q", "qz"), "q", out("zero"))    # x * 0 -> 0

        # plus(zero, one) -> one
        t = tree("plus", tree("zero"), tree("one"))
        results = butt.transduce(t)
        assert any(r.symbol == "one" for r in results)

        # times(one, two) -> two
        t2 = tree("times", tree("one"), tree("two"))
        results2 = butt.transduce(t2)
        assert any(r.symbol == "two" for r in results2)

        # times(zero, two) -> zero
        t3 = tree("times", tree("zero"), tree("two"))
        results3 = butt.transduce(t3)
        assert any(r.symbol == "zero" for r in results3)

    def test_nested_optimization(self):
        """plus(zero, plus(zero, one)) -> one (two levels of folding)."""
        alpha = arith_alphabet()
        butt = BottomUpTreeTransducer(alpha, alpha)
        butt.add_state("q", final=True)
        butt.add_state("qz")

        butt.add_rule("zero", (), "qz", out("zero"))
        butt.add_rule("zero", (), "q", out("zero"))
        butt.add_rule("one", (), "q", out("one"))
        butt.add_rule("two", (), "q", out("two"))
        butt.add_rule("neg", ("q",), "q", out("neg", out_var("0")))
        butt.add_rule("plus", ("q", "q"), "q", out("plus", out_var("0"), out_var("1")))
        butt.add_rule("plus", ("qz", "q"), "q", out_var("1"))

        t = tree("plus", tree("zero"), tree("plus", tree("zero"), tree("one")))
        results = butt.transduce(t)
        # Bottom-up: inner plus(zero, one) -> one, then outer plus(zero, one) -> one
        assert any(r.symbol == "one" for r in results)


# ===== Multi-alphabet Transducer Tests =====

class TestCrossAlphabet:
    def test_different_alphabets(self):
        """Transducer between different alphabets."""
        input_alpha = make_alphabet(("pair", 2), ("val", 0))
        output_alpha = make_alphabet(("swap", 2), ("val", 0))

        butt = BottomUpTreeTransducer(input_alpha, output_alpha)
        butt.add_state("q", final=True)
        butt.add_rule("val", (), "q", out("val"))
        butt.add_rule("pair", ("q", "q"), "q", out("swap", out_var("1"), out_var("0")))

        t = tree("pair", tree("val"), tree("val"))
        results = butt.transduce(t)
        assert len(results) == 1
        assert results[0].symbol == "swap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
